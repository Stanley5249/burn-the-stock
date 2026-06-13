use crate::label::compute_labels_rewards;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use tracing::instrument;

const CODE: PlSmallStr = PlSmallStr::from_static("code");
const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
const DATE: PlSmallStr = PlSmallStr::from_static("date");
const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");

const OPEN: PlSmallStr = PlSmallStr::from_static("open");
const HIGH: PlSmallStr = PlSmallStr::from_static("high");
const LOW: PlSmallStr = PlSmallStr::from_static("low");
const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

const OPEN_RETURN: PlSmallStr = PlSmallStr::from_static("open_return");
const HIGH_RETURN: PlSmallStr = PlSmallStr::from_static("high_return");
const LOW_RETURN: PlSmallStr = PlSmallStr::from_static("low_return");
const CLOSE_RETURN: PlSmallStr = PlSmallStr::from_static("close_return");
const VOLUME_RETURN: PlSmallStr = PlSmallStr::from_static("volume_return");

/// The five stationary features flattened per row, in column order. Width stays
/// at five so the model, batcher, and tensor shapes are unchanged from the raw
/// OHLCV layout this replaced.
pub(crate) const FEATURE_NAMES: [PlSmallStr; 5] = [
    OPEN_RETURN,
    HIGH_RETURN,
    LOW_RETURN,
    CLOSE_RETURN,
    VOLUME_RETURN,
];

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

/// Build the five stationary feature expressions from the raw OHLCV columns.
///
/// One uniform transform per channel: the natural log of its ratio to the prior
/// bar. The four prices share a single anchor, the previous close, so the
/// overnight gap (open), the intraday extremes (high, low), and the close-to-close
/// return all land in one comparable frame, and the old hand-built range, gap, and
/// body indicators are linear combinations the model can recover on its own. The
/// per-ticker `shift` runs `over` the ticker so it never reaches across stock
/// boundaries, which means the caller must have sorted by `[ticker, date]` first.
fn stationary_features() -> [Expr; 5] {
    let prev_close = col(CLOSE).shift(lit(1)).over([col(TICKER)]);
    let prev_volume = col(VOLUME).shift(lit(1)).over([col(TICKER)]);

    let natural_log = || lit(std::f64::consts::E);

    // Volume is a count, so a no-trade day has volume 0 and the bare ratio's log
    // is `-inf`. Add one share to both sides (log1p style) to keep it finite
    // while leaving real, large volumes essentially unchanged.
    let volume_return = ((col(VOLUME) + lit(1.0)) / (prev_volume + lit(1.0))).log(natural_log());

    let price_return = |price: PlSmallStr, alias: PlSmallStr| {
        (col(price) / prev_close.clone())
            .log(natural_log())
            .alias(alias)
    };

    [
        price_return(OPEN, OPEN_RETURN),
        price_return(HIGH, HIGH_RETURN),
        price_return(LOW, LOW_RETURN),
        price_return(CLOSE, CLOSE_RETURN),
        volume_return.alias(VOLUME_RETURN),
    ]
}

/// Replace each stationary feature with its cross-sectional z-score: subtract the
/// mean and divide by the standard deviation taken `over` all stocks on the same
/// date.
///
/// Market-wide moves (a crash, a rally, an earnings season) push every stock the
/// same way on a given day. Standardizing per date removes that common factor and
/// leaves only how a stock did relative to its peers, which is the signal that
/// survives in a weak-signal universe. Run after [`stationary_features`] and
/// before the `drop_nulls`, so each ticker's null first row is excluded from the
/// per-date statistics by polars and then dropped.
fn cross_sectional_zscore() -> [Expr; 5] {
    FEATURE_NAMES.map(|name| {
        let centered = col(name.clone()) - col(name.clone()).mean().over([col(DATE)]);
        let spread = col(name.clone()).std(1).over([col(DATE)]) + lit(1e-8);
        (centered / spread).alias(name)
    })
}

/// One ticker's history flattened out of polars into plain buffers, so window
/// extraction in the hot path is pointer-offset indexing rather than Arrow array
/// traversal. Rows are sorted ascending by date, and the three vectors share that
/// row order: `dates[i]`, `labels[i]`, and `features[i*5 .. i*5+5]` are the same
/// trading day.
#[derive(Clone)]
struct Ticker {
    name: PlSmallStr,
    /// Trading dates, ascending.
    dates: Vec<NaiveDate>,
    /// Row-major stationary features, length `dates.len() * 5`.
    features: Vec<f32>,
    /// One action label per row.
    labels: Vec<u8>,
    /// Signed realized return of each row's barrier outcome, one per row.
    rewards: Vec<f32>,
}

impl Ticker {
    fn rows(&self) -> usize {
        self.dates.len()
    }

    /// Split into rows `[0, at)` and `[at, rows)`, keeping the three buffers
    /// aligned (features are five wide per row).
    fn split_at(&self, at: usize) -> (Ticker, Ticker) {
        let (dates_left, dates_right) = self.dates.split_at(at);
        let (features_left, features_right) = self.features.split_at(at * FEATURE_NAMES.len());
        let (labels_left, labels_right) = self.labels.split_at(at);
        let (rewards_left, rewards_right) = self.rewards.split_at(at);

        let left = Ticker {
            name: self.name.clone(),
            dates: dates_left.to_vec(),
            features: features_left.to_vec(),
            labels: labels_left.to_vec(),
            rewards: rewards_left.to_vec(),
        };
        let right = Ticker {
            name: self.name.clone(),
            dates: dates_right.to_vec(),
            features: features_right.to_vec(),
            labels: labels_right.to_vec(),
            rewards: rewards_right.to_vec(),
        };

        (left, right)
    }
}

/// Backend-free store of every ticker's flattened history. This is the data
/// layer: it owns the loaded parquet data and the train/valid split and
/// ticker-subset transforms, but knows nothing about windows, tensors, or
/// devices. A [`crate::dataset::WindowDataset`] borrows it behind an `Arc` to
/// enumerate and materialize training samples.
pub struct TickerStore {
    tickers: Vec<Ticker>,
}

impl TickerStore {
    /// Load the parquet file into one flat [`Ticker`] per stock, deriving the
    /// stationary feature window from the raw OHLCV columns in polars once and
    /// then standardizing each feature cross-sectionally across the whole loaded
    /// universe per date.
    ///
    /// Each ticker's first row carries null features because its previous-bar
    /// reference does not exist, so it is dropped. The final `horizon` rows of
    /// each ticker are then dropped so every kept row has a full forward window for
    /// its triple-barrier label. Tickers too short to yield a single window are
    /// kept but simply produce no windows later. `take_profit`, `stop_loss`, and
    /// `horizon` set the barriers passed to [`compute_labels_rewards`].
    ///
    /// Rows with a non-positive close are dropped as corrupt. yfinance back
    /// adjustment can drive a delisted ticker's whole history negative, which
    /// would otherwise poison the log-return and ratio features and the barrier
    /// labels that compare prices.
    #[instrument(level = "info", skip_all, fields(path = %path))]
    pub fn load(
        path: &str,
        take_profit: f32,
        stop_loss: f32,
        horizon: usize,
    ) -> PolarsResult<Self> {
        let feature_expr = concat_arr(
            FEATURE_NAMES
                .map(|name| col(name).cast(DataType::Float32))
                .to_vec(),
        )
        .unwrap();

        let long = LazyFrame::scan_parquet(PlRefPath::new(path), ScanArgsParquet::default())?
            .select([
                col(CODE).cast(DataType::String).alias(TICKER),
                col(DATE).cast(DataType::Date),
                col(OPEN).cast(DataType::Float32),
                col(HIGH).cast(DataType::Float32),
                col(LOW).cast(DataType::Float32),
                col(CLOSE).cast(DataType::Float32),
                col(VOLUME).cast(DataType::Float32),
            ])
            .filter(col(CLOSE).gt(lit(0.0)))
            // Sort before deriving features so the per-ticker `shift` inside
            // `stationary_features` sees rows in date order.
            .sort([TICKER, DATE], SortMultipleOptions::new())
            .with_columns(stationary_features())
            // Standardize each feature across every stock on the same date. This
            // runs over the full loaded universe, before any ticker subsetting,
            // so the cross-section is the whole market.
            .with_columns(cross_sectional_zscore())
            // The only nulls are each ticker's first row, whose previous bar did
            // not exist, so drop them across all columns before packing the
            // feature array.
            .drop_nulls(None)
            .select([
                col(TICKER),
                col(DATE),
                feature_expr.alias(FEATURE),
                // Raw high/low/close survive the feature pipeline untouched, since
                // the z-score only rewrites the feature columns, and the barrier
                // labels need the intraday range to detect a touch.
                col(HIGH),
                col(LOW),
                col(CLOSE),
            ])
            .collect()?;

        let groups = long.partition_by_stable([TICKER], true)?;

        let mut tickers = Vec::with_capacity(groups.len());

        for group in groups {
            let height = group.height();

            // A labeled row needs a full `horizon`-bar forward window, so a ticker
            // with no more rows than the horizon yields nothing.
            if height <= horizon {
                continue;
            }

            let name: PlSmallStr = group.column(&TICKER)?.str()?.get(0).unwrap().into();

            // Labels and rewards are `horizon` short, already aligned to the rows we
            // keep after dropping the trailing horizon-less rows.
            let (labels, rewards) = compute_labels_rewards(
                group.column(&HIGH)?,
                group.column(&LOW)?,
                group.column(&CLOSE)?,
                take_profit,
                stop_loss,
                horizon,
            )?;

            let head = group.head(Some(height - horizon));

            // Flatten the Array<f32, 5> feature column into row-major features
            // once, so window extraction later is a contiguous slice.
            let features: Vec<f32> = head
                .column(&FEATURE)?
                .array()?
                .get_inner()
                .f32()?
                .into_no_null_iter()
                .collect();

            let dates: Vec<NaiveDate> = head
                .column(&DATE)?
                .date()?
                .as_date_iter()
                .flatten()
                .collect();

            tickers.push(Ticker {
                name,
                dates,
                features,
                labels,
                rewards,
            });
        }

        Ok(Self { tickers })
    }

    /// Randomly keep `count` tickers, chosen with `seed` so the subset is
    /// reproducible. Asking for at least as many tickers as exist is a no-op.
    /// Used to carve a small subset for overfit diagnostics, where we want a
    /// representative random sample rather than the first `count` by sort order.
    pub fn sample_tickers(mut self, count: usize, seed: u64) -> Self {
        if count >= self.tickers.len() {
            return self;
        }

        let mut rng = Rng::with_seed(seed);
        let indices = rng.choose_multiple(0..self.tickers.len(), count);

        let tickers = indices
            .into_iter()
            .map(|index| self.tickers[index].clone())
            .collect();

        self.tickers = tickers;
        self
    }

    /// Split every ticker at `cutoff` into an earlier train store and a later
    /// valid store. Tickers whose train or valid side has fewer than `steps`
    /// rows are dropped from that side. Errors if either side ends up empty.
    #[instrument(level = "info", skip_all, fields(steps))]
    pub fn train_valid_split(&self, cutoff: NaiveDate, steps: usize) -> PolarsResult<(Self, Self)> {
        let mut train_tickers = Vec::with_capacity(self.tickers.len());
        let mut valid_tickers = Vec::with_capacity(self.tickers.len());

        for ticker in &self.tickers {
            // Dates are ascending, so `partition_point` is the count of rows
            // strictly before the cutoff.
            let split = ticker.dates.partition_point(|&day| day < cutoff);
            let (train, valid) = ticker.split_at(split);

            if train.rows() >= steps {
                train_tickers.push(train);
            }
            if valid.rows() >= steps {
                valid_tickers.push(valid);
            }
        }

        polars_ensure!(
            !train_tickers.is_empty() && !valid_tickers.is_empty(),
            NoData: "train/valid split left one side empty; check cutoff and steps"
        );

        Ok((
            Self {
                tickers: train_tickers,
            },
            Self {
                tickers: valid_tickers,
            },
        ))
    }

    /// Per-class label counts across every ticker, indexed Sell 0, Hold 1, Buy 2,
    /// for logging the triple-barrier balance after a split.
    pub fn label_counts(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for ticker in &self.tickers {
            for &label in &ticker.labels {
                counts[usize::from(label)] += 1;
            }
        }
        counts
    }

    /// The latest date across every ticker, used to anchor a recent-window
    /// train/valid split. `None` only when no ticker has a dated row.
    pub fn max_date(&self) -> Option<NaiveDate> {
        self.tickers
            .iter()
            // Dates are ascending, so the last is the ticker's latest.
            .filter_map(|ticker| ticker.dates.last().copied())
            .max()
    }

    /// Enumerate every `steps`-length window of every ticker as a
    /// `(ticker_index, window_start)` pair. Tickers too short for a single
    /// window contribute none. This is the pool a [`crate::dataset::WindowDataset`]
    /// indexes into.
    pub(crate) fn enumerate_windows(&self, steps: usize) -> Vec<(u32, u32)> {
        let mut windows = Vec::new();

        for (ticker_index, ticker) in self.tickers.iter().enumerate() {
            if ticker.rows() < steps {
                continue;
            }
            let ticker_index = u32::try_from(ticker_index).unwrap();
            let last_start = ticker.rows() - steps;
            for start in 0..=u32::try_from(last_start).unwrap() {
                windows.push((ticker_index, start));
            }
        }

        windows
    }

    /// The first row each ticker occupies once every ticker's history is laid out
    /// end to end in ticker order, as the prefix sum of the per-ticker row counts.
    /// A `(ticker_index, start)` window from [`Self::enumerate_windows`] maps to the
    /// single absolute row `row_offsets()[ticker_index] + start`, which the batcher
    /// gathers against the resident tensors built by [`Self::resident_buffers`].
    pub(crate) fn row_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.tickers.len());
        let mut offset = 0u32;
        for ticker in &self.tickers {
            offsets.push(offset);
            offset += u32::try_from(ticker.rows()).unwrap();
        }
        offsets
    }

    /// Flatten every ticker's history into single contiguous buffers, in the same
    /// ticker order as [`Self::row_offsets`], so the batcher can upload them to the
    /// device once and then gather each batch on-device by absolute row. The three
    /// buffers share the row order: `features[row*5 .. row*5+5]`, `labels[row]`,
    /// and `rewards[row]` are the same trading day.
    pub(crate) fn resident_buffers(&self) -> ResidentBuffers {
        let stride = FEATURE_NAMES.len();
        let total: usize = self.tickers.iter().map(Ticker::rows).sum();

        let mut features = Vec::with_capacity(total * stride);
        let mut labels = Vec::with_capacity(total);
        let mut rewards = Vec::with_capacity(total);

        for ticker in &self.tickers {
            features.extend_from_slice(&ticker.features);
            labels.extend(ticker.labels.iter().map(|&label| i32::from(label)));
            rewards.extend_from_slice(&ticker.rewards);
        }

        ResidentBuffers {
            rows: total,
            features,
            labels,
            rewards,
        }
    }
}

/// Every ticker's history flattened into one set of row-aligned buffers, ready to
/// upload to the device as the batcher's resident gather tensors. Produced by
/// [`TickerStore::resident_buffers`]; `rows` is the shared length (`features` is
/// five wide per row).
pub(crate) struct ResidentBuffers {
    pub(crate) rows: usize,
    pub(crate) features: Vec<f32>,
    pub(crate) labels: Vec<i32>,
    pub(crate) rewards: Vec<f32>,
}

/// Synthetic builders shared across the crate's module tests, so `dataset` and
/// `batcher` can exercise a [`TickerStore`] without a parquet file on disk.
#[cfg(test)]
impl TickerStore {
    /// `n_tickers` tickers of `rows` rows each. Every feature slot in row `i`
    /// shares the value `base + i`, so `base` separates tickers and the per-row
    /// value rises monotonically. Labels cycle 0/1/2; dates ascend from the
    /// epoch.
    pub(crate) fn synthetic(n_tickers: i16, rows: i16) -> Self {
        let tickers = (0..n_tickers)
            .map(|t| make_ticker(&format!("t{t}"), f32::from(t) * 1000.0, rows))
            .collect();

        Self { tickers }
    }
}

/// Build one flat ticker of `rows` rows, used by the synthetic store builders
/// and the short-ticker test below.
#[cfg(test)]
fn make_ticker(name: &str, base: f32, rows: i16) -> Ticker {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let stride = FEATURE_NAMES.len();

    let dates = (0..i64::from(rows))
        .map(|i| epoch + chrono::Duration::days(i))
        .collect();

    let mut features = Vec::with_capacity(usize::from(rows.unsigned_abs()) * stride);
    for i in 0..rows {
        let value = base + f32::from(i);
        features.extend(std::iter::repeat_n(value, stride));
    }

    let labels = (0..rows).map(|i| u8::try_from(i % 3).unwrap()).collect();
    let rewards = (0..rows).map(|i| f32::from(i) * 0.01).collect();

    Ticker {
        name: name.into(),
        dates,
        features,
        labels,
        rewards,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(n_tickers: i16, rows: i16) -> TickerStore {
        TickerStore::synthetic(n_tickers, rows)
    }

    #[test]
    fn cross_sectional_zscore_centers_each_date() {
        // Two dates, three stocks each, with deliberately different scales so a
        // raw mean would be far from zero. Only `open_return` is exercised; the
        // other four columns just need to exist for the shared helper.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let next = epoch + chrono::Duration::days(1);
        let zeros = [0.0f32; 6];

        let frame = df!(
            "date" => [epoch, epoch, epoch, next, next, next],
            "open_return" => [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            "high_return" => zeros,
            "low_return" => zeros,
            "close_return" => zeros,
            "volume_return" => zeros,
        )
        .unwrap();

        let out = frame
            .lazy()
            .with_columns(cross_sectional_zscore())
            .collect()
            .unwrap();

        let standardized = out.column(&OPEN_RETURN).unwrap().f32().unwrap();

        // Each date's three values must average to ~0 after standardizing.
        let first: f32 = (0..3).map(|i| standardized.get(i).unwrap()).sum();
        let second: f32 = (3..6).map(|i| standardized.get(i).unwrap()).sum();
        assert!(first.abs() < 1e-5, "first date mean {first} not ~0");
        assert!(second.abs() < 1e-5, "second date mean {second} not ~0");

        // The ordering within a date is preserved, so the smallest stays lowest.
        assert!(standardized.get(0).unwrap() < standardized.get(2).unwrap());
    }

    #[test]
    fn enumerate_windows_skips_short_tickers() {
        let mut store = make_store(2, 6);
        // Append a ticker too short to yield any window of length 4.
        store.tickers.push(make_ticker("short", 9000.0, 3));

        let windows = store.enumerate_windows(4);

        // Each 6-row ticker yields 6 - 4 + 1 = 3 windows; the 3-row ticker none.
        assert_eq!(windows.len(), 6);
        // The short ticker is index 2, so no window references it.
        assert!(windows.iter().all(|&(ticker, _)| ticker != 2));
    }

    #[test]
    fn sample_tickers_keeps_a_reproducible_subset() {
        let store = make_store(10, 20);

        let first = store.sample_tickers(4, 7);
        assert_eq!(first.tickers.len(), 4);

        let again = make_store(10, 20).sample_tickers(4, 7);
        let first_names: Vec<_> = first.tickers.iter().map(|t| t.name.clone()).collect();
        let again_names: Vec<_> = again.tickers.iter().map(|t| t.name.clone()).collect();
        assert_eq!(first_names, again_names);
    }

    #[test]
    fn max_date_is_the_latest_row() {
        let store = make_store(3, 20);
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        assert_eq!(store.max_date(), Some(epoch + chrono::Duration::days(19)));
    }

    #[test]
    fn train_valid_split_partitions_rows() {
        let store = make_store(3, 20);
        let cutoff = NaiveDate::from_ymd_opt(1970, 1, 11).unwrap();

        let (train, valid) = store.train_valid_split(cutoff, 4).unwrap();

        assert_eq!(train.tickers.len(), 3);
        assert_eq!(valid.tickers.len(), 3);
        assert_eq!(train.tickers[0].rows(), 10);
        assert_eq!(valid.tickers[0].rows(), 10);
    }
}
