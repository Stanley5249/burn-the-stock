use crate::label::compute_labels_rewards;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use std::collections::HashMap;

const CODE: PlSmallStr = PlSmallStr::from_static("code");
const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
const DATE: PlSmallStr = PlSmallStr::from_static("date");
const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");
const INDUSTRY: PlSmallStr = PlSmallStr::from_static("industry");

const OPEN: PlSmallStr = PlSmallStr::from_static("open");
const HIGH: PlSmallStr = PlSmallStr::from_static("high");
const LOW: PlSmallStr = PlSmallStr::from_static("low");
const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

const LOG_RETURN: PlSmallStr = PlSmallStr::from_static("log_return");
const VOLUME_RATIO: PlSmallStr = PlSmallStr::from_static("volume_ratio");
const HL_RANGE: PlSmallStr = PlSmallStr::from_static("hl_range");
const GAP: PlSmallStr = PlSmallStr::from_static("gap");
const BODY: PlSmallStr = PlSmallStr::from_static("body");

/// The five stationary features flattened per row, in column order. Width stays
/// at five so the model, batcher, and tensor shapes are unchanged from the raw
/// OHLCV layout this replaced.
pub(crate) const FEATURE_NAMES: [PlSmallStr; 5] = [LOG_RETURN, VOLUME_RATIO, HL_RANGE, GAP, BODY];

/// Rolling window, in trading days, for the volume average that `volume_ratio`
/// divides by. The `min_periods` equals this window, so the leading rows of every
/// ticker are null and dropped as warmup at load.
const VOLUME_WINDOW: usize = 20;

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

/// Build the five stationary feature expressions from the raw OHLCV columns.
///
/// Each is an honest stationarity transform of OHLCV, no invented indicators:
/// price levels become log returns, raw volume becomes a log ratio to its own
/// rolling average, and the intraday bar is described by scale-free range and
/// body shape. The per-ticker `shift` and `rolling_mean` run `over` the ticker so
/// they never reach across stock boundaries, which means the caller must have
/// sorted by `[ticker, date]` first.
fn stationary_features() -> [Expr; 5] {
    let prev_close = col(CLOSE).shift(lit(1)).over([col(TICKER)]);

    let volume_mean = col(VOLUME)
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: VOLUME_WINDOW,
            min_periods: VOLUME_WINDOW,
            ..Default::default()
        })
        .over([col(TICKER)]);

    let natural_log = || lit(std::f64::consts::E);
    let high_low = col(HIGH) - col(LOW);

    // Volume is a count, so a no-trade day has volume 0 and the bare ratio's log
    // is `-inf`. Add one share to both sides (log1p style) to keep it finite
    // while leaving real, large volumes essentially unchanged.
    let volume_ratio = ((col(VOLUME) + lit(1.0)) / (volume_mean + lit(1.0))).log(natural_log());

    [
        (col(CLOSE) / prev_close.clone())
            .log(natural_log())
            .alias(LOG_RETURN),
        volume_ratio.alias(VOLUME_RATIO),
        (high_low.clone() / col(CLOSE)).alias(HL_RANGE),
        ((col(OPEN) - prev_close.clone()) / prev_close).alias(GAP),
        ((col(CLOSE) - col(OPEN)) / (high_low + lit(1e-8))).alias(BODY),
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
/// before the warmup `drop_nulls`, so null warmup values are excluded from the
/// per-date statistics by polars and then dropped.
fn cross_sectional_zscore() -> [Expr; 5] {
    FEATURE_NAMES.map(|name| {
        let centered = col(name.clone()) - col(name.clone()).mean().over([col(DATE)]);
        let spread = col(name.clone()).std(1).over([col(DATE)]) + lit(1e-8);
        (centered / spread).alias(name)
    })
}

/// Build the `ticker name -> industry index` map from a frame with `ticker`
/// and `industry` string columns. Distinct industries are indexed in sorted
/// order for a stable encoding, and the returned width includes one extra
/// bucket for tickers with an unknown industry.
pub(crate) fn index_industries(
    frame: &DataFrame,
) -> PolarsResult<(HashMap<PlSmallStr, usize>, usize)> {
    let names = frame.column(&TICKER)?.str()?;
    let industries = frame.column(&INDUSTRY)?.str()?;

    let mut distinct: Vec<&str> = industries.into_iter().flatten().collect();
    distinct.sort_unstable();
    distinct.dedup();

    let index_of: HashMap<&str, usize> = distinct
        .iter()
        .enumerate()
        .map(|(index, industry)| (*industry, index))
        .collect();

    let mut codes = HashMap::new();
    for (name, industry) in names.into_iter().zip(industries) {
        if let (Some(name), Some(industry)) = (name, industry) {
            codes.insert(name.into(), index_of[industry]);
        }
    }

    // The trailing bucket catches tickers whose industry is null or missing.
    Ok((codes, distinct.len() + 1))
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
    /// Signed forward return to the next swing extreme, one per row.
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

/// Backend-free store of every ticker's flattened history plus the industry
/// encoding. This is the data layer: it owns the loaded parquet data and the
/// train/valid split and ticker-subset transforms, but knows nothing about
/// windows, tensors, or devices. A [`crate::dataset::WindowDataset`] borrows it
/// behind an `Arc` to enumerate and materialize training samples.
pub struct TickerStore {
    tickers: Vec<Ticker>,
    /// Per-ticker industry index, aligned with `tickers`. Empty until
    /// [`Self::attach_industries`] runs; an attached ticker with no known
    /// industry resolves to the trailing unknown bucket (`n_industries - 1`).
    industry_of: Vec<usize>,
    /// One-hot width for the industry feature: distinct industries plus a final
    /// unknown bucket. Zero means no categorical feature is attached.
    n_industries: usize,
}

impl TickerStore {
    /// Load the parquet file into one flat [`Ticker`] per stock, deriving the
    /// stationary feature window from the raw OHLCV columns in polars once and
    /// then standardizing each feature cross-sectionally across the whole loaded
    /// universe per date.
    ///
    /// The leading rows of each ticker carry null features while the volume
    /// rolling average warms up, so they are dropped. The final label-less row of
    /// each ticker is then dropped so every kept row has a forward window for its
    /// label. Tickers too short to yield a single window are kept but simply
    /// produce no windows later. `threshold` sets the swing-reversal magnitude
    /// passed to [`compute_labels_rewards`].
    ///
    /// Rows with a non-positive close are dropped as corrupt. yfinance back
    /// adjustment can drive a delisted ticker's whole history negative, which
    /// would otherwise poison the log-return and ratio features and the swing
    /// labels that compare prices.
    pub fn load(path: &str, threshold: f32) -> PolarsResult<Self> {
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
            // Sort before deriving features so the per-ticker `shift` and
            // `rolling_mean` inside `stationary_features` see rows in date order.
            .sort([TICKER, DATE], SortMultipleOptions::new())
            .with_columns(stationary_features())
            // Standardize each feature across every stock on the same date. This
            // runs over the full loaded universe, before any ticker subsetting,
            // so the cross-section is the whole market.
            .with_columns(cross_sectional_zscore())
            // The only nulls are the warmup rows whose volume average and
            // previous close had too few prior days, so drop them across all
            // columns before packing the feature array.
            .drop_nulls(None)
            .select([
                col(TICKER),
                col(DATE),
                feature_expr.alias(FEATURE),
                col(CLOSE),
            ])
            .collect()?;

        let groups = long.partition_by_stable([TICKER], true)?;

        let mut tickers = Vec::with_capacity(groups.len());

        for group in groups {
            let height = group.height();

            // A ticker needs at least two rows: one to keep and one to drop as
            // the label-less last row.
            if height <= 1 {
                continue;
            }

            let name: PlSmallStr = group.column(&TICKER)?.str()?.get(0).unwrap().into();

            // Labels and rewards come from the full close series but are one short,
            // already aligned to the rows we keep after dropping the label-less last
            // row.
            let (labels, rewards) = compute_labels_rewards(group.column(&CLOSE)?, threshold)?;

            let head = group.head(Some(height - 1));

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

        Ok(Self {
            tickers,
            industry_of: Vec::new(),
            n_industries: 0,
        })
    }

    /// Attach the industry categorical feature from a `tickers.parquet` written
    /// by the `tickers` prefetch bin (columns `market`, `code`, `industry`).
    ///
    /// Distinct industries get stable indices in sorted order, and a final
    /// bucket absorbs every ticker whose industry is null or absent from the
    /// file. Run before [`Self::train_valid_split`] so the per-ticker encoding
    /// propagates to both sides.
    pub fn attach_industries(mut self, path: &str) -> PolarsResult<Self> {
        let frame = LazyFrame::scan_parquet(PlRefPath::new(path), ScanArgsParquet::default())?
            .select([
                col(CODE).cast(DataType::String).alias(TICKER),
                col(INDUSTRY).cast(DataType::String),
            ])
            .collect()?;

        let (industry_codes, n_industries) = index_industries(&frame)?;
        let unknown = n_industries - 1;

        self.industry_of = self
            .tickers
            .iter()
            .map(|ticker| industry_codes.get(&ticker.name).copied().unwrap_or(unknown))
            .collect();
        self.n_industries = n_industries;

        Ok(self)
    }

    /// Randomly keep `count` tickers, chosen with `seed` so the subset is
    /// reproducible. Asking for at least as many tickers as exist is a no-op.
    /// Used to carve a small subset for overfit diagnostics, where we want a
    /// representative random sample rather than the first `count` by sort order.
    pub fn sample_tickers(mut self, count: usize, seed: u64) -> Self {
        if count >= self.tickers.len() {
            return self;
        }

        let has_industries = !self.industry_of.is_empty();

        let mut rng = Rng::with_seed(seed);
        let indices = rng.choose_multiple(0..self.tickers.len(), count);

        let mut tickers = Vec::with_capacity(count);
        let mut industry_of = Vec::with_capacity(if has_industries { count } else { 0 });
        for index in indices {
            tickers.push(self.tickers[index].clone());
            if has_industries {
                industry_of.push(self.industry_of[index]);
            }
        }

        self.tickers = tickers;
        self.industry_of = industry_of;
        self
    }

    /// Split every ticker at `cutoff` into an earlier train store and a later
    /// valid store. Tickers whose train or valid side has fewer than `steps`
    /// rows are dropped from that side. Both stores keep the industry encoding;
    /// errors if either side ends up empty.
    pub fn train_valid_split(&self, cutoff: NaiveDate, steps: usize) -> PolarsResult<(Self, Self)> {
        let has_industries = !self.industry_of.is_empty();

        let mut train_tickers = Vec::with_capacity(self.tickers.len());
        let mut train_industry = Vec::new();
        let mut valid_tickers = Vec::with_capacity(self.tickers.len());
        let mut valid_industry = Vec::new();

        for (index, ticker) in self.tickers.iter().enumerate() {
            // Dates are ascending, so `partition_point` is the count of rows
            // strictly before the cutoff.
            let split = ticker.dates.partition_point(|&day| day < cutoff);
            let (train, valid) = ticker.split_at(split);

            if train.rows() >= steps {
                train_tickers.push(train);
                if has_industries {
                    train_industry.push(self.industry_of[index]);
                }
            }
            if valid.rows() >= steps {
                valid_tickers.push(valid);
                if has_industries {
                    valid_industry.push(self.industry_of[index]);
                }
            }
        }

        polars_ensure!(
            !train_tickers.is_empty() && !valid_tickers.is_empty(),
            NoData: "train/valid split left one side empty; check cutoff and steps"
        );

        let train = Self {
            tickers: train_tickers,
            industry_of: train_industry,
            n_industries: self.n_industries,
        };
        let valid = Self {
            tickers: valid_tickers,
            industry_of: valid_industry,
            n_industries: self.n_industries,
        };

        Ok((train, valid))
    }

    /// One-hot width of the industry feature, needed to size the model's
    /// categorical branch.
    pub fn n_industries(&self) -> usize {
        self.n_industries
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

    /// Materialize one window into `(stationary features [steps * 5], industry
    /// index, label, reward)`. The window is a contiguous span of the flat feature
    /// buffer copied out as is, the label and reward are the action and forward
    /// return on the window's last day, and the industry is the resolved bucket
    /// (0 when no industries are attached). The features were already made
    /// stationary at load, so no per-window rescaling happens here.
    pub(crate) fn window(
        &self,
        ticker_index: u32,
        start: u32,
        steps: usize,
    ) -> (Vec<f32>, usize, i32, f32) {
        let ticker = &self.tickers[ticker_index as usize];
        let start = start as usize;
        let stride = FEATURE_NAMES.len();

        let begin = start * stride;
        let end = begin + steps * stride;
        let technical = ticker.features[begin..end].to_vec();

        let last_day = start + steps - 1;
        let label = i32::from(ticker.labels[last_day]);
        let reward = ticker.rewards[last_day];
        let industry = self
            .industry_of
            .get(ticker_index as usize)
            .copied()
            .unwrap_or(0);

        (technical, industry, label, reward)
    }
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

        Self {
            tickers,
            industry_of: Vec::new(),
            n_industries: 0,
        }
    }

    /// Attach an explicit per-ticker industry encoding, bypassing the parquet
    /// `attach_industries` path.
    pub(crate) fn set_industries(mut self, industry_of: Vec<usize>, n_industries: usize) -> Self {
        self.industry_of = industry_of;
        self.n_industries = n_industries;
        self
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
        // raw mean would be far from zero. Only `log_return` is exercised; the
        // other four columns just need to exist for the shared helper.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let next = epoch + chrono::Duration::days(1);
        let zeros = [0.0f32; 6];

        let frame = df!(
            "date" => [epoch, epoch, epoch, next, next, next],
            "log_return" => [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            "volume_ratio" => zeros,
            "hl_range" => zeros,
            "gap" => zeros,
            "body" => zeros,
        )
        .unwrap();

        let out = frame
            .lazy()
            .with_columns(cross_sectional_zscore())
            .collect()
            .unwrap();

        let standardized = out.column(&LOG_RETURN).unwrap().f32().unwrap();

        // Each date's three values must average to ~0 after standardizing.
        let first: f32 = (0..3).map(|i| standardized.get(i).unwrap()).sum();
        let second: f32 = (3..6).map(|i| standardized.get(i).unwrap()).sum();
        assert!(first.abs() < 1e-5, "first date mean {first} not ~0");
        assert!(second.abs() < 1e-5, "second date mean {second} not ~0");

        // The ordering within a date is preserved, so the smallest stays lowest.
        assert!(standardized.get(0).unwrap() < standardized.get(2).unwrap());
    }

    #[test]
    fn index_industries_encodes_and_buckets_unknown() {
        let frame = df!(
            "ticker" => ["tse_2330", "otc_1240", "tse_2317", "tse_9999"],
            "industry" => [Some("24"), Some("16"), Some("24"), None],
        )
        .unwrap();

        let (codes, n_industries) = index_industries(&frame).unwrap();

        // Two distinct industries ("16", "24") plus the unknown bucket.
        assert_eq!(n_industries, 3);
        // Sorted order: "16" -> 0, "24" -> 1.
        assert_eq!(codes[&PlSmallStr::from_static("otc_1240")], 0);
        assert_eq!(codes[&PlSmallStr::from_static("tse_2330")], 1);
        assert_eq!(codes[&PlSmallStr::from_static("tse_2317")], 1);
        // The null-industry ticker is left out of the map entirely.
        assert!(!codes.contains_key(&PlSmallStr::from_static("tse_9999")));
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
