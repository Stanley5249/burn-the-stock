use crate::label::{LABEL_THRESHOLD, compute_labels};
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use std::collections::HashMap;

const CODE: PlSmallStr = PlSmallStr::from_static("code");
const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
const DATE: PlSmallStr = PlSmallStr::from_static("date");
const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");
const LABEL: PlSmallStr = PlSmallStr::from_static("label");
const INDUSTRY: PlSmallStr = PlSmallStr::from_static("industry");

const OPEN: PlSmallStr = PlSmallStr::from_static("open");
const HIGH: PlSmallStr = PlSmallStr::from_static("high");
const LOW: PlSmallStr = PlSmallStr::from_static("low");
const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

pub(crate) const FEATURE_NAMES: [PlSmallStr; 5] = [OPEN, HIGH, LOW, CLOSE, VOLUME];

/// Offsets into a flattened OHLCV row. OHLC occupy the first four slots, so the
/// close sits at index 3 and volume, the lone non-price feature, sits last.
const CLOSE_OFFSET: usize = 3;
const VOLUME_OFFSET: usize = 4;

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

/// Normalize one flattened `[steps * 5]` OHLCV window in place.
///
/// A GRU should learn the shape of a window, not the price level it happens to
/// sit at, so OHLC are divided by the window's last close and land near 1.0.
/// Volume lives on a wildly different scale, so it is `log1p` compressed and
/// then z-scored within the window to match the price ratios.
#[allow(clippy::cast_precision_loss)] // `steps` is a small window length
pub(crate) fn normalize_window(window: &mut [f32], steps: usize) {
    let stride = FEATURE_NAMES.len();

    let last_close = window[(steps - 1) * stride + CLOSE_OFFSET];

    for row in window.chunks_mut(stride) {
        if last_close != 0.0 {
            for price in &mut row[..VOLUME_OFFSET] {
                *price /= last_close;
            }
        }
        row[VOLUME_OFFSET] = row[VOLUME_OFFSET].max(0.0).ln_1p();
    }

    let volumes = || window.iter().skip(VOLUME_OFFSET).step_by(stride);

    let mean = volumes().sum::<f32>() / steps as f32;
    let variance = volumes().map(|v| (v - mean).powi(2)).sum::<f32>() / steps as f32;
    let std = variance.sqrt();

    for row in window.chunks_mut(stride) {
        row[VOLUME_OFFSET] = if std > f32::EPSILON {
            (row[VOLUME_OFFSET] - mean) / std
        } else {
            0.0
        };
    }
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
    /// Row-major OHLCV, length `dates.len() * 5`.
    features: Vec<f32>,
    /// One action label per row.
    labels: Vec<u8>,
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

        let left = Ticker {
            name: self.name.clone(),
            dates: dates_left.to_vec(),
            features: features_left.to_vec(),
            labels: labels_left.to_vec(),
        };
        let right = Ticker {
            name: self.name.clone(),
            dates: dates_right.to_vec(),
            features: features_right.to_vec(),
            labels: labels_right.to_vec(),
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
    /// Load the parquet file into one flat [`Ticker`] per stock.
    ///
    /// The final label-less row of each ticker is dropped so every kept row has
    /// a forward window for its label. Tickers too short to yield a single
    /// window are kept but simply produce no windows later.
    pub fn load(path: &str) -> PolarsResult<Self> {
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
                feature_expr.alias(FEATURE),
                col(CLOSE).cast(DataType::Float32),
            ])
            .sort([TICKER, DATE], SortMultipleOptions::new())
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

            // Labels come from the full close series but are one short, already
            // aligned to the rows we keep after dropping the label-less last row.
            let labels: Vec<u8> = compute_labels(group.column(&CLOSE)?, LABEL, LABEL_THRESHOLD)?
                .u8()?
                .into_no_null_iter()
                .collect();

            let head = group.head(Some(height - 1));

            // Flatten the Array<f32, 5> feature column into row-major OHLCV once,
            // so window extraction later is a contiguous slice.
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

    /// Materialize one window into `(normalized OHLCV [steps * 5], industry
    /// index, label)`. The window is a contiguous span of the flat OHLCV buffer
    /// copied out and normalized in place, the label is the action on the
    /// window's last day, and the industry is the resolved bucket (0 when no
    /// industries are attached).
    pub(crate) fn window(
        &self,
        ticker_index: u32,
        start: u32,
        steps: usize,
    ) -> (Vec<f32>, usize, i32) {
        let ticker = &self.tickers[ticker_index as usize];
        let start = start as usize;
        let stride = FEATURE_NAMES.len();

        let begin = start * stride;
        let end = begin + steps * stride;
        let mut technical = ticker.features[begin..end].to_vec();

        normalize_window(&mut technical, steps);

        let label = i32::from(ticker.labels[start + steps - 1]);
        let industry = self
            .industry_of
            .get(ticker_index as usize)
            .copied()
            .unwrap_or(0);

        (technical, industry, label)
    }
}

/// Synthetic builders shared across the crate's module tests, so `dataset` and
/// `batcher` can exercise a [`TickerStore`] without a parquet file on disk.
#[cfg(test)]
impl TickerStore {
    /// `n_tickers` tickers of `rows` rows each. Every OHLCV slot in row `i`
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

    Ticker {
        name: name.into(),
        dates,
        features,
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(n_tickers: i16, rows: i16) -> TickerStore {
        TickerStore::synthetic(n_tickers, rows)
    }

    #[test]
    fn normalize_window_scales_price_and_volume() {
        // Three rows of (open, high, low, close, volume), last close is 16.
        let mut window = vec![
            10.0, 11.0, 9.0, 10.0, 100.0, //
            12.0, 13.0, 11.0, 12.0, 200.0, //
            14.0, 15.0, 13.0, 16.0, 300.0, //
        ];

        normalize_window(&mut window, 3);

        // OHLC are divided by the window's last close, so it lands exactly on 1.
        assert!((window[2 * 5 + CLOSE_OFFSET] - 1.0).abs() < 1e-6);
        assert!((window[3] - 10.0 / 16.0).abs() < 1e-6);

        // Volume is z-scored, so it has near-zero mean and stays monotonic.
        let volumes = [window[4], window[9], window[14]];
        let mean = volumes.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-6);
        assert!(volumes[0] < volumes[1] && volumes[1] < volumes[2]);
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
