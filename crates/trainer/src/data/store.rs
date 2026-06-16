use std::path::Path;

use crate::data::label::compute_labels;
use chrono::NaiveDate;
use fastrand::Rng;
use polars::prelude::*;
use tracing::instrument;

use stock_model::features::{
    CLOSE, DATE, FEATURE, FEATURE_NAMES, HIGH, InferenceWindow, LOW, OPEN, TICKER, feature_array,
    standardized_features,
};

/// Scan the OHLCV parquet and run the shared feature transform.
fn scan_standardized(path: &Path) -> PolarsResult<LazyFrame> {
    let frame =
        LazyFrame::scan_parquet(PlRefPath::try_from_path(path)?, ScanArgsParquet::default())?;
    Ok(standardized_features(frame))
}

/// One stable-ordered frame per ticker, standardized features and raw prices selected.
fn load_groups(path: &Path) -> PolarsResult<Vec<DataFrame>> {
    let long = scan_standardized(path)?
        .select([
            col(TICKER),
            col(DATE),
            feature_array().alias(FEATURE),
            // Raw prices survive the z-score untouched; the barrier labels and the
            // backtest need them.
            col(OPEN),
            col(HIGH),
            col(LOW),
            col(CLOSE),
        ])
        .collect()?;

    long.partition_by_stable([TICKER], true)
}

/// Dates, flattened features, then the open, high, low, and close price vectors.
type TickerBuffers = (
    Vec<NaiveDate>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
);

/// Dates, flattened features, and the four raw price vectors from a group frame, all
/// in its row order.
fn ticker_buffers(frame: &DataFrame) -> PolarsResult<TickerBuffers> {
    let features: Vec<f32> = frame
        .column(&FEATURE)?
        .array()?
        .get_inner()
        .f32()?
        .into_no_null_iter()
        .collect();
    let dates: Vec<NaiveDate> = frame
        .column(&DATE)?
        .date()?
        .as_date_iter()
        .flatten()
        .collect();
    let open = price_column(frame, &OPEN)?;
    let high = price_column(frame, &HIGH)?;
    let low = price_column(frame, &LOW)?;
    let close = price_column(frame, &CLOSE)?;

    Ok((dates, features, open, high, low, close))
}

/// Pull a raw price column as `f32`, aligned with `dates`. `load_groups` already
/// ran the frame through `standardized_features`, whose `drop_nulls` removes any
/// row with a null in this column, so none survive to here.
fn price_column(frame: &DataFrame, name: &PlSmallStr) -> PolarsResult<Vec<f32>> {
    let column = frame.column(name)?.f32()?;
    polars_ensure!(!column.has_nulls(), InvalidOperation: "price column must not contain nulls");
    Ok(column.into_no_null_iter().collect())
}

/// One ticker's history flattened out of polars into plain buffers for fast window
/// extraction. Rows ascend by date; every vector shares that order, so `dates[i]`,
/// `labels[i]`, and `features[i*5 .. i*5+5]` are the same trading day.
#[derive(Clone)]
struct Ticker {
    name: PlSmallStr,
    dates: Vec<NaiveDate>,
    /// Row-major features, length `dates.len() * 5`.
    features: Vec<f32>,
    labels: Vec<u8>,
    /// Raw daily prices, untouched by the feature pipeline. The backtest fills buys
    /// at `low`, sells at `high`, marks at `close`, and uses `open` for pessimistic
    /// fills.
    open: Vec<f32>,
    high: Vec<f32>,
    low: Vec<f32>,
    close: Vec<f32>,
}

impl Ticker {
    fn rows(&self) -> usize {
        self.dates.len()
    }

    /// Split into rows `[0, at)` and `[at, rows)`, keeping every buffer aligned.
    fn split_at(&self, at: usize) -> (Ticker, Ticker) {
        let (dates_left, dates_right) = self.dates.split_at(at);
        let (features_left, features_right) = self.features.split_at(at * FEATURE_NAMES.len());
        let (labels_left, labels_right) = self.labels.split_at(at);
        let (open_left, open_right) = self.open.split_at(at);
        let (high_left, high_right) = self.high.split_at(at);
        let (low_left, low_right) = self.low.split_at(at);
        let (close_left, close_right) = self.close.split_at(at);

        let left = Ticker {
            name: self.name.clone(),
            dates: dates_left.to_vec(),
            features: features_left.to_vec(),
            labels: labels_left.to_vec(),
            open: open_left.to_vec(),
            high: high_left.to_vec(),
            low: low_left.to_vec(),
            close: close_left.to_vec(),
        };
        let right = Ticker {
            name: self.name.clone(),
            dates: dates_right.to_vec(),
            features: features_right.to_vec(),
            labels: labels_right.to_vec(),
            open: open_right.to_vec(),
            high: high_right.to_vec(),
            low: low_right.to_vec(),
            close: close_right.to_vec(),
        };

        (left, right)
    }
}

/// Backend-free store of every ticker's flattened history. Owns the loaded data
/// and the split and subset transforms, but knows nothing about windows, tensors,
/// or devices.
pub struct TickerStore {
    tickers: Vec<Ticker>,
}

impl TickerStore {
    /// Load the parquet into one flat [`Ticker`] per stock, standardized features and
    /// triple-barrier labels included. Each ticker's null first row and its trailing
    /// `horizon` label-less rows are dropped. `take_profit`, `stop_loss`, and
    /// `horizon` set the barriers for [`compute_labels`]. Non-positive closes
    /// are dropped as corrupt, since yfinance back-adjustment can drive a delisted
    /// history negative and poison the features.
    #[instrument(level = "info", skip_all, fields(path = %path.display()))]
    pub fn load(
        path: &Path,
        take_profit: f32,
        stop_loss: f32,
        horizon: usize,
    ) -> PolarsResult<Self> {
        let groups = load_groups(path)?;

        let mut tickers = Vec::with_capacity(groups.len());

        for group in groups {
            let height = group.height();

            // Too short for even one labeled row.
            if height <= horizon {
                continue;
            }

            let name: PlSmallStr = group.column(&TICKER)?.str()?.get(0).unwrap().into();

            // Already aligned to the kept rows after dropping the trailing horizon.
            let labels = compute_labels(
                group.column(&HIGH)?,
                group.column(&LOW)?,
                group.column(&CLOSE)?,
                take_profit,
                stop_loss,
                horizon,
            )?;

            let head = group.head(Some(height - horizon));
            let (dates, features, open, high, low, close) = ticker_buffers(&head)?;

            tickers.push(Ticker {
                name,
                dates,
                features,
                labels,
                open,
                high,
                low,
                close,
            });
        }

        Ok(Self { tickers })
    }

    /// Load every row of every ticker for the backtest, no trailing-horizon chop and
    /// no labels, so the most recent bars stay tradeable. `labels` are zeroed since
    /// inference never reads them.
    ///
    /// # Errors
    /// If the parquet cannot be scanned or a column has the wrong dtype.
    pub fn load_prices(path: &Path) -> PolarsResult<Self> {
        let groups = load_groups(path)?;

        let mut tickers = Vec::with_capacity(groups.len());

        for group in groups {
            if group.height() == 0 {
                continue;
            }

            let name: PlSmallStr = group.column(&TICKER)?.str()?.get(0).unwrap().into();
            let (dates, features, open, high, low, close) = ticker_buffers(&group)?;

            let rows = dates.len();
            tickers.push(Ticker {
                name,
                dates,
                features,
                labels: vec![0; rows],
                open,
                high,
                low,
                close,
            });
        }

        Ok(Self { tickers })
    }

    /// Every `steps`-length window whose last bar is on or after `cutoff`, as an
    /// [`InferenceWindow`] dated at that last bar. The window start may precede the
    /// cutoff, so a held-out day draws its `steps - 1` lookback from earlier bars.
    pub fn backtest_windows_since(&self, steps: usize, cutoff: NaiveDate) -> Vec<InferenceWindow> {
        let stride = FEATURE_NAMES.len();
        let mut windows = Vec::new();

        for ticker in &self.tickers {
            if ticker.rows() < steps {
                continue;
            }
            let last_start = ticker.rows() - steps;
            for start in 0..=last_start {
                let last = start + steps - 1;
                if ticker.dates[last] < cutoff {
                    continue;
                }
                windows.push(InferenceWindow {
                    ticker: ticker.name.to_string(),
                    date: ticker.dates[last],
                    features: ticker.features[start * stride..(last + 1) * stride].to_vec(),
                });
            }
        }

        windows
    }

    /// Each ticker's raw daily prices for the backtest, one [`TickerQuotes`] per
    /// ticker with the price vectors aligned to `dates`.
    pub fn quotes(&self) -> Vec<TickerQuotes> {
        self.tickers
            .iter()
            .map(|ticker| TickerQuotes {
                ticker: ticker.name.to_string(),
                dates: ticker.dates.clone(),
                open: ticker.open.clone(),
                high: ticker.high.clone(),
                low: ticker.low.clone(),
                close: ticker.close.clone(),
            })
            .collect()
    }

    /// Randomly keep `count` tickers, reproducible by `seed`, for overfit
    /// diagnostics. A no-op when `count` is at least the ticker count.
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

    /// Split every ticker at `cutoff` into an earlier train store and a later valid
    /// store. A side with fewer than `steps` rows is dropped. Errors if either side
    /// ends up empty.
    #[instrument(level = "info", skip_all, fields(steps))]
    pub fn train_valid_split(&self, cutoff: NaiveDate, steps: usize) -> PolarsResult<(Self, Self)> {
        let mut train_tickers = Vec::with_capacity(self.tickers.len());
        let mut valid_tickers = Vec::with_capacity(self.tickers.len());

        for ticker in &self.tickers {
            // Dates ascend, so this is the count of rows before the cutoff.
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

    /// Number of tickers in the store.
    pub fn ticker_count(&self) -> usize {
        self.tickers.len()
    }

    /// Per-class label counts, indexed Sell 0, Hold 1, Buy 2.
    pub fn label_counts(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        for ticker in &self.tickers {
            for &label in &ticker.labels {
                counts[usize::from(label)] += 1;
            }
        }
        counts
    }

    /// The latest date across every ticker, to anchor the split. `None` only when no
    /// ticker has a dated row.
    pub fn max_date(&self) -> Option<NaiveDate> {
        self.tickers
            .iter()
            // Dates ascend, so the last is the ticker's latest.
            .filter_map(|ticker| ticker.dates.last().copied())
            .max()
    }

    /// Every `steps`-length window as a `(ticker_index, window_start)` pair, the pool
    /// a [`crate::dataset::WindowDataset`] indexes into. Short tickers contribute none.
    pub(crate) fn enumerate_windows(&self, steps: usize) -> Vec<(u32, u32)> {
        let mut windows = Vec::new();

        for (ticker_index, ticker) in self.tickers.iter().enumerate() {
            if ticker.rows() < steps {
                continue;
            }
            let ticker_index = u32::try_from(ticker_index)
                .expect("ticker count exceeds u32; far larger than supported");
            let last_start = ticker.rows() - steps;
            for start in 0..=u32::try_from(last_start)
                .expect("row index exceeds u32; ticker far larger than supported")
            {
                windows.push((ticker_index, start));
            }
        }

        windows
    }

    /// Prefix sum of per-ticker row counts: a `(ticker_index, start)` window maps to
    /// the absolute row `row_offsets()[ticker_index] + start`.
    pub(crate) fn row_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.tickers.len());
        let mut offset = 0u32;
        for ticker in &self.tickers {
            offsets.push(offset);
            offset += u32::try_from(ticker.rows())
                .expect("row index exceeds u32; ticker far larger than supported");
        }
        offsets
    }

    /// Flatten every ticker's history into contiguous buffers in [`Self::row_offsets`]
    /// order, so the batcher uploads once and gathers each batch on-device by absolute
    /// row. The buffers share row order.
    pub(crate) fn resident_buffers(&self) -> ResidentBuffers {
        let stride = FEATURE_NAMES.len();
        let total: usize = self.tickers.iter().map(Ticker::rows).sum();

        let mut features = Vec::with_capacity(total * stride);
        let mut labels = Vec::with_capacity(total);

        for ticker in &self.tickers {
            features.extend_from_slice(&ticker.features);
            labels.extend(ticker.labels.iter().map(|&label| i32::from(label)));
        }

        ResidentBuffers {
            rows: total,
            features,
            labels,
        }
    }
}

/// Every ticker's history in row-aligned buffers, the batcher's resident gather
/// tensors. `rows` is the shared length; `features` is five wide per row.
pub(crate) struct ResidentBuffers {
    pub(crate) rows: usize,
    pub(crate) features: Vec<f32>,
    pub(crate) labels: Vec<i32>,
}

/// One ticker's raw daily prices for the backtest. The price vectors share `dates`'
/// row order.
pub struct TickerQuotes {
    pub ticker: String,
    pub dates: Vec<NaiveDate>,
    pub open: Vec<f32>,
    pub high: Vec<f32>,
    pub low: Vec<f32>,
    pub close: Vec<f32>,
}

/// Synthetic builders for the crate's module tests, so they can exercise a
/// [`TickerStore`] without a parquet file.
#[cfg(test)]
impl TickerStore {
    /// `n_tickers` tickers of `rows` rows each. Row `i`'s feature slots all hold
    /// `base + i`, so `base` separates tickers. Labels cycle 0/1/2.
    pub(crate) fn synthetic(n_tickers: i16, rows: i16) -> Self {
        let tickers = (0..n_tickers)
            .map(|t| make_ticker(&format!("t{t}"), f32::from(t) * 1000.0, rows))
            .collect();

        Self { tickers }
    }
}

/// Build one flat ticker of `rows` rows for the tests.
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

    // Prices rise with the row, one-unit intraday range.
    let close: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();
    let open = close.clone();
    let high: Vec<f32> = close.iter().map(|price| price + 1.0).collect();
    let low: Vec<f32> = close.iter().map(|price| price - 1.0).collect();

    Ticker {
        name: name.into(),
        dates,
        features,
        labels,
        open,
        high,
        low,
        close,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(n_tickers: i16, rows: i16) -> TickerStore {
        TickerStore::synthetic(n_tickers, rows)
    }

    #[test]
    fn enumerate_windows_skips_short_tickers() {
        let mut store = make_store(2, 6);
        // Too short for any window of length 4.
        store.tickers.push(make_ticker("short", 9000.0, 3));

        let windows = store.enumerate_windows(4);

        // Each 6-row ticker yields 3 windows; the 3-row ticker none.
        assert_eq!(windows.len(), 6);
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
    fn quotes_align_prices_with_dates() {
        let store = make_store(2, 8);
        let quotes = store.quotes();

        assert_eq!(quotes.len(), 2);
        let first = &quotes[0];
        // Every price vector matches the date count.
        assert_eq!(first.dates.len(), 8);
        assert_eq!(first.open.len(), 8);
        assert_eq!(first.high.len(), 8);
        assert_eq!(first.low.len(), 8);
        assert_eq!(first.close.len(), 8);
        // One-unit range around the close.
        assert!((first.high[3] - first.close[3] - 1.0).abs() < 1e-6);
        assert!((first.close[3] - first.low[3] - 1.0).abs() < 1e-6);
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
