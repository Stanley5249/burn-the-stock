//! Per-ticker standardized frames, the data store shared by training, backtest, and
//! the live trader. Feature engineering runs once over the full universe, then the
//! frame is partitioned by ticker. Windowing reads each ticker's contiguous feature
//! rows straight out of polars' buffer and gathers a batch into one flat host buffer,
//! a single upload per batch rather than a slice and stack per window.

use std::path::Path;

use burn::prelude::*;
use chrono::NaiveDate;
use miette::{IntoDiagnostic, Result, ensure};
use polars::prelude::*;

use crate::features::{
    CLOSE, DATE, FEATURE, HIGH, LOW, OPEN, TICKER, feature_array, standardized_features,
};
use crate::model::NUM_FEATURES;

/// Triple-barrier class per row, added by the trainer. Absent in the trader and the
/// backtest, which never read labels.
pub const LABEL: PlSmallStr = PlSmallStr::from_static("label");

/// One standardized frame per ticker, each sorted by date and holding `TICKER`, `DATE`,
/// the packed `FEATURE` array, and the raw `OPEN`/`HIGH`/`LOW`/`CLOSE` prices. The
/// trainer adds a `LABEL` column on top.
pub struct TickerFrames {
    pub frames: Vec<DataFrame>,
}

impl TickerFrames {
    /// Standardize a raw OHLCV frame and partition it by ticker. The single place the
    /// universe is turned into per-ticker frames.
    ///
    /// # Errors
    /// If the frame cannot be collected or partitioned.
    pub fn from_lazy(raw: LazyFrame) -> PolarsResult<Self> {
        let long = standardized_features(raw)
            .select([
                col(TICKER),
                col(DATE),
                feature_array().alias(FEATURE),
                col(OPEN),
                col(HIGH),
                col(LOW),
                col(CLOSE),
            ])
            .collect()?;

        Ok(Self {
            frames: long.partition_by_stable([TICKER], true)?,
        })
    }

    /// Scan the OHLCV parquet and build the per-ticker store, all rows, no labels.
    ///
    /// # Errors
    /// If the parquet cannot be scanned or standardized.
    pub fn load(path: &Path) -> PolarsResult<Self> {
        let raw =
            LazyFrame::scan_parquet(PlRefPath::try_from_path(path)?, ScanArgsParquet::default())?;
        Self::from_lazy(raw)
    }

    /// Each ticker's flattened f32 `FEATURE` values as one contiguous `Series`
    /// (`rows * NUM_FEATURES`, row-major), reusing polars' Arc-backed buffer. The gather
    /// source every window build reads from, paired with [`gather_windows`]. `rechunk`
    /// guarantees a single chunk so the window slice is contiguous.
    ///
    /// # Errors
    /// If a frame lacks an `f32` feature array column.
    pub fn feature_series(&self) -> PolarsResult<Vec<Series>> {
        self.frames
            .iter()
            .map(|frame| Ok(frame.column(&FEATURE)?.array()?.get_inner().rechunk()))
            .collect()
    }

    /// Each ticker's `LABEL` column as a `Series`, aligned to its feature rows. Only
    /// present after the trainer labels the store.
    ///
    /// # Errors
    /// If a frame lacks a label column.
    pub fn label_series(&self) -> PolarsResult<Vec<Series>> {
        self.frames
            .iter()
            .map(|frame| Ok(frame.column(&LABEL)?.as_materialized_series().clone()))
            .collect()
    }

    /// Each kept ticker's most recent `steps`-day window, keyed by ticker and its
    /// last-bar date. Tickers shorter than `steps` are skipped. Pair with
    /// [`feature_series`] and [`gather_windows`] to build the live trader's input,
    /// the same windowing path the backtest uses.
    ///
    /// [`feature_series`]: Self::feature_series
    ///
    /// # Errors
    /// If a frame's ticker or date column is malformed.
    ///
    /// # Panics
    /// If a retained frame has no rows, impossible since `rows >= steps >= 1`.
    pub fn latest_windows(&self, steps: usize) -> PolarsResult<Vec<Window>> {
        let mut windows = Vec::new();
        for (ticker_index, frame) in self.frames.iter().enumerate() {
            let rows = frame.height();
            if rows < steps {
                continue;
            }
            let date = date_endpoints(frame)?.expect("height >= steps >= 1").1;
            windows.push(Window {
                ticker_index,
                start: rows - steps,
                ticker: ticker_name(frame)?,
                date,
            });
        }
        Ok(windows)
    }

    /// Every window of length `steps`, in ticker-then-date order, as a [`StockItem`].
    /// Short tickers contribute none. The training dataset's pool.
    #[must_use]
    pub fn enumerate_windows(&self, steps: usize) -> Vec<StockItem> {
        let mut windows = Vec::new();
        for (ticker, frame) in self.frames.iter().enumerate() {
            let rows = frame.height();
            if rows < steps {
                continue;
            }
            for start in 0..=rows - steps {
                windows.push(StockItem { ticker, start });
            }
        }
        windows
    }

    /// Every `steps`-length window whose last bar is on or after `cutoff`, keyed by
    /// ticker and that last-bar date. The window may start before the cutoff, so a
    /// held-out day draws its lookback from earlier bars. The backtest's signals.
    ///
    /// # Errors
    /// If a frame's ticker or date column is malformed.
    pub fn windows_since(&self, steps: usize, cutoff: NaiveDate) -> PolarsResult<Vec<Window>> {
        let mut windows = Vec::new();
        for (ticker_index, frame) in self.frames.iter().enumerate() {
            let rows = frame.height();
            if rows < steps {
                continue;
            }
            let dates = date_buffer(frame)?;
            let ticker = ticker_name(frame)?;
            for start in 0..=rows - steps {
                let last = start + steps - 1;
                if dates[last] < cutoff {
                    continue;
                }
                windows.push(Window {
                    ticker_index,
                    start,
                    ticker: ticker.clone(),
                    date: dates[last],
                });
            }
        }
        Ok(windows)
    }

    /// Each ticker's raw daily prices for the backtest, aligned to its dates.
    ///
    /// # Errors
    /// If a frame's price or date column is malformed.
    pub fn quotes(&self) -> PolarsResult<Vec<TickerQuotes>> {
        self.frames.iter().map(quotes_of).collect()
    }

    /// The earliest and latest dated row across every ticker, to anchor a split. `None`
    /// when no ticker has a dated row.
    ///
    /// # Errors
    /// If a frame's date column is malformed.
    pub fn date_bounds(&self) -> PolarsResult<Option<(NaiveDate, NaiveDate)>> {
        let mut bounds: Option<(NaiveDate, NaiveDate)> = None;
        for frame in &self.frames {
            if let Some((first, last)) = date_endpoints(frame)? {
                bounds = Some(match bounds {
                    Some((lo, hi)) => (lo.min(first), hi.max(last)),
                    None => (first, last),
                });
            }
        }
        Ok(bounds)
    }

    /// Every ticker symbol in frame order, the run's universe.
    ///
    /// # Errors
    /// If a frame's ticker column is malformed.
    pub fn tickers(&self) -> PolarsResult<Vec<String>> {
        self.frames.iter().map(ticker_name).collect()
    }

    /// Split every ticker at `cutoff` into an earlier-train and a later-valid store. A
    /// side with fewer than `steps` rows is dropped. Errors if either side ends up empty.
    ///
    /// # Errors
    /// If a frame's date column is malformed or a split leaves one side empty.
    pub fn train_valid_split(&self, cutoff: NaiveDate, steps: usize) -> Result<(Self, Self)> {
        let mut train = Vec::with_capacity(self.frames.len());
        let mut valid = Vec::with_capacity(self.frames.len());

        for frame in &self.frames {
            let dates = date_buffer(frame).into_diagnostic()?;
            // Dates ascend, so this is the count of rows before the cutoff.
            let split = dates.partition_point(|&day| day < cutoff);
            let left = frame.head(Some(split));
            let right = frame.tail(Some(frame.height() - split));

            if left.height() >= steps {
                train.push(left);
            }
            if right.height() >= steps {
                valid.push(right);
            }
        }

        ensure!(
            !train.is_empty() && !valid.is_empty(),
            "train/valid split left one side empty; check cutoff and steps"
        );

        Ok((Self { frames: train }, Self { frames: valid }))
    }
}

/// A window coordinate: a ticker by its frame index and the start row of a
/// `steps`-length slice. The shared key into the resident feature tensors, produced by
/// the training dataset and the signal builders and consumed by [`gather_windows`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StockItem {
    pub ticker: usize,
    pub start: usize,
}

/// Assemble a `[windows.len(), steps, NUM_FEATURES]` tensor by copying each window's
/// contiguous rows into one flat host buffer, then a single upload. No per-window tensor
/// op, so the kernel does not grow with the batch and the JIT backend never compiles a
/// many-operand `stack`. The one windowing path, shared by training batches, backtest
/// chunks, and the trader's tails. `features` comes from [`TickerFrames::feature_series`],
/// indexed ticker-locally by [`StockItem::ticker`].
///
/// # Panics
/// If a series is not contiguous `f32`, which [`TickerFrames::feature_series`] guarantees.
#[must_use]
#[tracing::instrument(skip_all, fields(n = windows.len()))]
pub fn gather_windows<B: Backend>(
    features: &[Series],
    windows: &[StockItem],
    steps: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let count = windows.len();
    let mut flat = Vec::with_capacity(count * steps * NUM_FEATURES);
    for window in windows {
        let rows = features[window.ticker]
            .f32()
            .expect("f32 feature series")
            .cont_slice()
            .expect("contiguous feature series");
        let base = window.start * NUM_FEATURES;
        flat.extend_from_slice(&rows[base..base + steps * NUM_FEATURES]);
    }
    Tensor::from_data(TensorData::new(flat, [count, steps, NUM_FEATURES]), device)
}

/// One scored window: its `(ticker_index, start)` into the resident tensors, plus
/// the ticker and last-bar date that key the signal. Produced by [`TickerFrames::windows_since`]
/// for the backtest and [`TickerFrames::latest_windows`] for the trader.
pub struct Window {
    pub ticker_index: usize,
    pub start: usize,
    pub ticker: String,
    pub date: NaiveDate,
}

impl Window {
    /// The lightweight `(ticker_index, start)` coordinate for [`gather_windows`],
    /// dropping the ticker name and date that key the signal.
    #[must_use]
    pub fn item(&self) -> StockItem {
        StockItem {
            ticker: self.ticker_index,
            start: self.start,
        }
    }
}

/// One ticker's raw daily prices for the backtest, sharing `dates`' row order.
pub struct TickerQuotes {
    pub ticker: String,
    pub dates: Vec<NaiveDate>,
    pub open: Vec<f32>,
    pub high: Vec<f32>,
    pub low: Vec<f32>,
    pub close: Vec<f32>,
}

/// A frame's ticker, read from the first row of its `TICKER` column.
fn ticker_name(frame: &DataFrame) -> PolarsResult<String> {
    Ok(frame
        .column(&TICKER)?
        .str()?
        .get(0)
        .expect("partition group is non-empty")
        .to_owned())
}

/// A frame's first and last `DATE`, in one pass without materializing the column.
/// `None` when the frame has no dated row. Rows ascend, so `.1` is the latest bar.
fn date_endpoints(frame: &DataFrame) -> PolarsResult<Option<(NaiveDate, NaiveDate)>> {
    let mut dates = frame.column(&DATE)?.date()?.as_date_iter().flatten();
    let Some(first) = dates.next() else {
        return Ok(None);
    };
    Ok(Some((first, dates.last().unwrap_or(first))))
}

/// A frame's `DATE` column as `NaiveDate`s, in row order.
fn date_buffer(frame: &DataFrame) -> PolarsResult<Vec<NaiveDate>> {
    Ok(frame
        .column(&DATE)?
        .date()?
        .as_date_iter()
        .flatten()
        .collect())
}

/// A frame's price column as `f32`, in row order.
fn price_buffer(frame: &DataFrame, name: &PlSmallStr) -> PolarsResult<Vec<f32>> {
    Ok(frame.column(name)?.f32()?.into_no_null_iter().collect())
}

/// Pull one ticker's quotes out of its frame.
fn quotes_of(frame: &DataFrame) -> PolarsResult<TickerQuotes> {
    Ok(TickerQuotes {
        ticker: ticker_name(frame)?,
        dates: date_buffer(frame)?,
        open: price_buffer(frame, &OPEN)?,
        high: price_buffer(frame, &HIGH)?,
        low: price_buffer(frame, &LOW)?,
        close: price_buffer(frame, &CLOSE)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::{Flex, FlexDevice};

    /// `n_tickers` tickers of `rows` rows each. Row `i` fills every feature slot and the
    /// close with `ticker * 1000 + i`, so a window's value identifies its ticker and
    /// row; labels cycle 0/1/2; the price range is one unit.
    fn synthetic(n_tickers: i16, rows: i16) -> TickerFrames {
        use crate::features::FEATURE_NAMES;

        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let height = usize::try_from(rows).unwrap();

        let frames = (0..n_tickers)
            .map(|ticker| {
                let base = f32::from(ticker) * 1000.0;
                let value: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();

                let mut columns = vec![
                    Column::new(TICKER, vec![format!("t{ticker}"); height]),
                    Column::new(
                        DATE,
                        (0..i64::from(rows))
                            .map(|i| epoch + chrono::Duration::days(i))
                            .collect::<Vec<_>>(),
                    ),
                    Column::new(OPEN, value.clone()),
                    Column::new(HIGH, value.iter().map(|v| v + 1.0).collect::<Vec<_>>()),
                    Column::new(LOW, value.iter().map(|v| v - 1.0).collect::<Vec<_>>()),
                    Column::new(CLOSE, value.clone()),
                    Column::new(
                        LABEL,
                        (0..rows)
                            .map(|i| u8::try_from(i % 3).unwrap())
                            .collect::<Vec<_>>(),
                    ),
                ];
                for feature in FEATURE_NAMES {
                    columns.push(Column::new(feature, value.clone()));
                }

                DataFrame::new(height, columns)
                    .unwrap()
                    .lazy()
                    .with_column(feature_array().alias(FEATURE))
                    .select([
                        col(TICKER),
                        col(DATE),
                        col(FEATURE),
                        col(OPEN),
                        col(HIGH),
                        col(LOW),
                        col(CLOSE),
                        col(LABEL),
                    ])
                    .collect()
                    .unwrap()
            })
            .collect();

        TickerFrames { frames }
    }

    #[test]
    fn enumerate_windows_skips_short_tickers() {
        let store = synthetic(2, 6);
        let windows = store.enumerate_windows(4);
        // Each 6-row ticker yields 3 windows; none from a ticker shorter than 4.
        assert_eq!(windows.len(), 6);
        assert!(windows.iter().all(|item| item.ticker < 2));
    }

    #[test]
    fn gather_windows_slices_contiguous_rows() {
        let store = synthetic(2, 10);
        let features = store.feature_series().unwrap();

        // Ticker 0 from row 0, ticker 1 from row 1.
        let windows = [
            StockItem {
                ticker: 0,
                start: 0,
            },
            StockItem {
                ticker: 1,
                start: 1,
            },
        ];
        let tensor = gather_windows::<Flex>(&features, &windows, 4, &FlexDevice);

        assert_eq!(tensor.dims(), [2, 4, NUM_FEATURES]);
        let values = tensor.into_data().to_vec::<f32>().unwrap();
        let stride = NUM_FEATURES;
        for step in 0..4u8 {
            let row = usize::from(step);
            // First feature channel of each step is the row value.
            assert!((values[row * stride] - f32::from(step)).abs() < 1e-6);
            assert!((values[(4 + row) * stride] - (1001.0 + f32::from(step))).abs() < 1e-6);
        }
    }

    #[test]
    fn latest_windows_take_each_ticker_tail() {
        let store = synthetic(3, 8);
        let windows = store.latest_windows(4).unwrap();

        assert_eq!(windows.len(), 3);
        // An 8-row ticker's last 4-step window starts at row 4 and ends on row 7.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        assert_eq!(windows[0].ticker_index, 0);
        assert_eq!(windows[0].start, 4);
        assert_eq!(windows[0].ticker, "t0");
        assert_eq!(windows[0].date, epoch + chrono::Duration::days(7));
    }

    #[test]
    fn train_valid_split_partitions_rows() {
        let store = synthetic(3, 20);
        let cutoff = NaiveDate::from_ymd_opt(1970, 1, 11).unwrap();

        let (train, valid) = store.train_valid_split(cutoff, 4).unwrap();

        assert_eq!(train.frames.len(), 3);
        assert_eq!(valid.frames.len(), 3);
        assert_eq!(train.frames[0].height(), 10);
        assert_eq!(valid.frames[0].height(), 10);
    }

    #[test]
    fn date_bounds_span_first_to_last_row() {
        let store = synthetic(3, 20);
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        assert_eq!(
            store.date_bounds().unwrap(),
            Some((epoch, epoch + chrono::Duration::days(19)))
        );
    }
}
