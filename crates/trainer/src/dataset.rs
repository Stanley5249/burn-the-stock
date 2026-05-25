use arrow_array::{Float32Array, RecordBatch, StringArray};
use burn::data::dataset::Dataset;
use miette::{IntoDiagnostic, Result, miette};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::Path;

/// Number of trading days in each input window.
pub const WINDOW_SIZE: usize = 20;

/// OHLCV features per day.
pub const FEATURE_COUNT: usize = 5;

/// One sliding window of OHLCV data with a label.
#[derive(Clone, Debug)]
pub struct StockItem {
    /// Flattened OHLCV values, shape [`WINDOW_SIZE`] * [`FEATURE_COUNT`].
    pub features: Vec<f32>,
    /// 0 = sell, 1 = hold, 2 = buy.
    pub label: usize,
}

/// Dataset of sliding OHLCV windows built from a parquet file.
pub struct StockDataset {
    items: Vec<StockItem>,
}

impl StockDataset {
    /// Load the parquet file at `path` and build sliding windows.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O or schema mismatch.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).into_diagnostic()?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).into_diagnostic()?;
        let reader = builder.build().into_diagnostic()?;

        let mut batches: Vec<RecordBatch> = Vec::new();
        for batch in reader {
            batches.push(batch.into_diagnostic()?);
        }

        let items = build_items(&batches)?;
        Ok(Self { items })
    }
}

impl Dataset<StockItem> for StockDataset {
    fn get(&self, index: usize) -> Option<StockItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

// --- Internal helpers ---

/// Extract per-symbol row groups from `batches` and slice into windows.
fn build_items(batches: &[RecordBatch]) -> Result<Vec<StockItem>> {
    // Collect all rows sorted by (market, code, date) — the parquet is already
    // sorted this way by the Python aggregator, so we just iterate in order and
    // emit windows whenever the symbol changes.
    let mut items = Vec::new();

    for batch in batches {
        let codes = batch
            .column_by_name("code")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| miette!("missing or wrong-type 'code' column"))?;

        let open = float_col(batch, "open")?;
        let high = float_col(batch, "high")?;
        let low = float_col(batch, "low")?;
        let close = float_col(batch, "close")?;
        let volume = float_col(batch, "volume")?;

        // Group consecutive rows by symbol and emit windows whenever the symbol changes.
        let mut group_start = 0usize;
        let row_count = batch.num_rows();

        for row in 1..=row_count {
            let is_boundary = row == row_count || codes.value(row) != codes.value(group_start);

            if is_boundary {
                let symbol_rows = group_start..row;
                emit_windows(symbol_rows, open, high, low, close, volume, &mut items);
                group_start = row;
            }
        }
    }

    Ok(items)
}

fn float_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| miette!("missing or wrong-type '{name}' column"))
}

/// Slide a window of [`WINDOW_SIZE`] over the rows in `range` and label each window.
fn emit_windows(
    range: std::ops::Range<usize>,
    open: &Float32Array,
    high: &Float32Array,
    low: &Float32Array,
    close: &Float32Array,
    volume: &Float32Array,
    out: &mut Vec<StockItem>,
) {
    let rows: Vec<usize> = range.collect();
    if rows.len() < WINDOW_SIZE + 1 {
        return;
    }

    for window_start in 0..(rows.len() - WINDOW_SIZE) {
        let window = &rows[window_start..window_start + WINDOW_SIZE];
        let next_row = rows[window_start + WINDOW_SIZE];

        let mut features = Vec::with_capacity(WINDOW_SIZE * FEATURE_COUNT);
        for &row in window {
            features.push(open.value(row));
            features.push(high.value(row));
            features.push(low.value(row));
            features.push(close.value(row));
            features.push(volume.value(row));
        }

        let label = label_from_return(close.value(*window.last().unwrap()), close.value(next_row));
        out.push(StockItem { features, label });
    }
}

/// Classify next-day return into sell / hold / buy.
///
/// Thresholds are placeholders — tune before training.
fn label_from_return(current_close: f32, next_close: f32) -> usize {
    let return_pct = (next_close - current_close) / current_close;
    if return_pct < -0.01 {
        0 // sell
    } else if return_pct > 0.01 {
        2 // buy
    } else {
        1 // hold
    }
}
