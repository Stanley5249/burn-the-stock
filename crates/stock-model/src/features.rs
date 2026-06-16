//! Feature transform shared by training and inference: per-bar log-returns, then a
//! per-date cross-sectional z-score over the whole universe. One implementation
//! keeps the training distribution and the live inputs identical.

use chrono::NaiveDate;
use polars::prelude::*;

/// Raw source columns the transform reads.
pub const CODE: PlSmallStr = PlSmallStr::from_static("code");
pub const DATE: PlSmallStr = PlSmallStr::from_static("date");
pub const OPEN: PlSmallStr = PlSmallStr::from_static("open");
pub const HIGH: PlSmallStr = PlSmallStr::from_static("high");
pub const LOW: PlSmallStr = PlSmallStr::from_static("low");
pub const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
pub const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

/// Ticker code after the rename, and the packed feature array column name.
pub const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
pub const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");

const OPEN_RETURN: PlSmallStr = PlSmallStr::from_static("open_return");
const HIGH_RETURN: PlSmallStr = PlSmallStr::from_static("high_return");
const LOW_RETURN: PlSmallStr = PlSmallStr::from_static("low_return");
const CLOSE_RETURN: PlSmallStr = PlSmallStr::from_static("close_return");
const VOLUME_RETURN: PlSmallStr = PlSmallStr::from_static("volume_return");

/// The five standardized features in column order; width is fixed at five.
pub const FEATURE_NAMES: [PlSmallStr; 5] = [
    OPEN_RETURN,
    HIGH_RETURN,
    LOW_RETURN,
    CLOSE_RETURN,
    VOLUME_RETURN,
];

const fn col(name: PlSmallStr) -> Expr {
    Expr::Column(name)
}

/// Build the five stationary feature expressions: the natural log of each channel's
/// ratio to the prior bar. The four prices share one anchor, the previous close. The
/// per-ticker `shift` runs `over` the ticker, so the caller must sort by
/// `[ticker, date]` first.
fn stationary_features() -> [Expr; 5] {
    let prev_close = col(CLOSE)
        .shift(lit(1))
        .over([col(TICKER)])
        .expect("partition_by is non-empty");
    let prev_volume = col(VOLUME)
        .shift(lit(1))
        .over([col(TICKER)])
        .expect("partition_by is non-empty");

    let natural_log = || lit(std::f64::consts::E);

    // Volume is a count, so a no-trade day (0) makes the bare log -inf. Add one
    // share to both sides to keep it finite.
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

/// Replace each feature with its per-date cross-sectional z-score, removing the
/// market-wide common factor so only relative performance remains. Run after
/// [`stationary_features`] and before `drop_nulls`.
fn cross_sectional_zscore() -> [Expr; 5] {
    FEATURE_NAMES.map(|name| {
        let mean_over_date = col(name.clone())
            .mean()
            .over([col(DATE)])
            .expect("partition_by is non-empty");
        let std_over_date = col(name.clone())
            .std(1)
            .over([col(DATE)])
            .expect("partition_by is non-empty");
        let centered = col(name.clone()) - mean_over_date;
        let spread = std_over_date + lit(1e-8);
        (centered / spread).alias(name)
    })
}

/// Standardize a raw OHLCV frame into the model's feature columns, keeping raw
/// high/low/close for the barrier labels. The per-date z-score needs the full
/// ticker universe, not a single stock.
pub fn standardized_features(frame: LazyFrame) -> LazyFrame {
    frame
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
        // Sort so the per-ticker `shift` sees rows in date order.
        .sort([TICKER, DATE], SortMultipleOptions::new())
        .with_columns(stationary_features())
        .with_columns(cross_sectional_zscore())
        // The only nulls are each ticker's first row.
        .drop_nulls(None)
}

/// Pack the five feature columns into one row-major `Array<f32, 5>`.
///
/// # Panics
/// If the feature column list is empty, which [`FEATURE_NAMES`] never is.
pub fn feature_array() -> Expr {
    concat_arr(
        FEATURE_NAMES
            .map(|name| col(name).cast(DataType::Float32))
            .to_vec(),
    )
    .expect("feature columns are non-empty and uniformly f32")
}

/// One ticker's most recent `steps`-day feature window, the model input as of
/// [`InferenceWindow::date`].
pub struct InferenceWindow {
    pub ticker: String,
    /// The bar the prediction is made from.
    pub date: NaiveDate,
    /// Row-major standardized features, length `steps * 5`.
    pub features: Vec<f32>,
}

/// Each ticker's most recent `steps`-day window from a frame already run through
/// [`standardized_features`]. Tickers with fewer than `steps` rows are skipped.
///
/// # Errors
/// If the frame cannot be collected or its columns are malformed.
///
/// # Panics
/// If a partitioned group is empty, which `partition_by_stable` never yields.
pub fn latest_windows(features: LazyFrame, steps: usize) -> PolarsResult<Vec<InferenceWindow>> {
    let long = features
        .select([col(TICKER), col(DATE), feature_array().alias(FEATURE)])
        .collect()?;

    let groups = long.partition_by_stable([TICKER], true)?;
    let mut windows = Vec::with_capacity(groups.len());

    for group in groups {
        // Too short to fill the window.
        if group.height() < steps {
            continue;
        }
        let tail = group.tail(Some(steps));

        let ticker = tail.column(&TICKER)?.str()?.get(0).unwrap().to_owned();

        let features: Vec<f32> = tail
            .column(&FEATURE)?
            .array()?
            .get_inner()
            .f32()?
            .into_no_null_iter()
            .collect();

        // Last row is the bar to predict from.
        let date = tail
            .column(&DATE)?
            .date()?
            .as_date_iter()
            .flatten()
            .last()
            .expect("tail holds steps rows");

        windows.push(InferenceWindow {
            ticker,
            date,
            features,
        });
    }

    Ok(windows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_sectional_zscore_centers_each_date() {
        // Two dates, three stocks, different scales. Only open_return is checked;
        // the rest just need to exist.
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

        // Each date's values must average to ~0 after standardizing.
        let first: f32 = (0..3).map(|i| standardized.get(i).unwrap()).sum();
        let second: f32 = (3..6).map(|i| standardized.get(i).unwrap()).sum();
        assert!(first.abs() < 1e-5, "first date mean {first} not ~0");
        assert!(second.abs() < 1e-5, "second date mean {second} not ~0");

        // Ordering within a date is preserved.
        assert!(standardized.get(0).unwrap() < standardized.get(2).unwrap());
    }
}
