//! Feature transform shared by training and inference: per-bar stationary
//! signals, then the same signals as a per-date cross-sectional z-score. The
//! absolute half keeps the market move; the z-scored half keeps only rank. One
//! implementation keeps training and live inputs identical.

use std::f64::consts::E;

use polars::prelude::*;

/// Ticker code after the rename, and the packed feature array column name.
pub const TICKER: PlSmallStr = PlSmallStr::from_static("ticker");
pub const FEATURE: PlSmallStr = PlSmallStr::from_static("feature");

/// Raw source columns the transform reads.
pub const CODE: PlSmallStr = PlSmallStr::from_static("code");
pub const DATE: PlSmallStr = PlSmallStr::from_static("date");
pub const OPEN: PlSmallStr = PlSmallStr::from_static("open");
pub const HIGH: PlSmallStr = PlSmallStr::from_static("high");
pub const LOW: PlSmallStr = PlSmallStr::from_static("low");
pub const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
pub const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");

const OPEN_RETURN: PlSmallStr = PlSmallStr::from_static("open_return");
const HIGH_RETURN: PlSmallStr = PlSmallStr::from_static("high_return");
const LOW_RETURN: PlSmallStr = PlSmallStr::from_static("low_return");
const CLOSE_RETURN: PlSmallStr = PlSmallStr::from_static("close_return");
const VOLUME_RETURN: PlSmallStr = PlSmallStr::from_static("volume_return");
const RANGE: PlSmallStr = PlSmallStr::from_static("range");

const OPEN_RETURN_CS: PlSmallStr = PlSmallStr::from_static("open_return_cs");
const HIGH_RETURN_CS: PlSmallStr = PlSmallStr::from_static("high_return_cs");
const LOW_RETURN_CS: PlSmallStr = PlSmallStr::from_static("low_return_cs");
const CLOSE_RETURN_CS: PlSmallStr = PlSmallStr::from_static("close_return_cs");
const VOLUME_RETURN_CS: PlSmallStr = PlSmallStr::from_static("volume_return_cs");
const RANGE_CS: PlSmallStr = PlSmallStr::from_static("range_cs");

/// Standardized feature width of the technical input, matching the feature
/// column.
pub const NUM_FEATURES: usize = 12;

/// Absolute signal count; the z-scored half follows at `[i + NUM_BASE]`.
const NUM_BASE: usize = NUM_FEATURES / 2;

/// Feature column order: `NUM_FEATURES` long, so the polars list and the tensor width
/// are one number. Changing one without the other is a compile error.
pub const FEATURE_NAMES: [PlSmallStr; NUM_FEATURES] = [
    // absolute
    OPEN_RETURN,
    HIGH_RETURN,
    LOW_RETURN,
    CLOSE_RETURN,
    VOLUME_RETURN,
    RANGE,
    // z-scored
    OPEN_RETURN_CS,
    HIGH_RETURN_CS,
    LOW_RETURN_CS,
    CLOSE_RETURN_CS,
    VOLUME_RETURN_CS,
    RANGE_CS,
];

/// The absolute signals: log of each channel's ratio. Prices anchor on the
/// previous close; `range` is the same-day high/low ratio, a direction-free
/// volatility measure.
///
/// The per-ticker `shift` runs `over` the ticker, so the caller must sort by
/// `[ticker, date]` first.
fn stationary_features() -> [Expr; 6] {
    let prev_close = col(CLOSE)
        .shift(lit(1))
        .over([col(TICKER)])
        .expect("partition_by is non-empty");

    let prev_volume = col(VOLUME)
        .shift(lit(1))
        .over([col(TICKER)])
        .expect("partition_by is non-empty");

    // Volume is a count, so a no-trade day (0) makes the bare log -inf. Add one
    // share to both sides to keep it finite.
    let volume_return = ((col(VOLUME) + lit(1.0)) / (prev_volume + lit(1.0))).log(lit(E));

    let price_return = |price: PlSmallStr, alias: PlSmallStr| {
        (col(price) / prev_close.clone()).log(lit(E)).alias(alias)
    };

    [
        price_return(OPEN, OPEN_RETURN),
        price_return(HIGH, HIGH_RETURN),
        price_return(LOW, LOW_RETURN),
        price_return(CLOSE, CLOSE_RETURN),
        volume_return.alias(VOLUME_RETURN),
        (col(HIGH) / col(LOW)).log(lit(E)).alias(RANGE),
    ]
}

/// Each absolute signal's per-date cross-sectional z-score as a new column, dropping
/// the market-wide move so only rank remains. Run after [`stationary_features`].
fn cross_sectional_zscore() -> [Expr; NUM_BASE] {
    std::array::from_fn(|index| {
        let base = FEATURE_NAMES[index].clone();
        let mean_over_date = col(base.clone())
            .mean()
            .over([col(DATE)])
            .expect("partition_by is non-empty");

        let std_over_date = col(base.clone())
            .std(1)
            .over([col(DATE)])
            .expect("partition_by is non-empty");

        let centered = col(base) - mean_over_date;
        let spread = std_over_date + lit(1e-8);

        (centered / spread).alias(FEATURE_NAMES[index + NUM_BASE].clone())
    })
}

/// Standardize a raw OHLCV frame into the model's feature columns, keeping raw
/// high/low/close for the trainer's forward-MFE label. The per-date z-score needs the
/// full ticker universe, not a single stock.
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

/// Pack the [`FEATURE_NAMES`] columns into one row-major fixed-size `Array` of
/// `NUM_FEATURES` floats.
///
/// # Panics
/// If the feature column list is empty, which [`FEATURE_NAMES`] never
/// is.
pub fn feature_array() -> Expr {
    concat_arr(
        FEATURE_NAMES
            .map(|name| col(name).cast(DataType::Float32))
            .to_vec(),
    )
    .expect("feature columns are non-empty and uniformly f32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn cross_sectional_zscore_centers_each_date() {
        // Two dates, three stocks, different scales. Only open_return is checked;
        // the rest just need to exist for the cs map over every base name.
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
            "range" => zeros,
        )
        .unwrap();

        let out = frame
            .lazy()
            .with_columns(cross_sectional_zscore())
            .collect()
            .unwrap();

        let standardized = out.column(&OPEN_RETURN_CS).unwrap().f32().unwrap();

        // Each date's values must average to ~0 after standardizing.
        let first: f32 = (0..3).map(|i| standardized.get(i).unwrap()).sum();
        let second: f32 = (3..6).map(|i| standardized.get(i).unwrap()).sum();
        assert!(first.abs() < 1e-5, "first date mean {first} not ~0");
        assert!(second.abs() < 1e-5, "second date mean {second} not ~0");

        // Ordering within a date is preserved.
        assert!(standardized.get(0).unwrap() < standardized.get(2).unwrap());
    }
}
