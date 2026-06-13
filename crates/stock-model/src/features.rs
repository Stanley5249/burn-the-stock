//! The feature transform shared by training and inference.
//!
//! Both turn raw OHLCV into the same standardized input: per-bar log-returns, then
//! a per-date cross-sectional z-score over the whole universe. Sharing one
//! implementation is what keeps the model's training distribution and its live
//! inputs identical, no matter that one source is a parquet file and the other a
//! live price feed.

use chrono::NaiveDate;
use polars::prelude::*;

/// Source schema the transform reads. A caller's frame, whether scanned from
/// parquet or built from a live feed, must carry these raw columns.
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

/// The five standardized features flattened per row, in column order. Width is
/// five, which the model's input layer and the batcher's tensor shapes assume.
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

/// Standardize a raw OHLCV frame into the model's feature columns.
///
/// Casts the source schema, drops non-positive closes, sorts by `[ticker, date]`,
/// derives the per-bar log-returns, then standardizes each one cross-sectionally
/// per date over the whole frame. The output keeps the raw `high`/`low`/`close`
/// alongside the rewritten feature columns, since the barrier labels need the
/// intraday range. Each ticker's first row is dropped, its previous bar being
/// absent, so the result has no nulls.
///
/// The per-date z-score is taken over every row on a date, so the frame must hold
/// the full ticker universe, not a single stock, for the cross-section to match.
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
        // Sort before deriving features so the per-ticker `shift` sees rows in date
        // order.
        .sort([TICKER, DATE], SortMultipleOptions::new())
        .with_columns(stationary_features())
        .with_columns(cross_sectional_zscore())
        // The only nulls are each ticker's first row, whose previous bar did not
        // exist, so drop them before packing the feature array.
        .drop_nulls(None)
}

/// Pack the five standardized feature columns into one `Array<f32, 5>`, the
/// row-major layout the batcher and the inference path both flatten.
///
/// # Panics
///
/// Panics if the feature column list is empty, which [`FEATURE_NAMES`] never is.
pub fn feature_array() -> Expr {
    concat_arr(
        FEATURE_NAMES
            .map(|name| col(name).cast(DataType::Float32))
            .to_vec(),
    )
    .expect("feature columns are non-empty and uniformly f32")
}

/// One ticker's most recent `steps`-day standardized feature window, the model
/// input for predicting that ticker's action as of [`InferenceWindow::date`].
pub struct InferenceWindow {
    /// Ticker code.
    pub ticker: String,
    /// Most recent trading day in the window, the bar the prediction is made from.
    pub date: NaiveDate,
    /// Row-major standardized features, length `steps * 5`.
    pub features: Vec<f32>,
}

/// Take each ticker's most recent `steps`-day feature window from a frame already
/// run through [`standardized_features`]. Tickers with fewer than `steps` rows are
/// skipped. This is the inference counterpart to the training window enumeration,
/// keeping the latest rows rather than dropping a label horizon.
///
/// # Errors
///
/// Returns an error if the frame cannot be collected or its columns are malformed.
///
/// # Panics
///
/// Panics if a partitioned group is empty, which `partition_by_stable` never yields.
pub fn latest_windows(features: LazyFrame, steps: usize) -> PolarsResult<Vec<InferenceWindow>> {
    let long = features
        .select([col(TICKER), col(DATE), feature_array().alias(FEATURE)])
        .collect()?;

    let groups = long.partition_by_stable([TICKER], true)?;
    let mut windows = Vec::with_capacity(groups.len());

    for group in groups {
        // A shorter history cannot fill the model's window, so skip it.
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

        // The last row is the most recent trading day, the bar to predict from.
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
}
