//! The trainer's labeling layer: a forward maximum-favorable-excursion target, then a
//! per-date cross-sectional z-score so the model learns to rank stocks rather than
//! predict an absolute move. Labels are computed on the combined universe frame before
//! it is partitioned by ticker, so both steps are native polars `over` expressions.

use std::path::Path;

use miette::{IntoDiagnostic, Result};
use polars::prelude::*;
use stock_model::data::{LABEL, TickerFrames};
use stock_model::features::{DATE, HIGH, LOW, TICKER};

/// Intermediate raw MFE column, z-scored into [`LABEL`] then dropped before partition.
const RAW_MFE: PlSmallStr = PlSmallStr::from_static("raw_mfe");

/// Scan the OHLCV parquet, attach the forward-MFE rank label per row, and partition into
/// the per-ticker store. The label is the per-date z-score of
/// `max(high[t+1..=t+horizon]) / low[t+1] - 1`, so the score ranks the universe. The
/// trailing `horizon` rows per ticker have no forward outcome and are dropped, so the
/// store is `horizon` rows per ticker shorter than the source.
///
/// # Errors
/// If the parquet cannot be scanned or the label frame cannot be built.
pub fn load_labeled(path: &Path, horizon: usize) -> Result<TickerFrames> {
    let long = TickerFrames::standardized_long(path)?
        .with_columns([forward_mfe_expr(horizon)])
        .with_columns([zscore_label_expr()])
        // The only null labels are each ticker's trailing `horizon` rows.
        .filter(col(LABEL).is_not_null())
        .collect()
        .into_diagnostic()?
        .drop(&RAW_MFE)
        .into_diagnostic()?;

    TickerFrames::partition_long(&long).into_diagnostic()
}

/// Forward maximum favorable excursion per ticker: for entry `low[t+1]`, the best peak
/// `max(high[t+1..=t+horizon]) / low[t+1] - 1`, matching `sim_stock`'s buy-low/sell-high
/// fills. A forward window is a reversed trailing window, and `min_periods = horizon`
/// leaves the last `horizon` rows null so [`load_labeled`] drops them.
fn forward_mfe_expr(horizon: usize) -> Expr {
    assert!(horizon >= 1, "horizon must be at least one bar");

    let entry = col(LOW).shift(lit(-1));
    let peak = col(HIGH)
        .shift(lit(-1))
        .reverse()
        .rolling_max(RollingOptionsFixedWindow {
            window_size: horizon,
            min_periods: horizon,
            ..Default::default()
        })
        .reverse();

    (peak / entry - lit(1.0))
        .over([col(TICKER)])
        .expect("partition_by is non-empty")
        .alias(RAW_MFE)
}

/// Per-date cross-sectional z-score of [`RAW_MFE`], dropping the market-wide move so only
/// relative rank remains. Mirrors `stock_model`'s feature z-score.
fn zscore_label_expr() -> Expr {
    let mean = col(RAW_MFE)
        .mean()
        .over([col(DATE)])
        .expect("partition_by is non-empty");
    let std = col(RAW_MFE)
        .std(1)
        .over([col(DATE)])
        .expect("partition_by is non-empty");

    ((col(RAW_MFE) - mean) / (std + lit(1e-8))).alias(LABEL)
}

/// `n_tickers` tickers of `rows` rows each, for the dataset and batcher tests. Row `i`
/// fills every feature slot with `ticker * 1000 + i`, so a window's value identifies its
/// ticker and row; the label is `i` as `f32`.
#[cfg(test)]
pub(crate) fn synthetic(n_tickers: i16, rows: i16) -> TickerFrames {
    use stock_model::features::{FEATURE, FEATURE_NAMES, feature_array};

    let height = usize::try_from(rows).unwrap();

    let frames = (0..n_tickers)
        .map(|ticker| {
            let base = f32::from(ticker) * 1000.0;
            let value: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();

            let mut columns = vec![
                Column::new(TICKER, vec![format!("t{ticker}"); height]),
                Column::new(LABEL, (0..rows).map(f32::from).collect::<Vec<_>>()),
            ];
            for feature in FEATURE_NAMES {
                columns.push(Column::new(feature, value.clone()));
            }

            DataFrame::new(height, columns)
                .unwrap()
                .lazy()
                .with_column(feature_array().alias(FEATURE))
                .select([col(TICKER), col(FEATURE), col(LABEL)])
                .collect()
                .unwrap()
        })
        .collect();

    TickerFrames { frames }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn forward_mfe_is_peak_over_next_low() {
        // horizon 2, one ticker. t=0: entry=low[1]=10, peak=max(high[1..=2])=13 -> 0.3.
        // t=1: entry=low[2]=11, peak=max(high[2..=3])=13 -> 13/11-1. Last 2 rows null.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let dates: Vec<NaiveDate> = (0..4).map(|i| epoch + chrono::Duration::days(i)).collect();
        let frame = df!(
            "ticker" => ["t0"; 4],
            "date" => dates,
            "high" => [10.0f32, 11.0, 13.0, 12.0],
            "low" => [10.0f32, 10.0, 11.0, 11.0],
        )
        .unwrap();

        let out = frame
            .lazy()
            .with_columns([forward_mfe_expr(2)])
            .collect()
            .unwrap();
        let raw = out.column(&RAW_MFE).unwrap().f32().unwrap();

        assert!((raw.get(0).unwrap() - 0.3).abs() < 1e-6);
        assert!((raw.get(1).unwrap() - (13.0 / 11.0 - 1.0)).abs() < 1e-6);
        assert!(raw.get(2).is_none());
        assert!(raw.get(3).is_none());
    }

    #[test]
    fn zscore_label_centers_each_date() {
        // Two tickers, two dates. Each date's z-scored labels sum to ~0.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let next = epoch + chrono::Duration::days(1);
        let frame = df!(
            "date" => [epoch, next, epoch, next],
            "raw_mfe" => [1.0f32, 2.0, 3.0, 6.0],
        )
        .unwrap();

        let out = frame
            .lazy()
            .with_columns([zscore_label_expr()])
            .collect()
            .unwrap();
        let label = out.column(&LABEL).unwrap().f32().unwrap();

        // epoch rows are 0 and 2; next rows are 1 and 3.
        assert!((label.get(0).unwrap() + label.get(2).unwrap()).abs() < 1e-5);
        assert!((label.get(1).unwrap() + label.get(3).unwrap()).abs() < 1e-5);
        // The larger value of each date standardizes above the smaller.
        assert!(label.get(2).unwrap() > label.get(0).unwrap());
    }
}
