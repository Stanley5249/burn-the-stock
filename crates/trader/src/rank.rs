//! Model inference: score every ticker and rank them strongest first. The same path the
//! backtest uses, run over the latest window per ticker.

use std::path::Path;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use chrono::Duration;
use miette::{Context, IntoDiagnostic, Result};
use polars::prelude::*;
use stock_model::data::TickerFrames;
use stock_model::features::DATE;
use stock_model::inference::{InferenceConfig, score};

use crate::cli::Args;

type Backend = Wgpu;

/// Score every ticker on its latest window and return `(ticker, score)` sorted strongest
/// first.
///
/// # Errors
/// If the artifact config/model cannot be loaded or the parquet cannot be scanned.
#[tracing::instrument(skip_all)]
pub fn rank(args: &Args, device: &WgpuDevice) -> Result<Vec<(String, f32)>> {
    let config = InferenceConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;
    let model = config
        .model
        .init::<Backend>(device)
        .load_file(
            args.artifact_dir.join("model"),
            &CompactRecorder::new(),
            device,
        )
        .into_diagnostic()
        .wrap_err("fail to init model from artifact")?;

    // Trading days are sparser than calendar days, so over-reach the lookback and let
    // `latest_windows` trim each ticker to exactly `config.steps`.
    let lookback = i64::try_from(config.steps * 2 + 10)
        .into_diagnostic()
        .wrap_err("steps too large for the lookback window")?;
    let frame = recent_frame(&args.data, lookback).into_diagnostic()?;
    let store = TickerFrames::from_lazy(frame).into_diagnostic()?;

    let windows = store.latest_windows(config.steps).into_diagnostic()?;
    let features = store.feature_series().into_diagnostic()?;
    let predictions = score::<Backend>(&model, &features, &windows, config.steps, device);

    let mut ranked: Vec<(String, f32)> = windows
        .into_iter()
        .zip(predictions)
        .map(|(window, prediction)| (window.ticker, prediction.score))
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(ranked)
}

/// Keep the strongest names scoring above `threshold`, capped at `max`. Daily rotation sells
/// the whole book, so held names stay eligible and may be rebought.
#[must_use]
pub fn select_candidates(
    ranked: &[(String, f32)],
    threshold: f32,
    max: usize,
) -> Vec<(String, f32)> {
    ranked
        .iter()
        .filter(|(_, score)| *score > threshold)
        .take(max)
        .cloned()
        .collect()
}

/// Scan only the recent tail of the OHLCV parquet, keeping the last `lookback` calendar
/// days. The per-date z-score is unaffected since each retained date still holds the full
/// universe.
fn recent_frame(path: &Path, lookback: i64) -> PolarsResult<LazyFrame> {
    let frame =
        LazyFrame::scan_parquet(PlRefPath::try_from_path(path)?, ScanArgsParquet::default())?
            .with_column(col(DATE).cast(DataType::Date));

    let max_date = frame
        .clone()
        .select([col(DATE).max()])
        .collect()?
        .column(&DATE)?
        .date()?
        .as_date_iter()
        .flatten()
        .next()
        .expect("parquet has at least one dated row");

    let cutoff = max_date - Duration::days(lookback);

    Ok(frame.filter(col(DATE).gt_eq(lit(cutoff))))
}
