//! Model inference: score every ticker and rank them strongest first. The same path the
//! backtest uses, run over the latest window per ticker.

use burn::prelude::*;
use burn::record::CompactRecorder;
use miette::{Context, IntoDiagnostic, Result};
use stock_data::read::History;
use stock_model::data::TickerFrames;
use stock_model::inference::{InferenceConfig, score};

use crate::cli::Args;

/// Score every ticker on its latest window and return `(ticker, score)` sorted strongest
/// first.
///
/// # Errors
/// If the artifact config/model cannot be loaded or the parquet cannot be scanned.
#[tracing::instrument(skip_all)]
pub fn rank<B: Backend>(args: &Args, device: &B::Device) -> Result<Vec<(String, f32)>> {
    let config = InferenceConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;

    let model = config
        .model
        .init::<B>(device)
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
    let frame = History::scan(&args.data)?.recent(lookback)?.lazy();

    let store = TickerFrames::from_lazy(frame).into_diagnostic()?;

    let windows = store.latest_windows(config.steps).into_diagnostic()?;
    let features = store.feature_series().into_diagnostic()?;
    let predictions = score(
        &model,
        &features,
        &windows,
        config.steps,
        config.batch_size,
        device,
    );

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
