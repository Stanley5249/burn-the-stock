use std::fs::File;
use std::sync::Mutex;

use miette::{IntoDiagnostic, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

use crate::training::{RunOptions, TrainingConfig};

/// Install the global tracing subscriber that writes this run's structured log to
/// `{artifact_dir}/experiment.log`.
///
/// Burn's training builder writes the same file by default, so the caller disables
/// that with `with_application_logger(None)` and lets this subscriber own the file.
/// Owning it from the start means the data-loading spans, which run before training
/// begins, land in the same log as the per-epoch metrics. `FmtSpan::CLOSE` makes
/// every `#[instrument]` span emit its busy/idle time when it closes, so phase
/// timings need no manual clocks. The `wgpu` stack is silenced to keep the log
/// about the experiment rather than the GPU backend.
pub fn install_experiment_logger(artifact_dir: &str) -> Result<()> {
    let file = File::create(format!("{artifact_dir}/experiment.log")).into_diagnostic()?;

    // Keep our spans, the run-config record, and burn's early-stopping loss
    // trajectory at INFO, but drop the noise: burn's per-iteration `Iteration N`
    // chatter (884 of 911 lines in a short run), the checkpointer's "File exists"
    // warnings, the autotune-cache notes, and the wgpu backend's device dump.
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .parse_lossy("info,wgpu=off,naga=off");

    let layer = fmt::layer()
        .with_writer(Mutex::new(file))
        .with_ansi(false)
        .with_span_events(FmtSpan::CLOSE);

    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .init();

    Ok(())
}

/// Emit one structured `experiment` record of the flags and derived counts that
/// shaped this run, so reading `experiment.log` later ties the metrics back to the
/// configuration that produced them. `num_epochs` and `total_windows` are derived
/// after the load and split, so they are passed in rather than read off `config`.
pub fn log_run_config(
    config: &TrainingConfig,
    options: &RunOptions,
    n_industries: usize,
    total_windows: usize,
    num_epochs: usize,
) {
    tracing::info!(
        target: "experiment",
        steps = config.steps,
        batch_size = config.batch_size,
        epoch_size = config.epoch_size,
        passes = config.passes,
        num_epochs,
        total_windows,
        n_industries,
        d_hidden = config.model.d_hidden,
        dropout = config.model.dropout,
        learning_rate = config.learning_rate,
        label_threshold = config.label_threshold,
        fee = config.fee,
        reward_clip = config.reward_clip,
        seed = config.seed,
        valid_days = options.valid_days,
        valid_batches = ?options.valid_batches,
        max_tickers = ?options.max_tickers,
        patience = ?options.patience,
        "run config"
    );
}
