use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use miette::{IntoDiagnostic, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

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
pub fn install_experiment_logger(artifact_dir: &Path) -> Result<()> {
    let file = File::create(artifact_dir.join("experiment.log")).into_diagnostic()?;

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
