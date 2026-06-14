use std::fs::File;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use miette::{IntoDiagnostic, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{Layer, Registry, reload};

// Boxed so the stderr and file layers, which have different writer types, share one
// type the reload handle can swap between.
type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync>;

static RELOAD_HANDLE: OnceLock<reload::Handle<BoxedLayer, Registry>> = OnceLock::new();

/// Install the process-wide subscriber, logging to stderr until [`redirect_to_file`].
pub fn install() {
    let (layer, handle) = reload::Layer::new(boxed_layer(std::io::stderr, true));
    tracing_subscriber::registry().with(layer).init();
    let _ = RELOAD_HANDLE.set(handle);
}

/// Swap the writer to `{artifact_dir}/experiment.log` for the rest of the run. Burn
/// would otherwise own this file, so the caller disables its logger; owning it here
/// keeps the pre-training data-loading spans in the same log as the epoch metrics.
///
/// # Errors
///
/// If the log file cannot be created or the layer cannot be reloaded.
pub fn redirect_to_file(artifact_dir: &Path) -> Result<()> {
    let file = File::create(artifact_dir.join("experiment.log")).into_diagnostic()?;
    if let Some(handle) = RELOAD_HANDLE.get() {
        handle
            .reload(boxed_layer(Mutex::new(file), false))
            .into_diagnostic()?;
    }
    Ok(())
}

// `ansi` colors stderr but stays off for the file. `FmtSpan::CLOSE` emits each span's
// busy/idle time so phase timings need no manual clocks. `wgpu`/`naga` are silenced;
// INFO keeps our spans and burn's loss trajectory while dropping its per-iter chatter.
fn boxed_layer<W>(writer: W, ansi: bool) -> BoxedLayer
where
    W: for<'writer> fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .parse_lossy("info,wgpu=off,naga=off");

    fmt::layer()
        .with_writer(writer)
        .with_ansi(ansi)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(filter)
        .boxed()
}
