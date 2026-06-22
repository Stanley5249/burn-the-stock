use std::fs::File;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use miette::{IntoDiagnostic, Result};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{Layer, Registry, reload};

// Boxed so the stderr and file layers, with different writer types, share one type
// the reload handle can swap.
type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync>;

static RELOAD_HANDLE: OnceLock<reload::Handle<BoxedLayer, Registry>> = OnceLock::new();

/// Install the process-wide subscriber, logging to stderr until [`redirect_to_file`].
pub fn install() {
    let (layer, handle) = reload::Layer::new(boxed_layer(std::io::stderr, true));

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,stock_client=info,trainer=info"));

    tracing_subscriber::registry()
        .with(layer)
        .with(filter)
        .init();

    let _ = RELOAD_HANDLE.set(handle);
}

/// Swap the writer to `{artifact_dir}/experiment.log` for the rest of the run, so the
/// data-loading spans share a log with the epoch metrics. The caller disables burn's
/// own logger so it does not fight for the file.
///
/// # Errors
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

fn boxed_layer<W>(writer: W, ansi: bool) -> BoxedLayer
where
    W: for<'writer> fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    fmt::layer()
        .with_writer(writer)
        .with_ansi(ansi)
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .boxed()
}
