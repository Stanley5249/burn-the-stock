//! Refresh the OHLCV parquet before trading by running the Python downloader, so the trader
//! never scores stale data.

use miette::{IntoDiagnostic, Result, WrapErr, miette};
use tokio::process::Command;

/// Run `uv run python -m burn_the_stock.downloader`, streaming its output, and fail the run
/// on a non-zero exit so trading never proceeds on stale data.
///
/// # Errors
/// If the process cannot be spawned or exits non-zero.
pub async fn run_downloader() -> Result<()> {
    tracing::info!("refreshing OHLCV via downloader");
    let status = Command::new("uv")
        .args(["run", "python", "-m", "burn_the_stock.downloader"])
        .status()
        .await
        .into_diagnostic()
        .wrap_err("spawn downloader (is `uv` on PATH?)")?;

    if !status.success() {
        return Err(miette!("downloader exited with {status}"));
    }
    Ok(())
}
