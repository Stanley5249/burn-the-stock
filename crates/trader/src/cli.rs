//! Command-line arguments for the live trader.

use std::path::PathBuf;

use clap::Parser;
use stock_client::sim_stock::DEFAULT_TIMEOUT_MS;

#[derive(Parser, Debug, Clone)]
#[command(about = "Predict, then place the day's weighted orders on sim_stock")]
pub struct Args {
    /// Directory holding a training run's `config.json` and `model`.
    #[arg(long, default_value = "artifacts/latest")]
    pub artifact_dir: PathBuf,

    /// OHLCV parquet to score; only its recent tail is read. Must be current through
    /// yesterday's close, so the pre-trade refresh updates it before running.
    #[arg(long, default_value = "data/yfinance/stock_history.parquet")]
    pub data: PathBuf,

    /// Minimum predicted score (per-date z-scored MFE) to buy.
    #[arg(long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Target number of positions to hold, which also caps how many ranked candidates get
    /// quoted.
    #[arg(long, default_value_t = 100)]
    pub max_holdings: usize,

    /// Fraction of settled cash held back for later days.
    #[arg(long, default_value_t = 0.0)]
    pub buffer: f64,

    /// Delay between Fugle quote requests, to respect the rate limit.
    #[arg(long, default_value_t = 1000)]
    pub quote_delay_ms: u64,

    /// Directory holding the per-year TWSE holiday caches.
    #[arg(long, default_value = "data/twse")]
    pub holiday_cache: PathBuf,

    /// Per-request timeout in milliseconds for `sim_stock`, which is flaky and can hang.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_MS)]
    pub timeout_ms: u64,

    /// Skip the pre-trade refresh and score the data as-is.
    #[arg(long)]
    pub no_download: bool,

    /// Trade even if the data is not current through the required session. Only for the rare
    /// long-holiday case where the latest bar is legitimately old.
    #[arg(long)]
    pub allow_stale: bool,

    /// Plan and print the orders without placing them.
    #[arg(long)]
    pub dry_run: bool,
}
