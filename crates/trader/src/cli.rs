//! Command-line arguments for the live trader.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(about = "Predict, then place the day's weighted orders on sim_stock")]
pub struct Args {
    /// Directory holding a training run's `config.json` and `model`.
    #[arg(long, default_value = "artifacts/latest")]
    pub artifact_dir: PathBuf,

    /// OHLCV parquet to score; only its recent tail is read. Must be current through
    /// yesterday's close, so refresh it with the downloader before running.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    pub data: PathBuf,

    /// Cash ledger path; tracks settled cash and unsettled sale proceeds across runs.
    #[arg(long, default_value = "data/live-state.json")]
    pub state: PathBuf,

    /// Minimum predicted score (per-date z-scored MFE) to buy.
    #[arg(long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Target number of positions to hold, which also caps how many ranked candidates get
    /// quoted.
    #[arg(long, default_value_t = 100)]
    pub max_holdings: usize,

    /// Fraction of settled cash held back for later days.
    #[arg(long, default_value_t = 0.1)]
    pub buffer: f64,

    /// Calendar days until sale proceeds become spendable. The platform settles at
    /// 15:30-16:00, after the 13:00 order cutoff, so today's sells fund the next run (0).
    #[arg(long, default_value_t = 0)]
    pub settle_lag: i64,

    /// Delay between Fugle quote requests, to respect the rate limit.
    #[arg(long, default_value_t = 1100)]
    pub quote_delay_ms: u64,

    /// Cap on in-flight order requests placed against the platform at once.
    #[arg(long, default_value_t = 8)]
    pub order_concurrency: usize,

    /// Plan and print the orders without placing them.
    #[arg(long)]
    pub dry_run: bool,
}
