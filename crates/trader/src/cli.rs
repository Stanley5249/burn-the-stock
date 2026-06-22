//! Command-line arguments for the live trader.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
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

    /// Target number of positions to hold.
    #[arg(long, default_value_t = 100)]
    pub max_holdings: usize,

    /// How many top-ranked candidates to fetch quotes for.
    #[arg(long, default_value_t = 100)]
    pub shortlist: usize,

    /// Fraction of settled cash held back for later days.
    #[arg(long, default_value_t = 0.1)]
    pub buffer: f64,

    /// Calendar days until sale proceeds become spendable. The platform settles at
    /// 15:30-16:00, after the 13:00 order cutoff, so today's sells fund the next run (0).
    #[arg(long, default_value_t = 0)]
    pub settle_lag: i64,

    /// Use the laddered exits (time / model-sell / barriers) instead of the default, which
    /// sells the whole book every day to harvest the buy-low/sell-high spread.
    #[arg(long)]
    pub exit_ladder: bool,

    /// Trading days to hold before the time exit (laddered-exit mode only).
    #[arg(long, default_value_t = 20)]
    pub max_hold: usize,

    /// Take-profit exit as a fraction of price; 1.0 is effectively off.
    #[arg(long, default_value_t = 1.0)]
    pub take_profit: f64,

    /// Stop-loss exit as a fraction of price; 1.0 is effectively off.
    #[arg(long, default_value_t = 1.0)]
    pub stop_loss: f64,

    /// Delay between Fugle quote requests, to respect the rate limit.
    #[arg(long, default_value_t = 1100)]
    pub quote_delay_ms: u64,

    /// Plan and print the orders without placing them.
    #[arg(long)]
    pub dry_run: bool,
}
