//! Manual OHLCV refresh from Yahoo Finance into the consolidated history parquet.

use std::path::PathBuf;

use chrono::NaiveDate;
use clap::Parser;
use miette::Result;
use stock_data::refresh::refresh;
use stock_data::schema::{DEFAULT_FLOOR, DEFAULT_PATH};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(about = "Refresh the OHLCV history parquet from Yahoo Finance")]
struct Args {
    /// History parquet to update in place.
    #[arg(long, default_value = DEFAULT_PATH)]
    output: PathBuf,

    /// First bar to fetch when the parquet does not exist yet.
    #[arg(long, default_value = DEFAULT_FLOOR)]
    floor: NaiveDate,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    refresh(&args.output, args.floor).await
}
