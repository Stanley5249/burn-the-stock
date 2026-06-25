//! Manual OHLCV refresh from Yahoo Finance into the consolidated history parquet.

use std::path::PathBuf;

use chrono::{Local, NaiveDate, TimeDelta};
use clap::Parser;
use miette::Result;
use stock_client::sim_stock::SimStockClient;
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
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,stock=info")),
        )
        .init();

    let args = Args::parse();

    let client = SimStockClient::from_env(None, None)?;

    let stock_list = client.stock_list().await?;

    // Stop at yesterday; today's bar is not final until the market closes.
    let end = Local::now().date_naive() - TimeDelta::days(1);

    refresh(stock_list, &args.output, args.floor, end).await?;

    Ok(())
}
