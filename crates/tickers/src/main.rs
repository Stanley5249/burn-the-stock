//! Prefetch per-ticker static metadata (`industry`, `securityType`) from Fugle
//! into a parquet the trainer joins against price data. The universe comes from
//! the price parquet, not the list endpoint, which is only live during a session.
//! Each code needs its own `intraday/ticker` call, so the run is rate-limited and
//! meant to run once.

use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use polars::prelude::*;
use reqwest::header::{HeaderMap, HeaderValue};
use std::path::{Path, PathBuf};
use std::time::Duration;
use stock_client::fugle::fetch_ticker;

#[derive(Parser, Debug)]
#[command(about = "Prefetch Fugle ticker industry metadata into parquet")]
struct Args {
    /// Price parquet to read the (market, code) universe from.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    input: PathBuf,

    /// Output parquet path.
    #[arg(long, default_value = "data/yfinance/tickers.parquet")]
    output: PathBuf,

    /// Only fetch the first N tickers (for trial runs).
    #[arg(long)]
    limit: Option<usize>,

    /// Delay between Fugle requests in milliseconds (limit is roughly 60/min).
    #[arg(long, default_value_t = 1100)]
    delay_ms: u64,
}

/// Distinct `(market, code)` pairs from the price parquet, sorted for a
/// deterministic fetch order.
fn load_universe(input: &Path) -> miette::Result<Vec<(String, String)>> {
    let path = input
        .to_str()
        .ok_or_else(|| miette::miette!("input path is not valid UTF-8"))?;

    let frame = LazyFrame::scan_parquet(PlRefPath::new(path), ScanArgsParquet::default())
        .into_diagnostic()?
        .select([
            col("market").cast(DataType::String),
            col("code").cast(DataType::String),
        ])
        .unique(None, UniqueKeepStrategy::Any)
        .sort(["market", "code"], SortMultipleOptions::new())
        .collect()
        .into_diagnostic()?;

    let markets = frame
        .column("market")
        .into_diagnostic()?
        .str()
        .into_diagnostic()?;
    let codes = frame
        .column("code")
        .into_diagnostic()?
        .str()
        .into_diagnostic()?;

    let universe = markets
        .into_iter()
        .zip(codes)
        .filter_map(|pair| match pair {
            (Some(market), Some(code)) => Some((market.to_owned(), code.to_owned())),
            _ => None,
        })
        .collect();

    Ok(universe)
}

/// Build a Fugle HTTP client with the `X-API-KEY` header from the environment.
fn build_client() -> Result<reqwest::Client> {
    let api_key = std::env::var("FUGLE_API_KEY")
        .into_diagnostic()
        .wrap_err("`FUGLE_API_KEY` must be set")?;

    let mut headers = HeaderMap::new();
    headers.insert(
        "X-API-KEY",
        HeaderValue::from_str(&api_key).into_diagnostic()?,
    );

    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .into_diagnostic()
}

/// Columns accumulated from per-ticker fetches, ready to become a frame.
#[derive(Default)]
struct TickerColumns {
    markets: Vec<String>,
    codes: Vec<String>,
    industries: Vec<Option<String>>,
    security_types: Vec<Option<String>>,
}

/// Fetch metadata for every `(market, code)`, sleeping `delay` between requests.
/// A failed lookup is logged and skipped so one bad symbol does not abort the run.
async fn fetch_all(
    http: &reqwest::Client,
    universe: &[(String, String)],
    delay: Duration,
) -> TickerColumns {
    let mut columns = TickerColumns::default();

    for (index, (market, symbol)) in universe.iter().enumerate() {
        if index > 0 {
            tokio::time::sleep(delay).await;
        }

        match fetch_ticker(http, symbol).await {
            Ok(detail) => {
                tracing::info!(market = %market, symbol = %symbol, industry = ?detail.industry, "fetched");
                columns.markets.push(market.clone());
                columns.codes.push(symbol.clone());
                columns.industries.push(detail.industry);
                columns.security_types.push(detail.security_type);
            }
            Err(error) => {
                tracing::warn!(market = %market, symbol = %symbol, ?error, "skipped ticker");
            }
        }
    }

    columns
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt().init();

    let args = Args::parse();

    // The polars lazy scan runs its own runtime, so do it before entering one.
    let mut universe = load_universe(&args.input)?;

    if let Some(limit) = args.limit {
        universe.truncate(limit);
    }

    tracing::info!(total = universe.len(), "fetching ticker metadata");

    let http = build_client()?;
    let delay = Duration::from_millis(args.delay_ms);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .into_diagnostic()?;
    let columns = runtime.block_on(fetch_all(&http, &universe, delay));

    let mut frame = df!(
        "market" => columns.markets,
        "code" => columns.codes,
        "industry" => columns.industries,
        "security_type" => columns.security_types,
    )
    .into_diagnostic()?;

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).into_diagnostic()?;
    }

    let mut file = std::fs::File::create(&args.output).into_diagnostic()?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut frame)
        .into_diagnostic()?;

    tracing::info!(rows = frame.height(), path = %args.output.display(), "wrote tickers parquet");

    Ok(())
}
