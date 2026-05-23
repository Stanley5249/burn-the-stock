use chrono::{Local, NaiveDate};
use clap::Parser;
use miette::{IntoDiagnostic, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use std::path::{Path, PathBuf};
use stock_client::market_data::{
    fetch_candles_chunk, fetch_tickers, FugleCandleBar, FugleMarket, CANDLE_CHUNK_DAYS,
};

const ALL_MARKETS: &[FugleMarket] = &[FugleMarket::Tse, FugleMarket::Otc, FugleMarket::Esb];

#[derive(Parser)]
#[command(about = "Download historical daily OHLCV data via the Fugle API")]
struct Args {
    /// Fugle API key (falls back to FUGLE_API_KEY env var)
    #[arg(long, env = "FUGLE_API_KEY")]
    api_key: String,

    /// First date to download (inclusive)
    #[arg(long, default_value = "2016-01-01")]
    from: NaiveDate,

    /// Last date to download (inclusive), defaults to today
    #[arg(long)]
    to: Option<NaiveDate>,

    /// Output directory for CSV files
    #[arg(long, default_value = "data")]
    output: PathBuf,
}

// --- Setup ---

fn build_client(api_key: &str) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-API-KEY",
        HeaderValue::from_str(api_key).into_diagnostic()?,
    );
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .into_diagnostic()
}

// --- CSV writing ---

fn opt_f64(value: Option<f64>) -> String {
    value.map(|f| f.to_string()).unwrap_or_default()
}

fn write_csv_inner(path: &Path, symbol: &str, bars: &[FugleCandleBar]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path).into_diagnostic()?;
    writer
        .write_record([
            "date", "code", "open", "high", "low", "close", "change", "volume", "turnover",
        ])
        .into_diagnostic()?;

    for bar in bars {
        writer
            .write_record([
                &bar.date.to_string(),
                symbol,
                &opt_f64(bar.open),
                &opt_f64(bar.high),
                &opt_f64(bar.low),
                &opt_f64(bar.close),
                &opt_f64(bar.change),
                &opt_f64(bar.volume),
                &opt_f64(bar.turnover),
            ])
            .into_diagnostic()?;
    }

    writer.flush().into_diagnostic()
}

/// Write `bars` to a temporary file then atomically rename to `path`.
///
/// A crash mid-write leaves a `.tmp` file behind, not a partial `.csv`, so the
/// symbol is cleanly retried on the next run.
fn write_csv(path: &Path, symbol: &str, bars: &[FugleCandleBar]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    let result = write_csv_inner(&tmp_path, symbol, bars);
    if result.is_ok() {
        std::fs::rename(&tmp_path, path).into_diagnostic()?;
    } else {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

// --- Download logic ---

async fn download_symbol(
    http: &reqwest::Client,
    interval: &mut tokio::time::Interval,
    market: FugleMarket,
    symbol: &str,
    from: NaiveDate,
    to: NaiveDate,
    output: &Path,
) {
    let mut all_bars: Vec<FugleCandleBar> = Vec::new();
    let mut chunk_from = from;

    while chunk_from <= to {
        let chunk_to = (chunk_from + chrono::Duration::days(CANDLE_CHUNK_DAYS)).min(to);

        interval.tick().await;

        match fetch_candles_chunk(http, symbol, chunk_from, chunk_to).await {
            Ok(response) => all_bars.extend(response.data),
            Err(error) => {
                tracing::warn!(%symbol, %error, "fetch failed");
                return;
            }
        }

        chunk_from = chunk_to + chrono::Duration::days(1);
    }

    if all_bars.is_empty() {
        tracing::debug!(%symbol, "no bars, skipping");
        return;
    }

    let dir = output.join(market.as_str().to_lowercase());
    let path = dir.join(format!("{symbol}.csv"));
    match write_csv(&path, symbol, &all_bars) {
        Ok(()) => tracing::info!(%symbol, %market, bars = all_bars.len(), "saved"),
        Err(error) => tracing::warn!(%symbol, %error, "write failed"),
    }
}

async fn collect_pending(http: &reqwest::Client, output: &Path) -> Vec<(FugleMarket, String)> {
    let mut pending = Vec::new();

    for market in ALL_MARKETS {
        let dir = output.join(market.as_str().to_lowercase());

        if let Err(error) = std::fs::create_dir_all(&dir) {
            tracing::warn!(%market, %error, "could not create output directory");
            continue;
        }

        match fetch_tickers(http, *market).await {
            Ok(tickers) => {
                tracing::info!(market = %market, count = tickers.len(), "fetched tickers");
                for ticker in tickers {
                    let path = dir.join(format!("{}.csv", ticker.symbol));
                    if !path.exists() {
                        pending.push((*market, ticker.symbol));
                    }
                }
            }
            Err(error) => tracing::warn!(%market, %error, "failed to fetch tickers"),
        }
    }

    pending
}

async fn run_downloads(
    http: &reqwest::Client,
    pending: Vec<(FugleMarket, String)>,
    from: NaiveDate,
    to: NaiveDate,
    output: &Path,
) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    for (market, symbol) in pending {
        download_symbol(http, &mut interval, market, &symbol, from, to, output).await;
    }
}

// --- Entry point ---

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();
    let to = args.to.unwrap_or_else(|| Local::now().date_naive());

    let http = build_client(&args.api_key)?;

    tracing::info!(from = %args.from, %to, "starting download");

    let pending = collect_pending(&http, &args.output).await;
    tracing::info!(total = pending.len(), "symbols to download");

    run_downloads(&http, pending, args.from, to, &args.output).await;

    tracing::info!("download complete");

    Ok(())
}
