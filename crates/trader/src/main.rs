//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the settled-cash budget.

mod state;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use chrono::{Duration, Local, NaiveDate};
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use polars::prelude::*;
use portfolio::{
    DayBar, ExitReason, Fill, SELL_TAX_RATE, STARTING_CASH, affordable_shares, commission,
    exit_decision, score_weights, sell_price, tick_ceil,
};
use reqwest::header::{HeaderMap, HeaderValue};
use stock_client::fugle::{FugleQuote, fetch_quote};
use stock_client::sim_stock::SimStockClient;
use stock_model::data::TickerFrames;
use stock_model::features::DATE;
use stock_model::inference::{InferenceConfig, score};

use crate::state::LiveState;

type Backend = Wgpu;

#[derive(Parser, Debug)]
#[command(about = "Predict, then place the day's weighted orders on sim_stock")]
struct Args {
    /// Directory holding a training run's `config.json` and `model`.
    #[arg(long, default_value = "artifacts/latest")]
    artifact_dir: PathBuf,

    /// OHLCV parquet to score; only its recent tail is read. Must be current through
    /// yesterday's close, so refresh it with the downloader before running.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    data: PathBuf,

    /// Cash ledger path; tracks settled cash and unsettled sale proceeds across runs.
    #[arg(long, default_value = "data/live-state.json")]
    state: PathBuf,

    /// Minimum predicted score (per-date z-scored MFE) to buy.
    #[arg(long, default_value_t = 0.0)]
    threshold: f32,

    /// Target number of positions to hold.
    #[arg(long, default_value_t = 100)]
    max_holdings: usize,

    /// How many top-ranked candidates to fetch quotes for.
    #[arg(long, default_value_t = 100)]
    shortlist: usize,

    /// Fraction of settled cash held back for later days.
    #[arg(long, default_value_t = 0.1)]
    buffer: f64,

    /// Calendar days until sale proceeds settle into spendable cash.
    #[arg(long, default_value_t = 2)]
    settle_lag: i64,

    /// Trading days to hold before the time exit.
    #[arg(long, default_value_t = 20)]
    max_hold: usize,

    /// Take-profit exit as a fraction of price; 1.0 is effectively off.
    #[arg(long, default_value_t = 1.0)]
    take_profit: f64,

    /// Stop-loss exit as a fraction of price; 1.0 is effectively off.
    #[arg(long, default_value_t = 1.0)]
    stop_loss: f64,

    /// Delay between Fugle quote requests, to respect the rate limit.
    #[arg(long, default_value_t = 1100)]
    quote_delay_ms: u64,

    /// Plan and print the orders without placing them.
    #[arg(long)]
    dry_run: bool,
}

/// A planned exit: sell the whole position at the quote high.
struct Sell {
    code: String,
    shares: u64,
    price: f64,
    proceeds: f64,
    reason: ExitReason,
}

/// A planned entry: buy a whole-lot quantity at the quote low.
struct Buy {
    code: String,
    shares: u64,
    price: f64,
    cost: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv_override().ok();
    let args = Args::parse();
    let device = WgpuDevice::default();

    // Rank every ticker from data ending yesterday; today's bar is never read.
    let ranked = rank(&args, &device)?;
    let score_of: HashMap<String, f32> = ranked.iter().cloned().collect();

    // One Fugle-keyed client, shared with the sim client (sim ignores the extra header).
    let api_key = std::env::var("FUGLE_API_KEY")
        .into_diagnostic()
        .wrap_err("FUGLE_API_KEY must be set")?;
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-API-KEY",
        HeaderValue::from_str(&api_key).into_diagnostic()?,
    );
    let http = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .into_diagnostic()?;
    let sim = SimStockClient::from_env(http.clone()).into_diagnostic()?;

    // Positions come from the platform; cash comes from our own ledger.
    let holdings = sim.user_stocks().await.into_diagnostic()?;
    let held: HashSet<String> = holdings.iter().map(|h| h.stock_code_id.clone()).collect();

    let today = Local::now().date_naive();
    let mut state = LiveState::load_or_seed(&args.state, STARTING_CASH)?;
    state.settle(today);

    // Top above-gate names we do not already hold.
    let candidates: Vec<(String, f32)> = ranked
        .iter()
        .filter(|(ticker, score)| !held.contains(ticker) && *score > args.threshold)
        .take(args.shortlist)
        .cloned()
        .collect();

    // Quote held names (to sell) and candidates (to buy).
    let mut symbols: Vec<String> = holdings.iter().map(|h| h.stock_code_id.clone()).collect();
    symbols.extend(candidates.iter().map(|(ticker, _)| ticker.clone()));
    let quotes = fetch_quotes(&http, &symbols, args.quote_delay_ms).await;

    let sells = plan_sells(&holdings, &quotes, &score_of, &args, today);

    let budget = state.settled_cash * (1.0 - args.buffer);
    let open_slots = args.max_holdings.saturating_sub(holdings.len());
    let buys = plan_buys(&candidates, &quotes, budget, open_slots);

    print!(
        "{}",
        report(
            today,
            state.settled_cash,
            budget,
            holdings.len(),
            candidates.len(),
            &sells,
            &buys,
        )
    );

    if args.dry_run {
        println!("dry run: no orders placed");
        return Ok(());
    }

    let settle_date = today + Duration::days(args.settle_lag);
    for sell in &sells {
        sim.sell(&sell.code, sell.shares, sell.price)
            .await
            .into_diagnostic()
            .wrap_err_with(|| format!("sell {}", sell.code))?;
        state.record_sell(sell.proceeds, settle_date);
    }
    for buy in &buys {
        sim.buy(&buy.code, buy.shares, buy.price)
            .await
            .into_diagnostic()
            .wrap_err_with(|| format!("buy {}", buy.code))?;
        state.record_buy(buy.cost);
    }

    state.last_run = Some(today);
    state.save(&args.state)?;
    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}

/// Score every ticker on its latest window and return `(ticker, score)` sorted strongest
/// first. The same inference path as the backtest.
fn rank(args: &Args, device: &WgpuDevice) -> Result<Vec<(String, f32)>> {
    let config = InferenceConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;
    let model = config
        .model
        .init::<Backend>(device)
        .load_file(
            args.artifact_dir.join("model"),
            &CompactRecorder::new(),
            device,
        )
        .into_diagnostic()
        .wrap_err("fail to init model from artifact")?;

    // Trading days are sparser than calendar days, so over-reach the lookback and let
    // `latest_windows` trim each ticker to exactly `config.steps`.
    let lookback = i64::try_from(config.steps * 2 + 10)
        .into_diagnostic()
        .wrap_err("steps too large for the lookback window")?;
    let frame = recent_frame(&args.data, lookback).into_diagnostic()?;
    let store = TickerFrames::from_lazy(frame).into_diagnostic()?;

    let windows = store.latest_windows(config.steps).into_diagnostic()?;
    let features = store.feature_series().into_diagnostic()?;
    let predictions = score::<Backend>(&model, &features, &windows, config.steps, device);

    let mut ranked: Vec<(String, f32)> = windows
        .into_iter()
        .zip(predictions)
        .map(|(window, prediction)| (window.ticker, prediction.score))
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    Ok(ranked)
}

/// Scan only the recent tail of the OHLCV parquet, keeping the last `lookback` calendar
/// days. The per-date z-score is unaffected since each retained date still holds the full
/// universe.
fn recent_frame(path: &Path, lookback: i64) -> PolarsResult<LazyFrame> {
    let frame =
        LazyFrame::scan_parquet(PlRefPath::try_from_path(path)?, ScanArgsParquet::default())?
            .with_column(col(DATE).cast(DataType::Date));

    let max_date = frame
        .clone()
        .select([col(DATE).max()])
        .collect()?
        .column(&DATE)?
        .date()?
        .as_date_iter()
        .flatten()
        .next()
        .expect("parquet has at least one dated row");

    let cutoff = max_date - Duration::days(lookback);

    Ok(frame.filter(col(DATE).gt_eq(lit(cutoff))))
}

/// Fetch a quote per symbol, sequential and rate-limited. A failed symbol is logged and
/// skipped rather than aborting the run.
async fn fetch_quotes(
    http: &reqwest::Client,
    symbols: &[String],
    delay_ms: u64,
) -> HashMap<String, FugleQuote> {
    let mut quotes = HashMap::with_capacity(symbols.len());
    for symbol in symbols {
        match fetch_quote(http, symbol).await {
            Ok(quote) => {
                quotes.insert(symbol.clone(), quote);
            }
            Err(error) => eprintln!("quote {symbol} failed: {error}"),
        }
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    quotes
}

/// Decide each holding's exit on the shared ladder, selling the whole position at the quote
/// high. Holdings without a usable quote are left alone.
fn plan_sells(
    holdings: &[stock_client::types::UserStock],
    quotes: &HashMap<String, FugleQuote>,
    score_of: &HashMap<String, f32>,
    args: &Args,
    today: NaiveDate,
) -> Vec<Sell> {
    let mut sells = Vec::new();
    for holding in holdings {
        let Some(quote) = quotes.get(&holding.stock_code_id) else {
            continue;
        };
        let score = score_of.get(&holding.stock_code_id).copied().unwrap_or(0.0);
        let Some(bar) = quote_to_bar(quote, score) else {
            continue;
        };

        let entry_price = holding
            .beginning_price
            .to_string()
            .parse::<f64>()
            .unwrap_or(0.0);
        let days = days_held(holding.createtime, today);

        if let Some((_, reason)) = exit_decision(
            entry_price,
            days,
            &bar,
            args.take_profit,
            args.stop_loss,
            args.max_hold,
            Fill::LowHigh,
        ) {
            let price = sell_price(&bar, Fill::LowHigh);
            #[allow(
                clippy::cast_precision_loss,
                reason = "share counts are small lot multiples"
            )]
            let amount = price * holding.shares as f64;
            let proceeds = amount - commission(amount) - amount * SELL_TAX_RATE;
            sells.push(Sell {
                code: holding.stock_code_id.clone(),
                shares: holding.shares,
                price,
                proceeds,
                reason,
            });
        }
    }
    sells
}

/// Size buys by score over the budget, filling the open slots with the strongest quoted
/// candidates. Cash drops as each fills, so later names get what is left.
fn plan_buys(
    candidates: &[(String, f32)],
    quotes: &HashMap<String, FugleQuote>,
    budget: f64,
    open_slots: usize,
) -> Vec<Buy> {
    let priced: Vec<(&String, f32, f64)> = candidates
        .iter()
        .filter_map(|(ticker, score)| {
            let quote = quotes.get(ticker)?;
            let low = quote.low_price.or(quote.open_price)?;
            Some((ticker, *score, tick_ceil(low)))
        })
        .take(open_slots)
        .collect();

    let weights = score_weights(
        &priced
            .iter()
            .map(|(_, score, _)| *score)
            .collect::<Vec<_>>(),
    );

    let mut remaining = budget;
    let mut buys = Vec::new();
    for ((code, _, price), weight) in priced.iter().zip(weights) {
        let target = budget * weight;
        let shares = affordable_shares(target.min(remaining), *price, remaining);
        if shares <= 0.0 {
            continue;
        }
        let amount = price * shares;
        let cost = amount + commission(amount);
        remaining -= cost;
        buys.push(Buy {
            code: (*code).clone(),
            shares: lots(shares),
            price: *price,
            cost,
        });
    }
    buys
}

/// Build a one-day bar from a live quote for the exit ladder, or `None` when a price is
/// still missing before the first trade of the session.
#[allow(clippy::cast_possible_truncation, reason = "TWSE prices fit f32")]
fn quote_to_bar(quote: &FugleQuote, score: f32) -> Option<DayBar> {
    Some(DayBar {
        score,
        open: quote.open_price? as f32,
        low: quote.low_price? as f32,
        high: quote.high_price? as f32,
        close: quote.last_price.or(quote.open_price)? as f32,
    })
}

/// Trading-day-agnostic days since entry, from the platform's epoch-second timestamp.
fn days_held(createtime: i64, today: NaiveDate) -> usize {
    let entry =
        chrono::DateTime::from_timestamp(createtime, 0).map_or(today, |moment| moment.date_naive());
    usize::try_from((today - entry).num_days().max(0)).unwrap_or(0)
}

/// Whole nonnegative lot count from [`affordable_shares`] as the order quantity.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "affordable_shares returns whole nonnegative lot multiples"
)]
fn lots(shares: f64) -> u64 {
    shares as u64
}

/// Render the day's plan as one grouped block.
fn report(
    today: NaiveDate,
    settled_cash: f64,
    budget: f64,
    holdings: usize,
    candidates: usize,
    sells: &[Sell],
    buys: &[Buy],
) -> String {
    use std::fmt::Write as _;

    let proceeds: f64 = sells.iter().map(|sell| sell.proceeds).sum();
    let cost: f64 = buys.iter().map(|buy| buy.cost).sum();

    let mut out = String::new();
    let _ = writeln!(out, "Live plan {today}");
    let _ = writeln!(out, "  settled cash : {settled_cash:.0}");
    let _ = writeln!(out, "  buy budget   : {budget:.0}");
    let _ = writeln!(out, "  holdings     : {holdings}");
    let _ = writeln!(out, "  candidates   : {candidates}");
    let _ = writeln!(out, "  Sells ({}), proceeds {proceeds:.0}", sells.len());
    for sell in sells {
        let _ = writeln!(
            out,
            "    {:<8} {:>7} @ {:.2}  [{}]",
            sell.code, sell.shares, sell.price, sell.reason
        );
    }
    let _ = writeln!(out, "  Buys ({}), cost {cost:.0}", buys.len());
    for buy in buys {
        let _ = writeln!(
            out,
            "    {:<8} {:>7} @ {:.2}  ({:.0})",
            buy.code, buy.shares, buy.price, buy.cost
        );
    }
    out
}
