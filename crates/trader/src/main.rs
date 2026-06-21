//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the settled-cash budget.

mod cli;
mod execute;
mod plan;
mod quotes;
mod rank;
mod report;
mod state;

use std::collections::{HashMap, HashSet};

use burn::backend::wgpu::WgpuDevice;
use chrono::Local;
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use portfolio::STARTING_CASH;
use reqwest::header::{HeaderMap, HeaderValue};
use stock_client::sim_stock::SimStockClient;

use crate::cli::Args;
use crate::execute::execute;
use crate::plan::{plan_buys, plan_sells};
use crate::quotes::fetch_quotes;
use crate::rank::rank;
use crate::report::report;
use crate::state::LiveState;

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

    execute(&sim, &sells, &buys, &mut state, today, args.settle_lag).await?;
    state.last_run = Some(today);
    state.save(&args.state)?;
    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}
