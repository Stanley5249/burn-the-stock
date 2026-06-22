//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the settled-cash budget.

mod cli;
mod execute;
mod plan;
mod rank;
mod report;
mod state;

use burn::backend::wgpu::WgpuDevice;
use chrono::Local;
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use portfolio::STARTING_CASH;
use stock_client::fugle::{client, fetch_quotes};
use stock_client::sim_stock::SimStockClient;
use tracing_subscriber::EnvFilter;

use crate::cli::Args;
use crate::execute::execute;
use crate::plan::{plan_buys, plan_sells};
use crate::rank::rank;
use crate::report::report;
use crate::state::LiveState;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv_override().ok();
    // wgpu/cubecl emit noisy info logs, so default to warnings plus our own crates' info
    // (the per-order buy/sell confirmations). RUST_LOG overrides.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("warn,stock_client=info,trader=info")
        }))
        .init();
    let args = Args::parse();
    let device = WgpuDevice::default();

    // Rank every ticker from data ending yesterday; today's bar is never read.
    let ranked = rank(&args, &device)?;

    // One Fugle-keyed client, shared with the sim client (sim ignores the extra header).
    let api_key = std::env::var("FUGLE_API_KEY")
        .into_diagnostic()
        .wrap_err("FUGLE_API_KEY must be set")?;
    let http = client(&api_key).into_diagnostic()?;
    let sim = SimStockClient::from_env(http.clone()).into_diagnostic()?;

    // Positions come from the platform; cash comes from our own ledger.
    let holdings = sim.user_stocks().await.into_diagnostic()?;

    let today = Local::now().date_naive();
    let mut state = LiveState::load_or_seed(&args.state, STARTING_CASH)?;
    state.settle(today);

    // Daily rotation sells the whole book, so held names are buyable again (rebought at the
    // low after selling at the high) and stay in the candidate list.
    let candidates: Vec<(String, f32)> = ranked
        .iter()
        .filter(|(_, score)| *score > args.threshold)
        .take(args.max_holdings)
        .cloned()
        .collect();

    // Quote held names (to sell) and candidates (to buy); dedup since rotation lists held
    // names in both.
    let mut symbols: Vec<String> = holdings.iter().map(|h| h.stock_code_id.clone()).collect();
    symbols.extend(candidates.iter().map(|(ticker, _)| ticker.clone()));
    symbols.sort();
    symbols.dedup();
    let quotes = fetch_quotes(&http, &symbols, args.quote_delay_ms).await;

    let sells = plan_sells(&holdings, &quotes);

    // ponytail: rotation recycles capital only as fast as sells settle; if settlement lags
    // the order cutoff, a split-capital cadence is the follow-up.
    let budget = state.settled_cash * (1.0 - args.buffer);
    let buys = plan_buys(&candidates, &quotes, budget, args.max_holdings);

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
