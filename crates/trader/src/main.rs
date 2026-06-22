//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the settled-cash budget.

mod cli;
mod execute;
mod plan;
mod rank;
mod report;
mod state;

use std::collections::HashMap;

use burn::backend::wgpu::WgpuDevice;
use chrono::{Duration, Local};
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use portfolio::STARTING_CASH;
use stock_client::fugle::{FugleQuote, client, fetch_quotes};
use stock_client::sim_stock::SimStockClient;
use tracing_subscriber::EnvFilter;

use crate::cli::Args;
use crate::execute::{place_buys, place_sells};
use crate::plan::{plan_buys, plan_sells};
use crate::rank::{rank, select_candidates};
use crate::report::{report_buys, report_candidates, report_sells};
use crate::state::LiveState;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("warn,stock_client=info,trader=info")),
        )
        .init();

    let args = Args::parse();
    let device = WgpuDevice::default();

    // One Fugle-keyed client, shared with the sim client (sim ignores the extra header).
    let api_key = std::env::var("FUGLE_API_KEY")
        .into_diagnostic()
        .wrap_err("FUGLE_API_KEY must be set")?;
    let http = client(&api_key).into_diagnostic()?;
    let sim = SimStockClient::from_env(http.clone()).into_diagnostic()?;

    // Inference reads local parquet (today's bar is never read) and is the long pole; start it
    // now on a blocking thread and only await it once candidates are needed for the buys.
    let rank_task = tokio::task::spawn_blocking({
        let args = args.clone();
        let device = device.clone();
        move || rank(&args, &device)
    });

    // Positions come from the platform; cash from our own ledger. The sim_stock holdings
    // fetch is fast, unlike the paced Fugle quote sweep below.
    let holdings = sim
        .user_stocks()
        .await
        .into_diagnostic()
        .wrap_err("fetch holdings")?;

    let today = Local::now().date_naive();

    // A dry run only checks the model's picks, so skip the live Fugle quotes (a 110s+ paced
    // sweep) and the whole order flow; just rank and list the candidates.
    if args.dry_run {
        let ranked = rank_task
            .await
            .into_diagnostic()
            .wrap_err("ranking task panicked")??;
        let candidates = select_candidates(&ranked, args.threshold, args.max_holdings);

        print!("{}", report_candidates(today, holdings.len(), &candidates));
        return Ok(());
    }

    // Sells depend only on held names, never on the ranking, so quote them first and overlap
    // that fetch with `rank`.
    let held_symbols: Vec<String> = holdings.iter().map(|h| h.stock_code_id.clone()).collect();
    let held_quotes = fetch_quotes(&http, &held_symbols, args.quote_delay_ms).await;

    let mut state = LiveState::load_or_seed(&args.state, STARTING_CASH)?;
    state.settle(today);

    // ponytail: rotation recycles capital only as fast as sells settle; if settlement lags
    // the order cutoff, a split-capital cadence is the follow-up. Sells settle later, so the
    // budget reads settled cash once here and is unaffected by today's sells.
    let budget = state.settled_cash * (1.0 - args.buffer);

    let sells = plan_sells(&holdings, &held_quotes);
    print!(
        "{}",
        report_sells(today, state.settled_cash, budget, holdings.len(), &sells)
    );

    // Rank has overlapped the held-quote fetch and is almost certainly done; await it now to
    // get the candidates the buys size against.
    let ranked = rank_task
        .await
        .into_diagnostic()
        .wrap_err("ranking task panicked")??;
    let candidates = select_candidates(&ranked, args.threshold, args.max_holdings);

    // Quote only candidates not already covered by the held sweep, so Fugle never sees a
    // symbol twice across the two phases.
    let candidate_symbols: Vec<String> = candidates
        .iter()
        .map(|(ticker, _)| ticker.clone())
        .filter(|ticker| !held_quotes.contains_key(ticker))
        .collect();

    // Fire the sells (sim POSTs) while the candidate quote sweep runs on Fugle; the two hit
    // different hosts, so they overlap cleanly.
    let sell_future = place_sells(&sim, &sells, args.order_concurrency);
    let quote_future = fetch_quotes(&http, &candidate_symbols, args.quote_delay_ms);

    let (sell_result, candidate_quotes) = tokio::join!(sell_future, quote_future);
    sell_result?;

    let mut quotes: HashMap<String, FugleQuote> = held_quotes;
    quotes.extend(candidate_quotes);

    let buys = plan_buys(&candidates, &quotes, budget, args.max_holdings);
    print!("{}", report_buys(candidates.len(), &buys));

    place_buys(&sim, &buys, args.order_concurrency).await?;

    let settle_date = today + Duration::days(args.settle_lag);
    for sell in &sells {
        state.record_sell(sell.proceeds, settle_date);
    }
    for buy in &buys {
        state.record_buy(buy.cost);
    }
    state.last_run = Some(today);
    state.save(&args.state)?;
    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}
