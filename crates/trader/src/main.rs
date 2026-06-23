//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the usable-cash budget.

mod cli;
mod execute;
mod plan;
mod rank;
mod report;

use std::collections::HashMap;

use burn::backend::wgpu::WgpuDevice;
use chrono::Local;
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use stock_client::fugle::{FugleClient, FugleQuote};
use stock_client::sim_stock::SimStockClient;
use tracing_subscriber::EnvFilter;

use crate::cli::Args;
use crate::execute::{place_buys, place_sells};
use crate::plan::{plan_buys, plan_sells};
use crate::rank::{rank, select_candidates};
use crate::report::{report_buys, report_candidates, report_sells};

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

    let fugle = FugleClient::from_env()?;

    let sim_stock_client = SimStockClient::from_env(None)?;

    // login is stateful; do it once so the profile scrape below reuses the session cookie.
    sim_stock_client.login().await.wrap_err("login")?;

    // Inference reads local parquet (today's bar is never read) and is the long pole; start it
    // now on a blocking thread and only await it once candidates are needed for the buys.
    let rank_task = tokio::task::spawn_blocking({
        let args = args.clone();
        let device = device.clone();
        move || rank(&args, &device)
    });

    // Positions come from the platform. The sim_stock holdings fetch is fast, unlike the
    // paced Fugle quote sweep below.
    let holdings = sim_stock_client
        .user_stocks()
        .await
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
    let held_quotes = fugle.quotes(&held_symbols, args.quote_delay_ms).await;

    // The platform's usable balance is the source of truth for the budget; a failed scrape
    // stops the run rather than trading on a guess.
    let profile = sim_stock_client.profile().await.wrap_err("fetch profile")?;
    let budget = profile.usable_cash * (1.0 - args.buffer);

    let sells = plan_sells(&holdings, &held_quotes);
    print!(
        "{}",
        report_sells(today, budget, holdings.len(), &profile, &sells)
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
    let sell_future = place_sells(&sim_stock_client, &sells, args.order_concurrency);
    let quote_future = fugle.quotes(&candidate_symbols, args.quote_delay_ms);

    let (sell_result, candidate_quotes) = tokio::join!(sell_future, quote_future);
    sell_result?;

    let mut quotes: HashMap<String, FugleQuote> = held_quotes;
    quotes.extend(candidate_quotes);

    let buys = plan_buys(&candidates, &quotes, budget, args.max_holdings);
    print!("{}", report_buys(candidates.len(), &buys));

    place_buys(&sim_stock_client, &buys, args.order_concurrency).await?;

    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}
