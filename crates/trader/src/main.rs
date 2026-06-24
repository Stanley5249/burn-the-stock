//! Live trading loop on `sim_stock`: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the usable-cash budget.

mod calendar;
mod cli;
mod execute;
mod plan;
mod rank;
mod report;

use std::collections::HashMap;

use burn::backend::wgpu::WgpuDevice;
use chrono::{Datelike, NaiveDate, Utc};
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result, ensure};
use stock_client::fugle::{FugleClient, FugleQuote};
use stock_client::sim_stock::SimStockClient;
use tracing_subscriber::EnvFilter;

use crate::calendar::DayKind;
use crate::cli::Args;
use crate::execute::{place_buys, place_sells};
use crate::plan::{plan_buys, plan_sells};
use crate::rank::{data_max_date, rank, select_candidates};
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

    let datetime = Utc::now().with_timezone(&calendar::TAIPEI_OFFSET);

    ensure!(
        !calendar::in_maintenance(datetime),
        help = "re-run after 16:00",
        "sim_stock is in maintenance (15:30-16:00 Taipei), now {datetime}"
    );

    // The TWSE calendar decides which session these orders target and the last completed
    // session the model must have data through.
    let calendar = calendar::build(datetime.year(), &args.holiday_cache).await?;
    let today = datetime.date_naive();
    let today_kind = calendar.day_kind(today);
    let session = calendar.session(datetime);
    let target_session = session.target;

    // Reveal the trader's read of the calendar and why the orders land on this session.
    let reason = if target_session == today {
        format!("today is a {today_kind}, before the 13:00 cutoff")
    } else if matches!(today_kind, DayKind::Trading) {
        "today is past the 13:00 cutoff, so orders queue to the next session".to_string()
    } else {
        format!("today is a {today_kind}, so orders queue to the next session")
    };
    println!(
        "{}: {reason}; orders target {target_session}, data required through {}",
        datetime.format("%Y-%m-%d %H:%M %:z"),
        session.data_through,
    );

    // Refresh the OHLCV before scoring so a forgotten refresh never trades stale data.
    if !args.no_download {
        let floor: NaiveDate = stock_data::schema::DEFAULT_FLOOR
            .parse()
            .into_diagnostic()?;
        stock_data::refresh::refresh(&args.data, floor).await?;
    }

    let max_date = data_max_date(&args.data).wrap_err("read data max date")?;
    ensure!(
        max_date >= session.data_through || args.allow_stale,
        help =
            "the latest bar may not be published yet; retry after the close, or pass --allow-stale",
        "data is current through {max_date} but session {target_session} needs it through {}",
        session.data_through
    );

    let device = WgpuDevice::default();

    let fugle = FugleClient::from_env()?;

    let sim_stock_client = SimStockClient::from_env(None)?;

    // login is stateful; do it once so the profile scrape below reuses the session cookie.
    sim_stock_client.login().await.wrap_err("login")?;

    // Inference reads local parquet (the target session's bar is never read) and is the long
    // pole; start it now on a blocking thread and only await it once candidates are needed.
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

    // A dry run only checks the model's picks, so skip the live Fugle quotes (a 110s+ paced
    // sweep) and the whole order flow; just rank and list the candidates.
    if args.dry_run {
        let ranked = rank_task
            .await
            .into_diagnostic()
            .wrap_err("ranking task panicked")??;
        let candidates = select_candidates(&ranked, args.threshold, args.max_holdings);

        print!(
            "{}",
            report_candidates(target_session, holdings.len(), &candidates)
        );
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
        report_sells(target_session, budget, holdings.len(), &profile, &sells)
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

    let buys = plan_buys(&candidates, &quotes, budget);
    print!("{}", report_buys(candidates.len(), &buys));

    place_buys(&sim_stock_client, &buys, args.order_concurrency).await?;

    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}
