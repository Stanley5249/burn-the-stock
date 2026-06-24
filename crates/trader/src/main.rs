//! Live trading loop on sim stock: rank every ticker from data through yesterday, fetch
//! holdings and live Fugle quotes, then place the day's weighted orders. Sells exit on the
//! same ladder the backtest uses; buys size by score over the usable-cash budget.

mod calendar;
mod cli;
mod execute;
mod plan;
mod rank;
mod report;

use std::collections::HashSet;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use chrono::{DateTime, FixedOffset, NaiveDate, Utc};
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result, ensure};
use stock_client::fugle::FugleClient;
use stock_client::sim_stock::SimStockClient;
use stock_data::read::History;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

use crate::calendar::{Session, TradingCalendar};
use crate::cli::Args;
use crate::execute::{place_buys, place_sells};
use crate::plan::{plan_buys, plan_sells};
use crate::rank::{rank, select_candidates};
use crate::report::{report_buys, report_candidates, report_sells};

type Backend = Wgpu;

/// Guard the session and confirm the OHLCV is fresh enough to trade. Logs in to sim along the
/// way so the login round trip overlaps the calendar fetch.
async fn preflight(
    sim: &SimStockClient,
    args: &Args,
    datetime: DateTime<FixedOffset>,
) -> Result<Session> {
    let (calendar, ()) = tokio::try_join!(TradingCalendar::build(&args.holiday_cache), async {
        sim.login().await.wrap_err("login to sim stock")
    },)?;

    let today = datetime.date_naive();
    let day_kind = calendar.day_kind(today);
    let session = calendar.session(datetime);
    tracing::info!(%today, %day_kind, %session.date, %session.prior);

    // Refresh the OHLCV before scoring so a forgotten refresh never trades stale data.
    if !args.no_download {
        let floor: NaiveDate = stock_data::schema::DEFAULT_FLOOR
            .parse()
            .into_diagnostic()?;
        stock_data::refresh::refresh(&args.data, floor).await?;
    }

    let last_date = History::scan(&args.data)?
        .last_date()
        .wrap_err_with(|| format!("scan {} for the last date", args.data.display()))?;

    ensure!(
        last_date >= session.prior || args.allow_stale,
        help =
            "the latest bar may not be published yet; retry after the close, or pass --allow-stale",
        "data is current through {last_date} but session {} needs it through {}",
        session.date,
        session.prior
    );

    Ok(session)
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::try_new("warn,trader=info,stock=info").unwrap()),
        )
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .init();

    let args = Args::parse();

    let datetime = Utc::now().with_timezone(&calendar::TAIPEI_OFFSET);

    ensure!(
        !calendar::in_maintenance(datetime),
        help = "re-run after 16:00",
        "sim stock is in maintenance (15:30-16:00 Taipei), now {datetime}"
    );

    let device = WgpuDevice::default();
    let fugle = FugleClient::from_env()?;
    let sim = SimStockClient::from_env(None)?;

    let session = preflight(&sim, &args, datetime).await?;

    // Inference over the fresh data, overlapping the holdings fetch.
    let rank_task = tokio::task::spawn_blocking({
        let args = args.clone();
        let device = device.clone();
        move || rank::<Backend>(&args, &device)
    });

    let holdings = sim
        .user_stocks()
        .await
        .wrap_err("fetch sim stock holdings")?;

    let ranked = rank_task
        .await
        .into_diagnostic()
        .wrap_err("ranking task panicked")??;

    let candidates = select_candidates(&ranked, args.threshold, args.max_holdings);

    // A dry run only checks the model's picks, so skip the live Fugle quotes (a 110s+ paced
    // sweep) and the whole order flow.
    if args.dry_run {
        print!(
            "{}",
            report_candidates(session.date, holdings.len(), &candidates)
        );
        return Ok(());
    }

    // One sweep over everything we might trade, with profile hidden under the long Fugle sweep.
    let symbols: Vec<String> = holdings
        .iter()
        .map(|holding| holding.stock_code_id.clone())
        .chain(candidates.iter().map(|(ticker, _)| ticker.clone()))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let (profile, quotes) =
        tokio::join!(sim.profile(), fugle.quotes(&symbols, args.quote_delay_ms));
    let profile = profile.wrap_err("fetch profile")?;
    let budget = profile.usable_cash * (1.0 - args.buffer);

    let sells = plan_sells(&holdings, &quotes);
    let buys = plan_buys(&candidates, &quotes, budget);
    print!(
        "{}",
        report_sells(session.date, budget, holdings.len(), &profile, &sells)
    );
    print!("{}", report_buys(candidates.len(), &buys));

    // Sim server is single-threaded, so place exits fully before entries.
    place_sells(&sim, &sells).await?;
    place_buys(&sim, &buys).await?;

    println!("placed {} sells and {} buys", sells.len(), buys.len());

    Ok(())
}
