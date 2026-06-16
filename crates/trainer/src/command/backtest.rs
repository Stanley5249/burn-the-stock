use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use chrono::{Duration, NaiveDate};
use miette::{IntoDiagnostic, Result};
use polars::prelude::*;
use stock_model::inference::{Action, Predictor};

use crate::cli::{BacktestArgs, FillArg};
use crate::data::store::TickerStore;
use crate::portfolio::{
    self, BacktestConfig, BacktestReport, DayBar, Fill, RenderContext, STARTING_CASH, TradingDay,
};
use crate::training::TrainingConfig;

type InferenceBackend = Wgpu;

/// Run the portfolio backtest over the held-out split, reporting metrics and a CSV of
/// the daily account value.
pub fn run(args: &BacktestArgs) -> Result<()> {
    let device = WgpuDevice::default();

    // Only the window length is needed from the run's config; barriers and labels do
    // not apply to inference.
    let config = TrainingConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    // Price-only load keeps every row, so the most recent bars stay tradeable.
    let store = TickerStore::load_prices(&args.data).into_diagnostic()?;

    let max_date = store
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - Duration::days(args.valid_days);

    // Windows ending on or after the cutoff, lookback drawn from earlier bars. Index
    // each signal by (ticker, signal date).
    let windows = store.backtest_windows_since(config.steps, cutoff);
    let predictions = predictor.predict(&windows);

    let mut signals: HashMap<String, HashMap<NaiveDate, (f32, Action)>> = HashMap::new();
    for prediction in &predictions {
        signals
            .entry(prediction.ticker.clone())
            .or_default()
            .insert(
                prediction.date,
                (prediction.expected_edge, prediction.action),
            );
    }

    let fill = match args.fill {
        FillArg::LowHigh => Fill::LowHigh,
        FillArg::Open => Fill::Open,
    };

    let days = build_days(&store, &signals);

    let backtest_config = BacktestConfig {
        threshold: args.threshold,
        fill,
        max_holdings: args.max_holdings,
        starting_cash: STARTING_CASH,
        take_profit: f64::from(args.take_profit.unwrap_or(config.take_profit)),
        stop_loss: f64::from(args.stop_loss.unwrap_or(config.stop_loss)),
        max_hold_days: args.max_hold.unwrap_or(config.label_horizon),
    };
    let report = portfolio::run(&days, &backtest_config);

    let context = RenderContext {
        tickers: store.ticker_count(),
        windows_scored: windows.len(),
        threshold: args.threshold,
        fill,
    };
    print!("{}", portfolio::summary(&report, &context));

    let equity_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.artifact_dir.join("backtest-equity.csv"));
    let log_dir = equity_path.parent().unwrap_or_else(|| Path::new("."));
    let trades_path = log_dir.join("backtest-trades.csv");
    let actions_path = log_dir.join("backtest-actions.csv");

    write_equity_csv(&report, &equity_path)?;
    write_trades_csv(&report, &trades_path)?;
    write_actions_csv(&report, &actions_path)?;
    println!(
        "\nWrote equity curve to {}\nWrote trades to {}\nWrote actions to {}",
        equity_path.display(),
        trades_path.display(),
        actions_path.display(),
    );

    Ok(())
}

/// Assemble the time-ordered day stream. The previous day's signal drives today's
/// order (the no-look-ahead lag), priced from today's bar; missing-price bars are
/// skipped.
fn build_days(
    data: &TickerStore,
    signals: &HashMap<String, HashMap<NaiveDate, (f32, Action)>>,
) -> Vec<TradingDay> {
    let mut by_date: BTreeMap<NaiveDate, HashMap<String, DayBar>> = BTreeMap::new();

    for quotes in data.quotes() {
        let ticker_signals = signals.get(&quotes.ticker);
        for index in 1..quotes.dates.len() {
            let signal_date = quotes.dates[index - 1];
            let Some(&(score, action)) = ticker_signals.and_then(|map| map.get(&signal_date))
            else {
                continue;
            };

            let (open, low, high, close) = (
                quotes.open[index],
                quotes.low[index],
                quotes.high[index],
                quotes.close[index],
            );
            if !(open.is_finite() && low.is_finite() && high.is_finite() && close.is_finite()) {
                continue;
            }

            by_date.entry(quotes.dates[index]).or_default().insert(
                quotes.ticker.clone(),
                DayBar {
                    score,
                    action,
                    open,
                    low,
                    high,
                    close,
                },
            );
        }
    }

    by_date
        .into_iter()
        .map(|(date, bars)| TradingDay { date, bars })
        .collect()
}

/// Write a `DataFrame` to `path` as a header-row CSV.
fn write_csv(frame: &mut DataFrame, path: &Path) -> Result<()> {
    let mut file = std::fs::File::create(path).into_diagnostic()?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(frame)
        .into_diagnostic()
}

/// Daily account value, for plotting the equity curve.
fn write_equity_csv(report: &BacktestReport, path: &Path) -> Result<()> {
    let curve = &report.equity_curve;
    let mut frame = df!(
        "date" => curve.iter().map(|point| point.date).collect::<Vec<_>>(),
        "equity" => curve.iter().map(|point| point.equity).collect::<Vec<_>>(),
    )
    .into_diagnostic()?;
    write_csv(&mut frame, path)
}

/// Every completed round trip, one row per trade.
fn write_trades_csv(report: &BacktestReport, path: &Path) -> Result<()> {
    let trades = &report.trades;
    let mut frame = df!(
        "entry_date" => trades.iter().map(|t| t.entry_date).collect::<Vec<_>>(),
        "exit_date" => trades.iter().map(|t| t.exit_date).collect::<Vec<_>>(),
        "ticker" => trades.iter().map(|t| t.ticker.clone()).collect::<Vec<_>>(),
        "shares" => trades.iter().map(|t| t.shares).collect::<Vec<_>>(),
        "entry_price" => trades.iter().map(|t| t.entry_price).collect::<Vec<_>>(),
        "exit_price" => trades.iter().map(|t| t.exit_price).collect::<Vec<_>>(),
        "cost" => trades.iter().map(|t| t.cost).collect::<Vec<_>>(),
        "proceeds" => trades.iter().map(|t| t.proceeds).collect::<Vec<_>>(),
        "pnl" => trades.iter().map(|t| t.pnl).collect::<Vec<_>>(),
        "return_pct" => trades.iter().map(|t| t.return_pct).collect::<Vec<_>>(),
        "exit_reason" => trades.iter().map(|t| t.exit_reason.to_string()).collect::<Vec<_>>(),
    )
    .into_diagnostic()?;
    write_csv(&mut frame, path)
}

/// Every executed buy and sell, in order.
fn write_actions_csv(report: &BacktestReport, path: &Path) -> Result<()> {
    let events = &report.events;
    let mut frame = df!(
        "date" => events.iter().map(|e| e.date).collect::<Vec<_>>(),
        "ticker" => events.iter().map(|e| e.ticker.clone()).collect::<Vec<_>>(),
        "side" => events.iter().map(|e| e.side.to_string()).collect::<Vec<_>>(),
        "reason" => events.iter().map(|e| e.reason.to_string()).collect::<Vec<_>>(),
        "price" => events.iter().map(|e| e.price).collect::<Vec<_>>(),
        "shares" => events.iter().map(|e| e.shares).collect::<Vec<_>>(),
        "amount" => events.iter().map(|e| e.amount).collect::<Vec<_>>(),
        "cash_after" => events.iter().map(|e| e.cash_after).collect::<Vec<_>>(),
    )
    .into_diagnostic()?;
    write_csv(&mut frame, path)
}
