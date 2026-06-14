use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use chrono::{Duration, NaiveDate};
use miette::{IntoDiagnostic, Result};
use stock_model::inference::{Action, Predictor};

use crate::cli::{BacktestArgs, FillArg};
use crate::portfolio::{self, BacktestConfig, DayBar, Fill, STARTING_CASH, TradingDay};
use crate::store::TickerStore;
use crate::training::TrainingConfig;

type InferenceBackend = Wgpu;

/// Maximum stocks held at once; each new buy targets an equal tenth of equity.
const MAX_HOLDINGS: usize = 10;

/// Run the stateful portfolio backtest over the held-out split and report the
/// platform's performance metrics, plus a CSV of the daily account value.
pub fn run(args: &BacktestArgs) -> Result<()> {
    let device = WgpuDevice::default();

    // The full training config carries the barrier knobs the Predictor's inference
    // subset omits, so reload it to rebuild the same labels and split.
    let config = TrainingConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    let store = TickerStore::load(
        &args.data,
        config.take_profit,
        config.stop_loss,
        config.label_horizon,
    )
    .into_diagnostic()?;

    // Reproduce the training split and keep only the held-out tail, the same window
    // `eval` scores, so the backtest trades data the model never fit.
    let max_date = store
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - Duration::days(args.valid_days);
    let (_, valid) = store
        .train_valid_split(cutoff, config.steps)
        .into_diagnostic()?;

    // Score every window, then index each signal by (ticker, signal date).
    let windows = valid.backtest_windows(config.steps);
    let predictions = predictor.predict(&windows);

    let mut signals: HashMap<String, HashMap<NaiveDate, (f32, Action)>> = HashMap::new();
    for prediction in &predictions {
        signals
            .entry(prediction.ticker.clone())
            .or_default()
            .insert(prediction.date, (prediction.position, prediction.action));
    }

    let fill = match args.fill {
        FillArg::LowHigh => Fill::LowHigh,
        FillArg::Open => Fill::Open,
    };

    let days = build_days(&valid, &signals);

    let backtest_config = BacktestConfig {
        threshold: args.threshold,
        fill,
        max_holdings: MAX_HOLDINGS,
        starting_cash: STARTING_CASH,
    };
    let report = portfolio::run(&days, &backtest_config);

    portfolio::render(&report);

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.artifact_dir.join("backtest-equity.csv"));
    write_equity_csv(&report, &out_path)?;
    println!("\nWrote equity curve to {}", out_path.display());

    Ok(())
}

/// Assemble the time-ordered day stream the engine consumes. Each ticker's signal
/// from the window ending the previous trading day drives the order filled today
/// (the no-look-ahead lag), priced from today's bar. Bars with a missing price are
/// skipped.
fn build_days(
    valid: &TickerStore,
    signals: &HashMap<String, HashMap<NaiveDate, (f32, Action)>>,
) -> Vec<TradingDay> {
    let mut by_date: BTreeMap<NaiveDate, HashMap<String, DayBar>> = BTreeMap::new();

    for quotes in valid.quotes() {
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

/// Write the daily account value as a two-column CSV for plotting.
fn write_equity_csv(report: &portfolio::BacktestReport, path: &Path) -> Result<()> {
    use std::fmt::Write as _;

    let mut csv = String::from("date,equity\n");
    for point in &report.equity_curve {
        let _ = writeln!(csv, "{},{:.2}", point.date, point.equity);
    }
    std::fs::write(path, csv).into_diagnostic()
}
