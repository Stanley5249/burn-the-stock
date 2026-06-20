//! Live trading loop: load recent prices, predict an action per ticker, place orders.
//! The data load reads the parquet tail and order placement is still a stub, so this
//! runs the real inference path end to end without the network.

use std::path::{Path, PathBuf};

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use chrono::Duration;
use clap::Parser;
use miette::{Context, IntoDiagnostic, Result};
use polars::prelude::*;
use stock_model::class::Action;
use stock_model::data::TickerFrames;
use stock_model::features::DATE;
use stock_model::inference::{InferenceConfig, score};

type Backend = Wgpu;

#[derive(Parser, Debug)]
#[command(about = "Predict today's actions and place the implied orders")]
struct Args {
    /// Directory holding a training run's `config.json` and `model.mpk`.
    #[arg(long, default_value = "artifacts/latest")]
    artifact_dir: PathBuf,

    /// OHLCV parquet to score; only its recent tail is read.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    data: PathBuf,
}

/// Scan only the recent tail of the OHLCV parquet, keeping the last `lookback`
/// calendar days. That over-reaches the `steps` trading days the caller needs,
/// since trading days are sparser, and `latest_windows` trims each ticker down
/// afterward. The per-date z-score is unaffected since each retained date still
/// holds the full universe.
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

/// Submit the day's per-ticker actions as orders to `sim_stock`.
fn place_orders(decisions: &[(String, Action)]) -> Result<()> {
    todo!("submit orders to sim_stock: {decisions:?}")
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = WgpuDevice::default();

    let config = InferenceConfig::load(args.artifact_dir.join("config.json")).into_diagnostic()?;

    let model = config
        .model
        .init::<Backend>(&device)
        .load_file(
            args.artifact_dir.join("model"),
            &CompactRecorder::new(),
            &device,
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

    // Same inference path as the backtest: copy each ticker's tail out of the per-ticker
    // feature series, then one forward over every ticker.
    let windows = store.latest_windows(config.steps).into_diagnostic()?;
    let features = store.feature_series().into_diagnostic()?;
    let predictions = score::<Backend>(&model, &features, &windows, config.steps, &device);

    let decisions: Vec<(String, Action)> = windows
        .into_iter()
        .zip(predictions)
        .map(|(window, prediction)| (window.ticker, prediction.action))
        .collect();

    place_orders(&decisions)
}
