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
use stock_model::data::{TickerFrames, stack_windows};
use stock_model::features::DATE;
use stock_model::inference::{InferenceConfig, predict};

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

/// Scan only the recent tail of the OHLCV parquet, enough rows that each ticker keeps
/// `steps` standardized days after the per-ticker log-return drops its first bar. The
/// per-date z-score is unaffected since each retained date still holds the full
/// universe.
fn recent_frame(path: &Path, steps: usize) -> PolarsResult<LazyFrame> {
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

    // Trading days are sparser than calendar days, so over-reach the lookback and let
    // `latest_windows` trim each ticker to exactly `steps`.
    let lookback = i64::try_from(steps * 2 + 10).expect("steps far smaller than i64");
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

    let frame = recent_frame(&args.data, config.steps).into_diagnostic()?;
    let store = TickerFrames::from_lazy(frame).into_diagnostic()?;

    // Same windowing path as the backtest: slice each ticker's tail out of the
    // resident feature tensors rather than copying the FEATURE column back to the host.
    let windows = store.latest_windows(config.steps).into_diagnostic()?;
    let features = store
        .feature_tensors::<Backend>(&device)
        .into_diagnostic()?;
    let pairs: Vec<(u32, u32)> = windows.iter().map(|w| (w.ticker_index, w.start)).collect();
    let technical = stack_windows(&features, &pairs, config.steps, &device);

    // One forward over every ticker; the model is tiny and there is one window each.
    let decisions: Vec<(String, Action)> = windows
        .into_iter()
        .zip(predict(&model, technical))
        .map(|(window, prediction)| (window.ticker, prediction.action))
        .collect();

    place_orders(&decisions)
}
