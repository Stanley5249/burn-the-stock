#![allow(dead_code)]

mod dataloader;
mod label;
mod model;
mod training;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use clap::Parser;
use miette::Result;

use crate::model::StockModelConfig;
use crate::training::{TrainingConfig, train};

#[derive(Parser, Debug)]
#[command(about = "Train the stock action classifier")]
struct Args {
    /// Aggregated OHLCV history.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    data: String,

    /// Per-ticker industry metadata from the `tickers` prefetch bin.
    #[arg(long, default_value = "data/yfinance/tickers.parquet")]
    tickers: String,

    /// Directory for checkpoints, config, and the final model.
    #[arg(long, default_value = "artifacts")]
    artifact_dir: String,
}

type Backend = Autodiff<Wgpu>;

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let args = Args::parse();

    let device = WgpuDevice::default();

    // `n_industries` is a placeholder; `train` fills it from the loaded data.
    let config = TrainingConfig::new(StockModelConfig::new(0), AdamConfig::new());

    train::<Backend>(
        device,
        &args.data,
        &args.tickers,
        &args.artifact_dir,
        config,
    )
}
