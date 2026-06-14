mod batcher;
mod cli;
mod dataset;
mod label;
mod link;
mod logging;
mod metric;
mod model;
mod report;
mod store;
mod training;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use miette::{IntoDiagnostic, Result};
use stock_model::inference::Predictor;

use crate::cli::{Cli, Command, PredictArgs, TrainArgs};
use crate::store::TickerStore;
use crate::training::train;

type TrainBackend = Autodiff<Wgpu>;
type InferenceBackend = Wgpu;

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => run_train(&args),
        Command::Predict(args) => run_predict(&args),
    }
}

fn run_train(args: &TrainArgs) -> Result<()> {
    // The tracing subscriber is installed inside `train`, once the artifact dir is
    // known, so there is no logger setup here.
    let device = WgpuDevice::default();
    let config = args.training_config();
    let options = args.run_options();

    train::<TrainBackend>(&device, &args.data, &args.artifact_dir, &config, options)?;

    link::refresh_latest(&args.artifact_dir);

    Ok(())
}

fn run_predict(args: &PredictArgs) -> Result<()> {
    let device = WgpuDevice::default();

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    // Offline inference: read the most recent `steps` bars per ticker from the
    // parquet snapshot and run them through the exact training feature pipeline.
    // Live fetching and order placement live in the `trader` bin.
    let windows =
        TickerStore::load_inference_windows(&args.data, predictor.steps()).into_diagnostic()?;

    let predictions = predictor.predict(&windows);

    report::print(&predictions, args.min_position);

    Ok(())
}
