#![allow(dead_code)]

mod dataloader;
mod label;
mod model;
mod training;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use clap::Parser;
use miette::Result;

use crate::model::StockModelConfig;
use crate::training::{RunOptions, TrainingConfig, train};

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

    /// Training epochs.
    #[arg(long, default_value_t = 10)]
    num_epochs: usize,

    /// Batches per training epoch. Omit for one full pass over every window;
    /// set it to cap each epoch and make validation run more often.
    #[arg(long)]
    epoch_size: Option<usize>,

    /// Tickers per batch.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Window length fed to the GRU.
    #[arg(long, default_value_t = 30)]
    steps: usize,

    /// Validate on a fixed-seed subsample of this many batches, drawn across all
    /// tickers and dates and stable across epochs. Omit to sweep every window,
    /// which dwarfs a short training run.
    #[arg(long)]
    valid_batches: Option<usize>,

    /// Length in days of the recent window used for validation. Everything
    /// before it trains, so a smaller value leaves more data for training.
    #[arg(long, default_value_t = 180)]
    valid_days: i64,

    /// Keep only this many tickers, drawn at random by the seed. For overfit
    /// diagnostics on a small subset; omit to train on every ticker.
    #[arg(long)]
    max_tickers: Option<usize>,

    /// Learning rate for the optimizer.
    #[arg(long, default_value_t = 1.0e-3)]
    learning_rate: f64,

    /// L2 weight decay for the optimizer; 0 disables it.
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,

    /// Dropout probability in the fusion head.
    #[arg(long, default_value_t = 0.2)]
    dropout: f64,

    /// GRU hidden size. A smaller value trains faster.
    #[arg(long, default_value_t = 64)]
    d_hidden: usize,

    /// Stop early if validation loss does not improve for this many epochs.
    /// Omit to disable early stopping.
    #[arg(long)]
    patience: Option<usize>,
}

type Backend = Autodiff<Wgpu>;

fn main() -> Result<()> {
    // Do not install a global tracing subscriber here. `SupervisedTraining`
    // installs its own file logger (into the artifact dir); a subscriber set
    // first makes that install fail and dumps burn's internal logs onto the
    // console alongside the metrics renderer.
    let args = Args::parse();

    let device = WgpuDevice::default();

    let optimizer = if args.weight_decay > 0.0 {
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(args.weight_decay)))
    } else {
        AdamConfig::new()
    };

    // `n_industries` is a placeholder; `train` fills it from the loaded data.
    let model = StockModelConfig::new(0)
        .with_d_hidden(args.d_hidden)
        .with_dropout(args.dropout);

    let config = TrainingConfig::new(model, optimizer)
        .with_num_epochs(args.num_epochs)
        .with_epoch_size(args.epoch_size)
        .with_batch_size(args.batch_size)
        .with_steps(args.steps)
        .with_learning_rate(args.learning_rate);

    let options = RunOptions {
        valid_batches: args.valid_batches,
        max_tickers: args.max_tickers,
        valid_days: args.valid_days,
        patience: args.patience,
    };

    train::<Backend>(
        device,
        &args.data,
        &args.tickers,
        &args.artifact_dir,
        config,
        options,
    )
}
