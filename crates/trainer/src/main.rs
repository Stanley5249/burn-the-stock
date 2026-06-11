#![allow(dead_code)]

mod batcher;
mod dataset;
mod label;
mod metric;
mod model;
mod store;
mod training;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamWConfig;
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

    /// Number of full passes over the training data. Validation runs every
    /// epoch, so smaller epochs validate more often within a pass.
    #[arg(long)]
    passes: Option<usize>,

    /// Batches per training epoch, which sets the validation cadence. Each epoch
    /// samples this many batches without replacement.
    #[arg(long)]
    epoch_size: Option<usize>,

    /// Tickers per batch.
    #[arg(long)]
    batch_size: Option<usize>,

    /// Window length fed to the GRU.
    #[arg(long)]
    steps: Option<usize>,

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
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Swing-reversal magnitude for the oracle labels, as a fraction of price.
    #[arg(long)]
    label_threshold: Option<f32>,

    /// Round-trip transaction cost charged to a Buy in the EV metric, as a
    /// fraction of price.
    #[arg(long)]
    fee: Option<f32>,

    /// Symmetric clip on the per-row reward fed to the EV metric.
    #[arg(long)]
    reward_clip: Option<f32>,

    /// L2 weight decay for the optimizer.
    #[arg(long)]
    weight_decay: Option<f32>,

    /// Dropout probability in the fusion head.
    #[arg(long)]
    dropout: Option<f64>,

    /// GRU hidden size. A smaller value trains faster.
    #[arg(long)]
    d_hidden: Option<usize>,

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

    // `n_industries` is a placeholder; `train` fills it from the loaded data.
    let mut model = StockModelConfig::new(0);
    if let Some(d_hidden) = args.d_hidden {
        model = model.with_d_hidden(d_hidden);
    }
    if let Some(dropout) = args.dropout {
        model = model.with_dropout(dropout);
    }

    let mut optimizer_config = AdamWConfig::new();

    if let Some(weight_decay) = args.weight_decay {
        optimizer_config = optimizer_config.with_weight_decay(weight_decay);
    }

    let mut training_config = TrainingConfig::new(model, optimizer_config);
    if let Some(passes) = args.passes {
        training_config = training_config.with_passes(passes);
    }
    if let Some(epoch_size) = args.epoch_size {
        training_config = training_config.with_epoch_size(epoch_size);
    }
    if let Some(batch_size) = args.batch_size {
        training_config = training_config.with_batch_size(batch_size);
    }
    if let Some(steps) = args.steps {
        training_config = training_config.with_steps(steps);
    }
    if let Some(learning_rate) = args.learning_rate {
        training_config = training_config.with_learning_rate(learning_rate);
    }
    if let Some(label_threshold) = args.label_threshold {
        training_config = training_config.with_label_threshold(label_threshold);
    }
    if let Some(fee) = args.fee {
        training_config = training_config.with_fee(fee);
    }
    if let Some(reward_clip) = args.reward_clip {
        training_config = training_config.with_reward_clip(reward_clip);
    }

    let options = RunOptions {
        valid_batches: args.valid_batches,
        max_tickers: args.max_tickers,
        valid_days: args.valid_days,
        patience: args.patience,
    };

    train::<Backend>(
        &device,
        &args.data,
        &args.tickers,
        &args.artifact_dir,
        training_config,
        options,
    )
}
