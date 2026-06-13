mod batcher;
mod dataset;
mod label;
mod logging;
mod metric;
mod model;
mod store;
mod training;

use std::path::PathBuf;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamWConfig;
use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};
use stock_model::inference::{Action, Prediction, Predictor};
use stock_model::model::StockModelConfig;

use crate::store::TickerStore;
use crate::training::{RunOptions, TrainingConfig, train};

#[derive(Parser, Debug)]
#[command(about = "Stock action classifier: train a model or predict today's actions")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "clap subcommand variants must hold their Args type by value"
)]
enum Command {
    /// Train the model and write the run's artifacts.
    Train(TrainArgs),
    /// Predict an action per ticker from a trained model and a recent price snapshot.
    Predict(PredictArgs),
}

/// Every hyperparameter is an `Option` so an omitted flag falls through to the
/// default baked into the `Config` struct it feeds, keeping one source of truth for
/// defaults.
#[derive(Parser, Debug)]
struct TrainArgs {
    /// Aggregated OHLCV history.
    #[arg(
        long,
        default_value = "data/yfinance/stocks.parquet",
        help_heading = "Data"
    )]
    data: PathBuf,

    /// Directory for checkpoints, config, and the final model.
    #[arg(long, default_value = "artifacts", help_heading = "Data")]
    artifact_dir: PathBuf,

    /// GRU hidden size, the temporal summary width. A smaller value trains faster.
    #[arg(long, help_heading = "Model")]
    d_hidden: Option<usize>,

    /// Hidden width of the MLP head that maps the summary to action logits.
    #[arg(long, help_heading = "Model")]
    d_head: Option<usize>,

    /// Dropout probability in the head.
    #[arg(long, help_heading = "Model")]
    dropout: Option<f64>,

    /// Learning rate for the optimizer.
    #[arg(long, help_heading = "Optimizer")]
    learning_rate: Option<f64>,

    /// L2 weight decay for the `AdamW` optimizer.
    #[arg(long, help_heading = "Optimizer")]
    weight_decay: Option<f32>,

    /// `AdamW` first-moment decay (`beta_1`).
    #[arg(long, help_heading = "Optimizer")]
    beta_1: Option<f32>,

    /// `AdamW` second-moment decay (`beta_2`).
    #[arg(long, help_heading = "Optimizer")]
    beta_2: Option<f32>,

    /// `AdamW` numerical-stability term added to the denominator (epsilon).
    #[arg(long, help_heading = "Optimizer")]
    epsilon: Option<f32>,

    /// Number of full passes over the training data. Validation runs every
    /// epoch, so smaller epochs validate more often within a pass.
    #[arg(long, help_heading = "Training schedule")]
    passes: Option<usize>,

    /// Training batches per epoch, which sets the validation cadence. Each epoch
    /// samples this many batches without replacement. The validation-side
    /// counterpart is `--valid-batches`.
    #[arg(long, help_heading = "Training schedule")]
    batches_per_epoch: Option<usize>,

    /// Tickers per batch, the batch dimension shared by training and validation.
    #[arg(long, help_heading = "Training schedule")]
    batch_size: Option<usize>,

    /// GRU input window length in trading days. This is the sequence length fed to
    /// the model, not the number of optimizer updates.
    #[arg(long, help_heading = "Training schedule")]
    window_steps: Option<usize>,

    /// Random seed for the train/valid split, ticker and window sampling, and
    /// parameter initialization.
    #[arg(long, help_heading = "Training schedule")]
    seed: Option<u64>,

    /// Take-profit barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    take_profit: Option<f32>,

    /// Stop-loss barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    stop_loss: Option<f32>,

    /// Vertical-barrier horizon in trading days for the triple-barrier labels.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    label_horizon: Option<usize>,

    /// Round-trip transaction cost the Sharpe metric charges each position, as a
    /// fraction. Taiwan is 0.1425% brokerage on each of the buy and sell legs plus
    /// 0.3% sell tax, so 0.585% round trip.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    fee: Option<f32>,

    /// Validation batches per epoch: a fixed-seed subsample of this many batches,
    /// drawn across all tickers and dates and stable across epochs. Omit to sweep
    /// every window, which dwarfs a short training run. The training-side
    /// counterpart is `--batches-per-epoch`.
    #[arg(long, help_heading = "Validation")]
    valid_batches: Option<usize>,

    /// Length in days of the recent window used for validation. Everything
    /// before it trains, so a smaller value leaves more data for training.
    #[arg(long, default_value_t = 180, help_heading = "Validation")]
    valid_days: i64,

    /// Keep only this many tickers, drawn at random by the seed. For overfit
    /// diagnostics on a small subset; omit to train on every ticker.
    #[arg(long, help_heading = "Validation")]
    max_tickers: Option<usize>,

    /// Stop early if validation loss does not improve for this many epochs.
    /// Omit to disable early stopping.
    #[arg(long, help_heading = "Validation")]
    patience: Option<usize>,
}

/// Recent OHLCV snapshot, model artifacts, and the position gate for one
/// prediction run. The data fetch and the order placement are mocked, so this run
/// only reads a parquet and prints the actions a trader would take.
#[derive(Parser, Debug)]
struct PredictArgs {
    /// Directory holding a training run's `config.json` and `model.mpk`.
    #[arg(long, default_value = "artifacts/tbl")]
    artifact_dir: PathBuf,

    /// Recent OHLCV snapshot standing in for today's live feed. It must hold the
    /// full ticker universe so the per-date cross-sectional features match training.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    data: PathBuf,

    /// Only act on predictions whose long position exceeds this threshold, so weak
    /// signals stay flat. The position is `clamp(P(Buy) - P(Sell), 0)`.
    #[arg(long, default_value_t = 0.0)]
    min_position: f32,
}

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
    // known, by `logging::install_experiment_logger`. Burn's own file logger is
    // disabled there so this one owns `experiment.log` and also captures the
    // pre-training data-loading spans.
    let device = WgpuDevice::default();

    let mut model = StockModelConfig::new();
    if let Some(d_hidden) = args.d_hidden {
        model = model.with_d_hidden(d_hidden);
    }
    if let Some(d_head) = args.d_head {
        model = model.with_d_head(d_head);
    }
    if let Some(dropout) = args.dropout {
        model = model.with_dropout(dropout);
    }

    let mut optimizer_config = AdamWConfig::new();
    if let Some(weight_decay) = args.weight_decay {
        optimizer_config = optimizer_config.with_weight_decay(weight_decay);
    }
    if let Some(beta_1) = args.beta_1 {
        optimizer_config = optimizer_config.with_beta_1(beta_1);
    }
    if let Some(beta_2) = args.beta_2 {
        optimizer_config = optimizer_config.with_beta_2(beta_2);
    }
    if let Some(epsilon) = args.epsilon {
        optimizer_config = optimizer_config.with_epsilon(epsilon);
    }

    let mut training_config = TrainingConfig::new(model, optimizer_config);
    if let Some(passes) = args.passes {
        training_config = training_config.with_passes(passes);
    }
    if let Some(batches_per_epoch) = args.batches_per_epoch {
        training_config = training_config.with_epoch_size(batches_per_epoch);
    }
    if let Some(batch_size) = args.batch_size {
        training_config = training_config.with_batch_size(batch_size);
    }
    if let Some(window_steps) = args.window_steps {
        training_config = training_config.with_steps(window_steps);
    }
    if let Some(seed) = args.seed {
        training_config = training_config.with_seed(seed);
    }
    if let Some(learning_rate) = args.learning_rate {
        training_config = training_config.with_learning_rate(learning_rate);
    }
    if let Some(take_profit) = args.take_profit {
        training_config = training_config.with_take_profit(take_profit);
    }
    if let Some(stop_loss) = args.stop_loss {
        training_config = training_config.with_stop_loss(stop_loss);
    }
    if let Some(label_horizon) = args.label_horizon {
        training_config = training_config.with_label_horizon(label_horizon);
    }
    if let Some(fee) = args.fee {
        training_config = training_config.with_fee(fee);
    }

    let options = RunOptions {
        valid_batches: args.valid_batches,
        max_tickers: args.max_tickers,
        valid_days: args.valid_days,
        patience: args.patience,
    };

    train::<TrainBackend>(
        &device,
        &args.data,
        &args.artifact_dir,
        &training_config,
        options,
    )
}

fn run_predict(args: &PredictArgs) -> Result<()> {
    let device = WgpuDevice::default();

    let predictor = Predictor::<InferenceBackend>::load(&args.artifact_dir, device)?;

    // Mock the data feed: a real trader would pull today's OHLCV from the broker.
    // Here we read the most recent `steps` bars per ticker from a parquet snapshot
    // and run them through the exact training feature pipeline.
    let windows =
        TickerStore::load_inference_windows(&args.data, predictor.steps()).into_diagnostic()?;

    let predictions = predictor.predict(&windows);

    report(&predictions, args.min_position);

    Ok(())
}

/// Print the predictions and the orders a trader would place, sorted by how strong
/// the long signal is. The order placement itself is mocked, so this only reports.
fn report(predictions: &[Prediction], min_position: f32) {
    if predictions.is_empty() {
        println!("No tickers had enough history to fill the model's window.");
        return;
    }

    let as_of = predictions.iter().map(|p| p.date).max().expect("non-empty");
    let mut counts = [0usize; 3];
    for prediction in predictions {
        counts[prediction.action as usize] += 1;
    }
    println!(
        "Predictions as of {as_of} ({} tickers): Sell {}  Hold {}  Buy {}",
        predictions.len(),
        counts[Action::Sell as usize],
        counts[Action::Hold as usize],
        counts[Action::Buy as usize],
    );

    let mut orders: Vec<&Prediction> = predictions
        .iter()
        .filter(|p| p.position > min_position)
        .collect();
    orders.sort_by(|left, right| right.position.total_cmp(&left.position));

    println!("\nOrders (position > {min_position:.2}):");
    if orders.is_empty() {
        println!("  none above threshold; staying flat.");
        return;
    }

    println!(
        "  {:<8} {:<6} {:>7} {:>7} {:>7} {:>9}",
        "TICKER", "ACTION", "P(Sell)", "P(Hold)", "P(Buy)", "POSITION"
    );
    for order in &orders {
        let [sell, hold, buy] = order.probabilities;
        println!(
            "  {:<8} {:<6} {sell:>7.3} {hold:>7.3} {buy:>7.3} {:>9.3}",
            order.ticker,
            order.action.as_str(),
            order.position,
        );
    }

    // Mock the actioning: a real trader would size shares and call the sim_stock
    // buy/sell endpoints here.
    println!("\n[mock] would place {} buy order(s).", orders.len());
}
