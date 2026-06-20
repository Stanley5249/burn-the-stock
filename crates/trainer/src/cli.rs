use std::path::PathBuf;

use burn::optim::AdamWConfig;
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use stock_model::class::NUM_CLASSES;
use stock_model::model::StockModelConfig;

use crate::portfolio::Fill;
use crate::training::{RunOptions, TrainingConfig};

#[derive(Parser, Debug)]
#[command(about = "Stock action classifier: train a model or backtest a run")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "clap subcommand variants must hold their Args type by value"
)]
pub enum Command {
    /// Train the model and write the run's artifacts.
    Train(TrainArgs),
    /// Simulate a long-only portfolio over the held-out window under `sim_stock`
    /// rules, reporting cumulative return, win rate, and more.
    Backtest(BacktestArgs),
}

/// Every hyperparameter is an `Option` so an omitted flag falls through to its
/// `Config` default, keeping one source of truth.
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Aggregated OHLCV history.
    #[arg(
        long,
        default_value = "data/yfinance/stocks.parquet",
        help_heading = "Data"
    )]
    pub data: PathBuf,

    /// Output dir for this run's checkpoints, config, and model; `<parent>/latest`
    /// links here.
    #[arg(long, help_heading = "Data")]
    pub artifact_dir: PathBuf,

    /// GRU hidden size; smaller trains faster.
    #[arg(long, help_heading = "Model")]
    pub d_hidden: Option<usize>,

    /// MLP head hidden width.
    #[arg(long, help_heading = "Model")]
    pub d_head: Option<usize>,

    /// Dropout probability in the head.
    #[arg(long, help_heading = "Model")]
    pub dropout: Option<f64>,

    /// Learning rate for the optimizer.
    #[arg(long, help_heading = "Optimizer")]
    pub learning_rate: Option<f64>,

    /// `AdamW` L2 weight decay.
    #[arg(long, help_heading = "Optimizer")]
    pub weight_decay: Option<f32>,

    /// `AdamW` first-moment decay (`beta_1`).
    #[arg(long, help_heading = "Optimizer")]
    pub beta_1: Option<f32>,

    /// `AdamW` second-moment decay (`beta_2`).
    #[arg(long, help_heading = "Optimizer")]
    pub beta_2: Option<f32>,

    /// `AdamW` epsilon (denominator stability).
    #[arg(long, help_heading = "Optimizer")]
    pub epsilon: Option<f32>,

    /// Number of full passes over the training data.
    #[arg(long, help_heading = "Training schedule")]
    pub passes: Option<usize>,

    /// Training batches per epoch (validation cadence); counterpart of `--valid-batches`.
    #[arg(long, help_heading = "Training schedule")]
    pub batches_per_epoch: Option<usize>,

    /// Tickers per batch, the batch dimension shared by training and validation.
    #[arg(long, help_heading = "Training schedule")]
    pub batch_size: Option<usize>,

    /// Input window length in trading days.
    #[arg(long, help_heading = "Training schedule")]
    pub window_steps: Option<usize>,

    /// Seed for split, sampling, and init.
    #[arg(long, help_heading = "Training schedule")]
    pub seed: Option<u64>,

    /// Sell/Hold/Buy cross-entropy weights; default [2, 1, 2].
    #[arg(
        long,
        num_args = 3,
        value_names = ["SELL", "HOLD", "BUY"],
        help_heading = "Training schedule"
    )]
    pub class_weights: Option<Vec<f32>>,

    /// Weight on the soft expected-value loss term added to cross-entropy; 0 disables it.
    #[arg(long, help_heading = "Training schedule")]
    pub ev_weight: Option<f32>,

    /// Take-profit barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling")]
    pub take_profit: Option<f32>,

    /// Stop-loss barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling")]
    pub stop_loss: Option<f32>,

    /// Vertical-barrier horizon in trading days for the triple-barrier labels.
    #[arg(long, help_heading = "Labeling")]
    pub label_horizon: Option<usize>,

    /// Round-trip cost charged per position by the Sharpe metric (Taiwan: 0.585%).
    #[arg(long, help_heading = "Labeling")]
    pub fee: Option<f32>,

    /// Validation batches per epoch; 0 sweeps every window. Counterpart of
    /// `--batches-per-epoch`.
    #[arg(long, default_value_t = 200, help_heading = "Validation")]
    pub valid_batches: usize,

    /// Length in days of the recent validation window; everything before it trains.
    #[arg(long, default_value_t = 180, help_heading = "Validation")]
    pub valid_days: i64,

    /// Keep only this many tickers, drawn at random by the seed, for overfit
    /// diagnostics. Omit to train on every ticker.
    #[arg(long, help_heading = "Validation")]
    pub max_tickers: Option<usize>,

    /// Stop early after this many epochs without validation-loss improvement; 0
    /// disables.
    #[arg(long, default_value_t = 5, help_heading = "Validation")]
    pub patience: usize,
}

macro_rules! apply {
    ($target:ident, $flag:expr, $with:ident) => {
        if let Some(value) = $flag {
            $target = $target.$with(value);
        }
    };
}

impl TrainArgs {
    /// Fold the flags into a [`TrainingConfig`], leaving each omitted flag's
    /// config default untouched.
    pub fn training_config(&self) -> TrainingConfig {
        let mut model = StockModelConfig::new();
        apply!(model, self.d_hidden, with_d_hidden);
        apply!(model, self.d_head, with_d_head);
        apply!(model, self.dropout, with_dropout);

        let mut optimizer = AdamWConfig::new();
        apply!(optimizer, self.weight_decay, with_weight_decay);
        apply!(optimizer, self.beta_1, with_beta_1);
        apply!(optimizer, self.beta_2, with_beta_2);
        apply!(optimizer, self.epsilon, with_epsilon);

        let mut config = TrainingConfig::new(model, optimizer);
        apply!(config, self.passes, with_passes);
        apply!(config, self.batches_per_epoch, with_epoch_size);
        apply!(config, self.batch_size, with_batch_size);
        apply!(config, self.window_steps, with_steps);
        apply!(config, self.seed, with_seed);
        apply!(config, self.learning_rate, with_learning_rate);
        apply!(config, self.take_profit, with_take_profit);
        apply!(config, self.stop_loss, with_stop_loss);
        apply!(config, self.label_horizon, with_label_horizon);
        apply!(config, self.fee, with_fee);
        apply!(config, self.ev_weight, with_ev_weight);
        if let Some(class_weights) = &self.class_weights {
            // clap's num_args = 3 guarantees exactly NUM_CLASSES values.
            let weights: [f32; NUM_CLASSES] = class_weights
                .clone()
                .try_into()
                .expect("--class-weights takes exactly NUM_CLASSES values");
            config = config.with_class_weights(weights);
        }

        config
    }

    /// Gather the runtime knobs that shape one run.
    pub fn run_options(&self) -> RunOptions {
        RunOptions {
            // 0 means sweep every window / disable early stopping, mapped to `None`.
            valid_batches: (self.valid_batches != 0).then_some(self.valid_batches),
            max_tickers: self.max_tickers,
            valid_days: self.valid_days,
            patience: (self.patience != 0).then_some(self.patience),
        }
    }
}

/// Long-only portfolio backtest over the held-out window under `sim_stock` rules.
/// Barriers and the split come from the saved `config.json`; the buy gate, fill
/// model, and window are flags here.
#[derive(Parser, Debug)]
pub struct BacktestArgs {
    /// Directory holding a run's `config.json` and `model.mpk`. Defaults to the
    /// `latest` link.
    #[arg(long, default_value = "artifacts/latest")]
    pub artifact_dir: PathBuf,

    /// OHLCV history to backtest over. Must hold the full ticker universe so the
    /// cross-sectional features match training.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    pub data: PathBuf,

    /// Train/valid boundary to score from; the lookback reaches before it, so tradeable
    /// days are roughly the trading days after it. Defaults to the run's stored
    /// `valid_from`.
    #[arg(long)]
    pub valid_from: Option<NaiveDate>,

    /// Minimum expected edge `clamp(P(Buy)*tp - P(Sell)*sl, 0)` to buy, so weak signals
    /// stay in cash.
    #[arg(long, default_value_t = 0.0)]
    pub threshold: f32,

    /// Take-profit exit, a fraction of the entry price. Defaults to the run's config.
    #[arg(long)]
    pub take_profit: Option<f32>,

    /// Stop-loss exit, a fraction of the entry price. Defaults to the run's config.
    #[arg(long)]
    pub stop_loss: Option<f32>,

    /// Trading days to hold before a time exit. Defaults to the run's `label_horizon`.
    #[arg(long)]
    pub max_hold: Option<usize>,

    /// Most stocks held at once; each buy targets an equal `equity / slots`.
    #[arg(long, default_value_t = 10)]
    pub max_holdings: usize,

    /// Which prices fills happen at: `low-high` optimistic, `open` pessimistic.
    #[arg(long, value_enum, default_value_t = Fill::LowHigh)]
    pub fill: Fill,

    /// Equity-curve CSV path. Defaults to `<artifact_dir>/backtest-equity.csv`.
    #[arg(long)]
    pub out: Option<PathBuf>,
}
