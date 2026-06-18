use std::path::PathBuf;

use burn::optim::AdamWConfig;
use clap::{Parser, Subcommand};
use stock_model::class::NUM_CLASSES;
use stock_model::model::StockModelConfig;

use crate::training::{RunOptions, TrainingConfig};

#[derive(Parser, Debug)]
#[command(about = "Stock action classifier: train a model or predict today's actions")]
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

    /// Directory for this run's checkpoints, config, and model. Required, so each run
    /// is its own directory; `<parent>/latest` is pointed here afterward.
    #[arg(long, help_heading = "Data")]
    pub artifact_dir: PathBuf,

    /// GRU hidden size; smaller trains faster.
    #[arg(long, help_heading = "Model")]
    pub d_hidden: Option<usize>,

    /// Hidden width of the MLP head that maps the summary to action logits.
    #[arg(long, help_heading = "Model")]
    pub d_head: Option<usize>,

    /// Dropout probability in the head.
    #[arg(long, help_heading = "Model")]
    pub dropout: Option<f64>,

    /// Learning rate for the optimizer.
    #[arg(long, help_heading = "Optimizer")]
    pub learning_rate: Option<f64>,

    /// L2 weight decay for the `AdamW` optimizer.
    #[arg(long, help_heading = "Optimizer")]
    pub weight_decay: Option<f32>,

    /// `AdamW` first-moment decay (`beta_1`).
    #[arg(long, help_heading = "Optimizer")]
    pub beta_1: Option<f32>,

    /// `AdamW` second-moment decay (`beta_2`).
    #[arg(long, help_heading = "Optimizer")]
    pub beta_2: Option<f32>,

    /// `AdamW` numerical-stability term added to the denominator (epsilon).
    #[arg(long, help_heading = "Optimizer")]
    pub epsilon: Option<f32>,

    /// Number of full passes over the training data.
    #[arg(long, help_heading = "Training schedule")]
    pub passes: Option<usize>,

    /// Training batches per epoch, setting the validation cadence. Sampled without
    /// replacement; the valid-side counterpart is `--valid-batches`.
    #[arg(long, help_heading = "Training schedule")]
    pub batches_per_epoch: Option<usize>,

    /// Tickers per batch, the batch dimension shared by training and validation.
    #[arg(long, help_heading = "Training schedule")]
    pub batch_size: Option<usize>,

    /// GRU input window length in trading days (the sequence length).
    #[arg(long, help_heading = "Training schedule")]
    pub window_steps: Option<usize>,

    /// Random seed for the split, sampling, and parameter initialization.
    #[arg(long, help_heading = "Training schedule")]
    pub seed: Option<u64>,

    /// Sell Hold Buy cross-entropy weights, upweighting the rare actionable classes.
    /// Three floats; defaults to the config's [2, 1, 2].
    #[arg(
        long,
        num_args = 3,
        value_names = ["SELL", "HOLD", "BUY"],
        help_heading = "Training schedule"
    )]
    pub class_weights: Option<Vec<f32>>,

    /// Take-profit barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub take_profit: Option<f32>,

    /// Stop-loss barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub stop_loss: Option<f32>,

    /// Vertical-barrier horizon in trading days for the triple-barrier labels.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub label_horizon: Option<usize>,

    /// Round-trip transaction cost the Sharpe metric charges per position. Taiwan is
    /// 0.1425% per leg plus 0.3% sell tax, so 0.585% round trip.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub fee: Option<f32>,

    /// Validation batches per epoch, a fixed-seed subsample stable across epochs. Set
    /// 0 to sweep every window. Valid-side counterpart of `--batches-per-epoch`.
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

impl TrainArgs {
    /// Fold the flags into a [`TrainingConfig`], leaving each omitted flag's config
    /// default untouched.
    pub fn training_config(&self) -> TrainingConfig {
        let mut model = StockModelConfig::new();
        if let Some(d_hidden) = self.d_hidden {
            model = model.with_d_hidden(d_hidden);
        }
        if let Some(d_head) = self.d_head {
            model = model.with_d_head(d_head);
        }
        if let Some(dropout) = self.dropout {
            model = model.with_dropout(dropout);
        }

        let mut optimizer = AdamWConfig::new();
        if let Some(weight_decay) = self.weight_decay {
            optimizer = optimizer.with_weight_decay(weight_decay);
        }
        if let Some(beta_1) = self.beta_1 {
            optimizer = optimizer.with_beta_1(beta_1);
        }
        if let Some(beta_2) = self.beta_2 {
            optimizer = optimizer.with_beta_2(beta_2);
        }
        if let Some(epsilon) = self.epsilon {
            optimizer = optimizer.with_epsilon(epsilon);
        }

        let mut config = TrainingConfig::new(model, optimizer);
        if let Some(passes) = self.passes {
            config = config.with_passes(passes);
        }
        if let Some(batches_per_epoch) = self.batches_per_epoch {
            config = config.with_epoch_size(batches_per_epoch);
        }
        if let Some(batch_size) = self.batch_size {
            config = config.with_batch_size(batch_size);
        }
        if let Some(window_steps) = self.window_steps {
            config = config.with_steps(window_steps);
        }
        if let Some(seed) = self.seed {
            config = config.with_seed(seed);
        }
        if let Some(learning_rate) = self.learning_rate {
            config = config.with_learning_rate(learning_rate);
        }
        if let Some(take_profit) = self.take_profit {
            config = config.with_take_profit(take_profit);
        }
        if let Some(stop_loss) = self.stop_loss {
            config = config.with_stop_loss(stop_loss);
        }
        if let Some(label_horizon) = self.label_horizon {
            config = config.with_label_horizon(label_horizon);
        }
        if let Some(fee) = self.fee {
            config = config.with_fee(fee);
        }
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

/// Which intraday prices the simulated orders fill at.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum FillArg {
    /// Optimistic best case: buy at the day's low, sell at the day's high.
    LowHigh,
    /// Pessimistic comparison: buy and sell at the day's open.
    Open,
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

    /// Calendar days of the recent window to backtest. The lookback reaches before the
    /// cutoff, so tradeable days are roughly the trading days within this window. Match
    /// train's `--valid-days`.
    #[arg(long, default_value_t = 180)]
    pub valid_days: i64,

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
    #[arg(long, value_enum, default_value_t = FillArg::LowHigh)]
    pub fill: FillArg,

    /// Equity-curve CSV path. Defaults to `<artifact_dir>/backtest-equity.csv`.
    #[arg(long)]
    pub out: Option<PathBuf>,
}
