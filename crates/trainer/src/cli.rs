use std::path::PathBuf;

use burn::optim::AdamWConfig;
use clap::{Parser, Subcommand};
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
    /// Backtest a trained model over the held-out window and report realized
    /// return, Sharpe, and hit rate.
    Eval(EvalArgs),
    /// Simulate a stateful long-only portfolio over the held-out window under the
    /// `sim_stock` platform rules, reporting cumulative return, win rate, and more.
    Backtest(BacktestArgs),
}

/// Every hyperparameter is an `Option` so an omitted flag falls through to the
/// default baked into the `Config` struct it feeds, keeping one source of truth for
/// defaults.
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Aggregated OHLCV history.
    #[arg(
        long,
        default_value = "data/yfinance/stocks.parquet",
        help_heading = "Data"
    )]
    pub data: PathBuf,

    /// Directory for this run's checkpoints, config, and final model. Required, so
    /// each run gets its own directory rather than overwriting a shared default.
    /// After the run, the train command points `<parent>/latest` at this directory
    /// so predict and other tools can open the newest run without naming it.
    #[arg(long, help_heading = "Data")]
    pub artifact_dir: PathBuf,

    /// GRU hidden size, the temporal summary width. A smaller value trains faster.
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

    /// Number of full passes over the training data. Validation runs every
    /// epoch, so smaller epochs validate more often within a pass.
    #[arg(long, help_heading = "Training schedule")]
    pub passes: Option<usize>,

    /// Training batches per epoch, which sets the validation cadence. Each epoch
    /// samples this many batches without replacement. The validation-side
    /// counterpart is `--valid-batches`.
    #[arg(long, help_heading = "Training schedule")]
    pub batches_per_epoch: Option<usize>,

    /// Tickers per batch, the batch dimension shared by training and validation.
    #[arg(long, help_heading = "Training schedule")]
    pub batch_size: Option<usize>,

    /// GRU input window length in trading days. This is the sequence length fed to
    /// the model, not the number of optimizer updates.
    #[arg(long, help_heading = "Training schedule")]
    pub window_steps: Option<usize>,

    /// Random seed for the train/valid split, ticker and window sampling, and
    /// parameter initialization.
    #[arg(long, help_heading = "Training schedule")]
    pub seed: Option<u64>,

    /// Take-profit barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub take_profit: Option<f32>,

    /// Stop-loss barrier for the triple-barrier labels, as a fraction of price.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub stop_loss: Option<f32>,

    /// Vertical-barrier horizon in trading days for the triple-barrier labels.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub label_horizon: Option<usize>,

    /// Round-trip transaction cost the Sharpe metric charges each position, as a
    /// fraction. Taiwan is 0.1425% brokerage on each of the buy and sell legs plus
    /// 0.3% sell tax, so 0.585% round trip.
    #[arg(long, help_heading = "Labeling (triple-barrier)")]
    pub fee: Option<f32>,

    /// Validation batches per epoch: a fixed-seed subsample of this many batches,
    /// drawn across all tickers and dates and stable across epochs. Omit to sweep
    /// every window, which dwarfs a short training run. The training-side
    /// counterpart is `--batches-per-epoch`.
    #[arg(long, help_heading = "Validation")]
    pub valid_batches: Option<usize>,

    /// Length in days of the recent window used for validation. Everything
    /// before it trains, so a smaller value leaves more data for training.
    #[arg(long, default_value_t = 180, help_heading = "Validation")]
    pub valid_days: i64,

    /// Keep only this many tickers, drawn at random by the seed. For overfit
    /// diagnostics on a small subset; omit to train on every ticker.
    #[arg(long, help_heading = "Validation")]
    pub max_tickers: Option<usize>,

    /// Stop early if validation loss does not improve for this many epochs.
    /// Omit to disable early stopping.
    #[arg(long, help_heading = "Validation")]
    pub patience: Option<usize>,
}

impl TrainArgs {
    /// Fold the model, optimizer, and schedule flags into a [`TrainingConfig`]. Each
    /// omitted flag leaves its config default untouched, so the defaults live in one
    /// place rather than being restated here.
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

        config
    }

    /// Gather the runtime knobs that shape one run without touching the model or
    /// optimizer config.
    pub fn run_options(&self) -> RunOptions {
        RunOptions {
            valid_batches: self.valid_batches,
            max_tickers: self.max_tickers,
            valid_days: self.valid_days,
            patience: self.patience,
        }
    }
}

/// Held-out backtest of a trained model: replay it over the validation window and
/// report realized return, Sharpe, and hit rate. Reads a parquet snapshot and places
/// no orders, so live trading stays in the `trader` bin. The barrier and fee knobs
/// come from the run's saved `config.json`.
#[derive(Parser, Debug)]
pub struct EvalArgs {
    /// Directory holding a training run's `config.json` and `model.mpk`. Defaults to
    /// the `latest` link the train command refreshes after each run.
    #[arg(long, default_value = "artifacts/latest")]
    pub artifact_dir: PathBuf,

    /// OHLCV history to backtest over. It must hold the full ticker universe so the
    /// per-date cross-sectional features match training.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    pub data: PathBuf,

    /// Length in days of the recent window to evaluate. Match train's `--valid-days`
    /// so the held-out split lines up with the one the model never fit.
    #[arg(long, default_value_t = 180)]
    pub valid_days: i64,

    /// Count a window as a taken trade only when its long position exceeds this, so
    /// weak signals stay flat. The position is `clamp(P(Buy) - P(Sell), 0)`.
    #[arg(long, default_value_t = 0.0)]
    pub min_position: f32,
}

/// Which intraday prices the simulated orders fill at.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum FillArg {
    /// Optimistic best case: buy at the day's low, sell at the day's high.
    LowHigh,
    /// Pessimistic comparison: buy and sell at the day's open.
    Open,
}

/// Stateful long-only portfolio backtest over the held-out window, under the
/// `sim_stock` platform rules (100M capital, ten equal-weight slots, whole lots, buy
/// low / sell high, sell-side tax). Barriers and the split come from the saved
/// `config.json`; the buy gate, fill model, and window are flags here.
#[derive(Parser, Debug)]
pub struct BacktestArgs {
    /// Directory holding a training run's `config.json` and `model.mpk`. Defaults to
    /// the `latest` link the train command refreshes after each run.
    #[arg(long, default_value = "artifacts/latest")]
    pub artifact_dir: PathBuf,

    /// OHLCV history to backtest over. It must hold the full ticker universe so the
    /// per-date cross-sectional features match training.
    #[arg(long, default_value = "data/yfinance/stocks.parquet")]
    pub data: PathBuf,

    /// Length in days of the recent window to backtest. Match train's `--valid-days`
    /// so the held-out split lines up with the one the model never fit.
    #[arg(long, default_value_t = 180)]
    pub valid_days: i64,

    /// Minimum net-bullish score `clamp(P(Buy) - P(Sell), 0)` to buy a stock, so weak
    /// signals stay in cash.
    #[arg(long, default_value_t = 0.2)]
    pub threshold: f32,

    /// Which prices fills happen at: `low-high` is the optimistic best case, `open`
    /// the pessimistic comparison.
    #[arg(long, value_enum, default_value_t = FillArg::LowHigh)]
    pub fill: FillArg,

    /// Equity-curve CSV path. Defaults to `<artifact_dir>/backtest-equity.csv`.
    #[arg(long)]
    pub out: Option<PathBuf>,
}
