use std::path::Path;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{PartialDataset, SamplerDataset, SamplerDatasetOptions};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{ClassReduction, FBetaScoreMetric, LossMetric};
use burn::train::{
    Learner, LearnerSummary, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
};
use miette::{IntoDiagnostic, Result, bail};

use crate::data::store::TickerStore;
use crate::training::batcher::StockBatcher;
use crate::training::dataset::WindowDataset;
use crate::training::metric::{PrecisionClassMetric, SharpeMetric};
use crate::training::model::StockClassifier;
use stock_model::model::StockModelConfig;

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Take-profit barrier for the triple-barrier labels, as a positive fraction
    /// of the entry close.
    #[config(default = 0.09)]
    pub take_profit: f32,
    /// Stop-loss barrier for the triple-barrier labels, as a positive fraction of
    /// the entry close.
    #[config(default = 0.09)]
    pub stop_loss: f32,
    /// Vertical-barrier horizon in trading days for the triple-barrier labels.
    #[config(default = 25)]
    pub label_horizon: usize,
    /// Round-trip transaction cost the Sharpe metric charges each position, as a
    /// fraction. Taiwan brokerage is 0.1425% on each of the buy and sell legs, plus
    /// a 0.3% tax on the sell, so the default is 0.1425% * 2 + 0.3% = 0.585%.
    #[config(default = 0.005_85)]
    pub fee: f32,
    /// Number of full passes over the training data. With a fixed `epoch_size`
    /// this sets how many epochs run: `passes * windows / epoch_size`.
    #[config(default = 3)]
    pub passes: usize,
    /// Window length fed to the GRU.
    #[config(default = 30)]
    pub steps: usize,
    /// Tickers per batch, which is the batch size.
    #[config(default = 64)]
    pub batch_size: usize,
    /// Batches per epoch, which sets the validation cadence. Each epoch samples
    /// `epoch_size * batch_size` windows without replacement, so on a large
    /// dataset validation runs long before a full pass completes.
    #[config(default = 200)]
    pub epoch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
}

/// Runtime knobs that shape one run without touching the model or optimizer
/// config, kept separate so a baseline run and a diagnostic run differ only
/// here.
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// Fixed-seed validation subsample size in batches; `None` sweeps every
    /// window.
    pub valid_batches: Option<usize>,
    /// Random ticker-subset cap for overfit diagnostics; `None` uses every
    /// ticker.
    pub max_tickers: Option<usize>,
    /// Length in days of the recent window that validates; everything before it
    /// trains.
    pub valid_days: i64,
    /// Stop early if validation loss does not improve for this many epochs;
    /// `None` disables early stopping.
    pub patience: Option<usize>,
}

/// Run the full training loop.
///
/// `data_path`    - aggregated `stocks.parquet` with the OHLCV history.
/// `artifact_dir` - directory where checkpoints, config, and the final model land.
/// `options`      - runtime knobs, see [`RunOptions`].
///
/// # Errors
///
/// Returns an error if the data cannot be loaded or the artifacts cannot be saved.
#[allow(
    clippy::too_many_lines,
    reason = "linear pipeline reads better unsplit"
)]
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &Path,
    artifact_dir: &Path,
    config: &TrainingConfig,
    options: RunOptions,
) -> Result<()> {
    let RunOptions {
        valid_batches,
        max_tickers,
        valid_days,
        patience,
    } = options;

    std::fs::create_dir_all(artifact_dir).into_diagnostic()?;

    crate::logging::install_experiment_logger(artifact_dir)?;

    B::seed(device, config.seed);

    // Load the backend-free store once. The store carries the train/valid split
    // and tensors are produced later by two batchers, so there is no per-backend
    // copy of the data.
    let store = TickerStore::load(
        data_path,
        config.take_profit,
        config.stop_loss,
        config.label_horizon,
    )
    .into_diagnostic()?;

    // Trim to a random ticker subset before the split so both sides shrink
    // together.
    let store = match max_tickers {
        Some(count) => store.sample_tickers(count, config.seed),
        None => store,
    };

    // Anchor the split to the most recent date in the data so the last
    // `valid_days` validate and everything earlier trains. One global cutoff
    // keeps the split aligned across tickers.
    let max_date = store
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - chrono::Duration::days(valid_days);

    let (train_store, valid_store) = store
        .train_valid_split(cutoff, config.steps)
        .into_diagnostic()?;

    // Surface the triple-barrier class balance per split, indexed Sell 0, Hold 1,
    // Buy 2, so the take-profit and stop-loss knobs can be tuned toward an even
    // Sell/Hold/Buy mix.
    let train_counts = train_store.label_counts();
    let valid_counts = valid_store.label_counts();
    tracing::info!(
        train_sell = train_counts[0],
        train_hold = train_counts[1],
        train_buy = train_counts[2],
        valid_sell = valid_counts[0],
        valid_hold = valid_counts[1],
        valid_buy = valid_counts[2],
        "label balance"
    );

    config
        .save(artifact_dir.join("config.json"))
        .expect("config should be saved successfully");

    // Build the train pipeline. `SamplerDataset` caps each epoch to a fixed
    // window budget independent of the pool size and walks a reshuffled
    // permutation across the run, so `passes` full passes take
    // `passes * windows / epoch_size` epochs.
    let train_windows = WindowDataset::new(&train_store, config.steps);
    let total_windows = train_windows.len();
    if total_windows == 0 {
        bail!("no training windows; check steps and the train/valid split");
    }

    let epoch_items = (config.epoch_size * config.batch_size).min(total_windows);
    let num_epochs = (config.passes * total_windows).div_ceil(epoch_items).max(1);

    // The flags and config live in `config.json`; only the counts derived from them
    // are worth a log line, so a later read ties the schedule back to the data size
    // without restating what the saved config already holds.
    tracing::info!(total_windows, epoch_items, num_epochs, "training schedule");

    let train_sampler = SamplerDataset::new(
        train_windows,
        SamplerDatasetOptions::from(epoch_items)
            .without_replacement()
            .with_seed(config.seed),
    );

    let dataloader_train =
        DataLoaderBuilder::new(StockBatcher::<B>::new(config.steps, &train_store, device))
            .batch_size(config.batch_size)
            .set_device(device.clone())
            .build(train_sampler);

    // Build the valid pipeline on the inner backend, as burn expects. A full
    // sweep over every window dwarfs a training run, so when asked, cap a
    // once-shuffled pool: representative across tickers and dates, and stable
    // from one epoch to the next.
    let valid_batcher = StockBatcher::<B::InnerBackend>::new(config.steps, &valid_store, device);
    let valid_builder = || {
        DataLoaderBuilder::new(valid_batcher.clone())
            .batch_size(config.batch_size)
            .set_device(device.clone())
    };

    let dataloader_valid = match valid_batches {
        Some(batches) => {
            let dataset = WindowDataset::subsample(&valid_store, config.steps, config.seed);
            let cap = (batches * config.batch_size).min(dataset.len());
            valid_builder().build(PartialDataset::new(dataset, 0, cap))
        }
        None => valid_builder().build(WindowDataset::new(&valid_store, config.steps)),
    };

    let model = StockClassifier::new(&config.model, device);

    let optimizer = config.optimizer.init::<B, StockClassifier<B>>();

    let learner = Learner::new(model, optimizer, config.learning_rate);

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((
            FBetaScoreMetric::multiclass(1.0, 1, ClassReduction::Macro),
            SharpeMetric::new(config.fee),
            PrecisionClassMetric::new(2, "Buy"),
            LossMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        // The logger installed at startup owns `experiment.log`, so stop burn from
        // installing its own subscriber over it.
        .with_application_logger(None)
        .num_epochs(num_epochs)
        .summary();

    // Halt once validation loss stops improving, so a run does not sail past its
    // optimum and overfit.
    if let Some(patience) = patience {
        let strategy = MetricEarlyStoppingStrategy::new::<LossMetric<B::InnerBackend>>(
            &LossMetric::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: patience },
        );
        training = training.early_stopping(strategy);
    }

    let result = training.launch(learner);

    // burn's `launch` returns the final-epoch model, which early stopping leaves
    // `patience` epochs past the valid-loss optimum. Recover the best checkpoint by
    // valid loss and export that one, falling back to the final model if the summary
    // is unavailable. Save only the inner model, dropping the loss, so the artifact
    // loads straight into a `StockModel` for inference.
    let best_model = if let Some(epoch) = best_valid_loss_epoch(artifact_dir) {
        tracing::info!(best_epoch = epoch, "exporting best-checkpoint model");
        StockClassifier::<B::InnerBackend>::new(&config.model, device)
            .load_file(
                artifact_dir
                    .join("checkpoint")
                    .join(format!("model-{epoch}")),
                &CompactRecorder::new(),
                device,
            )
            .into_diagnostic()?
            .into_model()
    } else {
        tracing::warn!("no valid Loss summary; exporting final-epoch model");
        result.model.into_model()
    };

    best_model
        .save_file(artifact_dir.join("model"), &CompactRecorder::new())
        .into_diagnostic()?;

    Ok(())
}

/// Epoch (1-based, matching the checkpoint filenames) with the lowest validation
/// loss. burn exposes no built-in best-checkpoint accessor: `launch` only returns
/// the final-epoch model, so we recover the best epoch from `LearnerSummary`, burn's
/// own structured reader over the metric event store it writes during training (the
/// same data its internal `EventStoreClient::find_epoch` uses), not by parsing text.
fn best_valid_loss_epoch(artifact_dir: &Path) -> Option<usize> {
    let summary = LearnerSummary::new(artifact_dir, &["Loss"]).ok()?;
    let loss = summary.metrics.valid.iter().find(|m| m.name == "Loss")?;
    loss.entries
        .iter()
        .min_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|entry| entry.step)
}
