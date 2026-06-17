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
use miette::{IntoDiagnostic, Result, WrapErr, bail};

use crate::data::store::TickerStore;
use crate::training::batcher::StockBatcher;
use crate::training::dataset::WindowDataset;
use crate::training::metric::{ExpectedValueMetric, PrecisionClassMetric};
use crate::training::model::StockClassifier;
use stock_model::class::Action;
use stock_model::model::StockModelConfig;

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Take-profit barrier, a positive fraction of the entry close.
    #[config(default = 0.09)]
    pub take_profit: f32,
    /// Stop-loss barrier, a positive fraction of the entry close.
    #[config(default = 0.09)]
    pub stop_loss: f32,
    /// Vertical-barrier horizon in trading days.
    #[config(default = 25)]
    pub label_horizon: usize,
    /// Round-trip transaction cost per position. 0.1425% per leg plus 0.3% sell tax
    /// is 0.585%.
    #[config(default = 0.005_85)]
    pub fee: f32,
    /// Full passes over the training data; `passes * windows / epoch_size` epochs run.
    #[config(default = 3)]
    pub passes: usize,
    /// Window length fed to the GRU.
    #[config(default = 30)]
    pub steps: usize,
    /// Tickers per batch.
    #[config(default = 64)]
    pub batch_size: usize,
    /// Batches per epoch, setting the validation cadence. Each epoch samples
    /// `epoch_size * batch_size` windows without replacement.
    #[config(default = 200)]
    pub epoch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
}

/// Runtime knobs that shape one run without touching the model or optimizer config.
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// Fixed-seed validation subsample size in batches; `None` sweeps every window.
    pub valid_batches: Option<usize>,
    /// Random ticker-subset cap for overfit diagnostics; `None` uses every ticker.
    pub max_tickers: Option<usize>,
    /// Length in days of the recent validation window; everything before it trains.
    pub valid_days: i64,
    /// Epochs without validation-loss improvement before stopping; `None` disables.
    pub patience: Option<usize>,
}

/// Run the full training loop.
///
/// # Errors
/// If the data cannot be loaded or the artifacts cannot be saved.
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

    // Refuse an existing dir: a prior run's checkpoints and logs corrupt the
    // best-epoch selection.
    if artifact_dir.exists() {
        bail!(
            "artifact dir {} already exists; pick a fresh --artifact-dir",
            artifact_dir.display()
        );
    }
    std::fs::create_dir_all(artifact_dir).into_diagnostic()?;

    crate::logging::redirect_to_file(artifact_dir)?;

    B::seed(device, config.seed);

    // Load the backend-free store once; the batchers produce tensors later, so the
    // data is not copied per backend.
    let store = TickerStore::load(
        data_path,
        config.take_profit,
        config.stop_loss,
        config.label_horizon,
    )
    .into_diagnostic()?;

    // Trim before the split so both sides shrink together.
    let store = match max_tickers {
        Some(count) => store.sample_tickers(count, config.seed),
        None => store,
    };

    // One global cutoff `valid_days` before the latest date, aligned across tickers.
    let max_date = store
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - chrono::Duration::days(valid_days);

    let (train_store, valid_store) = store
        .train_valid_split(cutoff, config.steps)
        .into_diagnostic()?;

    // Class balance per split, to tune the barriers toward an even mix.
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

    // `SamplerDataset` caps each epoch to a fixed window budget and walks a reshuffled
    // permutation across the run.
    let train_windows = WindowDataset::new(&train_store, config.steps);
    let total_windows = train_windows.len();
    if total_windows == 0 {
        bail!("no training windows; check steps and the train/valid split");
    }

    let epoch_items = (config.epoch_size * config.batch_size).min(total_windows);
    let num_epochs = (config.passes * total_windows).div_ceil(epoch_items).max(1);

    // Log the derived counts, which `config.json` does not hold.
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

    // Valid pipeline on the inner backend. A full sweep dwarfs a training run, so when
    // asked, cap a once-shuffled pool, stable across epochs.
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
            ExpectedValueMetric::new(config.take_profit, config.stop_loss, config.fee),
            PrecisionClassMetric::new(Action::Buy),
            LossMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        // Our startup logger owns `experiment.log`; stop burn installing its own.
        .with_application_logger(None)
        .num_epochs(num_epochs)
        .summary();

    // Halt once validation loss stops improving, so the run does not overfit.
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

    // `launch` returns the final-epoch model, `patience` epochs past the valid-loss
    // optimum. Export the best checkpoint instead, falling back to the final model.
    // Save only the inner model so the artifact loads straight into a `StockModel`.
    let best_model = if let Some(epoch) = best_valid_loss_epoch(artifact_dir) {
        tracing::info!(best_epoch = epoch, "exporting best-checkpoint model");
        let checkpoint = artifact_dir
            .join("checkpoint")
            .join(format!("model-{epoch}"));
        StockClassifier::<B::InnerBackend>::new(&config.model, device)
            .load_file(&checkpoint, &CompactRecorder::new(), device)
            .into_diagnostic()
            .wrap_err_with(|| format!("loading best checkpoint {}", checkpoint.display()))?
            .into_model()
    } else {
        tracing::warn!("no valid Loss summary; exporting final-epoch model");
        result.model.into_model()
    };

    let model_path = artifact_dir.join("model");
    best_model
        .save_file(&model_path, &CompactRecorder::new())
        .into_diagnostic()
        .wrap_err_with(|| format!("saving model to {}", model_path.display()))?;

    Ok(())
}

/// Epoch (1-based) of the lowest-valid-loss checkpoint still on disk. burn has no
/// best-checkpoint accessor and its checkpointer prunes all but the last few plus its
/// own best, so we read every epoch's loss from `LearnerSummary` and filter to the
/// surviving files before taking the min.
fn best_valid_loss_epoch(artifact_dir: &Path) -> Option<usize> {
    let summary = LearnerSummary::new(artifact_dir, &["Loss"]).ok()?;
    let loss = summary.metrics.valid.iter().find(|m| m.name == "Loss")?;
    let checkpoint_dir = artifact_dir.join("checkpoint");
    loss.entries
        .iter()
        .filter(|entry| {
            checkpoint_dir
                .join(format!("model-{}.mpk", entry.step))
                .exists()
        })
        .min_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|entry| entry.step)
}
