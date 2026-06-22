use std::path::Path;

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::{PartialDataset, SamplerDataset, SamplerDatasetOptions};
use burn::data::dataset::{Dataset, InMemDataset};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::{
    Learner, LearnerSummary, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
};
use miette::{IntoDiagnostic, Result, WrapErr, bail};

use crate::label::load_labeled;
use crate::training::batcher::StockBatcher;
use crate::training::metric::CorrelationMetric;
use crate::training::model::StockRegressor;
use chrono::NaiveDate;
use fastrand::Rng;
use stock_model::data::{StockItem, TickerFrames};
use stock_model::model::StockModelConfig;

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Forward horizon in trading days the MFE target looks ahead.
    #[config(default = 25)]
    pub label_horizon: usize,
    /// Huber loss delta, the residual size where it switches from squared to linear.
    #[config(default = 1.0)]
    pub huber_delta: f32,
    /// Full passes over the training data; `passes * windows / epoch_size` epochs run.
    #[config(default = 3)]
    pub passes: usize,
    /// Window length fed to the GRU.
    #[config(default = 30)]
    pub steps: usize,
    /// Tickers per batch.
    #[config(default = "stock_model::inference::DEFAULT_BATCH_SIZE")]
    pub batch_size: usize,
    /// Batches per epoch, setting the validation cadence. Each epoch samples
    /// `epoch_size * batch_size` windows without replacement.
    #[config(default = 200)]
    pub epoch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    /// Resolved dataset provenance, written once the data loads. `None` until `train`
    /// fills it; the trader's partial load ignores it.
    pub split: Option<DataSplit>,
}

/// Resolved dataset provenance for a run, written after the data loads so the backtest
/// reloads the exact split and universe. The train window is `[data_start, valid_from)`
/// and the valid window is `[valid_from, valid_to]`.
#[derive(Config, Debug)]
pub struct DataSplit {
    /// First dated row in the run's data, the train window start.
    pub data_start: NaiveDate,
    /// Train/valid boundary; valid holds windows whose last bar is on or after it.
    pub valid_from: NaiveDate,
    /// Last dated row at training time, the valid window end.
    pub valid_to: NaiveDate,
    /// Ticker universe the run trained on, in frame order.
    pub tickers: Vec<String>,
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

    // Load, label, and partition the store once; the batchers produce tensors later, so
    // the data is not copied per backend.
    let store = load_labeled(data_path, config.label_horizon)?;

    // Trim before the split so both sides shrink together.
    let store = match max_tickers {
        Some(count) => sample_tickers(store, count, config.seed),
        None => store,
    };

    // One global cutoff `valid_days` before the latest date, aligned across tickers.
    let (data_start, max_date) = store
        .date_bounds()
        .into_diagnostic()?
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - chrono::Duration::days(valid_days);

    // Resolve the split to concrete dates and pin the universe, so the backtest reloads
    // this exact window regardless of newer bars.
    let tickers = store.tickers().into_diagnostic()?;
    let config = config
        .clone()
        .with_split(Some(DataSplit::new(data_start, cutoff, max_date, tickers)));

    let (train_store, valid_store) = store.train_valid_split(cutoff, config.steps)?;

    config
        .save(artifact_dir.join("config.json"))
        .expect("config should be saved successfully");

    // `SamplerDataset` caps each epoch to a fixed window budget and walks a reshuffled
    // permutation across the run.
    let train_windows = InMemDataset::new(train_store.enumerate_windows(config.steps));
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

    let dataloader_train = DataLoaderBuilder::new(StockBatcher::new(config.steps, &train_store))
        .batch_size(config.batch_size)
        .set_device(device.clone())
        .build(train_sampler);

    // Valid pipeline on the inner backend. A full sweep dwarfs a training run, so when
    // asked, cap a once-shuffled pool, stable across epochs.
    let valid_batcher = StockBatcher::new(config.steps, &valid_store);
    let valid_builder = || {
        DataLoaderBuilder::new(valid_batcher.clone())
            .batch_size(config.batch_size)
            .set_device(device.clone())
    };

    let dataloader_valid = match valid_batches {
        Some(batches) => {
            let dataset = subsample_windows(&valid_store, config.steps, config.seed);
            let cap = (batches * config.batch_size).min(dataset.len());
            valid_builder().build(PartialDataset::new(dataset, 0, cap))
        }
        None => valid_builder().build(InMemDataset::new(
            valid_store.enumerate_windows(config.steps),
        )),
    };

    let model = StockRegressor::new(&config, device);

    let optimizer = config.optimizer.init::<B, StockRegressor<B>>();

    let learner = Learner::new(model, optimizer, config.learning_rate);

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((CorrelationMetric::new(), LossMetric::new()))
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
        StockRegressor::<B::InnerBackend>::new(&config, device)
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

/// Randomly keep `count` tickers, reproducible by `seed`, for overfit diagnostics. A
/// no-op when `count` is at least the ticker count.
fn sample_tickers(store: TickerFrames, count: usize, seed: u64) -> TickerFrames {
    if count >= store.frames.len() {
        return store;
    }
    let mut rng = Rng::with_seed(seed);
    let indices = rng.choose_multiple(0..store.frames.len(), count);
    let frames = indices
        .into_iter()
        .map(|index| store.frames[index].clone())
        .collect();
    TickerFrames { frames }
}

/// Every window of `store`, shuffled once by `seed`, so a `PartialDataset` cap yields a
/// validation subsample drawn evenly across tickers and dates.
fn subsample_windows(store: &TickerFrames, steps: usize, seed: u64) -> InMemDataset<StockItem> {
    let mut windows = store.enumerate_windows(steps);
    Rng::with_seed(seed).shuffle(&mut windows);
    InMemDataset::new(windows)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::synthetic;

    #[test]
    fn subsample_windows_is_reproducible() {
        let store = synthetic(4, 20);
        let collect = |dataset: &InMemDataset<StockItem>| {
            (0..dataset.len())
                .map(|i| dataset.get(i))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            collect(&subsample_windows(&store, 4, 99)),
            collect(&subsample_windows(&store, 4, 99))
        );
    }
}
