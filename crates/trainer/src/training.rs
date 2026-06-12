use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{PartialDataset, SamplerDataset, SamplerDatasetOptions};
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{AccuracyMetric, ClassReduction, FBetaScoreMetric, LossMetric};
use burn::train::{Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining};
use miette::{IntoDiagnostic, Result, bail};

use crate::batcher::StockBatcher;
use crate::dataset::WindowDataset;
use crate::metric::{ExpectedValueMetric, PrecisionClassMetric};
use crate::model::{StockModel, StockModelConfig};
use crate::store::TickerStore;

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
    /// Swing-reversal magnitude for the oracle labels, as a fraction of price.
    /// Mirrors [`crate::label::LABEL_THRESHOLD`].
    #[config(default = 0.03)]
    pub label_threshold: f32,
    /// Round-trip transaction cost charged to a Buy in the EV metric, as a
    /// fraction. The `sim_stock` default is 0.1425% brokerage twice plus 0.3%
    /// sell tax.
    #[config(default = 0.005_85)]
    pub fee: f32,
    /// Symmetric clip on the per-row reward fed to the EV metric, taming
    /// penny-stock and inverse-ETF moves that would otherwise dominate.
    #[config(default = 1.0)]
    pub reward_clip: f32,
    /// Number of full passes over the training data. With a fixed `epoch_size`
    /// this sets how many epochs run: `passes * windows / epoch_size`.
    #[config(default = 1)]
    pub passes: usize,
    /// Window length fed to the GRU.
    #[config(default = 20)]
    pub steps: usize,
    /// Tickers per batch, which is the batch size.
    #[config(default = 64)]
    pub batch_size: usize,
    /// Batches per epoch, which sets the validation cadence. Each epoch samples
    /// `epoch_size * batch_size` windows without replacement, so on a large
    /// dataset validation runs long before a full pass completes.
    #[config(default = 1000)]
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
/// `tickers_path` - `tickers.parquet` with the per-ticker industry metadata.
/// `artifact_dir` - directory where checkpoints, config, and the final model land.
/// `options`      - runtime knobs, see [`RunOptions`].
///
/// # Errors
///
/// Returns an error if the data cannot be loaded or the artifacts cannot be saved.
pub fn train<B: AutodiffBackend>(
    device: &B::Device,
    data_path: &str,
    tickers_path: &str,
    artifact_dir: &str,
    mut config: TrainingConfig,
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
    let store = TickerStore::load(data_path, config.label_threshold)
        .into_diagnostic()?
        .attach_industries(tickers_path)
        .into_diagnostic()?;

    // Trim to a random ticker subset before the split so both sides shrink
    // together. Done after attach_industries, whose per-ticker encoding follows.
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

    let n_industries = train_store.n_industries();

    // The industry encoding is only known once the data is loaded, so size the
    // categorical branch from it before building the model and saving the config.
    config.model.n_industries = n_industries;
    config
        .save(format!("{artifact_dir}/config.json"))
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

    // One structured record of everything that shaped this run, so a later read of
    // `experiment.log` can tie the metrics back to the flags and derived counts
    // that produced them.
    crate::logging::log_run_config(&config, &options, n_industries, total_windows, num_epochs);

    let train_sampler = SamplerDataset::new(
        train_windows,
        SamplerDatasetOptions::from(epoch_items)
            .without_replacement()
            .with_seed(config.seed),
    );

    let dataloader_train = DataLoaderBuilder::new(StockBatcher::<B>::new(
        config.steps,
        n_industries,
        &train_store,
        device,
    ))
    .batch_size(config.batch_size)
    .set_device(device.clone())
    .build(train_sampler);

    // Build the valid pipeline on the inner backend, as burn expects. A full
    // sweep over every window dwarfs a training run, so when asked, cap a
    // once-shuffled pool: representative across tickers and dates, and stable
    // from one epoch to the next.
    let valid_batcher =
        StockBatcher::<B::InnerBackend>::new(config.steps, n_industries, &valid_store, device);
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

    let model = config.model.init::<B>(device);

    let optimizer = config.optimizer.init::<B, StockModel<B>>();

    let learner = Learner::new(model, optimizer, config.learning_rate);

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((
            AccuracyMetric::new(),
            FBetaScoreMetric::multiclass(1.0, 1, ClassReduction::Macro),
            ExpectedValueMetric::new(config.fee, config.reward_clip),
            PrecisionClassMetric::new(2, "Buy"),
            PrecisionClassMetric::new(0, "Sell"),
            LossMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        // The logger installed at startup owns `experiment.log`, so stop burn from
        // installing its own subscriber over it.
        .with_application_logger(None)
        .num_epochs(num_epochs)
        .summary();

    // Halt once validation loss stops improving, so a run does not sail past its
    // optimum and overfit. Monitors the same valid Loss metric registered above.
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

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .into_diagnostic()?;

    Ok(())
}
