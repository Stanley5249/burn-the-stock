use burn::data::dataloader::DataLoader;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{AccuracyMetric, ClassReduction, FBetaScoreMetric, LossMetric};
use burn::train::{Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining};
use miette::{IntoDiagnostic, Result};
use std::sync::Arc;

use crate::dataloader::{StockBatch, StockDataLoader};
use crate::model::{StockModel, StockModelConfig};

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Window length fed to the GRU.
    #[config(default = 30)]
    pub steps: usize,
    /// Tickers per batch, which is the batch size.
    #[config(default = 64)]
    pub batch_size: usize,
    /// Batches per epoch. `None` is one full pass over every window; `Some(k)`
    /// emits `k` reshuffled batches per epoch, which controls how often
    /// validation runs since burn only validates between epochs.
    #[config(default = "None")]
    pub epoch_size: Option<usize>,
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

    B::seed(device, config.seed);

    // Load once on the inner backend, then lift the train split up to the
    // autodiff backend. Validation stays on the inner backend, as burn expects.
    let base = StockDataLoader::<B::InnerBackend>::load(
        data_path,
        config.steps,
        config.batch_size,
        config.epoch_size,
        Some(config.seed),
        device.clone(),
    )
    .into_diagnostic()?
    .attach_industries(tickers_path)
    .into_diagnostic()?;

    // Trim to a random ticker subset before the split so both sides shrink
    // together. Done after attach_industries, whose name-keyed map is unaffected.
    let base = match max_tickers {
        Some(count) => base.sample_tickers(count, config.seed),
        None => base,
    };

    // Anchor the split to the most recent date in the data so the last
    // `valid_days` validate and everything earlier trains. One global cutoff
    // keeps the split aligned across tickers.
    let max_date = base
        .max_date()
        .expect("loaded data should have at least one dated row");
    let cutoff = max_date - chrono::Duration::days(valid_days);

    let (train_inner, valid) = base.train_valid_split(cutoff).into_diagnostic()?;

    let train = train_inner.to_backend::<B>(device.clone());

    // The industry encoding is only known once the data is loaded, so size the
    // categorical branch from it before building the model and saving the config.
    config.model.n_industries = train.n_industries();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("config should be saved successfully");

    let dataloader_train: Arc<dyn DataLoader<B, StockBatch<B>>> = Arc::new(train);

    // A full validation sweep over every window dwarfs a training run, so when
    // asked, replace it with a fixed-seed subsample: representative across all
    // tickers and dates, and stable from one epoch to the next.
    let valid = match valid_batches {
        Some(n) => valid.into_subsample(Some(n), config.seed),
        None => valid,
    };

    let dataloader_valid = Arc::new(valid);

    let model = config.model.init::<B>(device);

    let optimizer = config.optimizer.init::<B, StockModel<B>>();

    let learner = Learner::new(model, optimizer, config.learning_rate);

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metrics((
            AccuracyMetric::new(),
            FBetaScoreMetric::multiclass(1.0, 1, ClassReduction::Macro),
            LossMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
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
