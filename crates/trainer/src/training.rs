use burn::data::dataloader::DataLoader;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};
use chrono::NaiveDate;
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

/// First day of the validation split. Earlier rows train, this day onward validates.
const SPLIT_DATE: (i32, u32, u32) = (2024, 6, 1);

/// Run the full training loop.
///
/// `data_path`    - aggregated `stocks.parquet` with the OHLCV history.
/// `tickers_path` - `tickers.parquet` with the per-ticker industry metadata.
/// `artifact_dir` - directory where checkpoints, config, and the final model land.
/// `valid_batches` - optional cap on the validation sweep, for quick smoke runs.
/// `max_tickers` - optional cap on the ticker universe, drawn at random by the
///   seed, for overfit diagnostics on a small subset.
///
/// # Errors
///
/// Returns an error if the data cannot be loaded or the artifacts cannot be saved.
#[allow(clippy::needless_pass_by_value)]
pub fn train<B: AutodiffBackend>(
    device: B::Device,
    data_path: &str,
    tickers_path: &str,
    artifact_dir: &str,
    mut config: TrainingConfig,
    valid_batches: Option<usize>,
    max_tickers: Option<usize>,
) -> Result<()> {
    std::fs::create_dir_all(artifact_dir).into_diagnostic()?;

    B::seed(&device, config.seed);

    let (year, month, day) = SPLIT_DATE;
    let cutoff = NaiveDate::from_ymd_opt(year, month, day).expect("split date is valid");

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

    let dataloader_valid: Arc<dyn DataLoader<B::InnerBackend, StockBatch<B::InnerBackend>>> =
        Arc::new(valid);

    let model = config.model.init::<B>(&device);
    let optimizer = config.optimizer.init::<B, StockModel<B>>();
    let learner = Learner::new(model, optimizer, config.learning_rate);

    let result = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .into_diagnostic()?;

    Ok(())
}
