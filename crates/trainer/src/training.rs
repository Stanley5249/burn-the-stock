use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use miette::Result;

use crate::model::StockModelConfig;

/// Top-level training configuration.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: StockModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
}

/// Run the full training loop.
///
/// `data_path`    - path to the aggregated `stocks.parquet` file.
/// `artifact_dir` - directory where checkpoints and config are saved.
///
/// # Errors
///
/// Returns an error if the dataloader cannot be built.
#[allow(clippy::needless_pass_by_value)]
pub fn train<B: AutodiffBackend>(
    device: B::Device,
    _data_path: &str,
    artifact_dir: &str,
    config: TrainingConfig,
) -> Result<()> {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("config should be saved successfully");

    B::seed(&device, config.seed);

    // TODO: build StockDataLoader for train/valid splits.
    // TODO: call config.model.init::<B>(&device).
    // TODO: configure SupervisedTraining with AccuracyMetric + LossMetric.
    // TODO: launch Learner::new(model, config.optimizer.init(), config.learning_rate).
    // TODO: save result.model to artifact_dir.
    todo!("implement training loop")
}
