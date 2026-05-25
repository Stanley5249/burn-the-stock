use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::{InferenceStep, TrainOutput, TrainStep};

use crate::dataset::{FEATURE_COUNT, WINDOW_SIZE};

pub const NUM_CLASSES: usize = 3;

// --- Batch type ---

/// One batch of windows fed into the model.
#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape [batch, `WINDOW_SIZE` * `FEATURE_COUNT`].
    pub features: Tensor<B, 2>,
    /// Shape [batch] — class index 0/1/2.
    pub targets: Tensor<B, 1, Int>,
}

// --- Model ---

/// Burn model for stock action classification.
///
/// Input:  [batch, `WINDOW_SIZE` * `FEATURE_COUNT`]
/// Output: [batch, `NUM_CLASSES`] (logits)
#[derive(Module, Debug)]
pub struct StockModel<B: Backend> {
    // TODO: define layers (linear, norm, activation, dropout, …)
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> StockModel<B> {
    /// Build the model from `config`.
    pub fn new(_config: &StockModelConfig, _device: &B::Device) -> Self {
        todo!("initialize layers")
    }

    /// Forward pass.
    ///
    /// `input` has shape [batch, `WINDOW_SIZE` * `FEATURE_COUNT`].
    /// Returns logits with shape [batch, `NUM_CLASSES`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let _ = (WINDOW_SIZE, FEATURE_COUNT);
        let _ = input;
        todo!("forward pass")
    }

    fn forward_classification(
        &self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(features);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        ClassificationOutput::new(loss, logits, targets)
    }
}

// --- TrainStep (autodiff backend) ---

impl<B: AutodiffBackend> TrainStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// --- InferenceStep (inner backend for validation) ---

impl<B: Backend> InferenceStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.targets)
    }
}

// --- Config ---

/// Hyperparameters for `StockModel`.
#[derive(Config, Debug)]
pub struct StockModelConfig {
    // TODO: add hidden size, dropout, num layers, …
}

impl StockModelConfig {
    /// Instantiate the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StockModel<B> {
        StockModel::new(self, device)
    }
}
