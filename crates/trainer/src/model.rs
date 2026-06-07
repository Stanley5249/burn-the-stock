use crate::dataloader::StockBatch;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::ClassificationOutput;
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use std::marker::PhantomData;

pub const NUM_CLASSES: usize = 3;

/// Burn model for stock action classification.
///
/// Input:  technical `[batch, steps, 5]`, ticker `[batch, ticker_features]`
/// Output: `[batch, NUM_CLASSES]` (logits)
#[derive(Module, Debug)]
pub struct StockModel<B: Backend> {
    // TODO: define layers (linear, norm, activation, dropout, …)
    phantom: PhantomData<B>,
}

impl<B: Backend> StockModel<B> {
    pub fn new(_config: &StockModelConfig, _device: &B::Device) -> Self {
        todo!("initialize layers")
    }

    /// Returns logits with shape `[batch, NUM_CLASSES]`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn forward(&self, technical: Tensor<B, 3>, ticker: Tensor<B, 2>) -> Tensor<B, 2> {
        let _ = (technical, ticker);
        todo!("forward pass")
    }

    fn forward_classification(&self, batch: &StockBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.technical.clone(), batch.ticker.clone());
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.label.clone());
        ClassificationOutput::new(loss, logits, batch.label.clone())
    }
}

impl<B: AutodiffBackend> TrainStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(&batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(&batch)
    }
}

/// Hyperparameters for `StockModel`.
#[derive(Config, Debug)]
pub struct StockModelConfig {
    // TODO: add hidden size, dropout, num layers, …
}

impl StockModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StockModel<B> {
        StockModel::new(self, device)
    }
}
