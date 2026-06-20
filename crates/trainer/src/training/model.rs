use burn::backend::flex::{Flex, FlexDevice};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, ConfusionStatsInput, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use stock_model::class::NUM_CLASSES;
use stock_model::model::{StockModel, StockModelConfig};

use crate::training::batcher::StockBatch;
use crate::training::metric::StockEvalInput;

/// Training wrapper around [`StockModel`], adding the loss and the train/eval steps,
/// so the model crate stays free of training machinery.
#[derive(Module, Debug)]
pub struct StockClassifier<B: Backend> {
    model: StockModel<B>,
    loss: CrossEntropyLoss<B>,
}

impl<B: Backend> StockClassifier<B> {
    /// `class_weights` are the Sell, Hold, Buy cross-entropy weights, upweighting the
    /// rare actionable classes.
    pub fn new(
        config: &StockModelConfig,
        class_weights: [f32; NUM_CLASSES],
        device: &B::Device,
    ) -> Self {
        let model = config.init::<B>(device);
        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(class_weights.to_vec()))
            .init(device);

        Self { model, loss }
    }

    /// The architecture without the loss, to save for inference.
    pub fn into_model(self) -> StockModel<B> {
        self.model
    }

    #[tracing::instrument(skip_all)]
    fn forward_classification(&self, batch: &StockBatch<B>) -> StockOutput<B> {
        let logits = self.model.forward(batch.technical.clone());

        let loss = self.loss.forward(logits.clone(), batch.label.clone());

        StockOutput {
            loss,
            output: logits,
            targets: batch.label.clone(),
        }
    }
}

/// Model output carrying the classification fields. A local stand-in for burn's
/// `ClassificationOutput`, which the orphan rule blocks us from adapting to the
/// trade-aware metrics. Shapes: `output` `[batch, NUM_CLASSES]` and `targets`
/// `[batch]`.
pub struct StockOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ItemLazy for StockOutput<B> {
    // Metrics run on the synced CPU backend, matching burn's ClassificationOutput.
    type ItemSync = StockOutput<Flex>;

    #[tracing::instrument(skip_all)]
    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &FlexDevice;

        StockOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for StockOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for StockOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
        ConfusionStatsInput::new(
            self.output.clone(),
            self.targets.clone().one_hot(NUM_CLASSES).bool(),
        )
    }
}

impl<B: Backend> Adaptor<StockEvalInput<B>> for StockOutput<B> {
    fn adapt(&self) -> StockEvalInput<B> {
        StockEvalInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: AutodiffBackend> TrainStep for StockClassifier<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    #[tracing::instrument(skip_all)]
    fn step(&self, batch: StockBatch<B>) -> TrainOutput<StockOutput<B>> {
        let item = self.forward_classification(&batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StockClassifier<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> StockOutput<B> {
        self.forward_classification(&batch)
    }
}
