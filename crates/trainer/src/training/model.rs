use burn::backend::flex::{Flex, FlexDevice};
use burn::nn::loss::{HuberLoss, HuberLossConfig, Reduction};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use stock_model::model::StockModel;

use crate::training::batcher::StockBatch;
use crate::training::metric::StockEvalInput;
use crate::training::pipeline::TrainingConfig;

/// Training wrapper around [`StockModel`], adding the Huber loss and the train/eval
/// steps, so the model crate stays free of training machinery.
#[derive(Module, Debug)]
pub struct StockRegressor<B: Backend> {
    model: StockModel<B>,
    loss: HuberLoss,
}

impl<B: Backend> StockRegressor<B> {
    pub fn new(config: &TrainingConfig, device: &B::Device) -> Self {
        let model = config.model.init::<B>(device);
        let loss = HuberLossConfig::new(config.huber_delta).init();

        Self { model, loss }
    }

    /// The architecture without the loss, to save for inference.
    pub fn into_model(self) -> StockModel<B> {
        self.model
    }

    #[tracing::instrument(skip_all)]
    fn forward_regression(&self, batch: &StockBatch<B>) -> StockOutput<B> {
        let [batch_size, _, _] = batch.technical.dims();
        let prediction = self
            .model
            .forward(batch.technical.clone())
            .reshape([batch_size]);

        let loss = self
            .loss
            .forward(prediction.clone(), batch.target.clone(), Reduction::Mean);

        StockOutput {
            loss,
            output: prediction,
            targets: batch.target.clone(),
        }
    }
}

/// Model output carrying the regression fields. Shapes: `output` `[batch]` scores and
/// `targets` `[batch]`.
pub struct StockOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 1>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for StockOutput<B> {
    // Metrics run on the synced CPU backend.
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

impl<B: Backend> Adaptor<StockEvalInput<B>> for StockOutput<B> {
    fn adapt(&self) -> StockEvalInput<B> {
        StockEvalInput {
            predictions: self.output.clone(),
            targets: self.targets.clone(),
        }
    }
}

impl<B: AutodiffBackend> TrainStep for StockRegressor<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    #[tracing::instrument(skip_all)]
    fn step(&self, batch: StockBatch<B>) -> TrainOutput<StockOutput<B>> {
        let item = self.forward_regression(&batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StockRegressor<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> StockOutput<B> {
        self.forward_regression(&batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::AdamWConfig;
    use burn::tensor::ElementConversion;
    use stock_model::features::NUM_FEATURES;
    use stock_model::model::StockModelConfig;

    #[test]
    fn forward_regression_scores_each_row() {
        let device = FlexDevice;
        let config = TrainingConfig::new(StockModelConfig::new(), AdamWConfig::new());
        let model = StockRegressor::<Flex>::new(&config, &device);

        let technical = Tensor::<Flex, 3>::zeros([4, config.steps, NUM_FEATURES], &device);
        let target = Tensor::<Flex, 1>::zeros([4], &device);

        let output = model.forward_regression(&StockBatch { technical, target });

        assert_eq!(output.output.dims(), [4]);
        assert!(output.loss.into_scalar().elem::<f64>().is_finite());
    }
}
