use burn::backend::flex::{Flex, FlexDevice};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, ConfusionStatsInput, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use stock_model::class::{Action, BUY, NUM_CLASSES};
use stock_model::model::StockModel;

use crate::training::batcher::StockBatch;
use crate::training::metric::StockEvalInput;
use crate::training::pipeline::TrainingConfig;

/// Training wrapper around [`StockModel`], adding the loss and the train/eval steps,
/// so the model crate stays free of training machinery.
#[derive(Module, Debug)]
pub struct StockClassifier<B: Backend> {
    model: StockModel<B>,
    loss: CrossEntropyLoss<B>,
    // Barrier payoffs and round-trip cost for the soft expected-value term. Primitive
    // fields, so burn skips them in the record and the gradient.
    take_profit: f32,
    stop_loss: f32,
    fee: f32,
    ev_weight: f32,
}

impl<B: Backend> StockClassifier<B> {
    pub fn new(config: &TrainingConfig, device: &B::Device) -> Self {
        let model = config.model.init::<B>(device);
        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(config.class_weights.to_vec()))
            .init(device);

        Self {
            model,
            loss,
            take_profit: config.take_profit,
            stop_loss: config.stop_loss,
            fee: config.fee,
            ev_weight: config.ev_weight,
        }
    }

    /// The architecture without the loss, to save for inference.
    pub fn into_model(self) -> StockModel<B> {
        self.model
    }

    #[tracing::instrument(skip_all)]
    fn forward_classification(&self, batch: &StockBatch<B>) -> StockOutput<B> {
        let logits = self.model.forward(batch.technical.clone());

        let mut loss = self.loss.forward(logits.clone(), batch.label.clone());

        if self.ev_weight > 0.0 {
            let ev = negative_expected_value(
                logits.clone(),
                batch.label.clone(),
                self.take_profit,
                self.stop_loss,
                self.fee,
            );
            loss = loss + ev.mul_scalar(self.ev_weight);
        }

        StockOutput {
            loss,
            output: logits,
            targets: batch.label.clone(),
        }
    }
}

/// Negative soft expected value of the Buy policy, the differentiable surrogate for
/// `ExpectedValueMetric`. It swaps that metric's hard `argmax == Buy` indicator for the
/// softmax Buy probability, so each row contributes `buy_probability * reward`, where the
/// reward is the barrier payoff net of fee keyed on the true label. Minimizing it raises
/// expected value. Returned negated so it adds to the cross-entropy loss.
fn negative_expected_value<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    take_profit: f32,
    stop_loss: f32,
    fee: f32,
) -> Tensor<B, 1> {
    let buy_probability = softmax(logits, 1).narrow(1, BUY, 1).squeeze_dim(1);

    let is_buy = targets
        .clone()
        .equal_elem(i64::from(Action::Buy.class()))
        .float();
    let is_sell = targets.equal_elem(i64::from(Action::Sell.class())).float();
    let reward = (is_buy.mul_scalar(take_profit) - is_sell.mul_scalar(stop_loss)).sub_scalar(fee);

    (reward * buy_probability).mean().neg()
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;

    #[test]
    fn negative_ev_matches_reward_under_certain_buy() {
        let device = FlexDevice;
        // A saturated Buy logit drives buy_probability to ~1, so the loss approaches the
        // negated reward for a true Buy: -(take_profit - fee).
        let logits = Tensor::<Flex, 2>::from_data([[0.0, 0.0, 10.0]], &device);
        let targets = Tensor::<Flex, 1, Int>::from_data([2], &device);

        let loss = negative_expected_value(logits, targets, 0.09, 0.04, 0.01);
        let value = loss.into_scalar().elem::<f64>();

        assert!((value + 0.08).abs() < 1e-3);
    }
}
