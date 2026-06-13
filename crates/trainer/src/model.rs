use crate::batcher::StockBatch;
use crate::metric::StockEvalInput;
use burn::backend::flex::{Flex, FlexDevice};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, ConfusionStatsInput, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};

pub const NUM_CLASSES: usize = 3;

/// Sell, Hold, Buy
pub const CLASS_WEIGHTS: [f32; NUM_CLASSES] = [2.0, 1.0, 2.0];

/// Stationary feature width of the technical input, matching the dataloader's
/// feature column.
const NUM_FEATURES: usize = 5;

/// GRU classifier over the stationary feature window.
///
/// A two-layer GRU summarizes the window into its last hidden state, which a small
/// MLP head turns into the action logits.
///
/// Input:  technical `[batch, steps, 5]`
/// Output: `[batch, NUM_CLASSES]` (logits)
#[derive(Module, Debug)]
pub struct StockModel<B: Backend> {
    gru_1: Gru<B>,
    gru_1_norm: RmsNorm<B>,
    gru_2: Gru<B>,
    gru_2_norm: RmsNorm<B>,
    hidden: Linear<B>,
    head: Linear<B>,
    activation: Gelu,
    dropout: Dropout,
    loss: CrossEntropyLoss<B>,
}

impl<B: Backend> StockModel<B> {
    pub fn new(config: &StockModelConfig, device: &B::Device) -> Self {
        let gru_1 = GruConfig::new(NUM_FEATURES, config.d_hidden, true).init(device);

        let gru_1_norm = RmsNormConfig::new(config.d_hidden).init(device);

        let gru_2 = GruConfig::new(config.d_hidden, config.d_hidden, true).init(device);

        let gru_2_norm = RmsNormConfig::new(config.d_hidden).init(device);

        let hidden = LinearConfig::new(config.d_hidden, config.d_head).init(device);

        let head = LinearConfig::new(config.d_head, NUM_CLASSES).init(device);

        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(CLASS_WEIGHTS.to_vec()))
            .init(device);

        Self {
            gru_1,
            gru_1_norm,
            gru_2,
            gru_2_norm,
            hidden,
            head,
            activation: Gelu::new(),
            dropout: DropoutConfig::new(config.dropout).init(),
            loss,
        }
    }

    /// Returns logits with shape `[batch, NUM_CLASSES]`.
    pub fn forward(&self, technical: Tensor<B, 3>) -> Tensor<B, 2> {
        let temporal_1 = self.gru_1.forward(technical, None);

        let temporal_1 = self.gru_1_norm.forward(temporal_1);

        let temporal_2 = self.gru_2.forward(temporal_1, None);

        // Summarize the window by its last hidden state: [batch, d_hidden].
        let [batch, sequence, d_hidden] = temporal_2.dims();

        let summary = temporal_2
            .slice([0..batch, sequence - 1..sequence, 0..d_hidden])
            .reshape([batch, d_hidden]);
        let summary = self.gru_2_norm.forward(summary);

        let hidden = self.activation.forward(self.hidden.forward(summary));
        let hidden = self.dropout.forward(hidden);

        self.head.forward(hidden)
    }

    fn forward_classification(&self, batch: &StockBatch<B>) -> StockOutput<B> {
        let logits = self.forward(batch.technical.clone());

        let loss = self.loss.forward(logits.clone(), batch.label.clone());

        StockOutput {
            loss,
            output: logits,
            targets: batch.label.clone(),
            reward: batch.reward.clone(),
        }
    }
}

/// Model output carrying the reward alongside the usual classification fields.
///
/// A local stand-in for burn's `ClassificationOutput` so it can adapt to the
/// trade-aware Sharpe and precision metrics as well as the built-in Loss and
/// confusion-based (F-beta) metrics. The orphan rule blocks adding that adaptor to
/// the foreign type.
///
/// Input shapes: `output` `[batch, NUM_CLASSES]`, `targets` and `reward` `[batch]`.
pub struct StockOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
    pub reward: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for StockOutput<B> {
    // Metrics run on the synced CPU backend, matching burn's ClassificationOutput.
    type ItemSync = StockOutput<Flex>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets, reward] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .register(self.reward)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &FlexDevice;

        StockOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
            reward: Tensor::from_data(reward, device),
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
        StockEvalInput::new(
            self.output.clone(),
            self.targets.clone(),
            self.reward.clone(),
        )
    }
}

impl<B: AutodiffBackend> TrainStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> TrainOutput<StockOutput<B>> {
        let item = self.forward_classification(&batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for StockModel<B> {
    type Input = StockBatch<B>;
    type Output = StockOutput<B>;

    fn step(&self, batch: StockBatch<B>) -> StockOutput<B> {
        self.forward_classification(&batch)
    }
}

/// Hyperparameters for `StockModel`.
#[derive(Config, Debug)]
pub struct StockModelConfig {
    /// GRU hidden size, the temporal summary width.
    #[config(default = 64)]
    pub d_hidden: usize,
    /// Hidden width of the MLP head.
    #[config(default = 32)]
    pub d_head: usize,
    /// Dropout probability applied in the head.
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl StockModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StockModel<B> {
        StockModel::new(self, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::{Flex, FlexDevice};

    #[test]
    fn forward_outputs_logits() {
        let device = FlexDevice;
        let config = StockModelConfig::new();
        let model = config.init::<Flex>(&device);

        let technical = Tensor::<Flex, 3>::zeros([2, 8, NUM_FEATURES], &device);

        let logits = model.forward(technical);

        assert_eq!(logits.dims(), [2, NUM_CLASSES]);
    }
}
