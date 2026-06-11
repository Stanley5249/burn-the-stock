use crate::batcher::StockBatch;
use burn::backend::flex::{Flex, FlexDevice};
use burn::nn::gru::{Gru, GruConfig};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyInput, Adaptor, ConfusionStatsInput, ItemLazy, LossInput};
use burn::train::{InferenceStep, TrainOutput, TrainStep};

pub const NUM_CLASSES: usize = 3;

/// Sell, Hold, Buy
pub const CLASS_WEIGHTS: [f32; NUM_CLASSES] = [2.0, 1.0, 2.0];

/// OHLCV width of the technical input, matching the dataloader's feature column.
const NUM_FEATURES: usize = 5;

/// Multi-branch late-fusion classifier.
///
/// A GRU summarizes the OHLCV window into its last hidden state, a linear layer
/// embeds the industry one-hot, and the two branches are concatenated and run
/// through a small MLP head that emits the action logits.
///
/// Input:  technical `[batch, steps, 5]`, ticker `[batch, n_industries]`
/// Output: `[batch, NUM_CLASSES]` (logits)
#[derive(Module, Debug)]
pub struct StockModel<B: Backend> {
    gru: Gru<B>,
    gru_norm: RmsNorm<B>,
    industry: Linear<B>,
    industry_norm: RmsNorm<B>,
    fusion: Linear<B>,
    head: Linear<B>,
    activation: Gelu,
    dropout: Dropout,
    loss: CrossEntropyLoss<B>,
}

impl<B: Backend> StockModel<B> {
    pub fn new(config: &StockModelConfig, device: &B::Device) -> Self {
        let gru = GruConfig::new(NUM_FEATURES, config.d_hidden, true).init(device);

        let gru_norm = RmsNormConfig::new(config.d_hidden).init(device);

        let industry = LinearConfig::new(config.n_industries, config.d_industry).init(device);

        let industry_norm = RmsNormConfig::new(config.d_industry).init(device);

        let fusion =
            LinearConfig::new(config.d_hidden + config.d_industry, config.d_fusion).init(device);

        let head = LinearConfig::new(config.d_fusion, NUM_CLASSES).init(device);

        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(CLASS_WEIGHTS.to_vec()))
            .init(device);

        Self {
            gru,
            gru_norm,
            industry,
            industry_norm,
            fusion,
            head,
            activation: Gelu::new(),
            dropout: DropoutConfig::new(config.dropout).init(),
            loss,
        }
    }

    /// Returns logits with shape `[batch, NUM_CLASSES]`.
    pub fn forward(&self, technical: Tensor<B, 3>, ticker: Tensor<B, 2>) -> Tensor<B, 2> {
        let temporal = self.gru.forward(technical, None);

        // Summarize the window by its last hidden state: [batch, d_hidden].
        let [batch, sequence, d_hidden] = temporal.dims();

        let summary = temporal
            .slice([0..batch, sequence - 1..sequence, 0..d_hidden])
            .reshape([batch, d_hidden]);

        let categorical = self.industry.forward(ticker);

        let fused = Tensor::cat(vec![summary, categorical], 1);
        let hidden = self.activation.forward(self.fusion.forward(fused));
        let hidden = self.dropout.forward(hidden);

        self.head.forward(hidden)
    }

    fn forward_classification(&self, batch: &StockBatch<B>) -> StockOutput<B> {
        let logits = self.forward(batch.technical.clone(), batch.ticker.clone());

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
/// A local stand-in for burn's `ClassificationOutput` so it can adapt to the EV
/// metric's input as well as the built-in Accuracy, Loss, and confusion-based
/// metrics. The orphan rule blocks adding that adaptor to the foreign type.
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

impl<B: Backend> Adaptor<AccuracyInput<B>> for StockOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
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
    /// One-hot width of the industry feature, set from the dataloader.
    pub n_industries: usize,
    /// GRU hidden size, the temporal branch summary width.
    #[config(default = 64)]
    pub d_hidden: usize,
    /// Output width of the industry branch.
    #[config(default = 16)]
    pub d_industry: usize,
    /// Hidden width of the fusion head.
    #[config(default = 32)]
    pub d_fusion: usize,
    /// Dropout probability applied in the fusion head.
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
        let config = StockModelConfig::new(7);
        let model = config.init::<Flex>(&device);

        let technical = Tensor::<Flex, 3>::zeros([2, 8, NUM_FEATURES], &device);
        let ticker = Tensor::<Flex, 2>::zeros([2, 7], &device);

        let logits = model.forward(technical, ticker);

        assert_eq!(logits.dims(), [2, NUM_CLASSES]);
    }
}
