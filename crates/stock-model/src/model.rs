use burn::nn::gru::{Gru, GruConfig};
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;

/// Action classes the model scores: Sell, Hold, Buy.
pub const NUM_CLASSES: usize = 3;

/// Standardized feature width of the technical input, matching the feature column.
const NUM_FEATURES: usize = 5;

/// GRU classifier over the standardized feature window.
///
/// A two-layer GRU summarizes the window into its last hidden state, which a small
/// MLP head turns into the action logits. This is the architecture only: the loss
/// and the training step live with the trainer, so a model loaded for inference
/// carries no training machinery.
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
}

impl<B: Backend> StockModel<B> {
    #[must_use]
    pub fn new(config: &StockModelConfig, device: &B::Device) -> Self {
        let gru_1 = GruConfig::new(NUM_FEATURES, config.d_hidden, true).init(device);
        let gru_1_norm = RmsNormConfig::new(config.d_hidden).init(device);
        let gru_2 = GruConfig::new(config.d_hidden, config.d_hidden, true).init(device);
        let gru_2_norm = RmsNormConfig::new(config.d_hidden).init(device);
        let hidden = LinearConfig::new(config.d_hidden, config.d_head).init(device);
        let head = LinearConfig::new(config.d_head, NUM_CLASSES).init(device);

        Self {
            gru_1,
            gru_1_norm,
            gru_2,
            gru_2_norm,
            hidden,
            head,
            activation: Gelu::new(),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    /// Returns logits with shape `[batch, NUM_CLASSES]`. Dropout is inert on a plain
    /// backend, so this is the inference path as well as the training forward.
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
}

/// Hyperparameters for [`StockModel`].
#[derive(Config, Debug)]
pub struct StockModelConfig {
    /// GRU hidden size, the temporal summary width.
    #[config(default = 8)]
    pub d_hidden: usize,
    /// Hidden width of the MLP head.
    #[config(default = 32)]
    pub d_head: usize,
    /// Dropout probability applied in the head.
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl StockModelConfig {
    #[must_use]
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
