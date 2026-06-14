use std::marker::PhantomData;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::activation::softmax;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

/// Sell and Buy class indices, matching `crate::data::label::Label::class`.
const SELL: usize = 0;
const BUY: usize = 2;

/// Floor on the Sharpe denominator, guarding a near-constant-return batch from a
/// blown-up ratio.
const EPS: f32 = 1e-8;

/// Input shared by the trade-aware metrics: logits, true class, and signed forward
/// return per row.
pub struct StockEvalInput<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
    pub reward: Tensor<B, 1>,
}

impl<B: Backend> StockEvalInput<B> {
    pub fn new(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>, reward: Tensor<B, 1>) -> Self {
        Self {
            logits,
            targets,
            reward,
        }
    }
}

/// Per-trade Sharpe of the long-only soft policy. The softmax becomes a position
/// `clamp(P(Buy) - P(Sell), 0, 1)`; each row earns its forward `reward` scaled by it,
/// less the round-trip `fee`, and the batch of net returns gives `mean / std`.
///
/// Two caveats: this is a per-batch Sharpe averaged over the epoch (burn averages
/// per-batch values, and Sharpe is non-linear), so its scale rides on `batch_size`;
/// and the reward is a per-trade return over the holding horizon, not annualized.
#[derive(Clone)]
pub struct SharpeMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    fee: f32,
    _b: PhantomData<B>,
}

impl<B: Backend> SharpeMetric<B> {
    pub fn new(fee: f32) -> Self {
        Self {
            name: Arc::new("Sharpe".to_string()),
            state: NumericMetricState::default(),
            fee,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for SharpeMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size, _] = input.logits.dims();

        let probabilities = softmax(input.logits.clone(), 1);
        let probability_sell = probabilities
            .clone()
            .slice([0..batch_size, SELL..SELL + 1])
            .reshape([batch_size]);
        let probability_buy = probabilities
            .slice([0..batch_size, BUY..BUY + 1])
            .reshape([batch_size]);

        // Clamp away negatives: a Sell only vetoes a Buy, it never shorts.
        let position = (probability_buy - probability_sell).clamp_min(0.0);

        let net = position * input.reward.clone().sub_scalar(self.fee);

        let mean = net.clone().mean();
        let deviation = net.var(0).sqrt().add_scalar(EPS);
        let sharpe = (mean / deviation).into_scalar().elem::<f64>();

        // Weighting by batch size makes the epoch value the mean of per-batch Sharpes.
        self.state.update(
            sharpe,
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for SharpeMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

/// Precision for one action class, in percent: of the rows predicted as this class,
/// the fraction whose true label matches. Per class, since the macro F-beta hides
/// whether the rare Buy and Sell calls are the trustworthy ones.
#[derive(Clone)]
pub struct PrecisionClassMetric<B: Backend> {
    name: MetricName,
    class: i64,
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> PrecisionClassMetric<B> {
    pub fn new(class: i64, class_name: &str) -> Self {
        Self {
            name: Arc::new(format!("Precision {class_name}")),
            class,
            state: NumericMetricState::default(),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for PrecisionClassMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size, _] = input.logits.dims();
        let predictions = input.logits.clone().argmax(1).reshape([batch_size]);
        let predicted = predictions.equal_elem(self.class);

        let count =
            usize::try_from(predicted.clone().int().sum().into_scalar().elem::<i64>()).unwrap_or(0);

        let predicted = predicted.float();
        let actual = input.targets.clone().equal_elem(self.class).float();

        let predicted_count = predicted.clone().sum().into_scalar().elem::<f64>();
        let true_positive = (predicted * actual).sum().into_scalar().elem::<f64>();

        let value = if count > 0 {
            100.0 * true_positive / predicted_count
        } else {
            0.0
        };

        self.state.update(
            value,
            count,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for PrecisionClassMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::{Flex, FlexDevice};
    use burn::data::dataloader::Progress;

    fn meta() -> MetricMetadata {
        MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            global_progress: Progress {
                items_processed: 0,
                items_total: 1,
            },
            iteration: Some(0),
            lr: None,
        }
    }

    #[test]
    fn sharpe_rewards_confident_profitable_buys() {
        let device = FlexDevice;
        // All three rows lean hard toward Buy, so the position is ~1, and the
        // rewards are positive with a finite spread, so the net return series has a
        // clearly positive Sharpe.
        let logits = Tensor::<Flex, 2>::from_data(
            [[0.0, 0.0, 6.0], [0.0, 0.0, 6.0], [0.0, 0.0, 6.0]],
            &device,
        );
        let targets = Tensor::<Flex, 1, Int>::from_data([2, 2, 2], &device);
        let reward = Tensor::<Flex, 1>::from_data([0.08f32, 0.10, 0.12], &device);

        let mut metric = SharpeMetric::<Flex>::new(0.0);
        metric.update(&StockEvalInput::new(logits, targets, reward), &meta());

        assert!(metric.value().current() > 0.0);
    }

    #[test]
    fn precision_counts_true_buys() {
        let device = FlexDevice;
        // Predictions argmax to Buy, Buy, Hold.
        let logits = Tensor::<Flex, 2>::from_data(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            &device,
        );
        let targets = Tensor::<Flex, 1, Int>::from_data([2, 0, 1], &device);
        let reward = Tensor::<Flex, 1>::zeros([3], &device);

        let mut metric = PrecisionClassMetric::<Flex>::new(2, "Buy");
        metric.update(&StockEvalInput::new(logits, targets, reward), &meta());

        // Two rows predicted Buy, only the first is truly Buy, so 50%.
        assert!((metric.value().current() - 50.0).abs() < 1e-4);
    }
}
