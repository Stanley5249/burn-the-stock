use std::marker::PhantomData;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

/// Buy class index, matching `crate::label::Label::class`.
const BUY: i64 = 2;

/// Input shared by the trade-aware metrics: the class logits, the true class
/// index, and the signed forward return each row heads toward.
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

/// Expected value per Buy trade, in percent.
///
/// Long only and per sample: a predicted Buy earns the row's reward minus the
/// round-trip fee, while Hold and Sell stay flat at zero. Weighting each batch by
/// its Buy count makes the running value the mean payoff over every Buy the model
/// made across the epoch, which says directly whether its Buy calls pay. The TBL
/// reward is already bounded by the barrier, so no outlier clip is needed.
#[derive(Clone)]
pub struct ExpectedValueMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    fee: f32,
    _b: PhantomData<B>,
}

impl<B: Backend> ExpectedValueMetric<B> {
    pub fn new(fee: f32) -> Self {
        Self {
            name: Arc::new("EV".to_string()),
            state: NumericMetricState::default(),
            fee,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for ExpectedValueMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size, _] = input.logits.dims();
        let predictions = input.logits.clone().argmax(1).reshape([batch_size]);
        let is_buy = predictions.equal_elem(BUY);

        let count =
            usize::try_from(is_buy.clone().int().sum().into_scalar().elem::<i64>()).unwrap_or(0);

        // Long only: only Buy takes a position, so only its rows earn or lose.
        let payoff = input.reward.clone().sub_scalar(self.fee);

        let is_buy = is_buy.float();
        let trades = is_buy.clone().sum().into_scalar().elem::<f64>();
        let total = (payoff * is_buy).sum().into_scalar().elem::<f64>();

        let value = if count > 0 {
            100.0 * total / trades
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

impl<B: Backend> Numeric for ExpectedValueMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

/// Precision for one action class, in percent.
///
/// Of the rows predicted as this class, the fraction whose true label matches.
/// Weighting by the predicted count makes the running value the precision over
/// every such prediction in the epoch. Reported per class because the macro
/// F-beta hides whether the rare Buy and Sell calls are the trustworthy ones.
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
    fn ev_averages_buy_payoff() {
        let device = FlexDevice;
        // Predictions argmax to Buy, Hold, Buy.
        let logits = Tensor::<Flex, 2>::from_data(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &device,
        );
        let targets = Tensor::<Flex, 1, Int>::from_data([2, 1, 2], &device);
        let reward = Tensor::<Flex, 1>::from_data([0.10f32, 0.50, -0.04], &device);

        let mut metric = ExpectedValueMetric::<Flex>::new(0.01);
        metric.update(&StockEvalInput::new(logits, targets, reward), &meta());

        // Buy rows 0 and 2 only: (0.10 - 0.01) + (-0.04 - 0.01) = 0.04 over two
        // trades, so 2%. Hold contributes nothing.
        assert!((metric.value().current() - 2.0).abs() < 1e-4);
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
