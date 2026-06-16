use std::marker::PhantomData;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

/// Class indices, matching `crate::data::label::Label::class`.
const SELL: i64 = 0;
const BUY: i64 = 2;

/// Input shared by the trade-aware metrics: logits and true class per row.
pub struct StockEvalInput<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> StockEvalInput<B> {
    pub fn new(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Self {
        Self { logits, targets }
    }
}

/// Empirical expected value of the Buy policy, per opportunity. Over the rows
/// predicted Buy (`argmax == BUY`), each pays the round-trip `fee` and earns by its
/// true label: a true Buy `+take_profit`, a true Sell `-stop_loss`, a true Hold `0`.
/// Non-Buy rows score `0`. The batch mean is the per-name EV net of cost, so it
/// rises as the model both buys more and buys correctly.
#[derive(Clone)]
pub struct ExpectedValueMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    take_profit: f64,
    stop_loss: f64,
    fee: f64,
    _b: PhantomData<B>,
}

impl<B: Backend> ExpectedValueMetric<B> {
    pub fn new(take_profit: f32, stop_loss: f32, fee: f32) -> Self {
        Self {
            name: Arc::new("Expected value".to_string()),
            state: NumericMetricState::default(),
            take_profit: f64::from(take_profit),
            stop_loss: f64::from(stop_loss),
            fee: f64::from(fee),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for ExpectedValueMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size, _] = input.logits.dims();
        let predictions = input.logits.clone().argmax(1).reshape([batch_size]);
        let predicted_buy = predictions.equal_elem(BUY).float();

        let is_buy = input.targets.clone().equal_elem(BUY).float();
        let is_sell = input.targets.clone().equal_elem(SELL).float();

        // Round-trip fee on every taken position, plus the barrier payoff keyed on
        // the true label. Averaged across the whole batch, so the score scales with
        // how often Buy fires, not just per-trade quality.
        let payoff =
            is_buy.mul_scalar(self.take_profit) - is_sell.mul_scalar(self.stop_loss) - self.fee;
        let total = (payoff * predicted_buy).sum().into_scalar().elem::<f64>();
        let value = total / f64::from(u32::try_from(batch_size).expect("batch size fits in u32"));

        self.state.update(
            value,
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

impl<B: Backend> Numeric for ExpectedValueMetric<B> {
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
    fn expected_value_averages_net_payoff_over_batch() {
        let device = FlexDevice;
        // Rows 0-2 predict Buy, row 3 predicts Hold and is excluded. Each predicted
        // buy pays the fee; true labels Buy, Sell, Hold pay +tp, -sl, 0 on top.
        let logits = Tensor::<Flex, 2>::from_data(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            &device,
        );
        let targets = Tensor::<Flex, 1, Int>::from_data([2, 0, 1, 2], &device);

        let mut metric = ExpectedValueMetric::<Flex>::new(0.10, 0.04, 0.01);
        metric.update(&StockEvalInput::new(logits, targets), &meta());

        // Payoffs 0.09, -0.05, -0.01 summed over 3 predicted buys, over the batch of 4.
        assert!((metric.value().current() - 0.03 / 4.0).abs() < 1e-6);
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

        let mut metric = PrecisionClassMetric::<Flex>::new(2, "Buy");
        metric.update(&StockEvalInput::new(logits, targets), &meta());

        // Two rows predicted Buy, only the first is truly Buy, so 50%.
        assert!((metric.value().current() - 50.0).abs() < 1e-4);
    }
}
