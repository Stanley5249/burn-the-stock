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
const HOLD: i64 = 1;
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

/// Average net edge per trade the model would take. Over the rows predicted Buy
/// (`argmax == BUY`), each scores by its true label using the barrier magnitudes:
/// a true Buy earns `+take_profit`, a true Sell `-stop_loss`, a true Hold `-fee`.
/// The mean over those rows is the per-trade edge of the Buy action.
#[derive(Clone)]
pub struct BuyEdgeMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    take_profit: f64,
    stop_loss: f64,
    fee: f64,
    _b: PhantomData<B>,
}

impl<B: Backend> BuyEdgeMetric<B> {
    pub fn new(take_profit: f32, stop_loss: f32, fee: f32) -> Self {
        Self {
            name: Arc::new("Buy edge".to_string()),
            state: NumericMetricState::default(),
            take_profit: f64::from(take_profit),
            stop_loss: f64::from(stop_loss),
            fee: f64::from(fee),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for BuyEdgeMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size, _] = input.logits.dims();
        let predictions = input.logits.clone().argmax(1).reshape([batch_size]);
        let predicted_buy = predictions.equal_elem(BUY);

        let count = usize::try_from(
            predicted_buy
                .clone()
                .int()
                .sum()
                .into_scalar()
                .elem::<i64>(),
        )
        .unwrap_or(0);

        let predicted_buy = predicted_buy.float();
        let is_buy = input.targets.clone().equal_elem(BUY).float();
        let is_sell = input.targets.clone().equal_elem(SELL).float();
        let is_hold = input.targets.clone().equal_elem(HOLD).float();

        // Barrier payoff keyed on the true label, summed over predicted buys.
        let payoff = is_buy.mul_scalar(self.take_profit)
            - is_sell.mul_scalar(self.stop_loss)
            - is_hold.mul_scalar(self.fee);
        let predicted_count = predicted_buy.clone().sum().into_scalar().elem::<f64>();
        let total = (payoff * predicted_buy).sum().into_scalar().elem::<f64>();

        let value = if count > 0 {
            total / predicted_count
        } else {
            0.0
        };

        self.state
            .update(value, count, FormatOptions::new(self.name()).precision(4))
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

impl<B: Backend> Numeric for BuyEdgeMetric<B> {
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
    fn buy_edge_averages_barrier_payoff_over_predicted_buys() {
        let device = FlexDevice;
        // Rows 0-2 predict Buy, row 3 predicts Hold and is excluded. Of the predicted
        // buys, true labels are Buy, Sell, Hold, paying +tp, -sl, -fee.
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

        let mut metric = BuyEdgeMetric::<Flex>::new(0.10, 0.04, 0.01);
        metric.update(&StockEvalInput::new(logits, targets), &meta());

        // (0.10 - 0.04 - 0.01) / 3 predicted buys.
        assert!((metric.value().current() - 0.05 / 3.0).abs() < 1e-6);
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
