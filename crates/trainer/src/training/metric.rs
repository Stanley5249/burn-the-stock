use std::marker::PhantomData;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

/// Input for the rank metric: the predicted score and the true target per row.
pub struct StockEvalInput<B: Backend> {
    pub predictions: Tensor<B, 1>,
    pub targets: Tensor<B, 1>,
}

/// Pearson correlation between the predicted score and the target over the batch, the
/// information coefficient that tracks ranking quality. The target is already z-scored
/// per date, so this batch-level Pearson is a close proxy for the true per-date IC;
/// group rows by date to compute that IC exactly, should the proxy drift.
#[derive(Clone)]
pub struct CorrelationMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> CorrelationMetric<B> {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Correlation".to_string()),
            state: NumericMetricState::default(),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Default for CorrelationMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Metric for CorrelationMetric<B> {
    type Input = StockEvalInput<B>;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size] = input.predictions.dims();

        let predictions = input.predictions.clone();
        let targets = input.targets.clone();

        let mean_prediction = predictions.clone().mean().into_scalar().elem::<f64>();
        let mean_target = targets.clone().mean().into_scalar().elem::<f64>();

        let centered_prediction = predictions.sub_scalar(mean_prediction);
        let centered_target = targets.sub_scalar(mean_target);

        let covariance = (centered_prediction.clone() * centered_target.clone())
            .mean()
            .into_scalar()
            .elem::<f64>();
        let variance_prediction = (centered_prediction.clone() * centered_prediction)
            .mean()
            .into_scalar()
            .elem::<f64>();
        let variance_target = (centered_target.clone() * centered_target)
            .mean()
            .into_scalar()
            .elem::<f64>();

        let correlation =
            covariance / (variance_prediction.sqrt() * variance_target.sqrt() + 1e-12);

        self.state.update(
            correlation,
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

impl<B: Backend> Numeric for CorrelationMetric<B> {
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
    fn correlation_is_one_for_aligned_scores() {
        let device = FlexDevice;
        // A perfect linear relationship: targets = 2 * predictions, so corr == 1.
        let predictions = Tensor::<Flex, 1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        let targets = Tensor::<Flex, 1>::from_data([2.0, 4.0, 6.0, 8.0], &device);

        let mut metric = CorrelationMetric::<Flex>::new();
        metric.update(
            &StockEvalInput {
                predictions,
                targets,
            },
            &meta(),
        );

        assert!((metric.value().current() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn correlation_is_negative_for_reversed_scores() {
        let device = FlexDevice;
        let predictions = Tensor::<Flex, 1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
        let targets = Tensor::<Flex, 1>::from_data([4.0, 3.0, 2.0, 1.0], &device);

        let mut metric = CorrelationMetric::<Flex>::new();
        metric.update(
            &StockEvalInput {
                predictions,
                targets,
            },
            &meta(),
        );

        assert!((metric.value().current() + 1.0).abs() < 1e-4);
    }
}
