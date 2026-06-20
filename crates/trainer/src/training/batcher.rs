use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use stock_model::data::{StockItem, TickerFrames, stack_windows};

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, stationary_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size]` -- class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
}

/// Builds a [`StockBatch`] by slicing windows out of the per-ticker resident tensors.
/// The whole store is uploaded once into those tensors, so a batch is a pure on-device
/// slice and stack with no per-batch host transfer. The tensors clone by shared handle,
/// so the batcher is cheap to clone across the loader; train and valid build their own.
#[derive(Clone)]
pub struct StockBatcher<B: Backend> {
    steps: usize,
    features: Vec<Tensor<B, 2>>,
    labels: Vec<Tensor<B, 1, Int>>,
}

impl<B: Backend> StockBatcher<B> {
    /// Upload the store's per-ticker feature and label tensors once.
    ///
    /// # Panics
    /// If the store is unlabeled or a column has the wrong dtype, both store invariants.
    pub fn new(steps: usize, store: &TickerFrames, device: &B::Device) -> Self {
        Self {
            steps,
            features: store
                .feature_tensors(device)
                .expect("store has a feature column"),
            labels: store
                .label_tensors(device)
                .expect("labeled store has a label column"),
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let technical = stack_windows(&self.features, &items, self.steps, device);

        // The label comes from the window's last day.
        let label_slices = items
            .iter()
            .map(|item| {
                let last = item.start + self.steps - 1;
                self.labels[item.ticker].clone().slice(last..last + 1)
            })
            .collect();
        let label = Tensor::cat(label_slices, 0);

        StockBatch { technical, label }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic;
    use burn::backend::flex::{Flex, FlexDevice};
    use stock_model::model::NUM_FEATURES;

    type TestBackend = Flex;

    #[test]
    fn slices_windows_and_labels() {
        // Two tickers of ten rows; row `i` fills features with `base + i`, base
        // separating tickers (0 and 1000).
        let store = synthetic(2, 10);
        let batcher = StockBatcher::<TestBackend>::new(4, &store, &FlexDevice);

        // Ticker 0 from row 0, ticker 1 from row 1.
        let items = vec![
            StockItem {
                ticker: 0,
                start: 0,
            },
            StockItem {
                ticker: 1,
                start: 1,
            },
        ];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [2, 4, NUM_FEATURES]);
        assert_eq!(batch.label.dims(), [2]);

        // First window covers ticker 0 rows 0..4, second ticker 1 rows 1..5.
        let values = batch.technical.into_data().to_vec::<f32>().unwrap();
        let stride = NUM_FEATURES;
        for step in 0..4u8 {
            let row = usize::from(step);
            assert!((values[row * stride] - f32::from(step)).abs() < 1e-6);
            assert!((values[(4 + row) * stride] - (1001.0 + f32::from(step))).abs() < 1e-6);
        }

        // Labels come from each window's last row: ticker 0 row 3 (3 % 3 = 0) and
        // ticker 1 row 4 (4 % 3 = 1).
        let labels: Vec<i64> = batch.label.into_data().iter::<i64>().collect();
        assert_eq!(labels, vec![0, 1]);
    }
}
