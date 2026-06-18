use crate::data::store::TickerStore;
use crate::training::dataset::StockItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use stock_model::inference::gather_windows;
use stock_model::model::NUM_FEATURES;

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, stationary_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size]` -- class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
}

/// Builds a [`StockBatch`] by gathering windows from device-resident tensors.
///
/// The model is tiny, so per-batch cost is dominated by host->device transfer, not
/// compute. The whole store is uploaded once into the resident tensors below, so a
/// batch is a pure on-device gather and the only per-batch transfer is the window
/// start rows. Cheap to clone across the loader; train and valid each build their own.
#[derive(Clone)]
pub struct StockBatcher<B: Backend> {
    steps: usize,
    /// Resident `[rows, 5]` features.
    features: Tensor<B, 2>,
    /// Resident `[rows]` label per row.
    labels: Tensor<B, 1, Int>,
}

impl<B: Backend> StockBatcher<B> {
    /// Upload the store's flattened buffers to `device` once.
    pub fn new(steps: usize, store: &TickerStore, device: &B::Device) -> Self {
        let buffers = store.resident_buffers();
        let rows = buffers.rows;

        let features = Tensor::from_data(
            TensorData::new(buffers.features, [rows, NUM_FEATURES]),
            device,
        );
        let labels = Tensor::from_data(TensorData::new(buffers.labels, [rows]), device);

        Self {
            steps,
            features,
            labels,
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let count = items.len();
        let steps = self.steps;

        let start_values: Vec<u32> = items.iter().map(|item| item.start).collect();
        let starts = Tensor::<B, 1, Int>::from_data(TensorData::new(start_values, [count]), device);

        // Same on-device gather the live predictor runs, so the index math has one source.
        let technical = gather_windows(&self.features, &starts, steps);

        // The label comes from the window's last day.
        let steps = u32::try_from(steps)
            .expect("steps exceeds u32; window length far larger than supported");
        let last = starts.add_scalar(steps - 1);
        let label = self.labels.clone().select(0, last);

        StockBatch { technical, label }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::flex::{Flex, FlexDevice};

    type TestBackend = Flex;

    #[test]
    fn gathers_windows_and_labels() {
        // Two tickers of ten rows; row `i` fills all features with `base + i`, base
        // separating tickers (0 and 1000).
        let store = TickerStore::synthetic(2, 10);
        let batcher = StockBatcher::<TestBackend>::new(4, &store, &FlexDevice);

        // First ticker's row 0, and second ticker's row 1 at offset 10.
        let items = vec![StockItem { start: 0 }, StockItem { start: 11 }];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [2, 4, NUM_FEATURES]);
        assert_eq!(batch.label.dims(), [2]);

        // First window gathers ticker 0 rows 0..4, second ticker 1 rows 1..5.
        let values = batch.technical.into_data().to_vec::<f32>().unwrap();
        let stride = NUM_FEATURES;
        let expected_first = [0.0f32, 1.0, 2.0, 3.0];
        let expected_second = [1001.0f32, 1002.0, 1003.0, 1004.0];
        for step in 0..4 {
            assert!((values[step * stride] - expected_first[step]).abs() < 1e-6);
            assert!((values[(4 + step) * stride] - expected_second[step]).abs() < 1e-6);
        }

        // Labels come from each window's last row: ticker 0 row 3 (3 % 3 = 0) and
        // ticker 1 row 4 (4 % 3 = 1), so the u8 store labels convert by value.
        let labels: Vec<i64> = batch.label.into_data().iter::<i64>().collect();
        assert_eq!(labels, vec![0, 1]);
    }
}
