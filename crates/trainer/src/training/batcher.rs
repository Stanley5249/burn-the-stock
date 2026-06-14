use crate::data::store::TickerStore;
use crate::training::dataset::StockItem;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use stock_model::features::FEATURE_NAMES;

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, stationary_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size]` -- class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
    /// Shape `[batch_size]` -- signed realized return of the barrier outcome.
    pub reward: Tensor<B, 1>,
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
    /// Resident `[rows]` barrier return per row.
    rewards: Tensor<B, 1>,
}

impl<B: Backend> StockBatcher<B> {
    /// Upload the store's flattened buffers to `device` once.
    pub fn new(steps: usize, store: &TickerStore, device: &B::Device) -> Self {
        let buffers = store.resident_buffers();
        let rows = buffers.rows;

        let features = Tensor::from_data(
            TensorData::new(buffers.features, [rows, FEATURE_NAMES.len()]),
            device,
        );
        let labels = Tensor::from_data(TensorData::new(buffers.labels, [rows]), device);
        let rewards = Tensor::from_data(TensorData::new(buffers.rewards, [rows]), device);

        Self {
            steps,
            features,
            labels,
            rewards,
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let count = items.len();
        let steps = self.steps;

        let start_values: Vec<i32> = items
            .iter()
            .map(|item| {
                i32::try_from(item.start)
                    .expect("row index exceeds i32; dataset far larger than supported")
            })
            .collect();
        let starts = Tensor::<B, 1, Int>::from_data(TensorData::new(start_values, [count]), device);

        // `[count, steps]` row indices: each window start broadcast over `0..steps`.
        let offsets = Tensor::<B, 1, Int>::arange(
            0..i64::try_from(steps)
                .expect("steps exceeds i64; window length far larger than supported"),
            device,
        );
        let index = starts.clone().reshape([count, 1]).expand([count, steps])
            + offsets.reshape([1, steps]).expand([count, steps]);

        // One indexed read, then fold the flat rows into `[count, steps, features]`.
        let technical = self
            .features
            .clone()
            .select(0, index.reshape([count * steps]))
            .reshape([count, steps, FEATURE_NAMES.len()]);

        // Label and reward come from the window's last day.
        let last = starts.add_scalar(
            i32::try_from(steps)
                .expect("steps exceeds i32; window length far larger than supported")
                - 1,
        );
        let label = self.labels.clone().select(0, last.clone());
        let reward = self.rewards.clone().select(0, last);

        StockBatch {
            technical,
            label,
            reward,
        }
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

        // First ticker's row 0, and second ticker's row 0 at offset 10.
        let items = vec![StockItem { start: 0 }, StockItem { start: 10 }];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [2, 4, FEATURE_NAMES.len()]);
        assert_eq!(batch.label.dims(), [2]);

        // First window gathers ticker 0 rows 0..4, second ticker 1 rows 0..4.
        let values = batch.technical.into_data().to_vec::<f32>().unwrap();
        let stride = FEATURE_NAMES.len();
        let expected_first = [0.0f32, 1.0, 2.0, 3.0];
        let expected_second = [1000.0f32, 1001.0, 1002.0, 1003.0];
        for step in 0..4 {
            assert!((values[step * stride] - expected_first[step]).abs() < 1e-6);
            assert!((values[(4 + step) * stride] - expected_second[step]).abs() < 1e-6);
        }
    }
}
