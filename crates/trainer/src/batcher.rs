use crate::dataset::StockItem;
use crate::store::{FEATURE_NAMES, TickerStore};
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, stationary_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size, ticker_features]`.
    pub ticker: Tensor<B, 2>,
    /// Shape `[batch_size]` -- class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
    /// Shape `[batch_size]` -- signed forward return to the next swing extreme.
    pub reward: Tensor<B, 1>,
}

/// Builds a [`StockBatch`] by gathering windows from device-resident tensors.
///
/// The model is tiny, so the per-batch cost is dominated by moving feature data
/// host->device rather than by compute. To avoid that, the whole store is uploaded
/// once into the four resident tensors below, and a batch is then a pure on-device
/// gather: the only per-batch transfer is the `batch_size` window start rows. The
/// same resident set serves every batch, so the batcher is cheap to clone across
/// the loader. Train and validation each build their own set on their backend.
#[derive(Clone)]
pub struct StockBatcher<B: Backend> {
    steps: usize,
    n_industries: usize,
    /// Resident `[rows, 5]` stationary features for the whole store.
    features: Tensor<B, 2>,
    /// Resident `[rows]` action label per row.
    labels: Tensor<B, 1, Int>,
    /// Resident `[rows]` signed forward return per row.
    rewards: Tensor<B, 1>,
    /// Resident `[rows]` industry bucket per row, width zero when none attached.
    industry_row: Tensor<B, 1, Int>,
}

impl<B: Backend> StockBatcher<B> {
    /// Upload the store's flattened buffers to `device` once. `steps` is the window
    /// length and `n_industries` the one-hot width, both fixed for the run.
    pub fn new(steps: usize, n_industries: usize, store: &TickerStore, device: &B::Device) -> Self {
        let buffers = store.resident_buffers();
        let rows = buffers.rows;

        let features = Tensor::from_data(
            TensorData::new(buffers.features, [rows, FEATURE_NAMES.len()]),
            device,
        );
        let labels = Tensor::from_data(TensorData::new(buffers.labels, [rows]), device);
        let rewards = Tensor::from_data(TensorData::new(buffers.rewards, [rows]), device);

        // Left empty when no categorical feature is attached, matching the
        // width-zero ticker branch in `batch`.
        let industry_row = if n_industries == 0 {
            Tensor::zeros([0], device)
        } else {
            Tensor::from_data(TensorData::new(buffers.industry_row, [rows]), device)
        };

        Self {
            steps,
            n_industries,
            features,
            labels,
            rewards,
            industry_row,
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let count = items.len();
        let steps = self.steps;

        let start_values: Vec<i32> = items
            .iter()
            .map(|item| i32::try_from(item.start).unwrap())
            .collect();
        let starts = Tensor::<B, 1, Int>::from_data(TensorData::new(start_values, [count]), device);

        // Build the `[count, steps]` absolute row indices by broadcasting each
        // window start across the step offsets `0..steps`.
        let offsets = Tensor::<B, 1, Int>::arange(0..i64::try_from(steps).unwrap(), device);
        let index = starts.clone().reshape([count, 1]).expand([count, steps])
            + offsets.reshape([1, steps]).expand([count, steps]);

        // One indexed read pulls every window's rows out of the resident features,
        // then the flat rows fold back into `[count, steps, features]`.
        let technical = self
            .features
            .clone()
            .select(0, index.reshape([count * steps]))
            .reshape([count, steps, FEATURE_NAMES.len()]);

        // The label, reward, and industry are read on the window's last day.
        let last = starts.add_scalar(i32::try_from(steps).unwrap() - 1);
        let label = self.labels.clone().select(0, last.clone());
        let reward = self.rewards.clone().select(0, last.clone());

        let ticker = if self.n_industries == 0 {
            Tensor::zeros([count, 0], device)
        } else {
            self.industry_row
                .clone()
                .select(0, last)
                .one_hot(self.n_industries)
                .float()
        };

        StockBatch {
            technical,
            ticker,
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
    fn gathers_windows_labels_and_one_hot() {
        // Two tickers of ten rows; synthetic row `i` fills all five features with
        // `base + i`, where base separates tickers (0 and 1000).
        let store = TickerStore::synthetic(2, 10).set_industries(vec![1, 0], 2);
        let batcher = StockBatcher::<TestBackend>::new(4, 2, &store, &FlexDevice);

        // Absolute starts: the first ticker's row 0, and the second ticker's row 0
        // at its row offset 10.
        let items = vec![StockItem { start: 0 }, StockItem { start: 10 }];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [2, 4, FEATURE_NAMES.len()]);
        assert_eq!(batch.label.dims(), [2]);
        assert_eq!(batch.ticker.dims(), [2, 2]);

        // The first window gathers ticker 0 rows 0..4 (values 0..3), the second
        // gathers ticker 1 rows 0..4 (values 1000..1003), across all five features.
        let values = batch.technical.into_data().to_vec::<f32>().unwrap();
        let stride = FEATURE_NAMES.len();
        let expected_first = [0.0f32, 1.0, 2.0, 3.0];
        let expected_second = [1000.0f32, 1001.0, 1002.0, 1003.0];
        for step in 0..4 {
            assert!((values[step * stride] - expected_first[step]).abs() < 1e-6);
            assert!((values[(4 + step) * stride] - expected_second[step]).abs() < 1e-6);
        }

        // Industry is read on the last day: ticker 0 -> bucket 1, ticker 1 -> 0.
        let ticker = batch.ticker.into_data().to_vec::<f32>().unwrap();
        let expected_ticker = [0.0f32, 1.0, 1.0, 0.0];
        assert_eq!(ticker.len(), expected_ticker.len());
        for (got, want) in ticker.iter().zip(expected_ticker) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn ticker_is_width_zero_without_industries() {
        let store = TickerStore::synthetic(2, 10);
        let batcher = StockBatcher::<TestBackend>::new(4, 0, &store, &FlexDevice);

        let items = vec![StockItem { start: 0 }, StockItem { start: 10 }];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.ticker.dims(), [2, 0]);
    }
}
