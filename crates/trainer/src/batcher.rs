use crate::dataset::StockItem;
use crate::store::FEATURE_NAMES;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, ohlcv_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size, ticker_features]`.
    pub ticker: Tensor<B, 2>,
    /// Shape `[batch_size]` -- class index 0/1/2.
    pub label: Tensor<B, 1, Int>,
    /// Shape `[batch_size]` -- signed forward return to the next swing extreme.
    pub reward: Tensor<B, 1>,
}

/// Packs a `Vec<StockItem>` into one [`StockBatch`] on the target device.
///
/// Holds only the window length and industry width, so the same batch shape can
/// be produced on the autodiff backend for training and on the inner backend
/// for validation by instantiating two batchers over one shared store.
#[derive(Clone)]
pub struct StockBatcher<B: Backend> {
    steps: usize,
    n_industries: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> StockBatcher<B> {
    pub fn new(steps: usize, n_industries: usize) -> Self {
        Self {
            steps,
            n_industries,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher<B> {
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let count = items.len();

        let mut technical_data = Vec::with_capacity(count * self.steps * FEATURE_NAMES.len());
        let mut label_data = Vec::with_capacity(count);
        let mut reward_data = Vec::with_capacity(count);

        // One-hot industries, left empty when no categorical feature is attached.
        let mut industry_data = vec![0.0f32; count * self.n_industries];

        for (row, item) in items.iter().enumerate() {
            technical_data.extend_from_slice(&item.technical);
            label_data.push(item.label);
            reward_data.push(item.reward);
            if self.n_industries != 0 {
                industry_data[row * self.n_industries + item.industry] = 1.0;
            }
        }

        let technical = Tensor::from_data(
            TensorData::new(technical_data, [count, self.steps, FEATURE_NAMES.len()]),
            device,
        );

        let label = Tensor::from_data(TensorData::new(label_data, [count]), device);

        let reward = Tensor::from_data(TensorData::new(reward_data, [count]), device);

        let ticker = if self.n_industries == 0 {
            Tensor::zeros([count, 0], device)
        } else {
            Tensor::from_data(
                TensorData::new(industry_data, [count, self.n_industries]),
                device,
            )
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

    fn item(industry: usize, label: i32) -> StockItem {
        StockItem {
            technical: vec![0.0; 4 * FEATURE_NAMES.len()],
            industry,
            label,
            reward: 0.0,
        }
    }

    #[test]
    fn batch_shapes_and_one_hot() {
        let batcher = StockBatcher::<TestBackend>::new(4, 3);

        let items = vec![item(0, 0), item(2, 1), item(1, 2)];
        let batch = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [3, 4, FEATURE_NAMES.len()]);
        assert_eq!(batch.label.dims(), [3]);
        assert_eq!(batch.ticker.dims(), [3, 3]);

        // Every industry row is one-hot, so each row sums to exactly one.
        let row_sums = batch.ticker.sum_dim(1).into_data();
        for value in row_sums.to_vec::<f32>().unwrap() {
            assert!((value - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn ticker_is_width_zero_without_industries() {
        let batcher = StockBatcher::<TestBackend>::new(4, 0);

        let batch = batcher.batch(vec![item(0, 0), item(0, 1)], &FlexDevice);

        assert_eq!(batch.ticker.dims(), [2, 0]);
    }
}
