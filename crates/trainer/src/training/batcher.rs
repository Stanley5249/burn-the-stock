use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use polars::prelude::Series;
use stock_model::data::{StockItem, TickerFrames, gather_windows};

#[derive(Clone, Debug)]
pub struct StockBatch<B: Backend> {
    /// Shape `[batch_size, steps, stationary_features]`.
    pub technical: Tensor<B, 3>,
    /// Shape `[batch_size]` -- the per-date z-scored MFE target.
    pub target: Tensor<B, 1>,
}

/// Builds a [`StockBatch`] by copying each window's contiguous rows out of the
/// per-ticker feature `Series` into one flat host buffer, then a single upload. The
/// `Series` clone by shared buffer, so the batcher is cheap to clone across the loader's
/// workers; train and valid build their own. No backend type, the upload picks it up.
#[derive(Clone)]
pub struct StockBatcher {
    steps: usize,
    features: Vec<Series>,
    labels: Vec<Series>,
}

impl StockBatcher {
    /// Pull the store's per-ticker feature and label `Series`.
    ///
    /// # Panics
    /// If the store is unlabeled or a column has the wrong dtype, both store invariants.
    pub fn new(steps: usize, store: &TickerFrames) -> Self {
        Self {
            steps,
            features: store.feature_series().expect("store has a feature column"),
            labels: store
                .label_series()
                .expect("labeled store has a label column"),
        }
    }
}

impl<B: Backend> Batcher<B, StockItem, StockBatch<B>> for StockBatcher {
    #[tracing::instrument(skip_all, fields(n = items.len()))]
    fn batch(&self, items: Vec<StockItem>, device: &B::Device) -> StockBatch<B> {
        let technical = gather_windows::<B>(&self.features, &items, self.steps, device);

        // The target comes from the window's last day, indexed ticker-locally.
        let rows: Vec<f32> = items
            .iter()
            .map(|item| {
                self.labels[item.ticker]
                    .f32()
                    .expect("f32 label series")
                    .get(item.start + self.steps - 1)
                    .expect("row in range")
            })
            .collect();
        let target = Tensor::from_data(TensorData::new(rows, [items.len()]), device);

        StockBatch { technical, target }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::flex::{Flex, FlexDevice};
    use stock_model::features::NUM_FEATURES;

    use crate::label::synthetic;

    type TestBackend = Flex;

    #[test]
    fn slices_windows_and_labels() {
        // Two tickers of ten rows; row `i` fills features with `base + i`, base
        // separating tickers (0 and 1000).
        let store = synthetic(2, 10);
        let batcher = StockBatcher::new(4, &store);

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
        let batch: StockBatch<TestBackend> = batcher.batch(items, &FlexDevice);

        assert_eq!(batch.technical.dims(), [2, 4, NUM_FEATURES]);
        assert_eq!(batch.target.dims(), [2]);

        // First window covers ticker 0 rows 0..4, second ticker 1 rows 1..5.
        let values = batch.technical.into_data().to_vec::<f32>().unwrap();
        let stride = NUM_FEATURES;
        for step in 0..4u8 {
            let row = usize::from(step);
            assert!((values[row * stride] - f32::from(step)).abs() < 1e-6);
            assert!((values[(4 + row) * stride] - (1001.0 + f32::from(step))).abs() < 1e-6);
        }

        // Targets come from each window's last row: ticker 0 row 3 and ticker 1 row 4.
        let targets: Vec<f32> = batch.target.into_data().iter::<f32>().collect();
        assert_eq!(targets, vec![3.0, 4.0]);
    }
}
