use crate::store::TickerStore;
use burn::data::dataset::Dataset;
use fastrand::Rng;
use std::sync::Arc;

/// One materialized training sample: a stationary feature window plus its
/// resolved industry index and action label. Backend-free, so the same item
/// feeds the autodiff train batcher and the inner-backend valid batcher.
#[derive(Clone, Debug)]
pub struct StockItem {
    /// Row-major stationary features, length `steps * 5`.
    pub technical: Vec<f32>,
    /// Resolved industry bucket; 0 when no industries are attached.
    pub industry: usize,
    /// Action class index 0/1/2.
    pub label: i32,
    /// Signed forward return to the window's next swing extreme.
    pub reward: f32,
}

/// A [`Dataset`] over every `steps`-length window of a [`TickerStore`].
///
/// The store is shared behind an `Arc` so the train and valid datasets are
/// cheap to build and the dataset stays backend-free. `get` materializes one
/// window on demand, which lets burn's data loader parallelize it across
/// workers.
pub struct WindowDataset {
    store: Arc<TickerStore>,
    /// Every `(ticker_index, window_start)` pair this dataset indexes.
    windows: Vec<(u32, u32)>,
    steps: usize,
}

impl WindowDataset {
    /// Index every window of the store in ticker-then-date order.
    pub fn new(store: Arc<TickerStore>, steps: usize) -> Self {
        let windows = store.enumerate_windows(steps);
        Self {
            store,
            windows,
            steps,
        }
    }

    /// Like [`Self::new`] but shuffle the window order once with `seed`. Pairing
    /// this with a [`burn::data::dataset::transform::PartialDataset`] cap yields
    /// a validation subsample drawn evenly across tickers and dates rather than
    /// biased to the earliest ones, and stable across epochs.
    pub fn subsample(store: Arc<TickerStore>, steps: usize, seed: u64) -> Self {
        let mut windows = store.enumerate_windows(steps);
        Rng::with_seed(seed).shuffle(&mut windows);
        Self {
            store,
            windows,
            steps,
        }
    }
}

impl Dataset<StockItem> for WindowDataset {
    fn get(&self, index: usize) -> Option<StockItem> {
        let &(ticker_index, start) = self.windows.get(index)?;
        let (technical, industry, label, reward) =
            self.store.window(ticker_index, start, self.steps);
        Some(StockItem {
            technical,
            industry,
            label,
            reward,
        })
    }

    fn len(&self) -> usize {
        self.windows.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn len_counts_every_window() {
        let store = Arc::new(TickerStore::synthetic(3, 20));
        let dataset = WindowDataset::new(store, 4);

        // Each ticker yields 20 - 4 + 1 = 17 windows across 3 tickers.
        assert_eq!(dataset.len(), 17 * 3);
    }

    #[test]
    fn get_materializes_a_window_item() {
        let store = Arc::new(TickerStore::synthetic(2, 10).set_industries(vec![1, 0], 2));
        let dataset = WindowDataset::new(store, 4);

        let item = dataset.get(0).unwrap();

        // A window of 4 steps over 5 features is 20 floats.
        assert_eq!(item.technical.len(), 4 * 5);
        // First window is the first ticker, whose industry was set to 1.
        assert_eq!(item.industry, 1);
        // Labels cycle 0/1/2; the window's last day is row index 3 -> label 0.
        assert_eq!(item.label, 0);
        // Synthetic row `i` fills every slot with `base + i`; the first ticker's
        // base is 0, so the last row's slots pass through unscaled as 3.0.
        assert!((item.technical[3 * 5 + 3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn subsample_order_is_reproducible() {
        let first =
            WindowDataset::subsample(Arc::new(TickerStore::synthetic(4, 20)), 4, 99).windows;
        let again =
            WindowDataset::subsample(Arc::new(TickerStore::synthetic(4, 20)), 4, 99).windows;
        assert_eq!(first, again);
    }
}
