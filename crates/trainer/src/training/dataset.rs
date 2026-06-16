use crate::data::store::TickerStore;
use burn::data::dataset::Dataset;
use fastrand::Rng;

/// One training sample, reduced to the absolute store row where its window starts.
/// The batcher gathers the window and label from this index, so the item carries
/// no feature data and stays backend-free.
#[derive(Clone, Copy, Debug)]
pub struct StockItem {
    /// Absolute row of the window's first day; it spans `steps` rows from here.
    pub start: u32,
}

/// A [`Dataset`] over every `steps`-length window of a [`TickerStore`]. Each window
/// is resolved at construction into its absolute start row, so the dataset owns only
/// a flat `Vec<u32>` and `get` is a cheap lookup off the store's hot path.
pub struct WindowDataset {
    /// Absolute start row of every window, in ticker-then-date order.
    windows: Vec<u32>,
}

impl WindowDataset {
    /// Index every window of the store in ticker-then-date order.
    pub fn new(store: &TickerStore, steps: usize) -> Self {
        Self {
            windows: Self::absolute_starts(store, steps),
        }
    }

    /// Like [`Self::new`] but shuffle once with `seed`, so a `PartialDataset` cap
    /// yields a validation subsample drawn evenly across tickers and dates.
    pub fn subsample(store: &TickerStore, steps: usize, seed: u64) -> Self {
        let mut windows = Self::absolute_starts(store, steps);
        Rng::with_seed(seed).shuffle(&mut windows);
        Self { windows }
    }

    /// Resolve every `(ticker_index, start)` window into its absolute start row via
    /// the store's row-offset table.
    fn absolute_starts(store: &TickerStore, steps: usize) -> Vec<u32> {
        let offsets = store.row_offsets();
        store
            .enumerate_windows(steps)
            .into_iter()
            .map(|(ticker_index, start)| offsets[ticker_index as usize] + start)
            .collect()
    }
}

impl Dataset<StockItem> for WindowDataset {
    fn get(&self, index: usize) -> Option<StockItem> {
        self.windows
            .get(index)
            .copied()
            .map(|start| StockItem { start })
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
        let store = TickerStore::synthetic(3, 20);
        let dataset = WindowDataset::new(&store, 4);

        // 17 windows per ticker, 3 tickers.
        assert_eq!(dataset.len(), 17 * 3);
    }

    #[test]
    fn get_maps_window_to_absolute_start() {
        let store = TickerStore::synthetic(2, 10);
        let dataset = WindowDataset::new(&store, 4);

        assert_eq!(dataset.get(0).unwrap().start, 0);

        // 7 windows per 10-row ticker, so index 7 is the second ticker's first, at
        // offset 10.
        assert_eq!(dataset.get(7).unwrap().start, 10);
    }

    #[test]
    fn subsample_order_is_reproducible() {
        let first = WindowDataset::subsample(&TickerStore::synthetic(4, 20), 4, 99).windows;
        let again = WindowDataset::subsample(&TickerStore::synthetic(4, 20), 4, 99).windows;
        assert_eq!(first, again);
    }
}
