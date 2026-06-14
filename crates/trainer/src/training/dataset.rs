use crate::data::store::TickerStore;
use burn::data::dataset::Dataset;
use fastrand::Rng;

/// One training sample, reduced to the absolute row where its window starts in the
/// store's concatenated row numbering. The batcher gathers the feature window,
/// label, and reward from its resident device tensors using this index, so the item
/// carries no feature data and stays backend-free.
#[derive(Clone, Copy, Debug)]
pub struct StockItem {
    /// Absolute row of the window's first day. The window spans `steps` consecutive
    /// rows from here, and the label and reward are read at its last day.
    pub start: u32,
}

/// A [`Dataset`] over every `steps`-length window of a [`TickerStore`].
///
/// Each `(ticker_index, window_start)` from the store is resolved once at
/// construction into the single absolute start row the batcher gathers against, so
/// the dataset owns only a flat `Vec<u32>` and borrows nothing. `get` is then a
/// pointer-cheap lookup, which lets burn's data loader hand indices to the batcher
/// without touching the store on the hot path.
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

    /// Like [`Self::new`] but shuffle the window order once with `seed`. Pairing
    /// this with a [`burn::data::dataset::transform::PartialDataset`] cap yields
    /// a validation subsample drawn evenly across tickers and dates rather than
    /// biased to the earliest ones, and stable across epochs.
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

        // Each ticker yields 20 - 4 + 1 = 17 windows across 3 tickers.
        assert_eq!(dataset.len(), 17 * 3);
    }

    #[test]
    fn get_maps_window_to_absolute_start() {
        let store = TickerStore::synthetic(2, 10);
        let dataset = WindowDataset::new(&store, 4);

        // The first ticker's first window starts at absolute row 0.
        assert_eq!(dataset.get(0).unwrap().start, 0);

        // Each 10-row ticker yields 10 - 4 + 1 = 7 windows, so window index 7 is the
        // second ticker's first window, whose absolute start is its row offset 10.
        assert_eq!(dataset.get(7).unwrap().start, 10);
    }

    #[test]
    fn subsample_order_is_reproducible() {
        let first = WindowDataset::subsample(&TickerStore::synthetic(4, 20), 4, 99).windows;
        let again = WindowDataset::subsample(&TickerStore::synthetic(4, 20), 4, 99).windows;
        assert_eq!(first, again);
    }
}
