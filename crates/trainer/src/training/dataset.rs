use burn::data::dataset::Dataset;
use fastrand::Rng;
use stock_model::data::{StockItem, TickerFrames};

/// A [`Dataset`] over every `steps`-length window of a [`TickerFrames`]. Each window is
/// resolved at construction, so the dataset owns a flat `Vec<StockItem>` and `get` is a
/// cheap lookup off the store's hot path.
pub struct WindowDataset {
    windows: Vec<StockItem>,
}

impl WindowDataset {
    /// Index every window of the store in ticker-then-date order.
    pub fn new(store: &TickerFrames, steps: usize) -> Self {
        Self {
            windows: store.enumerate_windows(steps),
        }
    }

    /// Like [`Self::new`] but shuffle once with `seed`, so a `PartialDataset` cap yields
    /// a validation subsample drawn evenly across tickers and dates.
    pub fn subsample(store: &TickerFrames, steps: usize, seed: u64) -> Self {
        let mut windows = store.enumerate_windows(steps);
        Rng::with_seed(seed).shuffle(&mut windows);
        Self { windows }
    }
}

impl Dataset<StockItem> for WindowDataset {
    fn get(&self, index: usize) -> Option<StockItem> {
        self.windows.get(index).copied()
    }

    fn len(&self) -> usize {
        self.windows.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic;

    #[test]
    fn len_counts_every_window() {
        let store = synthetic(3, 20);
        let dataset = WindowDataset::new(&store, 4);

        // 17 windows per ticker, 3 tickers.
        assert_eq!(dataset.len(), 17 * 3);
    }

    #[test]
    fn get_maps_index_to_window() {
        let store = synthetic(2, 10);
        let dataset = WindowDataset::new(&store, 4);

        assert_eq!(
            (
                dataset.get(0).unwrap().ticker,
                dataset.get(0).unwrap().start
            ),
            (0, 0)
        );

        // 7 windows per 10-row ticker, so index 7 is the second ticker's first window.
        let seventh = dataset.get(7).unwrap();
        assert_eq!((seventh.ticker, seventh.start), (1, 0));
    }

    #[test]
    fn subsample_order_is_reproducible() {
        let first = WindowDataset::subsample(&synthetic(4, 20), 4, 99).windows;
        let again = WindowDataset::subsample(&synthetic(4, 20), 4, 99).windows;
        assert_eq!(first, again);
    }
}
