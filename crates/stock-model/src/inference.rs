use burn::config::Config;
use burn::prelude::*;
use polars::prelude::Series;

use crate::data::{Window, gather_windows};
use crate::model::{StockModel, StockModelConfig};

/// Default GPU batch: windows per inference forward pass, and tickers per training batch. A
/// run's `config.json` can override it, so inference follows the value the model trained with.
pub const DEFAULT_BATCH_SIZE: usize = 4096;

/// The inference slice of a run's config. `Config` ignores the extra training-only
/// fields, so this loads from the same `config.json`.
#[derive(Config, Debug)]
pub struct InferenceConfig {
    pub model: StockModelConfig,
    pub steps: usize,
    /// Windows per forward pass. Read from the run's config when present, so scoring matches
    /// the training batch; otherwise [`DEFAULT_BATCH_SIZE`].
    #[config(default = "DEFAULT_BATCH_SIZE")]
    pub batch_size: usize,
}

/// One window's model output, with no trading policy applied. Aligned by index to the
/// `technical` rows passed to [`predict`], so the caller recovers ticker and date.
pub struct Prediction {
    /// The predicted per-date z-scored MFE: higher means a stronger relative pick.
    pub score: f32,
}

/// Score an already-windowed `[rows, steps, NUM_FEATURES]` batch: one forward, one score
/// per row. The returned predictions align by index with the input rows. The caller
/// chunks to bound GPU memory, since this holds the whole batch at once.
///
/// # Panics
/// If the model output is not `f32`.
#[must_use]
pub fn predict<B: Backend>(model: &StockModel<B>, technical: Tensor<B, 3>) -> Vec<Prediction> {
    let flat = model
        .forward(technical)
        .into_data()
        .to_vec::<f32>()
        .expect("model output is f32");

    flat.into_iter().map(|score| Prediction { score }).collect()
}

/// Gather each window's features and score it, chunked to bound device memory. The
/// returned predictions align by index with `windows`. The one inference path shared by
/// the backtest and the live trader.
///
/// # Panics
/// If a feature series is not contiguous `f32`, which [`crate::data::TickerFrames::feature_series`] guarantees.
#[must_use]
pub fn score<B: Backend>(
    model: &StockModel<B>,
    features: &[Series],
    windows: &[Window],
    steps: usize,
    batch_size: usize,
    device: &B::Device,
) -> Vec<Prediction> {
    let mut predictions = Vec::with_capacity(windows.len());

    for chunk in windows.chunks(batch_size) {
        let items: Vec<_> = chunk.iter().map(Window::item).collect();

        let technical = gather_windows::<B>(features, &items, steps, device);

        predictions.extend(predict(model, technical));
    }
    predictions
}
