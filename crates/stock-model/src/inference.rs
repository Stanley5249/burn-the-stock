use burn::config::Config;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use polars::prelude::Series;

use crate::class::{Action, NUM_CLASSES, SELL};
use crate::data::{Window, gather_windows};
use crate::model::{StockModel, StockModelConfig};

/// Windows per forward pass, capping device memory on a universe-wide scoring run.
const SCORE_CHUNK: usize = 1024;

/// The inference slice of a run's config. `Config` ignores the extra training-only
/// fields, so this loads from the same `config.json`.
#[derive(Config, Debug)]
pub struct InferenceConfig {
    pub model: StockModelConfig,
    pub steps: usize,
}

/// One window's model output, with no trading policy applied. Aligned by index to the
/// `technical` rows passed to [`predict`], so the caller recovers ticker and date.
pub struct Prediction {
    /// Class probabilities in model order: Sell, Hold, Buy.
    pub probabilities: [f32; NUM_CLASSES],
    /// The argmax class.
    pub action: Action,
}

/// Score an already-windowed `[rows, steps, NUM_FEATURES]` batch: forward, softmax, and
/// argmax each row. The returned predictions align by index with the input rows. The
/// caller chunks to bound GPU memory, since this holds the whole batch at once.
///
/// # Panics
/// If the model output is not `f32` or a row does not hold `NUM_CLASSES` values.
#[must_use]
pub fn predict<B: Backend>(model: &StockModel<B>, technical: Tensor<B, 3>) -> Vec<Prediction> {
    let rows = technical.dims()[0];
    let probabilities = softmax(model.forward(technical), 1);

    // One host transfer; softmax is monotonic so its argmax agrees with the pre-softmax
    // argmax without a second device op.
    let flat = probabilities
        .into_data()
        .to_vec::<f32>()
        .expect("softmax output is f32");

    let mut predictions = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * NUM_CLASSES;
        let probabilities: [f32; NUM_CLASSES] = flat[base..base + NUM_CLASSES]
            .try_into()
            .expect("one row holds NUM_CLASSES probabilities");

        // Scans low to high and only replaces on a strict improvement, so a tie keeps
        // the earlier class, matching burn's argmax.
        let mut class = SELL;
        for (candidate, &probability) in probabilities.iter().enumerate().skip(1) {
            if probability > probabilities[class] {
                class = candidate;
            }
        }

        predictions.push(Prediction {
            probabilities,
            action: Action::from_class(class).expect("argmax is below NUM_CLASSES"),
        });
    }

    predictions
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
    device: &B::Device,
) -> Vec<Prediction> {
    let mut predictions = Vec::with_capacity(windows.len());

    for chunk in windows.chunks(SCORE_CHUNK) {
        let items: Vec<_> = chunk.iter().map(Window::item).collect();

        let technical = gather_windows::<B>(features, &items, steps, device);

        predictions.extend(predict(model, technical));
    }
    predictions
}
