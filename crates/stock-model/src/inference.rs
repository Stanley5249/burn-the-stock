use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use chrono::NaiveDate;
use miette::{IntoDiagnostic, Result};

use crate::features::{FEATURE_NAMES, InferenceWindow};
use crate::model::{NUM_CLASSES, StockModel, StockModelConfig};

/// Class indices in the model's output order, matching the labeler's Sell/Hold/Buy.
const SELL: usize = 0;
const BUY: usize = 2;

/// Windows per forward pass, capping GPU memory on a universe-wide backtest.
const BATCH_SIZE: usize = 1024;

/// The inference slice of a run's config. `Config` ignores the extra training-only
/// fields, so this loads from the same `config.json`.
#[derive(Config, Debug)]
pub struct InferenceConfig {
    pub model: StockModelConfig,
    pub steps: usize,
}

/// Predicted action for one ticker, the argmax of the model's class probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Sell,
    Hold,
    Buy,
}

impl Action {
    fn from_class(class: usize) -> Self {
        match class {
            SELL => Action::Sell,
            BUY => Action::Buy,
            _ => Action::Hold,
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Action::Sell => "Sell",
            Action::Hold => "Hold",
            Action::Buy => "Buy",
        }
    }
}

/// One ticker's inference result.
pub struct Prediction {
    pub ticker: String,
    /// The bar this prediction was made from.
    pub date: NaiveDate,
    /// Class probabilities in model order: Sell, Hold, Buy.
    pub probabilities: [f32; NUM_CLASSES],
    pub action: Action,
    /// Long-only soft position, `clamp(P(Buy) - P(Sell), 0)`. Zero stays flat, since
    /// a Sell only vetoes a Buy in a market that cannot short.
    pub position: f32,
}

/// A trained model bound to a device, carrying the `steps` it was trained with.
pub struct Predictor<B: Backend> {
    model: StockModel<B>,
    steps: usize,
    device: B::Device,
}

impl<B: Backend> Predictor<B> {
    /// Load the trained model and its config from `artifact_dir`. On a plain backend
    /// dropout is inert, so the forward pass is inference.
    ///
    /// # Errors
    /// If the config or model file is missing or cannot be read.
    pub fn load(artifact_dir: &Path, device: B::Device) -> Result<Self> {
        let config = InferenceConfig::load(artifact_dir.join("config.json")).into_diagnostic()?;

        let model = config
            .model
            .init::<B>(&device)
            .load_file(artifact_dir.join("model"), &CompactRecorder::new(), &device)
            .into_diagnostic()?;

        Ok(Self {
            model,
            steps: config.steps,
            device,
        })
    }

    /// Window length the model expects. Build [`Self::predict`] windows this long.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Score every window, in [`BATCH_SIZE`] chunks. Each window holds `steps * 5`
    /// features, as produced by [`crate::features::latest_windows`].
    ///
    /// # Panics
    /// If a window's feature length does not match `steps * 5`.
    #[must_use]
    pub fn predict(&self, windows: &[InferenceWindow]) -> Vec<Prediction> {
        let width = FEATURE_NAMES.len();
        let mut predictions = Vec::with_capacity(windows.len());

        for chunk in windows.chunks(BATCH_SIZE) {
            let mut features = Vec::with_capacity(chunk.len() * self.steps * width);
            for window in chunk {
                assert_eq!(
                    window.features.len(),
                    self.steps * width,
                    "window feature length does not match the model's steps"
                );
                features.extend_from_slice(&window.features);
            }

            let technical = Tensor::<B, 3>::from_data(
                TensorData::new(features, [chunk.len(), self.steps, width]),
                &self.device,
            );

            let probabilities = softmax(self.model.forward(technical), 1);

            // One host transfer per chunk.
            let flat = probabilities
                .into_data()
                .to_vec::<f32>()
                .expect("softmax output is f32");

            for (row, window) in chunk.iter().enumerate() {
                let offset = row * NUM_CLASSES;
                let probabilities: [f32; NUM_CLASSES] = flat[offset..offset + NUM_CLASSES]
                    .try_into()
                    .expect("one row holds NUM_CLASSES probabilities");

                predictions.push(Prediction {
                    ticker: window.ticker.clone(),
                    date: window.date,
                    probabilities,
                    action: Action::from_class(argmax(&probabilities)),
                    position: (probabilities[BUY] - probabilities[SELL]).max(0.0),
                });
            }
        }

        predictions
    }
}

/// Index of the largest probability; ties resolve to the lower index.
fn argmax(probabilities: &[f32]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}
