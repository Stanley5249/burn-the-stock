use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use miette::{IntoDiagnostic, Result};

use crate::class::{Action, NUM_CLASSES};
use crate::features::InferenceWindow;
use crate::model::{NUM_FEATURES, StockModel, StockModelConfig};

/// Windows per forward pass, capping GPU memory on a universe-wide backtest.
const BATCH_SIZE: usize = 1024;

/// The inference slice of a run's config. `Config` ignores the extra training-only
/// fields, so this loads from the same `config.json`.
#[derive(Config, Debug)]
pub struct InferenceConfig {
    pub model: StockModelConfig,
    pub steps: usize,
}

/// One row's model output, with no trading policy applied. Aligned by index to the
/// windows passed to [`Predictor::predict`], so the caller recovers ticker and date.
pub struct Prediction {
    /// Class probabilities in model order: Sell, Hold, Buy.
    pub probabilities: [f32; NUM_CLASSES],
    /// The argmax class.
    pub action: Action,
}

/// A trained model bound to a device, carrying the `steps` it trained with.
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

    /// Score every window, in [`BATCH_SIZE`] chunks. Each window holds `steps *
    /// NUM_FEATURES` features, as produced by [`crate::features::latest_windows`]. The
    /// returned rows align by index with `windows`.
    ///
    /// # Panics
    /// If a window's feature length does not match `steps * NUM_FEATURES`.
    #[must_use]
    pub fn predict(&self, windows: &[InferenceWindow]) -> Vec<Prediction> {
        let width = NUM_FEATURES;
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

            let logits = self.model.forward(technical);
            let probabilities = softmax(logits.clone(), 1);
            let classes = logits.argmax(1).reshape([chunk.len()]);

            // One host transfer per chunk for the probabilities and the argmax classes.
            let flat = probabilities
                .into_data()
                .to_vec::<f32>()
                .expect("softmax output is f32");
            let classes: Vec<i64> = classes.into_data().iter::<i64>().collect();

            for (row, class) in classes.into_iter().enumerate() {
                let offset = row * NUM_CLASSES;
                let probabilities: [f32; NUM_CLASSES] = flat[offset..offset + NUM_CLASSES]
                    .try_into()
                    .expect("one row holds NUM_CLASSES probabilities");

                let class = usize::try_from(class).expect("argmax index is non-negative");
                predictions.push(Prediction {
                    probabilities,
                    action: Action::from_class(class).expect("argmax is below NUM_CLASSES"),
                });
            }
        }

        predictions
    }
}
