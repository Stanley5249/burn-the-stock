use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use miette::{IntoDiagnostic, Result};

use crate::class::{Action, NUM_CLASSES, SELL};
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
/// `starts` passed to [`Predictor::predict`], so the caller recovers ticker and date.
pub struct Prediction {
    /// Class probabilities in model order: Sell, Hold, Buy.
    pub probabilities: [f32; NUM_CLASSES],
    /// The argmax class.
    pub action: Action,
}

/// A trained model bound to a device, carrying the `steps` it trained with.
pub struct Predictor<B: Backend> {
    pub model: StockModel<B>,
    pub steps: usize,
    pub device: B::Device,
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

    /// Score every window, in [`BATCH_SIZE`] chunks. `features` is the resident
    /// `[rows, NUM_FEATURES]` tensor and `starts` holds each window's absolute first
    /// row; each chunk slices `starts` and gathers on-device via [`gather_windows`], so
    /// the start indices upload once. The returned predictions align by index with
    /// `starts`.
    ///
    /// # Panics
    /// If any start plus `steps` exceeds the feature rows, an out-of-range gather.
    #[must_use]
    pub fn predict(&self, features: &Tensor<B, 2>, starts: &Tensor<B, 1, Int>) -> Vec<Prediction> {
        let total = starts.dims()[0];
        let mut predictions = Vec::with_capacity(total);

        let mut chunk_start = 0;
        while chunk_start < total {
            let chunk_end = (chunk_start + BATCH_SIZE).min(total);
            let count = chunk_end - chunk_start;

            let chunk_starts = starts.clone().slice(chunk_start..chunk_end);
            let technical = gather_windows(features, &chunk_starts, self.steps);

            let logits = self.model.forward(technical);
            let probabilities = softmax(logits, 1);

            // One host transfer per chunk; the action is this row's argmax, and
            // softmax is monotonic so it agrees with the pre-softmax argmax without
            // a second device op and its own host transfer.
            let flat = probabilities
                .into_data()
                .to_vec::<f32>()
                .expect("softmax output is f32");

            for row in 0..count {
                let base = row * NUM_CLASSES;
                let probabilities: [f32; NUM_CLASSES] = flat[base..base + NUM_CLASSES]
                    .try_into()
                    .expect("one row holds NUM_CLASSES probabilities");

                // Scans low to high and only replaces on a strict improvement, so a
                // tie keeps the earlier class, matching burn's argmax.
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

            chunk_start = chunk_end;
        }

        predictions
    }
}

/// Gather `steps`-length windows from a resident `[rows, NUM_FEATURES]` feature
/// tensor. Each window start in `starts` broadcasts over `0..steps` into
/// absolute row indices, then one on-device `select` reads them into `[count,
/// steps, NUM_FEATURES]`. Shared by the live predictor and the trainer's
/// batcher so the index math has one source.
///
/// # Panics
/// If `steps` exceeds `i64`, a window far longer than supported.
#[must_use]
pub fn gather_windows<B: Backend>(
    features: &Tensor<B, 2>,
    starts: &Tensor<B, 1, Int>,
    steps: usize,
) -> Tensor<B, 3> {
    let device = features.device();
    let count = starts.dims()[0];

    // `[count, steps]` row indices: each window start broadcast over `0..steps`.
    let offsets = Tensor::<B, 1, Int>::arange(
        0..i64::try_from(steps)
            .expect("steps exceeds i64; window length far larger than supported"),
        &device,
    );
    let index = starts.clone().reshape([count, 1]).expand([count, steps])
        + offsets.reshape([1, steps]).expand([count, steps]);

    features
        .clone()
        .select(0, index.reshape([count * steps]))
        .reshape([count, steps, NUM_FEATURES])
}
