//! The burn-coupled training pipeline: the data batcher, the window dataset, the
//! trade-aware metrics, the training wrapper around the shared model, and the loop
//! that ties them together. Only the loop and its config are public; the rest are
//! internal to this pipeline.

mod batcher;
mod dataset;
mod metric;
mod model;
mod runner;

pub use runner::{RunOptions, TrainingConfig, train};
