//! The burn-coupled training pipeline: the data batcher, the window dataset, the
//! trade-aware metrics, the training wrapper around the shared model, and the loop
//! that ties them together, plus the `latest` link the train command refreshes. The
//! loop, its config, and the `latest` link are public; the rest are internal to this
//! pipeline.

mod batcher;
mod dataset;
pub mod link;
mod metric;
mod model;
mod pipeline;

pub use pipeline::{RunOptions, TrainingConfig, train};
