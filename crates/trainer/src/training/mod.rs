//! The burn-coupled training pipeline: batcher, window dataset, trade-aware metrics,
//! training wrapper, and the loop tying them together, plus the `latest` link. Only
//! the loop, its config, and the link are public.

mod batcher;
mod dataset;
pub mod link;
mod metric;
mod model;
mod pipeline;

pub use pipeline::{RunOptions, TrainingConfig, train};
