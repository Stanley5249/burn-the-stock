//! Shared core for the stock bot: the model architecture, the feature transform,
//! and the inference path. Both binaries depend on this so training and live
//! trading turn prices into model inputs the same way, and run the same model.
//!
//! - [`features`] standardizes raw OHLCV into the model's input window.
//! - [`model`] is the GRU classifier architecture and its forward pass.
//! - [`inference`] loads a trained model and scores feature windows.

pub mod features;
pub mod inference;
pub mod model;
