//! Shared core for the stock bot: model architecture, feature transform, and
//! inference path. Both binaries depend on it so training and live trading turn
//! prices into the same model inputs.

pub mod features;
pub mod inference;
pub mod model;
