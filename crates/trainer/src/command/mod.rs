//! One module per CLI subcommand, each owning the orchestration that turns parsed
//! args into a run: the train loop, or an offline prediction.

pub mod predict;
pub mod train;
