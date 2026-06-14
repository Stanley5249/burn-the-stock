//! One module per CLI subcommand, each owning the orchestration that turns parsed
//! args into a run: the train loop, or a held-out backtest.

pub mod eval;
pub mod train;
