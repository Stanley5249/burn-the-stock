//! One module per CLI subcommand, each owning the orchestration that turns parsed
//! args into a run: the train loop, the held-out eval, or the portfolio backtest.

pub mod backtest;
pub mod eval;
pub mod train;
