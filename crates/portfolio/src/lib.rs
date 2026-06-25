//! Stateful long-only portfolio simulation under the sim stock rules. The engine
//! walks one trading day at a time over a pre-built [`TradingDay`] stream, realizing
//! profit only on a sell. Pure: signals and prices in, a [`BacktestReport`] out.
//!
//! Rules: 100M starting cash, equal weight across at most ten holdings, whole
//! 1,000-share lots, buys at the day's low and sells at the high (snapped to a legal
//! tick in range), 0.1425% commission per side, 0.3% sell-only tax.

mod engine;
mod metrics;
mod pricing;
mod report;
mod types;

pub use engine::{affordable_shares, run};
pub use pricing::{LOT, SELL_TAX_RATE, commission, sell_price, tick_ceil};
pub use report::{RenderContext, summary};
pub use types::{BacktestConfig, BacktestReport, DayBar, ExitReason, Fill, TradingDay, Weighting};

/// The platform's simulated starting balance.
pub const STARTING_CASH: f64 = 100_000_000.0;
