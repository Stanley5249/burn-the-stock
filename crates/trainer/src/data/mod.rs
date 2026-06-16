//! The backend-free data layer shared by training and backtest: the `TickerStore`
//! that loads and splits the OHLCV history, and the triple-barrier labeling. Neither
//! touches burn.

pub mod label;
pub mod store;
