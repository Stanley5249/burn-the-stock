//! The OHLCV parquet schema: the single source of truth for its column names.
//! `stock-model` re-exports the columns its feature transform reads.

use polars::prelude::PlSmallStr;

pub const CODE: PlSmallStr = PlSmallStr::from_static("code");
pub const DATE: PlSmallStr = PlSmallStr::from_static("date");
pub const OPEN: PlSmallStr = PlSmallStr::from_static("open");
pub const HIGH: PlSmallStr = PlSmallStr::from_static("high");
pub const LOW: PlSmallStr = PlSmallStr::from_static("low");
pub const CLOSE: PlSmallStr = PlSmallStr::from_static("close");
pub const VOLUME: PlSmallStr = PlSmallStr::from_static("volume");
pub const MARKET: PlSmallStr = PlSmallStr::from_static("market");

/// Default on-disk location of the consolidated history parquet.
pub const DEFAULT_PATH: &str = "data/yfinance/stock_history.parquet";

/// First bar to fetch when the history parquet does not exist yet.
pub const DEFAULT_FLOOR: &str = "2016-01-01";
