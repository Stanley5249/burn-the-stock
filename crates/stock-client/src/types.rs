use serde::{Deserialize, Serialize};

// sim server types

#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    pub result: String,
    pub status: String,
}

/// One entry from the sim server's [`stock_list`] endpoint.
#[derive(Debug, Deserialize)]
pub struct StockInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
}

/// One position as returned by the sim server's [`get_user_stocks`] endpoint.
#[derive(Debug, Deserialize)]
pub struct UserStock {
    pub usid: u64,
    pub stock_name: String,
    pub stock_code_id: String,
    pub shares: u64,
    /// Decimal string, e.g. `"344.00000"`. Parse with `str::parse::<f64>()` when needed.
    pub beginning_price: String,
    pub createtime: i64,
    pub user_uid_id: u64,
}

// OHLCV row shared by all market sources

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvRow {
    pub date: String,
    pub stock_code_id: String,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub change: Option<f64>,
    pub capacity: f64,
    pub turnover: f64,
    pub transaction_volume: f64,
}

// market type

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Market {
    Twse,
    Tpex,
    Esb,
}
