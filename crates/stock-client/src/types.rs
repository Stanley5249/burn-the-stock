use chrono::NaiveDate;
use rust_decimal::Decimal;
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
    pub kind: MarketType,
}

/// One position as returned by the sim server's [`get_user_stocks`] endpoint.
#[derive(Debug, Deserialize)]
pub struct UserStock {
    pub usid: u64,
    pub stock_name: String,
    pub stock_code_id: String,
    pub shares: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub beginning_price: Decimal,
    pub createtime: i64,
    pub user_uid_id: u64,
}

// OHLCV row shared by all market sources

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvRow {
    pub date: NaiveDate,
    pub stock_code_id: String,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub change: Option<f64>,
    pub capacity: u64,
    pub turnover: u64,
    pub transaction_volume: u64,
}

/// Market label as returned by the sim server's `/stock_type` endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum MarketType {
    #[serde(rename = "TWSE")]
    Twse,
    #[serde(rename = "ETF")]
    Etf,
    #[serde(rename = "OTC")]
    Otc,
    #[serde(rename = "ESB")]
    Esb,
}

impl std::fmt::Display for MarketType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketType::Twse => formatter.write_str("TWSE"),
            MarketType::Etf => formatter.write_str("ETF"),
            MarketType::Otc => formatter.write_str("OTC"),
            MarketType::Esb => formatter.write_str("ESB"),
        }
    }
}

/// Canonical market used to select which OHLCV data API to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiMarket {
    Twse,
    Tpex,
    Esb,
}

impl From<MarketType> for ApiMarket {
    fn from(market_type: MarketType) -> Self {
        match market_type {
            MarketType::Twse | MarketType::Etf => ApiMarket::Twse,
            MarketType::Otc => ApiMarket::Tpex,
            MarketType::Esb => ApiMarket::Esb,
        }
    }
}
