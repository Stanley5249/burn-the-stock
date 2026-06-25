use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer};

/// One position as returned by the sim server's `get_user_stocks` endpoint.
#[derive(Debug, Deserialize)]
pub struct UserStock {
    pub usid: u64,
    pub stock_name: String,
    pub stock_code_id: String,
    /// Position size in 張 (board lots of 1,000 shares), the platform's unit.
    pub shares: u64,
    #[serde(deserialize_with = "f64_from_str")]
    pub beginning_price: f64,
    #[serde(with = "chrono::serde::ts_seconds")]
    pub createtime: DateTime<Utc>,
    pub user_uid_id: u64,
}

/// The sim server sends prices as quoted strings like "34.40000", so parse to f64.
fn f64_from_str<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let raw = String::deserialize(deserializer)?;
    raw.parse().map_err(serde::de::Error::custom)
}

/// The account dashboard's headline numbers, scraped from the sim server's `profile/` page.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Spendable cash (可用餘額), already net of unsettled proceeds.
    pub usable_cash: f64,
    /// Total account value (資產總額): cash plus the marked value of holdings.
    pub total_assets: f64,
    /// Cumulative return (累積報酬率) as a percentage, e.g. `1.906` for 1.906%.
    pub cumulative_return: f64,
    /// Count of successful trades (交易成功次數).
    pub trade_count: u64,
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
