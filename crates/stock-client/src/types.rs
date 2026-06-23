use chrono::NaiveDate;
use miette::{Result, bail};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// One position as returned by the sim server's `get_user_stocks` endpoint.
#[derive(Debug, Deserialize)]
pub struct UserStock {
    pub usid: u64,
    pub stock_name: String,
    pub stock_code_id: String,
    /// Position size in 張 (board lots of 1,000 shares), the platform's unit.
    pub shares: u64,
    #[serde(with = "rust_decimal::serde::str")]
    pub beginning_price: Decimal,
    pub createtime: i64,
    pub user_uid_id: u64,
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

/// OHLCV row shared by all market sources.
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

impl OhlcvRow {
    /// # Errors
    /// If `stock_code_id` is empty or any price field is negative.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        date: NaiveDate,
        stock_code_id: String,
        open: Option<f64>,
        high: Option<f64>,
        low: Option<f64>,
        close: Option<f64>,
        change: Option<f64>,
        capacity: u64,
        turnover: u64,
        transaction_volume: u64,
    ) -> Result<Self> {
        if stock_code_id.is_empty() {
            bail!("empty stock_code_id");
        }
        for (name, value) in [
            ("open", open),
            ("high", high),
            ("low", low),
            ("close", close),
        ] {
            if value.is_some_and(f64::is_sign_negative) {
                bail!("{name} is negative");
            }
        }
        Ok(Self {
            date,
            stock_code_id,
            open,
            high,
            low,
            close,
            change,
            capacity,
            turnover,
            transaction_volume,
        })
    }
}

/// One symbol from the sim server's `stock_list` universe.
#[derive(Debug, Clone)]
pub struct StockListEntry {
    pub code: String,
    pub market_type: MarketType,
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
