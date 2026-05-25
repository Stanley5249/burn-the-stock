use crate::error::Result;
use crate::urls;
use chrono::NaiveDate;
use serde::Deserialize;

// --- Public API ---

/// Fetch the list of equity tickers for `market`.
///
/// # Errors
///
/// Returns an error on network or deserialization failure.
pub async fn fetch_tickers(
    http: &reqwest::Client,
    market: FugleMarket,
) -> Result<FugleTickersResponse> {
    let response: FugleTickersResponse = http
        .get(urls::FUGLE_INTRADAY_TICKERS)
        .query(&[
            ("type", "EQUITY"),
            ("exchange", market.exchange()),
            ("market", market.as_str()),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(response)
}

/// Fetch adjusted daily candles for `symbol` over a single `[from, to]` window.
///
/// The window must be at most [`CANDLE_CHUNK_DAYS`] days.
///
/// # Errors
///
/// Returns an error on network or deserialization failure.
pub async fn fetch_candles(
    http: &reqwest::Client,
    symbol: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<FugleCandlesResponse> {
    let url = format!("{}/{}", urls::FUGLE_HISTORICAL_CANDLES, symbol);

    let response = http
        .get(url)
        .query(&[
            ("timeframe", "D"),
            ("adjusted", "true"),
            ("sort", "asc"),
            ("from", from.to_string().as_str()),
            ("to", to.to_string().as_str()),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(response)
}

// --- Market enum ---

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum FugleMarket {
    Tse,
    Otc,
    Esb,
    Tib,
    Psb,
}

impl FugleMarket {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            FugleMarket::Tse => "TSE",
            FugleMarket::Otc => "OTC",
            FugleMarket::Esb => "ESB",
            FugleMarket::Tib => "TIB",
            FugleMarket::Psb => "PSB",
        }
    }

    #[must_use]
    pub fn exchange(self) -> &'static str {
        match self {
            FugleMarket::Tse => "TWSE",
            FugleMarket::Otc | FugleMarket::Esb | FugleMarket::Tib | FugleMarket::Psb => "TPEx",
        }
    }
}

impl std::fmt::Display for FugleMarket {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

// --- Response types ---

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FugleTickersResponse {
    pub date: String,
    pub r#type: String,
    pub exchange: String,
    pub market: Option<String>,
    #[serde(rename = "isNormal")]
    pub is_normal: Option<bool>,
    #[serde(rename = "isAttention")]
    pub is_attention: Option<bool>,
    #[serde(rename = "isDisposition")]
    pub is_disposition: Option<bool>,
    #[serde(rename = "isHalted")]
    pub is_halted: Option<bool>,
    pub data: Vec<FugleTickerItem>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FugleTickerItem {
    pub symbol: String,
    pub name: String,
    pub industry: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FugleCandlesResponse {
    pub symbol: String,
    pub r#type: String,
    pub exchange: String,
    pub market: String,
    pub timeframe: String,
    pub sort: Option<String>,
    pub adjusted: Option<bool>,
    pub data: Vec<FugleCandleBar>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FugleCandleBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
    pub turnover: Option<f64>,
    pub change: Option<f64>,
}
