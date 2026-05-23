use crate::error::{Error, Result};
use crate::urls;
use chrono::NaiveDate;
use serde::Deserialize;

/// Maximum number of days per historical candles request (API limit is ~1 year).
pub const CANDLE_CHUNK_DAYS: i64 = 364;

// --- Public API ---

/// Fetch the list of equity tickers for `market`.
///
/// # Errors
///
/// Returns an error on network or deserialization failure.
pub async fn fetch_tickers(
    http: &reqwest::Client,
    market: FugleMarket,
) -> Result<Vec<FugleTickerItem>> {
    let response = http
        .get(urls::FUGLE_INTRADAY_TICKERS)
        .query(&[
            ("type", "EQUITY"),
            ("exchange", market.exchange()),
            ("market", market.as_str()),
        ])
        .send()
        .await?
        .error_for_status()?
        .json::<FugleTickersResponse>()
        .await?;
    Ok(response.data)
}

/// Fetch adjusted daily candles for `symbol` over `[from, to]`.
///
/// The endpoint caps each request at ~1 year, so this function issues multiple
/// requests when the range is longer and concatenates the bars.
/// Bars are returned in ascending date order.
///
/// # Errors
///
/// Returns an error if `from > to`, or on network or deserialization failure.
pub async fn fetch_candles(
    http: &reqwest::Client,
    symbol: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<FugleCandlesResponse> {
    if from > to {
        return Err(Error::InvalidRow(format!(
            "from ({from}) is after to ({to})"
        )));
    }

    let first_to = (from + chrono::Duration::days(CANDLE_CHUNK_DAYS)).min(to);
    let mut result = fetch_candles_chunk(http, symbol, from, first_to).await?;
    let mut chunk_from = first_to + chrono::Duration::days(1);

    while chunk_from <= to {
        let chunk_to = (chunk_from + chrono::Duration::days(CANDLE_CHUNK_DAYS)).min(to);
        let chunk = fetch_candles_chunk(http, symbol, chunk_from, chunk_to).await?;
        result.data.extend(chunk.data);
        chunk_from = chunk_to + chrono::Duration::days(1);
    }

    Ok(result)
}

/// Fetch adjusted daily candles for `symbol` over a single `[from, to]` window.
///
/// The window must be at most [`CANDLE_CHUNK_DAYS`] days. For longer ranges use
/// [`fetch_candles`], which handles chunking automatically.
///
/// # Errors
///
/// Returns an error on network or deserialization failure.
pub async fn fetch_candles_chunk(
    http: &reqwest::Client,
    symbol: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<FugleCandlesResponse> {
    let url = format!("{}/{}", urls::FUGLE_HISTORICAL_CANDLES, symbol);
    Ok(http
        .get(&url)
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
        .json::<FugleCandlesResponse>()
        .await?)
}

// --- Market enum ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct FugleTickersResponse {
    pub date: String,
    pub data: Vec<FugleTickerItem>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FugleTickerItem {
    pub symbol: String,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FugleCandlesResponse {
    pub symbol: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub exchange: String,
    pub market: String,
    pub timeframe: String,
    pub data: Vec<FugleCandleBar>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FugleCandleBar {
    pub date: NaiveDate,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    pub turnover: Option<f64>,
    pub change: Option<f64>,
}
