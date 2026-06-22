use crate::error::{Error, Result};
use crate::urls;
use chrono::NaiveDate;
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;

/// Build an HTTP client carrying the Fugle `X-API-KEY` header on every request. The same
/// client also drives the sim trading API, which ignores the extra header.
///
/// # Errors
/// If the key is not a valid header value or the client cannot be built.
pub fn client(api_key: &str) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert("X-API-KEY", HeaderValue::from_str(api_key)?);
    Ok(reqwest::Client::builder()
        .default_headers(headers)
        .build()?)
}

/// Give up on a rate-limited symbol after this many backoff retries.
const MAX_QUOTE_RETRIES: u32 = 5;

/// Ceiling on the 429 backoff pace. Raise if the rate limit tightens.
const MAX_QUOTE_DELAY_MS: u64 = 8_000;

/// Fetch a quote per symbol, sequential and paced by `delay`. On a rate-limit (429) it doubles
/// the pace up to [`MAX_QUOTE_DELAY_MS`] and retries the same symbol, so the rest of the batch
/// self-tunes to the limit and coverage stays full. A non-rate-limit failure is logged and
/// skipped, and a symbol still limited after [`MAX_QUOTE_RETRIES`] backoffs is given up on.
#[tracing::instrument(skip(http, symbols), fields(symbols = symbols.len()))]
pub async fn fetch_quotes(
    http: &reqwest::Client,
    symbols: &[String],
    mut delay: u64,
) -> HashMap<String, FugleQuote> {
    let mut quotes = HashMap::with_capacity(symbols.len());

    for symbol in symbols {
        let mut retries = 0;
        loop {
            match fetch_quote(http, symbol).await {
                Ok(quote) => {
                    quotes.insert(symbol.clone(), quote);
                    break;
                }
                Err(Error::Http(error))
                    if retries < MAX_QUOTE_RETRIES
                        && error.status() == Some(StatusCode::TOO_MANY_REQUESTS) =>
                {
                    retries += 1;
                    delay = delay.saturating_mul(2).min(MAX_QUOTE_DELAY_MS);
                    tracing::warn!(%symbol, delay, retries, "Fugle rate limit, backing off");
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                Err(error) => {
                    tracing::warn!(%symbol, %error, "quote failed");
                    break;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    quotes
}

/// Fetch the list of equity tickers for `market`.
///
/// # Errors
/// Network or deserialization failure.
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

/// Fetch the static metadata for a single `symbol`.
///
/// # Errors
/// Network or deserialization failure.
pub async fn fetch_ticker(http: &reqwest::Client, symbol: &str) -> Result<FugleTickerDetail> {
    let url = format!("{}/{}", urls::FUGLE_INTRADAY_TICKER, symbol);

    let response = http
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(response)
}

/// Fetch adjusted daily candles for `symbol` over `[from, to]`, at most
/// [`CANDLE_CHUNK_DAYS`] days.
///
/// # Errors
/// Network or deserialization failure.
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

/// Fetch the live intraday quote for `symbol`, the morning range and last trade the live
/// trader sets limit prices from.
///
/// # Errors
/// Network or deserialization failure.
pub async fn fetch_quote(http: &reqwest::Client, symbol: &str) -> Result<FugleQuote> {
    let url = format!("{}/{}", urls::FUGLE_INTRADAY_QUOTE, symbol);

    let response = http
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    Ok(response)
}

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

/// Static metadata for a single ticker. No `deny_unknown_fields`: the endpoint
/// returns extra warrant and index fields we do not model.
#[derive(Clone, Debug, Deserialize)]
pub struct FugleTickerDetail {
    pub symbol: String,
    pub name: String,
    pub r#type: String,
    pub exchange: String,
    pub market: String,
    pub industry: Option<String>,
    #[serde(rename = "securityType")]
    pub security_type: Option<String>,
}

/// Live intraday quote. No `deny_unknown_fields`: the endpoint returns deep bid/ask,
/// total, and last-trade objects we do not model. Prices are `Option` since they are
/// absent before the first trade of the session.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FugleQuote {
    pub symbol: String,
    pub name: Option<String>,
    pub open_price: Option<f64>,
    pub high_price: Option<f64>,
    pub low_price: Option<f64>,
    pub last_price: Option<f64>,
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
