use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use url::Url;

use crate::urls::fugle as urls;
use miette::{IntoDiagnostic, Result, WrapErr};

/// Give up on a rate-limited symbol after this many backoff retries.
const MAX_QUOTE_RETRIES: u32 = 5;

/// Ceiling on the 429 backoff pace. Raise if the rate limit tightens.
const MAX_QUOTE_DELAY_MS: u64 = 8_000;

/// Fugle market data client. Owns a `reqwest::Client` carrying the `X-API-KEY` header.
pub struct FugleClient {
    pub client: reqwest::Client,
}

impl FugleClient {
    /// Build a client from `FUGLE_API_KEY`. The key rides on the `X-API-KEY` header; no cookie
    /// store, since Fugle is stateless.
    ///
    /// # Errors
    /// If the env var is missing, the key is not a valid header value, or the client fails to
    /// build.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("FUGLE_API_KEY")
            .into_diagnostic()
            .wrap_err("FUGLE_API_KEY must be set")?;

        let value = HeaderValue::from_str(&api_key)
            .into_diagnostic()
            .wrap_err("invalid api key")?;

        let mut headers = HeaderMap::new();
        // `from_static` panics on uppercase; header names are case-insensitive on the wire.
        headers.insert(HeaderName::from_static("x-api-key"), value);

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .into_diagnostic()
            .wrap_err("build fugle client")?;

        Ok(Self { client })
    }

    /// Fetch a quote per symbol, sequential and paced by `delay`. On a rate-limit (429) it
    /// doubles the pace up to [`MAX_QUOTE_DELAY_MS`] and retries the same symbol, so the rest of
    /// the batch self-tunes to the limit and coverage stays full. A non-rate-limit failure is
    /// logged and skipped, and a symbol still limited after [`MAX_QUOTE_RETRIES`] backoffs is
    /// given up on.
    #[tracing::instrument(skip_all, fields(symbols = symbols.len(), delay))]
    pub async fn quotes(&self, symbols: &[String], mut delay: u64) -> HashMap<String, FugleQuote> {
        let mut quotes = HashMap::with_capacity(symbols.len());

        for symbol in symbols {
            let url = match quote_url(symbol) {
                Ok(url) => url,
                Err(error) => {
                    tracing::warn!(%symbol, %error, "bad quote url");
                    continue;
                }
            };

            let mut retries = 0;
            loop {
                match self.request_quote(url.clone()).await {
                    Ok(quote) => {
                        quotes.insert(symbol.clone(), quote);
                        break;
                    }
                    Err(error)
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

    /// Fetch the static metadata for a single `symbol`.
    ///
    /// # Errors
    /// Network or deserialization failure.
    pub async fn ticker(&self, symbol: &str) -> Result<FugleTickerDetail> {
        let url = urls::base()
            .join(&format!("{}/{symbol}", urls::INTRADAY_TICKER))
            .into_diagnostic()
            .wrap_err("build ticker url")?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .into_diagnostic()?
            .error_for_status()
            .into_diagnostic()?
            .json()
            .await
            .into_diagnostic()
            .wrap_err("decode ticker")?;

        Ok(response)
    }

    /// Inner quote request kept at the reqwest level so [`quotes`](Self::quotes) can read the
    /// HTTP status for the 429 backoff.
    async fn request_quote(&self, url: Url) -> reqwest::Result<FugleQuote> {
        self.client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
    }
}

fn quote_url(symbol: &str) -> Result<Url> {
    urls::base()
        .join(&format!("{}/{symbol}", urls::INTRADAY_QUOTE))
        .into_diagnostic()
        .wrap_err("build quote url")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_url_appends_symbol() {
        assert_eq!(
            quote_url("2330").unwrap().as_str(),
            "https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/2330"
        );
    }
}
