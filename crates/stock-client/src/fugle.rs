use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use tokio_retry::RetryIf;
use tokio_retry::strategy::ExponentialBackoff;
use url::Url;

use crate::urls::fugle as urls;
use miette::{IntoDiagnostic, Result, WrapErr};

/// Give up on a rate-limited symbol after this many backoff retries.
pub const QUOTE_RETRIES: usize = 5;

/// Ceiling on the 429 backoff pace. Raise if the rate limit tightens.
pub const QUOTE_MAX_DELAY_MS: u64 = 8_000;

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
    /// doubles the pace up to [`QUOTE_MAX_DELAY_MS`] and retries the same symbol, so the rest of
    /// the batch self-tunes to the limit and coverage stays full. A non-rate-limit failure is
    /// logged and skipped, and a symbol still limited after [`QUOTE_RETRIES`] backoffs is
    /// given up on.
    #[tracing::instrument(skip_all, fields(symbols = symbols.len(), delay, fetched, priced))]
    pub async fn quotes(&self, symbols: &[String], mut delay: u64) -> HashMap<String, FugleQuote> {
        let mut quotes = HashMap::with_capacity(symbols.len());

        for (index, symbol) in symbols.iter().enumerate() {
            let url = match quote_url(symbol) {
                Ok(url) => url,
                Err(error) => {
                    tracing::warn!(%symbol, %error, "bad quote url");
                    continue;
                }
            };

            // Seed the per-symbol backoff from the current pace, then grow that pace on each 429
            // so the rest of the batch inherits the raised delay and self-tunes to the limit.
            let result = RetryIf::start(
                ExponentialBackoff::from_millis(2)
                    .factor(delay)
                    .max_delay(Duration::from_millis(QUOTE_MAX_DELAY_MS))
                    .take(QUOTE_RETRIES),
                || {
                    let url = url.clone();
                    async move { self.request_quote(url).await }
                },
                |error: &reqwest::Error| {
                    let limited = error.status() == Some(StatusCode::TOO_MANY_REQUESTS);
                    if limited {
                        delay = delay.saturating_mul(2).min(QUOTE_MAX_DELAY_MS);
                        tracing::warn!(%symbol, delay, "Fugle rate limit, backing off");
                    }
                    limited
                },
            )
            .await;

            match result {
                Ok(quote) => {
                    // Log each landing so the long sweep streams live; priced=false is a null
                    // quote (no session price yet).
                    tracing::info!(
                        %symbol,
                        done = index + 1,
                        priced = quote.open_price.is_some(),
                        last = ?quote.last_price,
                        "quote",
                    );
                    quotes.insert(symbol.clone(), quote);
                }
                Err(error) => {
                    tracing::warn!(%symbol, %error, "quote failed");
                }
            }
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }

        // priced counts quotes that have an open price, which means the session has started.
        let priced = quotes
            .values()
            .filter(|quote| quote.open_price.is_some())
            .count();
        let span = tracing::Span::current();
        span.record("fetched", quotes.len());
        span.record("priced", priced);

        quotes
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
