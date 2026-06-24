use std::time::Duration;

use chrono::{DateTime, Duration as ChronoDuration, NaiveDate};
use futures::stream::{self, StreamExt};
use miette::{IntoDiagnostic, Result, WrapErr, miette};
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use serde::Deserialize;
use url::Url;

use crate::urls::yahoo as urls;

/// Give up on a rate-limited symbol after this many backoff retries.
const MAX_CHART_RETRIES: u32 = 5;

/// First 429 backoff, doubled each retry.
const INITIAL_DELAY_MS: u64 = 500;

/// Ceiling on the 429 backoff pace. Raise if the rate limit tightens.
const MAX_CHART_DELAY_MS: u64 = 8_000;

/// In-flight chart requests; Yahoo is one symbol per request.
const CONCURRENCY: usize = 16;

/// Yahoo Finance chart client. Owns a cookie-storing `reqwest::Client` with a browser UA.
pub struct YahooClient {
    client: reqwest::Client,
}

/// Flat, unadjusted daily bars decoded from a chart response. Dates are local to the exchange.
/// `adjclose` is kept so the caller can fold the split/dividend adjustment into OHLC.
pub struct ChartBars {
    pub dates: Vec<NaiveDate>,
    pub open: Vec<Option<f64>>,
    pub high: Vec<Option<f64>>,
    pub low: Vec<Option<f64>>,
    pub close: Vec<Option<f64>>,
    pub volume: Vec<Option<f64>>,
    pub adjclose: Vec<Option<f64>>,
}

impl YahooClient {
    /// Build a cookie-storing client and prime the consent cookie.
    ///
    /// # Errors
    /// If the client fails to build.
    pub async fn new() -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static(urls::USER_AGENT));

        let client = reqwest::Client::builder()
            .cookie_store(true)
            .default_headers(headers)
            .build()
            .into_diagnostic()
            .wrap_err("build yahoo client")?;

        // chart API needs no crumb; add a cookie+crumb flow if Yahoo starts returning 401.
        let _ = client.get(urls::CONSENT).send().await;

        Ok(Self { client })
    }

    /// Fetch unadjusted daily bars for one `symbol`, named after the `/v8/finance/chart` endpoint.
    /// `end` is inclusive. On a 429 it doubles the backoff up to [`MAX_CHART_DELAY_MS`] and retries
    /// the same request.
    ///
    /// # Errors
    /// Network, deserialization, or empty/missing chart data.
    #[tracing::instrument(skip(self))]
    pub async fn chart(&self, symbol: &str, start: NaiveDate, end: NaiveDate) -> Result<ChartBars> {
        let url = chart_url(symbol, start, end)?;

        let mut delay = INITIAL_DELAY_MS;
        let mut retries = 0;
        let response = loop {
            match self.request_chart(url.clone()).await {
                Ok(response) => break response,
                Err(error)
                    if retries < MAX_CHART_RETRIES
                        && error.status() == Some(StatusCode::TOO_MANY_REQUESTS) =>
                {
                    retries += 1;
                    delay = delay.saturating_mul(2).min(MAX_CHART_DELAY_MS);
                    tracing::warn!(symbol, delay, retries, "Yahoo rate limit, backing off");
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                Err(error) => return Err(error).into_diagnostic().wrap_err("fetch chart"),
            }
        };

        bars(symbol, response)
    }

    /// Fetch many symbols concurrently. A failed symbol is logged and dropped, so the returned
    /// vec covers only the successes.
    #[tracing::instrument(skip_all, fields(symbols = symbols.len()))]
    pub async fn download(
        &self,
        symbols: &[String],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Vec<(String, ChartBars)> {
        stream::iter(symbols.iter().cloned())
            .map(|symbol| async move {
                match self.chart(&symbol, start, end).await {
                    Ok(bars) => Some((symbol, bars)),
                    Err(error) => {
                        tracing::warn!(%symbol, %error, "chart failed");
                        None
                    }
                }
            })
            .buffer_unordered(CONCURRENCY)
            .filter_map(|entry| async move { entry })
            .collect()
            .await
    }

    /// Inner request kept at the reqwest level so [`chart`](Self::chart) can read the HTTP status
    /// for the 429 backoff.
    async fn request_chart(&self, url: Url) -> reqwest::Result<ChartResponse> {
        self.client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
    }
}

fn chart_url(symbol: &str, start: NaiveDate, end: NaiveDate) -> Result<Url> {
    let mut url = Url::parse(&format!("{}{symbol}", urls::CHART_BASE))
        .into_diagnostic()
        .wrap_err("build chart url")?;

    // end is inclusive, period2 is exclusive, so push one day past it.
    let period1 = day_start_epoch(start);
    let period2 = day_start_epoch(end + ChronoDuration::days(1));

    url.query_pairs_mut()
        .append_pair("period1", &period1.to_string())
        .append_pair("period2", &period2.to_string())
        .append_pair("interval", "1d")
        .append_pair("events", "div|split|capitalGains")
        .append_pair("includePrePost", "false");

    Ok(url)
}

fn day_start_epoch(date: NaiveDate) -> i64 {
    date.and_hms_opt(0, 0, 0)
        .expect("midnight is always valid")
        .and_utc()
        .timestamp()
}

/// Decode the response into flat unadjusted columns: `date`, OHLC, `volume`, plus `adjclose` so
/// the caller can fold the adjustment in. Timestamps are shifted by the exchange offset so the
/// calendar date is local, not UTC.
fn bars(symbol: &str, response: ChartResponse) -> Result<ChartBars> {
    let result = response
        .chart
        .result
        .into_iter()
        .next()
        .ok_or_else(|| miette!("no chart data for {symbol}"))?;

    let quote = result
        .indicators
        .quote
        .into_iter()
        .next()
        .ok_or_else(|| miette!("no quote series for {symbol}"))?;
    let adjclose = result
        .indicators
        .adjclose
        .into_iter()
        .next()
        .map(|series| series.adjclose)
        .ok_or_else(|| miette!("no adjclose series for {symbol}"))?;

    let offset = result.meta.gmtoffset;
    let dates: Vec<NaiveDate> = result
        .timestamp
        .iter()
        .map(|ts| {
            DateTime::from_timestamp(ts + offset, 0)
                .map(|dt| dt.date_naive())
                .ok_or_else(|| miette!("bad timestamp {ts} for {symbol}"))
        })
        .collect::<Result<_>>()?;

    Ok(ChartBars {
        dates,
        open: quote.open,
        high: quote.high,
        low: quote.low,
        close: quote.close,
        volume: quote.volume,
        adjclose,
    })
}

#[derive(Deserialize)]
struct ChartResponse {
    chart: Chart,
}

#[derive(Deserialize)]
struct Chart {
    result: Vec<ResultData>,
}

#[derive(Deserialize)]
struct ResultData {
    meta: Meta,
    timestamp: Vec<i64>,
    indicators: Indicators,
}

#[derive(Deserialize)]
struct Meta {
    gmtoffset: i64,
}

#[derive(Deserialize)]
struct Indicators {
    quote: Vec<Quote>,
    adjclose: Vec<AdjClose>,
}

#[derive(Deserialize)]
struct Quote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct AdjClose {
    adjclose: Vec<Option<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_url_sets_range() {
        let url = chart_url(
            "2330.TW",
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
        )
        .unwrap();
        assert!(url.as_str().starts_with(
            "https://query1.finance.yahoo.com/v8/finance/chart/2330.TW?period1=1704067200"
        ));
        // period2 is one day past the inclusive end.
        assert!(url.as_str().contains("period2=1704240000"));
    }

    #[tokio::test]
    #[ignore = "hits live Yahoo Finance"]
    async fn live_chart_returns_bars() {
        let client = YahooClient::new().await.unwrap();
        let end = chrono::Local::now().date_naive();
        let start = end - ChronoDuration::days(30);
        let bars = client.chart("2330.TW", start, end).await.unwrap();
        assert!(!bars.dates.is_empty());
        assert!(bars.adjclose.iter().flatten().all(|value| *value > 0.0));
    }
}
