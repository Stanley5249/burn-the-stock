use chrono::NaiveDate;
use reqwest::header::{HeaderMap, HeaderValue};
use std::sync::LazyLock;
use stock_client::error::Error;
use stock_client::fugle::{FugleMarket, fetch_candles, fetch_ticker, fetch_tickers};

static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    dotenvy::dotenv().unwrap();

    let api_key = std::env::var("FUGLE_API_KEY").expect("`FUGLE_API_KEY` must be set");

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let mut headers = HeaderMap::new();
    headers.insert(
        "X-API-KEY",
        HeaderValue::from_str(&api_key).expect("invalid API key"),
    );

    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .expect("failed to build reqwest client")
});

fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_tickers_tse() {
    let response = fetch_tickers(&HTTP, FugleMarket::Tse).await.unwrap();
    let tickers = response.data;

    tracing::info!(count = tickers.len(), "TSE tickers");

    assert!(!tickers.is_empty(), "expected at least one TSE ticker");

    let tsmc = tickers.iter().find(|t| t.symbol == "2330");
    assert!(tsmc.is_some(), "TSMC (2330) not found in TSE tickers");
    tracing::info!(name = tsmc.unwrap().name, "TSMC");
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_tickers_otc() {
    let response = fetch_tickers(&HTTP, FugleMarket::Otc).await.unwrap();
    let tickers = response.data;

    tracing::info!(count = tickers.len(), "OTC tickers");

    assert!(!tickers.is_empty(), "expected at least one OTC ticker");
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_ticker_industry() {
    // A general stock and an ETF, both expected to carry an industry.
    for symbol in ["2330", "0050"] {
        let detail = fetch_ticker(&HTTP, symbol).await.unwrap();

        tracing::info!(
            symbol = detail.symbol,
            name = detail.name,
            industry = ?detail.industry,
            security_type = ?detail.security_type,
            "ticker detail"
        );

        assert_eq!(detail.symbol, symbol);

        let industry = detail.industry.as_deref().unwrap_or_default();
        assert!(!industry.is_empty(), "expected industry for {symbol}");
    }
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_candles_tsmc() {
    let from = date("2024-01-01");
    let to = date("2024-12-31");
    let response = fetch_candles(&HTTP, "2330", from, to).await.unwrap();

    tracing::info!(
        symbol = response.symbol,
        market = response.market,
        bars = response.data.len(),
        "TSMC candles"
    );

    assert_eq!(response.symbol, "2330");
    assert!(!response.data.is_empty(), "expected at least one candle");

    let first = &response.data[0];
    assert!(first.date >= from, "first bar date before requested from");
    assert!(first.close > 0.0, "expected close price");
    assert!(first.volume.is_some(), "expected volume");

    // Bars should be in ascending date order.
    for window in response.data.windows(2) {
        assert!(
            window[0].date < window[1].date,
            "candles not in ascending order"
        );
    }

    tracing::info!(
        date = %first.date,
        open = ?first.open,
        close = ?first.close,
        volume = ?first.volume,
        "first bar"
    );
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_candles_ten_years() {
    // The Fugle API rejects date ranges longer than 1 year with HTTP 400.
    let from = date("2016-01-01");
    let to = chrono::Local::now().date_naive();

    let error = fetch_candles(&HTTP, "2330", from, to).await.unwrap_err();

    tracing::info!(?error, "got expected error for >1-year range");

    match &error {
        Error::Http(error) => {
            let status = error.status();
            let is_4xx = status.is_some_and(|s| s.is_client_error());
            assert!(is_4xx, "expected 4xx status, got {status:?}");
        }
        other => panic!("expected HTTP error, got {other:?}"),
    }
}
