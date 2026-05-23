use chrono::NaiveDate;
use reqwest::header::{HeaderMap, HeaderValue};
use std::sync::LazyLock;
use stock_client::client::StockClient;
use stock_client::market_data::{FugleMarket, fetch_candles, fetch_tickers};

static CLIENT: LazyLock<StockClient> = LazyLock::new(|| {
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

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .expect("failed to build reqwest client");

    StockClient::from_env(client).expect("`STOCK_ACCOUNT` and `STOCK_PASSWORD` must be set")
});

fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}

// --- Trading API ---

#[tokio::test]
#[ignore = "requires network access"]
async fn test_stock_list_schema() {
    let stocks = CLIENT.stock_list().await.unwrap();
    let (code, stock_info) = stocks.iter().next().unwrap();
    tracing::info!(
        total = stocks.len(),
        first.code = code,
        first.stock_info = ?stock_info,
        "stock list"
    );
}

#[tokio::test]
#[ignore = "requires network access"]
async fn test_stock_market_schema() {
    let market = CLIENT.stock_market("2330").await.unwrap();
    tracing::info!(market = ?market, "stock market");
}

#[tokio::test]
#[ignore = "requires network access and credentials"]
async fn test_user_stocks_schema() {
    let stocks = CLIENT.user_stocks().await.unwrap();
    tracing::info!(count = stocks.len(), "user stocks");
    if let Some(first) = stocks.first() {
        tracing::info!(stock = ?first, "first user stock");
    }
}

// --- Fugle market data ---

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_tickers_tse() {
    let tickers = fetch_tickers(CLIENT.http(), FugleMarket::Tse)
        .await
        .unwrap();

    tracing::info!(count = tickers.len(), "TSE tickers");

    assert!(
        tickers.len() > 500,
        "expected hundreds of TSE stocks, got {}",
        tickers.len()
    );

    let tsmc = tickers.iter().find(|t| t.symbol == "2330");

    assert!(tsmc.is_some(), "TSMC (2330) not found in TSE tickers");

    tracing::info!(name = tsmc.unwrap().name, "TSMC");
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_tickers_otc() {
    let tickers = fetch_tickers(CLIENT.http(), FugleMarket::Otc)
        .await
        .unwrap();

    tracing::info!(count = tickers.len(), "OTC tickers");

    assert!(!tickers.is_empty(), "expected at least one OTC ticker");
}

#[tokio::test]
#[ignore = "requires network access and FUGLE_API_KEY"]
async fn test_fugle_candles_tsmc() {
    let from = date("2024-01-01");
    let to = date("2024-12-31");
    let response = fetch_candles(CLIENT.http(), "2330", from, to)
        .await
        .unwrap();

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
    assert!(first.close.is_some(), "expected close price");
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
    let from = date("2016-01-01");
    let to = chrono::Local::now().date_naive();
    let response = fetch_candles(CLIENT.http(), "2330", from, to)
        .await
        .unwrap();

    tracing::info!(bars = response.data.len(), "TSMC 10-year candles");

    assert!(
        response.data.len() > 1000,
        "expected >1000 daily bars, got {}",
        response.data.len()
    );
}
