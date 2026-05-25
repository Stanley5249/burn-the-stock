use reqwest::header::{HeaderMap, HeaderValue};
use std::sync::LazyLock;
use stock_client::sim_stock::SimStockClient;

static CLIENT: LazyLock<SimStockClient> = LazyLock::new(|| {
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

    SimStockClient::from_env(client).expect("`STOCK_ACCOUNT` and `STOCK_PASSWORD` must be set")
});

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
