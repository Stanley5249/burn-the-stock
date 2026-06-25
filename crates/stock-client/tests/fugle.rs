use std::sync::LazyLock;
use stock_client::fugle::FugleClient;

mod common;

static CLIENT: LazyLock<FugleClient> = LazyLock::new(common::fugle_client);

#[tokio::test]
#[ignore = "requires network access, FUGLE_API_KEY, and an open market session"]
async fn test_fugle_quote_tsmc() {
    let quotes = CLIENT.quotes(&["2330".to_string()], 0).await;
    let quote = quotes.get("2330").expect("quote for 2330");

    tracing::info!(
        symbol = quote.symbol,
        open = ?quote.open_price,
        high = ?quote.high_price,
        low = ?quote.low_price,
        last = ?quote.last_price,
        "TSMC quote"
    );

    assert_eq!(quote.symbol, "2330");
    // During a session the high is at or above the low.
    if let (Some(high), Some(low)) = (quote.high_price, quote.low_price) {
        assert!(high >= low, "high below low");
    }
}
