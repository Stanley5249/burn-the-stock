use std::sync::LazyLock;
use stock_client::sim_stock::SimStockClient;

mod common;

static CLIENT: LazyLock<SimStockClient> = LazyLock::new(common::sim_client);

#[tokio::test]
#[ignore = "requires network access and credentials"]
async fn test_user_stocks_schema() {
    let stocks = CLIENT.user_stocks().await.unwrap();
    tracing::info!(count = stocks.len(), "user stocks");
    if let Some(first) = stocks.first() {
        tracing::info!(stock = ?first, "first user stock");
    }
}

#[tokio::test]
#[ignore = "requires network access and credentials"]
async fn test_profile_schema() {
    // login is stateful; the profile scrape needs the session it establishes.
    CLIENT.login().await.unwrap();
    let profile = CLIENT.profile().await.unwrap();
    tracing::info!(profile = ?profile, "profile");
    assert!(profile.usable_cash > 0.0, "usable cash should be positive");
    assert!(
        profile.total_assets >= profile.usable_cash,
        "total assets should cover usable cash"
    );
}
