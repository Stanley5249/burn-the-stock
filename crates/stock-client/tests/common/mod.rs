//! Shared test bootstrap: load `.env` and init tracing once, then build clients.

// Each integration-test binary includes this module but uses only some helpers.
#![allow(dead_code)]

use std::sync::LazyLock;

use stock_client::client::default_client;
use stock_client::sim_stock::SimStockClient;

static INIT: LazyLock<()> = LazyLock::new(|| {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
});

/// Fugle-keyed client, no cookie store.
pub fn fugle_client() -> reqwest::Client {
    LazyLock::force(&INIT);
    let api_key = std::env::var("FUGLE_API_KEY").expect("FUGLE_API_KEY must be set");
    default_client(false, Some(&api_key)).expect("build fugle client")
}

/// `sim_stock` client (cookie store) from env credentials.
pub fn sim_client() -> SimStockClient {
    LazyLock::force(&INIT);
    SimStockClient::from_env(None, None).expect("build sim client")
}
