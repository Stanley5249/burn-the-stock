//! Shared test bootstrap: load `.env` and init tracing once, then build clients.

// Each integration-test binary includes this module but uses only some helpers.
#![allow(dead_code)]

use std::sync::LazyLock;

use stock_client::fugle::FugleClient;
use stock_client::sim_stock::SimStockClient;

static INIT: LazyLock<()> = LazyLock::new(|| {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
});

/// Fugle-keyed client (no cookie store) from env.
pub fn fugle_client() -> FugleClient {
    LazyLock::force(&INIT);
    FugleClient::from_env().expect("build fugle client")
}

/// sim stock client (cookie store) from env credentials.
pub fn sim_client() -> SimStockClient {
    LazyLock::force(&INIT);
    SimStockClient::from_env(None).expect("build sim client")
}
