use chrono::NaiveDate;
use std::sync::LazyLock;
use stock_client::client::StockClient;
use stock_client::market_data::fetch_stock_data;
use stock_client::types::{ApiMarket, MarketType, OhlcvRow};

static CLIENT: LazyLock<StockClient> = LazyLock::new(|| {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    StockClient::from_env().expect("`STOCK_ACCOUNT` and `STOCK_PASSWORD` must be set")
});

fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
}

fn assert_ohlcv_rows(rows: &[OhlcvRow], start: NaiveDate, end: NaiveDate) {
    assert!(!rows.is_empty(), "expected at least one row");

    for row in rows {
        assert!(
            row.date >= start && row.date <= end,
            "date out of range: {}",
            row.date
        );
        assert!(row.capacity.is_finite(), "capacity not finite");
        assert!(row.turnover.is_finite(), "turnover not finite");
        assert!(
            row.transaction_volume.is_finite(),
            "transaction_volume not finite"
        );
        if let Some(v) = row.open {
            assert!(v.is_finite(), "open not finite");
        }
        if let Some(v) = row.high {
            assert!(v.is_finite(), "high not finite");
        }
        if let Some(v) = row.low {
            assert!(v.is_finite(), "low not finite");
        }
        if let Some(v) = row.close {
            assert!(v.is_finite(), "close not finite");
        }
        if let Some(v) = row.change {
            assert!(v.is_finite(), "change not finite");
        }
    }
}

// --- trading API ---

#[tokio::test]
#[ignore = "requires network access"]
async fn test_stock_list_schema() {
    let stocks = CLIENT.stock_list().await.unwrap();
    let (sample_code, sample_info) = stocks.iter().next().unwrap();
    tracing::info!(
        total = stocks.len(),
        code = sample_code,
        name = %sample_info.name,
        kind = %sample_info.kind,
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

// --- market data ---

#[tokio::test]
#[ignore = "requires network access"]
async fn test_fetch_twse_schema() {
    let start = date("2024-03-01");
    let end = date("2024-03-31");
    let rows = fetch_stock_data(CLIENT.http(), "2330", start, end, ApiMarket::Twse)
        .await
        .unwrap();
    tracing::info!(count = rows.len(), "twse rows");
    assert_ohlcv_rows(&rows, start, end);
}

#[tokio::test]
#[ignore = "requires network access"]
async fn test_fetch_tpex_schema() {
    let start = date("2024-03-01");
    let end = date("2024-03-31");

    let stocks = CLIENT.stock_list().await.unwrap();
    let code = stocks
        .iter()
        .find(|(_, info)| info.kind == MarketType::Otc)
        .map(|(code, _)| code.clone())
        .expect("no OTC stock in list");

    let rows = fetch_stock_data(CLIENT.http(), &code, start, end, ApiMarket::Tpex)
        .await
        .unwrap();
    tracing::info!(count = rows.len(), code, "tpex rows");
    assert_ohlcv_rows(&rows, start, end);
}

#[tokio::test]
#[ignore = "requires network access"]
async fn test_fetch_esb_schema() {
    let start = date("2024-01-01");
    let end = date("2024-12-31");

    let stocks = CLIENT.stock_list().await.unwrap();
    let esb_codes: Vec<_> = stocks
        .iter()
        .filter(|(_, info)| info.kind == MarketType::Esb)
        .map(|(code, _)| code.clone())
        .take(20)
        .collect();

    assert!(!esb_codes.is_empty(), "no ESB stocks in list");

    for code in &esb_codes {
        let rows = fetch_stock_data(CLIENT.http(), code, start, end, ApiMarket::Esb)
            .await
            .unwrap();
        if !rows.is_empty() {
            tracing::info!(count = rows.len(), code, "esb rows");
            assert_ohlcv_rows(&rows, start, end);
            return;
        }
    }

    panic!(
        "no ESB data found for any of {} candidates in {start}..{end}",
        esb_codes.len()
    );
}
