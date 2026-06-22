use reqwest::header::{HeaderMap, HeaderValue};

pub fn http_client() -> reqwest::Client {
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
}
