use miette::{IntoDiagnostic, Result, WrapErr};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

/// Build a reqwest client. Set `sim_stock_login` to enable the cookie store the sim stock
/// login needs. Pass `fugle_api_key` to send the Fugle `X-API-KEY` header.
///
/// # Errors
/// If the api key is not a valid header value, or the client fails to build.
pub fn default_client(
    sim_stock_login: bool,
    fugle_api_key: Option<&str>,
) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();

    if let Some(fugle_api_key) = fugle_api_key {
        let value = HeaderValue::from_str(fugle_api_key)
            .into_diagnostic()
            .wrap_err("invalid api key")?;

        // `from_static` panics on uppercase; header names are case-insensitive on the wire.
        headers.insert(HeaderName::from_static("x-api-key"), value);
    }

    reqwest::Client::builder()
        .default_headers(headers)
        .cookie_store(sim_stock_login)
        .build()
        .into_diagnostic()
        .wrap_err("failed to build reqwest client")
}
