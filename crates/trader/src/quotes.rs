//! Live Fugle quote fetching and conversion to a one-day bar.

use std::collections::HashMap;

use portfolio::DayBar;
use stock_client::fugle::{FugleQuote, fetch_quote};

/// Fetch a quote per symbol, sequential and rate-limited. A failed symbol is logged and
/// skipped rather than aborting the run.
pub async fn fetch_quotes(
    http: &reqwest::Client,
    symbols: &[String],
    delay_ms: u64,
) -> HashMap<String, FugleQuote> {
    let mut quotes = HashMap::with_capacity(symbols.len());
    for symbol in symbols {
        match fetch_quote(http, symbol).await {
            Ok(quote) => {
                quotes.insert(symbol.clone(), quote);
            }
            Err(error) => eprintln!("quote {symbol} failed: {error}"),
        }
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    quotes
}

/// Build a one-day bar from a live quote, or `None` when a price is still missing before
/// the first trade of the session.
#[allow(clippy::cast_possible_truncation, reason = "TWSE prices fit f32")]
pub fn quote_to_bar(quote: &FugleQuote, score: f32) -> Option<DayBar> {
    Some(DayBar {
        score,
        open: quote.open_price? as f32,
        low: quote.low_price? as f32,
        high: quote.high_price? as f32,
        close: quote.last_price.or(quote.open_price)? as f32,
    })
}
