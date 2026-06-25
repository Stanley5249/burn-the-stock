//! Turn rankings, holdings, and live quotes into concrete sell and buy orders.

use std::collections::HashMap;

use portfolio::{
    DayBar, Fill, LOT, SELL_TAX_RATE, affordable_shares, commission, sell_price, tick_ceil,
};
use stock_client::fugle::FugleQuote;
use stock_client::types::UserStock;

/// A planned exit: sell the whole position at the quote high. `lots` is in 張 (the platform
/// order unit), `price` is per share.
pub struct Sell {
    pub code: String,
    pub lots: u64,
    pub price: f64,
    pub proceeds: f64,
}

/// A planned entry: buy at the quote low. `lots` is in 張, `price` is per share.
pub struct Buy {
    pub code: String,
    pub lots: u64,
    pub price: f64,
    pub cost: f64,
}

/// Plan the day's exits: sell every holding whole at the quote high to harvest the daily
/// buy-low/sell-high spread. Holdings without a usable quote are left alone.
pub fn plan_sells(holdings: &[UserStock], quotes: &HashMap<String, FugleQuote>) -> Vec<Sell> {
    let mut sells = Vec::new();
    for holding in holdings {
        let Some(quote) = quotes.get(&holding.stock_code_id) else {
            continue;
        };
        let Some(bar) = quote_to_bar(quote) else {
            continue;
        };

        let price = sell_price(&bar, Fill::LowHigh);
        // holding.shares is in 張; the traded value is per-share price times actual shares.
        #[allow(
            clippy::cast_precision_loss,
            reason = "lot counts are small whole numbers"
        )]
        let amount = price * holding.shares as f64 * LOT;
        let proceeds = amount - commission(amount) - amount * SELL_TAX_RATE;
        sells.push(Sell {
            code: holding.stock_code_id.clone(),
            lots: holding.shares,
            price,
            proceeds,
        });
    }
    sells
}

/// Size buys equal-weight over the budget, filling the open slots with the strongest quoted
/// candidates. Each name targets `budget / max_holdings`, matching the backtest engine. Cash
/// drops as each fills, so later names get what is left.
pub fn plan_buys(
    candidates: &[(String, f32)],
    quotes: &HashMap<String, FugleQuote>,
    budget: f64,
    max_holdings: usize,
) -> Vec<Buy> {
    let priced: Vec<(&String, f64)> = candidates
        .iter()
        .filter_map(|(ticker, _)| {
            let quote = quotes.get(ticker)?;
            let low = quote.low_price.or(quote.open_price)?;
            Some((ticker, tick_ceil(low)))
        })
        .collect();

    #[allow(
        clippy::cast_precision_loss,
        reason = "slot count is a small whole number"
    )]
    let target = budget / max_holdings.max(1) as f64;

    let mut remaining = budget;
    let mut buys = Vec::new();
    for (code, price) in priced {
        let shares = affordable_shares(target.min(remaining), price, remaining);
        if shares <= 0.0 {
            continue;
        }
        let amount = price * shares;
        let cost = amount + commission(amount);
        remaining -= cost;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "share/LOT is a small whole nonnegative lot count"
        )]
        let lots = (shares / LOT).round() as u64;
        buys.push(Buy {
            code: code.clone(),
            lots,
            price,
            cost,
        });
    }
    buys
}

/// Build a one-day bar from a live quote, or `None` when a price is still missing before the
/// first trade of the session.
#[allow(clippy::cast_possible_truncation, reason = "TWSE prices fit f32")]
fn quote_to_bar(quote: &FugleQuote) -> Option<DayBar> {
    Some(DayBar {
        score: 0.0,
        open: quote.open_price? as f32,
        low: quote.low_price? as f32,
        high: quote.high_price? as f32,
        close: quote.last_price.or(quote.open_price)? as f32,
    })
}
