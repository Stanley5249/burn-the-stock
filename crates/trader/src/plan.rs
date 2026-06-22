//! Turn rankings, holdings, and live quotes into concrete sell and buy orders.

use std::collections::HashMap;

use chrono::NaiveDate;
use portfolio::{
    DayBar, ExitReason, Fill, LOT, SELL_TAX_RATE, affordable_shares, commission, exit_decision,
    score_weights, sell_price, tick_ceil,
};
use stock_client::fugle::FugleQuote;
use stock_client::types::UserStock;

use crate::cli::Args;

/// A planned exit: sell the whole position at the quote high. `lots` is in 張 (the platform
/// order unit), `price` is per share.
pub struct Sell {
    pub code: String,
    pub lots: u64,
    pub price: f64,
    pub proceeds: f64,
    pub reason: ExitReason,
}

/// A planned entry: buy at the quote low. `lots` is in 張, `price` is per share.
pub struct Buy {
    pub code: String,
    pub lots: u64,
    pub price: f64,
    pub cost: f64,
}

/// Plan the day's exits, each sold whole at the quote high. By default (`exit_ladder` off)
/// every holding is sold to harvest the daily spread; with `exit_ladder` on, only holdings
/// the shared ladder flags exit. Holdings without a usable quote are left alone.
pub fn plan_sells(
    holdings: &[UserStock],
    quotes: &HashMap<String, FugleQuote>,
    score_of: &HashMap<String, f32>,
    args: &Args,
    today: NaiveDate,
) -> Vec<Sell> {
    let mut sells = Vec::new();
    for holding in holdings {
        let Some(quote) = quotes.get(&holding.stock_code_id) else {
            continue;
        };
        let score = score_of.get(&holding.stock_code_id).copied().unwrap_or(0.0);
        let Some(bar) = quote_to_bar(quote, score) else {
            continue;
        };

        let reason = if args.exit_ladder {
            let entry_price = holding
                .beginning_price
                .to_string()
                .parse::<f64>()
                .unwrap_or(0.0);
            let days = days_held(holding.createtime, today);
            match exit_decision(
                entry_price,
                days,
                &bar,
                args.take_profit,
                args.stop_loss,
                args.max_hold,
                Fill::LowHigh,
            ) {
                Some((_, reason)) => reason,
                None => continue,
            }
        } else {
            ExitReason::Rotate
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
            reason,
        });
    }
    sells
}

/// Size buys by score over the budget, filling the open slots with the strongest quoted
/// candidates. Cash drops as each fills, so later names get what is left.
pub fn plan_buys(
    candidates: &[(String, f32)],
    quotes: &HashMap<String, FugleQuote>,
    budget: f64,
    open_slots: usize,
) -> Vec<Buy> {
    let priced: Vec<(&String, f32, f64)> = candidates
        .iter()
        .filter_map(|(ticker, score)| {
            let quote = quotes.get(ticker)?;
            let low = quote.low_price.or(quote.open_price)?;
            Some((ticker, *score, tick_ceil(low)))
        })
        .take(open_slots)
        .collect();

    let weights = score_weights(
        &priced
            .iter()
            .map(|(_, score, _)| *score)
            .collect::<Vec<_>>(),
    );

    let mut remaining = budget;
    let mut buys = Vec::new();
    for ((code, _, price), weight) in priced.iter().zip(weights) {
        let target = budget * weight;
        let shares = affordable_shares(target.min(remaining), *price, remaining);
        if shares <= 0.0 {
            continue;
        }
        let amount = price * shares;
        let cost = amount + commission(amount);
        remaining -= cost;
        buys.push(Buy {
            code: (*code).clone(),
            lots: lots(shares),
            price: *price,
            cost,
        });
    }
    buys
}

/// Build a one-day bar from a live quote for the exit ladder, or `None` when a price is
/// still missing before the first trade of the session.
#[allow(clippy::cast_possible_truncation, reason = "TWSE prices fit f32")]
fn quote_to_bar(quote: &FugleQuote, score: f32) -> Option<DayBar> {
    Some(DayBar {
        score,
        open: quote.open_price? as f32,
        low: quote.low_price? as f32,
        high: quote.high_price? as f32,
        close: quote.last_price.or(quote.open_price)? as f32,
    })
}

/// Trading-day-agnostic days since entry, from the platform's epoch-second timestamp.
fn days_held(createtime: i64, today: NaiveDate) -> usize {
    let entry =
        chrono::DateTime::from_timestamp(createtime, 0).map_or(today, |moment| moment.date_naive());
    usize::try_from((today - entry).num_days().max(0)).unwrap_or(0)
}

/// Convert an actual share count from [`affordable_shares`] (a whole multiple of [`LOT`])
/// into the platform's order unit, 張 (board lots of 1,000 shares).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "share/LOT is a small whole nonnegative lot count"
)]
fn lots(shares: f64) -> u64 {
    (shares / LOT).round() as u64
}
