use super::types::{DayBar, Fill};

/// Shares per Taiwan lot. Counts stay f64 lot-multiples to avoid integer casts.
pub(super) const LOT: f64 = 1_000.0;
/// Commission charged on each buy and sell.
const COMMISSION_RATE: f64 = 0.001_425;
/// Minimum commission per transaction.
const MIN_COMMISSION: f64 = 20.0;
/// Securities transaction tax, charged on sells only.
pub(super) const SELL_TAX_RATE: f64 = 0.003;

/// Commission on a trade of `amount`, with the per-transaction floor.
pub(super) fn commission(amount: f64) -> f64 {
    (amount * COMMISSION_RATE).max(MIN_COMMISSION)
}

/// Round-trip cost as a fraction: commission on both legs plus the sell tax. The edge
/// gain a rotation must clear to be worth the churn.
pub(super) fn round_trip_cost() -> f64 {
    2.0 * COMMISSION_RATE + SELL_TAX_RATE
}

/// Tick size in NT$ for a price, by the platform's price bands.
fn tick_size(price: f64) -> f64 {
    if price < 10.0 {
        0.01
    } else if price < 50.0 {
        0.05
    } else if price < 100.0 {
        0.10
    } else if price < 500.0 {
        0.50
    } else if price < 1000.0 {
        1.00
    } else {
        5.00
    }
}

/// Largest legal tick `<= price`; the epsilon absorbs float error.
pub(super) fn tick_floor(price: f64) -> f64 {
    let tick = tick_size(price);
    ((price / tick) + 1e-9).floor() * tick
}

/// Smallest legal tick price `>= price`.
fn tick_ceil(price: f64) -> f64 {
    let tick = tick_size(price);
    ((price / tick) - 1e-9).ceil() * tick
}

/// Buy fill: lowest legal tick at or above the day's low (or open).
pub(super) fn buy_price(bar: &DayBar, fill: Fill) -> f64 {
    match fill {
        Fill::LowHigh => tick_ceil(f64::from(bar.low)),
        Fill::Open => tick_ceil(f64::from(bar.open)),
    }
}

/// Sell fill: highest legal tick at or below the day's high (or open).
pub(super) fn sell_price(bar: &DayBar, fill: Fill) -> f64 {
    match fill {
        Fill::LowHigh => tick_floor(f64::from(bar.high)),
        Fill::Open => tick_floor(f64::from(bar.open)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_rounding_matches_platform_examples() {
        // Buy rounds down, sell rounds up, across the price bands.
        assert!((tick_floor(8.256) - 8.25).abs() < 1e-9);
        assert!((tick_ceil(3.765) - 3.77).abs() < 1e-9);
        assert!((tick_floor(34.271) - 34.25).abs() < 1e-9);
        assert!((tick_ceil(66.256) - 66.30).abs() < 1e-9);
        assert!((tick_floor(499.831) - 499.50).abs() < 1e-9);
        assert!((tick_ceil(499.831) - 500.00).abs() < 1e-9);
        assert!((tick_floor(1456.256) - 1455.0).abs() < 1e-9);
        assert!((tick_ceil(1456.256) - 1460.0).abs() < 1e-9);
    }
}
