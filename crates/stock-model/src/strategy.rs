//! Trading policy applied to the model's probabilities. The edge formula is plain
//! math over a probability row, so the live trader and the backtest score signals the
//! same way without pulling in the inference machinery.

use burn::config::Config;

use crate::class::{BUY, NUM_CLASSES, SELL};

/// Long-only expected edge for one ticker, `clamp(P(Buy)*take_profit -
/// P(Sell)*stop_loss, 0)`. Zero stays flat, since a Sell only vetoes a Buy in a market
/// that cannot short.
#[must_use]
pub fn expected_edge(probabilities: &[f32; NUM_CLASSES], take_profit: f32, stop_loss: f32) -> f32 {
    (probabilities[BUY] * take_profit - probabilities[SELL] * stop_loss).max(0.0)
}

/// The strategy slice of a run's config. `Config` ignores the extra fields, so this
/// loads the barriers from the same `config.json` the trainer writes.
#[derive(Config, Debug)]
pub struct StrategyConfig {
    pub take_profit: f32,
    pub stop_loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_nets_buy_against_sell() {
        // P(Buy) 0.6 earns 0.10, P(Sell) 0.3 risks 0.05, so 0.06 - 0.015 = 0.045.
        let probabilities = [0.3, 0.1, 0.6];
        let edge = expected_edge(&probabilities, 0.10, 0.05);
        assert!((edge - 0.045).abs() < 1e-6);
    }

    #[test]
    fn edge_floors_at_zero() {
        // A dominant Sell drives the raw edge negative, so it clamps to flat.
        let probabilities = [0.8, 0.1, 0.1];
        assert!(expected_edge(&probabilities, 0.10, 0.05).abs() < 1e-6);
    }
}
