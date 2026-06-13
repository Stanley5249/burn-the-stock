use polars::prelude::*;

/// Trade action paired with the signed realized return of its triple-barrier
/// outcome. A Buy carries `+take_profit` and a Sell `-stop_loss`, the payoff of
/// exiting a long at the touched barrier, while a Hold carries the realized
/// close-to-close move at the vertical barrier. The reward is the per-sample
/// payoff the expected-value metric scores a Buy against.
///
/// The data-carrying variants rule out an explicit discriminant, so [`class`]
/// defines the 0/1/2 order the model's output index relies on.
///
/// [`class`]: Label::class
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Label {
    Sell(f32),
    Hold(f32),
    Buy(f32),
}

impl Label {
    /// Class index in the model's output order: Sell 0, Hold 1, Buy 2.
    pub fn class(self) -> u8 {
        match self {
            Label::Sell(_) => 0,
            Label::Hold(_) => 1,
            Label::Buy(_) => 2,
        }
    }

    /// Signed realized return of the labeled barrier outcome.
    pub fn reward(self) -> f32 {
        match self {
            Label::Sell(reward) | Label::Hold(reward) | Label::Buy(reward) => reward,
        }
    }
}

/// Label each row with the triple-barrier outcome of opening a long at its close.
///
/// For row `t` the entry is `close[t]`, with a take-profit barrier at
/// `entry * (1 + take_profit)` and a stop-loss barrier at `entry * (1 - stop_loss)`.
/// The next `horizon` bars are scanned in order and the first barrier touched wins:
/// the intraday high crossing take-profit is a [`Label::Buy`], the intraday low
/// crossing stop-loss is a [`Label::Sell`]. When one bar touches both, the
/// intraday order is unknown, so it falls back to the vertical-barrier
/// [`Label::Hold`]. A row whose price stays inside both barriers for the whole
/// horizon is also a Hold, closed at the vertical barrier.
///
/// The last `horizon` rows have no full forward window, so the result is
/// `horizon` shorter than the input, empty when `close.len() <= horizon`.
pub fn triple_barrier_labels(
    high: &[f32],
    low: &[f32],
    close: &[f32],
    take_profit: f32,
    stop_loss: f32,
    horizon: usize,
) -> Vec<Label> {
    assert!(
        take_profit.is_sign_positive() && take_profit.is_finite(),
        "take profit must be a positive fraction"
    );
    assert!(
        stop_loss.is_sign_positive() && stop_loss.is_finite(),
        "stop loss must be a positive fraction"
    );
    assert!(horizon >= 1, "horizon must be at least one bar");

    let rows = close.len();
    if rows <= horizon {
        return Vec::new();
    }

    (0..rows - horizon)
        .map(|t| {
            let entry = close[t];
            let upper = entry * (1.0 + take_profit);
            let lower = entry * (1.0 - stop_loss);

            for k in t + 1..=t + horizon {
                let hit_take_profit = high[k] >= upper;
                let hit_stop_loss = low[k] <= lower;

                // One bar touching both barriers hides the intraday order, so the
                // outcome is ambiguous and defaults to the vertical-barrier Hold.
                if hit_take_profit && hit_stop_loss {
                    return Label::Hold((close[k] - entry) / entry);
                }
                if hit_take_profit {
                    return Label::Buy(take_profit);
                }
                if hit_stop_loss {
                    return Label::Sell(-stop_loss);
                }
            }

            // Untouched through the horizon: closed at the vertical barrier and
            // labeled by its realized close-to-close move.
            Label::Hold((close[t + horizon] - entry) / entry)
        })
        .collect()
}

/// Build aligned `u8` class and `f32` reward vectors from `high`, `low`, and
/// `close` price columns.
///
/// Both are `horizon` rows shorter than the input, so callers must keep the
/// matching leading rows. Errors if a column is not `f32` or contains nulls.
pub fn compute_labels_rewards(
    high: &Column,
    low: &Column,
    close: &Column,
    take_profit: f32,
    stop_loss: f32,
    horizon: usize,
) -> PolarsResult<(Vec<u8>, Vec<f32>)> {
    let high = high.f32()?;
    let low = low.f32()?;
    let close = close.f32()?;

    polars_ensure!(
        !high.has_nulls() && !low.has_nulls() && !close.has_nulls(),
        InvalidOperation: "OHLC columns must not contain nulls"
    );

    let high: Vec<f32> = high.into_no_null_iter().collect();
    let low: Vec<f32> = low.into_no_null_iter().collect();
    let close: Vec<f32> = close.into_no_null_iter().collect();

    let labels = triple_barrier_labels(&high, &low, &close, take_profit, stop_loss, horizon);

    Ok(labels
        .iter()
        .map(|label| (label.class(), label.reward()))
        .unzip())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Class indices, matching `Label::class`.
    const SELL: u8 = 0;
    const HOLD: u8 = 1;
    const BUY: u8 = 2;

    #[track_caller]
    fn assert_classes(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        take_profit: f32,
        stop_loss: f32,
        horizon: usize,
        expected: &[u8],
    ) {
        let classes: Vec<u8> =
            triple_barrier_labels(high, low, close, take_profit, stop_loss, horizon)
                .iter()
                .map(|label| label.class())
                .collect();

        assert_eq!(classes, expected);
    }

    #[test]
    fn shorter_than_horizon_is_empty() {
        assert_classes(
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
            0.05,
            0.05,
            2,
            &[],
        );
    }

    #[test]
    fn take_profit_first_is_buy() {
        // Bar 1's high crosses the +5% barrier while its low stays above -5%.
        assert_classes(
            &[100.0, 106.0, 100.0],
            &[100.0, 101.0, 99.0],
            &[100.0, 105.0, 100.0],
            0.05,
            0.05,
            2,
            &[BUY],
        );
    }

    #[test]
    fn stop_loss_first_is_sell() {
        // Bar 1's low crosses the -5% barrier while its high stays below +5%.
        assert_classes(
            &[100.0, 101.0, 100.0],
            &[100.0, 94.0, 95.0],
            &[100.0, 95.0, 100.0],
            0.05,
            0.05,
            2,
            &[SELL],
        );
    }

    #[test]
    fn untouched_is_hold() {
        // Price drifts inside both barriers for the whole horizon.
        assert_classes(
            &[100.0, 102.0, 103.0],
            &[100.0, 98.0, 97.0],
            &[100.0, 101.0, 102.0],
            0.05,
            0.05,
            2,
            &[HOLD],
        );
    }

    #[test]
    fn both_barriers_in_one_bar_is_hold() {
        // Bar 1 touches both, so the ambiguous outcome defaults to Hold.
        assert_classes(
            &[100.0, 106.0, 100.0],
            &[100.0, 94.0, 99.0],
            &[100.0, 100.0, 100.0],
            0.05,
            0.05,
            2,
            &[HOLD],
        );
    }

    #[test]
    fn earlier_barrier_wins() {
        // Take-profit at bar 1 beats a deeper stop-loss at bar 2.
        assert_classes(
            &[100.0, 106.0, 100.0],
            &[100.0, 99.0, 90.0],
            &[100.0, 105.0, 90.0],
            0.05,
            0.05,
            2,
            &[BUY],
        );
    }

    #[test]
    fn touch_beyond_horizon_is_hold() {
        // With horizon 1, only bar 1 is checked, so the +6% spike at bar 2 is
        // outside row 0's window and the row closes flat as a Hold.
        let labels = triple_barrier_labels(
            &[100.0, 101.0, 106.0],
            &[100.0, 100.0, 100.0],
            &[100.0, 101.0, 106.0],
            0.05,
            0.05,
            1,
        );
        assert_eq!(labels[0].class(), HOLD);
    }

    #[test]
    fn buy_and_sell_rewards_are_the_barriers() {
        let buy = triple_barrier_labels(
            &[100.0, 106.0, 100.0],
            &[100.0, 101.0, 99.0],
            &[100.0, 105.0, 100.0],
            0.05,
            0.05,
            2,
        );
        assert!((buy[0].reward() - 0.05).abs() < 1e-6);

        let sell = triple_barrier_labels(
            &[100.0, 101.0, 100.0],
            &[100.0, 94.0, 95.0],
            &[100.0, 95.0, 100.0],
            0.05,
            0.05,
            2,
        );
        assert!((sell[0].reward() + 0.05).abs() < 1e-6);
    }

    #[test]
    fn hold_reward_is_the_vertical_move() {
        // Untouched row closes at the vertical barrier: close 102 from entry 100.
        let labels = triple_barrier_labels(
            &[100.0, 102.0, 103.0],
            &[100.0, 98.0, 97.0],
            &[100.0, 101.0, 102.0],
            0.05,
            0.05,
            2,
        );
        assert!((labels[0].reward() - 0.02).abs() < 1e-6);
    }
}
