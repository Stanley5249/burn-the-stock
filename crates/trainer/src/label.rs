use polars::prelude::*;
use std::cmp::Ordering;

/// Trade action paired with the signed forward return to the trend extreme it
/// targets. The reward is positive while the trend climbs toward its high and
/// negative while it falls toward its low, and it is the per-sample payoff the
/// expected-value metric scores a Buy against.
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

    /// Signed fractional move to the upcoming swing extreme.
    pub fn reward(self) -> f32 {
        match self {
            Label::Sell(reward) | Label::Hold(reward) | Label::Buy(reward) => reward,
        }
    }
}

/// Minimum reversal magnitude (fractional) that confirms a swing high or low.
pub const LABEL_THRESHOLD: f32 = 0.03;

/// Label each price (except the last) with the action a perfect trader would
/// take, each carrying the signed forward return to the extreme it heads toward.
///
/// Walks prices from the end and tracks the running swing high and low. A swing
/// reverses once the move away from its extreme exceeds `rel_threshold`, a fraction
/// of the current price. Every row carries the move to its upcoming extreme, while
/// only the label waits for the confirmed turning point. The last price has no
/// forward window, so the result is one shorter than `prices`.
pub fn swing_labels<P>(mut prices: P, rel_threshold: f32) -> Vec<Label>
where
    P: Iterator<Item = f32> + DoubleEndedIterator,
{
    enum Phase {
        Up { min: f32, max: f32 },
        Flat { flat: f32 },
        Down { min: f32, max: f32 },
    }

    assert!(
        rel_threshold.is_sign_positive() && rel_threshold.is_finite(),
        "relative threshold must be a positive number"
    );

    let mut phase = match prices.next_back() {
        Some(flat) => Phase::Flat { flat },
        None => return vec![],
    };

    let transition = move |price: f32| {
        let abs_threshold = price * rel_threshold;

        phase = match phase {
            Phase::Flat { flat } => match price.total_cmp(&flat) {
                Ordering::Less => Phase::Up {
                    min: price,
                    max: flat,
                },
                Ordering::Equal => Phase::Flat { flat },
                Ordering::Greater => Phase::Down {
                    min: flat,
                    max: price,
                },
            },
            Phase::Up { min, max } => {
                if price - min > abs_threshold {
                    Phase::Down { min, max: price }
                } else if price >= min {
                    Phase::Up { min, max }
                } else {
                    Phase::Up { min: price, max }
                }
            }
            Phase::Down { min, max } => {
                if max - price > abs_threshold {
                    Phase::Up { min: price, max }
                } else if max >= price {
                    Phase::Down { min, max }
                } else {
                    Phase::Down { min, max: price }
                }
            }
        };

        match phase {
            Phase::Flat { .. } => Label::Hold(0.0),
            // The reward points at the upcoming extreme even on Hold rows. Only the
            // label waits for the turning point, since while the price still drifts
            // off the running extreme it is not a confirmed reversal yet.
            Phase::Up { min, max } => {
                let reward = (max - price) / price;
                if price > min {
                    Label::Hold(reward)
                } else if max - price > abs_threshold {
                    Label::Buy(reward)
                } else {
                    Label::Hold(reward)
                }
            }
            Phase::Down { min, max } => {
                let reward = (min - price) / price;
                if price < max {
                    Label::Hold(reward)
                } else if price - min > abs_threshold {
                    Label::Sell(reward)
                } else {
                    Label::Hold(reward)
                }
            }
        }
    };

    let mut labels: Vec<Label> = prices.rev().map(transition).collect();

    labels.reverse();

    labels
}

/// Build aligned `u8` class and `f32` reward vectors from a `close` price column.
///
/// Both are one row shorter than `close`, so callers must align the feature rows
/// accordingly. Errors if `close` is not `f32` or contains nulls.
pub fn compute_labels_rewards(close: &Column, threshold: f32) -> PolarsResult<(Vec<u8>, Vec<f32>)> {
    let ca = close.f32()?;

    polars_ensure!(!ca.has_nulls(), InvalidOperation: "close column has nulls");

    let labels = swing_labels(ca.into_no_null_iter(), threshold);

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
    fn assert_classes(closes: &[f32], expected: &[u8], threshold: f32) {
        let classes: Vec<u8> = swing_labels(closes.iter().copied(), threshold)
            .iter()
            .map(|label| label.class())
            .collect();

        assert_eq!(classes, expected);
    }

    #[track_caller]
    fn assert_rewards(closes: &[f32], expected: &[f32], threshold: f32) {
        let rewards: Vec<f32> = swing_labels(closes.iter().copied(), threshold)
            .iter()
            .map(|label| label.reward())
            .collect();

        assert_eq!(rewards.len(), expected.len());
        for (got, want) in rewards.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "reward {got} != {want}");
        }
    }

    #[test]
    fn empty() {
        assert_classes(&[], &[], 0.03);
    }

    #[test]
    fn single_price() {
        assert_classes(&[100.0], &[], 0.03);
    }

    #[test]
    fn rise_is_buy() {
        assert_classes(&[100.0, 105.0], &[BUY], 0.03);
    }

    #[test]
    fn fall_is_sell() {
        assert_classes(&[100.0, 95.0], &[SELL], 0.03);
    }

    #[test]
    fn flat_is_hold() {
        assert_classes(&[100.0, 100.0], &[HOLD], 0.03);
    }

    #[test]
    fn minor_pullback_holds() {
        assert_classes(&[100.0, 101.0, 100.5], &[HOLD, HOLD], 0.03);
    }

    #[test]
    fn peak_then_trough() {
        assert_classes(&[100.0, 105.0, 100.0], &[BUY, SELL], 0.03);
    }

    #[test]
    fn holds_within_up_swing() {
        assert_classes(
            &[100.0, 105.0, 103.0, 105.0, 107.0, 100.0],
            &[BUY, HOLD, BUY, HOLD, SELL],
            0.03,
        );
    }

    #[test]
    fn holds_within_down_swing() {
        assert_classes(
            &[100.0, 93.0, 95.0, 97.0, 95.0, 100.0],
            &[SELL, BUY, BUY, HOLD, BUY],
            0.03,
        );
    }

    #[test]
    fn rise_reward_is_upside() {
        // The lone row heads up to 105 from 100, a 5% gain.
        assert_rewards(&[100.0, 105.0], &[0.05], 0.03);
    }

    #[test]
    fn fall_reward_is_downside() {
        // Heading down to 95 from 100 is a signed -5% move.
        assert_rewards(&[100.0, 95.0], &[-0.05], 0.03);
    }

    #[test]
    fn peak_then_trough_rewards() {
        // Row 0 rises 5% to the 105 peak, row 1 falls from 105 to the 100 trough.
        assert_rewards(
            &[100.0, 105.0, 100.0],
            &[0.05, (100.0 - 105.0) / 105.0],
            0.03,
        );
    }
}
