use polars::prelude::*;
use std::cmp::Ordering;

/// Trade action class. Discriminants match the model's output index order
/// and are the values stored in the label column and tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    Sell = 0,
    Hold = 1,
    Buy = 2,
}

/// Minimum reversal magnitude (fractional) that confirms a swing high or low.
pub const LABEL_THRESHOLD: f32 = 0.03;

/// Label each price (except the last) with the action a perfect trader would take.
///
/// Walks prices from the end and tracks the running swing high and low. A swing
/// reverses once the move away from its extreme exceeds `rel_threshold`, a fraction
/// of the current price. `f` maps each [`Label`] to the caller's output type. The
/// last price has no forward window, so the result is one shorter than `prices`.
pub fn swing_labels<P, B, F>(mut prices: P, rel_threshold: f32, f: F) -> Vec<B>
where
    P: Iterator<Item = f32> + DoubleEndedIterator,
    F: FnMut(Label) -> B,
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
            Phase::Flat { .. } => Label::Hold,
            // Only commit at the turning point. While the price still drifts off the
            // running extreme it is not a confirmed reversal yet, so keep it Hold.
            Phase::Up { min, max } => {
                if price > min {
                    Label::Hold
                } else if max - price > abs_threshold {
                    Label::Buy
                } else {
                    Label::Hold
                }
            }
            Phase::Down { min, max } => {
                if price < max {
                    Label::Hold
                } else if price - min > abs_threshold {
                    Label::Sell
                } else {
                    Label::Hold
                }
            }
        }
    };

    let mut labels: Vec<B> = prices.rev().map(transition).map(f).collect();

    labels.reverse();

    labels
}

/// Build a `u8` label column from a `close` price column via [`swing_labels`].
///
/// The result is one row shorter than `close`, so callers must align the feature
/// rows accordingly. Errors if `close` is not `f32` or contains nulls.
pub fn compute_labels(close: &Column, name: PlSmallStr, threshold: f32) -> PolarsResult<Column> {
    let ca = close.f32()?;

    polars_ensure!(!ca.has_nulls(), InvalidOperation: "close column has nulls");

    let labels = swing_labels(ca.into_no_null_iter(), threshold, |label| label as u8);

    Ok(Series::from_vec(name, labels).into_column())
}

#[cfg(test)]
mod tests {
    use super::Label::*;
    use super::*;
    use std::convert::identity;

    #[track_caller]
    fn assert_labels(closes: &[f32], expected: &[Label], threshold: f32) {
        let labels = swing_labels(closes.iter().copied(), threshold, identity);

        assert_eq!(labels, expected);
    }

    #[test]
    fn empty() {
        assert_labels(&[], &[], 0.03);
    }

    #[test]
    fn single_price() {
        assert_labels(&[100.0], &[], 0.03);
    }

    #[test]
    fn rise_is_buy() {
        assert_labels(&[100.0, 105.0], &[Buy], 0.03);
    }

    #[test]
    fn fall_is_sell() {
        assert_labels(&[100.0, 95.0], &[Sell], 0.03);
    }

    #[test]
    fn flat_is_hold() {
        assert_labels(&[100.0, 100.0], &[Hold], 0.03);
    }

    #[test]
    fn minor_pullback_holds() {
        assert_labels(&[100.0, 101.0, 100.5], &[Hold, Hold], 0.03);
    }

    #[test]
    fn peak_then_trough() {
        assert_labels(&[100.0, 105.0, 100.0], &[Buy, Sell], 0.03);
    }

    #[test]
    fn holds_within_up_swing() {
        assert_labels(
            &[100.0, 105.0, 103.0, 105.0, 107.0, 100.0],
            &[Buy, Hold, Buy, Hold, Sell],
            0.03,
        );
    }

    #[test]
    fn holds_within_down_swing() {
        assert_labels(
            &[100.0, 93.0, 95.0, 97.0, 95.0, 100.0],
            &[Sell, Buy, Buy, Hold, Buy],
            0.03,
        );
    }
}
