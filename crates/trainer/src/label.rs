//! The trainer's labeling layer on top of `stock_model::data::TickerFrames`:
//! triple-barrier labeling of the standardized frames. Loading and splitting live in the
//! shared store.

use miette::{IntoDiagnostic, Result, ensure};
use polars::prelude::*;
use stock_model::class::Action;
use stock_model::data::{LABEL, TickerFrames};
use stock_model::features::{CLOSE, HIGH, LOW};

/// Label every ticker frame with the triple-barrier class of opening a long at each
/// close, then drop the trailing `horizon` rows that have no forward outcome. The
/// barriers come from [`compute_labels`]; tickers too short for one labeled row are
/// dropped. The labeled store is `horizon` rows per ticker shorter than the input.
///
/// # Errors
/// If a frame's price columns are malformed or the label column cannot be attached.
pub fn into_labeled(
    frames: &TickerFrames,
    take_profit: f32,
    stop_loss: f32,
    horizon: usize,
) -> Result<TickerFrames> {
    let mut labeled = Vec::with_capacity(frames.frames.len());

    for frame in &frames.frames {
        let height = frame.height();
        // Too short for even one labeled row.
        if height <= horizon {
            continue;
        }

        let labels = compute_labels(
            frame.column(&HIGH).into_diagnostic()?,
            frame.column(&LOW).into_diagnostic()?,
            frame.column(&CLOSE).into_diagnostic()?,
            take_profit,
            stop_loss,
            horizon,
        )?;

        // `compute_labels` already drops the trailing horizon, so the head aligns.
        let mut head = frame.head(Some(height - horizon));
        head.with_column(Column::new(LABEL, labels))
            .into_diagnostic()?;
        labeled.push(head);
    }

    Ok(TickerFrames { frames: labeled })
}

/// Per-class label counts across the store, indexed Sell 0, Hold 1, Buy 2.
///
/// # Errors
/// If a frame lacks a `u8` label column.
pub fn label_counts(frames: &TickerFrames) -> PolarsResult<[usize; 3]> {
    let mut counts = [0usize; 3];
    for frame in &frames.frames {
        for label in frame.column(&LABEL)?.u8()?.into_no_null_iter() {
            counts[usize::from(label)] += 1;
        }
    }
    Ok(counts)
}

/// Aligned `u8` label classes from the price columns, `horizon` rows shorter than
/// the input. Errors if a column is not `f32` or has nulls.
fn compute_labels(
    high: &Column,
    low: &Column,
    close: &Column,
    take_profit: f32,
    stop_loss: f32,
    horizon: usize,
) -> Result<Vec<u8>> {
    let high = high.f32().into_diagnostic()?;
    let low = low.f32().into_diagnostic()?;
    let close = close.f32().into_diagnostic()?;

    ensure!(
        !high.has_nulls() && !low.has_nulls() && !close.has_nulls(),
        "OHLC columns must not contain nulls"
    );

    let high: Vec<f32> = high.into_no_null_iter().collect();
    let low: Vec<f32> = low.into_no_null_iter().collect();
    let close: Vec<f32> = close.into_no_null_iter().collect();

    let labels = triple_barrier_labels(&high, &low, &close, take_profit, stop_loss, horizon);

    Ok(labels.iter().map(|label| label.class()).collect())
}

/// Label each row with the triple-barrier outcome of opening a long at its close.
/// For entry `close[t]`, scan the next `horizon` bars: the first high crossing
/// `entry * (1 + take_profit)` is [`Action::Buy`], the first low crossing
/// `entry * (1 - stop_loss)` is [`Action::Sell`]. A bar touching both, or no touch
/// through the horizon, is [`Action::Hold`].
///
/// The result is `horizon` rows shorter than the input, empty when
/// `close.len() <= horizon`.
fn triple_barrier_labels(
    high: &[f32],
    low: &[f32],
    close: &[f32],
    take_profit: f32,
    stop_loss: f32,
    horizon: usize,
) -> Vec<Action> {
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

                // Both touched in one bar: order unknown, default to Hold.
                if hit_take_profit && hit_stop_loss {
                    return Action::Hold;
                }
                if hit_take_profit {
                    return Action::Buy;
                }
                if hit_stop_loss {
                    return Action::Sell;
                }
            }

            // Untouched: close at the vertical barrier.
            Action::Hold
        })
        .collect()
}

/// `n_tickers` tickers of `rows` rows each, for the dataset and batcher tests. Row `i`
/// fills every feature slot with `ticker * 1000 + i`, so a window's value identifies its
/// ticker and row; labels cycle 0/1/2.
#[cfg(test)]
pub(crate) fn synthetic(n_tickers: i16, rows: i16) -> TickerFrames {
    use stock_model::features::{FEATURE, FEATURE_NAMES, TICKER, feature_array};

    let height = usize::try_from(rows).unwrap();

    let frames = (0..n_tickers)
        .map(|ticker| {
            let base = f32::from(ticker) * 1000.0;
            let value: Vec<f32> = (0..rows).map(|i| base + f32::from(i)).collect();

            let mut columns = vec![
                Column::new(TICKER, vec![format!("t{ticker}"); height]),
                Column::new(
                    LABEL,
                    (0..rows)
                        .map(|i| u8::try_from(i % 3).unwrap())
                        .collect::<Vec<_>>(),
                ),
            ];
            for feature in FEATURE_NAMES {
                columns.push(Column::new(feature, value.clone()));
            }

            DataFrame::new(height, columns)
                .unwrap()
                .lazy()
                .with_column(feature_array().alias(FEATURE))
                .select([col(TICKER), col(FEATURE), col(LABEL)])
                .collect()
                .unwrap()
        })
        .collect();

    TickerFrames { frames }
}

#[cfg(test)]
mod tests {
    use super::*;

    use Action::{Buy, Hold, Sell};

    #[track_caller]
    fn assert_classes(
        high: &[f32],
        low: &[f32],
        close: &[f32],
        take_profit: f32,
        stop_loss: f32,
        horizon: usize,
        expected: &[Action],
    ) {
        let classes = triple_barrier_labels(high, low, close, take_profit, stop_loss, horizon);

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
            &[Buy],
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
            &[Sell],
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
            &[Hold],
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
            &[Hold],
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
            &[Buy],
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
        assert_eq!(labels[0], Hold);
    }
}
