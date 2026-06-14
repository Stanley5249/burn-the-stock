use chrono::NaiveDate;
use stock_model::inference::Prediction;

/// Convert a window or trade count to f64 exactly. These counts stay far below
/// `u32`, so the conversion is lossless; it panics only on an impossible count.
fn count_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("eval count fits in u32"))
}

/// Realized performance of the long-only policy over a held-out set, the offline
/// backtest's output. Two views sit side by side: pooled over every window scored
/// (flat positions included, comparable to the training Sharpe metric), and over the
/// traded subset whose long position cleared the gate.
pub struct EvalReport {
    /// Most recent bar scored, the backtest's as-of date. `None` when no window had
    /// enough history.
    pub as_of: Option<NaiveDate>,
    /// Windows scored across every ticker.
    pub windows: usize,
    /// Round-trip fee charged per unit of position, as a fraction.
    pub fee: f32,
    /// Position gate above which a window counts as a taken trade.
    pub min_position: f32,
    /// Mean net return per window, pooled over every window including flats.
    pub mean_net: f64,
    /// Pooled Sharpe `mean / population_std` of the per-window net returns.
    pub sharpe: f64,
    /// Sum of every window's net return.
    pub total_net: f64,
    /// Windows whose long position exceeded `min_position`.
    pub trades: usize,
    /// Mean net return over the traded subset.
    pub traded_mean_net: f64,
    /// Fraction of traded windows whose net return was positive.
    pub hit_rate: f64,
}

impl EvalReport {
    /// Score each window's realized net return `position * (reward - fee)`, the same
    /// map the Sharpe metric trains on, then pool it. `predictions` and `rewards`
    /// share an index, so `predictions[i].position` pairs with `rewards[i]`.
    /// `min_position` is the gate above which a window counts as a taken trade.
    pub fn aggregate(
        predictions: &[Prediction],
        rewards: &[f32],
        fee: f32,
        min_position: f32,
    ) -> Self {
        assert_eq!(
            predictions.len(),
            rewards.len(),
            "predictions and rewards must be aligned one to one"
        );

        let windows = predictions.len();
        let as_of = predictions.iter().map(|prediction| prediction.date).max();

        // Net return of the soft position at each window, the policy's realized
        // per-trade payoff less the turnover fee.
        let nets: Vec<f64> = predictions
            .iter()
            .zip(rewards)
            .map(|(prediction, &reward)| f64::from(prediction.position * (reward - fee)))
            .collect();

        let total_net: f64 = nets.iter().sum();
        let mean_net = if windows > 0 {
            total_net / count_f64(windows)
        } else {
            0.0
        };
        let variance = if windows > 0 {
            nets.iter().map(|net| (net - mean_net).powi(2)).sum::<f64>() / count_f64(windows)
        } else {
            0.0
        };
        let sharpe = if variance > 0.0 {
            mean_net / variance.sqrt()
        } else {
            0.0
        };

        // Traded subset: only windows whose long position cleared the gate.
        let traded_nets: Vec<f64> = predictions
            .iter()
            .zip(&nets)
            .filter(|(prediction, _)| prediction.position > min_position)
            .map(|(_, &net)| net)
            .collect();
        let trades = traded_nets.len();
        let traded_mean_net = if trades > 0 {
            traded_nets.iter().sum::<f64>() / count_f64(trades)
        } else {
            0.0
        };
        let wins = traded_nets.iter().filter(|&&net| net > 0.0).count();
        let hit_rate = if trades > 0 {
            count_f64(wins) / count_f64(trades)
        } else {
            0.0
        };

        Self {
            as_of,
            windows,
            fee,
            min_position,
            mean_net,
            sharpe,
            total_net,
            trades,
            traded_mean_net,
            hit_rate,
        }
    }
}

/// Print the backtest summary. The returns are per-trade over the triple-barrier
/// holding horizon, so the Sharpe is per-trade and neither daily nor annualized;
/// overlapping entry windows also break the IID assumption an annualization needs.
pub fn render(report: &EvalReport) {
    let Some(as_of) = report.as_of else {
        println!("No tickers had enough history to fill the model's window.");
        return;
    };

    println!("Backtest over the held-out window (as of {as_of}):");
    println!("  windows scored  : {}", report.windows);
    println!("  fee (round trip): {:.3}%", f64::from(report.fee) * 100.0);
    println!(
        "  mean net return : {:+.4}  (per window, pooled incl. flat)",
        report.mean_net
    );
    println!(
        "  pooled Sharpe   : {:.4}  (per-trade horizon, not annualized)",
        report.sharpe
    );
    println!("  total net return: {:+.3}", report.total_net);

    println!("\nTraded (position > {:.2}):", report.min_position);
    if report.trades == 0 {
        println!("  none above threshold; the policy stayed flat.");
        return;
    }
    println!("  trades          : {}", report.trades);
    println!("  mean net return : {:+.4}", report.traded_mean_net);
    println!("  hit rate        : {:.1}%", report.hit_rate * 100.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use stock_model::inference::Action;

    fn prediction(position: f32) -> Prediction {
        // Only `position` and `date` matter to the aggregation; the rest are filler.
        Prediction {
            ticker: "t".to_string(),
            date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            probabilities: [0.0, 0.0, 0.0],
            action: Action::Hold,
            position,
        }
    }

    #[test]
    fn aggregate_pools_nets_and_gates_trades() {
        // Three windows: a flat one (no fee, no trade), a winning trade, and a losing
        // one. With fee 0, net = position * reward.
        let predictions = vec![prediction(0.0), prediction(1.0), prediction(0.5)];
        let rewards = [0.10f32, 0.10, -0.20];

        let report = EvalReport::aggregate(&predictions, &rewards, 0.0, 0.0);

        // Nets: 0.0, 0.10, -0.10 -> total 0, mean 0.
        assert_eq!(report.windows, 3);
        assert!(report.total_net.abs() < 1e-9);
        assert!(report.mean_net.abs() < 1e-9);

        // Two windows cleared the gate of 0.0; one won and one lost.
        assert_eq!(report.trades, 2);
        assert!((report.hit_rate - 0.5).abs() < 1e-9);
        assert!(report.traded_mean_net.abs() < 1e-9);

        // Population std of {0, 0.1, -0.1} about mean 0 is sqrt(0.02/3), so the
        // pooled Sharpe is mean/std = 0.
        assert!(report.sharpe.abs() < 1e-9);
    }

    #[test]
    fn aggregate_charges_fee_against_the_position() {
        // One full-position window earning the fee exactly nets zero.
        let predictions = vec![prediction(1.0)];
        let rewards = [0.005f32];

        let report = EvalReport::aggregate(&predictions, &rewards, 0.005, 0.0);

        assert_eq!(report.trades, 1);
        assert!(report.total_net.abs() < 1e-7);
        // Net is exactly zero, which is not a win, so the hit rate is zero.
        assert!(report.hit_rate.abs() < 1e-9);
    }

    #[test]
    fn empty_input_has_no_as_of() {
        let report = EvalReport::aggregate(&[], &[], 0.005, 0.0);
        assert!(report.as_of.is_none());
        assert_eq!(report.windows, 0);
        assert_eq!(report.trades, 0);
    }
}
