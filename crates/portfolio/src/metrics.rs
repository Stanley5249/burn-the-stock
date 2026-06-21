use super::types::{BacktestReport, EquityPoint, Trade, TradeEvent};

/// Trading days per year, for the platform's linear annualization.
const ANNUAL_TRADING_DAYS: f64 = 252.0;

impl BacktestReport {
    pub(super) fn new(
        starting_cash: f64,
        trades: Vec<Trade>,
        events: Vec<TradeEvent>,
        equity_curve: Vec<EquityPoint>,
    ) -> Self {
        let final_equity = equity_curve
            .last()
            .map_or(starting_cash, |point| point.equity);
        let cumulative_return = (final_equity - starting_cash) / starting_cash;

        let trade_count = trades.len();
        let win_count = trades.iter().filter(|trade| trade.pnl > 0.0).count();
        let win_rate = ratio(win_count, trade_count);

        let gain: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        // Loss total is negative, so negate it for the ratio.
        let loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl).sum();
        let profit_factor = if loss < 0.0 {
            gain / -loss
        } else if gain > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_win_return = mean(trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.return_pct));
        let avg_loss_return = mean(trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.return_pct));

        let trading_days = equity_curve.len();
        #[allow(clippy::cast_precision_loss)]
        let annualized_return = if trading_days > 0 {
            cumulative_return / trading_days as f64 * ANNUAL_TRADING_DAYS
        } else {
            0.0
        };

        let sharpe = annualized_sharpe(&equity_curve);

        Self {
            starting_cash,
            final_equity,
            cumulative_return,
            annualized_return,
            trade_count,
            win_rate,
            profit_factor,
            avg_win_return,
            avg_loss_return,
            sharpe,
            trading_days,
            equity_curve,
            trades,
            events,
        }
    }
}

/// Annualized Sharpe of the daily equity returns, zero when the curve is too short
/// or flat.
fn annualized_sharpe(curve: &[EquityPoint]) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = curve
        .windows(2)
        .map(|pair| pair[1].equity / pair[0].equity - 1.0)
        .collect();
    let average = mean(returns.iter().copied());
    let deviation = mean(returns.iter().map(|value| (value - average).powi(2))).sqrt();
    if deviation == 0.0 {
        0.0
    } else {
        average / deviation * ANNUAL_TRADING_DAYS.sqrt()
    }
}

/// Mean of an iterator of returns, zero when empty.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0u32;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

/// `numerator / denominator` as a fraction, zero when the denominator is zero.
#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
