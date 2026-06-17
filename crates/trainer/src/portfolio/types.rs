use std::collections::HashMap;
use std::fmt;

use chrono::NaiveDate;
use stock_model::class::Action;

/// Why a holding was closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    TakeProfit,
    StopLoss,
    Time,
    Signal,
    Rotate,
    Final,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ExitReason::TakeProfit => "take_profit",
            ExitReason::StopLoss => "stop_loss",
            ExitReason::Time => "time",
            ExitReason::Signal => "signal",
            ExitReason::Rotate => "rotate",
            ExitReason::Final => "final",
        };
        f.write_str(s)
    }
}

/// Which side of the book an order executed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Side::Buy => "Buy",
            Side::Sell => "Sell",
        };
        f.write_str(s)
    }
}

/// Which intraday prices the simulated orders fill at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// Optimistic best case: buy at the day's low, sell at the day's high.
    LowHigh,
    /// Pessimistic comparison: buy and sell at the day's open.
    Open,
}

/// Knobs that shape one backtest run.
pub struct BacktestConfig {
    /// Minimum net-bullish score to consider buying a stock.
    pub threshold: f32,
    /// Which prices fills happen at.
    pub fill: Fill,
    /// Most stocks held at once; each new buy targets `equity / max_holdings`.
    pub max_holdings: usize,
    /// Opening balance.
    pub starting_cash: f64,
    /// Take-profit exit, a positive fraction of the entry price.
    pub take_profit: f64,
    /// Stop-loss exit, a positive fraction of the entry price.
    pub stop_loss: f64,
    /// Trading days to hold before a time exit, the vertical barrier.
    pub max_hold_days: usize,
}

/// One ticker's signal and prices on one trading day. `score` and `action` are the
/// model's call, already lagged to the previous close.
pub struct DayBar {
    pub score: f32,
    pub action: Action,
    pub open: f32,
    pub low: f32,
    pub high: f32,
    pub close: f32,
}

/// Every ticker's bar for one trading day, in date order across the run.
pub struct TradingDay {
    pub date: NaiveDate,
    pub bars: HashMap<String, DayBar>,
}

/// An open position.
pub(super) struct Holding {
    /// Whole shares held, always a multiple of [`super::pricing::LOT`].
    pub(super) shares: f64,
    /// Cash paid to open, including the buy commission.
    pub(super) cost: f64,
    /// Most recent close seen, used to value the holding on days it has no bar.
    pub(super) mark: f64,
    pub(super) entry_date: NaiveDate,
    pub(super) entry_price: f64,
    /// Day-loop index at entry, for counting trading days held.
    pub(super) entry_index: usize,
}

/// Mutable account state the day phases evolve: cash, open positions, and the trade
/// and action logs.
pub(super) struct Ledger {
    pub(super) cash: f64,
    pub(super) holdings: HashMap<String, Holding>,
    pub(super) trades: Vec<Trade>,
    pub(super) events: Vec<TradeEvent>,
}

/// A completed round trip.
pub struct Trade {
    pub ticker: String,
    pub entry_date: NaiveDate,
    pub exit_date: NaiveDate,
    pub entry_price: f64,
    pub exit_price: f64,
    pub shares: f64,
    /// Cash paid to open, including the buy commission.
    pub cost: f64,
    /// Cash received on close, net of commission and the sell tax.
    pub proceeds: f64,
    /// Net profit in cash, `proceeds - cost`.
    pub pnl: f64,
    /// Net return on cost, `pnl / cost`.
    pub return_pct: f64,
    /// Exit rule that closed the trade.
    pub exit_reason: ExitReason,
}

/// One executed buy or sell, for the action log.
pub struct TradeEvent {
    pub date: NaiveDate,
    pub ticker: String,
    pub side: Side,
    /// Exit rule for a sell, `None` for a buy.
    pub reason: Option<ExitReason>,
    pub price: f64,
    pub shares: f64,
    /// Gross fill value, `price * shares`, before fees.
    pub amount: f64,
    pub cash_after: f64,
}

/// The account value at the close of one trading day, for the equity curve CSV.
pub struct EquityPoint {
    pub date: NaiveDate,
    pub equity: f64,
}

/// The backtest outcome and the platform's performance metrics.
pub struct BacktestReport {
    pub starting_cash: f64,
    pub final_equity: f64,
    pub cumulative_return: f64,
    pub annualized_return: f64,
    pub trade_count: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win_return: f64,
    pub avg_loss_return: f64,
    /// Annualized Sharpe of the daily equity returns, the honest risk-adjusted figure.
    pub sharpe: f64,
    pub trading_days: usize,
    pub equity_curve: Vec<EquityPoint>,
    pub trades: Vec<Trade>,
    pub events: Vec<TradeEvent>,
}
