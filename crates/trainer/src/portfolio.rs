//! Stateful long-only portfolio simulation under the `sim_stock` rules. The engine
//! walks one trading day at a time over a pre-built [`TradingDay`] stream, realizing
//! profit only on a sell. Pure: signals and prices in, a [`BacktestReport`] out.
//!
//! Rules: 100M starting cash, equal weight across at most ten holdings, whole
//! 1,000-share lots, buys at the day's low and sells at the high (snapped to a legal
//! tick in range), 0.1425% commission per side, 0.3% sell-only tax.

use std::collections::HashMap;

use chrono::NaiveDate;
use stock_model::inference::Action;

/// The platform's simulated starting balance.
pub const STARTING_CASH: f64 = 100_000_000.0;

/// Shares per Taiwan lot. Counts stay f64 lot-multiples to avoid integer casts.
const LOT: f64 = 1_000.0;
/// Commission charged on each buy and sell.
const COMMISSION_RATE: f64 = 0.001_425;
/// Minimum commission per transaction.
const MIN_COMMISSION: f64 = 20.0;
/// Securities transaction tax, charged on sells only.
const SELL_TAX_RATE: f64 = 0.003;
/// Trading days per year, for the platform's linear annualization.
const ANNUAL_TRADING_DAYS: f64 = 252.0;

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
struct Holding {
    /// Whole shares held, always a multiple of [`LOT`].
    shares: f64,
    /// Cash paid to open, including the buy commission.
    cost: f64,
    /// Most recent close seen, used to value the holding on days it has no bar.
    mark: f64,
}

/// A completed round trip, kept just long enough to aggregate the trade metrics.
struct Trade {
    /// Net profit in cash after every fee and the sell tax.
    pnl: f64,
    /// Net return on cost, `pnl / cost`.
    return_pct: f64,
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
    pub trading_days: usize,
    pub equity_curve: Vec<EquityPoint>,
}

/// Commission on a trade of `amount`, with the per-transaction floor.
fn commission(amount: f64) -> f64 {
    (amount * COMMISSION_RATE).max(MIN_COMMISSION)
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
fn tick_floor(price: f64) -> f64 {
    let tick = tick_size(price);
    ((price / tick) + 1e-9).floor() * tick
}

/// Smallest legal tick price `>= price`.
fn tick_ceil(price: f64) -> f64 {
    let tick = tick_size(price);
    ((price / tick) - 1e-9).ceil() * tick
}

/// Buy fill: lowest legal tick at or above the day's low (or open).
fn buy_price(bar: &DayBar, fill: Fill) -> f64 {
    match fill {
        Fill::LowHigh => tick_ceil(f64::from(bar.low)),
        Fill::Open => tick_ceil(f64::from(bar.open)),
    }
}

/// Sell fill: highest legal tick at or below the day's high (or open).
fn sell_price(bar: &DayBar, fill: Fill) -> f64 {
    match fill {
        Fill::LowHigh => tick_floor(f64::from(bar.high)),
        Fill::Open => tick_floor(f64::from(bar.open)),
    }
}

/// Total mark-to-market value of the open holdings, using each ticker's close today
/// when it has a bar and its last seen mark otherwise.
fn holdings_value(holdings: &HashMap<String, Holding>, bars: &HashMap<String, DayBar>) -> f64 {
    holdings
        .iter()
        .map(|(ticker, holding)| {
            let mark = bars
                .get(ticker)
                .map_or(holding.mark, |bar| f64::from(bar.close));
            mark * holding.shares
        })
        .sum()
}

/// Convert a small count to f64. Backtest counts stay well below `u32`.
fn count_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("backtest count fits in u32"))
}

/// Whole-lot shares affordable for `budget`, trimmed until `cash` also covers the
/// commission. Zero when even one lot is out of reach.
fn affordable_shares(budget: f64, price: f64, cash: f64) -> f64 {
    let mut shares = (budget / (price * LOT)).floor() * LOT;
    while shares > 0.0 {
        let amount = price * shares;
        if amount + commission(amount) <= cash {
            break;
        }
        shares -= LOT;
    }
    shares
}

/// Run the simulation over `days` (ascending). Each day: sell Sell-flagged holdings
/// (and all on the final day), buy the strongest above-threshold names into open
/// slots, then mark to the closes.
pub fn run(days: &[TradingDay], config: &BacktestConfig) -> BacktestReport {
    let mut cash = config.starting_cash;
    let mut holdings: HashMap<String, Holding> = HashMap::new();
    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_curve: Vec<EquityPoint> = Vec::with_capacity(days.len());

    let last_index = days.len().saturating_sub(1);

    for (index, day) in days.iter().enumerate() {
        let is_last = index == last_index;

        sell_phase(
            day,
            is_last,
            config.fill,
            &mut holdings,
            &mut cash,
            &mut trades,
        );

        if !is_last {
            buy_phase(day, config, &mut holdings, &mut cash);
        }

        // Refresh marks for held tickers that traded today.
        for (ticker, holding) in &mut holdings {
            if let Some(bar) = day.bars.get(ticker) {
                holding.mark = f64::from(bar.close);
            }
        }
        let equity = cash
            + holdings
                .values()
                .map(|holding| holding.mark * holding.shares)
                .sum::<f64>();
        equity_curve.push(EquityPoint {
            date: day.date,
            equity,
        });
    }

    BacktestReport::new(config.starting_cash, &trades, equity_curve)
}

/// Sell holdings that flipped to Sell today, or every holding on the final day.
fn sell_phase(
    day: &TradingDay,
    is_last: bool,
    fill: Fill,
    holdings: &mut HashMap<String, Holding>,
    cash: &mut f64,
    trades: &mut Vec<Trade>,
) {
    let to_sell: Vec<String> = holdings
        .keys()
        .filter(|ticker| {
            is_last
                || day
                    .bars
                    .get(*ticker)
                    .is_some_and(|bar| bar.action == Action::Sell)
        })
        .cloned()
        .collect();

    for ticker in to_sell {
        let holding = holdings.remove(&ticker).expect("ticker is held");
        // Sell at the high when there is a bar, else at the last mark.
        let price = day
            .bars
            .get(&ticker)
            .map_or(holding.mark, |bar| sell_price(bar, fill));
        let amount = price * holding.shares;
        let proceeds = amount - commission(amount) - amount * SELL_TAX_RATE;
        *cash += proceeds;
        let pnl = proceeds - holding.cost;
        trades.push(Trade {
            pnl,
            return_pct: pnl / holding.cost,
        });
    }
}

/// Fill open slots with the strongest above-threshold names not already held.
fn buy_phase(
    day: &TradingDay,
    config: &BacktestConfig,
    holdings: &mut HashMap<String, Holding>,
    cash: &mut f64,
) {
    // Equal-weight target from equity at the buy phase start, so all of the day's
    // buys size against the same value.
    let equity = *cash + holdings_value(holdings, &day.bars);
    let target = equity / count_f64(config.max_holdings.max(1));

    let mut candidates: Vec<(&String, &DayBar)> = day
        .bars
        .iter()
        .filter(|(ticker, bar)| !holdings.contains_key(*ticker) && bar.score > config.threshold)
        .collect();
    // Strongest first, ticker breaking ties for determinism.
    candidates.sort_by(|left, right| {
        right
            .1
            .score
            .total_cmp(&left.1.score)
            .then_with(|| left.0.cmp(right.0))
    });

    for (ticker, bar) in candidates {
        if holdings.len() >= config.max_holdings {
            break;
        }
        let price = buy_price(bar, config.fill);
        let shares = affordable_shares(target.min(*cash), price, *cash);
        if shares <= 0.0 {
            continue;
        }
        let amount = price * shares;
        let cost = amount + commission(amount);
        *cash -= cost;
        holdings.insert(
            ticker.clone(),
            Holding {
                shares,
                cost,
                mark: f64::from(bar.close),
            },
        );
    }
}

impl BacktestReport {
    fn new(starting_cash: f64, trades: &[Trade], equity_curve: Vec<EquityPoint>) -> Self {
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
        let annualized_return = if trading_days > 0 {
            cumulative_return / count_f64(trading_days) * ANNUAL_TRADING_DAYS
        } else {
            0.0
        };

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
            trading_days,
            equity_curve,
        }
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
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        count_f64(numerator) / count_f64(denominator)
    }
}

/// Print the platform's performance metrics.
pub fn render(report: &BacktestReport) {
    println!("Portfolio backtest ({} trading days):", report.trading_days);
    println!("  starting cash    : {:>16.0}", report.starting_cash);
    println!("  final equity     : {:>16.0}", report.final_equity);
    println!(
        "  cumulative return: {:+.2}%",
        report.cumulative_return * 100.0
    );
    println!(
        "  annualized       : {:+.2}%",
        report.annualized_return * 100.0
    );
    println!("  trades           : {}", report.trade_count);
    println!("  win rate         : {:.1}%", report.win_rate * 100.0);
    if report.profit_factor.is_finite() {
        println!("  profit factor    : {:.2}", report.profit_factor);
    } else {
        println!("  profit factor    : inf (no losing trades)");
    }
    println!(
        "  avg win / loss   : {:+.2}% / {:+.2}%",
        report.avg_win_return * 100.0,
        report.avg_loss_return * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, day).unwrap()
    }

    fn bar(score: f32, action: Action, low: f32, high: f32, close: f32) -> DayBar {
        DayBar {
            score,
            action,
            open: low,
            low,
            high,
            close,
        }
    }

    fn config(starting_cash: f64, max_holdings: usize) -> BacktestConfig {
        BacktestConfig {
            threshold: 0.0,
            fill: Fill::LowHigh,
            max_holdings,
            starting_cash,
        }
    }

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

    #[test]
    fn single_winning_trade_is_hand_checked() {
        // Buy A on day 1 at low 10, hold through day 2, liquidate day 3 at high 20.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, Action::Buy, 10.0, 12.0, 11.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.1, Action::Hold, 13.0, 16.0, 15.0));
        let mut day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };
        day3.bars
            .insert("A".to_string(), bar(0.0, Action::Hold, 18.0, 20.0, 19.0));

        let report = run(&[day1, day2, day3], &config(1_000_000.0, 2));

        // Buy 50,000 shares at 10 (cost 500,712.5 incl. commission), sell at 20 for
        // proceeds 995,575 net of commission and tax, so final cash is 1,494,862.5.
        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trading_days, 3);
        assert!((report.final_equity - 1_494_862.5).abs() < 1e-2);
        assert!((report.cumulative_return - 0.494_862_5).abs() < 1e-6);
        assert!((report.win_rate - 1.0).abs() < 1e-9);
        assert!(report.profit_factor.is_infinite());
        // Net return on cost of the lone winner.
        assert!((report.avg_win_return - 0.988_32).abs() < 1e-4);
    }

    #[test]
    fn winner_and_loser_drive_win_rate_and_profit_factor() {
        // Two slots, two buys on day 1, both liquidated day 2: A rises, B falls.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.6, Action::Buy, 10.0, 11.0, 11.0));
        day1.bars
            .insert("B".to_string(), bar(0.5, Action::Buy, 100.0, 101.0, 100.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.0, Action::Sell, 19.0, 20.0, 20.0));
        day2.bars
            .insert("B".to_string(), bar(0.0, Action::Sell, 89.0, 90.0, 90.0));

        let report = run(&[day1, day2], &config(2_000_000.0, 2));

        assert_eq!(report.trade_count, 2);
        assert!((report.win_rate - 0.5).abs() < 1e-9);
        // A nets +989,725, B nets -94,866.75, so final equity is 2,894,858.25.
        assert!((report.final_equity - 2_894_858.25).abs() < 1e-2);
        assert!((report.profit_factor - 10.4327).abs() < 1e-3);
        assert!((report.avg_loss_return + 0.105_258).abs() < 1e-5);
        assert!(report.cumulative_return > 0.0);
    }

    #[test]
    fn weak_signals_stay_in_cash() {
        // Only candidate is below the threshold, so nothing is bought.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.05, Action::Buy, 10.0, 12.0, 11.0));
        let mut cfg = config(1_000_000.0, 10);
        cfg.threshold = 0.2;

        let report = run(&[day1], &cfg);

        assert_eq!(report.trade_count, 0);
        assert!((report.final_equity - 1_000_000.0).abs() < 1e-9);
    }
}
