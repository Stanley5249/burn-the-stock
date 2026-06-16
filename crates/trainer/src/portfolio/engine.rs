use std::collections::HashMap;

use chrono::NaiveDate;
use stock_model::inference::Action;

use super::count_f64;
use super::pricing::{
    LOT, SELL_TAX_RATE, buy_price, commission, round_trip_cost, sell_price, tick_floor,
};
use super::types::{
    BacktestConfig, BacktestReport, DayBar, EquityPoint, Holding, Ledger, Trade, TradeEvent,
    TradingDay,
};

/// Run the simulation over `days` (ascending). Each day: run the exit ladder (barriers,
/// the time stop, a model Sell, the final-day liquidation), rotate the weakest holdings
/// out for clearly stronger names, buy into open slots, then mark to the closes.
pub fn run(days: &[TradingDay], config: &BacktestConfig) -> BacktestReport {
    let mut ledger = Ledger {
        cash: config.starting_cash,
        holdings: HashMap::new(),
        trades: Vec::new(),
        events: Vec::new(),
    };
    let mut equity_curve: Vec<EquityPoint> = Vec::with_capacity(days.len());

    let last_index = days.len().saturating_sub(1);

    for (index, day) in days.iter().enumerate() {
        let is_last = index == last_index;

        sell_phase(day, index, is_last, config, &mut ledger);

        if !is_last {
            rotate_phase(day, config, &mut ledger);
            buy_phase(day, index, config, &mut ledger);
        }

        // Refresh marks for held tickers that traded today.
        for (ticker, holding) in &mut ledger.holdings {
            if let Some(bar) = day.bars.get(ticker) {
                holding.mark = f64::from(bar.close);
            }
        }
        let equity = ledger.cash
            + ledger
                .holdings
                .values()
                .map(|holding| holding.mark * holding.shares)
                .sum::<f64>();
        equity_curve.push(EquityPoint {
            date: day.date,
            equity,
        });
    }

    BacktestReport::new(
        config.starting_cash,
        ledger.trades,
        ledger.events,
        equity_curve,
    )
}

/// Exit price and reason for a holding today, or `None` to keep holding. Take-profit
/// wins a both-touch bar; then stop-loss, the time barrier, a model Sell, and finally
/// the forced liquidation on the last day. Rotation is handled in [`rotate_phase`].
fn exit_decision(
    holding: &Holding,
    bar: Option<&DayBar>,
    index: usize,
    is_last: bool,
    config: &BacktestConfig,
) -> Option<(f64, &'static str)> {
    let Some(bar) = bar else {
        // No bar today: only the final day can liquidate, at the last mark.
        return is_last.then_some((holding.mark, "final"));
    };

    let upper = holding.entry_price * (1.0 + config.take_profit);
    let lower = holding.entry_price * (1.0 - config.stop_loss);

    if f64::from(bar.high) >= upper {
        Some((tick_floor(upper), "take_profit"))
    } else if f64::from(bar.low) <= lower {
        Some((tick_floor(lower), "stop_loss"))
    } else if index - holding.entry_index >= config.max_hold_days {
        Some((tick_floor(f64::from(bar.close)), "time"))
    } else if bar.action == Action::Sell {
        Some((sell_price(bar, config.fill), "signal"))
    } else if is_last {
        Some((sell_price(bar, config.fill), "final"))
    } else {
        None
    }
}

/// Apply the exit ladder to every holding, booking each closed position.
fn sell_phase(
    day: &TradingDay,
    index: usize,
    is_last: bool,
    config: &BacktestConfig,
    ledger: &mut Ledger,
) {
    let exits: Vec<(String, f64, &'static str)> = ledger
        .holdings
        .iter()
        .filter_map(|(ticker, holding)| {
            exit_decision(holding, day.bars.get(ticker), index, is_last, config)
                .map(|(price, reason)| (ticker.clone(), price, reason))
        })
        .collect();

    for (ticker, price, reason) in exits {
        book_sale(ledger, day.date, &ticker, price, reason);
    }
}

/// Book one sell: realize proceeds net of commission and tax, log the event and the
/// closed trade.
fn book_sale(ledger: &mut Ledger, date: NaiveDate, ticker: &str, price: f64, reason: &'static str) {
    let holding = ledger.holdings.remove(ticker).expect("ticker is held");
    let amount = price * holding.shares;
    let proceeds = amount - commission(amount) - amount * SELL_TAX_RATE;
    ledger.cash += proceeds;
    let pnl = proceeds - holding.cost;
    ledger.events.push(TradeEvent {
        date,
        ticker: ticker.to_string(),
        side: "Sell",
        reason,
        price,
        shares: holding.shares,
        amount,
        cash_after: ledger.cash,
    });
    ledger.trades.push(Trade {
        ticker: ticker.to_string(),
        entry_date: holding.entry_date,
        exit_date: date,
        entry_price: holding.entry_price,
        exit_price: price,
        shares: holding.shares,
        cost: holding.cost,
        proceeds,
        pnl,
        return_pct: pnl / holding.cost,
        exit_reason: reason,
    });
}

/// Rotate the weakest holdings out for clearly stronger challengers when the book is
/// full, pairing each eviction to a distinct above-threshold name whose edge beats it
/// by more than one round-trip cost. The freed slots are refilled by [`buy_phase`].
fn rotate_phase(day: &TradingDay, config: &BacktestConfig, ledger: &mut Ledger) {
    if ledger.holdings.len() < config.max_holdings {
        // Open slots already, so buy_phase can take challengers without evicting.
        return;
    }

    // Strongest above-threshold names we do not hold, best first.
    let mut challengers: Vec<f32> = day
        .bars
        .iter()
        .filter(|(ticker, bar)| {
            !ledger.holdings.contains_key(*ticker) && bar.score > config.threshold
        })
        .map(|(_, bar)| bar.score)
        .collect();
    challengers.sort_by(|left, right| right.total_cmp(left));

    // Holdings priced today, weakest first, the ones a challenger could displace.
    let mut weakest: Vec<(String, f32)> = ledger
        .holdings
        .keys()
        .filter_map(|ticker| day.bars.get(ticker).map(|bar| (ticker.clone(), bar.score)))
        .collect();
    weakest.sort_by(|left, right| left.1.total_cmp(&right.1));

    let hurdle = round_trip_cost();
    let mut rotated = Vec::new();
    for ((ticker, held_edge), challenger) in weakest.into_iter().zip(challengers) {
        // Weakest holding vs best remaining challenger: once this fails, no stronger
        // holding clears the hurdle against a weaker challenger either.
        if f64::from(challenger) - f64::from(held_edge) > hurdle {
            rotated.push(ticker);
        } else {
            break;
        }
    }

    for ticker in rotated {
        let price = sell_price(&day.bars[&ticker], config.fill);
        book_sale(ledger, day.date, &ticker, price, "rotate");
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

/// Fill open slots with the strongest above-threshold names not already held.
fn buy_phase(day: &TradingDay, index: usize, config: &BacktestConfig, ledger: &mut Ledger) {
    // Equal-weight target from equity at the buy phase start, so all of the day's
    // buys size against the same value.
    let equity = ledger.cash + holdings_value(&ledger.holdings, &day.bars);
    let target = equity / count_f64(config.max_holdings.max(1));

    let mut candidates: Vec<(&String, &DayBar)> = day
        .bars
        .iter()
        .filter(|(ticker, bar)| {
            !ledger.holdings.contains_key(*ticker) && bar.score > config.threshold
        })
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
        if ledger.holdings.len() >= config.max_holdings {
            break;
        }
        let price = buy_price(bar, config.fill);
        let shares = affordable_shares(target.min(ledger.cash), price, ledger.cash);
        if shares <= 0.0 {
            continue;
        }
        let amount = price * shares;
        let cost = amount + commission(amount);
        ledger.cash -= cost;
        ledger.events.push(TradeEvent {
            date: day.date,
            ticker: ticker.clone(),
            side: "Buy",
            reason: "",
            price,
            shares,
            amount,
            cash_after: ledger.cash,
        });
        ledger.holdings.insert(
            ticker.clone(),
            Holding {
                shares,
                cost,
                mark: f64::from(bar.close),
                entry_date: day.date,
                entry_price: price,
                entry_index: index,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::Fill;
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
            // Barriers wide enough never to fire, so these tests exercise the
            // model-Sell and final-day exits only.
            take_profit: 100.0,
            stop_loss: 0.99,
            max_hold_days: usize::MAX,
        }
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

    #[test]
    fn outranked_holding_rotates_out() {
        // One slot: buy A day 1, then B outranks A day 2, so A rotates out for B.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, Action::Buy, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.2, Action::Hold, 11.0, 12.0, 11.0));
        day2.bars
            .insert("B".to_string(), bar(0.9, Action::Buy, 50.0, 51.0, 50.0));
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        let report = run(&[day1, day2, day3], &config(1_000_000.0, 1));

        let rotated = report
            .trades
            .iter()
            .find(|trade| trade.exit_reason == "rotate")
            .expect("A should rotate out");
        assert_eq!(rotated.ticker, "A");
        assert!((rotated.exit_price - 12.0).abs() < 1e-9);
    }

    #[test]
    fn marginal_challenger_does_not_rotate() {
        // Challenger B (0.505) beats A (0.500) by less than the round-trip cost, so the
        // full book holds A rather than churning into B for no net gain.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, Action::Buy, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.500, Action::Hold, 11.0, 12.0, 11.0));
        day2.bars
            .insert("B".to_string(), bar(0.505, Action::Buy, 50.0, 51.0, 50.0));
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        let report = run(&[day1, day2, day3], &config(1_000_000.0, 1));

        assert!(report.trades.iter().all(|t| t.exit_reason != "rotate"));
    }

    fn barrier_config(take_profit: f64, stop_loss: f64, max_hold_days: usize) -> BacktestConfig {
        BacktestConfig {
            threshold: 0.0,
            fill: Fill::LowHigh,
            max_holdings: 1,
            starting_cash: 1_000_000.0,
            take_profit,
            stop_loss,
            max_hold_days,
        }
    }

    /// One ticker bought day 1 at low 10, then a trigger bar, then a filler day so the
    /// trigger is not the forced final-day exit.
    fn run_exit_scenario(trigger: DayBar, config: &BacktestConfig) -> BacktestReport {
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, Action::Buy, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars.insert("A".to_string(), trigger);
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        run(&[day1, day2, day3], config)
    }

    #[test]
    fn take_profit_exit_fills_at_the_barrier() {
        // High 12 crosses the +10% barrier (11); sell at the barrier tick.
        let report = run_exit_scenario(
            bar(0.0, Action::Hold, 10.0, 12.0, 11.0),
            &barrier_config(0.1, 0.5, 100),
        );

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, "take_profit");
        assert!((report.trades[0].exit_price - 11.0).abs() < 1e-9);
        assert!(report.trades[0].pnl > 0.0);
    }

    #[test]
    fn stop_loss_exit_fills_at_the_barrier() {
        // Low 8 crosses the -10% barrier (9); sell at the barrier tick.
        let report = run_exit_scenario(
            bar(0.0, Action::Hold, 8.0, 10.0, 9.0),
            &barrier_config(0.5, 0.1, 100),
        );

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, "stop_loss");
        assert!((report.trades[0].exit_price - 9.0).abs() < 1e-9);
        assert!(report.trades[0].pnl < 0.0);
    }

    #[test]
    fn time_exit_sells_at_the_horizon_close() {
        // Neither barrier touched, but the one-day hold limit is reached on day 2.
        let report = run_exit_scenario(
            bar(0.0, Action::Hold, 10.0, 11.0, 10.5),
            &barrier_config(0.5, 0.5, 1),
        );

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, "time");
        assert!((report.trades[0].exit_price - 10.5).abs() < 1e-9);
    }

    #[test]
    fn both_barriers_in_one_bar_takes_profit() {
        // Bar touches both the +10% and -10% barriers; the optimistic rule takes profit.
        let report = run_exit_scenario(
            bar(0.0, Action::Hold, 8.0, 12.0, 10.0),
            &barrier_config(0.1, 0.1, 100),
        );

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, "take_profit");
        assert!((report.trades[0].exit_price - 11.0).abs() < 1e-9);
    }
}
