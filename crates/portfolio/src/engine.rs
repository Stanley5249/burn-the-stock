use std::collections::HashMap;

use chrono::NaiveDate;

use super::pricing::{
    LOT, SELL_TAX_RATE, buy_price, commission, round_trip_cost, sell_price, tick_floor,
};
use super::types::{
    BacktestConfig, BacktestReport, DayBar, EquityPoint, ExitReason, Fill, Holding, Ledger, Side,
    Trade, TradeEvent, TradingDay, Weighting,
};

/// Run the simulation over `days` (ascending). Each day: run the exit ladder (barriers,
/// the time stop, a model Sell, the final-day liquidation), rotate the weakest holdings
/// out for clearly stronger names, buy into open slots, then mark to the closes.
#[must_use]
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
            if config.rotate {
                rotate_phase(day, config, &mut ledger);
            }
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

/// Exit price and reason for a position today, or `None` to keep holding. Take-profit wins
/// a both-touch bar; then stop-loss, the time barrier, then a model Sell. The shared ladder
/// the backtest and the live trader both sell on, so the two never drift. `days_held` is
/// trading days since entry.
#[must_use]
pub fn exit_decision(
    entry_price: f64,
    days_held: usize,
    bar: &DayBar,
    take_profit: f64,
    stop_loss: f64,
    max_hold_days: usize,
    fill: Fill,
) -> Option<(f64, ExitReason)> {
    let upper = entry_price * (1.0 + take_profit);
    let lower = entry_price * (1.0 - stop_loss);

    if f64::from(bar.high) >= upper {
        Some((tick_floor(upper), ExitReason::TakeProfit))
    } else if f64::from(bar.low) <= lower {
        Some((tick_floor(lower), ExitReason::StopLoss))
    } else if days_held >= max_hold_days {
        Some((tick_floor(f64::from(bar.close)), ExitReason::Time))
    } else if bar.score < 0.0 {
        Some((sell_price(bar, fill), ExitReason::Signal))
    } else {
        None
    }
}

/// Backtest wrapper over [`exit_decision`], adding the no-bar and final-day liquidations
/// that only the simulation has.
fn holding_exit(
    holding: &Holding,
    bar: Option<&DayBar>,
    index: usize,
    is_last: bool,
    config: &BacktestConfig,
) -> Option<(f64, ExitReason)> {
    let Some(bar) = bar else {
        // No bar today: only the final day can liquidate, at the last mark.
        return is_last.then_some((holding.mark, ExitReason::Final));
    };

    exit_decision(
        holding.entry_price,
        index - holding.entry_index,
        bar,
        config.take_profit,
        config.stop_loss,
        config.max_hold_days,
        config.fill,
    )
    .or_else(|| is_last.then_some((sell_price(bar, config.fill), ExitReason::Final)))
}

/// Apply the exit ladder to every holding, booking each closed position.
fn sell_phase(
    day: &TradingDay,
    index: usize,
    is_last: bool,
    config: &BacktestConfig,
    ledger: &mut Ledger,
) {
    let exits: Vec<(String, f64, ExitReason)> = ledger
        .holdings
        .iter()
        .filter_map(|(ticker, holding)| {
            holding_exit(holding, day.bars.get(ticker), index, is_last, config)
                .map(|(price, reason)| (ticker.clone(), price, reason))
        })
        .collect();

    for (ticker, price, reason) in exits {
        book_sale(ledger, day.date, &ticker, price, reason);
    }
}

/// Book one sell: realize proceeds net of commission and tax, log the event and the
/// closed trade.
fn book_sale(ledger: &mut Ledger, date: NaiveDate, ticker: &str, price: f64, reason: ExitReason) {
    let holding = ledger.holdings.remove(ticker).expect("ticker is held");
    let amount = price * holding.shares;
    let proceeds = amount - commission(amount) - amount * SELL_TAX_RATE;
    ledger.cash += proceeds;
    let pnl = proceeds - holding.cost;
    ledger.events.push(TradeEvent {
        date,
        ticker: ticker.to_string(),
        side: Side::Sell,
        reason: Some(reason),
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
        book_sale(ledger, day.date, &ticker, price, ExitReason::Rotate);
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

/// Per-ticker buy target for score weighting: split `cash` across the top candidates that
/// fit the open slots, in proportion to each name's positive score. Names beyond the open
/// slots are omitted, so the buy loop targets them at zero and skips them.
fn score_targets(
    candidates: &[(&String, &DayBar)],
    holdings: &HashMap<String, Holding>,
    cash: f64,
    config: &BacktestConfig,
) -> HashMap<String, f64> {
    let open_slots = config.max_holdings.saturating_sub(holdings.len());
    let chosen = &candidates[..open_slots.min(candidates.len())];

    let weight = |score: f32| f64::from(score).max(0.0);
    let total: f64 = chosen.iter().map(|(_, bar)| weight(bar.score)).sum();

    if total <= 0.0 {
        // No positive scores: split evenly so the cash still deploys.
        #[allow(clippy::cast_precision_loss)]
        let even = cash / (chosen.len().max(1) as f64);
        return chosen
            .iter()
            .map(|(ticker, _)| ((*ticker).clone(), even))
            .collect();
    }

    chosen
        .iter()
        .map(|(ticker, bar)| ((*ticker).clone(), cash * weight(bar.score) / total))
        .collect()
}

/// Whole-lot shares affordable for `budget`, trimmed until `cash` also covers the
/// commission. Zero when even one lot is out of reach.
#[must_use]
pub fn affordable_shares(budget: f64, price: f64, cash: f64) -> f64 {
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

/// Fill open slots with the strongest above-threshold names not already held. Equal
/// weighting targets `equity / max_holdings` per name; score weighting splits the cash
/// budget across the open slots in proportion to each name's score.
fn buy_phase(day: &TradingDay, index: usize, config: &BacktestConfig, ledger: &mut Ledger) {
    // Equal-weight target from equity at the buy phase start, so all of the day's
    // buys size against the same value.
    let equity = ledger.cash + holdings_value(&ledger.holdings, &day.bars);
    #[allow(clippy::cast_precision_loss)]
    let equal_target = equity / config.max_holdings.max(1) as f64;

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

    let score_targets = match config.weighting {
        Weighting::Equal => HashMap::new(),
        Weighting::Score => score_targets(&candidates, &ledger.holdings, ledger.cash, config),
    };

    for (ticker, bar) in candidates {
        if ledger.holdings.len() >= config.max_holdings {
            break;
        }
        let target = match config.weighting {
            Weighting::Equal => equal_target,
            // Names past the open slots are absent, so they target zero and skip below.
            Weighting::Score => score_targets.get(ticker).copied().unwrap_or(0.0),
        };
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
            side: Side::Buy,
            reason: None,
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

    fn bar(score: f32, low: f32, high: f32, close: f32) -> DayBar {
        DayBar {
            score,
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
            weighting: Weighting::Equal,
            starting_cash,
            // Barriers wide enough never to fire, so these tests exercise the
            // model-Sell and final-day exits only.
            take_profit: 100.0,
            stop_loss: 0.99,
            max_hold_days: usize::MAX,
            rotate: true,
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
            .insert("A".to_string(), bar(0.5, 10.0, 12.0, 11.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.1, 13.0, 16.0, 15.0));
        let mut day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };
        day3.bars
            .insert("A".to_string(), bar(0.0, 18.0, 20.0, 19.0));

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
            .insert("A".to_string(), bar(0.6, 10.0, 11.0, 11.0));
        day1.bars
            .insert("B".to_string(), bar(0.5, 100.0, 101.0, 100.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(-0.1, 19.0, 20.0, 20.0));
        day2.bars
            .insert("B".to_string(), bar(-0.1, 89.0, 90.0, 90.0));

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
            .insert("A".to_string(), bar(0.05, 10.0, 12.0, 11.0));
        let mut cfg = config(1_000_000.0, 10);
        cfg.threshold = 0.2;

        let report = run(&[day1], &cfg);

        assert_eq!(report.trade_count, 0);
        assert!((report.final_equity - 1_000_000.0).abs() < 1e-9);
    }

    #[test]
    fn score_weighting_buys_more_of_the_stronger_name() {
        // Same price, B scores 3x A, so score weighting sizes B's position larger.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(1.0, 10.0, 10.0, 10.0));
        day1.bars
            .insert("B".to_string(), bar(3.0, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(1.0, 10.0, 10.0, 10.0));
        day2.bars
            .insert("B".to_string(), bar(3.0, 10.0, 10.0, 10.0));

        let mut cfg = config(1_000_000.0, 2);
        cfg.weighting = Weighting::Score;

        let report = run(&[day1, day2], &cfg);

        let shares = |ticker: &str| {
            report
                .trades
                .iter()
                .find(|trade| trade.ticker == ticker)
                .expect("ticker traded")
                .shares
        };
        assert!(shares("B") > shares("A"));
    }

    #[test]
    fn outranked_holding_rotates_out() {
        // One slot: buy A day 1, then B outranks A day 2, so A rotates out for B.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.2, 11.0, 12.0, 11.0));
        day2.bars
            .insert("B".to_string(), bar(0.9, 50.0, 51.0, 50.0));
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        let report = run(&[day1, day2, day3], &config(1_000_000.0, 1));

        let rotated = report
            .trades
            .iter()
            .find(|trade| trade.exit_reason == ExitReason::Rotate)
            .expect("A should rotate out");
        assert_eq!(rotated.ticker, "A");
        assert!((rotated.exit_price - 12.0).abs() < 1e-9);
    }

    #[test]
    fn rotation_off_holds_through_a_stronger_challenger() {
        // Same setup as outranked_holding_rotates_out, but rotation disabled: A stays.
        let mut day1 = TradingDay {
            date: date(1),
            bars: HashMap::new(),
        };
        day1.bars
            .insert("A".to_string(), bar(0.5, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.2, 11.0, 12.0, 11.0));
        day2.bars
            .insert("B".to_string(), bar(0.9, 50.0, 51.0, 50.0));
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        let mut cfg = config(1_000_000.0, 1);
        cfg.rotate = false;
        let report = run(&[day1, day2, day3], &cfg);

        assert!(
            report
                .trades
                .iter()
                .all(|t| t.exit_reason != ExitReason::Rotate)
        );
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
            .insert("A".to_string(), bar(0.5, 10.0, 10.0, 10.0));
        let mut day2 = TradingDay {
            date: date(2),
            bars: HashMap::new(),
        };
        day2.bars
            .insert("A".to_string(), bar(0.500, 11.0, 12.0, 11.0));
        day2.bars
            .insert("B".to_string(), bar(0.505, 50.0, 51.0, 50.0));
        let day3 = TradingDay {
            date: date(3),
            bars: HashMap::new(),
        };

        let report = run(&[day1, day2, day3], &config(1_000_000.0, 1));

        assert!(
            report
                .trades
                .iter()
                .all(|t| t.exit_reason != ExitReason::Rotate)
        );
    }

    fn barrier_config(take_profit: f64, stop_loss: f64, max_hold_days: usize) -> BacktestConfig {
        BacktestConfig {
            threshold: 0.0,
            fill: Fill::LowHigh,
            max_holdings: 1,
            weighting: Weighting::Equal,
            starting_cash: 1_000_000.0,
            take_profit,
            stop_loss,
            max_hold_days,
            rotate: true,
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
            .insert("A".to_string(), bar(0.5, 10.0, 10.0, 10.0));
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
        let report = run_exit_scenario(bar(0.0, 10.0, 12.0, 11.0), &barrier_config(0.1, 0.5, 100));

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, ExitReason::TakeProfit);
        assert!((report.trades[0].exit_price - 11.0).abs() < 1e-9);
        assert!(report.trades[0].pnl > 0.0);
    }

    #[test]
    fn stop_loss_exit_fills_at_the_barrier() {
        // Low 8 crosses the -10% barrier (9); sell at the barrier tick.
        let report = run_exit_scenario(bar(0.0, 8.0, 10.0, 9.0), &barrier_config(0.5, 0.1, 100));

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, ExitReason::StopLoss);
        assert!((report.trades[0].exit_price - 9.0).abs() < 1e-9);
        assert!(report.trades[0].pnl < 0.0);
    }

    #[test]
    fn time_exit_sells_at_the_horizon_close() {
        // Neither barrier touched, but the one-day hold limit is reached on day 2.
        let report = run_exit_scenario(bar(0.0, 10.0, 11.0, 10.5), &barrier_config(0.5, 0.5, 1));

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, ExitReason::Time);
        assert!((report.trades[0].exit_price - 10.5).abs() < 1e-9);
    }

    #[test]
    fn both_barriers_in_one_bar_takes_profit() {
        // Bar touches both the +10% and -10% barriers; the optimistic rule takes profit.
        let report = run_exit_scenario(bar(0.0, 8.0, 12.0, 10.0), &barrier_config(0.1, 0.1, 100));

        assert_eq!(report.trade_count, 1);
        assert_eq!(report.trades[0].exit_reason, ExitReason::TakeProfit);
        assert!((report.trades[0].exit_price - 11.0).abs() < 1e-9);
    }
}
