//! Render the day's plan for the terminal, one block per phase.

use chrono::NaiveDate;
use stock_client::types::Profile;

use crate::plan::{Buy, Sell};

/// Build the sell phase summary as one string, printed in a single write.
pub fn report_sells(
    today: NaiveDate,
    budget: f64,
    holdings: usize,
    profile: &Profile,
    sells: &[Sell],
) -> String {
    use std::fmt::Write as _;

    let proceeds: f64 = sells.iter().map(|sell| sell.proceeds).sum();

    let mut out = String::new();
    let _ = writeln!(out, "Live plan {today}");
    let _ = writeln!(out, "  usable cash  : {:.0}", profile.usable_cash);
    let _ = writeln!(out, "  total assets : {:.0}", profile.total_assets);
    let _ = writeln!(out, "  cum. return  : {:.3}%", profile.cumulative_return);
    let _ = writeln!(out, "  buy budget   : {budget:.0}");
    let _ = writeln!(out, "  holdings     : {holdings}");
    let _ = writeln!(out, "  Sells ({}), proceeds {proceeds:.0}", sells.len());

    for sell in sells {
        let _ = writeln!(
            out,
            "    {:<8} {:>5} 張 @ {:.2}",
            sell.code, sell.lots, sell.price
        );
    }

    out
}

/// Build the dry-run summary: the model's picks without live quotes, printed in a single
/// write. No prices, since the order flow and Fugle quotes are skipped.
pub fn report_candidates(
    today: NaiveDate,
    holdings: usize,
    candidates: &[(String, f32)],
) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "Dry run {today}");
    let _ = writeln!(out, "  holdings   : {holdings}");
    let _ = writeln!(out, "  candidates : {}", candidates.len());

    for (code, score) in candidates {
        let _ = writeln!(out, "    {code:<8} {score:>8.4}");
    }

    out
}

/// Build the buy phase summary as one string, printed in a single write.
pub fn report_buys(candidates: usize, buys: &[Buy]) -> String {
    use std::fmt::Write as _;

    let cost: f64 = buys.iter().map(|buy| buy.cost).sum();

    let mut out = String::new();
    let _ = writeln!(out, "  candidates   : {candidates}");
    let _ = writeln!(out, "  Buys ({}), cost {cost:.0}", buys.len());

    for buy in buys {
        let _ = writeln!(
            out,
            "    {:<8} {:>5} 張 @ {:.2}  ({:.0})",
            buy.code, buy.lots, buy.price, buy.cost
        );
    }

    out
}
