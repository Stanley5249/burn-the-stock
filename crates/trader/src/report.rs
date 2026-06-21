//! Render the day's plan as one grouped block for the terminal.

use chrono::NaiveDate;

use crate::plan::{Buy, Sell};

/// Build the day's plan summary as one string, printed in a single write.
pub fn report(
    today: NaiveDate,
    settled_cash: f64,
    budget: f64,
    holdings: usize,
    candidates: usize,
    sells: &[Sell],
    buys: &[Buy],
) -> String {
    use std::fmt::Write as _;

    let proceeds: f64 = sells.iter().map(|sell| sell.proceeds).sum();
    let cost: f64 = buys.iter().map(|buy| buy.cost).sum();

    let mut out = String::new();
    let _ = writeln!(out, "Live plan {today}");
    let _ = writeln!(out, "  settled cash : {settled_cash:.0}");
    let _ = writeln!(out, "  buy budget   : {budget:.0}");
    let _ = writeln!(out, "  holdings     : {holdings}");
    let _ = writeln!(out, "  candidates   : {candidates}");
    let _ = writeln!(out, "  Sells ({}), proceeds {proceeds:.0}", sells.len());
    for sell in sells {
        let _ = writeln!(
            out,
            "    {:<8} {:>7} @ {:.2}  [{}]",
            sell.code, sell.shares, sell.price, sell.reason
        );
    }
    let _ = writeln!(out, "  Buys ({}), cost {cost:.0}", buys.len());
    for buy in buys {
        let _ = writeln!(
            out,
            "    {:<8} {:>7} @ {:.2}  ({:.0})",
            buy.code, buy.shares, buy.price, buy.cost
        );
    }
    out
}
