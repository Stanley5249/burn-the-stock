use super::pricing::round_trip_cost;
use super::types::{BacktestReport, ExitReason, Fill, Trade};

/// Window-level context for the summary, the parts the report itself does not carry.
pub struct RenderContext {
    pub tickers: usize,
    pub windows_scored: usize,
    pub threshold: f32,
    pub fill: Fill,
}

/// NT$ with thousands separators, no decimals, e.g. `NT$100,000,000`.
fn format_ntd(value: f64) -> String {
    let rounded = format!("{value:.0}");
    let (sign, digits) = rounded
        .strip_prefix('-')
        .map_or(("", rounded.as_str()), |rest| ("-", rest));

    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    let len = digits.len();
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (len - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(char::from(byte));
    }

    format!("NT${sign}{grouped}")
}

/// Count trades by exit reason, e.g.
/// `take-profit 12 / stop-loss 9 / time 5 / signal 2 / rotate 4 / final 1`.
fn exit_tally(trades: &[Trade]) -> String {
    let count = |reason| trades.iter().filter(|t| t.exit_reason == reason).count();
    format!(
        "take-profit {} / stop-loss {} / time {} / signal {} / rotate {} / final {}",
        count(ExitReason::TakeProfit),
        count(ExitReason::StopLoss),
        count(ExitReason::Time),
        count(ExitReason::Signal),
        count(ExitReason::Rotate),
        count(ExitReason::Final),
    )
}

/// Build the grouped summary as one string, so the caller prints it in a single write.
#[must_use]
pub fn summary(report: &BacktestReport, context: &RenderContext) -> String {
    use std::fmt::Write as _;

    let fill = match context.fill {
        Fill::LowHigh => "low/high (optimistic)",
        Fill::Open => "open (pessimistic)",
    };
    let dates = match (report.equity_curve.first(), report.equity_curve.last()) {
        (Some(first), Some(last)) => format!("{} -> {}", first.date, last.date),
        _ => "none".to_string(),
    };
    let profit_factor = if report.profit_factor.is_finite() {
        format!("{:.2}", report.profit_factor)
    } else {
        "inf".to_string()
    };
    let exits = exit_tally(&report.trades);

    let mut out = String::new();
    let _ = writeln!(out, "Backtest summary");
    let _ = writeln!(out, "  Window");
    summary_row(&mut out, "dates", &dates);
    summary_row(&mut out, "trading days", &report.trading_days.to_string());
    summary_row(&mut out, "tickers scored", &context.tickers.to_string());
    summary_row(
        &mut out,
        "windows scored",
        &context.windows_scored.to_string(),
    );
    summary_row(
        &mut out,
        "buy gate",
        &format!("score > {:.2}", context.threshold),
    );
    summary_row(
        &mut out,
        "rotate hurdle",
        &format!(
            "{:.3}% edge gain (one round trip)",
            round_trip_cost() * 100.0
        ),
    );
    summary_row(&mut out, "fills", fill);
    let _ = writeln!(out, "  Performance");
    summary_row(&mut out, "starting cash", &format_ntd(report.starting_cash));
    summary_row(&mut out, "final equity", &format_ntd(report.final_equity));
    summary_row(
        &mut out,
        "cumulative",
        &format!("{:+.2}%", report.cumulative_return * 100.0),
    );
    summary_row(
        &mut out,
        "annualized",
        &format!("{:+.2}%", report.annualized_return * 100.0),
    );
    summary_row(
        &mut out,
        "sharpe",
        &format!("{:.2}  (daily returns, annualized)", report.sharpe),
    );
    summary_row(&mut out, "trades", &report.trade_count.to_string());
    summary_row(&mut out, "exits", &exits);
    summary_row(
        &mut out,
        "win rate",
        &format!("{:.1}%  (trades closed in profit)", report.win_rate * 100.0),
    );
    summary_row(
        &mut out,
        "profit factor",
        &format!("{profit_factor}  (gross profit / gross loss)"),
    );
    summary_row(
        &mut out,
        "average win / loss",
        &format!(
            "{:+.2}% / {:+.2}%  (return on cost)",
            report.avg_win_return * 100.0,
            report.avg_loss_return * 100.0
        ),
    );
    let _ = writeln!(
        out,
        "  Note: annualized is naive linear (x252 per day), unreliable under ~30 trading days"
    );

    out
}

/// Append one aligned `label : value` row under a summary section.
fn summary_row(out: &mut String, label: &str, value: &str) {
    use std::fmt::Write as _;
    let _ = writeln!(out, "    {label:<18}: {value}");
}
