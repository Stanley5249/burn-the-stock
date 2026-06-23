//! Taiwan trading calendar and session timing. Uses the TWSE holiday schedule (cached per
//! year) to decide which session the day's orders target and how fresh the data must be.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use chrono::{DateTime, Datelike, FixedOffset, NaiveDate, NaiveTime, Weekday};
use miette::{IntoDiagnostic, Result, WrapErr};
use stock_client::twse::Holiday;

/// `sim_stock` accepts orders before 1pm; a later run targets the next session.
pub const ORDER_CUTOFF: NaiveTime = NaiveTime::from_hms_opt(13, 0, 0).expect("valid time literal");

/// `sim_stock` is down for maintenance in this window, so the trader refuses to run.
const MAINTENANCE_START: NaiveTime =
    NaiveTime::from_hms_opt(15, 30, 0).expect("valid time literal");
const MAINTENANCE_END: NaiveTime = NaiveTime::from_hms_opt(16, 0, 0).expect("valid time literal");

/// Taiwan never observes DST, so a fixed +08:00 offset is exactly correct.
pub const TAIPEI_OFFSET: FixedOffset = FixedOffset::east_opt(8 * 3600).expect("valid offset");

/// True inside the `sim_stock` maintenance window `[15:30, 16:00)`.
#[must_use]
pub fn in_maintenance(now: DateTime<FixedOffset>) -> bool {
    let now = now.time();
    now >= MAINTENANCE_START && now < MAINTENANCE_END
}

/// Why a calendar day is or is not tradable, carrying the TWSE label when the schedule names
/// the day. Taiwan has had no makeup trading days since 2019, so a weekday the schedule does
/// not name always trades.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DayKind {
    Trading,
    Weekend,
    Holiday(String),
}

impl fmt::Display for DayKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DayKind::Trading => formatter.write_str("trading day"),
            DayKind::Weekend => formatter.write_str("weekend"),
            DayKind::Holiday(name) => write!(formatter, "holiday ({name})"),
        }
    }
}

/// The session the run's orders will execute in and the last completed session the model
/// needs data through. The model never reads the target session's own bar, so the data must
/// be current through the prior trading day.
#[derive(Clone, Copy, Debug)]
pub struct Session {
    pub target: NaiveDate,
    pub data_through: NaiveDate,
}

/// Trading calendar: weekends plus the TWSE closed dates are non-trading days. The named
/// days from the schedule are kept so the trader can report why a day is or is not tradable.
pub struct TradingCalendar {
    days: HashMap<NaiveDate, Holiday>,
}

impl TradingCalendar {
    #[must_use]
    pub fn new(holidays: impl IntoIterator<Item = Holiday>) -> Self {
        Self {
            days: holidays
                .into_iter()
                .map(|holiday| (holiday.date, holiday))
                .collect(),
        }
    }

    #[must_use]
    pub fn day_kind(&self, date: NaiveDate) -> DayKind {
        if let Some(holiday) = self.days.get(&date) {
            if holiday.closed {
                DayKind::Holiday(holiday.name.clone())
            } else {
                DayKind::Trading
            }
        } else if matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            DayKind::Weekend
        } else {
            DayKind::Trading
        }
    }

    #[must_use]
    pub fn is_trading_day(&self, date: NaiveDate) -> bool {
        matches!(self.day_kind(date), DayKind::Trading)
    }

    #[must_use]
    pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate {
        date.iter_days()
            .skip(1)
            .find(|day| self.is_trading_day(*day))
            .expect("a trading day always follows within a week")
    }

    #[must_use]
    pub fn prev_trading_day(&self, date: NaiveDate) -> NaiveDate {
        std::iter::successors(date.pred_opt(), NaiveDate::pred_opt)
            .find(|day| self.is_trading_day(*day))
            .expect("a trading day always precedes within a week")
    }

    /// Resolve which session the current run's orders hit. Orders placed before 1pm on a
    /// trading day are for that day; otherwise they queue to the next session.
    #[must_use]
    pub fn session(&self, now: DateTime<FixedOffset>) -> Session {
        let today = now.date_naive();
        let target = if self.is_trading_day(today) && now.time() < ORDER_CUTOFF {
            today
        } else {
            self.next_trading_day(today)
        };
        Session {
            target,
            data_through: self.prev_trading_day(target),
        }
    }
}

/// Load the named days for `year` from the per-year cache, fetching from TWSE on a miss.
/// The endpoint serves the current year only, so a missing cache for a past year falls back
/// to weekend-only (logged), which is fine since the only near-boundary lookback risk is a
/// late-December holiday.
// ponytail: year-boundary lookback falls back to weekday-only if prior-year cache is missing.
///
/// # Errors
/// Cache read/write or the TWSE fetch failing.
async fn load_or_fetch(year: i32, current_year: i32, cache_dir: &Path) -> Result<Vec<Holiday>> {
    let path = cache_dir.join(format!("holidays-{year}.json"));

    if path.exists() {
        let text = std::fs::read_to_string(&path)
            .into_diagnostic()
            .wrap_err_with(|| format!("read holiday cache {}", path.display()))?;
        return serde_json::from_str(&text)
            .into_diagnostic()
            .wrap_err("parse holiday cache");
    }

    if year != current_year {
        tracing::warn!(
            year,
            "no holiday cache and TWSE serves only the current year; using weekends only"
        );
        return Ok(Vec::new());
    }

    let holidays = stock_client::twse::fetch_holidays().await?;
    std::fs::create_dir_all(cache_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("create holiday cache dir {}", cache_dir.display()))?;
    let text = serde_json::to_string(&holidays)
        .into_diagnostic()
        .wrap_err("serialize holiday cache")?;
    std::fs::write(&path, text)
        .into_diagnostic()
        .wrap_err_with(|| format!("write holiday cache {}", path.display()))?;

    Ok(holidays)
}

/// Build the calendar for `year`, merging the prior year's cache if it is on disk so
/// `prev_trading_day` is accurate across the January boundary.
///
/// # Errors
/// Cache read/write or the TWSE fetch failing.
pub async fn build(year: i32, cache_dir: &Path) -> Result<TradingCalendar> {
    let mut holidays = load_or_fetch(year, year, cache_dir).await?;
    let prior = cache_dir.join(format!("holidays-{}.json", year - 1));
    if prior.exists() {
        holidays.extend(load_or_fetch(year - 1, year, cache_dir).await?);
    }
    Ok(TradingCalendar::new(holidays))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(date: &str, hour: u32, minute: u32) -> DateTime<FixedOffset> {
        NaiveDate::parse_from_str(date, "%Y-%m-%d")
            .unwrap()
            .and_hms_opt(hour, minute, 0)
            .unwrap()
            .and_local_timezone(TAIPEI_OFFSET)
            .unwrap()
    }

    fn date(date: &str) -> NaiveDate {
        NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap()
    }

    fn holiday(date_str: &str, name: &str, closed: bool) -> Holiday {
        Holiday {
            date: date(date_str),
            name: name.to_string(),
            closed,
        }
    }

    /// Calendar with Lunar New Year week, a holiday Monday (2026-06-22), and the post-holiday
    /// trading marker, so the scenarios below exercise weekends, holidays, markers, and gaps.
    fn calendar() -> TradingCalendar {
        TradingCalendar::new([
            holiday("2026-02-16", "農曆除夕及春節", true),
            holiday("2026-02-17", "農曆除夕及春節", true),
            holiday("2026-02-18", "農曆除夕及春節", true),
            holiday("2026-02-19", "農曆除夕及春節", true),
            holiday("2026-02-20", "農曆除夕及春節", true),
            holiday("2026-02-23", "農曆春節後開始交易日", false),
            holiday("2026-06-22", "測試假日", true),
        ])
    }

    #[test]
    fn day_kind_distinguishes_weekend_holiday_trading() {
        let calendar = calendar();
        assert_eq!(calendar.day_kind(date("2026-06-14")), DayKind::Weekend);
        assert!(matches!(
            calendar.day_kind(date("2026-06-22")),
            DayKind::Holiday(_)
        ));
        assert_eq!(calendar.day_kind(date("2026-02-23")), DayKind::Trading);
        assert_eq!(calendar.day_kind(date("2026-06-23")), DayKind::Trading);
    }

    #[test]
    fn sunday_targets_monday_needs_friday() {
        // 2026-06-14 is a Sunday; Monday 06-15 is the next session, prior session Friday 06-12.
        let session = calendar().session(at("2026-06-14", 9, 0));
        assert_eq!(session.target.to_string(), "2026-06-15");
        assert_eq!(session.data_through.to_string(), "2026-06-12");
    }

    #[test]
    fn monday_before_cutoff_targets_today() {
        let session = calendar().session(at("2026-06-15", 9, 0));
        assert_eq!(session.target.to_string(), "2026-06-15");
        assert_eq!(session.data_through.to_string(), "2026-06-12");
    }

    #[test]
    fn monday_after_cutoff_targets_tuesday() {
        let session = calendar().session(at("2026-06-15", 13, 1));
        assert_eq!(session.target.to_string(), "2026-06-16");
        assert_eq!(session.data_through.to_string(), "2026-06-15");
    }

    #[test]
    fn holiday_monday_then_tuesday_needs_friday() {
        // 2026-06-22 is a closed Monday; Tuesday 06-23 before cutoff trades, prior session is
        // Friday 06-19 because Monday was a holiday.
        let session = calendar().session(at("2026-06-23", 9, 0));
        assert_eq!(session.target.to_string(), "2026-06-23");
        assert_eq!(session.data_through.to_string(), "2026-06-19");
    }

    #[test]
    fn maintenance_window() {
        assert!(in_maintenance(at("2026-06-23", 15, 45)));
        assert!(!in_maintenance(at("2026-06-23", 16, 0)));
        assert!(!in_maintenance(at("2026-06-23", 15, 29)));
    }
}
