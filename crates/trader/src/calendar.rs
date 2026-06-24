//! Taiwan trading calendar and session timing. Uses the TWSE holiday schedule (cached per
//! year) to decide which session the day's orders target and how fresh the data must be.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use chrono::{DateTime, Datelike, FixedOffset, NaiveDate, NaiveTime, Weekday};
use miette::{IntoDiagnostic, Result, WrapErr};
use stock_client::twse::Holiday;
use walkdir::WalkDir;

/// sim stock accepts orders before 1pm; a later run targets the next session.
pub const ORDER_CUTOFF: NaiveTime = NaiveTime::from_hms_opt(13, 0, 0).unwrap();

/// sim stock is down for maintenance in this window, so the trader refuses to run.
const MAINTENANCE_START: NaiveTime = NaiveTime::from_hms_opt(15, 30, 0).unwrap();
const MAINTENANCE_END: NaiveTime = NaiveTime::from_hms_opt(16, 0, 0).unwrap();

/// Taiwan never observes DST, so a fixed +08:00 offset is exactly correct.
pub const TAIPEI_OFFSET: FixedOffset = FixedOffset::east_opt(8 * 3600).expect("valid offset");

/// True inside the sim stock maintenance window `[15:30, 16:00)`.
#[must_use]
pub fn in_maintenance(now: DateTime<FixedOffset>) -> bool {
    let now = now.time();
    MAINTENANCE_START <= now && now < MAINTENANCE_END
}

/// Why a calendar day is or is not tradable, carrying the TWSE label when the schedule names
/// the day. Taiwan has had no makeup trading days since 2019, so a weekday the schedule does
/// not name always trades.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DayKind {
    Trading(Option<String>),
    Weekend,
    Holiday(String),
}

impl fmt::Display for DayKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DayKind::Trading(None) => write!(f, "trading day"),
            DayKind::Trading(Some(desc)) => write!(f, "trading day ({desc})"),
            DayKind::Weekend => write!(f, "weekend"),
            DayKind::Holiday(desc) => write!(f, "holiday ({desc})"),
        }
    }
}

/// The session the run's orders will execute in and the last completed session the model
/// needs data through. The model never reads the target session's own bar, so the data must
/// be current through the prior trading day.
#[derive(Clone, Copy, Debug)]
pub struct Session {
    pub date: NaiveDate,
    pub prior: NaiveDate,
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
            let desc = holiday.desc.clone();
            if holiday.closed {
                DayKind::Holiday(desc)
            } else {
                DayKind::Trading(Some(desc))
            }
        } else if matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
            DayKind::Weekend
        } else {
            DayKind::Trading(None)
        }
    }

    #[must_use]
    pub fn is_trading_day(&self, date: NaiveDate) -> bool {
        matches!(self.day_kind(date), DayKind::Trading(_))
    }

    #[must_use]
    pub fn next_trading_day(&self, date: NaiveDate) -> NaiveDate {
        date.iter_days()
            .skip(1)
            .take(30)
            .find(|day| self.is_trading_day(*day))
            .expect("a trading day always follows within a month")
    }

    #[must_use]
    pub fn prev_trading_day(&self, date: NaiveDate) -> NaiveDate {
        std::iter::successors(date.pred_opt(), NaiveDate::pred_opt)
            .take(30)
            .find(|day| self.is_trading_day(*day))
            .expect("a trading day always precedes within a month")
    }

    /// Build the calendar by fetching the current year's holidays from TWSE, caching them if
    /// not already on disk, then loading all cached years via walkdir.
    ///
    /// # Errors
    /// TWSE fetch, cache read/write.
    pub async fn build(cache_dir: &Path) -> Result<Self> {
        let dir = cache_dir.join("twse").join("holidays");

        let fetched = stock_client::twse::fetch_holidays().await?;
        if let Some(first) = fetched.first() {
            let path = dir.join(format!("{}.json", first.date.year()));
            if !path.exists() {
                std::fs::create_dir_all(&dir)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("create holiday cache dir {}", dir.display()))?;
                let text = serde_json::to_string(&fetched)
                    .into_diagnostic()
                    .wrap_err("serialize holiday cache")?;
                std::fs::write(&path, text)
                    .into_diagnostic()
                    .wrap_err_with(|| format!("write holiday cache {}", path.display()))?;
            }
        }

        let mut holidays = Vec::new();
        if dir.exists() {
            for entry in WalkDir::new(&dir).min_depth(1).max_depth(1) {
                let entry = entry.into_diagnostic().wrap_err("walk holiday cache")?;
                if entry.path().extension().is_some_and(|e| e == "json") {
                    let text = std::fs::read_to_string(entry.path())
                        .into_diagnostic()
                        .wrap_err_with(|| format!("read {}", entry.path().display()))?;
                    let year_holidays: Vec<Holiday> = serde_json::from_str(&text)
                        .into_diagnostic()
                        .wrap_err_with(|| format!("parse {}", entry.path().display()))?;
                    holidays.extend(year_holidays);
                }
            }
        }

        Ok(Self::new(holidays))
    }

    /// Resolve which session the current run's orders hit. Orders placed before 1pm on a
    /// trading day are for that day; otherwise they queue to the next session.
    #[must_use]
    pub fn session(&self, now: DateTime<FixedOffset>) -> Session {
        let today = now.date_naive();
        let date = if self.is_trading_day(today) && now.time() < ORDER_CUTOFF {
            today
        } else {
            self.next_trading_day(today)
        };
        Session {
            date,
            prior: self.prev_trading_day(date),
        }
    }
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
            desc: name.to_string(),
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
        assert_eq!(
            calendar.day_kind(date("2026-02-23")),
            DayKind::Trading(None)
        );
        assert_eq!(
            calendar.day_kind(date("2026-06-23")),
            DayKind::Trading(None)
        );
    }

    #[test]
    fn sunday_targets_monday_needs_friday() {
        // 2026-06-14 is a Sunday; Monday 06-15 is the next session, prior session Friday 06-12.
        let session = calendar().session(at("2026-06-14", 9, 0));
        assert_eq!(session.date.to_string(), "2026-06-15");
        assert_eq!(session.prior.to_string(), "2026-06-12");
    }

    #[test]
    fn monday_before_cutoff_targets_today() {
        let session = calendar().session(at("2026-06-15", 9, 0));
        assert_eq!(session.date.to_string(), "2026-06-15");
        assert_eq!(session.prior.to_string(), "2026-06-12");
    }

    #[test]
    fn monday_after_cutoff_targets_tuesday() {
        let session = calendar().session(at("2026-06-15", 13, 1));
        assert_eq!(session.date.to_string(), "2026-06-16");
        assert_eq!(session.prior.to_string(), "2026-06-15");
    }

    #[test]
    fn holiday_monday_then_tuesday_needs_friday() {
        // 2026-06-22 is a closed Monday; Tuesday 06-23 before cutoff trades, prior session is
        // Friday 06-19 because Monday was a holiday.
        let session = calendar().session(at("2026-06-23", 9, 0));
        assert_eq!(session.date.to_string(), "2026-06-23");
        assert_eq!(session.prior.to_string(), "2026-06-19");
    }

    #[test]
    fn maintenance_window() {
        assert!(in_maintenance(at("2026-06-23", 15, 45)));
        assert!(!in_maintenance(at("2026-06-23", 16, 0)));
        assert!(!in_maintenance(at("2026-06-23", 15, 29)));
    }
}
