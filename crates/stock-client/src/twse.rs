//! TWSE open-data holiday schedule. Returns the named special days for the current year,
//! the authoritative source for the trading calendar (holidays, 補假 makeup days, and the
//! first/last trading-day markers around long closures).

use chrono::NaiveDate;
use miette::{IntoDiagnostic, Result, WrapErr};
use serde::{Deserialize, Serialize};

use crate::urls::twse as urls;

/// A named day from the schedule: either a closed day or a trading-day marker. The `name` is
/// the TWSE label for the day (e.g. the holiday name or "農曆春節前最後交易日").
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Holiday {
    pub date: NaiveDate,
    pub desc: String,
    pub closed: bool,
}

/// One raw row of the TWSE holiday schedule.
#[derive(Clone, Debug, Deserialize)]
struct HolidayEntry {
    #[serde(rename = "Name")]
    name: String,
    /// ROC-calendar date, `YYYMMDD` (e.g. `1150101` is ROC year 115 = 2026).
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Description")]
    description: String,
}

impl HolidayEntry {
    /// A row is a closed day unless its description marks an open trading day. The schedule
    /// mixes closures (`放假`/`補假`/`市場無交易`/春節) with first/last-trading-day markers.
    // ponytail: closed unless desc says 開始交易/最後交易; if TWSE rephrases, switch to a 放假/補假 allowlist.
    fn is_closed(&self) -> bool {
        !(self.description.contains("開始交易") || self.description.contains("最後交易"))
    }

    /// Parse the ROC `YYYMMDD` date to Gregorian. ROC year + 1911 = Gregorian year.
    fn parse_date(&self) -> Option<NaiveDate> {
        let digits = self.date.trim();
        let split = digits.len().checked_sub(4)?;
        let roc_year: i32 = digits.get(..split)?.parse().ok()?;
        let month: u32 = digits.get(split..split + 2)?.parse().ok()?;
        let day: u32 = digits.get(split + 2..)?.parse().ok()?;
        NaiveDate::from_ymd_opt(roc_year + 1911, month, day)
    }

    fn into_holiday(self) -> Option<Holiday> {
        Some(Holiday {
            date: self.parse_date()?,
            closed: self.is_closed(),
            desc: self.name,
        })
    }
}

/// Fetch the named special days from TWSE open data. The endpoint serves the current year
/// only, so the returned dates all fall in the current Gregorian year.
///
/// # Errors
/// Network failure or a response that does not decode as the holiday schedule.
pub async fn fetch_holidays() -> Result<Vec<Holiday>> {
    let entries: Vec<HolidayEntry> = reqwest::get(urls::HOLIDAY_SCHEDULE)
        .await
        .into_diagnostic()
        .wrap_err("fetch twse holiday schedule")?
        .error_for_status()
        .into_diagnostic()?
        .json()
        .await
        .into_diagnostic()
        .wrap_err("decode twse holiday schedule")?;

    Ok(entries
        .into_iter()
        .filter_map(HolidayEntry::into_holiday)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(date: &str, description: &str) -> HolidayEntry {
        HolidayEntry {
            name: String::new(),
            date: date.to_string(),
            description: description.to_string(),
        }
    }

    #[test]
    fn roc_date_parses_to_gregorian() {
        assert_eq!(
            entry("1150101", "依規定放假1日。").parse_date(),
            NaiveDate::from_ymd_opt(2026, 1, 1)
        );
    }

    #[test]
    fn trading_markers_are_open() {
        assert!(!entry("1150102", "國曆新年開始交易。").is_closed());
        assert!(!entry("1150211", "農曆春節前最後交易。").is_closed());
        assert!(!entry("1150223", "農曆春節後開始交易。").is_closed());
    }

    #[test]
    fn closures_are_closed() {
        assert!(entry("1150101", "依規定放假1日。").is_closed());
        assert!(entry("1150227", "於2月27日補假。").is_closed());
        assert!(entry("1150212", "").is_closed());
    }
}
