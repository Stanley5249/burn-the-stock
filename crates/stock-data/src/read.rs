//! Lazy, chainable reads over the OHLCV parquet. One scan implementation behind a
//! builder so every consumer composes only the slice it needs and collects once. The
//! same builder wraps any `LazyFrame`, so the standardized long frame and the merged
//! download frame share the date filters, column appends, and sink.

use std::path::Path;

use chrono::{Duration, NaiveDate};
use miette::{IntoDiagnostic, Result, miette};
use polars::prelude::*;

use crate::schema::DATE;

/// A pending query over history bars. Methods accumulate `LazyFrame` ops; `lazy`,
/// `collect`, `sink`, and `last_date` are the terminals.
#[derive(Clone)]
pub struct History {
    lazy: LazyFrame,
}

impl History {
    /// Scan the parquet and normalize `date` to the `Date` dtype.
    ///
    /// # Errors
    /// If the path is not valid UTF-8 or the parquet cannot be scanned.
    pub fn scan(path: &Path) -> Result<Self> {
        let source = PlRefPath::try_from_path(path).into_diagnostic()?;
        let lazy = LazyFrame::scan_parquet(source, ScanArgsParquet::default())
            .into_diagnostic()?
            .with_column(col(DATE).cast(DataType::Date));
        Ok(Self { lazy })
    }

    /// Wrap an already-built frame, e.g. the standardized or labeled long frame.
    #[must_use]
    pub fn from_lazy(lazy: LazyFrame) -> Self {
        Self { lazy }
    }

    /// Keep only bars on or after `start`.
    #[must_use]
    pub fn since(self, start: NaiveDate) -> Self {
        Self {
            lazy: self.lazy.filter(col(DATE).gt_eq(lit(start))),
        }
    }

    /// Keep only bars strictly before `end`. Exclusive, so `until(c)` and `since(c)`
    /// partition a frame cleanly at `c`.
    #[must_use]
    pub fn until(self, end: NaiveDate) -> Self {
        Self {
            lazy: self.lazy.filter(col(DATE).lt(lit(end))),
        }
    }

    /// Keep the last `lookback` calendar days, measured back from the freshest bar. The
    /// per-date cross-section stays intact since each retained date keeps its full universe.
    ///
    /// # Errors
    /// If the last-date probe cannot be collected.
    pub fn recent(self, lookback: i64) -> Result<Self> {
        let cutoff = self.last_date()? - Duration::days(lookback);
        Ok(self.since(cutoff))
    }

    /// Append or replace columns, e.g. label or `code`/`market` tags.
    #[must_use]
    pub fn with(self, exprs: Vec<Expr>) -> Self {
        Self {
            lazy: self.lazy.with_columns(exprs),
        }
    }

    /// The accumulated lazy frame.
    pub fn lazy(self) -> LazyFrame {
        self.lazy
    }

    /// Collect the accumulated lazy frame.
    ///
    /// # Errors
    /// If the query cannot be collected.
    pub fn collect(self) -> Result<DataFrame> {
        self.lazy.collect().into_diagnostic()
    }

    /// Collect and write the frame to `path` as a zstd parquet, creating parent dirs.
    ///
    /// # Errors
    /// If the query cannot be collected or the file cannot be written.
    pub fn sink(self, path: &Path) -> Result<()> {
        let mut frame = self.lazy.collect().into_diagnostic()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).into_diagnostic()?;
        }
        let mut file = std::fs::File::create(path).into_diagnostic()?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut frame)
            .into_diagnostic()?;
        Ok(())
    }

    /// The most recent dated bar, used to verify the data is fresh enough for trading.
    ///
    /// # Errors
    /// If the probe cannot be collected or the parquet holds no dated rows.
    pub fn last_date(&self) -> Result<NaiveDate> {
        self.lazy
            .clone()
            .select([col(DATE).max()])
            .collect()
            .into_diagnostic()?
            .column(&DATE)
            .into_diagnostic()?
            .date()
            .into_diagnostic()?
            .as_date_iter()
            .flatten()
            .next()
            .ok_or_else(|| miette!("parquet has no dated rows"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> History {
        let dates: Vec<NaiveDate> = (1..=5)
            .map(|day| NaiveDate::from_ymd_opt(2024, 1, day).unwrap())
            .collect();
        let df = df!(DATE => dates, "close" => &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        History::from_lazy(df.lazy())
    }

    fn dates_of(history: History) -> Vec<NaiveDate> {
        history
            .collect()
            .unwrap()
            .column(&DATE)
            .unwrap()
            .date()
            .unwrap()
            .as_date_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn since_and_until_partition_at_cutoff() {
        let cut = NaiveDate::from_ymd_opt(2024, 1, 3).unwrap();

        // since is inclusive, until is exclusive, so they partition at the cutoff.
        let before = dates_of(frame().until(cut));
        let after = dates_of(frame().since(cut));
        assert_eq!(before.len(), 2);
        assert_eq!(after.len(), 3);
        assert_eq!(after[0], cut);
    }

    #[test]
    fn last_date_is_the_freshest_bar() {
        assert_eq!(
            frame().last_date().unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 5).unwrap()
        );
    }
}
