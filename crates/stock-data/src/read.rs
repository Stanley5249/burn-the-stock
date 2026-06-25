//! Lazy, chainable reads over the OHLCV parquet. One scan implementation behind a
//! builder so every consumer composes only the slice it needs and collects once. The
//! same builder wraps any `LazyFrame`, so the standardized long frame and the merged
//! download frame share the date filters, column appends, and sink.

use std::path::Path;

use chrono::{Duration, NaiveDate};
use miette::{IntoDiagnostic, Result, miette};
use polars::prelude::*;

use crate::schema::{CLOSE, CODE, DATE, HIGH, LOW, MARKET, OPEN, VOLUME};

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
        let path = PlRefPath::try_from_path(path).into_diagnostic()?;

        let lazy = LazyFrame::scan_parquet(path, ScanArgsParquet::default())
            .into_diagnostic()?
            .select([
                col(MARKET).cast(DataType::String),
                col(CODE).cast(DataType::String),
                col(DATE).cast(DataType::Date),
                col(OPEN).cast(DataType::Float32),
                col(HIGH).cast(DataType::Float32),
                col(LOW).cast(DataType::Float32),
                col(CLOSE).cast(DataType::Float32),
                col(VOLUME).cast(DataType::Float32),
            ]);

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

    /// Write the frame to `path` as a zstd parquet, creating parent dirs, and return the
    /// materialized frame. The sink write and the in-memory collect run in one pass so the
    /// shared scan executes once, sparing callers a re-read of the file they just wrote.
    ///
    /// # Errors
    /// If the query cannot be collected or the file cannot be written.
    pub fn sink(self, path: &Path) -> Result<Self> {
        let sink = self
            .lazy
            .clone()
            .sink(
                SinkDestination::File {
                    target: SinkTarget::Path(PlRefPath::try_from_path(path).into_diagnostic()?),
                },
                FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions {
                    compression: ParquetCompression::Zstd(None),
                    ..Default::default()
                })),
                UnifiedSinkArgs {
                    mkdir: true,
                    maintain_order: false,
                    ..Default::default()
                },
            )
            .into_diagnostic()?;

        // Results follow input order, so the data is last. The sink writes the file and yields empty.
        let mut out = LazyFrame::collect_all_with_engine(
            vec![sink.logical_plan, self.lazy.logical_plan],
            Engine::InMemory,
            OptFlags::default(),
        )
        .into_diagnostic()?;

        let data = out
            .pop()
            .ok_or_else(|| miette!("collect_all returned no frames"))?;

        Ok(Self::from_lazy(data.lazy()))
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
            .ok_or_else(|| miette!("dataframe has no dated rows"))
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
