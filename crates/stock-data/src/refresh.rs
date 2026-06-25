//! Refresh the consolidated history parquet from Yahoo Finance: fetch the sim stock
//! universe, pull bars since the last stored date, append, dedup, and sink. The pure-Rust
//! replacement for the Python downloader.

use std::path::Path;

use chrono::{NaiveDate, TimeDelta};
use futures::{StreamExt, future, stream};
use miette::{Context, IntoDiagnostic, Result};
use polars::prelude::*;
use stock_client::types::MarketType;
use stock_client::yahoo::{ChartBars, YahooClient};

use crate::read::History;
use crate::schema::{ADJCLOSE, CLOSE, CODE, DATE, HIGH, LOW, MARKET, OPEN, VOLUME};

/// In-flight chart requests; Yahoo is one symbol per request.
const CONCURRENCY: usize = 16;

/// Abandon the refresh once this many chart requests fail, so a Yahoo outage stops fast instead
/// of grinding through the whole universe.
const MAX_ERRORS: usize = 16;

/// Assemble unadjusted bars into a frame, then fold the split/dividend adjustment in column-wise:
/// scale OHLC by `adjclose / close`, keep volume raw, drop rows with no usable close, and rename
/// adjclose to close.
fn adjusted_frame(bars: ChartBars) -> Result<LazyFrame> {
    let raw = df!(
        DATE => bars.dates,
        OPEN => bars.open,
        HIGH => bars.high,
        LOW => bars.low,
        CLOSE => bars.close,
        VOLUME => bars.volume,
        ADJCLOSE => bars.adjclose,
    )
    .into_diagnostic()
    .wrap_err("build dataframe from chart bars")?;

    let factor = col(ADJCLOSE) / col(CLOSE);

    Ok(raw.lazy().filter(factor.clone().is_finite()).select([
        col(DATE),
        (col(OPEN) * factor.clone()).alias(OPEN),
        (col(HIGH) * factor.clone()).alias(HIGH),
        (col(LOW) * factor).alias(LOW),
        col(ADJCLOSE).alias(CLOSE),
        col(VOLUME),
    ]))
}

/// Project a frame onto the canonical column order so strict vertical concat lines up with the
/// scanned history frame.
fn canonical(lazy: LazyFrame) -> LazyFrame {
    lazy.select([
        col(MARKET),
        col(CODE),
        col(DATE),
        col(OPEN),
        col(HIGH),
        col(LOW),
        col(CLOSE),
        col(VOLUME),
    ])
}

/// Concat the per-symbol and prior frames, drop rows with any null price, keep the newest
/// row per `(code, date)`, and sort. Frames must arrive oldest-first so `keep=Last` wins.
fn consolidate(frames: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
    Ok(concat(frames, UnionArgs::default())?
        .unique(Some(cols([MARKET, CODE, DATE])), UniqueKeepStrategy::Last)
        .sort([MARKET, CODE, DATE], SortMultipleOptions::new()))
}

/// Refresh `output` in place: fetch every universe symbol from the bar after its last stored
/// date (or `floor` on first run) through today, then append and rewrite the parquet.
///
/// # Errors
/// Network, parse, or parquet failure.
pub async fn refresh(
    entries: impl IntoIterator<Item = (String, MarketType)>,
    output: &Path,
    floor: NaiveDate,
    end: NaiveDate,
) -> Result<History> {
    let exists = output.exists();

    let history = History::scan(output)?;

    let start = if exists {
        history.last_date()? + TimeDelta::days(1)
    } else {
        floor
    };

    if start > end {
        tracing::info!(%end, "history already current");
        return Ok(history);
    }

    let client = YahooClient::new().await?;
    let client = &client;

    let entries = entries.into_iter().filter_map(|(code, market_type)| {
        let (suffix, market) = match market_type {
            MarketType::Twse | MarketType::Etf => ("TW", "tse"),
            MarketType::Otc => ("TWO", "otc"),
            MarketType::Esb => return None,
        };

        let symbol = format!("{code}.{suffix}");

        Some((market, code, symbol))
    });

    let bars: Vec<_> = stream::iter(entries)
        .map(async |(market, code, symbol)| {
            client
                .chart(&symbol, start, end)
                .await
                .ok()
                .map(|bars| (market, code, bars))
        })
        .buffer_unordered(CONCURRENCY)
        .scan(0, |errors, fetched| {
            *errors += usize::from(fetched.is_none());
            // Halt the stream once Yahoo has failed too many times rather than fetch every symbol.
            future::ready((*errors <= MAX_ERRORS).then_some(fetched))
        })
        .filter_map(future::ready)
        .collect()
        .await;

    // No bars means Yahoo gave us nothing, so leave the existing file untouched.
    if bars.is_empty() {
        tracing::warn!("yahoo returned no bars, leaving history unchanged");
        return Ok(history);
    }

    let mut frames: Vec<LazyFrame> = Vec::with_capacity(bars.len() + 1);

    if exists {
        frames.push(history.lazy());
    }

    for (market, code, bars) in bars {
        let frame =
            adjusted_frame(bars)?.with_columns([lit(market).alias(MARKET), lit(code).alias(CODE)]);
        frames.push(canonical(frame));
    }

    let merged = consolidate(frames).into_diagnostic()?;

    History::from_lazy(merged).sink(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjusted_frame_scales_ohlc_keeps_volume_drops_null() {
        let bars = ChartBars {
            dates: vec![
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
            ],
            open: vec![Some(100.0), None],
            high: vec![Some(110.0), Some(50.0)],
            low: vec![Some(90.0), Some(40.0)],
            close: vec![Some(100.0), None],
            volume: vec![Some(1000.0), Some(500.0)],
            adjclose: vec![Some(90.0), None],
        };

        let out = adjusted_frame(bars).unwrap().collect().unwrap();

        // Null-close row is dropped, adjclose column is gone.
        assert_eq!(out.height(), 1);
        assert_eq!(
            out.get_column_names(),
            [&DATE, &OPEN, &HIGH, &LOW, &CLOSE, &VOLUME]
        );

        // factor = 90/100 = 0.9 applied to OHLC, volume untouched.
        let close = out.column(&CLOSE).unwrap().f32().unwrap().get(0).unwrap();
        let open = out.column(&OPEN).unwrap().f32().unwrap().get(0).unwrap();
        let high = out.column(&HIGH).unwrap().f32().unwrap().get(0).unwrap();
        let volume = out.column(&VOLUME).unwrap().f32().unwrap().get(0).unwrap();
        assert!((close - 90.0).abs() < 1e-4);
        assert!((open - 90.0).abs() < 1e-4);
        assert!((high - 99.0).abs() < 1e-4);
        assert!((volume - 1000.0).abs() < 1e-4);
    }

    fn bar(code: &str, day: u32, close: f64) -> LazyFrame {
        df!(
            DATE => &[NaiveDate::from_ymd_opt(2024, 1, day).unwrap()],
            CODE => &[code],
            OPEN => &[close],
            HIGH => &[close],
            LOW => &[close],
            CLOSE => &[close],
            VOLUME => &[1.0],
            MARKET => &["tse"],
        )
        .unwrap()
        .lazy()
    }

    #[test]
    fn consolidate_keeps_newest_per_code_date() {
        // Prior bar for (2330, day 1) close 100; new frames re-fetch it at 200 and add day 2.
        let prior = bar("2330", 1, 100.0);
        let fresh = concat(
            [bar("2330", 1, 200.0), bar("2330", 2, 300.0)],
            UnionArgs::default(),
        )
        .unwrap();

        let out = consolidate(vec![prior, fresh]).unwrap().collect().unwrap();

        assert_eq!(out.height(), 2);
        let close = out.column(&CLOSE).unwrap().f64().unwrap();
        // Sorted by date: day 1 takes the re-fetched 200, day 2 is 300.
        assert_eq!(close.get(0), Some(200.0));
        assert_eq!(close.get(1), Some(300.0));
    }
}
