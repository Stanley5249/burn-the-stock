//! Refresh the consolidated history parquet from Yahoo Finance: fetch the sim stock
//! universe, pull bars since the last stored date, append, dedup, and sink. The pure-Rust
//! replacement for the Python downloader.

use std::collections::HashMap;
use std::path::Path;

use chrono::{Duration, Local, NaiveDate};
use miette::{IntoDiagnostic, Result};
use polars::prelude::*;
use stock_client::sim_stock::SimStockClient;
use stock_client::types::{MarketType, StockListEntry};
use stock_client::yahoo::{ChartBars, YahooClient};

use crate::read::History;
use crate::schema::{CLOSE, CODE, DATE, HIGH, LOW, MARKET, OPEN, VOLUME};

/// A universe symbol mapped to its Yahoo ticker and our `tse`/`otc` market label.
struct Symbol {
    code: String,
    yahoo: String,
    market: &'static str,
}

/// Map the sim stock universe to Yahoo symbols, dropping ESB names Yahoo does not carry.
fn classify(entries: Vec<StockListEntry>) -> Vec<Symbol> {
    entries
        .into_iter()
        .filter_map(|entry| {
            let (suffix, market) = match entry.market_type {
                MarketType::Twse | MarketType::Etf => (".TW", "tse"),
                MarketType::Otc => (".TWO", "otc"),
                MarketType::Esb => return None,
            };
            Some(Symbol {
                yahoo: format!("{}{suffix}", entry.code),
                code: entry.code,
                market,
            })
        })
        .collect()
}

/// Assemble unadjusted bars into a frame, then fold the split/dividend adjustment in column-wise:
/// scale OHLC by `adjclose / close`, keep volume raw, drop rows with no usable close, and rename
/// adjclose to close.
fn adjusted_frame(bars: ChartBars) -> PolarsResult<LazyFrame> {
    let raw = df!(
        DATE => bars.dates,
        OPEN => bars.open,
        HIGH => bars.high,
        LOW => bars.low,
        CLOSE => bars.close,
        VOLUME => bars.volume,
        "adjclose" => bars.adjclose,
    )?;

    let usable = col(CLOSE)
        .is_not_null()
        .and(col(CLOSE).is_not_nan())
        .and(col("adjclose").is_not_null())
        .and(col("adjclose").is_not_nan());
    let factor = col("adjclose") / col(CLOSE);

    Ok(raw.lazy().filter(usable).select([
        col(DATE),
        (col(OPEN) * factor.clone()).alias(OPEN),
        (col(HIGH) * factor.clone()).alias(HIGH),
        (col(LOW) * factor).alias(LOW),
        // adjusted close is just adjclose (close * adjclose / close).
        col("adjclose").alias(CLOSE),
        col(VOLUME),
    ]))
}

/// Project a frame onto the canonical column order so strict vertical concat lines up.
fn canonical(lazy: LazyFrame) -> LazyFrame {
    lazy.select([
        col(DATE),
        col(CODE),
        col(OPEN),
        col(HIGH),
        col(LOW),
        col(CLOSE),
        col(VOLUME),
        col(MARKET),
    ])
}

/// Concat the per-symbol and prior frames, drop rows with any null price, keep the newest
/// row per `(code, date)`, and sort. Frames must arrive oldest-first so `keep=Last` wins.
fn consolidate(frames: Vec<LazyFrame>) -> PolarsResult<LazyFrame> {
    let not_null = col(OPEN)
        .is_not_null()
        .and(col(HIGH).is_not_null())
        .and(col(LOW).is_not_null())
        .and(col(CLOSE).is_not_null());

    let on_code_date = Selector::ByName {
        names: vec![CODE, DATE].into(),
        strict: true,
    };

    Ok(concat(frames, UnionArgs::default())?
        .filter(not_null)
        .unique(Some(on_code_date), UniqueKeepStrategy::Last)
        .sort([MARKET, CODE, DATE], SortMultipleOptions::new()))
}

/// Refresh `output` in place: fetch every universe symbol from the bar after its last stored
/// date (or `floor` on first run) through today, then append and rewrite the parquet.
///
/// # Errors
/// Network, parse, or parquet failure.
pub async fn refresh(sim_client: &SimStockClient, output: &Path, floor: NaiveDate) -> Result<()> {
    let symbols = classify(sim_client.stock_list().await?);

    let exists = output.exists();
    let start = if exists {
        History::scan(output)?.last_date()? + Duration::days(1)
    } else {
        floor
    };
    let end = Local::now().date_naive();
    if start > end {
        tracing::info!(%end, "history already current");
        return Ok(());
    }

    let yahoo_symbols: Vec<String> = symbols.iter().map(|symbol| symbol.yahoo.clone()).collect();
    tracing::info!(symbols = yahoo_symbols.len(), %start, %end, "fetching bars");

    let client = YahooClient::new().await?;
    let fetched = client.download(&yahoo_symbols, start, end).await;
    if fetched.is_empty() {
        tracing::warn!("no bars fetched; leaving history unchanged");
        return Ok(());
    }

    let by_yahoo: HashMap<&str, &Symbol> = symbols
        .iter()
        .map(|symbol| (symbol.yahoo.as_str(), symbol))
        .collect();

    // Prior bars first so consolidate's keep=Last lets a re-fetched day overwrite them.
    let mut frames: Vec<LazyFrame> = Vec::new();
    if exists {
        frames.push(canonical(History::scan(output)?.lazy()));
    }
    for (yahoo, bars) in fetched {
        let Some(symbol) = by_yahoo.get(yahoo.as_str()) else {
            continue;
        };
        let frame = adjusted_frame(bars).into_diagnostic()?;
        frames.push(canonical(frame.with_columns([
            lit(symbol.code.as_str()).alias(CODE),
            lit(symbol.market).alias(MARKET),
        ])));
    }

    let merged = consolidate(frames).into_diagnostic()?;
    History::from_lazy(merged).sink(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_maps_markets_and_drops_esb() {
        let entries = ["TWSE", "ETF", "OTC", "ESB"]
            .into_iter()
            .zip(["2330", "0050", "6488", "1234"])
            .map(|(kind, code)| StockListEntry {
                code: code.to_string(),
                market_type: match kind {
                    "TWSE" => MarketType::Twse,
                    "ETF" => MarketType::Etf,
                    "OTC" => MarketType::Otc,
                    _ => MarketType::Esb,
                },
            })
            .collect();

        let symbols = classify(entries);
        let mapped: Vec<(&str, &str, &str)> = symbols
            .iter()
            .map(|s| (s.code.as_str(), s.yahoo.as_str(), s.market))
            .collect();

        // ESB dropped; TWSE and ETF both -> .TW/tse; OTC -> .TWO/otc.
        assert_eq!(
            mapped,
            [
                ("2330", "2330.TW", "tse"),
                ("0050", "0050.TW", "tse"),
                ("6488", "6488.TWO", "otc"),
            ]
        );
    }

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
        let close = out.column(&CLOSE).unwrap().f64().unwrap().get(0).unwrap();
        let open = out.column(&OPEN).unwrap().f64().unwrap().get(0).unwrap();
        let high = out.column(&HIGH).unwrap().f64().unwrap().get(0).unwrap();
        let volume = out.column(&VOLUME).unwrap().f64().unwrap().get(0).unwrap();
        assert!((close - 90.0).abs() < 1e-9);
        assert!((open - 90.0).abs() < 1e-9);
        assert!((high - 99.0).abs() < 1e-9);
        assert!((volume - 1000.0).abs() < 1e-9);
    }

    fn bar(code: &str, day: u32, close: f64) -> LazyFrame {
        canonical(
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
            .lazy(),
        )
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
