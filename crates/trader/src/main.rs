//! Live trading loop: fetch prices, predict an action per ticker, place orders. The
//! data fetch and order placement are mocked, so this runs the real inference path
//! end to end without the network.

use std::path::PathBuf;

use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use chrono::{Duration, NaiveDate};
use clap::Parser;
use miette::{IntoDiagnostic, Result};
use polars::prelude::*;
use stock_client::types::OhlcvRow;
use stock_model::features::{
    CLOSE, CODE, DATE, HIGH, LOW, OPEN, VOLUME, latest_windows, standardized_features,
};
use stock_model::inference::{Prediction, Predictor};

type Backend = Wgpu;

#[derive(Parser, Debug)]
#[command(about = "Predict today's actions and place the implied orders")]
struct Args {
    /// Directory holding a training run's `config.json` and `model.mpk`.
    #[arg(long, default_value = "artifacts/latest")]
    artifact_dir: PathBuf,

    /// Skip orders whose expected edge `clamp(P(Buy)*tp - P(Sell)*sl, 0)` is weaker
    /// than this, so the trader stays flat rather than churning fees.
    #[arg(long, default_value_t = 0.0)]
    min_edge: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = WgpuDevice::default();
    let predictor = Predictor::<Backend>::load(&args.artifact_dir, device)?;

    // Mock feed: synthesize rows of the same `OhlcvRow` shape the HTTP client
    // returns, so only this call changes once the real fetch lands.
    let rows = mock_fetch(predictor.steps());

    // Real pipeline from here: build a frame, apply the shared feature transform,
    // take each ticker's most recent window.
    let market = market_frame(&rows).into_diagnostic()?;
    let windows =
        latest_windows(standardized_features(market), predictor.steps()).into_diagnostic()?;

    let predictions = predictor.predict(&windows);

    place_orders(&predictions, args.min_edge);

    Ok(())
}

/// Lazy frame with the raw OHLCV schema the feature transform reads. Volume is each
/// row's traded capacity.
fn market_frame(rows: &[OhlcvRow]) -> PolarsResult<LazyFrame> {
    let codes: Vec<&str> = rows.iter().map(|row| row.stock_code_id.as_str()).collect();
    let dates: Vec<NaiveDate> = rows.iter().map(|row| row.date).collect();
    let opens: Vec<Option<f64>> = rows.iter().map(|row| row.open).collect();
    let highs: Vec<Option<f64>> = rows.iter().map(|row| row.high).collect();
    let lows: Vec<Option<f64>> = rows.iter().map(|row| row.low).collect();
    let closes: Vec<Option<f64>> = rows.iter().map(|row| row.close).collect();
    let volumes: Vec<u64> = rows.iter().map(|row| row.capacity).collect();

    let frame = df!(
        CODE => codes,
        DATE => dates,
        OPEN => opens,
        HIGH => highs,
        LOW => lows,
        CLOSE => closes,
        VOLUME => volumes,
    )?;

    Ok(frame.lazy())
}

/// Stand-in for the live feed. Each ticker drifts at its own pace so the per-date
/// cross-section has the variance the z-score needs. Synthetic, so actions are only
/// meaningful once swapped for the real HTTP fetch.
fn mock_fetch(steps: usize) -> Vec<OhlcvRow> {
    // One extra day so the first log-return row is not dropped.
    let days = steps + 1;
    let start = NaiveDate::from_ymd_opt(2026, 1, 5).unwrap();

    // (code, starting close); start seeds each ticker's drift rate.
    let universe = [
        ("2330", 1000.0_f64),
        ("2317", 100.0),
        ("2454", 1300.0),
        ("2308", 480.0),
        ("2412", 120.0),
        ("3008", 2600.0),
    ];

    let mut rows = Vec::with_capacity(universe.len() * days);
    for (code, start_close) in universe {
        let mut close = start_close;
        for offset in 0..days {
            // Drift proportional to price level, so the cross-section is not
            // degenerate.
            close += start_close * 0.003;
            let date = start + Duration::days(i64::try_from(offset).expect("small horizon"));
            let row = OhlcvRow::new(
                date,
                (*code).to_owned(),
                Some(close - 0.5),
                Some(close + 1.0),
                Some(close - 1.0),
                Some(close),
                None,
                1_000_000,
                0,
                0,
            )
            .expect("mock rows are valid");
            rows.push(row);
        }
    }

    rows
}

/// Place the orders the predictions imply, strongest signal first. Mocked: prints
/// the buys a real trader would size against cash and submit to `sim_stock`.
fn place_orders(predictions: &[Prediction], min_edge: f32) {
    let Some(as_of) = predictions.iter().map(|prediction| prediction.date).max() else {
        println!("No tickers had enough history to fill the model's window.");
        return;
    };

    let mut buys: Vec<&Prediction> = predictions
        .iter()
        .filter(|prediction| prediction.expected_edge > min_edge)
        .collect();
    buys.sort_by(|left, right| right.expected_edge.total_cmp(&left.expected_edge));

    println!(
        "As of {as_of}: {} tickers, {} actionable buys (edge > {min_edge:.3}).",
        predictions.len(),
        buys.len(),
    );

    for buy in buys {
        let [_, _, probability_buy] = buy.probabilities;
        // Real placement: SimStockClient::buy(&buy.ticker, shares, price).await
        println!(
            "  [mock] BUY {:<8} P(Buy) {probability_buy:.3}  edge {:.3}",
            buy.ticker, buy.expected_edge,
        );
    }
}
