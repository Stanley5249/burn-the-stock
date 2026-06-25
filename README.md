# burn-the-stock

An automated Taiwan stock trading bot built in Rust, using a burn neural network to decide
daily buy and sell orders.

## Goal

Starting from NT$100,000,000 in virtual funds, the bot trades Taiwan listed stocks
(TWSE, TPEX, ESB) and aims to maximize total asset value by market close on June 26, 2026.
At least 100 executed trades are required over the trading period.

Profit is measured as total asset value at close on 2026-06-26 minus the 100,000,000 starting capital.

## How it works

The `refresh` binary pulls daily OHLCV history from Yahoo Finance into a parquet, which the
`trainer` binary reads to train a neural network and save the model weights to disk.

The `trader` binary loads those weights and runs as a daemon. Each Taiwan trading day it
wakes around market open (09:00 CST), runs inference on recent price data, and places
buy or sell orders through the sim-server API.

## Crates

- `portfolio` - long-only portfolio simulation under sim stock rules, the backtest engine
- `stock-client` - HTTP clients for the sim trading API, Fugle quotes, Yahoo OHLCV, and TWSE holidays
- `stock-data` - OHLCV parquet store plus the `refresh` binary that fetches history
- `stock-model` - shared model architecture, feature transform, and inference path
- `trainer` - training and backtest CLI
- `trader` - live inference loop, order execution, and daily scheduling

## Setup

Copy `.env.example` to `.env` and fill in your credentials, then build.

```
cargo build --workspace --all-targets --release
```

## Usage

Fetch the latest price history into the parquet.

```
cargo run --release --bin refresh
```

Train a model. The flag defaults already encode the tuned setup, so only `--artifact-dir` is
required. Override any hyperparameter to sweep.

```
cargo run --release --bin trainer -- train --artifact-dir artifacts/run
```

Backtest a run over its held-out window.

```
cargo run --release --bin trainer -- backtest --artifact-dir artifacts/run
```
