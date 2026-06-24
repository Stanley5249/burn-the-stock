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

- `stock-client` - HTTP client for the sim trading API and TWSE/TPEX market data
- `trainer` - data pipeline, model training, and weight serialization
- `trader` - inference loop, order execution, and daily scheduling

## Setup

Copy `.env.example` to `.env` and fill in your credentials, then build and run.

```
cargo build --workspace --all-targets --release
```
