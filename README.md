# burn-the-stock

An automated Taiwan stock trading bot built in Rust, using a burn neural network to decide daily buy and sell orders.

See `report/` for further details.

## How it works

The `refresh` binary pulls daily OHLCV history from Yahoo Finance into a parquet. The `trainer` binary reads that parquet to train a neural network and save the model weights to disk.

The `trader` binary is a CLI you run once during a trading session. It refreshes the OHLCV through the prior session, loads the weights, ranks every ticker, then places the day's buy and sell orders through the sim-server API and exits.

Schedule it externally if you want it to run daily.

## Crates

- `stock-portfolio` - the backtest engine, a long-only portfolio simulation under sim stock rules
- `stock-client` - HTTP clients for the sim trading API, Fugle quotes, Yahoo OHLCV, and TWSE holidays
- `stock-data` - OHLCV parquet store plus the `refresh` binary that fetches history
- `stock-model` - shared model architecture, feature transform, and inference path
- `trainer` - training and backtest CLI
- `trader` - live inference loop, order execution, and daily scheduling

## Setup

Install Rust 1.95.0 or later via [rustup](https://rustup.rs), which provides `cargo` and `rustc`.

Then copy `.env.example` to `.env` and fill in your credentials.

## Usage

Fetch the latest price history into `data/yfinance/stock_history.parquet`.

```
cargo run --release --bin refresh
```

Train a model. Pass `--artifact-dir` to set where checkpoints, config, and weights are written; all other flags default to the tuned setup.

```
cargo run --release --bin trainer -- train --artifact-dir artifacts/run
```

Backtest the latest run over its held-out window. Defaults to `artifacts/latest`, which the train step links automatically.

```
cargo run --release --bin trainer -- backtest
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
