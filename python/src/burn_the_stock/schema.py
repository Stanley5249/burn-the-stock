"""Shared OHLCV schema for stored CSVs and the aggregated parquet."""

import polars as pl

SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "code": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
}
