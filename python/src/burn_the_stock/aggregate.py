"""Aggregate per-stock CSVs into a single parquet file with zstd compression."""

import argparse
import logging
from pathlib import Path

import polars as pl

import burn_the_stock.log
from burn_the_stock.schema import SCHEMA

logger = logging.getLogger(__name__)

SORT_KEY = ["market", "code", "date"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def read_market(market_dir: Path, market: str) -> pl.LazyFrame:
    """Lazily scan a market dir's CSVs, tagged with the market column."""
    market_col = pl.lit(market).cast(pl.Categorical).alias("market")
    return pl.scan_csv(market_dir, schema_overrides=SCHEMA).with_columns(market_col)


def save_parquet(frame: pl.LazyFrame, output: Path) -> None:
    """Drop NaN-price rows, sort, and stream to a zstd-compressed parquet."""
    output.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.filter(pl.all_horizontal(pl.col(PRICE_COLUMNS).is_not_nan()))
    clean.sort(SORT_KEY).sink_parquet(output, compression="zstd")

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("done output=%s size_mb=%.1f", output, size_mb)


def run(input_dir: Path, output: Path) -> None:
    """Aggregate per-stock CSVs from input_dir's tse/ and otc/ dirs into parquet."""
    frames = [
        read_market(input_dir / market, market)
        for market in ("tse", "otc")
        if (input_dir / market).is_dir()
    ]
    save_parquet(pl.concat(frames), output)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate stock CSVs into parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, metavar="DIR")
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output parquet file, defaults to INPUT/stocks.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    burn_the_stock.log.setup()
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path / "stocks.parquet"
    run(input_path, output_path)
