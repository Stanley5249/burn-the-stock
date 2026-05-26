"""Aggregate per-stock CSVs into a single parquet file with zstd compression."""

import argparse
import logging
from pathlib import Path

import polars as pl

import burn_the_stock.logging

logger = logging.getLogger(__name__)


def read_market(market_dir: Path, market: str) -> pl.LazyFrame:
    """Read all CSV files from a market directory and combine them.

    Args:
        market_dir: Directory containing per-stock CSV files.
        market: Market label to attach to each row (e.g. "tse" or "otc").

    Returns:
        Combined DataFrame with a market column, or an empty DataFrame when
        no CSV files are found.
    """
    market_col = pl.lit(market).cast(pl.Categorical).alias("market")

    df = pl.scan_csv(
        market_dir,
        try_parse_dates=True,
        schema_overrides=({"code": pl.String}),
    )

    return df.with_columns(market_col)


def run(input_dir: Path, output: Path) -> None:
    """Aggregate per-stock CSVs from input_dir into a single parquet file.

    Reads TSE and OTC subdirectories, combines them, sorts by market, code,
    and date, then writes the result as zstd-compressed parquet.

    Args:
        input_dir: Root directory with tse/ and otc/ subdirectories.
        output: Destination parquet file path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    tse = read_market(input_dir / "tse", "tse")
    otc = read_market(input_dir / "otc", "otc")

    df = pl.concat([tse, otc]).sort(["market", "code", "date"]).collect()

    df.write_parquet(output, compression="zstd")

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("done output=%s size_mb=%.1f", output, size_mb)

    print(df)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the aggregator script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Aggregate stock CSVs into parquet")
    parser.add_argument("--input", required=True, metavar="DIR")
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output parquet file (default: INPUT/stocks.parquet)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    burn_the_stock.logging.setup()
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path / "stocks.parquet"
    run(input_path, output_path)
