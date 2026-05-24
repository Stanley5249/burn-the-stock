"""Aggregate per-stock CSVs into a single parquet file with zstd compression."""

import argparse
import logging
from pathlib import Path

import polars as pl

import burn_the_stock

logger = logging.getLogger(__name__)


def read_market(market_dir: Path, market: str) -> pl.DataFrame:
    """Read all CSV files from a market directory and combine them.

    Args:
        market_dir: Directory containing per-stock CSV files.
        market: Market label to attach to each row (e.g. "tse" or "otc").

    Returns:
        Combined DataFrame with a market column, or an empty DataFrame when
        no CSV files are found.
    """
    csv_files = sorted(market_dir.glob("*.csv"))
    if not csv_files:
        return pl.DataFrame()

    schema_overrides = {"code": pl.String, "volume": pl.Float64}
    frames = [
        pl.read_csv(path, try_parse_dates=True, schema_overrides=schema_overrides)
        for path in csv_files
    ]

    combined = pl.concat(frames)
    combined = combined.with_columns(pl.lit(market).alias("market"))
    logger.info("market=%s files=%s rows=%s", market, len(csv_files), len(combined))
    return combined


def run(input_dir: Path, output: Path) -> None:
    """Aggregate per-stock CSVs from input_dir into a single parquet file.

    Reads TSE and OTC subdirectories, combines them, sorts by market, code,
    and date, then writes the result as zstd-compressed parquet.

    Args:
        input_dir: Root directory with tse/ and otc/ subdirectories.
        output: Destination parquet file path.
    """
    tse = read_market(input_dir / "tse", "tse")
    otc = read_market(input_dir / "otc", "otc")

    frames = [df for df in (tse, otc) if not df.is_empty()]
    if not frames:
        logger.error("no data found")
        return

    combined = pl.concat(frames)
    combined = combined.select(
        [
            "market",
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )
    combined = combined.sort(["market", "code", "date"])

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output, compression="zstd", compression_level=3)
    size_mb = output.stat().st_size / 1_048_576
    logger.info("done rows=%s output=%s size_mb=%.1f", len(combined), output, size_mb)


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
