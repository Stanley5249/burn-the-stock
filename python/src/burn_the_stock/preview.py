"""Quick lazy preview of a tabular data file (parquet, CSV, or TSV)."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.exceptions import PolarsError

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars import LazyFrame


def lazy_preview(file_path: str, rows: int = 10) -> None:
    """Print the first rows of a parquet, CSV, or TSV file using a lazy scan.

    Raises:
        PolarsError: If the file cannot be scanned or collected.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    scanners: dict[str, Callable[[str], LazyFrame]] = {
        ".parquet": pl.scan_parquet,
        ".csv": pl.scan_csv,
        ".tsv": pl.scan_csv,
    }

    if ext not in scanners:
        print(f"Unsupported format: {ext}")
        sys.exit(1)

    try:
        frame = scanners[ext](file_path)
        print(frame.head(rows).collect())
    except PolarsError as error:
        print(f"Error reading file: {error}")
        raise


def main() -> None:
    """Parse command-line arguments and run the preview."""
    parser = argparse.ArgumentParser(
        description="Preview data files lazily",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path_to_file", help="Path to the file to preview")
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to preview",
    )

    args = parser.parse_args()
    lazy_preview(args.path_to_file, rows=args.rows)


if __name__ == "__main__":
    main()
