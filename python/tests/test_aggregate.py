"""Tests for the aggregate module."""

from typing import TYPE_CHECKING

import polars as pl
from burn_the_stock.aggregate import read_market, run

if TYPE_CHECKING:
    from pathlib import Path

_CSV_CONTENT = """\
date,code,open,high,low,close,volume
2024-01-02,2330,570.0,575.0,568.0,572.0,12000000.0
2024-01-03,2330,573.0,578.0,571.0,576.0,11000000.0
"""


def test_read_market_empty(tmp_path: Path) -> None:
    """Return an empty DataFrame when the market directory has no CSV files."""
    market_dir = tmp_path / "tse"
    market_dir.mkdir()
    result = read_market(market_dir, "tse")
    assert result.is_empty()


def test_read_market_single(tmp_path: Path) -> None:
    """Return a combined DataFrame with a market column from one CSV file."""
    market_dir = tmp_path / "tse"
    market_dir.mkdir()
    (market_dir / "2330.csv").write_text(_CSV_CONTENT)
    result = read_market(market_dir, "tse")
    assert not result.is_empty()
    assert (result["market"] == "tse").all()


def test_run(tmp_path: Path) -> None:
    """Write a sorted parquet with all expected columns from TSE and OTC data."""
    tse_dir = tmp_path / "tse"
    otc_dir = tmp_path / "otc"
    tse_dir.mkdir()
    otc_dir.mkdir()
    (tse_dir / "2330.csv").write_text(_CSV_CONTENT)
    (otc_dir / "3081.csv").write_text(_CSV_CONTENT.replace("2330", "3081"))
    output = tmp_path / "stocks.parquet"
    run(tmp_path, output)
    assert output.exists()
    df = pl.read_parquet(output)
    expected_cols = ["market", "code", "date", "open", "high", "low", "close", "volume"]
    assert df.columns == expected_cols
    assert df["market"].dtype == pl.Categorical
    assert not df.is_empty()
