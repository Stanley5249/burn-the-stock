"""Integration tests for the downloader module."""

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from burn_the_stock.downloader import (
    OTC_SUFFIX,
    TSE_SUFFIX,
    batch_download,
    read_symbol,
    save_symbol,
)

if TYPE_CHECKING:
    from pathlib import Path


TEST_SYMBOLS = {
    "tse": "2330",  # TSMC
    "otc": "3081",  # Diodes Taiwan
}


def _download_and_save(
    code: str,
    suffix: str,
    period: str,
    tmp_path: Path,
) -> pd.DataFrame:
    ticker = code + suffix
    data = batch_download([ticker], period=period)
    saved = save_symbol(data, ticker, code, tmp_path, None)
    assert saved is not None, f"save_symbol failed for {ticker}"
    path = tmp_path / f"{code}.csv"
    assert path.exists()
    return pd.read_csv(path, index_col="date", parse_dates=True)


def test_tse(tmp_path: Path) -> None:
    """Download a TSE stock and verify the saved CSV has OHLCV data."""
    df = _download_and_save(TEST_SYMBOLS["tse"], TSE_SUFFIX, "1mo", tmp_path)
    assert not df.empty
    assert "close" in df.columns


def test_otc(tmp_path: Path) -> None:
    """Download an OTC stock and verify the saved CSV has OHLCV data."""
    df = _download_and_save(TEST_SYMBOLS["otc"], OTC_SUFFIX, "1mo", tmp_path)
    assert not df.empty
    assert "close" in df.columns


def _fake_batch(ticker: str, dates: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    columns = pd.MultiIndex.from_product(
        [[ticker], ["Open", "High", "Low", "Close", "Volume"]],
    )
    values = [[1.0, 2.0, 0.5, 1.5, 100.0] for _ in dates]
    return pd.DataFrame(values, index=index, columns=columns)


def test_incremental_merge(tmp_path: Path) -> None:
    """A second save appends new dates and keeps the old ones."""
    ticker = "2330.TW"
    first = _fake_batch(ticker, ["2026-06-10", "2026-06-11"])
    saved = save_symbol(first, ticker, "2330", tmp_path, None)
    assert saved is not None
    assert saved.get_column("date").max() == date(2026, 6, 11)

    existing = read_symbol(tmp_path / "2330.csv")
    second = _fake_batch(ticker, ["2026-06-11", "2026-06-12"])
    merged = save_symbol(second, ticker, "2330", tmp_path, existing)
    assert merged is not None
    assert merged.get_column("date").to_list() == [
        date(2026, 6, 10),
        date(2026, 6, 11),
        date(2026, 6, 12),
    ]


def test_incremental_merge_integer_volume_csv(tmp_path: Path) -> None:
    """An existing CSV with integer-formatted volume still merges with new bars."""
    path = tmp_path / "2330.csv"
    path.write_text(
        "date,code,open,high,low,close,volume\n2026-06-10,2330,1.0,2.0,0.5,1.5,100\n",
    )
    existing = read_symbol(path)
    assert existing is not None
    assert existing.schema["volume"] == pl.Float64

    batch = _fake_batch("2330.TW", ["2026-06-11"])
    merged = save_symbol(batch, "2330.TW", "2330", tmp_path, existing)
    assert merged is not None
    assert merged.schema["volume"] == pl.Float64
    assert merged.get_column("date").to_list() == [date(2026, 6, 10), date(2026, 6, 11)]


def test_save_symbol_no_new_keeps_existing(tmp_path: Path) -> None:
    """An empty batch returns the existing frame untouched."""
    ticker = "2330.TW"
    existing = save_symbol(
        _fake_batch(ticker, ["2026-06-10"]),
        ticker,
        "2330",
        tmp_path,
        None,
    )
    empty = pd.DataFrame()
    assert save_symbol(empty, ticker, "2330", tmp_path, existing) is existing


def test_read_symbol_missing(tmp_path: Path) -> None:
    """read_symbol returns None when no CSV exists yet."""
    assert read_symbol(tmp_path / "9999.csv") is None
