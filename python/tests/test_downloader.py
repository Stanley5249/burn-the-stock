"""Integration tests for the downloader module."""

from typing import TYPE_CHECKING

import pandas as pd
from burn_the_stock.downloader import (
    OTC_SUFFIX,
    TSE_SUFFIX,
    batch_download,
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
    assert save_symbol(data, ticker, code, tmp_path), f"save_symbol failed for {ticker}"
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
