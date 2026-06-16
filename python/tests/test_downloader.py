"""Integration tests for the downloader module."""

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from burn_the_stock import downloader
from burn_the_stock.downloader import (
    OTC_SUFFIX,
    TSE_SUFFIX,
    batch_download,
    read_symbol,
    save_symbol,
)
from burn_the_stock.schema import SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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


def test_save_symbol_skips_write_without_new_dates(tmp_path: Path) -> None:
    """A fetch adding no date beyond the last stored bar returns existing untouched."""
    first = save_symbol(
        _fake_batch("2330.TW", ["2026-06-10", "2026-06-11"]),
        "2330.TW",
        "2330",
        tmp_path,
        None,
    )
    assert first is not None

    stale = _fake_batch("2330.TW", ["2026-06-11"])
    assert save_symbol(stale, "2330.TW", "2330", tmp_path, first) is first


def test_save_symbol_drops_nan_price_bar(tmp_path: Path) -> None:
    """A fetched bar with NaN prices is dropped before the CSV is written."""
    ticker = "2330.TW"
    batch = _fake_batch(ticker, ["2026-06-10", "2026-06-11"])
    batch.loc[pd.Timestamp("2026-06-11"), (ticker, "Close")] = float("nan")

    saved = save_symbol(batch, ticker, "2330", tmp_path, None)
    assert saved is not None
    assert saved.get_column("date").to_list() == [date(2026, 6, 10)]


def test_read_symbol_keeps_raw_nan(tmp_path: Path) -> None:
    """read_symbol returns stored rows verbatim, including a NaN-price bar."""
    path = tmp_path / "2330.csv"
    path.write_text(
        "date,code,open,high,low,close,volume\n"
        "2026-06-12,2330,1.0,2.0,0.5,1.5,100.0\n"
        "2026-06-15,2330,NaN,NaN,NaN,NaN,371330.0\n",
    )
    existing = read_symbol(path)
    assert existing is not None
    assert existing.get_column("date").to_list() == [
        date(2026, 6, 12),
        date(2026, 6, 15),
    ]


def test_save_symbol_fills_hole(tmp_path: Path) -> None:
    """A refetch with an earlier max still merges, filling an interior gap."""
    existing = save_symbol(
        _fake_batch("2330.TW", ["2026-06-12", "2026-06-16"]),
        "2330.TW",
        "2330",
        tmp_path,
        None,
    )
    assert existing is not None
    assert date(2026, 6, 15) not in existing.get_column("date").to_list()

    refetch = _fake_batch("2330.TW", ["2026-06-15"])
    filled = save_symbol(refetch, "2330.TW", "2330", tmp_path, existing)
    assert filled is not None
    assert filled.get_column("date").to_list() == [
        date(2026, 6, 12),
        date(2026, 6, 15),
        date(2026, 6, 16),
    ]


def test_save_symbol_skips_when_unchanged(tmp_path: Path) -> None:
    """A refetch of identical bars rewrites nothing and returns the existing frame."""
    existing = save_symbol(
        _fake_batch("2330.TW", ["2026-06-10", "2026-06-11"]),
        "2330.TW",
        "2330",
        tmp_path,
        None,
    )
    assert existing is not None
    same = _fake_batch("2330.TW", ["2026-06-10", "2026-06-11"])
    assert save_symbol(same, "2330.TW", "2330", tmp_path, existing) is existing


def _one_bar(last: date) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [last],
            "code": ["2330"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
        },
        schema=SCHEMA,
    )


def test_fetch_updates_skips_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symbol already current through end is collected without downloading."""
    calls: list[object] = []
    monkeypatch.setattr(downloader, "fetch_and_save", lambda *a: calls.append(a) or [])

    existing: dict[str, pl.DataFrame | None] = {"2330": _one_bar(date(2026, 6, 15))}
    frames = downloader.fetch_updates(
        ["2330"], TSE_SUFFIX, tmp_path, existing, "2026-06-16",
    )

    assert calls == []
    assert len(frames) == 1
    assert "market" in frames[0].columns


def test_fetch_updates_fetches_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale symbol is fetched from the day after its last bar."""
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(downloader, "fetch_and_save", lambda *a: calls.append(a) or [])

    existing: dict[str, pl.DataFrame | None] = {"2330": _one_bar(date(2024, 1, 2))}
    downloader.fetch_updates(["2330"], TSE_SUFFIX, tmp_path, existing, "2026-06-16")

    assert len(calls) == 1
    group, _suffix, _dir, _existing, span = calls[0]
    assert group == ["2330"]
    assert span == {"start": "2024-01-03", "end": "2026-06-16"}


def test_find_stale_flags_old() -> None:
    """A symbol far behind the freshest is stale; a recent or missing one is not."""
    existing: dict[str, pl.DataFrame | None] = {
        "2330": _one_bar(date(2026, 6, 15)),
        "0058": _one_bar(date(2019, 1, 1)),
        "9999": None,
    }
    assert downloader.find_stale(existing) == {"0058"}


def test_dead_roundtrip(tmp_path: Path) -> None:
    """save_dead then load_dead returns the same set; a missing file reads empty."""
    assert downloader.load_dead(tmp_path) == set()
    downloader.save_dead(tmp_path, {"0080", "3454"})
    assert downloader.load_dead(tmp_path) == {"0080", "3454"}
