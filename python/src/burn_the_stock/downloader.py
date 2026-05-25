"""Download historical daily OHLCV data for Taiwan stocks via yfinance."""

import argparse
import logging
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, TypeAdapter

import burn_the_stock.logging

logger = logging.getLogger(__name__)

SIM_STOCK_LIST_URL = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/stock_list"

TSE_SUFFIX = ".TW"
OTC_SUFFIX = ".TWO"


# --- Pydantic models ---


class StockEntry(BaseModel):
    """A single stock entry from the sim stock API."""

    name: str
    type: Literal["ETF", "TWSE", "OTC", "ESB"]


_stock_list_adapter = TypeAdapter(dict[str, StockEntry])


# --- Symbol fetching ---


def fetch_sim_symbols() -> tuple[list[str], list[str]]:
    """Fetch the sim stock universe and split into TSE and OTC code lists.

    ESB stocks are skipped because yfinance does not carry emerging-board data.

    Returns:
        A tuple of (tse_codes, otc_codes).
    """
    response = httpx.get(SIM_STOCK_LIST_URL, timeout=30)
    response.raise_for_status()
    stock_list = _stock_list_adapter.validate_python(response.json())

    tse_codes: list[str] = []
    otc_codes: list[str] = []
    esb_count = 0

    for code, entry in stock_list.items():
        if entry.type in {"ETF", "TWSE"}:
            tse_codes.append(code)
        elif entry.type == "OTC":
            otc_codes.append(code)
        else:
            esb_count += 1

    if esb_count:
        logger.warning("skipped %s ESB stocks (not on yfinance)", esb_count)

    logger.info("fetched symbols tse=%s otc=%s", len(tse_codes), len(otc_codes))
    return tse_codes, otc_codes


# --- Batch download ---


def batch_download(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download all tickers at once via yfinance.

    Returns:
        A MultiIndex DataFrame grouped by ticker.
    """
    kwargs: dict[str, Any] = dict(
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
        multi_level_index=True,
        **({"period": period} if period else {"start": start, "end": end}),
    )
    return yf.download(tickers, **kwargs)


# --- Saving ---


def save_symbol(
    data: pd.DataFrame,
    ticker: str,
    symbol: str,
    output_dir: Path,
) -> bool:
    """Extract one ticker from a batch DataFrame and save to CSV.

    Returns:
        False (with a warning) if the ticker is absent or has no data, True otherwise.
    """
    try:
        df: pd.DataFrame = data[ticker].dropna(how="all").copy()
    except KeyError:
        logger.warning("skip ticker=%s: not in batch result", ticker)
        return False

    if df.empty:
        logger.warning("skip ticker=%s: no data", ticker)
        return False

    df.index.name = "date"
    df.columns = [col.lower() for col in df.columns]
    df.insert(0, "code", symbol)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    df.to_csv(path)
    logger.info("saved ticker=%s bars=%s path=%s", ticker, len(df), path)
    return True


# --- Orchestration ---


def run(
    symbols: list[str] | None,
    start: str | None,
    end: str | None,
    period: str | None,
    output: Path,
) -> None:
    """Download OHLCV data for the given symbols and write CSVs under output.

    When symbols is None the full sim stock universe is downloaded. Otherwise
    the sim API is queried once to classify each symbol by market.
    """
    if symbols is not None:
        all_tse, all_otc = fetch_sim_symbols()
        tse_set = set(all_tse)
        otc_set = set(all_otc)
        tse_codes = [s for s in symbols if s in tse_set]
        otc_codes = [s for s in symbols if s in otc_set]
        unknown = [s for s in symbols if s not in tse_set and s not in otc_set]
        if unknown:
            logger.warning("unknown or ESB symbols skipped: %s", unknown)
    else:
        tse_codes, otc_codes = fetch_sim_symbols()

    for codes, suffix, market_name in (
        (tse_codes, TSE_SUFFIX, "tse"),
        (otc_codes, OTC_SUFFIX, "otc"),
    ):
        if not codes:
            continue
        tickers = [code + suffix for code in codes]
        logger.info(
            "batch downloading market=%s count=%s",
            market_name,
            len(tickers),
        )
        data = batch_download(tickers, start=start, end=end, period=period)

        output_dir = output / market_name
        for code, ticker in zip(codes, tickers, strict=True):
            save_symbol(data, ticker, code, output_dir)

        logger.info("done market=%s", market_name)


# --- CLI ---


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the downloader script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download historical daily OHLCV data via yfinance",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help=(
            "Stock codes without suffix (omit to download the full sim stock universe)"
        ),
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--period",
        metavar="PERIOD",
        help=(
            "yfinance period string (e.g. 10y, 5y, max);"
            " mutually exclusive with --from/--to"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_date",
        default="2016-01-01",
        metavar="YYYY-MM-DD",
        help="Start date inclusive (default: 2016-01-01); ignored when --period is set",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
        help="End date exclusive (default: today); ignored when --period is set",
    )
    parser.add_argument(
        "--output",
        default="data",
        metavar="DIR",
        help="Output root directory (default: data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    burn_the_stock.logging.setup()

    args = parse_args()
    start = end = None
    if args.period is None:
        start = args.from_date
        end = args.to_date
    run(args.symbols, start, end, args.period, Path(args.output))
