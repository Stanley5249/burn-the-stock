"""Download daily OHLCV for Taiwan stocks via yfinance, updating existing CSVs."""

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import httpx
import polars as pl
import yfinance as yf
from pydantic import BaseModel, TypeAdapter

import burn_the_stock.aggregate
import burn_the_stock.log

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

SIM_STOCK_LIST_URL = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/stock_list"

TSE_SUFFIX = ".TW"
OTC_SUFFIX = ".TWO"

SUFFIX_MARKET = {TSE_SUFFIX: "tse", OTC_SUFFIX: "otc"}

COLUMNS = ["date", "code", "open", "high", "low", "close", "volume"]


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
    span = {"period": period} if period else {"start": start, "end": end}
    return yf.download(
        tickers,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
        multi_level_index=True,
        **span,
    )


# --- Saving ---


def to_long(data: pd.DataFrame, ticker: str, symbol: str) -> pl.DataFrame | None:
    """Extract one ticker from a batch DataFrame as a long-form Polars frame.

    Builds the frame from numpy arrays rather than pl.from_pandas, so no pyarrow
    is needed. yfinance dates arrive at second resolution which Polars rejects,
    so they are cast to day resolution to land directly on a Polars Date.

    Returns:
        A frame with COLUMNS, or None when the ticker is absent or empty.
    """
    try:
        df = cast("pd.DataFrame", data[ticker]).dropna(how="all")
    except KeyError:
        logger.warning("skip ticker=%s: not in batch result", ticker)
        return None

    if df.empty:
        logger.warning("skip ticker=%s: no data", ticker)
        return None

    df.index.name = "date"
    df = df.reset_index()
    df.columns = [str(col).lower() for col in df.columns]
    arrays = {
        name: df[name].to_numpy() for name in ("open", "high", "low", "close", "volume")
    }
    arrays["date"] = df["date"].to_numpy().astype("datetime64[D]")
    return (
        pl.DataFrame(arrays).with_columns(pl.lit(symbol).alias("code")).select(COLUMNS)
    )


def read_symbol(path: Path) -> pl.DataFrame | None:
    """Read a symbol's existing CSV, or None when it is absent.

    Returns:
        The stored frame with COLUMNS, or None.
    """
    if not path.exists():
        return None
    return pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides={"code": pl.String},
    ).select(COLUMNS)


def save_symbol(
    data: pd.DataFrame,
    ticker: str,
    symbol: str,
    output_dir: Path,
    existing: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Merge a ticker's new bars into its CSV and return the full frame.

    New dates are appended to the pre-read existing frame and the complete frame
    is returned so the caller can aggregate in memory without re-reading. When
    the batch carries nothing new the existing frame is returned untouched, and
    None means the symbol has no data anywhere.

    Returns:
        The complete per-symbol frame, or None when no data exists.
    """
    new = to_long(data, ticker, symbol)

    if new is None:
        return existing
    if existing is not None:
        new = pl.concat([existing, new]).unique("date", keep="last").sort("date")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.csv"
    new.write_csv(path)
    logger.info("saved ticker=%s bars=%s path=%s", ticker, len(new), path)
    return new


# --- Orchestration ---


def classify_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split requested symbols into TSE and OTC by the sim stock universe.

    Returns:
        A tuple of (tse_codes, otc_codes); unknown or ESB symbols are dropped.
    """
    all_tse, all_otc = fetch_sim_symbols()
    tse_set = set(all_tse)
    otc_set = set(all_otc)
    tse_codes = [code for code in symbols if code in tse_set]
    otc_codes = [code for code in symbols if code in otc_set]
    unknown = [code for code in symbols if code not in tse_set and code not in otc_set]
    if unknown:
        logger.warning("unknown or ESB symbols skipped: %s", unknown)
    return tse_codes, otc_codes


def merge_save(
    data: pd.DataFrame,
    codes: list[str],
    suffix: str,
    output_dir: Path,
    existing: dict[str, pl.DataFrame | None],
) -> list[pl.DataFrame]:
    """Save each code's slice, tagging the merged frames with their market.

    Returns:
        The merged per-symbol frames that had data.
    """
    market = pl.lit(SUFFIX_MARKET[suffix]).cast(pl.Categorical).alias("market")
    frames: list[pl.DataFrame] = []
    for code in codes:
        frame = save_symbol(data, code + suffix, code, output_dir, existing[code])
        if frame is not None:
            frames.append(frame.with_columns(market))
    return frames


def fetch_and_save(
    codes: list[str],
    suffix: str,
    output_dir: Path,
    existing: dict[str, pl.DataFrame | None],
    span: dict[str, str | None],
) -> list[pl.DataFrame]:
    """Download one date span for the codes and merge each into its CSV.

    Returns:
        The merged per-symbol frames that had data.
    """
    tickers = [code + suffix for code in codes]
    data = batch_download(tickers, **span)
    return merge_save(data, codes, suffix, output_dir, existing)


def run(
    symbols: list[str] | None,
    start: str,
    end: str | None,
    period: str | None,
    output: Path,
) -> pl.DataFrame | None:
    """Download OHLCV for the given symbols, creating or updating CSVs.

    New symbols are fetched from start; symbols with an existing CSV are fetched
    only from the day after their latest saved bar. Each existing CSV is read
    once and merged in memory, then the full combined dataset is returned for
    in-memory aggregation. When symbols is None the full sim stock universe is
    used.

    Returns:
        The combined dataset across every processed symbol, or None when nothing
        had data.
    """
    if symbols is not None:
        tse_codes, otc_codes = classify_symbols(symbols)
    else:
        tse_codes, otc_codes = fetch_sim_symbols()

    collected: list[pl.DataFrame] = []

    for codes, suffix in ((tse_codes, TSE_SUFFIX), (otc_codes, OTC_SUFFIX)):
        if not codes:
            continue
        market_name = SUFFIX_MARKET[suffix]
        output_dir = output / market_name
        existing = {code: read_symbol(output_dir / f"{code}.csv") for code in codes}

        if period is not None:
            logger.info("batch downloading all count=%s period=%s", len(codes), period)
            collected.extend(
                fetch_and_save(codes, suffix, output_dir, existing, {"period": period}),
            )
            logger.info("done market=%s", market_name)
            continue

        new_codes = [code for code in codes if existing[code] is None]
        update_codes = [code for code in codes if existing[code] is not None]

        if new_codes:
            logger.info("batch downloading new count=%s from=%s", len(new_codes), start)
            span = {"start": start, "end": end}
            collected.extend(
                fetch_and_save(new_codes, suffix, output_dir, existing, span),
            )

        if update_codes:
            last = min(
                cast("date", frame.get_column("date").max())
                for code in update_codes
                if (frame := existing[code]) is not None
            )
            update_start = (last + timedelta(days=1)).isoformat()
            logger.info(
                "batch downloading update count=%s from=%s",
                len(update_codes),
                update_start,
            )
            span = {"start": update_start, "end": end}
            collected.extend(
                fetch_and_save(update_codes, suffix, output_dir, existing, span),
            )

        logger.info(
            "done market=%s new=%s updated=%s",
            market_name,
            len(new_codes),
            len(update_codes),
        )

    if not collected:
        return None
    return pl.concat(collected).sort(["market", "code", "date"])


# --- CLI ---


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the downloader script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download daily OHLCV via yfinance, updating existing CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="Stock codes without suffix, omit for the full sim stock universe",
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--period",
        metavar="PERIOD",
        help=(
            "yfinance period string (e.g. 10y, 5y, max);"
            " mutually exclusive with --from/--to and ignores existing CSVs"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_date",
        default="2016-01-01",
        metavar="YYYY-MM-DD",
        help="Start date for new symbols; ignored when --period is set",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=datetime.now().astimezone().date().isoformat(),
        metavar="YYYY-MM-DD",
        help="End date exclusive; ignored when --period is set",
    )
    parser.add_argument(
        "--output",
        default="data/yfinance",
        metavar="DIR",
        help="Output root directory holding tse/ and otc/",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip rebuilding the parquet after downloading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    burn_the_stock.log.setup()

    args = parse_args()
    end = None if args.period else args.to_date
    output = Path(args.output)
    dataset = run(args.symbols, args.from_date, end, args.period, output)

    if not args.no_aggregate:
        parquet = output / "stocks.parquet"
        if args.symbols is None and dataset is not None:
            # A full run already holds every symbol in memory, so skip re-reading.
            burn_the_stock.aggregate.save_parquet(dataset, parquet)
        else:
            burn_the_stock.aggregate.run(output, parquet)
