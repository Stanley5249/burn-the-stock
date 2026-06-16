"""Download daily OHLCV for Taiwan stocks via yfinance, updating existing CSVs."""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import httpx
import polars as pl
import yfinance as yf
from pydantic import BaseModel, TypeAdapter

import burn_the_stock.aggregate
import burn_the_stock.log
from burn_the_stock.schema import SCHEMA

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

logger = logging.getLogger(__name__)

SIM_STOCK_LIST_URL = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/stock_list"

TSE_SUFFIX = ".TW"
OTC_SUFFIX = ".TWO"

SUFFIX_MARKET = {TSE_SUFFIX: "tse", OTC_SUFFIX: "otc"}

DEAD_FILE = "dead.json"
DEAD_STALE_DAYS = 90

PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Window:
    """A download date range: a yfinance period, or a start/end span."""

    start: str
    end: str | None
    period: str | None


class StockEntry(BaseModel):
    """A single stock entry from the sim stock API."""

    name: str
    type: Literal["ETF", "TWSE", "OTC", "ESB"]


_stock_list_adapter = TypeAdapter(dict[str, StockEntry])


def fetch_sim_symbols() -> tuple[list[str], list[str]]:
    """Fetch the sim stock universe split into TSE and OTC code lists."""
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


def batch_download(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download all tickers at once via yfinance, grouped by ticker."""
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


def to_long(data: pd.DataFrame, ticker: str, symbol: str) -> pl.DataFrame | None:
    """Extract one ticker from a batch frame as long-form, or None when empty."""
    try:
        df = cast("pd.DataFrame", data[ticker])
    except KeyError:
        logger.warning("skip ticker=%s: not in batch result", ticker)
        return None

    # numpy avoids the pyarrow pl.from_pandas needs for tz-aware yfinance dates.
    frame = pl.DataFrame(
        {
            "date": df.index.to_numpy().astype("datetime64[D]"),
            "code": symbol,
            "open": df["Open"].to_numpy(),
            "high": df["High"].to_numpy(),
            "low": df["Low"].to_numpy(),
            "close": df["Close"].to_numpy(),
            "volume": df["Volume"].to_numpy(),
        },
        schema=SCHEMA,
    ).filter(pl.all_horizontal(pl.col(PRICE_COLUMNS).is_not_nan()))

    if frame.is_empty():
        logger.warning("skip ticker=%s: no data", ticker)
        return None
    return frame


def read_symbol(path: Path) -> pl.DataFrame | None:
    """Read a symbol's existing CSV, or None when it is absent."""
    if not path.exists():
        return None
    return pl.read_csv(path, schema_overrides=SCHEMA)


def save_symbol(
    data: pd.DataFrame,
    ticker: str,
    symbol: str,
    output_dir: Path,
    existing: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Merge a ticker's new bars into its CSV, rewriting only when a row changes."""
    new = to_long(data, ticker, symbol)
    if new is None:
        return existing
    if existing is not None:
        merged = pl.concat([existing, new]).unique("date", keep="last").sort("date")
        if merged.equals(existing):
            return existing
        new = merged

    output_dir.mkdir(parents=True, exist_ok=True)
    new.write_csv(output_dir / f"{symbol}.csv")
    return new


def classify_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split requested symbols into TSE and OTC, dropping unknown or ESB codes."""
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
    """Save each code's slice, tagging the merged frames with their market."""
    market = pl.lit(SUFFIX_MARKET[suffix]).cast(pl.Categorical).alias("market")
    frames: list[pl.DataFrame] = []
    saved = 0
    for code in codes:
        before = existing[code]
        frame = save_symbol(data, code + suffix, code, output_dir, before)
        if frame is None:
            continue
        frames.append(frame.with_columns(market))
        if frame is not before:
            saved += 1
    if saved:
        logger.info("saved count=%s market=%s", saved, SUFFIX_MARKET[suffix])
    return frames


def fetch_and_save(
    codes: list[str],
    suffix: str,
    output_dir: Path,
    existing: dict[str, pl.DataFrame | None],
    span: Mapping[str, str | None],
) -> list[pl.DataFrame]:
    """Download one date span for the codes and merge each into its CSV."""
    tickers = [code + suffix for code in codes]
    data = batch_download(tickers, **span)
    return merge_save(data, codes, suffix, output_dir, existing)


def find_stale(existing: dict[str, pl.DataFrame | None]) -> set[str]:
    """Codes whose last saved bar is far behind the freshest, treated as delisted."""
    last_dates = {
        code: cast("date", frame.get_column("date").max())
        for code, frame in existing.items()
        if frame is not None
    }
    if not last_dates:
        return set()
    cutoff = max(last_dates.values()) - timedelta(days=DEAD_STALE_DAYS)
    return {code for code, last in last_dates.items() if last < cutoff}


def fetch_updates(
    update_codes: list[str],
    suffix: str,
    output_dir: Path,
    existing: dict[str, pl.DataFrame | None],
    end: str | None,
) -> list[pl.DataFrame]:
    """Collect symbols current through end from disk and fetch the rest in one batch."""
    market = pl.lit(SUFFIX_MARKET[suffix]).cast(pl.Categorical).alias("market")
    frames: list[pl.DataFrame] = []
    stale: dict[str, str] = {}
    for code in update_codes:
        frame = existing[code]
        if frame is None:
            continue
        last = cast("date", frame.get_column("date").max())
        update_start = (last + timedelta(days=1)).isoformat()
        if end is not None and update_start >= end:
            frames.append(frame.with_columns(market))
        else:
            stale[code] = update_start

    if frames:
        logger.info("up to date count=%s", len(frames))
    if stale:
        start = min(stale.values())
        logger.info(
            "batch downloading update count=%s from=%s to=%s",
            len(stale),
            start,
            end,
        )
        span = {"start": start, "end": end}
        frames.extend(fetch_and_save(list(stale), suffix, output_dir, existing, span))
    return frames


def load_dead(output: Path) -> set[str]:
    """Read the persisted set of symbols yfinance has no advancing data for."""
    path = output / DEAD_FILE
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def save_dead(output: Path, dead: set[str]) -> None:
    """Persist the dead-symbol set as a sorted JSON list."""
    path = output / DEAD_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(dead), indent=2))


def fetch_market(
    codes: list[str],
    suffix: str,
    output: Path,
    window: Window,
    *,
    force: bool,
) -> tuple[list[pl.DataFrame], set[str]]:
    """Fetch one market's live symbols and collect stale ones from disk."""
    market_name = SUFFIX_MARKET[suffix]
    output_dir = output / market_name
    existing = {code: read_symbol(output_dir / f"{code}.csv") for code in codes}
    market = pl.lit(market_name).cast(pl.Categorical).alias("market")

    if force:
        logger.info(
            "force re-fetch count=%s from=%s to=%s",
            len(codes),
            window.start,
            window.end,
        )
        span = {"start": window.start, "end": window.end}
        frames = fetch_and_save(codes, suffix, output_dir, existing, span)
        logger.info("done market=%s forced=%s", market_name, len(codes))
        return frames, set()

    stale = find_stale(existing)
    frames: list[pl.DataFrame] = [
        cast("pl.DataFrame", existing[code]).with_columns(market) for code in stale
    ]
    live = [code for code in codes if code not in stale]

    if window.period is not None:
        logger.info(
            "batch downloading all count=%s period=%s",
            len(live),
            window.period,
        )
        span = {"period": window.period}
        frames.extend(fetch_and_save(live, suffix, output_dir, existing, span))
        logger.info("done market=%s", market_name)
        return frames, set()

    new_codes = [code for code in live if existing[code] is None]
    update_codes = [code for code in live if existing[code] is not None]

    if new_codes:
        logger.info(
            "batch downloading new count=%s from=%s",
            len(new_codes),
            window.start,
        )
        span = {"start": window.start, "end": window.end}
        frames.extend(fetch_and_save(new_codes, suffix, output_dir, existing, span))

    frames.extend(fetch_updates(update_codes, suffix, output_dir, existing, window.end))

    missing = {code for code in new_codes if not (output_dir / f"{code}.csv").exists()}
    logger.info(
        "done market=%s new=%s existing=%s stale=%s",
        market_name,
        len(new_codes),
        len(update_codes),
        len(stale),
    )
    return frames, missing


def run(
    symbols: list[str] | None,
    window: Window,
    output: Path,
    *,
    force: bool,
) -> pl.DataFrame | None:
    """Download OHLCV for the given symbols, creating or updating CSVs.

    New symbols fetch from the window start, existing ones from the day after
    their latest bar. Stale symbols aggregate from disk without fetching, and
    symbols yfinance never returns go to dead.json. Force re-fetches everything.
    """
    dead = load_dead(output)
    if symbols is not None:
        tse_codes, otc_codes = classify_symbols(symbols)
    else:
        tse_codes, otc_codes = fetch_sim_symbols()
    tse_codes = [code for code in tse_codes if code not in dead]
    otc_codes = [code for code in otc_codes if code not in dead]

    collected: list[pl.DataFrame] = []
    new_dead: set[str] = set()
    for codes, suffix in ((tse_codes, TSE_SUFFIX), (otc_codes, OTC_SUFFIX)):
        if not codes:
            continue
        frames, missing = fetch_market(codes, suffix, output, window, force=force)
        collected.extend(frames)
        new_dead.update(missing)

    if new_dead:
        logger.info("marked dead count=%s", len(new_dead))
        save_dead(output, dead | new_dead)

    if not collected:
        return None
    dataset = pl.concat(collected).sort(["market", "code", "date"])
    latest = cast("date", dataset.get_column("date").max())
    logger.info("latest bar=%s symbols=%s", latest.isoformat(), len(collected))
    return dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        help="End date inclusive; ignored when --period is set",
    )
    parser.add_argument(
        "--output",
        default="data/yfinance",
        metavar="DIR",
        help="Output root directory holding tse/ and otc/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch the full --from..--to window and overwrite existing bars",
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
    end = None
    if not args.period:
        to_date = date.fromisoformat(args.to_date)
        end = (to_date + timedelta(days=1)).isoformat()
    window = Window(start=args.from_date, end=end, period=args.period)
    output = Path(args.output)
    dataset = run(args.symbols, window, output, force=args.force)

    if not args.no_aggregate:
        parquet = output / "stocks.parquet"
        if args.symbols is None and dataset is not None:
            # A full run already holds every symbol in memory, so skip re-reading.
            burn_the_stock.aggregate.save_parquet(dataset.lazy(), parquet)
        else:
            burn_the_stock.aggregate.run(output, parquet)
