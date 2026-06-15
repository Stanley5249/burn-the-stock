"""Fast live quotes from the Fugle intraday API for the trader's pre-1pm loop."""

import argparse
import asyncio
import logging
import os
from typing import Any

import httpx
import polars as pl
from dotenv import load_dotenv
from pydantic import BaseModel

import burn_the_stock.log

logger = logging.getLogger(__name__)

FUGLE_QUOTE_URL = "https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote"


class Quote(BaseModel):
    """A single intraday quote for one symbol."""

    symbol: str
    open: float
    high: float
    low: float
    last: float
    close: float | None
    is_closed: bool
    avg_price: float
    change_percent: float
    bid: float | None
    ask: float | None

    @classmethod
    def from_fugle(cls, symbol: str, raw: dict[str, Any]) -> Quote:
        """Build a Quote from a Fugle intraday/quote response body.

        Returns:
            The parsed Quote.
        """
        bids = raw.get("bids") or []
        asks = raw.get("asks") or []
        return cls(
            symbol=symbol,
            open=raw.get("openPrice", 0.0),
            high=raw.get("highPrice", 0.0),
            low=raw.get("lowPrice", 0.0),
            last=raw.get("lastPrice", 0.0),
            close=raw.get("closePrice"),
            is_closed=bool(raw.get("isClose")),
            avg_price=raw.get("avgPrice", 0.0),
            change_percent=raw.get("changePercent", 0.0),
            bid=bids[0]["price"] if bids else None,
            ask=asks[0]["price"] if asks else None,
        )


async def fetch_quote(
    client: httpx.AsyncClient,
    symbol: str,
) -> Quote | None:
    """Fetch one symbol's quote, returning None on any HTTP error.

    Returns:
        The Quote, or None when the request fails.
    """
    try:
        response = await client.get(f"{FUGLE_QUOTE_URL}/{symbol}", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("quote failed symbol=%s error=%s", symbol, error)
        return None
    return Quote.from_fugle(symbol, response.json())


async def fetch_all(symbols: list[str], api_key: str) -> dict[str, Quote]:
    """Fetch all symbols concurrently.

    Returns:
        A mapping of symbol to Quote, skipping any that failed.
    """
    headers = {"X-API-KEY": api_key}
    async with httpx.AsyncClient(headers=headers) as client:
        quotes = await asyncio.gather(*[fetch_quote(client, s) for s in symbols])
    return {quote.symbol: quote for quote in quotes if quote is not None}


def get_quotes(symbols: list[str], api_key: str | None = None) -> dict[str, Quote]:
    """Fetch quotes for symbols, reading FUGLE_API_KEY from the env if needed.

    Returns:
        A mapping of symbol to Quote.
    """
    if api_key is None:
        load_dotenv()
        api_key = os.environ["FUGLE_API_KEY"]
    return asyncio.run(fetch_all(symbols, api_key))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the quotes script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fetch live Fugle quotes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        default=["2330", "2454", "2317"],
        metavar="SYMBOL",
        help="Stock codes without suffix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    burn_the_stock.log.setup()
    args = parse_args()
    quotes = get_quotes(args.symbols)

    columns = ["symbol", "open", "high", "low", "last", "change_percent", "is_closed"]
    frame = pl.DataFrame([quote.model_dump() for quote in quotes.values()])
    with pl.Config(tbl_hide_dataframe_shape=True, float_precision=2):
        print(frame.select(columns))
