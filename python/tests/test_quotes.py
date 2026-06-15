"""Tests for the quotes module."""

from burn_the_stock.quotes import Quote


def test_from_fugle_full() -> None:
    """Parse a complete Fugle quote body."""
    raw = {
        "openPrice": 100.0,
        "highPrice": 110.0,
        "lowPrice": 95.0,
        "lastPrice": 108.0,
        "closePrice": 108.0,
        "isClose": True,
        "avgPrice": 104.0,
        "changePercent": 2.5,
        "bids": [{"price": 107.5, "size": 10}],
        "asks": [{"price": 108.5, "size": 8}],
    }
    quote = Quote.from_fugle("2330", raw)
    assert quote.symbol == "2330"
    assert quote.high == 110.0
    assert quote.is_closed is True
    assert quote.bid == 107.5
    assert quote.ask == 108.5


def test_from_fugle_no_book() -> None:
    """Missing bid/ask books parse to None without error."""
    quote = Quote.from_fugle("2330", {"lastPrice": 108.0})
    assert quote.bid is None
    assert quote.ask is None
    assert quote.is_closed is False
