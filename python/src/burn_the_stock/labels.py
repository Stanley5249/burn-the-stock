"""Replicate the Rust swing labeler and analyze its class and reward distribution.

This is a faithful port of ``crates/trainer/src/label.rs::swing_labels`` so the
label distribution measured here matches what the trainer actually feeds the
model. The port is checked against the same unit-test vectors as the Rust source
in ``python/tests/test_labels.py``.

On top of the discrete label, each row also gets the signed fractional move to
the trend extreme it is heading toward. That scalar is the per-sample reward the
expected-value loss and metric would use, so reporting its distribution tells us
how much profit actually sits in the Buy and Sell calls.
"""

import argparse
from dataclasses import dataclass

import polars as pl

# Class indices match the Rust enum: Sell = 0, Hold = 1, Buy = 2.
SELL, HOLD, BUY = 0, 1, 2
CLASS_NAME = {SELL: "Sell", HOLD: "Hold", BUY: "Buy"}

LABEL_THRESHOLD = 0.03


@dataclass
class Up:
    """Rising trend heading toward ``maximum`` from the running ``minimum``."""

    minimum: float
    maximum: float


@dataclass
class Flat:
    """Undecided trend sitting at ``flat`` until the first move."""

    flat: float


@dataclass
class Down:
    """Falling trend heading toward ``minimum`` from the running ``maximum``."""

    minimum: float
    maximum: float


Phase = Up | Flat | Down


def advance(phase: Phase, price: float, abs_threshold: float) -> Phase:
    """Step the swing state machine to the next price walked from the end.

    Args:
        phase: Current trend phase.
        price: The price being folded in, earlier in time than ``phase``.
        abs_threshold: Reversal magnitude in price units for this step.

    Returns:
        The phase after accounting for ``price``.
    """
    if isinstance(phase, Flat):
        if price < phase.flat:
            result = Up(price, phase.flat)
        elif price > phase.flat:
            result = Down(phase.flat, price)
        else:
            result = Flat(phase.flat)
    elif isinstance(phase, Up):
        if price - phase.minimum > abs_threshold:
            result = Down(phase.minimum, price)
        elif price >= phase.minimum:
            result = Up(phase.minimum, phase.maximum)
        else:
            result = Up(price, phase.maximum)
    elif phase.maximum - price > abs_threshold:
        result = Up(price, phase.maximum)
    elif phase.maximum >= price:
        result = Down(phase.minimum, phase.maximum)
    else:
        result = Down(phase.minimum, price)
    return result


def classify(phase: Phase, price: float, abs_threshold: float) -> tuple[int, float]:
    """Read the action label and reward off a phase at ``price``.

    The reward points at the upcoming extreme even on Hold rows. Only the label
    waits for the turning point, since while the price still drifts off the
    running extreme it is not a confirmed reversal yet.

    Args:
        phase: Trend phase at this row.
        price: The price at this row.
        abs_threshold: Reversal magnitude in price units for this step.

    Returns:
        The action class index and the signed fractional move to the extreme.
    """
    if isinstance(phase, Flat):
        return HOLD, 0.0
    if isinstance(phase, Up):
        reward = (phase.maximum - price) / price
        confirmed = price <= phase.minimum and phase.maximum - price > abs_threshold
        return (BUY if confirmed else HOLD), reward
    reward = (phase.minimum - price) / price
    confirmed = price >= phase.maximum and price - phase.minimum > abs_threshold
    return (SELL if confirmed else HOLD), reward


def swing_labels(
    prices: list[float],
    rel_threshold: float,
) -> tuple[list[int], list[float]]:
    """Label every price except the last with the action a perfect trader takes.

    Args:
        prices: Close prices in ascending date order.
        rel_threshold: Reversal magnitude that confirms a swing, as a fraction
            of price.

    Returns:
        The per-row labels and the signed fractional move to the extreme each
        row heads toward. Both lists are one shorter than ``prices``.
    """
    if not prices:
        return [], []

    phase: Phase = Flat(prices[-1])
    labels: list[int] = []
    rewards: list[float] = []

    # Walk from the second-to-last price back to the first, exactly as the Rust
    # closure consumes ``prices.rev()`` after taking the final price for Flat.
    for price in reversed(prices[:-1]):
        abs_threshold = price * rel_threshold
        phase = advance(phase, price, abs_threshold)
        label, reward = classify(phase, price, abs_threshold)
        labels.append(label)
        rewards.append(reward)

    labels.reverse()
    rewards.reverse()
    return labels, rewards


def label_frame(stocks: pl.DataFrame) -> pl.DataFrame:
    """Attach per-row label and reward to every ticker, dropping the last row.

    Args:
        stocks: Frame with ``date``, ``code``, and ``close`` columns.

    Returns:
        The same rows minus each ticker's label-less last row, with ``label``
        and ``reward`` columns added.
    """
    pieces: list[pl.DataFrame] = []
    for group in stocks.sort("date").partition_by("code", maintain_order=True):
        closes = group.get_column("close").to_list()
        if len(closes) <= 1:
            continue
        labels, rewards = swing_labels(closes, LABEL_THRESHOLD)
        # Labels align to every row but the last, so drop that final row to match.
        kept = group.head(len(labels))
        pieces.append(
            kept.with_columns(
                pl.Series("label", labels, dtype=pl.Int8),
                pl.Series("reward", rewards, dtype=pl.Float64),
            ),
        )
    return pl.concat(pieces)


def report(name: str, frame: pl.DataFrame) -> None:
    """Print the class shares and the Buy/Sell move magnitude of a frame.

    Args:
        name: Section label for the printed block.
        frame: A labeled frame from ``label_frame``.
    """
    total = frame.height
    print(f"\n=== {name} ({total:,} labeled rows) ===")

    counts = frame.group_by("label").agg(pl.len().alias("count")).sort("label")
    for row in counts.iter_rows(named=True):
        share = 100.0 * row["count"] / total if total else 0.0
        print(f"  {CLASS_NAME[row['label']]:>4}: {row['count']:>10,}  {share:5.1f}%")

    # The reward magnitude on the action classes is the profit the EV objective
    # would chase, so summarize it rather than the zero-filled Holds.
    for label in (BUY, SELL):
        present = frame.filter(pl.col("label") == label)
        if present.height == 0:
            continue
        stats = present.select(
            median=pl.col("reward").abs().median(),
            mean=pl.col("reward").abs().mean(),
            p90=pl.col("reward").abs().quantile(0.9),
            top=pl.col("reward").abs().max(),
        ).row(0, named=True)
        print(
            f"  {CLASS_NAME[label]} move%: "
            f"median {100 * stats['median']:.2f}  "
            f"mean {100 * stats['mean']:.2f}  "
            f"p90 {100 * stats['p90']:.2f}  "
            f"max {100 * stats['top']:.2f}",
        )


def main() -> None:
    """Parse arguments, label the data, and print the distribution report."""
    parser = argparse.ArgumentParser(description="Analyze swing-label distribution")
    parser.add_argument("--data", default="data/yfinance/stocks.parquet")
    parser.add_argument(
        "--valid-days",
        type=int,
        default=252,
        help="Recent-window length that validates, matching the trainer split.",
    )
    args = parser.parse_args()

    stocks = pl.read_parquet(args.data).select("date", "code", "close")

    # Mirror the trainer's correctness filter: a non-positive close is corrupt
    # back-adjustment, so drop it before labeling and report what went.
    nonpos = stocks.filter(pl.col("close") <= 0)
    print(
        f"dropping {nonpos.height:,} rows with close <= 0 across "
        f"{nonpos.get_column('code').n_unique()} tickers",
    )
    stocks = stocks.filter(pl.col("close") > 0)

    labeled = label_frame(stocks)

    report("overall", labeled)

    # Surface the largest moves so suspect penny-stock or ETF ticks are visible
    # before they are clipped in the metric.
    top = (
        labeled.with_columns(pl.col("reward").abs().alias("abs_reward"))
        .sort("abs_reward", descending=True)
        .head(10)
        .select("date", "code", "close", "reward")
    )
    print("\ntop 10 rows by absolute move:")
    print(top)

    # Mirror the trainer: one global cutoff at the latest date minus valid_days,
    # everything earlier trains and the recent window validates. Keep the cutoff
    # in expression space to avoid Python-level date arithmetic.
    cutoff = pl.col("date").max() - pl.duration(days=args.valid_days)
    print(f"\ncutoff date: {labeled.select(cutoff.alias('cutoff')).item()}")
    report("train", labeled.filter(pl.col("date") < cutoff))
    report("valid", labeled.filter(pl.col("date") >= cutoff))


if __name__ == "__main__":
    main()
