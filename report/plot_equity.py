"""Plot backtest equity curves against an equal-weight buy-and-hold baseline.

Each LABEL=CSV argument is one equity curve (CSV columns date,equity). The
baseline comes from the OHLCV parquet over the combined date window, equal-weight
buy-and-hold of every ticker priced on the first window day. The y-axis is log
scaled so curves of very different magnitude stay legible.

    uv run report/plot_equity.py honest=report/honest-equity.csv exploit=report/exploit-equity.csv
"""

import argparse

import altair as alt
import polars as pl

START_CASH = 100_000_000
BASELINE = "baseline"


def parse_curve(spec: str) -> tuple[str, str]:
    """Split a LABEL=CSV argument into its label and path."""
    label, sep, path = spec.partition("=")
    if not sep:
        msg = f"expected LABEL=CSV, got {spec!r}"
        raise argparse.ArgumentTypeError(msg)
    return label, path


def load_strategies(curves: list[tuple[str, str]]) -> pl.DataFrame:
    """Read each equity CSV (columns date,equity) and tag it with its label."""
    return pl.concat(
        pl.read_csv(path, try_parse_dates=True)
        .select("date", "equity")
        .with_columns(series=pl.lit(label))
        for label, path in curves
    )


def equal_weight_baseline(parquet: str, start: object, end: object) -> pl.DataFrame:
    """Equal-weight buy-and-hold equity from the parquet over [start, end].

    Enter every ticker priced on the first window day, then hold. Restricting to
    that universe avoids IPO-mid-window distortion. The stored close is already
    split and dividend adjusted, so there is no separate adjclose column.
    """
    window = (
        pl.scan_parquet(parquet)
        .filter(pl.col("date").is_between(start, end))
        .select("code", "date", "close")
        .sort("code", "date")
    )
    priced_at_start = (
        window.filter(pl.col("date") == start)
        .select("code")
        .collect()["code"]
        .to_list()
    )
    return (
        window.filter(pl.col("code").is_in(priced_at_start))
        .with_columns(norm=pl.col("close") / pl.col("close").first().over("code"))
        .group_by("date")
        .agg(equity=(pl.col("norm").mean() * START_CASH).cast(pl.Float64))
        .with_columns(series=pl.lit(BASELINE))
        .select("date", "equity", "series")
        .sort("date")
        .collect()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "curves",
        type=parse_curve,
        nargs="+",
        metavar="LABEL=CSV",
        help="one equity curve per arg, CSV columns date,equity",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="report/equity-curve.svg",
        help="output path",
    )
    parser.add_argument(
        "--parquet",
        default="data/yfinance/stock_history.parquet",
        help="OHLCV parquet for the baseline",
    )
    args = parser.parse_args()

    strategies = load_strategies(args.curves)

    start, end = strategies["date"].min(), strategies["date"].max()

    curves = pl.concat([strategies, equal_weight_baseline(args.parquet, start, end)])

    # pin the log domain to the data, otherwise nice=True snaps it out to whole
    # powers of 10 and leaves the curves squished in a thin band
    low, high = curves["equity"].min(), curves["equity"].max()

    chart = (
        alt.Chart(curves)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="date"),
            y=alt.Y(
                "equity:Q",
                title="equity (NT$)",
                scale=alt.Scale(
                    type="log", domain=[low * 0.95, high * 1.05], nice=False
                ),
            ),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
        )
        .properties(
            width=900,
            height=440,
            title="Backtest equity by fill rule vs equal-weight buy-and-hold",
        )
    )
    chart.save(args.out)

    print(f"window : {start} -> {end}")

    for label in curves["series"].unique(maintain_order=True):
        final = curves.filter(pl.col("series") == label).sort("date")["equity"][-1]

        print(f"{label:34s} {final / START_CASH - 1:+.2%}")

    print(f"wrote  : {args.out}")


if __name__ == "__main__":
    main()
