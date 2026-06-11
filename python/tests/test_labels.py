"""Tests for the swing labeler port.

These mirror the unit-test vectors in ``crates/trainer/src/label.rs`` so the
Python analyzer stays faithful to the Rust labeler the trainer uses.
"""

import pytest
from burn_the_stock.labels import BUY, HOLD, SELL, swing_labels

LABEL_CASES: list[tuple[list[float], list[int]]] = [
    ([], []),
    ([100.0], []),
    ([100.0, 105.0], [BUY]),
    ([100.0, 95.0], [SELL]),
    ([100.0, 100.0], [HOLD]),
    ([100.0, 101.0, 100.5], [HOLD, HOLD]),
    ([100.0, 105.0, 100.0], [BUY, SELL]),
    ([100.0, 105.0, 103.0, 105.0, 107.0, 100.0], [BUY, HOLD, BUY, HOLD, SELL]),
    ([100.0, 93.0, 95.0, 97.0, 95.0, 100.0], [SELL, BUY, BUY, HOLD, BUY]),
]


@pytest.mark.parametrize(("prices", "expected"), LABEL_CASES)
def test_swing_labels_match_rust(prices: list[float], expected: list[int]) -> None:
    """Reproduce the label vectors asserted by the Rust unit tests."""
    labels, _ = swing_labels(prices, 0.03)
    assert labels == expected


REWARD_CASES = [
    ([100.0, 105.0], [0.05]),
    ([100.0, 95.0], [-0.05]),
    ([100.0, 105.0, 100.0], [0.05, (100.0 - 105.0) / 105.0]),
]


@pytest.mark.parametrize(("prices", "expected"), REWARD_CASES)
def test_swing_rewards(prices: list[float], expected: list[float]) -> None:
    """Each row reports the signed fractional move to its trend extreme."""
    _, rewards = swing_labels(prices, 0.03)
    assert rewards == pytest.approx(expected)
