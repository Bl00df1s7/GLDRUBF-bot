"""Pluggable strategy layer for Golden Bot infrastructure.

The bot runtime depends only on this package's interface, so a strategy
(e.g. frozen C5) can be swapped without touching infrastructure modules.
"""

from src.strategies.base import Strategy, StrategyDecision
from src.strategies.c5_core import (
    Decision,
    MarginRateError,
    DataError,
    block_entry,
)
from src.strategies.c5_core import evaluate as c5_evaluate
from src.strategies.c5_signals import C5Strategy

STRATEGY_NAME = "C5"


def build_strategy():
    """Return the active strategy for the trading runtime."""
    return C5Strategy()


__all__ = [
    "Strategy",
    "StrategyDecision",
    "C5Strategy",
    "Decision",
    "DataError",
    "MarginRateError",
    "c5_evaluate",
    "block_entry",
    "STRATEGY_NAME",
    "build_strategy",
]
