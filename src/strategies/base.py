"""Strategy interface: signal generation must be isolated from infrastructure."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyDecision:
    """Frozen decision produced by a strategy for one closed candle."""

    entry: Optional[str] = None          # "LONG" | "SHORT" | None
    exit: Optional[str] = None           # "EXIT_LONG" | "EXIT_SHORT" | None
    atr: Optional[float] = None          # ATR14 of the candle
    reason: str = ""
    details: dict = field(default_factory=dict)


class Strategy:
    """Base contract implemented by every pluggable strategy."""

    name = "BASE"

    def prepare(self, df):
        """Attach indicator columns to an OHLCV dataframe (pure function)."""
        raise NotImplementedError

    def decide(self, last_closed, position_direction: Optional[str]) -> StrategyDecision:
        """Return entry/exit decision for the last closed candle."""
        raise NotImplementedError
