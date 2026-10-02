"""C5 frozen trend-following strategy (CNYRUBF_SPBFUT).

Signal logic is FROZEN. Do not add filters, optimization, ATR stops,
new entry/exit conditions or discretionary logic here.

Entry: Donchian breakout N = 10 (close crosses above/below the channel
built from the PREVIOUS N candles only — no lookahead).
Exit:  Donchian counter-channel M = 5 (long exits when close falls below
the 5-bar low of previous bars; short exits when close rises above the
5-bar high of previous bars).

ATR14 is computed for risk sizing only (see src/risk_budget.py); it is
NOT part of the signal.
"""

from typing import Optional

import pandas as pd

from config.settings import (
    C5_ATR_LEN,
    C5_ENTRY_DONCHIAN,
    C5_EXIT_DONCHIAN,
)
from src.indicators import calculate_atr
from src.strategies.base import Strategy, StrategyDecision


def compute_c5_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach frozen C5 indicator columns to an OHLCV dataframe.

    Pure function shared by the bot and the reference backtester so that
    both sides compute signals on identical definitions.
    """
    data = df.copy().reset_index(drop=True)

    # Entry channel: extremes of the previous N closed candles (shift(1)).
    data["c5_entry_upper"] = (
        data["high"].rolling(C5_ENTRY_DONCHIAN).max().shift(1)
    )
    data["c5_entry_lower"] = (
        data["low"].rolling(C5_ENTRY_DONCHIAN).min().shift(1)
    )

    # Exit channel: extremes of the previous M closed candles (shift(1)).
    data["c5_exit_upper"] = (
        data["high"].rolling(C5_EXIT_DONCHIAN).max().shift(1)
    )
    data["c5_exit_lower"] = (
        data["low"].rolling(C5_EXIT_DONCHIAN).min().shift(1)
    )

    # Frozen signal definition.
    data["c5_long_signal"] = data["close"] > data["c5_entry_upper"]
    data["c5_short_signal"] = data["close"] < data["c5_entry_lower"]
    data["c5_exit_long"] = data["close"] < data["c5_exit_lower"]
    data["c5_exit_short"] = data["close"] > data["c5_exit_upper"]

    # Risk input only (not a signal component).
    data["atr"] = calculate_atr(data, C5_ATR_LEN)

    return data


class C5Strategy(Strategy):
    """Golden Bot adapter around the frozen C5 signal logic."""

    name = "C5"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return compute_c5_indicators(df)

    def decide(
        self,
        last_closed,
        position_direction: Optional[str],
    ) -> StrategyDecision:
        """Return the frozen C5 decision for one closed candle.

        Exit takes priority over entry: while a position is open only the
        M=5 counter-channel can produce an exit; entries are evaluated
        when flat.
        """
        get = (
            last_closed.get
            if isinstance(last_closed, dict)
            else lambda k, d=None: getattr(last_closed, k, d)
        )

        long_sig = bool(get("c5_long_signal", False))
        short_sig = bool(get("c5_short_signal", False))
        atr = get("atr")

        atr_value = None
        if atr is not None and not pd.isna(atr):
            atr_value = float(atr)

        decision = StrategyDecision(atr=atr_value)

        if long_sig and short_sig:
            # Anomaly guard: contradictory breakout => no action, loud log.
            decision.reason = "DONCHIAN_ANOMALY_BOTH_SIGNALS"
            return decision

        if position_direction in ("LONG", "SHORT"):
            if position_direction == "LONG" and bool(get("c5_exit_long", False)):
                decision.exit = "EXIT_LONG"
                decision.reason = "C5_EXIT_DONCHIAN_M5_LOW"
            elif position_direction == "SHORT" and bool(get("c5_exit_short", False)):
                decision.exit = "EXIT_SHORT"
                decision.reason = "C5_EXIT_DONCHIAN_M5_HIGH"
            else:
                decision.reason = "HOLD_POSITION"
            return decision

        if long_sig:
            decision.entry = "LONG"
            decision.reason = "C5_ENTRY_DONCHIAN_N10_BREAKOUT_UP"
        elif short_sig:
            decision.entry = "SHORT"
            decision.reason = "C5_ENTRY_DONCHIAN_N10_BREAKOUT_DOWN"
        else:
            decision.reason = "NO_SIGNAL"

        return decision
