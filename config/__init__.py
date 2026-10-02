"""Configuration package."""
from .settings import (
    TIMEFRAME,
    DONCHIAN_LEN,
    ATR_LEN,
    SL_ATR,
    TP_PCT,
    BE_PCT,
    SAR_START,
    SAR_INC,
    SAR_MAX,
    TARGET_TICKER,
)

__all__ = [
    "TIMEFRAME",
    "DONCHIAN_LEN",
    "ATR_LEN",
    "SL_ATR",
    "TP_PCT",
    "BE_PCT",
    "SAR_START",
    "SAR_INC",
    "SAR_MAX",
    "TARGET_TICKER",
]
