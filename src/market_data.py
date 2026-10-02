"""
Market data loading and price retrieval.
SIGNAL ONLY MODE - Uses t_tech.invest if available, otherwise mock data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

try:
    from t_tech.invest import CandleInterval
    T_TECH_AVAILABLE = True
except ImportError:
    T_TECH_AVAILABLE = False
    CandleInterval = None

from src.client_factory import get_client


def quotation_to_float(value) -> float:
    """
    Safe conversion of Quotation to float.
    
    Args:
        value: Quotation object or numeric value
        
    Returns:
        Float value or np.nan if None
    """
    if value is None:
        return np.nan
    
    if isinstance(value, (int, float, np.number)):
        return float(value)
    
    if hasattr(value, "units") and hasattr(value, "nano"):
        return float(value.units) + float(value.nano) / 1_000_000_000
    
    if hasattr(value, "value"):
        return float(value.value)
    
    return float(value)


def candle_to_row(candle) -> dict:
    """Convert candle object to dictionary row."""
    return {
        "time": candle.time,
        "open": quotation_to_float(candle.open),
        "high": quotation_to_float(candle.high),
        "low": quotation_to_float(candle.low),
        "close": quotation_to_float(candle.close),
        "volume": candle.volume,
    }


def load_candles(
    token: str,
    uid: str,
    candles_count: int = 200,
    timeframe: str = "4H"
) -> pd.DataFrame:
    """
    Load recent candles from T-Invest API.
    
    Args:
        token: T-Invest API token
        uid: Instrument UID
        candles_count: Number of candles to load
        timeframe: Candle timeframe (e.g., "4H", "1H", "15m")
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        RuntimeError: If t_tech is not available or data cannot be loaded
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available. Install with: pip install t-tech")
    
    now_utc = datetime.now(timezone.utc)

    # Calculate days needed based on timeframe.
    # 1d = 1 candle/day, 4H = 6, 1H = 24, 15m = 96, 5m = 288 (24h bound).
    if timeframe == "1d":
        candles_per_day = 1
    elif timeframe == "4H":
        candles_per_day = 6
    elif timeframe == "1H":
        candles_per_day = 24
    elif timeframe == "15m":
        candles_per_day = 96
    elif timeframe == "5m":
        candles_per_day = 288
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    days = int(candles_count / candles_per_day) + 10
    start_date = now_utc - timedelta(days=days)

    rows = []
    current = start_date
    chunk = timedelta(days=90)

    # Map timeframe string to CandleInterval (T-Invest API).
    # NOTE: deliberately NO silent fallback to another interval — an
    # unknown timeframe must fail loudly, otherwise C5 daily Donchian/ATR
    # could be computed from wrong-bar data (lookahead-free contract).
    timeframe_map = {
        "1d": getattr(CandleInterval, 'CANDLE_INTERVAL_DAY', None),
        "4H": getattr(CandleInterval, 'CANDLE_INTERVAL_4_HOUR', None),
        "2H": getattr(CandleInterval, 'CANDLE_INTERVAL_2_HOUR', None),
        "1H": getattr(CandleInterval, 'CANDLE_INTERVAL_HOUR', None),
        "30m": getattr(CandleInterval, 'CANDLE_INTERVAL_30_MIN', None),
        "15m": getattr(CandleInterval, 'CANDLE_INTERVAL_15_MIN', None),
        "5m": getattr(CandleInterval, 'CANDLE_INTERVAL_5_MIN', None),
        "1m": getattr(CandleInterval, 'CANDLE_INTERVAL_1_MIN', None),
    }

    candle_interval = timeframe_map.get(timeframe)
    if candle_interval is None:
        raise ValueError(
            f"Неизвестный таймфрейм {timeframe!r}: явного маппинга в "
            "CandleInterval нет, silent fallback запрещён."
        )

    while current < now_utc:
        chunk_end = min(current + chunk, now_utc)
        
        with get_client(token) as services:
            response = services.market_data.get_candles(
                instrument_id=uid,
                from_=current,
                to=chunk_end,
                interval=candle_interval,
            )
        
        rows.extend(candle_to_row(candle) for candle in response.candles)
        current = chunk_end
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        return df
    
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    
    return df.tail(candles_count).reset_index(drop=True)


def get_current_price(token: str, uid: str) -> float:
    """
    Get current last price for instrument.
    
    Args:
        token: T-Invest API token
        uid: Instrument UID
        
    Returns:
        Current price as float
        
    Raises:
        RuntimeError: If t_tech is not available or price cannot be retrieved
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available")
    
    with get_client(token) as services:
        response = services.market_data.get_last_prices(
            instrument_id=[uid]
        )
    
    if not response.last_prices:
        raise RuntimeError("Не удалось получить текущую цену инструмента")
    
    return quotation_to_float(response.last_prices[0].price)


def is_trading_time(now: datetime = None) -> bool:
    """Return whether the current Moscow time is after the 03:00 session start."""
    current = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("Europe/Moscow"))
    return current.time() >= time(3, 0)


# ---------------------------------------------------------------------------
# C5 runtime data helpers (daily bars + closed 5m control candle)
# ---------------------------------------------------------------------------

MSK_TZ = ZoneInfo("Europe/Moscow")


def daily_bars_from_df(df, exclude_date=None):
    """Convert a daily-candle DataFrame into lookahead-free DailyBar list.

    * index is by calendar date (MSK);
    * ``exclude_date`` (today's MSK date) is ALWAYS dropped — the daily
      Donchian/ATR channel must not include the current day;
    * ``prev_close`` for True Range is H+L+C / 3 of the previous bar
      (frozen reference convention used by c5_core and the backtest).
    """
    from src.strategies.c5_core import DailyBar

    if df is None or len(df) == 0:
        return []
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert(MSK_TZ)
    d["date"] = d["time"].dt.date
    d = d.drop_duplicates("date").sort_values("date")
    if exclude_date is not None:
        d = d[d["date"] < exclude_date]

    rows = list(d.itertuples(index=False))
    bars = []
    for k, r in enumerate(rows):
        prev = None
        if k > 0:
            p = rows[k - 1]
            prev = (float(p.high) + float(p.low) + float(p.close)) / 3.0
        bars.append(DailyBar(trade_date=r.date, high=float(r.high),
                             low=float(r.low), close=float(r.close),
                             prev_close=prev))
    return bars


def latest_closed_control_candle(token, uid, timeframe, hour, minute,
                                 now=None, lookback_days=7):
    """Return the most recent CLOSED `hour:minute` candle of `timeframe`.

    A candle is considered closed only when its start timestamp equals the
    scheduled point AND that point is strictly in the past. Never returns
    an unclosed candle and never falls back to another timestamp.
    Returns (candle_row | None).
    """
    now = now or datetime.now(timezone.utc)
    df = load_candles(token, uid, candles_count=lookback_days * 288,
                      timeframe=timeframe)
    if df is None or df.empty:
        return None
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert(MSK_TZ)
    d = d[(d["time"].dt.hour == hour) & (d["time"].dt.minute == minute)]
    d = d[d["time"] <= now.astimezone(MSK_TZ)]
    if d.empty:
        return None
    return d.sort_values("time").iloc[-1]


def previous_same_tf_bar(token, uid, timeframe, control_time, count=6):
    """Previous closed bar of the same timeframe before control_time."""
    df = load_candles(token, uid, candles_count=count * 40, timeframe=timeframe)
    if df is None or df.empty:
        return None
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    ct = pd.to_datetime(control_time)
    if ct.tzinfo is None:
        ct = ct.tz_localize("UTC")
    d = d[d["time"] < ct]
    if d.empty:
        return None
    return d.sort_values("time").iloc[-1]
