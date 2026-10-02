"""
Technical indicators calculation.
SIGNAL ONLY mode - no trading operations.
"""

import pandas as pd
import numpy as np

from config.settings import DONCHIAN_LEN, ATR_LEN


def calculate_atr(df: pd.DataFrame, length: int = None) -> pd.Series:
    """
    Calculate Average True Range (ATR) using Wilder smoothing.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        length: ATR period (default: ATR_LEN from settings)
        
    Returns:
        ATR series
    """
    if length is None:
        length = ATR_LEN
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    prev_close = close.shift(1)
    
    # True Range calculation
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    # Wilder smoothing (exponential moving average with alpha = 1/length)
    atr = tr.copy()
    for i in range(length, len(tr)):
        if pd.isna(atr.iloc[i-1]):
            # First valid ATR is simple MA of first 'length' TR values
            atr.iloc[i] = tr.iloc[i-length+1:i+1].mean()
        else:
            atr.iloc[i] = (atr.iloc[i-1] * (length - 1) + tr.iloc[i]) / length
    
    return atr


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all required indicators.
    Donchian Channel uses only previous candles (shift by 1 to avoid lookahead).
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    data = df.copy().reset_index(drop=True)
    
    # ATR
    data["atr"] = calculate_atr(data, ATR_LEN)
    
    # Donchian Channel (only previous candles, shift by 1 to avoid lookahead)
    # For candle t, we use highs[t-DONCHIAN_LEN : t-1], not including candle t
    data["donchian_upper"] = data["high"].rolling(DONCHIAN_LEN).max().shift(1)
    data["donchian_lower"] = data["low"].rolling(DONCHIAN_LEN).min().shift(1)
    
    # Entry signals (based on close vs previous Donchian)
    data["long_signal"] = data["close"] > data["donchian_upper"]
    data["short_signal"] = data["close"] < data["donchian_lower"]
    
    return data


def calculate_sar(
    df: pd.DataFrame,
    start: float,
    inc: float,
    maximum: float
) -> tuple:
    """
    Calculate Parabolic SAR with reversal detection.
    Uses only closed candles, no repainting.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        start: Initial acceleration factor
        inc: AF increment
        maximum: Maximum AF value
        
    Returns:
        Tuple of (SAR values dict with sar, trend, reversal_up, reversal_down)
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    
    n = len(df)
    
    sar = np.full(n, np.nan)
    ep = np.full(n, np.nan)
    af = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)  # 0=unknown, 1=UP, -1=DOWN
    
    if n == 0:
        return {"sar": sar, "trend": trend, "reversal_up": np.zeros(n, dtype=bool), "reversal_down": np.zeros(n, dtype=bool)}
    
    # Initialize first candle - need at least 2 candles to determine initial trend
    if n >= 2:
        # Initial trend based on first two closes
        if close[1] > close[0]:
            trend[0] = 1
            ep[0] = high[0]
            sar[0] = low[0]
        else:
            trend[0] = -1
            ep[0] = low[0]
            sar[0] = high[0]
        af[0] = start
    else:
        sar[0] = close[0]
        ep[0] = close[0]
        af[0] = start
        trend[0] = 1
    
    reversal_up = np.zeros(n, dtype=bool)
    reversal_down = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_ep = ep[i - 1]
        prev_af = af[i - 1]
        prev_trend = trend[i - 1]
        
        if prev_trend == 1:  # UP trend
            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            if i >= 2:
                current_sar = min(current_sar, low[i - 1], low[i - 2])
            else:
                current_sar = min(current_sar, low[i - 1])
            
            # Check for reversal
            if low[i] < current_sar:
                trend[i] = -1
                sar[i] = prev_ep
                ep[i] = low[i]
                af[i] = start
                reversal_down[i] = True  # Reversal from UP to DOWN
            else:
                trend[i] = 1
                sar[i] = current_sar
                
                if high[i] > prev_ep:
                    ep[i] = high[i]
                    af[i] = min(prev_af + inc, maximum)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
                    
        elif prev_trend == -1:  # DOWN trend
            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            if i >= 2:
                current_sar = max(current_sar, high[i - 1], high[i - 2])
            else:
                current_sar = max(current_sar, high[i - 1])
            
            # Check for reversal
            if high[i] > current_sar:
                trend[i] = 1
                sar[i] = prev_ep
                ep[i] = high[i]
                af[i] = start
                reversal_up[i] = True  # Reversal from DOWN to UP
            else:
                trend[i] = -1
                sar[i] = current_sar
                
                if low[i] < prev_ep:
                    ep[i] = low[i]
                    af[i] = min(prev_af + inc, maximum)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            # Unknown trend, initialize
            trend[i] = 1 if close[i] > close[i-1] else -1
            sar[i] = close[i]
            ep[i] = high[i] if trend[i] == 1 else low[i]
            af[i] = start
    
    return {
        "sar": sar,
        "trend": trend,
        "reversal_up": reversal_up,
        "reversal_down": reversal_down
    }
