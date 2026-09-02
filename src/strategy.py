"""
Strategy logic - entry and exit signals.
SIGNAL ONLY mode - no trading operations.
"""

import numpy as np

from config.settings import SL_ATR, TP_PCT, BE_PCT

# Strategy parameters
BE_AS_EXIT = False  # By default, BE is not an exit, just SL move
OPPOSITE_ENTRY_CLOSES_POSITION = False
OPPOSITE_ENTRY_REVERSES_POSITION = False
CONSERVATIVE_SL_TP_SAME_BAR = True


def check_entry_signal(last_closed: dict) -> str:
    """
    Check for entry signal based on Donchian breakout.
    Always calculated regardless of position status.
    
    Args:
        last_closed: Last closed candle data with indicators
        
    Returns:
        "LONG", "SHORT", or None
    """
    long_signal = last_closed.get("long_signal", False)
    short_signal = last_closed.get("short_signal", False)
    
    # Check for anomaly (both signals true)
    if long_signal and short_signal:
        print("⚠️ Donchian anomaly: both long and short signals")
        return None

    if long_signal:
        return "LONG"
    elif short_signal:
        return "SHORT"
    else:
        return None


def should_close_by_monitor(health) -> tuple:
    """Return whether monitor recommendations require an emergency close."""
    alert_level = getattr(health, "alert_level", "NORMAL")
    distance_to_sl = getattr(health, "distance_to_sl_points", None)
    pressure_atr = getattr(health, "pressure_atr_mult", None)
    adverse_speed_atr = getattr(health, "adverse_speed_atr_mult", None)

    # PositionHealth currently exposes ATR-normalized pressure and speed,
    # while SL distance is compared against the monitor's 0.1% fallback ATR.
    entry_price = getattr(health, "entry_price", None)
    sl_threshold = abs(entry_price) * 0.0005 if entry_price else None

    if alert_level == "CRITICAL" and sl_threshold is not None and distance_to_sl is not None:
        if distance_to_sl < sl_threshold:
            return True, "MONITOR_CRITICAL"
    if alert_level == "STRUCTURE_BREAK" and adverse_speed_atr is not None:
        if adverse_speed_atr > 1.5:
            return True, "MONITOR_STRUCTURE"
    if alert_level == "FAST_ADVERSE" and pressure_atr is not None:
        if pressure_atr > 2.0:
            return True, "MONITOR_FAST_ADVERSE"
    return False, ""
def check_exit_conditions(
    position_state: dict,
    last_closed: dict,
    stored_levels: dict
) -> tuple:
    """
    Check for exit conditions when position exists.
    Uses OHLC of closed candle, not current price.
    
    Args:
        position_state: Position state dictionary
        last_closed: Last closed candle data (OHLC)
        stored_levels: Levels from state store (recommended_sl, be_trigger, be_activated, etc.)
        
    Returns:
        Tuple of (exit_signal, be_activated_new, recommended_sl_new, warnings)
    """
    warnings = []
    
    direction = position_state["direction"]
    entry_price = position_state.get("entry_price")
    
    if np.isnan(entry_price) if isinstance(entry_price, float) else not entry_price:
        return None, stored_levels.get("be_activated", False), stored_levels.get("recommended_sl"), warnings
    
    # Use stored levels if available
    recommended_sl = stored_levels.get("recommended_sl") or position_state.get("sl_price")
    tp_price = stored_levels.get("tp") or position_state.get("tp_price")
    be_trigger = stored_levels.get("be_trigger") or position_state.get("be_trigger")
    be_activated = stored_levels.get("be_activated", False)
    
    if recommended_sl is None or tp_price is None or be_trigger is None:
        warnings.append("Levels unavailable")
        return None, be_activated, recommended_sl, warnings
    
    # Get candle OHLC
    candle_high = last_closed.get("high")
    candle_low = last_closed.get("low")
    candle_close = last_closed.get("close")
    
    # Get SAR info
    sar_trend = last_closed.get("sar_trend", 1)
    sar_reversal_up = last_closed.get("sar_reversal_up", False)
    sar_reversal_down = last_closed.get("sar_reversal_down", False)
    
    exit_signal = None
    new_be_activated = be_activated
    new_recommended_sl = recommended_sl
    
    if direction == "LONG":
        # Check SL first (highest priority)
        if candle_low <= recommended_sl:
            exit_signal = "EXIT_SL"
        
        # Check TP
        elif candle_high >= tp_price:
            exit_signal = "EXIT_TP"
        
        # Check SAR reversal
        elif sar_reversal_down:
            exit_signal = "EXIT_SAR"
        
        # Check BE trigger (only if not yet activated)
        elif not be_activated and candle_close >= be_trigger:
            if BE_AS_EXIT:
                exit_signal = "EXIT_BE_STOP"
            else:
                exit_signal = "BE_TRIGGERED"
                new_be_activated = True
                # Move recommended SL to entry (or entry + offset)
                new_recommended_sl = entry_price
        
        # Check if BE stop is hit (after BE activated)
        elif be_activated and candle_low <= entry_price:
            exit_signal = "EXIT_BE_STOP"
    
    else:  # SHORT
        # Check SL first (highest priority)
        if candle_high >= recommended_sl:
            exit_signal = "EXIT_SL"
        
        # Check TP
        elif candle_low <= tp_price:
            exit_signal = "EXIT_TP"
        
        # Check SAR reversal
        elif sar_reversal_up:
            exit_signal = "EXIT_SAR"
        
        # Check BE trigger (only if not yet activated)
        elif not be_activated and candle_close <= be_trigger:
            if BE_AS_EXIT:
                exit_signal = "EXIT_BE_STOP"
            else:
                exit_signal = "BE_TRIGGERED"
                new_be_activated = True
                # Move recommended SL to entry (or entry - offset)
                new_recommended_sl = entry_price
        
        # Check if BE stop is hit (after BE activated)
        elif be_activated and candle_high >= entry_price:
            exit_signal = "EXIT_BE_STOP"
    
    return exit_signal, new_be_activated, new_recommended_sl, warnings
