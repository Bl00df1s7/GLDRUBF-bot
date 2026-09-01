"""
GLDRUBF Trading Strategy - Main Entry Point
SIGNAL ONLY MODE - No trading operations.

This script runs the complete GLDRUBF trading strategy analysis
and sends results to Telegram.

Usage:
    python -m src.main
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TARGET_TICKER, DONCHIAN_LEN, ATR_LEN, SAR_START, SAR_INC, SAR_MAX
from src.instruments import get_gldrubf_instrument
from src.market_data import load_candles, quotation_to_float
from src.indicators import calculate_atr, calculate_sar, prepare_indicators
from src.positions import find_gldrubf_position
from src.strategy import check_entry_signal, check_exit_conditions
from src.telegram_bot import send_telegram_message, format_status_message
from src.state_store import (
    load_state,
    save_state,
    check_candle_already_processed,
    check_position_changed,
    update_state_for_new_position,
    update_state_for_closed_position,
    activate_break_even,
    update_candle_processed,
    get_stored_levels,
    get_monitor_state,
    update_monitor_state,
)

try:
    from config.settings import (
        POSITION_MONITOR_ENABLED,
        MONITOR_TIMEFRAME,
        MONITOR_ONLY_WHEN_POSITION,
        UPDATE_ON_STATE_CHANGE_ONLY,
        SEND_RECOVERY_MESSAGE,
    )
except ImportError:
    # Fallback if settings not updated
    POSITION_MONITOR_ENABLED = False
    MONITOR_TIMEFRAME = "1H"
    MONITOR_ONLY_WHEN_POSITION = True
    UPDATE_ON_STATE_CHANGE_ONLY = True
    SEND_RECOVERY_MESSAGE = False


# Signal only mode - no trading
SIGNAL_ONLY = True

# Minimum data requirements
MIN_DATA_LENGTH = DONCHIAN_LEN + ATR_LEN + 20  # Donchian + ATR + SAR warmup + buffer


def is_candle_closed(candle_time: datetime, candle_duration: timedelta = timedelta(hours=4)) -> bool:
    """
    Check if a 4H candle is fully closed.
    
    Args:
        candle_time: Candle open time
        candle_duration: Duration of candle (4 hours)
        
    Returns:
        True if candle is closed
    """
    now_utc = datetime.now(timezone.utc)
    candle_close_time = candle_time + candle_duration
    return candle_close_time <= now_utc


def get_last_closed_candle(df):
    """
    Get the last fully closed 4H candle.
    
    Returns None if:
    - No closed candles found
    - Insufficient data
    - Candle data has NaN/None values
    """
    now_utc = datetime.now(timezone.utc)
    candle_duration = timedelta(hours=4)
    
    df = df.copy()
    df["candle_close_time"] = df["time"] + candle_duration
    
    # Filter only closed candles
    closed_candidates = df[df["candle_close_time"] <= now_utc]
    
    if closed_candidates.empty:
        return None, "WAIT_FOR_CLOSED_CANDLE"
    
    # Get last closed candle
    last_closed = closed_candidates.iloc[-1]
    
    # Validate OHLC data
    for col in ["open", "high", "low", "close"]:
        if pd.isna(last_closed[col]) or last_closed[col] is None:
            return None, f"DATA_INVALID: {col} is NaN"
    
    return last_closed, None


def check_data_sufficiency(df) -> tuple:
    """
    Check if we have enough data for reliable signals.
    
    Returns:
        Tuple of (is_sufficient, warning_message)
    """
    if len(df) < MIN_DATA_LENGTH:
        return False, f"INSUFFICIENT_DATA: have {len(df)}, need {MIN_DATA_LENGTH}"
    
    # Check for NaN in critical columns
    critical_cols = ["open", "high", "low", "close", "atr", "donchian_upper", "donchian_lower"]
    for col in critical_cols:
        if col in df.columns and df[col].iloc[-1] is None:
            if pd.isna(df[col].iloc[-1]):
                return False, f"DATA_WARNING: {col} is NaN for last candle"
    
    return True, None


def build_candle_data_for_state(last_closed: dict) -> dict:
    """Build candle data dictionary for state storage."""
    return {
        "timestamp": last_closed.get("time").isoformat() if last_closed.get("time") else None,
        "close": last_closed.get("close"),
    }


def main():
    """Main entry point for the strategy - SIGNAL ONLY mode."""
    
    # Validate required environment variables
    token = os.environ.get("T_SANDAPI")
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token:
        raise RuntimeError("Secret T_SANDAPI not found")
    if not bot_token:
        raise RuntimeError("Secret BOT_TOKEN not found")
    if not chat_id:
        raise RuntimeError("Secret TELEGRAM_CHAT_ID not found")
    
    print("=" * 70)
    print("GLDRUBF STRATEGY - SIGNAL ONLY MODE")
    print("=" * 70)
    
    # Load state
    state = load_state()
    print("\n📦 State loaded")
    
    # Get instrument details
    print("\n📊 Getting instrument info...")
    instrument = get_gldrubf_instrument(token)
    print(f"   Ticker: {instrument.ticker}")
    print(f"   UID: {instrument.uid}")
    
    # Load market data
    print("\n📈 Loading market data...")
    df_raw = load_candles(token, instrument.uid, candles_count=200)
    
    if df_raw.empty:
        error_msg = "Failed to load GLDRUBF candles"
        print(f"❌ {error_msg}")
        # Send error to Telegram
        send_telegram_message(
            bot_token, chat_id,
            f"❌ ERROR: {error_msg}\nРежим: SIGNAL ONLY, ручной."
        )
        return
    
    print(f"   Loaded {len(df_raw)} candles")
    
    # Check data sufficiency
    is_sufficient, data_warning = check_data_sufficiency(df_raw)
    if not is_sufficient:
        print(f"⚠️ {data_warning}")
        # Continue but mark as insufficient data
    
    # Calculate indicators
    print("\n📐 Calculating indicators...")
    df = prepare_indicators(df_raw)
    sar_result = calculate_sar(df, SAR_START, SAR_INC, SAR_MAX)
    df["sar"] = sar_result["sar"]
    df["sar_trend"] = sar_result["trend"]
    df["sar_reversal_up"] = sar_result["reversal_up"]
    df["sar_reversal_down"] = sar_result["reversal_down"]
    
    # Get last closed candle
    last_closed, candle_error = get_last_closed_candle(df)
    
    if candle_error == "WAIT_FOR_CLOSED_CANDLE":
        print("\n⏳ Candle not yet closed, skipping signal calculation")
        send_telegram_message(
            bot_token, chat_id,
            "⏳ Свеча еще не закрыта, расчет пропущен.\nРежим: SIGNAL ONLY, ручной."
        )
        return
    
    if candle_error:
        print(f"\n⚠️ {candle_error}")
        # Continue with warning
    
    if last_closed is None:
        send_telegram_message(
            bot_token, chat_id,
            f"⚠️ Ошибка данных свечи: {candle_error}\nРежим: SIGNAL ONLY, ручной."
        )
        return
    
    print(f"\n🕐 Last closed candle: {last_closed['time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Close: {last_closed['close']:.2f}")
    
    # Build candle data for state
    candle_data = build_candle_data_for_state({
        "time": last_closed["time"],
        "close": last_closed["close"],
    })
    
    # Check idempotency - skip if same candle processed without state change
    # But always process if position changed
    position_info = find_gldrubf_position(token, instrument)
    
    # Build position key
    if position_info:
        from src.state_store import _build_position_key
        current_position_key = _build_position_key({
            "direction": position_info["direction"],
            "entry_price": position_info.get("balance", 0),  # Will be refined below
            "account_id": position_info.get("account_id"),
            "instrument": "GLDRUBF",
        })
    else:
        current_position_key = None
    
    position_changed = check_position_changed(state, current_position_key)
    
    if not position_changed and check_candle_already_processed(state, candle_data):
        print("\n✅ Same candle already processed, no state change - skipping duplicate message")
        return
    
    print(f"\n🔍 Searching for GLDRUBF position...")
    print(f"   Position key: {current_position_key}")
    
    # Get position state
    if position_info is None:
        position_state = {
            "direction": "NONE",
            "quantity": 0.0,
            "account_id": None,
            "account_name": None,
            "entry_price": np.nan,
            "entry_atr": float(last_closed["atr"]) if not pd.isna(last_closed["atr"]) else np.nan,
            "sl_price": np.nan,
            "tp_price": np.nan,
            "be_trigger": np.nan,
            "sar_price": float(last_closed["sar"]) if not pd.isna(last_closed["sar"]) else np.nan,
            "instrument": "GLDRUBF",
        }
        print("   No position found")
    else:
        # Get average entry price from portfolio
        from t_tech.invest import Client
        account_id = position_info["account_id"]
        figi = position_info["position"].figi
        
        entry_price = np.nan
        
        try:
            with Client(token) as client:
                portfolio = client.operations.get_portfolio(account_id=account_id)
                
                for portfolio_pos in portfolio.positions:
                    if portfolio_pos.figi == figi:
                        entry_price = quotation_to_float(portfolio_pos.average_position_price)
                        break
        except Exception as e:
            print(f"⚠️ Could not get entry price: {e}")
        
        direction = position_info["direction"]
        
        # Get stored levels or calculate initial
        stored_levels = get_stored_levels(state)
        
        if stored_levels["initial_sl"] is None or position_changed:
            # Calculate initial levels
            atr = float(last_closed["atr"]) if not pd.isna(last_closed["atr"]) else 0
            
            if direction == "LONG":
                initial_sl = entry_price - atr * 3.0 if not np.isnan(entry_price) and atr > 0 else np.nan
                tp = entry_price * 1.07 if not np.isnan(entry_price) else np.nan
                be_trigger = entry_price * 1.02 if not np.isnan(entry_price) else np.nan
            else:  # SHORT
                initial_sl = entry_price + atr * 3.0 if not np.isnan(entry_price) and atr > 0 else np.nan
                tp = entry_price * 0.93 if not np.isnan(entry_price) else np.nan
                be_trigger = entry_price * 0.98 if not np.isnan(entry_price) else np.nan
            
            # Update state for new position
            state = update_state_for_new_position(
                state,
                {
                    "direction": direction,
                    "entry_price": float(entry_price) if not np.isnan(entry_price) else None,
                    "account_id": position_info.get("account_id"),
                    "instrument": "GLDRUBF",
                },
                float(initial_sl) if not np.isnan(initial_sl) else None,
                float(tp) if not np.isnan(tp) else None,
                float(be_trigger) if not np.isnan(be_trigger) else None,
            )
            stored_levels = get_stored_levels(state)
        
        position_state = {
            "direction": direction,
            "quantity": float(position_info["balance"]),
            "account_id": position_info["account_id"],
            "account_name": position_info.get("account_name"),
            "entry_price": float(entry_price) if not np.isnan(entry_price) else None,
            "entry_atr": atr,
            "sl_price": stored_levels["recommended_sl"],
            "tp_price": stored_levels["tp"],
            "be_trigger": stored_levels["be_trigger"],
            "sar_price": float(last_closed["sar"]) if not pd.isna(last_closed["sar"]) else np.nan,
            "instrument": "GLDRUBF",
        }
    
    # Check for entry signal (always calculated)
    entry_signal = check_entry_signal(last_closed)
    print(f"\n🎯 Entry signal: {entry_signal}")
    
    # Check for exit conditions (only if position exists)
    exit_signal = None
    warnings = []
    
    if position_state["direction"] in ("LONG", "SHORT"):
        stored_levels = get_stored_levels(state)
        exit_result = check_exit_conditions(position_state, last_closed, stored_levels)
        exit_signal, new_be_activated, new_recommended_sl, exit_warnings = exit_result
        warnings.extend(exit_warnings)
        
        # Update state if BE activated
        if new_be_activated and not stored_levels.get("be_activated", False):
            state = activate_break_even(state, new_recommended_sl)
        
        print(f"🚪 Exit signal: {exit_signal}")
    
    # Handle position closed scenario
    if state.get("position_key") and position_state["direction"] == "NONE":
        # Position was closed externally
        state = update_state_for_closed_position(state)
        print("🔄 Position closed externally, state reset")
    
    # Determine action (informational only)
    if position_state["direction"] == "NONE":
        if entry_signal == "LONG":
            action = "SIGNAL_OPEN_LONG"
        elif entry_signal == "SHORT":
            action = "SIGNAL_OPEN_SHORT"
        else:
            action = "WAIT"
    else:
        if exit_signal == "EXIT_SL":
            action = "SIGNAL_EXIT_SL"
        elif exit_signal == "EXIT_TP":
            action = "SIGNAL_EXIT_TP"
        elif exit_signal == "EXIT_SAR":
            action = "SIGNAL_EXIT_SAR"
        elif exit_signal == "EXIT_BE_STOP":
            action = "SIGNAL_EXIT_BE_STOP"
        elif exit_signal == "BE_TRIGGERED":
            action = "BE_TRIGGERED"
        else:
            action = "HOLD_POSITION"
    
    # Check for opposite entry signal warning
    if position_state["direction"] == "LONG" and entry_signal == "SHORT":
        warnings.append("Встречный SHORT-сигнал не является выходом по текущим правилам.")
    elif position_state["direction"] == "SHORT" and entry_signal == "LONG":
        warnings.append("Встречный LONG-сигнал не является выходом по текущим правилам.")
    
    # Update state
    state = update_candle_processed(state, candle_data, action, exit_signal)
    
    # Position monitor integration
    if POSITION_MONITOR_ENABLED and position_state["direction"] in ("LONG", "SHORT"):
        try:
            from src.position_monitor import (
                calculate_position_health,
                format_position_monitor_message,
                should_send_alert,
            )
            
            # Get monitor timeframe data (1H by default)
            df_monitor = load_candles(token, instrument.uid, candles_count=100, timeframe=MONITOR_TIMEFRAME)
            
            if not df_monitor.empty:
                # Calculate indicators for monitor timeframe
                df_monitor = prepare_indicators(df_monitor)
                
                # Get entry time (use signal candle time as fallback)
                entry_time = last_closed.get("time")  # Fallback to 4H signal candle
                
                # Calculate position health
                health = calculate_position_health(
                    position_state=position_state,
                    df_monitor=df_monitor,
                    df_4h=df,
                    stored_levels=stored_levels,
                    entry_time=entry_time,
                )
                
                # Get previous monitor state
                monitor_state = get_monitor_state(state)
                
                # Check if alert should be sent
                if should_send_alert(
                    health=health,
                    last_alert_level=monitor_state["last_alert_level"],
                    last_alert_reasons=monitor_state["last_alert_reasons"],
                    send_recovery=SEND_RECOVERY_MESSAGE,
                ):
                    # Format and send monitor alert
                    monitor_message = format_position_monitor_message(health)
                    
                    # Prepend header with instrument name
                    full_message = f"🟠 GLDRUBF · Мониторинг позиции\n{monitor_message}"
                    
                    send_telegram_message(bot_token, chat_id, full_message)
                    print("✅ Position monitor alert sent")
                    
                    # Update monitor state
                    state = update_monitor_state(
                        state,
                        health.alert_level,
                        health.alert_reasons,
                        health.last_closed_candle_time,
                    )
                else:
                    print("🔇 Position monitor: no state change, skipping alert")
            else:
                print("⚠️ No monitor timeframe data available")
        except Exception as e:
            print(f"⚠️ Position monitor error: {e}")
    
    save_state(state)
    print("\n💾 State saved")
    
    # Format and send Telegram message (main strategy status)
    print("\n📱 Sending status to Telegram...")
    
    # Use debug mode from settings
    from config.settings import TELEGRAM_DEBUG_MODE
    
    message = format_status_message(
        last_closed=last_closed,
        df=df,
        position_state=position_state,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        action=action,
        warnings=warnings,
        debug_mode=TELEGRAM_DEBUG_MODE,
    )
    
    send_telegram_message(bot_token, chat_id, message)
    print("✅ Status sent successfully")
    
    # Print final status
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print(f"Action: {action}")
    print(f"Position: {position_state['direction']}")
    if position_state["direction"] in ("LONG", "SHORT"):
        print(f"Quantity: {position_state['quantity']}")
        print(f"Entry: {position_state['entry_price']}")
        print(f"SL: {position_state['sl_price']}")
        print(f"TP: {position_state['tp_price']}")
    
    print("\n✅ Strategy execution completed (SIGNAL ONLY)")



if __name__ == "__main__":
    import pandas as pd
    main()
