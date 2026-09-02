"""
GLDRUBF Trading Strategy - Main Entry Point
Supports AUTO_TRADING and SIGNAL_ONLY modes.

This script runs the complete GLDRUBF trading strategy analysis
and optionally executes trades automatically.

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
from src.strategy import check_entry_signal, check_exit_conditions, should_close_by_monitor
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
        AUTO_TRADING_ENABLED,
        TELEGRAM_ENABLED,
    )
except ImportError:
    POSITION_MONITOR_ENABLED = False
    MONITOR_TIMEFRAME = "1H"
    MONITOR_ONLY_WHEN_POSITION = True
    UPDATE_ON_STATE_CHANGE_ONLY = True
    SEND_RECOVERY_MESSAGE = False
    AUTO_TRADING_ENABLED = False
    TELEGRAM_ENABLED = False


SIGNAL_ONLY = not AUTO_TRADING_ENABLED
MIN_DATA_LENGTH = DONCHIAN_LEN + ATR_LEN + 20


def is_candle_closed(candle_time: datetime, candle_duration: timedelta = timedelta(hours=4)) -> bool:
    now_utc = datetime.now(timezone.utc)
    candle_close_time = candle_time + candle_duration
    return candle_close_time <= now_utc


def get_last_closed_candle(df):
    now_utc = datetime.now(timezone.utc)
    candle_duration = timedelta(hours=4)

    df = df.copy()
    df["candle_close_time"] = df["time"] + candle_duration

    closed_candidates = df[df["candle_close_time"] <= now_utc]

    if closed_candidates.empty:
        return None, "WAIT_FOR_CLOSED_CANDLE"

    last_closed = closed_candidates.iloc[-1]

    for col in ["open", "high", "low", "close"]:
        if pd.isna(last_closed[col]) or last_closed[col] is None:
            return None, f"DATA_INVALID: {col} is NaN"

    return last_closed, None


def check_data_sufficiency(df) -> tuple:
    if len(df) < MIN_DATA_LENGTH:
        return False, f"INSUFFICIENT_DATA: have {len(df)}, need {MIN_DATA_LENGTH}"

    critical_cols = ["open", "high", "low", "close", "atr", "donchian_upper", "donchian_lower"]
    for col in critical_cols:
        if col in df.columns and df[col].iloc[-1] is None:
            if pd.isna(df[col].iloc[-1]):
                return False, f"DATA_WARNING: {col} is NaN for last candle"

    return True, None


def build_candle_data_for_state(last_closed: dict) -> dict:
    return {
        "timestamp": last_closed.get("time").isoformat() if last_closed.get("time") else None,
        "close": last_closed.get("close"),
    }


def _send_telegram_safe(bot_token, chat_id, message):
    """Send Telegram message only if enabled and credentials available."""
    if not TELEGRAM_ENABLED:
        return
    if not bot_token or not chat_id:
        return
    try:
        send_telegram_message(bot_token, chat_id, message)
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")


def _run():
    """Main entry point for the strategy."""

    token = os.environ.get("T_SANDAPI")
    if not token:
        raise RuntimeError("Secret T_SANDAPI not found")

    from src.observability import configure_logging
    from src.preflight import preflight_check

    logger, correlation_id = configure_logging()
    preflight_ok, preflight_message = preflight_check(token)
    logger.info(preflight_message, extra={"correlation_id": correlation_id})
    if not preflight_ok:
        raise RuntimeError(preflight_message)

    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    mode_str = "AUTO TRADING" if AUTO_TRADING_ENABLED else "SIGNAL ONLY"

    print("=" * 70)
    print(f"GLDRUBF STRATEGY - {mode_str} MODE")
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
        _send_telegram_safe(
            bot_token, chat_id,
            f"❌ ERROR: {error_msg}\nРежим: {mode_str}."
        )
        return

    print(f"   Loaded {len(df_raw)} candles")

    # Check data sufficiency
    is_sufficient, data_warning = check_data_sufficiency(df_raw)
    if not is_sufficient:
        print(f"⚠️ {data_warning}")

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
        _send_telegram_safe(
            bot_token, chat_id,
            f"⏳ Свеча еще не закрыта, расчет пропущен.\nРежим: {mode_str}."
        )
        return

    if candle_error:
        print(f"\n⚠️ {candle_error}")

    if last_closed is None:
        _send_telegram_safe(
            bot_token, chat_id,
            f"⚠️ Ошибка данных свечи: {candle_error}\nРежим: {mode_str}."
        )
        return

    print(f"\n🕐 Last closed candle: {last_closed['time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Close: {last_closed['close']:.2f}")

    # Build candle data for state
    candle_data = build_candle_data_for_state({
        "time": last_closed["time"],
        "close": last_closed["close"],
    })

    # Check idempotency
    position_info = find_gldrubf_position(token, instrument)

    if position_info:
        from src.state_store import _build_position_key
        current_position_key = _build_position_key({
            "direction": position_info["direction"],
            "entry_price": position_info.get("balance", 0),
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
        from src.client_factory import get_client
        account_id = position_info["account_id"]
        figi = position_info["position"].figi

        entry_price = np.nan

        try:
            with get_client(token) as client:
                portfolio = client.operations.get_portfolio(account_id=account_id)
                for portfolio_pos in portfolio.positions:
                    if portfolio_pos.figi == figi:
                        entry_price = quotation_to_float(portfolio_pos.average_position_price)
                        break
        except Exception as e:
            print(f"⚠️ Could not get entry price: {e}")

        direction = position_info["direction"]
        stored_levels = get_stored_levels(state)

        if stored_levels["initial_sl"] is None or position_changed:
            atr = float(last_closed["atr"]) if not pd.isna(last_closed["atr"]) else 0

            if direction == "LONG":
                initial_sl = entry_price - atr * 3.0 if not np.isnan(entry_price) and atr > 0 else np.nan
                tp = entry_price * 1.07 if not np.isnan(entry_price) else np.nan
                be_trigger = entry_price * 1.02 if not np.isnan(entry_price) else np.nan
            else:
                initial_sl = entry_price + atr * 3.0 if not np.isnan(entry_price) and atr > 0 else np.nan
                tp = entry_price * 0.93 if not np.isnan(entry_price) else np.nan
                be_trigger = entry_price * 0.98 if not np.isnan(entry_price) else np.nan

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

    # Check for entry signal
    entry_signal = check_entry_signal(last_closed)
    print(f"\n🎯 Entry signal: {entry_signal}")

    # Check for exit conditions
    exit_signal = None
    warnings = []

    if position_state["direction"] in ("LONG", "SHORT"):
        stored_levels = get_stored_levels(state)
        exit_result = check_exit_conditions(position_state, last_closed, stored_levels)
        exit_signal, new_be_activated, new_recommended_sl, exit_warnings = exit_result
        warnings.extend(exit_warnings)

        if new_be_activated and not stored_levels.get("be_activated", False):
            state = activate_break_even(state, new_recommended_sl)

            if AUTO_TRADING_ENABLED and position_state.get("account_id"):
                try:
                    from src.stop_orders import replace_protection_orders

                    be_stop_ids = replace_protection_orders(
                        token=token,
                        account_id=position_state["account_id"],
                        instrument_uid=instrument.uid,
                        quantity_lots=int(abs(position_state["quantity"])),
                        position_direction=position_state["direction"],
                        old_stop_order_ids=stored_levels.get("stop_order_ids", {}).values(),
                        stop_price=float(new_recommended_sl),
                        take_profit_price=float(stored_levels["tp"]),
                        min_price_increment=instrument.min_price_increment,
                    )
                    state["stop_order_ids"] = be_stop_ids
                    state["stop_order_id"] = be_stop_ids.get("stop_loss")
                    print("✅ Server-side stop moved to break-even")
                except Exception as be_error:
                    warnings.append(f"BE server stop update failed: {be_error}")
                    print(f"⚠️ BE server stop update failed: {be_error}")

        print(f"🚪 Exit signal: {exit_signal}")

    # Handle position closed scenario
    if state.get("position_key") and position_state["direction"] == "NONE":
        state = update_state_for_closed_position(state)
        print("🔄 Position closed externally, state reset")

    # Determine action
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

    # Keep candle bookkeeping reversible until an order is confirmed.
    action_state_snapshot = {
        key: state.get(key)
        for key in (
            "last_processed_candle_timestamp",
            "last_processed_candle_hash",
            "last_action",
            "last_exit_signal",
            "last_run_timestamp",
        )
    }

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

            df_monitor = load_candles(token, instrument.uid, candles_count=100, timeframe=MONITOR_TIMEFRAME)

            if not df_monitor.empty:
                df_monitor = prepare_indicators(df_monitor)
                entry_time = last_closed.get("time")

                health = calculate_position_health(
                    position_state=position_state,
                    df_monitor=df_monitor,
                    df_4h=df,
                    stored_levels=stored_levels,
                    entry_time=entry_time,
                )

                monitor_close, monitor_reason = should_close_by_monitor(health)
                if monitor_close:
                    exit_signal = monitor_reason
                    action = f"SIGNAL_EXIT_{monitor_reason}"
                    print(f"🚨 Monitor close recommendation: {monitor_reason}")

                monitor_state = get_monitor_state(state)

                if should_send_alert(
                    health=health,
                    last_alert_level=monitor_state["last_alert_level"],
                    last_alert_reasons=monitor_state["last_alert_reasons"],
                    send_recovery=SEND_RECOVERY_MESSAGE,
                ):
                    monitor_message = format_position_monitor_message(health)
                    full_message = f"🟠 GLDRUBF · Мониторинг позиции\n{monitor_message}"
                    _send_telegram_safe(bot_token, chat_id, full_message)
                    print("✅ Position monitor alert sent")

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

    # Re-record the final action when monitoring overrides the strategy action.
    state = update_candle_processed(state, candle_data, action, exit_signal)

    # === AUTO TRADING EXECUTION ===
    trade_result = None
    if AUTO_TRADING_ENABLED and action not in ("WAIT", "HOLD_POSITION", "BE_TRIGGERED"):
        print("\n🤖 AUTO TRADING: Executing signal...")
        try:
            from src.auto_trader import execute_signal
            from src.market_data import get_current_price

            current_price = get_current_price(token, instrument.uid)
            print(f"   Current price: {current_price:.2f}")

            account_id = position_state.get("account_id")

            if account_id is None:
                from src.client_factory import get_client
                with get_client(token) as client:
                    accounts = client.users.get_accounts().accounts
                    for acc in accounts:
                        if acc.status.name == "ACCOUNT_STATUS_OPEN":
                            account_id = acc.id
                            break

            if account_id:
                position_before_trade = dict(position_state)
                risk_blocked = False
                if action in ("SIGNAL_OPEN_LONG", "SIGNAL_OPEN_SHORT"):
                    from config.settings import MAX_DAILY_LOSS_PCT
                    from src.risk_manager import check_circuit_breaker

                    risk = check_circuit_breaker(
                        token=token,
                        account_id=account_id,
                        figi=instrument.figi,
                        state=state,
                        max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
                    )
                    if not risk.allowed:
                        risk_blocked = True
                        trade_result = (
                            f"❌ Trading halted: realized PnL={risk.realized_pnl:.2f}, "
                            f"loss={risk.loss_pct:.2%}"
                        )
                        print(f"   {trade_result}")
                        _send_telegram_safe(bot_token, chat_id, trade_result)

                if risk_blocked:
                    save_state(state)
                    return

                stop_loss = None
                take_profit = None
                if action in ("SIGNAL_OPEN_LONG", "SIGNAL_OPEN_SHORT"):
                    from config.settings import SL_ATR, TP_PCT

                    atr = float(last_closed["atr"])
                    if action == "SIGNAL_OPEN_LONG":
                        stop_loss = current_price - atr * SL_ATR
                        take_profit = current_price * (1 + TP_PCT)
                    else:
                        stop_loss = current_price + atr * SL_ATR
                        take_profit = current_price * (1 - TP_PCT)

                success, msg = execute_signal(
                    token=token,
                    account_id=account_id,
                    instrument_uid=instrument.uid,
                    instrument=instrument,
                    action=action,
                    position_state=position_state,
                    current_price=current_price,
                    entry_signal=entry_signal,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    state=state,
                )
                trade_result = f"{'✅' if success else '❌'} {msg}"
                print(f"   {trade_result}")
                if success and action in ("SIGNAL_OPEN_LONG", "SIGNAL_OPEN_SHORT"):
                    state["entry_time"] = datetime.now(timezone.utc).isoformat()
                    state["entry_price"] = current_price
                    state["entry_lots"] = int(state.get("entry_lots", 0) or 0)
                elif success and action.startswith("SIGNAL_EXIT_"):
                    from src.trade_logger import record_trade, calculate_round_trip_commission

                    entry_price = position_before_trade.get("entry_price")
                    if entry_price:
                        entry_time = datetime.fromisoformat(
                            state.get("entry_time", datetime.now(timezone.utc).isoformat())
                        )
                        lots = int(abs(position_before_trade.get("quantity", 0)))
                        pnl_points = (
                            current_price - float(entry_price)
                            if position_before_trade.get("direction") == "LONG"
                            else float(entry_price) - current_price
                        )
                        lot_size = int(getattr(instrument, "lot", 1) or 1)
                        commission = calculate_round_trip_commission(
                            float(entry_price), current_price, lots, lot_size
                        )
                        record_trade(
                            entry_time=entry_time,
                            exit_time=datetime.now(timezone.utc),
                            direction=position_before_trade["direction"],
                            entry_price=float(entry_price),
                            exit_price=current_price,
                            lots=lots,
                            pnl_gross=pnl_points * lots * lot_size,
                            commission=commission,
                            exit_reason=action.replace("SIGNAL_EXIT_", ""),
                            figi=instrument.figi,
                        )
                if not success:
                    for key, value in action_state_snapshot.items():
                        state[key] = value
            else:
                trade_result = "❌ No active account found"
                print(f"   {trade_result}")
                for key, value in action_state_snapshot.items():
                    state[key] = value
        except Exception as e:
            trade_result = f"❌ Auto-trading error: {e}"
            print(f"   {trade_result}")
            for key, value in action_state_snapshot.items():
                state[key] = value
    elif AUTO_TRADING_ENABLED:
        print(f"\n🤖 AUTO TRADING: No action needed ({action})")

    # === TELEGRAM NOTIFICATION (optional) ===
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

    if trade_result:
        message += f"\n\n🤖 Авто-трейдинг: {trade_result}"

    if TELEGRAM_ENABLED:
        print("\n📱 Sending status to Telegram...")
        _send_telegram_safe(bot_token, chat_id, message)
        print("✅ Status sent successfully")
    else:
        print("\n📱 Telegram disabled, skipping notification")

    save_state(state)
    print("\n💾 State saved")

    # Print final status
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    print(f"Mode:     {mode_str}")
    print(f"Action:   {action}")
    print(f"Position: {position_state['direction']}")
    if position_state["direction"] in ("LONG", "SHORT"):
        print(f"Quantity: {position_state['quantity']}")
        print(f"Entry:    {position_state['entry_price']}")
        print(f"SL:       {position_state['sl_price']}")
        print(f"TP:       {position_state['tp_price']}")
    if trade_result:
        print(f"Trade:    {trade_result}")

    logger.info("run_completed", extra={"correlation_id": correlation_id})
    print(f"\n✅ Strategy execution completed ({mode_str})")


def main():
    """Acquire the process lock and execute one bot run."""
    from src.lock import process_lock

    with process_lock(os.environ.get("LOCK_FILE", "/tmp/gldrubf.lock")):
        _run()


if __name__ == "__main__":
    main()
