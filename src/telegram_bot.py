"""
Telegram bot messaging functions.
"""

import requests
import numpy as np
import pandas as pd

from config.settings import TELEGRAM_DEBUG_MODE


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> dict:
    """
    Send message to Telegram chat.
    
    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID to send message to
        message: Message text (supports HTML)
        
    Returns:
        Telegram API response
        
    Raises:
        RuntimeError: If message sending fails
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    
    return data


def fmt_price(value) -> str:
    """Format price with space as thousand separator."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "н/д"
    return f"{float(value):,.2f}".replace(",", " ")


def format_status_message(
    last_closed: dict,
    df: pd.DataFrame,
    position_state: dict,
    entry_signal: str,
    exit_signal: str,
    action: str,
    warnings: list = None,
    debug_mode: bool = False,
) -> str:
    """
    Format strategy status message for Telegram - HUMAN READABLE FORMAT.
    Technical only, no macro, no investment recommendations.
    
    Args:
        last_closed: Last closed candle data with indicators
        df: Full DataFrame with market data
        position_state: Position state dictionary
        entry_signal: Entry signal ("LONG", "SHORT", or None)
        exit_signal: Exit signal string or None
        action: Action to take
        warnings: List of warning strings
        debug_mode: Show technical debug info
        
    Returns:
        Formatted message string
    """
    from datetime import timezone, timedelta
    
    warnings = warnings or []
    
    # Moscow timezone (UTC+3)
    msk_tz = timezone(timedelta(hours=3))
    
    # Convert candle time to Moscow time for display
    candle_time = last_closed.get("time")
    candle_end_time = last_closed.get("candle_close_time")
    if candle_time:
        # Handle pandas Timestamp (may be tz-naive)
        if hasattr(candle_time, 'tz') and candle_time.tz is None:
            candle_time = candle_time.tz_localize('UTC')
        candle_time_msk = candle_time.astimezone(msk_tz)
        if candle_end_time is None:
            candle_end_time = candle_time + timedelta(hours=4)
        if hasattr(candle_end_time, 'tz') and candle_end_time.tz is None:
            candle_end_time = candle_end_time.tz_localize('UTC')
        candle_end_time_msk = candle_end_time.astimezone(msk_tz)
        candle_time_str = (
            f"{candle_time_msk.strftime('%d.%m.%Y %H:%M')} - "
            f"{candle_end_time_msk.strftime('%H:%M')} МСК"
        )
    else:
        candle_time_str = "н/д"
    
    # Get indicator values
    donchian_upper = last_closed.get("donchian_upper")
    donchian_lower = last_closed.get("donchian_lower")
    atr = last_closed.get("atr")
    sar_value = last_closed.get("sar")
    sar_trend_int = last_closed.get("sar_trend", 1)
    sar_reversal_up = last_closed.get("sar_reversal_up", False)
    sar_reversal_down = last_closed.get("sar_reversal_down", False)
    
    close_price = last_closed.get("close")
    
    # SAR description
    if sar_trend_int == 1:
        sar_trend_str = "вверх"
    elif sar_trend_int == -1:
        sar_trend_str = "вниз"
    else:
        sar_trend_str = "н/д"
    
    # Check if SAR supports position
    direction = position_state.get("direction", "NONE")
    sar_supports = ""
    if direction == "LONG" and sar_trend_int == 1:
        sar_supports = ", поддерживает позицию"
    elif direction == "SHORT" and sar_trend_int == -1:
        sar_supports = ", поддерживает позицию"
    
    # Check if SAR reversed
    sar_reversed = ""
    if direction == "LONG" and sar_reversal_down:
        sar_reversed = ", развернулся вниз"
    elif direction == "SHORT" and sar_reversal_up:
        sar_reversed = ", развернулся вверх"
    
    # Donchian channel description
    channel_text = ""
    price_position = ""
    
    if donchian_upper is not None and donchian_lower is not None and close_price is not None:
        channel_text = f"Канал: {fmt_price(donchian_lower)} — {fmt_price(donchian_upper)}"
        
        if close_price > donchian_upper:
            price_position = "Пробой верхней границы канала"
        elif close_price < donchian_lower:
            price_position = "Пробой нижней границы канала"
        else:
            price_position = "Цена внутри канала"
    
    # ATR text
    atr_text = f"ATR: {fmt_price(atr)}" if atr is not None and not (isinstance(atr, float) and np.isnan(atr)) else "ATR: н/д"
    
    # Position block
    position_icon = "⚪"
    position_direction = "нет"
    position_block_lines = []
    
    if direction == "NONE":
        position_block_lines.append("Позиция: нет")
    else:
        position_icon = "🟢" if direction == "LONG" else "🔴"
        position_direction = direction
        entry_price = position_state.get("entry_price")
        
        # P&L calculation
        pnl_points = None
        pnl_pct = None
        if entry_price is not None and not (isinstance(entry_price, float) and np.isnan(entry_price)) and close_price is not None:
            if direction == "LONG":
                pnl_points = close_price - entry_price
                pnl_pct = (pnl_points / entry_price) * 100
            else:  # SHORT
                pnl_points = entry_price - close_price
                pnl_pct = (pnl_points / entry_price) * 100
        
        position_block_lines.append(f"Позиция: {position_icon} {direction}")
        position_block_lines.append(f"Вход: {fmt_price(entry_price)}")
        position_block_lines.append(f"Текущая цена свечи: {fmt_price(close_price)}")
        
        if pnl_points is not None:
            pnl_sign = "+" if pnl_points >= 0 else ""
            pct_sign = "+" if pnl_pct >= 0 else ""
            position_block_lines.append(f"P&L от входа: {pnl_sign}{pnl_points:.2f} пунктов / {pct_sign}{pnl_pct:.2f}%")
    
    # Levels block
    levels_block_lines = []
    sl_price = position_state.get("sl_price")
    tp_price = position_state.get("tp_price")
    be_trigger = position_state.get("be_trigger")
    be_activated = position_state.get("be_activated", False)
    
    if direction != "NONE" and close_price is not None:
        # SL distance
        if sl_price is not None and not (isinstance(sl_price, float) and np.isnan(sl_price)):
            if direction == "LONG":
                dist_to_sl = close_price - sl_price
            else:  # SHORT
                dist_to_sl = sl_price - close_price
            
            if dist_to_sl <= 0:
                sl_dist_text = "· задет по закрытой свече"
            else:
                sl_dist_text = f"· до стопа {dist_to_sl:.2f}"
            
            levels_block_lines.append(f"SL: {fmt_price(sl_price)} {sl_dist_text}")
        
        # TP distance
        if tp_price is not None and not (isinstance(tp_price, float) and np.isnan(tp_price)):
            if direction == "LONG":
                dist_to_tp = tp_price - close_price
            else:  # SHORT
                dist_to_tp = close_price - tp_price
            
            if dist_to_tp <= 0:
                tp_dist_text = "· достигнут по закрытой свече"
            else:
                tp_dist_text = f"· до цели {dist_to_tp:.2f}"
            
            levels_block_lines.append(f"TP: {fmt_price(tp_price)} {tp_dist_text}")
        
        # BE status
        if be_trigger is not None and not (isinstance(be_trigger, float) and np.isnan(be_trigger)):
            if be_activated:
                levels_block_lines.append("BE: активирован")
            else:
                if direction == "LONG":
                    dist_to_be = be_trigger - close_price
                else:  # SHORT
                    dist_to_be = close_price - be_trigger
                
                if dist_to_be <= 0:
                    be_dist_text = "· достигнут по закрытой свече"
                else:
                    be_dist_text = f"· осталось {dist_to_be:.2f}"
                
                levels_block_lines.append(f"BE: {fmt_price(be_trigger)} {be_dist_text}")
    
    # Signal block
    entry_signal_text = "нет"
    if entry_signal == "LONG":
        entry_signal_text = "LONG"
    elif entry_signal == "SHORT":
        entry_signal_text = "SHORT"
    
    exit_signal_text = "нет"
    if exit_signal:
        exit_signal_text = exit_signal
    
    # Status text
    status_text = "наблюдение"
    
    if direction == "NONE":
        if entry_signal == "LONG":
            status_text = "технический сигнал на вход LONG"
        elif entry_signal == "SHORT":
            status_text = "технический сигнал на вход SHORT"
    else:
        if exit_signal == "EXIT_SL":
            status_text = "технический сигнал на выход по SL"
        elif exit_signal == "EXIT_TP":
            status_text = "технический сигнал на выход по TP"
        elif exit_signal == "EXIT_SAR":
            status_text = "технический сигнал на выход по SAR"
        elif exit_signal == "EXIT_BE_STOP":
            status_text = "технический сигнал на выход по BE"
        elif exit_signal == "BE_TRIGGERED":
            status_text = "достигнут уровень безубытка, стоп переведен в BE"
        elif direction == "LONG":
            status_text = "держим LONG"
        elif direction == "SHORT":
            status_text = "держим SHORT"
    
    # Hypothetical levels (if no position but signal exists)
    hypothetical_block = ""
    if direction == "NONE" and entry_signal and close_price is not None:
        atr_val = float(atr) if atr is not None and not (isinstance(atr, float) and np.isnan(atr)) else 0
        
        if entry_signal == "LONG":
            hyp_entry = close_price
            hyp_sl = close_price - atr_val * 3.0 if atr_val > 0 else None
            hyp_tp = close_price * 1.07
            hyp_be = close_price * 1.02
        else:  # SHORT
            hyp_entry = close_price
            hyp_sl = close_price + atr_val * 3.0 if atr_val > 0 else None
            hyp_tp = close_price * 0.93
            hyp_be = close_price * 0.98
        
        hypothetical_block = (
            "\nГипотетические уровни:\n"
            f"Entry: {fmt_price(hyp_entry)}\n"
            f"SL: {fmt_price(hyp_sl)}\n"
            f"TP: {fmt_price(hyp_tp)}\n"
            f"BE: {fmt_price(hyp_be)}"
        )
    
    # Warnings block
    warnings_block = ""
    if warnings:
        warnings_block = "\n\n⚠️ Warnings:\n" + "\n".join(f"   - {w}" for w in warnings)
    
    # Opposite signal warning
    opposite_warning = ""
    if direction == "LONG" and entry_signal == "SHORT":
        opposite_warning = "\n⚠️ Встречный сигнал SHORT не является выходом по текущим правилам."
    elif direction == "SHORT" and entry_signal == "LONG":
        opposite_warning = "\n⚠️ Встречный сигнал LONG не является выходом по текущим правилам."
    
    # Build message
    header = "GLDRUBF · 4H · ручной режим"
    
    market_section = (
        "Рынок:\n"
        f"{channel_text}\n"
        f"{price_position}\n"
        f"{atr_text}\n"
        f"SAR: {sar_trend_str}{sar_supports}{sar_reversed}"
    )
    
    position_section = "\n".join(position_block_lines)
    
    levels_section = ""
    if levels_block_lines:
        levels_section = "\n\nУровни:\n" + "\n".join(levels_block_lines)
    
    signal_section = (
        "\n\nСигнал:\n"
        f"Вход: {entry_signal_text}\n"
        f"Выход: {exit_signal_text}"
    )
    
    status_section = f"\n\nСтатус: {status_text}"
    
    footer = "\n\nРучной режим. Ордера не отправляются."
    
    # Debug block (optional)
    debug_block = ""
    if debug_mode:
        debug_block = (
            "\n\n--- debug ---\n"
            f"entry_signal: {entry_signal}\n"
            f"exit_signal: {exit_signal}\n"
            f"action: {action}\n"
            f"sar_reversal_up: {sar_reversal_up}\n"
            f"sar_reversal_down: {sar_reversal_down}\n"
            f"be_activated: {be_activated}\n"
            f"last_candle: {candle_time_str}"
        )
    
    message = (
        f"{header}\n"
        f"Свеча: {candle_time_str}\n"
        f"Close: {fmt_price(close_price)}\n"
        "\n"
        f"{market_section}\n"
        f"{position_section}"
        f"{levels_section}"
        f"{signal_section}"
        f"{status_section}"
        f"{hypothetical_block}"
        f"{opposite_warning}"
        f"{warnings_block}"
        f"{footer}"
        f"{debug_block}"
    )
    
    return message
