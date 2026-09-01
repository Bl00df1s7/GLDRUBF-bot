"""
Position monitoring module - calculates position health metrics.
SIGNAL ONLY MODE - No trading operations.

This module tracks open positions and calculates:
- Structural levels (entry candle, previous H4 candle)
- Pressure against position
- Adverse speed
- MAE/MFE
- Distance to SL/TP/BE
- Correlated instruments dynamics
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from config.settings import (
    POSITION_MONITOR_ENABLED,
    MONITOR_TIMEFRAME,
    CRITICAL_SL_DISTANCE_ATR_MULT,
    PRESSURE_ENABLED,
    PRESSURE_CONSECUTIVE_CANDLES,
    PRESSURE_MIN_ATR_MULT,
    STRONG_PRESSURE_CONSECUTIVE_CANDLES,
    STRONG_PRESSURE_ATR_MULT,
    ADVERSE_SPEED_ENABLED,
    ADVERSE_SPEED_LOOKBACK_BARS,
    ADVERSE_SPEED_WARNING_ATR_MULT,
    ADVERSE_SPEED_CRITICAL_ATR_MULT,
    MAE_MFE_ENABLED,
    CORRELATED_INSTRUMENTS,
    CORRELATED_PERIODS,
    STRUCTURE_ENTRY_CANDLE_ENABLED,
    STRUCTURE_PREV_H4_CANDLE_ENABLED,
)


# Alert levels
@dataclass
class AlertLevel:
    NORMAL = "NORMAL"
    PRESSURE = "PRESSURE"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    FAST_ADVERSE = "FAST_ADVERSE"
    CRITICAL = "CRITICAL"


# Human-readable labels for alert levels
ALERT_LABELS = {
    AlertLevel.NORMAL: "🟢 Спокойно",
    AlertLevel.PRESSURE: "🟡 Давление против позиции",
    AlertLevel.STRUCTURE_BREAK: "🟠 Структура нарушена",
    AlertLevel.FAST_ADVERSE: "🔴 Быстрое движение против позиции",
    AlertLevel.CRITICAL: "🚨 Критическая зона",
}


@dataclass
class CorrelatedMetric:
    """Raw dynamics for correlated instrument."""
    instrument: str
    change_since_entry_pct: Optional[float] = None
    change_1h_pct: Optional[float] = None
    change_4h_pct: Optional[float] = None
    change_since_entry_points: Optional[float] = None
    change_1h_points: Optional[float] = None
    change_4h_points: Optional[float] = None


@dataclass
class PositionHealth:
    """Position health state."""
    # Basic info
    position_key: Optional[str] = None
    direction: Optional[str] = None  # LONG or SHORT
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    monitor_timeframe: str = "1H"
    last_closed_candle_time: Optional[datetime] = None
    last_closed_close: Optional[float] = None
    
    # P&L
    pnl_points: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    # Distances to levels
    distance_to_sl_points: Optional[float] = None
    distance_to_tp_points: Optional[float] = None
    distance_to_be_points: Optional[float] = None
    
    # MAE/MFE
    max_favorable_price: Optional[float] = None
    max_adverse_price: Optional[float] = None
    mfe_points: Optional[float] = None
    mae_points: Optional[float] = None
    drawback_from_peak_points: Optional[float] = None
    drawback_from_peak_pct: Optional[float] = None
    
    # Structural levels
    entry_candle_low: Optional[float] = None
    entry_candle_high: Optional[float] = None
    prev_h4_candle_low: Optional[float] = None
    prev_h4_candle_high: Optional[float] = None
    entry_structure_broken: bool = False
    prev_h4_structure_broken: bool = False
    
    # Pressure
    pressure_count: int = 0
    pressure_move_points: Optional[float] = None
    pressure_atr_mult: Optional[float] = None
    
    # Adverse speed
    adverse_speed_points: Optional[float] = None
    adverse_speed_atr_mult: Optional[float] = None
    
    # Correlated instruments
    correlated_metrics: List[CorrelatedMetric] = field(default_factory=list)
    
    # Alert state
    alert_level: str = AlertLevel.NORMAL
    alert_reasons: List[str] = field(default_factory=list)


def _calculate_pnl(direction: str, entry_price: float, current_price: float) -> tuple:
    """Calculate P&L in points and percentage."""
    if direction == "LONG":
        pnl_points = current_price - entry_price
    else:  # SHORT
        pnl_points = entry_price - current_price
    
    pnl_pct = (pnl_points / entry_price) * 100 if entry_price != 0 else None
    return pnl_points, pnl_pct


def _calculate_distance_to_level(
    direction: str,
    current_price: float,
    level_price: float,
    is_sl: bool = False
) -> Optional[float]:
    """Calculate distance to a level in points."""
    if level_price is None:
        return None
    
    if direction == "LONG":
        distance = current_price - level_price if is_sl else level_price - current_price
    else:  # SHORT
        distance = level_price - current_price if is_sl else current_price - level_price
    
    return distance


def _check_structure_break(
    direction: str,
    last_close: float,
    entry_candle_low: Optional[float],
    entry_candle_high: Optional[float],
    prev_h4_candle_low: Optional[float],
    prev_h4_candle_high: Optional[float]
) -> tuple:
    """Check if structural levels are broken."""
    entry_broken = False
    prev_h4_broken = False
    
    if direction == "LONG":
        if entry_candle_low is not None and last_close < entry_candle_low:
            entry_broken = True
        if prev_h4_candle_low is not None and last_close < prev_h4_candle_low:
            prev_h4_broken = True
    else:  # SHORT
        if entry_candle_high is not None and last_close > entry_candle_high:
            entry_broken = True
        if prev_h4_candle_high is not None and last_close > prev_h4_candle_high:
            prev_h4_broken = True
    
    return entry_broken, prev_h4_broken


def _calculate_pressure(
    df_monitor: pd.DataFrame,
    direction: str,
    monitor_atr: float
) -> tuple:
    """
    Calculate pressure against position.
    Returns (pressure_count, pressure_move_points, pressure_atr_mult).
    """
    if len(df_monitor) < 2:
        return 0, None, None
    
    closes = df_monitor['close'].values
    
    # Count consecutive candles against position
    pressure_count = 0
    series_start_idx = len(closes) - 1
    
    for i in range(len(closes) - 1, 0, -1):
        current_close = closes[i]
        prev_close = closes[i - 1]
        
        is_against = False
        if direction == "LONG":
            is_against = current_close < prev_close
        else:  # SHORT
            is_against = current_close > prev_close
        
        if is_against:
            pressure_count += 1
            series_start_idx = i - 1
        else:
            break
    
    if pressure_count == 0:
        return 0, None, None
    
    # Calculate total move against position
    start_close = closes[series_start_idx]
    last_close = closes[-1]
    
    if direction == "LONG":
        pressure_move = start_close - last_close
    else:  # SHORT
        pressure_move = last_close - start_close
    
    # Calculate in ATR units
    pressure_atr_mult = pressure_move / monitor_atr if monitor_atr > 0 else None
    
    return pressure_count, pressure_move, pressure_atr_mult


def _calculate_adverse_speed(
    df_monitor: pd.DataFrame,
    direction: str,
    monitor_atr: float,
    lookback_bars: int = 2
) -> tuple:
    """
    Calculate adverse speed over lookback bars.
    Returns (adverse_speed_points, adverse_speed_atr_mult).
    """
    if len(df_monitor) < lookback_bars + 1:
        return None, None
    
    closes = df_monitor['close'].values
    
    # Get close from lookback_bars ago and current close
    old_close = closes[-(lookback_bars + 1)]
    current_close = closes[-1]
    
    if direction == "LONG":
        adverse_move = old_close - current_close
    else:  # SHORT
        adverse_move = current_close - old_close
    
    # If move is negative (in favor), set to 0
    if adverse_move < 0:
        adverse_move = 0
    
    adverse_atr_mult = adverse_move / monitor_atr if monitor_atr > 0 else None
    
    return adverse_move, adverse_atr_mult


def _calculate_mae_mfe(
    df_monitor: pd.DataFrame,
    direction: str,
    entry_price: float
) -> dict:
    """
    Calculate MAE/MFE from entry.
    Returns dict with mfe_points, mae_points, max_favorable_price, max_adverse_price.
    """
    if len(df_monitor) == 0:
        return {
            'mfe_points': None,
            'mae_points': None,
            'max_favorable_price': None,
            'max_adverse_price': None,
        }
    
    highs = df_monitor['high'].values
    lows = df_monitor['low'].values
    
    if direction == "LONG":
        max_favorable = float(np.max(highs))
        max_adverse = float(np.min(lows))
        mfe_points = max_favorable - entry_price
        mae_points = entry_price - max_adverse
    else:  # SHORT
        max_favorable = float(np.min(lows))
        max_adverse = float(np.max(highs))
        mfe_points = entry_price - max_favorable
        mae_points = max_adverse - entry_price
    
    return {
        'mfe_points': mfe_points if mfe_points > 0 else None,
        'mae_points': mae_points if mae_points > 0 else None,
        'max_favorable_price': max_favorable,
        'max_adverse_price': max_adverse,
    }


def _get_structural_levels(
    df_4h: pd.DataFrame,
    entry_time: datetime
) -> dict:
    """
    Get structural levels from entry candle and previous H4 candle.
    Returns dict with entry_candle_low/high, prev_h4_candle_low/high.
    """
    result = {
        'entry_candle_low': None,
        'entry_candle_high': None,
        'prev_h4_candle_low': None,
        'prev_h4_candle_high': None,
    }
    
    if df_4h.empty:
        return result
    
    # Find entry candle (candle containing entry_time)
    entry_candle = None
    prev_h4_candle = None
    
    for idx, row in df_4h.iterrows():
        candle_time = row.get('time')
        if candle_time is None:
            continue
        
        candle_end = candle_time + timedelta(hours=4)
        
        # Check if this is the entry candle
        if entry_time is not None and candle_time <= entry_time < candle_end:
            entry_candle = row
        elif entry_time is not None and candle_end <= entry_time:
            # This candle closed before entry - could be previous
            if prev_h4_candle is None or row.get('time', datetime.min.replace(tzinfo=timezone.utc)) > prev_h4_candle.get('time', datetime.min.replace(tzinfo=timezone.utc)):
                prev_h4_candle = row
    
    # If entry_candle not found, use the last closed candle before entry
    if entry_candle is None and entry_time is not None:
        for idx, row in df_4h.iterrows():
            candle_time = row.get('time')
            if candle_time and candle_time <= entry_time:
                entry_candle = row
    
    # If still no entry candle, use most recent
    if entry_candle is None and len(df_4h) > 0:
        entry_candle = df_4h.iloc[-1]
    
    # Get previous H4 candle (the one before entry candle)
    if prev_h4_candle is None and len(df_4h) > 1:
        # Find index of entry candle
        if entry_candle is not None:
            try:
                entry_idx = df_4h.index.get_loc(entry_candle.name)
                if entry_idx > 0:
                    prev_h4_candle = df_4h.iloc[entry_idx - 1]
            except (KeyError, IndexError):
                pass
        
        # Fallback: use second-to-last
        if prev_h4_candle is None:
            prev_h4_candle = df_4h.iloc[-2]
    
    if entry_candle is not None:
        result['entry_candle_low'] = float(entry_candle.get('low'))
        result['entry_candle_high'] = float(entry_candle.get('high'))
    
    if prev_h4_candle is not None:
        result['prev_h4_candle_low'] = float(prev_h4_candle.get('low'))
        result['prev_h4_candle_high'] = float(prev_h4_candle.get('high'))
    
    return result


def calculate_position_health(
    position_state: dict,
    df_monitor: pd.DataFrame,
    df_4h: pd.DataFrame,
    stored_levels: dict,
    entry_time: Optional[datetime] = None,
    monitor_atr: Optional[float] = None
) -> PositionHealth:
    """
    Calculate complete position health state.
    
    Args:
        position_state: Position state from positions.py
        df_monitor: DataFrame with monitor timeframe candles (e.g., 1H)
        df_4h: DataFrame with 4H candles for structural levels
        stored_levels: Levels from state store (SL, TP, BE, etc.)
        entry_time: Entry time for the position
        monitor_atr: ATR value for monitor timeframe
    
    Returns:
        PositionHealth object with all metrics
    """
    health = PositionHealth()
    
    # Basic info
    direction = position_state.get('direction')
    if direction not in ('LONG', 'SHORT'):
        health.alert_level = AlertLevel.NORMAL
        return health
    
    entry_price = position_state.get('entry_price')
    if entry_price is None or (isinstance(entry_price, float) and np.isnan(entry_price)):
        health.alert_level = AlertLevel.NORMAL
        return health
    
    health.direction = direction
    health.entry_price = float(entry_price)
    health.entry_time = entry_time
    health.monitor_timeframe = MONITOR_TIMEFRAME
    
    # Get last closed candle
    if df_monitor.empty:
        health.alert_level = AlertLevel.NORMAL
        return health
    
    last_candle = df_monitor.iloc[-1]
    health.last_closed_candle_time = last_candle.get('time')
    health.last_closed_close = float(last_candle.get('close'))
    
    # Get levels
    sl_price = stored_levels.get('recommended_sl') or position_state.get('sl_price')
    tp_price = stored_levels.get('tp') or position_state.get('tp_price')
    be_trigger = stored_levels.get('be_trigger') or position_state.get('be_trigger')
    be_activated = stored_levels.get('be_activated', False)
    
    # Calculate P&L
    pnl_points, pnl_pct = _calculate_pnl(direction, float(entry_price), health.last_closed_close)
    health.pnl_points = pnl_points
    health.pnl_pct = pnl_pct
    
    # Calculate distances to levels
    if sl_price is not None:
        health.distance_to_sl_points = _calculate_distance_to_level(
            direction, health.last_closed_close, sl_price, is_sl=True
        )
    
    if tp_price is not None:
        health.distance_to_tp_points = _calculate_distance_to_level(
            direction, health.last_closed_close, tp_price, is_sl=False
        )
    
    if be_trigger is not None and not be_activated:
        health.distance_to_be_points = _calculate_distance_to_level(
            direction, health.last_closed_close, be_trigger, is_sl=False
        )
    
    # Get ATR for calculations
    if monitor_atr is None:
        monitor_atr = float(last_candle.get('atr')) if 'atr' in last_candle else None
    
    if monitor_atr is None or monitor_atr <= 0:
        monitor_atr = abs(health.last_closed_close * 0.001)  # Fallback: 0.1% of price
    
    # Structural levels
    if STRUCTURE_ENTRY_CANDLE_ENABLED or STRUCTURE_PREV_H4_CANDLE_ENABLED:
        struct_levels = _get_structural_levels(df_4h, entry_time)
        health.entry_candle_low = struct_levels['entry_candle_low']
        health.entry_candle_high = struct_levels['entry_candle_high']
        health.prev_h4_candle_low = struct_levels['prev_h4_candle_low']
        health.prev_h4_candle_high = struct_levels['prev_h4_candle_high']
        
        # Check structure breaks
        entry_broken, prev_h4_broken = _check_structure_break(
            direction,
            health.last_closed_close,
            health.entry_candle_low,
            health.entry_candle_high,
            health.prev_h4_candle_low,
            health.prev_h4_candle_high
        )
        health.entry_structure_broken = entry_broken
        health.prev_h4_structure_broken = prev_h4_broken
    
    # Pressure calculation
    if PRESSURE_ENABLED and monitor_atr:
        pressure_count, pressure_move, pressure_atr_mult = _calculate_pressure(
            df_monitor, direction, monitor_atr
        )
        health.pressure_count = pressure_count
        health.pressure_move_points = pressure_move
        health.pressure_atr_mult = pressure_atr_mult
    
    # Adverse speed calculation
    if ADVERSE_SPEED_ENABLED and monitor_atr:
        adv_speed, adv_atr_mult = _calculate_adverse_speed(
            df_monitor, direction, monitor_atr, ADVERSE_SPEED_LOOKBACK_BARS
        )
        health.adverse_speed_points = adv_speed
        health.adverse_speed_atr_mult = adv_atr_mult
    
    # MAE/MFE calculation
    if MAE_MFE_ENABLED:
        mae_mfe = _calculate_mae_mfe(df_monitor, direction, float(entry_price))
        health.mfe_points = mae_mfe['mfe_points']
        health.mae_points = mae_mfe['mae_points']
        health.max_favorable_price = mae_mfe['max_favorable_price']
        health.max_adverse_price = mae_mfe['max_adverse_price']
        
        # Calculate drawback from peak
        if health.mfe_points and health.mfe_points > 0:
            if direction == "LONG":
                drawback = health.max_favorable_price - health.last_closed_close
            else:  # SHORT
                drawback = health.last_closed_close - health.max_favorable_price
            
            health.drawback_from_peak_points = drawback if drawback > 0 else 0
            health.drawback_from_peak_pct = (drawback / health.mfe_points) * 100 if drawback > 0 else 0
    
    # Determine alert level and reasons
    reasons = []
    alert_level = AlertLevel.NORMAL
    
    # Check critical SL distance
    if health.distance_to_sl_points is not None and monitor_atr:
        if health.distance_to_sl_points <= monitor_atr * CRITICAL_SL_DISTANCE_ATR_MULT:
            reasons.append("Цена находится близко к стоп-лоссу")
            alert_level = AlertLevel.CRITICAL
    
    # Check structure breaks
    if health.entry_structure_broken:
        if direction == "LONG":
            reasons.append("Цена закрылась ниже лоу свечи входа")
        else:
            reasons.append("Цена закрылась выше хая свечи входа")
        if alert_level not in (AlertLevel.CRITICAL,):
            alert_level = AlertLevel.STRUCTURE_BREAK
    
    if health.prev_h4_structure_broken:
        if direction == "LONG":
            reasons.append("Цена закрылась ниже лоу предыдущей 4H-свечи")
        else:
            reasons.append("Цена закрылась выше хая предыдущей 4H-свечи")
        if alert_level not in (AlertLevel.CRITICAL,):
            alert_level = AlertLevel.STRUCTURE_BREAK
    
    # Check fast adverse movement
    if health.adverse_speed_atr_mult is not None:
        if health.adverse_speed_atr_mult >= ADVERSE_SPEED_CRITICAL_ATR_MULT:
            reasons.append(f"Очень быстрое движение против позиции: {health.adverse_speed_points:.1f} пунктов за {ADVERSE_SPEED_LOOKBACK_BARS} свечи")
            alert_level = AlertLevel.CRITICAL
        elif health.adverse_speed_atr_mult >= ADVERSE_SPEED_WARNING_ATR_MULT:
            reasons.append(f"Быстрое движение против позиции: {health.adverse_speed_points:.1f} пунктов за {ADVERSE_SPEED_LOOKBACK_BARS} свечи")
            if alert_level not in (AlertLevel.CRITICAL, AlertLevel.STRUCTURE_BREAK):
                alert_level = AlertLevel.FAST_ADVERSE
    
    # Check pressure
    if health.pressure_count >= PRESSURE_CONSECUTIVE_CANDLES and health.pressure_atr_mult is not None:
        if health.pressure_atr_mult >= PRESSURE_MIN_ATR_MULT:
            reasons.append(f"{health.pressure_count} свечи(и) подряд против позиции")
            reasons.append(f"Суммарно: {health.pressure_move_points:.1f} пунктов / {health.pressure_atr_mult:.2f} ATR")
            
            if health.pressure_count >= STRONG_PRESSURE_CONSECUTIVE_CANDLES and health.pressure_atr_mult >= STRONG_PRESSURE_ATR_MULT:
                reasons.append("Сильное давление против позиции")
            
            if alert_level not in (AlertLevel.CRITICAL, AlertLevel.STRUCTURE_BREAK, AlertLevel.FAST_ADVERSE):
                alert_level = AlertLevel.PRESSURE
    
    # Check BE re-entry after activation
    if be_activated and entry_price is not None:
        if direction == "LONG" and health.last_closed_close <= entry_price:
            reasons.append("Цена вернулась в зону безубытка после активации BE")
            if alert_level not in (AlertLevel.CRITICAL,):
                alert_level = AlertLevel.STRUCTURE_BREAK
        elif direction == "SHORT" and health.last_closed_close >= entry_price:
            reasons.append("Цена вернулась в зону безубытка после активации BE")
            if alert_level not in (AlertLevel.CRITICAL,):
                alert_level = AlertLevel.STRUCTURE_BREAK
    
    health.alert_level = alert_level
    health.alert_reasons = reasons
    
    return health


def format_position_monitor_message(health: PositionHealth) -> str:
    """
    Format position health into compact Telegram message.
    
    Args:
        health: PositionHealth object
    
    Returns:
        Formatted message string
    """
    from src.telegram_bot import fmt_price
    
    lines = []
    
    # Header with alert level
    label = ALERT_LABELS.get(health.alert_level, "⚪ Статус неизвестен")
    lines.append(f"{label}")
    lines.append("")
    
    # Basic position info
    if health.direction:
        direction_icon = "🟢" if health.direction == "LONG" else "🔴"
        lines.append(f"Позиция: {direction_icon} {health.direction}")
    
    if health.entry_price:
        lines.append(f"Вход: {fmt_price(health.entry_price)}")
    
    if health.last_closed_close:
        lines.append(f"Цена {health.monitor_timeframe} закрыта: {fmt_price(health.last_closed_close)}")
    
    # P&L
    if health.pnl_points is not None and health.pnl_pct is not None:
        pnl_sign = "+" if health.pnl_points >= 0 else ""
        pct_sign = "+" if health.pnl_pct >= 0 else ""
        lines.append(f"P&L: {pnl_sign}{health.pnl_points:.1f} / {pct_sign}{health.pnl_pct:.2f}%")
    
    lines.append("")
    
    # Distances to levels
    if health.distance_to_sl_points is not None:
        lines.append(f"До SL: {health.distance_to_sl_points:.1f}")
    
    if health.distance_to_tp_points is not None:
        lines.append(f"До TP: {health.distance_to_tp_points:.1f}")
    
    if health.distance_to_be_points is not None:
        lines.append(f"До BE: {health.distance_to_be_points:.1f}")
    
    lines.append("")
    
    # Structure block
    structure_lines = []
    if health.entry_structure_broken:
        if health.direction == "LONG":
            structure_lines.append("Пробит лоу свечи входа")
        else:
            structure_lines.append("Пробит хай свечи входа")
    
    if health.prev_h4_structure_broken:
        if health.direction == "LONG":
            structure_lines.append("Пробит лоу предыдущей 4H-свечи")
        else:
            structure_lines.append("Пробит хай предыдущей 4H-свечи")
    
    if structure_lines:
        lines.append("Структура:")
        lines.extend(f"   {line}" for line in structure_lines)
        lines.append("")
    
    # Pressure block
    if health.pressure_count > 0 and health.pressure_move_points is not None:
        lines.append("Давление:")
        lines.append(f"   {health.pressure_count} свечи(и) подряд против позиции")
        lines.append(f"   Суммарно: {health.pressure_move_points:.1f} пунктов / {health.pressure_atr_mult:.2f} ATR")
        lines.append("")
    
    # Speed block
    if health.adverse_speed_points is not None and health.adverse_speed_atr_mult is not None:
        lines.append("Скорость:")
        lines.append(f"   {health.adverse_speed_atr_mult:.1f} ATR за {ADVERSE_SPEED_LOOKBACK_BARS} свечи")
        lines.append("")
    
    # MAE/MFE block
    if health.mfe_points is not None or health.drawback_from_peak_points is not None:
        lines.append("Откат от максимума:")
        if health.drawback_from_peak_points is not None:
            pct_str = f" / {health.drawback_from_peak_pct:.0f}%" if health.drawback_from_peak_pct else ""
            lines.append(f"   {health.drawback_from_peak_points:.1f} пунктов{pct_str}")
        if health.mfe_points is not None:
            lines.append(f"   Максимум в плюс: {health.mfe_points:.1f} пунктов")
        if health.mae_points is not None:
            lines.append(f"   Максимальная просадка: {health.mae_points:.1f} пунктов")
        lines.append("")
    
    # Footer
    lines.append("Ручной режим. Ордера не отправляются.")
    
    return "\n".join(lines)


def should_send_alert(
    health: PositionHealth,
    last_alert_level: Optional[str],
    last_alert_reasons: List[str],
    send_recovery: bool = False
) -> bool:
    """
    Check if alert should be sent based on state change.
    
    Args:
        health: Current position health
        last_alert_level: Previous alert level
        last_alert_reasons: Previous alert reasons
        send_recovery: Whether to send recovery messages
    
    Returns:
        True if alert should be sent
    """
    # No position - no alert
    if health.direction not in ('LONG', 'SHORT'):
        return False
    
    # First alert for this position
    if last_alert_level is None:
        return health.alert_level != AlertLevel.NORMAL
    
    # Level changed
    if health.alert_level != last_alert_level:
        # Don't send recovery to NORMAL unless configured
        if health.alert_level == AlertLevel.NORMAL and not send_recovery:
            return False
        return True
    
    # Same level but new reasons
    if health.alert_level != AlertLevel.NORMAL:
        current_reasons_set = set(health.alert_reasons)
        last_reasons_set = set(last_alert_reasons)
        
        # New reason appeared
        new_reasons = current_reasons_set - last_reasons_set
        if new_reasons:
            return True
    
    return False
