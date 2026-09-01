"""
Tests for GLDRUBF position monitor module.
SIGNAL ONLY mode - no trading operations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.position_monitor import (
    calculate_position_health,
    format_position_monitor_message,
    should_send_alert,
    PositionHealth,
    AlertLevel,
    _calculate_pnl,
    _calculate_distance_to_level,
    _check_structure_break,
    _calculate_pressure,
    _calculate_adverse_speed,
    _calculate_mae_mfe,
)


class TestPNLCalculation(unittest.TestCase):
    """Test P&L calculation."""
    
    def test_long_pnl_positive(self):
        """LONG position with profit."""
        pnl_points, pnl_pct = _calculate_pnl("LONG", 100.0, 105.0)
        self.assertEqual(pnl_points, 5.0)
        self.assertAlmostEqual(pnl_pct, 5.0, places=2)
    
    def test_long_pnl_negative(self):
        """LONG position with loss."""
        pnl_points, pnl_pct = _calculate_pnl("LONG", 100.0, 95.0)
        self.assertEqual(pnl_points, -5.0)
        self.assertAlmostEqual(pnl_pct, -5.0, places=2)
    
    def test_short_pnl_positive(self):
        """SHORT position with profit."""
        pnl_points, pnl_pct = _calculate_pnl("SHORT", 100.0, 95.0)
        self.assertEqual(pnl_points, 5.0)
        self.assertAlmostEqual(pnl_pct, 5.0, places=2)
    
    def test_short_pnl_negative(self):
        """SHORT position with loss."""
        pnl_points, pnl_pct = _calculate_pnl("SHORT", 100.0, 105.0)
        self.assertEqual(pnl_points, -5.0)
        self.assertAlmostEqual(pnl_pct, -5.0, places=2)


class TestDistanceToLevel(unittest.TestCase):
    """Test distance to level calculation."""
    
    def test_long_distance_to_sl(self):
        """LONG: distance to SL."""
        dist = _calculate_distance_to_level("LONG", 100.0, 95.0, is_sl=True)
        self.assertEqual(dist, 5.0)
    
    def test_long_distance_to_tp(self):
        """LONG: distance to TP."""
        dist = _calculate_distance_to_level("LONG", 100.0, 110.0, is_sl=False)
        self.assertEqual(dist, 10.0)
    
    def test_short_distance_to_sl(self):
        """SHORT: distance to SL."""
        dist = _calculate_distance_to_level("SHORT", 100.0, 105.0, is_sl=True)
        self.assertEqual(dist, 5.0)
    
    def test_short_distance_to_tp(self):
        """SHORT: distance to TP."""
        dist = _calculate_distance_to_level("SHORT", 100.0, 90.0, is_sl=False)
        self.assertEqual(dist, 10.0)
    
    def test_none_level(self):
        """None level returns None."""
        dist = _calculate_distance_to_level("LONG", 100.0, None, is_sl=True)
        self.assertIsNone(dist)


class TestStructureBreak(unittest.TestCase):
    """Test structure break detection."""
    
    def test_long_entry_structure_break(self):
        """LONG: close below entry candle low."""
        entry_broken, prev_h4_broken = _check_structure_break(
            "LONG", 95.0, 100.0, None, None, None
        )
        self.assertTrue(entry_broken)
        self.assertFalse(prev_h4_broken)
    
    def test_long_prev_h4_structure_break(self):
        """LONG: close below previous H4 candle low."""
        entry_broken, prev_h4_broken = _check_structure_break(
            "LONG", 95.0, None, None, 100.0, None
        )
        self.assertFalse(entry_broken)
        self.assertTrue(prev_h4_broken)
    
    def test_short_entry_structure_break(self):
        """SHORT: close above entry candle high."""
        entry_broken, prev_h4_broken = _check_structure_break(
            "SHORT", 105.0, None, 100.0, None, None
        )
        self.assertTrue(entry_broken)
        self.assertFalse(prev_h4_broken)
    
    def test_short_prev_h4_structure_break(self):
        """SHORT: close above previous H4 candle high."""
        entry_broken, prev_h4_broken = _check_structure_break(
            "SHORT", 105.0, None, None, None, 100.0
        )
        self.assertFalse(entry_broken)
        self.assertTrue(prev_h4_broken)
    
    def test_no_structure_break(self):
        """No structure break when price holds levels."""
        entry_broken, prev_h4_broken = _check_structure_break(
            "LONG", 105.0, 100.0, None, 95.0, None
        )
        self.assertFalse(entry_broken)
        self.assertFalse(prev_h4_broken)


class TestPressureCalculation(unittest.TestCase):
    """Test pressure calculation."""
    
    def test_pressure_long_downtrend(self):
        """LONG: consecutive down candles."""
        df = pd.DataFrame({
            'close': [105, 104, 103, 102, 101],
        })
        
        count, move, atr_mult = _calculate_pressure(df, "LONG", 1.0)
        self.assertEqual(count, 4)
        self.assertEqual(move, 4.0)
        self.assertEqual(atr_mult, 4.0)
    
    def test_pressure_short_uptrend(self):
        """SHORT: consecutive up candles."""
        df = pd.DataFrame({
            'close': [101, 102, 103, 104, 105],
        })
        
        count, move, atr_mult = _calculate_pressure(df, "SHORT", 1.0)
        self.assertEqual(count, 4)
        self.assertEqual(move, 4.0)
        self.assertEqual(atr_mult, 4.0)
    
    def test_no_pressure_long_uptrend(self):
        """LONG: up candles don't create pressure."""
        df = pd.DataFrame({
            'close': [101, 102, 103, 104, 105],
        })
        
        count, move, atr_mult = _calculate_pressure(df, "LONG", 1.0)
        self.assertEqual(count, 0)
        self.assertIsNone(move)
        self.assertIsNone(atr_mult)
    
    def test_insufficient_data(self):
        """Not enough data for pressure calculation."""
        df = pd.DataFrame({'close': [100]})
        
        count, move, atr_mult = _calculate_pressure(df, "LONG", 1.0)
        self.assertEqual(count, 0)


class TestAdverseSpeed(unittest.TestCase):
    """Test adverse speed calculation."""
    
    def test_adverse_speed_long_drop(self):
        """LONG: fast drop against position."""
        df = pd.DataFrame({
            'close': [105, 104, 103, 100, 98],
        })
        
        speed, atr_mult = _calculate_adverse_speed(df, "LONG", 1.0, lookback_bars=2)
        self.assertEqual(speed, 5.0)  # 103 - 98
        self.assertEqual(atr_mult, 5.0)
    
    def test_adverse_speed_short_rally(self):
        """SHORT: fast rally against position."""
        df = pd.DataFrame({
            'close': [95, 96, 97, 100, 102],
        })
        
        speed, atr_mult = _calculate_adverse_speed(df, "SHORT", 1.0, lookback_bars=2)
        self.assertEqual(speed, 5.0)  # 102 - 97
        self.assertEqual(atr_mult, 5.0)
    
    def test_adverse_speed_in_favor(self):
        """Movement in favor of position returns 0."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 105],
        })
        
        speed, atr_mult = _calculate_adverse_speed(df, "LONG", 1.0, lookback_bars=2)
        self.assertEqual(speed, 0)  # Movement is in favor
        self.assertEqual(atr_mult, 0.0)  # 0 / ATR = 0


class TestMAEMFE(unittest.TestCase):
    """Test MAE/MFE calculation."""
    
    def test_long_mfe_mae(self):
        """LONG: MFE and MAE calculation."""
        df = pd.DataFrame({
            'high': [102, 105, 103, 101],
            'low': [99, 101, 100, 98],
        })
        
        result = _calculate_mae_mfe(df, "LONG", 100.0)
        self.assertEqual(result['max_favorable_price'], 105.0)
        self.assertEqual(result['mfe_points'], 5.0)
        self.assertEqual(result['max_adverse_price'], 98.0)
        self.assertEqual(result['mae_points'], 2.0)
    
    def test_short_mfe_mae(self):
        """SHORT: MFE and MAE calculation."""
        df = pd.DataFrame({
            'high': [102, 105, 103, 101],
            'low': [99, 101, 100, 98],
        })
        
        result = _calculate_mae_mfe(df, "SHORT", 100.0)
        self.assertEqual(result['max_favorable_price'], 98.0)
        self.assertEqual(result['mfe_points'], 2.0)
        self.assertEqual(result['max_adverse_price'], 105.0)
        self.assertEqual(result['mae_points'], 5.0)


class TestPositionHealthIntegration(unittest.TestCase):
    """Integration tests for position health calculation."""
    
    def setUp(self):
        """Set up test data."""
        self.position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        self.stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        # Create 1H monitor data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        self.df_monitor = pd.DataFrame({
            'time': dates,
            'open': [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
            'high': [101, 102, 103, 104, 103, 102, 101, 100, 99, 98],
            'low': [99, 100, 101, 102, 101, 100, 99, 98, 97, 96],
            'close': [100, 101, 102, 103, 102, 101, 100, 99, 98, 97],
            'atr': [1.0] * 10,
        })
        
        # Create 4H structural data
        dates_4h = pd.date_range(start='2024-01-01', periods=5, freq='4H')
        self.df_4h = pd.DataFrame({
            'time': dates_4h,
            'open': [100, 101, 102, 103, 102],
            'high': [101, 102, 103, 104, 103],
            'low': [99, 100, 101, 102, 101],
            'close': [100, 101, 102, 103, 102],
        })
    
    def test_normal_state(self):
        """Position in normal state."""
        # Use favorable data
        self.df_monitor['close'] = [100, 101, 102, 103, 104, 105, 104, 103, 102, 103]
        self.df_monitor['high'] = self.df_monitor['close'] + 1
        self.df_monitor['low'] = self.df_monitor['close'] - 1
        
        health = calculate_position_health(
            self.position_state,
            self.df_monitor,
            self.df_4h,
            self.stored_levels,
        )
        
        self.assertEqual(health.direction, "LONG")
        self.assertEqual(health.alert_level, AlertLevel.NORMAL)
    
    def test_structure_break_triggers_alert(self):
        """Structure break triggers STRUCTURE_BREAK alert."""
        # Price breaks below entry candle low
        self.df_monitor.iloc[-1, self.df_monitor.columns.get_loc('close')] = 95.0
        
        health = calculate_position_health(
            self.position_state,
            self.df_monitor,
            self.df_4h,
            self.stored_levels,
        )
        
        # Structure break should trigger at least STRUCTURE_BREAK level (or higher like CRITICAL)
        self.assertIn(health.alert_level, [AlertLevel.STRUCTURE_BREAK, AlertLevel.CRITICAL])
        self.assertTrue(len(health.alert_reasons) > 0)
    
    def test_pressure_triggers_alert(self):
        """Consecutive pressure triggers PRESSURE alert."""
        # Create downtrend
        self.df_monitor['close'] = [105, 104, 103, 102, 101, 100, 99, 98, 97, 96]
        
        health = calculate_position_health(
            self.position_state,
            self.df_monitor,
            self.df_4h,
            self.stored_levels,
        )
        
        # Should have pressure detected
        self.assertGreater(health.pressure_count, 0)


class TestAlertSpamPrevention(unittest.TestCase):
    """Test anti-spam logic."""
    
    def test_no_duplicate_alert_same_state(self):
        """Same state should not trigger duplicate alert."""
        health = PositionHealth(
            direction="LONG",
            alert_level=AlertLevel.PRESSURE,
            alert_reasons=["2 свечи подряд против позиции"],
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=["2 свечи подряд против позиции"],
        )
        
        self.assertFalse(should_send)
    
    def test_new_reason_triggers_alert(self):
        """New reason should trigger alert."""
        health = PositionHealth(
            direction="LONG",
            alert_level=AlertLevel.PRESSURE,
            alert_reasons=["2 свечи подряд против позиции", "Сильное давление"],
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=["2 свечи подряд против позиции"],
        )
        
        self.assertTrue(should_send)
    
    def test_level_change_triggers_alert(self):
        """Level change should trigger alert."""
        health = PositionHealth(
            direction="LONG",
            alert_level=AlertLevel.CRITICAL,
            alert_reasons=["Цена находится близко к стоп-лоссу"],
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=["2 свечи подряд против позиции"],
        )
        
        self.assertTrue(should_send)
    
    def test_recovery_not_sent_by_default(self):
        """Recovery to NORMAL not sent by default."""
        health = PositionHealth(
            direction="LONG",
            alert_level=AlertLevel.NORMAL,
            alert_reasons=[],
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=["2 свечи подряд против позиции"],
            send_recovery=False,
        )
        
        self.assertFalse(should_send)
    
    def test_recovery_sent_when_enabled(self):
        """Recovery to NORMAL sent when enabled."""
        health = PositionHealth(
            direction="LONG",
            alert_level=AlertLevel.NORMAL,
            alert_reasons=[],
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=["2 свечи подряд против позиции"],
            send_recovery=True,
        )
        
        self.assertTrue(should_send)
    
    def test_no_position_no_alert(self):
        """No position means no alert."""
        health = PositionHealth(
            direction=None,
            alert_level=AlertLevel.NORMAL,
        )
        
        should_send = should_send_alert(
            health,
            last_alert_level=AlertLevel.PRESSURE,
            last_alert_reasons=[],
        )
        
        self.assertFalse(should_send)


class TestMessageFormatting(unittest.TestCase):
    """Test message formatting."""
    
    def test_format_normal_state(self):
        """Format NORMAL state message."""
        health = PositionHealth(
            direction="LONG",
            entry_price=100.0,
            last_closed_close=102.0,
            pnl_points=2.0,
            pnl_pct=2.0,
            distance_to_sl_points=7.0,
            distance_to_tp_points=8.0,
            alert_level=AlertLevel.NORMAL,
        )
        
        message = format_position_monitor_message(health)
        
        self.assertIn("🟢 Спокойно", message)
        self.assertIn("LONG", message)
        self.assertIn("P&L", message)
    
    def test_format_pressure_state(self):
        """Format PRESSURE state message."""
        health = PositionHealth(
            direction="LONG",
            entry_price=100.0,
            last_closed_close=98.0,
            pnl_points=-2.0,
            pnl_pct=-2.0,
            pressure_count=2,
            pressure_move_points=2.0,
            pressure_atr_mult=1.0,
            alert_level=AlertLevel.PRESSURE,
            alert_reasons=["2 свечи подряд против позиции"],
        )
        
        message = format_position_monitor_message(health)
        
        self.assertIn("🟡 Давление против позиции", message)
        self.assertIn("Давление:", message)


class TestBugFixes(unittest.TestCase):
    """Test fixes for reported bugs."""
    
    def test_distance_to_be_equals_pnl_when_profitable(self):
        """
        Bug #1 fix: Distance to BE should equal P&L when position is profitable.
        
        For a SHORT position at entry 12259.10, current price 12165.90:
        - P&L = +93.2 points (profit)
        - Distance to BE (entry_price) should be exactly 93.2 points
        
        Note: Distance to BE uses is_be=True flag for correct calculation.
        """
        # Test SHORT position with is_be=True
        dist_short = _calculate_distance_to_level("SHORT", 12165.90, 12259.10, is_sl=False, is_be=True)
        self.assertAlmostEqual(dist_short, 93.2, places=1)
        
        # Test LONG position with is_be=True
        dist_long = _calculate_distance_to_level("LONG", 105.0, 100.0, is_sl=False, is_be=True)
        self.assertEqual(dist_long, 5.0)
        
        # Verify: distance to BE should always equal abs(P&L) when profitable
        pnl_short, _ = _calculate_pnl("SHORT", 12259.10, 12165.90)
        self.assertAlmostEqual(abs(pnl_short), dist_short, places=1)
    
    def test_structure_break_only_on_close_not_wick(self):
        """
        Bug #2 fix: Structure break only triggers on CLOSE beyond level, not wick.
        
        For SHORT at 12259.10, entry candle high must be >= 12259.10.
        Current close at 12165.90 should NOT trigger structure break
        because it's below entry price, not above entry candle high.
        """
        entry_candle_high = 12280.0  # Entry candle high (contains entry price)
        current_close = 12165.90     # Current price (below entry)
        
        entry_broken, prev_h4_broken = _check_structure_break(
            "SHORT", 
            current_close, 
            None,           # entry_candle_low (not relevant)
            entry_candle_high,
            None, None      # prev_h4 levels
        )
        
        # Should NOT be broken - price closed BELOW entry candle high
        self.assertFalse(entry_broken)
        self.assertFalse(prev_h4_broken)
    
    def test_short_structure_break_requires_close_above_high(self):
        """SHORT structure break requires close ABOVE entry candle high."""
        entry_candle_high = 12259.10
        
        # Close above high - should break
        entry_broken, _ = _check_structure_break(
            "SHORT", 12270.0, None, entry_candle_high, None, None
        )
        self.assertTrue(entry_broken)
        
        # Close at high - should NOT break (needs to be strictly above)
        entry_broken, _ = _check_structure_break(
            "SHORT", 12259.10, None, entry_candle_high, None, None
        )
        self.assertFalse(entry_broken)
        
        # Close below high - should NOT break
        entry_broken, _ = _check_structure_break(
            "SHORT", 12200.0, None, entry_candle_high, None, None
        )
        self.assertFalse(entry_broken)
    
    def test_entry_candle_validation_contains_entry_price(self):
        """Entry candle must contain entry price in its range."""
        from src.position_monitor import _get_structural_levels
        
        # Create 4H candles where entry price is within candle range
        dates_4h = pd.date_range(start='2024-01-01', periods=3, freq='4H')
        df_4h = pd.DataFrame({
            'time': dates_4h,
            'open': [12200, 12250, 12300],
            'high': [12300, 12350, 12400],  # Second candle contains 12259.10
            'low': [12150, 12200, 12250],   # Second candle contains 12259.10
            'close': [12250, 12300, 12350],
        })
        
        entry_time = dates_4h[1]  # During second candle
        entry_price = 12259.10
        direction = "SHORT"
        
        result = _get_structural_levels(df_4h, entry_time, entry_price, direction)
        
        # Entry candle high should be >= entry price
        self.assertIsNotNone(result['entry_candle_high'])
        self.assertGreaterEqual(result['entry_candle_high'], entry_price)
        
        # Entry candle low should be <= entry price
        self.assertIsNotNone(result['entry_candle_low'])
        self.assertLessEqual(result['entry_candle_low'], entry_price)
    
    def test_full_integration_real_case(self):
        """Integration test with real reported bug case."""
        position_state = {
            'direction': 'SHORT',
            'entry_price': 12259.10,
            'sl_price': 12619.10,
            'tp_price': 11400.0,
            'be_trigger': 12259.10,
        }
        
        stored_levels = {
            'recommended_sl': 12619.10,
            'tp': 11400.0,
            'be_trigger': 12259.10,
            'be_activated': False,
        }
        
        # Create 1H monitor data ending at 12165.90
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        df_monitor = pd.DataFrame({
            'time': dates,
            'open': [12200, 12180, 12150, 12100, 12080, 12090, 12110, 12130, 12150, 12160],
            'high': [12220, 12200, 12170, 12120, 12100, 12110, 12130, 12150, 12170, 12180],
            'low': [12180, 12150, 12100, 12070, 12060, 12070, 12090, 12110, 12130, 12140],
            'close': [12200, 12180, 12150, 12100, 12080, 12090, 12110, 12130, 12150, 12165.90],
            'atr': [55.0] * 10,
        })
        
        # Create 4H structural data
        dates_4h = pd.date_range(start='2024-01-01', periods=5, freq='4H')
        df_4h = pd.DataFrame({
            'time': dates_4h,
            'open': [12200, 12250, 12300, 12350, 12400],
            'high': [12280, 12320, 12380, 12420, 12450],  # Entry candle high >= 12259.10
            'low': [12180, 12230, 12280, 12320, 12380],   # Entry candle low <= 12259.10
            'close': [12250, 12300, 12350, 12400, 12420],
        })
        
        health = calculate_position_health(
            position_state,
            df_monitor,
            df_4h,
            stored_levels,
        )
        
        # Verify P&L is correct
        self.assertAlmostEqual(health.pnl_points, 93.2, places=1)
        
        # Verify distance to BE equals P&L (Bug #1 fix)
        self.assertIsNotNone(health.distance_to_be_points)
        self.assertAlmostEqual(health.distance_to_be_points, 93.2, places=1)
        
        # Verify structure is NOT broken (Bug #2 fix)
        # Current close 12165.90 is BELOW entry candle high (~12320)
        self.assertFalse(health.entry_structure_broken)


if __name__ == '__main__':
    unittest.main()
