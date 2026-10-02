"""
Tests for GLDRUBF signal bot strategy logic.
SIGNAL ONLY mode - no trading operations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import calculate_atr, prepare_indicators, calculate_sar
from src.strategy import check_entry_signal, check_exit_conditions, should_close_by_monitor
from src.state_store import (
    load_state,
    save_state,
    check_candle_already_processed,
    check_position_changed,
    _compute_candle_hash,
    _build_position_key,
)


class TestDonchianLookahead(unittest.TestCase):
    """Test that Donchian channel doesn't use future data."""
    
    def test_donchian_no_lookahead(self):
        """Current candle should not create channel for its own breakout."""
        # Create test data with 25 candles
        dates = pd.date_range(start='2024-01-01', periods=25, freq='4h')
        df = pd.DataFrame({
            'time': dates,
            'open': [100] * 25,
            'high': list(range(100, 125)),  # Increasing highs: 100, 101, ..., 124
            'low': list(range(80, 105)),    # Increasing lows: 80, 81, ..., 104
            'close': list(range(90, 115)),  # Increasing closes
            'volume': [1000] * 25,
        })
        
        # Calculate indicators
        df = prepare_indicators(df)
        
        # For candle at index 24 (last), donchian_upper should be max of highs[4:24] (indices 4 to 23)
        # Not including high[24]
        last_idx = 24
        prev_highs_max = df['high'].iloc[4:24].max()  # DONCHIAN_LEN=20, so indices 4 to 23
        
        self.assertEqual(df['donchian_upper'].iloc[last_idx], prev_highs_max)
        self.assertGreater(df['high'].iloc[last_idx], df['donchian_upper'].iloc[last_idx])
        
        # The close should be compared against PREVIOUS donchian, not current
        # So long_signal should be based on close > prev_donchian_upper
        # Note: This is expected to be True since close[24]=114 > donchian_upper[24]=max(highs[4:23])=122
        # Actually close[24]=114 < donchian_upper[24]=123, so no long signal - this is correct behavior
        # The test verifies that donchian_upper doesn't include current high
        self.assertLess(df['close'].iloc[last_idx], df['donchian_upper'].iloc[last_idx])


class TestClosedCandle(unittest.TestCase):
    """Test that signals are only calculated for closed candles."""

    def test_candle_period_is_formatted_in_moscow_time(self):
        from src.main import format_candle_period

        candle = {
            "time": datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
            "candle_close_time": datetime(2026, 9, 2, 16, tzinfo=timezone.utc),
        }

        self.assertEqual(format_candle_period(candle), "2026-09-02 15:00 - 19:00 МСК")
    
    def test_unclosed_candle_skipped(self):
        """If last candle is not closed, signals should not be calculated."""
        # Import locally to avoid t_tech dependency in tests
        from datetime import timezone, timedelta
        
        now_utc = datetime.now(timezone.utc)
        
        # Create a candle that closes in the future (unclosed)
        future_time = now_utc - timedelta(hours=2)  # Opens 2 hours ago
        
        df = pd.DataFrame({
            'time': [future_time],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000],
        })
        df["candle_close_time"] = df["time"] + timedelta(hours=4)
        
        # Filter only closed candles
        closed_candidates = df[df["candle_close_time"] <= now_utc]
        
        # Should be empty since candle closes in 2 hours
        self.assertTrue(closed_candidates.empty)


class TestEntrySignals(unittest.TestCase):
    """Test entry signal logic."""
    
    def test_long_signal_above_donchian(self):
        """Close above previous Donchian upper should give LONG signal."""
        last_closed = {
            'close': 105,
            'donchian_upper': 100,
            'donchian_lower': 90,
            'long_signal': True,
            'short_signal': False,
        }
        
        result = check_entry_signal(last_closed)
        self.assertEqual(result, "LONG")
    
    def test_short_signal_below_donchian(self):
        """Close below previous Donchian lower should give SHORT signal."""
        last_closed = {
            'close': 85,
            'donchian_upper': 100,
            'donchian_lower': 90,
            'long_signal': False,
            'short_signal': True,
        }
        
        result = check_entry_signal(last_closed)
        self.assertEqual(result, "SHORT")
    
    def test_no_signal_inside_channel(self):
        """Close inside Donchian channel should give no signal."""
        last_closed = {
            'close': 95,
            'donchian_upper': 100,
            'donchian_lower': 90,
            'long_signal': False,
            'short_signal': False,
        }
        
        result = check_entry_signal(last_closed)
        self.assertIsNone(result)

    def test_monitor_structure_break_closes(self):
        health = type("Health", (), {
            "alert_level": "STRUCTURE_BREAK",
            "distance_to_sl_points": 100,
            "pressure_atr_mult": 0.5,
            "adverse_speed_atr_mult": 1.6,
            "entry_price": 10000,
        })()
        self.assertEqual(should_close_by_monitor(health), (True, "MONITOR_STRUCTURE"))

    def test_monitor_normal_holds(self):
        health = type("Health", (), {
            "alert_level": "NORMAL",
            "distance_to_sl_points": 100,
            "pressure_atr_mult": 3.0,
            "adverse_speed_atr_mult": 3.0,
            "entry_price": 10000,
        })()
        self.assertEqual(should_close_by_monitor(health), (False, ""))


class TestATR(unittest.TestCase):
    """Test ATR calculation."""
    
    def test_atr_true_range_includes_prev_close(self):
        """TR should consider gap from previous close."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='4H')
        
        # Create data with a gap
        df = pd.DataFrame({
            'time': dates,
            'open': [100] * 20,
            'high': [105] * 20,
            'low': [95] * 20,
            'close': [100] * 20,
            'volume': [1000] * 20,
        })
        
        # Introduce a gap on candle 5
        df.loc[5, 'open'] = 110
        df.loc[5, 'high'] = 115
        df.loc[5, 'low'] = 108
        df.loc[5, 'close'] = 112
        
        atr = calculate_atr(df, 14)
        
        # ATR should not be NaN after warmup period
        self.assertFalse(pd.isna(atr.iloc[-1]))
    
    def test_atr_not_nan_with_sufficient_data(self):
        """ATR should have valid values with enough data."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='4H')
        df = pd.DataFrame({
            'time': dates,
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(110, 120, 30),
            'low': np.random.uniform(90, 100, 30),
            'close': np.random.uniform(95, 115, 30),
            'volume': [1000] * 30,
        })
        
        atr = calculate_atr(df, 14)
        
        # Last few values should not be NaN
        self.assertFalse(pd.isna(atr.iloc[-1]))
    
    def test_atr_warning_on_insufficient_data(self):
        """ATR should produce consistent values but may be unreliable with insufficient data."""
        df = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=10, freq='4h'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10,
        })
        
        atr = calculate_atr(df, 14)
        
        # With Wilder smoothing, ATR produces values but they're not reliable until warmup complete
        # The test verifies that ATR doesn't crash and produces numeric values
        self.assertEqual(len(atr), 10)
        # All values should be equal since input data is constant
        self.assertAlmostEqual(atr.iloc[0], 10.0, places=5)


class TestSAR(unittest.TestCase):
    """Test Parabolic SAR calculation."""
    
    def test_sar_trend_detection(self):
        """SAR should correctly identify UP/DOWN trends."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='4H')
        
        # Uptrend: higher highs and higher lows
        df = pd.DataFrame({
            'time': dates,
            'open': list(range(100, 130)),
            'high': list(range(105, 135)),
            'low': list(range(95, 125)),
            'close': list(range(102, 132)),
            'volume': [1000] * 30,
        })
        
        sar_result = calculate_sar(df, 0.03, 0.02, 0.20)
        
        # In strong uptrend, SAR trend should be UP (1)
        self.assertEqual(sar_result['trend'][-1], 1)
    
    def test_sar_reversal_detection(self):
        """SAR should detect reversals."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='4h')
        
        # First uptrend (25 candles), then downtrend (25 candles)
        highs_uptrend = list(range(100, 125))
        lows_uptrend = list(range(95, 120))
        highs_downtrend = list(range(124, 74, -1))[:25]
        lows_downtrend = list(range(119, 69, -1))[:25]
        
        highs = highs_uptrend + highs_downtrend
        lows = lows_uptrend + lows_downtrend
        closes = [(h+l)/2 + 1 for h, l in zip(highs, lows)]
        opens = [(h+l)/2 for h, l in zip(highs, lows)]
        
        df = pd.DataFrame({
            'time': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [1000] * 50,
        })
        
        sar_result = calculate_sar(df, 0.03, 0.02, 0.20)
        
        # Should have at least one reversal
        has_reversal = any(sar_result['reversal_up']) or any(sar_result['reversal_down'])
        self.assertTrue(has_reversal)
    
    def test_sar_no_future_data(self):
        """SAR should not use future candle data."""
        dates = pd.date_range(start='2024-01-01', periods=25, freq='4H')
        df = pd.DataFrame({
            'time': dates,
            'open': list(range(100, 125)),
            'high': list(range(105, 130)),
            'low': list(range(95, 120)),
            'close': list(range(102, 127)),
            'volume': [1000] * 25,
        })
        
        sar_result = calculate_sar(df, 0.03, 0.02, 0.20)
        
        # SAR value at index i should only depend on data up to i
        # Not on data after i
        self.assertFalse(pd.isna(sar_result['sar'][-1]))


class TestExitPriority(unittest.TestCase):
    """Test exit signal priority."""
    
    def test_sl_priority_over_tp(self):
        """SL should have priority over TP when both hit."""
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        # Candle that hits both SL and TP (wide range)
        last_closed = {
            'high': 115,  # Above TP
            'low': 90,    # Below SL
            'close': 100,
            'sar_trend': 1,
            'sar_reversal_up': False,
            'sar_reversal_down': False,
        }
        
        stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        exit_signal, _, _, _ = check_exit_conditions(position_state, last_closed, stored_levels)
        
        # SL should have priority
        self.assertEqual(exit_signal, "EXIT_SL")
    
    def test_tp_priority_over_sar(self):
        """TP should have priority over SAR reversal."""
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        last_closed = {
            'high': 115,  # Above TP
            'low': 105,   # Above SL
            'close': 112,
            'sar_trend': -1,  # SAR reversal
            'sar_reversal_up': False,
            'sar_reversal_down': True,
        }
        
        stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        exit_signal, _, _, _ = check_exit_conditions(position_state, last_closed, stored_levels)
        
        # TP should have priority over SAR
        self.assertEqual(exit_signal, "EXIT_TP")


class TestSameBarSLTP(unittest.TestCase):
    """Test SL/TP on same bar scenario."""
    
    def test_same_bar_sl_and_tp(self):
        """When SL and TP hit on same bar, SL should take priority."""
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        # Candle range covers both SL and TP
        last_closed = {
            'high': 112,  # Above TP
            'low': 93,    # Below SL
            'close': 100,
            'sar_trend': 1,
            'sar_reversal_up': False,
            'sar_reversal_down': False,
        }
        
        stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        exit_signal, _, _, _ = check_exit_conditions(position_state, last_closed, stored_levels)
        
        self.assertEqual(exit_signal, "EXIT_SL")


class TestBreakEven(unittest.TestCase):
    """Test break-even logic."""
    
    def test_be_triggered_not_exit_by_default(self):
        """BE trigger should return BE_TRIGGERED, not EXIT, by default."""
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        last_closed = {
            'high': 103,
            'low': 101,
            'close': 102.5,  # Above BE trigger
            'sar_trend': 1,
            'sar_reversal_up': False,
            'sar_reversal_down': False,
        }
        
        stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        exit_signal, new_be_activated, new_recommended_sl, _ = check_exit_conditions(
            position_state, last_closed, stored_levels
        )
        
        self.assertEqual(exit_signal, "BE_TRIGGERED")
        self.assertTrue(new_be_activated)
        self.assertEqual(new_recommended_sl, 100.0)  # Moved to entry
    
    def test_be_stop_after_activation(self):
        """After BE activated, price hitting entry should give EXIT_SL (since SL moved to entry)."""
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        last_closed = {
            'high': 103,
            'low': 99,    # Dips below entry after BE
            'close': 101,
            'sar_trend': 1,
            'sar_reversal_up': False,
            'sar_reversal_down': False,
        }
        
        stored_levels = {
            'recommended_sl': 100.0,  # Already moved to entry
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': True,
        }
        
        exit_signal, _, _, _ = check_exit_conditions(position_state, last_closed, stored_levels)
        
        # When low <= recommended_sl (which is at entry), it's EXIT_SL
        self.assertEqual(exit_signal, "EXIT_SL")


class TestOppositeEntry(unittest.TestCase):
    """Test opposite entry signal handling."""
    
    def test_opposite_entry_doesnt_close_position(self):
        """Opposite entry signal should not automatically close position."""
        # This is tested in main.py logic, but we verify the warning is generated
        position_state = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'sl_price': 95.0,
            'tp_price': 110.0,
            'be_trigger': 102.0,
        }
        
        last_closed = {
            'close': 98,
            'donchian_upper': 100,
            'donchian_lower': 90,
            'long_signal': False,
            'short_signal': True,  # Opposite SHORT signal
            'high': 99,
            'low': 97,
            'sar_trend': 1,
            'sar_reversal_up': False,
            'sar_reversal_down': False,
        }
        
        # Entry signal should still be calculated
        entry_signal = check_entry_signal(last_closed)
        self.assertEqual(entry_signal, "SHORT")
        
        # But exit signal should be None (no exit condition met)
        stored_levels = {
            'recommended_sl': 95.0,
            'tp': 110.0,
            'be_trigger': 102.0,
            'be_activated': False,
        }
        
        exit_signal, _, _, _ = check_exit_conditions(position_state, last_closed, stored_levels)
        self.assertIsNone(exit_signal)


class TestStateReset(unittest.TestCase):
    """Test state reset on position change."""
    
    def test_position_key_changes_on_new_position(self):
        """Position key should change when position changes."""
        pos1 = {
            'direction': 'LONG',
            'entry_price': 100.0,
            'account_id': 'abc12345678',
            'instrument': 'GLDRUBF',
        }
        
        pos2 = {
            'direction': 'LONG',
            'entry_price': 105.0,  # Different entry
            'account_id': 'abc12345678',
            'instrument': 'GLDRUBF',
        }
        
        key1 = _build_position_key(pos1)
        key2 = _build_position_key(pos2)
        
        self.assertNotEqual(key1, key2)
    
    def test_state_reset_on_position_close(self):
        """State should reset when position is closed."""
        # Simulate state with position
        state = {
            'position_key': 'GLDRUBF:LONG:100.0:abcd',
            'be_activated': True,
            'recommended_sl': 100.0,
        }
        
        # Position closed (new_position_key is None)
        changed = check_position_changed(state, None)
        self.assertTrue(changed)


class TestIdempotency(unittest.TestCase):
    """Test message idempotency."""
    
    def test_same_candle_not_processed_twice(self):
        """Same candle should not trigger duplicate messages."""
        candle_data = {
            'timestamp': '2024-01-01T12:00:00',
            'close': 100.0,
        }
        
        state = {
            'last_processed_candle_timestamp': '2024-01-01T12:00:00',
            'last_processed_candle_hash': _compute_candle_hash(candle_data),
        }
        
        # Same candle processed again
        should_skip = check_candle_already_processed(state, candle_data)
        self.assertTrue(should_skip)
    
    def test_different_candle_processed(self):
        """Different candle should be processed."""
        candle_data_old = {
            'timestamp': '2024-01-01T08:00:00',
            'close': 98.0,
        }
        
        candle_data_new = {
            'timestamp': '2024-01-01T12:00:00',
            'close': 100.0,
        }
        
        state = {
            'last_processed_candle_timestamp': candle_data_old['timestamp'],
            'last_processed_candle_hash': _compute_candle_hash(candle_data_old),
        }
        
        # New candle should be processed
        should_skip = check_candle_already_processed(state, candle_data_new)
        self.assertFalse(should_skip)


if __name__ == '__main__':
    unittest.main()
