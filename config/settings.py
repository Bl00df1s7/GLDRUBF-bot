"""
Strategy configuration parameters.
"""

# Timeframe
TIMEFRAME = "4H"

# Entry - Donchian Channel
DONCHIAN_LEN = 20

# Volatility - ATR
ATR_LEN = 14

# Risk Management
SL_ATR = 3.0          # Stop Loss in ATR units
TP_PCT = 0.07         # Take Profit as percentage (7%)
BE_PCT = 0.02         # Break-Even trigger as percentage (2%)

# Parabolic SAR
SAR_START = 0.03
SAR_INC = 0.02
SAR_MAX = 0.20

# Target instrument
TARGET_TICKER = "GLDRUBF"

# Telegram settings
TELEGRAM_DEBUG_MODE = False  # Show technical debug info in messages

# ============================================================
# POSITION MONITOR SETTINGS
# ============================================================

# General
POSITION_MONITOR_ENABLED = True
MONITOR_TIMEFRAME = "1H"
MONITOR_ONLY_WHEN_POSITION = True
UPDATE_ON_STATE_CHANGE_ONLY = True

# Fast timeframe
FAST_TIMEFRAME_ENABLED = False
FAST_TIMEFRAME = "15m"
FAST_TIMEFRAME_ONLY_CRITICAL = True

# Structural levels
STRUCTURE_ENTRY_CANDLE_ENABLED = True
STRUCTURE_PREV_H4_CANDLE_ENABLED = True
CRITICAL_SL_DISTANCE_ATR_MULT = 0.33

# Pressure
PRESSURE_ENABLED = True
PRESSURE_CONSECUTIVE_CANDLES = 2
PRESSURE_MIN_ATR_MULT = 0.5
STRONG_PRESSURE_CONSECUTIVE_CANDLES = 3
STRONG_PRESSURE_ATR_MULT = 1.0

# Adverse speed
ADVERSE_SPEED_ENABLED = True
ADVERSE_SPEED_LOOKBACK_BARS = 2
ADVERSE_SPEED_WARNING_ATR_MULT = 1.0
ADVERSE_SPEED_CRITICAL_ATR_MULT = 1.5

# MAE/MFE
MAE_MFE_ENABLED = True

# Correlated instruments
CORRELATED_INSTRUMENTS = []
CORRELATED_PERIODS = ["since_entry", "1H", "4H"]
CORRELATED_SHOW_INTERPRETATION = False

# Recovery message
SEND_RECOVERY_MESSAGE = False
