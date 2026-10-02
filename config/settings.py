"""
Strategy configuration parameters.

Legacy GLDRUBF parameters below are kept ONLY because the preserved legacy
strategy modules (src/strategy.py, src/positions.py, src/indicators.py,
src/position_monitor.py) still import them. The active trading runtime on
main uses the frozen C5 block at the bottom of this file.
Full legacy configuration lives in git branch: legacy/gldrubf-old-strategy
"""

# ============================================================
# LEGACY STRATEGY PARAMETERS (GLDRUBF) — not used by C5 runtime
# ============================================================

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
# Stage 2D: the active production instrument is CNYRUBF (C5 strategy).
# GLDRUBF belongs to legacy/gldrubf-old-strategy and must not appear in
# the runtime path. TARGET_TICKER kept as an alias so generic infra/tests
# resolve to the ACTIVE instrument, not a hidden GLDRUBF dependency.
TARGET_TICKER = "CNYRUBF"
LEGACY_TARGET_TICKER = "GLDRUBF"  # archive reference only — NOT used by runtime

# Trading mode
AUTO_TRADING_ENABLED = True  # Enable automatic trading (False = signal only)
RESERVE_RATIO = 0.10         # Keep 10% of deposit as reserve
MAX_DAILY_LOSS_PCT = 0.03    # Halt new entries after 3% realized loss

# Telegram settings
TELEGRAM_ENABLED = False     # Enable Telegram notifications
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

# ============================================================
# FROZEN C5 STRATEGY PARAMETERS (CNYRUBF_SPBFUT) — active on main
# Signal logic is frozen: do not add filters/optimization/stops.
# ============================================================

C5_TARGET_TICKER = "CNYRUBF"
C5_CLASS_CODE = "SPBFUT"
C5_INSTRUMENT_ISIN = "CNYRUBF_SPBFUT"

# Signal timeframe: DAILY closed bars for channels/ATR + one 5m control
# candle at 16:00 MSK per trading day (spec §CONTROL_TIME / §BAR_TIMEFRAME).
C5_TIMEFRAME = "1d"
C5_CONTROL_TIMEFRAME = "5m"
C5_CONTROL_HOUR_MS = 16          # 16:00 MSK control point, once per day

# Entry Donchian N = 10, Exit Donchian M = 5 (trend following, frozen).
# Channels are built from CLOSED DAILY bars shifted by 1 (no lookahead);
# the comparison price is the close of the 16:00 MSK 5m control candle.
C5_ENTRY_DONCHIAN = 10
C5_EXIT_DONCHIAN = 5

# ATR period used only for risk adjustment (not a signal component)
C5_ATR_LEN = 14

# ATR risk adjustment: risk_multiplier = clip(ATR_target / ATR14, 0.5, 1.5)
# NOTE: the ratio is target/current — higher volatility -> lower risk.
C5_ATR_TARGET = 0.1563
C5_RISK_MULT_MIN = 0.5
C5_RISK_MULT_MAX = 1.5

# Risk budget fraction of equity: risk_budget = equity * 0.20 * multiplier
C5_RISK_BUDGET_PCT = 0.20

# Instrument parameter: one CNYRUBF contract = 1000 CNY.
# This is NOT derived from `lot` (lot == 1 contract for SPBFUT futures).
C5_CONTRACT_SIZE_CNY = 1000.0

# Engineering / liquidity cap (NOT the primary risk control)
C5_MAX_QTY = 40

# Deployment stage limit: Stage 1 caps executed quantity at 5 contracts.
C5_STAGE_MAX_QTY = 5

# Operational safety limit: margin used / equity must stay <= 30%.
C5_ME_LIMIT = 0.30

# Execution guard: reject entry when expected slippage exceeds 0.10%.
C5_MAX_EXPECTED_SLIPPAGE = 0.0010

# Risk protection thresholds
C5_DAILY_LOSS_BREAKER = 0.10   # vs equity at start of trading day
C5_DRAWDOWN_FREEZE = 0.20      # vs historical equity peak: block entries
C5_KILL_SWITCH_DD = 0.30       # vs historical equity peak: full halt

# Margin refresh policy (no fallback values are permitted):
# refreshed at startup, before every new entry, after position change,
# after reconnect and after API recovery; MARGIN_RATE_CHANGED event on delta.
C5_MARGIN_REFRESH_BEFORE_ENTRY = True
C5_MARGIN_RATE_CHANGE_EPS = 1e-9

# Validity band for a live ГО rate coming from get_future(..., FULL).
# Out-of-band values are treated as MARGIN_API_UNAVAILABLE (NO ENTRY);
# there is NO fallback to remembered/hardcoded rates (0.0585 / 0.0576).
C5_MARGIN_RATE_MIN = 0.0
C5_MARGIN_RATE_MAX = 0.5

# C5 auto-trading mode switch (paper/live handled by T_Invest sandbox token)
AUTO_TRADING_ENABLED = True
