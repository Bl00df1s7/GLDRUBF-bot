"""
State store for signal bot - tracks position state between runs.
SIGNAL ONLY mode - no trading operations.
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

STATE_FILE = os.environ.get("STATE_FILE", "/tmp/gldrubf_state.json")


def _mask_account_id(account_id: str) -> str:
    """Mask account ID for logging/storage."""
    if not account_id or len(account_id) < 8:
        return "***"
    return f"{account_id[:4]}...{account_id[-4:]}"


def _compute_candle_hash(candle_data: dict) -> str:
    """Compute hash for candle to detect duplicates."""
    key_data = f"{candle_data.get('timestamp', '')}:{candle_data.get('close', 0)}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _build_position_key(position_state: dict) -> Optional[str]:
    """
    Build unique position key.
    Returns None if no position.
    """
    direction = position_state.get("direction")
    if direction == "NONE" or not direction:
        return None
    
    entry_price = position_state.get("entry_price", 0)
    # Use entry_price rounded to avoid floating point issues
    entry_key = f"{round(entry_price, 4) if entry_price else 'unknown'}"
    
    instrument = position_state.get("instrument", "GLDRUBF")
    account_masked = _mask_account_id(position_state.get("account_id", ""))
    
    return f"{instrument}:{direction}:{entry_key}:{account_masked}"


def load_state() -> dict:
    """Load state from file."""
    if not os.path.exists(STATE_FILE):
        return {
            "last_processed_candle_timestamp": None,
            "last_processed_candle_hash": None,
            "position_key": None,
            "direction_snapshot": "NONE",
            "entry_price_snapshot": None,
            "initial_sl": None,
            "recommended_sl": None,
            "tp": None,
            "be_trigger": None,
            "be_activated": False,
            "stop_order_ids": {},
            "stop_order_id": None,
            "entry_time": None,
            "entry_price": None,
            "entry_lots": 0,
            "daily_session_date": None,
            "daily_start_balance": None,
            "daily_realized_pnl": 0.0,
            "daily_loss_pct": 0.0,
            "TRADING_HALTED": False,
            "last_exit_signal": None,
            "last_action": None,
            "warnings": [],
            "last_run_timestamp": None,
            # Position monitor state
            "monitor_last_alert_level": None,
            "monitor_last_alert_reasons": [],
            "monitor_last_alert_candle_time": None,
            "monitor_last_state_hash": None,
        }
    
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            # Ensure all keys exist
            defaults = {
                "last_processed_candle_timestamp": None,
                "last_processed_candle_hash": None,
                "position_key": None,
                "direction_snapshot": "NONE",
                "entry_price_snapshot": None,
                "initial_sl": None,
                "recommended_sl": None,
                "tp": None,
                "be_trigger": None,
                "be_activated": False,
                "stop_order_ids": {},
                "stop_order_id": None,
                "entry_time": None,
                "entry_price": None,
                "entry_lots": 0,
                "daily_session_date": None,
                "daily_start_balance": None,
                "daily_realized_pnl": 0.0,
                "daily_loss_pct": 0.0,
                "TRADING_HALTED": False,
                "last_exit_signal": None,
                "last_action": None,
                "warnings": [],
                "last_run_timestamp": None,
                # Position monitor state
                "monitor_last_alert_level": None,
                "monitor_last_alert_reasons": [],
                "monitor_last_alert_candle_time": None,
                "monitor_last_state_hash": None,
            }
            for key, default_value in defaults.items():
                if key not in state:
                    state[key] = default_value
            return state
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ State load error: {e}")
        return {
            "last_processed_candle_timestamp": None,
            "last_processed_candle_hash": None,
            "position_key": None,
            "direction_snapshot": "NONE",
            "entry_price_snapshot": None,
            "initial_sl": None,
            "recommended_sl": None,
            "tp": None,
            "be_trigger": None,
            "be_activated": False,
            "stop_order_ids": {},
            "stop_order_id": None,
            "entry_time": None,
            "entry_price": None,
            "entry_lots": 0,
            "daily_session_date": None,
            "daily_start_balance": None,
            "daily_realized_pnl": 0.0,
            "daily_loss_pct": 0.0,
            "TRADING_HALTED": False,
            "last_exit_signal": None,
            "last_action": None,
            "warnings": ["State file corrupted, reset"],
            "last_run_timestamp": None,
            # Position monitor state
            "monitor_last_alert_level": None,
            "monitor_last_alert_reasons": [],
            "monitor_last_alert_candle_time": None,
            "monitor_last_state_hash": None,
        }


def save_state(state: dict) -> None:
    """Save state to file."""
    try:
        # Ensure directory exists
        state_dir = os.path.dirname(STATE_FILE)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except IOError as e:
        print(f"⚠️ State save error: {e}")


def check_candle_already_processed(state: dict, candle_data: dict) -> bool:
    """
    Check if this candle was already processed without state changes.
    Returns True if can skip (idempotency check).
    """
    last_hash = state.get("last_processed_candle_hash")
    last_timestamp = state.get("last_processed_candle_timestamp")
    
    current_hash = _compute_candle_hash(candle_data)
    current_timestamp = candle_data.get("timestamp")
    
    if last_hash == current_hash and last_timestamp == current_timestamp:
        return True
    
    return False


def check_position_changed(state: dict, new_position_key: Optional[str]) -> bool:
    """
    Check if position changed (new position or closed).
    Returns True if position changed.
    """
    old_key = state.get("position_key")
    
    if old_key is None and new_position_key is None:
        return False  # Both none - no change
    
    if old_key != new_position_key:
        return True  # Position changed
    
    return False


def update_state_for_new_position(
    state: dict,
    position_state: dict,
    initial_sl: float,
    tp: float,
    be_trigger: float
) -> dict:
    """
    Reset state for new position.
    Returns updated state.
    """
    state["position_key"] = _build_position_key(position_state)
    state["direction_snapshot"] = position_state.get("direction", "NONE")
    state["entry_price_snapshot"] = position_state.get("entry_price")
    state["initial_sl"] = initial_sl
    state["recommended_sl"] = initial_sl  # Initially same as initial
    state["tp"] = tp
    state["be_trigger"] = be_trigger
    state["be_activated"] = False
    state["stop_order_ids"] = {}
    state["stop_order_id"] = None
    state["entry_time"] = None
    state["entry_price"] = None
    state["entry_lots"] = 0
    state["last_exit_signal"] = None
    state["warnings"] = state.get("warnings", []) + ["New position detected, state reset"]
    
    # Reset monitor state for new position
    state["monitor_last_alert_level"] = None
    state["monitor_last_alert_reasons"] = []
    state["monitor_last_alert_candle_time"] = None
    state["monitor_last_state_hash"] = None
    
    return state


def update_state_for_closed_position(state: dict) -> dict:
    """
    Reset state when position is closed.
    Returns updated state.
    """
    state["position_key"] = None
    state["direction_snapshot"] = "NONE"
    state["entry_price_snapshot"] = None
    state["initial_sl"] = None
    state["recommended_sl"] = None
    state["tp"] = None
    state["be_trigger"] = None
    state["be_activated"] = False
    state["stop_order_ids"] = {}
    state["stop_order_id"] = None
    state["entry_time"] = None
    state["entry_price"] = None
    state["entry_lots"] = 0
    state["last_exit_signal"] = None
    state["warnings"] = state.get("warnings", []) + ["Position closed, state reset"]
    
    # Reset monitor state when position closes
    state["monitor_last_alert_level"] = None
    state["monitor_last_alert_reasons"] = []
    state["monitor_last_alert_candle_time"] = None
    state["monitor_last_state_hash"] = None
    
    return state


def activate_break_even(state: dict, new_recommended_sl: float) -> dict:
    """
    Activate break-even and update recommended SL.
    Returns updated state.
    """
    state["be_activated"] = True
    state["recommended_sl"] = new_recommended_sl
    state["warnings"] = state.get("warnings", []) + ["BE activated, SL moved"]
    
    return state


def update_candle_processed(state: dict, candle_data: dict, action: str, exit_signal: Optional[str]) -> dict:
    """
    Update state after processing candle.
    Returns updated state.
    """
    state["last_processed_candle_timestamp"] = candle_data.get("timestamp")
    state["last_processed_candle_hash"] = _compute_candle_hash(candle_data)
    state["last_action"] = action
    state["last_exit_signal"] = exit_signal
    state["last_run_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    return state


def get_stored_levels(state: dict) -> dict:
    """
    Get stored levels from state.
    Returns dictionary with levels or None values.
    """
    return {
        "initial_sl": state.get("initial_sl"),
        "recommended_sl": state.get("recommended_sl"),
        "tp": state.get("tp"),
        "be_trigger": state.get("be_trigger"),
        "be_activated": state.get("be_activated", False),
        "stop_order_ids": state.get("stop_order_ids", {}),
        "stop_order_id": state.get("stop_order_id"),
    }


def get_monitor_state(state: dict) -> dict:
    """
    Get position monitor state from state store.
    Returns dictionary with monitor alert state.
    """
    return {
        "last_alert_level": state.get("monitor_last_alert_level"),
        "last_alert_reasons": state.get("monitor_last_alert_reasons", []),
        "last_alert_candle_time": state.get("monitor_last_alert_candle_time"),
        "last_state_hash": state.get("monitor_last_state_hash"),
    }


def update_monitor_state(
    state: dict,
    alert_level: str,
    alert_reasons: list,
    candle_time: datetime
) -> dict:
    """
    Update position monitor state after sending alert.
    Returns updated state.
    """
    import hashlib
    
    # Compute hash of current state for deduplication
    state_data = f"{alert_level}:{','.join(sorted(alert_reasons))}"
    state_hash = hashlib.sha256(state_data.encode()).hexdigest()[:16]
    
    state["monitor_last_alert_level"] = alert_level
    state["monitor_last_alert_reasons"] = alert_reasons
    state["monitor_last_alert_candle_time"] = candle_time.isoformat() if candle_time else None
    state["monitor_last_state_hash"] = state_hash
    
    return state
