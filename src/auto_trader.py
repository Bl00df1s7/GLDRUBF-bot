"""
Auto-trading module for GLDRUBF strategy.
Executes trades via T-Invest API based on strategy signals.
"""

import numpy as np
import time
from typing import Optional, Tuple

try:
    from t_tech.invest import OrderDirection, OrderType
    T_TECH_AVAILABLE = True
except ImportError:
    T_TECH_AVAILABLE = False
    OrderDirection = None
    OrderType = None

from config.settings import TARGET_TICKER
from src.client_factory import get_client
from src.instruments import quantize_lots
from src.api_retry import retry_api_call


# Reserve ratio: keep 10% of deposit as reserve, trade with 90%
RESERVE_RATIO = 0.10
TRADE_RATIO = 1.0 - RESERVE_RATIO
ORDER_FILL_TIMEOUT_SECONDS = 10.0
ORDER_POLL_INTERVAL_SECONDS = 0.5


def _status_name(status) -> str:
    """Return an SDK enum status name or a string status unchanged."""
    return getattr(status, "name", str(status).split(".")[-1])


def _ensure_market_order_available(client, instrument_uid: str) -> None:
    """Reject an order when T-Invest disables API market trading."""
    from src.market_data import is_trading_time

    if not is_trading_time():
        raise RuntimeError("Trading session has not started (before 03:00 MSK)")
    status = retry_api_call(client.market_data.get_trading_status)(
        instrument_id=instrument_uid
    )
    if not status.api_trade_available_flag or not status.market_order_available_flag:
        status_name = _status_name(status.trading_status)
        raise RuntimeError(f"Market orders unavailable: {status_name}")


def wait_for_order_fill(client, account_id: str, order_id: str) -> tuple:
    """Poll an order until filled, partially filled, or terminal timeout."""
    deadline = time.monotonic() + ORDER_FILL_TIMEOUT_SECONDS
    last_state = None

    while True:
        last_state = retry_api_call(client.orders.get_order_state)(
            account_id=account_id,
            order_id=order_id,
        )
        status = _status_name(last_state.execution_report_status)
        lots_executed = int(getattr(last_state, "lots_executed", 0) or 0)

        if status == "EXECUTION_REPORT_STATUS_FILL":
            return last_state, lots_executed
        if status == "EXECUTION_REPORT_STATUS_PARTIALLYFILL" and lots_executed > 0:
            return last_state, lots_executed
        if status in {
            "EXECUTION_REPORT_STATUS_REJECTED",
            "EXECUTION_REPORT_STATUS_CANCELLED",
        }:
            return last_state, 0
        if time.monotonic() >= deadline:
            return last_state, lots_executed if status == "EXECUTION_REPORT_STATUS_PARTIALLYFILL" else 0
        time.sleep(ORDER_POLL_INTERVAL_SECONDS)


def get_account_balance(token: str, account_id: str) -> float:
    """
    Get total portfolio value (balance) for position sizing.

    Args:
        token: T-Invest API token
        account_id: Account ID

    Returns:
        Total portfolio value in RUB
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available")

    with get_client(token) as client:
        portfolio = client.operations.get_portfolio(account_id=account_id)

        # total_amount_portfolio is the total portfolio value
        from src.market_data import quotation_to_float
        total_value = quotation_to_float(portfolio.total_amount_portfolio)

        return total_value


def calculate_position_size(
    balance: float,
    current_price: float,
    lot_size: int,
    instrument_uid: str,
    token: str,
    account_id: str,
) -> int:
    """
    Calculate number of lots to trade based on available capital (90% of deposit).

    For futures, margin is used instead of full price. We estimate margin
    from the instrument's initial margin or use a conservative approach.

    Args:
        balance: Total portfolio value
        current_price: Current instrument price
        lot_size: Lot size of the instrument
        instrument_uid: Instrument UID
        token: API token
        account_id: Account ID

    Returns:
        Number of lots to trade (integer >= 0)
    """
    if balance <= 0 or current_price <= 0 or lot_size <= 0:
        return 0

    available_capital = balance * TRADE_RATIO

    # Futures are sized by blocked initial margin, not contract notional.
    margin_per_lot = _get_margin_per_lot(token, instrument_uid)

    if margin_per_lot <= 0:
        print("   ⚠️ Initial margin is unavailable; entry is blocked")
        return 0

    max_lots = int(available_capital / margin_per_lot)

    # Safety: never exceed reasonable limits
    max_lots = min(max_lots, 100)  # Hard cap
    max_lots = max(max_lots, 0)

    print(f"   💰 Balance: {balance:.2f} RUB")
    print(f"   💰 Available (90%): {available_capital:.2f} RUB")
    print(f"   💰 Margin per lot: {margin_per_lot:.2f} RUB")
    print(f"   📦 Position size: {max_lots} lots")

    return max_lots


def _get_margin_per_lot(token: str, instrument_uid: str) -> float:
    """
    Get the broker's initial margin per futures lot.
    Returns zero when the broker value is unavailable.
    """
    try:
        with get_client(token) as client:
            instruments = retry_api_call(client.instruments.futures)().instruments
            for instrument in instruments:
                if instrument.uid != instrument_uid:
                    continue
                from src.market_data import quotation_to_float
                buy_margin = quotation_to_float(instrument.initial_margin_on_buy)
                sell_margin = quotation_to_float(instrument.initial_margin_on_sell)
                return max(buy_margin, sell_margin, 0.0)
    except Exception as e:
        print(f"   ⚠️ Could not get margin ratio: {e}")

    return 0.0


def open_position(
    token: str,
    account_id: str,
    instrument_uid: str,
    direction: str,
    quantity_lots: int,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    state: Optional[dict] = None,
    min_price_increment=None,
) -> Tuple[bool, str]:
    """
    Open a new position (LONG or SHORT).

    Args:
        token: API token
        account_id: Account ID
        instrument_uid: Instrument UID
        direction: "LONG" or "SHORT"
        quantity_lots: Number of lots
        price: Current price (used for position sizing and protection levels)

    Returns:
        Tuple of (success, message)
    """
    if not T_TECH_AVAILABLE:
        return False, "t_tech.invest not available"

    if quantity_lots <= 0:
        return False, "Quantity must be > 0"
    if stop_loss is None or take_profit is None:
        return False, "Protective SL/TP prices are required"

    if direction == "LONG":
        order_direction = OrderDirection.ORDER_DIRECTION_BUY
    elif direction == "SHORT":
        order_direction = OrderDirection.ORDER_DIRECTION_SELL
    else:
        return False, f"Invalid direction: {direction}"

    try:
        with get_client(token) as client:
            _ensure_market_order_available(client, instrument_uid)
            # Use market order for immediate execution
            response = retry_api_call(client.orders.post_order)(
                instrument_id=instrument_uid,
                quantity=quantity_lots,
                direction=order_direction,
                account_id=account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
            )

            order_id = response.order_id
            order_state, lots_executed = wait_for_order_fill(client, account_id, order_id)
            order_status = _status_name(order_state.execution_report_status)
            print(f"   📋 Order {order_id}: {order_status}, executed={lots_executed}")

            if lots_executed <= 0:
                return False, f"Opening order not filled: {order_status}"

            from src.stop_orders import place_protection_orders

            try:
                stop_order_ids = place_protection_orders(
                    token=token,
                    account_id=account_id,
                    instrument_uid=instrument_uid,
                    quantity_lots=lots_executed,
                    position_direction=direction,
                    stop_price=stop_loss,
                    take_profit_price=take_profit,
                    min_price_increment=min_price_increment,
                )
            except Exception as protection_error:
                print(f"   ❌ Protective orders failed: {protection_error}")
                close_position(
                    token=token,
                    account_id=account_id,
                    instrument_uid=instrument_uid,
                    direction=direction,
                    quantity_lots=lots_executed,
                )
                return False, f"Protective orders failed; position close requested: {protection_error}"

            if state is not None:
                state["stop_order_ids"] = stop_order_ids
                state["stop_order_id"] = stop_order_ids.get("stop_loss")
                state["entry_lots"] = lots_executed

            return True, f"Opened {direction} {lots_executed} lots with server SL/TP"

    except Exception as e:
        error_msg = f"Failed to open {direction}: {e}"
        print(f"   ❌ {error_msg}")
        return False, error_msg


def close_position(
    token: str,
    account_id: str,
    instrument_uid: str,
    direction: str,
    quantity_lots: int,
    stop_order_ids: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Close an existing position.
    To close LONG -> SELL, to close SHORT -> BUY.

    Args:
        token: API token
        account_id: Account ID
        instrument_uid: Instrument UID
        direction: Current position direction ("LONG" or "SHORT")
        quantity_lots: Number of lots to close

    Returns:
        Tuple of (success, message)
    """
    if not T_TECH_AVAILABLE:
        return False, "t_tech.invest not available"

    if quantity_lots <= 0:
        return False, "Quantity must be > 0"

    if stop_order_ids:
        from src.stop_orders import cancel_protection_orders

        try:
            cancel_protection_orders(
                token=token,
                account_id=account_id,
                stop_order_ids=stop_order_ids.values(),
            )
        except Exception as cancel_error:
            return False, f"Protective order cancellation failed: {cancel_error}"

    # Opposite direction to close
    if direction == "LONG":
        close_direction = OrderDirection.ORDER_DIRECTION_SELL
    elif direction == "SHORT":
        close_direction = OrderDirection.ORDER_DIRECTION_BUY
    else:
        return False, f"Invalid direction: {direction}"

    try:
        with get_client(token) as client:
            _ensure_market_order_available(client, instrument_uid)
            response = retry_api_call(client.orders.post_order)(
                instrument_id=instrument_uid,
                quantity=quantity_lots,
                direction=close_direction,
                account_id=account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
            )

            order_state, lots_executed = wait_for_order_fill(
                client, account_id, response.order_id
            )
            order_status = _status_name(order_state.execution_report_status)
            print(f"   📋 Close order {response.order_id}: {order_status}, executed={lots_executed}")

            if lots_executed <= 0:
                return False, f"Closing order not filled: {order_status}"

            return True, f"Closed {direction} {lots_executed} lots"

    except Exception as e:
        error_msg = f"Failed to close {direction}: {e}"
        print(f"   ❌ {error_msg}")
        return False, error_msg


def execute_signal(
    token: str,
    account_id: str,
    instrument_uid: str,
    instrument,
    action: str,
    position_state: dict,
    current_price: float,
    entry_signal: Optional[str] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    state: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Execute trading action based on strategy signal.

    Args:
        token: API token
        account_id: Account ID
        instrument_uid: Instrument UID
        instrument: Instrument object (with lot size)
        action: Action string (SIGNAL_OPEN_LONG, SIGNAL_EXIT_SL, etc.)
        position_state: Current position state
        current_price: Current market price
        entry_signal: Entry signal direction if applicable

    Returns:
        Tuple of (success, message)
    """
    print(f"\n🔄 Executing action: {action}")

    lot_size = int(getattr(instrument, "lot", 1) or 1)
    min_price_increment = getattr(instrument, "min_price_increment", None)

    # OPEN POSITION
    if action in ("SIGNAL_OPEN_LONG", "SIGNAL_OPEN_SHORT"):
        direction = "LONG" if action == "SIGNAL_OPEN_LONG" else "SHORT"

        # Calculate position size
        balance = get_account_balance(token, account_id)
        quantity_lots = calculate_position_size(
            balance=balance,
            current_price=current_price,
            lot_size=lot_size,
            instrument_uid=instrument_uid,
            token=token,
            account_id=account_id,
        )

        if quantity_lots <= 0:
            return False, "Insufficient capital for position"

        quantity_lots = quantize_lots(quantity_lots * lot_size, lot_size) // lot_size
        if quantity_lots <= 0:
            return False, "Insufficient capital for one complete lot"

        return open_position(
            token=token,
            account_id=account_id,
            instrument_uid=instrument_uid,
            direction=direction,
            quantity_lots=quantity_lots,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            state=state,
            min_price_increment=min_price_increment,
        )

    # CLOSE POSITION
    elif action.startswith("SIGNAL_EXIT_") or action == "EXIT_BE_STOP":
        current_direction = position_state.get("direction", "NONE")
        quantity = position_state.get("quantity", 0)

        if current_direction == "NONE" or quantity <= 0:
            return False, "No position to close"

        # Convert balance to lots (balance is in contracts for futures)
        quantity_lots = int(abs(quantity))
        if quantity_lots <= 0:
            return False, "Position quantity is zero"

        result = close_position(
            token=token,
            account_id=account_id,
            instrument_uid=instrument_uid,
            direction=current_direction,
            quantity_lots=quantity_lots,
            stop_order_ids=(state or {}).get("stop_order_ids", {}),
        )
        if result[0] and state is not None:
            state["stop_order_ids"] = {}
            state["stop_order_id"] = None
        return result

    # NO ACTION
    elif action in ("WAIT", "HOLD_POSITION", "BE_TRIGGERED"):
        return True, f"No trade action needed: {action}"

    else:
        return False, f"Unknown action: {action}"
