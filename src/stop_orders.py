"""Server-side protective stop and take-profit orders."""

from typing import Dict, Iterable, List, Optional

from t_tech.invest import (
    StopOrderDirection,
    StopOrderExpirationType,
    StopOrderType,
)
from t_tech.invest.schemas import Quotation

from src.client_factory import get_client
from src.instruments import quantize_price


def quotation_from_float(value: float) -> Quotation:
    """Convert a decimal price to the SDK Quotation type."""
    units = int(value)
    nano = int(round((value - units) * 1_000_000_000))
    if nano >= 1_000_000_000:
        units += 1
        nano -= 1_000_000_000
    if nano < 0:
        units -= 1
        nano += 1_000_000_000
    return Quotation(units=units, nano=nano)


def _close_direction(direction: str) -> StopOrderDirection:
    """Return the stop-order direction that closes a position."""
    if direction == "LONG":
        return StopOrderDirection.STOP_ORDER_DIRECTION_SELL
    if direction == "SHORT":
        return StopOrderDirection.STOP_ORDER_DIRECTION_BUY
    raise ValueError(f"Invalid position direction: {direction}")


def place_protection_orders(
    token: str,
    account_id: str,
    instrument_uid: str,
    quantity_lots: int,
    position_direction: str,
    stop_price: float,
    take_profit_price: float,
    min_price_increment=None,
) -> Dict[str, str]:
    """Create server-side SL and TP orders for an open futures position."""
    if quantity_lots <= 0:
        raise ValueError("quantity_lots must be positive")
    if stop_price <= 0 or take_profit_price <= 0:
        raise ValueError("stop_price and take_profit_price must be positive")
    if min_price_increment is not None:
        stop_price = quantize_price(stop_price, min_price_increment)
        take_profit_price = quantize_price(take_profit_price, min_price_increment)

    close_direction = _close_direction(position_direction)
    common = {
        "instrument_id": instrument_uid,
        "quantity": quantity_lots,
        "direction": close_direction,
        "account_id": account_id,
        "expiration_type": StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
    }

    with get_client(token) as client:
        stop_response = client.stop_orders.post_stop_order(
            **common,
            stop_price=quotation_from_float(stop_price),
            stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
        )
        try:
            take_response = client.stop_orders.post_stop_order(
                **common,
                stop_price=quotation_from_float(take_profit_price),
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
            )
        except Exception:
            client.stop_orders.cancel_stop_order(
                account_id=account_id,
                stop_order_id=stop_response.stop_order_id,
            )
            raise

    return {
        "stop_loss": stop_response.stop_order_id,
        "take_profit": take_response.stop_order_id,
    }


def cancel_protection_orders(
    token: str,
    account_id: str,
    stop_order_ids: Iterable[str],
) -> List[str]:
    """Cancel stored protective stop-order IDs and return cancelled IDs."""
    ids = [stop_order_id for stop_order_id in stop_order_ids if stop_order_id]
    if not ids:
        return []

    cancelled: List[str] = []
    with get_client(token) as client:
        for stop_order_id in ids:
            client.stop_orders.cancel_stop_order(
                account_id=account_id,
                stop_order_id=stop_order_id,
            )
            cancelled.append(stop_order_id)
    return cancelled


def replace_protection_orders(
    token: str,
    account_id: str,
    instrument_uid: str,
    quantity_lots: int,
    position_direction: str,
    old_stop_order_ids: Iterable[str],
    stop_price: float,
    take_profit_price: float,
    min_price_increment=None,
) -> Dict[str, str]:
    """Replace server-side protection, rolling back cancellation when possible."""
    cancel_protection_orders(token, account_id, old_stop_order_ids)
    return place_protection_orders(
        token=token,
        account_id=account_id,
        instrument_uid=instrument_uid,
        quantity_lots=quantity_lots,
        position_direction=position_direction,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        min_price_increment=min_price_increment,
    )


def get_open_protection_orders(token: str, account_id: str) -> list:
    """Return currently active stop-orders for an account."""
    with get_client(token) as client:
        return client.stop_orders.get_stop_orders(account_id=account_id).stop_orders
