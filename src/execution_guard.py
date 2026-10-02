"""C5 execution guard: pre-entry market quality checks (CNYRUBF_SPBFUT).

Checks before ANY entry: bid, ask, spread, depth, expected execution
price, expected slippage, margin rate presence, equity, M/E.

If expected_slippage > 0.10% → NO_ENTRY with the reason logged.
The guard never mutates strategy parameters and never auto-corrects.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import C5_MAX_EXPECTED_SLIPPAGE
from src.client_factory import get_client
from src.market_data import quotation_to_float

logger = logging.getLogger("golden_bot.execution_guard")

NO_ENTRY = "NO_ENTRY"


@dataclass
class ExecutionGuardResult:
    """Snapshot of order-book quality plus the pass/fail decision."""

    allowed: bool
    reason: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    spread_pct: float = 0.0
    expected_price: Optional[float] = None
    expected_slippage_pct: float = 0.0
    depth_qty: int = 0
    failed_checks: List[str] = field(default_factory=list)


def fetch_order_book(token: str, instrument_uid: str, depth: int = 10):
    """Return (bids, asks) as [(price, qty), ...] using existing infra."""
    with get_client(token) as client:
        response = client.market_data.get_order_book(
            figi="", uid=instrument_uid, limit=depth
        )
    bids = [
        (quotation_to_float(l.price), int(l.quantity))
        for l in response.bids
        if getattr(l, "quantity", 0)
    ]
    asks = [
        (quotation_to_float(l.price), int(l.quantity))
        for l in response.asks
        if getattr(l, "quantity", 0)
    ]
    return bids, asks


def evaluate_execution_guard(
    *,
    direction: str,
    bids,
    asks,
    required_qty: int,
    max_slippage: float = C5_MAX_EXPECTED_SLIPPAGE,
    margin_available: bool = True,
    equity_available: bool = True,
    me_ok: bool = True,
) -> ExecutionGuardResult:
    """Pure evaluation of book quality + prerequisite flags.

    ``bids``/``asks`` are [(price, qty)] lists; sizing uses the same
    convention as src.sizing.compute_liquidity_qty.
    """
    failed: List[str] = []

    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None

    if not margin_available:
        failed.append("MARGIN_API_UNAVAILABLE")
    if not equity_available:
        failed.append("EQUITY_UNAVAILABLE")
    if not me_ok:
        failed.append("ME_LIMIT_EXCEEDED")
    if bid is None or ask is None:
        failed.append("NO_BID_OR_ASK")
        result = ExecutionGuardResult(False, NO_ENTRY, bid, ask,
                                      failed_checks=failed)
        logger.error("EXECUTION_GUARD %s: %s", NO_ENTRY, ", ".join(failed))
        return result
    if bid <= 0 or ask < bid:
        failed.append("INVALID_QUOTE")

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid if mid > 0 else float("inf")

    # Expected execution price: walk the side we trade against.
    book = asks if direction == "LONG" else bids
    filled = 0.0
    cost = 0.0
    for price, qty in book:
        take = min(float(qty), required_qty - filled)
        if take <= 0:
            break
        filled += take
        cost += price * take
    if filled <= 0:
        failed.append("EMPTY_DEPTH")
        expected_price = None
        expected_slippage = float("inf")
    else:
        vwap = cost / filled
        expected_price = vwap
        expected_slippage = abs(vwap - mid) / mid

    depth_qty = int(sum(q for _, q in book))

    if filled < required_qty:
        failed.append("INSUFFICIENT_DEPTH")
    if expected_slippage > max_slippage:
        failed.append(
            f"SLIPPAGE_GUARD:{expected_slippage:.4%}>{max_slippage:.2%}"
        )

    allowed = not failed
    reason = "OK" if allowed else NO_ENTRY

    result = ExecutionGuardResult(
        allowed=allowed,
        reason=reason,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_pct=spread_pct,
        expected_price=expected_price,
        expected_slippage_pct=expected_slippage,
        depth_qty=depth_qty,
        failed_checks=failed,
    )

    if not allowed:
        logger.error(
            "EXECUTION_GUARD %s before %s entry: %s "
            "(bid=%s ask=%s spread=%.4f%% slippage=%s depth=%d)",
            NO_ENTRY, direction, "; ".join(failed),
            bid, ask, spread_pct * 100,
            f"{expected_slippage:.4%}" if expected_price else "n/a",
            depth_qty,
        )
    else:
        logger.info(
            "EXECUTION_GUARD passed for %s x%d: bid=%s ask=%s "
            "expected_price=%s slippage=%.4f%%",
            direction, required_qty, bid, ask,
            expected_price, expected_slippage * 100,
        )
    return result
