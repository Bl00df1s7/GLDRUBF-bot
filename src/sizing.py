"""C5 position sizing: risk / margin / liquidity quantity limits.

    risk_qty      = floor(risk_budget / per-contract price risk)
    margin_qty    = floor(available_margin_capacity / live ГО per contract)
    liquidity_qty = contracts absorbable within the slippage guard
    final_qty     = min(risk_qty, margin_qty, liquidity_qty, MAX_QTY)
                    then capped by the Stage-1 deployment limit.

All margin inputs must be LIVE values from src.margin_provider; a missing
margin API blocks entry upstream (NO ENTRY), it never falls back here.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    C5_ATR_TARGET,
    C5_CONTRACT_SIZE_CNY,
    C5_MAX_EXPECTED_SLIPPAGE,
    C5_MAX_QTY,
    C5_ME_LIMIT,
    C5_RISK_BUDGET_PCT,
    C5_RISK_MULT_MAX,
    C5_RISK_MULT_MIN,
    C5_STAGE_MAX_QTY,
)


@dataclass
class SizingResult:
    """Full audit trail of one sizing computation."""

    allowed: bool
    reason: str
    risk_multiplier: float = 0.0
    risk_budget_rub: float = 0.0
    risk_qty: int = 0
    margin_qty: int = 0
    liquidity_qty: int = 0
    calculated_qty: int = 0
    final_qty: int = 0
    required_margin_rub: float = 0.0
    me_ratio: float = 0.0
    expected_slippage_pct: float = 0.0
    details: dict = field(default_factory=dict)


def compute_risk_multiplier(atr14: float) -> float:
    """Frozen ATR adjustment: clip(ATR_target / ATR14, 0.5, 1.5).

    Direction matters and is frozen:
      * ATR14 < target  → multiplier > 1 (lower volatility → higher risk)
      * ATR14 > target  → multiplier < 1 (higher volatility → lower risk)
    """
    if atr14 is None or atr14 <= 0 or math.isnan(atr14):
        raise ValueError(f"Invalid ATR14 for risk multiplier: {atr14!r}")
    ratio = C5_ATR_TARGET / atr14
    return max(C5_RISK_MULT_MIN, min(C5_RISK_MULT_MAX, ratio))


def compute_liquidity_qty(
    quantity_guess: int,
    notional_per_contract_rub: float,
    book_asks,
    direction: str,
    mid_price: float,
    max_slippage: float = C5_MAX_EXPECTED_SLIPPAGE,
) -> int:
    """Max contracts fillable inside the slippage guard using book depth.

    book_asks: iterable of (price, quantity) on the side we would trade
    against (asks when buying, bids when selling). Walks levels until the
    volume-weighted execution price drifts more than ``max_slippage``
    from the reference mid price.
    """
    if not book_asks or not mid_price or mid_price <= 0:
        return 0

    filled_qty = 0.0
    cost_weighted_price = 0.0
    best_limit = mid_price * (1 + max_slippage) if direction == "LONG" \
        else mid_price * (1 - max_slippage)

    for level in book_asks:
        try:
            price = float(level[0])
            qty = float(level[1])
        except (TypeError, IndexError, ValueError):
            continue
        if qty <= 0:
            continue
        # Ignore obviously bogus levels relative to notional scale.
        if price <= 0:
            continue
        if direction == "LONG" and price > best_limit:
            break
        if direction == "SHORT" and price < best_limit:
            break

        take = min(qty, quantity_guess - filled_qty)
        if take <= 0:
            break
        filled_qty += take
        cost_weighted_price += price * take
        if filled_qty >= quantity_guess:
            break

    if filled_qty <= 0:
        return 0

    vwap = cost_weighted_price / filled_qty
    expected_slippage = abs(vwap - mid_price) / mid_price
    if expected_slippage > max_slippage:
        return 0
    return int(math.floor(filled_qty))


def size_position(
    *,
    equity_rub: float,
    atr14: float,
    price_rub: float,
    direction: str,
    margin_per_contract_rub: float,
    current_position_qty: int = 0,
    cny_rub_rate: Optional[float] = None,
    book_asks=None,
    mid_price: Optional[float] = None,
    contract_size_cny: float = C5_CONTRACT_SIZE_CNY,
    max_qty: int = C5_MAX_QTY,
    stage_max_qty: int = C5_STAGE_MAX_QTY,
    me_limit: float = C5_ME_LIMIT,
) -> SizingResult:
    """Compute final_qty with full audit trail. Never raises to fallback.

    Args:
        equity_rub: account equity in RUB.
        atr14: ATR14 of last closed candle (instrument price units).
        price_rub: current instrument price in RUB.
        direction: "LONG" | "SHORT".
        margin_per_contract_rub: LIVE dlongClient/dshortClient value.
        current_position_qty: absolute contracts already held (0 at entry).
        cny_rub_rate: CNY/RUB FX rate for contract notional conversion.
        book_asks: depth on the traded side [(price, qty), ...].
        mid_price: reference mid price for slippage measurement.
    """
    if equity_rub is None or equity_rub <= 0:
        return SizingResult(False, "NO_EQUITY")
    if price_rub is None or price_rub <= 0:
        return SizingResult(False, "INVALID_PRICE")
    if margin_per_contract_rub is None or margin_per_contract_rub <= 0:
        # Explicitly NOT a fallback: caller must block entry on API failure.
        return SizingResult(False, "MARGIN_API_UNAVAILABLE")
    # CNY/RUB rate is needed only for the liquidity/slippage depth walk;
    # it never participates in the frozen risk budget (§7 of C5 spec).
    fx_ok = cny_rub_rate is not None and cny_rub_rate > 0

    try:
        multiplier = compute_risk_multiplier(atr14)
    except ValueError as exc:
        return SizingResult(False, "ATR_UNAVAILABLE", details={"error": str(exc)})

    # Frozen risk model (C5 spec §7–§8):
    #   risk_budget = equity * RISK_BASE * clip(ATR_target/ATR14, 0.5, 1.5)
    #   qty_risk    = floor(risk_budget / live ГО per contract)
    # The ГО-based denominator makes higher volatility shrink the size
    # automatically (multiplier < 1), matching the economic meaning.
    risk_budget = equity_rub * C5_RISK_BUDGET_PCT * multiplier
    risk_qty = int(math.floor(risk_budget / margin_per_contract_rub)) \
        if margin_per_contract_rub > 0 else 0

    # margin capacity under the operational M/E safety limit.
    available_margin_rub = max(0.0, equity_rub * me_limit
                               - current_position_qty * margin_per_contract_rub)
    margin_qty = int(math.floor(available_margin_rub / margin_per_contract_rub))

    # engineering / liquidity cap
    guess = max(risk_qty, 0)
    if fx_ok and book_asks and mid_price:
        liquidity_qty = compute_liquidity_qty(
            quantity_guess=max(guess, 1),
            notional_per_contract_rub=contract_size_cny * cny_rub_rate,
            book_asks=book_asks,
            direction=direction,
            mid_price=mid_price,
        )
    else:
        # No depth data supplied: the execution guard (src.execution_guard)
        # performs the mandatory slippage check before any order is sent;
        # here liquidity defaults to the engineering cap.
        liquidity_qty = int(max_qty)

    calculated = min(risk_qty, margin_qty, liquidity_qty, max_qty)
    final = min(calculated, stage_max_qty)

    if final <= 0:
        limiting = []
        if risk_qty <= 0:
            limiting.append("risk_qty")
        if margin_qty <= 0:
            limiting.append("margin_qty")
        if liquidity_qty <= 0:
            limiting.append("liquidity_qty")
        return SizingResult(
            False,
            f"NO_ENTRY_QTY_ZERO:{'+'.join(limiting) or 'cap'}",
            risk_multiplier=multiplier,
            risk_budget_rub=risk_budget,
            risk_qty=risk_qty,
            margin_qty=margin_qty,
            liquidity_qty=liquidity_qty,
        )

    projected_qty = current_position_qty + final
    required_margin = projected_qty * margin_per_contract_rub
    me_ratio = required_margin / equity_rub
    if me_ratio > me_limit:
        return SizingResult(
            False,
            f"ME_LIMIT_EXCEEDED:{me_ratio:.4f}>{me_limit:.2f}",
            risk_multiplier=multiplier,
            risk_budget_rub=risk_budget,
            risk_qty=risk_qty,
            margin_qty=margin_qty,
            liquidity_qty=liquidity_qty,
            required_margin_rub=required_margin,
            me_ratio=me_ratio,
        )

    return SizingResult(
        True,
        "OK",
        risk_multiplier=multiplier,
        risk_budget_rub=risk_budget,
        risk_qty=risk_qty,
        margin_qty=margin_qty,
        liquidity_qty=liquidity_qty,
        calculated_qty=calculated,
        final_qty=final,
        required_margin_rub=required_margin,
        me_ratio=me_ratio,
    )
