"""C5 cash manager: TMON funding loop + excess-cash parking (CNYRUBF).

The CNYRUBF contract is margined in RUB. When free RUB cash is not
enough to cover the required ГО for a C5 ENTRY, the runtime liquidates
TMON_N (MOEX funds-repository) bonds — strictly ONE bond per operation —
waiting for actual execution confirmation and refreshing position/cash
after every single fill before the next order is allowed.

TMON is never a signal source; it only frees cash for the frozen C5
strategy decision made upstream.

Guards:
  * MAX_TMON_SELLS caps one liquidation cycle; when the cap is reached
    and cash is still insufficient → NO_ENTRY with an explicit reason
    (INSUFFICIENT_CASH_AFTER_TMON_LIQUIDATION), no partial entry, no
    fallback.
  * Every step emits a structured diagnostic event
    (TMON_SELL / TMON_BUY / POSITION_REFRESH / ENTRY_BLOCKED).
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("golden_bot.cash_manager")

REASON_FUNDED = "FUNDED"
REASON_NO_TMON = "NO_TMON_POSITION"
REASON_CAP_REACHED = "INSUFFICIENT_CASH_AFTER_TMON_LIQUIDATION"
MAX_TMON_SELLS = 20


@dataclass
class CashResult:
    """Outcome of one funding / parking cycle."""

    ok: bool
    reason: str
    operations: int
    cash_after: float


def ensure_cash_for_entry(
    trader,
    account_id: str,
    required_margin_rub: float,
    *,
    reserve_rub: float = 0.0,
    max_sells: int = MAX_TMON_SELLS,
) -> CashResult:
    """Liquidate TMON one-by-one until cash >= required + reserve.

    After EACH submitted order the adapter waits for the actual fill
    (wait_for_order_fill) and refreshes the live cash/position snapshot
    before deciding whether another TMON SELL is needed.
    """
    target = required_margin_rub + reserve_rub
    cash = trader.get_free_cash(account_id)
    sells = 0

    while cash < target:
        tmon_qty = trader.get_tmon_quantity(account_id)
        if tmon_qty < 1:
            logger.error(
                "ENTRY_BLOCKED %s: cash=%.2f < required=%.2f, TMON=%d",
                REASON_NO_TMON, cash, target, int(tmon_qty),
            )
            return CashResult(False, REASON_NO_TMON, sells, cash)
        if sells >= max_sells:
            logger.error(
                "ENTRY_BLOCKED %s: cash=%.2f after %d sells, required=%.2f",
                REASON_CAP_REACHED, cash, sells, target,
            )
            return CashResult(False, REASON_CAP_REACHED, sells, cash)

        # One bond per operation; confirm the fill before continuing.
        sold = trader.sell_one_tmon(account_id)
        if not sold:
            logger.error("ENTRY_BLOCKED: TMON_SELL_%d_NOT_CONFIRMED", sells + 1)
            return CashResult(False, f"TMON_SELL_{sells + 1}_NOT_CONFIRMED",
                              sells, cash)
        sells += 1
        cash = trader.get_free_cash(account_id)  # POSITION_REFRESH
        logger.info(
            "TMON_SELL #%d confirmed; cash refreshed: %.2f (target %.2f)",
            sells, cash, target,
        )

    return CashResult(True, REASON_FUNDED, sells, cash)


def park_excess_cash(
    trader,
    account_id: str,
    keep_cash_rub: float,
    price_per_bond_rub: float,
    *,
    max_buys: int = MAX_TMON_SELLS,
    min_buffer_rub: float = 0.0,
) -> CashResult:
    """Buy TMON bonds one-by-one with surplus above ``keep_cash_rub``.

    Each BUY is confirmed and the cash snapshot refreshed before the
    next order. This is opportunistic idle-cash management: any failure
    simply stops the loop (the cash stays on the account).
    """
    buys = 0
    if price_per_bond_rub <= 0:
        return CashResult(False, "TMON_PRICE_UNAVAILABLE", 0,
                          trader.get_free_cash(account_id))

    cash = trader.get_free_cash(account_id)
    while cash - keep_cash_rub >= price_per_bond_rub + min_buffer_rub:
        if buys >= max_buys:
            break
        bought = trader.buy_one_tmon(account_id)
        if not bought:
            logger.warning("TMON_BUY #%d not confirmed, stopping parking",
                           buys + 1)
            break
        buys += 1
        cash = trader.get_free_cash(account_id)
        logger.info("TMON_BUY #%d confirmed; cash refreshed: %.2f",
                    buys, cash)

    return CashResult(True, "PARKED" if buys else "NO_EXCESS", buys, cash)
