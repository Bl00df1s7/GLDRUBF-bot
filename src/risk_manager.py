"""Daily loss circuit breaker for the GLDRUBF trading session."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from src.client_factory import get_client
from src.market_data import quotation_to_float

MOSCOW = ZoneInfo("Europe/Moscow")
SESSION_START = time(3, 0)


@dataclass
class RiskDecision:
    """Result of the daily circuit-breaker evaluation."""

    allowed: bool
    session_date: str
    realized_pnl: float
    loss_pct: float
    reason: str


def session_date(now: Optional[datetime] = None) -> str:
    """Return the trading-session date for a timestamp in Moscow time."""
    current = now or datetime.now(timezone.utc)
    local = current.astimezone(MOSCOW)
    if local.timetz().replace(tzinfo=None) < SESSION_START:
        local = local - timedelta(days=1)
    return local.date().isoformat()


def calculate_realized_pnl(operations: Iterable, figi: str) -> float:
    """Sum executed cash results for the target futures instrument."""
    total = 0.0
    for operation in operations:
        if getattr(operation, "figi", "") != figi:
            continue
        state = getattr(getattr(operation, "state", None), "name", "")
        if state and not state.endswith("EXECUTED"):
            continue
        operation_type = getattr(getattr(operation, "operation_type", None), "name", "")
        if operation_type and not any(
            marker in operation_type
            for marker in ("VARMARGIN", "FUTURE_EXPIRATION", "BROKER_FEE", "SERVICE_FEE")
        ):
            continue
        total += quotation_to_float(getattr(operation, "payment", 0))
    return total


def check_circuit_breaker(
    token: str,
    account_id: str,
    figi: str,
    state: dict,
    max_daily_loss_pct: float,
    now: Optional[datetime] = None,
) -> RiskDecision:
    """Evaluate and persist the daily loss limit for the current session."""
    current_session = session_date(now)
    if state.get("daily_session_date") != current_session:
        state["daily_session_date"] = current_session
        state["daily_start_balance"] = None
        state["TRADING_HALTED"] = False

    with get_client(token) as client:
        if state.get("daily_start_balance") is None:
            portfolio = client.operations.get_portfolio(account_id=account_id)
            state["daily_start_balance"] = quotation_to_float(
                portfolio.total_amount_portfolio
            )

        local_now = (now or datetime.now(timezone.utc)).astimezone(MOSCOW)
        start = datetime.combine(
            datetime.fromisoformat(current_session).date(), SESSION_START, tzinfo=MOSCOW
        )
        if local_now < start:
            start -= timedelta(days=1)
        operations_response = client.operations.get_operations(
            account_id=account_id,
            from_=start.astimezone(timezone.utc),
            to=(now or datetime.now(timezone.utc)),
            figi=figi,
        )

    realized_pnl = calculate_realized_pnl(operations_response.operations, figi)
    start_balance = float(state.get("daily_start_balance") or 0)
    loss_pct = max(0.0, -realized_pnl / start_balance) if start_balance > 0 else 0.0
    halted = bool(state.get("TRADING_HALTED", False)) or loss_pct >= max_daily_loss_pct
    state["TRADING_HALTED"] = halted
    state["daily_realized_pnl"] = realized_pnl
    state["daily_loss_pct"] = loss_pct

    reason = "daily loss limit reached" if halted else "daily loss limit not reached"
    return RiskDecision(halted is False, current_session, realized_pnl, loss_pct, reason)
