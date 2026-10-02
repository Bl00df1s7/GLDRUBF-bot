"""C5 execution adapters: the ONLY place that talks to the broker API.

Stage 2D architecture rule: paper and live runs share one business
logic (src.main.run_cycle → c5_core → margin_provider → protection →
cash_manager) and differ only in the execution adapter injected into
the runtime.

  * LiveExecutionAdapter  — real T-Invest orders (market, full volume);
    every order is confirmed via wait_for_order_fill before any state
    mutation or the next operation is allowed.
  * PaperExecutionAdapter — dry-run mirror of the same interface; it
    never touches the trading API but updates a simulated account so
    the E2E lifecycle can be audited identically.

The adapter performs NO strategy math: quantities arrive already frozen
and sized by c5_core. CNY entry/exit is always ONE operation with the
full quantity; TMON is strictly one bond per operation.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from config.settings import C5_CLASS_CODE, C5_TARGET_TICKER
from src.client_factory import get_client
from src.market_data import quotation_to_float

logger = logging.getLogger("golden_bot.execution")

TMON_TICKER_PREFIX = "TMON"


@dataclass
class AccountSnapshot:
    """Reconciled view of the broker state for one cycle."""

    account_id: str
    equity_rub: float
    free_cash_rub: float
    position_qty: int          # signed contracts of CNYRUBF
    entry_price: Optional[float]
    tmon_qty: int
    open_orders: int


@dataclass
class ExecutionResult:
    ok: bool
    reason: str
    order_id: Optional[str] = None
    qty_executed: int = 0
    details: dict = field(default_factory=dict)


class BaseExecutionAdapter:
    """Shared interface — business logic depends on this contract only."""

    def find_account(self) -> str:
        raise NotImplementedError

    def snapshot(self, account_id: str) -> AccountSnapshot:
        raise NotImplementedError

    def buy_cny(self, account_id: str, qty: int) -> ExecutionResult:
        raise NotImplementedError

    def sell_cny(self, account_id: str, qty: int) -> ExecutionResult:
        raise NotImplementedError

    def sell_one_tmon(self, account_id: str) -> ExecutionResult:
        raise NotImplementedError

    def buy_one_tmon(self, account_id: str) -> ExecutionResult:
        raise NotImplementedError

    def get_free_cash(self, account_id: str) -> float:
        return self.snapshot(account_id).free_cash_rub

    def get_tmon_quantity(self, account_id: str) -> int:
        return self.snapshot(account_id).tmon_qty

    def get_tmon_price(self, account_id: str) -> float:
        raise NotImplementedError


def _find_open_account(client) -> str:
    accounts = client.users.get_accounts().accounts
    for acc in accounts:
        status = getattr(acc, "status", None)
        name = getattr(status, "name", str(status))
        if "OPEN" in name.upper():
            return acc.id
    if not accounts:
        raise RuntimeError("No T-Invest accounts found")
    return accounts[0].id


class LiveExecutionAdapter(BaseExecutionAdapter):
    """Real broker execution. Orders are market, single-shot, confirmed."""

    def __init__(self, token: str, instrument_uid: str,
                 tmon_uid_resolver=None):
        self.token = token
        self.instrument_uid = instrument_uid
        self._tmon_uid = None
        self._resolve_tmon_uid = tmon_uid_resolver or self._default_tmon_resolver

    def _default_tmon_resolver(self, client) -> Optional[str]:
        for fut in client.instruments.futures().instruments:
            if str(fut.ticker).upper().startswith(TMON_TICKER_PREFIX):
                return fut.uid
        return None

    def _tmon_uid_or_raise(self, client) -> str:
        if self._tmon_uid is None:
            self._tmon_uid = self._resolve_tmon_uid(client)
        if not self._tmon_uid:
            raise RuntimeError("TMON instrument not found")
        return self._tmon_uid

    def find_account(self) -> str:
        with get_client(self.token) as client:
            return _find_open_account(client)

    def snapshot(self, account_id: str) -> AccountSnapshot:
        from src.api_retry import retry_api_call

        with get_client(self.token) as client:
            positions = retry_api_call(client.operations.get_positions)(
                account_id=account_id)
            portfolio = retry_api_call(client.operations.get_portfolio)(
                account_id=account_id)

            pos_qty = 0
            entry_price = None
            tmon_qty = 0
            for pos in positions.futures:
                ticker = str(getattr(pos, "ticker", "")).upper()
                if ticker == C5_TARGET_TICKER:
                    pos_qty = int(pos.balance)
                    entry_price = quotation_to_float(
                        getattr(pos, "average_position_price", None))
                elif ticker.startswith(TMON_TICKER_PREFIX):
                    tmon_qty += int(pos.balance)

            cash = sum(
                quotation_to_float(value)
                for value in positions.money
                if getattr(value, "currency", "rub").lower() in ("rub", "ruble")
            )
            blocked = sum(
                quotation_to_float(value)
                for value in positions.blocked
                if getattr(value, "currency", "rub").lower() in ("rub", "ruble")
            )
            free_cash = max(cash - blocked, 0.0)
            equity = quotation_to_float(portfolio.total_amount_portfolio)
            if not (equity and equity > 0):
                equity = free_cash + abs(pos_qty) * (entry_price or 0.0)

            try:
                open_orders = len(retry_api_call(
                    client.orders.get_orders)(account_id=account_id).orders)
            except Exception:
                open_orders = 0

        return AccountSnapshot(
            account_id=account_id,
            equity_rub=float(equity),
            free_cash_rub=float(free_cash),
            position_qty=pos_qty,
            entry_price=float(entry_price) if entry_price else None,
            tmon_qty=tmon_qty,
            open_orders=open_orders,
        )

    def get_tmon_price(self, account_id: str) -> float:
        from src.api_retry import retry_api_call
        from src.market_data import get_current_price

        with get_client(self.token) as client:
            uid = self._tmon_uid_or_raise(client)
        return get_current_price(self.token, uid)

    def _post_market_order(self, instrument_uid: str, account_id: str,
                           qty: int, buy: bool) -> ExecutionResult:
        from t_tech.invest import OrderDirection, OrderType

        from src.api_retry import retry_api_call
        from src.auto_trader import (_ensure_market_order_available,
                                     _status_name, wait_for_order_fill)

        direction = (OrderDirection.ORDER_DIRECTION_BUY if buy
                     else OrderDirection.ORDER_DIRECTION_SELL)
        with get_client(self.token) as client:
            _ensure_market_order_available(client, instrument_uid)
            response = retry_api_call(client.orders.post_order)(
                instrument_id=instrument_uid,
                quantity=int(qty),
                direction=direction,
                account_id=account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
            )
            order_id = response.order_id
            # An order ACCEPTED by the API is NOT executed. Wait for the
            # actual fill report before anything else may happen.
            state, lots_executed = wait_for_order_fill(
                client, account_id, order_id)
            status = _status_name(state.execution_report_status)

        if lots_executed <= 0:
            logger.error("ORDER_NOT_FILLED %s: %s qty=%d status=%s",
                         instrument_uid, "BUY" if buy else "SELL", qty, status)
            return ExecutionResult(False, f"ORDER_NOT_FILLED:{status}",
                                   order_id=order_id)
        return ExecutionResult(True, "FILLED", order_id=order_id,
                               qty_executed=int(lots_executed),
                               details={"status": status})

    def buy_cny(self, account_id: str, qty: int) -> ExecutionResult:
        return self._post_market_order(self.instrument_uid, account_id,
                                       qty, buy=True)

    def sell_cny(self, account_id: str, qty: int) -> ExecutionResult:
        return self._post_market_order(self.instrument_uid, account_id,
                                       qty, buy=False)

    def sell_one_tmon(self, account_id: str) -> ExecutionResult:
        from src.api_retry import retry_api_call
        with get_client(self.token) as client:
            uid = self._tmon_uid_or_raise(client)
        result = self._post_market_order(uid, account_id, 1, buy=False)
        if result.ok:
            logger.info("TMON_SELL confirmed: order=%s qty=1",
                        result.order_id)
        return result

    def buy_one_tmon(self, account_id: str) -> ExecutionResult:
        from src.api_retry import retry_api_call
        with get_client(self.token) as client:
            uid = self._tmon_uid_or_raise(client)
        result = self._post_market_order(uid, account_id, 1, buy=True)
        if result.ok:
            logger.info("TMON_BUY confirmed: order=%s qty=1",
                        result.order_id)
        return result


class PaperExecutionAdapter(BaseExecutionAdapter):
    """Dry-run adapter: identical interface, simulated fills, no orders.

    ``broker`` is an object exposing:
        snapshot(account_id) -> AccountSnapshot
        apply(order_dict)    -> None   (mutates simulated state)
    Tests inject their own fake broker; production paper mode uses
    SimulatedBroker built from a real read-only snapshot at startup.
    """

    def __init__(self, broker, account_id: str = "paper-account"):
        self.broker = broker
        self.account_id = account_id
        self.orders = []

    def find_account(self) -> str:
        return self.account_id

    def snapshot(self, account_id: str) -> AccountSnapshot:
        return self.broker.snapshot(account_id)

    def get_tmon_price(self, account_id: str) -> float:
        return float(getattr(self.broker, "tmon_price", 1000.0))

    def _apply(self, instrument: str, side: str, qty: int,
               price_ref: float) -> ExecutionResult:
        order = {"instrument": instrument, "side": side, "qty": int(qty)}
        self.orders.append(order)
        self.broker.apply(order)
        logger.info("PAPER_FILL %s %s x%d", instrument, side, qty)
        return ExecutionResult(True, "PAPER_FILLED",
                               order_id=f"paper-{len(self.orders)}",
                               qty_executed=int(qty))

    def buy_cny(self, account_id: str, qty: int) -> ExecutionResult:
        return self._apply(C5_TARGET_TICKER, "BUY", qty, 0.0)

    def sell_cny(self, account_id: str, qty: int) -> ExecutionResult:
        return self._apply(C5_TARGET_TICKER, "SELL", qty, 0.0)

    def sell_one_tmon(self, account_id: str) -> ExecutionResult:
        return self._apply(TMON_TICKER_PREFIX, "SELL", 1,
                          self.get_tmon_price(account_id))

    def buy_one_tmon(self, account_id: str) -> ExecutionResult:
        return self._apply(TMON_TICKER_PREFIX, "BUY", 1,
                          self.get_tmon_price(account_id))


class SimulatedBroker:
    """Minimal in-memory broker used by PAPER mode and integration tests."""

    def __init__(self, cash: float = 0.0, tmon_qty: int = 0,
                 position_qty: int = 0, equity: float = 0.0,
                 tmon_price: float = 1000.0, entry_price: float = 0.0):
        self.cash = cash
        self.tmon_qty = tmon_qty
        self.position_qty = position_qty
        self.equity = equity or cash
        self.tmon_price = tmon_price
        self.entry_price = entry_price

    def snapshot(self, account_id: str) -> AccountSnapshot:
        return AccountSnapshot(
            account_id=account_id,
            equity_rub=self.equity,
            free_cash_rub=self.cash,
            position_qty=self.position_qty,
            entry_price=self.entry_price or None,
            tmon_qty=self.tmon_qty,
            open_orders=0,
        )

    def apply(self, order: dict) -> None:
        side, qty = order["side"], int(order["qty"])
        if order["instrument"] == C5_TARGET_TICKER:
            delta = qty if side == "BUY" else -qty
            self.position_qty += delta
            self.cash -= qty * 87_750.0 if side == "BUY" else qty * 87_750.0
        elif order["instrument"].startswith(TMON_TICKER_PREFIX):
            if side == "SELL":
                self.tmon_qty -= qty
                self.cash += qty * self.tmon_price
            else:
                self.tmon_qty += qty
                self.cash -= qty * self.tmon_price
        self.equity = max(self.equity, 0.0)


def build_execution_adapter(token: str, instrument_uid: str):
    """Factory honoring MODE env: LIVE (default) / PAPER.

    LIVE must NEVER silently degrade to paper: adapter choice is explicit.
    """
    mode = os.environ.get("TRADING_MODE", "LIVE").upper()
    if mode == "PAPER":
        return PaperExecutionAdapter(SimulatedBroker()), "PAPER"
    if mode == "LIVE":
        return LiveExecutionAdapter(token, instrument_uid), "LIVE"
    raise ValueError(f"Unknown TRADING_MODE: {mode!r} (expected LIVE|PAPER)")
