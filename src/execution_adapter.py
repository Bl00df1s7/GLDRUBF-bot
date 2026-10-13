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

            # Fail closed (open_orders trustworthiness fix): a get_orders()
            # error must NOT be masked as open_orders=0 — that would let
            # reconcile() read an unknown state as "confirmed absence of
            # working orders".  The exception propagates unchanged, exactly
            # like the sibling get_positions()/get_portfolio() calls above;
            # bootstrap() then marks reconcile FAILED and no execution is
            # enabled (STARTUP_RECONCILE_FAILED).
            open_orders = len(retry_api_call(
                client.orders.get_orders)(account_id=account_id).orders)

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

        # STAGE 5A DEFENSE-IN-DEPTH gate — re-checked immediately before
        # the ONLY OrdersServiceApi.post_order call site of this adapter.
        # AUTO_TRADING_ENABLED!=true blocks real order flow regardless of
        # TRADING_MODE; PAPER mode never constructs a LiveExecutionAdapter
        # (build_execution_adapter routes PAPER → PaperExecutionAdapter),
        # so PAPER can physically not reach this line. Fail closed.
        from src.runtime_config import (TradingConfigError,
                                        enforce_trading_gate,
                                        load_trading_config)
        try:
            enforce_trading_gate(load_trading_config())
        except TradingConfigError as exc:
            logger.error("LIVE_ORDER_BLOCKED (fail closed): %s", exc)
            return ExecutionResult(False, f"TRADING_BLOCKED:{exc}")

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
               price_ref: float, fill_qty=None, **extra) -> ExecutionResult:
        # ``fill_qty`` (G-9C) lets tests/fakes simulate a PARTIAL fill of
        # an order SENT with ``qty`` contracts: the broker applies the
        # whole dispatched order (its position/margin accounting mirrors
        # what the exchange does for that order), while the execution
        # report carries ONLY the confirmed executed volume.  Default:
        # full fill — identical to the pre-G-9C behaviour.
        if fill_qty is None:
            fill_qty = qty
        fill_qty = int(fill_qty)
        order = {"instrument": instrument, "side": side, "qty": int(qty),
                 **extra}
        self.orders.append(order)
        self.broker.apply(order)
        logger.info("PAPER_FILL %s %s x%d (executed %d)",
                    instrument, side, qty, fill_qty)
        return ExecutionResult(True, "PAPER_FILLED",
                               order_id=f"paper-{len(self.orders)}",
                               qty_executed=fill_qty)

    def buy_cny(self, account_id: str, qty: int,
                margin_per_contract: Optional[float] = None,
                entry_price: Optional[float] = None,
                intent: Optional[str] = None,
                fill_qty=None) -> ExecutionResult:
        # ГО per contract (already computed by the runtime from the live
        # FutureBy rate — see c5_runtime._evaluate_entry) is forwarded to
        # SimulatedBroker as PAPER margin reservation metadata.  Passing
        # None here means "caller has no ГО"; SimulatedBroker FAILS CLOSED
        # on a futures BUY without it (no fallback constant allowed).
        # ``intent`` (G-9C) is an OPTIONAL order-metadata tag only — old
        # callers that omit it keep working unchanged.
        return self._apply(C5_TARGET_TICKER, "BUY", qty, 0.0,
                           fill_qty=fill_qty,
                           margin_per_contract=margin_per_contract,
                           entry_price=entry_price,
                           **({"intent": intent} if intent else {}))

    def sell_cny(self, account_id: str, qty: int,
                 margin_per_contract: Optional[float] = None,
                 entry_price: Optional[float] = None,
                 intent: Optional[str] = None,
                 fill_qty=None) -> ExecutionResult:
        # Legacy EXIT path (closing a LONG) keeps the exact two-argument
        # call signature — no extra metadata, identical order dict shape.
        # G-9C: opening a SHORT is ALSO a SELL of CNYRUBF contracts; the
        # runtime then forwards the live ГО / entry price so
        # SimulatedBroker can reserve margin and record the basis, and
        # tags the order with its durable intent ("OPEN_SHORT") so the
        # broker can tell an opening SELL from a closing one.
        extra = {}
        if intent:
            extra["intent"] = intent
        if margin_per_contract is not None:
            extra["margin_per_contract"] = margin_per_contract
        if entry_price is not None:
            extra["entry_price"] = entry_price
        return self._apply(C5_TARGET_TICKER, "SELL", qty, 0.0,
                           fill_qty=fill_qty, **extra)

    def sell_one_tmon(self, account_id: str) -> ExecutionResult:
        return self._apply(TMON_TICKER_PREFIX, "SELL", 1,
                          self.get_tmon_price(account_id))

    def buy_one_tmon(self, account_id: str) -> ExecutionResult:
        return self._apply(TMON_TICKER_PREFIX, "BUY", 1,
                          self.get_tmon_price(account_id))


class SimulatedBrokerError(RuntimeError):
    """PAPER fill refused by the simulated broker (fail closed).

    Raised when a futures BUY arrives without ``margin_per_contract`` in
    the order metadata or when free cash cannot cover the ГО — the
    broker never guesses a margin constant and never lets ``cash`` go
    negative on a margined entry.
    """


class SimulatedBroker:
    """Minimal in-memory broker used by PAPER mode and integration tests.

    Futures accounting mirrors what the real broker exposes: ``cash`` is
    AVAILABLE/free funds, buying a contract blocks only its initial
    margin (ГО), never the notional value; ``equity`` stays at the
    seeded snapshot value (MTM/P&L is out of scope for this model).
    """

    def __init__(self, cash: float = 0.0, tmon_qty: int = 0,
                 position_qty: int = 0, equity: float = 0.0,
                 tmon_price: float = 1000.0, entry_price: float = 0.0):
        self.cash = cash
        self.tmon_qty = tmon_qty
        self.position_qty = position_qty
        self.equity = equity or cash
        self.tmon_price = tmon_price
        self.entry_price = entry_price
        # ГО per contract blocked by the current long (set at BUY time
        # from the runtime-computed live margin; 0.0 = unknown basis).
        self.margin_locked_per_contract = 0.0
        # G-9C: symmetric field for the short side — ГО per contract
        # blocked by an OPEN_SHORT sell (0.0 = no short basis known).
        self.short_margin_locked_per_contract = 0.0

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

    def _reserve_short_margin(self, qty: int, mpc) -> None:
        """G-9C: reserve the initial margin (ГО) for a newly OPENED short.

        Same fail-closed rules as the long BUY path: the ГО must arrive in
        the order metadata (runtime-computed from the live FutureBy rate),
        a non-positive margin or insufficient free cash refuses the fill —
        the broker never guesses a constant and never lets ``cash`` go
        negative on a margined entry.
        """
        try:
            total_margin = float(mpc) * qty
        except (TypeError, ValueError):
            raise SimulatedBrokerError(
                f"PAPER futures SHORT open x{qty} requires "
                f"margin_per_contract in the order metadata, got "
                f"{mpc!r} — refusing to guess ГО (fail closed)")
        if total_margin <= 0:
            raise SimulatedBrokerError(
                f"PAPER futures SHORT open x{qty}: non-positive total "
                f"margin {total_margin}")
        if self.cash < total_margin:
            raise SimulatedBrokerError(
                f"PAPER futures SHORT open x{qty} rejected: cash "
                f"{self.cash:.2f} < required margin "
                f"{total_margin:.2f}")
        self.cash -= total_margin
        self.short_margin_locked_per_contract = total_margin / qty

    def apply(self, order: dict) -> None:
        side, qty = order["side"], int(order["qty"])
        if order["instrument"] == C5_TARGET_TICKER:
            # Futures PAPER accounting (broker semantics): buying a
            # CNYRUBF contract NEVER costs its notional value — the
            # exchange blocks only the initial margin (ГО).  ``cash`` is
            # available/free funds; a BUY reserves ``qty × ГО/contract``,
            # an EXIT releases it back.  The ГО must come from the order
            # metadata (runtime-computed from the live FutureBy rate);
            # fallback constants are forbidden — fail closed instead.
            # G-9C: an OPENING SELL (ENTER_SHORT routed through SELL) is
            # the symmetric case — it reserves ГО for the new short and
            # drives ``position_qty`` NEGATIVE; closing/reducing a LONG
            # stays exactly on the legacy path below.
            if qty <= 0:
                raise SimulatedBrokerError(
                    f"PAPER futures {side} with non-positive qty {qty}")
            intent = order.get("intent")
            opening_short = (side == "SELL"
                             and str(intent or "").upper() == "OPEN_SHORT")
            if opening_short:
                if self.position_qty >= 1:
                    # An OPEN_SHORT order may NEVER flip an existing long
                    # into a short (no accidental position reversal via
                    # an opening order): close the LONG first (separate
                    # slice), refuse here fail-closed.
                    raise SimulatedBrokerError(
                        f"PAPER OPEN_SHORT x{qty} refused while a LONG "
                        f"position ({self.position_qty}) is open — an "
                        "opening order must never reverse the position")
                if self.position_qty <= -1:
                    # A short is already open: a second opening order
                    # would stack/reserve without a managed basis — block
                    # until SHORT close-out is implemented (next slice).
                    raise SimulatedBrokerError(
                        f"PAPER OPEN_SHORT x{qty} refused while a SHORT "
                        f"position ({self.position_qty}) is already open")
                self._reserve_short_margin(qty, order.get("margin_per_contract"))
                ep = order.get("entry_price")
                if ep is not None:
                    self.entry_price = float(ep)
                self.position_qty -= qty
            elif side == "BUY":
                if self.position_qty <= -1:
                    # Closing a SHORT is also a BUY — but that is the NEXT
                    # dedicated slice; silently adding to the short (a
                    # random reversal) is forbidden fail-closed.
                    raise SimulatedBrokerError(
                        f"PAPER futures BUY x{qty} refused while a SHORT "
                        f"position ({self.position_qty}) is open — SHORT "
                        "close-out via BUY is not implemented yet")
                mpc = order.get("margin_per_contract")
                try:
                    total_margin = float(mpc) * qty
                except (TypeError, ValueError):
                    raise SimulatedBrokerError(
                        f"PAPER futures BUY x{qty} requires "
                        f"margin_per_contract in the order metadata, got "
                        f"{mpc!r} — refusing to guess ГО (fail closed)")
                if total_margin <= 0:
                    raise SimulatedBrokerError(
                        f"PAPER futures BUY x{qty}: non-positive total "
                        f"margin {total_margin}")
                if self.cash < total_margin:
                    raise SimulatedBrokerError(
                        f"PAPER futures BUY x{qty} rejected: cash "
                        f"{self.cash:.2f} < required margin "
                        f"{total_margin:.2f}")
                self.cash -= total_margin
                # Remember the per-contract reservation so EXIT can
                # release exactly what was blocked (last BUY wins while
                # flat; partial re-entries are out of scope for PAPER).
                self.margin_locked_per_contract = total_margin / qty
                ep = order.get("entry_price")
                if ep is not None:
                    self.entry_price = float(ep)
                self.position_qty += qty
            else:
                # SELL releases the ГО that the long BUY blocked.  The
                # per-contract reservation is remembered at BUY time; a
                # legacy/unknown cost basis releases nothing rather than
                # inventing a fallback constant (fail-safe, no free cash
                # printed out of thin air).
                held = max(0, self.position_qty)
                if qty > held:
                    # A plain (closing) SELL may NEVER sell more contracts
                    # than the long actually held — beyond-flat would be an
                    # accidental reversal into a short.  Refuse fail-closed.
                    raise SimulatedBrokerError(
                        f"PAPER futures SELL x{qty} refused: only {held} "
                        "long contract(s) held — an opening order must "
                        "not reverse the position (open SHORT requires "
                        "the explicit OPEN_SHORT intent)")
                released = qty * self.margin_locked_per_contract
                self.cash += released
                if held - qty == 0:
                    self.margin_locked_per_contract = 0.0
                self.position_qty -= qty
        elif order["instrument"].startswith(TMON_TICKER_PREFIX):
            if side == "SELL":
                self.tmon_qty -= qty
                self.cash += qty * self.tmon_price
            else:
                self.tmon_qty += qty
                self.cash -= qty * self.tmon_price
        self.equity = max(self.equity, 0.0)


class PaperStartupError(RuntimeError):
    """PAPER startup refused — the real read-only account snapshot is
    unavailable or non-positive. Fail closed: the runtime must NOT start
    with a zero/artificial balance (the correct reconcile then stops it
    with EQUITY_UNAVAILABLE anyway)."""


def _build_paper_broker(read_adapter) -> "SimulatedBroker":
    """Seed a SimulatedBroker from the REAL read-only account snapshot.

    Documented PaperExecutionAdapter contract: production paper mode uses
    a SimulatedBroker built from a real read-only snapshot at startup.
    ``read_adapter`` is a LiveExecutionAdapter used EXCLUSIVELY through
    its two read-only methods — find_account() (UsersService.get_accounts)
    and snapshot() (OperationsService.get_positions/get_portfolio +
    OrdersService.get_orders). Neither method can physically reach
    OrdersServiceApi.post_order(): that call site lives only inside
    LiveExecutionAdapter._post_market_order(), which this function never
    invokes (and which stays double-gated for LIVE runs).

    Propagates any read-API/account-discovery error unchanged (fail
    closed) and raises PaperStartupError when equity/cash are not
    positive — never substitutes a zero or artificial balance.
    """
    account_id = read_adapter.find_account()
    live_snap = read_adapter.snapshot(account_id)
    equity = float(live_snap.equity_rub or 0.0)
    cash = float(live_snap.free_cash_rub or 0.0)
    if equity <= 0 or cash <= 0:
        raise PaperStartupError(
            f"PAPER startup fail closed: real account {account_id} "
            f"snapshot has equity={equity} cash={cash} — refusing to "
            "seed SimulatedBroker with a zero/artificial balance")
    return SimulatedBroker(
        cash=cash,
        equity=equity,
        tmon_qty=int(live_snap.tmon_qty or 0),
        position_qty=int(live_snap.position_qty or 0),
        entry_price=float(live_snap.entry_price or 0.0),
    )


def build_execution_adapter(token: str, instrument_uid: str, config=None):
    """Factory honoring the Stage 5A safety gate (single source of truth).

    ``config`` is a validated ``src.runtime_config.TradingConfig``; when
    omitted it is loaded from the environment via the ONE parser. There is
    NO default mode and NO default to LIVE:

      * PAPER → PaperExecutionAdapter + SimulatedBroker seeded from the
        real read-only account snapshot (Stage 6D wiring fix — previously
        SimulatedBroker() defaulted to cash=0/equity=0 and the correct
        fail-closed reconcile stopped the runtime with EQUITY_UNAVAILABLE).
        The PAPER order-flow path still physically cannot reach
        OrdersServiceApi.post_order(): the read-only LiveExecutionAdapter
        is used exclusively for find_account()/snapshot(); any exception
        from that read API propagates unchanged (fail closed — no adapter,
        no zero-balance fallback).
      * LIVE  → LiveExecutionAdapter — allowed ONLY with an explicit
        TRADING_MODE=LIVE plus AUTO_TRADING_ENABLED=true (enforced by
        load_trading_config / enforce_trading_gate in src/runtime_config.py).
      * missing/invalid TRADING_MODE → TradingConfigError (fail closed).
    """
    from src.runtime_config import (load_trading_config, enforce_trading_gate)

    if config is None:
        config = load_trading_config()
    # Fail closed BEFORE any adapter exists: AUTO=false blocks everything,
    # including paper order flow.
    enforce_trading_gate(config)

    if config.is_paper:
        # Real read-only account snapshot seeds the simulated broker.
        # Same account discovery (_find_open_account) as LIVE ⇒ the exact
        # same real account_id the runtime will use downstream.
        read_adapter = LiveExecutionAdapter(token, instrument_uid)
        broker = _build_paper_broker(read_adapter)
        return PaperExecutionAdapter(broker), "PAPER"
    return LiveExecutionAdapter(token, instrument_uid), "LIVE"
