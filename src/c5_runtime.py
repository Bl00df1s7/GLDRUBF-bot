"""C5 production runtime orchestrator (Stage 2D).

This module is the ONLY active trading path of Golden Bot. It wires the
existing, reusable infrastructure together and delegates every trading
decision to the frozen C5 core (``src.strategies.c5_core``):

    Market Data (1d + closed 5m)
        ↓
    Time Gate (ENTRY only at 16:00 MSK; EXIT on every new closed 5m bar)
        ↓
    C5 Core  (signal + sizing — never re-implemented here)
        ↓
    Margin Provider (live get_future FULL, no fallback)
        ↓
    Execution Guard (slippage / book quality)
        ↓
    Protection (daily loss 10% / DD freeze 20% / kill switch 30%)
        ↓
    Cash Manager (TMON liquidation, one bond per operation)
        ↓
    Execution Adapter (paper/live differ ONLY here)
        ↓
    State Store (idempotency, recovery)

Legacy GLDRUBF strategy code (strategy.py, indicators.calculate_sar,
stop_orders, position_monitor, risk_manager.check_circuit_breaker) is
NOT imported anywhere in this module and must never be re-added.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

from config.settings import (
    C5_CLASS_CODE,
    C5_CONTRACT_SIZE_CNY,
    C5_CONTROL_HOUR_MS,
    C5_CONTROL_TIMEFRAME,
    C5_TARGET_TICKER,
    C5_TIMEFRAME,
)
from src.cash_manager import ensure_cash_for_entry
from src.execution_adapter import build_execution_adapter
from src.execution_guard import evaluate_execution_guard, fetch_order_book
from src.instruments import get_c5_instrument
from src.margin_provider import MarginApiUnavailable, MarginTracker
from src.market_data import daily_bars_from_df, load_candles
from src.protection import evaluate_protections
from src.runtime_config import (
    TradingConfig,
    TradingConfigError,
    enforce_trading_gate,
    load_trading_config,
)
from src.state_store import (
    StateSaveError,
    _mask_account_id as _mask_account_id_local,
    load_state,
    save_state,
    save_state_strict,
)
from src.strategies import c5_core

logger = logging.getLogger("golden_bot.c5_runtime")

MSK_TZ = ZoneInfo("Europe/Moscow")
UTC = timezone.utc

# Session window for intraday EXIT checks (MOEX currency futures session).
SESSION_START_MS = (10, 0)
SESSION_END_MS = (23, 40)

REASON_NO_DATA = "NO_MARKET_DATA"
REASON_CANDLE_NOT_CLOSED = "CONTROL_CANDLE_NOT_CLOSED"
REASON_ENTRY_NOT_AT_CONTROL_POINT = "ENTRY_OUTSIDE_CONTROL_POINT"
REASON_MARGIN_API_UNAVAILABLE = c5_core.REASON_MARGIN_API
REASON_PROTECTION_BLOCKED = "PROTECTION_BLOCKED"
REASON_KILL_SWITCH = "KILL_SWITCH"
REASON_SLIPPAGE_GUARD = "SLIPPAGE_GUARD"
REASON_RECONCILE_FAILED = "RECONCILE_FAILED"
REASON_TRADING_BLOCKED = "TRADING_BLOCKED_BY_CONFIG"
# G-5: EXIT execution outcome could not be verified (ok=True but the
# executed quantity is missing / invalid / contradictory). Distinct from
# an ordinary failure so operators can tell "unknown residual" apart from
# "order provably dead".
REASON_EXIT_UNVERIFIED = "EXIT_UNVERIFIED_PENDING_RECONCILE"
EXIT_UNVERIFIED_REASON = REASON_EXIT_UNVERIFIED
# G-7: the durable SELL-intent marker could not be persisted BEFORE
# dispatch — fail closed, the order was NOT sent.
REASON_SELL_INTENT_PERSIST_FAILED = "SELL_INTENT_PERSIST_FAILED"
# G-9C: an opening SHORT order (SELL) was dispatched but its execution
# outcome could not be verified from a trustworthy qty_executed — the
# runtime must NOT invent a position.  Distinct reason code so operators
# can tell "unknown entry fill" apart from an unverified EXIT.
ENTRY_UNVERIFIED_REASON = "ENTRY_UNVERIFIED_PENDING_RECONCILE"


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _msk(dt: datetime) -> datetime:
    return dt.astimezone(MSK_TZ)


def _candle_ts(row) -> str:
    """Canonical idempotency key for a candle row."""
    ts = pd.Timestamp(row["time"])
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.isoformat()


def control_point_passed(now_msk: datetime) -> bool:
    """True once the 16:00 MSK control point has passed today."""
    return now_msk.hour > C5_CONTROL_HOUR_MS or (
        now_msk.hour == C5_CONTROL_HOUR_MS and now_msk.minute >= 5
    )


def in_exit_session(now_msk: datetime) -> bool:
    start = now_msk.replace(hour=SESSION_START_MS[0], minute=SESSION_START_MS[1],
                            second=0, microsecond=0)
    end = now_msk.replace(hour=SESSION_END_MS[0], minute=SESSION_END_MS[1],
                          second=0, microsecond=0)
    return start <= now_msk <= end


def latest_closed_5m(df: pd.DataFrame, now_utc: datetime):
    """Newest fully CLOSED 5m candle (start+5m <= now). No fallback."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[d["time"] + timedelta(minutes=5) <= now_utc]
    if d.empty:
        return None
    return d.sort_values("time").iloc[-1]


def _to_decimal(value) -> Decimal:
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise c5_core.MarginRateError(f"unparsable margin rate {value!r}") from exc
    if not (0 < rate < 0.5):
        raise c5_core.MarginRateError(
            f"margin rate {rate} outside (0, 0.5) — stale/invalid API value")
    return rate


class ReconcileError(RuntimeError):
    """Ambiguous broker state — LIVE execution must not be enabled."""


def reconcile(adapter, account_id: str, state: dict) -> "object":
    """Startup/reconnect reconciliation.

    restore state → broker positions/orders/cash → cross-check against
    persisted position → margin refresh → protection state → enable.

    Raises ReconcileError when the broker view contradicts the persisted
    state in an ambiguous way (e.g. open orders while we believe the
    cycle finished). On any error the caller MUST NOT execute.
    """
    snapshot = adapter.snapshot(account_id)

    stored_qty = int(state.get("c5_position_qty") or 0)
    broker_qty = int(snapshot.position_qty or 0)
    if snapshot.open_orders and broker_qty == 0 and stored_qty == 0:
        # Unattributed working orders with no position — ambiguous state.
        raise ReconcileError(
            f"{snapshot.open_orders} open order(s) without known position; "
            "manual review required before execution")

    if broker_qty != stored_qty:
        logger.warning(
            "RECONCILE: broker position %d != stored %d — adopting broker",
            broker_qty, stored_qty)
        state["c5_position_qty"] = broker_qty
        if broker_qty == 0:
            state["c5_position_direction"] = None
        elif state.get("c5_position_direction") not in ("LONG", "SHORT"):
            state["c5_position_direction"] = "LONG" if broker_qty > 0 else "SHORT"

    # G-5 defect fix: the EXIT_UNVERIFIED freeze must NEVER be cleared
    # just because reconcile() ran.  The freeze exists precisely because
    # an SELL outcome was NOT verified — the runtime does not know how
    # much of the position actually closed.  A single broker snapshot
    # cannot resolve that ambiguity (qty unchanged may mean "nothing
    # filled" OR a fill still in flight; qty changed may be a partial
    # end-state OR a live order that will execute more later), so no
    # automatic heuristic here may invent a residual.
    #
    # G-5 (this fix): NO automatic heuristic may lift the freeze.  A
    # single snapshot cannot prove that the previous ambiguous SELL is
    # TERMINAL (no longer able to execute):
    #   * broker_qty == stored size may mean "nothing filled" AND "a
    #     working order that will fill later";
    #   * broker_qty changed (including to zero) may be the end-state of
    #     the old order AND a partial fill of an order that is still
    #     alive;
    #   * snapshot.open_orders == 0 is still NOT sufficient proof for
    #     lifting the freeze: SimulatedBroker.snapshot() ALWAYS reports 0
    #     (no simulated order book), so a zero there is not a broker
    #     confirmation.  For the LIVE adapter the trustworthiness was
    #     fixed separately (PR #37): a get_orders() error now propagates
    #     out of snapshot() instead of being masked as open_orders = 0,
    #     so startup reconcile FAILS CLOSED (bootstrap marks it FAILED
    #     and no execution is enabled — STARTUP_RECONCILE_FAILED).  A
    #     LIVE zero is therefore a CONFIRMED absence of working orders,
    #     but terminality of the previous ambiguous SELL still cannot be
    #     proven from a single snapshot, so the freeze rule above holds.
    # The adapter contract (ExecutionResult / snapshot) carries no
    # terminal order status usable for this confirmation, so lifting the
    # freeze automatically would invent guarantees the API does not
    # provide.  Fail closed: the freeze persists across cycles and
    # process restarts (state_store.C5_STATE_DEFAULTS) until an EXPLICIT
    # operator confirmation (manual / external reconciliation) records
    # the true residual via state["broker_position_confirmed"] = <int>.
    # That value is trusted as coming from OUTSIDE this runtime's own
    # accounting, so it IS independent confirmation; once present the
    # freeze lifts, the position re-baselines to it and the EXIT ledger
    # resets.  A non-int value (bool/float/str included) is rejected —
    # no coercion launders bad data into a clearance.
    # G-7: an unfinished SELL-intent marker means we cannot prove the
    # exit order never reached the broker.  A regular broker snapshot
    # (even one that matches the stored size) NEVER clears it — same
    # fail-closed rule as the freeze above.  It survives reconcile
    # untouched and blocks any further SELL dispatch in _evaluate_exit
    # until resolve_pending_sell_intent() (explicit operator action) or
    # a proven-terminal result removes it.
    if state.get("exit_pending_sell_intent") and \
            not state.get("exit_unverified_until_reconcile"):
        logger.error(
            "RECONCILE: pending SELL intent %s kept — broker snapshot is"
            " NOT proof the ambiguous exit order never reached the broker;"
            " explicit operator resolution required",
            state["exit_pending_sell_intent"])
    if state.get("exit_unverified_until_reconcile"):
        confirmed = state.get("broker_position_confirmed")
        if type(confirmed) is int:
            state.pop("exit_unverified_until_reconcile", None)
            state.pop("exit_confirmed_filled_qty", None)
            state.pop("broker_position_confirmed", None)
            if confirmed != broker_qty:
                # Belt-and-braces: keep the runtime and the recorded
                # broker truth in agreement (no divergence).
                state["c5_position_qty"] = confirmed
                state["c5_position_direction"] = (
                    None if confirmed == 0 else
                    ("LONG" if confirmed > 0 else "SHORT"))
                logger.warning(
                    "RECONCILE: EXIT_UNVERIFIED freeze lifted by explicit"
                    " operator confirmation; position re-baselined to %d",
                    confirmed)
            else:
                logger.warning(
                    "RECONCILE: EXIT_UNVERIFIED freeze lifted by explicit"
                    " operator confirmation (confirmed qty matches broker"
                    " snapshot %d)", broker_qty)
        else:
            logger.error(
                "RECONCILE: EXIT_UNVERIFIED freeze KEPT — the adapter"
                " contract cannot prove the previous SELL is terminal;"
                " manual/external broker reconciliation required before"
                " any further position operation (qty=%d,"
                " open_orders=%s is NOT trustworthy)",
                broker_qty, snapshot.open_orders)

    # Hard validation AFTER the freeze logic: an invalid equity view
    # raises below and the caller discards these state changes (G-1:
    # failed reconcile ⇒ NO EXECUTION, freeze untouched in the file).
    if snapshot.equity_rub is None or snapshot.equity_rub <= 0:
        raise ReconcileError("EQUITY_UNAVAILABLE during reconcile")

    state["reconcile_status"] = "OK"
    return snapshot


def _marker_operation(marker) -> str:
    """G-9C minimal extension of the G-7 durable SELL-intent marker.

    The marker field ``operation`` distinguishes an unfinished EXIT of a
    LONG (legacy semantics, pre-G-9C markers) from an unfinished OPEN_SHORT
    ENTRY dispatch.  Markers WITHOUT the field — every marker written by
    any earlier version of this runtime — are treated as "EXIT" so the
    established G-7 behaviour stays byte-for-byte compatible.  Anything
    unrecognised fails closed to "EXIT": the most restrictive legacy
    interpretation for guards that already assume exit semantics.
    """
    if not isinstance(marker, dict):
        return "EXIT"
    op = str(marker.get("operation") or "").strip().upper()
    return op if op in ("ENTRY", "EXIT") else "EXIT"


def resolve_pending_sell_intent(state: dict, *, candle_ts: str,
                                qty: int, operation: str = "EXIT") -> bool:
    """G-7 EXPLICIT operator resolution of a pending SELL intent marker.

    Deliberately narrow: it clears ONLY the marker whose candle timestamp
    AND requested quantity match exactly what is on disk.  A plain broker
    snapshot or a coinciding position size NEVER calls this function and
    NEVER lifts the block (same principle as the G-5 EXIT_UNVERIFIED
    freeze).  The operator must first verify at the broker terminal which
    fate the ambiguous SELL actually had; only then may this be used.

    G-9C: the check additionally verifies the marker OPERATION identity
    (EXIT vs OPEN_SHORT ENTRY).  The default ``operation="EXIT"`` keeps
    the exact old contract for legacy callers/tests: markers without the
    ``operation`` field resolve as EXIT (see _marker_operation), so every
    pre-G-9C clearance path behaves identically.  An ENTRY marker can
    only be cleared by an explicit ``operation="ENTRY"`` request — a
    coincidence of candle/qty with an unfinished entry never launders it
    into an exit clearance.
    """
    marker = state.get("exit_pending_sell_intent")
    if not isinstance(marker, dict):
        return False
    wanted_op = str(operation or "").strip().upper()
    if wanted_op not in ("ENTRY", "EXIT"):
        logger.error(
            "SELL_INTENT_RESOLVE refused: unknown operation %r — must be"
            " 'ENTRY' or 'EXIT'; no implicit clearance is allowed",
            operation)
        return False
    if _marker_operation(marker) != wanted_op:
        logger.error(
            "SELL_INTENT_RESOLVE refused: requested operation %r does not"
            " match the pending marker operation %r (%s) — an unfinished"
            " %s must never be cleared as if it were a %s",
            wanted_op, _marker_operation(marker), marker,
            _marker_operation(marker), wanted_op)
        return False
    if str(marker.get("candle_ts")) != str(candle_ts) \
            or int(marker.get("qty") or 0) != int(qty):
        logger.error(
            "SELL_INTENT_RESOLVE refused: requested (ts=%r, qty=%r) does"
            " not match the pending marker (ts=%r, qty=%r) — no implicit"
            " or partial clearance is allowed",
            candle_ts, qty, marker.get("candle_ts"), marker.get("qty"))
        return False
    state.pop("exit_pending_sell_intent", None)
    logger.warning(
        "SELL_INTENT resolved explicitly by operator for candle %s qty %d"
        " (operation=%s)", candle_ts, qty, wanted_op)
    return True


class C5Runtime:
    """One production cycle of the C5 bot."""

    def __init__(self, token: str, adapter=None, trader=None,
                 margin_tracker: Optional[MarginTracker] = None,
                 now_fn=_now_utc, config: Optional[TradingConfig] = None):
        self.token = token
        self.adapter = adapter
        self.trader = trader or adapter
        self._margin = margin_tracker
        self._now_fn = now_fn
        self.now_fn = now_fn
        self.instrument = None
        self.account_id = None
        # G-7: per-process marker of a SELL dispatch that this runtime has
        # already handed to the adapter without a proven-terminal outcome.
        # A brand-new process (restart) always starts with False, so a
        # durable intent marker loaded from disk can never be mistaken for
        # an operation legitimately in flight in THIS process — any further
        # dispatch attempt is then a blind re-send and stays blocked.
        self._sell_dispatch_in_flight = False
        # G-7: identity of the durable SELL-intent marker created by THIS
        # process (per-process counter folded into ``created_at``).  A
        # restarted process starts at 0, so it can never reproduce the
        # exact marker content written by the previous process before a
        # crash — the boundary fake uses this to tell the legitimate
        # first dispatch apart from a blind re-send.
        self._sell_intent_seq = 0
        # Stage 5A safety gate — parsed ONCE via src.runtime_config (the
        # single source of truth). Injected configs are used as-is; when
        # omitted the validated env snapshot is loaded here (no second
        # parser exists anywhere else). Fail-closed errors from an
        # invalid/missing TRADING_MODE propagate to the caller.
        self.config = config if config is not None else load_trading_config()

    # ── infrastructure wiring ────────────────────────────────────────
    def bootstrap(self, state: dict) -> Optional[object]:
        """Resolve instrument, adapter, account; reconcile; refresh margin.

        Returns the reconciled snapshot or None when startup failed
        (NO LIVE EXECUTION in that case). Already-resolved components are
        never re-fetched (idempotent bootstrap; also lets integration
        tests inject boundary fakes without touching the network).

        Stage 6C wiring fix: the CNYRUBF instrument is resolved from the
        REAL T-Invest API in BOTH modes. PAPER execution stays fully local
        (PaperExecutionAdapter + SimulatedBroker — it never reaches
        OrdersServiceApi.post_order), but the runtime still needs the real
        instrument UID for read-only market data (daily candles, closed 5m
        control candle, order book) and for the real broker/account state
        and futures margin queries. Skipping resolution in PAPER used to
        leave ``self.instrument`` as None and crash on
        ``self.instrument.uid``. Fail-closed: unresolved instrument ⇒ no
        adapter, no snapshot, NO EXECUTION.
        """
        if self.instrument is None:
            try:
                self.instrument = get_c5_instrument(self.token)
            except Exception as exc:
                logger.error("STARTUP: CNYRUBF instrument unavailable: %s", exc)
                return None

        if self.adapter is None:
            self.adapter, mode = build_execution_adapter(
                self.token, self.instrument.uid, config=self.config)
            self.trader = self.trader or self.adapter
            logger.info("STARTUP: execution adapter mode=%s", mode)

        if self.account_id is None:
            try:
                self.account_id = self.adapter.find_account()
            except Exception as exc:
                logger.error("STARTUP: account discovery failed: %s", exc)
                return None

        if self._margin is None:
            self._margin = MarginTracker(self.token, C5_TARGET_TICKER)

        # G-1: the two failure classes are deliberately separated.
        # ReconcileError / any other critical state-recovery error ⇒
        # broker view is untrusted ⇒ bootstrap fails, NO EXECUTION.
        # MarginApiUnavailable AFTER a successful reconcile ⇒ the open
        # position is reliably restored from the broker snapshot and
        # EXIT/HOLD need no ГО; only ENTRY stays fail-closed
        # (_evaluate_entry always does a mandatory refresh()).
        try:
            snapshot = reconcile(self.adapter, self.account_id, state)
        except Exception as exc:
            logger.error("RECONCILE failure — trading NOT enabled: %s", exc)
            state["reconcile_status"] = "FAILED"
            return None

        try:
            self._margin.refresh()
        except MarginApiUnavailable as exc:
            logger.warning(
                "MARGIN API unavailable at startup: %s — EXIT/HOLD remain "
                "enabled (position restored from broker snapshot); ENTRY "
                "stays fail-closed until a fresh margin check succeeds.",
                exc)
        return snapshot

    @property
    def margin_tracker(self) -> MarginTracker:
        if self._margin is None:
            self._margin = MarginTracker(self.token, C5_TARGET_TICKER)
        return self._margin

    # ── market data ──────────────────────────────────────────────────
    def fetch_daily_bars(self, now_msk: datetime):
        df = load_candles(self.token, self.instrument.uid,
                          candles_count=120, timeframe=C5_TIMEFRAME)
        bars = daily_bars_from_df(df, exclude_date=now_msk.date())
        return bars

    def fetch_control_candle(self, now_utc: datetime):
        df = load_candles(self.token, self.instrument.uid,
                          candles_count=7 * 288, timeframe=C5_CONTROL_TIMEFRAME)
        return latest_closed_5m(df, now_utc), df

    def _finalize_exit_state(self, state) -> None:
        """G-7: durably persist the post-dispatch EXIT outcome.

        The SELL has already been handed to the adapter at this point, so
        losing the resulting state on a write failure could allow a blind
        re-send after a restart.  A strict save is attempted; if it fails
        we retry once (transient errors are common) and then fall back to
        the best-effort writer — which still performs the atomic replace
        — while logging loudly that durability is NOT proven.  Whatever
        remains on disk in that worst case is either the new terminal
        state or the previously saved intent marker; both keep the next
        process from re-sending blindly (the marker blocks dispatch, see
        the G-7 guard in _evaluate_exit).
        """
        try:
            save_state_strict(state)
        except StateSaveError as exc:
            logger.error(
                "G-7: post-dispatch EXIT state save failed (%r) — one"
                " strict retry", exc)
            try:
                save_state_strict(state)
            except StateSaveError as exc2:
                logger.critical(
                    "G-7: EXIT final state durability NOT PROVEN (%r);"
                    " falling back to best-effort atomic write; pending"
                    " SELL-intent marker (if any) stays authoritative on"
                    " the last successful write", exc2)
                save_state(state)

    def _mark_sell_outcome_unverified(self, state) -> None:
        """G-7: label the durable SELL-intent marker of THIS cycle as an
        UNKNOWN-outcome residue (``SELL_OUTCOME_UNVERIFIED``).

        The order was handed to the broker and its fate is not proven
        terminal.  Relabeling lets the owning process's pre-guard
        recognise its own burned candle (marker status
        SELL_OUTCOME_UNVERIFIED + ``last_processed_5m_ts`` == the
        marker's candle_ts) and reach the ordinary DUPLICATE_CANDLE /
        freeze classification instead of masking it with the cycle-level
        block.  A restarted process never carries this knowledge — for
        it the marker remains an unfinished foreign operation and stays
        fully blocked (fail-closed).  The marker is NEVER cleared here:
        only a proven-terminal result or resolve_pending_sell_intent()
        removes it.  Mutates the in-memory dict only; persistence is the
        caller's responsibility (strict save before returning).
        """
        marker = state.get("exit_pending_sell_intent")
        if isinstance(marker, dict):
            marker["status"] = "SELL_OUTCOME_UNVERIFIED"

    # ── one full cycle ───────────────────────────────────────────────
    def run_cycle(self) -> c5_core.Decision:
        # STAGE 5A SAFETY GATE — evaluated BEFORE anything that can reach
        # an execution adapter. AUTO_TRADING_ENABLED=false (or missing /
        # invalid) blocks ALL order flow, paper included. This is the only
        # gate check; main.py stays a thin entrypoint and never re-parses.
        try:
            enforce_trading_gate(self.config)
        except TradingConfigError as exc:
            logger.error("TRADING_BLOCKED before execution: %s", exc)
            return c5_core.block_entry(_noop_decision(),
                                       REASON_TRADING_BLOCKED)

        now_utc = self.now_fn()
        now_msk = _msk(now_utc)

        state = load_state()

        # G-7: a NEW process always starts with an empty dispatch flag (set
        # in __init__), so a marker loaded from disk can never be mistaken
        # for an in-flight operation of this process (see _evaluate_exit /
        # set flag below).  Also expose the live per-cycle state dict to
        # test boundary fakes — purely observational, production logic
        # never reads it (see tests/test_stage5b_paper_runtime.py
        # ScriptedSellAdapter).
        self._last_state = state

        # G-7/G-9C: a durable unfinished SELL-intent marker on disk means
        # an order send (EXIT of a LONG — or, since G-9C, the opening
        # SHORT SELL) is NOT proven terminal.  Block the WHOLE cycle
        # before bootstrap/reconcile — no adapter call, no snapshot, no
        # position re-baselining could launder the unknown residual into
        # a fresh ENTRY or a blind re-send.  Only explicit operator
        # resolution (resolve_pending_sell_intent) clears it; ordinary
        # snapshots NEVER do.
        pending_marker = state.get("exit_pending_sell_intent")
        if isinstance(pending_marker, dict):
            # G-7 fail-closed restart guard.  A durable unfinished
            # SELL-intent marker means an exit order send is NOT proven
            # terminal.  Exception (per spec): the marker belongs to the
            # candle THIS runtime's ledger already processed
            # (marker.candle_ts == last_processed_5m_ts) AND the outcome
            # was recorded as non-terminal — an explicit
            # "SELL_OUTCOME_UNVERIFIED" marker status written by the
            # owning process right after dispatch.  The owning process
            # burned that candle and its own _evaluate_exit guards
            # (duplicate → freeze → pending) keep blocking every further
            # dispatch, so the cycle may proceed to bootstrap/reconcile
            # and reach the ordinary DUPLICATE_CANDLE classification for
            # the same bar.  A FRESH process never carries our
            # in-process dispatch flag, and any marker from a DIFFERENT
            # candle remains fully blocked: no adapter call, no
            # reconcile, no position re-baselining, no ENTRY, no blind
            # re-send.  Only resolve_pending_sell_intent() clears the
            # durable marker.  The flag NEVER replaces the marker.
            #
            # G-9C: this exception applies ONLY to EXIT markers.  An
            # OPEN_SHORT entry marker (operation="ENTRY") that survived
            # on disk is ALWAYS a full cycle block even for its own
            # burned candle: unlike an exit, an unverified ENTRY leaves
            # the runtime with NO trustworthy position at all, so
            # proceeding under the EXIT guards would let later cycles act
            # on an invented flat view.  Legacy markers without the
            # "operation" field keep the pre-G-9C semantics — they are
            # treated as EXIT (see _marker_operation).
            burned_by_ledger = str(pending_marker.get("candle_ts")) == \
                str(state.get("last_processed_5m_ts"))
            owns_burned_candle = False
            if self._sell_dispatch_in_flight:
                # Mid-dispatch re-entry inside the owning process only;
                # never set across cycles or restarts (cleared in the
                # dispatch finally, initialised False in __init__).
                owns_burned_candle = True
            elif burned_by_ledger and str(pending_marker.get("status")) \
                    == "SELL_OUTCOME_UNVERIFIED" \
                    and _marker_operation(pending_marker) == "EXIT":
                logger.error(
                    "PENDING_SELL_INTENT owned-by-this-process: unfinished"
                    " exit operation %s for already-burned candle %s —"
                    " proceeding under the EXIT_UNVERIFIED guards; NO"
                    " further SELL dispatch until explicit resolution",
                    pending_marker, pending_marker.get("candle_ts"))
                owns_burned_candle = True
            if not owns_burned_candle:
                logger.error(
                    "CYCLE_BLOCKED_PENDING_SELL_INTENT: unfinished SELL "
                    "intent %s survived on disk — no reconcile/ENTRY/EXIT "
                    "until explicit operator resolution",
                    state["exit_pending_sell_intent"])
                return c5_core.block_entry(_noop_decision(),
                                           REASON_EXIT_UNVERIFIED)

        snapshot = self.bootstrap(state)
        if snapshot is None:
            save_state(state)
            return c5_core.block_entry(
                _noop_decision(), "STARTUP_RECONCILE_FAILED")

        # Protection runs BEFORE any signal evaluation result is used.
        protection = evaluate_protections(state, snapshot.equity_rub,
                                          now=now_msk)

        position = _position_from_snapshot(snapshot)

        # ── EXIT path: independent of the 16:00 ENTRY gate ───────────
        if position is not None and in_exit_session(now_msk):
            decision = self._evaluate_exit(state, snapshot, position,
                                           now_utc, now_msk)
            save_state(state)
            return decision

        # ── ENTRY path: strictly at the 16:00 MSK control point ──────
        if position is None:
            decision = self._evaluate_entry(state, snapshot, now_utc,
                                            now_msk, protection)
            save_state(state)
            return decision

        save_state(state)
        return c5_core.block_entry(_noop_decision(), "POSITION_OPEN")

    # ── EXIT ─────────────────────────────────────────────────────────
    def _evaluate_exit(self, state, snapshot, position, now_utc,
                       now_msk) -> c5_core.Decision:
        candle, _df = self.fetch_control_candle(now_utc)
        if candle is None:
            logger.warning("EXIT_CHECK: no closed 5m candle available")
            return c5_core.block_entry(_noop_decision(), REASON_NO_DATA)

        ts = _candle_ts(candle)
        if state.get("last_processed_5m_ts") == ts:
            logger.info("IDEMPOTENT: 5m candle %s already processed", ts)
            return c5_core.block_entry(_noop_decision(), "DUPLICATE_CANDLE")

        # G-5: an EXIT whose execution outcome could not be verified
        # leaves the true broker residual unknown. Until a trustworthy
        # reconcile re-baselines the position (bootstrap → reconcile
        # adopts the broker view and clears this flag), NO further
        # operation with this position is allowed — neither a new SELL
        # (could duplicate an order that was actually accepted/filled,
        # same protection as G-4) nor any other action. Fail closed.
        if state.get("exit_unverified_until_reconcile"):
            logger.error(
                "EXIT_BLOCKED_UNVERIFIED: previous exit execution is not"
                " confirmed (%s); position operations frozen until"
                " reconcile. Candle %s burned to prevent blind re-send.",
                state["exit_unverified_until_reconcile"], ts)
            state["last_processed_5m_ts"] = ts
            return c5_core.block_entry(_noop_decision(),
                                       REASON_EXIT_UNVERIFIED)

        # G-7: a SELL-intent marker survived on disk. That means an exit
        # order send of THIS runtime is not proven terminal — after a
        # crash/restart we cannot tell whether the broker received it.
        # Re-sending blindly could duplicate a live/filled order, so ALL
        # SELL dispatch is blocked until explicit operator resolution
        # (resolve_pending_sell_intent) or a proven-terminal result below
        # clears it.  This guard is unconditional: even the very candle
        # that created the marker may NOT be dispatched twice while the
        # marker survives (a crash between dispatch and the terminal
        # save must never allow a second send on the same bar).  The
        # candle is burned exactly like the G-5/G-6 guards.
        pending = state.get("exit_pending_sell_intent")
        if isinstance(pending, dict):
            logger.error(
                "SELL_INTENT_PENDING: unfinished exit operation %s found"
                " on disk; re-send BLOCKED until explicit operator"
                " resolution. Candle %s burned to prevent blind re-send.",
                pending, ts)
            state["last_processed_5m_ts"] = ts
            return c5_core.block_entry(_noop_decision(),
                                       REASON_EXIT_UNVERIFIED)

        daily = self.fetch_daily_bars(now_msk)
        if len(daily) < max(c5_core.C5_EXIT_DONCHIAN,
                           c5_core.C5_ENTRY_DONCHIAN) + 1:
            logger.error("EXIT_CHECK: insufficient daily history (%d)", len(daily))
            return c5_core.block_entry(_noop_decision(),
                                       c5_core.REASON_DATA_INSUFFICIENT)

        bar = c5_core.Candle(dt=pd.Timestamp(candle["time"]).to_pydatetime(),
                             close=float(candle["close"]), is_closed=True)
        price = float(candle["close"])
        direction = position.direction
        try:
            rates = self.margin_tracker.current \
                or self.margin_tracker.refresh()
            rate = _to_decimal(rates.for_direction(direction))
        except (MarginApiUnavailable, c5_core.MarginRateError) as exc:
            logger.error("%s (exit sizing input invalid): %s",
                         REASON_MARGIN_API_UNAVAILABLE, exc)
            # G-1: EXIT itself does not need ГО — the EXIT/HOLD branch of
            # c5_core.evaluate consumes no margin-derived value.  Proceed
            # with margin_rate=None instead of blocking; the candle is NOT
            # burned before a real exit attempt.
            rate = None

        decision = c5_core.evaluate(
            daily_bars=daily,
            control_bar=bar,
            equity=snapshot.equity_rub,
            free_margin=snapshot.free_cash_rub,
            position=position,
            price=price,
            margin_rate=rate,
            now=now_msk,
        )

        if decision.action != "EXIT":
            # Non-exit candles are consumed as before; an EXIT candle is
            # marked processed only after the sell attempt below, so a
            # failed/unexecuted exit can be re-evaluated on the next bar
            # without burning the exit event on a margin-API error.
            state["last_processed_5m_ts"] = ts

        if decision.action == "EXIT":
            qty = abs(int(snapshot.position_qty))

            # G-7: durably record the SELL intent BEFORE dispatch. The
            # marker carries everything needed for a safe post-restart
            # assessment: status, instrument, account, direction, the
            # candle that triggered the exit and the requested contract
            # quantity. If this write fails the order is NOT sent — we
            # would rather miss one exit than duplicate an untracked one.
            # NOTE: `state` here is the same dict run_cycle() saves at the
            # end of the cycle, so any later successful save also persists
            # (or clears) this marker — no orphan can survive a confirmed
            # terminal outcome while the final save succeeded.
            self._sell_intent_seq += 1
            intent = {
                "status": "SELL_IN_FLIGHT",
                "side": "SELL",
                "instrument": C5_TARGET_TICKER,
                "account_id_masked": _mask_account_id_local(self.account_id),
                "direction": position.direction,
                "candle_ts": ts,
                "qty": qty,
                # G-7: per-process sequence number folded into the
                # timestamp makes the marker content unique to THIS
                # process's dispatch — a restarted runtime can never
                # byte-for-byte reproduce a crashed predecessor's marker.
                "created_at": "%s+seq%d" % (now_utc.isoformat(),
                                            self._sell_intent_seq),
            }
            state["exit_pending_sell_intent"] = intent
            # G-7: this process IS the one that just persisted the marker
            # and is about to hand the SELL to the adapter — set the
            # in-process dispatch flag BEFORE sell_cny so the boundary
            # fake can distinguish the legitimate first dispatch from a
            # blind re-send after a restart (a restarted process always
            # starts with the flag False).  Cleared below once the result
            # reaches a proven-terminal state.
            try:
                save_state_strict(state)
            except StateSaveError as exc:
                # Pre-dispatch failure: the order was provably NOT sent
                # (sell_cny has not been called yet — the dispatch flag
                # is set ONLY after this strict save succeeds, so the
                # marker on disk can never claim an in-flight send that
                # never happened).  Roll the in-memory marker back, burn
                # nothing, keep the candle unprocessed — the next cycle
                # retries the persist-and-send atom.  The on-disk
                # previous state never contained the marker (the strict
                # save failed), so there is nothing to clean there.
                state.pop("exit_pending_sell_intent", None)
                logger.error(
                    "%s: durable SELL intent could not be persisted (%r)"
                    " — SELL NOT dispatched, position untouched, candle"
                    " %s NOT burned", REASON_SELL_INTENT_PERSIST_FAILED,
                    exc, ts)
                return c5_core.block_entry(
                    _noop_decision(), REASON_SELL_INTENT_PERSIST_FAILED)

            # G-7: the marker is durably on disk NOW; this process is
            # about to hand the SELL to the adapter.  Set the per-process
            # dispatch flag immediately BEFORE sell_cny() (never earlier:
            # a pre-save failure must leave it False) and clear it
            # guaranteedly in the finally below — no path may leak it.
            self._sell_dispatch_in_flight = True
            try:
                result = self.adapter.sell_cny(self.account_id, qty)
            except Exception as exc:
                # G-6 defect fix: an adapter exception after the SELL was
                # dispatched leaves the fill outcome UNKNOWN — it proves
                # neither execution nor non-execution.  Fail closed exactly
                # like the UNVERIFIED branch below: the position is NOT
                # modified on assumptions, the candle is burned so no blind
                # re-send can duplicate a possibly-live order, and the
                # EXIT_UNVERIFIED freeze blocks every further position
                # operation until an EXPLICIT operator confirmation lifts
                # it (reconcile never clears it automatically).  Without
                # this guard the exception propagated out of run_cycle()
                # BEFORE save_state(), so the protective state did not
                # survive a restart and the next process could re-send
                # the SELL blindly.  run_cycle() always saves state after
                # _evaluate_exit returns, so returning here persists the
                # freeze (verified by tests/test_stage5b_paper_runtime.py
                # G-6 regression tests).
                logger.error(
                    "CNY_EXIT EXCEPTION from adapter after dispatch: %r —"
                    " outcome unknown; position untouched, candle %s"
                    " burned, EXIT_UNVERIFIED freeze set until explicit"
                    " reconciliation", exc, ts, exc_info=True)
                state["exit_failure_reason"] = EXIT_UNVERIFIED_REASON
                state["exit_unverified_until_reconcile"] = \
                    EXIT_UNVERIFIED_REASON
                state["last_processed_5m_ts"] = ts
                # G-7: the SELL-intent marker was persisted strictly to
                # disk BEFORE dispatch, so the unfinished operation is
                # already durable.  KEEP it here as well: an adapter
                # exception proves nothing about whether the request
                # reached the broker (the dispatch boundary records calls
                # only AFTER the adapter returns/raises — see
                # tests/test_stage5b_paper_runtime.py ScriptedSellAdapter),
                # and the freeze alone must survive even if the final save
                # below fails outright.  Mark the intent itself as an
                # UNKNOWN-outcome residue (SELL_OUTCOME_UNVERIFIED) so the
                # owning process's pre-guard recognises its own burned
                # candle; resolve_pending_sell_intent() /
                # a proven-terminal outcome are the only ways to clear it.
                self._mark_sell_outcome_unverified(state)
                # G-7: burn + freeze must be durable BEFORE returning —
                # otherwise a lost final save would leave an UNBURNED
                # exit candle plus an SELL_IN_FLIGHT marker on disk, and
                # a later process could reach _evaluate_exit for this bar
                # and re-send blindly.  Strict write first; if durability
                # cannot be proven, fall back to the best-effort atomic
                # writer (still an atomic replace) and log loudly.  The
                # kept durable marker is relabeled ONLY to the
                # SELL_OUTCOME_UNVERIFIED terminal-fate-unknown status
                # (never cleared, never size/ts-rewritten): that label is
                # what lets the OWNING process distinguish its own burned
                # candle from a foreign unfinished operation.
                try:
                    save_state_strict(state)
                except StateSaveError as exc2:
                    logger.error(
                        "G-7: post-dispatch freeze/burn persistence not"
                        " proven (%r) — falling back to best-effort atomic"
                        " write", exc2)
                    save_state(state)
                return c5_core.block_entry(_noop_decision(),
                                           REASON_EXIT_UNVERIFIED)
            finally:
                # G-7: guaranteed clear — EVERY exit path of the dispatch
                # try (normal return, exception branch return, or a
                # propagating error) leaves the per-process flag False.
                # It never survives into a later cycle and NEVER replaces
                # the durable marker on disk.
                self._sell_dispatch_in_flight = False
            status, filled = _classify_exit_execution(result, qty)
            if not result.ok:
                # G-4 path unchanged: ok=False carries no trustworthy
                # fill info — the confirmed-rejection classification
                # below decides re-attempt safety.
                status = "FAILED"

            if status == "FULL":
                # Confirmed FULL fill of the whole open size. Keeps the
                # pre-G-4 idempotency exactly: the candle is marked
                # processed and the position fields are cleared.
                state["last_processed_5m_ts"] = ts
                state.pop("exit_failure_reason", None)
                state.pop("exit_unverified_until_reconcile", None)
                # EXIT ledger resets with the closed position (the whole
                # open size is confirmed executed — nothing to re-baseline).
                state.pop("exit_confirmed_filled_qty", None)
                state["c5_position_qty"] = 0
                state["c5_position_direction"] = None
                state["c5_entry_price"] = None
                state["last_action"] = decision.reason
                # G-7: terminal CONFIRMED outcome — resolve the durable
                # intent and persist the COMPLETE final state strictly
                # (after every field update, so a successful save can
                # never leave a stale marker or a stale position size).
                state.pop("exit_pending_sell_intent", None)
                # G-7: strict durability FIRST (inside the guarded try —
                # the finally below clears the dispatch flag only AFTER
                # this write completed), then the terminal decision.
                self._finalize_exit_state(state)
                logger.info("CNY_EXIT executed: qty=%d order=%s reason=%s",
                            qty, result.order_id, decision.reason)
                return decision

            if status == "PARTIAL":
                # G-5: a freeze from an EARLIER unverified EXIT means the
                # fate of that prior order is still unknown.  A new
                # confirmed partial fill does NOT prove the old order is
                # terminal, so it must not lift the freeze; the SELL was
                # never supposed to reach the broker in this state (the
                # _evaluate_exit guard blocks it — fail-closed belt).
                if state.get("exit_unverified_until_reconcile"):
                    logger.error(
                        "CNY_EXIT PARTIAL while EXIT_UNVERIFIED freeze is"
                        " active — freeze KEPT (partial fill does not"
                        " prove the previous ambiguous order is terminal)")
                # Confirmed PARTIAL fill (ok=True, 0 < qty_executed < qty):
                # only the executed volume may be removed from the tracked
                # position. The remainder stays open with its direction
                # and entry price intact — clearing the whole position
                # here would silently lose track of an unsettled residual.
                signed = int(state.get("c5_position_qty") or 0)
                if abs(signed) != qty:
                    # The classification is relative to the size actually
                    # SENT to the broker; a stored sign/size inconsistent
                    # with that request makes the residual unknowable —
                    # fail closed instead of doing unsafe arithmetic.
                    status = "UNVERIFIED"
                else:
                    remaining_signed = (signed - filled
                                        if direction == "LONG"
                                        else signed + filled)
                    if abs(remaining_signed) == 0 or \
                            (remaining_signed > 0) != (signed > 0):
                        # Defensive fail-closed guard: a confirmed partial
                        # must never flip/clear the side; if the arithmetic
                        # contradicts the classification, treat as UNKNOWN.
                        status = "UNVERIFIED"
                    else:
                        state["last_processed_5m_ts"] = ts
                        state.pop("exit_failure_reason", None)
                        if not state.get("exit_unverified_until_reconcile"):
                            state.pop("exit_unverified_until_reconcile", None)
                        # G-5 EXIT ledger: remember the CONFIRMED filled
                        # volume so a later trustworthy reconcile can
                        # compute the expected residual (stored − fills).
                        state["exit_confirmed_filled_qty"] = int(
                            state.get("exit_confirmed_filled_qty") or 0) + filled
                        state["c5_position_qty"] = remaining_signed
                        state["c5_position_direction"] = direction
                        state["last_action"] = decision.reason
                        # G-7: confirmed PARTIAL — the dispatched order
                        # reached a terminal confirmed state for the sent
                        # volume; resolve the intent and persist the full
                        # residual state strictly (see FULL branch note).
                        state.pop("exit_pending_sell_intent", None)
                        # G-7: strict durability before returning the
                        # terminal PARTIAL state (flag cleared in the
                        # finally above this write completes).
                        self._finalize_exit_state(state)
                        logger.warning(
                            "CNY_EXIT PARTIAL: %d/%d filled order=%s "
                            "reason=%s — remaining %d kept under "
                            "management; no blind re-send of the residual "
                            "until this order is confirmed terminal",
                            filled, qty, result.order_id, decision.reason,
                            abs(remaining_signed))
                        return decision

            if status == "ZERO_FILL":
                # ok=True but zero confirmed executed quantity —
                # contradictory ExecutionResult. Nothing was proven to
                # have closed ⇒ fall through to the UNVERIFIED branch
                # below (position untouched, no invented remainder).
                logger.error(
                    "CNY_EXIT contradiction: adapter reported ok=True with "
                    "qty_executed=0 — treated as UNVERIFIED, position "
                    "untouched")

            # ── G-4 / G-5: failed or unverified EXIT — classify by
            #    execution certainty ──────────────────────────────────
            # CONFIRMED rejection (the exchange provably did NOT fill
            # the order): safe to keep the candle unburned so the next
            # run_cycle() re-evaluates the still-valid exit signal.
            # Status source: ExecutionResult.details["status"] when the
            # LIVE adapter could query it; otherwise the machine-readable
            # reason suffix ORDER_NOT_FILLED:<STATUS> (execution_adapter
            # ._post_market_order).  Unknown/absent status ⇒ UNKNOWN
            # outcome ⇒ do NOT auto-retry (a second SELL could duplicate
            # an order that was in fact accepted/filled).
            confirmed_dead = _is_confirmed_rejection(result)
            # G-7: the dispatch flag stays True THROUGH the whole
            # post-dispatch handling below (including the strict final
            # saves) and is cleared guaranteedly in the finally that
            # wraps this try — so a mid-cycle crash can never leave a
            # stale "in flight" claim alive in a later cycle.  The
            # RAM-only flag NEVER replaces the durable marker: UNVERIFIED
            # / unknown outcomes keep the marker on disk and stay
            # fail-closed through the freeze + pre-guard instead.
            state["exit_failure_reason"] = str(
                result.reason if not result.ok else EXIT_UNVERIFIED_REASON)
            if status == "UNVERIFIED" or (result.ok and status == "ZERO_FILL"):
                # G-5: adapter claimed success but the executed quantity
                # cannot be trusted (None / wrong type / negative / over
                # the requested size / contradictory ok=True+qty=0).
                # The true broker residual is unknown ⇒ do NOT invent it,
                # do NOT clear the position, and flag the state so no
                # further operation with this position happens until an
                # EXTERNAL confirmation re-baselines it (reconcile never
                # lifts this freeze automatically — a snapshot cannot
                # prove the ambiguous SELL is terminal).  The candle is
                # burned exactly like an UNKNOWN G-4 outcome — a blind
                # re-send could duplicate an order that was in fact
                # accepted.
                state["exit_unverified_until_reconcile"] = EXIT_UNVERIFIED_REASON
                state["last_processed_5m_ts"] = ts
                # G-7: the order WAS handed to the broker adapter and its
                # fate is unproven — KEEP the marker on disk (finalized
                # strictly above) so even a restart that somehow loses the
                # freeze still sees the unfinished operation. The marker
                # was already durably written before dispatch; nothing
                # further to persist here.  Mark the intent itself as an
                # UNKNOWN-outcome residue (SELL_OUTCOME_UNVERIFIED) so the
                # owning process's pre-guard recognises its own burned
                # candle and reaches the ordinary DUPLICATE_CANDLE /
                # freeze classification instead of masking it with the
                # cycle-level block.
                self._mark_sell_outcome_unverified(state)
                # G-7: burn + freeze must be durable BEFORE returning —
                # same fail-closed reasoning as the exception branch
                # above (strict write, best-effort fallback, marker is
                # never rewritten while kept).
                try:
                    save_state_strict(state)
                except StateSaveError as exc2:
                    logger.error(
                        "G-7: UNVERIFIED freeze/burn persistence not"
                        " proven (%r) — falling back to best-effort atomic"
                        " write", exc2)
                    save_state(state)
                logger.error(
                    "CNY_EXIT UNVERIFIED: ok=%s qty_executed=%r requested=%d"
                    " — position state frozen until reconcile; candle %s "
                    "burned, NO automatic re-attempt",
                    result.ok, getattr(result, "qty_executed", None),
                    qty, ts)
            elif confirmed_dead:
                # G-7: exchange PROVABLY rejected/cancelled the order —
                # no execution happened (existing G-4 contract), so the
                # intent is terminal and may be cleared; the candle stays
                # unburned and a safe re-attempt is allowed.
                state.pop("exit_pending_sell_intent", None)
                self._finalize_exit_state(state)
                state.pop("last_processed_5m_ts", None)
                logger.error(
                    "CNY_EXIT FAILED (confirmed rejection): %s — candle "
                    "%s NOT burned, exit will be re-attempted",
                    result.reason, ts)
            else:
                # TRADING_BLOCKED / unknown fill outcome: burn the candle
                # exactly as before G-4 — retrying blindly could create a
                # duplicate SELL while broker state is not verified.
                # G-7: outcome not proven terminal — KEEP the marker and
                # persist it strictly.  Relabel it to the UNKNOWN-fate
                # status first, so the owning process's pre-guard
                # recognises its own burned candle (marker candle_ts ==
                # ledger) and the next pass over this bar reaches the
                # ordinary DUPLICATE_CANDLE classification instead of the
                # cycle-level block; a restarted process still sees an
                # unfinished foreign operation and stays fully blocked.
                self._mark_sell_outcome_unverified(state)
                state["last_processed_5m_ts"] = ts
                self._finalize_exit_state(state)
                logger.error(
                    "CNY_EXIT FAILED (outcome not confirmed-rejected): %s"
                    " — no automatic re-attempt; reconcile on next start "
                    "remains the source of truth", result.reason)
        return decision

    # ── ENTRY ────────────────────────────────────────────────────────
    def _evaluate_entry(self, state, snapshot, now_utc, now_msk,
                        protection) -> c5_core.Decision:
        if not control_point_passed(now_msk):
            logger.info("ENTRY_GATE: before %02d:00 MSK — NO_ENTRY",
                        C5_CONTROL_HOUR_MS)
            return c5_core.block_entry(_noop_decision(),
                                       REASON_ENTRY_NOT_AT_CONTROL_POINT)

        candle, _df = self.fetch_control_candle(now_utc)
        if candle is None:
            return c5_core.block_entry(_noop_decision(), REASON_NO_DATA)

        ts = _candle_ts(candle)
        if state.get("last_c5_entry_control_timestamp") == ts:
            logger.info("IDEMPOTENT: control point %s already processed", ts)
            return c5_core.block_entry(_noop_decision(), "DUPLICATE_CANDLE")

        if not protection.entries_allowed:
            reason = (REASON_KILL_SWITCH if protection.halted
                      else REASON_PROTECTION_BLOCKED)
            logger.error("ENTRY_BLOCKED %s: %s", reason, protection.reason)
            return c5_core.block_entry(_noop_decision(), reason)

        daily = self.fetch_daily_bars(now_msk)
        min_len = max(c5_core.C5_ENTRY_DONCHIAN, c5_core.C5_EXIT_DONCHIAN) + 1
        if len(daily) < min_len:
            logger.error("ENTRY: insufficient daily history %d < %d",
                         len(daily), min_len)
            return c5_core.block_entry(_noop_decision(),
                                       c5_core.REASON_DATA_INSUFFICIENT)

        bar = c5_core.Candle(dt=pd.Timestamp(candle["time"]).to_pydatetime(),
                             close=float(candle["close"]), is_closed=True)
        price = float(candle["close"])

        # Live margin — mandatory fresh refresh before ANY entry.
        # No fallback: failure ⇒ NO_ENTRY.
        try:
            rates = self.margin_tracker.refresh()
        except MarginApiUnavailable as exc:
            logger.error("ENTRY_BLOCKED %s: %s", REASON_MARGIN_API_UNAVAILABLE,
                         exc)
            return c5_core.block_entry(_noop_decision(),
                                       REASON_MARGIN_API_UNAVAILABLE)

        # Direction is decided inside c5_core; probe both rates so sizing
        # uses exactly dlongClient/dshortClient matching the breakout side.
        long_rate = _to_decimal(rates.long_margin_rub)
        short_rate = _to_decimal(rates.short_margin_rub)

        decision = c5_core.evaluate(
            daily_bars=daily,
            control_bar=bar,
            equity=snapshot.equity_rub,
            free_margin=snapshot.free_cash_rub,
            position=None,
            price=price,
            margin_rate=long_rate if _is_long_signal(bar, daily) else short_rate,
            now=now_msk,
        )

        if decision.action not in ("ENTER_LONG", "ENTER_SHORT"):
            # Control point consumed even without a trade: it must never
            # be re-evaluated with same-bar data (duplicate-order safety).
            state["last_c5_entry_control_timestamp"] = ts
            return decision

        direction = "LONG" if decision.action == "ENTER_LONG" else "SHORT"
        rate = long_rate if direction == "LONG" else short_rate

        # Execution guard: bid/ask/spread/depth/slippage/prereqs.
        try:
            bids, asks = fetch_order_book(self.token, self.instrument.uid)
        except Exception as exc:
            logger.error("ENTRY_BLOCKED: order book unavailable: %s", exc)
            state["last_c5_entry_control_timestamp"] = ts
            return c5_core.block_entry(decision, "ORDER_BOOK_UNAVAILABLE")

        guard = evaluate_execution_guard(
            direction=direction, bids=bids, asks=asks,
            required_qty=decision.qty,
            margin_available=True, equity_available=True, me_ok=True,
        )
        if not guard.allowed:
            state["last_c5_entry_control_timestamp"] = ts
            reason = (REASON_SLIPPAGE_GUARD if any(
                f.startswith("SLIPPAGE") for f in guard.failed_checks)
                else "EXECUTION_GUARD_" + "_".join(guard.failed_checks))
            return c5_core.block_entry(decision, reason)

        # Cash manager: TMON liquidation ONE bond per operation until the
        # required ГО is covered; cap reached ⇒ NO_ENTRY (no partial fill).
        required_margin = float(rate * Decimal(str(price)) *
                                Decimal(str(C5_CONTRACT_SIZE_CNY)) *
                                decision.qty)
        cash = ensure_cash_for_entry(self.trader, self.account_id,
                                     required_margin)
        if not cash.ok:
            state["last_c5_entry_control_timestamp"] = ts
            logger.error("ENTRY_BLOCKED %s: required=%.2f cash=%.2f ops=%d",
                         cash.reason, required_margin, cash.cash_after,
                         cash.operations)
            return c5_core.block_entry(decision, cash.reason)

        # ── G-9C: durable SELL-intent BEFORE dispatch for the OPENING
        #    SHORT.  An opening SHORT is routed through SELL — the SAME
        #    side as a LONG exit — so without a pre-persisted marker a
        #    crash between send and the final save would leave no trace,
        #    and a restarted process could re-send the SELL blindly (or
        #    reconcile could launder an unknown residual into a fresh
        #    position).  The marker extends the G-7 contract with
        #    ``operation="ENTRY"`` (legacy EXIT markers simply lack the
        #    field — see _marker_operation) and carries instrument,
        #    masked account, direction, candle, requested qty and
        #    creation time.  A failed strict save means the order is NOT
        #    sent (fail closed; the control point stays unburned so the
        #    next cycle retries the persist-and-send atom).  An existing
        #    unfinished marker can never be overwritten here: the
        #    run_cycle pre-guard blocks the whole cycle while any marker
        #    survives on disk, so an unfinished LONG exit is never
        #    masked/masqueraded by a new SHORT entry intent.
        entry_intent = None
        if direction == "SHORT":
            self._sell_intent_seq += 1
            entry_intent = {
                "status": "SELL_IN_FLIGHT",
                "side": "SELL",
                "operation": "ENTRY",
                "intent": "OPEN_SHORT",
                "instrument": C5_TARGET_TICKER,
                "account_id_masked": _mask_account_id_local(self.account_id),
                "direction": direction,
                "candle_ts": ts,
                "qty": int(decision.qty),
                # Per-process sequence folded into created_at keeps this
                # marker unique to THIS dispatch (G-7 identity rule).
                "created_at": "%s+seq%d" % (now_utc.isoformat(),
                                            self._sell_intent_seq),
            }
            state["exit_pending_sell_intent"] = entry_intent
            try:
                save_state_strict(state)
            except StateSaveError as exc:
                # Pre-dispatch failure: sell_cny has NOT been called yet,
                # so the on-disk previous state provably contains no
                # marker from this attempt — roll the in-memory marker
                # back, burn NOTHING (the control point stays
                # unprocessed) and block.  Same contract as the G-7 EXIT
                # pre-save failure.
                state.pop("exit_pending_sell_intent", None)
                logger.error(
                    "%s: durable OPEN_SHORT intent could not be persisted"
                    " (%r) — SELL NOT dispatched, position untouched,"
                    " control point %s NOT burned",
                    REASON_SELL_INTENT_PERSIST_FAILED, exc, ts)
                return c5_core.block_entry(
                    _noop_decision(), REASON_SELL_INTENT_PERSIST_FAILED)
            # Marker is durably on disk NOW; this process owns the
            # in-flight dispatch (flag set strictly AFTER the successful
            # save, cleared guaranteedly below — same rule as G-7 EXIT).
            self._sell_dispatch_in_flight = True

        try:
            if direction == "LONG":
                # CNY BUY — the whole final_qty in ONE operation.  The live
                # ГО per contract computed by c5_core (FutureBy rate ×
                # price × size) is forwarded to the PAPER adapter so
                # SimulatedBroker can block the margin instead of inventing
                # a cost basis; LIVE ignores the extra kwargs (the exchange
                # blocks ГО itself).  ENTER_LONG behaviour unchanged.
                result = self.adapter.buy_cny(
                    self.account_id, decision.qty,
                    margin_per_contract=float(decision.margin_per_contract),
                    entry_price=guard.expected_price or price)
            else:
                # G-9C: ENTER_SHORT → opening SELL.  The ГО/entry metadata
                # and the OPEN_SHORT intent tag let SimulatedBroker reserve
                # margin, record the basis and refuse accidental reversals.
                result = self.adapter.sell_cny(
                    self.account_id, decision.qty,
                    margin_per_contract=float(decision.margin_per_contract),
                    entry_price=guard.expected_price or price,
                    intent="OPEN_SHORT")
        except Exception as exc:
            # G-6/G-9C: an adapter exception after dispatch proves
            # neither execution nor non-execution.  Fail closed: NO
            # position is invented, the control point is burned so no
            # blind re-send can duplicate a possibly-live order, and the
            # durable marker is relabeled SELL_OUTCOME_UNVERIFIED and
            # persisted strictly (best-effort fallback) so even a lost
            # final save leaves the unfinished operation on disk.  A
            # restarted process sees the ENTRY marker and stays fully
            # blocked at the run_cycle pre-guard until an explicit
            # operator resolution.
            logger.error(
                "CNY_ENTRY EXCEPTION from adapter after dispatch: %r —"
                " outcome unknown; position untouched, control point %s"
                " burned, pending SELL-intent marker kept until explicit"
                " reconciliation", exc, ts, exc_info=True)
            state["last_c5_entry_control_timestamp"] = ts
            state["last_action"] = ENTRY_UNVERIFIED_REASON
            if entry_intent is not None:
                entry_intent["status"] = "SELL_OUTCOME_UNVERIFIED"
                try:
                    save_state_strict(state)
                except StateSaveError as exc2:
                    logger.error(
                        "G-9C: post-dispatch ENTRY marker persistence not"
                        " proven (%r) — falling back to best-effort atomic"
                        " write", exc2)
                    save_state(state)
            return c5_core.block_entry(_noop_decision(),
                                       ENTRY_UNVERIFIED_REASON)
        finally:
            # Guaranteed clear of the per-process dispatch flag — every
            # exit path leaves it False; it NEVER replaces the durable
            # marker (same invariant as the G-7 EXIT finally).
            if entry_intent is not None:
                self._sell_dispatch_in_flight = False

        status, filled = _classify_entry_execution(result, decision.qty)
        if status == ENTRY_FILL_FAILED:
            # ok=False: nothing was confirmed executed.  The control
            # point is burned exactly like the pre-G-9C behaviour (a
            # market-order re-send on the same bar could duplicate an
            # accepted-but-unreported order); NO position is created.
            state["last_c5_entry_control_timestamp"] = ts
            logger.error("CNY_ENTRY FAILED: %s", result.reason)
            return c5_core.block_entry(decision,
                                       f"EXECUTION_FAILED:{result.reason}")

        if status in (ENTRY_FILL_UNVERIFIED, ENTRY_FILL_ZERO):
            # ok=True but the executed quantity cannot be trusted
            # (missing / wrong type / negative / over the request /
            # contradictory ok=True+qty=0).  G-9C: do NOT invent a
            # position — keep c5_position_qty/direction at their flat
            # values, burn the control point against a blind re-send and
            # KEEP the durable marker (relabeled UNKNOWN-fate) so the
            # unfinished operation survives any restart until explicit
            # operator resolution / trustworthy reconcile.
            logger.error(
                "CNY_ENTRY UNVERIFIED: ok=%s qty_executed=%r requested=%d"
                " — NO position recorded, control point %s burned, no"
                " automatic re-attempt", getattr(result, "ok", None),
                getattr(result, "qty_executed", None), decision.qty, ts)
            state["last_c5_entry_control_timestamp"] = ts
            state["last_action"] = ENTRY_UNVERIFIED_REASON
            if entry_intent is not None:
                entry_intent["status"] = "SELL_OUTCOME_UNVERIFIED"
                try:
                    save_state_strict(state)
                except StateSaveError as exc2:
                    logger.error(
                        "G-9C: UNVERIFIED ENTRY marker persistence not"
                        " proven (%r) — falling back to best-effort atomic"
                        " write", exc2)
                    save_state(state)
            return c5_core.block_entry(_noop_decision(),
                                       ENTRY_UNVERIFIED_REASON)

        # FULL / PARTIAL: the ONLY trustworthy volume source is the
        # confirmed qty_executed — NEVER the requested decision.qty.  A
        # partial entry records the smaller confirmed position; the
        # unfilled remainder is simply not ours (no phantom contracts).
        # For SHORT the tracked position is NEGATIVE.
        state["last_c5_entry_control_timestamp"] = ts
        state["c5_position_qty"] = filled if direction == "LONG" else -filled
        state["c5_position_direction"] = direction
        state["c5_entry_price"] = guard.expected_price
        state["margin_long_rate"] = float(long_rate)
        state["margin_short_rate"] = float(short_rate)
        state["last_action"] = decision.reason
        if entry_intent is not None:
            # Proven-terminal CONFIRMED outcome (FULL/PARTIAL): resolve
            # the durable intent and persist the COMPLETE final state
            # strictly — position fields first, marker removed last, so a
            # successful save can never pair a stale marker with a fresh
            # position or vice versa (runtime ↔ broker snapshot stay in
            # agreement; reconcile adopts the same negative qty).
            state.pop("exit_pending_sell_intent", None)
            try:
                save_state_strict(state)
            except StateSaveError as exc:
                logger.error(
                    "G-9C: post-dispatch ENTRY state save failed (%r) —"
                    " one strict retry", exc)
                try:
                    save_state_strict(state)
                except StateSaveError as exc2:
                    logger.critical(
                        "G-9C: ENTRY final state durability NOT PROVEN"
                        " (%r); falling back to best-effort atomic write;"
                        " pending SELL-intent marker (if any) stays"
                        " authoritative on the last successful write", exc2)
                    save_state(state)
        if status == ENTRY_FILL_PARTIAL:
            logger.warning(
                "CNY_ENTRY PARTIAL: %d/%d filled order=%s — confirmed"
                " position %s x%d recorded from qty_executed only",
                filled, decision.qty, result.order_id, direction, filled)
        logger.info(
            "CNY_ENTRY executed: %s x%d (requested %d) order=%s price~%.4f"
            " slippage=%.4f%% required_margin=%.2f (qty_risk=%d"
            " qty_margin=%d qty_raw=%d)",
            direction, filled, decision.qty, result.order_id,
            guard.expected_price or 0.0,
            (guard.expected_slippage_pct or 0.0) * 100, required_margin,
            decision.qty_risk, decision.qty_margin, decision.qty_raw)
        return decision


def _is_long_signal(bar, daily) -> bool:
    """Cheap directional pre-check used ONLY to pick dlong vs dshort rate.

    The authoritative signal remains c5_core.evaluate(); this mirrors its
    entry comparison (close vs Upper(10)/Lower(10)) without duplicating
    sizing/risk logic.
    """
    n = c5_core.C5_ENTRY_DONCHIAN
    upper = max(b.high for b in daily[-n:])
    lower = min(b.low for b in daily[-n:])
    return bar.close > upper


def _position_from_snapshot(snapshot):
    qty = int(snapshot.position_qty or 0)
    if qty == 0:
        return None
    return c5_core.Position(direction="LONG" if qty > 0 else "SHORT",
                            qty=abs(qty))


# ── G-4: EXIT failure classification (execution certainty) ────────────
# Terminal statuses the T-Invest SDK reports for an order that provably
# reached a NOT-FILLED end state (see auto_trader.wait_for_order_fill:
# REJECTED/CANCELLED return lots_executed=0).  Anything else — NEW/NONE/
# time_out placeholders, missing status, TRADING_BLOCKED (the adapter may
# have raised before we could be sure nothing was posted elsewhere) — is
# treated as UNKNOWN and must not trigger an automatic re-post.
CONFIRMED_REJECTION_STATUSES = frozenset({
    "EXECUTION_REPORT_STATUS_REJECTED",
    "EXECUTION_REPORT_STATUS_CANCELLED",
})


def _is_confirmed_rejection(result) -> bool:
    """True only when the execution result PROVES the order was not filled.

    Sources of truth, in priority order:
      1. ``result.details["status"]`` — set by LiveExecutionAdapter from
         the SDK order-state it actually queried;
      2. ``result.reason`` suffix ``ORDER_NOT_FILLED:<STATUS>`` — the
         machine-readable code emitted by the same adapter when
         lots_executed <= 0.
    A reason without a recognised terminal status (unknown placeholder,
    TRADING_BLOCKED, arbitrary error string) returns False: fail closed
    against a potentially duplicating re-send.
    """
    status = str((getattr(result, "details", None) or {}).get("status") or "")
    if not status:
        reason = str(getattr(result, "reason", "") or "")
        prefix = "ORDER_NOT_FILLED:"
        status = reason[len(prefix):] if reason.startswith(prefix) else ""
    return status in CONFIRMED_REJECTION_STATUSES


# ── G-5: EXIT fill classification (executed quantity certainty) ────────
# ``ExecutionResult.qty_executed`` semantics (verified against the LIVE
# adapter _post_market_order and auto_trader.wait_for_order_fill):
#   * integer, number of FUTURES CONTRACTS filled on the exchange —
#     the same unit as AccountSnapshot.position_qty / c5_position_qty
#     (signed contracts) and SimulatedBroker.position_qty;
#   * the LIVE adapter returns ok=True ONLY with lots_executed > 0
#     (int(lots)); ok=False results carry the default qty_executed=0;
#   * the PAPER adapter mirrors full fills: qty_executed == requested qty.
# A partial fill therefore surfaces as ok=True AND 0 < qty_executed < qty.
EXIT_FILL_FULL = "FULL"
EXIT_FILL_PARTIAL = "PARTIAL"
EXIT_FILL_ZERO = "ZERO_FILL"          # ok=True + qty 0 → contradictory
EXIT_FILL_UNVERIFIED = "UNVERIFIED"   # everything else cannot be trusted


def _classify_exit_execution(result, qty):
    """Classify one EXIT execution result against the requested size.

    Returns ``(status, filled)`` where status is one of FULL / PARTIAL /
    ZERO_FILL / UNVERIFIED and ``filled`` is the trustworthy executed
    contract count (only meaningful for FULL / PARTIAL).

    Fail-closed rules (no rounding, no coercion that could launder an
    invalid amount into a valid one):
      * ``qty`` must already be a positive int — otherwise UNVERIFIED;
      * ``ok=False`` ⇒ UNVERIFIED (the G-4 rejection/unknown branch
        decides whether a re-attempt is safe);
      * ``qty_executed`` missing/None, non-int, bool, negative or
        greater than the requested size ⇒ UNVERIFIED;
      * ``qty_executed == 0`` while ok=True ⇒ ZERO_FILL (contradictory
        result — treated as UNVERIFIED by the caller);
      * ``qty_executed == qty`` ⇒ FULL; ``0 < qty_executed < qty`` ⇒
        PARTIAL.
    """
    if type(qty) is not int or qty <= 0:
        return EXIT_FILL_UNVERIFIED, 0
    if not getattr(result, "ok", False):
        return EXIT_FILL_UNVERIFIED, 0
    raw = getattr(result, "qty_executed", None)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        return EXIT_FILL_UNVERIFIED, 0
    if raw < 0 or raw > qty:
        return EXIT_FILL_UNVERIFIED, 0
    if raw == 0:
        return EXIT_FILL_ZERO, 0
    if raw == qty:
        return EXIT_FILL_FULL, raw
    return EXIT_FILL_PARTIAL, raw


def _noop_decision() -> c5_core.Decision:
    """Neutral decision shell used by blocking paths (action=NO_ENTRY)."""
    return c5_core.Decision(
        action="NO_ENTRY", qty=0, reason="NO_ACTION",
        signal_price=0.0,
        donchian_entry_upper=0.0, donchian_entry_lower=0.0,
        donchian_exit_upper=0.0, donchian_exit_lower=0.0,
        atr14=0.0, atr_target=c5_core.C5_ATR_TARGET,
        risk_multiplier=1.0, risk_budget=0.0,
        margin_rate=Decimal("0"), margin_per_contract=Decimal("0"),
        required_margin=Decimal("0"), m_e_ratio=0.0,
        qty_risk=0, qty_margin=0, qty_liquidity=0, qty_raw=0,
    )


# ── G-9C: ENTRY fill classification (executed quantity certainty) ──────
# Same fail-closed contract as the EXIT classifier: ``ok`` alone NEVER
# proves a fill — only a valid ``qty_executed`` is a confirmation source.
#   * qty must be a positive int — otherwise UNVERIFIED;
#   * ok=False ⇒ FAILED (the adapter provably did not execute; safe to
#     keep the candle unburned and re-attempt);
#   * qty_executed missing/None/non-int/bool/negative ⇒ UNVERIFIED
#     (over-request is also UNVERIFIED — an inconsistent report cannot
#     launder a bigger position into existence);
#   * 0 while ok=True ⇒ ZERO_FILL (contradictory — treated UNVERIFIED);
#   * == qty ⇒ FULL; 0 < executed < qty ⇒ PARTIAL.
ENTRY_FILL_FULL = "FULL"
ENTRY_FILL_PARTIAL = "PARTIAL"
ENTRY_FILL_ZERO = "ZERO_FILL"
ENTRY_FILL_UNVERIFIED = "UNVERIFIED"
ENTRY_FILL_FAILED = "FAILED"


def _classify_entry_execution(result, qty):
    """Classify one ENTRY execution result against the requested size.

    Returns ``(status, filled)`` where ``filled`` is the trustworthy
    executed contract count (meaningful for FULL / PARTIAL only).  The
    caller MUST base any position update on ``filled``, never on the
    requested ``qty`` — see G-9C requirement "confirmed volume only".
    """
    if type(qty) is not int or qty <= 0:
        return ENTRY_FILL_UNVERIFIED, 0
    # G-9C: a None (or otherwise result-less) adapter response proves
    # NOTHING — it must never be treated as a failure-with-reason nor as
    # a fill.  Classify it UNVERIFIED so the caller keeps the durable
    # marker and blocks until explicit resolution.
    if result is None:
        return ENTRY_FILL_UNVERIFIED, 0
    if not getattr(result, "ok", False):
        return ENTRY_FILL_FAILED, 0
    raw = getattr(result, "qty_executed", None)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, int):
        return ENTRY_FILL_UNVERIFIED, 0
    if raw < 0 or raw > qty:
        return ENTRY_FILL_UNVERIFIED, 0
    if raw == 0:
        return ENTRY_FILL_ZERO, 0
    if raw == qty:
        return ENTRY_FILL_FULL, raw
    return ENTRY_FILL_PARTIAL, raw


def run_once(token: str) -> c5_core.Decision:
    """Single production cycle (used by workflow / scheduler).

    Stage 5A: the trading-safety config is loaded ONCE here via
    src.runtime_config (fail closed on missing/invalid TRADING_MODE) and
    injected into the runtime — there is no second parser downstream.
    """
    if not token:
        raise RuntimeError("T_SANDAPI is missing or empty (fail closed)")
    config = load_trading_config()
    runtime = C5Runtime(token, config=config)
    return runtime.run_cycle()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Fail closed BEFORE any network/token use: missing or invalid
    # TRADING_MODE aborts; AUTO_TRADING_ENABLED!=true aborts.
    try:
        config = load_trading_config()
    except TradingConfigError as exc:
        logger.error("STARTUP REFUSED (fail closed): %s", exc)
        raise SystemExit(2) from exc
    if not config.trading_allowed:
        logger.error(
            "AUTO_TRADING_ENABLED is not true — trading blocked before "
            "execution (mode=%s). Nothing was sent to the broker.",
            config.mode)
        return
    # Credential wiring: the ONLY accepted token env name is T_SANDAPI.
    # No fallback to legacy names; missing/empty token fails closed in
    # run_once(). The token value itself is never logged or printed.
    token = os.environ.get("T_SANDAPI", "")
    decision = run_once(token)
    logger.info("RUN_RESULT action=%s qty=%d reason=%s",
                decision.action, decision.qty, decision.reason)


if __name__ == "__main__":
    main()
