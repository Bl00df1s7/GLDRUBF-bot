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
    AUTO_TRADING_ENABLED,
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
from src.state_store import load_state, save_state
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

    if snapshot.equity_rub is None or snapshot.equity_rub <= 0:
        raise ReconcileError("EQUITY_UNAVAILABLE during reconcile")

    state["reconcile_status"] = "OK"
    return snapshot


class C5Runtime:
    """One production cycle of the C5 bot."""

    def __init__(self, token: str, adapter=None, trader=None,
                 margin_tracker: Optional[MarginTracker] = None,
                 now_fn=_now_utc):
        self.token = token
        self.adapter = adapter
        self.trader = trader or adapter
        self._margin = margin_tracker
        self._now_fn = now_fn
        self.now_fn = now_fn
        self.instrument = None
        self.account_id = None

    # ── infrastructure wiring ────────────────────────────────────────
    def bootstrap(self, state: dict) -> Optional[object]:
        """Resolve instrument, adapter, account; reconcile; refresh margin.

        Returns the reconciled snapshot or None when startup failed
        (NO LIVE EXECUTION in that case). Already-resolved components are
        never re-fetched (idempotent bootstrap; also lets integration
        tests inject boundary fakes without touching the network).
        """
        if self.instrument is None:
            try:
                self.instrument = get_c5_instrument(self.token)
            except Exception as exc:
                logger.error("STARTUP: CNYRUBF instrument unavailable: %s", exc)
                return None

        if self.adapter is None:
            self.adapter, mode = build_execution_adapter(
                self.token, self.instrument.uid)
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

        try:
            snapshot = reconcile(self.adapter, self.account_id, state)
            self._margin.refresh()
        except (ReconcileError, MarginApiUnavailable) as exc:
            logger.error("RECONCILE/MARGIN failure — trading NOT enabled: %s", exc)
            state["reconcile_status"] = "FAILED"
            return None
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

    # ── one full cycle ───────────────────────────────────────────────
    def run_cycle(self) -> c5_core.Decision:
        now_utc = self.now_fn()
        now_msk = _msk(now_utc)

        state = load_state()
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
            # EXIT itself does not need ГО, but the pure Decision contract
            # requires a valid rate; block with explicit reason, no guess.
            state["last_processed_5m_ts"] = ts
            return c5_core.block_entry(_noop_decision(),
                                       REASON_MARGIN_API_UNAVAILABLE)

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

        state["last_processed_5m_ts"] = ts

        if decision.action == "EXIT":
            qty = abs(int(snapshot.position_qty))
            result = self.adapter.sell_cny(self.account_id, qty)
            if result.ok:
                state["c5_position_qty"] = 0
                state["c5_position_direction"] = None
                state["c5_entry_price"] = None
                state["last_action"] = decision.reason
                logger.info("CNY_EXIT executed: qty=%d order=%s reason=%s",
                            qty, result.order_id, decision.reason)
            else:
                logger.error("CNY_EXIT FAILED: %s", result.reason)
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

        # CNY BUY — the whole final_qty in ONE operation.
        result = self.adapter.buy_cny(self.account_id, decision.qty)
        state["last_c5_entry_control_timestamp"] = ts
        if not result.ok:
            logger.error("CNY_ENTRY FAILED: %s", result.reason)
            return c5_core.block_entry(decision,
                                       f"EXECUTION_FAILED:{result.reason}")

        state["c5_position_qty"] = decision.qty if direction == "LONG" \
            else -decision.qty
        state["c5_position_direction"] = direction
        state["c5_entry_price"] = guard.expected_price
        state["margin_long_rate"] = float(long_rate)
        state["margin_short_rate"] = float(short_rate)
        state["last_action"] = decision.reason
        logger.info(
            "CNY_ENTRY executed: %s x%d order=%s price~%.4f slippage=%.4f%% "
            "required_margin=%.2f (qty_risk=%d qty_margin=%d qty_raw=%d)",
            direction, decision.qty, result.order_id,
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


def run_once(token: str) -> c5_core.Decision:
    """Single production cycle (used by workflow / scheduler)."""
    if not token:
        raise RuntimeError("SANDBOX_TOKEN / INVEST_TOKEN is empty")
    runtime = C5Runtime(token)
    return runtime.run_cycle()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    token = os.environ.get("SANDBOX_TOKEN") or os.environ.get("INVEST_TOKEN", "")
    mode = os.environ.get("TRADING_MODE", "PAPER").upper()
    if not AUTO_TRADING_ENABLED and mode != "PAPER":
        logger.error("AUTO_TRADING_ENABLED=false — LIVE execution disabled")
        return
    decision = run_once(token)
    logger.info("RUN_RESULT action=%s qty=%d reason=%s",
                decision.action, decision.qty, decision.reason)


if __name__ == "__main__":
    main()
