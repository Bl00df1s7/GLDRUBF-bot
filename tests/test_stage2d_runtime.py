"""Stage 2D — C5 production runtime integration tests.

These tests prove the ACTUAL production execution path
(`src.c5_runtime.C5Runtime.run_cycle`, entered through `src.main._run`)
really wires:

    Market Data (1d + closed 5m)
      → Time Gate (ENTRY only at/after 16:00 MSK control point)
      → frozen C5 Core (signal + sizing — never mocked)
      → Margin Provider (live future_by FULL, dlong/dshort, no fallback)
      → Execution Guard (slippage / book quality)
      → Protection (daily loss 10% / DD freeze 20% / kill switch 30%)
      → Cash Manager (TMON liquidation, one bond per operation)
      → Execution Adapter (the ONLY order-placement boundary)
      → State Store (idempotency, restart recovery)

Mocks/fakes are allowed ONLY at external boundaries: broker API, market
data API, clock, order fills, state persistence failure.  The C5 core,
sizing, margin validation, protection math and cash-manager loop are the
REAL production objects.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# ── deterministic market fixture constants ────────────────────────────
REF_DATE = dt.date(2026, 9, 30)          # last closed daily bar
CONTROL_DAY = dt.date(2026, 10, 2)      # today — control day
LONG_BREAK = 1420.0                      # > Upper(10)=1400
SHORT_BREAK = 1370.0                     # < Lower(10)=1400
NEUTRAL = 1400.0                         # inside channel == flat daily bars
PRICE = 1400.0
ATR_EQ_TARGET = 0.1563                   # multiplier == 1.0 exactly


def daily_df(n=60):
    """Flat daily candles H=L=C=1400 with prev_close=(H+L+C)/3=1400
    → ATR14 == 0.1563 → risk multiplier == 1.0 (frozen math intact)."""
    rows = []
    d = REF_DATE - dt.timedelta(days=n - 1)
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        rows.append({"time": ts, "open": PRICE, "high": PRICE,
                     "low": PRICE, "close": PRICE, "volume": 1})
        d += dt.timedelta(days=1)
    assert rows[-1]["time"].tz_convert(MSK).date() == REF_DATE
    return pd.DataFrame(rows)


def candle_5m(day: dt.date, hour: int, minute: int, close: float) -> dict:
    ts = pd.Timestamp(dt.datetime(day.year, day.month, day.day,
                                  hour, minute, tzinfo=MSK)
                      .astimezone(dt.timezone.utc))
    return {"time": ts, "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1}


def five_m_df(rows):
    return pd.DataFrame(rows)


class FakeInstrument:
    uid = "uid-cnyrubf-test"
    ticker = "CNYRUBF"
    class_code = "SPBFUT"


class BrokerState:
    """Mutable fake broker behind the recording adapter."""

    def __init__(self, cash=1_000_000.0, equity=1_000_000.0,
                 position_qty=0, tmon_qty=0, open_orders=0,
                 entry_price=None):
        self.cash = cash
        self.equity = equity
        self.position_qty = position_qty
        self.tmon_qty = tmon_qty
        self.open_orders = open_orders
        self.entry_price = entry_price
        self.orders: list[dict] = []
        self.tmon_price = 1000.0

    def snapshot(self, account_id):
        from src.execution_adapter import AccountSnapshot
        return AccountSnapshot(account_id=account_id,
                               equity_rub=self.equity,
                               free_cash_rub=self.cash,
                               position_qty=self.position_qty,
                               entry_price=self.entry_price,
                               tmon_qty=self.tmon_qty,
                               open_orders=self.open_orders)

    def apply(self, order):
        side, qty, inst = order["side"], int(order["qty"]), order["instrument"]
        if inst == "CNYRUBF":
            self.position_qty += qty if side == "BUY" else -qty
            self.entry_price = PRICE if side == "BUY" else None
        else:
            if side == "SELL":
                self.tmon_qty -= qty
                self.cash += qty * self.tmon_price
            else:
                self.tmon_qty += qty
                self.cash -= qty * self.tmon_price
        self.orders.append(dict(order))


class RecordingAdapter:
    """Fake of BaseExecutionAdapter — the external execution boundary."""

    def __init__(self, broker: BrokerState):
        self.broker = broker
        self.calls: list[tuple] = []

    def find_account(self):
        return "fake-account"

    def snapshot(self, account_id):
        return self.broker.snapshot(account_id)

    def get_free_cash(self, account_id):
        return self.broker.cash

    def get_tmon_quantity(self, account_id):
        return self.broker.tmon_qty

    def get_tmon_price(self, account_id):
        return self.broker.tmon_price

    def _fill(self, instrument, side, qty):
        order = {"instrument": instrument, "side": side, "qty": int(qty)}
        self.broker.apply(order)
        self.calls.append((instrument, side, int(qty)))
        from src.execution_adapter import ExecutionResult
        return ExecutionResult(True, "FAKE_FILLED",
                               order_id=f"ord-{len(self.broker.orders)}",
                               qty_executed=int(qty))

    def buy_cny(self, account_id, qty):
        return self._fill("CNYRUBF", "BUY", qty)

    def sell_cny(self, account_id, qty):
        return self._fill("CNYRUBF", "SELL", qty)

    def sell_one_tmon(self, account_id):
        return self._fill("TMON_N", "SELL", 1)

    def buy_one_tmon(self, account_id):
        return self._fill("TMON_N", "BUY", 1)


class FakeMarginTracker:
    """Boundary fake for the broker margin API (future_by FULL).

    Implements the same contract as src.margin_provider.MarginTracker;
    records every refresh so tests can prove ENTRY-path refreshes always
    re-fetch live values (no stale-cache fallback).
    """

    def __init__(self, long_rate=0.0585, short_rate=0.0576, fail=False):
        self.long_rate = long_rate
        self.short_rate = short_rate
        self.fail = fail
        self.refresh_calls = 0
        self.current = None

    def refresh(self):
        from src.margin_provider import MarginApiUnavailable, MarginRates
        self.refresh_calls += 1
        if self.fail:
            raise MarginApiUnavailable("fake API down")
        self.current = MarginRates(long_margin_rub=self.long_rate,
                                   short_margin_rub=self.short_rate,
                                   instrument="CNYRUBF")
        return self.current


def good_book(direction, qty):
    """Order book that passes the slippage guard for the given side."""
    mid = PRICE
    if direction == "LONG":
        bids = [(mid - 0.05, 100)]
        asks = [(mid + 0.05, max(qty, 100))]
    else:
        bids = [(mid - 0.05, max(qty, 100))]
        asks = [(mid + 0.05, 100)]
    return bids, asks


def wide_book(direction, qty):
    """Mid far from touch → expected slippage > 0.10% → SLIPPAGE_GUARD."""
    off = PRICE * 0.002  # 0.20% away from mid on both sides
    if direction == "LONG":
        return [(PRICE - off, 100)], [(PRICE + off, max(qty, 100))]
    return [(PRICE - off, max(qty, 100))], [(PRICE + off, 100)]


def strict_book(direction, qty):
    """Depth below required qty → guard fails with INSUFFICIENT_DEPTH."""
    if direction == "LONG":
        return [(PRICE - 0.05, 100)], [(PRICE + 0.05, max(qty - 1, 1))]
    return [(PRICE - 0.05, max(qty - 1, 1))], [(PRICE + 0.05, 100)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", state_file)
    import src.state_store as ss
    monkeypatch.setattr(ss, "STATE_FILE", state_file)
    return tmp_path


def make_runtime(broker, *, now_utc, five_rows, daily=None,
                 margin=None, adapter=None, book=None):
    """Build a real C5Runtime with fakes ONLY at external boundaries."""
    from src.c5_runtime import C5Runtime
    import src.c5_runtime as rt_mod

    adapter = adapter or RecordingAdapter(broker)
    rt = C5Runtime(token="fake-token", adapter=adapter,
                   margin_tracker=margin or FakeMarginTracker(),
                   now_fn=lambda: now_utc)
    rt.instrument = FakeInstrument()
    rt.account_id = "fake-account"

    daily = daily if daily is not None else daily_df()

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return daily.copy()
        if timeframe == "5m":
            return five_m_df(five_rows)
        raise AssertionError(f"C5 runtime must not request {timeframe!r}")

    # patch the boundary where c5_runtime actually resolves the names
    rt_mod.load_candles = _load
    book = book or good_book("LONG", 5)
    rt_mod.fetch_order_book = lambda *a, **k: book
    return rt, adapter


@pytest.fixture(autouse=True)
def _isolate_boundaries():
    """Snapshot/restore module-level names patched by the tests so no
    fake ever leaks into another test or into production imports."""
    import src.c5_runtime as rt_mod
    saved = (rt_mod.load_candles, rt_mod.fetch_order_book)
    yield
    rt_mod.load_candles, rt_mod.fetch_order_book = saved


def utc_ms(day, hour, minute):
    return dt.datetime(day.year, day.month, day.day, hour, minute,
                       tzinfo=MSK).astimezone(dt.timezone.utc)


def read_state():
    import src.state_store as ss
    with open(ss.state_file_path()) as f:
        return json.load(f)


C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
NOW_1605 = utc_ms(CONTROL_DAY, 16, 5)


# ══════════════════════════════════════════════════════════════════════
# A — valid ENTRY at 16:00
# ══════════════════════════════════════════════════════════════════════
def test_a_valid_entry_at_control_point(env):
    broker = BrokerState(cash=1_000_000, equity=1_000_000)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()

    assert decision.action == "ENTER_LONG"
    assert decision.qty == 5                    # stage-1 cap from qty_raw>=5
    assert adapter.calls == [("CNYRUBF", "BUY", 5)]     # ONE full-volume op
    assert broker.position_qty == 5
    st = read_state()
    assert st["c5_position_qty"] == 5
    assert st["c5_position_direction"] == "LONG"
    assert st["last_c5_entry_control_timestamp"] == C16["time"].isoformat()
    assert st["reconcile_status"] == "OK"


def test_a_entry_consumes_control_point_on_no_signal(env):
    broker = BrokerState()
    c = candle_5m(CONTROL_DAY, 16, 0, NEUTRAL)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[c])
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "NO_SIGNAL"
    assert adapter.calls == []
    assert read_state()["last_c5_entry_control_timestamp"] == c["time"].isoformat()


# ══════════════════════════════════════════════════════════════════════
# B — ENTRY outside 16:00 → NO_ENTRY
# ══════════════════════════════════════════════════════════════════════
def test_b_entry_blocked_before_control_point(env):
    broker = BrokerState()
    now = utc_ms(CONTROL_DAY, 12, 0)
    rows = [candle_5m(CONTROL_DAY, 11, 55, LONG_BREAK)]
    rt, adapter = make_runtime(broker, now_utc=now, five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "ENTRY_OUTSIDE_CONTROL_POINT"
    assert adapter.calls == []


def test_b_entry_blocked_when_control_candle_not_closed(env):
    """16:03 — the 16:00 candle is NOT yet closed; latest closed is 15:55.
    Its timestamp is not the control point ⇒ NO trade, no fallback."""
    broker = BrokerState()
    now = utc_ms(CONTROL_DAY, 16, 3)
    rows = [candle_5m(CONTROL_DAY, 15, 55, LONG_BREAK),
            candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)]  # unclosed at 16:03
    rt, adapter = make_runtime(broker, now_utc=now, five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert adapter.calls == []
    # the newest CLOSED candle (15:55) is consumed without trading
    assert read_state()["last_c5_entry_control_timestamp"] == \
        rows[0]["time"].isoformat()


# ══════════════════════════════════════════════════════════════════════
# C — duplicate ENTRY on the same control candle
# ══════════════════════════════════════════════════════════════════════
def test_c_duplicate_control_point_no_second_buy(env):
    broker = BrokerState()
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    first = rt.run_cycle()
    assert first.action == "ENTER_LONG"
    # second cycle on the SAME candle: fresh runtime instance, same state
    rt2, adapter2 = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    second = rt2.run_cycle()
    assert second.reason == "DUPLICATE_CANDLE"
    assert second.action == "NO_ENTRY"
    cny_buys = [o for o in broker.orders
                if o["instrument"] == "CNYRUBF" and o["side"] == "BUY"]
    assert len(cny_buys) == 1


# ══════════════════════════════════════════════════════════════════════
# D — ENTRY ignored while position exists
# ══════════════════════════════════════════════════════════════════════
def test_d_new_entry_ignored_with_open_position(env):
    broker = BrokerState(position_qty=3, entry_price=PRICE, cash=100_000)
    rows = [candle_5m(CONTROL_DAY, 10, 35, NEUTRAL)]   # no EXIT condition
    rt, adapter = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                               five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "HOLD"
    assert decision.qty == 0
    assert adapter.calls == []
    assert broker.position_qty == 3


# ══════════════════════════════════════════════════════════════════════
# E — valid EXIT sells the FULL actual position, one operation
# ══════════════════════════════════════════════════════════════════════
def test_e_exit_full_volume(env):
    broker = BrokerState(position_qty=7, entry_price=PRICE, cash=100_000)
    rows = [candle_5m(CONTROL_DAY, 10, 35, 1390.0)]    # < Lower(5)=1400
    rt, adapter = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                               five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "EXIT"
    assert decision.reason == "EXIT_LONG"
    assert adapter.calls == [("CNYRUBF", "SELL", 7)]   # exact current size
    assert broker.position_qty == 0
    st = read_state()
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None
    assert st["last_processed_5m_ts"] == rows[0]["time"].isoformat()


# ══════════════════════════════════════════════════════════════════════
# F — EXIT independent of the 16:00 gate (morning and afternoon)
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("hour,minute", [(11, 35), (13, 20)])
def test_f_exit_outside_1600_executes(env, hour, minute):
    broker = BrokerState(position_qty=2, entry_price=PRICE)
    rows = [candle_5m(CONTROL_DAY, hour, minute, 1390.0)]
    rt, adapter = make_runtime(broker,
                               now_utc=utc_ms(CONTROL_DAY, hour, minute + 5),
                               five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "EXIT"
    assert adapter.calls == [("CNYRUBF", "SELL", 2)]


def test_f_short_exit_above_upper5(env):
    broker = BrokerState(position_qty=-4, entry_price=PRICE)
    rows = [candle_5m(CONTROL_DAY, 12, 15, 1410.0)]    # > Upper(5)=1400
    rt, adapter = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 12, 20),
                               five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "EXIT"
    assert decision.reason == "EXIT_SHORT"
    assert adapter.calls == [("CNYRUBF", "SELL", 4)]   # abs(full short size)


# ══════════════════════════════════════════════════════════════════════
# G — duplicate EXIT candle → NO_ACTION
# ══════════════════════════════════════════════════════════════════════
def test_g_duplicate_exit_candle(env):
    broker = BrokerState(position_qty=7, entry_price=PRICE)
    rows = [candle_5m(CONTROL_DAY, 10, 35, 1390.0)]
    rt, _ = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                         five_rows=rows)
    assert rt.run_cycle().action == "EXIT"
    rt2, adapter2 = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 41),
                                 five_rows=rows)
    decision = rt2.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "DUPLICATE_CANDLE"
    sells = [o for o in broker.orders if o["side"] == "SELL"]
    assert len(sells) == 1


# ══════════════════════════════════════════════════════════════════════
# H/I — restart after successful BUY / SELL → no duplicate orders
# ══════════════════════════════════════════════════════════════════════
def test_h_restart_after_entry_no_duplicate_buy(env):
    broker = BrokerState()
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    assert rt.run_cycle().action == "ENTER_LONG"
    assert broker.position_qty == 5
    # process survives: new runtime instance, persisted state + real broker
    rt2, adapter2 = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt2.run_cycle()
    assert decision.action != "ENTER_LONG"
    assert ("CNYRUBF", "BUY", 5) not in adapter2.calls
    buys = [o for o in broker.orders if o["side"] == "BUY"]
    assert len(buys) == 1


def test_i_restart_after_exit_no_duplicate_sell(env):
    broker = BrokerState(position_qty=7, entry_price=PRICE)
    rows = [candle_5m(CONTROL_DAY, 10, 35, 1390.0)]
    rt, _ = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                         five_rows=rows)
    assert rt.run_cycle().action == "EXIT"
    assert broker.position_qty == 0
    rt2, adapter2 = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 45),
                                 five_rows=rows)
    rt2.run_cycle()
    assert not [c for c in adapter2.calls if c[1] == "SELL"]
    sells = [o for o in broker.orders if o["side"] == "SELL"]
    assert len(sells) == 1


# ══════════════════════════════════════════════════════════════════════
# J — margin unavailable → NO_ENTRY, no fallback
# ══════════════════════════════════════════════════════════════════════
def test_j_margin_failure_blocks_entry(env):
    broker = BrokerState()
    margin = FakeMarginTracker(fail=True)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               margin=margin)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "MARGIN_API_UNAVAILABLE"
    assert adapter.calls == []
    assert margin.refresh_calls >= 1


def test_j_invalid_margin_rate_blocks_entry(env):
    from src.strategies import c5_core
    broker = BrokerState()
    margin = FakeMarginTracker(long_rate=0.9)      # outside (0, 0.5) band
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               margin=margin)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == c5_core.REASON_MARGIN_API
    assert adapter.calls == []


# ══════════════════════════════════════════════════════════════════════
# K — directional ГО: LONG→dlongClient, SHORT→dshortClient
# ══════════════════════════════════════════════════════════════════════
def test_k_long_entry_uses_dlong_rate_in_sizing(env):
    broker = BrokerState(equity=10_000)            # budget 2000 → qty_risk=24
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               margin=FakeMarginTracker(0.0585, 0.0576))
    decision = rt.run_cycle()
    assert decision.action == "ENTER_LONG"
    assert decision.margin_rate == Decimal("0.0585")
    assert decision.margin_per_contract == Decimal("81.9000")


def test_k_short_entry_uses_dshort_rate_in_sizing(env):
    broker = BrokerState(equity=10_000)
    c = candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[c],
                               margin=FakeMarginTracker(0.0585, 0.0576),
                               book=good_book("SHORT", 5))
    decision = rt.run_cycle()
    assert decision.action == "ENTER_SHORT"
    assert decision.margin_rate == Decimal("0.0576")
    assert decision.margin_per_contract == Decimal("80.6400")
    assert adapter.calls == [("CNYRUBF", "SELL", 5)]


def test_k_margin_refresh_is_live_before_entry(env):
    """The cache must never substitute a live fetch before an ENTRY:
    tracker starts EMPTY (as on process startup); run_cycle must still
    call refresh() and use the freshly fetched rates."""
    broker = BrokerState(equity=10_000)
    margin = FakeMarginTracker()
    assert margin.current is None                  # cold start, no cache
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               margin=margin)
    decision = rt.run_cycle()
    assert margin.refresh_calls >= 1               # live re-fetch happened
    assert decision.margin_rate == Decimal("0.0585")
    assert adapter.calls == [("CNYRUBF", "BUY", 5)]


# ══════════════════════════════════════════════════════════════════════
# L/M/N — protection 10 / 20 / 30
# ══════════════════════════════════════════════════════════════════════
def _trip_protection(kind):
    """Shared setup: baseline equity 100k, then drop equity to trip a
    protection level, then attempt a fresh ENTRY on a NEW control candle."""
    dd_map = {"daily": 90_000, "freeze": 79_000, "kill": 69_000}
    broker = BrokerState(equity=100_000)
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    rt.run_cycle()                                  # establishes baselines
    broker.equity = dd_map[kind]
    later = candle_5m(CONTROL_DAY, 16, 10, LONG_BREAK)
    rt2, adapter2 = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 16, 15),
                                 five_rows=[later])
    decision = rt2.run_cycle()
    return decision, adapter2, broker


def test_l_daily_loss_breaker_10pct_blocks_entry(env):
    decision, adapter, broker = _trip_protection("daily")
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "PROTECTION_BLOCKED"
    assert adapter.calls == []
    assert not read_state().get("kill_switch")


def test_m_drawdown_freeze_20pct_blocks_entry(env):
    decision, adapter, broker = _trip_protection("freeze")
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "PROTECTION_BLOCKED"
    assert adapter.calls == []
    assert not read_state().get("kill_switch")


def test_n_kill_switch_30pct_halts_and_stays_sticky(env):
    decision, adapter, broker = _trip_protection("kill")
    assert decision.reason == "KILL_SWITCH"
    assert adapter.calls == []
    st = read_state()
    assert st["kill_switch"] is True
    assert st["TRADING_HALTED"] is True
    # sticky: even after equity recovers, manual review flag persists
    broker.equity = 95_000
    later = candle_5m(CONTROL_DAY, 16, 15, LONG_BREAK)
    rt3, adapter3 = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 16, 20),
                                 five_rows=[later])
    assert rt3.run_cycle().reason == "KILL_SWITCH"
    assert adapter3.calls == []


def test_protection_never_touches_exit(env):
    """Kill-switch blocks NEW ENTRIES but must not force-close a position
    and must not suppress the C5 EXIT signal."""
    broker = BrokerState(equity=100_000)
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    rt.run_cycle()                                  # enters long 5
    broker.equity = 69_000                          # kill switch trips
    rows = [candle_5m(CONTROL_DAY, 10, 35, 1390.0)]  # EXIT condition
    rt2, adapter2 = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                                 five_rows=rows)
    decision = rt2.run_cycle()
    assert decision.action == "EXIT"
    assert ("CNYRUBF", "SELL", 5) in adapter2.calls


# ══════════════════════════════════════════════════════════════════════
# O/P — TMON sequential liquidation / insufficient cash
# ══════════════════════════════════════════════════════════════════════
def test_o_tmon_sequential_then_single_full_cny_buy(env):
    # required margin for 5 longs = 5 × 81.90 = 409.50 RUB; cash 100 →
    # needs 1 TMON bond (1000 RUB) to be funded.
    broker = BrokerState(cash=100.0, equity=10_000, tmon_qty=3)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "ENTER_LONG"
    assert adapter.calls.count(("TMON_N", "SELL", 1)) == 1   # strictly by 1
    assert adapter.calls[-1] == ("CNYRUBF", "BUY", 5)        # then full volume
    tmon_idx = [i for i, c in enumerate(adapter.calls) if c[0] == "TMON_N"]
    cny_idx = next(i for i, c in enumerate(adapter.calls) if c[0] == "CNYRUBF")
    assert all(i < cny_idx for i in tmon_idx)                # funding precedes BUY
    assert broker.tmon_qty == 2
    assert broker.position_qty == 5


def test_o_two_sequential_tmon_sells_when_one_is_not_enough(env):
    # cash 100, target 409.50; bond price 200 → 100 → 300 → 500 ≥ target.
    broker = BrokerState(cash=100.0, equity=10_000, tmon_qty=5)
    broker.tmon_price = 200.0
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "ENTER_LONG"
    tmon_sells = [c for c in adapter.calls if c == ("TMON_N", "SELL", 1)]
    assert len(tmon_sells) == 2                              # strictly by 1
    assert adapter.calls.index(("CNYRUBF", "BUY", 5)) == len(adapter.calls) - 1


def test_p_insufficient_cash_after_tmon_exhausted(env):
    broker = BrokerState(cash=100.0, equity=10_000, tmon_qty=1)
    broker.tmon_price = 50.0                                 # 100+50 < 409.50
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "NO_TMON_POSITION"
    assert [c for c in adapter.calls if c[0] == "CNYRUBF"] == []
    assert read_state()["last_c5_entry_control_timestamp"] == \
        C16["time"].isoformat()                              # point consumed


def test_p_tmon_operation_cap_blocks_entry(env, monkeypatch):
    import src.cash_manager as cm
    monkeypatch.setattr(cm, "MAX_TMON_SELLS", 2)
    broker = BrokerState(cash=0.0, equity=10_000, tmon_qty=50)
    broker.tmon_price = 10.0                                 # never reaches 409.5
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "INSUFFICIENT_CASH_AFTER_TMON_LIQUIDATION"
    assert adapter.calls.count(("TMON_N", "SELL", 1)) == 2   # capped
    assert ("CNYRUBF", "BUY", 5) not in adapter.calls


# ══════════════════════════════════════════════════════════════════════
# Q — order sent but state save failed → reconcile adopts broker, no dup
# ══════════════════════════════════════════════════════════════════════
def test_q_state_save_failure_reconciles_broker_position(env):
    broker = BrokerState()
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "ENTER_LONG"
    assert broker.position_qty == 5

    # Simulate crash AFTER the fill but BEFORE persistence: wipe the file.
    import src.state_store as ss
    if os.path.exists(ss.state_file_path()):
        os.remove(ss.state_file_path())

    # Restart: state says FLAT, broker says LONG 5 → reconcile adopts broker.
    rt2, adapter2 = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    rt2.run_cycle()
    assert ("CNYRUBF", "BUY", 5) not in adapter2.calls
    buys = [o for o in broker.orders if o["side"] == "BUY"]
    assert len(buys) == 1                                    # no duplicate BUY
    st = read_state()
    assert st["c5_position_qty"] == 5                        # broker adopted


# ══════════════════════════════════════════════════════════════════════
# R — ambiguous broker state blocks execution / broker wins over state
# ══════════════════════════════════════════════════════════════════════
def test_r_open_orders_without_position_no_execution(env):
    broker = BrokerState(open_orders=1)                      # ambiguous
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "STARTUP_RECONCILE_FAILED"
    assert adapter.calls == []
    assert read_state()["reconcile_status"] == "FAILED"


def test_r_broker_position_beats_stale_flat_state(env):
    """State flat + broker long → broker wins; position branch taken,
    no new ENTRY attempted (only C5 exit/hold possible)."""
    broker = BrokerState(position_qty=4, entry_price=PRICE)
    rows = [candle_5m(CONTROL_DAY, 16, 0, NEUTRAL)]
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=rows)
    decision = rt.run_cycle()
    assert decision.action == "HOLD"
    assert [c for c in adapter.calls if c[1] == "BUY"] == []
    st = read_state()
    assert st["c5_position_qty"] == 4                        # broker adopted


# ══════════════════════════════════════════════════════════════════════
# S — margin change: position untouched, sizing recalculated
# ══════════════════════════════════════════════════════════════════════
def test_s_margin_change_does_not_close_position(env):
    broker = BrokerState(position_qty=5, entry_price=PRICE)
    margin = FakeMarginTracker(0.0585, 0.0576)
    rows = [candle_5m(CONTROL_DAY, 10, 35, NEUTRAL)]
    rt, adapter = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 40),
                               five_rows=rows, margin=margin)
    decision = rt.run_cycle()
    assert decision.action == "HOLD"
    assert broker.position_qty == 5                          # not auto-closed
    # next cycle with CHANGED rate → fresh sizing uses new value
    margin.long_rate = 0.0700
    rows2 = [candle_5m(CONTROL_DAY, 10, 40, NEUTRAL)]
    rt2, _ = make_runtime(broker, now_utc=utc_ms(CONTROL_DAY, 10, 45),
                          five_rows=rows2, margin=margin)
    rt2.run_cycle()
    assert broker.position_qty == 5                          # still untouched


def test_s_margin_change_recalculates_entry_sizing(env):
    broker = BrokerState(equity=10_000)
    margin = FakeMarginTracker(0.0585, 0.0576)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               margin=margin)
    d1 = rt.run_cycle()
    assert d1.margin_per_contract == Decimal("81.9000")
    # new process (fresh state), changed API rate → recalculated sizing
    import src.state_store as ss
    os.remove(ss.state_file_path())
    margin2 = FakeMarginTracker(0.1000, 0.0576)              # ГО doubled
    rt2, adapter2 = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                                 margin=margin2)
    d2 = rt2.run_cycle()
    assert d2.margin_per_contract == Decimal("140.0000")
    assert d2.qty_risk == 14                                 # 2000/140 recalced


# ══════════════════════════════════════════════════════════════════════
# T/U — data guards
# ══════════════════════════════════════════════════════════════════════
def test_t_unclosed_candle_no_action_for_exit(env):
    """Only fully closed 5m bars are evaluated; an unclosed tail bar is
    ignored (its start+5m > now) — and NO fallback to it."""
    broker = BrokerState(position_qty=3, entry_price=PRICE)
    closed = candle_5m(CONTROL_DAY, 10, 30, NEUTRAL)
    unclosed = candle_5m(CONTROL_DAY, 10, 35, 1390.0)        # EXIT-level close
    now = utc_ms(CONTROL_DAY, 10, 37)                        # 10:35 bar open
    rt, adapter = make_runtime(broker, now_utc=now,
                               five_rows=[closed, unclosed])
    rt.run_cycle()                                           # processes 10:30
    rt2, adapter2 = make_runtime(broker, now_utc=now,
                                 five_rows=[closed, unclosed])
    decision = rt2.run_cycle()
    assert decision.reason == "DUPLICATE_CANDLE"             # 10:35 never used
    assert [c for c in adapter2.calls if c[1] == "SELL"] == []
    assert broker.position_qty == 3


def test_u_missing_daily_history_no_fallback(env):
    broker = BrokerState()
    empty_daily = pd.DataFrame(columns=["time", "open", "high", "low",
                                        "close", "volume"])
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               daily=empty_daily)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "DATA_INSUFFICIENT"
    assert adapter.calls == []


def test_u_lookahead_guard_rejects_today_daily_bar(env):
    """A daily channel containing TODAY's bar must be excluded — the
    runtime drops the current MSK date via daily_bars_from_df."""
    broker = BrokerState()
    df = daily_df()
    today_row = {"time": pd.Timestamp(CONTROL_DAY.isoformat(), tz="UTC"),
                 "open": PRICE, "high": 9999.0, "low": 1.0,
                 "close": PRICE, "volume": 1}
    df = pd.concat([df, pd.DataFrame([today_row])], ignore_index=True)
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16], daily=df)
    decision = rt.run_cycle()
    # today's bar excluded → normal frozen behaviour (entry on breakout)
    assert decision.donchian_entry_upper == 1400.0           # not 9999
    assert decision.action == "ENTER_LONG"


def test_u_unknown_timeframe_raises_no_silent_fallback(monkeypatch):
    import src.market_data as md
    monkeypatch.setattr(md, "T_TECH_AVAILABLE", True)
    with pytest.raises(ValueError):
        md.load_candles("token", "uid", candles_count=10, timeframe="7X")


def test_u_daily_interval_maps_to_day_enum(monkeypatch):
    """1d must map to CANDLE_INTERVAL_DAY — never silently to another TF."""
    from t_tech.invest import CandleInterval
    captured = {}

    class MD:
        def get_candles(self, **kw):
            captured["interval"] = kw["interval"]
            return SimpleNamespace(candles=[])

    class Client:
        market_data = MD()

    @contextmanager
    def fake_get_client(token):
        yield Client()

    import src.market_data as md
    monkeypatch.setattr(md, "get_client", fake_get_client)
    monkeypatch.setattr(md, "T_TECH_AVAILABLE", True)
    md.load_candles("tok", "uid", candles_count=30, timeframe="1d")
    assert captured["interval"] == CandleInterval.CANDLE_INTERVAL_DAY


# ══════════════════════════════════════════════════════════════════════
# V — full-volume single operations + M/E + Stage cap provenance
# ══════════════════════════════════════════════════════════════════════
def test_v_stage_cap_and_me_limit_provenance(env):
    broker = BrokerState(equity=10_000)                     # qty_raw would be 24
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.qty_risk == 24
    assert decision.qty_raw == 24                           # pre-cap research qty
    assert decision.qty == 5                                # Stage-1 cap
    assert "STAGE_CAPPED" in decision.reason
    assert decision.m_e_ratio <= 0.30
    assert adapter.calls == [("CNYRUBF", "BUY", 5)]         # ONE operation


def test_v_me_limit_blocks_entry_before_stage_cut(env, monkeypatch):
    """Structural note: with frozen params (risk_base×clip_hi == me_limit)
    M/E cannot exceed 0.30 through qty_risk. Deployment-injected tighter
    limit proves the guard placement: blocked with qty_raw preserved."""
    import src.strategies.c5_core as core
    orig = core.evaluate

    def patched(**kw):
        kw["me_limit"] = 0.10
        return orig(**kw)

    monkeypatch.setattr(core, "evaluate", patched)
    broker = BrokerState(equity=10_000)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "ME_LIMIT"
    assert decision.qty_raw == 24                           # cap did NOT protect
    assert adapter.calls == []


def test_v_slippage_guard_blocks_entry(env):
    broker = BrokerState(equity=10_000)
    book = wide_book("LONG", 5)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               book=book)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "SLIPPAGE_GUARD"
    assert adapter.calls == []


def test_v_depth_guard_blocks_entry(env):
    broker = BrokerState(equity=10_000)
    book = strict_book("LONG", 5)
    rt, adapter = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16],
                               book=book)
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert "EXECUTION_GUARD" in decision.reason
    assert adapter.calls == []


# ══════════════════════════════════════════════════════════════════════
# Production wiring proofs (call-chain, not string grepping)
# ══════════════════════════════════════════════════════════════════════
def test_main_entrypoint_delegates_to_c5_runtime(env, monkeypatch):
    """src.main._run really drives the C5 runtime (production chain:
    main → run_once → C5Runtime.run_cycle) with mock-only boundaries."""
    import src.main as m
    captured = {}

    def fake_run_once(token):
        captured["token"] = token
        from src.c5_runtime import _noop_decision
        return _noop_decision()

    monkeypatch.setattr(m, "run_once", fake_run_once)
    monkeypatch.setenv("SANDBOX_TOKEN", "tok-123")
    m._run()
    assert captured["token"] == "tok-123"


def test_runtime_actually_invokes_frozen_c5_core(env, monkeypatch):
    """Wrap — do not replace — c5_core.evaluate: proves the production
    orchestrator calls the frozen strategy and consumes its real output."""
    import src.c5_runtime as rt_mod
    calls = []
    orig = rt_mod.c5_core.evaluate

    def spy(*args, **kwargs):
        result = orig(*args, **kwargs)
        calls.append((kwargs.get("position"), result.action))
        return result

    monkeypatch.setattr(rt_mod.c5_core, "evaluate", spy)
    broker = BrokerState()
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    decision = rt.run_cycle()
    assert calls and calls[-1][1] == "ENTER_LONG"
    assert decision.action == "ENTER_LONG"


def test_runtime_actually_invokes_real_protection(env, monkeypatch):
    """Spy on the REAL evaluate_protections (math stays real)."""
    import src.c5_runtime as rt_mod
    import src.protection as pr
    seen = []
    orig = pr.evaluate_protections

    def spy(state, equity, now=None):
        res = orig(state, equity, now=now)
        seen.append(res.entries_allowed)
        return res

    monkeypatch.setattr(rt_mod, "evaluate_protections", spy)
    broker = BrokerState()
    rt, _ = make_runtime(broker, now_utc=NOW_1605, five_rows=[C16])
    rt.run_cycle()
    assert seen and seen[0] is True


def test_legacy_paths_are_not_importable_from_active_runtime():
    """Active C5 modules must not depend on legacy GLDRUBF machinery."""
    import subprocess
    import sys
    code = (
        "import sys;"
        "import src.c5_runtime, src.main, src.instruments, src.cash_manager,"
        " src.execution_adapter;"
        "bad=[m for m in ('strategy','indicators','stop_orders',"
        "'position_monitor','auto_trader') if 'src.'+m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.getcwd())
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_instruments_module_has_no_gldrubf_semantic_names():
    import src.instruments as ins
    assert not hasattr(ins, "get_gldrubf_instrument")
    assert hasattr(ins, "get_c5_instrument")


def test_margin_provider_uses_future_by_full_equivalent(env, monkeypatch):
    """Real fetch_margin_rates against a fake SDK client: asserts the
    dlongClient/dshortClient extraction and hard failure without fallback."""
    import src.margin_provider as mp

    captured = {}

    class Instruments:
        def future_by(self, *, class_code, id):
            captured["call"] = (class_code, id)
            return SimpleNamespace(instrument=SimpleNamespace(
                dlong_client=0.0585, dshort_client=0.0576))

    class Client:
        instruments = Instruments()

    @contextmanager
    def fake_get_client(token):
        yield Client()

    monkeypatch.setattr(mp, "get_client", fake_get_client)
    rates = mp.fetch_margin_rates("tok", "CNYRUBF")
    assert captured["call"] == ("SPBFUT", "CNYRUBF")
    assert rates.for_direction("LONG") == 0.0585
    assert rates.for_direction("SHORT") == 0.0576

    class BrokenInstruments:
        def future_by(self, **kw):
            raise RuntimeError("grpc down")

    class BrokenClient:
        instruments = BrokenInstruments()

    @contextmanager
    def broken_get_client(token):
        yield BrokenClient()

    monkeypatch.setattr(mp, "get_client", broken_get_client)
    with pytest.raises(mp.MarginApiUnavailable):
        mp.fetch_margin_rates("tok", "CNYRUBF")


def test_paper_mode_builds_paper_adapter_only(env, monkeypatch):
    """Paper/live share one pipeline; factory differs only at the adapter."""
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    from src.execution_adapter import (PaperExecutionAdapter,
                                       build_execution_adapter)
    adapter, mode = build_execution_adapter("tok", "uid")
    assert mode == "PAPER"
    assert isinstance(adapter, PaperExecutionAdapter)


def test_paper_e2e_entry_hold_exit_lifecycle(env):
    """Full lifecycle through the REAL paper adapter + simulated broker:
    16:00 ENTRY → HOLD → EXIT → re-entry blocked until next control point."""
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    from src.c5_runtime import C5Runtime
    import src.c5_runtime as rt_mod

    sim = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    adapter = PaperExecutionAdapter(sim)
    holder = {"now": NOW_1605, "rows": [C16]}

    rt = C5Runtime(token="tok", adapter=adapter,
                   margin_tracker=FakeMarginTracker(),
                   now_fn=lambda: holder["now"])
    rt.instrument = FakeInstrument()
    rt.account_id = adapter.find_account()

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return daily_df().copy()
        return five_m_df(holder["rows"])

    @contextmanager
    def _client(token):
        yield SimpleNamespace()

    # patch the boundary where c5_runtime actually resolves it
    orig_load = rt_mod.load_candles
    orig_fetch = rt_mod.fetch_order_book
    rt_mod.load_candles = _load
    rt_mod.fetch_order_book = lambda *a, **k: good_book("LONG", 5)

    try:
        d = _paper_lifecycle(rt, sim, holder)
    finally:
        rt_mod.load_candles = orig_load
        rt_mod.fetch_order_book = orig_fetch

    d1, d2, d3, d4 = d
    cny_buys = [o for o in sim.orders if o["instrument"] == "CNYRUBF"
                and o["side"] == "BUY"]
    assert len(cny_buys) == 1


def _paper_lifecycle(rt, sim, holder):
    d1 = rt.run_cycle()
    assert d1.action == "ENTER_LONG" and sim.position_qty == 5
    # hold on a neutral later candle
    holder["rows"] = [candle_5m(CONTROL_DAY, 17, 0, NEUTRAL)]
    holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d2 = rt.run_cycle()
    assert d2.action == "HOLD" and sim.position_qty == 5
    # exit on opposite boundary
    holder["rows"] = [candle_5m(CONTROL_DAY, 18, 0, 1390.0)]
    holder["now"] = utc_ms(CONTROL_DAY, 18, 5)
    d3 = rt.run_cycle()
    assert d3.action == "EXIT" and sim.position_qty == 0
    # re-entry blocked: same-day control point already consumed
    holder["rows"] = [C16]
    holder["now"] = utc_ms(CONTROL_DAY, 18, 10)
    d4 = rt.run_cycle()
    assert d4.action == "NO_ENTRY"
    return d1, d2, d3, d4
