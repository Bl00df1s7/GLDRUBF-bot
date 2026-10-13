"""G-9C.1 — SHORT ENTRY (OPEN_SHORT via SELL) through the PAPER runtime.

Proves the REAL behaviour of the implemented slice, with fakes ONLY at
external boundaries (market data / clock / margin API / order book) and
the real PaperExecutionAdapter + SimulatedBroker + state_store behind it
(same fixture shape as tests/test_stage5b_paper_runtime.py).

Covered scenarios (task list):
  1. ENTER_SHORT dispatches a SELL tagged intent="OPEN_SHORT";
     ENTER_LONG keeps its previous BUY behaviour byte-for-byte.
  2. A confirmed SHORT open creates a NEGATIVE broker position and
     reserves the ГО (fail-closed without margin metadata).
  3. PARTIAL fills update the tracked position ONLY from the confirmed
     qty_executed, never from the requested decision.qty.
  4. ZERO_FILL / None result / adapter exception / UNVERIFIED qty do NOT
     create a phantom position.
  5. A save_state_strict() failure BEFORE dispatch sends NO order.
  6. An ambiguous outcome leaves a durable marker on disk.
  7. After a restart the unfinished ENTRY marker blocks any repeated
     SELL until explicit operator resolution.
  8. Legacy markers WITHOUT the "operation" field keep EXIT semantics.
  9. OPEN_SHORT over an open LONG and every other accidental reversal
     are refused fail-closed (broker level).

SHORT close-out via BUY is the NEXT dedicated slice and stays blocked —
nothing here claims the full short cycle is done.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

REF_DATE = dt.date(2026, 9, 30)
CONTROL_DAY = dt.date(2026, 10, 2)
LONG_BREAK = 1420.0
SHORT_BREAK = 1370.0
NEUTRAL = 1400.0
PRICE = 1400.0

# ══════════════════════════════════════════════════════════════════════
# boundary fakes — identical fixture math to Stage 5B:
#   daily H=L±DAILY_HL around C=1400 → Upper/Lower(10)=1400, ATR≈0.1563
#   LONG_BREAK=1420 > Upper → ENTER_LONG;  SHORT_BREAK=1370 < Lower →
#   ENTER_SHORT.  Sizing @ equity 1M → qty=2.
# ══════════════════════════════════════════════════════════════════════
DAILY_HL = 0.07815


def daily_df(n=60):
    import pandas as pd
    rows = []
    d = REF_DATE - dt.timedelta(days=n - 1)
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        rows.append({"time": ts, "open": PRICE, "high": PRICE + DAILY_HL,
                     "low": PRICE - DAILY_HL, "close": PRICE, "volume": 1})
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def candle_5m(day, hour, minute, close):
    import pandas as pd
    ts = pd.Timestamp(dt.datetime(day.year, day.month, day.day,
                                  hour, minute, tzinfo=MSK)
                      .astimezone(dt.timezone.utc))
    return {"time": ts, "open": close, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": 1}


def utc_ms(day, hour, minute):
    return dt.datetime(day.year, day.month, day.day, hour, minute,
                       tzinfo=MSK).astimezone(dt.timezone.utc)


class FakeInstrument:
    uid = "uid-cnyrubf-test"
    ticker = "CNYRUBF"
    class_code = "SPBFUT"


class FakeMarginTracker:
    def __init__(self):
        self.current = None
        self.refresh_calls = 0

    def refresh(self):
        from src.margin_provider import MarginRates
        self.refresh_calls += 1
        self.current = MarginRates(long_margin_rub=0.0585,
                                   short_margin_rub=0.0576,
                                   instrument="CNYRUBF")
        return self.current


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", state_file)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "true")
    import src.state_store as ss
    monkeypatch.setattr(ss, "STATE_FILE", state_file)
    return tmp_path


def read_state(state_env):
    return json.loads((state_env / "state.json").read_text())


@pytest.fixture
def market(monkeypatch):
    """Patch the two network boundaries of c5_runtime; restore after test."""
    import pandas as pd
    import src.c5_runtime as rt_mod

    holder = {"now": utc_ms(CONTROL_DAY, 16, 5),
              "rows": [],
              "daily": daily_df(),
              "load_calls": []}

    def _load(token, uid, candles_count=200, timeframe="4H"):
        holder["load_calls"].append(timeframe)
        if timeframe == "1d":
            return holder["daily"].copy()
        if timeframe == "5m":
            return pd.DataFrame(holder["rows"])
        raise AssertionError(f"C5 runtime must not request {timeframe!r}")

    saved_load, saved_book = rt_mod.load_candles, rt_mod.fetch_order_book
    rt_mod.load_candles = _load
    # Order book: ask 1400.05 / bid 1399.95 — a SELL entry crosses the
    # spread symmetrically to the LONG case (guard expected price ~ask).
    rt_mod.fetch_order_book = lambda *a, **k: ([(PRICE - 0.05, 100)],
                                               [(PRICE + 0.05, 100)])
    yield SimpleNamespace(holder=holder, mod=rt_mod)
    rt_mod.load_candles, rt_mod.fetch_order_book = saved_load, saved_book


def new_broker(cash=1_000_000):
    from src.execution_adapter import SimulatedBroker
    return SimulatedBroker(cash=cash, equity=cash)


def set_candles(market, *candles):
    market.holder["rows"] = list(candles)


def buy_orders(adapter):
    return [o for o in adapter.orders if o["side"] == "BUY"
            and o["instrument"] == "CNYRUBF"]


def sell_orders(adapter):
    return [o for o in adapter.orders if o["side"] == "SELL"
            and o["instrument"] == "CNYRUBF"]


# ══════════════════════════════════════════════════════════════════════
# Recording wrapper around the REAL PaperExecutionAdapter.
#
# ``scripted_sell`` lets a test inject arbitrary ExecutionResults (or
# exceptions) for OPENING sells ONLY — legacy/exit calls (without the
# extra kwargs) always delegate to the real broker so existing exit
# paths stay production.  The recorded call kwargs prove exactly what
# the runtime forwarded (ГО / entry_price / intent).
# ══════════════════════════════════════════════════════════════════════
class RecordingPaperAdapter:
    """Records every OPEN_SHORT dispatch and mirrors the PROVEN outcome
    into the real broker (same contract as the Stage-5B ScriptedSellAdapter
    applied to exits): FULL/PARTIAL → the broker applies the order with
    ONLY the confirmed executed qty; ZERO_FILL / UNVERIFIED / ok=False /
    exception / None → the broker state is NOT touched at all, so no
    phantom position can appear on either side."""

    def __init__(self, paper, scripted_sell=None, default_sell_raises=None):
        self._paper = paper
        self._scripted = list(scripted_sell or [])
        # Optional exception raised on any UNSCRIPTED OPEN_SHORT dispatch
        # (fault injection: simulate a send that never reaches the real
        # broker — durable marker must survive, position must not change).
        self._default_sell_raises = default_sell_raises
        self.sell_calls = []      # (qty, kwargs) per opening-sell attempt
        self.orders = paper.orders
        self.snapshot_calls = 0

    def find_account(self):
        return self._paper.find_account()

    def snapshot(self, account_id):
        self.snapshot_calls += 1
        return self._paper.snapshot(account_id)

    def buy_cny(self, account_id, qty, **kwargs):
        return self._paper.buy_cny(account_id, qty, **kwargs)

    def sell_cny(self, account_id, qty, **kwargs):
        from src.c5_runtime import _classify_entry_execution, \
            ENTRY_FILL_FULL, ENTRY_FILL_PARTIAL
        opening = bool(kwargs.get("intent"))
        if not opening:
            # Legacy EXIT path — production behaviour, never scripted.
            return self._paper.sell_cny(account_id, qty)
        self.sell_calls.append((int(qty), dict(kwargs)))
        if not self._scripted:
            # Default (unscripted) OPEN_SHORT dispatch: the REAL broker
            # executes it exactly as production PaperExecutionAdapter
            # would — full fill of the requested qty with ГО reserved.
            # Tests inject scripted_sell=[...] to force partial/zero/
            # ambiguous outcomes instead.
            return self._paper.sell_cny(account_id, qty, **kwargs)
        res = self._scripted.pop(0)
        if isinstance(res, BaseException):
            raise res
        if callable(res):
            res = res(qty, kwargs)
        if res is None:
            # adapter returned NOTHING — outcome unknown, broker untouched
            return res
        status, filled = _classify_entry_execution(res, int(qty))
        if status in (ENTRY_FILL_FULL, ENTRY_FILL_PARTIAL):
            # Mirror ONLY the confirmed volume through the REAL broker
            # accounting (margin reservation + negative position).
            self._paper.sell_cny(account_id, filled,
                                 margin_per_contract=kwargs.get(
                                     "margin_per_contract"),
                                 entry_price=kwargs.get("entry_price"),
                                 intent="OPEN_SHORT")
        return res

    def sell_one_tmon(self, account_id):
        return self._paper.sell_one_tmon(account_id)

    def buy_one_tmon(self, account_id):
        return self._paper.buy_one_tmon(account_id)

    def get_free_cash(self, account_id):
        return self._paper.get_free_cash(account_id)

    def get_tmon_quantity(self, account_id):
        return self._paper.get_tmon_quantity(account_id)

    def get_tmon_price(self, account_id):
        return self._paper.get_tmon_price(account_id)


def make_recording_runtime(market, broker, scripted_sell=None):
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter
    from src.runtime_config import TradingConfig

    adapter = RecordingPaperAdapter(PaperExecutionAdapter(broker),
                                    scripted_sell)
    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()
    return rt, adapter


def exec_res(ok, reason="PAPER_FILLED", qty_executed=None, order_id="ord-1"):
    from src.execution_adapter import ExecutionResult
    kw = {}
    if qty_executed is not None:
        kw["qty_executed"] = qty_executed
    return ExecutionResult(ok, reason, order_id=order_id, **kw)


SHORT_CANDLE = lambda: candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK)


# ── unit: entry execution classification is qty_executed-driven ───────
def test_g9c_classify_entry_execution_unit():
    from src.c5_runtime import (
        _classify_entry_execution, ENTRY_FILL_FULL, ENTRY_FILL_PARTIAL,
        ENTRY_FILL_ZERO, ENTRY_FILL_UNVERIFIED, ENTRY_FILL_FAILED)
    ER = exec_res
    assert _classify_entry_execution(ER(True, qty_executed=2), 2) \
        == (ENTRY_FILL_FULL, 2)
    assert _classify_entry_execution(ER(True, qty_executed=1), 2) \
        == (ENTRY_FILL_PARTIAL, 1)
    assert _classify_entry_execution(ER(True, qty_executed=0), 2) \
        == (ENTRY_FILL_ZERO, 0)
    assert _classify_entry_execution(ER(False, qty_executed=2), 2) \
        == (ENTRY_FILL_FAILED, 0)
    # a MISSING qty_executed attribute is never trustworthy
    from src.execution_adapter import ExecutionResult

    class _NoQty:
        ok = True
        reason = "OK"
    assert _classify_entry_execution(_NoQty(), 2) \
        == (ENTRY_FILL_UNVERIFIED, 0)          # attribute absent at all
    assert _classify_entry_execution(ExecutionResult(True, "OK"), 2) \
        == (ENTRY_FILL_ZERO, 0)                # default 0 → ZERO, no fill
    assert _classify_entry_execution(ER(True, qty_executed="2"), 2) \
        == (ENTRY_FILL_UNVERIFIED, 0)          # wrong type
    assert _classify_entry_execution(ER(True, qty_executed=True), 2) \
        == (ENTRY_FILL_UNVERIFIED, 0)          # bool is not a count
    assert _classify_entry_execution(ER(True, qty_executed=-1), 2) \
        == (ENTRY_FILL_UNVERIFIED, 0)
    assert _classify_entry_execution(ER(True, qty_executed=3), 2) \
        == (ENTRY_FILL_UNVERIFIED, 0)          # more than requested


# ══════════════════════════════════════════════════════════════════════
# 1 — ENTER_SHORT routes through SELL with intent="OPEN_SHORT";
#     ENTER_LONG keeps the plain BUY path.
# ══════════════════════════════════════════════════════════════════════
def test_short_entry_dispatches_sell_with_open_short_intent(state_env,
                                                            market):
    from src.execution_adapter import SimulatedBroker
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)

    d = rt.run_cycle()

    assert d.action == "ENTER_SHORT"
    assert d.qty == 2                          # frozen C5 sizing @ equity 1M
    # SELL dispatched with the OPEN_SHORT intent + ГО/entry metadata
    assert len(adapter.sell_calls) == 1
    qty, kwargs = adapter.sell_calls[0]
    assert qty == 2
    assert kwargs["intent"] == "OPEN_SHORT"
    # ГО per contract from c5_core: short rate × signal price × contract
    # size = 0.0576 × 1370 × 1000 (rate×price×size — NOT the guard price)
    assert kwargs["margin_per_contract"] == pytest.approx(78_912.0)
    # execution_guard walks the side WE TRADE AGAINST: for SHORT it sells
    # into BIDS → expected price is the bid-side VWAP 1399.95, NOT the ask
    # (src/execution_guard.py: `book = asks if direction == "LONG" else bids`).
    assert kwargs["entry_price"] == pytest.approx(1399.95)
    # NO BUY was ever attempted for a short entry
    assert buy_orders(adapter) == []
    st = read_state(state_env)
    assert st["c5_position_direction"] == "SHORT"
    assert st["c5_position_qty"] == -2
    assert not st.get("exit_pending_sell_intent")   # resolved after confirm


def test_long_entry_still_uses_plain_buy(state_env, market):
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK))
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)

    d = rt.run_cycle()

    assert d.action == "ENTER_LONG"
    assert len(buy_orders(adapter)) == 1
    assert sell_orders(adapter) == []
    assert adapter.sell_calls == []            # recording fake never invoked
    order = buy_orders(adapter)[0]
    assert order["qty"] == 2
    assert order["margin_per_contract"] == pytest.approx(83_070.0)  # 1420×1000×0.0585
    assert "intent" not in order               # pre-G-9C order shape intact
    assert broker.position_qty == 2
    st = read_state(state_env)
    assert st["c5_position_direction"] == "LONG"
    assert st["c5_position_qty"] == 2


# ══════════════════════════════════════════════════════════════════════
# 2 — confirmed SHORT open: negative position + ГО reservation (real
#     production adapter path, no faking inside the broker).
# ══════════════════════════════════════════════════════════════════════
def test_confirmed_short_open_negative_position_and_margin_reservation(
        state_env, market):
    from src.execution_adapter import PaperExecutionAdapter
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    import src.c5_runtime as rt_mod
    from src.runtime_config import TradingConfig
    paper = PaperExecutionAdapter(broker)
    rt = rt_mod.C5Runtime("fake-token", adapter=paper,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()

    d = rt.run_cycle()

    assert d.action == "ENTER_SHORT"
    # broker truth: NEGATIVE position, ГО reserved, basis recorded.
    # The reservation basis is decision.margin_per_contract computed by
    # c5_core from the SIGNAL price (1370 × 1000 × 0.0576 = 78 912), NOT
    # from the guard VWAP — same formula as the forwarded order metadata.
    assert broker.position_qty == -2
    assert broker.short_margin_locked_per_contract == pytest.approx(78_912.0)
    assert broker.cash == pytest.approx(1_000_000 - 2 * 78_912.0)
    # entry basis forwarded by the runtime = guard expected price on the
    # side we trade against (bids for a SELL) → 1399.95
    assert broker.entry_price == pytest.approx(1399.95)
    # runtime truth agrees with the broker snapshot (no divergence)
    st = read_state(state_env)
    assert st["c5_position_qty"] == -2
    assert st["c5_position_direction"] == "SHORT"
    # the dispatched order carries the OPEN_SHORT tag into the broker dict
    so = sell_orders(paper)
    assert len(so) == 1
    assert so[0]["intent"] == "OPEN_SHORT"
    assert so[0]["margin_per_contract"] == pytest.approx(78_912.0)


def test_short_open_without_margin_metadata_fails_closed():
    from src.execution_adapter import (SimulatedBroker,
                                       SimulatedBrokerError)
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2,
                 "intent": "OPEN_SHORT"})     # no margin_per_contract
    assert b.position_qty == 0                 # nothing mutated
    assert b.cash == 1_000_000


# ══════════════════════════════════════════════════════════════════════
# 3 — PARTIAL: only qty_executed is booked, never the requested qty.
# ══════════════════════════════════════════════════════════════════════
def test_partial_short_books_only_confirmed_qty(state_env, market):
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt, adapter = make_recording_runtime(
        market, broker, scripted_sell=[exec_res(True, qty_executed=1)])

    d = rt.run_cycle()

    assert d.action == "ENTER_SHORT"
    assert d.qty == 2                            # REQUEST was 2 …
    st = read_state(state_env)
    assert st["c5_position_qty"] == -1           # … but only 1 confirmed
    assert st["c5_position_direction"] == "SHORT"
    # the marker is gone: the outcome was proven terminal (confirmed fill)
    assert not st.get("exit_pending_sell_intent")


def test_partial_full_still_books_executed_not_requested(state_env, market):
    # sanity mirror for LONG: qty_executed drives the number too
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK))
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)
    d = rt.run_cycle()
    assert d.action == "ENTER_LONG"
    assert broker.position_qty == 2
    assert read_state(state_env)["c5_position_qty"] == 2


# ══════════════════════════════════════════════════════════════════════
# 4 — ZERO_FILL / None / exception / UNVERIFIED ⇒ NO phantom position.
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("outcome", ["zero", "none", "exception",
                                     "bad_qty", "rejected"])
def test_unproven_short_fill_creates_no_position(state_env, market,
                                                 outcome):
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    scripted = {
        "zero": exec_res(True, qty_executed=0),
        "none": None,                       # adapter returned NOTHING
        "exception": RuntimeError("network dropped mid-send"),
        "bad_qty": exec_res(True, qty_executed="two"),
        "rejected": exec_res(False, "ORDER_NOT_FILLED:REJECTED",
                             qty_executed=0),
    }[outcome]
    rt, adapter = make_recording_runtime(market, broker,
                                         scripted_sell=[scripted])

    d = rt.run_cycle()

    assert d.action == "NO_ENTRY"
    # no phantom position anywhere — runtime state AND broker stay flat
    st = read_state(state_env)
    assert st.get("c5_position_qty", 0) == 0
    assert st.get("c5_position_direction") in (None, "")
    assert broker.position_qty == 0
    assert broker.cash == 1_000_000          # no ГО reserved out of nowhere
    if outcome == "rejected":
        # ok=False is a proven rejection: EXECUTION_FAILED reason
        assert d.reason.startswith("EXECUTION_FAILED")
    elif outcome == "none":
        # G-9C contract: a None adapter response proves NEITHER execution
        # NOR non-execution — _classify_entry_execution() maps it to
        # UNVERIFIED (not FAILED), so no position is invented and the
        # durable ENTRY marker stays until explicit resolution.
        assert d.reason == "ENTRY_UNVERIFIED_PENDING_RECONCILE"
    else:
        assert d.reason == "ENTRY_UNVERIFIED_PENDING_RECONCILE"


# ══════════════════════════════════════════════════════════════════════
# 5 — pre-dispatch save_state_strict() failure ⇒ order NEVER sent.
# ══════════════════════════════════════════════════════════════════════
def test_pre_save_failure_blocks_short_dispatch(state_env, market):
    import src.state_store as ss
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)

    original = ss._atomic_write_state

    def flaky(state):
        # fail exactly while persisting the NEW OPEN_SHORT intent marker
        m = state.get("exit_pending_sell_intent")
        if isinstance(m, dict) and m.get("operation") == "ENTRY":
            raise IOError("simulated disk failure")
        return original(state)

    ss._atomic_write_state = flaky
    try:
        d = rt.run_cycle()
    finally:
        ss._atomic_write_state = original

    assert d.action == "NO_ENTRY"
    assert d.reason == "SELL_INTENT_PERSIST_FAILED"
    assert adapter.sell_calls == []             # broker NEVER saw the SELL
    assert broker.position_qty == 0
    st = read_state(state_env)
    assert not st.get("exit_pending_sell_intent")   # no marker leaked
    # control point NOT burned → healthy disk retries the atom next cycle
    assert st.get("last_c5_entry_control_timestamp") != \
        SHORT_CANDLE()["time"].isoformat()

    d2 = rt.run_cycle()
    assert d2.action == "ENTER_SHORT"
    assert len(adapter.sell_calls) == 1
    assert broker.position_qty == -2


# ══════════════════════════════════════════════════════════════════════
# 6/7 — ambiguous outcome leaves a durable marker; a restarted process
#       must NOT re-send the SELL until explicit operator resolution.
# ══════════════════════════════════════════════════════════════════════
def test_ambiguous_short_entry_leaves_durable_marker(state_env, market):
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt, adapter = make_recording_runtime(
        market, broker,
        scripted_sell=[RuntimeError("crashed before confirmation")])

    d = rt.run_cycle()

    assert d.reason == "ENTRY_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)
    marker = st.get("exit_pending_sell_intent")
    assert isinstance(marker, dict)
    assert marker["operation"] == "ENTRY"
    assert marker["side"] == "SELL"
    assert marker["intent"] == "OPEN_SHORT"
    assert marker["instrument"] == "CNYRUBF"
    assert marker["direction"] == "SHORT"
    assert marker["qty"] == 2
    assert marker["status"] == "SELL_OUTCOME_UNVERIFIED"
    assert marker["candle_ts"] == SHORT_CANDLE()["time"].isoformat()
    assert "created_at" in marker and "account_id_masked" in marker
    # no false position while fate is unknown
    assert st.get("c5_position_qty", 0) == 0
    assert broker.position_qty == 0


def test_restart_after_ambiguous_entry_blocks_resend(state_env, market):
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt1, ad1 = make_recording_runtime(
        market, broker,
        scripted_sell=[RuntimeError("crashed before confirmation")])
    rt1.run_cycle()
    assert ad1.sell_calls                        # first send happened

    # fresh process: same market/broker/state file, brand-new runtime
    rt2, ad2 = make_recording_runtime(market, broker)
    d2 = rt2.run_cycle()

    assert d2.action == "NO_ENTRY"
    assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    # G-9C hard invariant: NO repeated SELL after restart …
    assert ad2.sell_calls == []
    # … and reconcile never laundered the unknown residual into a
    # position: broker still flat, runtime state still flat, marker kept.
    assert broker.position_qty == 0
    st = read_state(state_env)
    assert isinstance(st.get("exit_pending_sell_intent"), dict)
    assert st["exit_pending_sell_intent"]["operation"] == "ENTRY"
    assert st.get("c5_position_qty", 0) == 0

    # wrong-operation clearance is refused (identity check, fail closed)
    ts = st["exit_pending_sell_intent"]["candle_ts"]
    from src.c5_runtime import resolve_pending_sell_intent
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=2,
                                       operation="EXIT") is False
    assert isinstance(st.get("exit_pending_sell_intent"), dict)
    # mismatched identity is refused too
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=1,
                                       operation="ENTRY") is False
    # explicit correct resolution clears it
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=2,
                                       operation="ENTRY") is True
    assert not st.get("exit_pending_sell_intent")


def test_reconcile_alone_never_clears_entry_marker(state_env, market):
    """Reconcile itself must not lift the block even when the broker
    snapshot happens to match a would-be short position."""
    set_candles(market, SHORT_CANDLE())
    broker = new_broker()
    rt1, ad1 = make_recording_runtime(
        market, broker,
        scripted_sell=[RuntimeError("crashed before confirmation")])
    rt1.run_cycle()
    # external reality: the order actually DID reach the broker
    broker.position_qty = -2
    st = read_state(state_env)
    assert isinstance(st.get("exit_pending_sell_intent"), dict)

    rt2, ad2 = make_recording_runtime(market, broker)
    d2 = rt2.run_cycle()

    assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert ad2.sell_calls == []                  # no blind re-send
    st2 = read_state(state_env)
    # marker SURVIVES reconcile untouched (snapshot is not proof)
    assert isinstance(st2.get("exit_pending_sell_intent"), dict)
    assert st2["exit_pending_sell_intent"]["operation"] == "ENTRY"


# ══════════════════════════════════════════════════════════════════════
# 8 — legacy markers WITHOUT "operation" keep the pre-G-9C EXIT meaning.
# ══════════════════════════════════════════════════════════════════════
LEGACY_MARKER = {
    "status": "SELL_IN_FLIGHT",
    "side": "SELL",
    "instrument": "CNYRUBF",
    "account_id_masked": "mask****",
    "candle_ts": "2026-10-02T13:00:00+00:00",
    "qty": 2,
    "created_at": "2026-10-02T13:05:00+00:00",
}


def test_legacy_marker_defaults_to_exit_operation():
    from src.c5_runtime import _marker_operation
    assert _marker_operation(dict(LEGACY_MARKER)) == "EXIT"
    assert _marker_operation({"operation": "entry"}) == "ENTRY"
    assert _marker_operation({"operation": "EXIT"}) == "EXIT"
    # unrecognised values fail closed to the most restrictive legacy op
    assert _marker_operation({"operation": "SHORT_OPEN"}) == "EXIT"
    assert _marker_operation(None) == "EXIT"


def test_legacy_marker_resolution_contract_unchanged():
    from src.c5_runtime import resolve_pending_sell_intent
    st = {"exit_pending_sell_intent": dict(LEGACY_MARKER)}
    # an EXIT-style request resolves it exactly like before G-9C
    assert resolve_pending_sell_intent(st, candle_ts=LEGACY_MARKER["candle_ts"],
                                       qty=2) is True
    assert not st.get("exit_pending_sell_intent")
    # a mislabelled ENTRY request must NOT launder a legacy exit marker
    st2 = {"exit_pending_sell_intent": dict(LEGACY_MARKER)}
    assert resolve_pending_sell_intent(st2,
                                       candle_ts=LEGACY_MARKER["candle_ts"],
                                       qty=2, operation="ENTRY") is False
    assert isinstance(st2.get("exit_pending_sell_intent"), dict)


def test_legacy_marker_survives_reconcile_as_before(state_env):
    """G-7 regression: a legacy (no-operation) marker is KEPT by
    reconcile — a broker snapshot never clears it."""
    from src.c5_runtime import reconcile
    from src.execution_adapter import AccountSnapshot
    st = {"c5_position_qty": 0, "c5_position_direction": None,
          "exit_pending_sell_intent": dict(LEGACY_MARKER)}
    snap = AccountSnapshot(account_id="acc", equity_rub=1_000_000,
                           free_cash_rub=1_000_000, position_qty=0,
                           entry_price=None, tmon_qty=0, open_orders=0)

    class _A:
        def snapshot(self, account_id):
            return snap

    reconcile(_A(), "acc", st)
    assert isinstance(st.get("exit_pending_sell_intent"), dict)
    assert st["exit_pending_sell_intent"] == LEGACY_MARKER


# ══════════════════════════════════════════════════════════════════════
# 9 — accidental reversals refused fail-closed at the broker level.
# ══════════════════════════════════════════════════════════════════════
def test_open_short_over_long_refused_no_flip():
    from src.execution_adapter import (SimulatedBroker,
                                       SimulatedBrokerError)
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 2,
             "margin_per_contract": 80_000.0, "entry_price": 1400.0})
    assert b.position_qty == 2
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2,
                 "margin_per_contract": 79_000.0, "entry_price": 1370.0,
                 "intent": "OPEN_SHORT"})
    assert b.position_qty == 2                  # NO accidental flip
    assert b.short_margin_locked_per_contract == 0.0


def test_plain_sell_cannot_reverse_or_oversell():
    from src.execution_adapter import (SimulatedBroker,
                                       SimulatedBrokerError)
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 2,
             "margin_per_contract": 80_000.0, "entry_price": 1400.0})
    cash_after_buy = b.cash
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 3})
    assert b.position_qty == 2                  # unchanged
    assert b.cash == cash_after_buy
    # a proper closing sell releases exactly the reserved ГО
    b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2})
    assert b.position_qty == 0
    assert b.cash == pytest.approx(1_000_000)
    assert b.margin_locked_per_contract == 0.0


def test_second_open_short_and_buy_over_short_refused():
    from src.execution_adapter import (SimulatedBroker,
                                       SimulatedBrokerError)
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 1,
             "margin_per_contract": 79_000.0, "entry_price": 1370.0,
             "intent": "OPEN_SHORT"})
    assert b.position_qty == -1
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 1,
                 "margin_per_contract": 79_000.0, "entry_price": 1370.0,
                 "intent": "OPEN_SHORT"})      # stacking refused
    assert b.position_qty == -1
    # SHORT close-out via BUY is the NEXT slice — refused, never silent
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 1,
                 "margin_per_contract": 79_000.0})
    assert b.position_qty == -1


def test_short_entry_over_existing_long_blocked_by_runtime(state_env,
                                                           market):
    """End-to-end: ENTER_SHORT can never flip an open LONG — the runtime
    routes a live position to the EXIT branch and the broker refuses any
    reversal WHILE THE LONG IS STILL OPEN; zero SELL reaches the broker
    as an OPEN_SHORT."""
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK))
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)
    assert rt.run_cycle().action == "ENTER_LONG"
    assert broker.position_qty == 2
    cash_with_long = broker.cash
    go_locked = broker.margin_locked_per_contract

    # WHILE THE LONG IS STILL OPEN: a direct OPEN_SHORT order must be
    # refused by the broker fail-closed — position, cash and ГО
    # reservation all stay byte-identical (no accidental reversal).
    from src.execution_adapter import SimulatedBrokerError
    with pytest.raises(SimulatedBrokerError):
        broker.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2,
                      "margin_per_contract": 79_000.0,
                      "entry_price": 1370.0, "intent": "OPEN_SHORT"})
    assert broker.position_qty == 2             # LONG intact, not flipped
    assert broker.cash == cash_with_long        # nothing reserved/released
    assert broker.margin_locked_per_contract == go_locked
    assert broker.short_margin_locked_per_contract == 0.0

    # new closed 5m bar whose close breaks BELOW Lower(10); the frozen
    # core evaluates this as an EXIT of the LONG (never a reversal), and
    # the production legacy sell path closes the position flat.
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK),
                candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK))
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    d = rt.run_cycle()

    assert d.action == "EXIT"
    assert broker.position_qty == 0
    # no OPEN_SHORT-tagged order was EVER dispatched while long
    assert all(o.get("intent") != "OPEN_SHORT" for o in adapter.orders)
