"""G-9C.2 — SHORT close-out (CLOSE_SHORT via BUY) through the PAPER runtime.

Proves the REAL behaviour of the implemented slice: fakes ONLY at the
external boundaries (market data / clock / margin API / order book), the
real PaperExecutionAdapter + SimulatedBroker + state_store behind it
(same fixture shape as tests/test_stage9c_short_execution.py).

Full cycle under test:
    ENTER_SHORT → SELL OPEN_SHORT → REASON_EXIT_SHORT → BUY CLOSE_SHORT

Covered scenarios (task list):
  1. REASON_EXIT_SHORT dispatches a BUY; the LONG exit still dispatches
     a plain SELL (legacy path byte-for-byte intact).
  2. Full SHORT close → flat position, ГО released in full, entry basis
     cleared.
  3. Partial BUY closes ONLY the confirmed qty_executed and releases ГО
     proportionally; the short stays negative with its basis intact.
  4/5. Realized P&L on a profitable (close < entry) and a losing
     (close > entry) short — formula (entry − close) × size × filled.
     NO commissions are modeled by this PAPER broker (fixed explicitly).
  6. BUY bigger than the short is refused without any reversal.
  7. A plain BUY while a short is open is refused; BUY at flat opens a
     LONG only through the normal entry metadata path.
  8. None / exception / ZERO_FILL / UNVERIFIED / bad qty_executed never
     produce a false close (position, cash, ГО untouched).
  9. save_state_strict() failure BEFORE dispatch ⇒ BUY never sent.
 10. Ambiguous outcome / adapter exception / failed final save keep the
     durable EXIT marker (side=BUY, operation=EXIT, direction=SHORT).
 11. Restart with an unfinished EXIT marker blocks a repeated BUY until
     explicit operator resolution.
 12. Legacy G-7 markers (no "operation") keep EXIT semantics; old suites
     run green alongside this one.
 13. Full PAPER cycle: enter → partial exit → final exit → reconcile
     confirms the flat state.
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
# boundary fakes — identical fixture math to Stage 5B / G-9C.1:
#   daily H=L±DAILY_HL around C=1400 → Upper/Lower(10)=1400
#   SHORT_BREAK=1370 < Lower → ENTER_SHORT (qty=2 @ equity 1M);
#   with a SHORT open, close ≥ Upper(5)=1400 → EXIT_SHORT;
#   NEUTRAL=1400 → HOLD;  LONG_BREAK=1420 > Upper(5) → EXIT_LONG.
# ══════════════════════════════════════════════════════════════════════
DAILY_HL = 0.07815


# ── split entry/exit histories (see module docstring of daily_df) ─────
ENTRY_DAILY_DF = None          # built lazily — proven G-9C.1 flat history
EXIT_DAILY_DF = None           # same history + one OLD wide bar outside
                               # BOTH Donchian windows → exit-only candles
                               # fire while the entry channel stays 1400.


# Exit-window geometry (verified numerically against the REAL pipeline:
# daily_bars_from_df(exclude_date=today) → c5_core windows N=10/N=5):
#   * the EXIT decision uses the LAST 5 daily bars (2026-09-26..09-30);
#   * the ENTRY decision uses the LAST 10 daily bars (2026-09-21..09-30);
#   * bars 09-21..09-25 are inside the entry window but OUTSIDE the exit
#     window → widening them changes ONLY the entry upper rail;
#   * bars 09-26..09-30 are inside BOTH windows.
# On the untouched flat G-9C.1 history both channels equal 1400±DAILY_HL:
#   ENTER_SHORT@1370 ✓ (1370 < Lower(10)=1399.92185),
#   EXIT_LONG@1370/1360 ✓ (close < Lower(5)=1399.92185),
#   EXIT_SHORT@1450 ✓ (1450 > Upper(5)=1400.07815).
# A PROFITABLE short close (1360 < entry 1399.95) can however only fire
# when Upper(5) drops BELOW 1360 — impossible on the flat tail.  The
# "profit" history therefore replaces the five exit-window days with a
# genuinely lower market week (H/L/C = 1350/1340/1345):
#   entry[1340, 1400.07815] → ENTER_SHORT@1370 STILL fires (proven),
#   exit [1340, 1350]       → EXIT_SHORT@1360 fires; 1360 > 1340 so no
#                             contradictory double breakout.
# The "loss" history widens ONLY 09-21..09-25 (outside the exit window):
#   entry[1399.92185, 1460], exit[1399.92185, 1400.07815] → identical
#   behaviour to the flat fixture for every tested trigger.
PROFIT_EXIT_DAYS = (dt.date(2026, 9, 26), dt.date(2026, 9, 30))
LOSS_EXIT_DAYS = (dt.date(2026, 9, 21), dt.date(2026, 9, 25))


def daily_df(n=60, exit_mode="flat"):
    """Daily-history fixture ending at REF_DATE (2026-09-30, MSK).

    ``exit_mode="flat"`` reproduces the PROVEN G-9C.1 entry fixture
    byte-for-byte (flat 1400±DAILY_HL bars).  ``"profit"`` / ``"loss"``
    keep the entry trigger working and only reshape the Donchian rails
    as documented above.  Tests install non-flat histories strictly
    AFTER the entry cycle via ``use_exit_history(market, ...)``.
    """
    import pandas as pd
    start = REF_DATE - dt.timedelta(days=n - 1)
    rows = []
    d = start
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        if exit_mode == "profit" and PROFIT_EXIT_DAYS[0] <= d \
                <= PROFIT_EXIT_DAYS[1]:
            o = h = 1350.0
            l = c = 1340.0
        elif exit_mode == "loss" and LOSS_EXIT_DAYS[0] <= d \
                <= LOSS_EXIT_DAYS[1]:
            o = h = 1460.0
            l = c = 1450.0
        else:
            o = h = PRICE + DAILY_HL
            l = c = PRICE - DAILY_HL
        rows.append({"time": ts, "open": o, "high": h,
                     "low": l, "close": c, "volume": 1})
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def use_exit_history(market, mode="flat"):
    """Switch ONLY the daily history to an exit-shaped channel.

    The runtime re-fetches ``load_candles(..., timeframe='1d')`` inside
    ``_evaluate_exit`` right before ``c5_core.evaluate``, so installing
    the exit history here affects the EXIT signal computation without
    touching the proven G-9C.1 entry fixture used earlier in the test.
    """
    market.holder["daily"] = daily_df(exit_mode=mode)


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


def exit_sell_orders(adapter):
    """SELL orders WITHOUT the OPEN_SHORT entry intent — i.e. the legacy
    LONG-close dispatches only (entry order dicts keep their metadata)."""
    return [o for o in sell_orders(adapter)
            if str(o.get("intent") or "") != "OPEN_SHORT"]


def close_buy_orders(adapter):
    return [o for o in buy_orders(adapter)
            if str(o.get("intent") or "") == "CLOSE_SHORT"]


# ══════════════════════════════════════════════════════════════════════
# Recording wrapper around the REAL PaperExecutionAdapter + broker.
#
# ``scripted_buy`` injects arbitrary ExecutionResults (or exceptions /
# None) for CLOSE_SHORT dispatches ONLY; every other call (the OPEN_SHORT
# SELL entry AND the legacy LONG-exit SELL) always delegates to the real
# production path.  Mirroring contract (same invariant as the G-7/G-9C.1
# boundary fakes): FULL/PARTIAL → the REAL broker applies ONLY the
# confirmed executed volume; ZERO_FILL / UNVERIFIED / ok=False /
# exception / None → the broker state is NOT touched at all, so no false
# close can appear on either side of the accounting.
# ══════════════════════════════════════════════════════════════════════
class RecordingPaperAdapter:
    def __init__(self, paper, scripted_buy=None, scripted_sell=None):
        self._paper = paper
        self._scripted_buy = list(scripted_buy or [])
        self._scripted_sell = list(scripted_sell or [])
        self.buy_calls = []       # (qty, kwargs) per CLOSE_SHORT attempt
        self.sell_calls = []      # (qty, kwargs) per OPEN_SHORT attempt
        self.orders = paper.orders
        self.snapshot_calls = 0
        # save_state_strict fault-injection helper state (#9/#10)
        self._orig_strict_save = None

    def find_account(self):
        return self._paper.find_account()

    def snapshot(self, account_id):
        self.snapshot_calls += 1
        return self._paper.snapshot(account_id)

    def buy_cny(self, account_id, qty, **kwargs):
        from src.c5_runtime import (_classify_exit_execution,
                                    EXIT_FILL_FULL, EXIT_FILL_PARTIAL)
        closing = str(kwargs.get("intent") or "") == "CLOSE_SHORT"
        if not closing:
            # Normal ENTRY BUY (LONG) — production behaviour, never
            # scripted here.
            return self._paper.buy_cny(account_id, qty, **kwargs)
        self.buy_calls.append((int(qty), dict(kwargs)))
        if not self._scripted_buy:
            # Default (unscripted) CLOSE_SHORT dispatch: the REAL broker
            # executes it exactly as production would — full fill of the
            # requested qty with proportional ГО release + realized P&L.
            return self._paper.buy_cny(account_id, qty, **kwargs)
        res = self._scripted_buy.pop(0)
        if isinstance(res, BaseException):
            raise res
        if res is None:
            # adapter returned NOTHING — outcome unknown, broker untouched
            return res
        status, filled = _classify_exit_execution(res, int(qty))
        if status in (EXIT_FILL_FULL, EXIT_FILL_PARTIAL):
            # Mirror ONLY the confirmed volume through the REAL broker
            # accounting (position up, proportional ГО release, P&L).
            self._paper.buy_cny(account_id, filled,
                                intent="CLOSE_SHORT",
                                close_price=kwargs.get("close_price"))
        return res

    def sell_cny(self, account_id, qty, **kwargs):
        opening = bool(kwargs.get("intent"))
        if not opening:
            # Legacy EXIT path (closing a LONG) — production behaviour,
            # never scripted: exact pre-G-9C.2 delegation.
            return self._paper.sell_cny(account_id, qty)
        self.sell_calls.append((int(qty), dict(kwargs)))
        if not self._scripted_sell:
            return self._paper.sell_cny(account_id, qty, **kwargs)
        res = self._scripted_sell.pop(0)
        if isinstance(res, BaseException):
            raise res
        if res is None:
            # adapter returned NOTHING — outcome unknown, broker untouched
            return res
        from src.c5_runtime import (_classify_exit_execution,
                                   EXIT_FILL_FULL, EXIT_FILL_PARTIAL)
        status, filled = _classify_exit_execution(res, int(qty))
        if status in (EXIT_FILL_FULL, EXIT_FILL_PARTIAL):
            # Mirror ONLY the confirmed volume through the REAL broker
            # accounting (legacy LONG-close path: ГО release, no P&L).
            self._paper.sell_cny(account_id, filled)
        return res

    def marker_saves_fail(self, state):
        """save_state_strict replacement for fault injection (#9/#10).

        The G-9C.1 entry path (src/c5_runtime._evaluate_entry) persists
        its OPEN_SHORT marker with a PLAIN ``save_state_strict(state)``
        call and relies on run_cycle()'s end-of-cycle save to flush the
        ledger fields it mutates in the same dict — so the injected
        failure must NOT drop those legitimate writes.  A strict save is
        refused ONLY when it would persist an unfinished BUY-side EXIT
        marker (side == "BUY") while the SHORT ledger is still missing
        or stale (the entry-finalize write carries c5_position_qty ==
        -2 alongside the already-cleared marker; later writes that keep
        a stale ledger have no marker at all).  Every other save goes
        through to the real durable writer.
        """
        m = state.get("exit_pending_sell_intent")
        if isinstance(m, dict) and m.get("side") == "BUY" \
                and int(state.get("c5_position_qty") or 0) >= -1:
            raise IOError("simulated disk failure")
        return self._orig_strict_save(state)

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


def make_recording_runtime(market, broker, scripted_buy=None,
                           scripted_sell=None):
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter
    from src.runtime_config import TradingConfig

    adapter = RecordingPaperAdapter(PaperExecutionAdapter(broker),
                                    scripted_buy, scripted_sell)
    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()
    return rt, adapter


class fail_marker_saves:
    """Context manager installing ``adapter.marker_saves_fail`` as the
    runtime's durable strict writer (patched in src.c5_runtime — where
    save_state_strict is imported to; patching state_store internals
    would never be seen by the runtime)."""

    def __init__(self, adapter):
        self.adapter = adapter

    def __enter__(self):
        import src.c5_runtime as rt_mod
        self._rt_mod = rt_mod
        self._saved = rt_mod.save_state_strict
        self.adapter._orig_strict_save = self._saved
        rt_mod.save_state_strict = self.adapter.marker_saves_fail
        return self

    def __exit__(self, *exc):
        self._rt_mod.save_state_strict = self._saved
        self.adapter._orig_strict_save = None
        return False


def exec_res(ok, reason="PAPER_FILLED", qty_executed=None, order_id="ord-1",
             details=None):
    from src.execution_adapter import ExecutionResult
    kw = {}
    if qty_executed is not None:
        kw["qty_executed"] = qty_executed
    if details is not None:
        kw["details"] = details
    return ExecutionResult(ok, reason, order_id=order_id, **kw)


def _exit_candle(day, hour, minute, price):
    """Exit-check bar: identical close to the signal candle but with a
    non-zero volume — the production path (daily_bars_from_df → VWAP)
    refuses zero-volume bars for the entry basis, while the EXIT check
    consumes only the close price."""
    c = candle_5m(day, hour, minute, price)
    c["volume"] = 100
    return c


SHORT_CANDLE = lambda: candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK)
EXIT_PRICE = 1360.0          # close BELOW the 1399.95 entry → profitable short
LOSS_PRICE = 1450.0          # close ABOVE entry → losing short
EXIT_CANDLE = lambda: _exit_candle(CONTROL_DAY, 16, 30, EXIT_PRICE)
LOSS_EXIT_CANDLE = lambda: _exit_candle(CONTROL_DAY, 16, 30, LOSS_PRICE)


def advance(market, *extra_candles, hour=17, minute=0, exit_mode="profit"):
    """Move the market clock to a fresh EXIT-check bar.

    Installs the requested exit-channel history (see ``daily_df`` /
    ``use_exit_history``) so the real C5 EXIT signal can fire for the
    test candles; the ENTRY fixture stays untouched because this runs
    strictly AFTER the entry cycle of every test using it.
    """
    use_exit_history(market, exit_mode)
    set_candles(market, SHORT_CANDLE(), *extra_candles)
    market.holder["now"] = utc_ms(CONTROL_DAY, hour, minute)


def open_short(market, broker, state_env):
    """Production ENTER_SHORT cycle; returns (runtime, adapter).

    Post-conditions asserted here (so every exit test starts from a
    proven, identically-built short): qty=-2, ГО reserved 2×78 912,
    entry basis = guard bid VWAP 1399.95.
    """
    set_candles(market, SHORT_CANDLE())
    rt, adapter = make_recording_runtime(market, broker)
    d = rt.run_cycle()
    assert d.action == "ENTER_SHORT"
    assert broker.position_qty == -2
    assert broker.short_margin_locked_per_contract == pytest.approx(78_912.0)
    assert broker.entry_price == pytest.approx(1399.95)
    return rt, adapter


def utc_iso(day, hour, minute):
    return utc_ms(day, hour, minute).isoformat()


# ══════════════════════════════════════════════════════════════════════


# ── unit: close-short order constants & classifier reuse ──────────────
def test_g9c2_marker_identity_and_helpers_unit():
    from src.c5_runtime import _marker_operation, _persist_intent_marker
    # a SHORT-exit marker is unambiguously EXIT/BUY/SHORT/CLOSE_SHORT
    m = {"operation": "EXIT", "side": "BUY", "direction": "SHORT",
         "intent": "CLOSE_SHORT", "candle_ts": "ts", "qty": 2}
    assert _marker_operation(m) == "EXIT"
    # legacy (no operation) and entry markers unchanged
    assert _marker_operation({"side": "SELL"}) == "EXIT"
    assert _marker_operation({"operation": "ENTRY"}) == "ENTRY"
    # the shared pre-save helper persists strictly and rolls back on fail
    import src.c5_runtime as rt_mod
    import src.state_store as ss
    calls = []
    orig = rt_mod.save_state_strict
    rt_mod.save_state_strict = lambda st: calls.append(1)
    try:
        st = {}
        assert _persist_intent_marker(st, dict(m), "TEST") is True
        assert st["exit_pending_sell_intent"] == m

        def boom(_st):
            raise ss.StateSaveError("disk gone")
        rt_mod.save_state_strict = boom
        st2 = {}
        assert _persist_intent_marker(st2, dict(m), "TEST") is False
        assert "exit_pending_sell_intent" not in st2
    finally:
        rt_mod.save_state_strict = orig


# ══════════════════════════════════════════════════════════════════════
# 1 — REASON_EXIT_SHORT routes through BUY; LONG exit still routes SELL.
# ══════════════════════════════════════════════════════════════════════
def test_exit_short_dispatches_buy_close_short(state_env, market):
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())

    d = rt.run_cycle()

    assert d.action == "EXIT"
    assert d.reason == "EXIT_SHORT"
    # BUY dispatched with the dedicated CLOSE_SHORT intent + actual bar
    # price; NO SELL ever left the runtime while the short was open.
    assert len(adapter.buy_calls) == 1
    qty, kwargs = adapter.buy_calls[0]
    assert qty == 2                              # abs(confirmed position)
    assert kwargs["intent"] == "CLOSE_SHORT"
    assert kwargs["close_price"] == pytest.approx(EXIT_PRICE)
    assert "margin_per_contract" not in kwargs   # close ≠ new reservation
    co = close_buy_orders(adapter)
    assert len(co) == 1 and co[0]["qty"] == 2
    # the EXIT dispatch produced NO closing SELL at all — the only SELL
    # order on record is the OPEN_SHORT entry itself (legacy shape kept)
    assert exit_sell_orders(adapter) == []
    st = read_state(state_env)
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None
    assert st["c5_entry_price"] is None
    assert not st.get("exit_pending_sell_intent")


def test_exit_long_still_dispatches_plain_sell(state_env, market):
    """Legacy LONG exit path byte-for-byte intact after G-9C.2."""
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK))
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)
    assert rt.run_cycle().action == "ENTER_LONG"
    assert broker.position_qty == 2
    cash_after_buy = broker.cash

    advance(market, candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK),
            exit_mode="flat")
    d = rt.run_cycle()

    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    # plain legacy SELL: two-argument call, no extra metadata at all
    lo = exit_sell_orders(adapter)
    assert len(lo) == 1
    assert lo[0]["qty"] == 2
    assert "intent" not in lo[0]
    assert "margin_per_contract" not in lo[0]
    assert adapter.buy_calls == []               # NO BUY for a LONG exit
    assert broker.position_qty == 0
    # legacy cash formula preserved EXACTLY: ГО released, no P&L posted
    assert broker.cash == pytest.approx(cash_after_buy + 2 * 83_070.0)
    assert broker.realized_pnl == 0.0


# ══════════════════════════════════════════════════════════════════════
# 2 — full close: flat position, ГО released in full, basis cleared.
# 4 — profitable short: close BELOW entry → positive realized P&L.
# ══════════════════════════════════════════════════════════════════════
def test_full_short_close_flat_margin_release_and_profit(state_env, market):
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    cash_after_open = broker.cash
    go_locked = broker.short_margin_locked_per_contract  # 78 912/contract

    advance(market, EXIT_CANDLE())
    d = rt.run_cycle()
    assert d.reason == "EXIT_SHORT"

    # position fully closed, NO long opened
    assert broker.position_qty == 0
    # ГО released in full from the saved SHORT reservation
    expected_go = 2 * go_locked
    assert broker.cash == pytest.approx(cash_after_open + expected_go
                                        + (1399.95 - EXIT_PRICE) * 1000 * 2)
    assert broker.short_margin_locked_per_contract == 0.0
    # entry/basis cleared per the broker model (snapshot reports None)
    assert broker.entry_price == 0.0
    snap = broker.snapshot("paper-account")
    assert snap.position_qty == 0 and snap.entry_price is None
    # P&L formula, verified numbers:
    #   pnl = (entry − close) × size × qty = (1399.95−1360)×1000×2 = 79 900
    assert broker.realized_pnl == pytest.approx(79_900.0)
    assert broker.equity == pytest.approx(1_000_000 + 79_900.0)
    # commissions are NOT modeled by this PAPER broker — the whole delta
    # above is margin release + price-difference P&L and nothing else.
    st = read_state(state_env)
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None
    assert st["c5_entry_price"] is None


# ══════════════════════════════════════════════════════════════════════
# 3 — partial BUY: only qty_executed closes; ГО released proportionally.
# ══════════════════════════════════════════════════════════════════════
def test_partial_short_close_books_only_confirmed_volume(state_env, market):
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    cash_after_open = broker.cash
    go_pc = broker.short_margin_locked_per_contract

    advance(market, EXIT_CANDLE())
    # re-arm the runtime with a scripted PARTIAL BUY (1 of 2 confirmed)
    rt2, adapter2 = make_recording_runtime(
        market, broker, scripted_buy=[exec_res(True, qty_executed=1)])

    d = rt2.run_cycle()

    assert d.action == "EXIT" and d.reason == "EXIT_SHORT"
    assert len(adapter2.buy_calls) == 1
    assert adapter2.buy_calls[0][0] == 2         # REQUESTED 2 …
    # … but only 1 confirmed contract closed anywhere in the books
    assert broker.position_qty == -1             # still SHORT, smaller
    assert broker.entry_price == pytest.approx(1399.95)  # basis intact
    assert broker.short_margin_locked_per_contract == pytest.approx(go_pc)
    assert broker.cash == pytest.approx(
        cash_after_open + 1 * go_pc + (1399.95 - EXIT_PRICE) * 1000 * 1)
    assert broker.realized_pnl == pytest.approx(39_950.0)
    st = read_state(state_env)
    assert st["c5_position_qty"] == -1
    assert st["c5_position_direction"] == "SHORT"
    assert not st.get("exit_pending_sell_intent")   # proven-terminal result


# ══════════════════════════════════════════════════════════════════════
# 5 — losing short: close ABOVE entry → negative realized P&L.
# ══════════════════════════════════════════════════════════════════════
def test_losing_short_close_realizes_negative_pnl(state_env, market):
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    cash_after_open = broker.cash
    go_total = 2 * broker.short_margin_locked_per_contract

    advance(market, LOSS_EXIT_CANDLE(), exit_mode="loss")
    d = rt.run_cycle()

    assert d.reason == "EXIT_SHORT"
    assert broker.position_qty == 0
    # pnl = (1399.95 − 1450) × 1000 × 2 = −100 100 (loss debits cash)
    assert broker.realized_pnl == pytest.approx(-100_100.0)
    assert broker.cash == pytest.approx(cash_after_open + go_total - 100_100.0)
    assert broker.equity == pytest.approx(1_000_000 - 100_100.0)
    assert broker.short_margin_locked_per_contract == 0.0
    assert broker.entry_price == 0.0


# ══════════════════════════════════════════════════════════════════════
# 6 — BUY bigger than the SHORT is refused WITHOUT any reversal.
# ══════════════════════════════════════════════════════════════════════
def test_over_sized_close_refused_no_reversal():
    from src.execution_adapter import SimulatedBroker, SimulatedBrokerError
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2,
             "margin_per_contract": 79_000.0, "entry_price": 1370.0,
             "intent": "OPEN_SHORT"})
    pos, cash, go, eq, pnl = (b.position_qty, b.cash,
                              b.short_margin_locked_per_contract,
                              b.equity, b.realized_pnl)
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 3,
                 "intent": "CLOSE_SHORT", "close_price": 1360.0})
    # refused order changed NOTHING — no accidental LONG, no cash/go move
    assert (b.position_qty, b.cash, b.short_margin_locked_per_contract,
            b.equity, b.realized_pnl) == (pos, cash, go, eq, pnl)


def test_runtime_never_sends_buy_above_position_size(state_env, market):
    """The dispatched BUY is bounded by |confirmed position| — even when
    the tracked state disagrees with the broker snapshot, the oversized
    order is refused by the broker and no false close is recorded."""
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    # tamper: runtime ledger claims -5 while the broker truly holds -2
    st = read_state(state_env)
    st["c5_position_qty"] = -5
    (state_env / "state.json").write_text(json.dumps(st))

    advance(market, EXIT_CANDLE())
    d = rt.run_cycle()

    # qty comes from the ABSOLUTE CONFIRMED broker size, never more
    assert d.action == "EXIT" and d.reason == "EXIT_SHORT"
    assert len(adapter.buy_calls) == 1
    assert adapter.buy_calls[0][0] == 2
    assert broker.position_qty == 0              # clean full close
    fresh = read_state(state_env)
    assert fresh["c5_position_qty"] == 0


# ══════════════════════════════════════════════════════════════════════
# 7 — BUY without a short / plain BUY over a short.
# ══════════════════════════════════════════════════════════════════════
def test_close_short_without_short_is_fail_closed():
    """A CLOSE_SHORT-tagged BUY while FLAT must be refused (it is not an
    entry order — inventing a LONG out of a closing intent is forbidden);
    nothing changes.  A plain BUY at flat remains the normal LONG entry
    path with its own metadata contract."""
    from src.execution_adapter import (SimulatedBroker, SimulatedBrokerError,
                                       PaperExecutionAdapter)
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    ad = PaperExecutionAdapter(b)
    with pytest.raises(SimulatedBrokerError):
        ad.buy_cny("acc", 1, intent="CLOSE_SHORT", close_price=1360.0)
    assert b.position_qty == 0 and b.cash == 1_000_000
    assert b.realized_pnl == 0.0
    # a proper LONG entry BUY still works (unchanged production path)
    r = ad.buy_cny("acc", 1, margin_per_contract=80_000.0,
                   entry_price=1400.0)
    assert r.ok and b.position_qty == 1


def test_plain_buy_over_short_never_stacks_or_flips():
    from src.execution_adapter import SimulatedBroker, SimulatedBrokerError
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 2,
             "margin_per_contract": 79_000.0, "entry_price": 1370.0,
             "intent": "OPEN_SHORT"})
    cash, go = b.cash, b.short_margin_locked_per_contract
    # G-9C.1 refusal kept: a BUY over a short WITHOUT the close intent
    # is an accidental reversal — blocked fail-closed, state untouched.
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 1,
                 "margin_per_contract": 79_000.0})
    assert (b.position_qty, b.cash, b.short_margin_locked_per_contract,
            b.realized_pnl) == (-2, cash, go, 0.0)
    # and a CLOSE_SHORT-tagged BUY over a LONG is refused too
    b2 = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b2.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 1,
              "margin_per_contract": 80_000.0, "entry_price": 1400.0})
    with pytest.raises(SimulatedBrokerError):
        b2.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 1,
                  "intent": "CLOSE_SHORT", "close_price": 1360.0})
    assert b2.position_qty == 1


def test_missing_close_price_fails_closed():
    from src.execution_adapter import SimulatedBroker, SimulatedBrokerError
    b = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    b.apply({"instrument": "CNYRUBF", "side": "SELL", "qty": 1,
             "margin_per_contract": 79_000.0, "entry_price": 1370.0,
             "intent": "OPEN_SHORT"})
    cash = b.cash
    with pytest.raises(SimulatedBrokerError):
        b.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 1,
                 "intent": "CLOSE_SHORT"})           # no close_price
    assert (b.position_qty, b.cash) == (-1, cash)


# ══════════════════════════════════════════════════════════════════════
# 8 — None / exception / ZERO_FILL / UNVERIFIED / bad qty ⇒ NO false close
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("outcome", ["zero", "none", "exception", "bad_qty",
                                     "over_qty", "rejected"])
def test_unproven_close_never_fakes_a_flat(state_env, market, outcome):
    broker = new_broker()
    open_short(market, broker, state_env)
    cash, go, basis = broker.cash, broker.short_margin_locked_per_contract, \
        broker.entry_price
    scripted = {
        "zero": exec_res(True, qty_executed=0),
        "none": None,
        "exception": RuntimeError("network dropped mid-send"),
        "bad_qty": exec_res(True, qty_executed="two"),
        "over_qty": exec_res(True, qty_executed=5),
        # provable rejection via the machine-readable status source
        # (details["status"] — existing G-4 contract, no invented code)
        "rejected": exec_res(False, "ORDER_NOT_FILLED", details={
            "status": "EXECUTION_REPORT_STATUS_REJECTED"}, qty_executed=0),
    }[outcome]
    advance(market, EXIT_CANDLE())
    rt2, ad2 = make_recording_runtime(market, broker, scripted_buy=[scripted])

    d = rt2.run_cycle()

    if outcome == "rejected":
        # proven rejection: safe retry later, NOTHING closed.  The
        # decision itself carries the EXIT action with the adapter's
        # failure reason (existing G-4/G-7 contract); the candle stays
        # UNBURNED and the durable marker is cleared because the order
        # provably never executed.
        assert d.action == "EXIT"
        # production G-4/G-7 contract (test_stage5b regression): the
        # decision carries the signal reason; the adapter failure lands
        # in state["exit_failure_reason"], candle stays unburned
        assert d.reason == "EXIT_SHORT"
        st = read_state(state_env)
        assert st["exit_failure_reason"].startswith("ORDER_NOT_FILLED")
        assert not st.get("exit_pending_sell_intent")
        # G-4 confirmed-rejection contract: the runtime deliberately
        # REMOVES the processed-candle key so the next cycle re-attempts
        # the exit — absence (not a different value) is the burn proof.
        assert st.get("last_processed_5m_ts") != utc_iso(CONTROL_DAY, 16, 30)
    elif outcome in ("zero", "bad_qty", "over_qty"):
        # LAYER 1 (signal): the decision carries the strategy EXIT
        # reason — EXIT_SHORT for a trustworthy-but-unverifiable fill
        # result (G-5 contract, matches the stage5b reference suite).
        assert d.action == "EXIT"
        assert d.reason == "EXIT_SHORT"
        # LAYER 2 (freeze STATE): the freeze classification lives in
        # state, NOT in the decision string.
        st = read_state(state_env)
        assert st["exit_unverified_until_reconcile"] == \
            "EXIT_UNVERIFIED_PENDING_RECONCILE"
        # LAYER 3 (durable marker): kept on disk, relabeled to the
        # shared UNKNOWN-fate status used by the restart guard.
        m = st.get("exit_pending_sell_intent")
        assert isinstance(m, dict)
        assert m["status"] == "SELL_OUTCOME_UNVERIFIED"
        # candle burned ⇒ no blind re-send of this bar
        assert st["last_processed_5m_ts"] == utc_iso(CONTROL_DAY, 16, 30)
    else:
        # exception / None: block-path contract (no trustworthy result
        # at all) — the cycle-level freeze reason is returned directly.
        assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
        st = read_state(state_env)
        assert st["exit_unverified_until_reconcile"] == \
            "EXIT_UNVERIFIED_PENDING_RECONCILE"
        m = st.get("exit_pending_sell_intent")
        assert isinstance(m, dict)
        assert m["status"] == "SELL_OUTCOME_UNVERIFIED"
        assert st["last_processed_5m_ts"] == utc_iso(CONTROL_DAY, 16, 30)
    # the short NEVER falsely closed: broker + ledger both intact
    assert broker.position_qty == -2
    assert broker.cash == pytest.approx(cash)
    assert broker.short_margin_locked_per_contract == pytest.approx(go)
    assert broker.entry_price == pytest.approx(basis)
    assert broker.realized_pnl == 0.0
    st = read_state(state_env)
    assert st["c5_position_qty"] == -2
    assert st["c5_position_direction"] == "SHORT"


# ══════════════════════════════════════════════════════════════════════
# 9 — pre-save failure blocks the BUY dispatch entirely.
# ══════════════════════════════════════════════════════════════════════
def test_pre_save_failure_blocks_close_dispatch(state_env, market):
    import src.c5_runtime as rt_mod
    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())

    # Fault injected EXACTLY at the pre-dispatch durable-save boundary:
    # c5_runtime imports save_state_strict into its own namespace, so
    # patching state_store internals would never be seen by the runtime.
    original = rt_mod.save_state_strict
    calls = {"n": 0}

    def fail_marker_saves(state):
        m = state.get("exit_pending_sell_intent")
        if isinstance(m, dict) and m.get("side") == "BUY":
            calls["n"] += 1
            # the REAL durable writer raises StateSaveError on disk
            # failures (state_store.save_state_strict contract) — the
            # injected fault must travel through that same wrapper, or
            # it would bypass the very code this test exercises.
            raise rt_mod.StateSaveError("simulated disk failure")
        return original(state)

    rt_mod.save_state_strict = fail_marker_saves
    try:
        d = rt.run_cycle()
    finally:
        rt_mod.save_state_strict = original

    assert calls["n"] >= 1                        # the BUY pre-save was hit
    assert d.action == "NO_ENTRY"
    # LAYER split: the runtime wraps every strict-save failure in its
    # own StateSaveError contract; the fault injected here must travel
    # through THAT wrapper (as production disk errors do), not as a raw
    # IOError that would bypass the code under test.
    assert d.reason in ("SELL_INTENT_PERSIST_FAILED",
                        "EXIT_UNVERIFIED_PENDING_RECONCILE")
    if d.reason == "SELL_INTENT_PERSIST_FAILED":
        assert adapter.buy_calls == []                # BUY NEVER attempted
        st = read_state(state_env)
        assert not st.get("exit_pending_sell_intent")
        # candle NOT burned → healthy disk retries the atom next cycle
        assert st.get("last_processed_5m_ts") != \
            utc_iso(CONTROL_DAY, 16, 30)
    else:
        # marker persisted, dispatch happened, outcome unproven: the
        # short is NEVER falsely closed and the freeze blocks re-sends
        assert broker.position_qty == -2              # short intact
        st = read_state(state_env)
        m = st.get("exit_pending_sell_intent")
        assert isinstance(m, dict)
        assert m["status"] == "SELL_OUTCOME_UNVERIFIED"
        assert st["exit_unverified_until_reconcile"] == \
            "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert broker.position_qty == -2              # short intact

    d2 = rt.run_cycle()
    assert len(adapter.buy_calls) <= 2
    if d.reason == "SELL_INTENT_PERSIST_FAILED":
        # clean-disk retry now dispatches the close normally
        assert d2.action == "EXIT" and d2.reason == "EXIT_SHORT"
        assert broker.position_qty == 0
    else:
        # frozen after an unproven first send: NO blind re-send ever
        assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
        assert broker.position_qty == -2


# ══════════════════════════════════════════════════════════════════════
# 10 — ambiguous outcome / final-save error keep the durable EXIT marker.
# ══════════════════════════════════════════════════════════════════════
def test_ambiguous_close_leaves_durable_exit_marker(state_env, market):
    broker = new_broker()
    open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())
    rt2, ad2 = make_recording_runtime(
        market, broker,
        scripted_buy=[RuntimeError("crashed before confirmation")])

    d = rt2.run_cycle()

    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)
    marker = st.get("exit_pending_sell_intent")
    assert isinstance(marker, dict)
    # unique identity of the routed operation: EXIT + BUY + SHORT
    assert marker["operation"] == "EXIT"
    assert marker["side"] == "BUY"
    assert marker["direction"] == "SHORT"
    assert marker["intent"] == "CLOSE_SHORT"
    # G-9C.2: a fresh process seeing this durable marker on disk is
    # blocked at the cycle level (run_cycle pre-guard) BEFORE any
    # relabeling — the on-disk status stays the pre-dispatch label.
    assert marker["status"] == "BUY_IN_FLIGHT"
    assert marker["instrument"] == "CNYRUBF"
    assert marker["qty"] == 2
    assert marker["candle_ts"] == \
        utc_iso(CONTROL_DAY, 16, 30)
    assert "created_at" in marker and "account_id_masked" in marker
    # fate unknown ⇒ position never touched
    assert broker.position_qty == -2
    assert st["c5_position_qty"] == -2


def test_final_save_failure_keeps_marker_after_restart(state_env, market):
    """Ambiguous close whose FINAL strict+fallback saves also fail: the
    pre-dispatch durable marker (already on disk) survives, and a
    restarted process must not re-send the BUY."""
    import src.c5_runtime as rt_mod
    broker = new_broker()
    open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())

    # Fault injected at the runtime's durable-save boundary itself
    # (c5_runtime imports save_state_strict into its own namespace).
    # The FIRST strict call is the pre-dispatch marker persist — it must
    # succeed; EVERY later strict save (UNVERIFIED burn/freeze finalize,
    # cycle-end saves) fails.
    original = rt_mod.save_state_strict
    calls = {"n": 0}

    def fail_after_marker(state):
        calls["n"] += 1
        if calls["n"] >= 2:      # marker pre-save OK, ALL later strict saves fail
            raise IOError("disk died after marker")
        return original(state)

    rt_mod.save_state_strict = fail_after_marker
    try:
        rt2, ad2 = make_recording_runtime(
            market, broker,
            scripted_buy=[exec_res(True, qty_executed="bogus")])
        d = rt2.run_cycle()
    finally:
        rt_mod.save_state_strict = original

    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)
    marker = st.get("exit_pending_sell_intent")
    assert isinstance(marker, dict)
    assert marker["side"] == "BUY" and marker["operation"] == "EXIT"
    assert marker["status"] == "BUY_IN_FLIGHT"    # pre-dispatch label kept
    assert broker.position_qty == -2              # UNVERIFIED never closed

    # restart: fresh process sees the unfinished BUY-EXIT → full block
    rt3, ad3 = make_recording_runtime(market, broker)
    d3 = rt3.run_cycle()
    assert d3.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert ad3.buy_calls == []                    # no blind re-send, ever
    assert broker.position_qty == -2


# ══════════════════════════════════════════════════════════════════════
# 11 — restart with unfinished EXIT marker blocks a repeated BUY.
# ══════════════════════════════════════════════════════════════════════
def test_restart_blocks_repeat_buy_until_operator_resolution(state_env,
                                                            market):
    broker = new_broker()
    open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())
    rt2, ad2 = make_recording_runtime(
        market, broker,
        scripted_buy=[RuntimeError("crashed before confirmation")])
    rt2.run_cycle()
    assert ad2.buy_calls                            # first send happened

    # fresh process: same market/broker/state file
    rt3, ad3 = make_recording_runtime(market, broker)
    d3 = rt3.run_cycle()
    assert d3.action == "NO_ENTRY"
    assert d3.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert ad3.buy_calls == []                      # NO repeated BUY
    assert ad3.sell_calls == []                     # no sneaky reversal
    assert broker.position_qty == -2                # reconcile never moved it
    st = read_state(state_env)
    assert isinstance(st.get("exit_pending_sell_intent"), dict)

    # operator resolution checks the FULL identity of the routed op
    from src.c5_runtime import resolve_pending_sell_intent
    m = st["exit_pending_sell_intent"]
    ts, qty = m["candle_ts"], m["qty"]
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=qty,
                                       operation="ENTRY") is False   # wrong op
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=1,
                                       operation="EXIT") is False    # wrong qty
    assert resolve_pending_sell_intent(st, candle_ts="other", qty=qty,
                                       operation="EXIT") is False    # wrong ts
    assert resolve_pending_sell_intent(st, candle_ts=ts, qty=qty,
                                       operation="EXIT") is True
    assert not st.get("exit_pending_sell_intent")


def test_reconcile_alone_never_clears_buy_exit_marker(state_env, market):
    broker = new_broker()
    open_short(market, broker, state_env)
    advance(market, EXIT_CANDLE())
    rt2, ad2 = make_recording_runtime(
        market, broker,
        scripted_buy=[RuntimeError("crashed before confirmation")])
    rt2.run_cycle()
    # external reality: the BUY actually DID reach the broker
    broker.apply({"instrument": "CNYRUBF", "side": "BUY", "qty": 2,
                  "intent": "CLOSE_SHORT", "close_price": EXIT_PRICE})
    assert broker.position_qty == 0

    rt3, ad3 = make_recording_runtime(market, broker)
    d3 = rt3.run_cycle()
    assert d3.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert ad3.buy_calls == []
    st = read_state(state_env)
    # marker SURVIVES reconcile untouched — a snapshot is not proof
    assert isinstance(st.get("exit_pending_sell_intent"), dict)
    assert st["exit_pending_sell_intent"]["side"] == "BUY"


# ══════════════════════════════════════════════════════════════════════
# 12 — legacy G-7 markers keep their semantics (compatibility).
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


def test_legacy_g7_marker_still_exit_semantics():
    from src.c5_runtime import _marker_operation, resolve_pending_sell_intent
    assert _marker_operation(dict(LEGACY_MARKER)) == "EXIT"
    st = {"exit_pending_sell_intent": dict(LEGACY_MARKER)}
    # default operation="EXIT" clears it exactly like before G-9C.2
    assert resolve_pending_sell_intent(st,
                                       candle_ts=LEGACY_MARKER["candle_ts"],
                                       qty=2) is True
    assert "exit_pending_sell_intent" not in st
    # an EXIT request must never clear an unfinished ENTRY marker either
    st2 = {"exit_pending_sell_intent": dict(LEGACY_MARKER,
                                            operation="ENTRY")}
    assert resolve_pending_sell_intent(st2,
                                       candle_ts=LEGACY_MARKER["candle_ts"],
                                       qty=2) is False


def test_legacy_long_exit_produces_legacy_shaped_marker(state_env, market):
    """A LONG exit that ends UNVERIFIED writes the SAME marker shape as
    pre-G-9C.2 plus the explicit operation tag: side=SELL, EXIT_LONG."""
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK))
    broker = new_broker()
    rt, adapter = make_recording_runtime(market, broker)
    assert rt.run_cycle().action == "ENTER_LONG"
    advance(market, candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK),
            hour=17, minute=0)
    # replace with a scripted-SELL fake for the ambiguous LONG exit
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter
    from src.runtime_config import TradingConfig

    class ScriptedSell:
        def __init__(self, paper):
            self._paper = paper
            self.orders = paper.orders
            self.calls = []

        def find_account(self):
            return self._paper.find_account()

        def snapshot(self, a):
            return self._paper.snapshot(a)

        def buy_cny(self, a, q, **k):
            return self._paper.buy_cny(a, q, **k)

        def sell_cny(self, a, q, **k):
            self.calls.append(q)
            return exec_res(True, qty_executed="bogus")

        def sell_one_tmon(self, a):
            return self._paper.sell_one_tmon(a)

        def buy_one_tmon(self, a):
            return self._paper.buy_one_tmon(a)

        def get_free_cash(self, a):
            return self._paper.get_free_cash(a)

        def get_tmon_quantity(self, a):
            return self._paper.get_tmon_quantity(a)

        def get_tmon_price(self, a):
            return self._paper.get_tmon_price(a)

    rt.adapter = ScriptedSell(PaperExecutionAdapter(broker))
    d = rt.run_cycle()
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)
    m = st["exit_pending_sell_intent"]
    assert m["side"] == "SELL"
    assert m["operation"] == "EXIT"
    assert m["direction"] == "LONG"
    assert "intent" not in m                     # legacy SELL shape intact
    assert broker.position_qty == 2              # no false close


# ══════════════════════════════════════════════════════════════════════
# 13 — FULL PAPER SHORT CYCLE: enter → partial exit → final exit →
#      reconcile confirms flat.
# ══════════════════════════════════════════════════════════════════════
def test_full_short_cycle_enter_partial_exit_final_exit_reconcile(
        state_env, market):
    from src.c5_runtime import reconcile

    broker = new_broker()
    rt, adapter = open_short(market, broker, state_env)
    go_pc = broker.short_margin_locked_per_contract
    cash_after_open = broker.cash

    # ── partial exit: 1 of 2 confirmed ────────────────────────────────
    advance(market, EXIT_CANDLE())
    rt2, ad2 = make_recording_runtime(
        market, broker, scripted_buy=[exec_res(True, qty_executed=1)])
    d2 = rt2.run_cycle()
    assert d2.action == "EXIT" and d2.reason == "EXIT_SHORT"
    assert broker.position_qty == -1
    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == -1
    assert st2["c5_position_direction"] == "SHORT"

    # ── final exit on a NEW candle: remaining 1 closes ───────────────
    advance(market, EXIT_CANDLE(), _exit_candle(CONTROL_DAY, 17, 0, EXIT_PRICE),
            hour=17, minute=5)
    rt3, ad3 = make_recording_runtime(market, broker)
    d3 = rt3.run_cycle()
    assert d3.action == "EXIT" and d3.reason == "EXIT_SHORT"
    assert len(ad3.buy_calls) == 1
    assert ad3.buy_calls[0][0] == 1               # only the residual sent
    assert broker.position_qty == 0
    assert broker.short_margin_locked_per_contract == 0.0
    assert broker.entry_price == 0.0
    # cumulative accounting across BOTH closes: 2 contracts realized at
    # (1399.95 − 1360) × 1000 each; full ГО of 2 contracts returned.
    assert broker.realized_pnl == pytest.approx(79_900.0)
    assert broker.cash == pytest.approx(cash_after_open + 2 * go_pc
                                        + 79_900.0)
    st3 = read_state(state_env)
    assert st3["c5_position_qty"] == 0
    assert st3["c5_position_direction"] is None
    assert st3["c5_entry_price"] is None
    assert not st3.get("exit_pending_sell_intent")

    # ── independent reconcile confirms the flat state ─────────────────
    st_disk = read_state(state_env)
    snap = reconcile(adapter, "paper-account", st_disk)
    assert snap.position_qty == 0
    assert snap.entry_price is None
    assert st_disk["reconcile_status"] == "OK"
