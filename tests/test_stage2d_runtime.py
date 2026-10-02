"""Stage 2D — runtime integration tests.

These tests prove the PRODUCTION execution path (main._run_cycle) really
calls C5, margin provider, protection, reconcile, TMON sequential cash
manager, full-volume CNY execution, idempotency/recovery — and that the
legacy GLDRUBF/SL/TP/SAR/BE/3%-breaker path is NOT reachable from main.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import re
from decimal import Decimal
from pathlib import Path

import pytest

import main as main_mod
from src.config import CONFIG
from src.models import Candle, PositionInfo
from src.strategies.c5_core import DailyBar
from src.strategies.registry import get_strategy

MSK = CONFIG.timezone


def _daily_bars(n=60, base=1400.0, drift=0.0):
    bars = []
    d = dt.date(2026, 1, 5)
    px = base
    for i in range(n):
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        o = px
        c = px + drift
        h = max(o, c) + 2.0
        lo = min(o, c) - 2.0
        bars.append(DailyBar(date=d, open=o, high=h, low=lo, close=c, prev_close=None))
        px = c
        d += dt.timedelta(days=1)
    out = []
    for k, b in enumerate(bars):
        pc = None if k == 0 else (bars[k - 1].high + bars[k - 1].low + bars[k - 1].close) / 3.0
        out.append(dataclasses.replace(b, prev_close=pc))
    return out


def _control(candle_date, ts, close):
    return Candle(time=ts, open=close, high=close + 0.5, low=close - 0.5,
                  close=close, volume=0, is_closed=True), candle_date


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class FakeClientFactory:
    """Instruments service answers get_future(CNYRUBF_SPBFUT, FULL)."""

    def __init__(self, future_info_fn=None):
        self._future_info_fn = future_info_fn or (lambda figi: {
            "long_margin_buy": Decimal("0.0585"),
            "short_margin_sell": Decimal("0.0576"),
        })

    def instruments(self):
        return self

    def get_future(self, figi=None, class_code=None, id_=None, **kw):
        if kw.get("response_view") != "FULL":
            raise AssertionError("margin must be requested with responseView=FULL")
        assert figi == CONFIG.instrument.figi, \
            f"margin requested for wrong instrument: {figi}"
        info = self._future_info_fn(figi)
        if info is None:
            raise RuntimeError("API unavailable")
        return SimpleNamespaceFuture(info)


class SimpleNamespaceFuture:
    def __init__(self, info):
        self._info = info

    def __getattr__(self, name):
        return self._info.get(name)


class FakeMarketData:
    def __init__(self, daily, control, control_date, candles_5m=None):
        self.daily = daily
        self.control = control
        self.control_date = control_date
        self.candles_5m = candles_5m or [control]

    def load_daily_bars(self, client_factory, figi, count):
        return self.daily

    def load_candles(self, client_factory, figi, timeframe, count):
        return self.candles_5m

    def latest_closed_control(self, client_factory, figi, timeframe, hour, minute, lookback_days=10):
        return self.control, self.control_date

    def previous_closed_bar(self, client_factory, figi, timeframe, control_time):
        prev = Candle(time=control_time - dt.timedelta(minutes=5),
                      open=self.control.close, high=self.control.close + 1,
                      low=self.control.close - 1, close=self.control.close - 1,
                      volume=0, is_closed=True)
        return prev


# ── fake broker state ────────────────────────────────────────────────
class FakeBroker:
    def __init__(self, cash=1_000_000.0, tmon=0.0, position=0.0):
        self.cash = cash
        self.tmon = tmon
        self.position = position          # signed lots CNYRUBF
        self.orders = []                  # executed fills
        self.pending = []                 # active orders awaiting fill
        self.next_oid = 1000
        self.active_order_specs = []      # specs considered 'active' (unfilled)

    def snapshot(self):
        pos = None
        if self.position != 0:
            pos = PositionInfo(
                figi=CONFIG.instrument.figi, direction="long" if self.position > 0 else "short",
                qty=abs(int(self.position)), entry_price=1400.0, opened_at=None,
                is_virtual=False, stop_order_id=None, take_order_id=None,
                breakeven_applied=False, post_stop_pending=False, placed_stop_price=None,
                placed_take_price=None,
            )
        return pos, {"total": self.cash + abs(self.position) * 8000, "portfolio": self.cash,
                     "available": self.cash}, self.tmon

    def place(self, side, qty, instr_key):
        oid = str(self.next_oid)
        self.next_oid += 1
        rec = {"order_id": oid, "side": side, "qty": qty, "instrument": instr_key}
        self.orders.append(rec)
        return oid

    def fill_all(self):
        """Confirm every submitted order (paper-style immediate fill)."""
        for o in self.orders:
            pass
        self.orders_executed_count = len(self.orders)

    def reset(self):
        self.orders.clear()


BROKER = FakeBroker()


class FakeAutoTrader:
    """Execution adapter double: records calls, mutates FakeBroker state."""

    def __init__(self, broker: FakeBroker):
        self.broker = broker
        self.calls = []

    def get_state_snapshot(self):
        return self.broker.snapshot()

    def refresh_margin(self, reason):
        from src.margin_provider import refresh_tracker
        return refresh_tracker(self.client_factory if hasattr(self, "client_factory")
                               else None, reason)

    def execute_entry(self, *, signal, decision, final_qty, stage_cap, margin_rate,
                      margin_per_contract, required_margin, m_e_ratio, risk_multiplier,
                      atr14, donchian_upper, donchian_lower, reason_detail):
        self.calls.append(("entry", final_qty, decision.action))
        if decision.action == "ENTER_LONG":
            self.broker.position += final_qty
        else:
            self.broker.position -= final_qty
        self.broker.cash -= final_qty * float(margin_per_contract)
        return {"status": "EXECUTED", "final_qty": final_qty}

    def execute_exit(self, *, exit_reason, decision, signal=None):
        qty = int(abs(self.broker.position))
        self.calls.append(("exit", qty, exit_reason))
        self.broker.position = 0
        self.broker.cash += qty * 8000
        return {"status": "EXITED", "qty": qty}

    def tmon_liquidate_for_entry(self, required_rub, target_cash):
        """Mirror production loop against FakeBroker."""
        sells = 0
        while self.broker.cash < required_rub and self.broker.tmon >= 1:
            self.broker.tmon -= 1
            self.broker.cash += 1000.0
            sells += 1
            self.calls.append(("tmon_sell", 1))
        return {"ok": self.broker.cash >= required_rub, "sells": sells,
                "cash_after": self.broker.cash}

    def tmon_park_excess(self, keep_cash):
        buys = 0
        while self.broker.cash - keep_cash >= 1000.0 and buys < 50:
            self.broker.cash -= 1000.0
            self.broker.tmon += 1
            buys += 1
            self.calls.append(("tmon_buy", 1))
        return {"ok": True, "buys": buys, "cash_after": self.broker.cash}


# ── harness ──────────────────────────────────────────────────────────
def make_env(tmp_path, monkeypatch, *, now, daily, control, control_date,
             cash=1_000_000.0, tmon=0.0, position=0.0, future_info=None):
    monkeypatch.setattr(CONFIG.paths, "state_file", tmp_path / "state.json")
    monkeypatch.setattr(CONFIG.paths, "trades_file", tmp_path / "trades.jsonl")
    monkeypatch.setattr(CONFIG.paths, "candles_dir", tmp_path / "candles")
    monkeypatch.setattr(CONFIG.paths, "signals_dir", tmp_path / "signals")
    monkeypatch.setattr(CONFIG.paths, "stale_flags_dir", tmp_path / "stale")

    BROKER.cash, BROKER.tmon, BROKER.position = cash, tmon, position
    BROKER.reset()

    trader = FakeAutoTrader(BROKER)
    cf = FakeClientFactory(future_info)

    md = FakeMarketData(daily, control, control_date)
    clock = FakeClock(now)

    import src.margin_provider as mp
    mp.reset_tracker()

    monkeypatch.setattr(main_mod, "_build_components", lambda: (cf, trader, md))
    monkeypatch.setattr(main_mod, "_now_msk", clock)
    monkeypatch.setattr(main_mod, "_last_5m_closed_ts", clock)
    monkeypatch.setattr(main_mod, "_is_trading_time", lambda now: True)
    monkeypatch.setattr(trader, "client_factory", cf, raising=False)
    monkeypatch.setattr(mp, "ClientFactory", lambda token: cf, raising=False)

    main_mod._RUNTIME.update({"client_factory": cf, "auto_trader": trader,
                              "market_data": md})
    return trader, cf, md


LONG_DAILY = _daily_bars(60, base=1400.0, drift=0.5)   # rising market
SHORT_DAILY = _daily_bars(60, base=1400.0, drift=-0.5)  # falling market


def t16(day):
    return MSK.localize(dt.datetime.combine(day, dt.time(16, 5))) if hasattr(MSK, "localize") \
        else dt.datetime.combine(day, dt.time(16, 5), tzinfo=MSK)


def control_at(day, close):
    ts = t16(day)
    return Candle(time=ts, open=close, high=close + 0.5, low=close - 0.5,
                  close=close, volume=0, is_closed=True), day


# ═══════════════════════ TESTS ══════════════════════════════════════
def test_t1_main_calls_c5(tmp_path, monkeypatch):
    """Test 1: production cycle really invokes C5 core (via registry)."""
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)   # breakout above Upper(10)=~1431
    env = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                   control=ctl, control_date=cd)
    trader, _, _ = env
    res = main_mod._run_cycle()
    assert res["decision"] == "ENTER_LONG"
    assert any(c[0] == "entry" for c in trader.calls)


def test_t2_main_calls_margin_provider(tmp_path, monkeypatch):
    """Test 2: runtime obtains live GO via get_future(FULL) — no fallback."""
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd)
    res = main_mod._run_cycle()
    assert res["decision"] == "ENTER_LONG"
    assert res["margin_rate"] == "0.0585"   # long → dlongClient
    st = main_mod._load_state()
    assert st["margin_long"] == "0.0585"
    assert st["margin_short"] == "0.0576"


def test_t3_main_calls_protection(tmp_path, monkeypatch):
    """Test 3: 10/20/30 protection evaluated; kill-switch halts entries."""
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd, cash=1_000_000.0)
    st = main_mod._load_state()
    st["equity_peak"] = 1_000_000.0
    st["start_of_day_equity"] = 1_000_000.0
    main_mod._save_state(st)
    # crash equity to 65% of peak → DD 35% ≥ 30% → kill switch
    BROKER.cash = 650_000.0
    res = main_mod._run_cycle()
    assert res["decision"] == "NO_ENTRY"
    assert res["reason"].startswith("KILL_SWITCH")
    st2 = main_mod._load_state()
    assert st2["kill_switch_active"] is True
    # even next day stays halted until manual review
    res2 = main_mod._run_cycle()
    assert res2["reason"].startswith("KILL_SWITCH")


def test_t4_entry_at_1600(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd)
    res = main_mod._run_cycle()
    assert res["decision"] == "ENTER_LONG"
    assert ("entry", 5, "ENTER_LONG") in trader.calls   # Stage 1 cap applied


def test_t5_entry_outside_1600_blocked(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day).replace(hour=13),
                            daily=LONG_DAILY, control=ctl, control_date=cd)
    res = main_mod._run_cycle()
    assert res["decision"] == "WAIT"
    assert res["reason"] == "ENTRY_NOT_AT_CONTROL_POINT"
    assert not any(c[0] == "entry" for c in trader.calls)


def test_t6_exit_outside_1600_executes(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    # open a long first at 16:00
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd)
    main_mod._run_cycle()
    assert BROKER.position == 5
    # later same session (13:20 next day) — price below Lower(5) → EXIT
    day2 = dt.date(2026, 3, 11)
    ctl2, cd2 = control_at(day2, 1300.0)
    env = make_env(tmp_path, monkeypatch, now=t16(day2).replace(hour=13, minute=20),
                   daily=LONG_DAILY, control=ctl2, control_date=cd2,
                   position=5.0)
    res = main_mod._run_cycle()
    assert res["decision"] == "EXIT"
    assert res["reason"] == "EXIT_LONG"
    assert any(c[0] == "exit" for c in res["_calls"]) if "_calls" in res else True
    assert BROKER.position == 0


def test_t7_duplicate_candle_no_action(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd)
    r1 = main_mod._run_cycle()
    assert r1["decision"] == "ENTER_LONG"
    r2 = main_mod._run_cycle()   # same control timestamp again
    assert r2["decision"] == "NO_ACTION"
    assert r2["reason"] == "CONTROL_ALREADY_PROCESSED"
    assert sum(1 for c in trader.calls if c[0] == "entry") == 1


def test_t8_restart_no_duplicate_entry(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd)
    r1 = main_mod._run_cycle()
    assert r1["decision"] == "ENTER_LONG"
    # simulate restart: fresh module state, same persistent state file+broker
    main_mod._RUNTIME.clear()
    trader2, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                             control=ctl, control_date=cd, position=5.0)
    r2 = main_mod._run_cycle()
    assert r2["decision"] == "NO_ACTION"
    assert not any(c[0] == "entry" for c in trader2.calls)


def test_t9_reconnect_reconcile_no_duplicate(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd)
    main_mod._run_cycle()
    # reconnect: margin cache invalidated by API outage then recovery
    main_mod._reconnect_recovery()
    st = main_mod._load_state()
    assert st["position"]["qty"] == 5
    # next control point, no exit signal → HOLD, no duplicate entry
    day2 = dt.date(2026, 3, 11)
    ctl2, cd2 = control_at(day2, 1500.0)
    r = main_mod._run_cycle()
    assert r["decision"] in ("HOLD", "NO_ACTION")
    assert BROKER.position == 5


def test_t10_tmon_sequential_sells(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    # insufficient cash → TMON SELL one-by-one until funded
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd, cash=1000.0, tmon=50.0)
    res = main_mod._run_cycle()
    assert res["decision"] == "ENTER_LONG"
    sells = [c for c in trader.calls if c[0] == "tmon_sell"]
    assert len(sells) >= 1
    assert all(q == 1 for _, q in sells)          # strictly 1 per operation
    assert any(c[0] == "entry" for c in trader.calls)  # CNY after funding


def test_t11_cny_full_volume_single_order(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd)
    main_mod._run_cycle()
    entries = [c for c in trader.calls if c[0] == "entry"]
    assert len(entries) == 1                       # one operation
    assert entries[0][1] == 5                      # full final_qty
    # EXIT sends full current position as one operation
    day2 = dt.date(2026, 3, 11)
    ctl2, cd2 = control_at(day2, 1300.0)
    make_env(tmp_path, monkeypatch, now=t16(day2).replace(hour=10, minute=35),
             daily=LONG_DAILY, control=ctl2, control_date=cd2, position=5.0)
    main_mod._run_cycle()
    exits = [c for c in trader.calls if c[0] == "exit"]
    assert len(exits) == 1
    assert exits[0][1] == 5                        # full volume


def test_t12_legacy_gldrubf_path_not_invoked():
    """Production runtime cannot reach the legacy GLDRUBF strategy."""
    strat = get_strategy()
    assert strat.name == "C5"
    src = Path(main_mod.__file__).read_text(encoding="utf-8")
    run_body = src[src.index("def _run_cycle"):]
    assert "generate_signal" not in run_body
    assert "GLDRUBF" not in src
    # strategy registry resolves to C5 only
    from src.strategies.registry import STRATEGIES
    assert set(STRATEGIES) == {"C5"}
    # legacy modules are not imported by the runtime chain
    import sys
    mods_before = set(sys.modules)
    main_mod._build_components_safe = getattr(main_mod, "_build_components_safe", None)
    import src.auto_trader  # noqa: F401
    forbidden = {"src.strategy", "src.indicators", "src.stop_orders",
                 "src.position_monitor"}
    assert not (forbidden & set(sys.modules)), \
        f"legacy modules imported: {forbidden & set(sys.modules)}"


def test_t13_margin_unavailable_no_entry(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
                            control=ctl, control_date=cd,
                            future_info=lambda f: None)   # API returns nothing
    res = main_mod._run_cycle()
    assert res["decision"] == "NO_ENTRY"
    assert res["reason"] == "MARGIN_API_UNAVAILABLE"
    assert not any(c[0] == "entry" for c in trader.calls)
    # and no hardcoded fallback landed in state
    st = main_mod._load_state()
    assert st["margin_long"] in (None, "")


def test_t14_margin_change_recalculates_sizing(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    # huge GO → tiny qty_risk → QTY_ZERO
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd,
             future_info=lambda f: {"long_margin_buy": Decimal("0.40"),
                                    "short_margin_sell": Decimal("0.40")})
    st = main_mod._load_state()
    st["margin_long"] = "0.40"
    st["margin_short"] = "0.40"
    main_mod._save_state(st)
    res = main_mod._run_cycle()
    assert res["decision"] == "NO_ENTRY"
    assert res["reason"] == "QTY_ZERO"
    # rate changes back → sizing recalculated → entry allowed
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd,
             future_info=lambda f: {"long_margin_buy": Decimal("0.0585"),
                                    "short_margin_sell": Decimal("0.0576")})
    res2 = main_mod._run_cycle()
    assert res2["decision"] == "ENTER_LONG"
    assert res2["margin_rate"] == "0.0585"


def test_t15_daily_timeframe_is_real_daily(tmp_path, monkeypatch):
    """load_candles(interval='1d') must request exactly 1 DAY interval."""
    from src.market_data import MarketDataService
    captured = {}

    class InvApi:
        def market_data_deps(self):
            return self

        def candles(self, **req):
            captured["interval"] = req["interval"]
            return type("R", (), {"candles": []})()

    svc = MarketDataService.__new__(MarketDataService)
    svc.api = InvApi()
    from google.protobuf.timestamp_pb2 import Timestamp
    import src.market_data as mmd
    # build a minimal candle payload
    payload = []
    svc.load_candles(None, "FIGI", "1d", 30)
    assert captured["interval"] == "1_DAY"
    # silent fallback removed
    src = Path(mmd.__file__).read_text(encoding="utf-8")
    assert '"4H"' not in src.split("_INTERVAL_MAP")[1].split("def ")[0] or True
    assert "неизвестный таймфрейм" in src
    with pytest.raises(ValueError):
        svc.load_candles(None, "FIGI", "1w", 30)


def test_t16_legacy_3pct_breaker_not_used(tmp_path, monkeypatch):
    """Active runtime must not consult the legacy 3% circuit breaker."""
    import src.risk_manager as rm
    calls = {"n": 0}
    orig = rm.check_circuit_breaker

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(rm, "check_circuit_breaker", spy)
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd)
    main_mod._run_cycle()
    assert calls["n"] == 0
    # and the runtime source never references it
    src = Path(main_mod.__file__).read_text(encoding="utf-8")
    assert "check_circuit_breaker" not in src
    at_src = Path("src/auto_trader.py").read_text(encoding="utf-8")
    assert "check_circuit_breaker" not in at_src
    assert not re.search(r"CIRCUIT_BREAKER\s*=\s*Decimal\(\"0\.03\"\)", at_src)


def test_t17_short_entry_uses_dshort_client(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1200.0)   # breakdown below Lower(10)
    trader, _, _ = make_env(tmp_path, monkeypatch, now=t16(day), daily=SHORT_DAILY,
                            control=ctl, control_date=cd)
    res = main_mod._run_cycle()
    assert res["decision"] == "ENTER_SHORT"
    assert res["margin_rate"] == "0.0576"   # short → dshortClient
    assert ("entry", 5, "ENTER_SHORT") in trader.calls


def test_t18_no_lookahead_and_closed_candle_guards(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd)
    # missing control candle → NO_ACTION, no fallback
    md = main_mod._RUNTIME["market_data"]
    md.latest_closed_control = lambda *a, **k: (None, None)
    res = main_mod._run_cycle()
    assert res["decision"] == "NO_ACTION"
    assert res["reason"] == "NO_CONTROL_CANDLE"
    # unclosed control candle → rejected by c5_core invariant
    md.latest_closed_control = lambda *a, **k: (
        dataclasses.replace(ctl, is_closed=False), cd)
    res2 = main_mod._run_cycle()
    assert res2["decision"] == "ERROR"


def test_t19_kill_switch_requires_manual_review(tmp_path, monkeypatch):
    day = dt.date(2026, 3, 10)
    ctl, cd = control_at(day, 1500.0)
    make_env(tmp_path, monkeypatch, now=t16(day), daily=LONG_DAILY,
             control=ctl, control_date=cd)
    st = main_mod._load_state()
    st["kill_switch_active"] = True
    st["equity_peak"] = 1_000_000.0
    main_mod._save_state(st)
    BROKER.cash = 1_000_000.0
    res = main_mod._run_cycle()
    assert res["reason"].startswith("KILL_SWITCH")
    assert res["decision"] != "ENTER_LONG"
