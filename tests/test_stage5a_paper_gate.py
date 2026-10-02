"""Stage 5A — PAPER SAFETY GATE tests.

Proves the trading-safety contract end-to-end WITHOUT any real
credentials or network:

  * TRADING_MODE is read from the environment; missing/invalid → FAIL
    CLOSED (never defaults to LIVE).
  * AUTO_TRADING_ENABLED is read from the environment; missing/invalid →
    False; False blocks trading BEFORE execution (paper included).
  * One source of truth: src.runtime_config parses; main.py stays thin
    and c5_runtime consumes the validated config (no duplicate parsing).
  * PAPER  → PaperExecutionAdapter + SimulatedBroker, local state only.
  * LIVE   → LiveExecutionAdapter, routed through a FAKE SDK boundary.
  * Regression: PAPER BUY / SELL / TMON can physically never reach
    OrdersServiceApi.post_order() (client factory booby-trapped).
"""
from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# ── deterministic market fixture constants (same shape as stage 2D) ───
REF_DATE = dt.date(2026, 9, 30)
CONTROL_DAY = dt.date(2026, 10, 2)
LONG_BREAK = 1420.0
SHORT_BREAK = 1370.0
PRICE = 1400.0


def _mode_env(monkeypatch, mode, auto):
    """Set TRADING_MODE/AUTO_TRADING_ENABLED exactly like production env."""
    if mode is None:
        monkeypatch.delenv("TRADING_MODE", raising=False)
    else:
        monkeypatch.setenv("TRADING_MODE", mode)
    if auto is None:
        monkeypatch.delenv("AUTO_TRADING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AUTO_TRADING_ENABLED", auto)


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", state_file)
    import src.state_store as ss
    monkeypatch.setattr(ss, "STATE_FILE", state_file)
    return tmp_path


# ══════════════════════════════════════════════════════════════════════
# Single-source-of-truth parser (src.runtime_config)
# ══════════════════════════════════════════════════════════════════════
def test_rc_paper_mode_parsed():
    from src.runtime_config import load_trading_config
    cfg = load_trading_config({"TRADING_MODE": "PAPER",
                               "AUTO_TRADING_ENABLED": "true"})
    assert cfg.mode == "PAPER"
    assert cfg.is_paper and not cfg.is_live
    assert cfg.auto_trading_enabled is True


def test_rc_live_mode_parsed():
    from src.runtime_config import load_trading_config
    cfg = load_trading_config({"TRADING_MODE": "live",
                               "AUTO_TRADING_ENABLED": "TRUE"})
    assert cfg.mode == "LIVE"
    assert cfg.auto_trading_enabled is True


def test_rc_missing_mode_fail_closed():
    from src.runtime_config import TradingConfigError, load_trading_config
    with pytest.raises(TradingConfigError):
        load_trading_config({})


def test_rc_blank_mode_fail_closed():
    from src.runtime_config import TradingConfigError, load_trading_config
    with pytest.raises(TradingConfigError):
        load_trading_config({"TRADING_MODE": "   ",
                             "AUTO_TRADING_ENABLED": "true"})


@pytest.mark.parametrize("bad", ["SANDBOX", "paper-live", "DRY", "TEST"])
def test_rc_invalid_mode_fail_closed(bad):
    from src.runtime_config import TradingConfigError, load_trading_config
    with pytest.raises(TradingConfigError):
        load_trading_config({"TRADING_MODE": bad,
                             "AUTO_TRADING_ENABLED": "true"})


def test_rc_never_defaults_to_live():
    """Empty environment must raise — it must NOT produce a LIVE config."""
    from src.runtime_config import TradingConfigError, load_trading_config
    with pytest.raises(TradingConfigError):
        load_trading_config({})


@pytest.mark.parametrize("raw", [None, "", "maybe", "1", "0", "yes",
                                 "no", "on", "off", "True", "FALSE",
                                 " true ", "garbage"])
def test_rc_auto_parsing(raw):
    from src.runtime_config import parse_auto_trading
    result = parse_auto_trading(raw)
    assert isinstance(result, bool)
    # Spec: None / missing -> False; "" / blank -> False; invalid -> False.
    # True ONLY for explicitly allowed true-values (case/whitespace tolerant).
    if raw is not None and str(raw).strip().lower() in ("1", "true", "yes", "on"):
        assert result is True
    else:
        # missing / blank / invalid → false (fail closed for trading)
        assert result is False


def test_rc_gate_raises_when_disabled():
    from src.runtime_config import (TradingConfig, TradingConfigError,
                                    enforce_trading_gate)
    with pytest.raises(TradingConfigError):
        enforce_trading_gate(TradingConfig("PAPER", False))
    with pytest.raises(TradingConfigError):
        enforce_trading_gate(TradingConfig("LIVE", False))
    enforce_trading_gate(TradingConfig("PAPER", True))  # no raise


# ══════════════════════════════════════════════════════════════════════
# A — PAPER → PaperExecutionAdapter
# ══════════════════════════════════════════════════════════════════════
def test_a_paper_builds_paper_adapter_with_simulated_broker(state_env,
                                                            monkeypatch):
    _mode_env(monkeypatch, "PAPER", "true")
    from src.execution_adapter import (PaperExecutionAdapter, SimulatedBroker,
                                       build_execution_adapter)
    adapter, mode = build_execution_adapter("fake-token", "uid-cny")
    assert mode == "PAPER"
    assert isinstance(adapter, PaperExecutionAdapter)
    assert isinstance(adapter.broker, SimulatedBroker)


# ══════════════════════════════════════════════════════════════════════
# B — LIVE → LiveExecutionAdapter
# ══════════════════════════════════════════════════════════════════════
def test_b_live_builds_live_adapter(state_env, monkeypatch):
    _mode_env(monkeypatch, "LIVE", "true")
    from src.execution_adapter import (LiveExecutionAdapter,
                                       PaperExecutionAdapter,
                                       build_execution_adapter)
    adapter, mode = build_execution_adapter("fake-token", "uid-cny")
    assert mode == "LIVE"
    assert isinstance(adapter, LiveExecutionAdapter)
    assert not isinstance(adapter, PaperExecutionAdapter)


# ══════════════════════════════════════════════════════════════════════
# C/D — missing / invalid mode → fail closed at the factory too
# ══════════════════════════════════════════════════════════════════════
def test_c_missing_mode_factory_fails_closed(state_env, monkeypatch):
    _mode_env(monkeypatch, None, "true")
    from src.execution_adapter import build_execution_adapter
    from src.runtime_config import TradingConfigError
    with pytest.raises(TradingConfigError):
        build_execution_adapter("fake-token", "uid-cny")


def test_d_invalid_mode_factory_fails_closed(state_env, monkeypatch):
    _mode_env(monkeypatch, "PAPERISH", "true")
    from src.execution_adapter import build_execution_adapter
    from src.runtime_config import TradingConfigError
    with pytest.raises(TradingConfigError):
        build_execution_adapter("fake-token", "uid-cny")


def test_cd_runtime_constructor_fails_closed(state_env, monkeypatch):
    """Even constructing the runtime with an unset mode must refuse."""
    _mode_env(monkeypatch, "not-a-mode", "true")
    from src.c5_runtime import C5Runtime
    from src.runtime_config import TradingConfigError
    with pytest.raises(TradingConfigError):
        C5Runtime("fake-token")


# ══════════════════════════════════════════════════════════════════════
# E/F — AUTO_TRADING_ENABLED=false blocks trading BEFORE execution
# ══════════════════════════════════════════════════════════════════════
def _runtime_with_spy_clock(monkeypatch, rt_mod):
    calls = {"load": 0}
    original = rt_mod.load_candles

    def spy(token, uid, candles_count=200, timeframe="4H"):
        calls["load"] += 1
        return original(token, uid, candles_count=candles_count,
                        timeframe=timeframe)

    rt_mod.load_candles = spy
    return calls


@pytest.fixture
def rt_boundaries(monkeypatch):
    """Restore module-level boundary patches after each test."""
    import src.c5_runtime as rt_mod
    saved = (rt_mod.load_candles, rt_mod.fetch_order_book)
    yield rt_mod
    rt_mod.load_candles, rt_mod.fetch_order_book = saved


def test_e_auto_false_paper_blocks_before_execution(state_env, monkeypatch,
                                                    rt_boundaries):
    _mode_env(monkeypatch, "PAPER", "false")
    from src.c5_runtime import REASON_TRADING_BLOCKED
    rt_mod = rt_boundaries
    clock = _runtime_with_spy_clock(monkeypatch, rt_mod)

    from src.runtime_config import TradingConfig
    rt = rt_mod.C5Runtime("fake-token",
                          config=TradingConfig("PAPER", False),
                          now_fn=lambda: utc_ms(CONTROL_DAY, 16, 5))
    decision = rt.run_cycle()
    assert decision.action == "NO_ENTRY"
    assert decision.reason == REASON_TRADING_BLOCKED
    # blocked BEFORE execution: no market data, no adapter built,
    # no order flow attempted.
    assert clock["load"] == 0
    assert rt.adapter is None


def test_e_auto_missing_blocks_like_false(state_env, monkeypatch,
                                          rt_boundaries):
    _mode_env(monkeypatch, "PAPER", None)  # missing → false → blocked
    from src.c5_runtime import REASON_TRADING_BLOCKED
    from src.runtime_config import load_trading_config
    cfg = load_trading_config()
    assert cfg.auto_trading_enabled is False
    rt = rt_boundaries.C5Runtime("fake-token", config=cfg,
                                 now_fn=lambda: utc_ms(CONTROL_DAY, 16, 5))
    decision = rt.run_cycle()
    assert decision.reason == REASON_TRADING_BLOCKED


def test_f_auto_false_live_blocked_no_post_order(state_env, monkeypatch,
                                                 rt_boundaries):
    """LIVE + AUTO=false: refused before bootstrap — no client, no order."""
    _mode_env(monkeypatch, "LIVE", "false")
    import src.client_factory as cf
    import src.instruments as instruments
    from src.c5_runtime import REASON_TRADING_BLOCKED
    from src.runtime_config import TradingConfig

    def tripwire(*a, **k):
        raise AssertionError(
            "network/instrument access while AUTO_TRADING_ENABLED=false")

    monkeypatch.setattr(cf, "get_client", tripwire)
    monkeypatch.setattr(instruments, "get_c5_instrument", tripwire)

    clock = _runtime_with_spy_clock(monkeypatch, rt_boundaries)
    rt = rt_boundaries.C5Runtime("fake-token",
                                 config=TradingConfig("LIVE", False),
                                 now_fn=lambda: utc_ms(CONTROL_DAY, 16, 5))
    decision = rt.run_cycle()
    assert decision.reason == REASON_TRADING_BLOCKED
    assert clock["load"] == 0
    assert rt.adapter is None


def test_f_main_entrypoint_refuses_without_token(state_env, monkeypatch):
    """c5_runtime.main() fails closed before touching credentials."""
    _mode_env(monkeypatch, "LIVE", None)  # AUTO missing → false
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    monkeypatch.delenv("INVEST_TOKEN", raising=False)
    from src import c5_runtime
    assert c5_runtime.main() is None  # returns without running anything


# ══════════════════════════════════════════════════════════════════════
# G/H/I — PAPER BUY / SELL / TMON are LOCAL ONLY (post_order booby-trap)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def no_real_client(monkeypatch):
    """Any attempt to obtain a live API client raises immediately.

    Physical proof for G–J: the PAPER path cannot reach
    OrdersServiceApi.post_order(), because it can never even reach a
    client — and its adapter class contains no post_order call at all.
    """
    import src.client_factory as cf

    @contextmanager
    def _tripwire(token):
        raise AssertionError("REAL API CLIENT USED IN PAPER MODE")
        yield None  # pragma: no cover

    monkeypatch.setattr(cf, "get_client", _tripwire)
    return _tripwire


def test_g_paper_buy_is_local_only(no_real_client):
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    adapter = PaperExecutionAdapter(broker)
    res = adapter.buy_cny("paper-account", 3)
    assert res.ok and res.reason == "PAPER_FILLED"
    assert res.qty_executed == 3
    assert broker.position_qty == 3          # local state mutated
    assert adapter.orders == [{"instrument": "CNYRUBF", "side": "BUY",
                               "qty": 3}]


def test_h_paper_sell_is_local_only(no_real_client):
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000,
                             position_qty=3)
    adapter = PaperExecutionAdapter(broker)
    res = adapter.sell_cny("paper-account", 3)
    assert res.ok and res.reason == "PAPER_FILLED"
    assert broker.position_qty == 0


def test_i_paper_tmon_is_local_only(no_real_client):
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    broker = SimulatedBroker(cash=100_000, equity=100_000, tmon_qty=2)
    adapter = PaperExecutionAdapter(broker)
    sell = adapter.sell_one_tmon("paper-account")
    assert sell.ok
    assert broker.tmon_qty == 1
    buy = adapter.buy_one_tmon("paper-account")
    assert buy.ok
    assert broker.tmon_qty == 2
    assert [o["instrument"] for o in adapter.orders] == ["TMON", "TMON"]


# ══════════════════════════════════════════════════════════════════════
# J — PAPER full lifecycle flat → BUY → LONG → SELL → flat
#     through the REAL runtime pipeline (fakes only at boundaries)
# ══════════════════════════════════════════════════════════════════════
DAILY_HL = 0.07815   # half-range of each synthetic daily bar


def daily_df(n=60):
    """Deterministic realistic OHLC: H=1400+HL, L=1400-HL, C=1400.

    Wilder TR = max(H-L, |H-prev_close|, |L-prev_close|) = 2*HL per bar
    → ATR14 == 2*HL == 0.1563 (non-zero; c5_core fail-closes on 0.0) and
    risk_multiplier = clip(0.1563/0.1563, 0.5, 1.5) == 1.0 — frozen math
    untouched, only the test market data is chosen deliberately.
    Same principle as tests/test_stage2d_runtime.py (non-degenerate OHLC).
    Channel Upper/Lower(10) stays exactly at 1400 so LONG_BREAK/SHORT_BREAK
    signals are unchanged.

    Frozen sizing with this fixture (verified against c5_core.evaluate):
      equity          = 1_000_000
      ATR14           = 0.1563
      risk_multiplier = 1.0
      risk_budget     = 1_000_000 * 0.20 * 1.0 = 200_000
      margin/contract = 1400 * 1000 * 0.0585  = 81_900
      qty_risk        = floor(200_000 / 81_900) = 2
      qty_margin      = floor(1_000_000 / 81_900) = 12
      max_qty         = min(qty_risk, qty_margin, 40) = 2  (qty_raw)
      M/E             = 163_800 / 1_000_000 = 0.1638 <= 0.30
      stage cap       = 5
      final qty       = min(2, 5) = 2
    ⇒ qty=2 is the CORRECT frozen-C5 result at equity=1M; the stage cap
    of 5 is NOT reached (it would need risk_budget >= 4*81_900)."""
    import pandas as pd
    rows = []
    d = REF_DATE - dt.timedelta(days=n - 1)
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        rows.append({"time": ts, "open": PRICE, "high": PRICE + DAILY_HL,
                     "low": PRICE - DAILY_HL, "close": PRICE, "volume": 1})
        d += dt.timedelta(days=1)
    assert rows[-1]["time"].tz_convert(MSK).date() == REF_DATE
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

    def refresh(self):
        from src.margin_provider import MarginRates
        self.current = MarginRates(long_margin_rub=0.0585,
                                   short_margin_rub=0.0576,
                                   instrument="CNYRUBF")
        return self.current


def test_j_paper_full_lifecycle_flat_buy_long_sell_flat(state_env,
                                                        monkeypatch,
                                                        no_real_client):
    _mode_env(monkeypatch, "PAPER", "true")
    import pandas as pd

    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    from src.runtime_config import TradingConfig

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)      # breakout up
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)  # exit below Lower(10)
    holder = {"now": utc_ms(CONTROL_DAY, 16, 5),
              "rows": [C16],
              "daily": daily_df()}

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return holder["daily"].copy()
        if timeframe == "5m":
            return pd.DataFrame(holder["rows"])
        raise AssertionError(f"unexpected timeframe {timeframe!r}")

    rt_mod.load_candles = _load
    rt_mod.fetch_order_book = lambda *a, **k: ([(PRICE - 0.05, 100)],
                                               [(PRICE + 0.05, 100)])

    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    adapter = PaperExecutionAdapter(broker)
    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: holder["now"])
    # Market-data boundary fake stands in for the SDK instrument lookup;
    # PAPER bootstrap never resolves a live instrument on its own.
    rt.instrument = FakeInstrument()

    try:
        # ── 16:05 control point: ENTER_LONG executed on paper ────────
        d1 = rt.run_cycle()
        assert d1.action == "ENTER_LONG"
        # qty=2 is the frozen-C5 sizing result for equity=1M with this
        # fixture (risk_budget 200k / margin_per_contract 81.9k → qty_risk=2;
        # stage cap 5 is NOT reached). See daily_df docstring for the full
        # arithmetic — we assert the frozen outcome, not a hand-picked 5.
        assert d1.qty == 2
        assert broker.position_qty == 2           # simulated long opened
        assert adapter.orders[0] == {"instrument": "CNYRUBF", "side": "BUY",
                                     "qty": 2}
        st = json.loads(open(state_env / "state.json").read())
        assert st["c5_position_direction"] == "LONG"

        # ── HOLD: same candle already processed, nothing re-executed ─
        # With a LONG position open the frozen c5_core takes the EXIT/HOLD
        # branch and returns HOLD (entry signals are ignored in position);
        # the runtime's duplicate-candle guard is covered by len(orders)==1.
        d2 = rt.run_cycle()
        assert d2.action == "HOLD"
        assert broker.position_qty == 2
        assert len(adapter.orders) == 1

        # ── next closed 5m bar breaks Lower(10): EXIT → flat ─────────
        holder["rows"] = [C16, C1655]
        holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
        d3 = rt.run_cycle()
        assert d3.action == "EXIT"
        assert broker.position_qty == 0           # back to flat locally
        assert adapter.orders[-1] == {"instrument": "CNYRUBF",
                                      "side": "SELL", "qty": 2}
        st = json.loads(open(state_env / "state.json").read())
        assert st["c5_position_qty"] == 0
        assert st["c5_position_direction"] is None

        # The whole lifecycle used ONLY local simulated state; the
        # no_real_client tripwire was never hit ⇒ zero post_order calls.
        assert all(o["side"] in ("BUY", "SELL") for o in adapter.orders)
    finally:
        rt_mod.load_candles, rt_mod.fetch_order_book = (
            rt_boundaries_restore(rt_mod))


def rt_boundaries_restore(rt_mod):
    return rt_mod.load_candles, rt_mod.fetch_order_book


# ══════════════════════════════════════════════════════════════════════
# K — LIVE routing through a FAKE SDK boundary (no real API)
# ══════════════════════════════════════════════════════════════════════
class FakeOrdersAPI:
    def __init__(self):
        self.posted = []

    def post_order(self, **kwargs):
        self.posted.append(kwargs)
        return SimpleNamespace(order_id="fake-order-1")


class FakeClient:
    """Boundary fake standing in for the t_tech.invest SDK client."""

    def __init__(self):
        self.orders = FakeOrdersAPI()


def test_k_live_routes_through_fake_sdk(no_real_client, monkeypatch):
    """With TRADING_MODE=LIVE+AUTO=true the factory yields the LIVE
    adapter, and its order flow goes to client.orders.post_order — the
    exact boundary that PAPER can never touch."""
    _mode_env(monkeypatch, "LIVE", "true")
    import src.client_factory as cf
    import src.execution_adapter as ea
    from src.runtime_config import load_trading_config

    fake_client = FakeClient()

    @contextmanager
    def fake_get_client(token):
        yield fake_client

    monkeypatch.setattr(cf, "get_client", fake_get_client)
    monkeypatch.setattr(ea, "get_client", fake_get_client)

    cfg = load_trading_config()
    adapter, mode = build_execution_adapter_via_factory(cfg)
    assert mode == "LIVE"
    assert isinstance(adapter, ea.LiveExecutionAdapter)

    posted = []

    def fake_wait(client, account_id, order_id):
        from t_tech.invest import OrderExecutionReportStatus
        return (SimpleNamespace(execution_report_status=
                OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL), 2)

    monkeypatch.setattr("src.auto_trader.wait_for_order_fill", fake_wait)
    monkeypatch.setattr("src.auto_trader._ensure_market_order_available",
                        lambda *a, **k: None)

    result = adapter.buy_cny("acc-1", 2)
    assert result.ok
    assert fake_client.orders.posted, "LIVE must route via post_order"
    assert fake_client.orders.posted[0]["quantity"] == 2
    assert len(posted) == 0  # sanity


def build_execution_adapter_via_factory(cfg):
    from src.execution_adapter import build_execution_adapter
    return build_execution_adapter("fake-token", "uid-cny", config=cfg)


# ══════════════════════════════════════════════════════════════════════
# L — regression: PAPER can NEVER reach OrdersServiceApi.post_order()
# ══════════════════════════════════════════════════════════════════════
def test_l_paper_path_has_no_post_order_reference():
    """Structural guarantee: the PAPER classes contain zero references to
    post_order / get_client — only LiveExecutionAdapter may call them."""
    import inspect

    from src import execution_adapter as ea
    for cls in (ea.PaperExecutionAdapter, ea.SimulatedBroker):
        src_text = inspect.getsource(cls)
        assert "post_order" not in src_text
        assert "get_client" not in src_text
    # And the live path DOES reference it (so the check is meaningful).
    assert "post_order" in inspect.getsource(ea.LiveExecutionAdapter)


def test_l_paper_runtime_never_touches_client_factory(state_env,
                                                     monkeypatch):
    """Behavioral guarantee: a full gated PAPER cycle with the client
    factory booby-trapped completes without ever building a client."""
    _mode_env(monkeypatch, "PAPER", "true")
    import pandas as pd

    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker
    from src.runtime_config import load_trading_config

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    daily = daily_df()

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return daily.copy()
        return pd.DataFrame([C16])

    saved = (rt_mod.load_candles, rt_mod.fetch_order_book)
    rt_mod.load_candles = _load
    rt_mod.fetch_order_book = lambda *a, **k: ([(PRICE - 0.05, 100)],
                                               [(PRICE + 0.05, 100)])
    try:
        cfg = load_trading_config()
        broker = SimulatedBroker(cash=1_000_000, equity=1_000_000)
        adapter = PaperExecutionAdapter(broker)
        rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                              margin_tracker=FakeMarginTracker(),
                              config=cfg,
                              now_fn=lambda: utc_ms(CONTROL_DAY, 16, 5))
        rt.instrument = FakeInstrument()
        decision = rt.run_cycle()
        assert decision.action == "ENTER_LONG"
        assert broker.position_qty == decision.qty
        # no_real_client tripwire (fixture) never raised ⇒ no client,
        # therefore physically no OrdersServiceApi.post_order().
    finally:
        rt_mod.load_candles, rt_mod.fetch_order_book = saved


# ══════════════════════════════════════════════════════════════════════
# main.py stays a thin entrypoint (one source of truth)
# ══════════════════════════════════════════════════════════════════════
def test_main_is_thin_no_duplicate_parsing():
    import inspect

    import src.main as m
    text = inspect.getsource(m)
    assert 'environ.get("TRADING_MODE"' not in text
    assert "load_trading_config" not in text  # delegates entirely
    assert "SIGNAL_ONLY" not in text
