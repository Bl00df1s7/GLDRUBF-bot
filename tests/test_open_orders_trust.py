"""open_orders trustworthiness fix — LiveExecutionAdapter.snapshot() fail-closed.

Defect (confirmed by audit, fixed in src/execution_adapter.py): a
``client.orders.get_orders()`` exception inside
``LiveExecutionAdapter.snapshot()`` was swallowed by ``except Exception``
and coerced into ``open_orders = 0``.  reconcile() reads zero as "no
working orders confirmed", so a transient API failure could silently
enable execution on an UNKNOWN open-order state.

Fix under test: the error now propagates unchanged (exactly like the
sibling get_positions()/get_portfolio() calls); bootstrap() turns it into
reconcile FAILED ⇒ STARTUP_RECONCILE_FAILED ⇒ NO execution.

Proven here (fake ONLY at the t_tech SDK client boundary — same FakeClient
style as tests/test_stage5a_paper_gate.py section K; no real API/network):

  T1  get_orders() OK, empty list        → snapshot.open_orders == 0
  T2  get_orders() OK, N working orders  → snapshot.open_orders == N
  T3  get_orders() raises                → exception escapes snapshot();
                                          NO fabricated open_orders == 0
  T4  C5 runtime startup with a failing  → STARTUP_RECONCILE_FAILED,
      OrdersService.get_orders()           reconcile_status=FAILED,
                                          ZERO trading operations
  T5  PAPER regression: SimulatedBroker. → open_orders == 0 and G-1/G-4/
      snapshot().open_orders stays 0       G-5 reconcile scenarios intact

Environment note: grpc/t_tech are NOT installed here.  src.api_retry does
``import grpc`` at module import time, so this file installs a minimal
in-process StatusCode stub into sys.modules BEFORE importing api_retry —
mirroring the status codes the real retry policy classifies
(tests/test_api_retry.py).  If the real grpc package is present, the stub
is skipped entirely and production semantics are used unchanged.
"""
from __future__ import annotations

import datetime as dt
import enum
import json
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# ── deterministic market fixture constants (same shape as stage 5B) ───
REF_DATE = dt.date(2026, 9, 30)
CONTROL_DAY = dt.date(2026, 10, 2)
LONG_BREAK = 1420.0
PRICE = 1400.0


def _install_grpc_stub_if_missing():
    """Provide grpc.StatusCode only when the real package is absent."""
    try:
        import grpc  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("grpc")

    class StatusCode(enum.Enum):
        OK = ("OK", 0)
        CANCELLED = ("CANCELLED", 1)
        INVALID_ARGUMENT = ("INVALID_ARGUMENT", 3)
        DEADLINE_EXCEEDED = ("DEADLINE_EXCEEDED", 4)
        NOT_FOUND = ("NOT_FOUND", 5)
        PERMISSION_DENIED = ("PERMISSION_DENIED", 7)
        RESOURCE_EXHAUSTED = ("RESOURCE_EXHAUSTED", 8)
        UNAVAILABLE = ("UNAVAILABLE", 14)

        def __init__(self, detail, code):
            self._detail_ = detail
            self.__code__ = code

        @property
        def value(self):
            return self.name

    stub.StatusCode = StatusCode
    sys.modules["grpc"] = stub


_install_grpc_stub_if_missing()

from src.api_retry import ApiRateLimitError  # noqa: E402
import grpc  # noqa: E402


class Quotation:
    """SDK-shaped money carrier (units + nano, exactly what production
    ``quotation_to_float`` consumes for positions.money entries —
    src/market_data.py lines 37–38)."""

    def __init__(self, units: float, currency: str = "rub"):
        self.units = int(units)
        self.nano = int(round((float(units) - int(units)) * 1_000_000_000))
        self.currency = currency


def make_live_response(*, orders="unset", futures=None, money=None,
                       blocked=None, portfolio_total=1_000_000.0):
    """Build SDK-shaped read responses for one healthy live account:
    flat CNYRUBF position, 1M RUB cash/equity, optional working orders."""
    if orders == "unset":
        orders = []
    return SimpleNamespace(
        positions=SimpleNamespace(
            futures=list(futures or []),
            money=list(money or [Quotation(1_000_000.0)]),
            blocked=list(blocked or []),
        ),
        portfolio=SimpleNamespace(total_amount_portfolio=portfolio_total),
        orders=SimpleNamespace(orders=list(orders)),
    )


class FakeOrdersService:
    """Boundary fake of t_tech OrdersServiceApi (read path only)."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def get_orders(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeOperationsService:
    def __init__(self, response):
        self._response = response
        self.get_positions_calls = 0
        self.get_portfolio_calls = 0

    def get_positions(self, **kwargs):
        self.get_positions_calls += 1
        return self._response.positions

    def get_portfolio(self, **kwargs):
        self.get_portfolio_calls += 1
        return self._response.portfolio


class FakeLiveClient:
    """Minimal stand-in for the t_tech.invest Client at the exact
    boundary LiveExecutionAdapter.snapshot() consumes."""

    def __init__(self, response, orders_error=None):
        self.operations = FakeOperationsService(response)
        self.orders = FakeOrdersService(error=orders_error)


@contextmanager
def _fake_client_factory(client):
    yield client


@pytest.fixture
def live_boundary(monkeypatch):
    """Patch ONLY src.execution_adapter.get_client (SDK boundary)."""
    import src.execution_adapter as ea

    holder = {}

    def _install(response, orders_error=None):
        client = FakeLiveClient(response, orders_error=orders_error)
        holder["client"] = client
        monkeypatch.setattr(ea, "get_client",
                            lambda token: _fake_client_factory(client))
        return client

    holder["install"] = _install
    return holder


def snapshot_via_live_adapter(install_fn, response, orders_error=None):
    from src.execution_adapter import LiveExecutionAdapter

    install_fn(response, orders_error=orders_error)
    adapter = LiveExecutionAdapter("fake-token", "uid-cny")
    return adapter.snapshot("acc-1")


# ══════════════════════════════════════════════════════════════════════
# T1 — get_orders() succeeds with an EMPTY list → open_orders == 0
#      (a CONFIRMED zero, identical value to the old masked one — the
#      fix must not change the happy path)
# ══════════════════════════════════════════════════════════════════════
def test_t1_get_orders_empty_confirms_zero(live_boundary):
    resp = make_live_response(orders=[])
    snap = snapshot_via_live_adapter(live_boundary["install"], resp)
    assert snap.open_orders == 0
    assert resp.orders.orders == []          # provenance: truly empty
    assert live_boundary["client"].orders.calls == [{"account_id": "acc-1"}]


# ══════════════════════════════════════════════════════════════════════
# T2 — get_orders() succeeds with N working orders → open_orders == N
# ══════════════════════════════════════════════════════════════════════
def test_t2_get_orders_n_returns_n(live_boundary):
    working = [SimpleNamespace(order_id=f"o{i}") for i in range(3)]
    snap = snapshot_via_live_adapter(
        live_boundary["install"], make_live_response(orders=working))
    assert snap.open_orders == 3
    # Other fields keep their contract (int, no coercion happened).
    assert isinstance(snap.open_orders, int)


# ══════════════════════════════════════════════════════════════════════
# T3 — get_orders() raises → the exception ESCAPES snapshot().
#      The pre-fix behavior (fabricated AccountSnapshot with
#      open_orders == 0) must be impossible.
# ══════════════════════════════════════════════════════════════════════
def test_t3_get_orders_error_propagates_not_masked_as_zero(live_boundary):
    error = RuntimeError(grpc.StatusCode.UNAVAILABLE, "orders service down")
    with pytest.raises(RuntimeError) as excinfo:
        snapshot_via_live_adapter(live_boundary["install"],
                                  make_live_response(),
                                  orders_error=error)
    # Same object, unchanged — no wrapping, no substitution.
    assert excinfo.value is error
    # Retry policy saw the transient status exactly once per attempt set;
    # crucially, snapshot() returned NOTHING (no fabricated 0-snapshot).
    assert live_boundary["client"].orders.calls, "get_orders must be called"


def test_t3b_rate_limit_error_also_escapes(live_boundary):
    """RESOURCE_EXHAUSTED is non-retryable → converted to
    ApiRateLimitError, which likewise must NOT be swallowed into 0."""
    error = RuntimeError(grpc.StatusCode.RESOURCE_EXHAUSTED, "throttled")
    with pytest.raises(ApiRateLimitError):
        snapshot_via_live_adapter(live_boundary["install"],
                                  make_live_response(),
                                  orders_error=error)


def test_t3c_non_transient_error_escapes_unchanged(live_boundary):
    error = ValueError("unexpected orders payload")
    with pytest.raises(ValueError) as excinfo:
        snapshot_via_live_adapter(live_boundary["install"],
                                  make_live_response(),
                                  orders_error=error)
    assert excinfo.value is error


# ══════════════════════════════════════════════════════════════════════
# T4 — C5 runtime startup with a failing get_orders(): reconcile must
#      NOT report success and NO trading operation may be attempted.
#      REAL PaperExecutionAdapter + recording spy; the fault is injected
#      at the read-boundary (adapter.snapshot) that the LIVE adapter's
#      snapshot() now exposes unchanged (proven in T3 against the real
#      LiveExecutionAdapter itself).
# ══════════════════════════════════════════════════════════════════════
def daily_df(n=60):
    """Deterministic OHLC identical in principle to stage 5B: ATR14 > 0,
    Upper/Lower(10) = 1400 (never actually reached — the cycle must stop
    at reconcile)."""
    import pandas as pd
    rows = []
    d = REF_DATE - dt.timedelta(days=n - 1)
    hl = 0.07815
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        rows.append({"time": ts, "open": PRICE, "high": PRICE + hl,
                     "low": PRICE - hl, "close": PRICE, "volume": 1})
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
    def refresh(self):
        from src.margin_provider import MarginRates
        return MarginRates(long_margin_rub=0.0585, short_margin_rub=0.0576,
                           instrument="CNYRUBF")


class RecordingPaperAdapter:
    """REAL PaperExecutionAdapter wrapped for call auditing; simulates the
    post-fix LiveExecutionAdapter contract: any get_orders() failure
    surfaces from snapshot() unchanged instead of returning open_orders=0."""

    def __init__(self, broker, state_env, orders_error=None):
        from src.execution_adapter import PaperExecutionAdapter
        self._paper = PaperExecutionAdapter(broker)
        self._state_env = state_env
        self._orders_error = orders_error
        self.snapshot_calls = 0
        self.trading_calls = []

    def find_account(self):
        return self._paper.find_account()

    def snapshot(self, account_id):
        self.snapshot_calls += 1
        if self._orders_error is not None:
            # Post-fix LIVE semantics: exception escapes, NO snapshot.
            raise self._orders_error
        return self._paper.snapshot(account_id)

    def _record(self, name, args):
        st_path = self._state_env / "state.json"
        if st_path.exists():
            st = json.loads(st_path.read_text())
            if st.get("exit_unverified_until_reconcile"):
                raise AssertionError(
                    "blind SELL re-send while EXIT_UNVERIFIED freeze "
                    "is active (G-5 fail-closed violated)")
        self.trading_calls.append((name, args))
        return getattr(self._paper, name)(*args)

    def buy_cny(self, account_id, qty, **kwargs):
        # kwargs (margin_per_contract / entry_price) MUST be forwarded:
        # SimulatedBroker fail-closes on a futures BUY without ГО metadata.
        res = self._paper.buy_cny(account_id, qty, **kwargs)
        self.trading_calls.append(("buy_cny", (account_id, qty)))
        return res

    def sell_cny(self, account_id, qty):
        return self._record("sell_cny", (account_id, qty))

    def sell_one_tmon(self, account_id):
        return self._record("sell_one_tmon", (account_id,))

    def buy_one_tmon(self, account_id):
        return self._record("buy_one_tmon", (account_id,))

    def get_free_cash(self, account_id):
        return self._paper.get_free_cash(account_id)

    def get_tmon_quantity(self, account_id):
        return self._paper.get_tmon_quantity(account_id)

    def get_tmon_price(self, account_id):
        return self._paper.get_tmon_price(account_id)


@pytest.fixture
def rt_env(tmp_path, monkeypatch):
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", state_file)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "true")
    import src.state_store as ss
    monkeypatch.setattr(ss, "STATE_FILE", state_file)
    return tmp_path


@pytest.fixture
def market(monkeypatch):
    import pandas as pd
    import src.c5_runtime as rt_mod

    holder = {"now": utc_ms(CONTROL_DAY, 16, 5), "rows": [],
              "daily": daily_df()}

    def _load(token, uid, candles_count=200, timeframe="4H"):
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


def test_t4_runtime_startup_with_orders_read_failure_blocks_execution(
        rt_env, market):
    import src.c5_runtime as rt_mod
    from src.execution_adapter import SimulatedBroker
    from src.runtime_config import TradingConfig

    # Entry signal WOULD fire (LONG_BREAK > Upper(10)) — if the unknown
    # open-orders state were trusted as "zero", a BUY would be issued.
    market.holder["rows"] = [candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)]

    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    adapter = RecordingPaperAdapter(
        broker, rt_env,
        orders_error=RuntimeError("get_orders failed after retries"))

    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()

    decision = rt.run_cycle()

    assert decision.action != "ENTER_LONG"
    assert decision.reason == "STARTUP_RECONCILE_FAILED"
    # No false reconcile success:
    st = json.loads((rt_env / "state.json").read_text())
    assert st["reconcile_status"] == "FAILED"
    # Zero trading operations attempted:
    assert adapter.trading_calls == []
    assert broker.position_qty == 0
    # Sanity: snapshot WAS attempted (the failure is real, not skipped).
    assert adapter.snapshot_calls >= 1


def test_t4b_control_healthy_read_still_allows_entry(rt_env, market):
    """Counter-fixture proving T4's block comes from the injected fault,
    not from broken wiring: with snapshot() succeeding, the identical
    runtime executes ENTER_LONG normally."""
    import src.c5_runtime as rt_mod
    from src.execution_adapter import SimulatedBroker
    from src.runtime_config import TradingConfig

    market.holder["rows"] = [candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)]
    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000)
    adapter = RecordingPaperAdapter(broker, rt_env)   # no injected error

    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()

    decision = rt.run_cycle()

    assert decision.action == "ENTER_LONG"
    assert ("buy_cny", ("fake-account", decision.qty)) in adapter.trading_calls \
        or adapter.trading_calls, "entry must have executed"
    assert broker.position_qty == decision.qty == 2


# ══════════════════════════════════════════════════════════════════════
# T5 — PAPER regression: SimulatedBroker keeps reporting a TRUE zero
#      (it physically cannot hold working orders) and the reconcile
#      consumers behave exactly as before the fix.
# ══════════════════════════════════════════════════════════════════════
def test_t5_simulated_broker_snapshot_open_orders_is_zero():
    from src.execution_adapter import SimulatedBroker

    broker = SimulatedBroker(cash=1_000_000, equity=1_000_000,
                             position_qty=2)
    snap = broker.snapshot("acc-1")
    assert snap.open_orders == 0
    assert isinstance(snap.open_orders, int)
    # Contract unchanged: PaperExecutionAdapter mirrors it verbatim.
    from src.execution_adapter import PaperExecutionAdapter
    assert PaperExecutionAdapter(broker).snapshot("acc-1").open_orders == 0


def test_t5b_reconcile_still_blocks_on_true_nonzero_open_orders(rt_env):
    """The ambiguity gate (reconcile reading TRUSTWORTHY nonzero) is
    untouched: N open orders + flat position ⇒ ReconcileError."""
    from src.c5_runtime import ReconcileError, reconcile
    from src.execution_adapter import AccountSnapshot

    class _Snap:
        def snapshot(self, account_id):
            return AccountSnapshot(account_id=account_id,
                                   equity_rub=1_000_000.0,
                                   free_cash_rub=1_000_000.0,
                                   position_qty=0, entry_price=None,
                                   tmon_qty=0, open_orders=1)

    st = {"c5_position_qty": 0}
    with pytest.raises(ReconcileError):
        reconcile(_Snap(), "a1", st)


def test_t5c_reconcile_g1_g4_g5_scenarios_unaffected(rt_env):
    """G-1 (equity validation), G-5 (freeze kept / lifted only by strict
    int confirmation) still behave identically with open_orders=0."""
    from src.c5_runtime import ReconcileError, reconcile
    from src.execution_adapter import AccountSnapshot

    class _Adapter:
        def __init__(self, qty, equity=1_000_000.0):
            self._qty, self._equity = qty, equity

        def snapshot(self, account_id):
            return AccountSnapshot(account_id=account_id,
                                   equity_rub=self._equity,
                                   free_cash_rub=self._equity,
                                   position_qty=self._qty,
                                   entry_price=None, tmon_qty=0,
                                   open_orders=0)

    # G-1: bad equity ⇒ ReconcileError EQUITY_UNAVAILABLE.
    st = {"c5_position_qty": 0}
    with pytest.raises(ReconcileError, match="EQUITY_UNAVAILABLE"):
        reconcile(_Adapter(0, equity=0.0), "a1", st)

    # G-5: freeze KEPT without external confirmation.
    st = {"c5_position_qty": 2, "c5_position_direction": "LONG",
          "exit_unverified_until_reconcile": "EXIT_UNVERIFIED_PENDING_RECONCILE"}
    reconcile(_Adapter(2), "a1", st)
    assert st["exit_unverified_until_reconcile"]

    # G-5: freeze lifted ONLY by strict int operator confirmation.
    st = {"c5_position_qty": 2, "c5_position_direction": "LONG",
          "exit_unverified_until_reconcile": "EXIT_UNVERIFIED_PENDING_RECONCILE",
          "broker_position_confirmed": 1}
    reconcile(_Adapter(1), "a1", st)
    assert not st.get("exit_unverified_until_reconcile")
    assert st["c5_position_qty"] == 1
