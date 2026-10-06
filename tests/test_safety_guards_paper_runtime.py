"""Safety-guard tests for the PAPER runtime (Stage 5B harness reused).

Every test drives the REAL ``src.c5_runtime.C5Runtime.run_cycle()`` in
TRADING_MODE=PAPER through the REAL ``PaperExecutionAdapter`` +
``SimulatedBroker``.  Fakes exist ONLY at the external boundaries that
are already faked in Stage 5B — market data (load_candles /
fetch_order_book), clock (now_fn) and the margin-rate API
(FakeMarginTracker).  Nothing inside c5_core, execution_guard,
cash_manager, protection or state_store is mocked, and no execution
adapter is monkeypatched.

Six safety branches are covered:

REACHABLE through the real production call chain (plain PASS):
  1. ORDER_BOOK_UNAVAILABLE  — fetch_order_book boundary raises;
     c5_runtime._evaluate_entry converts it to NO_ENTRY via
     c5_core.block_entry().
  2. NO_BID_OR_ASK           — empty book lists fail the pure
     execution guard early ("NO_BID_OR_ASK" in failed_checks).
  3. INVALID_QUOTE           — crossed/invalid top-of-book quote
     fails the guard with "INVALID_QUOTE".
  4. QTY_ZERO                — free_margin below one contract's ГО
     makes the frozen sizing pipeline floor qty_raw to 0 →
     NO_ENTRY/QTY_ZERO before any execution layer runs.

NOT REACHABLE through the real PAPER call chain (xfail strict=True —
documented production gaps, do NOT "fix" them by weakening the tests):

  GAP A — SimulatedBrokerError on BUY:
      The runtime's ``required_margin`` and the margin that
      ``SimulatedBroker.apply()`` charges come from the SAME set of
      inputs (same rate × price × C5_CONTRACT_SIZE_CNY × same qty).
      ``ensure_cash_for_entry()`` guarantees broker cash >= required
      margin BEFORE ``buy_cny`` is called, so a broker-side margin
      mismatch cannot occur through the allowed deterministic hooks.
      Triggering SimulatedBrokerError would require artificial
      desynchronisation (monkeypatched adapter/broker) or a production
      change — both forbidden here.

  GAP B — TMON_SELL_N_NOT_CONFIRMED:
      ``PaperExecutionAdapter.sell_one_tmon()`` always returns an OK
      ExecutionResult (it never translates a broker rejection into a
      failed result), and ``SimulatedBroker.apply()`` never refuses a
      TMON SELL even when ``tmon_qty < 1`` (it just decrements).  The
      cash_manager branch
      ``if not sold: → TMON_SELL_{n}_NOT_CONFIRMED`` is therefore
      unreachable through the genuine PAPER execution path without
      changing production code or substituting the execution adapter.

The two xfail tests assert the PRODUCTION-CORRECT behaviour (the
runtime must surface those failures as machine-readable NO_ENTRY
decisions).  They fail today — that failure is the regression marker
until the gaps are closed.  Do not delete them and do not convert them
into passing tests with weakened expectations.
"""
from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# ── deterministic market fixture constants (Stage 5B shape) ───────────
REF_DATE = dt.date(2026, 9, 30)         # last closed daily bar
CONTROL_DAY = dt.date(2026, 10, 2)      # control day (Friday)
LONG_BREAK = 1420.0                     # > Upper(10) = 1400
PRICE = 1400.0
DAILY_HL = 0.07815                      # half-range of each daily bar

# ГО per contract at LONG_BREAK with the 0.0585 long rate (frozen
# c5_core formula: price × C5_CONTRACT_SIZE_CNY(1000) × rate).
MARGIN_PER_CONTRACT_AT_LONG_BREAK = float(
    (Decimal(str(LONG_BREAK)) * Decimal("1000.0")
     * Decimal("0.0585")).quantize(Decimal("0.0001")))  # 83 070 RUB


# ══════════════════════════════════════════════════════════════════════
# boundary fakes (market data / clock / margin) — NOTHING inside
# c5_runtime, c5_core, execution_guard, cash_manager, protection or
# state_store is mocked.
# ══════════════════════════════════════════════════════════════════════
def daily_df(n=60):
    """Deterministic OHLC: H=1400+HL, L=1400-HL, C=1400.

    Wilder TR = 2*HL per bar → ATR14 ≈ 0.1563 (non-zero), risk
    multiplier = clip(0.1563/0.1563, 0.5, 1.5) = 1.0; channel stays at
    1400 so LONG_BREAK keeps its breakout semantics.
    """
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
    """Patch the network boundaries of c5_runtime; restore after test.

    ``holder["book"]`` controls what the REAL fetch_order_book wrapper
    hands to the REAL evaluate_execution_guard:
      * callable  → invoked (may raise → ORDER_BOOK_UNAVAILABLE);
      * tuple     → returned as-is (([], []) → NO_BID_OR_ASK);
      * None      → default healthy two-sided book.
    """
    import pandas as pd
    import src.c5_runtime as rt_mod

    holder = {"now": utc_ms(CONTROL_DAY, 16, 5),
              "rows": [],
              "daily": daily_df(),
              "book": None}

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return holder["daily"].copy()
        if timeframe == "5m":
            return pd.DataFrame(holder["rows"])
        raise AssertionError(f"C5 runtime must not request {timeframe!r}")

    def _book(token, uid, depth=10):
        b = holder["book"]
        if callable(b):
            return b()
        if b is None:
            return ([(PRICE - 0.05, 100)], [(PRICE + 0.05, 100)])
        return b

    saved_load, saved_book = rt_mod.load_candles, rt_mod.fetch_order_book
    rt_mod.load_candles = _load
    rt_mod.fetch_order_book = _book
    yield SimpleNamespace(holder=holder, mod=rt_mod)
    rt_mod.load_candles, rt_mod.fetch_order_book = saved_load, saved_book


def make_runtime(market, broker, *, config=None, flat=False):
    """Real C5Runtime + REAL PaperExecutionAdapter (production paper
    order flow, no recording fake, no monkeypatched adapter methods).

    ``flat=True`` injects an all-zero position snapshot so the cycle
    follows the ENTRY branch even while the simulated broker holds a
    position (same helper pattern as Stage 5B; isolates the entry
    guards from the EXIT branch without touching production logic).
    """
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter
    from src.runtime_config import TradingConfig

    adapter = PaperExecutionAdapter(broker)
    if flat:
        from src.execution_adapter import AccountSnapshot
        adapter.snapshot = lambda account_id: AccountSnapshot(
            account_id=account_id, equity_rub=broker.equity,
            free_cash_rub=broker.cash, position_qty=0, entry_price=None,
            tmon_qty=broker.tmon_qty, open_orders=0)
    cfg = config if config is not None else TradingConfig("PAPER", True)
    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=cfg,
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()   # PAPER bootstrap skips live lookup
    return rt, adapter


def new_broker(cash=1_000_000, tmon_qty=0):
    from src.execution_adapter import SimulatedBroker
    return SimulatedBroker(cash=cash, equity=1_000_000,
                           tmon_qty=tmon_qty)


def set_candles(market, *candles):
    market.holder["rows"] = list(candles)


def buy_orders(adapter):
    return [o for o in adapter.orders if o["side"] == "BUY"
            and o["instrument"] == "CNYRUBF"]


ENTRY_CANDLE = lambda: candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)


# ══════════════════════════════════════════════════════════════════════
# 1 — ORDER_BOOK_UNAVAILABLE (reachable: fetch_order_book boundary
#     raises → c5_runtime converts it to NO_ENTRY/ORDER_BOOK_UNAVAILABLE)
# ══════════════════════════════════════════════════════════════════════
def test_safety_1_order_book_unavailable_blocks_entry(state_env, market):
    set_candles(market, ENTRY_CANDLE())

    def boom(*a, **k):
        raise RuntimeError("order book endpoint down")

    market.holder["book"] = boom

    rt, adapter = make_runtime(market, new_broker())
    decision = rt.run_cycle()

    assert decision.action == "NO_ENTRY"
    assert decision.reason == "ORDER_BOOK_UNAVAILABLE"
    assert decision.qty == 0
    # nothing reached the execution layer
    assert adapter.orders == []
    assert buy_orders(adapter) == []
    # control point consumed exactly once (duplicate-order safety)
    state = read_state(state_env)
    assert state["last_c5_entry_control_timestamp"] is not None


# ══════════════════════════════════════════════════════════════════════
# 2 — NO_BID_OR_ASK (reachable: empty book → pure execution guard fails)
# ══════════════════════════════════════════════════════════════════════
def test_safety_2_no_bid_or_ask_blocks_entry(state_env, market):
    set_candles(market, ENTRY_CANDLE())
    market.holder["book"] = ([], [])

    rt, adapter = make_runtime(market, new_broker())
    decision = rt.run_cycle()

    assert decision.action == "NO_ENTRY"
    assert "NO_BID_OR_ASK" in decision.reason
    assert decision.qty == 0
    assert adapter.orders == []


# ══════════════════════════════════════════════════════════════════════
# 3 — INVALID_QUOTE (reachable: crossed top-of-book → guard fails)
# ══════════════════════════════════════════════════════════════════════
def test_safety_3_invalid_quote_blocks_entry(state_env, market):
    set_candles(market, ENTRY_CANDLE())
    # ask below bid (crossed/stale quote) → INVALID_QUOTE in the guard.
    # The pair is placed symmetrically around the mid so the expected
    # slippage vs. mid stays ≈0.35% and the guard reason is exactly
    # "EXECUTION_GUARD_INVALID_QUOTE" (a wide crossed pair would also
    # trip SLIPPAGE and the runtime maps that to "SLIPPAGE_GUARD").
    market.holder["book"] = ([(1405.0, 100)], [(1400.0, 100)])

    rt, adapter = make_runtime(market, new_broker())
    decision = rt.run_cycle()

    assert decision.action == "NO_ENTRY"
    assert "INVALID_QUOTE" in decision.reason
    assert decision.qty == 0
    assert adapter.orders == []


# ══════════════════════════════════════════════════════════════════════
# 4 — QTY_ZERO (reachable: free_margin < one contract ГО → frozen
#     sizing floors qty_raw to 0 → NO_ENTRY/QTY_ZERO pre-execution)
# ══════════════════════════════════════════════════════════════════════
def test_safety_4_qty_zero_blocks_entry(state_env, market):
    set_candles(market, ENTRY_CANDLE())
    # equity 1M keeps M/E sane; free cash below one contract's ГО
    # (83 070 RUB) → qty_margin = 0 → qty_raw = 0 → QTY_ZERO.
    broker = new_broker(cash=MARGIN_PER_CONTRACT_AT_LONG_BREAK - 1.0)

    rt, adapter = make_runtime(market, broker)
    decision = rt.run_cycle()

    assert decision.action == "NO_ENTRY"
    assert decision.reason == "QTY_ZERO"
    assert decision.qty == 0
    assert adapter.orders == []


# ══════════════════════════════════════════════════════════════════════
# 5 — TMON_SELL_N_NOT_CONFIRMED — GAP B: UNREACHABLE via real PAPER
#     execution path.  xfail(strict=True) is the fixed marker until the
#     production gap is closed (see module docstring).
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(
    strict=True,
    reason="Production gap B: PaperExecutionAdapter.sell_one_tmon() "
           "always returns an OK ExecutionResult and never translates a "
           "broker rejection into a failed result; SimulatedBroker.apply() "
           "never produces such a refusal (TMON SELL succeeds even when "
           "tmon_qty < 1). Therefore the cash_manager branch "
           "TMON_SELL_N_NOT_CONFIRMED is unreachable through the real "
           "PAPER execution path without changing production code or "
           "substituting the execution adapter.")
def test_safety_5_tmon_sell_not_confirmed_surfaces_as_no_entry(state_env,
                                                               market):
    """Production-correct expectation (fails today — documented gap).

    Scenario: funding loop needs TMON liquidation, but the broker has
    already run out of TMON mid-loop (simulating an unfilled/rejected
    sell).  The runtime MUST surface the cash-manager's
    ``TMON_SELL_<n>_NOT_CONFIRMED`` as a machine-readable NO_ENTRY.

    Today the real adapter reports every simulated TMON SELL as
    confirmed, so the cycle reaches BUY instead — the assertion below
    fails, which is exactly the xfail we pin here.
    """
    set_candles(market, ENTRY_CANDLE())
    # cash below required margin → ensure_cash_for_entry() enters the
    # TMON loop; broker starts with 1 TMON bond.
    broker = new_broker(cash=60_000.0, tmon_qty=1)

    rt, adapter = make_runtime(market, broker, flat=True)
    decision = rt.run_cycle()

    # Production-correct behaviour: unconfirmed TMON sell ⇒ NO_ENTRY
    # with the explicit cash-manager reason, and no CNY BUY executed.
    assert decision.action == "NO_ENTRY"
    assert decision.reason == "TMON_SELL_1_NOT_CONFIRMED"
    assert buy_orders(adapter) == []


# ══════════════════════════════════════════════════════════════════════
# 6 — SimulatedBrokerError on BUY — GAP A: UNREACHABLE via allowed
#     deterministic hooks.  xfail(strict=True) pinned until closed.
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(
    strict=True,
    reason="Production gap A: the runtime's required_margin and the "
           "margin charged by SimulatedBroker.apply() are computed from "
           "the same inputs (same rate x price x contract size x qty), "
           "and ensure_cash_for_entry() funds the account up to that "
           "value before buy_cny; the extra kwargs forwarded by "
           "_evaluate_entry overwrite any hook-injected margin metadata. "
           "Through the allowed deterministic hooks (market data, clock, "
           "margin rates, order book, broker seed) a broker-side margin "
           "mismatch cannot be produced, so SimulatedBrokerError on BUY "
           "is unreachable without artificial desynchronisation "
           "(monkeypatched adapter/broker) or a production change.")
def test_safety_6_simulated_broker_error_on_buy_surfaces_as_no_entry(
        state_env, market):
    """Production-correct expectation (fails today — documented gap).

    Scenario: a PAPER futures BUY whose margin the broker-side model
    rejects (e.g. insufficient margin) must NOT crash the runtime and
    must surface as NO_ENTRY/EXECUTION_FAILED:* with zero orders.

    Through the consistent production call chain this is unreachable:
    runtime and SimulatedBroker use the same required_margin, so the
    cycle completes the BUY successfully and the assertion below fails
    — the exact failure we pin as xfail(strict=True).
    """
    set_candles(market, ENTRY_CANDLE())
    # Fully funded, healthy book, valid signal → sizing yields qty=1
    # (free cash covers exactly one contract's ГО of 83 070 RUB).
    broker = new_broker(cash=MARGIN_PER_CONTRACT_AT_LONG_BREAK + 100.0)

    rt, adapter = make_runtime(market, broker)
    try:
        decision = rt.run_cycle()
    except Exception as exc:  # a raw SimulatedBrokerError crash also fails
        pytest.fail(f"SimulatedBrokerError leaked through the runtime "
                    f"instead of surfacing as NO_ENTRY: {exc!r}")

    # Production-correct behaviour: broker-refused BUY ⇒ NO_ENTRY with
    # an EXECUTION_FAILED reason, not a successful ENTER_LONG.
    assert decision.action == "NO_ENTRY"
    assert decision.reason.startswith("EXECUTION_FAILED"), \
        f"expected EXECUTION_FAILED, got {decision.reason}"
    assert buy_orders(adapter) == []
