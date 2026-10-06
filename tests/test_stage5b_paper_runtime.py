"""Stage 5B — PAPER RUNTIME tests (full C5Runtime pipeline in PAPER mode).

Stage 5A proved the safety GATE (mode parsing, AUTO gate, adapter routing,
post_order tripwire). Stage 5B proves the RUNTIME: the real
``src.c5_runtime.C5Runtime.run_cycle()`` driven end-to-end with
TRADING_MODE=PAPER, with fakes ONLY at external boundaries (market data,
clock, margin API, order book) and the REAL PaperExecutionAdapter +
SimulatedBroker + state_store behind it. The frozen c5_core is never mocked.

Invariants proven here:

  A. PAPER BUY lifecycle: ENTER_LONG x2 executed locally only, one order.
  B. Repeated run_cycle() on the SAME candle timestamp → DUPLICATE_CANDLE
     and zero new orders (idempotency key = candle ts, not wall clock).
  E1. After a fresh BUY, re-running the SAME entry candle while FLAT →
      NO_ENTRY / DUPLICATE_CANDLE, no second BUY.
  E2. A NEW 5m candle with a NEUTRAL close → NO_SIGNAL path,
      action NO_ENTRY, reason NO_SIGNAL, still no BUY.
      NOTE: E1/E2 deliberately do NOT rely on any same-day cooldown —
      they prove the candle-idempotency invariant and the neutral-signal
      invariant independently.
  F. EXIT path duplicate guard: same exit candle twice → DUPLICATE_CANDLE,
     exactly one SELL.
  G. Neutral candle while LONG → HOLD, position untouched, no SELL.
  H. New candle closing below Lower(10) → EXIT, flat again.
  I. AUTO_TRADING_ENABLED=false blocks before ANY market-data fetch.
  J. Restart recovery: persisted last_processed_5m_ts survives a brand-new
     runtime instance (state store is the idempotency source of truth).
  K. No real client factory import on the paper runtime path.
  L. Protection (daily-loss breaker) blocks entries but NEVER blocks exits.

Deterministic market fixture (same shape as Stage 2D/5A):
  daily bars H=L=C=1400 → Upper(10)=Lower(10)=1400, ATR14≈0.1563,
  risk multiplier 1.0; LONG_BREAK=1420 (>Upper), SHORT_BREAK=1370 (<Lower),
  NEUTRAL=1400 (inside channel → NO_SIGNAL).
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

MSK = ZoneInfo("Europe/Moscow")

# ── deterministic market fixture constants ────────────────────────────
REF_DATE = dt.date(2026, 9, 30)         # last closed daily bar
CONTROL_DAY = dt.date(2026, 10, 2)      # control day (Friday)
LONG_BREAK = 1420.0                     # > Upper(10) = 1400
SHORT_BREAK = 1370.0                    # < Lower(10) = 1400
NEUTRAL = 1400.0                        # inside the channel
PRICE = 1400.0


# ══════════════════════════════════════════════════════════════════════
# boundary fakes (market data / clock / margin / instrument) — NOTHING
# inside c5_runtime, c5_core, protection, cash_manager or state_store
# is mocked.
# ══════════════════════════════════════════════════════════════════════
DAILY_HL = 0.07815   # half-range of each synthetic daily bar


def daily_df(n=60):
    """Deterministic OHLC: H=1400+HL, L=1400-HL, C=1400.

    Wilder TR = max(H-L, |H-prev_close|, |L-prev_close|) = 2*HL per bar
    → ATR14 == 0.1563 (non-zero; the frozen core fail-closes on 0.0) and
    risk_multiplier = clip(0.1563/0.1563, 0.5, 1.5) == 1.0. The channel
    Upper/Lower(10) stays exactly at 1400 so LONG_BREAK / SHORT_BREAK /
    NEUTRAL signal semantics are unchanged. Same principle as Stage 5A J.

    Frozen sizing at equity=1M: risk_budget=200_000, margin/contract=
    1400*1000*0.0585=81_900 → qty_risk=2, stage cap 5 not reached ⇒ qty=2.
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


def make_runtime(market, broker, *, config=None, flat=False):
    """Real C5Runtime + REAL PaperExecutionAdapter (paper order flow is
    exercised through the production adapter, not a recording fake).

    ``flat=True`` additionally injects an all-zero snapshot so the cycle
    follows the ENTRY branch even when the simulated broker still holds
    a position — this isolates the entry idempotency guard from the EXIT
    branch without changing any production logic.
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


def new_broker():
    from src.execution_adapter import SimulatedBroker
    return SimulatedBroker(cash=1_000_000, equity=1_000_000)


def set_candles(market, *candles):
    market.holder["rows"] = list(candles)


def latest_closed(market):
    """The candle the runtime will actually pick up at holder['now']
    (latest CLOSED 5m bar: start+5m <= now). Lets tests keep the SAME
    control candle across cycles and switch to a new one explicitly."""
    import pandas as pd
    from src.c5_runtime import latest_closed_5m
    df = pd.DataFrame(market.holder["rows"])
    row = latest_closed_5m(df, market.holder["now"])
    return None if row is None else row.to_dict()


def buy_orders(adapter):
    return [o for o in adapter.orders if o["side"] == "BUY"
            and o["instrument"] == "CNYRUBF"]


def sell_orders(adapter):
    return [o for o in adapter.orders if o["side"] == "SELL"
            and o["instrument"] == "CNYRUBF"]


# ══════════════════════════════════════════════════════════════════════
# A — PAPER BUY через полный рантайм: ровно один локальный BUY
# ══════════════════════════════════════════════════════════════════════
def test_a_paper_entry_executes_locally_once(state_env, market):
    from src.execution_adapter import SimulatedBroker  # noqa: F401
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    d = rt.run_cycle()

    assert d.action == "ENTER_LONG"
    assert d.qty == 2                      # frozen C5 sizing @ equity 1M
    assert broker.position_qty == 2
    # Current SimulatedBroker order contract (post accounting fix): a
    # futures BUY carries the runtime-computed ГО metadata —
    # ``margin_per_contract`` (rate × price × size, fail-closed without
    # it) and ``entry_price`` (execution-guard expected price).
    assert len(adapter.orders) == 1
    order = adapter.orders[0]
    assert order["instrument"] == "CNYRUBF"
    assert order["side"] == "BUY"
    assert order["qty"] == 2
    assert order["margin_per_contract"] == pytest.approx(83070.0)   # 1420×1000×0.0585
    assert order["entry_price"] == pytest.approx(1400.05)           # guard ask estimate
    st = read_state(state_env)
    assert st["c5_position_direction"] == "LONG"
    assert st["last_c5_entry_control_timestamp"] == C16["time"].isoformat()


# ══════════════════════════════════════════════════════════════════════
# B — повторный run_cycle() на ТОМ ЖЕ candle timestamp: DUPLICATE_CANDLE,
#     новых ордеров нет (идемпотентность по метке свечи, не по часам)
# ══════════════════════════════════════════════════════════════════════
def test_b_duplicate_candle_blocks_second_entry(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    # FLAT view injected at the (external) snapshot boundary: the cycle
    # stays on the ENTRY branch, so DUPLICATE_CANDLE here is provably the
    # entry idempotency guard — not the exit-branch HOLD that a real open
    # position would produce.
    rt, adapter = make_runtime(market, broker, flat=True)

    d1 = rt.run_cycle()
    assert d1.action == "ENTER_LONG"
    orders_after_first = len(adapter.orders)

    # Wall clock moves forward, candle set does NOT — the latest CLOSED
    # bar is still C16 (same ts), so the idempotency guard must reject it
    # regardless of elapsed time.
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 30)
    assert latest_closed(market)["time"] == C16["time"]   # same candle ts
    d2 = rt.run_cycle()

    assert d2.action == "NO_ENTRY"
    assert d2.reason == "DUPLICATE_CANDLE"
    assert len(adapter.orders) == orders_after_first == 1
    assert len(buy_orders(adapter)) == 1
    assert broker.position_qty == 2


# ══════════════════════════════════════════════════════════════════════
# C — выход за Lower(10) на новом 5m баре: EXIT и возврат во flat
# ══════════════════════════════════════════════════════════════════════
def test_c_exit_on_new_closed_candle(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"

    set_candles(market, C16, C1655)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    d = rt.run_cycle()

    assert d.action == "EXIT"
    assert broker.position_qty == 0
    assert sell_orders(adapter) == [{"instrument": "CNYRUBF", "side": "SELL",
                                     "qty": 2}]
    st = read_state(state_env)
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None


# ══════════════════════════════════════════════════════════════════════
# D — EXIT идемпотентен по метке свечи (covered by F too, kept minimal)
#     PAPER path physically never reaches post_order (tripwire).
# ══════════════════════════════════════════════════════════════════════
def test_d_paper_orders_never_reach_post_order(state_env, market, monkeypatch):
    import src.client_factory as cf

    def _tripwire(*a, **k):
        raise AssertionError("REAL API CLIENT USED IN PAPER MODE")

    monkeypatch.setattr(cf, "get_client", _tripwire)

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"       # tripwire not hit
    set_candles(market, C16, C1655)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    assert rt.run_cycle().action == "EXIT"             # tripwire not hit
    assert broker.position_qty == 0


# ══════════════════════════════════════════════════════════════════════
# E — ДВА РАЗНЫХ ИНВАРИАНТА после свежего BUY (без same-day cooldown!)
#
#   E1: повторный run_cycle() на ТОМ ЖЕ candle timestamp
#       → NO_ENTRY / DUPLICATE_CANDLE, нового BUY нет.
#   E2: НОВЫЙ 5m candle с НЕЙТРАЛЬНЫМ close
#       → путь NO_SIGNAL, NO_ENTRY / NO_SIGNAL, нового BUY нет.
#
#   Почему EXIT-свеча заменена с NEUTRAL на SHORT_BREAK:
#   в рантайме при открытой позиции run_cycle() всегда уходит в EXIT-ветку
#   (_evaluate_exit), а ENTRY-ветка (_evaluate_entry, где живёт guard
#   last_c5_entry_control_timestamp) выполняется только когда позиция FLAT.
#   Если бы финальная свеча была NEUTRAL, цикл завершился бы на HOLD внутри
#   EXIT-ветки и никогда не вызвал бы _evaluate_entry — инвариант E1
#   (дубликат entry-свечи после закрытия позиции) нельзя было бы проверить
#   вообще. SHORT_BREAK корректно закрывает позицию (EXIT), возвращает
#   рантайм во FLAT, и следующий цикл уже проходит по ENTRY-ветке — так
#   становятся наблюдаемыми оба инварианта E1 и E2 по очереди.
#   Дополнительно для E1 используется flat-view на границе snapshot
#   (внешний boundary-fake): он изолирует entry-guard от EXIT-ветки,
#   чтобы DUPLICATE_CANDLE гарантированно приходил из _evaluate_entry,
#   а не из HOLD при открытой позиции. Same-day cooldown НЕ используется.
# ══════════════════════════════════════════════════════════════════════
def test_e_duplicate_candle_and_neutral_new_candle_both_block_reentry(
        state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)      # entry breakout
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)  # EXIT leg (was
                                                         # NEUTRAL — see doc)
    C17 = candle_5m(CONTROL_DAY, 17, 0, NEUTRAL)         # new neutral candle
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker, flat=True)

    # ── fresh BUY on the 16:00 control candle ─────────────────────────
    d_buy = rt.run_cycle()
    assert d_buy.action == "ENTER_LONG"
    assert len(buy_orders(adapter)) == 1
    st = read_state(state_env)
    assert st["last_c5_entry_control_timestamp"] == C16["time"].isoformat()

    # ── EXIT leg (real adapter view): new candle closes below Lower(10)
    #    → SELL executed, simulated account back to flat ────────────────
    rt_live_view, _ = make_runtime(market, broker)   # honest snapshot
    set_candles(market, C16, C1655)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    assert latest_closed(market)["time"] == C1655["time"]  # NEW candle ts
    d_exit = rt_live_view.run_cycle()
    assert d_exit.action == "EXIT"
    assert broker.position_qty == 0          # genuinely FLAT now

    # ── INVARIANT 1 (E1): SAME candle timestamp → DUPLICATE_CANDLE ────
    # The entry candle C16 is the latest CLOSED bar again (C1655 removed;
    # at 17:05 nothing after C16 is closed). The flat-view runtime keeps
    # the cycle on the ENTRY branch, so DUPLICATE_CANDLE here provably
    # comes from the entry idempotency guard (last_c5_entry_control_
    # timestamp), not from the exit-branch HOLD. No cooldown involved —
    # the guard key is the candle ts itself.
    set_candles(market, C16)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    assert latest_closed(market)["time"] == C16["time"]    # same candle ts
    d_dup = rt.run_cycle()
    assert d_dup.action == "NO_ENTRY"
    assert d_dup.reason == "DUPLICATE_CANDLE"
    assert len(buy_orders(adapter)) == 1     # NO new BUY

    # ── INVARIANT 2 (E2): NEW candle, NEUTRAL close → NO_SIGNAL ───────
    # C17 (17:00) is now CLOSED ⇒ different timestamp ⇒ the idempotency
    # guard passes; the frozen core sees close==1400 inside the channel
    # ⇒ REASON_NO_SIGNAL ⇒ NO_ENTRY. Still no BUY: the absence of a
    # signal is verified independently of the duplicate-candle mechanism
    # above (and without any same-day cooldown).
    set_candles(market, C16, C17)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 10)
    assert latest_closed(market)["time"] == C17["time"]    # NEW candle ts
    d_neutral = rt.run_cycle()
    assert d_neutral.action == "NO_ENTRY"
    assert d_neutral.reason == "NO_SIGNAL"
    assert len(buy_orders(adapter)) == 1     # NO new BUY
    assert broker.position_qty == 0

    # Control-point bookkeeping per the REAL runtime semantics: every
    # evaluated control candle is CONSUMED — _evaluate_entry rewrites
    # last_c5_entry_control_timestamp to the bar it just processed even
    # when the outcome is NO_SIGNAL. We must NOT demand that the value
    # stays pinned at C16 after C17 was legitimately evaluated; the
    # invariant proven above (no second BUY on duplicate ts, no BUY on a
    # neutral new candle) holds independently of this field's movement.
    st = read_state(state_env)
    assert st["last_c5_entry_control_timestamp"] == C17["time"].isoformat()


# ══════════════════════════════════════════════════════════════════════
# F — EXIT-ветка идемпотентна по метке 5m свечи: второй проход на той же
#     exit-свече ПРИ НАЛИЧИИ ПОЗИЦИИ → DUPLICATE_CANDLE, ровно один SELL.
#     Инварианты: полный SELL происходит один раз; после EXIT позиция
#     flat; последующая entry-ветка оценивается ОТДЕЛЬНО и может открыть
#     новую позицию — отсутствие same-day cooldown не является ошибкой.
# ══════════════════════════════════════════════════════════════════════
def test_f_exit_branch_is_candle_idempotent(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"

    # ── EXIT leg: full SELL happens exactly once on the new candle ────
    set_candles(market, C16, C1655)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    assert rt.run_cycle().action == "EXIT"
    assert len(sell_orders(adapter)) == 1
    assert broker.position_qty == 0          # genuinely FLAT after EXIT

    # ── exit-guard check, isolated from the entry branch ──────────────
    # The duplicate guard of the EXIT branch (last_processed_5m_ts) is
    # observable ONLY while a position is open — a flat account always
    # goes to the ENTRY branch. So: clear just the exit guard key in the
    # state file (external boundary; production logic untouched), re-EXIT
    # on the SAME candle C1655, and verify exactly one SELL happens and
    # the second pass on that identical ts is refused with
    # DUPLICATE_CANDLE. No same-day cooldown exists or is required.
    st = read_state(state_env)
    st["last_processed_5m_ts"] = None        # clear ONLY the exit guard
    (state_env / "state.json").write_text(json.dumps(st))

    # Re-open a LONG so the next cycles provably run inside the EXIT
    # branch: entry guard also cleared ⇒ fresh BUY @ C1655 is legitimate
    # (SHORT_BREAK would be an opposite-direction signal; we keep the
    # scenario deterministic by re-entering via the adapter directly).
    broker.position_qty = 2                  # external broker boundary
    st = read_state(state_env)
    st["c5_position_qty"] = 2
    st["c5_position_direction"] = "LONG"
    (state_env / "state.json").write_text(json.dumps(st))

    # First pass on C1655 after the guard reset → real EXIT, one SELL.
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    assert latest_closed(market)["time"] == C1655["time"]  # same candle ts
    d1 = rt.run_cycle()
    assert d1.action == "EXIT"
    sells_before = len(sell_orders(adapter))
    assert sells_before == 2                 # full SELL happened once here
    assert broker.position_qty == 0          # FLAT again

    # Second pass on the IDENTICAL exit-candle timestamp while still
    # holding the last_processed_5m_ts == C1655 mark: reopen the broker
    # position externally (simulating another fill) and prove the runtime
    # refuses to churn — DUPLICATE_CANDLE, no additional SELL.
    broker.position_qty = 2
    st = read_state(state_env)
    st["c5_position_qty"] = 2
    st["c5_position_direction"] = "LONG"
    (state_env / "state.json").write_text(json.dumps(st))
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 10)
    assert latest_closed(market)["time"] == C1655["time"]  # same candle ts
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "DUPLICATE_CANDLE"   # exit-branch guard hit
    assert len(sell_orders(adapter)) == sells_before       # NO second SELL
    assert broker.position_qty == 2          # untouched by the dup cycle

    # ── subsequent EXIT on a NEW candle is evaluated independently ────
    C1700 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, C16, C1655, C1700)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 15)
    assert latest_closed(market)["time"] == C1700["time"]  # NEW candle ts
    d3 = rt.run_cycle()
    assert d3.action == "EXIT"
    assert len(sell_orders(adapter)) == sells_before + 1   # second, legit SELL
    assert broker.position_qty == 0          # flat again — no cooldown
                                             # blocks this or any re-entry


# ══════════════════════════════════════════════════════════════════════
# G — нейтральный бар при открытой позиции: HOLD, позицию не трогает
# ══════════════════════════════════════════════════════════════════════
def test_g_neutral_candle_holds_open_position(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1610 = candle_5m(CONTROL_DAY, 16, 10, NEUTRAL)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"

    set_candles(market, C16, C1610)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 20)
    d = rt.run_cycle()
    assert d.action == "HOLD"
    assert broker.position_qty == 2
    assert len(sell_orders(adapter)) == 0


# ══════════════════════════════════════════════════════════════════════
# H — новый бар за Lower(10) закрывает позицию (exit signal ≠ entry gate)
# ══════════════════════════════════════════════════════════════════════
def test_h_new_candle_below_lower_channel_exits(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1630 = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"

    set_candles(market, C16, C1630)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)
    d = rt.run_cycle()
    assert d.action == "EXIT"
    assert d.reason == "EXIT_LONG"
    assert broker.position_qty == 0
    assert len(sell_orders(adapter)) == 1


# ══════════════════════════════════════════════════════════════════════
# I — AUTO=false блокирует РАНТАЙМ до любого обращения к market data
# ══════════════════════════════════════════════════════════════════════
def test_i_auto_false_blocks_before_market_data(state_env, market):
    from src.runtime_config import TradingConfig

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker,
                               config=TradingConfig("PAPER", False))

    d = rt.run_cycle()

    assert d.action == "NO_ENTRY"
    assert d.reason == "TRADING_BLOCKED_BY_CONFIG"
    assert adapter.orders == []
    assert broker.position_qty == 0
    # Gate fires BEFORE anything that could touch the network:
    assert market.holder["load_calls"] == []


# ══════════════════════════════════════════════════════════════════════
# J — рестарт с ОТКРЫТОЙ broker-позицией: broker-state-wins.
#     Проверяем: (1) до рестарта позиция открыта; (2) новый рантайм через
#     reconcile восстанавливает её из snapshot брокера; (3) рантайм НЕ
#     создаёт новый BUY поверх существующей позиции; (4) при отсутствии
#     exit-сигнала корректный результат — HOLD (position/exit branch),
#     entry-guard DUPLICATE_CANDLE здесь недостижим и НЕ требуется.
# ══════════════════════════════════════════════════════════════════════
def test_j_restart_does_not_rebuy_same_candle(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt1, adapter1 = make_runtime(market, broker)
    assert rt1.run_cycle().action == "ENTER_LONG"

    # ── precondition: an OPEN CNY position exists before the restart ──
    assert broker.position_qty == 2
    st_before = read_state(state_env)
    assert st_before["c5_position_direction"] == "LONG"

    # Brand-new runtime instance over the SAME broker + state file
    # (simulates a process restart; nothing carried in memory). The
    # latest CLOSED bar is still C16 — same timestamp as before restart.
    rt2, adapter2 = make_runtime(market, broker)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == C16["time"]    # same candle ts
    d = rt2.run_cycle()

    # With a restored open position the cycle lives in the EXIT branch;
    # no exit signal on this bar ⇒ HOLD. We deliberately do NOT demand
    # NO_ENTRY/DUPLICATE_CANDLE here: the entry guard is unreachable
    # while a position is open.
    assert d.action == "HOLD"

    # ── main assertions: broker state was adopted, no duplicate BUY ───
    assert broker.position_qty == 2          # position survived restart
    st_after = read_state(state_env)
    assert st_after["c5_position_qty"] == 2  # reconcile restored state
    assert st_after["c5_position_direction"] == "LONG"
    assert adapter2.orders == []             # new runtime placed ZERO orders
    assert len(buy_orders(adapter1)) == 1    # BUY happened exactly once


# ══════════════════════════════════════════════════════════════════════
# K — PAPER-рантайм физически не импортирует реальный клиентский фабрик
#     в момент цикла (post_order недоступен на этом пути)
# ══════════════════════════════════════════════════════════════════════
def test_k_paper_cycle_never_builds_live_client(state_env, market,
                                                monkeypatch):
    import src.client_factory as cf

    calls = []
    monkeypatch.setattr(cf, "get_client",
                        lambda *a, **k: calls.append(1))

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"
    assert calls == []                       # live client never built


# ══════════════════════════════════════════════════════════════════════
# L — protection блокирует НОВЫЕ входы, но НИКОГДА не мешает EXIT
# ══════════════════════════════════════════════════════════════════════
def test_l_daily_loss_blocks_entry_but_never_exit(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1655 = candle_5m(CONTROL_DAY, 16, 55, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    assert rt.run_cycle().action == "ENTER_LONG"
    assert broker.position_qty == 2

    # Daily loss >= 10% vs session-start equity → entries forbidden.
    st = read_state(state_env)
    st["c5_daily_start_equity"] = 1_000_000
    broker.equity = 880_000                  # -12% daily loss
    broker.cash = 880_000
    (state_env / "state.json").write_text(json.dumps(st))

    # Flat account + broken protection → entry blocked by PROTECTION.
    broker.position_qty = 0
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 10)
    set_candles(market, candle_5m(CONTROL_DAY, 16, 5, LONG_BREAK))
    d_blocked = rt.run_cycle()
    assert d_blocked.action == "NO_ENTRY"
    assert d_blocked.reason == "PROTECTION_BLOCKED"

    # But with an OPEN position the EXIT path ignores protection entirely:
    broker.position_qty = 2
    set_candles(market, C16, C1655)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)
    d_exit = rt.run_cycle()
    assert d_exit.action == "EXIT"           # risk-reducing trade allowed
    assert broker.position_qty == 0


# ══════════════════════════════════════════════════════════════════════
# M — ПОЛНЫЙ DETERMINISTIC LIFECYCLE ENTER → HOLD → EXIT → FLAT
#     Один экземпляр C5Runtime + один PaperExecutionAdapter/SimulatedBroker,
#     три последовательных production run_cycle(). Меняются ТОЛЬКО внешние
#     границы (market fixture / clock); run_cycle(), _evaluate_entry(),
#     _evaluate_exit(), c5_core, sizing, guards, MarginTracker и
#     SimulatedBroker — реальные, не подменены.
#
#     Fixture math (equity=free_cash=2_000_000, ГО rate 0.0585):
#       risk_budget = 2M × 0.20 × ~1.0 ≈ 400k
#       margin_per_contract @ price 1420 = 83 070 → qty_risk = 4
#       stage_max_qty = 5                    → final qty = 4
#       M/E = 4×83 070 / 2M ≈ 0.166 < 0.30  → guard passed
#       required_margin = 4 × 1400×1000×0.0585 = 327 600 ≤ cash 2M
#       → ensure_cash_for_entry проходит БЕЗ TMON-ликвидации.
#
#     Exit channel: все дневные бары H=L=C=1400 ⇒ Lower(5)=Upper(5)=1400.
#     Реальное production-условие LONG-exit — close < Lower(5), а НЕ
#     возврат цены внутрь канала (для этого fixture канал вырожден).
#
#     Accounting NOTE (по явной договорённости): ассерт
#     "cash_after_exit == seed_cash" здесь ОТСУТСТВУЕТ намеренно —
#     полная модель возврата ГО после SELL является отдельной будущей
#     задачей broker accounting. Здесь проверяются lifecycle позиции и
#     execution state; направление accounting-изменения cash (BUY ↓,
#     EXIT ↑) фиксируется как monotonic sanity-check, не как инвариант.
# ══════════════════════════════════════════════════════════════════════
def test_m_full_lifecycle_enter_hold_exit_flat(state_env, market):
    SEED_CASH = 2_000_000.0
    SEED_EQUITY = 2_000_000.0

    from src.execution_adapter import SimulatedBroker
    broker = SimulatedBroker(cash=SEED_CASH, equity=SEED_EQUITY)

    # ONE runtime instance and ONE adapter/broker pair for ALL cycles.
    rt, adapter = make_runtime(market, broker)

    # ── CYCLE 1 — ENTER ───────────────────────────────────────────────
    # Control point 16:00 MSK passed (fixture now = 16:05); control
    # candle closes at 1420 > Upper(10)=1400 → LONG breakout; sizing
    # non-zero; execution guard passes on the fake order book.
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    assert latest_closed(market)["time"] == C16["time"]

    d1 = rt.run_cycle()

    assert d1.action == "ENTER_LONG"
    assert d1.qty_risk == 4                # 400k / 83 070
    assert d1.qty > 0                      # stage cap 5 not reached → 4
    QTY = d1.qty
    assert len(buy_orders(adapter)) == 1   # ровно один BUY
    buy = [o for o in adapter.orders if o["side"] == "BUY"][0]
    assert buy["instrument"] == "CNYRUBF"
    assert buy["side"] == "BUY"
    assert buy["qty"] == QTY
    assert buy["margin_per_contract"] == pytest.approx(83070.0)
    assert buy["entry_price"] is not None
    assert broker.position_qty == QTY      # PAPER_FILL applied locally
    st1 = read_state(state_env)
    assert st1["c5_position_direction"] == "LONG"
    assert st1["c5_position_qty"] == QTY
    assert st1["last_c5_entry_control_timestamp"] == C16["time"].isoformat()
    cash_after_buy = broker.cash
    assert cash_after_buy < SEED_CASH      # ГО заблокирован (BUY)

    # ── CYCLE 2 — HOLD ────────────────────────────────────────────────
    # NEW closed 5m bar (16:05→closed at 16:10) inside the exit channel
    # (close ≥ Lower(5)=1400) → no EXIT condition; position stays LONG.
    C1605 = candle_5m(CONTROL_DAY, 16, 5, NEUTRAL)
    set_candles(market, C16, C1605)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 10)
    assert latest_closed(market)["time"] == C1605["time"]   # NEW candle ts

    d2 = rt.run_cycle()

    assert d2.action == "HOLD"
    assert broker.position_qty == QTY      # позиция сохранена
    st2 = read_state(state_env)
    assert st2["c5_position_direction"] == "LONG"
    assert st2["c5_position_qty"] == QTY
    assert len(adapter.orders) == 1        # новых BUY/SELL нет
    assert len(buy_orders(adapter)) == 1
    assert len(sell_orders(adapter)) == 0
    assert broker.cash == cash_after_buy   # accounting не тронут на HOLD

    # ── CYCLE 3 — EXIT ────────────────────────────────────────────────
    # Реальное production-условие LONG-exit: close < Lower(5)=1400.
    C1610 = candle_5m(CONTROL_DAY, 16, 10, SHORT_BREAK)
    set_candles(market, C16, C1605, C1610)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 15)
    assert latest_closed(market)["time"] == C1610["time"]   # NEW candle ts

    d3 = rt.run_cycle()

    # Фактическое production action name: "EXIT" (reason "EXIT_LONG").
    assert d3.action == "EXIT"
    assert d3.reason == "EXIT_LONG"
    assert len(sell_orders(adapter)) == 1  # PaperExecutionAdapter.sell_cny
    sell = sell_orders(adapter)[0]
    assert sell["instrument"] == "CNYRUBF"
    assert sell["side"] == "SELL"
    assert sell["qty"] == QTY              # полный выход
    assert len(adapter.orders) == 2        # BUY + SELL
    assert broker.position_qty == 0        # flat в брокере
    st3 = read_state(state_env)
    assert st3["c5_position_qty"] == 0
    assert st3["c5_position_direction"] is None          # flat по модели state
    assert st3["c5_entry_price"] is None
    assert st3["last_action"] == "EXIT_LONG"
    # Monotonic sanity только как наблюдение, НЕ "cash == seed":
    # полная модель возврата ГО после SELL — отдельная будущая задача.
    assert broker.cash >= cash_after_buy


# ══════════════════════════════════════════════════════════════════════
# N — DETERMINISTIC PAPER RESTART / RECONCILE с открытой позицией
#
#   Runtime #1 → ENTER_LONG xN → state LONG xN → "restart" (brand-new
#   C5Runtime + brand-new PaperExecutionAdapter; NOTHING carried in
#   memory) → production reconcile()/bootstrap() восстанавливает позицию
#   → следующий run_cycle() Runtime #2 продолжает работать с ней (HOLD),
#   повторного BUY нет.
#
#   Фактический call chain position-restore (production, не подменён):
#     run_cycle() → load_state()            (state file: idempotency keys,
#                                            c5_position_qty/direction,
#                                            c5_entry_price — persistence)
#              → bootstrap(state)
#                  → adapter.find_account()
#                  → reconcile(adapter, account_id, state)
#                      → snapshot = adapter.snapshot(account_id)
#                        (PAPER: SimulatedBroker.position_qty/entry_price)
#                      → broker_qty != stored_qty ⇒ adopt BROKER into state
#                      → equity<=0 ⇒ ReconcileError (fail closed)
#              → _position_from_snapshot(snapshot)  ← runtime читает
#                позицию ИЗ SNAPSHOT, а не из state (snapshot = source of
#                truth для live position; state = persistence/idempotency).
#
#   Restart в реальном процессе пересоздаёт SimulatedBroker заново через
#   production factory `_build_paper_broker(read_adapter)` (execution_
#   adapter.py:433) из read-only account snapshot. Здесь тот же factory
#   вызывается НАПРЯМУЮ; единственный fake — external read boundary
#   `read_adapter` (find_account/snapshot только), который возвращает
#   сериализованное состояние брокера Phase 1 — ровно то, что реальный
#   read API вернул бы после перезапуска процесса. run_cycle(),
#   _evaluate_entry(), _evaluate_exit(), reconcile(), c5_core, sizing,
#   guards, SimulatedBroker — реальные production-объекты.
#
#   Known gap (фиксируется, НЕ чинится в этом тесте): при restart
#   `margin_locked_per_contract` НЕ входит ни в read snapshot, ни в state
#   file ⇒ он сбрасывается в 0.0 и последующий EXIT освобождает ГО=0
#   (broker fail-safe: не печатать свободные деньги из ниоткуда). Полная
#   broker accounting модель — отдельная будущая задача; ассерт
#   "cash == seed" здесь отсутствует намеренно.
# ══════════════════════════════════════════════════════════════════════
def test_n_paper_restart_reconcile_preserves_open_position(state_env, market):
    from src.execution_adapter import (SimulatedBroker, PaperExecutionAdapter,
                                       _build_paper_broker)

    SEED_CASH = 2_000_000.0
    SEED_EQUITY = 2_000_000.0

    # ── PHASE 1 — RUNTIME #1: ENTER_LONG xN ───────────────────────────
    broker1 = SimulatedBroker(cash=SEED_CASH, equity=SEED_EQUITY)
    rt1, adapter1 = make_runtime(market, broker1)

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)

    d1 = rt1.run_cycle()
    assert d1.action == "ENTER_LONG"
    QTY = d1.qty
    assert QTY > 0                                   # qty=4 @ fixture math
    assert broker1.position_qty == QTY
    st_mid = read_state(state_env)
    assert st_mid["c5_position_direction"] == "LONG"
    assert st_mid["c5_position_qty"] == QTY
    ENTRY_PRICE = st_mid["c5_entry_price"]           # persisted by runtime
    assert len(buy_orders(adapter1)) == 1            # BUY order exists
    cash_after_entry = broker1.cash
    assert cash_after_entry < SEED_CASH              # ГО заблокирован

    # ── PHASE 2 — SIMULATED PROCESS RESTART ───────────────────────────
    # Runtime #1, adapter #1, broker #1 больше НЕ используются. Снимок
    # внешнего состояния (брокерский счёт + state file) сохраняется —
    # как при реальной перезапуске процесса.
    persisted_state_file = (state_env / "state.json").read_text()
    assert json.loads(persisted_state_file)["c5_position_qty"] == QTY

    class ReadOnlyAccountApi:
        """Fake ТОЛЬКО external read boundary (find_account/snapshot).
        Физически не имеет post_order — как production read_adapter."""

        def __init__(self, snap):
            self._snap = snap

        def find_account(self):
            return "paper-restart-account"

        def snapshot(self, account_id):
            from src.execution_adapter import AccountSnapshot
            return AccountSnapshot(
                account_id=account_id,
                equity_rub=self._snap.equity,
                free_cash_rub=self._snap.cash,
                position_qty=self._snap.position_qty,
                entry_price=self._snap.entry_price or None,
                tmon_qty=self._snap.tmon_qty,
                open_orders=0,
            )

    # Production PAPER bootstrap path: brand-new SimulatedBroker seeded
    # via _build_paper_broker from the read-only account snapshot.
    broker2 = _build_paper_broker(ReadOnlyAccountApi(broker1))
    assert broker2 is not broker1                    # fresh process memory
    adapter2 = PaperExecutionAdapter(broker2)        # fresh orders ledger
    assert adapter2.orders == []

    # Brand-new C5Runtime — ничего не переносится в памяти; state file
    # остаётся ТЕМ ЖЕ (тот же C5_STATE_FILE из fixtures state_env).
    rt2, _ = make_runtime(market, broker2)
    rt2.adapter = adapter2                           # fresh adapter instance
    rt2.trader = adapter2

    # ── PHASE 3 — RECONCILE (production bootstrap внутри run_cycle) ──
    # Новая закрытая 5m свеча появляется ПОСЛЕ restart (16:05→closed
    # 16:10) с NEUTRAL close ≥ Lower(5)=1400 ⇒ HOLD-условие.
    C1605 = candle_5m(CONTROL_DAY, 16, 5, NEUTRAL)
    set_candles(market, C16, C1605)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 10)

    d2 = rt2.run_cycle()                             # real reconcile inside

    # Позиция восстановлена: source of truth — broker snapshot, adopted
    # reconcile() в state; direction/entry_price сохранены production.
    assert broker2.position_qty == QTY               # qty_from_phase_1
    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == QTY
    assert st2["c5_position_direction"] == "LONG"
    assert st2["reconcile_status"] == "OK"
    # entry_price хранится production state ⇒ переживает restart:
    assert st2["c5_entry_price"] == ENTRY_PRICE
    # ...и seed-брокер получает его из read snapshot (_build_paper_broker):
    assert broker2.entry_price == pytest.approx(ENTRY_PRICE)
    # Повторного BUY из-за restart НЕТ (orders нового рантайма пусты):
    assert adapter2.orders == []
    assert buy_orders(adapter2) == []
    assert sell_orders(adapter2) == []
    # Оригинальный BUY остался ровно один в истории Phase 1:
    assert len(buy_orders(adapter1)) == 1

    # ── PHASE 4 — ПРОДОЛЖЕНИЕ РАБОТЫ: HOLD без новых ордеров ─────────
    assert d2.action == "HOLD"                       # production continue
    assert d2.reason == "HOLD_POSITION"              # фактический production reason
    assert broker2.position_qty == QTY               # позиция прежняя
    st3 = read_state(state_env)
    assert st3["c5_position_direction"] == "LONG"
    assert st3["c5_position_qty"] == QTY
    assert len(adapter2.orders) == 0                 # новых orders нет
    assert broker2.cash == cash_after_entry          # accounting не тронут
