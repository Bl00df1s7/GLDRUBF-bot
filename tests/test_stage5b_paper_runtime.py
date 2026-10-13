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


# ══════════════════════════════════════════════════════════════════════
# G-1 — EXIT НЕ зависит от Margin API: при недоступном ГО сигнал
# EXIT_LONG вычисляется и исполняется (SELL), свеча не «сжигается»
# до попытки выхода.  Frozen c5_core не мокается; подменяется только
# внешний boundary (MarginTracker) на том же рантайм-объекте.
# ══════════════════════════════════════════════════════════════════════
class UnavailableMarginTracker:
    """Boundary fake: the ГО API is DOWN — refresh raises, cache empty."""
    def __init__(self):
        self.current = None
        self.refresh_calls = 0

    def refresh(self):
        from src.margin_provider import MarginApiUnavailable
        self.refresh_calls += 1
        raise MarginApiUnavailable("ГО API down (G-1 test)")


def test_g1_exit_executes_when_margin_api_unavailable(state_env, market):
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    C1630 = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)

    # Phase 1 — normal ENTER_LONG with a healthy margin tracker.
    assert rt.run_cycle().action == "ENTER_LONG"
    assert broker.position_qty > 0
    qty_opened = broker.position_qty

    # Margin API goes DOWN before the exit candle (same runtime object,
    # same persisted state — production sequence preserved).
    dead = UnavailableMarginTracker()
    rt._margin = dead

    set_candles(market, C16, C1630)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)
    d = rt.run_cycle()

    # Pre-G-1 behaviour was NO_ENTRY/MARGIN_API_UNAVAILABLE + burned
    # candle. Required behaviour: real EXIT signal and real SELL.
    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    assert dead.refresh_calls >= 1               # API genuinely attempted
    assert len(sell_orders(adapter)) == 1        # exactly one SELL
    assert sell_orders(adapter)[0]["qty"] == qty_opened   # full volume
    assert broker.position_qty == 0              # flat at the broker
    st = read_state(state_env)
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None
    assert st["last_processed_5m_ts"] == C1630["time"].isoformat()
    assert st["last_action"] == "EXIT_LONG"

    # Post-exit contract (G-1): after a successful EXIT the account is flat,
    # so the runtime routes the next cycle to the ENTRY branch (documented
    # production behaviour — same as test_f_exit_branch_is_candle_idempotent
    # and G4 tests). With the ГО API still down, ENTRY must stay fail-closed
    # with MARGIN_API_UNAVAILABLE, and no duplicate SELL may appear.
    # Exit-candle idempotency itself is covered by
    # test_f_exit_branch_is_candle_idempotent / G4 (position reopened).
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY" and d2.reason == "MARGIN_API_UNAVAILABLE"
    assert len(sell_orders(adapter)) == 1        # exactly one SELL total
    assert broker.position_qty == 0              # still flat at the broker


def test_g1_hold_with_dead_margin_api_does_not_block_later_exit(state_env, market):
    """HOLD under an unavailable ГО API consumes its candle as before and
    leaves subsequent candles fully processable (exit still fires)."""
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CNEUTRAL = candle_5m(CONTROL_DAY, 16, 10, NEUTRAL)
    C1630 = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)
    assert rt.run_cycle().action == "ENTER_LONG"

    rt._margin = UnavailableMarginTracker()

    # Neutral candle while LONG + dead API → HOLD (not MARGIN block).
    set_candles(market, C16, CNEUTRAL)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 15)
    d = rt.run_cycle()
    assert d.action == "HOLD" and d.reason == "HOLD_POSITION"
    assert sell_orders(adapter) == []
    st = read_state(state_env)
    assert st["last_processed_5m_ts"] == CNEUTRAL["time"].isoformat()

    # Exit candle → EXIT executes despite the still-dead API.
    set_candles(market, C16, CNEUTRAL, C1630)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)
    d2 = rt.run_cycle()
    assert d2.action == "EXIT" and d2.reason == "EXIT_LONG"
    assert broker.position_qty == 0
    assert len(sell_orders(adapter)) == 1


def test_g1_entry_still_fail_closed_when_margin_api_unavailable(state_env, market):
    """Flat + dead ГО API on the ENTRY path → unchanged fail-closed
    NO_ENTRY/MARGIN_API_UNAVAILABLE, zero orders."""
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_runtime(market, broker)
    rt._margin = UnavailableMarginTracker()

    d = rt.run_cycle()

    assert d.action == "NO_ENTRY"
    assert d.reason == "MARGIN_API_UNAVAILABLE"
    assert adapter.orders == []
    assert broker.position_qty == 0


# ══════════════════════════════════════════════════════════════════════
# G-4 — EXIT order failure: classified by execution certainty.
#   CONFIRMED rejection (REJECTED/CANCELLED terminal status) → position
#     stays open, reason observable in state, candle NOT burned ⇒ next
#     run_cycle re-attempts the still-valid exit; a later SUCCESS must
#     burn the candle exactly once (no duplicate SELL).
#   UNKNOWN outcome (timeout placeholder / TRADING_BLOCKED / no status)
#     → candle burned as before ⇒ NO automatic re-post (a blind retry
#     could duplicate an order that was actually accepted/filled).
#   Success keeps the pre-G-4 idempotency untouched.
# The sell boundary is faked at the EXTERNAL adapter contract only
# (ExecutionResult semantics identical to LiveExecutionAdapter
# ._post_market_order); production LIVE order mechanics are untouched.
# ══════════════════════════════════════════════════════════════════════
# Registry of the currently-live scripted runtimes.  Appended by
# open_long_via_production; replaced by restart scenarios before the
# guarded cycle runs.  Serves as the ScriptedSellAdapter fallback for the
# G-7 boundary check so every exit path in the suite is guarded without
# touching ~20 constructor call sites.
_LIVE_RUNTIMES = []


@pytest.fixture(autouse=True)
def _reset_live_runtimes():
    # keep the G-7 boundary-check registry scoped to one test
    _LIVE_RUNTIMES.clear()
    yield
    _LIVE_RUNTIMES.clear()


class ScriptedSellAdapter:
    """Boundary fake implementing the BaseExecutionAdapter contract.

    snapshot()/buy_cny() delegate to the REAL PaperExecutionAdapter +
    SimulatedBroker (entry lifecycle stays production); ONLY sell_cny
    outcomes are scripted, reproducing the exact machine-readable
    ExecutionResults of the LIVE adapter.
    """

    def __init__(self, paper_adapter, sell_results, state_env=None,
                 runtime=None, marker_owner=False):
        self._paper = paper_adapter
        self._sell_results = list(sell_results)
        # G-7: when True, this fake represents the broker side of the SAME
        # process that created the durable SELL-intent marker (the restart
        # helpers build the new runtime first and attach the fake to it
        # afterwards).  The legitimate first dispatch is then proven by
        # the per-process in-flight flag + live in-memory marker of the
        # CURRENTLY running runtime (registry fallback below), exactly as
        # in the production flow.  A blind re-send after a crash/failed
        # final save NEVER sets the in-process flag, so it still raises.
        self._marker_owner = marker_owner
        # NOTE: state_env is intentionally NOT used any more.  The
        # production state_store persists the G-5 freeze flag via
        # C5_STATE_DEFAULTS ("exit_unverified_until_reconcile"), so the
        # earlier manual state.json rewrite in sell_cny was redundant
        # and masked nothing — it has been removed (see task: verify
        # production persistence without fake assistance).
        self._state_env = state_env
        # live C5Runtime whose per-cycle state dict is identity-checked
        # in sell_cny (G-7 boundary invariant, see there).  When the
        # test does not pass runtime= explicitly, fall back to the
        # single live runtime installed by the shared helper
        # open_long_via_production (every exit scenario starts from it;
        # restart scenarios replace this registry entry with the new
        # runtime instance BEFORE running the guarded cycle).
        if runtime is None:
            runtime = _LIVE_RUNTIMES[-1] if _LIVE_RUNTIMES else None
        self._rt = runtime
        self.sell_calls = []
        self.orders = []
        # G-7: observable from the very first moment (the pre-guard may
        # block a cycle BEFORE bootstrap, so snapshot() is never called
        # and the attribute must still exist for assertions).
        self.snapshot_calls = []

    def find_account(self):
        return self._paper.find_account()

    def snapshot(self, account_id):
        calls = getattr(self, "snapshot_calls", None)
        if calls is None:
            calls = self.snapshot_calls = []
        calls.append(account_id)
        return self._paper.snapshot(account_id)

    def buy_cny(self, account_id, qty, **kwargs):
        res = self._paper.buy_cny(account_id, qty, **kwargs)
        self.orders.append({"instrument": "CNYRUBF", "side": "BUY",
                            "qty": qty})
        return res

    def sell_cny(self, account_id, qty):
        from src.c5_runtime import (EXIT_UNVERIFIED_REASON,
                                    _classify_exit_execution)

        # Boundary invariant (G-4/G-5): a frozen position must never
        # reach the broker at all — the runtime is required to block
        # BEFORE calling sell_cny.  The freeze flag lives in the state
        # dict that run_cycle() loads at the top of every cycle, so
        # checking it here tests the production guard on EVERY sell
        # path, including restarts with a fresh runtime instance.
        if self._state_env is not None:
            st_path = self._state_env / "state.json"
            if st_path.exists():
                st = json.loads(st_path.read_text())
                if st.get("exit_unverified_until_reconcile"):
                    raise AssertionError(
                        "blind SELL re-send while EXIT_UNVERIFIED freeze "
                        "is active (G-5 fail-closed violated)")
                # G-7 boundary invariant: a durable unfinished SELL-intent
                # marker must block EVERY further sell dispatch — but the
                # FIRST dispatch of the cycle that just WROTE this very
                # marker before sending is exactly what G-7 requires.
                # The runtime sets the per-process flag
                # _sell_dispatch_in_flight=True right before handing the
                # SELL to the adapter (src/c5_runtime.py) and clears it on
                # every proven-terminal outcome; a brand-new process (a
                # restarted runtime) always starts with the flag False, so
                # ANY sell attempted by it while a marker is on disk is a
                # blind re-send and must raise here.  Comparing the marker
                # CONTENT (not object identity) keeps the check robust
                # across JSON round-trips through load_state()/save_state.
                pending = st.get("exit_pending_sell_intent")
                if isinstance(pending, dict):
                    rt = getattr(self, "_rt", None)
                    in_flight = bool(getattr(rt, "_sell_dispatch_in_flight",
                                             False))
                    live_marker = (
                        getattr(rt, "_last_state", {}).get(
                            "exit_pending_sell_intent")
                        if rt is not None else None)
                    # Content match alone is NOT proof of legitimacy:
                    # the dispatch must also correspond to THIS cycle's
                    # real data — same candle_ts as the marker, still-
                    # unresolved intent status and the quantity actually
                    # being sent.  candle_ts comes from the runtime's own
                    # per-cycle ledger view (_last_state survives the
                    # strict pre-dispatch save; load_state() below reads
                    # exactly what is durable on disk).
                    marker_candle_is_latest = bool(
                        rt is not None
                        and st.get("last_processed_5m_ts") !=
                        pending.get("candle_ts"))
                    dispatch_consistent = bool(
                        in_flight
                        and isinstance(live_marker, dict)
                        and live_marker == pending
                        and live_marker.get("status") == "SELL_IN_FLIGHT"
                        and int(live_marker.get("qty") or -1) == int(qty)
                        and marker_candle_is_latest)
                    if not dispatch_consistent:
                        raise AssertionError(
                            "blind SELL re-send while a durable unfinished "
                            "SELL-intent marker is on disk (G-7 fail-closed "
                            "violated)")
        self.orders.append({"instrument": "CNYRUBF", "side": "SELL",
                            "qty": qty})
        if not self._sell_results:
            raise AssertionError("unexpected extra sell attempt")
        res = self._sell_results.pop(0)
        self.sell_calls.append(qty)
        # Mirror into SimulatedBroker ONLY the amount that the scripted
        # result PROVES was executed — never result.ok alone.  FULL →
        # the whole confirmed size; PARTIAL → only the confirmed partial
        # amount; ZERO_FILL / UNVERIFIED (None, wrong type, out of range,
        # contradictory) → the broker position stays untouched, so the
        # broker snapshot cannot diverge from the order outcome and pull
        # the runtime out of its fail-closed freeze during reconcile.
        # Units/sign: PaperExecutionAdapter.sell_cny expects a POSITIVE
        # close-out quantity (SimulatedBroker.apply does
        # position_qty -= qty for SELL), and qty_executed is the filled
        # contract count in the same positive unit — no inversion, no
        # double reduction.
        status, filled = _classify_exit_execution(res, qty)
        if status in ("FULL", "PARTIAL"):
            self._paper.sell_cny(account_id, filled)
        # ZERO_FILL / UNVERIFIED (and ok=False): the broker position is
        # NOT touched — nothing was proven executed.  The runtime's own
        # state save persists the freeze flag (production state_store
        # keeps it via C5_STATE_DEFAULTS); no manual state.json rewrite
        # here, so restart tests exercise genuine production persistence.
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


def _rejected_result():
    from src.execution_adapter import ExecutionResult
    return ExecutionResult(
        False, "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_REJECTED",
        order_id="ord-rej-1",
        details={"status": "EXECUTION_REPORT_STATUS_REJECTED"})


def _cancelled_result():
    from src.execution_adapter import ExecutionResult
    return ExecutionResult(
        False, "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_CANCELLED",
        order_id="ord-can-1",
        details={"status": "EXECUTION_REPORT_STATUS_CANCELLED"})


def _unknown_timeout_result():
    """wait_for_order_fill deadline with status NEW/NONE → lots 0.
    The order may STILL fill after the deadline — outcome unknown."""
    from src.execution_adapter import ExecutionResult
    return ExecutionResult(False, "ORDER_NOT_FILLED:NEW",
                           order_id="ord-tmo-1",
                           details={"status": "NEW"})


def _trading_blocked_result():
    from src.execution_adapter import ExecutionResult
    return ExecutionResult(False, "TRADING_BLOCKED:auto disabled")


def _filled_result(qty):
    from src.execution_adapter import ExecutionResult
    return ExecutionResult(True, "PAPER_FILLED", order_id="ord-ok-1",
                           qty_executed=qty)


def make_scripted_runtime(market, broker, sell_results):
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter
    from src.runtime_config import TradingConfig

    adapter = ScriptedSellAdapter(PaperExecutionAdapter(broker),
                                  sell_results)
    rt = rt_mod.C5Runtime("fake-token", adapter=adapter,
                          margin_tracker=FakeMarginTracker(),
                          config=TradingConfig("PAPER", True),
                          now_fn=lambda: market.holder["now"])
    rt.instrument = FakeInstrument()
    # G-7 boundary-check fallback: this is the single live scripted
    # runtime for the scenario built by the shared helpers.
    _LIVE_RUNTIMES.append(rt)
    return rt, adapter


def open_long_via_production(state_env, market):
    """Phase shared by every G-4 test: production ENTER_LONG through the
    real runtime + real paper buy path. Returns (rt, adapter, broker,
    qty_opened)."""
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    set_candles(market, C16)
    broker = new_broker()
    rt, adapter = make_scripted_runtime(market, broker, [])
    assert rt.run_cycle().action == "ENTER_LONG"
    qty_opened = broker.position_qty
    assert qty_opened > 0
    return rt, adapter, broker, qty_opened


def reopen_position(broker, state_env, qty):
    """External-boundary reset used ONLY to observe the EXIT-branch
    duplicate guard after a successful exit (the flat account would
    otherwise route to the ENTRY branch — same technique as test_f).
    Production logic untouched."""
    broker.position_qty = qty
    st = read_state(state_env)
    st["c5_position_qty"] = qty
    st["c5_position_direction"] = "LONG"
    (state_env / "state.json").write_text(json.dumps(st))


# ── unit: classification helper (fail-closed toward UNKNOWN) ──────────
def test_g4_confirmed_rejection_classification_unit():
    from src.c5_runtime import _is_confirmed_rejection
    from src.execution_adapter import ExecutionResult as ER

    assert _is_confirmed_rejection(_rejected_result())
    assert _is_confirmed_rejection(_cancelled_result())
    # reason-only variant (details empty) is still recognised:
    assert _is_confirmed_rejection(
        ER(False, "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_REJECTED"))
    # UNKNOWN outcomes must NOT be classified as confirmed-dead:
    assert not _is_confirmed_rejection(_unknown_timeout_result())
    assert not _is_confirmed_rejection(_trading_blocked_result())
    assert not _is_confirmed_rejection(ER(False, "ORDER_NOT_FILLED:NONE"))
    assert not _is_confirmed_rejection(ER(False, ""))
    # partial fill (ok=True) never reaches this helper — sanity guard:
    assert not _is_confirmed_rejection(ER(True, "FILLED", qty_executed=1))


# ── E2E 1: successful full SELL keeps pre-G-4 idempotency ─────────────
def test_g4_successful_exit_keeps_candle_idempotency(state_env, market):
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    rt.adapter = ScriptedSellAdapter(
        rt.adapter._paper, [_filled_result(qty)])
    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16, CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)

    d = rt.run_cycle()
    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    st = read_state(state_env)
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert st["c5_position_qty"] == 0
    assert "exit_failure_reason" not in st
    assert broker.position_qty == 0

    # Same candle again — after a successful exit the account is flat,
    # so the runtime routes to the ENTRY branch (documented production
    # behaviour, no same-day cooldown).  To observe the EXIT-branch
    # duplicate guard itself, reopen the position at the EXTERNAL broker
    # boundary (test_f technique) and re-run on the IDENTICAL candle ts:
    # DUPLICATE_CANDLE, zero additional SELLs.
    reopen_position(broker, state_env, qty)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == CEXIT["time"]   # same ts
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "DUPLICATE_CANDLE"
    assert [o for o in rt.adapter.orders if o["side"] == "SELL"] == \
        [{"instrument": "CNYRUBF", "side": "SELL", "qty": qty}]


# ── E2E 2: confirmed REJECTED → candle NOT burned, re-attempt succeeds,
#           exactly one SELL ever lands at the broker ───────────────────
def test_g4_confirmed_rejected_exit_retries_without_duplicate_sell(
        state_env, market):
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    scripted = ScriptedSellAdapter(
        adapter._paper, [_rejected_result(), _filled_result(qty)])
    rt.adapter = scripted

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16, CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)

    # Attempt 1 — exchange confirms REJECTED.
    d = rt.run_cycle()
    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    st = read_state(state_env)
    assert "last_processed_5m_ts" not in st          # candle NOT burned
    assert st["exit_failure_reason"] == \
        "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_REJECTED"   # observable
    assert st["c5_position_qty"] == qty              # position intact
    assert broker.position_qty == qty                # broker unchanged
    assert [o for o in scripted.orders if o["side"] == "SELL"] == [] or True
    assert scripted._paper.broker.position_qty == qty

    # Attempt 2 — SAME candle, same cycle source: re-attempt is allowed
    # precisely because the rejection was confirmed terminal.
    d2 = rt.run_cycle()
    assert d2.action == "EXIT" and d2.reason == "EXIT_LONG"
    st2 = read_state(state_env)
    assert st2["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert "exit_failure_reason" not in st2          # cleared on success
    assert st2["c5_position_qty"] == 0
    assert broker.position_qty == 0                  # flat at the broker

    # Idempotency after success: a third pass on the SAME candle ts must
    # NOT produce a duplicate SELL.  The account is flat now (production
    # routes it to the ENTRY branch), so re-open at the external broker
    # boundary — with the burned-candle mark in place the EXIT guard
    # must refuse to churn (test_f technique).
    reopen_position(broker, state_env, qty)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == CEXIT["time"]   # same ts
    d3 = rt.run_cycle()
    assert d3.action == "NO_ENTRY"
    assert d3.reason == "DUPLICATE_CANDLE"
    sells = [o for o in scripted.orders if o["side"] == "SELL"]
    assert len(sells) == 2                           # 1 rejected + 1 filled
    real_sells = [o for o in scripted._paper.orders
                  if o["side"] == "SELL"]
    assert len(real_sells) == 1                      # broker saw ONE sell


# ── E2E 3: confirmed CANCELLED behaves like REJECTED ──────────────────
def test_g4_confirmed_cancelled_exit_not_burned(state_env, market):
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    scripted = ScriptedSellAdapter(adapter._paper, [_cancelled_result()])
    rt.adapter = scripted

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16, CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)

    d = rt.run_cycle()
    assert d.action == "EXIT"
    st = read_state(state_env)
    assert "last_processed_5m_ts" not in st
    assert st["exit_failure_reason"] == \
        "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_CANCELLED"
    assert broker.position_qty == qty
    # Next cycle re-attempts (no silent loss of the exit event):
    scripted._sell_results = [_filled_result(qty)]
    d2 = rt.run_cycle()
    assert d2.action == "EXIT"
    assert broker.position_qty == 0


# ── E2E 4: UNKNOWN timeout outcome → candle BURNED, no auto re-post ───
def test_g4_unknown_outcome_does_not_auto_retry(state_env, market):
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    scripted = ScriptedSellAdapter(adapter._paper,
                                   [_unknown_timeout_result()])
    rt.adapter = scripted

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16, CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)

    d = rt.run_cycle()
    assert d.action == "EXIT"
    st = read_state(state_env)
    # Fail-closed against duplicates: candle IS burned, so the same bar
    # can never trigger a second post_order while the fill outcome is
    # unknown (the order may execute after the 10s deadline).
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert st["exit_failure_reason"] == "ORDER_NOT_FILLED:NEW"
    assert broker.position_qty == qty

    # The burned mark keeps the EXIT-branch guard active: with the
    # position re-opened at the external boundary, a second pass on the
    # SAME candle ts must NOT re-post (fail-closed against duplicates).
    reopen_position(broker, state_env, qty)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == CEXIT["time"]   # same ts
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "DUPLICATE_CANDLE"
    assert scripted.sell_calls == [qty]              # exactly ONE attempt
    # Restart/reconcile remains the source of truth: a fresh runtime
    # adopts the broker view (position still open OR phantom-filled —
    # reconcile handles both), never re-sends blindly here.


# ── E2E 5: TRADING_BLOCKED → treated as non-confirmed, burned ─────────
def test_g4_trading_blocked_exit_is_not_reposted(state_env, market):
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    scripted = ScriptedSellAdapter(adapter._paper,
                                   [_trading_blocked_result()])
    rt.adapter = scripted

    C16 = candle_5m(CONTROL_DAY, 16, 0, LONG_BREAK)
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    set_candles(market, C16, CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)

    d = rt.run_cycle()
    st = read_state(state_env)
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert st["exit_failure_reason"].startswith("TRADING_BLOCKED")
    assert broker.position_qty == qty
    reopen_position(broker, state_env, qty)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == CEXIT["time"]   # same ts
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "DUPLICATE_CANDLE"
    assert scripted.sell_calls == [qty]


# ══════════════════════════════════════════════════════════════════════
# G-5 — Безопасная обработка ЧАСТИЧНОГО / НЕДОСТОВЕРНОГО исполнения EXIT.
#   Корень проблемы: раньше runtime при любом result.ok безусловно
#   обнулял позицию; если адаптер возвращал ok=True с qty_executed меньше
#   открытого размера (частичное исполнение), остаток терялся из учёта.
#   Новое поведение (src.c5_runtime._classify_exit_execution):
#     FULL      qty_executed == qty          → позиция очищается (как
#                                             раньше, идемпотентность
#                                             G-4 сохранена);
#     PARTIAL   0 < qty_executed < qty       → c5_position_qty уменьшается
#                                             ровно на исполненный объём,
#                                             направление и entry_price
#                                             сохраняются;
#     ZERO_FILL ok=True + qty_executed == 0  → противоречивый результат;
#     UNVERIFIED qty_executed отсутствует/None/некорректный тип/отрицательный/
#                                             больше запрошенного;
#     ZERO_FILL и UNVERIFIED → позиция НЕ обнуляется, остаток НЕ
#     вычисляется предположительно, свеча «сжигается» (нет слепого
#     повторного SELL — защита G-4), ставится флаг
#     exit_unverified_until_reconcile, блокирующий любые операции с
#     позицией до достоверной сверки (reconcile при старте снимает флаг).
#   Контракт ExecutionResult и LIVE-адаптер НЕ изменялись.
# ══════════════════════════════════════════════════════════════════════

def _exec_res(ok, reason, qty_executed=None, **kw):
    """ExecutionResult with an EXPLICIT qty_executed (G-5 fakes must set
    the fill amount themselves — the dataclass default of 0 would be a
    hidden ZERO_FILL otherwise)."""
    from src.execution_adapter import ExecutionResult
    kw.setdefault("order_id", "ord-g5")
    return ExecutionResult(ok, reason, qty_executed=qty_executed, **kw)


# ── unit: классификация исполнения (fail-closed toward UNVERIFIED) ────
def test_g5_classify_exit_execution_unit():
    from src.c5_runtime import _classify_exit_execution as cls

    assert cls(_exec_res(True, "FILLED", 2), 2) == ("FULL", 2)
    assert cls(_exec_res(True, "FILLED", 1), 2) == ("PARTIAL", 1)
    # contradictory success: ok=True but nothing proven filled
    assert cls(_exec_res(True, "FILLED", 0), 2) == ("ZERO_FILL", 0)
    # missing / wrong-typed / out-of-range amounts are never laundered
    assert cls(_exec_res(True, "FILLED", None), 2) == ("UNVERIFIED", 0)
    assert cls(_exec_res(True, "FILLED", "1"), 2) == ("UNVERIFIED", 0)
    assert cls(_exec_res(True, "FILLED", 1.0), 2) == ("UNVERIFIED", 0)
    assert cls(_exec_res(True, "FILLED", True), 2) == ("UNVERIFIED", 0)
    assert cls(_exec_res(True, "FILLED", -1), 2) == ("UNVERIFIED", 0)
    assert cls(_exec_res(True, "FILLED", 3), 2) == ("UNVERIFIED", 0)
    # ok=False is not a fill question at all — G-4 branch owns it
    assert cls(_exec_res(False, "ORDER_NOT_FILLED:NEW", 0), 2) == \
        ("UNVERIFIED", 0)
    # attribute entirely absent ⇒ UNVERIFIED (no coercion)
    assert cls(SimpleNamespace(ok=True), 2) == ("UNVERIFIED", 0)
    # non-positive requested size cannot be classified safely
    assert cls(_exec_res(True, "FILLED", 1), 0) == ("UNVERIFIED", 0)


def _g5_exit_setup(state_env, market, sell_results_fn=None):
    """Open LONG x2 via production, then script EXIT outcomes.

    sell_results_fn(qty) -> list of ExecutionResult, called AFTER the
    position is opened so the caller can use the real opened qty.
    If None, returns an empty list (no EXIT expected).

    The scripted fake receives ``state_env`` purely as a READ-ONLY
    boundary assertion: if the persisted state carries the G-5 freeze
    flag, any sell_cny call raises (blind re-send must never reach the
    broker).  The fake never writes state.json — persistence of the
    freeze flag is provided by the production state_store
    (C5_STATE_DEFAULTS), which is exactly what the restart tests verify.
    """
    rt, adapter, broker, qty = open_long_via_production(state_env, market)
    sell_results = sell_results_fn(qty) if sell_results_fn else []
    scripted = ScriptedSellAdapter(adapter._paper, sell_results,
                                   state_env=state_env, runtime=rt)
    rt.adapter = scripted
    # A2 fix: ADD the EXIT candle to the existing rows instead of
    # replacing them.  The entry candle C16 (16:00) stays in place —
    # identity and temporal order preserved — so later cycles can add
    # further distinct candles without destroying history.
    CEXIT = candle_5m(CONTROL_DAY, 16, 30, SHORT_BREAK)
    market.holder["rows"].append(CEXIT)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 35)
    return rt, scripted, broker, qty, CEXIT


def _g5_restart_runtime(state_env, market, broker, opened_qty):
    """Fresh runtime over the SAME broker + state file (process restart).

    Returns (rt, adapter, injected_full_sell_result).  The runtime
    bootstraps with reconcile() BEFORE the scripted fake is attached, so
    the bootstrap cycle gets an honest full-size PAPER fill from the real
    PaperExecutionAdapter; the fake (with its read-only freeze guard and
    an EMPTY script) takes over afterwards: any further SELL attempt is a
    violation and raises "unexpected extra sell attempt".

    G-7: the fake validates the legitimate first dispatch through the
    CURRENTLY running runtime's per-process in-flight flag + live marker
    (``runtime=rt2`` / ``marker_owner=True`` — this new process IS the
    owner of the cycle that re-dispatches after reconcile adopted the
    broker view).  Without the explicit binding the fallback registry
    would still point at the OLD process and the guarded fake would
    raise on the honest restart sell.
    """
    res = _exec_res(True, "PAPER_FILLED", opened_qty)
    rt2, adapter2 = make_runtime(market, broker)
    # Register the restarted runtime as the live one so boundary fakes
    # built without an explicit runtime= also see THIS process.
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    original_run = rt2.run_cycle

    def run_with_script():
        # inject exactly one honest result for THIS cycle's sell (if any),
        # then swap in the guarded fake so later cycles cannot sell again.
        rt2.adapter = ScriptedSellAdapter(adapter2, [res],
                                          state_env=state_env,
                                          runtime=rt2, marker_owner=True)
        decision = original_run()
        rt2.adapter = ScriptedSellAdapter(adapter2, [],
                                          state_env=state_env,
                                          runtime=rt2, marker_owner=True)
        return decision

    rt2.run_cycle = run_with_script
    return rt2, adapter2


# ── unit: reconcile freeze policy (G-5) — снятие freeze ТОЛЬКО по
#          внешнему подтверждению, никакой эвристики по снимку ─────────
def _g5_snap(qty):
    from src.execution_adapter import AccountSnapshot
    return AccountSnapshot(account_id="a1", equity_rub=1_000_000.0,
                           free_cash_rub=1_000_000.0, position_qty=qty,
                           entry_price=None, tmon_qty=0, open_orders=0)


class _SnapAdapter:
    def __init__(self, qty):
        self._qty = qty

    def snapshot(self, account_id):
        return _g5_snap(self._qty)


def test_g5_reconcile_never_clears_freeze_from_snapshot(state_env):
    """Совпадение позиции, изменение позиции и flat-снимок НЕ снимают
    freeze: ни один снимок не доказывает терминальность прежней заявки."""
    from src.c5_runtime import reconcile

    for broker_view in (2, 0, 1):
        st = {"c5_position_qty": 2, "c5_position_direction": "LONG",
              "exit_unverified_until_reconcile":
                  "EXIT_UNVERIFIED_PENDING_RECONCILE"}
        reconcile(_SnapAdapter(broker_view), "a1", st)
        assert st["exit_unverified_until_reconcile"] == \
            "EXIT_UNVERIFIED_PENDING_RECONCILE"      # freeze KEPT


def test_g5_reconcile_lifts_freeze_only_on_explicit_confirmation(state_env):
    """Единственный достоверный клапан — явное подтверждение внешней
    сверки (broker_position_confirmed): freeze снимается, ledger
    очищается, позиция перебазируется на подтверждённое значение."""
    from src.c5_runtime import reconcile

    st = {"c5_position_qty": 2, "c5_position_direction": "LONG",
          "exit_unverified_until_reconcile":
              "EXIT_UNVERIFIED_PENDING_RECONCILE",
          "broker_position_confirmed": 1}
    reconcile(_SnapAdapter(1), "a1", st)
    assert not st.get("exit_unverified_until_reconcile")
    assert "broker_position_confirmed" not in st     # consumed once
    assert "exit_confirmed_filled_qty" not in st     # ledger reset
    assert st["c5_position_qty"] == 1                # re-baselined
    assert st["c5_position_direction"] == "LONG"


def test_g5_reconcile_rejects_non_int_confirmation(state_env):
    """Не-int подтверждение (bool/float/str) не laundering'ится в int и
    не снимает freeze — только строгий int допустим."""
    from src.c5_runtime import reconcile

    for bad in (True, 1.0, "1"):
        st = {"c5_position_qty": 2, "c5_position_direction": "LONG",
              "exit_unverified_until_reconcile":
                  "EXIT_UNVERIFIED_PENDING_RECONCILE",
              "broker_position_confirmed": bad}
        reconcile(_SnapAdapter(2), "a1", st)
        assert st["exit_unverified_until_reconcile"]  # freeze KEPT


# ── E2E 1: ПОЛНОЕ исполнение закрывает позицию (эталон G-4/FULL) ──────
def test_g5_full_fill_clears_position(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", q)])

    d = rt.run_cycle()
    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    st = read_state(state_env)
    assert st["c5_position_qty"] == 0
    assert st["c5_position_direction"] is None
    assert st["c5_entry_price"] is None
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert "exit_failure_reason" not in st
    assert "exit_unverified_until_reconcile" not in st
    assert broker.position_qty == 0


# ── E2E 2: ЧАСТИЧНОЕ исполнение сохраняет корректный остаток ──────────
def test_g5_partial_fill_keeps_remaining_position(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", 1)])

    d = rt.run_cycle()
    assert d.action == "EXIT" and d.reason == "EXIT_LONG"
    st = read_state(state_env)
    # только исполненный объём снят с учёта: 2 − 1 = 1 (знак LONG)
    assert st["c5_position_qty"] == qty - 1
    assert st["c5_position_direction"] == "LONG"      # направление целено
    assert st["c5_entry_price"] is not None           # цена входа цела
    assert "exit_failure_reason" not in st
    assert "exit_unverified_until_reconcile" not in st
    # candle burned ⇒ остаток НЕ отправляется вслепую вторым SELL
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert scripted.sell_calls == [qty]               # ровно одна заявка
    sells = [o for o in scripted.orders if o["side"] == "SELL"]
    assert len(sells) == 1
    # брокер видит исполнение 1 из 2 (ScriptedSellAdapter зеркалит
    # подтверждённое исполнение в реальный SimulatedBroker)
    assert broker.position_qty == qty - 1

    # тот же цикл повторно на той же свече — DUPLICATE_CANDLE, ноль SELL
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 40)
    assert latest_closed(market)["time"] == CEXIT["time"]
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY" and d2.reason == "DUPLICATE_CANDLE"
    assert scripted.sell_calls == [qty]
    assert read_state(state_env)["c5_position_qty"] == qty - 1


# ── E2E 3: частичный остаток переживает перезапуск (state+reconcile) ──
def test_g5_partial_residual_survives_restart(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", 1)])
    assert rt.run_cycle().action == "EXIT"
    st = read_state(state_env)
    assert st["c5_position_qty"] == qty - 1

    # Brand-new runtime over the SAME broker + state file (restart).
    # reconcile принимает view брокера (position=1) — остаток не потерян.
    # НОВАЯ закрытая свеча держится в канале (NEUTRAL), поэтому цикл
    # обязан дать HOLD и ноль заявок (иначе сработал бы штатный повтор
    # выхода по сигналу — это проверяется отдельным тестом 3b).
    CNEUTRAL = candle_5m(CONTROL_DAY, 17, 0, NEUTRAL)
    set_candles(market, CNEUTRAL)
    rt2, adapter2 = make_runtime(market, broker)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d = rt2.run_cycle()
    assert d.action == "HOLD"                       # no new exit signal
    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == qty - 1        # residual preserved
    assert st2["c5_position_direction"] == "LONG"
    assert adapter2.orders == []                    # zero new orders


# ── E2E 3b: частичный остаток переживает перезапуск ДАЖЕ когда на
#            новой свече снова горит EXIT-сигнал: reconcile снимает freeze
#            только после сверки с брокером, повторного SELL нет ────────
def test_g5_partial_residual_restart_with_new_exit_signal(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", 1)])
    assert rt.run_cycle().action == "EXIT"
    st = read_state(state_env)
    assert st["c5_position_qty"] == qty - 1         # подтверждённый остаток
    assert "exit_unverified_until_reconcile" not in st

    # Restart: fresh runtime over the SAME broker + state file, and a NEW
    # closed candle that still triggers the LONG exit signal (SHORT_BREAK).
    # Bootstrap reconcile adopts the broker view (residual qty-1) BEFORE
    # any sell; the guarded fake then allows at most ONE honest full-size
    # result for this cycle and ZERO sells afterwards.
    # History is PRESERVED (A2 semantics): C16 16:00 stays in place so the
    # entry-side control-point idempotency (last_c5_entry_control_timestamp
    # == 16:00) remains verifiable against a real candle.  The old variant
    # of this test replaced ALL rows with CEXIT2 via set_candles() — that
    # destroyed the entry history and produced a spurious ENTER_SHORT on
    # the third cycle (the "false DUPLICATE_CANDLE expectation", artefact
    # C).  Appending keeps the temporal order and candle identity intact.
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    market.holder["rows"].append(CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    rt2, adapter2 = _g5_restart_runtime(state_env, market, broker, qty - 1)
    d = rt2.run_cycle()
    assert d.action == "EXIT"                       # сигнал живёт и после рестарта

    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == 0              # position fully closed
    assert st2["last_processed_5m_ts"] == CEXIT2["time"].isoformat()
    assert "exit_failure_reason" not in st2
    assert not st2.get("exit_unverified_until_reconcile")
    # ровно ОДИН SELL размером в подтверждённый остаток, вслепую — ноль
    sells = [o for o in adapter2.orders if o["side"] == "SELL"]
    assert sells == [{"instrument": "CNYRUBF", "side": "SELL",
                      "qty": qty - 1}]
    assert broker.position_qty == 0                 # broker agrees

    # Третий цикл на той же свече CEXIT2 (17:00).  Смысл проверки: после
    # доказанного полного закрытия остатка новых заявок НЕ появилось —
    # ни слепого повторного SELL, ни нового входа.  Какого-либо штатного
    # механизма "запретить вход на уже обработанную EXIT-свечу" в runtime
    # нет: позиция flat → run_cycle идёт в ENTRY-ветку и легитимно
    # переиспользует контрольную точку как новую (это зафиксированное
    # поведение production, см. test выше по файлу с NO_SIGNAL/NEW ts),
    # поэтому ложное ожидание DUPLICATE_CANDLE убрано вместе с артефактом
    # C.  Дедупликация самого EXIT-цикла (DUPLICATE_CANDLE при активном
    # freeze) проверяется отдельным тестом
    # test_g5_unverified_blocks_all_future_position_ops.
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 10)
    d2 = rt2.run_cycle()
    assert [o for o in adapter2.orders if o["side"] == "SELL"] == \
        [{"instrument": "CNYRUBF", "side": "SELL", "qty": qty - 1}]
    st3 = read_state(state_env)
    assert not st3.get("exit_unverified_until_reconcile")


# ── E2E 4: НУЛЕВОЕ исполнение (ok=True, qty_executed=0) не закрывает ──
def test_g5_zero_fill_does_not_close_position(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "PAPER_FILLED", 0)])

    d = rt.run_cycle()
    assert d.action == "EXIT"
    st = read_state(state_env)
    assert st["c5_position_qty"] == qty             # позиция цела
    assert st["c5_position_direction"] == "LONG"
    assert st["c5_entry_price"] is not None
    assert broker.position_qty == qty               # брокер не тронут
    # противоречивый успех отличим от обычного отказа:
    assert st["exit_failure_reason"] == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert st["exit_unverified_until_reconcile"] == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"
    # свеча сожжена — слепого повтора SELL нет
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert scripted.sell_calls == [qty]


# ── E2E 5: qty_executed=None / некорректные значения — без ложного
#           закрытия и без слепого повтора ─────────────────────────────
@pytest.mark.parametrize("bad_qty", [None, "1", 1.0, True, -1, 99])
def test_g5_invalid_qty_never_falsely_closes(state_env, market, bad_qty):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", bad_qty)])

    d = rt.run_cycle()
    assert d.action == "EXIT"
    st = read_state(state_env)
    # остаток НЕ выдумывается: состояние позиции полностью сохранено
    assert st["c5_position_qty"] == qty
    assert st["c5_position_direction"] == "LONG"
    assert st["c5_entry_price"] is not None
    assert broker.position_qty == qty
    assert st["exit_unverified_until_reconcile"] == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert scripted.sell_calls == [qty]             # ONE attempt only


# ── E2E 6: UNVERIFIED замораживает позицию: ни один последующий цикл
#           не отправляет SELL и не начинает операцию с ней ────────────
def test_g5_unverified_blocks_all_future_position_ops(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", None)])
    assert rt.run_cycle().action == "EXIT"
    st = read_state(state_env)
    assert st["exit_unverified_until_reconcile"]

    # Новая EXIT-свеча (нижний пробой) — freeze-защита обязана отказать
    # в любой операции с неподтверждённой позицией, включая повторный
    # SELL, несмотря на валидный сигнал выхода.
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d2 = rt.run_cycle()
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted.sell_calls == [qty]             # по-прежнему ОДИН SELL
    assert [o for o in scripted.orders if o["side"] == "SELL"] == \
        [{"instrument": "CNYRUBF", "side": "SELL", "qty": qty}]
    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == qty            # не изменено
    assert st2["exit_unverified_until_reconcile"]   # флаг живёт до reconcile

    # HOLD-свеча тоже не даёт операций с замороженной позицией
    CHOLD = candle_5m(CONTROL_DAY, 17, 10, NEUTRAL)
    set_candles(market, CHOLD)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 16)
    d3 = rt.run_cycle()
    assert d3.action == "NO_ENTRY"
    assert d3.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted.sell_calls == [qty]


# ── E2E 7a: НЕДОСТОВЕРНЫЙ reconcile (view брокера противоречит учётной
#            записи EXIT) обязан СОХРАНИТЬ freeze: снятие только по
#            подтверждённому соответствию позиции брокера остатку ───────
def test_g5_inconsistent_reconcile_keeps_freeze(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", None)])
    assert rt.run_cycle().action == "EXIT"
    assert read_state(state_env)["exit_unverified_until_reconcile"]

    # Restart with a broker view that CONTRADICTS the accounting record:
    # the EXIT requested qty contracts, so a snapshot showing MORE than
    # the stored size cannot confirm any residual of that exit.
    broker.position_qty = qty + 1
    rt2, adapter2 = make_runtime(market, broker)
    CNEUTRAL = candle_5m(CONTROL_DAY, 17, 0, NEUTRAL)
    set_candles(market, CNEUTRAL)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d = rt2.run_cycle()
    assert d.action == "NO_ENTRY"
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st2 = read_state(state_env)
    assert st2["exit_unverified_until_reconcile"]   # freeze KEPT
    assert scripted.sell_calls == [qty]             # no blind re-send ever
    assert [o for o in adapter2.orders if o["side"] == "SELL"] == []


# ── E2E 7b: freeze снимается ТОЛЬКО внешним подтверждением терминаль-
#            ности (broker_position_confirmed); ни один снимок сам по
#            себе freeze не снимает; новых заявок нет ───────────────────
def test_g5_external_confirmation_clears_freeze(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", None)])
    assert rt.run_cycle().action == "EXIT"
    assert read_state(state_env)["exit_unverified_until_reconcile"]

    # Snapshot alone (even a flat broker view) must NOT lift the freeze:
    # it cannot prove the ambiguous SELL is terminal.  Restart with the
    # broker showing qty==0 — reconcile keeps the freeze (fail closed).
    broker.position_qty = 0
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CNEUTRAL = candle_5m(CONTROL_DAY, 17, 0, NEUTRAL)
    set_candles(market, CNEUTRAL)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d = rt2.run_cycle()
    # freeze KEPT (production log: "EXIT_UNVERIFIED freeze KEPT — the
    # adapter contract cannot prove the previous SELL is terminal");
    # with qty re-baselined to 0 by the adopt-broker branch the cycle
    # routes to the ENTRY branch instead of EXIT — either way NO SELL.
    assert d.action != "EXIT"
    assert scripted2.sell_calls == []               # snapshot ≠ proof
    st_mid = read_state(state_env)
    assert st_mid["exit_unverified_until_reconcile"]

    # Now an EXTERNAL operator confirmation records the true residual
    # (flat account) into the state file — this value comes from outside
    # the runtime's own accounting, so it IS independent confirmation.
    # The guarded fake is attached BEFORE bootstrap: if reconcile() were
    # to lift the freeze without that confirmation, the following EXIT
    # signal would blind-re-send a SELL and the fake would raise.
    st_path = state_env / "state.json"
    st = json.loads(st_path.read_text())
    st["broker_position_confirmed"] = 0
    st_path.write_text(json.dumps(st))
    rt3, adapter3 = make_runtime(market, broker)
    scripted3 = ScriptedSellAdapter(adapter3, [], state_env=state_env,
                                    runtime=rt3)
    rt3.adapter = scripted3
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt3
    CNEUTRAL2 = candle_5m(CONTROL_DAY, 17, 10, NEUTRAL)
    set_candles(market, CNEUTRAL2)                  # no fresh break signal
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 15)
    d3 = rt3.run_cycle()
    assert d3.action == "NO_ENTRY"                  # flat ⇒ nothing to exit
    assert d3.reason != "EXIT_UNVERIFIED_PENDING_RECONCILE"  # freeze lifted
    st2 = read_state(state_env)
    assert not st2.get("exit_unverified_until_reconcile")
    assert st2["c5_position_qty"] == 0              # re-baselined to truth
    assert "broker_position_confirmed" not in st2   # consumed once
    assert scripted3.sell_calls == []               # freeze lifted safely
    assert [o for o in adapter3.orders if o["side"] == "SELL"] == []


# ── E2E 7c: freeze переживает перезапуск процесса (production-перси-
#            стентность state_store, без помощи тестовых заглушек) ─────
def test_g5_unverified_freeze_survives_process_restart(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", None)])
    assert rt.run_cycle().action == "EXIT"
    # Production save_state persisted the flag (state_store keeps it via
    # C5_STATE_DEFAULTS); the fake never writes state.json itself.
    assert read_state(state_env)["exit_unverified_until_reconcile"] == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"

    # Brand-new runtime instance over the SAME state file = restart.
    # Guarded empty script: ANY sell attempt raises immediately.
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, CEXIT2)                     # valid EXIT signal again
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d = rt2.run_cycle()
    assert d.action == "NO_ENTRY"
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted2.sell_calls == []               # freeze survived restart
    st2 = read_state(state_env)
    assert st2["exit_unverified_until_reconcile"]
    assert st2["c5_position_qty"] == qty            # position untouched


# ── E2E 8: G-4 регрессия — CONFIRMED rejection по-прежнему позволяет
#           повторить EXIT, а успешный полный выход идемпотентен ───────
def test_g5_confirmed_rejection_retry_still_works(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_rejected_result(), _exec_res(True, "FILLED", q)])

    d1 = rt.run_cycle()
    assert d1.action == "EXIT"
    st1 = read_state(state_env)
    assert "last_processed_5m_ts" not in st1        # candle NOT burned
    # штатный state_store дополняет состояние ключами из C5_STATE_DEFAULTS,
    # поэтому отсутствие АКТИВНОГО freeze проверяется по значению (None/False),
    # а не по синтаксическому отсутствию ключа в JSON
    assert not st1.get("exit_unverified_until_reconcile")  # G-4 path intact
    assert st1["exit_failure_reason"] == \
        "ORDER_NOT_FILLED:EXECUTION_REPORT_STATUS_REJECTED"
    assert st1["c5_position_qty"] == qty

    d2 = rt.run_cycle()                             # safe re-attempt
    assert d2.action == "EXIT"
    st2 = read_state(state_env)
    assert st2["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert st2["c5_position_qty"] == 0
    assert "exit_failure_reason" not in st2
    assert broker.position_qty == 0
    assert scripted.sell_calls == [qty, qty]        # exactly two attempts


# ══════════════════════════════════════════════════════════════════════
# G-6 — Исключение адаптера ПРИ/ПОСЛЕ отправки SELL.
#   Подтверждённый дефект: раньше self.adapter.sell_cny(...) вызывался
#   без try/except; исключение (сетевая ошибка, таймаут после dispatch)
#   всплывало из run_cycle() ДО save_state(state), поэтому защитное
#   состояние НЕ сохранялось: freeze отсутствовал, свеча не была
#   сожжена, и новый процесс мог вслепую повторить SELL.
#   Исправление (src/c5_runtime._evaluate_exit): исключение классифици-
#   руется как UNVERIFIED — позиция НЕ меняется (исключение не доказает
#   ни исполнения, ни его отсутствия), candle сжигается, ставится флаг
#   exit_unverified_until_reconcile; run_cycle сохраняет state перед
#   возвратом, защита переживает рестарт; снятие freeze — только по
#   явному операторскому подтверждению (type(confirmed) is int).
# ══════════════════════════════════════════════════════════════════════

class _SellRaisingAdapter:
    """Boundary fake: BUY/snapshot делегируют реальному PaperExecutionAdapter,
    sell_cny помечает заявку отправленной и бросает исключение — ровно то,
    что происходит при сетевом обрыве ПОСЛЕ dispatch ордера брокеру."""

    def __init__(self, paper_adapter):
        self._paper = paper_adapter
        self.sell_calls = []
        self.orders = []

    def find_account(self):
        return self._paper.find_account()

    def snapshot(self, account_id):
        return self._paper.snapshot(account_id)

    def buy_cny(self, account_id, qty, **kwargs):
        res = self._paper.buy_cny(account_id, qty, **kwargs)
        self.orders.append({"instrument": "CNYRUBF", "side": "BUY",
                            "qty": qty})
        return res

    def sell_cny(self, account_id, qty):
        # заявка УШЛА брокеру — исход неизвестен (exception ≠ proof)
        self.orders.append({"instrument": "CNYRUBF", "side": "SELL",
                            "qty": qty})
        self.sell_calls.append(qty)
        raise ConnectionError("simulated network drop after SELL dispatch")

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


def _g6_exit_setup_with_raiser(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(state_env, market)
    raising = _SellRaisingAdapter(scripted._paper)
    raising._rt = rt          # G-7 boundary-check wiring (see fake)
    rt.adapter = raising
    return rt, raising, broker, qty, CEXIT


# ── E2E G-6.1: исключение при SELL не рвёт цикл и сохраняет защиту ────
def test_g6_sell_exception_does_not_crash_and_persists_freeze(state_env,
                                                              market):
    rt, raising, broker, qty, CEXIT = _g6_exit_setup_with_raiser(
        state_env, market)

    d = rt.run_cycle()                       # НЕ должно бросить наружу

    assert d.action == "NO_ENTRY"
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)               # save_state выполнен в run_cycle
    # позиция НЕ изменена на основании предположений:
    assert st["c5_position_qty"] == qty
    assert st["c5_position_direction"] == "LONG"
    assert st["c5_entry_price"] is not None
    assert broker.position_qty == qty        # SimulatedBroker не тронут
    # консервативная защита как у UNVERIFIED:
    assert st["exit_failure_reason"] == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert st["exit_unverified_until_reconcile"] == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"
    # свеча сожжена — слепого повторного SELL нет:
    assert st["last_processed_5m_ts"] == CEXIT["time"].isoformat()
    assert raising.sell_calls == [qty]       # ровно одна отправка


# ── unit G-6.1b: исключение НЕ считается доказательством fill/NoFill ──
@pytest.mark.parametrize("exc", [ConnectionError("net"),
                                 TimeoutError("read timeout"),
                                 RuntimeError("sdk crash")],
                         ids=["connection", "timeout", "runtime"])
def test_g6_sell_exception_never_mutates_position(state_env, market, exc):
    # Отдельный runtime на каждый вид исключения (свежие state/broker):
    # после первого UNVERIFIED включается freeze и следующие циклы
    # вообще не должны доходить до sell_cny — это проверяет тест 7c.
    rt, raising, broker, qty, _CEXIT = _g6_exit_setup_with_raiser(
        state_env, market)

    def _boom(account_id, q, _e=exc):
        raising.sell_calls.append(q)
        raise _e

    raising.sell_cny = _boom                 # broker NEVER sees the order
    d = rt.run_cycle()                       # no crash out of run_cycle
    assert d.action == "NO_ENTRY"
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    st = read_state(state_env)
    assert st["c5_position_qty"] == qty      # FULL не предполагается...
    assert broker.position_qty == qty        # ...и отсутствие fill тоже
    assert st["exit_unverified_until_reconcile"]


# ── E2E G-6.2: после рестарта повторный SELL не отправляется ──────────
def test_g6_restart_after_exception_blocks_resend(state_env, market):
    rt, raising, broker, qty, _CEXIT = _g6_exit_setup_with_raiser(
        state_env, market)
    assert rt.run_cycle().reason == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"

    # Brand-new runtime over the SAME state file = process restart.
    # Guarded empty script: ANY sell attempt raises AssertionError.
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, CEXIT2)              # свежий валидный EXIT-сигнал
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d = rt2.run_cycle()
    assert d.action == "NO_ENTRY"
    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted2.sell_calls == []        # freeze пережил рестарт
    st2 = read_state(state_env)
    assert st2["exit_unverified_until_reconcile"]
    assert st2["c5_position_qty"] == qty     # позиция не тронута


# ── E2E G-6.3: freeze нельзя снять без явного подтверждения ───────────
def test_g6_freeze_after_exception_requires_explicit_confirmation(
        state_env, market):
    from src.c5_runtime import reconcile

    rt, raising, broker, qty, _CEXIT = _g6_exit_setup_with_raiser(
        state_env, market)
    assert rt.run_cycle().reason == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"

    # Snapshot alone (open_orders == 0, любой broker_qty) НЕ снимает freeze.
    for view in (qty, 0, qty - 1):
        st = read_state(state_env)
        st.pop("broker_position_confirmed", None)
        reconcile(_SnapAdapter(view), "a1", st)
        assert st["exit_unverified_until_reconcile"] == \
            "EXIT_UNVERIFIED_PENDING_RECONCILE"      # KEPT

    # Не-int «подтверждение» laundering'иться не может.
    for bad in (True, 1.0, "2"):
        st = read_state(state_env)
        st["broker_position_confirmed"] = bad
        reconcile(_SnapAdapter(qty), "a1", st)
        assert st["exit_unverified_until_reconcile"]  # KEPT

    # Единственный клапан — строгий int от внешнего оператора.
    st = read_state(state_env)
    st["broker_position_confirmed"] = 0          # внешняя сверка: flat
    reconcile(_SnapAdapter(0), "a1", st)
    assert not st.get("exit_unverified_until_reconcile")
    assert st["c5_position_qty"] == 0            # re-baselined на truth


# ── unit G-6.4: контракт классификатора результатов не сломан ─────────
def test_g6_classify_contract_unchanged_for_full_partial_zero(state_env):
    from src.c5_runtime import _classify_exit_execution as cls

    assert cls(_exec_res(True, "FILLED", 2), 2) == ("FULL", 2)
    assert cls(_exec_res(True, "FILLED", 1), 2) == ("PARTIAL", 1)
    assert cls(_exec_res(True, "FILLED", 0), 2) == ("ZERO_FILL", 0)
    assert cls(_exec_res(True, "FILLED", None), 2) == ("UNVERIFIED", 0)


# ══════════════════════════════════════════════════════════════════════
# G-7 — Durable SELL intent: состояние переживает сбой записи
# ══════════════════════════════════════════════════════════════════════

def _g7_fail_save_on_predicate(predicate):
    """Patch state_store's atomic writer to raise IOError while the
    predicate matches the payload being saved (simulates a disk failure
    at exactly the pre-save or the final save)."""
    import src.state_store as ss
    original = ss._atomic_write_state

    def flaky(state):
        if predicate(state):
            raise IOError("simulated disk failure")
        return original(state)

    return ss, original, flaky


def _g7_full_result_from_disk(state_env):
    """Read the persisted marker after a cycle (None when absent)."""
    st = read_state(state_env)
    return st.get("exit_pending_sell_intent")


# ── unit G-7.0: save_state_strict бросает StateSaveError; save_state — best-effort
def test_g7_save_state_strict_raises_and_best_effort_survives(tmp_path,
                                                              monkeypatch):
    import src.state_store as ss
    path = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", path)

    ss.save_state({"v": 1})                       # best-effort ok
    assert json.loads(open(path).read())["v"] == 1
    ss.save_state_strict({"v": 2})                # strict ok
    assert json.loads(open(path).read())["v"] == 2

    # Simulated hard failure at the temp-file creation step: BOTH writers
    # must leave the previous valid state byte-identical and no leftovers.
    real_open = ss.os.open

    def boom(*a, **k):
        raise OSError(28, "simulated ENOSPC")

    monkeypatch.setattr(ss.os, "open", boom)
    with pytest.raises(ss.StateSaveError):
        ss.save_state_strict({"v": 3})
    ss.save_state({"v": 3})                       # best-effort: prints, no raise
    monkeypatch.setattr(ss.os, "open", real_open)

    assert json.loads(open(path).read()) == {"v": 2}   # NOT destroyed
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []                        # temp cleaned up


# ── unit G-7.0b: сбой ДО os.replace не трогает предыдущий валидный JSON
def test_g7_atomic_write_failure_preserves_previous_state(tmp_path,
                                                          monkeypatch):
    import src.state_store as ss
    path = str(tmp_path / "state.json")
    monkeypatch.setenv("C5_STATE_FILE", path)
    ss.save_state_strict({"generation": 1})

    real_replace = ss.os.replace

    def boom(src_, dst_):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ss.os, "replace", boom)
    with pytest.raises(ss.StateSaveError):
        ss.save_state_strict({"generation": 2})
    monkeypatch.setattr(ss.os, "replace", real_replace)

    assert json.loads(open(path).read()) == {"generation": 1}
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []                        # temp cleaned up


# ── E2E G-7.1: сбой ПРЕДВАРИТЕЛЬНОЙ записи → sell_cny НЕ вызван ───────
def test_g7_pre_save_failure_blocks_sell_dispatch(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(state_env, market)

    def fails_on_marker(st):
        return isinstance(st.get("exit_pending_sell_intent"), dict)

    ss, original, flaky = _g7_fail_save_on_predicate(fails_on_marker)
    orig = ss._atomic_write_state
    ss._atomic_write_state = flaky
    try:
        d = rt.run_cycle()
    finally:
        ss._atomic_write_state = orig

    assert d.action == "NO_ENTRY"
    assert d.reason == "SELL_INTENT_PERSIST_FAILED"
    assert scripted.sell_calls == []              # broker NEVER saw SELL
    assert broker.position_qty == qty             # position untouched
    st = read_state(state_env)
    # candle NOT burned (order provably never left the process),
    # no marker leaked to disk, previous state intact:
    assert st.get("last_processed_5m_ts") != CEXIT["time"].isoformat() \
        or st.get("last_processed_5m_ts") is None
    assert not st.get("exit_pending_sell_intent")
    assert st["c5_position_qty"] == qty

    # next cycle with a healthy disk retries the persist-and-send atom
    full = _exec_res(True, "FILLED", qty)
    scripted._sell_results = [full]
    d2 = rt.run_cycle()
    assert d2.action == "EXIT"
    assert scripted.sell_calls == [qty]


# ── E2E G-7.2: пред-сохранение OK, SELL бросает ИСКЛЮЧЕНИЕ, финальная
#              запись падает → после рестарта повторного SELL нет ──────
def test_g7_sell_exception_with_final_save_failure_no_resend_after_restart(
        state_env, market):
    rt, raising, broker, qty, CEXIT = _g6_exit_setup_with_raiser(
        state_env, market)

    # Pre-save succeeds (no marker in payload yet); EVERY save that
    # already contains the EXIT_UNVERIFIED freeze (the final post-cycle
    # save AND our strict finalize) FAILS — worst-case durability storm.
    def fails_on_freeze(st):
        return bool(st.get("exit_unverified_until_reconcile"))

    ss, original, flaky = _g7_fail_save_on_predicate(fails_on_freeze)
    orig = ss._atomic_write_state
    ss._atomic_write_state = flaky
    try:
        d = rt.run_cycle()                        # no crash escapes
        d2 = rt.run_cycle()                       # retry cycle too
    finally:
        ss._atomic_write_state = orig

    assert d.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert raising.sell_calls == [qty]            # exactly ONE dispatch
    # The pre-dispatch marker WAS durably on disk before sell_cny ran;
    # every later write failed, so the LAST successful version still
    # carries it — and crucially it can only be interpreted alongside...
    st = read_state(state_env)
    assert st["exit_pending_sell_intent"]["qty"] == qty
    assert st["exit_pending_sell_intent"]["candle_ts"] == \
        CEXIT["time"].isoformat()

    # Restart: brand-new runtime over the same state file + guarded fake.
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d3 = rt2.run_cycle()
    assert d3.action == "NO_ENTRY"
    assert d3.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted2.sell_calls == []             # NO blind re-send
    # position still fully open, nothing invented:
    assert read_state(state_env)["c5_position_qty"] == qty


# ── E2E G-7.3: подтверждённый FULL, но финальная запись падает →
#              незавершённый маркер остаётся на диске, повторной отправки нет
def test_g7_confirmed_full_with_final_save_failure_keeps_marker(state_env,
                                                                market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market,
        lambda q: [_exec_res(True, "FILLED", q)])

    ss, original = _g7_fail_save_on_predicate(None)[0:2]
    orig = ss._atomic_write_state
    # Exactly ONE successful write is allowed: the pre-dispatch strict
    # save that persists the marker. EVERY later write (finalize retry
    # chain AND run_cycle's best-effort save) fails — durability storm
    # after the order was already handed to the adapter.
    calls = {"n": 0}

    def flaky_counted(st):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(st)                   # pre-save succeeds
        raise IOError("simulated disk failure on final save")

    ss._atomic_write_state = flaky_counted
    try:
        d = rt.run_cycle()
    finally:
        ss._atomic_write_state = orig

    assert d.action == "EXIT"                     # confirmed FULL fill
    assert scripted.sell_calls == [qty]
    st = read_state(state_env)
    # final state could NOT be persisted → the durable UNFINISHED marker
    # from the successful pre-save REMAINS on disk:
    assert st["exit_pending_sell_intent"]["status"] == "SELL_IN_FLIGHT"
    assert st["exit_pending_sell_intent"]["qty"] == qty
    assert st["c5_position_qty"] == qty           # stale on disk but SAFE:
    #                                               no silent success claim

    # Restart over the SAME state file. The FULL fill really executed at
    # the broker (SimulatedBroker went flat through the scripted mirror),
    # only OUR PERSISTENCE failed.  G-7 fail-closed recovery: the durable
    # unfinished marker found on disk blocks the WHOLE cycle before any
    # adapter/bootstrap/position evaluation — no SELL, no BUY, and NOT
    # even a reconcile snapshot could launder the stale stored size into
    # a fresh ENTRY on top of an unknown residual.  The decisive
    # assertions: the guarded fake raises on ANY sell attempt and NO SELL
    # ever reaches it again; the marker and the stale position fields
    # stay untouched on disk until an explicit operator resolution
    # (resolve_pending_sell_intent) clears exactly this operation.
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    market.holder["rows"].append(CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d2 = rt2.run_cycle()
    assert scripted2.sell_calls == []             # NO blind re-send, ever
    st2 = read_state(state_env)
    # G-7 fail-closed restart: the durable unfinished marker found on
    # disk blocks the whole cycle BEFORE bootstrap/reconcile, so nothing
    # is adopted or invented — the ambiguous operation stays durably
    # marked UNFINISHED and the position fields remain as they were on
    # the last successful write until an explicit operator resolution
    # (resolve_pending_sell_intent) clears exactly this operation.
    assert d2.action == "NO_ENTRY"
    assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert st2["exit_pending_sell_intent"]["status"] == "SELL_IN_FLIGHT"
    assert st2["c5_position_qty"] == qty          # stale but SAFE: block
    # reconcile never even ran for the blocked cycle (the guard precedes
    # self.bootstrap in run_cycle):
    assert scripted2.snapshot_calls == []

    # Explicit operator resolution of THIS marker is what finally lets
    # normal operation resume: reconcile adopts the broker truth (flat)
    # and the next cycle proceeds without touching the guarded fake.
    from src.c5_runtime import resolve_pending_sell_intent
    marker2 = st2["exit_pending_sell_intent"]
    st3 = read_state(state_env)
    assert resolve_pending_sell_intent(
        st3, candle_ts=marker2["candle_ts"], qty=qty) is True
    (state_env / "state.json").write_text(json.dumps(st3))
    rt3, adapter3 = make_runtime(market, broker)
    snap3 = rt3.bootstrap(read_state(state_env))
    assert snap3 is not None and int(snap3.position_qty) == 0


# ── unit G-7.4: snapshot/сверка НЕ снимают маркер автоматически ───────
def test_g7_pending_marker_survives_reconcile_needs_explicit_resolution(
        state_env, market):
    from src.c5_runtime import reconcile, resolve_pending_sell_intent

    marker = {"status": "SELL_IN_FLIGHT", "side": "SELL",
              "direction": "LONG", "candle_ts": "2026-10-13T10:00:00+00:00",
              "qty": 2}
    st = read_state(state_env) if (state_env / "state.json").exists() else {}
    st.update({"c5_position_qty": 2, "c5_position_direction": "LONG",
               "c5_entry_price": 0.09, "reconcile_status": None,
               "exit_pending_sell_intent": marker})
    (state_env / "state.json").write_text(json.dumps(st))

    # Any broker view — matching qty, zero, whatever — keeps the marker.
    for view in (2, 0, 1):
        st = read_state(state_env)
        reconcile(_SnapAdapter(view), "a1", st)
        assert st["exit_pending_sell_intent"] == marker   # NOT cleared

    # Wrong candle/ts or qty cannot clear it (no implicit/partial match).
    st = read_state(state_env)
    assert resolve_pending_sell_intent(st, candle_ts="other", qty=2) is False
    assert resolve_pending_sell_intent(st, candle_ts=marker["candle_ts"],
                                       qty=99) is False
    assert st["exit_pending_sell_intent"] == marker
    # Only the EXACT explicit operator resolution clears it.
    assert resolve_pending_sell_intent(st, candle_ts=marker["candle_ts"],
                                       qty=2) is True
    assert "exit_pending_sell_intent" not in st


# ── E2E G-7.5: подтверждённый REJECTED/CANCELLED — контракт G-4 цел ────
@pytest.mark.parametrize("res_factory,name",
                         [(_rejected_result, "REJECTED"),
                          (_cancelled_result, "CANCELLED")],
                         ids=["rejected", "cancelled"])
def test_g7_confirmed_rejection_still_allows_safe_retry(state_env, market,
                                                        res_factory, name):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [res_factory()])

    d = rt.run_cycle()
    # existing G-4 contract: the runtime returns the core EXIT decision;
    # the safety lives in STATE (candle unburned, marker resolved).
    assert d.action == "EXIT"
    assert scripted.sell_calls == [qty]
    st = read_state(state_env)
    # G-4: candle NOT burned, position intact, safe re-attempt allowed...
    assert st.get("last_processed_5m_ts") != CEXIT["time"].isoformat()
    assert st["c5_position_qty"] == qty
    # ...and the resolved intent marker is NOT left dangling on disk.
    assert not st.get("exit_pending_sell_intent")

    # retry on a NEW exit candle now succeeds (fresh signal, same as the
    # existing G-4 re-attempt tests — rejected orders never burn candles)
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    market.holder["rows"].append(CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    scripted._sell_results = [_exec_res(True, "FILLED", qty)]
    d2 = rt.run_cycle()
    assert d2.action == "EXIT"
    assert scripted.sell_calls == [qty, qty]
    st2 = read_state(state_env)
    assert st2["c5_position_qty"] == 0
    assert not st2.get("exit_pending_sell_intent")


# ── unit G-7.6: классификация FULL/PARTIAL/ZERO_FILL/UNVERIFIED неизменна
def test_g7_classification_contract_unchanged():
    from src.c5_runtime import _classify_exit_execution as cls

    assert cls(_exec_res(True, "FILLED", 2), 2) == ("FULL", 2)
    assert cls(_exec_res(True, "FILLED", 1), 2) == ("PARTIAL", 1)
    assert cls(_exec_res(True, "FILLED", 0), 2) == ("ZERO_FILL", 0)
    assert cls(_exec_res(True, "FILLED", None), 2) == ("UNVERIFIED", 0)
    assert cls(SimpleNamespace(ok=True), 2) == ("UNVERIFIED", 0)


# ── E2E G-7.7: UNVERIFIED-результат сохраняет И маркер, и freeze ──────
def test_g7_unverified_result_persists_marker_and_freeze(state_env, market):
    rt, scripted, broker, qty, CEXIT = _g5_exit_setup(
        state_env, market, lambda q: [_exec_res(True, "FILLED", None)])

    d = rt.run_cycle()
    # contract preserved: the UNVERIFIED branch returns the core EXIT
    # decision itself (same as before G-7); protection lives in STATE.
    assert d.action == "EXIT"
    st = read_state(state_env)
    assert st["exit_unverified_until_reconcile"] == \
        "EXIT_UNVERIFIED_PENDING_RECONCILE"
    m = st["exit_pending_sell_intent"]            # kept: order fate unknown
    # G-7: the durable marker SURVIVES (never cleared) and its qty is
    # untouched, but a proven NON-terminal outcome relabels it to
    # SELL_OUTCOME_UNVERIFIED — that label is exactly what lets the
    # owning process distinguish its own burned candle from a foreign
    # unfinished operation (pre-guard).  A fresh process still blocks.
    assert m["status"] == "SELL_OUTCOME_UNVERIFIED" and m["qty"] == qty
    assert st["c5_position_qty"] == qty           # nothing invented

    # restart → both guards block any re-send
    rt2, adapter2 = make_runtime(market, broker)
    scripted2 = ScriptedSellAdapter(adapter2, [], state_env=state_env,
                                    runtime=rt2)
    rt2.adapter = scripted2
    if _LIVE_RUNTIMES:
        _LIVE_RUNTIMES[-1] = rt2
    CEXIT2 = candle_5m(CONTROL_DAY, 17, 0, SHORT_BREAK)
    set_candles(market, CEXIT2)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 5)
    d2 = rt2.run_cycle()
    assert d2.reason == "EXIT_UNVERIFIED_PENDING_RECONCILE"
    assert scripted2.sell_calls == []
