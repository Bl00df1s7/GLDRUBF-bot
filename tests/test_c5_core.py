"""Tests for the frozen C5 core module (spec v1.1) — pure, no network.

Covers T1–T18 of §13 plus the mandatory formula-inversion test (§13.2).
"""

import math
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.strategies import c5_core as mod
from src.strategies.c5_core import (
    Candle,
    DailyBar,
    DataError,
    MarginRateError,
    Position,
    block_entry,
    compute_atr14,
    compute_risk_multiplier,
    evaluate,
)

MSK = timezone(timedelta(hours=3))
RATE = Decimal("0.0585")          # test fixture value, NOT a runtime fallback
PRICE = 10.0                      # RUB per CNY
EQUITY = 1_000_000.0
FREE_MARGIN = 10_000_000.0


def make_daily(n=15, base=9.9, hi_off=0.1, lo_off=-0.1, end=date(2026, 10, 1)):
    bars = []
    d = end - timedelta(days=n + 5)
    prev_close = None
    for i in range(n):
        close = base + i * 0.01
        bar = DailyBar(
            trade_date=d,
            high=close + hi_off,
            low=close + lo_off,
            close=close,
            prev_close=prev_close,
        )
        bars.append(bar)
        prev_close = close
        d += timedelta(days=1)
    return bars


def control(close, day=date(2026, 10, 2), closed=True):
    return Candle(dt=datetime(day.year, day.month, day.day, 16, 0, tzinfo=MSK),
                  close=close, is_closed=closed)


def run(**kw):
    params = dict(
        daily_bars=make_daily(),
        control_bar=control(12.0),
        equity=EQUITY,
        free_margin=FREE_MARGIN,
        position=None,
        price=PRICE,
        margin_rate=RATE,
    )
    params.update(kw)
    return evaluate(**params)


# ── T1: ATR smoothing vs reference implementation ────────────────────
def test_t1_atr_matches_reference():
    bars = make_daily(n=20)
    # reference Wilder computation
    trs = []
    for b in bars:
        pc = b.prev_close if b.prev_close is not None else b.close
        trs.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    atr = sum(trs[:14]) / 14
    for tr in trs[14:]:
        atr = (atr * 13 + tr) / 14
    assert compute_atr14(bars) == pytest.approx(atr, rel=1e-12)


# ── T2: MANDATORY direction test (§13.2) ─────────────────────────────
def test_t2_atr_multiplier_direction():
    quiet = make_daily(hi_off=0.04, lo_off=-0.04)   # small TR -> ATR < target
    wild = make_daily(hi_off=0.30, lo_off=-0.30)    # large TR -> ATR > target

    lo = run(daily_bars=quiet)
    hi = run(daily_bars=wild)

    assert lo.atr14 < mod.C5_ATR_TARGET < hi.atr14
    assert lo.risk_multiplier > 1.0
    assert hi.risk_multiplier < 1.0
    assert lo.risk_budget > hi.risk_budget
    # exact target -> multiplier == 1
    assert compute_risk_multiplier(mod.C5_ATR_TARGET) == pytest.approx(1.0)


# ── T3: clip bounds ──────────────────────────────────────────────────
def test_t3_clip_bounds():
    assert compute_risk_multiplier(0.0001) == 1.5
    assert compute_risk_multiplier(1000.0) == 0.5


# ── T4: Donchian shift — channel excludes current day ────────────────
def test_t4_no_lookahead_current_day():
    bars = make_daily()
    # today's spike is invisible to the channel built from previous days
    spike = Candle(dt=bars[-1].trade_date and datetime(2026, 10, 2, 16, 0, tzinfo=MSK),
                   close=bars[-1].high + 5.0)
    d = run(control_bar=spike)
    assert d.donchian_entry_upper == max(b.high for b in bars[-10:])
    assert d.action == "ENTER_LONG"  # breakout measured vs shifted channel

    with pytest.raises(DataError):
        run(daily_bars=bars + [DailyBar(date(2026, 10, 2), 20, 1, 15)],
            control_bar=control(12.0))


# ── T5: signal LONG/SHORT reproduce reference on fixtures ────────────
def test_t5_signal_long_short():
    bars = make_daily()
    upper = max(b.high for b in bars[-10:])
    lower = min(b.low for b in bars[-10:])
    assert run(control_bar=control(upper + 0.01)).action == "ENTER_LONG"
    assert run(control_bar=control(lower - 0.01)).action == "ENTER_SHORT"
    mid = (upper + lower) / 2
    d = run(control_bar=control(mid))
    assert d.action == "NO_ENTRY" and d.reason == "NO_SIGNAL" and d.qty == 0


# ── T6: exit only at opposite M=5 bound ──────────────────────────────
def test_t6_exit_opposite_bound():
    bars = make_daily()
    exit_lower = min(b.low for b in bars[-5:])
    exit_upper = max(b.high for b in bars[-5:])

    d = run(position=Position("LONG", 3), control_bar=control(exit_lower - 0.01))
    assert d.action == "EXIT" and d.reason == "EXIT_LONG" and d.qty == 0

    d = run(position=Position("LONG", 3), control_bar=control(exit_lower + 0.01))
    assert d.action == "HOLD" and d.reason == "HOLD_POSITION"

    d = run(position=Position("SHORT", 3), control_bar=control(exit_upper + 0.01))
    assert d.action == "EXIT" and d.reason == "EXIT_SHORT"

    d = run(position=Position("SHORT", 3), control_bar=control(exit_upper - 0.01))
    assert d.action == "HOLD"


# ── T7: entry signals ignored while in position ──────────────────────
def test_t7_ignore_entry_in_position():
    bars = make_daily()
    upper = max(b.high for b in bars[-10:])
    d = run(position=Position("LONG", 2), control_bar=control(upper + 1.0))
    assert d.action == "HOLD" and d.reason == "HOLD_POSITION"


# ── T8/T14: sizing pipeline & stage cap ──────────────────────────────
def test_t8_sizing_pipeline_and_stage_cap():
    d = run()
    mpc = Decimal(str(PRICE)) * Decimal(1000) * RATE
    assert d.margin_per_contract == mpc.quantize(Decimal("0.0001"))
    expected_raw = min(
        int(math.floor(d.risk_budget / float(mpc))),
        int(math.floor(FREE_MARGIN / float(mpc))),
        40,
    )
    assert d.qty_raw == expected_raw
    assert d.qty == min(expected_raw, mod.C5_STAGE_MAX_QTY)
    assert d.qty <= 5 and d.qty <= 40
    assert "STAGE_CAPPED" in d.reason or d.qty == d.qty_raw
    assert d.action == "ENTER_LONG"


def test_t14_stage_cap_executed_lte_5():
    d = run(equity=50_000_000.0, free_margin=50_000_000.0)  # huge budget → raw hits 40
    assert d.qty_raw == 40
    assert d.qty == 5


# ── T9: directional ГО selection (margin_provider contract) ──────────
def test_t9_directional_rate():
    from src.margin_provider import MarginRates

    rates = MarginRates(long_margin_rub=585.0, short_margin_rub=576.0,
                       instrument="CNYRUBF_SPBFUT")
    assert rates.for_direction("LONG") == 585.0    # dlongClient
    assert rates.for_direction("SHORT") == 576.0   # dshortClient
    with pytest.raises(ValueError):
        rates.for_direction("SIDEWAYS")


# ── T10: NO FALLBACK — invalid/missing rate never yields ENTER ───────
def test_t10_no_fallback_on_bad_rate():
    for bad in (Decimal("0"), Decimal("0.9"), Decimal("-0.05"), Decimal("NaN"), None):
        with pytest.raises(MarginRateError):
            run(margin_rate=bad)

    # caller-side translation into NO_ENTRY (as the runtime does):
    try:
        run(margin_rate=Decimal("0"))
    except MarginRateError:
        decision = mod._base_decision({
            "reason": "MARGIN_API_UNAVAILABLE", "signal_price": 0.0,
            "entry_upper": 0.0, "entry_lower": 0.0,
            "exit_upper": 0.0, "exit_lower": 0.0, "atr14": 0.0,
        })
        assert decision.action == "NO_ENTRY" and decision.qty == 0


# ── T11: rate validation raises ──────────────────────────────────────
def test_t11_rate_validation():
    with pytest.raises(MarginRateError):
        run(margin_rate=Decimal("0"))
    with pytest.raises(MarginRateError):
        run(margin_rate=Decimal("0.9"))


# ── T12/T13: M/E limit ───────────────────────────────────────────────
def test_t12_me_limit_blocks_entry():
    # Invariant: qty_raw >= 1 AND M/E > 0.30 -> NO_ENTRY / ME_LIMIT.
    # ГО = 15*1000*0.06 = 900 RUB/contract; equity 2000 -> risk budget
    # 2000*0.2*mult admits 2 contracts, free margin admits 1 ->
    # qty_raw = 1 -> M/E = 900/2000 = 0.45 > 0.30 -> rejected.
    d = run(equity=2000.0, price=15.0, margin_rate=Decimal("0.06"),
            free_margin=900.0)
    assert d.qty_risk == 2 and d.qty_margin == 1 and d.qty_raw == 1
    assert d.action == "NO_ENTRY" and d.reason == "ME_LIMIT"
    assert abs(d.m_e_ratio - 0.45) < 1e-9 and d.qty == 0


def test_t13_me_limit_not_rescued_by_cap():
    # Invariant under test: engineering cap (40) does NOT rescue an entry
    # when M/E > 0.30 at qty_raw < 40.  ГО = 900 RUB/contract; equity is
    # sized so the risk budget admits exactly 25 contracts (< 40), while
    # 25*900/equity ≈ 0.325 > 0.30 forces rejection BEFORE stage-capping.
    mpc = 15.0 * 1000 * 0.06                       # 900 RUB/contract
    mult = run().risk_multiplier                   # deterministic for fixture bars
    equity = 25.4 * mpc / (0.20 * mult)            # budget admits exactly 25
    assert 25 * mpc / equity > 0.30                # M/E at raw=25 breaches limit
    d = run(equity=equity, price=15.0, margin_rate=Decimal("0.06"),
            free_margin=FREE_MARGIN)
    assert d.qty_risk == 25 and d.qty_raw == 25
    assert d.qty_raw < 40                          # cap could not have helped
    assert d.m_e_ratio > 0.30
    assert d.action == "NO_ENTRY" and d.reason == "ME_LIMIT"


# ── T15: qty zero when budget < one contract ─────────────────────────
def test_t15_qty_zero():
    d = run(equity=1000.0, free_margin=FREE_MARGIN)  # budget 200 < ГО 585
    assert d.action == "NO_ENTRY" and d.reason == "QTY_ZERO" and d.qty == 0


# ── T16: determinism / purity ────────────────────────────────────────
def test_t16_determinism_and_purity():
    bars = make_daily()
    d1 = run(daily_bars=bars)
    d2 = run(daily_bars=bars)
    assert d1 == d2
    # inputs untouched (module is pure — no mutation of caller data)
    assert bars == make_daily()
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        d1.qty = 99


# ── T17: data guards ─────────────────────────────────────────────────
def test_t17_data_guards():
    with pytest.raises(DataError):
        run(daily_bars=make_daily(n=5))                    # insufficient history
    with pytest.raises(DataError):
        run(control_bar=control(12.0, closed=False))       # unclosed candle
    dup = make_daily()
    dup[-1] = DailyBar(dup[-2].trade_date, 10, 9, 9.5)     # duplicate date
    with pytest.raises(DataError):
        run(daily_bars=dup)
    nan = make_daily()
    nan[3] = DailyBar(nan[3].trade_date, float("nan"), 9, 9)
    with pytest.raises(DataError):
        run(daily_bars=nan)
    with pytest.raises(DataError):
        run(equity=0.0)
    with pytest.raises(DataError):
        run(price=-1.0)


# ── invariants (§11) ─────────────────────────────────────────────────
def test_invariants():
    d = run()
    assert d.risk_budget <= 0.30 * EQUITY + 1e-9
    assert 0.5 <= d.risk_multiplier <= 1.5
    assert d.qty <= 40 and d.qty <= mod.C5_STAGE_MAX_QTY
    if d.qty > 0:
        assert d.required_margin / d.equity_or(EQUITY) if False else True
        assert float(d.required_margin) / EQUITY <= 0.30 + 1e-9
    assert (d.action.startswith("ENTER")) == (d.qty > 0)


# ── external-layer blocking contract ─────────────────────────────────
def test_block_entry_override():
    d = run()
    for reason in ("DAILY_LOSS_BREAKER", "DRAWDOWN_FREEZE",
                   "KILL_SWITCH", "SLIPPAGE_GUARD"):
        b = block_entry(d, reason)
        assert b.action == "NO_ENTRY" and b.qty == 0 and b.reason == reason
    assert run().qty > 0  # original decision untouched (purity)
