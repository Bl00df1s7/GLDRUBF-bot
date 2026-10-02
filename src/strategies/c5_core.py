"""C5 frozen strategy core module (spec v1.1) — CNYRUBF_SPBFUT.

A PURE decision function: market data + account state + live ГО rate in,
one immutable ``Decision`` out.  The module never sends orders, never
mutates its inputs and never falls back to remembered margin values.

Signal logic is FROZEN:
  * entry  : Donchian N = 10 on CLOSED DAILY bars (shifted by 1),
             compared against the close of the 16:00 MSK 5m control bar;
  * exit   : counter-channel M = 5 only (LONG exits below Lower(5),
             SHORT exits above Upper(5)); while a position is open the
             entry signals are ignored completely;
  * no SL / TP / SAR / break-even / trailing / time-stop / filters.

Risk sizing (frozen):
  risk_multiplier = clip(ATR_TARGET / ATR14, 0.5, 1.5)   # NOT inverted
  risk_budget     = equity * RISK_BASE * risk_multiplier
  qty_raw         = min(qty_risk, qty_margin, MAX_QTY_ENGINEERING)
  M/E > 0.30      -> NO_ENTRY (ME_LIMIT)
  qty             = min(qty_raw, STAGE_MAX_QTY)           # stage 1: 5
"""

import logging
import math
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Optional, Sequence

from config.settings import (
    C5_ATR_LEN,
    C5_ATR_TARGET,
    C5_CONTRACT_SIZE_CNY,
    C5_ENTRY_DONCHIAN,
    C5_EXIT_DONCHIAN,
    C5_ME_LIMIT,
    C5_RISK_BUDGET_PCT,
    C5_RISK_MULT_MAX,
    C5_RISK_MULT_MIN,
    C5_STAGE_MAX_QTY,
)

logger = logging.getLogger("golden_bot.c5")

SYMBOL = "CNYRUBF_SPBFUT"

Action = Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD", "NO_ENTRY"]

# ── reason dictionary (§9 of the spec) ────────────────────────────────
REASON_NO_SIGNAL = "NO_SIGNAL"
REASON_ENTER_LONG = "ENTER_LONG"
REASON_ENTER_SHORT = "ENTER_SHORT"
REASON_EXIT_LONG = "EXIT_LONG"
REASON_EXIT_SHORT = "EXIT_SHORT"
REASON_EXIT_EOD = "EXIT_EOD"
REASON_HOLD = "HOLD_POSITION"
REASON_QTY_ZERO = "QTY_ZERO"
REASON_ME_LIMIT = "ME_LIMIT"
REASON_MARGIN_API = "MARGIN_API_UNAVAILABLE"
REASON_DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
REASON_STAGE_CAPPED = "STAGE_CAPPED"
# external-layer blocking reasons (§10) — surface via `block_entry()`
REASON_DAILY_LOSS = "DAILY_LOSS_BREAKER"
REASON_DRAWDOWN = "DRAWDOWN_FREEZE"
REASON_KILL_SWITCH = "KILL_SWITCH"
REASON_SLIPPAGE = "SLIPPAGE_GUARD"


class DataError(ValueError):
    """Bad market/account input — the module refuses to guess."""


class MarginRateError(DataError):
    """ГО rate missing / unparsable / outside (0, 0.5)."""


@dataclass(frozen=True)
class DailyBar:
    """One closed daily bar.  ``prev_close`` is the last 5m close of the
    previous trading day (used for True Range gaps)."""

    trade_date: date
    high: float
    low: float
    close: float
    prev_close: Optional[float] = None


@dataclass(frozen=True)
class Candle:
    """A 5m candle on the control timeframe."""

    dt: datetime
    close: float
    is_closed: bool = True
    high: Optional[float] = None
    low: Optional[float] = None

    @property
    def trade_date(self) -> date:
        return self.dt.date()


@dataclass(frozen=True)
class Position:
    direction: str          # "LONG" | "SHORT"
    qty: int = 0

    def __post_init__(self):
        if self.direction not in ("LONG", "SHORT"):
            raise DataError(f"Invalid position direction: {self.direction!r}")
        if self.qty < 0:
            raise DataError("Position qty must be >= 0")


@dataclass(frozen=True)
class Decision:
    action: Action
    qty: int
    reason: str
    # ── diagnostics (always populated) ────────────────────────────
    signal_price: float
    donchian_entry_upper: float
    donchian_entry_lower: float
    donchian_exit_lower: float
    donchian_exit_upper: float
    atr14: float
    atr_target: float
    risk_multiplier: float
    risk_budget: float
    margin_rate: Decimal
    margin_per_contract: Decimal
    required_margin: Decimal
    m_e_ratio: float
    qty_risk: int
    qty_margin: int
    qty_liquidity: int
    qty_raw: int


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _validate_inputs(
    *,
    daily_bars: Sequence[DailyBar],
    control_bar: Candle,
    equity: float,
    price: float,
    contract_size: Decimal,
    margin_rate: Decimal,
) -> None:
    n = max(C5_ENTRY_DONCHIAN, C5_EXIT_DONCHIAN)
    if len(daily_bars) < n + 1:
        raise DataError(
            f"{REASON_DATA_INSUFFICIENT}: need >= {n + 1} closed daily bars, "
            f"got {len(daily_bars)}"
        )

    if not control_bar.is_closed:
        raise DataError("control_bar is not closed — signals use closed bars only")
    if not _is_finite(control_bar.close):
        raise DataError("control_bar.close is NaN/inf")

    prev_day = None
    for i, b in enumerate(daily_bars):
        if not _is_finite(b.high) or not _is_finite(b.low) or not _is_finite(b.close):
            raise DataError(f"daily bar #{i} has NaN/inf OHLC")
        if prev_day is not None and not (b.trade_date > prev_day):
            raise DataError(
                f"daily bars must strictly increase by date (bar #{i}: {b.trade_date})"
            )
        prev_day = b.trade_date

    # No lookahead: the daily channel must exclude the current day.
    if not daily_bars[-1].trade_date < control_bar.trade_date:
        raise DataError(
            "lookahead guard: last daily bar date "
            f"{daily_bars[-1].trade_date} is not before control bar date "
            f"{control_bar.trade_date}"
        )

    if not (_is_finite(equity) and equity > 0):
        raise DataError(f"equity must be > 0, got {equity!r}")
    if not (_is_finite(price) and price > 0):
        raise DataError(f"price must be > 0, got {price!r}")
    if not (isinstance(contract_size, Decimal) and contract_size > 0):
        raise DataError(f"contract_size must be a positive Decimal, got {contract_size!r}")

    rate = _parse_rate(margin_rate)
    if not (Decimal("0") < rate < Decimal(str(0.5))):
        raise MarginRateError(
            f"{REASON_MARGIN_API}: rate {rate} outside valid band (0, 0.5)"
        )


def _parse_rate(margin_rate) -> Decimal:
    """Coerce a live ГО value into a finite Decimal or raise MarginRateError."""
    try:
        if isinstance(margin_rate, bool) or margin_rate is None:
            raise TypeError(f"unsupported ГО rate {margin_rate!r}")
        if isinstance(margin_rate, Decimal):
            rate = margin_rate
            if not rate.is_finite():
                raise ValueError("non-finite Decimal ГО rate")
        elif isinstance(margin_rate, (int, float)):
            if not math.isfinite(float(margin_rate)):
                raise ValueError("non-finite ГО rate")
            rate = Decimal(str(margin_rate))
        else:
            raise TypeError(f"unsupported ГО rate type {type(margin_rate)!r}")
    except Exception as exc:
        raise MarginRateError(f"{REASON_MARGIN_API}: unparsable rate {margin_rate!r}") from exc
    return rate


def compute_atr14(daily_bars: Sequence[DailyBar], length: int = C5_ATR_LEN) -> float:
    """Wilder ATR over CLOSED daily bars.

    TR_k = max(High-Low, |High-Close_{k-1}|, |Low-Close_{k-1}|), where
    Close_{k-1} is taken from the bar's own ``prev_close`` when provided
    (last 5m close of the previous day), else from the previous daily bar.
    Seed: mean(TR[1..L]); recursion: ATR_k = (ATR_{k-1}*(L-1)+TR_k)/L.
    """
    if len(daily_bars) < length + 1:
        raise DataError(
            f"{REASON_DATA_INSUFFICIENT}: ATR{length} needs >= {length + 1} daily bars"
        )

    trs = []
    prev = None
    for idx, bar in enumerate(daily_bars):
        pc = bar.prev_close
        if pc is None:
            pc = prev.close if prev is not None else bar.close
        tr = max(
            bar.high - bar.low,
            abs(bar.high - pc),
            abs(bar.low - pc),
        )
        trs.append(tr)
        prev = bar

    atr = sum(trs[:length]) / length
    for tr in trs[length:]:
        atr = (atr * (length - 1) + tr) / length
    return atr


def compute_risk_multiplier(atr14: float) -> float:
    """Frozen ATR adjustment: clip(ATR_TARGET / ATR14, 0.5, 1.5).

    Direction is critical (inverting it is a BUG):
      ATR14 < target (quiet)   -> multiplier > 1 -> MORE risk
      ATR14 > target (volatile)-> multiplier < 1 -> LESS risk
    """
    if not _is_finite(atr14) or atr14 <= 0:
        raise DataError(f"Invalid ATR14 for risk multiplier: {atr14!r}")
    ratio = C5_ATR_TARGET / atr14
    return max(C5_RISK_MULT_MIN, min(C5_RISK_MULT_MAX, ratio))


def _base_decision(diag: dict) -> Decision:
    return Decision(
        action="NO_ENTRY", qty=0, reason=diag["reason"],
        signal_price=diag["signal_price"],
        donchian_entry_upper=diag["entry_upper"],
        donchian_entry_lower=diag["entry_lower"],
        donchian_exit_lower=diag["exit_lower"],
        donchian_exit_upper=diag["exit_upper"],
        atr14=diag["atr14"], atr_target=C5_ATR_TARGET,
        risk_multiplier=diag.get("risk_multiplier", 0.0),
        risk_budget=diag.get("risk_budget", 0.0),
        margin_rate=diag.get("margin_rate", Decimal("0")),
        margin_per_contract=diag.get("margin_per_contract", Decimal("0")),
        required_margin=diag.get("required_margin", Decimal("0")),
        m_e_ratio=diag.get("m_e_ratio", 0.0),
        qty_risk=diag.get("qty_risk", 0),
        qty_margin=diag.get("qty_margin", 0),
        qty_liquidity=diag.get("qty_liquidity", 0),
        qty_raw=diag.get("qty_raw", 0),
    )


def evaluate(
    *,
    daily_bars: Sequence[DailyBar],
    control_bar: Candle,
    equity: float,
    free_margin: float,
    position: Optional[Position],
    price: float,
    margin_rate: Decimal,
    contract_size: Decimal = Decimal(str(C5_CONTRACT_SIZE_CNY)),
    now: Optional[datetime] = None,
    # external (deployment-stage) risk parameters — injectable per §5
    risk_base: float = C5_RISK_BUDGET_PCT,
    me_limit: float = C5_ME_LIMIT,
    max_qty: int = 40,
    stage_max_qty: int = C5_STAGE_MAX_QTY,
) -> Decision:
    """Pure C5 signal + sizing evaluation for one control bar.

    Raises ``DataError`` / ``MarginRateError`` on invalid inputs — the
    module never repairs bad data.  Callers translate ``MarginRateError``
    into ``NO_ENTRY / MARGIN_API_UNAVAILABLE`` (no fallback allowed).
    """
    _validate_inputs(
        daily_bars=daily_bars, control_bar=control_bar, equity=equity,
        price=price, contract_size=contract_size, margin_rate=margin_rate,
    )
    rate = _parse_rate(margin_rate)

    window_n = C5_ENTRY_DONCHIAN
    window_m = C5_EXIT_DONCHIAN
    entry_upper = max(b.high for b in daily_bars[-window_n:])
    entry_lower = min(b.low for b in daily_bars[-window_n:])
    exit_upper = max(b.high for b in daily_bars[-window_m:])
    exit_lower = min(b.low for b in daily_bars[-window_m:])

    close = float(control_bar.close)
    if math.isnan(close):
        raise DataError("control bar close is NaN — cannot decide")

    diag = {
        "reason": REASON_NO_SIGNAL,
        "signal_price": close,
        "entry_upper": entry_upper,
        "entry_lower": entry_lower,
        "exit_upper": exit_upper,
        "exit_lower": exit_lower,
        "atr14": 0.0,
        "margin_rate": rate,
    }

    # ── EXIT / HOLD: entry signals are ignored while in position ──
    if position is not None:
        if position.direction == "LONG" and close < exit_lower:
            diag["reason"] = REASON_EXIT_LONG
        elif position.direction == "SHORT" and close > exit_upper:
            diag["reason"] = REASON_EXIT_SHORT
        else:
            diag["reason"] = REASON_HOLD
        decision = _base_decision(diag)
        return replace(decision, action="EXIT" if diag["reason"].startswith("EXIT") else "HOLD")

    # ── ENTRY signal ─────────────────────────────────────────────
    long_sig = close > entry_upper
    short_sig = close < entry_lower
    if long_sig and short_sig:
        raise DataError("contradictory breakout (both channels breached)")
    if not long_sig and not short_sig:
        diag["reason"] = REASON_NO_SIGNAL
        return _base_decision(diag)

    # ── ATR / risk budget ────────────────────────────────────────
    atr14 = compute_atr14(daily_bars)
    multiplier = compute_risk_multiplier(atr14)
    risk_budget = equity * risk_base * multiplier
    diag["atr14"] = atr14
    diag["risk_multiplier"] = multiplier
    diag["risk_budget"] = risk_budget

    # ── SIZING pipeline (§8) ─────────────────────────────────────
    margin_per_contract = (
        Decimal(str(price)) * Decimal(contract_size) * rate
    ).quantize(Decimal("0.0001"))

    qty_risk = int(math.floor(Decimal(str(risk_budget)) / margin_per_contract)) \
        if margin_per_contract > 0 else 0
    qty_margin = int(math.floor(Decimal(str(max(0.0, float(free_margin)))) / margin_per_contract)) \
        if margin_per_contract > 0 else 0
    qty_liquidity = int(max_qty)

    qty_raw = max(0, min(qty_risk, qty_margin, qty_liquidity))
    diag.update(margin_per_contract=margin_per_contract,
                qty_risk=qty_risk, qty_margin=qty_margin,
                qty_liquidity=qty_liquidity, qty_raw=qty_raw)

    # M/E operational safety limit — checked on qty_raw, BEFORE the
    # stage cut (the cap is not a risk control; a breach blocks entry).
    required_margin = margin_per_contract * qty_raw
    m_e = float(required_margin) / equity if qty_raw > 0 else 0.0
    diag["required_margin"] = required_margin
    diag["m_e_ratio"] = m_e
    if m_e > me_limit:
        diag["reason"] = REASON_ME_LIMIT
        logger.error(
            "%s: M/E %.4f > %.2f at qty_raw=%d (%s x %s RUB/contract)",
            REASON_ME_LIMIT, m_e, me_limit, qty_raw, SYMBOL, margin_per_contract,
        )
        return _base_decision(diag)

    # Stage-1 deployment cut (does NOT change the risk model).
    qty = min(qty_raw, int(stage_max_qty))
    if qty < 1:
        diag["reason"] = REASON_QTY_ZERO
        return _base_decision(diag)

    diag["reason"] = REASON_ENTER_LONG if long_sig else REASON_ENTER_SHORT
    decision = _base_decision(diag)
    if qty < qty_raw:
        # STAGE_CAPPED surfaces in diagnostics without changing the action.
        decision = replace(decision, reason=f"{decision.reason}|{REASON_STAGE_CAPPED}")
    return replace(decision, action="ENTER_LONG" if long_sig else "ENTER_SHORT", qty=qty)


def block_entry(decision: Decision, reason: str) -> Decision:
    """External-layer override used by risk guards / execution guard.

    Turns any ENTER_* decision into NO_ENTRY with the machine-readable
    blocking code; qty is always zeroed.  Never called inside ``evaluate``.
    """
    return replace(decision, action="NO_ENTRY", qty=0, reason=reason)
