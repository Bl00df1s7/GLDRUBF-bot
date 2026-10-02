"""C5 risk protection: daily loss breaker, drawdown freeze, kill-switch.

Three independent mechanisms (all evaluated against live equity):

  * Daily loss breaker : loss >= 10% vs equity at start of trading day
                         → block NEW entries for the rest of the session.
  * Drawdown freeze    : DD >= 20% vs historical equity peak
                         → block NEW entries until equity recovers.
  * Kill-switch        : DD >= 30% vs historical equity peak
                         → full halt; manual review required to resume.
                         NO automatic strategy/parameter replacement.

State is persisted through the existing Golden Bot state_store so limits
survive restarts (no silent auto-correction of stored peaks).
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    C5_DAILY_LOSS_BREAKER,
    C5_DRAWDOWN_FREEZE,
    C5_KILL_SWITCH_DD,
)
from src.risk_manager import session_date

logger = logging.getLogger("golden_bot.protection")

EVENT_KILL_SWITCH = "KILL_SWITCH_HALT"


@dataclass
class ProtectionStatus:
    """Outcome of the protection evaluation for one cycle."""

    entries_allowed: bool
    halted: bool
    daily_loss_pct: float
    drawdown_pct: float
    reason: str


def evaluate_protections(
    state: dict,
    equity_rub: float,
    now=None,
) -> ProtectionStatus:
    """Update peak/daily baselines in ``state`` and evaluate the guards.

    The historical peak only ever moves upward; it is never silently
    lowered. Kill-switch, once tripped, stays tripped until a human
    clears ``kill_switch`` in the persisted state after manual review.
    """
    if equity_rub is None or equity_rub <= 0:
        return ProtectionStatus(False, bool(state.get("kill_switch")),
                                0.0, 0.0, "EQUITY_UNAVAILABLE")

    today = session_date(now)
    if state.get("c5_daily_session_date") != today:
        state["c5_daily_session_date"] = today
        state["c5_daily_start_equity"] = equity_rub
        logger.info("Daily baseline reset: session=%s equity=%.2f",
                    today, equity_rub)

    daily_start = float(state.get("c5_daily_start_equity") or equity_rub)
    peak = float(state.get("equity_peak") or 0.0)
    if equity_rub > peak:
        peak = equity_rub
        state["equity_peak"] = peak

    daily_loss_pct = max(0.0, (daily_start - equity_rub) / daily_start) \
        if daily_start > 0 else 0.0
    drawdown_pct = max(0.0, (peak - equity_rub) / peak) if peak > 0 else 0.0

    # Kill-switch has priority and is sticky until manual review.
    if state.get("kill_switch"):
        return ProtectionStatus(
            False, True, daily_loss_pct, drawdown_pct,
            f"KILL_SWITCH_ACTIVE (DD={drawdown_pct:.2%}); manual review required",
        )

    if drawdown_pct >= C5_KILL_SWITCH_DD:
        state["kill_switch"] = True
        state["TRADING_HALTED"] = True
        logger.critical(
            "%s: drawdown %.2f%% >= %.0f%% of peak %.2f — FULL HALT, "
            "manual review required. Strategy/parameters are NOT changed "
            "automatically.",
            EVENT_KILL_SWITCH, drawdown_pct * 100, C5_KILL_SWITCH_DD * 100, peak,
        )
        return ProtectionStatus(
            False, True, daily_loss_pct, drawdown_pct,
            f"KILL_SWITCH: DD {drawdown_pct:.2%} >= {C5_KILL_SWITCH_DD:.0%}",
        )

    if daily_loss_pct >= C5_DAILY_LOSS_BREAKER:
        return ProtectionStatus(
            False, False, daily_loss_pct, drawdown_pct,
            f"DAILY_LOSS_BREAKER: {daily_loss_pct:.2%} >= "
            f"{C5_DAILY_LOSS_BREAKER:.0%} of session-start equity",
        )

    if drawdown_pct >= C5_DRAWDOWN_FREEZE:
        return ProtectionStatus(
            False, False, daily_loss_pct, drawdown_pct,
            f"DRAWDOWN_FREEZE: {drawdown_pct:.2%} >= {C5_DRAWDOWN_FREEZE:.0%} "
            "of historical peak",
        )

    return ProtectionStatus(True, False, daily_loss_pct, drawdown_pct, "OK")


def note_trade_pnl(state: dict, realized_delta_rub: float) -> None:
    """Optional hook: legacy realized-PnL tracking kept separate.

    The C5 daily breaker compares live equity snapshots, so this hook is
    intentionally a no-op placeholder that preserves the old interface
    without double-counting.
    """
    return None
