"""Stage 5A — THE single source of truth for trading-safety config parsing.

Every trading-safety switch is read from the environment exactly once,
here; ``main.py`` stays a thin entrypoint and ``c5_runtime`` consumes the
validated result instead of re-parsing (no duplicated checks).

Rules (fail closed — safety over convenience):

  * TRADING_MODE          — MANDATORY. Only "PAPER" or "LIVE"
                            (case-insensitive) are valid. Missing, blank
                            or invalid → TradingConfigError (fail
                            closed). There is NO default mode at all —
                            not PAPER, and NEVER LIVE.
  * AUTO_TRADING_ENABLED  — optional. Parsed case-insensitively:
                            true/false, 1/0, yes/no, on/off.
                            Missing or invalid → False (trading blocked
                            before execution).

Trading may reach an execution adapter ONLY when:

    mode == LIVE  AND  auto_trading_enabled is True
        (explicit live opt-in), or
    mode == PAPER AND  auto_trading_enabled is True
        (simulated local state only — physically cannot place real orders).

AUTO_TRADING_ENABLED=false blocks ALL order flow (paper included) before
the execution layer is ever reached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TRADING_MODE_ENV = "TRADING_MODE"
AUTO_TRADING_ENV = "AUTO_TRADING_ENABLED"

MODE_PAPER = "PAPER"
MODE_LIVE = "LIVE"
VALID_MODES = (MODE_PAPER, MODE_LIVE)

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


class TradingConfigError(RuntimeError):
    """Fail-closed configuration error — no trading may proceed."""


def parse_auto_trading(raw: str | None) -> bool:
    """Parse AUTO_TRADING_ENABLED. Missing / blank / invalid → False."""
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    # "false", "0", "no", "off", "" and ANY unrecognized garbage all mean
    # the same thing for safety: not enabled.
    return False


@dataclass(frozen=True)
class TradingConfig:
    """Validated trading-safety configuration (immutable snapshot)."""

    mode: str
    auto_trading_enabled: bool

    @property
    def is_paper(self) -> bool:
        return self.mode == MODE_PAPER

    @property
    def is_live(self) -> bool:
        return self.mode == MODE_LIVE

    @property
    def trading_allowed(self) -> bool:
        """Gate evaluated BEFORE any execution call reaches the broker."""
        return self.auto_trading_enabled


def load_trading_config(environ: dict | None = None) -> TradingConfig:
    """Parse + validate TRADING_MODE / AUTO_TRADING_ENABLED from env.

    Raises TradingConfigError on missing/invalid TRADING_MODE — the bot
    must refuse to run rather than guess a mode (and never default LIVE).
    """
    env = os.environ if environ is None else environ

    raw_mode = env.get(TRADING_MODE_ENV)
    if raw_mode is None or str(raw_mode).strip() == "":
        # FAIL CLOSED — TRADING_MODE is MANDATORY. There is NO default
        # trading mode whatsoever (not PAPER, and never LIVE): the bot
        # refuses to construct a runtime until the operator explicitly
        # declares the mode. Missing value → hard error, exactly like an
        # invalid one; nothing downstream can fall back to LIVE.
        raise TradingConfigError(
            f"{TRADING_MODE_ENV} is required but missing/blank — there is "
            f"no default trading mode (fail closed; never defaults to "
            f"LIVE). Set {TRADING_MODE_ENV} to one of "
            f"{', '.join(VALID_MODES)} only when this is intentional.")
    mode = str(raw_mode).strip().upper()
    if mode not in VALID_MODES:
        # Invalid value → refuse completely (no adapter, no cycle).
        raise TradingConfigError(
            f"{TRADING_MODE_ENV}={raw_mode!r} is invalid — expected one of "
            f"{', '.join(VALID_MODES)} (fail closed; never defaults to LIVE)")

    enabled = parse_auto_trading(env.get(AUTO_TRADING_ENV))
    return TradingConfig(mode=mode, auto_trading_enabled=enabled)


def enforce_trading_gate(config: TradingConfig) -> None:
    """Raise TradingConfigError unless trading is explicitly allowed.

    Called by the runtime immediately BEFORE any order-flow method
    (buy_cny / sell_cny / TMON liquidation) can be reached.
    """
    if not config.trading_allowed:
        raise TradingConfigError(
            f"AUTO_TRADING_ENABLED is not true — trading is blocked "
            f"before execution (mode={config.mode}). Set "
            f"{AUTO_TRADING_ENV}=true only when this is intentional.")
