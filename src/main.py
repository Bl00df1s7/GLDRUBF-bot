"""Golden Bot — C5 production entry point (Stage 2D).

The ACTIVE runtime is the frozen C5 strategy on CNYRUBF_SPBFUT, wired in
``src.c5_runtime``. This module is a thin delegating entrypoint so that
existing launchers (``python -m src.main``, workflows, systemd) keep
working while the legacy GLDRUBF pipeline is removed from the active
path. The old GLDRUBF implementation lives only in the archive branch
``legacy/gldrubf-old-strategy``.

Usage:
    python -m src.main
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AUTO_TRADING_ENABLED, C5_TARGET_TICKER
from src.c5_runtime import C5Runtime, run_once
from src.state_store import load_state, save_state

SIGNAL_ONLY = not AUTO_TRADING_ENABLED


def _run():
    """Legacy entry kept as a thin delegation to the C5 runtime.

    The old GLDRUBF pipeline (strategy.py / SAR / SL / TP / break-even /
    stop_orders / position_monitor / 3% circuit breaker) is NOT executed
    here anymore; it exists only in the archive branch
    legacy/gldrubf-old-strategy.
    """
    token = os.environ.get("SANDBOX_TOKEN") or os.environ.get(
        "INVEST_TOKEN", "")
    if not token:
        raise RuntimeError("SANDBOX_TOKEN / INVEST_TOKEN is empty")
    decision = run_once(token)
    print(f"\n✅ C5 cycle completed: action={decision.action} "
          f"qty={decision.qty} reason={decision.reason}")
    return decision


def main():
    """Acquire the process lock and execute one bot run."""
    from src.lock import process_lock

    with process_lock(os.environ.get("LOCK_FILE", "/tmp/c5_bot.lock")):
        _run()


if __name__ == "__main__":
    main()
