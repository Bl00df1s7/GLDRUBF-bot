"""PAPER smoke-test runner with a diagnostic clock override.

Runs ONE full C5 production cycle (src.c5_runtime.C5Runtime.run_cycle) in
PAPER mode so the execution branch can be exercised safely:

    python3 scripts/paper_smoke_test.py --clock-epoch 1790946060
    python3 scripts/paper_smoke_test.py --clock-epoch "2026-10-02T16:01:00+03:00"

WHY THIS FILE EXISTS
--------------------
The ENTRY guard ``ENTRY_OUTSIDE_CONTROL_POINT`` (control_point_passed() in
src/c5_runtime.py) blocks entries before 16:05 MSK, so a plain smoke run
outside that window always ends with NO_ENTRY and never reaches the
execution branch.  C5Runtime already supports dependency-injected time via
its ``now_fn`` constructor parameter — this runner only exposes that hook
through a CLI flag.  It is PURELY DIAGNOSTIC:

  * src/c5_strategy.py / src/c5_runtime.py are NOT modified;
  * C5 parameters, candles, guards, sizing, execution logic are untouched;
  * there is NO force-entry: every guard still runs normally, the injected
    clock merely changes what "now" means for the cycle;
  * without --clock-epoch the behaviour is EXACTLY as before (C5Runtime's
    default now_fn uses the real wall clock);
  * all market/account data stay REAL read-only API calls (candles,
    equity snapshot, margin rates, order book).

SAFETY INVARIANTS (proven by Stage 5A/5B tests, re-checked here at startup)
---------------------------------------------------------------------------
* TRADING_MODE must be PAPER (fail closed otherwise);
* AUTO_TRADING_ENABLED must be true (the runtime gate enforces it again);
* PAPER execution goes through PaperExecutionAdapter + SimulatedBroker,
  which structurally has no ``post_order`` reference — the OrdersServiceApi
  write path is physically unreachable in PAPER.

Exit code: 0 when the cycle completes (regardless of decision), 1 on
configuration errors (fail closed, nothing executed).

DIAGNOSTIC FLAGS (test-branch convenience only; defaults keep production
behaviour EXACTLY intact):
  * --clock-now  — set the injected time to the REAL current wall clock.
    Lets you run "as if it is now" inside the [16:05, ...) MSK control
    window without typing epoch values.
  * --fresh-state — use a throwaway idempotency/state file instead of the
    shared default (/tmp/c5_runtime_state.json), so repeated runs at the
    same time are not blocked as DUPLICATE_CANDLE.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Make the repo root importable when launched as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_clock_epoch(value: str) -> datetime:
    """Parse --clock-epoch into an aware UTC datetime.

    Accepts either POSIX epoch seconds (int/float) or an ISO-8601 string.
    Naive ISO strings are interpreted as UTC; aware strings keep their own
    offset (so 16:01 MSK == '2026-10-02T16:01:00+03:00').
    """
    v = value.strip()
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except ValueError:
        pass
    iso = v[:-1] + "+00:00" if v.endswith("Z") else v
    dt_ = datetime.fromisoformat(iso)          # ValueError propagates (fail closed)
    if dt_.tzinfo is None:
        dt_ = dt_.replace(tzinfo=timezone.utc)
    return dt_.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_smoke_test",
        description="One PAPER-mode C5 cycle with an optional diagnostic "
                    "clock override (default: real wall clock).")
    p.add_argument(
        "--clock-epoch", default=None, metavar="EPOCH_OR_ISO",
        help="Diagnostic time override passed to C5Runtime as now_fn. "
             "POSIX seconds (e.g. 1790946060) or ISO-8601 "
             "(e.g. 2026-10-02T16:01:00+03:00). Omit to use the real clock.")
    p.add_argument(
        "--clock-now", action="store_true",
        help="Diagnostic convenience: inject the REAL current wall clock as "
             "now_fn (same effect as --clock-epoch with the current epoch, "
             "but no value to type). Use it to run inside the [16:05, ...) "
             "MSK control window 'as if it is now'.")
    p.add_argument(
        "--fresh-state", action="store_true",
        help="Use a throwaway C5_STATE_FILE for this run so repeated cycles "
             "at the same time are not blocked as DUPLICATE_CANDLE. Does NOT "
             "touch the shared default state file.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    token = os.environ.get("T_SANDAPI", "")
    if not token:
        print("FAIL CLOSED: T_SANDAPI is missing or empty", file=sys.stderr)
        return 1

    from src.runtime_config import load_trading_config
    config = load_trading_config()          # fail-closed on missing/invalid env
    if not config.is_paper:
        print("FAIL CLOSED: paper smoke test requires TRADING_MODE=PAPER",
              file=sys.stderr)
        return 1

    if args.clock_epoch is not None and args.clock_now:
        print("FAIL CLOSED: --clock-epoch and --clock-now are mutually "
              "exclusive", file=sys.stderr)
        return 1

    now_fn = None
    if args.clock_epoch is not None:
        fixed = parse_clock_epoch(args.clock_epoch)
        now_msk = fixed.astimezone(ZoneInfo("Europe/Moscow"))
        print(f"[smoke] DIAGNOSTIC clock override: {fixed.isoformat()} "
              f"(MSK: {now_msk.isoformat()})")
        now_fn = lambda: fixed              # noqa: E731 — injected via ctor only
    elif args.clock_now:
        fixed = datetime.now(timezone.utc)  # real wall clock, frozen per cycle
        now_msk = fixed.astimezone(ZoneInfo("Europe/Moscow"))
        print(f"[smoke] DIAGNOSTIC clock override (--clock-now): "
              f"{fixed.isoformat()} (MSK: {now_msk.isoformat()})")
        now_fn = lambda: fixed              # noqa: E731 — injected via ctor only

    if args.fresh_state:
        fd, path = tempfile.mkstemp(prefix="c5_smoke_state_", suffix=".json")
        os.close(fd)
        os.unlink(path)                     # start truly fresh; runtime recreates
        os.environ["C5_STATE_FILE"] = path  # resolved at call time by state_store
        print(f"[smoke] DIAGNOSTIC fresh state file: {path}")

    runtime_kwargs = {"config": config}
    if now_fn is not None:
        runtime_kwargs["now_fn"] = now_fn   # existing DI hook; default unchanged

    from src.c5_runtime import C5Runtime
    runtime = C5Runtime(token, **runtime_kwargs)
    decision = runtime.run_cycle()

    print(f"\nC5 cycle completed:\naction={decision.action}\n"
          f"qty={decision.qty}\nreason={decision.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
