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

EXECUTION-PROOF DIAGNOSTICS (--diag-market / --diag-margin / --diag-broker)
---------------------------------------------------------------------------
Purpose: reproduce ONE deterministic PAPER execution cycle
(C5 SIGNAL → sizing → risk guards → ORDER_SUBMIT(PAPER) → SimulatedBroker
fill → position state) without waiting for a real breakout and without
touching any trading logic.

The mocks patch ONLY external network boundaries — exactly the two seams
already faked by the Stage 5B integration tests (tests/
test_stage5b_paper_runtime.py):

  * --diag-market  src.c5_runtime.load_candles   → deterministic fixture
                   (daily channel ≈ 1400 + closed 16:00 MSK 5m candle at
                   1420 = a valid Upper(10) breakout) and
                   src.c5_runtime.fetch_order_book → tight book around
                   mid 1400 (passes the frozen slippage guard).
  * --diag-margin  injects a boundary fake margin_tracker via the
                   EXISTING C5Runtime ctor DI hook (same seam Stage 5B
                   uses) with rates dlong=0.0585 / dshort=0.0576.
  * --diag-broker  builds the REAL production paper stack locally:
                   LiveExecutionAdapter used EXCLUSIVELY through its two
                   read-only methods (find_account/snapshot) seeds
                   _build_paper_broker → REAL SimulatedBroker → REAL
                   PaperExecutionAdapter, injected via the existing ctor
                   DI hook.  No fake execution object is used; fills go
                   through the production PaperExecutionAdapter.buy_cny/
                   sell_cny → SimulatedBroker.apply code.

NEVER mocked or modified: c5_core, run_cycle(), _evaluate_entry(),
protections, execution guard, cash manager, PaperExecutionAdapter,
SimulatedBroker, LIVE execution, post_order.  With no --diag-* flag the
runner behaves EXACTLY as before (real candles / book / ГО / broker
seed).  The three flags form one coherent synthetic market scenario, so
they must be used together (fail closed otherwise).

Reproducible command (execution proof):

    python3 scripts/paper_smoke_test.py \
        --clock-epoch "2026-10-02T16:05:00+03:00" \
        --fresh-state --diag-market --diag-margin --diag-broker

Expected end state: action=ENTER_LONG qty=2, order log contains
PAPER_FILL CNYRUBF BUY x2, broker.position_qty=2, state file shows
c5_position_qty=2 / LONG.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
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
    p.add_argument(
        "--diag-market", action="store_true",
        help="Diagnostic execution-proof boundary fake: patch "
             "src.c5_runtime.load_candles (deterministic fixture: daily "
             "channel ~1400 + closed 16:00 MSK 5m candle at 1420 = valid "
             "Upper(10) breakout) and src.c5_runtime.fetch_order_book (tight "
             "book around mid 1400). c5_core / sizing / guards stay REAL. "
             "Must be combined with --diag-margin and --diag-broker.")
    p.add_argument(
        "--diag-margin", action="store_true",
        help="Diagnostic execution-proof boundary fake: inject a fake "
             "margin_tracker (dlong=0.0585 / dshort=0.0576, Stage 5B values) "
             "through the EXISTING C5Runtime ctor DI hook instead of the live "
             "ГО API. Must be combined with --diag-market and --diag-broker.")
    p.add_argument(
        "--diag-broker", action="store_true",
        help="Diagnostic execution-proof seed: build the REAL production "
             "paper stack locally — read-only LiveExecutionAdapter "
             "(find_account/snapshot ONLY) -> _build_paper_broker -> REAL "
             "SimulatedBroker -> REAL PaperExecutionAdapter — injected via "
             "the existing ctor DI hook. No fake execution object. Must be "
             "combined with --diag-market and --diag-margin.")
    return p


# ══════════════════════════════════════════════════════════════════════
# Execution-proof diagnostics — boundary fakes ONLY (market data / order
# book / ГО), identical seams to tests/test_stage5b_paper_runtime.py.
# Nothing inside c5_runtime / c5_core / protection / cash_manager /
# execution_adapter / SimulatedBroker is mocked or modified.
# ══════════════════════════════════════════════════════════════════════
DIAG_REF_DATE = dt.date(2026, 9, 30)      # last closed daily bar in fixture
DIAG_PRICE = 1400.0                       # daily channel Upper(10)=Lower(10)
DIAG_LONG_BREAK = 1420.0                  # > Upper(10) → ENTER_LONG signal
DIAG_DAILY_HL = 0.07815                   # half-range → ATR14 == 0.1563
DIAG_MARGIN_LONG = 0.0585                 # Stage 5B frozen fixture rates
DIAG_MARGIN_SHORT = 0.0576


def diag_daily_df(n=60):
    """Deterministic OHLC: H=1400+HL, L=1400-HL, C=1400 (Stage 5B shape)."""
    import pandas as pd
    rows = []
    d = DIAG_REF_DATE - dt.timedelta(days=n - 1)
    for _ in range(n):
        ts = pd.Timestamp(d.isoformat(), tz="UTC")
        rows.append({"time": ts, "open": DIAG_PRICE,
                     "high": DIAG_PRICE + DIAG_DAILY_HL,
                     "low": DIAG_PRICE - DIAG_DAILY_HL,
                     "close": DIAG_PRICE, "volume": 1})
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def diag_control_candle(now_msk):
    """Closed 5m candle starting at the most recent 5m boundary strictly
    BEFORE `now_msk` (start+5m <= now ⇒ genuinely closed).  At the standard
    16:05 proof time this is exactly the frozen 16:00 MSK control candle."""
    import pandas as pd
    m = (now_msk.minute // 5) * 5 - 5
    if m < 0:                       # hour rolled back — use previous hour end
        start = (now_msk.replace(minute=0, second=0, microsecond=0)
                 - dt.timedelta(minutes=5))
    else:
        start = now_msk.replace(minute=m, second=0, microsecond=0)
    return {"time": pd.Timestamp(start.astimezone(timezone.utc)),
            "open": DIAG_LONG_BREAK, "high": DIAG_LONG_BREAK + 0.5,
            "low": DIAG_LONG_BREAK - 0.5, "close": DIAG_LONG_BREAK,
            "volume": 1}


def install_diag_market(rt_mod, cycle_now_fn=None):
    """Patch the two network boundaries imported INTO src.c5_runtime
    (module-attribute seam — exactly like the Stage 5B `market` fixture).
    Returns the control-candle row the runtime will pick up.

    The control candle is anchored to the time the CYCLE will actually use
    (the injected diagnostic clock when present, otherwise the real wall
    clock), so latest_closed_5m() inside run_cycle() genuinely sees it as
    CLOSED (start+5m <= now)."""
    import pandas as pd
    ref = cycle_now_fn() if cycle_now_fn is not None else rt_mod._now_utc()
    now_msk = rt_mod._msk(ref)
    row = diag_control_candle(now_msk)
    daily = diag_daily_df()
    five_m = pd.DataFrame([row])

    def _load(token, uid, candles_count=200, timeframe="4H"):
        if timeframe == "1d":
            return daily.copy()
        if timeframe == "5m":
            return five_m.copy()
        raise AssertionError(f"C5 runtime must not request {timeframe!r}")

    rt_mod.load_candles = _load
    # Tight book around mid 1400 (same shape as the Stage 5B fixture):
    # spread 0.1 RUB ≈ 0.007% < frozen slippage limit 0.10%.
    rt_mod.fetch_order_book = lambda *a, **k: (
        [(DIAG_PRICE - 0.05, 100)], [(DIAG_PRICE + 0.05, 100)])
    return row


class DiagMarginTracker:
    """Boundary fake for the ГО API only — same duck-type contract the
    Stage 5B FakeMarginTracker uses (refresh()/current)."""

    def __init__(self):
        self.current = None
        self.refresh_calls = 0

    def refresh(self):
        from src.margin_provider import MarginRates
        self.refresh_calls += 1
        self.current = MarginRates(long_margin_rub=DIAG_MARGIN_LONG,
                                   short_margin_rub=DIAG_MARGIN_SHORT,
                                   instrument="CNYRUBF")
        return self.current


def build_diag_instrument(token):
    """Resolve the REAL CNYRUBF instrument via the normal read-only path;
    fall back to the Stage 5B test stub only when the API is unreachable
    (e.g. SDK absent / offline CI). Read-only either way."""
    try:
        from src.instruments import get_c5_instrument
        return get_c5_instrument(token)
    except Exception as exc:
        print(f"[smoke] DIAGNOSTIC instrument fallback (read-only API "
              f"unavailable: {exc}) — using Stage 5B test stub")

        class FakeInstrument:
            uid = "uid-cnyrubf-test"
            ticker = "CNYRUBF"
            class_code = "SPBFUT"
        return FakeInstrument()


def build_diag_broker(token, instrument_uid):
    """REAL production paper stack seeded from the real read-only account
    snapshot (LiveExecutionAdapter used EXCLUSIVELY through find_account/
    snapshot — the same code path build_execution_adapter routes PAPER to).
    Raises on any read-API failure (fail closed — no artificial balance)."""
    from src.execution_adapter import (LiveExecutionAdapter,
                                       PaperExecutionAdapter,
                                       _build_paper_broker)
    broker = _build_paper_broker(LiveExecutionAdapter(token, instrument_uid))
    return PaperExecutionAdapter(broker)


def print_diag_final_state(runtime, decision):
    """Post-cycle proof: simulated order log, broker position, persisted
    state — all read from objects the production cycle already mutated."""
    adapter = runtime.adapter
    broker = getattr(adapter, "broker", None)
    print("[diag] final state:")
    print(f"action={decision.action} qty={decision.qty} reason={decision.reason}")
    if broker is not None:
        try:
            print(f"position_qty={broker.position_qty} cash={broker.cash:.2f} "
                  f"equity={broker.equity:.2f}")
        except (AttributeError, TypeError):
            pass    # custom broker double without numeric fields — orders/state below
    orders = getattr(adapter, "orders", None)
    if orders is not None:
        print(f"orders={orders}")
    try:
        from src.state_store import load_state
        st = load_state()
        print(f"state: c5_position_qty={st.get('c5_position_qty')} "
              f"c5_position_direction={st.get('c5_position_direction')} "
              f"last_action={st.get('last_action')}")
    except Exception as exc:
        print(f"[diag] state file unreadable: {exc}")


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

    diag_flags = (args.diag_market, args.diag_margin, args.diag_broker)
    if any(diag_flags) and not all(diag_flags):
        # The three diagnostics form ONE coherent synthetic market scenario
        # (fixture candles @1400/1420 <-> fixture ГО <-> seeded broker).
        # Half-synthetic runs would silently mix real and fake market data.
        print("FAIL CLOSED: --diag-market, --diag-margin and --diag-broker "
              "must be used together", file=sys.stderr)
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

    import src.c5_runtime as rt_mod
    diag_on = bool(diag_flags[0])           # all-or-nothing (validated above)
    diag_candle = None
    if diag_on:
        # Boundary fakes ONLY — production trading logic stays untouched.
        diag_candle = install_diag_market(rt_mod, now_fn)
        print("[smoke] DIAGNOSTIC market fake installed: "
              "src.c5_runtime.load_candles + fetch_order_book patched "
              "(daily channel ~1400, closed 5m control candle close="
              f"{DIAG_LONG_BREAK}, tight book around mid {DIAG_PRICE})")
        runtime_kwargs["margin_tracker"] = DiagMarginTracker()
        print(f"[smoke] DIAGNOSTIC margin fake injected via ctor DI: "
              f"dlong={DIAG_MARGIN_LONG} dshort={DIAG_MARGIN_SHORT}")

    from src.c5_runtime import C5Runtime
    runtime = C5Runtime(token, **runtime_kwargs)

    if diag_on:
        instrument = build_diag_instrument(token)
        runtime.instrument = instrument     # idempotent bootstrap honours it
        try:
            adapter = build_diag_broker(token, instrument.uid)
        except Exception as exc:
            # Fail closed BEFORE any execution: no artificial balance.
            print(f"FAIL CLOSED: --diag-broker real read-only snapshot "
                  f"unavailable: {exc}", file=sys.stderr)
            return 1
        broker = getattr(adapter, "broker", None)
        runtime.adapter = adapter           # existing DI seam (ctor param
        runtime.trader = adapter            # already skips bootstrap wiring)
        seed = ""
        if broker is not None and all(hasattr(broker, a) for a in
                                      ("cash", "equity", "tmon_qty",
                                       "position_qty")):
            seed = (f"cash={broker.cash:.2f} equity={broker.equity:.2f} "
                    f"tmon={broker.tmon_qty} position_qty={broker.position_qty}")
        print("[smoke] DIAGNOSTIC broker seeded from REAL read-only account "
              f"snapshot: {seed} (REAL PaperExecutionAdapter + "
              "SimulatedBroker — no fake execution object)")

    decision = runtime.run_cycle()

    print(f"\nC5 cycle completed:\naction={decision.action}\n"
          f"qty={decision.qty}\nreason={decision.reason}")

    if diag_on:
        if decision.action in ("ENTER_LONG", "ENTER_SHORT", "EXIT"):
            print(f"[diag] ORDER_SUBMIT(PAPER): "
                  f"{'BUY' if decision.action == 'ENTER_LONG' else 'SELL'} "
                  f"CNYRUBF x{decision.qty} -> PAPER_FILL (SimulatedBroker)")
        if decision.qty == 0 and decision.reason not in ("NO_SIGNAL",):
            print(f"[diag] execution path NOT reached: blocked at "
                  f"{decision.reason} (all guards stayed active)")
        if diag_candle is not None:
            print(f"[diag] control candle used: {diag_candle['time'].isoformat()} "
                  f"close={diag_candle['close']}")
        print_diag_final_state(runtime, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
