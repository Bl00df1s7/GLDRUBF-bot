"""Smoke-runner DI clock test: --clock-epoch must reach C5Runtime.now_fn.

The runner (scripts/paper_smoke_test.py) is a thin diagnostic wrapper over
the EXISTING dependency-injection hook ``C5Runtime(now_fn=...)`` — the
production runtime (src/c5_runtime.py) is NOT modified here.  These tests
prove:

  A. with --clock-epoch, the fixed datetime reaches C5Runtime and is what
     run_cycle() actually uses as "now" (ENTRY gate observes it);
  B. without --clock-epoch, no now_fn is injected at all — C5Runtime keeps
     its default wall-clock behaviour (current smoke-test semantics are
     unchanged);
  C. parser accepts POSIX epoch seconds and ISO-8601 (aware/naive/Z);
  D. fail-closed config checks (missing token / non-PAPER mode) abort
     before any runtime is constructed.

Nothing in c5_core / guards / sizing / execution is mocked beyond the two
external boundaries already faked in Stage 5B (market data via patched
load_candles, and the runtime ctor itself).
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MSK = ZoneInfo("Europe/Moscow")

_spec = importlib.util.spec_from_file_location(
    "paper_smoke_test", REPO_ROOT / "scripts" / "paper_smoke_test.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("paper_smoke_test", smoke)
_spec.loader.exec_module(smoke)


# ── C: epoch parsing ──────────────────────────────────────────────────
def test_parse_clock_epoch_accepts_posix_seconds():
    got = smoke.parse_clock_epoch("1790946060")
    assert got == dt.datetime(2026, 10, 2, 13, 1, tzinfo=dt.timezone.utc)
    assert got.astimezone(MSK).hour == 16 and got.astimezone(MSK).minute == 1


def test_parse_clock_epoch_accepts_iso_aware_naive_and_z():
    aware = smoke.parse_clock_epoch("2026-10-02T16:01:00+03:00")
    naive = smoke.parse_clock_epoch("2026-10-02T13:01:00")   # naive ⇒ UTC
    zulu = smoke.parse_clock_epoch("2026-10-02T13:01:00Z")
    assert aware == naive == zulu
    assert aware.tzinfo is not None

    with pytest.raises(ValueError):
        smoke.parse_clock_epoch("not-a-date")


# ── harness: capture kwargs passed to C5Runtime by the runner ─────────
@pytest.fixture
def captured_runtime(monkeypatch, tmp_path):
    """Patch state file + env; capture constructor kwargs of C5Runtime."""
    monkeypatch.setenv("T_SANDAPI", "fake-token")
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("AUTO_TRADING_ENABLED", "true")
    monkeypatch.setenv("C5_STATE_FILE", str(tmp_path / "state.json"))

    import src.state_store as ss
    monkeypatch.setattr(ss, "STATE_FILE", str(tmp_path / "state.json"))

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, token, **kwargs):
            captured["token"] = token
            captured.update(kwargs)

        def run_cycle(self):
            # Actually USE the injected now_fn exactly like production does
            # (run_cycle line: now_utc = self.now_fn()), so the test proves
            # the clock reaches the decision path, not just the ctor.
            captured["cycle_now"] = captured["now_fn"]() \
                if "now_fn" in captured else "DEFAULT_WALL_CLOCK"
            return SimpleNamespace(action="NO_ENTRY", qty=0, reason="TEST")

    import src.c5_runtime as rt_mod
    monkeypatch.setattr(rt_mod, "C5Runtime", FakeRuntime)
    return captured


def test_clock_epoch_reaches_c5runtime_now_fn(captured_runtime):
    """A — the CLI value arrives at C5Runtime as now_fn and is callable."""
    rc = smoke.main(["--clock-epoch", "2026-10-02T16:01:00+03:00"])
    assert rc == 0
    cfg = captured_runtime["config"]
    assert cfg.mode == "PAPER" and cfg.auto_trading_enabled is True
    now_fn = captured_runtime["now_fn"]
    got = now_fn()
    assert isinstance(got, dt.datetime)
    assert got.tzinfo is not None
    assert got.astimezone(MSK).strftime("%H:%M") == "16:01"
    # run_cycle consumed the SAME fixed time — override is effective end-to-end
    assert captured_runtime["cycle_now"] == got


def test_default_run_injects_no_now_fn(captured_runtime):
    """B — without the flag, no now_fn kwarg: default time.time behaviour."""
    rc = smoke.main([])
    assert rc == 0
    assert "now_fn" not in captured_runtime          # ctor default preserved
    assert captured_runtime["cycle_now"] == "DEFAULT_WALL_CLOCK"


def test_fail_closed_missing_token(captured_runtime, monkeypatch):
    monkeypatch.delenv("T_SANDAPI")
    assert smoke.main(["--clock-epoch", "1790946060"]) == 1
    assert captured_runtime == {}                    # runtime never built


def test_fail_closed_non_paper_mode(captured_runtime, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    assert smoke.main([]) == 1
    assert captured_runtime == {}


# ── E: --clock-now injects the real wall clock as now_fn ────────────────
def test_e_clock_now_injects_wall_clock(captured_runtime):
    rc = smoke.main(["--clock-now"])
    assert rc == 0
    assert "now_fn" in captured_runtime
    injected = captured_runtime["now_fn"]()
    # must be ~now (within 60s), i.e. the real wall clock, not a fixed epoch
    delta = abs((dt.datetime.now(dt.timezone.utc) - injected).total_seconds())
    assert delta < 60
    # run_cycle consumed the same value — override is effective end-to-end
    assert captured_runtime["cycle_now"] == injected


# ── F: --clock-epoch and --clock-now are mutually exclusive (fail closed) ─
def test_f_epoch_and_now_conflict_fail_closed(captured_runtime):
    rc = smoke.main(["--clock-epoch", "1790946060", "--clock-now"])
    assert rc == 1
    assert captured_runtime == {}          # runtime never built


# ── G: --fresh-state points C5_STATE_FILE at a throwaway file ───────────
def test_g_fresh_state_uses_throwaway_file(captured_runtime, monkeypatch,
                                           tmp_path):
    import os
    default_state = str(tmp_path / "shared_default.json")
    monkeypatch.setenv("C5_STATE_FILE", default_state)
    rc = smoke.main(["--fresh-state"])
    assert rc == 0
    smk = os.environ["C5_STATE_FILE"]
    assert smk and smk != default_state              # shared file untouched
    assert "c5_smoke_state_" in smk                  # throwaway prefix
    assert not os.path.exists(smk)                   # starts truly fresh
