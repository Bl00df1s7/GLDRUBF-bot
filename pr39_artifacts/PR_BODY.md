# chore: untrack committed .pyc caches and refresh stale G-5 comment after PR #37

Follow-up cleanup after merge of PR #37 (`open_orders` trustworthiness fix, merge commit `f515f2d`). No behavioral changes.

## What this PR does

1. **Untracks 50 committed `.pyc` files** (index-only removal via `git rm --cached`; local files on disk are untouched).
   - `config/__pycache__/` — 2 files
   - `scripts/__pycache__/` — 1 file
   - `src/__pycache__/` — 24 files
   - `src/strategies/__pycache__/` — 4 files
   - `tests/__pycache__/` — 19 files
   - Verified before removal: all 50 blobs in `main` carry the Python 3.12 bytecode magic number (`cb 0d 0d 0a`) — they are cached byte-code artifacts, not data, and are regenerated at import time. They are covered by the existing `.gitignore` rules (`__pycache__/`, `*.pyc`), so they will not reappear as untracked noise.
2. **Updates the stale G-5 comment in `src/c5_runtime.py`** (comments only, inside `reconcile()`):
   - Old text claimed the LIVE adapter "silently coerces a `get_orders()` exception into 0" — that defect was fixed by PR #37.
   - New text reflects actual post-#37 behavior: a `get_orders()` error now propagates out of `snapshot()`, startup reconcile **fails closed** (`STARTUP_RECONCILE_FAILED`, execution is not enabled), and a LIVE `open_orders == 0` is a confirmed absence of working orders.
   - The **EXIT_UNVERIFIED freeze rule is preserved unchanged**: a single snapshot without open orders is still not sufficient to prove terminality of the previous ambiguous SELL (`SimulatedBroker.snapshot()` always reports 0; the adapter contract carries no terminal order status). The freeze is lifted only by explicit operator confirmation. Executable logic is byte-for-byte identical (verified by AST comparison with comments stripped).

## Explicitly NOT changed

- `.gitignore` — restored to exact `main` state; it does **not** appear in the diff. Rules `__pycache__/`, `*.pyc`, `.pytest_cache/` were already present in `main` and remain in force.
- `src/execution_adapter.py` — empty diff (the PR #37 fix is untouched).
- C5 strategy and parameters, `reconcile()` executable code, ENTRY/risk logic, `PaperExecutionAdapter`, `SimulatedBroker` — no changes.

## Diff summary

```
51 files changed, 11 insertions(+), 5 deletions(-)
  50 × D  *.pyc            (untracked from git, removed from index only)
   1 × M  src/c5_runtime.py (comment block in reconcile(), +11/-5 lines)
```

`.gitignore`: `git diff main -- .gitignore` is empty (blob hash `75c61823…` identical to `main`).

## Tests

Environment: Python 3.12.10, pytest 9.1.1.

- Targeted: `python -m pytest tests/test_open_orders_trust.py -p no:libtmux -q` → **10 passed**, exit code 0.
- Regression control on modules touched by the comment change
  (`AUTO_TRADING_ENABLED=true TRADING_MODE=PAPER python -m pytest tests/test_stage5b_paper_runtime.py tests/test_stage2d_runtime.py -p no:libtmux -q`):
  baseline worktree at `f515f2d` → 31 failed / 60 passed; this branch → **identical FAILED set (diff empty)** → **0 regressions**. The 31 failures are pre-existing sandbox-environment issues, unrelated to this PR.
- Known environment limitation (not a code defect): `tests/test_stop_orders.py` cannot be collected here due to the missing private package `t_tech` (ModuleNotFoundError); excluded from runs.

## Safety

No LIVE operations, no history rewrite, no merges. Pure hygiene change: git index cleanup + comment refresh.
