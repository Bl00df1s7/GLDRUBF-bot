# chore: untrack committed .pyc caches and refresh stale G-5 comment after PR #37

Supersedes the broken intermediate state (`.gitignore` emptied, 47 `.pyc` re-added,
`pr39_artifacts/` accidentally committed). This branch contains exactly one commit
on top of `main` (`f515f2d`).

## What is in this PR (final diff vs main)

```
51 files changed, 11 insertions(+), 5 deletions(-)
  50 × D  *.pyc   — removed from Git tracking only (files on disk untouched)
   1 × M  src/c5_runtime.py — comment-only change (stale G-5 note)
```

### 1. Untracking of `.pyc` (50 files)
- The list of deleted files matches **exactly** the 50 tracked `.pyc` present in
  `main` (`git ls-tree -r f515f2d --name-only | grep '\.pyc$'`) — verified by
  sorted-list diff: no extras, none missing.
- All 50 blobs were verified to carry the Python 3.12 bytecode magic number
  (`importlib.util.MAGIC_NUMBER`) — they are compiled caches, not data files,
  regenerated automatically at import time; nothing depends on them at runtime.
- Removed from the index only (`git rm --cached`); local working copies on disk
  were never deleted.
- After this change `git ls-files '*.pyc'` returns **0**.

### 2. `.gitignore` — NOT modified
- `git diff f515f2d HEAD -- .gitignore` is empty; blob hash identical to main
  (`75c61823b87164dde85ad0f2c78c300e011cc8a2`).
- Rules preserved: line 1 `__pycache__/`, line 2 `*.pyc`, line 3 `.pytest_cache/`.

### 3. `src/c5_runtime.py` — G-5 comment refresh (comment-only)
The stale claim that the LIVE adapter "silently coerces a get_orders() exception
into 0" was fixed by PR #37 and is replaced with the actual post-#37 behaviour:
- a `get_orders()` error now propagates out of `snapshot()`;
- startup reconcile **FAILS CLOSED** (`STARTUP_RECONCILE_FAILED`, execution not enabled);
- a LIVE `open_orders == 0` is therefore a confirmed absence of working orders.

The EXIT_UNVERIFIED freeze rule is **kept unchanged**: a single snapshot without
open orders is still not sufficient to prove terminality of the previous ambiguous
SELL (SimulatedBroker always reports 0), so the freeze lifts only on explicit
operator confirmation. Executable code is byte-identical to main (AST comparison
after stripping comments/docstrings passes).

## Explicitly NOT touched
`src/execution_adapter.py`, C5 strategy and parameters, `reconcile()` logic,
ENTRY/risk logic, `PaperExecutionAdapter`, `SimulatedBroker`. No executable
statement changed anywhere in this PR.

## Tests
Environment: Python 3.12.10, pytest 9.1.1.

- `python -m pytest tests/test_open_orders_trust.py -p no:libtmux -q` →
  **10 passed**, exit code 0.
- Regression control on affected modules (`test_stage5b_paper_runtime.py`,
  `test_stage2d_runtime.py`, PAPER env): failed-set identical to baseline
  `f515f2d` worktree → **0 regressions** (pre-existing sandbox failures only).
- Limitation: `tests/test_stop_orders.py` cannot be collected without the private
  `t_tech` package (environment issue, not a code defect).

## Safety
No LIVE operations. No history rewrite of `main`. Merge is up to maintainer review.
