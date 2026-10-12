# How to publish PR #39 (branch `audit-pr-37-open-orders-a9bcd`)

The correct final state is commit **`b4dc9d5`** (single parent `f515f2d` = main).
The previously pushed heads `5d9ff7f`, `a2bc86f`, `ce92268` are superseded and must
be replaced (force-with-lease) — this does NOT rewrite `main` history.

## Option A — push the prepared branch from a clone that has credentials

```bash
cd <your-clone-of-GLDRUBF-bot>
git fetch <sandbox-or-source-repo> b4dc9d5   # or copy the repo
git checkout -B audit-pr-37-open-orders-a9bcd b4dc9d5
git push --force-with-lease origin audit-pr-37-open-orders-a9bcd
```

GitHub will update existing PR #39 automatically (same head branch name).

## Option B — apply the patch file in your local clone

```bash
git checkout -B audit-pr-37-open-orders-a9bcd f515f2d
git am pr39_artifacts/0001-chore-untrack-committed-.pyc-caches-and-refresh-stal.patch
# verify:
git ls-files '*.pyc' | wc -l          # -> 0
git diff f515f2d HEAD --stat | tail -1 # -> 51 files changed, 11 insertions(+), 5 deletions(-)
git diff f515f2d HEAD -- .gitignore    # -> empty
git push --force-with-lease origin audit-pr-37-open-orders-a9bcd
```

## Post-publish checks

```bash
gh pr view 39 --json headRefOid,files --jq '.headRefOid'   # expect new SHA of b4dc9d5 content
gh pr diff 39 | grep -E '^(diff|deleted)' | grep -c 'pyc'  # expect 50 deletions
gh pr diff 39 | grep -c '^diff --git'                       # expect 51 files total
gh pr diff 39 | grep -c 'gitignore'                         # expect 0
```

## Contents of this artifacts folder

| File | Purpose |
|---|---|
| `0001-chore-untrack-committed-.pyc-caches-and-refresh-stal.patch` | Full commit as mailbox patch (`git am`-ready, includes binary blobs) |
| `pr39.diff` | Plain diff vs main (review / `git apply`) |
| `changed_files.txt` | name-status list of all 51 files |
| `diff_stat.txt` | Final diffstat summary |
| `pyc_deleted_50.txt` | Exact list of the 50 `.pyc` removed (matches main's tracked set) |
| `c5_runtime_comment.diff` | Comment-only hunk for focused review |
| `gitignore_diff_vs_main.txt` | Empty file — proof `.gitignore` is unchanged |
| `PR_BODY.md` | Ready-to-paste PR description |

Note: `pr39_artifacts/` itself is intentionally NOT part of the PR — it is not
tracked in commit `b4dc9d5` and does not appear in the diff.
