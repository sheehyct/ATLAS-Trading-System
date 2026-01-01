# Sharing Codex Local Changes with Claude Code

Claude Code noted that the Codex timeframe continuity work (commit `913c314`) only exists locally. Use one of these options to make it visible:

## 1) Push the local commit to a remote branch
```bash
git push origin 913c314:codex/timeframe-continuity-adapter
```
- Creates branch `codex/timeframe-continuity-adapter` on the remote without changing `main`.
- Claude Code can then `git fetch origin codex/timeframe-continuity-adapter` and review.
- If you are working in the web Codex environment, this is the only way for Claude Code (or anyone else) to see the changes—web Codex workspaces are isolated until you push to GitHub.

## 2) Export and share a patch file
```bash
git format-patch -1 913c314 -o /tmp/
# share the generated 0001-*.patch file
```
- Lets collaborators apply the exact commit via `git apply 0001-*.patch` or `git am 0001-*.patch`.

## 3) Hand over the changed files
The Codex commit touched these files:
- `strat/timeframe_continuity_adapter.py` (new)
- `tests/test_strat/test_timeframe_continuity_adapter.py` (new)
- `strat/paper_signal_scanner.py`
- `crypto/scanning/signal_scanner.py`
- `crypto/simulation/paper_trader.py`
- `docs/git_actions.md` (new)

Share those files directly if pushing or patch exchange is unavailable.

## Quick review commands for Claude Code
Once the commit is accessible (via branch or patch):
```bash
# fetch the branch
git fetch origin codex/timeframe-continuity-adapter
# inspect the commit
git show 913c314
# apply locally (scratch branch)
git checkout -b review/codex-continuity FETCH_HEAD
```
These steps avoid modifying `main` while enabling review of the Codex changes.
