# 01: Git Repositories & Code Migration

## Pre-Migration (on LAPTOP -- do this FIRST)

### Push All Unpushed Branches

CRITICAL: feature/strategies-momentum has 25 unpushed commits.

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace

# Push all branches with unpushed work
git push origin feature/strategies-momentum
git push origin feature/codebase-audit
git push origin docs/atlas-story
git push origin feature/strategies-reversion
git push origin release/v1.0
```

### Commit or Stash Uncommitted Work

vectorbt-workspace main has 17 untracked items:
```
.claude/commands/
.claude/hooks/git_commit_guard.py.bak
.claude/hooks/vbt_workflow_guardian.py
Books/
crypto/monitoring/
crypto/statarb/ (multiple files)
crypto/trading/README.md
docs/ATLAS_TRADING_SYSTEM_STATUS_GUIDE.md
docs/CLAUDE_CODE_ARCHITECTURE_PLAN.md
docs/EQUITY-76_INVESTIGATION_FINDINGS.md
docs/worktree_templates/
output/
tests/test_crypto/test_monitoring/
```

clri has 1 modified file: `docs/SESSION_LOG.md`

### Clean Up Temp Directories

The momentum worktree has 20 `tmpclaude-*` temp directories:
```bash
cd C:\Strat_Trading_Bot\atlas-strat-momentum-20251229
# Review and delete temp dirs
dir tmpclaude-* /b
# Then remove them
```

---

## On Desktop -- Clone Repos

### Clone vectorbt-workspace
```bash
mkdir C:\Strat_Trading_Bot
cd C:\Strat_Trading_Bot
git clone https://github.com/sheehyct/ATLAS-Trading-System.git vectorbt-workspace
cd vectorbt-workspace
```

### Clone clri
```bash
cd C:\Strat_Trading_Bot
git clone https://github.com/sheehyct/Crypto-Leverage-Risk-Index.git clri
```

### Recreate Worktrees
```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace

git worktree add ../atlas-audit-20251229 feature/codebase-audit
git worktree add ../atlas-story-20251229 docs/atlas-story
git worktree add ../atlas-strat-momentum-20251229 feature/strategies-momentum
git worktree add ../atlas-strat-reversion-20251229 feature/strategies-reversion
```

### Checkout Additional Local Branches
```bash
git checkout -b claude/review-deephaven-dashboards-01D1zAN3QJ1q1WNat2airUtf origin/claude/review-deephaven-dashboards-01D1zAN3QJ1q1WNat2airUtf
git checkout -b codex/timeframe-continuity-adapter origin/codex/timeframe-continuity-adapter
git checkout -b release/v1.0 origin/release/v1.0
git checkout main
```

### Optimize Repo
```bash
# Pack loose objects (laptop had 143 MB of loose objects, 4256 count)
git gc --aggressive
```

---

## Verification Checks

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| VBT remote | `git remote -v` | origin = `https://github.com/sheehyct/ATLAS-Trading-System.git` |
| CLRI remote | `git -C ../clri remote -v` | origin = `https://github.com/sheehyct/Crypto-Leverage-Risk-Index.git` |
| Branch | `git branch --show-current` | `main` |
| Worktrees | `git worktree list` | 5 entries (main + 4 worktrees) |
| Submodules | `git submodule status` | Empty output (none) |
| Fetch works | `git fetch origin --dry-run` | No auth errors |
| CLRI clean | `git -C ../clri status` | Clean working tree |

## Repo Details

### vectorbt-workspace
- Remote: `https://github.com/sheehyct/ATLAS-Trading-System.git`
- Branches: 7 local, ~15 remote (many `claude/*` and `codex/*`)
- Worktrees: 4 (audit, story, momentum, reversion)
- No submodules, no LFS, no stashes
- `.gitignore`: 157 lines (Python, venvs, secrets, data, MCP servers, VBT Pro)
- `.gitattributes`: export-ignore rules for session docs, Books, output

### clri
- Remote: `https://github.com/sheehyct/Crypto-Leverage-Risk-Index.git`
- Branches: main only
- No worktrees, no submodules, no LFS, no stashes
- `.gitignore`: 79 lines
- Docker-based deployment (Hetzner EU VPS via Railway)

---
Generated: 2026-02-09
