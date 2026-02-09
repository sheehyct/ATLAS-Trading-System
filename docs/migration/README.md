# ATLAS + CLRI Desktop Migration Guide

Migration from laptop to desktop for both workspaces.

## Workspaces

| Workspace | Repo | Size |
|-----------|------|------|
| vectorbt-workspace | https://github.com/sheehyct/ATLAS-Trading-System.git | ~3.5 GB (133 MB git, 2.6 GB venv, 516 MB Books) |
| clri | https://github.com/sheehyct/Crypto-Leverage-Risk-Index.git | ~6.2 MB |

## Execution Order

```
Phase 0: Pre-migration (on LAPTOP)
  - Push all unpushed branches
  - Commit/stash uncommitted work
  - Copy secrets (.env files) securely

Phase 1-3: Run in PARALLEL on desktop
  01-git-repos.md     -> Clone repos, recreate worktrees
  02-environment.md   -> Install Python, uv, TA-LIB, Node, dependencies
  03-claude-config.md -> Copy/update MCP servers, hooks, skills, plugins

Phase 4: Run AFTER 1-3 complete
  04-validation.md    -> 46-check smoke test plan
```

## Agent Team Usage

On the new desktop, spin up a migration team:
```
Create a migration team with 4 agents. Each reads and executes their
assigned checklist from docs/migration/:
- Agent 1: 01-git-repos.md
- Agent 2: 02-environment.md
- Agent 3: 03-claude-config.md
- Agent 4: 04-validation.md (blocked by agents 1-3)
Require plan approval before making changes.
```

## Critical Secrets (manual transfer only)

These files contain API keys and must be transferred securely (USB, encrypted transfer, etc.):

| File | Contains |
|------|----------|
| `vectorbt-workspace/.env` | Alpaca (3 accounts), ThetaData, Tiingo, AlphaVantage, Coinbase, OpenAI, GitHub, VBT Pro tokens |
| `clri/.env` | Database URL, Discord tokens, Coinglass API key |
| `C:\Dev\openmemory\.env` | OpenAI API key, DB path |
| `C:\Dev\openmemory-clri\.env` | OpenAI API key, DB path |

## Data to Copy (not in git)

| Source | Size | Notes |
|--------|------|-------|
| `C:\Dev\openmemory\data\atlas_memory.sqlite` | 16.5 MB | OpenMemory database |
| `C:\Dev\openmemory\backups\` | ~300 MB | 23 backup files |
| `vectorbt-workspace\Books\` | 516 MB | Export-ignored, not in git |
| SSH keys (`~\.ssh\`) | varies | Needed for VPS access |

## Items That Do NOT Need Migration (regenerable)

- `.venv/` directories (recreated by `uv sync`)
- `data_cache/`, `logs/`, `backtest_results/`
- `.playwright-mcp/` (reinstalled via `playwright install`)
- `node_modules/` in openmemory instances (reinstalled via `npm install`)
- Permission rules in `settings.local.json` (regenerate as you use tools)

---
Generated: 2026-02-09
