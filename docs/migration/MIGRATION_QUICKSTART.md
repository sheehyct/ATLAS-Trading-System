# Desktop Migration Quick Start

This is your personal checklist for migrating ATLAS + CLRI to the desktop.
The migration docs are in this directory. Claude Code can execute them using agent teams.

---

## What You Need Before Starting

### Already Done (on laptop):
- [x] All branches pushed to GitHub (0 unpushed commits)
- [x] All repos clean (no uncommitted changes)
- [x] Migration docs committed and pushed
- [x] Temp directories cleaned up

### On Desktop (you said this is ready):
- [x] Hardware set up
- [ ] Claude Code installed

### Manual Transfers (USB or secure copy):
These files are NOT in git and must be copied manually:

| File | Location on Laptop | Why |
|------|-------------------|-----|
| vectorbt-workspace .env | `C:\Strat_Trading_Bot\vectorbt-workspace\.env` | All API keys (Alpaca, ThetaData, Tiingo, Coinbase, OpenAI, GitHub, VBT Pro) |
| clri .env | `C:\Strat_Trading_Bot\clri\.env` | Database URL, Discord tokens, Coinglass key |
| OpenMemory ATLAS .env | `C:\Dev\openmemory\.env` | OpenAI API key |
| OpenMemory CLRI .env | `C:\Dev\openmemory-clri\.env` | OpenAI API key |
| OpenMemory DB | `C:\Dev\openmemory\data\atlas_memory.sqlite` | 16.5 MB memory database |
| OpenMemory backups | `C:\Dev\openmemory\backups\` | ~300 MB of backup history |
| SSH keys | `C:\Users\sheeh\.ssh\` | VPS access (atlas@178.156.223.251, root@46.225.51.247) |
| Books directory | `C:\Strat_Trading_Bot\vectorbt-workspace\Books\` | 516 MB, gitignored |
| TA-LIB C library | `C:\ta-lib\` | Or re-download from SourceForge (easier) |
| Claude skills | `C:\Users\sheeh\.claude\skills\` | 10 skill directories |
| Claude plans | `C:\Users\sheeh\.claude\plans\` | 46 plan files |
| Claude settings | `C:\Users\sheeh\.claude\settings.json` | User-level config |

### TA-LIB (the "hard" one that's actually easy):
1. Download `ta-lib-0.4.0-msvc.zip` from SourceForge
2. Extract to `C:\ta-lib`
3. Done. `uv sync` handles the Python wrapper.

---

## IMPORTANT: .venv Directory

The workspace was copied from another machine. The `.venv/` directory contains compiled
binaries (C extensions, .pyd files) tied to the source machine and WILL NOT WORK here.

**Before running Claude Code or uv sync:**
```powershell
cd C:\Strat_Trading_Bot\vectorbt-workspace
rmdir /s /q .venv
```

Then `uv sync` will create a fresh .venv with binaries compiled for this machine.
This is critical -- TA-LIB, numpy, scipy, and other C-extension packages will segfault
or fail to import if you try to use the copied .venv.

---

## Running the Migration with Claude Code

After installing Claude Code on the desktop, open a terminal in the vectorbt-workspace
directory and start Claude Code. Then paste this prompt:

```
I'm migrating this workspace from my laptop to this desktop. The files were
copied via flash drive. The migration documentation is in docs/migration/.
Please read all 5 files:

- docs/migration/README.md (overview and execution order)
- docs/migration/01-git-repos.md (repo cloning and worktree setup)
- docs/migration/02-environment.md (Python, uv, TA-LIB, dependencies)
- docs/migration/03-claude-config.md (MCP servers, hooks, skills, plugins, path updates)
- docs/migration/04-validation.md (46-check smoke test plan)

CRITICAL FIRST STEP: The .venv directory was copied from the old machine and
contains incompatible compiled binaries. Delete it and rebuild:
  rmdir /s /q .venv
  uv sync

Create an agent team with 4 teammates to execute the migration:

1. "git-setup" -- Execute 01-git-repos.md: verify worktrees are intact (they were
   copied, not cloned), verify remotes, verify all branches are in sync with GitHub

2. "env-setup" -- Execute 02-environment.md: DELETE the old .venv first, then verify
   system tools are installed, run uv sync to rebuild from scratch, verify VBT Pro
   and TA-LIB imports, check .env keys are present

3. "config-setup" -- Execute 03-claude-config.md: verify MCP server configs have
   correct paths for this machine, verify hooks exist and compile, verify skills
   are in place, check for any hardcoded paths that need updating

4. "validator" -- Execute 04-validation.md: run the full 46-check smoke test plan
   AFTER the other 3 teammates finish. This is blocked by tasks 1-3.

Require plan approval before any teammate makes changes. Use delegate mode.
Each teammate should report exactly which checks passed, failed, or were skipped.
```

### If paths are different on the desktop:
The migration docs assume the same directory structure as the laptop:
- `C:\Strat_Trading_Bot\` for repos
- `C:\Dev\openmemory\` for OpenMemory
- `C:\ta-lib\` for TA-LIB
- `C:\Users\<username>\.claude\` for Claude config

If your desktop uses different paths, tell Claude:
```
Note: On this machine the base paths are different:
- Repos are at: D:\Strat_Trading_Bot\ (instead of C:\)
- OpenMemory is at: D:\Dev\openmemory\
- My username is <new-username> (instead of sheeh)
Update all path references in the migration docs accordingly before executing.
```

---

## After Migration

Once all 46 validation checks pass:

1. **Delete the laptop copies** (optional, but prevents confusion about which is the "real" one)
2. **Update OpenMemory** with a note about the migration:
   ```
   Store in OpenMemory: "ATLAS workspace migrated from laptop to desktop on
   YYYY-MM-DD. All repos, environments, MCP servers, and configs verified working."
   ```
3. **Run a real STRAT scan** to confirm the full pipeline works end-to-end
4. **SSH to both VPS servers** to confirm connectivity from the new machine

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv sync` fails on VBT Pro | Check GITHUB_TOKEN is set in environment (not just .env) |
| TA-LIB import fails | Verify `C:\ta-lib\c\lib\ta_lib.lib` exists, check 64-bit match |
| MCP server won't start | Check .mcp.json paths match actual locations on this machine |
| OpenMemory connection error | Verify `npm install` ran in backend dir, check OM_DB_PATH in .env |
| SSH to VPS fails | Copy ~/.ssh/ from laptop, check `ssh-add` |
| Hooks don't fire | Check settings.json hook paths match new machine paths |
| Plugins missing | Run `claude plugin list` -- they should auto-enable from settings |
| Permission prompts flood | Normal on new machine. Approve as needed, they accumulate. |

---
Generated: 2026-02-09
