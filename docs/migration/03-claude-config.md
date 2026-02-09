# 03: Claude Code Configuration Migration

## MCP Servers

### vectorbt-workspace (.mcp.json)

4 servers. Update paths marked with **[UPDATE]** on new machine.

#### vectorbt-pro (stdio)
- Command: **[UPDATE]** `C:\Strat_Trading_Bot\vectorbt-workspace\.venv\Scripts\python.exe`
- Args: `-m vectorbtpro.mcp_server`
- Depends on: VBT Pro installed in .venv

#### openmemory (stdio)
- Command: **[UPDATE]** `C:\Dev\openmemory\node_modules\.bin\tsx.cmd`
- Args: **[UPDATE]** `C:\Dev\openmemory\backend\src\mcp-stdio.ts`
- Depends on: Node.js, npm install in openmemory backend

#### playwright (stdio)
- Command: `cmd /c npx -y @playwright/mcp@latest`
- No path updates needed (npx fetches latest)

#### ThetaData (SSE)
- URL: `http://localhost:25503/mcp/sse`
- No path updates needed
- Requires: ThetaData desktop app running locally

### clri (.claude/mcp.json)

#### playwright (stdio)
- Command: `npx -y @playwright/mcp@latest`
- No path updates needed

#### openmemory (stdio)
- Command: **[UPDATE]** `C:\Program Files\nodejs\npm.cmd`
- Args: `run mcp`
- CWD: **[UPDATE]** `C:\Dev\openmemory-clri\backend`
- Port: 8081 (different from ATLAS's 8080)

---

## OpenMemory Configuration

**IMPORTANT:** After copying via flash drive, you must rebuild `node_modules/` in both
OpenMemory instances (`npm install` in each `backend/` dir). The native `sqlite3` module
has compiled binaries tied to the source machine. See 02-environment.md Phase 5 for details.

A previous bug where `console.log` in the `oninitialized` callback corrupted the stdio
MCP transport has been fixed in the source code. The fix (changed to `console.error`)
is already in `backend/src/mcp/index.ts` and will copy over with the flash drive.

### ATLAS Instance (.env at C:\Dev\openmemory\.env)
```
OM_PORT=8080
OM_DB_PATH=C:/Dev/openmemory/data/atlas_memory.sqlite   # [UPDATE if path changes]
OM_EMBEDDINGS=openai
OPENAI_API_KEY=sk-proj-...                                # [SENSITIVE - copy securely]
OM_METADATA_BACKEND=sqlite
OM_VECTOR_BACKEND=sqlite
OM_VEC_DIM=1536
OM_DECAY_LAMBDA=0.01
```

### CLRI Instance (.env at C:\Dev\openmemory-clri\.env)
```
OM_PORT=8081
OM_DB_PATH=C:/Dev/openmemory-clri/data/clri_memory.sqlite  # [UPDATE if path changes]
OPENAI_API_KEY=sk-proj-...                                   # [SENSITIVE - same key as ATLAS]
```

---

## Settings Files

### User-Level Settings (~/.claude/settings.json)

Copy from laptop. Current contents:
```json
{
  "includeCoAuthoredBy": false,
  "statusLine": {
    "type": "command",
    "command": "npx -y ccstatusline@latest",
    "padding": 0
  },
  "enabledPlugins": {
    "frontend-design-pro@frontend-design-pro": true,
    "pyright-lsp@claude-plugins-official": true,
    "code-review@claude-plugins-official": true,
    "pr-review-toolkit@claude-plugins-official": true,
    "feature-dev@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "context7@claude-plugins-official": true,
    "frontend-design@claude-plugins-official": true,
    "figma@claude-plugins-official": true
  },
  "alwaysThinkingEnabled": true,
  "gitAttribution": false,
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```
No hardcoded paths -- copy as-is.

### Project Settings (vectorbt-workspace/.claude/settings.json)

Contains hooks with **[UPDATE]** absolute paths:
```
python "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/strat_code_guardian.py"
python "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/vbt_workflow_guardian.py"
python "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/strat_prompt_validator.py"
```
These are in the committed settings.json. If the desktop uses the same `C:\Strat_Trading_Bot\` path, no changes needed.

### Local Settings (vectorbt-workspace/.claude/settings.local.json)

Contains ~137 permission allow rules with hardcoded paths. These will regenerate as you use Claude Code on the new machine. The important parts to preserve:
- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: "1"` in env
- `enabledMcpjsonServers` list
- Hook definitions (use relative paths in this file)

---

## Hooks (3 files in .claude/hooks/)

All are pure Python with no hardcoded paths internally.

| Hook | Trigger | Purpose |
|------|---------|---------|
| `strat_code_guardian.py` | PreToolUse (Write/Edit) | Blocks STRAT file edits without consulting /strat-methodology |
| `vbt_workflow_guardian.py` | PreToolUse (Write/Edit) | Enforces VBT 5-Step Workflow |
| `strat_prompt_validator.py` | UserPromptSubmit | Advises using /strat-methodology for STRAT prompts |
| `git_commit_guard.py.bak` | DISABLED | Was: run pytest before git commit |

These live in the git repo -- they migrate with the clone. No action needed.

---

## Skills (10 directories in ~/.claude/skills/)

Copy the entire `C:\Users\sheeh\.claude\skills\` directory to the new machine's equivalent path.

| Skill | Key Files |
|-------|-----------|
| strat-methodology | SKILL.md, PATTERNS.md, TIMEFRAMES.md, EXECUTION.md, OPTIONS.md, references/ |
| thetadata-api | SKILL.md, ENDPOINTS.md, ERRORS.md, DATA_AVAILABILITY.md, GOTCHAS.md |
| backtesting-validation | SKILL.md, ATLAS_BUGS.md |
| options-trading | SKILL.md, GREEKS_REFERENCE.md, ATLAS_OPTIONS.md |
| position-sizing-risk | SKILL.md, ATLAS_SIZING.md, FORMULAS.md |
| mean-reversion-strategies | SKILL.md |
| momentum-strategies | SKILL.md |
| execution-quality | SKILL.md |
| portfolio-construction | SKILL.md |
| tech-product-review-writer | SKILL.md, VOICE_EXAMPLES.md, STRUCTURE_GUIDE.md, + more |

**Hardcoded paths in skills to update:**
- `tech-product-review-writer/QUICK_START.md` lines 8 and 150: `C:\Users\sheeh\Documents\tech-product-review-writer\`

---

## Commands (6 files in .claude/commands/)

These live in the git repo. Files with hardcoded paths:

| Command | Hardcoded Paths to Update |
|---------|--------------------------|
| `session-start.md` | `/c/Users/sheeh/.claude/projects/C--Strat-Trading-Bot-vectorbt-workspace/*.jsonl` |
| `session-end.md` | `C:\Dev\openmemory\data\atlas_memory.sqlite` -> `C:\Dev\openmemory\backups\` |
| `tech-debt.md` | `C:\Users\sheeh\.claude\plans\sharded-foraging-puppy.md` |
| `pre-commit.md` | None |
| `pull-logs.md` | `atlas@178.156.223.251` (VPS IP -- stays the same) |
| `test-focus.md` | None |

---

## CLAUDE.md References

`docs/CLAUDE.md` references skill paths that include the username:
```
C:/Users/sheeh/.claude/skills/strat-methodology/
C:/Users/sheeh/.claude/skills/thetadata-api/
C:/Users/sheeh/.claude/skills/backtesting-validation/
```
**[UPDATE]** if desktop username differs from `sheeh`.

---

## Plans Directory

Copy `C:\Users\sheeh\.claude\plans\` (46 plan files) to the new machine.
These are auto-generated names (e.g., `sharded-foraging-puppy.md`).

---

## Status Line

Current config in user settings:
```json
"statusLine": {
  "type": "command",
  "command": "npx -y ccstatusline@latest",
  "padding": 0
}
```
No path updates needed -- npx handles it.

Backup exists at `.claude/statusline.sh.backup` (14 KB).

---

## Plugins (9 enabled, cloud-managed)

Plugins are managed by Claude Code and will re-enable on the new machine:
1. frontend-design-pro@frontend-design-pro
2. pyright-lsp@claude-plugins-official
3. code-review@claude-plugins-official
4. pr-review-toolkit@claude-plugins-official
5. feature-dev@claude-plugins-official
6. code-simplifier@claude-plugins-official
7. context7@claude-plugins-official
8. frontend-design@claude-plugins-official
9. figma@claude-plugins-official

No manual installation needed -- just ensure `enabledPlugins` is in user settings.

---

## Permission Rules

The `settings.local.json` has ~137 allow rules with paths like:
```
C:\Users\sheeh\...
C:\Strat_Trading_Bot\...
C:\Dev\...
atlas@178.156.223.251 (VPS -- IP stays the same)
root@46.225.51.247 (CLRI VPS -- IP stays the same)
```

**Recommendation:** Do NOT migrate these. Let them regenerate naturally as you use Claude Code on the desktop. They accumulate from approving tool calls.

---

## Summary: All Paths Requiring Update

### CRITICAL (MCP servers break without these):
1. `.mcp.json` vectorbt-pro python path
2. `.mcp.json` openmemory tsx command path
3. `.mcp.json` openmemory script arg path
4. `clri/.claude/mcp.json` npm.cmd path
5. `clri/.claude/mcp.json` openmemory-clri cwd
6. openmemory `.env` OM_DB_PATH
7. openmemory-clri `.env` OM_DB_PATH

### HIGH (hooks break):
8-10. Three hook paths in `settings.json` (only if base directory changes)

### MEDIUM (specific features):
11-15. Command files and CLAUDE.md skill references

### IF USERNAME CHANGES:
All `C:\Users\sheeh\` paths become `C:\Users\<new-username>\`

---
Generated: 2026-02-09
