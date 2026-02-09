# 04: Post-Migration Validation (46 Checks)

Run this AFTER completing 01-git-repos, 02-environment, and 03-claude-config.

---

## Phase 1: System Prerequisites (8 checks)

### 1.1 Python 3.12
```
python --version
```
PASS: `Python 3.12.x`

### 1.2 uv
```
uv --version
```
PASS: `uv 0.9.x` or newer

### 1.3 Node.js
```
node --version
```
PASS: `v22.x.x`

### 1.4 npm
```
npm --version
```
PASS: `10.x.x`

### 1.5 Git
```
git --version
```
PASS: `git version 2.x`

### 1.6 Docker
```
docker --version && docker compose version
```
PASS: Both return version strings

### 1.7 TA-LIB C Library
```
dir C:\ta-lib\c\lib\ta_lib.lib
dir C:\ta-lib\c\include\ta_libc.h
```
PASS: Both files exist

### 1.8 Claude Code
```
claude --version
```
PASS: Returns version string

---

## Phase 2: Repository Setup (5 checks)

### 2.1 vectorbt-workspace remote and branch
```
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" remote -v
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" branch --show-current
```
PASS: origin = `https://github.com/sheehyct/ATLAS-Trading-System.git`, branch = `main`

### 2.2 Worktrees
```
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" worktree list
```
PASS: 5 entries (main + 4 worktrees: audit, story, momentum, reversion)

### 2.3 CLRI remote and branch
```
git -C "C:/Strat_Trading_Bot/clri" remote -v
git -C "C:/Strat_Trading_Bot/clri" branch --show-current
```
PASS: origin = `https://github.com/sheehyct/Crypto-Leverage-Risk-Index.git`, branch = `main`

### 2.4 No submodules
```
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" submodule status
```
PASS: Empty output

### 2.5 Repo integrity
```
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" fsck --no-dangling 2>&1 | head -5
```
PASS: No errors (dangling warnings OK)

---

## Phase 3: Environment Setup (7 checks)

### 3.1 uv sync
```
cd C:\Strat_Trading_Bot\vectorbt-workspace && uv sync
```
PASS: Exit code 0, VBT Pro installs without auth failures

### 3.2 VectorBT Pro
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "import vectorbtpro; print('VBT Pro version:', vectorbtpro.__version__)"
```
PASS: Prints version (e.g., `2025.12.31`)

### 3.3 TA-LIB Python wrapper
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "import talib; print('TA-Lib version:', talib.__version__)"
```
PASS: Prints `0.6.x` without DLL errors

### 3.4 Key packages
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "import alpaca; import dash; import pandas; import numpy; import plotly; import openai; import scipy; import optuna; print('All key packages OK')"
```
PASS: Prints `All key packages OK`

### 3.5 .env file has required keys
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "from dotenv import dotenv_values; env = dotenv_values('.env'); missing = [k for k in ['ALPACA_API_KEY','ALPACA_SECRET_KEY','GITHUB_TOKEN','VECTORBT_TOKEN'] if not env.get(k)]; print('PASS' if not missing else f'MISSING: {missing}')"
```
PASS: Prints `PASS`

### 3.6 GITHUB_TOKEN and VECTORBT_TOKEN accessible
```
python -c "import os; print('GITHUB_TOKEN:', 'SET' if os.environ.get('GITHUB_TOKEN') else 'MISSING'); print('VECTORBT_TOKEN:', 'SET' if os.environ.get('VECTORBT_TOKEN') else 'MISSING')"
```
PASS: Both show `SET`

### 3.7 Package count
```
uv pip list 2>&1 | find /c /v ""
```
PASS: ~350+ packages

---

## Phase 4: MCP Server Health (5 checks)

### 4.1 MCP server list
```
claude mcp list
```
PASS: Lists vectorbt-pro, openmemory, playwright, ThetaData

### 4.2 vectorbt-pro MCP (inside Claude Code)
```
mcp__vectorbt-pro__resolve_refnames(["vbt.Portfolio"])
```
PASS: Returns `OK vbt.Portfolio vectorbtpro.portfolio.base.Portfolio`

### 4.3 OpenMemory MCP
```
dir C:\Dev\openmemory\data\atlas_memory.sqlite
```
PASS: File exists. Then inside Claude Code:
```
mcp__openmemory__openmemory_list(limit=3)
```
PASS: Returns memories without errors

### 4.4 Playwright MCP (inside Claude Code)
```
mcp__playwright__browser_navigate(url="https://www.google.com")
mcp__playwright__browser_snapshot()
```
PASS: Snapshot returns page content

### 4.5 ThetaData MCP (conditional)
```
curl -s http://localhost:25503/v2/config 2>&1 | head -5
```
PASS: Returns JSON. SKIP if ThetaData desktop app not running.

---

## Phase 5: Claude Code Configuration (7 checks)

### 5.1 Hook files exist
```
python -c "import os; hooks=['strat_code_guardian.py','strat_prompt_validator.py','vbt_workflow_guardian.py']; [print(f'  {h}:', 'EXISTS' if os.path.isfile(f'C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/{h}') else 'MISSING') for h in hooks]"
```
PASS: All 3 show EXISTS

### 5.2 Hooks compile
```
python -m py_compile "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/strat_code_guardian.py"
python -m py_compile "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/strat_prompt_validator.py"
python -m py_compile "C:/Strat_Trading_Bot/vectorbt-workspace/.claude/hooks/vbt_workflow_guardian.py"
```
PASS: All exit code 0

### 5.3 Hook config in settings.json
```
python -c "import json; cfg=json.load(open('C:/Strat_Trading_Bot/vectorbt-workspace/.claude/settings.json')); h=cfg.get('hooks',{}); print(f'PreToolUse: {len(h.get(\"PreToolUse\",[]))}'); print(f'PostToolUse: {len(h.get(\"PostToolUse\",[]))}'); print(f'UserPromptSubmit: {len(h.get(\"UserPromptSubmit\",[]))}')"
```
PASS: PreToolUse: 2, PostToolUse: 1, UserPromptSubmit: 1

### 5.4 Skills exist (top 3)
```
python -c "import os; skills=['strat-methodology','thetadata-api','backtesting-validation']; [print(f'  {s}:', 'EXISTS' if os.path.isfile(os.path.expanduser(f'~/.claude/skills/{s}/SKILL.md')) else 'MISSING') for s in skills]"
```
PASS: All 3 show EXISTS

### 5.5 Skills directory count
```
python -c "import os; d=os.path.expanduser('~/.claude/skills'); dirs=[x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))]; print(f'Skills: {len(dirs)}'); print('PASS' if len(dirs)>=10 else 'FAIL')"
```
PASS: >= 10 directories

### 5.6 Plugins
```
claude plugin list 2>&1
```
PASS: Shows 9+ plugins

### 5.7 Commands exist
```
python -c "import os; d='C:/Strat_Trading_Bot/vectorbt-workspace/.claude/commands'; cmds=[f for f in os.listdir(d) if f.endswith('.md')]; print(f'Commands: {len(cmds)}'); print('PASS' if len(cmds)>=5 else 'FAIL')"
```
PASS: >= 5 commands

---

## Phase 6: Functional Tests (7 checks)

### 6.1 Full test suite
```
cd C:\Strat_Trading_Bot\vectorbt-workspace && uv run pytest tests/ -x -q --timeout=120 2>&1 | tail -20
```
PASS: Zero failures

### 6.2 Signal automation tests
```
cd C:\Strat_Trading_Bot\vectorbt-workspace && uv run pytest tests/test_signal_automation/ -x -q --timeout=120 2>&1 | tail -20
```
PASS: Zero failures

### 6.3 Alpaca API connectivity
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "
from dotenv import load_dotenv; load_dotenv('.env')
import os; from alpaca.trading.client import TradingClient
client = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=True)
acct = client.get_account()
print(f'Alpaca: {acct.status}, buying power: \${float(acct.buying_power):,.2f}')
print('PASS')
"
```
PASS: Shows ACTIVE status

### 6.4 ATLAS VPS SSH
```
ssh -o ConnectTimeout=10 -o BatchMode=yes atlas@178.156.223.251 "echo ATLAS_VPS_OK && uptime"
```
PASS: Prints `ATLAS_VPS_OK`. Requires SSH key setup on new machine.

### 6.5 CLRI VPS SSH
```
ssh -o ConnectTimeout=10 -o BatchMode=yes root@46.225.51.247 "echo CLRI_VPS_OK && uptime"
```
PASS: Prints `CLRI_VPS_OK`

### 6.6 ATLAS daemon status
```
ssh atlas@178.156.223.251 "sudo systemctl status atlas-daemon --no-pager | head -10"
```
PASS: Shows `active (running)`

### 6.7 Tiingo API
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "
from dotenv import load_dotenv; load_dotenv('.env')
import os, requests
r = requests.get('https://api.tiingo.com/api/test', headers={'Authorization': f'Token {os.environ.get(\"TIINGO_API_KEY\",\"\")}'})
print(f'Tiingo: {r.status_code}'); print('PASS' if r.status_code==200 else 'FAIL')
"
```
PASS: Status 200

---

## Phase 7: End-to-End Validation (7 checks)

### 7.1 STRAT scan (inside Claude Code)
```
mcp__claude_ai_STRAT_Stock_Scanner__analyze_strat_patterns(ticker="SPY", timeframe="1Day", days_back=5)
```
PASS: Returns STRAT pattern analysis with candle types

### 7.2 VBT Pro run_code (inside Claude Code)
```
mcp__vectorbt-pro__run_code(code="import vectorbtpro as vbt; print('VBT:', vbt.__version__); data = vbt.YFData.pull('SPY', period='5d'); print('Shape:', data.data['SPY'].shape); print('PASS')")
```
PASS: Prints version, data shape, and PASS

### 7.3 OpenMemory query (inside Claude Code)
```
mcp__openmemory__openmemory_query(query="ATLAS trading system", k=3)
```
PASS: Returns relevant memories

### 7.4 OpenMemory round-trip (inside Claude Code)
```
mcp__openmemory__openmemory_store(content="Migration validation test", tags=["migration-test"])
# Then:
mcp__openmemory__openmemory_query(query="Migration validation test", k=1)
```
PASS: Stored memory is retrievable

### 7.5 Dashboard import
```
C:/Strat_Trading_Bot/vectorbt-workspace/.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'C:/Strat_Trading_Bot/vectorbt-workspace')
from dashboard.app import app
print(f'Dashboard: {type(app).__name__}')
print('PASS' if 'Dash' in type(app).__name__ else 'FAIL')
"
```
PASS: Shows `Dashboard: Dash`

### 7.6 CLRI Docker (on VPS)
```
ssh root@46.225.51.247 "docker ps --format 'table {{.Names}}\t{{.Status}}' | head -5"
```
PASS: Shows running containers

### 7.7 Git fetch (auth check)
```
git -C "C:/Strat_Trading_Bot/vectorbt-workspace" fetch origin --dry-run 2>&1
```
PASS: No auth errors

---

## Scorecard

| Phase | Checks | Critical |
|-------|--------|----------|
| 1. Prerequisites | 8 | YES |
| 2. Repo Setup | 5 | YES |
| 3. Environment | 7 | YES |
| 4. MCP Servers | 5 | MOSTLY (ThetaData conditional) |
| 5. Claude Config | 7 | YES |
| 6. Functional Tests | 7 | YES |
| 7. End-to-End | 7 | YES |
| **TOTAL** | **46** | |

Run Phases 1-3 first (sequential). Phases 4-5 can run in parallel. Phase 6 needs 1-3 done. Phase 7 needs all prior phases.

---
Generated: 2026-02-09
