# 02: Environment, Dependencies & TA-LIB

## Phase 1: System Software

Install in this order:

### 1.1 Git for Windows
```powershell
winget install Git.Git
git --version
# PASS: git version 2.x
```

### 1.2 Python 3.12
```powershell
winget install Python.Python.3.12
python --version
# PASS: Python 3.12.x
```

### 1.3 uv (Python package manager)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
# PASS: uv 0.9.x or newer
```

### 1.4 Node.js LTS (v22)
```powershell
winget install OpenJS.NodeJS.LTS
node --version
# PASS: v22.x.x
npm --version
# PASS: 10.x.x
```

### 1.5 Docker Desktop
```powershell
winget install Docker.DockerDesktop
docker --version
# PASS: Docker version 29.x or newer
```

### 1.6 Claude Code
```powershell
npm install -g @anthropic-ai/claude-code
claude --version
# PASS: Returns version string
```

### 1.7 Global npm packages
```powershell
npm install -g @google/gemini-cli @railway/cli pyright
```

### 1.8 Visual Studio Build Tools (for TA-LIB compilation)
Required for `ta-lib` Python package to compile against the C library.
Install "Desktop development with C++" workload if not already present.

---

## Phase 2: TA-LIB Installation

### 2.1 Install TA-LIB C Library

1. Download `ta-lib-0.4.0-msvc.zip` from:
   https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-msvc.zip/download

2. Extract to `C:\ta-lib` so the structure is:
   ```
   C:\ta-lib\
     c\
       lib\
         ta_lib.lib
       include\
         ta_libc.h
   ```

3. That's it -- no build step needed. The pre-built .lib files work with MSVC.

### 2.2 Common Pitfalls

| Issue | Solution |
|-------|----------|
| "ta_libc.lib not found" | Extraction path is wrong -- must be exactly `C:\ta-lib` |
| Architecture mismatch | 64-bit Python requires 64-bit libs (the msvc.zip has both) |
| Compilation fails | Visual Studio Build Tools must be installed (MSVC compiler) |
| Alternative approach | `conda install -c conda-forge ta-lib` handles everything automatically |

### 2.3 Verify C Library
```powershell
dir C:\ta-lib\c\lib\ta_lib.lib
dir C:\ta-lib\c\include\ta_libc.h
# PASS: Both files exist
```

---

## Phase 3: vectorbt-workspace Environment

### 3.0 DELETE Old .venv (CRITICAL if workspace was copied from another machine)
The .venv directory contains compiled C extensions (.pyd files for numpy, scipy, TA-LIB, etc.)
that are tied to the source machine. They will segfault or fail to import on a different machine.
```powershell
cd C:\Strat_Trading_Bot\vectorbt-workspace
rmdir /s /q .venv
```
Then `uv sync` (step 3.3) will rebuild everything from scratch using the uv.lock file.
Do NOT try to "fix" the old .venv -- delete and rebuild.

### 3.1 Copy .env File
Transfer `vectorbt-workspace/.env` securely from laptop (or verify it was copied with the folder). Contains:
```
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_ENDPOINT
ALPACA_MID_KEY, ALPACA_MID_SECRET, ALPACA_MID_ENDPOINT
ALPACA_LARGE_KEY, ALPACA_LARGE_SECRET, ALPACA_LARGE_ENDPOINT
ALPACA_BASE_URL, DEFAULT_ACCOUNT
ALPHAVANTAGE_API_KEY
GITHUB_TOKEN          # Required for VBT Pro install from private repo
VECTORBT_TOKEN        # Required for VBT Pro runtime license
OPENAI_API_KEY
TIINGO_API_KEY
THETADATA_HOST, THETADATA_PORT, THETADATA_TIMEOUT, THETADATA_ENABLED
COINBASE_API_KEY, COINBASE_API_SECRET
COINBASE_READONLY_API_KEY, COINBASE_READONLY_API_SECRET
```

### 3.2 Set GITHUB_TOKEN for uv sync
The VBT Pro package installs from a private GitHub repo. uv needs the token:
```powershell
# Set in current shell (or add to system environment variables)
set GITHUB_TOKEN=<your-token-from-.env>
```

### 3.3 Install All Dependencies
```powershell
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv sync
```
This will:
- Create `.venv` with Python 3.12
- Install all 353 packages from `uv.lock`
- Install VectorBT Pro from `git+https://github.com/polakowo/vectorbt.pro.git@v2025.12.31`
- Compile TA-LIB Python wrapper against `C:\ta-lib`

### 3.4 Install Playwright Browsers
```powershell
uv run playwright install
```

### 3.5 Verify Key Packages
```powershell
uv run python -c "import vectorbtpro as vbt; print('VBT Pro:', vbt.__version__)"
# PASS: Prints version (e.g., 2025.12.31)

uv run python -c "import talib; print('TA-Lib:', talib.__version__)"
# PASS: Prints 0.6.x

uv run python -c "
import alpaca, dash, pandas, numpy, plotly, openai, scipy, optuna
print('All key packages imported successfully')
"
# PASS: No ImportError
```

### 3.6 Package Count Sanity Check
```powershell
uv pip list 2>&1 | find /c /v ""
# PASS: ~350+ packages
```

---

## Phase 4: CLRI Environment

CLRI is Docker-based. Local dev setup is optional.

### 4.1 Copy .env File
Transfer `clri/.env` securely. Contains:
```
DATABASE_URL, DISCORD_WEBHOOK_URL, SESSION_SECRET, INVITE_CODES
ENABLED_EXCHANGES, TRACKED_SYMBOLS
COINGLASS_API_KEY, ATLAS_EXPORT_PATH
DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID
```

### 4.2 Local Dev (optional)
```powershell
cd C:\Strat_Trading_Bot\clri
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: CLRI uses Python 3.11 in Docker (`python:3.11-slim`). Local venv can use 3.12 for dev.

---

## Phase 5: OpenMemory Setup

### IMPORTANT: node_modules and the stdio fix

Like .venv, the `node_modules/` directory contains native binaries (e.g., the `sqlite3`
module) compiled for the source machine. You MUST delete and rebuild it on the new machine.

Also, a critical bug was fixed in `backend/src/mcp/index.ts`: the `oninitialized` callback
was using `console.log` which writes to stdout -- the same pipe used by the MCP JSON-RPC
protocol. This caused the MCP connection to drop immediately after connecting with:
`JSON Parse error: Unexpected identifier "MCP"`. The fix changed `console.log` to
`console.error` so diagnostic messages go to stderr instead. This fix is already in the
source code and will travel with the flash drive copy.

### 5.1 ATLAS Instance
```powershell
# If copied via flash drive, the directory already exists.
# Delete and rebuild node_modules (native binaries are machine-specific):
cd C:\Dev\openmemory\backend
rmdir /s /q node_modules
npm install

# Verify .env exists and has correct paths:
#   OM_DB_PATH=C:/Dev/openmemory/data/atlas_memory.sqlite
#   OPENAI_API_KEY=sk-proj-...
# UPDATE OM_DB_PATH if directory structure differs on this machine.

# Verify database exists (should have 525+ memories):
dir C:\Dev\openmemory\data\atlas_memory.sqlite
```

### 5.2 CLRI Instance
```powershell
# Delete and rebuild node_modules:
cd C:\Dev\openmemory-clri\backend
rmdir /s /q node_modules
npm install

# Copy .env (port 8081, separate from ATLAS on 8080)
```

---

## Key Dependency Notes

| Package | Version | Install Method |
|---------|---------|----------------|
| VectorBT Pro | v2025.12.31 | Private GitHub via `uv sync` (needs GITHUB_TOKEN) |
| TA-LIB | 0.6.8 | C library from SourceForge + `uv sync` compiles wrapper |
| PyTorch | 2.9.1+cpu | CPU-only. For GPU: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Playwright | 1.57.0 | `uv sync` + `playwright install` for browsers |
| OpenMemory | 1.0.0 | Node.js backend, `npm install` |

## Python Config Files
- `.python-version`: `3.12`
- `pyproject.toml`: requires `>=3.12, <3.14`, 57 direct deps + 5 dev deps
- `uv.lock`: pins all 353 packages exactly (revision 3)

---
Generated: 2026-02-09
