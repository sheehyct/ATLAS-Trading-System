# Laptop Migration Guide: Gigabyte Aero X16 to Lenovo Legion 5i Pro

**Date:** October 21, 2025
**From:** Gigabyte Aero X16
**To:** Lenovo Legion 5i Pro (Gen 10) - Windows 11
**Specs:** Intel Core Ultra 9 275HX, 32GB RAM, RTX 5070
**Project:** ATLAS Trading System (VectorBT Pro)

---

## PART 1: PROGRAMS & DEPENDENCIES REFERENCE LIST

### Core Development Tools

#### 1. Python
- **Version Required:** Python 3.12.11 (EXACT - VectorBT Pro compatibility)
- **Current System Python:** 3.13.7 (but project uses 3.12.11 via .python-version)
- **Installation:** Download from https://www.python.org/downloads/
- **Important:** Install Python 3.12.11, NOT the latest 3.13.x
- **Installation Options:**
  - Add Python to PATH (check box during install)
  - Install for all users
  - Install pip

**Verification:**
```bash
python --version  # Should show Python 3.12.11
```

#### 2. Node.js & npm
- **Version Required:** Node.js v22.19.0 (or latest v22.x LTS)
- **npm Version:** 10.9.3 (comes with Node.js)
- **Installation:** Download from https://nodejs.org/
- **Recommendation:** Use Node.js LTS version

**Verification:**
```bash
node --version  # Should show v22.x.x
npm --version   # Should show 10.x.x
```

#### 3. Git
- **Version Required:** 2.51.0 or newer
- **Installation:** Download Git for Windows from https://git-scm.com/download/win
- **Important Options During Install:**
  - Use Git from the Windows Command Prompt (default)
  - Use bundled OpenSSH
  - Checkout Windows-style, commit Unix-style line endings
  - Use MinTTY terminal emulator
  - Enable Git Credential Manager

**Verification:**
```bash
git --version  # Should show git version 2.51.0 or newer
```

#### 4. uv (Python Package Manager)
- **Version Required:** 0.8.22 or newer
- **Installation:**
```powershell
# PowerShell (Run as Administrator)
irm https://astral.sh/uv/install.ps1 | iex
```
- **Alternative:**
```bash
pip install uv
```

**Verification:**
```bash
uv --version  # Should show uv 0.8.x
```

#### 5. Visual Studio (C++ Build Tools for TA-Lib)
- **Version Required:** Visual Studio 2022 or newer
- **Installation:** Download Visual Studio Community from https://visualstudio.microsoft.com/
- **CRITICAL Workloads to Install:**
  - Desktop development with C++
  - MSVC v143 build tools (or latest)
  - Windows 10/11 SDK
- **Purpose:** Required to compile TA-Lib C library
- **Note:** You can install just "Build Tools for Visual Studio" if you don't need the full IDE

**Verification:**
```powershell
# Check if Visual Studio is installed
Get-Command cl.exe -ErrorAction SilentlyContinue
# Should find cl.exe in Visual Studio directory
```

#### 6. Visual Studio Code
- **Version Required:** 1.105.1 or newer
- **Installation:** Download from https://code.visualstudio.com/
- **Important Options:**
  - Add "Open with Code" to context menu (both files and directories)
  - Add to PATH
  - Register Code as editor for supported file types

**Verification:**
```bash
code --version  # Should show 1.105.x or newer
```

---

### WSL2 (Windows Subsystem for Linux)

**CRITICAL:** WSL2 is required for optimal development experience and Claude Code compatibility.

#### Installation Steps:

**Step 1: Enable WSL**
```powershell
# PowerShell (Run as Administrator)
wsl --install
```

**Step 2: Install Ubuntu 20.04 (or Ubuntu 22.04 LTS)**
```powershell
wsl --install -d Ubuntu-20.04
```

**Step 3: Set WSL 2 as default**
```powershell
wsl --set-default-version 2
```

**Step 4: Verify Installation**
```powershell
wsl --list --verbose
# Should show Ubuntu-20.04 with VERSION 2
```

**Step 5: Configure Ubuntu**
```bash
# Inside WSL Ubuntu terminal
sudo apt update
sudo apt upgrade -y

# Install essential build tools
sudo apt install build-essential -y
```

---

### VS Code Extensions (30+ Extensions Required)

**CRITICAL Extensions (Install First):**

1. **anthropic.claude-code** - Claude Code extension (REQUIRED)
2. **ms-python.python** - Python language support
3. **ms-python.vscode-pylance** - Python IntelliSense
4. **ms-vscode-remote.remote-wsl** - WSL integration (CRITICAL)

**Python Development:**
- donjayamanne.python-environment-manager
- donjayamanne.python-extension-pack
- ms-python.debugpy
- ms-python.vscode-python-envs
- kevinrose.vsc-python-indent
- njpwerner.autodocstring
- hbenl.vscode-test-explorer
- littlefoxteam.vscode-python-test-adapter
- ms-vscode.test-adapter-converter

**Remote Development Pack:**
- ms-vscode-remote.remote-containers
- ms-vscode-remote.remote-ssh
- ms-vscode-remote.remote-ssh-edit
- ms-vscode-remote.vscode-remote-extensionpack
- ms-vscode.remote-explorer
- ms-vscode.remote-repositories
- ms-vscode.remote-server

**Git & Version Control:**
- eamodio.gitlens

**AI Assistants:**
- github.copilot (if you use it)
- github.copilot-chat (if you use it)
- openai.chatgpt (optional)

**Additional Tools:**
- batisteo.vscode-django
- ms-toolsai.jupyter-keymap
- ritwickdey.liveserver
- tomoki1207.pdf
- visualstudioexptteam.intellicode-api-usage-examples
- visualstudioexptteam.vscodeintellicode
- vscode-icons-team.vscode-icons
- wholroyd.jinja
- ms-vscode.azure-repos
- github.remotehub

**Installation Method:**
```bash
# After VS Code is installed, install all extensions via command line:
code --install-extension anthropic.claude-code
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-vscode-remote.remote-wsl
# ... (continue for all extensions)
```

---

### VectorBT Pro Specific Requirements

#### 1. GitHub Personal Access Token
- **Purpose:** Access to VectorBT Pro private repository
- **Generate at:** https://github.com/settings/tokens
- **Required Scopes:**
  - `repo` (Full control of private repositories)
- **Store in:** `.env` file as `GITHUB_TOKEN` and `VECTORBT_TOKEN`

#### 2. VectorBT Pro MCP Server Dependencies
**Already in pyproject.toml, installed via `uv sync`:**
- lmdbm==0.0.6
- bm25s==0.2.14
- tiktoken==0.12.0
- mcp>=1.0.0

#### 3. TA-Lib (Technical Analysis Library)
- **Version Required:** ta-lib>=0.6.0
- **C Library Location:** C:\ta-lib
- **Purpose:** 150+ technical analysis functions for strategy development
- **Installation:** See dedicated TA-Lib installation section below

---

### API Keys & Credentials Required

**Create `.env` file with the following:**

```bash
# Alpaca Paper Trading Accounts
ALPACA_API_KEY=your_base_api_key
ALPACA_SECRET_KEY=your_base_secret_key
ALPACA_ENDPOINT=https://paper-api.alpaca.markets

ALPACA_MID_KEY=your_mid_api_key
ALPACA_MID_SECRET=your_mid_secret_key
ALPACA_MID_ENDPOINT=https://paper-api.alpaca.markets

ALPACA_LARGE_KEY=your_large_api_key
ALPACA_LARGE_SECRET=your_large_secret_key
ALPACA_LARGE_ENDPOINT=https://paper-api.alpaca.markets

ALPACA_BASE_URL=https://paper-api.alpaca.markets
DEFAULT_ACCOUNT=MID

# VectorBT Pro GitHub Access
GITHUB_TOKEN=your_github_personal_access_token
VECTORBT_TOKEN=your_github_personal_access_token

# AlphaVantage (Optional)
ALPHAVANTAGE_API_KEY=your_alphavantage_key

# OpenAI API (for VectorBT Pro advanced features)
OPENAI_API_KEY=your_openai_api_key
```

**Sign Up Links:**
- Alpaca Markets: https://alpaca.markets/
- AlphaVantage: https://www.alphavantage.co/support/#api-key
- OpenAI: https://platform.openai.com/api-keys

---

### Project Dependencies (Installed via uv sync)

**All dependencies are in `pyproject.toml` and will be installed automatically.**

**Key Dependencies:**
- vectorbtpro @ git+https://github.com/polakowo/vectorbt.pro.git@v2025.10.15
- alpaca-py>=0.42.0
- alpaca-trade-api>=3.2.0
- pandas>=2.1.0
- numpy>=1.26.0
- matplotlib>=3.10.5
- jupyter>=1.1.1
- jupyterlab>=4.4.5
- openai>=1.51.0
- python-dotenv>=1.1.1
- rich>=13.7.1
- typer>=0.12.5
- ta-lib>=0.6.0
- pytest>=7.0.0
- black>=22.0.0
- ruff>=0.1.0
- mypy>=1.0.0

**Installation after setup:**
```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv sync
```

---

### TA-Lib Installation Guide (CRITICAL FOR ATLAS)

**IMPORTANT:** TA-Lib is required for technical analysis functions in ATLAS strategies. This is a multi-step process that must be completed before running strategies.

#### Prerequisites
- Visual Studio 2022 (or newer) with C++ development tools installed
- Git for Windows installed
- Python 3.12.11 installed

#### Step 1: Download TA-Lib C Library

1. Go to TA-Lib SourceForge: https://sourceforge.net/projects/ta-lib/
2. Click "Download" to download the source code archive
3. Extract the downloaded archive to `C:\ta-lib`
   - **CRITICAL:** Extract to EXACTLY `C:\ta-lib` (not `C:\ta-lib-0.4.0` or any other path)
   - After extraction, you should have: `C:\ta-lib\c\` directory

**Verify extraction:**
```powershell
Test-Path C:\ta-lib\c\make\cdr\win32\msvc
# Should return: True
```

#### Step 2: Build TA-Lib C Library

**IMPORTANT:** This requires Visual Studio C++ build tools.

1. Open "x64 Native Tools Command Prompt for VS 2022"
   - **Find it:** Start Menu → Visual Studio 2022 → x64 Native Tools Command Prompt
   - **MUST use this specific command prompt** (not regular PowerShell or CMD)

2. Navigate to the MSVC directory:
```cmd
cd C:\ta-lib\c\make\cdr\win32\msvc
```

3. Build the library:
```cmd
nmake
```

**Expected output:**
- Compilation messages for ~150 functions
- No errors (warnings are OK)
- Creates library files in `C:\ta-lib\c\lib`

**Troubleshooting Build Errors:**
- **Error:** "nmake is not recognized"
  - **Solution:** You're not using the correct command prompt. Use "x64 Native Tools Command Prompt for VS 2022"

- **Error:** "Cannot find compiler"
  - **Solution:** Install "Desktop development with C++" workload in Visual Studio

- **Error:** "Permission denied"
  - **Solution:** Run command prompt as Administrator

4. Set environment variables (in the same command prompt):
```cmd
set TA_INCLUDE_PATH=C:\ta-lib\c\include
set TA_LIBRARY_PATH=C:\ta-lib\c\lib
```

#### Step 3: Install Python TA-Lib Wrapper

**Option A: Using pip (Recommended for this setup)**

Since you're using uv and have the C library built, install via pip in your virtual environment:

```powershell
# Navigate to project
cd C:\Strat_Trading_Bot\vectorbt-workspace

# Set environment variables (PowerShell)
$env:TA_INCLUDE_PATH = "C:\ta-lib\c\include"
$env:TA_LIBRARY_PATH = "C:\ta-lib\c\lib"

# Install via uv
uv pip install ta-lib
```

**Option B: Manual installation from source**

If Option A fails, use the manual method:

```powershell
# Navigate to temp directory
cd $env:TEMP

# Clone TA-Lib Python wrapper
git clone https://github.com/mrjbq7/ta-lib.git
cd ta-lib

# Set environment variables
$env:TA_INCLUDE_PATH = "C:\ta-lib\c\include"
$env:TA_LIBRARY_PATH = "C:\ta-lib\c\lib"

# Build and install
python setup.py build_ext --include-dirs=C:\ta-lib\c\include --library-dirs=C:\ta-lib\c\lib
python setup.py install

# Or install into uv environment
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv pip install $env:TEMP\ta-lib
```

#### Step 4: Verify TA-Lib Installation

```powershell
# Test import
uv run python -c "import talib; print(f'TA-Lib {talib.__version__} loaded')"

# Verify all 158 functions available
uv run python -c "import talib; print(f'Functions available: {len(talib.get_functions())}')"
# Expected output: Functions available: 158
```

**If verification fails:**
- Ensure Visual Studio C++ tools are installed
- Verify `C:\ta-lib\c\lib` contains .lib files
- Rebuild the C library (Step 2)
- Try Option B (manual installation)

#### Step 5: Make Environment Variables Permanent (OPTIONAL)

To avoid setting environment variables every time:

```powershell
# Add to Windows environment variables permanently
[Environment]::SetEnvironmentVariable("TA_INCLUDE_PATH", "C:\ta-lib\c\include", "User")
[Environment]::SetEnvironmentVariable("TA_LIBRARY_PATH", "C:\ta-lib\c\lib", "User")
```

**Restart PowerShell after setting permanent variables.**

#### Common TA-Lib Installation Issues

**Issue 1: "error: command 'cl.exe' failed"**
- **Cause:** Visual Studio C++ tools not found
- **Solution:** Install "Desktop development with C++" in Visual Studio
- **Verify:** Open "x64 Native Tools Command Prompt" and run `cl.exe`

**Issue 2: "Cannot find ta_libc_cdr.lib"**
- **Cause:** C library not built correctly
- **Solution:** Re-run `nmake` in `C:\ta-lib\c\make\cdr\win32\msvc`
- **Verify:** Check `C:\ta-lib\c\lib` contains .lib files

**Issue 3: "ImportError: DLL load failed"**
- **Cause:** Environment variables not set or wrong paths
- **Solution:**
  ```powershell
  $env:TA_INCLUDE_PATH = "C:\ta-lib\c\include"
  $env:TA_LIBRARY_PATH = "C:\ta-lib\c\lib"
  ```
- **Verify paths are correct**

**Issue 4: "ta-lib already installed but import fails"**
- **Cause:** Installed in wrong Python environment or broken installation
- **Solution:**
  ```powershell
  uv pip uninstall ta-lib
  uv pip install ta-lib --no-cache-dir
  ```

#### TA-Lib in pyproject.toml

The project already includes TA-Lib in dependencies:

```toml
dependencies = [
    "ta-lib>=0.6.0",
    # ... other dependencies
]
```

After manual C library installation, `uv sync` will install the Python wrapper.

---

## PART 2: CLAUDE CODE INSTALLATION GUIDE

### ULTRA-DETAILED INSTALLATION GUIDE

This guide addresses the common Windows 11 installation issues you've experienced (multiple install/uninstall/reinstall cycles).

---

### PRE-INSTALLATION: CRITICAL SETUP STEPS

**STEP 1: Verify Prerequisites**

Before installing Claude Code, ensure the following are installed and working:

```powershell
# Run these in PowerShell to verify
node --version    # Should return v22.x.x
npm --version     # Should return 10.x.x
git --version     # Should return 2.51.0+
python --version  # Should return Python 3.12.11
wsl --list --verbose  # Should show Ubuntu with VERSION 2
```

**If ANY of these fail, STOP and install the missing prerequisite first.**

---

**STEP 2: Clean Any Previous Claude Code Installation**

```powershell
# PowerShell (Run as Administrator)

# Check if Claude Code is already installed
where.exe claude

# If it returns a path, uninstall first:
# 1. Check install location
Get-ChildItem -Path $env:USERPROFILE\.local\bin -Filter claude*

# 2. Remove old installation
Remove-Item -Path "$env:USERPROFILE\.local\bin\claude*" -Force -ErrorAction SilentlyContinue

# 3. Remove old config (BACKUP FIRST if you have settings)
# BACKUP: Copy C:\Users\sheeh\AppData\Roaming\Claude\* to a safe location
Remove-Item -Path "$env:APPDATA\Claude" -Recurse -Force -ErrorAction SilentlyContinue

# 4. Clear npm cache
npm cache clean --force
```

---

**STEP 3: Verify PowerShell Execution Policy**

```powershell
# Check current policy
Get-ExecutionPolicy

# If it's "Restricted", change it:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### INSTALLATION: OFFICIAL METHOD (Native Windows)

**CRITICAL:** Use the OFFICIAL native Windows installation method. Do NOT use npm global installation or WSL-only installation.

**STEP 1: Download & Install via PowerShell**

```powershell
# PowerShell (Regular user, NOT Administrator)
# IMPORTANT: Run from your user account, not elevated PowerShell

irm https://claude.ai/install.ps1 | iex
```

**What this does:**
- Downloads the latest stable Claude Code version
- Installs to `C:\Users\sheeh\.local\bin\`
- Adds to PATH automatically
- Creates native Windows executable

**Expected Output:**
```
Downloading Claude Code...
Installing to C:\Users\sheeh\.local\bin\
Installation complete!
```

**COMMON ERROR #1: "Unsupported OS Error"**
- **Cause:** Script detects OS incorrectly
- **Solution:** Ensure you're using PowerShell (not PowerShell ISE or CMD)
- **Alternative:** Use CMD method:
```cmd
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

**COMMON ERROR #2: "Bun ENOTCONN error"**
- **Cause:** PowerShell installer crashes with Bun runtime
- **Solution:** Close all terminals, restart PC, try again
- **Alternative:** Use manual installation (see Troubleshooting section)

---

**STEP 2: Verify Installation**

```powershell
# Close and reopen PowerShell (to refresh PATH)

# Check if claude is available
where.exe claude
# Should return: C:\Users\sheeh\.local\bin\claude.exe

# Check version
claude --version
# Should return: 2.0.x (Claude Code)

# Run doctor command
claude doctor
```

**Expected `claude doctor` Output:**
```
Claude Code Installation Check:

Installation method: native
Version: 2.0.x
Shell: PowerShell
Git: Installed (2.51.0)
Node: Installed (v22.19.0)
VS Code: Detected
```

**If `claude doctor` shows "unknown" installation method:**
- This is a KNOWN BUG (Issue #5540)
- As long as `claude --version` works, you can proceed
- The functionality is NOT affected

---

**STEP 3: Initial Authentication**

```powershell
# Run Claude Code for the first time
claude

# You will be prompted to:
# 1. Select authentication method: Choose "Claude Console"
# 2. Browser will open for OAuth authentication
# 3. Log in with your Anthropic account
# 4. Complete authentication flow

# After authentication, you'll see:
# "Successfully authenticated!"
```

**CRITICAL:** Make sure you have an active Anthropic account with billing enabled at https://console.anthropic.com/

---

### POST-INSTALLATION: CONFIGURATION

**STEP 1: Configure MCP Server for VectorBT Pro**

**IMPORTANT:** Claude Code uses a DIFFERENT config file than Claude Desktop.

**Claude Code Config Location:**
- Windows: `C:\Users\sheeh\AppData\Roaming\Claude\claude_desktop_config.json`

**DO NOT edit this file manually during initial setup.** Let Claude Code create it.

**Add VectorBT Pro MCP Server:**

```powershell
# Option 1: Use Claude Code CLI (RECOMMENDED)
claude mcp add vectorbt-pro `
  -s user `
  "C:\Strat_Trading_Bot\vectorbt-workspace\.venv\Scripts\python.exe" `
  -e GITHUB_TOKEN="your_github_token_here" `
  -- -m vectorbtpro.mcp_server

# Option 2: Manual config file edit (ADVANCED)
# Edit: C:\Users\sheeh\AppData\Roaming\Claude\claude_desktop_config.json
```

**Manual config file content:**
```json
{
  "mcpServers": {
    "vectorbt-pro": {
      "command": "C:\\Strat_Trading_Bot\\vectorbt-workspace\\.venv\\Scripts\\python.exe",
      "args": [
        "-m",
        "vectorbtpro.mcp_server"
      ],
      "env": {
        "GITHUB_TOKEN": "your_github_token_here"
      }
    }
  }
}
```

**Verify MCP Server:**
```powershell
claude mcp list

# Should show:
# vectorbt-pro (connected)
```

---

**STEP 2: Configure VS Code Integration**

**Install Claude Code Extension:**
```bash
code --install-extension anthropic.claude-code
```

**Verify Integration:**
1. Open VS Code
2. Press `Ctrl+Shift+P`
3. Type "Claude Code"
4. You should see Claude Code commands available

---

**STEP 3: Configure Project-Specific Settings**

**Create `.claude` directory in project root:**
```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace

# These files should already exist, but verify:
ls .claude\
# Should show:
# - settings.json
# - settings.local.json
# - development_state.json
```

**Your current `.claude/settings.json`:**
```json
{
  "statusLine": {
    "type": "command",
    "command": "ccstatusline",
    "padding": 0
  }
}
```

**Your current `.claude/settings.local.json`:**
```json
{
  "permissions": {
    "allow": [
      "Bash(code --list-extensions)",
      "Bash(claude --version)",
      "WebSearch",
      "mcp__vectorbt-pro__search",
      "Read(//c/Users/sheeh/.claude/**)",
      "... (and many more)"
    ],
    "deny": [],
    "ask": []
  },
  "outputStyle": "Explanatory"
}
```

**These files control Claude Code behavior in this specific project.**

---

### VERIFICATION CHECKLIST

**Run these commands to verify everything is working:**

```powershell
# 1. Claude Code installed
claude --version
# Expected: 2.0.x (Claude Code)

# 2. Claude Code authenticated
claude doctor
# Expected: Shows authentication status

# 3. MCP Server configured
claude mcp list
# Expected: Shows vectorbt-pro (connected)

# 4. VS Code extension installed
code --list-extensions | findstr claude-code
# Expected: anthropic.claude-code

# 5. Project environment ready
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv sync
# Expected: All dependencies installed

# 6. TA-Lib installed
uv run python -c "import talib; print(f'TA-Lib loaded: {len(talib.get_functions())} functions')"
# Expected: TA-Lib loaded: 158 functions

# 7. VectorBT Pro accessible
uv run python -c "import vectorbtpro as vbt; print(f'VBT Pro {vbt.__version__} loaded')"
# Expected: VBT Pro 2025.10.15 loaded

# 8. Test Claude Code in project
cd C:\Strat_Trading_Bot\vectorbt-workspace
claude
# Expected: Interactive Claude Code session starts
```

---

### TROUBLESHOOTING: COMMON WINDOWS 11 ISSUES

#### Issue #1: "claude: command not found" after installation

**Symptoms:**
- Installation completes successfully
- `where.exe claude` returns nothing
- `claude --version` fails

**Cause:** PATH not updated in current session

**Solution:**
```powershell
# 1. Close ALL PowerShell windows
# 2. Open NEW PowerShell window
# 3. Try again

# If still not working:
# 4. Check installation location
Get-ChildItem -Path $env:USERPROFILE\.local\bin -Filter claude*

# 5. Manually add to PATH
$env:Path += ";$env:USERPROFILE\.local\bin"

# 6. Verify
claude --version
```

**Permanent Fix:**
```powershell
# Add to Windows PATH permanently
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:USERPROFILE\.local\bin",
    "User"
)
```

---

#### Issue #2: "Cannot interact with Claude Code interface" (Issue #95)

**Symptoms:**
- Claude Code launches
- Welcome message displays
- Terminal does not accept input (frozen)

**Cause:** PowerShell input buffer issue

**Solution:**
```powershell
# 1. Use Windows Terminal instead of default PowerShell
# Download Windows Terminal from Microsoft Store

# 2. Run Claude Code in Windows Terminal:
wt.exe -p PowerShell

# Then:
claude

# 3. Alternative: Use WSL
wsl
claude
```

---

#### Issue #3: "Installation method keeps flipping" (Issue #5540)

**Symptoms:**
- `claude doctor` shows "unknown" or "npm-global"
- But you used native PowerShell install

**Cause:** Detection bug in Claude Code

**Impact:** NONE - functionality is NOT affected

**Solution:** Ignore this issue, it's cosmetic only

**Workaround:** If it bothers you, add this to your PowerShell profile:
```powershell
# No actual fix needed, just verify it works:
claude --version  # As long as this works, you're fine
```

---

#### Issue #4: "VS Code not detected" (Issue #5153 & #1276)

**Symptoms:**
- `claude doctor` shows "No available IDEs detected"
- VS Code is installed and working

**Cause:** Claude Code can't find VS Code in PATH

**Solution:**
```powershell
# 1. Verify VS Code in PATH
where.exe code
# Should return: C:\Users\sheeh\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd

# If not found:
# 2. Reinstall VS Code with "Add to PATH" option checked

# 3. Or manually add to PATH:
$env:Path += ";C:\Users\sheeh\AppData\Local\Programs\Microsoft VS Code\bin"

# 4. Restart Claude Code
```

---

#### Issue #5: "Git Bash not detected" (Issue #8674)

**Symptoms:**
- Claude Code requires git-bash
- Git is installed but not detected

**Cause:** Git Bash not in PATH

**Solution:**
```powershell
# 1. Verify Git installation
git --version

# 2. Find Git Bash location
Get-ChildItem -Path "C:\Program Files\Git" -Filter bash.exe -Recurse

# 3. Add Git Bash to PATH
$env:Path += ";C:\Program Files\Git\bin"

# 4. Verify
bash --version
```

---

#### Issue #6: PowerShell installer crashes (Issue #9060)

**Symptoms:**
- `irm https://claude.ai/install.ps1 | iex` crashes
- Error: "ENOTCONN" from Bun runtime

**Cause:** Bun v1.2.23 compatibility issue

**Solution:**
```powershell
# Option 1: Restart computer and try again
# (Often fixes temporary network/runtime issues)

# Option 2: Use CMD installer instead
# Open CMD (not PowerShell):
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd

# Option 3: Manual installation
# 1. Download binary directly from:
#    https://github.com/anthropics/claude-code/releases/latest
# 2. Extract to: C:\Users\sheeh\.local\bin\
# 3. Add to PATH manually (see Issue #1)
```

---

#### Issue #7: "OAuth authentication not supported" with WebFetch

**Symptoms:**
- Running commands that use WebFetch
- Error: "OAuth authentication is currently not supported"

**Cause:** This is NOT an installation issue

**Impact:** Only affects WebFetch tool, NOT core Claude Code functionality

**Solution:** This is expected behavior - WebFetch has restrictions. Use alternative methods:
- Use `WebSearch` instead of `WebFetch` for general queries
- For VectorBT Pro docs, use MCP server instead: `mcp__vectorbt-pro__search`

---

### MIGRATION-SPECIFIC: TRANSFERRING CONFIG

**What to backup from old laptop:**

1. **API Keys (CRITICAL)**
```bash
# Old laptop:
# Copy this file to external drive/cloud:
C:\Strat_Trading_Bot\vectorbt-workspace\.env
```

2. **Claude Code Settings (OPTIONAL)**
```bash
# Old laptop - backup these directories:
C:\Users\sheeh\.claude\                    # User-level settings
C:\Strat_Trading_Bot\vectorbt-workspace\.claude\  # Project-level settings
C:\Users\sheeh\AppData\Roaming\Claude\     # Claude Code config
```

3. **VS Code Settings (OPTIONAL)**
```bash
# Old laptop:
# VS Code settings sync should handle this automatically
# But backup just in case:
%APPDATA%\Code\User\settings.json
%APPDATA%\Code\User\keybindings.json
```

4. **Git Configuration**
```bash
# Old laptop - save your git config:
git config --global --list > git-config-backup.txt

# New laptop - restore:
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
# ... (apply other settings from backup)
```

**What to transfer to new laptop:**

1. **Entire Project Directory**
```powershell
# New laptop:
# After git is installed:
cd C:\Strat_Trading_Bot\
git clone https://github.com/yourusername/vectorbt-workspace.git
cd vectorbt-workspace

# Restore .env file (from backup)
# Copy .env from backup location to: C:\Strat_Trading_Bot\vectorbt-workspace\.env
```

2. **Install Dependencies**
```powershell
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv sync  # This reads pyproject.toml and installs everything
```

3. **Restore Claude Code Settings**
```powershell
# Copy backed up settings to new laptop:
# .claude\ folder -> C:\Strat_Trading_Bot\vectorbt-workspace\.claude\
# AppData\Roaming\Claude\ -> C:\Users\sheeh\AppData\Roaming\Claude\
```

---

### FINAL VERIFICATION: COMPLETE SYSTEM CHECK

**Run this verification script on new laptop:**

```powershell
# Save this as verify-installation.ps1

Write-Host "=== ATLAS Trading System - Installation Verification ===" -ForegroundColor Cyan

# 1. Python
Write-Host "`n1. Python:" -ForegroundColor Yellow
python --version

# 2. Node.js & npm
Write-Host "`n2. Node.js & npm:" -ForegroundColor Yellow
node --version
npm --version

# 3. Git
Write-Host "`n3. Git:" -ForegroundColor Yellow
git --version

# 4. uv
Write-Host "`n4. uv Package Manager:" -ForegroundColor Yellow
uv --version

# 5. Visual Studio C++ Tools
Write-Host "`n5. Visual Studio C++ Tools:" -ForegroundColor Yellow
Get-Command cl.exe -ErrorAction SilentlyContinue

# 6. VS Code
Write-Host "`n6. VS Code:" -ForegroundColor Yellow
code --version

# 7. WSL2
Write-Host "`n7. WSL2:" -ForegroundColor Yellow
wsl --list --verbose

# 8. Claude Code
Write-Host "`n8. Claude Code:" -ForegroundColor Yellow
claude --version
claude doctor

# 9. MCP Server
Write-Host "`n9. MCP Server:" -ForegroundColor Yellow
claude mcp list

# 10. TA-Lib Installation
Write-Host "`n10. TA-Lib:" -ForegroundColor Yellow
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv run python -c "import talib; print(f'TA-Lib loaded: {len(talib.get_functions())} functions')"

# 11. VectorBT Pro
Write-Host "`n11. VectorBT Pro:" -ForegroundColor Yellow
uv run python -c "import vectorbtpro as vbt; print(f'VBT Pro {vbt.__version__} loaded')"

# 12. VS Code Extensions
Write-Host "`n12. VS Code Extensions:" -ForegroundColor Yellow
code --list-extensions | Select-String -Pattern "anthropic|python|remote"

Write-Host "`n=== Verification Complete ===" -ForegroundColor Green
```

**Run the script:**
```powershell
.\verify-installation.ps1
```

**All checks should PASS before proceeding with development.**

---

## SUMMARY: INSTALLATION ORDER

**Follow this EXACT order to avoid issues:**

1. Install Windows 11 updates (ensure latest version)
2. Enable WSL and install Ubuntu 20.04
3. Install Git for Windows
4. Install Python 3.12.11 (EXACT version)
5. Install Node.js v22.x LTS
6. Install Visual Studio 2022 with C++ development tools (for TA-Lib)
7. Install uv package manager
8. Install VS Code with required extensions
9. Install TA-Lib C library (download, extract, build with nmake)
10. Clone project repository
11. Install TA-Lib Python wrapper (uv pip install ta-lib)
12. Run `uv sync` in project directory
13. Install Claude Code via PowerShell (native method)
14. Authenticate Claude Code
15. Configure MCP server for VectorBT Pro
16. Run verification script
17. Start development!

---

## HELPFUL RESOURCES

**Official Documentation:**
- Claude Code Docs: https://docs.claude.com/en/docs/claude-code/setup
- Claude Code GitHub: https://github.com/anthropics/claude-code
- VectorBT Pro: https://vectorbt.pro/ (requires subscription)
- TA-Lib Documentation: https://ta-lib.org/

**Troubleshooting:**
- Claude Code Issues: https://github.com/anthropics/claude-code/issues
- Windows Dev Setup: https://learn.microsoft.com/en-us/windows/dev-environment/
- TA-Lib Installation: https://sourceforge.net/projects/ta-lib/

**Support:**
- Claude Code Help: Run `/help` in Claude Code
- Submit Issues: https://github.com/anthropics/claude-code/issues

---

## NOTES FOR SUCCESS

1. **Use Native Windows Installation ONLY**
   - Do NOT use npm global install
   - Do NOT use WSL-only install (WSL is for development, not Claude Code)
   - Official PowerShell method is most stable

2. **Restart is Your Friend**
   - After installing ANY tool, restart PowerShell
   - If PATH issues, restart PC
   - Fresh start = fewer issues

3. **VS Code Extension Install Order Matters**
   - Install Remote-WSL extension BEFORE Python extension
   - Restart VS Code after installing extensions
   - Let VS Code detect Python interpreter automatically

4. **TA-Lib Installation is Critical**
   - Must complete BEFORE running strategies
   - Visual Studio C++ tools are MANDATORY
   - Use "x64 Native Tools Command Prompt" for nmake
   - Verify 158 functions available after installation

5. **MCP Server Configuration**
   - Must use FULL path to Python executable in virtual environment
   - Test MCP server connection with `claude mcp list` after setup
   - If "failed", check GitHub token and Python path

6. **Environment Variables**
   - Store ALL secrets in `.env` file (gitignored)
   - Never commit API keys to git
   - Backup `.env` file separately (it's critical)

7. **When In Doubt**
   - Read the error message carefully
   - Check GitHub issues (linked above)
   - Run `claude doctor` for diagnostics
   - Restart and try again (surprisingly effective)

---

**Good luck with the migration! This guide should eliminate the install/uninstall/reinstall cycle you experienced before.**

**If you encounter issues not covered here, document them and update this guide for future reference.**

---

**Document Version:** 1.1
**Last Updated:** October 21, 2025
**Status:** Ready for Migration
**Tested On:** Windows 11 (Gigabyte Aero X16)
**Target:** Windows 11 (Lenovo Legion 5i Pro - Intel Core Ultra 9 275HX, 32GB RAM, RTX 5070)
