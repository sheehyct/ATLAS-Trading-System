# MCP Server Setup and Configuration

**Last Updated:** October 22, 2025
**MCP Servers Installed:** 2 (VectorBT Pro, Playwright)

---

## Overview

Model Context Protocol (MCP) servers extend Claude Code's capabilities by providing specialized tools for external integrations. This document covers the MCP servers installed for the ATLAS trading system development.

**Why MCP Servers Matter for Trading Development:**
- Direct access to VectorBT Pro API and documentation
- Automated web scraping for premium research content
- Real-time data extraction from financial sites
- Integration with alternative data sources

---

## Installed MCP Servers

### 1. VectorBT Pro MCP Server

**Purpose:** Direct access to VectorBT Pro documentation, API reference, and code execution.

**Configuration:**
```json
{
  "vectorbt-pro": {
    "type": "stdio",
    "command": "C:\\Strat_Trading_Bot\\vectorbt-workspace\\.venv\\Scripts\\python.exe",
    "args": ["-m", "vectorbtpro.mcp_server"],
    "env": {
      "GITHUB_TOKEN": "[REDACTED]"
    }
  }
}
```

**Capabilities:**
- Search VectorBT Pro documentation (API, docs, examples, Discord messages)
- Resolve reference names (verify class/method existence)
- Find usage examples for specific APIs
- Get object attributes (like Python's dir())
- Get source code for any VBT object
- Run code snippets in Jupyter kernel

**Critical Usage:**
**MANDATORY** 5-step VBT verification workflow before implementing any VectorBT code:
1. SEARCH: `mcp__vectorbt-pro__search()` for patterns
2. VERIFY: `resolve_refnames()` to confirm methods exist
3. FIND: `mcp__vectorbt-pro__find()` for real-world usage
4. TEST: `mcp__vectorbt-pro__run_code()` minimal example
5. IMPLEMENT: Only after steps 1-4 pass

**Reference:** CLAUDE.md lines 115-303

---

### 2. Playwright MCP Server

**Purpose:** Browser automation for accessing premium research, web scraping, and alternative data extraction.

**Configuration:**
```json
{
  "playwright": {
    "type": "stdio",
    "command": "npx",
    "args": ["@playwright/mcp@latest", "--browser=chrome"],
    "env": {}
  }
}
```

**Capabilities:**
- Navigate to URLs
- Click elements, fill forms, type text
- Take screenshots and accessibility snapshots
- Monitor network requests and console messages
- Execute JavaScript on pages
- Handle authentication (persistent profiles)
- Access paywalled content after manual login

**Browser Profile Persistence:**
- Profile location: `%USERPROFILE%\AppData\Local\ms-playwright\mcp-chrome-profile`
- Cookies and session data saved automatically
- Login once per site, credentials persist across sessions

**Installation Verification:**
```bash
claude mcp list
# Should show:
# playwright: npx @playwright/mcp@latest --browser=chrome - Connected
```

**Test Command:**
Navigate to example.com to verify:
```javascript
await page.goto('https://example.com');
```

---

## Usage Patterns for Trading Development

### Pattern 1: Premium Research Access

**Use Case:** Access paywalled trading research during development sessions.

**Workflow:**
1. First time only: Navigate to premium site, log in manually
2. Browser saves credentials automatically to profile
3. Subsequent sessions: Already logged in, immediate access
4. Ask Claude to extract strategies, summarize research, compare sources

**Example Sites:**
- Medium premium articles
- Seeking Alpha premium content
- Bloomberg Terminal web interface (if subscribed)
- Premium options flow services (Unusual Whales, etc.)

### Pattern 2: Alternative Data Extraction

**Use Case:** Extract alternative data signals to enhance risk management.

**Workflow:**
1. Navigate to alternative data source (insider trading, options flow, etc.)
2. Extract data using accessibility tree
3. Parse structured data (JSON, tables, etc.)
4. Feed to PortfolioHeatManager for quality filtering

**Integration Point:**
```python
# utils/portfolio_heat.py
class PortfolioHeatManager:
    def can_accept_trade(self, symbol, position_risk, capital):
        # Existing heat check
        if new_heat > self.max_heat:
            return False

        # NEW: Alternative data quality filter (future)
        # quality = playwright_extract_signal_quality(symbol)
        # if quality < threshold:
        #     return False

        return True
```

### Pattern 3: Economic Calendar Monitoring

**Use Case:** Extract economic event data for regime detection.

**Workflow:**
1. Navigate to economic calendar (Investing.com, ForexFactory, etc.)
2. Extract events, dates, expected impact
3. Feed to GMM regime detection for context
4. Adjust strategy parameters based on upcoming events

### Pattern 4: Regulatory Monitoring

**Use Case:** Monitor SEC.gov for rule changes affecting algorithmic trading.

**Workflow:**
1. Navigate to SEC.gov/edgar or FINRA regulatory notices
2. Search for keywords: "algorithmic trading", "pattern day trading", etc.
3. Extract rule proposals and effective dates
4. Alert if changes affect current strategies

---

## Authentication Setup for Premium Sites

**First-Time Setup for Paywalled Content:**

1. **Start Playwright with visible browser:**
   - Browser opens automatically when MCP server starts
   - Headed mode by default (browser visible)

2. **Manual login to premium sites:**
   - Navigate to your premium research site
   - Complete login process (including MFA if required)
   - Browser automatically saves all credentials

3. **Verify persistence:**
   - Close browser
   - Restart Claude Code
   - Navigate to premium site again
   - Should already be logged in (no re-authentication needed)

**Profile Location:**
```
Windows: %USERPROFILE%\AppData\Local\ms-playwright\mcp-chrome-profile
```

**Clear credentials (if needed):**
Delete the profile directory to reset all saved sessions.

---

## Configuration File Location

**Claude Code MCP Config:**
```
C:\Users\sheeh\.claude.json
```

**Project-Level Config (if needed):**
```
C:\Strat_Trading_Bot\vectorbt-workspace\.mcp.json
```

Note: Currently using user-level config. To share with team, move configuration to project-level `.mcp.json` and commit to git.

---

## Verification Commands

**Check MCP Server Status:**
```bash
claude mcp list
```

Expected output:
```
vectorbt-pro: C:\...\python.exe -m vectorbtpro.mcp_server - Connected
playwright: npx @playwright/mcp@latest --browser=chrome - Connected
```

**Test VectorBT Pro MCP:**
```python
# Search for portfolio methods
mcp__vectorbt-pro__search("portfolio from_signals")
```

**Test Playwright MCP:**
```javascript
// Navigate to test page
await page.goto('https://example.com');
```

---

## Troubleshooting

### VectorBT Pro MCP Issues

**Problem:** Server shows "Disconnected"
**Solution:**
1. Verify virtual environment activated: `C:\Strat_Trading_Bot\vectorbt-workspace\.venv`
2. Check Python path in config matches actual venv location
3. Restart Claude Code

**Problem:** GitHub token expired
**Solution:**
1. Generate new GitHub personal access token
2. Update `GITHUB_TOKEN` in `.claude.json`
3. Restart Claude Code

### Playwright MCP Issues

**Problem:** Browser doesn't open
**Solution:**
1. Run `npx playwright install` to install browser binaries
2. Verify Chrome installed on system
3. Check `--browser=chrome` argument in config

**Problem:** Lost authentication to premium sites
**Solution:**
1. Delete browser profile directory
2. Restart Claude Code
3. Log in manually again

**Problem:** "Command not found: npx"
**Solution:**
1. Install Node.js from nodejs.org
2. Verify installation: `node --version` and `npm --version`
3. Restart Claude Code

---

## Security Considerations

**GitHub Token:**
- Personal access token stored in `.claude.json`
- Required for VectorBT Pro Discord message search
- Scope: `read:user` (minimal permissions)
- **DO NOT commit `.claude.json` to git**

**Browser Profile:**
- Contains all login credentials for premium sites
- Stored locally in Windows AppData
- Not synchronized or backed up
- Delete profile directory to clear all saved credentials

**Premium Site Credentials:**
- Browser stores cookies and session tokens
- Persistent across Claude Code sessions
- Automatically used when navigating to sites
- Manual logout required to clear

---

## Future MCP Server Additions (Potential)

**Database Servers:**
- ClickHouse MCP for tick data storage
- DuckDB MCP for analytics queries
- PostgreSQL MCP (if maintenance resumes)

**Financial Data:**
- Alpha Vantage MCP for market data
- Financial Datasets MCP for fundamentals
- Alpaca MCP for order execution

**Development Tools:**
- GitHub MCP for version control
- Jupyter MCP for notebook integration
- Shell command MCP for data pipelines

**Alternative Data:**
- Custom MCP server for options flow data
- Custom MCP server for insider trading alerts
- Custom MCP server for short interest tracking

---

## Reference Documentation

**MCP Protocol:**
- Official spec: https://modelcontextprotocol.io/
- Claude Code MCP docs: https://docs.claude.com/en/docs/claude-code/mcp

**Playwright MCP:**
- GitHub: https://github.com/microsoft/playwright-mcp
- Playwright docs: https://playwright.dev/

**VectorBT Pro MCP:**
- VBT docs: https://vectorbt.pro/
- VBT MCP tools: See `mcp__vectorbt-pro__*` functions

---

## Changelog

**October 22, 2025:**
- Initial MCP setup documentation created
- VectorBT Pro MCP: Configured and verified
- Playwright MCP: Installed with Chrome browser
- Both servers tested and confirmed working
- Profile persistence configured for premium research access

---

**Next Steps:**
1. Test Playwright with actual premium research site
2. Document specific premium sites being used
3. Create alternative data extraction examples
4. Build integration with PortfolioHeatManager (Phase 4)
