# HANDOFF - ATLAS Trading System Development

**Last Updated:** December 18, 2025 (Session EQUITY-19)
**Current Branch:** `main`
**Phase:** Paper Trading - MONITORING + Crypto STRAT Integration
**Status:** CRITICAL BUG FIXED - Bidirectional entry logic implemented

---

## Session EQUITY-19: CRITICAL Entry Logic Bug Fix (COMPLETE)

**Date:** December 18, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Bidirectional setup detection deployed to VPS

### The Bug (Now Fixed)

Our entry logic was detecting COMPLETED patterns (both bars closed) instead of SETUPS (forming bar as implicit "1"). This caused entries to be one bar late.

### The Fix: Bidirectional Setup Detection

Per STRAT methodology: "The forming bar is always treated as '1' until it breaks."

**Changes Made:**

1. **`detect_22_setups_nb()` (pattern_detector.py:1127-1233)**
   - Now returns 8 values instead of 5 (bidirectional triggers)
   - For 2U bar: `long_trigger = HIGH`, `short_trigger = LOW`
   - For 2D bar: `long_trigger = HIGH`, `short_trigger = LOW`
   - Entry monitor detects which bound breaks first

2. **`detect_322_setups_nb()` (pattern_detector.py:1236-1345)**
   - Same bidirectional returns for 3-2 patterns
   - Both continuation and reversal triggers returned

3. **`detect_outside_bar_setups_nb()` (pattern_detector.py:1348-1433) - NEW**
   - Detects pure 3-? setups (outside bar with forming bar as implicit "1")
   - Bidirectional: LONG trigger = bar HIGH, SHORT trigger = bar LOW

4. **Crypto Scanner (signal_scanner.py:674-860)**
   - Now creates TWO signals per directional/outside bar
   - One LONG signal, one SHORT signal
   - Entry monitor handles which triggers first

5. **Equities Scanner (paper_signal_scanner.py:858-1021)**
   - Same bidirectional signal creation (CALL/PUT)

### Test Results

- 297 STRAT tests passing (2 skipped)
- 14 signal automation tests passing
- Both daemons running on VPS

### Commits

- `3710c32` - fix(strat): implement bidirectional setup detection for correct entry timing

### Next Session Priorities

| Priority | Task | Details |
|----------|------|---------|
| Monitor | Watch daemon logs | Verify bidirectional signals generating correctly |
| Monitor | Test entry timing | Confirm entries on forming bar break, not bar close |
| Optional | Add 3-? outside bar setups | Use new `detect_outside_bar_setups_nb()` in scanners |

---

## Session EQUITY-18: 15m/30m Timeframe Scanning (COMPLETE)

**Date:** December 18, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - 15m/30m scanning deployed to VPS

### Changes Implemented

1. **Setup Validation (ported from crypto scanner)**
   - File: `strat/paper_signal_scanner.py` around line 1082
   - Validates X-1 patterns: bars must stay inside setup bar range
   - Validates X-2 patterns: entry level must not be already triggered

2. **15m/30m Data Fetching**
   - Added `'15m'` and `'30m'` support to `_fetch_data()` method
   - Uses Alpaca VBT `15Min` and `30Min` timeframes
   - Market hours filtering applied to all intraday data

3. **Config Updates** (`strat/signal_automation/config.py`)
   - Added `fifteen_min_cron`: `'0,15,30,45 9-15 * * mon-fri'`
   - Added `thirty_min_cron`: `'0,30 9-15 * * mon-fri'`
   - Added `scan_15m` and `scan_30m` enable flags
   - Updated `valid_timeframes` to include `'15m'` and `'30m'`

4. **Scheduler Updates** (`strat/signal_automation/scheduler.py`)
   - Added `add_15m_job()` method
   - Added `add_30m_job()` method

5. **Daemon Updates** (`strat/signal_automation/daemon.py`)
   - Added 15m/30m scan job registration in legacy mode
   - Extended `_is_intraday_entry_allowed()` with time thresholds:
     - 15m: 2-bar 9:45 AM, 3-bar 10:00 AM
     - 30m: 2-bar 10:00 AM, 3-bar 10:30 AM
     - 1H: 2-bar 10:30 AM, 3-bar 11:30 AM

### Test Results

- All tests pass: 311 passed, 2 skipped
- 15m data fetch verified: 594 bars for SPY
- 30m data fetch verified: 308 bars for SPY

### Commits

- `e631970` - feat(strat): add 15m/30m timeframe scanning support

### VPS Deployment

Daemon restarted with new code. Running with `--execute` flag.

---

## Session EQUITY-19 Priority Tasks

| Priority | Task | Details |
|----------|------|---------|
| **HIGH** | Enable 15m/30m scanning | Set `enable_htf_resampling=False` in config to use new jobs |
| Monitor | Watch daemon logs | Look for 15m/30m scans during market hours |
| Low | Dashboard integration | Show 15m/30m signals on dashboard |

### Note on HTF Resampling Mode

Currently, the daemon uses `enable_htf_resampling=True` by default, which uses a single 15-min base scan that resamples to all higher timeframes. The new 15m/30m scan jobs are in the "legacy" mode (`enable_htf_resampling=False`).

To enable the new dedicated 15m/30m scan jobs, either:
1. Set `enable_htf_resampling=False` in config
2. Or set env var `SIGNAL_ENABLE_HTF_RESAMPLING=false`

---

## Session EQUITY-17: Discord Alerts Fix + Premarket Analysis (COMPLETE)

**Date:** December 18, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Discord working, premarket test script ready

### Issues Fixed

1. **Discord Webhook Routing**
   - **Problem:** Equity options alerts were going to crypto-alerts channel
   - **Fix:** Added `DISCORD_EQUITY_WEBHOOK_URL` env var on VPS
   - **Config:** Updated `strat/signal_automation/config.py` to prefer equity webhook
   - **Result:** Alerts now arrive in trade-alerts channel

2. **Merged Claude Web Bug Fixes** (5 commits from `claude/review-daemon-strategy-T4kTM`)
   - Time filtering ("Let the Market Breathe") for hourly patterns
   - Discord alerts on ALL execution paths
   - Pattern tracking via OSI symbol for closed trades
   - Component count for 2-bar vs 3-bar detection

3. **SSH Key Configuration**
   - Removed passphrase from SSH key for easier VPS access

### Premarket Analysis Script Created

Created `scripts/premarket_pipeline_test.py` that:
- Fetches live 15m premarket data from Alpaca
- Classifies bars and detects patterns
- Sends Discord alerts for verification
- Verified 11 signals sent to correct Discord channel

### Bar Classification Verified

All 11 tickers verified correct classification (SPY, QQQ, IWM, DIA, AAPL, TSLA, MSFT, GOOGL, HOOD, QBTS, ACHR)

### Commits

- `6d4d420` - Merge branch 'claude/review-daemon-strategy-T4kTM' - bug fixes
- `25d5142` - fix(discord): use equity-specific webhook URL for signal daemon
- `f0e7ff4` - feat(scripts): add live premarket 15m data fetching to pipeline test

---

## Session EQUITY-18 Priority Tasks

| Priority | Task | Details |
|----------|------|---------|
| **HIGH** | 15m/30m/1H Scanning | Add faster timeframes to daemon for theoretical entries |
| Medium | Setup Validation | Add validation logic to equities scanner (consistency with crypto) |
| Monitor | Watch daemon logs | First scan at 9:00 AM ET |

### 15m/30m Scanning Design (Ready to Implement)

**Data Fetching:**
- 15m/30m: Use Alpaca direct fetch (already in premarket script)
- 1H+: Keep existing VBT fetch

**Schedule:**
- 15m: Scan at :00, :15, :30, :45
- 30m: Scan at :00, :30
- 1H: Scan at :00

**Time Filtering ("Let the Market Breathe"):**
```
15m 2-bar: 9:45 AM    15m 3-bar: 10:00 AM
30m 2-bar: 10:00 AM   30m 3-bar: 10:30 AM
1H 2-bar:  10:30 AM   1H 3-bar:  11:30 AM (existing)
```

### Setup Validation Enhancement (Optional)

**Context:** Crypto scanner has validation logic that equities lacks.

**What crypto has:**
```python
# For X-1 patterns: Valid if bars stay inside setup bar range
# For X-2 patterns: Valid if entry level NOT yet triggered
#   - LONG: valid if bar_high < entry_price
#   - SHORT: valid if bar_low > entry_price
```

**Why it matters:** Without validation, scanner may return "stale" setups where entry was already triggered historically.

**File:** `strat/paper_signal_scanner.py` around line 1075

**Reference:** See commit `79bffa9` in crypto scanner for validation logic.

**Priority:** Low - Not breaking anything, just consistency improvement.

---

## Session CRYPTO-16: CFM Products + X-2 SETUP Detection (COMPLETE)

**Date:** December 18, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Both fixes deployed to VPS

### Part 1: CFM Product Discovery

**Coinbase API DOES have BIP data** - just under a different product ID format than expected.

### Discovery Process

1. GPT 5.2 research suggested CFM-specific endpoints might exist
2. Tested `/api/v3/brokerage/cfm/balance_summary` - WORKS (user has CFM account)
3. Found FCM venue products in API product listing
4. **BIP-20DEC30-CDE** is the correct product ID (not BTC-PERP)

### Key Finding

| Product | Format | Venue | API Access |
|---------|--------|-------|------------|
| BTC-PERP-INTX | BTC-PERP-{VENUE} | INTX (International) | Works |
| BIP | BIP-{EXPIRY}-CDE | FCM/CDE (US Derivatives) | Works |
| ETP | ETP-{EXPIRY}-CDE | FCM/CDE (US Derivatives) | Works |

### Config Changes

Updated `crypto/config.py`:
- `CRYPTO_SYMBOLS` changed from `["BTC-PERP-INTX", "ETH-PERP-INTX"]` to `["BIP-20DEC30-CDE", "ETP-20DEC30-CDE"]`
- Added `SYMBOL_TO_BASE_ASSET` mapping (BIP->BTC, ETP->ETH)
- Added `FUTURES_EXPIRY` with expiration dates (2025-12-30)

### Pattern Detection Verified

| Symbol | Signals | Sample Magnitude |
|--------|---------|------------------|
| BIP-20DEC30-CDE | 9 | 16.47% (vs old 0.038%) |
| ETP-20DEC30-CDE | 9 | 24.28% |

**This solves the original problem** - patterns now have proper magnitudes and will pass filters.

### Commits

- `f080419` - feat(crypto): switch to CFM venue products (BIP/ETP) for pattern detection

### Deployment

- Pushed to GitHub
- Pulled on VPS
- Restarted atlas-crypto-daemon
- Daemon logs confirm: `NEW SIGNAL: BIP-20DEC30-CDE 3-2U LONG (1w) [COMPLETED]`

### Tradovate Decision

**NOT NEEDED** - $25/month subscription can be avoided. Coinbase API provides all required data.

### Contract Rollover Note

BIP/ETP are dated futures (expire Dec 30, 2025). Before expiration:
1. Check for next contract (e.g., BIP-20MAR31-CDE)
2. Update `CRYPTO_SYMBOLS` and `FUTURES_EXPIRY` in config

---

### Part 2: X-2 SETUP Detection for Crypto Scanner

**Problem:** User noticed 2U-2U-2D pattern forming on BIP but no alert fired.

**Root Cause:** Crypto scanner (`crypto/scanning/signal_scanner.py`) only detected X-1 setups (inside bar patterns), not X-2 setups (directional bar patterns).

**Additional Bug:** Setup validation logic checked if subsequent bars stayed "inside" the setup bar range - this was WRONG for X-2 patterns where entry is at a specific price level, not a range break.

### Fixes Applied

1. **Added 3-2 and 2-2 SETUP detection** (ported from equities scanner):
   - `detect_322_setups_nb()` - 3-2D waiting for 3-2D-2U, 3-2U waiting for 3-2U-2D
   - `detect_22_setups_nb()` - X-2D waiting for X-2D-2U, X-2U waiting for X-2U-2D

2. **Fixed setup validation logic**:
   - X-1 patterns: Valid if bars stay inside setup bar range
   - X-2 patterns: Valid if entry level not yet triggered

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/signal_scanner.py` | +218 lines - Added X-2 setup detection and fixed validation |

### Commits

- `f080419` - feat(crypto): switch to CFM venue products (BIP/ETP) for pattern detection
- `79bffa9` - feat(crypto): add 3-2 and 2-2 SETUP detection to signal scanner

### Verification

Daemon now detects SETUP signals:
```
NEW SIGNAL: BIP-20DEC30-CDE 3-2D-? LONG (4h) [SETUP]
NEW SIGNAL: BIP-20DEC30-CDE 2U-2D-? LONG (4h) [SETUP]
Added 4 SETUP signals to entry monitor
```

### Pending: Equities Scanner

The equities scanner (`strat/paper_signal_scanner.py`) does NOT have the same bug (it doesn't reject X-2 setups), but it also lacks validation. Adding the same validation would be a consistency improvement but is not critical.

---

### Session CRYPTO-17 Options

| Priority | Task |
|----------|------|
| Monitor | Verify BIP/ETP SETUP signals generate alerts |
| Observe | Watch for TRIGGER entries with proper magnitudes |
| Optional | Add setup validation to equities scanner for consistency |
| Future | Contract rollover before Dec 30, 2025 expiration |

---

## Session CRYPTO-15: Critical SETUP Detection Bug Fix (COMPLETE)

**Date:** December 17, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Fix deployed to VPS

### Problem Identified

User reported AAPL formed a 3-2D-2U pattern today but no alerts/orders were generated. Investigation revealed:

1. **Scanner detected 3-2D as COMPLETED** (historical PUT entry)
2. **No SETUP created for potential 3-2D-2U** (CALL at 2D bar's high)
3. When 2U formed live, nothing triggered because no SETUP was watching for it

### Root Cause

`_detect_setups()` in `paper_signal_scanner.py` only detected patterns ending in inside bars:
- 3-1 setups (outside + inside)
- 2-1 setups (directional + inside)

**MISSING:** Directional bar setups waiting for OPPOSITE direction breaks:
- 3-2D waiting for 3-2D-2U (CALL at 2D bar high)
- 3-2U waiting for 3-2U-2D (PUT at 2U bar low)

The numba functions `detect_322_setups_nb()` and `detect_22_setups_nb()` existed but were never called.

### Fix Applied

Added two new sections to `_detect_setups()`:

1. **3-2 Setups** using `detect_322_setups_nb()`:
   - 3-2D creates CALL SETUP (entry at 2D bar high)
   - 3-2U creates PUT SETUP (entry at 2U bar low)

2. **2-2 Setups** using `detect_22_setups_nb()`:
   - X-2D creates CALL SETUP (entry at 2D bar high)
   - X-2U creates PUT SETUP (entry at 2U bar low)

### Files Modified

| File | Changes |
|------|---------|
| `strat/paper_signal_scanner.py` | +165 lines - Added 3-2 and 2-2 SETUP detection sections |

### Verification

```python
# Mock 3-2D pattern
# Result: "3-2D-? CALL - Entry: $250.00, Signal Type: SETUP"
```

### STRAT Principle Applied

"Where is the next 2?" - After any directional bar closes, we watch BOTH directions. Pattern evolution:
- 3-2D can become 3-2D-2U (CALL entry at 2D bar high)
- 3-2U can become 3-2U-2D (PUT entry at 2U bar low)

### Commits

- `2226b8f` - fix(strat): add X-2D and X-2U opposite direction SETUP detection

### Deployment

- Pushed to GitHub
- Pulled on VPS
- Restarted atlas-daemon

### Session CRYPTO-16 Options

| Priority | Task |
|----------|------|
| Monitor | Watch for proper SETUP detections in daemon logs |
| Verify | Check if patterns like AAPL 3-2D-2U now generate alerts |
| Decision | Tradovate API subscription still pending ($25/mo) |

---

## Session CRYPTO-14: Tradovate API Research and Coinbase API Verification (COMPLETE)

**Date:** December 17, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Decision pending on Tradovate subscription

### Objective

Research Tradovate API as alternative data source for BIP futures, verify Coinbase API limitations.

### Key Findings

**1. Coinbase API Confirmed Limited to INTX Only**
- Tested multiple CDX product ID formats: `BTC-PERP`, `BTC_PERP`, `BTC-PERP-CDX`
- ALL returned 404 errors
- API only exposes 199 INTX products
- Web UI shows CDX products, but API does NOT

**2. Tradovate Has BIP**
- User confirmed BIP/BIPZ0 visible in Tradovate product search
- Contract available via Coinbase Derivatives Exchange connection
- API access requires $25/month subscription (free account doesn't include API)

**3. MFF Account Limitations**
- MyFundedFutures credentials do NOT grant Tradovate API access
- User created separate free Tradovate account
- API tier costs $25/month additional

### API Test Results (test_cfm_products.py)

| Product ID | Status |
|------------|--------|
| BTC-PERP | 404 Not Found |
| BTC_PERP | 404 Not Found |
| BTC-PERP-CDX | 404 Not Found |
| BTC-PERP-INTX | Works ($87,836) |

### Architecture Research

Tradovate API characteristics:
- WebSocket-based market data (not REST)
- `md/getChart` subscription for OHLCV
- Historical minute data back to 2017
- REST only for authentication

### Files Modified

| File | Changes |
|------|---------|
| `scripts/test_cfm_products.py` | Added CDX product ID tests |

### Plan Created

Implementation plan ready at: `C:\Users\sheeh\.claude\plans\sprightly-toasting-hinton.md`

Phases:
1. Abstract ExchangeClientBase interface (~50 LOC)
2. TradovateClient WebSocket implementation (~400-500 LOC)
3. Daemon integration with config-driven data source

### Decision Pending

User needs to decide on $25/month Tradovate API subscription:
- **Yes**: Proceed with integration (4-5 hours implementation)
- **No**: Explore workarounds (adjust INTX filters, manual TradingView export)

### Session CRYPTO-15 Options

| If Decision | Next Steps |
|-------------|------------|
| Subscribe to Tradovate | Implement TradovateClient per plan |
| Don't subscribe | Explore INTX filter adjustments or manual alternatives |

### Plan Mode Recommendation

**PLAN MODE: OFF** - Plan already exists, awaiting user decision.

---

## Session CRYPTO-13: CFM API Research and Non-Uniform Basis Discovery (COMPLETE)

**Date:** December 16, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Critical finding: INTX has non-uniform basis vs BIP

### Objective

Research CFM API for BIP data and validate pattern detection accuracy.

### Critical Discovery: Non-Uniform Basis Spread

Comparing INTX vs BIP for the same hourly 3-2D-2U pattern (Dec 16, 08:00-10:00 EST):

| Level | BIP | INTX | Spread |
|-------|-----|------|--------|
| 3 Bar HIGH | $89,495 | $87,800 | **$1,695** |
| 3 Bar LOW | $86,405 | $86,413 | **$8** |
| 2D Bar HIGH | $87,840 | $87,767 | **$73** |

**Key Finding:** The basis is NOT uniform - highs differ by $1,695 but lows are nearly identical!

### Why Trades Are Being Filtered

**BIP Magnitude:**
- Target: $89,495, Entry: $87,840
- Magnitude: $1,655 (1.88%) - PASSES filter

**INTX Magnitude:**
- Target: $87,800, Entry: $87,767
- Magnitude: $33 (0.038%) - **FAILS filter** (MIN_MAGNITUDE_PCT = 0.5%)

The INTX outside bar barely clears the 2D high, creating a "compressed" geometry that fails magnitude filters even though BIP shows a valid setup.

### API Research Findings

1. **CFM Products NOT in Advanced Trade API**
   - Tested: Only 199 INTX products available
   - BTC-PERP and ETH-PERP return 404 errors
   - CFM nano futures (BIP/EIP) not accessible

2. **Alternative Data Sources Identified**
   - **Tradovate API** (user has account via MyFundedFutures)
     - Contract ID: BIPZ0
     - Historical minute data available
   - **TradingView** (user has paid account with Coinbase integration)
     - Direct BIP charts available

### Files Created

| File | Purpose |
|------|---------|
| `scripts/test_cfm_products.py` | API test for product discovery |

### Session CRYPTO-14 Priorities

1. **Integrate Tradovate API** - Fetch BIP OHLCV data
2. **Create data source abstraction** - Support multiple data feeds
3. **Test pattern detection with BIP data** - Verify correct magnitude calculations

### Plan Mode Recommendation

**PLAN MODE: ON** - Tradovate API integration requires architectural planning.

---

## Session CRYPTO-12: Wrong Coinbase Product Discovery (COMPLETE)

**Date:** December 16, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Led to CRYPTO-13 research

### Critical Discovery

The crypto daemon is fetching data for the **WRONG PRODUCT**:

| Product | What Daemon Uses | What User Trades |
|---------|------------------|------------------|
| **BTC** | BTC-PERP-INTX (INTX venue) | BIP - Nano Bitcoin Perp (CFM venue) |
| **ETH** | ETH-PERP-INTX (INTX venue) | EIP - Nano Ether Perp (CFM venue, TBD) |

**Why This Matters:**
- BTC-PERP-INTX is on Coinbase International Exchange (non-US users)
- BIP (Nano Bitcoin) is on CFM (US users can access)
- Prices differ by ~$1,695 on the same bar!
- Pattern detection was working correctly, but on wrong data

### Data Comparison (08:00 EST bar on Dec 16)

| Source | HIGH |
|--------|------|
| BTC-PERP-INTX API | $87,800 |
| BIP (user's chart) | $89,495 |

### Investigation Steps Completed

1. Verified target calculation in `pattern_detector.py` is CORRECT
2. Compared Coinbase Advanced Trade API vs INTX API - both return same data
3. Discovered user views "Nano Bitcoin Perp Style Futures" (BIP), not BTC-PERP-INTX
4. CFM products (BIP, EIP) not found via Advanced Trade API - may need different endpoint

### Session CRYPTO-13 Priorities

1. **Find CFM API endpoint** - Locate correct API for BIP/EIP nano products
2. **Update crypto config** - Change symbols from INTX to CFM products
3. **Update Coinbase client** - Add CFM data fetching if needed
4. **Test pattern detection** - Verify with correct data source

### Files to Investigate

| File | Purpose |
|------|---------|
| `crypto/config.py` | Symbol configuration (BTC-PERP-INTX -> BIP) |
| `crypto/exchange/coinbase_client.py` | May need CFM API endpoint |

### Product Details (BIP - Nano Bitcoin)

- Contract size: 1/100th of Bitcoin
- Trading code: BIP
- Venue: CFM (Coinbase Financial Markets)
- Hours: 24/7 with 1-hour break Friday 5-6PM ET
- Expiry: December 2030 (5-year contract)

### Plan Mode Recommendation

**PLAN MODE: ON** - Need to research CFM API and update data fetching.

---

## Session CRYPTO-11: Equity Daemon Bug Fixes (COMPLETE)

**Date:** December 16, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Equity daemon now matches crypto daemon fixes

### Objective

Apply the same STRAT bug fixes from crypto daemon to equity daemon before market open.

### Bugs Fixed

**Bug 1: Live Bar as Setup Bar (paper_signal_scanner.py)**
- Added `last_bar_idx` skip in 3-1 and 2-1 setup detection loops (lines 694-703, 775-780)
- Mirrors fix in `crypto/scanning/signal_scanner.py`

**Bug 2: Unidirectional Trigger Checking (entry_monitor.py)**
- Replaced unidirectional CALL/PUT check with bidirectional monitoring
- Now checks BOTH `setup_bar_high` and `setup_bar_low`
- Implements "Where is the next 2?" per STRAT methodology
- Stores `_actual_direction` on trigger event

**Bug 3: Direction Handling (daemon.py)**
- Added `getattr(event, '_actual_direction', signal.direction)` logic
- Updates signal.direction if opposite break detected
- Logs "DIRECTION CHANGED" when pattern breaks opposite

### Files Modified

| File | Changes |
|------|---------|
| `strat/paper_signal_scanner.py` | +14 lines - live bar skip for 3-1 and 2-1 loops |
| `strat/signal_automation/entry_monitor.py` | +40/-16 lines - bidirectional trigger logic |
| `strat/signal_automation/daemon.py` | +14 lines - actual_direction handling |

### Skill File Sync

Copied `IMPLEMENTATION-BUGS.md` from project space to:
`C:/Users/sheeh/.claude/skills/strat-methodology/IMPLEMENTATION-BUGS.md`

### Verification

- 297 STRAT tests passing (2 skipped)
- 14 signal automation tests passing
- VPS daemon showing "DIRECTION CHANGED" logs correctly
- Bidirectional monitoring confirmed working:
  - `MSFT 2U-1-? CALL -> PUT`
  - `TSLA 3-1-? PUT -> CALL`
  - `QQQ 2D-1-? CALL -> PUT`

### Commits

| Hash | Message |
|------|---------|
| `adc61b4` | fix(strat): apply crypto daemon bug fixes to equity daemon |

### Session CRYPTO-12 Priorities

1. **Monitor equity daemon** - Watch for correct direction handling during market hours
2. **Live Trading Mode for Crypto** - Enable execution (deferred from CRYPTO-7)
3. **Options Backtest Bug** - Fix direction inference in backtest script

### Plan Mode Recommendation

**PLAN MODE: OFF** - Monitoring work.

---

## Session CRYPTO-10: Review and Merge STRAT Bug Fixes (COMPLETE)

**Date:** December 16, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Merged Claude Code for Web branch, updated tests

### Objective

Review and merge STRAT implementation bug fixes from Claude Code for Web session that discovered issues with:
1. Live bar being treated as setup bar for 3-bar patterns
2. Incorrect stop placement using inside bar instead of first directional bar

### Branch Reviewed

`claude/review-daemon-strategy-T4kTM` - 2 commits merged

### Bug 1: Live Bar as Setup Bar

**Root Cause:** The signal scanner was detecting the live (incomplete) bar as an inside bar for 3-bar setups. Entry would trigger on the SAME bar that was used as the setup.

**Example:** User reported "a 2D-1 BTC trade yesterday where the entry was on the live inside bar"

**Fix:** Exclude the last bar from setup detection in `crypto/scanning/signal_scanner.py`:
```python
last_bar_idx = len(setup_mask) - 1
for i in range(len(setup_mask)):
    if i == last_bar_idx:  # Skip live bar
        continue
```

### Bug 2: Incorrect Stop Placement

**Root Cause:** 2-1-2 pattern stops were placed at inside bar (index i-1) instead of first directional bar (index i-2).

**Example:** User reported "stop was the high of the live inside bar, instead of the 2D directional bar before it"

**Fix:** Changed stop placement in `strat/pattern_detector.py`:
```python
# BEFORE (incorrect):
stops[i] = low[i-1]   # Inside bar

# AFTER (per STRAT methodology):
stops[i] = low[i-2]   # First directional bar
```

### STRAT Methodology Validation

Per strat-methodology skill, for 2-1-2 patterns:
- Stop goes at the **first directional bar's extreme** (structural support/resistance)
- Inside bar is just consolidation; structural level comes from directional bar before it

| Pattern | Stop Location |
|---------|---------------|
| 2U-1-2U (bullish) | First 2U bar LOW |
| 2D-1-2D (bearish) | First 2D bar HIGH |
| 2D-1-2U (reversal) | First 2D bar LOW |
| 2U-1-2D (reversal) | First 2U bar HIGH |

Note: 3-1-2 patterns were ALREADY correct (using outside bar for stop).

### Files Modified

| File | Changes |
|------|---------|
| `strat/pattern_detector.py` | Stop placement: i-1 -> i-2 for 2-1-2 patterns |
| `crypto/scanning/signal_scanner.py` | Live bar exclusion for 3-bar setup detection |
| `tests/test_strat/test_pattern_detector.py` | Updated test expectations for correct stop values |

### Documentation Added

New file: `docs/Claude Skills/strat-methodology/IMPLEMENTATION-BUGS.md` (236 lines)
- Live bar vs closed bar detection rules
- 3-bar vs 2-bar pattern distinction
- Stop placement rules with visual examples
- Entry trigger mechanics
- Summary checklist for future implementations

### Commits

| Hash | Message |
|------|---------|
| `1b8e295` | fix(strat): exclude live bars from 3-bar setup detection and fix stop placement |
| `276d307` | docs(strat): add implementation bugs guide for Claude Code sessions |
| `739f2df` | test(strat): update 2-1-2 pattern tests for correct stop placement |

### Test Results

- 297 STRAT tests passing (2 skipped - API credentials)
- All 16 pattern detector tests pass with updated expectations

### Session CRYPTO-11 Priorities

1. **Deploy to VPS** - Push merged fixes to production
2. **Monitor crypto daemon** - Verify no more same-candle entry/exit trades
3. **Live Trading Mode** - Enable crypto execution (deferred from CRYPTO-7)

### Plan Mode Recommendation

**PLAN MODE: OFF** - Monitoring and verification work.

---

## Session CRYPTO-9: Cron Fix + Dashboard Closed Trades Enhancement (COMPLETE)

**Date:** December 15, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Cron fix deployed, dashboard enhanced

### Part 1: Critical Cron Day-of-Week Bug Fix

Investigate and fix why the equity/options STRAT daemon had zero activity during today's market session (Monday).

### Root Cause Discovered

**CRITICAL BUG:** All cron expressions used `1-5` for day_of_week, but APScheduler interprets this as Tuesday-Saturday, NOT Monday-Friday.

**APScheduler weekday mapping:**
- 0 = Monday
- 1 = Tuesday
- 2 = Wednesday
- 3 = Thursday
- 4 = Friday
- 5 = Saturday
- 6 = Sunday

**Impact:** Scans never ran on Mondays. Today is Monday, so zero scans occurred despite the daemon running.

### Evidence

```python
# Test with 1-5 (old config - WRONG)
CronTrigger(day_of_week='1-5')  # Tue-Sat
# Result: Next run = Tuesday (skips Monday)

# Test with mon-fri (new config - CORRECT)
CronTrigger(day_of_week='mon-fri')  # Mon-Fri
# Result: Next run = Monday (same day)
```

### Files Modified

| File | Changes |
|------|---------|
| `strat/signal_automation/config.py` | Fixed 4 cron expressions: 1-5 -> mon-fri, 5 -> fri |

### Key Fixes

1. **hourly_cron:** `'30 9-15 * * 1-5'` -> `'30 9-15 * * mon-fri'`
2. **daily_cron:** `'0 17 * * 1-5'` -> `'0 17 * * mon-fri'`
3. **weekly_cron:** `'0 18 * * 5'` -> `'0 18 * * fri'` (was running Saturday)
4. **base_scan_cron:** `'30,45,0,15 9-15 * * 1-5'` -> `'30,45,0,15 9-15 * * mon-fri'`

### Commits

| Hash | Message |
|------|---------|
| `553ee45` | fix(strat): correct cron day-of-week from 1-5 to mon-fri |

### Verification

After fix deployment, tested cron schedule on VPS:
- Before: Next run = Tomorrow (Tuesday) 9:00 AM
- After: Next run = Today (Monday) 3:15 PM

Scans now correctly scheduled for remaining market hours today (3:15, 3:30, 3:45 PM).

### Part 2: Dashboard Closed Trades Enhancement

Enhanced the crypto dashboard closed trades table with additional information.

**Files Modified:**

| File | Changes |
|------|---------|
| `crypto/simulation/paper_trader.py` | Added tfc_score field to SimulatedTrade |
| `crypto/scanning/daemon.py` | Pass TFC score from signal context when opening trades |
| `dashboard/components/crypto_panel.py` | Enhanced closed trades table display |

**Closed Trades Table Enhancements:**
- Pattern (pattern_type + timeframe) displayed with symbol
- Entry time displayed alongside entry price
- TFC score column with color highlighting (green for 3+/4)
- Consolidated columns (Entry/Exit show price + time)

**Commits:**

| Hash | Message |
|------|---------|
| `58a5cf4` | feat(crypto): enhance closed trades dashboard with pattern, entry time, TFC |

### Part 3: Equity/Options Closed Trades API Fix

User reported that the Options Trading tab showed 0 closed trades despite 5 positions closing today.

**Root Cause:** The `get_fill_activities()` method in `alpaca_trading_client.py` was calling `self.client.get_activities()` which doesn't exist in alpaca-py's TradingClient.

**Fix:** Changed to use raw API call: `self.client.get('/account/activities/FILL', params)`

**Additional fixes:**
- Date format changed to RFC3339 (`YYYY-MM-DDTHH:MM:SSZ`) as required by Alpaca API
- Response parsing updated since raw API returns dicts, not objects
- ISO timestamp parsing for transaction_time

**Files Modified:**

| File | Changes |
|------|---------|
| `integrations/alpaca_trading_client.py` | Fixed get_fill_activities() to use raw API call |

**Commits:**

| Hash | Message |
|------|---------|
| `68255f6` | fix(alpaca): use raw API for get_fill_activities() |

**Verification:**
- Before fix: 0 fills, error "TradingClient object has no attribute get_activities"
- After fix: 24 fills, 12 closed trades with P&L correctly calculated

### Crypto Daemon Status

The crypto daemon was working correctly - not affected by these bugs. It detected 21 signals across 6 scans today.

### Session CRYPTO-10 Priorities

1. **Monitor equity daemon scans** - Verify scans run tomorrow (Tuesday)
2. **Monitor pattern detection** - Verify bidirectional SETUP monitoring in production
3. **Live Trading Mode** for crypto (deferred from CRYPTO-7)

### Plan Mode Recommendation

**PLAN MODE: OFF** - Monitoring and operational work.

---

## Session CRYPTO-8: STRAT Pattern Transition Detection Fix (COMPLETE)

**Date:** December 15, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Critical STRAT entry logic fixed

### Objective

Fix incorrect trade entries caused by misclassifying bars and not detecting when SETUP patterns become 2-bar patterns.

### Root Cause Discovered

STRAT principle "Where is the next 2?" was not implemented correctly:
- A SETUP (X-1-?) was assigned predetermined direction (LONG or SHORT)
- When the inside bar broke OPPOSITE direction, the trade was still entered in original direction
- Example: 2U-1-? LONG setup, but bar broke DOWN → should be 2U-2D SHORT, not LONG

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/models.py` | Added prior_bar_type/high/low fields to track pattern transitions |
| `crypto/scanning/signal_scanner.py` | Populate prior bar info for all SETUP signals (2-1, 3-1) |
| `crypto/scanning/entry_monitor.py` | Rewrite check_triggers() to watch BOTH directions |
| `crypto/scanning/daemon.py` | Use actual_direction from trigger event, prevent duplicate positions |
| `crypto/config.py` | Lowered MIN_SIGNAL_RISK_REWARD from 1.5 to 1.0 |
| `dashboard/components/crypto_panel.py` | Added entry_time display to open positions |

### Key Fixes

1. **Bidirectional SETUP Monitoring**
   - Entry monitor now watches BOTH inside bar high AND low
   - Break above → X-1-2U → LONG
   - Break below → X-2D (2-bar pattern) → SHORT

2. **Pattern Transition Detection**
   - When inside bar breaks down: 2U-1-? → 2U-2D (reversal SHORT)
   - When inside bar breaks down: 2D-1-? → 2D-2D (continuation SHORT)
   - Prior bar info stored to identify resulting 2-bar pattern

3. **Duplicate Position Prevention**
   - Check for existing position before opening new trade
   - Prevents multiple positions on same symbol from different timeframes

4. **R:R Threshold Lowered**
   - Changed from 1.5 to 1.0 to allow daily timeframe patterns
   - Daily patterns typically have R:R 0.6-1.3 due to target placement

### Commits

| Hash | Message |
|------|---------|
| `6a80ed6` | fix(crypto): lower R:R threshold to 1.0 for daily patterns |
| `dcf6fc8` | fix(crypto): prevent duplicate positions and add entry times to dashboard |
| `8296211` | fix(crypto): detect SETUP→2-bar pattern transitions in entry monitor |

### Investigation: 4H BTC Misclassification

**Issue:** 4H BTC bar classified as inside (1) when it was actually 2D (broke down)

**Root Cause:** Bar classification was correct at scan time, but bar continued forming and broke down after scan. Entry monitor didn't detect the pattern change.

**Fix:** Entry monitor now checks if "inside bar" has broken its bounds in either direction:
- Broke UP → pattern completes as X-1-2U
- Broke DOWN → pattern becomes X-2D (2-bar)

### Session CRYPTO-9 Priorities

1. **Dashboard Closed Trades Enhancement**
   - Add pattern/setup traded to closed trades section
   - Add entry time (currently only exit time shown)
   - Add TFC score for each trade

2. **Monitor Pattern Detection**
   - Verify bidirectional SETUP monitoring works in production
   - Confirm 2-bar pattern detection (2U-2D, 2D-2U, etc.)

3. **Live Trading Mode** - Enable execution (deferred from CRYPTO-7)

### Plan Mode Recommendation

**PLAN MODE: OFF** - Dashboard enhancement is straightforward implementation.

---

## Session CRYPTO-7: Discord Trade Alerts + VPS Deployment (COMPLETE)

**Date:** December 15, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Entry/exit Discord alerts, VPS REST API deployed

### Objective

Wire Discord alerts for actual trade execution (entry/exit with P&L), deploy REST API to VPS.

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/daemon.py` | Added entry/exit alert calls, config flags for alert types |

### Key Features

1. **Discord Trade Alerts (NEW)**
   - Entry alerts: symbol, pattern, price, quantity, leverage
   - Exit alerts: symbol, exit reason, entry/exit price, P&L dollar and percent
   - Config flags to control alert types (entry/exit enabled, signal/trigger disabled)

2. **VPS Deployment (COMPLETE)**
   - REST API deployed and running on port 8080
   - Firewall opened: `sudo ufw allow 8080/tcp`
   - Both daemons running: atlas-daemon, atlas-crypto-daemon

### Config Flags Added

| Flag | Default | Purpose |
|------|---------|---------|
| `alert_on_trade_entry` | True | Alert when trade executes |
| `alert_on_trade_exit` | True | Alert when trade closes with P&L |
| `alert_on_signal_detection` | False | Alert on pattern detection (noisy) |
| `alert_on_trigger` | False | Alert when SETUP price hit |

### Commits

| Hash | Message |
|------|---------|
| `5c24481` | feat(crypto): wire Discord entry/exit trade alerts |

### Session CRYPTO-8 Priorities

1. **Debug Daily Alerts** - Add logging to find why daily timeframe patterns not alerting
2. **Live Trading Mode** - Enable execution in daemon (currently paper only)
3. **Position Exit Tracking** - Track stop/target hits in dashboard

### Plan Mode Recommendation

**PLAN MODE: OFF** - Debugging and operational work.

---

## Session CRYPTO-6: Dashboard Integration via REST API (COMPLETE)

**Date:** December 14, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - REST API + Dashboard crypto panel

### Objective

Add crypto paper trading panel to dashboard via REST API from VPS daemon.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/api/__init__.py` | API module init | 18 |
| `crypto/api/server.py` | Flask REST API server | 232 |
| `dashboard/data_loaders/crypto_loader.py` | Dashboard data loader | 311 |
| `dashboard/components/crypto_panel.py` | Dashboard panel component | 1012 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/daemon.py` | Added api_enabled, api_host, api_port config; _start_api_server() method |
| `dashboard/data_loaders/__init__.py` | Export CryptoDataLoader |
| `dashboard/app.py` | Import crypto components; init crypto_loader; add Crypto Trading tab; add callbacks |

### Key Features

1. **REST API (Port 8080)**
   - Runs as daemon thread (single service)
   - Endpoints: /health, /status, /positions, /signals, /performance, /trades
   - Auto-starts with daemon

2. **Dashboard Crypto Panel**
   - Account summary: balance, P&L, return %
   - Daemon status: running, leverage tier, scan counts
   - Open positions with unrealized P&L
   - Pending SETUP signals tab
   - Closed trades tab with summary
   - Performance metrics tab
   - 30-second auto-refresh

3. **Architecture**
   - VPS daemon exposes API on port 8080
   - Railway dashboard calls API via CRYPTO_API_URL env var
   - Clean separation of concerns

### Deployment Steps

**VPS:**
```bash
ssh atlas@178.156.223.251
cd ~/vectorbt-workspace && git pull
sudo ufw allow 8080/tcp
sudo systemctl restart atlas-crypto-daemon
curl http://localhost:8080/health
```

**Railway:**
1. Add env var: `CRYPTO_API_URL=http://178.156.223.251:8080`
2. Push to main (auto-deploys)

### Commits

| Hash | Message |
|------|---------|
| `c391111` | feat(crypto): add REST API and dashboard crypto trading panel |

### Investigation: TradingView vs Coinbase Data

During session, investigated discrepancies between Discord alerts and TradingView charts:

- **Finding:** TradingView was missing Dec 13-14 data, causing visual mismatch
- **Root Cause:** Different data sources (TradingView vs Coinbase INTX)
- **Verification:** Bar classification is CORRECT based on Coinbase data
- **Confirmed:** All 5 timeframes (1w, 1d, 4h, 1h, 15m) are being scanned
- **Note:** Use Coinbase data for analysis since we trade on Coinbase INTX

### BUG: Daily Timeframe Alerts Missing

**Issue:** ETH daily 2D-1-2D pattern (Dec 12-14) not alerted despite daemon running.
- Pattern confirmed in Coinbase data: Dec 12 (2D) -> Dec 13 (1) -> Dec 14 (2D)
- Entry trigger $3,078.49 was hit (Dec 14 low = $3,050)
- 4h/1h alerts working, but daily missing

**Investigate in CRYPTO-7:**
1. Add verbose logging for daily timeframe pattern detection
2. Check if daily bars detected correctly during incomplete bar periods
3. Verify `detected_time` handling for daily patterns

### Session CRYPTO-7 Priorities

1. **Debug Daily Alerts** - Add logging, fix missing daily timeframe signals
2. **Live Trading Mode** - Enable execution in daemon
3. **Position Exit Tracking** - Track stop/target hits in dashboard

### Plan Mode Recommendation

**PLAN MODE: OFF** - Execution enablement is operational work.

---

## Session CRYPTO-5: VPS Deployment and Discord Alerts (COMPLETE)

**Date:** December 14, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - VPS deployment, 60s position monitoring, Discord alerts

### Objective

Deploy crypto daemon to VPS for 24/7 operation and add Discord alerts for signals.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/alerters/__init__.py` | Alerters module init | 10 |
| `crypto/alerters/discord_alerter.py` | CryptoDiscordAlerter class | 520 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/daemon.py` | Added discord_webhook_url config, _on_poll callback, Discord integration |
| `crypto/scanning/entry_monitor.py` | Added on_poll callback for 60s position checks |
| `scripts/run_crypto_daemon.py` | Added Discord webhook env var support |
| `deploy/atlas-crypto-daemon.service` | Fixed argument order, added cache paths |

### Key Features

1. **VPS Deployment (LIVE)**
   - Daemon running 24/7 on 178.156.223.251
   - systemd service auto-starts on boot
   - Logs at `/home/atlas/vectorbt-workspace/crypto/logs/daemon.log`

2. **60-Second Position Monitoring**
   - Moved from 5-minute health loop to entry monitor poll
   - Faster stop/target exit detection
   - via `on_poll` callback in entry monitor

3. **Discord Alerts**
   - Rich embeds with color-coded signals (green=LONG, red=SHORT)
   - Leverage tier and TFC score in alerts
   - Trigger alerts for SETUP patterns
   - Separate crypto webhook configured

### Commits

| Hash | Message |
|------|---------|
| `2321b42` | fix(crypto): correct entry_price -> entry_trigger in CLI |
| `0bdadff` | fix(crypto): resolve pandas FutureWarning |
| `75452ea` | fix(deploy): add uv cache paths to systemd |
| `576196f` | fix(deploy): correct argument order |
| `6392b4f` | feat(crypto): add 60s position monitoring via poll |
| `6dd2bf9` | feat(crypto): add Discord alerts for crypto signals |
| `6460f33` | fix(crypto): improve Discord alerter import error logging |
| `67e8981` | fix(crypto): resolve circular import in Discord alerter |
| `1a7a230` | fix(crypto): pass now_et to leverage/intraday functions |

### VPS Status

```bash
# Check daemon status
ssh atlas@178.156.223.251 "sudo systemctl status atlas-crypto-daemon"

# View logs
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon -f"
```

### Session CRYPTO-6 Priorities

1. **Dashboard Integration** - Add crypto paper trading panel
2. **Live Trading** - Enable execution mode (currently signals only)
3. **Performance Tracking** - Aggregate crypto P&L metrics

### Plan Mode Recommendation

**PLAN MODE: ON** - Dashboard integration requires architectural planning.

---

## Session CRYPTO-4: Intraday Leverage and VPS Deployment (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Intraday leverage, VPS deployment, position monitoring

### Objective

Add time-based leverage tier switching and VPS deployment infrastructure.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/run_crypto_daemon.py` | CLI entry point for VPS daemon | 370 |
| `deploy/atlas-crypto-daemon.service` | systemd service file | 50 |
| `crypto/simulation/position_monitor.py` | Stop/target exit monitoring | 250 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/config.py` | Added intraday leverage window (6PM-4PM ET), helper functions |
| `crypto/__init__.py` | Export leverage helpers, bump to v0.4.0 |
| `crypto/scanning/daemon.py` | Time-based leverage in _execute_trade, position monitoring integration |
| `crypto/simulation/paper_trader.py` | Added stop/target/timeframe/pattern to SimulatedTrade |
| `crypto/simulation/__init__.py` | Export CryptoPositionMonitor, ExitSignal |

### Key Features

1. **Time-Based Leverage Tiers**
   - Intraday: 10x available 6PM-4PM ET (22 hours/day)
   - Swing: 4x available 24/7
   - Helper functions: `is_intraday_window()`, `get_max_leverage_for_symbol()`

2. **VPS Deployment**
   - CLI script with start, scan, status, positions, performance, leverage, reset commands
   - systemd service file for production deployment
   - Log file support for daemon mode

3. **Position Monitoring**
   - Stop/target prices stored with trades
   - CryptoPositionMonitor checks exits in health loop
   - Auto-close on stop loss or take profit

### Verified Working

```python
from crypto import is_intraday_window, get_max_leverage_for_symbol

# At 10AM ET (intraday window)
# BTC-PERP-INTX: 10x leverage

# At 5PM ET (4-6PM gap)
# BTC-PERP-INTX: 4x leverage (swing only)
```

```bash
uv run python scripts/run_crypto_daemon.py leverage
# Current Tier: INTRADAY (10x)
# Time until 4PM ET close: 0.1 hours
```

### VPS Deployment Commands

```bash
# On VPS (178.156.223.251)
sudo cp deploy/atlas-crypto-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable atlas-crypto-daemon
sudo systemctl start atlas-crypto-daemon
sudo journalctl -u atlas-crypto-daemon -f
```

### Bug Fix: SETUP Detection Across Inside Bars

**Issue:** SETUP patterns (2-1, 3-1) were only detected if on the last bar. Missed setups when subsequent bars were also inside bars (e.g., 2D-1-1 structure).

**Fix:** Now checks if inside bar high/low was broken by subsequent bars. If still valid, SETUP is included.

**Commit:** `6ed6169` - fix(crypto): detect SETUP patterns that remain valid across inside bars

### Session CRYPTO-5 Priorities

1. **VPS Deployment Test** - Deploy to VPS and verify 24/7 operation
2. **Entry Monitor Enhancement** - More frequent position checks (every 60s vs 5min)
3. **Discord Alerts** - Add crypto signal alerts to Discord
4. **Performance Tracking** - Dashboard integration for crypto paper trades

### Plan Mode Recommendation

**PLAN MODE: OFF** - VPS deployment is operational work.

---

## Session CRYPTO-3: Entry Monitor and Daemon (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Full automation stack implemented

### Objective

Build entry trigger monitoring and daemon orchestration for 24/7 crypto paper trading.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/scanning/entry_monitor.py` | 24/7 trigger polling with maintenance window | 320 |
| `crypto/scanning/daemon.py` | Main orchestration daemon | 675 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/__init__.py` | Export entry monitor and daemon classes |
| `crypto/__init__.py` | Export new classes, bump to v0.3.0 |

### Key Features

1. **CryptoEntryMonitor** - Polls prices every 60s, checks SETUP triggers
2. **CryptoSignalDaemon** - Orchestrates scanner (15min), monitor (1min), paper trader
3. **24/7 Operation** - No market hours filter
4. **Friday Maintenance Window** - Pauses during 5-6 PM ET
5. **Paper Trading Integration** - Wired to PaperTrader for simulated execution
6. **Signal Deduplication** - Prevents duplicate signals across scans

### Verified Working

```python
from crypto import CryptoSignalDaemon

daemon = CryptoSignalDaemon()
signals = daemon.run_scan_and_monitor()
# BTC: 3-2U LONG (1w) [COMPLETED]
# ETH: 2D-1-? LONG (1d) [SETUP] - trigger: $3,136.30

daemon.start(block=False)  # Background mode works
daemon.stop()
```

### Session CRYPTO-4 Priorities

1. **Intraday Leverage** - Update config for 10x intraday (6PM-4PM ET window)
2. **Paper Balance** - Update default to $1,000
3. **VPS Deployment** - Create CLI script for production daemon
4. **Position Monitoring** - Add stop/target tracking for open trades

### Plan Mode Recommendation

**PLAN MODE: OFF** - Implementation continues from established architecture.

---

## Session CRYPTO-2: Crypto STRAT Signal Scanner (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Core scanner implemented and verified

### Objective

Connect crypto module to Atlas STRAT engine for pattern detection on BTC/ETH perpetual futures.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/scanning/__init__.py` | Module initialization | 15 |
| `crypto/scanning/models.py` | CryptoDetectedSignal, CryptoSignalContext dataclasses | 90 |
| `crypto/scanning/signal_scanner.py` | CryptoSignalScanner - core pattern detection | 650 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/config.py` | Added MAINTENANCE_WINDOW config, signal filters |
| `crypto/data/state.py` | Added signal tracking methods (add_detected_signal, get_pending_setups, etc.) |
| `crypto/__init__.py` | Export scanning module, bump to v0.2.0 |

### Key Features

1. **CryptoSignalScanner** - Detects all STRAT patterns (2-2, 3-2, 3-2-2, 2-1-2, 3-1-2)
2. **24/7 Operation** - No market hours filter (crypto is 24/7)
3. **Friday Maintenance Window** - Handles 5-6 PM ET Coinbase INTX maintenance
4. **Multi-Timeframe** - Scans 1w, 1d, 4h, 1h, 15m
5. **TFC Score** - Full Timeframe Continuity calculation
6. **SETUP Detection** - Detects X-1 patterns waiting for live break

### Verified Working

```python
from crypto.scanning import CryptoSignalScanner

scanner = CryptoSignalScanner()
signals = scanner.scan_all_timeframes('BTC-PERP-INTX')
scanner.print_signals(signals)

# Found 13 signals across all timeframes
# Weekly: 3-2U LONG detected
# Daily: 3-1-2D SHORT detected
# 4h: Multiple patterns including SETUP signals
```

### Architecture

```
Coinbase OHLCV Data
       |
       v
classify_bars_nb() [from strat/bar_classifier.py - unchanged]
       |
       v
detect_*_patterns_nb() [from strat/pattern_detector.py - unchanged]
       |
       v
CryptoDetectedSignal objects
       |
       v
CryptoSystemState.add_detected_signal()
```

### Session CRYPTO-3 Priorities

1. **Entry Monitor** - Create `crypto/scanning/entry_monitor.py` for 24/7 trigger polling
2. **Daemon** - Create `crypto/scanning/daemon.py` orchestrator
3. **Paper Trading Integration** - Wire triggers to PaperTrader execution

### Plan Mode Recommendation

**PLAN MODE: OFF** - Architecture established, next session is implementation.

---

## Session 83K-82: Crypto Derivatives Module Integration (COMPLETE)

**Date:** December 12, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Core infrastructure ready for paper trading

### Objective

Integrate BTC/ETH/SOL derivatives trading capability into Atlas using Coinbase Advanced Trade API. This complements the existing equities STRAT options strategy with 24/7 crypto trading.

### Source Project

Adapted code from `C:\Cypto_Trading_Bot` (Gemini 3.0 Pro prototype) - kept Coinbase client, discarded duplicate STRAT code (Atlas has better implementation).

### Files Created

| File | Purpose |
|------|---------|
| `crypto/__init__.py` | Module initialization |
| `crypto/config.py` | Configuration (symbols, risk params, timeframes) |
| `crypto/exchange/__init__.py` | Exchange module |
| `crypto/exchange/coinbase_client.py` | Coinbase API client with public API fallback |
| `crypto/data/__init__.py` | Data module |
| `crypto/data/state.py` | System state management (bar classifications, positions) |
| `crypto/trading/__init__.py` | Trading module |
| `crypto/trading/sizing.py` | ATR-based position sizing with leverage limits |
| `crypto/simulation/__init__.py` | Simulation module |
| `crypto/simulation/paper_trader.py` | Paper trading with trade history, P&L tracking |

### Files Modified

| File | Changes |
|------|---------|
| `.env` | Added COINBASE_API_KEY, COINBASE_API_SECRET |
| `pyproject.toml` | Added coinbase-advanced-py>=1.8.2, removed alpaca-trade-api (conflict) |

### Key Features

1. **CoinbaseClient** (`crypto/exchange/coinbase_client.py`)
   - Historical OHLCV data with resampling (4h, 1w)
   - Public API fallback when auth fails (no auth needed for market data)
   - Simulation mode for paper trading (mock orders, positions)
   - Order creation (market, limit, stop)

2. **PaperTrader** (`crypto/simulation/paper_trader.py`)
   - Trade history with P&L calculation
   - FIFO matching for closed trades
   - Performance metrics (win rate, profit factor, expectancy)
   - JSON persistence for session continuity

3. **Position Sizing** (`crypto/trading/sizing.py`)
   - ATR-based sizing with leverage cap (default 8x)
   - Skip trade logic when leverage exceeds limit

4. **State Management** (`crypto/data/state.py`)
   - Multi-timeframe bar classifications
   - Continuity scoring (FTFC)
   - Veto checks (Weekly/Daily inside bars)

### Verified Working

```python
# Current prices fetching (via public API)
BTC-USD: $90,387
ETH-USD: $3,091
SOL-USD: $132

# OHLCV data for all timeframes
15m, 1h, 4h, 1d - all working

# Paper trading simulation
Trades open/close with P&L calculation working
```

### API Credentials - WORKING

New Coinbase API credentials generated and verified working:
- Authenticated API access confirmed
- 16 accounts found
- Perpetual futures products accessible

### Perpetual Futures Access - VERIFIED

| Product | Type | Status |
|---------|------|--------|
| `BTC-PERP-INTX` | Perpetual | Working ($90,177) |
| `ETH-PERP-INTX` | Perpetual | Working ($3,121) |

### Derivatives Infrastructure Added

| File | Purpose |
|------|---------|
| `crypto/config.py` | Updated with INTX symbols, leverage tiers, funding rates, margin |
| `crypto/trading/derivatives.py` | Funding cost calc, liquidation price, margin requirements |

**Leverage Tiers:**
- Intraday: 10x (close before 8h funding)
- Swing: 4x BTC/ETH, 3x SOL

**Funding:** 8h intervals (00:00, 08:00, 16:00 UTC), ~10% APR default

### Session 83K-83 Priorities

1. **Connect to Atlas STRAT** - Wire crypto data to `strat/` module for pattern detection
2. **Create Crypto Signal Scanner** - Similar to `strat/paper_signal_scanner.py`
3. **Test Paper Trading** - Validate simulation with real perp prices
4. **Dashboard Integration** - Add crypto section to Atlas dashboard

### Plan Mode Recommendation

**PLAN MODE: ON** - Next step is connecting crypto to Atlas STRAT engine for pattern detection.

---

## Session 83K-81: Dashboard P&L and Performance Tracking (COMPLETE)

**Date:** December 12, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Ready for Railway deployment

### What Was Implemented

#### Priority 1: Closed Position P&L Tracking (COMPLETE)

**Solution:** FIFO matching algorithm using Alpaca `/v2/account/activities/FILL` endpoint

**Files Modified:**
| File | Changes |
|------|---------|
| `integrations/alpaca_trading_client.py` | Added `get_fill_activities()`, `get_closed_trades()` with FIFO matching |
| `dashboard/data_loaders/options_loader.py` | Added `get_closed_trades()`, `get_closed_trades_summary()` |
| `dashboard/components/options_panel.py` | Added 4th tab "Closed Trades", `create_closed_trades_table()` |
| `dashboard/app.py` | Updated `update_options_signals()` callback for closed trades |

**Features:**
- 4th tab "Closed Trades" in Options panel
- FIFO matching for realized P&L calculation
- Summary row: Total P&L, Win Rate, W/L count
- Table columns: Contract, Qty, Entry, Exit, Realized P&L, Duration, Closed Date
- 30-day default lookback

#### Priority 2: Strategy Performance Tab Restructure (COMPLETE)

**Files Modified:**
| File | Changes |
|------|---------|
| `dashboard/config.py` | Added 'strat_options' and 'aggregate' to AVAILABLE_STRATEGIES |
| `dashboard/components/strategy_panel.py` | Changed default to 'strat_options' |
| `dashboard/app.py` | Updated 3 callbacks to handle STRAT Options strategy |

**Strategy Dropdown Options:**
- **STRAT Options (Live)** - Default, shows closed trades performance
- **Aggregate (All Strategies)** - Combined view
- **Opening Range Breakout** - Existing backtest
- **52-Week High Momentum** - Existing backtest

**STRAT Options Displays:**
- Equity Curve: Total P&L, Win Rate, Trade counts, Avg P&L
- Rolling Metrics: Bar chart of last 10 closed trades P&L
- Trade Distribution: Pie chart of wins vs losses

### VPS Deployment

```bash
ssh atlas@178.156.223.251
cd ~/vectorbt-workspace && git pull
# Dashboard auto-deploys via Railway
```

### Session 83K-82 Priorities

1. **Monitor Dashboard** - Verify closed trades display correctly on Railway
2. **Test FIFO Matching** - Verify with actual closed trades in paper account
3. **P3: Trade Progress to Target** (DEFER) - Signal-to-position linkage

### Plan Mode Recommendation

**PLAN MODE: OFF** - Features complete, monitoring phase.

---

## Session 83K-80: HTF Scanning Architecture Fix (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Deployed to VPS

### Solution: 15-Minute Base Resampling

For SETUP patterns (2-1-?, 3-1-?), entry is LIVE when price breaks inside bar. Previous fixed schedules missed entries. Now uses 15-min bars as base and resamples to all higher timeframes every 15 minutes.

### Files Modified

| File | Changes |
|------|---------|
| `strat/paper_signal_scanner.py` | Added resampling methods |
| `strat/signal_automation/config.py` | Added `enable_htf_resampling` |
| `strat/signal_automation/scheduler.py` | Added `add_base_scan_job()` |
| `strat/signal_automation/daemon.py` | Added `run_base_scan()` |

### Commits

```
04d8933 feat: implement 15-min base resampling for HTF scanning fix
```

---

## Session 83K-79: Comprehensive Project Audit (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Documentation fixed, unused code removed

- Fixed test count (913) and Phase 5 status (deployed)
- Deleted empty stub modules
- Archived 24 exploratory scripts

---

## Session 83K-78: Dashboard Enhancement + Watchlist Expansion (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Dashboard redesigned, watchlist expanded to 11 symbols

---

## Session 83K-77: Critical Bug Fix - Rapid Entry/Exit Safeguards (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Four safeguards implemented

- Minimum hold time (5 minutes)
- HISTORICAL_TRIGGERED check
- Thread-safe lock
- Market hours check

---

**ARCHIVED SESSIONS:**
- Sessions 1-66: `archives/sessions/HANDOFF_SESSIONS_01-66.md`
- Sessions 83K-2 to 83K-10: `archives/sessions/HANDOFF_SESSIONS_83K-2_to_83K-10.md`
- Sessions 83K-10 to 83K-19: `archives/sessions/HANDOFF_SESSIONS_83K-10_to_83K-19.md`
- Sessions 83K-20 to 83K-39: `archives/sessions/HANDOFF_SESSIONS_83K-20_to_83K-39.md`
- Sessions 83K-40 to 83K-46: `archives/sessions/HANDOFF_SESSIONS_83K-40_to_83K-46.md`
- Sessions 83K-52 to 83K-66: `archives/sessions/HANDOFF_SESSIONS_83K-52_to_83K-66.md`

---
