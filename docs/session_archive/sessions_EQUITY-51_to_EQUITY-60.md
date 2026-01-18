# Archived Sessions: EQUITY-51 to EQUITY-60

**Archived:** January 18, 2026
**Reason:** HANDOFF.md exceeded 1500 lines

---

## Session EQUITY-60: Dashboard Phase 4 - UI Polish (COMPLETE)

**Date:** January 13, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Dashboard Phase 4 committed
**Commit:** 69c030a

### What Was Accomplished

**1. Risk/Heat Threshold Consistency**

Fixed threshold alignment between config and UI:
- Heat gauge now shows 0-25% range (was 0-15%)
- Green zone: <8% (portfolio heat limit)
- Yellow zone: 8-15% (elevated)
- Red zone: >15% (danger)
- Config `position_size_limit`: 12% (keeps max at yellow, not red)

**2. Table Pagination and Sticky Headers**

Added to all tables in strat_analytics_panel.py:
- Sticky headers (position: sticky, top: 0)
- Max height with overflow scroll (~15 rows visible)
- "Showing X of Y" row count indicator
- Tables: trades, patterns, positions, pending signals

**3. Strategy Performance Labels**

- Dynamic equity curve title: shows selected strategy name instead of generic "Paper Trading Portfolio"
- Example: "52-Week High Momentum - Live" instead of "Paper Trading Portfolio"

**4. Header/Footer Cleanup**

- Header: Uses FONTS config instead of hardcoded font families
- Footer: Removed placeholder `#` links, added version and status info

### Files Modified

| File | Changes |
|------|---------|
| `dashboard/app.py` | +dynamic strategy title, +footer cleanup, +heat gauge thresholds |
| `dashboard/config.py` | +position_size_limit 0.12 (was 0.05) |
| `dashboard/components/header.py` | +FONTS import, consistent font families |
| `dashboard/components/risk_panel.py` | +aligned alert text with config |
| `dashboard/components/strat_analytics_panel.py` | +sticky headers, +table scroll, +row counts |

### Code Review

Ran feature-dev:code-reviewer, fixed 2 issues:
1. Position size limit now 12% to stay below 15% danger threshold
2. Header uses FONTS from config instead of hardcoded values

---

## Session EQUITY-59: Crypto Leverage-First Sizing + Stale Setup (COMPLETE)

**Date:** January 13, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Crypto sizing and validation ported

### What Was Accomplished

**1. Leverage-First Sizing (EQUITY-59)**

Added new sizing mode for crypto paper trading that uses full available leverage:
- **Function:** `calculate_position_size_leverage_first()` in `crypto/trading/sizing.py`
- **Config Flag:** `LEVERAGE_FIRST_SIZING = True` in `crypto/config.py`
- **Behavior:** Always uses max leverage (10x intraday, 4x swing) instead of risk-based sizing
- **Purpose:** Full capital deployment for data collection during paper trading

**2. Weekend Leverage Handling (EQUITY-59)**

Added Coinbase derivatives weekend leverage rules:
- **Function:** `is_weekend_leverage_window()` in `crypto/config.py`
- **Window:** Friday 4PM ET to Sunday 6PM ET = swing leverage only (4x)
- **Pattern:** Similar to equity futures weekend handling

**3. Stale Setup Validation (EQUITY-46 Port)**

Ported equity stale setup validation to crypto with 24/7 market adaptations:
- **Method:** `_is_setup_stale()` in `crypto/scanning/daemon.py`
- **Windows:** 1H=2h, 4H=8h, 1D=48h, 1W=2 weeks
- **Check Location:** `_on_trigger()` before trade execution

**4. Code Review Fixes**

Addressed issues found during code review:
- Added timezone conversion (`astimezone(UTC)`) for non-UTC timestamps
- Added input validation (zero stop distance, invalid prices) before leverage-first check
- Documented 1H staleness window rationale (2x bar width for late triggers)

### Files Modified

| File | Changes |
|------|---------|
| `crypto/config.py` | +LEVERAGE_FIRST_SIZING flag, +is_weekend_leverage_window() |
| `crypto/scanning/daemon.py` | +_is_setup_stale(), +leverage-first sizing integration, +input validation |
| `crypto/trading/sizing.py` | +calculate_position_size_leverage_first() |

### Tests

- 413 tests passed (strat + signal_automation)
- Stale setup validation tested with fresh/stale signals
- Weekend leverage window tested with various times

---

## Session EQUITY-58: 1H EOD Exit Bug Fix (COMPLETE)

**Date:** January 12, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Fix deployed to VPS

### Bug Fixed

**1H Positions Not Exiting at 15:59 ET**

- **Root Cause:** Line 664 used exact string match `pos.timeframe == '1H'`
- **Problem:** Fails for timeframe variants like '60MIN', '60M', or lowercase '1h'
- **Fix:** Changed to `pos.timeframe and pos.timeframe.upper() in ['1H', '60MIN', '60M']`
- **Location:** `strat/signal_automation/position_monitor.py:664`
- **Pattern Match:** Now consistent with line 484 (target adjustment) which handles same variants

### Investigation Findings

**1. EOD Exit Timing Confirmed:**
- Live trading exits at 15:59 ET (intentional per EQUITY-45)
- Backtesting uses 15:30 (potential future alignment needed)
- 15:59 was changed from 15:55 to "capture late momentum moves"

**2. Crypto Pipeline Gaps:**
- Missing stale setup validation (EQUITY-46)
- Missing Type 3 pattern invalidation (EQUITY-48)
- Missing TFC re-evaluation at entry (EQUITY-49)
- Crypto bypasses unified_pattern_detector.py

**3. TFC Scanner Inconsistency:**
- paper_signal_scanner.py uses 4 timeframes (missing 4H)
- Base classes define 5 timeframes including 4H
- Impact: TFC never evaluates 4H alignment

### Tests

- 65 signal automation tests passed (no regressions)

### Commit

- `cf5630d`: fix(position-monitor): handle timeframe variants in 1H EOD exit check (EQUITY-58)

---

## Session EQUITY-57: Critical Bug Fixes + Trade Audit (COMPLETE)

**Date:** January 12, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - All fixes deployed to VPS

### Critical Bugs Fixed

1. **Timezone Bug in "Let the Market Breathe" Filter (CRITICAL)**
   - Bug: `datetime.now().time()` returned UTC on VPS, not ET
   - Impact: 1H patterns entered at 9:35 AM instead of waiting for 10:30+
   - Fix: Use `datetime.now(pytz.timezone('America/New_York')).time()`
   - File: `strat/signal_automation/daemon.py`

2. **Resampled SETUP Signals Never Detected (CRITICAL)**
   - Bug: Line 1518 checked `index == len-1` but `_detect_setups` excludes last bar
   - Impact: Zero Daily/Weekly/Monthly SETUP signals were being monitored
   - Fix: Changed to `index >= len-4` matching non-resampled logic
   - File: `strat/paper_signal_scanner.py`

3. **Equity Curve Error**
   - Bug: `self.client.trading_client` should be `self.client.client`
   - Fix: Corrected attribute access path
   - File: `dashboard/data_loaders/live_loader.py`

4. **Docker Build Missing enriched_trades.json**
   - Bug: `.dockerignore` excluded `data/*.json`
   - Fix: Added exception `!data/enriched_trades.json`
   - Impact: Railway dashboard will now show pattern/TFC data

### Improvements

1. **Hybrid Entry Priority**
   - Changed: Sort by `(priority_rank, priority)` instead of just `priority`
   - Result: High-TFC signals execute before low-TFC when multiple trigger
   - File: `strat/signal_automation/entry_monitor.py`

2. **Enhanced /diagnostic Endpoint**
   - Added: enriched_records count, sample OSI symbols, trade pattern data
   - Purpose: Debug Railway pattern loading issues

3. **README.md Updated to v4.0**
   - Updated test count (1069 tests)
   - Added Phase 5 active components
   - Fixed STRAT patterns section (removed unimplemented Rev Strat)
   - Added TFC documentation

### Trade Audit Findings (Jan 12)

| Trade | Pattern | TFC | Result | Issue |
|-------|---------|-----|--------|-------|
| DIA PUT | 2U-2D (1D) | 1/4 | -$201 | Should have been filtered (low TFC) |
| ACHR CALL | 3-2U (1H) | 2/4 | +$7 | Entered at 9:35 (before 10:30 allowed) |

### STRAT Terminology Clarified

- **Rev Strat**: NOT implemented - 1-bar (X-1-3) and 2-bar (X-1-2U-2D) patterns
- **3-2-2 Continuation** (3-2U-2U): NOT traded
- **3-2-2 Reversal** (3-2U-2D): TRADED - target is Type 3 bar extreme

### Commits

- `24ed8f8`: Enhanced /diagnostic + fixed equity curve
- `752fc68`: Timezone fix for let-the-market-breathe filter
- `77cc5c4`: Resampled SETUP fix + hybrid priority + dockerignore
- `37733b7`: README.md update to v4.0

### Tests

- 1069 tests collected, 413 signal automation tests passing
- All fixes verified before deployment

---

## Session EQUITY-56: Dashboard Field Normalization + TFC Fix (COMPLETE)

**Date:** January 12, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Deployed to VPS and Railway

### Root Cause Analysis

1. **Field Name Mismatch (Bug 1):**
   - Alpaca returns: `realized_pnl`, `roi_percent`, `buy_price`, `sell_price`
   - Panel expects: `pnl`, `pnl_pct`, `entry_price`, `exit_price`
   - Result: Dashboard showed $0.00 for all P&L values

2. **TFC Calculation Bug (Bug 2):**
   - Historical data truncated to entry_time doesn't capture intrabar state
   - Detection timeframe bar may show as Type 1 when it was 2U/2D at entry
   - Per STRAT methodology: Entry = break = bar became 2U/2D = aligned
   - Fix: Count detection timeframe as aligned BY DEFINITION at entry

### Implementation

1. **`dashboard/data_loaders/options_loader.py`** (+8 lines)
   - Added field name normalization in `get_closed_trades()` loop
   - Maps Alpaca field names to panel-expected field names

2. **`scripts/backfill_trade_tfc.py`** (+14 lines)
   - Added detection timeframe alignment logic
   - If detection TF not in aligned list, add it (entry = aligned)
   - Recalculate TFC score and passes_flexible

3. **`data/enriched_trades.json`** (regenerated)
   - 38 trades processed (4 more recent trades found)
   - NO trades with TFC 0 (minimum is now 1/4)
   - TFC distribution: 1/4 (10), 2/4 (13), 3/4 (15), 4/4 (0)

### STRAT Methodology Clarification (User Input)

TFC at entry:
1. Higher timeframes (1M, 1W, 1D) checked for alignment
2. Detection timeframe bar starts as Type 1 (inside, waiting for break)
3. Entry trigger fires when detection TF bar becomes 2U/2D
4. **At entry moment:** Detection TF is aligned BY DEFINITION
5. TFC score = count of aligned timeframes (1-4)

**Key Insight:** Entry itself makes the detection timeframe bar become 2U/2D. They happen simultaneously.

### Tests

- 83 tests passed (dashboard + signal automation)
- Code review passed (95% and 88% confidence)

### Commits

| Hash | Description |
|------|-------------|
| f4c2449 | fix(dashboard): normalize field names and fix TFC calculation (EQUITY-56) |

### Files Modified

| File | Change |
|------|--------|
| `dashboard/data_loaders/options_loader.py` | Added field name normalization |
| `scripts/backfill_trade_tfc.py` | Added detection TF alignment fix |
| `data/enriched_trades.json` | Regenerated with corrected TFC scores |

---

## Session EQUITY-55: Retroactive TFC Backfill (COMPLETE)

**Date:** January 11, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Deployed to VPS

### Implementation

1. **Created `scripts/backfill_trade_tfc.py`** (529 lines)
   - Parses OSI symbols to extract underlying and direction
   - Fetches historical OHLC data at entry time
   - Uses `TimeframeContinuityChecker.check_flexible_continuity_at_datetime()` for historical TFC
   - Outputs to `data/enriched_trades.json` with summary statistics

2. **Modified `dashboard/data_loaders/options_loader.py`**
   - Added `_load_enriched_tfc_data()` method for O(1) OSI symbol lookup
   - Modified `get_closed_trades()` to merge enriched TFC data into live Alpaca data
   - Uses absolute paths for data files (works from any working directory)

3. **VPS Deployment**
   - Git pull: 4 files changed, 697 insertions
   - Daemon restarted
   - Backfill script executed: 34 trades processed, all 34 have pattern data

### Results

34 trades processed with retroactive TFC:
- **All 34 trades had TFC < 4** (no high-TFC trades in history)
- **All 34 trades have pattern data** (VPS signal store has OSI mappings)
- Overall win rate: 29.4%
- Total P&L: -$2,125

**Key Insight:** The trading account was taking trades without TFC alignment, which may explain the poor win rate. New signals (after EQUITY-54 fix) will have TFC calculated at detection time, allowing filtering for TFC >= 4.

### Tests

- 431 tests passed, 2 skipped (no regressions)
- Dashboard integration verified locally and on VPS

### Commits

| Hash | Description |
|------|-------------|
| 4ab1e8d | feat(dashboard): add retroactive TFC backfill script (EQUITY-55) |

### Files Modified

| File | Change |
|------|--------|
| `scripts/backfill_trade_tfc.py` | NEW - Backfill script |
| `dashboard/data_loaders/options_loader.py` | Added TFC merge logic |
| `data/enriched_trades.json` | NEW - Generated output (VPS) |

---

## Session EQUITY-54: Dashboard Bug Fixes (COMPLETE)

**Date:** January 11, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Bug fixes deployed to VPS

### Issues Fixed

1. **CryptoDataLoader.get_closed_trades() Parameter Mismatch**
   - Problem: Called with `days=30` but CryptoDataLoader expects `limit=50`
   - Fix: Check market type before calling method with correct parameter
   - File: `dashboard/app.py` line 1896-1900

2. **TFC Score Always 0 in Signals**
   - Problem: `scan_symbol_all_timeframes_resampled()` didn't call `evaluate_tfc()`
   - Fix: Added TFC evaluation for both COMPLETED and SETUP patterns
   - File: `strat/paper_signal_scanner.py` lines 1470-1486, 1523-1539

### Root Cause Analysis

The `scan_symbol_all_timeframes_resampled()` method (used by daemon's 15-min base scan) was using basic `_get_market_context()` which returns a `SignalContext` without TFC data. The working `scan_single_symbol()` method correctly called `evaluate_tfc()` for each pattern.

### Tests

- 413 tests passed (no regressions)
- Dashboard imports verified

### Commit

- `2dcb801`: fix(dashboard): crypto loader parameter and TFC calculation (EQUITY-54)

### Note

**Existing signals still have `tfc_score=0`** because they were detected before this fix. New signals detected after deployment will have correct TFC scores. TFC for existing trades shows as 0/4 on dashboard because the underlying signals lack TFC data.

---

## Session EQUITY-53: Testing and VPS Deployment (COMPLETE)

**Date:** January 10, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - EQUITY-52-A/B deployed, daemon restarted

### Overview

Testing and deployment session for EQUITY-52-A (Dashboard) and EQUITY-52-B (ATR targets/trailing) changes.

### Verified

1. **Tests Passed**
   - 413 tests passed (348 STRAT + 65 signal automation)
   - 2 skipped, 7 warnings (expected)

2. **Dashboard Local Testing**
   - Dashboard starts without errors via `python -m dashboard.app`
   - Alpaca LARGE connected ($10,330.95 equity)
   - Alpaca SMALL connected ($739.85 equity)
   - SignalStore loaded 170 signals (local), 352 signals (VPS)
   - CryptoDataLoader connected to VPS API

3. **VPS Deployment**
   - Git pull successful (11 files, +2150 lines)
   - `atlas-daemon` restarted and running
   - Daily audit scheduled for 4:30 PM ET
   - 352 signals loaded, 351 executions loaded

### Deployment Commands Used

```bash
# On VPS
ssh atlas@178.156.223.251 "cd /home/atlas/vectorbt-workspace && git pull origin main"
ssh atlas@178.156.223.251 "sudo systemctl restart atlas-daemon"
```

### Files Deployed (EQUITY-52-A/B Combined)

| File | Changes |
|------|---------|
| `dashboard/components/strat_analytics_panel.py` | NEW - 1005 lines |
| `dashboard/app.py` | +213 lines |
| `dashboard/data_loaders/live_loader.py` | +63 lines |
| `strat/unified_pattern_detector.py` | +139 lines (ATR targets) |
| `strat/signal_automation/position_monitor.py` | +181 lines (ATR trailing) |
| `strat/signal_automation/daemon.py` | +136 lines (daily audit) |
| `strat/signal_automation/alerters/discord_alerter.py` | +106 lines |
| `strat/signal_automation/signal_store.py` | +34 lines (OSI index) |

### Pending Verification

- Daily audit runs at 4:30 PM ET (scheduled, not yet executed)
- ATR calculations on new 3-2 patterns (no new patterns since deployment)
- Visual browser test of 6-tab dashboard structure

---

## Session EQUITY-52-A: Unified STRAT Analytics Dashboard (COMPLETE)

**Date:** January 10, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Workstream:** Terminal A (Dashboard Overhaul)
**Status:** COMPLETE - Core implementation done

### Overview

Implemented unified STRAT Analytics dashboard with market toggle and 6-tab structure.

### Key Changes

1. **Signal Lookup Fix (Root Cause)**
   - Added OSI symbol reverse index to SignalStore for O(1) lookup
   - Fixed pattern overwrite bug (fallback only when signal store lookup fails)
   - File: `strat/signal_automation/signal_store.py` (+34 lines)

2. **Unified STRAT Analytics Panel**
   - New file: `dashboard/components/strat_analytics_panel.py` (1005 lines)
   - Market dropdown: Equity Options / Crypto toggle
   - 6 sub-tabs: Overview, Patterns, TFC Comparison, Closed Trades, Pending, Equity Curve
   - TFC threshold: >= 4 for "WITH TFC" comparison
   - No emojis (CLAUDE.md compliance)

3. **Portfolio History API**
   - Added `get_portfolio_history()` to LiveDataLoader
   - Uses Alpaca `/v2/account/portfolio/history` endpoint
   - File: `dashboard/data_loaders/live_loader.py` (+63 lines)

4. **App Integration**
   - Replaced Options Trading + Crypto Trading tabs with single STRAT Analytics tab
   - Added callbacks for market toggle and 6 sub-tabs
   - File: `dashboard/app.py` (+138 lines)

### Files Modified

| File | Changes |
|------|---------|
| `strat/signal_automation/signal_store.py` | OSI reverse index, O(1) lookup |
| `dashboard/data_loaders/options_loader.py` | Fix pattern overwrite bug |
| `dashboard/data_loaders/live_loader.py` | Portfolio history method |
| `dashboard/components/strat_analytics_panel.py` | NEW - Unified panel |
| `dashboard/app.py` | Tab integration and callbacks |

### User Decisions

- TFC threshold >= 4 for "WITH TFC"
- No emojis (follow CLAUDE.md)
- Replace both Options and Crypto tabs with unified panel
- Market dropdown instead of separate tabs

### Tests

- 65 signal automation tests passed
- Code review: No issues >= 80 confidence

### Commit

- `b31b8ff`: feat(dashboard): add unified STRAT Analytics panel with market toggle (EQUITY-52)

---

## Session EQUITY-52-B: ATR Targets + Trailing Stops + Daily Audit (COMPLETE)

**Date:** January 10, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Workstream:** Terminal B (ATR + Audit)
**Status:** COMPLETE - All three parts implemented and tested

### Overview

Implemented three features for 3-2 patterns:
1. **ATR-based targets** - Replace fixed 1.5% with `ATR * 1.5` for dynamic scaling
2. **ATR-based trailing stops** - Activate at 0.75 ATR, trail by 1.0 ATR
3. **Automated daily audit** - Webhook-based report at 4:30 PM ET

### Part 1: ATR-Based Targets

**Problem:** Fixed percentage targets don't scale across price ranges (COIN $300 vs ACHR $8).

**Solution:**
- Added `calculate_atr_target()` function to `unified_pattern_detector.py`
- Added `_calculate_atr_from_df()` helper for ATR calculation
- 3-2 patterns now use: `target = entry +/- (ATR * 1.5)`
- 3-2 patterns bypass 1H 1.0x override (user decision)
- ATR stored in pattern dict as `atr_at_detection` for audit trail

**Files Modified:**
- `strat/unified_pattern_detector.py` (+92 lines)

### Part 2: ATR-Based Trailing Stops

**3-2 Pattern Trailing:**
- Activation: 0.75 ATR profit
- Trail distance: 1.0 ATR from high water mark

**Non-3-2 Patterns:** Keep existing percentage-based (0.5x R:R activation, 50% trail)

**Configuration Added:**
```python
use_atr_trailing_for_32: bool = True
atr_trailing_activation_multiple: float = 0.75
atr_trailing_distance_multiple: float = 1.0
```

**Files Modified:**
- `strat/signal_automation/position_monitor.py` (+181 lines)
  - New config options (lines 92-97)
  - New TrackedPosition fields (lines 174-178)
  - ATR initialization in `_create_tracked_position()`
  - New `_check_atr_trailing_stop()` method
  - Router in `_check_trailing_stop()` to select method

### Part 3: Automated Daily Audit

**Scheduled:** 4:30 PM ET daily via APScheduler

**Report Contents:**
- Trades today, wins, losses, total P&L
- Open positions summary
- Anomaly detection (expandable)

**Files Modified:**
- `strat/signal_automation/alerters/discord_alerter.py` (+106 lines)
  - New `send_daily_audit()` method with rich embed
- `strat/signal_automation/daemon.py` (+136 lines)
  - New `_generate_daily_audit()` method
  - New `_run_daily_audit()` method
  - Scheduler job at 4:30 PM ET

### Tests

| Suite | Result |
|-------|--------|
| STRAT tests | 348 passed, 2 skipped |
| Signal automation tests | 65 passed |

### User Decisions

| Decision | Choice |
|----------|--------|
| Discord implementation | Webhook first (can upgrade to bot later) |
| Trailing stop scope | 3-2 patterns only |
| 1H timeframe override | ATR for ALL timeframes (override EQUITY-36 for 3-2) |

---

## Session EQUITY-51: Analytics Tests + VPS Deployment + Pipeline Analysis (COMPLETE)

**Date:** January 8-9, 2026
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Tests added, VPS deployed, pipeline gaps documented

### Overview

Session covered three areas:
1. Added unit tests for trade analytics (18 tests)
2. Deployed EQUITY-47 through EQUITY-51 changes to VPS
3. Comprehensive pipeline gap analysis between equity and crypto

### Part 1: Trade Analytics Tests

Added 18 unit tests for `calculate_trade_analytics()` function:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestCalculateTradeAnalyticsEmpty | 2 | Empty/None input handling |
| TestCalculateTradeAnalyticsSingleTrade | 3 | Win/loss/breakeven single trades |
| TestCalculateTradeAnalyticsMultipleTrades | 4 | Aggregation, sorting, breakdowns |
| TestCalculateTradeAnalyticsMissingFields | 6 | None/missing field handling |
| TestCalculateTradeAnalyticsCalculationAccuracy | 3 | Win rate, avg P&L, mixed data |

**Commit:** `a59c384` - test(dashboard): add unit tests for trade analytics calculate function

### Part 2: VPS Deployment

Deployed all pending changes to VPS:
- EQUITY-47: TFC logging + filter rejection logging
- EQUITY-48: Type 3 evolution detection + signal lifecycle tracing
- EQUITY-49: TFC re-evaluation at entry trigger
- EQUITY-50: Trade analytics dashboard
- EQUITY-51: Stale 1H position fix + analytics tests

### Part 3: Pipeline Gap Analysis (Equity vs Crypto)

Launched parallel explore agents to map both pipelines. Key findings:

**Bug Fixes in Equity NOT in Crypto:**

| Session | Fix | Equity Location | Crypto Status |
|---------|-----|-----------------|---------------|
| EQUITY-46 | Stale setup validation | daemon.py:786-877 | MISSING |
| EQUITY-48 | Type 3 invalidation (intrabar) | position_monitor.py:1030-1056 | MISSING |
| EQUITY-49 | TFC re-evaluation at entry | daemon.py:933-1056 | MISSING |
| EQUITY-51 | Stale 1H position detection | position_monitor.py:1132-1180 | MISSING |

**Recommended Unification Strategy:**

1. Create shared execution validation module
2. Port stale/TFC/Type3 logic to crypto
3. Ensure unified_pattern_detector used by both

### Part 1b: Stale 1H Position Fix (Other Terminal)

Fixed critical bug where 1H trades entered on a previous trading day were not being exited until today's 15:59, instead of being exited immediately. The NFLX 3-2D 1H Put trade from Jan 7 at 10:48 was held overnight and only exited Jan 8 at 15:59.

### The Bug

**Root Cause:** EOD exit logic used `now.replace(hour=15, minute=59)` which always creates TODAY's 15:59, never checking if the position was entered on a previous trading day.

```python
# BROKEN - Always uses today's 15:59
eod_exit_time = now_et.replace(hour=15, minute=59, ...)
if now_et >= eod_exit_time:  # Only true after today's 15:59
    trigger_exit()
```

### The Fix

1. **Added `_is_stale_1h_position()` method** - Uses pandas_market_calendars to detect if entry was on a previous trading day
2. **Modified EOD exit check** - Stale 1H positions exit IMMEDIATELY, not at today's 15:59
3. **Fixed entry_time extraction** - Uses `execution.timestamp` instead of `datetime.now()` when syncing positions (critical for daemon restarts)

| Fix | Location | Change |
|-----|----------|--------|
| Stale detection | position_monitor.py:1132-1180 | New `_is_stale_1h_position()` method |
| Immediate exit | position_monitor.py:596-612 | Check stale before normal EOD |
| Entry timestamp | position_monitor.py:475-481 | Use execution.timestamp |

### Audit Findings

Audited all stale-related files:
- **position_monitor.py** - Fixed (2 bugs: stale check + entry_time)
- **entry_monitor.py** - OK (relies on daemon for stale checks)
- **daemon.py** - OK (has `_is_setup_stale()` from EQUITY-46)
- **signal_store.py** - OK (proper timestamp handling)
- **executor.py** - OK (proper timestamp persistence)

### Test Results

```
tests/test_signal_automation/ - 65 passed (7 new stale position tests)
tests/test_strat/             - 348 passed, 2 skipped
```

### Commit

`10e381f` - fix(position-monitor): detect and exit stale 1H positions from previous trading day (EQUITY-51)

### Files Modified

| File | Change |
|------|--------|
| `strat/signal_automation/position_monitor.py` | +_is_stale_1h_position(), +stale exit check, +entry_time fix |
| `tests/test_signal_automation/test_stale_1h_position.py` | NEW - 7 tests for stale position detection |

### Plan File

`C:\Users\sheeh\.claude\plans\quirky-tumbling-rabin.md`
