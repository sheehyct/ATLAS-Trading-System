# Archived Sessions: EQUITY-18 to CRYPTO-1

**Archived:** December 20, 2025 (Session EQUITY-29)
**Sessions:** EQUITY-18, EQUITY-17, CRYPTO-16 through CRYPTO-1, 83K-82 through 83K-77

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

### Commits

- `e631970` - feat(strat): add 15m/30m timeframe scanning support

---

## Session EQUITY-17: Discord Alerts Fix + Premarket Analysis (COMPLETE)

**Date:** December 18, 2025
**Status:** COMPLETE - Discord working, premarket test script ready

### Issues Fixed

1. **Discord Webhook Routing** - Equity alerts now go to trade-alerts channel
2. **Merged Claude Web Bug Fixes** - Time filtering, Discord alerts on all paths
3. **SSH Key Configuration** - Removed passphrase for easier VPS access

### Commits

- `6d4d420` - Merge branch 'claude/review-daemon-strategy-T4kTM'
- `25d5142` - fix(discord): use equity-specific webhook URL
- `f0e7ff4` - feat(scripts): add live premarket 15m data fetching

---

## Session CRYPTO-16: CFM Products + X-2 SETUP Detection (COMPLETE)

**Date:** December 18, 2025
**Status:** COMPLETE - Both fixes deployed to VPS

### Key Finding

BIP-20DEC30-CDE is the correct product ID for nano Bitcoin futures (not BTC-PERP).

### Config Changes

- `CRYPTO_SYMBOLS` changed to `["BIP-20DEC30-CDE", "ETP-20DEC30-CDE"]`
- Added `SYMBOL_TO_BASE_ASSET` mapping
- Added `FUTURES_EXPIRY` with expiration dates

### Commits

- `f080419` - feat(crypto): switch to CFM venue products
- `79bffa9` - feat(crypto): add 3-2 and 2-2 SETUP detection

---

## Session CRYPTO-15: Critical SETUP Detection Bug Fix (COMPLETE)

**Date:** December 17, 2025
**Status:** COMPLETE - Fix deployed to VPS

### Problem

AAPL formed a 3-2D-2U pattern but no alerts generated because scanner only detected patterns ending in inside bars.

### Fix

Added detection for directional bar setups waiting for opposite direction breaks.

### Commits

- `2226b8f` - fix(strat): add X-2D and X-2U opposite direction SETUP detection

---

## Session CRYPTO-14: Tradovate API Research (COMPLETE)

**Date:** December 17, 2025
**Status:** COMPLETE - Decision pending on Tradovate subscription

### Key Findings

- Coinbase API confirmed limited to INTX only
- Tradovate has BIP but requires $25/month subscription
- MFF account credentials do NOT grant Tradovate API access

---

## Session CRYPTO-13: CFM API Research (COMPLETE)

**Date:** December 16, 2025
**Status:** COMPLETE - Critical finding: INTX has non-uniform basis vs BIP

### Critical Discovery

INTX vs BIP price spreads are non-uniform:
- 3 Bar HIGH differs by $1,695
- 3 Bar LOW differs by only $8

This explains why trades were being filtered (compressed geometry on INTX).

---

## Session CRYPTO-12: Wrong Coinbase Product Discovery (COMPLETE)

**Date:** December 16, 2025
**Status:** COMPLETE - Led to CRYPTO-13 research

### Critical Discovery

Daemon was fetching BTC-PERP-INTX but user trades BIP (Nano Bitcoin).

---

## Session CRYPTO-11: Equity Daemon Bug Fixes (COMPLETE)

**Date:** December 16, 2025
**Status:** COMPLETE - Equity daemon now matches crypto daemon fixes

### Bugs Fixed

1. Live bar as setup bar - Added `last_bar_idx` skip
2. Unidirectional trigger checking - Now checks BOTH directions
3. Direction handling - Added `_actual_direction` logic

### Commits

- `adc61b4` - fix(strat): apply crypto daemon bug fixes to equity daemon

---

## Session CRYPTO-10: Review and Merge STRAT Bug Fixes (COMPLETE)

**Date:** December 16, 2025
**Status:** COMPLETE - Merged Claude Code for Web branch

### Bug 1: Live Bar as Setup Bar

Fix: Exclude last bar from setup detection.

### Bug 2: Incorrect Stop Placement

Fix: 2-1-2 stops now use first directional bar, not inside bar.

### Commits

- `1b8e295` - fix(strat): exclude live bars and fix stop placement
- `739f2df` - test(strat): update 2-1-2 pattern tests

---

## Session CRYPTO-9: Cron Fix + Dashboard Enhancement (COMPLETE)

**Date:** December 15, 2025
**Status:** COMPLETE - Cron fix deployed, dashboard enhanced

### Critical Bug

All cron expressions used `1-5` for day_of_week, but APScheduler interprets this as Tuesday-Saturday.

### Fix

Changed `1-5` to `mon-fri` in all cron expressions.

### Commits

- `553ee45` - fix(strat): correct cron day-of-week

---

## Session CRYPTO-8: STRAT Pattern Transition Detection Fix (COMPLETE)

**Date:** December 15, 2025
**Status:** COMPLETE - Critical STRAT entry logic fixed

### Root Cause

"Where is the next 2?" was not implemented correctly. SETUP patterns had predetermined direction.

### Fix

Entry monitor now watches BOTH inside bar high AND low.

### Commits

- `8296211` - fix(crypto): detect SETUP to 2-bar pattern transitions

---

## Session CRYPTO-7: Discord Trade Alerts + VPS Deployment (COMPLETE)

**Date:** December 15, 2025
**Status:** COMPLETE - Entry/exit Discord alerts, VPS REST API deployed

### Key Features

- Entry alerts with symbol, pattern, price, quantity, leverage
- Exit alerts with P&L dollar and percent
- VPS deployment on 178.156.223.251

---

## Session CRYPTO-6: Dashboard Integration via REST API (COMPLETE)

**Date:** December 14, 2025
**Status:** COMPLETE - REST API + Dashboard crypto panel

### Files Created

- `crypto/api/server.py` - Flask REST API server
- `dashboard/components/crypto_panel.py` - Dashboard panel (1012 lines)

---

## Session CRYPTO-5: VPS Deployment and Discord Alerts (COMPLETE)

**Date:** December 14, 2025
**Status:** COMPLETE - VPS deployment, 60s position monitoring

### Key Features

- VPS daemon running 24/7
- 60-second position monitoring
- Discord alerts with leverage tier and TFC score

---

## Session CRYPTO-4: Intraday Leverage and VPS Deployment (COMPLETE)

**Date:** December 13, 2025
**Status:** COMPLETE - Intraday leverage, position monitoring

### Key Features

- Time-based leverage tiers (10x intraday, 4x swing)
- CLI script for VPS daemon
- systemd service file

---

## Session CRYPTO-3: Entry Monitor and Daemon (COMPLETE)

**Date:** December 13, 2025
**Status:** COMPLETE - Full automation stack

### Key Features

- CryptoEntryMonitor - 60s trigger polling
- CryptoSignalDaemon - Orchestrates scanner and monitor
- Friday maintenance window handling

---

## Session CRYPTO-2: Crypto STRAT Signal Scanner (COMPLETE)

**Date:** December 13, 2025
**Status:** COMPLETE - Core scanner implemented

### Key Features

- CryptoSignalScanner for all STRAT patterns
- Multi-timeframe (1w, 1d, 4h, 1h, 15m)
- SETUP detection for X-1 patterns

---

## Session 83K-82: Crypto Derivatives Module Integration (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Core infrastructure ready

### Files Created

- `crypto/` module with exchange, data, trading, simulation submodules
- CoinbaseClient, PaperTrader, position sizing, state management

---

## Session 83K-81: Dashboard P&L and Performance Tracking (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Ready for Railway deployment

### Key Features

- FIFO matching for closed trades P&L
- Strategy Performance tab with STRAT Options option
- Closed Trades tab with summary row

---

## Session 83K-80: HTF Scanning Architecture Fix (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Deployed to VPS

### Solution

15-minute base resampling for all higher timeframes.

---

## Session 83K-79: Comprehensive Project Audit (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE

- Fixed test count, deleted unused code, archived scripts

---

## Session 83K-78: Dashboard Enhancement + Watchlist Expansion (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE

- Dashboard redesigned, watchlist expanded to 11 symbols

---

## Session 83K-77: Critical Bug Fix - Rapid Entry/Exit Safeguards (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE

- Minimum hold time, HISTORICAL_TRIGGERED check, thread-safe lock, market hours check
