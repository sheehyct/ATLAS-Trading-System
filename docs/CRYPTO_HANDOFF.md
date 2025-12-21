# CRYPTO HANDOFF - ATLAS Crypto Daemon Monitoring

**Last Updated:** December 21, 2025 (Session CRYPTO-MONITOR-2)
**Purpose:** Live monitoring of crypto STRAT daemon for pattern validation
**Related:** See `docs/HANDOFF.md` for equity options work

---

## Session CRYPTO-MONITOR-2: COMPLETE

**Date:** December 21, 2025
**Environment:** Claude Code Desktop
**Status:** CRITICAL BUGS FIXED AND DEPLOYED

### Objective

Trade audit of discord alerts received today - cross-reference with actual price data to identify pattern detection bugs.

### Critical Bugs Fixed

#### BUG 1: Stale 3-? Setups Never Invalidated (CRITICAL)

**Severity:** CRITICAL - Trades executing 7+ days late at wrong prices

**Evidence from Discord Alerts:**
```
4:34 AM:  ENTRY 3-2U (1d) LONG @ $3,000    <- 7 days late!
10:55 AM: ENTRY 3-2D (1w) LONG @ $2,975    <- Pattern-direction mismatch!
```

**Root Cause:**
In setup validation (signal_scanner.py), 3-? setups used `pass`:
```python
elif setup_pattern == "3-2":
    pass  # Always valid - let entry monitor handle  <- BUG!
```

This meant outside bar setups from WEEKS ago remained active, triggering trades at wrong times and prices.

**Fix Applied:**
```python
elif setup_pattern == "3-2":
    # MUST invalidate when range is broken - pattern COMPLETED
    if bar_high > setup_high or bar_low < setup_low:
        setup_still_valid = False
        break
```

**Files Modified:**
- `crypto/scanning/signal_scanner.py` (lines 1070-1078)
- `strat/paper_signal_scanner.py` (lines 1274-1282)

#### BUG 2: One-Position-Per-Symbol Limit

**Severity:** MEDIUM - Prevented analysis of trade accuracy

**Root Cause:**
Daemon had check that skipped ALL trades if any position existed for that symbol:
```python
existing_position = self.paper_trader.get_open_position(signal.symbol)
if existing_position:
    return  # Skip trade
```

**Fix Applied:**
Removed the check entirely to allow multiple positions per symbol for better data collection.

**File Modified:**
- `crypto/scanning/daemon.py` (lines 375-383)

### Verification

**Before Fix (Weekly setups):**
- 5 ACTIVE SETUPS including from Nov 2 (7 weeks old!) and Dec 7 (pattern already completed)

**After Fix (Weekly setups):**
- 1 ACTIVE SETUP from Dec 21 (today - valid current setup)

### Trade Audit Summary

| Alert Time | Pattern | Entry | Issue |
|------------|---------|-------|-------|
| 4:34 AM | 3-2U (1d) LONG | $3,000 | 7 days late, wrong price |
| 8:32 AM | EXIT [STOP] | - | Lost $28.18 (-1.72%) |
| 10:55 AM | 3-2D (1w) LONG | $2,975 | Pattern=bearish, Direction=bullish (impossible) |
| 11:24 AM | 3-2U (4h) LONG | $2,983 | Also late |

**Root Cause:** All due to stale 3-? setups triggering days/weeks after pattern actually completed.

### Commits

- `04fbebb` - fix(strat): invalidate stale 3-? setups and remove position limit (CRYPTO-MONITOR-2)

---

## Session CRYPTO-MONITOR-1: COMPLETE

**Date:** December 21, 2025
**Environment:** Claude Code Desktop
**Status:** BUGS IDENTIFIED - ROOT CAUSE FOUND

### Objective

Live monitoring of crypto daemon to:
1. Validate pattern classifications against actual bar data
2. Verify entry correctness when trades trigger
3. Flag missed opportunities (pattern detected -> no trade)
4. Check for bugs found in equity daemon

### Critical Findings

#### BUG 1: Signal Expiration Uses Bar Timestamp (ROOT CAUSE FOUND)

**Severity:** CRITICAL - All weekly/daily signals expire immediately

**Evidence:**
```
15:03:06 | Added 64 SETUP signals to entry monitor
15:03:52 | Removed 64 expired signals  <-- 46 seconds later!
```

**Root Cause:**
In `crypto/scanning/signal_scanner.py`, signals are created with:
```python
detected_time = p["timestamp"]  # This is the BAR timestamp, not scan time!
```

For weekly bars, `detected_time` = last Sunday's date (Dec 15).
When expiry check runs on Dec 21: `age_hours = 144 hours > 24 = EXPIRED`

**Fix Required:**
Change `detected_time` to current scan time (`datetime.now(timezone.utc)`)
or use a separate `scan_time` field for expiry calculations.

**Files to Modify:**
- `crypto/scanning/signal_scanner.py` (lines 1006-1009, 1109-1112)
- Add: `detected_time=datetime.now(timezone.utc)` instead of bar timestamp

#### BUG 2: Duplicate Signal Generation

**Severity:** HIGH - Wastes resources, inflates signal counts

**Evidence:**
```
BIP 3-? LONG (1w): 3 duplicates per scan
BIP 3-? SHORT (1w): 3 duplicates per scan
ETP 3-? LONG (1w): 6 duplicates per scan
ETP 3-? SHORT (1w): 6 duplicates per scan
```

**Impact:** 64 signals added when should be ~10 unique signals.

**Root Cause:** Scanner generates same pattern multiple times for same bar.

**Investigation Needed:** Check 3-? setup detection loops for duplicate logic.

---

## Monitoring Protocol

### Poll Frequency

| Check | Interval | Command |
|-------|----------|---------|
| Daemon logs | 60s | `ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon --since '1 minute ago' --no-pager"` |
| Health status | 5 min | Check HEALTH log line for stats |
| Trade events | On occurrence | Log TRIGGER, TRADE OPENED, TARGET/STOP HIT |

### Event Tracking

When **NEW SIGNAL [COMPLETED]** detected:
1. Log pattern, symbol, timeframe, direction
2. Fetch actual bar data from Coinbase
3. Verify pattern classification is correct
4. If wrong → flag for investigation

When **TRIGGER FIRED** detected:
1. Log trigger price vs current price
2. Verify trigger level matches setup bar high/low
3. Check entry direction is correct per STRAT rules

When **SETUP signal expires** without triggering:
1. Log as potential missed opportunity
2. Note if price later broke trigger level
3. Investigate signal expiration issue

---

## Known Bugs to Check (From Equity)

### Bug 1: SETUP Signals Executed Immediately

**Equity Fix:** Skip SETUP signals in `_execute_signals()` - wait for entry_monitor

**Check:** Does crypto `crypto/scanning/daemon.py` have SETUP handling?

**Location to check:** Look for `signal_type == 'SETUP'` handling in execution flow

### Bug 2: Forming Bar as Setup Bar

**Equity Fix:** Exclude `last_bar_idx` from setup detection loops

**Check:** Does `crypto/scanning/signal_scanner.py` exclude last bar?

**Location to check:** 3-2, 2-2, and 3-? setup detection loops

### Bug 3: Signal Expiration Too Fast

**Observed:** SETUP signals expire in ~60 seconds (or less)

**Evidence:**
```
Dec 18 11:34:00 | Added signal: BIP-20DEC30-CDE 3-2D-? LONG (4h)
Dec 18 11:34:55 | Removed 1 expired signals  <-- 55 seconds!
```

**Check:** Find signal TTL setting in entry_monitor

---

## Current State (Dec 21, 2025 - 15:15 UTC)

### Daemon Stats

| Metric | Value |
|--------|-------|
| Uptime | 1 day 22 hours |
| Scans | 174 |
| Signals | 12,617 |
| Triggers | 64 |
| Executions | 6 |
| Errors | 0 |

### Key Observation

64 triggers but only 6 executions = 58 triggers did NOT result in trades.
This gap is likely due to signals expiring before triggers can be processed.

### Active Symbols

- BIP-20DEC30-CDE (Bitcoin derivative)
- ETP-20DEC30-CDE (Ethereum derivative)

### Recent Patterns

| Symbol | TF | Pattern | Type |
|--------|-----|---------|------|
| BIP | 1w | 3-2U | COMPLETED |
| BIP | 1d | 2D-1-2U | COMPLETED |
| ETP | 1w | 3-2U | COMPLETED |
| ETP | 1d | 3-2D-2U | COMPLETED |

---

## Files Reference

| File | Purpose |
|------|---------|
| `docs/CRYPTO_DAEMON_ANALYSIS.md` | Full analysis doc with trade data |
| `crypto/scanning/daemon.py` | Main daemon loop |
| `crypto/scanning/signal_scanner.py` | Pattern detection |
| `crypto/scanning/entry_monitor.py` | Trigger monitoring |
| `crypto/simulation/position_monitor.py` | Position management |

---

## VPS Access

```bash
# Check status
ssh atlas@178.156.223.251 "sudo systemctl status atlas-crypto-daemon --no-pager"

# Recent logs
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon --since '5 minutes ago' --no-pager"

# New signals only
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon --since '5 minutes ago' --no-pager | grep -E 'NEW SIGNAL|TRIGGER|TRADE|TARGET|STOP'"

# Restart if needed
ssh atlas@178.156.223.251 "sudo systemctl restart atlas-crypto-daemon"
```

---

## STRAT Reference (Quick)

### Three Universal Truths
- Type 1: Inside previous bar range
- Type 2U/2D: Broke ONE side
- Type 3: Broke BOTH sides

### Pattern Validation
- 3-2 Pattern: 1.5% measured move target
- X-1-2 Patterns: Entry at inside bar high/low break
- Reversals (3-2-2, 2-2): Traditional magnitude to reference bar

### Entry Rule
- SETUP bar: Must be CLOSED
- ENTRY bar: Enter when trigger breaks (intrabar)

---

## Session Log

### CRYPTO-MONITOR-1 (Dec 21, 2025) - COMPLETE

**Status:** BUGS FIXED AND DEPLOYED

**Accomplishments:**
- [x] Started live monitoring of crypto daemon
- [x] Identified signal expiration bug root cause
- [x] Fixed Bug 1: detected_time now uses scan time (not bar timestamp)
- [x] Fixed Bug 2: signal_id uses setup_bar_timestamp for deduplication
- [x] Deployed fixes to VPS (commit ad62441)
- [x] Verified signals now persist beyond 60 seconds

**Bug Fixes Applied:**

| Bug | Root Cause | Fix |
|-----|------------|-----|
| Signal Expiration | `detected_time = bar_timestamp` (old bars > 24h = expired) | `detected_time = datetime.now(timezone.utc)` |
| Duplicate Signals | `signal_id` used `detected_time` (unique per scan) | `signal_id` uses `setup_bar_timestamp` (same bar = dedupe) |

**Remaining Issue:**
Multiple 3-? setups still appear per timeframe because each outside bar in the data creates a setup. This is expected behavior - the scanner detects all valid outside bars, not just the most recent. Future enhancement: limit 3-? setups to the most recent outside bar only.

**Commit:** ad62441 - fix(crypto): fix signal expiration and deduplication bugs (CRYPTO-MONITOR-1)

---

## Next Session Priorities (CRYPTO-MONITOR-2)

1. **Monitor Signal Persistence** - Verify signals stay active for 24 hours
2. **Watch for Triggers** - Check if triggers now fire when price breaks levels
3. **Limit 3-? Setups** - Consider reducing to only most recent outside bar
4. **Pattern Validation** - Verify detected patterns match actual bar data

---

**Note:** For equity options work, see `docs/HANDOFF.md` (Session EQUITY-30/31)
