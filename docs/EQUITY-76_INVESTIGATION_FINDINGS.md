# EQUITY-76 Investigation: AMD 3-2 Pattern Issues

**Date:** January 21, 2026
**Status:** Priority 0 FIXED (Session EQUITY-76) - Priorities 1-2 Deferred
**Plan File:** `C:\Users\sheeh\.claude\plans\cozy-wandering-starlight.md`

---

## Summary

Investigation into AMD 3-2U 1D trade showing wrong target (5.88% instead of 1.5%) and duplicate Discord alerts. Also discovered why only AMD is being traded despite 21 tickers in watchlist.

---

## Critical Bugs Found

### Priority 0: WRONG 3-2 Target Calculation - FIXED

**Location:** `strat/pattern_detector.py` lines 1402-1405 (`detect_outside_bar_setups_nb`)

**Bug:** Using bar_range (1R measured move) instead of 1.5% fixed target per STRAT methodology.

**Root Cause:** EQUITY-62 fixed `detect_32_patterns_nb` (historical/backtesting path) but missed `detect_outside_bar_setups_nb` (live setup detection path). Two different functions, only one was fixed.

**Fix Applied (EQUITY-76):**
```python
# Session EQUITY-76: Simple 1.5% target per strat-methodology
target_long[i] = high[i] * 1.015    # 1.5% above entry (underlying price)
target_short[i] = low[i] * 0.985    # 1.5% below entry (underlying price)
```

**Verification:** 66 tests passing (16 pattern_detector + 50 paper_signal_scanner)

---

### Priority 1: Duplicate Discord Alerts

**Location:** `strat/signal_automation/daemon.py` line 380 (`_on_entry_triggered`)

**Bug:** No deduplication check before `send_entry_alert()`. Three independent code paths can send alerts:
1. `_on_entry_triggered()` at line 380
2. `_execute_triggered_pattern()` at line 1314
3. `_execute_signals()` at line 1492

**Fix:** Add `_entry_alerts_sent` tracking set before sending alerts.

---

### Priority 2: Only AMD Being Traded (12 Blocking Mechanisms)

**Root Cause:** TFC (Timeframe Continuity) filtering is the main bottleneck.

**Why AMD passes:**
- AMD trends more → Type 2U/2D bars (directional)
- Higher TFC scores (4/4 common)

**Why others fail:**
- SPY/QQQ often consolidate → Type 1 (inside) bars
- Type 1 bars don't count toward TFC alignment
- TFC score drops below threshold → REJECTED

**12 Blocking Mechanisms Found:**

| # | Mechanism | Location |
|---|-----------|----------|
| 1 | SETUP signals wait for entry_monitor | daemon.py:1443-1448 |
| 2 | COMPLETED double-skip | daemon.py:1454-1459 |
| 3 | HISTORICAL_TRIGGERED status | executor.py:301-314 |
| 4 | Stale Setup (2 trading days for 1D) | daemon.py:955-969 |
| 5 | TFC re-evaluation at entry | daemon.py:1115-1122 |
| 6 | Magnitude < 0.5% | daemon.py:764-778 |
| 7 | TFC score < 2/3 | daemon.py:835-849 |
| 8 | Position limit (5 max) | executor.py:350-359 |
| 9 | Contract selection failure | executor.py:370-375 |
| 10 | Outside market hours | daemon.py:1159-1168 |
| 11 | "Let market breathe" timing | daemon.py:1331-1415 |
| 12 | Entry monitor failure | entry_monitor.py:240-246 |

**DIA Example:** Valid 2D-2U 1D pattern with passing TFC wasn't traded despite only 1 position open. Most likely blocked by #3 (HISTORICAL_TRIGGERED), #4 (Stale Setup), or #5 (TFC re-eval).

---

## Key STRAT Methodology Clarification

**3-? and 3-2 are the SAME pattern at different stages:**

| Stage | Description | Tradeable? |
|-------|-------------|------------|
| 3-? | Outside bar CLOSED, watching for forming bar to break | YES - enter on break |
| 3-2U/3-2D | Entry HAPPENED (live break occurred) | Entry already taken |
| Bar 2 closed | Missed the trade entirely | NO - don't chase |

**Key Rule:** Enter THE INSTANT price breaks trigger level. Do NOT wait for bar to close.

---

## Files Modified / To Modify

| File | Lines | Fix | Status |
|------|-------|-----|--------|
| `strat/pattern_detector.py` | 1402-1405, 1427-1429 | Change 1R to 1.5% for 3-2 targets | FIXED (EQUITY-76) |
| `strat/signal_automation/daemon.py` | 380 | Add alert deduplication | DEFERRED |
| `strat/signal_automation/daemon.py` | 301-314 | Review HISTORICAL_TRIGGERED | DEFERRED |
| `strat/signal_automation/daemon.py` | 955-969 | Review 2-day stale logic | DEFERRED |
| `strat/timeframe_continuity.py` | 155-170 | Fix Type 3 data handling | DEFERRED |

---

## OpenMemory Reference

Memory ID: `b21a816d-c8e4-49e6-bcf2-8d14632d7a16`
Tags: STRAT, bug-fix, 3-2-pattern, duplicate-alerts, TFC-filtering, EQUITY-76

---

## Next Steps (After Tech Debt)

1. Fix the 3-2 target calculation (Priority 0)
2. Add duplicate alert deduplication (Priority 1)
3. Review the 12 blocking mechanisms for over-filtering
4. Consider relaxing TFC requirements or adding logging for rejected signals
