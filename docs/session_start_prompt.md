# Session EQUITY-28 Startup

**Date:** December 20, 2025
**Priority:** CONTINUE TRADE AUDIT + MONITOR

## Context

Session EQUITY-27 fixed the critical forming bar bug and deployed it to VPS. Only ONE trade (AAPL Daily) was fully audited - remaining trades still need verification.

## Bug Fix Deployed

**Location:** `strat/paper_signal_scanner.py` (lines 906, 995, 1086)

**Fix Applied:** Added `last_bar_idx` exclusion to 3-2, 2-2, and 3-? setup loops. This prevents using incomplete (forming) bars as setup bars for daily/weekly/monthly timeframes.

**Commit:** `79d134b`

## Account Status (Dec 20)

**Equity Daemon (Alpaca Paper):**
- Starting: $3,000
- Current: $2,102.86 (-29.9%)
- Open Positions: 5 (AAPL PUT -55%, ACHR PUT 0%, DIA PUT +9%, GOOGL CALL +34%, QQQ CALL +19%)

## Trades Still Needing Audit

Only the AAPL Daily trade was fully analyzed in EQUITY-26. These trades need verification:

| # | Ticker | Pattern | TF | Entry Time | Status | Audited? |
|---|--------|---------|-----|------------|--------|----------|
| 1 | QQQ | 3-2D-2U CALL | 1W | Dec 18 11:02 AM | OPEN +$138 | NO |
| 2 | AAPL | 3-2D-2U CALL | 1D | Dec 18 11:02 AM | MAX_LOSS -$298 | YES (bug found) |
| 3 | AAPL | 3-2D PUT | 1W | Dec 18 11:02 AM | OPEN +$27 | NO |
| 4 | ACHR | 3-2U CALL | 1H | Dec 18 12:17 PM | STOP -$25 | NO |
| 5 | QBTS | 3-2U CALL | 1H | Dec 19 9:30 AM | TARGET +$78 | NO |
| 6 | GOOGL | 3-2U CALL | 1H | Dec 19 9:48 AM | OPEN -$25 | NO |
| 7 | ACHR | 3-2U-2D PUT | 1H | Dec 19 3:48 PM | OPEN | NO |

## "Let the Market Breathe" Clarification

Confirmed with user that hourly time thresholds are CORRECT:
- Only applies to 1H timeframe
- Previous day's last hourly bar lacks price continuity to next day's first bar (pre/post market gaps)
- 2-bar patterns: 10:30 AM (requires 2 hourly bars to form)
- 3-bar patterns: 11:30 AM (requires 3 hourly bars to form)

## Session Priorities

1. **CONTINUE AUDIT** - Verify remaining 6 trades with actual market data
2. **Monitor** - Watch for correct pattern detection with forming bar fix
3. **Optional** - Investigate why ETF patterns complete before being detected as SETUPs

## Required Reading

1. `docs/HANDOFF.md` - Session EQUITY-27 details
2. `docs/CLAUDE.md` - Development rules
3. Invoke `strat-methodology` skill before any STRAT code changes

## User Philosophy

"Accuracy over speed. No rushing to conclusions. No time limit."

The STRAT methodology requires precise bar classification and entry timing. Entry happens ON THE BREAK, but the SETUP must be based on CLOSED bars, not forming bars.
