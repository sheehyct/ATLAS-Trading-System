# CRYPTO HANDOFF - ATLAS Crypto Daemon Monitoring

**Last Updated:** December 21, 2025 (Session EQUITY-30)
**Purpose:** Live monitoring of crypto STRAT daemon for pattern validation
**Related:** See `docs/HANDOFF.md` for equity options work

---

## Session CRYPTO-MONITOR-1: Setup (PENDING)

**Date:** December 21, 2025
**Environment:** Claude Code Desktop
**Status:** READY TO START

### Objective

Live monitoring of crypto daemon to:
1. Validate pattern classifications against actual bar data
2. Verify entry correctness when trades trigger
3. Flag missed opportunities (pattern detected → no trade)
4. Check for bugs found in equity daemon

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

## Current State (Dec 21, 2025)

### Daemon Stats

| Metric | Value |
|--------|-------|
| Scans | 173 |
| Signals | 12,500+ |
| Triggers | 64 |
| Executions | 6 |
| Errors | 0 |

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

### CRYPTO-MONITOR-1 (Pending)
- [ ] Start live monitoring
- [ ] Validate first pattern detection
- [ ] Check for signal expiration issue
- [ ] Log any trade events

---

**Note:** For equity options work, see `docs/HANDOFF.md` (Session EQUITY-30/31)
