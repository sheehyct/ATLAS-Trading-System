# Crypto Daemon Analysis - December 21, 2025

**Purpose:** Offline analysis for Claude Code for Web
**Created:** Session EQUITY-30
**Data Range:** December 15-21, 2025

---

## CRITICAL: Handoff from Equity Bug Investigation

### Bugs Found in Equity Daemon (Session EQUITY-29 & EQUITY-30)

The following bugs were found and fixed in the equity options daemon. **The crypto daemon may have the same issues:**

#### Bug 1: SETUP Signals Executed Immediately (EQUITY-29)

**Problem:** SETUP signals were being executed in `_execute_signals()` immediately without waiting for entry_monitor to detect which trigger (HIGH or LOW) breaks first.

**Impact:** Trades entered in WRONG DIRECTION because bidirectional trigger checking was bypassed.

**Fix Applied:** Skip SETUP signals in `_execute_signals()` - they must wait for entry_monitor.

**Check in Crypto:** Does the crypto daemon have the same SETUP signal bypass?

#### Bug 2: Forming Bar Used as Setup Bar (EQUITY-27)

**Problem:** The scanner used the currently FORMING bar as a setup bar for pattern detection, even though its HIGH/LOW are not finalized.

**Impact:** Invalid patterns detected with incorrect trigger/stop/target levels.

**Fix Applied:** Exclude last bar index from setup detection loops.

**Check in Crypto:** Does `crypto/scanning/signal_scanner.py` have the same forming bar exclusion?

#### Bug 3: Timing Filter Bypass (EQUITY-30 - UNEXPLAINED)

**Problem:** Trade 6 (GOOGL 3-2U CALL 1H) entered at 09:48, before the 10:30 AM minimum for 1H 2-bar patterns.

**Impact:** Trades executed before valid patterns could exist.

**Status:** Timing filter IS implemented but was bypassed somehow. Root cause unknown.

**Check in Crypto:** Crypto has no "let the market breathe" timing (24/7 market), but check if similar filter bypasses exist.

---

## Daemon Status

| Metric | Value |
|--------|-------|
| Status | ACTIVE (running 1 day 22h) |
| Scans | 173 |
| Signals Generated | 12,500+ |
| Triggers Detected | 64 |
| Executions | 6 |
| Errors | 0 |
| Entry Monitor Poll | 60s |

---

## CRITICAL OBSERVATION: Signal Expiration Issue

### Evidence from Logs

```
Dec 18 11:34:00 | Added signal to monitor: BIP-20DEC30-CDE 3-2D-? LONG (4h)
Dec 18 11:34:00 | Added 4 SETUP signals to entry monitor
Dec 18 11:34:55 | Removed 1 expired signals  <-- 55 SECONDS LATER!
```

**Pattern repeats:**
```
Dec 18 11:49:01 | Added signal: BIP-20DEC30-CDE 3-2D-? LONG (4h)
Dec 18 11:50:01 | Removed 1 expired signals   <-- 60 SECONDS LATER!
```

**And again:**
```
Dec 18 12:04:02 | Added signal: BIP-20DEC30-CDE 3-2D-? LONG (4h)
Dec 18 12:04:06 | Removed 1 expired signals   <-- 4 SECONDS LATER!
```

### Problem Analysis

1. SETUP signals are added to entry_monitor
2. Entry_monitor checks every 60 seconds (poll_interval=60s)
3. Signals expire after ~60 seconds (or sometimes IMMEDIATELY)
4. Triggers never have time to fire

**This may explain why:**
- 64 triggers detected
- But only 6 executions
- Most SETUP signals expire before triggering

---

## Actual Trade Entries with Prices

### Trade Example: BTC-PERP (Dec 18)

```
SETUP RESOLVED: BTC-PERP-INTX 2D-1-? → X-1-2U LONG @ $86,430.10
TRIGGER FIRED: @ $86,476.00 (trigger: $86,430.10)
TRADE OPENED: SIM-crypto_daemon-00002 BUY @ $86,476.00
  - Stop: $85,265.10
  - Target: $87,766.90
  - Qty: 0.016832
  - Risk: $20.38
  - Leverage: 1.4x
EXIT (TARGET): @ $87,795.60 | P&L: +$22.21 (+1.5%)
```

### Trade Example: ETH-PERP (Dec 18)

```
SETUP RESOLVED: ETH-PERP-INTX 2D-1-? → X-1-2U LONG @ $2,837.34
TRIGGER FIRED: @ $2,839.35 (trigger: $2,837.34)
TRADE OPENED: SIM-crypto_daemon-00003 BUY @ $2,839.35
  - Stop: $2,788.91
  - Target: $2,924.72
  - Qty: 0.404070
  - Risk: $20.38
  - Leverage: 1.1x
EXIT (TARGET): @ $2,925.66 | P&L: +$34.88 (+3.0%)
```

---

## Trade Summary (Dec 15-21)

### Winners

| ID | Symbol | Pattern | Entry | Exit | P&L |
|----|--------|---------|-------|------|-----|
| 00002 | BTC-PERP | 2D-1-2U | $86,476.00 | $87,795.60 | +$22.21 |
| 00003 | ETH-PERP | 2D-1-2U | $2,839.35 | $2,925.66 | +$34.88 |
| 00006 | BIP-CDE | ? | ? | $88,720.00 | +$127.10 |
| 00009 | ETP-CDE | ? | ? | $2,945.00 | +$18.01 |
| 00010 | ETP-CDE | ? | ? | $3,005.50 | +$3.27 |
| 00011 | BIP-CDE | ? | ? | $87,885.00 | +$7.15 |
| 00014 | BIP-CDE | ? | ? | $88,695.00 | +$15.76 |

### Losers

| ID | Symbol | Stop | P&L |
|----|--------|------|-----|
| 00004 | BIP-CDE | $85,270.00 | -$21.27 |
| 00007 | ETP-CDE | $2,780.00 | -$22.60 |
| 00008 | ETP-CDE | $2,841.00 | -$31.54 |

---

## Current Signals (Dec 21)

### BIP-20DEC30-CDE (Bitcoin Derivative)

| TF | Pattern | Type | Direction |
|----|---------|------|-----------|
| 1w | 3-2U | COMPLETED | LONG |
| 1d | 2D-1-2U | COMPLETED | LONG |
| 1d | 2D-1-? | SETUP | SHORT |
| 1d | 2U-2D-? | SETUP | LONG |
| 4h | 3-? | SETUP | BIDIRECTIONAL |
| 1h | 3-? | SETUP | BIDIRECTIONAL |
| 15m | 3-2D | COMPLETED | SHORT |

### ETP-20DEC30-CDE (Ethereum Derivative)

| TF | Pattern | Type | Direction |
|----|---------|------|-----------|
| 1w | 3-2U | COMPLETED | LONG |
| 1d | 3-2D | COMPLETED | SHORT |
| 1d | 3-2D-2U | COMPLETED | LONG |
| 1d | 2D-2U-? | SETUP | SHORT |
| 4h | 3-? | SETUP | BIDIRECTIONAL |
| 1h | 3-2D | COMPLETED | SHORT |

---

## Key Questions for Investigation

### 1. Signal Expiration

- Why are SETUP signals expiring after 60 seconds (or less)?
- Is there a `signal_ttl` or expiration setting that's too short?
- Should signals persist until the next bar closes?

### 2. SETUP Signal Handling

- Is the crypto daemon executing SETUP signals immediately (like the equity bug)?
- Check `crypto/scanning/daemon.py` for SETUP handling in `_execute_signals()`

### 3. Forming Bar Exclusion

- Does `crypto/scanning/signal_scanner.py` exclude the last bar from setup detection?
- Look for `last_bar_idx` exclusion in 3-2, 2-2, and 3-? setup loops

### 4. Trigger Count vs Execution Count

- 64 triggers detected but only 6 executions
- Why are 58 triggers not resulting in trades?
- Check if triggers are being blocked or filtered

---

## Files to Investigate

| File | Purpose | Check For |
|------|---------|-----------|
| `crypto/scanning/daemon.py` | Main loop | SETUP signal handling, signal expiration |
| `crypto/scanning/signal_scanner.py` | Pattern detection | Forming bar exclusion |
| `crypto/scanning/entry_monitor.py` | Trigger monitoring | Signal TTL, expiration logic |
| `crypto/simulation/position_monitor.py` | Position management | Exit handling |

---

## STRAT Methodology Reference

### Three Universal Truths (From EQUITY-30)

Price can only move in one of three ways:
1. **Type 1:** Stays within previous bar range (inside)
2. **Type 2U/2D:** Breaks ONE side (high or low)
3. **Type 3:** Breaks BOTH sides (outside)

**Critical:** Once a boundary breaks, it cannot "unbreak."

### Intrabar Classification

- A forming bar CAN be classified before close based on what it has DONE
- SETUP bars must be CLOSED (their high/low define trigger levels)
- ENTRY bars can be classified intrabar (enter when trigger breaks)

### Pattern Rules

- **3-2 Pattern:** 1.5% measured move target
- **3-2-2 Reversal:** Traditional magnitude target (reference bar)
- **X-1-2 Patterns:** Enter when bar breaks inside bar's high/low

---

## VPS Commands

```bash
# Check daemon status
ssh atlas@178.156.223.251 "sudo systemctl status atlas-crypto-daemon"

# View recent logs
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon --since '1 hour ago' --no-pager"

# View trades
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon --since '2025-12-18' | grep -E '(TARGET|STOP|TRIGGER|TRADE)'"

# Restart daemon
ssh atlas@178.156.223.251 "sudo systemctl restart atlas-crypto-daemon"
```

---

## Summary for Claude Code for Web

1. **Signal Expiration Bug:** SETUP signals expire in 60 seconds, not enough time for triggers
2. **Check for Equity Bugs:** SETUP immediate execution, forming bar as setup bar
3. **Trigger vs Execution Gap:** 64 triggers but only 6 executions - investigate why
4. **Pattern Detection:** Many 3-? bidirectional setups being created
5. **Performance:** Recent trades show mix of wins/losses, but system is generating profit

**Priority Investigation:** Why are SETUP signals expiring so fast in entry_monitor?
