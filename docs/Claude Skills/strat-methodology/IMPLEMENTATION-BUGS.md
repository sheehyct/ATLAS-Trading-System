# STRAT Implementation Bugs & Nuances

**Purpose:** Document critical implementation nuances and bugs discovered during crypto daemon development
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [Live Bar vs Closed Bar Detection](#1-live-bar-vs-closed-bar-detection)
2. [3-Bar vs 2-Bar Pattern Distinction](#2-3-bar-vs-2-bar-pattern-distinction)
3. [Stop Placement Rules](#3-stop-placement-rules)
4. [Entry Trigger Mechanics](#4-entry-trigger-mechanics)
5. [Common Bugs Fixed](#5-common-bugs-fixed)

---

## 1. Live Bar vs Closed Bar Detection

### Critical Concept

**The last bar in any dataset is LIVE (incomplete) and cannot be used as a setup bar for 3-bar patterns.**

### Why This Matters

For 3-bar patterns like `2D-1-?` or `3-1-?`:
- The inside bar (middle bar) **MUST be closed/complete**
- We watch the **NEXT bar** (live bar) for the break
- Entry triggers when the live bar breaks the closed inside bar's high/low

### The Bug

```
WRONG: Detect live bar as inside bar → trigger entry when that SAME bar moves
RIGHT: Detect closed inside bar → trigger entry when NEXT bar breaks its levels
```

**Real Example (BTC-PERP-INTX 2D-1-? 4h SHORT):**
- At 3:05 PM EST, system detected a "2D-1-?" pattern
- The "inside bar" was actually the LIVE 4h candle (not yet closed)
- Entry triggered when that same candle's price moved beyond its own earlier bounds
- Trade entered AND exited on the same incomplete bar

### The Fix

In `signal_scanner.py`, exclude the last bar from setup detection:

```python
# CRITICAL: Exclude the last bar (live/incomplete bar) from 3-bar setup detection.
# For 3-bar patterns (X-1-?), the inside bar must be CLOSED.
last_bar_idx = len(setup_mask) - 1
for i in range(len(setup_mask)):
    # Skip the last bar - it's live/incomplete and cannot be a valid setup bar
    if i == last_bar_idx:
        continue
    if setup_mask[i]:
        # Process setup...
```

---

## 2. 3-Bar vs 2-Bar Pattern Distinction

### Critical Concept

**A directional bar followed by an inside bar does NOT automatically mean a 3-bar pattern (X-1-2). The inside bar's state determines the pattern type.**

### Pattern Type Determination

| Inside Bar State | Pattern Type | Entry Timing |
|------------------|--------------|--------------|
| **CLOSED** inside bar | 3-bar: `X-1-?` | Wait for NEXT bar to break |
| **LIVE** bar starts as inside, then breaks | 2-bar: `X-2U` or `X-2D` | Entry is LIVE at break |

### Example Scenarios

**Scenario A: Closed Inside Bar (3-bar pattern)**
```
Bar -2: 2D (closed) - First directional bar
Bar -1: 1 (closed) - Inside bar, CLOSED at bar end
Bar 0:  ? (live)   - WATCHING for break of Bar -1's high/low

Pattern: 2D-1-? (waiting)
Entry: When Bar 0 breaks Bar -1's high → becomes 2D-1-2U
       When Bar 0 breaks Bar -1's low  → becomes 2D-1-2D
```

**Scenario B: Live Bar Breaking (2-bar pattern)**
```
Bar -1: 2D (closed) - Directional bar
Bar 0:  Currently inside Bar -1's range (live)
        ...then breaks Bar -1's high

Pattern: 2D-2U (reversal)
The "inside" period was just the live bar consolidating before breaking.
```

### Why This Distinction Matters

- **3-bar patterns** have the inside bar as part of the structure - it's committed
- **2-bar patterns** form when a live bar that started as "inside" breaks out
- Treating a live inside bar as a completed setup bar causes premature entries

---

## 3. Stop Placement Rules

### Critical Rule

**For 2-1-2 patterns, stops go at the FIRST DIRECTIONAL BAR's extreme, NOT the inside bar.**

### Stop Placement by Pattern

| Pattern | Stop Location | Rationale |
|---------|---------------|-----------|
| `2U-1-2U` (bullish) | First 2U bar's LOW | Structural support |
| `2D-1-2D` (bearish) | First 2D bar's HIGH | Structural resistance |
| `2D-1-2U` (bullish reversal) | First 2D bar's LOW | Failed breakdown level |
| `2U-1-2D` (bearish reversal) | First 2U bar's HIGH | Failed breakout level |
| `3-1-2U` (bullish) | Outside bar's LOW | Wide structural stop |
| `3-1-2D` (bearish) | Outside bar's HIGH | Wide structural stop |

### The Bug

**WRONG (tighter, incorrect):**
```python
# Using inside bar for stops
stops[i] = low[i-1]   # Inside bar low for bullish
stops[i] = high[i-1]  # Inside bar high for bearish
```

**RIGHT (per STRAT methodology):**
```python
# Using first directional bar for stops
stops[i] = low[i-2]   # First directional bar low for bullish
stops[i] = high[i-2]  # First directional bar high for bearish
```

### Visual Example

```
2D-1-2U Pattern (Bullish Reversal):

        [2U]    ← Bar 0: Exit bar, entry trigger = HIGH of this bar
         |
       [1]      ← Bar -1: Inside bar (NOT used for stop)
         |
      [2D]      ← Bar -2: First directional bar, STOP = LOW of this bar
         |
    ----+----

Entry: Break above Bar 0's high
Stop:  Bar -2's low (NOT Bar -1's low)
```

---

## 4. Entry Trigger Mechanics

### "Inside Bar Break" Definition

**CORRECT:** The NEXT candle (live) breaks the PREVIOUS bar's high/low
**WRONG:** The inside bar hits its own high/low levels

### Trigger Timing

- Entry is **LIVE** at the break - does NOT wait for bar close
- For bullish: triggered when price exceeds inside bar's HIGH
- For bearish: triggered when price breaks inside bar's LOW

### Code Flow

```
1. Detect CLOSED inside bar (setup)
2. Store inside bar's high/low as trigger levels
3. Monitor NEXT (live) bar's price
4. When price breaks trigger level → ENTRY
5. Entry price = current market price at break (not trigger level)
```

### Important Nuance

The pattern name changes based on which direction breaks:
- `2D-1-?` + upside break → `2D-1-2U` (bullish reversal)
- `2D-1-?` + downside break → `2D-1-2D` (bearish continuation)

The `?` in `X-1-?` means "waiting for directional resolution."

---

## 5. Common Bugs Fixed

### Bug 1: Live Bar as Setup Bar

**Symptom:** Trade enters and exits on same candle
**Cause:** System detected live bar as inside bar for 3-bar setup
**File:** `crypto/scanning/signal_scanner.py`
**Fix:** Exclude last bar from setup detection loops

### Bug 2: Incorrect Stop Placement

**Symptom:** Stops too tight, hit more frequently than expected
**Cause:** Using inside bar high/low instead of first directional bar
**File:** `strat/pattern_detector.py`
**Fix:** Changed `stops[i] = high/low[i-1]` to `high/low[i-2]`

### Bug 3: Pattern Misidentification

**Symptom:** 2-bar patterns labeled as 3-bar patterns
**Cause:** Not distinguishing between closed inside bar (3-bar) and live bar that starts inside then breaks (2-bar)
**Understanding:** A live bar that starts as "inside" and then breaks is a 2-bar pattern, not 3-bar

### Bug 4: Missing Discord Entry Alerts (Equity Daemon)

**Symptom:** Orders fill on Alpaca but no Discord entry notification received
**Cause:** Two code paths for execution, but only one sends Discord alerts:
- `EntryMonitor` triggers → `_on_entry_triggered()` → sends Discord alert ✅
- `run_scan()` → `_execute_signals()` directly → NO Discord alert ❌

**Files:** `strat/signal_automation/daemon.py`
**Fix:** Added Discord entry alert to `_execute_signals()` code path

### Bug 5: Late-Day Hourly Pattern Entries ("Let the Market Breathe")

**Symptom:** Hourly patterns entering at 3:30 PM and exiting at 9:31 AM next day
**Cause:** No time-of-day filtering for hourly patterns - first bar hasn't closed

**Rule:** For hourly (1H) patterns only:
- 2-bar patterns: Earliest entry at **10:30 AM EST** (after first 1H bar closes)
- 3-bar patterns: Earliest entry at **11:30 AM EST** (after first two 1H bars close)
- Daily/Weekly/Monthly: No restriction (larger timeframes carry more significance)

**Files:**
- `strat/signal_automation/entry_monitor.py` - Added `is_hourly_entry_allowed()` method
- `strat/signal_automation/daemon.py` - Added `_is_hourly_entry_allowed()` and filtering in `_execute_signals()`

### Bug 6: Dashboard Closed Trades Missing Pattern Column

**Symptom:** Closed trades show "-" for pattern column
**Cause:** Alpaca doesn't store pattern info - need to correlate with signal store

**Fix:**
1. Added `executed_osi_symbol` field to `StoredSignal`
2. Store OSI symbol when signal executes (`set_executed_osi_symbol()`)
3. Look up pattern by OSI symbol when fetching closed trades (`get_signal_by_osi_symbol()`)
4. Added Pattern column to closed trades table

**Files:**
- `strat/signal_automation/signal_store.py` - Added `executed_osi_symbol` field and lookup methods
- `strat/signal_automation/daemon.py` - Store OSI symbol on execution
- `dashboard/data_loaders/options_loader.py` - Look up pattern from signal store
- `dashboard/components/options_panel.py` - Added Pattern column to table

---

## Summary Checklist

When implementing STRAT pattern detection:

- [ ] **Live bar exclusion:** Never use last bar as setup bar for 3-bar patterns
- [ ] **Stop placement:** Use first directional bar (index -2), not inside bar (index -1)
- [ ] **Pattern distinction:** Closed inside bar = 3-bar, live bar breaking = 2-bar
- [ ] **Entry timing:** Entry is LIVE at break, not waiting for bar close
- [ ] **Trigger definition:** "Inside bar break" = NEXT bar breaks PREVIOUS bar's levels
- [ ] **Time filtering:** Hourly patterns restricted to 10:30 AM (2-bar) / 11:30 AM (3-bar)
- [ ] **Discord alerts:** Ensure ALL execution paths send entry alerts
- [ ] **Pattern tracking:** Store OSI symbol on execution for closed trade correlation

---

## Related Files

- `crypto/scanning/signal_scanner.py` - Setup detection with live bar exclusion
- `strat/pattern_detector.py` - Pattern detection with correct stop placement
- `crypto/scanning/entry_monitor.py` - Entry trigger monitoring
- `strat/signal_automation/entry_monitor.py` - Hourly time filtering
- `strat/signal_automation/daemon.py` - Execution with Discord alerts and time filtering
- `strat/signal_automation/signal_store.py` - OSI symbol tracking for pattern correlation
- `dashboard/data_loaders/options_loader.py` - Closed trades with pattern lookup
- `dashboard/components/options_panel.py` - Pattern column in closed trades table

---

**Session:** CRYPTO-10 (Review daemon strategy)
**Commits:**
- `1b8e295` - fix(strat): exclude live bars from 3-bar setup detection and fix stop placement
- `TBD` - fix(equity): add time filtering, Discord alerts, and pattern tracking
