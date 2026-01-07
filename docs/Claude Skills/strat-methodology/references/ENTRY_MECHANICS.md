# STRAT Entry Mechanics - Precise Triggers, Stops, and Targets

**Purpose:** Exact mechanical entry/exit rules for algorithmic implementation
**Parent:** [SKILL.md](../SKILL.md)

---

## Critical Rule: Entry Timing

**ENTER THE INSTANT PRICE BREAKS TRIGGER - DO NOT WAIT FOR ENTRY BAR TO CLOSE**

### Intrabar Classification (Three Universal Truths)

A bar can be classified BEFORE it closes based on what it has done:
```
Bar opens -> Type 1 (no boundary broken yet)
Price breaks previous high -> Now AT LEAST 2U (could become 3)
Price breaks previous low -> Now AT LEAST 2D (could become 3)
Price breaks BOTH -> Type 3 (final - cannot change)
```

**Once a boundary is broken, it cannot be "unbroken."**

### Setup vs Entry Bar

| Bar Role | When Classified | Why |
|----------|-----------------|-----|
| **SETUP bar** | Must be CLOSED | Defines trigger/stop/target price levels |
| **ENTRY bar** | Classified intrabar | Enter the moment it breaks trigger |

### Timeframe Differences

| Timeframe | Bar Open Behavior |
|-----------|-------------------|
| **Intraday** (15m, 30m, 1H) | Open = Previous Close (no gap) - bar starts as Type 1 |
| **Daily+** (1D, 1W, 1M) | Gap possible - bar can OPEN as 2U, 2D, or 3 |

---

## Quick Reference Tables

### Entry Triggers

| Pattern | Entry Trigger | Formula |
|---------|---------------|---------|
| 3-1-2D | Low of 1 bar - 0.01 | `low[bar_1_idx] - 0.01` |
| 3-1-2U | High of 1 bar + 0.01 | `high[bar_1_idx] + 0.01` |
| 2D-1-2U | High of 1 bar + 0.01 | `high[bar_1_idx] + 0.01` |
| 2U-1-2D | Low of 1 bar - 0.01 | `low[bar_1_idx] - 0.01` |
| 2D-2U | High of 2D bar + 0.01 | `high[bar_2d_idx] + 0.01` |
| 2U-2D | Low of 2U bar - 0.01 | `low[bar_2u_idx] - 0.01` |
| 3-2D | Low of 3 bar - 0.01 | `low[bar_3_idx] - 0.01` |
| 3-2U | High of 3 bar + 0.01 | `high[bar_3_idx] + 0.01` |

### Stop Loss

| Pattern | Stop Loss | Formula |
|---------|-----------|---------|
| 3-1-2D | High of 3 bar | `high[bar_3_idx]` |
| 3-1-2U | Low of 3 bar | `low[bar_3_idx]` |
| 2D-1-2U | Low of first 2D bar | `low[bar_2d_idx]` |
| 2U-1-2D | High of first 2U bar | `high[bar_2u_idx]` |
| 2D-2U | Low of 2D bar | `low[bar_2d_idx]` |
| 2U-2D | High of 2U bar | `high[bar_2u_idx]` |
| 3-2D | High of 3 bar | `high[bar_3_idx]` |
| 3-2U | Low of 3 bar | `low[bar_3_idx]` |

### Targets

| Pattern | Target Method | Formula |
|---------|---------------|---------|
| 3-1-2D | Low of 3 bar | `low[bar_3_idx]` |
| 3-1-2U | High of 3 bar | `high[bar_3_idx]` |
| 2D-1-2U | High of first 2D bar | `high[bar_2d_idx]` |
| 2U-1-2D | Low of first 2U bar | `low[bar_2u_idx]` |
| 2D-2U | High of reference bar | `high[bar_ref_idx]` |
| 2U-2D | Low of reference bar | `low[bar_ref_idx]` |
| 3-2D | 1.5% below entry | `entry - (entry * 0.015)` |
| 3-2U | 1.5% above entry | `entry + (entry * 0.015)` |
| 3-2D-2U (Trade 2) | High of 3 bar | `high[bar_3_idx]` |
| 3-2U-2D (Trade 2) | Low of 3 bar | `low[bar_3_idx]` |

**Critical Target Rules:**
- 3-2 standalone patterns: 1.5% measured move
- 3-2-2 patterns (Trade 2 after reversal): Traditional magnitude (high/low of 3 bar)

---

## 2-2 Reversal Validation (CRITICAL)

For valid 2-2 reversal, reference bar (bar BEFORE pattern) MUST be directional:

| Reference Bar | Result |
|---------------|--------|
| 2U or 2D | Valid 2-2 Reversal |
| 1 (Inside) | NOT valid - Rev Strat pattern |
| 3 (Outside) | NOT valid - 3-2-2 pattern |

```python
def is_valid_22_reversal(bars, idx):
    """Reference bar must be directional (2U or 2D)."""
    if idx < 2:
        return False, None
    ref_bar = bars[idx-2]
    if ref_bar == 2 or ref_bar == -2:
        return True, '2-2_reversal'
    elif ref_bar == 1:
        return False, 'rev_strat'
    elif ref_bar == 3:
        return False, '3-2-2'
    return False, 'invalid'
```

---

## Gap Handling (Daily+ Timeframes)

**Intraday:** Previous bar close = current bar open (no gap handling needed)

**Daily and higher:** Gap may occur

```python
def handle_gap_entry(trigger, bar_open, direction):
    """If bar gaps through trigger, enter at open not trigger."""
    if direction == 'long':
        return bar_open if bar_open > trigger else trigger
    else:  # short
        return bar_open if bar_open < trigger else trigger
```

---

## "Let the Market Breathe" - Intraday Timing Rules

For intraday patterns, overnight gaps break bar continuity. Yesterday's last bar and today's first bar are NOT a valid pattern sequence.

### Hourly (1H) Patterns

| Pattern Type | Bars Needed | First Tradeable Time | Reasoning |
|--------------|-------------|----------------------|-----------|
| 2-bar (2-2, 3-2) | 1 closed + forming | **10:30 AM EST** | First bar (09:30-10:30) must close |
| 3-bar (3-2-2, 2-1-2, 3-1-2) | 2 closed + forming | **11:30 AM EST** | First two bars must close |

### Other Intraday Timeframes

| Timeframe | 2-bar Earliest | 3-bar Earliest |
|-----------|----------------|----------------|
| 15m | 09:45 AM | 10:00 AM |
| 30m | 10:00 AM | 10:30 AM |
| 1H | 10:30 AM | 11:30 AM |

### 15:30 Bar Rule (1H Timeframe)

The last hourly bar (15:30-16:00) is truncated to 30 minutes.

**Trades on 15:30 bar MUST exit before 16:00** - holding overnight exposes you to:
1. Gap risk against your position
2. Extra theta decay (options)
3. Pattern logic breaking across the overnight gap

---

## Pattern-Specific Mechanics

### 3-1-2D (Bearish Reversal)

**Structure:** 3 -> 1 -> 2D
- Bar[2]: Outside bar (broke both H[3] and L[3])
- Bar[1]: Inside bar (H <= H[2] AND L >= L[2])
- Bar[0]: 2D when price breaks L[1]

**Entry:** `low[bar_1_idx] - 0.01` - Enter INSTANT price breaks inside bar low
**Stop:** `high[bar_3_idx]` - High of outside bar
**Target:** `low[bar_3_idx]` - Low of outside bar

```python
def detect_312d(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == -2:
        return {
            'pattern': '3-1-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-2],
            'target': low[idx-2]
        }
    return None
```

### 3-1-2U (Bullish Reversal)

**Structure:** 3 -> 1 -> 2U
**Entry:** `high[bar_1_idx] + 0.01`
**Stop:** `low[bar_3_idx]`
**Target:** `high[bar_3_idx]`

```python
def detect_312u(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == 2:
        return {
            'pattern': '3-1-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-2],
            'target': high[idx-2]
        }
    return None
```

### 2D-1-2U (Bullish Reversal)

**Structure:** 2D -> 1 -> 2U
**Entry:** `high[bar_1_idx] + 0.01`
**Stop:** `low[bar_2d_idx]` - Low of first bar (2D)
**Target:** `high[bar_2d_idx]` - High of first bar (2D)

```python
def detect_2d12u(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-2] == -2 and bars[idx-1] == 1 and bars[idx] == 2:
        return {
            'pattern': '2D-1-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-2],
            'target': high[idx-2]
        }
    return None
```

### 2U-1-2D (Bearish Reversal)

**Structure:** 2U -> 1 -> 2D
**Entry:** `low[bar_1_idx] - 0.01`
**Stop:** `high[bar_2u_idx]`
**Target:** `low[bar_2u_idx]`

```python
def detect_2u12d(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-2] == 2 and bars[idx-1] == 1 and bars[idx] == -2:
        return {
            'pattern': '2U-1-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-2],
            'target': low[idx-2]
        }
    return None
```

### 2D-2U (Bullish Reversal)

**Structure:** ref -> 2D -> 2U (reference bar MUST be 2U or 2D)
**Entry:** `high[bar_2d_idx] + 0.01`
**Stop:** `low[bar_2d_idx]`
**Target:** `high[bar_ref_idx]` - High of reference bar

```python
def detect_2d2u(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-1] == -2 and bars[idx] == 2:
        ref_bar = bars[idx-2]
        if ref_bar != 2 and ref_bar != -2:
            return None  # Not valid 2-2 reversal
        return {
            'pattern': '2D-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-1],
            'target': high[idx-2]
        }
    return None
```

### 2U-2D (Bearish Reversal)

**Structure:** ref -> 2U -> 2D (reference bar MUST be 2U or 2D)
**Entry:** `low[bar_2u_idx] - 0.01`
**Stop:** `high[bar_2u_idx]`
**Target:** `low[bar_ref_idx]` - Low of reference bar

```python
def detect_2u2d(bars, high, low, idx):
    if idx < 2:
        return None
    if bars[idx-1] == 2 and bars[idx] == -2:
        ref_bar = bars[idx-2]
        if ref_bar != 2 and ref_bar != -2:
            return None  # Not valid 2-2 reversal
        return {
            'pattern': '2U-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-1],
            'target': low[idx-2]
        }
    return None
```

### 3-2D (Bearish, 1.5% Target)

**Structure:** 3 -> 2D
**Entry:** `low[bar_3_idx] - 0.01`
**Stop:** `high[bar_3_idx]`
**Target:** `entry - (entry * 0.015)` - 1.5% measured move

```python
def detect_32d(bars, high, low, idx):
    if idx < 1:
        return None
    if bars[idx-1] == 3 and bars[idx] == -2:
        entry_trigger = low[idx-1] - 0.01
        return {
            'pattern': '3-2D',
            'direction': 'short',
            'entry_trigger': entry_trigger,
            'stop': high[idx-1],
            'target': entry_trigger - (entry_trigger * 0.015),
            'target_method': '1.5_percent'
        }
    return None
```

### 3-2U (Bullish, 1.5% Target)

**Structure:** 3 -> 2U
**Entry:** `high[bar_3_idx] + 0.01`
**Stop:** `low[bar_3_idx]`
**Target:** `entry + (entry * 0.015)` - 1.5% measured move

```python
def detect_32u(bars, high, low, idx):
    if idx < 1:
        return None
    if bars[idx-1] == 3 and bars[idx] == 2:
        entry_trigger = high[idx-1] + 0.01
        return {
            'pattern': '3-2U',
            'direction': 'long',
            'entry_trigger': entry_trigger,
            'stop': low[idx-1],
            'target': entry_trigger + (entry_trigger * 0.015),
            'target_method': '1.5_percent'
        }
    return None
```

---

## 3-2-2 Sequential Patterns (Trade Flip)

When a 3-2 pattern reverses, this creates a second trade opportunity with TRADITIONAL magnitude targets.

### 3-2D-2U (Short then Long)

**Trade 1:** 3-2D short with 1.5% target
**Trade 2:** When Bar[0] becomes 2U
- Entry: `high[bar_2d_idx] + 0.01`
- Stop: `low[bar_2d_idx]`
- Target: `high[bar_3_idx]` - HIGH of 3 bar (traditional magnitude, NOT 1.5%)

**Trade Flip Logic:**
```
If Trade 1 (short) still OPEN when 2U triggers:
  1. IMMEDIATELY EXIT Trade 1
  2. ENTER Trade 2 (long)
  3. New target: high of 3 bar
```

### 3-2U-2D (Long then Short)

**Trade 1:** 3-2U long with 1.5% target
**Trade 2:** When Bar[0] becomes 2D
- Entry: `low[bar_2u_idx] - 0.01`
- Stop: `high[bar_2u_idx]`
- Target: `low[bar_3_idx]` - LOW of 3 bar (traditional magnitude, NOT 1.5%)

---

## VectorBT Pro Complete Detector

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

@njit
def classify_bars_nb(high, low):
    """Classify bars as Type 1, 2U, 2D, or 3."""
    n = len(high)
    bars = np.zeros(n, dtype=np.int8)
    bars[0] = 0
    for i in range(1, n):
        broke_high = high[i] > high[i-1]
        broke_low = low[i] < low[i-1]
        if broke_high and broke_low:
            bars[i] = 3
        elif broke_high:
            bars[i] = 2
        elif broke_low:
            bars[i] = -2
        else:
            bars[i] = 1
    return bars

@njit
def detect_entries_nb(bars, high, low):
    """Detect all STRAT patterns and return entry signals."""
    n = len(bars)
    entries = np.zeros(n, dtype=np.int8)
    entry_prices = np.zeros(n, dtype=np.float64)
    stops = np.zeros(n, dtype=np.float64)
    targets = np.zeros(n, dtype=np.float64)
    
    for i in range(2, n):
        # 3-1-2D
        if bars[i-2] == 3 and bars[i-1] == 1 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-2]
            targets[i] = low[i-2]
        
        # 3-1-2U
        elif bars[i-2] == 3 and bars[i-1] == 1 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-2]
            targets[i] = high[i-2]
        
        # 2D-1-2U
        elif bars[i-2] == -2 and bars[i-1] == 1 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-2]
            targets[i] = high[i-2]
        
        # 2U-1-2D
        elif bars[i-2] == 2 and bars[i-1] == 1 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-2]
            targets[i] = low[i-2]
        
        # 2D-2U (validate reference bar)
        elif bars[i-1] == -2 and bars[i] == 2:
            ref_bar = bars[i-2]
            if ref_bar == 2 or ref_bar == -2:
                entries[i] = 1
                entry_prices[i] = high[i-1] + 0.01
                stops[i] = low[i-1]
                targets[i] = high[i-2]
        
        # 2U-2D (validate reference bar)
        elif bars[i-1] == 2 and bars[i] == -2:
            ref_bar = bars[i-2]
            if ref_bar == 2 or ref_bar == -2:
                entries[i] = -1
                entry_prices[i] = low[i-1] - 0.01
                stops[i] = high[i-1]
                targets[i] = low[i-2]
        
        # 3-2D (1.5% target)
        elif bars[i-1] == 3 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-1]
            targets[i] = entry_prices[i] - (entry_prices[i] * 0.015)
        
        # 3-2U (1.5% target)
        elif bars[i-1] == 3 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-1]
            targets[i] = entry_prices[i] + (entry_prices[i] * 0.015)
    
    return entries, entry_prices, stops, targets
```

---

## Common Mistakes

### WRONG: Waiting for bar close
```python
# WRONG
if bars[i] == -2:  # Bar already closed
    enter_short()   # TOO LATE
```

### CORRECT: Enter on trigger break
```python
# CORRECT
if current_price < trigger:  # Intrabar
    enter_short()             # IMMEDIATE
```

### WRONG: Using current bar high as trigger
```python
# WRONG for 3-1-2 patterns
entry = high[idx]  # Current bar high - INCORRECT
```

### CORRECT: Using inside bar bounds
```python
# CORRECT for 3-1-2 patterns  
entry = high[idx-1] + 0.01  # Inside bar high + buffer
```

---

## Rev Strat Patterns (FUTURE IMPLEMENTATION)

Rev Strats are failed inside bar patterns that reverse. When X-1-2 triggers and next bar reverses:

| Original | Bar 4 | Result | Trade 2 Target |
|----------|-------|--------|----------------|
| 3-1-2D | 2U | 3-1-2D-2U Rev Strat | High of 3 bar |
| 3-1-2U | 2D | 3-1-2U-2D Rev Strat | Low of 3 bar |
| 2U-1-2D | 2U | 2U-1-2D-2U Rev Strat | High of 2U bar |
| 2D-1-2U | 2D | 2D-1-2U-2D Rev Strat | Low of 2D bar |

**Status:** Documented for reference, not yet implemented in detection logic.
