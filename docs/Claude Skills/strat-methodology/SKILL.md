---
name: strat-methodology
description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics (triggers, stops, targets), position management (break-even, trailing stops, rolling profits), and options integration (strike selection, Greeks, cheap options strategy). Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns. Requires VectorBT Pro for backtesting implementation.
---

# STRAT Methodology Implementation

**Version:** 2.3 (Per STRAT methodology best practices update)
**Purpose:** Navigation hub for STRAT trading system implementation

---

## Contents

- [Pre-Entry Checklist](#pre-entry-checklist)
- [The Three Universal Truths](#the-three-universal-truths)
- [Intrabar Classification](#intrabar-classification)
- [Timeframe Gap Handling](#timeframe-gap-handling)
- [Intraday Timing Rules](#let-the-market-breathe---intraday-timing-rules)
- [Quick Reference](#quick-reference)
- [File Navigation](#file-navigation)
- [Implementation Workflow](#implementation-workflow)
- [Common Mistakes](#common-mistakes)
- [Requirements](#requirements)

---

## Pre-Entry Checklist

Before entering ANY STRAT trade, verify:

| Check | Question | Fail Action |
|-------|----------|-------------|
| 1. Pattern Valid | Is bar sequence correct (e.g., 2D-1-2U not 1-2D-2U)? | No trade |
| 2. Setup Bar Closed | Is the setup bar (defines trigger) fully closed? | Wait |
| 3. Trigger Correct | Entry trigger = setup bar high/low + 0.01? | Recalculate |
| 4. Stop Defined | Stop = opposite extreme of pattern? | Define before entry |
| 5. Target Valid | Target exists (not blocked by S/R)? | Skip or reduce size |
| 6. Timeframe Timing | 1H: After 10:30 (2-bar) or 11:30 (3-bar)? | Wait |
| 7. TFC Alignment | Higher timeframes supportive or neutral? | Reduce size if against |

**Critical:** Enter ON THE BREAK, not at bar close. The moment price breaks
the trigger level, enter immediately.

---

## The Three Universal Truths

Price can ONLY move in one of three ways relative to the previous bar - no other possibility exists:

| Scenario | Condition | Bar Type | Can Change To |
|----------|-----------|----------|---------------|
| **1** | Price stays within previous bar range | Inside (1) | 2U, 2D, or 3 |
| **2** | Price takes out ONE side (high OR low) | Directional (2U or 2D) | 3 only |
| **3** | Price takes out BOTH sides (high AND low) | Outside (3) | Nothing - final |

**Critical Insight:** Once a boundary is broken, it cannot be "unbroken."
- A bar that breaks the high is AT MINIMUM a 2U (could become 3 if it later breaks low)
- A bar that breaks the low is AT MINIMUM a 2D (could become 3 if it later breaks high)
- A bar that breaks both is definitively a 3 - no further evolution possible

---

## Intrabar Classification

**You CAN classify a forming bar before it closes** based on what it has done:

```
Bar opens -> Type 1 (no boundary broken yet)
Price breaks previous high -> Now AT LEAST 2U
Price breaks previous low -> Now AT LEAST 2D
Price breaks BOTH -> Type 3 (final classification)
```

**Why this matters for entry:**
- SETUP bars must be CLOSED (their high/low define trigger levels)
- ENTRY bars can be classified intrabar (enter the moment trigger breaks)
- Once the entry bar breaks the trigger, you know it's at least 2U/2D - enter immediately

---

## Timeframe Gap Handling

| Timeframe | Open vs Previous Close | Implication |
|-----------|------------------------|-------------|
| **Intraday** (15m, 30m, 1H) | Same price (no gap) | Bar always starts as Type 1 |
| **Daily+** (1D, 1W, 1M) | Gap possible (pre/post market) | Bar can OPEN as 2U, 2D, or even 3 |

**Daily+ Example:** Yesterday closes at $500, overnight trading moves price, today opens at $520.
- If $520 > yesterday's high -> Bar OPENS as 2U (already broke high)
- Pattern may be COMPLETE at market open - entry is IMMEDIATE

---

## "Let the Market Breathe" - Intraday Timing Rules

For intraday patterns, overnight gaps break bar continuity. You cannot use yesterday's last bar with today's first bar as a pattern.

### Hourly (1H) Patterns

| Pattern Type | Bars Needed | First Tradeable Time |
|--------------|-------------|----------------------|
| 2-bar (2-2, 3-2) | 1 closed + forming | **10:30 AM EST** |
| 3-bar (3-2-2, 2-1-2, 3-1-2) | 2 closed + forming | **11:30 AM EST** |

### 15:30 Bar Rule

The last hourly bar (15:30-16:00) is truncated to 30 minutes. Trades entered on this bar **MUST exit before 16:00** - holding overnight exposes you to:
1. Gap risk against your position
2. Extra theta decay (options)
3. Pattern logic breaking across the overnight gap

---

## Critical Entry Rule

**ENTER THE INSTANT PRICE BREAKS TRIGGER - DO NOT WAIT FOR ENTRY BAR TO CLOSE**

- SETUP bar: Must be CLOSED (defines trigger/stop/target levels)
- ENTRY bar: Classified intrabar - enter when trigger breaks

---

## Quick Reference

### Bar Classifications

| Type | Condition | Meaning |
|------|-----------|---------|
| **1** | `H <= H[1] AND L >= L[1]` | Inside/Consolidation |
| **2U** | `H > H[1] AND L >= L[1]` | Bullish direction |
| **2D** | `H <= H[1] AND L < L[1]` | Bearish direction |
| **3** | `H > H[1] AND L < L[1]` | Outside/Expansion |

### Entry Triggers (Inside Bar Patterns)

| Pattern | Entry Trigger | Stop | Target |
|---------|---------------|------|--------|
| 3-1-2D | `L[1] - 0.01` | `H[3]` | `L[3]` |
| 3-1-2U | `H[1] + 0.01` | `L[3]` | `H[3]` |
| 2D-1-2U | `H[1] + 0.01` | `L[2D]` | `H[2D]` |
| 2U-1-2D | `L[1] - 0.01` | `H[2U]` | `L[2U]` |

### Entry Triggers (2-2 and 3-2 Patterns)

| Pattern | Entry Trigger | Stop | Target |
|---------|---------------|------|--------|
| 2D-2U | `H[2D] + 0.01` | `L[2D]` | `H[ref]` |
| 2U-2D | `L[2U] - 0.01` | `H[2U]` | `L[ref]` |
| 3-2D | `L[3] - 0.01` | `H[3]` | Entry - 1.5% |
| 3-2U | `H[3] + 0.01` | `L[3]` | Entry + 1.5% |

**Critical:** 3-2 uses 1.5% measured move. 3-2-2 (reversal) uses traditional magnitude.

**Note on 2-2 Patterns:**

| Pattern Type | Bars | Status | Use |
|--------------|------|--------|-----|
| 2D-2U (Reversal) | Bearish then Bullish | ACTIVE | Entry signal |
| 2U-2D (Reversal) | Bullish then Bearish | ACTIVE | Entry signal |
| 2U-2U (Continuation) | Bullish continuation | FUTURE | Position management |
| 2D-2D (Continuation) | Bearish continuation | FUTURE | Position management |

**Why Continuation is Deferred:**
Continuation patterns (2U-2U, 2D-2D) indicate trend strength but are NOT
new entry signals. They are used for:
- Confirming existing position is working
- Deciding whether to hold through minor pullbacks
- Trailing stop management

They do not provide new entry opportunities because you are already in the trade
if you entered on the original pattern.

### 3-2 Entry Timing Clarification

**Common Confusion:** 3-2 is a 2-bar pattern, not a 3-bar pattern.

```
Bar 1: Type 3 (outside bar) - defines stop at opposite extreme
Bar 2: Type 2U or 2D - direction of the move

Entry: ON THE BREAK when Bar 2 forms
- 3-2U: Enter when price breaks ABOVE Bar 1 (3) high
- 3-2D: Enter when price breaks BELOW Bar 1 (3) low
```

**Do NOT wait for a third bar.** Entry happens as Bar 2 is forming,
the instant price breaks the trigger level.

### Timeframe Continuity (4 C's)

The 4 C's are **diagnostic questions** to evaluate timeframe alignment:

| C | Question | What It Reveals |
|---|----------|-----------------|
| **Control** | Which participation group(s) control current price direction? | Identifies dominant force |
| **Confirm** | Are all participation groups confirming each other's direction? | Checks alignment |
| **Conflict** | Are any participation groups in conflict? | Identifies divergence |
| **Change** | Are any groups changing the continuity or direction of others? | Spots transitions |

**IMPORTANT:** The 4 C's are analytical questions, NOT position sizing rules or pattern categories.

### TFC Scoring

| Bar Type | Counts? | Direction Determined By |
|----------|---------|------------------------|
| Type 1 (Inside) | NO - indecision | N/A |
| Type 2U | YES | Bullish (broke high) |
| Type 2D | YES | Bearish (broke low) |
| Type 3 (Outside) | YES | Green = Bullish, Red = Bearish |

**Green/Red Candle Effect:**
- For Type 2: Conviction modifier (2U+Green = strong, 2U+Red = weak/conflicted)
- For Type 3: Direction determination (Green = bullish, Red = bearish)

---

## File Navigation

### Pattern Detection -> [PATTERNS.md](PATTERNS.md)

**When to read:** Implementing bar classification or pattern detection

**Contains:**
- Bar classification logic with exact operators
- 2-1-2, 3-1-2, 2-2 Reversal, 3-2 pattern definitions
- Edge cases (equal highs/lows)
- VectorBT Pro classifier implementation

### Entry/Exit Mechanics -> [references/ENTRY_MECHANICS.md](references/ENTRY_MECHANICS.md)

**When to read:** Building entry triggers, stops, and targets

**Contains:**
- Precise entry trigger formulas for all patterns
- Stop loss and target calculations
- 2-2 reversal reference bar validation
- 1.5% vs traditional magnitude rules
- Gap handling for daily+ timeframes
- VectorBT Pro entry detector implementation

### Timeframe Analysis -> [TIMEFRAMES.md](TIMEFRAMES.md)

**When to read:** Implementing multi-timeframe analysis

**Contains:**
- 4 C's framework implementation
- MOAF (Mother of All Flips) detection
- Timeframe relationships (Daily/60min/15min)
- Continuity scoring and trade quality matrix

### Position Management -> [EXECUTION.md](EXECUTION.md)

**When to read:** Managing positions after entry

**Contains:**
- 4-level entry priority system
- Position scaling and adding rules
- Break-even management
- Target management (50% at T1, 30% at T2, 20% runner)
- Trailing stops and profit rolling
- Pattern invalidation by Type 3 (exit priority)

### Options Integration -> [OPTIONS.md](OPTIONS.md)

**When to read:** Integrating options with STRAT patterns

**Contains:**
- Strike selection based on entry-to-target range
- Delta requirements (0.50-0.80)
- Expiration selection
- Position sizing for options

### Implementation Bugs -> [IMPLEMENTATION-BUGS.md](IMPLEMENTATION-BUGS.md)

**When to read:** Debugging live/closed bar issues, pattern detection bugs (Per STRAT methodology)

**Contains:**
- Live bar vs closed bar detection critical fixes
- 3-bar vs 2-bar pattern distinction
- Stop placement rules and common errors
- Entry trigger mechanics bugs discovered in production

---

## Implementation Workflow

### Step 1: Data Fetching

```python
from utils import fetch_us_stocks

data = fetch_us_stocks(
    'AAPL',
    start='2025-11-01',
    end='2025-11-20',
    timeframe='1d',
    source='alpaca',
    client_config=dict(api_key=key, secret_key=secret, paper=True)
)
# Timezone is America/New_York, weekend dates validated
```

### Step 2: Bar Classification

```python
@njit
def classify_bars_nb(high, low):
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
```

See [PATTERNS.md](PATTERNS.md) Section 1 for full implementation.

### Step 3: Pattern Detection

Scan bar sequence for valid patterns. See [PATTERNS.md](PATTERNS.md) Sections 2-6.

### Step 4: Entry Calculation

Calculate triggers, stops, targets. See [references/ENTRY_MECHANICS.md](references/ENTRY_MECHANICS.md).

**Critical:** Entry trigger = inside bar high/low (for X-1-2 patterns) or previous bar high/low (for 2-2/3-2 patterns), NOT current bar.

### Step 5: Timeframe Analysis

Check 4 C's, identify MOAF. See [TIMEFRAMES.md](TIMEFRAMES.md).

### Step 6: Position Management

Manage stops, targets, trailing. See [EXECUTION.md](EXECUTION.md).

---

## Common Mistakes

### Entry Timing
**Wrong:** Wait for bar to close as 2U/2D before entering
**Correct:** Enter instant price breaks trigger level

### Trigger Calculation (X-1-2 patterns)
**Wrong:** Use current bar high/low as trigger
**Correct:** Use inside bar (bar[1]) high/low as trigger

### Target Calculation (3-2 vs 3-2-2)
**Wrong:** Use 1.5% for all 3-2 variants
**Correct:** 1.5% only for standalone 3-2, traditional magnitude for 3-2-2

### 2-2 Reference Bar
**Wrong:** Accept any bar type before 2-2 pattern
**Correct:** Reference bar must be directional (2U or 2D)

### Bar Classification
**Wrong:** Use >= for high breaks, <= for low breaks
**Correct:** Use > for high breaks, < for low breaks (strict inequality)

---

## Requirements

- **Python 3.8+**
- **VectorBT Pro** (pip install vectorbtpro)
- **NumPy, Pandas**
- OHLC data with correct timezone (America/New_York for US)

---

## Quick Start

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

# Load data
data = vbt.YFData.pull('SPY', start='2024-01-01', end='2024-12-01')
high = data.get('High').values
low = data.get('Low').values

# Classify bars
bars = classify_bars_nb(high, low)

# Detect patterns and entries
# See references/ENTRY_MECHANICS.md for complete detector
```

---

**Remember:** STRAT is NOT discretionary. Follow the rules mechanically.

**Next steps by task:**
- Bar classification -> [PATTERNS.md](PATTERNS.md)
- Entry/exit mechanics -> [references/ENTRY_MECHANICS.md](references/ENTRY_MECHANICS.md)
- Timeframe analysis -> [TIMEFRAMES.md](TIMEFRAMES.md)
- Position management -> [EXECUTION.md](EXECUTION.md)
- Options trading -> [OPTIONS.md](OPTIONS.md)
