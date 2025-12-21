---
name: strat-methodology
description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics (triggers, stops, targets), position management (break-even, trailing stops, rolling profits), and options integration (strike selection, Greeks, cheap options strategy). Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns. Requires VectorBT Pro for backtesting implementation.
---

# STRAT Methodology Implementation

**Version:** 2.1
**Purpose:** Navigation hub for STRAT trading system implementation

---

## Critical Entry Rule

**ENTER THE INSTANT PRICE BREAKS TRIGGER - DO NOT WAIT FOR BAR CLOSE**

Pattern detection happens at bar close. Entry happens intrabar when trigger breaks. This is fundamental to STRAT and the most common implementation error.

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

### Timeframe Continuity (4 C's)

1. **Combo** - Multiple timeframes show same setup
2. **Confirm** - Lower TF confirms higher TF trigger
3. **Continue** - Multiple setups in same direction
4. **Consolidate** - Controlled pullback before continuation

---

## File Navigation

### Pattern Detection -> [PATTERNS.md](PATTERNS.md)

**When to read:** Implementing bar classification or pattern detection

**Contains:**
- Bar classification logic with exact operators
- 2-1-2, 3-1-2, 2-2, Rev Strat pattern definitions
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

### Options Integration -> [OPTIONS.md](OPTIONS.md)

**When to read:** Integrating options with STRAT patterns

**Contains:**
- Strike selection based on entry-to-target range
- Delta requirements (0.50-0.80)
- Expiration selection
- Position sizing for options

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
