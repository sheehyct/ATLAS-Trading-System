# STRAT Patterns - Bar Classification & Pattern Detection

**Purpose:** Detailed implementation guide for bar classification and STRAT pattern detection  
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [Bar Classification Logic](#1-bar-classification-logic)
2. [2-1-2 Patterns](#2-2-1-2-patterns)
3. [3-1-2 Patterns](#3-3-1-2-patterns)
4. [2-2 Patterns](#4-2-2-patterns)
5. [Rev Strat Patterns](#5-rev-strat-patterns)
6. [Pattern Variations](#6-pattern-variations)
7. [Invalid Patterns](#7-invalid-patterns)
8. [Mother Bar Identification](#8-mother-bar-identification)

---

## 1. Bar Classification Logic

### Core Classification Function

**ThinkScript Reference:**
```thinkscript
def insidebar = (H < H[1] and L > L[1]) or (H == H[1] and L > L[1]) or 
                (H < H[1] and L == L[1]) or (H == H[1] and L == L[1]);
def outsidebar = H > H[1] and L < L[1];
def twoup = H > H[1] and L >= L[1];
def twodown = H <= H[1] and L < L[1];
```

**Python Implementation:**
```python
def classify_bar(current_high, current_low, prev_high, prev_low):
    """
    Classify bar as Type 1, 2U, 2D, or 3.
    
    Args:
        current_high: High of current bar
        current_low: Low of current bar
        prev_high: High of previous bar
        prev_low: Low of previous bar
    
    Returns:
        1: Inside bar (consolidation)
        2: 2U (bullish directional)
       -2: 2D (bearish directional)
        3: Outside bar (expansion)
        0: Unclassifiable (first bar)
    """
    broke_high = current_high > prev_high
    broke_low = current_low < prev_low
    
    if broke_high and broke_low:
        return 3  # Outside bar
    elif broke_high:
        return 2  # 2U
    elif broke_low:
        return -2  # 2D
    else:
        return 1  # Inside bar
```

### Operator Precision

**Critical:** Use exact operators as specified. Incorrect operators lead to misclassification.

| Condition | Operator | Rationale |
|-----------|----------|-----------|
| Inside high | `H <= H[1]` | Equal high still counts as inside |
| Inside low | `L >= L[1]` | Equal low still counts as inside |
| 2U high | `H > H[1]` | Must strictly break prior high |
| 2U low | `L >= L[1]` | Can equal or be higher than prior low |
| 2D high | `H <= H[1]` | Can equal or be lower than prior high |
| 2D low | `L < L[1]` | Must strictly break prior low |
| Outside high | `H > H[1]` | Must exceed both bounds |
| Outside low | `L < L[1]` | Must exceed both bounds |

### Edge Cases

**Equal High + Higher Low = Inside (1)**
```
Bar 1: High=100, Low=95
Bar 2: High=100, Low=96
Classification: 1 (inside)
Reason: H <= H[1] (100 <= 100) AND L >= L[1] (96 >= 95)
```

**Lower High + Equal Low = Inside (1)**
```
Bar 1: High=100, Low=95
Bar 2: High=99, Low=95
Classification: 1 (inside)
Reason: H <= H[1] (99 <= 100) AND L >= L[1] (95 >= 95)
```

**Equal High + Equal Low = Inside (1)**
```
Bar 1: High=100, Low=95
Bar 2: High=100, Low=95
Classification: 1 (inside)
Reason: H <= H[1] (100 <= 100) AND L >= L[1] (95 >= 95)
```

**Higher High + Equal Low = 2U (2)**
```
Bar 1: High=100, Low=95
Bar 2: High=101, Low=95
Classification: 2 (2U)
Reason: H > H[1] (101 > 100) AND L >= L[1] (95 >= 95)
```

**Equal High + Lower Low = 2D (-2)**
```
Bar 1: High=100, Low=95
Bar 2: High=100, Low=94
Classification: -2 (2D)
Reason: H <= H[1] (100 <= 100) AND L < L[1] (94 < 95)
```

### VectorBT Pro Implementation

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

@njit
def classify_bars_nb(high, low):
    """
    Numba-compiled bar classifier for performance.
    
    Args:
        high: NumPy array of highs
        low: NumPy array of lows
    
    Returns:
        NumPy array of bar types: 0 (unclassifiable), 1 (inside), 
        2 (2U), -2 (2D), 3 (outside)
    """
    n = len(high)
    bars = np.zeros(n, dtype=np.int8)
    
    # First bar is unclassifiable
    bars[0] = 0
    
    for i in range(1, n):
        broke_high = high[i] > high[i-1]
        broke_low = low[i] < low[i-1]
        
        if broke_high and broke_low:
            bars[i] = 3  # Outside
        elif broke_high:
            bars[i] = 2  # 2U
        elif broke_low:
            bars[i] = -2  # 2D
        else:
            bars[i] = 1  # Inside
    
    return bars

# Usage
high = data.get('High').values
low = data.get('Low').values
bar_types = classify_bars_nb(high, low)
```

### Bar Classification Display Functions

**Human-Readable Output:**

The `strat` module provides display functions to format bar classifications for analysis:

```python
from strat import (
    classify_bars,
    format_bar_classifications,
    get_bar_sequence_string
)

# Classify bars
classifications = classify_bars(data['high'], data['low'])

# Format as list of strings
labels = format_bar_classifications(classifications, skip_reference=True)
# Output: ['2U', '1', '2U', '3']

# Format as comma-separated string (oldest to newest)
sequence = get_bar_sequence_string(classifications)
# Output: '2U, 1, 2U, 3'
```

**Use Cases:**
- **Pattern Verification:** Display last 10 bars to verify pattern structure
- **Debugging:** Compare classifications to TradingView STRAT indicator
- **Analysis:** Export bar sequences for pattern frequency analysis
- **Reporting:** Include bar sequence in trade signals/alerts

**Example - Last 10 Days Bar Sequence:**
```python
# Get last 10 days of classifications
recent_bars = get_bar_sequence_string(classifications[-10:])
print(f"Last 10 bars: {recent_bars}")
# Output: "2U, 2D, 1, 2U, 3, 2D, 2D, 1, 2U, 2U"
```

### Common Classification Errors

**ERROR 1: Using close price for classification**
```python
# ❌ WRONG
if close[i] > close[i-1]:
    return 2  # Incorrect - ignores high/low

# ✅ CORRECT
if high[i] > high[i-1] and low[i] >= low[i-1]:
    return 2  # Correct - uses high/low only
```

**ERROR 2: Using `>=` for 2U high**
```python
# ❌ WRONG
if high[i] >= high[i-1]:  # Equal high should be inside
    return 2

# ✅ CORRECT
if high[i] > high[i-1]:  # Must strictly break
    return 2
```

**ERROR 3: Ignoring first bar**
```python
# ❌ WRONG
bars = classify_all_bars()  # Assumes all bars classifiable

# ✅ CORRECT
bars[0] = 0  # Mark first bar as unclassifiable
for i in range(1, len(bars)):
    bars[i] = classify_bar(i)
```

---

## 2. 2-1-2 Patterns

### CRITICAL: Direction Determination

**The EXIT bar (last bar) determines trade direction, NOT the first bar.**

| Pattern | Exit Bar | Direction | Option |
|---------|----------|-----------|--------|
| 2U-1-2U | 2U | Bullish | CALL |
| 2D-1-2D | 2D | Bearish | PUT |
| 2D-1-2U | 2U | Bullish | CALL |
| **2U-1-2D** | **2D** | **Bearish** | **PUT** |

**WARNING:** `2U-1-2D` is BEARISH despite starting with 2U!

### Pattern Structure

**2U-1-2U (Bullish Continuation):**
```
Bar 1: 2U (breaks high, low >= prior) - SETUP bar
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2U (breaks high again) - EXIT bar determines direction
Trigger: High of Bar 3
Direction: BULLISH (CALL)
```

**2D-1-2D (Bearish Continuation):**
```
Bar 1: 2D (breaks low, high <= prior) - SETUP bar
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2D (breaks low again) - EXIT bar determines direction
Trigger: Low of Bar 3
Direction: BEARISH (PUT)
```

**2D-1-2U (Bullish Reversal):**
```
Bar 1: 2D (breaks low) - SETUP bar
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2U (breaks high) - EXIT bar determines direction
Trigger: High of Bar 3
Direction: BULLISH (CALL)
```

**2U-1-2D (Bearish Reversal):**
```
Bar 1: 2U (breaks high) - SETUP bar
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2D (breaks low) - EXIT bar determines direction
Trigger: Low of Bar 3
Direction: BEARISH (PUT)
```

### Detection Logic

```python
@njit
def detect_212_bull(bars, high, idx):
    """
    Detect bullish 2-1-2 pattern.
    
    Returns:
        (is_valid, trigger_price, stop_price)
    """
    if idx < 2:
        return False, 0.0, 0.0
    
    # Check pattern: 2U → 1 → 2U
    if bars[idx-2] == 2 and bars[idx-1] == 1 and bars[idx] == 2:
        trigger = high[idx]  # Trigger = high of final 2U
        stop = low[idx-2]    # Stop = low of first 2U
        return True, trigger, stop
    
    return False, 0.0, 0.0

@njit
def detect_212_bear(bars, low, high, idx):
    """
    Detect bearish 2-1-2 pattern.
    
    Returns:
        (is_valid, trigger_price, stop_price)
    """
    if idx < 2:
        return False, 0.0, 0.0
    
    # Check pattern: 2D → 1 → 2D
    if bars[idx-2] == -2 and bars[idx-1] == 1 and bars[idx] == -2:
        trigger = low[idx]   # Trigger = low of final 2D
        stop = high[idx-2]   # Stop = high of first 2D
        return True, trigger, stop
    
    return False, 0.0, 0.0
```

### Target Scenarios

**Bullish 2-1-2 Targets:**
```
T1 (Conservative): 1R (risk distance)
T2 (Base): 2R
T3 (Extension): 3R+ or next resistance
```

**Example:**
```
Entry: $100
Stop: $98 (2U low)
Risk: $2

T1: $102 (1R = $2 profit)
T2: $104 (2R = $4 profit)
T3: $106+ (3R = $6+ profit)
```

### Pattern Invalidation

**2-1-2 is invalidated if:**
1. Inside bar (Bar 2) breaks in opposite direction first
2. Stop is hit before trigger
3. Pattern takes >5 bars to complete
4. Lower timeframe shows opposite pattern forming

### VectorBT Pro Backtesting

```python
def backtest_212_pattern(data):
    """
    Backtest 2-1-2 bull and bear patterns.
    """
    high = data.get('High').values
    low = data.get('Low').values
    close = data.get('Close').values
    
    # Classify bars
    bars = classify_bars_nb(high, low)
    
    # Initialize signals
    entries = np.zeros(len(data), dtype=bool)
    exits = np.zeros(len(data), dtype=bool)
    stops = np.full(len(data), np.nan)
    
    for i in range(2, len(data)):
        # Bullish 2-1-2
        is_bull, trigger, stop = detect_212_bull(bars, high, i)
        if is_bull and close[i] > trigger:
            entries[i] = True
            stops[i] = stop
    
    # Create portfolio
    pf = vbt.Portfolio.from_signals(
        close, 
        entries, 
        exits,
        sl_stop=stops,
        freq='1D'
    )
    
    return pf
```

---

## 3. 3-1-2 Patterns

### Pattern Structure

**3-1-2U (Bullish):**
```
Bar 1: 3 (outside bar - expansion)
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2U (breaks high) - EXIT bar determines direction
Trigger: High of 2U bar
Direction: BULLISH (CALL)
```

**3-1-2D (Bearish):**
```
Bar 1: 3 (outside bar - expansion)
Bar 2: 1 (inside bar - consolidation)
Bar 3: 2D (breaks low) - EXIT bar determines direction
Trigger: Low of 2D bar
Direction: BEARISH (PUT)
```

### Detection Logic

```python
@njit
def detect_312_bull(bars, high, low, idx):
    """
    Detect bullish 3-1-2 pattern.
    
    Returns:
        (is_valid, trigger_price, stop_price)
    """
    if idx < 2:
        return False, 0.0, 0.0
    
    # Check pattern: 3 → 1 → 2U
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == 2:
        trigger = high[idx]    # Trigger = high of 2U
        stop = low[idx-2]      # Stop = low of outside bar
        return True, trigger, stop
    
    return False, 0.0, 0.0

@njit
def detect_312_bear(bars, low, high, idx):
    """
    Detect bearish 3-1-2 pattern.
    
    Returns:
        (is_valid, trigger_price, stop_price)
    """
    if idx < 2:
        return False, 0.0, 0.0
    
    # Check pattern: 3 → 1 → 2D
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == -2:
        trigger = low[idx]     # Trigger = low of 2D
        stop = high[idx-2]     # Stop = high of outside bar
        return True, trigger, stop
    
    return False, 0.0, 0.0
```

### Comparison to 2-1-2

| Feature | 2-1-2 | 3-1-2 |
|---------|-------|-------|
| **Setup bar** | 2U or 2D | 3 (outside) |
| **Stop distance** | Smaller | Larger |
| **Win rate** | Higher | Lower |
| **R:R ratio** | 2:1 - 3:1 | 3:1 - 5:1 |
| **Frequency** | More common | Less common |
| **Reliability** | More consistent | More volatile |

**3-1-2 Advantages:**
- Larger potential target
- Captures volatility expansion
- Often marks significant reversals

**3-1-2 Disadvantages:**
- Wider stop (higher risk)
- Less frequent setups
- More whipsaw potential

---

## 4. 2-2 Patterns

### Pattern Structure

**2U-2U (Bullish Continuation):**
```
Bar 1: 2U (initial break)
Bar 2: 2U (continuation) - EXIT bar determines direction
Trigger: High of Bar 2
Direction: BULLISH (CALL)
```

**2D-2D (Bearish Continuation):**
```
Bar 1: 2D (initial break)
Bar 2: 2D (continuation) - EXIT bar determines direction
Trigger: Low of Bar 2
Direction: BEARISH (PUT)
```

**2D-2U (Bullish Reversal):**
```
Bar 1: 2D (initial down)
Bar 2: 2U (reversal up) - EXIT bar determines direction
Trigger: High of Bar 2
Direction: BULLISH (CALL)
```

**2U-2D (Bearish Reversal):**
```
Bar 1: 2U (initial up)
Bar 2: 2D (reversal down) - EXIT bar determines direction
Trigger: Low of Bar 2
Direction: BEARISH (PUT)
```

### Detection Logic

```python
@njit
def detect_22_bull(bars, high, low, idx):
    """
    Detect bullish 2-2 continuation pattern.
    """
    if idx < 1:
        return False, 0.0, 0.0
    
    # Check pattern: 2U → 2U
    if bars[idx-1] == 2 and bars[idx] == 2:
        trigger = high[idx]    # Trigger = high of second 2U
        stop = low[idx-1]      # Stop = low of first 2U
        return True, trigger, stop
    
    return False, 0.0, 0.0

@njit
def detect_22_bear(bars, low, high, idx):
    """
    Detect bearish 2-2 continuation pattern.
    """
    if idx < 1:
        return False, 0.0, 0.0
    
    # Check pattern: 2D → 2D
    if bars[idx-1] == -2 and bars[idx] == -2:
        trigger = low[idx]     # Trigger = low of second 2D
        stop = high[idx-1]     # Stop = high of first 2D
        return True, trigger, stop
    
    return False, 0.0, 0.0
```

### Continuation vs Reversal

**2-2 Patterns are CONTINUATION patterns:**
- They require existing trend
- Trade in direction of momentum
- Typically occur mid-trend

**Do NOT confuse with 2-1-2:**
- 2-1-2 = Reversal (has inside bar)
- 2-2 = Continuation (no inside bar)

### Context Requirements

**Valid 2-2 requires:**
1. Prior trend in same direction (3+ bars minimum)
2. No major resistance/support nearby
3. Higher timeframe confirmation
4. Volume expansion on second 2U/2D

**Invalid 2-2 setups:**
- First bar after long consolidation
- Against higher timeframe trend
- At major support/resistance
- Weak volume on continuation

---

## 5. Rev Strat Patterns

### Pattern Structure

**Rev Strat** = Extreme reversal after extended move

**Bullish Rev Strat:**
```
Context: Extended downtrend (5+ 2D bars)
Pattern: 2D → 3 → 2U → 2U
Signal: Rapid reversal with expansion
```

**Bearish Rev Strat:**
```
Context: Extended uptrend (5+ 2U bars)
Pattern: 2U → 3 → 2D → 2D
Signal: Rapid reversal with expansion
```

### Detection Logic

```python
@njit
def detect_rev_strat_bull(bars, high, low, idx):
    """
    Detect bullish Rev Strat exhaustion pattern.
    """
    if idx < 5:
        return False, 0.0, 0.0
    
    # Check for extended downtrend
    downtrend_bars = 0
    for i in range(idx-5, idx):
        if bars[i] == -2:
            downtrend_bars += 1
    
    if downtrend_bars < 3:
        return False, 0.0, 0.0
    
    # Check reversal: 2D → 3 → 2U
    if bars[idx-2] == -2 and bars[idx-1] == 3 and bars[idx] == 2:
        trigger = high[idx]
        stop = low[idx-2]
        return True, trigger, stop
    
    return False, 0.0, 0.0
```

### Rev Strat Characteristics

**Key features:**
- Occurs after exhaustion (5+ bars in one direction)
- Often includes outside bar (Type 3)
- Large range bars
- High volume
- Emotional extremes (panic/euphoria)

**Target expectations:**
- T1: Retest of prior support/resistance
- T2: 50% retracement of trend
- T3: Full reversal to trend start

---

## 6. Pattern Variations

### Multiple Inside Bars

**Pattern:** 2U → 1 → 1 → 2U

```python
@njit
def detect_multi_inside_212(bars, high, low, idx):
    """
    Detect 2-1-2 with multiple inside bars.
    Max 3 inside bars allowed.
    """
    if idx < 3:
        return False, 0.0, 0.0
    
    # Check for 2U ... (1+) ... 2U pattern
    if bars[idx] != 2:
        return False, 0.0, 0.0
    
    inside_count = 0
    setup_bar_idx = -1
    
    for i in range(idx-1, max(idx-4, 0), -1):
        if bars[i] == 1:
            inside_count += 1
        elif bars[i] == 2:
            setup_bar_idx = i
            break
        else:
            return False, 0.0, 0.0
    
    if inside_count > 0 and inside_count <= 3 and setup_bar_idx != -1:
        trigger = high[idx]
        stop = low[setup_bar_idx]
        return True, trigger, stop
    
    return False, 0.0, 0.0
```

**Rules:**
- Maximum 3 consecutive inside bars
- Each additional inside bar reduces pattern strength
- Consider lower position size for 3+ inside bars

---

## 7. Invalid Patterns

### What NOT to Trade

**Invalid Pattern 1: 1-2-2**
```
Bar 1: 1 (inside)
Bar 2: 2U
Bar 3: 2U
Problem: No setup bar (missing initial 2U or 3)
```

**Invalid Pattern 2: 2U-2D-2U**
```
Bar 1: 2U
Bar 2: 2D (invalidation)
Bar 3: 2U
Problem: Direction change invalidates setup
```

**Invalid Pattern 3: Pattern at limit**
```
2-1-2 pattern forms at prior high/low
Problem: No room for target
```

**Invalid Pattern 4: Against higher TF**
```
Daily: Bearish trend
15min: Bullish 2-1-2
Problem: Trades against higher timeframe
```

### Invalidation Rules

**A pattern is invalidated when:**
1. Inside bar breaks opposite direction first
2. Stop is hit before trigger
3. Higher timeframe contradicts
4. Major S/R prevents target
5. Volume too low

---

## 8. Mother Bar Identification

### Definition

**Mother Bar** = Bar that contains multiple smaller bars on lower timeframe

**Example:**
```
Daily bar (Mother):
High: $100
Low: $95

15min bars inside:
Bar 1: $96-$97
Bar 2: $97-$98
Bar 3: $97.50-$98.50
All contained within $95-$100 range
```

### Detection Logic

```python
def identify_mother_bars(high_htf, low_htf, high_ltf, low_ltf, ratio=4):
    """
    Identify mother bars across timeframes.
    
    Args:
        high_htf: Higher timeframe highs
        low_htf: Higher timeframe lows
        high_ltf: Lower timeframe highs
        low_ltf: Lower timeframe lows
        ratio: Timeframe ratio (e.g., 4 for daily:4h)
    
    Returns:
        Boolean array marking mother bars
    """
    n_htf = len(high_htf)
    n_ltf = len(high_ltf)
    
    is_mother = np.zeros(n_htf, dtype=bool)
    
    for i in range(n_htf):
        # Get corresponding LTF bars
        ltf_start = i * ratio
        ltf_end = min(ltf_start + ratio, n_ltf)
        
        if ltf_end - ltf_start < ratio:
            continue
        
        # Check if all LTF bars are inside HTF bar
        ltf_high_max = np.max(high_ltf[ltf_start:ltf_end])
        ltf_low_min = np.min(low_ltf[ltf_start:ltf_end])
        
        if ltf_high_max <= high_htf[i] and ltf_low_min >= low_htf[i]:
            is_mother[i] = True
    
    return is_mother
```

### Trading Mother Bars

**Strategy:**
1. Identify mother bar on HTF
2. Wait for LTF pattern inside mother bar
3. Trade LTF pattern with HTF mother bar as stop

**Example:**
```
Daily mother bar: $95-$100
15min inside: 2-1-2 bull at $97
Entry: $98 (15min trigger)
Stop: $95 (daily low)
Target: $103 (5R based on daily range)
```

---

## Summary

**Pattern Priority:**
1. **2-1-2** - Most reliable reversal setup
2. **3-1-2** - Higher R:R reversal with wider stop
3. **2-2** - Continuation in strong trends
4. **Rev Strat** - Exhaustion/climax reversals
5. **Multiple inside** - Weaker setup, reduce size

**Key Takeaways:**
- Bar classification MUST be exact (correct operators)
- Trigger = High of 2U (bull) or Low of 2D (bear)
- Stop = Low of setup bar (bull) or High of setup bar (bear)
- Pattern validity requires proper context
- Higher timeframe confirmation improves win rate

**Next Steps:**
- For timeframe analysis → Read [TIMEFRAMES.md](TIMEFRAMES.md)
- For entry/exit mechanics → Read [EXECUTION.md](EXECUTION.md)
- For options integration → Read [OPTIONS.md](OPTIONS.md)
