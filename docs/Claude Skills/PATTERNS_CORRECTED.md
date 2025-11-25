# STRAT Patterns - Entry Triggers CORRECTED

## Critical Concept: Live Entry (Intrabar)

**STRAT entries are LIVE - you enter the MOMENT a bar becomes directional.**

### "Where Is The Next 2?"

Every bar starts as Type 1 at the open:
- At open: high = low = open price
- Technically "inside" previous bar for that instant
- Bar is Type 1 until it breaks previous high OR low

**You're watching for the bar to become directional (Type 2).**

When current bar breaks inside bar high/low → It becomes 2U/2D → **ENTER IMMEDIATELY**

---

## 3-1-2 Bullish Reversal

### Pattern Structure
- Bar[idx-2]: Type 3 (Outside bar)
- Bar[idx-1]: Type 1 (Inside bar) 
- Bar[idx]: Type 2U (Directional up)

### Entry Mechanics

**CRITICAL: Entry = Inside Bar High Break (LIVE)**

```python
# Pattern detection
if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == 2:
    # Entry trigger = Inside bar high (NOT Bar 3's high)
    trigger = high[idx-1]  # Inside bar high
    stop = low[idx-2]      # Outside bar low
    target = high[idx-2]   # Outside bar high (magnitude)
    entry_type = "LIVE_INTRABAR"  # Enter moment bar becomes 2U
```

### Real-Time Entry Example

**Setup:**
- Bar 1 (Outside): High=$102, Low=$95
- Bar 2 (Inside): High=$100, Low=$97 (closes)

**Bar 3 (Current Bar) - Live Action:**
```
Time    Price   Bar Status   Action
------  ------  -----------  ------------------
09:30   $98.00  Type 1       Bar opens
09:31   $99.50  Type 1       Still inside
09:32   $100.01 Type 2U      ✅ ENTER NOW!
09:35   $101.50 Type 2U      Already in position
09:40   $102.50 Type 2U      Bar closes
```

**Entry occurred at $100.01** - the MOMENT Bar 3 broke inside bar high.

**NOT at $102.50** (Bar 3's closing high).

### Entry Trigger Breakdown

| Element | Value | Explanation |
|---------|-------|-------------|
| **Pattern** | 3-1-2U | Outside → Inside → 2U |
| **Inside Bar High** | $100.00 | Bar 2's high |
| **Entry Trigger** | $100.01 | Inside bar high + $0.01 |
| **Entry Type** | LIVE | Intrabar, not bar close |
| **Stop Loss** | $95.00 | Outside bar low (Bar 1) |
| **Target** | $102.00 | Outside bar high (Bar 1) |

### Code Implementation

```python
def detect_312_bullish_live(high, low, bars, idx):
    """
    Detect 3-1-2 bullish with LIVE entry trigger
    
    Entry = Inside bar high (live intrabar entry)
    NOT Bar 3's high (that would be late entry)
    """
    # Pattern structure check
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == 2:
        return {
            'pattern': '3-1-2-bull',
            'trigger': high[idx-1],      # Inside bar high (LIVE)
            'stop': low[idx-2],          # Outside bar low
            'target': high[idx-2],       # Outside bar high
            'entry_type': 'LIVE_INTRABAR',
            'entry_bar': idx,            # Current bar (Bar 3)
            'inside_bar_high': high[idx-1],
            'explanation': 'Enter when Bar 3 breaks above inside bar high'
        }
    return None
```

### VectorBT Pro Implementation Note

**For live entry in backtesting:**
```python
# Use inside bar high as entry level
entries = (
    (bar_type == 2) &                    # Current bar is 2U
    (bar_type.shift(1) == 1) &           # Previous bar is inside
    (bar_type.shift(2) == 3) &           # 2 bars ago is outside
    (high > high.shift(1))               # Broke inside bar high
)

# Entry price = inside bar high + slippage
entry_price = high.shift(1) + 0.01  # Inside bar high
```

---

## 3-1-2 Bearish Reversal

### Pattern Structure
- Bar[idx-2]: Type 3 (Outside bar)
- Bar[idx-1]: Type 1 (Inside bar)
- Bar[idx]: Type 2D (Directional down)

### Entry Mechanics

**Entry = Inside Bar Low Break (LIVE)**

```python
if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == -2:
    trigger = low[idx-1]   # Inside bar low (LIVE entry)
    stop = high[idx-2]     # Outside bar high
    target = low[idx-2]    # Outside bar low (magnitude)
    entry_type = "LIVE_INTRABAR"
```

### Real-Time Entry Example

**Setup:**
- Bar 1 (Outside): High=$105, Low=$98
- Bar 2 (Inside): High=$103, Low=$100 (closes)

**Bar 3 Live:**
```
Time    Price   Bar Status   Action
------  ------  -----------  ------------------
09:30   $102.00 Type 1       Bar opens
09:31   $100.50 Type 1       Still inside
09:32   $99.99  Type 2D      ✅ ENTER SHORT NOW!
09:35   $98.50  Type 2D      Already short
09:40   $97.00  Type 2D      Bar closes
```

**Entry at $99.99** - moment Bar 3 broke inside bar low.

---

## 2-1-2 Bullish Reversal

### Pattern Structure
- Bar[idx-2]: Type 2D (Directional down)
- Bar[idx-1]: Type 1 (Inside bar)
- Bar[idx]: Type 2U (Directional up)

### Entry Mechanics

**Entry = Inside Bar High Break (LIVE)**

```python
if bars[idx-2] == -2 and bars[idx-1] == 1 and bars[idx] == 2:
    trigger = high[idx-1]  # Inside bar high (LIVE)
    stop = low[idx-2]      # First 2D bar low
    target = high[idx-2]   # First 2D bar high
    entry_type = "LIVE_INTRABAR"
```

### Key Difference from 3-1-2

**2-1-2 has TIGHTER magnitude:**
- 3-1-2 target = Outside bar extreme (larger range)
- 2-1-2 target = First 2D bar extreme (smaller range)

**Example:**
- Bar 1 (2D): High=$100, Low=$97
- Bar 2 (Inside): High=$99, Low=$98
- Bar 3 breaks $99.01 → Enter
- **Target: $100** (Bar 1 high, not some larger range)

---

## 2-1-2 Bearish Reversal

### Pattern Structure
- Bar[idx-2]: Type 2U (Directional up)
- Bar[idx-1]: Type 1 (Inside bar)
- Bar[idx]: Type 2D (Directional down)

### Entry Mechanics

```python
if bars[idx-2] == 2 and bars[idx-1] == 1 and bars[idx] == -2:
    trigger = low[idx-1]   # Inside bar low (LIVE)
    stop = high[idx-2]     # First 2U bar high
    target = low[idx-2]    # First 2U bar low
    entry_type = "LIVE_INTRABAR"
```

---

## Rev Strat Patterns (2-Bar Reversal)

### 1-3-2U Rev Strat (Bullish)

**Pattern:**
- Bar[idx-3]: Any (often Type 1 or 2D)
- Bar[idx-2]: Type 3 (Outside)
- Bar[idx-1]: Type 2D (Failed to reach magnitude)
- Bar[idx]: Type 2U (Reversal)

**Entry = High of Bar[idx-1] (the failed 2D bar)**

```python
if (bars[idx-3] in [1, -2] and 
    bars[idx-2] == 3 and 
    bars[idx-1] == -2 and 
    bars[idx] == 2):
    
    # Entry = High of the 2D bar that failed
    trigger = high[idx-1]  # NOT inside bar (there isn't one)
    stop = low[idx-1]      # Low of failed 2D
    target = high[idx-2]   # Outside bar high
    entry_type = "LIVE_INTRABAR"
```

**Key Difference:**
- No inside bar in Rev Strat
- Entry = High of the bar that's being reversed (the failed 2D)
- "Where is next 2?" = Bar reversing the failed breakout

### 1-3-2D Rev Strat (Bearish)

```python
if (bars[idx-3] in [1, 2] and 
    bars[idx-2] == 3 and 
    bars[idx-1] == 2 and 
    bars[idx] == -2):
    
    trigger = low[idx-1]   # Low of failed 2U bar
    stop = high[idx-1]     # High of failed 2U
    target = low[idx-2]    # Outside bar low
    entry_type = "LIVE_INTRABAR"
```

---

## 2-2 Continuation (Not Reversal)

### 2U-2U Bullish Continuation

**Pattern:**
- Bar[idx-1]: Type 2U
- Bar[idx]: Type 2U (momentum continues)

**Entry = High of first 2U bar**

```python
if bars[idx-1] == 2 and bars[idx] == 2:
    trigger = high[idx-1]  # First 2U bar high
    stop = low[idx-1]      # First 2U bar low
    target = None          # Use trailing stop
    entry_type = "LIVE_INTRABAR"
```

**Note:** No inside bar, direct momentum entry

---

## Summary: Entry Trigger Rules

| Pattern | Entry Trigger | Bar Breaking | Entry Type |
|---------|---------------|--------------|------------|
| **3-1-2 Bull** | Inside bar high | Bar 3 → 2U | LIVE |
| **3-1-2 Bear** | Inside bar low | Bar 3 → 2D | LIVE |
| **2-1-2 Bull** | Inside bar high | Bar 3 → 2U | LIVE |
| **2-1-2 Bear** | Inside bar low | Bar 3 → 2D | LIVE |
| **Rev Strat Bull** | Failed 2D high | Bar 4 → 2U | LIVE |
| **Rev Strat Bear** | Failed 2U low | Bar 4 → 2D | LIVE |
| **2-2 Bull** | First 2U high | Bar 2 → 2U | LIVE |
| **2-2 Bear** | First 2D low | Bar 2 → 2D | LIVE |

---

## Common Mistakes to Avoid

### ❌ WRONG: Waiting for Bar to Close
```python
# This waits for Bar 3 to close
trigger = high[idx]  # Bar 3's high (TOO LATE)
```

### ✅ CORRECT: Live Entry
```python
# Enter the moment inside bar breaks
trigger = high[idx-1]  # Inside bar high (LIVE)
```

### ❌ WRONG: Using Final Bar's High
**Example:**
- Inside bar: $100
- Bar 3 closes: $103
- Entry trigger = $103 ← **WRONG**

You'd be entering 3 points late!

### ✅ CORRECT: Using Inside Bar High
**Example:**
- Inside bar: $100  
- Entry trigger = $100.01 ← **CORRECT**
- Bar 3 might close at $103, but you entered at $100.01

---

## Why This Matters for Backtesting

**Wrong entry (Bar 3 high):**
```python
Entry: $103
Target: $105
Profit: $2
Win rate: 50%
```

**Correct entry (Inside bar high):**
```python
Entry: $100
Target: $105
Profit: $5
Win rate: 65% (better entry timing)
```

**Using wrong trigger systematically underperforms because:**
- Enters too late (3-5% of move missed)
- Worse risk/reward
- More stopped out before profit
- Backtests won't match real performance

---

## Implementation Checklist

For VectorBT Pro or any backtesting engine:

- [ ] Entry trigger = Inside bar extreme (for patterns with inside bars)
- [ ] Entry trigger = Previous bar extreme (for Rev Strats)
- [ ] Entry type = LIVE_INTRABAR (not bar close)
- [ ] Pattern detection checks idx, idx-1, idx-2 correctly
- [ ] Trigger references idx-1 (inside bar), not idx (current bar)
- [ ] Code validated against manual chart analysis
- [ ] Test cases confirm live entry timing

---

**Last Updated:** Session 8+ (Entry Trigger Correction)
**Status:** CORRECTED - Ready for Implementation
