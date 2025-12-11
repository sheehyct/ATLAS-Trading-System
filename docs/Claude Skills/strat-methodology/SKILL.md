---
name: strat-methodology
description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics (triggers, stops, targets), position management (break-even, trailing stops, rolling profits), and options integration (strike selection, Greeks, cheap options strategy). Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns. Requires VectorBT Pro for backtesting implementation.
---

# STRAT Methodology Implementation

**Version:** 2.0  
**Last Updated:** November 2025  
**Purpose:** Navigation hub for STRAT trading system implementation

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [When to Use This Skill](#when-to-use-this-skill)
3. [Core Components](#core-components)
4. [Implementation Workflow](#implementation-workflow)
5. [File Navigation](#file-navigation)
6. [Requirements](#requirements)

---

## Quick Reference

### Bar Type Classifications

| Type | Name | Condition | Meaning |
|------|------|-----------|---------|
| **1** | Inside | `H <= H[1] AND L >= L[1]` | Consolidation |
| **2U** | Up | `H > H[1] AND L >= L[1]` | Bullish direction |
| **2D** | Down | `H <= H[1] AND L < L[1]` | Bearish direction |
| **3** | Outside | `H > H[1] AND L < L[1]` | Expansion/volatility |

**Critical Rules:**
- Bar color is irrelevant
- First bar cannot be classified (no prior reference)
- Use strict `>` and `<` operators (not `>=` or `<=` for directional moves)
- Equal highs/lows favor inside classification

### Key Patterns

| Pattern | Structure | Trigger | Bias |
|---------|-----------|---------|------|
| **2-1-2 Bull** | 2U → 1 → 2U | Break of 2U high | Bullish |
| **2-1-2 Bear** | 2D → 1 → 2D | Break of 2D low | Bearish |
| **3-1-2 Bull** | 3 → 1 → 2U | Break of 2U high | Bullish |
| **3-1-2 Bear** | 3 → 1 → 2D | Break of 2D low | Bearish |
| **2-2 Bull** | 2U → 2U | Break of first 2U high | Continuation |
| **2-2 Bear** | 2D → 2D | Break of first 2D low | Continuation |

### Timeframe Continuity (4 C's)

1. **Combo** - Multiple timeframes show same setup
2. **Confirm** - Lower TF confirms higher TF trigger
3. **Continue** - Multiple setups in same direction
4. **Consolidate** - Controlled pullback before continuation

### Timeframe Hierarchy

```
Monthly > Weekly > Daily > 60min > 15min > 5min
```

| Timeframe | Role | Typical Hold |
|-----------|------|--------------|
| **Monthly** | Macro bias | Weeks to months |
| **Weekly** | Anchor/Swing | Days to weeks |
| **Daily** | Execution | 1-7 days |
| **60min/15min** | Entry timing | Hours |

### Entry Priorities

| Priority | Entry Type | Condition |
|----------|-----------|-----------|
| **1** | Full Target | Price exceeds target scenario |
| **2** | MOAF | Mother of All Flips triggered |
| **3** | Mechanical | Pattern trigger hit |
| **4** | Discretionary | All scenarios converge |

---

## When to Use This Skill

Use this skill when you need to:

- **Classify bars** - Determine if price action is Type 1, 2U, 2D, or 3
- **Detect patterns** - Identify 2-1-2, 3-1-2, 2-2, or Rev Strat setups
- **Analyze timeframes** - Check for multi-timeframe continuity
- **Calculate entries** - Determine mechanical entry triggers and stops
- **Manage positions** - Implement break-even, targets, and trailing stops
- **Integrate options** - Select strikes based on STRAT patterns
- **Build backtests** - Implement STRAT in VectorBT Pro

**Do NOT use this skill for:**
- General market analysis without STRAT methodology
- Fundamental analysis or earnings-based trading
- High-frequency trading (STRAT is positional/swing)
- Strategies incompatible with multi-timeframe analysis

---

## Core Components

### 1. Bar Classification
The foundation of STRAT is bar typing. Every bar must be classified correctly before pattern detection.

**Critical implementation details:**
- Use exact operator logic (see [PATTERNS.md](PATTERNS.md) Section 1)
- Handle edge cases (equal highs/lows)
- Account for first bar (unclassifiable)
- Ignore bar color/close price

### 2. Pattern Detection
Five primary patterns drive STRAT entries:
- 2-1-2 (reversal)
- 3-1-2 (reversal)
- 2-2 (continuation)
- Rev Strat (extreme reversal)
- Inside bar variations

**Detection logic:**
- Sequential bar type matching
- Trigger level calculation
- Invalidation rules
- See [PATTERNS.md](PATTERNS.md) Sections 2-6

### 3. Timeframe Continuity
Multi-timeframe analysis determines trade quality and position size.

**Key concepts:**
- 4 C's framework (Combo, Confirm, Continue, Consolidate)
- MOAF (Mother of All Flips)
- Institutional vs retail flips
- See [TIMEFRAMES.md](TIMEFRAMES.md)

### 4. Entry/Exit Mechanics
Mechanical rules for trade execution eliminate discretion.

**Entry framework:**
- 4-level priority system
- Trigger calculations
- Stop placement
- Target scenarios
- See [EXECUTION.md](EXECUTION.md) Section 1

### 5. Position Management
Rules-based management from entry to exit.

**Management phases:**
- Initial stop placement
- Break-even moves
- Profit-taking (50% at 1R, 30% at 2R, 20% at 3R+)
- Trailing stops
- Rolling profits to new setups
- See [EXECUTION.md](EXECUTION.md) Section 2

### 6. Options Integration
STRAT patterns provide directional bias and targets for options selection.

**Integration points:**
- Strike selection based on entry-to-target range
- Expected ROI calculation
- Greeks analysis (Delta 0.50-0.80)
- Position sizing (1-2% risk per trade)
- See [OPTIONS.md](OPTIONS.md)

---

## Implementation Workflow

### Step 0: Data Fetching (CRITICAL - MUST BE CORRECT)

**ZERO TOLERANCE:** Market data MUST be fetched with correct year and timezone. Failure causes 0% accuracy with TradingView and invalid trading signals.

```python
from utils import fetch_us_stocks

# CORRECT: Mandatory pattern for US market data
data = fetch_us_stocks(
    'AAPL',
    start='2025-11-01',  # CRITICAL: Correct year!
    end='2025-11-20',
    timeframe='1d',
    source='alpaca',  # or 'tiingo'
    client_config=dict(
        api_key=api_key,
        secret_key=secret_key,
        paper=True
    )
)

# Verify no weekend dates (auto-checked by fetch_us_stocks)
df = data.get()
# Timezone is automatically set to America/New_York
# Weekend dates are automatically validated
```

**Why This Matters:**
- Without `tz='America/New_York'`: UTC midnight shifts dates backward by 1 day
- Result: Weekend dates appear, 0% match with TradingView, complete pattern detection failure
- Test conducted 2025-11-19: Wrong pattern = 0% match, Correct pattern = 100% match

**See:** CLAUDE.md "CRITICAL: Date and Timezone Handling for Market Data"

### Step 1: Bar Classification
```
Input: OHLC data (with correct timezone!)
Process: Apply bar typing logic
Output: Bar sequence [1, 2U, 1, 2U, 3, 2D...]
→ See PATTERNS.md Section 1
```

### Step 2: Pattern Detection
```
Input: Bar sequence
Process: Scan for valid patterns
Output: Pattern matches with triggers
→ See PATTERNS.md Sections 2-6
```

### Step 3: Timeframe Analysis
```
Input: Patterns from multiple timeframes
Process: Check 4 C's, identify MOAF
Output: Trade quality score + TF alignment
→ See TIMEFRAMES.md Sections 1-3
```

### Step 4: Entry Calculation
```
Input: Pattern + TF analysis
Process: Calculate trigger, stop, targets
Output: Entry parameters
→ See EXECUTION.md Section 1
```

### Step 5: Position Management
```
Input: Active position
Process: Monitor price action, adjust stops/targets
Output: Exit signals
→ See EXECUTION.md Section 2
```

### Step 6: Options Selection (Optional)
```
Input: STRAT pattern + targets
Process: Select strikes, calculate ROI
Output: Options trade parameters
→ See OPTIONS.md
```

---

## File Navigation

### [PATTERNS.md](PATTERNS.md) (~650 lines)
**When to read:** Implementing bar classification or pattern detection

**Contains:**
1. Bar Classification Logic - Detailed operator logic and edge cases
2. 2-1-2 Patterns - Reversal pattern detection
3. 3-1-2 Patterns - Expansion reversal patterns
4. 2-2 Patterns - Continuation patterns
5. Rev Strat Patterns - Extreme reversal patterns
6. Pattern Variations - Inside bar scenarios
7. Invalid Patterns - What NOT to trade
8. Mother Bar Identification - Multi-timeframe pivots

### [TIMEFRAMES.md](TIMEFRAMES.md) (~1050 lines)
**When to read:** Implementing multi-timeframe analysis

**Contains:**
1. The 4 C's Framework - Combo, Confirm, Continue, Consolidate
2. MOAF (Mother of All Flips) - Institutional timeframe flips
3. Timeframe Relationships - How daily/60min/15min interact
4. Continuity Scoring - Quantifying alignment
5. Trade Quality Matrix - Position sizing based on TF alignment
6. **Weekly/Monthly Analysis** - Extended timeframe patterns
7. **Cascade Analysis** - Monthly→Weekly→Daily alignment
8. **Weekly/Monthly MOAF** - Major trend reversal detection

### [EXECUTION.md](EXECUTION.md) (~500 lines)
**When to read:** Building entry/exit logic

**Contains:**
1. Entry Decision Framework - 4-level priority system
2. Trigger Calculations - Exact entry levels
3. Stop Placement - Initial and trailing stops
4. Target Scenarios - 3 target levels per pattern
5. Position Scaling - Entry/exit scaling rules
6. Break-Even Management - When and how to move stops
7. Profit Rolling - Scaling into new setups

### [OPTIONS.md](OPTIONS.md) (~410 lines)
**When to read:** Integrating options with STRAT patterns

**Contains:**
1. Strike Selection Methodology - Based on entry-to-target range
2. Expected ROI Calculation - Breakeven vs target prices
3. Greeks Analysis - Delta, Theta, Vega requirements
4. Expiration Selection - Matching pattern timeframe
5. Position Sizing - Risk management for options
6. Liquidity Requirements - Volume and open interest
7. Cheap Options Strategy - 0DTE and weekly plays

---

## Requirements

### Software
- **Python 3.8+** - Core language
- **VectorBT Pro** - Backtesting engine
  - Install: `pip install vectorbtpro`
  - License required: https://vectorbt.pro
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Knowledge Prerequisites
- Understanding of OHLC candlestick data
- Basic Python programming
- Familiarity with pandas DataFrames
- Options basics (if using OPTIONS.md)

### Data Requirements
- **OHLC data** - High, Low, Open, Close prices
- **Timeframe alignment** - Daily, 60min, 15min recommended
- **Historical depth** - Minimum 500 bars for backtesting
- **Continuous contracts** - For futures (adjust for rolls)

### VectorBT Pro Setup
```python
import vectorbtpro as vbt

# Configure VectorBT Pro
vbt.settings.set_theme('seaborn')
vbt.settings['plotting']['layout']['width'] = 1400
vbt.settings['plotting']['layout']['height'] = 800
```

---

## Quick Start Example

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

# 1. Load data
data = vbt.YFData.pull('SPY', start='2023-01-01', end='2024-01-01')
high = data.get('High')
low = data.get('Low')

# 2. Classify bars (see PATTERNS.md Section 1 for full implementation)
@njit
def classify_bars_nb(high, low):
    bars = np.zeros(len(high), dtype=np.int8)
    bars[0] = 0  # First bar unclassifiable
    
    for i in range(1, len(high)):
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

bars = classify_bars_nb(high.values, low.values)
print(f"Bar distribution: {np.bincount(bars + 2)}")

# 3. Detect patterns (see PATTERNS.md Section 2)
# 4. Analyze timeframes (see TIMEFRAMES.md)
# 5. Calculate entries (see EXECUTION.md)
# 6. Backtest and analyze
```

---

## Common Issues

### Issue: Incorrect bar classification
**Solution:** Verify operator logic. Equal highs/lows should result in inside bar unless other bound is broken. See [PATTERNS.md](PATTERNS.md) Section 1 for edge cases.

### Issue: Patterns not triggering
**Solution:** Check trigger calculation. Trigger = high of 2U bar (bull) or low of 2D bar (bear), NOT the inside bar bounds. See [EXECUTION.md](EXECUTION.md) Section 2.

### Issue: Poor timeframe alignment
**Solution:** Ensure proper timeframe ratios (4:1 minimum). Daily/60min/15min is optimal. See [TIMEFRAMES.md](TIMEFRAMES.md) Section 3.

### Issue: Stops too tight
**Solution:** Use full bar range for stop. Stop = low of 2U pattern (bull) or high of 2D pattern (bear). See [EXECUTION.md](EXECUTION.md) Section 3.

---

## Additional Resources

### VectorBT Pro Documentation
- Official docs: https://vectorbt.pro/docs
- Community: https://vectorbt.pro/community
- Examples: https://github.com/polakowo/vectorbt

### STRAT Methodology
- Rob Smith's content (creator of STRAT)
- TheStrat.com community resources
- Practice with paper trading before live implementation

---

## Summary

The STRAT methodology provides a systematic approach to technical analysis through:
1. Objective bar classification
2. Pattern-based entries
3. Multi-timeframe confirmation
4. Mechanical execution rules
5. Options integration

**Remember:** STRAT is NOT discretionary. Follow the rules mechanically for consistent results.

**Next steps:**
- For bar classification → Read [PATTERNS.md](PATTERNS.md)
- For timeframe analysis → Read [TIMEFRAMES.md](TIMEFRAMES.md)  
- For entry/exit rules → Read [EXECUTION.md](EXECUTION.md)
- For options trading → Read [OPTIONS.md](OPTIONS.md)
