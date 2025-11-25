# STRAT Layer 2 Specification

**Version:** 1.0
**Date:** 2025-11-10
**Status:** Design specification (implementation pending)
**Source:** STRAT_SKILL_v2_0_OPTIMIZED.md (trained in Claude Desktop)

---

## Overview

STRAT (Sequential Time Recognition and Allocation Technique) is Layer 2 of the ATLAS trading system, providing bar-level pattern recognition and entry signal generation. Unlike ATLAS Layer 1 which identifies broad market regimes, STRAT operates on price action microstructure to detect specific reversal and continuation patterns.

**Dual function capability:**
- **Standalone mode:** Trade STRAT patterns independently (primary use case for $3k capital)
- **Integrated mode:** Use STRAT signals in confluence with ATLAS regime detection (optional)

**Key characteristics:**
- Timeframe independent (works on any interval: 1m, 5m, 15m, 60m, daily)
- Objective bar classification (no subjective interpretation)
- Multi-timeframe continuity analysis (alignment across 3+ timeframes)
- Options-first implementation (27x capital efficiency vs equities)

---

## Bar Classification System

### Core Concept

Every bar is classified into one of four types based on its relationship to the previous bar's high/low range:

```
Type 1 (Inside Bar):     Bar contained within previous bar's range
Type 2U (Directional):   Bar breaks previous high only
Type 2D (Directional):   Bar breaks previous low only
Type 3 (Outside Bar):    Bar breaks both previous high and low
```

### Classification Logic

**Operators:**
```python
# Given current bar (i) and previous bar (i-1)
high[i] > high[i-1]   # Breaks previous high
low[i] < low[i-1]     # Breaks previous low

# Bar type determination
if high[i] <= high[i-1] and low[i] >= low[i-1]:
    bar_type = 1  # Inside bar
elif high[i] > high[i-1] and low[i] >= low[i-1]:
    bar_type = 2  # 2U (up)
elif high[i] <= high[i-1] and low[i] < low[i-1]:
    bar_type = -2  # 2D (down)
elif high[i] > high[i-1] and low[i] < low[i-1]:
    bar_type = 3  # Outside bar
```

**VectorBT Pro implementation:**
```python
import numpy as np
from numba import njit

@njit
def classify_bar_nb(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Classify bars using STRAT methodology.

    Returns:
        Array of bar types: 1 (inside), 2 (up), -2 (down), 3 (outside)
    """
    n = len(high)
    bar_types = np.full(n, 0, dtype=np.int8)

    for i in range(1, n):
        breaks_high = high[i] > high[i-1]
        breaks_low = low[i] < low[i-1]

        if not breaks_high and not breaks_low:
            bar_types[i] = 1  # Inside
        elif breaks_high and not breaks_low:
            bar_types[i] = 2  # 2U
        elif not breaks_high and breaks_low:
            bar_types[i] = -2  # 2D
        else:
            bar_types[i] = 3  # Outside

    return bar_types
```

### Governing Range Concept

The "governing bar" is the most recent directional bar (type 2 or 3) that establishes the active trading range. Inside bars (type 1) inherit their parent's governing range.

```
Example sequence: [2U, 1, 1, 2U]
- Bar 0 (2U): Governs itself
- Bar 1 (1): Governed by bar 0
- Bar 2 (1): Governed by bar 0
- Bar 3 (2U): New governing bar
```

**Implementation:**
```python
@njit
def find_governing_bar_nb(bar_types: np.ndarray) -> np.ndarray:
    """
    Find the governing bar index for each bar.

    Returns:
        Array of indices pointing to each bar's governing bar
    """
    n = len(bar_types)
    governing = np.arange(n)  # Initialize to self

    last_directional = 0
    for i in range(1, n):
        if bar_types[i] == 1:  # Inside bar inherits
            governing[i] = last_directional
        else:  # Directional bar (2 or 3)
            last_directional = i
            governing[i] = i

    return governing
```

---

## Pattern Detection

### Primary Patterns

STRAT focuses on 6 high-probability reversal and continuation patterns:

1. **3-1-2 Bullish Reversal:** Outside bar → Inside bar → 2U breakout
2. **3-1-2 Bearish Reversal:** Outside bar → Inside bar → 2D breakdown
3. **2-1-2 Bullish Reversal:** 2D → Inside bar → 2U (failed breakdown)
4. **2-1-2 Bearish Reversal:** 2U → Inside bar → 2D (failed breakout)
5. **2-2 Bullish Continuation:** 2U → 2U (trending higher)
6. **2-2 Bearish Continuation:** 2D → 2D (trending lower)

### Pattern Detection Algorithm

**Unified detector:**
```python
@njit
def detect_pattern_nb(bar_types: np.ndarray) -> np.ndarray:
    """
    Detect all 6 STRAT patterns.

    Returns:
        Array of pattern codes:
        312 = 3-1-2 bull, -312 = 3-1-2 bear
        212 = 2-1-2 bull, -212 = 2-1-2 bear
        22 = 2-2 bull, -22 = 2-2 bear
        0 = no pattern
    """
    n = len(bar_types)
    patterns = np.zeros(n, dtype=np.int16)

    for i in range(2, n):
        # 3-1-2 patterns
        if bar_types[i-2] == 3 and bar_types[i-1] == 1:
            if bar_types[i] == 2:
                patterns[i] = 312  # Bullish
            elif bar_types[i] == -2:
                patterns[i] = -312  # Bearish

        # 2-1-2 patterns
        elif bar_types[i-1] == 1:
            if bar_types[i-2] == -2 and bar_types[i] == 2:
                patterns[i] = 212  # Bullish reversal
            elif bar_types[i-2] == 2 and bar_types[i] == -2:
                patterns[i] = -212  # Bearish reversal

        # 2-2 patterns
        elif bar_types[i-1] == 2 and bar_types[i] == 2:
            patterns[i] = 22  # Bullish continuation
        elif bar_types[i-1] == -2 and bar_types[i] == -2:
            patterns[i] = -22  # Bearish continuation

    return patterns
```

### Pattern "In Force" Concept

A pattern remains "in force" (active/valid) until one of three conditions occurs:

1. **Target hit:** Pattern achieves profit target
2. **Stop hit:** Pattern hits stop loss
3. **Invalidation:** New opposing pattern detected

```
Example: 2-1-2 bullish remains in force until:
- Price hits target (governing bar high + R-multiple), OR
- Price hits stop (governing bar low), OR
- A bearish pattern triggers
```

---

## Entry Rules and Price Calculations

### Entry Triggers

**Long entry (bullish patterns):**
```python
# Entry: Bar closes above governing bar high
entry_price = max(governing_high, close[pattern_bar])
```

**Short entry (bearish patterns):**
```python
# Entry: Bar closes below governing bar low
entry_price = min(governing_low, close[pattern_bar])
```

### Stop Loss Placement

**Long positions:**
```python
# Stop: 1 tick below governing bar low
stop_price = governing_low - tick_size
```

**Short positions:**
```python
# Stop: 1 tick above governing bar high
stop_price = governing_high + tick_size
```

### Profit Targets

**Risk-based targets (1R to 3R):**
```python
# Calculate risk (R)
risk = abs(entry_price - stop_price)

# Targets
target_1R = entry_price + (risk * 1.0)  # Long
target_2R = entry_price + (risk * 2.0)
target_3R = entry_price + (risk * 3.0)

# Position management: Scale out at each target
# 50% at 1R, 25% at 2R, 25% at 3R (or trail)
```

### Entry Validation

Before taking any entry, validate:

1. **Pattern completion:** All 3 bars present (or 2 for 2-2 patterns)
2. **Clean break:** Close beyond governing range (not just wick)
3. **Volume confirmation:** Above average volume on breakout bar (optional)
4. **No conflicting signals:** Check higher timeframe for alignment

---

## Timeframe Continuity

### The 4 C's Framework

Timeframe continuity refers to alignment (or lack thereof) across multiple timeframes. The relationship is classified into 4 categories:

**1. Control (all aligned):**
```
Daily: 2U pattern active
60min: 2U pattern active
15min: 2U pattern active
Result: Highest confidence, trend continuation likely
```

**2. Confirm (majority aligned):**
```
Daily: 2U pattern active
60min: 2U pattern active
15min: Inside bar (neutral)
Result: High confidence, waiting for 15min confirmation
```

**3. Conflict (mixed signals):**
```
Daily: 2U pattern active
60min: Inside bar
15min: 2D pattern active
Result: Choppy price action, reduce size or wait
```

**4. Change (majority reversing):**
```
Daily: 2D pattern active (was 2U)
60min: 2D pattern active (was 2U)
15min: 2D pattern active
Result: Trend reversal in progress, exit longs
```

### Continuity Scoring

**Implementation approach:**
```python
def calculate_continuity_score(timeframes: dict) -> float:
    """
    Calculate alignment across 3+ timeframes.

    Args:
        timeframes: {tf: pattern_direction} where direction is 1 (bull), -1 (bear), 0 (neutral)

    Returns:
        Score from -1.0 (all bearish) to +1.0 (all bullish)
    """
    directions = list(timeframes.values())
    score = sum(directions) / len(directions)
    return score

# Example usage
timeframes = {
    'daily': 1,   # Bullish pattern
    '60min': 1,   # Bullish pattern
    '15min': 0    # Inside bar (neutral)
}
score = calculate_continuity_score(timeframes)  # Returns 0.67 (confirm)
```

### Research Finding: 43.3% Alignment Rate

From STRAT skill training (Claude Desktop sessions):

> "When daily and 60min timeframes both show bullish patterns, the 15min timeframe aligns (also bullish) 43.3% of the time. This is significantly higher than the 33% random baseline."

**Trading implications:**
- **Control (100% alignment):** Rare, pursue aggressively when found
- **Confirm (67-80% alignment):** Most common setup, good risk/reward
- **Conflict (<50% alignment):** Avoid or reduce size significantly
- **Change (reversal):** Exit existing positions, consider counter-trend

---

## Position Management

### Scaling and Adjustment Rules

**Initial entry:**
- Start with 25-33% of intended full position size
- Enter on pattern trigger (governing range break)

**Adding to winners:**
```python
# Add 25% more on each of:
# 1. Next timeframe confirms (15min confirms 60min pattern)
# 2. Pattern reaches 1R profit
# 3. New pattern forms in same direction
# Maximum position: 100% of allocated capital
```

**Reducing losers:**
```python
# Reduce by 50% if:
# 1. Pattern goes against you (but stop not hit)
# 2. Higher timeframe shows conflict
# 3. Inside bar forms after entry (momentum pause)
```

### Trailing Stops

**After reaching 1R profit:**
```python
# Move stop to breakeven
trailing_stop = entry_price
```

**After reaching 2R profit:**
```python
# Trail at 1R (lock in minimum profit)
trailing_stop = entry_price + (risk * 1.0)
```

**After reaching 3R profit:**
```python
# Trail at 2R or use previous bar's low (bullish) / high (bearish)
trailing_stop = max(entry_price + (risk * 2.0), prev_bar_low)
```

---

## Options Integration

### Why Options for STRAT

**Capital efficiency:**
```
Equity position: $3,000 capital = $3,000 notional exposure
Options position: $3,000 capital = ~$80,000 notional exposure (27x leverage)

Example: SPY at $400
- 10 shares: $4,000 required
- 1 ATM call: $150 premium, controls $40,000 notional
```

### Strike Selection Algorithm

**For bullish patterns (calls):**
```python
# Target: Slightly out-of-the-money (OTM)
strike = next_strike_above(entry_price)

# If entry_price falls between strikes, use next one up
# Example: SPY entry at $450.25, strikes are $450/$451
# Selected strike: $451 call
```

**For bearish patterns (puts):**
```python
# Target: Slightly out-of-the-money (OTM)
strike = next_strike_below(entry_price)
```

### Expiration Selection

**Minimum holding period:**
```
Pattern timeframe: 15min → Minimum 2-3 days DTE
Pattern timeframe: 60min → Minimum 5-7 days DTE
Pattern timeframe: daily → Minimum 14-21 days DTE
```

**Never hold through:**
- Earnings announcements (enter after, or close before)
- FOMC meetings (high volatility risk)
- Last 3 days before expiration (gamma risk)

### Greeks Management

**Delta targeting:**
```python
# Aim for 0.40 - 0.60 delta at entry
# Higher delta = more stock-like behavior
# Lower delta = cheaper premium, less probability
ideal_delta_range = (0.40, 0.60)
```

**Theta consideration:**
```python
# Avoid positions with theta > 5% of premium per day
max_theta_ratio = 0.05
if abs(theta) / premium > max_theta_ratio:
    # Position will decay too fast, extend DTE
    pass
```

**Vega awareness:**
```python
# Volatility expansion helps long options
# Enter when IV < historical average (if possible)
# Exit if IV spike occurs without price movement (take profit)
```

### Position Sizing for Options

**Risk-based approach:**
```python
# Never risk more than 2% of capital per trade
account_value = 3000
risk_per_trade = account_value * 0.02  # $60

# If option premium is $150, max contracts:
premium = 150
max_contracts = risk_per_trade / premium  # 0.4 contracts

# Round down to whole number
contracts = int(max_contracts)  # 0 contracts (trade too risky)

# Solution: Use cheaper OTM strikes or wait for better setup
```

---

## Lessons from Previous STRAT Implementation

### Old System Analysis (C:\STRAT-Algorithmic-Trading-System-V3)

**What failed:**
1. **Superficial VectorBT integration:** Used VBT as data loader only, not for actual backtesting
2. **Index misalignment bugs:** Entry/exit calculations used wrong bar indices
3. **No position sizing logic:** Fixed contract counts, not dynamic risk-based
4. **Incomplete timeframe continuity:** Only checked alignment, didn't act on it

**What worked:**
1. **Bar classification logic:** Core algorithm was sound
2. **Pattern detection:** Correctly identified 3-1-2, 2-1-2, 2-2 formations
3. **Entry price calculations:** Governing range logic was accurate
4. **Options strike selection:** Algorithm produced valid strikes

### New Implementation Requirements

**Must have:**
1. **Full VBT integration:** Use VectorBT Pro Portfolio.from_signals() for backtesting
2. **Index alignment verification:** Test suite to catch off-by-one errors
3. **Dynamic position sizing:** PositionManager class with risk-based sizing
4. **Timeframe continuity actions:** Actual position adjustments based on alignment

**Verification workflow:**
1. **Search:** Use VBT MCP tools to find relevant API methods
2. **Verify:** Check documentation for parameter requirements
3. **Find:** Look for existing code examples in VBT docs
4. **Test:** Write unit tests before implementation
5. **Implement:** Integrate with confidence

---

## Implementation Checklist

### Phase 1: Bar Classification (Week 1)
- [ ] Implement classify_bar_nb() with Numba
- [ ] Implement find_governing_bar_nb()
- [ ] Create unit tests for all 4 bar types
- [ ] Verify against manual calculations on SPY daily data

### Phase 2: Pattern Detection (Week 2)
- [ ] Implement detect_pattern_nb() for all 6 patterns
- [ ] Create pattern visualization tool (ASCII charts)
- [ ] Unit tests for each pattern type
- [ ] Historical pattern frequency analysis (SPY 2020-2024)

### Phase 3: Entry/Exit Rules (Week 3)
- [ ] Implement entry price calculation
- [ ] Implement stop loss placement
- [ ] Implement profit targets (1R, 2R, 3R)
- [ ] Create EntrySignal class with validation

### Phase 4: Timeframe Continuity (Week 4)
- [ ] Multi-timeframe data alignment (VBT resampling)
- [ ] Continuity score calculation
- [ ] Position sizing adjustment based on alignment
- [ ] Validate 43.3% alignment finding on recent data

### Phase 5: Options Integration (Week 5)
- [ ] Strike selection algorithm
- [ ] Expiration selection logic
- [ ] Greeks calculation (delta, theta, vega)
- [ ] Options-specific position sizing

### Phase 6: Position Management (Week 6)
- [ ] PositionManager class implementation
- [ ] Scaling in/out logic
- [ ] Trailing stop calculation
- [ ] Portfolio-level risk management

### Phase 7: Backtesting (Week 7)
- [ ] VectorBT Portfolio integration
- [ ] Walk-forward validation setup
- [ ] Performance metrics calculation
- [ ] Compare vs buy-and-hold baseline

### Phase 8: Integration with ATLAS (Optional)
- [ ] Signal quality matrix implementation
- [ ] ATLAS crash veto logic
- [ ] Confluence scoring (ATLAS + STRAT agreement)
- [ ] Mixed deployment testing (paper ATLAS + live STRAT)

---

## Performance Expectations

### Backtesting Targets

**Minimum acceptable:**
- Win rate: 45%+ (pattern-based systems typically 40-50%)
- Average R-multiple: 1.5+ (winners larger than losers)
- Sharpe ratio: 1.0+ (risk-adjusted returns)
- Maximum drawdown: <20% (preserve capital)

**Stretch goals:**
- Win rate: 55%+
- Average R-multiple: 2.0+
- Sharpe ratio: 1.5+
- Maximum drawdown: <15%

### Live Trading Expectations

**First 30 trades (paper trading):**
- Focus on execution accuracy, not P&L
- Verify entry prices match backtested logic
- Confirm position sizing calculations
- Test order routing and fills

**After validation (live capital):**
- Start with $500-$1,000 allocation (not full $3k)
- Scale up as confidence builds
- Track slippage and commissions (often underestimated)
- Expect ~10% worse performance vs backtest (implementation gap)

---

## Limitations and Constraints

### What STRAT Does Not Do

1. **Predict market direction:** STRAT reacts to price action, doesn't forecast
2. **Work in all markets:** Requires sufficient volatility and liquidity
3. **Guarantee profits:** Some patterns fail, stops get hit
4. **Replace risk management:** Position sizing and stop losses still critical

### Market Conditions Where STRAT Struggles

1. **Low volatility environments:** Patterns form but don't move far enough to hit targets
2. **Overnight gaps:** Can blow through stops (use options to limit risk)
3. **Thin markets:** Wide bid-ask spreads erode edge
4. **High-frequency noise:** Very short timeframes (1min, 5min) less reliable

### Operational Constraints

1. **Data requirements:** Need clean OHLC data for all timeframes
2. **Computation speed:** Bar classification must run in real-time (<100ms)
3. **Order routing:** Options orders more complex than equities
4. **Capital requirements:** Minimum $3k for meaningful options positions

---

## Next Steps

**For immediate implementation:**
1. Begin Phase 1 (Bar Classification) using VBT MCP tools
2. Set up test suite framework in tests/test_strat/
3. Create strat/ directory structure matching regime/ layout
4. Reference STRAT_SKILL_v2_0_OPTIMIZED.md for detailed code examples

**For documentation:**
1. Update README.md with Layer 2 overview (done in this session)
2. Create INTEGRATION_ARCHITECTURE.md for ATLAS+STRAT confluence
3. Update project tree to show planned strat/ modules

**For validation:**
1. After Phase 7 (Backtesting), compare results to old STRAT system
2. Identify specific improvements from VBT integration
3. Document lessons learned in SESSION_28_STRAT_IMPLEMENTATION.md
