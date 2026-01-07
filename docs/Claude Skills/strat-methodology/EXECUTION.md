# STRAT Execution - Entry/Exit Mechanics & Position Management

**Purpose:** Implementation guide for mechanical entry/exit decisions and position management  
**Parent:** [SKILL.md](SKILL.md)

---

## Table of Contents

1. [Entry Decision Framework](#1-entry-decision-framework)
2. [Position Management](#2-position-management)
3. [Break-Even Management](#3-break-even-management)
4. [Target Management](#4-target-management)
5. [Trailing Stops](#5-trailing-stops)
6. [Rolling Profits](#6-rolling-profits)
7. [Risk Management](#7-risk-management)
8. [Pattern Invalidation by Type 3](#8-pattern-invalidation-by-type-3)

---

## 1. Entry Decision Framework

### 4-Level Priority System

**Entry Hierarchy** (from highest to lowest priority):

| Level | Entry Type | Description | Priority |
|-------|-----------|-------------|----------|
| **1** | Full Target | Price already exceeded target scenario | **HIGHEST** |
| **2** | MOAF | Daily+ timeframe flip | **HIGH** |
| **3** | Mechanical | Pattern trigger hit | **MEDIUM** |
| **4** | Discretionary | All scenarios converge | **LOW** |

---

### 1.1 Level 1: Full Target Entry

**Definition:** Price has already exceeded the target scenario of a setup you missed.

**Logic:** If price ran past target, there's momentum to continue.

**Example:**
```
2-1-2 Bull Setup:
Trigger: $100
Target 1: $102 (1R)
Target 2: $104 (2R)
Target 3: $106 (3R)

Current Price: $105 (exceeded T2, approaching T3)
Action: Enter immediately
New Target: $108+ (extension)
Stop: $103 (prior target as support)
```

**Detection Logic:**
```python
def check_full_target_entry(pattern, current_price, targets):
    """
    Check if price exceeded target and entry still valid.
    
    Args:
        pattern: Pattern dict with trigger/stop/targets
        current_price: Current market price
        targets: List of target prices [T1, T2, T3]
    
    Returns:
        (is_valid, entry_price, stop_price, new_target)
    """
    trigger = pattern['trigger']
    
    # Check if pattern triggered
    if pattern['bias'] == 'bull' and current_price <= trigger:
        return False, 0, 0, 0
    if pattern['bias'] == 'bear' and current_price >= trigger:
        return False, 0, 0, 0
    
    # Check if price exceeded any target
    exceeded_targets = []
    for i, target in enumerate(targets):
        if pattern['bias'] == 'bull' and current_price > target:
            exceeded_targets.append((i, target))
        elif pattern['bias'] == 'bear' and current_price < target:
            exceeded_targets.append((i, target))
    
    if len(exceeded_targets) >= 1:
        # Use highest exceeded target as new stop
        highest_exceeded = exceeded_targets[-1][1]
        
        # Calculate new target (1R beyond current price)
        risk = abs(current_price - highest_exceeded)
        new_target = current_price + risk if pattern['bias'] == 'bull' else current_price - risk
        
        return True, current_price, highest_exceeded, new_target
    
    return False, 0, 0, 0
```

**Risk Management:**
- Stop = Last exceeded target
- Size = 50-75% (momentum trade)
- Target = 1-2R from entry
- Exit fast if momentum stalls

---

### 1.2 Level 2: MOAF Entry

**Definition:** Daily or higher timeframe flips direction.

**Logic:** Institutional timeframe changes override all other signals.

**Example:**
```
Daily: Flips bullish (2-1-2 bull triggered at $98)
60min: Still bearish
15min: Still bearish

Entry: Immediately at market ($99)
Stop: Daily setup low ($96)
Target: Daily target ($104)
Size: 100% (highest conviction)
```

**Detection Logic:**
```python
def check_moaf_entry(daily_flip, daily_bias, daily_trigger, daily_stop, daily_target, current_price):
    """
    Check if MOAF entry is valid.
    
    Args:
        daily_flip: Boolean - did daily just flip?
        daily_bias: 'bull' or 'bear'
        daily_trigger: Daily trigger price
        daily_stop: Daily stop price
        daily_target: Daily target price
        current_price: Current market price
    
    Returns:
        (is_valid, entry_price, stop_price, target_price)
    """
    if not daily_flip:
        return False, 0, 0, 0
    
    # Check if trigger was hit
    if daily_bias == 'bull' and current_price > daily_trigger:
        return True, current_price, daily_stop, daily_target
    elif daily_bias == 'bear' and current_price < daily_trigger:
        return True, current_price, daily_stop, daily_target
    
    return False, 0, 0, 0
```

**Risk Management:**
- Stop = Daily setup stop
- Size = 100% (full size)
- Target = Daily target (minimum)
- Hold through lower TF noise

---

### 1.3 Level 3: Mechanical Entry

**Definition:** Standard STRAT pattern trigger is hit.

**Logic:** Pattern completes according to rules - enter mechanically.

**Example:**
```
2-1-2 Bull:
Bar 1: 2U at $95-$96
Bar 2: 1 at $95.50-$96.50
Bar 3: 2U at $96-$97 (trigger = $97)

Current Price: $97.10 (triggered)
Entry: $97.10
Stop: $95 (low of setup)
Target: $99 (2R)
```

**Detection Logic:**
```python
@njit
def check_mechanical_entry(bars, high, low, current_price, idx):
    """
    Check if mechanical entry triggered.
    
    Returns:
        (is_valid, entry_type, trigger, stop, target, bias)
    """
    # Check for 2-1-2 bull
    if idx >= 2:
        if bars[idx-2] == 2 and bars[idx-1] == 1 and bars[idx] == 2:
            trigger = high[idx-1] + 0.01  # Inside bar high + buffer
            stop = low[idx-2]
            risk = trigger - stop
            target = trigger + (risk * 2)  # 2R target

            if current_price > trigger:
                return True, '212_bull', trigger, stop, target, 'bull'

    # Check for 2-1-2 bear
    if idx >= 2:
        if bars[idx-2] == -2 and bars[idx-1] == 1 and bars[idx] == -2:
            trigger = low[idx-1] - 0.01  # Inside bar low - buffer
            stop = high[idx-2]
            risk = stop - trigger
            target = trigger - (risk * 2)  # 2R target
            
            if current_price < trigger:
                return True, '212_bear', trigger, stop, target, 'bear'
    
    # Check other patterns (3-1-2, 2-2, etc.)
    # ... additional pattern checks ...
    
    return False, None, 0, 0, 0, None
```

**Risk Management:**
- Stop = Pattern stop
- Size = Based on quality score (see TIMEFRAMES.md)
- Target = 2-3R
- Standard management rules

---

### 1.4 Level 4: Discretionary Entry

**Definition:** All scenarios converge but no mechanical trigger yet.

**Logic:** High conviction but requires judgment - lowest priority.

**Example:**
```
Daily: Bullish 2-1-2 forming (not triggered)
60min: Bullish 2-1-2 triggered
15min: Bullish 2-1-2 triggered
All align but no daily trigger yet

Discretionary: Enter on 15min, use daily stop
```

**When to use:**
- All timeframes align
- Pattern forming but not triggered
- Strong volume/momentum
- Critical support/resistance level

**Risk Management:**
- Stop = Wider (daily level)
- Size = 25-50% (discretionary = lower size)
- Target = Conservative (1-2R)
- Exit if setup fails to develop

---

## 2. Position Management

### Initial Position Entry

**Entry Scaling:**
- **Full entry:** 100% position at trigger (mechanical/MOAF)
- **Scaled entry:** 50% at trigger + 50% at confirmation
- **Discretionary:** 25-50% with plan to add

**Example - Scaled Entry:**
```
Pattern: 2-1-2 bull
Trigger: $100
Confirmation: Break above $101 (prior resistance)

Entry 1: 50% at $100 (trigger)
Entry 2: 50% at $101 (confirmation)
Average entry: $100.50
Stop: $98
```

---

### Position Scaling (Adding)

**When to add to position:**
1. Initial entry in profit (>0.5R)
2. New setup forms in same direction
3. Higher timeframe confirms
4. Stop moved to break-even

**When NOT to add:**
- Position underwater
- Against higher timeframe
- Near target
- Volatility expanding

**Example:**
```
Entry 1: $100, Size: 100 shares
Price moves to $102 (1R profit)
New 2-2 pattern forms
Entry 2: $102, Size: 50 shares (add 50%)
Total: 150 shares, Average: $100.67
```

---

## 3. Break-Even Management

### Break-Even Rules

**Move stop to break-even when:**
1. **0.5R profit:** Price reaches 50% of initial target
2. **New pattern:** Lower TF forms pattern in your direction
3. **Time-based:** Position held >1 day (swing trades)

**Implementation:**
```python
def manage_break_even(entry, stop, current_price, bars_held, bias):
    """
    Determine if stop should move to break-even.
    
    Args:
        entry: Entry price
        stop: Current stop price
        current_price: Current market price
        bars_held: Bars since entry
        bias: 'bull' or 'bear'
    
    Returns:
        (move_to_be, new_stop)
    """
    risk = abs(entry - stop)
    profit = (current_price - entry) if bias == 'bull' else (entry - current_price)
    
    # Rule 1: 0.5R profit
    if profit >= risk * 0.5:
        return True, entry
    
    # Rule 2: Time-based (24 bars for daily, 4 bars for 60min)
    if bars_held >= 24:
        if profit > 0:
            return True, entry
    
    return False, stop
```

**Break-Even Offset:**
- Add small buffer: Entry + $0.05 (stocks) or +1 tick (futures)
- Prevents stop-hunting
- Allows for minor whipsaw

---

## 4. Target Management

### Three-Target System

**Target Levels:**
```
T1 = Entry + 1R (50% exit)
T2 = Entry + 2R (30% exit)
T3 = Entry + 3R+ (20% runner)
```

**Example:**
```
Entry: $100
Stop: $98
Risk: $2

T1: $102 (1R = $2) → Exit 50%
T2: $104 (2R = $4) → Exit 30%
T3: $106+ (3R+ = $6+) → Exit 20% or trail
```

---

### Target Management Rules

**At T1 (50% exit):**
```python
def manage_t1(entry, stop, current_price, position_size, bias):
    """Manage T1 target."""
    risk = abs(entry - stop)
    t1 = entry + risk if bias == 'bull' else entry - risk
    
    if (bias == 'bull' and current_price >= t1) or \
       (bias == 'bear' and current_price <= t1):
        exit_size = position_size * 0.50
        new_stop = entry  # Move stop to break-even
        return exit_size, new_stop
    
    return 0, stop
```

**At T2 (30% exit):**
```python
def manage_t2(entry, stop, current_price, remaining_size, bias):
    """Manage T2 target."""
    risk = abs(entry - stop)
    t2 = entry + (risk * 2) if bias == 'bull' else entry - (risk * 2)
    
    if (bias == 'bull' and current_price >= t2) or \
       (bias == 'bear' and current_price <= t2):
        exit_size = remaining_size * 0.60  # 30% of original
        new_stop = entry + risk if bias == 'bull' else entry - risk  # Stop at T1
        return exit_size, new_stop
    
    return 0, stop
```

**At T3 (20% runner):**
```python
def manage_t3(entry, stop, current_price, remaining_size, bias):
    """Manage T3 target - runner with trailing stop."""
    risk = abs(entry - stop)
    t3 = entry + (risk * 3) if bias == 'bull' else entry - (risk * 3)
    
    if (bias == 'bull' and current_price >= t3) or \
       (bias == 'bear' and current_price <= t3):
        # Hold runner with trailing stop
        # Stop = 1R trail
        trail_distance = risk
        new_stop = current_price - trail_distance if bias == 'bull' else current_price + trail_distance
        return 0, new_stop  # Don't exit, just trail stop
    
    return 0, stop
```

---

### Early Target Adjustment

**If T1 hit quickly (<3 bars):**
- Extend T2 to 3R
- Extend T3 to 5R
- Increase runner size to 30%

**If T1 slow (>10 bars):**
- T2 becomes primary target
- Exit 80% at T2
- No runner

---

## 5. Trailing Stops

### Trailing Stop Methods

**Method 1: Fixed Distance Trail**
```python
def trail_stop_fixed(current_price, current_stop, trail_distance, bias):
    """
    Trail stop by fixed distance.
    
    Args:
        current_price: Current market price
        current_stop: Current stop level
        trail_distance: Distance to trail (in price units)
        bias: 'bull' or 'bear'
    
    Returns:
        new_stop
    """
    if bias == 'bull':
        potential_stop = current_price - trail_distance
        if potential_stop > current_stop:
            return potential_stop
    else:
        potential_stop = current_price + trail_distance
        if potential_stop < current_stop:
            return potential_stop
    
    return current_stop
```

**Method 2: STRAT Level Trail**
```python
def trail_stop_strat(bars, low, high, current_idx, bias):
    """
    Trail stop to most recent STRAT support/resistance.
    
    For bulls: Trail to most recent 2U low
    For bears: Trail to most recent 2D high
    """
    if bias == 'bull':
        # Find most recent 2U bar
        for i in range(current_idx - 1, max(0, current_idx - 10), -1):
            if bars[i] == 2:
                return low[i]
    else:
        # Find most recent 2D bar
        for i in range(current_idx - 1, max(0, current_idx - 10), -1):
            if bars[i] == -2:
                return high[i]
    
    return None
```

**Method 3: Time-Based Trail**
- After 5 bars: Trail to break-even + 0.5R
- After 10 bars: Trail to break-even + 1R
- After 20 bars: Trail to highest close - 0.5R

---

### When to Trail

**Trail stop when:**
1. Position >1R in profit
2. New STRAT level forms (2U low / 2D high)
3. Time-based milestones hit
4. Pattern completes on higher TF

**Do NOT trail when:**
- Position <0.5R profit
- High volatility (use wider stop)
- Approaching major target
- Inside bar consolidation forming

---

## 6. Rolling Profits

### Rolling Strategy

**Definition:** Exit profitable setup and immediately enter new setup in same direction.

**When to roll:**
1. Reach T2 (2R profit)
2. New pattern forms
3. Same or higher quality setup
4. Maintain trend exposure

**Example:**
```
Setup 1: 2-1-2 bull
Entry: $100
Exit at T2: $104 (2R profit, +$4)

Setup 2: 2-2 bull forms
Entry: $104
Risk: $2 (stop at $102)
Target: $108

Total exposure: $100 → $108 (8R potential)
Locked profit: $4 (2R)
Risk on Setup 2: $2 (1R)
```

---

### Rolling Rules

**Rule 1: Quality Requirement**
- New setup must be equal or higher quality
- Never roll into lower quality setup

**Rule 2: Profit Lock**
- Must have locked at least 1R profit
- Use 50% of profit for new entry

**Rule 3: Timeframe Alignment**
- Higher TF must still be aligned
- Don't roll against higher TF

**Implementation:**
```python
def evaluate_roll(current_setup, new_setup, locked_profit):
    """
    Evaluate if rolling to new setup is valid.
    
    Args:
        current_setup: Dict with current setup details
        new_setup: Dict with new setup details
        locked_profit: Profit locked from current setup (in R)
    
    Returns:
        (should_roll, new_position_size)
    """
    # Check quality
    if new_setup['quality_score'] < current_setup['quality_score']:
        return False, 0
    
    # Check profit lock
    if locked_profit < 1.0:  # Must lock at least 1R
        return False, 0
    
    # Check timeframe alignment
    if new_setup['daily_bias'] != current_setup['daily_bias']:
        return False, 0
    
    # Calculate new size (use 50% of profit)
    profit_amount = locked_profit * current_setup['risk_amount']
    new_size = (profit_amount * 0.5) / new_setup['risk_amount']
    
    return True, new_size
```

---

## 7. Risk Management

### Position Sizing Formula

**Standard Size:**
```
Position Size = (Account Risk %) / (Entry Risk %)

Example:
Account: $100,000
Risk per trade: 1% = $1,000
Entry: $100
Stop: $98
Risk per share: $2

Position Size = $1,000 / $2 = 500 shares
```

**Quality-Adjusted Size:**
```python
def calculate_position_size(account_size, risk_pct, entry, stop, quality_score):
    """
    Calculate position size based on account risk and quality.
    
    Args:
        account_size: Total account value
        risk_pct: Risk percentage (e.g., 0.01 for 1%)
        entry: Entry price
        stop: Stop price
        quality_score: Trade quality score (0-15)
    
    Returns:
        position_size (shares/contracts)
    """
    risk_amount = account_size * risk_pct
    risk_per_unit = abs(entry - stop)
    
    base_size = risk_amount / risk_per_unit
    
    # Adjust for quality
    if quality_score >= 10:
        multiplier = 1.5  # 150% for A+ setups
    elif quality_score >= 7:
        multiplier = 1.0  # 100% for A setups
    elif quality_score >= 5:
        multiplier = 0.75  # 75% for B setups
    elif quality_score >= 3:
        multiplier = 0.50  # 50% for C setups
    else:
        multiplier = 0.0  # Skip D setups
    
    return int(base_size * multiplier)
```

---

### Risk Per Trade Limits

| Setup Quality | Risk % | Max Positions | Total Risk % |
|---------------|--------|---------------|--------------|
| **A+ (12-15)** | 1.5% | 3 | 4.5% |
| **A (9-11)** | 1.0% | 5 | 5.0% |
| **B (6-8)** | 0.75% | 4 | 3.0% |
| **C (3-5)** | 0.50% | 2 | 1.0% |
| **D (0-2)** | 0% | 0 | 0% |

---

### Maximum Risk Rules

**Total Portfolio Risk:**
- Max 10% total risk across all positions
- Max 15% if all A+ setups
- Max 5% in single sector/instrument

**Correlation Risk:**
- Limit correlated positions (e.g., SPY + QQQ = 1.5 positions)
- Count ETF components (SPY includes AAPL)
- Reduce size if high correlation

**Time Risk:**
- Max 5 positions at once (quality >7)
- Max 3 positions if quality 5-7
- Max 1 position if quality 3-5

---

## 8. Pattern Invalidation by Type 3

### The Problem

**Pattern invalidation occurs when the entry bar evolves from Type 2 to Type 3.**

**Setup:** 2D-1-2D pattern
- Bar 1: 2D
- Bar 2: 1 (inside)
- Bar 3: Breaks low of Bar 2 → becomes 2D → ENTRY (short)
- Stop: High of Bar 1 (traditional)

**What if Bar 3 then breaks HIGH of Bar 2?**
- Bar 3 evolves: 2D → 3
- Pattern becomes: 2D-1-3
- The original pattern premise is **INVALIDATED**

**Why This Matters:**
1. Our trade was based on directional continuation (2D-1-2D)
2. Type 3 signals broadening/reversal potential
3. The premise of our trade no longer exists
4. Waiting for original stop may result in larger loss

---

### Exit Priority Order

**When managing an open position, check exits in this order:**

| Priority | Exit Type | Trigger | Action |
|----------|-----------|---------|--------|
| **1** | Target Hit | Price reaches target | Exit at target |
| **2** | Pattern Invalidated | Entry bar becomes Type 3 | **EXIT IMMEDIATELY** |
| **3** | Traditional Stop | Price hits stop level | Exit at stop |

**Critical:** Pattern invalidation takes priority over traditional stop. Exit at current price when detected - do not wait for original stop.

---

### Detection Logic

```python
def check_pattern_invalidation(
    pattern_type,
    entry_bar_idx,
    current_bar_type,
    entry_bar_original_type
):
    """
    Check if pattern has been invalidated by bar evolution.

    Args:
        pattern_type: Original pattern (e.g., '2D-1-2D', '2U-1-2U')
        entry_bar_idx: Index of the entry bar (Bar 3 in X-1-X patterns)
        current_bar_type: Current classification of entry bar
        entry_bar_original_type: What bar was when we entered

    Returns:
        dict with invalidation status and action
    """
    # If entry bar was Type 2 and is now Type 3, pattern is invalidated
    if entry_bar_original_type in [2, -2] and current_bar_type == 3:
        return {
            'invalidated': True,
            'reason': f'Entry bar evolved from {entry_bar_original_type} to Type 3',
            'action': 'EXIT_IMMEDIATELY',
            'note': 'Pattern premise no longer valid - broadening formation'
        }

    return {
        'invalidated': False,
        'reason': None,
        'action': 'HOLD',
        'note': None
    }
```

---

### Real-Time Monitoring (Numba)

```python
@njit
def monitor_bar_evolution_nb(
    high, low, prev_high, prev_low,
    entry_price, entry_type, position_open
):
    """
    Monitor for pattern invalidation during live bar.

    Called on each price update while position is open.

    Args:
        high: Current bar high (updating)
        low: Current bar low (updating)
        prev_high: Previous bar high (Bar 2 / inside bar)
        prev_low: Previous bar low (Bar 2 / inside bar)
        entry_price: Our entry price
        entry_type: 2 (long) or -2 (short)
        position_open: Boolean

    Returns:
        action: 0 = hold, 1 = exit (invalidated)
    """
    if not position_open:
        return 0

    # Check if bar has become Type 3
    broke_high = high > prev_high
    broke_low = low < prev_low

    if broke_high and broke_low:
        # Bar is now Type 3 - pattern invalidated
        return 1  # EXIT signal

    return 0  # HOLD
```

---

### Position Manager with Invalidation

```python
class StratPositionManager:
    """
    Manages STRAT positions with pattern invalidation handling.
    """

    def __init__(self):
        self.position = None

    def enter_position(self, direction, entry_price, pattern_info):
        """Record entry with pattern context."""
        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_bar_type': pattern_info['entry_bar_type'],
            'stop': pattern_info['stop'],
            'target': pattern_info['target'],
            'inside_bar_high': pattern_info['inside_bar_high'],
            'inside_bar_low': pattern_info['inside_bar_low'],
            'pattern': pattern_info['pattern']
        }

    def check_exit_conditions(self, current_price, current_high, current_low):
        """
        Check all exit conditions including pattern invalidation.

        Returns:
            dict with exit signal and reason
        """
        if self.position is None:
            return {'exit': False}

        # Priority 1: Check target hit
        if self.position['direction'] == 'long':
            if current_price >= self.position['target']:
                return {
                    'exit': True,
                    'reason': 'TARGET_HIT',
                    'price': self.position['target']
                }
        else:  # short
            if current_price <= self.position['target']:
                return {
                    'exit': True,
                    'reason': 'TARGET_HIT',
                    'price': self.position['target']
                }

        # Priority 2: Check pattern invalidation (Type 3 evolution)
        broke_high = current_high > self.position['inside_bar_high']
        broke_low = current_low < self.position['inside_bar_low']

        if broke_high and broke_low:
            # Entry bar became Type 3 - IMMEDIATE EXIT
            return {
                'exit': True,
                'reason': 'PATTERN_INVALIDATED',
                'note': 'Entry bar evolved to Type 3',
                'price': current_price  # Exit at current price
            }

        # Priority 3: Check traditional stop (only if not invalidated)
        if self.position['direction'] == 'long':
            if current_price <= self.position['stop']:
                return {
                    'exit': True,
                    'reason': 'STOP_HIT',
                    'price': self.position['stop']
                }
        else:  # short
            if current_price >= self.position['stop']:
                return {
                    'exit': True,
                    'reason': 'STOP_HIT',
                    'price': self.position['stop']
                }

        return {'exit': False}
```

---

### Patterns Affected

This invalidation logic applies to any pattern where entry is on a Type 2 bar:

| Pattern | Entry Bar | Invalidation Trigger |
|---------|-----------|---------------------|
| 2D-1-2D | 2D (Bar 3) | Bar 3 breaks inside bar high → becomes 3 |
| 2U-1-2U | 2U (Bar 3) | Bar 3 breaks inside bar low → becomes 3 |
| 2D-1-2U | 2U (Bar 3) | Bar 3 breaks inside bar low → becomes 3 |
| 2U-1-2D | 2D (Bar 3) | Bar 3 breaks inside bar high → becomes 3 |
| 3-1-2D | 2D (Bar 3) | Bar 3 breaks inside bar high → becomes 3 |
| 3-1-2U | 2U (Bar 3) | Bar 3 breaks inside bar low → becomes 3 |

---

### VectorBT Pro Integration

For backtesting, pattern invalidation must be checked BEFORE traditional stop:

```python
# Pseudocode for VectorBT signal generation
def generate_exit_signals(entries, highs, lows, inside_highs, inside_lows, stops, targets):
    """
    Generate exit signals with pattern invalidation priority.
    """
    exits = np.zeros(len(entries), dtype=np.int8)
    exit_prices = np.zeros(len(entries), dtype=np.float64)
    exit_reasons = np.empty(len(entries), dtype='U20')

    in_position = False
    position_idx = -1

    for i in range(len(entries)):
        if entries[i] != 0 and not in_position:
            in_position = True
            position_idx = i

        if in_position:
            # Priority 1: Check target
            # ... target logic ...

            # Priority 2: Check pattern invalidation
            if (highs[i] > inside_highs[position_idx] and
                lows[i] < inside_lows[position_idx]):
                exits[i] = 1
                exit_reasons[i] = 'INVALIDATED'
                # Exit price approximation: midpoint or close
                exit_prices[i] = (highs[i] + lows[i]) / 2
                in_position = False
                continue

            # Priority 3: Check traditional stop
            # ... stop logic ...

    return exits, exit_prices, exit_reasons
```

---

## Summary

**Entry Priority:**
1. Full Target (price exceeded target)
2. MOAF (daily flip)
3. Mechanical (pattern trigger)
4. Discretionary (convergence)

**Position Management:**
- Enter: Full or scaled (50% + 50%)
- T1: Exit 50% at 1R
- T2: Exit 30% at 2R
- T3: Trail 20% runner
- Break-even: Move stop at 0.5R profit

**Risk Management:**
- Size by quality score
- Max 1-1.5% risk per trade
- Max 10% total portfolio risk
- Trail stops in profit

**Key Rules:**
- Never add to losers
- Always move to break-even at T1
- Roll profits only into equal/better setups
- Size larger for higher quality

**Next Steps:**
- For pattern detection → Read [PATTERNS.md](PATTERNS.md)
- For timeframe analysis → Read [TIMEFRAMES.md](TIMEFRAMES.md)
- For options integration → Read [OPTIONS.md](OPTIONS.md)
