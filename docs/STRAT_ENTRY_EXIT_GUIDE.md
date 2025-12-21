# STRAT Entry/Exit Reference Guide

**Purpose:** Precise entry, stop, and target mechanics for algorithmic implementation
**Audience:** Claude Code / VectorBT Pro implementation
**Version:** 1.0

---

## Section 1: Intrabar Evolution (CRITICAL)

### How Bars Evolve

Every intraday bar starts as Type 1 (inside) and evolves based on price action:

```
Bar opens -> Type 1 (no boundary broken yet)
Price breaks previous high -> Type 2U
Price breaks previous low -> Type 2D
Price breaks BOTH -> Type 3
```

### Entry Timing Rules

| Timing | Description | Use Case |
|--------|-------------|----------|
| CORRECT | Enter the instant price breaks trigger level | Live trading, backtesting |
| WRONG | Wait for bar to close as 2U/2D | Misses entry, late signals |
| WRONG | Enter when price touches trigger from wrong side | Premature entry |

### Intraday vs Daily+ Gaps

**Intraday (1m, 5m, 15m, 60m):**
- Previous bar close = current bar open (no gap)
- Bar always starts as Type 1
- Entry triggers intrabar

**Daily and higher:**
- Gap possible (previous close != current open)
- Bar may open as 2U/2D/3 immediately if gap breaks boundaries
- If bar gaps through trigger: enter at open, not trigger price

### Gap Handling

```python
def handle_gap_entry(trigger, bar_open, direction):
    """
    Handle entry when bar gaps through trigger.
    
    Args:
        trigger: Theoretical trigger price
        bar_open: Actual bar open price
        direction: 'long' or 'short'
    
    Returns:
        actual_entry_price
    """
    if direction == 'long':
        if bar_open > trigger:
            return bar_open  # Gapped through, enter at open
        return trigger
    else:  # short
        if bar_open < trigger:
            return bar_open  # Gapped through, enter at open
        return trigger
```

---

## Section 2: 3-1-2 Patterns (Traditional Magnitude)

### 3-1-2D (Bearish Reversal)

**Pattern Structure:**
```
Bar[2] (3):  Outside bar - broke both previous high AND low
Bar[1] (1):  Inside bar - contained within Bar[2]
Bar[0] (2D): Starts as 1, becomes 2D when breaks Bar[1] low
```

**OHLC Example:**
```
Bar[2] (3):  O:500 H:800 L:300 C:450
Bar[1] (1):  O:450 H:600 L:445 C:525
Bar[0] (2D): O:525 H:550 L:295 C:330
```

**Classification Logic:**
```
Bar[2]: Assume prev_H=475, prev_L=350
        800 > 475 AND 300 < 350 -> Type 3 (outside)

Bar[1]: prev_H=800, prev_L=300
        600 <= 800 AND 445 >= 300 -> Type 1 (inside)

Bar[0]: prev_H=600, prev_L=445
        Bar opens at 525, starts as Type 1
        When price breaks 445, becomes Type 2D
        550 <= 600 AND 295 < 445 -> Type 2D
```

**Entry Mechanics:**
```
Entry Trigger: Low of 1 bar - 0.01 = 444.99
Entry Timing:  The INSTANT price prints 444.99 on Bar[0]
               DO NOT wait for Bar[0] to close
```

**Stop Loss:**
```
Stop: High of 3 bar = 800
```

**Target:**
```
Target: Low of 3 bar = 300
```

**Python Implementation:**
```python
def detect_312d(bars, high, low, idx):
    """
    Detect 3-1-2D pattern and return entry parameters.
    
    Args:
        bars: Array of bar classifications
        high: Array of high prices
        low: Array of low prices
        idx: Current bar index
    
    Returns:
        dict with pattern info or None
    """
    if idx < 2:
        return None
    
    # Check pattern: 3 -> 1 -> 2D
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == -2:
        return {
            'pattern': '3-1-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-2],
            'target': low[idx-2],
            'bar_3_idx': idx-2,
            'bar_1_idx': idx-1,
            'bar_2d_idx': idx
        }
    return None
```

---

### 3-1-2U (Bullish Reversal)

**Pattern Structure:**
```
Bar[2] (3):  Outside bar - broke both previous high AND low
Bar[1] (1):  Inside bar - contained within Bar[2]
Bar[0] (2U): Starts as 1, becomes 2U when breaks Bar[1] high
```

**OHLC Example:**
```
Bar[2] (3):  O:500 H:800 L:300 C:350
Bar[1] (1):  O:350 H:600 L:445 C:525
Bar[0] (2U): O:525 H:650 L:510 C:640
```

**Classification Logic:**
```
Bar[2]: 3 (outside - broke both bounds)
Bar[1]: 1 (inside - 600 <= 800 AND 445 >= 300)
Bar[0]: Starts as 1, becomes 2U when breaks 600
        650 > 600 AND 510 >= 445 -> Type 2U
```

**Entry Mechanics:**
```
Entry Trigger: High of 1 bar + 0.01 = 600.01
Entry Timing:  The INSTANT price prints 600.01 on Bar[0]
               DO NOT wait for Bar[0] to close
```

**Stop Loss:**
```
Stop: Low of 3 bar = 300
```

**Target:**
```
Target: High of 3 bar = 800
```

**Python Implementation:**
```python
def detect_312u(bars, high, low, idx):
    """Detect 3-1-2U pattern and return entry parameters."""
    if idx < 2:
        return None
    
    if bars[idx-2] == 3 and bars[idx-1] == 1 and bars[idx] == 2:
        return {
            'pattern': '3-1-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-2],
            'target': high[idx-2],
            'bar_3_idx': idx-2,
            'bar_1_idx': idx-1,
            'bar_2u_idx': idx
        }
    return None
```

---

## Section 3: 2-1-2 Patterns (Traditional Magnitude)

### 2D-1-2U (Bullish Reversal)

**Pattern Structure:**
```
Bar[2] (2D): Directional down bar
Bar[1] (1):  Inside bar - contained within Bar[2]
Bar[0] (2U): Starts as 1, becomes 2U when breaks Bar[1] high
```

**OHLC Example:**
```
Bar[2] (2D): O:520 H:525 L:480 C:490
Bar[1] (1):  O:490 H:510 L:485 C:505
Bar[0] (2U): O:505 H:530 L:500 C:525
```

**Classification Logic:**
```
Bar[2]: Assume prev_H=530, prev_L=500
        525 <= 530 AND 480 < 500 -> Type 2D

Bar[1]: prev_H=525, prev_L=480
        510 <= 525 AND 485 >= 480 -> Type 1 (inside)

Bar[0]: prev_H=510, prev_L=485
        Bar opens at 505, starts as Type 1
        When price breaks 510, becomes Type 2U
        530 > 510 AND 500 >= 485 -> Type 2U
```

**Entry Mechanics:**
```
Entry Trigger: High of 1 bar + 0.01 = 510.01
Entry Timing:  The INSTANT price prints 510.01 on Bar[0]
```

**Stop Loss:**
```
Stop: Low of first bar (2D) = 480
```

**Target:**
```
Target: High of first bar (2D) = 525
```

**Python Implementation:**
```python
def detect_2d12u(bars, high, low, idx):
    """Detect 2D-1-2U (bullish reversal) pattern."""
    if idx < 2:
        return None
    
    if bars[idx-2] == -2 and bars[idx-1] == 1 and bars[idx] == 2:
        return {
            'pattern': '2D-1-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-2],
            'target': high[idx-2],
            'bar_2d_idx': idx-2,
            'bar_1_idx': idx-1,
            'bar_2u_idx': idx
        }
    return None
```

---

### 2U-1-2D (Bearish Reversal)

**Pattern Structure:**
```
Bar[2] (2U): Directional up bar
Bar[1] (1):  Inside bar - contained within Bar[2]
Bar[0] (2D): Starts as 1, becomes 2D when breaks Bar[1] low
```

**OHLC Example:**
```
Bar[2] (2U): O:480 H:530 L:475 C:520
Bar[1] (1):  O:520 H:525 L:485 C:510
Bar[0] (2D): O:510 H:520 L:470 C:480
```

**Classification Logic:**
```
Bar[2]: Assume prev_H=485, prev_L=470
        530 > 485 AND 475 >= 470 -> Type 2U

Bar[1]: prev_H=530, prev_L=475
        525 <= 530 AND 485 >= 475 -> Type 1 (inside)

Bar[0]: prev_H=525, prev_L=485
        When price breaks 485, becomes Type 2D
        520 <= 525 AND 470 < 485 -> Type 2D
```

**Entry Mechanics:**
```
Entry Trigger: Low of 1 bar - 0.01 = 484.99
Entry Timing:  The INSTANT price prints 484.99 on Bar[0]
```

**Stop Loss:**
```
Stop: High of first bar (2U) = 530
```

**Target:**
```
Target: Low of first bar (2U) = 475
```

**Python Implementation:**
```python
def detect_2u12d(bars, high, low, idx):
    """Detect 2U-1-2D (bearish reversal) pattern."""
    if idx < 2:
        return None
    
    if bars[idx-2] == 2 and bars[idx-1] == 1 and bars[idx] == -2:
        return {
            'pattern': '2U-1-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-2],
            'target': low[idx-2],
            'bar_2u_idx': idx-2,
            'bar_1_idx': idx-1,
            'bar_2d_idx': idx
        }
    return None
```

---

## Section 4: 2-2 Reversal Patterns (Traditional Magnitude)

### Reference Bar Validation (CRITICAL)

For a valid 2-2 reversal, the reference bar (bar BEFORE the pattern) MUST be a directional bar (2U or 2D).

| Reference Bar | Result |
|---------------|--------|
| 2U or 2D | Valid 2-2 Reversal |
| 1 (Inside) | NOT valid - this is a Rev Strat (see Section 9) |
| 3 (Outside) | NOT valid - this is a 3-2-2 pattern (see Section 5) |

```python
def is_valid_22_reversal(bars, idx):
    """
    Validate that 2-2 reversal has correct reference bar.
    Reference bar (idx-2) MUST be directional (2U or 2D).
    """
    if idx < 2:
        return False, None
    
    ref_bar = bars[idx-2]
    
    if ref_bar == 2 or ref_bar == -2:
        return True, '2-2_reversal'
    elif ref_bar == 1:
        return False, 'rev_strat'  # Route to Rev Strat logic
    elif ref_bar == 3:
        return False, '3-2-2'  # Route to 3-2-2 logic
    else:
        return False, 'invalid'
```

---

### 2D-2U (Bullish Reversal)

**Pattern Structure:**
```
Bar[2] (ref): Reference bar BEFORE the pattern (defines target)
Bar[1] (2D):  Directional down bar
Bar[0] (2U):  Reverses direction, breaks Bar[1] high
```

**OHLC Example:**
```
Bar[2] (ref): O:500 H:540 L:490 C:510  <- Target reference
Bar[1] (2D):  O:510 H:515 L:470 C:480
Bar[0] (2U):  O:480 H:530 L:475 C:520
```

**Classification Logic:**
```
Bar[1]: prev_H=540, prev_L=490
        515 <= 540 AND 470 < 490 -> Type 2D

Bar[0]: prev_H=515, prev_L=470
        530 > 515 AND 475 >= 470 -> Type 2U
```

**Entry Mechanics:**
```
Entry Trigger: High of 2D bar + 0.01 = 515.01
Entry Timing:  The INSTANT price prints 515.01 on Bar[0]
```

**Stop Loss:**
```
Stop: Low of 2D bar = 470
```

**Target:**
```
Target: High of reference bar (bar BEFORE the 2D) = 540
```

**Python Implementation:**
```python
def detect_2d2u(bars, high, low, idx):
    """
    Detect 2D-2U (bullish reversal) pattern.
    VALIDATES that reference bar is directional (2U or 2D).
    """
    if idx < 2:
        return None
    
    # Check pattern structure
    if bars[idx-1] == -2 and bars[idx] == 2:
        
        # VALIDATE reference bar
        ref_bar = bars[idx-2]
        if ref_bar != 2 and ref_bar != -2:
            # Not a valid 2-2 reversal
            # ref_bar == 1 -> Rev Strat
            # ref_bar == 3 -> 3-2-2 pattern
            return None
        
        return {
            'pattern': '2D-2U',
            'direction': 'long',
            'entry_trigger': high[idx-1] + 0.01,
            'stop': low[idx-1],
            'target': high[idx-2],  # Reference bar high
            'ref_bar_type': ref_bar,
            'bar_ref_idx': idx-2,
            'bar_2d_idx': idx-1,
            'bar_2u_idx': idx
        }
    return None
```

---

### 2U-2D (Bearish Reversal)

**Pattern Structure:**
```
Bar[2] (ref): Reference bar BEFORE the pattern (defines target)
Bar[1] (2U):  Directional up bar
Bar[0] (2D):  Reverses direction, breaks Bar[1] low
```

**OHLC Example:**
```
Bar[2] (ref): O:510 H:520 L:470 C:490  <- Target reference
Bar[1] (2U):  O:490 H:540 L:485 C:530
Bar[0] (2D):  O:530 H:535 L:475 C:480
```

**Classification Logic:**
```
Bar[1]: prev_H=520, prev_L=470
        540 > 520 AND 485 >= 470 -> Type 2U

Bar[0]: prev_H=540, prev_L=485
        535 <= 540 AND 475 < 485 -> Type 2D
```

**Entry Mechanics:**
```
Entry Trigger: Low of 2U bar - 0.01 = 484.99
Entry Timing:  The INSTANT price prints 484.99 on Bar[0]
```

**Stop Loss:**
```
Stop: High of 2U bar = 540
```

**Target:**
```
Target: Low of reference bar (bar BEFORE the 2U) = 470
```

**Python Implementation:**
```python
def detect_2u2d(bars, high, low, idx):
    """
    Detect 2U-2D (bearish reversal) pattern.
    VALIDATES that reference bar is directional (2U or 2D).
    """
    if idx < 2:
        return None
    
    # Check pattern structure
    if bars[idx-1] == 2 and bars[idx] == -2:
        
        # VALIDATE reference bar
        ref_bar = bars[idx-2]
        if ref_bar != 2 and ref_bar != -2:
            # Not a valid 2-2 reversal
            # ref_bar == 1 -> Rev Strat
            # ref_bar == 3 -> 3-2-2 pattern
            return None
        
        return {
            'pattern': '2U-2D',
            'direction': 'short',
            'entry_trigger': low[idx-1] - 0.01,
            'stop': high[idx-1],
            'target': low[idx-2],  # Reference bar low
            'ref_bar_type': ref_bar,
            'bar_ref_idx': idx-2,
            'bar_2u_idx': idx-1,
            'bar_2d_idx': idx
        }
    return None
```

---

## Section 5: 3-2 Patterns (1.5% Measured Move)

### Why 1.5% Measured Move?

3-2 patterns have variable magnitude from the 3 bar, making traditional targets inconsistent. The 1.5% measured move simplifies target calculation:

```
Target = Entry +/- (Entry * 0.015)
```

---

### 3-2D (Bearish)

**Pattern Structure:**
```
Bar[1] (3):  Outside bar - broke both bounds
Bar[0] (2D): Starts as 1, becomes 2D when breaks Bar[1] low
```

**OHLC Example:**
```
Bar[1] (3):  O:400 H:500 L:300 C:325
Bar[0] (2D): O:325 H:340 L:280 C:290
```

**Classification Logic:**
```
Bar[1]: 3 (outside bar)
Bar[0]: prev_H=500, prev_L=300
        340 <= 500 AND 280 < 300 -> Type 2D
```

**Entry Mechanics:**
```
Entry Trigger: Low of 3 bar - 0.01 = 299.99
Entry Timing:  The INSTANT price prints 299.99 on Bar[0]
```

**Stop Loss:**
```
Stop: High of 3 bar = 500
```

**Target (1.5% Measured Move):**
```
Target = Entry - (Entry * 0.015)
Target = 299.99 - (299.99 * 0.015)
Target = 299.99 - 4.50
Target = 295.49
```

**Python Implementation:**
```python
def detect_32d(bars, high, low, idx):
    """Detect 3-2D pattern with 1.5% target."""
    if idx < 1:
        return None
    
    if bars[idx-1] == 3 and bars[idx] == -2:
        entry_trigger = low[idx-1] - 0.01
        target = entry_trigger - (entry_trigger * 0.015)
        
        return {
            'pattern': '3-2D',
            'direction': 'short',
            'entry_trigger': entry_trigger,
            'stop': high[idx-1],
            'target': target,
            'target_method': '1.5_percent',
            'bar_3_idx': idx-1,
            'bar_2d_idx': idx
        }
    return None
```

---

### 3-2U (Bullish)

**Pattern Structure:**
```
Bar[1] (3):  Outside bar - broke both bounds
Bar[0] (2U): Starts as 1, becomes 2U when breaks Bar[1] high
```

**OHLC Example:**
```
Bar[1] (3):  O:400 H:500 L:300 C:475
Bar[0] (2U): O:475 H:520 L:460 C:510
```

**Classification Logic:**
```
Bar[1]: 3 (outside bar)
Bar[0]: prev_H=500, prev_L=300
        520 > 500 AND 460 >= 300 -> Type 2U
```

**Entry Mechanics:**
```
Entry Trigger: High of 3 bar + 0.01 = 500.01
Entry Timing:  The INSTANT price prints 500.01 on Bar[0]
```

**Stop Loss:**
```
Stop: Low of 3 bar = 300
```

**Target (1.5% Measured Move):**
```
Target = Entry + (Entry * 0.015)
Target = 500.01 + (500.01 * 0.015)
Target = 500.01 + 7.50
Target = 507.51
```

**Python Implementation:**
```python
def detect_32u(bars, high, low, idx):
    """Detect 3-2U pattern with 1.5% target."""
    if idx < 1:
        return None
    
    if bars[idx-1] == 3 and bars[idx] == 2:
        entry_trigger = high[idx-1] + 0.01
        target = entry_trigger + (entry_trigger * 0.015)
        
        return {
            'pattern': '3-2U',
            'direction': 'long',
            'entry_trigger': entry_trigger,
            'stop': low[idx-1],
            'target': target,
            'target_method': '1.5_percent',
            'bar_3_idx': idx-1,
            'bar_2u_idx': idx
        }
    return None
```

---

### 3-2D-2U (Sequential: Short then Long)

**This is TWO sequential trades, not one pattern.**

**CRITICAL:** Trade 2 target is HIGH of 3 bar (traditional magnitude), NOT 1.5% measured move.
Only standalone 3-2 patterns use 1.5%. Once it becomes 3-2-2, use traditional magnitude.

**Trade 1: 3-2D (Short)**
```
Bar[2] (3):  O:400 H:500 L:300 C:325
Bar[1] (2D): O:325 H:340 L:280 C:290

Entry: 299.99 (low of 3 bar - 0.01)
Stop: 500 (high of 3 bar)
Target: 295.49 (1.5% below entry)
```

**Trade 2: Triggered when Bar[0] becomes 2U**
```
Bar[2] (3):  O:400 H:500 L:300 C:325  <- Target reference (HIGH = 500)
Bar[1] (2D): O:325 H:340 L:280 C:290
Bar[0] (2U): O:290 H:360 L:285 C:350

Entry: 340.01 (high of 2D bar + 0.01)
Stop: 280 (low of 2D bar)
Target: 500 (HIGH of 3 bar - traditional magnitude)
```

**Trade Flip Logic:**
```
If Trade 1 (3-2D short) is still OPEN when Bar[0] breaks 2D high:
  1. IMMEDIATELY EXIT Trade 1 at current price
  2. ENTER Trade 2 (long) at 340.01
  3. New target: 500 (high of 3 bar)
```

**Detection Logic:**
```python
def detect_32d2u(bars, high, low, idx):
    """
    Detect 3-2D-2U pattern evolution.
    Returns TWO trade signals if pattern completes.
    """
    if idx < 2:
        return None
    
    # Check for completed 3-2D-2U sequence
    if bars[idx-2] == 3 and bars[idx-1] == -2 and bars[idx] == 2:
        
        # Trade 1: Original 3-2D (may already be closed)
        trade1_entry = low[idx-2] - 0.01
        trade1_target = trade1_entry - (trade1_entry * 0.015)
        
        # Trade 2: New 2U reversal - TARGET IS HIGH OF 3 BAR
        trade2_entry = high[idx-1] + 0.01
        trade2_target = high[idx-2]  # HIGH of 3 bar (traditional magnitude)
        
        return {
            'pattern': '3-2D-2U',
            'trade_1': {
                'direction': 'short',
                'entry_trigger': trade1_entry,
                'stop': high[idx-2],
                'target': trade1_target,
                'target_method': '1.5_percent',
                'status': 'check_if_open'
            },
            'trade_2': {
                'direction': 'long',
                'entry_trigger': trade2_entry,
                'stop': low[idx-1],
                'target': trade2_target,
                'target_method': 'traditional_magnitude',
                'status': 'new_signal_flip_trade1'
            }
        }
    return None
```

---

### 3-2U-2D (Sequential: Long then Short)

**CRITICAL:** Trade 2 target is LOW of 3 bar (traditional magnitude), NOT 1.5% measured move.

**Trade 1: 3-2U (Long)**
```
Bar[2] (3):  O:400 H:500 L:300 C:475
Bar[1] (2U): O:475 H:520 L:460 C:510

Entry: 500.01 (high of 3 bar + 0.01)
Stop: 300 (low of 3 bar)
Target: 507.51 (1.5% above entry)
```

**Trade 2: Triggered when Bar[0] becomes 2D**
```
Bar[2] (3):  O:400 H:500 L:300 C:475  <- Target reference (LOW = 300)
Bar[1] (2U): O:475 H:520 L:460 C:510
Bar[0] (2D): O:510 H:515 L:440 C:450

Entry: 459.99 (low of 2U bar - 0.01)
Stop: 520 (high of 2U bar)
Target: 300 (LOW of 3 bar - traditional magnitude)
```

**Trade Flip Logic:**
```
If Trade 1 (3-2U long) is still OPEN when Bar[0] breaks 2U low:
  1. IMMEDIATELY EXIT Trade 1 at current price
  2. ENTER Trade 2 (short) at 459.99
  3. New target: 300 (low of 3 bar)
```

**Detection Logic:**
```python
def detect_32u2d(bars, high, low, idx):
    """
    Detect 3-2U-2D pattern evolution.
    Returns TWO trade signals if pattern completes.
    """
    if idx < 2:
        return None
    
    if bars[idx-2] == 3 and bars[idx-1] == 2 and bars[idx] == -2:
        
        # Trade 1: Original 3-2U (may already be closed)
        trade1_entry = high[idx-2] + 0.01
        trade1_target = trade1_entry + (trade1_entry * 0.015)
        
        # Trade 2: New 2D reversal - TARGET IS LOW OF 3 BAR
        trade2_entry = low[idx-1] - 0.01
        trade2_target = low[idx-2]  # LOW of 3 bar (traditional magnitude)
        
        return {
            'pattern': '3-2U-2D',
            'trade_1': {
                'direction': 'long',
                'entry_trigger': trade1_entry,
                'stop': low[idx-2],
                'target': trade1_target,
                'target_method': '1.5_percent',
                'status': 'check_if_open'
            },
            'trade_2': {
                'direction': 'short',
                'entry_trigger': trade2_entry,
                'stop': high[idx-1],
                'target': trade2_target,
                'target_method': 'traditional_magnitude',
                'status': 'new_signal_flip_trade1'
            }
        }
    return None
```

---

### NOT TRADED: Continuation Patterns

**CRITICAL:** The following patterns are NOT traded as new entries. When a continuation occurs, stay in the original position.

| Pattern | What Happens | Action |
|---------|--------------|--------|
| 3-2D-2D | 3-2D short continues down | HOLD original short position |
| 3-2U-2U | 3-2U long continues up | HOLD original long position |
| 3-1-2D-2D | 3-1-2D short continues down | HOLD original short position |
| 3-1-2U-2U | 3-1-2U long continues up | HOLD original long position |
| 2U-1-2D-2D | 2U-1-2D short continues down | HOLD original short position |
| 2D-1-2U-2U | 2D-1-2U long continues up | HOLD original long position |

**Why not trade continuations?**
- You're already in the position from the original pattern
- Adding to the position increases risk without a new setup
- The original target and stop remain valid

**Detection Logic:**
```python
def is_continuation(bars, idx):
    """
    Check if current bar is a continuation (same direction as previous).
    Returns True if this is a continuation, False if reversal or other.
    """
    if idx < 1:
        return False
    
    # Same direction = continuation
    # 2D followed by 2D
    if bars[idx-1] == -2 and bars[idx] == -2:
        return True
    
    # 2U followed by 2U
    if bars[idx-1] == 2 and bars[idx] == 2:
        return True
    
    return False
```

---

## Section 6: Rev Strat Patterns (FUTURE IMPLEMENTATION)

**STATUS: NOT YET IMPLEMENTED**

Rev Strats are "failed" inside bar patterns that reverse direction. When a 2-1-2 or 3-1-2 pattern is taken and the next bar reverses, this creates a Rev Strat setup.

### Rev Strat Classification

| Original Pattern | Bar 4 Type | Result | Trade 2 Target |
|------------------|------------|--------|----------------|
| 3-1-2D | 2U | 3-1-2D-2U (Rev Strat) | High of 3 bar |
| 3-1-2U | 2D | 3-1-2U-2D (Rev Strat) | Low of 3 bar |
| 2U-1-2D | 2U | 2U-1-2D-2U (Rev Strat) | High of 2U bar |
| 2D-1-2U | 2D | 2D-1-2U-2D (Rev Strat) | Low of 2D bar |

**Key Distinction:**
- If Bar 4 continues same direction (e.g., 3-1-2D-2D) = NOT Rev Strat, just continuation
- If Bar 4 reverses direction (e.g., 3-1-2D-2U) = Rev Strat

### Rev Strat 1-2-2 Bullish

**Pattern Structure:**
```
Bar[3] (ref): Reference bar (2U, 2D, or 3) - defines target
Bar[2] (1):   Inside bar
Bar[1] (2D):  Breaks low of inside bar (Trade 1: short)
Bar[0] (2U):  REVERSES, breaks high of 2D (Trade 2: long)
```

**OHLC Example:**
```
Bar[3] (3):  O:400 H:550 L:280 C:350
Bar[2] (1):  O:350 H:450 L:320 C:400
Bar[1] (2D): O:400 H:420 L:290 C:310
Bar[0] (2U): O:310 H:460 L:305 C:450
```

**Trade 1: 3-1-2D (Short)**
```
Entry: 319.99 (low of 1 bar - 0.01)
Stop: 550 (high of 3 bar)
Target: 280 (low of 3 bar)
```

**Trade 2: Rev Strat Bullish (Long)**
```
Entry: 420.01 (high of 2D bar + 0.01)
Stop: 290 (low of 2D bar)
Target: 550 (HIGH of reference bar - Bar[3])
```

**Trade Flip Logic:**
```
If Trade 1 (3-1-2D short) is still OPEN when Bar[0] breaks 2D high:
  1. IMMEDIATELY EXIT Trade 1 at current price
  2. ENTER Trade 2 (long) at 420.01
  3. New target: 550 (high of reference bar)
```

### Rev Strat 1-2-2 Bearish

**Pattern Structure:**
```
Bar[3] (ref): Reference bar (2U, 2D, or 3) - defines target
Bar[2] (1):   Inside bar
Bar[1] (2U):  Breaks high of inside bar (Trade 1: long)
Bar[0] (2D):  REVERSES, breaks low of 2U (Trade 2: short)
```

**OHLC Example:**
```
Bar[3] (3):  O:400 H:550 L:280 C:450
Bar[2] (1):  O:450 H:500 L:350 C:480
Bar[1] (2U): O:480 H:560 L:470 C:540
Bar[0] (2D): O:540 H:545 L:340 C:360
```

**Trade 1: 3-1-2U (Long)**
```
Entry: 500.01 (high of 1 bar + 0.01)
Stop: 280 (low of 3 bar)
Target: 550 (high of 3 bar)
```

**Trade 2: Rev Strat Bearish (Short)**
```
Entry: 469.99 (low of 2U bar - 0.01)
Stop: 560 (high of 2U bar)
Target: 280 (LOW of reference bar - Bar[3])
```

### Rev Strat vs Continuation Decision Tree

```python
def classify_pattern_evolution(bars, idx):
    """
    Classify whether a 4-bar sequence is Rev Strat or continuation.
    
    Returns:
        pattern_type: 'rev_strat', 'continuation', or None
    """
    if idx < 3:
        return None
    
    bar_1 = bars[idx-3]  # Reference bar
    bar_2 = bars[idx-2]  # Must be 1 (inside)
    bar_3 = bars[idx-1]  # First directional
    bar_4 = bars[idx]    # Current bar
    
    # Must have inside bar at position 2
    if bar_2 != 1:
        return None
    
    # Check for Rev Strat patterns
    # X-1-2D-2U (Bullish Rev Strat)
    if bar_3 == -2 and bar_4 == 2:
        return {
            'type': 'rev_strat',
            'direction': 'long',
            'pattern': f'{bar_1}-1-2D-2U',
            'target_bar_idx': idx-3,
            'target_side': 'high'
        }
    
    # X-1-2U-2D (Bearish Rev Strat)
    if bar_3 == 2 and bar_4 == -2:
        return {
            'type': 'rev_strat',
            'direction': 'short',
            'pattern': f'{bar_1}-1-2U-2D',
            'target_bar_idx': idx-3,
            'target_side': 'low'
        }
    
    # X-1-2D-2D (Continuation, not Rev Strat)
    if bar_3 == -2 and bar_4 == -2:
        return {
            'type': 'continuation',
            'direction': 'short',
            'pattern': f'{bar_1}-1-2D-2D',
            'note': 'Treat as original 3-1-2D pattern'
        }
    
    # X-1-2U-2U (Continuation, not Rev Strat)
    if bar_3 == 2 and bar_4 == 2:
        return {
            'type': 'continuation',
            'direction': 'long',
            'pattern': f'{bar_1}-1-2U-2U',
            'note': 'Treat as original 3-1-2U pattern'
        }
    
    return None
```

### Pattern Routing Logic

When detecting patterns, route based on reference bar type:

```python
def route_pattern(bars, high, low, idx):
    """
    Route pattern to correct detector based on reference bar.
    """
    if idx < 2:
        return None
    
    # Check for 2-2 pattern (bars[idx-1] and bars[idx] are directional)
    if (bars[idx-1] == 2 or bars[idx-1] == -2) and \
       (bars[idx] == 2 or bars[idx] == -2) and \
       bars[idx-1] != bars[idx]:  # Reversal
        
        ref_bar = bars[idx-2]
        
        if ref_bar == 2 or ref_bar == -2:
            # Valid 2-2 reversal
            return detect_22_reversal(bars, high, low, idx)
        
        elif ref_bar == 3:
            # 3-2-2 pattern
            return detect_322(bars, high, low, idx)
        
        elif ref_bar == 1:
            # Rev Strat (requires 4 bars)
            if idx >= 3:
                return detect_rev_strat(bars, high, low, idx)
        
    return None
```

---

## Section 7: Quick Reference Tables

### Entry Trigger Summary

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

### Stop Loss Summary

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

### Target Summary

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

**NOTE:** 3-2 patterns use 1.5% measured move. 3-2-2 patterns (Trade 2) use traditional magnitude.

---

## Section 8: VectorBT Pro Integration

### Important: Position Management for Trade Flips

**CRITICAL:** This simple detector returns entry signals only. For 3-2-2 patterns, you need additional position management logic:

```
Timeline:
Bar N:   3-2D fires -> entries[N] = -1 (short)
Bar N+1: 3-2D-2U fires -> entries[N+1] = 1 (long)

The entries array shows BOTH signals at different bars.
Position management must:
1. Detect you're already SHORT from bar N
2. EXIT the short when 3-2D-2U triggers at bar N+1
3. ENTER long at bar N+1
```

**Production Implementation:**
```python
# Track open positions and handle flips
def manage_positions(entries, entry_prices, stops, targets):
    """
    Convert raw entry signals to actual trades with flip handling.
    """
    position = 0  # 1=long, -1=short, 0=flat
    trades = []
    
    for i in range(len(entries)):
        if entries[i] != 0:
            # New signal detected
            if position != 0 and position != entries[i]:
                # FLIP: Exit current position, enter opposite
                trades.append({
                    'bar': i,
                    'action': 'flip',
                    'from': position,
                    'to': entries[i],
                    'exit_price': entry_prices[i],  # Exit at flip trigger
                    'entry_price': entry_prices[i],
                    'stop': stops[i],
                    'target': targets[i]
                })
            elif position == 0:
                # New entry from flat
                trades.append({
                    'bar': i,
                    'action': 'entry',
                    'direction': entries[i],
                    'entry_price': entry_prices[i],
                    'stop': stops[i],
                    'target': targets[i]
                })
            # If position == entries[i], it's a continuation - ignore
            
            position = entries[i]
    
    return trades
```

### Complete Pattern Detector

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

@njit
def classify_bars_nb(high, low):
    """Classify bars as Type 1, 2U, 2D, or 3."""
    n = len(high)
    bars = np.zeros(n, dtype=np.int8)
    bars[0] = 0  # First bar unclassifiable
    
    for i in range(1, n):
        broke_high = high[i] > high[i-1]
        broke_low = low[i] < low[i-1]
        
        if broke_high and broke_low:
            bars[i] = 3   # Outside
        elif broke_high:
            bars[i] = 2   # 2U
        elif broke_low:
            bars[i] = -2  # 2D
        else:
            bars[i] = 1   # Inside
    
    return bars


@njit
def detect_entries_nb(bars, high, low):
    """
    Detect all STRAT patterns and return entry signals.
    
    Returns:
        entries: Array of entry signals (1=long, -1=short, 0=none)
        entry_prices: Array of entry trigger prices
        stops: Array of stop prices
        targets: Array of target prices
    """
    n = len(bars)
    entries = np.zeros(n, dtype=np.int8)
    entry_prices = np.zeros(n, dtype=np.float64)
    stops = np.zeros(n, dtype=np.float64)
    targets = np.zeros(n, dtype=np.float64)
    
    for i in range(2, n):
        # 3-1-2D (Bearish)
        if bars[i-2] == 3 and bars[i-1] == 1 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-2]
            targets[i] = low[i-2]
        
        # 3-1-2U (Bullish)
        elif bars[i-2] == 3 and bars[i-1] == 1 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-2]
            targets[i] = high[i-2]
        
        # 2D-1-2U (Bullish Reversal)
        elif bars[i-2] == -2 and bars[i-1] == 1 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-2]
            targets[i] = high[i-2]
        
        # 2U-1-2D (Bearish Reversal)
        elif bars[i-2] == 2 and bars[i-1] == 1 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-2]
            targets[i] = low[i-2]
        
        # 2D-2U (Bullish Reversal) - VALIDATE reference bar is directional
        elif bars[i-1] == -2 and bars[i] == 2:
            ref_bar = bars[i-2]
            if ref_bar == 2 or ref_bar == -2:  # Valid 2-2 reversal
                entries[i] = 1
                entry_prices[i] = high[i-1] + 0.01
                stops[i] = low[i-1]
                targets[i] = high[i-2]  # Reference bar
            # If ref_bar == 1: Rev Strat (not implemented)
            # If ref_bar == 3: 3-2-2 pattern (handled separately)
        
        # 2U-2D (Bearish Reversal) - VALIDATE reference bar is directional
        elif bars[i-1] == 2 and bars[i] == -2:
            ref_bar = bars[i-2]
            if ref_bar == 2 or ref_bar == -2:  # Valid 2-2 reversal
                entries[i] = -1
                entry_prices[i] = low[i-1] - 0.01
                stops[i] = high[i-1]
                targets[i] = low[i-2]  # Reference bar
            # If ref_bar == 1: Rev Strat (not implemented)
            # If ref_bar == 3: 3-2-2 pattern (handled separately)
        
        # 3-2D (Bearish, 1.5% target) - ONLY if next bar doesn't reverse
        elif bars[i-1] == 3 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01
            stops[i] = high[i-1]
            targets[i] = entry_prices[i] - (entry_prices[i] * 0.015)
        
        # 3-2U (Bullish, 1.5% target) - ONLY if next bar doesn't reverse
        elif bars[i-1] == 3 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01
            stops[i] = low[i-1]
            targets[i] = entry_prices[i] + (entry_prices[i] * 0.015)
        
        # 3-2D-2U (Trade 2: Long with traditional magnitude)
        # Detects when 3-2D pattern reverses to 2U
        elif i >= 2 and bars[i-2] == 3 and bars[i-1] == -2 and bars[i] == 2:
            entries[i] = 1
            entry_prices[i] = high[i-1] + 0.01  # High of 2D bar
            stops[i] = low[i-1]                  # Low of 2D bar
            targets[i] = high[i-2]               # HIGH of 3 bar (traditional)
        
        # 3-2U-2D (Trade 2: Short with traditional magnitude)
        # Detects when 3-2U pattern reverses to 2D
        elif i >= 2 and bars[i-2] == 3 and bars[i-1] == 2 and bars[i] == -2:
            entries[i] = -1
            entry_prices[i] = low[i-1] - 0.01   # Low of 2U bar
            stops[i] = high[i-1]                 # High of 2U bar
            targets[i] = low[i-2]                # LOW of 3 bar (traditional)
    
    return entries, entry_prices, stops, targets


# Usage Example
data = vbt.YFData.pull('SPY', start='2024-01-01', end='2024-12-01')
high = data.get('High').values
low = data.get('Low').values

bars = classify_bars_nb(high, low)
entries, entry_prices, stops, targets = detect_entries_nb(bars, high, low)

# Create signals DataFrame
import pandas as pd
signals = pd.DataFrame({
    'bar_type': bars,
    'entry_signal': entries,
    'entry_price': entry_prices,
    'stop': stops,
    'target': targets
}, index=data.get('Close').index)

print(signals[signals['entry_signal'] != 0])
```

---

## Section 9: Common Mistakes

### WRONG: Waiting for bar close
```python
# WRONG - Detects pattern after bar closes
if bars[i] == -2:  # Bar already closed as 2D
    enter_short()   # TOO LATE
```

### CORRECT: Enter on trigger break
```python
# CORRECT - Enter when price breaks trigger
if current_price < trigger:  # Intrabar check
    enter_short()             # IMMEDIATE
```

### WRONG: Using inside bar bounds as trigger
```python
# WRONG - Using inside bar low as entry for 3-1-2D
entry = low[inside_bar_idx]  # INCORRECT
```

### CORRECT: Using inside bar bounds as trigger
```python
# CORRECT - Inside bar low IS the trigger
entry = low[inside_bar_idx] - 0.01  # CORRECT
```

### WRONG: Premature entry
```python
# WRONG - Entering before trigger is broken
if price_approaching_trigger:
    enter()  # PREMATURE
```

### CORRECT: Wait for trigger break
```python
# CORRECT - Only enter when trigger is broken
if price < trigger:  # For short
    enter()          # TRIGGERED
```

---

## Appendix: Pattern Visual Reference

```
3-1-2D (Bearish)
================
     |
   __|__    Entry --> break of 1 low
  |  3  |   
  |_____|      ___
     |        | 1 |
     |        |___|
   Target        |
     |          _|_
              | 2D |
              |____|


3-1-2U (Bullish)
================
   Target
     |          ___
     |        | 2U |
     |        |____|
     |           |
   __|__        _|_
  |  3  |      | 1 |
  |_____|      |___|
     |            |
     |    Entry --> break of 1 high


3-2D (Bearish, 1.5%)
====================
     |
   __|__    Entry --> break of 3 low
  |  3  |   
  |_____|     ___
     |       | 2D|
   Target    |___|
   (-1.5%)


3-2U (Bullish, 1.5%)
====================
   Target
   (+1.5%)    ___
             |2U |
             |___|
   __|__        
  |  3  |   Entry --> break of 3 high
  |_____|
     |
```
