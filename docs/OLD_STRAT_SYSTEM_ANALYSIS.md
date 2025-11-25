# Old STRAT System Analysis - Session 30

**Date:** November 13, 2025
**Location:** C:\STRAT-Algorithmic-Trading-System-V3
**Objective:** Understand what worked and what failed to inform new implementation

## Executive Summary

The old STRAT system had CORRECT bar classification and pattern detection logic in core/analyzer.py, but suffered from integration bugs in trading/strat_signals.py. The primary failures were:

1. Index calculation inconsistencies between analyzer.py (CORRECT) and strat_signals.py (BUGGY)
2. Superficial VBT integration without custom indicators or verification workflow
3. Manual DataFrame loops instead of vectorized operations
4. No testing framework to verify price calculations against known patterns

## What Worked - Keep These Components

### 1. Bar Classification Logic (analyzer.py:137-212)

**File:** core/analyzer.py
**Method:** STRATAnalyzer.classify_bars()
**Status:** CORRECT - Implements proper governing range tracking

**Key Implementation:**

```python
# Initialize governing range from first bar
governing_high = high_values[0]
governing_low = low_values[0]

for i in range(1, n_bars):
    current_high = high_values[i]
    current_low = low_values[i]

    # Check against GOVERNING range (not just previous bar)
    is_inside = (current_high <= governing_high) and (current_low >= governing_low)
    breaks_high = current_high > governing_high
    breaks_low = current_low < governing_low

    if is_inside:
        classifications[i] = 1  # Inside bar
        # Governing range DOES NOT change
    elif breaks_high and breaks_low:
        classifications[i] = 3  # Outside bar
        governing_high = current_high
        governing_low = current_low
    elif breaks_high:
        classifications[i] = 2  # 2U - Upward directional
        governing_high = current_high
        governing_low = current_low
    elif breaks_low:
        classifications[i] = -2  # 2D - Downward directional
        governing_high = current_high
        governing_low = current_low
```

**Why This Is Correct:**
- Consecutive inside bars reference the SAME governing range until broken by a 2 or 3
- Distinguishes 2U (2) from 2D (-2) for directional clarity
- Matches Rob Smith's STRAT methodology exactly

**Action:** Port this logic to VBT custom indicator with @njit compilation

### 2. Pattern Detection Logic (analyzer.py:521-674)

**File:** core/analyzer.py
**Methods:** _analyze_212_signal(), _analyze_312_signal()
**Status:** CORRECT - Proper pattern matching and price calculations

**2-1-2 Pattern Detection (lines 521-599):**

```python
# Pattern: 2-1-2 (Directional-Inside-Directional continuation)
bar1_idx = trigger_index - 2  # First directional bar
bar2_idx = trigger_index - 1  # Inside bar
bar3_idx = trigger_index      # Second directional bar (trigger)

# Bullish 2-1-2 (2U-1-2U):
if bar1_bullish and bar3['high'] > bar2['high']:
    direction = Direction.BULLISH
    trigger_price = bar2['high']  # Inside bar high
    entry_price = bar3['open']    # Execution price
    stop_loss = bar2['low']       # Inside bar low

    # MEASURED MOVE target (NOT structural level)
    pattern_height = bar1['high'] - bar1['low']
    target_price = trigger_price + pattern_height
```

**3-1-2 Pattern Detection (lines 605-674):**

```python
# Pattern: 3-1-2 (Outside-Inside-Directional reversal)
bar1_idx = trigger_index - 2  # Outside bar
bar2_idx = trigger_index - 1  # Inside bar
bar3_idx = trigger_index      # Directional bar (trigger)

# Bullish 3-1-2 (3-1-2U):
if bar3['high'] > bar2['high'] and bar3['low'] >= bar2['low']:
    direction = Direction.BULLISH
    trigger_price = bar2['high']  # Inside bar high
    entry_price = bar3['open']    # Execution price
    stop_loss = bar1['low']       # Outside bar low

    # MEASURED MOVE target using outside bar range
    pattern_height = bar1['high'] - bar1['low']
    target_price = trigger_price + pattern_height
```

**Why This Is Correct:**
- Uses MEASURED MOVE targets (projects pattern range)
- Proper index offsets for 3-bar patterns (idx-2, idx-1, idx)
- Matches STRAT methodology: enter at inside bar extreme, stop at structural level, target measured move

**Action:** Port this logic to VBT custom indicator with pattern detection

### 3. Timeframe Continuity Concept

**File:** core/analyzer.py (FTFC components)
**Status:** Implemented but not fully tested

**Concept:** Check multiple timeframes (5min, 15min, 60min, daily) for directional alignment. Higher alignment (4/4 TFs) = higher conviction.

**Action:** Defer to Layer 2 implementation, focus on single-timeframe patterns first

## What Failed - Critical Bugs to Fix

### Bug 1: Index Calculation Inconsistency (strat_signals.py:437-572)

**File:** trading/strat_signals.py
**Method:** _validate_and_create_signal()
**Status:** BUGGY - Wrong target price calculation

**The Bug (lines 493-508):**

```python
if pattern['inside_bar_idx'] is not None:
    # Pattern with inside bar (2-1-2, 3-1-2)
    inside_idx = pattern['inside_bar_idx']
    inside_high = data['high'].iloc[inside_idx]
    inside_low = data['low'].iloc[inside_idx]

    if pattern['direction'] == 'long':
        trigger_price = inside_high + self.TRIGGER_TOLERANCE
        stop_price = inside_low
        # BUG: Assumes idx-2 is always structural level
        target_price = data['high'].iloc[idx-2]  # LINE 503 BUG
    else:  # short
        trigger_price = inside_low - self.TRIGGER_TOLERANCE
        stop_price = inside_high
        # BUG: Assumes idx-2 is always structural level
        target_price = data['low'].iloc[idx-2]   # LINE 508 BUG
```

**Why This Is Wrong:**

For 3-1-2 pattern (idx = trigger bar):
- idx-2 = outside bar (Bar 1 of 3-1-2)
- This WORKS BY ACCIDENT for 3-1-2 because outside bar high/low IS a structural level

For 2-1-2 pattern (idx = trigger bar):
- idx-2 = first directional bar (Bar 1 of 2-1-2)
- This is WRONG - Bar 1 of 2-1-2 is NOT necessarily a structural reversal point
- Should use MEASURED MOVE (like analyzer.py does) or find structural level BEFORE pattern

**Root Cause:**
- strat_signals.py tried to use structural levels but didn't account for pattern differences
- analyzer.py uses MEASURED MOVE targets (correct approach)
- Integration between two systems created inconsistency

**Impact:**
- 2-1-2 patterns had incorrect targets (too conservative or too aggressive depending on market structure)
- 3-1-2 patterns worked by accident but for wrong reasons
- Caused confusion when comparing analyzer.py output vs strat_signals.py output

### Bug 2: Superficial VBT Integration

**File:** trading/strat_signals.py (entire file)
**Status:** FAILED - Used VBT incorrectly

**What Was Done (Wrong Approach):**

```python
# Called vbt.Portfolio.from_signals with basic entry/exit signals
# NO custom indicators for bar classification
# NO custom indicators for pattern detection
# Manual DataFrame loops to calculate signals
# Then passed pre-calculated signals to VBT
```

**Why This Failed:**
- VBT's power is in VECTORIZED custom indicators with @njit compilation
- Manual loops are 10-100x slower than vectorized operations
- No access to VBT's advanced features (signal cleaning, position sizing, metrics)
- Debugging was trial-and-error without VBT MCP server verification

**What Should Have Been Done:**

```python
# Step 1: Create VBT custom indicator for bar classification
@njit
def classify_bars_nb(high, low):
    # Numba-compiled bar classification
    # Returns classifications array
    pass

StratBarClassifier = vbt.IF(
    class_name='StratBarClassifier',
    input_names=['high', 'low'],
    output_names=['classification'],
).with_apply_func(classify_bars_nb)

# Step 2: Create VBT custom indicator for pattern detection
@njit
def detect_patterns_nb(classifications, high, low):
    # Vectorized pattern detection
    # Returns entries, exits, targets
    pass

StratPatternDetector = vbt.IF(
    class_name='StratPatternDetector',
    input_names=['classifications', 'high', 'low'],
    output_names=['entries', 'exits', 'targets'],
).with_apply_func(detect_patterns_nb)

# Step 3: Use custom indicators in Portfolio
pf = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=pattern_detector.entries,
    exits=pattern_detector.exits,
    size=position_sizer.size,
    # ... other parameters
)
```

### Bug 3: No VBT Verification Workflow

**Status:** CRITICAL - Caused 90% of implementation failures

**What Was Missing:**
1. No SEARCH of VBT documentation before implementing
2. No VERIFY that methods/parameters exist
3. No FIND of real-world VBT usage examples
4. No TEST with minimal examples before full implementation
5. Implemented full code, discovered bugs, repeated cycle 3-4 times

**Impact:**
- Wasted 40+ hours on trial-and-error debugging
- Assumed VBT methods existed that didn't (or had different signatures)
- Reinvented working VBT patterns incorrectly
- Discovered incompatibilities AFTER full implementation

**Solution:**
- MANDATORY 5-step VBT verification workflow (CLAUDE.md lines 115-303)
- Use VBT MCP server tools BEFORE writing any VBT code
- Test with mcp__vectorbt-pro__run_code() on minimal examples
- Only proceed to full implementation after verification passes

### Bug 4: No Test Suite for Price Calculations

**Status:** MISSING - No way to verify correctness

**What Was Missing:**
- No unit tests for bar classification on known data
- No verification tests for pattern detection with hand-calculated prices
- No comparison against TradingView STRAT indicator
- No CSV exports to manually verify entry/stop/target prices

**Impact:**
- Index bugs (Bug 1) went undetected until live testing
- No way to verify if analyzer.py or strat_signals.py was correct
- Debugging required live market data instead of synthetic test cases

**Solution:**
1. Create synthetic test data with KNOWN patterns:
   - 3-bar sequence with known high/low values
   - Hand-calculate expected trigger/stop/target prices
   - Assert VBT custom indicator output matches expected

2. Export VBT backtest signals to CSV
3. Compare against TradingView STRAT indicator (if available)
4. Manually verify first 10 signals match expected prices

## New Implementation Plan

### Phase 1: VBT Custom Indicator - Bar Classification

**File:** strat/bar_classifier.py
**Objective:** Port analyzer.py classify_bars() to VBT custom indicator

**Implementation:**

```python
from numba import njit
import numpy as np
import vectorbtpro as vbt

@njit
def classify_bars_nb(high, low):
    """
    Classify bars using STRAT methodology with governing range tracking.

    Returns:
        classifications: Array with values:
            -999 = unclassified reference bar (first bar)
               1 = inside bar
               2 = 2U (upward directional)
              -2 = 2D (downward directional)
               3 = outside bar
    """
    n_bars = len(high)
    classifications = np.zeros(n_bars, dtype=np.int32)
    classifications[0] = -999  # Reference bar

    # Initialize governing range
    governing_high = high[0]
    governing_low = low[0]

    for i in range(1, n_bars):
        current_high = high[i]
        current_low = low[i]

        is_inside = (current_high <= governing_high) and (current_low >= governing_low)
        breaks_high = current_high > governing_high
        breaks_low = current_low < governing_low

        if is_inside:
            classifications[i] = 1
        elif breaks_high and breaks_low:
            classifications[i] = 3
            governing_high = current_high
            governing_low = current_low
        elif breaks_high:
            classifications[i] = 2
            governing_high = current_high
            governing_low = current_low
        elif breaks_low:
            classifications[i] = -2
            governing_high = current_high
            governing_low = current_low

    return classifications

# Create VBT indicator
StratBarClassifier = vbt.IF(
    class_name='StratBarClassifier',
    input_names=['high', 'low'],
    output_names=['classification'],
).with_apply_func(classify_bars_nb)
```

**Testing:**

```python
# Test with synthetic data
import pandas as pd

# Create 5-bar sequence: Ref, 2U, 1, 2U, 3
test_data = pd.DataFrame({
    'high':  [100, 105, 104, 107, 110],
    'low':   [95,  98,  99,  101, 93],
    'close': [98,  103, 102, 105, 100]
})

classifier = StratBarClassifier.run(
    test_data['high'],
    test_data['low']
)

expected = np.array([-999, 2, 1, 2, 3])
assert np.array_equal(classifier.classification.values, expected), \
    f"Expected {expected}, got {classifier.classification.values}"

print("[PASS] Bar classification test")
```

### Phase 2: VBT Custom Indicator - Pattern Detection

**File:** strat/pattern_detector.py
**Objective:** Port analyzer.py pattern detection to VBT custom indicator

**Implementation:**

```python
@njit
def detect_312_patterns_nb(classifications, high, low):
    """
    Detect 3-1-2 patterns with entry/stop/target prices.

    Pattern: 3-1-2 (Outside-Inside-Directional)
    - Bar 1: Outside bar (3)
    - Bar 2: Inside bar (1)
    - Bar 3: Directional bar (2U or 2D)

    Returns:
        long_entries: Boolean array for long entries
        short_entries: Boolean array for short entries
        trigger_prices: Entry trigger prices (inside bar high/low)
        stop_prices: Stop loss prices (outside bar low/high)
        target_prices: Target prices (measured move)
    """
    n_bars = len(classifications)
    long_entries = np.zeros(n_bars, dtype=np.bool_)
    short_entries = np.zeros(n_bars, dtype=np.bool_)
    trigger_prices = np.full(n_bars, np.nan)
    stop_prices = np.full(n_bars, np.nan)
    target_prices = np.full(n_bars, np.nan)

    # Need at least 3 bars for pattern
    for i in range(2, n_bars):
        bar1_class = classifications[i-2]  # Outside bar
        bar2_class = classifications[i-1]  # Inside bar
        bar3_class = classifications[i]    # Directional bar

        # Check for 3-1-2 pattern
        if bar1_class == 3 and bar2_class == 1:
            # Bullish 3-1-2U
            if bar3_class == 2:  # 2U
                long_entries[i] = True
                trigger_prices[i] = high[i-1]  # Inside bar high
                stop_prices[i] = low[i-2]      # Outside bar low

                # Measured move target
                pattern_height = high[i-2] - low[i-2]
                target_prices[i] = trigger_prices[i] + pattern_height

            # Bearish 3-1-2D
            elif bar3_class == -2:  # 2D
                short_entries[i] = True
                trigger_prices[i] = low[i-1]   # Inside bar low
                stop_prices[i] = high[i-2]     # Outside bar high

                # Measured move target
                pattern_height = high[i-2] - low[i-2]
                target_prices[i] = trigger_prices[i] - pattern_height

    return long_entries, short_entries, trigger_prices, stop_prices, target_prices

# Create VBT indicator
Strat312Detector = vbt.IF(
    class_name='Strat312Detector',
    input_names=['classifications', 'high', 'low'],
    output_names=['long_entries', 'short_entries', 'trigger_prices', 'stop_prices', 'target_prices'],
).with_apply_func(detect_312_patterns_nb)
```

**Testing:**

```python
# Test with known 3-1-2 pattern
test_data = pd.DataFrame({
    'high':  [100, 110, 105, 112],  # Ref, Outside(3), Inside(1), 2U
    'low':   [95,  90,  92,  98]
})

classifier = StratBarClassifier.run(test_data['high'], test_data['low'])
detector = Strat312Detector.run(
    classifier.classification,
    test_data['high'],
    test_data['low']
)

# Bar 3 (idx=3) should be bullish 3-1-2U entry
assert detector.long_entries.iloc[3] == True
assert detector.trigger_prices.iloc[3] == 105  # Inside bar high (bar 2)
assert detector.stop_prices.iloc[3] == 90      # Outside bar low (bar 1)

# Target = trigger + pattern_height = 105 + (110-90) = 125
assert detector.target_prices.iloc[3] == 125

print("[PASS] 3-1-2 pattern detection test")
```

### Phase 3: Integration with ATLAS (Deferred)

**Objective:** Filter STRAT signals with ATLAS regime detection
**Status:** Defer to after Phase 1 and 2 are working standalone

**Integration Logic:**

```python
# Get ATLAS regime
regime = atlas_model.online_inference(spy_data, date='2024-01-15')

# Get STRAT signals
strat_signals = strat_detector.run(individual_stock_data)

# Filter signals by regime
if regime == 'TREND_BULL':
    # Only take bullish STRAT signals
    final_entries = strat_signals.long_entries
elif regime == 'TREND_BEAR':
    # Only take bearish STRAT signals
    final_entries = strat_signals.short_entries
elif regime == 'CRASH':
    # Risk-off: no entries
    final_entries = pd.Series(False, index=strat_signals.index)
else:  # TREND_NEUTRAL
    # Take both directions with reduced position size
    final_entries = strat_signals.long_entries | strat_signals.short_entries
```

## Implementation Checklist

### Pre-Implementation (MANDATORY)

- [x] Read HANDOFF.md (current state)
- [x] Read CLAUDE.md (development rules)
- [x] Analyze old STRAT system (this document)
- [ ] Query VBT documentation for custom indicator examples
- [ ] Test minimal VBT custom indicator example with run_code()

### Phase 1: Bar Classification

- [ ] Create strat/ directory structure
- [ ] Implement classify_bars_nb() function
- [ ] Create StratBarClassifier VBT indicator
- [ ] Test with synthetic data (5-bar sequence)
- [ ] Test with SPY 2020-2024 data
- [ ] Export CSV and compare to TradingView (if available)
- [ ] Add to tests/test_strat/test_bar_classifier.py

### Phase 2: Pattern Detection

- [ ] Implement detect_312_patterns_nb() function
- [ ] Create Strat312Detector VBT indicator
- [ ] Test with synthetic 3-1-2 pattern
- [ ] Implement detect_212_patterns_nb() function
- [ ] Create Strat212Detector VBT indicator
- [ ] Test with synthetic 2-1-2 pattern
- [ ] Backtest on SPY 2020-2024
- [ ] Compare vs buy-and-hold benchmark
- [ ] Add to tests/test_strat/test_pattern_detector.py

### Phase 3: Integration Testing

- [ ] Test ATLAS + STRAT signal filtering
- [ ] Measure signal quality matrix (HIGH/MEDIUM/LOW)
- [ ] Verify CRASH regime veto logic
- [ ] Paper trade for 30 days minimum
- [ ] Compare paper trade results to backtest

## Key Lessons for New Implementation

1. **ALWAYS use VBT custom indicators** - No manual DataFrame loops
2. **ALWAYS follow 5-step VBT verification workflow** - Prevents 90% of bugs
3. **ALWAYS test with synthetic data first** - Catch index bugs before live testing
4. **Use MEASURED MOVE targets** - Not structural levels with fixed index offsets
5. **Export CSV and manually verify first 10 signals** - Ensures correctness
6. **Port CORRECT logic from analyzer.py** - Not buggy logic from strat_signals.py
7. **Test each component independently** - Bar classification, pattern detection, integration
8. **Create test suite BEFORE implementation** - Catches bugs early

## Conclusion

The old STRAT system had the RIGHT algorithms (bar classification and pattern detection in analyzer.py) but WRONG integration (strat_signals.py index bugs and superficial VBT usage).

New implementation success depends on:
1. Porting CORRECT algorithms to VBT custom indicators
2. Using 5-step VBT verification workflow
3. Testing with synthetic data before live testing
4. Creating comprehensive test suite

Expected timeline:
- Phase 1 (Bar Classification): 2-3 hours
- Phase 2 (Pattern Detection): 4-6 hours
- Phase 3 (Integration Testing): 2-3 hours
- Total: 8-12 hours over 2-3 sessions

Next session should begin Phase 1: Bar Classification VBT custom indicator.
