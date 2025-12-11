# STRAT Bar Classification - Claude Code Implementation Guide

**Purpose:** Build the STRAT bar classification system in VectorBT Pro  
**Status:** Ready for Claude Code implementation  
**Priority:** Phase 1 - Bar Classification Engine

---

## Implementation Overview

We're building THREE custom indicators:
1. **StratBarClassifier** - Classifies individual bars as 1, 2U, 2D, or 3
2. **TimeframeContinuity** - Checks alignment across Monthly/Weekly/Daily
3. **StratPatternDetector** - Identifies 3-1-2, 2-1-2, etc. patterns

**Start with #1 only** - validate bar classification before building on it.

---

## Step 1: Bar Classification Logic

### Classification Rules (CRITICAL - Must Be Exact)

```python
def classify_bar(current_high, current_low, prev_high, prev_low):
    """
    Classify a single bar based on relationship to previous bar.
    
    Returns:
        1: Inside bar (within prev range)
        '2U': Directional up (broke prev high only)
        '2D': Directional down (broke prev low only)  
        3: Outside bar (broke both high and low)
    """
    broke_high = current_high > prev_high
    broke_low = current_low < prev_low
    
    if broke_high and broke_low:
        return 3  # Outside bar
    elif broke_high and not broke_low:
        return '2U'  # Directional up
    elif broke_low and not broke_high:
        return '2D'  # Directional down
    else:
        return 1  # Inside bar
```

### Why This Matters

**Bar classification is RELATIVE, not absolute:**
- A bar is NOT "2U" because it's green/bullish
- A bar is "2U" because it broke the previous bar's high
- Same bar could be 1, 2U, 2D, or 3 depending on previous bar

**Context requirement:**
- To classify bar at index `i`, you need bar at index `i-1`
- First bar in dataset CANNOT be classified (no previous bar)
- Must skip first bar in backtest

---

## Step 2: VectorBT Pro Implementation

### Code Structure

```python
import vectorbtpro as vbt
import numpy as np
from numba import njit

# ============================================================================
# STEP 1: Define the classification function (Numba-compiled for speed)
# ============================================================================

@njit
def classify_bar_nb(high, low, prev_high, prev_low):
    """
    Numba-compiled bar classification.
    
    Args:
        high: Current bar high
        low: Current bar low
        prev_high: Previous bar high
        prev_low: Previous bar low
    
    Returns:
        int: 1 (inside), 2 (2U), -2 (2D), 3 (outside)
        
    Note: Using integers for Numba compatibility
          2 = 2U, -2 = 2D
    """
    broke_high = high > prev_high
    broke_low = low < prev_low
    
    if broke_high and broke_low:
        return 3  # Outside
    elif broke_high:
        return 2  # 2U (directional up)
    elif broke_low:
        return -2  # 2D (directional down)
    else:
        return 1  # Inside


@njit
def apply_strat_classification_nb(high, low):
    """
    Apply bar classification to entire array.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        
    Returns:
        Array of classifications (1, 2, -2, 3)
    """
    n = len(high)
    result = np.empty(n, dtype=np.int32)
    
    # First bar cannot be classified (no previous)
    result[0] = 0  # Use 0 as "undefined"
    
    # Classify remaining bars
    for i in range(1, n):
        result[i] = classify_bar_nb(
            high[i], 
            low[i],
            high[i-1],
            low[i-1]
        )
    
    return result


# ============================================================================
# STEP 2: Create the indicator using IndicatorFactory
# ============================================================================

StratBarClassifier = vbt.IF(
    class_name='StratBarClassifier',
    input_names=['high', 'low'],
    output_names=['classification'],
    param_names=[]  # No parameters needed
).with_apply_func(
    apply_strat_classification_nb,
    takes_1d=False,  # Operates on 2D arrays (multiple symbols)
    keep_pd=True  # Preserve pandas DataFrame structure
)

# Register the indicator
vbt.IF.register_custom_indicator(StratBarClassifier, "StratBars")


# ============================================================================
# STEP 3: Helper functions to interpret results
# ============================================================================

def decode_classification(value):
    """Convert integer classification back to readable format."""
    if value == 0:
        return 'UNDEFINED'
    elif value == 1:
        return '1'
    elif value == 2:
        return '2U'
    elif value == -2:
        return '2D'
    elif value == 3:
        return '3'
    else:
        return 'UNKNOWN'


def add_readable_column(df):
    """Add human-readable classification column."""
    return df.applymap(decode_classification)


# ============================================================================
# STEP 4: Usage example
# ============================================================================

# Example: Load SPY data
data = vbt.BinanceData.pull(
    "BTCUSDT",
    start="2024-01-01",
    end="2024-12-31",
    timeframe="1d"
)

# Or with Alpaca (your actual data source):
# data = vbt.AlpacaData.pull(
#     "SPY",
#     start="2024-01-01", 
#     end="2024-12-31",
#     timeframe="1d"
# )

# Run classification
strat_bars = StratBarClassifier.run(
    high=data.high,
    low=data.low
)

# Get results
classifications = strat_bars.classification

# Print first 20 bars with readable format
print("First 20 bar classifications:")
print(add_readable_column(classifications.head(20)))

# Count bar types
print("\nBar type distribution:")
print(f"Inside bars (1): {(classifications == 1).sum().sum()}")
print(f"2U bars: {(classifications == 2).sum().sum()}")
print(f"2D bars: {(classifications == -2).sum().sum()}")
print(f"Outside bars (3): {(classifications == 3).sum().sum()}")
```

---

## Step 3: Validation Tests

### Test 1: Manual Verification

**Create test data with known patterns:**

```python
import pandas as pd

# Create simple test data
test_dates = pd.date_range('2024-01-01', periods=10, freq='D')
test_data = pd.DataFrame({
    'high': [10, 12, 11, 13, 12, 14, 13, 15, 11, 16],
    'low':  [8,  9,  10, 10, 11, 11, 12, 12, 10, 13]
}, index=test_dates)

# Expected classifications:
# Bar 0: UNDEFINED (no previous)
# Bar 1: 2U (high 12 > prev high 10, low 9 > prev low 8)
# Bar 2: Inside (high 11 < prev high 12, low 10 > prev low 9)
# Bar 3: 2U (high 13 > prev high 11, low 10 == prev low 10 - NO BREAK)
# ... etc

# Run classifier
result = apply_strat_classification_nb(
    test_data['high'].values,
    test_data['low'].values
)

# Print results
for i, (date, row) in enumerate(test_data.iterrows()):
    if i == 0:
        print(f"{date.date()}: H={row['high']}, L={row['low']} -> UNDEFINED")
    else:
        prev = test_data.iloc[i-1]
        classification = decode_classification(result[i])
        print(f"{date.date()}: H={row['high']}, L={row['low']} "
              f"(prev H={prev['high']}, L={prev['low']}) -> {classification}")
```

### Test 2: Known STRAT Patterns

**Verify against manually identified patterns on real chart:**

```python
# Load SPY data for a known date range
spy_data = vbt.AlpacaData.pull(
    "SPY",
    start="2024-01-01",
    end="2024-01-31",
    timeframe="1d"
)

# Run classification
spy_bars = StratBarClassifier.run(
    high=spy_data.high,
    low=spy_data.low
)

# Check specific dates where you know the pattern
# Example: Jan 15 was a 2U bar on daily chart
print(f"Jan 15 classification: {decode_classification(spy_bars.classification.loc['2024-01-15'])}")

# Export to CSV for manual verification in TradingView
results_df = spy_data.data.copy()
results_df['strat_bar'] = spy_bars.classification.values
results_df['readable'] = results_df['strat_bar'].apply(decode_classification)
results_df.to_csv('spy_strat_classification.csv')

print("Exported to spy_strat_classification.csv - verify against TradingView")
```

### Test 3: Edge Cases

```python
def test_edge_cases():
    """Test specific edge case scenarios."""
    
    # Test 1: Gap up (should be 2U)
    high = np.array([100.0, 110.0])
    low = np.array([95.0, 105.0])
    result = apply_strat_classification_nb(high, low)
    assert result[1] == 2, "Gap up should be 2U"
    
    # Test 2: Inside bar with equal high/low
    high = np.array([100.0, 100.0])
    low = np.array([95.0, 96.0])
    result = apply_strat_classification_nb(high, low)
    assert result[1] == 1, "Equal high with higher low should be inside"
    
    # Test 3: Outside bar
    high = np.array([100.0, 105.0])
    low = np.array([95.0, 90.0])
    result = apply_strat_classification_nb(high, low)
    assert result[1] == 3, "Higher high and lower low should be outside"
    
    # Test 4: Exactly equal (should be inside)
    high = np.array([100.0, 100.0])
    low = np.array([95.0, 95.0])
    result = apply_strat_classification_nb(high, low)
    assert result[1] == 1, "Equal OHLC should be inside"
    
    print("All edge case tests passed!")

test_edge_cases()
```

---

## Step 4: Multi-Timeframe Extension

**Once single-timeframe classification is validated**, extend to multiple timeframes:

```python
def classify_multiple_timeframes(data, timeframes=['1M', '1W', '1D']):
    """
    Classify bars across multiple timeframes.
    
    Args:
        data: VectorBT Data object
        timeframes: List of timeframe strings
        
    Returns:
        Dict of {timeframe: classification_series}
    """
    results = {}
    
    for tf in timeframes:
        # Resample to target timeframe
        resampled = data.resample(tf)
        
        # Run classification
        bars = StratBarClassifier.run(
            high=resampled.high,
            low=resampled.low
        )
        
        results[tf] = bars.classification
    
    return results


# Usage
spy_data = vbt.AlpacaData.pull("SPY", start="2023-01-01", timeframe="1d")
mtf_classifications = classify_multiple_timeframes(
    spy_data,
    timeframes=['1M', '1W', '1D']
)

# Check specific date across timeframes
check_date = '2024-01-15'
print(f"\nClassifications for {check_date}:")
for tf, classifications in mtf_classifications.items():
    value = classifications.loc[check_date]
    print(f"{tf}: {decode_classification(value)}")
```

---

## Step 5: Integration with Timeframe Continuity

**After bar classification is validated**, build continuity checker:

```python
def check_timeframe_continuity(classifications_dict, direction='bullish'):
    """
    Check if all timeframes show continuity.
    
    Args:
        classifications_dict: Dict from classify_multiple_timeframes
        direction: 'bullish' or 'bearish'
        
    Returns:
        Boolean series indicating continuity at each timestamp
    """
    target_value = 2 if direction == 'bullish' else -2  # 2U or 2D
    
    # Get the finest granularity timeframe (last in dict)
    base_index = list(classifications_dict.values())[-1].index
    
    # For each timestamp, check if ALL timeframes are aligned
    continuity = pd.Series(True, index=base_index)
    
    for tf, classifications in classifications_dict.items():
        # Align to base index
        aligned = classifications.reindex(base_index, method='ffill')
        
        # Check if this timeframe matches target
        continuity &= (aligned == target_value)
    
    return continuity


# Usage
continuity = check_timeframe_continuity(
    mtf_classifications,
    direction='bullish'
)

print(f"\nDates with full bullish continuity:")
print(continuity[continuity == True].index.tolist())
```

---

## Critical Implementation Notes

### DO NOT:
- ❌ Classify first bar (no previous bar to compare to)
- ❌ Use bar color (green/red) for classification
- ❌ Look ahead (use future bars to classify current)
- ❌ Modify classification logic without re-validating

### DO:
- ✅ Skip first bar (index 0) in all analysis
- ✅ Use strict > and < comparisons (not >=, <=)
- ✅ Validate against manual chart analysis
- ✅ Test edge cases (gaps, equal prices, etc.)
- ✅ Export results to CSV for visual verification

### Performance Considerations:
- Use `@njit` for speed (Numba compilation)
- Process all symbols at once (vectorized)
- Cache results (don't re-calculate unnecessarily)

---

## Troubleshooting

### Issue: "IndexError: index 0 is out of bounds"
**Solution:** Skipping first bar in analysis? Use `result[1:]` not `result[0:]`

### Issue: Classifications don't match TradingView
**Solution:** 
1. Check data alignment (TradingView might use different open/close times)
2. Verify you're comparing same timeframe
3. Export both to CSV and compare side-by-side

### Issue: "All bars classified as 2U or 2D"
**Solution:** Check if using correct comparison operators (> not >=)

### Issue: "Results change when re-running"
**Solution:** Non-deterministic? Should not happen with fixed data. Check for:
- Random seed issues
- Data fetching inconsistencies
- Look-ahead bias in logic

---

## Success Criteria

**Phase 1 is complete when:**
- ✅ Bar classification runs without errors
- ✅ Manual verification shows 95%+ accuracy vs TradingView
- ✅ Edge cases all pass
- ✅ Multi-timeframe classification works
- ✅ Can identify known 3-1-2 and 2-1-2 patterns manually
- ✅ Exported CSV matches visual chart analysis

**Next steps after Phase 1:**
- Build pattern detector (3-1-2, 2-1-2, etc.)
- Add magnitude calculation
- Integrate with Portfolio backtesting
- Test on options data

---

## Quick Start Commands

```bash
# Create new file
touch strat_classifier.py

# Install dependencies (if needed)
pip install vectorbtpro --break-system-packages

# Run validation
python strat_classifier.py

# Run tests
pytest test_strat_classifier.py
```

---

## File Structure

```
project/
├── strat_classifier.py      # Main indicator code
├── test_strat_classifier.py # Validation tests
├── examples/
│   ├── basic_usage.py       # Simple example
│   ├── multi_timeframe.py   # MTF example
│   └── continuity_check.py  # Continuity example
└── validation/
    ├── spy_test_data.csv    # Known good data
    └── expected_results.csv # Expected classifications
```

---

## Next Session Checklist

When resuming in Claude Code:
- [ ] Read this document completely
- [ ] Verify VectorBT Pro is installed
- [ ] Create strat_classifier.py file
- [ ] Copy classification code
- [ ] Run basic test with SPY data
- [ ] Validate against manual analysis
- [ ] Export results for visual verification
- [ ] Run all edge case tests
- [ ] Document any issues encountered

---

**Ready to implement!** Pass this file to Claude Code and begin with Step 1.
