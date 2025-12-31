# Session EQUITY-37: Options Backtest Pattern Detection Bugs

**Date:** 2025-12-29
**Status:** INCOMPLETE - Critical bugs discovered requiring fixes
**Previous Session:** EQUITY-36 (trade audit system)

## Executive Summary

During validation of the unified options backtest (`scripts/backtest_strat_options_unified.py`), we discovered **critical pattern detection bugs** that make backtest results unreliable and inconsistent with paper trading.

## Critical Bugs Discovered

### Bug 1: Pattern Ordering (CRITICAL)

**Location:** `strat/tier1_detector.py:302-324`

**Problem:** The `Tier1Detector.detect_patterns()` method appends patterns in TYPE order, not chronological order:
```python
all_signals = []
# 3-1-2 patterns added first (all 6 years)
signals_312 = self._detect_312(...)
all_signals.extend(signals_312)
# Then 2-1-2 patterns (all 6 years)
signals_212 = self._detect_212(...)
all_signals.extend(signals_212)
# Then 2-2 patterns (all 6 years)
signals_22 = self._detect_22(...)
all_signals.extend(signals_22)
```

**Impact:** When using `--limit 20` for testing, we got ALL 20 3-1-2 patterns spanning 6 years (2019-2024) and ZERO 2-1-2 or 2-2 patterns. The first 20 patterns in the list are:
```
1.  2019-01-14: 3-1-2D
2.  2019-03-06: 3-1-2D
3.  2019-12-24: 3-1-2D
...
20. 2024-12-24: 3-1-2U
21. 2019-01-25: 2D-1-2U  <-- 2-1-2 patterns start here!
```

**Fix Required:** Sort patterns chronologically after detection:
```python
patterns.sort(key=lambda x: x['timestamp'])
```

### Bug 2: Missing Pattern Types (CRITICAL)

**Location:** `scripts/backtest_strat_options_unified.py:312-377` (uses Tier1Detector)

**Problem:** The backtest only includes 3 pattern types via Tier1Detector:
- 3-1-2 (Up/Down)
- 2-1-2 (all variants)
- 2-2 (Up only, Down excluded)

**Missing from backtest but traded in paper trading:**
- **3-2 patterns** (outside bar followed by directional)
- **3-2-2 patterns** (outside-directional-reversal)
- **2-2 Down patterns** (excluded via `include_22_down=False`)

**Paper trading scanner patterns:** `['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']`

### Bug 3: Detection Method Mismatch

**Problem:** Paper trading and backtest use different detection methods:

| Aspect | Paper Trading Scanner | Backtest |
|--------|----------------------|----------|
| Detection | Direct numba functions | Tier1Detector wrapper |
| Pattern types | ALL 5 types | Only 3 types |
| 2-2 Down | Included | Excluded |
| 3-2, 3-2-2 | Included | Missing |

## Pattern Statistics (SPY 1D, 2019-2024)

From our debug analysis:
```
Total patterns detected by Tier1Detector: 290

Pattern breakdown:
  2D-2U (2-2 Up): 181     <- Most common!
  2U-1-2D: 29
  2U-1-2U: 29
  2D-1-2U: 22
  3-1-2U: 12
  2D-1-2D: 9
  3-1-2D: 8
```

**Key insight:** 2-2 Up patterns are 62% of all patterns (181/290), but our first test only showed 3-1-2 patterns (20/290 = 7%).

## Files Involved

### Primary Files
- `scripts/backtest_strat_options_unified.py` - Unified backtest script (needs pattern detection fix)
- `strat/tier1_detector.py` - Pattern detector wrapper (has ordering bug)
- `strat/paper_signal_scanner.py` - Paper trading scanner (reference implementation)

### Pattern Detection (Reference)
- `strat/pattern_detector.py` - Core numba pattern detectors
- `strat/bar_classifier.py` - Bar classification (1, 2U, 2D, 3)

## Recommended Fix

Replace `Tier1Detector` usage in backtest with direct pattern detection matching `PaperSignalScanner._detect_patterns`:

```python
def detect_patterns_inline(data, timeframe):
    """Detect ALL STRAT patterns (matching paper trading scanner)."""
    ALL_PATTERNS = ['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']

    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    classifications = classify_bars_nb(high, low)

    patterns = []
    for pattern_type in ALL_PATTERNS:
        if pattern_type == '2-2':
            result = detect_22_patterns_nb(classifications, high, low)
        elif pattern_type == '3-2':
            result = detect_32_patterns_nb(classifications, high, low)
        elif pattern_type == '3-2-2':
            result = detect_322_patterns_nb(classifications, high, low)
        elif pattern_type == '2-1-2':
            result = detect_212_patterns_nb(classifications, high, low)
        elif pattern_type == '3-1-2':
            result = detect_312_patterns_nb(classifications, high, low)
        # ... process patterns

    # CRITICAL: Sort chronologically
    patterns.sort(key=lambda x: x['timestamp'])
    return patterns
```

## Verified Working Components

1. **Bar classification** - `classify_bars_nb` produces correct 1/2U/2D/3 classifications (verified against chart)
2. **Individual pattern detectors** - `detect_312_patterns_nb`, `detect_212_patterns_nb`, etc. work correctly
3. **ThetaData integration** - Options pricing queries work
4. **Alpaca data** - Split-adjusted pricing works
5. **Gap-through detection** - EQUITY-36 fix working
6. **EOD exit logic** - EQUITY-36 fix working

## Test Results (Before Fix)

With `--limit 20` (broken due to ordering bug):
- 18 trades executed (all 3-1-2 patterns only)
- 16.7% win rate
- -$6,665 P&L
- Date range: 2019-01-14 to 2023-01-05

**These results are INVALID** because they only tested 3-1-2 patterns.

## Next Steps

1. **Fix pattern detection** in `backtest_strat_options_unified.py`:
   - Use direct numba functions (like PaperSignalScanner)
   - Include ALL 5 pattern types
   - Sort chronologically by timestamp

2. **Re-run full backtest** without `--limit` flag

3. **Compare pattern distribution** between backtest and paper trading expectations

4. **Validate sample trades** against actual charts

## Key Data Points

- **SPY prices:** SPLIT-ADJUSTED (not dividend-adjusted) via Alpaca `adjustment='split'`
- **Backtest period:** 2019-01-01 to 2025-01-01 (6 years)
- **Total patterns (SPY 1D):** ~290 with current detection, likely more with 3-2/3-2-2

## OpenMemory Context

Previous sessions referenced:
- Session 83K-19: Price mismatch bug (dividend vs split adjustment)
- Session 83K-21: Entry exceeds target skip
- Session 83K-55/56: Entry timing fixes
- Session 83K-64: DTE/max_holding alignment
- EQUITY-36: Gap-through, EOD exit, 1H R:R fixes

## Command to Continue

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace
# Read this handoff first, then fix pattern detection
```
