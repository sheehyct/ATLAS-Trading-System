# CRITICAL: Position Sizing Method Clarification for Week 1

**Date:** October 14, 2025  
**Priority:** BLOCKER - Must resolve before Week 1 implementation  
**Audience:** Claude Code (implementation agent)  
**Author:** Claude (project analysis and architecture)

---

## Executive Summary

**DECISION: Use ATR-based position sizing for Strategy 2 (ORB) in Week 1.**

There is a documentation mismatch that must be resolved immediately. The clarification document received suggests implementing Yang-Zhang or Garman-Klass semi-volatility for Week 1, but this is **incorrect for Strategy 2 (ORB)**, which requires ATR-based position sizing.

**This document provides the definitive answer and implementation path.**

---

## 🔴 CRITICAL: VectorBT Pro Compliance Requirement

**ALL CODE MUST BE VECTORBT PRO COMPATIBLE**

This implementation uses **VectorBT Pro** as the backtesting framework. All position sizing code MUST:

- ✅ Use VectorBT Pro native data structures (pandas Series/DataFrame)
- ✅ Be vectorized (operate on entire arrays, not loops)
- ✅ Return position sizes in VectorBT Pro expected format
- ✅ Work with `vbt.Portfolio.from_signals()` size parameter
- ✅ Handle VectorBT Pro's internal indexing and alignment

**Reference Documentation:**
```
C:\Strat_Trading_Bot\vectorbt-workspace\VectorBT Pro Official Documentation
```

**If unsure about VectorBT Pro implementation:**
1. STOP coding
2. Consult the official documentation at the path above
3. Verify your approach matches VectorBT Pro patterns
4. Test with small synthetic data first

**Common VectorBT Pro Requirements:**
- Position sizes must be pandas Series with matching index to price data
- Use `size_type='amount'` for share counts or `size_type='percent'` for percentages
- Stops must be specified via `sl_stop` parameter (not manual exit signals)
- All calculations must preserve index alignment

**This is NON-NEGOTIABLE. VectorBT Pro incompatible code will fail at runtime.**

---

## The Confusion: Three Position Sizing Methods

Your research documents contain **three distinct position sizing approaches**, each designed for **different strategy types**:

### Method A: Downside Semi-Deviation (Article #6)
- **Formula:** `downside_vol = returns[returns < 0].rolling(60).std() * sqrt(252)`
- **Use Case:** Return-based position sizing
- **Strategy Type:** NOT in current implementation plan
- **When to Use:** Potentially for future enhancements (not Week 1-9)

### Method B: Garman-Klass Semi-Volatility (Article #6)
- **Formula:** `GK_Var = 0.5 × [ln(High/Low)]² - (2ln(2) - 1) × [ln(Close/Open)]²`
- **Use Case:** **Multi-asset momentum portfolio position sizing** (inverse volatility weighting)
- **Strategy Type:** Momentum portfolios with 20+ stocks, sector rotation
- **When to Use:** Week 4-6+ when building momentum portfolio strategies
- **Results:** Sortino 1.44, Sharpe 1.07, 24.75% CAGR
- **NOT FOR:** Single-asset intraday breakout strategies

### Method C: Yang-Zhang Volatility (Article #8)
- **Formula:** `Overnight + k×Open-Close + (1-k)×Rogers-Satchell`
- **Use Case:** **GMM regime detection feature engineering** (not position sizing)
- **Strategy Type:** Gaussian Mixture Model regime classification
- **When to Use:** Week 7-9 when implementing regime detection
- **Results:** 1.00 Sharpe, -14.68% max DD vs -34.10% SPY
- **NOT FOR:** Direct position sizing

### Method D: ATR-Based Position Sizing (Session 2B + HANDOFF.md)
- **Formula:** `position_size = (capital × risk%) / (ATR × multiplier)`
- **Use Case:** **Strategies with ATR-based stop losses** (Turtle Trading, ORB, breakouts)
- **Strategy Type:** Intraday breakouts, trend following with ATR stops
- **When to Use:** **Week 1 for Strategy 2 (ORB)** ← THIS IS THE ANSWER
- **Results:** 95% confidence from Session 2B testing
- **Critical Bug Fix:** Must include capital constraint `min(risk_based, capital_based)`

---

## The Answer: What to Implement for Week 1

### For Strategy 2 (ORB) - Week 1 Implementation

**USE: Method D (ATR-Based Position Sizing with Capital Constraint)**

**Why This Is Correct:**

1. **ORB uses ATR-based stops:** Strategy 2 uses 2.5× ATR stop losses
2. **Position sizing must match stops:** You size based on how you exit
3. **Already tested and validated:** Session 2B confirmed 95% confidence
4. **Documented in HANDOFF.md:** This is the approved approach
5. **The bug was capital constraints, not the volatility estimator**

**From Your HANDOFF.md:**
```markdown
Strategy 2: Opening Range Breakout (ORB) - READY TO START

Critical Implementation Requirements:
1. Volume confirmation 2.0× (hardcoded in Phase 1.3)
2. Capital-constrained position sizing (from fix)  ← ATR-BASED
3. NO signal exits (only EOD + stops)

Phase 1.4: Exit Logic (Day 2, 4 hours)
- EOD exit at 3:55 PM ET
- 2.5× ATR stops  ← THIS IS WHY WE USE ATR-BASED SIZING
- NO signal exits
```

---

## Common VectorBT Pro Pitfalls to Avoid

### ❌ Pitfall 1: Using Python min() Instead of np.minimum()

**WRONG:**
```python
position_size = min(position_size_risk, position_size_capital)  # Only works for scalars
```

**CORRECT:**
```python
position_size = np.minimum(position_size_risk, position_size_capital)  # Works for Series
```

**Why:** Python's `min()` doesn't work on pandas Series. Use `np.minimum()` for element-wise minimum.

---

### ❌ Pitfall 2: Losing Index Alignment

**WRONG:**
```python
position_sizes = []
for i in range(len(close)):
    size = calculate_single_position(close.iloc[i], atr.iloc[i])
    position_sizes.append(size)
return position_sizes  # Returns list, not Series with index
```

**CORRECT:**
```python
# Vectorized operation maintains index
position_size = np.minimum(
    (init_cash * risk_pct) / (atr * atr_multiplier),
    init_cash / close
)
return position_size  # Returns Series with original index
```

**Why:** VectorBT Pro requires index-aligned Series for proper backtesting.

---

### ❌ Pitfall 3: Wrong size_type Parameter

**WRONG:**
```python
pf = vbt.Portfolio.from_signals(
    size=0.02,           # Trying to size by 2% of capital
    size_type='amount'   # But 'amount' expects share count
)
```

**CORRECT - Option 1 (Shares):**
```python
pf = vbt.Portfolio.from_signals(
    size=position_sizes,  # Number of shares (e.g., 16 shares)
    size_type='amount'    # Matches our ATR-based calculation
)
```

**CORRECT - Option 2 (Percentage):**
```python
pf = vbt.Portfolio.from_signals(
    size=0.02,           # 2% of capital
    size_type='percent'  # Explicitly percent
)
```

**Why:** Our ATR-based function returns share counts, so use `size_type='amount'`.

---

### ❌ Pitfall 4: Manual Stop-Loss Exits

**WRONG:**
```python
# Trying to manually implement stops
stop_hit = low <= (entry_price - stop_distance)
exits = eod_exit | stop_hit  # Manually combining exits
```

**CORRECT:**
```python
pf = vbt.Portfolio.from_signals(
    entries=long_entries,
    exits=eod_exit,              # Only signal exits
    sl_stop=atr_series * 2.5,    # Let VectorBT Pro handle stops
    size_type='amount'
)
```

**Why:** VectorBT Pro's `sl_stop` parameter handles intrabar stops correctly. Manual implementation misses fills.

---

### ❌ Pitfall 5: Not Handling NaN Values

**WRONG:**
```python
# Assuming ATR is always valid
position_size = (init_cash * 0.02) / (atr * 2.5)
# If ATR has NaN, position_size will have NaN
```

**CORRECT:**
```python
# Handle edge cases
atr_clean = atr.fillna(method='ffill')  # Forward fill NaN
atr_clean = atr_clean.replace(0, np.nan).fillna(1.0)  # Replace zeros

position_size = (init_cash * 0.02) / (atr_clean * 2.5)
position_size = position_size.fillna(0)  # No position if still NaN
```

**Why:** Real data has gaps, halts, and edge cases. Defensive programming prevents crashes.

---

## VectorBT Pro Integration Checklist

Before submitting code, verify:

- [ ] All calculations use pandas/numpy operations (no Python loops)
- [ ] Input is pandas Series with DatetimeIndex
- [ ] Output is pandas Series with same index as input
- [ ] Used `np.minimum()` not `min()` for element-wise operations
- [ ] Position sizes tested with `vbt.Portfolio.from_signals()`
- [ ] Correct `size_type` parameter ('amount' for shares)
- [ ] Stop losses use `sl_stop` parameter (not manual exits)
- [ ] Edge cases handled (NaN, zero, negative values)
- [ ] No warnings or errors from VectorBT Pro
- [ ] Consult documentation if any uncertainty

**Documentation Reference:**
```
C:\Strat_Trading_Bot\vectorbt-workspace\VectorBT Pro Official Documentation
```

---

## Implementation: Create utils/position_sizing.py

### Week 1 Deliverable (ATR-Based)

```python
"""
Position Sizing Utilities for Algorithmic Trading Strategies

This module provides position sizing functions matched to stop-loss methodologies.
Each strategy type requires a different position sizing approach.

CRITICAL: All functions are designed for VectorBT Pro compatibility.
- Input: pandas Series with DatetimeIndex (aligned to price data)
- Output: pandas Series with same index (VectorBT Pro requirement)
- Operations: Fully vectorized (no loops, VectorBT Pro best practice)

Reference: C:\Strat_Trading_Bot\vectorbt-workspace\VectorBT Pro Official Documentation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def calculate_position_size_atr(
    init_cash: float,
    close: Union[float, pd.Series],
    atr: Union[float, pd.Series],
    atr_multiplier: float = 2.5,
    risk_pct: float = 0.02
) -> Tuple[Union[float, pd.Series], Union[float, pd.Series], Union[bool, pd.Series]]:
    """
    Calculate position size for strategies with ATR-based stop losses.
    
    *** VECTORBT PRO COMPATIBLE ***
    
    Use this function for:
    - Opening Range Breakout (ORB)
    - Turtle Trading systems
    - Any strategy with ATR-based stops
    
    The critical fix from Phase 0: Adds capital constraint to prevent
    position sizes exceeding 100% of capital.
    
    VectorBT Pro Integration:
        >>> # Example usage with VectorBT Pro
        >>> position_sizes, actual_risks, constrained = calculate_position_size_atr(
        ...     init_cash=10000,
        ...     close=data['Close'],  # pandas Series
        ...     atr=atr_series,       # pandas Series with matching index
        ...     atr_multiplier=2.5,
        ...     risk_pct=0.02
        ... )
        >>> 
        >>> # Use with VectorBT Pro Portfolio
        >>> pf = vbt.Portfolio.from_signals(
        ...     close=data['Close'],
        ...     entries=long_entries,
        ...     exits=long_exits,
        ...     size=position_sizes,        # Our output here
        ...     size_type='amount',          # Shares, not percentage
        ...     init_cash=10000,
        ...     sl_stop=atr_series * 2.5,   # ATR-based stops
        ...     fees=0.001
        ... )
    
    Args:
        init_cash: Account capital ($10,000 typically)
        close: Current price or price series (pandas Series for VectorBT Pro)
        atr: Average True Range value or series (pandas Series for VectorBT Pro)
        atr_multiplier: Stop distance in ATR units (2.5 for ORB, 2.0 for Turtle)
        risk_pct: Risk per trade as decimal (0.02 = 2%)
    
    Returns:
        tuple: (position_size_shares, actual_risk_pct, constrained_flag)
            - position_size_shares: Number of shares to buy (pandas Series if input is Series)
            - actual_risk_pct: Realized risk (may be < target if constrained)
            - constrained_flag: True if capital constraint was applied
            
        All return values maintain index alignment with input Series (VectorBT Pro requirement)
    
    Example:
        >>> size, risk, constrained = calculate_position_size_atr(
        ...     init_cash=10000,
        ...     close=480,
        ...     atr=5.0,
        ...     atr_multiplier=2.5,
        ...     risk_pct=0.02
        ... )
        >>> print(f"Buy {size:.0f} shares, actual risk {risk:.2%}")
        Buy 16 shares, actual risk 2.00%
    """
    # Calculate stop distance in dollars
    stop_distance = atr * atr_multiplier
    
    # Risk-based position size (what we'd like to buy for 2% risk)
    position_size_risk = (init_cash * risk_pct) / stop_distance
    
    # Capital-based maximum (can't buy more than 100% of capital)
    position_size_capital = init_cash / close
    
    # Take minimum (capital constraint fix from Phase 0)
    # Using np.minimum for vectorized operation (VectorBT Pro compatible)
    position_size = np.minimum(position_size_risk, position_size_capital)
    
    # Calculate actual risk achieved
    actual_risk = (position_size * stop_distance) / init_cash
    
    # Flag if constrained by capital (not risk)
    # For Series, this creates a boolean Series (VectorBT Pro compatible)
    constrained = position_size == position_size_capital
    
    # VectorBT Pro Note: If input is pandas Series, output will be Series with same index
    # This ensures proper alignment in vbt.Portfolio.from_signals()
    
    return position_size, actual_risk, constrained


def calculate_position_size_garman_klass(
    init_cash: float,
    close: pd.Series,
    semi_vol: pd.Series,
    target_vol: float = 0.15,
    max_position: float = 1.0
) -> pd.Series:
    """
    Calculate position size using Garman-Klass semi-volatility (inverse vol weighting).
    
    Use this function for:
    - Multi-asset momentum portfolios (20+ stocks)
    - Sector rotation strategies
    - Inverse volatility weighted portfolios
    
    DO NOT USE for:
    - Single-asset strategies
    - Intraday breakout strategies (use ATR-based instead)
    - Strategies with ATR-based stops
    
    This is from Article #6 research and achieves Sortino 1.44.
    
    Implementation: Week 4-6+ (NOT Week 1)
    
    Args:
        init_cash: Account capital
        close: Price series for asset
        semi_vol: Garman-Klass semi-volatility series (annualized)
        target_vol: Target portfolio volatility (0.15 = 15%)
        max_position: Maximum position size as fraction of capital (1.0 = 100%)
    
    Returns:
        pd.Series: Position sizes as fraction of capital
    
    Note:
        This function will be implemented in Week 4-6 when building momentum
        portfolio strategies. Not needed for Strategy 2 (ORB).
    """
    # Inverse volatility weighting
    raw_weight = target_vol / semi_vol
    
    # Normalize to sum to 1.0 across all assets
    # (This requires portfolio-level calculation, not shown here)
    
    # Apply maximum position constraint
    position_size = np.minimum(raw_weight, max_position)
    
    return position_size


def calculate_position_size_fixed_fraction(
    init_cash: float,
    close: Union[float, pd.Series],
    fraction: float = 0.02
) -> Union[float, pd.Series]:
    """
    Simple fixed-fraction position sizing (2% of capital per position).
    
    Use this function for:
    - Equal-weight portfolios
    - Simple baseline testing
    - When no volatility data available
    
    DO NOT USE for production strategies (too simplistic).
    
    Args:
        init_cash: Account capital
        close: Current price or price series
        fraction: Fraction of capital per position (0.02 = 2%)
    
    Returns:
        Position size in shares
    """
    return (init_cash * fraction) / close


# Edge case handling utilities

def validate_position_size(
    position_size: Union[float, pd.Series],
    init_cash: float,
    close: Union[float, pd.Series],
    max_pct: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate position sizes for common issues.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for negative sizes
    if np.any(position_size < 0):
        return False, "Negative position sizes detected"
    
    # Check for NaN or Inf
    if np.any(~np.isfinite(position_size)):
        return False, "NaN or Inf position sizes detected"
    
    # Check capital constraint
    position_value = position_size * close
    if np.any(position_value > init_cash * max_pct):
        return False, f"Position exceeds {max_pct*100:.0f}% of capital"
    
    return True, "Valid"
```

---

## Gate 1 Verification Criteria (Week 1)

After implementing ATR-based position sizing, run Strategy 2 (ORB) backtest and verify:

### Technical Functionality
- ✅ Code runs without VBT errors (zero tolerance)
- ✅ VBT accepts position sizing format
- ✅ No runtime errors or NaN values (zero tolerance)

### Position Size Constraints
- ✅ Mean position size: 10-30% range (STRICT)
- ✅ Max position size: ≤100% (zero violations allowed)
- ✅ Position sizes are never negative (zero tolerance)

### Risk-Adjusted Performance
- ✅ Sortino Ratio improvement: +0.3 MINIMUM vs current Strategy 2
  - **This criterion applies WHEN comparing position sizing methods**
  - **For Week 1:** We're fixing the capital constraint bug, not comparing volatility estimators
  - **Success:** Mean position size in 10-30% range with no violations

### Edge Case Handling
- ✅ Handles zero ATR gracefully (use minimum stop distance)
- ✅ Handles extremely low ATR (capital constraint activates)
- ✅ Handles extremely high ATR (position sizes very small but valid)

---

## When to Use Each Method: Decision Tree

```
Are you building Strategy 2 (ORB) in Week 1?
│
├─ YES → Use ATR-based position sizing (Method D)
│         - ORB uses 2.5× ATR stops
│         - Position sizing must match stop method
│         - Already tested in Session 2B
│
└─ NO → What are you building?
    │
    ├─ Multi-asset momentum portfolio (20+ stocks)?
    │   └─ YES → Use Garman-Klass semi-vol (Method B)
    │             - Week 4-6+ implementation
    │             - Inverse volatility weighting
    │             - Article #6: Sortino 1.44
    │
    ├─ Regime detection system (GMM)?
    │   └─ YES → Use Yang-Zhang vol (Method C)
    │             - Week 7-9 implementation
    │             - Feature engineering (not position sizing)
    │             - Article #8: 1.00 Sharpe
    │
    └─ Simple equal-weight portfolio?
        └─ YES → Use fixed-fraction (Method E)
                  - Simple baseline only
                  - Not for production
```

---

## Timeline: When Each Method Gets Implemented

### Week 1-2: ATR-Based Position Sizing
- **Implement:** `calculate_position_size_atr()`
- **Test:** Strategy 2 (ORB) with capital constraints
- **Verify:** Gate 1 PASS criteria
- **Status:** CURRENT PRIORITY

### Week 3-4: Mean Reversion Fix
- **Use:** ATR-based position sizing (already implemented)
- **No new position sizing methods needed**

### Week 4-6: Momentum Portfolio Strategies (FUTURE)
- **Implement:** `calculate_position_size_garman_klass()`
- **Implement:** Garman-Klass variance calculator
- **Test:** Multi-asset portfolios (20+ stocks)
- **Verify:** Sortino improvement vs equal-weight

### Week 7-9: GMM Regime Detection (FUTURE)
- **Implement:** `calculate_yang_zhang_volatility()`
- **Use:** As feature for regime classification
- **NOT for position sizing directly**

---

## Critical Error to Avoid

**DO NOT implement Garman-Klass or Yang-Zhang volatility for Strategy 2 (ORB) in Week 1.**

**Why this would be wrong:**

1. **ORB uses ATR-based stops** (2.5× ATR)
   - You MUST size positions based on your stop-loss method
   - Using semi-volatility for sizing would be mismatched to ATR stops

2. **Garman-Klass is for multi-asset portfolios**
   - Designed for inverse volatility weighting across 20+ stocks
   - Strategy 2 is single-asset (SPY only)
   - Wrong tool for the job

3. **Yang-Zhang is for regime detection**
   - Used as feature input to GMM classifier
   - Not designed for direct position sizing
   - Completely different use case

4. **Already tested ATR approach**
   - Session 2B validated ATR-based sizing with 95% confidence
   - HANDOFF.md documents this as approved path
   - No reason to change approaches

---

## What Caused This Confusion

Looking at the clarification document received, it appears there was confusion about:

1. **"Your framework mentions Yang-Zhang + Garman-Klass"**
   - These ARE in the research documents (correctly identified)
   - But they're for DIFFERENT strategies (not ORB)
   - The framework has multiple strategies across Week 1-9

2. **"Gate 1 requires semi-volatility position sizing"**
   - This is incorrect
   - Gate 1 requires CORRECT position sizing for Strategy 2 (ORB)
   - That means ATR-based with capital constraints

3. **Mixing research articles with implementation timeline**
   - Article #6 (Garman-Klass) → Week 4-6 momentum portfolios
   - Article #8 (Yang-Zhang) → Week 7-9 regime detection
   - NOT for Week 1 Strategy 2 (ORB)

---

## Action Items for Claude Code

### ✅ CORRECT Implementation Path

1. **Create `utils/position_sizing.py`** with:
   - `calculate_position_size_atr()` (full implementation above)
   - Capital constraint: `min(risk_based, capital_based)`
   - Edge case handling for zero/low ATR

2. **Add unit tests** (`tests/test_position_sizing.py`):
   - Test capital constraint activates when needed
   - Test no position exceeds 100% of capital
   - Test edge cases (zero ATR, extreme ATR, negative close)
   - Test vectorized operation (pd.Series input)

3. **Integration test** with Strategy 2 (ORB):
   - Run backtest with corrected position sizing
   - Verify mean position size: 10-30% range
   - Verify max position size: ≤100%
   - Verify no NaN or Inf values

4. **Document Gate 1 results**:
   - Position size distribution (mean, max, percentiles)
   - Actual risk per trade (should be ~2% when not constrained)
   - Percentage of trades constrained by capital
   - Performance metrics (Sharpe, Sortino, etc.)

### ❌ INCORRECT Implementation Path (DO NOT DO THIS)

1. ❌ Implementing Garman-Klass semi-volatility for ORB
2. ❌ Implementing Yang-Zhang volatility for ORB
3. ❌ Mixing position sizing methods (ATR + semi-vol hybrid)
4. ❌ Comparing semi-vol to ATR for ORB (wrong comparison)

---

## Summary: The Definitive Answer

**Question:** Which position sizing method for Week 1 / Strategy 2 (ORB)?

**Answer:** ATR-based position sizing with capital constraint.

**Implementation:** Create `utils/position_sizing.py` with `calculate_position_size_atr()`.

**Verification:** Gate 1 verifies mean position size 10-30%, max ≤100%.

**Future Work:**
- Week 4-6: Garman-Klass for momentum portfolios
- Week 7-9: Yang-Zhang for regime detection

**Critical:** Do not implement the wrong method for the wrong strategy type.

---

## Approval and Sign-Off

**This document represents the definitive implementation path for Week 1.**

Any questions or concerns about this decision should be raised BEFORE implementation begins.

**Approved by:** Claude (Architecture & Research Analysis)  
**For implementation by:** Claude Code (Implementation Agent)  
**Date:** October 14, 2025  
**Status:** READY FOR IMPLEMENTATION

---

**Questions? Review the decision tree above or trace through the HANDOFF.md documentation for Strategy 2 (ORB) requirements.**
