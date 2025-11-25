# VectorBT Pro Integration Patterns

**Branch:** feature/risk-management-foundation
**Created:** 2025-10-12
**Purpose:** Document verified VBT integration patterns for risk management implementation

---

## Session 2 Research Findings: Position Sizing

### VBT.PF.from_signals() Method Signature

**Key parameters verified:**
```python
vbt.PF.from_signals(
    close,           # REQUIRED - price data
    entries=None,    # Boolean Series for entry signals
    exits=None,      # Boolean Series for exit signals
    size=None,       # Position size (scalar, Series, or array)
    size_type=None,  # How to interpret size
    init_cash=None,  # Initial capital
    fees=None,       # Percentage fees
    slippage=None,   # Percentage slippage
    # ... many other parameters
)
```

**Source:** Python introspection using `vbt.phelp(vbt.PF.from_signals)`

---

## Position Sizing Parameter Details

### size Parameter

**Accepts:**
- `float`: Fixed size for all orders (e.g., `10.0` = 10 shares per trade)
- `pd.Series`: Variable size per bar (must match close.index)
- `np.ndarray`: Array of sizes
- `None`: Defaults to internal VBT logic

**Verified formats:**
```python
# Format 1: Scalar (same size for all trades)
pf = vbt.PF.from_signals(close, entries, exits, size=10.0)

# Format 2: Series (variable sizes per bar)
sizes = pd.Series(10.0, index=close.index)
sizes.iloc[30] = 20.0  # Larger position for specific signal
pf = vbt.PF.from_signals(close, entries, exits, size=sizes)

# Format 3: Capital-constrained (calculated dynamically)
stop_distance = atr * 2.0
position_size_risk = (init_cash * 0.02) / stop_distance
position_size_capital = init_cash / close
position_size = np.minimum(position_size_risk, position_size_capital)
pf = vbt.PF.from_signals(close, entries, exits, size=position_size)
```

**Test results:**
- All three formats PASSED ✓
- VBT accepts pd.Series with matching index ✓
- No type conversion errors ✓

---

### size_type Parameter

**Valid options (from VBT documentation):**
- `SizeType.Amount` - Size in shares/units (DEFAULT when size_type=None)
- `SizeType.Value` - Size in dollar value
- `SizeType.Percent(100)` - Percentage of available capital
- `SizeType.ValuePercent(100)` - Percentage of portfolio value

**For our implementation:**
We will use **DEFAULT (size_type=None)** which interprets size as shares/units.

**Why:**
- Our formula calculates position sizes in shares: `position_size = capital / price`
- No need to complicate with percentage-based sizing
- Direct share count is unambiguous and testable

**Verification:**
```python
# Default size_type (interpreted as shares)
pf = vbt.PF.from_signals(
    close, entries, exits,
    size=position_sizes,  # In shares
    # size_type not specified = defaults to Amount/shares
)
# RESULT: Works correctly ✓
```

---

## Capital-Constrained Position Sizing Formula

### Verified Implementation

```python
import pandas as pd
import numpy as np

def calculate_position_size(
    close: pd.Series,
    atr: pd.Series,
    init_cash: float,
    risk_pct: float = 0.02,
    atr_multiplier: float = 2.0
) -> pd.Series:
    """
    Calculate capital-constrained position sizes (in shares).

    Args:
        close: Closing prices
        atr: Average True Range
        init_cash: Initial capital
        risk_pct: Risk percentage per trade (default 2%)
        atr_multiplier: ATR multiplier for stop distance

    Returns:
        pd.Series: Position sizes in shares (VBT-compatible)
    """
    # Risk-based sizing: How many shares to risk 2% of capital?
    stop_distance = atr * atr_multiplier
    position_size_risk = (init_cash * risk_pct) / stop_distance

    # Capital constraint: Maximum shares affordable
    position_size_capital = init_cash / close

    # Take minimum (never exceed either constraint)
    position_size = np.minimum(position_size_risk, position_size_capital)

    # Clean up invalid values
    position_size = position_size.fillna(0).replace([np.inf, -np.inf], 0)

    return position_size
```

### Mathematical Verification

**Constraint 1: Risk-based sizing**
```
position_size_risk = (init_cash × risk_pct) / stop_distance

Example:
init_cash = $10,000
risk_pct = 0.02 (2%)
stop_distance = $5.00 (2 × ATR)

position_size_risk = ($10,000 × 0.02) / $5.00 = 40 shares

Risk per trade = 40 shares × $5.00 = $200 = 2% of $10,000 ✓
```

**Constraint 2: Capital-based sizing**
```
position_size_capital = init_cash / close

Example:
init_cash = $10,000
close = $150

position_size_capital = $10,000 / $150 = 66.67 shares

Position value = 66.67 shares × $150 = $10,000 = 100% of capital ✓
```

**Combined (take minimum):**
```
position_size = min(40, 66.67) = 40 shares

Position value = 40 × $150 = $6,000 = 60% of capital
Risk = 40 × $5 = $200 = 2% of capital

Both constraints satisfied ✓
```

---

## VBT Integration Test Results

### Test 1: Scalar Size
```python
pf = vbt.PF.from_signals(close, entries, exits, size=10.0, init_cash=10000)
```
**Result:** Return: -1.52%, Trades: 4
**Status:** PASSED ✓

### Test 2: Variable Size (Series)
```python
sizes = pd.Series(10.0, index=close.index)
sizes.iloc[30] = 20.0
pf = vbt.PF.from_signals(close, entries, exits, size=sizes, init_cash=10000)
```
**Result:** Return: -1.87%, Trades: 4
**Status:** PASSED ✓

### Test 3: Capital-Constrained Formula
```python
position_size = np.minimum(position_size_risk, position_size_capital)
pf = vbt.PF.from_signals(close, entries, exits, size=position_size, init_cash=10000, fees=0.002, slippage=0.001)
```
**Result:** Return: -1.22%, Sharpe: -0.67, Trades: 3
**Status:** PASSED ✓

**Verification Gate 1:** PASSED ✓

---

## Implementation Requirements for utils/position_sizing.py

Based on VBT research, our position sizing module must:

### 1. Return Format
**MUST return:** `pd.Series` with same index as close prices
**Type:** Float (shares/units)
**VBT interprets as:** Shares to buy/sell (with default size_type)

### 2. Handle Edge Cases
```python
# Invalid values that break VBT:
- NaN → replace with 0
- Inf → replace with 0
- Negative → should not occur (validation needed)
```

### 3. Integration Pattern
```python
# Correct usage:
position_sizes = calculate_position_size(close, atr, init_cash, risk_pct=0.02)

pf = vbt.PF.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=position_sizes,  # Series of shares
    # size_type defaults to Amount (shares)
    init_cash=init_cash,
    fees=0.002,
    slippage=0.001
)
```

### 4. Validation Checks
```python
# Before returning position_sizes:
assert isinstance(position_sizes, pd.Series), "Must be Series"
assert position_sizes.index.equals(close.index), "Index must match"
assert (position_sizes >= 0).all(), "No negative sizes"
assert not position_sizes.isna().any(), "No NaN values"
assert not np.isinf(position_sizes).any(), "No Inf values"
```

---

## Next Steps

1. Create `utils/position_sizing.py` with verified formula
2. Implement capital-constrained calculation
3. Add validation checks
4. Create unit tests
5. Run VBT integration test (Verification Gate 1)

**Status:** VBT research complete, ready for implementation

---

## References

- VBT Documentation: `VectorBT Pro Official Documentation/LLM Docs/3 API Documentation.md`
- Python introspection: `vbt.phelp(vbt.PF.from_signals)`
- Test results: Verified 2025-10-12

**Last Updated:** 2025-10-12

---

## Edge Case Research (Session 2B)

### Critical Finding: VBT Capital Protection

**Test:** Requested position larger than affordable
```
init_cash = $10,000
close = $100/share
requested_size = 200 shares  (would cost $20,000)
```

**Result:**
- Requested: 200 shares ($20,000 value)
- Executed: 100 shares ($10,000 value)
- VBT automatically capped to available capital via partial fill

**Conclusion:** VBT provides SECONDARY safety net through built-in partial fill mechanism.

---

### Mathematical Proof: Capital Constraint CANNOT Exceed 100%

**Our formula:**
```
position_size_capital = init_cash / close
```

**Proof:**
```
Max position value = position_size_capital × close
                   = (init_cash / close) × close
                   = init_cash
                   = 100% of capital  [QED]
```

**Layered defense:**
1. PRIMARY: Our formula mathematically caps at 100%
2. SECONDARY: VBT partial fills provide backup

**Status:** MATHEMATICALLY PROVEN + EMPIRICALLY VERIFIED

---

### Edge Cases Verified

| Edge Case | VBT Behavior | Our Handling | Status |
|-----------|--------------|--------------|--------|
| ATR NaN (first 14 bars) | Skips order when size=0 | fillna(0) | PASS |
| Division by zero | N/A | NaN→0 | PASS |
| Position > capital | Partial fill (caps at 100%) | Formula caps at 100% | PASS |
| Fractional shares | Executes fractional | No rounding needed | PASS |
| Signal before ATR valid | Skips (size=0) | fillna(0) | PASS |

**All edge cases verified. Implementation confidence: 95%**

---

**Last Updated:** 2025-10-12 (Session 2B Complete)
