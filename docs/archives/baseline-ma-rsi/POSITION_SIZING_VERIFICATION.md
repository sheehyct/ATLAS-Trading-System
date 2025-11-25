# Position Sizing Verification Results - Strategy 1

**Date:** 2025-10-11
**Strategy:** Strategy 1 Baseline (MA200 + RSI)
**Test Period:** 2021-10-11 to 2025-10-08 (4 years, 1003 daily bars)
**Status:** **CRITICAL ISSUE IDENTIFIED**

---

## Executive Summary

**VERDICT: FAIL** - Position sizing formula is mathematically correct but produces oversized positions that exceed available capital.

**Critical Finding:** 83% of calculated position sizes exceed 50% of capital, with mean position size of 81.8% and maximum of 142.6% of capital. This is physically impossible to execute and explains the 43% discrepancy in actual vs expected losses (-2.86% vs -2.00%).

**Root Cause:** The formula `position_size = (capital * risk%) / stop_distance` correctly calculates shares needed for target risk, but doesn't constrain position size by available capital. When ATR is small (tight stops), it calculates huge position sizes.

**Impact on Strategy 2:** **MUST FIX** before implementing Strategy 2 (ORB), or the same bug will be inherited.

---

## Verification Results

### Summary Table

| Metric | Value | Status |
|--------|-------|--------|
| Theoretical Risk | 2.00% | PASS (formula correct) |
| Actual Avg Loss | -2.86% | FAIL (43% higher than expected) |
| Discrepancy | 4.86% | FAIL (> 0.5% threshold) |
| Mean Position Size | 81.8% of capital | FAIL (should be 10-30%) |
| Positions > 50% capital | 832 (83.0%) | CRITICAL |
| Max Position Size | 142.6% of capital | IMPOSSIBLE |
| ATR as % of Price | 1.39% | PASS (reasonable) |

---

## Detailed Analysis

### ATR Statistics

```
Mean ATR: $6.55
ATR as % of price: 1.39%
Min ATR: $3.29 | Max ATR: $20.34

Stop Distance (2x ATR):
Mean: $13.11
As % of price: 2.78%
```

**Analysis:** ATR values are reasonable (1.39% of price is normal for SPY). The 2x ATR stop distance averages 2.78% of price, which is appropriate. **This is not the problem.**

### Position Sizing Statistics

```
Mean position size: 17 shares
Mean position value: $8,181.47
Mean position as % of portfolio: 81.8%
Min: 0.0% | Max: 142.6%

Positions > 50% of capital: 832 (83.0%)
Positions < 5% of capital: 14 (1.4%)
Zero positions: 14 (1.4%)
```

**Analysis:** **THIS IS THE PROBLEM.** The formula calculates an average of 17 shares, which at SPY ~$480 = $8,181 (81.8% of $10k capital). This is far too large.

**Why This Happens:**

Example calculation:
- init_cash = $10,000
- risk = 2% = $200
- ATR = $5 (low volatility)
- stop_distance = $5 × 2.0 = $10
- **position_size = $200 / $10 = 20 shares**
- SPY price = $480
- **Position value = 20 × $480 = $9,600 (96% of capital!)**

When ATR is small, the stop distance is small, so the formula calculates a large number of shares to achieve 2% risk. But buying that many shares can exceed available capital.

### Theoretical vs Actual Risk

```
Theoretical Risk Per Trade:
  Expected: 2.00%
  Actual: 2.00%
  Discrepancy: 0.00%
```

**Analysis:** The mathematical formula is **CORRECT**. If we could actually execute the calculated position sizes, we would achieve exactly 2% risk per trade. **The formula logic is sound.**

### Backtest Results

```
Total trades: 45
Losing trades: 14

Theoretical avg loss (from formula): 2.00%
Actual avg loss (from backtest): -2.86%
Discrepancy: 4.86%
```

**Analysis:** When VectorBT executes the backtest, it tries to buy the calculated shares but is constrained by available capital. This creates a mismatch:
- Formula says: "Buy 20 shares for 2% risk"
- VectorBT says: "You only have $10k, can only buy 20 shares if SPY < $500"
- If SPY = $480, it buys 20 shares (96% of capital)
- If stop hits (-2.78%), loss = 0.96 × 0.0278 = **-2.67%**
- Add fees (0.2%) + slippage (0.1%) = **-2.97%**

This explains the -2.86% actual loss vs -2.00% expected.

### Extreme Losses

```
Min loss: -7.12%
Max loss: -0.19%
Std dev: 2.04%

Found 2 trades with losses > 5%
```

**Analysis:** Some trades lost over 7% (3.5× the expected 2%). These are likely cases where:
1. Position size was 100%+ of capital (forced into margin or constrained by VectorBT)
2. Stop slipped beyond 2× ATR due to gap down
3. Multiple positions were open simultaneously (portfolio heat > 2%)

---

## Root Cause Explanation

### The Formula (Correct Math)

```python
stop_distance = atr * 2.0  # ATR-based stop
position_size = (init_cash * 0.02) / stop_distance  # Shares for 2% risk
```

**This formula is mathematically correct** for calculating the number of shares needed to risk 2% if the stop is hit.

**Proof:**
- If stop distance = $10 and we buy 20 shares
- Stop loss = $10 × 20 shares = $200
- $200 / $10,000 capital = 2% risk ✓

### The Problem (Missing Constraint)

The formula doesn't check:
```python
position_value = position_size * current_price
if position_value > init_cash:
    # PROBLEM: Can't buy this many shares!
```

**When ATR is low:**
- stop_distance is small
- Formula calculates large position_size
- position_value exceeds available capital
- Impossible to execute

**Example:**
- SPY = $480, ATR = $3.29 (min from data)
- stop_distance = $3.29 × 2.0 = $6.58
- position_size = ($10,000 × 0.02) / $6.58 = **30.4 shares**
- position_value = 30.4 × $480 = **$14,592**
- **146% of capital - IMPOSSIBLE!**

This matches our observed max position of 142.6%.

---

## The Solution

### Option 1: Add Capital Constraint (Recommended)

```python
stop_distance = atr * 2.0
position_size_risk = (init_cash * 0.02) / stop_distance  # Risk-based

# NEW: Cap by available capital
position_size_capital = init_cash / close  # Capital-based
position_size = min(position_size_risk, position_size_capital)

# Recalculate actual risk with constrained position
actual_risk = (position_size * stop_distance) / init_cash
```

**Pros:**
- Ensures position never exceeds 100% of capital
- Still targets 2% risk when possible
- Automatically scales down in low-volatility environments

**Cons:**
- In low volatility, actual risk < 2% (safer but lower returns)

### Option 2: Use Percentage of Capital (Alternative)

```python
# Always risk a fixed % of capital
position_size = (init_cash * 0.20) / close  # 20% of capital in shares
stop_distance = (init_cash * 0.02) / position_size  # Adjust stop to achieve 2% risk
```

**Pros:**
- Simpler logic
- Never exceeds capital

**Cons:**
- Changes stop distance instead of position size
- May create stops too tight or too wide

### Option 3: Kelly Fraction with Capital Constraint (Advanced)

```python
# Calculate Kelly fraction based on strategy win rate and R:R
kelly_full = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
kelly_fraction = kelly_full * 0.25  # Use quarter-Kelly for safety

position_size_kelly = (init_cash * kelly_fraction) / close
position_size_risk = (init_cash * 0.02) / stop_distance
position_size = min(position_size_kelly, position_size_risk, init_cash / close)
```

**Pros:**
- Theoretically optimal
- Adapts to strategy performance

**Cons:**
- More complex
- Requires ongoing win rate / R:R calculation

---

## Recommendation for Strategy 2

**MUST IMPLEMENT Option 1** before proceeding to Strategy 2 (ORB):

```python
def calculate_position_size(init_cash, close, atr, atr_multiplier=2.5, risk_pct=0.02):
    """
    Calculate position size with capital constraint.

    Args:
        init_cash: Account size
        close: Current price
        atr: Average True Range
        atr_multiplier: Stop distance multiplier
        risk_pct: Risk per trade (default 0.02 = 2%)

    Returns:
        tuple: (position_size_shares, actual_risk_pct, constrained)
    """
    stop_distance = atr * atr_multiplier

    # Risk-based position size
    position_size_risk = (init_cash * risk_pct) / stop_distance

    # Capital-based maximum
    position_size_capital = init_cash / close

    # Take minimum (capital constraint)
    position_size = min(position_size_risk, position_size_capital)

    # Calculate actual risk achieved
    actual_risk = (position_size * stop_distance) / init_cash

    # Flag if constrained
    constrained = position_size == position_size_capital

    return position_size, actual_risk, constrained
```

**Usage in Strategy 2:**

```python
# In generate_signals() or backtest()
position_sizes = []
actual_risks = []

for i in range(len(data)):
    size, risk, constrained = calculate_position_size(
        init_cash=init_cash,
        close=data['Close'].iloc[i],
        atr=atr.iloc[i],
        atr_multiplier=2.5,  # ORB uses wider stops
        risk_pct=0.02
    )
    position_sizes.append(size)
    actual_risks.append(risk)

    if constrained:
        # Log warning: position was capped by capital
        pass

# Use in VectorBT
pf = vbt.PF.from_signals(
    close=close,
    entries=long_entries,
    size=position_sizes,
    size_type="amount",
    ...
)
```

---

## Implications

### For Strategy 1:
- Results are still valid for comparison purposes
- But absolute performance metrics are slightly inflated (positions were larger than intended)
- The strategy would NOT be executable as-is in live trading
- Re-running with fixed position sizing would likely show:
  - Lower total return (smaller positions)
  - Lower max drawdown (smaller positions)
  - Similar Sharpe ratio (proportional scaling)

### For Strategy 2 (ORB):
- **MUST implement capital-constrained position sizing**
- Cannot proceed to Phase 1 implementation without this fix
- ORB uses 2.5× ATR stops (vs 2.0× for Strategy 1), so problem is slightly less severe
- But intraday strategies may have even lower ATR, making this more critical

### For Strategy 3 (TFC Hybrid):
- Same fix applies
- Will inherit corrected position sizing from Strategy 2
- May need portfolio heat management (multiple strategies running simultaneously)

---

## Action Items

### Immediate (Before Strategy 2):
- [ ] Implement capital-constrained position sizing function
- [ ] Add to shared utilities module (`utils/position_sizing.py`)
- [ ] Unit test with edge cases:
  - [ ] Very low ATR (high volatility)
  - [ ] Very high ATR (low volatility)
  - [ ] Price near stop distance
  - [ ] Zero ATR (NaN handling)
- [ ] Document in code with clear comments

### Phase 1 (Strategy 2 Implementation):
- [ ] Use corrected position sizing from start
- [ ] Verify in Phase 0 equivalent for Strategy 2
- [ ] Monitor % of constrained trades in backtests
- [ ] If > 30% constrained, consider increasing risk_pct or using tighter stops

### Optional (Future Enhancement):
- [ ] Implement portfolio heat management for multi-strategy
- [ ] Add Kelly Criterion position sizing option
- [ ] Create position sizing comparison study

---

## Conclusion

**Position sizing verification: FAIL**

**Root cause:** Formula is mathematically correct but produces oversized positions when ATR is low. Missing capital constraint allows impossible position sizes (> 100% of capital).

**Solution:** Implement capital-constrained position sizing using `min(risk_based, capital_based)`.

**Status:** **BLOCKER** - Cannot proceed to Strategy 2 implementation until this is fixed.

**Next Steps:**
1. Create `utils/position_sizing.py` with corrected function
2. Add unit tests
3. Proceed to Phase 1 (Strategy 2 Implementation)

---

**Verified by:** Claude Code
**Review Status:** Awaiting human team review
**Approval Required:** Yes (critical bug affects all strategies)
