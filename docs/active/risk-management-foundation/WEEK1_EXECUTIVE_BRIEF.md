# Week 1 Implementation - Executive Brief for Claude Code

**Date:** October 14, 2025  
**Status:** READY TO PROCEED  
**Full Documentation:** See CRITICAL_POSITION_SIZING_CLARIFICATION.md

---

## 🔴 CRITICAL: VectorBT Pro Compliance

**ALL CODE MUST BE VECTORBT PRO COMPATIBLE**

**Requirements:**
- ✅ Vectorized operations (no loops)
- ✅ Pandas Series with matching index to price data
- ✅ Works with `vbt.Portfolio.from_signals()`
- ✅ Proper index alignment for all calculations

**Documentation Path:**
```
C:\Strat_Trading_Bot\vectorbt-workspace\VectorBT Pro Official Documentation
```

**If unsure:** STOP and consult documentation BEFORE coding.

---

## TL;DR - The Answer

**Question:** Which position sizing for Week 1 / Strategy 2 (ORB)?

**Answer:** ATR-based with capital constraint (Method D)

**Why:** ORB uses 2.5× ATR stops, so position sizing MUST be ATR-based.

**NOT:** Garman-Klass (that's for momentum portfolios in Week 4-6)

**NOT:** Yang-Zhang (that's for regime detection in Week 7-9)

---

## Week 1 Implementation Tasks

### Task 1: Create `utils/position_sizing.py`

```python
def calculate_position_size_atr(init_cash, close, atr, atr_multiplier=2.5, risk_pct=0.02):
    """
    ATR-based position sizing with capital constraint fix.
    
    VECTORBT PRO COMPATIBLE - Fully vectorized, maintains index alignment.
    """
    stop_distance = atr * atr_multiplier
    position_size_risk = (init_cash * risk_pct) / stop_distance
    position_size_capital = init_cash / close
    position_size = np.minimum(position_size_risk, position_size_capital)  # Vectorized min
    actual_risk = (position_size * stop_distance) / init_cash
    constrained = position_size == position_size_capital
    return position_size, actual_risk, constrained
```

**VectorBT Pro Usage:**
```python
# Calculate position sizes (returns pandas Series)
position_sizes, actual_risks, constrained = calculate_position_size_atr(
    init_cash=10000,
    close=data['Close'],      # pandas Series
    atr=atr_series,           # pandas Series with matching index
    atr_multiplier=2.5,
    risk_pct=0.02
)

# Use with VectorBT Pro
pf = vbt.Portfolio.from_signals(
    close=data['Close'],
    entries=long_entries,
    exits=long_exits,
    size=position_sizes,      # Our calculated sizes (pandas Series)
    size_type='amount',       # Shares, not percentage
    init_cash=10000,
    sl_stop=atr_series * 2.5, # ATR-based stops
    fees=0.001
)
```

**This fixes the bug from Phase 0:** Adds `np.minimum(risk_based, capital_based)` constraint.

---

### Task 2: Unit Tests (`tests/test_position_sizing.py`)

Test these scenarios:
- ✅ Capital constraint activates (low ATR → large position)
- ✅ No position exceeds 100% of capital
- ✅ Edge cases (zero ATR, extreme ATR, negative close)
- ✅ Vectorized operation (pd.Series input)
- ✅ **VectorBT Pro compatibility:**
  - Input: pandas Series with DatetimeIndex
  - Output: pandas Series with same index
  - Index alignment preserved
  - No loops used (vectorized operations only)

---

### Task 3: Gate 1 Verification

Run Strategy 2 (ORB) backtest with corrected position sizing and verify:

**VectorBT Pro Compliance:**
- ✅ Position sizes are pandas Series (not numpy array or list)
- ✅ Index matches close price data exactly
- ✅ VectorBT Pro accepts the format without errors
- ✅ No manual loops (all operations vectorized)

**Position Size Constraints:**
- Mean position size: 10-30% range (STRICT)
- Max position size: ≤100% (zero violations)
- No NaN or Inf values

**Technical:**
- VBT runs without errors
- No runtime exceptions

**Performance:**
- Compare to baseline (current Strategy 2)
- Document improvement

---

## Why This Is Correct

From your HANDOFF.md:

```markdown
Strategy 2: Opening Range Breakout (ORB)

Phase 1.4: Exit Logic
- 2.5× ATR stops  ← THIS IS WHY WE USE ATR-BASED SIZING

Phase 1.5-1.6: VectorBT Integration
- Use CORRECTED position sizing  ← ATR with capital constraint
```

**Match position sizing to stop methodology.** ORB uses ATR stops, so position sizing must be ATR-based.

---

## What NOT To Do

❌ Don't implement Garman-Klass semi-volatility (wrong strategy type)  
❌ Don't implement Yang-Zhang volatility (wrong use case)  
❌ Don't mix multiple position sizing methods  
❌ Don't compare semi-vol to ATR for ORB (apples to oranges)

---

## Timeline

**Week 1-2:** ATR-based (THIS WEEK)  
**Week 4-6:** Garman-Klass (momentum portfolios - FUTURE)  
**Week 7-9:** Yang-Zhang (regime detection - FUTURE)

---

## Commit Structure

```
Commit 1: "feat: add ATR-based position sizing with capital constraint"
  - utils/position_sizing.py
  - calculate_position_size_atr() with min() fix

Commit 2: "test: add position sizing unit tests"
  - tests/test_position_sizing.py
  - Test capital constraint, edge cases, vectorization

Commit 3: "test: Gate 1 verification for Strategy 2 (ORB)"
  - Run backtest with corrected sizing
  - Document position size distribution
  - Verify PASS criteria

Commit 4: "docs: Gate 1 results and validation"
  - Mean position size: X%
  - Max position size: Y%
  - Pass/Fail decision
```

---

## Questions Before You Start?

**Q: Should I implement Garman-Klass too?**  
A: No. That's Week 4-6 for momentum portfolios. Week 1 is ATR only.

**Q: Should I compare ATR vs semi-vol performance?**  
A: No. They're for different strategy types. Not a valid comparison.

**Q: What about Yang-Zhang volatility?**  
A: That's Week 7-9 for regime detection (GMM features). Not position sizing.

**Q: How do I know I'm done?**  
A: Gate 1 PASS criteria met (mean position size 10-30%, max ≤100%).

---

## Ready to Proceed?

If you have read this brief and the full clarification document, you are ready to:

1. Create `utils/position_sizing.py` (ATR-based only)
2. Write unit tests
3. Run Gate 1 verification
4. Document results

**Estimated time:** 3-4 hours (SHORT SESSION) for foundation  
**OR:** 5-6 hours (LONG SESSION) for complete Gate 1 verification

---

**Full details:** CRITICAL_POSITION_SIZING_CLARIFICATION.md (20+ pages)

**Status:** Ready for implementation ✅
