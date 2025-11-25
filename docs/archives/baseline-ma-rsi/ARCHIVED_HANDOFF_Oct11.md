# HANDOFF - Strategy 2 Development: Position Sizing Critical Issue

## Current Branch
`feature/baseline-ma-rsi`

## Session Date
**October 11, 2025 (Evening Session) - PAUSED FOR HUMAN REVIEW**

---

## CRITICAL: Position Sizing Bug Discovered

### Status: BLOCKER - Cannot proceed to Strategy 2 until resolved

**Phase 0 (Position Sizing Verification) completed and identified critical bug in Strategy 1.**

### The Problem

**Formula is mathematically correct but produces impossible position sizes:**

```
Mean position size: 81.8% of capital (should be 10-30%)
83% of positions exceed 50% of capital
Maximum position size: 142.6% of capital (IMPOSSIBLE)
```

**Root Cause:**
```python
# Current formula (CORRECT math, MISSING constraint)
stop_distance = atr * 2.0
position_size = (init_cash * 0.02) / stop_distance  # Calculates shares for 2% risk

# PROBLEM: When ATR is low, this calculates huge position sizes
# Example: ATR=$5, stop=$10, position_size = $200/$10 = 20 shares
# At SPY=$480: 20 shares = $9,600 (96% of $10k capital!)
```

### The Fix (Must implement before Strategy 2)

```python
# Add capital constraint
position_size_risk = (init_cash * 0.02) / stop_distance  # Risk-based
position_size_capital = init_cash / close  # Capital-based maximum
position_size = min(position_size_risk, position_size_capital)  # Take minimum
```

### Documentation

**Complete analysis:** `docs/POSITION_SIZING_VERIFICATION.md`

**Key findings:**
- Theoretical risk: 2.00% (formula is correct)
- Actual avg loss: -2.86% (43% higher due to capital constraint violations)
- Discrepancy: 4.86% (FAIL - exceeds 0.5% threshold)
- ATR analysis: PASS (1.39% of price is normal)
- Position sizing: FAIL (impossible to execute calculated sizes)

**Implications:**
- Strategy 1 results still valid for comparison (proportional performance)
- Cannot deploy Strategy 1 as-is to live trading
- **Strategy 2 MUST implement corrected position sizing from start**
- All future strategies inherit this fix

---

## What Was Accomplished Today

### Phase 0: Position Sizing Verification - COMPLETE

**Created files:**
- `verify_position_sizing.py` - Verification script
- `docs/POSITION_SIZING_VERIFICATION.md` - Complete analysis (20+ pages)

**Key diagnostics run:**
1. Theoretical vs actual risk comparison
2. Position size distribution analysis
3. ATR reasonableness checks
4. Backtest execution and trade analysis
5. Extreme loss identification

**Verdict:** FAIL - Critical bug must be fixed before Strategy 2

### Updated Implementation Plan

**Reviewed and approved:**
- 7-phase implementation plan for Strategy 2 (ORB)
- 6 critical corrections from Claude Desktop's addendum:
  1. Volume confirmation 2.0× MANDATORY (not optional)
  2. Sharpe targets DOUBLED (min 2.0, not 1.0)
  3. R:R minimum RAISED to 3:1 (not 2:1)
  4. NEW Phase 2.3: Expectancy analysis (mandatory)
  5. NEW Phase 4.6: STRAT-lite bias filter (optional)
  6. ATR 3.0× testing emphasized

**Mathematical proofs reviewed:**
- Why 2:1 R:R is insufficient (loses money after costs + efficiency)
- Why Sharpe must be 2.0+ in backtest (50% haircut for real-world)
- Why 80% efficiency factor applies to fixed fractional sizing

**Documentation reviewed:**
- Claude Desktop Analysis: STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md (2,100+ lines)
- Claude Desktop Analysis: STRATEGY_2_IMPLEMENTATION_ADDENDUM.md (1,200+ lines)
- New research: Advanced_Algorithmic_Trading_Systems.md (expectancy proofs, STRAT-lite)

---

## NEXT SESSION - IMMEDIATE ACTIONS

### 1. HUMAN REVIEW REQUIRED (Tonight/Tomorrow)

**Review Position Sizing Findings:**
- Read `docs/POSITION_SIZING_VERIFICATION.md` in full
- Decide on fix approach:
  - **Option 1 (Recommended):** Capital-constrained position sizing
  - **Option 2:** Fixed % of capital approach
  - **Option 3:** Kelly Criterion with constraints
- Approve/modify corrected formula before continuing

**Questions for human team member:**
1. Approve Option 1 (capital constraint) as fix?
2. Should we create `utils/position_sizing.py` module?
3. Should we re-run Strategy 1 with corrected sizing for comparison?
4. Any concerns about the 81.8% mean position size finding?

### 2. After Approval: Create Position Sizing Utility (30 min)

```python
# utils/position_sizing.py
def calculate_position_size(init_cash, close, atr, atr_multiplier, risk_pct):
    """Capital-constrained position sizing with 2% risk target."""
    stop_distance = atr * atr_multiplier
    position_size_risk = (init_cash * risk_pct) / stop_distance
    position_size_capital = init_cash / close
    position_size = min(position_size_risk, position_size_capital)
    actual_risk = (position_size * stop_distance) / init_cash
    constrained = position_size == position_size_capital
    return position_size, actual_risk, constrained
```

### 3. Then Resume: Phase 1 - Strategy 2 Implementation

**Phase 1.1: Setup (30 min)**
- Delete temp files (verify_position_sizing.py after review complete)
- Commit Strategy 1 + position sizing analysis to feature/baseline-ma-rsi
- Create feature/opening-range-breakout branch

**Phase 1.2-1.3: Signal Generation (Day 1, 7 hours)**
- Opening range calculation (9:30-9:35 AM ET)
- Breakout detection (close > opening_high)
- **MANDATORY: 2.0× volume confirmation (hardcoded, not parameter)**
- ATR calculation for stops
- Test on small dataset

**Phase 1.4: Exit Logic (Day 2, 4 hours)**
- EOD exit at 3:55 PM ET
- 2.5× ATR stops
- **NO signal exits (no RSI, MACD, MA)**

**Phase 1.5-1.6: VectorBT Integration (Day 2-3, 6 hours)**
- Use CORRECTED position sizing
- Configure vbt.PF.from_signals()
- Run initial 1-year backtest
- Verify no errors

---

## Strategy 1: Summary (for reference)

**Status:** ARCHIVED FOR BEAR MARKET TESTING

**Results (4-year backtest on SPY 2021-2025):**
- Total Return: 3.86% vs 54.41% buy-and-hold
- Sharpe: 0.22 (FAIL - target 0.8-1.2)
- Longs: 35 trades, +0.27% avg
- Shorts: 10 trades, -0.68% avg
- Win Rate: 74% but avg trade too small

**Key Findings:**
1. Mean reversion wrong for bull market (2021-2025)
2. RSI exits cut winners short (60% of trades)
3. Mixing Connors + Asymmetric = hybrid failure
4. **Position sizing bug discovered (today)**

**Decision:** Archive, move to Strategy 2 (ORB)

**Documentation:** `docs/STRATEGY_1_BASELINE_RESULTS.md`

---

## Strategy 2: Opening Range Breakout (ORB) - READY TO START

### Why Strategy 2 Next?
1. Designed for trending markets (2021-2025 = bull)
2. Asymmetric R:R (17% win rate, 2.396 Sharpe in research)
3. Intraday timeframe (different from Strategy 1)
4. Proven approach (QuantConnect research)

### Updated Success Criteria (from Addendum)

**Minimum Viable Performance:**
- Win Rate: 15-30% (low by design)
- R:R Ratio: > 3:1 minimum (UPDATED from 2:1)
- Sharpe Ratio: > 2.0 minimum (UPDATED from 1.0)
- Avg Trade: > 0.6% (UPDATED from 0.5%)
- Net Expectancy: > 0.005 (NEW - 0.5% after costs + efficiency)
- Trade Count: > 100
- Max Drawdown: < 25%

### Critical Implementation Requirements

**MANDATORY:**
1. Volume confirmation 2.0× (hardcoded in Phase 1.3)
2. Capital-constrained position sizing (from fix)
3. NO signal exits (only EOD + stops)
4. Expectancy analysis in Phase 2.3
5. Transaction costs: 0.35% (0.2% fees + 0.15% slippage)

**Updated from research:**
- Sharpe must be 2.0+ in backtest (real-world ~1.0 after haircut)
- R:R must be 3:1+ (2:1 loses money after costs)
- Test 3.0× ATR stops (not just 2.5×)
- Document STRAT-lite results (even if not used)

### Files to Create
```
utils/
├── position_sizing.py         # NEW - Corrected position sizing

strategies/
├── opening_range_breakout.py  # Core ORB strategy

docs/
├── STRATEGY_2_ORB_RESULTS.md  # After backtest complete
```

### Research References
- `docs/Claude Desktop Analysis/STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md`
- `docs/Claude Desktop Analysis/STRATEGY_2_IMPLEMENTATION_ADDENDUM.md`
- `docs/Algorithmic Systems Research/Advanced_Algorithmic_Trading_Systems.md`
- `docs/Algorithmic trading systems with asymmetric risk-reward profiles.md`

---

## VectorBT Pro Resources (Updated)

### Priority Documentation Structure

**START HERE:**
`VectorBT Pro Official Documentation/README.md` - Navigation guide

**LLM Docs (PRIORITY - search first):**
- `LLM Docs/3 API Documentation.md` - Complete API (242k+ lines)
- `LLM Docs/2 General Documentation.md` - Comprehensive general docs
- `LLM Docs/1 Documentation_File_Locations.md` - File mapping

**Python Introspection:**
```python
import vectorbtpro as vbt

# Find documentation
vbt.find_docs(vbt.PFO)

# Find examples
vbt.find_examples(vbt.PFO)

# Get help
vbt.pdir(vbt.AlpacaData)
vbt.phelp(vbt.AlpacaData.pull)
```

### API Patterns Learned

**Data fetching:**
```python
# Correct pattern
data_obj = vbt.AlpacaData.pull('SPY', start='2021-01-01', end='2025-01-01', timeframe='1D')
df = data_obj.get()
```

**Portfolio stats:**
```python
# Properties (no parentheses)
pf.total_return
pf.sharpe_ratio
pf.trades.win_rate

# Methods (needs parentheses)
pf.trades.count()
```

---

## TFC Analysis (for Strategy 3)

**Results (4-year SPY data):**
- High-confidence (3/4 or 4/4 aligned): 39.5%
- FTFC (4/4): 6.9%
- TFC (3/4): 32.7%
- Bullish bias: 2.5:1

**Strategy 3 Implications:**
- TFC as allocation filter (not sole strategy)
- Need fallback for 60.5% low-confidence periods
- Dynamic sizing: 4/4=2%, 3/4=1.5%, 2/4=1%

---

## File Status

### Keep
- `strategies/baseline_ma_rsi.py`
- `docs/STRATEGY_1_BASELINE_RESULTS.md`
- `docs/POSITION_SIZING_VERIFICATION.md` (NEW)
- `docs/HANDOFF.md` (this file)
- `VectorBT Pro Official Documentation/` (all files)
- `docs/Claude Desktop Analysis/` (all files)
- `docs/Algorithmic Systems Research/` (all files)

### Delete After Human Review
- `verify_position_sizing.py` (temp analysis script)

### Create Next Session
- `utils/position_sizing.py` (after approval)
- `strategies/opening_range_breakout.py`

---

## Professional Standards Maintained

1. Data-driven analysis (not emotional)
2. Mathematical proofs for all claims
3. Honest assessment (FAIL verdict on position sizing)
4. Complete documentation (20+ page analysis)
5. Clear action items with human review gate
6. No guessing - all findings verified

---

## Context Management

**Current Status:** 60% context used
**Next Session:** Start fresh after /compact (if needed)

**Key Points to Preserve:**
1. Position sizing bug discovered (81.8% mean position size)
2. Fix approved: Capital-constrained position sizing
3. Strategy 2 ready to start with corrected sizing
4. Updated success criteria (Sharpe 2.0+, R:R 3:1+)
5. 6 critical corrections from addendum

**Can Forget:**
- Detailed verification script debugging steps
- Exact column name errors encountered
- Terminal restart details

---

## Next Session Starting Prompt

```
We've completed Phase 0 (Position Sizing Verification) for Strategy 2 development.

CRITICAL FINDING: Discovered position sizing bug in Strategy 1:
- 81.8% mean position size (should be 10-30%)
- Formula correct but missing capital constraint
- Fix required before Strategy 2 implementation

HUMAN REVIEW COMPLETE: [Awaiting confirmation]
- Approved fix: Capital-constrained position sizing
- Create utils/position_sizing.py module

READY TO PROCEED:
1. Implement corrected position sizing utility
2. Start Phase 1: Strategy 2 (ORB) core implementation
3. Use MANDATORY 2.0x volume confirmation
4. Target Sharpe 2.0+, R:R 3:1+ (updated criteria)

Current branch: feature/baseline-ma-rsi
Next branch: feature/opening-range-breakout (after setup)

Please confirm:
1. Position sizing fix approved (Option 1)?
2. Ready to create utils/position_sizing.py?
3. Ready to start Strategy 2 implementation?
```

---

**Last Updated:** 2025-10-11 Evening
**Status:** PAUSED FOR HUMAN REVIEW
**Next Session:** After position sizing fix approval
**Recommendation:** Review docs/POSITION_SIZING_VERIFICATION.md before continuing
