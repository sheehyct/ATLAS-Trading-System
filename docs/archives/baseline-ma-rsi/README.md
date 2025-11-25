# Archive: feature/baseline-ma-rsi Branch

**Archived:** October 12, 2025
**Branch Status:** Frozen for reference (not deleted)
**Reason:** Strategic pivot based on diagnostic framework analysis

---

## What This Branch Tested

**Primary Goals:**
- Strategy 1: Baseline MA/RSI mean reversion (Connors RSI approach)
- Two-branch comparison: TFC confidence scoring vs simple MA filter
- Position sizing with 2% risk per trade approach

**Key Components Developed:**
- Opening Range Breakout (ORB) - documented but not implemented
- TFC-based confidence scoring system
- Multi-timeframe data management
- STRAT pattern detection and classification

---

## Key Findings (Why Archived)

### 1. Position Sizing Bug Discovered (81.8% mean position size)
- **Problem:** Formula mathematically correct but missing capital constraint
- **Impact:** Impossible position sizes (max 142.6% of capital)
- **Root Cause:** `position_size = (init_cash * 0.02) / stop_distance` with no upper bound
- **Learning:** Capital constraints are NOT optional, they're mandatory

**Documentation:** `POSITION_SIZING_VERIFICATION.md` (20+ page analysis)

### 2. Missing Risk Layers Identified
From diagnostic framework analysis, discovered THREE missing risk layers:
1. **Position sizing constraint** (found, but incomplete)
2. **Portfolio heat management** (completely missing - no 6-8% total exposure limit)
3. **Regime awareness** (never considered - trading same rules in bull/bear markets)

**Documentation:** `../../research/Medium_Articles_Research_Findings.md`

### 3. TFC Confidence Scoring = Overfitting Risk
- **Parameters:** 6-7 (weights, thresholds, confidence levels)
- **Required data:** 60-140 independent observations
- **Available data:** ~50-70 effective observations (correlation-adjusted)
- **Verdict:** ABANDON (not "fix") - guaranteed overfitting

**Cross-validation consensus:** All three analyses (Opus 4.1, Desktop Sonnet 4.5, Web Sonnet 4.5) recommended abandoning TFC scoring

### 4. Retail vs Professional Psychology
The 81.8% position sizing bug was a **symptom** of retail thinking:
- Optimizing for high win rates (70%+) instead of expectancy
- Maximizing capital utilization instead of managing portfolio heat
- Building complex indicators to chase precision instead of accepting asymmetric R:R
- Holding losers 14 days hoping for recovery instead of cutting at 2-3 days

**Key insight:** Technical fixes won't work without mindset correction

---

## What Worked (Preserved Knowledge)

### Positive Results:
1. **VectorBT Pro integration** - Working data pipeline, backtesting framework
2. **Multi-timeframe manager** - Production-grade MTF data handling
3. **STRAT classification** - Accurate bar classification and pattern detection
4. **Strategy 1 results** - Provided baseline comparison (3.86% vs 54.41% SPY)

### Reusable Components:
- `core/analyzer.py` - STRAT bar classification (working correctly)
- `data/mtf_manager.py` - Market-aligned multi-timeframe data
- `data/alpaca.py` - Alpaca data fetching
- VectorBT Pro documentation navigation patterns

---

## Why Preserved (Not Deleted)

**Negative results are VALUABLE:**
- Documents what DOESN'T work (prevents repeating mistakes)
- Shows evolution of thinking (retail → professional approach)
- Preserves context for "why we tried X and abandoned it"
- Position sizing bug analysis is reference material for future risk management

**Academic standard:** Failed experiments are published, not hidden

---

## Related Active Work

**New branch:** `feature/risk-management-foundation`

**New approach:**
1. Implement ALL three risk layers FIRST (position sizing, portfolio heat, regime detection)
2. THEN build strategies on solid foundation
3. Abandon TFC confidence scoring (6+ parameters)
4. Replace Strategy 1 RSI logic with 5-day washout (proven approach)

**Documentation:** `../../active/risk-management-foundation/`

---

## Files in This Archive

### Strategy Documentation
- `STRATEGY_OVERVIEW.md` - Two-branch approach (TFC vs Baseline)
- `IMPLEMENTATION_PLAN.md` - Step-by-step build guide (1100 lines)
- `BRANCH_COMPARISON.md` - TFC vs Baseline analysis
- `STRATEGY_1_BASELINE_RESULTS.md` - Actual backtest results

### Analysis & Decisions
- `POSITION_SIZING_VERIFICATION.md` - 20-page mathematical analysis of bug
- `ARCHIVED_HANDOFF_Oct11.md` - Session state when pivot occurred

---

## Lessons Learned (Applied to New Branch)

1. **Risk management is infrastructure, not strategy**
   - Build position sizing + portfolio heat + regime detection FIRST
   - Strategies inherit these layers (not build them ad-hoc)

2. **Parameter count predicts overfitting**
   - 2-3 parameters = robust
   - 6+ parameters = guaranteed overfitting
   - No exceptions

3. **Professional mindset required**
   - Expectancy > win rate
   - Portfolio heat limits are hard gates, not guidelines
   - Accept 40-50% cash allocation as normal
   - Cut losers fast (2-3 days for mean reversion, not 14)

4. **VBT documentation is mandatory**
   - NEVER code without checking VBT docs first
   - Python introspection (`vbt.phelp()`) prevents errors
   - LLM Docs folder is PRIORITY resource

---

## Access to Original Branch Code

**Git branch:** `feature/baseline-ma-rsi` (if still exists)

**To view original code:**
```bash
git checkout feature/baseline-ma-rsi
# Or view specific files without checkout:
git show feature/baseline-ma-rsi:strategies/baseline_ma_rsi.py
```

---

## Questions About This Archive?

**Why not delete?**
→ Negative results are valuable. "We tried X, here's why it failed" prevents future mistakes.

**Can we resurrect TFC scoring?**
→ Not recommended. 6+ parameters = overfitting regardless of implementation quality.

**Was Strategy 1 completely wrong?**
→ No - the APPROACH was correct (mean reversion in bull markets fails), but implementation had bugs.

**What if GMM regime detection also fails?**
→ Then we document it in archives/risk-management-foundation/ with same rigor.

---

**Last Updated:** October 12, 2025
**Archived By:** Claude Code (session post /clear due to context limits)
**Cross-Reference:** `../../research/diagnostic_framework.md` for failure taxonomy
