# Consolidated Research Findings - Algorithmic Trading System

**Last Updated:** October 18, 2025
**Status:** Authoritative consolidated reference
**Purpose:** Single source of truth for research-backed trading principles

---

## How to Use This Document

This document consolidates 7+ research files that contained contradictions, redundancy, and outdated guidance.

**What was consolidated:**
- Advanced_Algorithmic_Trading_Systems.md
- Algorithmic Trading System.md
- Algorithmic trading systems with asymmetric risk-reward profiles.md
- CRITICAL_POSITION_SIZING_CLARIFICATION.md
- STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md
- WEEK1_EXECUTIVE_BRIEF.md (moved to active development)

**What remains separate (not consolidated):**
- Medium_Articles_Research_Findings.md - Foundational cross-validated research (READ THIS FIRST)
- VALIDATION_PROTOCOL.md - Complete 5-phase testing methodology
- STRATEGY_2_IMPLEMENTATION_ADDENDUM.md - ORB-specific implementation corrections

**Original files:** Archived in docs/research/ARCHIVE/ for historical reference

---

## Part 1: Foundational Principles

### Expectancy Framework (Mathematical Truth)

**Core Formula:**
```
Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss) - Transaction Costs

MUST be >0.005 (0.5%) after costs and efficiency factor
```

**Why 2:1 R:R is INSUFFICIENT:**
```
50% win rate @ 2:1 R:R
Expectancy = (0.5 × 2) - (0.5 × 1) = 0.5 = 50% per trade (gross)

Transaction costs: 0.35% (0.2% fees + 0.15% slippage)
Net expectancy: 0.5% - 0.35% = 0.15% (BARELY POSITIVE)

Conclusion: Need 3:1 R:R MINIMUM for robust edge
```

**Why Sharpe Must Be 2.0+ in Backtest:**
```
Backtest Sharpe: 2.0
Real-world haircut: 50%
Live Sharpe: ~1.0 (acceptable)

If backtest Sharpe = 1.0:
Live Sharpe = ~0.5 (UNACCEPTABLE)
```

**Research Source:** Advanced Algorithmic Trading Systems analysis + empirical validation

---

### Asymmetric Risk-Reward Profiles

**Definition:**
Strategies where average win > average loss by ratio of 3:1 or greater.

**Why Traditional Metrics Fail:**
- Standard deviation treats upside and downside equally
- A strategy with 30% win rate @ 5:1 R:R beats 70% win rate @ 1:1 R:R
- Expectancy > Win Rate for asymmetric strategies

**Preferred Metrics:**
```
1. Sortino Ratio (uses downside deviation only)
   = Return / Downside Deviation
   Target: >1.5

2. Expectancy
   = (Win% × Avg Win) - (Loss% × Avg Loss)
   Target: >0.5%

3. Payoff Ratio
   = Avg Win / Avg Loss
   Target: >3.0
```

**Example:**
```
Strategy A (Mean Reversion):
Win Rate: 70%
Avg Win: 1%
Avg Loss: 3%
Expectancy: (0.7 × 1%) - (0.3 × 3%) = -0.2% (NEGATIVE)

Strategy B (Asymmetric Breakout):
Win Rate: 30%
Avg Win: 5%
Avg Loss: 1%
Expectancy: (0.3 × 5%) - (0.7 × 1%) = 0.8% (POSITIVE)

Conclusion: Strategy B is superior despite lower win rate
```

**Research Source:** Algorithmic trading systems with asymmetric risk-reward profiles

---

### Transaction Cost Reality Check

**Published Research Warning:**
> "Correcting transaction cost assumptions from 0.1% to the more realistic 0.2% per trade transformed profitable backtests into unprofitable reality"

**Conservative Cost Assumptions (MANDATORY):**
```
Fees: 0.2% per trade (0.002 in VBT)
Slippage: 0.15% per trade (based on Article research, Oct 18)
Total Round Trip: 0.35%
```

**Implication:**
- Strategy must generate >0.5% per trade AFTER costs
- Backtests with 0.1% fees are MISLEADING
- Always test with 0.2% fees minimum

**Research Source:** Multiple article validation + STRATEGY_2_IMPLEMENTATION_ADDENDUM

---

## Part 2: Position Sizing - Resolved Standards

### CRITICAL CORRECTION: Volume Confirmation Threshold

**CONTRADICTION RESOLVED:**

**OLD (CLAUDE.md, outdated):**
- Volume > 1.5x average

**CURRENT (AUTHORITATIVE):**
- Volume > 2.0x average (MANDATORY, hardcoded per research)
- Source: STRATEGY_2_IMPLEMENTATION_ADDENDUM.md + Article research Oct 18

**Implementation:**
```python
volume_ma = data['volume'].rolling(20).mean()
volume_confirmation = data['volume'] > (volume_ma * 2.0)  # 2.0x MANDATORY

entry_signal = (
    (price > breakout_level) &
    (volume > volume_ma * 2.0) &  # NOT 1.5x - MUST be 2.0x
    (other_conditions)
)
```

**Rationale:**
- Article research: "Volume indicators measure trend strength and reduce false signals"
- 2.0x threshold backed by empirical testing
- 1.5x was preliminary estimate, 2.0x is research-validated

---

### Capital-Constrained ATR Position Sizing

**THE BUG (Phase 0):**
```python
# WRONG - Missing capital constraint
position_size = (init_cash * 0.02) / stop_distance
# Result: 81.8% mean, 142.6% max (VIOLATED 100% LIMIT)
```

**THE FIX (Implemented Oct 13-15):**
```python
# CORRECT - Dual constraint
stop_distance = atr * atr_multiplier
position_size_risk = (init_cash * risk_pct) / stop_distance
position_size_capital = init_cash / close
position_size = np.minimum(position_size_risk, position_size_capital)  # Hard constraint
```

**Mathematical Proof Position Never Exceeds 100%:**
```
position_size_capital = init_cash / close
Max position value = position_size_capital × close
                   = (init_cash / close) × close
                   = init_cash
                   = 100% of capital [QED]
```

**Status:** VERIFIED in Gate 1 (Oct 14, 2025)

---

### Position Sizing Parameters - CORRECTED

**CONTRADICTION RESOLVED:**

**OLD (Documentation):**
- risk_pct = 0.02 (2%)

**CURRENT (Empirically Validated):**
- risk_pct = 0.01 (1%) RECOMMENDED for 10-30% target
- risk_pct = 0.02 (2%) produces 40.6% mean (exceeds target)

**Evidence:**
```
Gate 1 Results (Oct 14):
risk_pct = 0.02 → Mean 40.6%, Max 44.5%
Gate 1 Recommendation:
risk_pct = 0.01 → Expected mean ~20% (within 10-30%)
```

**Parameter Optimization Matrix:**

| Configuration | Expected Mean | Target Range | Status |
|---------------|---------------|--------------|--------|
| risk_pct=0.02, atr_mult=2.5 | 40.6% | 10-30% | EXCEEDS |
| risk_pct=0.01, atr_mult=2.5 | ~20% | 10-30% | OPTIMAL |
| risk_pct=0.015, atr_mult=3.0 | ~22% | 10-30% | OPTIMAL |
| risk_pct=0.02, atr_mult=3.5 | ~25% | 10-30% | ACCEPTABLE |

**Recommendation:** Use risk_pct=0.01 as DEFAULT for ORB strategy

**Source:** GATE1_RESULTS.md + CRITICAL_POSITION_SIZING_CLARIFICATION.md consolidation

---

### Position Sizing Methodology Selection

**CLARIFICATION: Which method for which strategy?**

**ATR-Based (Current Implementation):**
- Use for: Strategies with ATR-based stops (ORB, breakouts)
- Formula: position_size = (capital × risk%) / (ATR × multiplier)
- Status: IMPLEMENTED (utils/position_sizing.py)

**Garman-Klass Semi-Volatility (Future):**
- Use for: Momentum portfolios (downside risk measurement)
- Formula: Uses OHLC range-based volatility
- Status: NOT STARTED (Week 4-6 future work)
- NOT for ORB strategy (wrong use case)

**Yang-Zhang Volatility (Future):**
- Use for: Regime detection features (GMM clustering)
- Formula: Combines overnight + intraday volatility
- Status: NOT STARTED (Week 2-3 GMM implementation)
- NOT for position sizing (wrong use case)

**DO NOT:**
- Mix position sizing methods across same strategy
- Use Garman-Klass for ATR-stop strategies
- Use Yang-Zhang for position sizing (it's for regime detection)

**Source:** CRITICAL_POSITION_SIZING_CLARIFICATION.md + WEEK1_EXECUTIVE_BRIEF.md

---

## Part 3: Performance Targets - Resolved Standards

### Sharpe Ratio Targets (CONTRADICTIONS RESOLVED)

**ORIGINAL CONTRADICTIONS:**
- VALIDATION_PROTOCOL.md: Sharpe >0.8 acceptable
- STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md: ORB target 2.396
- STRATEGY_2_IMPLEMENTATION_ADDENDUM.md: Minimum 2.0

**AUTHORITATIVE RESOLUTION:**

**Backtest Targets (Conservative Assumptions):**
```
Minimum Acceptable: Sharpe >1.0
Good: Sharpe >1.5
Excellent: Sharpe >2.0
ORB Specific: Target 2.0+ (was 2.396, lowered to realistic)
```

**Paper Trading Targets:**
```
Minimum Viable: Sharpe >0.8
Exceptional: Sharpe >1.2
```

**Live Trading Targets:**
```
Acceptable: Sharpe >0.5
Good: Sharpe >0.8
Excellent: Sharpe >1.0
```

**Haircut Assumption:**
```
Backtest Sharpe 2.0 → Paper 1.5 → Live 1.0 (50% total haircut)
```

**Rationale:**
- Backtest overstates due to perfect fills, no emotional override
- Paper trading closer to reality but still simulation
- Live trading includes slippage, emotions, execution delays

**Source:** Consolidation of all three conflicting documents + empirical adjustment

---

### Risk-Reward Targets

**Minimum R:R Standards (NOT negotiable):**
```
All Strategies: 3:1 minimum (was 2:1, RAISED per research)
ORB Specific: 3:1 minimum, 4:1 target
Mean Reversion: 2:1 acceptable (smaller moves expected)
```

**Why 3:1 Minimum:**
```
50% win rate @ 2:1 R:R:
Gross expectancy: 0.5%
Transaction costs: 0.35%
Net expectancy: 0.15% (MARGINAL)

50% win rate @ 3:1 R:R:
Gross expectancy: 1.0%
Transaction costs: 0.35%
Net expectancy: 0.65% (ROBUST)
```

**Source:** STRATEGY_2_IMPLEMENTATION_ADDENDUM.md + expectancy math

---

### Win Rate Expectations

**Strategy-Specific Targets:**

**ORB (Asymmetric Breakout):**
- Expected: 15-30%
- Acceptable: >20%
- Red Flag: <15% (strategy broken)

**Mean Reversion (5-Day Washout):**
- Expected: 60-70%
- Acceptable: >55%
- Red Flag: <50% (no edge)

**GMM Regime (Long-Only):**
- Expected: 60-80% (in-regime only)
- Acceptable: >55%
- Red Flag: <50%

**DO NOT:**
- Compare win rates across strategy types
- Target 90%+ win rates (overfitted)
- Reject 30% win rate if R:R >3:1 (expectancy matters more)

**Source:** Strategy-specific research consolidation

---

## Part 4: Strategy Implementation Standards

### Opening Range Breakout (ORB) - Authoritative Spec

**Entry Requirements (ALL MANDATORY):**
```python
# 1. Opening Range Definition
opening_range_start = '09:30'  # NYSE open
opening_range_end = '10:00'    # 30 minutes (NOT negotiable)
or_high = data.between_time('09:30', '10:00')['high'].max()
or_low = data.between_time('09:30', '10:00')['low'].max()

# 2. Breakout Signal
price_breakout = data['close'] > or_high  # Long only

# 3. Volume Confirmation (MANDATORY 2.0x)
volume_ma = data['volume'].rolling(20).mean()
volume_confirmation = data['volume'] > (volume_ma * 2.0)  # 2.0x HARDCODED

# 4. Directional Bias
directional_bias = data['close'].iloc[0] > data['open'].iloc[0]  # Opening bar

# 5. Combined Entry
entry_signal = price_breakout & volume_confirmation & directional_bias
```

**Exit Requirements:**
```python
# Primary Exit: End of Day (MANDATORY)
eod_exit = data.index.time >= pd.Timestamp('15:55').time()  # 3:55 PM ET

# Secondary Exit: ATR Stop (2.5x)
atr_stop = entry_price - (atr * 2.5)
stop_exit = data['close'] < atr_stop
```

**Position Sizing:**
```python
# Use Week 1 implementation
position_size = calculate_position_size_atr(
    init_cash=10000,
    close=data['close'],
    atr=atr,
    atr_multiplier=2.5,
    risk_pct=0.01  # 1% (NOT 2% - see position sizing section)
)
```

**Backtest Requirements:**
```
Fees: 0.002 (0.2%)
Slippage: 0.0015 (0.15%)
Target Sharpe: >2.0
Target R:R: >3:1
Expected Win Rate: 15-30%
Trade Count: >100 (10 years data)
```

**Source:** STRATEGY_2_IMPLEMENTATION_ADDENDUM.md (authoritative ORB spec)

---

### TFC (The Strat) Methodology - CLARIFICATION

**CONTRADICTION RESOLVED:**

**ABANDONED (CONFIRMED):**
- TFC Confidence Scoring System (6-7 parameters)
- Weighted combinations of continuity metrics
- Complex scoring algorithms

**REASON:**
```
Parameters: 6-7
Required observations: 60-140 (statistical minimum)
Available observations: ~50-70 (SPY daily data)
Overfitting probability: >85%

Cross-validation consensus (3 independent analyses):
ABANDON TFC scoring entirely
```

**KEPT (WORKING CODE):**
- core/analyzer.py - STRAT bar classification (2-1-3 system)
- core/components.py - Pattern detection (2-2 continuation, reversals)
- core/triggers.py - Intrabar mechanics

**POTENTIAL FUTURE USE:**
- Simple TFC alignment (4/4 or 3/4 aligned bars)
- ZERO parameters (binary: aligned or not aligned)
- Can test AFTER GMM if desired (low priority)

**Source:** Medium_Articles_Research_Findings.md + HANDOFF.md confirmation

---

## Part 5: Risk Management Framework

### Three-Layer Risk Architecture

**Layer 1: Position Sizing Constraint (IMPLEMENTED)**
```
Capital constraint: position_value <= 100% of capital
Risk constraint: position_risk <= 2% of capital (or 1% per empirical results)
Implementation: utils/position_sizing.py
Status: COMPLETE (Gate 1 PASSED Oct 14)
```

**Layer 2: Portfolio Heat Management (NOT STARTED)**
```
Definition: Total risk across ALL open positions
Hard Limit: 6-8% of total portfolio (NO EXCEPTIONS)
Gate Function: can_take_position(proposed_risk, current_heat) → bool

Example:
Capital: $100,000
Position 1: $2,000 at risk (2%)
Position 2: $2,500 at risk (2.5%)
Position 3: $2,000 at risk (2%)
Total Heat: $6,500 (6.5%)

New Signal: Would add $2,000 risk
Decision: PASS (would exceed 8% limit)

Implementation: utils/portfolio_heat.py (TO CREATE)
Status: NOT STARTED (Week 1 incomplete)
```

**Layer 3: Regime Awareness (NOT STARTED)**
```
Purpose: When to trade vs when to stay out
Method: GMM regime detection (Yang-Zhang vol + SMA crossover)
Proven Improvement: Sharpe 0.63→1.00, MDD -34%→-15%

Implementation: utils/regime_detector.py (TO CREATE)
Status: NOT STARTED (Week 2-3 future work)
```

**Source:** Medium_Articles_Research_Findings.md Concept 4 + HANDOFF.md Layer descriptions

---

### Market Hours Filtering (CRITICAL - MANDATORY)

**THE RULE:**
ALL data MUST be filtered to NYSE regular trading hours (RTH) BEFORE resampling or analysis.

**Failure to do this:**
- Creates phantom bars on weekends/holidays
- Invalid STRAT classifications (Saturday 2-2 reversals that don't exist)
- Trades execute on non-trading days (impossible in live)

**Correct Implementation:**
```python
import pandas_market_calendars as mcal

# 1. Get NYSE calendar
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start, end_date=end)
trading_days = schedule.index

# 2. Filter BEFORE resampling
df = df[df.index.date.isin(trading_days.date)]

# 3. THEN resample
resampled = df.resample('1H', origin='start').agg(...)
```

**NEVER:**
- Resample first, then filter (creates phantom bars)
- Assume weekends/holidays excluded (they're not in raw data)
- Skip verification (use assert to confirm no weekends)

**Verification:**
```python
# After filtering, verify no weekends
weekends = df.index.dayofweek.isin([5, 6])
assert weekends.sum() == 0, f"Found {weekends.sum()} weekend bars"
```

**Source:** CLAUDE.md + multiple debugging sessions

---

## Part 6: VectorBT Pro Integration Patterns

### Position Sizing Integration (VERIFIED)

**Correct Usage:**
```python
import vectorbtpro as vbt

# Calculate position sizes (returns pd.Series of shares)
position_sizes = calculate_position_size_atr(
    init_cash=10000,
    close=data['close'],
    atr=atr_series,
    atr_multiplier=2.5,
    risk_pct=0.01
)

# Use with VectorBT Pro
pf = vbt.PF.from_signals(
    close=data['close'],
    entries=long_entries,
    exits=long_exits,
    size=position_sizes,      # pd.Series of shares
    size_type=None,           # Defaults to 'amount' (shares)
    init_cash=10000,
    fees=0.002,
    slippage=0.0015
)
```

**Verified Formats:**
- Scalar: size=10.0 (fixed shares) ✓
- Series: size=pd.Series(...) (variable per bar) ✓
- Array: size=np.array(...) (also works) ✓

**VBT Capital Protection:**
- If requested position > available capital → VBT automatically caps via partial fill
- This is SECONDARY safety (our formula is PRIMARY)
- Tested empirically: Requested 200 shares @ $100 with $10k → Executed 100 shares

**Source:** VBT_INTEGRATION_PATTERNS.md (Session 2A-2B research)

---

### Data Fetching Patterns

**Alpaca Data via VBT Pro:**
```python
# Set credentials
vbt.AlpacaData.set_custom_settings(
    client_config=dict(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )
)

# Fetch data
data = vbt.AlpacaData.pull(
    'SPY',
    start='2024-01-01',
    end='2024-12-31',
    timeframe='5Min'  # or '1D', '1H', etc.
).get()  # Use .get() method (NOT .data attribute)
```

**Known Issues:**
- Daily timeframe sometimes returns 0 bars (API or entitlement issue)
- Adding tz='America/New_York' can cause 0 bars (API parsing issue)
- Workaround: Fetch without tz, then use .tz_localize() post-fetch

**Source:** WEEK2_HANDOFF.md debugging notes + VBT_INTEGRATION_PATTERNS.md

---

## Part 7: Validation & Testing Standards

### Out-of-Sample Testing (MANDATORY)

**Train-Test Split:**
```
Full Data: 2016-01-01 to 2025-12-31 (10 years)
Train: 2016-01-01 to 2023-01-01 (70%)
Test: 2023-01-01 to 2025-12-31 (30%)
```

**Rules:**
1. Optimize parameters on TRAIN data only
2. Test on TEST data (never optimize on test)
3. Out-of-sample Sharpe must be within 0.3 of in-sample Sharpe

**Pass Criteria:**
```
In-sample Sharpe: 2.2
Out-of-sample Sharpe: >1.9 (PASS)

In-sample Sharpe: 2.2
Out-of-sample Sharpe: 1.5 (FAIL - gap >0.3, overfitted)
```

**Source:** VALIDATION_PROTOCOL.md Phase 2

---

### Walk-Forward Analysis

**Configuration:**
```
Train Period: 365 days (1 year)
Test Period: 90 days (3 months)
Step Forward: 30 days (1 month)

Creates overlapping windows:
Window 1: Train Jan2019-Dec2019, Test Jan2020-Mar2020
Window 2: Train Feb2019-Jan2020, Test Feb2020-Apr2020
... (rolling forward monthly)
```

**Efficiency Metric:**
```
WF Efficiency = (Passing Windows / Total Windows)

>70%: Robust strategy (PASS)
50-70%: Marginal (monitor closely)
<50%: Unreliable (FAIL)
```

**Parameter Stability Test:**
```python
# After walk-forward, check if parameters stable
param_std = wf_results['optimal_param'].std()

if param_std > 0.2 * param_mean:
    print("Parameters vary >20% - unstable strategy")
else:
    print("Parameters stable - robust strategy")
```

**Source:** VALIDATION_PROTOCOL.md Phase 3

---

### Red Flags for Overfitting

| Sign | What It Means | Action |
|------|---------------|--------|
| Backtest Sharpe >3.0 | Almost certainly overfitted | Re-validate with simpler params |
| Perfect equity curve | No realistic drawdowns | Lookahead bias in code |
| 90%+ win rate | Too good to be true | Data leakage somewhere |
| Huge gap (in-sample vs out-sample) | Parameters don't generalize | Reduce parameters |
| Parameters change wildly in WF | Unstable strategy | Use fixed params |

**Research Warning:**
> "Backtested Sharpe ratios exceeding 3.0 typically indicate overfitting rather than genuine edge"

**Source:** VALIDATION_PROTOCOL.md overfitting section

---

## Part 8: Professional Development Standards

### VBT-First Methodology (MANDATORY)

**The Workflow:**
```
1. Read VBT README (navigation guide)
2. Search VBT LLM Docs for relevant methods
3. Use Python introspection (vbt.phelp(), vbt.pdir())
4. Test minimal example BEFORE full implementation
5. ONLY THEN write production code
```

**Example:**
```python
# STEP 1: Search documentation
vbt.find_docs(vbt.PF.from_signals)

# STEP 2: Get method signature
vbt.phelp(vbt.PF.from_signals)

# STEP 3: Test minimal example
close = pd.Series([100, 101, 102])
entries = pd.Series([True, False, False])
exits = pd.Series([False, False, True])

pf = vbt.PF.from_signals(close, entries, exits, size=10, init_cash=10000)
print(pf.total_return())  # Does it work?

# STEP 4: Only now implement full production code
```

**Consequences of Violation:**
- Implementation errors (ORB 0 bars after filtering example)
- Wasted time on debugging avoidable issues
- QuantGPT round-trips (could have been avoided with VBT docs)

**Source:** CLAUDE.md + empirical learning from mistakes

---

### Documentation Standards

**Active Development:**
```
Location: docs/active/<branch-name>/
Purpose: Current implementation docs
Lifecycle: Created during implementation, archived when branch merged/abandoned
```

**Research (Timeless):**
```
Location: docs/research/
Purpose: Mathematical truths, validated methodologies
Lifecycle: NEVER changes (add new files, don't edit existing)
```

**Archives:**
```
Location: docs/archives/<branch-name>/
Purpose: Historical decisions, negative results (valuable learning)
Lifecycle: Frozen (never edited after archiving)
```

**History:**
```
Location: docs/history/
Purpose: Session-by-session timeline
Lifecycle: Append-only (never edit past entries)
```

**Source:** DOCUMENTATION_INDEX.md + file management policy

---

## Part 9: What NOT To Do (Critical Don'ts)

### Abandoned Approaches (DO NOT PURSUE)

**1. TFC Confidence Scoring**
- 6-7 parameters = guaranteed overfitting
- Requires 60-140 observations, have ~50
- Cross-validation consensus: ABANDON
- Overfitting probability: >85%

**2. Strategy 1 RSI Mean Reversion (Current Implementation)**
- Average trade +0.27%, 14-day holds
- RSI too sensitive (cuts winners short)
- Replacement: 5-day washout (60-70% win rate)

**3. 2:1 R:R Targets**
- Net expectancy only 0.15% after 0.35% costs
- RAISED to 3:1 minimum (net expectancy 0.65%)

**4. Position Sizing Without Capital Constraint**
- Original bug: 81.8% mean, 142.6% max
- FIXED: Dual constraint (risk + capital)

**5. Resampling Before Market Hours Filtering**
- Creates phantom weekend/holiday bars
- Invalid STRAT classifications
- ALWAYS filter to trading days first

**Source:** Medium_Articles_Research_Findings.md + empirical failures

---

### Common Mistakes to Avoid

**VBT Integration:**
- Assuming methods exist without verification
- Skipping the README navigation guide
- Not testing minimal examples first
- Using .data attribute instead of .get() method

**Position Sizing:**
- Using 2% risk without checking if mean exceeds 30%
- Forgetting capital constraint
- Mixing sizing methodologies (ATR for some, vol for others)

**Strategy Development:**
- Optimizing on out-of-sample data
- Accepting Sharpe <1.0 in backtest
- Ignoring transaction costs
- Not validating market hours filtering

**Documentation:**
- Working without reading HANDOFF.md
- Assuming docs are current (check last updated date)
- Not updating HANDOFF at end of session

**Source:** Consolidation of all mistake documentation

---

## Part 10: Quick Reference Tables

### Position Sizing Quick Reference

| Parameter | Old (Docs) | New (Empirical) | Rationale |
|-----------|-----------|-----------------|-----------|
| risk_pct | 0.02 (2%) | 0.01 (1%) | Achieves 10-30% target |
| atr_multiplier | 2.5 | 2.5 | Optimal for ORB |
| volume_threshold | 1.5x | 2.0x | Research-validated |

### Performance Targets Quick Reference

| Metric | Backtest | Paper | Live | Notes |
|--------|----------|-------|------|-------|
| Sharpe | >2.0 | >0.8 | >0.5 | 50% haircut assumed |
| R:R | >3:1 | >3:1 | >2:1 | Live execution harder |
| Win Rate (ORB) | 15-30% | 15-30% | 15-30% | Asymmetric strategy |
| Win Rate (MR) | 60-70% | 55-65% | 50-60% | Higher frequency |
| Max Drawdown | <25% | <25% | <30% | Live has slippage |

### Validation Phases Quick Reference

| Phase | Duration | Key Metric | Pass Criteria |
|-------|----------|------------|---------------|
| Unit Tests | Week 1 | All tests pass | 100% pass rate |
| Backtest | Weeks 2-3 | Sharpe, R:R | Sharpe >2.0, R:R >3:1 |
| Walk-Forward | Weeks 4-5 | WF Efficiency | >70% |
| Paper Trading | Months 3-8 | Paper Sharpe | >0.8 |
| Live Trading | Month 9+ | Live Sharpe | >0.5 |

---

## Contradictions Resolved Summary

### 1. Volume Confirmation
**Was:** 1.5x vs 2.0x (conflicting docs)
**Now:** 2.0x MANDATORY (research-backed)

### 2. Position Sizing Risk %
**Was:** 2% (documentation standard)
**Now:** 1% recommended (empirical validation)

### 3. Sharpe Targets
**Was:** 0.8 vs 2.0 vs 2.396 (conflicting standards)
**Now:** Backtest >2.0, Paper >0.8, Live >0.5 (phased targets)

### 4. R:R Minimums
**Was:** 2:1 acceptable
**Now:** 3:1 minimum (transaction cost math)

### 5. TFC Approach
**Was:** Some docs still referenced scoring
**Now:** Scoring ABANDONED, classification code KEPT

### 6. Position Sizing Methods
**Was:** Confusion about ATR vs Garman-Klass vs Yang-Zhang
**Now:** ATR for ORB (current), others for future strategies

### 7. Market Hours Filtering
**Was:** Implicit requirement
**Now:** MANDATORY CRITICAL rule with code examples

---

## Using This Document

**For New Features:**
1. Read relevant section here first
2. Cross-reference with Medium_Articles_Research_Findings.md for deeper theory
3. Check VALIDATION_PROTOCOL.md for testing requirements
4. Check STRATEGY_2_IMPLEMENTATION_ADDENDUM.md if working on ORB

**For Debugging:**
1. Check "What NOT To Do" section
2. Review VBT Integration Patterns
3. Verify market hours filtering applied
4. Check position sizing constraints

**For Performance Issues:**
1. Review Performance Targets section
2. Check if out-of-sample tested
3. Verify transaction costs applied
4. Check Red Flags for Overfitting

**For Research Questions:**
1. THIS document for consolidated practical guidance
2. Medium_Articles_Research_Findings.md for deep theory
3. VALIDATION_PROTOCOL.md for testing methodology

---

## Files Consolidated Into This Document

**Archived (moved to docs/research/ARCHIVE/):**
1. Advanced_Algorithmic_Trading_Systems.md - General algo trading (136 lines)
2. Algorithmic Trading System.md - More general trading (69 lines)
3. Algorithmic trading systems with asymmetric risk-reward profiles.md - Asymmetric strategies (145 lines)
4. CRITICAL_POSITION_SIZING_CLARIFICATION.md - Position sizing deep dive (687 lines)
5. STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md - Comprehensive strategy guide (2139 lines)

**Moved to Active Development (docs/active/risk-management-foundation/):**
6. WEEK1_EXECUTIVE_BRIEF.md - Week 1 implementation brief (222 lines)

**Remaining Separate (NOT consolidated):**
- Medium_Articles_Research_Findings.md - Foundational cross-validated research (1550 lines)
- VALIDATION_PROTOCOL.md - Complete 5-phase testing methodology (700 lines)
- STRATEGY_2_IMPLEMENTATION_ADDENDUM.md - ORB-specific corrections (1260 lines)

**Already in Active Development (correct location):**
- GATE1_RESULTS.md
- VBT_INTEGRATION_PATTERNS.md
- WEEK2_HANDOFF.md

**Project-Specific (separate):**
- PROJECT_AUDIT_2025-10-16.md (should move to docs/active/)
- DOCUMENTATION_INDEX.md (navigation guide, will be updated)

---

**Total Consolidated:** 3,398 lines → This single authoritative document
**Total Eliminated Redundancy:** ~60% (multiple overlapping explanations)
**Contradictions Resolved:** 7 major conflicts documented and resolved

---

**Last Updated:** October 18, 2025
**Next Review:** After completing current implementation phase
**Maintained By:** Updated as new research findings are validated
