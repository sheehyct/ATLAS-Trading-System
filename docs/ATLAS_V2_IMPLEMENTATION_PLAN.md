# ATLAS v2.0 Implementation Plan (Layer 1)

**Created**: October 26, 2025
**Updated**: November 2025 (Session 20 - Multi-Layer Architecture Context)
**Status**: Layer 1 (ATLAS) nearing completion - Phase F validation next
**Architecture Source**: docs/SYSTEM_ARCHITECTURE/

**CRITICAL CONTEXT - Multi-Layer Architecture (Session 20)**:
- **Layer 1 (ATLAS)**: Regime detection + equity strategies (THIS DOCUMENT)
- **Layer 2 (STRAT)**: Pattern recognition (Sessions 22-27, PENDING)
- **Layer 3 (Execution)**: Capital-aware deployment - options ($3k optimal) OR equities ($10k+ optimal)

**Implementation Status**:
- Academic Jump Model (Layer 1): Phases A-E COMPLETE, Phase F validation next (Session 21)
- ATLAS equity strategies: PENDING (will implement after Phase F OR defer for STRAT priority)
- STRAT integration: PENDING (begins Session 22 after Phase F completes)

**Capital Requirements for Layer 1 (ATLAS)**:
- Minimum Viable Capital: $10,000 (equity strategies require full position sizing)
- With $3,000: CAPITAL CONSTRAINED (0.06% actual vs 2% target risk)
- User Decision: $3,000 starting capital (risk management, not undercapitalization)
- Deployment Plan: Paper trade ATLAS ($10k sim) + Live trade STRAT options ($3k real)

---

## Implementation Overview

This plan implements ATLAS v2.0 as Layer 1 in a multi-layer trading architecture. ATLAS provides regime detection (COMPLETE via Academic Jump Model) and optional equity strategy execution (PENDING).

**UPDATED PRIORITY (Session 20)**: Focus on completing Phase F validation for Academic Jump Model (Layer 1 regime detection), then prioritize STRAT integration (Layer 2) over ATLAS equity strategies due to $3,000 capital constraint making STRAT + Options the optimal deployment path.

**Key Principle**: Validate each component before building dependencies. No strategy proceeds to Phase 2 until Phase 1 foundations are proven.

---

## Phase 1: Foundation Infrastructure (Weeks 1-2)

**Goal**: Establish core abstractions and regime detection before any strategy implementation.

### 1.1 Directory Structure Setup

**Files to Create**:
```
strategies/
  __init__.py
  base_strategy.py          # Abstract base class

regime/
  __init__.py
  jump_model.py             # Jump Model regime detection
  regime_allocator.py       # Regime-based capital allocation

backtesting/
  __init__.py
  backtest_engine.py        # VectorBT Pro wrapper
  walk_forward.py           # Walk-forward validation
```

**Files Already Exist** (no changes needed):
```
utils/position_sizing.py    # Gate 1 PASSED
utils/portfolio_heat.py     # Gate 2 PASSED
data/alpaca_client.py       # Working
data/mtf_manager.py         # Working
```

**Validation Criteria**:
- [ ] All __init__.py files created
- [ ] Import structure works (no circular imports)
- [ ] pytest discovers all test modules

---

### 1.2 BaseStrategy Abstract Class

**File**: `strategies/base_strategy.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/2_DIRECTORY_STRUCTURE_AND_STRATEGY_IMPLEMENTATION.md` (lines 89-340)

**Critical Components**:
1. Abstract methods: `generate_signals()`, `calculate_position_size()`, `validate_parameters()`
2. Concrete method: `backtest()` - integrates with VectorBT Pro
3. Performance metrics: `get_performance_metrics()` - standardized across all strategies

**VectorBT Integration Pattern** (VERIFIED from existing code):
```python
pf = vbt.Portfolio.from_signals(
    close=data['Close'],              # Capital case
    entries=signals['entry_signal'],
    exits=signals['exit_signal'],
    size=position_sizes,              # Share counts (integers)
    size_type='amount',               # CRITICAL: 'amount' not 'value'
    init_cash=initial_capital,
    fees=0.0015,                      # 15 bps
    slippage=0.0015,                  # 15 bps
    sl_stop=stop_distance,
    freq='1D'
)
```

**Validation Criteria**:
- [ ] BaseStrategy class passes mypy type checking
- [ ] All abstract methods defined correctly
- [ ] backtest() method integrates with VectorBT Pro
- [ ] get_performance_metrics() returns all required metrics
- [ ] Test strategy (simple SMA crossover) can inherit and run

**Testing**:
```python
# tests/test_core/test_base_strategy.py

class SimpleTestStrategy(BaseStrategy):
    """Minimal strategy for testing BaseStrategy interface."""
    # Implement required methods with simple logic
    pass

def test_base_strategy_interface():
    """Verify BaseStrategy can be inherited and used."""
    strategy = SimpleTestStrategy()
    # Test signal generation
    # Test position sizing
    # Test backtest execution
    # Test performance metrics
```

---

### 1.3 Jump Model Regime Detection

**File**: `regime/jump_model.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 432-476)

**Algorithm**:
```python
# Yang-Zhang volatility calculation
yz_vol = calculate_yang_zhang_vol(ohlc, window=20)

# Normalized jump metric
jump_metric = abs(returns.iloc[-1]) / yz_vol.iloc[-1]

# Logistic probability function
jump_prob = 1 / (1 + np.exp(-jump_metric))

# Regime classification
if jump_prob > 0.70:
    if returns.iloc[-1] > 0:
        regime = "TREND_BULL"
    else:
        regime = "TREND_BEAR"
elif jump_prob > 0.30:
    regime = "TREND_NEUTRAL"
else:
    regime = "CRASH"
```

**Validation Criteria**:
- [ ] Yang-Zhang volatility calculation matches reference implementation
- [ ] Jump probability in [0, 1] range
- [ ] Regime classification deterministic
- [ ] Historical accuracy >70% on known bull/bear periods (2020-2024)

**Testing**:
```python
# Test on known market regimes
# Mar 2020: Should detect CRASH or TREND_BEAR
# Apr-Dec 2020: Should detect TREND_BULL
# 2022: Should detect TREND_BEAR
# 2023-2024: Should detect TREND_BULL
```

---

### 1.4 Regime Allocator

**File**: `regime/regime_allocator.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 484-564)

**Allocation Tables**:
```python
BULL_ALLOCATION = {
    '52_week_high': 0.30,
    'quality_momentum': 0.25,
    'semi_vol_momentum': 0.15,
    'ibs_mean_reversion': 0.10,
    'orb': 0.10,
    'cash': 0.10,
}

NEUTRAL_ALLOCATION = {
    '52_week_high': 0.20,
    'quality_momentum': 0.30,
    'ibs_mean_reversion': 0.20,
    'cash': 0.30,
}

BEAR_CONSERVATIVE = {
    'quality_momentum': 0.20,
    'cash': 0.80,
}
```

**Validation Criteria**:
- [ ] All allocations sum to 1.0
- [ ] Regime detection triggers correct allocation
- [ ] Allocation transitions smooth (no whipsaw)

---

## Phase 2: Foundation Strategies (Weeks 3-4)

**Goal**: Implement and validate Tier 1 foundation strategies.

### 2.1 52-Week High Momentum Strategy

**File**: `strategies/high_momentum_52w.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 149-196)

**Logic**:
```python
# Entry Signal
price_52w_high = close.rolling(252).max()
distance_from_high = close / price_52w_high
entry_signal = distance_from_high >= 0.90

# Exit Signal
exit_signal = distance_from_high < 0.70

# Rebalance: Semi-annual (February, August)
```

**Performance Targets**:
- Sharpe Ratio: 0.8-1.2
- Win Rate: 50-60%
- CAGR: 10-15%
- Max Drawdown: -25% to -30%

**Validation Criteria**:
- [ ] Backtest on SPY (2015-2024) within target ranges
- [ ] Walk-forward degradation <30%
- [ ] Works in TREND_BULL and TREND_NEUTRAL regimes
- [ ] Turnover ~50% semi-annually (verify transaction cost feasibility)

---

### 2.2 Quality-Momentum Strategy

**File**: `strategies/quality_momentum.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 199-255)

**Logic**:
```python
# Quality Score (filter bottom 50%)
quality_score = (
    0.40 * roe_rank +
    0.30 * earnings_quality +
    0.30 * (1 / leverage_rank)
)
quality_filter = quality_score >= 0.50

# Momentum Score
momentum_score = price.pct_change(252).shift(21)

# Combined Signal
eligible = quality_filter
momentum_rank = momentum_score[eligible].rank(pct=True)
entry_signal = momentum_rank >= 0.50
```

**Performance Targets**:
- Sharpe Ratio: 1.3-1.7
- Win Rate: 55-65%
- CAGR: 15-22%
- Max Drawdown: -18% to -22%

**Critical Note**: Quality-Momentum works in ALL regimes (including TREND_BEAR), making it the portfolio anchor.

**Data Requirements**:
- Fundamental data: ROE, earnings quality, leverage
- Source: TBD (Alpaca provides fundamentals?)
- Fallback: Use quality ETF proxies (QUAL, SPHQ) as starting point

**Validation Criteria**:
- [ ] Backtest on quality universe (2015-2024)
- [ ] Quality filter reduces left-tail risk (verify lower max DD)
- [ ] Works in bear markets (2018, 2020, 2022 performance)
- [ ] Walk-forward degradation <30%

---

## Phase 3: Tactical Strategies (Weeks 5-6)

### 3.1 Semi-Volatility Momentum

**File**: `strategies/semi_vol_momentum.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 263-315)

**Logic**: Already documented in v1.0, retain implementation approach.

**Modification Required**:
- Add regime filter (only TREND_BULL)
- Add volatility circuit breaker (exit if market vol >22%)

---

### 3.2 IBS Mean Reversion

**File**: `strategies/ibs_mean_reversion.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 318-377)

**Logic**:
```python
# Internal Bar Strength
ibs = (close - low) / (high - low)

# Entry Signal
entry_signal = (
    (ibs < 0.20) &
    (close > sma_200) &
    (volume > volume_ma * 2.0)
)

# Exit Signals (dual system)
exit_signal_profit = ibs > 0.80
exit_signal_time = days_held >= 3
exit_signal_stop = close < (entry_price - 2.5 * atr)
```

**Performance Targets**:
- Sharpe Ratio: 1.5-2.0
- Win Rate: 65-75%
- Average Hold: 1-3 days

**Validation Criteria**:
- [ ] Backtests on liquid stocks (SPY, QQQ components)
- [ ] Volume confirmation reduces false signals
- [ ] Time stop prevents dead money
- [ ] Works best in TREND_NEUTRAL (verify negative correlation with momentum)

---

## Phase 4: ORB Modifications (Week 7)

**Goal**: Enhance existing ORB strategy with v2.0 requirements.

**File**: `strategies/orb.py` (EXISTS - modify)

**Required Changes**:
1. Add transaction cost analysis (0.15-0.25% per trade)
2. Restrict to S&P 500 only (most liquid)
3. Add minimum $50M daily volume filter
4. Regime filter: TREND_BULL only

**Validation Criteria**:
- [ ] Transaction cost impact analyzed
- [ ] Net Sharpe after costs: 1.2-1.8 (down from 1.5-2.5)
- [ ] Volume filter reduces slippage
- [ ] Walk-forward validation with realistic costs

---

## Phase 5: Portfolio Manager (Week 8)

**Goal**: Coordinate multiple strategies with regime-aware allocation.

**File**: `core/portfolio_manager.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/3_CORE_COMPONENTS_RISK_MANAGEMENT_AND_BACKTESTING_REQUIREMENTS.md` (lines 4-135)

**Key Methods**:
- `detect_regime()`: Call Jump Model
- `allocate_capital()`: Regime-based allocation
- `check_portfolio_heat()`: Aggregate risk across strategies
- `run_multi_strategy_backtest()`: Coordinated backtest

**Validation Criteria**:
- [ ] Regime detection works in backtest
- [ ] Capital allocation respects regime rules
- [ ] Portfolio heat never exceeds 8%
- [ ] Multi-strategy coordination reduces correlation

---

## Phase 6: Walk-Forward Validation (Week 9-10)

**Goal**: Validate all strategies out-of-sample.

**File**: `backtesting/walk_forward.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/4_WALK_FORWARD_VALIDATION_PERFORMANCE_TARGETS_AND_DEPLOYMENT.md`

**Configuration**:
```python
WALK_FORWARD_CONFIG = {
    'train_period': 365,  # 1 year
    'test_period': 90,    # 3 months
    'step_forward': 30,   # 1 month
    'min_trades': 20,
}
```

**Acceptance Criteria**:
- Average degradation <30%: PASS
- WF efficiency >50%: PASS
- Parameter stability: std dev <20% of mean

**If Strategy Fails Walk-Forward**:
1. Review parameter optimization (possible overfit)
2. Simplify strategy logic
3. Consider removing from portfolio
4. Do NOT proceed to paper trading

---

## Phase 7: Bear Protection (Week 11)

**Goal**: Implement tiered bear market protection.

**File**: `strategies/bear_protection.py`

**Implementation Source**: `docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md` (lines 568-685)

**Tiers**:
1. Conservative: 80% cash, 20% quality-momentum
2. Moderate: 65% cash, 15% low-vol ETF (USMV), 20% quality
3. Aggressive: 60% cash, 10% low-vol, 15% managed futures (DBMF), 15% quality

**Validation Requirements**:
- [ ] Jump Model accurately predicts TREND_BEAR (>70% accuracy)
- [ ] Backtest shows 5-10% improvement vs pure cash
- [ ] Duration estimation somewhat reliable (within 30 days)

**Implementation Priority**: Start with CONSERVATIVE only, add tiers after 6+ months validation.

---

## Phase 8: Paper Trading Deployment (Week 12+)

**Goal**: Deploy validated strategies to Alpaca paper trading.

**Requirements Before Deployment**:
- [ ] All strategies pass walk-forward validation
- [ ] Portfolio heat management proven (<8% limit)
- [ ] Transaction costs analyzed and acceptable
- [ ] 100+ trades minimum per strategy in backtest
- [ ] All unit tests passing
- [ ] HANDOFF.md updated with deployment status

**Paper Trading Duration**: Minimum 6 months before considering live trading.

**Performance Monitoring**:
- Daily: Portfolio heat, position count
- Weekly: Strategy performance vs backtest
- Monthly: Regime detection accuracy, allocation adherence

---

## Development Workflow

### For Each Strategy Implementation:

**Step 1: VBT MCP Research** (MANDATORY)
```python
# Search for relevant VBT functionality
mcp__vectorbt-pro__search(
    query="portfolio from_signals position sizing",
    asset_names=["examples", "api"],
    max_tokens=2000
)

# Verify API methods exist
mcp__vectorbt-pro__resolve_refnames(
    refnames=["vbt.Portfolio.from_signals"]
)

# Find usage examples
mcp__vectorbt-pro__find(
    refnames=["vbt.Portfolio.from_signals"],
    asset_names=["examples", "messages"]
)

# Test minimal example
mcp__vectorbt-pro__run_code(
    code="..."  # Minimal test
)
```

**Step 2: Implement Strategy**
- Inherit from BaseStrategy
- Implement required methods
- Add type hints
- Write docstrings

**Step 3: Unit Tests**
```python
# tests/test_strategies/test_52w_high.py

def test_signal_generation():
    """Test entry/exit signals."""
    pass

def test_position_sizing():
    """Test position size calculations."""
    pass

def test_backtest_execution():
    """Test full backtest runs."""
    pass

def test_performance_metrics():
    """Test metric extraction."""
    pass
```

**Step 4: Backtest Validation**
- Run backtest on historical data (2015-2024)
- Verify performance within target ranges
- Check edge cases (no NaN, no Inf)
- Validate column name consistency (Capital case)

**Step 5: Walk-Forward Validation**
- Run walk-forward analysis
- Check degradation <30%
- Verify parameter stability
- Document results in HANDOFF.md

**Step 6: Integration**
- Add to Portfolio Manager
- Test multi-strategy coordination
- Verify regime-based allocation
- Check portfolio heat management

---

## Critical Success Factors

### Code Quality Gates:

**Gate 1: Type Safety**
```bash
uv run mypy strategies/ regime/ core/
# Must pass with no errors
```

**Gate 2: Test Coverage**
```bash
uv run pytest --cov=strategies --cov=regime --cov=core --cov-report=html
# Target: >80% coverage
```

**Gate 3: VBT Integration**
```python
# All strategies must use VERIFIED VBT patterns
# No assumptions about API without verification
# Test minimal examples before full implementation
```

**Gate 4: Performance Validation**
```python
# Backtest must meet performance targets
# Walk-forward degradation <30%
# All metrics within acceptable ranges
```

---

## Risk Management Checkpoints

Before proceeding to next phase:

**After Phase 1**:
- [ ] BaseStrategy interface proven with test strategy
- [ ] Jump Model regime detection >70% accurate
- [ ] Regime allocator sums correctly

**After Phase 2**:
- [ ] Both foundation strategies pass walk-forward
- [ ] Performance targets met on backtests
- [ ] Transaction costs analyzed

**After Phase 3**:
- [ ] Tactical strategies complement foundation (low correlation)
- [ ] Portfolio Sharpe >1.0 in multi-strategy backtest
- [ ] All regimes have viable allocation

**After Phase 4**:
- [ ] ORB transaction costs acceptable
- [ ] All 5 strategies validated individually
- [ ] Ready for portfolio integration

**After Phase 5**:
- [ ] Multi-strategy backtest Sharpe >1.3
- [ ] Portfolio heat never exceeded 8%
- [ ] Regime transitions smooth

**After Phase 6**:
- [ ] ALL strategies pass walk-forward
- [ ] Parameter stability confirmed
- [ ] No overfitting detected

**After Phase 7**:
- [ ] Bear protection improves risk-adjusted returns
- [ ] Conservative tier ready for deployment
- [ ] Moderate/Aggressive tiers documented for future

**After Phase 8**:
- [ ] Paper trading matches backtest expectations (within 30%)
- [ ] No operational issues (data quality, execution)
- [ ] Risk limits enforced automatically

---

## Documentation Standards

### Update HANDOFF.md After Each Phase:

**Phase Completion Template**:
```markdown
## Phase X Complete: [Name]

**Date**: [Date]
**Duration**: [Days]
**Status**: COMPLETE

**Deliverables**:
- [ ] [Item 1]
- [ ] [Item 2]

**Validation Results**:
- Metric 1: [Result] (Target: [Target])
- Metric 2: [Result] (Target: [Target])

**Key Findings**:
- Finding 1
- Finding 2

**Next Actions**:
- Action 1
- Action 2
```

---

## OpenMemory Integration

**Store Critical Findings**:
```python
# After each phase completion
mcp__openmemory__openmemory_store(
    content="Phase X complete: [Summary of results, key metrics, findings]",
    tags=["phase_completion", "atlas_v2", "strategy_validation"]
)

# For walk-forward results
mcp__openmemory__openmemory_store(
    content="52-Week High WF results: Degradation 15%, Sharpe 1.1, Win Rate 55%",
    tags=["walk_forward", "52w_high", "validation"]
)
```

**Query for Similar Patterns**:
```python
# Before implementing new strategy
mcp__openmemory__openmemory_query(
    query="quality momentum strategy implementation backtesting",
    k=5
)
```

---

## Current Status

**Phase 1**: Ready to begin
**Next Action**: Implement BaseStrategy abstract class
**Blockers**: None
**Questions**:
1. Confirm data source for fundamental data (Quality-Momentum strategy)
2. Preferred order of implementation (BaseStrategy first, or Jump Model first?)
3. Should we commit architecture docs before starting implementation?

---

## Timeline Estimate

**Optimistic** (Full-time, no blockers): 8-10 weeks
**Realistic** (Part-time, normal blockers): 12-16 weeks
**Conservative** (Learning curve, data issues): 16-20 weeks

**First Milestone** (Phase 1 complete): 1-2 weeks
**Second Milestone** (Foundation strategies validated): 3-4 weeks
**Third Milestone** (All strategies validated): 6-8 weeks
**Final Milestone** (Paper trading ready): 10-12 weeks

---

**Last Updated**: October 26, 2025
**Next Review**: After Phase 1 completion
