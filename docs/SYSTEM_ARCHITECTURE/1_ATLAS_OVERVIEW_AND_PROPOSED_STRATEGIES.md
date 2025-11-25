# ATLAS System Architecture v2.0 - Layer 1 (Regime Detection)
## Evidence-Based Multi-Strategy Trading System

**Document Purpose**: This document describes ATLAS as Layer 1 in a multi-layer trading architecture. ATLAS provides regime detection and equity strategy execution. This is ONE component in a larger unified system (ATLAS + STRAT + Options).

**CRITICAL CONTEXT - Multi-Layer Architecture**:
- **Layer 1 (ATLAS)**: Regime detection + equity strategies (THIS DOCUMENT)
- **Layer 2 (STRAT)**: Pattern recognition for precise entry/exit levels (Sessions 22-27, PENDING)
- **Layer 3 (Execution)**: Capital-aware deployment - options ($3k optimal) OR equities ($10k+ optimal)

**Integration Status**: Layer 1 (ATLAS) nearing completion (Phase F validation next). Layers 2-3 implementation begins after Phase F completes.

**Capital Requirements for Layer 1 (ATLAS Equity Strategies)**:
- Minimum Viable Capital: $10,000 (full position sizing capability)
- With $3,000: CAPITAL CONSTRAINED, sub-optimal performance (0.06% actual risk vs 2% target)
- Recommendation: Paper trade ATLAS with $10k simulated while building capital, deploy STRAT+Options live with $3k

**Target Audience**: Development Team (Quantitative Developers)
**Version**: 2.0 (Layer 1 Implementation)
**Date**: November 2025 (Updated Session 20)
**System**: ATLAS (Adaptive Trading with Layered Asset System)

**Key Changes from v1.0**:
- Replaced GMM Regime Detection with simpler Jump Model approach
- Added 52-Week High Momentum and Quality-Momentum as foundation strategies (replacing Pairs Trading)
- Replaced Five-Day Washout with IBS Mean Reversion (superior Sharpe ratio 1.5-2.0)
- Retained Opening Range Breakout with enhanced volume confirmation requirements
- Retained Semi-Volatility Momentum Portfolio (validated academic foundation)
- Added tiered bear market protection framework (cash, low-vol ETFs, managed futures)
- Refined position sizing expectations to match empirical performance data

**Note**: These changes reflect extensive academic and research article analysis. Strategies remain subject to modification based on backtest results, walk-forward validation outcomes, and evolving market conditions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Strategic Philosophy](#strategic-philosophy)
3. [Core Strategy Pyramid](#core-strategy-pyramid)
4. [Regime Detection Framework](#regime-detection-framework)
5. [Technology Stack](#technology-stack)
6. [Directory Structure](#directory-structure)
7. [Strategy Implementations](#strategy-implementations)
8. [Core Components](#core-components)
9. [Risk Management Framework](#risk-management-framework)
10. [Backtesting Requirements](#backtesting-requirements)
11. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Architecture Philosophy

**Evidence-Based Strategy Selection**: All strategies must have academic validation or documented track records spanning multiple market cycles. No theoretical-only approaches.

**Modular Strategy Design**: Each strategy is self-contained with its own signal generation, position sizing, and exit logic. Strategies share common infrastructure but remain independent.

**Portfolio-Level Orchestration**: A central portfolio manager coordinates multiple strategies, enforces risk limits, and manages capital allocation based on market regime.

**Regime-Aware Allocation**: Capital allocation adjusts dynamically based on Jump Model regime detection, with tiered bear market protection replacing leveraged inverse ETFs.

**Vectorized Operations**: All calculations use pandas/numpy vectorization. No Python loops for data processing. VectorBT Pro compatibility is mandatory.

**Walk-Forward Validation**: All strategies must pass out-of-sample testing with <30% performance degradation from in-sample results.

### System Layers

```
+---------------------------------------------+
|   Regime Detection (Jump Model)            |
|   - Market state classification            |
|   - Capital allocation decisions           |
|   - Bear market protection activation      |
+---------------------------------------------+
                    |
        +-----------+-----------+
        |                       |
+-------------------+   +-------------------+
| Portfolio Manager |   | Bear Protection   |
| (Orchestration)   |   | - Cash allocation |
| - Capital alloc   |   | - Low-vol ETFs    |
| - Portfolio heat  |   | - Managed futures |
| - Multi-strategy  |   +-------------------+
+-------------------+
          |
  +-------+-------+-------+-------+
  |       |       |       |       |
+-------+ +-----+ +-----+ +-----+ +-------+
| 52Wk  | | QM  | | SVM | | IBS | | ORB   |
| High  | |     | |     | | MR  | |       |
+-------+ +-----+ +-----+ +-----+ +-------+
  |       |       |       |       |
  +-------+-------+-------+-------+
                |
+---------------------------------------------+
|        Risk Management Layer                |
|   - Position sizing (ATR-based)             |
|   - Portfolio heat (6-8% max)               |
|   - Stop loss system (multi-layer)          |
|   - Volume confirmation                     |
+---------------------------------------------+
                |
+---------------------------------------------+
|          Data Management                    |
|   - Multi-timeframe alignment               |
|   - Technical indicators (TA-Lib)           |
|   - Market hours filtering                  |
|   - Data quality validation                 |
+---------------------------------------------+
```

---

## Strategic Philosophy

### Why This Architecture (v2.0)

**Rationale**: These changes were made after extensive analysis of academic research papers, peer-reviewed studies, and empirical performance data across multiple market cycles. The v2.0 architecture prioritizes evidence-based strategies with documented track records over theoretical approaches.

**Key Improvements from v1.0**:

1. **Simplified Regime Detection**: Jump Model approach (44% annual turnover) provides superior performance vs GMM/HMM methods (141% turnover) with significantly lower complexity and transaction costs.

2. **Foundation Strategy Addition**: 
   - **52-Week High Momentum**: Lowest turnover (50% semi-annual), highest robustness, works in bull AND neutral regimes
   - **Quality-Momentum**: Validated Sharpe 1.55, works across all regimes due to quality filter providing downside protection

3. **Enhanced Mean Reversion**: IBS Mean Reversion (Sharpe 1.5-2.0) provides superior risk-adjusted returns vs Five-Day Washout, with daily signals creating more trading opportunities.

4. **Refined Breakout Strategy**: Opening Range Breakout retained but enhanced with mandatory 2x volume confirmation, reducing false breakout entries by ~40%.

5. **Proven Volatility Strategy**: Semi-Volatility Momentum retained with academic validation from Moreira & Muir (2017) showing Sharpe improvement from 0.8 -> 1.6-1.7.

**Disclaimer**: Strategy composition remains subject to change based on:
- Walk-forward validation results
- Out-of-sample performance degradation  
- Transaction cost analysis
- Regime-specific performance patterns
- Market microstructure changes

### Expected Portfolio Characteristics

| Metric | Conservative Target | Aspirational Target | Critical Threshold |
|--------|---------------------|---------------------|-------------------|
| **Portfolio Sharpe** | 1.0-1.3 | 1.5-1.8 | <0.5 (fail) |
| **CAGR** | 12-18% | 20-25% | <8% (fail) |
| **Max Drawdown** | -20% to -25% | -15% to -18% | >-30% (fail) |
| **Win Rate** | 45-55% | 55-65% | <40% (concerning) |
| **Volatility** | 15-18% | 12-15% | >25% (excessive) |

**Critical**: These targets assume proper position sizing, portfolio heat management, and regime-based allocation. Individual strategy performance will vary significantly.

---

## Core Strategy Pyramid

### Tier 1: Foundation Strategies (60-70% allocation in bull markets)

#### Strategy 1: 52-Week High Momentum Portfolio

**Academic Foundation**: Novy-Marx (2012), George & Hwang (2004) - documented momentum effect with lowest turnover

**Strategy Logic**:
```python
# Entry Signal
price_52w_high = close.rolling(252).max()
distance_from_high = close / price_52w_high

entry_signal = distance_from_high >= 0.90  # Within 10% of 52-week high

# Universe Selection
eligible_universe = sp500_constituents  # Or Russell 1000
position_count = 20-50  # Equal weight or volatility-scaled

# Exit Signal
exit_signal = distance_from_high < 0.70  # 30% off highs

# Rebalance: Semi-annual (February, August)
```

**Performance Targets**:
- Sharpe Ratio: 0.8-1.2
- Turnover: ~50% semi-annually
- Win Rate: 50-60%
- CAGR: 10-15%
- Max Drawdown: -25% to -30%

**Regime Allocation**:
- TREND_BULL: 30-40% of portfolio
- TREND_NEUTRAL: 20-25% of portfolio (STILL WORKS)
- TREND_BEAR: 0% (exit all positions)

**Implementation Priority**: PHASE 1 (Highest priority - foundation strategy)

**Risk Management**:
- Position sizing: Equal-weight OR inverse-volatility weighted
- Portfolio heat: Included in 6-8% total limit
- Stop loss: Violation of 70% threshold (30% off highs)
- Rebalance discipline: Semi-annual only, no intra-period adjustments

**Critical Success Factors**:
- Simple methodology = high robustness
- Works in neutral markets (unique advantage)
- Lowest maintenance of all strategies
- Perfect for cash account constraints

---

#### Strategy 2: Quality-Momentum Combination

**Academic Foundation**: Asness, Frazzini, Pedersen (2018) "Quality Minus Junk" + Jegadeesh & Titman momentum

**Strategy Logic**:
```python
# Quality Metrics (remove bottom 50%)
quality_score = (
    0.40 * roe_rank +           # Return on Equity
    0.30 * earnings_quality +    # Accruals ratio
    0.30 * (1 / leverage_rank)   # Low leverage preferred
)

quality_filter = quality_score >= 0.50  # Top 50% by quality

# Momentum Score (rank top 50%)
momentum_score = price.pct_change(252).shift(21)  # 12-month return, 1-month lag

# Combined Signal
eligible = quality_filter
momentum_rank = momentum_score[eligible].rank(pct=True)
entry_signal = momentum_rank >= 0.50  # Top 50% of quality stocks

# Position Count
position_count = 20-30

# Exit Signal
quarterly_rebalance = True  # Refresh quality + momentum quarterly
exit_signal = ~(quality_filter & (momentum_rank >= 0.40))  # Buffer to reduce turnover
```

**Performance Targets**:
- Sharpe Ratio: 1.3-1.7 (validated 1.55 in research)
- Turnover: 50-80% quarterly
- Win Rate: 55-65%
- CAGR: 15-22%
- Max Drawdown: -18% to -22%

**Regime Allocation**:
- TREND_BULL: 25-30% of portfolio
- TREND_NEUTRAL: 30-35% of portfolio (quality protects)
- TREND_BEAR: 20-30% of portfolio (DEFENSIVE - quality prevents blow-up)

**Implementation Priority**: PHASE 1 (Highest priority - all-weather strategy)

**Risk Management**:
- Position sizing: Inverse-volatility weighted preferred
- Quality filter prevents blow-up risk
- Quarterly rebalance maintains discipline
- No leverage allowed (quality premium sufficient)

**Critical Success Factors**:
- Quality filter reduces left-tail risk in bear markets
- Momentum enhances returns in bull markets
- Works across ALL market regimes (unique)
- 40% buffer on exit reduces excessive turnover

---

### Tier 2: Tactical Strategies (20-30% allocation)

#### Strategy 3: Semi-Volatility Momentum

**Academic Foundation**: Moreira & Muir (2017) "Volatility-Managed Portfolios"

**Strategy Logic**:
```python
# Realized Volatility (60-day lookback)
returns = close.pct_change()
realized_vol = returns.rolling(60).std() * np.sqrt(252)  # Annualized

# Target Volatility
target_vol = 0.15  # 15% annualized

# Position Scaling
vol_scalar = target_vol / realized_vol
vol_scalar = vol_scalar.clip(0.5, 2.0)  # Limit to 50%-200% of base position

# Momentum Signal (standard 12-1 momentum)
momentum = close.pct_change(252).shift(21)
entry_signal = momentum > 0  # Positive momentum only

# Position Size
base_position_size = calculate_position_size_atr(capital, close, atr)
adjusted_position = base_position_size * vol_scalar

# Regime Filter
active_regime = (realized_vol < 0.18) & (trend_bull == True)
```

**Performance Targets**:
- Sharpe Ratio: 1.4-1.8 (improvement from 0.8 base momentum)
- Turnover: ~100% annually (monthly rebalance)
- Win Rate: 50-60%
- CAGR: 15-20%
- Max Drawdown: -15% to -20%

**Regime Allocation**:
- TREND_BULL + Low Vol: 15-20% of portfolio
- TREND_BULL + High Vol: 5-10% of portfolio (reduced)
- TREND_NEUTRAL: 0% (sit out)
- TREND_BEAR: 0% (sit out)

**Implementation Priority**: PHASE 2 (After foundation strategies proven)

**Risk Management**:
- Volatility scaling is CORE mechanism (not optional)
- Hard cap at 2.0x leverage equivalent
- Circuit breaker: Exit all positions if portfolio vol >22%
- Only trade when market vol <18%

**Critical Success Factors**:
- Volatility targeting reduces crash risk
- Works best in trending, stable markets
- Requires accurate volatility estimation
- Monthly rebalance discipline critical

---

#### Strategy 4: IBS Mean Reversion (Replaces 5-Day Washout)

**Academic Foundation**: Connors Research, validated across 20+ years

**Strategy Logic**:
```python
# Internal Bar Strength
ibs = (close - low) / (high - low)

# Entry Signal
entry_signal = (
    (ibs < 0.20) &  # Closed in bottom 20% of daily range
    (close > sma_200) &  # Above 200-day MA (uptrend filter)
    (volume > volume_ma * 2.0)  # 2x volume confirmation MANDATORY
)

# Exit Signals (dual system)
exit_signal_profit = ibs > 0.80  # Closed in top 80% of range
exit_signal_time = days_held >= 3  # Maximum 3-day hold
exit_signal_stop = close < (entry_price - 2.5 * atr)  # ATR-based stop

exit_signal = exit_signal_profit | exit_signal_time | exit_signal_stop

# Position Limit
max_concurrent_positions = 3  # Limit correlation risk
```

**Performance Targets**:
- Sharpe Ratio: 1.5-2.0 (superior to 5-day washout)
- Turnover: High (daily signals)
- Win Rate: 65-75%
- Average Hold: 1-3 days
- CAGR: 8-12%
- Max Drawdown: -10% to -12%

**Regime Allocation**:
- TREND_BULL: 5-10% of portfolio
- TREND_NEUTRAL/CHOP: 15-20% of portfolio (THRIVES in chop)
- TREND_BEAR: 0% (mean reversion fails in crashes)

**Implementation Priority**: PHASE 2 (After foundation strategies)

**Risk Management**:
- Volume confirmation is MANDATORY (2x threshold non-negotiable)
- Max 3 concurrent positions (reduce correlation)
- 3-day time stop prevents dead money
- Only trade stocks >$50M daily volume

**Critical Success Factors**:
- Daily signals provide more opportunities than 5-day
- Volume confirmation filters false signals
- Short hold period reduces overnight risk
- Works best in choppy markets (negative correlation with momentum)

**Why NOT 5-Day Washout**:
- IBS has superior documented Sharpe ratio (1.5-2.0 vs unknown)
- Daily signals vs weekly = more opportunities
- Simpler logic = less parameter overfitting risk
- Better academic validation

---

### Tier 3: Opportunistic Strategies (10-20% allocation)

#### Strategy 5: Opening Range Breakout (ORB)

**Implementation Status**: Already implemented (strategies/orb.py)

**Modifications Required** (based on research):
```python
# CRITICAL: Transaction cost constraint
min_daily_volume = 50_000_000  # $50M daily volume minimum
eligible_universe = sp500_only  # Most liquid only

# Volume confirmation (MANDATORY - research validated)
volume_threshold = volume_20d_avg * 2.0

entry_signal = (
    (high > opening_range_high) &
    (volume > volume_threshold)  # 2x volume REQUIRED
)

# Regime Filter
regime_filter = trend_bull_only  # NOT neutral or bear
```

**Performance Targets** (adjusted for transaction costs):
- Sharpe Ratio: 1.2-1.8 (reduced from 1.5-2.5 due to costs)
- Win Rate: 15-25% (asymmetric strategy)
- Average Win: 3-5x average loss
- CAGR: 10-18%
- Max Drawdown: -20% to -25%

**Regime Allocation**:
- TREND_BULL: 5-10% of portfolio (most liquid only)
- TREND_NEUTRAL: 0% (breakouts fail in range)
- TREND_BEAR: 0% (false breakouts increase)

**Implementation Priority**: PHASE 3 (Already exists, needs modification)

**Critical Modifications Required**:
1. Add transaction cost analysis (0.15-0.25% per trade)
2. Implement volume confirmation (2x threshold)
3. Restrict to most liquid stocks only (S&P 500)
4. Reduce trading frequency to minimize costs
5. Increase per-trade size (amortize costs)

**Risk Management**:
- Only trade TREND_BULL regime
- Volume confirmation non-negotiable
- Transaction cost analysis before every trade
- Consider reducing frequency vs v1.0 implementation

---

## Regime Detection Framework

### Jump Model Implementation (Replaces GMM)

**Why Jump Model vs GMM**:
- Simpler: 3 parameters vs 6-7 for GMM
- More robust: Lower overfitting risk
- Faster: Real-time classification vs batch processing
- Proven: Used by institutional traders

**Jump Model Logic**:
```python
# Jump Probability Calculation
def calculate_jump_probability(returns, window=20):
    """
    Calculate probability of regime jump using Yang-Zhang volatility.
    
    High probability = Trend
    Low probability = Mean reversion
    """
    # Yang-Zhang volatility estimator
    yz_vol = calculate_yang_zhang_vol(ohlc, window)
    
    # Normalized jump metric
    jump_metric = abs(returns.iloc[-1]) / yz_vol.iloc[-1]
    
    # Probability via logistic function
    jump_prob = 1 / (1 + np.exp(-jump_metric))
    
    return jump_prob

# Regime Classification
jump_prob = calculate_jump_probability(returns)

if jump_prob > 0.70:
    if returns.iloc[-1] > 0:
        regime = "TREND_BULL"
    else:
        regime = "TREND_BEAR"
elif jump_prob > 0.30:
    regime = "TREND_NEUTRAL"
else:
    regime = "CRASH"  # Extreme volatility spike
```

**Regime States**:
1. **TREND_BULL** (Jump prob >70%, positive return)
2. **TREND_BEAR** (Jump prob >70%, negative return)
3. **TREND_NEUTRAL** (Jump prob 30-70%)
4. **CRASH** (Jump prob >90% or special indicators)

**Regime-Based Capital Allocation**:

### TREND_BULL (Jump confidence >70%, positive direction)
```python
BULL_ALLOCATION = {
    '52_week_high': 0.30,      # 30% - Foundation
    'quality_momentum': 0.25,   # 25% - All-weather
    'semi_vol_momentum': 0.15,  # 15% - Volatility-scaled
    'ibs_mean_reversion': 0.10, # 10% - Opportunistic
    'orb': 0.10,                # 10% - Asymmetric
    'cash': 0.10,               # 10% - Dry powder
}
# Total deployed: 90%
```

### TREND_NEUTRAL (Jump confidence 30-70%)
```python
NEUTRAL_ALLOCATION = {
    '52_week_high': 0.20,       # 20% - Still works
    'quality_momentum': 0.30,    # 30% - Quality protects
    'ibs_mean_reversion': 0.20,  # 20% - Thrives in chop
    'cash': 0.30,                # 30% - Defensive
}
# Total deployed: 70%
# Note: NO semi-vol momentum, NO ORB
```

### TREND_BEAR (Jump confidence >70%, negative direction)
```python
# Tiered Approach Based on Conviction
BEAR_CONSERVATIVE = {
    'quality_momentum': 0.20,    # 20% - Quality defense
    'cash': 0.80,                # 80% - Maximum preservation
}

BEAR_MODERATE = {
    'quality_momentum': 0.20,    # 20% - Quality defense
    'min_vol_etf': 0.15,         # 15% - USMV or SPLV
    'cash': 0.65,                # 65% - Reduced from 80%
}

BEAR_AGGRESSIVE = {
    'quality_momentum': 0.15,    # 15% - Minimal equity
    'min_vol_etf': 0.10,         # 10% - Low-vol ETF
    'managed_futures': 0.15,     # 15% - Crisis alpha
    'cash': 0.60,                # 60% - Base preservation
}
```

### CRASH (Jump prob >30% for crash indicators)
```python
CRASH_ALLOCATION = {
    'cash': 0.90,                # 90% - Maximum preservation
    'quality_momentum': 0.10,    # 10% - Only highest quality IF ANY
}
# All other strategies: 0%
```

**Bear Market Protection Selection Framework**:

```python
def select_bear_protection_tier(jump_confidence, expected_duration, risk_tolerance):
    """
    Decision tree for bear market protection level.
    
    Returns: 'conservative', 'moderate', or 'aggressive'
    """
    if jump_confidence < 0.70:
        return 'conservative'  # Uncertain = play it safe
    
    if expected_duration < 90:  # <3 months
        return 'conservative'  # Short bear = cash sufficient
    
    if risk_tolerance == 'low':
        return 'moderate'  # Low risk = add low-vol only
    
    if expected_duration > 180:  # >6 months
        return 'aggressive'  # Long bear = full protection suite
    
    return 'moderate'  # Default middle ground
```

---

## Bear Market Protection Framework

### Evidence-Based Hierarchy

Based on research from CASH_VS_OTHER_BEAR_MARKET_STRATEGIES.md:

**Performance in Bear Markets (Descending Order)**:
1. Managed Futures: +10-20% during major equity bear markets
2. Low-Vol ETFs: 2:1 improvement vs market (5% loss vs 10% market loss)
3. Quality-Momentum: Defensive characteristics, limited losses
4. Cash: 0% return, 0% risk
5. Treasury Bonds: UNRELIABLE (regime-dependent)6. Inverse Leveraged ETFs: SYSTEMATIC FAILURE (never use)

### Tier 1: Low-Volatility ETF Protection

**Recommended ETFs**:
- **USMV** (iShares MSCI USA Min Vol): 0.15% ER, optimizes portfolio-level volatility
- **SPLV** (Invesco S&P 500 Low Vol): 0.25% ER, simple low-volatility approach
- **PTLC** (Pacer Trendpilot US Large Cap): 0.60% ER, trend-following layer

**Empirical Performance**:
- 2022 Bear Market: USMV -9.4% vs -17% category average
- 2025 Tariff Correction: Low-vol outperformed by 5-8%
- Crisis Protection: Typically lose 50-60% less than market

**Implementation**:
```python
# Bear market allocation
if regime == "TREND_BEAR" and tier == "moderate":
    allocate_to_etf(
        symbol="USMV",
        allocation=0.15,  # 15% of portfolio
        rationale="Proven 2:1 bear market protection"
    )
```

**Critical Notes**:
- NOT a perfect hedge (still loses money in bears)
- 2:1 improvement is REALISTIC expectation
- Works best in sustained bears (3-12 months)
- Liquid, exchange-traded, cash account compatible
- No daily rebalancing decay like leveraged products

---

### Tier 2: Managed Futures Protection

**Recommended ETFs**:
- **DBMF** (iMGP DBi Managed Futures): 0.85% ER
- **KMLM** (KFA Mount Lucas Managed Futures): 0.90% ER
- **CTA** (Simplify Managed Futures): 0.55% ER

**Empirical Performance**:
- Typical bear market return: +10-20%
- Correlation to equities: Near zero
- Volatility: 15-25% annualized
- Crisis alpha: Significant positive during crashes

**Academic Foundation**:
- 800 years of trend-following validation
- Used by CalPERS and major institutions
- Positive skewness (rare for alternatives)
- Reduces portfolio maximum drawdowns

**Implementation**:
```python
# Aggressive bear protection
if regime == "TREND_BEAR" and tier == "aggressive":
    allocate_to_etf(
        symbol="DBMF",
        allocation=0.15,  # 15% of portfolio
        rationale="Crisis alpha + trend following"
    )
```

**Critical Notes**:
- High expense ratios (0.55-0.90%)
- Can underperform in choppy markets
- Requires sustained trends to work
- Adds complexity (not passive)
- But provides TRUE crisis protection

---

### Bear Protection Decision Tree

```
Is Jump Model confident in bear market? (>70%)
|
+-- NO: Use CONSERVATIVE tier (80% cash, 20% quality)
|
+-- YES: Check expected duration
    |
    +-- Short (<3 months): CONSERVATIVE tier
    |
    +-- Medium/Long (3+ months): Check confidence level
        |
        +-- 70-80% confidence: MODERATE tier
        |   (20% quality, 15% low-vol, 65% cash)
        |
        +-- >80% confidence: AGGRESSIVE tier
            (15% quality, 10% low-vol, 15% managed futures, 60% cash)
```


**Implementation Priority**:
1. Start with CONSERVATIVE (Phase 1-2)
2. Add MODERATE after regime detection proven (Phase 3)
3. Consider AGGRESSIVE only after 6+ months validation (Phase 4+)

**Validation Requirements Before Use**:
- [ ] Jump Model accurately predicts TREND_BEAR (>70% accuracy)
- [ ] Backtest shows 5-10% improvement vs pure cash
- [ ] Duration estimation somewhat reliable (within 30 days)
- [ ] Risk tolerance properly assessed
- [ ] ETF costs understood and acceptable

---

## Technology Stack

### Core Technologies

**Python 3.13.7**:
- Primary development language
- UV package manager for virtual environment
- All code must be Python 3.10+ compatible

**VectorBT Pro 2025.10.15**:
- Primary backtesting framework
- Vectorized operations mandatory
- Native TA-Lib support
- OpenAI API integration for embeddings

**TA-Lib (Technical Analysis Library)**:
- MUST use TA-Lib for all technical indicators
- VectorBT Pro natively supports TA-Lib
- More reliable than custom implementations
- Established industry standard

**Alpaca Algo Trader Plus**:
- Historical data source
- Paper trading execution
- Commission-free trading
- Real-time market data

**OpenAI API**:
- Model: text-embedding-3-small (embeddings)
- Model: gpt-4o-mini (completions)
- Cost optimization: Target <$10/month
- Used for VectorBT Pro semantic search

### Development Infrastructure

**VS Code**:
- Primary development environment
- Integrated terminal for UV commands
- Python extension required

**UV Package Manager**:
- Faster than pip/conda
- Better dependency resolution
- Virtual environment management

**Git Version Control**:
- All code must be version controlled
- Branching strategy: main, dev, feature/*
- No direct commits to main

**Testing Framework**:
- pytest for unit tests
- pytest-cov for coverage analysis
- Target: >80% code coverage

---

## Multi-Layer Integration Architecture

### System Architecture Overview

ATLAS implements a flexible multi-layer architecture where each layer can operate independently or in conjunction with other layers:

**Layer 1 (ATLAS): Regime Detection**
- Market state classification: TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH
- Statistical jump model with academic validation (33 years historical testing)
- Primary output: Regime signal for filtering and position sizing
- Status: Validated (March 2020 crash: 77% detection rate)

**Layer 2 (STRAT): Pattern Recognition**
- Bar-level pattern detection: 3-1-2, 2-1-2, 2-2 reversals
- Multi-timeframe continuity analysis (alignment across 3+ timeframes)
- Precise entry/exit levels via governing range methodology
- Status: Design complete, implementation pending

**Layer 3 (Execution): Capital-Aware Routing**
- Options execution for $3k+ accounts (27x capital efficiency)
- Equity execution for $10k+ accounts (lower leverage, lower risk)
- Dynamic position sizing based on capital constraints
- Status: Integration pending after Layer 2 implementation

**Layer 4 (Credit Spreads): Crash Protection**
- Credit market monitoring (IG-HY spreads, TED spread, VIX term structure)
- Cross-asset validation of CRASH regime detection
- Veto power for high-risk trades during credit stress
- Status: Deferred pending Layer 2 completion

### Layer Independence Principle

**Critical design decision:** Each layer operates independently and must be profitable standalone.

**Three deployment modes:**

1. **Standalone ATLAS:** Regime-based filtering without pattern detection
   - Use case: Trader prefers broad regime signals over pattern precision
   - Capital: $10k+ (equity strategies)
   - Status: Ready for paper trading

2. **Standalone STRAT:** Pattern-based trading without regime context
   - Use case: Trader prioritizes price action patterns
   - Capital: $3k+ (options strategies)
   - Status: Design complete, implementation pending

3. **Integrated ATLAS+STRAT:** Confluence trading combining both systems
   - Use case: Highest signal quality via multi-layer agreement
   - Capital: $20k+ (both systems deployed)
   - Status: Deferred until both layers validated independently

**Rationale for independence:**
- Trader choice: Different traders prefer different approaches
- Capital flexibility: $3k accounts can run STRAT, $10k can run ATLAS
- System robustness: If one layer fails, the other continues
- Development timeline: Layers developed and tested independently

### Integration Framework (Mode 3 Only)

When operating in integrated mode, signals are classified by confluence level:

**High quality signals (full position size):**
```python
if regime == 'TREND_BULL' and strat_pattern == '3-1-2 Bullish' and continuity >= 0.67:
    signal_quality = 'HIGH'  # Both layers agree, timeframes aligned
    position_size = 1.0      # Full size
```

**Medium quality signals (half position size):**
```python
if regime == 'TREND_NEUTRAL' and strat_pattern in valid_patterns:
    signal_quality = 'MEDIUM'  # No regime bias, pattern-dependent
    position_size = 0.5        # Half size
```

**Reject signals (no trade):**
```python
if regime == 'CRASH' and strat_pattern in bullish_patterns:
    signal_quality = 'REJECT'  # ATLAS veto power
    position_size = 0.0        # No trade
```

**ATLAS CRASH veto:** When ATLAS detects CRASH regime, all bullish signals are automatically rejected regardless of STRAT pattern quality. This veto power is based on March 2020 validation where 77% CRASH detection would have prevented significant losses.

### Capital Deployment Strategy

**Validation requirements (all deployment modes):**
- Paper trading: 6 months minimum, 100+ trades
- Execution accuracy: 95%+ (fills match expected prices)
- Performance match: Within 20% of backtest expectations
- Risk controls: 100% compliance with position sizing limits

**Recommended allocation post-validation:**

| Capital Level | Recommended Mode | Rationale |
|--------------|------------------|-----------|
| $3k - $9k | Standalone STRAT (options) | Capital efficient, lower minimum |
| $10k - $19k | Standalone ATLAS or STRAT | Sufficient for either approach |
| $20k+ | Integrated (both layers) | Confluence trading, diversification |

**Current deployment status:**
- ATLAS Layer 1: Paper trading validation in progress
- STRAT Layer 2: Implementation pending after documentation complete
- Integration: Deferred until both layers independently profitable

See `INTEGRATION_ARCHITECTURE.md` for detailed deployment modes and signal quality matrix.
See `CAPITAL_DEPLOYMENT_GUIDE.md` for capital allocation decision tree and risk management by account size.

---