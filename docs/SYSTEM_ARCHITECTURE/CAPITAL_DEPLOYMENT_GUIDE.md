# Capital Deployment Guide

**Version:** 1.0
**Date:** 2025-11-10
**Purpose:** Define capital allocation strategies for different account sizes

---

## Overview

ATLAS is designed to work with multiple capital levels by offering two distinct execution approaches:

1. **Equity strategies** (ATLAS Layer 1 + existing strategies): Minimum $10k recommended
2. **Options strategies** (STRAT Layer 2): Minimum $3k viable

This guide explains why these minimums exist, demonstrates capital efficiency calculations, and provides deployment strategies for different account sizes.

**Important:** All strategies begin with paper trading validation regardless of capital level. Live deployment occurs only after thorough testing validates implementation correctness and profitability.

---

## Capital Efficiency: Options vs Equities

### The 27x Multiplier

**Core concept:** Options provide approximately 27x notional exposure compared to equities for the same capital outlay.

**Example calculation (SPY at $450):**

**Equity position:**
```
Capital available: $3,000
Position size: $3,000 / $450 = 6.67 shares (round to 6 shares)
Notional exposure: 6 shares × $450 = $2,700
Capital efficiency: 1.0x (baseline)
```

**Options position (slightly OTM call):**
```
Capital available: $3,000
Option premium: $150 per contract (example: $455 strike, 7 DTE)
Contracts: $3,000 / ($150 × 100) = 2 contracts
Notional exposure: 2 contracts × 100 shares × $450 = $90,000
Capital efficiency: $90,000 / $3,000 = 30x

Note: Actual multiplier varies by strike/expiration, but 25-30x typical for OTM options
```

**Practical difference:**
- **Equities:** $3k controls $2.7k notional (undercapitalized for meaningful returns)
- **Options:** $3k controls $90k notional (sufficient leverage for 1-3% market moves)

### Why $3k Is Too Small for Equity Strategies

**Position sizing constraint:**

ATLAS equity strategies use ATR-based position sizing with 2% risk per trade:

```python
# Example: SPY daily ATR = $9.00, ATR multiplier = 2.5x
stop_distance = 9.00 × 2.5 = $22.50

# 2% risk rule
risk_per_trade = $3,000 × 0.02 = $60

# Position size calculation
shares = risk_per_trade / stop_distance = $60 / $22.50 = 2.67 shares (round to 2)

# Actual position value
position_value = 2 shares × $450 = $900

# Capital utilization
utilization = $900 / $3,000 = 30%  # Only using 30% of capital per trade
```

**Problem:** With $3k capital and 2% risk, position sizes become too small to justify:
- Commissions erode returns (even commission-free has SEC fees and slippage)
- Can only hold 3-4 positions maximum
- Insufficient diversification across strategies
- ATR stops force tiny position sizes

**Conclusion:** $3k is systematically undercapitalized for equity strategies using proper risk management.

### Why $3k Works for Options Strategies

**Position sizing with defined risk:**

Options have defined maximum loss (premium paid), eliminating the need for ATR-based stops:

```python
# Same $3k capital, 2% risk rule
risk_per_trade = $3,000 × 0.02 = $60

# Options approach: Premium is the risk
max_premium_per_contract = $60 / 100 shares = $0.60 per share

# Typical OTM option: $1.50 premium
# Adjust to meet risk limit:
contracts = $60 / ($1.50 × 100) = 0.4 contracts (round to 0)

# This fails - $60 risk too tight for meaningful options
# Solution: Use 5-10% risk for options (higher percentage acceptable due to defined risk)
risk_per_trade = $3,000 × 0.05 = $150
contracts = $150 / ($1.50 × 100) = 1 contract

# Notional exposure
exposure = 1 contract × 100 shares × $450 = $45,000

# Capital efficiency
efficiency = $45,000 / $3,000 = 15x
```

**Advantage:** Even with conservative 5% risk per trade, options provide 15-30x leverage enabling meaningful returns on small capital.

---

## Validation Requirements Before Live Deployment

### Paper Trading Validation (All Strategies)

**Minimum paper trading period:**
- ATLAS equity strategies: 6 months, 100+ trades
- STRAT options strategies: 6 months, 100+ trades
- Integrated mode: 6 months after both systems validated independently

**Validation criteria:**
1. Execution accuracy: 95%+ (fills match expected prices within slippage tolerance)
2. Implementation correctness: No index errors, calculation bugs, or logic flaws
3. Performance match: Paper results within 20% of backtest expectations
4. Risk controls: 100% compliance with position sizing and portfolio heat limits
5. Profitability: Positive expectancy over validation period

**Why paper trading is mandatory:**
- New implementation may have bugs despite thorough testing
- Real market conditions differ from backtest assumptions
- Slippage and fill rates need validation
- Emotional execution differs from automated (if manual elements exist)

**Current status:**
- ATLAS Layer 1: Validation in progress (regime detection validated, full strategy testing pending)
- STRAT Layer 2: Design phase (paper trading begins after implementation complete)

---

## Capital Allocation Matrix (Post-Validation)

### $3,000 Capital

**Recommended: STRAT Standalone (Options)**

**Allocation:**
- 100% STRAT pattern trading with options
- 5% risk per trade (acceptable for defined-risk positions)
- Maximum 3-4 concurrent positions

**Example portfolio:**
```
Position 1: SPY 3-1-2 bullish → 1 call contract ($150 premium)
Position 2: QQQ 2-2 continuation → 1 call contract ($180 premium)
Position 3: IWM 2-1-2 reversal → 1 put contract ($120 premium)

Total capital at risk: $450 (15% of $3k)
Total notional exposure: ~$120,000 (40x leverage)
```

**Not recommended:**
- ATLAS equity strategies (position sizes too small)
- Integrated mode (insufficient capital for both systems)

---

### $5,000 - $9,000 Capital

**Recommended: STRAT Standalone (Options)**

**Allocation:**
- 100% options via STRAT patterns
- 4-6 concurrent positions
- 5% risk per trade

**Alternative: Testing allocation**
- Can paper trade both ATLAS and STRAT simultaneously
- Determines which system better matches trading style
- No capital constraints during paper phase

---

### $10,000 - $20,000 Capital

**Recommended: ATLAS Standalone (Equities) OR STRAT Standalone (Options)**

**Option A: ATLAS equities ($10k)**
```
Position sizing example:
- SPY position: 10 shares ($4,500 notional, 2% risk = $200)
- QQQ position: 8 shares ($3,200 notional, 2% risk = $200)
- IWM position: 15 shares ($2,250 notional, 2% risk = $200)

Total: 3 positions, $10k deployed, diversified across instruments
```

**Option B: STRAT options ($10k)**
```
Much higher notional exposure:
- 6-8 concurrent options positions
- 3-5% risk per trade
- Notional exposure: $300k-$500k
- Higher return potential, higher risk
```

**Option C: Mixed allocation ($15k+)**
```
$10k: ATLAS equities (4-5 positions)
$5k: STRAT options (3-4 positions)

Benefits:
- Diversification across execution methods
- Uncorrelated return streams
- Test integration readiness
```

---

### $20,000+ Capital

**Recommended: Integrated Mode (ATLAS + STRAT)**

**Allocation strategy:**
```
$12k: ATLAS equity positions (regime-based)
$8k: STRAT options positions (pattern-based)

Integration logic:
- High quality signals (confluence): Full size on both systems
- Medium quality: Half size or single-system deployment
- Reject signals: No trade
```

**Benefits:**
- Sufficient capital to diversify across both approaches
- Confluence trading improves signal quality
- Regime awareness prevents counter-trend pattern trades
- Pattern precision improves regime-based entries

**Example integrated trade:**
```
ATLAS: Detects TREND_BULL regime
STRAT: 3-1-2 bullish pattern on SPY 15min
Continuity: Control (all timeframes aligned)

Signal quality: HIGH

Execution:
- ATLAS: 20 shares SPY ($9,000 equity position)
- STRAT: 2 call contracts ($300 premium, $90k notional exposure)

Total capital: $9,300 deployed
Total exposure: $99,000 notional (10.6x average leverage)
```

---

## Deployment Timeline (Recommended Path)

### Phase 1: Paper Trading ATLAS (Months 1-6)

**Capital:** $10k simulated

**Objectives:**
- Validate regime detection implementation
- Test strategy execution (ORB, mean reversion, pairs trading)
- Verify position sizing and risk management
- Collect performance data vs backtests

**Success criteria:**
- 100+ trades executed
- Performance within 20% of backtest expectations
- No implementation bugs discovered
- Positive expectancy demonstrated

**Status:** In progress (regime detection validated, strategy testing pending)

### Phase 2: Paper Trading STRAT (Months 1-6, parallel or sequential)

**Capital:** $3k simulated

**Objectives:**
- Validate pattern detection implementation
- Test options execution (strikes, expirations, fills)
- Verify timeframe continuity logic
- Measure slippage on options fills

**Success criteria:**
- 100+ options trades executed
- Pattern detection matches manual analysis
- Options fills within acceptable slippage
- Positive expectancy on pattern trades

**Status:** Not started (design phase complete, implementation pending)

### Phase 3: Live ATLAS Deployment (Month 7+)

**Capital:** $10k live (after paper validation passes)

**Objectives:**
- Deploy validated ATLAS strategies with real capital
- Continue monitoring for implementation drift
- Compare live vs paper results
- Build confidence for 6+ months

**Risk management:**
- Start at 50% position size for first month
- Scale to full size after validation
- Maintain stop losses and portfolio heat limits

### Phase 4: Live STRAT Deployment (Month 7+, after paper validation)

**Capital:** $3k live (after paper validation passes)

**Objectives:**
- Deploy validated STRAT strategies with real capital
- Validate options fill quality in live markets
- Monitor pattern effectiveness vs paper results

**Risk management:**
- Start at 50% position size (0.5 contracts where available)
- Limited to 2-3 concurrent positions initially
- Scale to full deployment after 30+ live trades

### Phase 5: Integration (Month 13+, both systems profitable)

**Capital:** $13k+ live (both systems validated independently)

**Objectives:**
- Enable signal quality matrix
- Trade confluence opportunities
- Compare integrated vs standalone performance

**Risk:** Start integrated signals at 50% size, scale after validation

---

## Risk Management by Capital Level

### $3k Capital (STRAT Options)

**Position sizing:**
- 5% risk per trade = $150 maximum loss
- Typical option premium: $1.00 - $2.00 per share
- Result: 1 contract per trade

**Portfolio heat limit:**
- Maximum concurrent positions: 4
- Maximum portfolio risk: 20% ($600)
- Four positions at 5% each = 20% total risk exposure

**Stop loss:**
- Options: No traditional stop (premium is max loss)
- Exit rule: Close at 50% premium loss or pattern invalidation

### $10k Capital (ATLAS Equities)

**Position sizing:**
- 2% risk per trade = $200 maximum loss
- ATR-based stops (typically $15-$25 for SPY)
- Result: 8-15 shares per trade

**Portfolio heat limit:**
- Maximum concurrent positions: 6-8
- Maximum portfolio risk: 12-16% ($1,200-$1,600)
- Each position 2% risk × 6-8 positions

**Stop loss:**
- ATR-based: 2.5× ATR from entry
- Trailing: Move to breakeven at +1R, trail at +2R

### $20k Capital (Integrated)

**Position sizing:**
- ATLAS equities: 2% per trade ($400 max risk)
- STRAT options: 3-5% per trade ($600-$1,000 max risk)

**Portfolio heat limit:**
- ATLAS positions: 6-8 max (12-16% portfolio risk)
- STRAT positions: 4-6 max (12-30% portfolio risk)
- Combined: Monitor total notional exposure

**Risk adjustment:**
- High quality signals: Full size
- Medium quality: Half size (reduce percentage risk)
- Integrated positions: Higher size when both systems agree

---

## Summary

**Key takeaways:**

1. **Paper trading first:** All strategies validated for 6 months minimum before live capital
2. **$3k minimum (post-validation):** STRAT options only, not ATLAS equities
3. **$10k minimum (post-validation):** ATLAS equities viable, or continue STRAT options
4. **$20k minimum (post-validation):** Integrated mode (ATLAS + STRAT confluence)
5. **Capital efficiency:** Options provide 25-30x leverage vs equities
6. **Validation criteria:** Execution accuracy, implementation correctness, profitability

**Current deployment status:**
- ATLAS: Paper trading validation in progress
- STRAT: Implementation pending, paper trading will follow
- Integrated mode: Deferred until both systems independently validated

**Capital allocation is not just about total dollars - it's about matching execution method to account size constraints and maintaining rigorous validation standards before risking real capital.**
