# Quantitative Trading System: Strategy 1 Analysis & Strategy 2 Design Specification

**Document Purpose:** Independent analysis of Strategy 1 results and detailed specifications for Strategy 2 development  
**Intended Audience:** Claude Code (implementation), Development Team (review)  
**Date:** 2025-10-11  
**Status:** APPROVED FOR STRATEGY 2 DEVELOPMENT

---

## TL;DR for Claude Code: Critical Constraints

**Before implementing ANY strategy, you MUST:**

1. ✅ **Verify position sizing formula** - Calculate actual risk vs expected 2%, document results
2. ✅ **Match philosophy to implementation** - Don't mix mean reversion entries with trend-following exits
3. ✅ **Test across market regimes** - Bull, bear, ranging periods separately
4. ✅ **Use realistic transaction costs** - 0.2% fees + 0.1% slippage minimum

**Strategy 2 (ORB) Non-Negotiables:**

- **Exit logic:** Hard stops ONLY (no RSI/signal exits that cut winners)
- **Win rate expectation:** 15-25% (low by design - asymmetric strategy)
- **R:R target:** 2.5:1 to 4:1 minimum (winners must be 2.5x-4x losers)
- **Regime filter:** Consider TFC alignment as entry filter (optional but recommended)
- **Trade count:** Need 100+ trades for statistical significance

**Red flags that mean STOP and debug:**
- Win rate > 60% (indicates you've added mean reversion logic)
- Average trade < 0.5% (below transaction cost viability)
- Losses > wins in dollar terms (inverted asymmetry)
- < 50 trades in 4-year backtest (insufficient statistical power)

---

## Table of Contents

1. [Strategy 1 Post-Mortem: What Went Wrong](#strategy-1-post-mortem)
2. [Critical Learnings for All Future Strategies](#critical-learnings)
3. [Position Sizing Verification Protocol](#position-sizing-verification)
4. [Strategy 2 (ORB) Design Specification](#strategy-2-design-specification)
5. [Implementation Checklist](#implementation-checklist)
6. [Testing and Validation Protocol](#testing-validation-protocol)

---

## Strategy 1 Post-Mortem: What Went Wrong

### Executive Summary of Failure

**Strategy:** MA200 + RSI(2) Mean Reversion  
**Result:** 3.86% return vs 54.41% buy-and-hold (underperformance: -50.55%)  
**Root Cause:** Philosophical mismatch - mixed Connors mean reversion (high win rate, quick exits) with asymmetric trend-following exits (low win rate, let winners run)

### The Numbers

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Total Return | 3.86% | N/A | FAIL |
| Sharpe Ratio | 0.22 | 0.8-1.2 | FAIL |
| Win Rate (Long) | 74.3% | 75% | ✓ PASS |
| Avg Trade (Long) | +0.27% | 0.5-0.66% | FAIL |
| Risk:Reward | 1:0.48 | 1:2+ | INVERTED |
| Trade Count | 45 | 100+ | INSUFFICIENT |
| Max Drawdown | -11.53% | <20% | ✓ PASS |

### What We Got Right vs Wrong

**✓ RIGHT:**
- Win rate: 74.3% (matches Connors research 75%)
- Entry signal logic (RSI < 15 in uptrend)
- VectorBT Pro implementation quality
- Realistic transaction costs (0.3% total)

**✗ WRONG:**
- Exit logic: Mixed RSI signals (mean reversion) with TP/SL (trend-following)
- Average trade: +0.27% vs expected 0.5-0.66%
- Trade count: 45 vs needed 100+ for significance
- Losses 2x winners (-2.86% vs +1.38%) - wrong asymmetry direction

### The Critical Discovery: Exit Type Analysis

```
Exit Type Distribution (45 total trades):
- Signal Exit (RSI > 85):  27 trades (60%) → +0.81% avg
- Time Exit (14 days):     13 trades (29%) → -2.75% avg
- Take-Profit (4x ATR):     2 trades (4%) → +3.00% avg
- Stop-Loss (2x ATR):       3 trades (7%) → +3.48% avg (shorts!)
```

**Key Insight:** Only 4.4% of trades hit the asymmetric take-profit target. RSI overbought signal cut 60% of trades at +0.81% average - well before the 2:1 target could be reached.

**This proves:** You cannot impose asymmetric exits on mean reversion entries. They are mutually exclusive philosophies.

### The Philosophical Conflict Explained

**Connors Mean Reversion Logic:**
```
Philosophy: Price deviates from mean → quickly reverts → capture small moves
Entry: RSI < 15 (extreme oversold)
Exit: RSI > 85 (extreme overbought) - QUICK exit on reversion
Expected: High win rate (75%), low R:R (1:1 to 1:1.5)
Works in: Ranging, choppy, volatile markets
```

**Asymmetric Trend-Following Logic:**
```
Philosophy: Momentum continues → let winners run → large gains offset losses
Entry: Breakout, momentum confirmation
Exit: Hard stops only (2-3x ATR), let winners run to targets
Expected: Low win rate (20-40%), high R:R (2:1 to 5:1)
Works in: Trending markets (bull or bear)
```

**What Strategy 1 Did (HYBRID FAILURE):**
```
Entry: Connors RSI < 15 (mean reversion signal)
Exit 1: RSI > 85 (mean reversion - 60% of trades) → +0.81% avg
Exit 2: 4x ATR take-profit (asymmetric - 4% of trades) → +3.00% avg
Exit 3: 2x ATR stop-loss (asymmetric - 7% of trades)
Exit 4: 14-day timeout (neither - 29% of trades) → -2.75% avg

Result: Entries seek mean reversion, exits seek trends
        Neither philosophy works properly
        Win rate high (74%) but gains tiny (+0.27%)
```

### Why It Failed in 2021-2025 Specifically

**Market Regime Analysis:**
- 2021-2025: Strong bull market (+54% SPY)
- 71.5% bullish bias (284 bullish vs 113 bearish bars via TFC analysis)
- Low volatility, strong trends (except 2022)

**Why Mean Reversion Fails in Bull Markets:**

1. **Rare oversold conditions**
   - RSI < 15 occurs infrequently in uptrends
   - Result: Only 45 trades in 4 years (11/year)

2. **Shallow pullbacks**
   - When RSI < 15 triggers, bounce is small
   - Trend resumes before RSI > 85 reached
   - Result: 29% of trades hit 14-day timeout at -2.75% avg

3. **Fighting the trend**
   - 22% of trades were shorts (RSI > 85 below MA200)
   - Shorts lost -0.68% avg fighting bull market
   - Result: Dragged overall performance down

**The 2022 Exception:**

Strategy likely performed **better** in 2022 when SPY dropped -18%. This is when mean reversion shines - high volatility, frequent oscillations around the mean. But 2022 was only 1 out of 4 years (25% of test period), insufficient to overcome losses in trending years.

### Short Trade Breakdown (Critical Finding)

**Performance by Direction:**

| Direction | Trades | Avg Return | Win Rate | Avg Winner | Avg Loser |
|-----------|--------|------------|----------|------------|-----------|
| **LONG**  | 35 (78%) | +0.27% | 74.3% | +1.20% | -2.42% |
| **SHORT** | 10 (22%) | -0.68% | 50.0% | +2.29% | -3.64% |

**Short Trade Analysis:**

- Occurred primarily in 2022-2023 volatile period
- 30% hit stops as market kept rising (bull market)
- 40% timed out losing (-4.35% avg on time exits)
- Only 50% win rate vs 74% for longs

**Why Shorts Failed:**

Mean reversion short logic:
```python
short_signal = (rsi > 85) & (close < ma200)
# Translation: Overbought in downtrend, expect drop
```

The problem: 2021-2025 was 71.5% bullish. When this signal triggered:
1. Often it was a brief dip in an ongoing bull market
2. "Downtrend" (price < MA200) was temporary
3. Price bounced back up, stopping out shorts
4. Result: Fighting the trend = consistent losses

**Decision:** Shorts MUST be disabled for production deployment in bull market regimes.

### Position Sizing: Unverified Critical Gap

**Current Implementation:**
```python
stop_distance = atr * 2.0  # 2x ATR
position_size = (init_cash * 0.02) / stop_distance
```

**Expected Outcome:**
- Risk per trade: 2.00% of capital
- Average loss: -2.00% (if stops execute perfectly)

**Actual Outcome:**
- Average loss: -2.86%
- Discrepancy: -0.86% (43% higher than expected)

**Possible Causes:**

1. **Fees/slippage** - 0.3% total costs, but doesn't explain full -0.86%
2. **ATR scaling issue** - If ATR is too small, position sizes become too large
3. **Stop slippage** - Stops not executing at exact 2x ATR levels
4. **Multiple positions** - Portfolio heat exceeding single-position risk

**CRITICAL:** This must be verified before Strategy 2. If position sizing is wrong, Strategy 2 will inherit the same bug and you'll waste weeks debugging the wrong strategy.

### Statistical Significance: The 45 Trade Problem

**Your concern was correct:** 45 trades is statistically insufficient.

**Confidence Interval Math:**

With 45 trades and 74% win rate:
- Standard error: √(0.74 × 0.26 / 45) = 6.5%
- 95% confidence interval: 74% ± 13% = **61% to 87%**

This means the "true" win rate could be anywhere from 61% (mediocre) to 87% (excellent) - a massive uncertainty range.

**Minimum Sample Size for ±5% Precision:**
```
n = (1.96² × 0.74 × 0.26) / 0.05² = 296 trades
```

You have 45. You need **6.6x more trades** for statistical confidence at ±5%.

**But Even 296 Trades Isn't Enough:**

296 trades across 2021-2025 only validates **bull market performance**. To properly validate:
- 296 trades in bull markets (2021, 2023-2025)
- 296 trades in bear markets (2022, 2008, 2018)
- 296 trades in ranging markets (2015-2016, 2019)

**Total needed: ~900 trades across 10-15 years minimum**

With 11 trades/year average, this strategy would need **82 years of data** to validate properly.

**Conclusion:** This strategy is fundamentally untestable on available data for the regime it's designed for.

### Comparison to Academic Research

**Why Published Research Shows 0.5-0.66% Per Trade:**

The Connors research likely differs in these ways:

1. **Time period:** 30+ years includes multiple ranging/volatile markets (not just 2021-2025 bull)
2. **Exit logic:** Pure RSI exits (RSI > 85 only), no hybrid TP/SL/time
3. **Universe:** Multiple stocks, not single SPY index
4. **Regime selection:** May filter for suitable market conditions

**Your +0.27% vs Research 0.5-0.66%:**

The 59% performance gap is explained by:
- **Wrong regime:** Bull market vs mixed regimes (-30% impact estimated)
- **Exit conflict:** Hybrid exits vs pure RSI (-20% impact estimated)
- **Single instrument:** SPY vs diversified (-9% impact estimated)

**Conclusion:** Your implementation is probably correct - the research just doesn't apply to 2021-2025 SPY bull market specifically.

### VectorBT Pro API Learnings (Preserve for Strategy 2)

**CRITICAL PATTERNS:**

**1. Properties vs Methods**
```python
# Portfolio-level stats are PROPERTIES (no parentheses)
pf.total_return          # ✓ Correct
pf.sharpe_ratio          # ✓ Correct
pf.max_drawdown          # ✓ Correct

# Trade-level count is METHOD (needs parentheses)
pf.trades.count()        # ✓ Correct - returns integer
pf.trades.count          # ✗ Wrong - returns <bound method>

# Trade-level stats are PROPERTIES
pf.trades.win_rate       # ✓ Correct
pf.trades.profit_factor  # ✓ Correct
```

**2. Time-Based Exits**
```python
# ✗ WRONG (causes KeyError)
pf = vbt.PF.from_signals(
    time_delta_format='days',  # 'days' is not valid enum
    td_stop=14
)

# ✓ CORRECT
pf = vbt.PF.from_signals(
    td_stop=pd.Timedelta(days=14),  # Use Timedelta object
    freq='1D'
)

# ✓ ALTERNATIVE (string format also works)
td_stop="14 days"
```

**3. Getting Trade Statistics**
```python
# Recommended approach
trade_count = pf.trades.count()  # Call once, store result

stats = {
    'total_return': pf.total_return,
    'sharpe_ratio': pf.sharpe_ratio,
    'total_trades': trade_count,
    'avg_trade_return': pf.trades.returns.mean() if trade_count > 0 else 0.0
}

# Alternative (comprehensive stats)
all_stats = pf.trades.stats()  # Returns Series with all metrics
```

### Final Verdict on Strategy 1

**Status:** ARCHIVE - Not viable for 2021-2025 bull market regime

**Reasons:**
1. Longs barely profitable (+0.27%, below 0.3% transaction costs)
2. Shorts losing (-0.68%, fighting bull market)
3. Wrong regime (mean reversion fails in trending markets)
4. Exit conflict (Connors quick exits vs asymmetric targets)
5. Insufficient trade count (45 vs needed 296+ minimum)

**MAY Be Viable For:**
- Bear market / ranging market regimes (2022-style volatility)
- As fallback in Strategy 3 hybrid (when TFC low + ranging detected)
- Different asset class (individual volatile stocks vs SPY index)

**NOT Viable For:**
- Production deployment as-is
- Paper trading in current bull market
- Further optimization (diminishing returns)

**Preservation Value:**
- Code is clean and working
- VectorBT patterns documented
- Can be reactivated in future bear market
- Useful as reference for what NOT to do (philosophical mismatch)

---

## Critical Learnings for All Future Strategies

### Learning 1: Match Philosophy to Implementation

**The Cardinal Rule:** Entry logic and exit logic must serve the **same trading philosophy**.

**Valid Combinations:**

✓ **Mean Reversion:**
- Entry: Oversold/overbought indicators (RSI, Bollinger Bands)
- Exit: Signal reversal (RSI crosses back, price touches opposite band)
- Expectation: High win rate (60-75%), low R:R (1:1 to 1:1.5)

✓ **Trend-Following:**
- Entry: Breakouts, momentum confirmation (MACD, moving average cross)
- Exit: Hard stops only, let winners run to targets
- Expectation: Low win rate (20-40%), high R:R (2:1 to 5:1)

✓ **Hybrid (Complex):**
- Entry: Regime detection FIRST → Choose appropriate strategy
- Exit: Match the selected strategy's exit logic
- Expectation: Varies by regime, requires dynamic allocation

**Invalid Combinations:**

✗ **What Strategy 1 Did:**
- Entry: Mean reversion (RSI < 15)
- Exit: Trend-following (TP/SL targets)
- Result: Neither philosophy works, both conflict

✗ **Other Failures to Avoid:**
- Entry: Breakout (trend) + Exit: RSI overbought (mean reversion)
- Entry: Moving average cross (trend) + Exit: Bollinger Band (mean reversion)
- Entry: Momentum (trend) + Exit: Quick profit target (scalping)

**Test for Philosophical Consistency:**

Ask these questions:
1. "Am I expecting price to continue the move (trend) or reverse (mean reversion)?"
2. "Do my exits align with that expectation?"
3. "Would a professional trader using this philosophy approve of my exit logic?"

If answers conflict → redesign before implementing.

### Learning 2: Regime Matters More Than Strategy

**The Reality:** Every strategy has optimal and suboptimal market regimes.

**Strategy 1's Regime Mismatch:**
- Designed for: Ranging, volatile markets (2022, 2008-2009, 2015-2016)
- Tested on: Strong bull market (2021-2025, +54% SPY)
- Result: 3.86% vs 54% benchmark (massive underperformance)

**Regime Classification Framework:**

| Regime Type | Characteristics | Best Strategies | Example Periods |
|-------------|----------------|-----------------|-----------------|
| **Strong Trend Up** | ADX > 25, price > MA200, consistent higher highs | Breakout, momentum, trend-following | 2021, 2023-2025 |
| **Strong Trend Down** | ADX > 25, price < MA200, consistent lower lows | Trend-following shorts, put options | 2008, 2022 (briefly) |
| **Ranging/Choppy** | ADX < 20, price oscillates around MA200 | Mean reversion, pairs trading | 2015-2016, 2019 |
| **High Volatility** | ATR percentile > 80, frequent gaps | Mean reversion, vol selling (options) | 2020 COVID, 2022 |
| **Low Volatility** | ATR percentile < 20, narrow ranges | Trend-following (when breaks), skip trading | 2017, early 2020 |
| **Transitional** | ADX 20-25, conflicting signals | Reduce position sizes, wait for clarity | Market turning points |

**For Strategy 2 (ORB):**

✓ Designed for: Strong trending markets (up or down)  
✓ 2021-2025 regime: Bull market, perfect fit  
✓ Expected: Should perform well in backtests

**For Strategy 3 (TFC Hybrid):**

Must detect regime dynamically:
```python
if regime == "strong_trend":
    allocation = {"ORB": 0.70, "Mean_Rev": 0.30}
elif regime == "ranging":
    allocation = {"ORB": 0.30, "Mean_Rev": 0.70}
else:  # transitional
    allocation = {"ORB": 0.50, "Mean_Rev": 0.50}
```

### Learning 3: Transaction Costs Are Non-Negotiable

**Strategy 1's Reality Check:**

- Average long trade: +0.27%
- Transaction costs: 0.3% (0.2% fees + 0.1% slippage)
- Net expected: **-0.03% per trade** (losing money on average)

**Minimum Viability Thresholds:**

| Transaction Cost | Min Avg Trade for Breakeven | Recommended Target |
|------------------|------------------------------|-------------------|
| 0.15% (institutional) | 0.15% | 0.30%+ |
| 0.20% (retail, low-cost broker) | 0.20% | 0.40%+ |
| 0.30% (retail, realistic) | 0.30% | 0.60%+ |
| 0.50% (crypto, high slippage) | 0.50% | 1.00%+ |

**For Strategy 2 Design:**

Target: 0.60%+ average trade minimum

Why? If research shows 17% win rate:
- Need large winners to offset frequent losers
- 83% of trades lose money (pay 0.3% costs each)
- 17% of trades must win big enough to:
  1. Pay their own 0.3% costs
  2. Offset all loser costs
  3. Generate net profit

Example math:
- 100 trades @ 17% win rate = 17 winners, 83 losers
- Loser costs: 83 × 0.3% = 24.9% total
- Winner costs: 17 × 0.3% = 5.1% total
- Total costs: 30% of capital
- Winners must generate: 30% + target profit (e.g. 20%) = 50%
- Per winner: 50% / 17 = 2.94% average
- Per trade overall: (17 × 2.94% - 83 × 0.3%) / 100 = 0.25% average

**Critical:** If your backtest shows < 0.5% average trade, stop and debug. You're likely:
1. Cutting winners too early (signal exits on trend strategy)
2. Position sizing too small
3. Stops too tight relative to volatility

### Learning 4: Statistical Significance Requirements

**The 45 Trade Problem:**

With 45 trades:
- 95% confidence interval: ±13% on win rate
- Cannot distinguish between 61% (bad) and 87% (excellent)
- Conclusions are unreliable

**Minimum Sample Sizes:**

| Confidence Level | Margin of Error | Min Trades (50% win rate) | Min Trades (25% win rate) |
|------------------|-----------------|---------------------------|---------------------------|
| ±10% | Low confidence | 96 | 72 |
| ±5% | Standard | 384 | 288 |
| ±3% | High confidence | 1,067 | 800 |
| ±1% | Very high | 9,604 | 7,203 |

**Practical Targets for Strategy 2:**

- Minimum: 100 trades (±10% confidence, barely acceptable)
- Good: 200 trades (±7% confidence)
- Excellent: 400+ trades (±5% confidence)

**How to Achieve Higher Trade Counts:**

1. **Longer time periods** - Test 10+ years instead of 4
2. **Multiple symbols** - Test on S&P 500 stocks, not just SPY
3. **Multiple timeframes** - Daily + weekly + monthly signals
4. **Lower entry threshold** - More aggressive signal criteria (carefully)

**Warning Signs:**

- < 50 trades in 4 years → Strategy too selective, unreliable
- < 100 trades in 10 years → Same issue
- < 20 trades/year → Consider looser entry criteria or different strategy

### Learning 5: Position Sizing Verification is Non-Negotiable

**Why Strategy 1's Unverified Position Sizing is Critical:**

- Expected avg loss: -2.00% (2% risk per trade)
- Actual avg loss: -2.86%
- Discrepancy: -0.86% (43% higher)

**What This Means:**

Either:
1. Position sizing formula is wrong → All strategies will have wrong risk
2. Stops are slipping beyond 2x ATR → Need wider stops or different assets
3. Multiple positions open simultaneously → Portfolio heat exceeding single-position risk
4. Fees/slippage calculation is wrong → Using wrong cost assumptions

**Before Strategy 2, YOU MUST verify:**

```python
# Verification script (see Position Sizing Verification section)
signals = strategy.generate_signals(data)
close = data['Close']
atr = signals['atr']
stop_distance = atr * 2.0

# Calculate theoretical position sizing
position_size = init_cash * 0.02 / stop_distance
position_value = position_size * close
position_pct = position_value / init_cash

# Calculate theoretical risk
theoretical_risk = (stop_distance / close) * position_pct

print(f"Position size: {position_pct.mean():.1%} of portfolio")
print(f"Theoretical risk: {theoretical_risk.mean():.2%}")
print(f"Expected: 2.00%")
print(f"ATR as % of price: {(atr/close).mean():.2%}")

# Compare to actual losses
actual_avg_loss = pf.trades.losing_returns.mean()
print(f"Actual avg loss: {actual_avg_loss:.2%}")
print(f"Discrepancy: {actual_avg_loss - theoretical_risk.mean():.2%}")
```

**If discrepancy > 0.5%, you have a bug. Find it before Strategy 2.**

### Learning 6: Exit Logic Determines Success More Than Entry

**Strategy 1's Exit Analysis:**

| Exit Type | % of Trades | Avg Return | Impact on Overall Performance |
|-----------|-------------|------------|------------------------------|
| Signal (RSI > 85) | 60% | +0.81% | +0.49% contribution |
| Time (14 days) | 29% | -2.75% | -0.80% contribution |
| Take-Profit | 4% | +3.00% | +0.12% contribution |
| Stop-Loss | 7% | +3.48% | +0.24% contribution |
| **TOTAL** | 100% | - | **+0.05% average** |

**Key Insight:** Time exits (-0.80% contribution) destroyed more value than take-profits (+0.12%) and stops (+0.24%) combined added.

**The Exit Hierarchy Principle:**

Good exits follow this priority:
1. **Stop-loss** - Always execute, no exceptions (survival)
2. **Profit target** - Execute when reached (lock in gains)
3. **Signal exit** - Execute when trend/momentum changes (confirmation)
4. **Time exit** - Execute ONLY if position not developing as expected (last resort)

**What Strategy 1 Did Wrong:**

Priority was effectively:
1. Signal exit (60% of trades) - Cut winners early
2. Time exit (29% of trades) - Gave up on non-developing trades
3. Take-profit (4% of trades) - Almost never reached
4. Stop-loss (7% of trades) - Rarely hit

**For Strategy 2 (ORB):**

Exit hierarchy should be:
1. **Stop-loss** (2-3x ATR) - Primary risk management
2. **End-of-day close** (intraday strategy) - Natural time boundary
3. **Profit target** (optional, 2:1 minimum) - Lock in outsized gains
4. **NO signal exits** - Don't use RSI/MACD to cut winners

**Exit Logic Testing:**

Before finalizing any strategy, extract exit type distribution:
```python
trades = pf.trades.records_readable
exit_types = trades['Stop Type']  # Or equivalent field
exit_analysis = trades.groupby('Stop Type')['Return'].agg(['count', 'mean'])
print(exit_analysis)

# Red flags:
# - If signal exits > 50% → You're cutting winners
# - If time exits have negative avg → Your timeframe is wrong
# - If take-profits < 20% → Your targets are unrealistic
```

---

## Position Sizing Verification Protocol

**CRITICAL:** Run this verification before implementing Strategy 2. If position sizing is broken, all strategies will fail.

### Step 1: Theoretical Position Size Calculation

```python
import pandas as pd
import numpy as np
import vectorbtpro as vbt

# Load Strategy 1 for verification
from strategies.baseline_ma_rsi import BaselineStrategy

# Load data
from data.mtf_manager import MarketAlignedMTFManager
manager = MarketAlignedMTFManager(symbol='SPY')
daily_data = manager.get_daily_data('2021-10-11', '2025-10-09')

# Generate signals
strategy = BaselineStrategy()
signals = strategy.generate_signals(daily_data)

# Extract components
close = daily_data['Close']
atr = signals['atr']
stop_distance = atr * 2.0

# Calculate position sizing
init_cash = 10000
risk_per_trade = 0.02
position_size = (init_cash * risk_per_trade) / stop_distance

# Replace inf with 0 (same as strategy code)
position_size = position_size.replace([np.inf, -np.inf], 0).fillna(0)

# Calculate as % of portfolio
position_value = position_size * close
position_pct = position_value / init_cash

# Calculate theoretical risk
stop_pct = stop_distance / close
theoretical_risk_pct = stop_pct * position_pct

# Basic statistics
print("=" * 60)
print("POSITION SIZING VERIFICATION")
print("=" * 60)
print(f"\nATR Statistics:")
print(f"  Mean ATR: ${atr.mean():.2f}")
print(f"  ATR as % of price: {(atr/close * 100).mean():.2f}%")
print(f"  Min ATR: ${atr.min():.2f} | Max ATR: ${atr.max():.2f}")

print(f"\nStop Distance (2x ATR):")
print(f"  Mean: ${stop_distance.mean():.2f}")
print(f"  As % of price: {(stop_distance/close * 100).mean():.2f}%")

print(f"\nPosition Sizing:")
print(f"  Mean position size: {position_size.mean():.0f} shares")
print(f"  Mean position value: ${position_value.mean():.2f}")
print(f"  Mean position as % of portfolio: {(position_pct * 100).mean():.1f}%")
print(f"  Min: {(position_pct * 100).min():.1f}% | Max: {(position_pct * 100).max():.1f}%")

print(f"\nTheoretical Risk Per Trade:")
print(f"  Expected: 2.00%")
print(f"  Actual: {(theoretical_risk_pct * 100).mean():.2f}%")
print(f"  Discrepancy: {((theoretical_risk_pct * 100).mean() - 2.0):.2f}%")

# Sanity checks
print(f"\n{'=' * 60}")
print("SANITY CHECKS:")
print(f"{'=' * 60}")

# Check 1: Position size distribution
large_positions = (position_pct > 0.50).sum()
tiny_positions = (position_pct < 0.05).sum()
zero_positions = (position_size == 0).sum()

print(f"\n1. Position Size Distribution:")
print(f"   - Positions > 50% of capital: {large_positions} ({large_positions/len(position_pct)*100:.1f}%)")
print(f"   - Positions < 5% of capital: {tiny_positions} ({tiny_positions/len(position_pct)*100:.1f}%)")
print(f"   - Zero positions: {zero_positions} ({zero_positions/len(position_pct)*100:.1f}%)")
print(f"   ⚠️  WARNING if > 50%: {large_positions if large_positions > 0 else 'PASS ✓'}")
print(f"   ⚠️  WARNING if < 5% majority: {'FAIL - positions too small' if tiny_positions > len(position_pct)*0.5 else 'PASS ✓'}")

# Check 2: ATR reasonableness
atr_pct = (atr / close * 100)
if atr_pct.mean() < 0.5:
    print(f"\n2. ATR Analysis:")
    print(f"   ⚠️  WARNING: ATR very low ({atr_pct.mean():.2f}% of price)")
    print(f"   This may cause oversized positions.")
elif atr_pct.mean() > 5.0:
    print(f"\n2. ATR Analysis:")
    print(f"   ⚠️  WARNING: ATR very high ({atr_pct.mean():.2f}% of price)")
    print(f"   This may cause undersized positions.")
else:
    print(f"\n2. ATR Analysis: PASS ✓ ({atr_pct.mean():.2f}% of price is reasonable)")

# Check 3: Risk calculation correctness
risk_discrepancy = abs((theoretical_risk_pct * 100).mean() - 2.0)
print(f"\n3. Risk Calculation:")
print(f"   Expected: 2.00%")
print(f"   Actual: {(theoretical_risk_pct * 100).mean():.2f}%")
if risk_discrepancy > 0.2:
    print(f"   ⚠️  WARNING: Discrepancy {risk_discrepancy:.2f}% > 0.2% threshold")
    print(f"   Position sizing formula may be incorrect.")
else:
    print(f"   PASS ✓ - Discrepancy {risk_discrepancy:.2f}% within tolerance")
```

### Step 2: Actual vs Theoretical Comparison

```python
# Run backtest
pf = strategy.backtest(daily_data, init_cash=10000, fees=0.002, slippage=0.001, risk_per_trade=0.02)

# Extract actual trade results
trades = pf.trades.records_readable
losing_trades = trades[trades['Return'] < 0]

if len(losing_trades) > 0:
    actual_avg_loss_pct = losing_trades['Return'].mean() * 100
    
    print(f"\n{'=' * 60}")
    print("ACTUAL VS THEORETICAL COMPARISON:")
    print(f"{'=' * 60}")
    print(f"\nTheoretical avg loss (from formula): {(theoretical_risk_pct * 100).mean():.2f}%")
    print(f"Actual avg loss (from backtest): {actual_avg_loss_pct:.2f}%")
    print(f"Discrepancy: {abs(actual_avg_loss_pct - (theoretical_risk_pct * 100).mean()):.2f}%")
    
    # Analyze discrepancy
    discrepancy = abs(actual_avg_loss_pct - (theoretical_risk_pct * 100).mean())
    
    if discrepancy < 0.3:
        print(f"\n✓ PASS - Discrepancy < 0.3%, position sizing working correctly")
    elif discrepancy < 0.5:
        print(f"\n⚠️  BORDERLINE - Discrepancy {discrepancy:.2f}%")
        print(f"   Could be due to fees (0.2%) + slippage (0.1%)")
    else:
        print(f"\n❌ FAIL - Discrepancy {discrepancy:.2f}% > 0.5%")
        print(f"\nPossible causes:")
        print(f"  1. Position sizing formula incorrect")
        print(f"  2. Stops not executing at 2x ATR (slippage, gaps)")
        print(f"  3. Multiple positions open simultaneously (portfolio heat)")
        print(f"  4. Fees/slippage higher than assumed")
        
        # Additional diagnostics
        print(f"\nDetailed diagnostics:")
        print(f"  - Losing trades: {len(losing_trades)}")
        print(f"  - Min loss: {losing_trades['Return'].min() * 100:.2f}%")
        print(f"  - Max loss: {losing_trades['Return'].max() * 100:.2f}%")
        print(f"  - Std dev: {losing_trades['Return'].std() * 100:.2f}%")
        
        # Check for extreme losses
        extreme_losses = losing_trades[losing_trades['Return'] < -0.05]  # < -5%
        if len(extreme_losses) > 0:
            print(f"\n  ⚠️  Found {len(extreme_losses)} trades with losses > 5%:")
            print(extreme_losses[['Entry Timestamp', 'Exit Timestamp', 'Return', 'PnL']])
else:
    print("\n⚠️  No losing trades found - cannot verify position sizing")
    print("   This is suspicious for a mean reversion strategy.")
```

### Step 3: Visual Verification (Optional)

```python
import matplotlib.pyplot as plt

# Plot position size distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Position size over time
axes[0, 0].plot(position_pct.index, position_pct * 100, label='Position %', alpha=0.7)
axes[0, 0].axhline(y=50, color='r', linestyle='--', label='50% (Too Large)')
axes[0, 0].axhline(y=5, color='orange', linestyle='--', label='5% (Too Small)')
axes[0, 0].set_title('Position Size as % of Portfolio Over Time')
axes[0, 0].set_ylabel('Position %')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: ATR as % of price over time
axes[0, 1].plot(atr.index, (atr/close)*100, label='ATR % of Price', color='green', alpha=0.7)
axes[0, 1].set_title('ATR as % of Price Over Time')
axes[0, 1].set_ylabel('ATR %')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Theoretical risk over time
axes[1, 0].plot(theoretical_risk_pct.index, theoretical_risk_pct * 100, label='Theoretical Risk', color='purple', alpha=0.7)
axes[1, 0].axhline(y=2.0, color='b', linestyle='--', label='Target: 2%')
axes[1, 0].set_title('Theoretical Risk Per Trade Over Time')
axes[1, 0].set_ylabel('Risk %')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Position size distribution histogram
axes[1, 1].hist(position_pct * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].axvline(x=50, color='r', linestyle='--', label='50% (Warning)')
axes[1, 1].axvline(x=(position_pct * 100).mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {(position_pct * 100).mean():.1f}%')
axes[1, 1].set_title('Position Size Distribution')
axes[1, 1].set_xlabel('Position %')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('position_sizing_verification.png', dpi=150, bbox_inches='tight')
print("\n✓ Chart saved to: position_sizing_verification.png")
plt.show()
```

### Step 4: Document Results

Create `POSITION_SIZING_VERIFICATION.md`:

```markdown
# Position Sizing Verification Results

**Date:** [Current Date]
**Strategy:** Strategy 1 Baseline (MA200 + RSI)
**Test Period:** 2021-10-11 to 2025-10-09

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Theoretical Risk | X.XX% | [PASS/FAIL] |
| Actual Avg Loss | X.XX% | [PASS/FAIL] |
| Discrepancy | X.XX% | [< 0.5% = PASS] |
| Mean Position Size | XX.X% | [5-50% = PASS] |
| ATR as % of Price | X.XX% | [0.5-5% = PASS] |

## Verdict

[✓ PASS / ❌ FAIL]

**Reasoning:**
[Explain if position sizing is working correctly or what's wrong]

## Implications for Strategy 2

[If PASS: Position sizing formula verified, safe to use for Strategy 2]
[If FAIL: Must fix position sizing before Strategy 2 implementation]

## Action Items

- [ ] [List any required fixes]
- [ ] [Additional verification steps]
```

### Expected Outcomes

**If PASS (discrepancy < 0.5%):**
- Position sizing formula is correct
- Safe to proceed with Strategy 2
- Document verified formula for reference

**If FAIL (discrepancy > 0.5%):**
- **DO NOT proceed to Strategy 2**
- Debug position sizing formula
- Possible fixes:
  1. Adjust risk_per_trade to account for slippage
  2. Change ATR multiplier (try 1.5x or 2.5x instead of 2.0x)
  3. Add portfolio heat management (limit simultaneous positions)
  4. Increase slippage/fees assumptions

**Common Issues and Fixes:**

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Actual loss >> theoretical | Stops slipping beyond 2x ATR | Widen stops to 2.5x or 3x ATR |
| Position sizes > 50% | ATR too small for formula | Add minimum stop distance (e.g. $1) |
| Position sizes < 5% | ATR too large for formula | Increase risk_per_trade to 3% |
| High variance in losses | Inconsistent execution | Check for gap risk, earnings, low liquidity |

---

## Strategy 2 (ORB) Design Specification

### Overview

**Strategy:** Opening Range Breakout (ORB)  
**Philosophy:** Asymmetric trend-following (low win rate, high R:R)  
**Optimal Regime:** Trending markets (bull or bear)  
**Expected Win Rate:** 15-25% (research shows 17%)  
**Expected R:R:** 2.5:1 to 4:1 (winners 2.5x-4x losers)  
**Expected Sharpe:** 1.5-2.5 (research shows 2.396)

### Research Foundation

**Source:** QuantConnect research (2016 backtest, top 1000 US equities)

**Key Results:**
- Win rate: 17% (83% losses)
- Sharpe ratio: 2.396 (vs 0.836 benchmark)
- Beta: -0.042 (market neutral)
- Winners: 2.5-4x larger than losers

**Critical Insight:** Only 1 in 6 trades wins, but when they win, they win BIG. This is asymmetric design at its core.

### Strategy Logic

**Entry Conditions:**

1. **Universe Selection (Daily at Open):**
   ```python
   # Screen for liquid stocks
   min_price = 5.00
   max_price = 500.00  # Avoid extreme high-priced stocks
   min_atr = 0.50
   min_volume = 200000  # $200k+ daily volume
   
   # Filter for "in play" stocks
   volume_ma_20 = volume.rolling(20).mean()
   relative_volume_5min = volume_5min / volume_ma_20  # Opening 5-min bar
   
   # Select top 20 by relative volume
   top_candidates = relative_volume_5min.nlargest(20)
   ```

2. **Opening Range Definition:**
   ```python
   # First 5 minutes of trading (9:30-9:35 AM ET)
   opening_high = high['09:30:00':'09:35:00'].max()
   opening_low = low['09:30:00':'09:35:00'].min()
   opening_close = close['09:35:00']
   
   # Directional bias
   bullish_bar = opening_close > open['09:30:00']
   bearish_bar = opening_close < open['09:30:00']
   ```

3. **Entry Signals:**
   ```python
   # Long entry (bullish opening bar)
   long_entry = (close > opening_high) & bullish_bar
   
   # Short entry (bearish opening bar) - OPTIONAL in bull market
   short_entry = (close < opening_low) & bearish_bar
   ```

4. **Volume Confirmation (Optional but Recommended):**
   ```python
   # Require volume surge on breakout
   breakout_volume = volume.rolling(5).mean()
   volume_surge = breakout_volume > volume_ma_20 * 1.5
   
   confirmed_long_entry = long_entry & volume_surge
   ```

**Exit Conditions:**

1. **Primary Exit: End-of-Day Close**
   ```python
   # Exit all positions at 3:55 PM ET (5 min before close)
   exit_time = pd.Timestamp('15:55:00').time()
   eod_exit = True  # Exit regardless of P&L
   ```

2. **Stop-Loss: 2-3x ATR**
   ```python
   atr_14 = vbt.talib("ATR").run(high, low, close, timeperiod=14).real
   stop_distance = atr_14 * 2.5  # Slightly wider than Strategy 1
   
   long_stop = entry_price - stop_distance
   short_stop = entry_price + stop_distance
   ```

3. **Take-Profit: 2:1 Minimum (Optional)**
   ```python
   # Optional - research shows many winners run further
   profit_target = stop_distance * 2.0
   
   long_tp = entry_price + profit_target
   short_tp = entry_price - profit_target
   ```

4. **NO Signal Exits:**
   ```python
   # DO NOT exit based on:
   # - RSI overbought/oversold
   # - MACD crossover
   # - Moving average touch
   # 
   # These will cut winners early (Strategy 1 lesson)
   ```

**Position Sizing:**

```python
# Fixed fractional (same as Strategy 1)
risk_per_trade = 0.02  # 2% of capital
position_size = (capital * risk_per_trade) / stop_distance

# Portfolio heat management
max_portfolio_heat = 0.08  # 8% max total risk
max_positions = 20  # Diversification (research used 20)

# Equal-weighted across positions
position_size_adjusted = position_size / num_open_positions
if total_portfolio_heat + new_trade_risk > max_portfolio_heat:
    skip_trade()  # Don't exceed heat limit
```

**Risk Management:**

```python
# 1. Per-trade risk: 2%
# 2. Portfolio heat: Max 8% (4 positions at 2% each)
# 3. Max positions: 20 (from research)
# 4. Stop-loss: ALWAYS execute, no exceptions
# 5. Shorts: Disable in bull market (learned from Strategy 1)
```

### Implementation Architecture

**File Structure:**
```
strategies/
├── __init__.py
├── orb_intraday.py          # Main strategy class
├── orb_daily.py              # Daily adaptation (if needed)
└── backtest_orb.py           # Backtesting script
```

**Class Structure:**

```python
class OpeningRangeBreakout:
    """
    Opening Range Breakout strategy for intraday trading.
    
    Philosophy: Asymmetric trend-following
    Entry: Breakout of opening 5-minute range with volume confirmation
    Exit: End-of-day close (primary), stop-loss (risk management)
    
    Expected: 15-25% win rate, 2.5:1+ R:R, 1.5+ Sharpe
    """
    
    def __init__(
        self,
        opening_minutes: int = 5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        risk_per_trade: float = 0.02,
        max_portfolio_heat: float = 0.08,
        max_positions: int = 20,
        volume_surge_multiplier: float = 1.5,
        enable_shorts: bool = False  # Disable by default (bull market)
    ):
        """Initialize ORB strategy with asymmetric parameters."""
        self.opening_minutes = opening_minutes
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.max_positions = max_positions
        self.volume_surge_multiplier = volume_surge_multiplier
        self.enable_shorts = enable_shorts
    
    def screen_universe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Screen for liquid, in-play stocks.
        
        Filters:
        - Price: $5-500
        - ATR: > $0.50
        - Volume: > $200k daily
        - Relative volume: Top 20 on opening 5-min bar
        """
        pass
    
    def calculate_opening_range(self, intraday_data: pd.DataFrame) -> dict:
        """
        Calculate opening range from first N minutes.
        
        Returns:
        - opening_high: High of opening range
        - opening_low: Low of opening range
        - opening_close: Close of opening range
        - directional_bias: 'bullish' or 'bearish'
        """
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> dict:
        """
        Generate entry/exit signals.
        
        Returns dict with:
        - long_entries: Boolean series
        - short_entries: Boolean series (if enabled)
        - stop_distances: ATR-based stops
        - volume_confirmed: Boolean series
        """
        pass
    
    def backtest(
        self,
        data: pd.DataFrame,
        init_cash: float = 10000,
        fees: float = 0.002,
        slippage: float = 0.001
    ) -> vbt.Portfolio:
        """
        Run backtest with proper ORB exit logic.
        
        CRITICAL: No signal exits (RSI, MACD, etc.)
        Primary exit: End-of-day close
        Risk exit: Stop-loss only
        """
        pass
    
    def get_performance_stats(self, pf: vbt.Portfolio) -> dict:
        """
        Extract performance metrics.
        
        Focus on:
        - Win rate (expect 15-25%)
        - Avg winner vs avg loser (expect 2.5:1+)
        - Sharpe ratio (expect 1.5+)
        - Trade count (need 100+ for significance)
        """
        pass
```

### Critical Design Decisions

**Decision 1: Intraday vs Daily Execution**

**Option A: Pure Intraday (Matches Research)**
- Uses 5-minute bars during market hours
- Entry on breakout of opening range
- Exit at 3:55 PM ET (5 min before close)
- Requires intraday broker API execution

**Option B: Daily Adaptation (Simpler)**
- Uses daily bars only
- Opening range = First day of week high/low
- Breakout = Next day close > opening high
- Exit at end of day or next day open
- Different from research, needs separate validation

**Recommendation:** Start with **Option A (Pure Intraday)** since Alpaca supports it. This matches research methodology and expected performance.

**Decision 2: Single Symbol vs Universe Scanning**

**Option A: Single Symbol (SPY)**
- Simpler implementation
- Faster backtesting
- Easier debugging
- May not match research (tested on 1000 stocks)

**Option B: Universe Scanning (S&P 500)**
- Matches research methodology
- Better diversification
- More complex implementation
- Requires multi-symbol backtesting

**Recommendation:** Start with **Option A (SPY only)** for initial validation. If Sharpe > 1.0 and win rate 15-25%, expand to universe scanning.

**Decision 3: Volume Confirmation**

Research mentions "abnormally high opening volume" but doesn't specify exact threshold.

**Options:**
- No volume filter (simpler)
- 1.5x average volume (moderate)
- 2.0x average volume (conservative)
- 2.5x average volume (very selective)

**Recommendation:** Start with **1.5x** as optional parameter. Test sensitivity:
- If win rate too low (< 15%), reduce to 1.2x or remove
- If trade count too low (< 100 in 4 years), reduce threshold
- If win rate acceptable but Sharpe low, increase to 2.0x

**Decision 4: Stop-Loss Width**

Research doesn't specify stop distance, but asymmetric document suggests 2-3x ATR.

**Recommendation:** Start with **2.5x ATR** (slightly wider than Strategy 1's 2.0x). Rationale:
- Intraday volatility higher than daily
- Want to avoid premature stops on noise
- Winners need room to run (asymmetric profile)
- If stops hit > 30% of time, widen to 3x ATR

**Decision 5: Take-Profit Targets**

**Option A: No Take-Profit (Let Winners Run)**
- Matches asymmetric philosophy
- Allows outsized gains
- Exit only at EOD or stop-loss

**Option B: 2:1 Take-Profit (Lock In Some Gains)**
- Reduces variance
- Ensures some trades hit targets
- May limit upside on big moves

**Recommendation:** Start with **Option A (No TP)**. The research shows winners 2.5-4x losers, suggesting they let winners run fully. Add take-profit only if:
- Drawdowns too large (> 25%)
- Want to lock in gains on 50% of position at 2:1
- Testing shows improved Sharpe with partial profits

### Parameter Testing Matrix

After initial implementation, test these parameter combinations:

| Parameter | Values to Test | Expected Impact |
|-----------|----------------|-----------------|
| Opening Range Minutes | 5, 10, 15, 30 | Longer = fewer false breakouts, fewer trades |
| ATR Multiplier | 2.0, 2.5, 3.0 | Higher = lower stop-hit rate, lower win rate |
| Volume Surge | 1.2x, 1.5x, 2.0x, None | Higher = fewer trades, higher quality |
| Risk Per Trade | 1%, 2%, 3% | Higher = larger positions, higher variance |
| Enable Shorts | True, False | True may hurt in bull market |

**Walk-Forward Optimization:**
- In-sample: 2 years
- Out-of-sample: 6 months
- Step forward: 3 months
- Repeat through entire dataset

**Goal:** Find parameter combination where out-of-sample Sharpe matches in-sample within 20%.

### Expected Performance Benchmarks

Based on research and regime analysis:

| Metric | Conservative | Base Case | Optimistic | Red Flag |
|--------|--------------|-----------|------------|----------|
| Win Rate | 15% | 17-20% | 25% | > 30% * |
| Avg Winner | +2.5% | +3.0% | +4.0% | < 2.0% |
| Avg Loser | -1.0% | -1.2% | -1.5% | > -2.0% |
| R:R Ratio | 2:1 | 2.5:1 | 3:1 | < 1.5:1 |
| Sharpe Ratio | 1.0 | 1.5-2.0 | 2.5+ | < 0.8 |
| Max Drawdown | 25% | 20% | 15% | > 30% |
| Trade Count (4yr) | 80 | 100-150 | 200+ | < 50 |
| Avg Trade | 0.5% | 0.8% | 1.2% | < 0.4% |

*Red Flag: If win rate > 30%, you've likely added mean reversion logic by mistake (signal exits cutting winners). Check exit type distribution.

### Implementation Phases

**Phase 1: Core Implementation (Week 1)**
- [ ] Implement basic ORB logic (opening range, breakout detection)
- [ ] Add ATR-based stops
- [ ] Add EOD exit logic
- [ ] Test on SPY only, 4 years data
- [ ] Verify win rate 15-25% and R:R > 2:1

**Phase 2: Position Sizing & Risk Management (Week 1)**
- [ ] Implement fixed fractional position sizing
- [ ] Add portfolio heat management
- [ ] Test max positions limit
- [ ] Verify position sizing (same protocol as Strategy 1)
- [ ] Ensure stops execute correctly

**Phase 3: Validation & Testing (Week 2)**
- [ ] Walk-forward analysis (2yr in-sample, 6mo out-of-sample)
- [ ] Parameter sensitivity testing
- [ ] Exit type distribution analysis
- [ ] Compare to research expectations
- [ ] Document any deviations from expected performance

**Phase 4: Enhancement (Week 2-3)**
- [ ] Add volume confirmation filter
- [ ] Test take-profit targets (optional)
- [ ] Add regime filter (TFC-based, optional)
- [ ] Expand to multi-symbol universe (S&P 500)
- [ ] Re-validate with universe scanning

**Phase 5: Paper Trading Preparation (Week 3-4)**
- [ ] Create live execution module
- [ ] Test broker API integration (Alpaca)
- [ ] Implement real-time data feeds
- [ ] Add monitoring and alerts
- [ ] Create dashboard for tracking

### Integration with TFC (Future Strategy 3)

Once Strategy 2 is validated, integrate with TFC for Strategy 3:

```python
# Dynamic allocation based on regime
tfc_score = calculate_tfc_score(data)  # From TFC analysis
adx = calculate_adx(data)

if (tfc_score > 0.66) and (adx > 25):
    # Strong trend confirmed by TFC
    allocation = {"ORB": 0.80, "Mean_Rev": 0.20}
elif (tfc_score < 0.33) and (adx < 20):
    # Ranging market confirmed by TFC
    allocation = {"ORB": 0.20, "Mean_Rev": 0.80}
else:
    # Mixed signals
    allocation = {"ORB": 0.50, "Mean_Rev": 0.50}
```

### Common Pitfalls to Avoid

**Pitfall 1: Adding Signal Exits**

❌ **WRONG:**
```python
# Don't add these exits to ORB:
exit_rsi_overbought = rsi > 70
exit_macd_cross = macd_histogram < 0
exit_ma_touch = close < ma200

long_exits = eod_exit | exit_rsi_overbought | exit_macd_cross | exit_ma_touch
```

✓ **CORRECT:**
```python
# Only these exits:
long_exits = eod_exit  # Primary exit
# Stop-loss handled separately by VectorBT (sl_stop parameter)
```

**Why:** Signal exits cut winners early (Strategy 1 lesson). ORB needs winners to run to full EOD or stop-loss for asymmetric profile to work.

**Pitfall 2: Wrong Win Rate Expectations**

❌ **WRONG:** "Win rate is only 17%, strategy must be broken"

✓ **CORRECT:** "Win rate is 17%, matches research. Check if winners are 2.5x+ losers."

**Why:** Asymmetric strategies by design have low win rates. The magic is in the R:R ratio, not the win rate.

**Pitfall 3: Optimizing for Higher Win Rate**

❌ **WRONG:** Add filters to increase win rate to 50%+

✓ **CORRECT:** Accept 15-25% win rate, optimize for higher R:R ratio

**Why:** Increasing win rate usually decreases R:R ratio (cutting winners). Net result is worse performance.

**Pitfall 4: Testing Only on Bull Market**

❌ **WRONG:** Backtest only 2021-2025, conclude strategy works

✓ **CORRECT:** Test on 2008-2009 (financial crisis), 2020 (COVID), 2022 (bear market)

**Why:** ORB should work in any trending market (up or down). If it only works in bull markets, it's not truly trend-following.

**Pitfall 5: Ignoring Transaction Costs**

❌ **WRONG:** Use 0.05% fees + 0.05% slippage (unrealistic for intraday)

✓ **CORRECT:** Use 0.2% fees + 0.15% slippage minimum (intraday execution is worse)

**Why:** Intraday strategies face:
- Wider bid-ask spreads
- Market impact on entry/exit
- Partial fills on limit orders
- Slippage on stop-losses

**Realistic costs for intraday:** 0.3-0.4% total (higher than daily strategies)

### Success Criteria for Strategy 2

Before proceeding to Strategy 3, Strategy 2 must meet:

**Minimum Viable Performance:**
- [ ] Win rate: 15-30% (within expected range)
- [ ] R:R ratio: > 2:1 (winners exceed losers by 2x)
- [ ] Sharpe ratio: > 1.0 (good risk-adjusted returns)
- [ ] Avg trade: > 0.5% (above transaction costs)
- [ ] Trade count: > 100 (statistical significance)
- [ ] Max drawdown: < 25% (survivable)

**Validation Requirements:**
- [ ] Walk-forward OOS Sharpe within 20% of IS
- [ ] Performance consistent across multiple market regimes
- [ ] Exit type distribution: < 20% stop-losses (winners running)
- [ ] Position sizing verified (actual risk = 2% ± 0.5%)

**Documentation Requirements:**
- [ ] Complete performance stats documented
- [ ] Exit type analysis completed
- [ ] Parameter sensitivity tested
- [ ] Regime performance breakdown
- [ ] Comparison to research expectations

**If ANY minimum criteria fails:** Debug before proceeding. Common issues:
- Win rate > 30%: Check for signal exits cutting winners
- R:R < 2:1: Stops too tight or targets too close
- Sharpe < 1.0: Transaction costs too high or drawdowns too large
- Avg trade < 0.5%: Position sizing too small or exits too early

---

## Implementation Checklist

### Pre-Implementation (Before Writing Code)

**Strategy 1 Completion:**
- [ ] Position sizing verification completed
- [ ] Results documented in POSITION_SIZING_VERIFICATION.md
- [ ] All bugs identified and understood
- [ ] Lessons learned documented

**Strategy 2 Design Review:**
- [ ] Read this document in full
- [ ] Understand asymmetric philosophy (low win rate, high R:R)
- [ ] Confirm Alpaca supports intraday execution
- [ ] Review research paper/source for ORB strategy
- [ ] Clarify any ambiguous design decisions

**Infrastructure Check:**
- [ ] VectorBT Pro installed and working
- [ ] TA-Lib installed and working
- [ ] Alpaca API credentials configured
- [ ] Data pipeline tested (5-minute bars)
- [ ] Git branch created: feature/orb-strategy

### Implementation Phase 1: Core Logic

**Signal Generation:**
- [ ] Calculate opening range (first 5 min high/low)
- [ ] Detect breakouts (close > opening_high)
- [ ] Add directional bias check (opening bar close > open)
- [ ] Calculate ATR for stops
- [ ] Test signal generation on small dataset

**Entry Logic:**
- [ ] Implement long entry conditions
- [ ] Implement short entry conditions (optional, disabled by default)
- [ ] Add volume confirmation filter (optional)
- [ ] Test entry signals produce expected count (~25-50 per year)

**Exit Logic:**
- [ ] Implement EOD exit (3:55 PM ET)
- [ ] Implement ATR-based stops (2.5x multiplier)
- [ ] **CRITICAL:** Verify NO signal exits (RSI, MACD, MA)
- [ ] Test exit conditions

**Position Sizing:**
- [ ] Copy verified formula from Strategy 1 (if verification passed)
- [ ] Implement portfolio heat management
- [ ] Implement max positions limit
- [ ] Add position size clipping (handle inf, negative, nan)

### Implementation Phase 2: Backtesting Integration

**VectorBT Portfolio Setup:**
- [ ] Configure `vbt.PF.from_signals()` with correct parameters
- [ ] Set fees and slippage (0.2% + 0.15% = 0.35% total)
- [ ] Add stop-loss (`sl_stop` parameter)
- [ ] Add EOD exit (can't use `td_stop`, need custom exit signal)
- [ ] Set frequency to '5T' or '5min' (5-minute bars)

**Data Pipeline:**
- [ ] Fetch 4 years of 5-minute data for SPY
- [ ] Verify RTH-only filtering (9:30 AM - 4:00 PM ET)
- [ ] Check for data quality issues (gaps, missing bars)
- [ ] Resample if needed (daily bars for initial test)

**Backtest Execution:**
- [ ] Run initial backtest on SPY 2021-2025
- [ ] Extract performance metrics
- [ ] Verify backtest completed without errors
- [ ] Check trade count (expect 100-200 for SPY over 4 years)

### Implementation Phase 3: Analysis & Validation

**Performance Analysis:**
- [ ] Calculate win rate (expect 15-25%)
- [ ] Calculate R:R ratio (expect 2.5:1+)
- [ ] Calculate Sharpe ratio (expect 1.5+)
- [ ] Calculate max drawdown (expect < 25%)
- [ ] Calculate avg trade return (expect 0.5%+)

**Exit Type Analysis:**
- [ ] Extract trade records
- [ ] Categorize by exit type (EOD vs stop-loss)
- [ ] Calculate avg return per exit type
- [ ] Verify < 20% of trades hitting stops (winners running)
- [ ] **RED FLAG:** If > 50% hit stops, strategy isn't working

**Position Sizing Verification:**
- [ ] Run position sizing verification script (same as Strategy 1)
- [ ] Verify actual avg loss = 2% ± 0.5%
- [ ] Check position size distribution (5-50% of capital)
- [ ] Verify no extreme positions (> 50% of capital)

**Statistical Significance:**
- [ ] Count total trades (need 100+ minimum)
- [ ] Calculate confidence interval on win rate
- [ ] If < 100 trades, consider:
  - Longer backtest period (6-10 years)
  - Lower entry threshold (less selective)
  - Multiple symbols (universe scanning)

### Implementation Phase 4: Parameter Testing

**Sensitivity Analysis:**
- [ ] Test opening range: 5, 10, 15, 30 minutes
- [ ] Test ATR multiplier: 2.0, 2.5, 3.0
- [ ] Test volume surge: None, 1.2x, 1.5x, 2.0x
- [ ] Test risk per trade: 1%, 2%, 3%
- [ ] Document which parameters are most sensitive

**Walk-Forward Analysis:**
- [ ] Split data: 2 years IS, 6 months OOS
- [ ] Optimize on IS period
- [ ] Test on OOS period
- [ ] Calculate IS vs OOS Sharpe difference
- [ ] **PASS:** If OOS Sharpe within 20% of IS
- [ ] **FAIL:** If OOS Sharpe < 50% of IS (overfitting)

**Regime Testing:**
- [ ] Test on 2021 (bull)
- [ ] Test on 2022 (bear/volatile)
- [ ] Test on 2023-2025 (bull)
- [ ] Compare performance across regimes
- [ ] Verify strategy works in multiple regimes

### Implementation Phase 5: Documentation

**Performance Report:**
- [ ] Create STRATEGY_2_ORB_RESULTS.md
- [ ] Document all metrics vs expectations
- [ ] Include exit type analysis
- [ ] Include regime breakdown
- [ ] Include parameter sensitivity results

**Code Documentation:**
- [ ] Add docstrings to all methods
- [ ] Comment critical sections (entry/exit logic)
- [ ] Document any deviations from research
- [ ] Add type hints to function signatures

**Handoff Documentation:**
- [ ] Update HANDOFF.md with Strategy 2 status
- [ ] Document any bugs encountered
- [ ] Document VectorBT patterns learned
- [ ] List action items for Strategy 3

### Pre-Production Checklist

**If Strategy 2 Passes All Tests:**
- [ ] Position sizing verified
- [ ] Win rate 15-30%
- [ ] R:R > 2:1
- [ ] Sharpe > 1.0
- [ ] 100+ trades
- [ ] Walk-forward OOS within 20% of IS
- [ ] Works in multiple regimes

**Ready for Paper Trading:**
- [ ] Create paper trading script
- [ ] Integrate with Alpaca API
- [ ] Add real-time monitoring
- [ ] Set up alerts (drawdown, P&L, errors)
- [ ] Define stop criteria (when to halt paper trading)

**If Strategy 2 Fails:**
- [ ] Document why it failed
- [ ] Compare to research expectations
- [ ] Identify root causes
- [ ] Decide: Debug vs abandon vs modify
- [ ] Update STRATEGY_2_ORB_RESULTS.md with failure analysis

---

## Testing and Validation Protocol

### Level 1: Unit Tests (Components)

**Purpose:** Verify individual components work correctly in isolation.

**Signal Generation Tests:**
```python
def test_opening_range_calculation():
    """Test opening range is calculated correctly."""
    # Create sample 5-minute data
    data = create_sample_intraday_data()
    
    strategy = OpeningRangeBreakout(opening_minutes=5)
    opening_range = strategy.calculate_opening_range(data)
    
    # Verify opening_high is max of first 5 minutes
    assert opening_range['opening_high'] == data.loc['09:30':'09:35', 'High'].max()
    
    # Verify opening_low is min of first 5 minutes
    assert opening_range['opening_low'] == data.loc['09:30':'09:35', 'Low'].min()
    
    # Verify directional bias
    opening_close = data.loc['09:35', 'Close']
    opening_open = data.loc['09:30', 'Open']
    expected_bias = 'bullish' if opening_close > opening_open else 'bearish'
    assert opening_range['directional_bias'] == expected_bias

def test_breakout_detection():
    """Test breakout signals generated correctly."""
    data = create_sample_intraday_data()
    strategy = OpeningRangeBreakout()
    
    signals = strategy.generate_signals(data)
    
    # Long entry should trigger when close > opening_high
    long_entries = signals['long_entries']
    
    # Check at least one entry triggered
    assert long_entries.sum() > 0
    
    # Verify entries only after opening range period
    assert long_entries.loc[:'09:35'].sum() == 0  # No entries during opening range

def test_atr_calculation():
    """Test ATR calculated correctly for stops."""
    data = create_sample_daily_data()
    strategy = OpeningRangeBreakout(atr_period=14, atr_stop_multiplier=2.5)
    
    signals = strategy.generate_signals(data)
    atr = signals['atr']
    stop_distance = signals['stop_distance']
    
    # Verify stop_distance = atr * multiplier
    assert np.allclose(stop_distance, atr * 2.5)
    
    # Verify ATR is positive
    assert (atr > 0).all()
```

**Position Sizing Tests:**
```python
def test_position_sizing_formula():
    """Test position sizing calculates correct risk."""
    data = create_sample_data()
    strategy = OpeningRangeBreakout(risk_per_trade=0.02, atr_stop_multiplier=2.5)
    
    signals = strategy.generate_signals(data)
    
    close = data['Close']
    atr = signals['atr']
    stop_distance = atr * 2.5
    
    # Calculate position size
    init_cash = 10000
    position_size = (init_cash * 0.02) / stop_distance
    
    # Calculate theoretical risk
    position_value = position_size * close
    theoretical_risk = (stop_distance / close) * (position_value / init_cash)
    
    # Verify risk ~= 2% (allowing for rounding)
    assert np.allclose(theoretical_risk.mean(), 0.02, atol=0.002)

def test_portfolio_heat_management():
    """Test portfolio heat limits enforced."""
    # Create scenario with multiple open positions
    strategy = OpeningRangeBreakout(max_portfolio_heat=0.08)
    
    open_positions = [
        {'risk': 0.02},  # 2%
        {'risk': 0.02},  # 2%
        {'risk': 0.02},  # 2%
    ]
    
    current_heat = sum(p['risk'] for p in open_positions)  # 6%
    new_trade_risk = 0.02  # 2%
    
    # Should allow (6% + 2% = 8% = max)
    assert current_heat + new_trade_risk <= strategy.max_portfolio_heat
    
    # Add another position
    open_positions.append({'risk': 0.02})
    current_heat = sum(p['risk'] for p in open_positions)  # 8%
    
    # Should block next trade (8% + 2% = 10% > 8% max)
    assert current_heat + new_trade_risk > strategy.max_portfolio_heat
```

**Exit Logic Tests:**
```python
def test_eod_exit_timing():
    """Test EOD exits trigger at correct time."""
    data = create_sample_intraday_data()
    strategy = OpeningRangeBreakout()
    
    signals = strategy.generate_signals(data)
    
    # Check EOD exit signal present at 3:55 PM
    assert signals['eod_exit'].loc['15:55'].any()
    
    # Check no EOD exits before 3:55 PM
    assert not signals['eod_exit'].loc[:'15:50'].any()

def test_stop_loss_execution():
    """Test stops placed correctly."""
    data = create_sample_data()
    strategy = OpeningRangeBreakout(atr_stop_multiplier=2.5)
    
    signals = strategy.generate_signals(data)
    
    # Get entry price and stop distance
    entry_price = 100.0
    stop_distance = signals['stop_distance'].iloc[0]
    
    # Long stop should be below entry
    long_stop = entry_price - stop_distance
    assert long_stop < entry_price
    
    # Verify stop distance is 2.5x ATR
    atr = signals['atr'].iloc[0]
    assert np.isclose(stop_distance, atr * 2.5)
```

### Level 2: Integration Tests (Full Backtest)

**Purpose:** Verify strategy components work together correctly.

**Backtest Execution Test:**
```python
def test_full_backtest_execution():
    """Test complete backtest runs without errors."""
    # Load real data
    from data.mtf_manager import MarketAlignedMTFManager
    manager = MarketAlignedMTFManager(symbol='SPY')
    data = manager.get_intraday_data('2024-01-01', '2024-12-31', timeframe='5min')
    
    # Run strategy
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data, init_cash=10000, fees=0.002, slippage=0.0015)
    
    # Verify portfolio created
    assert pf is not None
    
    # Verify trades occurred
    trade_count = pf.trades.count()
    assert trade_count > 0, "No trades generated"
    
    # Verify basic metrics calculable
    assert pf.total_return is not None
    assert pf.sharpe_ratio is not None
    assert pf.max_drawdown is not None

def test_backtest_metrics_reasonable():
    """Test backtest produces reasonable metrics."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    # Win rate should be in expected range (10-30%)
    win_rate = pf.trades.win_rate
    assert 0.10 <= win_rate <= 0.30, f"Win rate {win_rate:.2%} outside expected range"
    
    # Should have reasonable number of trades (50+ over 4 years)
    trade_count = pf.trades.count()
    assert trade_count >= 50, f"Only {trade_count} trades over 4 years"
    
    # Sharpe should be positive
    assert pf.sharpe_ratio > 0, f"Negative Sharpe: {pf.sharpe_ratio}"
    
    # Max drawdown should be survivable (< 50%)
    assert pf.max_drawdown < 0.50, f"Drawdown {pf.max_drawdown:.2%} too large"

def test_position_sizing_integration():
    """Test position sizing works in full backtest."""
    data = load_test_data('SPY', '2024-01-01', '2024-12-31')
    strategy = OpeningRangeBreakout(risk_per_trade=0.02)
    pf = strategy.backtest(data, init_cash=10000)
    
    # Extract losing trades
    trades = pf.trades.records_readable
    losing_trades = trades[trades['Return'] < 0]
    
    if len(losing_trades) > 0:
        avg_loss_pct = losing_trades['Return'].mean() * 100
        
        # Should be close to 2% (allowing for fees/slippage)
        assert -3.0 <= avg_loss_pct <= -1.5, \
            f"Avg loss {avg_loss_pct:.2f}% outside expected range (-3% to -1.5%)"
```

### Level 3: Validation Tests (Performance)

**Purpose:** Verify strategy meets performance expectations.

**Win Rate Validation:**
```python
def test_win_rate_in_expected_range():
    """Test win rate matches asymmetric strategy expectations."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    win_rate = pf.trades.win_rate
    
    # Research shows 17% win rate, allow 10-30% range
    assert 0.10 <= win_rate <= 0.30, \
        f"Win rate {win_rate:.2%} outside expected range (10-30%)"
    
    # If win rate > 30%, likely have mean reversion logic
    if win_rate > 0.30:
        print("⚠️  WARNING: Win rate suspiciously high for asymmetric strategy")
        print("   Check for signal exits cutting winners early")

def test_risk_reward_ratio():
    """Test R:R ratio matches asymmetric expectations."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    trades = pf.trades.records_readable
    winning_trades = trades[trades['Return'] > 0]
    losing_trades = trades[trades['Return'] < 0]
    
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        avg_winner = winning_trades['Return'].mean()
        avg_loser = abs(losing_trades['Return'].mean())
        rr_ratio = avg_winner / avg_loser
        
        # Should be at least 2:1, ideally 2.5:1+
        assert rr_ratio >= 2.0, \
            f"R:R ratio {rr_ratio:.2f}:1 below minimum 2:1"
        
        print(f"✓ R:R ratio: {rr_ratio:.2f}:1")
        
        if rr_ratio < 2.5:
            print("⚠️  R:R ratio below ideal 2.5:1, consider widening stops")

def test_sharpe_ratio_threshold():
    """Test Sharpe ratio meets minimum threshold."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    sharpe = pf.sharpe_ratio
    
    # Minimum acceptable: 1.0
    # Research target: 1.5-2.5
    assert sharpe >= 1.0, \
        f"Sharpe {sharpe:.2f} below minimum 1.0"
    
    if sharpe < 1.5:
        print(f"⚠️  Sharpe {sharpe:.2f} below research target 1.5-2.5")

def test_average_trade_above_costs():
    """Test avg trade return exceeds transaction costs."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data, fees=0.002, slippage=0.0015)
    
    avg_trade = pf.trades.returns.mean()
    transaction_cost = 0.0035  # 0.2% fees + 0.15% slippage
    
    # Should exceed costs by at least 50%
    assert avg_trade >= transaction_cost * 1.5, \
        f"Avg trade {avg_trade:.4f} barely exceeds costs {transaction_cost}"
    
    print(f"✓ Avg trade {avg_trade:.4f} ({avg_trade/transaction_cost:.1f}x costs)")
```

**Exit Type Validation:**
```python
def test_exit_type_distribution():
    """Test exits follow asymmetric pattern (few stops, mostly EOD)."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    trades = pf.trades.records_readable
    
    # Categorize exits (this depends on VectorBT field names)
    eod_exits = trades[trades['Stop Type'] == 'Time']  # or similar
    stop_exits = trades[trades['Stop Type'] == 'StopLoss']  # or similar
    
    eod_pct = len(eod_exits) / len(trades)
    stop_pct = len(stop_exits) / len(trades)
    
    # Asymmetric strategy: most trades should hit EOD (winners running)
    assert eod_pct >= 0.50, \
        f"Only {eod_pct:.1%} EOD exits - winners may not be running"
    
    # Stop hits should be minority (< 30%)
    assert stop_pct <= 0.30, \
        f"Too many stop hits ({stop_pct:.1%}) - stops may be too tight"
    
    print(f"✓ Exit distribution: {eod_pct:.1%} EOD, {stop_pct:.1%} stops")
```

**Statistical Significance Validation:**
```python
def test_statistical_significance():
    """Test trade count sufficient for conclusions."""
    data = load_test_data('SPY', '2021-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    trade_count = pf.trades.count()
    
    # Minimum: 100 trades for ±10% confidence on win rate
    assert trade_count >= 100, \
        f"Only {trade_count} trades - need 100+ for statistical significance"
    
    # Calculate confidence interval
    win_rate = pf.trades.win_rate
    std_error = np.sqrt(win_rate * (1 - win_rate) / trade_count)
    margin_of_error = 1.96 * std_error  # 95% confidence
    
    print(f"✓ Win rate: {win_rate:.1%} ± {margin_of_error:.1%} (95% CI)")
    
    if margin_of_error > 0.10:
        print(f"⚠️  Large confidence interval ({margin_of_error:.1%})")
        print(f"   Need {int(384 * win_rate * (1-win_rate) / 0.05**2)} trades for ±5% CI")
```

### Level 4: Regime Tests (Robustness)

**Purpose:** Verify strategy works across different market conditions.

**Bull Market Test:**
```python
def test_bull_market_performance():
    """Test strategy performs in bull markets."""
    # 2023-2024 was strong bull market
    data = load_test_data('SPY', '2023-01-01', '2024-12-31')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    # Should be profitable in bull market (ORB designed for trends)
    assert pf.total_return > 0, "Negative returns in bull market"
    assert pf.sharpe_ratio > 0.5, f"Low Sharpe ({pf.sharpe_ratio:.2f}) in bull market"
    
    print(f"✓ Bull market: {pf.total_return:.2%} return, {pf.sharpe_ratio:.2f} Sharpe")

def test_bear_market_performance():
    """Test strategy performs in bear markets."""
    # 2022 was bear market (-18% SPY)
    data = load_test_data('SPY', '2022-01-01', '2022-12-31')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    # Should still work (shorts or reduced drawdown)
    # If enable_shorts=False, may underperform but shouldn't crash
    print(f"✓ Bear market: {pf.total_return:.2%} return, {pf.sharpe_ratio:.2f} Sharpe")
    
    # Compare to buy-and-hold
    bnh_return = (data['Close'][-1] / data['Close'][0]) - 1
    print(f"  Buy-and-hold: {bnh_return:.2%}")
    
    # Strategy should outperform or lose less
    assert pf.total_return >= bnh_return * 0.5, \
        "Strategy lost more than 50% of buy-and-hold loss"

def test_volatile_market_performance():
    """Test strategy performs in high volatility periods."""
    # 2020 COVID was extremely volatile
    data = load_test_data('SPY', '2020-01-01', '2020-12-31')
    strategy = OpeningRangeBreakout()
    pf = strategy.backtest(data)
    
    # High volatility should increase trade count
    trade_count = pf.trades.count()
    print(f"✓ Volatile period: {trade_count} trades")
    
    # Should still maintain risk management
    assert pf.max_drawdown < 0.40, \
        f"Excessive drawdown ({pf.max_drawdown:.2%}) in volatile period"
```

### Level 5: Walk-Forward Tests (Overfitting Detection)

**Purpose:** Verify strategy isn't overfit to historical data.

**Walk-Forward Analysis:**
```python
def test_walk_forward_analysis():
    """Test strategy robustness via walk-forward analysis."""
    data = load_test_data('SPY', '2019-01-01', '2025-10-01')
    strategy = OpeningRangeBreakout()
    
    # Define walk-forward windows
    windows = [
        {'IS': ('2019-01-01', '2020-12-31'), 'OOS': ('2021-01-01', '2021-06-30')},
        {'IS': ('2019-07-01', '2021-06-30'), 'OOS': ('2021-07-01', '2021-12-31')},
        {'IS': ('2020-01-01', '2021-12-31'), 'OOS': ('2022-01-01', '2022-06-30')},
        {'IS': ('2020-07-01', '2022-06-30'), 'OOS': ('2022-07-01', '2022-12-31')},
        {'IS': ('2021-01-01', '2022-12-31'), 'OOS': ('2023-01-01', '2023-06-30')},
        {'IS': ('2021-07-01', '2023-06-30'), 'OOS': ('2023-07-01', '2023-12-31')},
    ]
    
    results = []
    for window in windows:
        # In-sample
        is_data = data[window['IS'][0]:window['IS'][1]]
        is_pf = strategy.backtest(is_data)
        is_sharpe = is_pf.sharpe_ratio
        
        # Out-of-sample
        oos_data = data[window['OOS'][0]:window['OOS'][1]]
        oos_pf = strategy.backtest(oos_data)
        oos_sharpe = oos_pf.sharpe_ratio
        
        results.append({
            'window': f"{window['IS'][0]} to {window['OOS'][1]}",
            'is_sharpe': is_sharpe,
            'oos_sharpe': oos_sharpe,
            'degradation': (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe != 0 else 0
        })
    
    # Check average degradation
    avg_degradation = np.mean([r['degradation'] for r in results])
    
    print("\nWalk-Forward Results:")
    for r in results:
        print(f"  {r['window']}")
        print(f"    IS Sharpe: {r['is_sharpe']:.2f} | OOS Sharpe: {r['oos_sharpe']:.2f}")
        print(f"    Degradation: {r['degradation']:.1%}")
    
    print(f"\nAverage degradation: {avg_degradation:.1%}")
    
    # PASS: If OOS within 20-30% of IS
    assert avg_degradation <= 0.30, \
        f"Average degradation {avg_degradation:.1%} > 30% - likely overfit"
    
    if avg_degradation <= 0.20:
        print("✓ EXCELLENT: OOS performance within 20% of IS")
    elif avg_degradation <= 0.30:
        print("✓ GOOD: OOS performance within 30% of IS")
```

### Test Execution Workflow

**Step 1: Run Unit Tests**
```bash
# Test individual components
pytest tests/test_orb_signals.py -v
pytest tests/test_orb_position_sizing.py -v
pytest tests/test_orb_exits.py -v
```

**Step 2: Run Integration Tests**
```bash
# Test full backtest
pytest tests/test_orb_backtest.py -v
```

**Step 3: Run Validation Tests**
```bash
# Test performance expectations
pytest tests/test_orb_validation.py -v
```

**Step 4: Run Regime Tests**
```bash
# Test across market conditions
pytest tests/test_orb_regimes.py -v
```

**Step 5: Run Walk-Forward Tests**
```bash
# Test for overfitting
pytest tests/test_orb_walk_forward.py -v
```

**If All Tests Pass:**
- Strategy is ready for manual review
- Proceed to creating STRATEGY_2_ORB_RESULTS.md
- Consider paper trading preparation

**If Any Tests Fail:**
- Debug failed component
- Re-run full test suite
- Document what was wrong and how it was fixed
- Do NOT proceed to next phase until all tests pass

---

## Conclusion

This document provides:

1. **Complete forensic analysis of Strategy 1 failure** - Understanding what went wrong and why
2. **Critical learnings for all future strategies** - Patterns to follow and pitfalls to avoid
3. **Position sizing verification protocol** - Must complete before Strategy 2
4. **Detailed Strategy 2 (ORB) design specification** - Every design decision documented
5. **Comprehensive testing and validation protocol** - Ensure quality before deployment

**Next Steps for Claude Code:**

1. ✅ **Read this document completely** - Every section is important
2. ✅ **Run position sizing verification** - Critical prerequisite
3. ✅ **Implement Strategy 2 (ORB)** - Following specification exactly
4. ✅ **Execute testing protocol** - No shortcuts
5. ✅ **Document results** - Create STRATEGY_2_ORB_RESULTS.md

**Key Principles to Remember:**

- **Match philosophy to implementation** - No mixing mean reversion + trend-following
- **Regime matters more than strategy** - Test across multiple market conditions
- **Transaction costs are non-negotiable** - Avg trade must exceed costs
- **Exit logic determines success** - Don't cut winners with signal exits
- **Statistical significance is mandatory** - Need 100+ trades minimum

**Success Criteria:**

Strategy 2 must achieve:
- Win rate: 15-30%
- R:R ratio: > 2:1
- Sharpe ratio: > 1.0
- Avg trade: > 0.5%
- Trade count: > 100
- Walk-forward OOS within 20% of IS

If Strategy 2 meets these criteria, it will be ready for:
- Paper trading (3-6 months minimum)
- Integration into Strategy 3 (TFC hybrid)
- Potential production deployment

**Final Note:**

This is professional quantitative trading. **Speed is not the goal - accuracy is.** Take the time to:
- Verify position sizing before starting
- Test thoroughly at each phase
- Document everything
- Don't skip validation steps

A strategy that takes 2 extra weeks but works correctly is infinitely better than a strategy rushed in 1 week that fails in production.

**Good luck with Strategy 2 implementation.**

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-11  
**Status:** APPROVED FOR STRATEGY 2 DEVELOPMENT  
**Prepared By:** Independent Quantitative Analyst (Claude Desktop)  
**For:** Development Team (Claude Code + Human Team Lead)
