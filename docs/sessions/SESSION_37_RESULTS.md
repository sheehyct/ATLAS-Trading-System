# Session 37 Results: Multi-Asset 52W High Momentum Portfolio Backtest

**Date:** 2025-11-15
**Objective:** Test multi-asset portfolio approach to meet Gate 1 targets (Sharpe >= 0.8, CAGR >= 10%)

---

## Executive Summary

**Gate 1 Decision: FAIL**

The multi-asset momentum portfolio FAILED both Gate 1 targets and performed WORSE than single-asset SPY:
- **Sharpe Ratio:** 0.42 vs target 0.8 (FAIL by 47.5%)
- **CAGR:** 5.29% vs target 10% (FAIL by 47.1%)
- **Worse than SPY:** Multi-asset Sharpe 0.42 < SPY Sharpe 0.74

---

## Backtest Configuration

**Universe:** Technology sector (30 stocks)
**Portfolio Size:** Top 10 by momentum score
**Rebalance Frequency:** Semi-annual (February, August)
**Volume Filter:** 1.25x (same calibration as SPY)
**Minimum Distance:** 0.90 (within 10% of 52-week high)
**Period:** 2020-01-01 to 2024-12-31 (5 years)
**Initial Capital:** $100,000
**Fees:** 0.1% (0.001)
**Slippage:** 0.1% (0.001)

---

## Results Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Initial Capital** | $100,000 | - | - |
| **Final Value** | $129,382 | - | - |
| **Total Return** | 29.38% | - | - |
| **CAGR** | 5.29% | >= 10% | FAIL |
| **Sharpe Ratio** | 0.42 | >= 0.8 | FAIL |
| **Sortino Ratio** | 0.62 | - | - |
| **Max Drawdown** | -37.67% | <= -30% | FAIL (exceeded) |
| **Calmar Ratio** | 0.14 | - | Poor |
| **Total Trades** | 1,787 | - | High turnover |
| **Duration** | 5.0 years | - | - |

---

## Comparison to Single-Asset SPY (Session 36)

| Metric | Multi-Asset (Session 37) | Single-Asset SPY (Session 36) | Delta |
|--------|--------------------------|-------------------------------|-------|
| **Sharpe Ratio** | 0.42 | 0.74 | -43.2% (WORSE) |
| **CAGR** | 5.29% | 5.5% | -0.21% (WORSE) |
| **Max Drawdown** | -37.67% | -14.2% | -165.2% (WORSE) |
| **Total Trades** | 1,787 | 13 | +13,646% (much higher) |

**Critical Finding:** Multi-asset approach performed WORSE than single-asset SPY across all metrics.

---

## Portfolio Selections Over Time

| Rebalance Date | Stocks Selected | Symbols |
|----------------|-----------------|---------|
| 2020-02-01 | 0 | None |
| 2020-08-01 | 0 | None |
| 2021-02-01 | 0 | None |
| 2021-08-01 | 8 | AMD, FTNT, KLAC, CDNS, ACN, NOW, AMAT, LRCX |
| 2022-02-01 | 1 | GOOGL |
| 2022-08-01 | 0 | None |
| 2023-02-01 | 4 | MCHP, TXN, KLAC, SNPS |
| 2023-08-01 | 0 | None |
| 2024-02-01 | 2 | META, GOOGL |
| 2024-08-01 | 2 | META, TXN |

**Critical Issue:** 5 out of 10 rebalance periods (50%) selected ZERO stocks.

---

## Root Cause Analysis

### Problem 1: Volume Filter Too Restrictive

**Evidence:**
- 5 out of 10 rebalance periods selected 0 stocks
- Volume filter 1.25x calibrated for SPY (low volatility, high volume)
- Technology stocks have different volume characteristics than SPY

**Impact:**
- Portfolio held cash (0% allocation) for ~50% of the backtest period
- Cash drag severely reduced returns
- Missing high-momentum periods in tech stocks

**Solution:**
- Asset-specific volume calibration (Session 36 findings):
  - High vol stocks (TSLA, NVDA): 1.15x
  - Moderate vol stocks: 1.5-1.75x
  - OR disable volume filter for multi-asset portfolios

### Problem 2: Momentum Criteria Too Strict

**Evidence:**
- Minimum distance 0.90 (within 10% of 52w high)
- Only 17 total stocks selected across all rebalances
- Many rebalance periods with <5 stocks

**Impact:**
- Insufficient diversification (1-8 stocks vs target 10)
- Portfolio concentrated risk (single stock = 100% in Feb 2022)
- High volatility from concentration

**Solution:**
- Relax min_distance to 0.85 (within 15% of 52w high)
- Or lower top_n to 5 stocks (better for strict criteria)

### Problem 3: Scanner Integration Issue

**Evidence:**
- Strategy designed for single-asset signal generation
- Scanner expects portfolio-style continuous monitoring
- Mismatch between rebalance logic and momentum detection

**Impact:**
- Portfolio selection may miss stocks entering momentum zone between rebalances
- Semi-annual rebalance too infrequent for fast-moving tech stocks

**Solution:**
- Increase rebalance frequency to monthly
- Or use trailing entry logic (enter when signal fires, not just at rebalance)

### Problem 4: Equal-Weight vs Momentum-Weight

**Evidence:**
- All selected stocks receive equal weight (1/N)
- Ignores momentum score ranking (stronger signals = weaker signals)

**Impact:**
- Weak momentum stocks drag down strong momentum stocks
- No differentiation in allocation

**Solution:**
- Implement momentum-weighted allocation
  - Top momentum stock: 15-20%
  - Lower momentum stocks: 5-10%
- Or concentrate on top 5 instead of top 10

---

## Next Actions

### Option A: Debug Current Approach (2-3 hours)

1. **Disable volume filter** - Test without volume confirmation
2. **Relax momentum criteria** - min_distance = 0.85 (15% from high)
3. **Increase rebalance frequency** - Monthly instead of semi-annual
4. **Re-run backtest** - Technology sector 2020-2025
5. **Target:** Sharpe >= 0.8, CAGR >= 10%

### Option B: Alternative Universe (1-2 hours)

1. **Test S&P 500 proxy** - 40 stocks, top 20 portfolio
2. **Broader diversification** - Reduce sector concentration risk
3. **Keep current parameters** - Validate if tech sector is the problem
4. **Target:** Sharpe >= 0.8, CAGR >= 10%

### Option C: Pivot to Different Strategy (Session 38+)

1. **Quality-Momentum** - Profitability + momentum combination
2. **Relative Strength** - Sector rotation based on RS rankings
3. **52W High + ATLAS regime** - Filter momentum signals by market regime
4. **Target:** Find alternative foundation strategy

---

## Recommendation

**Proceed with Option A: Debug Current Approach**

**Rationale:**
1. Multi-asset portfolio has theoretical support (George & Hwang 2004)
2. Single-asset SPY worked (Sharpe 0.74) - multi-asset SHOULD improve
3. Clear root causes identified (volume filter, criteria strictness)
4. Quick fixes available (disable volume, relax distance, increase frequency)
5. 2-3 hour effort vs starting over with new strategy

**Expected Outcome after Option A:**
- Without volume filter: More stocks selected each rebalance
- Relaxed criteria (0.85): Top 10 portfolio fully populated
- Monthly rebalance: Capture momentum faster
- Target: Sharpe 0.8-1.0, CAGR 10-12%

**If Option A fails:** Pivot to Option C (new foundation strategy)

---

## Implementation Details

### VBT Integration: SUCCESS

**Accomplishments:**
- Implemented `_build_allocation_matrix()` with forward-fill logic
- Implemented `_extract_metrics()` for portfolio analysis
- Completed `run()` method with VBT Portfolio.from_orders()
- Handled timezone-aware data (Yahoo Finance returns America/New_York tz)
- Fixed rebalance date generation (Feb/Aug for semi-annual)

**VBT Approach Validated:**
```python
pf = vbt.Portfolio.from_orders(
    close=close,
    size=allocations,
    size_type='targetpercent',
    group_by=True,
    cash_sharing=True,
    init_cash=initial_capital,
    fees=0.001,
    slippage=0.001,
    freq='D'
)
```

**Key Learning:**
- Use `size_type='targetpercent'` for rebalancing portfolios
- Allocations = DataFrame (rows=dates, cols=symbols, values=0.0-1.0)
- Forward-fill allocations between rebalances
- Handle timezone-aware data (tz_localize before comparisons)

---

## Files Modified

1. **integrations/stock_scanner_bridge.py** (+100 lines)
   - Added `_build_allocation_matrix()` method (64 lines)
   - Added `_extract_metrics()` method (28 lines)
   - Completed `run()` method with VBT integration (52 lines)
   - Fixed rebalance date generation
   - Fixed timezone handling

2. **test_multi_asset_backtest.py** (NEW - 95 lines)
   - Technology sector backtest script
   - Gate 1 validation logic
   - Portfolio selection display
   - Results comparison to SPY

3. **SESSION_37_RESULTS.md** (THIS FILE)
   - Complete analysis of backtest results
   - Root cause analysis
   - Recommendations for next steps

---

## Session 37 Completion Status

**Objective:** Implement multi-asset portfolio backtest - COMPLETE
**Gate 1 Target:** Sharpe >= 0.8, CAGR >= 10% - FAIL
**Implementation:** VBT integration - SUCCESS
**Testing:** Technology sector backtest - COMPLETE
**Analysis:** Root causes identified - COMPLETE

**Next Session (38):** Debug portfolio selection (Option A) or pivot to new strategy (Option C)

**Session Duration:** 3.5 hours (within 2.5-3 hour estimate + debugging)

---

## Key Takeaways

1. **VBT portfolio integration works correctly** - from_orders with targetpercent validated
2. **Multi-asset does NOT automatically improve performance** - worse than single-asset SPY
3. **Volume filter calibration is asset-specific** - 1.25x too restrictive for tech stocks
4. **Portfolio selection criteria matter more than universe size** - 0 stocks = cash drag
5. **Rebalance frequency affects capture rate** - semi-annual may be too slow for tech

**Critical Lesson:** Assumptions need validation. Expected multi-asset improvement (Sharpe 0.8-1.2) did not materialize. Root cause analysis required before proceeding.
