# Credit Spread Strategy - Critical Bugs & Analysis

**Date**: November 8, 2025
**Status**: 🚨 STRATEGY DOES NOT ADD VALUE OVER BUY-AND-HOLD

---

## Executive Summary

**CRITICAL FINDING**: The credit spread timing strategy **underperforms simple SSO buy-and-hold by 37%**.

| Strategy | Return | Multiple |
|----------|--------|----------|
| **SSO Buy & Hold** | **1,439%** | **15.39x** ⭐ BEST |
| Video Claim | 1,537% | 16.37x |
| Our Backtest | 970% | 9.70x |
| SPY Buy & Hold | 669% | 7.69x |

**The strategy loses 37% vs simply buying and holding SSO**, making it fundamentally flawed.

---

## Bug #1: Video Signals Don't Match Stated Rules

### Stated Rules:
- **ENTRY**: Credit spread falls **35%** from recent 60-day high
- **EXIT**: Credit spread rises **40%** from recent 60-day low **AND** crosses above 330-day EMA

### Reality Check (60-day lookback):

**ENTRIES - NONE meet the -35% threshold:**
```
Date         Spread  EMA    Recent High  % from High  MEETS RULE?
2003-04-03   7.31    8.63   8.69         -15.9%       ❌ NO
2006-05-04   2.93    3.54   3.37         -13.1%       ❌ NO
2009-04-30   13.45   12.89  18.86        -28.7%       ❌ NO (close)
2012-03-13   5.97    6.61   7.45         -19.9%       ❌ NO
2016-07-12   5.46    6.15   6.57         -16.9%       ❌ NO
2019-12-13   3.71    4.02   4.40         -15.7%       ❌ NO
2020-05-21   7.08    5.11   10.87        -34.9%       ❌ NO (very close!)
2023-07-15   [weekend - no data]         -            ❌ NO
```

**EXITS - NONE meet the +40% threshold:**
```
Date         Spread  EMA    Recent Low   % from Low   Above EMA?  MEETS RULE?
1998-08-18   3.84    3.02   3.04         +26.3%       ✓ Yes       ❌ NO
2005-04-14   3.75    3.88   2.71         +38.4%       ❌ No       ❌ NO
2007-07-19   3.20    2.99   2.41         +32.8%       ✓ Yes       ❌ NO
2011-08-04   6.16    5.84   4.79         +28.6%       ✓ Yes       ❌ NO
2014-10-09   4.52    4.16   3.72         +21.5%       ✓ Yes       ❌ NO
2018-12-05   4.29    3.70   3.16         +35.8%       ✓ Yes       ❌ NO
2020-02-26   4.27    3.92   3.38         +26.3%       ✓ Yes       ❌ NO
2022-03-14   4.16    3.61   3.01         +38.2%       ✓ Yes       ❌ NO
```

### Conclusion:
**The video's signal dates were NOT generated using their stated rules with 60-day lookback.**

Possible explanations:
1. Different lookback period (252-day gets 75% match but generates 600+ signals)
2. Different calculation method (expanding window vs rolling window)
3. Manual signal selection
4. Error in video's explanation

---

## Bug #2: Strategy Underperforms Buy-and-Hold

### Performance Comparison (2006-06-21 to 2025-11-07):

```
BENCHMARK COMPARISON:
┌────────────────────────┬──────────┬──────────┬────────────┐
│ Strategy               │ Return   │ Multiple │ vs SSO B&H │
├────────────────────────┼──────────┼──────────┼────────────┤
│ SSO Buy & Hold         │ 1,439%   │ 15.39x   │ 0%         │
│ Video Claim            │ 1,537%   │ 16.37x   │ +6.4%      │
│ Our Backtest           │   970%   │  9.70x   │ -37.0% ❌  │
│ SPY Buy & Hold         │   669%   │  7.69x   │ -50.0%     │
└────────────────────────┴──────────┴──────────┴────────────┘
```

**Key Insight**:
- Our strategy (9.7x) beats SPY (7.69x) by 26%
- Our strategy (9.7x) **loses to SSO (15.39x) by 37%** ← THIS IS THE PROBLEM
- The QuantStats report shows vs SPY, making the strategy look better than it is

### Why This Matters:
**You could have simply bought SSO and held it for 15.39x instead of using the strategy for 9.7x.**

The "credit spread timing" is destroying value, not adding it.

---

## Bug #3: Misleading Benchmark in QuantStats

Our QuantStats tearsheet shows:
```
Strategy:   969.74%
Benchmark:  936.54%  ← This is SPY, not SSO!
```

**This makes the strategy look good** (3.3% better than benchmark), but the **correct benchmark should be SSO**:
```
Strategy:   969.74%
Correct Benchmark: 1,439.10%  ← SSO
UNDERPERFORMANCE: -37.0% ❌
```

---

## Bug #4: Lookback Period Mystery

Testing different lookback periods for "recent high/low":

| Lookback | Entry Matches | Exit Matches | Total Match Rate | Total Signals Generated |
|----------|---------------|--------------|------------------|-------------------------|
| 20 days  | 0/8           | 1/8          | 6.2%             | 23                      |
| 40 days  | 0/8           | 3/8          | 18.8%            | 68                      |
| 60 days  | 1/8           | 5/8          | 37.5%            | 129                     |
| 90 days  | 1/8           | 7/8          | 50.0%            | 220                     |
| 120 days | 4/8           | 7/8          | 68.8%            | 324                     |
| 252 days | 5/8           | 7/8          | 75.0%            | **691** ⚠️               |

**Problem**:
- 60-day lookback (as stated in video) gives only 37.5% match
- 252-day lookback gives 75% match but generates 691 signals (way too many)
- No lookback period generates the exact 16 video signals with correct timing

**Conclusion**: The video's signal generation algorithm is unclear or incorrectly explained.

---

## Bug #5: Video Claim is Suspiciously Close to Buy-and-Hold

```
Video Claim:        16.37x
SSO Buy-and-Hold:   15.39x
Difference:         +6.4%
```

**The video's 16.37x return is almost identical to SSO buy-and-hold (15.39x).**

Possible explanations:
1. Video made an error and actually compared SSO buy-and-hold
2. Video used different date range or ETF
3. Video's signals are more optimal than we can replicate
4. Video's backtest had errors

---

## Root Cause Analysis

### Why Our Strategy Underperforms SSO Buy-and-Hold:

**Time Out of Market = Opportunity Cost**

```
SSO Buy-and-Hold: 100% in market, captures ALL upside
Our Strategy:      59% in market, misses 41% of upside

Key Periods Missed:
- 2007-2008: Out during crash ✓ (good)
- 2008-2009: OUT during recovery ❌ (bad - missed 2x gain)
- 2011-2012: Out during consolidation ✓ (neutral)
- 2014-2016: OUT during bull run ❌ (bad - missed gains)
- 2020:      Quick exit/re-entry (mixed)
- 2022-2023: OUT during recovery ❌ (bad)
```

**The Problem**: Credit spread signals exit too early and re-enter too late, missing major recovery moves.

---

## Signal Algorithm Issues

### Issue 1: "Since Last Signal" Algorithm
Our best-performing algorithm (50% match) uses:
- Track state since last entry/exit
- Compare current spread to recent high/low since state change
- This creates expanding window, not fixed lookback

**Problem**:
- Still only 50% match with video
- Generates different entry/exit timing
- Results in worse performance

### Issue 2: Fixed Lookback Windows
All fixed lookback periods (20, 60, 120, 252 days) fail to replicate video signals:
- Too short: miss signals
- Too long: generate excessive signals

### Issue 3: Exit Condition Complexity
Exit requires TWO conditions:
1. Rise 40% from low
2. Cross above EMA

**Our analysis shows** most video exits don't meet condition #1, suggesting:
- Video may use different threshold
- Video may use different "low" calculation
- Video may prioritize EMA crossover over 40% threshold

---

## Fees and Costs Impact

### Our Assumptions:
- Trading fees: 0.1% (10 basis points) per trade
- Total fees paid: £823.47 over 19 years
- 15 orders (7-8 trades)

### Impact Analysis:
```
Without fees: ~9.78x
With fees:     9.70x
Fee drag:     -0.8%
```

**Conclusion**: Fees are negligible. The underperformance is due to signal timing, not costs.

---

## Comparison to Video Metrics

| Metric | Video Claim | Our Backtest | Difference |
|--------|-------------|--------------|------------|
| Total Return | 16.37x | 9.70x | -40.7% |
| Number of Trades | "8 trades" | 7 trades | -1 trade |
| Win Rate | Not stated | 83.33% | N/A |
| Max Drawdown | Not stated | -45.50% | N/A |
| Time in Market | Not stated | 59.33% | N/A |

---

## What Would Make This Strategy Viable?

### Option 1: More Accurate Signals (Get to 16.37x)
**Requirements**:
- Replicate video's exact signal algorithm (currently 50% match)
- Need 90%+ match to approach video's 16.37x performance
- Even then, only 6% better than SSO buy-and-hold

**Verdict**: Not worth the effort for 6% improvement

---

### Option 2: Use 3x Leverage (SPXL instead of SSO)
**If SSO (2x) underperforms, SPXL (3x) might work better?**

**Problem**:
- SPXL has higher volatility decay
- Credit spread timing would need to be even more precise
- Likely to perform even worse than current results

**Verdict**: High risk, likely worse results

---

### Option 3: Improve Exit Timing
**Current exits are too early**, missing recovery rallies.

**Possible improvements**:
- Delay exit until stronger confirmation
- Use multiple timeframe analysis
- Add momentum filter

**Problem**:
- Would make strategy more complex
- No guarantee of beating buy-and-hold
- Overfitting risk

**Verdict**: Not recommended without extensive walk-forward testing

---

## Final Verdict

### Strategy Viability: ❌ NOT VIABLE

**Reasons**:
1. **Underperforms simple buy-and-hold by 37%**
2. Signals don't match stated rules
3. Cannot replicate video's claimed performance
4. Video's claim (16.37x) is suspiciously close to buy-and-hold (15.39x)
5. Even if we matched video's signals perfectly, only 6% better than buy-and-hold

### Recommendation for ATLAS Integration

**DO NOT INTEGRATE** into ATLAS system.

**Alternative Use Cases**:
1. **Academic Exercise**: Study credit spreads as market indicator (not trading signal)
2. **Risk-Off Signal**: Use credit spread widening as warning sign, but don't act on timing
3. **Regime Indicator**: Incorporate into ATLAS 4-regime classification, not as standalone signal

---

## Lessons Learned

### For Future Strategy Evaluation:

1. **Always compare to correct benchmark**
   - If using leveraged ETF, compare to leveraged benchmark
   - Our QuantStats initially compared to SPY, hiding the underperformance

2. **Verify signal logic before implementation**
   - Video's stated rules didn't match their actual signals
   - 50% signal match = fundamentally different algorithm

3. **Opportunity cost matters**
   - Time out of market has real cost
   - Need to outperform buy-and-hold to justify complexity

4. **Don't trust YouTube strategy claims**
   - Video's 16.37x is suspiciously close to SSO buy-and-hold
   - Possible errors or misleading presentation

5. **Use QuantStats for professional reporting**
   - But verify benchmark selection
   - Always check absolute performance vs buy-and-hold

---

## Action Items

### ✅ Completed:
- [x] Identified signal generation discrepancies
- [x] Discovered strategy underperformance vs buy-and-hold
- [x] Tested multiple lookback periods
- [x] Generated professional QuantStats tearsheet
- [x] Documented all findings

### ❌ Do NOT Pursue:
- [ ] ~~Attempt to improve signal algorithm~~ (not worth it)
- [ ] ~~Test with SPXL (3x)~~ (higher risk, likely worse)
- [ ] ~~Integrate into ATLAS~~ (strategy doesn't add value)

### ✅ Next Steps (if desired):
- [ ] Study credit spreads as indicator only (not trading signal)
- [ ] Research why timing strategies tend to underperform in bull markets
- [ ] Document this as case study for strategy evaluation methodology

---

## Conclusion

The Credit Spread Leveraged ETF Strategy, as presented in the video, **cannot be replicated with acceptable accuracy** and **underperforms simple buy-and-hold** by a significant margin.

**Key takeaway**: Sometimes the best strategy is the simplest one - buy and hold a leveraged index fund if you believe in long-term equity growth.

**For ATLAS**: This analysis reinforces the importance of the 4-regime system. A regime-based approach that modulates exposure (rather than binary on/off) may avoid the opportunity cost pitfalls of this credit spread timing strategy.

---

**Analysis Date**: November 8, 2025
**Files**: See strategy_research/credit_spread/ for all supporting code and data
**Status**: Analysis complete, strategy rejected for ATLAS integration
