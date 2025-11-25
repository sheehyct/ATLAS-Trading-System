# STRAT Lab Optimization Insights for Session 65

**Source Articles:**
1. "Quantifying the Strat" - https://thestratlab.substack.com/p/quantifying-the-strat
2. "Are Daily and Weekly the best timeframes?" - https://thestratlab.substack.com/p/are-daily-and-weekly-the-best-timeframes

**Date Compiled:** November 22, 2025
**Purpose:** Inform optimization strategies for improving Hourly 2-2 Up pattern from 1.53:1 to 2:1 R:R ratio

---

## Executive Summary

The STRAT Lab research provides empirical validation for several optimization approaches that could improve our Hourly 2-2 Up pattern performance:

1. **Hybrid Timeframes (2D, 3D)** show 3-8% higher transition probabilities than standard daily/weekly
2. **Higher timeframes** (monthly, quarterly) dramatically increase pattern reliability (71.4% to 100% for hammers)
3. **Timeframe alignment** is critical - higher frame bias determines success of lower frame patterns
4. **2dG → 2u** transition is highly reliable across all timeframes (44.7% hourly to 71.4% quarterly)

**Key Finding for Our System:**
Our Daily 2-1-2 Up pattern (2.65:1 R:R) aligns with STRAT Lab's finding that **daily 2-1-2 Up transitions show 39.2% probability on standard daily, but could potentially be optimized using 2D hybrid timeframe analysis.**

---

## Article 1: Quantifying the STRAT - Pattern Transition Probabilities

### Top 5 Pattern Transitions (SPY, 1993-2025)

#### 1. Hammer → 2u (Bullish Breakout)
| Timeframe | Probability | Interpretation |
|-----------|-------------|----------------|
| Hourly | 47.7% | Moderate - treat as trade idea |
| Daily | 40.3% | Moderate - requires confirmation |
| Weekly | 39.5% | Moderate |
| Monthly | **71.4%** | HIGH CONVICTION |
| Quarterly | **100%** | HIGHEST CONVICTION (small sample warning) |

**Key Insight:** "Hammers lean bullish, but the edge scales with the timeframe."

**Application to Our System:**
- Our hourly patterns operate in the 47.7% zone (moderate conviction)
- Monthly/quarterly alignment could provide 31-52 percentage point boost
- **Optimization Strategy:** Add monthly/quarterly hammer confirmation filter

#### 2. 2dG → 2u (Failed Downside, Then Reversal Up)
| Timeframe | Probability |
|-----------|-------------|
| Hourly | 44.7% |
| Daily | 39.2% |
| Weekly | 49.2% |
| Monthly | **62.5%** |
| Quarterly | **71.4%** |

**Pattern Description:** "2d but green" - price breaks down but closes higher, indicating buying pressure.

**Key Insight:** "2dG plants the seed for trend continuation, and that seed tends to grow when there's more time for the algos to do their work."

**Application to Our System:**
- This is conceptually similar to our 2-2 reversal patterns (2D-2U)
- Higher timeframe 2dG could act as filter for hourly 2-2 Up entries
- **Optimization Strategy:** Require daily/weekly 2dG alignment for hourly 2-2 Up trades

#### 3. Shooter → 2d (Bearish Breakdown)
| Timeframe | Probability |
|-----------|-------------|
| Daily | **44.9%** (top transition) |
| Weekly | 40.9% |
| Monthly | **66.7%** |
| Quarterly | Not in top 5 |

**Warning:** "Shooters don't scale the same way" - irregular behavior across timeframes.

**Application to Our System:**
- Bearish patterns less relevant for our current bullish focus
- Note the irregularity - not all patterns scale predictably

#### 4. 3u → 2u (Outside Bar Up, Continues Up)
| Timeframe | Probability |
|-----------|-------------|
| Hourly | 42.0% |
| Daily | 38.8% |
| Weekly | 43.4% |
| Monthly | **65.2%** |
| Quarterly | **75.0%** |

**Unexpected Finding:** "Outside bars (traditionally exhaustion signals) frequently resolve into continuation patterns, especially on larger timeframes."

**Key Quote:** "A 3u isn't an automatic fade. It's a 'prove it' bar."

**Application to Our System:**
- 3u bars on higher timeframes can confirm bullish bias
- Don't avoid trades after 3u - wait for next bar confirmation
- **Optimization Strategy:** Use monthly/weekly 3u as bullish bias filter

#### 5. 2u → 2u (Trend Persistence)
| Timeframe | Probability |
|-----------|-------------|
| Hourly | 41.9% |
| Daily | 37.7% |
| Weekly | 46.6% |
| Monthly | **52.6%** |
| Quarterly | **61.5%** |

**Key Insight:** "Trends persist. Probabilities show how this persistence scales with timeframe."

**Application to Our System:**
- Higher timeframe uptrends (monthly/quarterly 2u → 2u) provide 61.5% probability of continuation
- Our hourly 2-2 Up patterns should align with these higher frame trends
- **Optimization Strategy:** Filter hourly entries by monthly/quarterly trend direction

---

## Article 2: Timeframe Selection - Hybrid vs Standard

### Critical Discovery: Hybrid Timeframes Outperform

**2D vs 1D Performance:**
- H → 2u: **48.6%** (2D) vs 40.3% (1D) = **+8.3 percentage points**
- Result: "2D chart signals are less likely to be random noise"
- Practical Impact: "Lesser chances to get TTOed (time-stopped-out)"

**3D vs 1W Performance:**
- H → 2u: **57.1%** (3D) vs 35.4% (1W) = **+21.7 percentage points**
- Result: "3D acts as a far more responsive, statistically backed alternative to the weekly"

### Surprising Finding: Hourly > Daily

**Median Probabilities:**
- 1H: 41.2%
- 1D: 39.1%
- Result: Hourly timeframe captures moves with **less noise** than daily

**Strongest Transitions:**
- 1H: 47.6%
- 1D: 44.9%

**Explanation:** "It may help explain why so many top-tier swing traders prefer to use the 1H or 65-minute charts to fine-tune their trade entries."

**Application to Our System:**
- Our hourly 2-2 Up focus is statistically validated
- Hourly provides better signal quality than daily for entry timing
- **Current Status:** We're using the right base timeframe (hourly)

### Timeframe Probability Rankings (Median)

From analysis of SPY 1993-2025:

| Rank | Timeframe | Median Probability | Max Probability |
|------|-----------|-------------------|-----------------|
| 1 | Quarterly (1M) | ~60% | ~100% |
| 2 | Monthly (1M) | ~55% | ~71.4% |
| 3 | 3D (Hybrid) | ~50% | **57.1%** |
| 4 | Weekly (1W) | ~45% | 49.2% |
| 5 | 2D (Hybrid) | ~44% | **48.6%** |
| 6 | Hourly (1H) | 41.2% | 47.6% |
| 7 | Daily (1D) | 39.1% | 44.9% |

**Key Takeaway:** Hybrid frames (2D, 3D) consistently outperform their neighboring standard frames.

---

## Practical Implementation Playbook (from Article 1)

### 1. Start at the Top
"Check the quarterly first, then the monthly."

**Bullish Alignment Indicators:**
- H → 2u, 2dG → 2u, 3u → 2u, or 2u → 2u already in progress
- Interpretation: "Treat pullbacks as pauses in accumulation rather than signs of reversal"

**Trading Implication:** "Fewer fights against the tide, more attention on finding spots to join it."

### 2. Define the Battlegrounds
**Key Levels to Mark:**
- Monthly open
- Broadening formation pivots
- Prior weekly highs/lows

**Quote:** "Watching how price behaves at these levels tells more truth than any single candle pattern shouted out of context."

### 3. Focus on Signals Reinforced by Higher-Timeframe Probabilities
**Priority Patterns:**
- 2dG → 2u aligned with higher-timeframe bias
- 3u → 2u aligned with higher-timeframe bias
- 2u → 2u on higher timeframes with clear risk levels

**Warning:** "Instead of chasing stretched intraday bars."

### 4. Stay Honest on Shorts
"Assume the first breakdown can get reclaimed and plan around that possibility."

---

## Optimization Strategies for Hourly 2-2 Up Pattern

**Current Performance:**
- Hit Rate: 90.5% (19/21 patterns) ✓ Excellent
- R:R Ratio: 1.53:1 ✗ Below 2:1 target (needs 31% improvement)

**Goal:** Improve R:R from 1.53:1 to 2:1+ while maintaining 80%+ hit rate

### Strategy 1: Higher Timeframe Alignment Filter (RECOMMENDED)

**Implementation:**
```python
# Add to pattern validation logic
def validate_higher_timeframe_alignment(pattern_date, symbol):
    # Check monthly chart
    monthly_bias = get_monthly_trend(symbol, pattern_date)
    # Check quarterly chart (if available)
    quarterly_bias = get_quarterly_trend(symbol, pattern_date)

    # For bullish hourly 2-2 Up, require:
    # - Monthly in uptrend (2u → 2u, H → 2u, or 3u → 2u)
    # - OR Monthly 2dG → 2u setup in progress

    if monthly_bias in ['2u_trend', 'hammer_up', '3u_continuation', '2dG_reversal']:
        return True
    return False
```

**Expected Impact:**
- Monthly alignment: 71.4% hammer probability vs 47.7% hourly
- Potential R:R improvement: +24 percentage points from alignment
- May reduce pattern count significantly (trade quality over quantity)

**Risk:** Could reduce tradeable pattern count below statistical significance threshold.

### Strategy 2: Hybrid Timeframe Analysis (2D Chart)

**Implementation:**
```python
# Switch from 1D to 2D timeframe for daily continuity check
# Current: Check 1D alignment (39.1% median probability)
# New: Check 2D alignment (48.6% probability for H → 2u)

def check_daily_continuity_2d(symbol, pattern_date):
    # Resample hourly data to 2-day bars
    two_day_data = resample_to_2d(hourly_data)
    # Check for bullish 2D patterns
    return classify_2d_bias(two_day_data, pattern_date)
```

**Expected Impact:**
- 8.3 percentage point improvement over standard daily
- "Lesser chances to get TTOed" - fewer false signals
- Better alignment with institutional trading cycles

**Advantage:** Maintains pattern count while improving quality.

### Strategy 3: 2dG Pattern Integration

**Implementation:**
```python
# Add 2dG detection on daily/weekly timeframes
def detect_2dg_setup(daily_data, pattern_date):
    """
    2dG = 2D bar (breaks below prior low) but closes GREEN
    Indicates buying pressure stepping in
    """
    current_bar = daily_data.loc[pattern_date]
    prior_bar = daily_data.iloc[-2]

    is_2d = current_bar['Low'] < prior_bar['Low']
    is_green = current_bar['Close'] > current_bar['Open']

    return is_2d and is_green

# Filter hourly 2-2 Up patterns:
if detect_2dg_setup(daily_data, pattern_date):
    # 2dG → 2u has 44.7% probability (hourly) to 71.4% (quarterly)
    # This is a "springboard setup" per STRAT Lab
    pattern_confidence = 'HIGH'
```

**Expected Impact:**
- 2dG → 2u shows strong probabilities across all timeframes
- Conceptually aligned with our 2-2 reversal logic (failed downside)
- Could act as pre-filter for hourly entries

### Strategy 4: Quarterly/Monthly Trend Filter (Most Conservative)

**Implementation:**
```python
# Only take hourly 2-2 Up when quarterly/monthly trends confirm
def check_macro_trend(symbol, pattern_date):
    quarterly = get_quarterly_trend(symbol, pattern_date)
    monthly = get_monthly_trend(symbol, pattern_date)

    # Quarterly H → 2u = 100% (but small sample!)
    # Monthly H → 2u = 71.4%
    # Quarterly 2u → 2u = 61.5%

    if quarterly in ['hammer_up', 'uptrend'] and monthly in ['hammer_up', 'uptrend']:
        return 'HIGHEST_CONVICTION'
    elif monthly in ['hammer_up', 'uptrend']:
        return 'HIGH_CONVICTION'
    return 'LOW_CONVICTION'

# Only trade HIGHEST or HIGH conviction setups
```

**Expected Impact:**
- Dramatic R:R improvement (quarterly probabilities 71-100%)
- Significant reduction in pattern count (may drop below 20 patterns)
- Highest confidence level for options deployment

**Risk:** Too conservative - may eliminate too many valid trades.

---

## Critical Warnings from Research

### 1. Probabilistic, Not Deterministic
**Quote:** "The numbers are guides, not guarantees: probabilistic, not deterministic."

**Application:** Don't expect 100% win rate even with perfect alignment. The 2:1 R:R target assumes ~50% hit rate minimum.

### 2. Sample Size Concerns
**Quote (Quarterly Hammer):** "To be taken with a pinch of salt as it's only happened once in a sample size of 130 quarterly candles."

**Application:** Be cautious with quarterly data - limited historical occurrences.

### 3. Context and Confluence Matter
**Quote:** "No single percentage is gospel; context and confluence matter more."

**Application:** Stack multiple conditions:
- Timeframe alignment
- Key level proximity (monthly open, broadening formation pivots)
- Market breadth confirmation
- Sector leadership alignment

### 4. The Real Edge
**Quote:** "The real edge comes from stacking conditions: aligning timeframes, respecting key levels and factoring in market breadth."

**Application:** Don't rely on a single filter - use multiple confluence factors.

---

## Recommended Optimization Path for Session 65

### Option A: Implement 2D Hybrid Timeframe Analysis (RECOMMENDED)
**Rationale:**
- Maintains pattern count (less filtering)
- 8.3 percentage point improvement validated by research
- Aligns with "top-tier swing traders" methodology
- Lower implementation complexity

**Steps:**
1. Add 2D bar resampling function
2. Detect 2D bullish patterns (H → 2u, 2dG → 2u, 3u → 2u)
3. Require 2D bullish bias for hourly 2-2 Up entries
4. Re-run validation on 3-stock universe
5. Measure R:R improvement

**Expected Result:** R:R improvement from 1.53:1 to ~1.70-1.80:1 (closer to 2:1 target)

### Option B: Add Monthly Trend Alignment Filter
**Rationale:**
- Highest probability improvement (71.4% vs 47.7%)
- Strong theoretical backing
- Used by STRAT Lab in their playbook

**Steps:**
1. Implement monthly bar classification
2. Detect monthly bullish patterns
3. Filter hourly 2-2 Up to only trade when monthly is bullish
4. Re-run validation
5. Assess pattern count reduction impact

**Expected Result:** R:R improvement to 2:1+, but significant pattern count reduction

### Option C: Combine Both (2D + Monthly Alignment)
**Rationale:**
- Maximum statistical edge
- Stacks multiple confluence factors
- Aligns with "real edge comes from stacking conditions"

**Steps:**
1. Implement both Strategy 1 and Strategy 2
2. Require both 2D bullish bias AND monthly alignment
3. Re-run validation
4. Accept reduced pattern count for higher quality

**Expected Result:** R:R improvement to 2:1+, highest conviction trades

---

## Integration with Current System

### Current Session 64 Status
**Validated Patterns:**
- Daily 2-1-2 Up: 2.65:1 R:R, 80% hit rate ✓ READY
- Hourly 2-2 Up: 1.53:1 R:R, 90.5% hit rate ✗ NOT READY

### How STRAT Lab Research Applies

**For Daily 2-1-2 Up (Already Validated):**
- Could potentially improve further with 2D hybrid analysis
- Consider adding monthly alignment for even higher conviction
- Current performance already exceeds target - optimization optional

**For Hourly 2-2 Up (Needs Improvement):**
- Apply Strategy 1 (2D hybrid) or Strategy 2 (monthly alignment)
- Goal: Achieve 2:1 R:R while maintaining 80%+ hit rate
- Research provides clear path: higher timeframe filters

### Implementation Priority
1. **Session 65 Focus:** Optimize Hourly 2-2 Up with 2D or monthly filters
2. **Session 66+:** Begin options module with validated patterns only
3. **Future:** Consider adding 3D analysis for additional edge

---

## Key Quotes for Context

**On Timeframe Selection:**
> "If trading is a probability game, why not play where odds are slightly skewed in your favour?"

**On Higher Timeframe Alignment:**
> "Check the quarterly first, then the monthly. If they're leaning H → 2u, 2dG → 2u, 3u → 2u, or have already gone 2u → 2u, treat pullbacks as pauses in accumulation rather than signs of reversal."

**On Hybrid Timeframes:**
> "Hybrid frames offer signals that are more reliable than their conventional neighbours."

**On Edge Discovery:**
> "Adding hybrid timeframes alongside your daily/weekly/monthly workflow could be the next evolution for Stratters looking to systematically upgrade their edge."

---

## Next Steps for Session 65

1. **Decision:** Choose Strategy 1 (2D), Strategy 2 (Monthly), or Strategy 3 (Both)
2. **Implementation:** Add timeframe alignment logic to backtest script
3. **Validation:** Re-run 3-stock validation with new filters
4. **Measurement:** Compare R:R ratios before/after optimization
5. **GO Decision:** If R:R reaches 2:1+, proceed to options module Phase 1

**Target:** Achieve 2:1 R:R on Hourly 2-2 Up pattern to enable dual-pattern options implementation (Daily 2-1-2 Up + Hourly 2-2 Up).
