# STRAT + Options Trading Strategy - Implementation Handoff Document

**Date:** November 6, 2025  
**Purpose:** Seamless context handoff for STRAT methodology implementation with options integration  
**Status:** Design Complete, Implementation Pending

---

## Executive Summary

We are implementing a systematic options trading strategy using Rob Smith's STRAT methodology with full timeframe continuity filtering. The strategy solves the capital efficiency problem ($3,000 minimum viable) while maintaining institutional-grade risk management through long calls/puts only.

**Key Innovation:** Using timeframe continuity (Monthly/Weekly/Daily/4H/1H alignment) to identify high-probability setups, then executing with options contracts to achieve capital efficiency at small scale.

---

## Capital Structure Decisions

### Minimum Viable Capital: $3,000
- **Position sizing:** Single contract at $300-500 premium = 10-17% of capital
- **Concurrent positions:** 2-3 maximum
- **Max deployed capital:** 40% at any time
- **Settlement:** T+1 (every-other-day rotation possible with single pool)

### Instrument Selection Strategy
**NOT trading sector ETFs directly.** Instead:
1. **Scan sector ETFs daily** (XLF, XLRE, XLP, etc.) for relative strength
2. **Drill down to constituent stocks** showing leadership within strong sectors
3. **Apply STRAT patterns** to individual stocks ($40-80 price range optimal)
4. **Execute with options** on the constituent stocks

**Why this works:**
- Better price points for $3k capital constraints
- More volatile instruments = better option leverage
- Sector filtering adds alpha layer (manual selection edge)
- "Relative strength within relative strength" approach

---

## STRAT Methodology - Core Understanding

### Bar Type Classification (The Foundation)

```
Scenario 1 (Inside Bar): Current bar entirely within previous bar range
- Signals: Consolidation, indecision, equilibrium
- Visual: High < Previous High AND Low > Previous Low

Scenario 2 (Directional Bar): Takes out one side of previous bar
- 2U (Two Up): Breaks previous high but not low
- 2D (Two Down): Breaks previous low but not high
- Signals: Trend continuation or reversal depending on context

Scenario 3 (Outside Bar): Takes out BOTH sides of previous bar
- Breaks previous high AND previous low
- Signals: Broadening formation, increased volatility
- Key for magnitude calculation
```

### Pattern Recognition

**Patterns are just sequences of bar types:**
- 2-1-2: Directional → Inside → Opposite Direction (Reversal)
- 3-1-2: Outside → Inside → Directional (Reversal with larger magnitude)
- 2-2: Directional → Same Direction (Continuation)
- 1-2-2: Inside → Directional → Opposite Direction (False breakout reversal)

**CRITICAL:** Bar classification depends on previous bar's high/low
- Must load context bars before backtest period
- Bar scenario is RELATIVE, not absolute

### Timeframe Continuity (The Filter)

**Full Timeframe Continuity = High Probability Setup**

For BULLISH continuity, ALL must be true:
- Monthly: 2U (current month broke previous month high)
- Weekly: 2U (current week broke previous week high)
- Daily: 2U (current day broke previous day high)
- 4H: 2U (optional, increases conviction)
- 1H: 2U (execution timeframe)

For BEARISH continuity, ALL must be 2D.

**Partial continuity (any Scenario 1 bars) = LOWER PROBABILITY**
- This is where traders "get chopped up"
- Inside bars on higher timeframes = avoid or reduce position size

### Magnitude Targets (The Exit)

**Magnitude = Previous bar's opposite extreme:**
- 3-1-2 Down: Target = Scenario 3 bar's LOW
- 3-1-2 Up: Target = Scenario 3 bar's HIGH
- 2-1-2 Down: Target = First 2U bar's LOW (tighter than 3-1-2)
- 2-1-2 Up: Target = First 2D bar's HIGH

**Key insight:** Scenario 3 patterns have LARGER magnitude targets (wider range)

### Entry/Exit Rules

**Entry:**
- Entry ONLY on break of inside bar in direction of pattern
- 3-1-2 Down: Enter when price breaks BELOW inside bar low
- 3-1-2 Up: Enter when price breaks ABOVE inside bar high

**Stop Loss:**
- Opposite side of inside bar
- 3-1-2 Down: Stop above inside bar high
- 3-1-2 Up: Stop below inside bar low

**Take Profit:**
- Magnitude 1: First target (scale out 50%?)
- Magnitude 2+: Let remainder run with trail
- Exit when higher timeframe continuity breaks

---

## Options Integration Strategy

### Why Options Solve the Capital Problem

**With $3,000 equity positions:**
- 60 shares × $50 = $3,000 notional
- $0.30 stop = $18 risk = 0.6% risk (below target)
- Winner at $0.90 = $54 profit = 1.8% account gain

**With $3,000 options positions:**
- 5 contracts × $300 premium = $1,500 deployed (50% of capital)
- Controls ~$25,000 notional equivalent
- Winner at 100% = $1,500 profit = 50% account gain
- **Capital efficiency = 27× better**

### The Critical Insight: Doubling Signals

**Traditional approach (shares only):**
- Wait for inside bars to form
- Wait for break
- Theta decay during consolidation kills you

**Options approach:**
- Can trade directional bars (2U/2D) directly when full continuity exists
- Don't need inside bar setups (though they're still valid)
- Avoid inside bar periods (consolidation = theta decay death)
- **Effectively doubles available signals**

### Contract Selection Framework

**Variables to optimize:**
1. DTE (Days to Expiration): 7, 14, 21, or 30 days?
2. Strike distance: ATM, 1 OTM, 2 OTM, 3 OTM?
3. IV Percentile: Does option expensiveness matter?

**The Math Problem:**

Example: 3-1-2 Down pattern
- Current price: $148
- Magnitude target: $145 (Scenario 3 low)
- Expected move: $3

**Strike Selection Decision:**
```
ATM Put ($148): Premium $5 → Breakeven $143 (needs $5 move, not $3)
1 OTM Put ($145): Premium $2.50 → Breakeven $142.50 (still needs $5.50)
2 OTM Put ($143): Premium $3 → Breakeven $140 (needs $8 move!)
```

**The Question:** Which strike optimizes expectancy given:
- Magnitude distance = $3
- Full timeframe continuity probability
- IV environment
- DTE selected

**Systematic Decision Logic (To Be Backtested):**

```python
IF full_timeframe_continuity:  # High conviction
    IF iv_percentile < 30:  # Options cheap
        → Buy ATM or 1 OTM (pay for delta, minimize theta)
    ELIF iv_percentile > 70:  # Options expensive
        → Buy 2-3 OTM (cheaper, let IV crush work for you)
    ELSE:  # Normal IV
        → Buy 1 OTM (balance)
        
ELIF partial_continuity:  # Lower conviction
    → Buy ATM only OR skip trade
```

---

## Implementation Roadmap

### Phase 1: Data Preparation & Bar Classification
**Objective:** Build the bar classification engine

**Requirements:**
- Historical OHLC data (Monthly, Weekly, Daily, 4H, 1H)
- Context bars: Start backtest at day 252, but load from day 1
- At least 36 months of data for proper context

**Deliverables:**
```python
def classify_bar(current_high, current_low, prev_high, prev_low):
    """Returns: 1, '2U', '2D', or 3"""
    
def get_bar_scenario(data, timeframe, index):
    """Returns bar scenario for any timeframe at any point"""
```

### Phase 2: Timeframe Continuity Engine
**Objective:** Identify when all timeframes align

**Deliverables:**
```python
def check_full_continuity(symbol, date, direction='bullish'):
    """
    Returns: True/False for full continuity
    Checks: Monthly, Weekly, Daily (4H/1H optional)
    """
    
def get_continuity_strength(symbol, date):
    """
    Returns: 0-5 (number of aligned timeframes)
    Useful for position sizing adjustments
    """
```

### Phase 3: Pattern Recognition
**Objective:** Identify specific STRAT patterns on daily timeframe

**Start with ONE pattern:** 3-1-2 (chosen for larger magnitude targets)

**Deliverables:**
```python
def identify_pattern(daily_data, index):
    """
    Returns: {
        'pattern': '3-1-2-down' or '3-1-2-up',
        'entry': price_at_inside_bar_break,
        'magnitude': target_price,
        'stop': stop_loss_price,
        'scenario_3_range': high - low  # For magnitude calculation
    }
    """
```

### Phase 4: Equity Backtest (Validate Pattern Edge)
**Objective:** Prove STRAT patterns reach magnitude targets

**Test Universe:**
- 50-100 stocks across sectors
- $40-80 price range (optimal for capital)
- Volume > 500K daily

**Success Criteria:**
- Magnitude hit rate > 60% with full continuity
- Risk-reward > 2:1
- Win rate validates options application

**Metrics to Track:**
```python
results = {
    'total_signals': int,
    'full_continuity_signals': int,
    'partial_continuity_signals': int,
    'magnitude_hit_rate_full': float,  # Key metric
    'magnitude_hit_rate_partial': float,
    'avg_days_to_magnitude': float,  # Critical for DTE selection
    'max_adverse_excursion': float,  # For stop loss validation
    'avg_profit_per_trade': float,
    'max_drawdown': float
}
```

### Phase 5: Options Simulation
**Objective:** Find optimal DTE and strike selection

**Once equity edge validated, test:**
- DTE options: [7, 14, 21, 30]
- Strike distances: [ATM, 1 OTM, 2 OTM, 3 OTM]
- All combinations = 16 scenarios per pattern

**For each scenario, calculate:**
```python
for signal in validated_signals:
    for dte in [7, 14, 21, 30]:
        for strike_offset in [0, 1, 2, 3]:
            # Simulate option trade
            premium = get_historical_option_price(...)
            
            if reached_magnitude_before_expiry:
                profit = intrinsic_value(magnitude) - premium
            else:
                profit = -premium  # Or partial value
            
            record_result(dte, strike_offset, profit)
```

**Optimization Output:**
- **Best DTE** (which expiration captures most magnitude moves)
- **Best strike distance** (optimal leverage vs probability)
- **IV impact** (does option expensiveness matter?)

### Phase 6: Build Production Scanner
**Objective:** Real-time pattern detection system

**Components:**
1. Daily sector scan (identify strong sectors via XLF, XLRE, etc.)
2. Constituent analysis (which stocks driving sector strength)
3. Timeframe continuity check (Monthly/Weekly/Daily/4H/1H)
4. Pattern recognition (3-1-2, 2-1-2, etc.)
5. Options chain analysis (optimal strike selection)
6. Alert system (notify when high-probability setup forms)

---

## Critical Questions Requiring Backtest Answers

### Question 1: Magnitude Hit Rate by Pattern
**Need to know:** 
- 3-1-2 with full continuity: X% reach magnitude
- 2-1-2 with full continuity: Y% reach magnitude
- How does partial continuity impact hit rate?

**Why it matters:** 
- High hit rate (>70%) = can use OTM strikes (cheaper, more leverage)
- Medium hit rate (50-70%) = ATM strikes (need delta)
- Low hit rate (<50%) = pattern doesn't work, try different one

### Question 2: Optimal DTE
**Need to know:**
- Average days to reach magnitude when full continuity exists
- Distribution: 50% within X days, 90% within Y days

**Why it matters:**
- If magnitude hits in 3-5 days → 7 DTE optimal
- If magnitude hits in 7-10 days → 14 DTE optimal
- Shorter DTE = cheaper premium but less time to be right

### Question 3: Strike Distance Sweet Spot
**Need to know:**
- Expected magnitude move size vs option premium breakeven

**Why it matters:**
- If magnitude move = $3, but ATM premium = $5, you LOSE even when right
- Need to find strike where: magnitude move > premium + buffer

### Question 4: IV Percentile Impact
**Need to know:**
- Does buying options when IV is cheap improve returns?
- Does IV crush after entry hurt or help?

**Why it matters:**
- High IV = expensive options, need larger move to profit
- Low IV = cheap options, but less volatility might mean slower magnitude reach

---

## Data Requirements

### For Equity Backtest:
- ✅ Already have: Alpaca historical data (OHLCV)
- ✅ Timeframes: Monthly, Weekly, Daily, 4H, 1H
- ✅ Can get: 2+ years history minimum

### For Options Backtest:
- ❌ Need: Historical options chain data
- Required fields:
  - Strike prices
  - Bid/Ask spreads
  - Option premium (mid-point)
  - Implied volatility
  - Greeks (Delta, Theta, Vega)
  - Open interest, volume

**Options Data Providers:**
1. **Polygon.io** (~$200/month) - Historical options data
2. **ThetaData** ($350-900/month) - Comprehensive options data
3. **CBOE DataShop** - One-time purchase for specific periods
4. **HistoricalOptionData.com** - Specific datasets

**Alternative:** Skip historical options data initially, use Black-Scholes to estimate premiums
- Less accurate but sufficient for initial validation
- Can refine with real data once concept proven

---

## Risk Management Framework

### Position Sizing Rules
```python
# Per trade
max_risk_per_trade = account_size * 0.15  # $450 on $3k account
max_position_size = $500 premium  # Single contract limit

# Portfolio level
max_deployed_capital = account_size * 0.40  # $1,200 max deployed
max_concurrent_positions = 3

# Risk checks
if (current_deployed + new_position_cost) > max_deployed_capital:
    reject_trade()
    
if len(open_positions) >= max_concurrent_positions:
    reject_trade()
```

### Exit Management
```python
# Scale out approach
if price_reaches_magnitude_1:
    close_50_percent_of_position()
    move_stop_to_breakeven()
    
if higher_timeframe_continuity_breaks:
    close_remaining_position()
    
if dte_remaining < 3:
    close_remaining_position()  # Avoid theta decay acceleration
    
# Max hold time
if days_in_trade > 14:
    evaluate_exit()  # Pattern didn't play out
```

### Portfolio Heat Monitoring
```python
total_risk = sum([position.max_loss for position in open_positions])
portfolio_heat = total_risk / account_size

if portfolio_heat > 0.06:  # 6% max exposure
    reject_new_trades()
    
# Options have defined risk = premium paid
# Unlike shares, can't lose more than initial investment
```

---

## VectorBT Pro Implementation Notes

### Required VectorBT Features
1. **Custom indicator support** - For bar classification
2. **Multi-timeframe analysis** - For continuity checks
3. **Options backtesting** - Data-agnostic, feed options contracts as columns
4. **Custom entry/exit logic** - For magnitude-based targets
5. **Portfolio simulation** - For position sizing and heat management

### Implementation Strategy

**Use VectorBT Pro's Portfolio engine:**
```python
import vectorbtpro as vbt

# Bar classification as custom indicator
@vbt.IF.reg_vbt_indicator('strat_bars')
class StratBars(vbt.IF.BaseIndicator):
    """Classifies bars as 1, 2U, 2D, or 3"""
    
# Timeframe continuity as signal generator
@vbt.IF.reg_vbt_indicator('timeframe_continuity')
class TimeframeContinuity(vbt.IF.BaseIndicator):
    """Checks alignment across Monthly/Weekly/Daily"""
    
# Pattern recognition
@vbt.IF.reg_vbt_indicator('strat_patterns')  
class StratPatterns(vbt.IF.BaseIndicator):
    """Identifies 3-1-2, 2-1-2, etc."""
```

**Critical:** Must verify each VectorBT Pro API call before implementation
- Use VectorBT Pro MCP server for documentation search
- Check examples in project knowledge
- Follow mandatory 5-step verification: SEARCH → VERIFY API → FIND EXAMPLES → TEST → IMPLEMENT

---

## Development Environment Setup

### Tools Stack
- **Primary IDE:** VS Code (production implementation)
- **Exploration IDE:** Cursor (rapid prototyping ONLY, never deploy directly)
- **Package Manager:** UV
- **Python:** 3.13.7
- **Core Libraries:**
  - VectorBT Pro 2025.10.15
  - TA-Lib (technical indicators)
  - Alpaca Algo Trader Plus (data + paper trading)
  
### MCP Server Integration
- VectorBT Pro MCP server (documentation access)
- Must be enabled in Claude desktop environment
- Not available on mobile

### Development Workflow
1. **Research phase:** Use Claude with VectorBT MCP access
2. **Design phase:** Outline implementation in artifacts
3. **Implementation:** Claude Code for production code
4. **Testing:** Rapid iteration in Cursor (exploratory only)
5. **Validation:** Comprehensive backtests in VectorBT Pro
6. **Paper Trading:** Alpaca paper account validation
7. **Live Trading:** Deploy to $3k live account

---

## Pattern Priority for Testing

### Phase 1: Start with 3-1-2 Pattern
**Why this pattern first:**
- Outside bar (Scenario 3) provides largest magnitude targets
- Clear entry signal (inside bar break)
- Larger expected move = better for options leverage
- Well-documented in STRAT literature

**Test on:**
- Daily timeframe (most common for STRAT day/swing traders)
- Require full continuity (Monthly/Weekly/Daily all 2U or 2D)
- 50-100 stock universe

### Phase 2: Add 2-1-2 Pattern
**If 3-1-2 validates:**
- Test 2-1-2 for comparison
- Smaller magnitude but potentially higher frequency
- May have better win rate (tighter targets)

### Phase 3: Explore 2-2 and 1-2-2
**Expansion patterns:**
- 2-2 continuation (trend following)
- 1-2-2 reversal (false breakout traps)

---

## Open Questions & Next Steps

### Questions for User
1. **Pattern priority:** Confirm 3-1-2 as starting point?
2. **Test universe:** Specific sectors or broad (Top 100 by volume)?
3. **Options data:** Budget for historical options data subscription?
4. **Timeline:** What's realistic timeframe for Phase 1-3 completion?

### Immediate Next Steps (Desktop Session)
1. ✅ Enable VectorBT Pro MCP server in Claude desktop
2. Search VectorBT Pro documentation for:
   - Multi-timeframe indicator examples
   - Custom signal generation patterns
   - Options backtesting approaches
3. Build bar classification indicator (test on SPY first)
4. Validate bar classification against known STRAT patterns
5. Build timeframe continuity checker
6. Test on single stock (e.g., AAPL) before expanding universe

---

## Critical Success Factors

### Must Have
- ✅ Bar classification 100% accurate (foundation of everything)
- ✅ Timeframe continuity check validated
- ✅ Pattern recognition tested against manual chart analysis
- ✅ Magnitude hit rate > 60% with full continuity
- ✅ Options simulation shows positive expectancy

### Should Have
- Multi-symbol validation (prevent overfitting)
- Walk-forward analysis (prove robustness)
- Monte Carlo simulation (understand variance)
- Transaction cost modeling (realistic execution)

### Nice to Have
- Real-time scanner for pattern detection
- Automated options chain analysis
- Alert system for high-probability setups
- Performance dashboard

---

## Key Insights from Discussion

1. **Sector rotation + stock selection > trading sectors directly**
   - Use sector ETFs as filters, trade constituents

2. **Options solve capital efficiency, not leverage**
   - We're not "gambling" with options
   - We're achieving proper position sizing at small scale

3. **Timeframe continuity = the edge**
   - Not the patterns themselves
   - Alignment across timeframes = institutional participation

4. **Don't wait for inside bars with options**
   - Inside bars = consolidation = theta decay
   - Can trade directional bars when full continuity exists
   - This doubles available signals

5. **Magnitude targets are objective**
   - Not discretionary "take profit when feels right"
   - Scenario 3 low/high = clear price target
   - This makes options strike selection systematic

6. **Start simple, expand systematically**
   - One pattern (3-1-2)
   - One timeframe (daily)
   - Validate edge
   - Then expand

---

## Handoff Checklist for Next Session

When resuming implementation, verify:
- [ ] VectorBT Pro MCP server is enabled
- [ ] Can access project knowledge files
- [ ] Have read this handoff document completely
- [ ] Understand STRAT bar classification logic
- [ ] Understand timeframe continuity requirement
- [ ] Understand magnitude target calculation
- [ ] Know which pattern we're testing first (3-1-2)
- [ ] Ready to write actual VectorBT Pro code

---

## Document Version Control
- **Version:** 1.0
- **Created:** 2025-11-06
- **Last Updated:** 2025-11-06
- **Next Review:** After Phase 1 completion (bar classification validated)

---

## Contact Context
- **User:** Chris (ADHD, operates quantitative trading firm)
- **Background:** Discretionary STRAT trader, systematizing approach
- **Constraints:** $3,000 capital challenge, cash account, Options Level 2
- **Goal:** Build systematic options strategy using STRAT + timeframe continuity

**User's Edge:** Years of discretionary STRAT trading experience, knows patterns work, needs systematic implementation to scale and remove emotional execution barriers.

---

*End of Handoff Document*
