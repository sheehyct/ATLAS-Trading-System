# Credit Spread Leveraged ETF Strategy - Complete Rules

**Strategy Name**: "Tier's Credit Spreads Leverage ETF Strategy"

**Source**: Transcript from video about leveraged ETF trading strategy using credit spreads

---

## 1. DATA REQUIREMENTS

### Credit Spread Indicator
- **Ticker**: ICE BofA US High Yield Option-Adjusted Spread
- **FRED Code**: BAMLH0A0HYM2
- **Data Available**: From 1996 onwards
- **Update Frequency**: Daily (slightly lagged for retail, but sufficient per the video)

### Trading Instrument
**Preferred (in order of preference):**
1. **SPXL** - 3x leveraged S&P 500 ETF (data from 2008+)
2. **UPRO** - 3x leveraged S&P 500 ETF (alternative to SPXL)
3. **SSO** - 2x leveraged S&P 500 ETF (data from 2006+, used in video backtest)

**Benchmark:**
- SPY or SPX for performance comparison

---

## 2. CORE STRATEGY RULES

### Entry Signal (BUY Signal)
**Condition**: Credit spreads fall **35% from their recent highs**

**Execution**:
- Enter with **100% of portfolio** into the leveraged ETF
- No scaling in - full position immediately
- Use next available trading day if signal occurs on weekend/holiday

**Reasoning** (from transcript):
> "Now why are you going a hundred percent and not scaling in well one for simplicity and two because you've identified that this is now about to be a positive trending market so there's no point leaving some of your cash and cash reserve"

---

### Exit Signal (SELL Signal)
**Conditions** (BOTH must be met):
1. Credit spreads rise **40% from their recent lows**, AND
2. Credit spreads **cross above the 330-day exponential moving average**

**Execution**:
- Exit **100% of position** - sell entirely
- Move to **100% cash**
- Use next available trading day if signal occurs on weekend/holiday

**Quote from transcript**:
> "when credit spreads rise 40% from their recent lows and cross above the 330 day exponential moving average that's a sign that you want to get out"

---

## 3. TECHNICAL SPECIFICATIONS

### The 330-Day EMA
- Applied to the credit spread data (not the price data)
- Standard exponential moving average calculation
- Must cross above this line for exit signal (in addition to 40% rise condition)

### "Recent Highs" and "Recent Lows" Definition
This is **the most discretionary part** of the strategy. From the transcript:

**Recent Low Definition**:
- Used for calculating the 40% rise for exit signals
- Appears to be the lowest point since the last entry signal
- Presenter mentions "more than six months before so potentially that's not the recent low"
- Suggests a rolling lookback of ~6 months, but somewhat subjective

**Recent High Definition**:
- Used for calculating the 35% fall for entry signals
- Appears to be the highest point since the last exit signal
- May consider multiple local highs during the out-of-market period

**Implementation Note**: This discretionary element may require testing multiple approaches:
- Option A: Rolling window (e.g., 180-day lookback)
- Option B: Highest/lowest since last signal change
- Option C: Peak detection algorithm (scipy.signal.find_peaks)

---

## 4. POSITION SIZING

- **When IN**: 100% allocated to leveraged ETF
- **When OUT**: 100% in cash (earning 0% return in backtest)
- **No partial positions**: Binary on/off strategy
- **No leverage beyond the ETF**: Do not use margin on top of SPXL/SSO/UPRO

---

## 5. HISTORICAL SIGNAL DATES (from video walkthrough)

These are the exact dates mentioned in the video - useful for validation:

| Date | Action | Credit Spread Level | Notes |
|------|--------|---------------------|-------|
| Aug 18, 1998 | EXIT | Up 40% from recent low | Before dot-com crash |
| Apr 3, 2003 | ENTRY | Down 35% from recent high | After dot-com bear market |
| Apr 14, 2005 | EXIT | Up 40% + crossed 330 EMA | |
| May 4, 2006 | ENTRY | Down 35% from recent high | |
| Jul 19, 2007 | EXIT | Up 40% + crossed 330 EMA | **Before 2008 crash!** |
| Apr 30, 2009 | ENTRY | Down 35% from recent high | After 2008 crash bottom |
| Aug 4, 2011 | EXIT | Up 40% + crossed 330 EMA | |
| Mar 13, 2012 | ENTRY | Down 35% from recent high | |
| Oct 9, 2014 | EXIT | Up 40% + crossed 330 EMA | |
| Jul 12, 2016 | ENTRY | Down 35% from recent high | |
| Dec 5, 2018 | EXIT | Up 40% + crossed 330 EMA | |
| Dec 13, 2019 | ENTRY | Down 35% from recent high | |
| Feb 26, 2020 | EXIT | Up 40% + crossed 330 EMA | **Before COVID crash!** (3% loss) |
| May 21, 2020 | ENTRY | Down 35% from recent high | After COVID bottom |
| Mar 14, 2022 | EXIT | Up 40% + crossed 330 EMA | |
| Jul 15, 2023 | ENTRY | Down 35% from recent high | Still in as of video |

---

## 6. PERFORMANCE CLAIMS (from video)

### Using SSO (2x leveraged) from July 2007 to 2024:
- **Starting Capital**: £10,000
- **Ending Capital**: £163,651
- **Total Return**: 16.3x (1,536% gain)
- **SPX Return (same period)**: 3.8x (280% gain)
- **Note**: With 3x leverage (SPXL/UPRO), estimated ~25x return

### Individual Trade Results (SSO):
1. Apr 30, 2009 @ $2.97 → Aug 4, 2011 @ $5.43 = **+82%**
2. Mar 13, 2012 @ $7.17 → Oct 9, 2014 @ $13.74 = **+92%**
3. Jul 12, 2016 @ $17.45 → Dec 5, 2018 @ $27.07 = **+55%**
4. Dec 13, 2019 @ $36.36 → Feb 26, 2020 @ $35.03 = **-3%** (avoided COVID crash)
5. May 21, 2020 @ $28.54 → Mar 14, 2022 @ $55.75 = **+95%**
6. Jul 15, 2023 @ $59.97 → (still in) @ $96.21 = **+60%**

---

## 7. RATIONALE & EDGE CASES

### Why This Works
**From transcript**:
> "Credit spreads are forward-looking instruments which means that they're trying to sniff out something before it happens and what it's trying to sniff out essentially is signs of stress on the economy or on the market before they actually rear their ugly head"

**Key Insight**:
- Low/tight credit spreads = Low risk, positive trending markets
- High/wide credit spreads = High stress, choppy/bear markets
- The strategy catches **positive trends** with leverage while **avoiding drawdowns**

### Volatility Decay Issue (addressed by strategy)
- Leveraged ETFs suffer from volatility decay in choppy markets
- This strategy specifically avoids choppy/bearish periods
- Only deploys leverage during confirmed uptrends (per credit spread signals)

### Edge Cases Mentioned:

1. **False Signal (1 instance noted)**:
   - One period showed credit spreads signaled exit, but market continued trending up
   - Presenter: "it kind of made a mistake here but it's no big deal over the course of a long term"

2. **Small Loss Example**:
   - Dec 2019 - Feb 2020: -3% loss, but avoided the entire COVID crash
   - Shows the strategy prioritizes capital preservation

3. **Long Out-of-Market Periods**:
   - Sometimes out of market for 1-2 years
   - Example: 2007-2009 (avoided entire 2008 crash)
   - Example: 1998-2003 (avoided dot-com crash)

---

## 8. IMPLEMENTATION CONSIDERATIONS

### Data Alignment
- Credit spread data updates daily (but slightly lagged)
- Use end-of-day pricing for signals
- Assume execution at next day's open (or same day close with conservative approach)

### Transaction Costs
- Video mentions "fees associated with the ETF" eating into returns slightly
- Consider:
  - ETF expense ratios (SPXL: ~0.95%, SSO: ~0.90%)
  - Trading commissions (can assume $0 for modern brokers)
  - Slippage (minimal for highly liquid ETFs)

### Starting Period
- **Optimal**: Start from 2006 (SSO data available, includes 2008 test)
- **Alternative**: Start from 2009 (SPXL data available)
- **Historical validation**: Can validate signals back to 1998 using credit spread data

---

## 9. BACKTEST VALIDATION REQUIREMENTS

To ensure accuracy, the backtest should:

1. **Replicate Historical Signals**: Match the dates in Section 5
2. **Replicate Trade Returns**: Match the SSO trade returns in Section 6
3. **Replicate Final Performance**: Achieve ~16.3x from £10,000 starting in July 2007 (SSO)
4. **Handle Edge Cases**: Properly identify the -3% loss before COVID crash
5. **Handle Missing Data**: Account for weekends/holidays gracefully

---

## 10. OPEN QUESTIONS FOR IMPLEMENTATION

### Critical Questions:
1. **Local high/low calculation**: What exact algorithm should we use?
   - Rolling 180-day window?
   - Highest/lowest since last signal?
   - Peak detection with specific parameters?

2. **Signal timing**: Same day or next day execution?
   - Video suggests signals trigger on close, execution next day

3. **Both exit conditions**: Do they need to trigger on the same day?
   - Or can 40% rise happen first, then EMA cross later?
   - Video examples suggest simultaneous or very close together

4. **EMA calculation start date**: Should we have a warm-up period for the 330-day EMA?
   - Need at least 330 days of credit spread data before first signal

### Questions for User:
- Should we use conservative (next-day) or aggressive (same-day) execution?
- Should we include cash returns while out of market (T-bills/money market)?
- What slippage/commission assumptions to use?

---

## 11. NEXT STEPS

1. Download ICE BofA credit spread data (BAMLH0A0HYM2)
2. Download SSO, SPXL, UPRO, and SPY price data
3. Implement signal generation logic (multiple approaches for local high/low)
4. Validate against historical dates from video
5. Run full backtest with VectorBT Pro
6. Compare results to video claims
7. Generate performance metrics and drawdown analysis
8. Create walk-forward validation if strategy is robust
