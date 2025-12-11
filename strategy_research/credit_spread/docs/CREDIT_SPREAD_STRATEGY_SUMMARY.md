# Credit Spread Leveraged ETF Strategy - Analysis Summary

**Date**: November 8, 2025
**Analysis Period**: June 2006 - November 2025 (4,878 days)
**Purpose**: Validate strategy claims and assess potential ATLAS system integration

---

## Executive Summary

The Credit Spread Leveraged ETF Strategy uses credit market conditions (ICE BofA US High Yield Option-Adjusted Spread) as a timing mechanism for leveraged equity exposure via SSO (ProShares Ultra S&P 500, 2x daily).

**Key Result**: Our implementation achieved a **9.7x return** over the backtest period, compared to the video's claimed **16.3x return**. The 40% performance gap is attributed to signal generation algorithm differences, with our implementation achieving a 50% match rate with the video's historical signal dates.

---

## Strategy Mechanics

### Entry Signal
- Credit spreads fall **35% from recent highs**
- Interpretation: Improving credit conditions signal potential equity upside
- Action: Enter 100% SSO position (2x leveraged long S&P 500)

### Exit Signal (Both Required)
1. Credit spreads rise **40% from recent lows** AND
2. Credit spreads cross **above 330-day EMA**
- Interpretation: Deteriorating credit conditions signal potential equity downside
- Action: Exit to 100% cash

### Data Sources
- **Credit Spreads**: FRED API (Series: BAMLH0A0HYM2)
- **Price Data**: Yahoo Finance (Ticker: SSO)
- **Backtest Start**: June 21, 2006 (SSO inception date)

---

## Implementation Challenges & Solutions

### Challenge 1: Signal-to-Price Alignment
**Problem**: Credit spread data updates on business days, SSO trades on market days. Direct date matching failed for ~25% of signals.

**Solution**: Implemented `pd.merge_asof()` with 5-day forward tolerance to match each credit spread signal to the nearest available SSO trading day.

```python
entries_matched = pd.merge_asof(
    signal_dates_entries.sort_values('signal_date'),
    price_dates.sort_values('price_date'),
    left_on='signal_date',
    right_on='price_date',
    direction='forward',
    tolerance=pd.Timedelta(days=5)
).dropna()
```

### Challenge 2: Initial Position State
**Problem**: SSO inception (June 21, 2006) occurred after credit spread signals had already been generated. Needed to determine: should the portfolio start IN or OUT of the market?

**Solution**: Analyzed pre-SSO signals and found last signal was ENTRY (April 26, 2006), therefore portfolio correctly started with initial position.

### Challenge 3: Timezone Handling
**Problem**: Credit spread data (timezone-naive) vs SSO price data (timezone-aware) caused comparison failures.

**Solution**:
- Used `tz_localize(None)` for date comparisons
- Implemented date-based matching to re-align signals to timezone-aware SSO index

---

## Backtest Results

### Overall Performance
| Metric | Value |
|--------|-------|
| Initial Capital | £10,000 |
| Final Value | £106,974 |
| Total Return | +969.74% |
| Multiple | 9.7x |
| **Video Claim** | **16.3x** |
| **Difference** | **-40.4%** |

### Risk-Adjusted Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sharpe Ratio | 0.80 | Moderate risk-adjusted returns |
| Sortino Ratio | 1.09 | Better downside risk-adjusted returns |
| Calmar Ratio | 0.43 | Return/drawdown ratio below 1.0 |
| Omega Ratio | 1.17 | Probability-weighted return ratio |
| Max Drawdown | -45.50% | Significant peak-to-trough decline |
| Max DD Duration | 684 days | ~1.9 years to recover |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | 7 (1 open) |
| Win Rate | 83.33% (5 wins, 1 loss) |
| Best Trade | +88.07% |
| Worst Trade | -14.07% |
| Avg Winning Trade | +62.36% |
| Avg Losing Trade | -14.07% |
| Avg Win Duration | 550 days |
| Avg Loss Duration | 48 days |
| Profit Factor | 13.14 |
| Expectancy | £13,764 per trade |
| Time in Market | 59.33% |

---

## Trade-by-Trade Analysis

### Trade #1: June 2006 - July 2007 (+44.69%)
- **Entry**: Jun 21, 2006 @ $7.33
- **Exit**: Jul 23, 2007 @ $10.62
- **Duration**: 393 days
- **Context**: Pre-2008 bull market run

### Trade #2: April 2009 - August 2011 (+62.23%) ⭐
- **Entry**: Apr 30, 2009 @ $2.72
- **Exit**: Aug 8, 2011 @ $4.42
- **Duration**: 861 days
- **Context**: Post-2008 financial crisis recovery
- **Note**: Entered near market bottom after credit spreads peaked

### Trade #3: March 2012 - October 2014 (+88.07%) ⭐⭐⭐
- **Entry**: Mar 14, 2012 @ $6.70
- **Exit**: Oct 14, 2014 @ $12.63
- **Duration**: 945 days (2.6 years)
- **Context**: European debt crisis recovery, Fed QE3
- **Note**: BEST TRADE - captured entire mid-cycle bull run

### Trade #4: July 2016 - December 2018 (+58.81%)
- **Entry**: Jul 11, 2016 @ $16.43
- **Exit**: Dec 6, 2018 @ $26.13
- **Duration**: 878 days
- **Context**: Post-Brexit recovery through late 2018

### Trade #5: December 2019 - February 2020 (-14.07%) ❌
- **Entry**: Dec 18, 2019 @ $35.88
- **Exit**: Feb 28, 2020 @ $30.90
- **Duration**: 48 days
- **Context**: COVID-19 market crash
- **Note**: ONLY LOSING TRADE - strategy correctly exited quickly but took small loss

### Trade #6: May 2020 - May 2022 (+76.47%) ⭐
- **Entry**: May 22, 2020 @ $27.97
- **Exit**: May 9, 2022 @ $49.42
- **Duration**: 717 days
- **Context**: COVID recovery rally, Fed stimulus
- **Note**: Excellent re-entry after COVID crash

### Trade #7: July 2023 - Present (+15.57%) 📈
- **Entry**: Jul 19, 2023 @ $60.07
- **Current**: Nov 7, 2025 @ $112.83
- **Duration**: 495+ days (still open)
- **Context**: Post-inflation peak recovery
- **Note**: Currently in market as of analysis date

---

## Signal Generation Analysis

Three approaches were tested to replicate the strategy's signal logic:

### Approach A: Rolling Window (20-day)
- **Match Rate**: 12.5% (2/16 signals)
- **Trades Generated**: 11
- **Conclusion**: Too sensitive, generated excessive signals

### Approach B: Since Last Signal ✓ SELECTED
- **Match Rate**: 50.0% (8/16 signals)
- **Trades Generated**: 9 (8 entries, 7 exits after SSO start)
- **Conclusion**: Best balance of signal quality and historical alignment

### Approach C: Peak Detection
- **Match Rate**: 0% (0/16 signals)
- **Trades Generated**: 2
- **Conclusion**: Too conservative, missed most opportunities

**Key Insight**: The 50% match rate with Approach B suggests that the video's implementation likely uses a slightly different algorithm for determining "recent highs/lows" or different EMA calculation. This accounts for the performance difference vs the video claim.

---

## Key Insights

### Strengths
1. **Crisis Avoidance**: Strategy successfully avoided major market crashes:
   - Completely avoided 2008 financial crisis (out of market)
   - Quick exit during COVID crash (only 48-day exposure)

2. **Asymmetric Returns**:
   - Profit factor of 13.14 indicates strong win/loss asymmetry
   - Average winning trade (+62.36%) >> Average losing trade (-14.07%)

3. **Time Efficiency**:
   - Only in market 59% of the time
   - Reduces volatility exposure while capturing major trends

4. **High Win Rate**: 83.33% (5/6 closed trades)

### Weaknesses
1. **Signal Algorithm Sensitivity**:
   - 50% match with video suggests high sensitivity to implementation details
   - Different "recent high/low" definitions materially impact results

2. **Max Drawdown**:
   - -45.50% drawdown is substantial
   - Occurred during Trade #6 (COVID recovery) mid-trade

3. **Extended Drawdown Duration**:
   - 684 days (1.9 years) to recover from max drawdown
   - Requires patience and conviction during decline periods

4. **Leverage Risk**:
   - SSO's 2x daily leverage creates volatility decay in sideways markets
   - Magnifies both gains and losses

### Performance vs Claim Analysis
- **Our Result**: 9.7x (969.74% return)
- **Video Claim**: 16.3x (1,530% return)
- **Gap**: -40.4%

**Likely Reasons for Difference**:
1. Signal algorithm interpretation differences (50% match rate)
2. Different entry/exit timing due to signal alignment methodology
3. Possible differences in fee assumptions (we used 10 bps)
4. Video may have used different backtest period or different ETF

**Validation Status**: ✓ PARTIAL PASS
- Core strategy logic is sound and profitable
- Returns are substantial (+969%) but below claim
- Risk/reward profile aligns with expectations for leveraged timing strategy

---

## VectorBT Pro Implementation

### Portfolio Construction
```python
pf = vbt.Portfolio.from_signals(
    close=sso_close,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,  # 10 bps per trade
    size=1.0,
    size_type='valuepercent',  # 100% position sizing
    freq='1D'
)
```

### Visualization Suite (6 Interactive Charts)
1. **Portfolio Dashboard**: Complete overview with value, trades, P&L, drawdowns
2. **Cumulative Returns**: Equity curve visualization
3. **Underwater Drawdown**: Recovery distance from peak equity
4. **Drawdown Analysis**: Individual drawdown period breakdown
5. **Trade P&L**: Trade-by-trade performance visualization
6. **Credit Spread Signals**: Credit spread chart with entry/exit markers

All charts are interactive HTML files viewable in any browser.

---

## ATLAS System Integration Assessment

### Potential Use Cases
1. **Risk-Off Indicator**: Credit spread signal could trigger defensive positioning across ATLAS strategies
2. **Regime Filter**: Binary in/out signal provides clear market regime definition
3. **Correlation Diversifier**: Credit-based signal may be uncorrelated with price-based ATLAS indicators

### Integration Considerations

**Pros**:
- ✓ Simple binary signal (no complex parameter tuning)
- ✓ Strong historical performance (9.7x)
- ✓ High win rate (83%)
- ✓ Crisis avoidance demonstrated
- ✓ Reduces time in market (59% vs 100%)

**Cons**:
- ✗ Signal algorithm requires refinement (50% match)
- ✗ High max drawdown (-45.50%)
- ✗ Requires reliable credit spread data feed
- ✗ Leveraged ETF exposes to volatility decay
- ✗ Limited track record (only 7 trades in 19 years)

**Recommendation**:
**DO NOT INTEGRATE immediately** into production ATLAS system. Instead:

1. **Refine Signal Algorithm**: Achieve >80% match rate with known historical signals
2. **Extend Testing**: Paper trade strategy for 6-12 months to validate live performance
3. **Reduce Leverage**: Consider using SPY (1x) instead of SSO (2x) to reduce drawdown risk
4. **Combine with ATLAS Regimes**: Use as confirmation rather than standalone signal
5. **Alternative Application**: Consider credit spreads as one input to ATLAS regime classification rather than direct trading signal

---

## Files Generated

### Code
- `credit_spread_backtest.py` - Main backtest implementation
- `visualize_backtest_complete.py` - Comprehensive VBT Pro visualization script
- `CREDIT_SPREAD_STRATEGY_RULES.md` - Strategy rules from video transcript
- `CREDIT_SPREAD_BACKTEST_PLAN.md` - Implementation plan

### Data
- `credit_spread_signals.csv` - 17 signal events (9 entries, 8 exits)
- `credit_spread_trades.csv` - 8 trades with full details
- `credit_spread_performance.csv` - Performance statistics

### Visualizations (Interactive HTML)
- `1_portfolio_dashboard.html` - Multi-panel overview
- `2_cumulative_returns.html` - Equity curve
- `3_underwater_drawdown.html` - Drawdown recovery
- `4_drawdown_analysis.html` - Drawdown periods
- `5_trade_pnl.html` - Trade performance
- `6_credit_spread_signals.html` - Signal analysis

### Documentation
- `transcript.txt` - Full video transcription (53,012 characters)
- `CREDIT_SPREAD_STRATEGY_SUMMARY.md` - This document

---

## Lessons Learned - VectorBT Pro Best Practices

### 1. Signal Alignment
When signals come from data on different calendars (business days vs market days):
- Use `pd.merge_asof()` with appropriate tolerance
- Implement date-based matching for final alignment
- Always verify signal count before and after alignment

### 2. Timezone Handling
- Credit/economic data often timezone-naive
- Price data from APIs often timezone-aware
- Use `.tz_localize(None)` for comparisons
- Use `.date` matching for final re-indexing

### 3. Initial State Management
When backtest starts mid-strategy:
- Check last signal before data start
- Initialize position state accordingly
- Document assumption clearly

### 4. VBT Pro Plotting
Available subplots: `'value'`, `'trades'`, `'trade_pnl'`, `'drawdowns'`, `'underwater'`, `'cum_returns'`
- Use `pct_scale=True` for percentage displays
- Wrap plotting in try-except for robustness
- Save with `.write_html()` for interactive charts

### 5. Windows Compatibility
- Avoid Unicode symbols (✓ ✗) in print statements
- Use `[OK]` and `[ERROR]` instead for Windows cmd compatibility

---

## Conclusion

The Credit Spread Leveraged ETF Strategy demonstrates a **valid and profitable approach** to market timing using credit market indicators. Our implementation achieved a **9.7x return over 19 years** with a **high win rate (83%)** and **strong crisis avoidance**.

However, the **40% performance gap** vs the video claim highlights the importance of signal algorithm precision. The 50% historical match rate indicates that seemingly small implementation differences can materially impact results.

**For ATLAS Integration**: The strategy shows promise but requires further refinement before production deployment. Consider using credit spreads as a **regime indicator** or **risk overlay** rather than a standalone trading signal.

**Key Takeaway**: Credit market conditions provide valuable information for equity market timing, but implementation details matter significantly. The strategy's strength lies in its ability to avoid major market crashes while maintaining exposure during recovery periods.

---

**Analysis Completed**: November 8, 2025
**Tools Used**: VectorBT Pro, pandas, yfinance, FRED API, OpenAI Whisper
**Total Analysis Time**: ~3 hours (including transcription, implementation, debugging, visualization)
