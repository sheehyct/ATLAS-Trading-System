# Strategy Overview: Momentum-Mean Reversion System

## Executive Summary

This system combines multi-timeframe momentum analysis with short-term mean reversion timing to generate swing trading signals for equities. The strategy aims for 2-week hold periods with 55-65% win rates and Sharpe ratios of 0.8-1.5.

**Core Principle:** Buy short-term dips in confirmed multi-timeframe uptrends.

## Two-Branch Approach

### Branch 1: TFC Confidence Score (Experimental)
- **Innovation:** Uses existing STRAT TFC (Time Frame Continuity) as multi-timeframe momentum filter
- **Signal:** Multi-factor confidence scoring (TFC + RSI + MACD + Volume)
- **Goal:** Test if TFC outperforms simple moving average as trend filter

### Branch 2: Baseline MA+RSI (Control)
- **Proven:** Research-documented strategy with 75% win rates over 30 years
- **Signal:** Price > 200-day MA + RSI(2) < 15
- **Goal:** Establish benchmark performance for comparison

## Research Foundation

### Academic Validation (From Algorithmic Trading Research Document)

**Combined Momentum-Mean Reversion:**
> "Combined momentum-mean reversion strategies consistently outperform single-approach systems across intraday, swing, and longer timeframes, with documented **Sharpe ratios of 1.0-2.5**"

**Connors RSI(2) Strategy:**
> "Achieved **75% win rates** with 0.5-0.66% gains per trade over 293 trades since 1993"

**MACD + RSI + Filter:**
> "**73% win rate** over 235 trades with 0.88% average gain per trade"

**Time Horizon Findings:**
> "Momentum dominates at **6-12 month horizons** while mean reversion performs best at **short horizons (days)** and very long horizons (3-5 years)"

**Risk Management:**
> "The Turtle Trading risk management approach of **allocating 2% per trade based on ATR with 2 ATR-based stop losses** provides a proven framework"

## Strategy Components

### 1. Momentum Filter (Long-Term Trend)

**Branch 1:** TFC Score
- Calculates alignment across hourly, daily, weekly timeframes
- Score 0-100% indicating trend strength
- Bullish filter: TFC > 70%
- Bearish filter: TFC < 30%

**Branch 2:** 200-Day Moving Average
- Simple, proven trend filter
- Bullish filter: Price > 200-day MA
- Bearish filter: Price < 200-day MA

**Purpose:** Prevents counter-trend trading (buying dips in bear markets)

### 2. Mean Reversion Signal (Entry Timing)

**Both Branches:** RSI(2) Oversold/Overbought
- RSI period: 2 days (NOT the standard 14-day)
- Long entry: RSI(2) < 15 (extreme short-term oversold)
- Short entry: RSI(2) > 85 (extreme short-term overbought)
- Research validation: 75% win rate documented

**Why RSI(2) instead of RSI(14):**
> "RSI strategies work best with **short periods (2-6 days)** rather than standard 14-day settings for mean reversion applications"

**Purpose:** Times entry at temporary weakness within established trend

### 3. Confidence Scoring (Branch 1 Only)

**Multi-factor weighted score (0-100):**
```python
confidence = (
    tfc_score * 0.30 +        # Multi-timeframe trend alignment
    rsi_score * 0.40 +        # Mean reversion strength
    macd_score * 0.20 +       # Momentum confirmation
    volume_score * 0.10       # Volume surge confirmation
)
```

**Entry thresholds:**
- High confidence (70+): Full position size (2% risk)
- Medium confidence (50-69): Half position size (1% risk)
- Low confidence (<50): No trade

**Optimization:** Weights optimized via walk-forward analysis

### 4. Exit Rules

**Profit Targets:**
- Primary: 2:1 reward-risk ratio (2 × ATR gain)
- Alternative: RSI(2) > 85 (overbought, mean reversion complete)

**Stop Loss:**
- 2 × ATR below entry price (Turtle Trading standard)
- Adjusts to volatility (wider in volatile markets, tighter in calm)

**Time-based:**
- Max hold: 14 days (2-week swing trade target)
- If no profit target or stop hit by day 14, exit at market

**Research validation:**
> "For short-term swing trading (days to weeks)... Expected Sharpe ratios range from **1.0-2.0** for retail implementations with 50-65% win rates"

## Risk Management

### Position Sizing
- **Risk per trade:** 2% of account equity
- **Calculation:** Position_size = (Account × 0.02) / (Entry - Stop_price)
- **Max open risk:** 10% of account (max 5 concurrent positions)

### Portfolio Rules
- **Max positions:** 5 simultaneous trades
- **Diversification:** No more than 2 positions in same sector
- **Correlation limit:** No positions in highly correlated stocks (>0.7)

### Account Size Requirements
- **Minimum:** $5,000 (allows proper position sizing)
- **Recommended:** $10,000+ (better diversification)

## Market Selection

### Asset Class
- **Primary:** US equities (NYSE, NASDAQ)
- **Future:** Cryptocurrency (one coin only, likely BTC or ETH)
- **Future:** Equity options (long calls/puts only)

### Stock Criteria
- **Liquidity:** Average volume > 1M shares/day
- **Price:** $10-500 per share (avoid penny stocks, extreme high prices)
- **Volatility:** ATR suitable for swing trading (not ultra-low, not ultra-high)
- **Market cap:** > $1B (avoid micro-caps with poor data quality)

## Performance Expectations

### Realistic Targets (Year 1)

**Win Rate:**
- Target: 55-65%
- Excellent: 65-75%
- Baseline threshold: >50% (if below, strategy doesn't work)

**Sharpe Ratio:**
- Target: 0.8-1.2
- Excellent: 1.2-1.8
- Baseline threshold: >0.5 (if below, risk-adjusted returns inadequate)

**Annual Return:**
- Target: 12-20% (after fees)
- Excellent: 20-30%
- Baseline: Must beat S&P 500 (~10% average)

**Max Drawdown:**
- Target: 15-25%
- Acceptable: 25-35%
- Failure threshold: >40% (position sizing broken)

**Research reality check:**
> "For retail algorithmic traders, **Sharpe ratios above 1.0 represent excellent performance**, while institutional traders typically target 1.5-2.5 as sustainable"

> "**Reported Sharpe ratios should be discounted by approximately 50%** due to data mining bias"

### Failure Criteria (Stop Development)
- Sharpe ratio < 0.3 after 6 months paper trading
- Win rate < 45% after 100+ trades
- Max drawdown > 40% at any point
- Strategy degrades over time (Sharpe declining month-over-month)

## Testing Protocol

### Phase 1: Backtesting (Months 1-2)
- Test period: 5+ years of historical data
- Include: 2008 crisis, 2020 COVID crash, bull markets, bear markets
- Walk-forward analysis: 1-year training, 3-month testing windows
- Conservative assumptions: 0.2% fees per trade, 0.1% slippage

### Phase 2: Paper Trading (Months 3-8)
- **Account 1 (Conservative):** $10k paper, 1% risk, confidence >75
- **Account 2 (Moderate):** $25k paper, 2% risk, confidence >60
- **Account 3 (Aggressive):** $50k paper, 2% risk, confidence >50
- Duration: Minimum 6 months (100+ trades)
- Weekly review: Track Sharpe, win rate, drawdown

### Phase 3: Live Trading (Month 9+)
- Start size: $1,000-5,000 (NOT full account)
- Scale up: Increase 20% per quarter if profitable
- Monitor: Compare live vs paper performance (should be similar)

## Technology Stack

### Core Infrastructure
- **Language:** Python 3.12+
- **Backtesting:** VectorBT Pro ($25/month)
- **Data:** Alpaca Algo Trader Plus (~$100/month)
- **Environment:** UV package manager

### Key Libraries
- `vectorbtpro` - Backtesting engine
- `alpaca-py` - Market data and trading API
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pandas-market-calendars` - Market hours filtering

### Existing Codebase
- `core/analyzer.py` - TFC calculation, bar classification
- `data/mtf_manager.py` - Multi-timeframe data alignment
- `data/alpaca.py` - Market data fetching
- `trading/strat_signals.py` - Signal generation (to be refactored)

## Documentation Structure

```
docs/
├── STRATEGY_OVERVIEW.md        (This file - high-level summary)
├── IMPLEMENTATION_PLAN.md      (Step-by-step build guide)
├── BRANCH_COMPARISON.md        (Branch 1 vs Branch 2 analysis)
├── VALIDATION_PROTOCOL.md      (Testing methodology)
├── PAPER_TRADING_GUIDE.md      (Using 3 Alpaca paper accounts)
├── WALK_FORWARD_ML.md          (Advanced: ML weight optimization)
├── HANDOFF.md                  (Current development status)
└── CLAUDE.md                   (Development guidelines)
```

## Critical Success Factors

### What Must Work
1. **Data quality:** Alpaca data must be clean, adjusted for splits/dividends
2. **TFC calculation:** Must accurately measure multi-timeframe alignment
3. **Risk management:** 2% position sizing must be enforced
4. **Walk-forward:** Out-of-sample results must match in-sample (no overfitting)

### What Can Fail (Acceptable)
1. **TFC underperforms MA:** Use Branch 2 baseline instead
2. **Confidence scoring adds no value:** Simplify to binary entry rules
3. **Specific stocks don't work:** Adjust stock selection criteria
4. **Win rate below 75%:** 55-65% is still profitable with good risk-reward

### Red Flags (Abort If Seen)
1. **Backtest Sharpe > 3.0:** Almost certainly overfitted
2. **Paper trading drastically worse than backtest:** Strategy doesn't work live
3. **Large divergence between in-sample and out-of-sample:** Overfitting
4. **Performance degrades month-over-month:** Market regime changed, strategy obsolete

## Research Warnings Applied

> "**44% of published trading strategies fail to replicate on new data** according to a 2014 study"

**Our mitigation:** Walk-forward analysis, conservative fee assumptions, 30% out-of-sample testing

> "Transaction costs destroy more strategies than any other factor... correcting from **0.1% to 0.2% per trade transformed profitable backtests into unprofitable reality**"

**Our mitigation:** Use 0.2% fees + 0.1% slippage in all backtests

> "**Simple systems with 2-3 parameters demonstrate greater robustness** than complex multi-indicator approaches"

**Our mitigation:** Branch 2 has 2 parameters (MA period, RSI threshold), Branch 1 has 4-5 (weights)

> "**MACD alone generates win rates below 50%**, RSI alone involves high risk in trending markets"

**Our mitigation:** Combine multiple indicators, never use single indicator

## Next Steps

1. **Review this overview** - Ensure alignment with goals
2. **Read IMPLEMENTATION_PLAN.md** - Understand build sequence
3. **Read BRANCH_COMPARISON.md** - Decide which branch to build first
4. **Read VALIDATION_PROTOCOL.md** - Understand testing requirements
5. **Choose branch** - Start with Branch 2 (proven) or Branch 1 (innovative)
6. **Begin development** - Follow implementation plan step-by-step

## Questions to Answer Before Building

- [ ] Do I understand why we're testing two branches? (TFC vs MA comparison)
- [ ] Am I comfortable with 55-65% win rates? (Not 75-85%)
- [ ] Can I commit to 6 months of paper trading? (Before risking real money)
- [ ] Do I accept 44% chance of failure? (Even with good research)
- [ ] Am I prepared to abandon TFC if it underperforms? (No sunk cost fallacy)

**If yes to all above, proceed to IMPLEMENTATION_PLAN.md**
