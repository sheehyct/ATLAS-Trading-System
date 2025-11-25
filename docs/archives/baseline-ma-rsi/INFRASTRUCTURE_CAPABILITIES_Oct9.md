# Infrastructure Capabilities Assessment

**Date:** 2025-10-09
**Purpose:** Document confirmed capabilities for three-strategy development approach

## Data Infrastructure

### Alpaca Algo Trader Plus ($99/month)

**Confirmed Capabilities:**
- Historical equity data since 2016
- Intraday bars: 1min, 5min, 15min, 1hour
- Daily, weekly, monthly bars
- No restriction on historical data access
- Rate limit: 10,000 API calls/minute
- Real-time data: All US Stock Exchanges (SIP feed)
- Paper trading: 3 accounts of different sizes

**User Confirmation:**
- 4+ years of 5-minute historical bars available
- Tested and verified during STRAT system development

### VectorBT Pro Integration

**Alpaca Data Fetching:**
```python
vbt.AlpacaData.pull(
    symbol,
    start="YYYY-MM-DD",
    end="YYYY-MM-DD",
    timeframe="5 minutes",  # Supports: 1min, 5min, 15min, 1hour, 1day, etc.
    tz='US/Eastern'
)
```

**Resampling Capabilities:**
- Pandas-style .resample() with origin alignment
- Market-aligned bins (9:30 not 10:00 for hourly)
- OHLCV aggregation: first, max, min, last, sum
- Custom timeframe conversion

**Existing MTF Manager (data/mtf_manager.py):**
- Fetches 5-minute base data
- Resamples to: 5min, 15min, 1H, 1D, 1W, 1M
- Market hours filtering (RTH only: 9:30-16:00 ET)
- Eastern Time with DST handling
- Market-aligned hourly bars (9:30-10:30, 10:30-11:30, etc.)
- Status: Production-grade, tested

## VectorBT Pro Advanced Features

**Confirmed from Documentation:**

### Multi-Timeframe Analysis
- Native support for MTF indicator calculations
- TA-Lib indicators support timeframe parameter
- Can run indicators on different timeframes simultaneously
- Resample output back to original frame for signal logic

### Regime Detection
- ADX indicator built-in: vbt.talib('ADX')
- Boolean threshold comparisons: adx > 25 for trending
- Conditional strategy enabling based on regime flags

### Dynamic Position Sizing
- Time-varying weights based on any logic
- Regime-based allocation shifts (60-70% to breakout vs mean reversion)
- Custom allocation functions

### Complex Exit Hierarchies
- Multiple exit types: SL (stop loss), TSL (trailing stop), TP (take profit)
- Time-based exits: td_stop=pd.Timedelta(days=14)
- Regime-based exits
- First-triggered exit logic
- Exit type tracking in portfolio results

### Backtesting
```python
pf = vbt.PF.from_signals(
    close=close,
    entries=long_entries,
    exits=long_exits,
    size=position_size,          # Dynamic or fixed
    fees=0.002,                  # 0.2% per trade
    slippage=0.001,              # 0.1% slippage
    sl_stop=stop_distance,       # ATR-based stop loss
    tp_stop=target_distance,     # 2:1 profit target
    td_stop=time_exit,           # Time-based exit
    max_open_positions=20        # Portfolio heat management
)
```

## Automated Execution

### Alpaca Trading API

**Order Types Supported:**
- Market orders
- Limit orders
- Stop orders
- Stop-limit orders
- Trailing stop orders
- Bracket orders (entry + SL + TP)
- OCO (One-Cancels-Other)
- OTO (One-Triggers-Other)

**Order Parameters:**
- time_in_force: day, gtc, ioc, fok
- extended_hours: true/false for pre/post market
- Fractional shares supported
- notional for dollar-based sizing

**Execution:**
- API-driven order submission (no manual intervention required)
- Order status tracking via webhooks or polling
- Automatic fill notifications

## Strategy-Specific Requirements

### Strategy 1: Opening Range Breakout

**Data Requirements:**
- [CONFIRMED] 5-minute bars (base data from MTF manager)
- [CONFIRMED] 4+ years historical (confirmed available)
- [CONFIRMED] Volume data (included in OHLCV)
- [CONFIRMED] Intraday timestamps for time-based exits

**Infrastructure:**
- [CONFIRMED] Can pull 5-minute bars from Alpaca
- [CONFIRMED] Existing MarketAlignedMTFManager provides 5-minute RTH data
- [CONFIRMED] VectorBT Pro supports intraday backtesting
- [CONFIRMED] Time-based exits: td_stop=pd.Timedelta(hours=6.5) for market close

**Gap:** [TODO] Opening volume surge detection logic (needs to be built)

### Strategy 2: MA+RSI Baseline

**Data Requirements:**
- [CONFIRMED] Daily bars (resample from 5-minute)
- [CONFIRMED] 4+ years historical
- [CONFIRMED] TA-Lib indicators: SMA, RSI, ATR

**Infrastructure:**
- [CONFIRMED] MTF manager resamples to daily (_resample_daily())
- [CONFIRMED] VectorBT Pro has native TA-Lib integration
- [CONFIRMED] Simple entry/exit logic supported

**Gap:** None - all capabilities exist

### Strategy 3: TFC + Regime Detection

**Data Requirements:**
- [CONFIRMED] Multi-timeframe: hourly, daily, weekly
- [CONFIRMED] ADX for regime detection
- [CONFIRMED] MACD, RSI for confirmation
- [CONFIRMED] ATR for position sizing

**Infrastructure:**
- [CONFIRMED] Existing MTF manager provides all timeframes
- [CONFIRMED] Existing core/analyzer.py has TFC calculation logic
- [CONFIRMED] VectorBT Pro supports ADX, MACD, RSI, ATR
- [CONFIRMED] Dynamic position sizing supported

**Gap:** [TODO] ADX regime detection integration (needs to be added to existing TFC logic)

## Resource Utilization Analysis

### Current Plan vs Proposed Plan

**Current Plan (from previous docs):**
- 1 strategy (MA+RSI) tested at 3 risk levels (conservative/moderate/aggressive)
- Uses only 1 timeframe (daily)
- Single backtest approach

**Proposed Plan:**
- 3 strategies tested at appropriate risk levels
- Uses multiple timeframes (5min, daily, hourly/daily/weekly)
- Parallel paper trading on 3 accounts

**Infrastructure Utilization:**

| Resource | Current Plan | Proposed Plan | Efficiency Gain |
|----------|--------------|---------------|-----------------|
| 3 Paper Accounts | Risk variations of same strategy | 3 different strategies | Tests 3 approaches vs 1 |
| Alpaca Data | Daily bars only | 5min + resampled timeframes | Full data spectrum |
| VectorBT Pro MTF | Unused | Fully utilized | Leverages existing code |
| 5-min historical data | Unused | Opening range breakout | Novel strategy enabled |
| ADX/Regime capabilities | Unused | Strategy 3 | Advanced feature testing |

**Conclusion:** Proposed plan utilizes 100% of paid infrastructure vs approximately 40% in current plan.

## Development Sequence Justification

### Strategy 1: MA+RSI Baseline (Build First)

**Why First:**
- Simplest logic (150-200 lines of code)
- Uses daily bars (single timeframe)
- No complex dependencies
- Validates core backtesting workflow
- Establishes baseline performance metrics
- Research-proven (Connors RSI 75% win rate)

**Success Criteria:**
- Backtest completes without errors
- Sharpe ratio > 0.8
- Win rate 60-75%
- Max drawdown < 20%
- Walk-forward efficiency > 70%

**Deliverables:**
- strategies/baseline_ma_rsi.py
- Backtest results report
- Performance metrics vs research expectations
- Parameter sensitivity analysis

**Estimated Time:** 1-2 weeks

### Strategy 2: Opening Range Breakout (Build Second)

**Why Second:**
- Novel approach (not in original plan)
- Different time horizon (intraday vs swing)
- Tests 5-minute data handling
- Validates volume analysis capabilities
- Psychological stress test (17% win rate)
- Fastest feedback (100 trades in 1 week)

**Dependencies:**
- [CONFIRMED] 5-minute data infrastructure (exists in MTF manager)
- [CONFIRMED] Intraday backtesting (VectorBT Pro supports)
- [CONFIRMED] Time-based exits (VectorBT Pro supports)
- [TODO] Opening volume surge detection (needs to be built)

**Success Criteria:**
- Sharpe ratio > 1.5
- Win rate 15-25% (asymmetric profile)
- Avg win / avg loss > 2.5:1
- Max drawdown < 25%
- Handles 5-20 trades/day

**Deliverables:**
- strategies/opening_range_breakout.py
- Volume surge detection module
- Intraday backtest results
- Comparison to research (2.396 Sharpe target)

**Estimated Time:** 2-3 weeks

### Strategy 3: TFC + Regime Detection (Build Third)

**Why Third:**
- Most complex (400+ lines of code)
- Builds on existing TFC infrastructure
- Adds novel regime detection layer
- Combines multiple indicators
- Dynamic strategy selection
- Tests if complexity adds value

**Dependencies:**
- [CONFIRMED] MTF manager (exists, production-grade)
- [CONFIRMED] TFC calculation (core/analyzer.py)
- [TODO] ADX regime detection (needs integration)
- [TODO] Dynamic allocation logic (needs to be built)
- [TODO] Multi-strategy portfolio (needs to be built)

**Success Criteria:**
- Sharpe ratio > 1.0 (must beat baseline by 0.3+)
- Win rate 45-60%
- TFC confidence scoring improves expectancy
- Regime detection reduces drawdowns
- Walk-forward efficiency > 70%

**Deliverables:**
- strategies/tfc_regime_detection.py
- Enhanced TFC module with regime detection
- Multi-strategy portfolio manager
- Comparison report: TFC vs Baseline
- Decision matrix: Does complexity justify returns?

**Estimated Time:** 3-4 weeks

## Total Timeline

**Sequential Development:**
- Strategy 1: Weeks 1-2
- Strategy 2: Weeks 3-5
- Strategy 3: Weeks 6-9

**Parallel Paper Trading (After Development):**
- Account 1: Opening Range Breakout
- Account 2: MA+RSI Baseline
- Account 3: TFC + Regime Detection
- Duration: 6 months minimum (100+ trades per strategy)

**Total Time to Live Trading Decision:** 9 weeks development + 6 months paper = approximately 8 months

## Risk Assessment

### Infrastructure Risks
- [CONFIRMED] Data availability: 5-min bars, 4+ years
- [CONFIRMED] API rate limits: 10,000/min (sufficient)
- [CONFIRMED] Execution capabilities: Automated via Alpaca API
- [WARNING] Resampling accuracy: Existing MTF manager is tested, but should verify edge cases
- [WARNING] Backtest vs live divergence: Will discover in paper trading

### Development Risks
- [WARNING] Opening range volume detection: Novel logic, needs validation
- [WARNING] Regime detection accuracy: ADX threshold tuning required
- [WARNING] TFC scoring calibration: Existing code, but may need adjustment for swing trades
- [WARNING] Multi-strategy portfolio heat: Complex logic, easy to introduce bugs

### Psychological Risks
- [WARNING] Opening range 17% win rate: User states no emotional trading, but 8-loss streaks will test this
- [CONFIRMED] Paper trading safety: No real money risk during 6-month validation

## Next Steps

1. Confirm sequenced build order (Strategy 1 -> 2 -> 3)
2. Begin Strategy 1 design (MA+RSI baseline)
3. Create strategy template structure
4. Document coding standards for strategy modules
5. Set up testing framework (backtest validation, not permanent tests)

---

**Status:** Infrastructure assessment complete. Ready to proceed with Strategy 1 development.
