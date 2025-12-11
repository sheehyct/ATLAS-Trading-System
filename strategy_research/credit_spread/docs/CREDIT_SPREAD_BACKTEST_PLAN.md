# Credit Spread Strategy Backtest - VectorBT Pro Implementation Plan

## Overview

Backtest the "Tier's Credit Spreads Leverage ETF Strategy" using VectorBT Pro with comprehensive visualization, metrics, and analysis capabilities.

**Strategy Summary**: Binary on/off strategy trading SPXL/SSO/UPRO based on credit spread signals to capture positive trends while avoiding volatility decay.

---

## MANDATORY: 5-Step VBT Pro Workflow

Before ANY implementation, we MUST follow the 5-step verification process from CLAUDE.md:

### Step 1: SEARCH Documentation
Search for relevant VBT Pro features:
- Custom indicator creation
- Portfolio.from_signals() usage
- Binary signal strategies
- Performance metrics and plotting
- Regime-based strategies

### Step 2: VERIFY API Methods Exist
Verify these methods/classes exist:
- `vbt.Portfolio.from_signals`
- `vbt.IF` (IndicatorFactory)
- `vbt.Data.download`
- Plotting methods for signals and performance
- Performance metrics (Sharpe, Sortino, max DD, etc.)

### Step 3: FIND Real Examples
Find examples of:
- Binary on/off strategies
- Custom indicators with state tracking (for tracking recent highs/lows)
- Signal visualization overlays
- Multi-asset comparison plots

### Step 4: TEST Minimal Example
Create minimal test with:
- Dummy credit spread data
- Simple signals (35% fall = entry, 40% rise = exit)
- Portfolio backtest
- Basic plotting

### Step 5: IMPLEMENT Full System
Only after Steps 1-4 pass successfully.

---

## VectorBT Pro Features to Leverage

Based on CLAUDE.md 5-step workflow, we will use:

### 1. Data Management
- `vbt.YFData.download()` for price data (SPXL, SSO, UPRO, SPY)
- `pandas_datareader` or FRED API for credit spread data (BAMLH0A0HYM2)
- Data alignment and resampling

### 2. Custom Indicator Creation
**Credit Spread Signal Generator** (IndicatorFactory):
```python
@njit
def credit_spread_signals_nb(spreads, ema_330):
    """
    Generate entry/exit signals from credit spreads.

    Entry: Spreads fall 35% from recent high
    Exit: Spreads rise 40% from recent low AND cross above 330-day EMA
    """
    # Numba-compiled for performance
    # Track recent highs/lows dynamically
    # Return entry/exit boolean arrays
    pass

CreditSpreadSignals = vbt.IF(
    class_name='CreditSpreadSignals',
    input_names=['spreads', 'ema'],
    output_names=['entries', 'exits', 'recent_high', 'recent_low']
).with_apply_func(credit_spread_signals_nb)
```

### 3. Portfolio Backtesting
```python
pf = vbt.Portfolio.from_signals(
    close=sso_prices,
    entries=entries,
    exits=exits,
    size=1.0,  # 100% of portfolio
    size_type='valuepercent',  # Use percentage of portfolio value
    init_cash=10000,
    fees=0.0009,  # ~0.09% expense ratio for SSO
    freq='D'
)
```

### 4. Performance Metrics (Built-in)
VBT Pro provides extensive metrics out-of-the-box:

**Returns Analysis:**
- `pf.total_return` - Total return
- `pf.annual_return` - Annualized return
- `pf.cumulative_returns` - Cumulative returns over time

**Risk Metrics:**
- `pf.sharpe_ratio` - Risk-adjusted returns
- `pf.sortino_ratio` - Downside risk-adjusted returns
- `pf.calmar_ratio` - Return vs max drawdown
- `pf.max_drawdown` - Maximum peak-to-trough decline
- `pf.value_at_risk` - VaR at various confidence levels

**Trade Analysis:**
- `pf.trades.count()` - Number of trades
- `pf.trades.win_rate` - Percentage of winning trades
- `pf.trades.profit_factor` - Gross profit / gross loss
- `pf.trades.avg_winning_trade` - Average win size
- `pf.trades.avg_losing_trade` - Average loss size
- `pf.trades.duration.mean()` - Average trade duration

**Exposure Metrics:**
- `pf.market_exposure` - Time in market percentage
- `pf.gross_exposure` - Portfolio exposure over time

### 5. Visualization & Plotting

VBT Pro has powerful built-in plotting:

**a) Strategy Performance Plot:**
```python
# Complete strategy overview
pf.plot().show()
```

**b) Signal Visualization:**
```python
# Plot credit spreads with entry/exit signals
fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True)

# Subplot 1: Credit spreads + signals + EMA
fig.add_trace(
    vbt.go.Scatter(x=spreads.index, y=spreads, name='Credit Spreads'),
    row=1, col=1
)
fig.add_trace(
    vbt.go.Scatter(x=ema_330.index, y=ema_330, name='330-day EMA'),
    row=1, col=1
)
# Add entry/exit markers
pf.plot_signals(fig=fig, row=1, col=1)

# Subplot 2: Portfolio value
fig.add_trace(
    vbt.go.Scatter(x=pf.value.index, y=pf.value, name='Portfolio Value'),
    row=2, col=1
)
```

**c) Trade Analysis Plot:**
```python
# Visualize all trades
pf.trades.plot().show()
```

**d) Drawdown Analysis:**
```python
# Plot drawdown curve
pf.drawdowns.plot().show()
```

**e) Returns Distribution:**
```python
# Plot returns histogram
pf.returns.vbt.plot_hist().show()
```

**f) Comparison Plot (Strategy vs SPY):**
```python
# Compare multiple strategies
vbt.Portfolio.compare_performances([
    ('Credit Spread Strategy', pf),
    ('SPY Buy & Hold', pf_spy)
]).plot().show()
```

### 6. Advanced Analysis Features

**Walk-Forward Validation:**
```python
# Test strategy robustness across different periods
wfo = vbt.WFO(
    n_splits=5,  # 5 periods
    split_every='365 days',
    look_ahead='0 days'
)

# Run walk-forward optimization
wfo_results = wfo.run(
    portfolio_func=run_strategy,
    data=data
)
```

**Monte Carlo Simulation:**
```python
# Test strategy under randomized scenarios
mc = pf.returns.vbt.monte_carlo(
    n=1000,  # 1000 simulations
    seed=42
)

# Plot Monte Carlo results
mc.plot_distribution().show()
```

**Regime Analysis:**
```python
# Analyze performance by market regime
regimes = identify_market_regimes(spy_data)
pf.trades.by_regime(regimes).plot_metrics().show()
```

---

## Implementation Steps

### Phase 1: Data Acquisition & Validation
1. Download credit spread data from FRED (BAMLH0A0HYM2)
2. Download SSO, SPXL, UPRO, SPY price data
3. Verify data quality and alignment
4. Handle missing data, weekends, holidays
5. Create validation dataset matching video examples (July 2007 - 2024)

**VBT Pro Tools:**
- `vbt.YFData.download()` for price data
- `pandas_datareader.fred` for FRED data
- Data alignment with `vbt.data.align_index()`

### Phase 2: Signal Generation Logic
1. Calculate 330-day EMA on credit spreads
2. Implement recent high/low tracking algorithm
3. Generate entry signals (35% fall from recent high)
4. Generate exit signals (40% rise from recent low + EMA cross)
5. Validate signals against video dates (Section 5 of rules doc)

**VBT Pro Tools:**
- Custom IndicatorFactory for signal generation
- `mcp__vectorbt-pro__run_code()` for testing signal logic
- Export signals to CSV for manual verification

### Phase 3: Minimal Backtest (Validation)
1. Test with SSO data (2007-2024)
2. Verify signal dates match video (Table in Section 5)
3. Verify trade returns match video claims (Section 6)
4. Check final performance: ~16.3x return from £10,000
5. Debug discrepancies if any

**VBT Pro Tools:**
- `vbt.Portfolio.from_signals()`
- `pf.trades.records_readable` for trade inspection
- `pf.total_return` for performance validation

### Phase 4: Comprehensive Analysis
1. Generate full performance metrics suite
2. Compare vs SPY benchmark
3. Test with SPXL (3x leverage) for comparison
4. Analyze drawdowns and time in market
5. Calculate risk-adjusted metrics (Sharpe, Sortino, Calmar)

**VBT Pro Tools:**
- All built-in metrics from `pf.*`
- `pf.stats()` for comprehensive stats table
- Comparison plots vs benchmark

### Phase 5: Visualization Suite
1. Credit spread chart with signals overlay
2. Portfolio equity curve with entry/exit markers
3. Drawdown analysis plot
4. Trade distribution analysis
5. Returns histogram and distribution
6. Comparison plot (Strategy vs SPY vs SPXL)

**VBT Pro Tools:**
- `pf.plot()` for comprehensive plots
- `vbt.make_subplots()` for custom layouts
- `pf.trades.plot()` for trade visualization
- `pf.drawdowns.plot()` for drawdown analysis

### Phase 6: Sensitivity Analysis
1. Test different local high/low lookback periods
2. Test different entry/exit thresholds (30%, 35%, 40%, 45%)
3. Test impact of transaction costs
4. Test with different leveraged instruments (2x vs 3x)
5. Document parameter sensitivity

**VBT Pro Tools:**
- Parameter grid search with `vbt.Portfolio.from_signals()`
- Heatmap visualization of parameter sensitivity
- `vbt.ParamOptimizer` for systematic testing

### Phase 7: Robustness Testing
1. Walk-forward validation
2. Out-of-sample testing (2020-2024 if trained on 2006-2019)
3. Monte Carlo simulation
4. Bootstrap analysis
5. Stress testing (simulate 2008-like crash scenarios)

**VBT Pro Tools:**
- `vbt.WFO()` for walk-forward optimization
- `pf.returns.vbt.monte_carlo()` for Monte Carlo
- Bootstrap resampling with VBT

---

## Expected Deliverables

### 1. Code Files
- `credit_spread_data.py` - Data download and preparation
- `credit_spread_signals.py` - Signal generation custom indicator
- `credit_spread_backtest.py` - Main backtest execution
- `credit_spread_analysis.py` - Performance analysis and visualization
- `credit_spread_validation.py` - Validate against video claims

### 2. Visualizations
- `credit_spreads_signals.html` - Interactive credit spread chart with signals
- `portfolio_performance.html` - Equity curve with trade markers
- `drawdown_analysis.html` - Drawdown chart
- `strategy_comparison.html` - Strategy vs benchmarks
- `trade_analysis.html` - Trade distribution and statistics
- `sensitivity_heatmap.html` - Parameter sensitivity analysis

### 3. Reports
- `backtest_results.csv` - All trades with entry/exit prices
- `performance_metrics.txt` - Comprehensive performance statistics
- `validation_report.md` - Comparison vs video claims
- `sensitivity_analysis.csv` - Parameter sensitivity results

### 4. Documentation
- Updated `CREDIT_SPREAD_STRATEGY_RULES.md` with findings
- Implementation notes and lessons learned
- Discrepancy analysis (if results differ from video)

---

## Critical Validation Checkpoints

### Checkpoint 1: Signal Generation
**Criteria:**
- Entry/exit dates match video Table (Section 5) within 1-2 days
- If >3 days off, investigate local high/low calculation

**VBT Pro Validation:**
```python
# Export signals to CSV
signals_df = pd.DataFrame({
    'date': data.index,
    'spread': spreads,
    'entry': entries,
    'exit': exits,
    'recent_high': recent_highs,
    'recent_low': recent_lows
})
signals_df[signals_df['entry'] | signals_df['exit']].to_csv('signals.csv')

# Compare to video dates
video_dates = ['1998-08-18', '2003-04-03', ...]
for date in video_dates:
    actual = signals_df.loc[date]
    print(f"{date}: Entry={actual['entry']}, Exit={actual['exit']}")
```

### Checkpoint 2: Trade Returns
**Criteria:**
- SSO trade returns match Section 6 claims within 5%
- Example: Apr 30, 2009 → Aug 4, 2011 should be ~82% gain

**VBT Pro Validation:**
```python
trades = pf.trades.records_readable
for idx, trade in trades.iterrows():
    entry_date = trade['Entry Date']
    exit_date = trade['Exit Date']
    return_pct = trade['Return %']

    # Compare to video claims
    print(f"{entry_date} → {exit_date}: {return_pct:.1%}")
```

### Checkpoint 3: Final Performance
**Criteria:**
- £10,000 starting July 2007 → ~£163,651 by 2024 (16.3x)
- Allow ±10% tolerance for data differences

**VBT Pro Validation:**
```python
initial_value = pf.init_cash
final_value = pf.final_value
total_return = pf.total_return

print(f"Initial: £{initial_value:,.0f}")
print(f"Final: £{final_value:,.0f}")
print(f"Multiple: {final_value/initial_value:.1f}x")
print(f"Total Return: {total_return:.1%}")

assert 14.0 < (final_value/initial_value) < 18.0, "Performance outside tolerance"
```

---

## Open Questions to Resolve

### 1. Local High/Low Algorithm
**Question**: Exact algorithm for tracking "recent highs" and "recent lows"

**Test Approaches**:
- Option A: Rolling 180-day window
- Option B: Highest/lowest since last signal change
- Option C: Peak detection (scipy.signal.find_peaks)

**Resolution Strategy**:
- Implement all 3 approaches
- Compare signal dates vs video Table
- Select approach with best match

**VBT Pro Testing**:
```python
# Test all 3 approaches
for approach in ['rolling_180', 'since_last_signal', 'peak_detection']:
    signals = generate_signals(spreads, approach=approach)
    dates = signals[signals['entry'] | signals['exit']].index

    # Compare to video dates
    match_rate = calculate_match_rate(dates, video_dates)
    print(f"{approach}: {match_rate:.1%} match")
```

### 2. Signal Execution Timing
**Question**: Same-day close or next-day open execution?

**Test Both**:
- Conservative: Signal triggers at close, execute next day's open
- Aggressive: Signal triggers and executes same day's close

**Impact Assessment**:
```python
# Test both timing assumptions
pf_conservative = backtest(signals, execution='next_open')
pf_aggressive = backtest(signals, execution='same_close')

print(f"Conservative: {pf_conservative.total_return:.1%}")
print(f"Aggressive: {pf_aggressive.total_return:.1%}")
print(f"Difference: {abs(pf_conservative.total_return - pf_aggressive.total_return):.1%}")
```

### 3. Exit Condition Simultaneity
**Question**: Must both exit conditions trigger on same day?

**Test Scenarios**:
- Strict: 40% rise AND EMA cross on same day
- Relaxed: 40% rise first, then EMA cross within 5 days
- Either: Whichever comes first

**VBT Pro Analysis**:
```python
# Analyze actual exit patterns
exits_40pct = (spreads > recent_lows * 1.4)
exits_ema = (spreads > ema_330)

# Check how often they coincide
simultaneous = (exits_40pct & exits_ema).sum()
total_exits = exits_40pct.sum()

print(f"Simultaneous: {simultaneous}/{total_exits} ({simultaneous/total_exits:.1%})")
```

---

## Risk Management Considerations

### Strategy-Specific Risks

1. **Volatility Decay Risk**
   - Leveraged ETFs lose value in choppy markets
   - Strategy specifically designed to avoid this
   - Validation: Measure performance in choppy vs trending periods

2. **Extended Drawdown Risk**
   - Strategy can be out of market 1-2 years
   - No gains during out periods (opportunity cost)
   - Validation: Compare opportunity cost vs crash avoidance

3. **Signal Lag Risk**
   - Exit signal may come after sell-off begins
   - Example: Feb 2020 exit had -3% loss before COVID crash
   - Validation: Measure average lag from peak to exit signal

4. **Leveraged ETF Decay**
   - Daily rebalancing causes tracking error
   - Expense ratios (~0.9% annually)
   - Validation: Compare SPXL/SSO vs 2x/3x SPY theoretical

### VBT Pro Risk Analysis Tools

```python
# Analyze risk metrics by period
risk_analysis = pf.stats([
    'max_drawdown',
    'avg_drawdown',
    'max_drawdown_duration',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'value_at_risk_95'
])

print(risk_analysis)

# Drawdown analysis
dd = pf.drawdowns
print(f"Max Drawdown: {dd.max_drawdown:.2%}")
print(f"Avg Drawdown: {dd.avg_drawdown:.2%}")
print(f"Longest Drawdown: {dd.max_duration} days")

# Underwater plot (time spent in drawdown)
pf.plot_underwater().show()
```

---

## Success Criteria

### Must Pass (Critical):
1. Signal dates match video ±3 days (>80% match rate)
2. SSO trade returns match video ±10%
3. Final performance 14x-18x (video claims 16.3x)
4. No trades on weekends or holidays
5. 100% position sizing when in market

### Should Pass (Important):
1. Sharpe ratio >1.0
2. Maximum drawdown <-40%
3. Win rate >60%
4. Market exposure 50-70% (reflecting in/out periods)
5. Outperforms SPY by >2x

### Nice to Have (Stretch Goals):
1. Sortino ratio >1.5
2. Calmar ratio >1.0
3. Profit factor >2.0
4. Walk-forward validation shows consistency
5. Monte Carlo 95% confidence interval includes video results

---

## Next Steps

1. **Follow 5-Step VBT Pro Workflow** (MANDATORY before coding)
   - STEP 1: Search VBT docs for signal strategies
   - STEP 2: Verify Portfolio.from_signals API
   - STEP 3: Find examples of binary on/off strategies
   - STEP 4: Test minimal example with dummy data
   - STEP 5: Implement full backtest

2. **Start with Data Acquisition**
   - Download FRED credit spread data
   - Download price data for SSO, SPXL, SPY
   - Verify data quality and alignment

3. **Validate Signal Logic**
   - Implement 3 approaches to local high/low tracking
   - Compare signal dates vs video
   - Select best approach

4. **Run Minimal Backtest**
   - Test with SSO (2007-2024)
   - Verify against video claims
   - Debug any discrepancies

5. **Generate Comprehensive Analysis**
   - Full metrics suite
   - Visualization plots
   - Sensitivity analysis

---

**Status**: Ready to begin 5-step VBT Pro workflow
**Priority**: HIGH - This is a feasible and interesting strategy to validate
**Estimated Time**: 4-6 hours (with proper VBT Pro verification)
