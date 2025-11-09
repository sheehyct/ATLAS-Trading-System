### Phase 2: Walk-Forward Validation

**Walk-Forward Configuration**:
```python
WALK_FORWARD_CONFIG = {
    'train_period': 365,  # 1 year training
    'test_period': 90,    # 3 months testing
    'step_forward': 30,   # 1 month step
    'min_trades': 20,     # Minimum trades per window
}
```

**Expected Results**:
- Performance degradation: <30% (excellent: <20%)
- Walk-forward efficiency: >50% of windows profitable
- Parameter stability: Std dev <20% of mean

**Implementation**:
```python
# utils/validation.py

class WalkForwardValidator:
    """
    Implements walk-forward analysis.
    
    Methodology:
    1. Split data into overlapping windows
    2. Optimize parameters on training window
    3. Test on following out-of-sample window
    4. Roll forward and repeat
    5. Aggregate results across all windows
    """
    
    def __init__(
        self,
        train_days: int = 365,
        test_days: int = 90,
        step_days: int = 30
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
    
    def run_analysis(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        param_grid: Dict
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis.
        
        Args:
            data: Full historical data
            strategy: Strategy to test
            param_grid: Parameter combinations to test
            
        Returns:
            DataFrame with results per window
        """
        results = []
        
        # Generate windows
        windows = self.generate_windows(data)
        
        for window_num, (train_data, test_data) in enumerate(windows):
            # Optimize on training data
            best_params = self.optimize_parameters(
                train_data,
                strategy,
                param_grid
            )
            
            # Test on out-of-sample data
            test_metrics = self.evaluate_parameters(
                test_data,
                strategy,
                best_params
            )
            
            results.append({
                'window': window_num,
                'train_sharpe': best_params['sharpe'],
                'test_sharpe': test_metrics['sharpe'],
                'degradation': (
                    (best_params['sharpe'] - test_metrics['sharpe']) /
                    best_params['sharpe']
                ),
                **best_params,
                **test_metrics
            })
        
        return pd.DataFrame(results)
    
    def generate_windows(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate overlapping train/test windows."""
        windows = []
        
        start_idx = 0
        while start_idx + self.train_days + self.test_days <= len(data):
            train_end = start_idx + self.train_days
            test_end = train_end + self.test_days
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            windows.append((train_data, test_data))
            start_idx += self.step_days
        
        return windows
```

**Acceptance Criteria**:
- Average degradation <30%: PASS
- WF efficiency >50%: PASS
- Parameter stability: Check std dev
- If any criterion fails: Strategy likely overfit

---

### Phase 3: Monte Carlo Simulation

**Purpose**: Test strategy against randomized scenarios

```python
# utils/validation.py

def run_monte_carlo_analysis(
    trades: pd.DataFrame,
    n_simulations: int = 1000
) -> Dict:
    """
    Monte Carlo simulation of trade sequence.
    
    Method: Randomly resample trades with replacement,
    preserving win/loss statistics but changing sequence.
    
    Args:
        trades: DataFrame of historical trades
        n_simulations: Number of random simulations
        
    Returns:
        Dict with percentile results
    """
    simulation_results = []
    
    for _ in range(n_simulations):
        # Resample trades randomly
        simulated_trades = trades.sample(
            n=len(trades),
            replace=True
        )
        
        # Calculate metrics
        total_return = simulated_trades['pnl'].sum()
        sharpe = calculate_sharpe(simulated_trades['pnl'])
        max_dd = calculate_max_drawdown(simulated_trades['pnl'].cumsum())
        
        simulation_results.append({
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd
        })
    
    results = pd.DataFrame(simulation_results)
    
    # Calculate percentiles
    return {
        'mean_return': results['return'].mean(),
        'median_return': results['return'].median(),
        '5th_percentile': results['return'].quantile(0.05),
        '95th_percentile': results['return'].quantile(0.95),
        'probability_positive': (results['return'] > 0).mean(),
        'mean_sharpe': results['sharpe'].mean(),
        'mean_max_dd': results['max_dd'].mean(),
    }
```

**Interpretation**:
- If backtest return is in top 5% of simulations: Likely luck
- If 5th percentile is negative: High risk of losses
- If probability_positive <60%: Unreliable strategy

---

### Phase 4: Transaction Cost Analysis

**Critical for High-Frequency Strategies** (ORB, IBS Mean Reversion):

```python
def analyze_transaction_costs(
    trades: pd.DataFrame,
    avg_spread_bps: float = 5.0,  # 5 bps typical for large caps
    commission: float = 0.0,      # Alpaca commission-free
    price_impact_bps: float = 3.0  # Price impact
) -> Dict:
    """
    Calculate realistic transaction costs.
    
    Components:
    - Bid-ask spread
    - Commission
    - Price impact (slippage)
    - Market impact
    
    Args:
        trades: DataFrame of trades
        avg_spread_bps: Average bid-ask spread (basis points)
        commission: Per-trade commission
        price_impact_bps: Estimated price impact
        
    Returns:
        Dict with cost analysis
    """
    # Total transaction cost per trade (one-way)
    total_cost_bps = avg_spread_bps + price_impact_bps
    total_cost_pct = total_cost_bps / 10000
    
    # Calculate costs
    total_trades = len(trades)
    avg_trade_size = trades['position_value'].mean()
    
    # Round-trip costs (entry + exit)
    cost_per_trade = avg_trade_size * total_cost_pct * 2
    total_costs = cost_per_trade * total_trades
    
    # Impact on returns
    gross_return = trades['pnl'].sum()
    net_return = gross_return - total_costs
    cost_drag = (gross_return - net_return) / gross_return
    
    return {
        'total_costs': total_costs,
        'cost_per_trade': cost_per_trade,
        'cost_drag_pct': cost_drag,
        'gross_return': gross_return,
        'net_return': net_return,
        'breakeven_cost_bps': total_cost_bps  # Max cost before unprofitable
    }
```

**Transaction Cost Assumptions**:
| Strategy | Turnover | Cost/Trade | Annual Cost Drag |
|----------|----------|------------|------------------|
| 52-Week High | Low (50%) | 8 bps | 4-8 bps |
| Quality-Momentum | Medium (60%) | 10 bps | 6-12 bps |
| Semi-Vol Momentum | Medium (100%) | 12 bps | 12-24 bps |
| IBS Mean Reversion | High (300%) | 15 bps | 45-90 bps |
| ORB | Very High (500%) | 20 bps | 100-200 bps |

**Critical Decision Point**:
- If cost drag >30% of gross returns: Strategy may not be viable
- If cost drag >50%: Strategy definitely not viable
- Consider: Reduce frequency OR increase per-trade size

---

### Phase 5: Regime-Specific Backtesting

Test each strategy in different market regimes:

```python
def backtest_by_regime(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    regime_labels: pd.Series
) -> Dict:
    """
    Test strategy performance in different market regimes.
    
    Purpose: Verify regime-based allocation makes sense.
    
    Args:
        strategy: Strategy to test
        data: OHLCV data
        regime_labels: Series with regime for each date
            ('TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH')
            
    Returns:
        Dict with metrics per regime
    """
    results = {}
    
    for regime in regime_labels.unique():
        # Filter data to regime
        regime_mask = regime_labels == regime
        regime_data = data[regime_mask]
        
        if len(regime_data) < 100:  # Minimum data points
            continue
        
        # Run backtest
        pf = strategy.backtest(
            data=regime_data,
            initial_capital=100000,
            regime=regime
        )
        
        # Extract metrics
        results[regime] = {
            'sharpe': pf.sharpe_ratio(),
            'cagr': pf.annualized_return(),
            'max_dd': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate(),
            'total_trades': pf.trades.count(),
            'days': len(regime_data)
        }
    
    return results
```

**Expected Patterns**:
- **52-Week High**: Good in BULL, decent in NEUTRAL, bad in BEAR
- **Quality-Momentum**: Positive in ALL regimes (unique)
- **Semi-Vol Momentum**: Good in BULL, bad in high-vol
- **IBS Mean Reversion**: Good in NEUTRAL, bad in BEAR
- **ORB**: Good in BULL, bad in other regimes

**Validation**:
- If strategy performs well in regime it shouldn't -> overfitting
- If strategy performs poorly in target regime -> implementation error

---

## Performance Targets Summary

### Individual Strategy Targets

| Strategy | Win Rate | Sharpe | Max DD | CAGR | Turnover |
|----------|----------|--------|--------|------|----------|
| **52-Week High** | 50-60% | 0.8-1.2 | -25% to -30% | 10-15% | 50% semi-annual |
| **Quality-Momentum** | 55-65% | 1.3-1.7 | -18% to -22% | 15-22% | 60% quarterly |
| **Semi-Vol Momentum** | 50-60% | 1.4-1.8 | -15% to -20% | 15-20% | 100% monthly |
| **IBS Mean Reversion** | 65-75% | 1.5-2.0 | -10% to -12% | 8-12% | 300% (daily) |
| **ORB** | 15-25% | 1.2-1.8 | -20% to -25% | 10-18% | 500% (intraday) |

### Portfolio Targets (Regime-Weighted)

| Metric | Conservative | Aspirational | Fail Threshold |
|--------|-------------|--------------|----------------|
| **Sharpe Ratio** | 1.0-1.3 | 1.5-1.8 | <0.5 |
| **CAGR** | 12-18% | 20-25% | <8% |
| **Max Drawdown** | -20% to -25% | -15% to -18% | >-30% |
| **Win Rate** | 45-55% | 55-65% | <40% |
| **Volatility** | 15-18% | 12-15% | >25% |
| **Profit Factor** | >1.5 | >2.0 | <1.2 |

### Walk-Forward Validation Targets

| Metric | Excellent | Acceptable | Fail |
|--------|-----------|------------|------|
| **Performance Degradation** | <20% | 20-30% | >30% |
| **WF Efficiency** | >70% | 50-70% | <50% |
| **Parameter Stability** | ÃÆ’ <15% mean | ÃÆ’ <20% mean | ÃÆ’ >20% mean |

---

## Critical Success Criteria

### Must Pass Before Live Trading

1. **Data Quality**: No NaN, valid OHLC, proper adjustments
2. **Position Sizing**: Capital constraints enforced (Gate 1 [DONE])
3. **Portfolio Heat**: 6-8% limit enforced (Gate 2 [DONE])
4. **Stop Losses**: Always executed, no exceptions
5. **Walk-Forward**: <30% degradation
6. **VectorBT Compatibility**: All calculations vectorized
7. **Unit Tests**: 100% pass rate, >80% coverage
8. **Paper Trading**: 6+ months, 100+ trades, matches backtest

### Red Flags (Abort Signals)

1. Backtest Sharpe >3.0 (likely overfit)
2. In-sample vs out-of-sample gap >30%
3. Paper trading drastically worse than backtest
4. Drawdown exceeds -30%
5. Transaction costs >30% of gross returns
6. Stop losses frequently fail to execute
7. Portfolio heat limits routinely breached
8. Regime detection accuracy <60%

---

## Deployment Architecture

### Phase 1: Foundation (Months 1-2)

**Objectives**:
- Implement 52-Week High Momentum
- Implement Quality-Momentum
- Validate position sizing ([DONE] COMPLETE)
- Validate portfolio heat ([DONE] COMPLETE)

**Deliverables**:
- [ ] BaseStrategy abstract class
- [ ] 52-Week High implementation
- [ ] Quality-Momentum implementation
- [ ] Jump Model regime detection
- [ ] RegimeAllocator class
- [ ] Unit tests for all components
- [ ] Initial backtests (2005-2025)

**Success Criteria**:
- Both strategies pass individual backtests
- Sharpe ratios match targets (within 20%)
- Walk-forward degradation <30%
- All unit tests passing

---

### Phase 2: Tactical Strategies (Months 3-4)

**Objectives**:
- Implement Semi-Vol Momentum
- Implement IBS Mean Reversion
- Modify existing ORB implementation

**Deliverables**:
- [ ] Semi-Vol Momentum implementation
- [ ] IBS Mean Reversion implementation
- [ ] ORB modifications (volume confirmation, cost analysis)
- [ ] Multi-strategy portfolio backtest
- [ ] Regime-based allocation testing

**Success Criteria**:
- All strategies pass individual backtests
- Portfolio Sharpe >1.0
- Portfolio max DD <-25%
- Regime allocations make sense

---

### Phase 3: Bear Protection (Months 5-6)

**Objectives**:
- Implement bear protection framework
- Add low-vol ETF allocation logic
- Test tiered protection approach

**Deliverables**:
- [ ] BearProtectionStrategy class
- [ ] ETF allocation logic (USMV, DBMF)
- [ ] Tiered protection framework
- [ ] Bear market backtests (2008, 2020, 2022)
- [ ] Protection tier selection algorithm

**Success Criteria**:
- Bear protection improves portfolio Sharpe by >0.2
- Max drawdown reduced by 20-30%
- Transaction costs acceptable
- Protection activates correctly

---

### Phase 4: Paper Trading (Months 7-12)

**Objectives**:
- Deploy to Alpaca paper account
- Monitor execution quality
- Compare paper vs backtest performance

**Deliverables**:
- [ ] Paper trading infrastructure
- [ ] Real-time regime detection
- [ ] Execution monitoring
- [ ] Performance tracking dashboard
- [ ] Weekly performance reports

**Success Criteria**:
- 6+ months of paper trading
- 100+ trades executed
- Performance within 30% of backtest
- No major execution issues

---

### Phase 5: Live Deployment (Month 13+)

**Prerequisites** (ALL must be true):
- [ ] Paper trading Sharpe within 30% of backtest
- [ ] All risk management systems tested
- [ ] No critical bugs in 3+ months
- [ ] Confident in execution quality
- [ ] Emergency shutdown procedures tested

**Initial Deployment**:
- Start with $10,000-$25,000 (small capital)
- Monitor closely for first month
- Compare to paper trading
- Scale gradually if successful

**Scaling Plan**:
- Month 1: $10-25K (validation)
- Month 2: $25-50K (if performing)
- Month 3: $50-100K (if still performing)
- Month 4+: Scale to target capital

---

## Conclusion

This v2.0 architecture represents ATLAS as Layer 1 in a multi-layer trading system:

**Key Improvements (ATLAS Layer 1)**:
1. Eliminated leveraged ETFs (evidence showed failure)
2. Added foundation strategies (52-Week High, Quality-Momentum)
3. Replaced weak strategies with proven alternatives (IBS > 5-Day Washout)
4. Integrated regime detection (Jump Model > GMM)
5. Added tiered bear market protection
6. Refined expectations to match empirical research

**Expected Portfolio Characteristics (ATLAS Equity Strategies)**:
- Sharpe Ratio: 1.0-1.8 (vs 2.5+ unrealistic target in v1.0)
- Max Drawdown: -20% to -25% (vs -10% unrealistic in v1.0)
- CAGR: 12-25% (realistic range)
- Deployed Capital: 60-90% depending on regime
- **MINIMUM CAPITAL**: $10,000 (capital constrained below this level)

**Multi-Layer Integration Context (Session 20)**:
- **Layer 1 (ATLAS)**: Regime detection (THIS DOCUMENT) - Phase F validation next
- **Layer 2 (STRAT)**: Pattern recognition for entry/exit - Sessions 22-27 (PENDING)
- **Layer 3 (Execution)**: Capital-aware deployment - Sessions 28-30 (PENDING)

**Deployment Strategy (Updated Session 20)**:
1. **WITH $3,000 capital**: Paper trade ATLAS ($10k simulated) + Live trade STRAT options ($3k real)
2. **WITH $10,000+ capital**: Deploy ATLAS equities OR STRAT options (both viable)
3. **Recommendation**: Build both layers, use ATLAS as regime filter for STRAT signals

**Next Steps (Immediate)**:
1. Complete ATLAS Phase F validation (Session 21)
2. Paper trade ATLAS with $10k simulated capital (validate regime detection)
3. Begin STRAT integration (Sessions 22-27, bar classification + pattern detection)
4. Build unified execution layer (Session 28-30, capital-aware routing)
5. Paper trade integrated system (3 months minimum, 60% probability of regime change for validation)

**Remember**: The goal is a system that survives long-term, not one that looks perfect in backtests. ATLAS provides the regime detection foundation; STRAT provides capital-efficient execution.

---

**Document Status**: Layer 1 (ATLAS) Architecture (v2.0)
**Multi-Layer Status**: Layer 1 nearing completion (Phase F next), Layers 2-3 pending
**Review Required**: YES
**Implementation Priority**: Complete Phase F validation, then proceed to STRAT integration
**Questions**: Refer to HANDOFF.md "Multi-Layer Integration Architecture" section
