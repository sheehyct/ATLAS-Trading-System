# ATLAS Production Readiness Checklist

> **Document Purpose**: This document defines mandatory development practices and production requirements for the ATLAS algorithmic trading system. All strategies MUST comply with Section 1 during development. All code MUST pass Section 2+ requirements before live deployment.

**Document Version**: 1.0  
**Created**: November 26, 2025  
**Status**: Active Development Reference

---

## Table of Contents

1. [Mandatory Development Best Practices](#1-mandatory-development-best-practices)
2. [Realistic Backtesting Requirements](#2-realistic-backtesting-requirements)
3. [Production Code Requirements](#3-production-code-requirements)
4. [Deployment Infrastructure](#4-deployment-infrastructure)
5. [Monitoring & Alerting](#5-monitoring--alerting)
6. [Risk Management Systems](#6-risk-management-systems)
7. [Pre-Launch Checklist](#7-pre-launch-checklist)
8. [Post-Launch Monitoring](#8-post-launch-monitoring)
9. [Options Trading Requirements](#9-options-trading-requirements)
---

## 1. Mandatory Development Best Practices

> ⚠️ **CRITICAL**: These practices are NON-NEGOTIABLE during development. Backtests that ignore these requirements will produce misleading results and WILL fail in production.

### 1.1 Transaction Cost Modeling

**Every backtest MUST include realistic transaction costs.**

| Cost Component | Minimum Model | Recommended Model |
|----------------|---------------|-------------------|
| **Commission** | $0 (Alpaca is commission-free) | $0 |
| **SEC Fee** | $0.0000278 per $ sold | Include in slippage |
| **FINRA TAF** | $0.000166 per share (max $8.30) | Include in slippage |
| **Bid-Ask Spread** | 0.05% per trade | 0.10% for < $50 stocks |
| **Slippage** | 0.05% per trade | 0.10-0.20% for larger orders |

**VectorBT Pro Implementation:**
```python
# MANDATORY: Include in ALL backtests
from vectorbtpro import Portfolio

# Minimum acceptable cost model
pf = Portfolio.from_signals(
    close=close_prices,
    entries=entries,
    exits=exits,
    fees=0.001,  # 0.10% total (spread + slippage combined)
    slippage=0.001,  # Additional 0.10% slippage
    freq='1D'
)

# Recommended: Separate spread and slippage
pf = Portfolio.from_signals(
    close=close_prices,
    entries=entries,
    exits=exits,
    fees=0.0005,      # 0.05% for spread
    slippage=0.001,   # 0.10% for market impact
    freq='1D'
)

# For intraday strategies (ORB, STRAT): Use higher costs
pf_intraday = Portfolio.from_signals(
    close=close_prices,
    entries=entries,
    exits=exits,
    fees=0.001,       # 0.10% spread (wider intraday)
    slippage=0.002,   # 0.20% slippage (faster execution needed)
    freq='1Min'
)
```

**Cost Sensitivity Analysis (REQUIRED):**
```python
def cost_sensitivity_analysis(close, entries, exits):
    """
    Run backtest across multiple cost assumptions.
    Strategy must remain profitable at pessimistic costs.
    """
    cost_scenarios = {
        'optimistic': {'fees': 0.0005, 'slippage': 0.0005},
        'realistic': {'fees': 0.001, 'slippage': 0.001},
        'pessimistic': {'fees': 0.0015, 'slippage': 0.002},
        'worst_case': {'fees': 0.002, 'slippage': 0.003},
    }
    
    results = {}
    for scenario, costs in cost_scenarios.items():
        pf = Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            **costs
        )
        results[scenario] = {
            'total_return': pf.total_return,
            'sharpe': pf.sharpe_ratio,
            'max_dd': pf.max_drawdown,
            'win_rate': pf.trades.win_rate
        }
    
    return pd.DataFrame(results).T

# ACCEPTANCE CRITERIA:
# - Strategy must be profitable at 'pessimistic' costs
# - Sharpe ratio must remain > 0.5 at 'pessimistic' costs
# - If only profitable at 'optimistic' costs, REJECT strategy
```

### 1.2 Slippage Modeling

**Slippage is the difference between expected and actual execution price.**

**Causes of Slippage:**
1. Market orders execute at next available price
2. Large orders move the market
3. Fast-moving markets during news/volatility
4. Delays between signal and execution

**Slippage Model by Order Size:**
```python
def calculate_realistic_slippage(order_value: float, avg_daily_volume: float, volatility: float) -> float:
    """
    Calculate realistic slippage based on order characteristics.
    
    Args:
        order_value: Dollar value of order
        avg_daily_volume: Average daily dollar volume of stock
        volatility: 20-day realized volatility (annualized)
    
    Returns:
        Expected slippage as decimal (0.001 = 0.1%)
    """
    # Base slippage (market microstructure)
    base_slippage = 0.0005  # 0.05% minimum
    
    # Market impact: larger orders relative to volume = more slippage
    participation_rate = order_value / avg_daily_volume
    market_impact = participation_rate * 0.1  # 10% of participation rate
    
    # Volatility adjustment: higher vol = more slippage
    vol_adjustment = (volatility / 0.20) * 0.0005  # Scaled to 20% baseline vol
    
    total_slippage = base_slippage + market_impact + vol_adjustment
    
    # Cap at reasonable maximum
    return min(total_slippage, 0.02)  # Max 2% slippage

# Example usage in backtest:
# $10,000 order in stock with $50M daily volume and 25% volatility
slippage = calculate_realistic_slippage(10000, 50_000_000, 0.25)
# Result: ~0.12% slippage
```

**VectorBT Pro Slippage Modes:**
```python
# Fixed percentage slippage (simplest)
pf = Portfolio.from_signals(
    ...,
    slippage=0.001  # 0.1% on every trade
)

# Variable slippage based on volatility (better)
# Calculate per-bar slippage based on ATR
atr = talib.ATR(high, low, close, timeperiod=14)
slippage_pct = (atr / close) * 0.5  # 50% of ATR as slippage

pf = Portfolio.from_signals(
    ...,
    slippage=slippage_pct  # Array of per-bar slippage
)
```

### 1.3 Partial Fill Simulation

**In production, orders may only partially fill. Backtests must account for this.**

**When Partial Fills Occur:**
- Large orders relative to available liquidity
- Fast-moving markets
- Limit orders at edge of spread
- Low-volume stocks

**Partial Fill Model:**
```python
def simulate_partial_fills(
    order_shares: int,
    avg_minute_volume: float,
    order_type: str = 'market'
) -> float:
    """
    Estimate fill rate for an order.
    
    Returns:
        Expected fill rate (0.0 to 1.0)
    """
    # Market orders almost always fill, but may take multiple prints
    if order_type == 'market':
        if order_shares < avg_minute_volume * 0.1:
            return 1.0  # Small order, full fill
        elif order_shares < avg_minute_volume * 0.5:
            return 0.95  # Medium order, likely full fill
        else:
            return 0.80  # Large order, may have issues
    
    # Limit orders have lower fill rates
    elif order_type == 'limit':
        if order_shares < avg_minute_volume * 0.1:
            return 0.85  # Small limit order
        elif order_shares < avg_minute_volume * 0.5:
            return 0.60  # Medium limit order
        else:
            return 0.40  # Large limit order
    
    return 1.0

# In backtesting, apply fill rate to position sizing
def adjusted_position_size(target_shares: int, symbol: str, data: pd.DataFrame) -> int:
    """Adjust position size for expected partial fills."""
    avg_volume = data['volume'].rolling(20).mean().iloc[-1]
    fill_rate = simulate_partial_fills(target_shares, avg_volume / 390)  # 390 minutes/day
    
    # If fill rate < 80%, reduce position to what we expect to get
    if fill_rate < 0.8:
        return int(target_shares * fill_rate)
    return target_shares
```

**Liquidity Filter (MANDATORY):**
```python
def liquidity_filter(symbols: list, min_dollar_volume: float = 10_000_000) -> list:
    """
    Filter out illiquid stocks that will have execution problems.
    
    MANDATORY: Apply this filter BEFORE backtesting.
    """
    liquid_symbols = []
    
    for symbol in symbols:
        # Get 20-day average dollar volume
        bars = get_historical_bars(symbol, days=30)
        avg_dollar_volume = (bars['close'] * bars['volume']).rolling(20).mean().iloc[-1]
        
        if avg_dollar_volume >= min_dollar_volume:
            liquid_symbols.append(symbol)
        else:
            logger.debug(f"Filtered out {symbol}: ${avg_dollar_volume:,.0f} avg volume")
    
    return liquid_symbols

# Minimum thresholds by strategy type:
# - Swing trading (daily): $10M average daily dollar volume
# - Day trading (intraday): $50M average daily dollar volume  
# - Options strategies: $20M underlying + 1000 OI on strikes
```

### 1.4 Look-Ahead Bias Prevention

**Look-ahead bias occurs when your backtest uses information that wouldn't be available at decision time.**

**Common Sources of Look-Ahead Bias:**

| Bias Type | Example | Solution |
|-----------|---------|----------|
| **Future price in signal** | Using close price to generate signal for that bar | Use previous bar's close |
| **Point-in-time data** | Using final earnings (restated) vs. initial release | Use as-reported data |
| **Indicator calculation** | RSI using full dataset | Calculate incrementally |
| **Rebalancing timing** | Rebalancing at exact close | Add realistic delay |

**Correct Signal Generation:**
```python
# WRONG: Using current bar's close to decide current bar's trade
def bad_signal_generation(data):
    sma = data['close'].rolling(20).mean()
    # This signal uses today's close to trade today - LOOK-AHEAD BIAS
    signal = data['close'] > sma
    return signal

# CORRECT: Signal based on previous bar, execution on current bar
def correct_signal_generation(data):
    sma = data['close'].rolling(20).mean()
    # Signal from yesterday, trade today
    signal = data['close'].shift(1) > sma.shift(1)
    return signal

# CORRECT for intraday: Signal on bar close, execution on NEXT bar open
def correct_intraday_signal(data):
    # Calculate indicator at bar close
    rsi = talib.RSI(data['close'], timeperiod=14)
    
    # Signal generated at close of bar N
    raw_signal = rsi < 30
    
    # Trade executes at open of bar N+1
    # Shift signal forward by 1 bar
    tradeable_signal = raw_signal.shift(1)
    
    # Use NEXT bar's open as execution price, not current close
    execution_price = data['open'].shift(-1)  # This is for backtest reference only
    
    return tradeable_signal
```

**VectorBT Pro Correct Usage:**
```python
# CORRECT: entries/exits are based on data available at decision time
# VectorBT executes on the NEXT bar after signal by default

# For end-of-day strategies:
entries = (close.shift(1) > sma.shift(1)) & (close.shift(2) <= sma.shift(2))
# Signal based on yesterday's data, execute at today's close

# For intraday with next-bar execution:
pf = Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    upon_opposite_entry='close',  # Close existing position
    accumulate=False,
    # VectorBT executes on the bar where signal is True
    # Ensure your signal is shifted appropriately
)
```

### 1.5 Survivorship Bias Prevention

**Survivorship bias occurs when backtests only include stocks that still exist today, ignoring delisted companies.**

**Impact of Survivorship Bias:**
- Overestimates returns by 1-2% annually
- Particularly affects momentum strategies (winners survive, losers delist)
- Small-cap strategies most affected

**Mitigation Strategies:**
```python
# Option 1: Use survivorship-bias-free data sources
# - CRSP (academic gold standard)
# - Tiingo (includes delisted stocks with proper end dates)
# - Norgate Data (includes delisted with full history)

# Option 2: Point-in-time universe construction
def get_universe_at_date(target_date: date, index: str = 'SPY') -> list:
    """
    Get the actual constituents of an index at a historical date.
    NOT the current constituents applied backwards.
    """
    # This requires historical constituent data
    # Sources: Bloomberg, FactSet, or reconstruct from SEC filings
    pass

# Option 3: Apply delisting returns
def apply_delisting_returns(returns: pd.Series, delisting_info: pd.DataFrame) -> pd.Series:
    """
    Apply delisting returns to stocks that were delisted.
    Delisted stocks often have -30% to -100% final returns.
    """
    adjusted_returns = returns.copy()
    
    for symbol, info in delisting_info.iterrows():
        if info['delisting_reason'] == 'bankruptcy':
            # Total loss
            adjusted_returns.loc[symbol, info['delist_date']:] = -1.0
        elif info['delisting_reason'] == 'merger':
            # Usually positive, use actual merger premium
            adjusted_returns.loc[symbol, info['delist_date']] = info['merger_premium']
        elif info['delisting_reason'] == 'going_private':
            # Usually positive premium
            adjusted_returns.loc[symbol, info['delist_date']] = info['premium']
    
    return adjusted_returns

# MINIMUM REQUIREMENT: Document universe construction method
# State explicitly: "Universe is current S&P 500 constituents" 
# and acknowledge this introduces ~1-2% annual survivorship bias
```

### 1.6 Walk-Forward Validation (MANDATORY)

**Every strategy MUST pass walk-forward validation before consideration for production.**

**Walk-Forward Protocol:**
```python
def walk_forward_validation(
    data: pd.DataFrame,
    strategy_func: callable,
    train_period: int = 252,  # 1 year training
    test_period: int = 63,    # 3 months testing
    min_trades_per_fold: int = 10
) -> dict:
    """
    Mandatory walk-forward validation for all strategies.
    
    Returns validation results including:
    - Out-of-sample Sharpe ratio
    - In-sample vs out-of-sample degradation
    - Parameter stability across folds
    """
    results = {
        'folds': [],
        'is_sharpes': [],
        'oos_sharpes': [],
        'is_returns': [],
        'oos_returns': [],
        'parameters': []
    }
    
    total_bars = len(data)
    current_idx = train_period
    fold_num = 0
    
    while current_idx + test_period <= total_bars:
        fold_num += 1
        
        # Define train/test windows
        train_start = current_idx - train_period
        train_end = current_idx
        test_start = current_idx
        test_end = current_idx + test_period
        
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Optimize on training data
        best_params, is_metrics = strategy_func.optimize(train_data)
        
        # Test on out-of-sample data (NO REOPTIMIZATION)
        oos_metrics = strategy_func.backtest(test_data, params=best_params)
        
        # Validate minimum trade count
        if oos_metrics['trade_count'] < min_trades_per_fold:
            logger.warning(f"Fold {fold_num}: Only {oos_metrics['trade_count']} trades")
        
        # Store results
        results['folds'].append(fold_num)
        results['is_sharpes'].append(is_metrics['sharpe'])
        results['oos_sharpes'].append(oos_metrics['sharpe'])
        results['is_returns'].append(is_metrics['total_return'])
        results['oos_returns'].append(oos_metrics['total_return'])
        results['parameters'].append(best_params)
        
        # Move to next fold
        current_idx += test_period
    
    # Calculate summary statistics
    results['summary'] = {
        'avg_is_sharpe': np.mean(results['is_sharpes']),
        'avg_oos_sharpe': np.mean(results['oos_sharpes']),
        'sharpe_degradation': 1 - (np.mean(results['oos_sharpes']) / np.mean(results['is_sharpes'])),
        'oos_sharpe_std': np.std(results['oos_sharpes']),
        'param_stability': calculate_param_stability(results['parameters']),
        'total_folds': fold_num,
        'profitable_folds_pct': sum(1 for r in results['oos_returns'] if r > 0) / fold_num
    }
    
    return results

# ACCEPTANCE CRITERIA (ALL must pass):
# 1. OOS Sharpe degradation < 30% (ideally < 20%)
# 2. OOS Sharpe > 0.5 average across folds
# 3. Parameter stability σ < 20% of mean
# 4. > 60% of folds profitable
# 5. No single fold with drawdown > 25%
```

### 1.7 Monte Carlo Simulation (MANDATORY)

**Monte Carlo simulation tests strategy robustness to random variations.**

```python
def monte_carlo_validation(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    confidence_level: float = 0.95
) -> dict:
    """
    Run Monte Carlo simulation on trade sequence.
    Tests robustness to trade ordering and random variation.
    """
    original_equity = calculate_equity_curve(trades)
    original_sharpe = calculate_sharpe(original_equity)
    original_max_dd = calculate_max_drawdown(original_equity)
    
    simulated_sharpes = []
    simulated_max_dds = []
    simulated_final_returns = []
    
    for _ in range(n_simulations):
        # Shuffle trade order (bootstrap)
        shuffled_trades = trades.sample(frac=1, replace=True)
        
        # Calculate metrics on shuffled sequence
        sim_equity = calculate_equity_curve(shuffled_trades)
        simulated_sharpes.append(calculate_sharpe(sim_equity))
        simulated_max_dds.append(calculate_max_drawdown(sim_equity))
        simulated_final_returns.append(sim_equity.iloc[-1] / sim_equity.iloc[0] - 1)
    
    # Calculate confidence intervals
    sharpe_ci = np.percentile(simulated_sharpes, [(1-confidence_level)*100/2, 100 - (1-confidence_level)*100/2])
    dd_ci = np.percentile(simulated_max_dds, [5, 95])
    return_ci = np.percentile(simulated_final_returns, [5, 95])
    
    return {
        'original_sharpe': original_sharpe,
        'simulated_sharpe_mean': np.mean(simulated_sharpes),
        'simulated_sharpe_std': np.std(simulated_sharpes),
        'sharpe_95_ci': sharpe_ci,
        'original_max_dd': original_max_dd,
        'simulated_max_dd_95': np.percentile(simulated_max_dds, 95),
        'max_dd_95_ci': dd_ci,
        'return_95_ci': return_ci,
        'probability_of_loss': sum(1 for r in simulated_final_returns if r < 0) / n_simulations,
        'probability_of_ruin': sum(1 for dd in simulated_max_dds if dd > 0.50) / n_simulations
    }

# ACCEPTANCE CRITERIA:
# 1. 95% CI for Sharpe does not include 0
# 2. Probability of loss < 20%
# 3. Probability of ruin (>50% DD) < 5%
# 4. Simulated mean Sharpe within 20% of original
```

### 1.8 Overfitting Prevention

**The 50% Haircut Rule**: Assume your out-of-sample performance will be 50% worse than your in-sample performance.

```python
def apply_overfitting_haircut(backtest_results: dict, haircut: float = 0.50) -> dict:
    """
    Apply realistic overfitting adjustment to backtest results.
    
    Industry standard: Expect 50% degradation from backtest to live.
    """
    adjusted = {}
    
    # Metrics that degrade
    adjusted['expected_annual_return'] = backtest_results['annual_return'] * (1 - haircut)
    adjusted['expected_sharpe'] = backtest_results['sharpe_ratio'] * (1 - haircut)
    adjusted['expected_win_rate'] = 0.5 + (backtest_results['win_rate'] - 0.5) * (1 - haircut)
    
    # Metrics that get worse (drawdowns increase)
    adjusted['expected_max_drawdown'] = backtest_results['max_drawdown'] * (1 + haircut)
    
    # Calculate if still viable
    adjusted['still_viable'] = (
        adjusted['expected_sharpe'] > 0.5 and
        adjusted['expected_annual_return'] > 0.08 and
        adjusted['expected_max_drawdown'] < 0.30
    )
    
    return adjusted

# RED FLAGS - Strategy likely overfit if:
# - Backtest Sharpe > 3.0
# - Win rate > 70%
# - No losing months
# - Works only on specific time period
# - Requires many parameters (>5)
# - Sensitive to small parameter changes
```

---

## 2. Realistic Backtesting Requirements

### 2.1 Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **History Length** | 5 years (2019-2024) | 15-20 years |
| **Must Include** | COVID crash (2020) | 2008 GFC, 2022 bear |
| **Data Quality** | Adjusted for splits/dividends | Corporate actions verified |
| **Frequency** | Daily for swing | 1-minute for intraday |

### 2.2 Backtest Configuration Template

```python
# MANDATORY BACKTEST CONFIGURATION
# Copy this template for all strategy backtests

BACKTEST_CONFIG = {
    # Transaction Costs (MANDATORY)
    'fees': 0.001,           # 0.10% spread
    'slippage': 0.001,       # 0.10% slippage
    
    # Capital & Position Sizing
    'init_cash': 10_000,     # Match your actual capital
    'size_type': 'percent',  # Percent of equity
    'size': 0.10,            # Max 10% per position
    
    # Execution Assumptions
    'accumulate': False,     # No pyramiding unless explicit
    'allow_partial': True,   # Allow partial fills
    
    # Risk Management
    'sl_stop': 0.02,         # 2% stop loss
    'tp_stop': None,         # Strategy-specific take profit
    
    # Validation Periods
    'train_start': '2010-01-01',
    'train_end': '2019-12-31',
    'test_start': '2020-01-01',
    'test_end': '2024-12-31',
    
    # Walk-Forward Settings
    'wf_train_period': 252,  # 1 year
    'wf_test_period': 63,    # 3 months
    
    # Monte Carlo Settings
    'mc_simulations': 1000,
    'mc_confidence': 0.95,
}

def run_compliant_backtest(strategy, data, config=BACKTEST_CONFIG):
    """Run a backtest that meets all mandatory requirements."""
    
    # 1. Apply liquidity filter
    liquid_symbols = liquidity_filter(data.columns, min_dollar_volume=10_000_000)
    data = data[liquid_symbols]
    
    # 2. Generate signals (with proper shifting to avoid look-ahead)
    signals = strategy.generate_signals(data)
    
    # 3. Run backtest with realistic costs
    pf = vbt.Portfolio.from_signals(
        close=data,
        entries=signals['entries'],
        exits=signals['exits'],
        fees=config['fees'],
        slippage=config['slippage'],
        init_cash=config['init_cash'],
        size=config['size'],
        size_type=config['size_type'],
        sl_stop=config['sl_stop'],
        accumulate=config['accumulate']
    )
    
    # 4. Run cost sensitivity analysis
    cost_sensitivity = cost_sensitivity_analysis(data, signals['entries'], signals['exits'])
    
    # 5. Run walk-forward validation
    wf_results = walk_forward_validation(
        data, strategy,
        train_period=config['wf_train_period'],
        test_period=config['wf_test_period']
    )
    
    # 6. Run Monte Carlo simulation
    mc_results = monte_carlo_validation(pf.trades.records, n_simulations=config['mc_simulations'])
    
    # 7. Apply overfitting haircut
    adjusted_expectations = apply_overfitting_haircut({
        'annual_return': pf.annual_return,
        'sharpe_ratio': pf.sharpe_ratio,
        'win_rate': pf.trades.win_rate,
        'max_drawdown': pf.max_drawdown
    })
    
    return {
        'portfolio': pf,
        'cost_sensitivity': cost_sensitivity,
        'walk_forward': wf_results,
        'monte_carlo': mc_results,
        'adjusted_expectations': adjusted_expectations,
        'passes_validation': _check_acceptance_criteria(wf_results, mc_results, adjusted_expectations)
    }
```

### 2.3 Acceptance Criteria Summary

| Metric | Threshold | Hard Fail |
|--------|-----------|-----------|
| **OOS Sharpe** | > 0.5 | < 0.3 |
| **Sharpe Degradation** | < 30% | > 50% |
| **Profitable at Pessimistic Costs** | Required | No |
| **Monte Carlo P(Loss)** | < 20% | > 35% |
| **Monte Carlo P(Ruin)** | < 5% | > 15% |
| **Max Drawdown (expected)** | < 25% | > 35% |
| **Walk-Forward Efficiency** | > 50% | < 30% |
| **Minimum Trades** | > 100 | < 30 |

---

## 3. Production Code Requirements

### 3.1 Order Execution

**Every production order MUST have:**

```python
class ProductionOrder:
    """Minimum requirements for production orders."""
    
    # REQUIRED fields
    symbol: str
    side: Literal['buy', 'sell']
    qty: int
    order_type: Literal['market', 'limit', 'stop', 'stop_limit']
    time_in_force: Literal['day', 'gtc', 'ioc', 'fok']
    
    # REQUIRED for limit orders
    limit_price: Optional[float]
    
    # REQUIRED metadata
    strategy_id: str          # Which strategy generated this
    signal_timestamp: datetime # When signal was generated
    
    # REQUIRED risk checks (validated before submission)
    position_size_pct: float  # Must be < MAX_POSITION_SIZE
    portfolio_heat_after: float  # Must be < MAX_PORTFOLIO_HEAT
    
    # REQUIRED handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
```

**Order State Machine:**
```
CREATED → SUBMITTED → ACCEPTED → [PARTIAL_FILL] → FILLED
                  ↓                    ↓            ↓
              REJECTED              CANCELED    COMPLETED
                  ↓                    ↓
               ALERT              RECONCILE
```

### 3.2 State Management

**Required State Persistence:**

```python
# All of these MUST survive restarts:

PERSISTENT_STATE = {
    # Position tracking
    'open_positions': {},      # {symbol: {entry_price, qty, strategy, stop_loss, take_profit}}
    
    # Order tracking
    'pending_orders': {},      # Orders submitted but not filled
    'today_orders': [],        # All orders today (for daily limits)
    
    # Risk tracking
    'daily_pnl': 0.0,          # Reset at market open
    'consecutive_losses': 0,
    'portfolio_heat': 0.0,
    
    # Strategy state
    'regime': 'NORMAL',        # Current market regime
    'last_regime_check': None,
    
    # System state
    'last_heartbeat': None,
    'startup_time': None,
    'circuit_breaker_status': False,
}
```

### 3.3 Error Handling

**Required Error Categories:**

```python
class TradingError(Exception):
    """Base class for trading errors."""
    pass

class OrderRejectedError(TradingError):
    """Order was rejected by broker."""
    # Action: Log, alert, DO NOT retry
    pass

class InsufficientFundsError(TradingError):
    """Not enough buying power."""
    # Action: Log, alert, reduce position size
    pass

class MarketClosedError(TradingError):
    """Attempted to trade when market closed."""
    # Action: Queue for next open or cancel
    pass

class DataQualityError(TradingError):
    """Bad data received."""
    # Action: Use cached data or skip signal
    pass

class RateLimitError(TradingError):
    """API rate limit hit."""
    # Action: Exponential backoff, retry
    pass

class ConnectionError(TradingError):
    """Network/API connection failed."""
    # Action: Retry with backoff
    pass

class CircuitBreakerError(TradingError):
    """Trading halted by circuit breaker."""
    # Action: Alert, wait for manual reset
    pass
```

### 3.4 Logging Requirements

**Every trade action MUST be logged:**

```python
# Required log format
LOG_FORMAT = {
    'timestamp': 'ISO8601',
    'level': 'DEBUG|INFO|WARNING|ERROR|CRITICAL',
    'component': 'strategy|execution|risk|data',
    'action': 'signal|order|fill|cancel|error',
    'details': {}  # Context-specific
}

# Required log events:
# - Strategy signal generated
# - Order submitted
# - Order filled (including fill price, slippage)
# - Order rejected (including reason)
# - Position opened
# - Position closed (including P&L)
# - Stop loss triggered
# - Circuit breaker tripped
# - System startup/shutdown
# - Error with stack trace
```

---

## 4. Deployment Infrastructure

### 4.1 VPS Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4 cores |
| **RAM** | 2 GB | 4 GB |
| **Storage** | 20 GB SSD | 50 GB SSD |
| **Network** | 100 Mbps | 1 Gbps |
| **Uptime SLA** | 99.9% | 99.99% |
| **Location** | US East | NYC/Virginia |

### 4.2 Process Management

**Required: Auto-restart on crash**

```yaml
# docker-compose.yml
services:
  atlas-trader:
    restart: always
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 60s
      timeout: 10s
      retries: 3
```

OR

```ini
# systemd service
[Service]
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=3
```

### 4.3 Environment Configuration

```bash
# .env file (NEVER commit to git)

# Alpaca Credentials
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Change for live

# Monitoring
DISCORD_WEBHOOK_URL=
HEALTHCHECK_PING_URL=

# Risk Limits
MAX_DAILY_LOSS_PCT=5.0
MAX_POSITION_SIZE_PCT=10.0
MAX_PORTFOLIO_HEAT_PCT=8.0
MAX_CONSECUTIVE_LOSSES=5

# System
LOG_LEVEL=INFO
ENVIRONMENT=paper  # paper or live
```

---

## 5. Monitoring & Alerting

### 5.1 Required Alerts

| Event | Severity | Channel | Response Time |
|-------|----------|---------|---------------|
| System startup | INFO | Discord | - |
| System shutdown | WARNING | Discord | - |
| Order filled | INFO | Log only | - |
| Order rejected | WARNING | Discord | 1 hour |
| Position opened | INFO | Discord | - |
| Stop loss triggered | WARNING | Discord | - |
| Daily loss > 3% | WARNING | Discord | 15 min |
| Daily loss > 5% | CRITICAL | Discord + SMS | 5 min |
| Circuit breaker | CRITICAL | Discord + SMS | Immediate |
| Connection lost | ERROR | Discord | 5 min |
| No heartbeat (5 min) | CRITICAL | External monitor | Immediate |

### 5.2 Daily Reports

**Automated daily report at 4:30 PM ET:**

```
📊 ATLAS Daily Report - 2025-11-26

Portfolio Value: $10,523.45 (+$123.45 / +1.19%)
Day's P&L: +$87.23 (+0.84%)
Open Positions: 3
Trades Today: 5 (3 wins, 2 losses)

Position Summary:
├── AAPL: +2.3% (50 shares @ $198.50)
├── MSFT: -0.8% (25 shares @ $412.30)
└── NVDA: +1.1% (10 shares @ $875.20)

Risk Status:
├── Portfolio Heat: 4.2% (limit: 8%)
├── Max Position: 8.5% (limit: 10%)
└── Daily Loss: -0.84% (limit: -5%)

System Status: ✅ Healthy
├── Uptime: 12h 34m
├── Last Heartbeat: 2 min ago
└── Circuit Breaker: OFF
```

---

## 6. Risk Management Systems

### 6.1 Circuit Breaker Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Daily loss | > 5% | Halt all trading |
| Consecutive losses | ≥ 5 | Halt all trading |
| Failed orders | ≥ 3 in 1 hour | Halt all trading |
| Abnormal spread | > 2% | Skip symbol |
| No data (5 min) | During market hours | Alert + pause |
| Memory usage | > 80% | Alert |
| Manual trigger | Anytime | Immediate halt |

### 6.2 Position Limits

```python
POSITION_LIMITS = {
    # Per-position limits
    'max_position_pct': 10.0,      # Max 10% of portfolio in one stock
    'max_position_value': 5000,    # Max $5,000 per position (for small accounts)
    
    # Portfolio limits
    'max_portfolio_heat': 8.0,     # Max 8% total at risk
    'max_open_positions': 10,      # Max 10 concurrent positions
    'max_sector_exposure': 30.0,   # Max 30% in one sector
    
    # Daily limits
    'max_daily_trades': 50,        # Prevent runaway trading
    'max_daily_loss_pct': 5.0,     # Stop trading if down 5%
    
    # Per-trade limits
    'min_position_value': 500,     # Minimum position size
    'max_spread_pct': 1.0,         # Don't trade if spread > 1%
}
```

### 6.3 Kill Switch

```python
class KillSwitch:
    """
    Emergency stop for all trading activity.
    Can be triggered manually or automatically.
    """
    
    def __init__(self, alpaca_client, alert_manager):
        self.api = alpaca_client
        self.alerts = alert_manager
        self._is_active = False
    
    async def activate(self, reason: str):
        """Activate kill switch - liquidate all positions."""
        self._is_active = True
        
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        await self.alerts.send(f"🚨 KILL SWITCH: {reason}", level="critical")
        
        # Cancel all open orders
        await self.api.cancel_all_orders()
        
        # Liquidate all positions
        positions = await self.api.get_positions()
        for position in positions:
            await self.api.submit_order(
                symbol=position.symbol,
                qty=abs(int(position.qty)),
                side='sell' if float(position.qty) > 0 else 'buy',
                type='market',
                time_in_force='day'
            )
        
        await self.alerts.send(
            f"Liquidated {len(positions)} positions",
            level="critical"
        )
```

---

## 7. Pre-Launch Checklist

### 7.1 Strategy Validation (Must Pass ALL)

- [ ] Walk-forward validation: OOS Sharpe degradation < 30%
- [ ] Monte Carlo: P(Loss) < 20%, P(Ruin) < 5%
- [ ] Cost sensitivity: Profitable at pessimistic costs
- [ ] Overfitting check: Expected Sharpe > 0.5 after 50% haircut
- [ ] Minimum 100 trades in backtest
- [ ] Tested across multiple market regimes (bull, bear, sideways)
- [ ] No look-ahead bias (verified via code review)
- [ ] Liquidity filter applied to universe

### 7.2 Code Quality (Must Pass ALL)

- [ ] All error types handled with appropriate responses
- [ ] Retry logic with exponential backoff for network calls
- [ ] State persistence implemented and tested
- [ ] Startup reconciliation with broker verified
- [ ] Graceful shutdown implemented
- [ ] All secrets in environment variables (not hardcoded)
- [ ] Logging covers all trade actions
- [ ] Unit tests pass with > 80% coverage
- [ ] Integration tests pass

### 7.3 Infrastructure (Must Pass ALL)

- [ ] VPS provisioned with auto-restart enabled
- [ ] Docker container builds and runs successfully
- [ ] Health check endpoint functional
- [ ] Heartbeat monitoring configured
- [ ] Discord/Telegram alerts working
- [ ] Log rotation configured
- [ ] Backup/restore procedure documented
- [ ] Firewall configured (only necessary ports open)

### 7.4 Paper Trading (Must Complete)

- [ ] Minimum 4 weeks of paper trading
- [ ] Minimum 50 trades executed
- [ ] Paper trading P&L within 30% of backtest expectation
- [ ] No unexpected errors or crashes
- [ ] Order execution matches expectations
- [ ] All alert types triggered and received
- [ ] System restart tested (verify state recovery)
- [ ] Circuit breaker tested
- [ ] Kill switch tested

### 7.5 Go-Live Approval

- [ ] All above sections completed and documented
- [ ] Risk limits configured for live account
- [ ] Live API credentials secured
- [ ] Initial capital deposited
- [ ] Emergency procedures documented
- [ ] Rollback plan documented
- [ ] First week monitoring plan in place

---

## 8. Post-Launch Monitoring

### 8.1 First Week Protocol

| Day | Focus | Actions |
|-----|-------|---------|
| 1 | Watch everything | Monitor all trades in real-time |
| 2-3 | Execution quality | Compare fills to expected prices |
| 4-5 | System stability | Verify no memory leaks, crashes |
| 6-7 | Performance check | Compare to paper trading baseline |

### 8.2 Ongoing Monitoring

**Daily:**
- Review daily P&L report
- Check for any errors in logs
- Verify heartbeat is healthy
- Review any alerts triggered

**Weekly:**
- Compare live performance to backtest expectations
- Review execution quality (slippage analysis)
- Check resource usage trends
- Review and address any warnings

**Monthly:**
- Full performance review vs. benchmarks
- Parameter stability check
- Strategy correlation analysis
- Consider reoptimization if needed

### 8.3 Performance Degradation Response

| Degradation | Threshold | Response |
|-------------|-----------|----------|
| Sharpe ratio | < 50% of expected | Investigate, consider pause |
| Win rate | < 40% | Review recent trades for pattern |
| Max drawdown | > 150% of expected | Reduce position sizes |
| Execution cost | > 2x backtest | Review order types, timing |

# Section 9: Options Trading Requirements

> **CRITICAL**: Options trading introduces significantly different risks and complexities compared to equity trading. This section defines mandatory requirements specific to options strategies including STRAT pattern-based options execution.

---

## 9.1 Options-Specific Backtesting Requirements

### 9.1.1 Bid-Ask Spread Modeling (CRITICAL)

**Options spreads are 10-100x wider than equities. This is the #1 source of backtest-to-live performance degradation.**

| Underlying Price | Typical Equity Spread | Typical Option Spread | Spread as % of Premium |
|-----------------|----------------------|----------------------|----------------------|
| $100 stock | $0.01-0.03 (0.01-0.03%) | $0.05-0.20 | 2-10% of premium |
| $500 stock | $0.05-0.10 (0.01-0.02%) | $0.10-0.50 | 1-5% of premium |
| SPY/QQQ | $0.01 (0.002%) | $0.02-0.10 | 0.5-3% of premium |

**Minimum Spread Model for Backtesting:**
```python
def calculate_options_spread_cost(option_price: float, underlying_price: float, 
                                   dte: int, moneyness: float) -> float:
    """
    Calculate realistic bid-ask spread for options.
    
    Args:
        option_price: Mid-price of option
        underlying_price: Current stock price
        dte: Days to expiration
        moneyness: strike/underlying (1.0 = ATM, <1.0 = ITM call)
    
    Returns:
        Expected spread as decimal of option price
    """
    # Base spread - even liquid options have minimum spread
    base_spread = 0.02  # 2% minimum
    
    # DTE adjustment: shorter dated = wider spreads
    if dte < 7:
        dte_adj = 0.03  # Near expiration = wider
    elif dte < 21:
        dte_adj = 0.01  # Sweet spot
    else:
        dte_adj = 0.02  # Longer dated = less liquid
    
    # Moneyness adjustment: OTM options have wider spreads
    otm_distance = abs(moneyness - 1.0)
    moneyness_adj = otm_distance * 0.10  # 10% of OTM distance
    
    # Price adjustment: cheap options have wider relative spreads
    if option_price < 1.0:
        price_adj = 0.05  # $0.05 spread on $1 option = 5%
    elif option_price < 5.0:
        price_adj = 0.02
    else:
        price_adj = 0.01
    
    total_spread = base_spread + dte_adj + moneyness_adj + price_adj
    
    # Cap at 20% (options with >20% spread should not be traded)
    return min(total_spread, 0.20)

# MANDATORY: Apply in all options backtests
# Half spread on entry, half spread on exit = full spread round-trip
def apply_options_spread(entry_price: float, exit_price: float, spread_pct: float) -> tuple:
    """Apply bid-ask spread to entry and exit prices."""
    # Entry: pay ask (mid + half spread)
    adjusted_entry = entry_price * (1 + spread_pct / 2)
    # Exit: receive bid (mid - half spread)  
    adjusted_exit = exit_price * (1 - spread_pct / 2)
    return adjusted_entry, adjusted_exit
```

**VectorBT Pro Options Cost Model:**
```python
# MANDATORY for all options backtests
pf = vbt.Portfolio.from_signals(
    close=option_prices,
    entries=entries,
    exits=exits,
    fees=0.0065,      # Per-contract commission ($0.65) as % of typical premium
    slippage=0.03,    # 3% slippage (MINIMUM for options)
    # OR use per-trade slippage array calculated above
)

# For cheap options (<$2 premium): Use higher slippage
pf_cheap_options = vbt.Portfolio.from_signals(
    close=option_prices,
    entries=entries,
    exits=exits,
    fees=0.0065,
    slippage=0.05,    # 5% for cheap options
)
```

### 9.1.2 Implied Volatility (IV) Modeling

**Backtests using historical IV must account for the volatility risk premium.**

| IV Scenario | Adjustment | Use Case |
|-------------|-----------|----------|
| Historical IV | Actual IV at time | Most realistic |
| Constant IV | Use 20-30 IV for equities | Simplified model |
| IV Crush | Model post-earnings IV drop | Event strategies |

**IV Crush Modeling (REQUIRED for earnings strategies):**
```python
def model_iv_crush(current_iv: float, event_type: str, dte_to_event: int) -> float:
    """
    Model expected IV behavior around events.
    
    Returns:
        Post-event IV estimate
    """
    crush_factors = {
        'earnings': 0.40,      # IV typically drops 40-60% post-earnings
        'fed_meeting': 0.20,   # Fed events: 20-30% crush
        'product_launch': 0.30 # Major announcements
    }
    
    base_crush = crush_factors.get(event_type, 0.0)
    
    # IV builds into event - more crush if entering close to event
    if dte_to_event <= 1:
        crush_mult = 1.0  # Full crush
    elif dte_to_event <= 5:
        crush_mult = 0.7  # Most of crush
    else:
        crush_mult = 0.4  # Partial crush
    
    post_event_iv = current_iv * (1 - base_crush * crush_mult)
    return max(post_event_iv, 0.10)  # Floor at 10% IV

# Example: Backtest must use post-event IV for exit pricing
entry_iv = 0.45  # 45% IV before earnings
exit_iv = model_iv_crush(entry_iv, 'earnings', dte_to_event=1)
# exit_iv = ~0.27 (40% crush)
```

### 9.1.3 Greeks-Based Position Sizing

**Options position sizing must account for delta exposure, not notional value.**

```python
def calculate_options_position_size(
    account_value: float,
    option_delta: float,
    option_price: float,
    underlying_price: float,
    max_delta_exposure: float = 0.10,  # Max 10% delta exposure
    max_premium_risk: float = 0.02     # Max 2% premium at risk
) -> int:
    """
    Calculate position size based on delta exposure and premium risk.
    
    Returns:
        Number of contracts to trade
    """
    # Delta-based sizing: How much underlying exposure
    delta_per_contract = abs(option_delta) * 100 * underlying_price
    max_delta_value = account_value * max_delta_exposure
    contracts_by_delta = int(max_delta_value / delta_per_contract)
    
    # Premium-based sizing: Max loss if option goes to zero
    premium_per_contract = option_price * 100
    max_premium = account_value * max_premium_risk
    contracts_by_premium = int(max_premium / premium_per_contract)
    
    # Use the more conservative limit
    return min(contracts_by_delta, contracts_by_premium, 10)  # Cap at 10 contracts

# Example for ATLAS STRAT options
# $10,000 account, 0.45 delta call, $3.50 premium, $150 underlying
contracts = calculate_options_position_size(
    account_value=10000,
    option_delta=0.45,
    option_price=3.50,
    underlying_price=150.0,
    max_delta_exposure=0.10,
    max_premium_risk=0.02
)
# Result: min(1, 0) = 0 contracts (premium too expensive for small account)
# With $3k account: Need to use cheaper options or smaller max_premium_risk
```

### 9.1.4 Theta Decay Modeling

**Time decay is NOT linear. Backtests must model realistic theta decay curves.**

```python
def model_theta_decay(dte: int, option_type: str = 'atm') -> float:
    """
    Model daily theta decay as percentage of option value.
    
    ATM options decay faster as expiration approaches.
    """
    if option_type == 'atm':
        if dte > 30:
            daily_decay = 0.02   # ~2% daily decay
        elif dte > 14:
            daily_decay = 0.03   # ~3% daily decay
        elif dte > 7:
            daily_decay = 0.05   # ~5% daily decay (accelerating)
        elif dte > 3:
            daily_decay = 0.08   # ~8% daily decay
        else:
            daily_decay = 0.15   # ~15% daily decay (rapid)
    elif option_type == 'otm':
        # OTM options decay faster
        daily_decay = model_theta_decay(dte, 'atm') * 1.5
    else:  # itm
        # ITM options decay slower (more intrinsic value)
        daily_decay = model_theta_decay(dte, 'atm') * 0.5
    
    return daily_decay

# MANDATORY: Apply theta decay to held positions in backtest
def apply_holding_period_theta(entry_price: float, dte_at_entry: int, 
                                holding_days: int, option_type: str) -> float:
    """Calculate option value after holding period due to theta alone."""
    remaining_value = entry_price
    for day in range(holding_days):
        dte = dte_at_entry - day
        if dte <= 0:
            return 0.0  # Option expired worthless
        daily_decay = model_theta_decay(dte, option_type)
        remaining_value *= (1 - daily_decay)
    return remaining_value
```

---

## 9.2 Options-Specific Data Requirements

### 9.2.1 Historical Options Data Sources

| Source | Cost | Data Quality | Use Case |
|--------|------|--------------|----------|
| **ThetaData** | $80/mo Standard | High (tick-level) | Primary for ATLAS |
| OptionMetrics | $$$$ (institutional) | Highest | Academic research |
| CBOE DataShop | $$$ | High | Professional |
| Yahoo Finance | Free | Low (EOD only) | NOT for backtesting |

**ThetaData Integration (ATLAS Standard):**
```python
from integrations.thetadata_client import ThetaDataRESTClient

client = ThetaDataRESTClient()
if client.connect():
    # Get historical option chain
    expirations = client.get_expirations('SPY', min_dte=7, max_dte=45)
    
    # Get ATM strikes
    underlying_price = client.get_underlying_price('SPY')
    strikes = client.get_strikes('SPY', expirations[0])
    atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
    
    # Get historical quote with Greeks
    quote = client.get_quote(
        symbol='SPY',
        expiration=expirations[0],
        strike=atm_strike,
        right='call'
    )
    
    # MANDATORY: Verify data quality
    assert quote['bid'] > 0, "Invalid bid price"
    assert quote['ask'] > quote['bid'], "Inverted market"
    assert quote['ask'] - quote['bid'] < quote['mid'] * 0.10, "Spread too wide"
```

### 9.2.2 Greeks Calculation Requirements

**All options backtests MUST include Greeks calculations.**

| Greek | Purpose | Model Requirement |
|-------|---------|-------------------|
| **Delta** | Directional exposure | Black-Scholes or better |
| **Gamma** | Delta change rate | Required for intraday |
| **Theta** | Time decay | Required for all |
| **Vega** | IV sensitivity | Required for all |
| **Rho** | Rate sensitivity | Optional (minor impact) |

```python
from strat.greeks import calculate_greeks

# MANDATORY: Calculate Greeks at entry and monitor during hold
entry_greeks = calculate_greeks(
    underlying_price=150.0,
    strike=155.0,
    dte=14,
    risk_free_rate=0.05,
    implied_volatility=0.30,
    option_type='call'
)

# Validate reasonable Greeks
assert 0 < entry_greeks['delta'] < 1, "Invalid delta"
assert entry_greeks['gamma'] > 0, "Invalid gamma"
assert entry_greeks['theta'] < 0, "Calls should have negative theta"
assert entry_greeks['vega'] > 0, "Invalid vega"
```

---

## 9.3 Options-Specific Risk Management

### 9.3.1 Greeks-Based Position Limits

```python
OPTION_POSITION_LIMITS = {
    # Portfolio-level Greeks limits
    'max_portfolio_delta': 0.30,      # Max 30% net delta exposure
    'max_portfolio_gamma': 0.05,      # Max 5% gamma (prevents delta whipsaw)
    'max_portfolio_theta': -0.02,     # Max 2% daily theta bleed
    'max_portfolio_vega': 0.10,       # Max 10% vega exposure
    
    # Per-position limits
    'max_position_delta': 0.10,       # Max 10% delta per position
    'max_contracts_per_symbol': 10,   # Max 10 contracts per underlying
    'max_premium_at_risk': 0.05,      # Max 5% of account in one option
    
    # DTE limits
    'min_dte_entry': 7,               # Don't enter < 7 DTE (gamma risk)
    'max_dte_entry': 45,              # Don't enter > 45 DTE (capital tied up)
    'forced_exit_dte': 3,             # Force exit at 3 DTE
    
    # Spread limits
    'max_spread_pct': 0.10,           # Don't trade if spread > 10% of mid
    'max_spread_absolute': 0.50,      # Don't trade if spread > $0.50
}
```

### 9.3.2 Options Circuit Breakers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Portfolio delta | > 40% | Reduce positions |
| Single position loss | > 50% of premium | Exit position |
| Portfolio theta bleed | > 3% daily | Reduce short-dated positions |
| IV spike (VIX) | > 25% increase | Halt new entries |
| Spread widening | > 15% of mid | Halt trading that option |
| Underlying gap | > 5% | Review all options positions |

```python
class OptionsCircuitBreaker:
    def check_greeks_limits(self, portfolio_greeks: dict) -> tuple[bool, str]:
        """Check if portfolio Greeks exceed limits."""
        
        if abs(portfolio_greeks['delta']) > OPTION_POSITION_LIMITS['max_portfolio_delta']:
            return True, f"Portfolio delta {portfolio_greeks['delta']:.2%} exceeds limit"
        
        if portfolio_greeks['theta'] < -OPTION_POSITION_LIMITS['max_portfolio_theta']:
            return True, f"Portfolio theta bleed {portfolio_greeks['theta']:.2%} exceeds limit"
        
        if portfolio_greeks['vega'] > OPTION_POSITION_LIMITS['max_portfolio_vega']:
            return True, f"Portfolio vega {portfolio_greeks['vega']:.2%} exceeds limit"
        
        return False, "OK"
    
    def check_position_health(self, position: dict) -> tuple[bool, str]:
        """Check individual position for exit signals."""
        
        # DTE-based forced exit
        if position['dte'] <= OPTION_POSITION_LIMITS['forced_exit_dte']:
            return True, f"Forced exit: DTE {position['dte']} below minimum"
        
        # Loss-based exit
        current_value = position['current_price'] * position['contracts'] * 100
        entry_value = position['entry_price'] * position['contracts'] * 100
        loss_pct = (current_value - entry_value) / entry_value
        
        if loss_pct < -0.50:  # 50% loss
            return True, f"Loss limit: {loss_pct:.1%} loss on position"
        
        return False, "OK"
```

### 9.3.3 Assignment and Exercise Risk

**For American-style options (most equity options):**

```python
def check_early_exercise_risk(position: dict, dividend_date: date = None) -> dict:
    """
    Check for early assignment/exercise risk.
    
    Risk factors:
    - Deep ITM options near expiration
    - ITM calls before ex-dividend date
    - ITM puts when rates are high
    """
    risks = []
    
    # Deep ITM risk (delta > 0.90)
    if abs(position['delta']) > 0.90 and position['dte'] < 7:
        risks.append({
            'type': 'deep_itm',
            'severity': 'high',
            'action': 'Consider closing before expiration'
        })
    
    # Dividend risk for calls
    if position['type'] == 'call' and dividend_date:
        days_to_div = (dividend_date - date.today()).days
        if days_to_div <= 2 and position['delta'] > 0.80:
            risks.append({
                'type': 'dividend_exercise',
                'severity': 'high', 
                'action': 'Close before ex-dividend or accept assignment'
            })
    
    # Expiration week pin risk
    if position['dte'] <= 5:
        risks.append({
            'type': 'expiration_week',
            'severity': 'medium',
            'action': 'Monitor for pin risk and gamma exposure'
        })
    
    return risks

# Level 1 Account Constraints (User's Schwab account)
LEVEL_1_CONSTRAINTS = {
    'allowed': [
        'long_stock',
        'long_call',
        'long_put', 
        'cash_secured_put',
        'straddle',
        'strangle'
    ],
    'prohibited': [
        'short_stock',
        'naked_call',
        'naked_put',
        'credit_spread',
        'debit_spread',
        'iron_condor'
    ]
}

def validate_order_for_account_level(order: dict, account_level: int = 1) -> bool:
    """Validate that order is allowed for account level."""
    if account_level == 1:
        if order['strategy'] in LEVEL_1_CONSTRAINTS['prohibited']:
            raise ValueError(f"Strategy {order['strategy']} not allowed for Level 1 account")
    return True
```

---

## 9.4 Options Execution Best Practices

### 9.4.1 Order Type Selection

| Scenario | Recommended Order Type | Rationale |
|----------|----------------------|-----------|
| Liquid options (SPY, QQQ) | Limit at mid | Tight spreads, good fills |
| Less liquid options | Limit at natural | Avoid paying full spread |
| Urgent exit | Limit at market side | Ensure fill |
| NEVER use | Market orders | Uncontrolled slippage |

```python
def determine_option_order_price(
    bid: float, 
    ask: float, 
    urgency: str = 'normal'
) -> float:
    """
    Determine limit price for options order.
    
    NEVER use market orders for options.
    """
    mid = (bid + ask) / 2
    spread = ask - bid
    
    if urgency == 'normal':
        # Try to get filled at mid
        return round(mid, 2)
    
    elif urgency == 'want_fill':
        # Pay 25% of spread for better fill probability
        return round(mid + spread * 0.25, 2)  # For buys
        # return round(mid - spread * 0.25, 2)  # For sells
    
    elif urgency == 'urgent':
        # Pay 50% of spread (near market)
        return round(mid + spread * 0.50, 2)  # For buys
    
    elif urgency == 'immediate':
        # Pay full ask (equivalent to market)
        return round(ask, 2)  # For buys
    
    return round(mid, 2)
```

### 9.4.2 Fill Monitoring and Adjustment

```python
class OptionsOrderManager:
    def __init__(self, broker_client):
        self.client = broker_client
        self.max_wait_seconds = 30
        self.price_improvement_steps = 3
    
    async def execute_with_price_ladder(
        self, 
        symbol: str,
        contracts: int,
        side: str,
        initial_price: float,
        bid: float,
        ask: float
    ) -> dict:
        """
        Execute options order with price ladder.
        Start at mid, walk toward market if not filled.
        """
        spread = ask - bid
        step_size = spread / (self.price_improvement_steps + 1)
        
        for step in range(self.price_improvement_steps + 1):
            if side == 'buy':
                price = initial_price + (step * step_size)
            else:
                price = initial_price - (step * step_size)
            
            order = await self.client.submit_order(
                symbol=symbol,
                qty=contracts,
                side=side,
                type='limit',
                limit_price=round(price, 2),
                time_in_force='ioc'  # Immediate or cancel
            )
            
            if order.filled_qty > 0:
                return {
                    'filled': True,
                    'qty': order.filled_qty,
                    'price': order.filled_avg_price,
                    'steps': step + 1
                }
            
            await asyncio.sleep(1)  # Brief pause between attempts
        
        return {'filled': False, 'qty': 0, 'price': None, 'steps': self.price_improvement_steps}
```

---

## 9.5 Options Backtesting Validation

### 9.5.1 Required Validation Checks

```python
OPTIONS_BACKTEST_VALIDATION = {
    # Trade-level checks
    'min_trades': 50,                    # Minimum for statistical significance
    'max_single_trade_impact': 0.20,     # No single trade > 20% of total P/L
    
    # Cost checks
    'spread_cost_included': True,        # MANDATORY
    'commission_included': True,         # MANDATORY ($0.65/contract)
    'theta_decay_modeled': True,         # MANDATORY for holds > 1 day
    
    # Greeks checks
    'entry_delta_range': (0.30, 0.70),   # Sweet spot for directional trades
    'max_gamma_exposure': 0.10,          # Limit gamma risk
    
    # Realism checks
    'no_fill_at_mid_assumption': True,   # Must assume slippage
    'spread_filter_applied': True,       # Skip illiquid options
    'dte_filter_applied': True,          # Stay in liquid DTE range
}

def validate_options_backtest(backtest_results: dict) -> tuple[bool, list]:
    """Validate options backtest meets requirements."""
    failures = []
    
    if backtest_results['total_trades'] < OPTIONS_BACKTEST_VALIDATION['min_trades']:
        failures.append(f"Insufficient trades: {backtest_results['total_trades']} < 50")
    
    if not backtest_results.get('spread_cost_applied', False):
        failures.append("CRITICAL: Bid-ask spread cost not modeled")
    
    if not backtest_results.get('theta_decay_applied', False):
        failures.append("CRITICAL: Theta decay not modeled for multi-day holds")
    
    # Check for unrealistic results (likely missing costs)
    if backtest_results['win_rate'] > 0.80 and backtest_results['sharpe'] > 3.0:
        failures.append("WARNING: Results too good - verify cost modeling")
    
    return len(failures) == 0, failures
```

### 9.5.2 Options-Specific Monte Carlo

```python
def options_monte_carlo(
    trades: pd.DataFrame,
    n_simulations: int = 10000,
    account_size: float = 10000
) -> dict:
    """
    Monte Carlo simulation for options strategies.
    
    Key differences from equity Monte Carlo:
    - Models theta decay on losing trades
    - Models IV expansion/contraction
    - Models total loss scenarios (options go to $0)
    """
    results = []
    
    for _ in range(n_simulations):
        # Resample trades with replacement
        simulated_trades = trades.sample(n=len(trades), replace=True)
        
        # Add realistic options-specific variance
        for idx, trade in simulated_trades.iterrows():
            # IV uncertainty: +/- 20% on exit IV
            iv_shock = np.random.normal(1.0, 0.20)
            simulated_trades.loc[idx, 'pnl'] *= iv_shock
            
            # Theta uncertainty: could be worse if held longer
            if trade['holding_days'] > 0:
                theta_shock = np.random.uniform(1.0, 1.5)  # Up to 50% worse theta
                if trade['pnl'] < 0:  # Only affects losing trades
                    simulated_trades.loc[idx, 'pnl'] *= theta_shock
        
        # Calculate equity curve
        cumulative_pnl = simulated_trades['pnl'].cumsum()
        final_equity = account_size + cumulative_pnl.iloc[-1]
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        
        results.append({
            'final_equity': final_equity,
            'total_return': (final_equity - account_size) / account_size,
            'max_drawdown': max_drawdown / account_size,
            'max_consecutive_losses': calculate_max_consecutive_losses(simulated_trades)
        })
    
    results_df = pd.DataFrame(results)
    
    return {
        'median_return': results_df['total_return'].median(),
        'p5_return': results_df['total_return'].quantile(0.05),
        'p95_return': results_df['total_return'].quantile(0.95),
        'probability_of_loss': (results_df['final_equity'] < account_size).mean(),
        'probability_of_ruin': (results_df['final_equity'] < account_size * 0.5).mean(),
        'median_max_drawdown': results_df['max_drawdown'].median(),
        'p95_max_drawdown': results_df['max_drawdown'].quantile(0.95)
    }
```

---

## 9.6 Options Pre-Launch Checklist

### 9.6.1 Strategy Validation (Must Pass ALL)

- [ ] Bid-ask spread costs modeled (minimum 3% slippage for options)
- [ ] Per-contract commission included ($0.65/contract)
- [ ] Theta decay modeled for all multi-day holds
- [ ] IV crush modeled if trading around events
- [ ] Greeks limits enforced (delta, gamma, vega)
- [ ] DTE filters applied (7-45 DTE recommended)
- [ ] Spread width filter applied (< 10% of mid)
- [ ] Minimum 50 trades in backtest
- [ ] Profitable at pessimistic spread assumptions (5% slippage)
- [ ] Monte Carlo P(Loss) < 30%, P(Ruin) < 10%

### 9.6.2 Account Requirements

- [ ] Options approval level verified (Level 1, 2, etc.)
- [ ] Strategies match account level permissions
- [ ] Buying power requirements understood
- [ ] Assignment risk procedures documented
- [ ] Exercise procedures documented

### 9.6.3 Data & Execution

- [ ] Options data feed connected (ThetaData)
- [ ] Greeks calculation verified against market data
- [ ] Limit order execution tested (NEVER market orders)
- [ ] Fill monitoring implemented
- [ ] Position management tested (rolls, adjustments)

### 9.6.4 Risk Management

- [ ] Greeks-based position limits configured
- [ ] DTE-based forced exit implemented
- [ ] Loss limit per position configured
- [ ] Portfolio-level Greeks monitoring active
- [ ] Circuit breakers tested

---

## 9.7 Options-Specific Alerts

```python
OPTIONS_ALERTS = {
    'position_alerts': [
        {'condition': 'dte <= 5', 'message': 'Position approaching expiration', 'level': 'warning'},
        {'condition': 'loss_pct > 0.30', 'message': 'Position down 30%', 'level': 'warning'},
        {'condition': 'loss_pct > 0.50', 'message': 'Position down 50% - exit recommended', 'level': 'critical'},
        {'condition': 'delta > 0.90', 'message': 'Deep ITM - assignment risk', 'level': 'warning'},
    ],
    'portfolio_alerts': [
        {'condition': 'portfolio_delta > 0.40', 'message': 'Portfolio delta exceeds 40%', 'level': 'warning'},
        {'condition': 'portfolio_theta < -0.03', 'message': 'Daily theta bleed > 3%', 'level': 'warning'},
        {'condition': 'vix_change > 0.20', 'message': 'VIX up 20%+ - review positions', 'level': 'critical'},
    ],
    'execution_alerts': [
        {'condition': 'spread_pct > 0.10', 'message': 'Wide spread detected', 'level': 'warning'},
        {'condition': 'fill_rate < 0.80', 'message': 'Poor fill rate today', 'level': 'warning'},
    ]
}
```

---

## Appendix: Options Quick Reference

### Minimum Options Backtest Configuration

```python
# Copy-paste this into every options backtest

# 1. Load historical options data (NOT theoretical prices)
options_data = thetadata_fetcher.get_historical_options(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2024-12-31',
    min_dte=7,
    max_dte=45,
    moneyness_range=(0.95, 1.05)  # Near ATM
)

# 2. Filter for liquid options
options_data = options_data[
    (options_data['bid'] > 0) &
    (options_data['ask'] > options_data['bid']) &
    ((options_data['ask'] - options_data['bid']) / options_data['mid'] < 0.10)
]

# 3. Run backtest with realistic costs
pf = vbt.Portfolio.from_signals(
    close=options_data['mid'],
    entries=entries.shift(1),      # No look-ahead
    exits=exits.shift(1),          # No look-ahead
    fees=0.01,                     # 1% for $0.65 commission on ~$6.50 avg premium
    slippage=0.03,                 # 3% minimum for options
    init_cash=10_000,
    size=0.05,                     # 5% max per position (options are leveraged)
)

# 4. Validate results
assert pf.trades.count() >= 50, "Insufficient trades"
print(f"Sharpe: {pf.sharpe_ratio:.2f}")
print(f"Win Rate: {pf.trades.win_rate:.1%}")
print(f"Max DD: {pf.max_drawdown:.1%}")
```

### Options Acceptance Thresholds

```python
OPTIONS_ACCEPTANCE = {
    'min_oos_sharpe': 0.3,              # Lower than equity (higher variance)
    'max_sharpe_degradation': 0.40,      # More degradation expected
    'max_probability_of_loss': 0.30,     # Higher than equity
    'max_probability_of_ruin': 0.10,     # Max 10% chance of 50% drawdown
    'max_expected_drawdown': 0.40,       # Options have higher drawdowns
    'min_trades': 50,                    # Statistical significance
    'max_spread_cost_impact': 0.50,      # Spread costs < 50% of gross profits
}
```

### Level 1 Account Strategy Mapping

| STRAT Pattern | Options Strategy | Level 1 Allowed? |
|---------------|-----------------|------------------|
| 2-1-2 Bullish | Long Call | YES |
| 2-1-2 Bearish | Long Put | YES |
| 3-1-2 Bullish | Long Call | YES |
| 3-1-2 Bearish | Long Put | YES |
| 2-2 Continuation | Long Call/Put | YES |
| Reversal | Straddle/Strangle | YES |
| Any | Vertical Spread | NO (Level 2+) |
| Any | Iron Condor | NO (Level 3+) |

---

**Document Maintenance:**
- Review after any options-related production incident
- Update spread assumptions based on live trading experience
- Add new Greeks-based rules as strategies evolve


---

## Appendix A: Quick Reference

### Minimum Backtest Configuration

```python
# Copy-paste this into every backtest
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries.shift(1),  # No look-ahead
    exits=exits.shift(1),      # No look-ahead
    fees=0.001,                # 0.1% spread
    slippage=0.001,            # 0.1% slippage
    init_cash=10_000,          # Your actual capital
    size=0.10,                 # 10% max per position
    sl_stop=0.02,              # 2% stop loss
)
```

### Acceptance Thresholds

```python
ACCEPTANCE = {
    'min_oos_sharpe': 0.5,
    'max_sharpe_degradation': 0.30,
    'max_probability_of_loss': 0.20,
    'max_probability_of_ruin': 0.05,
    'max_expected_drawdown': 0.25,
    'min_walk_forward_efficiency': 0.50,
    'min_trades': 100,
}
```

### Production Checklist (Abbreviated)

```
□ Strategy passes validation
□ Code handles all errors
□ State persists across restarts
□ Alerts configured and tested
□ Paper traded 4+ weeks
□ Performance matches expectations
□ Risk limits configured
□ Kill switch tested
```

---

**Document Maintenance:**
- Review quarterly or after any production incident
- Update thresholds based on live trading experience
- Add new failure modes as discovered

**Version History:**
- v1.0 (2025-11-26): Initial creation
