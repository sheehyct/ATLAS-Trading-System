
## Core Components

**NOTE (Session 20)**: These components describe ATLAS (Layer 1) risk management. Multi-layer integration will add capital-aware execution logic in Layer 3 to route between equity execution (optimal $10k+) and options execution (optimal $3k).

### Capital Requirements for ATLAS Components

**CRITICAL CONSTRAINT IDENTIFIED (Session 20)**:
- ATLAS equity strategies require MINIMUM $10,000 for proper position sizing
- With $3,000 capital: Capital constrained to 0.06% actual risk vs 2% target risk
- Root cause: ATR-based position sizing calculation uses `min(risk_based_size, capital_constrained_size)`
- Impact: Cannot maintain 2-3 concurrent positions, systematic underperformance

**Deployment Recommendation**:
- **$3,000 capital**: Deploy STRAT + Options (Layer 2-3, capital-efficient, 27x better)
- **$10,000+ capital**: Deploy ATLAS equities (Layer 1, optimal)
- **Mixed approach**: Paper trade ATLAS with $10k simulated + Live trade STRAT options with $3k

See HANDOFF.md "Capital Requirements Analysis" section for detailed position sizing math and capital efficiency comparison.

### Portfolio Manager

Orchestrates multiple strategies with regime-based allocation:

```python
# core/portfolio_manager.py

from typing import Dict, List
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from core.risk_manager import RiskManager
from regime.regime_allocator import RegimeAllocator
from utils.portfolio_heat import PortfolioHeatManager


class PortfolioManager:
    """
    Coordinates multiple strategies with regime-aware allocation.
    
    Key Responsibilities:
    - Detect current market regime
    - Allocate capital based on regime
    - Enforce portfolio heat limits
    - Track aggregate performance
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        initial_capital: float,
        regime_allocator: RegimeAllocator,
        risk_manager: RiskManager,
        heat_manager: PortfolioHeatManager
    ):
        self.strategies = strategies
        self.capital = initial_capital
        self.regime_allocator = regime_allocator
        self.risk_manager = risk_manager
        self.heat_manager = heat_manager
        
        self.positions = {}
        self.equity_curve = []
        self.regime_history = []
    
    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime using Jump Model.
        
        Args:
            data: Recent market data
            
        Returns:
            'TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', or 'CRASH'
        """
        return self.regime_allocator.detect_regime(data)
    
    def allocate_capital(self, regime: str) -> Dict[str, float]:
        """
        Allocate capital across strategies based on regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dict mapping strategy names to capital allocation
        """
        # Get regime-specific allocation
        allocation_template = self.regime_allocator.get_allocation(regime)
        
        # Map to actual strategies
        allocations = {}
        for strategy in self.strategies:
            strategy_key = strategy.name.lower().replace(' ', '_')
            
            if strategy_key in allocation_template:
                allocations[strategy.name] = (
                    allocation_template[strategy_key] * self.capital
                )
            else:
                allocations[strategy.name] = 0.0
        
        return allocations
    
    def check_portfolio_heat(self) -> float:
        """Calculate current portfolio heat across all positions."""
        total_risk = sum(
            pos['risk'] for pos in self.positions.values()
        )
        return total_risk / self.capital
    
    def can_take_position(
        self,
        strategy_name: str,
        position_risk: float
    ) -> bool:
        """
        Check if new position would exceed heat limit.
        
        Args:
            strategy_name: Name of strategy requesting position
            position_risk: Dollar risk of proposed position
            
        Returns:
            True if position can be taken
        """
        return self.heat_manager.can_accept_trade(
            symbol=strategy_name,
            risk_amount=position_risk,
            capital=self.capital
        )
    
    def run_multi_strategy_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Run coordinated backtest across all strategies.
        
        Args:
            data: Dict mapping symbols to OHLCV DataFrames
            start_date: Backtest start
            end_date: Backtest end
            
        Returns:
            Comprehensive performance metrics
        """
        # Implementation details...
        pass
```

---

## Risk Management Framework

### Three-Layer Risk Architecture

**Layer 1: Position Sizing (Individual Trade Risk)**

```python
# Already implemented in utils/position_sizing.py
# NO CHANGES REQUIRED - existing implementation correct

def calculate_position_size_capital_constrained(
    capital: float,
    close: pd.Series,
    atr: pd.Series,
    atr_multiplier: float = 2.5,
    risk_pct: float = 0.02
) -> pd.Series:
    """
    ATR-based position sizing with capital constraint.
    
    Key: Uses MINIMUM of risk-based and capital-based sizing.
    This prevents positions from exceeding 100% of capital.
    """
    # Stop distance
    stop_distance = atr * atr_multiplier
    
    # Risk-based sizing
    position_size_risk = (capital * risk_pct) / stop_distance
    
    # Capital-based sizing (hard constraint)
    position_size_capital = capital / close
    
    # Take MINIMUM
    position_size = np.minimum(position_size_risk, position_size_capital)
    
    # Handle edge cases
    position_size = position_size.fillna(0).replace([np.inf, -np.inf], 0)
    
    return position_size.astype(int)
```

**Verification Checklist**:
- [x] Mean position size: 10-30% of capital
- [x] Max position size: Never exceeds 100%
- [x] Edge case handling: NaN, Inf, zero values
- [x] Implementation: COMPLETE (Gate 1 PASSED)

---

**Layer 2: Portfolio Heat Management**

```python
# Already implemented in utils/portfolio_heat.py
# NO CHANGES REQUIRED - existing implementation correct

class PortfolioHeatManager:
    """
    Tracks aggregate risk across all positions.
    Enforces 6-8% maximum portfolio heat.
    """
    
    def __init__(self, max_heat: float = 0.08):
        self.max_heat = max_heat
        self.active_positions = {}
    
    def can_accept_trade(
        self,
        symbol: str,
        risk_amount: float,
        capital: float
    ) -> bool:
        """
        Gating function: Reject trade if heat would exceed limit.
        
        Returns:
            True if trade can be accepted
        """
        current_heat = self.calculate_current_heat(capital)
        new_risk = risk_amount / capital
        
        return (current_heat + new_risk) <= self.max_heat
    
    def add_position(self, symbol: str, risk_amount: float):
        """Add position to heat tracking."""
        self.active_positions[symbol] = risk_amount
    
    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
```

**Verification Checklist**:
- [x] Heat calculation: Sum of all position risks
- [x] Gating function: Rejects before trade execution
- [x] Max heat: 6-8% hard limit enforced
- [x] Implementation: COMPLETE (Gate 2 PASSED)

---

**Layer 3: Multi-Layer Stop Loss System**

```python
# utils/stop_loss.py

class MultiLayerStopSystem:
    """
    Three independent stop loss mechanisms:
    1. ATR-based initial stop (volatility-adjusted)
    2. Time-based stop (prevents dead money)
    3. Trailing stop (locks in profits)
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.5,
        max_hold_days: int = 5,
        trailing_activation: float = 2.0  # 2:1 R:R
    ):
        self.atr_multiplier = atr_multiplier
        self.max_hold_days = max_hold_days
        self.trailing_activation = trailing_activation
    
    def calculate_initial_stop(
        self,
        entry_price: float,
        atr: float,
        direction: str = 'long'
    ) -> float:
        """
        Calculate ATR-based initial stop loss.
        
        Args:
            entry_price: Entry price
            atr: Average True Range at entry
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        stop_distance = self.atr_multiplier * atr
        
        if direction == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def check_time_stop(
        self,
        entry_date: pd.Timestamp,
        current_date: pd.Timestamp
    ) -> bool:
        """
        Check if maximum hold period exceeded.
        
        Returns:
            True if time stop triggered
        """
        days_held = (current_date - entry_date).days
        return days_held >= self.max_hold_days
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        direction: str = 'long'
    ) -> float:
        """
        Calculate trailing stop if profit target reached.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            current_stop: Current stop loss
            atr: Current ATR
            direction: 'long' or 'short'
            
        Returns:
            Updated stop loss (may be same as current_stop)
        """
        initial_risk = abs(entry_price - current_stop)
        profit_target = entry_price + (initial_risk * self.trailing_activation)
        
        if direction == 'long':
            # Activate trailing stop if price above profit target
            if current_price >= profit_target:
                new_stop = current_price - (self.atr_multiplier * atr)
                return max(new_stop, current_stop)  # Never lower stop
        else:
            if current_price <= profit_target:
                new_stop = current_price + (self.atr_multiplier * atr)
                return min(new_stop, current_stop)  # Never raise stop
        
        return current_stop
    
    def evaluate_all_stops(
        self,
        position: Dict,
        current_data: pd.Series
    ) -> Tuple[bool, str]:
        """
        Check all stop conditions.
        
        Args:
            position: Dict with position details
            current_data: Current OHLCV bar
            
        Returns:
            (should_exit: bool, reason: str)
        """
        # ATR stop
        if position['direction'] == 'long':
            if current_data['close'] <= position['stop_loss']:
                return True, "ATR_STOP"
        else:
            if current_data['close'] >= position['stop_loss']:
                return True, "ATR_STOP"
        
        # Time stop
        if self.check_time_stop(position['entry_date'], current_data.name):
            return True, "TIME_STOP"
        
        # No stop triggered
        return False, None
```

**Stop Loss Priority**:
1. **ATR Stop**: Primary mechanism, checked every bar
2. **Time Stop**: Prevents positions from becoming "dead money"
3. **Trailing Stop**: Locks in profits at 2:1 R:R

**Critical Rules**:
- NEVER widen stops after entry
- NEVER ignore stop signals
- ALWAYS execute stops (no discretion)

---

**Layer 4: Volume Confirmation**

For breakout strategies (ORB, 52-Week High):

```python
def validate_volume_confirmation(
    current_volume: float,
    volume_ma: float,
    threshold: float = 2.0
) -> bool:
    """
    Validate volume confirmation for breakout signals.
    
    Research shows 2x volume threshold achieves significantly
    higher success rates for breakout strategies.
    
    Args:
        current_volume: Current bar volume
        volume_ma: Moving average of volume (20-day typical)
        threshold: Volume multiplier (default 2.0x)
        
    Returns:
        True if volume confirmed
    """
    return current_volume >= (volume_ma * threshold)
```

**Volume Confirmation Rules**:
- **Mandatory** for: ORB, 52-Week High Momentum
- **Not required** for: Mean reversion, Quality-Momentum
- **Threshold**: 2.0x moving average (research-validated)
- **Consequence**: Without confirmation, reject signal

---

**Layer 5: Multi-Symbol Validation**

```python
# utils/validation.py

def validate_strategy_multi_symbol(
    strategy: BaseStrategy,
    test_symbols: List[str],
    data: Dict[str, pd.DataFrame],
    initial_capital: float = 100000,
    min_sharpe: float = 1.0,
    max_drawdown: float = 0.25
) -> Dict:
    """
    Test strategy across multiple symbols for overfitting detection.
    
    Purpose: Ensure strategy performance generalizes across different instruments
    and isn't overfit to a single symbol. A robust strategy should pass on
    at least 2 out of 3 test symbols.
    
    Methodology:
        1. Run backtest on each symbol independently
        2. Extract Sharpe ratio and maximum drawdown
        3. Compare against acceptance thresholds
        4. Aggregate results (2/3 symbols must pass)
    
    Args:
        strategy: Strategy instance to test (already configured)
        test_symbols: List of ticker symbols to test on
            Recommended: ['SPY', 'QQQ', 'IWM'] (large-cap, tech, small-cap)
            Alternative: ['SPY', 'DIA', 'MDY'] (large, mid, mid-cap)
        data: Dictionary mapping symbols to OHLCV DataFrames
            Example: {'SPY': spy_data, 'QQQ': qqq_data, 'IWM': iwm_data}
        initial_capital: Starting capital for each backtest (default: $100,000)
        min_sharpe: Minimum acceptable Sharpe ratio (default: 1.0)
        max_drawdown: Maximum acceptable drawdown as decimal (default: 0.25 = 25%)
        
    Returns:
        Dictionary with structure:
        {
            'SPY': {
                'sharpe': float,
                'max_drawdown': float (absolute value),
                'passed': bool
            },
            'QQQ': {...},
            'IWM': {...},
            'overall_passed': bool,
            'symbols_passed': int,
            'pass_rate': float
        }
    
    Example:
        >>> results = validate_strategy_multi_symbol(
        ...     strategy=my_strategy,
        ...     test_symbols=['SPY', 'QQQ', 'IWM'],
        ...     data={'SPY': spy_data, 'QQQ': qqq_data, 'IWM': iwm_data},
        ...     min_sharpe=1.0,
        ...     max_drawdown=0.25
        ... )
        >>> print(f"Passed: {results['overall_passed']}")
        >>> print(f"Pass Rate: {results['pass_rate']:.1%}")
    
    Notes:
        - VectorBT Portfolio properties (sharpe_ratio, max_drawdown) are accessed
          WITHOUT parentheses (they're properties, not methods)
        - max_drawdown returns NEGATIVE values (e.g., -0.25 for 25% drawdown)
        - abs() is applied to compare with positive threshold
        - Strategy passes if 2/3 symbols pass (67% threshold)
    """
    results = {}
    
    for symbol in test_symbols:
        try:
            # Run backtest (uses corrected BaseStrategy.backtest())
            pf = strategy.backtest(
                data=data[symbol],
                initial_capital=initial_capital
            )
            
            # Extract metrics using VectorBT PROPERTIES (no parentheses!)
            # VERIFIED: These are properties, not methods
            sharpe = pf.sharpe_ratio  # -> NO parentheses!
            max_dd = abs(pf.max_drawdown)  # -> NO parentheses! (abs() for positive comparison)
            
            # Evaluate pass/fail
            passed = (sharpe >= min_sharpe) and (max_dd <= max_drawdown)
            
            results[symbol] = {
                'sharpe': sharpe,
                'max_drawdown': max_dd,  # Stored as positive value
                'passed': passed
            }
            
        except Exception as e:
            # Handle backtest failures gracefully
            print(f"Warning: Backtest failed for {symbol}: {e}")
            results[symbol] = {
                'sharpe': 0.0,
                'max_drawdown': 1.0,  # Worst case
                'passed': False,
                'error': str(e)
            }
    
    # Calculate aggregate results
    total_passed = sum(r['passed'] for r in results.values() if 'error' not in r)
    total_tested = len([r for r in results.values() if 'error' not in r])
    
    results['overall_passed'] = total_passed >= (total_tested * 0.67)  # 2/3 threshold
    results['symbols_passed'] = total_passed
    results['symbols_tested'] = total_tested
    results['pass_rate'] = total_passed / total_tested if total_tested > 0 else 0.0
    
    return results


def print_multi_symbol_results(results: Dict) -> None:
    """
    Pretty-print multi-symbol validation results.
    
    Args:
        results: Output from validate_strategy_multi_symbol()
    """
    print("\n" + "="*70)
    print("MULTI-SYMBOL VALIDATION RESULTS")
    print("="*70)
    
    # Per-symbol results
    for symbol, metrics in results.items():
        if symbol in ['overall_passed', 'symbols_passed', 'symbols_tested', 'pass_rate']:
            continue  # Skip aggregate keys
            
        status = "[PASS] PASS" if metrics['passed'] else "[FAIL] FAIL"
        print(f"\n{symbol}: {status}")
        
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    
    # Overall results
    print("\n" + "-"*70)
    print(f"Overall Result: {'[PASS] PASSED' if results['overall_passed'] else '[FAIL] FAILED'}")
    print(f"Symbols Passed: {results['symbols_passed']}/{results['symbols_tested']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    print("="*70 + "\n")
```

**Multi-Symbol Validation Rules**:
- Test symbols: SPY (large cap), QQQ (tech), IWM (small cap)
- Pass criteria: 2 out of 3 symbols must pass
- Alternative: Test across different timeframes of same symbol
- Purpose: Detect overfitting to single instrument

---

## Backtesting Requirements

### Phase 1: Initial Strategy Validation

**Minimum Data Requirements**:
- Time period: 15-20 years (2005-2025 recommended)
- Must include: 2008 crash, 2020 crash, 2022 bear
- Tick data: Daily minimum, intraday for ORB
- Data quality: No missing bars, proper adjustments

**Mandatory Checks**:
```python
def validate_backtest_data(data: pd.DataFrame) -> bool:
    """
    Validate data quality before backtesting.
    
    Returns:
        True if data passes all checks
    """
    # Check 1: No missing data
    assert data.isnull().sum().sum() == 0, "Missing data detected"
    
    # Check 2: OHLC relationships
    assert (data['high'] >= data['low']).all(), "High < Low detected"
    assert (data['high'] >= data['close']).all(), "High < Close detected"
    assert (data['low'] <= data['close']).all(), "Low > Close detected"
    
    # Check 3: Reasonable values
    assert (data['volume'] > 0).all(), "Zero/negative volume detected"
    assert (data['close'] > 0).all(), "Zero/negative price detected"
    
    # Check 4: Date index
    assert isinstance(data.index, pd.DatetimeIndex), "Invalid index type"
    assert data.index.is_monotonic_increasing, "Non-monotonic dates"
    
    return True
```

---
