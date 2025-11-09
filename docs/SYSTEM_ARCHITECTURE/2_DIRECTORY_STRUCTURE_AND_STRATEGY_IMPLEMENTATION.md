## Directory Structure

**NOTE (Session 20)**: This directory structure describes ATLAS (Layer 1) components only. Directory structure will expand significantly when STRAT (Layer 2) integration begins in Sessions 22-27. Expected additions: `strat/` directory for bar classification and pattern detection components.

```
atlas-trading-system/
|
+-- strategies/                    # ATLAS equity strategy implementations
|   +-- __init__.py
|   +-- base_strategy.py          # Abstract base class
|   +-- high_momentum_52w.py      # NEW: 52-week high momentum
|   +-- quality_momentum.py       # NEW: Quality-Momentum combo
|   +-- semi_vol_momentum.py      # Semi-volatility scaled momentum
|   +-- ibs_mean_reversion.py     # NEW: IBS mean reversion
|   +-- orb.py                    # Opening Range Breakout (EXISTS)
|   +-- bear_protection.py        # NEW: Bear market allocation logic
|
+-- regime/                        # ATLAS regime detection (Layer 1 output)
|   +-- __init__.py
|   +-- academic_jump_model.py    # Academic Jump Model (Phases A-E COMPLETE)
|   +-- academic_features.py      # Feature calculation for Jump Model
|   +-- base_regime_detector.py   # Abstract base class
|   +-- regime_allocator.py       # Regime-based capital allocation (PENDING)
|
+-- core/                          # Core system components
|   +-- __init__.py
|   +-- portfolio_manager.py      # Multi-strategy orchestration
|   +-- risk_manager.py           # Portfolio heat & circuit breakers
|   +-- position_sizer.py         # Position sizing calculations (EXISTS)
|   +-- analyzer.py               # Performance analytics (EXISTS)
|   +-- config.py                 # System configuration
|
+-- data/                          # Data management
|   +-- __init__.py
|   +-- alpaca_client.py          # Alpaca API interface (EXISTS)
|   +-- mtf_manager.py            # Multi-timeframe alignment (EXISTS)
|   +-- indicators.py             # TA-Lib wrapper functions
|   +-- validators.py             # Data quality checks
|
+-- utils/                         # Utility functions
|   +-- __init__.py
|   +-- position_sizing.py        # Position sizing utilities (EXISTS)
|   +-- portfolio_heat.py         # Portfolio heat management (EXISTS)
|   +-- logger.py                 # Logging configuration
|   +-- metrics.py                # Performance metrics
|   +-- validation.py             # Walk-forward validation
|
+-- backtesting/                   # Backtesting infrastructure
|   +-- __init__.py
|   +-- backtest_engine.py        # VectorBT Pro wrapper
|   +-- walk_forward.py           # Walk-forward analysis
|   +-- report_generator.py       # Performance reporting
|
+-- tests/                         # Test suite
|   +-- __init__.py
|   +-- conftest.py               # Pytest configuration
|   +-- test_strategies/          # Strategy tests
|   +-- test_core/                # Core component tests
|   +-- test_integration/         # Integration tests
|   +-- test_validation/          # Walk-forward validation tests
|
+-- notebooks/                     # Jupyter notebooks
|   +-- research/                 # Strategy research
|   +-- analysis/                 # Performance analysis
|   +-- visualization/            # Results visualization
|
+-- docs/                          # Documentation
|   +-- HANDOFF.md               # Session continuity
|   +-- CLAUDE.md                # Development workflows
|   +-- VALIDATION_PROTOCOL.md   # Validation requirements
|   +-- STRATEGY_SPECS/          # Individual strategy specs
|
+-- .venv/                        # Virtual environment (UV-managed)
+-- pyproject.toml               # UV project configuration
+-- .gitignore                   # Git ignore patterns
+-- README.md                    # Project overview
```


**File Naming Conventions**:
- Snake_case for Python files: `quality_momentum.py`
- UPPER_CASE for documentation: `HANDOFF.md`
- Test files prefixed with `test_`: `test_orb.py`

---

## Strategy Implementations

### BaseStrategy Abstract Class

All strategies must inherit from BaseStrategy and implement required methods:

```python
# strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import vectorbtpro as vbt


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Design Philosophy:
    - Enforce consistent interface across strategies
    - Separate signal generation from execution
    - Enable portfolio-level orchestration
    - Facilitate walk-forward validation
    """
    
    def __init__(
        self,
        name: str,
        universe: str,
        rebalance_frequency: str,
        regime_compatibility: Dict[str, bool]
    ):
        """
        Args:
            name: Strategy identifier
            universe: 'sp500', 'russell1000', 'nasdaq100'
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'semi_annual'
            regime_compatibility: {'TREND_BULL': True, 'TREND_NEUTRAL': False, ...}
        """
        self.name = name
        self.universe = universe
        self.rebalance_frequency = rebalance_frequency
        self.regime_compatibility = regime_compatibility
    
    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: str = None
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals.
        
        Args:
            data: OHLCV data with technical indicators
            regime: Current market regime (optional)
            
        Returns:
            DataFrame with 'entry_signal' and 'exit_signal' columns
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate position sizes for each signal.
        
        Args:
            data: OHLCV data
            capital: Available capital
            signals: Entry/exit signals
            
        Returns:
            Series with position sizes in shares
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters are within acceptable ranges.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        pass
    
    def should_trade_in_regime(self, regime: str) -> bool:
        """
        Check if strategy should trade in current regime.
        
        Args:
            regime: 'TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH'
            
        Returns:
            True if strategy should be active
        """
        return self.regime_compatibility.get(regime, False)
    
    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float,
        regime: str = None
    ) -> vbt.Portfolio:
        """
        Run backtest using VectorBT Pro.
        
        This method integrates child strategy's signals and position sizing
        with VectorBT Pro's Portfolio simulator. Uses VERIFIED API patterns
        from existing base_strategy.py implementation.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
                Required columns: Open, High, Low, Close, Volume
            initial_capital: Starting capital in dollars
            regime: Optional market regime filter
            
        Returns:
            vbt.Portfolio object with backtest results
            
        Raises:
            ValueError: If signals or position sizes have mismatched indexes
            
        VBT Integration Notes:
            - Uses from_signals() method (signal-based simulator)
            - size_type='amount': Position sizes are share counts
            - sl_stop: ATR-based stop losses from generate_signals()
            - Fully vectorized (no Python loops)
        """
        # Generate signals from child class
        signals = self.generate_signals(data, regime)
        
        # Validate required signals exist
        required_keys = {'entry_signal', 'exit_signal'}
        if not required_keys.issubset(signals.keys()):
            missing = required_keys - signals.keys()
            raise ValueError(
                f"generate_signals() must return {required_keys}. "
                f"Missing: {missing}"
            )
        
        # Extract stop distance if provided (optional)
        stop_distance = signals.get('stop_distance', None)
        
        # Calculate position sizes from child class
        position_sizes = self.calculate_position_size(
            data,
            initial_capital,
            stop_distance if stop_distance is not None else signals
        )
        
        # Validate position sizes
        if not isinstance(position_sizes, pd.Series):
            raise ValueError(
                f"calculate_position_size() must return pd.Series, "
                f"got {type(position_sizes)}"
            )
        
        if not position_sizes.index.equals(data.index):
            raise ValueError(
                "Position sizes index must match data index. "
                "Use data.index when creating position_sizes Series."
            )
        
        # Run VectorBT Pro backtest with VERIFIED API pattern
        # Pattern source: base_strategy.py (tested and working)
        pf = vbt.Portfolio.from_signals(
            close=data['Close'],                # Note: Capital 'C' for consistency
            entries=signals['entry_signal'],
            exits=signals['exit_signal'],
            size=position_sizes,                # pd.Series of share counts
            size_type='amount',                 # CRITICAL: 'amount' = shares, not dollars
            init_cash=initial_capital,
            fees=0.0015,                        # 15 bps commission
            slippage=0.0015,                    # 15 bps slippage
            sl_stop=stop_distance,              # ATR-based stops (if provided)
            freq='1D'                           # Daily data frequency
        )
        
        return pf
    
    def get_performance_metrics(self, pf: vbt.Portfolio) -> Dict[str, float]:
        """
        Extract standardized performance metrics from VBT Portfolio.
        
        Provides consistent metrics across all strategies for comparison
        and reporting. All strategies use the same metric definitions.
        
        CRITICAL: VectorBT Portfolio properties (total_return, sharpe_ratio, etc.)
        are accessed WITHOUT parentheses. They are properties, not methods.
        
        Args:
            pf: VectorBT Portfolio object from backtest()
            
        Returns:
            Dictionary with standardized performance metrics
            
        Metrics Included:
            - total_return: Overall portfolio return (decimal, e.g., 0.25 = 25%)
            - sharpe_ratio: Risk-adjusted return (annualized)
            - sortino_ratio: Downside risk-adjusted return
            - max_drawdown: Largest peak-to-trough decline (decimal)
            - win_rate: Percentage of winning trades (0-1)
            - profit_factor: Gross profit / gross loss
            - expectancy: Average P&L per trade (in dollars)
            - avg_win: Average winning trade P&L
            - avg_loss: Average losing trade P&L
            - total_trades: Number of closed trades
            - avg_trade_duration: Average holding period (in days)
            
        Source: base_strategy.py + STRATEGY_1_BASELINE_RESULTS.md (verified patterns)
        """
        # Get trade statistics using VERIFIED API patterns
        # CRITICAL: Use .winning and .losing accessors, NOT array indexing
        winning_trades = pf.trades.winning
        losing_trades = pf.trades.losing
        
        # Calculate win rate safely
        total_trades = pf.trades.count()  # -> METHOD (needs parentheses)
        win_rate = winning_trades.count() / total_trades if total_trades > 0 else 0.0
        
        # Calculate average P&L using verified patterns
        avg_win = winning_trades.returns.mean() if winning_trades.count() > 0 else 0.0
        avg_loss = losing_trades.returns.mean() if losing_trades.count() > 0 else 0.0
        
        # Calculate expectancy: (Win% x Avg Win) + (Loss% x Avg Loss)
        expectancy = (
            (win_rate * avg_win) + 
            ((1 - win_rate) * avg_loss)
        ) if total_trades > 0 else 0.0
        
        return {
            # Portfolio-level metrics (PROPERTIES - no parentheses!)
            'total_return': pf.total_return,      # -> PROPERTY
            'sharpe_ratio': pf.sharpe_ratio,      # -> PROPERTY
            'sortino_ratio': pf.sortino_ratio,    # -> PROPERTY
            'max_drawdown': pf.max_drawdown,      # -> PROPERTY
            
            # Trade-level metrics (using VERIFIED patterns)
            'win_rate': win_rate,
            'profit_factor': pf.trades.profit_factor,  # -> PROPERTY
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            # Volume metrics
            'total_trades': total_trades,  # From count() method above
            'avg_trade_duration': pf.trades.duration.mean() if total_trades > 0 else 0.0,
        }
```

---

### Strategy Implementation Template

Template for implementing new strategies:

```python
# strategies/example_strategy.py

from strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
import talib


class ExampleStrategy(BaseStrategy):
    """
    Example strategy implementation template.
    
    This template demonstrates the correct interface for inheriting from
    BaseStrategy. All custom strategies must implement these three methods:
    - validate_parameters()
    - generate_signals()
    - calculate_position_size()
    
    Academic Foundation: [Your citation here]
    Expected Performance: Sharpe X.X, Win Rate X%, Max DD -X%
    
    Implementation Notes:
        - Column names MUST be capital case: Close, High, Low, Open, Volume
        - generate_signals() MUST return dict with 'entry_signal', 'exit_signal', 'stop_distance'
        - calculate_position_size() parameter is stop_distance (pd.Series), NOT signals
        - Position sizes must be share counts (integers), not dollar amounts
        - All Series must have same index as input data
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 0.15,
        atr_multiplier: float = 2.5,
        risk_pct: float = 0.02,
        **kwargs
    ):
        """
        Initialize strategy with parameters.
        
        Args:
            lookback_period: Period for technical indicators (default: 20)
            entry_threshold: RSI threshold for entries (default: 0.15)
            atr_multiplier: ATR multiplier for stop loss (default: 2.5)
            risk_pct: Risk per trade as decimal (default: 0.02 = 2%)
            **kwargs: Additional arguments passed to BaseStrategy
        """
        super().__init__(
            name="Example Strategy",
            universe="sp500",
            rebalance_frequency="daily",
            regime_compatibility={
                'TREND_BULL': True,
                'TREND_NEUTRAL': False,
                'TREND_BEAR': False,
                'CRASH': False
            }
        )
        
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.atr_multiplier = atr_multiplier
        self.risk_pct = risk_pct
        
        self.validate_parameters()
    
    def validate_parameters(self) -> bool:
        """
        Validate parameters are in acceptable ranges.
        
        Raises:
            AssertionError: If any parameter is out of acceptable range
            
        Returns:
            True if all parameters valid
        """
        assert 10 <= self.lookback_period <= 50, \
            f"lookback_period {self.lookback_period} outside range [10, 50]"
        assert 0.0 < self.entry_threshold < 1.0, \
            f"entry_threshold {self.entry_threshold} outside range (0, 1)"
        assert 1.0 <= self.atr_multiplier <= 5.0, \
            f"atr_multiplier {self.atr_multiplier} outside range [1.0, 5.0]"
        assert 0.001 <= self.risk_pct <= 0.05, \
            f"risk_pct {self.risk_pct} outside range [0.001, 0.05]"
        return True
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: str = None
    ) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals and stop distances.
        
        CRITICAL: This method MUST return a dictionary with these keys:
        - 'entry_signal': Boolean Series (True = enter trade)
        - 'exit_signal': Boolean Series (True = exit trade)
        - 'stop_distance': Float Series (stop loss distance in price units)
        
        Args:
            data: OHLCV DataFrame with capital case columns (Close, High, Low, Open, Volume)
            regime: Optional market regime filter (TREND_BULL, TREND_NEUTRAL, etc.)
            
        Returns:
            Dict with 'entry_signal', 'exit_signal', 'stop_distance' keys
            All Series must have same index as input data
            
        Example:
            >>> signals = strategy.generate_signals(data)
            >>> assert 'entry_signal' in signals
            >>> assert 'exit_signal' in signals
            >>> assert 'stop_distance' in signals
            >>> assert signals['entry_signal'].index.equals(data.index)
        """
        # Regime filter (exit if regime incompatible)
        if regime and not self.should_trade_in_regime(regime):
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': pd.Series(0.0, index=data.index)
            }
        
        # Calculate indicators using TA-Lib (NOTE: Capital case column names!)
        rsi = talib.RSI(data['Close'], timeperiod=self.lookback_period)
        
        # Calculate ATR for stop loss distances
        atr = talib.ATR(
            data['High'],
            data['Low'],
            data['Close'],
            timeperiod=14
        )
        
        # Entry signal (RSI oversold)
        entry_signal = rsi < (self.entry_threshold * 100)  # Convert to 0-100 scale
        
        # Exit signal (RSI overbought)
        exit_signal = rsi > ((1.0 - self.entry_threshold) * 100)
        
        # Stop loss distance (ATR-based)
        stop_distance = atr * self.atr_multiplier
        
        # Return required dictionary format
        return {
            'entry_signal': entry_signal.fillna(False),  # No NaN values
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance.fillna(0.0)
        }
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series  # -> CORRECT: stop_distance, not signals!
    ) -> pd.Series:
        """
        Calculate ATR-based position sizes.
        
        CRITICAL: This method signature MUST match what BaseStrategy.backtest() expects:
        - Third parameter is stop_distance (pd.Series), NOT signals (pd.DataFrame)
        - Return value is pd.Series of share counts (integers)
        - Index must match input data.index
        
        Args:
            data: OHLCV DataFrame (same as generate_signals input)
            capital: Current account capital in dollars
            stop_distance: Stop loss distances from generate_signals()
            
        Returns:
            pd.Series of share counts (integers, not dollars)
            Index matches data.index
            
        Example:
            >>> position_sizes = strategy.calculate_position_size(data, 10000, stop_distance)
            >>> assert isinstance(position_sizes, pd.Series)
            >>> assert position_sizes.index.equals(data.index)
            >>> assert all(position_sizes >= 0)  # No negative positions
        """
        from utils.position_sizing import calculate_position_size_capital_constrained
        
        # Calculate position sizes using utility function
        # NOTE: Uses capital case 'Close' column
        position_sizes = calculate_position_size_capital_constrained(
            capital=capital,
            close=data['Close'],  # -> Capital case!
            atr=stop_distance / self.atr_multiplier,  # Convert stop_distance back to ATR
            atr_multiplier=self.atr_multiplier,
            risk_pct=self.risk_pct
        )
        
        # Ensure position sizes are non-negative integers
        position_sizes = position_sizes.fillna(0).clip(lower=0).astype(int)
        
        # Validate output
        assert isinstance(position_sizes, pd.Series), \
            f"Must return pd.Series, got {type(position_sizes)}"
        assert position_sizes.index.equals(data.index), \
            "Position sizes index must match data.index"
        
        return position_sizes
```

---
