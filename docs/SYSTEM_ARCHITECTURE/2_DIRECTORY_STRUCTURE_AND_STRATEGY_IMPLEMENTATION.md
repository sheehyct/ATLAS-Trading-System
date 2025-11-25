## Directory Structure

**NOTE (Session 28)**: This directory structure shows both implemented (Layer 1) and planned (Layers 2-4) components. STRAT Layer 2 implementation begins Session 29.

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
+-- regime/                        # ATLAS regime detection (Layer 1) - VALIDATED
|   +-- __init__.py
|   +-- academic_jump_model.py    # Academic Jump Model (VALIDATED Session 27)
|   +-- academic_features.py      # Feature calculation for Jump Model
|   +-- base_regime_detector.py   # Abstract base class
|   +-- regime_mapper.py          # Four-regime classification
|   +-- regime_allocator.py       # Regime-based capital allocation (PENDING)
|
+-- strat/                         # STRAT pattern recognition (Layer 2) - PENDING
|   +-- __init__.py
|   +-- bar_classifier.py         # Bar classification (1, 2U, 2D, 3)
|   +-- pattern_detector.py       # Pattern detection (3-1-2, 2-1-2, 2-2, Rev Strat)
|   +-- governing_range.py        # Governing range tracking logic
|   +-- timeframe_continuity.py   # Multi-timeframe alignment (4 C's)
|   +-- entry_calculator.py       # Entry/stop/target price calculations
|   +-- position_manager.py       # Scaling in/out, trailing stops
|
+-- core/                          # Core system components
|   +-- __init__.py
|   +-- portfolio_manager.py      # Multi-strategy orchestration
|   +-- risk_manager.py           # Portfolio heat & circuit breakers
|   +-- position_sizer.py         # Position sizing calculations (EXISTS)
|   +-- order_validator.py        # Pre-submission order validation (NEW Session 43)
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
+-- integrations/                  # External service integrations (EXISTS)
|   +-- __init__.py
|   +-- tiingo_data_fetcher.py    # Tiingo market data API (EXISTS)
|   +-- stock_scanner_bridge.py   # Strategy signal scanning (EXISTS)
|   +-- alpaca_trading_client.py  # Alpaca trading API - order execution (NEW Session 42)
|   +-- validate_tiingo_vs_alpaca.py  # Data validation (EXISTS)
|
+-- scripts/                       # Execution and analysis scripts (EXISTS)
|   +-- execute_52w_rebalance.py  # 52W strategy rebalancing (NEW Session 43)
|   +-- schedule_rebalance.sh     # Semi-annual scheduler (NEW Session 44)
|   +-- view_portfolio_status.py  # Portfolio monitoring dashboard (NEW Session 44)
|   +-- backtest_52w_high.py      # 52W backtesting (EXISTS)
|   +-- backtest_atlas_integration.py  # ATLAS+STRAT backtesting (EXISTS)
|
+-- utils/                         # Utility functions
|   +-- __init__.py
|   +-- position_sizing.py        # Position sizing utilities (EXISTS)
|   +-- portfolio_heat.py         # Portfolio heat management (EXISTS)
|   +-- logger.py                 # Logging configuration
|   +-- execution_logger.py       # Execution audit trail (NEW Session 43)
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
|   +-- test_regime/              # Layer 1 tests (48/63 passing Session 27)
|   +-- test_strat/               # Layer 2 tests (PENDING)
|   +-- test_strategies/          # Strategy tests
|   +-- test_core/                # Core component tests
|   +-- test_integrations/        # External service integration tests (NEW Session 42)
|   +-- test_execution/           # Execution pipeline tests (NEW Session 43-44)
|   +-- test_integration/         # Integration tests
|   +-- test_validation/          # Walk-forward validation tests
|
+-- logs/                          # Execution logs (gitignored, NEW Session 42)
|   +-- execution_{date}.log      # Daily execution logs
|   +-- trades_{date}.csv         # Trade audit trail
|   +-- errors_{date}.log         # Error logs
|   +-- archive/                  # Archived logs (90+ days old)
|
+-- notebooks/                     # Jupyter notebooks
|   +-- research/                 # Strategy research
|   +-- analysis/                 # Performance analysis
|   +-- visualization/            # Results visualization
|
+-- docs/                          # Documentation
|   +-- HANDOFF.md               # Session continuity
|   +-- CLAUDE.md                # Development workflows
|   +-- SESSION_26_27_LAMBDA_RECALIBRATION.md  # Technical reports
|   +-- SYSTEM_ARCHITECTURE/     # Architecture documentation
|       +-- 1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md
|       +-- 2_DIRECTORY_STRUCTURE_AND_STRATEGY_IMPLEMENTATION.md
|       +-- 3_CORE_COMPONENTS_RISK_MANAGEMENT_AND_BACKTESTING_REQUIREMENTS.md
|       +-- 4_WALK_FORWARD_VALIDATION_PERFORMANCE_TARGETS_AND_DEPLOYMENT.md
|       +-- 5_EXECUTION_ARCHITECTURE.md  # NEW Session 42
|       +-- STRAT_LAYER_SPECIFICATION.md
|       +-- INTEGRATION_ARCHITECTURE.md
|       +-- CAPITAL_DEPLOYMENT_GUIDE.md
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

## Execution Layer Components (Post-Validation)

**NOTE:** These components are used ONLY after strategies pass paper trading validation (6+ months, 100+ trades). They are NOT part of the backtesting infrastructure.

### Directory Distinctions

**data/ vs integrations/**

**data/**: Internal data pipeline components
- Data loading and transformation
- Data quality validation
- Cache management
- Focus: Getting data INTO the system
- Example: `data/alpaca_client.py` (historical OHLCV data fetching)

**integrations/**: External service integrations
- API wrappers for external services
- Data fetching from external sources
- Order execution to external brokers
- Validation across external data sources
- Focus: Interfacing WITH external systems
- Examples:
  - `integrations/tiingo_data_fetcher.py` (data integration)
  - `integrations/stock_scanner_bridge.py` (strategy integration)
  - `integrations/alpaca_trading_client.py` (trading execution integration)

### AlpacaTradingClient Interface (integrations/)

Wrapper for Alpaca Trading API focused on order execution (distinct from data fetching):

```python
# integrations/alpaca_trading_client.py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging
import time

class AlpacaTradingClient:
    """
    Alpaca Trading API wrapper for order execution.

    Responsibilities:
    - Order submission (market, limit orders)
    - Position monitoring
    - Account status queries
    - Order status tracking
    - Retry logic with exponential backoff
    - Error handling for network/API failures

    NOT responsible for:
    - Historical data fetching (use data/alpaca_client.py)
    - Strategy signal generation (use strategies/)
    - Position sizing calculations (use utils/position_sizing.py)
    - Risk validation (use core/order_validator.py)

    Design Philosophy:
    - Separation of concerns: Trading vs data are distinct
    - Audit trail: Log all API calls via ExecutionLogger
    - Resilience: Retry transient failures automatically
    - Testability: Mock TradingClient in tests
    """

    def __init__(self, account: str = 'LARGE', logger: Optional[logging.Logger] = None):
        """
        Initialize trading client.

        Args:
            account: Account name ('SMALL', 'MID', 'LARGE')
            logger: ExecutionLogger instance for audit trail
        """

    def connect() -> bool:
        """Connect to Alpaca Trading API."""

    def get_account() -> dict:
        """Get account info (equity, buying power, positions)."""

    def get_positions() -> List[dict]:
        """Get all current positions."""

    def submit_market_order(symbol: str, qty: int, side: str) -> dict:
        """
        Submit market order.

        Args:
            symbol: Ticker symbol
            qty: Share quantity
            side: 'buy' or 'sell'

        Returns:
            Order dict with order_id, status, submitted_at
        """

    def _retry_api_call(func, *args, max_retries: int = 3) -> Any:
        """
        Retry API calls with exponential backoff.

        Retry schedule: 1s, 2s, 4s (max 3 attempts)
        Logs each retry attempt at WARNING level
        Raises exception after max retries exhausted
        """
```

### OrderValidator Interface (core/)

Pre-submission validation to prevent invalid orders:

```python
# core/order_validator.py

class OrderValidator:
    """
    Validates orders before broker submission.

    Checks:
    - Sufficient buying power
    - Position size within limits (max 15% per position)
    - Portfolio heat constraints (max 8% total risk)
    - Symbol validity (proper format, tradable)
    - Market hours (9:30-16:00 ET, NYSE calendar)
    - Regime compliance (CRASH regime = 0% allocation)
    - No duplicate orders (prevent double submission)

    Design Philosophy:
    - Fail fast: Validate before submitting, not after
    - Clear errors: Return specific validation failure reasons
    - Configurable limits: Position size and heat limits are tunable
    - Testable: All checks are pure functions
    """

    def __init__(self, max_position_pct: float = 0.15, max_portfolio_heat: float = 0.08):
        """
        Initialize validator with risk limits.

        Args:
            max_position_pct: Maximum position size as % of portfolio (default 15%)
            max_portfolio_heat: Maximum total risk as % of portfolio (default 8%)
        """

    def validate_buying_power(account_info: dict, order_value: float) -> Tuple[bool, str]:
        """Check sufficient buying power for order."""

    def validate_position_size(order_value: float, portfolio_value: float) -> Tuple[bool, str]:
        """Check position size within 15% limit."""

    def validate_regime_compliance(regime: str, allocation_pct: float) -> Tuple[bool, str]:
        """
        Enforce regime allocation rules.

        Rules:
        - TREND_BULL: 100% max allocation
        - TREND_NEUTRAL: 70% max allocation
        - TREND_BEAR: 30% max allocation
        - CRASH: 0% allocation (NO new orders)
        """

    def validate_order_batch(orders: List[dict], account_info: dict, regime: str) -> dict:
        """
        Validate batch of orders before submission.

        Returns:
            {
                'valid': bool,
                'errors': List[str],     # Critical errors (prevent submission)
                'warnings': List[str]    # Non-critical warnings (log but allow)
            }
        """
```

### ExecutionLogger Interface (utils/)

Centralized logging for all execution events (audit trail):

```python
# utils/execution_logger.py

import logging
import pandas as pd
from datetime import datetime

class ExecutionLogger:
    """
    Audit trail for all trading operations.

    Logs to:
    - Console: INFO and above (human-readable monitoring)
    - File: logs/execution_{date}.log (all levels)
    - CSV: logs/trades_{date}.csv (trade events only)
    - File: logs/errors_{date}.log (errors only)

    Design Philosophy:
    - Compliance: Complete audit trail for regulatory review
    - Debugging: Detailed logs for troubleshooting
    - Analysis: CSV format for trade performance analysis
    - Separation: Errors isolated for quick review
    """

    def __init__(self, log_dir: str = 'logs/'):
        """Initialize logger with log directory."""

    def log_order_submission(symbol: str, qty: int, side: str, order_type: str, order_id: str):
        """Log order submission event."""

    def log_order_fill(order_id: str, fill_price: float, fill_qty: int, commission: float):
        """Log order fill event."""

    def log_order_rejection(order_id: str, reason: str):
        """Log order rejection event."""

    def log_error(component: str, error_msg: str, exc_info: Optional[Exception] = None):
        """Log error event with stack trace."""

    def log_reconciliation(target_positions: dict, actual_positions: dict, discrepancies: List[str]):
        """Log position reconciliation results."""
```

### Execution Script Pattern (scripts/)

Main orchestration script for strategy rebalancing:

```python
# scripts/execute_52w_rebalance.py

"""
52-Week High Momentum Strategy Rebalancing Script.

Workflow:
1. Check if rebalance date (Feb 1, Aug 1)
2. Initialize components (logger, trading client, validator)
3. Fetch current state (account, positions, regime)
4. Generate signals (stock scanner, top 10 momentum)
5. Apply regime allocation (BULL=100%, NEUTRAL=70%, BEAR=30%, CRASH=0%)
6. Calculate target positions (equal weight)
7. Generate rebalancing orders (close, adjust, open)
8. Validate orders (OrderValidator checks)
9. Submit orders (if not dry-run mode)
10. Monitor fills (5 min timeout)
11. Reconcile (compare target vs actual)
12. Summary report (log results)

Usage:
    # Dry-run (calculates orders but doesn't submit)
    python scripts/execute_52w_rebalance.py --dry-run

    # Live execution on rebalance date
    python scripts/execute_52w_rebalance.py

    # Force execution on non-rebalance date
    python scripts/execute_52w_rebalance.py --force --date 2025-11-18

    # Custom universe and top N
    python scripts/execute_52w_rebalance.py --universe sp500 --top-n 20
"""

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--date', type=str)
    parser.add_argument('--universe', default='tech', choices=['tech', 'sp500'])
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()

    # Initialize components
    logger = ExecutionLogger()
    trading_client = AlpacaTradingClient(account='LARGE', logger=logger)
    validator = OrderValidator()

    # ... (see 5_EXECUTION_ARCHITECTURE.md for complete flow)
```

For complete execution pipeline specifications, error handling, and monitoring standards, see `docs/SYSTEM_ARCHITECTURE/5_EXECUTION_ARCHITECTURE.md`.

---
