# ATLAS System Architecture Reference (Layer 1)
## Development Guide for Implementation

**Document Purpose**: This is the definitive reference for ATLAS as Layer 1 in a multi-layer trading architecture. ATLAS provides regime detection and equity strategy execution. This is ONE component in a larger unified system (ATLAS + STRAT + Options).

**CRITICAL CONTEXT - Multi-Layer Architecture (Session 20)**:
- **Layer 1 (ATLAS)**: Regime detection + equity strategies (THIS DOCUMENT)
- **Layer 2 (STRAT)**: Pattern recognition for precise entry/exit levels (PENDING - Sessions 22-27)
- **Layer 3 (Execution)**: Capital-aware deployment - options ($3k optimal) OR equities ($10k+ optimal)

**Integration Status**: Layer 1 (ATLAS) nearing completion (Phase F validation next). Layers 2-3 implementation begins after Phase F completes.

**Capital Requirements for ATLAS (Layer 1)**:
- Minimum Viable Capital: $10,000 (full position sizing capability)
- With $3,000: CAPITAL CONSTRAINED, sub-optimal performance
- Recommendation: Paper trade ATLAS with $10k simulated, deploy STRAT+Options live with $3k

**Target Audience**: Development Team (Quantitative Developers)
**Version**: 2.0 (Layer 1 Implementation)
**Last Updated**: November 2025 (Updated Session 20)
**System**: ATLAS Algorithmic Trading System

**Related Documentation**:
- See HANDOFF.md "Multi-Layer Integration Architecture" for integration approach
- See CLAUDE.md "STRAT Integration Development Rules" for Layer 2 development guidelines
- See Claude Desktop Layer Proposals for STRAT implementation details

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Directory Structure](#directory-structure)
4. [Strategy Implementations](#strategy-implementations)
5. [Core Components](#core-components)
6. [Data Pipeline](#data-pipeline)
7. [Risk Management Framework](#risk-management-framework)
8. [Testing Requirements](#testing-requirements)
9. [Integration Patterns](#integration-patterns)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Architecture Philosophy

**Modular Strategy Design**: Each strategy is self-contained with its own signal generation, position sizing, and exit logic. Strategies share common infrastructure but remain independent.

**Portfolio-Level Orchestration**: A central portfolio manager coordinates multiple strategies, enforces risk limits, and manages capital allocation.

**Vectorized Operations**: All calculations use pandas/numpy vectorization. No Python loops for data processing. VectorBT Pro compatibility is mandatory.

**Walk-Forward Validation**: All strategies must pass out-of-sample testing with <30% performance degradation from in-sample results.

### System Layers

```
┌─────────────────────────────────────────┐
│   Portfolio Manager (Orchestration)     │
│   - Capital allocation                  │
│   - Portfolio heat management           │
│   - Multi-strategy coordination         │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────┐       ┌───────▼──────┐
│  Strategy 1  │  ...  │  Strategy N  │
│  - Signals   │       │  - Signals   │
│  - Sizing    │       │  - Sizing    │
│  - Exits     │       │  - Exits     │
└───────┬──────┘       └───────┬──────┘
        │                       │
        └───────────┬───────────┘
                    │
┌─────────────────────────────────────────┐
│        Risk Management Layer            │
│   - Position sizing                     │
│   - Stop loss enforcement               │
│   - Portfolio heat limits               │
│   - Drawdown circuit breakers           │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│          Data Management                │
│   - Multi-timeframe alignment           │
│   - Technical indicators                │
│   - Market hours filtering              │
│   - Data quality validation             │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Data Source (Alpaca)            │
└─────────────────────────────────────────┘
```

---

## Technology Stack

### Core Framework

```python
# pyproject.toml or requirements.txt
[project.dependencies]
python = ">=3.12"
vectorbtpro = ">=0.28.0"     # Backtesting engine
alpaca-py = ">=0.26.0"        # Market data & trading
pandas = ">=2.0.0"            # Data manipulation
numpy = ">=1.24.0"            # Numerical computing
ta-lib = ">=0.4.28"           # Technical indicators
scikit-learn = ">=1.3.0"      # ML (GMM, StandardScaler)
statsmodels = ">=0.14.0"      # Statistics (cointegration)
pandas-market-calendars = ">=4.3"  # Market hours
python-dotenv = ">=1.0.0"     # Environment variables
loguru = ">=0.7.0"            # Logging
pytest = ">=7.4.0"            # Testing
pydantic = ">=2.0.0"          # Configuration validation

[project.dev-dependencies]
ruff = ">=0.1.0"              # Linting/formatting
mypy = ">=1.7.0"              # Type checking
pytest-cov = ">=4.1.0"        # Coverage reporting
```

### Why These Tools

**VectorBT Pro**: 
- Native TA-Lib integration
- Vectorized operations (10-100x faster than loops)
- Multi-timeframe support out-of-the-box
- Portfolio-level analytics
- Custom indicator support

**Alpaca**:
- Clean, adjusted OHLCV data
- 4+ years of 5-minute bars available
- Paper trading API (matches live execution)
- WebSocket streaming for live data
- Commission-free trading

**TA-Lib**:
- Industry standard (used by professionals)
- 150+ technical indicators
- Properly handles edge cases (NaN, lookback periods)
- C-based implementation (fast)

**UV Package Manager**:
- 10-100x faster than pip
- Reproducible environments
- Better dependency resolution
- Compatible with existing Python ecosystem

---

## Directory Structure

```
vectorbt-workspace/
│
├── strategies/                    # Strategy implementations
│   ├── __init__.py
│   ├── base_strategy.py          # Abstract base class
│   ├── gmm_regime.py             # Gaussian Mixture Model regime detection
│   ├── mean_reversion.py         # Five-day washout mean reversion
│   ├── orb.py                    # Opening Range Breakout
│   ├── pairs_trading.py          # Statistical arbitrage
│   └── momentum_portfolio.py     # Semi-volatility scaled momentum
│
├── core/                          # Core system components
│   ├── __init__.py
│   ├── portfolio_manager.py      # Multi-strategy orchestration
│   ├── risk_manager.py           # Portfolio heat & circuit breakers
│   ├── position_sizer.py         # Position sizing calculations
│   ├── analyzer.py               # Performance analytics
│   └── config.py                 # System configuration
│
├── data/                          # Data management
│   ├── __init__.py
│   ├── alpaca_client.py          # Alpaca API interface
│   ├── mtf_manager.py            # Multi-timeframe alignment
│   ├── indicators.py             # Custom indicator library
│   └── validators.py             # Data quality checks
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── position_sizing.py        # Position sizing utilities
│   ├── logger.py                 # Logging configuration
│   ├── metrics.py                # Performance metrics
│   └── validation.py             # Walk-forward validation
│
├── backtesting/                   # Backtesting infrastructure
│   ├── __init__.py
│   ├── backtest_engine.py        # VectorBT Pro wrapper
│   ├── walk_forward.py           # Walk-forward analysis
│   └── report_generator.py       # Performance reporting
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_strategies/          # Strategy tests
│   ├── test_risk_management/     # Risk management tests
│   ├── test_data/                # Data pipeline tests
│   └── test_integration/         # Integration tests
│
├── configs/                       # Configuration files
│   ├── strategies.yaml           # Strategy parameters
│   ├── risk_limits.yaml          # Risk management settings
│   └── data_sources.yaml         # Data source configuration
│
├── logs/                          # Log files
│   ├── backtest/                 # Backtest logs
│   ├── paper_trading/            # Paper trading logs
│   └── live/                     # Live trading logs
│
├── results/                       # Backtest results
│   ├── walk_forward/             # Walk-forward analysis
│   ├── optimization/             # Parameter optimization
│   └── reports/                  # Performance reports
│
├── notebooks/                     # Analysis notebooks
│   ├── strategy_research/        # Strategy development
│   └── performance_analysis/     # Results analysis
│
├── docs/                          # Documentation
│   ├── STRATEGY_OVERVIEW.md
│   ├── IMPLEMENTATION_PLAN.md
│   └── API_REFERENCE.md
│
├── pyproject.toml                 # Project configuration
├── .env.example                   # Environment variables template
└── README.md                      # Project overview
```

---

## Strategy Implementations

### Base Strategy Interface

All strategies inherit from this abstract base class:

```python
# strategies/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """Strategy configuration with validation."""
    name: str
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.05)
    max_positions: int = Field(default=5, ge=1, le=20)
    enable_shorts: bool = False
    commission_rate: float = Field(default=0.0015, ge=0.0, le=0.01)
    slippage: float = Field(default=0.0015, ge=0.0, le=0.01)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signals(): Entry/exit signal generation
    - calculate_position_size(): Position sizing logic
    - get_strategy_name(): Strategy identifier
    
    Common functionality provided:
    - VectorBT Pro integration
    - Performance metrics calculation
    - Risk management validation
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate strategy configuration."""
        if self.config.risk_per_trade > 0.03:
            raise ValueError(f"Risk per trade too high: {self.config.risk_per_trade}")
        
        if self.config.commission_rate + self.config.slippage > 0.005:
            raise ValueError("Combined costs exceed 0.5%")
    
    @abstractmethod
    def generate_signals(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Generate trading signals.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            
        Returns:
            Dictionary containing:
            - 'long_entries': Boolean series for long entries
            - 'long_exits': Boolean series for long exits
            - 'short_entries': Boolean series for short entries (optional)
            - 'short_exits': Boolean series for short exits (optional)
            - 'stop_distance': Float series for stop loss distances
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes.
        
        Args:
            data: OHLCV DataFrame
            capital: Current account capital
            stop_distance: Stop loss distances
            
        Returns:
            Position sizes (number of shares) as pandas Series
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging/reporting."""
        pass
    
    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ):
        """
        Run backtest using VectorBT Pro.
        
        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital
            
        Returns:
            VectorBT Portfolio object
        """
        import vectorbtpro as vbt
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Calculate position sizes
        stop_distance = signals.get('stop_distance')
        position_sizes = self.calculate_position_size(
            data, initial_capital, stop_distance
        )
        
        # Run VectorBT backtest
        pf = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=signals['long_entries'],
            exits=signals['long_exits'],
            short_entries=signals.get('short_entries'),
            short_exits=signals.get('short_exits'),
            size=position_sizes,
            size_type='amount',
            init_cash=initial_capital,
            fees=self.config.commission_rate,
            slippage=self.config.slippage,
            sl_stop=stop_distance,
            freq='1D'
        )
        
        return pf
    
    def get_performance_metrics(self, pf) -> Dict:
        """Extract standardized performance metrics."""
        return {
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'sortino_ratio': pf.sortino_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate(),
            'profit_factor': pf.trades.profit_factor(),
            'avg_trade': pf.trades.returns.mean(),
            'total_trades': pf.trades.count(),
            'avg_winner': pf.trades.returns[pf.trades.returns > 0].mean(),
            'avg_loser': pf.trades.returns[pf.trades.returns < 0].mean(),
        }
```

### Strategy 1: GMM Regime Detection

```python
# strategies/gmm_regime.py

from strategies.base_strategy import BaseStrategy, StrategyConfig
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class RegimeMapping:
    """Maps GMM clusters to market regimes."""
    cluster_to_regime: Dict[int, str]
    mean_returns: Dict[int, float]
    cluster_counts: Dict[int, int]
    training_end_date: pd.Timestamp
    is_valid: bool


class GMMRegimeStrategy(BaseStrategy):
    """
    Gaussian Mixture Model regime detection with walk-forward validation.
    
    Features:
    - Yang-Zhang volatility (20-day)
    - Normalized SMA 20/50 crossover
    
    Regimes:
    - Bullish: 100% long
    - Neutral: 0% (cash)
    - Bearish: 0% (cash)
    
    Refit frequency: 63 days (quarterly)
    Minimum training: 252 days (1 year)
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.feature_cols = ["YangZhang_Vol_lag", "SMA_Cross_Norm_lag"]
        self.gmm_components = 3
        self.min_training_days = 252
        self.refit_frequency = 63
        self.min_cluster_samples = 10
        self.previous_mapping: Optional[RegimeMapping] = None
        
        # Models (will be fit during walk-forward)
        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.regime_mapping: Optional[RegimeMapping] = None
    
    def get_strategy_name(self) -> str:
        return "GMM_Regime_Detection"
    
    def calculate_yang_zhang_volatility(
        self, 
        high: pd.Series,
        low: pd.Series,
        open_price: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator.
        
        Formula: σ²_YZ = σ²_overnight + k·σ²_open_close + (1-k)·σ²_Rogers_Satchell
        
        More comprehensive than simple historical volatility.
        """
        # Overnight component (close to open)
        overnight = np.log(open_price / close.shift(1))
        overnight_var = overnight.rolling(window).var()
        
        # Open-to-close component
        open_close = np.log(close / open_price)
        open_close_var = open_close.rolling(window).var()
        
        # Rogers-Satchell component
        hl = np.log(high / low)
        hc = np.log(high / close)
        ho = np.log(high / open_price)
        lc = np.log(low / close)
        lo = np.log(low / open_price)
        
        rs = (ho * (ho - hc) + lo * (lo - lc)).rolling(window).mean()
        
        # Weight factor
        k = 0.34 / (1 + (window + 1) / (window - 1))
        
        # Combine components
        yang_zhang_var = overnight_var + k * open_close_var + (1 - k) * rs
        
        # Annualize
        yang_zhang_vol = np.sqrt(yang_zhang_var * 252)
        
        return yang_zhang_vol
    
    def calculate_sma_cross_normalized(
        self,
        close: pd.Series,
        short_window: int = 20,
        long_window: int = 50
    ) -> pd.Series:
        """
        Calculate normalized SMA crossover signal.
        
        Returns continuous signal: (SMA_short - SMA_long) / SMA_long
        Positive = bullish momentum, Negative = bearish momentum
        """
        sma_short = close.rolling(short_window).mean()
        sma_long = close.rolling(long_window).mean()
        
        normalized_cross = (sma_short - sma_long) / sma_long
        
        return normalized_cross
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for regime detection.
        
        Critical: Features must be lagged by 1 day to prevent look-ahead bias.
        """
        df = data.copy()
        
        # Calculate Yang-Zhang volatility
        df['YangZhang_Vol'] = self.calculate_yang_zhang_volatility(
            high=df['high'],
            low=df['low'],
            open_price=df['open'],
            close=df['close'],
            window=20
        )
        
        # Calculate normalized SMA crossover
        df['SMA_Cross_Norm'] = self.calculate_sma_cross_normalized(
            close=df['close'],
            short_window=20,
            long_window=50
        )
        
        # LAG FEATURES BY 1 DAY (prevent look-ahead)
        df['YangZhang_Vol_lag'] = df['YangZhang_Vol'].shift(1)
        df['SMA_Cross_Norm_lag'] = df['SMA_Cross_Norm'].shift(1)
        
        # Calculate forward returns for regime mapping (only used in training)
        df['Returns_OO'] = df['open'].pct_change()  # Open-to-open returns
        
        return df
    
    def create_regime_mapping(
        self,
        X_train: np.ndarray,
        gmm_model: GaussianMixture,
        returns_forward: np.ndarray,
        train_end_date: pd.Timestamp
    ) -> RegimeMapping:
        """
        Map GMM clusters to regimes based on forward returns in training data.
        
        CRITICAL: Uses ONLY training period returns to avoid look-ahead bias.
        """
        # Predict clusters for training data
        clusters = gmm_model.predict(X_train)
        
        # Calculate mean forward return for each cluster
        cluster_returns = {}
        cluster_counts = {}
        
        for cluster_id in range(self.gmm_components):
            mask = clusters == cluster_id
            cluster_counts[cluster_id] = mask.sum()
            
            if mask.sum() >= self.min_cluster_samples:
                cluster_returns[cluster_id] = returns_forward[mask].mean()
            else:
                cluster_returns[cluster_id] = np.nan
        
        # Check if all clusters are sufficiently populated
        is_valid = all(
            count >= self.min_cluster_samples 
            for count in cluster_counts.values()
        )
        
        # Sort clusters by return and assign regimes
        valid_clusters = {
            k: v for k, v in cluster_returns.items() 
            if not np.isnan(v)
        }
        
        if len(valid_clusters) >= 3:
            sorted_clusters = sorted(valid_clusters.items(), key=lambda x: x[1])
            regime_mapping = {
                sorted_clusters[0][0]: "Bearish",   # Lowest returns
                sorted_clusters[1][0]: "Neutral",   # Middle returns
                sorted_clusters[2][0]: "Bullish"    # Highest returns
            }
        else:
            # Fallback to previous mapping if available
            if self.previous_mapping and self.previous_mapping.is_valid:
                return self.previous_mapping
            else:
                # Default mapping (alphabetical)
                regime_mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
                is_valid = False
        
        mapping = RegimeMapping(
            cluster_to_regime=regime_mapping,
            mean_returns=cluster_returns,
            cluster_counts=cluster_counts,
            training_end_date=train_end_date,
            is_valid=is_valid
        )
        
        # Store for potential fallback
        if is_valid:
            self.previous_mapping = mapping
        
        return mapping
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate regime-based entry/exit signals with walk-forward validation.
        """
        # Engineer features
        df_features = self.engineer_features(data)
        
        # Initialize regime predictions
        regime_predictions = pd.Series(
            "Neutral", 
            index=df_features.index, 
            name="Regime"
        )
        
        # Walk-forward loop
        last_refit_idx = 0
        fold_num = 0
        
        for i in range(len(df_features)):
            # Check if it's time to refit
            days_since_refit = i - last_refit_idx
            
            if i >= self.min_training_days and days_since_refit >= self.refit_frequency:
                fold_num += 1
                
                # Define training window
                train_start = 0
                train_end = i
                
                # Extract training features
                X_train = df_features.loc[
                    train_start:train_end-1, 
                    self.feature_cols
                ].values
                
                # Remove NaN rows
                valid_mask = ~np.isnan(X_train).any(axis=1)
                X_train_clean = X_train[valid_mask]
                
                # Fit StandardScaler
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train_clean)
                
                # Fit GMM
                self.gmm = GaussianMixture(
                    n_components=self.gmm_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                self.gmm.fit(X_train_scaled)
                
                # Create regime mapping
                returns_forward_train = df_features.loc[
                    train_start:train_end-1, 
                    "Returns_OO"
                ].shift(-1).values[valid_mask]
                
                self.regime_mapping = self.create_regime_mapping(
                    X_train_scaled,
                    self.gmm,
                    returns_forward_train,
                    df_features.index[train_end-1]
                )
                
                last_refit_idx = i
            
            # Predict current observation using frozen models
            if self.scaler is not None and self.gmm is not None:
                X_current = df_features.loc[
                    i:i, 
                    self.feature_cols
                ].values
                
                if not np.isnan(X_current).any():
                    X_current_scaled = self.scaler.transform(X_current)
                    cluster_pred = self.gmm.predict(X_current_scaled)[0]
                    regime_pred = self.regime_mapping.cluster_to_regime.get(
                        cluster_pred, 
                        "Neutral"
                    )
                    regime_predictions.iloc[i] = regime_pred
        
        # Generate trading signals based on regime
        long_entries = regime_predictions == "Bullish"
        long_exits = regime_predictions != "Bullish"
        
        # Stop distance (not used since we exit on regime change)
        stop_distance = pd.Series(0.0, index=data.index)
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'stop_distance': stop_distance,
            'regime': regime_predictions
        }
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Position size for GMM: 100% allocation when Bullish, 0% otherwise.
        """
        signals = self.generate_signals(data)
        regime = signals['regime']
        
        # Binary sizing: 100% or 0%
        position_sizes = pd.Series(0.0, index=data.index)
        
        bullish_mask = regime == "Bullish"
        position_sizes[bullish_mask] = capital / data.loc[bullish_mask, 'close']
        
        return position_sizes
```

### Strategy 2: Five-Day Washout Mean Reversion

```python
# strategies/mean_reversion.py

from strategies.base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np


class MeanReversionStrategy(BaseStrategy):
    """
    Five-day washout mean reversion system.
    
    Entry: Close below 5-day low (washout) + 200-day MA uptrend
    Exit: Price above 5-day MA OR 7-day time limit
    Stop: 2x ATR below entry
    
    Designed for: Major indices (SPY, QQQ)
    Expected: 67% win rate, 4-5% CAGR, -10% max DD
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.washout_period = 5
        self.ma_filter_period = 200
        self.exit_ma_period = 5
        self.max_hold_days = 7
        self.atr_period = 14
        self.atr_stop_multiplier = 2.0
    
    def get_strategy_name(self) -> str:
        return "Five_Day_Washout_Mean_Reversion"
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate mean reversion signals."""
        df = data.copy()
        
        # Calculate 5-day rolling low (washout level)
        five_day_low = df['low'].rolling(self.washout_period).min()
        
        # Washout signal: close below 5-day low
        washout = df['close'] < five_day_low.shift(1)
        
        # Trend filter: 200-day MA
        ma_200 = df['close'].rolling(self.ma_filter_period).mean()
        uptrend = df['close'] > ma_200
        
        # Entry signal: washout in uptrend
        long_entries = washout & uptrend
        
        # Exit signal: price recovers above 5-day MA
        ma_5 = df['close'].rolling(self.exit_ma_period).mean()
        mean_revert_complete = df['close'] > ma_5
        
        # Calculate ATR for stops
        atr = self.calculate_atr(df, self.atr_period)
        stop_distance = atr * self.atr_stop_multiplier
        
        # Exit conditions (will be OR'd together in VectorBT)
        long_exits = mean_revert_complete
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'stop_distance': stop_distance,
            'time_exit_days': self.max_hold_days
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        ATR-based position sizing with capital constraint.
        """
        # Risk-based size
        position_size_risk = (capital * self.config.risk_per_trade) / stop_distance
        
        # Capital constraint
        position_size_capital = capital / data['close']
        
        # Take minimum (prevent oversizing)
        position_size = np.minimum(position_size_risk, position_size_capital)
        
        # Fill NaN with 0
        position_size = position_size.fillna(0)
        
        return position_size
```

### Strategy 3: Opening Range Breakout

```python
# strategies/orb.py

from strategies.base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout for intraday momentum.
    
    Entry: Breakout of first 5-minute range + 2.0x volume surge
    Exit: End-of-day (3:55 PM ET) OR stop loss
    
    Designed for: Liquid stocks with high relative volume
    Expected: 15-25% win rate, 2.5+ Sharpe, 3:1+ R:R
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.opening_minutes = 5
        self.atr_period = 14
        self.atr_stop_multiplier = 2.5
        self.volume_surge_multiplier = 2.0
        self.eod_exit_time = '15:55:00'
    
    def get_strategy_name(self) -> str:
        return "Opening_Range_Breakout"
    
    def calculate_opening_range(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate opening range from first 5 minutes.
        
        Args:
            data: Intraday OHLCV data (5-minute bars)
            
        Returns:
            DataFrame with opening_high, opening_low, opening_close columns
        """
        # Extract opening period (9:30-9:35 AM ET)
        opening_period = data.between_time('09:30', '09:35')
        
        # Calculate daily opening range
        opening_high = opening_period.groupby(
            opening_period.index.date
        )['high'].max()
        
        opening_low = opening_period.groupby(
            opening_period.index.date
        )['low'].min()
        
        opening_close = opening_period.groupby(
            opening_period.index.date
        )['close'].last()
        
        opening_open = opening_period.groupby(
            opening_period.index.date
        )['open'].first()
        
        # Determine directional bias
        bullish_bar = opening_close > opening_open
        
        # Convert to DataFrame
        opening_range = pd.DataFrame({
            'opening_high': opening_high,
            'opening_low': opening_low,
            'opening_close': opening_close,
            'bullish_bar': bullish_bar
        })
        
        # Forward-fill to match intraday data
        opening_range.index = pd.to_datetime(opening_range.index)
        opening_range = opening_range.reindex(
            data.index.date, 
            method='ffill'
        )
        opening_range.index = data.index
        
        return opening_range
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate ORB signals."""
        # Calculate opening range
        opening_range = self.calculate_opening_range(data)
        
        # Breakout signals
        long_breakout = (
            (data['close'] > opening_range['opening_high']) &
            opening_range['bullish_bar']
        )
        
        short_breakout = (
            (data['close'] < opening_range['opening_low']) &
            ~opening_range['bullish_bar']
        )
        
        # Volume confirmation
        volume_ma_20 = data['volume'].rolling(20).mean()
        volume_surge = data['volume'] > (
            volume_ma_20 * self.volume_surge_multiplier
        )
        
        # Confirmed entries
        long_entries = long_breakout & volume_surge
        
        if self.config.enable_shorts:
            short_entries = short_breakout & volume_surge
        else:
            short_entries = pd.Series(False, index=data.index)
        
        # Exit at end-of-day
        eod_exit = data.index.time >= pd.Timestamp(
            self.eod_exit_time
        ).time()
        
        long_exits = eod_exit
        short_exits = eod_exit
        
        # Calculate ATR for stops
        atr = self.calculate_atr(data, self.atr_period)
        stop_distance = atr * self.atr_stop_multiplier
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'stop_distance': stop_distance
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR (using daily bars for intraday strategy)."""
        # Resample intraday to daily for ATR calculation
        daily = data.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        high = daily['high']
        low = daily['low']
        close = daily['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # Forward-fill to intraday frequency
        atr = atr.reindex(data.index, method='ffill')
        
        return atr
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """ATR-based position sizing with capital constraint."""
        position_size_risk = (capital * self.config.risk_per_trade) / stop_distance
        position_size_capital = capital / data['close']
        position_size = np.minimum(position_size_risk, position_size_capital)
        return position_size.fillna(0)
```

### Strategy 4: Pairs Trading

```python
# strategies/pairs_trading.py

from strategies.base_strategy import BaseStrategy, StrategyConfig
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np
from typing import Tuple


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage via pairs trading.
    
    Entry: Spread diverges >2 std dev from mean
    Exit: Spread converges to mean (z-score crosses zero)
    Stop: Spread exceeds 3 std dev (relationship breakdown)
    
    Expected: 70-80% win rate, market-neutral returns
    """
    
    def __init__(
        self, 
        config: StrategyConfig,
        stock_a_symbol: str,
        stock_b_symbol: str
    ):
        super().__init__(config)
        self.stock_a_symbol = stock_a_symbol
        self.stock_b_symbol = stock_b_symbol
        self.spread_window = 60
        self.entry_z_score = 2.0
        self.exit_z_score = 0.0
        self.stop_z_score = 3.0
        self.min_cointegration_pvalue = 0.05
    
    def get_strategy_name(self) -> str:
        return f"Pairs_Trading_{self.stock_a_symbol}_{self.stock_b_symbol}"
    
    def test_cointegration(
        self, 
        stock_a: pd.Series, 
        stock_b: pd.Series
    ) -> Tuple[bool, float]:
        """
        Test if two price series are cointegrated using CADF.
        
        Returns:
            (is_cointegrated, p_value)
        """
        # Fit linear regression
        model = OLS(stock_a, stock_b).fit()
        residuals = model.resid
        
        # Augmented Dickey-Fuller test on residuals
        adf_result = adfuller(residuals)
        p_value = adf_result[1]
        
        # If p < 0.05, reject null hypothesis (stationary residuals)
        is_cointegrated = p_value < self.min_cointegration_pvalue
        
        return is_cointegrated, p_value
    
    def calculate_spread_z_score(
        self,
        stock_a_price: pd.Series,
        stock_b_price: pd.Series
    ) -> pd.Series:
        """
        Calculate normalized spread z-score.
        
        Spread = Stock_A / Stock_B
        Z-score = (Spread - Mean) / Std
        """
        # Calculate price ratio
        spread = stock_a_price / stock_b_price
        
        # Rolling statistics
        spread_mean = spread.rolling(self.spread_window).mean()
        spread_std = spread.rolling(self.spread_window).std()
        
        # Z-score
        z_score = (spread - spread_mean) / spread_std
        
        return z_score
    
    def generate_signals(
        self, 
        data_a: pd.DataFrame,
        data_b: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Generate pairs trading signals.
        
        Args:
            data_a: OHLCV for stock A
            data_b: OHLCV for stock B
        """
        # Test cointegration
        is_cointegrated, p_value = self.test_cointegration(
            data_a['close'], 
            data_b['close']
        )
        
        if not is_cointegrated:
            raise ValueError(
                f"Stocks not cointegrated (p={p_value:.4f}). "
                f"Cannot trade this pair."
            )
        
        # Calculate spread z-score
        z_score = self.calculate_spread_z_score(
            data_a['close'], 
            data_b['close']
        )
        
        # Entry signals
        long_spread_entry = z_score < -self.entry_z_score
        short_spread_entry = z_score > self.entry_z_score
        
        # Exit signals (mean reversion complete)
        exit_signal = abs(z_score) < abs(self.exit_z_score)
        
        # Stop loss (relationship breakdown)
        stop_signal = abs(z_score) > self.stop_z_score
        
        # Combine exits
        long_exits = exit_signal | stop_signal
        short_exits = exit_signal | stop_signal
        
        # For pairs trading, "long" = long A + short B
        # "short" = short A + long B
        return {
            'long_spread_entry': long_spread_entry,
            'short_spread_entry': short_spread_entry,
            'long_exits': long_exits,
            'short_exits': short_exits,
            'z_score': z_score,
            'is_cointegrated': is_cointegrated,
            'cointegration_pvalue': p_value
        }
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Equal dollar allocation to each leg.
        
        For pairs: $5000 long A, $5000 short B (market-neutral)
        """
        capital_per_leg = capital / 2
        position_size = capital_per_leg / data['close']
        return position_size
```

### Strategy 5: Semi-Volatility Momentum Portfolio

```python
# strategies/momentum_portfolio.py

from strategies.base_strategy import BaseStrategy, StrategyConfig
import pandas as pd
import numpy as np


class MomentumPortfolioStrategy(BaseStrategy):
    """
    Semi-volatility scaled momentum for multi-asset portfolios.
    
    Position sizing: Inverse Garman-Klass semi-volatility
    Signal: SMA 20/50 crossover
    Target: 15% portfolio volatility
    
    Expected: 1.44 Sortino, 24.75% CAGR
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.sma_short = 20
        self.sma_long = 50
        self.semi_vol_window = 60
        self.target_portfolio_vol = 0.15
        self.max_position_weight = 0.10
        self.min_position_weight = 0.01
    
    def get_strategy_name(self) -> str:
        return "Semi_Volatility_Momentum_Portfolio"
    
    def calculate_garman_klass_semi_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        open_price: pd.Series,
        close: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        Calculate Garman-Klass semi-volatility (downside only).
        
        Formula: GK_Var = 0.5 × [ln(H/L)]² - (2ln(2)-1) × [ln(C/O)]²
        """
        # Garman-Klass variance components
        hl_component = 0.5 * (np.log(high / low)) ** 2
        co_component = -(2 * np.log(2) - 1) * (np.log(close / open_price)) ** 2
        
        # Combine
        gk_variance = hl_component + co_component
        
        # Filter for negative returns only (semi-volatility)
        returns = close.pct_change()
        negative_mask = returns < 0
        downside_variance = gk_variance.copy()
        downside_variance[~negative_mask] = np.nan
        
        # Rolling mean
        semi_vol_rolling = downside_variance.rolling(window).mean()
        
        # Annualize
        semi_vol_annualized = np.sqrt(semi_vol_rolling * 252)
        
        # Forward-fill NaN
        semi_vol_annualized = semi_vol_annualized.fillna(method='ffill')
        
        # Fallback for remaining NaN
        semi_vol_annualized = semi_vol_annualized.fillna(0.20)  # Default 20% vol
        
        return semi_vol_annualized
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate momentum signals."""
        # Calculate SMAs
        sma_short = data['close'].rolling(self.sma_short).mean()
        sma_long = data['close'].rolling(self.sma_long).mean()
        
        # Momentum signal
        long_entries = sma_short > sma_long
        long_exits = sma_short < sma_long
        
        # Calculate semi-volatility for position sizing
        semi_vol = self.calculate_garman_klass_semi_volatility(
            high=data['high'],
            low=data['low'],
            open_price=data['open'],
            close=data['close'],
            window=self.semi_vol_window
        )
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'semi_vol': semi_vol,
            'stop_distance': pd.Series(0.0, index=data.index)
        }
    
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Inverse semi-volatility position sizing.
        
        Lower semi-vol → Larger position
        Higher semi-vol → Smaller position
        """
        signals = self.generate_signals(data)
        semi_vol = signals['semi_vol']
        
        # Inverse volatility weight
        inv_vol_weight = self.target_portfolio_vol / semi_vol
        
        # Normalize by portfolio size (assume 20 assets)
        portfolio_size = 20
        position_weight = inv_vol_weight / portfolio_size
        
        # Clip to min/max
        position_weight = position_weight.clip(
            self.min_position_weight,
            self.max_position_weight
        )
        
        # Convert to share count
        position_size = (capital * position_weight) / data['close']
        
        return position_size
```

---

## Core Components

### Portfolio Manager

```python
# core/portfolio_manager.py

from typing import Dict, List
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from core.risk_manager import RiskManager


class PortfolioManager:
    """
    Orchestrates multiple strategies with portfolio-level risk management.
    """
    
    def __init__(
        self,
        strategies: List[BaseStrategy],
        capital: float,
        risk_manager: RiskManager
    ):
        self.strategies = strategies
        self.capital = capital
        self.risk_manager = risk_manager
        self.positions = {}
        self.equity_curve = []
    
    def allocate_capital(self) -> Dict[str, float]:
        """
        Allocate capital across strategies.
        
        Default: Equal weight
        Advanced: Risk parity, volatility targeting
        """
        strategy_count = len(self.strategies)
        allocation_per_strategy = self.capital / strategy_count
        
        allocations = {
            strategy.get_strategy_name(): allocation_per_strategy
            for strategy in self.strategies
        }
        
        return allocations
    
    def check_portfolio_heat(self) -> float:
        """Calculate total portfolio heat across all positions."""
        total_heat = sum(
            pos['risk'] for pos in self.positions.values()
        )
        return total_heat
    
    def can_take_position(
        self, 
        strategy_name: str, 
        position_risk: float
    ) -> bool:
        """
        Check if new position would exceed heat limit.
        """
        current_heat = self.check_portfolio_heat()
        new_heat = current_heat + position_risk
        
        max_heat = self.risk_manager.max_portfolio_heat
        
        if new_heat > max_heat:
            return False
        
        return True
    
    def run_multi_strategy_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float
    ) -> Dict:
        """
        Run coordinated backtest across all strategies.
        
        Args:
            data: Dict mapping symbol -> OHLCV DataFrame
            initial_capital: Starting capital
            
        Returns:
            Combined portfolio results
        """
        allocations = self.allocate_capital()
        strategy_results = {}
        
        # Run each strategy independently
        for strategy in self.strategies:
            strategy_name = strategy.get_strategy_name()
            strategy_capital = allocations[strategy_name]
            
            # Get data for this strategy
            # (May need multiple symbols for pairs trading)
            strategy_data = data.get(strategy_name)
            
            # Run backtest
            pf = strategy.backtest(strategy_data, strategy_capital)
            
            strategy_results[strategy_name] = {
                'portfolio': pf,
                'metrics': strategy.get_performance_metrics(pf)
            }
        
        # Combine results
        combined_results = self.combine_strategy_results(strategy_results)
        
        return combined_results
    
    def combine_strategy_results(
        self, 
        strategy_results: Dict
    ) -> Dict:
        """
        Aggregate performance across strategies.
        """
        # Extract equity curves
        equity_curves = []
        for name, result in strategy_results.items():
            pf = result['portfolio']
            equity = pf.value()
            equity_curves.append(equity)
        
        # Combine (simple sum)
        combined_equity = sum(equity_curves)
        
        # Calculate portfolio metrics
        returns = combined_equity.pct_change()
        
        portfolio_metrics = {
            'total_return': (combined_equity.iloc[-1] / combined_equity.iloc[0]) - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (combined_equity / combined_equity.cummax() - 1).min(),
            'strategy_results': strategy_results
        }
        
        return portfolio_metrics
```

### Risk Manager

```python
# core/risk_manager.py

from typing import Dict, Optional
import pandas as pd


class RiskManager:
    """
    Portfolio-level risk management.
    
    - Portfolio heat limits
    - Drawdown circuit breakers
    - Position size validation
    """
    
    def __init__(
        self,
        max_portfolio_heat: float = 0.08,
        max_position_risk: float = 0.02,
        drawdown_thresholds: Optional[Dict[float, str]] = None
    ):
        self.max_portfolio_heat = max_portfolio_heat
        self.max_position_risk = max_position_risk
        
        if drawdown_thresholds is None:
            self.drawdown_thresholds = {
                0.10: 'WARNING',
                0.15: 'REDUCE_SIZE',
                0.20: 'STOP_TRADING',
                0.25: 'CRITICAL'
            }
        else:
            self.drawdown_thresholds = drawdown_thresholds
        
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.trading_enabled = True
        self.risk_multiplier = 1.0
    
    def update_equity(self, equity: float):
        """Update equity and check for circuit breakers."""
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Check drawdown
        drawdown = self.calculate_drawdown()
        self.check_circuit_breakers(drawdown)
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        return drawdown
    
    def check_circuit_breakers(self, drawdown: float):
        """
        Trigger actions based on drawdown thresholds.
        """
        if drawdown >= 0.20:
            self.trading_enabled = False
            self.risk_multiplier = 0.0
            print(f"CIRCUIT BREAKER: Trading halted at {drawdown:.1%} drawdown")
        
        elif drawdown >= 0.15:
            self.risk_multiplier = 0.5
            print(f"RISK REDUCTION: Position size reduced 50% at {drawdown:.1%} drawdown")
        
        elif drawdown >= 0.10:
            print(f"WARNING: {drawdown:.1%} drawdown reached")
        
        else:
            # Normal operations
            self.trading_enabled = True
            self.risk_multiplier = 1.0
    
    def validate_position_size(
        self,
        position_size: float,
        price: float,
        capital: float
    ) -> bool:
        """
        Validate position size doesn't exceed limits.
        """
        position_value = position_size * price
        position_pct = position_value / capital
        
        if position_pct > 1.0:
            print(f"Position size exceeds capital: {position_pct:.1%}")
            return False
        
        if position_pct > 0.50:
            print(f"Warning: Large position size: {position_pct:.1%}")
        
        return True
```

### Position Sizer Utilities

```python
# utils/position_sizing.py

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_position_size_atr(
    capital: float,
    close: pd.Series,
    atr: pd.Series,
    atr_multiplier: float = 2.5,
    risk_pct: float = 0.02
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    ATR-based position sizing with capital constraint.
    
    Args:
        capital: Account size
        close: Current price
        atr: Average True Range
        atr_multiplier: Stop distance multiplier
        risk_pct: Risk per trade (default 2%)
    
    Returns:
        (position_size, actual_risk, constrained_flag)
    """
    # Stop distance
    stop_distance = atr * atr_multiplier
    
    # Risk-based position size
    position_size_risk = (capital * risk_pct) / stop_distance
    
    # Capital constraint
    position_size_capital = capital / close
    
    # Take minimum
    position_size = np.minimum(position_size_risk, position_size_capital)
    
    # Actual risk achieved
    actual_risk = (position_size * stop_distance) / capital
    
    # Constrained flag
    constrained = position_size == position_size_capital
    
    return position_size, actual_risk, constrained


def calculate_position_size_semi_volatility(
    capital: float,
    close: pd.Series,
    semi_vol: pd.Series,
    target_vol: float = 0.15,
    portfolio_size: int = 20,
    max_weight: float = 0.10,
    min_weight: float = 0.01
) -> pd.Series:
    """
    Semi-volatility based position sizing (inverse volatility).
    
    Args:
        capital: Account size
        close: Current price
        semi_vol: Garman-Klass semi-volatility
        target_vol: Target portfolio volatility (15%)
        portfolio_size: Number of assets in portfolio
        max_weight: Maximum position weight (10%)
        min_weight: Minimum position weight (1%)
    
    Returns:
        Position sizes (number of shares)
    """
    # Inverse volatility weight
    inv_vol_weight = target_vol / semi_vol
    
    # Normalize by portfolio size
    position_weight = inv_vol_weight / portfolio_size
    
    # Clip to limits
    position_weight = position_weight.clip(min_weight, max_weight)
    
    # Convert to shares
    position_size = (capital * position_weight) / close
    
    return position_size


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion optimal position size.
    
    Formula: f* = (p × b - q) / b
    where:
        p = win rate
        q = loss rate (1 - p)
        b = win/loss ratio
    
    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss == 0:
        return 0.0
    
    win_loss_ratio = abs(avg_win / avg_loss)
    loss_rate = 1 - win_rate
    
    kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
    
    # Apply Kelly fraction constraint (never exceed 25%)
    kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
    
    return kelly_fraction
```

---

## Risk Management Framework

### Overview: Integrated De-Risking Strategy

**Philosophy:** Risk management must be integrated from day one, not added as an afterthought. Multiple layers of protection ensure system robustness across different failure modes.

**Core Principle:** Risk controls should be independently verifiable, mathematically sound, and enforce hard limits without exceptions.

### The Five Risk Management Layers

```
Layer 1: Position Sizing (Individual Trade Risk)
    |
    v
Layer 2: Portfolio Heat (Aggregate Risk Exposure)
    |
    v  
Layer 3: Stop Loss System (Multi-Layer Exits)
    |
    v
Layer 4: Volume Confirmation (Signal Quality)
    |
    v
Layer 5: Multi-Symbol Validation (Robustness Check)
```

---

### Layer 1: Position Sizing (Capital-Constrained)

**Purpose:** Limit risk per individual trade to 2% of capital while respecting absolute capital constraints.

**Implementation:**
```python
# utils/position_sizing.py

def calculate_position_size_capital_constrained(
    capital: float,
    close: pd.Series,
    atr: pd.Series,
    atr_multiplier: float = 2.5,
    risk_pct: float = 0.02
) -> pd.Series:
    """
    ATR-based position sizing with absolute capital constraint.
    
    Key Innovation: Uses MINIMUM of risk-based and capital-based sizing.
    This prevents position sizes from exceeding 100% of capital.
    
    Args:
        capital: Account size
        close: Current price
        atr: Average True Range  
        atr_multiplier: Stop distance (default 2.5x ATR)
        risk_pct: Risk per trade (default 2%)
        
    Returns:
        Position sizes in shares (never exceeds capital / price)
    """
    # Stop distance
    stop_distance = atr * atr_multiplier
    
    # Risk-based sizing (traditional approach)
    position_size_risk = (capital * risk_pct) / stop_distance
    
    # Capital-based sizing (hard constraint)
    position_size_capital = capital / close
    
    # Take MINIMUM - ensures we never exceed available capital
    position_size = np.minimum(position_size_risk, position_size_capital)
    
    # Handle edge cases
    position_size = position_size.fillna(0).replace([np.inf, -np.inf], 0)
    
    return position_size
```

**Verification:**
- Mean position size: 10-30% of capital (target range)
- Max position size: Never exceeds 100% of capital (hard constraint)
- Position size = 0 when ATR is NaN or invalid

**Mathematical Proof:**
```
Given: capital = C, close = P, stop_distance = S
Risk-based: size_risk = (C * 0.02) / S
Capital-based: size_capital = C / P

Position value = size * P
If size = size_risk: value = [(C * 0.02) / S] * P
If size = size_capital: value = [C / P] * P = C

By taking min(size_risk, size_capital), we ensure:
  position_value <= C (always)
```

---

### Layer 2: Portfolio Heat Management

**Purpose:** Limit total risk exposure across all open positions to 6-8% of capital.

**Implementation:**
```python
# utils/portfolio_heat.py

class PortfolioHeatManager:
    """
    Tracks aggregate risk across all positions.
    Rejects new trades if total portfolio heat would exceed limit.
    """
    
    def __init__(self, max_heat: float = 0.08):
        """
        Args:
            max_heat: Maximum portfolio heat (default 8%)
        """
        self.max_heat = max_heat
        self.active_positions = {}  # symbol -> risk_amount
    
    def calculate_current_heat(self, capital: float) -> float:
        """
        Calculate current portfolio heat as percentage of capital.
        
        Returns:
            Current heat (0.0 to 1.0)
        """
        total_risk = sum(self.active_positions.values())
        return total_risk / capital
    
    def can_accept_trade(
        self,
        symbol: str,
        position_risk: float,
        capital: float
    ) -> bool:
        """
        Check if new trade would exceed heat limit.
        
        Args:
            symbol: Symbol for new trade
            position_risk: Dollar risk for new trade
            capital: Current account size
            
        Returns:
            True if trade accepted, False if rejected
        """
        current_heat = self.calculate_current_heat(capital)
        new_heat = (sum(self.active_positions.values()) + position_risk) / capital
        
        if new_heat > self.max_heat:
            print(f"REJECTED: Trade would increase heat from {current_heat:.1%} to {new_heat:.1%}")
            print(f"Max heat: {self.max_heat:.1%}")
            return False
        
        return True
    
    def add_position(self, symbol: str, risk_amount: float):
        """Add new position to heat tracking."""
        self.active_positions[symbol] = risk_amount
    
    def remove_position(self, symbol: str):
        """Remove position from heat tracking (trade closed)."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
    
    def update_position_risk(self, symbol: str, new_risk: float):
        """Update risk for existing position (e.g., trailing stop moved)."""
        if symbol in self.active_positions:
            self.active_positions[symbol] = new_risk
```

**Usage Example:**
```python
heat_manager = PortfolioHeatManager(max_heat=0.08)
capital = 100000

# Existing positions
heat_manager.add_position('SPY', 2000)  # $2,000 at risk
heat_manager.add_position('QQQ', 2500)  # $2,500 at risk
heat_manager.add_position('IWM', 2000)  # $2,000 at risk

# Current heat: $6,500 / $100,000 = 6.5%

# New signal for AAPL with $2,000 risk
if heat_manager.can_accept_trade('AAPL', 2000, capital):
    # Would be 8.5% - REJECTED
    print("Taking trade")
else:
    # Heat would exceed 8% limit
    print("Trade rejected - heat limit")
```

**Professional Standard:**
- Max heat 6-8% (hard limit, NO exceptions)
- Heat limit enforced BEFORE trade entry
- Heat recalculated as stops trail (risk reduces over time)

---

### Layer 3: Multi-Layer Stop Loss System

**Purpose:** Protect capital through multiple independent exit mechanisms.

**Why Multiple Layers:** Single stop type = single point of failure. Multiple stops provide redundancy and handle different market conditions.

**Implementation:**
```python
# utils/stop_loss.py

class MultiLayerStopSystem:
    """
    Three-layer stop loss system:
    1. ATR-based initial stop (volatility-adjusted)
    2. Time-based stop (prevents indefinite holding)
    3. Trailing stop (locks in profits)
    """
    
    def __init__(
        self,
        entry_price: float,
        entry_time: pd.Timestamp,
        atr: float,
        atr_multiplier: float = 2.5,
        max_hold_days: int = 5,
        trailing_activation_rr: float = 2.0,
        trailing_distance_atr: float = 1.5
    ):
        """
        Args:
            entry_price: Entry price for position
            entry_time: Entry timestamp
            atr: ATR at entry
            atr_multiplier: Initial stop distance (default 2.5x ATR)
            max_hold_days: Maximum days to hold (default 5)
            trailing_activation_rr: R:R to activate trailing (default 2:1)
            trailing_distance_atr: Trailing stop distance (default 1.5x ATR)
        """
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.atr = atr
        
        # Layer 1: ATR stop
        self.atr_stop = entry_price - (atr * atr_multiplier)
        self.stop_distance = atr * atr_multiplier
        
        # Layer 2: Time stop
        self.max_hold_time = entry_time + pd.Timedelta(days=max_hold_days)
        
        # Layer 3: Trailing stop
        self.trailing_stop = None
        self.trailing_active = False
        self.trailing_activation_price = entry_price + (self.stop_distance * trailing_activation_rr)
        self.trailing_distance = atr * trailing_distance_atr
    
    def get_active_stop(self, current_price: float, current_time: pd.Timestamp) -> tuple:
        """
        Determine which stop is active.
        
        Returns:
            (stop_price, stop_type)
        """
        # Check if time stop triggered
        if current_time >= self.max_hold_time:
            return (current_price, 'TIME_STOP')
        
        # Activate trailing stop if 2:1 R:R achieved
        if not self.trailing_active and current_price >= self.trailing_activation_price:
            self.trailing_active = True
            self.trailing_stop = current_price - self.trailing_distance
        
        # Update trailing stop if active
        if self.trailing_active:
            new_trailing = current_price - self.trailing_distance
            if new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing
            
            # Use tightest of ATR or trailing
            if self.trailing_stop > self.atr_stop:
                return (self.trailing_stop, 'TRAILING_STOP')
        
        # Default: ATR stop
        return (self.atr_stop, 'ATR_STOP')
    
    def is_stop_triggered(self, current_price: float, current_time: pd.Timestamp) -> tuple:
        """
        Check if any stop is triggered.
        
        Returns:
            (is_triggered: bool, stop_price: float, stop_type: str)
        """
        stop_price, stop_type = self.get_active_stop(current_price, current_time)
        
        # Time stop always triggers at max hold
        if stop_type == 'TIME_STOP':
            return (True, stop_price, stop_type)
        
        # Price stops trigger if current price below stop
        is_triggered = current_price <= stop_price
        return (is_triggered, stop_price, stop_type)
```

**Stop Priority Logic:**
1. Time stop triggers at day 5 regardless of price (prevents indefinite holding)
2. Trailing stop activates at 2:1 R:R (locks in profit)
3. ATR stop as baseline (volatility-adjusted protection)

**Professional Standard:**
- NEVER remove stops once placed
- NEVER widen stops after entry
- Trail stops only tighter, never wider
- Time stop prevents "hope trading"

---

### Layer 4: Volume Confirmation

**Purpose:** Filter false signals by requiring strong volume on breakouts.

**Research Basis:** Article finding - "Volume indicators measure trend strength and reduce false signals"

**Implementation:**
```python
# Applied to ALL price breakout strategies

def add_volume_confirmation(
    price_signal: pd.Series,
    volume: pd.Series,
    volume_threshold: float = 1.5,
    volume_window: int = 20
) -> pd.Series:
    """
    Add volume confirmation to price breakout signals.
    
    Args:
        price_signal: Boolean series of price breakout signals
        volume: Volume data
        volume_threshold: Multiplier for average volume (default 1.5x)
        volume_window: Lookback for volume average (default 20)
        
    Returns:
        Confirmed signals (price breakout AND volume confirmation)
    """
    # Calculate rolling average volume
    volume_ma = volume.rolling(volume_window).mean()
    
    # Volume must exceed threshold * average
    volume_confirmed = volume > (volume_ma * volume_threshold)
    
    # Combine with price signal
    confirmed_signal = price_signal & volume_confirmed
    
    return confirmed_signal
```

**Application to ORB Strategy:**
```python
# Opening Range Breakout with volume confirmation
price_breakout = close > opening_range_high
volume_filter = volume > (volume.rolling(20).mean() * 1.5)
atr_filter = atr > atr_threshold

# Final entry signal
entry_signal = price_breakout & volume_filter & atr_filter
```

**Verification:**
- Backtest WITH volume filter
- Backtest WITHOUT volume filter  
- Volume filter should:
  - Reduce number of trades
  - Improve win rate
  - Improve Sharpe ratio
  - Reduce max drawdown

**Professional Standard:**
- Minimum 1.5x average volume for breakouts
- Higher thresholds (2.0x) for volatile stocks
- Volume confirmation MANDATORY for all breakout strategies

---

### Layer 5: Multi-Symbol Validation

**Purpose:** Verify strategy robustness across different market segments.

**Research Basis:** Article finding - "Single symbol optimization often leads to overfitting"

**Implementation:**
```python
# validation/multi_symbol_test.py

def validate_strategy_multi_symbol(
    strategy,
    test_symbols: list = ['SPY', 'QQQ', 'IWM'],
    min_sharpe: float = 1.5,
    max_drawdown: float = 0.15
) -> dict:
    """
    Test strategy across multiple symbols to verify robustness.
    
    Args:
        strategy: Strategy instance to test
        test_symbols: List of symbols (default: SPY, QQQ, IWM)
        min_sharpe: Minimum acceptable Sharpe (default 1.5)
        max_drawdown: Maximum acceptable drawdown (default 15%)
        
    Returns:
        Validation results with pass/fail for each symbol
    """
    results = {}
    
    for symbol in test_symbols:
        # Fetch data
        data = fetch_data(symbol, start_date, end_date)
        
        # Run backtest
        pf = strategy.backtest(data, capital=100000)
        
        # Check criteria
        sharpe = pf.sharpe_ratio
        max_dd = pf.max_drawdown
        
        passed = (sharpe >= min_sharpe) and (max_dd <= max_drawdown)
        
        results[symbol] = {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'passed': passed
        }
    
    # Strategy must pass on ALL symbols
    all_passed = all(r['passed'] for r in results.values())
    
    results['validation_passed'] = all_passed
    
    return results
```

**Validation Criteria:**
- Test on minimum 3 symbols representing different market segments
- SPY (S&P 500), QQQ (Nasdaq), IWM (Russell 2000) recommended
- Strategy must meet performance criteria on ALL symbols
- If strategy only works on 1 symbol → overfitted → REJECT

**Professional Standard:**
- Multi-symbol validation BEFORE declaring strategy production-ready
- Document performance across all test symbols
- Reject strategies with high variance across symbols
- Prefer strategies with consistent performance

---

### Enhanced Backtesting Requirements

**Cost Modeling (MANDATORY):**
```python
# Must include BOTH fees and slippage in ALL backtests

pf = vbt.PF.from_signals(
    close,
    entries,
    exits,
    size=position_sizes,
    fees=0.002,      # 0.2% transaction fees
    slippage=0.0015, # 0.15% slippage estimate
    init_cash=100000
)

# Total round-trip cost: 0.35%
# Article finding: "Slippage is a major algorithmic trading risk"
```

**Benchmark Comparisons (MANDATORY):**
```python
def compare_to_benchmarks(
    strategy_returns: pd.Series,
    price_data: pd.DataFrame,
    spy_data: pd.DataFrame
) -> dict:
    """
    Compare strategy to buy-and-hold and SPY benchmark.
    
    Strategy must outperform BOTH to justify complexity.
    """
    # Buy and Hold
    bh_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1
    
    # SPY benchmark
    spy_return = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[0]) - 1
    
    # Strategy
    strategy_return = strategy_returns.sum()
    
    # Sharpe ratios
    strategy_sharpe = calculate_sharpe(strategy_returns)
    bh_sharpe = calculate_sharpe(price_data['close'].pct_change())
    spy_sharpe = calculate_sharpe(spy_data['close'].pct_change())
    
    results = {
        'strategy_return': strategy_return,
        'bh_return': bh_return,
        'spy_return': spy_return,
        'strategy_sharpe': strategy_sharpe,
        'bh_sharpe': bh_sharpe,
        'spy_sharpe': spy_sharpe,
        'beats_bh': strategy_return > bh_return,
        'beats_spy': strategy_return > spy_return
    }
    
    # Strategy must beat both
    results['benchmark_validation_passed'] = (
        results['beats_bh'] and results['beats_spy']
    )
    
    return results
```

**Additional Metrics to Track:**
```python
# Performance Analysis
win_rate = wins / total_trades
risk_reward_ratio = avg_win / avg_loss
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
max_consecutive_losses = calculate_max_consecutive_losses(trades)

# Professional Standard
assert win_rate >= 0.40, "Win rate too low"
assert risk_reward_ratio >= 2.0, "Risk:reward insufficient"
assert expectancy > 0, "Negative expectancy"
assert max_consecutive_losses <= 8, "Drawdown risk too high"
```

---

### Risk Management Integration Checklist

**Before declaring strategy production-ready:**

**Layer 1: Position Sizing**
- [ ] Capital-constrained formula implemented
- [ ] Mean position size 10-30% of capital
- [ ] Max position size never exceeds 100%
- [ ] Edge cases handled (NaN, Inf, zero volume)

**Layer 2: Portfolio Heat**
- [ ] Heat manager tracks all open positions
- [ ] 6-8% max heat hard limit enforced
- [ ] Heat recalculated as stops trail
- [ ] New trades rejected if heat exceeded

**Layer 3: Stop Losses**
- [ ] ATR-based initial stop
- [ ] Time-based max hold (5 days)
- [ ] Trailing stop (activates at 2:1 R:R)
- [ ] Never widen stops after entry

**Layer 4: Volume Confirmation**
- [ ] Volume filter on all breakout entries
- [ ] Minimum 1.5x average volume threshold
- [ ] Backtest comparison (with/without volume)
- [ ] Volume filter improves Sharpe ratio

**Layer 5: Multi-Symbol Validation**
- [ ] Tested on SPY, QQQ, IWM minimum
- [ ] Meets performance criteria on all symbols
- [ ] Consistent behavior across symbols
- [ ] No single-symbol overfitting

**Enhanced Backtesting:**
- [ ] Slippage included (0.15%)
- [ ] Transaction fees included (0.2%)
- [ ] Benchmarked vs buy-and-hold
- [ ] Benchmarked vs SPY
- [ ] Strategy beats both benchmarks

**If ANY checkbox unchecked: Strategy NOT ready for production**

---

## Data Pipeline

### Alpaca Data Client

```python
# data/alpaca_client.py

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
from typing import Optional


class AlpacaDataClient:
    """
    Interface to Alpaca market data API.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.client = StockHistoricalDataClient(api_key, secret_key)
    
    def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = '5Min',
        adjustment: str = 'all'
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            adjustment: Price adjustment ('raw', 'split', 'dividend', 'all')
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Convert timeframe string to Alpaca TimeFrame
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, TimeFrame.Minute),
            '15Min': TimeFrame(15, TimeFrame.Minute),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day
        }
        
        tf = timeframe_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Create request
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=datetime.fromisoformat(start_date),
            end=datetime.fromisoformat(end_date),
            timeframe=tf,
            adjustment=adjustment
        )
        
        # Fetch data
        bars = self.client.get_stock_bars(request)
        
        # Convert to DataFrame
        df = bars.df
        
        # Standardize column names
        df.columns = ['open', 'high', 'low', 'close', 'volume', 
                      'trade_count', 'vwap']
        
        # Drop unnecessary columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Check for common data quality issues.
        """
        issues = []
        
        # Check for NaN
        if df.isnull().any().any():
            issues.append("Contains NaN values")
        
        # Check for zero volume
        if (df['volume'] == 0).any():
            issues.append("Contains zero volume bars")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['close']) |
            (df['low'] > df['close']) |
            (df['high'] < df['open']) |
            (df['low'] > df['open'])
        )
        if invalid_ohlc.any():
            issues.append("Invalid OHLC relationships")
        
        # Check for weekend data (should be filtered by Alpaca)
        if (df.index.dayofweek >= 5).any():
            issues.append("Contains weekend data")
        
        if issues:
            print("Data quality issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
```

### Multi-Timeframe Manager

```python
# data/mtf_manager.py

import pandas as pd
from typing import Dict, List


class MultiTimeframeManager:
    """
    Manages multi-timeframe data alignment.
    """
    
    def __init__(self, base_timeframe: str = '5Min'):
        self.base_timeframe = base_timeframe
        self.timeframes = {
            '5Min': '5T',
            '15Min': '15T',
            '1Hour': '1H',
            '1Day': '1D'
        }
    
    def resample_to_timeframe(
        self,
        data: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to target timeframe.
        
        Args:
            data: OHLCV DataFrame at base timeframe
            target_timeframe: Target timeframe ('1Hour', '1Day', etc.)
        
        Returns:
            Resampled DataFrame
        """
        freq = self.timeframes.get(target_timeframe)
        if freq is None:
            raise ValueError(f"Invalid timeframe: {target_timeframe}")
        
        # Resample OHLCV
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop NaN rows (non-trading periods)
        resampled = resampled.dropna()
        
        return resampled
    
    def align_timeframes(
        self,
        data: pd.DataFrame,
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Create aligned multi-timeframe dataset.
        
        Returns dict mapping timeframe -> DataFrame
        """
        aligned = {}
        
        for tf in timeframes:
            aligned[tf] = self.resample_to_timeframe(data, tf)
        
        return aligned
```

---

## Testing Requirements

### Unit Tests

```python
# tests/test_strategies/test_gmm_regime.py

import pytest
import pandas as pd
import numpy as np
from strategies.gmm_regime import GMMRegimeStrategy, StrategyConfig


def test_gmm_initialization():
    """Test GMM strategy initialization."""
    config = StrategyConfig(name="GMM_Test")
    strategy = GMMRegimeStrategy(config)
    
    assert strategy.gmm_components == 3
    assert strategy.min_training_days == 252
    assert strategy.refit_frequency == 63


def test_yang_zhang_volatility():
    """Test Yang-Zhang volatility calculation."""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100) * 2,
        'high': 102 + np.random.randn(100) * 2,
        'low': 98 + np.random.randn(100) * 2,
        'close': 100 + np.random.randn(100) * 2,
    }, index=dates)
    
    config = StrategyConfig(name="GMM_Test")
    strategy = GMMRegimeStrategy(config)
    
    vol = strategy.calculate_yang_zhang_volatility(
        high=data['high'],
        low=data['low'],
        open_price=data['open'],
        close=data['close'],
        window=20
    )
    
    # Check output
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(data)
    assert vol.iloc[-1] > 0  # Volatility should be positive
    assert not np.isnan(vol.iloc[-1])  # Last value should be valid


def test_feature_engineering():
    """Test feature engineering with lag."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100) * 2,
        'high': 102 + np.random.randn(100) * 2,
        'low': 98 + np.random.randn(100) * 2,
        'close': 100 + np.random.randn(100) * 2,
    }, index=dates)
    
    config = StrategyConfig(name="GMM_Test")
    strategy = GMMRegimeStrategy(config)
    
    df_features = strategy.engineer_features(data)
    
    # Check columns exist
    assert 'YangZhang_Vol_lag' in df_features.columns
    assert 'SMA_Cross_Norm_lag' in df_features.columns
    
    # Check lag (value at index i should equal unlagged at i-1)
    assert pd.isna(df_features['YangZhang_Vol_lag'].iloc[0])
    assert df_features['YangZhang_Vol_lag'].iloc[50] == df_features['YangZhang_Vol'].iloc[49]


def test_regime_mapping():
    """Test regime mapping creation."""
    config = StrategyConfig(name="GMM_Test")
    strategy = GMMRegimeStrategy(config)
    
    # Mock training data
    X_train = np.random.randn(100, 2)
    returns_forward = np.random.randn(100) * 0.01
    
    # Fit GMM
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_train)
    
    # Create mapping
    mapping = strategy.create_regime_mapping(
        X_train, gmm, returns_forward, pd.Timestamp('2020-12-31')
    )
    
    # Check mapping structure
    assert len(mapping.cluster_to_regime) == 3
    assert all(regime in ["Bearish", "Neutral", "Bullish"] 
               for regime in mapping.cluster_to_regime.values())
    assert mapping.is_valid in [True, False]
```

### Integration Tests

```python
# tests/test_integration/test_full_backtest.py

import pytest
import pandas as pd
from strategies.gmm_regime import GMMRegimeStrategy, StrategyConfig
from data.alpaca_client import AlpacaDataClient


@pytest.fixture
def sample_data():
    """Load sample historical data for testing."""
    # In real tests, load from parquet file
    # For now, generate synthetic data
    dates = pd.date_range('2019-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic price movement
    returns = np.random.randn(len(dates)) * 0.01
    price = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.002),
        'high': price * (1 + abs(np.random.randn(len(dates))) * 0.005),
        'low': price * (1 - abs(np.random.randn(len(dates))) * 0.005),
        'close': price,
        'volume': np.random.randint(1e6, 10e6, len(dates))
    }, index=dates)
    
    return data


def test_gmm_full_backtest(sample_data):
    """Test full GMM strategy backtest."""
    config = StrategyConfig(
        name="GMM_Test",
        risk_per_trade=0.02,
        commission_rate=0.0015,
        slippage=0.0015
    )
    
    strategy = GMMRegimeStrategy(config)
    
    # Run backtest
    pf = strategy.backtest(sample_data, initial_capital=10000)
    
    # Check results
    metrics = strategy.get_performance_metrics(pf)
    
    # Sanity checks
    assert metrics['total_trades'] > 0, "No trades executed"
    assert metrics['sharpe_ratio'] is not None
    assert -1.0 <= metrics['max_drawdown'] <= 0.0
    assert 0.0 <= metrics['win_rate'] <= 1.0
    
    # Performance targets (may not always hit in synthetic data)
    print(f"\nGMM Backtest Results:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total Trades: {metrics['total_trades']}")
```

### Walk-Forward Validation Tests

```python
# tests/test_validation/test_walk_forward.py

import pytest
from utils.validation import WalkForwardValidator


def test_walk_forward_analysis(sample_data):
    """Test walk-forward validation."""
    from strategies.mean_reversion import MeanReversionStrategy
    
    config = StrategyConfig(name="MR_Test")
    strategy = MeanReversionStrategy(config)
    
    validator = WalkForwardValidator(
        train_days=365,
        test_days=90,
        step_days=30
    )
    
    # Define parameter grid
    param_grid = {
        'washout_period': [3, 5, 7],
        'exit_ma_period': [3, 5, 7]
    }
    
    # Run walk-forward
    results = validator.run_analysis(sample_data, strategy, param_grid)
    
    # Check results
    assert len(results) > 0
    assert 'train_sharpe' in results.columns
    assert 'test_sharpe' in results.columns
    assert 'degradation' in results.columns
    
    # Calculate metrics
    avg_degradation = results['degradation'].mean()
    wf_efficiency = (results['test_sharpe'] > 0.5).sum() / len(results)
    
    print(f"\nWalk-Forward Results:")
    print(f"  Average Degradation: {avg_degradation:.2%}")
    print(f"  WF Efficiency: {wf_efficiency:.2%}")
    
    # Acceptance criteria
    assert avg_degradation < 0.30, "Performance degradation too high (overfitting)"
    assert wf_efficiency > 0.50, "WF efficiency too low (unreliable)"
```

---

## Integration Patterns

### VectorBT Pro Integration

```python
# All strategies must follow this pattern for VectorBT Pro compatibility

import vectorbtpro as vbt

def backtest_with_vectorbt(signals, data, config):
    """
    Standard VectorBT Pro backtest pattern.
    """
    pf = vbt.Portfolio.from_signals(
        # Price data
        close=data['close'],
        open=data['open'],  # Optional, for better execution
        high=data['high'],  # Required for stop losses
        low=data['low'],    # Required for stop losses
        
        # Entry signals
        entries=signals['long_entries'],
        exits=signals['long_exits'],
        short_entries=signals.get('short_entries'),
        short_exits=signals.get('short_exits'),
        
        # Position sizing
        size=signals['position_sizes'],
        size_type='amount',  # 'amount' = shares, 'percent' = % of capital
        
        # Risk management
        sl_stop=signals.get('stop_distance'),  # Stop loss
        tp_stop=signals.get('profit_target'),  # Take profit (optional)
        td_stop=pd.Timedelta(days=signals.get('time_exit_days', 0)),  # Time exit
        
        # Costs
        init_cash=config.initial_capital,
        fees=config.commission_rate,
        slippage=config.slippage,
        
        # Other
        freq='1D',  # Or '5Min' for intraday
        call_seq='auto'  # Order sizing
    )
    
    return pf
```

### Configuration Management

```python
# configs/strategies.yaml

gmm_regime:
  name: "GMM_Regime_Detection"
  risk_per_trade: 0.02
  max_positions: 1
  enable_shorts: false
  commission_rate: 0.0015
  slippage: 0.0015
  gmm_components: 3
  min_training_days: 252
  refit_frequency: 63

mean_reversion:
  name: "Five_Day_Washout"
  risk_per_trade: 0.02
  max_positions: 3
  enable_shorts: false
  washout_period: 5
  ma_filter_period: 200
  max_hold_days: 7
  atr_stop_multiplier: 2.0

orb:
  name: "Opening_Range_Breakout"
  risk_per_trade: 0.02
  max_positions: 20
  enable_shorts: false
  opening_minutes: 5
  atr_stop_multiplier: 2.5
  volume_surge_multiplier: 2.0
```

```python
# core/config.py

import yaml
from pathlib import Path
from typing import Dict


def load_strategy_configs() -> Dict:
    """Load strategy configurations from YAML."""
    config_path = Path('configs/strategies.yaml')
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    return configs


def get_strategy_config(strategy_name: str) -> Dict:
    """Get configuration for specific strategy."""
    configs = load_strategy_configs()
    
    if strategy_name not in configs:
        raise ValueError(f"No configuration for strategy: {strategy_name}")
    
    return configs[strategy_name]
```

---

## Deployment Architecture

### Development Environment

```bash
# Project setup with UV
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
uv pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=strategies --cov=core

# Run backtest
python -m strategies.gmm_regime --symbol SPY --start 2019-01-01 --end 2024-01-01
```

### Paper Trading Deployment

```python
# scripts/run_paper_trading.py

from strategies.gmm_regime import GMMRegimeStrategy, StrategyConfig
from data.alpaca_client import AlpacaDataClient
from core.config import get_strategy_config
import os


def main():
    # Load configuration
    config_dict = get_strategy_config('gmm_regime')
    config = StrategyConfig(**config_dict)
    
    # Initialize strategy
    strategy = GMMRegimeStrategy(config)
    
    # Initialize Alpaca client
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    client = AlpacaDataClient(api_key, secret_key)
    
    # Fetch recent data for signal generation
    data = client.fetch_bars(
        symbol='SPY',
        start_date='2023-01-01',
        end_date='2024-12-31',
        timeframe='1Day'
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Get latest signal
    latest_regime = signals['regime'].iloc[-1]
    latest_entry = signals['long_entries'].iloc[-1]
    
    print(f"Current Regime: {latest_regime}")
    print(f"Entry Signal: {latest_entry}")
    
    # Execute trade if signal (via Alpaca trading API)
    # ... implementation here


if __name__ == '__main__':
    main()
```

### Logging Configuration

```python
# utils/logger.py

from loguru import logger
import sys


def setup_logger(log_dir: str = 'logs'):
    """Configure structured logging."""
    logger.remove()  # Remove default handler
    
    # Console logging (INFO+)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # File logging (DEBUG+)
    logger.add(
        f"{log_dir}/debug.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    )
    
    # Error logging (ERROR+)
    logger.add(
        f"{log_dir}/errors.log",
        rotation="1 week",
        retention="90 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}\n{exception}"
    )
    
    return logger
```

---

## Performance Targets Summary

### Individual Strategy Targets

| Strategy | Win Rate | Sharpe | Max DD | CAGR | Notes |
|----------|----------|--------|--------|------|-------|
| GMM Regime | 55-65% | 0.8-1.0 | -15% | 8-12% | Defensive, low correlation |
| Mean Reversion | 65-75% | 0.6-0.9 | -12% | 5-8% | High win rate, small gains |
| ORB | 15-25% | 1.5-2.5 | -25% | 15-25% | Asymmetric, low win rate |
| Pairs Trading | 70-80% | 1.0-1.4 | -10% | 6-10% | Market-neutral |
| Momentum Portfolio | 50-60% | 1.0-1.5 | -20% | 12-20% | Multi-asset |

### Portfolio Targets (Equal Allocation)

| Metric | Target | Excellent | Critical Threshold |
|--------|--------|-----------|-------------------|
| Sharpe Ratio | > 1.0 | > 1.5 | < 0.5 (fail) |
| Max Drawdown | < 25% | < 20% | > 30% (fail) |
| Win Rate | 45-55% | > 60% | < 40% (concerning) |
| CAGR | 10-15% | > 18% | < 5% (fail) |
| Profit Factor | > 1.5 | > 2.0 | < 1.2 (fail) |

### Walk-Forward Validation Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Performance Degradation | < 20% | Excellent |
| Performance Degradation | 20-30% | Acceptable |
| Performance Degradation | > 30% | Overfitted (fail) |
| WF Efficiency | > 70% | 70%+ windows profitable |
| Parameter Stability | σ < 20% mean | Consistent parameters |

---

## Critical Success Criteria

### Must Pass Before Live Trading

1. **Data Quality**: No NaN, valid OHLC relationships, proper adjustments
2. **Position Sizing**: Capital constraints enforced, no position > 100% capital
3. **Stop Losses**: Always executed, no exceptions
4. **Portfolio Heat**: Never exceed 8% total exposure
5. **Walk-Forward**: Out-of-sample within 30% of in-sample performance
6. **VectorBT Compatibility**: All calculations vectorized, proper index alignment
7. **Unit Tests**: 100% pass rate, >80% code coverage
8. **Paper Trading**: 6+ months, 100+ trades, performance matches backtest

### Red Flags (Abort Signals)

1. Backtest Sharpe > 3.0 (likely overfitted)
2. In-sample vs out-of-sample gap > 30%
3. Paper trading drastically worse than backtest
4. Drawdown exceeds 25%
5. Execution differs significantly from backtest
6. Data quality issues persist
7. Stop losses frequently fail to execute
8. Portfolio heat limits routinely breached

---

## Development Checklist

### Phase 1: Strategy Implementation

- [ ] Implement BaseStrategy interface
- [ ] Implement GMM Regime Detection
- [ ] Implement Mean Reversion
- [ ] Implement Opening Range Breakout
- [ ] Implement Pairs Trading
- [ ] Implement Momentum Portfolio
- [ ] Write unit tests for each strategy
- [ ] Verify VectorBT Pro compatibility

### Phase 2: Core Components

- [ ] Implement PortfolioManager
- [ ] Implement RiskManager
- [ ] Implement position sizing utilities
- [ ] Write integration tests
- [ ] Test multi-strategy coordination

### Phase 3: Data Pipeline

- [ ] Implement AlpacaDataClient
- [ ] Implement MultiTimeframeManager
- [ ] Add data quality validators
- [ ] Test data fetching and alignment

### Phase 4: Backtesting

- [ ] Run individual strategy backtests
- [ ] Run portfolio-level backtest
- [ ] Generate performance reports
- [ ] Document results

### Phase 5: Walk-Forward Validation

- [ ] Implement walk-forward framework
- [ ] Run validation for each strategy
- [ ] Analyze parameter stability
- [ ] Check performance degradation

### Phase 6: Paper Trading

- [ ] Deploy to Alpaca paper account
- [ ] Monitor for 6+ months
- [ ] Compare paper vs backtest
- [ ] Document any discrepancies

### Phase 7: Live Deployment (If All Tests Pass)

- [ ] Start with small capital
- [ ] Monitor execution quality
- [ ] Scale gradually
- [ ] Continuous review and adjustment

---

## Conclusion

This document defines the complete architecture for a production-ready algorithmic trading system. All components are designed to work together following industry best practices:

- **Modular Design**: Each strategy is independent but shares common infrastructure
- **Vectorized Operations**: All calculations use pandas/numpy for performance
- **Risk Management**: Multiple layers of protection (position, portfolio, drawdown)
- **Validation**: Rigorous walk-forward testing prevents overfitting
- **Testing**: Comprehensive unit and integration tests
- **Monitoring**: Structured logging and performance tracking

Development should proceed through all phases systematically, with each phase building on the previous. No shortcuts are permitted on testing, validation, or risk management.

**Remember**: The goal is to build a system that survives long-term, not one that looks good in backtests. Robust validation, conservative position sizing, and disciplined execution are more important than optimization for maximum returns.

---

**Document Status**: Reference Architecture (Complete)  
**Next Steps**: Begin Phase 1 implementation following this specification  
**Questions**: Refer to project knowledge documents or consult development team