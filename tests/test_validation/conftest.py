"""
Fixtures for validation module tests.

Provides mock strategies and synthetic data for testing validators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from validation.protocols import BacktestResult


class MockStrategy:
    """
    Mock strategy for testing validators.

    Implements StrategyProtocol with configurable behavior.
    """

    def __init__(
        self,
        base_sharpe: float = 1.0,
        sharpe_degradation: float = 0.2,
        win_rate: float = 0.55,
        avg_trades_per_fold: int = 20,
        default_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize mock strategy.

        Args:
            base_sharpe: Sharpe ratio to return for in-sample tests
            sharpe_degradation: How much to degrade OOS Sharpe (0.2 = 20% worse)
            win_rate: Win rate for generated trades
            avg_trades_per_fold: Average number of trades to generate per fold
            default_params: Default parameters for optimization
        """
        self.base_sharpe = base_sharpe
        self.sharpe_degradation = sharpe_degradation
        self.win_rate = win_rate
        self.avg_trades_per_fold = avg_trades_per_fold
        self.default_params = default_params or {'sma_period': 20, 'threshold': 0.5}
        self._call_count = 0

    def backtest(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """Run mock backtest."""
        self._call_count += 1
        params = params or self.default_params

        # Generate trades
        n_trades = max(5, int(len(data) / 10))
        trades = self._generate_trades(n_trades, data.index)

        # Calculate metrics
        returns = trades['pnl_pct']
        sharpe = self._calculate_sharpe(returns)

        # Apply degradation if this looks like OOS (subsequent calls with same params)
        if self._call_count > 1:
            sharpe *= (1 - self.sharpe_degradation)

        total_return = trades['pnl_pct'].sum()
        win_rate = (trades['pnl'] > 0).mean()
        equity_curve = self._generate_equity_curve(data.index, total_return)
        max_dd = self._calculate_max_dd(equity_curve)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            trades=trades,
            equity_curve=equity_curve,
            trade_count=len(trades),
            parameters=params,
            start_date=data.index[0].to_pydatetime() if hasattr(data.index[0], 'to_pydatetime') else data.index[0],
            end_date=data.index[-1].to_pydatetime() if hasattr(data.index[-1], 'to_pydatetime') else data.index[-1],
        )

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """Run mock optimization."""
        # Reset call count for new optimization
        self._call_count = 0

        param_grid = param_grid or {
            'sma_period': [10, 20, 30],
            'threshold': [0.3, 0.5, 0.7]
        }

        # Simulate finding best params
        best_params = {k: v[len(v)//2] for k, v in param_grid.items()}

        # Add some noise to make params vary between folds
        np.random.seed(len(data))  # Seed based on data length for reproducibility
        best_params['sma_period'] = int(best_params.get('sma_period', 20) + np.random.randint(-5, 6))
        best_params['threshold'] = round(best_params.get('threshold', 0.5) + np.random.uniform(-0.1, 0.1), 2)

        result = self.backtest(data, best_params)
        return best_params, result

    def generate_signals(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Generate mock signals."""
        np.random.seed(42)
        n = len(data)
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = np.random.random(n) < 0.05
        signals['exit'] = np.random.random(n) < 0.05
        signals['direction'] = 1
        return signals

    def _generate_trades(self, n_trades: int, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate mock trades."""
        np.random.seed(42)

        # Generate random dates within the index
        trade_indices = sorted(np.random.choice(len(index) - 10, size=n_trades, replace=False))

        trades = []
        for i, idx in enumerate(trade_indices):
            is_winner = np.random.random() < self.win_rate
            pnl = np.random.uniform(50, 200) if is_winner else -np.random.uniform(30, 100)
            pnl_pct = pnl / 10000  # Assuming $10k account

            trades.append({
                'trade_id': i,
                'entry_date': index[idx],
                'exit_date': index[min(idx + np.random.randint(1, 10), len(index) - 1)],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'days_held': np.random.randint(1, 10),
                'direction': 1,
            })

        return pd.DataFrame(trades)

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret == 0:
            return 0.0
        return (mean_ret / std_ret) * np.sqrt(252) * self.base_sharpe

    def _generate_equity_curve(self, index: pd.DatetimeIndex, total_return: float) -> pd.Series:
        """Generate mock equity curve."""
        n = len(index)
        # Create smooth equity curve ending at total_return
        t = np.linspace(0, 1, n)
        noise = np.random.randn(n).cumsum() * 0.01
        equity = 10000 * (1 + total_return * t + noise - noise[0])
        return pd.Series(equity, index=index)

    def _calculate_max_dd(self, equity: pd.Series) -> float:
        """Calculate max drawdown from equity curve."""
        if len(equity) < 2:
            return 0.0
        running_max = equity.expanding().max()
        drawdown = (running_max - equity) / running_max
        return drawdown.max()


class PoorStrategy(MockStrategy):
    """Strategy that fails validation criteria."""

    def __init__(self):
        super().__init__(
            base_sharpe=0.3,  # Below minimum
            sharpe_degradation=0.5,  # 50% degradation (above 30% max)
            win_rate=0.40,  # Below 50%
            avg_trades_per_fold=3,  # Few trades
        )


class ExcellentStrategy(MockStrategy):
    """Strategy that easily passes validation."""

    def __init__(self):
        super().__init__(
            base_sharpe=2.0,  # Strong Sharpe
            sharpe_degradation=0.1,  # Only 10% degradation
            win_rate=0.65,  # Good win rate
            avg_trades_per_fold=30,  # Many trades
        )


@pytest.fixture
def mock_strategy():
    """Fixture providing default mock strategy."""
    return MockStrategy()


@pytest.fixture
def poor_strategy():
    """Fixture providing strategy that fails validation."""
    return PoorStrategy()


@pytest.fixture
def excellent_strategy():
    """Fixture providing strategy that passes validation easily."""
    return ExcellentStrategy()


@pytest.fixture
def synthetic_daily_data():
    """
    Generate synthetic daily OHLCV data for testing.

    Returns 3 years (756 trading days) of data - enough for ~8 folds.
    """
    np.random.seed(42)

    n_days = 756  # 3 years
    start_date = datetime(2021, 1, 1)

    # Generate dates (weekdays only)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Generate prices with trend and noise
    returns = np.random.randn(n_days) * 0.02 + 0.0003  # Slight upward drift
    prices = 100 * (1 + returns).cumprod()

    # Generate OHLCV
    data = pd.DataFrame(index=dates)
    data['Open'] = prices * (1 + np.random.randn(n_days) * 0.005)
    data['High'] = prices * (1 + np.abs(np.random.randn(n_days) * 0.01))
    data['Low'] = prices * (1 - np.abs(np.random.randn(n_days) * 0.01))
    data['Close'] = prices
    data['Volume'] = np.random.randint(1000000, 10000000, n_days)

    return data


@pytest.fixture
def short_data():
    """
    Generate short dataset (insufficient for walk-forward).

    Returns only 200 days - not enough for default config.
    """
    np.random.seed(42)

    n_days = 200  # Less than train_period (252)
    start_date = datetime(2024, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    returns = np.random.randn(n_days) * 0.02
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame(index=dates)
    data['Open'] = prices
    data['High'] = prices * 1.01
    data['Low'] = prices * 0.99
    data['Close'] = prices
    data['Volume'] = np.random.randint(1000000, 10000000, n_days)

    return data


@pytest.fixture
def minimal_fold_data():
    """
    Generate data for exactly 1 fold.

    Returns 315 days (252 train + 63 test).
    """
    np.random.seed(42)

    n_days = 315
    start_date = datetime(2024, 1, 1)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    returns = np.random.randn(n_days) * 0.02
    prices = 100 * (1 + returns).cumprod()

    data = pd.DataFrame(index=dates)
    data['Open'] = prices
    data['High'] = prices * 1.01
    data['Low'] = prices * 0.99
    data['Close'] = prices
    data['Volume'] = np.random.randint(1000000, 10000000, n_days)

    return data
