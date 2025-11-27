"""
ATLAS Production Readiness Validation - Protocols Module

Defines the core protocols and base dataclasses for the validation framework.
Uses Python Protocol for structural typing - strategies don't need to inherit.

Session 83C: Foundation for ATLAS compliance validation per
ATLAS_PRODUCTION_READINESS_CHECKLIST.md Sections 1.6, 1.7.

Usage:
    from validation.protocols import StrategyProtocol, BacktestResult

    class MyStrategy:  # No inheritance needed
        def backtest(self, data, params=None):
            ...
        def optimize(self, data, param_grid=None):
            ...
        def generate_signals(self, data, params=None):
            ...
"""

from dataclasses import dataclass, field
from typing import Protocol, Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class BacktestResult:
    """
    Unified format for strategy backtest output.

    Required for all validators - strategies must return this format
    or a compatible structure.

    Attributes:
        total_return: Total percentage return over backtest period
        sharpe_ratio: Risk-adjusted return (annualized)
        max_drawdown: Maximum peak-to-trough decline (as positive number)
        win_rate: Percentage of winning trades (0-1)
        trades: DataFrame with individual trade records
        equity_curve: Time series of portfolio value
        trade_count: Number of trades executed
        parameters: Strategy parameters used for this backtest
        start_date: Backtest start date
        end_date: Backtest end date

    Required trades DataFrame columns:
        - pnl: Trade profit/loss in dollars
        - pnl_pct: Trade profit/loss as percentage
        - entry_date: Trade entry timestamp
        - exit_date: Trade exit timestamp
        - days_held: Number of days position was held

    Optional trades DataFrame columns (for pattern metrics):
        - pattern_type: STRAT pattern type (e.g., '2-1-2U', '3-1-2D')
        - exit_type: How trade was closed ('TARGET', 'STOP', 'EXPIRATION')
        - symbol: Underlying symbol
        - direction: Trade direction (1=long, -1=short)
    """
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    trade_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def __post_init__(self):
        """Calculate trade_count if not set."""
        if self.trade_count == 0 and self.trades is not None:
            self.trade_count = len(self.trades)

    @property
    def annual_return(self) -> float:
        """Annualized return based on equity curve duration."""
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0

        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            return 0.0

        years = days / 365.0
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses."""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0.0

        profits = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        losses = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())

        if losses == 0:
            return float('inf') if profits > 0 else 0.0

        return profits / losses

    @property
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0.0

        if len(self.trades) == 0:
            return 0.0

        return self.trades['pnl'].mean()

    @property
    def avg_winner(self) -> float:
        """Average winning trade P/L."""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0.0

        winners = self.trades[self.trades['pnl'] > 0]['pnl']
        return winners.mean() if len(winners) > 0 else 0.0

    @property
    def avg_loser(self) -> float:
        """Average losing trade P/L (negative number)."""
        if self.trades is None or 'pnl' not in self.trades.columns:
            return 0.0

        losers = self.trades[self.trades['pnl'] < 0]['pnl']
        return losers.mean() if len(losers) > 0 else 0.0

    @property
    def avg_days_held(self) -> float:
        """Average holding period in days."""
        if self.trades is None or 'days_held' not in self.trades.columns:
            return 0.0

        return self.trades['days_held'].mean()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'trade_count': self.trade_count,
            'annual_return': self.annual_return,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_winner': self.avg_winner,
            'avg_loser': self.avg_loser,
            'avg_days_held': self.avg_days_held,
            'parameters': self.parameters,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            f"Total Return:    {self.total_return:.2%}",
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}",
            f"Max Drawdown:    {self.max_drawdown:.2%}",
            f"Win Rate:        {self.win_rate:.1%}",
            f"Trade Count:     {self.trade_count}",
            f"Profit Factor:   {self.profit_factor:.2f}",
            f"Expectancy:      ${self.expectancy:.2f}",
            f"Avg Days Held:   {self.avg_days_held:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class StrategyProtocol(Protocol):
    """
    Protocol defining the interface for validatable strategies.

    Uses structural typing - strategies don't need to inherit from this class,
    they just need to implement the required methods with compatible signatures.

    Required methods:
        backtest: Run strategy on historical data
        optimize: Find optimal parameters on training data
        generate_signals: Generate entry/exit signals
    """

    def backtest(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            params: Optional strategy parameters (uses defaults if None)

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        ...

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters on training data.

        Args:
            data: OHLCV DataFrame for parameter optimization
            param_grid: Dict of parameter names to lists of values to test
                       If None, uses strategy's default optimization grid

        Returns:
            Tuple of (best_params, best_backtest_result)
        """
        ...

    def generate_signals(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals without backtesting.

        Args:
            data: OHLCV DataFrame
            params: Optional strategy parameters

        Returns:
            DataFrame with at least 'entry' and 'exit' boolean columns
            May include 'direction' (1/-1), 'stop', 'target' columns
        """
        ...


class ValidatorProtocol(Protocol):
    """
    Protocol for validation components.

    All validators (walk-forward, Monte Carlo, etc.) should implement
    this interface for consistent orchestration.
    """

    def validate(self, *args, **kwargs) -> Any:
        """
        Run validation and return results.

        Specific arguments depend on validator type.
        Returns a results dataclass specific to the validator.
        """
        ...

    def passes(self, results: Any) -> bool:
        """
        Check if results meet acceptance criteria.

        Args:
            results: Results from validate() method

        Returns:
            True if validation passes, False otherwise
        """
        ...


# Type aliases for clarity
ParameterDict = Dict[str, Any]
ParameterGrid = Dict[str, List[Any]]
TradesDataFrame = pd.DataFrame
EquityCurve = pd.Series
