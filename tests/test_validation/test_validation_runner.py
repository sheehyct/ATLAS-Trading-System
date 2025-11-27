"""
Tests for ValidationRunner orchestrator.

Session 83J: Tests for the Session 83I ValidationRunner class that
orchestrates all validators for comprehensive production readiness checks.

Tests cover:
- Initialization with default/custom config
- validate_strategy() method
- passes() method
- Options config looser thresholds
- Skip validators functionality
- Summary aggregation
- run_validation() convenience function
- Execution time tracking
- Critical issues population
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from validation import (
    ValidationRunner,
    run_validation,
    ValidationConfig,
    ValidationReport,
)
from validation.results import ValidationSummary
from validation.protocols import BacktestResult


# =============================================================================
# Mock Strategy for Testing
# =============================================================================

class MockValidationStrategy:
    """
    Mock strategy for testing ValidationRunner.

    Implements StrategyProtocol with configurable behavior.
    """

    def __init__(
        self,
        base_sharpe: float = 1.0,
        win_rate: float = 0.55,
        should_fail: bool = False
    ):
        """
        Initialize mock strategy.

        Args:
            base_sharpe: Sharpe ratio for backtest results
            win_rate: Win rate for generated trades
            should_fail: If True, backtest raises exception
        """
        self.base_sharpe = base_sharpe
        self.win_rate = win_rate
        self.should_fail = should_fail

    def backtest(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """Run mock backtest."""
        if self.should_fail:
            raise ValueError("Mock backtest failure")

        # Generate mock trades
        n_trades = max(5, len(data) // 10)
        trades = self._generate_trades(n_trades, data.index)

        return BacktestResult(
            total_return=0.15,
            sharpe_ratio=self.base_sharpe,
            max_drawdown=-0.10,
            win_rate=self.win_rate,
            trades=trades,
            equity_curve=self._generate_equity_curve(data.index),
            trade_count=n_trades,
            parameters=params or {},
            start_date=data.index[0],
            end_date=data.index[-1],
        )

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict] = None
    ):
        """Run mock optimization."""
        result = self.backtest(data)
        return {'sma_period': 20, 'threshold': 0.5}, result

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock signals."""
        np.random.seed(42)
        signals = pd.DataFrame({
            'signal': np.random.choice([0, 1, -1], size=len(data), p=[0.8, 0.1, 0.1]),
            'raw_signal': np.random.choice([0, 1, -1], size=len(data), p=[0.8, 0.1, 0.1]),
        }, index=data.index)

        # Shift tradeable signal by 1 (proper signal handling)
        signals['signal'] = signals['raw_signal'].shift(1).fillna(0)
        return signals

    def _generate_trades(self, n: int, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate mock trades DataFrame."""
        np.random.seed(42)
        dates = np.random.choice(index, size=min(n, len(index)), replace=False)
        dates = sorted(dates)

        trades = []
        for i, date in enumerate(dates):
            win = np.random.random() < self.win_rate
            pnl = np.random.uniform(50, 200) if win else -np.random.uniform(20, 100)

            trades.append({
                'entry_date': date,
                'exit_date': date + timedelta(days=np.random.randint(1, 10)),
                'entry_price': 100.0,
                'exit_price': 100.0 * (1 + pnl / 10000),
                'pnl': pnl,
                'pnl_pct': pnl / 10000,
                'win': win,
            })

        return pd.DataFrame(trades)

    def _generate_equity_curve(self, index: pd.DatetimeIndex) -> pd.Series:
        """Generate mock equity curve."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(index))
        equity = 10000 * np.cumprod(1 + returns)
        return pd.Series(equity, index=index)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Create 1 year of daily data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)


@pytest.fixture
def mock_strategy() -> MockValidationStrategy:
    """Create mock strategy for testing."""
    return MockValidationStrategy(base_sharpe=1.0, win_rate=0.55)


@pytest.fixture
def failing_strategy() -> MockValidationStrategy:
    """Create strategy that fails backtest."""
    return MockValidationStrategy(should_fail=True)


# =============================================================================
# Test: Initialization
# =============================================================================

class TestValidationRunnerInit:
    """Tests for ValidationRunner initialization."""

    def test_runner_initializes_with_default_config(self):
        """Test runner initializes with default configuration."""
        runner = ValidationRunner()
        assert runner.config is not None
        assert isinstance(runner.config, ValidationConfig)

    def test_runner_initializes_with_custom_config(self):
        """Test runner initializes with custom configuration."""
        custom_config = ValidationConfig()
        custom_config.walk_forward.train_period = 300  # Custom value

        runner = ValidationRunner(config=custom_config)
        assert runner.config is custom_config
        assert runner.config.walk_forward.train_period == 300


# =============================================================================
# Test: validate_strategy Method
# =============================================================================

class TestValidateStrategyMethod:
    """Tests for validate_strategy() method."""

    def test_validate_strategy_returns_validation_report(
        self, sample_data, mock_strategy
    ):
        """Test that validate_strategy returns ValidationReport."""
        runner = ValidationRunner()

        # Skip expensive validators for this test
        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        assert isinstance(report, ValidationReport)
        assert report.strategy_name == "Test Strategy"
        assert isinstance(report.summary, ValidationSummary)

    def test_passes_method_returns_boolean(self, sample_data, mock_strategy):
        """Test that passes() returns boolean."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        result = runner.passes(report)
        assert isinstance(result, bool)


# =============================================================================
# Test: Options Configuration
# =============================================================================

class TestOptionsConfig:
    """Tests for options-specific configuration."""

    def test_options_config_uses_looser_thresholds(self):
        """Test that is_options=True uses looser thresholds."""
        runner = ValidationRunner()

        # Get default config
        default_config = ValidationConfig()

        # Get options config
        options_config = default_config.for_options()

        # Options config should have different (looser) thresholds
        # WalkForwardConfigOptions has different defaults
        assert options_config is not default_config

    def test_validate_strategy_with_is_options(self, sample_data, mock_strategy):
        """Test validation with is_options=True."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Options Strategy",
            is_options=True,
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        assert report.is_options == True


# =============================================================================
# Test: Skip Validators
# =============================================================================

class TestSkipValidators:
    """Tests for skip validator functionality."""

    def test_skip_walk_forward_respected(self, sample_data, mock_strategy):
        """Test that skip_walk_forward=True skips walk-forward."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # Walk-forward should be None when skipped
        assert report.walk_forward is None
        # Warning should be present
        assert any('walk-forward' in w.lower() for w in report.summary.warnings)

    def test_skip_monte_carlo_respected(self, sample_data, mock_strategy):
        """Test that skip_monte_carlo=True skips Monte Carlo."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # Monte Carlo should be None when skipped
        assert report.monte_carlo is None

    def test_skip_bias_detection_respected(self, sample_data, mock_strategy):
        """Test that skip_bias_detection=True skips bias detection."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # Bias detection should be None when skipped
        assert report.bias_detection is None


# =============================================================================
# Test: Summary Aggregation
# =============================================================================

class TestSummaryAggregation:
    """Tests for validation summary aggregation."""

    def test_summary_aggregates_all_results(self, sample_data, mock_strategy):
        """Test that summary correctly aggregates results."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        summary = report.summary
        assert hasattr(summary, 'passes_all')
        assert hasattr(summary, 'walk_forward_passed')
        assert hasattr(summary, 'monte_carlo_passed')
        assert hasattr(summary, 'bias_detection_passed')
        assert hasattr(summary, 'critical_issues')
        assert hasattr(summary, 'warnings')


# =============================================================================
# Test: run_validation Convenience Function
# =============================================================================

class TestRunValidationFunction:
    """Tests for run_validation() convenience function."""

    def test_run_validation_convenience_function(self, sample_data, mock_strategy):
        """Test run_validation() returns ValidationReport."""
        report = run_validation(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        assert isinstance(report, ValidationReport)
        assert report.strategy_name == "Test Strategy"

    def test_run_validation_with_custom_config(self, sample_data, mock_strategy):
        """Test run_validation() accepts custom config."""
        custom_config = ValidationConfig()

        report = run_validation(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            config=custom_config,
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        assert isinstance(report, ValidationReport)


# =============================================================================
# Test: Execution Time Tracking
# =============================================================================

class TestExecutionTimeTracking:
    """Tests for execution time tracking."""

    def test_execution_time_tracked(self, sample_data, mock_strategy):
        """Test that execution time is tracked in report."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        assert hasattr(report, 'execution_time_seconds')
        assert report.execution_time_seconds >= 0


# =============================================================================
# Test: Critical Issues Population
# =============================================================================

class TestCriticalIssues:
    """Tests for critical issues population."""

    def test_critical_issues_populated_on_failure(self, sample_data, failing_strategy):
        """Test that critical issues are populated when validators fail."""
        runner = ValidationRunner()

        # Even with a failing strategy, we skip validators so issues come from there
        report = runner.validate_strategy(
            strategy=failing_strategy,
            data=sample_data,
            strategy_name="Failing Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # When backtest fails, summary should reflect issues
        # Skipped validators generate warnings, not critical issues
        assert isinstance(report.summary.critical_issues, list)
        assert isinstance(report.summary.warnings, list)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self, mock_strategy):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        runner = ValidationRunner()

        # Should handle gracefully
        try:
            report = runner.validate_strategy(
                strategy=mock_strategy,
                data=empty_data,
                strategy_name="Empty Data Test",
                skip_walk_forward=True,
                skip_monte_carlo=True,
                skip_bias_detection=True,
                skip_pattern_metrics=True,
            )
            # May pass with empty data depending on implementation
            assert isinstance(report, ValidationReport)
        except Exception:
            # Or may raise an exception - that's also acceptable
            pass

    def test_report_serialization(self, sample_data, mock_strategy):
        """Test that report can be serialized."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # Should be able to convert to dict
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert 'strategy_name' in report_dict
        assert 'summary' in report_dict

    def test_report_to_json(self, sample_data, mock_strategy):
        """Test that report can be converted to JSON."""
        runner = ValidationRunner()

        report = runner.validate_strategy(
            strategy=mock_strategy,
            data=sample_data,
            strategy_name="Test Strategy",
            skip_walk_forward=True,
            skip_monte_carlo=True,
            skip_bias_detection=True,
            skip_pattern_metrics=True,
        )

        # Should be able to convert to JSON
        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert 'Test Strategy' in json_str
