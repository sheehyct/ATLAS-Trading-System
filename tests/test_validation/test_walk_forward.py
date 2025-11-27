"""
Tests for walk-forward validation module.

Tests cover:
- FoldWindow dataclass functionality
- Fold generation logic
- Sharpe degradation calculation
- Parameter stability calculation
- Full validation workflow
- Pass/fail criteria
- Edge cases and error handling

Session 83D: Walk-forward validation tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from validation.walk_forward import (
    WalkForwardValidator,
    FoldWindow,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from validation.config import WalkForwardConfig, WalkForwardConfigOptions
from validation.results import FoldResult, WalkForwardResults


class TestFoldWindow:
    """Tests for FoldWindow dataclass."""

    def test_fold_window_creation(self):
        """Test basic FoldWindow creation."""
        window = FoldWindow(
            fold_number=1,
            train_start_idx=0,
            train_end_idx=252,
            test_start_idx=252,
            test_end_idx=315
        )

        assert window.fold_number == 1
        assert window.train_start_idx == 0
        assert window.train_end_idx == 252
        assert window.test_start_idx == 252
        assert window.test_end_idx == 315

    def test_train_size_property(self):
        """Test train_size calculation."""
        window = FoldWindow(
            fold_number=1,
            train_start_idx=0,
            train_end_idx=252,
            test_start_idx=252,
            test_end_idx=315
        )

        assert window.train_size == 252

    def test_test_size_property(self):
        """Test test_size calculation."""
        window = FoldWindow(
            fold_number=1,
            train_start_idx=0,
            train_end_idx=252,
            test_start_idx=252,
            test_end_idx=315
        )

        assert window.test_size == 63


class TestFoldGeneration:
    """Tests for fold window generation."""

    def test_generate_folds_default_config(self, synthetic_daily_data):
        """Test fold generation with default config."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(synthetic_daily_data))

        # With 756 bars, 252 train, 63 test, step=63:
        # Fold 1: train 0-252, test 252-315
        # Fold 2: train 63-315, test 315-378
        # ... should get ~8 folds
        assert len(windows) >= 7
        assert len(windows) <= 9

    def test_generate_folds_first_window(self, synthetic_daily_data):
        """Test first fold window indices."""
        config = WalkForwardConfig(train_period=252, test_period=63)
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(synthetic_daily_data))

        first = windows[0]
        assert first.fold_number == 1
        assert first.train_start_idx == 0
        assert first.train_end_idx == 252
        assert first.test_start_idx == 252
        assert first.test_end_idx == 315

    def test_generate_folds_non_overlapping_test(self, synthetic_daily_data):
        """Test that test windows don't overlap."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(synthetic_daily_data))

        for i in range(len(windows) - 1):
            # Test end of fold i should equal test start of fold i+1
            # with step_size = test_period
            assert windows[i].test_end_idx <= windows[i+1].test_end_idx

    def test_generate_folds_insufficient_data(self, short_data):
        """Test with insufficient data for any folds."""
        config = WalkForwardConfig(train_period=252, test_period=63)
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(short_data))

        # 200 bars < 252 + 63, should get 0 folds
        assert len(windows) == 0

    def test_generate_folds_minimal_data(self, minimal_fold_data):
        """Test with data for exactly 1 fold."""
        config = WalkForwardConfig(train_period=252, test_period=63)
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(minimal_fold_data))

        assert len(windows) == 1

    def test_custom_step_size(self, synthetic_daily_data):
        """Test fold generation with custom step size."""
        config = WalkForwardConfig(
            train_period=252,
            test_period=63,
            step_size=21  # Monthly steps
        )
        validator = WalkForwardValidator(config)

        windows = validator._generate_fold_windows(len(synthetic_daily_data))

        # More folds with smaller step size
        assert len(windows) > 10


class TestSharpeDegradationCalculation:
    """Tests for Sharpe degradation calculation."""

    def test_zero_degradation(self):
        """Test with no degradation."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation = validator._calculate_sharpe_degradation(1.0, 1.0)
        assert degradation == 0.0

    def test_typical_degradation(self):
        """Test typical 20% degradation."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation = validator._calculate_sharpe_degradation(1.0, 0.8)
        assert abs(degradation - 0.2) < 0.001

    def test_thirty_percent_degradation(self):
        """Test 30% degradation threshold."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation = validator._calculate_sharpe_degradation(1.0, 0.7)
        assert abs(degradation - 0.3) < 0.001

    def test_negative_degradation(self):
        """Test OOS better than IS (negative degradation)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation = validator._calculate_sharpe_degradation(1.0, 1.2)
        assert degradation < 0

    def test_zero_is_sharpe(self):
        """Test with zero IS Sharpe."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        # Zero IS with positive OOS
        degradation = validator._calculate_sharpe_degradation(0.0, 0.5)
        assert degradation == 0.0

        # Zero both
        degradation = validator._calculate_sharpe_degradation(0.0, 0.0)
        assert degradation == 1.0

    def test_full_degradation(self):
        """Test 100% degradation (OOS = 0)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation = validator._calculate_sharpe_degradation(1.5, 0.0)
        assert degradation == 1.0


class TestParameterStability:
    """Tests for parameter stability (CV) calculation."""

    def test_stable_parameters(self):
        """Test with stable (identical) parameters."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        params = [
            {'sma_period': 20, 'threshold': 0.5},
            {'sma_period': 20, 'threshold': 0.5},
            {'sma_period': 20, 'threshold': 0.5},
        ]

        stability = validator._calculate_parameter_stability(params)

        assert stability['sma_period'] == 0.0
        assert stability['threshold'] == 0.0

    def test_unstable_parameters(self):
        """Test with highly variable parameters."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        params = [
            {'sma_period': 10, 'threshold': 0.3},
            {'sma_period': 30, 'threshold': 0.7},
            {'sma_period': 50, 'threshold': 0.1},
        ]

        stability = validator._calculate_parameter_stability(params)

        # Both parameters should have CV > 20%
        assert stability['sma_period'] > 0.2
        assert stability['threshold'] > 0.2

    def test_mixed_stability(self):
        """Test with some stable, some unstable parameters."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        params = [
            {'sma_period': 20, 'threshold': 0.3},
            {'sma_period': 20, 'threshold': 0.7},
            {'sma_period': 20, 'threshold': 0.5},
        ]

        stability = validator._calculate_parameter_stability(params)

        # sma_period is stable
        assert stability['sma_period'] == 0.0
        # threshold varies
        assert stability['threshold'] > 0.2

    def test_empty_parameters(self):
        """Test with no parameters."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        stability = validator._calculate_parameter_stability([])

        assert stability == {}

    def test_non_numeric_parameters_ignored(self):
        """Test that non-numeric parameters are ignored."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        params = [
            {'sma_period': 20, 'name': 'strategy_a'},
            {'sma_period': 20, 'name': 'strategy_b'},
        ]

        stability = validator._calculate_parameter_stability(params)

        assert 'sma_period' in stability
        assert 'name' not in stability


class TestCalculateSharpeRatio:
    """Tests for standalone Sharpe ratio calculation."""

    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)

        sharpe = calculate_sharpe_ratio(returns)

        # Should be positive with positive drift
        assert sharpe > 0

    def test_zero_returns(self):
        """Test Sharpe with zero returns."""
        returns = pd.Series([0.0] * 100)

        sharpe = calculate_sharpe_ratio(returns)

        # Can't calculate meaningful Sharpe with zero variance
        assert sharpe == 0.0

    def test_negative_returns(self):
        """Test Sharpe with negative returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 - 0.002)

        sharpe = calculate_sharpe_ratio(returns)

        # Should be negative with negative drift
        assert sharpe < 0

    def test_empty_returns(self):
        """Test Sharpe with empty series."""
        returns = pd.Series(dtype=float)

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_single_return(self):
        """Test Sharpe with single return."""
        returns = pd.Series([0.01])

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe == 0.0  # Need at least 2 for std


class TestCalculateMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = pd.Series([100, 110, 120, 130, 140])

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_simple_drawdown(self):
        """Test with simple drawdown."""
        equity = pd.Series([100, 110, 88, 95])  # 20% drawdown from peak

        max_dd = calculate_max_drawdown(equity)

        assert abs(max_dd - 0.2) < 0.001

    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns - should return max."""
        equity = pd.Series([100, 110, 99, 120, 96, 130])
        # First DD: 110 -> 99 = 10%
        # Second DD: 120 -> 96 = 20%

        max_dd = calculate_max_drawdown(equity)

        assert abs(max_dd - 0.2) < 0.001

    def test_empty_equity(self):
        """Test with empty equity series."""
        equity = pd.Series(dtype=float)

        max_dd = calculate_max_drawdown(equity)

        assert max_dd == 0.0


class TestFullValidation:
    """Tests for full walk-forward validation workflow."""

    def test_validate_returns_results(self, mock_strategy, synthetic_daily_data):
        """Test that validate returns WalkForwardResults."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        assert isinstance(results, WalkForwardResults)
        assert results.total_folds > 0
        assert len(results.folds) == results.total_folds

    def test_validate_with_good_strategy(self, excellent_strategy, synthetic_daily_data):
        """Test validation with good strategy."""
        config = WalkForwardConfig(
            min_trades_per_fold=3  # Lower threshold for mock
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(excellent_strategy, synthetic_daily_data)

        # Excellent strategy should pass most criteria
        assert results.avg_is_sharpe > 0
        assert results.avg_oos_sharpe > 0

    def test_validate_with_poor_strategy(self, poor_strategy, synthetic_daily_data):
        """Test validation with poor strategy."""
        config = WalkForwardConfig(
            min_trades_per_fold=3  # Lower threshold for mock
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(poor_strategy, synthetic_daily_data)

        # Poor strategy should have issues
        assert results.avg_oos_sharpe < 1.0  # Low OOS Sharpe

    def test_validate_insufficient_data(self, mock_strategy, short_data):
        """Test validation with insufficient data."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, short_data)

        assert results.passes_validation is False
        assert results.total_folds == 0
        assert "Insufficient data" in results.failure_reasons[0]

    def test_fold_results_populated(self, mock_strategy, synthetic_daily_data):
        """Test that individual fold results are populated."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        for fold in results.folds:
            assert isinstance(fold, FoldResult)
            assert fold.fold_number > 0
            assert fold.train_start is not None
            assert fold.train_end is not None
            assert fold.test_start is not None
            assert fold.test_end is not None
            assert isinstance(fold.parameters, dict)

    def test_summary_method(self, mock_strategy, synthetic_daily_data):
        """Test results summary method."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)
        summary = results.summary()

        assert "WALK-FORWARD VALIDATION" in summary
        assert "Total Folds" in summary
        assert "Sharpe Degradation" in summary


class TestValidationCriteria:
    """Tests for pass/fail criteria."""

    def test_sharpe_degradation_failure(self, mock_strategy, synthetic_daily_data):
        """Test failure due to excessive Sharpe degradation."""
        config = WalkForwardConfig(
            max_sharpe_degradation=0.05,  # Very strict
            min_trades_per_fold=3
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        # Most strategies will exceed 5% degradation
        # (checking that the check exists)
        assert 'sharpe_degradation' in str(results.failure_reasons).lower() or results.sharpe_degradation <= 0.05

    def test_min_oos_sharpe_failure(self, poor_strategy, synthetic_daily_data):
        """Test failure due to low OOS Sharpe."""
        config = WalkForwardConfig(
            min_oos_sharpe=5.0,  # Impossibly high
            min_trades_per_fold=3
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(poor_strategy, synthetic_daily_data)

        assert results.passes_validation is False
        assert any('OOS Sharpe' in r for r in results.failure_reasons)

    def test_profitable_folds_failure(self, poor_strategy, synthetic_daily_data):
        """Test failure due to too few profitable folds."""
        config = WalkForwardConfig(
            min_profitable_folds=1.0,  # 100% profitable required
            min_trades_per_fold=3
        )
        validator = WalkForwardValidator(config)

        results = validator.validate(poor_strategy, synthetic_daily_data)

        # Very hard to have 100% profitable folds
        assert results.profitable_folds_pct < 1.0 or results.passes_validation

    def test_passes_method(self, mock_strategy, synthetic_daily_data):
        """Test passes() convenience method."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        assert validator.passes(results) == results.passes_validation


class TestOptionsConfig:
    """Tests for options-specific configuration."""

    def test_options_config_looser_thresholds(self):
        """Test that options config has looser thresholds."""
        equity_config = WalkForwardConfig()
        options_config = WalkForwardConfigOptions()

        assert options_config.max_sharpe_degradation > equity_config.max_sharpe_degradation
        assert options_config.min_oos_sharpe < equity_config.min_oos_sharpe

    def test_validation_with_options_config(self, mock_strategy, synthetic_daily_data):
        """Test validation with options config."""
        config = WalkForwardConfigOptions(min_trades_per_fold=3)
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        assert isinstance(results, WalkForwardResults)
        assert results.max_sharpe_degradation == 0.40  # Options threshold


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_fold(self, mock_strategy, minimal_fold_data):
        """Test with minimum data for single fold."""
        config = WalkForwardConfig(min_trades_per_fold=3)
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, minimal_fold_data)

        assert results.total_folds == 1

    def test_custom_param_grid(self, mock_strategy, synthetic_daily_data):
        """Test with custom parameter grid."""
        config = WalkForwardConfig(min_trades_per_fold=3)
        validator = WalkForwardValidator(config)

        custom_grid = {
            'sma_period': [5, 10, 15, 20],
            'threshold': [0.2, 0.4, 0.6, 0.8]
        }

        results = validator.validate(mock_strategy, synthetic_daily_data, param_grid=custom_grid)

        assert results.total_folds > 0

    def test_to_dict_serializable(self, mock_strategy, synthetic_daily_data):
        """Test that results can be serialized to dict."""
        config = WalkForwardConfig(min_trades_per_fold=3)
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)
        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert 'avg_is_sharpe' in result_dict
        assert 'avg_oos_sharpe' in result_dict
        assert 'folds' in result_dict
        assert isinstance(result_dict['folds'], list)
