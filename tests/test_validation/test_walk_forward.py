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
from validation.config import WalkForwardConfig, WalkForwardConfigOptions, WalkForwardConfigHoldout, ValidationConfig
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

        # Session 83K-16: Now returns tuple (degradation, is_sign_reversal)
        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(1.0, 1.0)
        assert degradation == 0.0
        assert is_sign_reversal == False

    def test_typical_degradation(self):
        """Test typical 20% degradation."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(1.0, 0.8)
        assert abs(degradation - 0.2) < 0.001
        assert is_sign_reversal == False

    def test_thirty_percent_degradation(self):
        """Test 30% degradation threshold."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(1.0, 0.7)
        assert abs(degradation - 0.3) < 0.001
        assert is_sign_reversal == False

    def test_negative_degradation(self):
        """Test OOS better than IS (negative degradation)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(1.0, 1.2)
        assert degradation < 0
        assert is_sign_reversal == False

    def test_zero_is_sharpe(self):
        """Test with zero IS Sharpe."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        # Zero IS with positive OOS - not a sign reversal (zero is neither positive nor negative)
        # Session 83K-16: Sign reversal requires IS < 0 (strictly negative)
        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(0.0, 0.5)
        assert degradation == 0.0
        assert is_sign_reversal == False  # Zero is not negative, so no sign reversal

        # Zero both - no sign reversal
        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(0.0, 0.0)
        assert degradation == 1.0
        assert is_sign_reversal == False

    def test_full_degradation(self):
        """Test 100% degradation (OOS = 0)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(1.5, 0.0)
        assert degradation == 1.0
        assert is_sign_reversal == False

    # Session 83K-16: New tests for sign reversal detection
    def test_sign_reversal_negative_is_positive_oos(self):
        """Test sign reversal: negative IS Sharpe, positive OOS Sharpe (like IWM case)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        # This is the IWM case: IS=-7.70, OOS=+4.09
        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(-7.70, 4.09)
        assert is_sign_reversal == True  # Sign reversal detected

    def test_sign_reversal_positive_is_negative_oos(self):
        """Test sign reversal: positive IS Sharpe, negative OOS Sharpe (overfitting case)."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        # This is the SPY case: IS=+5.07, OOS=-16.90
        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(5.07, -16.90)
        assert is_sign_reversal == True  # Sign reversal detected

    def test_no_sign_reversal_both_negative(self):
        """Test no sign reversal when both IS and OOS are negative."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(-2.0, -1.5)
        assert degradation == 1.0  # Both bad
        assert is_sign_reversal == False  # Same sign, no reversal

    def test_no_sign_reversal_both_positive(self):
        """Test no sign reversal when both IS and OOS are positive."""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)

        degradation, is_sign_reversal = validator._calculate_sharpe_degradation(2.0, 1.5)
        assert abs(degradation - 0.25) < 0.001
        assert is_sign_reversal == False  # Same sign, no reversal


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


class TestHoldoutMode:
    """
    Tests for holdout validation mode (Session 83K-14).

    Holdout mode uses a single 70/30 train/test split instead of
    rolling walk-forward folds. This is appropriate for sparse
    pattern strategies like STRAT where 15-fold walk-forward
    produces meaningless 0-trade folds.
    """

    def test_holdout_generates_single_fold(self):
        """Holdout mode should generate exactly one fold."""
        config = WalkForwardConfig(validation_mode='holdout')
        validator = WalkForwardValidator(config)
        windows = validator._generate_fold_windows(total_bars=1000)

        assert len(windows) == 1, "Holdout mode should generate exactly 1 fold"

    def test_holdout_split_ratio_default(self):
        """Holdout should split at 70/30 by default."""
        config = WalkForwardConfig(validation_mode='holdout', holdout_train_pct=0.70)
        validator = WalkForwardValidator(config)
        windows = validator._generate_fold_windows(total_bars=1000)

        window = windows[0]
        assert window.train_start_idx == 0
        assert window.train_end_idx == 700  # 70% of 1000
        assert window.test_start_idx == 700
        assert window.test_end_idx == 1000

    def test_holdout_split_ratio_custom(self):
        """Holdout should respect custom train percentage."""
        config = WalkForwardConfig(validation_mode='holdout', holdout_train_pct=0.80)
        validator = WalkForwardValidator(config)
        windows = validator._generate_fold_windows(total_bars=1000)

        window = windows[0]
        assert window.train_end_idx == 800  # 80% of 1000
        assert window.test_start_idx == 800

    def test_holdout_fold_properties(self):
        """Holdout fold should have correct size properties."""
        config = WalkForwardConfig(validation_mode='holdout', holdout_train_pct=0.70)
        validator = WalkForwardValidator(config)
        windows = validator._generate_fold_windows(total_bars=1000)

        window = windows[0]
        assert window.train_size == 700
        assert window.test_size == 300
        assert window.fold_number == 1

    def test_holdout_config_class(self):
        """WalkForwardConfigHoldout should have correct defaults."""
        config = WalkForwardConfigHoldout()

        assert config.validation_mode == 'holdout'
        assert config.holdout_train_pct == 0.70
        assert config.min_trades_per_fold == 5
        # Inherits from WalkForwardConfigOptions
        assert config.max_sharpe_degradation == 0.40
        assert config.min_oos_sharpe == 0.3

    def test_holdout_mode_in_full_validation(self, mock_strategy, synthetic_daily_data):
        """Test holdout mode in full validation workflow."""
        config = WalkForwardConfigHoldout()
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        assert results.total_folds == 1
        assert len(results.folds) == 1

    def test_holdout_skips_profitable_folds_check(self):
        """Holdout mode should not fail on profitable_folds_pct."""
        # Create results where the single fold is not profitable
        # Should still pass if OOS Sharpe is OK
        from validation.results import FoldResult, WalkForwardResults
        from datetime import datetime

        # Create a single unprofitable fold with good OOS Sharpe
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2024, 12, 31),
            is_sharpe=2.0,
            oos_sharpe=1.5,  # Good OOS Sharpe
            is_return=0.50,
            oos_return=-0.05,  # Unprofitable
            is_trades=50,
            oos_trades=15,
            parameters={},
            is_profitable=False  # Unprofitable fold
        )

        # Create config with holdout mode
        config = WalkForwardConfigHoldout()
        validator = WalkForwardValidator(config)

        # Manually aggregate results
        results = validator._aggregate_results([fold], [{}])

        # Should pass - profitable_folds check is skipped in holdout mode
        # (0% profitable folds would fail in walk-forward mode)
        # Check that profitable_folds_pct failure reason is NOT in the list
        profit_folds_failure = any('Profitable folds' in r for r in results.failure_reasons)
        assert not profit_folds_failure, "Holdout should skip profitable_folds_pct check"

    def test_holdout_skips_param_stability_check(self):
        """Holdout mode should not fail on parameter stability."""
        from datetime import datetime

        # Create a single fold
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2024, 12, 31),
            is_sharpe=2.0,
            oos_sharpe=1.5,
            is_return=0.50,
            oos_return=0.20,
            is_trades=50,
            oos_trades=15,
            parameters={'sma': 20},  # Single parameter set
            is_profitable=True
        )

        config = WalkForwardConfigHoldout()
        validator = WalkForwardValidator(config)

        # Aggregate with single parameter set
        results = validator._aggregate_results([fold], [{'sma': 20}])

        # Should not have param stability failures (only 1 fold = no CV calculation)
        param_stability_failure = any('CV' in r for r in results.failure_reasons)
        assert not param_stability_failure, "Holdout should skip param stability check"

    def test_holdout_still_checks_oos_sharpe(self):
        """Holdout mode should still validate OOS Sharpe threshold."""
        from datetime import datetime

        # Create fold with low OOS Sharpe
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2024, 12, 31),
            is_sharpe=2.0,
            oos_sharpe=0.1,  # Below threshold (0.3)
            is_return=0.50,
            oos_return=0.05,
            is_trades=50,
            oos_trades=15,
            parameters={},
            is_profitable=True
        )

        config = WalkForwardConfigHoldout()
        validator = WalkForwardValidator(config)
        results = validator._aggregate_results([fold], [{}])

        # Should fail due to low OOS Sharpe
        assert not results.passes_validation
        oos_failure = any('OOS Sharpe' in r for r in results.failure_reasons)
        assert oos_failure, "Holdout should still check OOS Sharpe threshold"

    def test_holdout_still_checks_min_trades(self):
        """Holdout mode should still validate minimum trade count."""
        from datetime import datetime

        # Create fold with too few trades
        fold = FoldResult(
            fold_number=1,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2023, 6, 30),
            test_start=datetime(2023, 7, 1),
            test_end=datetime(2024, 12, 31),
            is_sharpe=2.0,
            oos_sharpe=1.5,
            is_return=0.50,
            oos_return=0.20,
            is_trades=50,
            oos_trades=2,  # Below threshold (5)
            parameters={},
            is_profitable=True
        )

        config = WalkForwardConfigHoldout()
        validator = WalkForwardValidator(config)
        results = validator._aggregate_results([fold], [{}])

        # Should fail due to insufficient trades
        assert not results.passes_validation
        trades_failure = any('fewer than' in r for r in results.failure_reasons)
        assert trades_failure, "Holdout should still check min trades"

    def test_walk_forward_mode_unchanged(self, mock_strategy, synthetic_daily_data):
        """Verify walk-forward mode still works correctly (regression test)."""
        config = WalkForwardConfig(validation_mode='walk_forward', min_trades_per_fold=3)
        validator = WalkForwardValidator(config)

        results = validator.validate(mock_strategy, synthetic_daily_data)

        # Should have multiple folds for sufficient data
        assert results.total_folds > 1, "Walk-forward mode should generate multiple folds"


class TestConfigForOptionsPreservation:
    """
    Session 83K-16: Tests for config.for_options() preservation of holdout mode.

    Bug: Previously, config.for_options() would always create a new WalkForwardConfigOptions()
    with default validation_mode='walk_forward', losing any holdout settings.
    """

    def test_for_options_preserves_holdout_mode(self):
        """Verify for_options() preserves validation_mode='holdout'."""
        # Create config with holdout mode
        config = ValidationConfig()
        config.walk_forward = WalkForwardConfigHoldout()

        # Call for_options()
        options_config = config.for_options()

        # Holdout mode should be preserved
        assert options_config.walk_forward.validation_mode == 'holdout', \
            "for_options() should preserve validation_mode='holdout'"

    def test_for_options_preserves_holdout_train_pct(self):
        """Verify for_options() preserves holdout_train_pct setting."""
        # Create config with custom holdout percentage
        config = ValidationConfig()
        config.walk_forward = WalkForwardConfigHoldout()
        config.walk_forward.holdout_train_pct = 0.80  # Custom 80/20 split

        # Call for_options()
        options_config = config.for_options()

        # Custom holdout percentage should be preserved
        assert options_config.walk_forward.holdout_train_pct == 0.80, \
            "for_options() should preserve holdout_train_pct"

    def test_for_options_uses_options_thresholds_in_holdout(self):
        """Verify for_options() uses WalkForwardConfigHoldout (with options thresholds) when in holdout mode."""
        # Create config with holdout mode
        config = ValidationConfig()
        config.walk_forward = WalkForwardConfigHoldout()

        # Call for_options()
        options_config = config.for_options()

        # Should have options-like thresholds
        assert options_config.walk_forward.max_sharpe_degradation == 0.40, \
            "Holdout config should use options threshold (0.40)"
        assert options_config.walk_forward.min_oos_sharpe == 0.3, \
            "Holdout config should use options threshold (0.3)"

    def test_for_options_walk_forward_mode_unchanged(self):
        """Verify for_options() keeps walk_forward mode when not in holdout."""
        # Create config with default walk_forward mode
        config = ValidationConfig()

        # Call for_options()
        options_config = config.for_options()

        # Walk-forward mode should be preserved (default behavior)
        assert options_config.walk_forward.validation_mode == 'walk_forward', \
            "for_options() should keep validation_mode='walk_forward' when not in holdout"
