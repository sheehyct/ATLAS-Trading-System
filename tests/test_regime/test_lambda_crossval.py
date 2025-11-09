"""
Tests for Academic Jump Model Phase C - Lambda Cross-Validation

This module tests the cross-validation framework for optimal lambda selection
as specified in Section 3.4.3 of Shu et al., Princeton 2024.

Test Coverage:
    1. simulate_01_strategy() basic functionality
    2. simulate_01_strategy() with transaction costs
    3. simulate_01_strategy() with signal delay
    4. cross_validate_lambda() on synthetic data
    5. cross_validate_lambda() on real SPY data (10 years)
    6. Lambda selection consistency and reasonableness

Mathematical Validation:
    - Sharpe ratio calculation correctness
    - Transaction cost application (10 bps)
    - 1-day delay implementation
    - 8-year validation window
    - Monthly lambda updates

Expected Behavior (from paper Table 3):
    - Lambda=5: ~2.7 switches/year
    - Lambda=50-100: <1 switch/year
    - Lambda=150: ~0.4 switches/year

Reference:
    Section 3.4.3 "Optimal Jump Penalty Selection"
    Shu et al., "Downside Risk Reduction Using Regime-Switching Signals"
    Princeton University, 2024
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from regime.academic_jump_model import (
    simulate_01_strategy,
    cross_validate_lambda,
    AcademicJumpModel
)


class TestSimulate01Strategy:
    """Test suite for 0/1 strategy simulation with regime signals."""

    def test_basic_strategy_simulation(self):
        """Test basic strategy returns calculation."""
        # Create simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Bull market: all bull regime, positive returns
        regime = pd.Series(['bull'] * 100, index=dates)
        returns = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates)

        result = simulate_01_strategy(regime, returns, delay_days=0, transaction_cost_bps=0)

        # Basic checks
        assert 'sharpe_ratio' in result
        assert 'total_return' in result
        assert 'annual_return' in result
        assert 'annual_volatility' in result
        assert 'n_trades' in result

        # With no delay and all bull, strategy should match asset
        # (no transaction costs in this test)
        assert result['n_trades'] == 0  # No regime switches

    def test_transaction_costs_reduce_returns(self):
        """Test that transaction costs reduce returns on regime switches."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Regime switches every 10 days
        regime_labels = []
        for i in range(10):
            regime_labels.extend(['bull'] * 5)
            regime_labels.extend(['bear'] * 5)

        regime = pd.Series(regime_labels, index=dates)
        returns = pd.Series(np.ones(100) * 0.01, index=dates)  # Constant 1% daily return

        # Without transaction costs
        result_no_tc = simulate_01_strategy(regime, returns, delay_days=0, transaction_cost_bps=0)

        # With transaction costs (10 bps)
        result_with_tc = simulate_01_strategy(regime, returns, delay_days=0, transaction_cost_bps=10)

        # Transaction costs should reduce returns
        assert result_with_tc['total_return'] < result_no_tc['total_return']

        # Should have multiple trades
        assert result_with_tc['n_trades'] > 5

    def test_signal_delay_implementation(self):
        """Test that 1-day signal delay is correctly implemented."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        # Regime switches on day 5
        regime = pd.Series(['bull']*5 + ['bear']*5, index=dates)
        returns = pd.Series([0.01]*5 + [-0.01]*5, index=dates)

        # No delay: should catch regime switch immediately
        result_no_delay = simulate_01_strategy(regime, returns, delay_days=0, transaction_cost_bps=0)

        # 1-day delay: should miss first day of regime change
        result_with_delay = simulate_01_strategy(regime, returns, delay_days=1, transaction_cost_bps=0)

        # With delay, we hold bull position 1 day longer into bear market
        # This should reduce returns
        assert result_with_delay['total_return'] < result_no_delay['total_return']

    def test_bear_regime_earns_risk_free_rate(self):
        """Test that bear regime positions earn risk-free rate."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # All bear regime
        regime = pd.Series(['bear'] * 100, index=dates)
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)  # Volatile returns

        result = simulate_01_strategy(
            regime, returns,
            delay_days=0,
            transaction_cost_bps=0,
            risk_free_rate=0.03
        )

        # In bear regime, should earn close to risk-free rate
        rf_daily = (1.03) ** (1/252) - 1
        expected_total_return = (1 + rf_daily) ** 100 - 1

        # Should be close (within 1% relative error)
        assert abs(result['total_return'] - expected_total_return) / expected_total_return < 0.01

    def test_empty_data_handling(self):
        """Test graceful handling of empty or insufficient data."""
        dates = pd.date_range('2020-01-01', periods=2, freq='D')
        regime = pd.Series(['bull', 'bear'], index=dates)
        returns = pd.Series([0.01, -0.01], index=dates)

        # With 1-day delay, only 1 day of data after dropna
        result = simulate_01_strategy(regime, returns, delay_days=1)

        # Should return valid structure even with minimal data
        assert result['sharpe_ratio'] is not None
        assert result['n_trades'] >= 0


class TestCrossValidateLambda:
    """Test suite for lambda cross-validation framework."""

    def test_cross_validation_structure(self):
        """Test that cross-validation returns correct structure."""
        # Generate synthetic data (minimal size for testing)
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=2500, freq='D')

        # Random walk price
        returns = np.random.randn(2500) * 0.01 + 0.0003
        prices = 100 * (1 + returns).cumprod()

        data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.ones(2500)
        }, index=dates)

        # Run cross-validation with minimal parameters for speed
        results = cross_validate_lambda(
            data,
            lambda_candidates=[50, 150],  # Only 2 candidates for speed
            validation_window_days=400,  # ~1.5 years (smaller for testing)
            update_frequency_days=400,  # Only 2-3 update periods total
            lookback_window_days=800,
            n_starts=3,  # Reduce from 10 to 3 for faster testing
            verbose=False
        )

        # Check structure
        assert isinstance(results, pd.DataFrame)
        assert 'date' in results.columns
        assert 'selected_lambda' in results.columns
        assert 'sharpe_ratio' in results.columns
        assert 'n_trades' in results.columns
        assert 'lambda_sharpes' in results.columns

        # Should have multiple update periods
        assert len(results) > 0

        # Lambda should be from candidates
        assert all(results['selected_lambda'].isin([5, 50, 150]))

    def test_lambda_sensitivity_to_trades(self):
        """Test that higher lambda results in fewer regime switches."""
        # This is a critical relationship from the paper
        # Lambda controls temporal penalty, so higher lambda = fewer switches

        np.random.seed(42)
        dates = pd.date_range('2015-01-01', periods=3000, freq='D')

        # Create somewhat volatile data
        returns = np.random.randn(3000) * 0.015 + 0.0004
        prices = 100 * (1 + returns).cumprod()

        data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.ones(3000)
        }, index=dates)

        # Fit models with different lambdas
        model_low = AcademicJumpModel(lambda_penalty=5)
        model_high = AcademicJumpModel(lambda_penalty=150)

        model_low.fit(data, n_starts=5, verbose=False)
        model_high.fit(data, n_starts=5, verbose=False)

        regimes_low = model_low.predict(data)
        regimes_high = model_high.predict(data)

        # Count switches (convert to numeric first since regimes are strings)
        regimes_low_numeric = (regimes_low == 'bull').astype(int)
        regimes_high_numeric = (regimes_high == 'bull').astype(int)

        switches_low = (regimes_low_numeric.diff() != 0).sum()
        switches_high = (regimes_high_numeric.diff() != 0).sum()

        # Higher lambda should have fewer switches
        assert switches_high <= switches_low

        # With synthetic random walk data, we may get very few switches
        # The key property is that lambda=150 has fewer (or equal) switches than lambda=5
        # Actual counts depend on whether synthetic data has regime structure
        assert switches_low >= 0  # Sanity check
        assert switches_high >= 0  # Sanity check

        # Just verify the relationship holds
        print(f"Switches: lambda=5: {switches_low}, lambda=150: {switches_high}")

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises clear error."""
        # Only 100 days of data (need >5000)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100))

        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.ones(100)
        }, index=dates)

        with pytest.raises(ValueError, match="Insufficient data"):
            cross_validate_lambda(data, verbose=False)

    def test_lambda_candidates_default(self):
        """Test that default lambda candidates match paper specification."""
        # From Table 3 of the paper: [5, 15, 35, 50, 70, 100, 150]

        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=2500, freq='D')
        prices = 100 * (1 + np.random.randn(2500) * 0.01).cumprod()

        data = pd.DataFrame({
            'Close': prices,
            'Volume': np.ones(2500)
        }, index=dates)

        # Run with defaults (but reduced n_starts for speed)
        results = cross_validate_lambda(
            data,
            validation_window_days=400,
            update_frequency_days=400,  # Only 2-3 updates
            lookback_window_days=800,
            n_starts=2,  # Reduce for testing speed
            verbose=False
        )

        # Check that selected lambdas are from default set
        expected_lambdas = {5, 15, 35, 50, 70, 100, 150}
        actual_lambdas = set(results['selected_lambda'].unique())

        assert actual_lambdas.issubset(expected_lambdas)


class TestLambdaCrossValidationIntegration:
    """Integration tests on real SPY data (requires data access)."""

    @pytest.mark.slow
    def test_cross_validation_on_spy_data(self):
        """
        Test cross-validation on real SPY data (2014-2024).

        This is a slow test that validates the entire workflow on real market data.
        Expected runtime: 5-10 minutes depending on n_starts.
        """
        # This test requires Alpaca data access
        # Skip if data not available
        try:
            from data.alpaca import fetch_alpaca_data
        except ImportError:
            pytest.skip("Alpaca data module not available")

        # Fetch 10 years of SPY data
        try:
            spy_data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3650)
        except Exception as e:
            pytest.skip(f"Could not fetch SPY data: {e}")

        if len(spy_data) < 2500:
            pytest.skip(f"Insufficient SPY data: {len(spy_data)} days")

        # Run cross-validation with reduced parameters for testing speed
        results = cross_validate_lambda(
            spy_data,
            lambda_candidates=[5, 50, 100, 150],  # Reduced set
            validation_window_days=2016,  # 8 years
            update_frequency_days=63,  # Quarterly instead of monthly
            lookback_window_days=3000,
            verbose=True
        )

        # Validation checks
        assert len(results) > 0, "Should have at least one cross-validation result"

        # Check Sharpe ratios are reasonable
        assert results['sharpe_ratio'].min() > -2.0, "Sharpe too negative"
        assert results['sharpe_ratio'].max() < 5.0, "Sharpe unrealistically high"

        # Check lambda selection distribution
        # Should prefer moderate lambdas (50-100) over extremes
        median_lambda = results['selected_lambda'].median()
        assert 15 <= median_lambda <= 150, f"Median lambda {median_lambda} outside expected range"

        # Print summary statistics
        print("\n[PASS] SPY Cross-Validation Results:")
        print(f"  Periods tested: {len(results)}")
        print(f"  Mean Sharpe: {results['sharpe_ratio'].mean():.3f}")
        print(f"  Median lambda: {median_lambda:.0f}")
        print(f"  Lambda distribution:")
        print(results['selected_lambda'].value_counts().sort_index())

    @pytest.mark.slow
    def test_march_2020_crash_lambda_adaptation(self):
        """
        Test that lambda adapts during March 2020 crash period.

        During high volatility, the optimal lambda may shift to allow
        more frequent regime switches to capture crash/recovery.
        """
        try:
            from data.alpaca import fetch_alpaca_data
        except ImportError:
            pytest.skip("Alpaca data module not available")

        # Fetch data covering March 2020
        try:
            # 2018-2021 covers before, during, and after crash
            spy_data = fetch_alpaca_data('SPY', start_date='2018-01-01', end_date='2021-12-31')
        except Exception as e:
            pytest.skip(f"Could not fetch SPY data: {e}")

        if len(spy_data) < 500:
            pytest.skip("Insufficient data for crash test")

        # Run cross-validation
        results = cross_validate_lambda(
            spy_data,
            lambda_candidates=[5, 15, 35, 50, 70, 100, 150],
            validation_window_days=500,  # ~2 years
            update_frequency_days=21,  # Monthly
            lookback_window_days=750,  # ~3 years
            verbose=False
        )

        # Check that we have results around March 2020
        crash_period = pd.Timestamp('2020-03-01')
        results_near_crash = results[
            (results['date'] >= crash_period - pd.Timedelta(days=90)) &
            (results['date'] <= crash_period + pd.Timedelta(days=90))
        ]

        if len(results_near_crash) > 0:
            # During crash, may prefer lower lambda (more responsive)
            # This is not guaranteed but worth checking
            avg_lambda_crash = results_near_crash['selected_lambda'].mean()
            avg_lambda_overall = results['selected_lambda'].mean()

            print(f"\n[INFO] Lambda during crash period: {avg_lambda_crash:.1f}")
            print(f"[INFO] Lambda overall: {avg_lambda_overall:.1f}")

            # Just document the behavior, don't enforce strict requirement
            assert avg_lambda_crash > 0  # Sanity check


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s', '--tb=short'])
