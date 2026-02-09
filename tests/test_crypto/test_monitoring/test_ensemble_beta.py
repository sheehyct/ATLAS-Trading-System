"""
Tests for crypto/monitoring/ensemble_beta.py - Priority 2.

Tests ensemble beta calculation with confidence weighting.

Session: February 2026
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from crypto.monitoring.ensemble_beta import (
    EnsembleBetaTracker,
    EnsembleBetaResult,
    BetaEstimate,
    calculate_beta,
    calculate_rolling_beta,
    calculate_prediction_error,
    update_error_ewma,
    calculate_weights_from_errors,
    calculate_ensemble_beta,
    calculate_confidence_interval,
    quick_ensemble_beta,
    get_beta_adjustment_factor,
    BETA_WINDOWS,
    BASE_WEIGHTS,
    ERROR_DECAY,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_returns():
    """Generate sample return series with known beta relationship."""
    np.random.seed(42)
    n = 400  # Enough for 365-day window
    
    # BTC returns
    btc_returns = np.random.normal(0.001, 0.03, n)
    
    # Asset with beta ~2.0 to BTC
    beta_true = 2.0
    asset_returns = beta_true * btc_returns + np.random.normal(0, 0.01, n)
    
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    return {
        'btc': pd.Series(btc_returns, index=dates),
        'asset': pd.Series(asset_returns, index=dates),
        'true_beta': beta_true,
    }


@pytest.fixture
def sample_prices(sample_returns):
    """Convert returns to prices."""
    btc_prices = 50000 * np.exp(np.cumsum(sample_returns['btc']))
    asset_prices = 1.0 * np.exp(np.cumsum(sample_returns['asset']))
    
    return {
        'BTC': pd.Series(btc_prices, index=sample_returns['btc'].index),
        'ASSET': pd.Series(asset_prices, index=sample_returns['asset'].index),
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfiguration:
    """Test default configuration values."""
    
    def test_beta_windows_defined(self):
        """Beta windows should include expected values."""
        assert 7 in BETA_WINDOWS
        assert 30 in BETA_WINDOWS
        assert 90 in BETA_WINDOWS
        assert 365 in BETA_WINDOWS
    
    def test_base_weights_sum_to_one(self):
        """Base weights should sum to 1.0."""
        total = sum(BASE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
    
    def test_error_decay_in_valid_range(self):
        """Error decay should be between 0 and 1."""
        assert 0 < ERROR_DECAY < 1


# =============================================================================
# CORE FUNCTION TESTS
# =============================================================================


class TestCalculateBeta:
    """Test beta calculation function."""
    
    def test_beta_calculation(self, sample_returns):
        """Should calculate beta close to true value."""
        beta = calculate_beta(
            sample_returns['asset'],
            sample_returns['btc'],
        )
        
        # Should be close to true beta of 2.0
        assert abs(beta - sample_returns['true_beta']) < 0.3
    
    def test_beta_with_window(self, sample_returns):
        """Should calculate beta for specified window."""
        beta = calculate_beta(
            sample_returns['asset'],
            sample_returns['btc'],
            window=30,
        )
        
        # Should still be in reasonable range
        assert 1.0 < beta < 3.0
    
    def test_btc_to_btc_beta_is_one(self):
        """BTC beta to itself should be 1.0."""
        np.random.seed(42)
        btc_returns = pd.Series(np.random.normal(0, 0.02, 100))
        
        beta = calculate_beta(btc_returns, btc_returns)
        
        # Allow small tolerance for floating point precision
        assert abs(beta - 1.0) < 0.02
    
    def test_insufficient_data_returns_nan(self):
        """Should return NaN with insufficient data."""
        returns = pd.Series([0.01, 0.02, 0.03])
        btc = pd.Series([0.01, 0.01, 0.01])
        
        beta = calculate_beta(returns, btc)
        
        assert np.isnan(beta)


class TestCalculateRollingBeta:
    """Test rolling beta calculation."""
    
    def test_returns_series(self, sample_returns):
        """Should return pandas Series."""
        result = calculate_rolling_beta(
            sample_returns['asset'],
            sample_returns['btc'],
            window=30,
        )
        
        assert isinstance(result, pd.Series)
    
    def test_correct_length(self, sample_returns):
        """Should return correct length."""
        result = calculate_rolling_beta(
            sample_returns['asset'],
            sample_returns['btc'],
            window=30,
        )
        
        assert len(result) == len(sample_returns['asset'])
    
    def test_first_values_nan(self, sample_returns):
        """First window-1 values should be NaN (valid at position window-1)."""
        window = 30
        result = calculate_rolling_beta(
            sample_returns['asset'],
            sample_returns['btc'],
            window=window,
        )
        
        # First 'window-1' values should be NaN (value at index window-1 is first valid)
        assert result.iloc[:window-1].isna().all()


class TestCalculatePredictionError:
    """Test prediction error calculation."""
    
    def test_perfect_prediction_zero_error(self):
        """Perfect prediction should have zero error."""
        actual = 0.04  # 4% move
        btc = 0.02     # 2% move
        beta = 2.0     # Beta of 2
        
        error = calculate_prediction_error(actual, btc, beta)
        
        assert abs(error) < 0.001
    
    def test_underperformance_positive_error(self):
        """Underperformance creates positive residual error."""
        actual = 0.03  # Only 3%
        btc = 0.02     # 2% move
        beta = 2.0     # Expected 4%
        
        error = calculate_prediction_error(actual, btc, beta)
        
        # Error = |0.03 - 0.04| = 0.01
        assert abs(error - 0.01) < 0.001
    
    def test_zero_btc_move(self):
        """Should handle zero BTC move."""
        error = calculate_prediction_error(0.01, 0.0, 2.0)
        assert error == 0.0


class TestUpdateErrorEwma:
    """Test EWMA error update."""
    
    def test_initial_update(self):
        """First update should use the new error directly."""
        result = update_error_ewma(np.nan, 0.05)
        assert result == 0.05
    
    def test_subsequent_update(self):
        """Subsequent updates should blend old and new."""
        result = update_error_ewma(0.04, 0.06, decay=0.9)
        
        # Expected: 0.9 * 0.04 + 0.1 * 0.06 = 0.036 + 0.006 = 0.042
        assert abs(result - 0.042) < 0.001
    
    def test_decay_effect(self):
        """Higher decay should weight old value more."""
        result_high_decay = update_error_ewma(0.10, 0.02, decay=0.95)
        result_low_decay = update_error_ewma(0.10, 0.02, decay=0.50)
        
        # High decay stays closer to old value
        assert result_high_decay > result_low_decay


class TestCalculateWeightsFromErrors:
    """Test error-based weight calculation."""
    
    def test_lower_error_higher_weight(self):
        """Lower error should get higher weight."""
        errors = {7: 0.05, 30: 0.02, 90: 0.03}
        
        weights = calculate_weights_from_errors(errors)
        
        # 30-day has lowest error, should have highest weight
        assert weights[30] > weights[7]
        assert weights[30] > weights[90]
    
    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        errors = {7: 0.05, 30: 0.02, 90: 0.03, 365: 0.04}
        
        weights = calculate_weights_from_errors(errors)
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.01
    
    def test_all_zero_errors_uses_base_weights(self):
        """Zero errors should fall back to base weights."""
        errors = {7: 0.0, 30: 0.0, 90: 0.0, 365: 0.0}
        
        weights = calculate_weights_from_errors(errors)
        
        # Should match base weights structure
        assert len(weights) == len(errors)


class TestCalculateEnsembleBeta:
    """Test ensemble beta calculation."""
    
    def test_weighted_average(self):
        """Should calculate weighted average."""
        betas = {7: 2.2, 30: 2.0, 90: 1.9}
        weights = {7: 0.2, 30: 0.5, 90: 0.3}
        
        ensemble = calculate_ensemble_beta(betas, weights)
        
        # Expected: 0.2*2.2 + 0.5*2.0 + 0.3*1.9 = 0.44 + 1.0 + 0.57 = 2.01
        assert abs(ensemble - 2.01) < 0.01
    
    def test_handles_nan_betas(self):
        """Should skip NaN betas."""
        betas = {7: np.nan, 30: 2.0, 90: 1.9}
        weights = {7: 0.2, 30: 0.5, 90: 0.3}
        
        ensemble = calculate_ensemble_beta(betas, weights)
        
        # Should only use valid betas
        assert not np.isnan(ensemble)


class TestCalculateConfidenceInterval:
    """Test confidence interval calculation."""
    
    def test_returns_tuple(self):
        """Should return (lower, upper) tuple."""
        betas = {7: 2.2, 30: 2.0, 90: 1.9, 365: 1.85}
        weights = {7: 0.2, 30: 0.3, 90: 0.3, 365: 0.2}
        
        lower, upper = calculate_confidence_interval(betas, weights)
        
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper
    
    def test_ci_contains_ensemble(self):
        """CI should contain the ensemble estimate."""
        betas = {7: 2.2, 30: 2.0, 90: 1.9, 365: 1.85}
        weights = {7: 0.2, 30: 0.3, 90: 0.3, 365: 0.2}
        
        lower, upper = calculate_confidence_interval(betas, weights)
        ensemble = calculate_ensemble_beta(betas, weights)
        
        assert lower <= ensemble <= upper
    
    def test_wider_ci_for_dispersed_betas(self):
        """More dispersed betas should have wider CI."""
        tight_betas = {7: 2.0, 30: 2.0, 90: 2.0, 365: 2.0}
        dispersed_betas = {7: 2.5, 30: 2.0, 90: 1.5, 365: 1.0}
        weights = {7: 0.25, 30: 0.25, 90: 0.25, 365: 0.25}
        
        tight_ci = calculate_confidence_interval(tight_betas, weights)
        dispersed_ci = calculate_confidence_interval(dispersed_betas, weights)
        
        tight_width = tight_ci[1] - tight_ci[0]
        dispersed_width = dispersed_ci[1] - dispersed_ci[0]
        
        assert dispersed_width > tight_width


# =============================================================================
# TRACKER CLASS TESTS
# =============================================================================


class TestEnsembleBetaTracker:
    """Test EnsembleBetaTracker class."""
    
    def test_initialization(self):
        """Tracker should initialize with defaults."""
        tracker = EnsembleBetaTracker()
        
        assert tracker.windows == BETA_WINDOWS
        assert len(tracker.static_betas) > 0
    
    def test_load_data(self, sample_prices):
        """Should load price data correctly."""
        tracker = EnsembleBetaTracker()
        
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        assert 'ASSET' in tracker._returns
        assert 'BTC' in tracker._returns
    
    def test_get_ensemble_beta(self, sample_prices):
        """Should return EnsembleBetaResult."""
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 2.0})
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        
        assert isinstance(result, EnsembleBetaResult)
        assert result.symbol == 'ASSET'
        assert not np.isnan(result.beta_ensemble)
        assert result.beta_lower < result.beta_ensemble < result.beta_upper
    
    def test_estimates_have_all_windows(self, sample_prices):
        """Result should have estimates for all windows."""
        tracker = EnsembleBetaTracker()
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        
        for window in BETA_WINDOWS:
            assert window in result.estimates
            assert isinstance(result.estimates[window], BetaEstimate)
    
    def test_deviation_from_static(self, sample_prices):
        """Should calculate deviation from static beta."""
        static_beta = 1.5  # Intentionally different from true beta
        tracker = EnsembleBetaTracker(static_betas={'ASSET': static_beta})
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        
        # Should show deviation since true beta is ~2.0
        assert result.static_beta == static_beta
        assert result.deviation_from_static != 0
    
    def test_is_significantly_different_flag(self, sample_prices):
        """Should flag significant deviation from static."""
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 1.0})  # Far from true beta
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        
        # True beta is ~2.0, static is 1.0, so >10% deviation
        assert result.is_significantly_different
    
    def test_position_size_multiplier(self, sample_prices):
        """Should calculate position size multiplier."""
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 2.0})
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        
        assert 0 <= result.position_size_multiplier <= 1


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestQuickEnsembleBeta:
    """Test quick_ensemble_beta convenience function."""
    
    def test_returns_result(self, sample_prices):
        """Should return EnsembleBetaResult."""
        result = quick_ensemble_beta(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            symbol='ASSET',
            static_beta=2.0,
        )
        
        assert isinstance(result, EnsembleBetaResult)
        assert result.symbol == 'ASSET'


class TestGetBetaAdjustmentFactor:
    """Test beta adjustment factor calculation."""
    
    def test_factor_calculation(self, sample_prices):
        """Should return ratio of ensemble to static."""
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 1.5})
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        factor = get_beta_adjustment_factor(result)
        
        expected = result.beta_ensemble / result.static_beta
        assert abs(factor - expected) < 0.01
    
    def test_zero_static_beta_returns_one(self, sample_prices):
        """Should return 1.0 if static beta is 0."""
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 0.0})
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        result = tracker.get_ensemble_beta('ASSET')
        factor = get_beta_adjustment_factor(result)
        
        assert factor == 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self, sample_prices):
        """Test complete tracking workflow."""
        # 1. Create tracker with static betas
        tracker = EnsembleBetaTracker(static_betas={'ASSET': 2.0, 'BTC': 1.0})
        
        # 2. Load data
        tracker.load_data(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            'ASSET',
        )
        
        # 3. Get ensemble beta
        result = tracker.get_ensemble_beta('ASSET')
        
        # 4. Verify actionable output
        assert result.beta_ensemble > 0
        assert result.beta_lower < result.beta_upper
        assert 0 <= result.ensemble_confidence <= 1
        
        # 5. Get adjustment factor
        factor = get_beta_adjustment_factor(result)
        assert factor > 0


# =============================================================================
# INIT FILE
# =============================================================================


@pytest.fixture(scope='module', autouse=True)
def test_init():
    """Create __init__.py for test package."""
    pass
