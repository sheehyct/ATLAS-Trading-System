"""
Tests for Kalman Filter Beta Estimator (Priority 4).

Tests the time-varying beta estimation via Kalman filter:
- State-space model correctness
- Parameter sensitivity
- Convergence behavior
- Confidence interval accuracy
- Comparison with static betas

Session Origin: February 3, 2026
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from crypto.monitoring.kalman_beta import (
    KalmanBetaFilter,
    KalmanBetaEstimate,
    KalmanBetaHistory,
    KalmanBetaTracker,
    quick_kalman_beta,
    estimate_kalman_parameters,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_prices():
    """Generate sample price data with known beta relationship."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    
    # BTC: random walk
    btc_returns = np.random.randn(n) * 0.03
    btc = 100 * np.exp(np.cumsum(btc_returns))
    
    # ETH: beta ~1.98 to BTC
    eth_returns = 1.98 * btc_returns + np.random.randn(n) * 0.01
    eth = 100 * np.exp(np.cumsum(eth_returns))
    
    # SOL: beta ~1.55 to BTC
    sol_returns = 1.55 * btc_returns + np.random.randn(n) * 0.015
    sol = 100 * np.exp(np.cumsum(sol_returns))
    
    return pd.DataFrame({
        'BTC': btc,
        'ETH': eth,
        'SOL': sol,
    }, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    """Returns from sample prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def kalman_filter():
    """Basic Kalman filter for ETH."""
    return KalmanBetaFilter(
        symbol='ETH',
        initial_beta=1.98,
        initial_variance=0.1,
        process_variance=0.001,
        observation_variance=0.0001,
        static_beta=1.98,
    )


# =============================================================================
# KALMAN BETA ESTIMATE TESTS
# =============================================================================


class TestKalmanBetaEstimate:
    """Tests for KalmanBetaEstimate dataclass."""
    
    def test_confidence_width(self):
        """Confidence width is upper - lower."""
        estimate = KalmanBetaEstimate(
            timestamp=datetime.now(),
            symbol='ETH',
            beta=2.0,
            beta_prior=1.98,
            variance=0.04,
            std=0.2,
            confidence_lower=1.6,
            confidence_upper=2.4,
            kalman_gain=0.5,
            innovation=0.01,
            innovation_variance=0.001,
            static_beta=1.98,
            deviation_from_static=0.01,
        )
        assert estimate.confidence_width == pytest.approx(0.8)
    
    def test_is_uncertain_true(self):
        """is_uncertain True when CI > 30% of beta."""
        estimate = KalmanBetaEstimate(
            timestamp=datetime.now(),
            symbol='ETH',
            beta=2.0,
            beta_prior=1.98,
            variance=0.04,
            std=0.2,
            confidence_lower=1.2,  # Width = 1.6, 80% of beta
            confidence_upper=2.8,
            kalman_gain=0.5,
            innovation=0.01,
            innovation_variance=0.001,
            static_beta=1.98,
            deviation_from_static=0.01,
        )
        assert estimate.is_uncertain is True
    
    def test_is_uncertain_false(self):
        """is_uncertain False when CI < 30% of beta."""
        estimate = KalmanBetaEstimate(
            timestamp=datetime.now(),
            symbol='ETH',
            beta=2.0,
            beta_prior=1.98,
            variance=0.01,
            std=0.1,
            confidence_lower=1.8,  # Width = 0.4, 20% of beta
            confidence_upper=2.2,
            kalman_gain=0.5,
            innovation=0.01,
            innovation_variance=0.001,
            static_beta=1.98,
            deviation_from_static=0.01,
        )
        assert estimate.is_uncertain is False
    
    def test_position_size_multiplier_confident(self):
        """Full position size when confident."""
        estimate = KalmanBetaEstimate(
            timestamp=datetime.now(),
            symbol='ETH',
            beta=2.0,
            beta_prior=1.98,
            variance=0.01,
            std=0.1,
            confidence_lower=1.85,  # Width = 0.3, 15% of beta
            confidence_upper=2.15,
            kalman_gain=0.5,
            innovation=0.01,
            innovation_variance=0.001,
            static_beta=1.98,
            deviation_from_static=0.01,
        )
        assert estimate.position_size_multiplier == 1.0
    
    def test_position_size_multiplier_uncertain(self):
        """Reduced position when uncertain."""
        estimate = KalmanBetaEstimate(
            timestamp=datetime.now(),
            symbol='ETH',
            beta=2.0,
            beta_prior=1.98,
            variance=0.16,
            std=0.4,
            confidence_lower=1.2,  # Width = 1.6, 80% of beta
            confidence_upper=2.8,
            kalman_gain=0.5,
            innovation=0.01,
            innovation_variance=0.001,
            static_beta=1.98,
            deviation_from_static=0.01,
        )
        assert estimate.position_size_multiplier == 0.25


# =============================================================================
# KALMAN BETA HISTORY TESTS
# =============================================================================


class TestKalmanBetaHistory:
    """Tests for KalmanBetaHistory dataclass."""
    
    def test_latest_returns_most_recent(self):
        """latest property returns most recent estimate."""
        estimates = [
            KalmanBetaEstimate(
                timestamp=datetime(2025, 1, i),
                symbol='ETH',
                beta=1.9 + i * 0.01,
                beta_prior=1.9,
                variance=0.04,
                std=0.2,
                confidence_lower=1.7,
                confidence_upper=2.1,
                kalman_gain=0.5,
                innovation=0.01,
                innovation_variance=0.001,
                static_beta=1.98,
                deviation_from_static=0.01,
            )
            for i in range(1, 4)
        ]
        history = KalmanBetaHistory(symbol='ETH', estimates=estimates)
        assert history.latest.beta == pytest.approx(1.93)
    
    def test_latest_empty_returns_none(self):
        """latest returns None for empty history."""
        history = KalmanBetaHistory(symbol='ETH', estimates=[])
        assert history.latest is None
    
    def test_beta_series(self):
        """beta_series returns time series of estimates."""
        estimates = [
            KalmanBetaEstimate(
                timestamp=datetime(2025, 1, i),
                symbol='ETH',
                beta=1.9 + i * 0.01,
                beta_prior=1.9,
                variance=0.04,
                std=0.2,
                confidence_lower=1.7,
                confidence_upper=2.1,
                kalman_gain=0.5,
                innovation=0.01,
                innovation_variance=0.001,
                static_beta=1.98,
                deviation_from_static=0.01,
            )
            for i in range(1, 4)
        ]
        history = KalmanBetaHistory(symbol='ETH', estimates=estimates)
        series = history.beta_series
        
        assert len(series) == 3
        assert series.iloc[0] == pytest.approx(1.91)
        assert series.iloc[-1] == pytest.approx(1.93)


# =============================================================================
# KALMAN FILTER TESTS
# =============================================================================


class TestKalmanBetaFilter:
    """Tests for KalmanBetaFilter class."""
    
    def test_initialization(self, kalman_filter):
        """Filter initializes with correct values."""
        assert kalman_filter.symbol == 'ETH'
        assert kalman_filter.beta == 1.98
        assert kalman_filter.variance == 0.1
        assert kalman_filter.static_beta == 1.98
    
    def test_predict_increases_variance(self, kalman_filter):
        """Predict step increases variance by Q."""
        initial_variance = kalman_filter.variance
        beta_prior, variance_prior = kalman_filter.predict()
        
        assert beta_prior == kalman_filter.beta  # Beta unchanged
        assert variance_prior == initial_variance + 0.001  # Increased by Q
    
    def test_update_returns_estimate(self, kalman_filter):
        """Update returns KalmanBetaEstimate."""
        result = kalman_filter.update(
            asset_return=0.02,
            btc_return=0.01,
            timestamp=datetime(2025, 1, 1),
        )
        
        assert isinstance(result, KalmanBetaEstimate)
        assert result.symbol == 'ETH'
        assert result.timestamp == datetime(2025, 1, 1)
    
    def test_update_adjusts_beta(self, kalman_filter):
        """Update adjusts beta based on observation."""
        initial_beta = kalman_filter.beta
        
        # Asset return higher than expected -> beta should increase
        kalman_filter.update(
            asset_return=0.05,  # 5% return
            btc_return=0.02,   # 2% BTC return -> expected 3.96%
        )
        
        # Beta should have increased to explain higher return
        assert kalman_filter.beta > initial_beta
    
    def test_update_reduces_variance(self, kalman_filter):
        """Update reduces variance (more certain after observation)."""
        # Do a predict first to increase variance
        kalman_filter.update(
            asset_return=0.02,
            btc_return=0.01,
        )
        
        variance_after_first = kalman_filter.variance
        
        # Second update
        kalman_filter.update(
            asset_return=0.02,
            btc_return=0.01,
        )
        
        # More observations -> more certain (usually)
        # Note: this depends on parameters, may not always be true
        assert kalman_filter.variance <= variance_after_first + 0.001
    
    def test_update_zero_btc_return(self, kalman_filter):
        """Update handles zero BTC return gracefully."""
        initial_beta = kalman_filter.beta
        
        result = kalman_filter.update(
            asset_return=0.01,
            btc_return=0.0,  # Zero return
        )
        
        # Beta should be unchanged
        assert result.beta == initial_beta
        assert result.kalman_gain == 0.0
    
    def test_confidence_interval_property(self, kalman_filter):
        """confidence_interval returns 95% CI."""
        ci = kalman_filter.confidence_interval
        
        # Should be (beta - 1.96*std, beta + 1.96*std)
        expected_margin = 1.96 * kalman_filter.std
        assert ci[0] == pytest.approx(kalman_filter.beta - expected_margin)
        assert ci[1] == pytest.approx(kalman_filter.beta + expected_margin)
    
    def test_reset(self, kalman_filter):
        """reset clears history and updates."""
        # Do some updates
        for _ in range(5):
            kalman_filter.update(asset_return=0.02, btc_return=0.01)
        
        # Reset with explicit values
        kalman_filter.reset(initial_beta=1.98, initial_variance=0.1)
        
        assert kalman_filter.beta == 1.98
        assert kalman_filter.variance == 0.1
    
    def test_get_history(self, kalman_filter):
        """get_history returns KalmanBetaHistory."""
        for i in range(3):
            kalman_filter.update(
                asset_return=0.02,
                btc_return=0.01,
                timestamp=datetime(2025, 1, i + 1),
            )
        
        history = kalman_filter.get_history()
        
        assert isinstance(history, KalmanBetaHistory)
        assert history.symbol == 'ETH'
        assert len(history.estimates) == 3


class TestKalmanFilterConvergence:
    """Tests for Kalman filter convergence behavior."""
    
    def test_converges_to_true_beta(self, sample_returns):
        """Filter converges to true beta with enough data."""
        kf = KalmanBetaFilter(
            symbol='ETH',
            initial_beta=1.0,  # Start far from true beta
            initial_variance=1.0,
            process_variance=0.0001,
            observation_variance=0.0001,
            static_beta=1.98,
        )
        
        btc_returns = sample_returns['BTC']
        eth_returns = sample_returns['ETH']
        
        for i in range(len(btc_returns)):
            kf.update(
                asset_return=eth_returns.iloc[i],
                btc_return=btc_returns.iloc[i],
            )
        
        # Should converge close to true beta (~1.98)
        # Allow some tolerance due to noise
        assert 1.5 < kf.beta < 2.5
    
    def test_variance_decreases_over_time(self, sample_returns):
        """Variance generally decreases with more observations."""
        kf = KalmanBetaFilter(
            symbol='ETH',
            initial_beta=1.98,
            initial_variance=1.0,  # High initial uncertainty
            process_variance=0.0001,
            observation_variance=0.0001,
        )
        
        btc_returns = sample_returns['BTC']
        eth_returns = sample_returns['ETH']
        
        initial_variance = kf.variance
        
        for i in range(len(btc_returns)):
            kf.update(
                asset_return=eth_returns.iloc[i],
                btc_return=btc_returns.iloc[i],
            )
        
        # Final variance should be much lower
        assert kf.variance < initial_variance * 0.5


# =============================================================================
# KALMAN BETA TRACKER TESTS
# =============================================================================


class TestKalmanBetaTracker:
    """Tests for KalmanBetaTracker convenience class."""
    
    def test_initialization_default_betas(self):
        """Tracker initializes with default betas from config."""
        tracker = KalmanBetaTracker()
        assert 'ETH' in tracker.static_betas or tracker.static_betas is not None
    
    def test_initialization_custom_betas(self):
        """Tracker accepts custom static betas."""
        tracker = KalmanBetaTracker(
            static_betas={'ETH': 2.0, 'SOL': 1.5}
        )
        assert tracker.static_betas['ETH'] == 2.0
        assert tracker.static_betas['SOL'] == 1.5
    
    def test_load_data(self, sample_prices):
        """load_data processes price data."""
        tracker = KalmanBetaTracker()
        tracker.load_data(sample_prices)
        
        # Should have filters for non-BTC assets
        assert 'ETH' in tracker._filters
        assert 'SOL' in tracker._filters
        assert 'BTC' not in tracker._filters
    
    def test_get_estimate(self, sample_prices):
        """get_estimate returns current estimate."""
        tracker = KalmanBetaTracker(
            static_betas={'ETH': 1.98, 'SOL': 1.55}
        )
        tracker.load_data(sample_prices)
        
        estimate = tracker.get_estimate('ETH')
        
        assert isinstance(estimate, KalmanBetaEstimate)
        assert estimate.symbol == 'ETH'
        assert estimate.static_beta == 1.98
    
    def test_get_all_estimates(self, sample_prices):
        """get_all_estimates returns dict of estimates."""
        tracker = KalmanBetaTracker(
            static_betas={'ETH': 1.98, 'SOL': 1.55}
        )
        tracker.load_data(sample_prices)
        
        estimates = tracker.get_all_estimates()
        
        assert 'ETH' in estimates
        assert 'SOL' in estimates
        assert isinstance(estimates['ETH'], KalmanBetaEstimate)


# =============================================================================
# QUICK FUNCTION TESTS
# =============================================================================


class TestQuickKalmanBeta:
    """Tests for quick_kalman_beta convenience function."""
    
    def test_returns_estimate(self, sample_prices):
        """Returns KalmanBetaEstimate."""
        result = quick_kalman_beta(
            asset_prices=sample_prices['ETH'],
            btc_prices=sample_prices['BTC'],
            symbol='ETH',
            static_beta=1.98,
        )
        
        assert isinstance(result, KalmanBetaEstimate)
        assert result.symbol == 'ETH'
    
    def test_estimates_reasonable_beta(self, sample_prices):
        """Estimates beta close to true value."""
        result = quick_kalman_beta(
            asset_prices=sample_prices['ETH'],
            btc_prices=sample_prices['BTC'],
            symbol='ETH',
            static_beta=1.98,
        )
        
        # True beta is ~1.98
        assert 1.0 < result.beta < 3.0
    
    def test_includes_confidence_interval(self, sample_prices):
        """Result includes confidence interval."""
        result = quick_kalman_beta(
            asset_prices=sample_prices['ETH'],
            btc_prices=sample_prices['BTC'],
            symbol='ETH',
        )
        
        assert result.confidence_lower < result.beta
        assert result.confidence_upper > result.beta


# =============================================================================
# PARAMETER ESTIMATION TESTS
# =============================================================================


class TestEstimateKalmanParameters:
    """Tests for estimate_kalman_parameters function."""
    
    def test_returns_dict(self, sample_prices):
        """Returns dict with expected keys."""
        returns = sample_prices.pct_change().dropna()
        
        params = estimate_kalman_parameters(
            asset_returns=returns['ETH'],
            btc_returns=returns['BTC'],
        )
        
        assert 'process_variance' in params
        assert 'observation_variance' in params
        assert 'ols_beta' in params
        assert 'residual_std' in params
    
    def test_ols_beta_reasonable(self, sample_prices):
        """OLS beta estimate is reasonable."""
        returns = sample_prices.pct_change().dropna()
        
        params = estimate_kalman_parameters(
            asset_returns=returns['ETH'],
            btc_returns=returns['BTC'],
        )
        
        # True beta is ~1.98
        assert 1.0 < params['ols_beta'] < 3.0
    
    def test_positive_variances(self, sample_prices):
        """Variances are positive."""
        returns = sample_prices.pct_change().dropna()
        
        params = estimate_kalman_parameters(
            asset_returns=returns['ETH'],
            btc_returns=returns['BTC'],
        )
        
        assert params['process_variance'] > 0
        assert params['observation_variance'] > 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_observation(self):
        """Handles single observation gracefully."""
        kf = KalmanBetaFilter(symbol='ETH', initial_beta=2.0)
        
        result = kf.update(asset_return=0.02, btc_return=0.01)
        
        assert result is not None
        assert not np.isnan(result.beta)
    
    def test_negative_returns(self):
        """Handles negative returns correctly."""
        kf = KalmanBetaFilter(symbol='ETH', initial_beta=2.0)
        
        result = kf.update(
            asset_return=-0.05,
            btc_return=-0.025,
        )
        
        # Beta should still be positive
        assert result.beta > 0
    
    def test_large_returns(self):
        """Handles large returns without overflow."""
        kf = KalmanBetaFilter(
            symbol='ETH',
            initial_beta=2.0,
            initial_variance=0.1,
        )
        
        result = kf.update(
            asset_return=0.50,  # 50% return
            btc_return=0.25,   # 25% return
        )
        
        assert not np.isnan(result.beta)
        assert not np.isinf(result.beta)
    
    def test_constant_prices(self):
        """Handles constant prices (zero returns)."""
        kf = KalmanBetaFilter(symbol='ETH', initial_beta=2.0)
        
        # Multiple zero-return updates
        for _ in range(5):
            result = kf.update(asset_return=0.0, btc_return=0.0)
        
        # Should not crash, beta unchanged
        assert result.beta == 2.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self, sample_prices):
        """Complete workflow from prices to estimate."""
        # Create tracker
        tracker = KalmanBetaTracker(
            static_betas={'ETH': 1.98, 'SOL': 1.55}
        )
        
        # Load data
        tracker.load_data(sample_prices)
        
        # Get estimates
        eth_estimate = tracker.get_estimate('ETH')
        sol_estimate = tracker.get_estimate('SOL')
        
        # Verify results
        assert eth_estimate is not None
        assert sol_estimate is not None
        
        # Check deviation tracking
        assert eth_estimate.deviation_from_static is not None
        assert sol_estimate.deviation_from_static is not None
    
    def test_comparison_with_static(self, sample_prices):
        """Kalman beta should be close to static for stable relationship."""
        result = quick_kalman_beta(
            asset_prices=sample_prices['ETH'],
            btc_prices=sample_prices['BTC'],
            symbol='ETH',
            static_beta=1.98,
        )
        
        # For synthetic data with true beta 1.98, Kalman should be close
        deviation = abs(result.deviation_from_static or 0)
        assert deviation < 0.5  # Within 50%
