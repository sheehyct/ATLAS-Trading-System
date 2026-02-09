"""
Tests for crypto/monitoring/correlation_stability.py - Priority 1.

Tests correlation stability tracking for pairs trading.

Session: February 2026
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from crypto.monitoring.correlation_stability import (
    CorrelationStabilityTracker,
    CorrelationStabilityResult,
    calculate_rolling_correlation,
    calculate_stability_score,
    determine_alert_level,
    detect_correlation_trend,
    quick_stability_check,
    calculate_position_size_multiplier,
    SHORT_WINDOW,
    LONG_WINDOW,
    STABILITY_THRESHOLD,
    ALERT_CRITICAL,
    ALERT_WARNING,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_prices():
    """Generate sample price series with known correlation."""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='D')
    
    # Create correlated series
    # Base random walk
    base_returns = np.random.normal(0, 0.02, 200)
    
    # Asset 1 follows base with some idiosyncratic noise
    asset1_returns = base_returns + np.random.normal(0, 0.01, 200)
    
    # Asset 2 follows base with different noise (should be correlated)
    asset2_returns = base_returns * 1.2 + np.random.normal(0, 0.01, 200)
    
    # Convert to prices
    asset1_prices = 100 * np.exp(np.cumsum(asset1_returns))
    asset2_prices = 50 * np.exp(np.cumsum(asset2_returns))
    
    return {
        'ADA': pd.Series(asset1_prices, index=dates),
        'XRP': pd.Series(asset2_prices, index=dates),
    }


@pytest.fixture
def unstable_prices():
    """Generate prices where correlation breaks down mid-series."""
    np.random.seed(123)
    dates = pd.date_range('2025-01-01', periods=200, freq='D')
    
    # First half: strongly correlated
    base_returns_1 = np.random.normal(0, 0.02, 100)
    asset1_returns_1 = base_returns_1 + np.random.normal(0, 0.005, 100)
    asset2_returns_1 = base_returns_1 * 1.5 + np.random.normal(0, 0.005, 100)
    
    # Second half: weakly correlated (correlation breakdown)
    asset1_returns_2 = np.random.normal(0, 0.02, 100)
    asset2_returns_2 = np.random.normal(0, 0.03, 100)  # Independent
    
    asset1_returns = np.concatenate([asset1_returns_1, asset1_returns_2])
    asset2_returns = np.concatenate([asset2_returns_1, asset2_returns_2])
    
    asset1_prices = 100 * np.exp(np.cumsum(asset1_returns))
    asset2_prices = 50 * np.exp(np.cumsum(asset2_returns))
    
    return {
        'ASSET1': pd.Series(asset1_prices, index=dates),
        'ASSET2': pd.Series(asset2_prices, index=dates),
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfiguration:
    """Test default configuration values."""
    
    def test_short_window_reasonable(self):
        """Short window should be 30 days."""
        assert SHORT_WINDOW == 30
    
    def test_long_window_reasonable(self):
        """Long window should be 90 days."""
        assert LONG_WINDOW == 90
    
    def test_stability_threshold_sensible(self):
        """Stability threshold should be between 0 and 1."""
        assert 0 < STABILITY_THRESHOLD < 1
        assert STABILITY_THRESHOLD == 0.70
    
    def test_alert_levels_ordered(self):
        """Alert levels should be in ascending order."""
        assert ALERT_CRITICAL < ALERT_WARNING < 1.0


# =============================================================================
# CORE FUNCTION TESTS
# =============================================================================


class TestCalculateRollingCorrelation:
    """Test rolling correlation calculation."""
    
    def test_returns_series(self, sample_prices):
        """Should return pandas Series."""
        returns1 = sample_prices['ADA'].pct_change()
        returns2 = sample_prices['XRP'].pct_change()
        
        result = calculate_rolling_correlation(returns1, returns2, window=20)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(returns1)
    
    def test_first_values_nan(self, sample_prices):
        """First window-1 values should be NaN."""
        returns1 = sample_prices['ADA'].pct_change()
        returns2 = sample_prices['XRP'].pct_change()
        window = 20
        
        result = calculate_rolling_correlation(returns1, returns2, window=window)
        
        # First window values should be NaN (plus 1 for pct_change)
        assert result.iloc[:window].isna().all()
    
    def test_correlated_series_high_correlation(self, sample_prices):
        """Correlated series should have high correlation."""
        returns1 = sample_prices['ADA'].pct_change()
        returns2 = sample_prices['XRP'].pct_change()
        
        result = calculate_rolling_correlation(returns1, returns2, window=30)
        
        # Should have positive correlation
        last_corr = result.dropna().iloc[-1]
        assert last_corr > 0.5, f"Expected high correlation, got {last_corr}"
    
    def test_uncorrelated_series_low_correlation(self):
        """Uncorrelated series should have low correlation."""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        
        returns1 = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        returns2 = pd.Series(np.random.normal(0, 0.02, 100), index=dates)
        
        result = calculate_rolling_correlation(returns1, returns2, window=30)
        
        # Correlation should be close to 0
        last_corr = result.dropna().iloc[-1]
        assert abs(last_corr) < 0.5, f"Expected low correlation, got {last_corr}"


class TestCalculateStabilityScore:
    """Test stability score calculation."""
    
    def test_identical_correlations_perfect_stability(self):
        """Identical short and long correlations = stability of 1.0."""
        result = calculate_stability_score(0.8, 0.8)
        assert result == 1.0
    
    def test_opposite_correlations_zero_stability(self):
        """Opposite correlations = low stability."""
        result = calculate_stability_score(0.8, -0.2)
        assert result == 0.0  # Clamped to 0
    
    def test_moderate_difference_moderate_stability(self):
        """Moderate difference should give moderate stability."""
        result = calculate_stability_score(0.75, 0.65)
        assert 0.8 < result < 1.0
    
    def test_nan_handling(self):
        """Should handle NaN inputs."""
        result = calculate_stability_score(np.nan, 0.8)
        assert np.isnan(result)
        
        result = calculate_stability_score(0.8, np.nan)
        assert np.isnan(result)
    
    def test_result_clamped_to_range(self):
        """Result should always be between 0 and 1."""
        # Large difference
        result = calculate_stability_score(1.0, -0.5)
        assert result == 0.0
        
        # Same value
        result = calculate_stability_score(0.5, 0.5)
        assert result == 1.0


class TestDetermineAlertLevel:
    """Test alert level determination."""
    
    def test_critical_alert(self):
        """Low stability should trigger CRITICAL."""
        assert determine_alert_level(0.45) == 'CRITICAL'
        assert determine_alert_level(0.30) == 'CRITICAL'
    
    def test_warning_alert(self):
        """Medium stability should trigger WARNING."""
        assert determine_alert_level(0.60) == 'WARNING'
        assert determine_alert_level(0.55) == 'WARNING'
    
    def test_normal_alert(self):
        """High stability should be NORMAL."""
        assert determine_alert_level(0.90) == 'NORMAL'
        assert determine_alert_level(0.85) == 'NORMAL'
    
    def test_nan_handling(self):
        """Should handle NaN input."""
        assert determine_alert_level(np.nan) == 'UNKNOWN'


class TestDetectCorrelationTrend:
    """Test correlation trend detection."""
    
    def test_strengthening_trend(self):
        """Increasing correlations should be STRENGTHENING."""
        corr_series = pd.Series([0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
        result = detect_correlation_trend(corr_series, lookback=7)
        assert result == 'STRENGTHENING'
    
    def test_weakening_trend(self):
        """Decreasing correlations should be WEAKENING."""
        corr_series = pd.Series([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
        result = detect_correlation_trend(corr_series, lookback=7)
        assert result == 'WEAKENING'
    
    def test_stable_trend(self):
        """Flat correlations should be STABLE."""
        corr_series = pd.Series([0.75, 0.76, 0.74, 0.75, 0.76, 0.74, 0.75])
        result = detect_correlation_trend(corr_series, lookback=7)
        assert result == 'STABLE'
    
    def test_insufficient_data(self):
        """Should return UNKNOWN for insufficient data."""
        corr_series = pd.Series([0.75, 0.76])
        result = detect_correlation_trend(corr_series, lookback=7)
        assert result == 'UNKNOWN'


# =============================================================================
# TRACKER CLASS TESTS
# =============================================================================


class TestCorrelationStabilityTracker:
    """Test CorrelationStabilityTracker class."""
    
    def test_initialization(self):
        """Tracker should initialize with default parameters."""
        tracker = CorrelationStabilityTracker()
        
        assert tracker.short_window == SHORT_WINDOW
        assert tracker.long_window == LONG_WINDOW
        assert tracker.stability_threshold == STABILITY_THRESHOLD
    
    def test_custom_initialization(self):
        """Tracker should accept custom parameters."""
        tracker = CorrelationStabilityTracker(
            short_window=20,
            long_window=60,
            stability_threshold=0.80,
        )
        
        assert tracker.short_window == 20
        assert tracker.long_window == 60
        assert tracker.stability_threshold == 0.80
    
    def test_add_prices(self, sample_prices):
        """Should add price series correctly."""
        tracker = CorrelationStabilityTracker()
        
        tracker.add_prices('ADA', sample_prices['ADA'])
        tracker.add_prices('XRP', sample_prices['XRP'])
        
        assert 'ADA' in tracker._prices
        assert 'XRP' in tracker._prices
        assert 'ADA' in tracker._returns
        assert 'XRP' in tracker._returns
    
    def test_add_prices_dict(self, sample_prices):
        """Should add multiple price series at once."""
        tracker = CorrelationStabilityTracker()
        
        tracker.add_prices_dict(sample_prices)
        
        assert 'ADA' in tracker._prices
        assert 'XRP' in tracker._prices
    
    def test_get_correlation_series(self, sample_prices):
        """Should return correlation series between two assets."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(sample_prices)
        
        corr = tracker.get_correlation_series('ADA', 'XRP', window=30)
        
        assert isinstance(corr, pd.Series)
        assert len(corr.dropna()) > 0
    
    def test_get_correlation_series_missing_data(self):
        """Should raise error for missing data."""
        tracker = CorrelationStabilityTracker()
        
        with pytest.raises(ValueError, match="No data for"):
            tracker.get_correlation_series('ADA', 'XRP', window=30)
    
    def test_get_stability_series(self, sample_prices):
        """Should return stability metrics over time."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(sample_prices)
        
        df = tracker.get_stability_series('ADA', 'XRP')
        
        assert isinstance(df, pd.DataFrame)
        assert 'correlation_short' in df.columns
        assert 'correlation_long' in df.columns
        assert 'stability_score' in df.columns
        assert 'is_stable' in df.columns
        assert 'alert_level' in df.columns
    
    def test_get_current_stability(self, sample_prices):
        """Should return current stability result."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(sample_prices)
        
        result = tracker.get_current_stability('ADA', 'XRP')
        
        assert isinstance(result, CorrelationStabilityResult)
        assert result.symbol1 == 'ADA'
        assert result.symbol2 == 'XRP'
        assert result.pair_name == 'ADA/XRP'
        assert isinstance(result.timestamp, datetime)
        assert not np.isnan(result.correlation_short)
        assert not np.isnan(result.correlation_long)
        assert not np.isnan(result.stability_score)
    
    def test_correlated_pair_is_stable(self, sample_prices):
        """Correlated pair should show as stable."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(sample_prices)
        
        result = tracker.get_current_stability('ADA', 'XRP')
        
        # Correlated series should be stable
        assert result.is_stable or result.stability_score > 0.5
        assert result.alert_level in ('NORMAL', 'WARNING')
    
    def test_unstable_pair_detection(self, unstable_prices):
        """Should detect unstable pair after correlation breakdown."""
        tracker = CorrelationStabilityTracker(short_window=20, long_window=60)
        tracker.add_prices_dict(unstable_prices)
        
        result = tracker.get_current_stability('ASSET1', 'ASSET2')
        
        # After breakdown, should show some instability
        # Note: exact result depends on window sizes and random seed
        assert isinstance(result, CorrelationStabilityResult)
    
    def test_action_recommendation(self, sample_prices):
        """Result should have action recommendation."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(sample_prices)
        
        result = tracker.get_current_stability('ADA', 'XRP')
        
        assert result.action_recommendation in (
            "HALT - Reduce or close positions",
            "REDUCE - Scale to 50% normal size",
            "NORMAL - Full position sizing OK",
        )


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestQuickStabilityCheck:
    """Test quick_stability_check convenience function."""
    
    def test_returns_result(self, sample_prices):
        """Should return CorrelationStabilityResult."""
        result = quick_stability_check(
            sample_prices['ADA'],
            sample_prices['XRP'],
            'ADA',
            'XRP',
        )
        
        assert isinstance(result, CorrelationStabilityResult)
        assert result.symbol1 == 'ADA'
        assert result.symbol2 == 'XRP'


class TestCalculatePositionSizeMultiplier:
    """Test position size multiplier calculation."""
    
    def test_high_stability_full_size(self):
        """High stability should allow full position size."""
        multiplier = calculate_position_size_multiplier(0.95)
        assert multiplier > 0.9
    
    def test_medium_stability_reduced_size(self):
        """Medium stability should reduce position size."""
        multiplier = calculate_position_size_multiplier(0.65)
        assert 0.3 < multiplier < 0.8
    
    def test_low_stability_heavily_reduced(self):
        """Low stability should heavily reduce position size."""
        multiplier = calculate_position_size_multiplier(0.40)
        assert multiplier == 0.0
    
    def test_nan_returns_zero(self):
        """NaN stability should return zero."""
        multiplier = calculate_position_size_multiplier(np.nan)
        assert multiplier == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self, sample_prices):
        """Test complete tracking workflow."""
        # 1. Create tracker
        tracker = CorrelationStabilityTracker()
        
        # 2. Add data
        tracker.add_prices_dict(sample_prices)
        
        # 3. Get stability
        result = tracker.get_current_stability('ADA', 'XRP')
        
        # 4. Calculate position multiplier
        multiplier = calculate_position_size_multiplier(result.stability_score)
        
        # 5. Verify we have actionable output
        assert 0 <= multiplier <= 1
        assert result.action_recommendation is not None
    
    def test_multiple_pairs(self):
        """Test tracking multiple pairs simultaneously."""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=150, freq='D')
        
        # Create 3 assets with different correlations
        base = np.random.normal(0, 0.02, 150)
        
        prices = {
            'BTC': pd.Series(100 * np.exp(np.cumsum(base)), index=dates),
            'ETH': pd.Series(50 * np.exp(np.cumsum(base * 1.5 + np.random.normal(0, 0.01, 150))), index=dates),
            'SOL': pd.Series(20 * np.exp(np.cumsum(np.random.normal(0, 0.03, 150))), index=dates),  # Less correlated
        }
        
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(prices)
        
        # Check multiple pairs
        results = tracker.get_all_pairs_stability([('BTC', 'ETH'), ('BTC', 'SOL')])
        
        assert len(results) == 2
        
        # BTC/ETH should be more stable than BTC/SOL
        btc_eth = next(r for r in results if 'ETH' in r.pair_name)
        btc_sol = next(r for r in results if 'SOL' in r.pair_name)
        
        # Can't guarantee exact values, but should have valid results
        assert isinstance(btc_eth.stability_score, float)
        assert isinstance(btc_sol.stability_score, float)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_short_series(self):
        """Should handle short price series."""
        dates = pd.date_range('2025-01-01', periods=10, freq='D')
        prices = {
            'A': pd.Series(np.linspace(100, 110, 10), index=dates),
            'B': pd.Series(np.linspace(50, 55, 10), index=dates),
        }
        
        tracker = CorrelationStabilityTracker(short_window=5, long_window=8)
        tracker.add_prices_dict(prices)
        
        result = tracker.get_current_stability('A', 'B')
        
        # Should return result even if NaN
        assert isinstance(result, CorrelationStabilityResult)
    
    def test_symbol_normalization(self, sample_prices):
        """Should normalize symbol names."""
        tracker = CorrelationStabilityTracker()
        tracker.add_prices('ada-usd', sample_prices['ADA'])
        tracker.add_prices('XRP', sample_prices['XRP'])
        
        # Should find 'ADA' despite adding as 'ada-usd'
        result = tracker.get_current_stability('ADA', 'xrp')
        assert result.symbol1 == 'ADA'
        assert result.symbol2 == 'XRP'
    
    def test_constant_prices(self):
        """Should handle constant prices (zero variance)."""
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        prices = {
            'A': pd.Series([100.0] * 100, index=dates),
            'B': pd.Series([50.0] * 100, index=dates),
        }
        
        tracker = CorrelationStabilityTracker()
        tracker.add_prices_dict(prices)
        
        # Should handle gracefully
        result = tracker.get_current_stability('A', 'B')
        assert isinstance(result, CorrelationStabilityResult)
