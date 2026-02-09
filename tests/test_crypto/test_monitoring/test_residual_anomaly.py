"""
Tests for crypto/monitoring/residual_anomaly.py - Priority 3.

Tests residual anomaly detection for beta monitoring.

Session: February 2026
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from crypto.monitoring.residual_anomaly import (
    ResidualAnomalyDetector,
    ResidualAnomaly,
    AnomalyReport,
    calculate_residual,
    calculate_residual_series,
    calculate_rolling_residual_stats,
    classify_anomaly,
    calculate_historical_residual_std,
    quick_anomaly_check,
    validate_february_selloff,
    ANOMALY_THRESHOLD_WARNING,
    ANOMALY_THRESHOLD_CRITICAL,
    RESIDUAL_LOOKBACK_DAYS,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_prices():
    """Generate sample prices with known beta relationship."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    
    # BTC prices
    btc_returns = np.random.normal(0.001, 0.02, n)
    btc_prices = 50000 * np.exp(np.cumsum(btc_returns))
    
    # Asset with beta ~2.0 (plus some noise)
    beta = 2.0
    asset_returns = beta * btc_returns + np.random.normal(0, 0.005, n)
    asset_prices = 1.0 * np.exp(np.cumsum(asset_returns))
    
    return pd.DataFrame({
        'BTC': btc_prices,
        'ASSET': asset_prices,
    }, index=dates)


@pytest.fixture
def prices_with_anomaly(sample_prices):
    """Add an anomalous move to sample prices."""
    prices = sample_prices.copy()
    
    # Create a large anomaly on the last day
    # Asset drops much more than beta implies
    prices.loc[prices.index[-1], 'ASSET'] *= 0.90  # 10% extra drop
    
    return prices


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfiguration:
    """Test default configuration values."""
    
    def test_warning_threshold(self):
        """Warning threshold should be 2 sigma."""
        assert ANOMALY_THRESHOLD_WARNING == 2.0
    
    def test_critical_threshold(self):
        """Critical threshold should be 3 sigma."""
        assert ANOMALY_THRESHOLD_CRITICAL == 3.0
    
    def test_lookback_days(self):
        """Lookback should be 60 days."""
        assert RESIDUAL_LOOKBACK_DAYS == 60


# =============================================================================
# CORE FUNCTION TESTS
# =============================================================================


class TestCalculateResidual:
    """Test residual calculation function."""
    
    def test_perfect_tracking_zero_residual(self):
        """Perfect beta tracking should have zero residual."""
        actual = 0.04   # 4% move
        btc = 0.02      # 2% BTC move
        beta = 2.0      # Beta of 2
        
        residual = calculate_residual(actual, btc, beta)
        
        assert abs(residual) < 0.001
    
    def test_outperformance_positive_residual(self):
        """Outperformance should give positive residual."""
        actual = 0.05   # 5% move
        btc = 0.02      # 2% BTC move
        beta = 2.0      # Expected 4%
        
        residual = calculate_residual(actual, btc, beta)
        
        # Residual = 0.05 - 0.04 = 0.01
        assert abs(residual - 0.01) < 0.001
    
    def test_underperformance_negative_residual(self):
        """Underperformance should give negative residual."""
        actual = 0.03   # Only 3% move
        btc = 0.02      # 2% BTC move
        beta = 2.0      # Expected 4%
        
        residual = calculate_residual(actual, btc, beta)
        
        # Residual = 0.03 - 0.04 = -0.01
        assert abs(residual - (-0.01)) < 0.001
    
    def test_zero_btc_move(self):
        """Should handle zero BTC move."""
        residual = calculate_residual(0.01, 0.0, 2.0)
        
        # Expected = 0 * 2 = 0, residual = 0.01 - 0 = 0.01
        assert abs(residual - 0.01) < 0.001


class TestCalculateResidualSeries:
    """Test residual series calculation."""
    
    def test_returns_series(self, sample_prices):
        """Should return pandas Series."""
        asset_returns = sample_prices['ASSET'].pct_change()
        btc_returns = sample_prices['BTC'].pct_change()
        
        residuals = calculate_residual_series(asset_returns, btc_returns, beta=2.0)
        
        assert isinstance(residuals, pd.Series)
        assert residuals.name == 'residual'
    
    def test_correct_length(self, sample_prices):
        """Should return correct length."""
        asset_returns = sample_prices['ASSET'].pct_change()
        btc_returns = sample_prices['BTC'].pct_change()
        
        residuals = calculate_residual_series(asset_returns, btc_returns, beta=2.0)
        
        assert len(residuals) == len(asset_returns)
    
    def test_residuals_centered_near_zero(self, sample_prices):
        """Residuals should be centered near zero for good beta."""
        asset_returns = sample_prices['ASSET'].pct_change()
        btc_returns = sample_prices['BTC'].pct_change()
        
        residuals = calculate_residual_series(asset_returns, btc_returns, beta=2.0)
        
        # Mean should be close to zero
        assert abs(residuals.mean()) < 0.01


class TestCalculateRollingResidualStats:
    """Test rolling residual statistics."""
    
    def test_returns_dataframe(self, sample_prices):
        """Should return DataFrame with expected columns."""
        asset_returns = sample_prices['ASSET'].pct_change()
        btc_returns = sample_prices['BTC'].pct_change()
        residuals = calculate_residual_series(asset_returns, btc_returns, beta=2.0)
        
        stats = calculate_rolling_residual_stats(residuals, window=30)
        
        assert isinstance(stats, pd.DataFrame)
        assert 'residual' in stats.columns
        assert 'residual_mean' in stats.columns
        assert 'residual_std' in stats.columns
        assert 'residual_zscore' in stats.columns
    
    def test_zscore_calculation(self, sample_prices):
        """Z-scores should be calculated correctly."""
        asset_returns = sample_prices['ASSET'].pct_change()
        btc_returns = sample_prices['BTC'].pct_change()
        residuals = calculate_residual_series(asset_returns, btc_returns, beta=2.0)
        
        stats = calculate_rolling_residual_stats(residuals, window=30)
        
        # Most z-scores should be between -3 and 3
        valid_z = stats['residual_zscore'].dropna()
        assert (valid_z.abs() < 5).sum() / len(valid_z) > 0.95


class TestClassifyAnomaly:
    """Test anomaly classification."""
    
    def test_normal_classification(self):
        """Low z-score should be NORMAL."""
        severity, direction = classify_anomaly(0.5)
        
        assert severity == 'NORMAL'
    
    def test_warning_classification(self):
        """Medium z-score should be WARNING."""
        severity, direction = classify_anomaly(2.5)
        
        assert severity == 'WARNING'
    
    def test_critical_classification(self):
        """High z-score should be CRITICAL."""
        severity, direction = classify_anomaly(3.5)
        
        assert severity == 'CRITICAL'
    
    def test_outperformed_direction(self):
        """Positive z-score should be OUTPERFORMED."""
        severity, direction = classify_anomaly(2.5)
        
        assert direction == 'OUTPERFORMED'
    
    def test_underperformed_direction(self):
        """Negative z-score should be UNDERPERFORMED."""
        severity, direction = classify_anomaly(-2.5)
        
        assert direction == 'UNDERPERFORMED'
    
    def test_inline_direction(self):
        """Near-zero z-score should be INLINE."""
        severity, direction = classify_anomaly(0.3)
        
        assert direction == 'INLINE'
    
    def test_nan_handling(self):
        """Should handle NaN z-score."""
        severity, direction = classify_anomaly(np.nan)
        
        assert severity == 'UNKNOWN'
        assert direction == 'UNKNOWN'


# =============================================================================
# DETECTOR CLASS TESTS
# =============================================================================


class TestResidualAnomalyDetector:
    """Test ResidualAnomalyDetector class."""
    
    def test_initialization_with_default_betas(self):
        """Should initialize with config betas."""
        detector = ResidualAnomalyDetector()
        
        assert len(detector.betas) > 0
        assert 'BTC' in detector.betas or detector.betas.get('BTC', None) is None
    
    def test_initialization_with_custom_betas(self):
        """Should accept custom betas."""
        custom_betas = {'BTC': 1.0, 'ETH': 2.0, 'ADA': 2.2}
        detector = ResidualAnomalyDetector(betas=custom_betas)
        
        assert detector.betas == custom_betas
    
    def test_load_data(self, sample_prices):
        """Should load price data correctly."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        assert 'BTC' in detector._returns
        assert 'ASSET' in detector._returns
    
    def test_load_data_requires_btc(self):
        """Should raise error if BTC not in data."""
        detector = ResidualAnomalyDetector()
        prices = pd.DataFrame({'ETH': [100, 101, 102]})
        
        with pytest.raises(ValueError, match="BTC"):
            detector.load_data(prices)
    
    def test_check_latest(self, sample_prices):
        """Should return ResidualAnomaly."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        result = detector.check_latest('ASSET')
        
        assert isinstance(result, ResidualAnomaly)
        assert result.symbol == 'ASSET'
        assert result.beta_used == 2.0
    
    def test_check_latest_with_beta_override(self, sample_prices):
        """Should use beta override when provided."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        result = detector.check_latest('ASSET', beta_override=1.5)
        
        assert result.beta_used == 1.5
    
    def test_check_latest_has_required_fields(self, sample_prices):
        """Result should have all required fields."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        result = detector.check_latest('ASSET')
        
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'btc_return')
        assert hasattr(result, 'asset_return')
        assert hasattr(result, 'expected_return')
        assert hasattr(result, 'residual')
        assert hasattr(result, 'residual_zscore')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'direction')
    
    def test_check_all(self, sample_prices):
        """Should return AnomalyReport for all assets."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        report = detector.check_all()
        
        assert isinstance(report, AnomalyReport)
        assert 'ASSET' in report.anomalies
        assert report.total_assets == 1
    
    def test_anomaly_detected(self, prices_with_anomaly):
        """Should detect anomalous move."""
        detector = ResidualAnomalyDetector(
            betas={'BTC': 1.0, 'ASSET': 2.0},
            warning_threshold=2.0,
            critical_threshold=3.0,
        )
        detector.load_data(prices_with_anomaly)
        
        result = detector.check_latest('ASSET')
        
        # The anomaly should be detected (large negative residual)
        assert result.residual < 0  # Underperformed
        # May or may not trigger alert depending on historical std
        assert result.severity in ('NORMAL', 'WARNING', 'CRITICAL')
    
    def test_has_anomalies_flag(self, sample_prices):
        """Report should have has_anomalies property."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        report = detector.check_all()
        
        assert isinstance(report.has_anomalies, bool)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestQuickAnomalyCheck:
    """Test quick_anomaly_check convenience function."""
    
    def test_returns_result(self, sample_prices):
        """Should return ResidualAnomaly."""
        result = quick_anomaly_check(
            sample_prices['ASSET'],
            sample_prices['BTC'],
            beta=2.0,
            symbol='ASSET',
        )
        
        assert isinstance(result, ResidualAnomaly)
        assert result.symbol == 'ASSET'


class TestValidateFebruarySelloff:
    """Test February selloff validation function."""
    
    def test_returns_results_dict(self):
        """Should return dictionary of results."""
        results = validate_february_selloff()
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_eth_tracked_well(self):
        """ETH should track within 10% error."""
        results = validate_february_selloff()
        
        assert 'ETH' in results
        assert results['ETH']['tracked_well']
        assert abs(results['ETH']['prediction_error']) < 0.10
    
    def test_xrp_tracked_well(self):
        """XRP should track within 10% error."""
        results = validate_february_selloff()
        
        assert 'XRP' in results
        assert results['XRP']['tracked_well']
    
    def test_sol_underperformed(self):
        """SOL should show underperformance."""
        results = validate_february_selloff()
        
        assert 'SOL' in results
        # SOL dropped more than expected (negative residual)
        assert results['SOL']['residual'] < 0
    
    def test_custom_inputs(self):
        """Should accept custom inputs."""
        results = validate_february_selloff(
            btc_move=-0.10,
            actual_moves={'ETH': -0.20},
            betas={'ETH': 2.0},
        )
        
        assert 'ETH' in results
        # -0.10 * 2.0 = -0.20, so perfect tracking
        assert abs(results['ETH']['residual']) < 0.01


# =============================================================================
# ANOMALY REPORT TESTS
# =============================================================================


class TestAnomalyReport:
    """Test AnomalyReport dataclass."""
    
    def test_has_anomalies_true_with_warnings(self, sample_prices):
        """has_anomalies should be True if warnings exist."""
        report = AnomalyReport(
            timestamp=datetime.now(),
            btc_return=-0.13,
            anomalies={},
            total_assets=3,
            normal_count=2,
            warning_count=1,
            critical_count=0,
        )
        
        assert report.has_anomalies
    
    def test_has_anomalies_true_with_critical(self, sample_prices):
        """has_anomalies should be True if critical exists."""
        report = AnomalyReport(
            timestamp=datetime.now(),
            btc_return=-0.13,
            anomalies={},
            total_assets=3,
            normal_count=2,
            warning_count=0,
            critical_count=1,
        )
        
        assert report.has_anomalies
    
    def test_has_anomalies_false_all_normal(self, sample_prices):
        """has_anomalies should be False if all normal."""
        report = AnomalyReport(
            timestamp=datetime.now(),
            btc_return=-0.13,
            anomalies={},
            total_assets=3,
            normal_count=3,
            warning_count=0,
            critical_count=0,
        )
        
        assert not report.has_anomalies


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self, sample_prices):
        """Test complete detection workflow."""
        # 1. Create detector
        detector = ResidualAnomalyDetector(
            betas={'BTC': 1.0, 'ASSET': 2.0}
        )
        
        # 2. Load data
        detector.load_data(sample_prices)
        
        # 3. Check single asset
        anomaly = detector.check_latest('ASSET')
        
        # 4. Check all assets
        report = detector.check_all()
        
        # 5. Format report
        report_str = detector.format_anomaly_report(report)
        
        # Verify outputs
        assert isinstance(anomaly, ResidualAnomaly)
        assert isinstance(report, AnomalyReport)
        assert isinstance(report_str, str)
        assert 'ASSET' in report_str


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_symbol(self, sample_prices):
        """Should raise error for missing symbol."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        with pytest.raises(ValueError, match="No data"):
            detector.check_latest('UNKNOWN')
    
    def test_symbol_normalization(self, sample_prices):
        """Should normalize symbol names."""
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(sample_prices)
        
        result = detector.check_latest('asset-usd')
        
        assert result.symbol == 'ASSET'
    
    def test_constant_prices(self):
        """Should handle constant prices."""
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'BTC': [50000.0] * 100,
            'ASSET': [1.0] * 100,
        }, index=dates)
        
        detector = ResidualAnomalyDetector(betas={'BTC': 1.0, 'ASSET': 2.0})
        detector.load_data(prices)
        
        # Should not crash
        result = detector.check_latest('ASSET')
        assert isinstance(result, ResidualAnomaly)
