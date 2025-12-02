"""
Session 82: Comprehensive Pricing Accuracy Test Suite

Tests for validating ThetaData vs Black-Scholes pricing accuracy.
These tests validate the accuracy metrics module and integration points.

Test Categories:
1. PricingAccuracyMetrics unit tests
2. AccuracyResults validation tests
3. Integration with ThetaDataOptionsFetcher
4. Threshold validation tests
"""

import pytest
import numpy as np
from datetime import datetime

from tests.metrics.pricing_accuracy import (
    PricingAccuracyMetrics,
    AccuracyResults,
    PricingComparison,
    compare_prices,
    validate_pricing_accuracy,
)


class TestPricingComparisonDataclass:
    """Test PricingComparison dataclass behavior."""

    def test_basic_comparison_creation(self):
        """Test creating a basic pricing comparison."""
        comp = PricingComparison(
            real_price=5.25,
            synthetic_price=5.10,
            underlying_price=450.0,
            strike=455.0,
            dte=30,
            option_type='call',
        )

        assert comp.real_price == 5.25
        assert comp.synthetic_price == 5.10
        assert comp.underlying_price == 450.0

    def test_moneyness_calculation_call_itm(self):
        """Test moneyness for ITM call."""
        comp = PricingComparison(
            real_price=15.0,
            synthetic_price=14.5,
            underlying_price=465.0,  # 3.3% above strike
            strike=450.0,
            dte=30,
            option_type='call',
        )

        assert comp.moneyness == 'itm'

    def test_moneyness_calculation_call_otm(self):
        """Test moneyness for OTM call."""
        comp = PricingComparison(
            real_price=2.0,
            synthetic_price=2.1,
            underlying_price=440.0,  # 2.2% below strike
            strike=450.0,
            dte=30,
            option_type='call',
        )

        assert comp.moneyness == 'otm'

    def test_moneyness_calculation_put_itm(self):
        """Test moneyness for ITM put."""
        comp = PricingComparison(
            real_price=12.0,
            synthetic_price=11.8,
            underlying_price=440.0,  # Below strike - ITM for put
            strike=450.0,
            dte=30,
            option_type='put',
        )

        assert comp.moneyness == 'itm'

    def test_moneyness_calculation_atm(self):
        """Test moneyness for ATM option."""
        comp = PricingComparison(
            real_price=8.0,
            synthetic_price=7.9,
            underlying_price=450.0,  # At the money
            strike=450.0,
            dte=30,
            option_type='call',
        )

        assert comp.moneyness == 'atm'


class TestPricingAccuracyMetrics:
    """Test PricingAccuracyMetrics class."""

    def test_empty_metrics(self):
        """Test metrics with no comparisons."""
        metrics = PricingAccuracyMetrics()
        results = metrics.calculate()

        assert results.n_samples == 0
        assert results.mae == float('inf')
        assert results.mape == 100.0

    def test_single_comparison(self):
        """Test metrics with a single comparison."""
        metrics = PricingAccuracyMetrics()
        metrics.add_comparison(
            real_price=5.0,
            synthetic_price=5.0,
            option_type='call',
        )

        results = metrics.calculate()

        assert results.n_samples == 1
        assert results.mae == 0.0
        assert results.mape == 0.0

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        metrics = PricingAccuracyMetrics()

        # Add known comparisons
        metrics.add_comparison(real_price=5.0, synthetic_price=5.5)  # error = 0.5
        metrics.add_comparison(real_price=3.0, synthetic_price=2.5)  # error = -0.5
        metrics.add_comparison(real_price=4.0, synthetic_price=4.2)  # error = 0.2

        results = metrics.calculate()

        # MAE = (0.5 + 0.5 + 0.2) / 3 = 0.4
        assert results.mae == pytest.approx(0.4, abs=0.01)

    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(real_price=5.0, synthetic_price=6.0)  # error = 1.0
        metrics.add_comparison(real_price=3.0, synthetic_price=3.0)  # error = 0.0

        results = metrics.calculate()

        # RMSE = sqrt((1.0^2 + 0^2) / 2) = sqrt(0.5) ≈ 0.707
        assert results.rmse == pytest.approx(0.707, abs=0.01)

    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(real_price=10.0, synthetic_price=11.0)  # 10% error
        metrics.add_comparison(real_price=10.0, synthetic_price=10.0)  # 0% error

        results = metrics.calculate()

        # MAPE = (10% + 0%) / 2 = 5%
        assert results.mape == pytest.approx(5.0, abs=0.1)

    def test_correlation_calculation(self):
        """Test correlation calculation."""
        metrics = PricingAccuracyMetrics()

        # Add perfectly correlated prices
        metrics.add_comparison(real_price=5.0, synthetic_price=5.5)
        metrics.add_comparison(real_price=6.0, synthetic_price=6.5)
        metrics.add_comparison(real_price=7.0, synthetic_price=7.5)

        results = metrics.calculate()

        # Perfect linear relationship should give correlation ~1.0
        assert results.correlation == pytest.approx(1.0, abs=0.01)

    def test_bias_calculation(self):
        """Test systematic bias calculation."""
        metrics = PricingAccuracyMetrics()

        # Synthetic consistently over-prices
        metrics.add_comparison(real_price=5.0, synthetic_price=5.2)
        metrics.add_comparison(real_price=6.0, synthetic_price=6.3)
        metrics.add_comparison(real_price=7.0, synthetic_price=7.1)

        results = metrics.calculate()

        # Positive bias = synthetic over-pricing
        assert results.bias > 0

    def test_call_put_breakdown(self):
        """Test call/put MAE breakdown."""
        metrics = PricingAccuracyMetrics()

        # Calls with small errors
        metrics.add_comparison(real_price=5.0, synthetic_price=5.1, option_type='call')
        metrics.add_comparison(real_price=6.0, synthetic_price=6.1, option_type='call')

        # Puts with larger errors
        metrics.add_comparison(real_price=4.0, synthetic_price=4.5, option_type='put')
        metrics.add_comparison(real_price=3.0, synthetic_price=3.6, option_type='put')

        results = metrics.calculate()

        assert results.call_mae < results.put_mae

    def test_invalid_prices_skipped(self):
        """Test that invalid prices are skipped."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(real_price=5.0, synthetic_price=5.1)
        metrics.add_comparison(real_price=-1.0, synthetic_price=5.0)  # Invalid
        metrics.add_comparison(real_price=5.0, synthetic_price=0.0)  # Invalid

        results = metrics.calculate()

        assert results.n_samples == 1

    def test_clear_comparisons(self):
        """Test clearing comparisons."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(real_price=5.0, synthetic_price=5.1)
        metrics.add_comparison(real_price=6.0, synthetic_price=6.1)

        metrics.clear()

        assert len(metrics.comparisons) == 0


class TestAccuracyResults:
    """Test AccuracyResults methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = AccuracyResults(
            mae=0.5,
            rmse=0.7,
            mape=10.0,
            correlation=0.95,
            bias=0.1,
            n_samples=100,
            real_mean=5.0,
            synthetic_mean=5.1,
        )

        d = results.to_dict()

        assert d['mae'] == 0.5
        assert d['correlation'] == 0.95
        assert d['n_samples'] == 100

    def test_to_json(self):
        """Test JSON serialization."""
        results = AccuracyResults(
            mae=0.5,
            rmse=0.7,
            mape=10.0,
            correlation=0.95,
            bias=0.1,
            n_samples=100,
            real_mean=5.0,
            synthetic_mean=5.1,
        )

        json_str = results.to_json()

        assert '"mae": 0.5' in json_str
        assert '"n_samples": 100' in json_str

    def test_passes_threshold_success(self):
        """Test threshold check passing."""
        results = AccuracyResults(
            mae=0.5,
            rmse=0.7,
            mape=10.0,  # Below 15%
            correlation=0.95,  # Above 0.9
            bias=0.1,
            n_samples=100,
            real_mean=5.0,
            synthetic_mean=5.1,
        )

        assert results.passes_threshold(max_mape=15.0, min_correlation=0.9)

    def test_passes_threshold_failure_mape(self):
        """Test threshold check failing on MAPE."""
        results = AccuracyResults(
            mae=1.0,
            rmse=1.5,
            mape=20.0,  # Above 15%
            correlation=0.95,
            bias=0.2,
            n_samples=100,
            real_mean=5.0,
            synthetic_mean=5.2,
        )

        assert not results.passes_threshold(max_mape=15.0, min_correlation=0.9)

    def test_passes_threshold_failure_correlation(self):
        """Test threshold check failing on correlation."""
        results = AccuracyResults(
            mae=0.5,
            rmse=0.7,
            mape=10.0,
            correlation=0.8,  # Below 0.9
            bias=0.1,
            n_samples=100,
            real_mean=5.0,
            synthetic_mean=5.1,
        )

        assert not results.passes_threshold(max_mape=15.0, min_correlation=0.9)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_compare_prices_basic(self):
        """Test quick price comparison."""
        real = [5.0, 6.0, 7.0, 8.0]
        synthetic = [5.1, 5.9, 7.2, 7.9]

        results = compare_prices(real, synthetic)

        assert results.n_samples == 4
        assert results.mae > 0

    def test_validate_pricing_accuracy_pass(self):
        """Test validation passing."""
        results = AccuracyResults(
            mae=0.3,
            rmse=0.4,
            mape=8.0,
            correlation=0.96,
            bias=0.05,
            n_samples=50,
            real_mean=5.0,
            synthetic_mean=5.05,
        )

        passed, msg = validate_pricing_accuracy(results, max_mape=15.0)

        assert passed
        assert 'Passed' in msg

    def test_validate_pricing_accuracy_fail_samples(self):
        """Test validation failing on sample count."""
        results = AccuracyResults(
            mae=0.3,
            rmse=0.4,
            mape=8.0,
            correlation=0.96,
            bias=0.05,
            n_samples=5,  # Below min_samples=10
            real_mean=5.0,
            synthetic_mean=5.05,
        )

        passed, msg = validate_pricing_accuracy(results, min_samples=10)

        assert not passed
        assert 'Insufficient samples' in msg


class TestThresholdAnalysis:
    """Test threshold analysis features."""

    def test_within_threshold_percentages(self):
        """Test percentage within thresholds."""
        metrics = PricingAccuracyMetrics()

        # 50% within 1%, 75% within 5%, 100% within 10%
        metrics.add_comparison(real_price=100.0, synthetic_price=100.5)  # 0.5%
        metrics.add_comparison(real_price=100.0, synthetic_price=102.0)  # 2%
        metrics.add_comparison(real_price=100.0, synthetic_price=104.0)  # 4%
        metrics.add_comparison(real_price=100.0, synthetic_price=108.0)  # 8%

        results = metrics.calculate()

        assert results.within_1pct == pytest.approx(25.0, abs=1.0)  # 1 of 4
        assert results.within_5pct == pytest.approx(75.0, abs=1.0)  # 3 of 4
        assert results.within_10pct == pytest.approx(100.0, abs=1.0)  # 4 of 4


class TestSummaryReport:
    """Test summary report generation."""

    def test_summary_contains_all_sections(self):
        """Test that summary contains all expected sections."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(
            real_price=5.0,
            synthetic_price=5.1,
            option_type='call',
            real_iv=0.15,
            synthetic_iv=0.20,
        )
        metrics.add_comparison(
            real_price=4.0,
            synthetic_price=4.2,
            option_type='put',
            real_iv=0.16,
            synthetic_iv=0.20,
        )

        summary = metrics.summary()

        assert 'PRICING ACCURACY SUMMARY' in summary
        assert 'Core Metrics' in summary
        assert 'MAE' in summary
        assert 'MAPE' in summary
        assert 'Correlation' in summary
        assert 'Threshold Analysis' in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_identical_prices(self):
        """Test when all prices are identical."""
        metrics = PricingAccuracyMetrics()

        for _ in range(10):
            metrics.add_comparison(real_price=5.0, synthetic_price=5.0)

        results = metrics.calculate()

        assert results.mae == 0.0
        assert results.mape == 0.0
        assert results.bias == 0.0

    def test_single_unique_value(self):
        """Test with single unique value (correlation undefined)."""
        metrics = PricingAccuracyMetrics()
        metrics.add_comparison(real_price=5.0, synthetic_price=5.0)

        results = metrics.calculate()

        # Correlation with single point should be 0 (undefined)
        assert results.correlation == 0.0

    def test_very_small_prices(self):
        """Test with very small prices."""
        metrics = PricingAccuracyMetrics()

        metrics.add_comparison(real_price=0.05, synthetic_price=0.06)
        metrics.add_comparison(real_price=0.10, synthetic_price=0.12)

        results = metrics.calculate()

        assert results.n_samples == 2
        assert results.mae > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
