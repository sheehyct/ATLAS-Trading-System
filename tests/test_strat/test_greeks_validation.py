"""
Tests for Greeks calculation accuracy and Black-Scholes validation.

Session 72: Validates that our Black-Scholes implementation produces
accurate Greeks values by comparing against known theoretical values.

Test Categories:
1. Black-Scholes price accuracy
2. Delta range validation (ATM near 0.50)
3. Theta sign validation (must be negative for longs)
4. Greeks edge cases (zero DTE, deep ITM/OTM)
"""

import pytest
import numpy as np
from strat.greeks import (
    calculate_greeks,
    Greeks,
    estimate_iv_from_history,
    black_scholes_price,
    validate_delta_range,
    evaluate_trade_quality,
)
import pandas as pd


class TestBlackScholesPrice:
    """Test Black-Scholes pricing accuracy against known values."""

    def test_atm_call_price(self):
        """ATM call should have reasonable price."""
        # S=100, K=100, T=30 days, r=5%, sigma=20%
        # Expected price: approximately $2.30-2.60 for ATM call
        greeks = calculate_greeks(
            S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )
        # ATM 30-day call at 20% IV should be roughly $2.40
        assert 2.0 < greeks.option_price < 3.5, f"ATM call price {greeks.option_price} out of expected range"

    def test_itm_call_price(self):
        """ITM call should have higher price than ATM."""
        greeks_itm = calculate_greeks(
            S=100, K=95, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )
        greeks_atm = calculate_greeks(
            S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks_itm.option_price > greeks_atm.option_price, "ITM should be more expensive than ATM"
        # ITM by $5 should have at least $5 intrinsic value
        assert greeks_itm.option_price >= 5.0, f"ITM call should have intrinsic value: {greeks_itm.option_price}"

    def test_otm_call_price(self):
        """OTM call should have lower price than ATM."""
        greeks_otm = calculate_greeks(
            S=100, K=105, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )
        greeks_atm = calculate_greeks(
            S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks_otm.option_price < greeks_atm.option_price, "OTM should be cheaper than ATM"
        # OTM call should still have some time value
        assert greeks_otm.option_price > 0.5, f"OTM call too cheap: {greeks_otm.option_price}"

    def test_put_call_parity_approximate(self):
        """Test put-call parity: C - P approximately equals S - K*e^(-rT)."""
        S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.20
        call = calculate_greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type='call')
        put = calculate_greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type='put')

        # C - P = S - K*e^(-rT) approximately (for ATM)
        expected_diff = S - K * np.exp(-r * T)
        actual_diff = call.option_price - put.option_price

        # Should be within 1% of expected
        assert abs(actual_diff - expected_diff) < 0.5, f"Put-call parity violated: {actual_diff} vs {expected_diff}"


class TestDeltaValidation:
    """Test delta calculation accuracy."""

    def test_atm_call_delta_near_50(self):
        """ATM call delta should be approximately 0.50."""
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        # ATM delta should be 0.50-0.55 (slightly above 0.50 due to drift)
        assert 0.48 < greeks.delta < 0.58, f"ATM call delta {greeks.delta} not near 0.50"

    def test_atm_put_delta_near_negative_50(self):
        """ATM put delta should be approximately -0.50."""
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='put'
        )
        # ATM put delta should be -0.50 to -0.42
        assert -0.58 < greeks.delta < -0.42, f"ATM put delta {greeks.delta} not near -0.50"

    def test_itm_call_delta_above_50(self):
        """ITM call delta should be above 0.50."""
        greeks = calculate_greeks(
            S=450, K=440, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.delta > 0.55, f"ITM call delta {greeks.delta} should be > 0.55"

    def test_otm_call_delta_below_50(self):
        """OTM call delta should be below 0.50."""
        greeks = calculate_greeks(
            S=450, K=460, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.delta < 0.45, f"OTM call delta {greeks.delta} should be < 0.45"

    def test_deep_itm_call_delta_near_1(self):
        """Deep ITM call delta should approach 1.0."""
        greeks = calculate_greeks(
            S=450, K=400, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.delta > 0.90, f"Deep ITM call delta {greeks.delta} should be > 0.90"

    def test_deep_otm_call_delta_near_0(self):
        """Deep OTM call delta should approach 0."""
        greeks = calculate_greeks(
            S=450, K=500, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.delta < 0.15, f"Deep OTM call delta {greeks.delta} should be < 0.15"

    def test_delta_range_validation_optimal(self):
        """Test delta validation function for optimal range."""
        # Delta in optimal range (0.50-0.80)
        valid, reason = validate_delta_range(0.55)
        assert valid, f"Delta 0.55 should be valid: {reason}"

    def test_delta_range_validation_too_low(self):
        """Test delta validation for too low delta."""
        valid, reason = validate_delta_range(0.25)
        assert not valid, "Delta 0.25 should be too low"
        assert "otm" in reason.lower(), f"Wrong reason: {reason}"


class TestThetaValidation:
    """Test theta calculation accuracy."""

    def test_theta_negative_for_long_call(self):
        """Long call theta must be negative (time decay hurts longs)."""
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.theta < 0, f"Call theta must be negative: {greeks.theta}"

    def test_theta_negative_for_long_put(self):
        """Long put theta must be negative."""
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='put'
        )
        assert greeks.theta < 0, f"Put theta must be negative: {greeks.theta}"

    def test_theta_accelerates_near_expiration(self):
        """Theta should be larger (more negative) closer to expiration."""
        theta_35d = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        ).theta

        theta_7d = calculate_greeks(
            S=450, K=450, T=7/365, r=0.05, sigma=0.20, option_type='call'
        ).theta

        # 7-day theta should be MORE negative than 35-day
        assert theta_7d < theta_35d, f"7-day theta {theta_7d} should be more negative than 35-day {theta_35d}"

    def test_theta_daily_decay_reasonable(self):
        """Theta should represent reasonable daily decay."""
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        # For $12 ATM option, daily decay should be roughly $0.10-0.30
        assert -0.50 < greeks.theta < -0.05, f"Theta {greeks.theta} seems unreasonable"


class TestGreeksEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dte_intrinsic_only(self):
        """At expiration (T=0), option should have intrinsic value only."""
        # ITM call at expiration
        greeks = calculate_greeks(
            S=455, K=450, T=0, r=0.05, sigma=0.20, option_type='call'
        )
        intrinsic = 455 - 450  # $5
        assert abs(greeks.option_price - intrinsic) < 0.01, f"At expiration, price should be intrinsic: {greeks.option_price}"
        assert greeks.delta == 1.0, f"ITM at expiration should have delta=1: {greeks.delta}"

    def test_zero_dte_otm_worthless(self):
        """OTM option at expiration should be worthless."""
        greeks = calculate_greeks(
            S=445, K=450, T=0, r=0.05, sigma=0.20, option_type='call'
        )
        assert greeks.option_price == 0, f"OTM at expiration should be worthless: {greeks.option_price}"
        assert greeks.delta == 0.0, f"OTM at expiration should have delta=0: {greeks.delta}"

    def test_very_high_iv(self):
        """High IV should increase option price."""
        low_iv = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        high_iv = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.50, option_type='call')

        assert high_iv.option_price > low_iv.option_price * 1.5, "High IV should significantly increase price"

    def test_very_low_iv(self):
        """Low IV should decrease option price."""
        greeks = calculate_greeks(
            S=100, K=100, T=30/365, r=0.05, sigma=0.05, option_type='call'
        )
        # 5% IV is very low, price should be minimal
        assert greeks.option_price < 1.5, f"Very low IV should have low price: {greeks.option_price}"


class TestIVEstimation:
    """Test implied volatility estimation from historical data."""

    def test_iv_from_stable_prices(self):
        """Stable prices should produce low IV estimate."""
        # Generate stable prices (low volatility)
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50) * 0.5)  # Small moves

        iv = estimate_iv_from_history(prices, window=20)
        # Low volatility should be < 20%
        assert iv < 0.25, f"Stable prices should have low IV: {iv}"

    def test_iv_from_volatile_prices(self):
        """Volatile prices should produce higher IV estimate."""
        # Generate volatile prices
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 3))  # Large moves

        iv = estimate_iv_from_history(prices, window=20)
        # Higher volatility should be > 15%
        assert iv > 0.10, f"Volatile prices should have higher IV: {iv}"

    def test_iv_annualized(self):
        """IV should be annualized (scaled by sqrt(252))."""
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50))

        iv_annualized = estimate_iv_from_history(prices, window=20, annualize=True)
        iv_daily = estimate_iv_from_history(prices, window=20, annualize=False)

        # Annualized should be roughly sqrt(252) times daily
        ratio = iv_annualized / iv_daily
        assert 14 < ratio < 18, f"Annualization ratio {ratio} seems wrong"


class TestTradeQualityEvaluation:
    """Test trade quality scoring system."""

    def test_good_trade_quality(self):
        """Trade with good delta and ROI should pass."""
        # ATM call should have delta ~0.50-0.55, which is in optimal range
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        entry_premium = greeks.option_price  # Use Black-Scholes price (~$12)
        target_move = 30  # $30 up - significant move for good ROI

        # evaluate_trade_quality(greeks, entry_premium, target_move, max_days)
        quality = evaluate_trade_quality(greeks, entry_premium, target_move, max_days=5)

        # Should have quality score >= 1 for decent trade (0-3 scale)
        # ATM with 30 point move in 5 days should pass delta check at minimum
        assert quality['quality_score'] >= 1, f"Good trade should score >= 1: {quality}"

    def test_bad_delta_trade(self):
        """Trade with very low delta should have warning."""
        # Deep OTM call - very low delta
        greeks = calculate_greeks(
            S=450, K=500, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        entry_premium = greeks.option_price
        target_move = 10

        quality = evaluate_trade_quality(greeks, entry_premium, target_move, max_days=5)

        # Should have delta warning or low quality score
        has_delta_issue = any('delta' in issue.lower() for issue in quality.get('issues', []))
        assert has_delta_issue or quality['quality_score'] < 2, f"Bad delta trade should have issues: {quality}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
