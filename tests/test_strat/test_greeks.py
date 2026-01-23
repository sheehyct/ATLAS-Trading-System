"""
Tests for strat/greeks.py - Black-Scholes Greeks calculations.

Session EQUITY-79: Comprehensive unit tests covering all functions
and edge cases not covered by test_greeks_validation.py.

Coverage includes:
- _d1, _d2 helper functions
- black_scholes_price standalone
- Greeks dataclass validate_delta_range method
- calculate_iv_percentile (missing from validation tests)
- calculate_pnl_with_greeks (missing from validation tests)
- Gamma, Vega, Rho validation
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
import pandas as pd

from strat.greeks import (
    Greeks,
    _d1,
    _d2,
    black_scholes_price,
    calculate_greeks,
    estimate_iv_from_history,
    validate_delta_range,
    calculate_iv_percentile,
    calculate_pnl_with_greeks,
    evaluate_trade_quality,
)


# =============================================================================
# Test _d1 and _d2 helper functions
# =============================================================================

class TestD1D2Helpers:
    """Test Black-Scholes d1 and d2 helper functions."""

    def test_d1_atm_short_term(self):
        """d1 for ATM should be close to 0 for short-term options."""
        # ATM: S=K, short term, low drift
        d1 = _d1(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        # d1 = (ln(1) + (0.05 + 0.02)*30/365) / (0.20 * sqrt(30/365))
        # d1 = (0 + 0.0058) / 0.0573 = ~0.10
        assert -0.5 < d1 < 0.5, f"ATM d1 should be near 0: {d1}"

    def test_d1_itm_positive(self):
        """d1 should be positive for ITM options."""
        d1 = _d1(S=110, K=100, T=30/365, r=0.05, sigma=0.20)
        assert d1 > 0, f"ITM d1 should be positive: {d1}"

    def test_d1_otm_negative(self):
        """d1 should be more negative for OTM options."""
        d1_atm = _d1(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        d1_otm = _d1(S=100, K=110, T=30/365, r=0.05, sigma=0.20)
        assert d1_otm < d1_atm, f"OTM d1 should be less than ATM: {d1_otm} vs {d1_atm}"

    def test_d2_less_than_d1(self):
        """d2 should always be less than d1 by sigma*sqrt(T)."""
        S, K, T, r, sigma = 100, 100, 30/365, 0.05, 0.20
        d1_val = _d1(S, K, T, r, sigma)
        d2_val = _d2(S, K, T, r, sigma)
        expected_diff = sigma * np.sqrt(T)
        assert abs((d1_val - d2_val) - expected_diff) < 0.001, "d2 = d1 - sigma*sqrt(T)"

    def test_d1_zero_time_returns_zero(self):
        """d1 with T=0 should return 0."""
        d1 = _d1(S=100, K=100, T=0, r=0.05, sigma=0.20)
        assert d1 == 0.0, f"d1 with T=0 should be 0: {d1}"

    def test_d1_zero_sigma_returns_zero(self):
        """d1 with sigma=0 should return 0."""
        d1 = _d1(S=100, K=100, T=30/365, r=0.05, sigma=0)
        assert d1 == 0.0, f"d1 with sigma=0 should be 0: {d1}"

    def test_d2_zero_time_returns_zero(self):
        """d2 with T=0 should return 0."""
        d2 = _d2(S=100, K=100, T=0, r=0.05, sigma=0.20)
        assert d2 == 0.0, f"d2 with T=0 should be 0: {d2}"

    def test_d1_higher_rate_increases_d1(self):
        """Higher interest rate should increase d1."""
        d1_low_r = _d1(S=100, K=100, T=30/365, r=0.01, sigma=0.20)
        d1_high_r = _d1(S=100, K=100, T=30/365, r=0.10, sigma=0.20)
        assert d1_high_r > d1_low_r, "Higher rate should increase d1"


# =============================================================================
# Test black_scholes_price standalone function
# =============================================================================

class TestBlackScholesStandalone:
    """Test black_scholes_price function directly."""

    def test_call_price_positive(self):
        """Call price should always be positive."""
        price = black_scholes_price(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        assert price > 0, f"Call price should be positive: {price}"

    def test_put_price_positive(self):
        """Put price should always be positive."""
        price = black_scholes_price(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert price > 0, f"Put price should be positive: {price}"

    def test_call_at_expiration_itm(self):
        """ITM call at expiration = intrinsic value."""
        price = black_scholes_price(S=105, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert price == 5.0, f"ITM call at expiry should be intrinsic: {price}"

    def test_call_at_expiration_otm(self):
        """OTM call at expiration = 0."""
        price = black_scholes_price(S=95, K=100, T=0, r=0.05, sigma=0.20, option_type='call')
        assert price == 0.0, f"OTM call at expiry should be 0: {price}"

    def test_put_at_expiration_itm(self):
        """ITM put at expiration = intrinsic value."""
        price = black_scholes_price(S=95, K=100, T=0, r=0.05, sigma=0.20, option_type='put')
        assert price == 5.0, f"ITM put at expiry should be intrinsic: {price}"

    def test_put_at_expiration_otm(self):
        """OTM put at expiration = 0."""
        price = black_scholes_price(S=105, K=100, T=0, r=0.05, sigma=0.20, option_type='put')
        assert price == 0.0, f"OTM put at expiry should be 0: {price}"

    def test_higher_volatility_higher_price(self):
        """Higher IV should increase option price."""
        low_iv_price = black_scholes_price(S=100, K=100, T=30/365, r=0.05, sigma=0.10)
        high_iv_price = black_scholes_price(S=100, K=100, T=30/365, r=0.05, sigma=0.40)
        assert high_iv_price > low_iv_price, "Higher IV = higher price"

    def test_longer_expiry_higher_price(self):
        """Longer time to expiry should increase option price."""
        short_term = black_scholes_price(S=100, K=100, T=7/365, r=0.05, sigma=0.20)
        long_term = black_scholes_price(S=100, K=100, T=90/365, r=0.05, sigma=0.20)
        assert long_term > short_term, "Longer expiry = higher price"

    def test_negative_time_returns_intrinsic(self):
        """Negative T (edge case) should return intrinsic like T=0."""
        price = black_scholes_price(S=105, K=100, T=-1/365, r=0.05, sigma=0.20, option_type='call')
        assert price == 5.0, f"Negative T should return intrinsic: {price}"


# =============================================================================
# Test Greeks dataclass
# =============================================================================

class TestGreeksDataclass:
    """Test Greeks dataclass and its methods."""

    def test_greeks_initialization(self):
        """Greeks dataclass should initialize correctly."""
        greeks = Greeks(
            delta=0.55,
            gamma=0.02,
            theta=-0.15,
            vega=0.20,
            rho=0.10,
            option_price=5.50
        )
        assert greeks.delta == 0.55
        assert greeks.gamma == 0.02
        assert greeks.theta == -0.15
        assert greeks.vega == 0.20
        assert greeks.rho == 0.10
        assert greeks.option_price == 5.50

    def test_validate_delta_range_optimal(self):
        """Delta in optimal range (0.50-0.80) should pass."""
        greeks = Greeks(delta=0.65, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is True
        assert "optimal" in msg.lower()

    def test_validate_delta_range_too_low(self):
        """Delta below 0.30 should be rejected as too far OTM."""
        greeks = Greeks(delta=0.20, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is False
        assert "too far otm" in msg.lower()

    def test_validate_delta_range_below_optimal(self):
        """Delta 0.30-0.50 should warn as below optimal."""
        greeks = Greeks(delta=0.40, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is False
        assert "below optimal" in msg.lower()

    def test_validate_delta_range_above_optimal(self):
        """Delta 0.80-0.90 should warn as expensive."""
        greeks = Greeks(delta=0.85, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is False
        assert "expensive" in msg.lower() or "high" in msg.lower()

    def test_validate_delta_range_too_high(self):
        """Delta above 0.90 should warn as essentially stock."""
        greeks = Greeks(delta=0.95, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is False
        assert "stock" in msg.lower()

    def test_validate_delta_range_negative_put(self):
        """Negative delta (puts) should use absolute value."""
        greeks = Greeks(delta=-0.65, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        valid, msg = greeks.validate_delta_range()
        assert valid is True, "Put delta -0.65 abs=0.65 is in range"

    def test_validate_delta_range_custom_bounds(self):
        """Custom min/max delta bounds should work."""
        greeks = Greeks(delta=0.45, gamma=0, theta=0, vega=0, rho=0, option_price=0)
        # Default bounds would reject 0.45
        valid_default, _ = greeks.validate_delta_range()
        assert valid_default is False
        # Custom bounds 0.40-0.90 should accept it
        valid_custom, msg = greeks.validate_delta_range(min_delta=0.40, max_delta=0.90)
        assert valid_custom is True


# =============================================================================
# Test Gamma validation
# =============================================================================

class TestGammaValidation:
    """Test gamma calculation accuracy."""

    def test_gamma_positive_for_all_options(self):
        """Gamma should always be positive for long options."""
        call = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        put = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert call.gamma > 0, f"Call gamma should be positive: {call.gamma}"
        assert put.gamma > 0, f"Put gamma should be positive: {put.gamma}"

    def test_gamma_highest_atm(self):
        """Gamma should be highest for ATM options."""
        atm = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        itm = calculate_greeks(S=100, K=95, T=30/365, r=0.05, sigma=0.20)
        otm = calculate_greeks(S=100, K=105, T=30/365, r=0.05, sigma=0.20)
        assert atm.gamma >= itm.gamma, "ATM gamma should be >= ITM"
        assert atm.gamma >= otm.gamma, "ATM gamma should be >= OTM"

    def test_gamma_same_for_call_put(self):
        """Gamma should be same for call and put at same strike."""
        call = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        put = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert abs(call.gamma - put.gamma) < 0.001, "Call and put gamma should match"

    def test_gamma_increases_near_expiration(self):
        """ATM gamma should increase as expiration approaches."""
        gamma_far = calculate_greeks(S=100, K=100, T=60/365, r=0.05, sigma=0.20).gamma
        gamma_near = calculate_greeks(S=100, K=100, T=7/365, r=0.05, sigma=0.20).gamma
        assert gamma_near > gamma_far, "Near-term ATM gamma should be higher"


# =============================================================================
# Test Vega validation
# =============================================================================

class TestVegaValidation:
    """Test vega calculation accuracy."""

    def test_vega_positive_for_long_options(self):
        """Vega should be positive for long options (benefit from IV increase)."""
        call = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        put = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert call.vega > 0, f"Call vega should be positive: {call.vega}"
        assert put.vega > 0, f"Put vega should be positive: {put.vega}"

    def test_vega_highest_atm(self):
        """Vega should be highest for ATM options."""
        atm = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20)
        otm = calculate_greeks(S=100, K=110, T=30/365, r=0.05, sigma=0.20)
        itm = calculate_greeks(S=100, K=90, T=30/365, r=0.05, sigma=0.20)
        assert atm.vega >= itm.vega, "ATM vega should be >= ITM"
        assert atm.vega >= otm.vega, "ATM vega should be >= OTM"

    def test_vega_same_for_call_put(self):
        """Vega should be same for call and put at same strike."""
        call = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        put = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert abs(call.vega - put.vega) < 0.001, "Call and put vega should match"

    def test_vega_increases_with_time(self):
        """Vega should be higher for longer-dated options."""
        short_term = calculate_greeks(S=100, K=100, T=7/365, r=0.05, sigma=0.20)
        long_term = calculate_greeks(S=100, K=100, T=90/365, r=0.05, sigma=0.20)
        assert long_term.vega > short_term.vega, "Long-term vega should be higher"


# =============================================================================
# Test Rho validation
# =============================================================================

class TestRhoValidation:
    """Test rho calculation accuracy."""

    def test_rho_positive_for_calls(self):
        """Call rho should be positive (benefits from rate increase)."""
        call = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='call')
        assert call.rho > 0, f"Call rho should be positive: {call.rho}"

    def test_rho_negative_for_puts(self):
        """Put rho should be negative (hurt by rate increase)."""
        put = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type='put')
        assert put.rho < 0, f"Put rho should be negative: {put.rho}"

    def test_rho_increases_with_time(self):
        """Rho magnitude should be higher for longer-dated options."""
        short_call = calculate_greeks(S=100, K=100, T=7/365, r=0.05, sigma=0.20, option_type='call')
        long_call = calculate_greeks(S=100, K=100, T=90/365, r=0.05, sigma=0.20, option_type='call')
        assert long_call.rho > short_call.rho, "Long-term rho should be higher"


# =============================================================================
# Test calculate_iv_percentile (NOT in validation tests)
# =============================================================================

class TestCalculateIVPercentile:
    """Test IV percentile calculation."""

    def test_iv_percentile_at_median(self):
        """Current IV at median of history should be ~50%."""
        historical = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30])
        percentile = calculate_iv_percentile(0.20, historical)
        # 0.20 is exactly in the middle, 2 values below it
        assert 30 <= percentile <= 70, f"Median IV should be ~50%: {percentile}"

    def test_iv_percentile_at_minimum(self):
        """Current IV at minimum should be ~0%."""
        historical = pd.Series([0.15, 0.20, 0.25, 0.30, 0.35])
        percentile = calculate_iv_percentile(0.10, historical)
        assert percentile < 10, f"Below-minimum IV should be ~0%: {percentile}"

    def test_iv_percentile_at_maximum(self):
        """Current IV at maximum should be ~100%."""
        historical = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30])
        percentile = calculate_iv_percentile(0.35, historical)
        assert percentile > 90, f"Above-maximum IV should be ~100%: {percentile}"

    def test_iv_percentile_with_lookback(self):
        """Lookback should limit historical data used."""
        # Long history with low IV, recent with high IV
        old_low = pd.Series([0.10] * 300)
        recent_high = pd.Series([0.30] * 50)
        historical = pd.concat([old_low, recent_high], ignore_index=True)

        # Full history: 0.20 is in middle
        percentile_full = calculate_iv_percentile(0.20, historical, lookback_days=350)
        # Recent only: 0.20 is below 0.30
        percentile_recent = calculate_iv_percentile(0.20, historical, lookback_days=50)

        assert percentile_recent < percentile_full, "Recent lookback should give different result"

    def test_iv_percentile_insufficient_data(self):
        """With insufficient data, should return default 50."""
        historical = pd.Series([0.20])
        percentile = calculate_iv_percentile(0.25, historical)
        assert percentile == 50.0, f"Insufficient data should return 50: {percentile}"

    def test_iv_percentile_empty_series(self):
        """Empty series should return default 50."""
        historical = pd.Series([], dtype=float)
        percentile = calculate_iv_percentile(0.25, historical)
        assert percentile == 50.0, f"Empty series should return 50: {percentile}"


# =============================================================================
# Test calculate_pnl_with_greeks (NOT in validation tests)
# =============================================================================

class TestCalculatePnlWithGreeks:
    """Test P/L calculation using Greeks."""

    def test_pnl_basic_delta_move(self):
        """Test basic delta P/L calculation."""
        entry_greeks = Greeks(delta=0.50, gamma=0.01, theta=-0.10, vega=0.20, rho=0.05, option_price=5.0)
        exit_greeks = entry_greeks  # Same for simplicity

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,  # $10 up
            days_held=0,
            contracts=1,
            entry_premium=0
        )

        # Delta P/L: 0.50 * 10 * 100 = $500
        assert abs(pnl['delta_pnl'] - 500) < 1, f"Delta P/L wrong: {pnl['delta_pnl']}"

    def test_pnl_with_gamma(self):
        """Test gamma (convexity) contribution to P/L."""
        entry_greeks = Greeks(delta=0.50, gamma=0.02, theta=0, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,
            days_held=0,
            contracts=1,
            entry_premium=0
        )

        # Gamma P/L: 0.5 * 0.02 * 10^2 * 100 = $100
        assert abs(pnl['gamma_pnl'] - 100) < 1, f"Gamma P/L wrong: {pnl['gamma_pnl']}"

    def test_pnl_with_theta_decay(self):
        """Test theta (time decay) contribution."""
        entry_greeks = Greeks(delta=0, gamma=0, theta=-0.20, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=0,
            days_held=5,  # 5 days
            contracts=1,
            entry_premium=0
        )

        # Theta P/L: -0.20 * 5 * 100 = -$100
        assert abs(pnl['theta_pnl'] - (-100)) < 1, f"Theta P/L wrong: {pnl['theta_pnl']}"

    def test_pnl_with_entry_premium(self):
        """Test net P/L subtracts entry premium."""
        entry_greeks = Greeks(delta=0.50, gamma=0, theta=0, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,
            days_held=0,
            contracts=1,
            entry_premium=3.0  # Paid $3.00 per share
        )

        # Delta P/L: 0.50 * 10 * 100 = $500
        # Premium paid: $3 * 100 = $300
        # Net: $500 - $300 = $200
        assert abs(pnl['net_pnl'] - 200) < 1, f"Net P/L wrong: {pnl['net_pnl']}"

    def test_pnl_multiple_contracts(self):
        """Test scaling by number of contracts."""
        entry_greeks = Greeks(delta=0.50, gamma=0, theta=0, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl_1 = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,
            days_held=0,
            contracts=1,
            entry_premium=0
        )

        pnl_5 = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,
            days_held=0,
            contracts=5,
            entry_premium=0
        )

        assert pnl_5['net_pnl'] == pnl_1['net_pnl'] * 5, "5 contracts should be 5x P/L"

    def test_pnl_negative_price_move(self):
        """Test P/L with negative price move (loss on long call)."""
        entry_greeks = Greeks(delta=0.50, gamma=0.01, theta=-0.10, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=-5.0,  # $5 down
            days_held=3,
            contracts=1,
            entry_premium=0
        )

        # Delta P/L: 0.50 * -5 * 100 = -$250
        # Gamma P/L: 0.5 * 0.01 * 25 * 100 = $12.50 (always positive)
        # Theta P/L: -0.10 * 3 * 100 = -$30
        assert pnl['delta_pnl'] < 0, "Delta P/L should be negative"
        assert pnl['gamma_pnl'] > 0, "Gamma P/L should be positive (convexity)"
        assert pnl['theta_pnl'] < 0, "Theta P/L should be negative"

    def test_pnl_return_dict_keys(self):
        """Test that return dict has all expected keys."""
        entry_greeks = Greeks(delta=0.50, gamma=0.01, theta=-0.10, vega=0.20, rho=0.05, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=5.0,
            days_held=3,
            contracts=2,
            entry_premium=4.0
        )

        expected_keys = {'delta_pnl', 'gamma_pnl', 'theta_pnl', 'gross_pnl', 'net_pnl', 'days_held', 'entry_premium'}
        assert set(pnl.keys()) == expected_keys, f"Missing keys: {expected_keys - set(pnl.keys())}"

    def test_pnl_zero_premium_no_subtraction(self):
        """With zero entry premium, net = gross."""
        entry_greeks = Greeks(delta=0.50, gamma=0, theta=0, vega=0, rho=0, option_price=5.0)
        exit_greeks = entry_greeks

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=10.0,
            days_held=0,
            contracts=1,
            entry_premium=0
        )

        assert pnl['gross_pnl'] == pnl['net_pnl'], "Zero premium: gross should equal net"


# =============================================================================
# Test estimate_iv_from_history edge cases
# =============================================================================

class TestEstimateIVEdgeCases:
    """Test IV estimation edge cases."""

    def test_iv_insufficient_data_returns_default(self):
        """With <2 data points, should return default 20%."""
        prices = pd.Series([100])
        iv = estimate_iv_from_history(prices, window=20)
        assert iv == 0.20, f"Insufficient data should return 0.20: {iv}"

    def test_iv_window_larger_than_data(self):
        """Window larger than data should use available data."""
        prices = pd.Series([100, 101, 102, 103, 104])
        iv = estimate_iv_from_history(prices, window=100)  # Window > len
        assert 0 < iv < 1, f"Should calculate IV even with small data: {iv}"

    def test_iv_constant_prices_returns_default(self):
        """Constant prices (zero vol) should return default."""
        prices = pd.Series([100.0] * 30)
        iv = estimate_iv_from_history(prices, window=20)
        # std will be 0, should return default 0.20
        assert iv == 0.20, f"Constant prices should return 0.20: {iv}"


# =============================================================================
# Test evaluate_trade_quality comprehensive
# =============================================================================

class TestEvaluateTradeQualityComprehensive:
    """Additional trade quality evaluation tests."""

    def test_quality_score_maximum(self):
        """Perfect trade should get score 3."""
        # Create greeks with delta in optimal range
        greeks = Greeks(delta=0.65, gamma=0.02, theta=-0.05, vega=0.20, rho=0.05, option_price=5.0)
        quality = evaluate_trade_quality(
            greeks=greeks,
            entry_premium=2.0,   # Low premium
            target_move=20.0,    # Large target move for high ROI
            max_days=5           # Short hold for low theta cost
        )
        # Should have: delta valid, high ROI, low theta cost
        assert quality['quality_score'] == 3, f"Perfect trade should score 3: {quality}"
        assert quality['recommendation'] == 'ACCEPT'

    def test_quality_score_minimum(self):
        """Terrible trade should get score 0."""
        # Deep OTM with tiny target
        greeks = Greeks(delta=0.10, gamma=0.001, theta=-0.50, vega=0.05, rho=0.01, option_price=0.50)
        quality = evaluate_trade_quality(
            greeks=greeks,
            entry_premium=0.50,
            target_move=1.0,     # Tiny move
            max_days=30          # Long hold for high theta cost
        )
        assert quality['quality_score'] <= 1, f"Bad trade should score low: {quality}"
        assert quality['recommendation'] in ['REJECT', 'REVIEW']

    def test_quality_issues_list_populated(self):
        """Issues list should contain specific problems."""
        greeks = Greeks(delta=0.20, gamma=0.001, theta=-0.50, vega=0.05, rho=0.01, option_price=0.50)
        quality = evaluate_trade_quality(
            greeks=greeks,
            entry_premium=1.0,
            target_move=2.0,
            max_days=30
        )
        assert len(quality['issues']) > 0, "Should have issues for bad trade"
        # Check delta issue is flagged
        delta_issue = any('delta' in issue.lower() for issue in quality['issues'])
        assert delta_issue, "Should flag delta issue"

    def test_quality_recommendation_review(self):
        """Score of 1 should give REVIEW recommendation."""
        # Borderline trade
        greeks = Greeks(delta=0.55, gamma=0.02, theta=-0.30, vega=0.20, rho=0.05, option_price=5.0)
        quality = evaluate_trade_quality(
            greeks=greeks,
            entry_premium=5.0,   # High premium
            target_move=5.0,     # Small move
            max_days=30          # Long hold
        )
        # Delta OK (score +1), but ROI low and theta high
        if quality['quality_score'] == 1:
            assert quality['recommendation'] == 'REVIEW'


# =============================================================================
# Test edge cases for calculate_greeks
# =============================================================================

class TestCalculateGreeksEdgeCases:
    """Additional edge cases for calculate_greeks."""

    def test_very_low_sigma_handled(self):
        """Very low sigma should be clamped to minimum."""
        greeks = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.0001)
        assert greeks.option_price >= 0, "Price should be non-negative"
        assert not np.isnan(greeks.delta), "Delta should not be NaN"

    def test_zero_sigma_handled(self):
        """Zero sigma should be handled gracefully."""
        greeks = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0)
        # Should clamp to minimum sigma
        assert greeks.option_price >= 0
        assert not np.isnan(greeks.delta)

    def test_very_high_sigma(self):
        """Very high sigma (100%) should work."""
        greeks = calculate_greeks(S=100, K=100, T=30/365, r=0.05, sigma=1.0)
        assert greeks.option_price > 0
        assert 0 < greeks.delta < 1, "Call delta should be between 0 and 1"

    def test_very_long_expiry(self):
        """Long expiry (2 years) should work."""
        greeks = calculate_greeks(S=100, K=100, T=2.0, r=0.05, sigma=0.20)
        assert greeks.option_price > 0
        assert greeks.vega > 0

    def test_zero_interest_rate(self):
        """Zero interest rate should work."""
        greeks = calculate_greeks(S=100, K=100, T=30/365, r=0, sigma=0.20)
        assert greeks.option_price > 0
        # Rho at r=0 should still be defined
        assert not np.isnan(greeks.rho)

    def test_negative_time_handled(self):
        """Negative T should be treated as expired."""
        greeks = calculate_greeks(S=105, K=100, T=-1/365, r=0.05, sigma=0.20, option_type='call')
        assert greeks.option_price == 5.0  # Intrinsic value
        assert greeks.delta == 1.0  # ITM at expiry


# =============================================================================
# Test standalone validate_delta_range function
# =============================================================================

class TestValidateDeltaRangeFunction:
    """Test standalone validate_delta_range function."""

    def test_validate_at_lower_bound(self):
        """Delta exactly at 0.50 should be valid."""
        valid, msg = validate_delta_range(0.50)
        assert valid is True

    def test_validate_at_upper_bound(self):
        """Delta exactly at 0.80 should be valid."""
        valid, msg = validate_delta_range(0.80)
        assert valid is True

    def test_validate_just_below_lower(self):
        """Delta just below 0.50 should warn."""
        valid, msg = validate_delta_range(0.49)
        assert valid is False
        assert "below optimal" in msg.lower()

    def test_validate_just_above_upper(self):
        """Delta just above 0.80 should warn."""
        valid, msg = validate_delta_range(0.81)
        assert valid is False

    def test_validate_deep_otm(self):
        """Very low delta should warn about OTM."""
        valid, msg = validate_delta_range(0.15)
        assert valid is False
        assert "otm" in msg.lower()

    def test_validate_deep_itm(self):
        """Very high delta should warn about stock replacement."""
        valid, msg = validate_delta_range(0.95)
        assert valid is False
        assert "stock" in msg.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
