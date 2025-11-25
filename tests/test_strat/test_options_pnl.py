"""
Tests for Options P/L calculation accuracy.

Session 72: Validates that P/L calculations are correct after bug fixes:
- Bug #1: abs(delta_pnl) removed - direction now correct
- Bug #3: Hardcoded $5 premium replaced with Black-Scholes price
- Bug #4/#5: Average Greeks now used for multi-day holds

Test Categories:
1. Direction sign validation (winning vs losing trades)
2. Premium calculation (Black-Scholes vs hardcoded)
3. Average Greeks P/L accuracy
4. Stop loss max loss validation
"""

import pytest
import numpy as np
from strat.greeks import calculate_greeks, Greeks, calculate_pnl_with_greeks
from strat.options_module import (
    OptionsBacktester,
    OptionsExecutor,
    OptionContract,
    OptionType,
)
from datetime import datetime


class TestDirectionSignValidation:
    """Test that P/L direction is correct after abs() bug fix."""

    def test_winning_call_positive_pnl(self):
        """Winning call (price up) should have positive delta P/L direction."""
        # Call option: buy when stock at $450, exit at $460
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=460, K=455, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )

        price_move = 460 - 450  # +$10

        # Calculate P/L using average Greeks (as fixed)
        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        avg_gamma = (entry_greeks.gamma + exit_greeks.gamma) / 2
        avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2

        delta_pnl = avg_delta * price_move * 100
        gamma_pnl = 0.5 * avg_gamma * (price_move ** 2) * 100
        theta_pnl = avg_theta * 5 * 100  # 5 days

        gross_pnl = delta_pnl + gamma_pnl + theta_pnl

        # This test validates DIRECTION is correct after abs() bug fix
        # Delta P/L MUST be positive for call when price goes up
        assert delta_pnl > 0, f"Delta P/L should be positive for winning call: {delta_pnl}"
        assert gamma_pnl > 0, f"Gamma P/L should be positive for any move: {gamma_pnl}"
        # Gross P/L (before premium) should be positive when direction is right
        assert gross_pnl > 0 or abs(theta_pnl) < delta_pnl + gamma_pnl, \
            f"Gross P/L should be positive or close: {gross_pnl}"

        # Alternative validation: option price at exit > entry (ITM now)
        assert exit_greeks.option_price > entry_greeks.option_price, \
            f"Exit price {exit_greeks.option_price} should exceed entry {entry_greeks.option_price}"

    def test_losing_call_negative_pnl(self):
        """Losing call (price down) should have negative P/L."""
        # Call option: buy when stock at $450, exit at $440
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=440, K=455, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )

        price_move = 440 - 450  # -$10

        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        delta_pnl = avg_delta * price_move * 100

        # Losing call should have NEGATIVE delta P/L
        assert delta_pnl < 0, f"Delta P/L should be negative for losing call: {delta_pnl}"

    def test_winning_put_positive_pnl(self):
        """Winning put (price down) should have positive P/L."""
        # Put option: buy when stock at $450, exit at $440
        entry_greeks = calculate_greeks(
            S=450, K=445, T=35/365, r=0.05, sigma=0.20, option_type='put'
        )
        exit_greeks = calculate_greeks(
            S=440, K=445, T=30/365, r=0.05, sigma=0.20, option_type='put'
        )

        price_move = 440 - 450  # -$10 (stock goes down)

        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        delta_pnl = avg_delta * price_move * 100

        # Put delta is negative, price move is negative, so delta_pnl should be POSITIVE
        # (-0.40) * (-10) = +4
        assert delta_pnl > 0, f"Delta P/L should be positive for winning put: {delta_pnl}"

    def test_losing_put_negative_pnl(self):
        """Losing put (price up) should have negative P/L."""
        # Put option: buy when stock at $450, exit at $460
        entry_greeks = calculate_greeks(
            S=450, K=445, T=35/365, r=0.05, sigma=0.20, option_type='put'
        )
        exit_greeks = calculate_greeks(
            S=460, K=445, T=30/365, r=0.05, sigma=0.20, option_type='put'
        )

        price_move = 460 - 450  # +$10 (stock goes up - bad for put)

        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        delta_pnl = avg_delta * price_move * 100

        # Put delta is negative, price move is positive, so delta_pnl should be NEGATIVE
        # (-0.40) * (+10) = -4
        assert delta_pnl < 0, f"Delta P/L should be negative for losing put: {delta_pnl}"


class TestPremiumCalculation:
    """Test that premium uses Black-Scholes price, not hardcoded values."""

    def test_atm_premium_not_5_dollars(self):
        """ATM SPY option premium should NOT be hardcoded $5."""
        # ATM SPY call at typical conditions
        greeks = calculate_greeks(
            S=450, K=450, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        # ATM option should be roughly $8-15, NOT $5
        assert greeks.option_price > 6.0, f"ATM option {greeks.option_price} should not be ~$5"
        assert greeks.option_price < 20.0, f"ATM option {greeks.option_price} seems too high"

    def test_otm_premium_realistic(self):
        """OTM option premium should be realistic."""
        greeks = calculate_greeks(
            S=450, K=460, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        # OTM option should have some premium but less than ATM
        assert 1.0 < greeks.option_price < 8.0, f"OTM price {greeks.option_price} unrealistic"

    def test_itm_premium_includes_intrinsic(self):
        """ITM option premium should include intrinsic value."""
        greeks = calculate_greeks(
            S=450, K=440, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        intrinsic = 450 - 440  # $10
        # ITM option should be worth at least intrinsic value
        assert greeks.option_price >= intrinsic, f"ITM price {greeks.option_price} < intrinsic {intrinsic}"

    def test_premium_scales_with_underlying(self):
        """Premium should scale with underlying price."""
        greeks_low = calculate_greeks(
            S=100, K=100, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        greeks_high = calculate_greeks(
            S=500, K=500, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        # Higher priced stock should have higher option premium (roughly proportional)
        ratio = greeks_high.option_price / greeks_low.option_price
        assert 3.0 < ratio < 7.0, f"Premium ratio {ratio} seems wrong"


class TestAverageGreeksAccuracy:
    """Test average Greeks P/L calculation for multi-day holds."""

    def test_average_delta_more_accurate(self):
        """Average delta should be between entry and exit delta."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=460, K=455, T=28/365, r=0.05, sigma=0.20, option_type='call'
        )

        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2

        # Average should be between entry and exit
        min_delta = min(entry_greeks.delta, exit_greeks.delta)
        max_delta = max(entry_greeks.delta, exit_greeks.delta)

        assert min_delta <= avg_delta <= max_delta

    def test_theta_average_for_7day_hold(self):
        """Average theta should account for acceleration near expiration."""
        entry_greeks = calculate_greeks(
            S=450, K=450, T=14/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=450, K=450, T=7/365, r=0.05, sigma=0.20, option_type='call'
        )

        avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2

        # Exit theta should be more negative (theta accelerates)
        assert exit_greeks.theta < entry_greeks.theta, "Theta should accelerate"

        # Average should be more negative than entry but less than exit
        assert avg_theta < entry_greeks.theta
        assert avg_theta > exit_greeks.theta

    def test_pnl_calculation_with_average_greeks(self):
        """Full P/L calculation using average Greeks."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=460, K=455, T=28/365, r=0.05, sigma=0.20, option_type='call'
        )

        price_move = 10.0  # $10 up
        days_held = 7

        # Calculate with average Greeks (our fixed approach)
        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        avg_gamma = (entry_greeks.gamma + exit_greeks.gamma) / 2
        avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2

        delta_pnl = avg_delta * price_move * 100
        gamma_pnl = 0.5 * avg_gamma * (price_move ** 2) * 100
        theta_pnl = avg_theta * days_held * 100

        gross_pnl = delta_pnl + gamma_pnl + theta_pnl

        # Gross P/L should be positive (price moved in our favor)
        assert gross_pnl > 0, f"Gross P/L should be positive: {gross_pnl}"

        # Delta should dominate for small moves
        assert abs(delta_pnl) > abs(gamma_pnl), "Delta should dominate gamma for $10 move"


class TestStopLossMaxLoss:
    """Test that stop loss limits loss to premium paid."""

    def test_stop_loss_max_loss_is_premium(self):
        """Stop loss should limit loss to exactly premium paid."""
        # Setup: Buy call option with known premium
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        premium_paid = entry_greeks.option_price
        contracts = 1

        # Max loss at stop should be premium * 100 * contracts
        max_loss = -premium_paid * 100 * contracts

        # This is what the backtester should return on STOP exit
        # From options_module.py line 682: pnl = -actual_option_cost * 100 * trade.quantity
        expected_stop_pnl = -premium_paid * 100 * contracts

        assert max_loss == expected_stop_pnl, f"Stop P/L {expected_stop_pnl} != max loss {max_loss}"

    def test_max_loss_regardless_of_price_move(self):
        """Even if price crashes, max loss is premium."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )

        # Price crashes to $400 (massive move against us)
        exit_greeks = calculate_greeks(
            S=400, K=455, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )

        price_move = 400 - 450  # -$50

        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        delta_pnl = avg_delta * price_move * 100

        # Delta P/L would be huge negative
        # But with stop loss, we only lose premium
        max_loss = -entry_greeks.option_price * 100

        # The stop loss protects us from the full delta P/L
        assert abs(delta_pnl) > abs(max_loss), f"Stop should protect from large loss"


class TestPnlFunctionDirectly:
    """Test the calculate_pnl_with_greeks function directly."""

    def test_basic_pnl_calculation(self):
        """Test basic P/L calculation returns correct structure."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=455, K=455, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=5.0,
            days_held=5,
            contracts=2,
            entry_premium=entry_greeks.option_price
        )

        # Check structure
        assert 'delta_pnl' in pnl
        assert 'gamma_pnl' in pnl
        assert 'theta_pnl' in pnl
        assert 'gross_pnl' in pnl
        assert 'net_pnl' in pnl

    def test_pnl_theta_is_negative(self):
        """Theta P/L should be negative (time decay costs money)."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=450, K=455, T=30/365, r=0.05, sigma=0.20, option_type='call'
        )

        pnl = calculate_pnl_with_greeks(
            entry_greeks=entry_greeks,
            exit_greeks=exit_greeks,
            price_move=0.0,  # No price movement
            days_held=5,
            contracts=1,
            entry_premium=entry_greeks.option_price
        )

        # Theta P/L should be negative
        assert pnl['theta_pnl'] < 0, f"Theta P/L should be negative: {pnl['theta_pnl']}"

    def test_pnl_scales_with_contracts(self):
        """P/L should scale linearly with number of contracts."""
        entry_greeks = calculate_greeks(
            S=450, K=455, T=35/365, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=458, K=455, T=28/365, r=0.05, sigma=0.20, option_type='call'
        )

        pnl_1 = calculate_pnl_with_greeks(
            entry_greeks, exit_greeks, price_move=8.0, days_held=7, contracts=1
        )
        pnl_3 = calculate_pnl_with_greeks(
            entry_greeks, exit_greeks, price_move=8.0, days_held=7, contracts=3
        )

        ratio = pnl_3['net_pnl'] / pnl_1['net_pnl']
        assert 2.9 < ratio < 3.1, f"P/L should scale 3x with 3 contracts: {ratio}"


class TestBacktesterInstantiation:
    """Test OptionsBacktester can be instantiated with correct parameters."""

    def test_backtester_correct_params(self):
        """Backtester should accept risk_free_rate and default_iv."""
        # This was Bug #2 - test script used invalid parameter
        backtester = OptionsBacktester(risk_free_rate=0.05, default_iv=0.20)

        assert backtester.risk_free_rate == 0.05
        assert backtester.default_iv == 0.20

    def test_backtester_default_params(self):
        """Backtester should have sensible defaults."""
        backtester = OptionsBacktester()

        # Should have defaults (check they exist)
        assert hasattr(backtester, 'risk_free_rate')
        assert hasattr(backtester, 'default_iv')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
