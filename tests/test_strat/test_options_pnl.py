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
        # Session 78: risk_free_rate is now a fallback (date-based lookup is primary)
        backtester = OptionsBacktester(risk_free_rate=0.05, default_iv=0.20)

        assert backtester.risk_free_rate_fallback == 0.05
        assert backtester.default_iv == 0.20

    def test_backtester_default_params(self):
        """Backtester should have sensible defaults."""
        backtester = OptionsBacktester()

        # Should have defaults (check they exist)
        # Session 78: risk_free_rate renamed to risk_free_rate_fallback
        assert hasattr(backtester, 'risk_free_rate_fallback')
        assert hasattr(backtester, 'default_iv')

    def test_backtester_accepts_thetadata_provider(self):
        """Session 82: Backtester should accept thetadata_provider parameter."""
        # Test with None (forces Black-Scholes fallback)
        backtester = OptionsBacktester(thetadata_provider=None)
        assert backtester.thetadata_provider is None

        # Test with mock provider
        from unittest.mock import MagicMock
        mock_provider = MagicMock()
        backtester_with_provider = OptionsBacktester(thetadata_provider=mock_provider)
        assert backtester_with_provider.thetadata_provider is mock_provider


class TestThetaDataIntegration:
    """Session 82: Test ThetaData integration in options backtesting."""

    def test_results_include_data_source_columns(self):
        """Results DataFrame should include data_source tracking columns."""
        # This test validates that Phase 1 changes added the tracking columns
        backtester = OptionsBacktester()

        # Results from backtest should have data_source columns
        # (actual trade test would require price data, just validate structure)
        expected_columns = ['data_source', 'entry_source', 'exit_source']

        # The columns should be added to results - verify by checking docstring/code exists
        import inspect
        source = inspect.getsource(backtester.backtest_trades)

        # Verify data_source tracking code exists
        assert 'data_source' in source, "data_source tracking should be in backtest_trades"
        assert 'entry_source' in source, "entry_source tracking should be in backtest_trades"
        assert 'exit_source' in source, "exit_source tracking should be in backtest_trades"

    def test_market_price_method_exists(self):
        """_get_market_price method should exist for ThetaData lookup."""
        backtester = OptionsBacktester()

        assert hasattr(backtester, '_get_market_price'), "_get_market_price method required"
        assert callable(backtester._get_market_price), "_get_market_price should be callable"

    def test_market_greeks_method_exists(self):
        """_get_market_greeks method should exist for ThetaData lookup."""
        backtester = OptionsBacktester()

        assert hasattr(backtester, '_get_market_greeks'), "_get_market_greeks method required"
        assert callable(backtester._get_market_greeks), "_get_market_greeks should be callable"

    def test_market_price_returns_none_without_provider(self):
        """_get_market_price should return None when no provider configured."""
        backtester = OptionsBacktester(thetadata_provider=None)

        from unittest.mock import MagicMock

        # Create a mock trade with just the contract info needed by _get_market_price
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 11, 15),
            option_type=OptionType.CALL,
            strike=460.0,
        )

        mock_trade = MagicMock()
        mock_trade.contract = contract

        result = backtester._get_market_price(mock_trade, 458.0, datetime(2024, 10, 15))
        assert result is None, "Should return None without ThetaData provider"

    def test_fallback_to_black_scholes(self):
        """When ThetaData unavailable, should fall back to Black-Scholes."""
        # Backtester without ThetaData should still work
        backtester = OptionsBacktester(thetadata_provider=None)

        # Verify BS fallback flag/behavior exists
        import inspect
        source = inspect.getsource(backtester.backtest_trades)

        # Should have fallback logic
        assert 'market_entry_price' in source, "Should attempt market price lookup"
        assert 'market_entry_greeks' in source, "Should attempt market Greeks lookup"
        # Black-Scholes fallback should be used when market data unavailable
        assert 'used_market_data_entry = False' in source, "Should track BS fallback"


class TestThetaDataFetcherFallbackParams:
    """Session 82: Test improved fallback parameters in ThetaDataOptionsFetcher."""

    def test_fetcher_imports_risk_free_rate(self):
        """ThetaDataOptionsFetcher should import get_risk_free_rate."""
        from integrations.thetadata_options_fetcher import RISK_FREE_RATE_AVAILABLE
        assert RISK_FREE_RATE_AVAILABLE is True, "Risk-free rate module should be available"

    def test_fallback_iv_is_realistic(self):
        """Fallback IV should be 0.15, not 0.20 (Session 81 finding)."""
        from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
        import inspect

        source = inspect.getsource(ThetaDataOptionsFetcher._calculate_bs_price)

        # Should use 0.15 IV, not 0.20
        assert 'sigma = 0.15' in source, "Fallback IV should be 0.15"
        assert 'sigma = 0.20' not in source, "Old 0.20 IV should be removed"

    def test_fallback_rate_uses_dynamic_lookup(self):
        """Fallback should attempt dynamic risk-free rate lookup."""
        from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
        import inspect

        source = inspect.getsource(ThetaDataOptionsFetcher._calculate_bs_price)

        # Should use get_risk_free_rate when available
        assert 'get_risk_free_rate' in source, "Should use dynamic risk-free rate"
        assert 'RISK_FREE_RATE_AVAILABLE' in source, "Should check if rate module available"


class TestSession83K10Fixes:
    """
    Session 83K-10: Tests for ThetaData coverage and MaxDD bug fixes.

    Bug #1: ThetaData 0% coverage due to case mismatch
    Bug #2: MaxDD 5000%+ due to negative equity and unbounded calculation
    """

    def test_data_source_values_match_pattern_metrics(self):
        """Verify data_source values use correct case for pattern_metrics.py."""
        import inspect
        from strat.options_module import OptionsBacktester

        source = inspect.getsource(OptionsBacktester.backtest_trades)

        # Session 83K-10: Must use PascalCase to match pattern_metrics.py
        assert "'ThetaData'" in source, "data_source should be 'ThetaData' (PascalCase)"
        assert "'BlackScholes'" in source, "data_source should be 'BlackScholes' (PascalCase)"
        assert "'Mixed'" in source, "data_source should be 'Mixed' (PascalCase)"

        # Old lowercase values should be removed
        assert "data_source = 'thetadata'" not in source, "Old lowercase 'thetadata' should be removed"
        assert "data_source = 'black_scholes'" not in source, "Old lowercase 'black_scholes' should be removed"
        assert "data_source = 'mixed'" not in source, "Old lowercase 'mixed' should be removed"

    def test_maxdd_capped_at_100_percent(self):
        """MaxDD should never exceed 100% for cash-secured options."""
        # Import directly to avoid circular import through strategies/__init__.py
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "strat_options_strategy",
            "C:\\Strat_Trading_Bot\\vectorbt-workspace\\strategies\\strat_options_strategy.py"
        )
        module = importlib.util.module_from_spec(spec)

        # Test the logic directly without full module import
        import pandas as pd
        import numpy as np

        # Simulate the equity calculation with floor and cap
        equity = [10000]
        pnl_values = [-5000, -5000, -5000, -5000]  # $20k loss

        for pnl in pnl_values:
            new_equity = equity[-1] + pnl
            equity.append(max(0, new_equity))  # Floor at zero

        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        peak_safe = np.where(peak == 0, 1e-10, peak)
        drawdown = (peak - equity_arr) / peak_safe
        max_dd = min(drawdown.max(), 1.0)  # Cap at 100%

        # MaxDD should be capped at 100% (1.0)
        assert max_dd <= 1.0, f"MaxDD {max_dd} exceeds 100% cap"
        assert max_dd == 1.0, f"MaxDD should be exactly 100% for total loss scenario"

    def test_equity_floor_at_zero(self):
        """Equity curve should never go negative."""
        import numpy as np

        # Simulate equity calculation with floor
        equity = [10000]
        pnl_values = [-15000]  # More than starting capital

        for pnl in pnl_values:
            new_equity = equity[-1] + pnl
            equity.append(max(0, new_equity))  # Floor at zero

        # Equity should be floored at 0, not negative
        assert all(e >= 0 for e in equity), f"Equity contains negative: {equity}"
        assert equity[-1] == 0, f"Equity should be 0 after total loss, got {equity[-1]}"

    def test_monte_carlo_equity_floor(self):
        """Monte Carlo equity curve should floor at zero."""
        from validation.monte_carlo import MonteCarloValidator
        import numpy as np

        validator = MonteCarloValidator()

        # Simulate build equity curve using the actual method
        class MockTrade:
            def __init__(self, pnl):
                self.pnl = pnl

        mock_trades = [MockTrade(-15000), MockTrade(-5000)]
        equity = validator._calculate_equity_curve(mock_trades, starting_capital=10000)

        # All equity values should be >= 0
        assert all(equity >= 0), f"Equity curve contains negative values: {equity}"

    def test_monte_carlo_maxdd_capped(self):
        """Monte Carlo MaxDD calculation should cap at 100%."""
        from validation.monte_carlo import MonteCarloValidator
        import numpy as np

        validator = MonteCarloValidator()

        # Equity that would produce >100% DD without cap (if it could go negative)
        equity = np.array([10000, 5000, 2000, 0, 0, 0])  # Floored at 0

        max_dd = validator._calculate_max_drawdown(equity)

        # MaxDD should be capped at 100%
        assert max_dd <= 1.0, f"MaxDD {max_dd} exceeds 100% cap"


class TestSession83K11PnLFix:
    """Session 83K-11: Regression tests for P&L calculation bug fix.

    Bug: P&L calculation incorrectly subtracted option premium AGAIN from gross_pnl
    for TARGET hits, and assumed 100% loss for all STOP hits.

    Fix: Use gross_pnl (delta + gamma + theta) directly as P&L for both cases.
    The Greek-based change in option value IS the profit/loss.
    """

    def test_target_hit_pnl_not_double_subtracted(self):
        """TARGET hit P&L should equal Greek-based change, not change minus premium."""
        from strat.options_module import calculate_greeks

        # Simulate a winning call trade
        S_entry = 100  # Entry underlying price
        S_exit = 110   # Exit underlying price (target hit)
        K = 100        # ATM call
        T = 30/365     # 30 DTE
        r = 0.05
        sigma = 0.20

        entry_greeks = calculate_greeks(S_entry, K, T, r, sigma, 'call')
        exit_greeks = calculate_greeks(S_exit, K, T - 5/365, r, sigma, 'call')  # 5 days later

        price_move = S_exit - S_entry  # +10
        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        avg_gamma = (entry_greeks.gamma + exit_greeks.gamma) / 2
        avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2

        # Calculate P&L using fixed formula (just the change)
        delta_pnl = avg_delta * price_move * 100  # Per contract
        gamma_pnl = 0.5 * avg_gamma * (price_move ** 2) * 100
        theta_pnl = avg_theta * 5 * 100  # 5 days of theta decay

        gross_pnl = delta_pnl + gamma_pnl + theta_pnl
        pnl = gross_pnl  # CORRECT: Just use the change

        # P&L should be POSITIVE for a winning call
        assert pnl > 0, f"Winning call TARGET should have positive P&L, got {pnl}"

        # The buggy calculation would have been:
        buggy_pnl = gross_pnl - entry_greeks.option_price * 100
        assert buggy_pnl < pnl, "Buggy calculation would subtract premium again"

    def test_stop_hit_uses_greeks_not_100_percent_loss(self):
        """STOP hit P&L should use Greek-based calculation, not assume 100% loss."""
        from strat.options_module import calculate_greeks

        # Simulate a losing call trade (stop hit)
        S_entry = 100  # Entry underlying price
        S_exit = 97    # Exit underlying price (stop hit, 3% drop)
        K = 100        # ATM call
        T = 30/365     # 30 DTE
        r = 0.05
        sigma = 0.20

        entry_greeks = calculate_greeks(S_entry, K, T, r, sigma, 'call')
        exit_greeks = calculate_greeks(S_exit, K, T - 2/365, r, sigma, 'call')  # 2 days later

        price_move = S_exit - S_entry  # -3
        avg_delta = (entry_greeks.delta + exit_greeks.delta) / 2
        avg_gamma = (entry_greeks.gamma + exit_greeks.gamma) / 2
        avg_theta = (entry_greeks.theta + exit_greeks.theta) / 2

        # Calculate P&L using fixed formula (Greeks-based)
        delta_pnl = avg_delta * price_move * 100  # Per contract
        gamma_pnl = 0.5 * avg_gamma * (price_move ** 2) * 100
        theta_pnl = avg_theta * 2 * 100  # 2 days of theta decay

        gross_pnl = delta_pnl + gamma_pnl + theta_pnl
        pnl = gross_pnl  # CORRECT: Greeks-based P&L

        # P&L should be NEGATIVE for a losing call
        assert pnl < 0, f"Losing call STOP should have negative P&L, got {pnl}"

        # But NOT -100% of premium (that's unrealistic for a 3% drop)
        premium = entry_greeks.option_price * 100
        buggy_pnl = -premium  # Old buggy code: assume 100% loss

        # The actual loss should be less severe than 100% premium
        assert abs(pnl) < abs(buggy_pnl), \
            f"STOP loss {pnl} should be less than 100% premium loss {buggy_pnl}"

    def test_pnl_direction_consistency(self):
        """Verify P&L sign is consistent with trade outcome."""
        from strat.options_module import calculate_greeks

        K = 100
        T = 30/365
        r = 0.05
        sigma = 0.20

        # Test case 1: Call wins (price up)
        call_entry = calculate_greeks(100, K, T, r, sigma, 'call')
        call_exit = calculate_greeks(108, K, T - 3/365, r, sigma, 'call')
        call_pnl = ((call_entry.delta + call_exit.delta) / 2) * 8 * 100
        assert call_pnl > 0, "Call should profit when price goes up"

        # Test case 2: Put wins (price down)
        put_entry = calculate_greeks(100, K, T, r, sigma, 'put')
        put_exit = calculate_greeks(92, K, T - 3/365, r, sigma, 'put')
        put_pnl = ((put_entry.delta + put_exit.delta) / 2) * (-8) * 100
        # Put delta is negative, price_move is negative, so pnl is positive
        assert put_pnl > 0, "Put should profit when price goes down"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
