"""
Tests for crypto/trading/sizing.py

Comprehensive tests for position sizing functions:
- calculate_position_size (risk-based sizing)
- should_skip_trade (leverage constraint checking)
- calculate_stop_distance_for_leverage
- calculate_position_size_leverage_first

Session EQUITY-70: Phase 3 Test Coverage.
"""

import pytest
from crypto.trading.sizing import (
    calculate_position_size,
    should_skip_trade,
    calculate_stop_distance_for_leverage,
    calculate_position_size_leverage_first,
)


# =============================================================================
# CALCULATE_POSITION_SIZE TESTS
# =============================================================================


class TestCalculatePositionSizeBasic:
    """Basic functionality tests for calculate_position_size."""

    def test_basic_position_size(self):
        """Basic position sizing with normal inputs."""
        size, leverage, risk = calculate_position_size(
            account_value=10000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=98000,  # 2% stop
            max_leverage=8.0
        )

        # $10k account, 2% risk = $200 target risk
        # 2% stop distance means position notional = $200 / 0.02 = $10,000
        # Leverage = $10k / $10k = 1x (no leverage)
        assert size > 0
        assert leverage > 0
        assert risk > 0
        assert leverage <= 8.0

    def test_position_size_returns_tuple(self):
        """Function returns tuple of three floats."""
        result = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=50000,
            stop_price=49000,
            max_leverage=8.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_position_size_base_currency_calculation(self):
        """Position size correctly converts to base currency."""
        size, _, _ = calculate_position_size(
            account_value=10000,
            risk_percent=0.02,
            entry_price=100000,  # $100k BTC
            stop_price=99000,    # 1% stop
            max_leverage=8.0
        )

        # With 1% stop and 2% risk on $10k = $200 risk
        # Position notional = $200 / 0.01 = $20,000
        # Position size in BTC = $20,000 / $100,000 = 0.2 BTC
        assert pytest.approx(size, rel=0.01) == 0.2


class TestCalculatePositionSizeLeverageCapping:
    """Tests for leverage capping behavior."""

    def test_leverage_capped_at_max(self):
        """Position size is capped when leverage would exceed max."""
        _, leverage, _ = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,     # 2% risk = $20
            entry_price=100000,
            stop_price=99900,      # 0.1% stop - would require 20x leverage
            max_leverage=8.0
        )

        # Should cap at 8x leverage
        assert leverage == 8.0

    def test_actual_risk_reduced_when_capped(self):
        """Actual risk is reduced when leverage is capped."""
        _, leverage, actual_risk = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,     # 2% target risk = $20
            entry_price=100000,
            stop_price=99900,      # 0.1% stop
            max_leverage=8.0
        )

        # Capped at 8x, so position = $8000 notional
        # 0.1% stop on $8000 = $8 actual risk (less than $20 target)
        assert leverage == 8.0
        assert actual_risk < 20  # Less than target
        assert pytest.approx(actual_risk, rel=0.01) == 8.0

    def test_no_capping_when_within_limits(self):
        """No capping when leverage is within limits."""
        size, leverage, risk = calculate_position_size(
            account_value=10000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=95000,      # 5% stop - requires only 0.4x leverage
            max_leverage=8.0
        )

        # 2% risk on $10k = $200
        # 5% stop means position = $200 / 0.05 = $4000
        # Leverage = $4000 / $10000 = 0.4x (no capping needed)
        assert leverage < 8.0
        assert pytest.approx(risk, rel=0.01) == 200.0


class TestCalculatePositionSizeEdgeCases:
    """Edge case tests for calculate_position_size."""

    def test_zero_account_value_returns_zeros(self):
        """Zero account value returns all zeros."""
        size, leverage, risk = calculate_position_size(
            account_value=0,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=99000,
            max_leverage=8.0
        )

        assert size == 0.0
        assert leverage == 0.0
        assert risk == 0.0

    def test_negative_account_value_returns_zeros(self):
        """Negative account value returns all zeros."""
        size, leverage, risk = calculate_position_size(
            account_value=-1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=99000,
            max_leverage=8.0
        )

        assert size == 0.0
        assert leverage == 0.0
        assert risk == 0.0

    def test_zero_entry_price_returns_zeros(self):
        """Zero entry price returns all zeros."""
        size, leverage, risk = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=0,
            stop_price=99000,
            max_leverage=8.0
        )

        assert size == 0.0
        assert leverage == 0.0
        assert risk == 0.0

    def test_zero_stop_price_returns_zeros(self):
        """Zero stop price returns all zeros."""
        size, leverage, risk = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=0,
            max_leverage=8.0
        )

        assert size == 0.0
        assert leverage == 0.0
        assert risk == 0.0

    def test_same_entry_and_stop_returns_zeros(self):
        """Same entry and stop (zero distance) returns all zeros."""
        size, leverage, risk = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=100000,  # Same as entry
            max_leverage=8.0
        )

        assert size == 0.0
        assert leverage == 0.0
        assert risk == 0.0

    def test_stop_above_entry_for_short(self):
        """Stop above entry (short position) calculates correctly."""
        size, leverage, risk = calculate_position_size(
            account_value=10000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=102000,  # Stop above for short
            max_leverage=8.0
        )

        # Should still work - uses abs() for stop distance
        assert size > 0
        assert leverage > 0
        assert risk > 0


# =============================================================================
# SHOULD_SKIP_TRADE TESTS
# =============================================================================


class TestShouldSkipTrade:
    """Tests for should_skip_trade function."""

    def test_skip_when_leverage_exceeds_max(self):
        """Trade should be skipped when required leverage exceeds max."""
        skip, reason = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,     # 2% risk
            entry_price=100000,
            stop_price=99900,      # 0.1% stop - requires 20x
            max_leverage=8.0
        )

        assert skip is True
        assert "leverage" in reason.lower()
        assert "20.0x" in reason

    def test_no_skip_when_within_limits(self):
        """Trade should not be skipped when within leverage limits."""
        skip, reason = should_skip_trade(
            account_value=10000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=95000,      # 5% stop - only 0.4x leverage
            max_leverage=8.0
        )

        assert skip is False
        assert reason == "Trade acceptable"

    def test_skip_when_entry_zero(self):
        """Trade should be skipped with zero entry price."""
        skip, reason = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,
            entry_price=0,
            stop_price=99000,
            max_leverage=8.0
        )

        assert skip is True
        assert "invalid" in reason.lower()

    def test_skip_when_stop_zero(self):
        """Trade should be skipped with zero stop price."""
        skip, reason = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=0,
            max_leverage=8.0
        )

        assert skip is True
        assert "invalid" in reason.lower()

    def test_skip_when_same_entry_and_stop(self):
        """Trade should be skipped when entry equals stop."""
        skip, reason = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=100000,
            max_leverage=8.0
        )

        assert skip is True
        assert "zero" in reason.lower()

    def test_returns_tuple(self):
        """Function returns tuple of bool and str."""
        result = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=99000,
            max_leverage=8.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# =============================================================================
# CALCULATE_STOP_DISTANCE_FOR_LEVERAGE TESTS
# =============================================================================


class TestCalculateStopDistanceForLeverage:
    """Tests for calculate_stop_distance_for_leverage function."""

    def test_basic_stop_distance_calculation(self):
        """Basic stop distance calculation."""
        distance = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.02,     # 2% = $20 risk
            entry_price=100000,
            target_leverage=4.0    # $4000 position
        )

        # $20 risk / $4000 position = 0.5% stop
        # 0.5% of $100,000 = $500 stop distance
        assert pytest.approx(distance, rel=0.01) == 500.0

    def test_higher_leverage_means_tighter_stop(self):
        """Higher leverage requires tighter stop for same risk."""
        distance_4x = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            target_leverage=4.0
        )

        distance_8x = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            target_leverage=8.0
        )

        # Higher leverage = smaller stop distance
        assert distance_8x < distance_4x
        assert distance_8x == pytest.approx(distance_4x / 2, rel=0.01)

    def test_higher_risk_means_wider_stop(self):
        """Higher risk percentage allows wider stop."""
        distance_2pct = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.02,     # 2%
            entry_price=100000,
            target_leverage=4.0
        )

        distance_4pct = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.04,     # 4%
            entry_price=100000,
            target_leverage=4.0
        )

        # Double risk = double stop distance
        assert distance_4pct > distance_2pct
        assert distance_4pct == pytest.approx(distance_2pct * 2, rel=0.01)


# =============================================================================
# CALCULATE_POSITION_SIZE_LEVERAGE_FIRST TESTS
# =============================================================================


class TestCalculatePositionSizeLeverageFirst:
    """Tests for calculate_position_size_leverage_first function."""

    def test_basic_leverage_first_sizing(self):
        """Basic leverage-first position sizing."""
        size, leverage, risk = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=98000,      # 2% stop
            leverage=4.0
        )

        # $1000 at 4x = $4000 position
        # $4000 / $100k = 0.04 BTC
        assert pytest.approx(size, rel=0.01) == 0.04
        assert leverage == 4.0
        # 2% stop on $4000 = $80 risk
        assert pytest.approx(risk, rel=0.01) == 80.0

    def test_leverage_first_returns_specified_leverage(self):
        """Leverage-first always returns specified leverage."""
        _, leverage_10x, _ = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=99900,      # Tight stop (0.1%)
            leverage=10.0
        )

        _, leverage_4x, _ = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=99900,
            leverage=4.0
        )

        assert leverage_10x == 10.0
        assert leverage_4x == 4.0

    def test_risk_floats_with_leverage_first(self):
        """Actual risk floats based on stop distance."""
        _, _, risk_tight = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=99900,      # 0.1% stop
            leverage=4.0
        )

        _, _, risk_wide = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=95000,      # 5% stop
            leverage=4.0
        )

        # Same leverage, different stops = different risk
        assert risk_wide > risk_tight
        # $4000 position: 0.1% stop = $4, 5% stop = $200
        assert pytest.approx(risk_tight, rel=0.01) == 4.0
        assert pytest.approx(risk_wide, rel=0.01) == 200.0

    def test_leverage_first_edge_cases(self):
        """Edge cases for leverage-first sizing."""
        # Zero account value
        size, lev, risk = calculate_position_size_leverage_first(
            account_value=0,
            entry_price=100000,
            stop_price=99000,
            leverage=4.0
        )
        assert size == 0.0
        assert lev == 0.0
        assert risk == 0.0

        # Zero entry price
        size, lev, risk = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=0,
            stop_price=99000,
            leverage=4.0
        )
        assert size == 0.0

        # Zero stop price
        size, lev, risk = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=0,
            leverage=4.0
        )
        assert size == 0.0

    def test_leverage_first_zero_leverage_defaults_to_one(self):
        """Zero or negative leverage defaults to 1x."""
        size, leverage, risk = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=99000,
            leverage=0
        )

        # Should use 1x leverage
        assert leverage == 1.0
        # $1000 / $100k = 0.01 BTC
        assert pytest.approx(size, rel=0.01) == 0.01

    def test_leverage_first_same_entry_stop_zero_risk(self):
        """Same entry and stop results in zero risk."""
        size, leverage, risk = calculate_position_size_leverage_first(
            account_value=1000,
            entry_price=100000,
            stop_price=100000,
            leverage=4.0
        )

        # Position is calculated, but risk is zero
        assert pytest.approx(size, rel=0.01) == 0.04
        assert leverage == 4.0
        assert risk == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSizingIntegration:
    """Integration tests combining multiple functions."""

    def test_risk_based_vs_leverage_first_comparison(self):
        """Compare risk-based and leverage-first sizing."""
        # Same scenario
        account = 1000
        entry = 100000
        stop = 98000  # 2% stop

        # Risk-based: 2% risk = $20
        size_risk, lev_risk, actual_risk = calculate_position_size(
            account_value=account,
            risk_percent=0.02,
            entry_price=entry,
            stop_price=stop,
            max_leverage=8.0
        )

        # Leverage-first: 4x leverage
        size_lev, lev_lev, risk_lev = calculate_position_size_leverage_first(
            account_value=account,
            entry_price=entry,
            stop_price=stop,
            leverage=4.0
        )

        # Risk-based targets fixed risk ($20)
        assert pytest.approx(actual_risk, rel=0.01) == 20.0

        # Leverage-first uses fixed leverage (4x), risk floats
        assert lev_lev == 4.0
        assert risk_lev > actual_risk  # Higher leverage = higher risk

    def test_skip_trade_and_sizing_consistency(self):
        """should_skip_trade and calculate_position_size are consistent."""
        # Scenario that requires high leverage
        skip, _ = should_skip_trade(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=99900,  # 0.1% stop
            max_leverage=8.0
        )

        _, leverage, _ = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=99900,
            max_leverage=8.0
        )

        if skip:
            # If skipped, leverage would have been capped
            assert leverage == 8.0  # Capped at max
        else:
            # If not skipped, leverage is within limits
            assert leverage <= 8.0

    def test_stop_distance_enables_target_leverage(self):
        """Stop distance calculation enables target leverage."""
        # Calculate stop distance for 4x leverage
        distance = calculate_stop_distance_for_leverage(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            target_leverage=4.0
        )

        # Use that stop distance and verify we get 4x
        stop_price = 100000 - distance
        size, leverage, _ = calculate_position_size(
            account_value=1000,
            risk_percent=0.02,
            entry_price=100000,
            stop_price=stop_price,
            max_leverage=8.0
        )

        # Leverage should be approximately 4x
        assert pytest.approx(leverage, rel=0.05) == 4.0
