"""
Tests for crypto/trading/derivatives.py - Perpetual futures utilities.

Tests cover:
- get_leverage_for_tier() - tier lookup
- calculate_funding_cost() - long/short funding
- get_next_funding_time() - datetime calculation
- time_to_funding() - hours remaining
- calculate_initial_margin() - margin requirements
- calculate_maintenance_margin() - maintenance calc
- calculate_liquidation_price() - liq price formula
- should_close_before_funding() - decision logic
- calculate_effective_leverage() - current leverage
- is_leverage_safe() - tier compliance check

Session: EQUITY-82
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from crypto.trading.derivatives import (
    get_leverage_for_tier,
    calculate_funding_cost,
    get_next_funding_time,
    time_to_funding,
    calculate_initial_margin,
    calculate_maintenance_margin,
    calculate_liquidation_price,
    should_close_before_funding,
    calculate_effective_leverage,
    is_leverage_safe,
)


# =============================================================================
# LEVERAGE TIER TESTS
# =============================================================================


class TestGetLeverageForTier:
    """Tests for get_leverage_for_tier function."""

    def test_btc_intraday_10x(self):
        """BTC intraday should be 10x."""
        leverage = get_leverage_for_tier("BTC-USD", "intraday")
        assert leverage == 10.0

    def test_btc_swing_4x(self):
        """BTC swing should be 4x."""
        leverage = get_leverage_for_tier("BTC-USD", "swing")
        assert leverage == 4.0

    def test_eth_intraday_10x(self):
        """ETH intraday should be 10x."""
        leverage = get_leverage_for_tier("ETH-USD", "intraday")
        assert leverage == 10.0

    def test_sol_intraday_5x(self):
        """SOL intraday should be 5x."""
        leverage = get_leverage_for_tier("SOL-USD", "intraday")
        assert leverage == 5.0

    def test_ada_swing_3x(self):
        """ADA swing should be 3x."""
        leverage = get_leverage_for_tier("ADA-USD", "swing")
        assert leverage == 3.0

    def test_symbol_without_suffix(self):
        """Symbol without -USD suffix should work."""
        leverage = get_leverage_for_tier("BTC", "intraday")
        assert leverage == 10.0

    def test_unknown_tier_defaults_to_swing(self):
        """Unknown tier should default to swing."""
        leverage = get_leverage_for_tier("BTC", "unknown_tier")
        assert leverage == 4.0  # BTC swing

    def test_unknown_symbol_defaults_to_4x(self):
        """Unknown symbol should default to 4x."""
        leverage = get_leverage_for_tier("UNKNOWN", "swing")
        assert leverage == 4.0


# =============================================================================
# FUNDING COST TESTS
# =============================================================================


class TestCalculateFundingCost:
    """Tests for calculate_funding_cost function."""

    def test_long_pays_positive_funding(self):
        """Long position should pay when funding rate is positive."""
        cost = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=8,  # One funding period
            funding_rate=0.0001,  # 0.01%
        )
        # Long pays: -$10,000 * 0.0001 * 1 = -$1
        assert cost < 0  # Negative = cost

    def test_short_receives_positive_funding(self):
        """Short position should receive when funding rate is positive."""
        cost = calculate_funding_cost(
            position_size_usd=10000,
            side="SELL",
            holding_hours=8,
            funding_rate=0.0001,
        )
        # Short receives: $10,000 * 0.0001 * 1 = $1
        assert cost > 0  # Positive = income

    def test_long_receives_negative_funding(self):
        """Long position should receive when funding rate is negative."""
        cost = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=8,
            funding_rate=-0.0001,
        )
        # Long pays negative rate = receives money
        assert cost > 0

    def test_multiple_funding_periods(self):
        """Holding through multiple funding periods multiplies payment."""
        single = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=8,
            funding_rate=0.0001,
        )
        double = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=16,  # Two periods
            funding_rate=0.0001,
        )
        assert abs(double) == pytest.approx(abs(single) * 2, rel=0.01)

    def test_partial_funding_period(self):
        """Partial period should be pro-rated."""
        full = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=8,
            funding_rate=0.0001,
        )
        half = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=4,  # Half period
            funding_rate=0.0001,
        )
        assert abs(half) == pytest.approx(abs(full) / 2, rel=0.01)

    def test_zero_holding_time(self):
        """Zero holding time should have zero funding cost."""
        cost = calculate_funding_cost(
            position_size_usd=10000,
            side="BUY",
            holding_hours=0,
            funding_rate=0.0001,
        )
        assert cost == 0.0


# =============================================================================
# FUNDING TIME TESTS
# =============================================================================


class TestGetNextFundingTime:
    """Tests for get_next_funding_time function."""

    def test_returns_datetime(self):
        """Should return a datetime object."""
        result = get_next_funding_time()
        assert isinstance(result, datetime)

    def test_next_funding_in_future(self):
        """Next funding time should be in the future."""
        result = get_next_funding_time()
        assert result > datetime.utcnow()

    def test_next_funding_within_8_hours(self):
        """Next funding should be within 8 hours."""
        result = get_next_funding_time()
        delta = result - datetime.utcnow()
        assert delta.total_seconds() <= 8 * 3600


class TestTimeToFunding:
    """Tests for time_to_funding function."""

    def test_returns_tuple(self):
        """Should return tuple of (hours, formatted_string)."""
        result = time_to_funding()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hours_positive(self):
        """Hours should be positive."""
        hours, _ = time_to_funding()
        assert hours > 0

    def test_hours_within_8(self):
        """Hours should be within 8."""
        hours, _ = time_to_funding()
        assert hours <= 8

    def test_formatted_string_contains_to_funding(self):
        """Formatted string should contain 'to funding'."""
        _, formatted = time_to_funding()
        assert "to funding" in formatted


# =============================================================================
# MARGIN CALCULATION TESTS
# =============================================================================


class TestCalculateInitialMargin:
    """Tests for calculate_initial_margin function."""

    def test_btc_initial_margin(self):
        """BTC initial margin calculation."""
        margin = calculate_initial_margin(position_size_usd=10000, symbol="BTC-USD")
        # Default 10% initial margin
        assert margin == pytest.approx(1000, rel=0.1)

    def test_symbol_without_suffix(self):
        """Symbol without suffix should work."""
        margin = calculate_initial_margin(position_size_usd=10000, symbol="BTC")
        assert margin > 0

    def test_margin_proportional_to_position(self):
        """Margin should be proportional to position size."""
        margin_1k = calculate_initial_margin(1000, "BTC")
        margin_10k = calculate_initial_margin(10000, "BTC")
        assert margin_10k == 10 * margin_1k


class TestCalculateMaintenanceMargin:
    """Tests for calculate_maintenance_margin function."""

    def test_maintenance_less_than_initial(self):
        """Maintenance margin should be less than initial margin."""
        initial = calculate_initial_margin(10000, "BTC")
        maintenance = calculate_maintenance_margin(10000, "BTC")
        assert maintenance < initial

    def test_btc_maintenance_margin(self):
        """BTC maintenance margin calculation."""
        margin = calculate_maintenance_margin(position_size_usd=10000, symbol="BTC-USD")
        # Default 5% maintenance margin
        assert margin == pytest.approx(500, rel=0.1)

    def test_margin_proportional_to_position(self):
        """Margin should be proportional to position size."""
        margin_1k = calculate_maintenance_margin(1000, "BTC")
        margin_10k = calculate_maintenance_margin(10000, "BTC")
        assert margin_10k == 10 * margin_1k


# =============================================================================
# LIQUIDATION PRICE TESTS
# =============================================================================


class TestCalculateLiquidationPrice:
    """Tests for calculate_liquidation_price function."""

    def test_long_liq_below_entry(self):
        """Long liquidation price should be below entry."""
        liq = calculate_liquidation_price(
            entry_price=90000,
            side="BUY",
            leverage=10,
            symbol="BTC",
        )
        assert liq < 90000

    def test_short_liq_above_entry(self):
        """Short liquidation price should be above entry."""
        liq = calculate_liquidation_price(
            entry_price=90000,
            side="SELL",
            leverage=10,
            symbol="BTC",
        )
        assert liq > 90000

    def test_higher_leverage_closer_liq(self):
        """Higher leverage should have liquidation closer to entry."""
        liq_5x = calculate_liquidation_price(90000, "BUY", leverage=5, symbol="BTC")
        liq_10x = calculate_liquidation_price(90000, "BUY", leverage=10, symbol="BTC")

        distance_5x = abs(90000 - liq_5x)
        distance_10x = abs(90000 - liq_10x)

        assert distance_10x < distance_5x

    def test_10x_long_approximately_10_percent(self):
        """10x long should liquidate at approximately 10% move."""
        liq = calculate_liquidation_price(90000, "BUY", leverage=10, symbol="BTC")
        # Distance should be around 5-10% (accounting for maintenance margin)
        distance_pct = (90000 - liq) / 90000
        assert 0.04 <= distance_pct <= 0.15

    def test_symmetry_for_short(self):
        """Short liquidation distance should be symmetric to long."""
        long_liq = calculate_liquidation_price(90000, "BUY", leverage=10, symbol="BTC")
        short_liq = calculate_liquidation_price(90000, "SELL", leverage=10, symbol="BTC")

        long_distance = 90000 - long_liq
        short_distance = short_liq - 90000

        assert long_distance == pytest.approx(short_distance, rel=0.01)


# =============================================================================
# SHOULD CLOSE BEFORE FUNDING TESTS
# =============================================================================


class TestShouldCloseBeforeFunding:
    """Tests for should_close_before_funding function."""

    def test_returns_tuple(self):
        """Should return tuple of (bool, reason)."""
        result = should_close_before_funding(
            entry_time=datetime.utcnow(),
            side="BUY",
            expected_funding_rate=0.0001,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_far_from_funding_no_close(self):
        """Far from funding should not recommend close."""
        # Mock time_to_funding to return > 0.5 hours
        with patch('crypto.trading.derivatives.time_to_funding') as mock_ttf:
            mock_ttf.return_value = (2.0, "2h to funding")
            should_close, reason = should_close_before_funding(
                entry_time=datetime.utcnow(),
                side="BUY",
                expected_funding_rate=0.0001,
            )
            assert should_close == False
            assert reason == ""

    def test_close_funding_long_positive_rate(self):
        """Close to funding with long + positive rate should recommend close."""
        with patch('crypto.trading.derivatives.time_to_funding') as mock_ttf:
            mock_ttf.return_value = (0.3, "18m to funding")
            should_close, reason = should_close_before_funding(
                entry_time=datetime.utcnow(),
                side="BUY",
                expected_funding_rate=0.0001,
            )
            assert should_close == True
            assert "long" in reason.lower()

    def test_close_funding_short_negative_rate(self):
        """Close to funding with short + negative rate should recommend close."""
        with patch('crypto.trading.derivatives.time_to_funding') as mock_ttf:
            mock_ttf.return_value = (0.3, "18m to funding")
            should_close, reason = should_close_before_funding(
                entry_time=datetime.utcnow(),
                side="SELL",
                expected_funding_rate=-0.0001,
            )
            assert should_close == True
            assert "short" in reason.lower()

    def test_close_funding_short_positive_rate_no_close(self):
        """Close to funding with short + positive rate should NOT recommend close."""
        with patch('crypto.trading.derivatives.time_to_funding') as mock_ttf:
            mock_ttf.return_value = (0.3, "18m to funding")
            should_close, reason = should_close_before_funding(
                entry_time=datetime.utcnow(),
                side="SELL",
                expected_funding_rate=0.0001,
            )
            # Short receives positive funding, so no need to close
            assert should_close == False


# =============================================================================
# EFFECTIVE LEVERAGE TESTS
# =============================================================================


class TestCalculateEffectiveLeverage:
    """Tests for calculate_effective_leverage function."""

    def test_basic_calculation(self):
        """Basic leverage calculation."""
        leverage = calculate_effective_leverage(
            account_equity=10000,
            position_size_usd=50000,
        )
        assert leverage == 5.0

    def test_10x_leverage(self):
        """10x leverage calculation."""
        leverage = calculate_effective_leverage(
            account_equity=1000,
            position_size_usd=10000,
        )
        assert leverage == 10.0

    def test_zero_equity_returns_zero(self):
        """Zero equity should return zero leverage."""
        leverage = calculate_effective_leverage(
            account_equity=0,
            position_size_usd=10000,
        )
        assert leverage == 0.0

    def test_negative_equity_returns_zero(self):
        """Negative equity should return zero leverage."""
        leverage = calculate_effective_leverage(
            account_equity=-1000,
            position_size_usd=10000,
        )
        assert leverage == 0.0

    def test_fractional_leverage(self):
        """Fractional leverage should be calculated correctly."""
        leverage = calculate_effective_leverage(
            account_equity=10000,
            position_size_usd=5000,
        )
        assert leverage == 0.5


# =============================================================================
# LEVERAGE SAFETY TESTS
# =============================================================================


class TestIsLeverageSafe:
    """Tests for is_leverage_safe function."""

    def test_returns_tuple(self):
        """Should return tuple of (is_safe, warning_message)."""
        result = is_leverage_safe(
            account_equity=10000,
            position_size_usd=50000,
            tier="intraday",
            symbol="BTC-USD",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_safe_leverage(self):
        """Leverage within limits should be safe."""
        is_safe, warning = is_leverage_safe(
            account_equity=10000,
            position_size_usd=50000,  # 5x leverage
            tier="intraday",
            symbol="BTC-USD",  # 10x max
        )
        assert is_safe == True

    def test_unsafe_leverage(self):
        """Leverage exceeding max should be unsafe."""
        is_safe, warning = is_leverage_safe(
            account_equity=10000,
            position_size_usd=150000,  # 15x leverage
            tier="intraday",
            symbol="BTC-USD",  # 10x max
        )
        assert is_safe == False
        assert "exceeds" in warning.lower()

    def test_warning_when_approaching_max(self):
        """Should warn when approaching max leverage."""
        is_safe, warning = is_leverage_safe(
            account_equity=10000,
            position_size_usd=85000,  # 8.5x leverage
            tier="intraday",
            symbol="BTC-USD",  # 10x max, 80% = 8x
        )
        assert is_safe == True  # Still safe
        assert "approaching" in warning.lower()

    def test_swing_tier_lower_max(self):
        """Swing tier should have lower max leverage."""
        # 5x should be safe for intraday (10x max) but unsafe for swing (4x max)
        intraday_safe, _ = is_leverage_safe(
            account_equity=10000,
            position_size_usd=50000,  # 5x
            tier="intraday",
            symbol="BTC-USD",
        )
        swing_safe, _ = is_leverage_safe(
            account_equity=10000,
            position_size_usd=50000,  # 5x
            tier="swing",
            symbol="BTC-USD",
        )
        assert intraday_safe == True
        assert swing_safe == False
