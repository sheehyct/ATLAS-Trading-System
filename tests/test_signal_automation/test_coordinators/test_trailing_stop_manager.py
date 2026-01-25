"""
Tests for TrailingStopManager coordinator.

EQUITY-90: Phase 4.2 - Extracted trailing stop logic from PositionMonitor.
"""

import pytest
from datetime import datetime, timedelta

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)
from strat.signal_automation.coordinators.trailing_stop_manager import TrailingStopManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default monitoring config."""
    return MonitoringConfig()


@pytest.fixture
def manager(config):
    """TrailingStopManager with default config."""
    return TrailingStopManager(config=config)


@pytest.fixture
def call_position():
    """Basic bullish position for testing."""
    return TrackedPosition(
        osi_symbol='SPY250120C00450000',
        signal_key='SPY_1D_2U_test',
        symbol='SPY',
        direction='CALL',
        entry_trigger=450.0,
        target_price=460.0,
        stop_price=445.0,
        pattern_type='2-1-2U',
        timeframe='1D',
        entry_price=5.50,
        contracts=2,
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        current_price=6.00,
        unrealized_pnl=100.0,
        unrealized_pct=0.10,
        underlying_price=455.0,
        dte=10,
        target_1x=455.0,
        original_target=460.0,
        actual_entry_underlying=450.0,
        contracts_remaining=2,
        high_water_mark=0.0,  # Not yet set
        trailing_stop_active=False,
        trailing_stop_price=0.0,
        use_atr_trailing=False,
        atr_at_detection=0.0,
        atr_trail_distance=0.0,
    )


@pytest.fixture
def put_position():
    """Basic bearish position for testing."""
    return TrackedPosition(
        osi_symbol='SPY250120P00450000',
        signal_key='SPY_1D_2D_test',
        symbol='SPY',
        direction='PUT',
        entry_trigger=450.0,
        target_price=440.0,
        stop_price=455.0,
        pattern_type='2-1-2D',
        timeframe='1D',
        entry_price=5.50,
        contracts=2,
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        current_price=6.00,
        unrealized_pnl=100.0,
        unrealized_pct=0.10,
        underlying_price=445.0,
        dte=10,
        target_1x=445.0,
        original_target=440.0,
        actual_entry_underlying=450.0,
        contracts_remaining=2,
        high_water_mark=0.0,
        trailing_stop_active=False,
        trailing_stop_price=0.0,
        use_atr_trailing=False,
        atr_at_detection=0.0,
        atr_trail_distance=0.0,
    )


@pytest.fixture
def atr_call_position():
    """3-2 pattern position with ATR trailing stop."""
    return TrackedPosition(
        osi_symbol='SPY250120C00450000',
        signal_key='SPY_1D_32U_test',
        symbol='SPY',
        direction='CALL',
        entry_trigger=450.0,
        target_price=456.75,  # 1.5% for 3-2
        stop_price=445.0,
        pattern_type='3-2U',
        timeframe='1D',
        entry_price=5.50,
        contracts=2,
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        current_price=6.00,
        unrealized_pnl=100.0,
        unrealized_pct=0.10,
        underlying_price=455.0,
        dte=10,
        target_1x=455.0,
        original_target=456.75,
        actual_entry_underlying=450.0,
        contracts_remaining=2,
        high_water_mark=0.0,
        trailing_stop_active=False,
        trailing_stop_price=0.0,
        use_atr_trailing=True,
        atr_at_detection=4.0,  # $4 ATR
        atr_trail_distance=4.0,  # 1.0 ATR pre-calculated
        atr_activation_level=453.0,  # 0.75 ATR = $3 profit
    )


# =============================================================================
# Test TrailingStopManager Initialization
# =============================================================================


class TestTrailingStopManagerInit:
    """Tests for TrailingStopManager initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = MonitoringConfig()
        manager = TrailingStopManager(config=config)
        assert manager.config is config

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = MonitoringConfig(
            trailing_stop_activation_rr=0.75,
            trailing_stop_pct=0.40,
        )
        manager = TrailingStopManager(config=config)
        assert manager.config.trailing_stop_activation_rr == 0.75
        assert manager.config.trailing_stop_pct == 0.40


# =============================================================================
# Test Percentage-Based Trailing Stop (Non-3-2 Patterns)
# =============================================================================


class TestPercentageTrailingStop:
    """Tests for percentage-based trailing stop logic."""

    def test_updates_high_water_mark_call(self, manager, call_position):
        """Test high water mark update for calls."""
        call_position.underlying_price = 455.0
        manager.check(call_position)
        assert call_position.high_water_mark == 455.0

        # Price goes higher
        call_position.underlying_price = 457.0
        manager.check(call_position)
        assert call_position.high_water_mark == 457.0

        # Price drops - HWM should not change
        call_position.underlying_price = 456.0
        manager.check(call_position)
        assert call_position.high_water_mark == 457.0

    def test_updates_high_water_mark_put(self, manager, put_position):
        """Test high water mark update for puts (lowest price)."""
        put_position.underlying_price = 445.0
        manager.check(put_position)
        assert put_position.high_water_mark == 445.0

        # Price goes lower (better for puts)
        put_position.underlying_price = 443.0
        manager.check(put_position)
        assert put_position.high_water_mark == 443.0

        # Price rises - HWM should not change
        put_position.underlying_price = 444.0
        manager.check(put_position)
        assert put_position.high_water_mark == 443.0

    def test_activates_at_0_5x_rr_call(self, manager, call_position):
        """Test trailing stop activates at 0.5x R:R for calls."""
        # Risk = |450 - 445| = $5
        # Activation = 0.5 * $5 = $2.50 profit -> price $452.50
        call_position.underlying_price = 452.50
        manager.check(call_position)
        assert call_position.trailing_stop_active is True

    def test_does_not_activate_below_threshold(self, manager, call_position):
        """Test trailing stop does not activate below threshold."""
        # Need $2.50 profit, only at $2
        call_position.underlying_price = 452.0
        manager.check(call_position)
        assert call_position.trailing_stop_active is False

    def test_activates_at_0_5x_rr_put(self, manager, put_position):
        """Test trailing stop activates at 0.5x R:R for puts."""
        # Risk = |450 - 455| = $5
        # Activation = 0.5 * $5 = $2.50 profit -> price $447.50
        put_position.underlying_price = 447.50
        manager.check(put_position)
        assert put_position.trailing_stop_active is True

    def test_trailing_stop_hit_call(self, manager, call_position):
        """Test trailing stop hit for calls."""
        # Activate trailing stop
        call_position.underlying_price = 455.0  # $5 profit, well above 0.5x R:R
        manager.check(call_position)
        assert call_position.trailing_stop_active is True
        assert call_position.high_water_mark == 455.0

        # Trail = HWM - (50% of profit)
        # Profit from entry = 455 - 450 = $5
        # Trail amount = $5 * 0.5 = $2.50
        # Trail price = 455 - 2.50 = $452.50
        assert call_position.trailing_stop_price == 452.50

        # Price drops to trail level
        call_position.underlying_price = 452.0
        call_position.unrealized_pct = 0.05  # Still in profit
        result = manager.check(call_position)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_STOP
        assert '452.50' in result.details

    def test_trailing_stop_hit_put(self, manager, put_position):
        """Test trailing stop hit for puts."""
        # Activate trailing stop
        put_position.underlying_price = 445.0  # $5 profit
        manager.check(put_position)
        assert put_position.trailing_stop_active is True
        assert put_position.high_water_mark == 445.0

        # Trail = HWM + (50% of profit)
        # Profit from entry = 450 - 445 = $5
        # Trail amount = $5 * 0.5 = $2.50
        # Trail price = 445 + 2.50 = $447.50

        # Price rises to trail level
        put_position.underlying_price = 448.0
        put_position.unrealized_pct = 0.05
        result = manager.check(put_position)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_STOP

    def test_trailing_stop_blocked_if_no_profit(self, manager, call_position):
        """Test trailing stop blocked if option P/L negative."""
        # Activate trailing stop
        call_position.underlying_price = 455.0
        manager.check(call_position)
        assert call_position.trailing_stop_active is True

        # Price drops to trail level but option at loss
        call_position.underlying_price = 452.0
        call_position.unrealized_pct = -0.05  # 5% loss
        result = manager.check(call_position)
        assert result is None  # Blocked


# =============================================================================
# Test ATR-Based Trailing Stop (3-2 Patterns)
# =============================================================================


class TestATRTrailingStop:
    """Tests for ATR-based trailing stop logic."""

    def test_routes_to_atr_trailing(self, manager, atr_call_position):
        """Test that 3-2 patterns use ATR trailing."""
        assert atr_call_position.use_atr_trailing is True
        manager.check(atr_call_position)
        # Should not crash, just update HWM

    def test_activates_at_0_75_atr(self, manager, atr_call_position):
        """Test ATR trailing activates at 0.75 ATR profit."""
        # ATR = $4, activation = 0.75 * $4 = $3 profit -> price $453
        atr_call_position.underlying_price = 453.0
        manager.check(atr_call_position)
        assert atr_call_position.trailing_stop_active is True

    def test_does_not_activate_below_threshold(self, manager, atr_call_position):
        """Test ATR trailing does not activate below 0.75 ATR."""
        atr_call_position.underlying_price = 452.5  # Only $2.50 profit, need $3
        manager.check(atr_call_position)
        assert atr_call_position.trailing_stop_active is False

    def test_atr_trailing_stop_hit(self, manager, atr_call_position):
        """Test ATR trailing stop hit."""
        # Activate and set HWM
        atr_call_position.underlying_price = 456.0  # $6 profit
        manager.check(atr_call_position)
        assert atr_call_position.trailing_stop_active is True
        assert atr_call_position.high_water_mark == 456.0

        # Trail = HWM - 1.0 ATR = 456 - 4 = $452
        assert atr_call_position.trailing_stop_price == 452.0

        # Price drops to trail level
        atr_call_position.underlying_price = 451.5
        atr_call_position.unrealized_pct = 0.03  # Small profit
        result = manager.check(atr_call_position)
        assert result is not None
        assert result.reason == ExitReason.TRAILING_STOP
        assert 'ATR' in result.details

    def test_atr_trailing_not_hit_above_trail(self, manager, atr_call_position):
        """Test no exit when price above ATR trail."""
        atr_call_position.underlying_price = 456.0
        manager.check(atr_call_position)

        # Price drops but stays above trail (452)
        atr_call_position.underlying_price = 453.0
        result = manager.check(atr_call_position)
        assert result is None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestTrailingStopEdgeCases:
    """Tests for edge cases in trailing stop logic."""

    def test_uses_actual_entry_not_trigger(self, manager, call_position):
        """Test uses actual_entry_underlying for gap-through scenarios."""
        # Gap through: trigger was $450, actual entry was $448
        call_position.actual_entry_underlying = 448.0
        call_position.underlying_price = 450.5  # $2.50 profit from actual entry

        # Risk = |448 - 445| = $3
        # Activation = 0.5 * $3 = $1.50 profit -> price $449.50
        # At $450.50, profit is $2.50 which exceeds $1.50
        manager.check(call_position)
        assert call_position.trailing_stop_active is True

    def test_falls_back_to_trigger_if_no_actual_entry(self, manager, call_position):
        """Test falls back to entry_trigger if actual_entry_underlying is 0."""
        call_position.actual_entry_underlying = 0.0
        call_position.underlying_price = 452.50  # $2.50 profit from trigger

        # Uses entry_trigger = $450
        # Risk = |450 - 445| = $5
        # Activation = 0.5 * $5 = $2.50
        manager.check(call_position)
        assert call_position.trailing_stop_active is True

    def test_hwm_initialized_on_first_check(self, manager, call_position):
        """Test high water mark is initialized on first check."""
        assert call_position.high_water_mark == 0.0
        manager.check(call_position)
        assert call_position.high_water_mark == 455.0

    def test_no_exit_if_not_active(self, manager, call_position):
        """Test no exit signal if trailing stop not active."""
        call_position.underlying_price = 450.0  # At entry, no profit
        result = manager.check(call_position)
        assert result is None
        assert call_position.trailing_stop_active is False

    def test_custom_activation_threshold(self, call_position):
        """Test custom activation threshold."""
        config = MonitoringConfig(trailing_stop_activation_rr=1.0)  # 1.0x R:R
        manager = TrailingStopManager(config=config)

        # Risk = $5, need 1.0 * $5 = $5 profit -> price $455
        call_position.underlying_price = 454.0  # Only $4 profit
        manager.check(call_position)
        assert call_position.trailing_stop_active is False

        call_position.underlying_price = 455.0  # $5 profit
        manager.check(call_position)
        assert call_position.trailing_stop_active is True

    def test_custom_trail_percentage(self, call_position):
        """Test custom trail percentage."""
        config = MonitoringConfig(trailing_stop_pct=0.25)  # 25% trail
        manager = TrailingStopManager(config=config)

        # Activate
        call_position.underlying_price = 455.0
        manager.check(call_position)

        # Trail = HWM - (25% of profit) = 455 - (5 * 0.25) = 455 - 1.25 = $453.75
        assert call_position.trailing_stop_price == 453.75

    def test_minimum_profit_threshold(self, call_position):
        """Test minimum profit threshold blocks exit."""
        config = MonitoringConfig(trailing_stop_min_profit_pct=0.10)  # 10% min
        manager = TrailingStopManager(config=config)

        # Activate
        call_position.underlying_price = 455.0
        manager.check(call_position)

        # Hit trail but option only at 5% profit
        call_position.underlying_price = 452.0
        call_position.unrealized_pct = 0.05  # 5% < 10% min
        result = manager.check(call_position)
        assert result is None  # Blocked

        # At 10% profit, should exit
        call_position.unrealized_pct = 0.10
        result = manager.check(call_position)
        assert result is not None
