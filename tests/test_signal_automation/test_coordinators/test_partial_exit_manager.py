"""
Tests for PartialExitManager coordinator.

EQUITY-90: Phase 4.3 - Extracted partial exit logic from PositionMonitor.
"""

import pytest
from datetime import datetime, timedelta

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitReason,
    MonitoringConfig,
)
from strat.signal_automation.coordinators.partial_exit_manager import PartialExitManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default monitoring config."""
    return MonitoringConfig()


@pytest.fixture
def manager(config):
    """PartialExitManager with default config."""
    return PartialExitManager(config=config)


@pytest.fixture
def call_position():
    """Basic multi-contract bullish position."""
    return TrackedPosition(
        osi_symbol='SPY250120C00450000',
        signal_key='SPY_1D_2U_test',
        symbol='SPY',
        direction='CALL',
        entry_trigger=450.0,
        target_price=460.0,
        target_1x=455.0,  # 1.0x R:R target
        stop_price=445.0,
        pattern_type='2-1-2U',
        timeframe='1D',
        entry_price=5.50,
        contracts=4,  # Multi-contract
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        underlying_price=452.0,
        partial_exit_done=False,
    )


@pytest.fixture
def put_position():
    """Basic multi-contract bearish position."""
    return TrackedPosition(
        osi_symbol='SPY250120P00450000',
        signal_key='SPY_1D_2D_test',
        symbol='SPY',
        direction='PUT',
        entry_trigger=450.0,
        target_price=440.0,
        target_1x=445.0,  # 1.0x R:R target
        stop_price=455.0,
        pattern_type='2-1-2D',
        timeframe='1D',
        entry_price=5.50,
        contracts=4,
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        underlying_price=448.0,
        partial_exit_done=False,
    )


# =============================================================================
# Test PartialExitManager Initialization
# =============================================================================


class TestPartialExitManagerInit:
    """Tests for PartialExitManager initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = MonitoringConfig()
        manager = PartialExitManager(config=config)
        assert manager.config is config

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = MonitoringConfig(partial_exit_pct=0.75)  # 75% instead of 50%
        manager = PartialExitManager(config=config)
        assert manager.config.partial_exit_pct == 0.75


# =============================================================================
# Test Skip Conditions
# =============================================================================


class TestPartialExitSkipConditions:
    """Tests for conditions that skip partial exit."""

    def test_skips_single_contract(self, manager, call_position):
        """Test partial exit skips positions with 1 contract."""
        call_position.contracts = 1
        call_position.underlying_price = 456.0  # Above target_1x
        result = manager.check(call_position)
        assert result is None

    def test_skips_if_already_done(self, manager, call_position):
        """Test partial exit skips if already done."""
        call_position.partial_exit_done = True
        call_position.underlying_price = 456.0  # Above target_1x
        result = manager.check(call_position)
        assert result is None


# =============================================================================
# Test Call Partial Exit
# =============================================================================


class TestCallPartialExit:
    """Tests for CALL partial exit logic."""

    def test_triggers_above_target_1x(self, manager, call_position):
        """Test partial exit triggers when CALL price above target_1x."""
        call_position.underlying_price = 456.0  # Above 455 target_1x
        result = manager.check(call_position)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT
        assert result.contracts_to_close == 2  # 50% of 4

    def test_triggers_at_exact_target_1x(self, manager, call_position):
        """Test partial exit triggers at exact target_1x."""
        call_position.underlying_price = 455.0  # At target_1x
        result = manager.check(call_position)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT

    def test_does_not_trigger_below_target_1x(self, manager, call_position):
        """Test partial exit does not trigger below target_1x."""
        call_position.underlying_price = 454.0  # Below 455 target_1x
        result = manager.check(call_position)
        assert result is None


# =============================================================================
# Test Put Partial Exit
# =============================================================================


class TestPutPartialExit:
    """Tests for PUT partial exit logic."""

    def test_triggers_below_target_1x(self, manager, put_position):
        """Test partial exit triggers when PUT price below target_1x."""
        put_position.underlying_price = 444.0  # Below 445 target_1x
        result = manager.check(put_position)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT
        assert result.contracts_to_close == 2  # 50% of 4

    def test_triggers_at_exact_target_1x(self, manager, put_position):
        """Test partial exit triggers at exact target_1x."""
        put_position.underlying_price = 445.0  # At target_1x
        result = manager.check(put_position)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT

    def test_does_not_trigger_above_target_1x(self, manager, put_position):
        """Test partial exit does not trigger above target_1x."""
        put_position.underlying_price = 446.0  # Above 445 target_1x
        result = manager.check(put_position)
        assert result is None


# =============================================================================
# Test Contract Calculation
# =============================================================================


class TestContractCalculation:
    """Tests for contracts to close calculation."""

    def test_50_percent_of_4_contracts(self, manager, call_position):
        """Test 50% of 4 contracts = 2."""
        call_position.contracts = 4
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert result.contracts_to_close == 2

    def test_50_percent_of_3_contracts_rounds_up(self, manager, call_position):
        """Test 50% of 3 contracts rounds up to 2."""
        call_position.contracts = 3
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert result.contracts_to_close == 2  # 1.5 rounds to 2

    def test_50_percent_of_2_contracts(self, manager, call_position):
        """Test 50% of 2 contracts = 1."""
        call_position.contracts = 2
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert result.contracts_to_close == 1

    def test_minimum_1_contract(self, manager, call_position):
        """Test minimum 1 contract is closed."""
        call_position.contracts = 2
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert result.contracts_to_close >= 1

    def test_custom_partial_exit_pct(self, call_position):
        """Test custom partial exit percentage."""
        config = MonitoringConfig(partial_exit_pct=0.75)  # 75%
        manager = PartialExitManager(config=config)
        call_position.contracts = 4
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert result.contracts_to_close == 3  # 75% of 4


# =============================================================================
# Test Exit Signal Details
# =============================================================================


class TestExitSignalDetails:
    """Tests for exit signal details."""

    def test_signal_has_correct_fields(self, manager, call_position):
        """Test exit signal has all required fields."""
        call_position.underlying_price = 456.0
        result = manager.check(call_position)

        assert result.osi_symbol == call_position.osi_symbol
        assert result.signal_key == call_position.signal_key
        assert result.reason == ExitReason.PARTIAL_EXIT
        assert result.underlying_price == 456.0
        assert result.contracts_to_close == 2

    def test_signal_details_include_target(self, manager, call_position):
        """Test exit signal details include target info."""
        call_position.underlying_price = 456.0
        result = manager.check(call_position)
        assert '1.0x R:R' in result.details
        assert '$455.00' in result.details  # target_1x
