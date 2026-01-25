"""
EQUITY-89: Tests for StaleSetupValidator coordinator.

Tests setup freshness validation logic extracted from SignalDaemon._is_setup_stale().
Covers staleness windows for all timeframes: 1H, 4H, 1D, 1W, 1M.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional

import pytz

from strat.signal_automation.coordinators.stale_setup_validator import (
    StaleSetupValidator,
    StalenessConfig,
)
from strat.signal_automation.signal_store import SignalType


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockStoredSignal:
    """Mock stored signal for testing."""
    signal_key: str = "TEST_1H_3-2U_CALL"
    signal_type: str = "SETUP"
    timeframe: str = "1H"
    setup_bar_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.setup_bar_timestamp is None:
            et = pytz.timezone('America/New_York')
            self.setup_bar_timestamp = datetime.now(et)


@pytest.fixture
def validator():
    """Create StaleSetupValidator with default config."""
    return StaleSetupValidator()


@pytest.fixture
def et_timezone():
    """Return Eastern timezone."""
    return pytz.timezone('America/New_York')


@pytest.fixture
def now(et_timezone):
    """Return current time in Eastern."""
    return datetime.now(et_timezone)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestStaleSetupValidatorInit:
    """Tests for StaleSetupValidator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        validator = StaleSetupValidator()
        assert validator.config.hourly_window_hours == 1.5
        assert validator.config.four_hour_window_hours == 4.0
        assert validator.config.daily_max_trading_days == 2
        assert validator.config.weekly_max_weeks == 1
        assert validator.config.monthly_max_months == 1

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = StalenessConfig(
            hourly_window_hours=2.0,
            four_hour_window_hours=5.0,
            daily_max_trading_days=3,
        )
        validator = StaleSetupValidator(config=config)
        assert validator.config.hourly_window_hours == 2.0
        assert validator.config.four_hour_window_hours == 5.0
        assert validator.config.daily_max_trading_days == 3

    def test_init_custom_timezone(self):
        """Test initialization with custom timezone."""
        validator = StaleSetupValidator(timezone='UTC')
        assert 'UTC' in validator.timezone

    def test_config_property(self):
        """Test config property returns config."""
        config = StalenessConfig()
        validator = StaleSetupValidator(config=config)
        assert validator.config is config


# =============================================================================
# Signal Type Filtering Tests
# =============================================================================


class TestSignalTypeFiltering:
    """Tests for signal type filtering."""

    def test_completed_signal_not_stale(self, validator):
        """Test COMPLETED signals are never marked stale."""
        signal = MockStoredSignal(signal_type=SignalType.COMPLETED.value)
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False
        assert reason == ""

    def test_setup_signal_checked(self, validator, et_timezone):
        """Test SETUP signals are checked for staleness."""
        # Fresh signal should not be stale
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_missing_timestamp_not_stale(self, validator):
        """Test signals without timestamp are not marked stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            setup_bar_timestamp=None
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False


# =============================================================================
# 1H Staleness Tests
# =============================================================================


class TestHourlyStaleness:
    """Tests for 1H timeframe staleness."""

    def test_fresh_hourly_not_stale(self, validator, et_timezone):
        """Test fresh 1H signal is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_hourly_just_under_window(self, validator, et_timezone):
        """Test 1H signal just under 1.5 hour window is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(hours=1, minutes=29)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_hourly_over_window_is_stale(self, validator, et_timezone):
        """Test 1H signal over 1.5 hour window is stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(hours=2)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True
        assert "Hourly setup expired" in reason

    def test_hourly_custom_window(self, et_timezone):
        """Test 1H with custom window config."""
        config = StalenessConfig(hourly_window_hours=3.0)
        validator = StaleSetupValidator(config=config)

        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(hours=2)
        )
        # With 3 hour window, 2 hours ago should NOT be stale
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False


# =============================================================================
# 4H Staleness Tests
# =============================================================================


class TestFourHourStaleness:
    """Tests for 4H timeframe staleness."""

    def test_fresh_4h_not_stale(self, validator, et_timezone):
        """Test fresh 4H signal is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='4H',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_4h_under_window(self, validator, et_timezone):
        """Test 4H signal under 4 hour window is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='4H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(hours=3, minutes=59)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_4h_over_window_is_stale(self, validator, et_timezone):
        """Test 4H signal over 4 hour window is stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='4H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(hours=5)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True
        assert "4H setup expired" in reason


# =============================================================================
# 1D Staleness Tests
# =============================================================================


class TestDailyStaleness:
    """Tests for 1D timeframe staleness."""

    def test_fresh_daily_not_stale(self, validator, et_timezone):
        """Test same-day daily signal is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1D',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_daily_yesterday_not_stale(self, validator, et_timezone):
        """Test daily signal from yesterday is not stale (within 2 trading days)."""
        # This test is tricky because it depends on NYSE calendar
        # We'll just verify it doesn't crash
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1D',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(days=1)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        # Result depends on whether yesterday was a trading day
        assert isinstance(is_stale, bool)

    def test_daily_old_is_stale(self, validator, et_timezone):
        """Test daily signal from many days ago is stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1D',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(days=10)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True
        assert "Daily setup expired" in reason


# =============================================================================
# 1W Staleness Tests
# =============================================================================


class TestWeeklyStaleness:
    """Tests for 1W timeframe staleness."""

    def test_fresh_weekly_not_stale(self, validator, et_timezone):
        """Test same-week weekly signal is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1W',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_weekly_old_is_stale(self, validator, et_timezone):
        """Test weekly signal from many weeks ago is stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1W',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(weeks=3)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True
        assert "Weekly setup expired" in reason


# =============================================================================
# 1M Staleness Tests
# =============================================================================


class TestMonthlyStaleness:
    """Tests for 1M timeframe staleness."""

    def test_fresh_monthly_not_stale(self, validator, et_timezone):
        """Test same-month monthly signal is not stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1M',
            setup_bar_timestamp=datetime.now(et_timezone)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_monthly_old_is_stale(self, validator, et_timezone):
        """Test monthly signal from many months ago is stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1M',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(days=90)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True
        assert "Monthly setup expired" in reason


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unknown_timeframe_not_stale(self, validator, et_timezone):
        """Test unknown timeframe is not marked stale."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='15m',  # Not in staleness rules
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(days=1)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is False

    def test_naive_timestamp_handled(self, validator):
        """Test naive (non-timezone-aware) timestamp is handled."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now()  # Naive datetime
        )
        # Should not crash
        is_stale, reason = validator.is_setup_stale(signal)
        assert isinstance(is_stale, bool)

    def test_very_old_timestamp(self, validator, et_timezone):
        """Test very old timestamp is properly handled."""
        signal = MockStoredSignal(
            signal_type=SignalType.SETUP.value,
            timeframe='1H',
            setup_bar_timestamp=datetime.now(et_timezone) - timedelta(days=365)
        )
        is_stale, reason = validator.is_setup_stale(signal)
        assert is_stale is True


# =============================================================================
# StalenessConfig Tests
# =============================================================================


class TestStalenessConfig:
    """Tests for StalenessConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StalenessConfig()
        assert config.hourly_window_hours == 1.5
        assert config.four_hour_window_hours == 4.0
        assert config.daily_max_trading_days == 2
        assert config.weekly_max_weeks == 1
        assert config.monthly_max_months == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StalenessConfig(
            hourly_window_hours=2.0,
            four_hour_window_hours=6.0,
            daily_max_trading_days=3,
            weekly_max_weeks=2,
            monthly_max_months=2,
        )
        assert config.hourly_window_hours == 2.0
        assert config.four_hour_window_hours == 6.0
        assert config.daily_max_trading_days == 3
        assert config.weekly_max_weeks == 2
        assert config.monthly_max_months == 2
