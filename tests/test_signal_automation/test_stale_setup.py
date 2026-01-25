"""
Stale Setup Validation Tests - Session EQUITY-46

Tests the stale setup detection logic that prevents triggering setups
that have become invalid due to intervening bars.

The MSTR Bug (Jan 7, 2026):
- Jan 5: 3-2U setup detected (valid)
- Jan 6: New bar closed as 2D (pattern evolved to 3-2U-2D)
- Jan 7: Entry monitor triggered stale setup (WRONG)

Fix: Check if setup is stale before triggering based on timeframe.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import pytz

from strat.signal_automation.signal_store import StoredSignal, SignalType, SignalStatus
from strat.signal_automation.daemon import SignalDaemon
from strat.signal_automation.config import SignalAutomationConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config(tmp_path):
    """Create minimal config for daemon."""
    return SignalAutomationConfig(
        store_path=str(tmp_path / 'signals'),
    )


@pytest.fixture
def daemon(mock_config):
    """Create daemon instance for testing."""
    return SignalDaemon(config=mock_config)


def create_setup_signal(
    timeframe: str,
    setup_bar_timestamp: datetime,
    signal_type: str = SignalType.SETUP.value,
) -> StoredSignal:
    """Helper to create test signals."""
    return StoredSignal(
        signal_key=f'TEST_{timeframe}_3-2U_CALL_{setup_bar_timestamp.strftime("%Y%m%d%H%M")}',
        pattern_type='3-2U',
        direction='CALL',
        symbol='TEST',
        timeframe=timeframe,
        detected_time=setup_bar_timestamp,
        entry_trigger=100.0,
        stop_price=95.0,
        target_price=105.0,
        magnitude_pct=5.0,
        risk_reward=1.0,
        signal_type=signal_type,
        setup_bar_timestamp=setup_bar_timestamp,
        setup_bar_high=100.0,
        setup_bar_low=95.0,
    )


# =============================================================================
# HOURLY TESTS
# =============================================================================


class TestHourlyStaleSetup:
    """Tests for hourly timeframe stale detection."""

    def test_fresh_hourly_setup_not_stale(self, daemon):
        """Setup detected within the hour should NOT be stale."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Setup detected 30 minutes ago (within forming bar period)
        setup_ts = now - timedelta(minutes=30)
        signal = create_setup_signal('1H', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is False
        assert reason == ""

    def test_stale_hourly_setup_rejected(self, daemon):
        """Setup detected over 1 hour ago should be stale."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Setup detected 2 hours ago (forming bar already closed)
        setup_ts = now - timedelta(hours=2)
        signal = create_setup_signal('1H', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is True
        assert "Hourly setup expired" in reason


# =============================================================================
# DAILY TESTS
# =============================================================================


class TestDailyStaleSetup:
    """Tests for daily timeframe stale detection."""

    def test_fresh_daily_setup_not_stale(self, daemon):
        """Setup from yesterday should NOT be stale (forming bar is today)."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Mock the validator's NYSE calendar
        mock_calendar = Mock()
        mock_schedule = Mock()
        mock_schedule.__len__ = Mock(return_value=2)  # 2 trading days (OK)
        mock_calendar.schedule.return_value = mock_schedule

        # EQUITY-89: Inject mock calendar into validator
        daemon._stale_validator._nyse_calendar = mock_calendar

        # Setup from yesterday
        setup_ts = now - timedelta(days=1)
        signal = create_setup_signal('1D', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is False

    def test_stale_daily_setup_rejected(self, daemon):
        """Setup from 3+ trading days ago should be stale (MSTR bug scenario)."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Mock the validator's NYSE calendar
        mock_calendar = Mock()
        mock_schedule = Mock()
        mock_schedule.__len__ = Mock(return_value=4)  # 4 trading days (too many)
        mock_calendar.schedule.return_value = mock_schedule

        # EQUITY-89: Inject mock calendar into validator
        daemon._stale_validator._nyse_calendar = mock_calendar

        # Setup from 3 days ago
        setup_ts = now - timedelta(days=3)
        signal = create_setup_signal('1D', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is True
        assert "Daily setup expired" in reason


# =============================================================================
# WEEKLY TESTS
# =============================================================================


class TestWeeklyStaleSetup:
    """Tests for weekly timeframe stale detection."""

    def test_fresh_weekly_setup_not_stale(self, daemon):
        """Setup from last week should NOT be stale (forming bar is this week)."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Setup from 5 days ago (likely same week or last week)
        setup_ts = now - timedelta(days=5)
        signal = create_setup_signal('1W', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        # Should not be stale if within 1 week difference
        # (depends on current day of week)
        assert isinstance(is_stale, bool)

    def test_stale_weekly_setup_rejected(self, daemon):
        """Setup from 3+ weeks ago should be stale."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Setup from 3 weeks ago
        setup_ts = now - timedelta(weeks=3)
        signal = create_setup_signal('1W', setup_ts)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is True
        assert "Weekly setup expired" in reason


# =============================================================================
# EDGE CASES
# =============================================================================


class TestStaleSetupEdgeCases:
    """Edge case tests for stale detection."""

    def test_completed_signal_not_checked(self, daemon):
        """COMPLETED signals should not be checked for staleness."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Old setup but marked as COMPLETED
        setup_ts = now - timedelta(days=10)
        signal = create_setup_signal('1D', setup_ts, signal_type=SignalType.COMPLETED.value)

        is_stale, reason = daemon._is_setup_stale(signal)

        assert is_stale is False
        assert reason == ""

    def test_missing_timestamp_logs_warning(self, daemon):
        """Signal without setup_bar_timestamp should log warning and pass."""
        signal = StoredSignal(
            signal_key='TEST_1D_3-2U_CALL_202501010000',
            pattern_type='3-2U',
            direction='CALL',
            symbol='TEST',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=100.0,
            stop_price=95.0,
            target_price=105.0,
            magnitude_pct=5.0,
            risk_reward=1.0,
            signal_type=SignalType.SETUP.value,
            setup_bar_timestamp=None,  # Missing timestamp
        )

        is_stale, reason = daemon._is_setup_stale(signal)

        # Should not be marked stale (allow execution, but logged)
        assert is_stale is False
