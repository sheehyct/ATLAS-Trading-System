"""
Stale 1H Position Tests - Session EQUITY-51

Tests the stale 1H position detection logic that ensures hourly trades
entered on a previous trading day are exited immediately, not at today's 15:59.

The NFLX Bug (Jan 7-8, 2026):
- Jan 7 10:48: NFLX 3-2D 1H Put entered
- Jan 7 15:59: Should have exited (same day EOD rule)
- Jan 8 15:59: Actually exited (WRONG - held overnight)

Root Cause: EOD exit logic used now.replace(hour=15, minute=59) which
always creates TODAY's 15:59, not checking if position was entered yesterday.

Fix: Check if entry_time was on a previous trading day, exit immediately.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pytz

from strat.signal_automation.position_monitor import PositionMonitor, MonitoringConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Create minimal config for position monitor."""
    return MonitoringConfig(
        eod_exit_hour=15,
        eod_exit_minute=59,
    )


@pytest.fixture
def monitor(mock_config):
    """Create position monitor instance for testing."""
    return PositionMonitor(
        config=mock_config,
        trading_client=None,
        signal_store=None,
        executor=None,
    )


# =============================================================================
# STALE 1H POSITION TESTS
# =============================================================================


class TestStale1HPositionDetection:
    """Tests for stale 1H position detection."""

    def test_same_day_entry_not_stale(self, monitor):
        """Position entered today should NOT be stale."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Entry 2 hours ago today
        entry_time = now - timedelta(hours=2)

        is_stale = monitor._is_stale_1h_position(entry_time)

        assert is_stale is False

    def test_previous_day_entry_is_stale(self, monitor):
        """Position entered yesterday should be stale."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Entry yesterday at same time
        entry_time = now - timedelta(days=1)

        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            # Mock schedule showing 2 trading days (entry day + today)
            mock_calendar = Mock()
            mock_schedule = Mock()
            mock_schedule.__len__ = Mock(return_value=2)
            mock_calendar.schedule.return_value = mock_schedule
            mock_get_calendar.return_value = mock_calendar

            is_stale = monitor._is_stale_1h_position(entry_time)

            assert is_stale is True

    def test_weekend_position_is_stale(self, monitor):
        """Position entered Friday but checked Monday should be stale."""
        et = pytz.timezone('America/New_York')

        # Mock: Friday 3pm
        friday = datetime(2026, 1, 9, 15, 0, tzinfo=et)  # Friday
        # Mock: Monday 10am
        monday = datetime(2026, 1, 12, 10, 0, tzinfo=et)  # Monday

        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            with patch('strat.signal_automation.position_monitor.datetime') as mock_datetime:
                mock_datetime.now.return_value = monday

                # Mock schedule showing 2 trading days (Friday + Monday)
                mock_calendar = Mock()
                mock_schedule = Mock()
                mock_schedule.__len__ = Mock(return_value=2)
                mock_calendar.schedule.return_value = mock_schedule
                mock_get_calendar.return_value = mock_calendar

                is_stale = monitor._is_stale_1h_position(friday)

                assert is_stale is True

    def test_naive_datetime_handled(self, monitor):
        """Entry time without timezone info should be handled gracefully."""
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Naive datetime from 2 days ago
        entry_time = datetime.now() - timedelta(days=2)

        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            # Mock schedule showing 3 trading days
            mock_calendar = Mock()
            mock_schedule = Mock()
            mock_schedule.__len__ = Mock(return_value=3)
            mock_calendar.schedule.return_value = mock_schedule
            mock_get_calendar.return_value = mock_calendar

            is_stale = monitor._is_stale_1h_position(entry_time)

            assert is_stale is True

    def test_timezone_aware_datetime_handled(self, monitor):
        """Entry time with timezone info should be handled correctly."""
        et = pytz.timezone('America/New_York')
        utc = pytz.UTC

        # Entry in UTC timezone from yesterday
        entry_time = datetime.now(utc) - timedelta(days=1)

        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            # Mock schedule showing 2 trading days
            mock_calendar = Mock()
            mock_schedule = Mock()
            mock_schedule.__len__ = Mock(return_value=2)
            mock_calendar.schedule.return_value = mock_schedule
            mock_get_calendar.return_value = mock_calendar

            is_stale = monitor._is_stale_1h_position(entry_time)

            assert is_stale is True


class TestStale1HPositionEdgeCases:
    """Edge case tests for stale 1H position detection."""

    def test_early_morning_same_day_not_stale(self, monitor):
        """Position entered at market open, checked end of day, not stale."""
        et = pytz.timezone('America/New_York')

        # Entry at 9:30 AM today, check at 3:30 PM today
        today = datetime.now(et).date()
        entry_time = et.localize(datetime.combine(today, datetime.strptime("09:30", "%H:%M").time()))

        is_stale = monitor._is_stale_1h_position(entry_time)

        assert is_stale is False

    def test_single_trading_day_not_stale(self, monitor):
        """When schedule shows only 1 trading day, position is not stale."""
        et = pytz.timezone('America/New_York')

        # Entry time from a few hours ago
        entry_time = datetime.now(et) - timedelta(hours=3)

        with patch('pandas_market_calendars.get_calendar') as mock_get_calendar:
            # Mock schedule showing only 1 trading day
            mock_calendar = Mock()
            mock_schedule = Mock()
            mock_schedule.__len__ = Mock(return_value=1)
            mock_calendar.schedule.return_value = mock_schedule
            mock_get_calendar.return_value = mock_calendar

            is_stale = monitor._is_stale_1h_position(entry_time)

            assert is_stale is False
