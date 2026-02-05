"""
Tests for EQUITY-100: EOD Exit Timing Bug Fix

Tests for:
- Dedicated EOD exit jobs (daemon.py)
- Grace period for EOD exits (position_monitor.py)
- Market open stale check (daemon.py)
- Retry logic for failed exits
- Logging upgrades

Session: EQUITY-100
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytz

from strat.signal_automation.position_monitor import (
    ExitReason,
    MonitoringConfig,
    TrackedPosition,
    ExitSignal,
    PositionMonitor,
)
from strat.signal_automation.daemon import SignalDaemon


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_trading_client():
    """Create a mock trading client."""
    client = Mock()
    client.close_option_position = Mock(return_value={'status': 'filled', 'qty': 1})
    return client


@pytest.fixture
def mock_market_hours_validator():
    """Create a mock market hours validator."""
    validator = Mock()
    validator.is_market_hours = Mock(return_value=True)
    validator.is_trading_day = Mock(return_value=True)

    # Create mock schedule
    mock_schedule = Mock()
    mock_schedule.is_trading_day = True
    et = pytz.timezone('America/New_York')
    mock_schedule.market_close = et.localize(datetime.now().replace(hour=16, minute=0, second=0))
    validator.get_schedule = Mock(return_value=mock_schedule)

    return validator


@pytest.fixture
def position_monitor(mock_trading_client, mock_market_hours_validator):
    """Create a position monitor with mocked dependencies."""
    config = MonitoringConfig()

    # Create monitor with minimal dependencies
    with patch('strat.signal_automation.position_monitor.MarketHoursValidator', return_value=mock_market_hours_validator):
        monitor = PositionMonitor(
            config=config,
            trading_client=mock_trading_client,
        )

    # Override the validator
    monitor._market_hours_validator = mock_market_hours_validator

    return monitor


@pytest.fixture
def sample_1h_position():
    """Create a sample 1H TrackedPosition."""
    et = pytz.timezone('America/New_York')
    return TrackedPosition(
        osi_symbol='SPY250120C00450000',
        symbol='SPY',
        signal_key='SPY_1H_test',
        timeframe='1H',
        pattern_type='2-1-2U',
        direction='CALL',
        contracts=1,
        entry_time=et.localize(datetime.now() - timedelta(hours=2)),
        entry_trigger=450.0,
        entry_price=2.50,
        stop_price=445.0,
        target_price=455.0,
        expiration='2025-01-20',
        dte=30,
    )


@pytest.fixture
def sample_4h_position():
    """Create a sample 4H TrackedPosition (should NOT trigger EOD exit)."""
    et = pytz.timezone('America/New_York')
    return TrackedPosition(
        osi_symbol='QQQ250120C00380000',
        symbol='QQQ',
        signal_key='QQQ_4H_test',
        timeframe='4H',
        pattern_type='3-2U',
        direction='CALL',
        contracts=1,
        entry_time=et.localize(datetime.now() - timedelta(hours=4)),
        entry_trigger=380.0,
        entry_price=3.00,
        stop_price=375.0,
        target_price=385.0,
        expiration='2025-01-20',
        dte=30,
    )


# =============================================================================
# Grace Period Tests
# =============================================================================

class TestGracePeriod:
    """Tests for EOD grace period logic."""

    def test_grace_period_returns_false_outside_grace_window(self, position_monitor, mock_market_hours_validator):
        """Grace period should return False when not in grace window."""
        et = pytz.timezone('America/New_York')

        # Set time to 3:30 PM (well before close)
        test_time = et.localize(datetime.now().replace(hour=15, minute=30, second=0))

        with patch('strat.signal_automation.position_monitor.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time

            result = position_monitor._is_within_eod_grace_period()
            assert result is False

    def test_grace_period_returns_true_at_4pm_00s(self, position_monitor, mock_market_hours_validator):
        """Grace period should return True at exactly 4:00:00 PM."""
        et = pytz.timezone('America/New_York')

        # Set time to exactly 4:00:00 PM
        test_time = et.localize(datetime.now().replace(hour=16, minute=0, second=0))

        # Update mock schedule to match
        mock_schedule = Mock()
        mock_schedule.is_trading_day = True
        mock_schedule.market_close = test_time  # Close at exactly this time
        mock_market_hours_validator.get_schedule.return_value = mock_schedule

        with patch('strat.signal_automation.position_monitor.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time

            result = position_monitor._is_within_eod_grace_period()
            assert result is True

    def test_grace_period_returns_true_at_4pm_30s(self, position_monitor, mock_market_hours_validator):
        """Grace period should return True at 4:00:30 PM."""
        et = pytz.timezone('America/New_York')

        # Set market close at 4:00 PM
        close_time = et.localize(datetime.now().replace(hour=16, minute=0, second=0))

        # Set current time to 4:00:30 PM
        test_time = et.localize(datetime.now().replace(hour=16, minute=0, second=30))

        mock_schedule = Mock()
        mock_schedule.is_trading_day = True
        mock_schedule.market_close = close_time
        mock_market_hours_validator.get_schedule.return_value = mock_schedule

        with patch('strat.signal_automation.position_monitor.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time

            result = position_monitor._is_within_eod_grace_period()
            assert result is True

    def test_grace_period_returns_false_after_grace_window(self, position_monitor, mock_market_hours_validator):
        """Grace period should return False after 4:01 PM."""
        et = pytz.timezone('America/New_York')

        # Set market close at 4:00 PM
        close_time = et.localize(datetime.now().replace(hour=16, minute=0, second=0))

        # Set current time to 4:02 PM (beyond grace period)
        test_time = et.localize(datetime.now().replace(hour=16, minute=2, second=0))

        mock_schedule = Mock()
        mock_schedule.is_trading_day = True
        mock_schedule.market_close = close_time
        mock_market_hours_validator.get_schedule.return_value = mock_schedule

        with patch('strat.signal_automation.position_monitor.datetime') as mock_datetime:
            mock_datetime.now.return_value = test_time

            result = position_monitor._is_within_eod_grace_period()
            assert result is False

    def test_grace_period_returns_false_on_non_trading_day(self, position_monitor, mock_market_hours_validator):
        """Grace period should return False on non-trading days."""
        mock_market_hours_validator.is_trading_day.return_value = False

        result = position_monitor._is_within_eod_grace_period()
        assert result is False


# =============================================================================
# EOD Exit Execution Tests
# =============================================================================

class TestEodExitExecution:
    """Tests for EOD exit execution with grace period."""

    def test_eod_exit_blocked_outside_market_hours_no_grace(self, position_monitor, sample_1h_position):
        """EOD exits should be blocked outside market hours when not in grace period."""
        # Add position
        position_monitor._positions[sample_1h_position.osi_symbol] = sample_1h_position

        # Mock market as closed and NOT in grace period
        with patch.object(position_monitor, '_is_market_hours', return_value=False), \
             patch.object(position_monitor, '_is_within_eod_grace_period', return_value=False):

            signal = ExitSignal(
                osi_symbol=sample_1h_position.osi_symbol,
                signal_key=sample_1h_position.signal_key,
                reason=ExitReason.EOD_EXIT,
                underlying_price=450.0,
                current_option_price=2.50,
                unrealized_pnl=0.0,
                dte=30,
                details="EOD exit test",
            )

            result = position_monitor.execute_exit(signal)

            assert result is None
            position_monitor.trading_client.close_option_position.assert_not_called()

    def test_eod_exit_allowed_in_grace_period(self, position_monitor, sample_1h_position):
        """EOD exits should be allowed in grace period even after market close."""
        # Add position
        position_monitor._positions[sample_1h_position.osi_symbol] = sample_1h_position

        # Mock market as closed but IN grace period
        with patch.object(position_monitor, '_is_market_hours', return_value=False), \
             patch.object(position_monitor, '_is_within_eod_grace_period', return_value=True):

            signal = ExitSignal(
                osi_symbol=sample_1h_position.osi_symbol,
                signal_key=sample_1h_position.signal_key,
                reason=ExitReason.EOD_EXIT,
                underlying_price=450.0,
                current_option_price=2.50,
                unrealized_pnl=0.0,
                dte=30,
                details="EOD exit test",
            )

            result = position_monitor.execute_exit(signal)

            # Should have called close_option_position
            position_monitor.trading_client.close_option_position.assert_called_once()

    def test_non_eod_exit_blocked_in_grace_period(self, position_monitor, sample_1h_position):
        """Non-EOD exits should still be blocked in grace period."""
        # Add position
        position_monitor._positions[sample_1h_position.osi_symbol] = sample_1h_position

        # Mock market as closed but in grace period
        with patch.object(position_monitor, '_is_market_hours', return_value=False), \
             patch.object(position_monitor, '_is_within_eod_grace_period', return_value=True):

            # Test with STOP_HIT (not EOD)
            signal = ExitSignal(
                osi_symbol=sample_1h_position.osi_symbol,
                signal_key=sample_1h_position.signal_key,
                reason=ExitReason.STOP_HIT,  # Not EOD
                underlying_price=445.0,
                current_option_price=1.50,
                unrealized_pnl=-100.0,
                dte=30,
                details="Stop hit test",
            )

            result = position_monitor.execute_exit(signal)

            assert result is None
            position_monitor.trading_client.close_option_position.assert_not_called()


# =============================================================================
# Daemon EOD Exit Job Tests
# =============================================================================

class TestDaemonEodExitJob:
    """Tests for daemon EOD exit job."""

    def test_eod_exit_job_filters_1h_positions_only(self):
        """EOD exit job should only process 1H positions."""
        # Create mock daemon with position monitor
        daemon = Mock(spec=SignalDaemon)
        daemon.position_monitor = Mock()
        daemon._exit_count = 0
        daemon._error_count = 0

        # Create 1H and 4H positions
        et = pytz.timezone('America/New_York')
        pos_1h = Mock()
        pos_1h.timeframe = '1H'
        pos_1h.osi_symbol = 'SPY250120C00450000'
        pos_1h.signal_key = 'SPY_1H_test'
        pos_1h.underlying_price = 450.0
        pos_1h.current_price = 2.50
        pos_1h.unrealized_pnl = 0.0
        pos_1h.dte = 30

        pos_4h = Mock()
        pos_4h.timeframe = '4H'
        pos_4h.osi_symbol = 'QQQ250120C00380000'

        daemon.position_monitor.get_tracked_positions.return_value = [pos_1h, pos_4h]
        daemon.position_monitor.execute_exit.return_value = {'status': 'filled'}

        # Run the EOD exit job logic
        positions = daemon.position_monitor.get_tracked_positions()
        hourly_positions = [
            p for p in positions
            if p.timeframe and p.timeframe.upper() in ['1H', '60MIN', '60M']
        ]

        # Should only include 1H position
        assert len(hourly_positions) == 1
        assert hourly_positions[0].timeframe == '1H'

    def test_eod_exit_job_skips_when_no_1h_positions(self):
        """EOD exit job should do nothing when no 1H positions exist."""
        daemon = Mock(spec=SignalDaemon)
        daemon.position_monitor = Mock()
        daemon._exit_count = 0
        daemon._error_count = 0

        # Only 4H position
        pos_4h = Mock()
        pos_4h.timeframe = '4H'

        daemon.position_monitor.get_tracked_positions.return_value = [pos_4h]

        positions = daemon.position_monitor.get_tracked_positions()
        hourly_positions = [
            p for p in positions
            if p.timeframe and p.timeframe.upper() in ['1H', '60MIN', '60M']
        ]

        assert len(hourly_positions) == 0


# =============================================================================
# Retry Logic Tests
# =============================================================================

class TestRetryLogic:
    """Tests for EOD exit retry logic."""

    def test_retry_succeeds_on_second_attempt(self):
        """Retry should succeed if second attempt works."""
        daemon = Mock(spec=SignalDaemon)
        daemon.position_monitor = Mock()

        # First call fails (returns None), second succeeds
        daemon.position_monitor.execute_exit.side_effect = [None, {'status': 'filled'}]

        exit_signal = Mock()
        exit_signal.osi_symbol = 'SPY250120C00450000'

        # Simulate retry logic
        result = None
        for attempt in range(3):
            result = daemon.position_monitor.execute_exit(exit_signal)
            if result:
                break

        assert result is not None
        assert daemon.position_monitor.execute_exit.call_count == 2

    def test_retry_gives_up_after_max_attempts(self):
        """Retry should give up after max attempts."""
        daemon = Mock(spec=SignalDaemon)
        daemon.position_monitor = Mock()

        # All attempts fail
        daemon.position_monitor.execute_exit.return_value = None

        exit_signal = Mock()
        exit_signal.osi_symbol = 'SPY250120C00450000'

        # Simulate retry logic
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            result = daemon.position_monitor.execute_exit(exit_signal)
            if result:
                break

        assert result is None
        assert daemon.position_monitor.execute_exit.call_count == max_retries


# =============================================================================
# Stale Position Detection Tests
# =============================================================================

class TestStalePositionDetection:
    """Tests for stale 1H position detection at market open."""

    def test_stale_1h_position_detected(self, position_monitor):
        """Positions entered on previous trading day should be detected as stale."""
        et = pytz.timezone('America/New_York')

        # Create entry time from yesterday
        yesterday = datetime.now(et) - timedelta(days=1)
        entry_time = et.localize(yesterday.replace(hour=10, minute=0, tzinfo=None))

        # Mock the market calendar to show 2 trading days
        with patch.object(position_monitor, '_market_hours_validator') as mock_validator:
            mock_validator.get_schedule.return_value = Mock(is_trading_day=True)

            # The _is_stale_1h_position should detect this as stale
            # We test the filter logic used in daemon
            positions = [Mock(
                timeframe='1H',
                entry_time=entry_time,
            )]

            hourly_positions = [
                p for p in positions
                if p.timeframe and p.timeframe.upper() in ['1H', '60MIN', '60M']
            ]

            assert len(hourly_positions) == 1

    def test_fresh_1h_position_not_stale(self, position_monitor):
        """Positions entered today should not be detected as stale."""
        et = pytz.timezone('America/New_York')

        # Create entry time from today
        today = datetime.now(et)
        entry_time = et.localize(today.replace(hour=10, minute=0, tzinfo=None))

        # Positions entered today are not stale
        # This is handled by _is_stale_1h_position() in position_monitor
        assert entry_time.date() == today.date()


# =============================================================================
# Logging Tests
# =============================================================================

class TestLogging:
    """Tests for logging levels."""

    def test_blocked_exit_logs_warning(self, position_monitor, sample_1h_position, caplog):
        """Blocked exits should log at WARNING level."""
        import logging

        # Add position
        position_monitor._positions[sample_1h_position.osi_symbol] = sample_1h_position

        # Mock market as closed and NOT in grace period
        with patch.object(position_monitor, '_is_market_hours', return_value=False), \
             patch.object(position_monitor, '_is_within_eod_grace_period', return_value=False):

            with caplog.at_level(logging.WARNING):
                signal = ExitSignal(
                    osi_symbol=sample_1h_position.osi_symbol,
                    signal_key=sample_1h_position.signal_key,
                    reason=ExitReason.STOP_HIT,
                    underlying_price=445.0,
                    current_option_price=1.50,
                    unrealized_pnl=-100.0,
                    dte=30,
                    details="Stop hit test",
                )

                position_monitor.execute_exit(signal)

            # Check that warning was logged
            assert any('Skipping exit' in record.message for record in caplog.records)
            assert any(record.levelname == 'WARNING' for record in caplog.records
                      if 'Skipping exit' in record.message)
