"""
Tests for strat/signal_automation/scheduler.py

Covers:
- SignalScheduler initialization
- Cron expression parsing
- Job addition methods (hourly, daily, weekly, monthly, 15m, 30m, base scan)
- Job management (run_job_now, start, shutdown, pause, resume)
- Status and stats reporting
- Event handlers
- Market hours checking
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, time
import pytz

from strat.signal_automation.scheduler import SignalScheduler
from strat.signal_automation.config import ScheduleConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_scheduler():
    """Create a mocked BackgroundScheduler."""
    with patch('strat.signal_automation.scheduler.BackgroundScheduler') as mock_bg:
        scheduler_instance = MagicMock()
        mock_bg.return_value = scheduler_instance
        yield scheduler_instance


@pytest.fixture
def scheduler(mock_scheduler):
    """Create a SignalScheduler with mocked APScheduler."""
    return SignalScheduler()


@pytest.fixture
def default_config():
    """Create default ScheduleConfig."""
    return ScheduleConfig()


@pytest.fixture
def custom_config():
    """Create custom ScheduleConfig with all scans disabled."""
    config = ScheduleConfig()
    config.scan_15m = False
    config.scan_30m = False
    config.scan_hourly = False
    config.scan_daily = False
    config.scan_weekly = False
    config.scan_monthly = False
    config.enable_htf_resampling = False
    return config


# =============================================================================
# Initialization Tests
# =============================================================================

class TestSignalSchedulerInit:
    """Test SignalScheduler initialization."""

    def test_default_initialization(self, mock_scheduler):
        """Default initialization creates scheduler with ET timezone."""
        scheduler = SignalScheduler()

        assert scheduler.config is not None
        assert str(scheduler.timezone) == 'America/New_York'
        assert scheduler._is_running is False
        assert len(scheduler._jobs) == 0
        assert len(scheduler._job_stats) == 0

    def test_initialization_with_custom_config(self, mock_scheduler):
        """Initialization with custom config."""
        config = ScheduleConfig()
        config.scan_hourly = False

        scheduler = SignalScheduler(config=config)

        assert scheduler.config.scan_hourly is False

    def test_initialization_with_custom_timezone(self, mock_scheduler):
        """Initialization with custom timezone."""
        scheduler = SignalScheduler(timezone='America/Los_Angeles')

        assert str(scheduler.timezone) == 'America/Los_Angeles'

    def test_scheduler_created_with_job_defaults(self, mock_scheduler):
        """BackgroundScheduler created with correct job defaults."""
        scheduler = SignalScheduler()

        # Verify BackgroundScheduler was called with expected arguments
        mock_scheduler.add_listener.assert_called()  # Event listeners added

    def test_market_hours_constants(self, mock_scheduler):
        """Market hours constants are set correctly."""
        assert SignalScheduler.MARKET_OPEN == time(9, 30)
        assert SignalScheduler.MARKET_CLOSE == time(16, 0)


# =============================================================================
# Cron Parsing Tests
# =============================================================================

class TestCronParsing:
    """Test cron expression parsing."""

    def test_parse_valid_cron(self, scheduler):
        """Parse valid 5-part cron expression."""
        result = scheduler._parse_cron('30 9-15 * * mon-fri')

        assert result['minute'] == '30'
        assert result['hour'] == '9-15'
        assert result['day'] == '*'
        assert result['month'] == '*'
        assert result['day_of_week'] == 'mon-fri'

    def test_parse_cron_with_lists(self, scheduler):
        """Parse cron with comma-separated values."""
        result = scheduler._parse_cron('0,15,30,45 9-15 * * mon-fri')

        assert result['minute'] == '0,15,30,45'
        assert result['hour'] == '9-15'

    def test_parse_cron_specific_day(self, scheduler):
        """Parse cron with specific day."""
        result = scheduler._parse_cron('0 18 * * fri')

        assert result['day_of_week'] == 'fri'

    def test_parse_cron_specific_date(self, scheduler):
        """Parse cron with specific date."""
        result = scheduler._parse_cron('0 18 28 * *')

        assert result['day'] == '28'

    def test_parse_invalid_cron_raises(self, scheduler):
        """Invalid cron expression raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            scheduler._parse_cron('30 9-15 *')  # Only 3 parts

    def test_parse_cron_too_many_parts(self, scheduler):
        """Cron with too many parts raises ValueError."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            scheduler._parse_cron('30 9-15 * * mon-fri extra')


# =============================================================================
# Hourly Job Tests
# =============================================================================

class TestAddHourlyJob:
    """Test add_hourly_job method."""

    def test_add_hourly_job_when_enabled(self, scheduler, mock_scheduler):
        """Add hourly job when scan_hourly is enabled."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_hourly_job(callback)

        assert result == 'scan_hourly'
        assert 'hourly' in scheduler._jobs
        assert 'scan_hourly' in scheduler._job_stats
        mock_scheduler.add_job.assert_called_once()

    def test_add_hourly_job_when_disabled(self, mock_scheduler, custom_config):
        """Add hourly job returns None when disabled."""
        scheduler = SignalScheduler(config=custom_config)
        callback = Mock()

        result = scheduler.add_hourly_job(callback)

        assert result is None
        mock_scheduler.add_job.assert_not_called()

    def test_add_hourly_job_custom_id(self, scheduler, mock_scheduler):
        """Add hourly job with custom job ID."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'custom_hourly'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_hourly_job(callback, job_id='custom_hourly')

        assert result == 'custom_hourly'

    def test_hourly_job_stats_initialized(self, scheduler, mock_scheduler):
        """Hourly job stats are properly initialized."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_scheduler.add_job.return_value = mock_job

        scheduler.add_hourly_job(callback)

        stats = scheduler._job_stats['scan_hourly']
        assert stats['name'] == 'Hourly Scan'
        assert stats['run_count'] == 0
        assert stats['error_count'] == 0
        assert stats['missed_count'] == 0
        assert stats['last_run'] is None
        assert stats['last_status'] == 'pending'


# =============================================================================
# Daily Job Tests
# =============================================================================

class TestAddDailyJob:
    """Test add_daily_job method."""

    def test_add_daily_job_when_enabled(self, scheduler, mock_scheduler):
        """Add daily job when scan_daily is enabled."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_daily'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_daily_job(callback)

        assert result == 'scan_daily'
        assert 'daily' in scheduler._jobs

    def test_add_daily_job_when_disabled(self, mock_scheduler, custom_config):
        """Add daily job returns None when disabled."""
        scheduler = SignalScheduler(config=custom_config)
        callback = Mock()

        result = scheduler.add_daily_job(callback)

        assert result is None


# =============================================================================
# Weekly Job Tests
# =============================================================================

class TestAddWeeklyJob:
    """Test add_weekly_job method."""

    def test_add_weekly_job_when_enabled(self, scheduler, mock_scheduler):
        """Add weekly job when scan_weekly is enabled."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_weekly'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_weekly_job(callback)

        assert result == 'scan_weekly'
        assert 'weekly' in scheduler._jobs

    def test_add_weekly_job_when_disabled(self, mock_scheduler, custom_config):
        """Add weekly job returns None when disabled."""
        scheduler = SignalScheduler(config=custom_config)
        callback = Mock()

        result = scheduler.add_weekly_job(callback)

        assert result is None


# =============================================================================
# Monthly Job Tests
# =============================================================================

class TestAddMonthlyJob:
    """Test add_monthly_job method."""

    def test_add_monthly_job_when_enabled(self, scheduler, mock_scheduler):
        """Add monthly job when scan_monthly is enabled."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_monthly'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_monthly_job(callback)

        assert result == 'scan_monthly'
        assert 'monthly' in scheduler._jobs

    def test_add_monthly_job_when_disabled(self, mock_scheduler, custom_config):
        """Add monthly job returns None when disabled."""
        scheduler = SignalScheduler(config=custom_config)
        callback = Mock()

        result = scheduler.add_monthly_job(callback)

        assert result is None


# =============================================================================
# 15m and 30m Job Tests
# =============================================================================

class TestAdd15mJob:
    """Test add_15m_job method."""

    def test_add_15m_job_when_enabled(self, mock_scheduler):
        """Add 15m job when scan_15m is enabled."""
        config = ScheduleConfig()
        config.scan_15m = True
        scheduler = SignalScheduler(config=config)

        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_15m'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_15m_job(callback)

        assert result == 'scan_15m'
        assert '15m' in scheduler._jobs

    def test_add_15m_job_when_disabled(self, scheduler, mock_scheduler):
        """Add 15m job returns None when disabled (default)."""
        callback = Mock()

        result = scheduler.add_15m_job(callback)

        assert result is None


class TestAdd30mJob:
    """Test add_30m_job method."""

    def test_add_30m_job_when_enabled(self, mock_scheduler):
        """Add 30m job when scan_30m is enabled."""
        config = ScheduleConfig()
        config.scan_30m = True
        scheduler = SignalScheduler(config=config)

        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_30m'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_30m_job(callback)

        assert result == 'scan_30m'
        assert '30m' in scheduler._jobs

    def test_add_30m_job_when_disabled(self, scheduler, mock_scheduler):
        """Add 30m job returns None when disabled (default)."""
        callback = Mock()

        result = scheduler.add_30m_job(callback)

        assert result is None


# =============================================================================
# Base Scan Job Tests (HTF Resampling)
# =============================================================================

class TestAddBaseScanJob:
    """Test add_base_scan_job method."""

    def test_add_base_scan_job_when_enabled(self, scheduler, mock_scheduler):
        """Add base scan job when HTF resampling is enabled."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_base'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_base_scan_job(callback)

        assert result == 'scan_base'
        assert 'base' in scheduler._jobs

    def test_add_base_scan_job_when_disabled(self, mock_scheduler, custom_config):
        """Add base scan job returns None when HTF resampling disabled."""
        scheduler = SignalScheduler(config=custom_config)
        callback = Mock()

        result = scheduler.add_base_scan_job(callback)

        assert result is None

    def test_base_scan_job_stats(self, scheduler, mock_scheduler):
        """Base scan job stats are properly initialized."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_base'
        mock_scheduler.add_job.return_value = mock_job

        scheduler.add_base_scan_job(callback)

        stats = scheduler._job_stats['scan_base']
        assert stats['name'] == '15-Min Multi-TF Scan'


# =============================================================================
# Health Check and Interval Job Tests
# =============================================================================

class TestAddHealthCheckJob:
    """Test add_health_check_job method."""

    def test_add_health_check_job(self, scheduler, mock_scheduler):
        """Add health check job."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'health_check'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_health_check_job(callback)

        assert result == 'health_check'
        assert 'health_check' in scheduler._jobs

    def test_add_health_check_job_custom_interval(self, scheduler, mock_scheduler):
        """Add health check job with custom interval."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'health_check'
        mock_scheduler.add_job.return_value = mock_job

        scheduler.add_health_check_job(callback, interval_seconds=60)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs[1]['seconds'] == 60


class TestAddIntervalJob:
    """Test add_interval_job method."""

    def test_add_interval_job(self, scheduler, mock_scheduler):
        """Add generic interval job."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'custom_interval'
        mock_scheduler.add_job.return_value = mock_job

        result = scheduler.add_interval_job(
            callback,
            interval_seconds=120,
            job_id='custom_interval',
            job_name='Custom Interval Job'
        )

        assert result == 'custom_interval'
        assert 'custom_interval' in scheduler._jobs

    def test_add_interval_job_stats(self, scheduler, mock_scheduler):
        """Interval job stats are properly initialized."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'custom_interval'
        mock_scheduler.add_job.return_value = mock_job

        scheduler.add_interval_job(
            callback,
            interval_seconds=120,
            job_id='custom_interval',
            job_name='Custom Job'
        )

        stats = scheduler._job_stats['custom_interval']
        assert stats['name'] == 'Custom Job'
        assert stats['run_count'] == 0


# =============================================================================
# Job Management Tests
# =============================================================================

class TestRunJobNow:
    """Test run_job_now method."""

    def test_run_job_now_existing_job(self, scheduler, mock_scheduler):
        """Run existing job immediately."""
        # Setup job
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_scheduler.add_job.return_value = mock_job
        mock_scheduler.get_job.return_value = mock_job

        scheduler.add_hourly_job(callback)
        result = scheduler.run_job_now('hourly')

        assert result is True
        mock_job.modify.assert_called_once()

    def test_run_job_now_nonexistent_job(self, scheduler, mock_scheduler):
        """Run nonexistent job returns False."""
        result = scheduler.run_job_now('nonexistent')

        assert result is False

    def test_run_job_now_job_not_in_scheduler(self, scheduler, mock_scheduler):
        """Run job that exists in tracking but not in scheduler."""
        scheduler._jobs['test'] = 'test_id'
        mock_scheduler.get_job.return_value = None

        result = scheduler.run_job_now('test')

        assert result is False


# =============================================================================
# Start/Shutdown/Pause/Resume Tests
# =============================================================================

class TestSchedulerLifecycle:
    """Test scheduler start, shutdown, pause, resume."""

    def test_start_scheduler(self, scheduler, mock_scheduler):
        """Start scheduler."""
        scheduler.start()

        assert scheduler._is_running is True
        mock_scheduler.start.assert_called_once()

    def test_start_scheduler_already_running(self, scheduler, mock_scheduler):
        """Start scheduler when already running logs warning."""
        scheduler._is_running = True

        scheduler.start()

        mock_scheduler.start.assert_not_called()

    def test_shutdown_scheduler(self, scheduler, mock_scheduler):
        """Shutdown scheduler."""
        scheduler._is_running = True

        scheduler.shutdown()

        assert scheduler._is_running is False
        mock_scheduler.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_scheduler_not_running(self, scheduler, mock_scheduler):
        """Shutdown scheduler when not running logs warning."""
        scheduler._is_running = False

        scheduler.shutdown()

        mock_scheduler.shutdown.assert_not_called()

    def test_shutdown_scheduler_no_wait(self, scheduler, mock_scheduler):
        """Shutdown scheduler without waiting."""
        scheduler._is_running = True

        scheduler.shutdown(wait=False)

        mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_pause_scheduler(self, scheduler, mock_scheduler):
        """Pause scheduler."""
        scheduler.pause()

        mock_scheduler.pause.assert_called_once()

    def test_resume_scheduler(self, scheduler, mock_scheduler):
        """Resume scheduler."""
        scheduler.resume()

        mock_scheduler.resume.assert_called_once()

    def test_is_running_property(self, scheduler):
        """is_running property returns correct value."""
        assert scheduler.is_running is False

        scheduler._is_running = True
        assert scheduler.is_running is True


# =============================================================================
# Status and Stats Tests
# =============================================================================

class TestGetNextRunTimes:
    """Test get_next_run_times method."""

    def test_get_next_run_times_with_jobs(self, scheduler, mock_scheduler):
        """Get next run times for existing jobs."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_job.next_run_time = datetime(2024, 1, 15, 10, 30)
        mock_scheduler.add_job.return_value = mock_job
        mock_scheduler.get_job.return_value = mock_job

        scheduler.add_hourly_job(callback)
        result = scheduler.get_next_run_times()

        assert 'hourly' in result
        assert result['hourly'] == datetime(2024, 1, 15, 10, 30)

    def test_get_next_run_times_job_not_found(self, scheduler, mock_scheduler):
        """Get next run times when job not in scheduler."""
        scheduler._jobs['test'] = 'test_id'
        mock_scheduler.get_job.return_value = None

        result = scheduler.get_next_run_times()

        assert result['test'] is None


class TestGetJobStats:
    """Test get_job_stats method."""

    def test_get_job_stats(self, scheduler, mock_scheduler):
        """Get job stats."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_job.next_run_time = datetime(2024, 1, 15, 10, 30)
        mock_scheduler.add_job.return_value = mock_job
        mock_scheduler.get_job.return_value = mock_job

        scheduler.add_hourly_job(callback)
        result = scheduler.get_job_stats()

        assert 'scan_hourly' in result
        assert result['scan_hourly']['name'] == 'Hourly Scan'

    def test_get_job_stats_returns_copy(self, scheduler, mock_scheduler):
        """Get job stats returns a copy of top-level dict."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_scheduler.add_job.return_value = mock_job
        mock_scheduler.get_job.return_value = mock_job

        scheduler.add_hourly_job(callback)
        result = scheduler.get_job_stats()

        # Verify it's a copy (new dict object)
        assert result is not scheduler._job_stats

        # Add new key to returned dict
        result['new_key'] = 'test'

        # Original should not have new key
        assert 'new_key' not in scheduler._job_stats


class TestGetStatus:
    """Test get_status method."""

    def test_get_status(self, scheduler, mock_scheduler):
        """Get overall scheduler status."""
        scheduler._is_running = True

        with patch.object(scheduler, 'is_market_hours', return_value=True):
            result = scheduler.get_status()

        assert result['running'] is True
        assert result['timezone'] == 'America/New_York'
        assert result['market_hours'] is True
        assert 'jobs_count' in result
        assert 'jobs' in result
        assert 'next_runs' in result


class TestGetJobsSummary:
    """Test get_jobs_summary method."""

    def test_get_jobs_summary(self, scheduler, mock_scheduler):
        """Get jobs summary."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_job.next_run_time = datetime(2024, 1, 15, 10, 30)
        mock_scheduler.add_job.return_value = mock_job
        mock_scheduler.get_job.return_value = mock_job

        scheduler.add_hourly_job(callback)
        result = scheduler.get_jobs_summary()

        assert len(result) == 1
        assert result[0]['name'] == 'hourly'
        assert result[0]['job_id'] == 'scan_hourly'
        assert result[0]['run_count'] == 0


# =============================================================================
# Event Handler Tests
# =============================================================================

class TestEventHandlers:
    """Test job event handlers."""

    def test_on_job_executed_updates_stats(self, scheduler, mock_scheduler):
        """Job executed event updates stats."""
        # Setup job
        scheduler._job_stats['test_job'] = {
            'name': 'Test',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        # Create mock event
        mock_event = MagicMock()
        mock_event.job_id = 'test_job'

        scheduler._on_job_executed(mock_event)

        assert scheduler._job_stats['test_job']['run_count'] == 1
        assert scheduler._job_stats['test_job']['last_status'] == 'success'
        assert scheduler._job_stats['test_job']['last_run'] is not None

    def test_on_job_error_updates_stats(self, scheduler, mock_scheduler):
        """Job error event updates stats."""
        scheduler._job_stats['test_job'] = {
            'name': 'Test',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        mock_event = MagicMock()
        mock_event.job_id = 'test_job'
        mock_event.exception = Exception("Test error")

        scheduler._on_job_error(mock_event)

        assert scheduler._job_stats['test_job']['error_count'] == 1
        assert scheduler._job_stats['test_job']['last_status'] == 'error'
        assert 'Test error' in scheduler._job_stats['test_job']['last_error']

    def test_on_job_missed_updates_stats(self, scheduler, mock_scheduler):
        """Job missed event updates stats."""
        scheduler._job_stats['test_job'] = {
            'name': 'Test',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        mock_event = MagicMock()
        mock_event.job_id = 'test_job'

        scheduler._on_job_missed(mock_event)

        assert scheduler._job_stats['test_job']['missed_count'] == 1
        assert scheduler._job_stats['test_job']['last_status'] == 'missed'

    def test_event_handlers_ignore_unknown_jobs(self, scheduler, mock_scheduler):
        """Event handlers ignore unknown job IDs."""
        mock_event = MagicMock()
        mock_event.job_id = 'unknown_job'

        # Should not raise
        scheduler._on_job_executed(mock_event)
        scheduler._on_job_error(mock_event)
        scheduler._on_job_missed(mock_event)


# =============================================================================
# Market Hours Tests
# =============================================================================

class TestIsMarketHours:
    """Test is_market_hours method."""

    def test_is_market_hours_during_trading(self, scheduler, mock_scheduler):
        """Market hours check during trading day."""
        import pandas as pd

        with patch('pandas_market_calendars.get_calendar') as mock_get_cal:
            mock_calendar = MagicMock()
            mock_get_cal.return_value = mock_calendar

            # Create schedule DataFrame
            tz = pytz.timezone('America/New_York')
            schedule = pd.DataFrame({
                'market_open': [pd.Timestamp('2024-01-15 09:30:00', tz='America/New_York')],
                'market_close': [pd.Timestamp('2024-01-15 16:00:00', tz='America/New_York')],
            })
            mock_calendar.schedule.return_value = schedule

            # Mock current time to be during market hours
            mock_now = tz.localize(datetime(2024, 1, 15, 12, 0))
            with patch.object(scheduler.timezone, 'localize', return_value=mock_now):
                with patch('strat.signal_automation.scheduler.datetime') as mock_dt:
                    mock_dt.now.return_value = mock_now

                    result = scheduler.is_market_hours()

        assert result is True

    def test_is_market_hours_holiday(self, scheduler, mock_scheduler):
        """Market hours check on holiday (empty schedule)."""
        import pandas as pd

        with patch('pandas_market_calendars.get_calendar') as mock_get_cal:
            mock_calendar = MagicMock()
            mock_get_cal.return_value = mock_calendar

            # Empty schedule = market closed
            mock_calendar.schedule.return_value = pd.DataFrame()

            result = scheduler.is_market_hours()

        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestSchedulerIntegration:
    """Integration tests for SignalScheduler."""

    def test_add_multiple_jobs(self, scheduler, mock_scheduler):
        """Add multiple jobs to scheduler."""
        callback = Mock()
        mock_job = MagicMock()
        mock_scheduler.add_job.return_value = mock_job

        # Mock different job IDs
        mock_scheduler.add_job.side_effect = [
            MagicMock(id='scan_hourly'),
            MagicMock(id='scan_daily'),
            MagicMock(id='scan_weekly'),
        ]

        scheduler.add_hourly_job(callback)
        scheduler.add_daily_job(callback)
        scheduler.add_weekly_job(callback)

        assert len(scheduler._jobs) == 3
        assert 'hourly' in scheduler._jobs
        assert 'daily' in scheduler._jobs
        assert 'weekly' in scheduler._jobs

    def test_full_lifecycle(self, scheduler, mock_scheduler):
        """Test full scheduler lifecycle."""
        callback = Mock()
        mock_job = MagicMock()
        mock_job.id = 'scan_hourly'
        mock_scheduler.add_job.return_value = mock_job

        # Add job
        scheduler.add_hourly_job(callback)

        # Start
        scheduler.start()
        assert scheduler.is_running is True

        # Pause
        scheduler.pause()
        mock_scheduler.pause.assert_called_once()

        # Resume
        scheduler.resume()
        mock_scheduler.resume.assert_called_once()

        # Shutdown
        scheduler.shutdown()
        assert scheduler.is_running is False

    def test_event_stats_accumulate(self, scheduler, mock_scheduler):
        """Event stats accumulate over multiple events."""
        scheduler._job_stats['test_job'] = {
            'name': 'Test',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        mock_event = MagicMock()
        mock_event.job_id = 'test_job'

        # Multiple executions
        scheduler._on_job_executed(mock_event)
        scheduler._on_job_executed(mock_event)
        scheduler._on_job_executed(mock_event)

        assert scheduler._job_stats['test_job']['run_count'] == 3

        # Some errors
        mock_event.exception = Exception("Error")
        scheduler._on_job_error(mock_event)

        assert scheduler._job_stats['test_job']['error_count'] == 1
        assert scheduler._job_stats['test_job']['last_status'] == 'error'
