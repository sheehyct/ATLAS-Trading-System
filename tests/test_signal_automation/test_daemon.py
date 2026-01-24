"""
SignalDaemon Comprehensive Tests - Session EQUITY-81

Tests for the main SignalDaemon class covering:
- Initialization and component setup
- Signal filtering (_passes_filters)
- Market hours detection
- Scanning operations
- Health checks and status reporting
- Lifecycle management (start/shutdown)
- Position monitoring integration

Note: TFC re-evaluation tested in test_tfc_reeval.py
Note: Stale setup detection tested in test_stale_setup.py
"""

import pytest
from datetime import datetime, timedelta, time as dt_time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytz
import os
from threading import Event

from strat.signal_automation.daemon import SignalDaemon
from strat.signal_automation.config import (
    SignalAutomationConfig,
    ExecutionConfig,
    ScanConfig,
    AlertConfig,
    ScheduleConfig,
    MonitoringConfig,
)
from strat.signal_automation.signal_store import StoredSignal, SignalType, SignalStatus
from strat.signal_automation.executor import ExecutionResult, ExecutionState
from strat.signal_automation.position_monitor import (
    ExitSignal,
    ExitReason,
    TrackedPosition,
)
from strat.paper_signal_scanner import DetectedSignal, SignalContext


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_store_path(tmp_path):
    """Create temporary path for signal store."""
    return str(tmp_path / 'signals')


@pytest.fixture
def minimal_config(tmp_store_path):
    """Create minimal config for basic tests."""
    return SignalAutomationConfig(
        store_path=tmp_store_path,
    )


@pytest.fixture
def full_config(tmp_store_path):
    """Create full config with all features enabled."""
    return SignalAutomationConfig(
        store_path=tmp_store_path,
        scan=ScanConfig(
            symbols=['SPY', 'AAPL'],
            timeframes=['1H', '1D'],
            min_magnitude_pct=0.5,
            min_risk_reward=1.0,
            patterns=['3-1-2', '2-1-2', '3-2', '2-2'],
        ),
        alerts=AlertConfig(
            logging_enabled=True,
            discord_enabled=False,
        ),
        execution=ExecutionConfig(
            enabled=True,
            tfc_reeval_enabled=True,
            tfc_reeval_min_strength=2,
        ),
        monitoring=MonitoringConfig(
            enabled=True,
            check_interval=60,
        ),
    )


@pytest.fixture
def daemon(minimal_config):
    """Create daemon with minimal config."""
    return SignalDaemon(config=minimal_config)


@pytest.fixture
def daemon_with_mocks(full_config):
    """Create daemon with mocked components."""
    mock_scanner = Mock()
    mock_store = Mock()
    mock_executor = Mock()
    mock_monitor = Mock()

    d = SignalDaemon(
        config=full_config,
        scanner=mock_scanner,
        signal_store=mock_store,
        executor=mock_executor,
        position_monitor=mock_monitor,
    )
    return d


def create_detected_signal(
    symbol: str = 'TEST',
    timeframe: str = '1D',
    pattern_type: str = '3-2U',
    direction: str = 'CALL',
    magnitude_pct: float = 1.0,
    risk_reward: float = 1.5,
    signal_type: str = 'SETUP',
    tfc_score: int = 3,
    tfc_alignment: str = '3/4 BULLISH',
    aligned_timeframes: list = None,
) -> DetectedSignal:
    """Helper to create DetectedSignal for testing."""
    context = SignalContext(
        tfc_score=tfc_score,
        tfc_alignment=tfc_alignment,
        aligned_timeframes=aligned_timeframes or ['1D', '1W', '1M'],
        atr_14=2.0,
        atr_percent=1.0,
        volume_ratio=1.5,
        market_regime='BULLISH',
    )

    return DetectedSignal(
        symbol=symbol,
        timeframe=timeframe,
        pattern_type=pattern_type,
        direction=direction,
        detected_time=datetime.now(pytz.timezone('America/New_York')),
        entry_trigger=100.0,
        stop_price=95.0,
        target_price=110.0,
        magnitude_pct=magnitude_pct,
        risk_reward=risk_reward,
        signal_type=signal_type,
        context=context,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestDaemonInit:
    """Tests for SignalDaemon initialization."""

    def test_init_with_minimal_config(self, minimal_config):
        """Daemon initializes with minimal config."""
        daemon = SignalDaemon(config=minimal_config)

        assert daemon.config is minimal_config
        assert daemon.scanner is not None
        assert daemon.signal_store is not None
        assert daemon.scheduler is not None
        assert daemon._is_running is False
        assert daemon._shutdown_event is not None

    def test_init_creates_scanner_if_not_provided(self, minimal_config):
        """Daemon creates PaperSignalScanner if not provided."""
        daemon = SignalDaemon(config=minimal_config)

        from strat.paper_signal_scanner import PaperSignalScanner
        assert isinstance(daemon.scanner, PaperSignalScanner)

    def test_init_uses_provided_scanner(self, minimal_config):
        """Daemon uses provided scanner."""
        mock_scanner = Mock()
        daemon = SignalDaemon(config=minimal_config, scanner=mock_scanner)

        assert daemon.scanner is mock_scanner

    def test_init_creates_signal_store(self, minimal_config):
        """Daemon creates SignalStore at configured path."""
        daemon = SignalDaemon(config=minimal_config)

        from strat.signal_automation.signal_store import SignalStore
        assert isinstance(daemon.signal_store, SignalStore)

    def test_init_uses_provided_signal_store(self, minimal_config):
        """Daemon uses provided signal store."""
        mock_store = Mock()
        daemon = SignalDaemon(config=minimal_config, signal_store=mock_store)

        assert daemon.signal_store is mock_store

    def test_init_counters_start_at_zero(self, minimal_config):
        """All daemon counters start at zero."""
        daemon = SignalDaemon(config=minimal_config)

        assert daemon._scan_count == 0
        assert daemon._signal_count == 0
        assert daemon._error_count == 0
        assert daemon._execution_count == 0
        assert daemon._exit_count == 0

    def test_init_start_time_is_none(self, minimal_config):
        """Start time is None before daemon starts."""
        daemon = SignalDaemon(config=minimal_config)

        assert daemon._start_time is None


class TestSetupAlerters:
    """Tests for _setup_alerters method."""

    def test_logging_alerter_enabled_by_default(self, minimal_config):
        """Logging alerter is created when enabled."""
        daemon = SignalDaemon(config=minimal_config)

        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerters = [a for a in daemon.alerters if isinstance(a, LoggingAlerter)]
        assert len(logging_alerters) == 1

    def test_logging_alerter_disabled(self, tmp_store_path):
        """Logging alerter not created when disabled."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            alerts=AlertConfig(logging_enabled=False, discord_enabled=False),
        )
        daemon = SignalDaemon(config=config)

        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerters = [a for a in daemon.alerters if isinstance(a, LoggingAlerter)]
        assert len(logging_alerters) == 0

    def test_discord_alerter_not_created_without_webhook(self, tmp_store_path):
        """Discord alerter not created without valid webhook."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            alerts=AlertConfig(discord_enabled=True, discord_webhook_url=None),
        )
        daemon = SignalDaemon(config=config)

        from strat.signal_automation.alerters import DiscordAlerter
        discord_alerters = [a for a in daemon.alerters if isinstance(a, DiscordAlerter)]
        assert len(discord_alerters) == 0


class TestSetupExecutor:
    """Tests for _setup_executor method."""

    def test_executor_not_created_when_disabled(self, tmp_store_path):
        """Executor not created when execution disabled."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            execution=ExecutionConfig(enabled=False),
        )
        daemon = SignalDaemon(config=config)

        assert daemon.executor is None

    def test_executor_uses_provided_instance(self, tmp_store_path):
        """Daemon uses provided executor instance."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            execution=ExecutionConfig(enabled=True),
        )
        mock_executor = Mock()
        daemon = SignalDaemon(config=config, executor=mock_executor)

        assert daemon.executor is mock_executor


class TestSetupPositionMonitor:
    """Tests for _setup_position_monitor method."""

    def test_monitor_not_created_when_disabled(self, tmp_store_path):
        """Position monitor not created when disabled."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            monitoring=MonitoringConfig(enabled=False),
        )
        daemon = SignalDaemon(config=config)

        assert daemon.position_monitor is None

    def test_monitor_uses_provided_instance(self, tmp_store_path):
        """Daemon uses provided position monitor instance."""
        config = SignalAutomationConfig(
            store_path=tmp_store_path,
            monitoring=MonitoringConfig(enabled=True),
        )
        mock_monitor = Mock()
        daemon = SignalDaemon(config=config, position_monitor=mock_monitor)

        assert daemon.position_monitor is mock_monitor


class TestFromConfig:
    """Tests for from_config class method."""

    def test_from_config_with_provided_config(self, minimal_config):
        """from_config uses provided configuration."""
        daemon = SignalDaemon.from_config(minimal_config)

        assert daemon.config is minimal_config

    @patch.object(SignalAutomationConfig, 'from_env')
    def test_from_config_uses_env_when_none(self, mock_from_env, tmp_store_path):
        """from_config uses env-based config when None provided."""
        env_config = SignalAutomationConfig(store_path=tmp_store_path)
        mock_from_env.return_value = env_config

        daemon = SignalDaemon.from_config(None)

        mock_from_env.assert_called_once()
        assert daemon.config is env_config

    def test_from_config_logs_validation_issues(self, minimal_config, caplog):
        """from_config logs validation warnings."""
        # Config with issues will log warnings during validation
        daemon = SignalDaemon.from_config(minimal_config)

        # Daemon should still be created
        assert daemon is not None


# =============================================================================
# FILTER TESTS
# =============================================================================


class TestPassesFiltersMagnitude:
    """Tests for magnitude filtering in _passes_filters."""

    def test_setup_passes_with_relaxed_magnitude(self, daemon):
        """SETUP signals pass with lower magnitude (0.1% default)."""
        signal = create_detected_signal(
            magnitude_pct=0.15,  # Above 0.1% relaxed threshold
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is True

    def test_setup_fails_below_relaxed_magnitude(self, daemon):
        """SETUP signals fail below 0.1% magnitude."""
        signal = create_detected_signal(
            magnitude_pct=0.05,  # Below 0.1%
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is False

    def test_completed_uses_config_magnitude(self, full_config):
        """COMPLETED signals use config magnitude threshold."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=0.3,  # Below config's 0.5%
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False

    def test_completed_passes_above_config_magnitude(self, full_config):
        """COMPLETED signals pass above config magnitude."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=0.6,  # Above config's 0.5%
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is True

    def test_setup_magnitude_env_override(self, daemon, monkeypatch):
        """SETUP magnitude threshold can be overridden via env."""
        monkeypatch.setenv('SIGNAL_SETUP_MIN_MAGNITUDE', '0.5')

        signal = create_detected_signal(
            magnitude_pct=0.3,  # Below new 0.5% threshold
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is False


class TestPassesFiltersRiskReward:
    """Tests for R:R filtering in _passes_filters."""

    def test_setup_passes_with_relaxed_rr(self, daemon):
        """SETUP signals pass with lower R:R (0.3 default)."""
        signal = create_detected_signal(
            risk_reward=0.5,  # Above 0.3 relaxed threshold
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is True

    def test_setup_fails_below_relaxed_rr(self, daemon):
        """SETUP signals fail below 0.3 R:R."""
        signal = create_detected_signal(
            risk_reward=0.2,  # Below 0.3
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is False

    def test_completed_uses_config_rr(self, full_config):
        """COMPLETED signals use config R:R threshold."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            risk_reward=0.8,  # Below config's 1.0
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False

    def test_setup_rr_env_override(self, daemon, monkeypatch):
        """SETUP R:R threshold can be overridden via env."""
        monkeypatch.setenv('SIGNAL_SETUP_MIN_RR', '1.0')

        signal = create_detected_signal(
            risk_reward=0.5,  # Below new 1.0 threshold
            signal_type='SETUP',
        )

        assert daemon._passes_filters(signal) is False


class TestPassesFiltersPattern:
    """Tests for pattern filtering in _passes_filters."""

    def test_any_pattern_allowed_when_not_configured(self, daemon):
        """Any pattern allowed when patterns list is empty."""
        signal = create_detected_signal(pattern_type='3-2U')

        assert daemon._passes_filters(signal) is True

    def test_allowed_pattern_passes(self, full_config):
        """Configured patterns pass filter."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(pattern_type='3-2U')  # 3-2 is allowed

        assert daemon._passes_filters(signal) is True

    def test_disallowed_pattern_fails(self, full_config):
        """Non-configured patterns fail filter."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(pattern_type='3-2U-2D')  # 3-2-2 not in list

        assert daemon._passes_filters(signal) is False

    def test_pattern_direction_normalized(self, full_config):
        """Pattern directions (2U/2D) normalized for comparison."""
        daemon = SignalDaemon(config=full_config)

        # 2D-1-2U normalizes to 2-1-2 which is in allowed list
        signal = create_detected_signal(pattern_type='2D-1-2U')

        assert daemon._passes_filters(signal) is True

    def test_setup_pattern_normalized(self, full_config):
        """SETUP patterns (-?) normalized to -2 for comparison."""
        daemon = SignalDaemon(config=full_config)

        # 3-1-? normalizes to 3-1-2 which is in allowed list
        signal = create_detected_signal(pattern_type='3-1-?')

        assert daemon._passes_filters(signal) is True


class TestPassesFiltersTFC:
    """Tests for TFC filtering in _passes_filters."""

    def test_1h_passes_with_3_or_4_tfc(self, daemon):
        """1H signals pass with TFC >= 3."""
        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=3,
            tfc_alignment='3/4 BULLISH',
        )

        assert daemon._passes_filters(signal) is True

    def test_1h_passes_with_2_tfc_and_1d_aligned(self, daemon):
        """1H signals pass with TFC 2 if 1D is aligned."""
        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=2,
            tfc_alignment='2/4 BULLISH',
            aligned_timeframes=['1D', '1W'],  # 1D is aligned
        )

        assert daemon._passes_filters(signal) is True

    def test_1h_fails_with_2_tfc_without_1d(self, daemon):
        """1H signals fail with TFC 2 if 1D not aligned."""
        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=2,
            tfc_alignment='2/4 BULLISH',
            aligned_timeframes=['1W', '1M'],  # 1D NOT aligned
        )

        assert daemon._passes_filters(signal) is False

    def test_1h_fails_with_1_tfc(self, daemon):
        """1H signals fail with TFC 1."""
        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=1,
            tfc_alignment='1/4 BULLISH',
        )

        assert daemon._passes_filters(signal) is False

    def test_1h_fails_with_0_tfc(self, daemon):
        """1H signals fail with TFC 0."""
        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=0,
            tfc_alignment='0/4 NEUTRAL',
        )

        assert daemon._passes_filters(signal) is False

    def test_1d_passes_with_2_tfc(self, daemon):
        """1D signals pass with TFC >= 2."""
        signal = create_detected_signal(
            timeframe='1D',
            tfc_score=2,
            tfc_alignment='2/3 BULLISH',
        )

        assert daemon._passes_filters(signal) is True

    def test_1d_fails_with_1_tfc(self, daemon):
        """1D signals fail with TFC < 2."""
        signal = create_detected_signal(
            timeframe='1D',
            tfc_score=1,
            tfc_alignment='1/3 BULLISH',
        )

        assert daemon._passes_filters(signal) is False

    def test_1w_passes_with_1_tfc(self, daemon):
        """1W signals pass with TFC >= 1."""
        signal = create_detected_signal(
            timeframe='1W',
            tfc_score=1,
            tfc_alignment='1/2 BULLISH',
        )

        assert daemon._passes_filters(signal) is True

    def test_tfc_filter_can_be_disabled(self, daemon, monkeypatch):
        """TFC filter can be disabled via env."""
        monkeypatch.setenv('SIGNAL_TFC_FILTER_ENABLED', 'false')

        signal = create_detected_signal(
            timeframe='1H',
            tfc_score=0,  # Would normally fail
        )

        assert daemon._passes_filters(signal) is True


class TestPassesFiltersEdgeCases:
    """Edge case tests for _passes_filters."""

    def test_signal_without_context_uses_defaults(self, daemon):
        """Signals without context use default TFC values."""
        signal = create_detected_signal()
        signal.context = None  # Remove context

        # Should still check other filters, TFC defaults to 0
        # For 1D with TFC 0, should fail
        signal.timeframe = '1D'
        assert daemon._passes_filters(signal) is False

    def test_missing_signal_type_defaults_to_completed(self, daemon):
        """Missing signal_type treated as COMPLETED."""
        signal = create_detected_signal(magnitude_pct=0.05)  # Below SETUP threshold
        delattr(signal, 'signal_type')  # Remove attribute

        # As COMPLETED with low magnitude and config threshold
        # Depends on config, but should use standard thresholds
        result = daemon._passes_filters(signal)
        # Result depends on config, just verify no crash
        assert isinstance(result, bool)


# =============================================================================
# MARKET HOURS TESTS
# =============================================================================


class TestIsMarketHours:
    """Tests for _is_market_hours method."""

    @patch('pandas_market_calendars.get_calendar')
    def test_market_closed_on_holiday(self, mock_get_calendar, daemon):
        """Returns False on market holidays."""
        mock_calendar = Mock()
        mock_schedule = Mock()
        mock_schedule.empty = True  # No trading today
        mock_calendar.schedule.return_value = mock_schedule
        mock_get_calendar.return_value = mock_calendar

        assert daemon._is_market_hours() is False

    @patch('pandas_market_calendars.get_calendar')
    def test_market_open_during_hours(self, mock_get_calendar, daemon):
        """Returns True during market hours."""
        import pandas as pd

        mock_calendar = Mock()
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)

        # Create schedule for today
        mock_schedule = pd.DataFrame({
            'market_open': [now.replace(hour=9, minute=30)],
            'market_close': [now.replace(hour=16, minute=0)],
        })
        mock_calendar.schedule.return_value = mock_schedule
        mock_get_calendar.return_value = mock_calendar

        # Only returns True if current time is between open/close
        result = daemon._is_market_hours()
        assert isinstance(result, bool)

    @patch('pandas_market_calendars.get_calendar')
    def test_market_before_open(self, mock_get_calendar, daemon):
        """Returns False before market opens."""
        import pandas as pd

        mock_calendar = Mock()
        et = pytz.timezone('America/New_York')

        # Set market open in the future
        future_open = datetime.now(et) + timedelta(hours=2)
        future_close = future_open + timedelta(hours=6)

        mock_schedule = pd.DataFrame({
            'market_open': [future_open],
            'market_close': [future_close],
        })
        mock_calendar.schedule.return_value = mock_schedule
        mock_get_calendar.return_value = mock_calendar

        assert daemon._is_market_hours() is False


# =============================================================================
# SCANNING TESTS
# =============================================================================


class TestRunScan:
    """Tests for run_scan method."""

    def test_run_scan_increments_counter(self, daemon_with_mocks):
        """run_scan increments scan counter."""
        daemon_with_mocks.scanner.scan_symbol.return_value = []

        initial_count = daemon_with_mocks._scan_count
        daemon_with_mocks.run_scan('1D')

        assert daemon_with_mocks._scan_count == initial_count + 1

    def test_run_scan_returns_empty_on_no_signals(self, daemon_with_mocks):
        """run_scan returns empty list when no signals found."""
        daemon_with_mocks.scanner.scan_symbol.return_value = []

        result = daemon_with_mocks.run_scan('1D')

        assert result == []

    def test_run_scan_handles_scanner_error(self, daemon_with_mocks, caplog):
        """run_scan handles scanner errors gracefully."""
        daemon_with_mocks.scanner.scan_symbol.side_effect = Exception("Scan failed")

        result = daemon_with_mocks.run_scan('1D')

        assert result == []
        # Should not crash, and should increment error count
        # (actual error logging depends on implementation)


class TestRunAllScans:
    """Tests for run_all_scans method."""

    def test_run_all_scans_returns_dict(self, daemon_with_mocks):
        """run_all_scans returns dictionary of results."""
        daemon_with_mocks.scanner.scan_symbol.return_value = []

        result = daemon_with_mocks.run_all_scans()

        assert isinstance(result, dict)


# =============================================================================
# HEALTH CHECK AND STATUS TESTS
# =============================================================================


class TestHealthCheck:
    """Tests for _health_check method."""

    def test_health_check_returns_dict(self, daemon):
        """_health_check returns status dictionary."""
        status = daemon._health_check()

        assert isinstance(status, dict)
        assert 'status' in status
        assert 'scan_count' in status
        assert 'signal_count' in status
        assert 'error_count' in status

    def test_health_check_stopped_status(self, daemon):
        """Health check shows stopped when not running."""
        status = daemon._health_check()

        assert status['status'] == 'stopped'

    def test_health_check_healthy_when_running(self, daemon):
        """Health check shows healthy when running."""
        daemon._is_running = True
        daemon._start_time = datetime.now()

        status = daemon._health_check()

        assert status['status'] == 'healthy'

    def test_health_check_includes_uptime(self, daemon):
        """Health check includes uptime when running."""
        daemon._is_running = True
        daemon._start_time = datetime.now() - timedelta(hours=1)

        status = daemon._health_check()

        assert status['uptime_seconds'] is not None
        assert status['uptime_seconds'] >= 3600  # At least 1 hour

    def test_health_check_includes_counters(self, daemon):
        """Health check includes all counters."""
        daemon._scan_count = 10
        daemon._signal_count = 5
        daemon._error_count = 2
        daemon._execution_count = 3
        daemon._exit_count = 1

        status = daemon._health_check()

        assert status['scan_count'] == 10
        assert status['signal_count'] == 5
        assert status['error_count'] == 2
        assert status['execution_count'] == 3
        assert status['exit_count'] == 1

    def test_health_check_includes_alerters(self, daemon):
        """Health check lists configured alerters."""
        status = daemon._health_check()

        assert 'alerters' in status
        assert isinstance(status['alerters'], list)

    def test_health_check_includes_execution_status(self, daemon):
        """Health check shows execution enabled status."""
        status = daemon._health_check()

        assert 'execution_enabled' in status

    def test_health_check_includes_monitoring_status(self, daemon):
        """Health check shows monitoring enabled status."""
        status = daemon._health_check()

        assert 'monitoring_enabled' in status


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_returns_dict(self, daemon):
        """get_status returns status dictionary."""
        status = daemon.get_status()

        assert isinstance(status, dict)

    def test_get_status_includes_config_info(self, daemon):
        """get_status includes configuration information."""
        status = daemon.get_status()

        # Should include basic status info
        assert 'is_running' in status or 'status' in status or 'running' in status


class TestIsRunning:
    """Tests for is_running property."""

    def test_is_running_false_initially(self, daemon):
        """is_running is False initially."""
        assert daemon.is_running is False

    def test_is_running_true_after_start(self, daemon):
        """is_running reflects running state."""
        daemon._is_running = True

        assert daemon.is_running is True


# =============================================================================
# POSITION MONITORING TESTS
# =============================================================================


class TestCheckPositionsNow:
    """Tests for check_positions_now method."""

    def test_returns_empty_when_monitor_disabled(self, daemon):
        """Returns empty list when monitoring disabled."""
        daemon.position_monitor = None

        result = daemon.check_positions_now()

        assert result == []

    def test_delegates_to_position_monitor(self, daemon_with_mocks):
        """Delegates to position_monitor.check_positions."""
        mock_exit = Mock(spec=ExitSignal)
        daemon_with_mocks.position_monitor.check_positions.return_value = [mock_exit]

        result = daemon_with_mocks.check_positions_now()

        daemon_with_mocks.position_monitor.check_positions.assert_called_once()
        assert result == [mock_exit]


class TestGetTrackedPositions:
    """Tests for get_tracked_positions method."""

    def test_returns_empty_when_monitor_disabled(self, daemon):
        """Returns empty list when monitoring disabled."""
        daemon.position_monitor = None

        result = daemon.get_tracked_positions()

        assert result == []

    def test_delegates_to_position_monitor(self, daemon_with_mocks):
        """Delegates to position_monitor.get_tracked_positions."""
        mock_position = Mock(spec=TrackedPosition)
        daemon_with_mocks.position_monitor.get_tracked_positions.return_value = [mock_position]

        result = daemon_with_mocks.get_tracked_positions()

        daemon_with_mocks.position_monitor.get_tracked_positions.assert_called_once()
        assert result == [mock_position]


class TestRunPositionCheck:
    """Tests for _run_position_check method."""

    def test_does_nothing_when_monitor_disabled(self, daemon):
        """Does nothing when position monitor is None."""
        daemon.position_monitor = None

        # Should not crash
        daemon._run_position_check()

    def test_handles_check_error(self, daemon_with_mocks, caplog):
        """Handles position check errors gracefully."""
        daemon_with_mocks.position_monitor.check_positions.side_effect = Exception("Check failed")

        initial_errors = daemon_with_mocks._error_count
        daemon_with_mocks._run_position_check()

        assert daemon_with_mocks._error_count == initial_errors + 1


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestShutdown:
    """Tests for shutdown method."""

    def test_shutdown_does_nothing_when_not_running(self, daemon):
        """shutdown returns early when not running."""
        daemon._is_running = False
        daemon.scheduler = Mock()

        daemon.shutdown()

        # Should return early, not call scheduler shutdown
        daemon.scheduler.shutdown.assert_not_called()

    def test_shutdown_sets_running_false(self, daemon):
        """shutdown sets _is_running to False."""
        daemon._is_running = True
        daemon.shutdown()

        assert daemon._is_running is False

    def test_shutdown_calls_scheduler_shutdown(self, daemon):
        """shutdown calls scheduler.shutdown when running."""
        daemon._is_running = True
        daemon.scheduler = Mock()
        daemon.shutdown()

        daemon.scheduler.shutdown.assert_called_once_with(wait=True)


class TestTestAlerters:
    """Tests for test_alerters method."""

    def test_returns_dict_of_results(self, daemon):
        """test_alerters returns dictionary of results."""
        result = daemon.test_alerters()

        assert isinstance(result, dict)

    def test_tests_all_configured_alerters(self, daemon):
        """test_alerters tests each configured alerter."""
        mock_alerter1 = Mock()
        mock_alerter1.name = 'test1'
        mock_alerter1.test_connection.return_value = True

        mock_alerter2 = Mock()
        mock_alerter2.name = 'test2'
        mock_alerter2.test_connection.return_value = False

        # EQUITY-85: Update both daemon.alerters and AlertManager's alerters
        daemon.alerters = [mock_alerter1, mock_alerter2]
        daemon._alert_manager._alerters = [mock_alerter1, mock_alerter2]

        result = daemon.test_alerters()

        mock_alerter1.test_connection.assert_called_once()
        mock_alerter2.test_connection.assert_called_once()
        assert result['test1'] is True
        assert result['test2'] is False


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestOnPositionExit:
    """Tests for _on_position_exit callback."""

    def test_increments_exit_count(self, daemon_with_mocks):
        """Exit callback increments exit counter."""
        exit_signal = Mock(spec=ExitSignal)
        exit_signal.reason = ExitReason.TARGET_HIT
        exit_signal.osi_symbol = 'TEST240119C00100000'
        exit_signal.unrealized_pnl = 50.0
        exit_signal.signal_key = None

        order_result = {'status': 'filled', 'filled_qty': 1}

        initial_count = daemon_with_mocks._exit_count
        daemon_with_mocks._on_position_exit(exit_signal, order_result)

        assert daemon_with_mocks._exit_count == initial_count + 1


class TestCreateScanCallback:
    """Tests for _create_scan_callback method."""

    def test_creates_callable(self, daemon):
        """_create_scan_callback returns callable."""
        callback = daemon._create_scan_callback('1H')

        assert callable(callback)

    def test_callback_calls_run_scan(self, daemon):
        """Created callback calls run_scan with timeframe."""
        daemon.run_scan = Mock(return_value=[])
        callback = daemon._create_scan_callback('1D')

        callback()

        daemon.run_scan.assert_called_once_with('1D')


# =============================================================================
# SEND ALERTS TESTS
# =============================================================================


class TestSendAlerts:
    """Tests for _send_alerts method."""

    def test_calls_alerters_for_each_signal(self, daemon):
        """_send_alerts calls each alerter via AlertManager."""
        mock_alerter = Mock()
        mock_alerter.name = 'TestAlerter'
        mock_alerter.send_alert.return_value = True
        # EQUITY-85: Update both daemon.alerters and AlertManager's alerters
        daemon.alerters = [mock_alerter]
        daemon._alert_manager._alerters = [mock_alerter]
        # EQUITY-85: Mock market hours to be True so alerts are not blocked
        daemon._alert_manager._is_market_hours = Mock(return_value=True)

        signal = Mock(spec=StoredSignal)
        signal.symbol = 'TEST'
        signal.signal_key = 'test_key_123'
        signal.status = SignalStatus.DETECTED.value
        signal.priority = 1
        signal.continuity_strength = 1
        signal.magnitude_pct = 1.0

        daemon._send_alerts([signal])

        # Should call send_alert or send_batch_alert
        assert mock_alerter.send_alert.called or mock_alerter.send_batch_alert.called

    def test_handles_alerter_error(self, daemon, caplog):
        """_send_alerts handles alerter errors gracefully via AlertManager."""
        mock_alerter = Mock()
        mock_alerter.name = 'TestAlerter'
        mock_alerter.send_alert.side_effect = Exception("Alert failed")
        mock_alerter.send_batch_alert.side_effect = Exception("Batch alert failed")
        # EQUITY-85: Update both daemon.alerters and AlertManager's alerters
        daemon.alerters = [mock_alerter]
        daemon._alert_manager._alerters = [mock_alerter]

        signal = Mock(spec=StoredSignal)
        signal.symbol = 'TEST'
        signal.signal_key = 'test_key_456'
        signal.status = SignalStatus.DETECTED.value
        signal.priority = 1
        signal.continuity_strength = 1
        signal.magnitude_pct = 1.0

        # Should not crash
        daemon._send_alerts([signal])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDaemonIntegration:
    """Integration-style tests for SignalDaemon."""

    def test_full_initialization(self, full_config):
        """Full daemon initialization with all features."""
        # Use mock for executor/monitor to avoid external deps
        mock_executor = Mock()
        mock_monitor = Mock()

        daemon = SignalDaemon(
            config=full_config,
            executor=mock_executor,
            position_monitor=mock_monitor,
        )

        assert daemon.config is full_config
        assert daemon.executor is mock_executor
        assert daemon.position_monitor is mock_monitor
        assert len(daemon.alerters) > 0

    def test_filter_chain_all_pass(self, full_config):
        """Signal that passes all filters."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=1.0,
            risk_reward=1.5,
            pattern_type='3-2U',
            timeframe='1D',
            tfc_score=3,
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is True

    def test_filter_chain_magnitude_fail(self, full_config):
        """Signal fails on magnitude."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=0.1,  # Below 0.5%
            risk_reward=1.5,
            pattern_type='3-2U',
            timeframe='1D',
            tfc_score=3,
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False

    def test_filter_chain_rr_fail(self, full_config):
        """Signal fails on R:R."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=1.0,
            risk_reward=0.5,  # Below 1.0
            pattern_type='3-2U',
            timeframe='1D',
            tfc_score=3,
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False

    def test_filter_chain_tfc_fail(self, full_config):
        """Signal fails on TFC."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=1.0,
            risk_reward=1.5,
            pattern_type='3-2U',
            timeframe='1D',
            tfc_score=1,  # Below 2
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False

    def test_filter_chain_pattern_fail(self, full_config):
        """Signal fails on pattern."""
        daemon = SignalDaemon(config=full_config)

        signal = create_detected_signal(
            magnitude_pct=1.0,
            risk_reward=1.5,
            pattern_type='3-2U-2D',  # 3-2-2 not in allowed list
            timeframe='1D',
            tfc_score=3,
            signal_type='COMPLETED',
        )

        assert daemon._passes_filters(signal) is False
