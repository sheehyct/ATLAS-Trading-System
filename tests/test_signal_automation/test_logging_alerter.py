"""
Tests for strat/signal_automation/alerters/logging_alerter.py

Covers:
- JSONFormatter for structured logging
- LoggingAlerter initialization
- send_alert single signal logging
- send_batch_alert batch logging
- test_connection health check
- Scan lifecycle logging (started, completed, error)
- Daemon lifecycle logging (started, stopped)
- Health check logging
- Position exit logging
- Throttling integration

Session EQUITY-79: Test coverage for logging alerter module.
"""

import pytest
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from strat.signal_automation.alerters.logging_alerter import (
    JSONFormatter,
    LoggingAlerter,
)
from strat.signal_automation.signal_store import StoredSignal


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def log_file(temp_log_dir):
    """Create a path for a temporary log file."""
    return str(temp_log_dir / "test_signals.log")


@pytest.fixture
def alerter(log_file):
    """Create a LoggingAlerter instance with console output disabled."""
    return LoggingAlerter(
        log_file=log_file,
        level='DEBUG',
        console_output=False
    )


@pytest.fixture
def alerter_with_console(log_file):
    """Create a LoggingAlerter instance with console output enabled."""
    return LoggingAlerter(
        log_file=log_file,
        level='INFO',
        console_output=True
    )


@pytest.fixture
def mock_signal():
    """Create a mock StoredSignal for testing."""
    signal = MagicMock(spec=StoredSignal)
    signal.signal_key = 'SPY_3-1-2U_CALL_1H_20240115'
    signal.symbol = 'SPY'
    signal.pattern_type = '3-1-2U'
    signal.direction = 'CALL'
    signal.timeframe = '1H'
    signal.entry_trigger = 480.50
    signal.target_price = 487.50
    signal.stop_price = 477.00
    signal.risk_reward = 2.0
    signal.magnitude_pct = 1.5
    signal.vix = 18.5
    signal.market_regime = 'TREND_BULL'
    signal.detected_time = datetime(2024, 1, 15, 10, 30)
    signal.continuity_strength = 4
    signal.priority = 1
    return signal


@pytest.fixture
def mock_exit_signal():
    """Create a mock exit signal for position exit logging."""
    exit_signal = MagicMock()
    exit_signal.osi_symbol = 'SPY241220C00480000'
    exit_signal.signal_key = 'SPY_3-1-2U_CALL_1H_20240115'
    exit_signal.reason = MagicMock(value='TARGET_HIT')
    exit_signal.unrealized_pnl = 150.00
    exit_signal.underlying_price = 485.50
    exit_signal.current_option_price = 6.50
    exit_signal.dte = 15
    return exit_signal


# =============================================================================
# JSONFormatter Tests
# =============================================================================

class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_format_basic_record(self):
        """Format a basic log record to JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed['level'] == 'INFO'
        assert parsed['message'] == 'Test message'
        assert 'timestamp' in parsed

    def test_format_includes_module_and_function(self):
        """Format includes module and function name."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test.module',
            level=logging.WARNING,
            pathname='test_module.py',
            lineno=20,
            msg='Warning message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        assert 'module' in parsed
        assert 'function' in parsed

    def test_format_includes_signal_data(self):
        """Format includes signal_data when present."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Signal detected',
            args=(),
            exc_info=None
        )
        record.signal_data = {
            'symbol': 'SPY',
            'pattern': '3-1-2U',
            'direction': 'CALL'
        }
        result = formatter.format(record)
        parsed = json.loads(result)

        assert 'signal' in parsed
        assert parsed['signal']['symbol'] == 'SPY'
        assert parsed['signal']['pattern'] == '3-1-2U'

    def test_format_includes_extra_data(self):
        """Format includes extra data when present."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=10,
            msg='Health check',
            args=(),
            exc_info=None
        )
        record.extra = {'uptime': 3600, 'signals_today': 5}
        result = formatter.format(record)
        parsed = json.loads(result)

        assert 'extra' in parsed
        assert parsed['extra']['uptime'] == 3600
        assert parsed['extra']['signals_today'] == 5

    def test_format_timestamp_is_iso(self):
        """Timestamp should be ISO format."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        # Should be parseable as ISO timestamp
        timestamp = parsed['timestamp']
        assert 'T' in timestamp  # ISO format has T separator


# =============================================================================
# LoggingAlerter Initialization Tests
# =============================================================================

class TestLoggingAlerterInit:
    """Tests for LoggingAlerter initialization."""

    def test_init_creates_log_directory(self, tmp_path):
        """Initialization creates log directory if it doesn't exist."""
        log_file = str(tmp_path / "new_dir" / "signals.log")
        alerter = LoggingAlerter(log_file=log_file, console_output=False)

        assert Path(log_file).parent.exists()

    def test_init_sets_log_file_path(self, alerter, log_file):
        """Alerter stores log file path."""
        assert alerter.log_file == Path(log_file)

    def test_init_creates_logger(self, alerter):
        """Alerter creates dedicated logger."""
        assert alerter.logger is not None
        assert alerter.logger.name == 'strat.signals'

    def test_init_sets_log_level(self, log_file):
        """Alerter sets correct log level."""
        alerter = LoggingAlerter(log_file=log_file, level='WARNING', console_output=False)
        assert alerter.logger.level == logging.WARNING

    def test_init_with_debug_level(self, log_file):
        """Alerter accepts DEBUG level."""
        alerter = LoggingAlerter(log_file=log_file, level='DEBUG', console_output=False)
        assert alerter.logger.level == logging.DEBUG

    def test_init_prevents_propagation(self, alerter):
        """Logger does not propagate to root logger."""
        assert alerter.logger.propagate is False

    def test_init_clears_existing_handlers(self, log_file):
        """Initialization clears existing handlers."""
        alerter1 = LoggingAlerter(log_file=log_file, console_output=False)
        alerter2 = LoggingAlerter(log_file=log_file, console_output=False)
        # Both use same logger name, handlers should be reset
        assert len(alerter2.logger.handlers) > 0

    def test_init_adds_file_handler(self, alerter):
        """Alerter adds rotating file handler."""
        from logging.handlers import RotatingFileHandler
        file_handlers = [h for h in alerter.logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1

    def test_init_with_console_adds_stream_handler(self, alerter_with_console):
        """Alerter with console_output adds StreamHandler."""
        stream_handlers = [h for h in alerter_with_console.logger.handlers if isinstance(h, logging.StreamHandler)]
        # At least 1 stream handler (console)
        assert len(stream_handlers) >= 1

    def test_init_without_console_no_stream_handler(self, alerter):
        """Alerter without console_output has no StreamHandler (only RotatingFileHandler)."""
        from logging.handlers import RotatingFileHandler
        # Should only have RotatingFileHandler (which is a subclass of StreamHandler for files)
        handlers = alerter.logger.handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], RotatingFileHandler)

    def test_init_inherits_from_base_alerter(self, alerter):
        """LoggingAlerter inherits from BaseAlerter."""
        from strat.signal_automation.alerters.base import BaseAlerter
        assert isinstance(alerter, BaseAlerter)
        assert alerter.name == 'logging'


# =============================================================================
# send_alert Tests
# =============================================================================

class TestSendAlert:
    """Tests for send_alert method."""

    def test_send_alert_returns_true(self, alerter, mock_signal):
        """send_alert always returns True."""
        result = alerter.send_alert(mock_signal)
        assert result is True

    def test_send_alert_logs_to_file(self, alerter, mock_signal, log_file):
        """send_alert writes to log file."""
        alerter.send_alert(mock_signal)

        # Force flush by closing handlers
        for handler in alerter.logger.handlers:
            handler.flush()

        # Check file has content
        with open(log_file, 'r') as f:
            content = f.read()
            assert 'SPY' in content
            assert '3-1-2U' in content

    def test_send_alert_includes_signal_data(self, alerter, mock_signal, log_file):
        """send_alert logs signal data as JSON."""
        alerter.send_alert(mock_signal)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            # Should be valid JSON with signal data
            log_entry = json.loads(content.strip())
            assert 'signal' in log_entry
            assert log_entry['signal']['symbol'] == 'SPY'

    def test_send_alert_includes_tfc_data(self, alerter, mock_signal, log_file):
        """send_alert includes TFC continuity_strength and priority."""
        alerter.send_alert(mock_signal)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            assert log_entry['signal']['continuity_strength'] == 4
            assert log_entry['signal']['priority'] == 1

    def test_send_alert_throttled_returns_true(self, alerter, mock_signal):
        """Throttled alert still returns True (not a failure)."""
        # First alert
        alerter.send_alert(mock_signal)
        # Immediate second alert should be throttled
        result = alerter.send_alert(mock_signal)
        assert result is True

    def test_send_alert_records_for_throttling(self, alerter, mock_signal):
        """send_alert records alert time for throttling."""
        alerter.send_alert(mock_signal)
        assert mock_signal.signal_key in alerter._last_alert_time


# =============================================================================
# send_batch_alert Tests
# =============================================================================

class TestSendBatchAlert:
    """Tests for send_batch_alert method."""

    def test_send_batch_alert_returns_true(self, alerter, mock_signal):
        """send_batch_alert always returns True."""
        result = alerter.send_batch_alert([mock_signal])
        assert result is True

    def test_send_batch_alert_empty_list(self, alerter):
        """send_batch_alert handles empty list."""
        result = alerter.send_batch_alert([])
        assert result is True

    def test_send_batch_alert_logs_summary(self, alerter, mock_signal, log_file):
        """send_batch_alert logs batch summary."""
        mock_signal2 = MagicMock(spec=StoredSignal)
        mock_signal2.signal_key = 'QQQ_2-1-2D_PUT_1H_20240115'
        mock_signal2.symbol = 'QQQ'
        mock_signal2.pattern_type = '2-1-2D'
        mock_signal2.direction = 'PUT'
        mock_signal2.timeframe = '1H'
        mock_signal2.entry_trigger = 400.00
        mock_signal2.target_price = 392.00
        mock_signal2.stop_price = 404.00
        mock_signal2.risk_reward = 2.0
        mock_signal2.magnitude_pct = 2.0
        mock_signal2.vix = 18.5
        mock_signal2.market_regime = 'TREND_BEAR'
        mock_signal2.detected_time = datetime(2024, 1, 15, 10, 30)
        mock_signal2.continuity_strength = 3
        mock_signal2.priority = 1

        alerter.send_batch_alert([mock_signal, mock_signal2])

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'BATCH ALERT' in content
            assert '2 signals' in content

    def test_send_batch_alert_logs_each_signal(self, alerter, mock_signal, log_file):
        """send_batch_alert logs each signal individually."""
        mock_signal2 = MagicMock(spec=StoredSignal)
        mock_signal2.signal_key = 'AAPL_3-2U_CALL_1D_20240115'
        mock_signal2.symbol = 'AAPL'
        mock_signal2.pattern_type = '3-2U'
        mock_signal2.direction = 'CALL'
        mock_signal2.timeframe = '1D'
        mock_signal2.entry_trigger = 180.00
        mock_signal2.target_price = 183.00
        mock_signal2.stop_price = 178.00
        mock_signal2.risk_reward = 1.5
        mock_signal2.magnitude_pct = 1.7
        mock_signal2.vix = 18.5
        mock_signal2.market_regime = 'TREND_NEUTRAL'
        mock_signal2.detected_time = datetime(2024, 1, 15, 10, 30)
        mock_signal2.continuity_strength = 3
        mock_signal2.priority = 2

        alerter.send_batch_alert([mock_signal, mock_signal2])

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'SPY' in content
            assert 'AAPL' in content


# =============================================================================
# test_connection Tests
# =============================================================================

class TestTestConnection:
    """Tests for test_connection method."""

    def test_connection_returns_true(self, alerter):
        """test_connection returns True when logging works."""
        result = alerter.test_connection()
        assert result is True

    def test_connection_logs_test_message(self, alerter, log_file):
        """test_connection logs a test message."""
        alerter.test_connection()

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'connection test' in content.lower()


# =============================================================================
# Scan Lifecycle Logging Tests
# =============================================================================

class TestScanLifecycleLogging:
    """Tests for scan lifecycle logging methods."""

    def test_log_scan_started(self, alerter, log_file):
        """log_scan_started logs scan start."""
        alerter.log_scan_started('1H', ['SPY', 'QQQ', 'AAPL'])

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'SCAN STARTED' in content
            assert '1H' in content
            assert 'SPY' in content

    def test_log_scan_completed(self, alerter, log_file):
        """log_scan_completed logs scan completion."""
        alerter.log_scan_completed('1H', signals_found=5, duration_seconds=2.35)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'SCAN COMPLETED' in content
            assert '5 signals' in content
            assert '2.35' in content

    def test_log_scan_error(self, alerter, log_file):
        """log_scan_error logs scan errors."""
        alerter.log_scan_error('1H', 'Connection timeout')

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'SCAN ERROR' in content
            assert 'Connection timeout' in content


# =============================================================================
# Daemon Lifecycle Logging Tests
# =============================================================================

class TestDaemonLifecycleLogging:
    """Tests for daemon lifecycle logging methods."""

    def test_log_daemon_started(self, alerter, log_file):
        """log_daemon_started logs daemon start."""
        alerter.log_daemon_started()

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'DAEMON STARTED' in content

    def test_log_daemon_stopped_default_reason(self, alerter, log_file):
        """log_daemon_stopped logs with default reason."""
        alerter.log_daemon_stopped()

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'DAEMON STOPPED' in content
            assert 'shutdown' in content

    def test_log_daemon_stopped_custom_reason(self, alerter, log_file):
        """log_daemon_stopped logs with custom reason."""
        alerter.log_daemon_stopped(reason='User interrupt')

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'DAEMON STOPPED' in content
            assert 'User interrupt' in content


# =============================================================================
# Health Check Logging Tests
# =============================================================================

class TestHealthCheckLogging:
    """Tests for health check logging."""

    def test_log_health_check(self, alerter, log_file):
        """log_health_check logs health status."""
        status = {
            'uptime_seconds': 3600,
            'signals_today': 15,
            'positions_open': 3,
            'circuit_state': 'NORMAL'
        }
        alerter.log_health_check(status)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'HEALTH CHECK' in content

    def test_log_health_check_includes_extra(self, alerter, log_file):
        """log_health_check includes status as extra data."""
        status = {'uptime': 1800, 'memory_mb': 256}
        alerter.log_health_check(status)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            assert 'extra' in log_entry
            assert log_entry['extra']['uptime'] == 1800


# =============================================================================
# Position Exit Logging Tests
# =============================================================================

class TestPositionExitLogging:
    """Tests for position exit logging."""

    def test_log_position_exit_profit(self, alerter, mock_exit_signal, log_file):
        """log_position_exit logs profitable exit."""
        order_result = {'id': 'order123', 'status': 'filled'}
        alerter.log_position_exit(mock_exit_signal, order_result)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'POSITION EXIT' in content
            assert 'PROFIT' in content
            assert 'TARGET_HIT' in content

    def test_log_position_exit_loss(self, alerter, log_file):
        """log_position_exit logs losing exit."""
        exit_signal = MagicMock()
        exit_signal.osi_symbol = 'SPY241220P00470000'
        exit_signal.signal_key = 'SPY_3-1-2D_PUT_1H_20240115'
        exit_signal.reason = MagicMock(value='STOP_HIT')
        exit_signal.unrealized_pnl = -75.00
        exit_signal.underlying_price = 475.50
        exit_signal.current_option_price = 1.25
        exit_signal.dte = 10

        order_result = {'id': 'order456'}
        alerter.log_position_exit(exit_signal, order_result)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'POSITION EXIT' in content
            assert 'LOSS' in content
            assert 'STOP_HIT' in content

    def test_log_position_exit_includes_details(self, alerter, mock_exit_signal, log_file):
        """log_position_exit includes exit details as extra."""
        order_result = {'id': 'order789'}
        alerter.log_position_exit(mock_exit_signal, order_result)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            assert 'extra' in log_entry
            assert log_entry['extra']['osi_symbol'] == 'SPY241220C00480000'
            assert log_entry['extra']['pnl'] == 150.00

    def test_log_position_exit_none_order_result(self, alerter, mock_exit_signal, log_file):
        """log_position_exit handles None order result."""
        alerter.log_position_exit(mock_exit_signal, None)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            log_entry = json.loads(content.strip())
            assert log_entry['extra']['order_id'] is None

    def test_log_position_exit_string_reason(self, alerter, log_file):
        """log_position_exit handles string reason (not enum)."""
        exit_signal = MagicMock()
        exit_signal.osi_symbol = 'SPY241220C00490000'
        exit_signal.signal_key = 'SPY_2-1-2U_CALL_1H'
        exit_signal.reason = 'MANUAL_EXIT'  # String, not enum
        exit_signal.unrealized_pnl = 50.00
        exit_signal.underlying_price = 492.00
        exit_signal.current_option_price = 3.50
        exit_signal.dte = 20

        alerter.log_position_exit(exit_signal, {'id': 'order999'})

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            content = f.read()
            assert 'MANUAL_EXIT' in content


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for logging alerter."""

    def test_full_signal_workflow(self, alerter, mock_signal, log_file):
        """Test complete signal workflow logging."""
        # Start daemon
        alerter.log_daemon_started()

        # Start scan
        alerter.log_scan_started('1H', ['SPY', 'QQQ'])

        # Signal detected
        alerter.send_alert(mock_signal)

        # Scan completed
        alerter.log_scan_completed('1H', signals_found=1, duration_seconds=1.5)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 4  # At least 4 log entries

    def test_multiple_log_entries_valid_json(self, alerter, mock_signal, log_file):
        """All log entries should be valid JSON."""
        alerter.log_daemon_started()
        alerter.send_alert(mock_signal)
        alerter.log_scan_completed('1H', 1, 1.0)

        for handler in alerter.logger.handlers:
            handler.flush()

        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    # Each line should be valid JSON
                    parsed = json.loads(line.strip())
                    assert 'timestamp' in parsed
                    assert 'level' in parsed

    def test_log_file_rotation_config(self, log_file):
        """Verify file handler is configured for rotation."""
        from logging.handlers import RotatingFileHandler

        alerter = LoggingAlerter(log_file=log_file, console_output=False)
        file_handlers = [h for h in alerter.logger.handlers if isinstance(h, RotatingFileHandler)]

        assert len(file_handlers) == 1
        handler = file_handlers[0]
        assert handler.maxBytes == 10 * 1024 * 1024  # 10 MB
        assert handler.backupCount == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
