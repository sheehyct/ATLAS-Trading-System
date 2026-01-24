"""
EQUITY-85: Tests for HealthMonitor coordinator.

Tests health check generation, daily audit statistics,
paper trades loading, and alerter integration.
"""

import json
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from strat.signal_automation.coordinators.health_monitor import HealthMonitor, DaemonStats


class TestDaemonStats:
    """Tests for DaemonStats dataclass."""

    def test_default_values(self):
        """DaemonStats has sensible defaults."""
        stats = DaemonStats()

        assert stats.start_time is None
        assert stats.is_running is False
        assert stats.scan_count == 0
        assert stats.signal_count == 0
        assert stats.execution_count == 0
        assert stats.exit_count == 0
        assert stats.error_count == 0

    def test_custom_values(self):
        """DaemonStats accepts custom values."""
        start = datetime.now()
        stats = DaemonStats(
            start_time=start,
            is_running=True,
            scan_count=10,
            signal_count=5,
            execution_count=3,
            exit_count=2,
            error_count=1,
        )

        assert stats.start_time == start
        assert stats.is_running is True
        assert stats.scan_count == 10
        assert stats.signal_count == 5
        assert stats.execution_count == 3
        assert stats.exit_count == 2
        assert stats.error_count == 1


class TestHealthMonitorInit:
    """Tests for HealthMonitor initialization."""

    def test_init_with_required_args(self):
        """HealthMonitor initializes with required arguments."""
        stats = DaemonStats()
        signal_store = Mock()
        alerters = [Mock()]

        monitor = HealthMonitor(
            stats=stats,
            signal_store=signal_store,
            alerters=alerters,
        )

        assert monitor._stats == stats
        assert monitor._signal_store == signal_store
        assert monitor._alerters == alerters

    def test_init_with_optional_args(self):
        """HealthMonitor accepts optional arguments."""
        stats = DaemonStats()
        scheduler = Mock()
        position_monitor = Mock()
        paper_path = Path('/tmp/test_trades.json')
        error_callback = Mock()

        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(),
            alerters=[],
            scheduler=scheduler,
            position_monitor=position_monitor,
            paper_trades_path=paper_path,
            on_error=error_callback,
            executor_enabled=True,
        )

        assert monitor._scheduler == scheduler
        assert monitor._position_monitor == position_monitor
        assert monitor._paper_trades_path == paper_path
        assert monitor._on_error == error_callback
        assert monitor._executor_enabled is True

    def test_default_paper_trades_path(self):
        """Uses default paper trades path when not specified."""
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(),
            alerters=[],
        )

        # Should contain 'paper_trades/paper_trades.json'
        assert 'paper_trades' in str(monitor._paper_trades_path)
        assert monitor._paper_trades_path.name == 'paper_trades.json'

    def test_paper_trades_path_property(self):
        """paper_trades_path property returns configured path."""
        custom_path = Path('/custom/path/trades.json')
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(),
            alerters=[],
            paper_trades_path=custom_path,
        )

        assert monitor.paper_trades_path == custom_path


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.fixture
    def running_stats(self):
        """Create stats for a running daemon."""
        return DaemonStats(
            start_time=datetime.now(),
            is_running=True,
            scan_count=100,
            signal_count=50,
            execution_count=10,
            exit_count=5,
            error_count=2,
        )

    @pytest.fixture
    def mock_signal_store(self):
        """Create mock signal store with signals."""
        store = Mock()
        store.__len__ = Mock(return_value=25)
        return store

    @pytest.fixture
    def mock_scheduler(self):
        """Create mock scheduler."""
        scheduler = Mock()
        scheduler.get_status.return_value = {'jobs': 5, 'running': True}
        return scheduler

    def test_health_check_running_daemon(self, running_stats, mock_signal_store, mock_scheduler):
        """Health check returns correct status for running daemon."""
        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[],
            scheduler=mock_scheduler,
            executor_enabled=True,
        )

        status = monitor.health_check()

        assert status['status'] == 'healthy'
        assert status['uptime_seconds'] is not None
        assert status['uptime_seconds'] >= 0
        assert status['scan_count'] == 100
        assert status['signal_count'] == 50
        assert status['execution_count'] == 10
        assert status['exit_count'] == 5
        assert status['error_count'] == 2
        assert status['signals_in_store'] == 25
        assert status['execution_enabled'] is True
        assert status['monitoring_enabled'] is False  # No position_monitor

    def test_health_check_stopped_daemon(self):
        """Health check returns 'stopped' status when not running."""
        stats = DaemonStats(is_running=False)

        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
        )

        status = monitor.health_check()

        assert status['status'] == 'stopped'
        assert status['uptime_seconds'] is None

    def test_health_check_with_position_monitor(self, running_stats, mock_signal_store):
        """Health check includes position monitor stats."""
        position_monitor = Mock()
        position_monitor.get_stats.return_value = {'active_positions': 3}

        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[],
            position_monitor=position_monitor,
        )

        status = monitor.health_check()

        assert status['monitoring_enabled'] is True
        assert status['monitoring'] == {'active_positions': 3}
        position_monitor.get_stats.assert_called_once()

    def test_health_check_alerter_names(self, running_stats, mock_signal_store):
        """Health check includes alerter names."""
        alerter1 = Mock()
        alerter1.name = 'Discord'
        alerter2 = Mock()
        alerter2.name = 'Logging'

        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[alerter1, alerter2],
        )

        status = monitor.health_check()

        assert status['alerters'] == ['Discord', 'Logging']

    def test_health_check_scheduler_status(self, running_stats, mock_signal_store, mock_scheduler):
        """Health check includes scheduler status."""
        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[],
            scheduler=mock_scheduler,
        )

        status = monitor.health_check()

        assert status['scheduler'] == {'jobs': 5, 'running': True}
        mock_scheduler.get_status.assert_called_once()

    def test_health_check_logs_via_logging_alerter(self, running_stats, mock_signal_store):
        """Health check logs status via LoggingAlerter."""
        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerter = Mock(spec=LoggingAlerter)
        logging_alerter.name = 'Logging'

        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[logging_alerter],
        )

        monitor.health_check()

        logging_alerter.log_health_check.assert_called_once()

    def test_health_check_handles_position_monitor_error(self, running_stats, mock_signal_store):
        """Health check handles position monitor errors gracefully."""
        position_monitor = Mock()
        position_monitor.get_stats.side_effect = Exception("Monitor error")

        monitor = HealthMonitor(
            stats=running_stats,
            signal_store=mock_signal_store,
            alerters=[],
            position_monitor=position_monitor,
        )

        status = monitor.health_check()

        assert 'error' in status['monitoring']


class TestGenerateDailyAudit:
    """Tests for generate_daily_audit method."""

    @pytest.fixture
    def temp_trades_file(self):
        """Create a temporary paper trades file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trades = [
                {
                    'exit_time': date.today().isoformat() + 'T10:30:00',
                    'pnl_dollars': 150.0,
                },
                {
                    'exit_time': date.today().isoformat() + 'T14:00:00',
                    'pnl_dollars': -50.0,
                },
                {
                    'exit_time': '2025-01-01T10:00:00',  # Old trade
                    'pnl_dollars': 200.0,
                },
            ]
            json.dump(trades, f)
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    def test_audit_counts_todays_trades(self, temp_trades_file):
        """Audit correctly counts today's trades."""
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            paper_trades_path=temp_trades_file,
        )

        audit = monitor.generate_daily_audit()

        assert audit['trades_today'] == 2  # Only 2 trades from today
        assert audit['wins'] == 1
        assert audit['losses'] == 1
        assert audit['total_pnl'] == 100.0  # 150 - 50

    def test_audit_calculates_profit_factor(self, temp_trades_file):
        """Audit calculates profit factor correctly."""
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            paper_trades_path=temp_trades_file,
        )

        audit = monitor.generate_daily_audit()

        # profit_factor = gross_profit / gross_loss = 150 / 50 = 3.0
        assert audit['profit_factor'] == 3.0

    def test_audit_handles_missing_file(self):
        """Audit handles missing paper trades file."""
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        audit = monitor.generate_daily_audit()

        assert audit['trades_today'] == 0
        assert len(audit['anomalies']) == 1
        assert 'not found' in audit['anomalies'][0]

    def test_audit_handles_invalid_json(self):
        """Audit handles invalid JSON in trades file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json {')
            f.flush()
            temp_path = Path(f.name)

        try:
            monitor = HealthMonitor(
                stats=DaemonStats(),
                signal_store=Mock(__len__=Mock(return_value=0)),
                alerters=[],
                paper_trades_path=temp_path,
            )

            audit = monitor.generate_daily_audit()

            assert audit['trades_today'] == 0
            assert len(audit['anomalies']) == 1
        finally:
            temp_path.unlink(missing_ok=True)

    def test_audit_includes_date(self):
        """Audit includes today's date."""
        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        audit = monitor.generate_daily_audit()

        assert audit['date'] == date.today().isoformat()

    def test_audit_includes_open_positions(self):
        """Audit includes open positions from position monitor."""
        position = Mock()
        position.symbol = 'AAPL'
        position.pattern_type = '3-2U'
        position.timeframe = '1H'
        position.unrealized_pnl = 75.0
        position.unrealized_pct = 15.0

        position_monitor = Mock()
        position_monitor.get_tracked_positions.return_value = [position]

        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            position_monitor=position_monitor,
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        audit = monitor.generate_daily_audit()

        assert len(audit['open_positions']) == 1
        assert audit['open_positions'][0]['symbol'] == 'AAPL'
        assert audit['open_positions'][0]['pattern_type'] == '3-2U'
        assert audit['open_positions'][0]['unrealized_pnl'] == 75.0

    def test_audit_handles_no_losses(self):
        """Audit handles case with no losses (avoids division by zero)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trades = [
                {
                    'exit_time': date.today().isoformat() + 'T10:30:00',
                    'pnl_dollars': 100.0,
                },
            ]
            json.dump(trades, f)
            f.flush()
            temp_path = Path(f.name)

        try:
            monitor = HealthMonitor(
                stats=DaemonStats(),
                signal_store=Mock(__len__=Mock(return_value=0)),
                alerters=[],
                paper_trades_path=temp_path,
            )

            audit = monitor.generate_daily_audit()

            assert audit['profit_factor'] == 0.0  # No losses, so 0
        finally:
            temp_path.unlink(missing_ok=True)

    def test_audit_handles_trades_dict_format(self):
        """Audit handles trades in dict format with 'trades' key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                'trades': [
                    {
                        'exit_time': date.today().isoformat() + 'T10:30:00',
                        'pnl_dollars': 200.0,
                    },
                ]
            }
            json.dump(data, f)
            f.flush()
            temp_path = Path(f.name)

        try:
            monitor = HealthMonitor(
                stats=DaemonStats(),
                signal_store=Mock(__len__=Mock(return_value=0)),
                alerters=[],
                paper_trades_path=temp_path,
            )

            audit = monitor.generate_daily_audit()

            assert audit['trades_today'] == 1
            assert audit['total_pnl'] == 200.0
        finally:
            temp_path.unlink(missing_ok=True)


class TestRunDailyAudit:
    """Tests for run_daily_audit method."""

    def test_sends_audit_to_alerters(self):
        """run_daily_audit sends audit data to alerters with send_daily_audit method."""
        alerter = Mock()
        alerter.name = 'Discord'

        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[alerter],
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        monitor.run_daily_audit()

        alerter.send_daily_audit.assert_called_once()
        # Check audit data was passed
        audit_data = alerter.send_daily_audit.call_args[0][0]
        assert 'date' in audit_data
        assert 'trades_today' in audit_data

    def test_skips_alerters_without_send_daily_audit(self):
        """run_daily_audit skips alerters without send_daily_audit method."""
        alerter_with = Mock()
        alerter_with.name = 'Discord'

        alerter_without = Mock(spec=[])  # No methods
        alerter_without.name = 'Simple'

        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[alerter_with, alerter_without],
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        # Should not raise
        monitor.run_daily_audit()

        alerter_with.send_daily_audit.assert_called_once()

    def test_handles_alerter_error(self):
        """run_daily_audit handles alerter errors gracefully."""
        alerter = Mock()
        alerter.name = 'Discord'
        alerter.send_daily_audit.side_effect = Exception("Network error")

        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[alerter],
            paper_trades_path=Path('/nonexistent/trades.json'),
        )

        # Should not raise
        monitor.run_daily_audit()

    def test_calls_error_callback_on_failure(self):
        """run_daily_audit calls error callback on complete failure."""
        error_callback = Mock()

        monitor = HealthMonitor(
            stats=DaemonStats(),
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
            paper_trades_path=Path('/nonexistent/trades.json'),
            on_error=error_callback,
        )

        # Patch generate_daily_audit to raise
        with patch.object(monitor, 'generate_daily_audit', side_effect=Exception("Audit failed")):
            monitor.run_daily_audit()

        error_callback.assert_called_once()


class TestUpdateStats:
    """Tests for update_stats method."""

    def test_updates_single_stat(self):
        """update_stats updates a single stat."""
        stats = DaemonStats()
        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
        )

        monitor.update_stats(scan_count=42)

        assert stats.scan_count == 42

    def test_updates_multiple_stats(self):
        """update_stats updates multiple stats at once."""
        stats = DaemonStats()
        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
        )

        monitor.update_stats(
            scan_count=100,
            signal_count=50,
            error_count=5,
        )

        assert stats.scan_count == 100
        assert stats.signal_count == 50
        assert stats.error_count == 5

    def test_ignores_unknown_stats(self):
        """update_stats ignores unknown stat names."""
        stats = DaemonStats()
        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
        )

        # Should not raise
        monitor.update_stats(unknown_field=123)

        assert not hasattr(stats, 'unknown_field')


class TestIntegration:
    """Integration tests for HealthMonitor."""

    def test_full_health_monitoring_workflow(self):
        """Test complete health monitoring workflow."""
        # Setup
        stats = DaemonStats(
            start_time=datetime.now(),
            is_running=True,
            scan_count=50,
        )

        signal_store = Mock()
        signal_store.__len__ = Mock(return_value=10)

        scheduler = Mock()
        scheduler.get_status.return_value = {'jobs': 3}

        position_monitor = Mock()
        position_monitor.get_stats.return_value = {'active': 2}
        position = Mock(
            symbol='SPY',
            pattern_type='2-1-2U',
            timeframe='1D',
            unrealized_pnl=500.0,
            unrealized_pct=25.0,
        )
        position_monitor.get_tracked_positions.return_value = [position]

        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerter = Mock(spec=LoggingAlerter)
        logging_alerter.name = 'Logging'

        # Create monitor
        monitor = HealthMonitor(
            stats=stats,
            signal_store=signal_store,
            alerters=[logging_alerter],
            scheduler=scheduler,
            position_monitor=position_monitor,
            executor_enabled=True,
        )

        # Perform health check
        health = monitor.health_check()

        # Verify
        assert health['status'] == 'healthy'
        assert health['scan_count'] == 50
        assert health['signals_in_store'] == 10
        assert health['monitoring']['active'] == 2
        logging_alerter.log_health_check.assert_called_once()

    def test_health_check_with_subsequent_stat_updates(self):
        """Test that stat updates are reflected in subsequent health checks."""
        stats = DaemonStats(is_running=True)
        monitor = HealthMonitor(
            stats=stats,
            signal_store=Mock(__len__=Mock(return_value=0)),
            alerters=[],
        )

        # Initial health check
        health1 = monitor.health_check()
        assert health1['scan_count'] == 0

        # Update stats
        monitor.update_stats(scan_count=10)

        # Second health check should reflect update
        health2 = monitor.health_check()
        assert health2['scan_count'] == 10
