"""
EQUITY-84: Tests for AlertManager coordinator.

Tests alert routing, market hours blocking, signal sorting,
and error handling for Discord and logging alerters.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from strat.signal_automation.coordinators.alert_manager import AlertManager
from strat.signal_automation.signal_store import SignalStatus


class TestAlertManagerInit:
    """Tests for AlertManager initialization."""

    def test_init_with_required_args(self):
        """AlertManager initializes with required arguments."""
        alerters = [Mock()]
        signal_store = Mock()
        config = Mock(alert_on_signal_detection=True)
        is_market_hours = Mock(return_value=True)

        manager = AlertManager(
            alerters=alerters,
            signal_store=signal_store,
            config_alerts=config,
            is_market_hours_fn=is_market_hours,
        )

        assert manager._alerters == alerters
        assert manager._signal_store == signal_store
        assert manager._config == config

    def test_init_with_error_callback(self):
        """AlertManager accepts optional error callback."""
        error_callback = Mock()

        manager = AlertManager(
            alerters=[],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
            on_error=error_callback,
        )

        assert manager._on_error == error_callback

    def test_alerters_property(self):
        """alerters property returns alerter list."""
        alerters = [Mock(), Mock()]

        manager = AlertManager(
            alerters=alerters,
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        assert manager.alerters == alerters


class TestSendSignalAlerts:
    """Tests for send_signal_alerts method."""

    @pytest.fixture
    def mock_signal(self):
        """Create a mock signal."""
        signal = Mock()
        signal.status = SignalStatus.DETECTED.value
        signal.signal_key = "test_key_123"
        signal.priority = 5
        signal.continuity_strength = 3
        signal.magnitude_pct = 1.5
        return signal

    @pytest.fixture
    def mock_alerter(self):
        """Create a mock logging alerter."""
        alerter = Mock()
        alerter.name = "TestAlerter"
        alerter.send_alert.return_value = True
        return alerter

    @pytest.fixture
    def manager_with_alerter(self, mock_alerter):
        """Create AlertManager with a mock alerter."""
        return AlertManager(
            alerters=[mock_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=True),
        )

    def test_sends_alerts_during_market_hours(self, manager_with_alerter, mock_alerter, mock_signal):
        """Alerts are sent during market hours."""
        manager_with_alerter.send_signal_alerts([mock_signal])

        mock_alerter.send_alert.assert_called_once_with(mock_signal)

    def test_blocks_alerts_outside_market_hours(self, mock_alerter, mock_signal):
        """Alerts are blocked outside market hours."""
        manager = AlertManager(
            alerters=[mock_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=False),  # Outside market hours
        )

        manager.send_signal_alerts([mock_signal])

        mock_alerter.send_alert.assert_not_called()

    def test_marks_signals_alerted_outside_market_hours(self, mock_signal):
        """Signals are marked as alerted even outside market hours."""
        signal_store = Mock()
        manager = AlertManager(
            alerters=[Mock()],
            signal_store=signal_store,
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=False),
        )

        manager.send_signal_alerts([mock_signal])

        signal_store.mark_alerted.assert_called_once_with(mock_signal.signal_key)

    def test_skips_historical_triggered_signals(self):
        """Historical triggered signals are not marked as alerted again."""
        signal = Mock()
        signal.status = SignalStatus.HISTORICAL_TRIGGERED.value
        signal.signal_key = "historical_key"

        signal_store = Mock()
        manager = AlertManager(
            alerters=[Mock()],
            signal_store=signal_store,
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=False),
        )

        manager.send_signal_alerts([signal])

        signal_store.mark_alerted.assert_not_called()

    def test_sorts_signals_by_priority(self, mock_alerter):
        """Signals are sorted by priority before alerting."""
        # Remove send_batch_alert so individual alerts are used
        del mock_alerter.send_batch_alert

        signals = []
        for i, (priority, strength, mag) in enumerate([(1, 1, 0.5), (3, 2, 1.5), (2, 3, 1.0)]):
            s = Mock()
            s.status = SignalStatus.DETECTED.value
            s.signal_key = f"key_{i}"
            s.priority = priority
            s.continuity_strength = strength
            s.magnitude_pct = mag
            signals.append(s)

        manager = AlertManager(
            alerters=[mock_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=True),
        )

        manager.send_signal_alerts(signals)

        # Verify send_alert was called 3 times
        assert mock_alerter.send_alert.call_count == 3
        # First call should be highest priority (3)
        first_call_signal = mock_alerter.send_alert.call_args_list[0][0][0]
        assert first_call_signal.priority == 3

    def test_uses_batch_alert_for_multiple_signals(self, mock_alerter):
        """Uses batch_alert method when alerter supports it and multiple signals."""
        mock_alerter.send_batch_alert = Mock(return_value=True)

        signals = [Mock(status=SignalStatus.DETECTED.value, signal_key="k1"),
                   Mock(status=SignalStatus.DETECTED.value, signal_key="k2")]
        for s in signals:
            s.priority = 1
            s.continuity_strength = 1
            s.magnitude_pct = 1.0

        manager = AlertManager(
            alerters=[mock_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=True),
        )

        manager.send_signal_alerts(signals)

        mock_alerter.send_batch_alert.assert_called_once()
        mock_alerter.send_alert.assert_not_called()

    def test_handles_alerter_error(self, mock_alerter, mock_signal):
        """Handles alerter errors gracefully."""
        mock_alerter.send_alert.side_effect = Exception("Network error")
        error_callback = Mock()

        manager = AlertManager(
            alerters=[mock_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=True),
            on_error=error_callback,
        )

        # Should not raise
        manager.send_signal_alerts([mock_signal])

        error_callback.assert_called_once()


class TestDiscordAlerterRouting:
    """Tests for Discord-specific alerter routing."""

    @pytest.fixture
    def mock_discord_alerter(self):
        """Create a mock Discord alerter."""
        # Create a proper mock that isinstance checks work with
        from strat.signal_automation.alerters import DiscordAlerter
        alerter = Mock(spec=DiscordAlerter)
        alerter.name = "Discord"
        alerter.send_alert.return_value = True
        return alerter

    def test_blocks_discord_when_alert_on_signal_detection_false(self, mock_discord_alerter):
        """Discord alerts are blocked when alert_on_signal_detection is False."""
        signal = Mock()
        signal.status = SignalStatus.DETECTED.value
        signal.signal_key = "test_key"
        signal.priority = 1
        signal.continuity_strength = 1
        signal.magnitude_pct = 1.0

        signal_store = Mock()
        manager = AlertManager(
            alerters=[mock_discord_alerter],
            signal_store=signal_store,
            config_alerts=Mock(alert_on_signal_detection=False),  # Disabled
            is_market_hours_fn=Mock(return_value=True),
        )

        manager.send_signal_alerts([signal])

        mock_discord_alerter.send_alert.assert_not_called()
        # Signal should still be marked as alerted
        signal_store.mark_alerted.assert_called_once_with(signal.signal_key)

    def test_allows_discord_when_alert_on_signal_detection_true(self, mock_discord_alerter):
        """Discord alerts are sent when alert_on_signal_detection is True."""
        signal = Mock()
        signal.status = SignalStatus.DETECTED.value
        signal.signal_key = "test_key"
        signal.priority = 1
        signal.continuity_strength = 1
        signal.magnitude_pct = 1.0

        manager = AlertManager(
            alerters=[mock_discord_alerter],
            signal_store=Mock(),
            config_alerts=Mock(alert_on_signal_detection=True),
            is_market_hours_fn=Mock(return_value=True),
        )

        manager.send_signal_alerts([signal])

        mock_discord_alerter.send_alert.assert_called_once()


class TestSendEntryAlert:
    """Tests for send_entry_alert method."""

    def test_sends_discord_entry_alert(self):
        """Sends entry alert via Discord alerter."""
        from strat.signal_automation.alerters import DiscordAlerter
        discord = Mock(spec=DiscordAlerter)
        discord.name = "Discord"

        signal = Mock()
        result = Mock()

        manager = AlertManager(
            alerters=[discord],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        manager.send_entry_alert(signal, result)

        discord.send_entry_alert.assert_called_once_with(signal, result)

    def test_sends_logging_entry_alert(self):
        """Logs entry via logging alerter (fallback if log_execution not available)."""
        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerter = Mock(spec=LoggingAlerter)
        logging_alerter.name = "Logging"

        signal = Mock()
        signal.symbol = "AAPL"
        signal.direction = "CALL"
        result = Mock()

        manager = AlertManager(
            alerters=[logging_alerter],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        # Should not raise even without log_execution
        manager.send_entry_alert(signal, result)

    def test_handles_entry_alert_error(self):
        """Handles entry alert errors gracefully."""
        from strat.signal_automation.alerters import DiscordAlerter
        discord = Mock(spec=DiscordAlerter)
        discord.name = "Discord"
        discord.send_entry_alert.side_effect = Exception("Send failed")

        error_callback = Mock()
        manager = AlertManager(
            alerters=[discord],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
            on_error=error_callback,
        )

        # Should not raise
        manager.send_entry_alert(Mock(), Mock())

        error_callback.assert_called_once()


class TestSendExitAlert:
    """Tests for send_exit_alert method."""

    def test_sends_discord_exit_alert(self):
        """Sends exit alert via Discord alerter."""
        from strat.signal_automation.alerters import DiscordAlerter
        discord = Mock(spec=DiscordAlerter)
        discord.name = "Discord"

        exit_signal = Mock()
        exit_signal.reason = Mock(value="TARGET_HIT")
        exit_signal.osi_symbol = "AAPL240119C00150000"
        exit_signal.unrealized_pnl = 250.00

        signal = Mock()
        signal.symbol = "AAPL"
        signal.pattern_type = "3-2U"
        signal.timeframe = "1H"
        signal.direction = "CALL"

        manager = AlertManager(
            alerters=[discord],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        manager.send_exit_alert(exit_signal, {}, signal)

        discord.send_simple_exit_alert.assert_called_once_with(
            symbol="AAPL",
            pattern_type="3-2U",
            timeframe="1H",
            direction="CALL",
            exit_reason="TARGET_HIT",
            pnl=250.00,
        )

    def test_sends_exit_alert_without_signal(self):
        """Sends exit alert when original signal not available."""
        from strat.signal_automation.alerters import DiscordAlerter
        discord = Mock(spec=DiscordAlerter)
        discord.name = "Discord"

        exit_signal = Mock()
        exit_signal.reason = Mock(value="STOP_HIT")
        exit_signal.osi_symbol = "SPY240119P00450000"
        exit_signal.unrealized_pnl = -100.00

        manager = AlertManager(
            alerters=[discord],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        manager.send_exit_alert(exit_signal, {}, signal=None)

        discord.send_simple_exit_alert.assert_called_once()
        call_kwargs = discord.send_simple_exit_alert.call_args[1]
        assert call_kwargs['symbol'] == "SPY"  # Extracted from OSI symbol
        assert call_kwargs['pattern_type'] == "Unknown"
        assert call_kwargs['exit_reason'] == "STOP_HIT"

    def test_sends_logging_exit_alert(self):
        """Sends exit alert via logging alerter."""
        from strat.signal_automation.alerters import LoggingAlerter
        logging_alerter = Mock(spec=LoggingAlerter)
        logging_alerter.name = "Logging"

        exit_signal = Mock()
        order_result = {'order_id': '123'}

        manager = AlertManager(
            alerters=[logging_alerter],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        manager.send_exit_alert(exit_signal, order_result)

        logging_alerter.log_position_exit.assert_called_once_with(exit_signal, order_result)

    def test_handles_exit_alert_error(self):
        """Handles exit alert errors gracefully."""
        from strat.signal_automation.alerters import DiscordAlerter
        discord = Mock(spec=DiscordAlerter)
        discord.name = "Discord"
        discord.send_simple_exit_alert.side_effect = Exception("Network error")

        error_callback = Mock()
        manager = AlertManager(
            alerters=[discord],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
            on_error=error_callback,
        )

        exit_signal = Mock()
        exit_signal.reason = Mock(value="TARGET_HIT")
        exit_signal.osi_symbol = "AAPL240119C00150000"
        exit_signal.unrealized_pnl = 100.0

        # Should not raise
        manager.send_exit_alert(exit_signal, {}, Mock(symbol="AAPL", pattern_type="3-2U", timeframe="1H", direction="CALL"))

        error_callback.assert_called_once()


class TestTestAlerters:
    """Tests for test_alerters method."""

    def test_returns_success_status_for_each_alerter(self):
        """Returns dict with success status for each alerter."""
        alerter1 = Mock()
        alerter1.name = "Discord"
        alerter1.test_connection.return_value = True

        alerter2 = Mock()
        alerter2.name = "Logging"
        alerter2.test_connection.return_value = True

        manager = AlertManager(
            alerters=[alerter1, alerter2],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        results = manager.test_alerters()

        assert results == {"Discord": True, "Logging": True}

    def test_handles_alerter_test_failure(self):
        """Handles alerter test failures gracefully."""
        alerter = Mock()
        alerter.name = "Discord"
        alerter.test_connection.return_value = False

        manager = AlertManager(
            alerters=[alerter],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        results = manager.test_alerters()

        assert results == {"Discord": False}

    def test_handles_alerter_test_exception(self):
        """Handles alerter test exceptions gracefully."""
        alerter = Mock()
        alerter.name = "Discord"
        alerter.test_connection.side_effect = Exception("Connection failed")

        manager = AlertManager(
            alerters=[alerter],
            signal_store=Mock(),
            config_alerts=Mock(),
            is_market_hours_fn=Mock(),
        )

        results = manager.test_alerters()

        assert results == {"Discord": False}
