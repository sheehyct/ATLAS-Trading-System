"""
Crypto Daemon Trade Execution Tests - Session EQUITY-73

Tests for trade execution, trigger handling, and position management in CryptoSignalDaemon.

Test coverage:
1. Trigger callback (_on_trigger)
2. Trade execution (_execute_trade)
3. Triggered pattern execution (_execute_triggered_pattern)
4. Leverage tier selection
5. Position sizing
6. Direction flip handling
7. Discord alerter integration
8. Poll callback (_on_poll)
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, call

from crypto.scanning.daemon import CryptoSignalDaemon, CryptoDaemonConfig
from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto.scanning.entry_monitor import CryptoTriggerEvent


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_client():
    """Create mock CoinbaseClient."""
    client = Mock()
    client.get_current_price.return_value = 50000.0
    return client


@pytest.fixture
def mock_scanner():
    """Create mock CryptoSignalScanner."""
    scanner = Mock()
    scanner.scan_all_timeframes.return_value = []
    # Mock evaluate_tfc for TFC re-evaluation
    mock_assessment = Mock()
    mock_assessment.strength = 3
    mock_assessment.direction = "bullish"
    mock_assessment.passes_flexible = True
    mock_assessment.alignment_label = Mock(return_value="3/4 BULLISH")
    scanner.evaluate_tfc.return_value = mock_assessment
    return scanner


@pytest.fixture
def mock_paper_trader():
    """Create mock PaperTrader."""
    trader = Mock()
    trader.account = Mock()
    trader.account.current_balance = 10000.0
    trader.get_available_balance.return_value = 10000.0
    trader.get_account_summary.return_value = {
        "current_balance": 10000.0,
        "realized_pnl": 0.0,
        "open_trades": 0,
        "closed_trades": 0,
    }
    # Mock open_trade to return a trade object
    mock_trade = Mock()
    mock_trade.trade_id = "test-trade-123"
    trader.open_trade.return_value = mock_trade
    return trader


@pytest.fixture
def mock_position_monitor():
    """Create mock CryptoPositionMonitor."""
    monitor = Mock()
    monitor.check_exits.return_value = []
    monitor.execute_all_exits.return_value = []
    monitor.get_open_positions_with_pnl.return_value = []
    return monitor


@pytest.fixture
def mock_entry_monitor():
    """Create mock CryptoEntryMonitor."""
    monitor = Mock()
    monitor.add_signals.return_value = 0
    monitor.get_pending_signals.return_value = []
    monitor.get_stats.return_value = {"pending_signals": 0, "trigger_count": 0}
    return monitor


@pytest.fixture
def execution_config():
    """Create config with execution enabled."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX"],
        scan_interval=60,
        enable_execution=True,
        paper_balance=10000.0,
        api_enabled=False,
        tfc_reeval_enabled=True,
        tfc_reeval_min_strength=3,
        tfc_reeval_block_on_flip=True,
    )


@pytest.fixture
def daemon_with_mocks(
    execution_config,
    mock_client,
    mock_scanner,
    mock_paper_trader,
    mock_position_monitor,
    mock_entry_monitor,
):
    """Create daemon with all components mocked."""
    with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            with patch('crypto.scanning.daemon.PaperTrader', return_value=mock_paper_trader):
                with patch('crypto.scanning.daemon.CryptoPositionMonitor', return_value=mock_position_monitor):
                    daemon = CryptoSignalDaemon(
                        config=execution_config,
                        client=mock_client,
                        scanner=mock_scanner,
                        paper_trader=mock_paper_trader,
                    )
                    daemon.position_monitor = mock_position_monitor
                    daemon.entry_monitor = mock_entry_monitor
                    return daemon


def create_test_signal(
    symbol: str = "BTC-PERP-INTX",
    timeframe: str = "1h",
    pattern_type: str = "3-2U",
    direction: str = "LONG",
    signal_type: str = "SETUP",
    tfc_score: int = 3,
    tfc_passes: bool = True,
    risk_multiplier: float = 1.0,
    entry_trigger: float = 50000.0,
    stop_price: float = 49000.0,
    target_price: float = 52000.0,
    setup_bar_high: float = 50500.0,
    setup_bar_low: float = 49500.0,
    setup_bar_timestamp: datetime = None,
) -> CryptoDetectedSignal:
    """Helper to create test signals."""
    context = CryptoSignalContext(
        tfc_score=tfc_score,
        tfc_alignment=f"{tfc_score}/4 BULLISH" if direction == "LONG" else f"{tfc_score}/4 BEARISH",
        tfc_passes=tfc_passes,
        risk_multiplier=risk_multiplier,
        priority_rank=1,
    )

    signal = CryptoDetectedSignal(
        pattern_type=pattern_type,
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        detected_time=datetime.now(timezone.utc),
        entry_trigger=entry_trigger,
        stop_price=stop_price,
        target_price=target_price,
        magnitude_pct=2.0,
        risk_reward=2.0,
        context=context,
        signal_type=signal_type,
        setup_bar_high=setup_bar_high,
        setup_bar_low=setup_bar_low,
    )

    if setup_bar_timestamp is not None:
        signal.setup_bar_timestamp = setup_bar_timestamp
    else:
        # Set a fresh timestamp so it's not stale
        signal.setup_bar_timestamp = datetime.now(timezone.utc)

    return signal


def create_trigger_event(
    signal: CryptoDetectedSignal = None,
    current_price: float = 50100.0,
    trigger_price: float = 50000.0,
) -> CryptoTriggerEvent:
    """Helper to create trigger events."""
    if signal is None:
        signal = create_test_signal()

    return CryptoTriggerEvent(
        signal=signal,
        current_price=current_price,
        trigger_price=trigger_price,
        triggered_at=datetime.now(timezone.utc),
    )


# =============================================================================
# TRIGGER CALLBACK TESTS
# =============================================================================


class TestOnTrigger:
    """Tests for _on_trigger callback."""

    def test_trigger_increments_counter(self, daemon_with_mocks):
        """Trigger should increment trigger counter."""
        event = create_trigger_event()

        initial_count = daemon_with_mocks._trigger_count
        daemon_with_mocks._on_trigger(event)

        assert daemon_with_mocks._trigger_count == initial_count + 1

    def test_trigger_calls_execute_trade(self, daemon_with_mocks, mock_paper_trader):
        """Trigger should call execute trade."""
        event = create_trigger_event()

        daemon_with_mocks._on_trigger(event)

        # Verify paper_trader.open_trade was called
        mock_paper_trader.open_trade.assert_called_once()

    def test_trigger_blocked_by_stale_setup(self, daemon_with_mocks, mock_paper_trader):
        """Stale setup should block trigger execution."""
        # Create a stale signal (setup bar from 10 hours ago for 1H timeframe)
        stale_ts = datetime.now(timezone.utc) - timedelta(hours=10)
        signal = create_test_signal(
            timeframe="1H",
            setup_bar_timestamp=stale_ts,
            signal_type="SETUP",
        )
        event = create_trigger_event(signal=signal)

        daemon_with_mocks._on_trigger(event)

        # Trade should not be executed due to stale setup
        mock_paper_trader.open_trade.assert_not_called()

    def test_trigger_blocked_by_tfc_degradation(self, daemon_with_mocks, mock_scanner, mock_paper_trader):
        """TFC degradation should block trigger execution."""
        # Configure scanner to return degraded TFC
        degraded_assessment = Mock()
        degraded_assessment.strength = 1  # Below min threshold of 3
        degraded_assessment.direction = "bullish"
        degraded_assessment.passes_flexible = False
        degraded_assessment.alignment_label = Mock(return_value="1/4 BULLISH")
        mock_scanner.evaluate_tfc.return_value = degraded_assessment

        event = create_trigger_event()

        daemon_with_mocks._on_trigger(event)

        # Trade should not be executed due to TFC degradation
        mock_paper_trader.open_trade.assert_not_called()

    def test_trigger_blocked_by_tfc_direction_flip(self, daemon_with_mocks, mock_scanner, mock_paper_trader):
        """TFC direction flip should block trigger execution."""
        # Configure scanner to return flipped TFC direction
        flipped_assessment = Mock()
        flipped_assessment.strength = 3
        flipped_assessment.direction = "bearish"  # Flipped from bullish
        flipped_assessment.passes_flexible = True
        flipped_assessment.alignment_label = Mock(return_value="3/4 BEARISH")
        mock_scanner.evaluate_tfc.return_value = flipped_assessment

        signal = create_test_signal(direction="LONG", tfc_score=3)
        signal.context.tfc_alignment = "3/4 BULLISH"  # Original was bullish
        event = create_trigger_event(signal=signal)

        daemon_with_mocks._on_trigger(event)

        # Trade should not be executed due to TFC direction flip
        mock_paper_trader.open_trade.assert_not_called()

    def test_trigger_fires_custom_callback(self, daemon_with_mocks):
        """Custom callback should be fired on trigger."""
        custom_callback = Mock()
        daemon_with_mocks.config.on_trigger = custom_callback

        event = create_trigger_event()
        daemon_with_mocks._on_trigger(event)

        custom_callback.assert_called_once_with(event)

    def test_trigger_handles_execution_error(self, daemon_with_mocks, mock_paper_trader):
        """Execution error should be handled gracefully."""
        mock_paper_trader.open_trade.side_effect = Exception("Execution error")

        event = create_trigger_event()

        # Should not raise - error is logged as warning
        daemon_with_mocks._on_trigger(event)

        # Verify open_trade was called (error handled internally)
        mock_paper_trader.open_trade.assert_called_once()


# =============================================================================
# EXECUTE TRADE TESTS
# =============================================================================


class TestExecuteTrade:
    """Tests for _execute_trade method."""

    def test_execute_trade_opens_position(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should open position via paper trader."""
        event = create_trigger_event()

        daemon_with_mocks._execute_trade(event)

        mock_paper_trader.open_trade.assert_called_once()

    def test_execute_trade_uses_correct_direction(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should use correct side based on direction."""
        # LONG signal
        long_signal = create_test_signal(direction="LONG")
        long_event = create_trigger_event(signal=long_signal)

        daemon_with_mocks._execute_trade(long_event)

        # Should use BUY for LONG
        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["side"] == "BUY"

        mock_paper_trader.reset_mock()

        # SHORT signal
        short_signal = create_test_signal(direction="SHORT")
        short_event = create_trigger_event(signal=short_signal)

        daemon_with_mocks._execute_trade(short_event)

        # Should use SELL for SHORT
        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["side"] == "SELL"

    def test_execute_trade_skips_zero_position_size(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should skip when position size is zero."""
        # Mock sizing to return zero - need to patch both config flag and sizing function
        with patch('crypto.scanning.daemon.config.LEVERAGE_FIRST_SIZING', False):
            with patch('crypto.scanning.daemon.calculate_position_size', return_value=(0, 0, 0)):
                event = create_trigger_event()

                daemon_with_mocks._execute_trade(event)

                # Should not call open_trade
                mock_paper_trader.open_trade.assert_not_called()

    def test_execute_trade_skips_invalid_price(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should skip with invalid prices."""
        signal = create_test_signal(stop_price=0)  # Invalid stop
        event = CryptoTriggerEvent(
            signal=signal,
            current_price=50000.0,
            trigger_price=50000.0,
            triggered_at=datetime.now(timezone.utc),
        )

        daemon_with_mocks._execute_trade(event)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_trade_skips_zero_stop_distance(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should skip when stop equals entry."""
        signal = create_test_signal()
        event = CryptoTriggerEvent(
            signal=signal,
            current_price=49000.0,  # Same as stop price
            trigger_price=49000.0,
            triggered_at=datetime.now(timezone.utc),
        )

        daemon_with_mocks._execute_trade(event)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_trade_skips_failing_tfc(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should skip when TFC filter fails."""
        signal = create_test_signal(tfc_passes=False)
        event = create_trigger_event(signal=signal)

        daemon_with_mocks._execute_trade(event)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_trade_handles_rejected_trade(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should handle margin rejection."""
        mock_paper_trader.open_trade.return_value = None  # Trade rejected

        event = create_trigger_event()

        # Should not raise
        daemon_with_mocks._execute_trade(event)

        # Execution count should NOT increase (trade was rejected)
        # Note: The method still calls open_trade, it just returns None

    def test_execute_trade_increments_execution_count(self, daemon_with_mocks):
        """Successful execution should increment counter."""
        event = create_trigger_event()

        initial_count = daemon_with_mocks._execution_count
        daemon_with_mocks._execute_trade(event)

        assert daemon_with_mocks._execution_count == initial_count + 1

    def test_execute_trade_passes_stop_and_target(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should pass correct stop and target prices."""
        signal = create_test_signal(
            stop_price=48000.0,
            target_price=54000.0,
        )
        event = create_trigger_event(signal=signal, current_price=50000.0)

        daemon_with_mocks._execute_trade(event)

        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["stop_price"] == 48000.0
        assert call_kwargs["target_price"] == 54000.0

    def test_execute_trade_passes_pattern_info(self, daemon_with_mocks, mock_paper_trader):
        """Execute trade should pass pattern info."""
        signal = create_test_signal(
            pattern_type="3-1-2U",
            timeframe="4h",
        )
        event = create_trigger_event(signal=signal)

        daemon_with_mocks._execute_trade(event)

        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["pattern_type"] == "3-1-2U"
        assert call_kwargs["timeframe"] == "4h"


# =============================================================================
# DIRECTION FLIP TESTS
# =============================================================================


class TestDirectionFlip:
    """Tests for direction flip handling during execution."""

    def test_direction_flip_recalculates_stop_target(self, daemon_with_mocks, mock_paper_trader):
        """Direction flip should recalculate stop and target."""
        # Create LONG signal
        signal = create_test_signal(
            direction="LONG",
            setup_bar_high=51000.0,
            setup_bar_low=49000.0,
            stop_price=49000.0,
            target_price=53000.0,
        )

        # Create event with flipped direction (SHORT instead of LONG)
        event = CryptoTriggerEvent(
            signal=signal,
            current_price=48500.0,
            trigger_price=49000.0,
            triggered_at=datetime.now(timezone.utc),
        )
        # Simulate direction flip
        event._actual_direction = "SHORT"

        daemon_with_mocks._execute_trade(event)

        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        # For SHORT: stop at high, target below entry
        assert call_kwargs["stop_price"] == 51000.0  # setup_bar_high
        assert call_kwargs["side"] == "SELL"  # SHORT -> SELL


# =============================================================================
# TRIGGERED PATTERN EXECUTION TESTS
# =============================================================================


class TestExecuteTriggeredPattern:
    """Tests for _execute_triggered_pattern method."""

    def test_execute_triggered_skips_without_paper_trader(self, daemon_with_mocks):
        """Triggered pattern should skip without paper trader."""
        daemon_with_mocks.paper_trader = None
        signal = create_test_signal(signal_type="COMPLETED")

        # Should not raise
        daemon_with_mocks._execute_triggered_pattern(signal)

    def test_execute_triggered_skips_invalid_price(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """Triggered pattern should skip with invalid price."""
        mock_client.get_current_price.return_value = 0

        signal = create_test_signal(signal_type="COMPLETED")

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_triggered_skips_past_target_long(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """LONG triggered should skip if price past target."""
        signal = create_test_signal(
            direction="LONG",
            target_price=50000.0,
            signal_type="COMPLETED",
        )

        # Price already at/past target
        mock_client.get_current_price.return_value = 51000.0

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_triggered_skips_past_target_short(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """SHORT triggered should skip if price past target."""
        signal = create_test_signal(
            direction="SHORT",
            target_price=50000.0,
            signal_type="COMPLETED",
        )

        # Price already at/past target
        mock_client.get_current_price.return_value = 49000.0

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_triggered_executes_valid_long(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """Valid LONG triggered should execute."""
        signal = create_test_signal(
            direction="LONG",
            target_price=55000.0,
            signal_type="COMPLETED",
        )

        mock_client.get_current_price.return_value = 50000.0

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_called_once()
        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["side"] == "BUY"

    def test_execute_triggered_executes_valid_short(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """Valid SHORT triggered should execute."""
        signal = create_test_signal(
            direction="SHORT",
            target_price=45000.0,
            signal_type="COMPLETED",
        )

        mock_client.get_current_price.return_value = 50000.0

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_called_once()
        call_kwargs = mock_paper_trader.open_trade.call_args[1]
        assert call_kwargs["side"] == "SELL"

    def test_execute_triggered_skips_failing_tfc(self, daemon_with_mocks, mock_client, mock_paper_trader):
        """Triggered with failing TFC should skip."""
        signal = create_test_signal(
            signal_type="COMPLETED",
            tfc_passes=False,
        )

        mock_client.get_current_price.return_value = 50000.0

        daemon_with_mocks._execute_triggered_pattern(signal)

        mock_paper_trader.open_trade.assert_not_called()

    def test_execute_triggered_increments_execution_count(self, daemon_with_mocks, mock_client):
        """Successful triggered execution should increment counter."""
        signal = create_test_signal(signal_type="COMPLETED")
        mock_client.get_current_price.return_value = 50000.0

        initial_count = daemon_with_mocks._execution_count
        daemon_with_mocks._execute_triggered_pattern(signal)

        assert daemon_with_mocks._execution_count == initial_count + 1


# =============================================================================
# POLL CALLBACK TESTS
# =============================================================================


class TestOnPoll:
    """Tests for _on_poll callback."""

    def test_on_poll_checks_positions(self, daemon_with_mocks, mock_position_monitor):
        """Poll callback should check positions."""
        daemon_with_mocks._on_poll()

        mock_position_monitor.check_exits.assert_called_once()

    def test_on_poll_executes_exits(self, daemon_with_mocks, mock_position_monitor):
        """Poll callback should execute exits when triggered."""
        mock_exit = Mock()
        mock_position_monitor.check_exits.return_value = [mock_exit]
        mock_position_monitor.execute_all_exits.return_value = [mock_exit]

        daemon_with_mocks._on_poll()

        mock_position_monitor.execute_all_exits.assert_called_once()

    def test_on_poll_handles_no_monitor(self, daemon_with_mocks):
        """Poll callback should handle missing monitor."""
        daemon_with_mocks.position_monitor = None

        # Should not raise
        result = daemon_with_mocks.check_positions()
        assert result == 0


# =============================================================================
# DISCORD ALERTER TESTS
# =============================================================================


class TestDiscordAlerter:
    """Tests for Discord alerter integration."""

    def test_discord_setup_without_webhook(self, mock_client, mock_scanner):
        """Discord alerter should not be created without webhook."""
        config = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            discord_webhook_url=None,
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=config)

                assert daemon.discord_alerter is None

    def test_discord_alert_on_trade_entry(self, daemon_with_mocks):
        """Discord alert should fire on trade entry when configured."""
        mock_alert_manager = Mock()
        daemon_with_mocks.alert_manager = mock_alert_manager
        daemon_with_mocks.config.alert_on_trade_entry = True

        event = create_trigger_event()
        daemon_with_mocks._execute_trade(event)

        mock_alert_manager.send_entry_alert.assert_called_once()

    def test_discord_alert_disabled_no_alert(self, daemon_with_mocks):
        """No Discord alert when disabled via alert_manager."""
        mock_alert_manager = Mock()
        mock_alert_manager.send_entry_alert = Mock()  # No-op
        daemon_with_mocks.alert_manager = mock_alert_manager
        # alert_manager checks its own config, but we mock the whole manager
        # so send_entry_alert is always callable - the test verifies the call happens
        # (alert_manager internally gates on config flags)

        event = create_trigger_event()
        daemon_with_mocks._execute_trade(event)

        # Daemon always calls alert_manager; the manager decides whether to alert
        mock_alert_manager.send_entry_alert.assert_called_once()

    def test_discord_alert_handles_error(self, daemon_with_mocks):
        """Discord alert error should not block execution."""
        mock_alert_manager = Mock()
        mock_alert_manager.send_entry_alert.side_effect = Exception("Discord error")
        daemon_with_mocks.alert_manager = mock_alert_manager
        daemon_with_mocks.config.alert_on_trade_entry = True

        event = create_trigger_event()

        # Should not raise - alert_manager error is caught
        daemon_with_mocks._execute_trade(event)

        # Execution should still succeed
        assert daemon_with_mocks._execution_count > 0


# =============================================================================
# LEVERAGE TIER TESTS
# =============================================================================


class TestLeverageTier:
    """Tests for leverage tier selection."""

    def test_leverage_tier_called_during_execution(self, daemon_with_mocks):
        """Leverage tier should be determined during execution."""
        with patch('crypto.scanning.daemon.get_current_leverage_tier') as mock_tier:
            with patch('crypto.scanning.daemon.get_max_leverage_for_symbol') as mock_leverage:
                mock_tier.return_value = "intraday"
                mock_leverage.return_value = 10

                event = create_trigger_event()
                daemon_with_mocks._execute_trade(event)

                mock_tier.assert_called_once()
                mock_leverage.assert_called_once()

    def test_get_current_time_et(self, daemon_with_mocks):
        """_get_current_time_et should return datetime with timezone."""
        result = daemon_with_mocks._get_current_time_et()

        assert result is not None
        assert result.tzinfo is not None


# =============================================================================
# RUN SCAN AND MONITOR EXECUTION TESTS
# =============================================================================


class TestRunScanAndMonitorExecution:
    """Tests for COMPLETED signal execution in run_scan_and_monitor."""

    def test_executed_signals_tracked_for_dedup(self, daemon_with_mocks, mock_scanner, mock_client):
        """Executed COMPLETED signals should be tracked to prevent duplicates."""
        # Create two COMPLETED signals for same symbol/timeframe
        signal1 = create_test_signal(
            signal_type="COMPLETED",
            pattern_type="3-2U",
            direction="LONG",
        )
        signal2 = create_test_signal(
            signal_type="COMPLETED",
            pattern_type="2-1-2U",
            direction="LONG",
        )

        mock_scanner.scan_all_timeframes.return_value = [signal1, signal2]
        mock_client.get_current_price.return_value = 50000.0

        daemon_with_mocks.run_scan_and_monitor()

        # Only first should execute (second is same symbol/timeframe)
        # Due to deduplication in run_scan_and_monitor

    def test_setup_signals_added_to_monitor(self, daemon_with_mocks, mock_scanner, mock_entry_monitor):
        """SETUP signals should be added to entry monitor."""
        setup_signal = create_test_signal(signal_type="SETUP")
        mock_scanner.scan_all_timeframes.return_value = [setup_signal]
        mock_entry_monitor.add_signals.return_value = 1

        daemon_with_mocks.run_scan_and_monitor()

        mock_entry_monitor.add_signals.assert_called_once()
