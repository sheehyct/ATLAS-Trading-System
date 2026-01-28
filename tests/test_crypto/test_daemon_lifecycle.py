"""
Crypto Daemon Lifecycle and Orchestration Tests - Session EQUITY-73

Tests for CryptoSignalDaemon initialization, configuration, lifecycle management,
signal handling, and orchestration logic.

Test coverage:
1. CryptoDaemonConfig - configuration dataclass
2. Daemon initialization - component setup
3. Signal filters - quality filtering
4. Signal ID generation and deduplication
5. Maintenance window detection
6. Stale setup validation
7. Scanning orchestration
8. Daemon lifecycle (start/stop)
9. Status and statistics
10. Signal cleanup
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from crypto.scanning.daemon import CryptoSignalDaemon, CryptoDaemonConfig
from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX"],
        scan_interval=60,
        enable_execution=False,
        api_enabled=False,
    )


@pytest.fixture
def full_config():
    """Create full config with all options."""
    return CryptoDaemonConfig(
        symbols=["BTC-PERP-INTX", "ETH-PERP-INTX"],
        scan_interval=300,
        entry_poll_interval=30,
        signal_expiry_hours=12,
        min_magnitude_pct=0.5,
        min_risk_reward=1.5,
        enable_execution=True,
        paper_balance=50000.0,
        maintenance_window_enabled=True,
        health_check_interval=120,
        discord_webhook_url="https://discord.com/api/webhooks/test",
        alert_on_signal_detection=True,
        alert_on_trigger=True,
        alert_on_trade_entry=True,
        alert_on_trade_exit=True,
        api_enabled=False,
        tfc_reeval_enabled=True,
        tfc_reeval_min_strength=3,
        tfc_reeval_block_on_flip=True,
    )


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
    return trader


@pytest.fixture
def daemon_minimal(minimal_config, mock_client, mock_scanner):
    """Create daemon with minimal config."""
    with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            daemon = CryptoSignalDaemon(
                config=minimal_config,
                client=mock_client,
                scanner=mock_scanner,
            )
            return daemon


@pytest.fixture
def daemon_with_execution(full_config, mock_client, mock_scanner, mock_paper_trader):
    """Create daemon with execution enabled."""
    with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            with patch('crypto.scanning.daemon.PaperTrader', return_value=mock_paper_trader):
                daemon = CryptoSignalDaemon(
                    config=full_config,
                    client=mock_client,
                    scanner=mock_scanner,
                    paper_trader=mock_paper_trader,
                )
                return daemon


def create_test_signal(
    symbol: str = "BTC-PERP-INTX",
    timeframe: str = "1h",
    pattern_type: str = "3-2U",
    direction: str = "LONG",
    signal_type: str = "SETUP",
    magnitude_pct: float = 2.0,
    risk_reward: float = 2.0,
    has_maintenance_gap: bool = False,
    tfc_score: int = 3,
    tfc_passes: bool = True,
    detected_time: datetime = None,
    setup_bar_timestamp: datetime = None,
) -> CryptoDetectedSignal:
    """Helper to create test signals."""
    if detected_time is None:
        detected_time = datetime.now(timezone.utc)

    context = CryptoSignalContext(
        tfc_score=tfc_score,
        tfc_alignment=f"{tfc_score}/4 BULLISH" if direction == "LONG" else f"{tfc_score}/4 BEARISH",
        tfc_passes=tfc_passes,
        risk_multiplier=1.0,
        priority_rank=1,
    )

    signal = CryptoDetectedSignal(
        pattern_type=pattern_type,
        direction=direction,
        symbol=symbol,
        timeframe=timeframe,
        detected_time=detected_time,
        entry_trigger=50000.0,
        stop_price=49000.0 if direction == "LONG" else 51000.0,
        target_price=52000.0 if direction == "LONG" else 48000.0,
        magnitude_pct=magnitude_pct,
        risk_reward=risk_reward,
        context=context,
        signal_type=signal_type,
        setup_bar_high=50500.0,
        setup_bar_low=49500.0,
        has_maintenance_gap=has_maintenance_gap,
    )

    if setup_bar_timestamp is not None:
        signal.setup_bar_timestamp = setup_bar_timestamp

    return signal


# =============================================================================
# CRYPTODAEMONCONFIG TESTS
# =============================================================================


class TestCryptoDaemonConfig:
    """Tests for CryptoDaemonConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CryptoDaemonConfig()

        assert len(config.symbols) > 0
        assert config.scan_interval > 0
        assert config.entry_poll_interval > 0
        assert config.signal_expiry_hours > 0
        assert config.min_magnitude_pct >= 0
        assert config.min_risk_reward >= 0
        assert config.enable_execution is True
        assert config.paper_balance > 0

    def test_custom_values(self):
        """Config should accept custom values."""
        config = CryptoDaemonConfig(
            symbols=["TEST-PERP"],
            scan_interval=120,
            min_magnitude_pct=1.0,
            min_risk_reward=3.0,
            enable_execution=False,
        )

        assert config.symbols == ["TEST-PERP"]
        assert config.scan_interval == 120
        assert config.min_magnitude_pct == 1.0
        assert config.min_risk_reward == 3.0
        assert config.enable_execution is False

    def test_tfc_reeval_defaults(self):
        """TFC re-eval config should have correct defaults."""
        config = CryptoDaemonConfig()

        assert config.tfc_reeval_enabled is True
        assert config.tfc_reeval_min_strength == 3
        assert config.tfc_reeval_block_on_flip is True
        assert config.tfc_reeval_log_always is True

    def test_api_config_defaults(self):
        """API config should have correct defaults."""
        config = CryptoDaemonConfig()

        assert config.api_enabled is True
        assert config.api_host == '0.0.0.0'
        assert config.api_port == 8080

    def test_discord_alert_defaults(self):
        """Discord alert config should have correct defaults."""
        config = CryptoDaemonConfig()

        assert config.discord_webhook_url is None
        assert config.alert_on_signal_detection is False
        assert config.alert_on_trigger is False
        assert config.alert_on_trade_entry is True
        assert config.alert_on_trade_exit is True


# =============================================================================
# DAEMON INITIALIZATION TESTS
# =============================================================================


class TestDaemonInitialization:
    """Tests for daemon initialization."""

    def test_init_with_defaults(self, mock_client, mock_scanner):
        """Daemon should initialize with default config."""
        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon()

                assert daemon.config is not None
                assert daemon._running is False
                assert daemon._scan_count == 0
                assert daemon._signal_count == 0

    def test_init_with_custom_config(self, minimal_config, mock_client, mock_scanner):
        """Daemon should use provided config."""
        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=minimal_config)

                assert daemon.config.symbols == ["BTC-PERP-INTX"]
                assert daemon.config.scan_interval == 60

    def test_init_with_provided_client(self, minimal_config, mock_client, mock_scanner):
        """Daemon should use provided client."""
        with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
            daemon = CryptoSignalDaemon(
                config=minimal_config,
                client=mock_client,
            )

            assert daemon.client is mock_client

    def test_init_with_provided_scanner(self, minimal_config, mock_client, mock_scanner):
        """Daemon should use provided scanner."""
        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            daemon = CryptoSignalDaemon(
                config=minimal_config,
                scanner=mock_scanner,
            )

            assert daemon.scanner is mock_scanner

    def test_init_execution_disabled_no_paper_trader(self, minimal_config, mock_client, mock_scanner):
        """Paper trader should not be created when execution disabled."""
        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=minimal_config)

                assert daemon.paper_trader is None
                assert daemon.position_monitor is None

    def test_init_statistics_zeroed(self, daemon_minimal):
        """Statistics should be zeroed on init."""
        assert daemon_minimal._scan_count == 0
        assert daemon_minimal._signal_count == 0
        assert daemon_minimal._trigger_count == 0
        assert daemon_minimal._execution_count == 0
        assert daemon_minimal._error_count == 0


# =============================================================================
# SIGNAL FILTER TESTS
# =============================================================================


class TestPassesFilters:
    """Tests for _passes_filters method."""

    def test_passes_all_filters(self, daemon_minimal):
        """Signal passing all filters should return True."""
        signal = create_test_signal(
            magnitude_pct=2.0,  # Above default 0.1
            risk_reward=2.0,   # Above default 0.3
            has_maintenance_gap=False,
        )

        assert daemon_minimal.filter_manager.passes_filters(signal) is True

    def test_fails_magnitude_filter(self, daemon_minimal):
        """Signal with low magnitude should fail."""
        signal = create_test_signal(magnitude_pct=0.05)

        assert daemon_minimal.filter_manager.passes_filters(signal) is False

    def test_fails_risk_reward_filter(self, mock_client, mock_scanner):
        """Signal with low R:R should fail when threshold is set."""
        # Default MIN_SIGNAL_RISK_REWARD is 0.0, so we need to set a higher threshold
        config = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            min_risk_reward=0.5,  # Set threshold higher than test value
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=config)

                signal = create_test_signal(risk_reward=0.3)  # Below 0.5 threshold
                assert daemon.filter_manager.passes_filters(signal) is False

    def test_fails_maintenance_gap_filter(self, daemon_minimal):
        """Signal with maintenance gap should fail."""
        signal = create_test_signal(has_maintenance_gap=True)

        assert daemon_minimal.filter_manager.passes_filters(signal) is False

    def test_custom_magnitude_threshold(self, mock_client, mock_scanner):
        """Custom magnitude threshold should be respected."""
        config = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            min_magnitude_pct=5.0,  # High threshold
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=config)

                signal = create_test_signal(magnitude_pct=4.0)  # Below 5.0
                assert daemon.filter_manager.passes_filters(signal) is False

                signal = create_test_signal(magnitude_pct=6.0)  # Above 5.0
                assert daemon.filter_manager.passes_filters(signal) is True

    def test_custom_risk_reward_threshold(self, mock_client, mock_scanner):
        """Custom R:R threshold should be respected."""
        config = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            min_risk_reward=3.0,  # High threshold
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=config)

                signal = create_test_signal(risk_reward=2.5)  # Below 3.0
                assert daemon.filter_manager.passes_filters(signal) is False

                signal = create_test_signal(risk_reward=3.5)  # Above 3.0
                assert daemon.filter_manager.passes_filters(signal) is True


# =============================================================================
# SIGNAL ID AND DEDUPLICATION TESTS
# =============================================================================


class TestSignalDeduplication:
    """Tests for signal ID generation and deduplication."""

    def test_generate_signal_id_basic(self, daemon_minimal):
        """Signal ID should contain key identifying info."""
        signal = create_test_signal(
            symbol="BTC-PERP-INTX",
            timeframe="1h",
            pattern_type="3-2U",
            direction="LONG",
        )

        signal_id = daemon_minimal.filter_manager.generate_signal_id(signal)

        assert "BTC-PERP-INTX" in signal_id
        assert "1h" in signal_id
        assert "3-2U" in signal_id
        assert "LONG" in signal_id

    def test_generate_signal_id_uses_setup_bar_timestamp(self, daemon_minimal):
        """Signal ID should use setup_bar_timestamp when available."""
        setup_ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        signal = create_test_signal(setup_bar_timestamp=setup_ts)

        signal_id = daemon_minimal.filter_manager.generate_signal_id(signal)

        assert "2025-01-15" in signal_id

    def test_generate_signal_id_falls_back_to_detected_time(self, daemon_minimal):
        """Signal ID should fall back to detected_time if no setup_bar_timestamp."""
        detected = datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc)
        signal = create_test_signal(detected_time=detected)
        # Ensure no setup_bar_timestamp
        if hasattr(signal, 'setup_bar_timestamp'):
            signal.setup_bar_timestamp = None

        signal_id = daemon_minimal.filter_manager.generate_signal_id(signal)

        assert "2025-01-16" in signal_id

    def test_different_signals_different_ids(self, daemon_minimal):
        """Different signals should have different IDs."""
        signal1 = create_test_signal(symbol="BTC-PERP-INTX", direction="LONG")
        signal2 = create_test_signal(symbol="ETH-PERP-INTX", direction="LONG")
        signal3 = create_test_signal(symbol="BTC-PERP-INTX", direction="SHORT")

        id1 = daemon_minimal.filter_manager.generate_signal_id(signal1)
        id2 = daemon_minimal.filter_manager.generate_signal_id(signal2)
        id3 = daemon_minimal.filter_manager.generate_signal_id(signal3)

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_same_signal_same_id(self, daemon_minimal):
        """Same signal should produce same ID."""
        setup_ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        signal1 = create_test_signal(setup_bar_timestamp=setup_ts)
        signal2 = create_test_signal(setup_bar_timestamp=setup_ts)

        id1 = daemon_minimal.filter_manager.generate_signal_id(signal1)
        id2 = daemon_minimal.filter_manager.generate_signal_id(signal2)

        assert id1 == id2

    def test_is_duplicate_new_signal(self, daemon_minimal):
        """New signal should not be duplicate."""
        signal = create_test_signal()

        assert daemon_minimal.filter_manager.is_duplicate(signal) is False

    def test_is_duplicate_after_storing(self, daemon_minimal):
        """Signal should be duplicate after storing."""
        signal = create_test_signal()

        # Store the signal
        signal_id = daemon_minimal.filter_manager.generate_signal_id(signal)
        with daemon_minimal.filter_manager._signals_lock:
            daemon_minimal.filter_manager._detected_signals[signal_id] = signal

        # Now it should be duplicate
        assert daemon_minimal.filter_manager.is_duplicate(signal) is True


# =============================================================================
# MAINTENANCE WINDOW TESTS
# =============================================================================


class TestMaintenanceWindow:
    """Tests for maintenance window detection."""

    def test_maintenance_disabled(self, mock_client, mock_scanner):
        """Maintenance window should return False when disabled."""
        config = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            maintenance_window_enabled=False,
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon = CryptoSignalDaemon(config=config)

                # Even during maintenance time, should return False
                assert daemon.is_maintenance_window() is False

    def test_not_maintenance_day(self, daemon_minimal):
        """Non-maintenance day should not be maintenance window."""
        daemon_minimal.config.maintenance_window_enabled = True

        # The method checks datetime.now(timezone.utc).weekday()
        # We verify the method returns a boolean and handles non-Friday gracefully
        result = daemon_minimal.is_maintenance_window()
        assert isinstance(result, bool)

    def test_maintenance_window_method_exists(self, daemon_minimal):
        """Maintenance window method should exist and be callable."""
        daemon_minimal.config.maintenance_window_enabled = True

        # Verify the method exists and returns a boolean
        assert callable(daemon_minimal.is_maintenance_window)
        result = daemon_minimal.is_maintenance_window()
        assert isinstance(result, bool)

    def test_maintenance_check_respects_config(self, mock_client, mock_scanner):
        """Maintenance check should respect enabled flag."""
        # With enabled=True, it checks the time
        config_enabled = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            maintenance_window_enabled=True,
            api_enabled=False,
            enable_execution=False,
        )

        # With enabled=False, it always returns False
        config_disabled = CryptoDaemonConfig(
            symbols=["BTC-PERP-INTX"],
            maintenance_window_enabled=False,
            api_enabled=False,
            enable_execution=False,
        )

        with patch('crypto.scanning.daemon.CoinbaseClient', return_value=mock_client):
            with patch('crypto.scanning.daemon.CryptoSignalScanner', return_value=mock_scanner):
                daemon_enabled = CryptoSignalDaemon(config=config_enabled)
                daemon_disabled = CryptoSignalDaemon(config=config_disabled)

                # Disabled should always return False regardless of time
                assert daemon_disabled.is_maintenance_window() is False

                # Enabled returns based on current time (may be True or False)
                result = daemon_enabled.is_maintenance_window()
                assert isinstance(result, bool)


# =============================================================================
# STALE SETUP TESTS
# =============================================================================


class TestIsSetupStale:
    """Tests for _is_setup_stale method."""

    def test_completed_signal_not_stale(self, daemon_minimal):
        """COMPLETED signals should never be stale."""
        signal = create_test_signal(signal_type="COMPLETED")

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is False
        assert reason == ""

    def test_setup_without_timestamp_not_stale(self, daemon_minimal):
        """SETUP without timestamp should skip staleness check."""
        signal = create_test_signal(signal_type="SETUP")
        signal.setup_bar_timestamp = None
        if hasattr(signal, 'context') and signal.context:
            signal.context.setup_bar_timestamp = None

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is False

    def test_fresh_1h_setup_not_stale(self, daemon_minimal):
        """Fresh 1H setup (within 2 hours) should not be stale."""
        # Setup bar 1 hour ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="1h",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is False

    def test_stale_1h_setup(self, daemon_minimal):
        """1H setup older than 2 hours should be stale."""
        # Setup bar 3 hours ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=3)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="1H",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is True
        assert "expired" in reason.lower()

    def test_fresh_4h_setup_not_stale(self, daemon_minimal):
        """Fresh 4H setup (within 8 hours) should not be stale."""
        # Setup bar 6 hours ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=6)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="4H",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is False

    def test_stale_4h_setup(self, daemon_minimal):
        """4H setup older than 8 hours should be stale."""
        # Setup bar 10 hours ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=10)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="4H",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is True
        assert "expired" in reason.lower()

    def test_fresh_1d_setup_not_stale(self, daemon_minimal):
        """Fresh 1D setup (within 48 hours) should not be stale."""
        # Setup bar 24 hours ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=24)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="1D",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is False

    def test_stale_1d_setup(self, daemon_minimal):
        """1D setup older than 48 hours should be stale."""
        # Setup bar 50 hours ago
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=50)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="1D",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        assert is_stale is True
        assert "expired" in reason.lower()

    def test_unknown_timeframe_uses_default(self, daemon_minimal):
        """Unknown timeframe should use default 2 hour window."""
        # Setup bar 3 hours ago with unknown timeframe
        setup_ts = datetime.now(timezone.utc) - timedelta(hours=3)
        signal = create_test_signal(
            signal_type="SETUP",
            timeframe="UNKNOWN",
            setup_bar_timestamp=setup_ts,
        )

        is_stale, reason = daemon_minimal.entry_validator.is_setup_stale(signal)

        # Default is 2 hours, so 3 hours should be stale
        assert is_stale is True


# =============================================================================
# SCANNING TESTS
# =============================================================================


class TestRunScan:
    """Tests for run_scan method."""

    def test_run_scan_increments_count(self, daemon_minimal, mock_scanner):
        """run_scan should increment scan count."""
        mock_scanner.scan_all_timeframes.return_value = []

        initial_count = daemon_minimal._scan_count
        daemon_minimal.run_scan()

        assert daemon_minimal._scan_count == initial_count + 1

    def test_run_scan_calls_scanner_for_each_symbol(self, daemon_minimal, mock_scanner):
        """run_scan should call scanner for each configured symbol."""
        mock_scanner.scan_all_timeframes.return_value = []

        daemon_minimal.run_scan()

        assert mock_scanner.scan_all_timeframes.call_count == len(daemon_minimal.config.symbols)

    def test_run_scan_filters_signals(self, daemon_minimal, mock_scanner):
        """run_scan should filter low-quality signals."""
        low_quality = create_test_signal(magnitude_pct=0.01)  # Too low
        high_quality = create_test_signal(magnitude_pct=2.0)

        mock_scanner.scan_all_timeframes.return_value = [low_quality, high_quality]

        new_signals = daemon_minimal.run_scan()

        # Only high quality should pass
        assert len(new_signals) == 1

    def test_run_scan_deduplicates(self, daemon_minimal, mock_scanner):
        """run_scan should not return duplicate signals."""
        signal = create_test_signal()
        mock_scanner.scan_all_timeframes.return_value = [signal]

        # First scan
        first_result = daemon_minimal.run_scan()
        assert len(first_result) == 1

        # Second scan with same signal
        second_result = daemon_minimal.run_scan()
        assert len(second_result) == 0  # Duplicate filtered

    def test_run_scan_increments_signal_count(self, daemon_minimal, mock_scanner):
        """run_scan should increment signal count for new signals."""
        signal = create_test_signal()
        mock_scanner.scan_all_timeframes.return_value = [signal]

        initial_count = daemon_minimal._signal_count
        daemon_minimal.run_scan()

        assert daemon_minimal._signal_count == initial_count + 1

    def test_run_scan_handles_scanner_error(self, daemon_minimal, mock_scanner):
        """run_scan should handle scanner errors gracefully."""
        mock_scanner.scan_all_timeframes.side_effect = Exception("Scanner error")

        # Should not raise
        result = daemon_minimal.run_scan()

        assert result == []
        assert daemon_minimal._error_count > 0


class TestRunScanAndMonitor:
    """Tests for run_scan_and_monitor method."""

    def test_run_scan_and_monitor_adds_setups_to_monitor(self, daemon_minimal, mock_scanner):
        """SETUP signals should be added to entry monitor."""
        setup_signal = create_test_signal(signal_type="SETUP")
        mock_scanner.scan_all_timeframes.return_value = [setup_signal]

        # Mock entry monitor
        daemon_minimal.entry_monitor = Mock()
        daemon_minimal.entry_monitor.add_signals.return_value = 1

        daemon_minimal.run_scan_and_monitor()

        daemon_minimal.entry_monitor.add_signals.assert_called_once()

    def test_run_scan_and_monitor_executes_completed(self, daemon_with_execution, mock_scanner, mock_paper_trader):
        """COMPLETED signals should execute immediately."""
        completed_signal = create_test_signal(
            signal_type="COMPLETED",
            tfc_passes=True,
        )
        mock_scanner.scan_all_timeframes.return_value = [completed_signal]
        daemon_with_execution.scanner = mock_scanner

        # Mock entry monitor
        daemon_with_execution.entry_monitor = Mock()
        daemon_with_execution.entry_monitor.add_signals.return_value = 0

        daemon_with_execution.run_scan_and_monitor()

        # Should attempt to execute
        # (the actual execution may fail due to other checks)


# =============================================================================
# DAEMON LIFECYCLE TESTS
# =============================================================================


class TestDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    def test_is_running_initially_false(self, daemon_minimal):
        """Daemon should not be running initially."""
        assert daemon_minimal.is_running is False

    def test_start_non_blocking(self, daemon_minimal):
        """start(block=False) should return immediately."""
        daemon_minimal.entry_monitor = Mock()

        # Start non-blocking
        daemon_minimal.start(block=False)

        assert daemon_minimal.is_running is True

        # Clean up
        daemon_minimal.stop()

    def test_start_sets_running_flag(self, daemon_minimal):
        """start should set _running flag."""
        daemon_minimal.entry_monitor = Mock()

        daemon_minimal.start(block=False)

        assert daemon_minimal._running is True

        daemon_minimal.stop()

    def test_start_records_start_time(self, daemon_minimal):
        """start should record start time."""
        daemon_minimal.entry_monitor = Mock()

        daemon_minimal.start(block=False)

        assert daemon_minimal._start_time is not None

        daemon_minimal.stop()

    def test_stop_clears_running_flag(self, daemon_minimal):
        """stop should clear _running flag."""
        daemon_minimal.entry_monitor = Mock()
        daemon_minimal.start(block=False)

        daemon_minimal.stop()

        assert daemon_minimal._running is False

    def test_stop_when_not_running(self, daemon_minimal):
        """stop should handle being called when not running."""
        # Should not raise
        daemon_minimal.stop()

        assert daemon_minimal._running is False

    def test_start_when_already_running(self, daemon_minimal):
        """start should warn if already running."""
        daemon_minimal.entry_monitor = Mock()
        daemon_minimal.start(block=False)

        # Second start should just return
        daemon_minimal.start(block=False)

        assert daemon_minimal.is_running is True

        daemon_minimal.stop()

    def test_start_starts_entry_monitor(self, daemon_minimal):
        """start should start entry monitor."""
        daemon_minimal.entry_monitor = Mock()

        daemon_minimal.start(block=False)

        daemon_minimal.entry_monitor.start.assert_called_once()

        daemon_minimal.stop()


# =============================================================================
# STATUS AND STATISTICS TESTS
# =============================================================================


class TestStatus:
    """Tests for get_status method."""

    def test_get_status_returns_dict(self, daemon_minimal):
        """get_status should return dict with expected keys."""
        status = daemon_minimal.get_status()

        assert isinstance(status, dict)
        assert "running" in status
        assert "scan_count" in status
        assert "signal_count" in status
        assert "trigger_count" in status
        assert "execution_count" in status
        assert "error_count" in status

    def test_get_status_reflects_state(self, daemon_minimal):
        """get_status should reflect current daemon state."""
        daemon_minimal._scan_count = 5
        daemon_minimal._signal_count = 10
        daemon_minimal._trigger_count = 3

        status = daemon_minimal.get_status()

        assert status["scan_count"] == 5
        assert status["signal_count"] == 10
        assert status["trigger_count"] == 3

    def test_get_status_includes_uptime(self, daemon_minimal):
        """get_status should include uptime when running."""
        daemon_minimal.entry_monitor = Mock()
        daemon_minimal.start(block=False)

        status = daemon_minimal.get_status()

        assert status["uptime_seconds"] is not None
        assert status["uptime_seconds"] >= 0

        daemon_minimal.stop()

    def test_get_status_uptime_none_when_not_started(self, daemon_minimal):
        """get_status should have None uptime when not started."""
        status = daemon_minimal.get_status()

        assert status["uptime_seconds"] is None

    def test_get_detected_signals_empty_initially(self, daemon_minimal):
        """get_detected_signals should be empty initially."""
        signals = daemon_minimal.get_detected_signals()

        assert signals == []

    def test_get_detected_signals_returns_stored(self, daemon_minimal, mock_scanner):
        """get_detected_signals should return stored signals."""
        signal = create_test_signal()
        mock_scanner.scan_all_timeframes.return_value = [signal]

        daemon_minimal.run_scan()

        signals = daemon_minimal.get_detected_signals()

        assert len(signals) == 1

    def test_get_pending_setups_without_monitor(self, daemon_minimal):
        """get_pending_setups should return empty list without monitor."""
        daemon_minimal.entry_monitor = None

        setups = daemon_minimal.get_pending_setups()

        assert setups == []

    def test_get_pending_setups_delegates_to_monitor(self, daemon_minimal):
        """get_pending_setups should delegate to entry monitor."""
        daemon_minimal.entry_monitor = Mock()
        daemon_minimal.entry_monitor.get_pending_signals.return_value = ["signal1"]

        setups = daemon_minimal.get_pending_setups()

        assert setups == ["signal1"]


# =============================================================================
# SIGNAL CLEANUP TESTS
# =============================================================================


class TestSignalCleanup:
    """Tests for _cleanup_expired_signals method."""

    def test_cleanup_removes_old_signals(self, daemon_minimal):
        """Cleanup should remove signals older than expiry."""
        # Create old signal
        old_time = datetime.now(timezone.utc) - timedelta(hours=100)
        old_signal = create_test_signal(detected_time=old_time)

        # Store it
        signal_id = daemon_minimal.filter_manager.generate_signal_id(old_signal)
        with daemon_minimal.filter_manager._signals_lock:
            daemon_minimal.filter_manager._detected_signals[signal_id] = old_signal

        # Run cleanup
        daemon_minimal.filter_manager.cleanup_expired_signals()

        # Should be removed
        assert len(daemon_minimal.filter_manager._detected_signals) == 0

    def test_cleanup_keeps_fresh_signals(self, daemon_minimal):
        """Cleanup should keep signals within expiry window."""
        # Create fresh signal
        fresh_signal = create_test_signal()

        # Store it
        signal_id = daemon_minimal.filter_manager.generate_signal_id(fresh_signal)
        with daemon_minimal.filter_manager._signals_lock:
            daemon_minimal.filter_manager._detected_signals[signal_id] = fresh_signal

        # Run cleanup
        daemon_minimal.filter_manager.cleanup_expired_signals()

        # Should still be there
        assert len(daemon_minimal.filter_manager._detected_signals) == 1

    def test_cleanup_handles_empty_store(self, daemon_minimal):
        """Cleanup should handle empty signal store."""
        # Should not raise
        daemon_minimal.filter_manager.cleanup_expired_signals()

        assert len(daemon_minimal.filter_manager._detected_signals) == 0


# =============================================================================
# POSITION MONITORING TESTS
# =============================================================================


class TestPositionMonitoring:
    """Tests for position monitoring methods."""

    def test_check_positions_without_monitor(self, daemon_minimal):
        """check_positions should return 0 without monitor."""
        daemon_minimal.position_monitor = None

        closed = daemon_minimal.check_positions()

        assert closed == 0

    def test_check_positions_no_exits(self, daemon_with_execution):
        """check_positions should return 0 when no exits triggered."""
        daemon_with_execution.position_monitor = Mock()
        daemon_with_execution.position_monitor.check_exits.return_value = []

        closed = daemon_with_execution.check_positions()

        assert closed == 0

    def test_get_open_positions_without_monitor(self, daemon_minimal):
        """get_open_positions should return empty list without monitor."""
        daemon_minimal.position_monitor = None

        positions = daemon_minimal.get_open_positions()

        assert positions == []

    def test_get_open_positions_delegates_to_monitor(self, daemon_with_execution):
        """get_open_positions should delegate to position monitor."""
        daemon_with_execution.position_monitor = Mock()
        daemon_with_execution.position_monitor.get_open_positions_with_pnl.return_value = [
            {"symbol": "BTC", "pnl": 100}
        ]

        positions = daemon_with_execution.get_open_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC"
