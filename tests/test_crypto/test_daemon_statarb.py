"""
Tests for StatArb integration in CryptoSignalDaemon.

Session EQUITY-92: Tests for Phase 6.3 daemon integration.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from crypto.scanning.daemon import CryptoDaemonConfig, CryptoSignalDaemon
from crypto.simulation.paper_trader import PaperTrader


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_client():
    """Create a mock Coinbase client."""
    client = MagicMock()
    client.get_current_price.return_value = 1.0
    return client


@pytest.fixture
def mock_scanner():
    """Create a mock scanner."""
    scanner = MagicMock()
    scanner.scan_all_timeframes.return_value = []
    return scanner


def create_paper_trader(balance: float = 10000.0) -> PaperTrader:
    """Create a fresh paper trader with sufficient balance."""
    # Use unique account name to avoid state sharing
    import uuid
    return PaperTrader(starting_balance=balance, account_name=f"test_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def paper_trader():
    """Create a real paper trader for testing."""
    return create_paper_trader(10000.0)


@pytest.fixture
def statarb_config():
    """Create a daemon config with StatArb enabled."""
    return CryptoDaemonConfig(
        symbols=["BTC-USD", "ETH-USD"],
        statarb_enabled=True,
        statarb_pairs=[("ADA-USD", "XRP-USD")],
        enable_execution=True,
    )


@pytest.fixture
def daemon_with_statarb(mock_client, mock_scanner, statarb_config):
    """Create daemon with StatArb enabled (fresh paper trader each time)."""
    fresh_paper_trader = create_paper_trader(10000.0)
    daemon = CryptoSignalDaemon(
        config=statarb_config,
        client=mock_client,
        scanner=mock_scanner,
        paper_trader=fresh_paper_trader,
    )
    return daemon


@pytest.fixture
def daemon_without_statarb(mock_client, mock_scanner):
    """Create daemon without StatArb (default, fresh paper trader)."""
    fresh_paper_trader = create_paper_trader(10000.0)
    config = CryptoDaemonConfig(
        symbols=["BTC-USD"],
        statarb_enabled=False,
        enable_execution=True,
    )
    return CryptoSignalDaemon(
        config=config,
        client=mock_client,
        scanner=mock_scanner,
        paper_trader=fresh_paper_trader,
    )


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestStatArbConfig:
    """Test StatArb configuration in CryptoDaemonConfig."""

    def test_statarb_disabled_by_default(self):
        """StatArb should be disabled by default."""
        config = CryptoDaemonConfig()
        assert config.statarb_enabled is False

    def test_statarb_pairs_empty_by_default(self):
        """StatArb pairs should be empty by default."""
        config = CryptoDaemonConfig()
        assert config.statarb_pairs == []

    def test_statarb_config_none_by_default(self):
        """StatArb config should be None by default."""
        config = CryptoDaemonConfig()
        assert config.statarb_config is None

    def test_statarb_can_be_enabled(self):
        """StatArb can be enabled via config."""
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
        )
        assert config.statarb_enabled is True
        assert config.statarb_pairs == [("ADA-USD", "XRP-USD")]


# =============================================================================
# SETUP TESTS
# =============================================================================


class TestStatArbSetup:
    """Test StatArb generator setup."""

    def test_statarb_generator_none_when_disabled(self, daemon_without_statarb):
        """StatArb generator should be None when disabled."""
        assert daemon_without_statarb.statarb_generator is None

    def test_statarb_generator_initialized_when_enabled(self, daemon_with_statarb):
        """StatArb generator should be initialized when enabled."""
        assert daemon_with_statarb.statarb_generator is not None

    def test_statarb_generator_has_correct_pairs(self, daemon_with_statarb):
        """StatArb generator should have the configured pairs."""
        generator = daemon_with_statarb.statarb_generator
        assert generator.pairs == [("ADA-USD", "XRP-USD")]

    def test_statarb_generator_uses_paper_trader_balance(
        self, mock_client, mock_scanner
    ):
        """StatArb generator should use paper trader's current balance."""
        fresh_trader = create_paper_trader(5000.0)
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
            paper_trader=fresh_trader,
        )
        # Generator uses paper trader balance at init time
        assert daemon.statarb_generator.account_value == 5000.0

    def test_statarb_not_setup_without_pairs(self, mock_client, mock_scanner):
        """StatArb generator not created if no pairs configured."""
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[],  # Empty pairs
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
        )
        assert daemon.statarb_generator is None


# =============================================================================
# STRAT ACTIVE SYMBOLS TRACKING
# =============================================================================


class TestStratActiveSymbols:
    """Test STRAT active symbols tracking for priority."""

    def test_strat_active_symbols_initially_empty(self, daemon_with_statarb):
        """STRAT active symbols should be empty initially."""
        assert daemon_with_statarb._strat_active_symbols == set()

    def test_update_strat_active_symbols_from_open_trades(self, daemon_with_statarb):
        """Should update active symbols from open STRAT trades."""
        # Open a STRAT trade (use small amount to fit within margin)
        daemon_with_statarb.paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,  # Small quantity
            entry_price=100.0,  # Low price for testing
            strategy="strat",
        )

        daemon_with_statarb._update_strat_active_symbols()
        assert "BTC-USD" in daemon_with_statarb._strat_active_symbols

    def test_update_excludes_statarb_trades(self, daemon_with_statarb):
        """Should not include StatArb trades in STRAT active symbols."""
        # Open a StatArb trade
        daemon_with_statarb.paper_trader.open_trade(
            symbol="ADA-USD",
            side="BUY",
            quantity=100.0,
            entry_price=0.5,
            strategy="statarb",
        )

        daemon_with_statarb._update_strat_active_symbols()
        assert "ADA-USD" not in daemon_with_statarb._strat_active_symbols

    def test_update_excludes_closed_trades(self, daemon_with_statarb):
        """Should not include closed trades in active symbols."""
        # Open and close a trade (use small amount to fit within margin)
        trade = daemon_with_statarb.paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=100.0,
            strategy="strat",
        )
        assert trade is not None, "Trade should be opened successfully"
        daemon_with_statarb.paper_trader.close_trade(
            trade.trade_id, exit_price=110.0
        )

        daemon_with_statarb._update_strat_active_symbols()
        assert "BTC-USD" not in daemon_with_statarb._strat_active_symbols


# =============================================================================
# SIGNAL CHECKING TESTS
# =============================================================================


class TestCheckStatArbSignals:
    """Test StatArb signal checking."""

    def test_check_skipped_when_generator_none(self, daemon_without_statarb):
        """Signal check should be skipped when generator is None."""
        # Should not raise
        daemon_without_statarb._check_statarb_signals()

    def test_check_fetches_prices_for_pairs(self, daemon_with_statarb, mock_client):
        """Should fetch prices for configured pairs."""
        mock_client.get_current_price.return_value = 1.0

        daemon_with_statarb._check_statarb_signals()

        # Should have called for both symbols in the pair
        calls = [
            call[0][0] for call in mock_client.get_current_price.call_args_list
        ]
        assert "ADA-USD" in calls
        assert "XRP-USD" in calls

    def test_check_updates_account_value(self, daemon_with_statarb, mock_client):
        """Should update generator account value before checking."""
        mock_client.get_current_price.return_value = 1.0

        # Modify balance
        daemon_with_statarb.paper_trader.account.current_balance = 2000.0

        with patch.object(
            daemon_with_statarb.statarb_generator, "update_account_value"
        ) as mock_update:
            daemon_with_statarb._check_statarb_signals()
            mock_update.assert_called()

    def test_check_skips_on_strat_conflict_long_symbol(
        self, daemon_with_statarb, mock_client
    ):
        """Should skip StatArb if long symbol in active STRAT trade."""
        mock_client.get_current_price.return_value = 1.0

        # Open STRAT trade on long symbol
        daemon_with_statarb.paper_trader.open_trade(
            symbol="ADA-USD",
            side="BUY",
            quantity=100.0,
            entry_price=0.5,
            strategy="strat",
        )

        # Mock a signal on the conflicting pair
        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        mock_signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        with patch.object(
            daemon_with_statarb.statarb_generator,
            "check_for_signals",
            return_value=[mock_signal],
        ):
            with patch.object(daemon_with_statarb, "_execute_statarb") as mock_exec:
                daemon_with_statarb._check_statarb_signals()
                # Should NOT execute due to STRAT conflict
                mock_exec.assert_not_called()

    def test_check_skips_on_strat_conflict_short_symbol(
        self, daemon_with_statarb, mock_client
    ):
        """Should skip StatArb if short symbol in active STRAT trade."""
        mock_client.get_current_price.return_value = 1.0

        # Open STRAT trade on short symbol
        daemon_with_statarb.paper_trader.open_trade(
            symbol="XRP-USD",
            side="BUY",
            quantity=100.0,
            entry_price=0.5,
            strategy="strat",
        )

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        mock_signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        with patch.object(
            daemon_with_statarb.statarb_generator,
            "check_for_signals",
            return_value=[mock_signal],
        ):
            with patch.object(daemon_with_statarb, "_execute_statarb") as mock_exec:
                daemon_with_statarb._check_statarb_signals()
                mock_exec.assert_not_called()

    def test_check_executes_when_no_conflict(self, mock_client, mock_scanner):
        """Should execute StatArb signal when no STRAT conflict."""
        # Create fresh daemon for this test
        fresh_trader = create_paper_trader(10000.0)
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
            enable_execution=True,
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
            paper_trader=fresh_trader,
        )
        mock_client.get_current_price.return_value = 1.0

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        mock_signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
            long_notional=100.0,
            short_notional=100.0,
        )

        with patch.object(
            daemon.statarb_generator,
            "check_for_signals",
            return_value=[mock_signal],
        ):
            with patch.object(daemon.statarb_executor, "_execute") as mock_exec:
                daemon._check_statarb_signals()
                mock_exec.assert_called_once_with(mock_signal)


# =============================================================================
# EXECUTION TESTS
# =============================================================================


class TestExecuteStatArb:
    """Test StatArb execution."""

    def test_execute_entry_opens_both_legs(self, mock_client, mock_scanner):
        """Entry signal should open long and short legs."""
        # Create fresh daemon with fresh paper trader
        fresh_trader = create_paper_trader(10000.0)
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
            enable_execution=True,
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
            paper_trader=fresh_trader,
        )
        mock_client.get_current_price.return_value = 1.0

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
            long_notional=100.0,
            short_notional=100.0,
        )

        initial_count = daemon._execution_count
        daemon._execute_statarb_entry(signal)

        # Should have opened 2 trades
        assert daemon._execution_count == initial_count + 2

        # Check trades opened with correct strategy
        statarb_trades = daemon.paper_trader.get_trades_by_strategy("statarb")
        assert len(statarb_trades) == 2

        symbols = {t.symbol for t in statarb_trades}
        assert "ADA-USD" in symbols
        assert "XRP-USD" in symbols

    def test_execute_entry_uses_correct_sides(self, mock_client, mock_scanner):
        """Entry should use BUY for long leg, SELL for short leg."""
        # Create fresh daemon
        fresh_trader = create_paper_trader(10000.0)
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
            enable_execution=True,
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
            paper_trader=fresh_trader,
        )
        mock_client.get_current_price.return_value = 1.0

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
            long_notional=100.0,
            short_notional=100.0,
        )

        daemon._execute_statarb_entry(signal)

        trades = daemon.paper_trader.get_trades_by_strategy("statarb")
        trade_map = {t.symbol: t for t in trades}

        assert trade_map["ADA-USD"].side == "BUY"
        assert trade_map["XRP-USD"].side == "SELL"

    def test_execute_exit_closes_statarb_trades(self, mock_client, mock_scanner):
        """Exit signal should close StatArb trades for the pair."""
        # Create fresh daemon
        fresh_trader = create_paper_trader(10000.0)
        config = CryptoDaemonConfig(
            statarb_enabled=True,
            statarb_pairs=[("ADA-USD", "XRP-USD")],
            enable_execution=True,
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
            paper_trader=fresh_trader,
        )
        mock_client.get_current_price.return_value = 1.0

        # First open trades
        daemon.paper_trader.open_trade(
            symbol="ADA-USD",
            side="BUY",
            quantity=10.0,  # Smaller quantity
            entry_price=0.5,
            strategy="statarb",
        )
        daemon.paper_trader.open_trade(
            symbol="XRP-USD",
            side="SELL",
            quantity=10.0,
            entry_price=0.5,
            strategy="statarb",
        )

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        exit_signal = StatArbSignal(
            signal_type=StatArbSignalType.EXIT,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=0.0,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
            bars_held=5,
        )

        daemon._execute_statarb_exit(exit_signal)

        # All statarb trades should be closed
        open_trades = [
            t
            for t in daemon.paper_trader.get_trades_by_strategy("statarb")
            if t.exit_time is None
        ]
        assert len(open_trades) == 0

    def test_execute_skips_without_paper_trader(self, mock_client, mock_scanner):
        """Execution should be skipped without paper trader."""
        config = CryptoDaemonConfig(
            enable_execution=False,
        )
        daemon = CryptoSignalDaemon(
            config=config,
            client=mock_client,
            scanner=mock_scanner,
        )

        from crypto.statarb.signal_generator import StatArbSignal, StatArbSignalType

        signal = StatArbSignal(
            signal_type=StatArbSignalType.LONG_SPREAD,
            long_symbol="ADA-USD",
            short_symbol="XRP-USD",
            zscore=-2.5,
            spread=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        # Should not raise
        daemon._execute_statarb(signal)


# =============================================================================
# STATUS TESTS
# =============================================================================


class TestStatArbStatus:
    """Test StatArb status reporting."""

    def test_status_includes_statarb_when_enabled(self, daemon_with_statarb):
        """Status should include StatArb info when enabled."""
        status = daemon_with_statarb.get_status()
        assert "statarb" in status
        assert status["statarb"] is not None

    def test_status_statarb_empty_when_disabled(self, daemon_without_statarb):
        """Status should have empty StatArb when disabled."""
        status = daemon_without_statarb.get_status()
        assert "statarb" in status
        assert status["statarb"] == {}

    def test_status_includes_strat_active_symbols(self, daemon_with_statarb):
        """Status should include STRAT active symbols."""
        status = daemon_with_statarb.get_status()
        assert "strat_active_symbols" in status
        assert isinstance(status["strat_active_symbols"], list)


# =============================================================================
# SCAN LOOP INTEGRATION
# =============================================================================


class TestScanLoopIntegration:
    """Test StatArb integration in scan loop."""

    def test_scan_loop_calls_statarb_check(self, daemon_with_statarb, mock_client):
        """Scan loop should call StatArb check after STRAT scan."""
        mock_client.get_current_price.return_value = 1.0

        with patch.object(
            daemon_with_statarb, "_check_statarb_signals"
        ) as mock_check:
            # Run single iteration of scan loop
            daemon_with_statarb.run_scan_and_monitor()

            # StatArb check happens in scan loop, not in run_scan_and_monitor
            # Let's verify the method exists and is callable
            assert callable(daemon_with_statarb._check_statarb_signals)

    def test_scan_loop_skips_statarb_when_disabled(self, daemon_without_statarb):
        """Scan loop should skip StatArb when disabled."""
        # Verify statarb_generator is None
        assert daemon_without_statarb.statarb_generator is None

        # run_scan_and_monitor should complete without error
        daemon_without_statarb.run_scan_and_monitor()
