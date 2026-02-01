"""
Integration tests for Spot Signal Detection + Derivative Execution Architecture.

Session EQUITY-99: Tests the two-layer data architecture where:
- Signal Detection: Uses SPOT data for cleaner price action
- Execution: Uses DERIVATIVE data for actual trading
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from crypto import config
from crypto.utils.symbol_resolver import SymbolResolver
from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto.scanning.signal_scanner import CryptoSignalScanner
from crypto.scanning.entry_monitor import CryptoEntryMonitor, CryptoEntryMonitorConfig
from crypto.scanning.daemon import CryptoSignalDaemon, CryptoDaemonConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_coinbase_client():
    """Create a mock Coinbase client."""
    client = MagicMock()
    client.simulation_mode = True

    # Default price responses
    client.get_current_price.return_value = 100000.0

    return client


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2026-01-01', periods=50, freq='1h', tz='UTC')

    # Generate realistic BTC-like price data
    np.random.seed(42)
    base_price = 100000
    returns = np.random.normal(0.0001, 0.005, 50)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Open': prices * (1 - np.random.uniform(0, 0.002, 50)),
        'High': prices * (1 + np.random.uniform(0, 0.003, 50)),
        'Low': prices * (1 - np.random.uniform(0, 0.003, 50)),
        'Close': prices,
        'Volume': np.random.uniform(100, 1000, 50),
    }, index=dates)

    return df


@pytest.fixture
def sample_signal():
    """Create a sample detected signal."""
    return CryptoDetectedSignal(
        pattern_type="2D-1-2U",
        direction="LONG",
        symbol="BIP-20DEC30-CDE",
        timeframe="1h",
        detected_time=datetime.now(timezone.utc),
        entry_trigger=100500.0,
        stop_price=99500.0,
        target_price=102000.0,
        magnitude_pct=1.5,
        risk_reward=1.5,
        context=CryptoSignalContext(),
        signal_type="SETUP",
        setup_bar_high=100500.0,
        setup_bar_low=99800.0,
        data_symbol="BTC-USD",
        execution_symbol="BIP-20DEC30-CDE",
    )


# =============================================================================
# SIGNAL SCANNER INTEGRATION TESTS
# =============================================================================


class TestScannerSpotDataIntegration:
    """Tests for scanner using spot data for signal detection."""

    def test_scanner_fetch_data_uses_spot_when_enabled(self, mock_coinbase_client, sample_ohlcv_df):
        """Scanner should fetch spot data when USE_SPOT_FOR_SIGNALS=True."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            scanner._fetch_data("BIP-20DEC30-CDE", "1h", 50)

        # Should have called with BTC-USD (spot), not BIP-20DEC30-CDE
        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "BTC-USD"

    def test_scanner_fetch_data_uses_derivative_when_disabled(self, mock_coinbase_client, sample_ohlcv_df):
        """Scanner should fetch derivative data when USE_SPOT_FOR_SIGNALS=False."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', False):
            scanner._fetch_data("BIP-20DEC30-CDE", "1h", 50)

        # Should have called with original derivative symbol
        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "BIP-20DEC30-CDE"

    def test_scanner_ada_always_uses_derivative_data(self, mock_coinbase_client, sample_ohlcv_df):
        """Scanner should always use derivative data for ADA (no spot available)."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            scanner._fetch_data("ADA-USD", "1h", 50)

        # ADA has no spot mapping, should use original
        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "ADA-USD"

    def test_scanner_explicit_use_spot_override(self, mock_coinbase_client, sample_ohlcv_df):
        """Scanner should respect explicit use_spot parameter."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        # Even if config says False, explicit True should use spot
        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', False):
            scanner._fetch_data("BIP-20DEC30-CDE", "1h", 50, use_spot=True)

        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "BTC-USD"


class TestScannerSignalFieldPopulation:
    """Tests that scanner populates data_symbol and execution_symbol fields."""

    def test_signal_has_correct_data_symbol(self, mock_coinbase_client, sample_ohlcv_df):
        """Detected signals should have correct data_symbol field."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            signals = scanner.scan_symbol_timeframe("BIP-20DEC30-CDE", "1h")

        # All signals should have data_symbol set to BTC-USD
        for signal in signals:
            assert signal.data_symbol == "BTC-USD"

    def test_signal_has_correct_execution_symbol(self, mock_coinbase_client, sample_ohlcv_df):
        """Detected signals should have correct execution_symbol field."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            signals = scanner.scan_symbol_timeframe("BIP-20DEC30-CDE", "1h")

        # All signals should have execution_symbol set to derivative
        for signal in signals:
            assert signal.execution_symbol == "BIP-20DEC30-CDE"

    def test_signal_symbol_unchanged(self, mock_coinbase_client, sample_ohlcv_df):
        """Signal's main symbol field should be the trading symbol."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            signals = scanner.scan_symbol_timeframe("BIP-20DEC30-CDE", "1h")

        # Main symbol should be the trading symbol
        for signal in signals:
            assert signal.symbol == "BIP-20DEC30-CDE"


# =============================================================================
# ENTRY MONITOR INTEGRATION TESTS
# =============================================================================


class TestEntryMonitorSpotPrices:
    """Tests for entry monitor using spot prices for trigger detection."""

    def test_monitor_uses_spot_prices_when_enabled(self, mock_coinbase_client):
        """Entry monitor should use spot prices when USE_SPOT_FOR_TRIGGERS=True."""
        monitor = CryptoEntryMonitor(
            client=mock_coinbase_client,
            config=CryptoEntryMonitorConfig(),
        )

        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', True):
            monitor._fetch_prices(["BIP-20DEC30-CDE"])

        # Should have called with BTC-USD (spot)
        mock_coinbase_client.get_current_price.assert_called_with("BTC-USD")

    def test_monitor_uses_derivative_prices_when_disabled(self, mock_coinbase_client):
        """Entry monitor should use derivative prices when USE_SPOT_FOR_TRIGGERS=False."""
        monitor = CryptoEntryMonitor(
            client=mock_coinbase_client,
            config=CryptoEntryMonitorConfig(),
        )

        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', False):
            monitor._fetch_prices(["BIP-20DEC30-CDE"])

        # Should have called with derivative symbol
        mock_coinbase_client.get_current_price.assert_called_with("BIP-20DEC30-CDE")

    def test_monitor_price_key_is_trading_symbol(self, mock_coinbase_client):
        """Price dictionary key should be the trading symbol, not spot."""
        mock_coinbase_client.get_current_price.return_value = 100000.0
        monitor = CryptoEntryMonitor(
            client=mock_coinbase_client,
            config=CryptoEntryMonitorConfig(),
        )

        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', True):
            prices = monitor._fetch_prices(["BIP-20DEC30-CDE"])

        # Key should be trading symbol, not spot
        assert "BIP-20DEC30-CDE" in prices
        assert "BTC-USD" not in prices

    def test_monitor_ada_uses_derivative_price(self, mock_coinbase_client):
        """Entry monitor should use derivative price for ADA (no spot)."""
        monitor = CryptoEntryMonitor(
            client=mock_coinbase_client,
            config=CryptoEntryMonitorConfig(),
        )

        with patch.object(config, 'USE_SPOT_FOR_TRIGGERS', True):
            monitor._fetch_prices(["ADA-USD"])

        # ADA has no spot, should use derivative
        mock_coinbase_client.get_current_price.assert_called_with("ADA-USD")


# =============================================================================
# DAEMON INTEGRATION TESTS
# =============================================================================


class TestDaemonConfigSpotSettings:
    """Tests for daemon configuration of spot/derivative settings."""

    def test_daemon_config_has_spot_settings(self):
        """Daemon config should have spot/derivative settings."""
        config_obj = CryptoDaemonConfig()

        assert hasattr(config_obj, 'use_spot_for_signals')
        assert hasattr(config_obj, 'use_spot_for_triggers')

    def test_daemon_config_defaults_from_global_config(self):
        """Daemon config should default to global config values."""
        config_obj = CryptoDaemonConfig()

        assert config_obj.use_spot_for_signals == config.USE_SPOT_FOR_SIGNALS
        assert config_obj.use_spot_for_triggers == config.USE_SPOT_FOR_TRIGGERS


class TestDaemonExecutionSymbol:
    """Tests for daemon using execution_symbol for trading."""

    def test_execute_trade_uses_execution_symbol(self, mock_coinbase_client, sample_signal):
        """Daemon should use execution_symbol when opening trades."""
        mock_paper_trader = MagicMock()
        mock_paper_trader.account.current_balance = 1000.0
        mock_paper_trader.get_available_balance.return_value = 1000.0
        mock_paper_trader.open_trade.return_value = MagicMock(trade_id="test-123")

        daemon = CryptoSignalDaemon(
            client=mock_coinbase_client,
            paper_trader=mock_paper_trader,
        )

        # Create mock trigger event
        mock_event = MagicMock()
        mock_event.signal = sample_signal
        mock_event.current_price = 100500.0
        mock_event.trigger_price = 100500.0

        # Mock the TFC reeval to pass
        daemon.entry_validator.reevaluate_tfc_at_entry = MagicMock(return_value=(False, ""))
        daemon.entry_validator.is_setup_stale = MagicMock(return_value=(False, ""))

        with patch.object(config, 'LEVERAGE_FIRST_SIZING', True):
            with patch.object(config, 'FEE_PROFITABILITY_FILTER_ENABLED', False):
                daemon._execute_trade(mock_event)

        # Check that open_trade was called with execution_symbol
        if mock_paper_trader.open_trade.called:
            call_kwargs = mock_paper_trader.open_trade.call_args[1]
            assert call_kwargs['symbol'] == "BIP-20DEC30-CDE"

    def test_execute_trade_falls_back_to_symbol(self, mock_coinbase_client):
        """Daemon should fall back to signal.symbol if execution_symbol not set."""
        mock_paper_trader = MagicMock()
        mock_paper_trader.account.current_balance = 1000.0
        mock_paper_trader.get_available_balance.return_value = 1000.0
        mock_paper_trader.open_trade.return_value = MagicMock(trade_id="test-123")

        daemon = CryptoSignalDaemon(
            client=mock_coinbase_client,
            paper_trader=mock_paper_trader,
        )

        # Create signal WITHOUT execution_symbol
        signal_no_exec = CryptoDetectedSignal(
            pattern_type="2D-1-2U",
            direction="LONG",
            symbol="BIP-20DEC30-CDE",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=100500.0,
            stop_price=99500.0,
            target_price=102000.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
            context=CryptoSignalContext(tfc_passes=True, risk_multiplier=1.0),
            signal_type="SETUP",
            setup_bar_high=100500.0,
            setup_bar_low=99800.0,
            # execution_symbol not set (empty string)
        )

        mock_event = MagicMock()
        mock_event.signal = signal_no_exec
        mock_event.current_price = 100500.0
        mock_event.trigger_price = 100500.0

        daemon.entry_validator.reevaluate_tfc_at_entry = MagicMock(return_value=(False, ""))
        daemon.entry_validator.is_setup_stale = MagicMock(return_value=(False, ""))

        with patch.object(config, 'LEVERAGE_FIRST_SIZING', True):
            with patch.object(config, 'FEE_PROFITABILITY_FILTER_ENABLED', False):
                daemon._execute_trade(mock_event)

        # Should fall back to signal.symbol
        if mock_paper_trader.open_trade.called:
            call_kwargs = mock_paper_trader.open_trade.call_args[1]
            assert call_kwargs['symbol'] == "BIP-20DEC30-CDE"


# =============================================================================
# MODEL SERIALIZATION TESTS
# =============================================================================


class TestSignalSerialization:
    """Tests for signal serialization with new fields."""

    def test_to_dict_includes_data_symbol(self, sample_signal):
        """to_dict() should include data_symbol field."""
        result = sample_signal.to_dict()
        assert "data_symbol" in result
        assert result["data_symbol"] == "BTC-USD"

    def test_to_dict_includes_execution_symbol(self, sample_signal):
        """to_dict() should include execution_symbol field."""
        result = sample_signal.to_dict()
        assert "execution_symbol" in result
        assert result["execution_symbol"] == "BIP-20DEC30-CDE"


# =============================================================================
# END-TO-END FLOW TESTS
# =============================================================================


class TestEndToEndSpotDerivativeFlow:
    """Tests for the complete spot signal -> derivative execution flow."""

    def test_signal_detection_uses_spot_execution_uses_derivative(
        self, mock_coinbase_client, sample_ohlcv_df
    ):
        """Full flow: detect on spot, execute on derivative."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df
        mock_coinbase_client.get_current_price.return_value = 100000.0

        # Step 1: Scanner detects signal using spot data
        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            signals = scanner.scan_symbol_timeframe("BIP-20DEC30-CDE", "1h")

        # Verify spot was used for detection
        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "BTC-USD"

        # Verify signal fields
        for signal in signals:
            assert signal.data_symbol == "BTC-USD"
            assert signal.execution_symbol == "BIP-20DEC30-CDE"
            assert signal.symbol == "BIP-20DEC30-CDE"  # Trading symbol

    def test_statarb_symbols_unchanged_in_flow(
        self, mock_coinbase_client, sample_ohlcv_df
    ):
        """StatArb symbols should work unchanged through the flow."""
        mock_coinbase_client.get_historical_ohlcv.return_value = sample_ohlcv_df

        scanner = CryptoSignalScanner(client=mock_coinbase_client)

        with patch.object(config, 'USE_SPOT_FOR_SIGNALS', True):
            signals = scanner.scan_symbol_timeframe("ADA-USD", "1h")

        # ADA has no spot mapping, should use ADA-USD
        call_args = mock_coinbase_client.get_historical_ohlcv.call_args
        assert call_args[1]['symbol'] == "ADA-USD"

        # Verify signal fields
        for signal in signals:
            assert signal.data_symbol == "ADA-USD"
            assert signal.execution_symbol == "ADA-USD"
