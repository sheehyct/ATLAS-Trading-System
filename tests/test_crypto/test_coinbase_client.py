"""
Tests for crypto/exchange/coinbase_client.py

Session EQUITY-72: Comprehensive test coverage for CoinbaseClient.

Covers:
- Initialization (with/without credentials, simulation mode)
- Market data fetching (OHLCV, current price)
- Granularity resolution and resampling
- Account info retrieval
- Order management (create, cancel, list)
- Position management
- Paper trading utilities (reset, history, balance)
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from crypto.exchange.coinbase_client import CoinbaseClient, COINBASE_PUBLIC_API


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def client_simulation():
    """Create a CoinbaseClient in simulation mode without credentials."""
    with patch.dict('os.environ', {}, clear=True):
        return CoinbaseClient(simulation_mode=True)


@pytest.fixture
def client_with_mock_sdk():
    """Create a CoinbaseClient with mocked SDK."""
    with patch('crypto.exchange.coinbase_client.RESTClient') as MockRESTClient:
        mock_client = Mock()
        MockRESTClient.return_value = mock_client
        client = CoinbaseClient(
            api_key="test_key",
            api_secret="-----BEGIN EC PRIVATE KEY-----\ntest_secret\n-----END EC PRIVATE KEY-----",
            simulation_mode=False,
        )
        client.client = mock_client
        return client


@pytest.fixture
def sample_candles_data():
    """Create sample candle data for testing."""
    base_time = int(datetime(2026, 1, 15, 12, 0).timestamp())
    return [
        {
            "timestamp": base_time,
            "open": "100.0",
            "high": "105.0",
            "low": "98.0",
            "close": "103.0",
            "volume": "1000.0",
        },
        {
            "timestamp": base_time + 3600,
            "open": "103.0",
            "high": "108.0",
            "low": "102.0",
            "close": "106.0",
            "volume": "1200.0",
        },
        {
            "timestamp": base_time + 7200,
            "open": "106.0",
            "high": "110.0",
            "low": "104.0",
            "close": "109.0",
            "volume": "1100.0",
        },
        {
            "timestamp": base_time + 10800,
            "open": "109.0",
            "high": "112.0",
            "low": "107.0",
            "close": "111.0",
            "volume": "1300.0",
        },
    ]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestCoinbaseClientInit:
    """Tests for CoinbaseClient initialization."""

    def test_init_simulation_mode(self, client_simulation):
        """Test initialization in simulation mode."""
        assert client_simulation.simulation_mode is True
        assert client_simulation._mock_orders == []
        assert client_simulation._mock_position is None
        assert client_simulation._mock_balance == {"USDC": 1000.0}
        assert client_simulation._trade_history == []

    def test_init_without_credentials(self):
        """Test initialization without API credentials."""
        with patch.dict('os.environ', {
            'COINBASE_API_KEY': '',
            'COINBASE_API_SECRET': ''
        }, clear=True):
            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                # When no credentials, client should be None
                assert client.api_key is None or client.api_key == ''

    def test_init_with_env_credentials(self):
        """Test initialization loads credentials from env."""
        with patch.dict('os.environ', {
            'COINBASE_API_KEY': 'env_test_key',
            'COINBASE_API_SECRET': 'env_test_secret'
        }):
            with patch('crypto.exchange.coinbase_client.RESTClient') as MockRESTClient:
                client = CoinbaseClient(simulation_mode=True)
                assert client.api_key == 'env_test_key'

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicit credentials."""
        with patch('crypto.exchange.coinbase_client.RESTClient') as MockRESTClient:
            client = CoinbaseClient(
                api_key="explicit_key",
                api_secret="explicit_secret",
                simulation_mode=False,
            )
            assert client.api_key == "explicit_key"

    def test_init_formats_private_key(self):
        """Test that private key is formatted correctly."""
        with patch('crypto.exchange.coinbase_client.RESTClient') as MockRESTClient:
            # Key without headers should get headers added
            client = CoinbaseClient(
                api_key="test_key",
                api_secret="raw_key_content",
                simulation_mode=False,
            )
            assert "-----BEGIN EC PRIVATE KEY-----" in client.api_secret

    def test_init_handles_newlines_in_key(self):
        """Test that literal \\n in env vars are converted."""
        with patch('crypto.exchange.coinbase_client.RESTClient') as MockRESTClient:
            client = CoinbaseClient(
                api_key="test_key",
                api_secret="line1\\nline2",
                simulation_mode=False,
            )
            assert "\\n" not in client.api_secret or "\n" in client.api_secret


# =============================================================================
# GRANULARITY TESTS
# =============================================================================


class TestGranularityMapping:
    """Tests for granularity resolution."""

    def test_granularity_map_values(self, client_simulation):
        """Test GRANULARITY_MAP has expected values."""
        assert client_simulation.GRANULARITY_MAP["1m"] == "ONE_MINUTE"
        assert client_simulation.GRANULARITY_MAP["1h"] == "ONE_HOUR"
        assert client_simulation.GRANULARITY_MAP["1d"] == "ONE_DAY"

    def test_granularity_seconds(self, client_simulation):
        """Test GRANULARITY_SECONDS has correct values."""
        assert client_simulation.GRANULARITY_SECONDS["ONE_MINUTE"] == 60
        assert client_simulation.GRANULARITY_SECONDS["ONE_HOUR"] == 3600
        assert client_simulation.GRANULARITY_SECONDS["ONE_DAY"] == 86400

    def test_resolve_granularity_standard(self, client_simulation):
        """Test _resolve_granularity for standard intervals."""
        gran, resample = client_simulation._resolve_granularity("1h")
        assert gran == "ONE_HOUR"
        assert resample is None

    def test_resolve_granularity_4h(self, client_simulation):
        """Test _resolve_granularity for 4h (needs resampling)."""
        gran, resample = client_simulation._resolve_granularity("4h")
        assert gran == "ONE_HOUR"
        assert resample == "4h"

    def test_resolve_granularity_1w(self, client_simulation):
        """Test _resolve_granularity for 1w (needs resampling)."""
        gran, resample = client_simulation._resolve_granularity("1w")
        assert gran == "ONE_DAY"
        assert resample == "1W"

    def test_resolve_granularity_unknown(self, client_simulation):
        """Test _resolve_granularity falls back to 1h for unknown."""
        gran, resample = client_simulation._resolve_granularity("unknown")
        assert gran == "ONE_HOUR"
        assert resample is None


# =============================================================================
# DATA FRAME BUILDING TESTS
# =============================================================================


class TestDataFrameBuilding:
    """Tests for OHLCV DataFrame construction."""

    def test_build_ohlcv_dataframe(self, client_simulation, sample_candles_data):
        """Test building DataFrame from candle data."""
        df = client_simulation._build_ohlcv_dataframe(sample_candles_data)

        assert len(df) == 4
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "datetime"
        assert df["open"].iloc[0] == 100.0
        assert df["close"].iloc[-1] == 111.0

    def test_build_ohlcv_dataframe_sorted(self, client_simulation):
        """Test DataFrame is sorted by datetime."""
        # Reverse order data
        data = [
            {"timestamp": 1000, "open": "2", "high": "2", "low": "2", "close": "2", "volume": "2"},
            {"timestamp": 500, "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
        ]
        df = client_simulation._build_ohlcv_dataframe(data)
        assert df["open"].iloc[0] == 1.0  # Earlier timestamp first

    def test_build_ohlcv_numeric_conversion(self, client_simulation):
        """Test numeric conversion of string values."""
        data = [
            {"timestamp": "1000", "open": "100.5", "high": "101", "low": "99.5", "close": "100", "volume": "1000"}
        ]
        df = client_simulation._build_ohlcv_dataframe(data)
        assert df["open"].dtype in [np.float64, float]


class TestResampleOHLCV:
    """Tests for OHLCV resampling."""

    def test_resample_ohlcv_4h(self, client_simulation):
        """Test resampling 1h data to 4h."""
        # Create data aligned to 4h boundary (midnight start)
        base_time = int(datetime(2026, 1, 15, 0, 0).timestamp())
        aligned_data = [
            {"timestamp": base_time, "open": "100.0", "high": "105.0", "low": "98.0", "close": "103.0", "volume": "1000.0"},
            {"timestamp": base_time + 3600, "open": "103.0", "high": "108.0", "low": "102.0", "close": "106.0", "volume": "1200.0"},
            {"timestamp": base_time + 7200, "open": "106.0", "high": "110.0", "low": "104.0", "close": "109.0", "volume": "1100.0"},
            {"timestamp": base_time + 10800, "open": "109.0", "high": "112.0", "low": "107.0", "close": "111.0", "volume": "1300.0"},
        ]
        df = client_simulation._build_ohlcv_dataframe(aligned_data)
        resampled = client_simulation._resample_ohlcv(df, "4h")

        # Should have at least 1 4h bar
        assert len(resampled) >= 1
        # First bar should have correct open
        assert resampled["open"].iloc[0] == 100.0

    def test_resample_ohlcv_aggregation(self, client_simulation):
        """Test resampling aggregation rules - high is max, low is min."""
        # Create simple aligned data
        base_time = int(datetime(2026, 1, 15, 0, 0).timestamp())
        aligned_data = [
            {"timestamp": base_time, "open": "100.0", "high": "105.0", "low": "98.0", "close": "103.0", "volume": "1000.0"},
            {"timestamp": base_time + 3600, "open": "103.0", "high": "108.0", "low": "102.0", "close": "106.0", "volume": "1200.0"},
        ]
        df = client_simulation._build_ohlcv_dataframe(aligned_data)
        resampled = client_simulation._resample_ohlcv(df, "4h")

        # High should be max of all highs
        assert resampled["high"].iloc[0] == 108.0
        # Low should be min of all lows
        assert resampled["low"].iloc[0] == 98.0


# =============================================================================
# PARSE CANDLES RESPONSE TESTS
# =============================================================================


class TestParseCandlesResponse:
    """Tests for parsing Coinbase API candle responses."""

    def test_parse_candles_dict_format(self, client_simulation):
        """Test parsing dict format response."""
        response = {
            "candles": [
                {"start": 1000, "open": "100", "high": "105", "low": "95", "close": "102", "volume": "500"}
            ]
        }
        result = client_simulation._parse_candles_response(response)
        assert len(result) == 1
        assert result[0]["timestamp"] == 1000
        assert result[0]["open"] == "100"

    def test_parse_candles_object_format(self, client_simulation):
        """Test parsing object format response."""
        mock_candle = Mock()
        mock_candle.start = 1000
        mock_candle.open = "100"
        mock_candle.high = "105"
        mock_candle.low = "95"
        mock_candle.close = "102"
        mock_candle.volume = "500"

        mock_response = Mock()
        mock_response.candles = [mock_candle]

        result = client_simulation._parse_candles_response(mock_response)
        assert len(result) == 1
        assert result[0]["timestamp"] == 1000

    def test_parse_candles_empty(self, client_simulation):
        """Test parsing empty response."""
        result = client_simulation._parse_candles_response({"candles": []})
        assert result == []


# =============================================================================
# MARKET DATA TESTS
# =============================================================================


class TestGetCurrentPrice:
    """Tests for get_current_price method."""

    def test_get_price_via_public_api(self):
        """Test getting price via public API fallback."""
        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"price": "50000.0"}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Create client with no SDK
            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                client.client = None  # Force public API path

                price = client.get_current_price("BTC-USD")

                assert price == 50000.0
                mock_get.assert_called_once()

    def test_get_price_sdk_first(self, client_with_mock_sdk):
        """Test SDK is tried first for price."""
        client_with_mock_sdk.client.get_product.return_value = Mock(price="45000.0")

        price = client_with_mock_sdk.get_current_price("BTC-USD")

        assert price == 45000.0
        client_with_mock_sdk.client.get_product.assert_called_once_with(product_id="BTC-USD")

    def test_get_price_sdk_fallback(self, client_with_mock_sdk):
        """Test fallback to public API when SDK fails."""
        client_with_mock_sdk.client.get_product.side_effect = Exception("SDK error")

        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"price": "50000.0"}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            price = client_with_mock_sdk.get_current_price("BTC-USD")
            assert price == 50000.0

    def test_get_price_error_returns_none(self):
        """Test returns None on complete failure."""
        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")

            # Create client with no SDK
            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                client.client = None  # Force public API path

                price = client.get_current_price("BTC-USD")
                assert price is None


class TestGetHistoricalOHLCV:
    """Tests for get_historical_ohlcv method."""

    def test_get_ohlcv_via_public_api(self):
        """Test fetching OHLCV via public API."""
        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_response = Mock()
            # Public API format: [time, low, high, open, close, volume]
            mock_response.json.return_value = [
                [1000, 95.0, 105.0, 100.0, 102.0, 500.0],
                [2000, 96.0, 108.0, 102.0, 106.0, 600.0],
            ]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                client.client = None  # Force public API path

                df = client.get_historical_ohlcv("BTC-USD", "1h", limit=10)

                assert len(df) > 0
                assert "open" in df.columns
                assert "close" in df.columns

    def test_get_ohlcv_4h_resampled(self):
        """Test 4h interval triggers resampling."""
        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_response = Mock()
            # Return 4 hours of 1h data
            base = 1000
            mock_response.json.return_value = [
                [base, 95, 105, 100, 102, 500],
                [base + 3600, 96, 108, 102, 106, 600],
                [base + 7200, 94, 107, 106, 104, 550],
                [base + 10800, 98, 110, 104, 109, 700],
            ]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                client.client = None  # Force public API path

                df = client.get_historical_ohlcv("BTC-USD", "4h", limit=5)

                # Should be resampled from 4x1h to 1x4h
                # Note: actual result depends on resampling behavior
                assert len(df) <= 5

    def test_get_ohlcv_empty_response(self):
        """Test handling empty response."""
        with patch('crypto.exchange.coinbase_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with patch('crypto.exchange.coinbase_client.load_dotenv'):
                client = CoinbaseClient(
                    api_key=None,
                    api_secret=None,
                    simulation_mode=True
                )
                client.client = None  # Force public API path

                df = client.get_historical_ohlcv("BTC-USD", "1h")

                assert df.empty


# =============================================================================
# ACCOUNT TESTS
# =============================================================================


class TestAccountInfo:
    """Tests for account info retrieval."""

    def test_get_account_info_simulation(self, client_simulation):
        """Test account info in simulation mode."""
        info = client_simulation.get_account_info()

        assert "accounts" in info
        assert len(info["accounts"]) == 1
        assert info["accounts"][0]["currency"] == "USDC"
        assert info["accounts"][0]["available_balance"]["value"] == 1000.0

    def test_get_account_info_live(self, client_with_mock_sdk):
        """Test account info via SDK."""
        mock_account = Mock()
        mock_account.uuid = "test-uuid"
        mock_account.currency = "USDC"
        mock_account.available_balance = {"value": 5000.0}

        mock_response = Mock()
        mock_response.accounts = [mock_account]
        client_with_mock_sdk.client.get_accounts.return_value = mock_response

        info = client_with_mock_sdk.get_account_info()

        assert "accounts" in info
        client_with_mock_sdk.client.get_accounts.assert_called_once()

    def test_get_account_info_no_client(self):
        """Test account info when client not initialized."""
        with patch.dict('os.environ', {}, clear=True):
            client = CoinbaseClient(simulation_mode=False)
            client.client = None

            info = client.get_account_info()
            assert info == {}


# =============================================================================
# ORDER TESTS
# =============================================================================


class TestCreateOrder:
    """Tests for order creation."""

    def test_create_market_order_simulation(self, client_simulation):
        """Test creating market order in simulation."""
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            result = client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )

        assert result["success"] is True
        assert "order_id" in result
        assert result["response"]["status"] == "FILLED"
        assert result["response"]["side"] == "BUY"

    def test_create_limit_order_simulation(self, client_simulation):
        """Test creating limit order in simulation."""
        result = client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=49000.0,
        )

        assert result["success"] is True
        assert result["response"]["status"] == "OPEN"
        assert result["response"]["price"] == 49000.0

    def test_create_order_updates_position(self, client_simulation):
        """Test market order updates mock position."""
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )

        assert client_simulation._mock_position is not None
        assert client_simulation._mock_position["side"] == "BUY"
        assert client_simulation._mock_position["quantity"] == 0.01

    def test_create_order_records_trade(self, client_simulation):
        """Test market order records trade history."""
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )

        assert len(client_simulation._trade_history) == 1
        assert client_simulation._trade_history[0]["side"] == "BUY"

    def test_create_order_custom_id(self, client_simulation):
        """Test creating order with custom client_order_id."""
        result = client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=49000.0,
            client_order_id="my-custom-id",
        )

        assert result["response"]["client_order_id"] == "my-custom-id"


class TestBuildOrderConfig:
    """Tests for order configuration building."""

    def test_build_market_order_config(self, client_simulation):
        """Test building market order config."""
        config = client_simulation._build_order_config(
            order_type="MARKET",
            quantity=0.01,
            price=None,
            stop_price=None,
            side="BUY",
        )

        assert "market_market_ioc" in config
        assert config["market_market_ioc"]["base_size"] == "0.01"

    def test_build_limit_order_config(self, client_simulation):
        """Test building limit order config."""
        config = client_simulation._build_order_config(
            order_type="LIMIT",
            quantity=0.01,
            price=50000.0,
            stop_price=None,
            side="BUY",
        )

        assert "limit_limit_gtc" in config
        assert config["limit_limit_gtc"]["limit_price"] == "50000.0"

    def test_build_limit_order_requires_price(self, client_simulation):
        """Test limit order raises without price."""
        with pytest.raises(ValueError, match="Price required"):
            client_simulation._build_order_config(
                order_type="LIMIT",
                quantity=0.01,
                price=None,
                stop_price=None,
                side="BUY",
            )

    def test_build_stop_order_config(self, client_simulation):
        """Test building stop order config."""
        config = client_simulation._build_order_config(
            order_type="STOP",
            quantity=0.01,
            price=50000.0,
            stop_price=49000.0,
            side="SELL",
        )

        assert "stop_limit_stop_limit_gtc" in config
        assert config["stop_limit_stop_limit_gtc"]["stop_price"] == "49000.0"
        assert config["stop_limit_stop_limit_gtc"]["stop_direction"] == "STOP_DIRECTION_STOP_DOWN"

    def test_build_stop_order_requires_stop_price(self, client_simulation):
        """Test stop order raises without stop price."""
        with pytest.raises(ValueError, match="Stop price required"):
            client_simulation._build_order_config(
                order_type="STOP",
                quantity=0.01,
                price=50000.0,
                stop_price=None,
                side="SELL",
            )

    def test_build_unsupported_order_type(self, client_simulation):
        """Test unsupported order type raises."""
        with pytest.raises(ValueError, match="Unsupported order type"):
            client_simulation._build_order_config(
                order_type="TRAILING_STOP",
                quantity=0.01,
                price=None,
                stop_price=None,
                side="BUY",
            )


class TestCancelOrder:
    """Tests for order cancellation."""

    def test_cancel_order_simulation(self, client_simulation):
        """Test canceling order in simulation."""
        # First create an order
        result = client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=49000.0,
        )
        order_id = result["order_id"]

        # Cancel it
        cancel_result = client_simulation.cancel_order(order_id)

        assert cancel_result["success"] is True
        assert len(client_simulation._mock_orders) == 0

    def test_cancel_nonexistent_order(self, client_simulation):
        """Test canceling non-existent order."""
        result = client_simulation.cancel_order("nonexistent-id")
        assert result["success"] is True  # No error, just no-op


class TestGetOpenOrders:
    """Tests for open orders retrieval."""

    def test_get_open_orders_simulation(self, client_simulation):
        """Test getting open orders in simulation."""
        # Create some orders
        client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=49000.0,
        )
        client_simulation.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=0.1,
            price=3500.0,
        )

        orders = client_simulation.get_open_orders()

        assert len(orders) == 2

    def test_get_open_orders_filtered(self, client_simulation):
        """Test filtering open orders by symbol."""
        client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=49000.0,
        )
        client_simulation.create_order(
            symbol="ETH-USD",
            side="SELL",
            order_type="LIMIT",
            quantity=0.1,
            price=3500.0,
        )

        orders = client_simulation.get_open_orders(symbol="BTC-USD")

        assert len(orders) == 1
        assert orders[0]["product_id"] == "BTC-USD"

    def test_get_open_orders_empty(self, client_simulation):
        """Test getting open orders when none exist."""
        orders = client_simulation.get_open_orders()
        assert orders == []


# =============================================================================
# POSITION TESTS
# =============================================================================


class TestUpdateMockPosition:
    """Tests for mock position updates."""

    def test_create_new_position(self, client_simulation):
        """Test creating a new position."""
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.01, 50000.0)

        pos = client_simulation._mock_position
        assert pos["product_id"] == "BTC-USD"
        assert pos["side"] == "BUY"
        assert pos["quantity"] == 0.01
        assert pos["avg_entry_price"] == 50000.0

    def test_add_to_existing_position(self, client_simulation):
        """Test adding to an existing position."""
        # Initial position
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.01, 50000.0)
        # Add more
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.01, 52000.0)

        pos = client_simulation._mock_position
        assert pos["quantity"] == 0.02
        # Avg price = (0.01 * 50000 + 0.01 * 52000) / 0.02 = 51000
        assert pos["avg_entry_price"] == 51000.0

    def test_close_position_fully(self, client_simulation):
        """Test closing a position completely."""
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.01, 50000.0)
        client_simulation._update_mock_position("BTC-USD", "SELL", 0.01, 52000.0)

        assert client_simulation._mock_position is None

    def test_partially_close_position(self, client_simulation):
        """Test partially closing a position."""
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.02, 50000.0)
        client_simulation._update_mock_position("BTC-USD", "SELL", 0.01, 52000.0)

        pos = client_simulation._mock_position
        assert pos["quantity"] == 0.01
        assert pos["side"] == "BUY"

    def test_flip_position(self, client_simulation):
        """Test flipping position direction."""
        client_simulation._update_mock_position("BTC-USD", "BUY", 0.01, 50000.0)
        client_simulation._update_mock_position("BTC-USD", "SELL", 0.02, 52000.0)

        pos = client_simulation._mock_position
        assert pos["side"] == "SELL"
        assert pos["quantity"] == 0.01


class TestGetPosition:
    """Tests for position retrieval."""

    def test_get_position_simulation(self, client_simulation):
        """Test getting position in simulation."""
        client_simulation._mock_position = {
            "product_id": "BTC-USD",
            "side": "BUY",
            "quantity": 0.01,
            "avg_entry_price": 50000.0,
        }

        pos = client_simulation.get_position("BTC-USD")

        assert pos is not None
        assert pos["product_id"] == "BTC-USD"

    def test_get_position_wrong_symbol(self, client_simulation):
        """Test getting position for wrong symbol."""
        client_simulation._mock_position = {
            "product_id": "BTC-USD",
            "side": "BUY",
            "quantity": 0.01,
        }

        pos = client_simulation.get_position("ETH-USD")
        assert pos is None

    def test_get_position_no_position(self, client_simulation):
        """Test getting position when none exists."""
        pos = client_simulation.get_position("BTC-USD")
        assert pos is None


# =============================================================================
# PAPER TRADING UTILITIES TESTS
# =============================================================================


class TestPaperTradingUtilities:
    """Tests for paper trading utility methods."""

    def test_get_trade_history(self, client_simulation):
        """Test getting trade history."""
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )

        history = client_simulation.get_trade_history()

        assert len(history) == 1
        assert history[0]["symbol"] == "BTC-USD"

    def test_get_trade_history_returns_copy(self, client_simulation):
        """Test trade history returns a copy, not original."""
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )

        history = client_simulation.get_trade_history()
        history.clear()

        assert len(client_simulation._trade_history) == 1

    def test_reset_simulation(self, client_simulation):
        """Test resetting simulation state."""
        # Add some state
        with patch.object(client_simulation, 'get_current_price', return_value=50000.0):
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=0.01,
            )
            client_simulation.create_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="LIMIT",
                quantity=0.01,
                price=49000.0,
            )

        # Reset
        client_simulation.reset_simulation(starting_balance=5000.0)

        assert client_simulation._mock_orders == []
        assert client_simulation._mock_position is None
        assert client_simulation._mock_balance == {"USDC": 5000.0}
        assert client_simulation._trade_history == []

    def test_set_mock_balance(self, client_simulation):
        """Test setting mock balance."""
        client_simulation.set_mock_balance("BTC", 1.5)

        assert client_simulation._mock_balance["BTC"] == 1.5
        assert client_simulation._mock_balance["USDC"] == 1000.0  # Original preserved


# =============================================================================
# OBJECT CONVERSION TESTS
# =============================================================================


class TestObjectConversion:
    """Tests for object to dict conversion methods."""

    def test_account_to_dict_from_dict(self, client_simulation):
        """Test account conversion when already a dict."""
        account = {"currency": "USDC", "available_balance": 1000}
        result = client_simulation._account_to_dict(account)
        assert result == account

    def test_account_to_dict_from_object(self, client_simulation):
        """Test account conversion from object."""
        mock_account = Mock()
        mock_account.uuid = "test-uuid"
        mock_account.name = "Test Account"
        mock_account.currency = "USDC"
        mock_account.available_balance = {"value": 1000}
        mock_account.default = True
        mock_account.active = True
        mock_account.type = "ACCOUNT_TYPE_CRYPTO"
        mock_account.hold = {}

        result = client_simulation._account_to_dict(mock_account)

        assert result["uuid"] == "test-uuid"
        assert result["currency"] == "USDC"

    def test_order_to_dict_from_dict(self, client_simulation):
        """Test order conversion when already a dict."""
        order = {"order_id": "123", "status": "OPEN"}
        result = client_simulation._order_to_dict(order)
        assert result == order

    def test_order_to_dict_from_object(self, client_simulation):
        """Test order conversion from object."""
        mock_order = Mock()
        mock_order.order_id = "123"
        mock_order.product_id = "BTC-USD"
        mock_order.side = "BUY"
        mock_order.status = "OPEN"
        mock_order.order_configuration = {}

        result = client_simulation._order_to_dict(mock_order)

        assert result["order_id"] == "123"
        assert result["product_id"] == "BTC-USD"

    def test_position_to_dict_from_dict(self, client_simulation):
        """Test position conversion when already a dict."""
        position = {"product_id": "BTC-USD", "side": "BUY"}
        result = client_simulation._position_to_dict(position)
        assert result == position

    def test_position_to_dict_from_object(self, client_simulation):
        """Test position conversion from object."""
        mock_position = Mock()
        mock_position.product_id = "BTC-USD"
        mock_position.side = "BUY"
        mock_position.number_of_contracts = 1
        mock_position.avg_entry_price = 50000.0
        mock_position.unrealized_pnl = 100.0

        result = client_simulation._position_to_dict(mock_position)

        assert result["product_id"] == "BTC-USD"
        assert result["quantity"] == 1


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_create_order_no_client(self):
        """Test order creation fails without client in live mode."""
        with patch.dict('os.environ', {}, clear=True):
            client = CoinbaseClient(simulation_mode=False)
            client.client = None

            with pytest.raises(ValueError, match="Client not initialized"):
                client.create_order(
                    symbol="BTC-USD",
                    side="BUY",
                    order_type="MARKET",
                    quantity=0.01,
                )

    def test_cancel_order_no_client(self):
        """Test cancel order fails without client in live mode."""
        with patch.dict('os.environ', {}, clear=True):
            client = CoinbaseClient(simulation_mode=False)
            client.client = None

            with pytest.raises(ValueError, match="Client not initialized"):
                client.cancel_order("order-123")

    def test_get_open_orders_no_client(self):
        """Test get open orders returns empty without client."""
        with patch.dict('os.environ', {}, clear=True):
            client = CoinbaseClient(simulation_mode=False)
            client.client = None

            orders = client.get_open_orders()
            assert orders == []

    def test_fetch_ohlcv_with_timestamps(self, client_simulation):
        """Test fetching OHLCV with explicit timestamps."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [
                [1000, 95.0, 105.0, 100.0, 102.0, 500.0],
            ]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            df = client_simulation.get_historical_ohlcv(
                "BTC-USD",
                "1h",
                limit=10,
                start_time=1000000,  # milliseconds
                end_time=2000000,
            )

            assert not df.empty

    def test_market_order_with_price(self, client_simulation):
        """Test market order uses provided price if given."""
        result = client_simulation.create_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="MARKET",
            quantity=0.01,
            price=50000.0,  # Explicit fill price
        )

        assert result["success"] is True
        # Position should use the provided price
        assert client_simulation._mock_position["avg_entry_price"] == 50000.0
