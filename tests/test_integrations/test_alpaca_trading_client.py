"""
Tests for AlpacaTradingClient

Tests cover:
- Connection and initialization
- Account information retrieval
- Order submission (market, limit)
- Position tracking
- Order status and cancellation
- Retry logic with exponential backoff
- Rate limiting
- Error handling
- Parameter validation

Note: Uses mocked API calls (no real API requests).
"""

import os
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderType
from alpaca.common.exceptions import APIError

from integrations.alpaca_trading_client import AlpacaTradingClient


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv('APCA_API_KEY_ID', 'test_key')
    monkeypatch.setenv('APCA_API_SECRET_KEY', 'test_secret')
    monkeypatch.setenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')


@pytest.fixture
def mock_account():
    """Mock Alpaca account object."""
    account = Mock()
    account.id = 'test_account_123'
    account.equity = 10000.0
    account.cash = 8000.0
    account.buying_power = 8000.0
    account.portfolio_value = 10000.0
    account.pattern_day_trader = False
    account.trading_blocked = False
    account.account_blocked = False
    return account


@pytest.fixture
def mock_order():
    """Mock Alpaca order object."""
    order = Mock()
    order.id = 'order_123'
    order.symbol = 'SPY'
    order.qty = 10
    order.side = OrderSide.BUY
    order.type = OrderType.MARKET
    order.status = OrderStatus.ACCEPTED
    order.time_in_force = TimeInForce.DAY
    order.limit_price = None
    order.filled_qty = 0
    order.filled_avg_price = None
    order.submitted_at = datetime.now()
    order.filled_at = None
    return order


@pytest.fixture
def mock_position():
    """Mock Alpaca position object."""
    position = Mock()
    position.symbol = 'SPY'
    position.qty = 10
    position.avg_entry_price = 450.00
    position.market_value = 4500.00
    position.unrealized_pl = 50.00
    position.unrealized_plpc = 0.0111
    position.current_price = 455.00
    return position


class TestAlpacaTradingClientInitialization:
    """Test client initialization and configuration."""

    def test_init_with_valid_account(self, mock_env):
        """Test initialization with valid account."""
        client = AlpacaTradingClient(account='LARGE')

        assert client.account == 'LARGE'
        assert client.account_config['capital'] == 10000
        assert client.api_key == 'test_key'
        assert client.secret_key == 'test_secret'
        assert client.connected is False
        assert client.client is None

    def test_init_with_invalid_account(self, mock_env):
        """Test initialization with invalid account raises error."""
        with pytest.raises(ValueError, match="Invalid account"):
            AlpacaTradingClient(account='INVALID')

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization without API key raises error."""
        monkeypatch.delenv('APCA_API_KEY_ID', raising=False)
        monkeypatch.delenv('APCA_API_SECRET_KEY', raising=False)

        with pytest.raises(ValueError, match="Missing Alpaca API credentials"):
            AlpacaTradingClient(account='LARGE')

    def test_custom_logger(self, mock_env):
        """Test initialization with custom logger."""
        import logging
        custom_logger = logging.getLogger('test_logger')

        client = AlpacaTradingClient(account='LARGE', logger=custom_logger)

        assert client.logger == custom_logger


class TestAlpacaTradingClientConnection:
    """Test connection functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_connect_success(self, mock_trading_client, mock_env, mock_account):
        """Test successful connection."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        result = client.connect()

        assert result is True
        assert client.connected is True
        assert client.client == mock_client_instance
        mock_trading_client.assert_called_once_with(
            api_key='test_key',
            secret_key='test_secret',
            paper=True
        )

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_connect_failure(self, mock_trading_client, mock_env):
        """Test connection failure."""
        mock_trading_client.side_effect = Exception("Connection failed")

        client = AlpacaTradingClient(account='LARGE')
        result = client.connect()

        assert result is False
        assert client.connected is False
        assert client.client is None


class TestAlpacaTradingClientAccountOperations:
    """Test account information operations."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_account_success(self, mock_trading_client, mock_env, mock_account):
        """Test getting account information."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        account_info = client.get_account()

        assert account_info['account_id'] == 'test_account_123'
        assert account_info['equity'] == 10000.0
        assert account_info['cash'] == 8000.0
        assert account_info['buying_power'] == 8000.0
        assert account_info['pattern_day_trader'] is False

    def test_get_account_not_connected(self, mock_env):
        """Test getting account when not connected raises error."""
        client = AlpacaTradingClient(account='LARGE')

        with pytest.raises(RuntimeError, match="Not connected"):
            client.get_account()


class TestAlpacaTradingClientOrderSubmission:
    """Test order submission operations."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_submit_market_order_buy(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test submitting market buy order."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.submit_market_order('SPY', 10, 'buy')

        assert order['id'] == 'order_123'
        assert order['symbol'] == 'SPY'
        assert order['qty'] == 10
        assert order['side'] == 'buy'
        assert order['type'] == 'market'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_submit_market_order_sell(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test submitting market sell order."""
        mock_order.side = OrderSide.SELL
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.submit_market_order('SPY', 10, 'sell')

        assert order['side'] == 'sell'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_submit_limit_order(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test submitting limit order."""
        mock_order.type = OrderType.LIMIT
        mock_order.limit_price = 450.00
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.submit_limit_order('SPY', 10, 'buy', 450.00)

        assert order['type'] == 'limit'
        assert order['limit_price'] == 450.00

    def test_submit_order_invalid_params(self, mock_env):
        """Test order submission with invalid parameters."""
        client = AlpacaTradingClient(account='LARGE')
        client.connected = True
        client.client = Mock()

        # Invalid symbol
        with pytest.raises(ValueError, match="Invalid symbol"):
            client.submit_market_order('', 10, 'buy')

        # Invalid quantity
        with pytest.raises(ValueError, match="positive integer"):
            client.submit_market_order('SPY', -5, 'buy')

        # Invalid side
        with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
            client.submit_market_order('SPY', 10, 'invalid')

        # Invalid limit price
        with pytest.raises(ValueError, match="must be positive"):
            client.submit_limit_order('SPY', 10, 'buy', -100.0)


class TestAlpacaTradingClientPositions:
    """Test position tracking operations."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_position_exists(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_position
    ):
        """Test getting existing position."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.return_value = mock_position
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        position = client.get_position('SPY')

        assert position is not None
        assert position['symbol'] == 'SPY'
        assert position['qty'] == 10
        assert position['side'] == 'long'
        assert position['avg_entry_price'] == 450.00

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_position_not_exists(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test getting non-existent position returns None."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.side_effect = APIError(
            "position does not exist"
        )
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        position = client.get_position('AAPL')

        assert position is None

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_list_positions(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_position
    ):
        """Test listing all positions."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_all_positions.return_value = [mock_position]
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        positions = client.list_positions()

        assert len(positions) == 1
        assert positions[0]['symbol'] == 'SPY'


class TestAlpacaTradingClientOrderManagement:
    """Test order management operations."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_order(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test getting order by ID."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_order_by_id.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.get_order('order_123')

        assert order['id'] == 'order_123'
        assert order['symbol'] == 'SPY'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_cancel_order_success(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test successful order cancellation."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.cancel_order_by_id.return_value = None
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        result = client.cancel_order('order_123')

        assert result is True
        mock_client_instance.cancel_order_by_id.assert_called_once_with('order_123')

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_cancel_order_failure(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test failed order cancellation."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.cancel_order_by_id.side_effect = APIError("Order not found")
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        result = client.cancel_order('order_123')

        assert result is False


class TestAlpacaTradingClientRetryLogic:
    """Test retry logic and error handling."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_retry_on_timeout(self, mock_trading_client, mock_env, mock_account):
        """Test retry on timeout error."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.side_effect = [
            APIError("Connection timeout"),
            APIError("Connection timeout"),
            mock_account  # Third attempt succeeds
        ]
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        # Should succeed after retries
        account = client.get_account()
        assert account['account_id'] == 'test_account_123'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_retry_exhausted(self, mock_trading_client, mock_env, mock_account):
        """Test all retries exhausted."""
        mock_client_instance = Mock()
        # All attempts fail
        mock_client_instance.get_account.side_effect = [
            mock_account,  # Initial connect succeeds
            APIError("Connection timeout"),
            APIError("Connection timeout"),
            APIError("Connection timeout")
        ]
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        # Should raise after max retries
        with pytest.raises(APIError):
            client.get_account()

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_no_retry_on_non_retryable_error(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test no retry on non-retryable errors."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.side_effect = [
            mock_account,  # Initial connect succeeds
            APIError("Invalid symbol")  # Non-retryable error
        ]
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        # Should raise immediately without retry
        with pytest.raises(APIError):
            client.get_account()


class TestAlpacaTradingClientRateLimiting:
    """Test rate limiting functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_rate_limit_enforcement(self, mock_trading_client, mock_env, mock_account):
        """Test rate limit is enforced."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        # Set artificially low rate limit for testing
        client.max_requests_per_minute = 5

        # Make requests up to limit
        start_time = time.time()
        for _ in range(6):
            client.get_account()
        elapsed = time.time() - start_time

        # Should have been throttled (waited at least 1 second)
        assert elapsed >= 1.0

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_rate_limit_window_reset(self, mock_trading_client, mock_env, mock_account):
        """Test rate limit window resets correctly."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        # Add old timestamps (should be cleaned up)
        client.request_timestamps = [time.time() - 120]  # 2 minutes ago

        # Should not affect current requests
        client._check_rate_limit()
        assert len(client.request_timestamps) == 0
