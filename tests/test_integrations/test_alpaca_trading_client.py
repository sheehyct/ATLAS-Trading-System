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
    """Mock get_alpaca_credentials to return test credentials.

    Note: AlpacaTradingClient uses config.settings.get_alpaca_credentials()
    which reads from .env file. We must mock the function, not env vars.
    """
    def mock_get_credentials(account):
        return {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }

    monkeypatch.setattr(
        'integrations.alpaca_trading_client.get_alpaca_credentials',
        mock_get_credentials
    )


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
        """Test initialization without API key raises error.

        Note: Mock get_alpaca_credentials to return empty credentials.
        """
        def mock_get_credentials_empty(account):
            return {
                'api_key': '',
                'secret_key': '',
                'base_url': 'https://paper-api.alpaca.markets'
            }

        monkeypatch.setattr(
            'integrations.alpaca_trading_client.get_alpaca_credentials',
            mock_get_credentials_empty
        )

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
        """Test retry on timeout error.

        Note: connect() calls get_account() internally, so side_effect list
        must account for both connect() and subsequent get_account() calls.
        """
        mock_client_instance = Mock()
        # First call: connect() success, then timeouts, then success
        mock_client_instance.get_account.side_effect = [
            mock_account,  # connect() internal call succeeds
            APIError("Connection timeout"),  # First retry
            APIError("Connection timeout"),  # Second retry
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


# =============================================================================
# OPTIONS TRADING TESTS (Session 83K-47)
# =============================================================================

@pytest.fixture
def mock_option_contract():
    """Mock Alpaca option contract object."""
    contract = Mock()
    contract.symbol = 'SPY241220C00450000'
    contract.underlying_symbol = 'SPY'
    contract.strike_price = 450.0
    contract.expiration_date = datetime(2024, 12, 20)
    contract.type = Mock(value='call')
    contract.status = Mock(value='active')
    contract.tradable = True
    return contract


@pytest.fixture
def mock_option_position():
    """Mock Alpaca option position object."""
    position = Mock()
    position.symbol = 'SPY241220C00450000'
    position.qty = 2
    position.avg_entry_price = 5.50
    position.market_value = 1200.00
    position.cost_basis = 1100.00
    position.unrealized_pl = 100.00
    position.unrealized_plpc = 0.0909
    position.current_price = 6.00
    return position


class TestAlpacaOptionsContractDiscovery:
    """Test options contract discovery functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_option_contracts_success(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_option_contract
    ):
        """Test getting available option contracts."""
        mock_response = Mock()
        mock_response.option_contracts = [mock_option_contract]

        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_option_contracts.return_value = mock_response
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        contracts = client.get_option_contracts('SPY')

        assert len(contracts) == 1
        assert contracts[0]['symbol'] == 'SPY241220C00450000'
        assert contracts[0]['underlying'] == 'SPY'
        assert contracts[0]['strike'] == 450.0
        assert contracts[0]['type'] == 'call'
        assert contracts[0]['tradable'] is True

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_option_contracts_with_filters(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_option_contract
    ):
        """Test getting option contracts with filters."""
        mock_response = Mock()
        mock_response.option_contracts = [mock_option_contract]

        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_option_contracts.return_value = mock_response
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        contracts = client.get_option_contracts(
            underlying='SPY',
            contract_type='call',
            strike_price_gte=440.0,
            strike_price_lte=460.0,
        )

        assert len(contracts) == 1
        mock_client_instance.get_option_contracts.assert_called_once()


class TestAlpacaOptionsOrderSubmission:
    """Test options order submission functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_submit_option_market_order(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test submitting options market order."""
        mock_order.symbol = 'SPY241220C00450000'
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.submit_option_market_order(
            symbol='SPY241220C00450000',
            qty=2,
            side='buy'
        )

        assert order['symbol'] == 'SPY241220C00450000'
        assert order['asset_class'] == 'option'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_submit_option_limit_order(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_order
    ):
        """Test submitting options limit order."""
        mock_order.symbol = 'SPY241220C00450000'
        mock_order.type = OrderType.LIMIT
        mock_order.limit_price = 5.50

        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.submit_option_limit_order(
            symbol='SPY241220C00450000',
            qty=2,
            side='buy',
            limit_price=5.50
        )

        assert order['symbol'] == 'SPY241220C00450000'
        assert order['asset_class'] == 'option'
        assert order['limit_price'] == 5.50

    def test_submit_option_order_invalid_symbol_format(self, mock_env):
        """Test options order with invalid OCC symbol."""
        client = AlpacaTradingClient(account='LARGE')
        client.connected = True
        client.client = Mock()

        # Too short symbol (not OCC format)
        with pytest.raises(ValueError, match="Invalid OCC option symbol"):
            client.submit_option_market_order('SPY', 2, 'buy')

    def test_submit_option_order_invalid_qty(self, mock_env):
        """Test options order with invalid quantity."""
        client = AlpacaTradingClient(account='LARGE')
        client.connected = True
        client.client = Mock()

        with pytest.raises(ValueError, match="positive integer"):
            client.submit_option_market_order('SPY241220C00450000', -1, 'buy')


class TestAlpacaOptionsPositions:
    """Test options position tracking functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_option_position_exists(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_option_position
    ):
        """Test getting existing option position."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.return_value = mock_option_position
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        position = client.get_option_position('SPY241220C00450000')

        assert position is not None
        assert position['symbol'] == 'SPY241220C00450000'
        assert position['qty'] == 2
        assert position['avg_entry_price'] == 5.50
        assert position['asset_class'] == 'option'

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_option_position_not_exists(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test getting non-existent option position."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.side_effect = APIError(
            "position does not exist"
        )
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        position = client.get_option_position('SPY241220C00450000')

        assert position is None

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_list_option_positions(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_option_position
    ):
        """Test listing all option positions."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_all_positions.return_value = [mock_option_position]
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        positions = client.list_option_positions()

        assert len(positions) == 1
        assert positions[0]['symbol'] == 'SPY241220C00450000'
        assert positions[0]['asset_class'] == 'option'


class TestAlpacaOptionsPositionClose:
    """Test options position closing functionality."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_close_option_position(
        self,
        mock_trading_client,
        mock_env,
        mock_account,
        mock_option_position,
        mock_order
    ):
        """Test closing an option position."""
        mock_order.symbol = 'SPY241220C00450000'
        mock_order.side = OrderSide.SELL

        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.return_value = mock_option_position
        mock_client_instance.submit_order.return_value = mock_order
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        order = client.close_option_position('SPY241220C00450000')

        assert order is not None
        mock_client_instance.submit_order.assert_called_once()

    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_close_option_position_no_position(
        self,
        mock_trading_client,
        mock_env,
        mock_account
    ):
        """Test closing non-existent position raises error."""
        mock_client_instance = Mock()
        mock_client_instance.get_account.return_value = mock_account
        mock_client_instance.get_open_position.side_effect = APIError(
            "position does not exist"
        )
        mock_trading_client.return_value = mock_client_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        with pytest.raises(RuntimeError, match="No open position"):
            client.close_option_position('SPY241220C00450000')


# =============================================================================
# MARKET DATA TESTS (Session 83K-50)
# =============================================================================


@pytest.fixture
def mock_quote():
    """Mock stock quote object."""
    quote = Mock()
    quote.bid_price = 600.00
    quote.ask_price = 600.10
    quote.bid_size = 100
    quote.ask_size = 150
    quote.timestamp = datetime.now()
    return quote


class TestAlpacaMarketData:
    """Test market data methods (Session 83K-50)."""

    @patch('integrations.alpaca_trading_client.StockHistoricalDataClient')
    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_stock_quote(
        self,
        mock_trading_client,
        mock_data_client,
        mock_env,
        mock_account,
        mock_quote
    ):
        """Test fetching single stock quote."""
        # Setup trading client
        mock_trading_instance = Mock()
        mock_trading_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_trading_instance

        # Setup data client
        mock_data_instance = Mock()
        mock_data_instance.get_stock_latest_quote.return_value = {'SPY': mock_quote}
        mock_data_client.return_value = mock_data_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        quote = client.get_stock_quote('SPY')

        assert quote is not None
        assert quote['symbol'] == 'SPY'
        assert quote['bid'] == 600.00
        assert quote['ask'] == 600.10
        assert quote['mid'] == 600.05

    @patch('integrations.alpaca_trading_client.StockHistoricalDataClient')
    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_stock_quotes_batch(
        self,
        mock_trading_client,
        mock_data_client,
        mock_env,
        mock_account
    ):
        """Test fetching multiple stock quotes."""
        # Create mock quotes for multiple symbols
        spy_quote = Mock()
        spy_quote.bid_price = 600.00
        spy_quote.ask_price = 600.10
        spy_quote.bid_size = 100
        spy_quote.ask_size = 150
        spy_quote.timestamp = datetime.now()

        aapl_quote = Mock()
        aapl_quote.bid_price = 200.00
        aapl_quote.ask_price = 200.20
        aapl_quote.bid_size = 50
        aapl_quote.ask_size = 75
        aapl_quote.timestamp = datetime.now()

        # Setup trading client
        mock_trading_instance = Mock()
        mock_trading_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_trading_instance

        # Setup data client
        mock_data_instance = Mock()
        mock_data_instance.get_stock_latest_quote.return_value = {
            'SPY': spy_quote,
            'AAPL': aapl_quote
        }
        mock_data_client.return_value = mock_data_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        quotes = client.get_stock_quotes(['SPY', 'AAPL'])

        assert len(quotes) == 2
        assert 'SPY' in quotes
        assert 'AAPL' in quotes
        assert quotes['SPY']['mid'] == 600.05
        assert quotes['AAPL']['mid'] == 200.10

    @patch('integrations.alpaca_trading_client.StockHistoricalDataClient')
    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_stock_price(
        self,
        mock_trading_client,
        mock_data_client,
        mock_env,
        mock_account,
        mock_quote
    ):
        """Test convenience method for getting stock price."""
        # Setup trading client
        mock_trading_instance = Mock()
        mock_trading_instance.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_trading_instance

        # Setup data client
        mock_data_instance = Mock()
        mock_data_instance.get_stock_latest_quote.return_value = {'SPY': mock_quote}
        mock_data_client.return_value = mock_data_instance

        client = AlpacaTradingClient(account='LARGE')
        client.connect()

        price = client.get_stock_price('SPY')

        assert price == 600.05  # mid of 600.00 and 600.10

    @patch('integrations.alpaca_trading_client.StockHistoricalDataClient')
    @patch('integrations.alpaca_trading_client.TradingClient')
    def test_get_stock_quote_not_connected(
        self,
        mock_trading_client,
        mock_data_client,
        mock_env
    ):
        """Test quote fetch fails gracefully when not connected."""
        client = AlpacaTradingClient(account='LARGE')
        # Don't call connect()

        with pytest.raises(RuntimeError, match="Not connected"):
            client.get_stock_quote('SPY')


# =============================================================================
# EXECUTION PERSISTENCE TESTS (Session 83K-50)
# =============================================================================


class TestExecutionPersistence:
    """Test execution persistence functionality (Session 83K-50)."""

    def test_execution_result_serialization(self):
        """Test ExecutionResult to_dict and from_dict."""
        from strat.signal_automation.executor import ExecutionResult, ExecutionState

        # Create test result
        result = ExecutionResult(
            signal_key='SPY_1D_3-2U_202412060000',
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='test-order-123',
            osi_symbol='SPY241220C00600000',
            strike=600.0,
            expiration='2024-12-20',
            contracts=2,
            premium=5.50,
            side='buy'
        )

        # Serialize and deserialize
        data = result.to_dict()
        restored = ExecutionResult.from_dict(data)

        assert restored.signal_key == result.signal_key
        assert restored.state == result.state
        assert restored.order_id == result.order_id
        assert restored.osi_symbol == result.osi_symbol
        assert restored.strike == result.strike
        assert restored.expiration == result.expiration
        assert restored.contracts == result.contracts
        assert restored.premium == result.premium
        assert restored.side == result.side

    def test_executor_persistence_save_load(self, tmp_path):
        """Test executor saves and loads executions from disk."""
        from strat.signal_automation.executor import (
            SignalExecutor,
            ExecutorConfig,
            ExecutionResult,
            ExecutionState
        )

        # Create executor with temp path
        config = ExecutorConfig(persistence_path=str(tmp_path / 'executions'))
        executor = SignalExecutor(config=config)

        # Add test execution
        result = ExecutionResult(
            signal_key='TEST_SIGNAL_1',
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='test-123',
            osi_symbol='SPY241220C00600000'
        )
        executor._executions['TEST_SIGNAL_1'] = result
        executor._save()

        # Create new executor to load from disk
        executor2 = SignalExecutor(config=config)

        assert 'TEST_SIGNAL_1' in executor2._executions
        loaded = executor2._executions['TEST_SIGNAL_1']
        assert loaded.signal_key == 'TEST_SIGNAL_1'
        assert loaded.state == ExecutionState.ORDER_SUBMITTED
        assert loaded.order_id == 'test-123'

    def test_executor_persistence_survives_restart(self, tmp_path):
        """Test executions survive daemon restart."""
        from strat.signal_automation.executor import (
            SignalExecutor,
            ExecutorConfig,
            ExecutionResult,
            ExecutionState
        )
        import json

        # Simulate first run - save some executions
        config = ExecutorConfig(persistence_path=str(tmp_path / 'executions'))
        executor = SignalExecutor(config=config)

        for i in range(3):
            result = ExecutionResult(
                signal_key=f'SIGNAL_{i}',
                state=ExecutionState.ORDER_SUBMITTED,
                order_id=f'order-{i}'
            )
            executor._executions[f'SIGNAL_{i}'] = result

        executor._save()

        # Delete executor (simulate restart)
        del executor

        # Simulate second run - verify all loaded
        executor2 = SignalExecutor(config=config)

        assert len(executor2._executions) == 3
        for i in range(3):
            assert f'SIGNAL_{i}' in executor2._executions
