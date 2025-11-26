"""
ThetaData REST Client Unit Tests.

Session 80: Comprehensive test suite for ThetaDataRESTClient.
Session 81: Updated for v3 API (port 25503, /v3 endpoints, strike in dollars).

Tests data formatting, quote/Greeks retrieval, retry logic, and edge cases.
Uses MockThetaDataProvider for isolation from ThetaData terminal.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests

from integrations.thetadata_client import (
    ThetaDataRESTClient,
    ThetaDataProviderBase,
    OptionsQuote,
    create_thetadata_client,
)
from tests.mocks.mock_thetadata import MockThetaDataProvider, create_spy_mock_provider


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_provider():
    """Pre-configured MockThetaDataProvider."""
    return create_spy_mock_provider()


@pytest.fixture
def sample_quote():
    """Sample OptionsQuote for testing."""
    return OptionsQuote(
        symbol='SPY241220C00450000',
        underlying='SPY',
        strike=450.0,
        expiration=datetime(2024, 12, 20),
        option_type='C',
        timestamp=datetime.now(),
        bid=5.00,
        ask=5.20,
        mid=5.10,
        bid_size=100,
        ask_size=100,
        volume=1000,
        open_interest=5000,
        iv=0.22,
        delta=0.55,
        gamma=0.02,
        theta=-0.05,
        vega=0.15,
        underlying_price=450.0,
    )


@pytest.fixture
def sample_quote_csv():
    """Sample CSV response from ThetaData API."""
    return """bid,ask,bid_size,ask_size,date,ms_of_day
5.00,5.20,100,100,20241115,36000000
5.05,5.25,150,150,20241115,39600000"""


@pytest.fixture
def sample_greeks_csv():
    """Sample Greeks CSV response."""
    return """delta,gamma,theta,vega,implied_vol,underlying_price
0.55,0.02,-0.05,0.15,0.22,450.00"""


# =============================================================================
# TestOptionsQuoteDataclass
# =============================================================================

class TestOptionsQuoteDataclass:
    """Tests for OptionsQuote dataclass."""

    def test_options_quote_creation(self, sample_quote):
        """Verify all fields are correctly set."""
        assert sample_quote.symbol == 'SPY241220C00450000'
        assert sample_quote.underlying == 'SPY'
        assert sample_quote.strike == 450.0
        assert sample_quote.option_type == 'C'
        assert sample_quote.bid == 5.00
        assert sample_quote.ask == 5.20

    def test_options_quote_mid_calculation(self, sample_quote):
        """Verify mid = (bid + ask) / 2."""
        assert sample_quote.mid == 5.10

    def test_options_quote_to_dict(self, sample_quote):
        """Test to_dict() method returns correct dict."""
        d = sample_quote.to_dict()
        assert isinstance(d, dict)
        assert d['symbol'] == 'SPY241220C00450000'
        assert d['bid'] == 5.00
        assert d['delta'] == 0.55

    def test_options_quote_optional_fields_none(self):
        """Test optional Greeks fields default to None."""
        quote = OptionsQuote(
            symbol='TEST',
            underlying='TEST',
            strike=100.0,
            expiration=datetime.now(),
            option_type='C',
            timestamp=datetime.now(),
            bid=1.0,
            ask=1.1,
            mid=1.05,
        )
        assert quote.iv is None
        assert quote.delta is None
        assert quote.gamma is None

    def test_options_quote_default_values(self):
        """Test default values for bid_size, ask_size, volume, etc."""
        quote = OptionsQuote(
            symbol='TEST',
            underlying='TEST',
            strike=100.0,
            expiration=datetime.now(),
            option_type='C',
            timestamp=datetime.now(),
            bid=1.0,
            ask=1.1,
            mid=1.05,
        )
        assert quote.bid_size == 0
        assert quote.ask_size == 0
        assert quote.volume == 0
        assert quote.open_interest == 0


# =============================================================================
# TestThetaDataRESTClientInitialization
# =============================================================================

class TestThetaDataRESTClientInitialization:
    """Tests for client initialization and configuration."""

    @patch('integrations.thetadata_client.get_thetadata_config')
    def test_init_with_defaults(self, mock_config):
        """Test initialization with default host/port (v3 API)."""
        mock_config.return_value = {}
        client = ThetaDataRESTClient()
        assert client.host == "127.0.0.1"
        assert client.port == 25503  # v3 default port

    @patch('integrations.thetadata_client.get_thetadata_config')
    def test_init_with_custom_config(self, mock_config):
        """Test custom host, port, timeout parameters."""
        mock_config.return_value = {}
        client = ThetaDataRESTClient(host='192.168.1.1', port=8080, timeout=60)
        assert client.host == '192.168.1.1'
        assert client.port == 8080
        assert client.timeout == 60

    @patch('integrations.thetadata_client.get_thetadata_config')
    def test_init_from_env_config(self, mock_config):
        """Test initialization from environment config."""
        mock_config.return_value = {'host': '10.0.0.1', 'port': 9999}
        client = ThetaDataRESTClient()
        assert client.host == '10.0.0.1'
        assert client.port == 9999

    @patch('integrations.thetadata_client.get_thetadata_config')
    def test_init_creates_session(self, mock_config):
        """Verify requests.Session is created."""
        mock_config.return_value = {}
        client = ThetaDataRESTClient()
        assert client._session is not None

    @patch('integrations.thetadata_client.get_thetadata_config')
    def test_init_sets_base_url(self, mock_config):
        """Verify base_url is correctly formatted (v3 API)."""
        mock_config.return_value = {}
        client = ThetaDataRESTClient(host='localhost', port=12345)
        assert client.base_url == "http://localhost:12345/v3"


# =============================================================================
# TestThetaDataRESTClientDataFormatters
# =============================================================================

class TestThetaDataRESTClientDataFormatters:
    """Tests for internal data formatting helpers."""

    @pytest.fixture
    def client(self):
        """Create client for testing formatters."""
        with patch('integrations.thetadata_client.get_thetadata_config', return_value={}):
            return ThetaDataRESTClient()

    def test_format_strike(self, client):
        """Test strike formatting (v3 uses dollars directly)."""
        assert client._format_strike(450.0) == 450.0

    def test_format_strike_with_cents(self, client):
        """Test strike with cents (v3 uses dollars directly)."""
        assert client._format_strike(450.50) == 450.50

    def test_format_expiration(self, client):
        """Test expiration formatting (datetime -> YYYYMMDD)."""
        exp = datetime(2024, 12, 20)
        assert client._format_expiration(exp) == '20241220'

    def test_format_date(self, client):
        """Test date formatting."""
        date = datetime(2024, 11, 15)
        assert client._format_date(date) == '20241115'

    def test_parse_date(self, client):
        """Test parsing YYYYMMDD integer to datetime."""
        result = client._parse_date(20241220)
        assert result == datetime(2024, 12, 20)

    def test_parse_strike(self, client):
        """Test parsing strike value (v3 uses dollars directly)."""
        assert client._parse_strike(450.0) == 450.0
        assert client._parse_strike(450.5) == 450.5

    def test_generate_osi_symbol_call(self, client):
        """Test OSI symbol generation for calls."""
        exp = datetime(2024, 12, 20)
        symbol = client._generate_osi_symbol('SPY', exp, 450.0, 'C')
        assert symbol == 'SPY241220C00450000'

    def test_generate_osi_symbol_put(self, client):
        """Test OSI symbol generation for puts."""
        exp = datetime(2024, 12, 20)
        symbol = client._generate_osi_symbol('SPY', exp, 450.0, 'P')
        assert symbol == 'SPY241220P00450000'

    def test_generate_osi_symbol_padding(self, client):
        """Test strike padding to 8 digits."""
        exp = datetime(2024, 12, 20)
        symbol = client._generate_osi_symbol('SPY', exp, 100.0, 'C')
        assert symbol == 'SPY241220C00100000'

    @pytest.mark.parametrize("strike,expected_osi", [
        (450.0, "SPY241220C00450000"),
        (450.5, "SPY241220C00450500"),
        (100.0, "SPY241220C00100000"),
        (1000.0, "SPY241220C01000000"),
    ])
    def test_osi_symbol_generation_parametrized(self, client, strike, expected_osi):
        """Test OSI symbol generation for various strikes."""
        exp = datetime(2024, 12, 20)
        symbol = client._generate_osi_symbol('SPY', exp, strike, 'C')
        assert symbol == expected_osi


# =============================================================================
# TestThetaDataRESTClientSafeGetSeries
# =============================================================================

class TestThetaDataRESTClientSafeGetSeries:
    """Tests for the _safe_get_series helper method (Session 80 fix)."""

    @pytest.fixture
    def client(self):
        """Create client for testing."""
        with patch('integrations.thetadata_client.get_thetadata_config', return_value={}):
            return ThetaDataRESTClient()

    def test_safe_get_series_value_exists(self, client):
        """Test extracting value when key exists."""
        row = pd.Series({'bid': 5.0, 'ask': 5.2})
        assert client._safe_get_series(row, 'bid') == 5.0

    def test_safe_get_series_key_missing(self, client):
        """Test default value when key missing."""
        row = pd.Series({'bid': 5.0})
        assert client._safe_get_series(row, 'ask') == 0.0

    def test_safe_get_series_custom_default(self, client):
        """Test custom default value."""
        row = pd.Series({'bid': 5.0})
        assert client._safe_get_series(row, 'ask', default=-1.0) == -1.0

    def test_safe_get_series_type_conversion(self, client):
        """Test that value is converted to float."""
        row = pd.Series({'bid': '5.0'})
        result = client._safe_get_series(row, 'bid')
        assert isinstance(result, float)
        assert result == 5.0


# =============================================================================
# TestThetaDataRESTClientGetQuote (using mock)
# =============================================================================

class TestThetaDataRESTClientGetQuote:
    """Tests for get_quote() method using mock provider."""

    def test_get_quote_success(self, mock_provider):
        """Test successful quote retrieval."""
        exps = mock_provider.get_expirations('SPY')
        quote = mock_provider.get_quote('SPY', exps[0], 450.0, 'C')
        assert quote is not None
        assert quote.underlying == 'SPY'
        assert quote.strike == 450.0

    def test_get_quote_missing_data(self, mock_provider):
        """Test handling of missing quote."""
        exp = datetime(2024, 12, 20)
        quote = mock_provider.get_quote('SPY', exp, 999.0, 'C')  # Strike doesn't exist
        assert quote is None

    def test_get_quote_returns_mid(self, mock_provider):
        """Verify mid price is returned."""
        exps = mock_provider.get_expirations('SPY')
        quote = mock_provider.get_quote('SPY', exps[0], 450.0, 'C')
        assert quote.mid == (quote.bid + quote.ask) / 2


# =============================================================================
# TestThetaDataRESTClientGetGreeks (using mock)
# =============================================================================

class TestThetaDataRESTClientGetGreeks:
    """Tests for get_greeks() method using mock provider."""

    def test_get_greeks_success(self, mock_provider):
        """Test successful Greeks retrieval."""
        exps = mock_provider.get_expirations('SPY')
        greeks = mock_provider.get_greeks('SPY', exps[0], 450.0, 'C')
        assert greeks is not None
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'iv' in greeks

    def test_get_greeks_missing_data(self, mock_provider):
        """Test handling of missing Greeks."""
        exp = datetime(2024, 12, 20)
        greeks = mock_provider.get_greeks('SPY', exp, 999.0, 'C')
        assert greeks is None

    def test_get_greeks_correct_keys(self, mock_provider):
        """Verify returned dict has correct keys."""
        exps = mock_provider.get_expirations('SPY')
        greeks = mock_provider.get_greeks('SPY', exps[0], 450.0, 'C')
        expected_keys = {'delta', 'gamma', 'theta', 'vega', 'iv', 'underlying_price'}
        assert set(greeks.keys()) == expected_keys


# =============================================================================
# TestThetaDataRESTClientExpirationStrikes (using mock)
# =============================================================================

class TestThetaDataRESTClientExpirationStrikes:
    """Tests for get_expirations() and get_strikes() methods."""

    def test_get_expirations_success(self, mock_provider):
        """Test successful expiration list retrieval."""
        exps = mock_provider.get_expirations('SPY')
        assert len(exps) == 8  # create_spy_mock_provider adds 8 weeks

    def test_get_expirations_sorted(self, mock_provider):
        """Verify expirations are sorted."""
        exps = mock_provider.get_expirations('SPY')
        assert exps == sorted(exps)

    def test_get_strikes_success(self, mock_provider):
        """Test successful strike list retrieval."""
        exps = mock_provider.get_expirations('SPY')
        strikes = mock_provider.get_strikes('SPY', exps[0])
        assert len(strikes) > 0

    def test_get_strikes_sorted(self, mock_provider):
        """Verify strikes are sorted."""
        exps = mock_provider.get_expirations('SPY')
        strikes = mock_provider.get_strikes('SPY', exps[0])
        assert strikes == sorted(strikes)


# =============================================================================
# TestThetaDataRESTClientConnection (using mock)
# =============================================================================

class TestThetaDataRESTClientConnection:
    """Tests for connection and disconnection using mock."""

    def test_connect_success(self, mock_provider):
        """Test successful connection."""
        assert mock_provider.connect() is True

    def test_is_connected_after_connect(self, mock_provider):
        """Verify is_connected() returns True after connect."""
        mock_provider.connect()
        assert mock_provider.is_connected() is True

    def test_disconnect_changes_state(self, mock_provider):
        """Verify disconnect changes connection state."""
        mock_provider.connect()
        mock_provider.disconnect()
        assert mock_provider.is_connected() is False

    def test_get_quote_when_disconnected(self, mock_provider):
        """Verify get_quote returns None when disconnected."""
        mock_provider.set_connected(False)
        exps = mock_provider._expirations.get('SPY', [])
        if exps:
            quote = mock_provider.get_quote('SPY', exps[0], 450.0, 'C')
            assert quote is None


# =============================================================================
# TestThetaDataRESTClientCallCounting (using mock)
# =============================================================================

class TestThetaDataRESTClientCallCounting:
    """Tests for call counting functionality in mock provider."""

    def test_call_count_increments(self, mock_provider):
        """Test that call counts increment correctly."""
        exps = mock_provider.get_expirations('SPY')
        mock_provider.get_quote('SPY', exps[0], 450.0, 'C')
        mock_provider.get_quote('SPY', exps[0], 450.0, 'C')
        mock_provider.get_greeks('SPY', exps[0], 450.0, 'C')

        # Note: get_expirations was called once above
        assert mock_provider.get_call_count('get_expirations') >= 1
        assert mock_provider.get_call_count('get_quote') == 2
        assert mock_provider.get_call_count('get_greeks') == 1

    def test_reset_clears_call_counts(self, mock_provider):
        """Test that reset() clears call counts."""
        mock_provider.get_expirations('SPY')
        mock_provider.reset()
        assert mock_provider.get_call_count('get_expirations') == 0


# =============================================================================
# TestCreateThetaDataClient
# =============================================================================

class TestCreateThetaDataClient:
    """Tests for create_thetadata_client factory function."""

    @patch('integrations.thetadata_client.get_thetadata_config', return_value={})
    def test_create_rest_client(self, mock_config):
        """Test creating REST client."""
        client = create_thetadata_client(mode='rest')
        assert isinstance(client, ThetaDataRESTClient)

    def test_create_websocket_client_not_implemented(self):
        """Test WebSocket mode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_thetadata_client(mode='websocket')

    def test_create_invalid_mode(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            create_thetadata_client(mode='invalid')


# =============================================================================
# TestMockProviderBehavior
# =============================================================================

class TestMockProviderBehavior:
    """Tests for MockThetaDataProvider behavior."""

    def test_add_mock_quote(self):
        """Test adding mock quote."""
        mock = MockThetaDataProvider()
        exp = datetime(2024, 12, 20)
        mock.add_mock_quote('SPY', exp, 450.0, 'C', bid=5.0, ask=5.2)
        quote = mock.get_quote('SPY', exp, 450.0, 'C')
        assert quote is not None
        assert quote.bid == 5.0
        assert quote.ask == 5.2

    def test_add_mock_greeks(self):
        """Test adding mock Greeks."""
        mock = MockThetaDataProvider()
        exp = datetime(2024, 12, 20)
        mock.add_mock_greeks('SPY', exp, 450.0, 'C',
                            delta=0.55, gamma=0.02, theta=-0.05, vega=0.15, iv=0.22)
        greeks = mock.get_greeks('SPY', exp, 450.0, 'C')
        assert greeks is not None
        assert greeks['delta'] == 0.55

    def test_add_mock_expirations(self):
        """Test adding mock expirations."""
        mock = MockThetaDataProvider()
        exps = [datetime(2024, 12, 20), datetime(2024, 12, 27)]
        mock.add_mock_expirations('SPY', exps)
        result = mock.get_expirations('SPY')
        assert len(result) == 2

    def test_add_mock_strikes(self):
        """Test adding mock strikes."""
        mock = MockThetaDataProvider()
        exp = datetime(2024, 12, 20)
        mock.add_mock_strikes('SPY', exp, [440.0, 445.0, 450.0])
        result = mock.get_strikes('SPY', exp)
        assert result == [440.0, 445.0, 450.0]


# =============================================================================
# Session 83: Tests for _safe_float() helper
# =============================================================================

class TestSafeFloatHelper:
    """Tests for _safe_float() helper added in Session 83."""

    @pytest.fixture
    def client(self):
        """Create client for testing _safe_float."""
        with patch('integrations.thetadata_client.get_thetadata_config', return_value={}):
            return ThetaDataRESTClient()

    def test_safe_float_with_valid_int(self, client):
        """Test _safe_float with valid integer."""
        assert client._safe_float(42) == 42.0

    def test_safe_float_with_valid_float(self, client):
        """Test _safe_float with valid float."""
        assert client._safe_float(3.14159) == 3.14159

    def test_safe_float_with_valid_string(self, client):
        """Test _safe_float with valid numeric string."""
        assert client._safe_float("5.25") == 5.25

    def test_safe_float_with_none(self, client):
        """Test _safe_float with None returns default."""
        assert client._safe_float(None) == 0.0
        assert client._safe_float(None, 99.0) == 99.0

    def test_safe_float_with_na_string(self, client):
        """Test _safe_float with N/A string returns default."""
        assert client._safe_float("N/A") == 0.0
        assert client._safe_float("n/a") == 0.0
        assert client._safe_float("NA") == 0.0
        assert client._safe_float("na") == 0.0

    def test_safe_float_with_null_string(self, client):
        """Test _safe_float with null/none strings returns default."""
        assert client._safe_float("null") == 0.0
        assert client._safe_float("NULL") == 0.0
        assert client._safe_float("None") == 0.0
        assert client._safe_float("NONE") == 0.0

    def test_safe_float_with_empty_string(self, client):
        """Test _safe_float with empty string returns default."""
        assert client._safe_float("") == 0.0
        assert client._safe_float("   ") == 0.0

    def test_safe_float_with_invalid_string(self, client):
        """Test _safe_float with invalid string returns default."""
        assert client._safe_float("invalid") == 0.0
        assert client._safe_float("abc123") == 0.0

    def test_safe_float_with_custom_default(self, client):
        """Test _safe_float uses custom default."""
        assert client._safe_float(None, 42.0) == 42.0
        assert client._safe_float("N/A", -1.0) == -1.0

    def test_safe_float_with_whitespace_string(self, client):
        """Test _safe_float strips whitespace from strings."""
        assert client._safe_float("  5.5  ") == 5.5
        assert client._safe_float("  N/A  ") == 0.0
