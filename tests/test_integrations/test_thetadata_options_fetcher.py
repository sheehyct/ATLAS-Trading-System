"""
ThetaData Options Fetcher Unit Tests.

Session 80: Comprehensive test suite for ThetaDataOptionsFetcher.
Tests caching, fallback logic, price/Greeks retrieval, and edge cases.

Uses MockThetaDataProvider for isolation from ThetaData terminal.
"""

import pytest
import pickle
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
from integrations.thetadata_client import OptionsQuote
from tests.mocks.mock_thetadata import MockThetaDataProvider, create_spy_mock_provider


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_provider():
    """Pre-configured MockThetaDataProvider."""
    return create_spy_mock_provider()


@pytest.fixture
def fetcher_with_mock(mock_provider, tmp_path):
    """ThetaDataOptionsFetcher with mock provider and temp cache dir."""
    fetcher = ThetaDataOptionsFetcher(
        provider=mock_provider,
        cache_dir=str(tmp_path / 'cache'),
        use_cache=True,
        fallback_to_bs=True,
        auto_connect=False,
    )
    # Mock is already connected
    fetcher._connected = True
    return fetcher


@pytest.fixture
def fetcher_no_cache(mock_provider, tmp_path):
    """ThetaDataOptionsFetcher with caching disabled."""
    fetcher = ThetaDataOptionsFetcher(
        provider=mock_provider,
        cache_dir=str(tmp_path / 'cache'),
        use_cache=False,
        fallback_to_bs=True,
        auto_connect=False,
    )
    fetcher._connected = True
    return fetcher


@pytest.fixture
def disconnected_fetcher(tmp_path):
    """Fetcher with disconnected mock provider for fallback testing."""
    mock = MockThetaDataProvider(auto_connect=False)
    mock.set_connected(False)
    fetcher = ThetaDataOptionsFetcher(
        provider=mock,
        cache_dir=str(tmp_path / 'cache'),
        use_cache=True,
        fallback_to_bs=True,
        auto_connect=False,
    )
    fetcher._connected = False
    return fetcher


@pytest.fixture
def sample_option_params(mock_provider):
    """Sample option parameters for testing."""
    exps = mock_provider.get_expirations('SPY')
    return {
        'underlying': 'SPY',
        'expiration': exps[0] if exps else datetime(2024, 12, 20),
        'strike': 450.0,
        'option_type': 'C',
        'as_of': datetime(2024, 11, 15),
        'underlying_price': 450.0,
    }


# =============================================================================
# TestThetaDataOptionsFetcherInitialization
# =============================================================================

class TestThetaDataOptionsFetcherInitialization:
    """Tests for fetcher initialization."""

    def test_init_with_provider(self, mock_provider, tmp_path):
        """Test initialization with custom provider."""
        fetcher = ThetaDataOptionsFetcher(
            provider=mock_provider,
            cache_dir=str(tmp_path),
            auto_connect=False,
        )
        assert fetcher._provider is mock_provider

    def test_init_cache_dir_creation(self, tmp_path):
        """Verify cache directory is created."""
        cache_dir = tmp_path / 'new_cache'
        fetcher = ThetaDataOptionsFetcher(
            provider=None,
            cache_dir=str(cache_dir),
            use_cache=True,
            auto_connect=False,
        )
        assert cache_dir.exists()

    def test_is_available_when_connected(self, fetcher_with_mock):
        """Test is_available property when connected."""
        assert fetcher_with_mock.is_available is True

    def test_is_available_when_disconnected(self, disconnected_fetcher):
        """Test is_available returns False when disconnected."""
        assert disconnected_fetcher.is_available is False


# =============================================================================
# TestThetaDataOptionsFetcherCaching
# =============================================================================

class TestThetaDataOptionsFetcherCaching:
    """Tests for cache management."""

    def test_cache_key_generation(self, fetcher_with_mock):
        """Test cache key format."""
        exp = datetime(2024, 12, 20)
        as_of = datetime(2024, 11, 15)
        key = fetcher_with_mock._get_cache_key('SPY', exp, 450.0, 'C', as_of, 'quote')
        assert 'SPY' in key
        assert '20241220' in key
        assert '450_00' in key
        assert 'C' in key
        assert '20241115' in key
        assert 'quote' in key

    def test_cache_key_includes_all_params(self, fetcher_with_mock):
        """Verify all params in cache key."""
        exp = datetime(2024, 12, 20)
        as_of = datetime(2024, 11, 15)
        key = fetcher_with_mock._get_cache_key('AAPL', exp, 175.5, 'P', as_of, 'greeks')
        assert 'AAPL' in key
        assert '175_50' in key
        assert 'P' in key
        assert 'greeks' in key

    def test_save_and_load_from_cache(self, fetcher_with_mock, tmp_path):
        """Test saving and loading from cache."""
        cache_key = 'test_cache_key.pkl'
        test_data = {'bid': 5.0, 'ask': 5.2, 'mid': 5.1}

        fetcher_with_mock._save_to_cache(cache_key, test_data)
        loaded = fetcher_with_mock._load_from_cache(cache_key)

        assert loaded is not None
        assert loaded['mid'] == 5.1

    def test_cache_expired_returns_none(self, fetcher_with_mock, tmp_path):
        """Test expired cache returns None."""
        cache_dir = fetcher_with_mock.cache_dir
        cache_key = 'expired_cache.pkl'
        cache_path = cache_dir / cache_key

        # Create cache file
        with open(cache_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        # Set modification time to 8 days ago (beyond 7-day TTL)
        eight_days_ago = time.time() - (8 * 24 * 60 * 60)
        os.utime(cache_path, (eight_days_ago, eight_days_ago))

        loaded = fetcher_with_mock._load_from_cache(cache_key)
        assert loaded is None

    def test_cache_valid_within_ttl(self, fetcher_with_mock, tmp_path):
        """Test cache valid when within TTL."""
        cache_key = 'valid_cache.pkl'
        test_data = {'test': 'data'}

        fetcher_with_mock._save_to_cache(cache_key, test_data)
        loaded = fetcher_with_mock._load_from_cache(cache_key)

        assert loaded is not None
        assert loaded == test_data

    def test_cache_disabled(self, fetcher_no_cache, sample_option_params):
        """Test no caching when use_cache=False."""
        params = sample_option_params
        fetcher_no_cache.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Check no cache files created
        cache_files = list(fetcher_no_cache.cache_dir.glob('*.pkl'))
        assert len(cache_files) == 0

    def test_clear_cache_all(self, fetcher_with_mock, sample_option_params):
        """Test clearing all cache files."""
        params = sample_option_params

        # Create some cache
        fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Clear all cache
        cleared = fetcher_with_mock.clear_cache()
        cache_files = list(fetcher_with_mock.cache_dir.glob('*.pkl'))
        assert len(cache_files) == 0

    def test_clear_cache_by_underlying(self, fetcher_with_mock, tmp_path):
        """Test clearing cache for specific underlying."""
        # Create cache for SPY
        cache_key_spy = 'SPY_20241220_450_00_C_20241115_quote.pkl'
        cache_key_aapl = 'AAPL_20241220_175_00_C_20241115_quote.pkl'

        fetcher_with_mock._save_to_cache(cache_key_spy, {'test': 'spy'})
        fetcher_with_mock._save_to_cache(cache_key_aapl, {'test': 'aapl'})

        # Clear only SPY cache
        fetcher_with_mock.clear_cache('SPY')

        # SPY should be gone, AAPL should remain
        assert not (fetcher_with_mock.cache_dir / cache_key_spy).exists()
        assert (fetcher_with_mock.cache_dir / cache_key_aapl).exists()


# =============================================================================
# TestThetaDataOptionsFetcherGetOptionPrice
# =============================================================================

class TestThetaDataOptionsFetcherGetOptionPrice:
    """Tests for get_option_price() method."""

    def test_get_option_price_from_provider(self, fetcher_with_mock, sample_option_params):
        """Test price from ThetaData provider."""
        params = sample_option_params
        price = fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )
        assert price is not None
        assert isinstance(price, float)

    def test_get_option_price_from_cache(self, fetcher_with_mock, sample_option_params, mock_provider):
        """Test price from cache."""
        params = sample_option_params

        # First call - from provider
        fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Reset call count
        initial_count = mock_provider.get_call_count('get_quote')

        # Second call - should be from cache
        price2 = fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Provider should not be called again
        assert mock_provider.get_call_count('get_quote') == initial_count

    def test_get_option_price_caches_result(self, fetcher_with_mock, sample_option_params):
        """Verify result is cached."""
        params = sample_option_params

        fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Check cache file exists
        cache_files = list(fetcher_with_mock.cache_dir.glob('*quote*.pkl'))
        assert len(cache_files) > 0

    def test_get_option_price_returns_mid(self, fetcher_with_mock, sample_option_params):
        """Verify mid price is returned."""
        params = sample_option_params
        price = fetcher_with_mock.get_option_price(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        # Price should be roughly the mid of bid/ask
        assert price is not None
        # Mock adds quotes with bid ~5.0, ask ~5.2 for ATM
        assert 4.0 < price < 8.0  # Reasonable range for SPY ATM option


# =============================================================================
# TestThetaDataOptionsFetcherGetOptionGreeks
# =============================================================================

class TestThetaDataOptionsFetcherGetOptionGreeks:
    """Tests for get_option_greeks() method."""

    def test_get_option_greeks_from_provider(self, fetcher_with_mock, sample_option_params):
        """Test Greeks from ThetaData provider."""
        params = sample_option_params
        greeks = fetcher_with_mock.get_option_greeks(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )
        assert greeks is not None
        assert 'delta' in greeks
        assert 'gamma' in greeks

    def test_get_option_greeks_from_cache(self, fetcher_with_mock, sample_option_params, mock_provider):
        """Test Greeks from cache."""
        params = sample_option_params

        # First call
        fetcher_with_mock.get_option_greeks(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        initial_count = mock_provider.get_call_count('get_greeks')

        # Second call - from cache
        fetcher_with_mock.get_option_greeks(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )

        assert mock_provider.get_call_count('get_greeks') == initial_count

    def test_get_option_greeks_dict_keys(self, fetcher_with_mock, sample_option_params):
        """Verify dict has correct keys."""
        params = sample_option_params
        greeks = fetcher_with_mock.get_option_greeks(
            params['underlying'], params['expiration'], params['strike'],
            params['option_type'], params['as_of'], params['underlying_price']
        )
        expected_keys = {'delta', 'gamma', 'theta', 'vega', 'iv', 'underlying_price'}
        assert expected_keys.issubset(set(greeks.keys()))


# =============================================================================
# TestThetaDataOptionsFetcherBlackScholesFallback
# =============================================================================

class TestThetaDataOptionsFetcherBlackScholesFallback:
    """Tests for Black-Scholes fallback behavior."""

    def test_bs_fallback_when_disconnected(self, disconnected_fetcher):
        """Test fallback when provider unavailable."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        price = disconnected_fetcher.get_option_price(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=450.0
        )

        # Should fall back to Black-Scholes
        assert price is not None
        assert isinstance(price, float)

    def test_bs_fallback_when_data_missing(self, fetcher_with_mock):
        """Test fallback when no quote data."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        # Request strike that doesn't exist in mock
        price = fetcher_with_mock.get_option_price(
            'SPY', exp, 999.0, 'C', as_of, underlying_price=450.0
        )

        # Should fall back to Black-Scholes
        assert price is not None

    def test_bs_fallback_requires_underlying_price(self, disconnected_fetcher):
        """Test None when no underlying_price for fallback."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        price = disconnected_fetcher.get_option_price(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=None
        )

        assert price is None

    def test_bs_greeks_fallback(self, disconnected_fetcher):
        """Test B-S Greeks calculation."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        greeks = disconnected_fetcher.get_option_greeks(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=450.0
        )

        assert greeks is not None
        assert 'delta' in greeks
        # ATM call delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.7


# =============================================================================
# TestThetaDataOptionsFetcherEdgeCases
# =============================================================================

class TestThetaDataOptionsFetcherEdgeCases:
    """Tests for edge cases and error handling."""

    def test_expired_option(self, disconnected_fetcher):
        """Test handling of expired options."""
        exp = datetime.now() - timedelta(days=10)  # Already expired
        as_of = datetime.now()

        price = disconnected_fetcher.get_option_price(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=450.0
        )

        # For expired option, should return intrinsic value
        # ATM option has no intrinsic value
        assert price is not None
        assert price >= 0

    def test_zero_dte_option(self, disconnected_fetcher):
        """Test handling of zero DTE option."""
        exp = datetime.now()  # Same day
        as_of = datetime.now()

        # ITM call (strike < underlying)
        price = disconnected_fetcher.get_option_price(
            'SPY', exp, 440.0, 'C', as_of, underlying_price=450.0
        )

        # Should return intrinsic value
        if price is not None:
            assert price >= 10.0  # At least (450 - 440)

    def test_provider_returns_none(self, fetcher_with_mock):
        """Test None propagation from provider."""
        exp = datetime(2099, 12, 20)  # Far future, no data in mock
        as_of = datetime(2024, 11, 15)

        # Without underlying_price, should return None
        price = fetcher_with_mock.get_option_price(
            'UNKNOWN', exp, 100.0, 'C', as_of, underlying_price=None
        )

        assert price is None


# =============================================================================
# TestThetaDataOptionsFetcherCacheTTL
# =============================================================================

class TestThetaDataOptionsFetcherCacheTTL:
    """Tests for cache TTL validation (Session 80 fix)."""

    def test_cache_ttl_uses_timedelta(self, tmp_path):
        """Test that cache TTL uses timedelta comparison (not .days truncation)."""
        mock = MockThetaDataProvider()
        fetcher = ThetaDataOptionsFetcher(
            provider=mock,
            cache_dir=str(tmp_path),
            cache_ttl_days=7,
            auto_connect=False,
        )

        cache_key = 'test_ttl.pkl'
        cache_path = fetcher.cache_dir / cache_key

        # Create cache file
        with open(cache_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        # Set modification time to 6 days 23 hours ago (should still be valid)
        almost_seven_days = time.time() - (6 * 24 * 60 * 60 + 23 * 60 * 60)
        os.utime(cache_path, (almost_seven_days, almost_seven_days))

        assert fetcher._is_cache_valid(cache_path) is True

    def test_cache_ttl_expired_at_boundary(self, tmp_path):
        """Test cache expired just after TTL."""
        mock = MockThetaDataProvider()
        fetcher = ThetaDataOptionsFetcher(
            provider=mock,
            cache_dir=str(tmp_path),
            cache_ttl_days=7,
            auto_connect=False,
        )

        cache_key = 'test_expired.pkl'
        cache_path = fetcher.cache_dir / cache_key

        # Create cache file
        with open(cache_path, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        # Set modification time to 7 days 1 hour ago (should be expired)
        seven_days_plus = time.time() - (7 * 24 * 60 * 60 + 1 * 60 * 60)
        os.utime(cache_path, (seven_days_plus, seven_days_plus))

        assert fetcher._is_cache_valid(cache_path) is False


# =============================================================================
# TestThetaDataOptionsFetcherGetQuoteWithGreeks
# =============================================================================

class TestThetaDataOptionsFetcherGetQuoteWithGreeks:
    """Tests for get_quote_with_greeks() method."""

    def test_get_quote_with_greeks_fallback_to_bs(self, disconnected_fetcher):
        """Test fallback creates OptionsQuote with Greeks using Black-Scholes."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        quote = disconnected_fetcher.get_quote_with_greeks(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=450.0
        )

        assert quote is not None
        assert quote.delta is not None
        assert quote.gamma is not None
        assert quote.bid > 0
        assert quote.ask > 0

    def test_get_quote_with_greeks_returns_osi_symbol(self, disconnected_fetcher):
        """Test that returned quote has valid OSI symbol."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        quote = disconnected_fetcher.get_quote_with_greeks(
            'SPY', exp, 450.0, 'C', as_of, underlying_price=450.0
        )

        assert quote is not None
        assert 'SPY' in quote.symbol
        assert 'C' in quote.symbol


# =============================================================================
# Session 83: Tests for ATLAS-compliant spread model
# =============================================================================

class TestSpreadModel:
    """Tests for _estimate_spread_pct() added in Session 83."""

    @pytest.fixture
    def fetcher(self, tmp_path):
        """Create fetcher for testing spread model."""
        mock = MockThetaDataProvider()
        return ThetaDataOptionsFetcher(
            provider=mock,
            cache_dir=str(tmp_path),
            auto_connect=False,
        )

    def test_spread_atm_sweet_spot_dte(self, fetcher):
        """Test ATM option with optimal DTE has narrow spread."""
        # ATM (moneyness = 1.0), 14 DTE (sweet spot), $5 option
        spread = fetcher._estimate_spread_pct(
            option_price=5.0,
            underlying_price=100.0,
            strike=100.0,
            dte=14
        )
        # Expected: base 2% + dte 1% + moneyness 0% + price 2% = 5%
        assert 0.03 <= spread <= 0.07

    def test_spread_otm_wider(self, fetcher):
        """Test OTM options have wider spreads."""
        # OTM by 10% (strike 110, underlying 100)
        spread_otm = fetcher._estimate_spread_pct(
            option_price=2.0,
            underlying_price=100.0,
            strike=110.0,
            dte=14
        )
        # OTM distance = 0.10, so moneyness_adj = 0.01 (10% of 0.10)
        # Expected: wider than ATM due to moneyness adjustment
        assert spread_otm > 0.05

    def test_spread_short_dte_wider(self, fetcher):
        """Test near-expiration options have wider spreads."""
        # Same option but 3 DTE vs 14 DTE
        spread_short = fetcher._estimate_spread_pct(
            option_price=5.0,
            underlying_price=100.0,
            strike=100.0,
            dte=3
        )
        spread_normal = fetcher._estimate_spread_pct(
            option_price=5.0,
            underlying_price=100.0,
            strike=100.0,
            dte=14
        )
        # Short DTE should have wider spread
        assert spread_short > spread_normal

    def test_spread_cheap_option_wider(self, fetcher):
        """Test cheap options (<$1) have wider relative spreads."""
        spread_cheap = fetcher._estimate_spread_pct(
            option_price=0.50,
            underlying_price=100.0,
            strike=100.0,
            dte=14
        )
        spread_expensive = fetcher._estimate_spread_pct(
            option_price=10.0,
            underlying_price=100.0,
            strike=100.0,
            dte=14
        )
        # Cheap options should have wider spread
        assert spread_cheap > spread_expensive

    def test_spread_capped_at_20_percent(self, fetcher):
        """Test spread is capped at 20% per ATLAS checklist."""
        # Deep OTM, near expiration, cheap option - maximum spread scenario
        spread = fetcher._estimate_spread_pct(
            option_price=0.10,
            underlying_price=100.0,
            strike=150.0,  # 50% OTM
            dte=1
        )
        # Should not exceed 20%
        assert spread <= 0.20

    def test_spread_applied_to_bid_ask(self, disconnected_fetcher):
        """Test that spread is correctly applied to bid/ask in _create_bs_quote."""
        exp = datetime.now() + timedelta(days=30)
        as_of = datetime.now()

        quote = disconnected_fetcher._create_bs_quote(
            underlying='SPY',
            expiration=exp,
            strike=450.0,
            option_type='C',
            as_of=as_of,
            underlying_price=450.0
        )

        assert quote is not None
        # Bid should be less than mid
        assert quote.bid < quote.mid
        # Ask should be greater than mid
        assert quote.ask > quote.mid
        # Spread should be symmetric around mid
        bid_diff = quote.mid - quote.bid
        ask_diff = quote.ask - quote.mid
        assert abs(bid_diff - ask_diff) < 0.01  # Allow small floating point error
