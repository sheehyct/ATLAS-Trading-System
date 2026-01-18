"""
Tests for TiingoDataFetcher - Session EQUITY-71

Comprehensive test coverage for integrations/tiingo_data_fetcher.py including:
- Initialization with/without API key
- Fetch with single and multiple symbols
- Timeframe conversion
- Caching behavior (read, write, bypass)
- Cache update and clearing
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / 'tiingo_cache'
    cache_dir.mkdir(parents=True)
    return str(cache_dir)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame like Tiingo returns."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [101.0, 102.0, 103.0, 104.0, 105.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
    }, index=dates)


@pytest.fixture
def mock_tiingo_client(sample_ohlcv_df):
    """Create mock TiingoClient."""
    client = Mock()
    client.get_dataframe.return_value = sample_ohlcv_df.copy()
    return client


@pytest.fixture
def mock_vbt_data():
    """Create mock VBT Data object."""
    data = Mock()
    data.shape = (5, 5)
    data.index = pd.date_range('2024-01-01', periods=5, freq='D')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return data


# =============================================================================
# TEST INITIALIZATION
# =============================================================================


class TestTiingoDataFetcherInit:
    """Tests for TiingoDataFetcher initialization."""

    def test_init_with_api_key(self, temp_cache_dir):
        """Test initialization with explicit API key."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='default_key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                fetcher = TiingoDataFetcher(api_key='test_key', cache_dir=temp_cache_dir)

                assert fetcher.api_key == 'test_key'
                MockClient.assert_called_once_with({
                    'api_key': 'test_key',
                    'session': True
                })

    def test_init_without_api_key_uses_config(self, temp_cache_dir):
        """Test initialization uses config when no API key provided."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='config_key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)

                assert fetcher.api_key == 'config_key'

    def test_init_creates_cache_directory(self, tmp_path):
        """Test initialization creates cache directory."""
        cache_dir = tmp_path / 'new_cache_dir'

        with patch('integrations.tiingo_data_fetcher.TiingoClient'):
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                TiingoDataFetcher(cache_dir=str(cache_dir))

                assert cache_dir.exists()


# =============================================================================
# TEST FETCH
# =============================================================================


class TestFetch:
    """Tests for TiingoDataFetcher.fetch()."""

    def test_fetch_single_symbol(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetching single symbol."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    mock_data = Mock()
                    mock_vbt.Data.from_data.return_value = mock_data

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    result = fetcher.fetch('SPY', '2024-01-01', '2024-01-05', use_cache=False)

                    mock_client.get_dataframe.assert_called_once_with(
                        'SPY',
                        startDate='2024-01-01',
                        endDate='2024-01-05',
                        frequency='daily'
                    )
                    assert result == mock_data

    def test_fetch_normalizes_symbol_to_list(self, temp_cache_dir, sample_ohlcv_df):
        """Test single symbol string is converted to list."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', use_cache=False)

                    # Should call get_dataframe for 'SPY'
                    mock_client.get_dataframe.assert_called()
                    call_args = mock_client.get_dataframe.call_args
                    assert call_args[0][0] == 'SPY'

    def test_fetch_multiple_symbols(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetching multiple symbols."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch(['SPY', 'QQQ'], '2024-01-01', '2024-01-05', use_cache=False)

                    # Should call get_dataframe twice
                    assert mock_client.get_dataframe.call_count == 2

    def test_fetch_default_end_date(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetch uses today as default end date."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', use_cache=False)

                    call_args = mock_client.get_dataframe.call_args
                    end_date = call_args[1]['endDate']
                    # Should be today's date
                    assert end_date == datetime.now().strftime('%Y-%m-%d')

    def test_fetch_renames_columns(self, temp_cache_dir, sample_ohlcv_df):
        """Test columns are renamed to VBT convention."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', use_cache=False)

                    # Check DataFrame passed to vbt.Data.from_data has renamed columns
                    call_args = mock_vbt.Data.from_data.call_args
                    df = call_args[0][0]
                    assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']


class TestFetchTimeframes:
    """Tests for timeframe conversion in fetch."""

    @pytest.mark.parametrize('input_tf,expected_freq', [
        ('1d', 'daily'),
        ('1D', 'daily'),
        ('daily', 'daily'),
        ('1w', 'weekly'),
        ('1W', 'weekly'),
        ('weekly', 'weekly'),
        ('1m', 'monthly'),
        ('1M', 'monthly'),
        ('monthly', 'monthly'),
        ('1y', 'annually'),
        ('1Y', 'annually'),
        ('annually', 'annually'),
    ])
    def test_timeframe_conversion(self, temp_cache_dir, sample_ohlcv_df, input_tf, expected_freq):
        """Test timeframe to frequency conversion."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', timeframe=input_tf, use_cache=False)

                    call_args = mock_client.get_dataframe.call_args
                    assert call_args[1]['frequency'] == expected_freq

    def test_unknown_timeframe_defaults_to_daily(self, temp_cache_dir, sample_ohlcv_df):
        """Test unknown timeframe defaults to daily."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', timeframe='4h', use_cache=False)

                    call_args = mock_client.get_dataframe.call_args
                    assert call_args[1]['frequency'] == 'daily'


# =============================================================================
# TEST CACHING
# =============================================================================


class TestCaching:
    """Tests for caching behavior."""

    def test_fetch_saves_to_cache(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetch saves data to cache file."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', '2024-01-05', use_cache=False)

                    # Check cache file was created
                    cache_files = list(Path(temp_cache_dir).glob('SPY_*.pkl'))
                    assert len(cache_files) == 1

    def test_fetch_uses_cache_when_available(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetch uses cache when available."""
        # Pre-create cache file
        cache_file = Path(temp_cache_dir) / 'SPY_2024-01-01_2024-01-05_daily.pkl'
        renamed_df = sample_ohlcv_df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })[['Open', 'High', 'Low', 'Close', 'Volume']]
        renamed_df.to_pickle(cache_file)

        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', '2024-01-05', use_cache=True)

                    # Should NOT call API when cache exists
                    mock_client.get_dataframe.assert_not_called()

    def test_fetch_bypasses_cache_when_disabled(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetch bypasses cache when use_cache=False."""
        # Pre-create cache file
        cache_file = Path(temp_cache_dir) / 'SPY_2024-01-01_2024-01-05_daily.pkl'
        sample_ohlcv_df.to_pickle(cache_file)

        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', '2024-01-05', use_cache=False)

                    # Should call API even though cache exists
                    mock_client.get_dataframe.assert_called_once()


# =============================================================================
# TEST UPDATE CACHE
# =============================================================================


class TestUpdateCache:
    """Tests for update_cache method."""

    def test_update_cache_single_symbol(self, temp_cache_dir, sample_ohlcv_df):
        """Test updating cache for single symbol."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.update_cache('SPY')

                    # Should call API with full date range
                    mock_client.get_dataframe.assert_called_once()
                    call_args = mock_client.get_dataframe.call_args
                    assert call_args[0][0] == 'SPY'
                    assert call_args[1]['startDate'] == '1990-01-01'

    def test_update_cache_multiple_symbols(self, temp_cache_dir, sample_ohlcv_df):
        """Test updating cache for multiple symbols."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.update_cache(['SPY', 'QQQ', 'IWM'])

                    # Should call API for each symbol
                    assert mock_client.get_dataframe.call_count == 3

    def test_update_cache_custom_start_date(self, temp_cache_dir, sample_ohlcv_df):
        """Test updating cache with custom start date."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt'):
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.update_cache('SPY', start_date='2020-01-01')

                    call_args = mock_client.get_dataframe.call_args
                    assert call_args[1]['startDate'] == '2020-01-01'


# =============================================================================
# TEST CLEAR CACHE
# =============================================================================


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache_all(self, temp_cache_dir):
        """Test clearing all cache files."""
        # Create some cache files
        (Path(temp_cache_dir) / 'SPY_2024-01-01_2024-01-05_daily.pkl').touch()
        (Path(temp_cache_dir) / 'QQQ_2024-01-01_2024-01-05_daily.pkl').touch()
        (Path(temp_cache_dir) / 'IWM_2024-01-01_2024-01-05_daily.pkl').touch()

        with patch('integrations.tiingo_data_fetcher.TiingoClient'):
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                fetcher.clear_cache()

                # All cache files should be deleted
                cache_files = list(Path(temp_cache_dir).glob('*.pkl'))
                assert len(cache_files) == 0

    def test_clear_cache_specific_symbol(self, temp_cache_dir):
        """Test clearing cache for specific symbol."""
        # Create some cache files
        (Path(temp_cache_dir) / 'SPY_2024-01-01_2024-01-05_daily.pkl').touch()
        (Path(temp_cache_dir) / 'SPY_2024-02-01_2024-02-05_daily.pkl').touch()
        (Path(temp_cache_dir) / 'QQQ_2024-01-01_2024-01-05_daily.pkl').touch()

        with patch('integrations.tiingo_data_fetcher.TiingoClient'):
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                fetcher.clear_cache('SPY')

                # Only QQQ file should remain
                cache_files = list(Path(temp_cache_dir).glob('*.pkl'))
                assert len(cache_files) == 1
                assert 'QQQ' in cache_files[0].name

    def test_clear_cache_no_files(self, temp_cache_dir):
        """Test clearing cache when no files exist."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient'):
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                from integrations.tiingo_data_fetcher import TiingoDataFetcher

                fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                fetcher.clear_cache()  # Should not raise error


# =============================================================================
# TEST VBT DATA OUTPUT
# =============================================================================


class TestVbtDataOutput:
    """Tests for VBT Data object output."""

    def test_returns_vbt_data_object(self, temp_cache_dir, sample_ohlcv_df):
        """Test fetch returns VBT Data object."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    expected_data = Mock()
                    mock_vbt.Data.from_data.return_value = expected_data

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    result = fetcher.fetch('SPY', '2024-01-01', use_cache=False)

                    assert result == expected_data
                    mock_vbt.Data.from_data.assert_called_once()

    def test_single_symbol_no_multiindex(self, temp_cache_dir, sample_ohlcv_df):
        """Test single symbol returns data without MultiIndex."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch('SPY', '2024-01-01', use_cache=False)

                    # DataFrame passed to vbt.Data.from_data should have simple columns
                    call_args = mock_vbt.Data.from_data.call_args
                    df = call_args[0][0]
                    assert not isinstance(df.columns, pd.MultiIndex)

    def test_multiple_symbols_combined(self, temp_cache_dir, sample_ohlcv_df):
        """Test multiple symbols are combined into single DataFrame."""
        with patch('integrations.tiingo_data_fetcher.TiingoClient') as MockClient:
            with patch('integrations.tiingo_data_fetcher.get_tiingo_key', return_value='key'):
                with patch('integrations.tiingo_data_fetcher.vbt') as mock_vbt:
                    mock_client = Mock()
                    mock_client.get_dataframe.return_value = sample_ohlcv_df.copy()
                    MockClient.return_value = mock_client

                    from integrations.tiingo_data_fetcher import TiingoDataFetcher

                    fetcher = TiingoDataFetcher(cache_dir=temp_cache_dir)
                    fetcher.fetch(['SPY', 'QQQ'], '2024-01-01', use_cache=False)

                    # DataFrame should be combined
                    call_args = mock_vbt.Data.from_data.call_args
                    df = call_args[0][0]
                    # Combined DataFrame should have MultiIndex columns
                    assert isinstance(df.columns, pd.MultiIndex)
