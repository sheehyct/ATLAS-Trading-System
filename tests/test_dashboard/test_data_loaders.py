"""
Dashboard Data Loader Functional Tests

Tests actual data loading logic with mocked external dependencies.
Goes beyond smoke tests to verify:
- Data normalization and field mapping
- Calculation accuracy (win rates, P&L, summaries)
- Error handling and graceful degradation
- Caching behavior

Session EQUITY-74: Phase 3 Test Coverage for dashboard data loaders.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np


# =============================================================================
# CRYPTO DATA LOADER TESTS
# =============================================================================


class TestCryptoDataLoaderConnection:
    """Test CryptoDataLoader connection handling."""

    def test_init_with_custom_url(self):
        """CryptoDataLoader accepts custom API URL."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            loader = CryptoDataLoader(api_url='http://localhost:9999')

            assert loader.api_url == 'http://localhost:9999'

    def test_init_with_env_var(self):
        """CryptoDataLoader uses CRYPTO_API_URL env var."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch.dict('os.environ', {'CRYPTO_API_URL': 'http://env-url:8080'}):
            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {'status': 'ok'}
                mock_get.return_value.raise_for_status = MagicMock()

                loader = CryptoDataLoader()

                assert loader.api_url == 'http://env-url:8080'

    def test_connection_failure_sets_init_error(self):
        """Failed connection sets init_error."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            loader = CryptoDataLoader(api_url='http://invalid:8080')

            assert loader._connected is False
            assert loader.init_error is not None
            assert "Connection refused" in loader.init_error

    def test_connection_success_sets_connected(self):
        """Successful connection sets _connected flag."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            loader = CryptoDataLoader(api_url='http://test:8080')

            assert loader._connected is True
            assert loader.init_error is None

    def test_health_check_fails_on_bad_status(self):
        """Health check fails if status is not 'ok'."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'error'}
            mock_get.return_value.raise_for_status = MagicMock()

            loader = CryptoDataLoader(api_url='http://test:8080')

            assert loader._connected is False


class TestCryptoDataLoaderFetch:
    """Test CryptoDataLoader data fetching methods."""

    @pytest.fixture
    def connected_loader(self):
        """Create a connected CryptoDataLoader with mocked connection."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            from dashboard.data_loaders.crypto_loader import CryptoDataLoader
            loader = CryptoDataLoader(api_url='http://test:8080')
            yield loader

    def test_get_daemon_status_returns_dict(self, connected_loader):
        """get_daemon_status returns dictionary."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                'running': True,
                'uptime': 3600,
                'scan_count': 100
            }
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_daemon_status()

            assert isinstance(result, dict)
            assert result.get('running') is True
            assert result.get('uptime') == 3600

    def test_get_daemon_status_returns_empty_on_error(self):
        """get_daemon_status returns empty dict on error."""
        import requests as req
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            # First call for health check succeeds
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            loader = CryptoDataLoader(api_url='http://test:8080')

        # Now make the next request fail with RequestException (which is caught)
        with patch('requests.get') as mock_get:
            mock_get.side_effect = req.RequestException("Network error")

            result = loader.get_daemon_status()

            assert result == {}

    def test_is_daemon_running_true(self, connected_loader):
        """is_daemon_running returns True when running."""
        with patch.object(connected_loader, 'get_daemon_status') as mock_status:
            mock_status.return_value = {'running': True}

            assert connected_loader.is_daemon_running() is True

    def test_is_daemon_running_false(self, connected_loader):
        """is_daemon_running returns False when not running."""
        with patch.object(connected_loader, 'get_daemon_status') as mock_status:
            mock_status.return_value = {'running': False}

            assert connected_loader.is_daemon_running() is False

    def test_get_open_positions_returns_list(self, connected_loader):
        """get_open_positions returns list of positions."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [
                {'symbol': 'BTC-USD', 'side': 'BUY', 'quantity': 0.1},
                {'symbol': 'ETH-USD', 'side': 'BUY', 'quantity': 1.0}
            ]
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_open_positions()

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]['symbol'] == 'BTC-USD'

    def test_get_open_positions_returns_empty_when_not_connected(self):
        """get_open_positions returns empty list when not connected."""
        from dashboard.data_loaders.crypto_loader import CryptoDataLoader

        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")

            loader = CryptoDataLoader(api_url='http://invalid:8080')
            result = loader.get_open_positions()

            assert result == []

    def test_get_position_count(self, connected_loader):
        """get_position_count returns correct count."""
        with patch.object(connected_loader, 'get_open_positions') as mock_pos:
            mock_pos.return_value = [{'symbol': 'BTC'}, {'symbol': 'ETH'}]

            assert connected_loader.get_position_count() == 2


class TestCryptoDataLoaderSignals:
    """Test CryptoDataLoader signal methods."""

    @pytest.fixture
    def connected_loader(self):
        """Create a connected CryptoDataLoader."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            from dashboard.data_loaders.crypto_loader import CryptoDataLoader
            return CryptoDataLoader(api_url='http://test:8080')

    def test_normalize_signal_adds_display_fields(self, connected_loader):
        """_normalize_signal adds display-friendly fields."""
        signal = {
            'pattern_type': '3-1-2U',
            'entry_trigger': 50000,
            'target_price': 52000,
            'stop_price': 49000,
            'detected_time': '2024-01-15T10:30:00Z'
        }

        result = connected_loader._normalize_signal(signal)

        assert result['pattern'] == '3-1-2U'
        assert result['entry'] == 50000
        assert result['target'] == 52000
        assert result['stop'] == 49000
        assert 'detected_time_display' in result

    def test_normalize_signal_handles_missing_time(self, connected_loader):
        """_normalize_signal handles missing detected_time."""
        signal = {
            'pattern_type': '2-1-2D',
            'entry_trigger': 100,
        }

        result = connected_loader._normalize_signal(signal)

        assert result['detected_time_display'] == ''


class TestCryptoDataLoaderTrades:
    """Test CryptoDataLoader trade history methods."""

    @pytest.fixture
    def connected_loader(self):
        """Create a connected CryptoDataLoader."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            from dashboard.data_loaders.crypto_loader import CryptoDataLoader
            return CryptoDataLoader(api_url='http://test:8080')

    def test_get_closed_trades_normalizes_fields(self, connected_loader):
        """get_closed_trades normalizes field names for dashboard."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [
                {
                    'pattern_type': '3-2U',
                    'pnl_percent': 5.5,
                    'pnl': 100,
                    'entry': 50000,
                    'exit': 52750
                }
            ]
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_closed_trades(limit=10)

            assert len(result) == 1
            assert result[0]['pattern'] == '3-2U'
            assert result[0]['pnl_pct'] == 5.5
            assert result[0]['entry_price'] == 50000
            assert result[0]['exit_price'] == 52750

    def test_get_closed_trades_handles_none_values(self, connected_loader):
        """get_closed_trades handles None values gracefully."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [
                {
                    'pattern_type': None,
                    'pnl_percent': None,
                    'pnl': None
                }
            ]
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_closed_trades(limit=10)

            assert result[0]['pattern'] == 'Unclassified'
            assert result[0]['pnl_pct'] == 0
            assert result[0]['pnl'] == 0


class TestCryptoDataLoaderSummary:
    """Test CryptoDataLoader summary calculation methods."""

    @pytest.fixture
    def connected_loader(self):
        """Create a connected CryptoDataLoader."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            from dashboard.data_loaders.crypto_loader import CryptoDataLoader
            return CryptoDataLoader(api_url='http://test:8080')

    def test_get_closed_trades_summary_calculates_correctly(self, connected_loader):
        """get_closed_trades_summary calculates win rate correctly."""
        with patch.object(connected_loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = [
                {'pnl': 100},  # win
                {'pnl': 50},   # win
                {'pnl': -30},  # loss
                {'pnl': 80},   # win
                {'pnl': -20},  # loss
            ]

            result = connected_loader.get_closed_trades_summary()

            assert result['trade_count'] == 5
            assert result['win_count'] == 3
            assert result['loss_count'] == 2
            assert result['win_rate'] == 60.0  # 3/5 * 100
            assert result['total_pnl'] == 180  # 100+50-30+80-20

    def test_get_closed_trades_summary_handles_empty(self, connected_loader):
        """get_closed_trades_summary handles empty trades."""
        with patch.object(connected_loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = []

            result = connected_loader.get_closed_trades_summary()

            assert result['trade_count'] == 0
            assert result['win_rate'] == 0.0
            assert result['total_pnl'] == 0.0

    def test_get_closed_trades_summary_handles_all_wins(self, connected_loader):
        """get_closed_trades_summary handles all winning trades."""
        with patch.object(connected_loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = [
                {'pnl': 100},
                {'pnl': 200},
            ]

            result = connected_loader.get_closed_trades_summary()

            assert result['win_rate'] == 100.0
            assert result['loss_count'] == 0

    def test_get_closed_trades_summary_handles_all_losses(self, connected_loader):
        """get_closed_trades_summary handles all losing trades."""
        with patch.object(connected_loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = [
                {'pnl': -100},
                {'pnl': -50},
            ]

            result = connected_loader.get_closed_trades_summary()

            assert result['win_rate'] == 0.0
            assert result['win_count'] == 0


class TestCryptoDataLoaderAccountHistory:
    """Test CryptoDataLoader account history methods."""

    @pytest.fixture
    def connected_loader(self):
        """Create a connected CryptoDataLoader."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'status': 'ok'}
            mock_get.return_value.raise_for_status = MagicMock()

            from dashboard.data_loaders.crypto_loader import CryptoDataLoader
            return CryptoDataLoader(api_url='http://test:8080')

    def test_get_account_history_from_api(self, connected_loader):
        """get_account_history uses API endpoint when available."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [
                {'timestamp': '2024-01-01T00:00:00', 'equity': 10000},
                {'timestamp': '2024-01-02T00:00:00', 'equity': 10100}
            ]
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_account_history(days=90)

            assert len(result) == 2
            assert result[0]['equity'] == 10000
            assert result[1]['equity'] == 10100

    def test_get_account_history_normalizes_field_names(self, connected_loader):
        """get_account_history normalizes various field names."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = [
                {'date': '2024-01-01', 'balance': 10000},  # alternate field names
                {'date': '2024-01-02', 'current_balance': 10100}
            ]
            mock_get.return_value.raise_for_status = MagicMock()

            result = connected_loader.get_account_history(days=90)

            assert result[0]['equity'] == 10000
            assert result[1]['equity'] == 10100


# =============================================================================
# LIVE DATA LOADER TESTS
# =============================================================================


class TestLiveDataLoaderInit:
    """Test LiveDataLoader initialization."""

    def test_init_stores_account(self):
        """LiveDataLoader stores account tier."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='LARGE')

            assert loader.account == 'LARGE'

    def test_init_connection_failure_sets_error(self):
        """LiveDataLoader sets init_error on connection failure."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_client.side_effect = ValueError("Invalid credentials")

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')

            assert loader.init_error is not None
            assert loader.client is None


class TestLiveDataLoaderPositions:
    """Test LiveDataLoader position methods."""

    def test_get_current_positions_returns_dataframe(self):
        """get_current_positions returns pandas DataFrame."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_instance.list_positions.return_value = [
                {'symbol': 'AAPL', 'qty': 10, 'market_value': 1500},
                {'symbol': 'GOOGL', 'qty': 5, 'market_value': 750}
            ]
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')
            result = loader.get_current_positions()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_get_current_positions_returns_empty_when_no_client(self):
        """get_current_positions returns empty DataFrame when client is None."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_client.side_effect = ValueError("No credentials")

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')
            result = loader.get_current_positions()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_get_current_positions_returns_empty_on_no_positions(self):
        """get_current_positions returns empty DataFrame when no positions."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_instance.list_positions.return_value = []
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')
            result = loader.get_current_positions()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestLiveDataLoaderAccount:
    """Test LiveDataLoader account methods."""

    def test_get_account_status_returns_dict(self):
        """get_account_status returns dictionary."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_instance.get_account.return_value = {
                'equity': 10000,
                'cash': 5000,
                'buying_power': 20000
            }
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')
            result = loader.get_account_status()

            assert isinstance(result, dict)
            assert result['equity'] == 10000

    def test_get_account_status_returns_empty_when_no_client(self):
        """get_account_status returns empty dict when client is None."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_client.side_effect = ValueError("No credentials")

            from dashboard.data_loaders.live_loader import LiveDataLoader
            loader = LiveDataLoader(account='TEST')
            result = loader.get_account_status()

            assert result == {}


class TestLiveDataLoaderMarket:
    """Test LiveDataLoader market status methods."""

    def test_get_market_status_returns_dict(self):
        """get_market_status returns dictionary with is_open."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('pandas_market_calendars.get_calendar') as mock_cal:
                mock_calendar = MagicMock()
                mock_calendar.schedule.return_value = pd.DataFrame({'market_open': [1]})
                mock_cal.return_value = mock_calendar

                from dashboard.data_loaders.live_loader import LiveDataLoader
                loader = LiveDataLoader(account='TEST')
                result = loader.get_market_status()

                assert 'is_open' in result
                assert 'timestamp' in result


# =============================================================================
# OPTIONS DATA LOADER TESTS
# =============================================================================


class TestOptionsDataLoaderInit:
    """Test OptionsDataLoader initialization."""

    def test_init_with_vps_api_url(self):
        """OptionsDataLoader uses VPS API when URL provided."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.options_loader import OptionsDataLoader
            loader = OptionsDataLoader(vps_api_url='http://vps:8080')

            assert loader.use_remote is True
            assert loader.vps_api_url == 'http://vps:8080'

    def test_init_without_vps_api_uses_local(self):
        """OptionsDataLoader uses local SignalStore without VPS URL."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                loader = OptionsDataLoader(vps_api_url='')

                assert loader.use_remote is False


class TestOptionsDataLoaderOCCParsing:
    """Test OptionsDataLoader OCC symbol parsing."""

    def test_parse_occ_symbol_standard(self):
        """_parse_occ_symbol parses standard OCC format."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                loader = OptionsDataLoader(vps_api_url='')

                result = loader._parse_occ_symbol('SPY241220C00600000')

                assert 'SPY' in result
                assert '12/20' in result
                assert '$600' in result
                assert 'C' in result

    def test_parse_occ_symbol_put(self):
        """_parse_occ_symbol handles PUT options."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                loader = OptionsDataLoader(vps_api_url='')

                result = loader._parse_occ_symbol('AAPL250117P00200000')

                assert 'AAPL' in result
                assert 'P' in result
                assert '$200' in result

    def test_parse_occ_symbol_returns_original_on_invalid(self):
        """_parse_occ_symbol returns original on invalid input."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                loader = OptionsDataLoader(vps_api_url='')

                result = loader._parse_occ_symbol('INVALID')

                assert result == 'INVALID'

    def test_parse_occ_symbol_handles_empty(self):
        """_parse_occ_symbol handles empty string."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                loader = OptionsDataLoader(vps_api_url='')

                result = loader._parse_occ_symbol('')

                assert result == ''


class TestOptionsDataLoaderSummary:
    """Test OptionsDataLoader summary calculation methods."""

    @pytest.fixture
    def loader(self):
        """Create OptionsDataLoader with mocked dependencies."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            with patch('dashboard.data_loaders.options_loader._get_signal_store_classes') as mock_store:
                mock_store.return_value = (MagicMock(), MagicMock())

                from dashboard.data_loaders.options_loader import OptionsDataLoader
                return OptionsDataLoader(vps_api_url='')

    def test_get_closed_trades_summary_calculates_correctly(self, loader):
        """get_closed_trades_summary calculates statistics correctly."""
        with patch.object(loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = [
                {'realized_pnl': 100},  # win
                {'realized_pnl': 200},  # win
                {'realized_pnl': -50},  # loss
                {'realized_pnl': -100}, # loss
            ]

            result = loader.get_closed_trades_summary(days=30)

            assert result['total_trades'] == 4
            assert result['win_count'] == 2
            assert result['loss_count'] == 2
            assert result['win_rate'] == 50.0
            assert result['total_pnl'] == 150  # 100+200-50-100
            assert result['avg_pnl'] == 37.5   # 150/4
            assert result['avg_win'] == 150.0  # (100+200)/2
            assert result['avg_loss'] == -75.0 # (-50-100)/2
            assert result['largest_win'] == 200
            assert result['largest_loss'] == -100

    def test_get_closed_trades_summary_handles_empty(self, loader):
        """get_closed_trades_summary handles empty trades."""
        with patch.object(loader, 'get_closed_trades') as mock_trades:
            mock_trades.return_value = []

            result = loader.get_closed_trades_summary(days=30)

            assert result['total_trades'] == 0
            assert result['win_rate'] == 0.0
            assert result['total_pnl'] == 0.0
            assert result['avg_win'] == 0.0
            assert result['avg_loss'] == 0.0


class TestOptionsDataLoaderAPIFetch:
    """Test OptionsDataLoader API fetching methods."""

    @pytest.fixture
    def remote_loader(self):
        """Create OptionsDataLoader in remote mode."""
        with patch('integrations.alpaca_trading_client.AlpacaTradingClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.connect.return_value = True
            mock_client.return_value = mock_instance

            from dashboard.data_loaders.options_loader import OptionsDataLoader
            yield OptionsDataLoader(vps_api_url='http://vps:8080')

    def test_normalize_api_signal_maps_fields(self, remote_loader):
        """_normalize_api_signal maps API field names to dashboard names."""
        signal_data = {
            'pattern_type': '3-1-2U',
            'target_price': 150.0,
            'stop_price': 140.0,
            'detected_time': '2024-01-15T10:30:00'
        }

        result = remote_loader._normalize_api_signal(signal_data)

        assert result['pattern'] == '3-1-2U'
        assert result['target'] == 150.0
        assert result['stop'] == 140.0
        assert result['detected_time'] == '2024-01-15 10:30'

    def test_fetch_from_api_returns_empty_on_error(self, remote_loader):
        """_fetch_from_api returns empty list on request error."""
        import requests as req
        with patch('requests.get') as mock_get:
            mock_get.side_effect = req.RequestException("Network error")

            result = remote_loader._fetch_from_api('/signals/active')

            assert result == []


# =============================================================================
# REGIME DATA LOADER TESTS
# =============================================================================


class TestRegimeDataLoaderInit:
    """Test RegimeDataLoader initialization."""

    def test_init_creates_atlas_model(self):
        """RegimeDataLoader initializes ATLAS model."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader.atlas_model is not None

    def test_init_handles_model_failure(self):
        """RegimeDataLoader handles ATLAS model initialization failure."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.side_effect = Exception("Model init failed")

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader.atlas_model is None


class TestRegimeDataLoaderCaching:
    """Test RegimeDataLoader caching behavior."""

    def test_get_current_regime_uses_cache(self):
        """get_current_regime returns cached value for same day."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            # Manually set cache
            today = date.today()
            loader._current_regime_cache['current_regime'] = (
                today, 'TREND_BULL', datetime.now()
            )

            result = loader.get_current_regime()

            assert result['regime'] == 'TREND_BULL'
            assert result['cached'] is True

    def test_clear_cache_empties_all_caches(self):
        """clear_cache removes all cached data."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            # Add some cached data
            loader._current_regime_cache['test'] = ('data',)
            loader._daily_cache['test'] = ('data',)

            loader.clear_cache()

            assert len(loader._current_regime_cache) == 0
            assert len(loader._daily_cache) == 0


class TestRegimeDataLoaderAllocation:
    """Test RegimeDataLoader allocation percentage calculation."""

    def test_get_allocation_pct_trend_bull(self):
        """_get_allocation_pct returns 100% for TREND_BULL."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader._get_allocation_pct('TREND_BULL') == 100

    def test_get_allocation_pct_trend_neutral(self):
        """_get_allocation_pct returns 70% for TREND_NEUTRAL."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader._get_allocation_pct('TREND_NEUTRAL') == 70

    def test_get_allocation_pct_trend_bear(self):
        """_get_allocation_pct returns 30% for TREND_BEAR."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader._get_allocation_pct('TREND_BEAR') == 30

    def test_get_allocation_pct_crash(self):
        """_get_allocation_pct returns 0% for CRASH."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader._get_allocation_pct('CRASH') == 0

    def test_get_allocation_pct_unknown(self):
        """_get_allocation_pct returns 0% for unknown regime."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            assert loader._get_allocation_pct('UNKNOWN_REGIME') == 0


class TestRegimeDataLoaderErrorHandling:
    """Test RegimeDataLoader error handling."""

    def test_error_response_format(self):
        """_error_response returns properly formatted dict."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            from dashboard.data_loaders.regime_loader import RegimeDataLoader
            loader = RegimeDataLoader()

            result = loader._error_response("Test error message")

            assert result['regime'] == 'UNKNOWN'
            assert result['error'] == "Test error message"
            assert result['allocation_pct'] == 0
            assert result['cached'] is False

    def test_get_vix_status_returns_defaults_on_error(self):
        """get_vix_status returns default dict on error."""
        with patch('regime.academic_jump_model.AcademicJumpModel') as mock_model:
            mock_model.return_value = MagicMock()

            with patch('regime.vix_spike_detector.VIXSpikeDetector') as mock_vix:
                mock_vix.side_effect = Exception("VIX fetch failed")

                from dashboard.data_loaders.regime_loader import RegimeDataLoader
                loader = RegimeDataLoader()

                result = loader.get_vix_status()

                assert result['vix_current'] == 0.0
                assert result['is_crash'] is False
                assert result['triggers'] == []


# =============================================================================
# BACKTEST DATA LOADER TESTS
# =============================================================================


class TestBacktestDataLoaderInit:
    """Test BacktestDataLoader initialization."""

    def test_init_sets_default_results_dir(self):
        """BacktestDataLoader uses default results directory."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()

        assert loader.results_dir is not None
        assert 'results' in str(loader.results_dir)

    def test_init_accepts_custom_results_dir(self):
        """BacktestDataLoader accepts custom results directory."""
        from pathlib import Path
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader

        custom_dir = Path('/custom/results')
        loader = BacktestDataLoader(results_dir=custom_dir)

        assert loader.results_dir == custom_dir


class TestBacktestDataLoaderMethods:
    """Test BacktestDataLoader data methods."""

    def test_load_backtest_returns_true(self):
        """load_backtest returns True on success."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()

        result = loader.load_backtest('orb')

        assert result is True
        assert loader.current_strategy == 'orb'
        assert loader.portfolio is not None

    def test_get_equity_curve_returns_series(self):
        """get_equity_curve returns pandas Series."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()
        loader.load_backtest('test')

        result = loader.get_equity_curve()

        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_get_equity_curve_returns_empty_when_no_portfolio(self):
        """get_equity_curve returns empty Series when no portfolio loaded."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()

        result = loader.get_equity_curve()

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_get_trades_returns_dataframe(self):
        """get_trades returns pandas DataFrame."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()
        loader.load_backtest('test')

        result = loader.get_trades()

        assert isinstance(result, pd.DataFrame)
        assert 'entry_date' in result.columns
        assert 'pnl' in result.columns

    def test_get_trades_returns_empty_when_no_portfolio(self):
        """get_trades returns empty DataFrame when no portfolio loaded."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()

        result = loader.get_trades()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_performance_metrics_returns_dict(self):
        """get_performance_metrics returns dictionary."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()
        loader.load_backtest('test')

        result = loader.get_performance_metrics()

        assert isinstance(result, dict)
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result

    def test_get_performance_metrics_returns_empty_when_no_portfolio(self):
        """get_performance_metrics returns empty dict when no portfolio."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()

        result = loader.get_performance_metrics()

        assert result == {}

    def test_get_returns_returns_series(self):
        """get_returns returns pandas Series."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()
        loader.load_backtest('test')

        result = loader.get_returns()

        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_clear_cache_resets_state(self):
        """clear_cache resets portfolio and strategy."""
        from dashboard.data_loaders.backtest_loader import BacktestDataLoader
        loader = BacktestDataLoader()
        loader.load_backtest('test')

        loader.clear_cache()

        assert loader.portfolio is None
        assert loader.current_strategy is None


# =============================================================================
# YF DATA FETCH HELPER TESTS
# =============================================================================


class TestYFDataFetch:
    """Test yfinance data fetching helper."""

    def test_fetch_yf_data_returns_dataframe(self):
        """_fetch_yf_data returns DataFrame with OHLCV columns."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = pd.DataFrame({
                'Open': [100, 101],
                'High': [105, 106],
                'Low': [99, 100],
                'Close': [104, 105],
                'Volume': [1000000, 1100000]
            }, index=pd.DatetimeIndex(['2024-01-01', '2024-01-02']))
            mock_ticker.return_value = mock_ticker_instance

            from dashboard.data_loaders.regime_loader import _fetch_yf_data
            result = _fetch_yf_data('SPY', '2024-01-01', '2024-01-03')

            assert isinstance(result, pd.DataFrame)
            assert 'Close' in result.columns

    def test_fetch_yf_data_returns_empty_on_no_data(self):
        """_fetch_yf_data returns empty DataFrame when no data."""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            from dashboard.data_loaders.regime_loader import _fetch_yf_data
            result = _fetch_yf_data('INVALID', '2024-01-01', '2024-01-03')

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
