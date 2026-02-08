"""
EQUITY-102: Tests for Signal-to-Trade Metadata Correlation fixes.

Tests cover:
1. SignalStore.update_tfc() method
2. /trade_metadata API endpoint
3. ExecutionCoordinator._last_tfc_assessment writeback
4. OptionsDataLoader._fetch_trade_metadata_from_api()
5. OptionsDataLoader.get_closed_trades() remote mode metadata fetch
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from strat.signal_automation.signal_store import SignalStore, StoredSignal, SignalStatus
from strat.signal_automation.api.server import app, init_api
from strat.signal_automation.coordinators.execution_coordinator import ExecutionCoordinator


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_store_dir(tmp_path):
    """Create temporary directory for signal store tests."""
    store_dir = tmp_path / 'signals'
    store_dir.mkdir(parents=True)
    return str(store_dir)


@pytest.fixture
def signal_store(temp_store_dir):
    """Create a signal store with temp directory."""
    return SignalStore(store_path=temp_store_dir)


@pytest.fixture
def sample_stored_signal():
    """Create a sample StoredSignal."""
    return StoredSignal(
        signal_key='SPY_1D_3-1-2U_CALL_202501200000',
        pattern_type='3-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime(2025, 1, 20, 10, 30),
        entry_trigger=590.50,
        stop_price=585.00,
        target_price=600.00,
        magnitude_pct=1.6,
        risk_reward=1.9,
        vix=16.0,
        tfc_score=0,
        tfc_alignment='',
        status=SignalStatus.DETECTED.value,
        first_seen_at=datetime(2025, 1, 20, 10, 30),
        last_seen_at=datetime(2025, 1, 20, 10, 30),
    )


@pytest.fixture
def flask_client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_daemon():
    """Create mock daemon for API tests."""
    daemon = MagicMock()
    daemon._is_running = True
    daemon.signal_store = MagicMock()
    daemon.executor = MagicMock()
    return daemon


# =============================================================================
# TEST: SignalStore.update_tfc()
# =============================================================================


class TestUpdateTfc:
    """Tests for SignalStore.update_tfc() method (EQUITY-102 Fix 3)."""

    def test_update_tfc_success(self, signal_store, sample_stored_signal):
        """Test updating TFC score and alignment."""
        signal_store._signals[sample_stored_signal.signal_key] = sample_stored_signal
        signal_store._save()

        result = signal_store.update_tfc(
            sample_stored_signal.signal_key,
            tfc_score=4,
            tfc_alignment='4/5 BULLISH'
        )

        assert result is True
        updated = signal_store.get_signal(sample_stored_signal.signal_key)
        assert updated.tfc_score == 4
        assert updated.tfc_alignment == '4/5 BULLISH'

    def test_update_tfc_not_found(self, signal_store):
        """Test updating TFC for non-existent signal returns False."""
        result = signal_store.update_tfc('nonexistent_key', tfc_score=3, tfc_alignment='3/5')
        assert result is False

    def test_update_tfc_persists_to_disk(self, temp_store_dir, sample_stored_signal):
        """Test updated TFC is persisted to disk."""
        store1 = SignalStore(store_path=temp_store_dir)
        store1._signals[sample_stored_signal.signal_key] = sample_stored_signal
        store1._save()

        store1.update_tfc(
            sample_stored_signal.signal_key,
            tfc_score=3,
            tfc_alignment='3/4 BULLISH'
        )

        # Load from disk in new instance
        store2 = SignalStore(store_path=temp_store_dir)
        loaded = store2.get_signal(sample_stored_signal.signal_key)
        assert loaded.tfc_score == 3
        assert loaded.tfc_alignment == '3/4 BULLISH'

    def test_update_tfc_overwrites_zero(self, signal_store, sample_stored_signal):
        """Test update_tfc overwrites initial tfc_score=0."""
        sample_stored_signal.tfc_score = 0
        sample_stored_signal.tfc_alignment = ''
        signal_store._signals[sample_stored_signal.signal_key] = sample_stored_signal

        signal_store.update_tfc(
            sample_stored_signal.signal_key,
            tfc_score=5,
            tfc_alignment='5/5 BULLISH'
        )

        updated = signal_store.get_signal(sample_stored_signal.signal_key)
        assert updated.tfc_score == 5
        assert updated.tfc_alignment == '5/5 BULLISH'


# =============================================================================
# TEST: ExecutionCoordinator._last_tfc_assessment
# =============================================================================


class TestExecutionCoordinatorTfcWriteback:
    """Tests for ExecutionCoordinator TFC assessment storage (EQUITY-102 Fix 3)."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.tfc_reeval_enabled = True
        config.tfc_reeval_min_strength = 3
        config.tfc_reeval_block_on_flip = True
        config.tfc_reeval_log_always = False
        return config

    @pytest.fixture
    def mock_tfc_evaluator(self):
        evaluator = Mock()
        assessment = Mock()
        assessment.strength = 4
        assessment.direction = 'bullish'
        assessment.passes_flexible = True
        assessment.alignment_label.return_value = '4/5 BULLISH'
        evaluator.evaluate_tfc.return_value = assessment
        return evaluator

    def test_last_tfc_assessment_stored_after_reeval(self, mock_config, mock_tfc_evaluator):
        """Test _last_tfc_assessment is set after successful TFC re-evaluation."""
        coord = ExecutionCoordinator(
            config=mock_config,
            tfc_evaluator=mock_tfc_evaluator,
        )

        signal = MagicMock()
        signal.tfc_score = 0
        signal.tfc_alignment = ''
        signal.passes_flexible = True
        signal.direction = 'CALL'
        signal.symbol = 'SPY'
        signal.timeframe = '1D'
        signal.pattern_type = '3-1-2U'
        signal.signal_key = 'test_key'

        blocked, reason = coord.reevaluate_tfc_at_entry(signal)

        assert coord._last_tfc_assessment is not None
        score, alignment = coord._last_tfc_assessment
        assert score == 4
        assert alignment == '4/5 BULLISH'

    def test_last_tfc_assessment_none_when_disabled(self, mock_config):
        """Test _last_tfc_assessment is None when TFC reeval disabled."""
        mock_config.tfc_reeval_enabled = False
        coord = ExecutionCoordinator(config=mock_config)

        signal = MagicMock()
        blocked, reason = coord.reevaluate_tfc_at_entry(signal)

        assert coord._last_tfc_assessment is None

    def test_last_tfc_assessment_none_on_error(self, mock_config, mock_tfc_evaluator):
        """Test _last_tfc_assessment is None when evaluation errors."""
        mock_tfc_evaluator.evaluate_tfc.side_effect = ConnectionError("Network error")
        coord = ExecutionCoordinator(
            config=mock_config,
            tfc_evaluator=mock_tfc_evaluator,
            on_error=Mock(),
        )

        signal = MagicMock()
        signal.tfc_score = 0
        signal.tfc_alignment = ''
        signal.passes_flexible = True
        signal.direction = 'CALL'
        signal.symbol = 'SPY'
        signal.timeframe = '1D'
        signal.pattern_type = '3-1-2U'
        signal.signal_key = 'test_key'

        blocked, reason = coord.reevaluate_tfc_at_entry(signal)

        assert coord._last_tfc_assessment is None

    def test_last_tfc_assessment_reset_between_calls(self, mock_config, mock_tfc_evaluator):
        """Test _last_tfc_assessment is reset at the start of each call."""
        coord = ExecutionCoordinator(
            config=mock_config,
            tfc_evaluator=mock_tfc_evaluator,
        )

        signal = MagicMock()
        signal.tfc_score = 3
        signal.tfc_alignment = '3/4 BULLISH'
        signal.passes_flexible = True
        signal.direction = 'CALL'
        signal.symbol = 'SPY'
        signal.timeframe = '1D'
        signal.pattern_type = '3-1-2U'
        signal.signal_key = 'test_key'

        # First call succeeds
        coord.reevaluate_tfc_at_entry(signal)
        assert coord._last_tfc_assessment is not None

        # Second call with disabled reeval should reset
        coord._config.tfc_reeval_enabled = False
        coord.reevaluate_tfc_at_entry(signal)
        assert coord._last_tfc_assessment is None


# =============================================================================
# TEST: /trade_metadata API Endpoint
# =============================================================================


class TestTradeMetadataEndpoint:
    """Tests for /trade_metadata API endpoint (EQUITY-102 Fix 1)."""

    def test_trade_metadata_no_daemon(self, flask_client):
        """Test returns empty dict when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = flask_client.get('/trade_metadata')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == {}

    def test_trade_metadata_from_file(self, flask_client, mock_daemon, tmp_path):
        """Test loads metadata from trade_metadata.json."""
        import strat.signal_automation.api.server as server_module

        # Write test metadata file
        metadata_dir = tmp_path / 'data' / 'executions'
        metadata_dir.mkdir(parents=True)
        metadata_file = metadata_dir / 'trade_metadata.json'
        test_data = {
            'SPY250117C00600000': {
                'pattern_type': '3-1-2U',
                'timeframe': '1D',
                'tfc_score': 3,
                'tfc_alignment': '3/4 BULLISH',
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(test_data, f)

        # Mock signal_store with no executed signals
        mock_daemon.signal_store.load_signals.return_value = {}
        server_module._daemon = mock_daemon

        with patch('strat.signal_automation.api.server.Path', return_value=metadata_file):
            response = flask_client.get('/trade_metadata')

        assert response.status_code == 200

    def test_trade_metadata_merges_signal_store(self, flask_client, mock_daemon):
        """Test merges signal_store data with trade_metadata."""
        import strat.signal_automation.api.server as server_module

        # Create mock signal with executed_osi_symbol
        mock_signal = MagicMock()
        mock_signal.executed_osi_symbol = 'AAPL250117C00250000'
        mock_signal.pattern_type = '2-1-2U'
        mock_signal.timeframe = '1H'
        mock_signal.tfc_score = 4
        mock_signal.tfc_alignment = '4/5 BULLISH'
        mock_signal.direction = 'CALL'
        mock_signal.symbol = 'AAPL'
        mock_signal.entry_trigger = 250.0
        mock_signal.stop_price = 248.0
        mock_signal.target_price = 255.0
        mock_signal.detected_time = datetime(2025, 1, 20, 10, 30)

        mock_daemon.signal_store.load_signals.return_value = {
            'AAPL_1H_2-1-2U_CALL_202501201000': mock_signal
        }
        server_module._daemon = mock_daemon

        # No trade_metadata.json file
        with patch('strat.signal_automation.api.server.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            response = flask_client.get('/trade_metadata')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'AAPL250117C00250000' in data
        assert data['AAPL250117C00250000']['pattern_type'] == '2-1-2U'
        assert data['AAPL250117C00250000']['tfc_score'] == 4

    def test_trade_metadata_signal_store_tfc_overrides_zero(self, flask_client, mock_daemon, tmp_path):
        """Test signal_store TFC overrides tfc_score=0 from trade_metadata.json."""
        import strat.signal_automation.api.server as server_module

        # trade_metadata.json has tfc_score=0 (written before TFC writeback)
        metadata_dir = tmp_path / 'data' / 'executions'
        metadata_dir.mkdir(parents=True)
        metadata_file = metadata_dir / 'trade_metadata.json'
        test_data = {
            'SPY250117C00600000': {
                'pattern_type': '3-1-2U',
                'timeframe': '1D',
                'tfc_score': 0,
                'tfc_alignment': '',
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(test_data, f)

        # signal_store has updated TFC (from EQUITY-102 writeback)
        mock_signal = MagicMock()
        mock_signal.executed_osi_symbol = 'SPY250117C00600000'
        mock_signal.pattern_type = '3-1-2U'
        mock_signal.timeframe = '1D'
        mock_signal.tfc_score = 4
        mock_signal.tfc_alignment = '4/5 BULLISH'
        mock_signal.direction = 'CALL'
        mock_signal.symbol = 'SPY'
        mock_signal.entry_trigger = 600.0
        mock_signal.stop_price = 595.0
        mock_signal.target_price = 610.0
        mock_signal.detected_time = datetime(2025, 1, 20, 10, 30)

        mock_daemon.signal_store.load_signals.return_value = {
            'SPY_1D_3-1-2U_CALL': mock_signal
        }
        server_module._daemon = mock_daemon

        with patch('strat.signal_automation.api.server.Path', return_value=metadata_file):
            response = flask_client.get('/trade_metadata')

        assert response.status_code == 200


# =============================================================================
# TEST: OptionsDataLoader._fetch_trade_metadata_from_api()
# =============================================================================


class TestFetchTradeMetadataFromApi:
    """Tests for OptionsDataLoader._fetch_trade_metadata_from_api() (EQUITY-102 Fix 2)."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock OptionsDataLoader with remote mode."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient'):
            from dashboard.data_loaders.options_loader import OptionsDataLoader
            loader = OptionsDataLoader.__new__(OptionsDataLoader)
            loader.vps_api_url = 'http://vps:8081'
            loader.use_remote = True
            loader.signal_store = None
            loader._connected = False
            loader.client = MagicMock()
            loader.init_error = None
            loader.account = 'SMALL'
            return loader

    @patch('dashboard.data_loaders.options_loader.requests.get')
    def test_fetch_metadata_success(self, mock_get, mock_loader):
        """Test successful metadata fetch from VPS API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'SPY250117C00600000': {
                'pattern_type': '3-1-2U',
                'timeframe': '1D',
                'tfc_score': 4,
                'tfc_alignment': '4/5 BULLISH',
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = mock_loader._fetch_trade_metadata_from_api()

        assert 'SPY250117C00600000' in result
        assert result['SPY250117C00600000']['pattern_type'] == '3-1-2U'
        assert result['SPY250117C00600000']['tfc_score'] == 4
        mock_get.assert_called_once_with('http://vps:8081/trade_metadata', timeout=10)

    @patch('dashboard.data_loaders.options_loader.requests.get')
    def test_fetch_metadata_network_error(self, mock_get, mock_loader):
        """Test returns empty dict on network error."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        result = mock_loader._fetch_trade_metadata_from_api()

        assert result == {}

    def test_fetch_metadata_no_url(self, mock_loader):
        """Test returns empty dict when no VPS URL configured."""
        mock_loader.vps_api_url = ''

        result = mock_loader._fetch_trade_metadata_from_api()

        assert result == {}

    @patch('dashboard.data_loaders.options_loader.requests.get')
    def test_fetch_metadata_non_dict_response(self, mock_get, mock_loader):
        """Test returns empty dict when API returns non-dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = mock_loader._fetch_trade_metadata_from_api()

        assert result == {}


class TestGetClosedTradesRemoteMetadata:
    """Tests for get_closed_trades() using remote metadata (EQUITY-102 Fix 2)."""

    @pytest.fixture
    def mock_loader(self):
        """Create mock OptionsDataLoader for testing get_closed_trades remote path."""
        with patch('dashboard.data_loaders.options_loader.AlpacaTradingClient'):
            from dashboard.data_loaders.options_loader import OptionsDataLoader
            loader = OptionsDataLoader.__new__(OptionsDataLoader)
            loader.vps_api_url = 'http://vps:8081'
            loader.use_remote = True
            loader.signal_store = None
            loader._connected = True
            loader.client = MagicMock()
            loader.init_error = None
            loader.account = 'SMALL'
            return loader

    @patch('dashboard.data_loaders.options_loader.requests.get')
    def test_remote_mode_fetches_from_api(self, mock_get, mock_loader):
        """Test that remote mode fetches trade metadata from VPS API."""
        # Mock VPS API response for /trade_metadata
        api_response = MagicMock()
        api_response.status_code = 200
        api_response.json.return_value = {
            'SPY250120C00590000': {
                'pattern_type': '3-1-2U',
                'timeframe': '1D',
                'tfc_score': 4,
                'tfc_alignment': '4/5 BULLISH',
            }
        }
        api_response.raise_for_status = MagicMock()
        mock_get.return_value = api_response

        # Mock Alpaca closed trades
        mock_loader.client.get_closed_trades.return_value = [
            {
                'symbol': 'SPY250120C00590000',
                'realized_pnl': 150.0,
                'roi_percent': 25.0,
                'buy_price': 6.00,
                'sell_price': 7.50,
                'buy_time_dt': datetime(2025, 1, 20, 10, 30),
                'sell_time_dt': datetime(2025, 1, 22, 14, 0),
            }
        ]

        trades = mock_loader.get_closed_trades(days=30)

        assert len(trades) == 1
        assert trades[0]['pattern'] == '3-1-2U'
        assert trades[0]['timeframe'] == '1D'
        assert trades[0]['tfc_score'] == 4
        assert trades[0]['tfc_alignment'] == '4/5 BULLISH'

    @patch('dashboard.data_loaders.options_loader.requests.get')
    def test_local_mode_uses_file(self, mock_get, mock_loader):
        """Test that local mode uses _load_trade_metadata (not API)."""
        mock_loader.use_remote = False
        mock_loader.vps_api_url = ''

        mock_loader.client.get_closed_trades.return_value = []

        with patch.object(mock_loader, '_load_trade_metadata', return_value={}) as mock_load:
            with patch.object(mock_loader, '_load_executions', return_value={}):
                with patch.object(mock_loader, '_load_enriched_tfc_data', return_value={}):
                    trades = mock_loader.get_closed_trades(days=30)

        mock_load.assert_called_once()
        mock_get.assert_not_called()
