"""
Tests for strat/signal_automation/api/server.py - Equity Daemon REST API.

EQUITY-83: Phase 3 test coverage for equity API server.

Tests cover:
- init_api function
- Health endpoint
- Status endpoint
- Positions endpoint
- Positions with signals endpoint
- Signals endpoints (all, triggered, setups, pending, closed, by_category)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
import json

from strat.signal_automation.api.server import app, init_api, _signal_to_dict


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_daemon():
    """Create mock daemon instance."""
    daemon = MagicMock()
    daemon._is_running = True
    daemon._start_time = datetime(2025, 1, 23, 10, 0, 0)
    daemon._scan_count = 100
    daemon._signal_count = 50
    daemon._execution_count = 10
    daemon._error_count = 2

    # Mock executor
    daemon.executor = MagicMock()
    daemon.executor.get_positions.return_value = [
        {'symbol': 'SPY250117C00600000', 'unrealized_plpc': 0.15}
    ]

    # Mock signal_store
    daemon.signal_store = MagicMock()

    return daemon


@pytest.fixture
def mock_signal():
    """Create mock signal object."""
    signal = MagicMock()
    signal.signal_key = 'SPY_1H_3-1-2_20250123'
    signal.symbol = 'SPY'
    signal.timeframe = '1H'
    signal.pattern_type = '3-1-2'
    signal.direction = 'CALL'
    signal.entry_trigger = 600.0
    signal.stop_price = 595.0
    signal.target_price = 610.0
    signal.magnitude_pct = 1.5
    signal.risk_reward = 2.0
    signal.signal_type = 'SETUP'
    signal.status = 'DETECTED'
    signal.detected_time = datetime(2025, 1, 23, 10, 30, 0)
    signal.executed_osi_symbol = None
    signal.tfc_score = 3
    signal.tfc_alignment = '3/4'
    return signal


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestSignalToDict:
    """Tests for _signal_to_dict helper."""

    def test_converts_signal_to_dict(self, mock_signal):
        """Test signal object is converted to dict."""
        result = _signal_to_dict(mock_signal)

        assert result['symbol'] == 'SPY'
        assert result['timeframe'] == '1H'
        assert result['pattern_type'] == '3-1-2'
        assert result['direction'] == 'CALL'
        assert result['entry_trigger'] == 600.0
        assert result['tfc_score'] == 3

    def test_handles_missing_attributes(self):
        """Test handles objects with missing attributes."""
        partial_signal = MagicMock(spec=[])

        result = _signal_to_dict(partial_signal)

        assert result['symbol'] == ''
        assert result['timeframe'] == ''


# =============================================================================
# init_api Tests
# =============================================================================

class TestInitApi:
    """Tests for init_api function."""

    def test_init_api_sets_daemon(self, mock_daemon):
        """Test init_api sets global daemon reference."""
        import strat.signal_automation.api.server as server_module

        init_api(mock_daemon)

        assert server_module._daemon is mock_daemon


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_no_daemon(self, client):
        """Test health returns 503 when daemon not initialized."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/health')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_health_with_daemon(self, client, mock_daemon):
        """Test health returns OK when daemon running."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/health')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
        assert data['daemon_running'] is True


# =============================================================================
# Status Endpoint Tests
# =============================================================================

class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_no_daemon(self, client):
        """Test status returns 503 when daemon not initialized."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/status')

        assert response.status_code == 503

    def test_status_with_daemon(self, client, mock_daemon):
        """Test status returns daemon status."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/status')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['running'] is True
        assert data['scan_count'] == 100
        assert data['signal_count'] == 50


# =============================================================================
# Positions Endpoint Tests
# =============================================================================

class TestPositionsEndpoint:
    """Tests for /positions endpoint."""

    def test_positions_no_daemon(self, client):
        """Test positions returns empty list when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_positions_no_executor(self, client, mock_daemon):
        """Test positions handles missing executor."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.executor = None
        server_module._daemon = mock_daemon

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_positions_with_signal_lookup(self, client, mock_daemon, mock_signal):
        """Test positions includes pattern/timeframe from signal."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_signal_by_osi_symbol.return_value = mock_signal
        server_module._daemon = mock_daemon

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['pattern'] == '3-1-2'
        assert data[0]['timeframe'] == '1H'


# =============================================================================
# Positions With Signals Endpoint Tests
# =============================================================================

class TestPositionsWithSignalsEndpoint:
    """Tests for /positions_with_signals endpoint."""

    def test_positions_with_signals_no_daemon(self, client):
        """Test returns empty when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/positions_with_signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_positions_with_signals_links_data(self, client, mock_daemon, mock_signal):
        """Test positions are linked to signal data."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_signal_by_osi_symbol.return_value = mock_signal
        mock_daemon.executor._trading_client = MagicMock()
        mock_daemon.executor._trading_client.get_stock_quotes.return_value = {
            'SPY': {'mid': 605.0}
        }
        server_module._daemon = mock_daemon

        response = client.get('/positions_with_signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['name'] == 'SPY 3-1-2 1H'
        assert data[0]['entry'] == 600.0
        assert data[0]['target'] == 610.0


# =============================================================================
# Signals Endpoint Tests
# =============================================================================

class TestSignalsEndpoint:
    """Tests for /signals endpoint."""

    def test_signals_no_daemon(self, client):
        """Test signals returns empty when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_signals_default_query(self, client, mock_daemon, mock_signal):
        """Test signals returns recent signals by default."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_recent_signals.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals')

        assert response.status_code == 200
        mock_daemon.signal_store.get_recent_signals.assert_called_with(days=7)

    def test_signals_filter_by_symbol(self, client, mock_daemon, mock_signal):
        """Test signals filters by symbol."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_signals_by_symbol.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals?symbol=SPY')

        mock_daemon.signal_store.get_signals_by_symbol.assert_called_with('SPY')

    def test_signals_filter_by_timeframe(self, client, mock_daemon, mock_signal):
        """Test signals filters by timeframe."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_signals_by_timeframe.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals?timeframe=1H')

        mock_daemon.signal_store.get_signals_by_timeframe.assert_called_with('1H')

    def test_signals_filter_by_status(self, client, mock_daemon, mock_signal):
        """Test signals filters by status."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_recent_signals.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals?status=DETECTED')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1


# =============================================================================
# Triggered Signals Endpoint Tests
# =============================================================================

class TestTriggeredSignalsEndpoint:
    """Tests for /signals/triggered endpoint."""

    def test_triggered_no_daemon(self, client):
        """Test returns empty when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/signals/triggered')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_triggered_returns_signals(self, client, mock_daemon, mock_signal):
        """Test returns triggered signals."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_historical_triggered_signals.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals/triggered')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1


# =============================================================================
# Setup Signals Endpoint Tests
# =============================================================================

class TestSetupSignalsEndpoint:
    """Tests for /signals/setups endpoint."""

    def test_setups_returns_signals(self, client, mock_daemon, mock_signal):
        """Test returns setup signals."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_setup_signals_for_monitoring.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals/setups')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1


# =============================================================================
# Pending Signals Endpoint Tests
# =============================================================================

class TestPendingSignalsEndpoint:
    """Tests for /signals/pending endpoint (alias for setups)."""

    def test_pending_is_alias_for_setups(self, client, mock_daemon, mock_signal):
        """Test pending is alias for setups."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_setup_signals_for_monitoring.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals/pending')

        assert response.status_code == 200
        mock_daemon.signal_store.get_setup_signals_for_monitoring.assert_called()


# =============================================================================
# Closed Signals Endpoint Tests
# =============================================================================

class TestClosedSignalsEndpoint:
    """Tests for /signals/closed endpoint."""

    def test_closed_no_daemon(self, client):
        """Test returns empty when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/signals/closed')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_closed_filters_executed_signals(self, client, mock_daemon, mock_signal):
        """Test filters to executed signals only."""
        import strat.signal_automation.api.server as server_module
        mock_signal.executed_osi_symbol = 'SPY250117C00600000'
        mock_signal.status = 'CONVERTED'
        mock_daemon.signal_store.get_recent_signals.return_value = [mock_signal]
        server_module._daemon = mock_daemon

        response = client.get('/signals/closed')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1

    def test_closed_respects_days_param(self, client, mock_daemon):
        """Test respects days query parameter."""
        import strat.signal_automation.api.server as server_module
        mock_daemon.signal_store.get_recent_signals.return_value = []
        server_module._daemon = mock_daemon

        response = client.get('/signals/closed?days=14')

        mock_daemon.signal_store.get_recent_signals.assert_called_with(days=14)


# =============================================================================
# Signals By Category Endpoint Tests
# =============================================================================

class TestSignalsByCategoryEndpoint:
    """Tests for /signals/by_category endpoint."""

    def test_by_category_no_daemon(self, client):
        """Test returns empty categories when no daemon."""
        import strat.signal_automation.api.server as server_module
        server_module._daemon = None

        response = client.get('/signals/by_category')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['setups'] == []
        assert data['triggered'] == []
        assert data['low_magnitude'] == []

    def test_by_category_categorizes_signals(self, client, mock_daemon):
        """Test categorizes signals correctly."""
        import strat.signal_automation.api.server as server_module

        def create_mock_signal(signal_type, magnitude, status):
            """Create mock signal with all required attributes."""
            s = MagicMock()
            s.signal_type = signal_type
            s.magnitude_pct = magnitude
            s.status = status
            # Add all attributes needed by _signal_to_dict
            s.signal_key = f'TEST_{signal_type}'
            s.symbol = 'SPY'
            s.timeframe = '1H'
            s.pattern_type = '3-1-2'
            s.direction = 'CALL'
            s.entry_trigger = 600.0
            s.stop_price = 595.0
            s.target_price = 610.0
            s.risk_reward = 2.0
            s.detected_time = '2025-01-23 10:00:00'
            s.executed_osi_symbol = None
            s.tfc_score = 3
            s.tfc_alignment = '3/4'
            return s

        setup_signal = create_mock_signal('SETUP', 1.0, 'DETECTED')
        low_mag_signal = create_mock_signal('SETUP', 0.3, 'DETECTED')
        triggered_signal = create_mock_signal('TRIGGERED', 1.0, 'TRIGGERED')

        mock_daemon.signal_store.get_recent_signals.return_value = [
            setup_signal, low_mag_signal, triggered_signal
        ]
        server_module._daemon = mock_daemon

        response = client.get('/signals/by_category')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['setups']) == 1
        assert len(data['low_magnitude']) == 1
        assert len(data['triggered']) == 1
