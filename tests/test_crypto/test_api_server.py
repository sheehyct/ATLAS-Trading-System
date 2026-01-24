"""
Tests for crypto/api/server.py - Crypto Daemon REST API.

EQUITY-83: Phase 3 test coverage for crypto API server.

Tests cover:
- init_api function
- Health endpoint
- Status endpoint
- Positions endpoint
- Signals endpoint
- Performance endpoint
- Trades endpoint with filtering
"""

import pytest
from unittest.mock import MagicMock, patch
import json

# Import the Flask app and functions
from crypto.api.server import app, init_api, _daemon


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
    daemon.get_status.return_value = {
        'running': True,
        'uptime_seconds': 3600,
        'scan_count': 100
    }
    daemon.get_open_positions.return_value = [
        {'symbol': 'BTC', 'size': 0.1, 'pnl': 50.0},
        {'symbol': 'ETH', 'size': 1.0, 'pnl': -20.0}
    ]
    daemon.get_pending_setups.return_value = []
    daemon.paper_trader = MagicMock()
    daemon.paper_trader.get_account_summary.return_value = {
        'equity': 10000,
        'cash': 5000
    }
    daemon.paper_trader.get_performance_metrics.return_value = {
        'total_pnl': 500,
        'win_rate': 0.65
    }
    daemon.paper_trader.get_trade_history.return_value = [
        {'symbol': 'BTC', 'status': 'CLOSED', 'pnl': 100},
        {'symbol': 'ETH', 'status': 'OPEN', 'pnl': -50}
    ]
    return daemon


# =============================================================================
# init_api Tests
# =============================================================================

class TestInitApi:
    """Tests for init_api function."""

    def test_init_api_sets_daemon(self, mock_daemon):
        """Test init_api sets global daemon reference."""
        import crypto.api.server as server_module

        init_api(mock_daemon)

        assert server_module._daemon is mock_daemon


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_no_daemon(self, client):
        """Test health returns 503 when daemon not initialized."""
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/health')

        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_health_with_daemon(self, client, mock_daemon):
        """Test health returns OK when daemon running."""
        import crypto.api.server as server_module
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
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/status')

        assert response.status_code == 503

    def test_status_with_daemon(self, client, mock_daemon):
        """Test status returns daemon status."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/status')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['running'] is True
        assert data['uptime_seconds'] == 3600


# =============================================================================
# Positions Endpoint Tests
# =============================================================================

class TestPositionsEndpoint:
    """Tests for /positions endpoint."""

    def test_positions_no_daemon(self, client):
        """Test positions returns empty list when no daemon."""
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_positions_with_daemon(self, client, mock_daemon):
        """Test positions returns position list."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2
        assert data[0]['symbol'] == 'BTC'

    def test_positions_handles_exception(self, client, mock_daemon):
        """Test positions handles exceptions gracefully."""
        import crypto.api.server as server_module
        mock_daemon.get_open_positions.side_effect = Exception("API Error")
        server_module._daemon = mock_daemon

        response = client.get('/positions')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []


# =============================================================================
# Signals Endpoint Tests
# =============================================================================

class TestSignalsEndpoint:
    """Tests for /signals endpoint."""

    def test_signals_no_daemon(self, client):
        """Test signals returns empty list when no daemon."""
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_signals_with_dict_signals(self, client, mock_daemon):
        """Test signals handles dict signals."""
        import crypto.api.server as server_module
        mock_daemon.get_pending_setups.return_value = [
            {'symbol': 'BTC', 'pattern_type': '3-1-2'}
        ]
        server_module._daemon = mock_daemon

        response = client.get('/signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['symbol'] == 'BTC'

    def test_signals_with_object_signals(self, client, mock_daemon):
        """Test signals handles object signals with to_dict."""
        import crypto.api.server as server_module

        signal_obj = MagicMock()
        signal_obj.to_dict.return_value = {'symbol': 'ETH', 'pattern_type': '2-1-2'}
        mock_daemon.get_pending_setups.return_value = [signal_obj]
        server_module._daemon = mock_daemon

        response = client.get('/signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data[0]['symbol'] == 'ETH'

    def test_signals_fallback_serialization(self, client, mock_daemon):
        """Test signals fallback for objects without to_dict."""
        import crypto.api.server as server_module

        signal_obj = MagicMock(spec=[])  # No to_dict
        signal_obj.symbol = 'SOL'
        signal_obj.timeframe = '1H'
        signal_obj.pattern_type = '3-2'
        signal_obj.direction = 'CALL'
        signal_obj.entry_trigger = 100.0
        signal_obj.stop_price = 95.0
        signal_obj.target_price = 110.0
        signal_obj.magnitude_pct = 1.5
        signal_obj.risk_reward = 2.0
        signal_obj.signal_type = 'SETUP'
        signal_obj.detected_time = '2025-01-23 10:00:00'

        mock_daemon.get_pending_setups.return_value = [signal_obj]
        server_module._daemon = mock_daemon

        response = client.get('/signals')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data[0]['symbol'] == 'SOL'


# =============================================================================
# Performance Endpoint Tests
# =============================================================================

class TestPerformanceEndpoint:
    """Tests for /performance endpoint."""

    def test_performance_no_daemon(self, client):
        """Test performance returns empty when no daemon."""
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/performance')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['account_summary'] == {}
        assert data['performance_metrics'] == {}

    def test_performance_no_paper_trader(self, client, mock_daemon):
        """Test performance handles missing paper trader."""
        import crypto.api.server as server_module
        mock_daemon.paper_trader = None
        server_module._daemon = mock_daemon

        response = client.get('/performance')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['account_summary'] == {}

    def test_performance_with_data(self, client, mock_daemon):
        """Test performance returns account and metrics."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/performance')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['account_summary']['equity'] == 10000
        assert data['performance_metrics']['win_rate'] == 0.65


# =============================================================================
# Trades Endpoint Tests
# =============================================================================

class TestTradesEndpoint:
    """Tests for /trades endpoint."""

    def test_trades_no_daemon(self, client):
        """Test trades returns empty when no daemon."""
        import crypto.api.server as server_module
        server_module._daemon = None

        response = client.get('/trades')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_trades_all(self, client, mock_daemon):
        """Test trades returns all trades."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/trades')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2

    def test_trades_filter_open(self, client, mock_daemon):
        """Test trades filters by open status."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/trades?status=open')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['status'] == 'OPEN'

    def test_trades_filter_closed(self, client, mock_daemon):
        """Test trades filters by closed status."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/trades?status=closed')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['status'] == 'CLOSED'

    def test_trades_with_limit(self, client, mock_daemon):
        """Test trades respects limit parameter."""
        import crypto.api.server as server_module
        server_module._daemon = mock_daemon

        response = client.get('/trades?limit=10')

        mock_daemon.paper_trader.get_trade_history.assert_called_with(limit=10)
