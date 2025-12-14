"""
Crypto Data Loader for Dashboard - Session CRYPTO-6

Fetches crypto trading data from VPS daemon REST API for dashboard display.
Provides methods matching the dashboard data loader interface pattern.

Configuration:
    Set CRYPTO_API_URL environment variable to point to VPS daemon.
    Default: http://178.156.223.251:8080

Usage:
    from dashboard.data_loaders.crypto_loader import CryptoDataLoader

    loader = CryptoDataLoader()
    if loader._connected:
        positions = loader.get_open_positions()
        signals = loader.get_pending_signals()
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Default VPS API URL
DEFAULT_CRYPTO_API_URL = 'http://178.156.223.251:8080'


class CryptoDataLoader:
    """
    Load crypto trading data from VPS daemon REST API.

    Implements the dashboard data loader interface pattern:
    - _connected flag for connection status
    - init_error for error tracking
    - Methods return List[Dict] or Dict (never raw objects)
    - Graceful degradation with empty defaults
    """

    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the crypto data loader.

        Args:
            api_url: Optional API URL (overrides CRYPTO_API_URL env var)
        """
        self.api_url = api_url or os.getenv('CRYPTO_API_URL', DEFAULT_CRYPTO_API_URL)
        self._connected = False
        self.init_error: Optional[str] = None

        # Test connection on init
        try:
            self._test_connection()
            self._connected = True
            logger.info(f"CryptoDataLoader connected to {self.api_url}")
        except Exception as e:
            self.init_error = str(e)
            logger.warning(f"CryptoDataLoader init error: {e}")

    def _test_connection(self) -> bool:
        """
        Test connection to API.

        Returns:
            True if connection successful

        Raises:
            Exception if connection fails
        """
        response = requests.get(f"{self.api_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('status') != 'ok':
            raise Exception(f"API health check failed: {data}")
        return True

    def _fetch(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch data from API endpoint.

        Args:
            endpoint: API endpoint path (e.g., '/status')
            params: Optional query parameters

        Returns:
            JSON response data or None on error
        """
        url = f"{self.api_url.rstrip('/')}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    # =========================================================================
    # STATUS METHODS
    # =========================================================================

    def get_daemon_status(self) -> Dict:
        """
        Get full daemon status.

        Returns:
            Dict with daemon status including running, uptime, scan counts,
            leverage tier, entry monitor stats, paper trader summary.
            Empty dict on error.
        """
        data = self._fetch('/status')
        return data if data else {}

    def is_daemon_running(self) -> bool:
        """
        Check if daemon is running.

        Returns:
            True if daemon is running and responsive
        """
        status = self.get_daemon_status()
        return status.get('running', False)

    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================

    def get_account_summary(self) -> Dict:
        """
        Get paper trading account summary.

        Returns:
            Dict with account_name, starting_balance, current_balance,
            realized_pnl, return_percent, open_trades, closed_trades.
            Empty dict on error.
        """
        data = self._fetch('/performance')
        if data and 'account_summary' in data:
            return data['account_summary']
        return {}

    def get_performance_metrics(self) -> Dict:
        """
        Get trading performance metrics.

        Returns:
            Dict with total_trades, winning_trades, losing_trades, win_rate,
            total_pnl, gross_profit, gross_loss, profit_factor, avg_win,
            avg_loss, expectancy, largest_win, largest_loss.
            Empty dict on error.
        """
        data = self._fetch('/performance')
        if data and 'performance_metrics' in data:
            return data['performance_metrics']
        return {}

    # =========================================================================
    # POSITION METHODS
    # =========================================================================

    def get_open_positions(self) -> List[Dict]:
        """
        Get open positions with unrealized P&L.

        Returns:
            List of position dicts with trade_id, symbol, side, quantity,
            entry_price, current_price, unrealized_pnl, unrealized_pnl_percent,
            stop_price, target_price, stop_distance_pct, target_distance_pct,
            timeframe, pattern_type, entry_time.
            Empty list on error.
        """
        data = self._fetch('/positions')
        if isinstance(data, list):
            return data
        return []

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.get_open_positions())

    # =========================================================================
    # SIGNAL METHODS
    # =========================================================================

    def get_pending_signals(self) -> List[Dict]:
        """
        Get SETUP signals awaiting trigger.

        Returns:
            List of signal dicts normalized for dashboard display.
            Empty list on error.
        """
        data = self._fetch('/signals')
        if not isinstance(data, list):
            return []

        # Normalize signals for dashboard display
        return [self._normalize_signal(s) for s in data]

    def _normalize_signal(self, signal: Dict) -> Dict:
        """
        Normalize API signal data for dashboard display.

        Adds display-friendly aliases and formats datetime strings.
        """
        # Add display-friendly aliases
        signal['pattern'] = signal.get('pattern_type', '')
        signal['entry'] = signal.get('entry_trigger', 0)
        signal['target'] = signal.get('target_price', 0)
        signal['stop'] = signal.get('stop_price', 0)

        # Format detected_time for display
        detected_time = signal.get('detected_time')
        if detected_time and isinstance(detected_time, str):
            try:
                # Handle ISO format with timezone
                dt_str = detected_time.replace('Z', '+00:00')
                if 'T' in dt_str:
                    dt = datetime.fromisoformat(dt_str)
                    signal['detected_time_display'] = dt.strftime('%Y-%m-%d %H:%M')
                else:
                    signal['detected_time_display'] = detected_time
            except ValueError:
                signal['detected_time_display'] = detected_time
        else:
            signal['detected_time_display'] = str(detected_time) if detected_time else ''

        return signal

    # =========================================================================
    # TRADE HISTORY METHODS
    # =========================================================================

    def get_trade_history(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get trade history.

        Args:
            status: Filter by 'open' or 'closed' (optional)
            limit: Maximum trades to return (default 50)

        Returns:
            List of trade dicts with trade_id, symbol, side, quantity,
            entry_price, exit_price, pnl, pnl_percent, status, etc.
            Empty list on error.
        """
        params = {'limit': limit}
        if status:
            params['status'] = status

        data = self._fetch('/trades', params=params)
        if isinstance(data, list):
            return data
        return []

    def get_open_trades(self) -> List[Dict]:
        """Get open trades only."""
        return self.get_trade_history(status='open')

    def get_closed_trades(self, limit: int = 50) -> List[Dict]:
        """Get closed trades only."""
        return self.get_trade_history(status='closed', limit=limit)

    def get_closed_trades_summary(self, limit: int = 50) -> Dict:
        """
        Get closed trades summary.

        Args:
            limit: Maximum trades to analyze

        Returns:
            Dict with total_pnl, trade_count, win_count, loss_count, win_rate
        """
        trades = self.get_closed_trades(limit=limit)

        if not trades:
            return {
                'total_pnl': 0.0,
                'trade_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
            }

        total_pnl = sum(t.get('pnl', 0) or 0 for t in trades)
        win_count = sum(1 for t in trades if (t.get('pnl') or 0) > 0)
        loss_count = sum(1 for t in trades if (t.get('pnl') or 0) < 0)
        trade_count = len(trades)
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0

        return {
            'total_pnl': total_pnl,
            'trade_count': trade_count,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
        }


# Export for dashboard
__all__ = ['CryptoDataLoader']
