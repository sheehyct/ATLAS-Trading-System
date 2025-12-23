"""
Options Data Loader for Dashboard

Fetches live options signals and positions for the options trading panel.

Data Sources:
- Signal Store: Pending/active STRAT signals from signal_automation
- Alpaca API: Live option positions via AlpacaTradingClient

Session 83K-75: Initial implementation for dashboard integration.
Session 83K-76: Added VPS API support for Railway deployment.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import requests

from integrations.alpaca_trading_client import AlpacaTradingClient

logger = logging.getLogger(__name__)

# VPS Signal API URL - set this for Railway deployments
VPS_SIGNAL_API_URL = os.getenv('VPS_SIGNAL_API_URL', '')

# Lazy import SignalStore only when needed (avoids numba dependency on Railway)
SignalStore = None
StoredSignal = None


def _get_signal_store_classes():
    """Lazy import SignalStore to avoid numba dependency on Railway."""
    global SignalStore, StoredSignal
    if SignalStore is None:
        from strat.signal_automation.signal_store import SignalStore as SS
        from strat.signal_automation.signal_store import StoredSignal as SSig
        SignalStore = SS
        StoredSignal = SSig
    return SignalStore, StoredSignal


class OptionsDataLoader:
    """Load options signals and positions for dashboard."""

    def __init__(self, account: str = 'SMALL', vps_api_url: str = None):
        """
        Initialize the options data loader.

        Args:
            account: Alpaca account tier ('SMALL', 'MID', 'LARGE')
            vps_api_url: Optional VPS signal API URL (overrides env var)
        """
        self.account = account
        self.client = AlpacaTradingClient(account=account)
        self._connected = False
        self.init_error = None

        # VPS API URL for remote signal fetching (Railway deployment)
        self.vps_api_url = vps_api_url or VPS_SIGNAL_API_URL
        self.use_remote = bool(self.vps_api_url)

        # Only create local signal store if not using remote API
        if self.use_remote:
            logger.info(f"OptionsDataLoader using VPS API: {self.vps_api_url}")
            self.signal_store = None
        else:
            logger.info("OptionsDataLoader using local SignalStore")
            SignalStoreClass, _ = _get_signal_store_classes()
            self.signal_store = SignalStoreClass()

        # Try to connect to Alpaca on init
        try:
            self._connected = self.client.connect()
            if self._connected:
                logger.info(f"OptionsDataLoader connected to Alpaca {account} account")
            else:
                self.init_error = "Alpaca connection returned False"
                logger.warning("OptionsDataLoader: Alpaca connection failed")
        except Exception as e:
            self.init_error = str(e)
            logger.warning(f"OptionsDataLoader init error: {e}")

    def connect(self) -> bool:
        """Connect to Alpaca if not already connected."""
        if not self._connected:
            try:
                self._connected = self.client.connect()
            except Exception as e:
                logger.error(f"OptionsDataLoader connect error: {e}")
                self._connected = False
        return self._connected

    def _fetch_from_api(self, endpoint: str) -> List[Dict]:
        """
        Fetch signals from VPS API.

        Args:
            endpoint: API endpoint path (e.g., '/signals/active')

        Returns:
            List of signal dictionaries from API
        """
        if not self.vps_api_url:
            return []

        url = f"{self.vps_api_url.rstrip('/')}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # API returns list directly for these endpoints
            if isinstance(data, list):
                return [self._normalize_api_signal(s) for s in data]
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching from VPS API {url}: {e}")
            return []

    def _normalize_api_signal(self, signal_data: Dict) -> Dict:
        """
        Normalize API signal data to dashboard format.

        Maps API field names to dashboard expected names and
        converts datetime strings to formatted display strings.
        """
        # Map API field names to dashboard expected names
        signal_data['pattern'] = signal_data.get('pattern_type', '')
        signal_data['target'] = signal_data.get('target_price', 0)
        signal_data['stop'] = signal_data.get('stop_price', 0)

        # Parse and format detected_time if present
        detected_time = signal_data.get('detected_time')
        if detected_time and isinstance(detected_time, str):
            try:
                dt = datetime.fromisoformat(detected_time)
                signal_data['detected_time'] = dt.strftime('%Y-%m-%d %H:%M')
            except ValueError:
                pass

        # Parse and format triggered_at if present
        triggered_at = signal_data.get('triggered_at')
        if triggered_at and isinstance(triggered_at, str):
            try:
                dt = datetime.fromisoformat(triggered_at)
                signal_data['triggered_at'] = dt.strftime('%Y-%m-%d %H:%M')
            except ValueError:
                pass

        return signal_data

    def get_pending_signals(self) -> List[Dict]:
        """
        Get SETUP signals awaiting entry trigger.

        Returns:
            List of signal dictionaries ready for monitoring
        """
        try:
            if self.use_remote:
                return self._fetch_from_api('/signals/pending')
            signals = self.signal_store.get_setup_signals_for_monitoring()
            return [self._signal_to_dict(s) for s in signals]
        except Exception as e:
            logger.error(f"Error getting pending signals: {e}")
            return []

    def get_active_signals(self) -> List[Dict]:
        """
        Get all non-expired signals.

        Returns:
            List of signal dictionaries (all statuses except EXPIRED/CONVERTED)
        """
        try:
            if self.use_remote:
                return self._fetch_from_api('/signals/active')
            all_signals = self.signal_store.load_signals()
            active = [
                s for s in all_signals.values()
                if s.status not in ('EXPIRED', 'CONVERTED')
            ]
            # Sort by detected_time descending (newest first)
            active.sort(key=lambda s: s.detected_time or datetime.min, reverse=True)
            return [self._signal_to_dict(s) for s in active]
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []

    def get_triggered_signals(self) -> List[Dict]:
        """
        Get signals that have been triggered (entry hit).

        Returns:
            List of triggered signal dictionaries
        """
        try:
            if self.use_remote:
                return self._fetch_from_api('/signals/triggered')
            all_signals = self.signal_store.load_signals()
            triggered = [
                s for s in all_signals.values()
                if s.status in ('TRIGGERED', 'HISTORICAL_TRIGGERED')
            ]
            # Sort by triggered_at descending
            triggered.sort(key=lambda s: s.triggered_at or datetime.min, reverse=True)
            return [self._signal_to_dict(s) for s in triggered]
        except Exception as e:
            logger.error(f"Error getting triggered signals: {e}")
            return []

    def get_setup_signals(self, min_magnitude: float = 0.5) -> List[Dict]:
        """
        Get SETUP signals awaiting entry trigger that meet magnitude threshold.

        Args:
            min_magnitude: Minimum magnitude % to include (default 0.5%)

        Returns:
            List of setup signal dictionaries meeting criteria
        """
        try:
            if self.use_remote:
                # Get pending signals from API and filter locally
                signals = self._fetch_from_api('/signals/pending')
                return [s for s in signals if s.get('magnitude_pct', 0) >= min_magnitude]
            signals = self.signal_store.get_setup_signals_for_monitoring()
            filtered = [s for s in signals if s.magnitude_pct >= min_magnitude]
            filtered.sort(key=lambda s: s.detected_time or datetime.min, reverse=True)
            return [self._signal_to_dict(s) for s in filtered]
        except Exception as e:
            logger.error(f"Error getting setup signals: {e}")
            return []

    def get_low_magnitude_signals(self, max_magnitude: float = 0.5) -> List[Dict]:
        """
        Get signals that don't meet magnitude threshold (for review).

        Args:
            max_magnitude: Maximum magnitude % to include (signals below this)

        Returns:
            List of low magnitude signal dictionaries
        """
        try:
            if self.use_remote:
                # Get all active signals and filter locally
                signals = self._fetch_from_api('/signals/active')
                return [s for s in signals
                        if s.get('magnitude_pct', 0) < max_magnitude
                        and s.get('status') not in ('EXPIRED', 'CONVERTED')]
            all_signals = self.signal_store.load_signals()
            low_mag = [
                s for s in all_signals.values()
                if s.magnitude_pct < max_magnitude
                and s.status not in ('EXPIRED', 'CONVERTED')
            ]
            low_mag.sort(key=lambda s: s.detected_time or datetime.min, reverse=True)
            return [self._signal_to_dict(s) for s in low_mag]
        except Exception as e:
            logger.error(f"Error getting low magnitude signals: {e}")
            return []

    def get_signals_by_category(self) -> Dict[str, List[Dict]]:
        """
        Get all signals organized by category for tabbed display.

        Returns:
            Dictionary with keys: 'setups', 'triggered', 'low_magnitude'
        """
        return {
            'setups': self.get_setup_signals(),
            'triggered': self.get_triggered_signals(),
            'low_magnitude': self.get_low_magnitude_signals()
        }

    def get_option_positions(self) -> List[Dict]:
        """
        Get live option positions from Alpaca.

        Returns:
            List of position dictionaries with P&L data
        """
        if not self._connected:
            self.connect()
        if not self._connected:
            logger.warning("Cannot get positions: not connected to Alpaca")
            return []

        try:
            positions = self.client.list_option_positions()
            # Enhance positions with calculated fields
            for pos in positions:
                # Calculate P&L percentage if not present
                if 'unrealized_plpc' not in pos and pos.get('cost_basis', 0) > 0:
                    pos['unrealized_plpc'] = (
                        pos.get('unrealized_pl', 0) / pos['cost_basis'] * 100
                    )
                # Parse OCC symbol for display
                pos['display_contract'] = self._parse_occ_symbol(pos.get('symbol', ''))
            return positions
        except Exception as e:
            logger.error(f"Error getting option positions: {e}")
            return []

    def get_account_summary(self) -> Dict:
        """
        Get account summary for options panel.

        Returns:
            Dictionary with equity, buying_power, etc.
        """
        if not self._connected:
            self.connect()
        if not self._connected:
            return {'error': 'Not connected'}

        try:
            account = self.client.get_account()
            return {
                'equity': float(account.get('equity', 0)),
                'buying_power': float(account.get('buying_power', 0)),
                'cash': float(account.get('cash', 0)),
                'portfolio_value': float(account.get('portfolio_value', 0)),
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {'error': str(e)}

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        """
        Get closed option trades with realized P&L using FIFO matching.

        Fetches fill activities from Alpaca and matches sells to buys
        to calculate realized P&L for closed positions.

        Also correlates trades with originating STRAT signals to include
        pattern information.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of closed trade dictionaries with realized P&L and pattern info
        """
        if not self._connected:
            self.connect()
        if not self._connected:
            logger.warning("Cannot get closed trades: not connected to Alpaca")
            return []

        try:
            from datetime import timedelta
            after = datetime.now() - timedelta(days=days)

            # Get closed trades from Alpaca client
            closed_trades = self.client.get_closed_trades(
                after=after,
                options_only=True
            )

            # Load executions to link patterns
            executions = self._load_executions()

            # Enhance with display-friendly fields
            for trade in closed_trades:
                osi_symbol = trade.get('symbol', '')

                # Parse OCC symbol for display
                trade['display_contract'] = self._parse_occ_symbol(osi_symbol)

                # Look up pattern from signal store
                trade['pattern'] = '-'
                trade['timeframe'] = '-'
                if self.signal_store and osi_symbol:
                    signal = self.signal_store.get_signal_by_osi_symbol(osi_symbol)
                    if signal:
                        trade['pattern'] = signal.pattern_type
                        trade['timeframe'] = signal.timeframe

                # Format timestamps for display
                if trade.get('buy_time_dt'):
                    trade['buy_time_display'] = trade['buy_time_dt'].strftime(
                        '%m/%d %H:%M'
                    )
                else:
                    trade['buy_time_display'] = ''

                if trade.get('sell_time_dt'):
                    trade['sell_time_display'] = trade['sell_time_dt'].strftime(
                        '%m/%d %H:%M'
                    )
                else:
                    trade['sell_time_display'] = ''

                # Link pattern from execution data
                trade['pattern'] = self._get_pattern_for_trade(
                    trade.get('symbol', ''),
                    executions
                )

            return closed_trades

        except Exception as e:
            logger.error(f"Error getting closed trades: {e}")
            return []

    def _load_executions(self) -> Dict:
        """Load executions from disk to link patterns to trades."""
        try:
            import json
            executions_file = Path('data/executions/executions.json')
            if executions_file.exists():
                with open(executions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load executions: {e}")
        return {}

    def _get_pattern_for_trade(self, osi_symbol: str, executions: Dict) -> str:
        """
        Get pattern type for a trade by matching OSI symbol to execution.

        Args:
            osi_symbol: OCC option symbol (e.g., DIA251226P00485000)
            executions: Dictionary of executions keyed by signal_key

        Returns:
            Pattern string (e.g., '2U-1-?') or empty string if not found
        """
        if not executions:
            return ''

        # Find execution that matches this OSI symbol
        for signal_key, exec_data in executions.items():
            if exec_data.get('osi_symbol') == osi_symbol:
                # Extract pattern from signal_key
                # Format: SYMBOL_TIMEFRAME_PATTERN_TIMESTAMP
                # e.g., DIA_1D_2U-1-?_202512161400
                parts = signal_key.split('_')
                if len(parts) >= 3:
                    return parts[2]  # Pattern is the 3rd part
                break

        return ''

    def get_closed_trades_summary(self, days: int = 30) -> Dict:
        """
        Get summary statistics for closed trades.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            Dictionary with summary stats (total P&L, win rate, etc.)
        """
        trades = self.get_closed_trades(days=days)

        if not trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
            }

        total_pnl = sum(t['realized_pnl'] for t in trades)
        winners = [t for t in trades if t['realized_pnl'] > 0]
        losers = [t for t in trades if t['realized_pnl'] <= 0]

        return {
            'total_trades': len(trades),
            'total_pnl': round(total_pnl, 2),
            'win_count': len(winners),
            'loss_count': len(losers),
            'win_rate': round(len(winners) / len(trades) * 100, 1) if trades else 0.0,
            'avg_pnl': round(total_pnl / len(trades), 2) if trades else 0.0,
            'avg_win': round(
                sum(t['realized_pnl'] for t in winners) / len(winners), 2
            ) if winners else 0.0,
            'avg_loss': round(
                sum(t['realized_pnl'] for t in losers) / len(losers), 2
            ) if losers else 0.0,
            'largest_win': round(
                max(t['realized_pnl'] for t in winners), 2
            ) if winners else 0.0,
            'largest_loss': round(
                min(t['realized_pnl'] for t in losers), 2
            ) if losers else 0.0,
        }

    def get_positions_with_signals(self) -> List[Dict]:
        """
        Get live positions linked to their original signals for trade progress tracking.

        Session EQUITY-33: Enables the "Trade Progress to Target" chart.

        Returns:
            List of trade dictionaries with:
            - name: Display name (e.g., "SPY 3-1-2U 1H")
            - entry: Entry trigger price (underlying stock)
            - current: Current underlying stock price
            - target: Target price (underlying stock)
            - stop: Stop price (underlying stock)
            - direction: CALL or PUT
            - pnl_pct: Current P&L percentage
        """
        positions = self.get_option_positions()
        if not positions:
            return []

        # Load executions and signals to link
        executions = self._load_executions()

        # Load all signals from store for target/stop data
        all_signals = {}
        if self.signal_store:
            try:
                for signal in self.signal_store.get_all():
                    all_signals[signal.signal_key] = signal
            except Exception as e:
                logger.debug(f"Could not load signals: {e}")

        # Collect unique underlying symbols to fetch prices
        underlying_symbols = set()
        position_signals = []

        for pos in positions:
            osi_symbol = pos.get('symbol', '')

            # Find the signal_key for this position
            signal = None
            for key, exec_data in executions.items():
                if exec_data.get('osi_symbol') == osi_symbol:
                    signal = all_signals.get(key)
                    break

            if signal:
                underlying_symbols.add(signal.symbol)
                position_signals.append((pos, signal))

        # Fetch current underlying prices
        underlying_prices = {}
        if underlying_symbols and self._connected:
            try:
                quotes = self.client.get_stock_quotes(list(underlying_symbols))
                for symbol, quote in quotes.items():
                    if isinstance(quote, dict):
                        underlying_prices[symbol] = quote.get('mid', quote.get('last', 0))
                    elif isinstance(quote, (int, float)):
                        underlying_prices[symbol] = float(quote)
            except Exception as e:
                logger.debug(f"Could not fetch underlying prices: {e}")

        # Build trade data
        trades = []
        for pos, signal in position_signals:
            current_underlying = underlying_prices.get(signal.symbol, signal.entry_trigger)

            trade = {
                'name': f"{signal.symbol} {signal.pattern_type} {signal.timeframe}",
                'entry': signal.entry_trigger,
                'current': current_underlying,
                'target': signal.target_price,
                'stop': signal.stop_price,
                'direction': signal.direction,
                'pnl_pct': float(pos.get('unrealized_plpc', 0)),
                'osi_symbol': pos.get('symbol', ''),
            }
            trades.append(trade)

        return trades

    def _signal_to_dict(self, signal: StoredSignal) -> Dict:
        """
        Convert StoredSignal to dashboard-friendly dictionary.

        Args:
            signal: StoredSignal object from signal store

        Returns:
            Dictionary with display-ready fields
        """
        return {
            'signal_key': signal.signal_key,
            'symbol': signal.symbol,
            'timeframe': signal.timeframe,
            'pattern': signal.pattern_type,
            'direction': signal.direction,
            'entry_trigger': signal.entry_trigger,
            'target': signal.target_price,
            'stop': signal.stop_price,
            'magnitude_pct': signal.magnitude_pct,
            'risk_reward': signal.risk_reward,
            'status': signal.status,
            'signal_type': getattr(signal, 'signal_type', 'COMPLETED'),
            'vix': signal.vix,
            'regime': signal.market_regime,
            'detected_time': (
                signal.detected_time.strftime('%Y-%m-%d %H:%M')
                if signal.detected_time else None
            ),
            'triggered_at': (
                signal.triggered_at.strftime('%Y-%m-%d %H:%M')
                if signal.triggered_at else None
            ),
        }

    def _parse_occ_symbol(self, occ_symbol: str) -> str:
        """
        Parse OCC option symbol into human-readable format.

        Example: SPY241220C00600000 -> SPY 12/20 $600C

        Args:
            occ_symbol: OCC format option symbol

        Returns:
            Human-readable contract string
        """
        if not occ_symbol or len(occ_symbol) < 15:
            return occ_symbol

        try:
            # OCC format: ROOT(1-6) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
            # Find where date starts (6 digits after root)
            root = occ_symbol[:-15]  # Everything before last 15 chars
            date_str = occ_symbol[-15:-9]  # YYMMDD
            option_type = occ_symbol[-9]  # C or P
            strike_raw = occ_symbol[-8:]  # 8 digits

            # Parse date
            yy = date_str[:2]
            mm = date_str[2:4]
            dd = date_str[4:6]

            # Parse strike (divide by 1000)
            strike = int(strike_raw) / 1000

            # Format: SPY 12/20 $600C
            return f"{root} {mm}/{dd} ${strike:.0f}{option_type}"
        except Exception:
            return occ_symbol


# Export for dashboard
__all__ = ['OptionsDataLoader']
