"""
Options Data Loader for Dashboard

Fetches live options signals and positions for the options trading panel.

Data Sources:
- Signal Store: Pending/active STRAT signals from signal_automation
- Alpaca API: Live option positions via AlpacaTradingClient

Session 83K-75: Initial implementation for dashboard integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from integrations.alpaca_trading_client import AlpacaTradingClient
from strat.signal_automation.signal_store import SignalStore, StoredSignal

logger = logging.getLogger(__name__)


class OptionsDataLoader:
    """Load options signals and positions for dashboard."""

    def __init__(self, account: str = 'SMALL'):
        """
        Initialize the options data loader.

        Args:
            account: Alpaca account tier ('SMALL', 'MID', 'LARGE')
        """
        self.account = account
        self.client = AlpacaTradingClient(account=account)
        self.signal_store = SignalStore()
        self._connected = False
        self.init_error = None

        # Try to connect on init
        try:
            self._connected = self.client.connect()
            if self._connected:
                logger.info(f"OptionsDataLoader connected to Alpaca {account} account")
            else:
                self.init_error = "Alpaca connection returned False"
                logger.warning(f"OptionsDataLoader: Alpaca connection failed")
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

    def get_pending_signals(self) -> List[Dict]:
        """
        Get SETUP signals awaiting entry trigger.

        Returns:
            List of signal dictionaries ready for monitoring
        """
        try:
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
            all_signals = self.signal_store.load_signals()
            triggered = [
                s for s in all_signals.values()
                if s.status in ('TRIGGERED', 'HISTORICAL_TRIGGERED')
            ]
            return [self._signal_to_dict(s) for s in triggered]
        except Exception as e:
            logger.error(f"Error getting triggered signals: {e}")
            return []

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
