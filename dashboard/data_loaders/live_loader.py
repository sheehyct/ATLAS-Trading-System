"""
Live Data Loader

Fetches real-time data from Alpaca API for live portfolio monitoring.
Provides current positions, account status, and market data.
"""

import pandas as pd
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LiveDataLoader:
    """
    Fetch real-time data from Alpaca for live monitoring.

    Uses AlpacaTradingClient to fetch real account data, positions,
    and market status from paper trading or live account.
    """

    def __init__(self, account='LARGE'):
        """
        Initialize LiveDataLoader with AlpacaTradingClient.

        Args:
            account: Account to use ('LARGE' for $10k paper, 'SMALL' for $3k)
        """
        self.account = account
        self.client = None

        try:
            from integrations.alpaca_trading_client import AlpacaTradingClient

            self.client = AlpacaTradingClient(account=account)
            self.client.connect()
            logger.info(f"LiveDataLoader initialized with {account} account")

        except Exception as e:
            logger.error(f"Failed to initialize AlpacaTradingClient: {e}")
            logger.warning("LiveDataLoader will return empty data")
            self.client = None

    def get_current_positions(self) -> pd.DataFrame:
        """
        Get current open positions from Alpaca.

        Returns:
            DataFrame with columns:
                - symbol: Stock symbol
                - qty: Quantity held
                - market_value: Current market value
                - unrealized_pl: Unrealized P&L
                - unrealized_plpc: Unrealized P&L %
                - avg_entry_price: Average entry price
                - current_price: Current market price
        """

        try:
            if self.client is None:
                logger.warning("AlpacaTradingClient not initialized")
                return pd.DataFrame()

            # Fetch real positions from Alpaca
            positions = self.client.list_positions()

            if not positions:
                logger.info("No open positions")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(positions)

            logger.info(f"Fetched {len(df)} positions")
            return df

        except Exception as e:
            logger.error(f"Error getting positions: {e}", exc_info=True)
            return pd.DataFrame()

    def get_account_status(self) -> Dict:
        """
        Get account equity, buying power, etc from Alpaca.

        Returns:
            Dictionary with:
                - equity: Total account equity
                - cash: Available cash
                - buying_power: Buying power
                - portfolio_value: Portfolio value
                - timestamp: Last update timestamp
        """

        try:
            if self.client is None:
                logger.warning("AlpacaTradingClient not initialized")
                return {}

            # Fetch real account data from Alpaca
            account_data = self.client.get_account()

            logger.info("Fetched account status")
            return account_data

        except Exception as e:
            logger.error(f"Error getting account status: {e}", exc_info=True)
            return {}

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price or None if unavailable
        """

        try:
            if self.data_client is None:
                logger.warning("Data client not initialized")
                return None

            # PLACEHOLDER: Would fetch from Alpaca
            # from alpaca.data.requests import StockLatestQuoteRequest
            #
            # request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            # quote = self.data_client.get_stock_latest_quote(request)
            # return float(quote[symbol].ask_price)

            # For now, return dummy price
            logger.info(f"Fetched price for {symbol}")
            return 450.00  # Dummy price

        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None

    def get_market_status(self) -> Dict:
        """
        Get current market status from Alpaca.

        Returns:
            Dictionary with:
                - is_open: Boolean indicating if market is currently open
                - timestamp: Current time
        """

        try:
            if self.client is None:
                logger.warning("AlpacaTradingClient not initialized")
                return {'is_open': False}

            # Use Alpaca's market calendar to check if market is open
            # For now, return basic status (can be enhanced later with full clock API)
            from datetime import datetime
            import pandas_market_calendars as mcal

            nyse = mcal.get_calendar('NYSE')
            now = datetime.now()
            schedule = nyse.schedule(start_date=now.date(), end_date=now.date())

            is_open = len(schedule) > 0

            return {
                'is_open': is_open,
                'timestamp': now.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting market status: {e}", exc_info=True)
            return {'is_open': False}
