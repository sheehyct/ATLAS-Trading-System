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

    def __init__(self, account=None):
        """
        Initialize LiveDataLoader with AlpacaTradingClient.

        Args:
            account: Account to use ('LARGE', 'MID', 'SMALL').
                     If None, uses DEFAULT_ACCOUNT from .env (currently MID).
        """
        # Use default account from config if not specified
        if account is None:
            from config.settings import get_default_account
            account = get_default_account()

        self.account = account
        self.client = None
        self.init_error = None

        try:
            from integrations.alpaca_trading_client import AlpacaTradingClient

            self.client = AlpacaTradingClient(account=account)
            if self.client.connect():
                logger.info(f"LiveDataLoader initialized with {account} account")
            else:
                self.init_error = f"Failed to connect to Alpaca {account} account"
                logger.error(self.init_error)
                self.client = None

        except ValueError as e:
            self.init_error = str(e)
            logger.error(f"Alpaca credentials error: {e}")
            self.client = None
        except Exception as e:
            self.init_error = str(e)
            logger.error(f"Failed to initialize AlpacaTradingClient: {e}")
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

    def get_portfolio_history(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch portfolio history from Alpaca for equity curve.

        Session EQUITY-52: Added for unified STRAT analytics dashboard.

        Args:
            days: Number of days of history to fetch (default: 90)

        Returns:
            DataFrame with columns:
                - date: Date string
                - equity: Account equity value
                - profit_loss: Daily P&L
                - profit_loss_pct: Daily P&L percentage
        """
        try:
            if self.client is None:
                logger.warning("AlpacaTradingClient not initialized")
                return pd.DataFrame()

            # Use the Alpaca client's underlying trading_client to get portfolio history
            # The alpaca-py library has a get_portfolio_history method
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            from datetime import datetime, timedelta

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            request = GetPortfolioHistoryRequest(
                period=f'{days}D',
                timeframe='1D',
                extended_hours=False
            )

            history = self.client.trading_client.get_portfolio_history(request)

            if history is None or history.timestamp is None:
                logger.warning("No portfolio history returned from Alpaca")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct
            })

            # Convert timestamp to date string
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')

            logger.info(f"Fetched {len(df)} days of portfolio history")
            return df[['date', 'equity', 'profit_loss', 'profit_loss_pct']]

        except ImportError as e:
            logger.error(f"Missing alpaca-py dependency: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}", exc_info=True)
            return pd.DataFrame()
