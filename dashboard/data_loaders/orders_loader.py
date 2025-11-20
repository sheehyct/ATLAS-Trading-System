"""
Order History Data Loader

Loads order execution history from CSV log files in logs/ directory.
Provides recent orders, fill history, and execution statistics.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OrdersDataLoader:
    """
    Load and format order execution data from CSV logs.

    Reads trade logs from logs/trades_{date}.csv files and provides
    formatted data for dashboard display.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize OrdersDataLoader.

        Args:
            log_dir: Optional path to logs directory (defaults to logs/)
        """
        self.log_dir = log_dir or Path(__file__).parent.parent.parent / 'logs'

        if not self.log_dir.exists():
            logger.warning(f"Logs directory not found: {self.log_dir}")

        logger.info(f"OrdersDataLoader initialized with log_dir: {self.log_dir}")

    def get_recent_orders(self, days: int = 7) -> pd.DataFrame:
        """
        Get recent orders from CSV log files.

        Args:
            days: Number of days to look back (default: 7)

        Returns:
            DataFrame with columns:
                - timestamp: Order timestamp
                - symbol: Stock symbol
                - action: Order action (BUY/SELL/OPEN/CLOSE/ADJUST)
                - qty: Quantity
                - price: Fill price
                - order_type: Order type (market/limit)
                - order_id: Alpaca order ID
                - status: Order status (submitted/filled/rejected)
                - error: Error message if any
        """

        try:
            if not self.log_dir.exists():
                logger.warning(f"Logs directory does not exist: {self.log_dir}")
                return pd.DataFrame()

            dfs = []

            # Read CSV files for last N days
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                csv_file = self.log_dir / f'trades_{date}.csv'

                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        dfs.append(df)
                        logger.debug(f"Loaded {len(df)} orders from {csv_file.name}")
                    except Exception as e:
                        logger.error(f"Error reading {csv_file}: {e}")

            if not dfs:
                logger.info("No order history found")
                return pd.DataFrame(columns=[
                    'timestamp', 'symbol', 'action', 'qty', 'price',
                    'order_type', 'order_id', 'status', 'error'
                ])

            # Combine and sort by timestamp
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.sort_values('timestamp', ascending=False)

            logger.info(f"Loaded {len(combined)} total orders from {len(dfs)} log files")
            return combined

        except Exception as e:
            logger.error(f"Error loading recent orders: {e}", exc_info=True)
            return pd.DataFrame()

    def get_filled_orders(self, days: int = 7) -> pd.DataFrame:
        """
        Get only filled orders (filter out submissions/rejections).

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with only filled orders
        """

        try:
            orders = self.get_recent_orders(days=days)

            if orders.empty:
                return orders

            # Filter for filled orders only
            filled = orders[orders['status'] == 'filled'].copy()

            logger.info(f"Found {len(filled)} filled orders out of {len(orders)} total")
            return filled

        except Exception as e:
            logger.error(f"Error getting filled orders: {e}", exc_info=True)
            return pd.DataFrame()

    def get_order_summary(self, days: int = 30) -> Dict:
        """
        Get summary statistics for recent orders.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with:
                - total: Total number of orders
                - filled: Number of filled orders
                - rejected: Number of rejected orders
                - pending: Number of pending orders
                - fill_rate: Percentage of orders filled
        """

        try:
            orders = self.get_recent_orders(days=days)

            if orders.empty:
                return {
                    'total': 0,
                    'filled': 0,
                    'rejected': 0,
                    'pending': 0,
                    'fill_rate': 0.0
                }

            total = len(orders)
            filled = len(orders[orders['status'] == 'filled'])
            rejected = len(orders[orders['status'] == 'rejected'])
            pending = len(orders[orders['status'] == 'submitted'])

            fill_rate = (filled / total * 100) if total > 0 else 0.0

            summary = {
                'total': total,
                'filled': filled,
                'rejected': rejected,
                'pending': pending,
                'fill_rate': fill_rate
            }

            logger.info(f"Order summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Error calculating order summary: {e}", exc_info=True)
            return {
                'total': 0,
                'filled': 0,
                'rejected': 0,
                'pending': 0,
                'fill_rate': 0.0
            }

    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        """
        Get details for a specific order by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            Dictionary with order details or None if not found
        """

        try:
            orders = self.get_recent_orders(days=90)  # Search last 90 days

            if orders.empty:
                return None

            # Find order by ID
            matches = orders[orders['order_id'] == order_id]

            if matches.empty:
                logger.warning(f"Order not found: {order_id}")
                return None

            # Return most recent match (in case of multiple events)
            order = matches.iloc[0].to_dict()
            logger.info(f"Found order: {order_id}")

            return order

        except Exception as e:
            logger.error(f"Error getting order by ID: {e}", exc_info=True)
            return None
