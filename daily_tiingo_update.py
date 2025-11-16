"""
Daily data update workflow for Tiingo.

Run this once per day after market close to keep data current.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from integrations.tiingo_data_fetcher import TiingoDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def daily_update():
    """
    Update all index data with latest prices.
    """
    logging.info("Starting daily Tiingo data update...")

    # API key should be set in environment variable TIINGO_API_KEY
    # Verify it exists before proceeding
    if not os.environ.get('TIINGO_API_KEY'):
        raise ValueError("TIINGO_API_KEY environment variable not set")

    # Initialize fetcher
    fetcher = TiingoDataFetcher()

    # Symbols to update (index-level only for free tier)
    symbols = ['SPY', 'QQQ', 'IWM']

    # Update each symbol (fetches from inception to latest)
    fetcher.update_cache(symbols, start_date='1990-01-01')

    # Verify latest data is from today or yesterday
    print("\nVerifying data freshness...")
    print("=" * 80)

    for symbol in symbols:
        data = fetcher.fetch(symbol, start_date='2025-01-01', use_cache=True)
        latest_date = data.wrapper.index[-1]
        days_old = (datetime.now(latest_date.tz).date() - latest_date.date()).days

        if days_old <= 1:
            logging.info(f"[PASS] {symbol}: Latest data is {latest_date.date()} ({days_old} days old)")
        else:
            logging.warning(f"[WARN] {symbol}: Latest data is {latest_date.date()} ({days_old} days old) - may need investigation")

    logging.info("Daily update complete!")
    print("=" * 80)


if __name__ == "__main__":
    daily_update()
