"""
VectorBT Pro compatible Tiingo data fetcher.

Features:
- Local caching to minimize API calls
- Automatic data updates
- VectorBT Pro Data object output
- Cross-validation with Alpaca

Session 70: Updated to use centralized config.settings
"""

import vectorbtpro as vbt
import pandas as pd
from tiingo import TiingoClient
from datetime import datetime
from pathlib import Path

# Use centralized config (loads from root .env with all credentials)
from config.settings import get_tiingo_key


class TiingoDataFetcher:
    """
    Fetch historical data from Tiingo and format for VectorBT Pro.

    Usage:
        fetcher = TiingoDataFetcher()
        data = fetcher.fetch('SPY', start_date='1993-01-01', end_date='2025-11-15')
    """

    def __init__(self, api_key=None, cache_dir='./data/tiingo_cache'):
        """
        Initialize Tiingo data fetcher.

        Args:
            api_key: Tiingo API key (or use TIINGO_API_KEY env variable via config.settings)
            cache_dir: Directory for caching downloaded data
        """
        # Use centralized config for API key
        self.api_key = api_key or get_tiingo_key()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Tiingo client
        config = {
            'api_key': self.api_key,
            'session': True
        }
        self.client = TiingoClient(config)

    def fetch(self, symbols, start_date, end_date=None, timeframe='1d', use_cache=True):
        """
        Fetch historical data for symbols.

        Args:
            symbols: Single symbol string or list of symbols
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime), defaults to today
            timeframe: Timeframe ('1d', '1w', '1m', '1y')
            use_cache: Use cached data if available

        Returns:
            VectorBT Pro Data object
        """
        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Convert timeframe to Tiingo frequency
        frequency_map = {
            '1d': 'daily',
            '1D': 'daily',
            'daily': 'daily',
            '1w': 'weekly',
            '1W': 'weekly',
            'weekly': 'weekly',
            '1m': 'monthly',
            '1M': 'monthly',
            'monthly': 'monthly',
            '1y': 'annually',
            '1Y': 'annually',
            'annually': 'annually'
        }
        frequency = frequency_map.get(timeframe, 'daily')

        # Fetch data for each symbol
        data_dict = {}

        for symbol in symbols:
            # Check cache
            cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{frequency}.pkl"

            if use_cache and cache_file.exists():
                print(f"Loading {symbol} from cache...")
                df = pd.read_pickle(cache_file)
            else:
                print(f"Fetching {symbol} from Tiingo API...")
                df = self.client.get_dataframe(
                    symbol,
                    startDate=start_date,
                    endDate=end_date,
                    frequency=frequency
                )

                # Rename columns to match VectorBT Pro convention
                df = df.rename(columns={
                    'adjOpen': 'Open',
                    'adjHigh': 'High',
                    'adjLow': 'Low',
                    'adjClose': 'Close',
                    'adjVolume': 'Volume'
                })

                # Keep only OHLCV columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                # Save to cache
                df.to_pickle(cache_file)
                print(f"  Cached {len(df)} days")

            data_dict[symbol] = df

        # Combine into single DataFrame if multiple symbols
        if len(symbols) == 1:
            # For single symbol, return data WITHOUT MultiIndex to match Alpaca behavior
            # This ensures compatibility with existing validation scripts
            combined_df = data_dict[symbols[0]]
        else:
            combined_df = pd.concat(data_dict, axis=1)

        # Convert to VectorBT Pro Data object
        vbt_data = vbt.Data.from_data(combined_df)

        return vbt_data

    def update_cache(self, symbols, start_date='1990-01-01'):
        """
        Update cached data to latest available date.

        Args:
            symbols: Single symbol or list of symbols to update
            start_date: Historical start date for full cache
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        end_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in symbols:
            print(f"Updating {symbol}...")
            # Force fresh download
            self.fetch(symbol, start_date, end_date, use_cache=False)

        print(f"Updated {len(symbols)} symbols")

    def clear_cache(self, symbol=None):
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            pattern = f"{symbol}_*.pkl"
        else:
            pattern = "*.pkl"

        files = list(self.cache_dir.glob(pattern))
        for file in files:
            file.unlink()

        print(f"Cleared {len(files)} cache files")


# Example usage
if __name__ == "__main__":
    import os

    # API key should be set in environment variable TIINGO_API_KEY
    # or passed to TiingoDataFetcher(api_key='your_key_here')

    # Initialize fetcher
    fetcher = TiingoDataFetcher()

    # Fetch 30 years of SPY data
    print("=" * 80)
    print("Fetching 30 years of SPY data...")
    print("=" * 80)

    spy_data = fetcher.fetch(
        symbols='SPY',
        start_date='1993-01-01',
        end_date='2025-11-15',
        timeframe='1d'
    )

    print(f"\nData object: {spy_data}")
    print(f"Shape: {spy_data.shape}")
    print(f"Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
    print(f"Columns: {spy_data.columns.tolist()}")

    # Extract DataFrame for inspection
    spy_df = spy_data.get('SPY')
    print(f"\nFirst 5 rows:")
    print(spy_df.head())
    print(f"\nLast 5 rows:")
    print(spy_df.tail())

    print("\n" + "=" * 80)
    print("VectorBT Pro Data object created successfully!")
    print("=" * 80)
