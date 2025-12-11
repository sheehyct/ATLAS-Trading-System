"""
VIX Data Fetcher and Analysis Module

Session 83K-53: Created for VIX correlation with bars-to-magnitude analysis.

Per CLAUDE.md: VIX data ONLY from yfinance (not available on Alpaca).

Usage:
    from analysis.vix_data import fetch_vix_data, get_vix_at_date, categorize_vix

    # Fetch VIX data
    vix = fetch_vix_data('2020-01-01', '2024-12-31')

    # Get VIX for specific date
    vix_value = get_vix_at_date(vix, pd.Timestamp('2024-01-15'))

    # Categorize VIX
    bucket = categorize_vix(vix_value)  # Returns 1-5 (LOW to EXTREME)
    name = get_vix_bucket_name(bucket)  # Returns 'LOW', 'NORMAL', etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


# VIX bucket definitions
# Session 83K-53: Aligned with GATE_1_ML_FRAMEWORK.md
VIX_BUCKETS: List[Tuple[float, float, int]] = [
    (0, 15, 1),             # LOW - calm markets
    (15, 20, 2),            # NORMAL - typical conditions
    (20, 30, 3),            # ELEVATED - increased uncertainty
    (30, 40, 4),            # HIGH - significant fear
    (40, float('inf'), 5),  # EXTREME - panic/crisis
]

VIX_BUCKET_NAMES = {
    0: 'UNKNOWN',
    1: 'LOW',
    2: 'NORMAL',
    3: 'ELEVATED',
    4: 'HIGH',
    5: 'EXTREME',
}

# Cache path for VIX data
VIX_CACHE_DIR = Path('data_cache')


def fetch_vix_data(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_dir: Path = None,
) -> pd.Series:
    """
    Fetch VIX close prices from yfinance with optional caching.

    Per CLAUDE.md Rule 4: VIX data ONLY from yfinance.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    use_cache : bool
        Whether to use cached data if available
    cache_dir : Path
        Directory for cache files (default: data_cache/)

    Returns
    -------
    pd.Series
        VIX close prices indexed by date (timezone-naive)
    """
    import yfinance as yf

    cache_dir = cache_dir or VIX_CACHE_DIR
    cache_file = cache_dir / 'vix_daily.parquet'

    # Check cache
    if use_cache and cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            # Normalize index to timezone-naive
            if hasattr(cached.index, 'tz') and cached.index.tz is not None:
                cached.index = cached.index.tz_localize(None)

            cached_min = cached.index.min()
            cached_max = cached.index.max()

            if cached_min <= pd.Timestamp(start_date) and cached_max >= pd.Timestamp(end_date):
                # Filter to requested range
                mask = (cached.index >= start_date) & (cached.index <= end_date)
                return cached.loc[mask, 'Close']
        except Exception:
            pass  # Cache invalid, refetch

    # Fetch from yfinance
    print(f"Fetching VIX data from {start_date} to {end_date}...")
    ticker = yf.Ticker('^VIX')
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No VIX data returned for {start_date} to {end_date}")

    # Normalize index to timezone-naive
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Cache for future use
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    print(f"VIX data cached to {cache_file}")

    return df['Close']


def get_vix_at_date(
    vix_series: pd.Series,
    date: pd.Timestamp,
    max_lookback_days: int = 7,
) -> Optional[float]:
    """
    Get VIX value for a specific date, with fallback to previous trading day.

    Handles weekends and holidays by looking back up to max_lookback_days.

    Parameters
    ----------
    vix_series : pd.Series
        VIX close prices indexed by date
    date : pd.Timestamp
        Date to look up
    max_lookback_days : int
        Maximum days to look back for valid data

    Returns
    -------
    Optional[float]
        VIX value, or None if no data found
    """
    # Normalize date to timezone-naive and start of day
    date = pd.Timestamp(date)
    if hasattr(date, 'tz') and date.tz is not None:
        date = date.tz_localize(None)
    date = date.normalize()

    # Normalize VIX series index if needed
    if hasattr(vix_series.index, 'tz') and vix_series.index.tz is not None:
        vix_series = vix_series.copy()
        vix_series.index = vix_series.index.tz_localize(None)

    # Try exact date first
    if date in vix_series.index:
        return float(vix_series.loc[date])

    # Fall back to previous trading day
    for i in range(1, max_lookback_days + 1):
        prev_date = date - pd.Timedelta(days=i)
        if prev_date in vix_series.index:
            return float(vix_series.loc[prev_date])

    return None


def categorize_vix(vix_value: float) -> int:
    """
    Categorize VIX value into bucket (1-5).

    Bucket definitions:
        1: LOW (0-15) - Calm markets, low fear
        2: NORMAL (15-20) - Typical market conditions
        3: ELEVATED (20-30) - Increased uncertainty
        4: HIGH (30-40) - Significant fear
        5: EXTREME (40+) - Panic/crisis mode

    Parameters
    ----------
    vix_value : float
        VIX index value

    Returns
    -------
    int
        Bucket number (0 if unknown/invalid, 1-5 otherwise)
    """
    if vix_value is None or (isinstance(vix_value, float) and np.isnan(vix_value)):
        return 0  # Unknown

    for low, high, bucket in VIX_BUCKETS:
        if low <= vix_value < high:
            return bucket

    return 5  # Extreme (shouldn't reach here due to inf upper bound)


def get_vix_bucket_name(bucket: int) -> str:
    """
    Get human-readable name for VIX bucket.

    Parameters
    ----------
    bucket : int
        Bucket number (0-5)

    Returns
    -------
    str
        Bucket name (e.g., 'LOW', 'NORMAL', 'ELEVATED', 'HIGH', 'EXTREME')
    """
    return VIX_BUCKET_NAMES.get(bucket, 'UNKNOWN')


def add_vix_to_trades(
    trades_df: pd.DataFrame,
    entry_date_col: str = 'entry_date',
    vix_start_buffer_days: int = 30,
) -> pd.DataFrame:
    """
    Add VIX data columns to a trades DataFrame.

    Adds:
        - vix_at_entry: VIX value on entry date
        - vix_bucket: Bucket number (1-5)
        - vix_bucket_name: Human-readable bucket name

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trade data
    entry_date_col : str
        Column name containing entry dates
    vix_start_buffer_days : int
        Days before first trade to start VIX fetch (for lookback)

    Returns
    -------
    pd.DataFrame
        DataFrame with VIX columns added
    """
    df = trades_df.copy()

    # Parse dates
    df[entry_date_col] = pd.to_datetime(df[entry_date_col])

    # Get date range
    min_date = df[entry_date_col].min() - pd.Timedelta(days=vix_start_buffer_days)
    max_date = df[entry_date_col].max() + pd.Timedelta(days=7)

    # Fetch VIX data
    vix = fetch_vix_data(
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d'),
    )

    # Add VIX columns
    df['vix_at_entry'] = df[entry_date_col].apply(lambda d: get_vix_at_date(vix, d))
    df['vix_bucket'] = df['vix_at_entry'].apply(categorize_vix)
    df['vix_bucket_name'] = df['vix_bucket'].apply(get_vix_bucket_name)

    return df
