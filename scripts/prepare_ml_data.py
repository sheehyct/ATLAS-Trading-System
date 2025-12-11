"""
ML Data Preparation Script for STRAT Options Validation

Created: Session 83K-40 (December 4, 2025)
Purpose: Prepare trade data for Gate 1 ML optimization

This script:
1. Merges non-hourly trade CSVs for ML-eligible patterns (2-2, 3-2)
2. Engineers features per GATE_1_ML_FRAMEWORK.md
3. Creates temporal splits (60/20/20) with purging/embargo
4. Exports ML-ready datasets

Usage:
    uv run python scripts/prepare_ml_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configuration
TRADES_DIR = Path('validation_results/session_83k/trades')
OUTPUT_DIR = Path('validation_results/ml')
ML_ELIGIBLE_PATTERNS = ['2-2', '3-2']
NON_HOURLY_TIMEFRAMES = ['1D', '1W', '1M']

# Feature encoding maps
TIMEFRAME_ENCODING = {'1D': 1, '1W': 5, '1M': 21}
PATTERN_ENCODING = {'2-2': 1, '3-2': 2}
SYMBOL_ENCODING = {'SPY': 1, 'QQQ': 2, 'AAPL': 3, 'IWM': 4, 'DIA': 5, 'NVDA': 6}

# VIX bucket boundaries (per GATE_1_ML_FRAMEWORK.md)
VIX_BUCKETS = [
    (0, 15, 1),      # Low volatility
    (15, 20, 2),     # Normal
    (20, 30, 3),     # Elevated
    (30, 40, 4),     # High
    (40, float('inf'), 5)  # Extreme
]


def load_and_merge_trades(patterns: List[str], timeframes: List[str]) -> pd.DataFrame:
    """
    Load and merge trade CSVs for specified patterns and timeframes.

    Parameters
    ----------
    patterns : List[str]
        Pattern types to include (e.g., ['2-2', '3-2'])
    timeframes : List[str]
        Timeframes to include (e.g., ['1D', '1W', '1M'])

    Returns
    -------
    pd.DataFrame
        Merged trades with timeframe column added
    """
    all_trades = []

    for pattern in patterns:
        for tf in timeframes:
            # Find all files matching pattern_timeframe_*_trades.csv
            file_pattern = f"{pattern}_{tf}_*_trades.csv"
            files = list(TRADES_DIR.glob(file_pattern))

            for file_path in files:
                df = pd.read_csv(file_path)
                df['timeframe'] = tf
                df['source_file'] = file_path.name
                all_trades.append(df)

    if not all_trades:
        raise ValueError(f"No trade files found for patterns {patterns}, timeframes {timeframes}")

    merged = pd.concat(all_trades, ignore_index=True)

    # Convert dates to datetime (handle timezone-aware strings)
    merged['entry_date'] = pd.to_datetime(merged['entry_date'], utc=True).dt.tz_convert('America/New_York')
    merged['exit_date'] = pd.to_datetime(merged['exit_date'], utc=True).dt.tz_convert('America/New_York')
    merged['pattern_timestamp'] = pd.to_datetime(merged['pattern_timestamp'], utc=True).dt.tz_convert('America/New_York')

    # Sort by entry date (required for temporal splits)
    merged = merged.sort_values('entry_date').reset_index(drop=True)

    print(f"Loaded {len(merged)} trades from {len(all_trades)} files")
    return merged


def fetch_vix_data_cached(start_date: str, end_date: str) -> pd.Series:
    """
    Fetch VIX data from Yahoo Finance with caching.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.Series
        VIX close prices indexed by date
    """
    import yfinance as yf

    cache_file = OUTPUT_DIR / 'vix_cache.parquet'

    # Check cache
    if cache_file.exists():
        try:
            cached = pd.read_parquet(cache_file)
            cached_min = cached.index.min()
            cached_max = cached.index.max()
            # Handle timezone-aware cached data
            if cached_min.tzinfo is not None:
                cached_min = cached_min.tz_localize(None)
                cached_max = cached_max.tz_localize(None)
            if cached_min <= pd.Timestamp(start_date) and cached_max >= pd.Timestamp(end_date):
                return cached['Close']
        except Exception:
            pass  # Cache invalid, refetch

    # Fetch from yfinance
    print(f"Fetching VIX data from {start_date} to {end_date}...")
    ticker = yf.Ticker('^VIX')
    df = ticker.history(start=start_date, end=end_date)

    # Cache
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)

    return df['Close']


def get_vix_for_date(vix_series: pd.Series, date: pd.Timestamp) -> Optional[float]:
    """Get VIX value for a specific date, handling weekends/holidays."""
    date = pd.Timestamp(date).normalize().tz_localize(None)

    # Try exact date first
    if date in vix_series.index:
        return vix_series.loc[date]

    # Fall back to previous trading day
    for i in range(1, 8):
        prev_date = date - pd.Timedelta(days=i)
        if prev_date in vix_series.index:
            return vix_series.loc[prev_date]

    return None


def categorize_vix(vix_value: float) -> int:
    """Categorize VIX into buckets per GATE_1_ML_FRAMEWORK.md."""
    if vix_value is None or np.isnan(vix_value):
        return 0  # Unknown

    for low, high, bucket in VIX_BUCKETS:
        if low <= vix_value < high:
            return bucket
    return 5  # Extreme


def calculate_win(row: pd.Series) -> int:
    """Calculate if trade was a win (1) or loss (0)."""
    return 1 if row['pnl'] > 0 else 0


def extract_pattern_from_filename(source_file: str) -> str:
    """
    Extract base pattern from source filename.

    Note: Session 83K-38 discovered 3-2 and 3-2-2 patterns were mislabeled
    as 3-1-2 in CSV exports. The filename is the reliable source of truth.

    Filename format: {pattern}_{timeframe}_{symbol}_trades.csv
    Examples: 2-2_1D_SPY_trades.csv, 3-2_1W_QQQ_trades.csv
    """
    # Extract pattern from filename (first part before _1D, _1W, etc.)
    parts = source_file.split('_')
    if parts:
        pattern = parts[0]
        # Normalize pattern names
        if pattern in ['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']:
            return pattern
    return 'unknown'


def engineer_features(df: pd.DataFrame, vix_series: pd.Series) -> pd.DataFrame:
    """
    Engineer ML features per GATE_1_ML_FRAMEWORK.md.

    Features added:
    - timeframe_encoded: 1D=1, 1W=5, 1M=21
    - pattern_encoded: 2-2=1, 3-2=2
    - symbol_encoded: SPY=1, QQQ=2, etc.
    - day_of_week: Mon=1 to Fri=5
    - vix_close: VIX at entry date
    - vix_bucket: Categorized VIX level
    - win: 1 if P&L > 0, else 0

    Parameters
    ----------
    df : pd.DataFrame
        Merged trade data
    vix_series : pd.Series
        VIX close prices indexed by date

    Returns
    -------
    pd.DataFrame
        Trade data with engineered features
    """
    df = df.copy()

    # Timeframe encoding
    df['timeframe_encoded'] = df['timeframe'].map(TIMEFRAME_ENCODING)

    # Pattern encoding (use filename as source of truth due to Session 83K-38 bug)
    df['pattern_base'] = df['source_file'].apply(extract_pattern_from_filename)
    df['pattern_encoded'] = df['pattern_base'].map(PATTERN_ENCODING)

    # Symbol encoding
    df['symbol_encoded'] = df['symbol'].map(SYMBOL_ENCODING)

    # Day of week (Mon=1 to Fri=5)
    df['day_of_week'] = df['entry_date'].dt.dayofweek + 1

    # VIX features
    print("Adding VIX features...")
    df['vix_close'] = df['entry_date'].apply(lambda d: get_vix_for_date(vix_series, d))
    df['vix_bucket'] = df['vix_close'].apply(categorize_vix)

    # Win/loss indicator
    df['win'] = df.apply(calculate_win, axis=1)

    # Exit type encoding
    df['target_hit'] = (df['exit_type'] == 'TARGET').astype(int)
    df['stop_hit'] = (df['exit_type'] == 'STOP').astype(int)

    # Delta bucket (for analysis)
    df['delta_bucket'] = pd.cut(
        df['entry_delta'].abs(),
        bins=[0, 0.35, 0.50, 0.65, 0.80, 1.0],
        labels=[1, 2, 3, 4, 5]  # OTM to Deep ITM
    )

    return df


def create_temporal_splits(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    purge_days: int = 5,
    embargo_days: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test splits with purging and embargo.

    Parameters
    ----------
    df : pd.DataFrame
        Trade data sorted by entry_date
    train_frac : float
        Fraction of data for training (default: 0.60)
    val_frac : float
        Fraction of data for validation (default: 0.20)
    test_frac : float
        Fraction of data for testing (default: 0.20)
    purge_days : int
        Days to remove at split boundaries (default: 5)
    embargo_days : int
        Days to skip after each split (default: 2)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames
    """
    n = len(df)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    # Get split dates
    train_end_date = df.iloc[train_end_idx - 1]['entry_date']
    val_end_date = df.iloc[val_end_idx - 1]['entry_date']

    # Apply purge: remove trades within purge_days of split boundaries
    train_cutoff = train_end_date - pd.Timedelta(days=purge_days)
    val_start_cutoff = train_end_date + pd.Timedelta(days=embargo_days)
    val_cutoff = val_end_date - pd.Timedelta(days=purge_days)
    test_start_cutoff = val_end_date + pd.Timedelta(days=embargo_days)

    # Create splits
    train_df = df[df['entry_date'] <= train_cutoff].copy()
    val_df = df[(df['entry_date'] >= val_start_cutoff) & (df['entry_date'] <= val_cutoff)].copy()
    test_df = df[df['entry_date'] >= test_start_cutoff].copy()

    print(f"\nTemporal Splits Created:")
    print(f"  Train: {len(train_df)} trades ({len(train_df)/n*100:.1f}%)")
    print(f"    Date range: {train_df['entry_date'].min()} to {train_df['entry_date'].max()}")
    print(f"  Validation: {len(val_df)} trades ({len(val_df)/n*100:.1f}%)")
    print(f"    Date range: {val_df['entry_date'].min()} to {val_df['entry_date'].max()}")
    print(f"  Test: {len(test_df)} trades ({len(test_df)/n*100:.1f}%)")
    print(f"    Date range: {test_df['entry_date'].min()} to {test_df['entry_date'].max()}")
    print(f"  Purge gap: {purge_days} days, Embargo: {embargo_days} days")

    return train_df, val_df, test_df


def calculate_baseline_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate baseline metrics with static delta=0.65.

    Parameters
    ----------
    df : pd.DataFrame
        Trade data

    Returns
    -------
    Dict
        Baseline metrics
    """
    metrics = {
        'n_trades': len(df),
        'total_pnl': df['pnl'].sum(),
        'avg_pnl': df['pnl'].mean(),
        'win_rate': df['win'].mean() * 100,
        'avg_delta': df['entry_delta'].abs().mean(),
        'target_rate': df['target_hit'].mean() * 100,
        'stop_rate': df['stop_hit'].mean() * 100,
    }

    # Sharpe approximation (using daily P&L stddev)
    if len(df) > 1:
        daily_returns = df.groupby(df['entry_date'].dt.date)['pnl'].sum()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            metrics['sharpe'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            metrics['sharpe'] = 0
    else:
        metrics['sharpe'] = 0

    return metrics


def print_summary_statistics(df: pd.DataFrame, title: str = "Dataset Summary"):
    """Print summary statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"\nTrades: {len(df)}")
    print(f"Date Range: {df['entry_date'].min().date()} to {df['entry_date'].max().date()}")

    print(f"\nBy Pattern:")
    for pattern in df['pattern_base'].unique():
        subset = df[df['pattern_base'] == pattern]
        print(f"  {pattern}: {len(subset)} trades, Avg P&L: ${subset['pnl'].mean():.2f}")

    print(f"\nBy Timeframe:")
    for tf in sorted(df['timeframe'].unique()):
        subset = df[df['timeframe'] == tf]
        print(f"  {tf}: {len(subset)} trades, Avg P&L: ${subset['pnl'].mean():.2f}")

    print(f"\nBy VIX Bucket:")
    for bucket in sorted(df['vix_bucket'].unique()):
        if bucket == 0:
            continue  # Skip unknown
        subset = df[df['vix_bucket'] == bucket]
        vix_range = [b for b in VIX_BUCKETS if b[2] == bucket][0]
        print(f"  {vix_range[0]}-{vix_range[1] if vix_range[1] != float('inf') else '+'}: {len(subset)} trades, Win Rate: {subset['win'].mean()*100:.1f}%")

    print(f"\nOverall Metrics:")
    metrics = calculate_baseline_metrics(df)
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Avg P&L: ${metrics['avg_pnl']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Target Rate: {metrics['target_rate']:.1f}%")
    print(f"  Sharpe (approx): {metrics['sharpe']:.2f}")


def main():
    """Main execution function."""
    print("="*60)
    print("ML Data Preparation - Session 83K-40")
    print("="*60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and merge trades
    print("\nStep 1: Loading and merging non-hourly trades...")
    trades = load_and_merge_trades(ML_ELIGIBLE_PATTERNS, NON_HOURLY_TIMEFRAMES)

    # Get date range for VIX data
    start_date = (trades['entry_date'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (trades['entry_date'].max() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')

    # Step 2: Fetch VIX data
    print("\nStep 2: Fetching VIX data...")
    vix_series = fetch_vix_data_cached(start_date, end_date)
    # Normalize VIX index for lookup
    vix_series.index = vix_series.index.tz_localize(None)
    print(f"VIX data range: {vix_series.index.min().date()} to {vix_series.index.max().date()}")

    # Step 3: Engineer features
    print("\nStep 3: Engineering features...")
    trades = engineer_features(trades, vix_series)

    # Step 4: Summary statistics (before splits)
    print_summary_statistics(trades, "Full Dataset Summary (Before Splits)")

    # Step 5: Create temporal splits
    print("\nStep 5: Creating temporal splits...")
    train_df, val_df, test_df = create_temporal_splits(trades)

    # Step 6: Calculate baseline metrics for training set
    print("\nStep 6: Baseline Metrics (Training Set - Static Delta=0.65)")
    train_metrics = calculate_baseline_metrics(train_df)
    for key, value in train_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Step 7: Export datasets
    print("\nStep 7: Exporting datasets...")

    # Full dataset
    full_output = OUTPUT_DIR / 'ml_trades_full.csv'
    trades.to_csv(full_output, index=False)
    print(f"  Full dataset: {full_output} ({len(trades)} trades)")

    # Split datasets
    train_output = OUTPUT_DIR / 'ml_trades_train.csv'
    val_output = OUTPUT_DIR / 'ml_trades_val.csv'
    test_output = OUTPUT_DIR / 'ml_trades_test.csv'

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"  Train set: {train_output} ({len(train_df)} trades)")
    print(f"  Validation set: {val_output} ({len(val_df)} trades)")
    print(f"  Test set (SACRED): {test_output} ({len(test_df)} trades)")

    # Step 8: Export metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'session': '83K-40',
        'patterns': ML_ELIGIBLE_PATTERNS,
        'timeframes': NON_HOURLY_TIMEFRAMES,
        'total_trades': len(trades),
        'train_trades': len(train_df),
        'val_trades': len(val_df),
        'test_trades': len(test_df),
        'train_metrics': train_metrics,
        'vix_date_range': [start_date, end_date],
        'features': list(trades.columns),
    }

    import json
    metadata_output = OUTPUT_DIR / 'ml_data_metadata.json'
    with open(metadata_output, 'w') as f:
        # Convert any non-serializable types
        def convert(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        json.dump(metadata, f, indent=2, default=convert)
    print(f"  Metadata: {metadata_output}")

    print("\n" + "="*60)
    print("ML Data Preparation COMPLETE")
    print("="*60)
    print("\nNext Steps (Session 83K-41):")
    print("  1. Train delta optimization model on train set")
    print("  2. Tune hyperparameters on validation set")
    print("  3. DO NOT TOUCH test set until final evaluation")


if __name__ == '__main__':
    main()
