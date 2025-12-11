#!/usr/bin/env python3
"""
Pattern Magnitude Analysis Script for STRAT Options Trading

Session 83K-26: Analyze magnitude distribution across patterns/symbols
to inform DTE and delta selection rules.

Key insight: Small magnitude patterns (<0.3%) may not be profitable for options
due to theta decay exceeding delta gains.

Usage:
    uv run python scripts/analyze_pattern_magnitudes.py
    uv run python scripts/analyze_pattern_magnitudes.py --patterns 3-1-2 2-1-2
    uv run python scripts/analyze_pattern_magnitudes.py --symbols SPY QQQ
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fetch_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data using cache (primary), Alpaca, or Tiingo (fallback)."""
    import os

    # Try cache first
    cache_path = project_root / 'data_cache' / f'{symbol}_1D.parquet'
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                logger.debug(f"Loaded {symbol} from cache")
                return df
        except Exception as e:
            logger.debug(f"Cache read failed for {symbol}: {e}")

    # Try Alpaca
    try:
        import vectorbtpro as vbt
        alpaca_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
        alpaca_secret = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')

        if alpaca_key and alpaca_secret:
            data = vbt.AlpacaData.pull(
                symbol,
                start=start_date,
                end=end_date,
                timeframe='1 day',
                adjustment='split'
            )
            df = data.get()
            if not df.empty:
                return df
    except Exception as e:
        logger.debug(f"Alpaca failed for {symbol}: {e}")

    # Fallback to Tiingo
    try:
        from integrations.tiingo_data_fetcher import TiingoDataFetcher
        fetcher = TiingoDataFetcher()
        df = fetcher.fetch_daily_data(symbol, start_date, end_date)
        if not df.empty:
            return df
    except Exception as e:
        logger.debug(f"Tiingo failed for {symbol}: {e}")

    raise ValueError(f"Could not fetch data for {symbol}")


def detect_patterns(data: pd.DataFrame, pattern_types: List[str]) -> pd.DataFrame:
    """Detect STRAT patterns and calculate magnitudes."""
    from strat.tier1_detector import Tier1Detector

    detector = Tier1Detector(
        min_continuation_bars=0,  # Get ALL patterns, no filtering
    )

    try:
        # Convert pattern names: '3-1-2' -> '312', '2-1-2' -> '212', '2-2' -> '22'
        internal_patterns = []
        for p in pattern_types:
            internal = p.replace('-', '')
            internal_patterns.append(internal)

        # Detect all patterns at once
        signals = detector.detect_patterns(data, pattern_types=internal_patterns)

        if not signals:
            return pd.DataFrame()

        # Convert PatternSignal objects to DataFrame
        pattern_data = []
        for signal in signals:
            pattern_data.append({
                'timestamp': signal.timestamp,
                'pattern_type': signal.pattern_type,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_price': signal.stop_price,
                'target_price': signal.target_price,
                'continuation_bars': signal.continuation_bars,
                'is_filtered': signal.is_filtered,
                'risk_reward': signal.risk_reward
            })

        return pd.DataFrame(pattern_data)

    except Exception as e:
        logger.warning(f"Error detecting patterns: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def calculate_magnitude_stats(patterns: pd.DataFrame) -> Dict[str, Any]:
    """Calculate magnitude statistics for patterns."""
    if patterns.empty:
        return {}

    # Calculate magnitude percentage
    patterns = patterns.copy()
    patterns['magnitude_pct'] = abs(patterns['target_price'] - patterns['entry_price']) / patterns['entry_price'] * 100

    # Bucket by magnitude
    patterns['mag_bucket'] = pd.cut(
        patterns['magnitude_pct'],
        bins=[0, 0.2, 0.3, 0.5, 1.0, 2.0, 100],
        labels=['<0.2%', '0.2-0.3%', '0.3-0.5%', '0.5-1%', '1-2%', '>2%']
    )

    return patterns


def analyze_symbol(symbol: str, pattern_types: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Analyze patterns for a single symbol."""
    try:
        data = fetch_price_data(symbol, start_date, end_date)
        patterns = detect_patterns(data, pattern_types)

        if patterns.empty:
            return pd.DataFrame()

        patterns = calculate_magnitude_stats(patterns)
        patterns['symbol'] = symbol

        return patterns

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return pd.DataFrame()


def print_magnitude_report(all_patterns: pd.DataFrame):
    """Print comprehensive magnitude analysis report."""
    if all_patterns.empty:
        print("No patterns found!")
        return

    print()
    print("=" * 80)
    print("STRAT PATTERN MAGNITUDE ANALYSIS")
    print("=" * 80)
    print(f"Total Patterns: {len(all_patterns)}")
    print(f"Date Range: {all_patterns['timestamp'].min()} to {all_patterns['timestamp'].max()}")
    print()

    # Overall magnitude distribution
    print("-" * 80)
    print("MAGNITUDE DISTRIBUTION (ALL PATTERNS)")
    print("-" * 80)

    mag_dist = all_patterns.groupby('mag_bucket', observed=True).agg({
        'magnitude_pct': ['count', 'mean']
    }).round(3)
    mag_dist.columns = ['Count', 'Avg_Mag%']
    mag_dist['Pct_of_Total'] = (mag_dist['Count'] / len(all_patterns) * 100).round(1)
    print(mag_dist.to_string())
    print()

    # By pattern type
    print("-" * 80)
    print("BY PATTERN TYPE")
    print("-" * 80)

    for pattern in all_patterns['pattern_type'].unique():
        pdata = all_patterns[all_patterns['pattern_type'] == pattern]
        print(f"\n{pattern} ({len(pdata)} patterns):")
        print(f"  Mean Magnitude: {pdata['magnitude_pct'].mean():.3f}%")
        print(f"  Median Magnitude: {pdata['magnitude_pct'].median():.3f}%")
        print(f"  Min/Max: {pdata['magnitude_pct'].min():.3f}% / {pdata['magnitude_pct'].max():.3f}%")

        # Small magnitude count
        small = (pdata['magnitude_pct'] < 0.3).sum()
        print(f"  Patterns < 0.3%: {small} ({small/len(pdata)*100:.1f}%)")

    print()

    # By symbol
    print("-" * 80)
    print("BY SYMBOL")
    print("-" * 80)

    symbol_stats = all_patterns.groupby('symbol').agg({
        'magnitude_pct': ['count', 'mean', 'median'],
        'pattern_type': lambda x: x.value_counts().to_dict()
    }).round(3)

    for symbol in all_patterns['symbol'].unique():
        sdata = all_patterns[all_patterns['symbol'] == symbol]
        print(f"\n{symbol} ({len(sdata)} patterns):")
        print(f"  Mean Magnitude: {sdata['magnitude_pct'].mean():.3f}%")
        print(f"  Median Magnitude: {sdata['magnitude_pct'].median():.3f}%")
        small = (sdata['magnitude_pct'] < 0.3).sum()
        print(f"  Patterns < 0.3%: {small} ({small/len(sdata)*100:.1f}%)")

    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS FOR OPTIONS TRADING")
    print("=" * 80)

    small_mag_pct = (all_patterns['magnitude_pct'] < 0.3).sum() / len(all_patterns) * 100

    print(f"\n1. MAGNITUDE FILTER:")
    print(f"   {small_mag_pct:.1f}% of patterns have magnitude < 0.3%")
    if small_mag_pct > 20:
        print(f"   RECOMMENDATION: Filter out patterns with magnitude < 0.3%")
        print(f"   This would remove {(all_patterns['magnitude_pct'] < 0.3).sum()} patterns")
    else:
        print(f"   Small magnitude patterns are rare - filtering optional")

    print(f"\n2. DTE SELECTION BY MAGNITUDE:")
    print(f"   Suggested rules:")
    print(f"   - Magnitude >= 1.0%: DTE 21-45 days (standard)")
    print(f"   - Magnitude 0.5-1.0%: DTE 14-21 days")
    print(f"   - Magnitude 0.3-0.5%: DTE 7-14 days (faster theta decay)")
    print(f"   - Magnitude < 0.3%: SKIP or same-week expiry only")

    print(f"\n3. DELTA SELECTION BY MAGNITUDE:")
    print(f"   Suggested rules:")
    print(f"   - Magnitude >= 1.0%: Delta 0.40-0.60 (OTM okay)")
    print(f"   - Magnitude 0.5-1.0%: Delta 0.50-0.70 (ATM preferred)")
    print(f"   - Magnitude 0.3-0.5%: Delta 0.60-0.80 (ITM required)")
    print(f"   - Magnitude < 0.3%: Delta 0.70+ or SKIP")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze STRAT pattern magnitudes")
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'NVDA'],
                        help='Symbols to analyze')
    parser.add_argument('--patterns', nargs='+', default=['3-1-2', '2-1-2', '2-2'],
                        help='Pattern types to analyze')
    parser.add_argument('--start', default='2020-01-01', help='Start date')
    parser.add_argument('--end', default='2025-12-01', help='End date')
    parser.add_argument('--csv', help='Export results to CSV')

    args = parser.parse_args()

    print(f"Analyzing patterns: {args.patterns}")
    print(f"Symbols: {args.symbols}")
    print(f"Period: {args.start} to {args.end}")
    print()

    all_patterns = []

    for symbol in args.symbols:
        print(f"Processing {symbol}...", end=" ", flush=True)
        patterns = analyze_symbol(symbol, args.patterns, args.start, args.end)
        if not patterns.empty:
            all_patterns.append(patterns)
            print(f"found {len(patterns)} patterns")
        else:
            print("no patterns found")

    if not all_patterns:
        print("No patterns found across any symbols!")
        return

    combined = pd.concat(all_patterns, ignore_index=True)

    # Print report
    print_magnitude_report(combined)

    # Export to CSV if requested
    if args.csv:
        output_path = Path(args.csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"\nResults exported to: {output_path}")


if __name__ == '__main__':
    main()
