"""
Session 73: Delta Accuracy Validation Script

Validates that the data-driven strike selection achieves 60%+ delta accuracy
(strikes in the optimal 0.50-0.80 delta range).

Baseline: 20.9% with old 0.3x geometric formula
Target: 60%+ with new delta-targeting algorithm

Usage:
    uv run python scripts/validate_delta_accuracy.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from strat.options_module import OptionsExecutor, OptionType
from strat.tier1_detector import Tier1Detector, PatternSignal, Timeframe
from strat.greeks import calculate_greeks
from integrations.tiingo_data_fetcher import TiingoDataFetcher


def calculate_strike_delta(
    strike: float,
    underlying_price: float,
    option_type: str,
    dte: int = 35,
    iv: float = 0.20,
    r: float = 0.05
) -> float:
    """Calculate the delta for a given strike."""
    T = dte / 365.0
    greeks = calculate_greeks(
        S=underlying_price,
        K=strike,
        T=T,
        r=r,
        sigma=iv,
        option_type=option_type
    )
    return abs(greeks.delta)


def geometric_strike_selection(
    entry: float,
    target: float,
    is_call: bool,
    underlying_price: float
) -> float:
    """Original 0.3x geometric formula for baseline comparison."""
    if is_call:
        strike = entry + (0.3 * (target - entry))
    else:
        strike = entry - (0.3 * (entry - target))

    # Round to standard interval
    if underlying_price < 100:
        interval = 1.0
    elif underlying_price < 500:
        interval = 5.0
    else:
        interval = 10.0

    return round(strike / interval) * interval


def validate_delta_accuracy(
    symbols: List[str] = None,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31',
    delta_range: Tuple[float, float] = (0.50, 0.80)
) -> Dict:
    """
    Validate delta accuracy of the new data-driven strike selection.

    Args:
        symbols: List of symbols to test (default: SPY, QQQ, IWM)
        start_date: Start date for pattern detection
        end_date: End date for pattern detection
        delta_range: Target delta range (default 0.50-0.80)

    Returns:
        Dictionary with validation results
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']

    print("=" * 70)
    print("Session 73: Delta Accuracy Validation")
    print("=" * 70)
    print(f"\nTarget: 60%+ of strikes in delta range {delta_range}")
    print(f"Baseline: 20.9% (old 0.3x geometric formula)")
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Date range: {start_date} to {end_date}")
    print("-" * 70)

    # Initialize components
    fetcher = TiingoDataFetcher()
    detector = Tier1Detector()
    executor = OptionsExecutor()

    # Collect all results
    all_results = []

    for symbol in symbols:
        print(f"\n[{symbol}] Fetching data and detecting patterns...")

        try:
            # Fetch weekly data
            data = fetcher.fetch(symbol, start_date=start_date, end_date=end_date, timeframe='1W')
            df = data.get()

            if df is None or len(df) < 20:
                print(f"  -> Skipped: Insufficient data")
                continue

            # Detect Tier 1 patterns
            signals = detector.detect_patterns(df, timeframe=Timeframe.WEEKLY)

            if not signals:
                print(f"  -> No patterns detected")
                continue

            print(f"  -> {len(signals)} patterns detected")

            # Calculate IV from historical data
            close_col = 'close' if 'close' in df.columns else 'Close'
            returns = df[close_col].pct_change().dropna()
            iv = returns.std() * np.sqrt(252)  # Annualized
            iv = max(0.10, min(iv, 1.0))  # Clamp

            # Process each signal
            for signal in signals:
                # CRITICAL: Use entry price as underlying price (trade execution time)
                underlying_price = signal.entry_price
                is_call = signal.direction == 1
                option_type = 'call' if is_call else 'put'

                # Get entry/target
                entry = signal.entry_price
                target = signal.target_price

                # Method 1: Old geometric formula (baseline)
                old_strike = geometric_strike_selection(entry, target, is_call, underlying_price)
                old_delta = calculate_strike_delta(old_strike, underlying_price, option_type, iv=iv)

                # Method 2: New data-driven selection
                new_strike, new_delta, _ = executor._select_strike(
                    signal=signal,
                    underlying_price=underlying_price,
                    option_type=OptionType.CALL if is_call else OptionType.PUT,
                    iv=iv,
                    dte=35
                )

                # If new_delta is None (fallback was used), calculate it
                if new_delta is None:
                    new_delta = calculate_strike_delta(new_strike, underlying_price, option_type, iv=iv)
                else:
                    new_delta = abs(new_delta)

                # Check if in range
                old_in_range = delta_range[0] <= old_delta <= delta_range[1]
                new_in_range = delta_range[0] <= new_delta <= delta_range[1]

                all_results.append({
                    'symbol': symbol,
                    'timestamp': signal.timestamp,
                    'pattern_type': signal.pattern_type.value,
                    'direction': 'BULL' if is_call else 'BEAR',
                    'entry': entry,
                    'target': target,
                    'underlying_price': underlying_price,
                    'iv': iv,
                    'old_strike': old_strike,
                    'old_delta': old_delta,
                    'old_in_range': old_in_range,
                    'new_strike': new_strike,
                    'new_delta': new_delta,
                    'new_in_range': new_in_range,
                    'used_fallback': new_delta == old_delta  # Approximate check
                })

        except Exception as e:
            print(f"  -> Error: {e}")
            continue

    # Calculate summary statistics
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("\n[ERROR] No results to analyze!")
        return {'error': 'No patterns detected'}

    # Summary metrics
    total_patterns = len(results_df)
    old_in_range_count = results_df['old_in_range'].sum()
    new_in_range_count = results_df['new_in_range'].sum()

    old_accuracy = (old_in_range_count / total_patterns) * 100
    new_accuracy = (new_in_range_count / total_patterns) * 100
    improvement = new_accuracy - old_accuracy

    old_avg_delta = results_df['old_delta'].mean()
    new_avg_delta = results_df['new_delta'].mean()

    # Fallback rate
    fallback_count = results_df['used_fallback'].sum()
    fallback_rate = (fallback_count / total_patterns) * 100

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTotal patterns analyzed: {total_patterns}")
    print(f"\n{'Metric':<30} {'Old (0.3x)':<15} {'New (Delta)':<15} {'Target':<10}")
    print("-" * 70)
    print(f"{'Delta Accuracy':<30} {old_accuracy:>12.1f}%  {new_accuracy:>12.1f}%  {'60%+':<10}")
    print(f"{'Average Delta':<30} {old_avg_delta:>12.3f}   {new_avg_delta:>12.3f}   {'0.60-0.70':<10}")
    print(f"{'Fallback Rate':<30} {'N/A':<15} {fallback_rate:>12.1f}%  {'<20%':<10}")

    print("\n" + "-" * 70)
    print(f"Improvement: {improvement:+.1f} percentage points")

    # Success check
    target_met = new_accuracy >= 60
    print("\n" + "=" * 70)
    if target_met:
        print(f"SUCCESS: Delta accuracy {new_accuracy:.1f}% >= 60% target")
    else:
        print(f"NEEDS IMPROVEMENT: Delta accuracy {new_accuracy:.1f}% < 60% target")
    print("=" * 70)

    # Distribution by pattern type
    print("\n[Pattern Type Breakdown]")
    pattern_stats = results_df.groupby('pattern_type').agg({
        'new_in_range': ['sum', 'count'],
        'new_delta': 'mean'
    }).round(3)
    pattern_stats.columns = ['in_range', 'total', 'avg_delta']
    pattern_stats['accuracy'] = (pattern_stats['in_range'] / pattern_stats['total'] * 100).round(1)
    print(pattern_stats[['total', 'in_range', 'accuracy', 'avg_delta']].to_string())

    # Distribution by symbol
    print("\n[Symbol Breakdown]")
    symbol_stats = results_df.groupby('symbol').agg({
        'new_in_range': ['sum', 'count'],
        'new_delta': 'mean'
    }).round(3)
    symbol_stats.columns = ['in_range', 'total', 'avg_delta']
    symbol_stats['accuracy'] = (symbol_stats['in_range'] / symbol_stats['total'] * 100).round(1)
    print(symbol_stats[['total', 'in_range', 'accuracy', 'avg_delta']].to_string())

    return {
        'total_patterns': total_patterns,
        'old_accuracy': old_accuracy,
        'new_accuracy': new_accuracy,
        'improvement': improvement,
        'old_avg_delta': old_avg_delta,
        'new_avg_delta': new_avg_delta,
        'fallback_rate': fallback_rate,
        'target_met': target_met,
        'results_df': results_df
    }


if __name__ == '__main__':
    results = validate_delta_accuracy()
