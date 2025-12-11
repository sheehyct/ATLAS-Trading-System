#!/usr/bin/env python3
"""
Bars-to-Magnitude Analysis Script

Session 83K-53: Analyze bars_to_magnitude data from validation results.

Analyzes:
1. Bars-to-magnitude by pattern type
2. Bars-to-magnitude by timeframe
3. Bars-to-magnitude by VIX bucket
4. Cross-tabulation: pattern x timeframe x VIX

Usage:
    uv run python scripts/analyze_bars_to_magnitude.py
    uv run python scripts/analyze_bars_to_magnitude.py --results-dir validation_results/session_83k
    uv run python scripts/analyze_bars_to_magnitude.py --output-format json
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.strat_validator import get_ticker_category, TICKER_CATEGORIES


def load_trade_csvs(results_dir: Path) -> pd.DataFrame:
    """
    Load and merge all trade CSVs from results directory.

    Parameters
    ----------
    results_dir : Path
        Directory containing trade CSV files

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all trades
    """
    all_trades = []

    # Look for CSV files matching validation output pattern
    csv_patterns = [
        'strat_validation_*.csv',
        '*_trades.csv',
        'trades_*.csv',
    ]

    for pattern in csv_patterns:
        for csv_file in results_dir.glob(f'**/{pattern}'):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_trades.append(df)
                print(f"Loaded: {csv_file.name} ({len(df)} trades)")
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

    if not all_trades:
        # Also check scripts/ directory for validation output
        scripts_dir = results_dir.parent / 'scripts' if 'validation' in str(results_dir) else Path('scripts')
        for csv_file in scripts_dir.glob('strat_validation_*.csv'):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_trades.append(df)
                print(f"Loaded: {csv_file.name} ({len(df)} trades)")
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

    if not all_trades:
        raise FileNotFoundError(f"No trade CSV files found in {results_dir}")

    merged = pd.concat(all_trades, ignore_index=True)
    print(f"\nTotal trades loaded: {len(merged)}")
    return merged


def analyze_by_pattern(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate bars_to_magnitude statistics by pattern type.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with bars_to_magnitude column

    Returns
    -------
    Dict
        Statistics per pattern type
    """
    stats = {}

    # Filter to TARGET exits only (where bars_to_magnitude is meaningful)
    if 'exit_type' in trades.columns:
        target_trades = trades[trades['exit_type'] == 'TARGET'].copy()
    elif 'magnitude_hit' in trades.columns:
        target_trades = trades[trades['magnitude_hit'] == True].copy()
    else:
        target_trades = trades.copy()

    if 'bars_to_magnitude' not in target_trades.columns:
        print("Warning: bars_to_magnitude column not found, using days_held as proxy")
        if 'days_held' in target_trades.columns:
            target_trades['bars_to_magnitude'] = target_trades['days_held']
        else:
            return stats

    pattern_col = 'pattern_type' if 'pattern_type' in trades.columns else 'pattern'

    for pattern in target_trades[pattern_col].unique():
        subset = target_trades[target_trades[pattern_col] == pattern]
        btm = subset['bars_to_magnitude'].dropna()

        if len(btm) > 0:
            stats[pattern] = {
                'count': len(btm),
                'mean_bars': round(btm.mean(), 2),
                'median_bars': round(btm.median(), 2),
                'std_bars': round(btm.std(), 2),
                'max_bars': int(btm.max()),
                'p75_bars': round(btm.quantile(0.75), 2),
                'p90_bars': round(btm.quantile(0.90), 2),
            }

    return stats


def analyze_by_timeframe(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate bars_to_magnitude statistics by timeframe.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with bars_to_magnitude column

    Returns
    -------
    Dict
        Statistics per timeframe
    """
    stats = {}

    # Filter to TARGET exits
    if 'exit_type' in trades.columns:
        target_trades = trades[trades['exit_type'] == 'TARGET'].copy()
    elif 'magnitude_hit' in trades.columns:
        target_trades = trades[trades['magnitude_hit'] == True].copy()
    else:
        target_trades = trades.copy()

    if 'bars_to_magnitude' not in target_trades.columns:
        if 'days_held' in target_trades.columns:
            target_trades['bars_to_magnitude'] = target_trades['days_held']
        else:
            return stats

    tf_col = 'detection_timeframe' if 'detection_timeframe' in trades.columns else 'timeframe'

    for tf in target_trades[tf_col].unique():
        subset = target_trades[target_trades[tf_col] == tf]
        btm = subset['bars_to_magnitude'].dropna()

        if len(btm) > 0:
            stats[tf] = {
                'count': len(btm),
                'mean_bars': round(btm.mean(), 2),
                'median_bars': round(btm.median(), 2),
                'std_bars': round(btm.std(), 2),
                'max_bars': int(btm.max()),
                'p75_bars': round(btm.quantile(0.75), 2),
                'p90_bars': round(btm.quantile(0.90), 2),
            }

    return stats


def analyze_by_vix_bucket(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate bars_to_magnitude statistics by VIX bucket.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with bars_to_magnitude and vix_bucket columns

    Returns
    -------
    Dict
        Statistics per VIX bucket
    """
    stats = {}

    if 'vix_bucket' not in trades.columns and 'vix_bucket_name' not in trades.columns:
        print("Warning: No VIX data in trades")
        return stats

    # Filter to TARGET exits
    if 'exit_type' in trades.columns:
        target_trades = trades[trades['exit_type'] == 'TARGET'].copy()
    elif 'magnitude_hit' in trades.columns:
        target_trades = trades[trades['magnitude_hit'] == True].copy()
    else:
        target_trades = trades.copy()

    if 'bars_to_magnitude' not in target_trades.columns:
        if 'days_held' in target_trades.columns:
            target_trades['bars_to_magnitude'] = target_trades['days_held']
        else:
            return stats

    vix_col = 'vix_bucket_name' if 'vix_bucket_name' in trades.columns else 'vix_bucket'

    for bucket in target_trades[vix_col].unique():
        if pd.isna(bucket) or bucket == 'UNKNOWN' or bucket == 0:
            continue

        subset = target_trades[target_trades[vix_col] == bucket]
        btm = subset['bars_to_magnitude'].dropna()

        if len(btm) > 0:
            stats[str(bucket)] = {
                'count': len(btm),
                'mean_bars': round(btm.mean(), 2),
                'median_bars': round(btm.median(), 2),
                'std_bars': round(btm.std(), 2),
                'vix_correlation_hypothesis': 'faster' if btm.mean() < 2 else 'slower',
            }

    return stats


def generate_dte_recommendations(pattern_stats: Dict, tf_stats: Dict) -> Dict[str, Any]:
    """
    Generate DTE recommendations based on bars-to-magnitude data.

    Parameters
    ----------
    pattern_stats : Dict
        Statistics by pattern
    tf_stats : Dict
        Statistics by timeframe

    Returns
    -------
    Dict
        DTE recommendations
    """
    recommendations = {}

    # Base DTE by timeframe (current defaults)
    base_dte = {
        '1H': 3,
        '1D': 21,
        '1W': 35,
        '1M': 75,
    }

    # Timeframe multipliers (bars to days)
    tf_multiplier = {
        '1H': 1/6.5,  # 6.5 market hours per day
        '1D': 1,
        '1W': 5,      # 5 trading days per week
        '1M': 21,     # ~21 trading days per month
    }

    for tf, stats in tf_stats.items():
        if tf not in base_dte:
            continue

        mean_bars = stats['mean_bars']
        p90_bars = stats['p90_bars']
        multiplier = tf_multiplier.get(tf, 1)

        # Calculate days to magnitude
        mean_days = mean_bars * multiplier
        p90_days = p90_bars * multiplier

        # Recommended DTE = P90 days + buffer
        recommended_dte = int(p90_days + max(7, p90_days * 0.3))

        recommendations[tf] = {
            'current_dte': base_dte[tf],
            'mean_days_to_magnitude': round(mean_days, 1),
            'p90_days_to_magnitude': round(p90_days, 1),
            'recommended_dte': recommended_dte,
            'assessment': 'OK' if base_dte[tf] >= recommended_dte else 'INCREASE DTE',
        }

    return recommendations


def print_report(
    pattern_stats: Dict,
    tf_stats: Dict,
    vix_stats: Dict,
    dte_recommendations: Dict,
):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("BARS-TO-MAGNITUDE ANALYSIS REPORT")
    print("Session 83K-53")
    print("=" * 70)

    # By Pattern
    print("\n--- BY PATTERN TYPE ---")
    print(f"{'Pattern':<15} {'Count':<8} {'Mean':<8} {'Median':<8} {'P90':<8} {'Max':<8}")
    print("-" * 55)
    for pattern, stats in sorted(pattern_stats.items()):
        print(f"{pattern:<15} {stats['count']:<8} {stats['mean_bars']:<8} {stats['median_bars']:<8} {stats['p90_bars']:<8} {stats['max_bars']:<8}")

    # By Timeframe
    print("\n--- BY TIMEFRAME ---")
    print(f"{'TF':<8} {'Count':<8} {'Mean':<8} {'Median':<8} {'P90':<8} {'Max':<8}")
    print("-" * 48)
    for tf, stats in sorted(tf_stats.items()):
        print(f"{tf:<8} {stats['count']:<8} {stats['mean_bars']:<8} {stats['median_bars']:<8} {stats['p90_bars']:<8} {stats['max_bars']:<8}")

    # By VIX Bucket
    if vix_stats:
        print("\n--- BY VIX BUCKET ---")
        print(f"{'VIX':<12} {'Count':<8} {'Mean':<8} {'Median':<8} {'Hypothesis':<12}")
        print("-" * 48)
        for bucket, stats in sorted(vix_stats.items()):
            print(f"{bucket:<12} {stats['count']:<8} {stats['mean_bars']:<8} {stats['median_bars']:<8} {stats['vix_correlation_hypothesis']:<12}")

    # DTE Recommendations
    if dte_recommendations:
        print("\n--- DTE RECOMMENDATIONS ---")
        print(f"{'TF':<8} {'Current':<10} {'Mean Days':<12} {'P90 Days':<12} {'Recommended':<12} {'Status':<10}")
        print("-" * 64)
        for tf, rec in sorted(dte_recommendations.items()):
            print(f"{tf:<8} {rec['current_dte']:<10} {rec['mean_days_to_magnitude']:<12} {rec['p90_days_to_magnitude']:<12} {rec['recommended_dte']:<12} {rec['assessment']:<10}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Analyze bars-to-magnitude data')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='validation_results',
        help='Directory containing validation results'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load trades
    try:
        trades = load_trade_csvs(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTip: Run validation first:")
        print("  uv run python scripts/run_atlas_validation_83k.py")
        sys.exit(1)

    # Run analyses
    pattern_stats = analyze_by_pattern(trades)
    tf_stats = analyze_by_timeframe(trades)
    vix_stats = analyze_by_vix_bucket(trades)
    dte_recommendations = generate_dte_recommendations(pattern_stats, tf_stats)

    if args.output_format == 'json':
        output = {
            'by_pattern': pattern_stats,
            'by_timeframe': tf_stats,
            'by_vix_bucket': vix_stats,
            'dte_recommendations': dte_recommendations,
            'total_trades': len(trades),
        }
        json_str = json.dumps(output, indent=2)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(json_str)
            print(f"Results saved to: {args.output_file}")
        else:
            print(json_str)
    else:
        print_report(pattern_stats, tf_stats, vix_stats, dte_recommendations)

        if args.output_file:
            # Save JSON anyway for programmatic use
            output = {
                'by_pattern': pattern_stats,
                'by_timeframe': tf_stats,
                'by_vix_bucket': vix_stats,
                'dte_recommendations': dte_recommendations,
            }
            with open(args.output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults also saved to: {args.output_file}")


if __name__ == '__main__':
    main()
