#!/usr/bin/env python3
"""
Cross-Instrument STRAT Pattern Comparison

Session 83K-53: Compare pattern performance across different instrument types.

Analyzes:
1. Performance by ticker category (index ETF, sector ETF, mega cap, etc.)
2. Bars-to-magnitude differences across instrument types
3. Beta classification analysis (high-beta vs low-beta behavior)

Usage:
    uv run python scripts/analyze_cross_instrument.py
    uv run python scripts/analyze_cross_instrument.py --results-dir validation_results
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


# Beta classification (approximate based on typical behavior)
BETA_CLASSIFICATION = {
    'high_beta': ['QQQ', 'IWM', 'TQQQ', 'SOXL', 'ARKK', 'TSLA', 'NVDA', 'META', 'XLK'],
    'medium_beta': ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'MDY', 'XLF', 'XLE', 'XLB', 'XLI'],
    'low_beta': ['DIA', 'XLV', 'XLU'],
}


def get_beta_class(symbol: str) -> str:
    """Return beta classification for a symbol."""
    for beta_class, symbols in BETA_CLASSIFICATION.items():
        if symbol in symbols:
            return beta_class
    return 'medium_beta'


def load_trade_csvs(results_dir: Path) -> pd.DataFrame:
    """Load and merge all trade CSVs from results directory."""
    all_trades = []

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


def add_classifications(trades: pd.DataFrame) -> pd.DataFrame:
    """Add ticker category and beta classification to trades."""
    df = trades.copy()

    symbol_col = 'symbol' if 'symbol' in df.columns else 'ticker'

    if symbol_col in df.columns:
        df['ticker_category'] = df[symbol_col].apply(get_ticker_category)
        df['beta_class'] = df[symbol_col].apply(get_beta_class)
    else:
        df['ticker_category'] = 'unknown'
        df['beta_class'] = 'unknown'

    return df


def analyze_by_category(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare performance across ticker categories.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with ticker_category column

    Returns
    -------
    Dict
        Performance statistics per category
    """
    stats = {}

    for category in trades['ticker_category'].unique():
        if pd.isna(category) or category == 'unknown' or category == 'other':
            continue

        subset = trades[trades['ticker_category'] == category]

        # Calculate performance metrics
        total_trades = len(subset)
        if total_trades == 0:
            continue

        # Win rate (if pnl or magnitude_hit available)
        if 'pnl' in subset.columns:
            wins = (subset['pnl'] > 0).sum()
            win_rate = wins / total_trades
            total_pnl = subset['pnl'].sum()
            avg_pnl = subset['pnl'].mean()
        elif 'magnitude_hit' in subset.columns:
            wins = subset['magnitude_hit'].sum()
            win_rate = wins / total_trades
            total_pnl = None
            avg_pnl = None
        else:
            win_rate = None
            total_pnl = None
            avg_pnl = None

        # Bars to magnitude (if available)
        if 'bars_to_magnitude' in subset.columns:
            btm_subset = subset[subset['magnitude_hit'] == True] if 'magnitude_hit' in subset.columns else subset
            btm = btm_subset['bars_to_magnitude'].dropna()
            mean_btm = btm.mean() if len(btm) > 0 else None
            median_btm = btm.median() if len(btm) > 0 else None
        elif 'days_held' in subset.columns:
            btm_subset = subset[subset['magnitude_hit'] == True] if 'magnitude_hit' in subset.columns else subset
            btm = btm_subset['days_held'].dropna()
            mean_btm = btm.mean() if len(btm) > 0 else None
            median_btm = btm.median() if len(btm) > 0 else None
        else:
            mean_btm = None
            median_btm = None

        stats[category] = {
            'trade_count': total_trades,
            'win_rate': round(win_rate, 3) if win_rate is not None else None,
            'total_pnl': round(total_pnl, 2) if total_pnl is not None else None,
            'avg_pnl': round(avg_pnl, 2) if avg_pnl is not None else None,
            'mean_bars_to_magnitude': round(mean_btm, 2) if mean_btm is not None else None,
            'median_bars_to_magnitude': round(median_btm, 2) if median_btm is not None else None,
            'symbols': list(subset['symbol'].unique()) if 'symbol' in subset.columns else [],
        }

    return stats


def analyze_by_beta(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare performance by beta classification.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with beta_class column

    Returns
    -------
    Dict
        Performance statistics per beta class
    """
    stats = {}

    for beta_class in ['high_beta', 'medium_beta', 'low_beta']:
        subset = trades[trades['beta_class'] == beta_class]

        if len(subset) == 0:
            continue

        total_trades = len(subset)

        # Win rate
        if 'pnl' in subset.columns:
            wins = (subset['pnl'] > 0).sum()
            win_rate = wins / total_trades
            avg_pnl = subset['pnl'].mean()
        elif 'magnitude_hit' in subset.columns:
            wins = subset['magnitude_hit'].sum()
            win_rate = wins / total_trades
            avg_pnl = None
        else:
            win_rate = None
            avg_pnl = None

        # Bars to magnitude
        if 'bars_to_magnitude' in subset.columns:
            btm_subset = subset[subset['magnitude_hit'] == True] if 'magnitude_hit' in subset.columns else subset
            btm = btm_subset['bars_to_magnitude'].dropna()
            mean_btm = btm.mean() if len(btm) > 0 else None
        elif 'days_held' in subset.columns:
            btm_subset = subset[subset['magnitude_hit'] == True] if 'magnitude_hit' in subset.columns else subset
            btm = btm_subset['days_held'].dropna()
            mean_btm = btm.mean() if len(btm) > 0 else None
        else:
            mean_btm = None

        stats[beta_class] = {
            'trade_count': total_trades,
            'win_rate': round(win_rate, 3) if win_rate is not None else None,
            'avg_pnl': round(avg_pnl, 2) if avg_pnl is not None else None,
            'mean_bars_to_magnitude': round(mean_btm, 2) if mean_btm is not None else None,
        }

    return stats


def analyze_pattern_by_category(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Cross-tabulation: Pattern performance by ticker category.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data

    Returns
    -------
    Dict
        Nested dict: category -> pattern -> stats
    """
    stats = {}

    pattern_col = 'pattern_type' if 'pattern_type' in trades.columns else 'pattern'

    for category in trades['ticker_category'].unique():
        if pd.isna(category) or category == 'unknown' or category == 'other':
            continue

        cat_subset = trades[trades['ticker_category'] == category]
        stats[category] = {}

        for pattern in cat_subset[pattern_col].unique():
            subset = cat_subset[cat_subset[pattern_col] == pattern]

            if len(subset) < 5:  # Need minimum trades for meaningful stats
                continue

            if 'magnitude_hit' in subset.columns:
                win_rate = subset['magnitude_hit'].mean()
            elif 'pnl' in subset.columns:
                win_rate = (subset['pnl'] > 0).mean()
            else:
                win_rate = None

            stats[category][pattern] = {
                'trade_count': len(subset),
                'win_rate': round(win_rate, 3) if win_rate is not None else None,
            }

    return stats


def print_report(
    category_stats: Dict,
    beta_stats: Dict,
    pattern_by_category: Dict,
):
    """Print formatted cross-instrument report."""
    print("\n" + "=" * 70)
    print("CROSS-INSTRUMENT COMPARISON REPORT")
    print("Session 83K-53")
    print("=" * 70)

    # By Ticker Category
    print("\n--- BY TICKER CATEGORY ---")
    print(f"{'Category':<15} {'Trades':<10} {'Win Rate':<10} {'Avg P&L':<12} {'Mean BTM':<10}")
    print("-" * 57)
    for category, stats in sorted(category_stats.items()):
        wr = f"{stats['win_rate']:.1%}" if stats['win_rate'] is not None else "N/A"
        pnl = f"${stats['avg_pnl']:.0f}" if stats['avg_pnl'] is not None else "N/A"
        btm = f"{stats['mean_bars_to_magnitude']:.1f}" if stats['mean_bars_to_magnitude'] is not None else "N/A"
        print(f"{category:<15} {stats['trade_count']:<10} {wr:<10} {pnl:<12} {btm:<10}")

    # By Beta Classification
    print("\n--- BY BETA CLASSIFICATION ---")
    print(f"{'Beta Class':<15} {'Trades':<10} {'Win Rate':<10} {'Avg P&L':<12} {'Mean BTM':<10}")
    print("-" * 57)
    for beta_class, stats in sorted(beta_stats.items()):
        wr = f"{stats['win_rate']:.1%}" if stats['win_rate'] is not None else "N/A"
        pnl = f"${stats['avg_pnl']:.0f}" if stats['avg_pnl'] is not None else "N/A"
        btm = f"{stats['mean_bars_to_magnitude']:.1f}" if stats['mean_bars_to_magnitude'] is not None else "N/A"
        print(f"{beta_class:<15} {stats['trade_count']:<10} {wr:<10} {pnl:<12} {btm:<10}")

    # Key Insights
    print("\n--- KEY INSIGHTS ---")

    # Best performing category
    best_cat = None
    best_wr = 0
    for cat, stats in category_stats.items():
        if stats['win_rate'] is not None and stats['win_rate'] > best_wr:
            best_wr = stats['win_rate']
            best_cat = cat
    if best_cat:
        print(f"Best category: {best_cat} ({best_wr:.1%} win rate)")

    # Fastest moves (lowest BTM)
    fastest_cat = None
    lowest_btm = float('inf')
    for cat, stats in category_stats.items():
        if stats['mean_bars_to_magnitude'] is not None and stats['mean_bars_to_magnitude'] < lowest_btm:
            lowest_btm = stats['mean_bars_to_magnitude']
            fastest_cat = cat
    if fastest_cat:
        print(f"Fastest moves: {fastest_cat} ({lowest_btm:.1f} bars to magnitude)")

    # Beta comparison
    if 'high_beta' in beta_stats and 'low_beta' in beta_stats:
        high_btm = beta_stats['high_beta'].get('mean_bars_to_magnitude')
        low_btm = beta_stats['low_beta'].get('mean_bars_to_magnitude')
        if high_btm is not None and low_btm is not None:
            if high_btm < low_btm:
                print(f"High-beta moves faster: {high_btm:.1f} vs {low_btm:.1f} bars")
            else:
                print(f"Low-beta moves faster: {low_btm:.1f} vs {high_btm:.1f} bars")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Cross-instrument comparison analysis')
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
        print("  uv run python scripts/run_atlas_validation_83k.py --universe expanded")
        sys.exit(1)

    # Add classifications
    trades = add_classifications(trades)

    # Run analyses
    category_stats = analyze_by_category(trades)
    beta_stats = analyze_by_beta(trades)
    pattern_by_category = analyze_pattern_by_category(trades)

    if args.output_format == 'json':
        output = {
            'by_category': category_stats,
            'by_beta': beta_stats,
            'pattern_by_category': pattern_by_category,
            'total_trades': len(trades),
            'unique_symbols': list(trades['symbol'].unique()) if 'symbol' in trades.columns else [],
        }
        json_str = json.dumps(output, indent=2)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(json_str)
            print(f"Results saved to: {args.output_file}")
        else:
            print(json_str)
    else:
        print_report(category_stats, beta_stats, pattern_by_category)

        if args.output_file:
            output = {
                'by_category': category_stats,
                'by_beta': beta_stats,
                'pattern_by_category': pattern_by_category,
            }
            with open(args.output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nResults also saved to: {args.output_file}")


if __name__ == '__main__':
    main()
