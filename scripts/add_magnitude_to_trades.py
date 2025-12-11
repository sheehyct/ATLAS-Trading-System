#!/usr/bin/env python3
"""
Session 83K-29: Post-hoc Magnitude Calculation for Trade CSVs

Calculates magnitude_pct from entry_price and target_price in existing trade CSVs.
Formula: magnitude_pct = abs(target_price - entry_price) / entry_price * 100

Usage:
    python scripts/add_magnitude_to_trades.py
    python scripts/add_magnitude_to_trades.py --analyze-only
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json


def load_trade_csvs(trades_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all trade CSV files from the trades directory."""
    trades = {}
    for csv_file in trades_dir.glob("*_trades.csv"):
        try:
            df = pd.read_csv(csv_file)
            run_id = csv_file.stem.replace("_trades", "")
            trades[run_id] = df
            print(f"Loaded {run_id}: {len(df)} trades")
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    return trades


def calculate_magnitude_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add magnitude_pct column to trade DataFrame."""
    df = df.copy()

    # Calculate magnitude percentage
    # magnitude_pct = abs(target_price - entry_price) / entry_price * 100
    df['magnitude_pct'] = abs(df['target_price'] - df['entry_price']) / df['entry_price'] * 100

    # Also calculate actual move percentage for comparison
    df['actual_move_pct'] = abs(df['exit_price'] - df['entry_price']) / df['entry_price'] * 100

    # Calculate risk percentage (stop distance)
    df['risk_pct'] = abs(df['stop_price'] - df['entry_price']) / df['entry_price'] * 100

    # Calculate R:R ratio (magnitude / risk)
    df['rr_ratio'] = df['magnitude_pct'] / df['risk_pct']

    return df


def analyze_magnitude_by_pattern(trades: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze magnitude distribution by pattern type."""
    all_trades = []

    for run_id, df in trades.items():
        if len(df) == 0:
            continue

        # Parse run_id: pattern_timeframe_symbol
        parts = run_id.rsplit("_", 2)
        if len(parts) == 3:
            pattern, timeframe, symbol = parts
        else:
            pattern = run_id
            timeframe = "unknown"
            symbol = "unknown"

        df = df.copy()
        df['run_id'] = run_id
        df['pattern'] = pattern
        df['timeframe'] = timeframe
        df['underlying'] = symbol

        # Calculate magnitude if not already present
        if 'magnitude_pct' not in df.columns:
            df = calculate_magnitude_pct(df)

        all_trades.append(df)

    if not all_trades:
        return pd.DataFrame()

    combined = pd.concat(all_trades, ignore_index=True)
    return combined


def generate_pattern_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by pattern."""
    if combined.empty:
        return pd.DataFrame()

    summary = combined.groupby('pattern').agg({
        'pnl': ['count', 'sum', 'mean'],
        'magnitude_pct': ['mean', 'median', 'std', 'min', 'max'],
        'actual_move_pct': ['mean', 'median'],
        'risk_pct': ['mean', 'median'],
        'rr_ratio': ['mean', 'median'],
        'exit_type': lambda x: (x == 'TARGET').sum() / len(x) * 100  # Win rate proxy
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'pnl_count': 'trade_count',
        'pnl_sum': 'total_pnl',
        'pnl_mean': 'avg_pnl',
        'exit_type_<lambda>': 'target_hit_rate'
    })

    return summary


def generate_symbol_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by symbol."""
    if combined.empty:
        return pd.DataFrame()

    summary = combined.groupby('underlying').agg({
        'pnl': ['count', 'sum', 'mean'],
        'magnitude_pct': ['mean', 'median'],
        'actual_move_pct': ['mean'],
        'rr_ratio': ['mean'],
        'exit_type': lambda x: (x == 'TARGET').sum() / len(x) * 100
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={
        'pnl_count': 'trade_count',
        'pnl_sum': 'total_pnl',
        'pnl_mean': 'avg_pnl',
        'exit_type_<lambda>': 'target_hit_rate'
    })

    return summary


def generate_magnitude_buckets(combined: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by magnitude bucket."""
    if combined.empty:
        return pd.DataFrame()

    # Define magnitude buckets
    bins = [0, 0.3, 0.5, 1.0, 2.0, 5.0, float('inf')]
    labels = ['<0.3%', '0.3-0.5%', '0.5-1.0%', '1.0-2.0%', '2.0-5.0%', '>5.0%']

    combined['mag_bucket'] = pd.cut(combined['magnitude_pct'], bins=bins, labels=labels)

    bucket_summary = combined.groupby('mag_bucket', observed=True).agg({
        'pnl': ['count', 'sum', 'mean'],
        'exit_type': lambda x: (x == 'TARGET').sum() / len(x) * 100 if len(x) > 0 else 0
    }).round(4)

    bucket_summary.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'target_hit_rate']

    return bucket_summary


def main():
    parser = argparse.ArgumentParser(description='Calculate magnitude_pct for trade CSVs')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not modify files')
    parser.add_argument('--trades-dir', type=str, default='validation_results/session_83k/trades',
                        help='Directory containing trade CSV files')
    parser.add_argument('--output-dir', type=str, default='validation_results/session_83k/analysis',
                        help='Output directory for analysis reports')
    args = parser.parse_args()

    trades_dir = Path(args.trades_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Session 83K-29: Magnitude Analysis")
    print("=" * 70)
    print()

    # Load all trade CSVs
    print("Loading trade CSVs...")
    trades = load_trade_csvs(trades_dir)
    print(f"\nTotal files loaded: {len(trades)}")

    # Calculate magnitude for each trade
    print("\nCalculating magnitude_pct for all trades...")
    for run_id, df in trades.items():
        if len(df) > 0:
            trades[run_id] = calculate_magnitude_pct(df)

    # Combine all trades for analysis
    combined = analyze_magnitude_by_pattern(trades)
    total_trades = len(combined)
    print(f"\nTotal trades across all runs: {total_trades}")

    if total_trades == 0:
        print("No trades to analyze.")
        return

    # Generate summaries
    print("\n" + "=" * 70)
    print("MAGNITUDE ANALYSIS RESULTS")
    print("=" * 70)

    # 1. Pattern summary
    print("\n--- By Pattern ---")
    pattern_summary = generate_pattern_summary(combined)
    print(pattern_summary.to_string())
    pattern_summary.to_csv(output_dir / 'pattern_magnitude_summary.csv')

    # 2. Symbol summary
    print("\n--- By Symbol ---")
    symbol_summary = generate_symbol_summary(combined)
    print(symbol_summary.to_string())
    symbol_summary.to_csv(output_dir / 'symbol_magnitude_summary.csv')

    # 3. Magnitude bucket analysis
    print("\n--- By Magnitude Bucket ---")
    bucket_summary = generate_magnitude_buckets(combined)
    print(bucket_summary.to_string())
    bucket_summary.to_csv(output_dir / 'magnitude_bucket_summary.csv')

    # 4. Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Total Trades: {total_trades}")
    print(f"Mean Magnitude: {combined['magnitude_pct'].mean():.4f}%")
    print(f"Median Magnitude: {combined['magnitude_pct'].median():.4f}%")
    print(f"Mean R:R Ratio: {combined['rr_ratio'].mean():.4f}")
    print(f"Median R:R Ratio: {combined['rr_ratio'].median():.4f}")
    print(f"Target Hit Rate: {(combined['exit_type'] == 'TARGET').sum() / len(combined) * 100:.2f}%")
    print(f"Total P&L: ${combined['pnl'].sum():,.2f}")
    print(f"Avg P&L per Trade: ${combined['pnl'].mean():.2f}")

    # 5. Low magnitude analysis (potential theta decay issues)
    low_mag = combined[combined['magnitude_pct'] < 0.3]
    print(f"\nLow Magnitude (<0.3%) Trades: {len(low_mag)} ({len(low_mag)/total_trades*100:.1f}%)")
    if len(low_mag) > 0:
        print(f"  - Win Rate: {(low_mag['exit_type'] == 'TARGET').sum() / len(low_mag) * 100:.1f}%")
        print(f"  - Avg P&L: ${low_mag['pnl'].mean():.2f}")

    # Save combined data with magnitude
    if not args.analyze_only:
        print("\n" + "=" * 70)
        print("Saving updated CSVs with magnitude_pct...")
        for run_id, df in trades.items():
            output_file = trades_dir / f"{run_id}_trades.csv"
            df.to_csv(output_file, index=False)
            print(f"  Updated: {output_file.name}")

    # Save combined analysis
    combined.to_csv(output_dir / 'all_trades_with_magnitude.csv', index=False)
    print(f"\nSaved combined analysis to: {output_dir / 'all_trades_with_magnitude.csv'}")

    # Save summary JSON
    summary_json = {
        'total_trades': int(total_trades),
        'mean_magnitude_pct': float(combined['magnitude_pct'].mean()),
        'median_magnitude_pct': float(combined['magnitude_pct'].median()),
        'mean_rr_ratio': float(combined['rr_ratio'].mean()),
        'target_hit_rate': float((combined['exit_type'] == 'TARGET').sum() / len(combined) * 100),
        'total_pnl': float(combined['pnl'].sum()),
        'low_magnitude_count': int(len(low_mag)),
        'low_magnitude_pct': float(len(low_mag)/total_trades*100),
        'patterns': pattern_summary.to_dict(),
        'symbols': symbol_summary.to_dict()
    }

    with open(output_dir / 'magnitude_analysis_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
