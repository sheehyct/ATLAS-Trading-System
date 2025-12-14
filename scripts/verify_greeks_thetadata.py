"""
Greeks Verification Script for STRAT Options Backtest

Session 85: Verify that strike selection produces reasonable deltas
using ThetaData's historical Greeks endpoint.

STRAT Methodology Target: Delta 0.30-0.80 range
- Strikes within [Entry, Target] range
- Best Strike = Entry + 0.3 * (Target - Entry) for calls

Usage:
    uv run python scripts/verify_greeks_thetadata.py --symbol SPY --risk 7 --sample 30
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from integrations.thetadata_client import ThetaDataRESTClient


def load_trades(symbol: str, risk_pct: float, reports_dir: str = 'reports') -> pd.DataFrame:
    """Load trades from ThetaData backtest CSV."""
    risk_int = int(risk_pct)
    filepath = Path(reports_dir) / f'options_backtest_{symbol}_{risk_int}pct_thetadata.csv'

    if not filepath.exists():
        raise FileNotFoundError(f"Backtest file not found: {filepath}")

    df = pd.read_csv(filepath)
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['expiration'] = pd.to_datetime(df['expiration'])

    return df


def sample_trades(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Sample trades across the time range for representative verification.

    Samples evenly across time periods to get a good distribution.
    """
    if len(df) <= n:
        return df

    # Sort by entry date and sample evenly
    df_sorted = df.sort_values('entry_date').reset_index(drop=True)

    # Calculate indices to sample evenly across time
    indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)

    return df_sorted.iloc[indices].reset_index(drop=True)


def verify_greeks(
    client: ThetaDataRESTClient,
    trades: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Query ThetaData for Greeks at entry and analyze delta distribution.

    Returns DataFrame with Greeks data added.
    """
    results = []

    for idx, row in trades.iterrows():
        entry_date = row['entry_date']
        expiration = row['expiration']
        strike = row['strike']
        option_type = 'C' if row['option_type'] == 'call' else 'P'
        symbol = 'SPY'  # Assuming SPY for now

        if verbose:
            print(f"  [{idx+1}/{len(trades)}] {entry_date.date()} - ${strike} {option_type} exp {expiration.date()}...", end=' ')

        # Get Greeks from ThetaData
        greeks = client.get_greeks(
            underlying=symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            as_of=entry_date
        )

        result = row.to_dict()

        if greeks:
            result['greeks_delta'] = greeks.get('delta', None)
            result['greeks_gamma'] = greeks.get('gamma', None)
            result['greeks_theta'] = greeks.get('theta', None)
            result['greeks_vega'] = greeks.get('vega', None)
            result['greeks_iv'] = greeks.get('iv', None)
            result['greeks_underlying'] = greeks.get('underlying_price', None)
            result['greeks_source'] = 'ThetaData'

            if verbose:
                delta = greeks.get('delta', 0)
                print(f"Delta: {delta:.3f}")
        else:
            result['greeks_delta'] = None
            result['greeks_gamma'] = None
            result['greeks_theta'] = None
            result['greeks_vega'] = None
            result['greeks_iv'] = None
            result['greeks_underlying'] = None
            result['greeks_source'] = 'N/A'

            if verbose:
                print("No data")

        results.append(result)

    return pd.DataFrame(results)


def analyze_deltas(df: pd.DataFrame) -> dict:
    """Analyze delta distribution and generate summary metrics."""
    # Filter to trades with valid delta
    valid = df[df['greeks_delta'].notna()].copy()

    if len(valid) == 0:
        return {'error': 'No valid Greeks data found'}

    # For puts, delta is negative - take absolute value for analysis
    valid['abs_delta'] = valid['greeks_delta'].abs()

    metrics = {
        'total_sampled': len(df),
        'valid_greeks': len(valid),
        'success_rate': len(valid) / len(df) * 100,

        # Delta statistics
        'delta_mean': valid['abs_delta'].mean(),
        'delta_median': valid['abs_delta'].median(),
        'delta_std': valid['abs_delta'].std(),
        'delta_min': valid['abs_delta'].min(),
        'delta_max': valid['abs_delta'].max(),

        # Target range (0.30-0.80)
        'in_target_range': ((valid['abs_delta'] >= 0.30) & (valid['abs_delta'] <= 0.80)).sum(),
        'in_target_pct': ((valid['abs_delta'] >= 0.30) & (valid['abs_delta'] <= 0.80)).mean() * 100,

        # Distribution buckets
        'delta_0_20': (valid['abs_delta'] < 0.20).sum(),
        'delta_20_30': ((valid['abs_delta'] >= 0.20) & (valid['abs_delta'] < 0.30)).sum(),
        'delta_30_50': ((valid['abs_delta'] >= 0.30) & (valid['abs_delta'] < 0.50)).sum(),
        'delta_50_70': ((valid['abs_delta'] >= 0.50) & (valid['abs_delta'] < 0.70)).sum(),
        'delta_70_80': ((valid['abs_delta'] >= 0.70) & (valid['abs_delta'] <= 0.80)).sum(),
        'delta_80_plus': (valid['abs_delta'] > 0.80).sum(),

        # By option type
        'calls_mean_delta': valid[valid['option_type'] == 'call']['greeks_delta'].mean() if len(valid[valid['option_type'] == 'call']) > 0 else None,
        'puts_mean_delta': valid[valid['option_type'] == 'put']['greeks_delta'].mean() if len(valid[valid['option_type'] == 'put']) > 0 else None,

        # IV statistics
        'iv_mean': valid['greeks_iv'].mean() if 'greeks_iv' in valid else None,
        'iv_median': valid['greeks_iv'].median() if 'greeks_iv' in valid else None,
    }

    return metrics


def print_report(metrics: dict, df: pd.DataFrame):
    """Print formatted verification report."""
    print("\n" + "=" * 70)
    print("GREEKS VERIFICATION REPORT - THETADATA")
    print("=" * 70)

    print(f"\nData Quality:")
    print(f"  Total trades sampled: {metrics['total_sampled']}")
    print(f"  Valid Greeks data: {metrics['valid_greeks']} ({metrics['success_rate']:.1f}%)")

    print(f"\nDelta Statistics (absolute values):")
    print(f"  Mean:   {metrics['delta_mean']:.3f}")
    print(f"  Median: {metrics['delta_median']:.3f}")
    print(f"  Std:    {metrics['delta_std']:.3f}")
    print(f"  Range:  {metrics['delta_min']:.3f} - {metrics['delta_max']:.3f}")

    print(f"\nTarget Range (0.30-0.80):")
    print(f"  In range: {metrics['in_target_range']} trades ({metrics['in_target_pct']:.1f}%)")

    print(f"\nDelta Distribution:")
    print(f"  < 0.20 (deep OTM):     {metrics['delta_0_20']} trades")
    print(f"  0.20-0.30 (OTM):       {metrics['delta_20_30']} trades")
    print(f"  0.30-0.50 (slight OTM):{metrics['delta_30_50']} trades")
    print(f"  0.50-0.70 (ATM/ITM):   {metrics['delta_50_70']} trades")
    print(f"  0.70-0.80 (ITM):       {metrics['delta_70_80']} trades")
    print(f"  > 0.80 (deep ITM):     {metrics['delta_80_plus']} trades")

    if metrics.get('calls_mean_delta') is not None:
        print(f"\nBy Option Type:")
        print(f"  Calls mean delta: {metrics['calls_mean_delta']:.3f}")
        if metrics.get('puts_mean_delta') is not None:
            print(f"  Puts mean delta:  {metrics['puts_mean_delta']:.3f}")

    if metrics.get('iv_mean') is not None and metrics['iv_mean'] > 0:
        print(f"\nImplied Volatility:")
        print(f"  Mean IV:   {metrics['iv_mean']:.2%}")
        print(f"  Median IV: {metrics['iv_median']:.2%}")

    # Show sample trades with Greeks
    valid = df[df['greeks_delta'].notna()].copy()
    if len(valid) > 0:
        print(f"\nSample Trades with Greeks (first 10):")
        print("-" * 70)
        cols = ['entry_date', 'pattern_type', 'strike', 'option_type', 'greeks_delta', 'outcome']
        sample = valid.head(10)[cols].copy()
        sample['entry_date'] = sample['entry_date'].dt.strftime('%Y-%m-%d')
        sample['greeks_delta'] = sample['greeks_delta'].apply(lambda x: f"{x:.3f}")
        print(sample.to_string(index=False))

    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    if metrics['in_target_pct'] >= 70:
        print("  PASS: Majority of strikes have deltas in target range (0.30-0.80)")
    elif metrics['in_target_pct'] >= 50:
        print("  ACCEPTABLE: Over half of strikes have deltas in target range")
    else:
        print("  REVIEW: Less than half of strikes in target delta range")
        print("  Consider adjusting strike selection formula")

    if metrics['delta_mean'] > 0.80:
        print("  NOTE: Average delta is high (ITM) - consider more OTM strikes")
    elif metrics['delta_mean'] < 0.30:
        print("  NOTE: Average delta is low (OTM) - consider strikes closer to ATM")
    else:
        print(f"  GOOD: Average delta ({metrics['delta_mean']:.3f}) is within target range")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify Greeks for STRAT options backtest')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol (default: SPY)')
    parser.add_argument('--risk', type=float, default=7.0, help='Risk level (default: 7.0)')
    parser.add_argument('--sample', type=int, default=30, help='Number of trades to sample (default: 30)')
    parser.add_argument('--all', action='store_true', help='Verify all trades (overrides --sample)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file for results')
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    print("=" * 70)
    print("STRAT OPTIONS GREEKS VERIFICATION")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Risk level: {args.risk}%")
    print(f"Sample size: {'ALL' if args.all else args.sample}")

    # Load trades
    print("\n[1/4] Loading trades...")
    trades = load_trades(args.symbol, args.risk)
    print(f"  Loaded {len(trades)} trades")

    # Sample trades
    print("\n[2/4] Sampling trades...")
    if args.all:
        sample = trades
    else:
        sample = sample_trades(trades, args.sample)
    print(f"  Sampled {len(sample)} trades for verification")
    print(f"  Date range: {sample['entry_date'].min().date()} to {sample['entry_date'].max().date()}")

    # Connect to ThetaData
    print("\n[3/4] Connecting to ThetaData...")
    client = ThetaDataRESTClient()
    if not client.connect():
        print("  ERROR: Failed to connect to ThetaData terminal")
        print("  Make sure terminal is running: cd C:\\thetaterminal && java -jar thetaterminalv3.jar")
        return 1
    print("  Connected")

    # Verify Greeks
    print("\n[4/4] Fetching Greeks from ThetaData...")
    results = verify_greeks(client, sample)

    # Disconnect
    client.disconnect()

    # Analyze results
    metrics = analyze_deltas(results)

    # Print report
    print_report(metrics, results)

    # Save results if requested
    if args.output:
        output_path = Path('reports') / args.output
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
