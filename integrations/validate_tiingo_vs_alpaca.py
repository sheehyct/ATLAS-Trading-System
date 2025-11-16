"""
Cross-validate Tiingo data against Alpaca for quality assurance.

Compares:
- Price correlation (OHLC)
- Mean absolute percentage error (MAPE)
- Maximum price differences
- Volume correlation
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.tiingo_data_fetcher import TiingoDataFetcher


def validate_data_sources(symbol='SPY', start_date='2018-01-01', end_date='2025-11-15'):
    """
    Compare Tiingo vs Alpaca data for quality assurance.

    Args:
        symbol: Symbol to validate
        start_date: Start date for comparison
        end_date: End date for comparison

    Returns:
        dict: Validation results
    """
    print(f"Validating {symbol} from {start_date} to {end_date}")
    print("=" * 80)

    # Fetch from both sources
    print("\n1. Fetching from Tiingo...")
    tiingo_fetcher = TiingoDataFetcher()
    tiingo_data = tiingo_fetcher.fetch(symbol, start_date, end_date)
    tiingo_df = tiingo_data.get(symbol)

    print("2. Fetching from Yahoo Finance...")
    alpaca_data = vbt.YFData.pull(symbol, start=start_date, end=end_date, timeframe='1d')
    # YFData.get() uses lowercase internally, extract the dataframe directly
    alpaca_df = pd.DataFrame({
        'Open': alpaca_data.open.values,
        'High': alpaca_data.high.values,
        'Low': alpaca_data.low.values,
        'Close': alpaca_data.close.values,
        'Volume': alpaca_data.volume.values
    }, index=alpaca_data.wrapper.index)

    # Align dates (both should have same trading days)
    print("\n3. Aligning dates...")
    common_dates = tiingo_df.index.intersection(alpaca_df.index)
    tiingo_aligned = tiingo_df.loc[common_dates]
    alpaca_aligned = alpaca_df.loc[common_dates]

    print(f"   Tiingo days: {len(tiingo_df)}")
    print(f"   Yahoo Finance days: {len(alpaca_df)}")
    print(f"   Common days: {len(common_dates)}")

    # Compare prices
    print("\n4. Comparing prices...")
    results = {}

    for col in ['Open', 'High', 'Low', 'Close']:
        # Calculate correlation
        correlation = tiingo_aligned[col].corr(alpaca_aligned[col])

        # Calculate mean absolute percentage error
        mape = np.mean(np.abs(
            (tiingo_aligned[col] - alpaca_aligned[col]) / alpaca_aligned[col]
        )) * 100

        # Max difference
        max_diff = np.max(np.abs(tiingo_aligned[col] - alpaca_aligned[col]))

        results[col] = {
            'correlation': correlation,
            'mape': mape,
            'max_diff': max_diff
        }

        print(f"\n   {col}:")
        print(f"      Correlation: {correlation:.6f}")
        print(f"      MAPE: {mape:.4f}%")
        print(f"      Max Difference: ${max_diff:.2f}")

    # Volume comparison
    vol_correlation = tiingo_aligned['Volume'].corr(alpaca_aligned['Volume'])
    print(f"\n   Volume:")
    print(f"      Correlation: {vol_correlation:.6f}")

    # Overall assessment
    print("\n5. Overall Assessment")
    print("-" * 80)

    avg_correlation = np.mean([r['correlation'] for r in results.values()])
    avg_mape = np.mean([r['mape'] for r in results.values()])

    if avg_correlation > 0.9999 and avg_mape < 0.1:
        status = "EXCELLENT - Data sources highly aligned"
    elif avg_correlation > 0.999 and avg_mape < 0.5:
        status = "GOOD - Data sources well aligned"
    elif avg_correlation > 0.99:
        status = "ACCEPTABLE - Minor discrepancies detected"
    else:
        status = "POOR - Significant discrepancies, investigate"

    print(f"   Average Correlation: {avg_correlation:.6f}")
    print(f"   Average MAPE: {avg_mape:.4f}%")
    print(f"   Status: {status}")

    return results


if __name__ == "__main__":
    # API key should be set in environment variable TIINGO_API_KEY before running
    # or set it here for testing: os.environ['TIINGO_API_KEY'] = 'your_key_here'

    # Validate SPY
    print("\n" + "=" * 80)
    print("TIINGO VS YAHOO FINANCE DATA VALIDATION")
    print("=" * 80)

    results = validate_data_sources('SPY')

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
