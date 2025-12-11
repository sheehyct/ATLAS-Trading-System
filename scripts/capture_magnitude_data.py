#!/usr/bin/env python3
"""
Session 83K-27: Capture magnitude data from validation runs.

This script runs the options backtest and captures magnitude_pct for each trade
to analyze magnitude distribution across patterns.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.strat_validator import DataFetcher
from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig
from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher

# Symbols and patterns to validate
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL']
PATTERNS = ['3-1-2', '2-1-2', '2-2']


def run_pattern_backtest(symbol: str, pattern: str, fetcher: DataFetcher,
                          thetadata_fetcher: ThetaDataOptionsFetcher = None) -> pd.DataFrame:
    """Run backtest for a single symbol/pattern combination and return trade data."""
    print(f"  Running {pattern} on {symbol}...")

    # Get price data
    try:
        data = fetcher.get_data(symbol, '1D')
    except Exception as e:
        print(f"    ERROR fetching data: {e}")
        return pd.DataFrame()

    # Initialize strategy with config
    config = STRATOptionsConfig(
        pattern_types=[pattern],
        symbol=symbol,
        timeframe='1D'
    )
    strategy = STRATOptionsStrategy(config=config)

    # Wire ThetaData if available
    if thetadata_fetcher and hasattr(strategy, '_backtester'):
        strategy._backtester._options_fetcher = thetadata_fetcher
        strategy._backtester._use_market_prices = True

    # Run backtest
    try:
        backtest_result = strategy.backtest(data)

        # BacktestResult has a trades attribute containing the DataFrame
        # Session 83K-29: Use len() instead of .empty for safer check
        if backtest_result.trades is None or len(backtest_result.trades) == 0:
            print(f"    No trades for {pattern}_{symbol}")
            return pd.DataFrame()

        trades_df = backtest_result.trades.copy()

        # Add metadata
        trades_df['symbol'] = symbol
        trades_df['pattern_type_cat'] = pattern

        print(f"    Found {len(trades_df)} trades")
        return trades_df

    except Exception as e:
        print(f"    ERROR in backtest: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("Session 83K-27: Magnitude Data Capture")
    print("=" * 70)

    # Initialize fetchers
    fetcher = DataFetcher()

    # Try to connect ThetaData
    thetadata = None
    try:
        from integrations.thetadata_client import ThetaDataRESTClient
        client = ThetaDataRESTClient()
        if client.connect():
            thetadata = ThetaDataOptionsFetcher(client)
            print("[OK] ThetaData connected")
        else:
            print("[WARN] ThetaData not available, using Black-Scholes")
    except Exception as e:
        print(f"[WARN] ThetaData error: {e}")

    # Collect all trade data
    all_trades = []

    for pattern in PATTERNS:
        print(f"\n--- Pattern: {pattern} ---")
        for symbol in SYMBOLS:
            trades = run_pattern_backtest(symbol, pattern, fetcher, thetadata)
            if not trades.empty:
                all_trades.append(trades)

    # Combine results
    if not all_trades:
        print("\nNo trades found!")
        return

    df = pd.concat(all_trades, ignore_index=True)

    # Save to CSV
    output_path = project_root / 'output' / 'magnitude_analysis_83k27.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(df)} trades to {output_path}")

    # Analyze magnitude distribution
    print("\n" + "=" * 70)
    print("MAGNITUDE ANALYSIS")
    print("=" * 70)

    if 'magnitude_pct' in df.columns:
        print("\nMagnitude by Pattern Type:")
        mag_by_pattern = df.groupby('pattern_type')['magnitude_pct'].agg(['mean', 'std', 'count'])
        print(mag_by_pattern.to_string())

        print("\nMagnitude Distribution:")
        for threshold in [0.3, 0.5, 1.0]:
            below = (df['magnitude_pct'] < threshold).sum()
            pct = below / len(df) * 100
            print(f"  < {threshold}%: {below} trades ({pct:.1f}%)")

        print("\nMagnitude by Pattern + Symbol:")
        mag_by_both = df.groupby(['pattern_type', 'symbol'])['magnitude_pct'].agg(['mean', 'count'])
        print(mag_by_both.to_string())

        print("\nP&L by Magnitude Bucket:")
        df['mag_bucket'] = pd.cut(df['magnitude_pct'], bins=[0, 0.3, 0.5, 1.0, 100],
                                   labels=['<0.3%', '0.3-0.5%', '0.5-1.0%', '>1.0%'])
        pnl_by_mag = df.groupby('mag_bucket', observed=True)['pnl'].agg(['sum', 'mean', 'count'])
        pnl_by_mag['win_rate'] = df.groupby('mag_bucket', observed=True)['win'].mean()
        print(pnl_by_mag.to_string())
    else:
        print("[ERROR] magnitude_pct column not found in results!")
        print(f"Available columns: {df.columns.tolist()}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total trades: {len(df)}")
    print(f"Patterns: {df['pattern_type'].unique().tolist()}")
    print(f"Symbols: {df['symbol'].unique().tolist()}")
    if 'pnl' in df.columns:
        print(f"Total P&L: ${df['pnl'].sum():,.2f}")
        print(f"Win Rate: {df['win'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
