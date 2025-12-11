"""Quick analysis of Weekly and Daily validation results."""
import pandas as pd
import glob
import os
import sys

timeframe = sys.argv[1] if len(sys.argv) > 1 else '1W'

# Load all trades for specified timeframe
files = glob.glob(f'validation_results/session_83k/trades/*_{timeframe}_*.csv')
all_trades = []
for f in files:
    df = pd.read_csv(f)
    parts = os.path.basename(f).replace('_trades.csv', '').split('_')
    df['pattern'] = parts[0]
    df['timeframe'] = parts[1]
    df['symbol'] = parts[2]
    all_trades.append(df)

if all_trades:
    trades = pd.concat(all_trades, ignore_index=True)
    print(f'Total Weekly Trades: {len(trades)}')
    print()

    # Overall stats
    total_pnl = trades['pnl'].sum()
    avg_pnl = trades['pnl'].mean()
    print(f'Total P&L: ${total_pnl:,.2f}')
    print(f'Average P&L: ${avg_pnl:,.2f}')
    print()

    # By pattern
    print('Weekly Performance by Pattern:')
    print('=' * 70)
    pattern_stats = trades.groupby('pattern').agg({
        'pnl': ['count', 'sum', 'mean'],
    }).round(2)
    pattern_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    pattern_stats = pattern_stats.sort_values('avg_pnl', ascending=False)

    for pattern, row in pattern_stats.iterrows():
        print(f'{pattern:8} | {int(row.trades):3} trades | Total: ${row.total_pnl:>10,.2f} | Avg: ${row.avg_pnl:>8,.2f}')

    print()
    print('Weekly Performance by Symbol:')
    print('=' * 70)
    symbol_stats = trades.groupby('symbol').agg({
        'pnl': ['count', 'sum', 'mean'],
    }).round(2)
    symbol_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    symbol_stats = symbol_stats.sort_values('avg_pnl', ascending=False)

    for symbol, row in symbol_stats.iterrows():
        print(f'{symbol:6} | {int(row.trades):3} trades | Total: ${row.total_pnl:>10,.2f} | Avg: ${row.avg_pnl:>8,.2f}')

    # Magnitude analysis
    if 'magnitude_pct' in trades.columns:
        print()
        print('Weekly Magnitude Analysis:')
        print('=' * 70)
        bins = [0, 0.3, 0.5, 1.0, 2.0, 5.0, float('inf')]
        labels = ['<0.3%', '0.3-0.5%', '0.5-1.0%', '1.0-2.0%', '2.0-5.0%', '>5.0%']
        trades['mag_bucket'] = pd.cut(trades['magnitude_pct'], bins=bins, labels=labels)
        mag_stats = trades.groupby('mag_bucket', observed=True).agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        mag_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
        for bucket, row in mag_stats.iterrows():
            if row.trades > 0:
                print(f'{bucket:12} | {int(row.trades):3} trades | Avg: ${row.avg_pnl:>8,.2f}')

    # Pattern x Symbol breakdown
    print()
    print('Weekly Performance by Pattern x Symbol:')
    print('=' * 70)
    combo_stats = trades.groupby(['pattern', 'symbol']).agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    combo_stats.columns = ['trades', 'total_pnl', 'avg_pnl']
    combo_stats = combo_stats.sort_values('avg_pnl', ascending=False)
    for (pattern, symbol), row in combo_stats.head(15).iterrows():
        print(f'{pattern:8} {symbol:5} | {int(row.trades):3} trades | Avg: ${row.avg_pnl:>8,.2f}')

else:
    print('No Weekly trade files found')
