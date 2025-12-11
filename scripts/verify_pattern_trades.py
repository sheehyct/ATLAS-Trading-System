#!/usr/bin/env python3
"""Quick pattern trade verification for Session 83K-62."""

import pandas as pd
from pathlib import Path

trades_dir = Path('validation_results/session_83k/trades')
patterns = ['2-2', '3-2', '2-1-2', '3-1-2', '3-2-2']

print('=' * 90)
print('PATTERN TRADE VERIFICATION - Session 83K-62')
print('=' * 90)

for pattern in patterns:
    pattern_files = list(trades_dir.glob(f'{pattern}_1D_SPY_trades.csv'))
    if not pattern_files:
        pattern_files = list(trades_dir.glob(f'{pattern}_1D_*_trades.csv'))

    if not pattern_files:
        print(f'\n{pattern}: No files found')
        continue

    df = pd.read_csv(pattern_files[0])
    if df.empty:
        print(f'\n{pattern}: Empty')
        continue

    print(f'\n--- {pattern} ({len(df)} trades from {pattern_files[0].name}) ---')

    # Bullish
    bullish = df[df['direction'] == 1]
    if len(bullish) > 0:
        t = bullish.iloc[0]
        entry_date = str(t.get('entry_date', 'N/A'))[:10]
        print(f"  Bullish: {entry_date} | Entry ${t['entry_price']:.2f} -> Target ${t['target_price']:.2f} | {t['exit_type']} | P&L ${t['pnl']:.2f}")
    else:
        print("  Bullish: No trades")

    # Bearish
    bearish = df[df['direction'] == -1]
    if len(bearish) > 0:
        t = bearish.iloc[0]
        entry_date = str(t.get('entry_date', 'N/A'))[:10]
        print(f"  Bearish: {entry_date} | Entry ${t['entry_price']:.2f} -> Target ${t['target_price']:.2f} | {t['exit_type']} | P&L ${t['pnl']:.2f}")
    else:
        print("  Bearish: No trades")

print('\n' + '=' * 90)
print('VERIFICATION COMPLETE')
print('=' * 90)
