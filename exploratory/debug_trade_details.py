"""Debug script to extract detailed trade data from SPY 1D 3-1-2 strategy."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from integrations.tiingo_data_fetcher import TiingoDataFetcher
from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig

print('='*70)
print('SPY 1D 3-1-2 DETAILED ANALYSIS (Post P&L Fix)')
print('='*70)

# Get data
fetcher = TiingoDataFetcher()
spy_vbt = fetcher.fetch('SPY', start_date='2020-01-01', end_date='2025-01-01', timeframe='1d')
spy_data = spy_vbt.get()
print(f'Loaded {len(spy_data)} bars')

# Configure strategy
config = STRATOptionsConfig(
    pattern_types=['3-1-2'],
    timeframe='1D',
    min_continuation_bars=2,
    include_22_down=True,
    symbol='SPY',
)

strategy = STRATOptionsStrategy(config=config)

# Run backtest
result = strategy.backtest(spy_data)

print(f'\n=== Backtest Results ===')
print(f'  Trade Count: {result.trade_count}')
print(f'  Total Return: {result.total_return:.4f} ({result.total_return*100:.2f}%)')
print(f'  Sharpe Ratio: {result.sharpe_ratio:.4f}')
print(f'  Max Drawdown: {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)')
print(f'  Win Rate: {result.win_rate:.4f} ({result.win_rate*100:.1f}%)')

# Get individual trades
if result.trades is not None and not result.trades.empty:
    trades = result.trades

    # P&L Summary
    total_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else 0
    avg_pnl = trades['pnl'].mean() if 'pnl' in trades.columns else 0

    winners = trades[trades['pnl'] > 0] if 'pnl' in trades.columns else pd.DataFrame()
    losers = trades[trades['pnl'] <= 0] if 'pnl' in trades.columns else pd.DataFrame()

    print(f'\n=== P&L Summary ===')
    print(f'  Total P&L: ${total_pnl:.2f}')
    print(f'  Average P&L: ${avg_pnl:.2f}')
    print(f'  Winners: {len(winners)} trades')
    print(f'  Losers: {len(losers)} trades')

    if len(winners) > 0:
        print(f'  Avg Winning Trade: ${winners["pnl"].mean():.2f}')
        print(f'  Total Wins: ${winners["pnl"].sum():.2f}')
    if len(losers) > 0:
        print(f'  Avg Losing Trade: ${losers["pnl"].mean():.2f}')
        print(f'  Total Losses: ${losers["pnl"].sum():.2f}')

    if len(losers) > 0 and losers['pnl'].sum() != 0:
        profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum())
        print(f'  Profit Factor: {profit_factor:.2f}')

    # Return analysis
    if 'return_pct' in trades.columns:
        print(f'\n=== Return Analysis ===')
        print(f'  Avg Return: {trades["return_pct"].mean()*100:.1f}%')
        print(f'  Min Return: {trades["return_pct"].min()*100:.1f}%')
        print(f'  Max Return: {trades["return_pct"].max()*100:.1f}%')
        print(f'  Std Dev: {trades["return_pct"].std()*100:.1f}%')

        if trades['return_pct'].min() < -1.0:
            print(f'  *** WARNING: Min return < -100% - BUG EXISTS! ***')
        else:
            print(f'  *** Returns are realistic (no return < -100%) ***')

    # Individual trades
    print(f'\n=== Individual Trades ({len(trades)}) ===')
    print(f'{"#":<3} {"Date":<12} {"Dir":<8} {"Entry$":<10} {"Exit$":<10} {"P&L":<12} {"Return":<10} {"Exit Reason":<12}')
    print('-'*80)

    for i, (idx, row) in enumerate(trades.iterrows(), 1):
        date = str(row.get('pattern_timestamp', ''))[:10]
        direction = str(row.get('direction_label', 'N/A'))[:7]
        entry = row.get('entry_premium', 0)
        exit_p = row.get('exit_premium', 0)
        pnl = row.get('pnl', 0)
        ret = row.get('return_pct', 0) * 100
        exit_r = str(row.get('exit_reason', 'N/A'))[:11]

        print(f'{i:<3} {date:<12} {direction:<8} ${entry:<9.2f} ${exit_p:<9.2f} ${pnl:<11.2f} {ret:<9.1f}% {exit_r:<12}')

    # Correlation analysis
    print(f'\n=== Correlations ===')
    if 'entry_premium' in trades.columns and 'pnl' in trades.columns:
        corr = trades['entry_premium'].corr(trades['pnl'])
        print(f'  Entry Premium vs P&L: {corr:.3f}')

    if 'delta' in trades.columns and 'pnl' in trades.columns:
        corr = trades['delta'].corr(trades['pnl'])
        print(f'  Delta vs P&L: {corr:.3f}')

else:
    print('No trade details available')

print('\n' + '='*70)
print('Analysis Complete')
print('='*70)
