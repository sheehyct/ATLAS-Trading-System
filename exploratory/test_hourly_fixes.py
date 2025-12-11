"""Test hourly validation fixes from Session 83K-33."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig
from validation.strat_validator import DataFetcher


def main():
    fetcher = DataFetcher()
    data = fetcher.get_data('SPY', '1H', '2024-01-01', '2024-06-30')

    # Debug: Check bar timestamps
    print('Sample bar timestamps:')
    for i in range(min(10, len(data))):
        ts = data.index[i]
        print(f'  {ts} | hour attr: {ts.hour if hasattr(ts, "hour") else "N/A"}')

    config = STRATOptionsConfig(
        pattern_types=['2-2'],
        timeframe='1H',
        symbol='SPY'
    )

    strategy = STRATOptionsStrategy(config=config)
    result = strategy.backtest(data)

    if result.trades is not None and not result.trades.empty:
        trades_df = result.trades
        print(f'\nTrades: {len(trades_df)}')
        print(f'Total P&L: ${trades_df["pnl"].sum():,.0f}')
        print(f'Avg P&L: ${trades_df["pnl"].mean():,.0f}')

        print(f'\nExit Type Distribution:')
        for exit_type, count in trades_df['exit_type'].value_counts().items():
            print(f'  {exit_type}: {count}')

        print(f'\nTrade Details:')
        for i, row in trades_df.iterrows():
            entry = str(row.get('entry_date', 'N/A'))[:19]
            exit_d = str(row.get('exit_date', 'N/A'))[:19]
            exit_t = row.get('exit_type', '?')
            pnl = row['pnl']
            print(f'  {entry} -> {exit_d} | {exit_t} | PnL: {pnl:.0f}')
    else:
        print('No trades generated')


if __name__ == '__main__':
    main()
