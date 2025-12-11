"""Analyze why 2-2 and 2-1-2 underperform on hourly vs 3-bar patterns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig
from validation.strat_validator import DataFetcher

def analyze_pattern(pattern, symbol='SPY', start='2022-01-01', end='2024-11-30'):
    fetcher = DataFetcher()
    data = fetcher.get_data(symbol, '1H', start, end)
    
    config = STRATOptionsConfig(
        pattern_types=[pattern],
        timeframe='1H',
        symbol=symbol
    )
    
    strategy = STRATOptionsStrategy(config=config)
    result = strategy.backtest(data)
    
    if result.trades is None or result.trades.empty:
        return None
    
    trades = result.trades
    stats = {
        'pattern': pattern,
        'trades': len(trades),
        'total_pnl': trades['pnl'].sum(),
        'avg_pnl': trades['pnl'].mean(),
        'win_rate': (trades['pnl'] > 0).mean() * 100,
    }
    
    # Exit type breakdown
    for exit_type in ['TARGET', 'STOP', 'TIME_EXIT']:
        subset = trades[trades['exit_type'] == exit_type]
        if len(subset) > 0:
            stats[f'{exit_type.lower()}_count'] = len(subset)
            stats[f'{exit_type.lower()}_avg'] = subset['pnl'].mean()
        else:
            stats[f'{exit_type.lower()}_count'] = 0
            stats[f'{exit_type.lower()}_avg'] = 0
    
    return stats

def main():
    patterns = ['2-2', '2-1-2', '3-1-2', '3-2', '3-2-2']
    
    print("Hourly Pattern Analysis (SPY, 2022-2024)")
    print("=" * 80)
    
    results = []
    for pattern in patterns:
        stats = analyze_pattern(pattern)
        if stats:
            results.append(stats)
    
    # Print comparison
    print(f"\n{'Pattern':<10} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win %':<8}")
    print("-" * 48)
    for s in results:
        print(f"{s['pattern']:<10} {s['trades']:<8} ${s['total_pnl']:>9,.0f} ${s['avg_pnl']:>8,.0f} {s['win_rate']:>6.1f}%")
    
    print(f"\n{'Pattern':<10} {'TARGET #':<10} {'TARGET Avg':<12} {'STOP #':<10} {'STOP Avg':<12} {'TIME #':<10} {'TIME Avg':<12}")
    print("-" * 86)
    for s in results:
        print(f"{s['pattern']:<10} {s['target_count']:<10} ${s['target_avg']:>9,.0f} {s['stop_count']:<10} ${s['stop_avg']:>9,.0f} {s['time_exit_count']:<10} ${s['time_exit_avg']:>9,.0f}")
    
    print("\n" + "=" * 80)
    print("Analysis Summary:")
    print("-" * 80)
    
    # Find patterns with high TIME_EXIT %
    for s in results:
        time_pct = (s['time_exit_count'] / s['trades']) * 100 if s['trades'] > 0 else 0
        stop_pct = (s['stop_count'] / s['trades']) * 100 if s['trades'] > 0 else 0
        target_pct = (s['target_count'] / s['trades']) * 100 if s['trades'] > 0 else 0
        print(f"{s['pattern']}: TARGET {target_pct:.0f}% | STOP {stop_pct:.0f}% | TIME_EXIT {time_pct:.0f}%")

if __name__ == '__main__':
    main()
