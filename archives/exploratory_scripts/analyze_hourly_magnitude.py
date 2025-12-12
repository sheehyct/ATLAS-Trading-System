"""Analyze magnitude impact on hourly pattern performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig
from validation.strat_validator import DataFetcher

def analyze_by_magnitude(pattern, symbol='SPY', start='2022-01-01', end='2024-11-30'):
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
    
    # Calculate magnitude for each trade
    if 'entry_price' in trades.columns and 'target_price' in trades.columns:
        trades['magnitude_pct'] = abs(trades['target_price'] - trades['entry_price']) / trades['entry_price'] * 100
        
        # Group by magnitude buckets
        buckets = [
            (0, 0.3, '<0.3%'),
            (0.3, 0.5, '0.3-0.5%'),
            (0.5, 1.0, '0.5-1.0%'),
            (1.0, float('inf'), '>1.0%'),
        ]
        
        print(f"\n{pattern} Pattern - Magnitude Analysis:")
        print("-" * 60)
        print(f"{'Magnitude':<12} {'Trades':<8} {'Avg P&L':<12} {'Win Rate':<10}")
        print("-" * 60)
        
        for low, high, label in buckets:
            bucket_trades = trades[(trades['magnitude_pct'] >= low) & (trades['magnitude_pct'] < high)]
            if len(bucket_trades) > 0:
                avg_pnl = bucket_trades['pnl'].mean()
                win_rate = (bucket_trades['pnl'] > 0).mean() * 100
                print(f"{label:<12} {len(bucket_trades):<8} ${avg_pnl:>9,.0f} {win_rate:>8.1f}%")
        
        print(f"\nOverall avg magnitude: {trades['magnitude_pct'].mean():.2f}%")
        print(f"Overall trades: {len(trades)}, Avg P&L: ${trades['pnl'].mean():,.0f}")
    else:
        print(f"{pattern}: magnitude data not available")

def main():
    patterns = ['2-2', '3-2']
    for pattern in patterns:
        analyze_by_magnitude(pattern)

if __name__ == '__main__':
    main()
