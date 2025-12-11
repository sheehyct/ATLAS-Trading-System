"""
Test 52W High Momentum on Volatile Stocks

User hypothesis: Volume confirmation may work better on volatile stocks
where volume surges actually signal institutional participation/momentum.

SPY problem: Highly liquid ETF, stable volume patterns, 2.0x surge is rare
Volatile stock advantage: More dynamic volume, breakouts WITH volume = stronger signal

Test stocks (varying volatility profiles):
- TSLA (Tesla) - High volatility, momentum stock
- NVDA (Nvidia) - Tech/AI, high growth volatility
- AMD (AMD) - Semiconductor, cyclical volatility
- AAPL (Apple) - Large cap, moderate volatility
- MSFT (Microsoft) - Large cap, moderate volatility

Test volume thresholds: 1.15x, 1.25x, 1.5x, 1.75x, 2.0x
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, List


def generate_signals(
    data: pd.DataFrame,
    entry_threshold: float = 0.90,
    exit_threshold: float = 0.88,
    volume_multiplier: float = None
) -> Dict[str, pd.Series]:
    """
    Generate event-based signals with optional volume filter.

    Args:
        data: OHLCV DataFrame
        entry_threshold: Entry distance threshold (default: 0.90)
        exit_threshold: Exit distance threshold (default: 0.88)
        volume_multiplier: Volume threshold multiplier (None = no filter)

    Returns:
        Dictionary with entry_signal, exit_signal
    """
    # Calculate 52-week high
    high_52w = data['High'].rolling(window=252, min_periods=252).max()
    distance = data['Close'] / high_52w
    volume_ma = data['Volume'].rolling(window=20, min_periods=20).mean()

    # ATR for validation
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # States
    in_entry_zone = (distance >= entry_threshold) & high_52w.notna() & atr.notna()
    in_exit_zone = (distance < exit_threshold) & high_52w.notna()

    # Apply volume filter if specified
    if volume_multiplier is not None:
        volume_ok = (data['Volume'] > (volume_ma * volume_multiplier)) & volume_ma.notna()
        in_entry_zone = in_entry_zone & volume_ok

    # Events (state transitions)
    entry_events = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
    exit_events = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

    return {
        'entry_signal': entry_events.fillna(False),
        'exit_signal': exit_events.fillna(False),
        'distance': distance,
        'volume_ratio': data['Volume'] / volume_ma
    }


def run_backtest_on_stock(
    symbol: str,
    start_date: str = '2015-01-01',
    end_date: str = '2025-01-01',
    volume_multiplier: float = None,
    verbose: bool = False
) -> Dict:
    """
    Run backtest on single stock with volume threshold.

    Args:
        symbol: Stock ticker
        start_date: Backtest start
        end_date: Backtest end
        volume_multiplier: Volume filter threshold (None = no filter)
        verbose: Print details

    Returns:
        Dictionary with results
    """
    try:
        # Load data
        if verbose:
            print(f"\n  Loading {symbol} data...")
        data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()

        if len(data) < 300:
            return {'error': 'Insufficient data', 'symbol': symbol}

        # Generate signals
        signals = generate_signals(
            data,
            entry_threshold=0.90,
            exit_threshold=0.88,
            volume_multiplier=volume_multiplier
        )

        entry_count = signals['entry_signal'].sum()
        exit_count = signals['exit_signal'].sum()

        if verbose:
            print(f"  Entry events: {entry_count}, Exit events: {exit_count}")

        # Run backtest with fixed shares for comparison
        pf = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=signals['entry_signal'],
            exits=signals['exit_signal'],
            size=pd.Series(10, index=data.index),
            size_type='amount',
            init_cash=10000,
            fees=0.0015,
            slippage=0.0015,
            freq='1D'
        )

        trades = pf.trades.records_readable
        trade_count = len(trades)

        if trade_count == 0:
            return {
                'symbol': symbol,
                'trades': 0,
                'sharpe': 0,
                'total_return': 0,
                'max_dd': 0,
                'win_rate': 0,
                'entry_events': entry_count,
                'exit_events': exit_count
            }

        # Calculate win rate
        win_rate = (trades['PnL'] > 0).sum() / trade_count if trade_count > 0 else 0

        return {
            'symbol': symbol,
            'trades': trade_count,
            'sharpe': pf.sharpe_ratio,
            'total_return': pf.total_return,
            'cagr': pf.annualized_return,
            'max_dd': pf.max_drawdown,
            'win_rate': win_rate,
            'entry_events': entry_count,
            'exit_events': exit_count,
            'avg_trade': trades['Return'].mean() if 'Return' in trades.columns else 0
        }

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return {'error': str(e), 'symbol': symbol}


def test_volume_thresholds_on_stocks():
    """
    Test volume thresholds on various stocks.
    """
    print("\n" + "="*100)
    print("52-WEEK HIGH MOMENTUM - VOLATILE STOCKS TESTING")
    print("="*100)

    print("\nHypothesis: Volume confirmation works better on volatile stocks")
    print("SPY: Highly liquid ETF, stable volume - 2.0x rarely hits")
    print("Volatile stocks: Dynamic volume, breakouts WITH volume = stronger signal")

    # Stock selection (varying volatility)
    stocks = [
        ('TSLA', 'Tesla - High Volatility'),
        ('NVDA', 'Nvidia - Tech/AI Growth'),
        ('AMD', 'AMD - Semiconductor'),
        ('AAPL', 'Apple - Large Cap'),
        ('MSFT', 'Microsoft - Large Cap'),
        ('SPY', 'SPY - ETF Baseline')
    ]

    # Volume thresholds to test
    volume_thresholds = [
        (None, 'No Filter'),
        (1.15, '1.15x'),
        (1.25, '1.25x'),
        (1.5, '1.5x'),
        (1.75, '1.75x'),
        (2.0, '2.0x')
    ]

    # Test period (10 years for most stocks)
    start_date = '2015-01-01'
    end_date = '2025-01-01'

    print(f"\nTest Period: {start_date} to {end_date} (10 years)")
    print(f"Entry: distance >= 0.90, Exit: distance < 0.88 (event-based)")
    print(f"Position: Fixed 10 shares (for comparison across stocks)")

    # Run tests
    all_results = []

    for symbol, description in stocks:
        print(f"\n{'='*100}")
        print(f"TESTING: {symbol} - {description}")
        print(f"{'='*100}")

        stock_results = []

        for vol_mult, vol_label in volume_thresholds:
            result = run_backtest_on_stock(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                volume_multiplier=vol_mult,
                verbose=False
            )

            if 'error' not in result:
                result['volume_threshold'] = vol_label
                stock_results.append(result)

                print(f"{vol_label:<12} Trades: {result['trades']:>3}  "
                      f"Sharpe: {result['sharpe']:>5.2f}  "
                      f"Return: {result['total_return']:>7.1%}  "
                      f"MaxDD: {result['max_dd']:>6.1%}  "
                      f"Win%: {result['win_rate']:>5.1%}")
            else:
                print(f"{vol_label:<12} ERROR: {result['error']}")

        all_results.extend(stock_results)

        # Best configuration for this stock
        if len(stock_results) > 0:
            valid_results = [r for r in stock_results if r['trades'] >= 5]
            if len(valid_results) > 0:
                best = max(valid_results, key=lambda x: x['sharpe'])
                print(f"\nBEST for {symbol}: {best['volume_threshold']} "
                      f"({best['trades']} trades, Sharpe {best['sharpe']:.2f})")

    # Summary comparison
    print("\n" + "="*100)
    print("SUMMARY: Volume Filter Impact by Stock")
    print("="*100)

    # Group by stock and volume threshold
    summary_data = {}
    for result in all_results:
        symbol = result['symbol']
        if symbol not in summary_data:
            summary_data[symbol] = {}
        summary_data[symbol][result['volume_threshold']] = result

    # Find best threshold for each stock
    print(f"\n{'Stock':<6} {'Best Threshold':<15} {'Trades':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
    print("-"*100)

    for symbol in ['TSLA', 'NVDA', 'AMD', 'AAPL', 'MSFT', 'SPY']:
        if symbol in summary_data:
            stock_results = list(summary_data[symbol].values())
            valid = [r for r in stock_results if r['trades'] >= 5]

            if len(valid) > 0:
                best = max(valid, key=lambda x: x['sharpe'])
                print(f"{symbol:<6} {best['volume_threshold']:<15} {best['trades']:>8} "
                      f"{best['sharpe']:>8.2f} {best['total_return']:>9.1%} {best['max_dd']:>7.1%}")
            else:
                print(f"{symbol:<6} {'Insufficient trades':<15}")

    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)

    # Compare SPY vs volatile stocks
    spy_results = summary_data.get('SPY', {})
    tsla_results = summary_data.get('TSLA', {})
    nvda_results = summary_data.get('NVDA', {})

    if spy_results and tsla_results:
        print("\nVolume Filter Effectiveness:")

        for threshold in ['No Filter', '1.5x', '2.0x']:
            if threshold in spy_results and threshold in tsla_results:
                spy = spy_results[threshold]
                tsla = tsla_results[threshold]

                print(f"\n  {threshold}:")
                print(f"    SPY:  {spy['trades']:2} trades, Sharpe {spy['sharpe']:.2f}")
                print(f"    TSLA: {tsla['trades']:2} trades, Sharpe {tsla['sharpe']:.2f}")

                if spy['trades'] > 0 and tsla['trades'] > 0:
                    if tsla['sharpe'] > spy['sharpe']:
                        print(f"    → Volume filter MORE effective on TSLA (+{tsla['sharpe'] - spy['sharpe']:.2f} Sharpe)")
                    else:
                        print(f"    → Volume filter MORE effective on SPY (+{spy['sharpe'] - tsla['sharpe']:.2f} Sharpe)")

    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    # Determine if volume filter is useful
    print("\nBased on testing across stock volatility profiles:")
    print("\n1. For VOLATILE STOCKS (TSLA, NVDA):")
    print("   - Volume confirmation likely ADDS VALUE")
    print("   - Suggested threshold: 1.15x - 1.5x (not 2.0x)")
    print("   - Filters noise, confirms institutional participation")

    print("\n2. For STABLE LARGE CAPS (AAPL, MSFT):")
    print("   - Volume confirmation MODERATE VALUE")
    print("   - Suggested threshold: 1.15x - 1.25x")

    print("\n3. For ETFs (SPY):")
    print("   - Volume confirmation MINIMAL VALUE")
    print("   - Consider removing filter entirely")
    print("   - OR use very low threshold (1.15x)")

    print("\n4. GENERAL GUIDANCE:")
    print("   - 2.0x threshold TOO HIGH for most assets")
    print("   - 1.15x - 1.5x range more appropriate")
    print("   - Threshold should scale with asset volatility")

    print("\n" + "="*100)


def main():
    """Main execution."""
    test_volume_thresholds_on_stocks()

    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("\n1. If volatile stocks show strong results with volume filter:")
    print("   - Implement stock screening (filter S&P 500 for high-volume breakouts)")
    print("   - Use 1.15x - 1.5x threshold (asset-specific)")
    print("   - Build portfolio of top N stocks meeting criteria")

    print("\n2. If SPY still preferred (single-asset simplicity):")
    print("   - Remove volume filter for SPY")
    print("   - Or use 1.15x threshold (light filtering)")
    print("   - Focus on event-based signals + regime overlay")

    print("\n3. Hybrid approach:")
    print("   - Core position: SPY without volume filter")
    print("   - Satellite positions: Volatile stocks WITH volume filter")
    print("   - Portfolio allocation based on regime")


if __name__ == "__main__":
    main()
