"""
Test Different Exit Thresholds

Current issue: Exit at distance < 0.70 (30% off highs) is too extreme.
SPY rarely drops that much, so we have:
- 54 entry events
- 5 exit events
- Only 3 complete trade cycles

Test alternative exit thresholds to find optimal entry/exit balance.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt


def generate_signals_with_exit_threshold(
    data: pd.DataFrame,
    entry_threshold: float = 0.90,
    exit_threshold: float = 0.70
) -> dict:
    """Generate event signals with configurable exit threshold."""

    # Calculate components
    high_52w = data['High'].rolling(window=252, min_periods=252).max()
    distance = data['Close'] / high_52w
    volume_ma = data['Volume'].rolling(window=20, min_periods=20).mean()

    # ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # States
    in_entry_zone = (distance >= entry_threshold) & high_52w.notna() & atr.notna()
    in_exit_zone = (distance < exit_threshold) & high_52w.notna()

    # Events (state transitions)
    entry_events = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
    exit_events = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

    return {
        'entry_signal': entry_events.fillna(False),
        'exit_signal': exit_events.fillna(False),
        'distance': distance
    }


def test_exit_thresholds():
    """Test different exit thresholds to find optimal balance."""

    print("\n" + "="*90)
    print("TESTING EXIT THRESHOLDS")
    print("="*90)

    # Load data
    print("\nLoading SPY data...")
    data = vbt.YFData.pull('SPY', start='2005-01-01', end='2025-01-01').get()
    print(f"Loaded {len(data)} days")

    # Entry threshold fixed at 0.90 (within 10% of highs)
    entry_threshold = 0.90

    # Test different exit thresholds
    exit_thresholds = [
        (0.70, "0.70 (Current - 30% off highs)"),
        (0.75, "0.75 (25% off highs)"),
        (0.80, "0.80 (20% off highs)"),
        (0.85, "0.85 (15% off highs)"),
        (0.88, "0.88 (12% off highs - exit zone boundary)"),
    ]

    results = []

    print("\n" + "="*90)
    print(f"{'Exit Threshold':<40} {'Entry Events':>12} {'Exit Events':>12} {'Trades':>10} {'Sharpe':>8}")
    print("-"*90)

    for exit_value, exit_label in exit_thresholds:
        # Generate signals
        signals = generate_signals_with_exit_threshold(
            data,
            entry_threshold=entry_threshold,
            exit_threshold=exit_value
        )

        entry_count = signals['entry_signal'].sum()
        exit_count = signals['exit_signal'].sum()

        # Run backtest
        try:
            pf = vbt.Portfolio.from_signals(
                close=data['Close'],
                entries=signals['entry_signal'],
                exits=signals['exit_signal'],
                size=pd.Series(10, index=data.index),  # Fixed 10 shares
                size_type='amount',
                init_cash=10000,
                fees=0.0015,
                slippage=0.0015,
                freq='1D'
            )

            trades = pf.trades.records_readable
            trade_count = len(trades)
            sharpe = pf.sharpe_ratio
            total_return = pf.total_return
            max_dd = pf.max_drawdown

            print(f"{exit_label:<40} {entry_count:>12} {exit_count:>12} {trade_count:>10} {sharpe:>8.2f}")

            results.append({
                'exit_threshold': exit_value,
                'exit_label': exit_label,
                'entry_events': entry_count,
                'exit_events': exit_count,
                'trades': trade_count,
                'sharpe': sharpe,
                'total_return': total_return,
                'max_dd': max_dd
            })

        except Exception as e:
            print(f"{exit_label:<40} {entry_count:>12} {exit_count:>12} {'ERROR':>10}")
            print(f"  Error: {e}")

    # Detailed results
    print("\n" + "="*90)
    print("DETAILED RESULTS")
    print("="*90)

    print(f"\n{'Exit Threshold':<40} {'Trades':>10} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10}")
    print("-"*90)

    for r in results:
        print(f"{r['exit_label']:<40} {r['trades']:>10} {r['sharpe']:>8.2f} {r['total_return']:>9.1%} {r['max_dd']:>9.1%}")

    # Recommendation
    print("\n" + "="*90)
    print("RECOMMENDATION")
    print("="*90)

    # Find threshold with best balance (enough trades, good Sharpe)
    valid_results = [r for r in results if r['trades'] >= 10]

    if len(valid_results) > 0:
        best = max(valid_results, key=lambda x: x['sharpe'])
        print(f"\nRECOMMENDED: {best['exit_label']}")
        print(f"  Entry events: {best['entry_events']}")
        print(f"  Exit events: {best['exit_events']}")
        print(f"  Trades: {best['trades']}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Total Return: {best['total_return']:.1%}")
        print(f"  Max Drawdown: {best['max_dd']:.1%}")

        print(f"\nRATIONALE:")
        print(f"  - Entry at distance >= 0.90 (within 10% of 52w high)")
        print(f"  - Exit at distance < {best['exit_threshold']:.2f}")
        print(f"  - Creates balanced entry/exit cycle")
        print(f"  - Produces {best['trades']} trades over 20 years (avg {best['trades']/20:.1f}/year)")

    else:
        print("\nNO THRESHOLD produces sufficient trades")
        print("\nAlternative approach needed:")
        print("1. Consider semi-annual REBALANCE strategy (not signal-based)")
        print("2. Or use 'stay-in-zone' logic (hold while >= 0.90, exit when < 0.90)")

    print("="*90)


def main():
    """Main execution."""
    print("\n" + "="*90)
    print("EXIT THRESHOLD OPTIMIZATION")
    print("="*90)

    print("\nProblem: Current exit threshold (0.70 = 30% off highs) is too extreme")
    print("Result: Many entry events, few exit events, only 3 complete trade cycles")
    print("\nSolution: Test tighter exit thresholds to create more balanced cycles")

    test_exit_thresholds()


if __name__ == "__main__":
    main()
