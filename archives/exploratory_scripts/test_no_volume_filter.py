"""
Test Strategy Without Volume Filter

Based on validation results, we're getting only 13 trades with 1.25x volume filter.
Earlier testing showed 20 trades WITHOUT volume filter (test_exit_thresholds.py).

This script tests whether removing volume filter entirely gives better results.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from strategies.high_momentum_52w import HighMomentum52W
from strategies.base_strategy import StrategyConfig


def compare_with_without_volume_filter():
    """Compare performance with and without volume filter."""

    print("\n" + "="*100)
    print("TESTING: VOLUME FILTER vs NO FILTER")
    print("="*100)

    # Load data
    print("\nLoading SPY data (2005-2025)...")
    data = vbt.YFData.pull('SPY', start='2005-01-01', end='2025-01-01').get()
    print(f"Loaded {len(data)} days")

    initial_capital = 10000

    # Test configurations
    configs = [
        {'label': 'No Volume Filter', 'volume_mult': None},
        {'label': '1.15x Volume', 'volume_mult': 1.15},
        {'label': '1.25x Volume (Current)', 'volume_mult': 1.25},
        {'label': '1.5x Volume', 'volume_mult': 1.5},
    ]

    results = []

    print("\n" + "="*100)
    print(f"{'Configuration':<25} {'Entries':>10} {'Exits':>10} {'Trades':>10} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-"*100)

    for cfg in configs:
        # Create strategy config
        strategy_config = StrategyConfig(
            name="52-Week High Momentum",
            universe="SPY",
            rebalance_frequency="signal_based",
            regime_compatibility={
                'TREND_BULL': True,
                'TREND_NEUTRAL': True,
                'TREND_BEAR': False,
                'CRASH': False
            },
            risk_per_trade=0.02,
            max_positions=1
        )

        # Initialize strategy
        strategy = HighMomentum52W(
            config=strategy_config,
            atr_multiplier=2.5,
            volume_multiplier=cfg['volume_mult']
        )

        # Generate signals
        signals = strategy.generate_signals(data, regime=None)

        entry_count = signals['entry_signal'].sum()
        exit_count = signals['exit_signal'].sum()

        # Calculate position sizes
        position_sizes = strategy.calculate_position_size(
            data=data,
            capital=initial_capital,
            stop_distance=signals['stop_distance']
        )

        # Run backtest
        try:
            pf = vbt.Portfolio.from_signals(
                close=data['Close'],
                entries=signals['entry_signal'],
                exits=signals['exit_signal'],
                size=position_sizes,
                size_type='amount',
                init_cash=initial_capital,
                fees=0.0015,
                slippage=0.0015,
                freq='1D'
            )

            trades = pf.trades.records_readable
            trade_count = len(trades)
            sharpe = pf.sharpe_ratio
            cagr = pf.annualized_return
            max_dd = pf.max_drawdown
            total_return = pf.total_return

            win_rate = (trades['PnL'] > 0).sum() / trade_count if trade_count > 0 else 0

            print(f"{cfg['label']:<25} {entry_count:>10} {exit_count:>10} {trade_count:>10} {sharpe:>10.2f} {cagr:>9.1%} {max_dd:>9.1%}")

            results.append({
                'config': cfg['label'],
                'volume_mult': cfg['volume_mult'],
                'entry_events': entry_count,
                'exit_events': exit_count,
                'trades': trade_count,
                'sharpe': sharpe,
                'cagr': cagr,
                'max_dd': max_dd,
                'total_return': total_return,
                'win_rate': win_rate
            })

        except Exception as e:
            print(f"{cfg['label']:<25} {entry_count:>10} {exit_count:>10} {'ERROR':>10}")
            print(f"  Error: {e}")

    # Detailed comparison
    print("\n" + "="*100)
    print("DETAILED COMPARISON")
    print("="*100)

    print(f"\n{'Configuration':<25} {'Trades':>10} {'Sharpe':>10} {'CAGR':>10} {'Return':>12} {'MaxDD':>10} {'Win%':>10} {'Gate1':>10}")
    print("-"*100)

    for r in results:
        gate1_pass = (
            r['trades'] >= 20 and
            r['sharpe'] >= 0.8 and
            r['cagr'] >= 0.10 and
            r['max_dd'] >= -0.30 and
            0.50 <= r['win_rate'] <= 0.70
        )

        gate1_status = "PASS" if gate1_pass else "FAIL"

        print(f"{r['config']:<25} {r['trades']:>10} {r['sharpe']:>10.2f} {r['cagr']:>9.1%} {r['total_return']:>11.1%} {r['max_dd']:>9.1%} {r['win_rate']:>9.1%} {gate1_status:>10}")

    # Recommendation
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)

    # Find best configuration
    valid_results = [r for r in results if r['trades'] >= 10]

    if len(valid_results) > 0:
        best_sharpe = max(valid_results, key=lambda x: x['sharpe'])
        best_cagr = max(valid_results, key=lambda x: x['cagr'])
        most_trades = max(valid_results, key=lambda x: x['trades'])

        print(f"\nBest Sharpe: {best_sharpe['config']} ({best_sharpe['sharpe']:.2f})")
        print(f"Best CAGR: {best_cagr['config']} ({best_cagr['cagr']:.1%})")
        print(f"Most Trades: {most_trades['config']} ({most_trades['trades']} trades)")

        # Check if any configuration passes Gate 1
        passing_configs = [r for r in results if (
            r['trades'] >= 20 and
            r['sharpe'] >= 0.8 and
            r['cagr'] >= 0.10 and
            r['max_dd'] >= -0.30
        )]

        if len(passing_configs) > 0:
            print(f"\nCONFIGURATIONS PASSING GATE 1: {len(passing_configs)}")
            for r in passing_configs:
                print(f"  - {r['config']}: Sharpe {r['sharpe']:.2f}, CAGR {r['cagr']:.1%}, {r['trades']} trades")
        else:
            print(f"\nNO CONFIGURATIONS PASS GATE 1")
            print("\nClosest to targets:")
            # Find configuration closest to passing all targets
            for r in results:
                targets_met = 0
                if r['trades'] >= 20: targets_met += 1
                if r['sharpe'] >= 0.8: targets_met += 1
                if r['cagr'] >= 0.10: targets_met += 1
                if r['max_dd'] >= -0.30: targets_met += 1
                if 0.50 <= r['win_rate'] <= 0.70: targets_met += 1

                if targets_met >= 3:
                    print(f"  - {r['config']}: {targets_met}/5 targets met")

    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    no_filter = next((r for r in results if r['volume_mult'] is None), None)
    with_filter = next((r for r in results if r['volume_mult'] == 1.25), None)

    if no_filter and with_filter:
        print(f"\nNo Filter vs 1.25x Filter:")
        print(f"  No Filter:   {no_filter['trades']:2} trades, Sharpe {no_filter['sharpe']:.2f}, CAGR {no_filter['cagr']:.1%}")
        print(f"  1.25x Filter: {with_filter['trades']:2} trades, Sharpe {with_filter['sharpe']:.2f}, CAGR {with_filter['cagr']:.1%}")

        if no_filter['sharpe'] > with_filter['sharpe']:
            sharpe_improvement = no_filter['sharpe'] - with_filter['sharpe']
            print(f"\n  Volume filter REDUCES performance by {sharpe_improvement:.2f} Sharpe points")
        else:
            sharpe_improvement = with_filter['sharpe'] - no_filter['sharpe']
            print(f"\n  Volume filter IMPROVES performance by {sharpe_improvement:.2f} Sharpe points")

        if no_filter['trades'] > with_filter['trades']:
            trade_diff = no_filter['trades'] - with_filter['trades']
            print(f"  Volume filter REDUCES trade count by {trade_diff} trades")
        else:
            trade_diff = with_filter['trades'] - no_filter['trades']
            print(f"  Volume filter INCREASES trade count by {trade_diff} trades")

    print("\n" + "="*100)

    return results


def main():
    """Main execution."""
    results = compare_with_without_volume_filter()

    print("\nKEY FINDINGS:")
    print("1. Academic research (George & Hwang 2004) does NOT specify volume requirement")
    print("2. Volume filter may be reducing trade count below minimum threshold")
    print("3. Testing across thresholds shows tradeoff: filter quality vs quantity")
    print("\nNext steps depend on whether any configuration passes Gate 1.")


if __name__ == "__main__":
    main()
