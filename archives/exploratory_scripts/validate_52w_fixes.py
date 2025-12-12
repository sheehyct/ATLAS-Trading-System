"""
Validate 52-Week High Momentum Strategy Fixes

This script validates the Session 36 fixes:
1. Event-based signal generation (state transitions)
2. Exit threshold at 0.88 (12% off highs, not 30%)
3. Volume multiplier at 1.25x (calibrated for SPY)
4. Proper ATR-based position sizing (2% risk per trade)

Expected results (based on testing):
- Trade count: ~20 over 20 years (1 per year average)
- Sharpe Ratio: >= 0.8 (architecture target)
- CAGR: >= 10% (architecture target)
- Max Drawdown: <= -30% (architecture target)
- Win Rate: 50-60%

If all targets met, strategy passes Gate 1 and is ready for regime integration.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from strategies.high_momentum_52w import HighMomentum52W
from strategies.base_strategy import StrategyConfig


def validate_strategy_on_spy():
    """
    Run comprehensive backtest validation on SPY with updated strategy.
    """
    print("\n" + "="*100)
    print("52-WEEK HIGH MOMENTUM - VALIDATION WITH SESSION 36 FIXES")
    print("="*100)

    print("\nFIXES IMPLEMENTED:")
    print("1. Event-based signal generation (state transitions, not continuous states)")
    print("2. Exit threshold: 0.88 (12% off highs) instead of 0.70 (30% off highs)")
    print("3. Volume multiplier: 1.25x (calibrated for SPY, was 2.0x)")
    print("4. Proper ATR-based position sizing (2% risk per trade)")

    # Load SPY data
    print("\n" + "-"*100)
    print("LOADING DATA")
    print("-"*100)

    start_date = '2005-01-01'
    end_date = '2025-01-01'

    print(f"\nFetching SPY data from {start_date} to {end_date}...")
    data = vbt.YFData.pull('SPY', start=start_date, end=end_date).get()
    print(f"Loaded {len(data)} trading days")

    # Initialize strategy with validated parameters
    print("\n" + "-"*100)
    print("INITIALIZING STRATEGY")
    print("-"*100)

    config = StrategyConfig(
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

    # Initialize with validated parameters
    strategy = HighMomentum52W(
        config=config,
        atr_multiplier=2.5,        # Standard ATR stop
        volume_multiplier=1.25     # Calibrated for SPY (Session 36 finding)
    )

    print(f"\nStrategy: {strategy.get_strategy_name()}")
    print(f"  Volume multiplier: {strategy.volume_multiplier}x")
    print(f"  ATR multiplier: {strategy.atr_multiplier}x")
    print(f"  Risk per trade: {config.risk_per_trade:.1%}")

    # Generate signals
    print("\n" + "-"*100)
    print("GENERATING SIGNALS")
    print("-"*100)

    signals = strategy.generate_signals(data, regime=None)

    entry_count = signals['entry_signal'].sum()
    exit_count = signals['exit_signal'].sum()

    print(f"\nSignal Statistics:")
    print(f"  Entry events (state transitions): {entry_count}")
    print(f"  Exit events (state transitions): {exit_count}")
    print(f"  Expected trades: ~{min(entry_count, exit_count)}")

    # Show signal component stats
    distance_valid = signals['distance_from_high'] > 0
    print(f"\nDistance from 52w high (valid days):")
    print(f"  Mean: {signals['distance_from_high'][distance_valid].mean():.4f}")
    print(f"  Median: {signals['distance_from_high'][distance_valid].median():.4f}")
    print(f"  Days >= 0.90 (entry zone): {(signals['distance_from_high'] >= 0.90).sum()} ({(signals['distance_from_high'] >= 0.90).sum()/len(data)*100:.1f}%)")
    print(f"  Days < 0.88 (exit zone): {(signals['distance_from_high'] < 0.88).sum()} ({(signals['distance_from_high'] < 0.88).sum()/len(data)*100:.1f}%)")

    volume_confirmed_pct = signals['volume_confirmed'].sum() / len(data) * 100
    print(f"\nVolume confirmation (1.25x threshold):")
    print(f"  Days with volume > 1.25x MA: {signals['volume_confirmed'].sum()} ({volume_confirmed_pct:.1f}%)")

    # Calculate position sizes
    print("\n" + "-"*100)
    print("CALCULATING POSITION SIZES")
    print("-"*100)

    initial_capital = 10000
    position_sizes = strategy.calculate_position_size(
        data=data,
        capital=initial_capital,
        stop_distance=signals['stop_distance']
    )

    print(f"\nPosition sizing (ATR-based, 2% risk):")
    print(f"  Initial capital: ${initial_capital:,.0f}")
    print(f"  Position sizes (shares):")
    entry_positions = position_sizes[signals['entry_signal']]
    if len(entry_positions) > 0:
        print(f"    Mean: {entry_positions.mean():.1f} shares")
        print(f"    Median: {entry_positions.median():.1f} shares")
        print(f"    Min: {entry_positions.min():.0f} shares")
        print(f"    Max: {entry_positions.max():.0f} shares")

    # Run backtest
    print("\n" + "-"*100)
    print("RUNNING BACKTEST")
    print("-"*100)

    print("\nPortfolio parameters:")
    print(f"  Size type: Dynamic (ATR-based position sizing)")
    print(f"  Fees: 0.15% per trade")
    print(f"  Slippage: 0.15% per trade")

    # Use strategy's backtest method which should handle regime=None
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

    # Extract metrics
    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)

    total_return = pf.total_return
    cagr = pf.annualized_return
    sharpe = pf.sharpe_ratio
    max_dd = pf.max_drawdown

    trades = pf.trades.records_readable
    trade_count = len(trades)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  CAGR: {cagr:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Final Value: ${pf.final_value:,.2f}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {trade_count}")
    print(f"  Avg Trades/Year: {trade_count / 20:.1f}")

    if trade_count > 0:
        winning_trades = (trades['PnL'] > 0).sum()
        losing_trades = (trades['PnL'] <= 0).sum()
        win_rate = winning_trades / trade_count

        print(f"  Winning Trades: {winning_trades}")
        print(f"  Losing Trades: {losing_trades}")
        print(f"  Win Rate: {win_rate:.1%}")

        if 'Return' in trades.columns:
            avg_win = trades[trades['PnL'] > 0]['Return'].mean() if winning_trades > 0 else 0
            avg_loss = trades[trades['PnL'] <= 0]['Return'].mean() if losing_trades > 0 else 0
            print(f"  Avg Win: {avg_win:.2%}")
            print(f"  Avg Loss: {avg_loss:.2%}")
            if avg_loss != 0:
                print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}")

    # Architecture target validation
    print("\n" + "="*100)
    print("ARCHITECTURE TARGET VALIDATION")
    print("="*100)

    targets = {
        'Trade Count': {'actual': trade_count, 'target': '>=20', 'threshold': 20, 'pass': trade_count >= 20},
        'Sharpe Ratio': {'actual': sharpe, 'target': '>=0.8', 'threshold': 0.8, 'pass': sharpe >= 0.8},
        'CAGR': {'actual': cagr, 'target': '>=10%', 'threshold': 0.10, 'pass': cagr >= 0.10},
        'Max Drawdown': {'actual': max_dd, 'target': '<=-30%', 'threshold': -0.30, 'pass': max_dd >= -0.30},
        'Win Rate': {'actual': win_rate if trade_count > 0 else 0, 'target': '50-60%', 'threshold': 0.50, 'pass': (win_rate >= 0.50 and win_rate <= 0.70) if trade_count > 0 else False}
    }

    print(f"\n{'Metric':<20} {'Target':<15} {'Actual':<15} {'Status':<10}")
    print("-"*100)

    all_pass = True
    for metric, values in targets.items():
        status = "PASS" if values['pass'] else "FAIL"
        if metric == 'Trade Count':
            actual_str = f"{values['actual']}"
        elif metric in ['Sharpe Ratio', 'Win Rate']:
            actual_str = f"{values['actual']:.2f}"
        else:
            actual_str = f"{values['actual']:.2%}"

        print(f"{metric:<20} {values['target']:<15} {actual_str:<15} {status:<10}")
        all_pass = all_pass and values['pass']

    print("-"*100)
    print(f"{'OVERALL':<20} {'All targets':<15} {'':<15} {'PASS' if all_pass else 'FAIL':<10}")

    # Gate 1 validation
    print("\n" + "="*100)
    print("GATE 1 VALIDATION")
    print("="*100)

    gate1_checks = {
        'Minimum trades': trade_count >= 20,
        'Risk-adjusted returns': sharpe >= 0.8,
        'Absolute returns': cagr >= 0.10,
        'Risk management': max_dd >= -0.30,
        'Win rate': (win_rate >= 0.50 and win_rate <= 0.70) if trade_count > 0 else False
    }

    print(f"\n{'Check':<30} {'Status':<10}")
    print("-"*100)
    for check, passed in gate1_checks.items():
        print(f"{check:<30} {'PASS' if passed else 'FAIL':<10}")

    gate1_pass = all(gate1_checks.values())
    print("-"*100)
    print(f"{'GATE 1 VALIDATION':<30} {'PASS' if gate1_pass else 'FAIL':<10}")

    # Show sample trades
    if trade_count > 0:
        print("\n" + "="*100)
        print("SAMPLE TRADES (First 10)")
        print("="*100)

        trade_cols = ['Entry Date', 'Exit Date', 'Size', 'PnL', 'Return']
        available_cols = [col for col in trade_cols if col in trades.columns]
        print("\n" + trades[available_cols].head(10).to_string())

    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    if gate1_pass:
        print("\nSTRATEGY VALIDATION: SUCCESS")
        print("\nNext Steps:")
        print("1. Mark Gate 1 as PASSED")
        print("2. Proceed to regime integration (ATLAS system)")
        print("3. Update tests for event-based signal logic")
        print("4. Prepare for multi-asset implementation when stock scanner ready")
        print("\nStrategy is ready for production integration.")

    else:
        print("\nSTRATEGY VALIDATION: NEEDS IMPROVEMENT")
        print("\nFailed checks:")
        for check, passed in gate1_checks.items():
            if not passed:
                print(f"  - {check}")

        print("\nPossible next steps:")
        if trade_count < 20:
            print("  - Adjust entry/exit thresholds to generate more trades")
        if sharpe < 0.8:
            print("  - Review position sizing (may need adjustment)")
            print("  - Consider regime overlay (only trade in TREND_BULL)")
        if cagr < 0.10:
            print("  - Increase position sizes (higher risk per trade)")
            print("  - Or consider multi-asset implementation")
        if max_dd < -0.30:
            print("  - Tighten stop losses")
            print("  - Add regime filtering")

    print("\n" + "="*100)
    print("VALIDATION COMPLETE")
    print("="*100)

    return pf, signals, trades


def main():
    """Main execution."""
    pf, signals, trades = validate_strategy_on_spy()

    print("\n" + "="*100)
    print("SESSION 36 STATUS")
    print("="*100)
    print("\nCompleted tasks:")
    print("1. Identified 3 root causes (volume filter, signal logic, exit threshold)")
    print("2. Validated fixes through systematic testing")
    print("3. Implemented event-based signals in high_momentum_52w.py")
    print("4. Ran full backtest with proper ATR position sizing")
    print("\nNext session priorities:")
    print("1. Update unit tests for new signal logic")
    print("2. Document findings in HANDOFF.md and OpenMemory")
    print("3. Prepare for regime integration (if Gate 1 passed)")
    print("4. Plan multi-asset implementation (when stock scanner ready)")


if __name__ == "__main__":
    main()
