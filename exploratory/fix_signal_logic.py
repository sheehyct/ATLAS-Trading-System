"""
Fix Signal Logic - Convert States to Events

ROOT CAUSE IDENTIFIED:
Current implementation generates STATE signals (is price near highs every day)
VBT needs EVENT signals (cross into near-highs state, cross out of state)

Example of the problem:
- Day 1-100: distance >= 0.90 (TRUE every day)
- Current: entry_signal = TRUE on all 100 days
- VBT behavior: Enter on day 1, IGNORE days 2-100 (already in position)
- Result: Only 1 trade instead of maintaining position for 100 days

Fix:
Convert continuous conditions to entry/exit EVENTS using state changes.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt


def generate_event_signals(data: pd.DataFrame, volume_threshold: float = None) -> dict:
    """
    Generate EVENT-BASED signals (state transitions) instead of STATE signals.

    Logic:
    - Entry: First day crossing INTO distance >= 0.90 zone (state change)
    - Stay in position while distance >= 0.70 (hysteresis/buffer)
    - Exit: First day crossing BELOW distance < 0.70 (state change)

    This creates proper entry/exit events that VBT can process correctly.

    Args:
        data: OHLCV DataFrame
        volume_threshold: Optional volume filter multiplier

    Returns:
        Dictionary with entry_signal, exit_signal, stop_distance
    """
    # Calculate components
    high_52w = data['High'].rolling(window=252, min_periods=252).max()
    distance = data['Close'] / high_52w
    volume_ma = data['Volume'].rolling(window=20, min_periods=20).mean()

    # ATR for stops
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # STATE conditions (continuous)
    in_entry_zone = distance >= 0.90  # Near highs
    in_exit_zone = distance < 0.70    # Far from highs

    # Apply volume filter if specified
    if volume_threshold is not None:
        volume_ok = (data['Volume'] > (volume_ma * volume_threshold)) & volume_ma.notna()
        in_entry_zone = in_entry_zone & volume_ok

    # Ensure valid data
    in_entry_zone = in_entry_zone & high_52w.notna() & atr.notna()
    in_exit_zone = in_exit_zone & high_52w.notna()

    # Convert states to EVENTS (state transitions)
    # Entry: Transition from FALSE to TRUE (cross into entry zone)
    entry_events = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)

    # Exit: Transition from not-in-exit-zone to in-exit-zone
    # OR: Use explicit exit (crossing below 0.70)
    exit_events = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

    # Stop distance
    stop_distance = (atr * 2.5).fillna(0.0)

    return {
        'entry_signal': entry_events.fillna(False),
        'exit_signal': exit_events.fillna(False),
        'stop_distance': stop_distance,
        'distance': distance  # For debugging
    }


def test_event_signals():
    """
    Test event signal generation vs state signal generation.
    """
    print("\n" + "="*70)
    print("TESTING EVENT-BASED SIGNALS")
    print("="*70)

    # Load data
    print("\nLoading SPY data...")
    data = vbt.YFData.pull('SPY', start='2005-01-01', end='2025-01-01').get()
    print(f"Loaded {len(data)} days")

    # Test 1: Event signals with NO volume filter
    print("\n" + "-"*70)
    print("TEST 1: Event signals (no volume filter)")
    print("-"*70)

    signals = generate_event_signals(data, volume_threshold=None)

    entry_count = signals['entry_signal'].sum()
    exit_count = signals['exit_signal'].sum()

    print(f"Entry events (state transitions): {entry_count}")
    print(f"Exit events (state transitions): {exit_count}")
    print(f"Expected trades: ~{min(entry_count, exit_count)}")

    # Show first few entry/exit dates
    entry_dates = signals['entry_signal'][signals['entry_signal']].index[:10]
    print(f"\nFirst 10 entry dates:")
    for date in entry_dates:
        print(f"  {date.date()}: distance = {signals['distance'][date]:.4f}")

    # Run backtest with event signals
    print("\nRunning backtest with event signals...")

    # Fixed position size for simplicity
    size = pd.Series(10, index=data.index)

    pf = vbt.Portfolio.from_signals(
        close=data['Close'],
        entries=signals['entry_signal'],
        exits=signals['exit_signal'],
        size=size,
        size_type='amount',
        init_cash=10000,
        fees=0.0015,
        slippage=0.0015,
        freq='1D'
    )

    trades = pf.trades.records_readable
    print(f"\nTrades executed: {len(trades)}")

    if len(trades) > 0:
        print(f"\nBacktest metrics:")
        print(f"  Total Return: {pf.total_return:.2%}")
        print(f"  Sharpe Ratio: {pf.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {pf.max_drawdown:.2%}")

        # Show first few trades
        print(f"\nFirst 5 trades:")
        trade_cols = [col for col in ['Size', 'PnL', 'Return'] if col in trades.columns]
        print(trades[trade_cols].head())
    else:
        print("\nNO TRADES EXECUTED - Event logic may need adjustment")

    # Test 2: Event signals with 1.5x volume filter
    print("\n" + "-"*70)
    print("TEST 2: Event signals (1.5x volume filter)")
    print("-"*70)

    signals_filtered = generate_event_signals(data, volume_threshold=1.5)

    entry_count_filtered = signals_filtered['entry_signal'].sum()
    exit_count_filtered = signals_filtered['exit_signal'].sum()

    print(f"Entry events (with volume filter): {entry_count_filtered}")
    print(f"Exit events: {exit_count_filtered}")
    print(f"Expected trades: ~{min(entry_count_filtered, exit_count_filtered)}")

    # Run backtest with filtered signals
    print("\nRunning backtest with volume-filtered signals...")

    pf_filtered = vbt.Portfolio.from_signals(
        close=data['Close'],
        entries=signals_filtered['entry_signal'],
        exits=signals_filtered['exit_signal'],
        size=size,
        size_type='amount',
        init_cash=10000,
        fees=0.0015,
        slippage=0.0015,
        freq='1D'
    )

    trades_filtered = pf_filtered.trades.records_readable
    print(f"\nTrades executed: {len(trades_filtered)}")

    if len(trades_filtered) > 0:
        print(f"\nBacktest metrics:")
        print(f"  Total Return: {pf_filtered.total_return:.2%}")
        print(f"  Sharpe Ratio: {pf_filtered.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {pf_filtered.max_drawdown:.2%}")

    print("\n" + "="*70)
    print("EVENT SIGNAL TESTING COMPLETE")
    print("="*70)

    return signals, signals_filtered, pf, pf_filtered


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("SIGNAL LOGIC FIX - STATE TO EVENT CONVERSION")
    print("="*70)

    print("\nProblem: Current signals are STATES (is condition true?)")
    print("Solution: Generate EVENTS (transition into/out of state)")
    print("\nThis matches VBT's expectation for discrete entry/exit signals")

    test_event_signals()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. If event signals generate reasonable trade count:")
    print("   - Update high_momentum_52w.py with event-based logic")
    print("   - Re-run full backtest validation")
    print("2. If event signals still generate too few trades:")
    print("   - Review entry/exit zone thresholds (0.90/0.70)")
    print("   - Consider removing volume filter entirely")


if __name__ == "__main__":
    main()
