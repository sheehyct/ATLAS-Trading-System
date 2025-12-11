"""
Debug Backtest Logic - Why are signals not converting to trades?

The debug script shows:
- No volume filter: 3,905 entry signals
- But backtest executes: only 3 trades

This script investigates the backtest execution logic to find the disconnect.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt


def load_spy_data() -> pd.DataFrame:
    """Load SPY data."""
    print("Loading SPY data...")
    spy_data = vbt.YFData.pull('SPY', start='2005-01-01', end='2025-01-01').get()
    print(f"Loaded {len(spy_data)} days")
    return spy_data


def debug_backtest_execution(data: pd.DataFrame):
    """
    Debug why signals aren't converting to trades.

    Hypothesis: Issue is in how we're passing signals/sizes to VBT.
    """
    print("\n" + "="*70)
    print("DEBUGGING BACKTEST EXECUTION")
    print("="*70)

    # Generate signals (NO volume filter)
    print("\n1. Generating signals (no volume filter)...")
    high_52w = data['High'].rolling(window=252, min_periods=252).max()
    distance = data['Close'] / high_52w
    volume_ma = data['Volume'].rolling(window=20, min_periods=20).mean()

    # Calculate ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # Entry signal (no volume filter)
    entries = (
        (distance >= 0.90) &
        high_52w.notna() &
        volume_ma.notna() &
        atr.notna()
    )

    # Exit signal
    exits = (distance < 0.70) & high_52w.notna()

    print(f"   Entry signals (TRUE count): {entries.sum()}")
    print(f"   Exit signals (TRUE count): {exits.sum()}")

    # Position sizing
    print("\n2. Calculating position sizes...")
    capital = 10000
    risk_pct = 0.02
    atr_mult = 2.5
    stop_distance = atr * atr_mult

    # Risk-based sizing
    risk_amount = capital * risk_pct  # $200
    position_sizes = risk_amount / stop_distance  # shares based on risk

    # Capital constraint
    max_shares = capital / data['Close']
    position_sizes = np.minimum(position_sizes, max_shares)

    # Fill NaN with 0
    position_sizes = position_sizes.fillna(0)

    # Integer shares
    position_sizes_int = position_sizes.astype(int)

    print(f"   Position sizes calculated: {len(position_sizes_int)} values")
    print(f"   Non-zero position sizes: {(position_sizes_int > 0).sum()}")
    print(f"   Mean position size (where >0): {position_sizes_int[position_sizes_int > 0].mean():.1f} shares")

    # Size array FOR VBT
    print("\n3. Creating size array for VBT...")

    # ORIGINAL APPROACH (from test_volume_thresholds.py)
    size_original = pd.Series(0, index=data.index)
    size_original[entries] = position_sizes_int[entries]

    print(f"   Size array (original approach):")
    print(f"     Total length: {len(size_original)}")
    print(f"     Non-zero values: {(size_original > 0).sum()}")
    print(f"     Sum of all sizes: {size_original.sum()}")

    # ALTERNATIVE APPROACH: Use position sizes directly
    size_direct = position_sizes_int.copy()

    print(f"\n   Size array (direct approach):")
    print(f"     Total length: {len(size_direct)}")
    print(f"     Non-zero values: {(size_direct > 0).sum()}")
    print(f"     Sum of all sizes: {size_direct.sum()}")

    # Show first few entry dates with sizes
    entry_dates = entries[entries].index[:10]
    print(f"\n   First 10 entry dates and sizes:")
    for date in entry_dates:
        print(f"     {date.date()}: {size_original[date]} shares (Close: ${data.loc[date, 'Close']:.2f})")

    # Test VBT backtest with ORIGINAL approach
    print("\n4. Running VBT backtest (ORIGINAL size approach)...")
    try:
        pf_original = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=entries,
            exits=exits,
            size=size_original,
            size_type='amount',
            init_cash=capital,
            fees=0.0015,
            slippage=0.0015,
            freq='1D'
        )

        trades_original = pf_original.trades.records_readable
        print(f"   Trades executed (original): {len(trades_original)}")

        if len(trades_original) > 0:
            print("\n   First few trades:")
            print(trades_original[['Entry Date', 'Exit Date', 'Size', 'Entry Price', 'Exit Price', 'PnL']].head())
        else:
            print("   NO TRADES EXECUTED!")

    except Exception as e:
        print(f"   [FAIL] Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    # Test VBT backtest with DIRECT approach
    print("\n5. Running VBT backtest (DIRECT size approach)...")
    try:
        pf_direct = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=entries,
            exits=exits,
            size=size_direct,
            size_type='amount',
            init_cash=capital,
            fees=0.0015,
            slippage=0.0015,
            freq='1D'
        )

        trades_direct = pf_direct.trades.records_readable
        print(f"   Trades executed (direct): {len(trades_direct)}")

        if len(trades_direct) > 0:
            print("\n   First few trades:")
            print(trades_direct[['Entry Date', 'Exit Date', 'Size', 'Entry Price', 'Exit Price', 'PnL']].head())

    except Exception as e:
        print(f"   [FAIL] Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    # Test VBT backtest with FIXED SIZE (simplest possible)
    print("\n6. Running VBT backtest (FIXED 10 shares)...")
    try:
        size_fixed = pd.Series(10, index=data.index)

        pf_fixed = vbt.Portfolio.from_signals(
            close=data['Close'],
            entries=entries,
            exits=exits,
            size=size_fixed,
            size_type='amount',
            init_cash=capital,
            fees=0.0015,
            slippage=0.0015,
            freq='1D'
        )

        trades_fixed = pf_fixed.trades.records_readable
        print(f"   Trades executed (fixed 10 shares): {len(trades_fixed)}")

        if len(trades_fixed) > 0:
            print("\n   First few trades:")
            print(trades_fixed[['Entry Date', 'Exit Date', 'Size', 'Entry Price', 'Exit Price', 'PnL']].head(10))

    except Exception as e:
        print(f"   [FAIL] Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("BACKTEST LOGIC DEBUG")
    print("="*70)
    print("\nGoal: Understand why 3,905 signals become only 3 trades")

    data = load_spy_data()
    debug_backtest_execution(data)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review VBT from_signals() documentation")
    print("2. Check if 'accumulate=True' or other parameters needed")
    print("3. Verify signal alignment vs VBT expectations")


if __name__ == "__main__":
    main()
