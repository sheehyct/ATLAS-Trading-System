"""
Debug Script: 52-Week High Momentum Signal Generation Analysis

Purpose: Identify why the 52W High strategy generates only 3 trades in 20 years.

This script:
1. Loads SPY data (2005-2025)
2. Calculates all signal components step-by-step
3. Exports detailed CSV for manual inspection
4. Generates statistics on condition frequencies
5. Identifies which filter is blocking signals

Expected findings:
- If volume_ok days < 50: Volume filter is the bottleneck
- If distance_ok days > 500 but entry_signals < 10: Volume filter is blocking
- If volume_ratio rarely exceeds 1.5x: 2.0x threshold is too high for SPY

Output:
- 52w_high_signals_debug.csv (daily signal components)
- Console statistics showing bottleneck analysis
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt


def load_spy_data(start_date: str = '2005-01-01', end_date: str = '2025-01-01') -> pd.DataFrame:
    """
    Load SPY historical data using VectorBT Pro.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n{'='*70}")
    print("LOADING SPY DATA")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}")

    spy_data = vbt.YFData.pull('SPY', start=start_date, end=end_date).get()

    print(f"Total trading days: {len(spy_data)}")
    print(f"Actual range: {spy_data.index[0]} to {spy_data.index[-1]}")

    return spy_data


def calculate_signal_components(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all signal components for debugging.

    Replicates the exact logic from high_momentum_52w.py generate_signals()
    but returns intermediate values for analysis.

    Args:
        data: OHLCV DataFrame

    Returns:
        DataFrame with all signal components
    """
    print(f"\n{'='*70}")
    print("CALCULATING SIGNAL COMPONENTS")
    print(f"{'='*70}")

    # Calculate 52-week high (252 trading days)
    print("Calculating 52-week high (252-day rolling max)...")
    high_52w = data['High'].rolling(window=252, min_periods=252).max()

    # Calculate distance from 52-week high
    print("Calculating distance from 52w high...")
    distance_from_high = data['Close'] / high_52w

    # Calculate volume moving average (20-day)
    print("Calculating 20-day volume moving average...")
    volume_ma_20 = data['Volume'].rolling(window=20, min_periods=20).mean()

    # Volume ratio (current volume / 20-day average)
    print("Calculating volume ratio...")
    volume_ratio = data['Volume'] / volume_ma_20

    # Calculate ATR for completeness (14-period standard)
    print("Calculating ATR (14-period)...")
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # Boolean conditions (matching strategy implementation)
    print("Calculating boolean conditions...")

    # Distance condition: Within 10% of 52w high
    distance_ok = (distance_from_high >= 0.90) & high_52w.notna()

    # Volume condition: 2.0x threshold (CURRENT IMPLEMENTATION)
    volume_ok_2x = (volume_ratio > 2.0) & volume_ma_20.notna()

    # Alternative volume conditions for comparison
    volume_ok_1_5x = (volume_ratio > 1.5) & volume_ma_20.notna()
    volume_ok_1_25x = (volume_ratio > 1.25) & volume_ma_20.notna()
    volume_ok_none = volume_ma_20.notna()  # No volume filter

    # Entry signals (all conditions must be TRUE)
    entry_signal_2x = distance_ok & volume_ok_2x & atr.notna()
    entry_signal_1_5x = distance_ok & volume_ok_1_5x & atr.notna()
    entry_signal_1_25x = distance_ok & volume_ok_1_25x & atr.notna()
    entry_signal_none = distance_ok & volume_ok_none & atr.notna()

    # Exit signal: 30% off highs
    exit_signal = (distance_from_high < 0.70) & high_52w.notna()

    # Create debug DataFrame
    debug_df = pd.DataFrame({
        'Close': data['Close'],
        'High': data['High'],
        'Low': data['Low'],
        'Volume': data['Volume'],
        'high_52w': high_52w,
        'distance': distance_from_high,
        'volume_ma_20': volume_ma_20,
        'volume_ratio': volume_ratio,
        'atr': atr,
        'distance_ok': distance_ok,
        'volume_ok_2x': volume_ok_2x,
        'volume_ok_1_5x': volume_ok_1_5x,
        'volume_ok_1_25x': volume_ok_1_25x,
        'entry_signal_2x': entry_signal_2x,
        'entry_signal_1_5x': entry_signal_1_5x,
        'entry_signal_1_25x': entry_signal_1_25x,
        'entry_signal_none': entry_signal_none,
        'exit_signal': exit_signal
    }, index=data.index)

    print(f"Signal components calculated for {len(debug_df)} days")

    return debug_df


def generate_statistics(debug_df: pd.DataFrame):
    """
    Generate comprehensive statistics to identify bottlenecks.

    Args:
        debug_df: DataFrame with signal components
    """
    print(f"\n{'='*70}")
    print("SIGNAL GENERATION STATISTICS")
    print(f"{'='*70}")

    # Total days (excluding NaN due to lookback periods)
    valid_days = debug_df['distance'].notna().sum()
    print(f"\nTotal valid days (after warmup): {valid_days}")

    # First valid signal date (after 252-day warmup)
    first_valid = debug_df[debug_df['distance'].notna()].index[0]
    last_valid = debug_df.index[-1]
    print(f"Valid signal period: {first_valid.date()} to {last_valid.date()}")

    # Distance condition analysis
    print(f"\n{'DISTANCE CONDITION (Within 10% of 52w high)':^70}")
    print("-" * 70)
    distance_ok_days = debug_df['distance_ok'].sum()
    distance_ok_pct = (distance_ok_days / valid_days) * 100
    print(f"Days with distance >= 0.90: {distance_ok_days} ({distance_ok_pct:.1f}%)")

    # Distance distribution
    distance_stats = debug_df['distance'].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"\nDistance from 52w high percentiles:")
    print(f"  25th: {distance_stats['25%']:.3f}")
    print(f"  50th: {distance_stats['50%']:.3f}")
    print(f"  75th: {distance_stats['75%']:.3f}")
    print(f"  90th: {distance_stats['90%']:.3f}")
    print(f"  95th: {distance_stats['95%']:.3f}")
    print(f"  99th: {distance_stats['99%']:.3f}")

    # Volume condition analysis
    print(f"\n{'VOLUME CONDITION ANALYSIS':^70}")
    print("-" * 70)

    volume_ok_2x_days = debug_df['volume_ok_2x'].sum()
    volume_ok_2x_pct = (volume_ok_2x_days / valid_days) * 100
    print(f"Days with volume > 2.0x MA: {volume_ok_2x_days} ({volume_ok_2x_pct:.1f}%)")

    volume_ok_1_5x_days = debug_df['volume_ok_1_5x'].sum()
    volume_ok_1_5x_pct = (volume_ok_1_5x_days / valid_days) * 100
    print(f"Days with volume > 1.5x MA: {volume_ok_1_5x_days} ({volume_ok_1_5x_pct:.1f}%)")

    volume_ok_1_25x_days = debug_df['volume_ok_1_25x'].sum()
    volume_ok_1_25x_pct = (volume_ok_1_25x_days / valid_days) * 100
    print(f"Days with volume > 1.25x MA: {volume_ok_1_25x_days} ({volume_ok_1_25x_pct:.1f}%)")

    # Volume ratio distribution
    volume_ratio_stats = debug_df['volume_ratio'].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"\nVolume ratio (Volume / 20-day MA) percentiles:")
    print(f"  25th: {volume_ratio_stats['25%']:.2f}x")
    print(f"  50th: {volume_ratio_stats['50%']:.2f}x")
    print(f"  75th: {volume_ratio_stats['75%']:.2f}x")
    print(f"  90th: {volume_ratio_stats['90%']:.2f}x")
    print(f"  95th: {volume_ratio_stats['95%']:.2f}x")
    print(f"  99th: {volume_ratio_stats['99%']:.2f}x")
    print(f"  Max: {volume_ratio_stats['max']:.2f}x")

    # Entry signal counts (with different volume thresholds)
    print(f"\n{'ENTRY SIGNAL COUNTS (Distance + Volume Filter)':^70}")
    print("-" * 70)

    entry_2x = debug_df['entry_signal_2x'].sum()
    entry_1_5x = debug_df['entry_signal_1_5x'].sum()
    entry_1_25x = debug_df['entry_signal_1_25x'].sum()
    entry_none = debug_df['entry_signal_none'].sum()

    print(f"With 2.0x volume filter (CURRENT): {entry_2x} signals")
    print(f"With 1.5x volume filter:           {entry_1_5x} signals")
    print(f"With 1.25x volume filter:          {entry_1_25x} signals")
    print(f"Without volume filter:             {entry_none} signals")

    # Signal loss analysis
    if entry_none > 0:
        loss_2x = ((entry_none - entry_2x) / entry_none) * 100
        loss_1_5x = ((entry_none - entry_1_5x) / entry_none) * 100
        loss_1_25x = ((entry_none - entry_1_25x) / entry_none) * 100

        print(f"\nSignal loss due to volume filter:")
        print(f"  2.0x threshold: {loss_2x:.1f}% of potential signals blocked")
        print(f"  1.5x threshold: {loss_1_5x:.1f}% of potential signals blocked")
        print(f"  1.25x threshold: {loss_1_25x:.1f}% of potential signals blocked")

    # Overlap analysis (both conditions TRUE)
    print(f"\n{'OVERLAP ANALYSIS (Distance AND Volume)':^70}")
    print("-" * 70)

    both_ok_2x = (debug_df['distance_ok'] & debug_df['volume_ok_2x']).sum()
    both_ok_pct_2x = (both_ok_2x / valid_days) * 100
    print(f"Days with BOTH distance >= 0.90 AND volume > 2.0x: {both_ok_2x} ({both_ok_pct_2x:.1f}%)")

    # Exit signal count
    exit_signals = debug_df['exit_signal'].sum()
    exit_pct = (exit_signals / valid_days) * 100
    print(f"\nDays with exit signal (distance < 0.70): {exit_signals} ({exit_pct:.1f}%)")

    # Bottleneck identification
    print(f"\n{'BOTTLENECK IDENTIFICATION':^70}")
    print("=" * 70)

    if distance_ok_days < 100:
        print("BOTTLENECK: Distance filter")
        print(f"Only {distance_ok_days} days within 10% of 52w high - strategy may not suit SPY")
    elif volume_ok_2x_days < 100:
        print("BOTTLENECK: Volume filter (2.0x threshold)")
        print(f"Only {volume_ok_2x_days} days with 2.0x volume surge")
        print(f"Consider reducing threshold to 1.5x ({volume_ok_1_5x_days} days) or removing filter")
    elif entry_2x < 50:
        print("BOTTLENECK: Combined filters (conjunction too rare)")
        print(f"Distance OK: {distance_ok_days} days, Volume OK: {volume_ok_2x_days} days")
        print(f"Overlap: {both_ok_2x} days - filters rarely align")
    else:
        print("NO CLEAR BOTTLENECK - Further investigation needed")

    print("=" * 70)


def export_csv(debug_df: pd.DataFrame, filename: str = '52w_high_signals_debug.csv'):
    """
    Export debug DataFrame to CSV for manual inspection.

    Args:
        debug_df: DataFrame with signal components
        filename: Output CSV filename
    """
    print(f"\n{'='*70}")
    print("EXPORTING DEBUG CSV")
    print(f"{'='*70}")

    # Round numeric columns for readability
    export_df = debug_df.copy()
    export_df['Close'] = export_df['Close'].round(2)
    export_df['High'] = export_df['High'].round(2)
    export_df['Low'] = export_df['Low'].round(2)
    export_df['high_52w'] = export_df['high_52w'].round(2)
    export_df['distance'] = export_df['distance'].round(4)
    export_df['volume_ma_20'] = export_df['volume_ma_20'].round(0)
    export_df['volume_ratio'] = export_df['volume_ratio'].round(2)
    export_df['atr'] = export_df['atr'].round(2)

    # Export to CSV
    export_df.to_csv(filename)
    print(f"CSV exported to: {filename}")
    print(f"Total rows: {len(export_df)}")

    # Show sample of entry signal days
    entry_days_2x = export_df[export_df['entry_signal_2x']]
    if len(entry_days_2x) > 0:
        print(f"\nEntry signal days (2.0x volume filter): {len(entry_days_2x)}")
        print("\nSample entry signal days:")
        print(entry_days_2x[['Close', 'distance', 'volume_ratio', 'entry_signal_2x']].head(10))
    else:
        print("\nNo entry signals with 2.0x volume filter (as expected from backtest)")

    # Show days that would signal with lower threshold
    entry_days_1_5x = export_df[export_df['entry_signal_1_5x']]
    if len(entry_days_1_5x) > 0:
        print(f"\nEntry signal days (1.5x volume filter): {len(entry_days_1_5x)}")
        print("\nSample entry signals (1.5x threshold):")
        print(entry_days_1_5x[['Close', 'distance', 'volume_ratio', 'entry_signal_1_5x']].head(10))


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("52-WEEK HIGH MOMENTUM STRATEGY - SIGNAL GENERATION DEBUG")
    print("="*70)
    print("\nPurpose: Identify why strategy generates only 3 trades in 20 years")
    print("Hypothesis: 2.0x volume filter is too restrictive for SPY")

    # Load data
    try:
        data = load_spy_data(start_date='2005-01-01', end_date='2025-01-01')
    except Exception as e:
        print(f"\n[FAIL] Data loading failed: {e}")
        return

    # Calculate signal components
    try:
        debug_df = calculate_signal_components(data)
    except Exception as e:
        print(f"\n[FAIL] Signal calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate statistics
    try:
        generate_statistics(debug_df)
    except Exception as e:
        print(f"\n[FAIL] Statistics generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Export CSV
    try:
        export_csv(debug_df)
    except Exception as e:
        print(f"\n[FAIL] CSV export failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review statistics above to confirm bottleneck")
    print("2. Inspect 52w_high_signals_debug.csv for specific dates")
    print("3. If volume filter is bottleneck -> Test threshold variations")
    print("4. If distance filter is bottleneck -> Strategy may not suit SPY")


if __name__ == "__main__":
    main()
