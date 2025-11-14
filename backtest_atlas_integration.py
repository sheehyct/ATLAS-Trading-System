r"""
SPY Backtest Comparison: STRAT-only vs STRAT+ATLAS vs Buy-and-Hold

Compares three trading approaches on SPY 2020-2024:
    1. STRAT-only: Trade all pattern signals (no regime filtering)
    2. STRAT+ATLAS: Trade only HIGH/MEDIUM quality signals (regime filtered)
    3. Buy-and-hold: SPY benchmark

Metrics compared:
    - Total return
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Number of trades

Purpose:
    Validate that ATLAS regime filtering improves STRAT signal quality.

Expected results:
    - STRAT+ATLAS should have higher Sharpe (better risk-adjusted returns)
    - STRAT+ATLAS should have lower max drawdown (regime filter reduces risk)
    - STRAT+ATLAS should have fewer trades (only high-quality signals)
    - Both STRAT strategies should beat buy-and-hold (validates Layer 2)

Reference:
    docs/SYSTEM_ARCHITECTURE/INTEGRATION_ARCHITECTURE.md
    Session 32 pattern detection validation (100% accuracy verified)
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime

# Import STRAT components
from strat.bar_classifier import classify_bars
from strat.pattern_detector import detect_patterns
from strat.atlas_integration import filter_strat_signals, get_position_size_multiplier, combine_pattern_signals

# Import ATLAS components
from regime.academic_jump_model import AcademicJumpModel


def load_spy_data(start_date='2020-01-01', end_date='2024-12-31'):
    """
    Load SPY daily data using VBT.

    Returns:
    --------
    pd.DataFrame with OHLCV columns
    """
    print(f"Loading SPY data from {start_date} to {end_date}...")

    # Use yfinance via VBT
    spy_data = vbt.YFData.pull('SPY', start=start_date, end=end_date, timeframe='1d')
    data = spy_data.get()  # Get DataFrame

    print(f"  Loaded {len(data)} bars")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    return data


def run_strat_detection(data):
    """
    Run STRAT bar classification and pattern detection.

    Returns:
    --------
    dict with:
        - classifications: Bar types (1, 2, -2, 3)
        - pattern_entries: Combined pattern entry signals
        - pattern_directions: Combined pattern directions
        - entries_312, stops_312, targets_312, directions_312
        - entries_212, stops_212, targets_212, directions_212
    """
    print("\nRunning STRAT bar classification...")

    # Extract price data
    high = data['High'].values
    low = data['Low'].values

    # Classify bars
    classifications = classify_bars(high, low)

    print(f"  Bar classification complete: {len(classifications)} bars")
    print(f"  Distribution: Inside={np.sum(classifications==1)}, "
          f"2U={np.sum(classifications==2)}, "
          f"2D={np.sum(classifications==-2)}, "
          f"Outside={np.sum(classifications==3)}")

    print("\nRunning STRAT pattern detection...")

    # Detect patterns
    result = detect_patterns(classifications, high, low)

    # Unpack results
    entries_312 = result.entries_312.values
    stops_312 = result.stops_312.values
    targets_312 = result.targets_312.values
    directions_312 = result.directions_312.values

    entries_212 = result.entries_212.values
    stops_212 = result.stops_212.values
    targets_212 = result.targets_212.values
    directions_212 = result.directions_212.values

    # Combine patterns (3-1-2 has priority)
    pattern_entries = np.logical_or(entries_312, entries_212)
    pattern_directions = combine_pattern_signals(
        entries_312, directions_312,
        entries_212, directions_212
    )

    # Count patterns
    num_312 = np.sum(entries_312)
    num_212 = np.sum(entries_212)
    num_total = np.sum(pattern_entries)

    print(f"  Patterns detected: {num_total} total ({num_312} 3-1-2, {num_212} 2-1-2)")
    print(f"  Bullish patterns: {np.sum(pattern_directions==1)}")
    print(f"  Bearish patterns: {np.sum(pattern_directions==-1)}")

    return {
        'classifications': classifications,
        'pattern_entries': pattern_entries,
        'pattern_directions': pattern_directions,
        'entries_312': entries_312,
        'stops_312': stops_312,
        'targets_312': targets_312,
        'directions_312': directions_312,
        'entries_212': entries_212,
        'stops_212': stops_212,
        'targets_212': targets_212,
        'directions_212': directions_212,
    }


def run_atlas_detection(data):
    """
    Run ATLAS regime detection.

    Returns:
    --------
    pd.Series : ATLAS regimes ('TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH')
    """
    print("\nRunning ATLAS regime detection...")

    # Initialize ATLAS model
    atlas = AcademicJumpModel()

    # Prepare data (ATLAS needs OHLCV with 'Close' column)
    atlas_data = data.copy()

    # Ensure column names match ATLAS expectations
    if 'close' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'close': 'Close'})
    if 'high' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'high': 'High'})
    if 'low' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'low': 'Low'})
    if 'volume' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'volume': 'Volume'})

    # Run online inference (lookback=1000 days for 2020-2024 period)
    # Note: Architecture spec recommends 3000 days, but using 1000 for demonstration
    lookback = 1000
    try:
        atlas_regimes, lambda_series, theta_df = atlas.online_inference(
            atlas_data,
            lookback=lookback,  # Reduced from 3000 to fit 2020-2024 data
            default_lambda=10.0  # Session 24 calibrated value
        )
    except Exception as e:
        print(f"  Warning: ATLAS inference failed: {e}")
        print("  Using fallback: all TREND_NEUTRAL")
        atlas_regimes = pd.Series('TREND_NEUTRAL', index=data.index)
        lookback = 0

    # Align regimes with STRAT data
    print(f"  ATLAS returned {len(atlas_regimes)} days (after {lookback}-day lookback)")
    print(f"  STRAT data has {len(data)} days")

    # Reindex to match STRAT data and forward-fill NaNs
    atlas_regimes_aligned = atlas_regimes.reindex(data.index)

    # Fill leading NaNs with TREND_NEUTRAL (safe default for lookback period)
    atlas_regimes_aligned = atlas_regimes_aligned.fillna('TREND_NEUTRAL')

    # Count regime distribution
    regime_counts = atlas_regimes_aligned.value_counts()
    print(f"  Regime distribution (after filling lookback period with TREND_NEUTRAL):")
    for regime, count in regime_counts.items():
        pct = count / len(atlas_regimes_aligned) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    return atlas_regimes_aligned


def backtest_strat_only(data, strat_results):
    """
    Backtest STRAT-only: Trade all pattern signals (no regime filtering).

    Returns:
    --------
    vbt.Portfolio
    """
    print("\n[Scenario 1] STRAT-only backtest...")

    # Entry signals: All patterns
    entries = pd.Series(strat_results['pattern_entries'], index=data.index)

    # Exit signals: Simple approach - exit after N days or on opposite pattern
    # For simplicity, hold for 5 days (typical STRAT magnitude target timeframe)
    exits = entries.shift(5).fillna(False)

    # Run backtest
    pf = vbt.PF.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001  # 0.1% commission
    )

    print(f"  Total return: {pf.total_return:.2%}")
    print(f"  Sharpe ratio: {pf.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {pf.max_drawdown:.2%}")
    print(f"  Total trades: {pf.stats()['Total Trades']}")

    return pf


def backtest_strat_atlas(data, strat_results, atlas_regimes):
    """
    Backtest STRAT+ATLAS: Trade only HIGH/MEDIUM quality signals.

    Returns:
    --------
    vbt.Portfolio
    """
    print("\n[Scenario 2] STRAT+ATLAS backtest...")

    # Filter signals by regime
    pattern_directions = pd.Series(strat_results['pattern_directions'], index=data.index)
    signal_qualities = filter_strat_signals(atlas_regimes, pattern_directions)
    position_multipliers = get_position_size_multiplier(signal_qualities)

    # Entry signals: Only HIGH and MEDIUM quality (multiplier > 0)
    entries = pd.Series(strat_results['pattern_entries'], index=data.index) & (position_multipliers > 0)

    # Exit signals: Same as STRAT-only (5 days)
    exits = entries.shift(5).fillna(False)

    # Position sizing based on signal quality
    # HIGH = 100% allocation, MEDIUM = 50% allocation
    size = position_multipliers[entries]

    # Run backtest
    pf = vbt.PF.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        size=size if len(size) > 0 else 1.0,  # Use multipliers if trades exist
        init_cash=10000,
        fees=0.001  # 0.1% commission
    )

    print(f"  Total return: {pf.total_return:.2%}")
    print(f"  Sharpe ratio: {pf.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {pf.max_drawdown:.2%}")
    print(f"  Total trades: {pf.stats()['Total Trades']}")

    # Count signal quality distribution
    quality_counts = signal_qualities[entries].value_counts()
    print(f"  Signal quality distribution:")
    for quality, count in quality_counts.items():
        print(f"    {quality}: {count} trades")

    return pf


def backtest_buy_and_hold(data):
    """
    Backtest buy-and-hold SPY benchmark.

    Returns:
    --------
    vbt.Portfolio
    """
    print("\n[Scenario 3] Buy-and-hold benchmark...")

    # Entry: Buy on first day
    entries = pd.Series(False, index=data.index)
    entries.iloc[0] = True

    # Exit: Never (hold forever)
    exits = pd.Series(False, index=data.index)

    # Run backtest
    pf = vbt.PF.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001  # 0.1% commission
    )

    print(f"  Total return: {pf.total_return:.2%}")
    print(f"  Sharpe ratio: {pf.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {pf.max_drawdown:.2%}")

    return pf


def compare_results(pf_strat_only, pf_strat_atlas, pf_buy_hold):
    """
    Compare backtest results across all three scenarios.

    Prints summary table and analysis.
    """
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison table
    comparison = pd.DataFrame({
        'STRAT-only': [
            pf_strat_only.total_return,
            pf_strat_only.sharpe_ratio,
            pf_strat_only.max_drawdown,
            pf_strat_only.stats()['Total Trades'],
        ],
        'STRAT+ATLAS': [
            pf_strat_atlas.total_return,
            pf_strat_atlas.sharpe_ratio,
            pf_strat_atlas.max_drawdown,
            pf_strat_atlas.stats()['Total Trades'],
        ],
        'Buy-and-Hold': [
            pf_buy_hold.total_return,
            pf_buy_hold.sharpe_ratio,
            pf_buy_hold.max_drawdown,
            1,  # Single trade (buy and hold)
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades'])

    print(comparison.to_string())

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check if STRAT+ATLAS improves over STRAT-only
    if pf_strat_atlas.sharpe_ratio > pf_strat_only.sharpe_ratio:
        improvement = ((pf_strat_atlas.sharpe_ratio / pf_strat_only.sharpe_ratio) - 1) * 100
        print(f"[PASS] STRAT+ATLAS Sharpe improved by {improvement:.1f}% (regime filtering works)")
    else:
        print(f"[WARN] STRAT+ATLAS Sharpe did not improve (check signal quality matrix)")

    # Check if STRAT+ATLAS reduces drawdown (drawdowns are negative, so abs for comparison)
    if abs(pf_strat_atlas.max_drawdown) < abs(pf_strat_only.max_drawdown):
        reduction = ((abs(pf_strat_only.max_drawdown) - abs(pf_strat_atlas.max_drawdown))
                     / abs(pf_strat_only.max_drawdown)) * 100
        print(f"[PASS] STRAT+ATLAS reduced max drawdown by {reduction:.1f}% (risk management works)")
    else:
        print(f"[WARN] STRAT+ATLAS did not reduce drawdown")

    # Check if STRAT beats buy-and-hold
    if pf_strat_atlas.total_return > pf_buy_hold.total_return:
        print(f"[PASS] STRAT+ATLAS beat buy-and-hold (validates Layer 2)")
    else:
        print(f"[INFO] STRAT+ATLAS did not beat buy-and-hold (expected in strong bull market)")

    print("=" * 80)


def main():
    """Run complete backtest comparison."""
    print("=" * 80)
    print("SPY BACKTEST COMPARISON: STRAT-only vs STRAT+ATLAS vs Buy-and-Hold")
    print("=" * 80)

    # Load data
    data = load_spy_data('2020-01-01', '2024-12-31')

    # Run STRAT detection
    strat_results = run_strat_detection(data)

    # Run ATLAS detection
    atlas_regimes = run_atlas_detection(data)

    # Run backtests
    pf_strat_only = backtest_strat_only(data, strat_results)
    pf_strat_atlas = backtest_strat_atlas(data, strat_results, atlas_regimes)
    pf_buy_hold = backtest_buy_and_hold(data)

    # Compare results
    compare_results(pf_strat_only, pf_strat_atlas, pf_buy_hold)

    print("\nBacktest complete!")


if __name__ == '__main__':
    main()
