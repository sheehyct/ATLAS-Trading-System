"""
Volume Threshold Testing Script

Tests 52W High Momentum strategy with different volume confirmation thresholds
to identify the optimal setting that balances trade frequency with performance.

Thresholds tested:
- None (no volume filter - pure 52w high momentum per original research)
- 1.25x (light filtering)
- 1.5x (moderate filtering)
- 1.75x (aggressive filtering)
- 2.0x (current implementation)

Architecture Targets:
- Sharpe Ratio: 0.8-1.2
- CAGR: 10-15%
- Win Rate: 50-60%
- Max Drawdown: -25% to -30%
- Minimum trades: 50+ over 20 years (2.5/year average)

Output:
- Comparison table of all threshold variants
- Recommendation based on target compliance
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict


def load_spy_data(start_date: str = '2005-01-01', end_date: str = '2025-01-01') -> pd.DataFrame:
    """Load SPY historical data."""
    print(f"\nLoading SPY data ({start_date} to {end_date})...")
    spy_data = vbt.YFData.pull('SPY', start=start_date, end=end_date).get()
    print(f"Loaded {len(spy_data)} trading days")
    return spy_data


def generate_signals_with_threshold(
    data: pd.DataFrame,
    volume_threshold: float = None,
    atr_multiplier: float = 2.5
) -> Dict[str, pd.Series]:
    """
    Generate signals with configurable volume threshold.

    Replicates high_momentum_52w.py logic but with parametrized volume filter.

    Args:
        data: OHLCV DataFrame
        volume_threshold: Volume multiplier (None = no filter, 1.5 = 1.5x MA, etc.)
        atr_multiplier: ATR stop multiplier (default: 2.5)

    Returns:
        Dictionary with entry_signal, exit_signal, stop_distance
    """
    # Calculate 52-week high (252 trading days)
    high_52w = data['High'].rolling(window=252, min_periods=252).max()

    # Calculate distance from 52-week high
    distance_from_high = data['Close'] / high_52w

    # Calculate volume moving average
    volume_ma_20 = data['Volume'].rolling(window=20, min_periods=20).mean()

    # Calculate ATR for stop loss
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=14, adjust=False, min_periods=14).mean()

    # Distance condition: Within 10% of 52w high
    distance_ok = (distance_from_high >= 0.90) & high_52w.notna()

    # Volume condition: Apply threshold if provided
    if volume_threshold is not None:
        volume_confirmed = (data['Volume'] > (volume_ma_20 * volume_threshold)) & volume_ma_20.notna()
        entry_signal = distance_ok & volume_confirmed & atr.notna()
    else:
        # No volume filter
        entry_signal = distance_ok & volume_ma_20.notna() & atr.notna()

    # Exit signal: 30% off highs
    exit_signal = (distance_from_high < 0.70) & high_52w.notna()

    # Stop distance
    stop_distance = (atr * atr_multiplier).fillna(0.0)

    return {
        'entry_signal': entry_signal.fillna(False),
        'exit_signal': exit_signal.fillna(False),
        'stop_distance': stop_distance
    }


def calculate_position_sizes(
    data: pd.DataFrame,
    capital: float,
    stop_distance: pd.Series,
    risk_pct: float = 0.02
) -> pd.Series:
    """
    Calculate position sizes using ATR-based risk management.

    Args:
        data: OHLCV DataFrame
        capital: Initial capital
        stop_distance: Stop loss distances
        risk_pct: Risk per trade (default: 2%)

    Returns:
        Position sizes as integer share counts
    """
    # Risk-based position sizing
    close = data['Close']

    # Avoid division by zero
    position_sizes = pd.Series(0, index=data.index)

    # Calculate where stop_distance > 0
    valid_stops = stop_distance > 0

    # Risk amount per trade
    risk_amount = capital * risk_pct

    # Position size = risk amount / stop distance
    position_sizes[valid_stops] = risk_amount / stop_distance[valid_stops]

    # Convert to share counts (capital constraint)
    max_shares = capital / close
    position_sizes = np.minimum(position_sizes, max_shares)

    # Integer shares only
    position_sizes = position_sizes.astype(int)

    return position_sizes


def run_backtest_with_threshold(
    data: pd.DataFrame,
    volume_threshold: float = None,
    initial_capital: float = 10000,
    threshold_label: str = "Unknown"
) -> Dict:
    """
    Run backtest with specified volume threshold.

    Args:
        data: OHLCV DataFrame
        volume_threshold: Volume multiplier (None = no filter)
        initial_capital: Starting capital
        threshold_label: Label for display

    Returns:
        Dictionary with backtest results and metrics
    """
    print(f"\nBacktesting with volume threshold: {threshold_label}...")

    # Generate signals
    signals = generate_signals_with_threshold(data, volume_threshold)

    # Calculate position sizes
    position_sizes = calculate_position_sizes(
        data,
        capital=initial_capital,
        stop_distance=signals['stop_distance'],
        risk_pct=0.02
    )

    # Entry/exit positions (only enter when signal, exit when exit signal or stop hit)
    entries = signals['entry_signal']
    exits = signals['exit_signal']

    # Size array: position sizes on entry dates
    size = pd.Series(0, index=data.index)
    size[entries] = position_sizes[entries]

    # Run VBT backtest
    pf = vbt.Portfolio.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        size=size,
        size_type='amount',
        init_cash=initial_capital,
        fees=0.0015,
        slippage=0.0015,
        freq='1D'
    )

    # Extract metrics
    total_return = pf.total_return
    cagr = pf.annualized_return
    sharpe = pf.sharpe_ratio
    sortino = pf.sortino_ratio
    max_dd = pf.max_drawdown

    # Trade statistics
    trades = pf.trades.records_readable
    total_trades = len(trades)

    if total_trades > 0:
        win_rate = (trades['PnL'] > 0).sum() / total_trades
        avg_trade = (trades['Return'].mean()) if 'Return' in trades.columns else 0
    else:
        win_rate = 0
        avg_trade = 0

    print(f"  Total Trades: {total_trades}")
    print(f"  Sharpe: {sharpe:.2f}, CAGR: {cagr:.2%}, MaxDD: {max_dd:.2%}")

    return {
        'threshold_label': threshold_label,
        'threshold_value': volume_threshold if volume_threshold else 0,
        'total_trades': total_trades,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'portfolio': pf
    }


def compare_results(results: list):
    """
    Compare backtest results across all threshold variants.

    Args:
        results: List of result dictionaries
    """
    print("\n" + "="*100)
    print("VOLUME THRESHOLD COMPARISON")
    print("="*100)

    # Create comparison DataFrame
    comparison = pd.DataFrame(results)

    # Format for display
    print(f"\n{'Threshold':<15} {'Trades':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Win Rate':>10} {'Status':>15}")
    print("-"*100)

    for _, row in comparison.iterrows():
        threshold = row['threshold_label']
        trades = int(row['total_trades'])
        sharpe = row['sharpe']
        cagr = row['cagr']
        max_dd = row['max_drawdown']
        win_rate = row['win_rate']

        # Determine status vs targets
        sharpe_ok = sharpe >= 0.8
        cagr_ok = cagr >= 0.10
        max_dd_ok = abs(max_dd) <= 0.30
        trades_ok = trades >= 50
        win_rate_ok = 0.50 <= win_rate <= 0.70  # Extended upper bound slightly

        all_ok = sharpe_ok and cagr_ok and max_dd_ok and trades_ok

        if all_ok:
            status = "MEETS TARGETS"
        elif sharpe_ok and cagr_ok and trades_ok:
            status = "ACCEPTABLE"
        elif trades < 10:
            status = "TOO FEW TRADES"
        else:
            status = "BELOW TARGETS"

        print(f"{threshold:<15} {trades:>8} {sharpe:>8.2f} {cagr:>7.1%} {max_dd:>7.1%} {win_rate:>9.1%} {status:>15}")

    # Architecture targets reminder
    print("\n" + "-"*100)
    print("ARCHITECTURE TARGETS:")
    print("  Sharpe Ratio: >= 0.8")
    print("  CAGR: 10-15%")
    print("  Max Drawdown: <= -30%")
    print("  Win Rate: 50-60%")
    print("  Minimum Trades: >= 50 (over 20 years)")
    print("-"*100)

    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    # Find best threshold
    valid_results = comparison[comparison['total_trades'] >= 50]

    if len(valid_results) == 0:
        print("\nNO THRESHOLD meets minimum trade count (50+) - Consider removing volume filter")
        print("\nSuggested approach:")
        print("1. Remove volume filter entirely (align with George & Hwang 2004 original research)")
        print("2. Volume confirmation is NOT mentioned in academic 52w high momentum papers")
        print("3. The 2.0x threshold was borrowed from ORB research (different strategy)")
    else:
        # Rank by Sharpe ratio
        best = valid_results.loc[valid_results['sharpe'].idxmax()]

        print(f"\nRECOMMENDED: {best['threshold_label']}")
        print(f"  Trades: {int(best['total_trades'])}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  CAGR: {best['cagr']:.2%}")
        print(f"  Max Drawdown: {best['max_drawdown']:.2%}")
        print(f"  Win Rate: {best['win_rate']:.1%}")

        # Compare to current implementation (2.0x)
        current = comparison[comparison['threshold_label'] == '2.0x (Current)'].iloc[0]
        print(f"\nIMPROVEMENT vs Current (2.0x):")
        print(f"  Trades: +{int(best['total_trades'] - current['total_trades'])}")
        print(f"  Sharpe: {best['sharpe'] - current['sharpe']:+.2f}")
        print(f"  CAGR: {best['cagr'] - current['cagr']:+.2%}")

    print("="*100)


def main():
    """Main execution function."""
    print("\n" + "="*100)
    print("52-WEEK HIGH MOMENTUM - VOLUME THRESHOLD OPTIMIZATION")
    print("="*100)
    print("\nTesting 5 variants to identify optimal volume confirmation threshold")
    print("Expected: Lower thresholds increase trade count and improve risk-adjusted returns")

    # Load data
    try:
        data = load_spy_data()
    except Exception as e:
        print(f"\n[FAIL] Data loading failed: {e}")
        return

    # Test thresholds
    thresholds = [
        (None, "No Filter"),
        (1.25, "1.25x"),
        (1.5, "1.5x"),
        (1.75, "1.75x"),
        (2.0, "2.0x (Current)")
    ]

    results = []

    for threshold_value, threshold_label in thresholds:
        try:
            result = run_backtest_with_threshold(
                data,
                volume_threshold=threshold_value,
                threshold_label=threshold_label
            )
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Backtest failed for {threshold_label}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare results
    if len(results) > 0:
        compare_results(results)
    else:
        print("\n[FAIL] No successful backtests to compare")

    print("\n" + "="*100)
    print("THRESHOLD TESTING COMPLETE")
    print("="*100)
    print("\nNext Steps:")
    print("1. Review recommendation above")
    print("2. Implement optimal threshold in high_momentum_52w.py")
    print("3. Update tests to reflect new threshold")
    print("4. Re-run full validation backtest")


if __name__ == "__main__":
    main()
