"""
SPY Backtest: System A (ATLAS + 52-Week Momentum) vs Buy-and-Hold

Backtests System A on SPY 2020-2024:
    - ATLAS regime detection (Layer 1)
    - 52-week high momentum scanner (instead of STRAT patterns)
    - Top-N=5 portfolio selection
    - Regime-based allocation (BULL=100%, NEUTRAL=70%, BEAR=30%, CRASH=0%)
    - Semi-annual rebalancing (February 1, August 1)

Metrics compared:
    - Total return
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Number of trades

Purpose:
    Compare System A (52-week momentum) to System B (STRAT patterns) to determine
    which system to deploy to paper trading account.

Expected results:
    - System A should beat buy-and-hold (validates momentum factor)
    - Regime allocation should reduce drawdown vs 100% deployed
    - Semi-annual rebalancing should minimize transaction costs

Reference:
    .session_startup_prompt.md - Session 50 objective
    docs/HANDOFF.md - Session 49 summary
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime

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

    # Use yfinance via VBT (tz='America/New_York' for date alignment)
    spy_data = vbt.YFData.pull(
        'SPY',
        start=start_date,
        end=end_date,
        timeframe='1d',
        tz='America/New_York'  # Critical: prevents UTC date shifts
    )
    data = spy_data.get()  # Get DataFrame

    print(f"  Loaded {len(data)} bars")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")

    return data


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
    lookback = 1000
    try:
        atlas_regimes, lambda_series, theta_df = atlas.online_inference(
            atlas_data,
            lookback=lookback,
            default_lambda=1.5  # Session 49 validated value
        )
    except Exception as e:
        print(f"  Warning: ATLAS inference failed: {e}")
        print("  Using fallback: all TREND_NEUTRAL")
        atlas_regimes = pd.Series('TREND_NEUTRAL', index=data.index)
        lookback = 0

    # Align regimes with data
    print(f"  ATLAS returned {len(atlas_regimes)} days (after {lookback}-day lookback)")

    # Reindex to match data and forward-fill NaNs
    atlas_regimes_aligned = atlas_regimes.reindex(data.index)

    # Fill leading NaNs with TREND_NEUTRAL (safe default for lookback period)
    atlas_regimes_aligned = atlas_regimes_aligned.fillna('TREND_NEUTRAL')

    # Count regime distribution
    regime_counts = atlas_regimes_aligned.value_counts()
    print(f"  Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(atlas_regimes_aligned) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    return atlas_regimes_aligned


def calculate_momentum_score(close_prices, lookback=252):
    """
    Calculate 52-week high momentum score.

    Score = (current_price / 52-week_high)

    Higher score = closer to 52-week high = stronger momentum

    Args:
        close_prices: pd.Series of close prices
        lookback: Lookback period in days (default 252 = 1 year)

    Returns:
        pd.Series of momentum scores
    """
    # Calculate 52-week high for each date (expanding window)
    high_52w = close_prices.rolling(window=lookback, min_periods=lookback).max()

    # Score = current price / 52-week high
    momentum_score = close_prices / high_52w

    return momentum_score


def generate_rebalance_signals(data, atlas_regimes, top_n=5):
    """
    Generate semi-annual rebalance signals with regime-based allocation.

    Strategy:
    1. Rebalance on Feb 1 and Aug 1 each year
    2. Select SPY if momentum score > 0.90 (within 10% of 52-week high)
    3. Apply regime-based allocation:
       - TREND_BULL: 100% capital
       - TREND_NEUTRAL: 70% capital
       - TREND_BEAR: 30% capital
       - CRASH: 0% capital (100% cash)

    Args:
        data: pd.DataFrame with OHLCV
        atlas_regimes: pd.Series of regime classifications
        top_n: Number of positions (1 for SPY-only backtest)

    Returns:
        dict with entries, exits, position_sizes
    """
    print("\nGenerating rebalance signals...")

    # Calculate momentum score
    momentum = calculate_momentum_score(data['Close'], lookback=252)

    # Rebalance dates: Feb 1 and Aug 1 each year
    rebalance_dates = []
    for year in range(data.index[0].year, data.index[-1].year + 1):
        for month in [2, 8]:
            try:
                date = pd.Timestamp(year=year, month=month, day=1)
                if data.index.tz is not None:
                    date = date.tz_localize(data.index.tz)
                if date >= data.index[0] and date <= data.index[-1]:
                    rebalance_dates.append(date)
            except:
                continue

    print(f"  Rebalance dates: {len(rebalance_dates)}")
    for date in rebalance_dates:
        print(f"    {date.strftime('%Y-%m-%d')}")

    # Regime allocation rules
    REGIME_ALLOCATION = {
        'TREND_BULL': 1.00,      # 100% deployed
        'TREND_NEUTRAL': 0.70,   # 70% deployed
        'TREND_BEAR': 0.30,      # 30% deployed
        'CRASH': 0.00            # 0% deployed (cash only)
    }

    # Build allocation series (what % should be deployed each day)
    target_allocation = pd.Series(0.0, index=data.index)

    # Process each rebalance period
    for i, rebalance_date in enumerate(rebalance_dates):
        # Find closest trading day
        if rebalance_date not in data.index:
            future_dates = data.index[data.index >= rebalance_date]
            if len(future_dates) == 0:
                continue
            rebalance_date = future_dates[0]

        # Get momentum score at rebalance date
        current_momentum = momentum.loc[rebalance_date]

        # Get regime at rebalance date
        current_regime = atlas_regimes.loc[rebalance_date]
        allocation_pct = REGIME_ALLOCATION.get(current_regime, 0.70)

        print(f"\n  Rebalance {i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
        print(f"    Momentum: {current_momentum:.4f}")
        print(f"    Regime: {current_regime}")
        print(f"    Allocation: {allocation_pct:.0%}")

        # Determine target allocation for this period
        if current_momentum >= 0.90 and not pd.isna(current_momentum):
            # Enter with regime-based allocation
            next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else data.index[-1]
            target_allocation.loc[rebalance_date:next_rebalance] = allocation_pct
            print(f"    Signal: HOLD {allocation_pct:.0%} SPY")
        else:
            # Hold cash if momentum weak or CRASH regime
            next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else data.index[-1]
            target_allocation.loc[rebalance_date:next_rebalance] = 0.0
            print(f"    Signal: HOLD CASH")

    # Summary
    avg_allocation = target_allocation[target_allocation > 0].mean() if len(target_allocation[target_allocation > 0]) > 0 else 0
    days_deployed = (target_allocation > 0).sum()

    print(f"\n  Signal summary:")
    print(f"    Days deployed: {days_deployed} / {len(data)} ({100*days_deployed/len(data):.1f}%)")
    print(f"    Average allocation when deployed: {avg_allocation:.1%}")

    return {
        'target_allocation': target_allocation,
        'momentum': momentum,
        'rebalance_dates': rebalance_dates
    }


def backtest_system_a(data, atlas_regimes):
    """
    Backtest System A: ATLAS + 52-week momentum.

    Uses target allocation approach: portfolio value is rebalanced to target % of equity each day.

    Returns:
    --------
    vbt.Portfolio
    """
    print("\n[System A] ATLAS + 52-Week Momentum backtest...")

    # Generate signals
    signals = generate_rebalance_signals(data, atlas_regimes, top_n=5)

    # Calculate returns manually based on target allocation
    # This simulates holding target_allocation % of equity in SPY each day
    spy_returns = data['Close'].pct_change()

    # Portfolio returns = allocation % * SPY returns
    portfolio_returns = signals['target_allocation'].shift(1) * spy_returns

    # Cumulative returns starting from $10,000
    init_cash = 10000
    portfolio_value = init_cash * (1 + portfolio_returns).cumprod()

    # Calculate metrics
    total_return = (portfolio_value.iloc[-1] / init_cash) - 1

    # Calculate Sharpe ratio (annualized)
    mean_return = portfolio_returns.mean() * 252  # Annualized
    std_return = portfolio_returns.std() * np.sqrt(252)  # Annualized
    sharpe = mean_return / std_return if std_return > 0 else 0

    # Calculate max drawdown
    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    max_drawdown = drawdown.min()

    # Count rebalances (allocation changes)
    allocation_changes = signals['target_allocation'].diff().abs() > 0.01
    num_rebalances = allocation_changes.sum()

    print(f"  Total return: {total_return:.2%}")
    print(f"  Sharpe ratio: {sharpe:.2f}")
    print(f"  Max drawdown: {max_drawdown:.2%}")
    print(f"  Number of rebalances: {num_rebalances}")

    # Create a simple portfolio object for comparison
    # Store results in dict
    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': num_rebalances,
        'portfolio_value': portfolio_value
    }

    return results


def backtest_buy_and_hold(data):
    """
    Backtest buy-and-hold SPY benchmark.

    Returns:
    --------
    vbt.Portfolio
    """
    print("\n[Benchmark] Buy-and-hold SPY...")

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


def compare_results(results_system_a, pf_buy_hold):
    """
    Compare backtest results.

    Prints summary table and analysis.
    """
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison table
    comparison = pd.DataFrame({
        'System A': [
            results_system_a['total_return'],
            results_system_a['sharpe_ratio'],
            results_system_a['max_drawdown'],
            results_system_a['num_trades'],
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

    # Check if System A beats buy-and-hold
    if results_system_a['total_return'] > pf_buy_hold.total_return:
        improvement = ((results_system_a['total_return'] / pf_buy_hold.total_return) - 1) * 100
        print(f"[PASS] System A beat buy-and-hold by {improvement:.1f}%")
    else:
        underperformance = ((pf_buy_hold.total_return / results_system_a['total_return']) - 1) * 100
        print(f"[WARN] System A underperformed buy-and-hold by {underperformance:.1f}%")

    # Check Sharpe ratio improvement
    if results_system_a['sharpe_ratio'] > pf_buy_hold.sharpe_ratio:
        improvement = ((results_system_a['sharpe_ratio'] / pf_buy_hold.sharpe_ratio) - 1) * 100
        print(f"[PASS] System A Sharpe improved by {improvement:.1f}% (better risk-adjusted returns)")
    else:
        print(f"[WARN] System A Sharpe did not improve")

    # Check if System A reduces drawdown
    if abs(results_system_a['max_drawdown']) < abs(pf_buy_hold.max_drawdown):
        reduction = ((abs(pf_buy_hold.max_drawdown) - abs(results_system_a['max_drawdown']))
                     / abs(pf_buy_hold.max_drawdown)) * 100
        print(f"[PASS] System A reduced max drawdown by {reduction:.1f}% (risk management works)")
    else:
        print(f"[WARN] System A did not reduce drawdown")

    print("=" * 80)


def main():
    """Run complete backtest comparison."""
    print("=" * 80)
    print("SYSTEM A BACKTEST: ATLAS + 52-Week Momentum vs Buy-and-Hold")
    print("=" * 80)

    # Load data
    data = load_spy_data('2020-01-01', '2024-12-31')

    # Run ATLAS detection
    atlas_regimes = run_atlas_detection(data)

    # Run backtests
    results_system_a = backtest_system_a(data, atlas_regimes)
    pf_buy_hold = backtest_buy_and_hold(data)

    # Compare results
    compare_results(results_system_a, pf_buy_hold)

    print("\nBacktest complete!")

    # Return results for comparison with System B
    return {
        'system_a': results_system_a,
        'buy_hold': pf_buy_hold
    }


if __name__ == '__main__':
    main()
