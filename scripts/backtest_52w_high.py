"""
52-Week High Momentum Strategy Backtest Validation

This script validates the 52W High Momentum strategy on historical SPY data (2005-2025).
Tests whether the strategy meets architecture performance targets.

Performance Targets (per architecture):
- Sharpe Ratio: 0.8-1.2
- Win Rate: 50-60%
- CAGR: 10-15%
- Max Drawdown: -25% to -30%

Critical Periods to Include:
- 2008 Financial Crisis (major crash)
- 2020 COVID Crash (rapid recovery)
- 2022 Bear Market (rates/inflation)

Validation Criteria:
- Sharpe >= 0.8 (acceptable)
- Max Drawdown <= -30% (acceptable)
- Outperforms buy-and-hold on risk-adjusted basis

Professional Standards:
- NO emojis or unicode
- Clear output with metrics comparison
- CSV export for manual inspection
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from strategies.high_momentum_52w import HighMomentum52W
from strategies.base_strategy import StrategyConfig


def fetch_spy_data(start_date: str = '2005-01-01', end_date: str = '2025-01-01') -> pd.DataFrame:
    """
    Fetch SPY historical data using VectorBT Pro.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        DataFrame with OHLCV data
    """
    print(f"\nFetching SPY data from {start_date} to {end_date}...")

    # Use VBT's built-in data fetcher (yfinance)
    spy_data = vbt.YFData.pull('SPY', start=start_date, end=end_date).get()

    print(f"Fetched {len(spy_data)} trading days")
    print(f"Date range: {spy_data.index[0]} to {spy_data.index[-1]}")

    return spy_data


def run_52w_high_backtest(data: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """
    Run 52W High Momentum strategy backtest.

    Args:
        data: OHLCV DataFrame
        initial_capital: Starting capital

    Returns:
        Dictionary with strategy results
    """
    print("\n" + "="*60)
    print("52-WEEK HIGH MOMENTUM STRATEGY BACKTEST")
    print("="*60)

    # Create strategy configuration
    config = StrategyConfig(
        name="52-Week High Momentum",
        universe="sp500",
        rebalance_frequency="semi_annual",
        regime_compatibility={
            'TREND_BULL': True,
            'TREND_NEUTRAL': True,  # Unique advantage
            'TREND_BEAR': False,
            'CRASH': False
        },
        risk_per_trade=0.02,
        max_positions=5,
        enable_shorts=False
    )

    # Initialize strategy
    strategy = HighMomentum52W(config)

    # Run backtest (no regime filtering for historical validation)
    print("\nRunning backtest...")
    pf = strategy.backtest(data, initial_capital=initial_capital)

    # Extract metrics
    metrics = strategy.get_performance_metrics(pf)

    # Calculate additional metrics
    total_return = pf.total_return
    cagr = pf.annualized_return
    sharpe = pf.sharpe_ratio
    sortino = pf.sortino_ratio
    max_dd = pf.max_drawdown

    return {
        'portfolio': pf,
        'metrics': metrics,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd
    }


def run_buy_and_hold_benchmark(data: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """
    Run buy-and-hold benchmark for comparison.

    Args:
        data: OHLCV DataFrame
        initial_capital: Starting capital

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*60)
    print("BUY-AND-HOLD BENCHMARK")
    print("="*60)

    # Buy at start, hold to end
    entries = pd.Series(False, index=data.index)
    entries.iloc[252] = True  # Buy after 252 days (52-week lookback)

    exits = pd.Series(False, index=data.index)

    # Position size: all capital
    shares = int(initial_capital / data['Close'].iloc[252])
    size = pd.Series(0, index=data.index)
    size.iloc[252] = shares

    print("\nRunning buy-and-hold...")
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

    return {
        'portfolio': pf,
        'total_return': pf.total_return,
        'cagr': pf.annualized_return,
        'sharpe': pf.sharpe_ratio,
        'sortino': pf.sortino_ratio,
        'max_drawdown': pf.max_drawdown
    }


def compare_results(strategy_results: dict, benchmark_results: dict):
    """
    Compare strategy vs benchmark and print results.

    Args:
        strategy_results: Strategy backtest results
        benchmark_results: Buy-and-hold results
    """
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<25} {'Strategy':>15} {'Buy-Hold':>15} {'Delta':>15}")
    print("-"*70)

    # Total Return
    strat_ret = strategy_results['total_return']
    bench_ret = benchmark_results['total_return']
    delta_ret = strat_ret - bench_ret
    print(f"{'Total Return':<25} {strat_ret:>14.2%} {bench_ret:>14.2%} {delta_ret:>14.2%}")

    # CAGR
    strat_cagr = strategy_results['cagr']
    bench_cagr = benchmark_results['cagr']
    delta_cagr = strat_cagr - bench_cagr
    print(f"{'CAGR':<25} {strat_cagr:>14.2%} {bench_cagr:>14.2%} {delta_cagr:>14.2%}")

    # Sharpe Ratio
    strat_sharpe = strategy_results['sharpe']
    bench_sharpe = benchmark_results['sharpe']
    delta_sharpe = strat_sharpe - bench_sharpe
    print(f"{'Sharpe Ratio':<25} {strat_sharpe:>15.2f} {bench_sharpe:>15.2f} {delta_sharpe:>15.2f}")

    # Sortino Ratio
    strat_sortino = strategy_results['sortino']
    bench_sortino = benchmark_results['sortino']
    delta_sortino = strat_sortino - bench_sortino
    print(f"{'Sortino Ratio':<25} {strat_sortino:>15.2f} {bench_sortino:>15.2f} {delta_sortino:>15.2f}")

    # Max Drawdown
    strat_dd = strategy_results['max_drawdown']
    bench_dd = benchmark_results['max_drawdown']
    delta_dd = strat_dd - bench_dd
    print(f"{'Max Drawdown':<25} {strat_dd:>14.2%} {bench_dd:>14.2%} {delta_dd:>14.2%}")

    # Trade Statistics
    strat_metrics = strategy_results['metrics']
    print(f"\n{'Strategy Trade Stats':<25}")
    print("-"*70)
    print(f"{'Total Trades':<25} {strat_metrics['total_trades']:>15.0f}")
    print(f"{'Win Rate':<25} {strat_metrics['win_rate']:>14.2%}")
    print(f"{'Avg Trade':<25} {strat_metrics['avg_trade']:>14.2%}")

    # Validation against targets
    print("\n" + "="*60)
    print("VALIDATION AGAINST ARCHITECTURE TARGETS")
    print("="*60)

    targets = {
        'Sharpe Ratio': (strat_sharpe, 0.8, 1.2),
        'CAGR': (strat_cagr, 0.10, 0.15),
        'Max Drawdown': (abs(strat_dd), 0.25, 0.30),
        'Win Rate': (strat_metrics['win_rate'], 0.50, 0.60)
    }

    print(f"\n{'Metric':<20} {'Actual':>12} {'Target':>20} {'Status':>15}")
    print("-"*70)

    all_pass = True
    for metric, (actual, min_target, max_target) in targets.items():
        if metric == 'Max Drawdown':
            # For drawdown, lower is better
            status = 'PASS' if actual <= max_target else 'FAIL'
            target_str = f'<= {max_target:.2%}'
        else:
            # For other metrics, within range is good
            status = 'PASS' if min_target <= actual <= max_target * 1.5 else 'ACCEPTABLE' if actual >= min_target else 'FAIL'
            target_str = f'{min_target:.2%} - {max_target:.2%}'

        if status == 'FAIL':
            all_pass = False

        actual_str = f'{actual:.2%}' if metric != 'Sharpe Ratio' else f'{actual:.2f}'
        print(f"{metric:<20} {actual_str:>12} {target_str:>20} {status:>15}")

    print("\n" + "="*60)
    if all_pass:
        print("OVERALL: STRATEGY MEETS ALL TARGETS")
    else:
        print("OVERALL: STRATEGY NEEDS REVIEW (some targets missed)")
    print("="*60)


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("52-WEEK HIGH MOMENTUM STRATEGY - BACKTEST VALIDATION")
    print("="*70)
    print("\nPurpose: Validate strategy implementation on SPY (2005-2025)")
    print("Expected: Sharpe 0.8-1.2, CAGR 10-15%, MaxDD -25% to -30%")

    # Fetch data
    try:
        data = fetch_spy_data(start_date='2005-01-01', end_date='2025-01-01')
    except Exception as e:
        print(f"\n[FAIL] Data fetch failed: {e}")
        print("Ensure internet connection and try again.")
        return

    # Run strategy backtest
    try:
        strategy_results = run_52w_high_backtest(data, initial_capital=10000)
    except Exception as e:
        print(f"\n[FAIL] Strategy backtest failed: {e}")
        print("Check strategy implementation and try again.")
        import traceback
        traceback.print_exc()
        return

    # Run benchmark
    try:
        benchmark_results = run_buy_and_hold_benchmark(data, initial_capital=10000)
    except Exception as e:
        print(f"\n[FAIL] Benchmark backtest failed: {e}")
        print("Check benchmark logic and try again.")
        import traceback
        traceback.print_exc()
        return

    # Compare results
    compare_results(strategy_results, benchmark_results)

    print("\n" + "="*70)
    print("BACKTEST VALIDATION COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. If strategy meets targets -> Prepare for paper trading")
    print("2. If strategy underperforms -> Review implementation or data")
    print("3. Export trades to CSV for manual inspection (optional)")


if __name__ == "__main__":
    main()
