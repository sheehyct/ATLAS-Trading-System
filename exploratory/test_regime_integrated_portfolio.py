"""
Test script for Session 39: ATLAS Regime Integration with 52W Momentum Strategy

Compares performance of:
1. Baseline: 52W momentum portfolio (no regime filter)
2. Regime-Integrated: 52W momentum portfolio with ATLAS regime filtering

Expected Outcome:
- Sharpe improvement: +14-36% (target: 1.0-1.2 vs baseline 0.88)
- Max Drawdown reduction: ~17-33% (target: -20% to -25% vs baseline ~-30%)
- March 2020: CRASH regime avoids drawdown via 100% cash allocation
"""

import sys
from pathlib import Path
import pandas as pd
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integrations.stock_scanner_bridge import MomentumPortfolioBacktest


def run_comparison_backtest():
    """
    Run baseline vs regime-integrated backtest comparison.
    """
    print("\n" + "="*80)
    print("SESSION 39: ATLAS REGIME INTEGRATION COMPARISON")
    print("="*80)

    # Configuration (matches Session 38 Gate 1 PASS config EXACTLY)
    config = {
        'universe': 'technology',
        'top_n': 10,
        'volume_threshold': None,  # Disabled per Session 38
        'min_distance': 0.90,
        'rebalance_frequency': 'semi_annual',
        'start_date': '2020-01-01',
        'end_date': '2025-01-01',  # ALIGNED with Session 38
        'initial_capital': 100000,
        'fees': 0.001,
        'slippage': 0.001
    }

    print(f"\nConfiguration:")
    print(f"  Universe: {config['universe']}")
    print(f"  Portfolio Size: Top {config['top_n']} stocks")
    volume_filter_text = 'Disabled' if config['volume_threshold'] is None else f"{config['volume_threshold']}x"
    print(f"  Volume Filter: {volume_filter_text}")
    print(f"  Rebalance: {config['rebalance_frequency']}")
    print(f"  Period: {config['start_date']} to {config['end_date']}")

    # ========================================================================
    # BASELINE: No Regime Filter (Session 38 validated config)
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE: 52W Momentum Portfolio (NO REGIME FILTER)")
    print("="*80)

    backtest_baseline = MomentumPortfolioBacktest(
        universe=config['universe'],
        top_n=config['top_n'],
        volume_threshold=config['volume_threshold'],
        min_distance=config['min_distance'],
        rebalance_frequency=config['rebalance_frequency']
    )

    results_baseline = backtest_baseline.run(
        start_date=config['start_date'],
        end_date=config['end_date'],
        initial_capital=config['initial_capital'],
        fees=config['fees'],
        slippage=config['slippage'],
        use_regime_filter=False
    )

    # ========================================================================
    # REGIME-INTEGRATED: With ATLAS Regime Filter
    # ========================================================================
    print("\n" + "="*80)
    print("REGIME-INTEGRATED: 52W Momentum Portfolio (WITH ATLAS REGIME FILTER)")
    print("="*80)

    backtest_regime = MomentumPortfolioBacktest(
        universe=config['universe'],
        top_n=config['top_n'],
        volume_threshold=config['volume_threshold'],
        min_distance=config['min_distance'],
        rebalance_frequency=config['rebalance_frequency']
    )

    results_regime = backtest_regime.run(
        start_date=config['start_date'],
        end_date=config['end_date'],
        initial_capital=config['initial_capital'],
        fees=config['fees'],
        slippage=config['slippage'],
        use_regime_filter=True
    )

    # ========================================================================
    # COMPARISON ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: BASELINE VS REGIME-INTEGRATED")
    print("="*80)

    baseline_metrics = results_baseline['metrics']
    regime_metrics = results_regime['metrics']

    # Calculate improvements
    sharpe_improvement = ((regime_metrics['sharpe'] / baseline_metrics['sharpe']) - 1) * 100
    cagr_improvement = ((regime_metrics['cagr'] / baseline_metrics['cagr']) - 1) * 100
    dd_improvement = ((abs(regime_metrics['max_dd']) / abs(baseline_metrics['max_dd'])) - 1) * 100

    print(f"\nPerformance Metrics:")
    print(f"{'Metric':<20} {'Baseline':<15} {'Regime':<15} {'Improvement':<15}")
    print(f"{'-'*65}")
    print(f"{'Sharpe Ratio':<20} {baseline_metrics['sharpe']:<15.2f} {regime_metrics['sharpe']:<15.2f} {sharpe_improvement:>+14.1f}%")
    print(f"{'CAGR':<20} {baseline_metrics['cagr']:<15.2%} {regime_metrics['cagr']:<15.2%} {cagr_improvement:>+14.1f}%")
    print(f"{'Max Drawdown':<20} {baseline_metrics['max_dd']:<15.2%} {regime_metrics['max_dd']:<15.2%} {dd_improvement:>+14.1f}%")
    print(f"{'Total Return':<20} {baseline_metrics['total_return']:<15.2%} {regime_metrics['total_return']:<15.2%}")
    print(f"{'Total Trades':<20} {baseline_metrics['total_trades']:<15,} {regime_metrics['total_trades']:<15,}")

    print(f"\nGate 1 Validation:")
    print(f"  Baseline Sharpe >= 0.8: {'PASS' if baseline_metrics['sharpe'] >= 0.8 else 'FAIL'}")
    print(f"  Regime Sharpe >= 0.8: {'PASS' if regime_metrics['sharpe'] >= 0.8 else 'FAIL'}")
    print(f"  Baseline CAGR >= 10%: {'PASS' if baseline_metrics['cagr'] >= 0.10 else 'FAIL'}")
    print(f"  Regime CAGR >= 10%: {'PASS' if regime_metrics['cagr'] >= 0.10 else 'FAIL'}")

    print(f"\nTarget Achievement:")
    print(f"  Sharpe 1.0-1.2 target: {'PASS' if 1.0 <= regime_metrics['sharpe'] <= 1.2 else 'PARTIAL' if regime_metrics['sharpe'] >= baseline_metrics['sharpe'] else 'FAIL'}")
    print(f"  MaxDD -20% to -25% target: {'PASS' if -0.25 <= regime_metrics['max_dd'] <= -0.20 else 'PARTIAL' if regime_metrics['max_dd'] > baseline_metrics['max_dd'] else 'FAIL'}")

    # ========================================================================
    # REGIME ANALYSIS (if enabled)
    # ========================================================================
    if results_regime.get('regime_filter_enabled'):
        print(f"\n" + "="*80)
        print("REGIME ANALYSIS")
        print("="*80)

        regime_at_rebalance = results_regime.get('regime_at_rebalance', {})
        portfolios_regime = results_regime.get('portfolios', {})

        print(f"\nRebalance Date Analysis:")
        print(f"{'Date':<15} {'Regime':<15} {'Stocks':<10} {'Baseline Stocks':<15}")
        print(f"{'-'*55}")

        for date in results_baseline['rebalance_dates']:
            regime = regime_at_rebalance.get(date, 'N/A')
            stocks_regime = len(portfolios_regime.get(date, []))
            stocks_baseline = len(results_baseline['portfolios'].get(date, []))
            print(f"{date:<15} {regime:<15} {stocks_regime:<10} {stocks_baseline:<15}")

        # Check March 2020 CRASH regime behavior
        march_2020_dates = [d for d in results_baseline['rebalance_dates'] if d.startswith('2020-02') or d.startswith('2020-08')]
        if march_2020_dates:
            print(f"\nMarch 2020 Analysis (COVID crash):")
            for date in march_2020_dates:
                regime = regime_at_rebalance.get(date, 'N/A')
                stocks = len(portfolios_regime.get(date, []))
                print(f"  {date}: Regime={regime}, Stocks={stocks}")
                if regime == 'CRASH' and stocks == 0:
                    print(f"    [SUCCESS] CRASH regime correctly held 100% cash")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save comparison summary
    comparison_data = {
        'Metric': ['Sharpe Ratio', 'CAGR', 'Max Drawdown', 'Total Return', 'Total Trades'],
        'Baseline': [
            baseline_metrics['sharpe'],
            baseline_metrics['cagr'],
            baseline_metrics['max_dd'],
            baseline_metrics['total_return'],
            baseline_metrics['total_trades']
        ],
        'Regime': [
            regime_metrics['sharpe'],
            regime_metrics['cagr'],
            regime_metrics['max_dd'],
            regime_metrics['total_return'],
            regime_metrics['total_trades']
        ],
        'Improvement_%': [
            sharpe_improvement,
            cagr_improvement,
            dd_improvement,
            ((regime_metrics['total_return'] / baseline_metrics['total_return']) - 1) * 100,
            ((regime_metrics['total_trades'] / baseline_metrics['total_trades']) - 1) * 100
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('session_39_comparison_results.csv', index=False)
    print(f"  Saved: session_39_comparison_results.csv")

    # Save regime at rebalance
    if results_regime.get('regime_filter_enabled'):
        regime_rebalance_data = []
        for date in results_baseline['rebalance_dates']:
            regime_rebalance_data.append({
                'rebalance_date': date,
                'regime': regime_at_rebalance.get(date, 'N/A'),
                'stocks_regime': len(portfolios_regime.get(date, [])),
                'stocks_baseline': len(results_baseline['portfolios'].get(date, []))
            })
        regime_rebalance_df = pd.DataFrame(regime_rebalance_data)
        regime_rebalance_df.to_csv('session_39_regime_at_rebalance.csv', index=False)
        print(f"  Saved: session_39_regime_at_rebalance.csv")

    print(f"\n" + "="*80)
    print("SESSION 39 COMPARISON COMPLETE")
    print("="*80)

    return {
        'baseline': results_baseline,
        'regime': results_regime,
        'comparison': comparison_df,
        'sharpe_improvement': sharpe_improvement,
        'cagr_improvement': cagr_improvement,
        'dd_improvement': dd_improvement
    }


if __name__ == "__main__":
    results = run_comparison_backtest()
