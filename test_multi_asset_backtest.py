"""
Test Multi-Asset 52W Momentum Portfolio

Session 37 validation:
- Technology sector (30 stocks)
- 2020-2025 period
- Top 10 portfolio
- Semi-annual rebalance
- Target: Sharpe >= 0.8, CAGR >= 10%
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from integrations.stock_scanner_bridge import MomentumPortfolioBacktest


def main():
    """Run technology sector backtest."""

    print("\n" + "="*80)
    print("SESSION 38: Multi-Asset 52W Momentum Portfolio Test - Phase 1")
    print("="*80)

    # Initialize backtest
    # PHASE 1: Disable volume filter (Session 38 debugging)
    backtest = MomentumPortfolioBacktest(
        universe='technology',
        top_n=10,
        volume_threshold=None,  # CHANGED: Disable volume filter
        rebalance_frequency='semi_annual'
    )

    # Run backtest (2020-2025)
    results = backtest.run(
        start_date='2020-01-01',
        end_date='2025-01-01',
        initial_capital=100000
    )

    # Display selected portfolios
    print(f"\n{'='*80}")
    print("PORTFOLIO SELECTIONS OVER TIME")
    print(f"{'='*80}")

    for date, stocks in results['portfolios'].items():
        print(f"\n{date}: {len(stocks)} stocks")
        print(f"  {', '.join(stocks)}")

    # Gate 1 decision
    metrics = results['metrics']
    sharpe_pass = metrics['sharpe'] >= 0.8
    cagr_pass = metrics['cagr'] >= 0.10

    print(f"\n{'='*80}")
    print("GATE 1 VALIDATION DECISION")
    print(f"{'='*80}")

    if sharpe_pass and cagr_pass:
        decision = "PASS"
        print(f"Decision: {decision}")
        print("  Both Sharpe and CAGR exceed Gate 1 targets.")
        print("  Recommendation: Proceed to regime integration (Session 38)")
    elif sharpe_pass or cagr_pass:
        decision = "PARTIAL"
        print(f"Decision: {decision}")
        if sharpe_pass:
            print("  Sharpe meets target but CAGR below threshold.")
        else:
            print("  CAGR meets target but Sharpe below threshold.")
        print("  Recommendation: Analyze portfolio composition, consider sector diversification")
    else:
        decision = "FAIL"
        print(f"Decision: {decision}")
        print("  Neither Sharpe nor CAGR meet Gate 1 targets.")
        print("  Recommendation: Debug portfolio selection logic or try different universe")

    print(f"\nComparison to single-asset SPY (Session 36):")
    print(f"  SPY Sharpe: 0.74 | Multi-Asset Sharpe: {metrics['sharpe']:.2f}")
    print(f"  SPY CAGR: 5.5% | Multi-Asset CAGR: {metrics['cagr']:.2%}")

    # Save results
    print(f"\n{'='*80}")
    print("Results saved to SESSION_37_RESULTS.md")
    print(f"{'='*80}")

    return results, decision


if __name__ == "__main__":
    results, decision = main()

    print("\n\nBacktest complete.")
    print(f"Gate 1 Decision: {decision}")
