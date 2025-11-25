"""
Phase 1: Continuation Bar Impact Analysis

Analyzes how continuation bars (directional bars after pattern entry) impact:
1. Hit rates (already validated in Session 57)
2. Risk-reward ratios (NEW - this is the key question)
3. Average wins vs average losses
4. Overall expectancy

HYPOTHESIS: Patterns with 2+ continuation bars have BOTH higher hit rates AND better R:R ratios
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent
VALIDATION_FILES = {
    '1H': BASE_DIR / 'strat_validation_1H.csv',
    '1D': BASE_DIR / 'strat_validation_1D.csv',
    '1W': BASE_DIR / 'strat_validation_1W.csv',
    '1M': BASE_DIR / 'strat_validation_1M.csv'
}

def calculate_rr_metrics(df):
    """Calculate risk-reward metrics for a group of patterns"""

    # Filter to patterns where we have outcome data
    winners = df[df['magnitude_hit'] == True].copy()
    losers = df[df['magnitude_hit'] == False].copy()

    if len(df) == 0:
        return {
            'pattern_count': 0,
            'hit_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'rr_ratio': 0.0,
            'expectancy': 0.0
        }

    # Calculate metrics
    hit_rate = len(winners) / len(df) if len(df) > 0 else 0.0
    avg_win = winners['actual_pnl_pct'].mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers['actual_pnl_pct'].mean()) if len(losers) > 0 else 0.0

    # Risk-reward ratio
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # Expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)
    expectancy = (hit_rate * avg_win) - ((1 - hit_rate) * avg_loss)

    return {
        'pattern_count': len(df),
        'hit_rate': hit_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'rr_ratio': rr_ratio,
        'expectancy': expectancy
    }


def analyze_continuation_bar_groups(df, timeframe):
    """Analyze patterns grouped by continuation bar count"""

    print(f"\n{'='*80}")
    print(f"TIMEFRAME: {timeframe}")
    print(f"{'='*80}")

    # Group patterns
    group_0_1 = df[df['continuation_bars'] <= 1]
    group_2_plus = df[df['continuation_bars'] >= 2]

    # Calculate metrics for each group
    metrics_0_1 = calculate_rr_metrics(group_0_1)
    metrics_2_plus = calculate_rr_metrics(group_2_plus)

    # Display comparison
    print(f"\n0-1 Continuation Bars (N={metrics_0_1['pattern_count']}):")
    print(f"  Hit Rate:     {metrics_0_1['hit_rate']:.1%}")
    print(f"  Avg Win:      {metrics_0_1['avg_win_pct']:.2%}")
    print(f"  Avg Loss:     {metrics_0_1['avg_loss_pct']:.2%}")
    print(f"  R:R Ratio:    {metrics_0_1['rr_ratio']:.2f}:1")
    print(f"  Expectancy:   {metrics_0_1['expectancy']:.2%}")

    print(f"\n2+ Continuation Bars (N={metrics_2_plus['pattern_count']}):")
    print(f"  Hit Rate:     {metrics_2_plus['hit_rate']:.1%}")
    print(f"  Avg Win:      {metrics_2_plus['avg_win_pct']:.2%}")
    print(f"  Avg Loss:     {metrics_2_plus['avg_loss_pct']:.2%}")
    print(f"  R:R Ratio:    {metrics_2_plus['rr_ratio']:.2f}:1")
    print(f"  Expectancy:   {metrics_2_plus['expectancy']:.2%}")

    # Calculate improvements
    if metrics_0_1['hit_rate'] > 0:
        hit_rate_improvement = ((metrics_2_plus['hit_rate'] - metrics_0_1['hit_rate'])
                               / metrics_0_1['hit_rate']) * 100
    else:
        hit_rate_improvement = 0

    if metrics_0_1['rr_ratio'] > 0:
        rr_improvement = ((metrics_2_plus['rr_ratio'] - metrics_0_1['rr_ratio'])
                         / metrics_0_1['rr_ratio']) * 100
    else:
        rr_improvement = 0

    print(f"\nIMPROVEMENTS (2+ vs 0-1 bars):")
    print(f"  Hit Rate:     {hit_rate_improvement:+.1f}%")
    print(f"  R:R Ratio:    {rr_improvement:+.1f}%")
    print(f"  Expectancy:   {(metrics_2_plus['expectancy'] - metrics_0_1['expectancy']):.2%} absolute")

    # CRITICAL FINDING indicator
    if metrics_2_plus['rr_ratio'] >= 1.5 and metrics_2_plus['hit_rate'] >= 0.65:
        print(f"\n*** CRITICAL FINDING: 2+ bar filter achieves TARGET METRICS ***")
        print(f"    R:R {metrics_2_plus['rr_ratio']:.2f}:1 >= 1.5:1 target")
        print(f"    Hit Rate {metrics_2_plus['hit_rate']:.1%} >= 65% target")
    elif metrics_2_plus['rr_ratio'] >= 2.0:
        print(f"\n*** EXCELLENT: 2+ bar filter achieves 2:1 R:R target ***")

    return {
        '0-1_bars': metrics_0_1,
        '2+_bars': metrics_2_plus,
        'improvements': {
            'hit_rate_pct': hit_rate_improvement,
            'rr_pct': rr_improvement,
            'expectancy_abs': metrics_2_plus['expectancy'] - metrics_0_1['expectancy']
        }
    }


def analyze_by_pattern_type(df, timeframe):
    """Analyze continuation bar impact by specific pattern type"""

    print(f"\n{'-'*80}")
    print(f"PATTERN TYPE BREAKDOWN - {timeframe}")
    print(f"{'-'*80}")

    pattern_types = df['pattern_type'].unique()

    for pattern_type in sorted(pattern_types):
        pattern_df = df[df['pattern_type'] == pattern_type]

        if len(pattern_df) < 5:  # Skip if too few patterns
            continue

        print(f"\n{pattern_type}:")

        group_0_1 = pattern_df[pattern_df['continuation_bars'] <= 1]
        group_2_plus = pattern_df[pattern_df['continuation_bars'] >= 2]

        metrics_0_1 = calculate_rr_metrics(group_0_1)
        metrics_2_plus = calculate_rr_metrics(group_2_plus)

        print(f"  0-1 bars (N={metrics_0_1['pattern_count']}): "
              f"{metrics_0_1['hit_rate']:.1%} hit, {metrics_0_1['rr_ratio']:.2f}:1 R:R")
        print(f"  2+ bars  (N={metrics_2_plus['pattern_count']}): "
              f"{metrics_2_plus['hit_rate']:.1%} hit, {metrics_2_plus['rr_ratio']:.2f}:1 R:R")

        # Highlight exceptional patterns
        if metrics_2_plus['rr_ratio'] >= 2.0 and metrics_2_plus['pattern_count'] >= 5:
            print(f"  *** {pattern_type} with 2+ bars achieves 2:1 R:R target! ***")


def main():
    """Run comprehensive continuation bar analysis"""

    print("="*80)
    print("PHASE 1: CONTINUATION BAR IMPACT ON RISK-REWARD RATIOS")
    print("="*80)
    print("\nOBJECTIVE: Determine if filtering for 2+ continuation bars improves R:R")
    print("HYPOTHESIS: 2+ bars = higher hit rate AND better R:R ratios")

    results = {}

    for timeframe, filepath in VALIDATION_FILES.items():
        if not filepath.exists():
            print(f"\nWARNING: {filepath} not found, skipping {timeframe}")
            continue

        # Load data
        df = pd.read_csv(filepath)

        # Overall analysis
        results[timeframe] = analyze_continuation_bar_groups(df, timeframe)

        # Pattern-specific analysis
        analyze_by_pattern_type(df, timeframe)

    # Summary report
    print(f"\n\n{'='*80}")
    print("SUMMARY: CONTINUATION BAR FILTER RECOMMENDATION")
    print(f"{'='*80}")

    for timeframe, result in results.items():
        metrics_2_plus = result['2+_bars']

        print(f"\n{timeframe} with 2+ continuation bars:")
        print(f"  Hit Rate: {metrics_2_plus['hit_rate']:.1%}")
        print(f"  R:R:      {metrics_2_plus['rr_ratio']:.2f}:1")
        print(f"  Status:   ", end="")

        if metrics_2_plus['rr_ratio'] >= 2.0:
            print("EXCEEDS TARGET (2:1)")
        elif metrics_2_plus['rr_ratio'] >= 1.5:
            print("ACCEPTABLE (1.5:1+)")
        else:
            print("BELOW TARGET (needs further optimization)")

    # GO/NO-GO decision for mandatory filtering
    print(f"\n{'-'*80}")
    print("RECOMMENDATION:")
    print(f"{'-'*80}")

    weekly_2plus = results.get('1W', {}).get('2+_bars', {})

    if weekly_2plus.get('rr_ratio', 0) >= 1.5:
        print("MANDATORY FILTER: Require 2+ continuation bars for all patterns")
        print(f"  Rationale: Weekly 2-2 Up achieves {weekly_2plus['rr_ratio']:.2f}:1 R:R")
        print(f"             with {weekly_2plus['hit_rate']:.1%} hit rate")
        print("\nNEXT STEP: Proceed to Phase 2 (target multipliers) with this filter active")
    else:
        print("CONTINUE OPTIMIZATION: 2+ bar filter helps but insufficient alone")
        print("NEXT STEP: Proceed to Phase 2 (target multipliers) for further improvement")


if __name__ == '__main__':
    main()
