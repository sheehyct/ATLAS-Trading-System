"""
Analyze Impact of 2D Hybrid Optimization (Session 65)

Compares Session 64 (baseline, no 2D) vs Session 65 (with 2D hybrid)
to measure R:R improvement, pattern count change, and hit rate impact.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("SESSION 65 - 2D HYBRID OPTIMIZATION IMPACT ANALYSIS")
print("=" * 80)
print()

# Session 64 baseline metrics (from analyze_continuation_bar_impact.py after bug fix)
print("SESSION 64 BASELINE (No 2D Hybrid):")
print("-" * 80)

baseline_hourly_22_up = {
    'pattern_count': 21,
    'hit_rate': 0.905,  # 19/21
    'avg_win_pct': 0.60,
    'avg_loss_pct': 0.39,
    'rr_ratio': 1.53
}

baseline_daily_212_up = {
    'pattern_count': 10,
    'hit_rate': 0.80,  # 8/10
    'avg_win_pct': 2.56,
    'avg_loss_pct': 0.97,
    'rr_ratio': 2.65
}

print(f"Hourly 2-2 Up + 2+ Continuation Bars:")
print(f"  Pattern Count: {baseline_hourly_22_up['pattern_count']}")
print(f"  Hit Rate: {baseline_hourly_22_up['hit_rate']:.1%}")
print(f"  R:R Ratio: {baseline_hourly_22_up['rr_ratio']:.2f}:1")
print(f"  Status: FAILS 2:1 target (24% below)")
print()

print(f"Daily 2-1-2 Up + 2+ Continuation Bars:")
print(f"  Pattern Count: {baseline_daily_212_up['pattern_count']}")
print(f"  Hit Rate: {baseline_daily_212_up['hit_rate']:.1%}")
print(f"  R:R Ratio: {baseline_daily_212_up['rr_ratio']:.2f}:1")
print(f"  Status: EXCEEDS 2:1 target (33% above)")
print()

# Load Session 65 results (with 2D hybrid)
print("=" * 80)
print("SESSION 65 RESULTS (With 2D Hybrid):")
print("-" * 80)

hourly_df = pd.read_csv('scripts/strat_validation_1H.csv')
daily_df = pd.read_csv('scripts/strat_validation_1D.csv')

def calculate_metrics(df, pattern_type, min_cont_bars=2):
    """Calculate metrics for a specific pattern with continuation bar filter."""
    subset = df[
        (df['pattern_type'] == pattern_type) &
        (df['continuation_bars'] >= min_cont_bars)
    ]

    if len(subset) == 0:
        return None

    winners = subset[subset['magnitude_hit'] == True]
    losers = subset[subset['magnitude_hit'] == False]

    if len(losers) == 0:
        avg_loss_pct = 0.01  # Avoid division by zero
    else:
        avg_loss_pct = abs(losers['actual_pnl_pct'].mean())

    avg_win_pct = winners['actual_pnl_pct'].mean() if len(winners) > 0 else 0
    rr_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0

    return {
        'pattern_count': len(subset),
        'hit_rate': len(winners) / len(subset),
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'rr_ratio': rr_ratio,
        'winners': len(winners),
        'losers': len(losers)
    }

# Analyze Hourly 2-2 Up
hourly_22_up_metrics = calculate_metrics(hourly_df, '2-2 Up', min_cont_bars=2)

if hourly_22_up_metrics:
    print(f"Hourly 2-2 Up + 2+ Continuation Bars:")
    print(f"  Pattern Count: {hourly_22_up_metrics['pattern_count']} (vs {baseline_hourly_22_up['pattern_count']} baseline)")
    print(f"  Hit Rate: {hourly_22_up_metrics['hit_rate']:.1%} (vs {baseline_hourly_22_up['hit_rate']:.1%} baseline)")
    print(f"  Avg Win: {hourly_22_up_metrics['avg_win_pct']:.2f}% (vs {baseline_hourly_22_up['avg_win_pct']:.2f}% baseline)")
    print(f"  Avg Loss: {hourly_22_up_metrics['avg_loss_pct']:.2f}% (vs {baseline_hourly_22_up['avg_loss_pct']:.2f}% baseline)")
    print(f"  R:R Ratio: {hourly_22_up_metrics['rr_ratio']:.2f}:1 (vs {baseline_hourly_22_up['rr_ratio']:.2f}:1 baseline)")

    # Calculate improvements
    count_change = (hourly_22_up_metrics['pattern_count'] - baseline_hourly_22_up['pattern_count']) / baseline_hourly_22_up['pattern_count'] * 100
    rr_improvement = (hourly_22_up_metrics['rr_ratio'] - baseline_hourly_22_up['rr_ratio']) / baseline_hourly_22_up['rr_ratio'] * 100
    hit_rate_change = (hourly_22_up_metrics['hit_rate'] - baseline_hourly_22_up['hit_rate']) * 100

    print()
    print(f"  Pattern Count Change: {count_change:+.1f}%")
    print(f"  R:R Improvement: {rr_improvement:+.1f}%")
    print(f"  Hit Rate Change: {hit_rate_change:+.1f} percentage points")
    print()

    # Status check
    if hourly_22_up_metrics['rr_ratio'] >= 2.0 and hourly_22_up_metrics['pattern_count'] >= 15:
        status = "SUCCESS - Meets 2:1 R:R target with sufficient patterns"
    elif hourly_22_up_metrics['rr_ratio'] >= 1.70:
        status = "PARTIAL SUCCESS - Close to 2:1, consider Phase 3 (monthly filter)"
    elif hourly_22_up_metrics['pattern_count'] < 15:
        status = "INSUFFICIENT DATA - Pattern count too low for significance"
    else:
        status = "FAILURE - 2D hybrid did not improve R:R significantly"

    print(f"  Status: {status}")
else:
    print("ERROR: No Hourly 2-2 Up patterns found with 2+ continuation bars!")

print()

# Analyze Daily 2-1-2 Up
daily_212_up_metrics = calculate_metrics(daily_df, '2-1-2 Up', min_cont_bars=2)

if daily_212_up_metrics:
    print(f"Daily 2-1-2 Up + 2+ Continuation Bars:")
    print(f"  Pattern Count: {daily_212_up_metrics['pattern_count']} (vs {baseline_daily_212_up['pattern_count']} baseline)")
    print(f"  Hit Rate: {daily_212_up_metrics['hit_rate']:.1%} (vs {baseline_daily_212_up['hit_rate']:.1%} baseline)")
    print(f"  R:R Ratio: {daily_212_up_metrics['rr_ratio']:.2f}:1 (vs {baseline_daily_212_up['rr_ratio']:.2f}:1 baseline)")

    if daily_212_up_metrics['rr_ratio'] >= 2.0:
        print(f"  Status: STILL EXCEEDS 2:1 target")
    else:
        print(f"  Status: WARNING - R:R degraded below 2:1 target!")
else:
    print("ERROR: No Daily 2-1-2 Up patterns found with 2+ continuation bars!")

print()

# Phase 2 Decision
print("=" * 80)
print("PHASE 2 DECISION:")
print("-" * 80)

if hourly_22_up_metrics:
    if hourly_22_up_metrics['rr_ratio'] >= 2.0 and hourly_22_up_metrics['pattern_count'] >= 15:
        print("SCENARIO A: SUCCESS")
        print("  - Hourly 2-2 Up now meets 2:1 R:R target")
        print("  - Pattern count sufficient (>=15)")
        print("  - RECOMMENDATION: Proceed to options module with BOTH patterns")
        print("    * Daily 2-1-2 Up (validated)")
        print("    * Hourly 2-2 Up (validated with 2D optimization)")

    elif hourly_22_up_metrics['rr_ratio'] >= 1.70 and hourly_22_up_metrics['pattern_count'] >= 12:
        print("SCENARIO B: PARTIAL SUCCESS")
        print(f"  - Hourly 2-2 Up R:R improved to {hourly_22_up_metrics['rr_ratio']:.2f}:1 (close to 2:1)")
        print(f"  - Pattern count: {hourly_22_up_metrics['pattern_count']} (acceptable)")
        print("  - RECOMMENDATION: Implement Phase 3 (Monthly Alignment Filter)")
        print("    * Expected: +23.7 pp improvement from monthly bias")
        print("    * Risk: Pattern count may reduce to 10-12")
        print("    * Timeline: 1-2 hours additional work")

    elif hourly_22_up_metrics['pattern_count'] < 15:
        print("SCENARIO C: INSUFFICIENT DATA")
        print(f"  - Pattern count: {hourly_22_up_metrics['pattern_count']} (below 15 minimum)")
        print("  - 2D hybrid reduced pattern count below statistical significance")
        print("  - RECOMMENDATION: Proceed with Daily 2-1-2 Up only (conservative)")
        print("    * OR expand to 10-stock validation for more data")

    else:
        print("SCENARIO D: FAILURE")
        print(f"  - R:R: {hourly_22_up_metrics['rr_ratio']:.2f}:1 (no meaningful improvement)")
        print("  - 2D hybrid optimization did not achieve expected results")
        print("  - RECOMMENDATION: Proceed with Daily 2-1-2 Up only")
        print("    * Investigate why 2D hybrid failed (data issue? implementation bug?)")
else:
    print("ERROR: Cannot make decision - no Hourly 2-2 Up data!")

print()
print("=" * 80)
print("END ANALYSIS")
print("=" * 80)
