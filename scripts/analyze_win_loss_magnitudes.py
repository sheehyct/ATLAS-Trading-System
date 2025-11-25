"""
Analyze actual win and loss magnitudes to understand R:R ratio problem.

HYPOTHESIS: Losses are too large relative to wins, suggesting stops are too wide.

ANALYSIS:
1. Compare entry-to-stop distance vs entry-to-target distance
2. Analyze actual loss magnitudes when stops hit
3. Determine if tighter stops would improve R:R without excessive stop-outs
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
VALIDATION_FILES = {
    '1H': BASE_DIR / 'strat_validation_1H.csv',
    '1D': BASE_DIR / 'strat_validation_1D.csv',
    '1W': BASE_DIR / 'strat_validation_1W.csv',
}


def analyze_stop_distances(df, timeframe):
    """Analyze stop and target distances from entry"""

    print(f"\n{'='*80}")
    print(f"TIMEFRAME: {timeframe}")
    print(f"{'='*80}")

    # Calculate distances
    df = df.copy()
    df['stop_distance_pct'] = abs((df['stop_price'] - df['entry_price']) / df['entry_price'] * 100)
    df['target_distance_pct'] = abs((df['target_price'] - df['entry_price']) / df['entry_price'] * 100)
    df['planned_rr'] = df['target_distance_pct'] / df['stop_distance_pct']

    # Overall statistics
    print(f"\nPLANNED R:R STATISTICS:")
    print(f"  Average planned R:R: {df['planned_rr'].mean():.2f}:1")
    print(f"  Median planned R:R:  {df['planned_rr'].median():.2f}:1")
    print(f"  Min planned R:R:     {df['planned_rr'].min():.2f}:1")
    print(f"  Max planned R:R:     {df['planned_rr'].max():.2f}:1")

    print(f"\nSTOP DISTANCE STATISTICS:")
    print(f"  Average stop distance: {df['stop_distance_pct'].mean():.2f}%")
    print(f"  Median stop distance:  {df['stop_distance_pct'].median():.2f}%")
    print(f"  Min stop distance:     {df['stop_distance_pct'].min():.2f}%")
    print(f"  Max stop distance:     {df['stop_distance_pct'].max():.2f}%")

    print(f"\nTARGET DISTANCE STATISTICS:")
    print(f"  Average target distance: {df['target_distance_pct'].mean():.2f}%")
    print(f"  Median target distance:  {df['target_distance_pct'].median():.2f}%")

    # Analyze actual outcomes
    winners = df[df['magnitude_hit'] == True].copy()
    losers = df[df['stop_hit'] == True].copy()

    print(f"\nACTUAL OUTCOME STATISTICS:")
    print(f"  Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"  Losers:  {len(losers)} ({len(losers)/len(df)*100:.1f}%)")

    if len(winners) > 0:
        print(f"\nWINNER ANALYSIS:")
        print(f"  Average win: {winners['actual_pnl_pct'].mean():.2f}%")
        print(f"  Median win:  {winners['actual_pnl_pct'].median():.2f}%")

    if len(losers) > 0:
        print(f"\nLOSER ANALYSIS:")
        print(f"  Average loss: {abs(losers['actual_pnl_pct'].mean()):.2f}%")
        print(f"  Median loss:  {abs(losers['actual_pnl_pct'].median()):.2f}%")

        # Key insight: Are actual losses close to stop distance?
        losers['actual_loss_pct'] = abs(losers['actual_pnl_pct'])
        losers['loss_vs_stop'] = losers['actual_loss_pct'] / losers['stop_distance_pct']

        print(f"\nSTOP EFFICIENCY (actual loss / planned stop distance):")
        print(f"  Average: {losers['loss_vs_stop'].mean():.2f}x")
        print(f"  Median:  {losers['loss_vs_stop'].median():.2f}x")

        if losers['loss_vs_stop'].median() < 0.7:
            print(f"\n  *** INSIGHT: Stops hitting prematurely (median {losers['loss_vs_stop'].median():.2f}x) ***")
            print(f"      Patterns aren't using full stop distance before reversing")
            print(f"      Opportunity: Tighter stops could improve R:R without increasing stop-outs")

    # Calculate R:R ratio
    if len(winners) > 0 and len(losers) > 0:
        actual_rr = winners['actual_pnl_pct'].mean() / abs(losers['actual_pnl_pct'].mean())
        print(f"\nACTUAL R:R RATIO: {actual_rr:.2f}:1")

        planned_rr_avg = df['planned_rr'].mean()
        print(f"PLANNED R:R RATIO: {planned_rr_avg:.2f}:1")
        print(f"R:R DEGRADATION: {(planned_rr_avg - actual_rr) / planned_rr_avg * 100:.1f}%")


def analyze_by_continuation_bars(df, timeframe):
    """Analyze how continuation bars affect stop hits"""

    print(f"\n{'-'*80}")
    print(f"CONTINUATION BAR ANALYSIS - {timeframe}")
    print(f"{'-'*80}")

    # Calculate distances for entire dataframe first
    df = df.copy()
    df['stop_distance_pct'] = abs((df['stop_price'] - df['entry_price']) / df['entry_price'] * 100)
    df['target_distance_pct'] = abs((df['target_price'] - df['entry_price']) / df['entry_price'] * 100)

    group_0_1 = df[df['continuation_bars'] <= 1]
    group_2_plus = df[df['continuation_bars'] >= 2]

    for name, group in [("0-1 Continuation Bars", group_0_1), ("2+ Continuation Bars", group_2_plus)]:
        losers = group[group['stop_hit'] == True]

        if len(losers) > 0:
            print(f"\n{name} (N={len(group)}):")
            print(f"  Stop-out rate: {len(losers)/len(group)*100:.1f}%")
            print(f"  Avg loss when stopped: {abs(losers['actual_pnl_pct'].mean()):.2f}%")
            print(f"  Avg stop distance:     {losers['stop_distance_pct'].mean():.2f}%")

            losers_copy = losers.copy()
            losers_copy['actual_loss_pct'] = abs(losers_copy['actual_pnl_pct'])
            losers_copy['loss_vs_stop'] = losers_copy['actual_loss_pct'] / losers_copy['stop_distance_pct']

            print(f"  Stop efficiency:       {losers_copy['loss_vs_stop'].median():.2f}x")


def main():
    """Analyze win/loss magnitudes across timeframes"""

    print("="*80)
    print("WIN/LOSS MAGNITUDE ANALYSIS")
    print("="*80)
    print("\nOBJECTIVE: Understand why R:R ratios are poor")
    print("HYPOTHESIS: Stops are too wide relative to actual price movement")

    for timeframe, filepath in VALIDATION_FILES.items():
        if not filepath.exists():
            continue

        df = pd.read_csv(filepath)
        analyze_stop_distances(df, timeframe)
        analyze_by_continuation_bars(df, timeframe)

    print(f"\n\n{'='*80}")
    print("STOP ADJUSTMENT RECOMMENDATIONS")
    print(f"{'='*80}")
    print("\nBased on the analysis above:")
    print("1. If stop efficiency < 0.7x: Stops too wide, tighten without harming performance")
    print("2. If stop efficiency ~1.0x: Stops correctly sized, problem is target distance")
    print("3. If stop efficiency > 1.0x: Stops too tight, patterns need more room")


if __name__ == '__main__':
    main()
