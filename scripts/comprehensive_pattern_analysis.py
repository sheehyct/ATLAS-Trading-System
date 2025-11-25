"""
Comprehensive STRAT Pattern Analysis - Session 67

PURPOSE:
    Test ALL 4 core pattern types (3-1-2, 2-1-2, 2-2, 3-2) across ALL 4 timeframes
    (1H, 1D, 1W, 1M) with P/L EXPECTANCY as PRIMARY metric.

STRATEGIC SHIFT:
    Sessions 65-66 optimized patterns in isolation. Session 67 implements systematic
    testing of ALL patterns with SAME methodology, ranked by ACTUAL performance.

KEY PRINCIPLE:
    P/L Expectancy > Win Rate

    Example: 99 losses × $1 + 1 win × $1000 = 1% win rate but $901 profit = EXCELLENT

    P/L Expectancy = (Avg Win × Win Rate) - (Avg Loss × Loss Rate)

OUTPUT:
    1. Ranked table by P/L expectancy (highest to lowest)
    2. CSV export with all 32 combinations
    3. Top 3-5 pattern recommendations for Phase 2 (50-stock validation)
    4. GO/NO-GO decision criteria

USAGE:
    # 3-stock quick test (30 min)
    python scripts/comprehensive_pattern_analysis.py --symbols AAPL MSFT GOOGL --start 2024-01-01 --end 2025-01-01

    # 10-stock extended test (2 hrs)
    python scripts/comprehensive_pattern_analysis.py --symbols AAPL MSFT GOOGL AMD INTC QCOM JPM BAC WFC UNH --start 2023-01-01 --end 2025-01-01

    # 50-stock full test (8-12 hrs)
    python scripts/comprehensive_pattern_analysis.py --full-universe --start 2020-01-01 --end 2025-01-01

SESSION: 67 (2025-11-23)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# CRITICAL: Import existing validation infrastructure (no modifications needed)
# This reuses the 1107-line backtest_strat_equity_validation.py as-is
from scripts.backtest_strat_equity_validation import EquityValidationBacktest, VALIDATION_CONFIG


def calculate_combination_metrics(subset_df: pd.DataFrame) -> dict:
    """
    Calculate P/L expectancy and all metrics for a single pattern+timeframe combination.

    PRIMARY METRIC: P/L Expectancy = (avg_win × win_rate) - (avg_loss × loss_rate)

    Parameters:
    -----------
    subset_df : pd.DataFrame
        Subset of patterns for specific pattern_type + detection_timeframe combination

    Returns:
    --------
    dict with keys:
        - pl_expectancy (float): PRIMARY METRIC for ranking
        - pattern_count (int): Total patterns detected
        - win_rate (float): Magnitude hit rate (0-1)
        - loss_rate (float): Stop hit rate (0-1)
        - unresolved_rate (float): Neither hit nor stopped (0-1)
        - avg_win_pct (float): Average winning trade %
        - avg_loss_pct (float): Average losing trade % (absolute value)
        - risk_reward (float): avg_win / avg_loss ratio
        - median_bars_to_magnitude (float): Median bars to target
        - avg_continuation_bars (float): Average continuation bars after entry

    Edge Cases Handled:
    -------------------
    - Zero patterns: Returns all metrics as 0 or NaN
    - No wins: avg_win = 0, win_rate = 0
    - No losses: avg_loss = 0, loss_rate = 0, R:R = inf (capped at 999)
    - Division by zero: Protected with conditionals
    """
    total = len(subset_df)

    # Edge case: No patterns
    if total == 0:
        return {
            'pl_expectancy': 0.0,
            'pattern_count': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'unresolved_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'risk_reward': 0.0,
            'median_bars_to_magnitude': np.nan,
            'avg_continuation_bars': np.nan
        }

    # Count outcomes
    hits = subset_df['magnitude_hit'].sum()
    stops = subset_df['stop_hit'].sum()
    unresolved = total - (hits + stops)

    # Rates (0-1 range)
    win_rate = hits / total
    loss_rate = stops / total
    unresolved_rate = unresolved / total

    # Separate wins and losses
    wins = subset_df[subset_df['magnitude_hit'] == True]
    losses = subset_df[subset_df['stop_hit'] == True]

    # Average win/loss percentages
    avg_win = wins['actual_pnl_pct'].mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses['actual_pnl_pct'].mean()) if len(losses) > 0 else 0.0

    # PRIMARY METRIC: P/L Expectancy
    # Example: 80% win rate × 2% avg win - 20% loss rate × 1% avg loss
    #          = 1.6% - 0.2% = 1.4% expectancy
    pl_expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)

    # Risk-Reward Ratio (guard division by zero)
    if avg_loss > 0:
        risk_reward = avg_win / avg_loss
    else:
        # No losses = infinite R:R, cap at 999 for display
        risk_reward = 999.0 if avg_win > 0 else 0.0

    # Quality metrics
    magnitude_hits = subset_df[subset_df['magnitude_hit'] == True]
    median_bars = magnitude_hits['bars_to_magnitude'].median() if len(magnitude_hits) > 0 else np.nan

    avg_cont_bars = subset_df['continuation_bars'].mean() if 'continuation_bars' in subset_df.columns else np.nan

    return {
        'pl_expectancy': pl_expectancy,
        'pattern_count': total,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'unresolved_rate': unresolved_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'risk_reward': risk_reward,
        'median_bars_to_magnitude': median_bars,
        'avg_continuation_bars': avg_cont_bars
    }


def build_results_matrix(combined_df: pd.DataFrame, min_count: int = 10) -> dict:
    """
    Build results matrix for all 32 combinations (8 patterns × 4 timeframes).

    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined DataFrame with all patterns from all timeframes
        Must have columns: pattern_type, detection_timeframe, magnitude_hit, etc.

    min_count : int
        Minimum pattern count to include in ranking (default 10)
        Combinations with < min_count patterns are tracked separately

    Returns:
    --------
    dict with keys:
        - 'ranked': List of dicts with metrics for combinations >= min_count
        - 'insufficient': List of dicts for combinations < min_count
        - 'zero_patterns': List of (pattern_type, timeframe) tuples with 0 patterns
    """
    pattern_types = ['3-1-2 Up', '3-1-2 Down', '2-1-2 Up', '2-1-2 Down',
                     '2-2 Up', '2-2 Down', '3-2 Up', '3-2 Down']
    timeframes = ['1H', '1D', '1W', '1M']

    ranked_combinations = []
    insufficient_combinations = []
    zero_pattern_combinations = []

    for pattern_type in pattern_types:
        for timeframe in timeframes:
            # Filter to specific combination
            subset = combined_df[
                (combined_df['pattern_type'] == pattern_type) &
                (combined_df['detection_timeframe'] == timeframe)
            ]

            count = len(subset)

            if count == 0:
                # Track but don't include in any list
                zero_pattern_combinations.append((pattern_type, timeframe))
            elif count < min_count:
                # Calculate metrics but mark as insufficient
                metrics = calculate_combination_metrics(subset)
                metrics['pattern_type'] = pattern_type
                metrics['timeframe'] = timeframe
                insufficient_combinations.append(metrics)
            else:
                # Calculate metrics and include in ranking
                metrics = calculate_combination_metrics(subset)
                metrics['pattern_type'] = pattern_type
                metrics['timeframe'] = timeframe
                ranked_combinations.append(metrics)

    return {
        'ranked': ranked_combinations,
        'insufficient': insufficient_combinations,
        'zero_patterns': zero_pattern_combinations
    }


def create_ranked_dataframe(ranked_combinations: list) -> pd.DataFrame:
    """
    Create DataFrame sorted by P/L expectancy (descending).

    Parameters:
    -----------
    ranked_combinations : list of dict
        Combinations with sufficient pattern count

    Returns:
    --------
    pd.DataFrame sorted by pl_expectancy descending
    """
    if len(ranked_combinations) == 0:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'pattern_type', 'timeframe', 'pl_expectancy', 'pattern_count',
            'win_rate', 'loss_rate', 'unresolved_rate', 'avg_win_pct',
            'avg_loss_pct', 'risk_reward', 'median_bars_to_magnitude',
            'avg_continuation_bars'
        ])

    df = pd.DataFrame(ranked_combinations)

    # Sort by P/L expectancy (PRIMARY METRIC) descending
    df = df.sort_values('pl_expectancy', ascending=False).reset_index(drop=True)

    # Add rank column (1-indexed)
    df.insert(0, 'rank', range(1, len(df) + 1))

    return df


def print_ranked_table(ranked_df: pd.DataFrame, top_n: int = 10):
    """
    Print formatted console table with top N patterns.

    Parameters:
    -----------
    ranked_df : pd.DataFrame
        Sorted DataFrame from create_ranked_dataframe()
    top_n : int
        Number of top patterns to display (default 10)
    """
    print("\n" + "=" * 110)
    print("RANKED BY P/L EXPECTANCY (Top {})".format(min(top_n, len(ranked_df))))
    print("=" * 110)
    print()

    if len(ranked_df) == 0:
        print("NO PATTERNS WITH SUFFICIENT DATA (minimum 10 patterns required)")
        return

    # Header
    print(f"{'Rank':<6} {'Pattern':<12} {'TF':<4} {'P/L Exp':<9} {'Count':<7} {'Win%':<7} {'R:R':<7} {'Avg Win':<9} {'Avg Loss':<9} {'Med Bars':<9}")
    print("-" * 110)

    # Rows (top N)
    for _, row in ranked_df.head(top_n).iterrows():
        rank = int(row['rank'])
        pattern = row['pattern_type']
        tf = row['timeframe']
        pl_exp = row['pl_expectancy']
        count = int(row['pattern_count'])
        win_rate = row['win_rate']
        rr = row['risk_reward']
        avg_win = row['avg_win_pct']
        avg_loss = row['avg_loss_pct']
        med_bars = row['median_bars_to_magnitude']

        print(f"{rank:<6} {pattern:<12} {tf:<4} {pl_exp:>7.2f}%  {count:<7} {win_rate:>5.1%}  {rr:>5.2f}  {avg_win:>7.2f}%  {avg_loss:>7.2f}%  {med_bars:>7.1f}")

    print()


def print_insufficient_data_table(insufficient_combinations: list, zero_patterns: list):
    """
    Print table of combinations with insufficient data (< 10 patterns).

    Parameters:
    -----------
    insufficient_combinations : list of dict
        Combinations with 1-9 patterns
    zero_patterns : list of tuples
        (pattern_type, timeframe) tuples with 0 patterns
    """
    print("\n" + "=" * 110)
    print("INSUFFICIENT DATA COMBINATIONS (< 10 patterns)")
    print("=" * 110)
    print()

    if len(insufficient_combinations) == 0 and len(zero_patterns) == 0:
        print("ALL COMBINATIONS HAVE SUFFICIENT DATA")
        return

    # Header
    print(f"{'Pattern':<12} {'TF':<4} {'Count':<7} {'Win%':<7} {'P/L Exp':<9} {'Notes':<40}")
    print("-" * 110)

    # Insufficient data (1-9 patterns)
    for combo in insufficient_combinations:
        pattern = combo['pattern_type']
        tf = combo['timeframe']
        count = int(combo['pattern_count'])
        win_rate = combo['win_rate']
        pl_exp = combo['pl_expectancy']
        notes = f"Too few patterns for statistical significance"

        print(f"{pattern:<12} {tf:<4} {count:<7} {win_rate:>5.1%}  {pl_exp:>7.2f}%  {notes:<40}")

    # Zero patterns
    for pattern, tf in zero_patterns:
        print(f"{pattern:<12} {tf:<4} {'0':<7} {'N/A':<7} {'N/A':<9} {'No patterns detected (rare pattern or continuity filter)':<40}")

    print()


def identify_top_performers(ranked_df: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """
    Identify top 3-5 patterns for Phase 2 (50-stock validation).

    Parameters:
    -----------
    ranked_df : pd.DataFrame
        Sorted DataFrame with all combinations
    criteria : dict
        Selection criteria with keys:
        - min_pl_expectancy: Minimum P/L expectancy (e.g., 0.003 = 0.3%)
        - min_win_rate: Minimum win rate (e.g., 0.70 = 70%)
        - min_pattern_count: Minimum patterns on small test (extrapolates to 100+ on 50 stocks)

    Returns:
    --------
    pd.DataFrame with top performers meeting ALL criteria
    """
    top_performers = ranked_df[
        (ranked_df['pl_expectancy'] >= criteria['min_pl_expectancy']) &
        (ranked_df['win_rate'] >= criteria['min_win_rate']) &
        (ranked_df['pattern_count'] >= criteria['min_pattern_count'])
    ].copy()

    return top_performers


def print_top_performers_summary(top_performers: pd.DataFrame, criteria: dict, test_config: dict):
    """
    Print summary of top performers selected for Phase 2.

    Parameters:
    -----------
    top_performers : pd.DataFrame
        Patterns meeting all selection criteria
    criteria : dict
        Selection criteria used
    test_config : dict
        Test configuration (symbols, period, etc.)
    """
    print("\n" + "=" * 110)
    print("TOP PERFORMERS FOR PHASE 2 VALIDATION (50-Stock Universe)")
    print("=" * 110)
    print()

    if len(top_performers) == 0:
        print("NO PATTERNS MET ALL SELECTION CRITERIA")
        print()
        print("CRITERIA:")
        print(f"  - P/L Expectancy >= {criteria['min_pl_expectancy']*100:.1f}%")
        print(f"  - Win Rate >= {criteria['min_win_rate']*100:.0f}%")
        print(f"  - Pattern Count >= {criteria['min_pattern_count']} (extrapolates to 100+ on 50 stocks)")
        print()
        print("RECOMMENDATION: Lower criteria or investigate why patterns underperform")
        return

    print(f"SELECTED PATTERNS: {len(top_performers)}")
    print()

    for idx, row in top_performers.iterrows():
        rank = int(row['rank'])
        pattern = row['pattern_type']
        tf = row['timeframe']
        pl_exp = row['pl_expectancy']
        count = int(row['pattern_count'])
        win_rate = row['win_rate']
        rr = row['risk_reward']

        print(f"{rank}. {pattern} @ {tf}")
        print(f"   P/L Expectancy: {pl_exp:.2f}%")
        print(f"   Pattern Count: {count} (on {len(test_config['symbols'])} stocks)")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Risk-Reward: {rr:.2f}:1")
        print()

    print("CRITERIA MET:")
    print(f"  [X] P/L Expectancy >= {criteria['min_pl_expectancy']*100:.1f}%")
    print(f"  [X] Win Rate >= {criteria['min_win_rate']*100:.0f}%")
    print(f"  [X] Pattern Count >= {criteria['min_pattern_count']}")
    print()

    # GO/NO-GO recommendation
    if len(top_performers) >= 3:
        print("GO DECISION: Proceed to Phase 2 (50-stock validation)")
        print(f"Rationale: {len(top_performers)} patterns meet all criteria, sufficient for options module")
    elif len(top_performers) >= 1:
        print("CONDITIONAL GO: Proceed with caution")
        print(f"Rationale: Only {len(top_performers)} pattern(s) meet criteria, may want to expand selection")
    else:
        print("NO-GO DECISION: Do not proceed to Phase 2 yet")
        print("Rationale: No patterns meet minimum criteria")
    print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Comprehensive STRAT Pattern Analysis - Session 67')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to test (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--full-universe', action='store_true', help='Use full 50-stock universe')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-count', type=int, default=10, help='Minimum patterns per combination (default 10)')
    parser.add_argument('--output-dir', type=str, default='scripts', help='Output directory for CSV (default scripts/)')

    args = parser.parse_args()

    # Configure test
    if args.full_universe:
        symbols = [
            # Technology (10)
            'INTC', 'AMD', 'QCOM', 'TXN', 'ADI', 'MRVL', 'ON', 'MCHP', 'SWKS', 'NXPI',
            # Financials (10)
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'AXP', 'USB', 'PNC',
            # Healthcare (10)
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'LLY', 'BMY',
            # Consumer (10)
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            # Industrials (10)
            'HON', 'UNP', 'CAT', 'BA', 'GE', 'MMM', 'DE', 'LMT', 'RTX', 'UPS'
        ]
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default: 3-stock quick test
        symbols = ['AAPL', 'MSFT', 'GOOGL']

    start_date = args.start if args.start else '2024-01-01'
    end_date = args.end if args.end else '2025-01-01'

    test_config = {
        'symbols': symbols,
        'start': start_date,
        'end': end_date,
        'min_count': args.min_count
    }

    # Print configuration
    print("\n" + "=" * 110)
    print("COMPREHENSIVE PATTERN ANALYSIS - SESSION 67")
    print("=" * 110)
    print()
    print("Test Configuration:")
    print(f"  Symbols: {', '.join(symbols)} ({len(symbols)} stocks)")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Patterns: 8 types (3-1-2, 2-1-2, 2-2, 3-2 Up/Down)")
    print(f"  Timeframes: Will be configured based on data availability")
    print(f"  Minimum count: {args.min_count} patterns per combination")
    print()

    # Step 1: Run existing validation framework (reuses backtest_strat_equity_validation.py)
    print("Step 1: Running pattern detection across all timeframes...")
    print("(This may take 30 minutes to 12 hours depending on stock count and period)")
    print()

    # Create custom config with user's symbols and dates
    import copy
    custom_config = copy.deepcopy(VALIDATION_CONFIG)
    custom_config['stock_universe']['symbols'] = symbols
    custom_config['stock_universe']['count'] = len(symbols)
    custom_config['backtest_period']['start'] = start_date
    custom_config['backtest_period']['end'] = end_date

    # TEMPORARY: Skip hourly timeframe (Alpaca 401 error, Tiingo doesn't support hourly)
    # Test with daily/weekly/monthly only (24 combinations instead of 32)
    custom_config['timeframes']['detection'] = ['1D', '1W', '1M']
    custom_config['timeframes']['base'] = '1D'  # Use daily as base instead of hourly
    print("NOTE: Skipping hourly timeframe (Alpaca credentials issue)")
    print("      Testing daily/weekly/monthly only (24 combinations)")
    print()

    try:
        # Create validator with custom config
        validator = EquityValidationBacktest(config=custom_config)

        # Run validation (returns dict[timeframe -> DataFrame])
        raw_results = validator.run_validation()

    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        print("Check that all required data is available and .env is configured")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Combine all timeframes into single DataFrame
    print("Step 2: Combining results from all timeframes...")
    all_patterns = []
    for tf, df in raw_results.items():
        if 'detection_timeframe' not in df.columns:
            df['detection_timeframe'] = tf
        all_patterns.append(df)

    combined_df = pd.concat(all_patterns, ignore_index=True)
    print(f"Total patterns detected: {len(combined_df)}")
    print()

    # Step 3: Build results matrix for 32 combinations
    print("Step 3: Calculating metrics for 32 combinations...")
    results = build_results_matrix(combined_df, min_count=args.min_count)
    print(f"  Ranked combinations: {len(results['ranked'])}")
    print(f"  Insufficient data: {len(results['insufficient'])}")
    print(f"  Zero patterns: {len(results['zero_patterns'])}")
    print()

    # Step 4: Create ranked DataFrame
    ranked_df = create_ranked_dataframe(results['ranked'])

    # Step 5: Print ranked table (top 10)
    print_ranked_table(ranked_df, top_n=10)

    # Step 6: Print insufficient data combinations
    print_insufficient_data_table(results['insufficient'], results['zero_patterns'])

    # Step 7: Export to CSV
    output_path = Path(args.output_dir) / 'pattern_ranking_by_expectancy.csv'
    ranked_df.to_csv(output_path, index=False)
    print(f"Full results exported to: {output_path}")
    print()

    # Step 8: Identify top performers for Phase 2
    selection_criteria = {
        'min_pl_expectancy': 0.003,  # 0.3% minimum
        'min_win_rate': 0.70,  # 70% minimum
        'min_pattern_count': 8  # Extrapolates to 100+ on 50 stocks
    }

    top_performers = identify_top_performers(ranked_df, selection_criteria)
    print_top_performers_summary(top_performers, selection_criteria, test_config)

    print("=" * 110)
    print("SESSION 67 PHASE 1 COMPLETE")
    print("=" * 110)
    print()


if __name__ == '__main__':
    main()
