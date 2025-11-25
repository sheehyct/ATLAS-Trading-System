"""
2-1-2 Pattern Filter Sensitivity Analysis - Session 69

Tests 2-1-2 patterns across 8 filter configurations to compare with 3-1-2 results.
Adapted from Session 68 test_312_isolation.py.

Objective: Determine if 2-1-2 patterns have similar filter sensitivity to 3-1-2.

Output: 212_filter_analysis_matrix.csv with pattern counts and metrics for each config
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime

# Import Session 67 infrastructure (proven, tested code)
from scripts.backtest_strat_equity_validation import EquityValidationBacktest, VALIDATION_CONFIG


# Filter configurations to test (same as 3-1-2 analysis for comparison)
FILTER_CONFIGS = {
    'A_NoFilters': {
        'require_full_continuity': False,
        'use_flexible_continuity': False,
        'min_continuity_strength': 0,
        'require_continuation_bars': False,
        'min_continuation_bars': 0
    },
    'B_Continuity1': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 1,
        'require_continuation_bars': False,
        'min_continuation_bars': 0
    },
    'C_Continuity2': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 2,
        'require_continuation_bars': False,
        'min_continuation_bars': 0
    },
    'D_Continuity3': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 3,
        'require_continuation_bars': False,
        'min_continuation_bars': 0
    },
    'E_Cont1_Bars2': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 1,
        'require_continuation_bars': True,
        'min_continuation_bars': 2
    },
    'F_Cont2_Bars2': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 2,
        'require_continuation_bars': True,
        'min_continuation_bars': 2
    },
    'G_Cont3_Bars2': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 3,
        'require_continuation_bars': True,
        'min_continuation_bars': 2
    },
    'H_Session67Baseline': {
        'require_full_continuity': False,
        'use_flexible_continuity': True,
        'min_continuity_strength': 3,
        'require_continuation_bars': True,
        'min_continuation_bars': 2
    }
}


def calculate_212_metrics(patterns_df):
    """
    Calculate metrics for 2-1-2 patterns only.

    Returns dict with pattern counts and P/L metrics.
    """
    # Filter to 2-1-2 patterns (all 4 types: 2U-1-2U, 2D-1-2D, 2D-1-2U, 2U-1-2D)
    patterns_212 = patterns_df[patterns_df['pattern_type'].isin(['2-1-2 Up', '2-1-2 Down'])]

    if len(patterns_212) == 0:
        return {
            'total_212': 0,
            '212_up': 0,
            '212_down': 0,
            'pl_expectancy': np.nan,
            'win_rate': np.nan,
            'avg_win_pct': np.nan,
            'avg_loss_pct': np.nan,
            'rr_ratio': np.nan,
            'median_bars': np.nan
        }

    # Calculate metrics
    wins = patterns_212[patterns_212['magnitude_hit'] == True]
    losses = patterns_212[patterns_212['magnitude_hit'] == False]

    win_count = len(wins)
    loss_count = len(losses)
    total = len(patterns_212)

    win_rate = win_count / total if total > 0 else 0
    loss_rate = loss_count / total if total > 0 else 0

    avg_win_pct = wins['actual_pnl_pct'].mean() if win_count > 0 else 0
    avg_loss_pct = abs(losses['actual_pnl_pct'].mean()) if loss_count > 0 else 0

    pl_expectancy = (avg_win_pct * win_rate) - (avg_loss_pct * loss_rate)
    rr_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 999.0

    median_bars = patterns_212['bars_to_magnitude'].median()

    # Count by direction
    up_count = len(patterns_212[patterns_212['pattern_type'] == '2-1-2 Up'])
    down_count = len(patterns_212[patterns_212['pattern_type'] == '2-1-2 Down'])

    return {
        'total_212': total,
        '212_up': up_count,
        '212_down': down_count,
        'pl_expectancy': pl_expectancy,
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'rr_ratio': rr_ratio,
        'median_bars': median_bars
    }


def test_single_config(config_name, filter_config, symbols, start_date, end_date, timeframe):
    """
    Test 2-1-2 patterns with a single filter configuration.

    Returns dict with results for this config + timeframe combination.
    """
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name} @ {timeframe}")
    print(f"{'='*70}")
    print(f"  Continuity Enabled: {filter_config.get('use_flexible_continuity', False)}")
    print(f"  Min Strength: {filter_config.get('min_continuity_strength', 0)}")
    print(f"  Continuation Bars: {filter_config.get('require_continuation_bars', False)} "
          f"(min {filter_config.get('min_continuation_bars', 0)})")

    # Create custom config (clone VALIDATION_CONFIG and modify filters)
    test_config = {
        'backtest_period': {
            'start': start_date,
            'end': end_date,
            'rationale': f'Session 69 filter test - {config_name}'
        },
        'stock_universe': {
            'symbols': symbols,
            'count': len(symbols)
        },
        'timeframes': {
            'detection': [timeframe],  # Only test one timeframe at a time
            'continuity_check': ['1M', '1W', '1D', '4H', '1H']
        },
        'pattern_types': ['2-1-2 Up', '2-1-2 Down'],  # Only 2-1-2 patterns
        'filters': filter_config,
        'metrics': {
            'magnitude_window': 5,
            'max_holding_bars': 40,
            'magnitude_hit_threshold': 0.50,
            'risk_reward_threshold': 2.0,
            'min_pattern_count': 100
        }
    }

    # Create validator with custom config
    validator = EquityValidationBacktest(config=test_config)

    # Run validation
    try:
        results = validator.run_validation()

        # Extract patterns for this timeframe
        if timeframe not in results or len(results[timeframe]) == 0:
            print(f"  No patterns found")
            return {
                'config': config_name,
                'timeframe': timeframe,
                **calculate_212_metrics(pd.DataFrame())
            }

        patterns_df = results[timeframe]

        # Calculate metrics
        metrics = calculate_212_metrics(patterns_df)

        # Print summary
        print(f"\n  RESULTS:")
        print(f"    Total 2-1-2: {metrics['total_212']}")
        print(f"    Up: {metrics['212_up']}, Down: {metrics['212_down']}")
        if metrics['total_212'] > 0:
            print(f"    Win Rate: {metrics['win_rate']:.1%}")
            print(f"    P/L Expectancy: {metrics['pl_expectancy']:.2%}")
            print(f"    R:R Ratio: {metrics['rr_ratio']:.2f}")

        return {
            'config': config_name,
            'timeframe': timeframe,
            **metrics
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config': config_name,
            'timeframe': timeframe,
            **calculate_212_metrics(pd.DataFrame())
        }


def main():
    """
    Main execution: Test all 8 filter configs across 3 timeframes.
    """
    print("=" * 80)
    print("2-1-2 PATTERN FILTER SENSITIVITY ANALYSIS - SESSION 69")
    print("=" * 80)
    print("\nObjective: Compare 2-1-2 filter sensitivity to 3-1-2 (Session 68)")
    print("\nTest Parameters:")

    # Same dataset as Session 67/68 (for comparison)
    symbols = ['AAPL', 'MSFT', 'AMD', 'AMZN', 'TSLA', 'JPM', 'GS',
               'UNH', 'JNJ', 'XOM', 'CAT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    timeframes = ['1D', '1W', '1M']  # Skip 1H (Alpaca 401 error)

    print(f"  Symbols: {len(symbols)} stocks")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Configurations: {len(FILTER_CONFIGS)}")
    print(f"  Total Tests: {len(FILTER_CONFIGS) * len(timeframes)}")

    # Collect all results
    all_results = []

    # Test each configuration
    for config_name, filter_config in FILTER_CONFIGS.items():
        for timeframe in timeframes:
            result = test_single_config(
                config_name=config_name,
                filter_config=filter_config,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            all_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save to CSV
    output_file = 'scripts/212_filter_analysis_matrix.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {output_file}")
    print(f"{'='*80}")

    # Print summary tables
    print("\n" + "="*80)
    print("SUMMARY: 2-1-2 Pattern Counts by Configuration")
    print("="*80)

    pivot_counts = results_df.pivot(index='config', columns='timeframe', values='total_212')
    print("\nTotal 2-1-2 Patterns:")
    print(pivot_counts.to_string())

    pivot_winrate = results_df.pivot(index='config', columns='timeframe', values='win_rate')
    print("\nWin Rate (%):")
    print((pivot_winrate * 100).round(1).to_string())

    pivot_expectancy = results_df.pivot(index='config', columns='timeframe', values='pl_expectancy')
    print("\nP/L Expectancy (%):")
    print((pivot_expectancy * 100).round(2).to_string())

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return results_df


if __name__ == "__main__":
    main()
