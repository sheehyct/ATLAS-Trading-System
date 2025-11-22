"""
Quick 3-stock multi-timeframe test validation.

Tests AAPL, MSFT, GOOGL for 1 year (2024-2025) on all 4 timeframes (1H, 1D, 1W, 1M)
to establish empirical magnitude hit rates before running full 50-stock validation.
"""

import sys
sys.path.insert(0, 'C:\\Strat_Trading_Bot\\vectorbt-workspace')

from scripts.backtest_strat_equity_validation import EquityValidationBacktest

# Create validator with 3-stock test configuration
test_config = {
    'backtest_period': {
        'start': '2024-01-01',  # 1 year for speed
        'end': '2025-01-01',
        'rationale': '1 year test for quick validation'
    },
    'stock_universe': {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],  # 3 liquid tech stocks
        'count': 3,
        'rationale': '3-stock test before full 50-stock validation'
    },
    'timeframes': {
        'base': '1H',  # Base data download
        'detection': ['1H', '1D', '1W', '1M'],  # Timeframes to detect patterns on
        'continuity_check': ['1M', '1W', '1D', '4H', '1H'],
        'rationale': 'Multi-timeframe pattern detection + alignment test'
    },
    'pattern_types': ['3-1-2 Up', '3-1-2 Down', '2-1-2 Up', '2-1-2 Down'],
    'filters': {
        'require_full_continuity': False,  # Changed to flexible (Session 56 Bug #3 fix)
        'use_flexible_continuity': True,   # Use timeframe-appropriate continuity
        'min_continuity_strength': 3,      # Require 3/5 timeframes aligned minimum
        'use_atlas_regime': False,  # Test patterns alone first
        'min_pattern_quality': 'HIGH'
    },
    'metrics': {
        'magnitude_window': 5,  # Bars to check for magnitude hit
        'max_holding_bars': 30,  # Maximum bars before exit
        'magnitude_hit_threshold': 0.50,  # 50% realistic per research
        'risk_reward_threshold': 2.0,
        'min_pattern_count': 10  # Lower for 3-stock test
    }
}

print("=" * 80)
print("3-STOCK MULTI-TIMEFRAME TEST VALIDATION")
print("=" * 80)
print(f"Stocks: {test_config['stock_universe']['symbols']}")
print(f"Period: {test_config['backtest_period']['start']} to {test_config['backtest_period']['end']}")
print(f"Detection timeframes: {test_config['timeframes']['detection']}")
print(f"Continuity timeframes: {test_config['timeframes']['continuity_check']}")
print(f"Pattern types: {test_config['pattern_types']}")
print(f"Flexible continuity: {test_config['filters']['use_flexible_continuity']} (min {test_config['filters']['min_continuity_strength']}/5 TFs)")
print(f"Max holding bars: {test_config['metrics']['max_holding_bars']}")
print("=" * 80)
print()

# Run validation
try:
    validator = EquityValidationBacktest(test_config)
    all_results = validator.run_validation()

    print("\n" + "=" * 80)
    print("EMPIRICAL HIT RATES BY DETECTION TIMEFRAME")
    print("=" * 80)

    for timeframe, results_df in all_results.items():
        if len(results_df) == 0:
            print(f"\n{timeframe}: NO PATTERNS FOUND")
            continue

        total = len(results_df)
        hits = results_df['magnitude_hit'].sum()
        hit_rate = hits / total if total > 0 else 0

        print(f"\n{timeframe}:")
        print(f"  Total patterns: {total}")
        print(f"  Magnitude hits: {hits} ({hit_rate:.1%})")

        # Breakdown by pattern type
        for pattern_type in test_config['pattern_types']:
            subset = results_df[results_df['pattern_type'] == pattern_type]
            if len(subset) > 0:
                subset_hits = subset['magnitude_hit'].sum()
                subset_rate = subset_hits / len(subset)
                print(f"    {pattern_type}: {subset_hits}/{len(subset)} ({subset_rate:.1%})")

    print("\n" + "=" * 80)
    print("SUCCESS: Multi-timeframe pattern detection working!")
    print("Ready to run full 50-stock validation.")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
