r"""
Manual Verification Script: ATLAS Integration with March 2020 CRASH Veto

Purpose:
    Verify that ATLAS CRASH regime correctly vetoes bullish STRAT patterns.
    Focus on March 2020 crash period where ATLAS detected 77% CRASH regime.

Verification:
    - Export CSV showing date, regime, pattern, signal quality, trade decision
    - Verify 0 bullish trades during CRASH days (veto power working)
    - Manually inspect signal quality assignments

Reference:
    docs/SYSTEM_ARCHITECTURE/INTEGRATION_ARCHITECTURE.md lines 142-164
    Historical validation: March 2020 ATLAS detected 17 CRASH days (77%)
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt

# Import STRAT components
from strat.bar_classifier import classify_bars
from strat.pattern_detector import detect_patterns
from strat.atlas_integration import (
    filter_strat_signals,
    get_position_size_multiplier,
    combine_pattern_signals
)

# Import ATLAS components
from regime.academic_jump_model import AcademicJumpModel


def load_march_2020_data():
    """Load SPY data for March 2020 crash period (expanded for context)."""
    print("Loading SPY data for March 2020 crash period...")

    # Load wider period to have enough lookback for ATLAS
    spy_data = vbt.YFData.pull('SPY', start='2015-01-01', end='2020-04-30', timeframe='1d')
    data = spy_data.get()

    print(f"  Loaded {len(data)} bars (2015-01-01 to 2020-04-30)")
    print(f"  March 2020 period: 2020-03-01 to 2020-03-31")

    return data


def run_verification(data):
    """Run ATLAS + STRAT integration and export verification CSV."""
    print("\n" + "=" * 80)
    print("ATLAS INTEGRATION VERIFICATION - MARCH 2020 CRASH")
    print("=" * 80)

    # Run bar classification
    print("\nStep 1: Running STRAT bar classification...")
    high = data['High'].values
    low = data['Low'].values
    classifications = classify_bars(high, low)
    print(f"  Classified {len(classifications)} bars")

    # Run pattern detection
    print("\nStep 2: Running STRAT pattern detection...")
    result = detect_patterns(classifications, high, low)

    # Unpack results
    entries_312 = result.entries_312.values
    directions_312 = result.directions_312.values
    entries_212 = result.entries_212.values
    directions_212 = result.directions_212.values

    # Combine patterns
    pattern_entries = np.logical_or(entries_312, entries_212)
    pattern_directions = combine_pattern_signals(
        entries_312, directions_312,
        entries_212, directions_212
    )

    num_patterns = np.sum(pattern_entries)
    print(f"  Detected {num_patterns} patterns")

    # Run ATLAS regime detection
    print("\nStep 3: Running ATLAS regime detection...")
    atlas = AcademicJumpModel()

    # Ensure column names match ATLAS expectations
    atlas_data = data.copy()
    if 'close' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'close': 'Close'})
    if 'high' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'high': 'High'})
    if 'low' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'low': 'Low'})
    if 'volume' in atlas_data.columns:
        atlas_data = atlas_data.rename(columns={'volume': 'Volume'})

    # Run online inference
    atlas_regimes, _, _ = atlas.online_inference(
        atlas_data,
        lookback=1000,
        default_lambda=10.0
    )

    # Align with data
    atlas_regimes_aligned = atlas_regimes.reindex(data.index).fillna('TREND_NEUTRAL')

    # Count regimes in March 2020
    march_2020_mask = (data.index >= '2020-03-01') & (data.index <= '2020-03-31')
    march_regimes = atlas_regimes_aligned[march_2020_mask]
    regime_counts = march_regimes.value_counts()

    print(f"  March 2020 regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(march_regimes) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    # Apply integration filtering
    print("\nStep 4: Applying ATLAS-STRAT integration filtering...")
    pattern_directions_series = pd.Series(pattern_directions, index=data.index)
    signal_qualities = filter_strat_signals(atlas_regimes_aligned, pattern_directions_series)
    position_multipliers = get_position_size_multiplier(signal_qualities)

    # Create verification DataFrame
    print("\nStep 5: Creating verification CSV...")
    verification_df = pd.DataFrame({
        'Close': data['Close'].values,
        'Bar_Type': classifications,
        'Pattern_Entry': pattern_entries,
        'Pattern_Direction': pattern_directions,
        'Pattern_Name': [''] * len(data),
        'ATLAS_Regime': atlas_regimes_aligned.values,
        'Signal_Quality': signal_qualities.values,
        'Position_Multiplier': position_multipliers.values,
        'Trade_Taken': (pattern_entries & (position_multipliers.values > 0))
    }, index=data.index)

    # Label pattern names
    def get_pattern_name(entry_312, entry_212, direction):
        if not (entry_312 or entry_212):
            return ''
        if entry_312:
            return '3-1-2 Bullish' if direction > 0 else '3-1-2 Bearish'
        else:
            return '2-1-2 Bullish' if direction > 0 else '2-1-2 Bearish'

    for idx in range(len(verification_df)):
        e312 = entries_312[idx]
        e212 = entries_212[idx]
        direction = pattern_directions[idx]
        verification_df.loc[verification_df.index[idx], 'Pattern_Name'] = get_pattern_name(e312, e212, direction)

    # Filter to March 2020 only
    march_verification = verification_df[march_2020_mask].copy()

    # Export full March 2020 period
    output_file = 'atlas_integration_verification_march2020.csv'
    march_verification.to_csv(output_file, index=False)
    print(f"  Exported: {output_file} ({len(march_verification)} days)")

    # Analysis
    print("\n" + "=" * 80)
    print("VERIFICATION ANALYSIS")
    print("=" * 80)

    # Count patterns in March 2020
    march_patterns = march_verification[march_verification['Pattern_Entry'] == True]
    print(f"\nPatterns detected in March 2020: {len(march_patterns)}")
    if len(march_patterns) > 0:
        print(f"  Bullish patterns: {len(march_patterns[march_patterns['Pattern_Direction'] > 0])}")
        print(f"  Bearish patterns: {len(march_patterns[march_patterns['Pattern_Direction'] < 0])}")

    # Check CRASH veto
    crash_days = march_verification[march_verification['ATLAS_Regime'] == 'CRASH']
    print(f"\nCRASH days in March 2020: {len(crash_days)}")

    if len(crash_days) > 0:
        crash_bullish_patterns = crash_days[
            (crash_days['Pattern_Entry'] == True) &
            (crash_days['Pattern_Direction'] > 0)
        ]
        crash_bullish_trades = crash_days[
            (crash_days['Pattern_Entry'] == True) &
            (crash_days['Pattern_Direction'] > 0) &
            (crash_days['Trade_Taken'] == True)
        ]

        print(f"\nCRASH VETO VERIFICATION:")
        print(f"  Bullish patterns on CRASH days: {len(crash_bullish_patterns)}")
        print(f"  Bullish trades taken on CRASH days: {len(crash_bullish_trades)}")

        if len(crash_bullish_patterns) > 0 and len(crash_bullish_trades) == 0:
            print(f"  [PASS] CRASH veto working: {len(crash_bullish_patterns)} bullish patterns rejected")
        elif len(crash_bullish_patterns) == 0:
            print(f"  [INFO] No bullish patterns detected during CRASH days")
        else:
            print(f"  [FAIL] CRASH veto failed: {len(crash_bullish_trades)} bullish trades taken!")

    # Signal quality distribution
    march_patterns_with_quality = march_verification[march_verification['Pattern_Entry'] == True]
    if len(march_patterns_with_quality) > 0:
        quality_counts = march_patterns_with_quality['Signal_Quality'].value_counts()
        print(f"\nSignal quality distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} patterns")

    # Trades taken
    march_trades = march_verification[march_verification['Trade_Taken'] == True]
    print(f"\nTrades taken in March 2020: {len(march_trades)}")
    if len(march_trades) > 0:
        print(f"  Bullish trades: {len(march_trades[march_trades['Pattern_Direction'] > 0])}")
        print(f"  Bearish trades: {len(march_trades[march_trades['Pattern_Direction'] < 0])}")

    print("=" * 80)
    print(f"\nVerification complete! Review {output_file} for detailed analysis.")
    print("=" * 80)


def main():
    """Run March 2020 verification."""
    data = load_march_2020_data()
    run_verification(data)


if __name__ == '__main__':
    main()
