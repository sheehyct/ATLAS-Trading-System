"""
Manual Verification Script for STRAT Pattern Detection

Creates synthetic patterns with known entry/stop/target values and exports to CSV
for manual inspection.

Usage:
    python verify_pattern_detection.py

Output:
    - pattern_detection_verification_312.csv (3-1-2 patterns)
    - pattern_detection_verification_212.csv (2-1-2 patterns)
"""

import numpy as np
import pandas as pd
from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import detect_all_patterns_nb


def create_312_bullish_pattern():
    """
    Create synthetic 3-1-2 bullish pattern.

    Pattern:
        Bar 0: Reference (H=100, L=95)
        Bar 1: Outside (H=110, L=90, range=20)
        Bar 2: Inside (H=105, L=95)
        Bar 3: 2U directional (H=112, L=96) - TRIGGER

    Expected:
        Stop: 90.0 (outside bar low)
        Target: 125.0 (105 + 20)
        Direction: Bullish (1)
    """
    high = np.array([100.0, 110.0, 105.0, 112.0, 115.0])
    low = np.array([95.0, 90.0, 95.0, 96.0, 97.0])

    return high, low, "3-1-2 Bullish"


def create_212_bearish_pattern():
    """
    Create synthetic 2-1-2 bearish pattern.

    Pattern:
        Bar 0: Reference (H=100, L=95)
        Bar 1: 2D directional (H=99, L=90, range=9)
        Bar 2: Inside (H=98, L=91)
        Bar 3: 2D directional (H=97, L=88) - TRIGGER

    Expected:
        Stop: 98.0 (inside bar high)
        Target: 82.0 (91 - 9)
        Direction: Bearish (-1)
    """
    high = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
    low = np.array([95.0, 90.0, 91.0, 88.0, 87.0])

    return high, low, "2-1-2 Bearish"


def verify_pattern_detection():
    """Run pattern detection on synthetic data and export to CSV."""

    print("=" * 70)
    print("STRAT Pattern Detection Verification")
    print("=" * 70)

    # Test 1: 3-1-2 Bullish Pattern
    print("\n1. Testing 3-1-2 Bullish Pattern...")
    high1, low1, name1 = create_312_bullish_pattern()

    classifications1 = classify_bars_nb(high1, low1)
    (entries_312_1, stops_312_1, targets_312_1, directions_312_1,
     entries_212_1, stops_212_1, targets_212_1, directions_212_1) = detect_all_patterns_nb(
        classifications1, high1, low1
    )

    # Create DataFrame
    df1 = pd.DataFrame({
        'Bar': range(len(high1)),
        'High': high1,
        'Low': low1,
        'Classification': classifications1,
        '312_Entry': entries_312_1,
        '312_Stop': stops_312_1,
        '312_Target': targets_312_1,
        '312_Direction': directions_312_1,
        '212_Entry': entries_212_1,
        '212_Stop': stops_212_1,
        '212_Target': targets_212_1,
        '212_Direction': directions_212_1
    })

    # Add expected values for manual verification
    df1['Expected_312_Stop'] = [np.nan, np.nan, np.nan, 90.0, np.nan]
    df1['Expected_312_Target'] = [np.nan, np.nan, np.nan, 125.0, np.nan]
    df1['Expected_312_Dir'] = [0, 0, 0, 1, 0]

    print(f"  Detected {np.sum(entries_312_1)} 3-1-2 patterns")
    if np.any(entries_312_1):
        idx = np.where(entries_312_1)[0][0]
        print(f"  Pattern at index {idx}:")
        print(f"    Stop: {stops_312_1[idx]} (expected: 90.0)")
        print(f"    Target: {targets_312_1[idx]} (expected: 125.0)")
        print(f"    Direction: {directions_312_1[idx]} (expected: 1)")

        match = (abs(stops_312_1[idx] - 90.0) < 0.01 and
                 abs(targets_312_1[idx] - 125.0) < 0.01 and
                 directions_312_1[idx] == 1)
        print(f"    Verification: {'PASS' if match else 'FAIL'}")

    # Save to CSV
    csv_file_1 = 'pattern_detection_verification_312.csv'
    df1.to_csv(csv_file_1, index=False, float_format='%.2f')
    print(f"  Exported to {csv_file_1}")

    # Test 2: 2-1-2 Bearish Pattern
    print("\n2. Testing 2-1-2 Bearish Pattern...")
    high2, low2, name2 = create_212_bearish_pattern()

    classifications2 = classify_bars_nb(high2, low2)
    (entries_312_2, stops_312_2, targets_312_2, directions_312_2,
     entries_212_2, stops_212_2, targets_212_2, directions_212_2) = detect_all_patterns_nb(
        classifications2, high2, low2
    )

    # Create DataFrame
    df2 = pd.DataFrame({
        'Bar': range(len(high2)),
        'High': high2,
        'Low': low2,
        'Classification': classifications2,
        '312_Entry': entries_312_2,
        '312_Stop': stops_312_2,
        '312_Target': targets_312_2,
        '312_Direction': directions_312_2,
        '212_Entry': entries_212_2,
        '212_Stop': stops_212_2,
        '212_Target': targets_212_2,
        '212_Direction': directions_212_2
    })

    # Add expected values
    df2['Expected_212_Stop'] = [np.nan, np.nan, np.nan, 98.0, np.nan]
    df2['Expected_212_Target'] = [np.nan, np.nan, np.nan, 82.0, np.nan]
    df2['Expected_212_Dir'] = [0, 0, 0, -1, 0]

    print(f"  Detected {np.sum(entries_212_2)} 2-1-2 patterns")
    if np.any(entries_212_2):
        idx = np.where(entries_212_2)[0][0]
        print(f"  Pattern at index {idx}:")
        print(f"    Stop: {stops_212_2[idx]} (expected: 98.0)")
        print(f"    Target: {targets_212_2[idx]} (expected: 82.0)")
        print(f"    Direction: {directions_212_2[idx]} (expected: -1)")

        match = (abs(stops_212_2[idx] - 98.0) < 0.01 and
                 abs(targets_212_2[idx] - 82.0) < 0.01 and
                 directions_212_2[idx] == -1)
        print(f"    Verification: {'PASS' if match else 'FAIL'}")

    # Save to CSV
    csv_file_2 = 'pattern_detection_verification_212.csv'
    df2.to_csv(csv_file_2, index=False, float_format='%.2f')
    print(f"  Exported to {csv_file_2}")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {csv_file_1}")
    print(f"  2. {csv_file_2}")
    print("\nManual verification:")
    print("  1. Open CSV files in Excel or text editor")
    print("  2. Verify stop/target prices match expected values")
    print("  3. Verify direction matches expected (1=bullish, -1=bearish)")
    print("  4. Check that NaN values appear where no pattern exists")
    print("\nExpected Results:")
    print("  - 3-1-2 pattern at index 3: Stop=90, Target=125, Dir=1")
    print("  - 2-1-2 pattern at index 3: Stop=98, Target=82, Dir=-1")
    print("=" * 70)


if __name__ == "__main__":
    verify_pattern_detection()
