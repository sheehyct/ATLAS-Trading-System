"""
Manual verification script for bar classification.

Creates CSV export of synthetic and real data classifications for manual review.
Run this script to verify bar classifications are correct before proceeding to Phase 2.

Usage:
    uv run python verify_bar_classification.py
"""

import numpy as np
import pandas as pd
from strat.bar_classifier import StratBarClassifier

def verify_synthetic_data():
    """Verify classification with known synthetic sequences."""
    print("=" * 80)
    print("SYNTHETIC DATA VERIFICATION")
    print("=" * 80)

    # Test 1: Known 5-bar sequence
    print("\n1. Known 5-Bar Sequence (Ref, 2U, 1, 2U, 3)")
    print("-" * 80)

    test_data = pd.DataFrame({
        'high': [100, 105, 104, 107, 110],
        'low': [95, 98, 99, 101, 93],
    })

    result = StratBarClassifier.run(test_data['high'], test_data['low'])

    verification = pd.DataFrame({
        'high': test_data['high'],
        'low': test_data['low'],
        'classification': result.classification,
        'expected': [-999, 2, 1, 2, 3],
        'bar_type': ['Ref', '2U', '1', '2U', '3'],
        'reason': [
            'Reference bar (first bar)',
            'Breaks high (105 > 100) -> 2U, gov_range: 105/95',
            'Inside (104 <= 105 and 99 >= 95) -> 1, gov_range: unchanged',
            'Breaks high (107 > 105) -> 2U, gov_range: 107/95',
            'Breaks both (110 > 107 and 93 < 95) -> 3, gov_range: 110/93'
        ]
    })

    print(verification.to_string(index=True))

    match = (verification['classification'] == verification['expected']).all()
    print(f"\nVerification: {'PASS' if match else 'FAIL'}")

    # Test 2: Governing range persistence
    print("\n\n2. Governing Range Persistence (3 Inside Bars, Then 2U)")
    print("-" * 80)

    test_data2 = pd.DataFrame({
        'high': [100, 99, 98, 97, 105],
        'low': [90, 91, 92, 93, 91],
    })

    result2 = StratBarClassifier.run(test_data2['high'], test_data2['low'])

    verification2 = pd.DataFrame({
        'high': test_data2['high'],
        'low': test_data2['low'],
        'classification': result2.classification,
        'expected': [-999, 1, 1, 1, 2],
        'bar_type': ['Ref', '1', '1', '1', '2U'],
        'reason': [
            'Reference bar, gov_range: 100/90',
            'Inside (99 <= 100, 91 >= 90), gov_range: unchanged',
            'Inside (98 <= 100, 92 >= 90), gov_range: unchanged',
            'Inside (97 <= 100, 93 >= 90), gov_range: unchanged',
            'Breaks high (105 > 100), gov_range: 105/91'
        ]
    })

    print(verification2.to_string(index=True))

    match2 = (verification2['classification'] == verification2['expected']).all()
    print(f"\nVerification: {'PASS' if match2 else 'FAIL'}")

    # Export to CSV
    verification.to_csv('bar_classification_verification_5bar.csv', index=True)
    verification2.to_csv('bar_classification_verification_persistence.csv', index=True)

    print("\n\nCSV files exported:")
    print("  - bar_classification_verification_5bar.csv")
    print("  - bar_classification_verification_persistence.csv")


def verify_random_data():
    """Verify classification with random data (visual inspection)."""
    print("\n\n" + "=" * 80)
    print("RANDOM DATA VERIFICATION (First 20 Bars)")
    print("=" * 80)

    np.random.seed(42)
    high = 100 + np.cumsum(np.random.randn(100) * 2)
    low = high - np.abs(np.random.randn(100) * 5)

    result = StratBarClassifier.run(high, low)

    verification = pd.DataFrame({
        'high': high[:20],
        'low': low[:20],
        'classification': result.classification.values[:20],
        'bar_type': result.classification.values[:20].astype(int).astype(str)
    })

    # Map classifications to readable names
    type_map = {'-999': 'Ref', '1': 'Inside', '2': '2U', '-2': '2D', '3': 'Outside'}
    verification['bar_type'] = verification['bar_type'].map(type_map)

    print(verification.to_string(index=True))

    # Distribution check
    value_counts = pd.Series(result.classification.values).value_counts()
    print(f"\n\nDistribution across all {len(high)} bars:")
    print(f"  Reference: {value_counts.get(-999, 0)}")
    print(f"  Inside (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0) / len(high) * 100:.1f}%)")
    print(f"  2U (2): {value_counts.get(2, 0)} ({value_counts.get(2, 0) / len(high) * 100:.1f}%)")
    print(f"  2D (-2): {value_counts.get(-2, 0)} ({value_counts.get(-2, 0) / len(high) * 100:.1f}%)")
    print(f"  Outside (3): {value_counts.get(3, 0)} ({value_counts.get(3, 0) / len(high) * 100:.1f}%)")

    # Export full dataset
    full_verification = pd.DataFrame({
        'high': high,
        'low': low,
        'classification': result.classification.values,
    })
    full_verification.to_csv('bar_classification_verification_random.csv', index=True)

    print("\nCSV file exported: bar_classification_verification_random.csv")


def performance_check():
    """Check performance with large dataset."""
    print("\n\n" + "=" * 80)
    print("PERFORMANCE CHECK")
    print("=" * 80)

    import time

    for n_bars in [1000, 10000]:
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(n_bars) * 2)
        low = high - np.abs(np.random.randn(n_bars) * 5)

        start = time.time()
        result = StratBarClassifier.run(high, low)
        elapsed = time.time() - start

        print(f"{n_bars:,} bars: {elapsed:.4f}s ({n_bars/elapsed:,.0f} bars/second)")


if __name__ == '__main__':
    verify_synthetic_data()
    verify_random_data()
    performance_check()

    print("\n\n" + "=" * 80)
    print("MANUAL VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Open the exported CSV files in Excel/editor")
    print("2. Manually verify first 10 bars match expected classifications")
    print("3. Check that governing range logic is correct")
    print("4. Verify no NaN/inf values in output")
    print("\nIf all checks pass, Phase 1 is COMPLETE!")
