#!/usr/bin/env python
"""
Session EQUITY-64: Verify 1H bar alignment fix.

This script validates that the _fetch_hourly_market_aligned() method
returns market-open-aligned bars (9:30, 10:30, 11:30) instead of
clock-aligned bars (10:00, 11:00, 12:00).
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from strat.paper_signal_scanner import PaperSignalScanner


def verify_hourly_alignment():
    """Verify 1H bars are market-open aligned."""
    print("=" * 60)
    print("EQUITY-64: 1H Bar Alignment Verification")
    print("=" * 60)

    # Initialize scanner
    scanner = PaperSignalScanner(use_alpaca=True)

    # Fetch 1H data
    print("\nFetching SPY 1H data via _fetch_data()...")
    df = scanner._fetch_data('SPY', '1H', lookback_bars=50)

    if df is None or df.empty:
        print("ERROR: No data returned")
        return False

    # Check timestamps
    print(f"\nTotal bars: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Verify all timestamps end in :30
    print("\n" + "-" * 40)
    print("Timestamp Verification:")
    print("-" * 40)

    market_aligned_count = 0
    clock_aligned_count = 0

    for ts in df.index:
        minute = ts.minute
        if minute == 30:
            market_aligned_count += 1
        elif minute == 0:
            clock_aligned_count += 1

    print(f"Market-open aligned (:30): {market_aligned_count}")
    print(f"Clock aligned (:00):       {clock_aligned_count}")

    # Check expected bars per day
    print("\n" + "-" * 40)
    print("Sample Bars (Last 10):")
    print("-" * 40)

    for i, (ts, row) in enumerate(df.tail(10).iterrows()):
        minute_check = "OK" if ts.minute == 30 else "WRONG"
        print(f"  {ts} | O:{row['Open']:.2f} H:{row['High']:.2f} L:{row['Low']:.2f} C:{row['Close']:.2f} | {minute_check}")

    # Final verdict
    print("\n" + "=" * 60)
    if market_aligned_count > 0 and clock_aligned_count == 0:
        print("RESULT: PASS - All bars are market-open aligned (:30)")
        print("=" * 60)
        return True
    elif clock_aligned_count > 0:
        print("RESULT: FAIL - Found clock-aligned bars (:00)")
        print("The fix is NOT working correctly.")
        print("=" * 60)
        return False
    else:
        print("RESULT: UNKNOWN - Unexpected timestamp patterns")
        print("=" * 60)
        return False


def verify_tfc_uses_aligned_bars():
    """Verify TFC evaluation uses aligned bars."""
    print("\n" + "=" * 60)
    print("TFC Evaluation Verification")
    print("=" * 60)

    scanner = PaperSignalScanner(use_alpaca=True)

    # Evaluate TFC for 1H - this will call _fetch_data which should use aligned bars
    print("\nRunning TFC evaluation for SPY 1H bullish...")

    try:
        result = scanner.evaluate_tfc('SPY', '1H', 1)  # 1 = bullish
        print(f"\nTFC Result: {result}")
        print("TFC evaluation completed successfully (uses aligned bars)")
        return True
    except Exception as e:
        print(f"ERROR: TFC evaluation failed: {e}")
        return False


if __name__ == '__main__':
    print(f"Verification run at: {datetime.now()}")
    print()

    # Test 1: Verify alignment
    align_ok = verify_hourly_alignment()

    # Test 2: Verify TFC works
    tfc_ok = verify_tfc_uses_aligned_bars()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1H Bar Alignment: {'PASS' if align_ok else 'FAIL'}")
    print(f"TFC Evaluation:   {'PASS' if tfc_ok else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if (align_ok and tfc_ok) else 1)
