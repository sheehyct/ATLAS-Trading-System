"""
Tests for STRAT pattern detection VBT custom indicator.

Test Coverage:
    - Synthetic pattern tests (known 3-1-2 and 2-1-2 patterns)
    - Edge case tests (insufficient bars, no patterns, all inside bars)
    - VBT integration tests (indicator factory, multi-output)
    - Performance benchmarks (>100k bars/sec target)

All tests use hand-calculated expected values for entry/stop/target prices.
"""

import pytest
import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import (
    detect_312_patterns_nb,
    detect_212_patterns_nb,
    detect_all_patterns_nb,
    StratPatternDetector
)


class TestSynthetic312Patterns:
    """Test 3-1-2 pattern detection with known synthetic patterns."""

    def test_312_bullish_pattern(self):
        """
        Test 3-1-2 bullish pattern with hand-calculated expected values.

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: Outside (H=110, L=90, range=20)
            Bar 2: Inside (H=105, L=95)
            Bar 3: 2U directional (H=112, L=96) - TRIGGER

        Expected:
            Entry: Index 3
            Stop: 90.0 (outside bar low)
            Target: 105.0 + 20.0 = 125.0 (trigger + range)
            Direction: 1 (bullish)
        """
        high = np.array([100.0, 110.0, 105.0, 112.0])
        low = np.array([95.0, 90.0, 95.0, 96.0])

        # Step 1: Classify bars
        classifications = classify_bars_nb(high, low)
        assert classifications[0] == -999, "Bar 0 should be reference"
        assert classifications[1] == 3, "Bar 1 should be outside"
        assert classifications[2] == 1, "Bar 2 should be inside"
        assert classifications[3] == 2, "Bar 3 should be 2U"

        # Step 2: Detect 3-1-2 patterns
        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications, high, low
        )

        # Verify pattern detected at index 3
        assert entries[3] == True, "Entry should be True at index 3"
        assert abs(stops[3] - 90.0) < 0.01, f"Stop should be 90.0, got {stops[3]}"
        assert abs(targets[3] - 125.0) < 0.01, f"Target should be 125.0, got {targets[3]}"
        assert directions[3] == 1, f"Direction should be 1 (bullish), got {directions[3]}"

        # Verify no patterns at other indices
        assert entries[0] == False
        assert entries[1] == False
        assert entries[2] == False

    def test_312_bearish_pattern(self):
        """
        Test 3-1-2 bearish pattern.

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: Outside (H=110, L=90, range=20)
            Bar 2: Inside (H=105, L=95)
            Bar 3: 2D directional (H=103, L=88) - TRIGGER

        Expected:
            Entry: Index 3
            Stop: 110.0 (outside bar high)
            Target: 95.0 - 20.0 = 75.0 (trigger - range)
            Direction: -1 (bearish)
        """
        high = np.array([100.0, 110.0, 105.0, 103.0])
        low = np.array([95.0, 90.0, 95.0, 88.0])

        classifications = classify_bars_nb(high, low)
        assert classifications[3] == -2, "Bar 3 should be 2D"

        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications, high, low
        )

        # Verify pattern detected at index 3
        assert entries[3] == True
        assert abs(stops[3] - 110.0) < 0.01, f"Stop should be 110.0, got {stops[3]}"
        assert abs(targets[3] - 75.0) < 0.01, f"Target should be 75.0, got {targets[3]}"
        assert directions[3] == -1, f"Direction should be -1 (bearish), got {directions[3]}"


class TestSynthetic212Patterns:
    """Test 2-1-2 pattern detection with known synthetic patterns."""

    def test_212_bullish_continuation(self):
        """
        Test 2-1-2 bullish continuation (2U-1-2U).

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: 2U directional (H=105, L=96, range=9)
            Bar 2: Inside (H=104, L=97)
            Bar 3: 2U directional (H=108, L=98) - TRIGGER

        Expected:
            Entry: Index 3
            Stop: 97.0 (inside bar low)
            Target: 104.0 + 9.0 = 113.0
            Direction: 1 (bullish)
        """
        high = np.array([100.0, 105.0, 104.0, 108.0])
        low = np.array([95.0, 96.0, 97.0, 98.0])

        classifications = classify_bars_nb(high, low)
        assert classifications[1] == 2, "Bar 1 should be 2U"
        assert classifications[2] == 1, "Bar 2 should be inside"
        assert classifications[3] == 2, "Bar 3 should be 2U"

        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, high, low
        )

        assert entries[3] == True
        assert abs(stops[3] - 97.0) < 0.01, f"Stop should be 97.0, got {stops[3]}"
        assert abs(targets[3] - 113.0) < 0.01, f"Target should be 113.0, got {targets[3]}"
        assert directions[3] == 1

    def test_212_bearish_continuation(self):
        """
        Test 2-1-2 bearish continuation (2D-1-2D).

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: 2D directional (H=99, L=90, range=9)
            Bar 2: Inside (H=98, L=91)
            Bar 3: 2D directional (H=97, L=88) - TRIGGER

        Expected:
            Entry: Index 3
            Stop: 98.0 (inside bar high)
            Target: 91.0 - 9.0 = 82.0
            Direction: -1 (bearish)
        """
        high = np.array([100.0, 99.0, 98.0, 97.0])
        low = np.array([95.0, 90.0, 91.0, 88.0])

        classifications = classify_bars_nb(high, low)
        assert classifications[1] == -2, "Bar 1 should be 2D"
        assert classifications[2] == 1, "Bar 2 should be inside"
        assert classifications[3] == -2, "Bar 3 should be 2D"

        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, high, low
        )

        assert entries[3] == True
        assert abs(stops[3] - 98.0) < 0.01
        assert abs(targets[3] - 82.0) < 0.01
        assert directions[3] == -1

    def test_212_bullish_reversal(self):
        """
        Test 2-1-2 bullish reversal (2D-1-2U) - failed breakdown.

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: 2D directional (H=99, L=90, range=9)
            Bar 2: Inside (H=98, L=91)
            Bar 3: 2U directional (H=102, L=92) - TRIGGER (reversal!)

        Expected:
            Entry: Index 3
            Stop: 91.0 (inside bar low)
            Target: 98.0 + 9.0 = 107.0
            Direction: 1 (bullish reversal)
        """
        high = np.array([100.0, 99.0, 98.0, 102.0])
        low = np.array([95.0, 90.0, 91.0, 92.0])

        classifications = classify_bars_nb(high, low)
        assert classifications[1] == -2, "Bar 1 should be 2D"
        assert classifications[2] == 1, "Bar 2 should be inside"
        assert classifications[3] == 2, "Bar 3 should be 2U (reversal)"

        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, high, low
        )

        assert entries[3] == True
        assert abs(stops[3] - 91.0) < 0.01
        assert abs(targets[3] - 107.0) < 0.01
        assert directions[3] == 1  # Bullish reversal

    def test_212_bearish_reversal(self):
        """
        Test 2-1-2 bearish reversal (2U-1-2D) - failed breakout.

        Pattern:
            Bar 0: Reference (H=100, L=95)
            Bar 1: 2U directional (H=105, L=96, range=9)
            Bar 2: Inside (H=104, L=97)
            Bar 3: 2D directional (H=103, L=93) - TRIGGER (reversal!)

        Expected:
            Entry: Index 3
            Stop: 104.0 (inside bar high)
            Target: 97.0 - 9.0 = 88.0
            Direction: -1 (bearish reversal)
        """
        high = np.array([100.0, 105.0, 104.0, 103.0])
        low = np.array([95.0, 96.0, 97.0, 93.0])

        classifications = classify_bars_nb(high, low)
        assert classifications[1] == 2, "Bar 1 should be 2U"
        assert classifications[2] == 1, "Bar 2 should be inside"
        assert classifications[3] == -2, "Bar 3 should be 2D (reversal)"

        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, high, low
        )

        assert entries[3] == True
        assert abs(stops[3] - 104.0) < 0.01
        assert abs(targets[3] - 88.0) < 0.01
        assert directions[3] == -1  # Bearish reversal


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_insufficient_bars_312(self):
        """Test 3-1-2 detection with insufficient bars (needs 3 minimum)."""
        high = np.array([100.0, 105.0])
        low = np.array([95.0, 98.0])

        classifications = classify_bars_nb(high, low)
        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications, high, low
        )

        # No patterns should be detected
        assert np.sum(entries) == 0, "No patterns should be detected with <3 bars"

    def test_insufficient_bars_212(self):
        """Test 2-1-2 detection with insufficient bars (needs 3 minimum)."""
        high = np.array([100.0, 105.0])
        low = np.array([95.0, 98.0])

        classifications = classify_bars_nb(high, low)
        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, high, low
        )

        assert np.sum(entries) == 0

    def test_all_inside_bars(self):
        """Test with all inside bars - no patterns possible."""
        # All bars inside 110/90 range
        high = np.array([100.0, 110.0, 105.0, 104.0, 103.0])
        low = np.array([95.0, 90.0, 95.0, 96.0, 97.0])

        classifications = classify_bars_nb(high, low)
        # Should be: Ref, Outside, Inside, Inside, Inside
        assert classifications[2] == 1
        assert classifications[3] == 1
        assert classifications[4] == 1

        # No 3-1-2 patterns (need directional bar after inside)
        entries_312, _, _, _ = detect_312_patterns_nb(classifications, high, low)
        assert np.sum(entries_312) == 0, "No 3-1-2 patterns with all inside bars"

        # No 2-1-2 patterns (need directional bars)
        entries_212, _, _, _ = detect_212_patterns_nb(classifications, high, low)
        assert np.sum(entries_212) == 0, "No 2-1-2 patterns with all inside bars"

    def test_no_nan_in_entries(self):
        """Verify entry arrays never contain NaN."""
        high = np.array([100.0, 110.0, 105.0, 112.0])
        low = np.array([95.0, 90.0, 95.0, 96.0])

        classifications = classify_bars_nb(high, low)
        entries_312, stops_312, targets_312, _ = detect_312_patterns_nb(
            classifications, high, low
        )

        # Entry array should be bool (no NaN possible)
        assert entries_312.dtype == np.bool_
        assert not np.any(np.isnan(stops_312[entries_312]))  # No NaN where pattern exists
        assert not np.any(np.isnan(targets_312[entries_312]))  # No NaN where pattern exists


class TestVBTIntegration:
    """Test VBT IndicatorFactory integration."""

    def test_vbt_indicator_outputs(self):
        """Test VBT indicator has all 8 expected outputs."""
        high = np.array([100.0, 110.0, 105.0, 112.0])
        low = np.array([95.0, 90.0, 95.0, 96.0])

        classifications = classify_bars_nb(high, low)
        result = StratPatternDetector.run(classifications, high, low)

        # Verify all 8 outputs exist
        assert hasattr(result, 'entries_312')
        assert hasattr(result, 'stops_312')
        assert hasattr(result, 'targets_312')
        assert hasattr(result, 'directions_312')
        assert hasattr(result, 'entries_212')
        assert hasattr(result, 'stops_212')
        assert hasattr(result, 'targets_212')
        assert hasattr(result, 'directions_212')

    def test_vbt_indicator_312_accuracy(self):
        """Test VBT indicator produces correct 3-1-2 values."""
        high = np.array([100.0, 110.0, 105.0, 112.0])
        low = np.array([95.0, 90.0, 95.0, 96.0])

        classifications = classify_bars_nb(high, low)
        result = StratPatternDetector.run(classifications, high, low)

        # Verify 3-1-2 pattern at index 3
        assert result.entries_312.iloc[3] == True
        assert abs(result.stops_312.iloc[3] - 90.0) < 0.01
        assert abs(result.targets_312.iloc[3] - 125.0) < 0.01
        assert result.directions_312.iloc[3] == 1

    def test_vbt_indicator_212_accuracy(self):
        """Test VBT indicator produces correct 2-1-2 values."""
        high = np.array([100.0, 99.0, 98.0, 97.0])
        low = np.array([95.0, 90.0, 91.0, 88.0])

        classifications = classify_bars_nb(high, low)
        result = StratPatternDetector.run(classifications, high, low)

        # Verify 2-1-2 pattern at index 3
        assert result.entries_212.iloc[3] == True
        assert abs(result.stops_212.iloc[3] - 98.0) < 0.01
        assert abs(result.targets_212.iloc[3] - 82.0) < 0.01
        assert result.directions_212.iloc[3] == -1


class TestPerformance:
    """Performance benchmarks for pattern detection."""

    def test_performance_1000_bars(self):
        """Test pattern detection on 1,000 bars (<0.01s target)."""
        np.random.seed(42)
        n = 1000

        # Generate realistic random walk data
        high = 100 + np.cumsum(np.random.randn(n) * 2)
        low = high - np.abs(np.random.randn(n) * 5)

        classifications = classify_bars_nb(high, low)

        start_time = time.time()
        entries_312, stops_312, targets_312, dirs_312 = detect_312_patterns_nb(
            classifications, high, low
        )
        entries_212, stops_212, targets_212, dirs_212 = detect_212_patterns_nb(
            classifications, high, low
        )
        elapsed = time.time() - start_time

        bars_per_sec = n / elapsed if elapsed > 0 else float('inf')

        print(f"\nPerformance (1,000 bars):")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Throughput: {bars_per_sec:,.0f} bars/sec")
        print(f"  3-1-2 patterns detected: {np.sum(entries_312)}")
        print(f"  2-1-2 patterns detected: {np.sum(entries_212)}")

        # Target: >100k bars/sec
        assert bars_per_sec > 100_000, f"Performance too slow: {bars_per_sec:,.0f} bars/sec"
        assert elapsed < 0.01, f"Took {elapsed:.4f}s, target <0.01s"

    def test_performance_10000_bars(self):
        """Test pattern detection on 10,000 bars (<0.1s target)."""
        np.random.seed(42)
        n = 10000

        high = 100 + np.cumsum(np.random.randn(n) * 2)
        low = high - np.abs(np.random.randn(n) * 5)

        classifications = classify_bars_nb(high, low)

        start_time = time.time()
        entries_312, _, _, _ = detect_312_patterns_nb(classifications, high, low)
        entries_212, _, _, _ = detect_212_patterns_nb(classifications, high, low)
        elapsed = time.time() - start_time

        bars_per_sec = n / elapsed if elapsed > 0 else float('inf')

        print(f"\nPerformance (10,000 bars):")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Throughput: {bars_per_sec:,.0f} bars/sec")
        print(f"  3-1-2 patterns detected: {np.sum(entries_312)}")
        print(f"  2-1-2 patterns detected: {np.sum(entries_212)}")

        # Target: >100k bars/sec
        assert bars_per_sec > 100_000, f"Performance too slow: {bars_per_sec:,.0f} bars/sec"
        assert elapsed < 0.1, f"Took {elapsed:.4f}s, target <0.1s"

    def test_numba_compilation(self):
        """Verify @njit Numba compilation is working."""
        # First call compiles
        high = np.array([100.0, 110.0, 105.0, 112.0, 115.0])
        low = np.array([95.0, 90.0, 95.0, 96.0, 97.0])
        classifications = classify_bars_nb(high, low)

        # First run (with JIT compilation)
        start1 = time.time()
        detect_312_patterns_nb(classifications, high, low)
        time1 = time.time() - start1

        # Second run (already compiled)
        start2 = time.time()
        detect_312_patterns_nb(classifications, high, low)
        time2 = time.time() - start2

        print(f"\nNumba compilation check:")
        print(f"  First run (with JIT): {time1:.6f}s")
        print(f"  Second run (compiled): {time2:.6f}s")
        print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: Inf")

        # Second run should be much faster (or immeasurably fast)
        assert time2 <= time1, "Second run should be faster or equal (compiled)"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
