"""
Quick test for 2-2 pattern detection

Tests both 2D-2U (bullish reversal) and 2U-2D (bearish reversal) patterns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from strat.pattern_detector import detect_22_patterns_nb

def test_2d_2u_bullish_reversal():
    """Test 2D-2U bullish reversal pattern (failed breakdown) - MEASURED MOVE FALLBACK"""
    print("TEST 1: 2D-2U Bullish Reversal (Measured Move Fallback)")
    print("=" * 60)

    # Create synthetic data:
    # Bar 0: 2D directional down (H=105, L=95, classification=-2)
    # Bar 1: 2U directional up (H=110, L=100, classification=2) - TRIGGER
    # No previous 2D bar before index 0, so measured move fallback expected

    classifications = np.array([-2.0, 2.0], dtype=np.float64)
    high = np.array([105.0, 110.0], dtype=np.float64)
    low = np.array([95.0, 100.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    # Expected (SESSION 62 UPDATE):
    # - Pattern detected at index 1 (2U bar)
    # - Entry: 105 (bar 0 high, Session 59 correction)
    # - Stop: 95 (2D bar low)
    # - Stop distance: 105 - 95 = 10
    # - Target: 105 + (10 * 1.5) = 120 (measured move fallback)
    # - Direction: 1 (bullish)

    entry_price = high[0]  # Session 59: entry is bar i-1 high
    stop_distance = entry_price - low[0]  # 105 - 95 = 10
    expected_target = entry_price + (stop_distance * 1.5)  # 105 + 15 = 120

    print(f"Pattern detected: {entries[1]}")
    print(f"Entry price: {entry_price:.2f} (bar 0 high)")
    print(f"Stop price: {stops[1]:.2f} (expected: 95.00)")
    print(f"Stop distance: {stop_distance:.2f}")
    print(f"Target price: {targets[1]:.2f} (expected: {expected_target:.2f} - measured move)")
    print(f"Direction: {directions[1]} (expected: 1 for bullish)")
    print()

    # Verify (SESSION 62 expectations)
    assert entries[1] == True, "Pattern should be detected at index 1"
    assert directions[1] == 1, "Direction should be bullish (1)"
    assert abs(stops[1] - 95.0) < 0.01, f"Stop should be 95.00, got {stops[1]}"
    assert abs(targets[1] - expected_target) < 0.01, f"Target should be {expected_target:.2f} (measured move), got {targets[1]}"
    assert targets[1] > entry_price, "Target must be above entry for bullish trade"

    print("PASS - 2D-2U Bullish Reversal (Measured Move)")
    print()


def test_2u_2d_bearish_reversal():
    """Test 2U-2D bearish reversal pattern (failed breakout) - MEASURED MOVE FALLBACK"""
    print("TEST 2: 2U-2D Bearish Reversal (Measured Move Fallback)")
    print("=" * 60)

    # Create synthetic data:
    # Bar 0: 2U directional up (H=110, L=100, classification=2)
    # Bar 1: 2D directional down (H=105, L=95, classification=-2) - TRIGGER
    # No previous 2U bar before index 0, so measured move fallback expected

    classifications = np.array([2.0, -2.0], dtype=np.float64)
    high = np.array([110.0, 105.0], dtype=np.float64)
    low = np.array([100.0, 95.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    # Expected (SESSION 62 UPDATE):
    # - Pattern detected at index 1 (2D bar)
    # - Entry: 100 (bar 0 low, Session 59 correction)
    # - Stop: 110 (2U bar high)
    # - Stop distance: 110 - 100 = 10
    # - Target: 100 - (10 * 1.5) = 85 (measured move fallback)
    # - Direction: -1 (bearish)

    entry_price = low[0]  # Session 59: entry is bar i-1 low
    stop_distance = high[0] - entry_price  # 110 - 100 = 10
    expected_target = entry_price - (stop_distance * 1.5)  # 100 - 15 = 85

    print(f"Pattern detected: {entries[1]}")
    print(f"Entry price: {entry_price:.2f} (bar 0 low)")
    print(f"Stop price: {stops[1]:.2f} (expected: 110.00)")
    print(f"Stop distance: {stop_distance:.2f}")
    print(f"Target price: {targets[1]:.2f} (expected: {expected_target:.2f} - measured move)")
    print(f"Direction: {directions[1]} (expected: -1 for bearish)")
    print()

    # Verify (SESSION 62 expectations)
    assert entries[1] == True, "Pattern should be detected at index 1"
    assert directions[1] == -1, "Direction should be bearish (-1)"
    assert abs(stops[1] - 110.0) < 0.01, f"Stop should be 110.00, got {stops[1]}"
    assert abs(targets[1] - expected_target) < 0.01, f"Target should be {expected_target:.2f} (measured move), got {targets[1]}"
    assert targets[1] < entry_price, "Target must be below entry for bearish trade"

    print("PASS - 2U-2D Bearish Reversal (Measured Move)")
    print()


def test_no_false_positives():
    """Test that non-2-2 patterns are NOT detected"""
    print("TEST 3: No False Positives")
    print("=" * 60)

    # Create data without 2-2 patterns:
    # Bar 0: 2U
    # Bar 1: 2U (continuation, not reversal)
    # Bar 2: 1 (inside bar)

    classifications = np.array([2.0, 2.0, 1.0], dtype=np.float64)
    high = np.array([110.0, 115.0, 112.0], dtype=np.float64)
    low = np.array([100.0, 105.0, 108.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    # Expected: No patterns detected
    print(f"Patterns detected: {entries.sum()}")
    print(f"Expected: 0")
    print()

    assert entries.sum() == 0, "Should not detect any 2-2 patterns"

    print("PASS - No False Positives")
    print()


def test_2d_2d_2u_compound_reversal():
    """Test 2D-2D-2U compound bullish reversal pattern"""
    print("TEST 4: 2D-2D-2U Compound Bullish Reversal (CORRECTED)")
    print("=" * 60)

    # Create synthetic data (User's example):
    # Bar 0: 2D (H=105, L=95, classification=-2) <- Previous 2D
    # Bar 1: 2D (H=100, L=90, classification=-2) <- Part of reversal
    # Bar 2: 2U (H=107, L=96, classification=2) - TRIGGER

    classifications = np.array([-2.0, -2.0, 2.0], dtype=np.float64)
    high = np.array([105.0, 100.0, 107.0], dtype=np.float64)
    low = np.array([95.0, 90.0, 96.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    # Expected (CORRECTED):
    # - Pattern detected at index 2 (2U bar)
    # - Entry: 107 (2U bar high)
    # - Stop: 90 (2D bar i-1 low)
    # - Target: 105 (high of bar 0, previous 2D NOT in reversal)
    # - Direction: 1 (bullish)

    print(f"Pattern detected: {entries[2]}")
    print(f"Entry price: {high[2]:.2f} (expected: 107.00)")
    print(f"Stop price: {stops[2]:.2f} (expected: 90.00)")
    print(f"Target price: {targets[2]:.2f} (expected: 105.00 - previous 2D high)")
    print(f"Direction: {directions[2]} (expected: 1 for bullish)")
    print()

    # Verify
    assert entries[2] == True, "Pattern should be detected at index 2"
    assert directions[2] == 1, "Direction should be bullish (1)"
    assert abs(stops[2] - 90.0) < 0.01, f"Stop should be 90.00, got {stops[2]}"
    assert abs(targets[2] - 105.0) < 0.01, f"Target should be 105.00 (previous 2D high), got {targets[2]}"

    print("PASS - 2D-2D-2U Compound Reversal")
    print()


def test_2u_2u_2d_compound_reversal():
    """Test 2U-2U-2D compound bearish reversal pattern"""
    print("TEST 5: 2U-2U-2D Compound Bearish Reversal (CORRECTED)")
    print("=" * 60)

    # Create synthetic data:
    # Bar 0: 2U (H=115, L=105, classification=2) <- Previous 2U
    # Bar 1: 2U (H=120, L=110, classification=2) <- Part of reversal
    # Bar 2: 2D (H=118, L=108, classification=-2) - TRIGGER

    classifications = np.array([2.0, 2.0, -2.0], dtype=np.float64)
    high = np.array([115.0, 120.0, 118.0], dtype=np.float64)
    low = np.array([105.0, 110.0, 108.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    # Expected (CORRECTED):
    # - Pattern detected at index 2 (2D bar)
    # - Entry: 108 (2D bar low)
    # - Stop: 120 (2U bar i-1 high)
    # - Target: 105 (low of bar 0, previous 2U NOT in reversal)
    # - Direction: -1 (bearish)

    print(f"Pattern detected: {entries[2]}")
    print(f"Entry price: {low[2]:.2f} (expected: 108.00)")
    print(f"Stop price: {stops[2]:.2f} (expected: 120.00)")
    print(f"Target price: {targets[2]:.2f} (expected: 105.00 - previous 2U low)")
    print(f"Direction: {directions[2]} (expected: -1 for bearish)")
    print()

    # Verify
    assert entries[2] == True, "Pattern should be detected at index 2"
    assert directions[2] == -1, "Direction should be bearish (-1)"
    assert abs(stops[2] - 120.0) < 0.01, f"Stop should be 120.00, got {stops[2]}"
    assert abs(targets[2] - 105.0) < 0.01, f"Target should be 105.00 (previous 2U low), got {targets[2]}"

    print("PASS - 2U-2U-2D Compound Reversal")
    print()


def test_valid_geometry_2d_2u():
    """Test 2D-2U compound pattern where previous 2D creates VALID geometry (Session 62)"""
    print("TEST 6: 2D-2U Valid Geometry (STRAT Methodology)")
    print("=" * 60)

    # Bar 0: 2D at $195 (H=195, L=190)
    # Bar 1: 2D at $193 (H=193, L=188)
    # Bar 2: 2U at $197 (H=197, L=191) - TRIGGER

    # Entry: $193 (bar 1 high, Session 59 correction)
    # Stop: $188 (bar 1 low)
    # Previous 2D: bar 0 high = $195
    # Geometry: $195 > $193 entry - VALID!
    # Expected target: $195 (previous 2D high - STRAT methodology)

    classifications = np.array([-2.0, -2.0, 2.0], dtype=np.float64)
    high = np.array([195.0, 193.0, 197.0], dtype=np.float64)
    low = np.array([190.0, 188.0, 191.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    print(f"Entry: {high[1]:.2f} (bar 1 high)")
    print(f"Stop: {stops[2]:.2f} (bar 1 low)")
    print(f"Previous 2D high: {high[0]:.2f}")
    print(f"Target: {targets[2]:.2f} (should be 195.00 - previous 2D high)")
    print(f"Geometry valid: {targets[2] > high[1]}")
    print()

    assert abs(targets[2] - 195.0) < 0.01, f"Target should use previous 2D high (195), got {targets[2]}"
    assert targets[2] > high[1], "Target must be above entry for bullish trade"
    print("PASS - Valid Geometry (STRAT Methodology)")
    print()


def test_invalid_geometry_2d_2u():
    """Test 2D-2U compound where previous 2D creates INVALID geometry (Session 62)"""
    print("TEST 7: 2D-2U Invalid Geometry (Measured Move Fallback)")
    print("=" * 60)

    # Bar 0: 2D at OLD price $170 (H=170, L=165)
    # Bar 1: 2D at CURRENT price $193 (H=193, L=188)
    # Bar 2: 2U at $197 (H=197, L=191) - TRIGGER

    # Entry: $193 (bar 1 high)
    # Stop: $188 (bar 1 low)
    # Previous 2D: bar 0 high = $170
    # Geometry: $170 < $193 entry - INVALID!
    # Expected: Measured move = $193 + ($193-$188) * 1.5 = $193 + $7.50 = $200.50

    classifications = np.array([-2.0, -2.0, 2.0], dtype=np.float64)
    high = np.array([170.0, 193.0, 197.0], dtype=np.float64)
    low = np.array([165.0, 188.0, 191.0], dtype=np.float64)

    entries, stops, targets, directions = detect_22_patterns_nb(classifications, high, low)

    stop_distance = high[1] - low[1]  # $193 - $188 = $5
    expected_target = high[1] + (stop_distance * 1.5)  # $193 + $7.50 = $200.50

    print(f"Entry: {high[1]:.2f}")
    print(f"Stop: {stops[2]:.2f}")
    print(f"Previous 2D high: {high[0]:.2f} (INVALID - below entry)")
    print(f"Stop distance: {stop_distance:.2f}")
    print(f"Target: {targets[2]:.2f} (should be {expected_target:.2f} - measured move)")
    print(f"Geometry valid: {targets[2] > high[1]}")
    print()

    assert abs(targets[2] - expected_target) < 0.01, \
        f"Target should use measured move ({expected_target:.2f}), got {targets[2]}"
    assert targets[2] > high[1], "Target must be above entry for bullish trade"
    print("PASS - Invalid Geometry (Measured Move Fallback)")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("2-2 PATTERN DETECTION TEST SUITE (Session 62 - Geometric Validation)")
    print("=" * 60)
    print()

    test_2d_2u_bullish_reversal()
    test_2u_2d_bearish_reversal()
    test_no_false_positives()
    test_2d_2d_2u_compound_reversal()
    test_2u_2u_2d_compound_reversal()
    test_valid_geometry_2d_2u()
    test_invalid_geometry_2d_2u()

    print("=" * 60)
    print("ALL 7 TESTS PASSED!")
    print("=" * 60)
