r"""
Test Suite for ATLAS-STRAT Integration Layer

Tests signal quality matrix, CRASH veto logic, and position sizing.

Test Coverage:
    - HIGH quality signal detection (regime + pattern aligned)
    - MEDIUM quality signal detection (neutral regime)
    - REJECT signal detection (counter-trend)
    - CRASH veto power (absolute reject for bullish patterns)
    - Position size multipliers (1.0, 0.5, 0.0)
    - Pattern combination logic (3-1-2 priority over 2-1-2)
    - Vectorized operations (pandas Series inputs)
    - Edge cases (no pattern, invalid inputs)

Reference:
    strat/atlas_integration.py
    docs/SYSTEM_ARCHITECTURE/INTEGRATION_ARCHITECTURE.md
"""

import pytest
import numpy as np
import pandas as pd

from strat.atlas_integration import (
    filter_strat_signals,
    get_position_size_multiplier,
    combine_pattern_signals,
)


# ============================================================================
# HIGH QUALITY SIGNAL TESTS
# ============================================================================


def test_high_quality_bull_regime_bull_pattern():
    """Test HIGH: TREND_BULL + bullish pattern (3-1-2 or 2-1-2 bullish)."""
    quality = filter_strat_signals('TREND_BULL', 1)
    assert quality == 'HIGH', "Bullish regime + bullish pattern should be HIGH quality"


def test_high_quality_bear_regime_bear_pattern():
    """Test HIGH: TREND_BEAR + bearish pattern (3-1-2 or 2-1-2 bearish)."""
    quality = filter_strat_signals('TREND_BEAR', -1)
    assert quality == 'HIGH', "Bearish regime + bearish pattern should be HIGH quality"


def test_high_quality_position_size():
    """Test HIGH quality signals get 100% position size (multiplier=1.0)."""
    multiplier = get_position_size_multiplier('HIGH')
    assert multiplier == 1.0, "HIGH quality should have 100% position size"


# ============================================================================
# MEDIUM QUALITY SIGNAL TESTS
# ============================================================================


def test_medium_quality_neutral_regime_bull_pattern():
    """Test MEDIUM: TREND_NEUTRAL + bullish pattern."""
    quality = filter_strat_signals('TREND_NEUTRAL', 1)
    assert quality == 'MEDIUM', "Neutral regime + bullish pattern should be MEDIUM quality"


def test_medium_quality_neutral_regime_bear_pattern():
    """Test MEDIUM: TREND_NEUTRAL + bearish pattern."""
    quality = filter_strat_signals('TREND_NEUTRAL', -1)
    assert quality == 'MEDIUM', "Neutral regime + bearish pattern should be MEDIUM quality"


def test_medium_quality_crash_regime_bear_pattern():
    """Test MEDIUM: CRASH + bearish pattern (shorts allowed in crash, but reduced size)."""
    quality = filter_strat_signals('CRASH', -1)
    assert quality == 'MEDIUM', "CRASH regime + bearish pattern should be MEDIUM (shorts ok, reduced size)"


def test_medium_quality_position_size():
    """Test MEDIUM quality signals get 50% position size (multiplier=0.5)."""
    multiplier = get_position_size_multiplier('MEDIUM')
    assert multiplier == 0.5, "MEDIUM quality should have 50% position size"


# ============================================================================
# REJECT SIGNAL TESTS
# ============================================================================


def test_reject_counter_trend_bull_regime_bear_pattern():
    """Test REJECT: TREND_BULL + bearish pattern (counter-trend)."""
    quality = filter_strat_signals('TREND_BULL', -1)
    assert quality == 'REJECT', "Bullish regime + bearish pattern should be REJECT (counter-trend)"


def test_reject_counter_trend_bear_regime_bull_pattern():
    """Test REJECT: TREND_BEAR + bullish pattern (counter-trend)."""
    quality = filter_strat_signals('TREND_BEAR', 1)
    assert quality == 'REJECT', "Bearish regime + bullish pattern should be REJECT (counter-trend)"


def test_reject_no_pattern():
    """Test REJECT: Any regime + no pattern (direction=0)."""
    for regime in ['TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH']:
        quality = filter_strat_signals(regime, 0)
        assert quality == 'REJECT', f"{regime} + no pattern should be REJECT"


def test_reject_position_size():
    """Test REJECT signals get 0% position size (multiplier=0.0)."""
    multiplier = get_position_size_multiplier('REJECT')
    assert multiplier == 0.0, "REJECT quality should have 0% position size (no trade)"


# ============================================================================
# CRASH VETO TESTS (CRITICAL)
# ============================================================================


def test_crash_veto_power():
    """
    Test CRASH veto: CRASH regime REJECTS all bullish patterns.

    CRITICAL TEST: This is the highest priority rule in the signal quality matrix.

    Historical validation: March 2020 had 77% CRASH regime detection.
    During this period, bullish STRAT patterns would have failed.
    Veto power prevents losses from counter-trend trades.
    """
    quality = filter_strat_signals('CRASH', 1)
    assert quality == 'REJECT', "CRASH regime MUST veto bullish patterns (absolute veto power)"


def test_crash_veto_absolute_priority():
    """Test CRASH veto has ABSOLUTE priority over pattern strength."""
    # Even if pattern is bullish (strong signal), CRASH veto overrides
    quality = filter_strat_signals('CRASH', 1)
    assert quality == 'REJECT', "CRASH veto is ABSOLUTE (highest priority rule)"

    # But bearish patterns allowed during CRASH (shorts ok, reduced size)
    quality_bear = filter_strat_signals('CRASH', -1)
    assert quality_bear == 'MEDIUM', "CRASH allows bearish patterns (shorts ok, reduced size)"


# ============================================================================
# VECTORIZED OPERATION TESTS
# ============================================================================


def test_vectorized_pandas_series():
    """Test filter_strat_signals with pandas Series inputs."""
    # Create test data
    regimes = pd.Series(['TREND_BULL', 'CRASH', 'TREND_NEUTRAL', 'TREND_BEAR'])
    directions = pd.Series([1, 1, -1, -1])

    # Apply filtering
    qualities = filter_strat_signals(regimes, directions)

    # Verify results
    assert isinstance(qualities, pd.Series), "Should return pandas Series"
    assert len(qualities) == 4, "Should have same length as input"
    assert qualities.iloc[0] == 'HIGH', "TREND_BULL + bullish = HIGH"
    assert qualities.iloc[1] == 'REJECT', "CRASH + bullish = REJECT (veto)"
    assert qualities.iloc[2] == 'MEDIUM', "TREND_NEUTRAL + bearish = MEDIUM"
    assert qualities.iloc[3] == 'HIGH', "TREND_BEAR + bearish = HIGH"


def test_vectorized_numpy_array():
    """Test filter_strat_signals with numpy array inputs."""
    # Create test data
    regimes = np.array(['TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL'])
    directions = np.array([1, -1, 0])

    # Apply filtering
    qualities = filter_strat_signals(regimes, directions)

    # Verify results (should convert to Series internally)
    assert len(qualities) == 3, "Should have same length as input"
    assert qualities.iloc[0] == 'HIGH', "TREND_BULL + bullish = HIGH"
    assert qualities.iloc[1] == 'HIGH', "TREND_BEAR + bearish = HIGH"
    assert qualities.iloc[2] == 'REJECT', "TREND_NEUTRAL + no pattern = REJECT"


def test_position_size_multiplier_vectorized():
    """Test get_position_size_multiplier with pandas Series."""
    qualities = pd.Series(['HIGH', 'MEDIUM', 'REJECT', 'HIGH'])
    multipliers = get_position_size_multiplier(qualities)

    assert isinstance(multipliers, pd.Series), "Should return pandas Series"
    assert len(multipliers) == 4, "Should have same length as input"
    assert multipliers.iloc[0] == 1.0, "HIGH = 1.0"
    assert multipliers.iloc[1] == 0.5, "MEDIUM = 0.5"
    assert multipliers.iloc[2] == 0.0, "REJECT = 0.0"
    assert multipliers.iloc[3] == 1.0, "HIGH = 1.0"


# ============================================================================
# PATTERN COMBINATION TESTS
# ============================================================================


def test_combine_patterns_312_only():
    """Test combine_pattern_signals with only 3-1-2 pattern."""
    entries_312 = np.array([False, True, False, True])
    directions_312 = np.array([0, 1, 0, -1])
    entries_212 = np.array([False, False, False, False])
    directions_212 = np.array([0, 0, 0, 0])

    combined = combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)

    assert len(combined) == 4, "Should have same length as input"
    assert combined[0] == 0, "No pattern"
    assert combined[1] == 1, "3-1-2 bullish"
    assert combined[2] == 0, "No pattern"
    assert combined[3] == -1, "3-1-2 bearish"


def test_combine_patterns_212_only():
    """Test combine_pattern_signals with only 2-1-2 pattern."""
    entries_312 = np.array([False, False, False])
    directions_312 = np.array([0, 0, 0])
    entries_212 = np.array([True, False, True])
    directions_212 = np.array([1, 0, -1])

    combined = combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)

    assert len(combined) == 3, "Should have same length as input"
    assert combined[0] == 1, "2-1-2 bullish"
    assert combined[1] == 0, "No pattern"
    assert combined[2] == -1, "2-1-2 bearish"


def test_combine_patterns_312_priority():
    """
    Test 3-1-2 takes priority over 2-1-2 when both trigger on same bar.

    Rationale: 3-1-2 is stronger reversal signal (outside bar creates more range).
    """
    # Both patterns trigger on same bar (index 0)
    entries_312 = np.array([True])
    directions_312 = np.array([1])  # 3-1-2 bullish
    entries_212 = np.array([True])
    directions_212 = np.array([-1])  # 2-1-2 bearish (conflicting)

    combined = combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)

    assert combined[0] == 1, "3-1-2 bullish should override 2-1-2 bearish (priority)"


def test_combine_patterns_pandas_series():
    """Test combine_pattern_signals with pandas Series inputs."""
    index = pd.date_range('2024-01-01', periods=3, freq='D')

    entries_312 = pd.Series([False, True, False], index=index)
    directions_312 = pd.Series([0, 1, 0], index=index)
    entries_212 = pd.Series([True, False, True], index=index)
    directions_212 = pd.Series([-1, 0, 1], index=index)

    combined = combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)

    assert isinstance(combined, pd.Series), "Should return pandas Series"
    assert len(combined) == 3, "Should have same length as input"
    assert combined.iloc[0] == -1, "2-1-2 bearish (no 3-1-2 conflict)"
    assert combined.iloc[1] == 1, "3-1-2 bullish"
    assert combined.iloc[2] == 1, "2-1-2 bullish (no 3-1-2 conflict)"
    assert combined.index.equals(index), "Should preserve index"


# ============================================================================
# SIGNAL QUALITY MATRIX COMPLETENESS TESTS
# ============================================================================


def test_signal_matrix_all_regimes_covered():
    """Test all 4 ATLAS regimes are handled correctly."""
    regimes = ['TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH']

    for regime in regimes:
        # Bullish pattern
        quality_bull = filter_strat_signals(regime, 1)
        assert quality_bull in ['HIGH', 'MEDIUM', 'REJECT'], \
            f"{regime} + bullish must return valid quality"

        # Bearish pattern
        quality_bear = filter_strat_signals(regime, -1)
        assert quality_bear in ['HIGH', 'MEDIUM', 'REJECT'], \
            f"{regime} + bearish must return valid quality"

        # No pattern
        quality_none = filter_strat_signals(regime, 0)
        assert quality_none == 'REJECT', \
            f"{regime} + no pattern must return REJECT"


def test_signal_matrix_specific_combinations():
    """
    Test specific regime-pattern combinations match architecture spec.

    Reference: docs/SYSTEM_ARCHITECTURE/INTEGRATION_ARCHITECTURE.md lines 107-138
    """
    # HIGH quality combinations
    high_combos = [
        ('TREND_BULL', 1),   # Bull regime + bull pattern
        ('TREND_BEAR', -1),  # Bear regime + bear pattern
    ]

    for regime, direction in high_combos:
        quality = filter_strat_signals(regime, direction)
        assert quality == 'HIGH', f"{regime} + direction={direction} should be HIGH"

    # MEDIUM quality combinations
    medium_combos = [
        ('TREND_NEUTRAL', 1),   # Neutral + bull pattern
        ('TREND_NEUTRAL', -1),  # Neutral + bear pattern
        ('CRASH', -1),          # Crash + bear pattern (shorts ok, reduced size)
    ]

    for regime, direction in medium_combos:
        quality = filter_strat_signals(regime, direction)
        assert quality == 'MEDIUM', f"{regime} + direction={direction} should be MEDIUM"

    # REJECT combinations
    reject_combos = [
        ('TREND_BULL', -1),  # Bull regime + bear pattern (counter-trend)
        ('TREND_BEAR', 1),   # Bear regime + bull pattern (counter-trend)
        ('CRASH', 1),        # Crash + bull pattern (VETO)
    ]

    for regime, direction in reject_combos:
        quality = filter_strat_signals(regime, direction)
        assert quality == 'REJECT', f"{regime} + direction={direction} should be REJECT"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_invalid_regime_handling():
    """Test behavior with invalid/unknown regime values."""
    # Invalid regime should not crash, return REJECT as safe default
    quality = filter_strat_signals('INVALID_REGIME', 1)
    assert quality == 'REJECT', "Invalid regime should return REJECT (safe default)"


def test_position_size_invalid_quality():
    """Test position_size_multiplier with invalid quality."""
    multiplier = get_position_size_multiplier('INVALID')
    assert multiplier == 0.0, "Invalid quality should return 0.0 (safe default)"


def test_empty_pattern_arrays():
    """Test combine_pattern_signals with empty arrays."""
    entries_312 = np.array([], dtype=bool)
    directions_312 = np.array([], dtype=np.int8)
    entries_212 = np.array([], dtype=bool)
    directions_212 = np.array([], dtype=np.int8)

    combined = combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)

    assert len(combined) == 0, "Empty input should return empty output"


# ============================================================================
# INTEGRATION TEST (REALISTIC SCENARIO)
# ============================================================================


def test_realistic_integration_scenario():
    """
    Test realistic scenario: SPY trading with mixed signals.

    Simulates 5 trading days with different regime-pattern combinations.
    """
    # Simulate 5 days of ATLAS regimes
    dates = pd.date_range('2024-03-01', periods=5, freq='D')
    atlas_regimes = pd.Series([
        'TREND_BULL',     # Day 1
        'TREND_BULL',     # Day 2
        'TREND_NEUTRAL',  # Day 3
        'CRASH',          # Day 4 (market crash)
        'TREND_BEAR',     # Day 5 (post-crash)
    ], index=dates)

    # Simulate STRAT pattern detections
    pattern_directions = pd.Series([
        1,   # Day 1: Bullish pattern
        -1,  # Day 2: Bearish pattern (counter-trend!)
        1,   # Day 3: Bullish pattern (neutral regime)
        1,   # Day 4: Bullish pattern (CRASH veto!)
        -1,  # Day 5: Bearish pattern
    ], index=dates)

    # Filter signals
    signal_qualities = filter_strat_signals(atlas_regimes, pattern_directions)
    position_sizes = get_position_size_multiplier(signal_qualities)

    # Verify results
    assert signal_qualities.iloc[0] == 'HIGH', "Day 1: Bull regime + bull pattern = HIGH"
    assert position_sizes.iloc[0] == 1.0, "Day 1: Full size"

    assert signal_qualities.iloc[1] == 'REJECT', "Day 2: Bull regime + bear pattern = REJECT"
    assert position_sizes.iloc[1] == 0.0, "Day 2: No trade"

    assert signal_qualities.iloc[2] == 'MEDIUM', "Day 3: Neutral regime = MEDIUM"
    assert position_sizes.iloc[2] == 0.5, "Day 3: Half size"

    assert signal_qualities.iloc[3] == 'REJECT', "Day 4: CRASH veto bullish pattern = REJECT"
    assert position_sizes.iloc[3] == 0.0, "Day 4: No trade (veto power)"

    assert signal_qualities.iloc[4] == 'HIGH', "Day 5: Bear regime + bear pattern = HIGH"
    assert position_sizes.iloc[4] == 1.0, "Day 5: Full size"

    # Count trades
    trades_taken = (position_sizes > 0).sum()
    assert trades_taken == 3, "Should take 3 trades out of 5 signals (2 rejected)"


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
