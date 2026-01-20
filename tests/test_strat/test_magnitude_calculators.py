"""
Tests for strat/magnitude_calculators.py

Covers:
- MagnitudeResult dataclass
- Numba helper functions (validate_target_geometry, calculate_measured_move, etc.)
- OptionA_PreviousOutsideBar calculator
- OptionB_SwingPivot calculator
- OptionC_MeasuredMove calculator
- Factory function get_all_calculators
"""

import pytest
import numpy as np
from strat.magnitude_calculators import (
    MagnitudeResult,
    validate_target_geometry_nb,
    calculate_measured_move_nb,
    calculate_rr_ratio_nb,
    find_previous_outside_bar_nb,
    find_swing_high_nb,
    find_swing_low_nb,
    MagnitudeCalculator,
    OptionA_PreviousOutsideBar,
    OptionB_SwingPivot,
    OptionC_MeasuredMove,
    get_all_calculators,
)


# =============================================================================
# MagnitudeResult Dataclass Tests
# =============================================================================

class TestMagnitudeResult:
    """Test MagnitudeResult dataclass."""

    def test_creation_with_all_fields(self):
        """MagnitudeResult can be created with all fields."""
        result = MagnitudeResult(
            target_price=307.50,
            method_used="measured_move",
            rr_ratio=1.5,
            lookback_distance=10
        )
        assert result.target_price == 307.50
        assert result.method_used == "measured_move"
        assert result.rr_ratio == 1.5
        assert result.lookback_distance == 10

    def test_creation_without_lookback(self):
        """MagnitudeResult can be created without lookback_distance."""
        result = MagnitudeResult(
            target_price=307.50,
            method_used="measured_move",
            rr_ratio=1.5
        )
        assert result.lookback_distance is None

    def test_repr_formatting(self):
        """MagnitudeResult __repr__ formats correctly."""
        result = MagnitudeResult(
            target_price=307.50,
            method_used="swing_pivot",
            rr_ratio=2.0
        )
        repr_str = repr(result)
        assert "target=307.50" in repr_str
        assert "method='swing_pivot'" in repr_str
        assert "R:R=2.00" in repr_str

    def test_different_method_types(self):
        """MagnitudeResult accepts different method types."""
        for method in ["previous_outside", "swing_pivot", "measured_move"]:
            result = MagnitudeResult(
                target_price=100.0,
                method_used=method,
                rr_ratio=1.0
            )
            assert result.method_used == method


# =============================================================================
# Numba Helper Function Tests
# =============================================================================

class TestValidateTargetGeometry:
    """Test validate_target_geometry_nb function."""

    def test_bullish_valid_geometry(self):
        """Bullish: target above entry is valid."""
        # Entry $300, stop $295, target $310 - bullish valid
        assert validate_target_geometry_nb(300.0, 295.0, 310.0, 1) is True

    def test_bullish_invalid_geometry(self):
        """Bullish: target below entry is invalid."""
        # Entry $300, stop $295, target $290 - bullish INVALID
        assert validate_target_geometry_nb(300.0, 295.0, 290.0, 1) is False

    def test_bearish_valid_geometry(self):
        """Bearish: target below entry is valid."""
        # Entry $300, stop $305, target $290 - bearish valid
        assert validate_target_geometry_nb(300.0, 305.0, 290.0, -1) is True

    def test_bearish_invalid_geometry(self):
        """Bearish: target above entry is invalid."""
        # Entry $300, stop $305, target $310 - bearish INVALID
        assert validate_target_geometry_nb(300.0, 305.0, 310.0, -1) is False

    def test_invalid_direction_returns_false(self):
        """Invalid direction returns False."""
        assert validate_target_geometry_nb(300.0, 295.0, 310.0, 0) is False
        assert validate_target_geometry_nb(300.0, 295.0, 310.0, 2) is False

    def test_target_equal_to_entry(self):
        """Target equal to entry is invalid (no profit)."""
        assert validate_target_geometry_nb(300.0, 295.0, 300.0, 1) is False
        assert validate_target_geometry_nb(300.0, 305.0, 300.0, -1) is False


class TestCalculateMeasuredMove:
    """Test calculate_measured_move_nb function."""

    def test_bullish_measured_move(self):
        """Bullish measured move projects upward."""
        # Entry $300, stop $295, 1.5x = $300 + $5 * 1.5 = $307.50
        target = calculate_measured_move_nb(300.0, 295.0, 1, 1.5)
        assert target == pytest.approx(307.50)

    def test_bearish_measured_move(self):
        """Bearish measured move projects downward."""
        # Entry $300, stop $305, 1.5x = $300 - $5 * 1.5 = $292.50
        target = calculate_measured_move_nb(300.0, 305.0, -1, 1.5)
        assert target == pytest.approx(292.50)

    def test_different_multipliers(self):
        """Different multipliers produce different targets."""
        # Entry $100, stop $95 (risk = $5)
        target_1x = calculate_measured_move_nb(100.0, 95.0, 1, 1.0)
        target_2x = calculate_measured_move_nb(100.0, 95.0, 1, 2.0)
        target_3x = calculate_measured_move_nb(100.0, 95.0, 1, 3.0)

        assert target_1x == pytest.approx(105.0)
        assert target_2x == pytest.approx(110.0)
        assert target_3x == pytest.approx(115.0)

    def test_invalid_direction_returns_entry(self):
        """Invalid direction returns entry price."""
        target = calculate_measured_move_nb(300.0, 295.0, 0, 1.5)
        assert target == 300.0

    def test_large_stop_distance(self):
        """Large stop distance produces proportionally larger target."""
        # Entry $100, stop $80 (risk = $20), 1.5x = $100 + $30 = $130
        target = calculate_measured_move_nb(100.0, 80.0, 1, 1.5)
        assert target == pytest.approx(130.0)


class TestCalculateRRRatio:
    """Test calculate_rr_ratio_nb function."""

    def test_1_to_1_ratio(self):
        """1:1 risk-reward calculation."""
        # Entry $100, stop $95 (risk $5), target $105 (reward $5)
        rr = calculate_rr_ratio_nb(100.0, 95.0, 105.0)
        assert rr == pytest.approx(1.0)

    def test_1_5_to_1_ratio(self):
        """1.5:1 risk-reward calculation."""
        # Entry $100, stop $95 (risk $5), target $107.50 (reward $7.50)
        rr = calculate_rr_ratio_nb(100.0, 95.0, 107.50)
        assert rr == pytest.approx(1.5)

    def test_2_to_1_ratio(self):
        """2:1 risk-reward calculation."""
        # Entry $100, stop $95 (risk $5), target $110 (reward $10)
        rr = calculate_rr_ratio_nb(100.0, 95.0, 110.0)
        assert rr == pytest.approx(2.0)

    def test_zero_risk_returns_zero(self):
        """Zero risk returns 0 R:R."""
        rr = calculate_rr_ratio_nb(100.0, 100.0, 110.0)
        assert rr == 0.0

    def test_bearish_rr_calculation(self):
        """Bearish R:R calculation works correctly."""
        # Entry $100, stop $105 (risk $5), target $90 (reward $10)
        rr = calculate_rr_ratio_nb(100.0, 105.0, 90.0)
        assert rr == pytest.approx(2.0)


class TestFindPreviousOutsideBar:
    """Test find_previous_outside_bar_nb function."""

    def test_finds_outside_bar(self):
        """Finds previous outside bar (Type 3)."""
        # Classifications: ... 3, 2, 2, 1, 2 (index 0-4)
        classifications = np.array([3.0, 2.0, 2.0, 1.0, 2.0])
        idx = find_previous_outside_bar_nb(classifications, 4, max_lookback=100)
        assert idx == 0

    def test_finds_nearest_outside_bar(self):
        """Finds nearest (most recent) outside bar."""
        # Classifications: 3, 2, 3, 1, 2 (index 0-4)
        classifications = np.array([3.0, 2.0, 3.0, 1.0, 2.0])
        idx = find_previous_outside_bar_nb(classifications, 4, max_lookback=100)
        assert idx == 2  # Nearest Type 3

    def test_handles_negative_type_3(self):
        """Handles negative Type 3 (abs(class) == 3)."""
        # Classifications with -3
        classifications = np.array([2.0, -3.0, 2.0, 1.0, 2.0])
        idx = find_previous_outside_bar_nb(classifications, 4, max_lookback=100)
        assert idx == 1

    def test_returns_negative_one_when_not_found(self):
        """Returns -1 when no outside bar found."""
        # No Type 3 bars
        classifications = np.array([2.0, 2.0, 1.0, 2.0, 2.0])
        idx = find_previous_outside_bar_nb(classifications, 4, max_lookback=100)
        assert idx == -1

    def test_respects_max_lookback(self):
        """Respects max_lookback limit."""
        # Type 3 at index 0, but max_lookback=2 from index 4
        classifications = np.array([3.0, 2.0, 2.0, 2.0, 2.0])
        idx = find_previous_outside_bar_nb(classifications, 4, max_lookback=2)
        assert idx == -1  # Can't look back far enough

    def test_empty_array(self):
        """Handles empty array."""
        classifications = np.array([], dtype=np.float64)
        # This should handle gracefully
        idx = find_previous_outside_bar_nb(classifications, 0, max_lookback=100)
        assert idx == -1


class TestFindSwingHigh:
    """Test find_swing_high_nb function."""

    def test_finds_swing_high_n2(self):
        """Finds N=2 swing high above threshold."""
        # Create array with clear swing high at index 5
        # Pattern: lower, lower, HIGH, lower, lower
        high = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 104.0, 103.0, 102.0, 101.0])
        # Index 5 (110.0) is swing high: higher than 2 bars before and 2 bars after
        idx = find_swing_high_nb(high, start_idx=9, threshold=105.0, n_bars=2, max_lookback=50)
        assert idx == 5

    def test_respects_threshold(self):
        """Only finds swing high above threshold."""
        # Swing high at 110, but threshold is 115
        high = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 104.0, 103.0, 102.0, 101.0])
        idx = find_swing_high_nb(high, start_idx=9, threshold=115.0, n_bars=2, max_lookback=50)
        assert idx == -1  # Not above threshold

    def test_returns_negative_one_when_no_swing(self):
        """Returns -1 when no swing high found."""
        # Monotonically decreasing - no swing high
        high = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0])
        idx = find_swing_high_nb(high, start_idx=9, threshold=90.0, n_bars=2, max_lookback=50)
        assert idx == -1

    def test_n3_requires_more_confirmation(self):
        """N=3 swing requires 3 bars on each side."""
        # Create swing that satisfies N=2 but not N=3
        high = np.array([100.0, 101.0, 102.0, 110.0, 102.0, 101.0, 100.0])
        # Index 3 is N=2 swing but not N=3 (needs 3 bars each side)
        idx_n2 = find_swing_high_nb(high, start_idx=6, threshold=105.0, n_bars=2, max_lookback=50)
        idx_n3 = find_swing_high_nb(high, start_idx=6, threshold=105.0, n_bars=3, max_lookback=50)
        # N=2 should find it, N=3 should not (not enough bars on sides)
        assert idx_n2 == 3
        assert idx_n3 == -1

    def test_short_array_handling(self):
        """Handles arrays too short for swing detection."""
        high = np.array([100.0, 110.0, 100.0])  # Only 3 bars
        idx = find_swing_high_nb(high, start_idx=2, threshold=105.0, n_bars=2, max_lookback=50)
        assert idx == -1  # Not enough bars for N=2 swing


class TestFindSwingLow:
    """Test find_swing_low_nb function."""

    def test_finds_swing_low_n2(self):
        """Finds N=2 swing low below threshold."""
        # Create array with clear swing low at index 5
        low = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 90.0, 96.0, 97.0, 98.0, 99.0])
        # Index 5 (90.0) is swing low
        idx = find_swing_low_nb(low, start_idx=9, threshold=95.0, n_bars=2, max_lookback=50)
        assert idx == 5

    def test_respects_threshold(self):
        """Only finds swing low below threshold."""
        # Swing low at 90, but threshold is 85
        low = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 90.0, 96.0, 97.0, 98.0, 99.0])
        idx = find_swing_low_nb(low, start_idx=9, threshold=85.0, n_bars=2, max_lookback=50)
        assert idx == -1  # Not below threshold

    def test_returns_negative_one_when_no_swing(self):
        """Returns -1 when no swing low found."""
        # Monotonically increasing - no swing low
        low = np.array([90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0])
        idx = find_swing_low_nb(low, start_idx=9, threshold=100.0, n_bars=2, max_lookback=50)
        assert idx == -1

    def test_finds_nearest_swing_low(self):
        """Finds nearest (most recent) swing low."""
        # Two swing lows
        low = np.array([100.0, 99.0, 85.0, 99.0, 100.0, 99.0, 88.0, 99.0, 100.0, 100.0])
        # Swing lows at index 2 (85) and index 6 (88)
        idx = find_swing_low_nb(low, start_idx=9, threshold=95.0, n_bars=2, max_lookback=50)
        assert idx == 6  # Nearest one


# =============================================================================
# OptionC_MeasuredMove Tests
# =============================================================================

class TestOptionCMeasuredMove:
    """Test OptionC_MeasuredMove calculator."""

    def test_default_multiplier(self):
        """Default multiplier is 1.5."""
        calc = OptionC_MeasuredMove()
        assert calc.multiplier == 1.5

    def test_custom_multiplier(self):
        """Accepts custom multiplier."""
        calc = OptionC_MeasuredMove(multiplier=2.0)
        assert calc.multiplier == 2.0

    def test_name_includes_multiplier(self):
        """Name includes multiplier value."""
        calc = OptionC_MeasuredMove(multiplier=1.5)
        assert "1.5x" in calc.name

    def test_bullish_target_calculation(self):
        """Bullish target calculated correctly."""
        calc = OptionC_MeasuredMove(multiplier=1.5)
        high = np.array([100.0] * 10)
        low = np.array([95.0] * 10)
        classifications = np.array([2.0] * 10)

        result = calc.calculate_target(
            entry_price=300.0,
            stop_price=295.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        assert result.target_price == pytest.approx(307.50)
        assert result.method_used == "measured_move"
        assert result.rr_ratio == pytest.approx(1.5)
        assert result.lookback_distance is None

    def test_bearish_target_calculation(self):
        """Bearish target calculated correctly."""
        calc = OptionC_MeasuredMove(multiplier=1.5)
        high = np.array([100.0] * 10)
        low = np.array([95.0] * 10)
        classifications = np.array([2.0] * 10)

        result = calc.calculate_target(
            entry_price=300.0,
            stop_price=305.0,
            direction=-1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        assert result.target_price == pytest.approx(292.50)
        assert result.method_used == "measured_move"
        assert result.rr_ratio == pytest.approx(1.5)

    def test_always_uses_measured_move_method(self):
        """Method is always 'measured_move'."""
        calc = OptionC_MeasuredMove()
        high = np.array([100.0] * 10)
        low = np.array([95.0] * 10)
        classifications = np.array([3.0] * 10)  # All Type 3 bars

        result = calc.calculate_target(
            entry_price=100.0,
            stop_price=95.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Even with Type 3 bars available, still uses measured move
        assert result.method_used == "measured_move"


# =============================================================================
# OptionA_PreviousOutsideBar Tests
# =============================================================================

class TestOptionAPreviousOutsideBar:
    """Test OptionA_PreviousOutsideBar calculator."""

    def test_default_parameters(self):
        """Default parameters are set correctly."""
        calc = OptionA_PreviousOutsideBar()
        assert calc.max_lookback == 100
        assert calc.fallback_multiplier == 1.5

    def test_custom_parameters(self):
        """Accepts custom parameters."""
        calc = OptionA_PreviousOutsideBar(max_lookback=50, fallback_multiplier=2.0)
        assert calc.max_lookback == 50
        assert calc.fallback_multiplier == 2.0

    def test_name_property(self):
        """Name property returns correct value."""
        calc = OptionA_PreviousOutsideBar()
        assert calc.name == "Option_A_PreviousOutsideBar"

    def test_finds_previous_outside_bar_bullish(self):
        """Finds previous outside bar for bullish pattern."""
        calc = OptionA_PreviousOutsideBar()

        # Setup: Outside bar at index 2 with high=310
        high = np.array([300.0, 305.0, 310.0, 302.0, 301.0, 303.0])
        low = np.array([295.0, 298.0, 290.0, 298.0, 299.0, 300.0])
        classifications = np.array([2.0, 2.0, 3.0, 1.0, 2.0, 2.0])

        result = calc.calculate_target(
            entry_price=303.0,
            stop_price=300.0,
            direction=1,  # Bullish
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Should use high of previous outside bar (310)
        assert result.target_price == pytest.approx(310.0)
        assert result.method_used == "previous_outside"
        assert result.lookback_distance == 3  # 5 - 2 = 3

    def test_finds_previous_outside_bar_bearish(self):
        """Finds previous outside bar for bearish pattern."""
        calc = OptionA_PreviousOutsideBar()

        # Setup: Outside bar at index 2 with low=285
        high = np.array([300.0, 305.0, 310.0, 302.0, 301.0, 300.0])
        low = np.array([295.0, 298.0, 285.0, 298.0, 299.0, 297.0])
        classifications = np.array([2.0, 2.0, 3.0, 1.0, 2.0, 2.0])

        result = calc.calculate_target(
            entry_price=297.0,
            stop_price=300.0,
            direction=-1,  # Bearish
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Should use low of previous outside bar (285)
        assert result.target_price == pytest.approx(285.0)
        assert result.method_used == "previous_outside"

    def test_falls_back_on_invalid_geometry(self):
        """Falls back to measured move when geometry is invalid."""
        calc = OptionA_PreviousOutsideBar()

        # Setup: Outside bar with high BELOW entry (invalid for bullish)
        high = np.array([300.0, 305.0, 295.0, 302.0, 301.0, 303.0])  # Outside bar high=295 < entry=303
        low = np.array([295.0, 298.0, 290.0, 298.0, 299.0, 300.0])
        classifications = np.array([2.0, 2.0, 3.0, 1.0, 2.0, 2.0])

        result = calc.calculate_target(
            entry_price=303.0,
            stop_price=300.0,
            direction=1,  # Bullish but outside bar high is below entry
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Should fall back to measured move
        assert result.method_used == "measured_move"
        assert result.rr_ratio == pytest.approx(1.5)

    def test_falls_back_when_no_outside_bar(self):
        """Falls back to measured move when no outside bar found."""
        calc = OptionA_PreviousOutsideBar()

        # No Type 3 bars
        high = np.array([300.0, 305.0, 302.0, 301.0, 303.0, 304.0])
        low = np.array([295.0, 298.0, 298.0, 299.0, 300.0, 301.0])
        classifications = np.array([2.0, 2.0, 2.0, 1.0, 2.0, 2.0])  # No Type 3

        result = calc.calculate_target(
            entry_price=304.0,
            stop_price=301.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        assert result.method_used == "measured_move"

    def test_falls_back_when_not_enough_bars(self):
        """Falls back when pattern_idx - 2 < 0."""
        calc = OptionA_PreviousOutsideBar()

        high = np.array([300.0, 305.0])
        low = np.array([295.0, 298.0])
        classifications = np.array([3.0, 2.0])

        result = calc.calculate_target(
            entry_price=305.0,
            stop_price=298.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=1  # pattern_idx - 2 = -1
        )

        assert result.method_used == "measured_move"


# =============================================================================
# OptionB_SwingPivot Tests
# =============================================================================

class TestOptionBSwingPivot:
    """Test OptionB_SwingPivot calculator."""

    def test_default_parameters(self):
        """Default parameters are set correctly."""
        calc = OptionB_SwingPivot()
        assert calc.n_bars == 2
        assert calc.max_lookback == 50
        assert calc.fallback_multiplier == 1.5

    def test_custom_parameters(self):
        """Accepts custom parameters."""
        calc = OptionB_SwingPivot(n_bars=3, max_lookback=100, fallback_multiplier=2.0)
        assert calc.n_bars == 3
        assert calc.max_lookback == 100
        assert calc.fallback_multiplier == 2.0

    def test_name_includes_n_bars(self):
        """Name includes n_bars value."""
        calc2 = OptionB_SwingPivot(n_bars=2)
        calc3 = OptionB_SwingPivot(n_bars=3)
        assert "N2" in calc2.name
        assert "N3" in calc3.name

    def test_finds_swing_high_bullish(self):
        """Finds swing high for bullish pattern."""
        calc = OptionB_SwingPivot(n_bars=2)

        # Create data with swing high at index 3
        # Need at least 2 bars before and after for N=2 swing
        high = np.array([300.0, 305.0, 308.0, 320.0, 308.0, 305.0, 310.0, 312.0, 311.0, 313.0])
        low = np.array([295.0, 300.0, 303.0, 305.0, 303.0, 300.0, 305.0, 307.0, 308.0, 310.0])
        classifications = np.array([2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0])

        result = calc.calculate_target(
            entry_price=313.0,
            stop_price=310.0,
            direction=1,  # Bullish
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=9  # bar3_idx = 8, search starts from 7
        )

        # Should find swing high at index 3 (320.0)
        assert result.target_price == pytest.approx(320.0)
        assert result.method_used == "swing_pivot"
        assert result.lookback_distance == 6  # 9 - 3

    def test_finds_swing_low_bearish(self):
        """Finds swing low for bearish pattern."""
        calc = OptionB_SwingPivot(n_bars=2)

        # Create data with swing low at index 3
        high = np.array([300.0, 295.0, 292.0, 290.0, 292.0, 295.0, 290.0, 288.0, 289.0, 287.0])
        low = np.array([295.0, 290.0, 287.0, 280.0, 287.0, 290.0, 285.0, 283.0, 284.0, 282.0])
        classifications = np.array([2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0])

        result = calc.calculate_target(
            entry_price=282.0,
            stop_price=285.0,
            direction=-1,  # Bearish
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=9
        )

        # Should find swing low at index 3 (280.0)
        assert result.target_price == pytest.approx(280.0)
        assert result.method_used == "swing_pivot"

    def test_respects_threshold_bullish(self):
        """Only finds swing high above 3-bar's high."""
        calc = OptionB_SwingPivot(n_bars=2)

        # Swing high at 310, but 3-bar's high is 315 (threshold)
        high = np.array([300.0, 305.0, 308.0, 310.0, 308.0, 305.0, 310.0, 312.0, 315.0, 316.0])
        low = np.array([295.0, 300.0, 303.0, 305.0, 303.0, 300.0, 305.0, 307.0, 310.0, 312.0])
        classifications = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0])

        result = calc.calculate_target(
            entry_price=316.0,
            stop_price=312.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=9  # bar3_idx=8 has high=315
        )

        # Swing high at index 3 (310.0) is below threshold (315), should fall back
        assert result.method_used == "measured_move"

    def test_falls_back_when_no_swing(self):
        """Falls back to measured move when no swing found."""
        calc = OptionB_SwingPivot(n_bars=2)

        # Monotonically increasing - no swing
        high = np.array([300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0])
        low = np.array([295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0])
        classifications = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0])

        result = calc.calculate_target(
            entry_price=309.0,
            stop_price=304.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=9
        )

        assert result.method_used == "measured_move"

    def test_falls_back_not_enough_data(self):
        """Falls back when not enough data for swing detection."""
        calc = OptionB_SwingPivot(n_bars=2)

        # Only 3 bars - not enough for pattern + swing detection
        high = np.array([300.0, 305.0, 310.0])
        low = np.array([295.0, 300.0, 305.0])
        classifications = np.array([3.0, 1.0, 2.0])

        result = calc.calculate_target(
            entry_price=310.0,
            stop_price=305.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=2
        )

        assert result.method_used == "measured_move"


# =============================================================================
# get_all_calculators Factory Tests
# =============================================================================

class TestGetAllCalculators:
    """Test get_all_calculators factory function."""

    def test_returns_four_calculators(self):
        """Returns list of 4 calculators."""
        calculators = get_all_calculators()
        assert len(calculators) == 4

    def test_all_are_magnitude_calculators(self):
        """All returned objects are MagnitudeCalculator instances."""
        calculators = get_all_calculators()
        for calc in calculators:
            assert isinstance(calc, MagnitudeCalculator)

    def test_contains_option_a(self):
        """Contains OptionA_PreviousOutsideBar."""
        calculators = get_all_calculators()
        option_a = [c for c in calculators if isinstance(c, OptionA_PreviousOutsideBar)]
        assert len(option_a) == 1

    def test_contains_option_b_n2_and_n3(self):
        """Contains OptionB_SwingPivot with N=2 and N=3."""
        calculators = get_all_calculators()
        option_b = [c for c in calculators if isinstance(c, OptionB_SwingPivot)]
        assert len(option_b) == 2

        n_values = [c.n_bars for c in option_b]
        assert 2 in n_values
        assert 3 in n_values

    def test_contains_option_c(self):
        """Contains OptionC_MeasuredMove."""
        calculators = get_all_calculators()
        option_c = [c for c in calculators if isinstance(c, OptionC_MeasuredMove)]
        assert len(option_c) == 1

    def test_unique_names(self):
        """All calculators have unique names."""
        calculators = get_all_calculators()
        names = [c.name for c in calculators]
        assert len(names) == len(set(names))


# =============================================================================
# Integration Tests
# =============================================================================

class TestMagnitudeCalculatorIntegration:
    """Integration tests for magnitude calculators."""

    def test_all_calculators_handle_same_input(self):
        """All calculators can process the same input without errors."""
        calculators = get_all_calculators()

        high = np.array([100.0, 105.0, 110.0, 115.0, 110.0, 105.0, 100.0, 105.0, 110.0, 112.0])
        low = np.array([95.0, 100.0, 105.0, 108.0, 105.0, 100.0, 95.0, 100.0, 105.0, 107.0])
        classifications = np.array([2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0, 1.0, 2.0])

        for calc in calculators:
            result = calc.calculate_target(
                entry_price=112.0,
                stop_price=107.0,
                direction=1,
                high=high,
                low=low,
                classifications=classifications,
                pattern_idx=9
            )

            assert isinstance(result, MagnitudeResult)
            assert result.target_price > 0
            assert result.rr_ratio > 0
            assert result.method_used in ["previous_outside", "swing_pivot", "measured_move"]

    def test_bullish_vs_bearish_targets(self):
        """Bullish targets are above entry, bearish below."""
        calc = OptionC_MeasuredMove()

        high = np.array([100.0] * 10)
        low = np.array([95.0] * 10)
        classifications = np.array([2.0] * 10)

        # Bullish
        bullish_result = calc.calculate_target(
            entry_price=100.0,
            stop_price=95.0,
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Bearish
        bearish_result = calc.calculate_target(
            entry_price=100.0,
            stop_price=105.0,
            direction=-1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        assert bullish_result.target_price > 100.0
        assert bearish_result.target_price < 100.0

    def test_rr_ratio_consistency(self):
        """R:R ratio matches expected calculation."""
        calc = OptionC_MeasuredMove(multiplier=2.0)

        high = np.array([100.0] * 10)
        low = np.array([95.0] * 10)
        classifications = np.array([2.0] * 10)

        result = calc.calculate_target(
            entry_price=100.0,
            stop_price=95.0,  # Risk = $5
            direction=1,
            high=high,
            low=low,
            classifications=classifications,
            pattern_idx=5
        )

        # Target should be 100 + 5*2 = 110, reward = $10, R:R = 2.0
        assert result.target_price == pytest.approx(110.0)
        assert result.rr_ratio == pytest.approx(2.0)
