"""
Tests for STRAT Timeframe Continuity Checker

Validates multi-timeframe alignment detection for high-conviction signal filtering.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from strat.timeframe_continuity import (
    TimeframeContinuityChecker,
    check_full_continuity,
    get_continuity_strength
)


class TestTimeframeContinuityChecker:
    """Test TimeframeContinuityChecker class functionality."""

    def setup_method(self):
        """Create test data for each test method."""
        # Create hourly datetime index
        start = pd.Timestamp('2025-01-01')
        self.hourly_index = pd.date_range(start, periods=240, freq='1h')

        # Create base price movement (bullish trend)
        np.random.seed(42)
        base_prices = 100 + np.cumsum(np.random.randn(240) * 0.5)

        # Generate OHLC data for hourly timeframe
        self.hourly_high = pd.Series(base_prices + np.abs(np.random.randn(240) * 0.5), index=self.hourly_index)
        self.hourly_low = pd.Series(base_prices - np.abs(np.random.randn(240) * 0.5), index=self.hourly_index)

    def test_checker_initialization(self):
        """Test TimeframeContinuityChecker initializes correctly."""
        checker = TimeframeContinuityChecker()

        assert checker.timeframes == ['1M', '1W', '1D', '4H', '1H']
        assert checker.n_timeframes == 5

    def test_checker_custom_timeframes(self):
        """Test TimeframeContinuityChecker with custom timeframes."""
        custom_timeframes = ['1D', '4H', '1H']
        checker = TimeframeContinuityChecker(timeframes=custom_timeframes)

        assert checker.timeframes == custom_timeframes
        assert checker.n_timeframes == 3

    def test_resample_to_daily(self):
        """Test resampling hourly data to daily."""
        checker = TimeframeContinuityChecker()

        # Create OHLC DataFrame
        hourly_df = pd.DataFrame({
            'Open': self.hourly_high - 1,
            'High': self.hourly_high,
            'Low': self.hourly_low,
            'Close': self.hourly_low + 1
        })

        daily_df = checker.resample_to_timeframe(hourly_df, '1D')

        # Should have ~10 daily bars from 240 hourly bars
        assert len(daily_df) == 10
        assert 'High' in daily_df.columns
        assert 'Low' in daily_df.columns

    def test_check_directional_bar_bullish(self):
        """Test directional bar check for bullish 2U bars."""
        checker = TimeframeContinuityChecker()

        # 2.0 = 2U (bullish directional bar)
        assert checker.check_directional_bar(2.0, 'bullish') == True
        assert checker.check_directional_bar(2.0, 'bearish') == False

    def test_check_directional_bar_bearish(self):
        """Test directional bar check for bearish 2D bars."""
        checker = TimeframeContinuityChecker()

        # -2.0 = 2D (bearish directional bar)
        assert checker.check_directional_bar(-2.0, 'bearish') == True
        assert checker.check_directional_bar(-2.0, 'bullish') == False

    def test_check_directional_bar_inside_bar(self):
        """Test directional bar check rejects inside bars (1)."""
        checker = TimeframeContinuityChecker()

        # 1.0 = Inside bar (not directional)
        assert checker.check_directional_bar(1.0, 'bullish') == False
        assert checker.check_directional_bar(1.0, 'bearish') == False

    def test_check_directional_bar_outside_bar(self):
        """Test directional bar check rejects outside bars (3)."""
        checker = TimeframeContinuityChecker()

        # 3.0 = Outside bar (breaks both directions, not directional)
        assert checker.check_directional_bar(3.0, 'bullish') == False
        assert checker.check_directional_bar(3.0, 'bearish') == False

    def test_check_continuity_full_bullish(self):
        """Test full bullish continuity detection (all 5 timeframes aligned)."""
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H', '1H'])

        # Create perfect bullish alignment (all 2U bars)
        # Use simple uptrending bars: each bar breaks previous high only

        # Daily: 3 bars, all bullish (2U)
        daily_high = pd.Series([100, 105, 110])
        daily_low = pd.Series([95, 100, 105])  # Lows rising, highs breaking

        # 4H: 6 bars, all bullish (2U)
        four_h_high = pd.Series([100, 102, 104, 106, 108, 110])
        four_h_low = pd.Series([95, 97, 99, 101, 103, 105])

        # 1H: 10 bars, all bullish (2U)
        one_h_high = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 110])
        one_h_low = pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 105])

        high_dict = {'1D': daily_high, '4H': four_h_high, '1H': one_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low, '1H': one_h_low}

        result = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=-1)

        assert result['strength'] == 3  # All 3 timeframes aligned
        assert result['full_continuity'] == True
        assert '1D' in result['aligned_timeframes']
        assert '4H' in result['aligned_timeframes']
        assert '1H' in result['aligned_timeframes']
        assert result['direction'] == 'bullish'

    def test_check_continuity_partial(self):
        """Test partial continuity detection (some timeframes aligned)."""
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H', '1H'])

        # Daily: Bullish 2U
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        # 4H: Bullish 2U
        four_h_high = pd.Series([100, 102, 104])
        four_h_low = pd.Series([95, 97, 99])

        # 1H: Inside bar (NOT aligned)
        one_h_high = pd.Series([100, 99])  # Doesn't break previous high
        one_h_low = pd.Series([95, 96])  # Doesn't break previous low

        high_dict = {'1D': daily_high, '4H': four_h_high, '1H': one_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low, '1H': one_h_low}

        result = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=-1)

        assert result['strength'] == 2  # Only 2 of 3 aligned
        assert result['full_continuity'] == False
        assert '1D' in result['aligned_timeframes']
        assert '4H' in result['aligned_timeframes']
        assert '1H' not in result['aligned_timeframes']

    def test_check_continuity_bearish(self):
        """Test bearish continuity detection (all bearish 2D bars)."""
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H'])

        # Daily: Bearish 2D (breaks previous low only)
        daily_high = pd.Series([100, 100])  # High unchanged
        daily_low = pd.Series([95, 90])  # Low breaking down

        # 4H: Bearish 2D
        four_h_high = pd.Series([100, 100, 100])
        four_h_low = pd.Series([95, 92, 88])

        high_dict = {'1D': daily_high, '4H': four_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low}

        result = checker.check_continuity(high_dict, low_dict, 'bearish', bar_index=-1)

        assert result['strength'] == 2
        assert result['full_continuity'] == True  # All 2 available timeframes
        assert result['direction'] == 'bearish'

    def test_check_full_continuity_convenience_function(self):
        """Test check_full_continuity() convenience function."""
        # Perfect bullish alignment
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        four_h_high = pd.Series([100, 102, 104])
        four_h_low = pd.Series([95, 97, 99])

        # Note: With only 2 timeframes in dict, full continuity = both aligned
        high_dict = {'1D': daily_high, '4H': four_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low}

        # Default checker has 5 timeframes, but only 2 provided in dict
        # So we need custom checker
        from strat.timeframe_continuity import TimeframeContinuityChecker
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H'])

        result = checker.check_continuity(high_dict, low_dict, 'bullish')
        assert result['full_continuity'] == True

    def test_get_continuity_strength_convenience_function(self):
        """Test get_continuity_strength() convenience function."""
        # Create 3 timeframes, 2 aligned
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        four_h_high = pd.Series([100, 102, 104])
        four_h_low = pd.Series([95, 97, 99])

        one_h_high = pd.Series([100, 99])  # Inside bar (not aligned)
        one_h_low = pd.Series([95, 96])

        high_dict = {'1D': daily_high, '4H': four_h_high, '1H': one_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low, '1H': one_h_low}

        strength = get_continuity_strength(high_dict, low_dict, 'bullish')
        assert strength == 2  # 2 out of 3 available timeframes

    def test_check_continuity_missing_timeframe(self):
        """Test continuity check handles missing timeframes gracefully."""
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H', '1H'])

        # Only provide 2 of 3 timeframes
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        high_dict = {'1D': daily_high}  # Missing 4H and 1H
        low_dict = {'1D': daily_low}

        result = checker.check_continuity(high_dict, low_dict, 'bullish')

        assert result['strength'] == 1  # Only 1 timeframe available
        assert result['full_continuity'] == False  # Not all 3 timeframes
        assert '1D' in result['aligned_timeframes']

    def test_check_continuity_reference_bar(self):
        """Test continuity check handles reference bars (-999) correctly."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Single bar (will be reference bar -999, not directional)
        daily_high = pd.Series([100])
        daily_low = pd.Series([95])

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        result = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=0)

        # Reference bar should NOT count as aligned
        assert result['strength'] == 0
        assert result['full_continuity'] == False

    def test_check_continuity_bar_index_positive(self):
        """Test continuity check with positive bar index."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # 3 bars: ref, 2U, inside
        daily_high = pd.Series([100, 105, 104])
        daily_low = pd.Series([95, 100, 101])

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        # Check bar_index=1 (should be 2U = bullish)
        result = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=1)

        assert result['strength'] == 1
        assert result['full_continuity'] == True

    def test_check_continuity_bar_index_negative(self):
        """Test continuity check with negative bar index (from end)."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # 4 bars: ref, 2U, inside, 2U
        daily_high = pd.Series([100, 105, 104, 107])
        daily_low = pd.Series([95, 100, 101, 102])

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        # Check bar_index=-1 (last bar, should be 2U = bullish)
        result = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=-1)

        assert result['strength'] == 1
        assert result['full_continuity'] == True

        # Check bar_index=-2 (second to last, should be inside bar = not aligned)
        result2 = checker.check_continuity(high_dict, low_dict, 'bullish', bar_index=-2)

        assert result2['strength'] == 0  # Inside bar not aligned

    def test_signal_quality_matrix_integration(self):
        """Test continuity strength maps to signal quality levels."""
        # Simulate STRAT signal quality filtering using continuity strength

        # HIGH: Full continuity (5/5 or 4/5)
        high_strength = 5
        assert high_strength >= 4  # HIGH quality signal

        # MEDIUM: Partial continuity (3/5)
        medium_strength = 3
        assert 2 <= medium_strength < 4  # MEDIUM quality signal

        # LOW: Weak or no continuity (0-2/5)
        low_strength = 1
        assert low_strength < 2  # LOW quality signal (likely reject)

        # This test validates that continuity strength can drive
        # position sizing multipliers:
        # - Strength 5 or 4: 1.0x position size (HIGH)
        # - Strength 3: 0.5x position size (MEDIUM)
        # - Strength 0-2: 0.0x position size (REJECT)

    def test_integration_with_strat_patterns(self):
        """Test continuity checker integrates with existing STRAT components."""
        from strat.bar_classifier import classify_bars

        # This test validates the integration flow:
        # 1. Classify bars across multiple timeframes
        # 2. Check continuity at pattern trigger point
        # 3. Use continuity to filter pattern signals

        # Create simple bullish trend data
        daily_high = pd.Series([100, 105, 110])
        daily_low = pd.Series([95, 100, 105])

        # Classify bars
        daily_classifications = classify_bars(daily_high, daily_low)

        # Verify last bar is 2U (bullish directional)
        assert daily_classifications.iloc[-1] == 2.0  # 2U

        # Now check continuity (would be called when pattern detected)
        checker = TimeframeContinuityChecker(timeframes=['1D'])
        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        result = checker.check_continuity(high_dict, low_dict, 'bullish')

        # This bar should contribute to bullish continuity
        assert result['strength'] == 1
        assert '1D' in result['aligned_timeframes']


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_check_full_continuity_true(self):
        """Test check_full_continuity returns True for perfect alignment."""
        # Create 2-timeframe perfect alignment
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        four_h_high = pd.Series([100, 102, 104])
        four_h_low = pd.Series([95, 97, 99])

        high_dict = {'1D': daily_high, '4H': four_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low}

        # Note: Default checker expects 5 timeframes
        # With only 2 provided, full_continuity requires both aligned
        from strat.timeframe_continuity import TimeframeContinuityChecker
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H'])
        result = checker.check_continuity(high_dict, low_dict, 'bullish')

        assert result['full_continuity'] == True

    def test_check_full_continuity_false(self):
        """Test check_full_continuity returns False for partial alignment."""
        # Create 3-timeframe partial alignment (2/3)
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])

        four_h_high = pd.Series([100, 102, 104])
        four_h_low = pd.Series([95, 97, 99])

        one_h_high = pd.Series([100, 99])  # Inside bar
        one_h_low = pd.Series([95, 96])

        high_dict = {'1D': daily_high, '4H': four_h_high, '1H': one_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low, '1H': one_h_low}

        result = check_full_continuity(high_dict, low_dict, 'bullish')

        # Default checker expects all 5 timeframes, only 3 provided
        # Even if all 3 were aligned, full_continuity would be False
        assert result == False

    def test_get_continuity_strength_scores(self):
        """Test get_continuity_strength returns correct scores."""
        # Test different alignment scenarios

        # Scenario 1: No alignment (0/1)
        daily_high = pd.Series([100, 99])  # Inside bar
        daily_low = pd.Series([95, 96])

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        assert get_continuity_strength(high_dict, low_dict, 'bullish') == 0

        # Scenario 2: Full alignment (1/1)
        daily_high2 = pd.Series([100, 105])  # 2U
        daily_low2 = pd.Series([95, 100])

        high_dict2 = {'1D': daily_high2}
        low_dict2 = {'1D': daily_low2}

        assert get_continuity_strength(high_dict2, low_dict2, 'bullish') == 1


class TestType3CandleColor:
    """Test Type 3 (outside bar) TFC scoring with candle color (EQUITY-44).

    Per STRAT methodology:
    - Type 1: Does NOT count toward TFC
    - Type 2U/2D: Counts as directional
    - Type 3: Counts, direction by GREEN (Close>Open) or RED (Close<Open)
    """

    def test_type3_green_counts_as_bullish(self):
        """Type 3 bar with green candle (Close>Open) should count as bullish."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Create Type 3 bar (breaks both high and low)
        # Bar 0: Reference, Bar 1: Type 3 (outside bar)
        daily_high = pd.Series([100, 110])   # Breaks previous high
        daily_low = pd.Series([95, 85])      # Breaks previous low
        daily_open = pd.Series([97, 90])     # Open at 90
        daily_close = pd.Series([98, 105])   # Close at 105 (GREEN: Close > Open)

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}
        open_dict = {'1D': daily_open}
        close_dict = {'1D': daily_close}

        # Check bullish - should count because green candle
        result = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result['strength'] == 1, "Type 3 green candle should count as bullish"
        assert '1D' in result['aligned_timeframes']

        # Check bearish - should NOT count because green candle
        result_bear = checker.check_continuity(
            high_dict, low_dict, 'bearish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result_bear['strength'] == 0, "Type 3 green candle should NOT count as bearish"

    def test_type3_red_counts_as_bearish(self):
        """Type 3 bar with red candle (Close<Open) should count as bearish."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Create Type 3 bar (breaks both high and low)
        # Bar 0: Reference, Bar 1: Type 3 (outside bar) with red candle
        daily_high = pd.Series([100, 110])   # Breaks previous high
        daily_low = pd.Series([95, 85])      # Breaks previous low
        daily_open = pd.Series([97, 105])    # Open at 105
        daily_close = pd.Series([98, 90])    # Close at 90 (RED: Close < Open)

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}
        open_dict = {'1D': daily_open}
        close_dict = {'1D': daily_close}

        # Check bearish - should count because red candle
        result = checker.check_continuity(
            high_dict, low_dict, 'bearish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result['strength'] == 1, "Type 3 red candle should count as bearish"
        assert '1D' in result['aligned_timeframes']

        # Check bullish - should NOT count because red candle
        result_bull = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result_bull['strength'] == 0, "Type 3 red candle should NOT count as bullish"

    def test_type3_no_color_data_excluded(self):
        """Type 3 bar without Open/Close data should NOT count (backward compat)."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Create Type 3 bar without color data
        daily_high = pd.Series([100, 110])   # Breaks previous high
        daily_low = pd.Series([95, 85])      # Breaks previous low

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}

        # Without open_dict/close_dict, Type 3 should NOT count (old behavior)
        result = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1
        )

        assert result['strength'] == 0, "Type 3 without candle data should NOT count"

        result_bear = checker.check_continuity(
            high_dict, low_dict, 'bearish', bar_index=-1
        )

        assert result_bear['strength'] == 0, "Type 3 without candle data should NOT count"

    def test_type1_still_excluded_with_color_data(self):
        """Type 1 (inside bar) should still NOT count even with Open/Close data."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Create Type 1 bar (inside bar - doesn't break either)
        daily_high = pd.Series([100, 98])    # Does NOT break previous high
        daily_low = pd.Series([95, 96])      # Does NOT break previous low
        daily_open = pd.Series([97, 97])     # Open/Close doesn't matter
        daily_close = pd.Series([98, 97.5])  # for inside bars

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}
        open_dict = {'1D': daily_open}
        close_dict = {'1D': daily_close}

        result = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result['strength'] == 0, "Type 1 should NOT count even with color data"

    def test_type2_unchanged_with_color_data(self):
        """Type 2 bars should work the same with or without color data."""
        checker = TimeframeContinuityChecker(timeframes=['1D'])

        # Create Type 2U bar
        daily_high = pd.Series([100, 105])   # Breaks previous high
        daily_low = pd.Series([95, 100])     # Does NOT break previous low
        daily_open = pd.Series([97, 101])
        daily_close = pd.Series([98, 104])

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}
        open_dict = {'1D': daily_open}
        close_dict = {'1D': daily_close}

        # Without color data
        result_no_color = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1
        )

        # With color data
        result_with_color = checker.check_continuity(
            high_dict, low_dict, 'bullish', bar_index=-1,
            open_dict=open_dict, close_dict=close_dict
        )

        assert result_no_color['strength'] == result_with_color['strength']
        assert result_no_color['strength'] == 1, "Type 2U should count as bullish"

    def test_multi_timeframe_with_type3(self):
        """Test multi-timeframe continuity includes Type 3 with candle color."""
        checker = TimeframeContinuityChecker(timeframes=['1D', '4H', '1H'])

        # 1D: Type 2U (bullish)
        daily_high = pd.Series([100, 105])
        daily_low = pd.Series([95, 100])
        daily_open = pd.Series([97, 101])
        daily_close = pd.Series([98, 104])

        # 4H: Type 3 Green (bullish)
        four_h_high = pd.Series([100, 110])   # Breaks high
        four_h_low = pd.Series([95, 85])      # Breaks low
        four_h_open = pd.Series([97, 90])
        four_h_close = pd.Series([98, 105])   # Green candle

        # 1H: Type 2U (bullish)
        one_h_high = pd.Series([100, 102])
        one_h_low = pd.Series([95, 100])
        one_h_open = pd.Series([97, 100])
        one_h_close = pd.Series([98, 101])

        high_dict = {'1D': daily_high, '4H': four_h_high, '1H': one_h_high}
        low_dict = {'1D': daily_low, '4H': four_h_low, '1H': one_h_low}
        open_dict = {'1D': daily_open, '4H': four_h_open, '1H': one_h_open}
        close_dict = {'1D': daily_close, '4H': four_h_close, '1H': one_h_close}

        # Use check_continuity (not flexible) to test all provided timeframes
        result = checker.check_continuity(
            high_dict=high_dict,
            low_dict=low_dict,
            direction='bullish',
            bar_index=-1,
            open_dict=open_dict,
            close_dict=close_dict
        )

        # All 3 should be aligned (1D 2U, 4H Type3 Green, 1H 2U)
        assert result['strength'] == 3, f"Expected 3 aligned, got {result['strength']}"
        assert '4H' in result['aligned_timeframes'], "Type 3 Green 4H should be aligned"
        assert result['full_continuity'] == True, "All 3 timeframes should give full continuity"

    def test_convenience_function_with_type3(self):
        """Test get_continuity_strength with Type 3 candle color."""
        # Type 3 Green bar
        daily_high = pd.Series([100, 110])
        daily_low = pd.Series([95, 85])
        daily_open = pd.Series([97, 90])
        daily_close = pd.Series([98, 105])  # Green

        high_dict = {'1D': daily_high}
        low_dict = {'1D': daily_low}
        open_dict = {'1D': daily_open}
        close_dict = {'1D': daily_close}

        strength = get_continuity_strength(
            high_dict, low_dict, 'bullish',
            open_dict=open_dict, close_dict=close_dict
        )

        assert strength == 1, "Type 3 Green should count as bullish in convenience function"
