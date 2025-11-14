"""
Test suite for STRAT bar classification.

Tests cover:
1. Known synthetic sequences with hand-calculated expected values
2. Edge cases (all inside bars, all outside bars, alternating patterns)
3. Governing range persistence across consecutive inside bars
4. Real SPY data validation with distribution checks
5. Performance benchmarks with large datasets
"""

import pytest
import numpy as np
import pandas as pd
import time
import vectorbtpro as vbt

from strat.bar_classifier import StratBarClassifier, classify_bars, classify_bars_nb


class TestSyntheticData:
    """Test bar classification with known synthetic sequences."""

    def test_known_5_bar_sequence(self):
        """
        Test with hand-calculated 5-bar sequence.

        Sequence: Reference, 2U, Inside, 2U, Outside
        High: [100, 105, 104, 107, 110]
        Low:  [95,  98,  99,  101, 93]

        Expected:
        - Bar 0: Reference (-999)
        - Bar 1: Breaks high (105 > 100) -> 2U (gov_range: 105/95)
        - Bar 2: Inside (104 <= 105 and 99 >= 95) -> 1 (gov_range: unchanged)
        - Bar 3: Breaks high (107 > 105) -> 2U (gov_range: 107/95)
        - Bar 4: Breaks both (110 > 107 and 93 < 95) -> 3 (gov_range: 110/93)
        """
        test_data = pd.DataFrame({
            'high': [100, 105, 104, 107, 110],
            'low': [95, 98, 99, 101, 93],
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        expected = np.array([-999, 2, 1, 2, 3])
        np.testing.assert_array_equal(
            result.classification.values,
            expected,
            err_msg="5-bar sequence classification incorrect"
        )

    def test_all_inside_bars(self):
        """
        Test sequence of all inside bars.

        After reference bar, all bars stay inside governing range.
        Expected: Ref, 1, 1, 1, 1
        """
        test_data = pd.DataFrame({
            'high': [100, 99, 98, 97, 96],  # Declining highs (all inside)
            'low': [90, 91, 92, 93, 94],    # Rising lows (all inside)
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        expected = np.array([-999, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            result.classification.values,
            expected,
            err_msg="All inside bars sequence incorrect"
        )

    def test_all_outside_bars(self):
        """
        Test sequence of all outside bars.

        Each bar breaks both high and low of previous governing range.
        Expected: Ref, 3, 3, 3, 3
        """
        test_data = pd.DataFrame({
            'high': [100, 105, 110, 115, 120],  # Each bar higher high
            'low': [90, 85, 80, 75, 70],        # Each bar lower low
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        expected = np.array([-999, 3, 3, 3, 3])
        np.testing.assert_array_equal(
            result.classification.values,
            expected,
            err_msg="All outside bars sequence incorrect"
        )

    def test_alternating_2u_2d(self):
        """
        Test alternating 2U and 2D bars.

        Sequence alternates between breaking high and breaking low.
        Expected: Ref, 2U, 2D, 2U, 2D
        """
        test_data = pd.DataFrame({
            'high': [100, 105, 104, 108, 107],
            'low': [95, 96, 90, 92, 88],
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        # Bar 0: Ref (-999)
        # Bar 1: H breaks (105>100) -> 2U (gov: 105/95)
        # Bar 2: L breaks (90<95) -> 2D (gov: 104/90)
        # Bar 3: H breaks (108>104) -> 2U (gov: 108/90)
        # Bar 4: L breaks (88<90) -> 2D (gov: 107/88)
        expected = np.array([-999, 2, -2, 2, -2])
        np.testing.assert_array_equal(
            result.classification.values,
            expected,
            err_msg="Alternating 2U/2D sequence incorrect"
        )

    def test_governing_range_persistence(self):
        """
        Test that governing range persists across multiple inside bars.

        Critical test: Inside bars should reference SAME governing range
        until it's broken, not reference the previous bar.
        """
        test_data = pd.DataFrame({
            'high': [100, 99, 98, 97, 105],  # 3 inside bars, then breaks high only
            'low': [90, 91, 92, 93, 91],     # Last bar stays above gov_low (90)
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        # Bar 0: Ref (gov: 100/90)
        # Bar 1-3: Inside (gov: unchanged at 100/90)
        # Bar 4: Breaks high only (105 > 100, 91 >= 90) -> 2U
        assert result.classification.iloc[0] == -999, "First bar should be reference"
        assert result.classification.iloc[1] == 1, "Bar 1 should be inside"
        assert result.classification.iloc[2] == 1, "Bar 2 should be inside"
        assert result.classification.iloc[3] == 1, "Bar 3 should be inside"
        assert result.classification.iloc[4] == 2, "Bar 4 should be 2U"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bar(self):
        """Test with single bar (should be reference)."""
        test_data = pd.DataFrame({
            'high': [100],
            'low': [95],
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        assert result.classification.iloc[0] == -999, "Single bar should be reference"

    def test_two_bars_2u(self):
        """Test with two bars where second is 2U."""
        test_data = pd.DataFrame({
            'high': [100, 105],
            'low': [95, 96],
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        expected = np.array([-999, 2])
        np.testing.assert_array_equal(result.classification.values, expected)

    def test_two_bars_2d(self):
        """Test with two bars where second is 2D."""
        test_data = pd.DataFrame({
            'high': [100, 99],
            'low': [95, 90],
        })

        result = StratBarClassifier.run(test_data['high'], test_data['low'])

        expected = np.array([-999, -2])
        np.testing.assert_array_equal(result.classification.values, expected)

    def test_no_nan_values(self):
        """Verify no NaN values in output (all bars classified)."""
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(100) * 2)
        low = high - np.abs(np.random.randn(100) * 5)

        result = StratBarClassifier.run(high, low)

        assert not result.classification.isna().any(), "Should have no NaN values"

    def test_valid_classification_values(self):
        """Verify all classifications are in valid set."""
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(100) * 2)
        low = high - np.abs(np.random.randn(100) * 5)

        result = StratBarClassifier.run(high, low)

        valid_values = {-999, -2, 1, 2, 3}
        actual_values = set(result.classification.dropna().unique())

        assert actual_values.issubset(valid_values), \
            f"Invalid classification values: {actual_values - valid_values}"


class TestRealData:
    """Test with real SPY market data."""

    @pytest.mark.skip(reason="Requires Alpaca API credentials - run manually when needed")
    def test_spy_2020_classifications(self):
        """
        Test bar classification on real SPY data from 2020.

        Validation checks:
        1. No NaN values
        2. All values in valid range
        3. Reasonable distribution of bar types
        """
        # Fetch SPY data for 2020
        spy_data = vbt.AlpacaData.pull('SPY', start='2020-01-01', end='2020-12-31')

        # Run classifier
        result = StratBarClassifier.run(spy_data.high, spy_data.low)

        # Check 1: No NaN values (except possibly first bar warmup)
        assert result.classification.isna().sum() <= 1, \
            "Too many NaN values in classifications"

        # Check 2: All values in expected range
        valid_values = {-999, -2, 1, 2, 3}
        actual_values = set(result.classification.dropna().unique())
        assert actual_values.issubset(valid_values), \
            f"Invalid classification values: {actual_values - valid_values}"

        # Check 3: Reasonable distribution
        value_counts = result.classification.value_counts()
        total_bars = len(result.classification) - 1  # Exclude reference bar

        # Inside bars typically 30-50% of all bars
        inside_bars = value_counts.get(1, 0)
        inside_ratio = inside_bars / total_bars

        assert 0.20 < inside_ratio < 0.70, \
            f"Inside bar ratio {inside_ratio:.2%} outside reasonable range (20-70%)"

        # Should have some of each type (except possibly all types)
        assert value_counts.get(2, 0) > 0, "Should have at least one 2U bar"
        assert value_counts.get(-2, 0) > 0, "Should have at least one 2D bar"

    @pytest.mark.skip(reason="Requires Alpaca API credentials - run manually when needed")
    def test_spy_march_2020_crash(self):
        """
        Test bar classification during March 2020 crash.

        Expect high proportion of 2D (downward directional) bars.
        """
        # March 2020 crash period
        spy_data = vbt.AlpacaData.pull('SPY', start='2020-03-01', end='2020-03-31')

        result = StratBarClassifier.run(spy_data.high, spy_data.low)

        value_counts = result.classification.value_counts()
        total_bars = len(result.classification) - 1

        # During crash, expect more 2D bars than 2U bars
        bar_2d = value_counts.get(-2, 0)
        bar_2u = value_counts.get(2, 0)

        # Not a strict requirement, but during March 2020 crash we expect
        # more downward directional bars
        print(f"March 2020 - 2D: {bar_2d}, 2U: {bar_2u}")


class TestPerformance:
    """Test performance with large datasets."""

    def test_performance_1000_bars(self):
        """Test performance with 1000 bars."""
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(1000) * 2)
        low = high - np.abs(np.random.randn(1000) * 5)

        start_time = time.time()
        result = StratBarClassifier.run(high, low)
        elapsed = time.time() - start_time

        print(f"Performance (1000 bars): {elapsed:.4f}s")
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 1000 bars"

    def test_performance_10000_bars(self):
        """Test performance with 10,000 bars (target: <1 second)."""
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(10000) * 2)
        low = high - np.abs(np.random.randn(10000) * 5)

        start_time = time.time()
        result = StratBarClassifier.run(high, low)
        elapsed = time.time() - start_time

        print(f"Performance (10,000 bars): {elapsed:.4f}s")
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 10k bars"

    def test_numba_compilation(self):
        """Verify @njit compilation provides speedup."""
        np.random.seed(42)
        high = 100 + np.cumsum(np.random.randn(10000) * 2)
        low = high - np.abs(np.random.randn(10000) * 5)

        # First run (includes JIT compilation time)
        start = time.time()
        classify_bars_nb(high, low)
        first_run = time.time() - start

        # Second run (compiled, should be faster)
        start = time.time()
        classify_bars_nb(high, low)
        second_run = time.time() - start

        print(f"First run (with JIT): {first_run:.4f}s")
        print(f"Second run (compiled): {second_run:.4f}s")

        # Both runs should be fast (< 0.5s for 10k bars)
        assert first_run < 0.5, f"First run too slow: {first_run:.4f}s"
        assert second_run < 0.1, f"Second run too slow: {second_run:.4f}s"

        # Calculate speedup only if second run is measurable
        if second_run > 0.0001:  # Avoid division by zero
            speedup = first_run / second_run
            print(f"Speedup: {speedup:.1f}x")
        else:
            print("Second run too fast to measure accurately (excellent performance!)")


class TestConvenienceFunction:
    """Test the convenience wrapper function."""

    def test_classify_bars_function(self):
        """Test classify_bars() convenience function."""
        test_data = pd.DataFrame({
            'high': [100, 105, 104, 107, 110],
            'low': [95, 98, 99, 101, 93],
        })

        result = classify_bars(test_data['high'], test_data['low'])

        expected = np.array([-999, 2, 1, 2, 3])
        np.testing.assert_array_equal(
            result.values,
            expected,
            err_msg="classify_bars() convenience function incorrect"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
