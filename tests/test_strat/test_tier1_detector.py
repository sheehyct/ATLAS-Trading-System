"""
Tests for Tier1Detector continuation bar filter logic.

Session 83K-5: Comprehensive test coverage for continuation bar filter fix.
The fix allows inside bars (1) to pass without breaking the count,
while still breaking on reversal bars and outside bars (3).

Session 83K-8: Tests updated for look-ahead bias fix.
All patterns are now RETURNED (no filtering). Continuation bars are counted
for analytics only. Tests verify counting logic, not filtering behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from strat.tier1_detector import (
    Tier1Detector, PatternSignal, PatternType, Timeframe
)


class TestContinuationFilterLogic:
    """Test the _apply_continuation_filter method."""

    @pytest.fixture
    def detector(self):
        """Create detector with min 2 continuation bars."""
        return Tier1Detector(min_continuation_bars=2)

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC DataFrame with 10 bars."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
        }, index=dates)

    def create_signal(self, timestamp, direction=1):
        """Helper to create a PatternSignal."""
        pattern_type = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN
        return PatternSignal(
            timestamp=timestamp,
            pattern_type=pattern_type,
            direction=direction,
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            timeframe=Timeframe.DAILY,
            continuation_bars=0,
            is_filtered=False
        )

    def test_counts_consecutive_directional_bars_bullish(self, detector, sample_data):
        """Bullish pattern: 2U -> 2U -> 2U = count 3."""
        # Classifications: [ref, pattern, pattern, pattern, 2U, 2U, 2U, ...]
        classifications = np.array([-999, 3, 1, 2, 2, 2, 2, 1, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 3

    def test_counts_consecutive_directional_bars_bearish(self, detector, sample_data):
        """Bearish pattern: 2D -> 2D -> 2D = count 3."""
        classifications = np.array([-999, 3, 1, -2, -2, -2, -2, 1, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=-1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 3

    def test_inside_bar_allows_continuation(self, detector, sample_data):
        """Inside bars (1) should NOT break count: 2U -> 1 -> 2U -> 2U = count 3."""
        # Key test: Inside bar at position 5 should not break the sequence
        classifications = np.array([-999, 3, 1, 2, 2, 1, 2, 2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Should count: bar 4 (2U) + skip bar 5 (1) + bar 6 (2U) + bar 7 (2U) = 3
        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 3
        assert filtered[0].is_filtered is True

    def test_outside_bar_breaks_continuation(self, detector, sample_data):
        """Outside bars (3) should break count: 2U -> 3 -> 2U = count 1."""
        classifications = np.array([-999, 3, 1, 2, 2, 3, 2, 2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        # Pass signal through filter - it modifies signal even if not returned
        detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Should only count bar 4 (2U), then break at bar 5 (3)
        assert signal.continuation_bars == 1
        assert signal.is_filtered is False

    def test_reversal_bar_breaks_continuation_bullish(self, detector, sample_data):
        """For bullish: 2D reversal bar should break: 2U -> 2D = count 1."""
        classifications = np.array([-999, 3, 1, 2, 2, -2, 2, 2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Should only count bar 4 (2U), then break at bar 5 (2D reversal)
        assert signal.continuation_bars == 1
        assert signal.is_filtered is False

    def test_reversal_bar_breaks_continuation_bearish(self, detector, sample_data):
        """For bearish: 2U reversal bar should break: 2D -> 2U = count 1."""
        classifications = np.array([-999, 3, 1, -2, -2, 2, -2, -2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=-1)

        detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Should only count bar 4 (2D), then break at bar 5 (2U reversal)
        assert signal.continuation_bars == 1
        assert signal.is_filtered is False

    def test_mixed_sequence_with_inside_bars(self, detector, sample_data):
        """Complex sequence: 2U -> 1 -> 2U -> 2D = count 2."""
        # This tests that inside bars are properly skipped
        classifications = np.array([-999, 3, 1, 2, 2, 1, 2, -2, 2, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # bar 4 (2U): count 1
        # bar 5 (1): skip, continue
        # bar 6 (2U): count 2
        # bar 7 (2D): break (reversal)
        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 2

    def test_low_continuation_count_still_returned(self, detector, sample_data):
        """
        Session 83K-8: Patterns with < min_continuation_bars are now RETURNED.
        Continuation bars counted for analytics, but no filtering.
        """
        # Only 1 continuation bar before reversal
        classifications = np.array([-999, 3, 1, 2, 2, -2, 2, 2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Session 83K-8: All patterns returned (no filtering)
        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 1
        # is_filtered is still set for analytics (False = below threshold)
        assert filtered[0].is_filtered is False

    def test_minimum_bars_filter_passes(self, detector, sample_data):
        """Patterns with >= min_continuation_bars should pass."""
        # Exactly 2 continuation bars
        classifications = np.array([-999, 3, 1, 2, 2, 2, -2, 2, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 2
        assert filtered[0].is_filtered is True

    def test_lookahead_capped_at_5_bars(self, detector):
        """Max lookforward should be 5 bars."""
        dates = pd.date_range('2024-01-01', periods=15, freq='D')
        data = pd.DataFrame({
            'Open': [100] * 15,
            'High': [105] * 15,
            'Low': [95] * 15,
            'Close': [102] * 15,
        }, index=dates)

        # All 2U bars after pattern - should cap at 5
        classifications = np.array([-999, 3, 1, 2] + [2] * 11)
        signal = self.create_signal(data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], data, classifications
        )

        # Session 83K-8: Pattern returned (no filtering)
        assert len(filtered) == 1
        # Should cap at 5 even though more 2U bars available
        assert filtered[0].continuation_bars == 5

    def test_all_inside_bars_counts_zero(self, detector, sample_data):
        """
        All inside bars after pattern = 0 continuation (but no break).
        Session 83K-8: Pattern still returned despite 0 continuation bars.
        """
        classifications = np.array([-999, 3, 1, 2, 1, 1, 1, 1, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Session 83K-8: Pattern returned (no filtering)
        assert len(filtered) == 1
        # Inside bars are skipped, no 2U found, count = 0
        assert filtered[0].continuation_bars == 0
        assert filtered[0].is_filtered is False

    def test_empty_signals_list(self, detector, sample_data):
        """Empty signals list should return empty list."""
        classifications = np.array([0] * 10)
        filtered = detector._apply_continuation_filter(
            [], sample_data, classifications
        )
        assert filtered == []

    def test_signal_timestamp_not_found(self, detector, sample_data):
        """Signal with timestamp not in data should be skipped."""
        classifications = np.array([0] * 10)
        bad_timestamp = pd.Timestamp('2025-01-01')  # Not in sample_data
        signal = self.create_signal(bad_timestamp, direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        assert len(filtered) == 0


class TestSessionExample:
    """Test the specific example from Session 83K-4."""

    def test_session_83k4_example(self):
        """
        Trace through Session 83K-4 example:
        Sequence: 3-1-2D-2D-1-2U-2D-2D-1-2U

        Pattern 1 (3-1-2D at bars 0-1-2, bearish):
        - Bar 3 (2D): COUNT = 1
        - Bar 4 (1): ALLOW (skip)
        - Bar 5 (2U): BREAK (reversal for bearish)
        Result: 1 continuation bar

        Session 83K-8: Pattern now RETURNED (no filtering) with count=1.
        """
        detector = Tier1Detector(min_continuation_bars=2)

        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
        }, index=dates)

        # Sequence: 3-1-2D-2D-1-2U-2D-2D-1-2U
        # Index:    0-1- 2- 3-4- 5- 6- 7-8- 9
        classifications = np.array([3, 1, -2, -2, 1, 2, -2, -2, 1, 2])

        # Pattern at index 2 (3-1-2D at bars 0-1-2), bearish
        signal = PatternSignal(
            timestamp=dates[2],
            pattern_type=PatternType.PATTERN_312_DOWN,
            direction=-1,
            entry_price=95.0,
            stop_price=105.0,
            target_price=85.0,
            timeframe=Timeframe.DAILY,
            continuation_bars=0,
            is_filtered=False
        )

        filtered = detector._apply_continuation_filter(
            [signal], data, classifications
        )

        # Session 83K-8: Pattern returned (no filtering)
        assert len(filtered) == 1

        # Bar 3 (2D): COUNT = 1
        # Bar 4 (1): ALLOW (skip)
        # Bar 5 (2U): BREAK (reversal for bearish)
        assert filtered[0].continuation_bars == 1
        # is_filtered = False (below threshold) but pattern still returned
        assert filtered[0].is_filtered is False


class TestSession83K8FilterRemoval:
    """
    Session 83K-8: Tests verifying look-ahead bias fix.
    All patterns are now returned; continuation bars for analytics only.
    """

    @pytest.fixture
    def detector(self):
        """Create detector with min 2 continuation bars."""
        return Tier1Detector(min_continuation_bars=2)

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC DataFrame with 15 bars."""
        dates = pd.date_range('2024-01-01', periods=15, freq='D')
        return pd.DataFrame({
            'Open': [100] * 15,
            'High': [105] * 15,
            'Low': [95] * 15,
            'Close': [102] * 15,
        }, index=dates)

    def create_signal(self, timestamp, direction=1):
        """Helper to create a PatternSignal."""
        pattern_type = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN
        return PatternSignal(
            timestamp=timestamp,
            pattern_type=pattern_type,
            direction=direction,
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            timeframe=Timeframe.DAILY,
            continuation_bars=0,
            is_filtered=False
        )

    def test_all_patterns_returned_no_filtering(self, detector, sample_data):
        """
        CRITICAL: Verify that ALL patterns are returned, not filtered.
        This is the core fix - removing look-ahead bias.
        """
        # Create 3 signals with different continuation counts
        classifications = np.array([
            -999, 3, 1, 2,  # Pattern 1 at index 3
            -2, 1, 1, 1,     # Count 0 for pattern 1 (immediate reversal)
            3, 1, 2, 2,      # Pattern 2 at index 10
            2, 2, 2          # Count 5 for pattern 2
        ])

        signals = [
            self.create_signal(sample_data.index[3], direction=1),  # Will have count=0
            self.create_signal(sample_data.index[10], direction=1),  # Will have count>=2
        ]

        filtered = detector._apply_continuation_filter(
            signals, sample_data, classifications
        )

        # ALL signals should be returned (Session 83K-8 fix)
        assert len(filtered) == 2
        # Both patterns returned regardless of count
        assert filtered[0].continuation_bars == 0  # Low count - still returned
        assert filtered[1].continuation_bars >= 2   # High count - also returned

    def test_continuation_bars_analytics_only(self, detector, sample_data):
        """
        Verify continuation_bars is computed for analytics but not used for filtering.
        Pattern with 0 continuation bars should be returned.
        """
        # Signal that would have been REJECTED under old behavior
        # Immediate reversal = 0 continuation bars
        classifications = np.array([-999, 3, 1, 2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1])
        signal = self.create_signal(sample_data.index[3], direction=1)

        filtered = detector._apply_continuation_filter(
            [signal], sample_data, classifications
        )

        # Pattern returned with count recorded (Session 83K-8 fix)
        assert len(filtered) == 1
        assert filtered[0].continuation_bars == 0  # Count is captured
        assert filtered[0].is_filtered is False     # Below threshold for analytics
        # Pattern is NOT rejected based on count

    def test_detector_accepts_min_bars_zero(self):
        """
        Session 83K-8: ValueError removed for min_continuation_bars < 2.
        Detector should accept any value for backward compatibility.
        """
        # This would have raised ValueError before Session 83K-8
        detector = Tier1Detector(min_continuation_bars=0)
        assert detector.min_continuation_bars == 0

        detector = Tier1Detector(min_continuation_bars=1)
        assert detector.min_continuation_bars == 1
