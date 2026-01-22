"""
Tests for strat/timeframe_continuity_adapter.py

Covers:
- ContinuityAssessment dataclass and alignment_label method
- strength_to_risk_multiplier function
- strength_to_priority_rank function
- TimeframeContinuityAdapter initialization and evaluate method

Session EQUITY-77: Extended test coverage for TFC adapter module.
"""

import pytest
from unittest.mock import patch

pd = pytest.importorskip("pandas")

from strat.timeframe_continuity_adapter import (
    ContinuityAssessment,
    TimeframeContinuityAdapter,
    strength_to_priority_rank,
    strength_to_risk_multiplier,
)


# =============================================================================
# Original Tests (Preserved)
# =============================================================================

def test_strength_mappings():
    assert strength_to_risk_multiplier(5) == 1.0
    assert strength_to_risk_multiplier(3) == 0.5
    assert strength_to_risk_multiplier(1) == 0.0

    assert strength_to_priority_rank(5) == 1
    assert strength_to_priority_rank(3) == 2
    assert strength_to_priority_rank(0) == 4


def test_adapter_returns_continuity_assessment():
    adapter = TimeframeContinuityAdapter(timeframes=["1M", "1W", "1D"])

    def _fetch(_: str):
        return pd.DataFrame(
            {
                "High": [100, 102, 104, 106],
                "Low": [99, 101, 103, 105],
            }
        )

    assessment = adapter.evaluate(
        fetcher=_fetch,
        detection_timeframe="1D",
        direction="bullish",
    )

    assert assessment.passes_flexible is True
    assert assessment.strength == 3
    assert assessment.risk_multiplier == 0.5
    assert assessment.priority_rank == 2


# =============================================================================
# ContinuityAssessment Tests (EQUITY-77)
# =============================================================================

class TestContinuityAssessmentDefaults:
    """Tests for ContinuityAssessment default values."""

    def test_default_strength(self):
        """Default strength is 0."""
        ca = ContinuityAssessment()
        assert ca.strength == 0

    def test_default_passes_flexible(self):
        """Default passes_flexible is False."""
        ca = ContinuityAssessment()
        assert ca.passes_flexible is False

    def test_default_risk_multiplier(self):
        """Default risk_multiplier is 1.0."""
        ca = ContinuityAssessment()
        assert ca.risk_multiplier == 1.0

    def test_default_priority_rank(self):
        """Default priority_rank is 5."""
        ca = ContinuityAssessment()
        assert ca.priority_rank == 5


class TestContinuityAssessmentAlignmentLabel:
    """Tests for ContinuityAssessment.alignment_label method."""

    def test_full_bullish_alignment(self):
        """Full bullish alignment label."""
        ca = ContinuityAssessment(
            strength=4,
            required_timeframes=['1M', '1W', '1D', '1H'],
            direction='bullish',
        )
        assert ca.alignment_label() == '4/4 BULLISH'

    def test_partial_bearish_alignment(self):
        """Partial bearish alignment label."""
        ca = ContinuityAssessment(
            strength=2,
            required_timeframes=['1W', '1D', '1H'],
            direction='bearish',
        )
        assert ca.alignment_label() == '2/3 BEARISH'

    def test_no_direction_label(self):
        """Label without direction."""
        ca = ContinuityAssessment(
            strength=3,
            required_timeframes=['1M', '1W', '1D', '4H'],
            direction='',
        )
        assert ca.alignment_label() == '3/4'

    def test_none_required_timeframes(self):
        """Label with None required_timeframes."""
        ca = ContinuityAssessment(
            strength=0,
            required_timeframes=None,
            direction='bullish',
        )
        assert '0/1' in ca.alignment_label()


# =============================================================================
# strength_to_risk_multiplier Tests (EQUITY-77)
# =============================================================================

class TestStrengthToRiskMultiplierExtended:
    """Extended tests for strength_to_risk_multiplier."""

    def test_strength_4_returns_1(self):
        """Strength 4 returns 1.0."""
        assert strength_to_risk_multiplier(4) == 1.0

    def test_strength_2_returns_0(self):
        """Strength 2 returns 0.0."""
        assert strength_to_risk_multiplier(2) == 0.0

    def test_strength_0_returns_0(self):
        """Strength 0 returns 0.0."""
        assert strength_to_risk_multiplier(0) == 0.0

    def test_negative_strength_returns_0(self):
        """Negative strength returns 0.0."""
        assert strength_to_risk_multiplier(-1) == 0.0


# =============================================================================
# strength_to_priority_rank Tests (EQUITY-77)
# =============================================================================

class TestStrengthToPriorityRankExtended:
    """Extended tests for strength_to_priority_rank."""

    def test_strength_4_returns_rank_1(self):
        """Strength 4 returns rank 1."""
        assert strength_to_priority_rank(4) == 1

    def test_strength_2_returns_rank_3(self):
        """Strength 2 returns rank 3."""
        assert strength_to_priority_rank(2) == 3

    def test_strength_1_returns_rank_4(self):
        """Strength 1 returns rank 4."""
        assert strength_to_priority_rank(1) == 4

    def test_negative_strength_returns_rank_4(self):
        """Negative strength returns rank 4."""
        assert strength_to_priority_rank(-1) == 4


# =============================================================================
# TimeframeContinuityAdapter Tests (EQUITY-77)
# =============================================================================

class TestTimeframeContinuityAdapterInit:
    """Tests for TimeframeContinuityAdapter initialization."""

    def test_default_timeframes(self):
        """Default timeframes are set."""
        adapter = TimeframeContinuityAdapter()
        assert adapter.timeframes == ["1M", "1W", "1D", "4H", "1H"]

    def test_custom_timeframes(self):
        """Custom timeframes are stored."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D", "4H"])
        assert adapter.timeframes == ["1D", "4H"]

    def test_default_min_strength(self):
        """Default min_strength is 3."""
        adapter = TimeframeContinuityAdapter()
        assert adapter.min_strength == 3

    def test_custom_min_strength(self):
        """Custom min_strength is stored."""
        adapter = TimeframeContinuityAdapter(min_strength=2)
        assert adapter.min_strength == 2


class TestTimeframeContinuityAdapterEvaluate:
    """Tests for TimeframeContinuityAdapter.evaluate method."""

    def test_evaluate_empty_data(self):
        """evaluate handles empty fetcher gracefully."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D"])

        def empty_fetcher(_):
            return None

        result = adapter.evaluate(
            fetcher=empty_fetcher,
            detection_timeframe='1D',
            direction='bullish',
        )

        assert result.strength == 0
        assert result.passes_flexible is False

    def test_evaluate_normalizes_direction(self):
        """Direction is normalized to lowercase."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D"])

        def fetcher(_):
            return pd.DataFrame({'High': [100], 'Low': [99]})

        with patch.object(adapter.checker, 'check_flexible_continuity') as mock:
            mock.return_value = {'strength': 1, 'passes_flexible': False}
            result = adapter.evaluate(
                fetcher=fetcher,
                detection_timeframe='1D',
                direction='BULLISH',
            )

        assert result.direction == 'bullish'

    def test_evaluate_normalizes_timeframe(self):
        """Detection timeframe is normalized to uppercase."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D"])

        def fetcher(_):
            return pd.DataFrame({'High': [100], 'Low': [99]})

        with patch.object(adapter.checker, 'check_flexible_continuity') as mock:
            mock.return_value = {'strength': 1, 'passes_flexible': False}
            result = adapter.evaluate(
                fetcher=fetcher,
                detection_timeframe='1d',
                direction='bullish',
            )

        assert result.detection_timeframe == '1D'

    def test_evaluate_uses_aliases(self):
        """Timeframe aliases are passed to fetcher."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D"])
        called_with = []

        def tracking_fetcher(tf):
            called_with.append(tf)
            return pd.DataFrame({'High': [100], 'Low': [99]})

        with patch.object(adapter.checker, 'check_flexible_continuity') as mock:
            mock.return_value = {'strength': 1, 'passes_flexible': False}
            adapter.evaluate(
                fetcher=tracking_fetcher,
                detection_timeframe='1D',
                direction='bullish',
                timeframe_aliases={'1D': 'day'},
            )

        assert 'day' in called_with

    def test_evaluate_missing_columns(self):
        """Fetcher returning incomplete data is handled."""
        adapter = TimeframeContinuityAdapter(timeframes=["1D"])

        def bad_fetcher(_):
            return pd.DataFrame({'Close': [100]})  # Missing High/Low

        result = adapter.evaluate(
            fetcher=bad_fetcher,
            detection_timeframe='1D',
            direction='bullish',
        )

        assert result.strength == 0
