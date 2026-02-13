"""Unit tests for CandidateScorer."""

import pytest
from strat.ticker_selection.scorer import CandidateScorer, TickerSelectionConfig


@pytest.fixture
def scorer():
    return CandidateScorer(TickerSelectionConfig())


def _make_kwargs(**overrides):
    """Build default scoring kwargs with optional overrides."""
    defaults = dict(
        symbol='TEST',
        pattern_type='3-1-2U',
        signal_type='SETUP',
        direction='CALL',
        timeframe='1D',
        is_bidirectional=True,
        entry_trigger=100.0,
        stop_price=97.0,
        target_price=106.0,
        current_price=99.0,
        tfc_score=3,
        tfc_alignment='3/4 BULLISH',
        tfc_direction='bullish',
        tfc_passes_flexible=True,
        tfc_risk_multiplier=0.5,
        tfc_priority_rank=2,
        atr_percent=2.5,
        dollar_volume=50_000_000,
    )
    defaults.update(overrides)
    return defaults


class TestBasePattern:
    def test_strips_direction_suffix(self, scorer):
        assert scorer._base_pattern('3-1-2U') == '3-1-2'
        assert scorer._base_pattern('2-1-2D') == '2-1-2'
        assert scorer._base_pattern('2D-2U') == '2-2'
        assert scorer._base_pattern('3-2') == '3-2'


class TestTFCScoring:
    def test_perfect_tfc(self, scorer):
        assert scorer._score_tfc(4) == 100

    def test_three_of_four(self, scorer):
        assert scorer._score_tfc(3) == 75

    def test_two_of_four(self, scorer):
        assert scorer._score_tfc(2) == 50

    def test_poor_tfc(self, scorer):
        assert scorer._score_tfc(1) == 0
        assert scorer._score_tfc(0) == 0


class TestATRScoring:
    def test_sweet_spot(self, scorer):
        assert scorer._score_atr(2.5) == 100
        assert scorer._score_atr(3.0) == 100

    def test_high_volatility(self, scorer):
        assert scorer._score_atr(5.0) == 75

    def test_low_volatility(self, scorer):
        assert scorer._score_atr(1.5) == 40

    def test_very_low(self, scorer):
        assert scorer._score_atr(0.5) == 20


class TestCompositeScore:
    def test_score_in_range(self, scorer):
        result = scorer.score(**_make_kwargs())
        assert 0 <= result.composite_score <= 100

    def test_high_quality_candidate(self, scorer):
        result = scorer.score(**_make_kwargs(
            tfc_score=4,
            pattern_type='3-1-2U',
            atr_percent=3.0,
        ))
        assert result.composite_score >= 70

    def test_low_quality_candidate(self, scorer):
        result = scorer.score(**_make_kwargs(
            tfc_score=1,
            pattern_type='3-2',
            atr_percent=0.5,
        ))
        assert result.composite_score < 40

    def test_risk_reward_calculated(self, scorer):
        result = scorer.score(**_make_kwargs(
            entry_trigger=100.0,
            stop_price=97.0,
            target_price=106.0,
        ))
        assert result.risk_reward == pytest.approx(2.0, abs=0.01)

    def test_distance_to_trigger(self, scorer):
        result = scorer.score(**_make_kwargs(
            current_price=99.0,
            entry_trigger=100.0,
        ))
        assert result.distance_to_trigger_pct == pytest.approx(1.0, abs=0.1)

    def test_breakdown_sums_to_composite(self, scorer):
        result = scorer.score(**_make_kwargs())
        bd = result.breakdown
        total = bd.tfc_component + bd.pattern_component + bd.proximity_component + bd.atr_component
        assert abs(total - result.composite_score) < 1.0  # Rounding tolerance
