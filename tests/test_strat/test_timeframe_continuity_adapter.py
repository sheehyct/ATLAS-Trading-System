import pytest

pd = pytest.importorskip("pandas")

from strat.timeframe_continuity_adapter import (
    TimeframeContinuityAdapter,
    strength_to_priority_rank,
    strength_to_risk_multiplier,
)


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
