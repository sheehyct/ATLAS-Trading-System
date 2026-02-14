"""
Composite scoring for ticker selection candidates.

Scores 0-100 with configurable weights across four dimensions:
  TFC alignment (40%), Pattern quality (25%),
  Proximity to trigger (20%), ATR volatility (15%).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from strat.ticker_selection.config import TickerSelectionConfig

logger = logging.getLogger(__name__)

# Pattern quality scores (higher = better setup)
_PATTERN_SCORES: Dict[str, float] = {
    '3-1-2': 100,
    '2-1-2': 85,
    '2-2': 70,       # Reversal variant
    '3-2-2': 55,
    '3-2': 40,
}


@dataclass
class ScoringBreakdown:
    """Detailed scoring breakdown for a candidate."""
    tfc_component: float = 0.0
    pattern_component: float = 0.0
    proximity_component: float = 0.0
    atr_component: float = 0.0
    composite: float = 0.0


@dataclass
class ScoredCandidate:
    """A fully scored candidate with all metadata."""
    symbol: str
    composite_score: float
    rank: int = 0

    # Pattern info
    pattern_type: str = ''
    base_pattern: str = ''
    signal_type: str = ''
    direction: str = ''
    timeframe: str = ''
    is_bidirectional: bool = False

    # Price levels
    entry_trigger: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    current_price: float = 0.0
    distance_to_trigger_pct: float = 0.0

    # TFC
    tfc_score: int = 0
    tfc_alignment: str = ''
    tfc_direction: str = ''
    tfc_passes_flexible: bool = False
    tfc_risk_multiplier: float = 1.0
    tfc_priority_rank: int = 5

    # Metrics
    atr_percent: float = 0.0
    dollar_volume: float = 0.0
    risk_reward: float = 0.0

    # Scoring detail
    breakdown: ScoringBreakdown = field(default_factory=ScoringBreakdown)


class CandidateScorer:
    """Scores and ranks ticker selection candidates."""

    def __init__(self, config: Optional[TickerSelectionConfig] = None):
        self.config = config or TickerSelectionConfig()

    def score(
        self,
        symbol: str,
        pattern_type: str,
        signal_type: str,
        direction: str,
        timeframe: str,
        is_bidirectional: bool,
        entry_trigger: float,
        stop_price: float,
        target_price: float,
        current_price: float,
        tfc_score: int,
        tfc_alignment: str,
        tfc_direction: str,
        tfc_passes_flexible: bool,
        tfc_risk_multiplier: float,
        tfc_priority_rank: int,
        atr_percent: float,
        dollar_volume: float,
    ) -> ScoredCandidate:
        """
        Compute composite score for a candidate.

        Returns a ScoredCandidate with breakdown.
        """
        # Extract base pattern (strip direction suffix like U/D)
        base = self._base_pattern(pattern_type)

        # TFC component: 4/4=100, 3/4=75, 2/4=50, <2=0
        tfc_raw = self._score_tfc(tfc_score)

        # Pattern component
        pattern_raw = _PATTERN_SCORES.get(base, 30)

        # Continuations score very low (not typically traded)
        if self._is_continuation(pattern_type):
            pattern_raw = 15  # Minimal score, effectively filtered by ranking

        # Proximity component
        proximity_raw = self._score_proximity(
            current_price, entry_trigger, atr_percent, direction
        )

        # ATR component
        atr_raw = self._score_atr(atr_percent)

        # Weighted composite
        cfg = self.config
        composite = (
            tfc_raw * cfg.weight_tfc
            + pattern_raw * cfg.weight_pattern
            + proximity_raw * cfg.weight_proximity
            + atr_raw * cfg.weight_atr
        )

        # Risk-reward bonus
        rr = 0.0
        if stop_price and entry_trigger:
            risk = abs(entry_trigger - stop_price)
            reward = abs(target_price - entry_trigger) if target_price else 0.0
            rr = reward / risk if risk > 0 else 0.0

        # Distance to trigger
        dist_pct = 0.0
        if entry_trigger and current_price:
            dist_pct = abs(entry_trigger - current_price) / current_price * 100

        breakdown = ScoringBreakdown(
            tfc_component=round(tfc_raw * cfg.weight_tfc, 1),
            pattern_component=round(pattern_raw * cfg.weight_pattern, 1),
            proximity_component=round(proximity_raw * cfg.weight_proximity, 1),
            atr_component=round(atr_raw * cfg.weight_atr, 1),
            composite=round(composite, 1),
        )

        return ScoredCandidate(
            symbol=symbol,
            composite_score=round(composite, 1),
            pattern_type=pattern_type,
            base_pattern=base,
            signal_type=signal_type,
            direction=direction,
            timeframe=timeframe,
            is_bidirectional=is_bidirectional,
            entry_trigger=entry_trigger,
            stop_price=stop_price,
            target_price=target_price,
            current_price=current_price,
            distance_to_trigger_pct=round(dist_pct, 2),
            tfc_score=tfc_score,
            tfc_alignment=tfc_alignment,
            tfc_direction=tfc_direction,
            tfc_passes_flexible=tfc_passes_flexible,
            tfc_risk_multiplier=tfc_risk_multiplier,
            tfc_priority_rank=tfc_priority_rank,
            atr_percent=atr_percent,
            dollar_volume=dollar_volume,
            risk_reward=round(rr, 2),
            breakdown=breakdown,
        )

    @staticmethod
    def _base_pattern(pattern_type: str) -> str:
        """Strip direction suffix to get base pattern name.

        Examples: "3-1-2U" -> "3-1-2", "2D-2U" -> "2-2"
        """
        return re.sub(r'(\d)[UD]', r'\1', pattern_type)

    @staticmethod
    def _is_continuation(pattern_type: str) -> bool:
        """Check if pattern is a continuation (same direction, e.g., 2U-2U, 2D-2D).

        Continuations are NOT typically traded in STRAT methodology.
        Only reversals (direction change, e.g., 2D-2U, 2U-2D) create new entries.
        """
        segments = pattern_type.split('-')
        directions = [s[-1] for s in segments if s.endswith(('U', 'D')) and len(s) >= 2]
        if len(directions) >= 2:
            return len(set(directions)) == 1
        return False

    @staticmethod
    def _score_tfc(tfc_score: int) -> float:
        """Score TFC alignment: 4/4=100, 3/4=75, 2/4=50, <2=0."""
        if tfc_score >= 4:
            return 100
        elif tfc_score == 3:
            return 75
        elif tfc_score == 2:
            return 50
        return 0

    @staticmethod
    def _score_proximity(current_price: float, trigger: float, atr_pct: float, direction: str = 'CALL') -> float:
        """
        Score proximity to trigger.

        Sweet spot: 0.2-0.8 ATR from trigger = 100.
        Already triggered (price past trigger) = 0.
        Too far (> 1.5 ATR) = ramp down.

        Args:
            direction: 'CALL' or 'PUT' -- determines which side of trigger
                       means "already triggered".
        """
        if not trigger or not current_price or not atr_pct:
            return 50  # Neutral if missing data

        atr_dollars = current_price * atr_pct / 100
        if atr_dollars <= 0:
            return 50

        distance = abs(current_price - trigger)
        distance_in_atrs = distance / atr_dollars

        # Already past the trigger
        if direction == 'CALL' and current_price > trigger:
            return 0
        elif direction == 'PUT' and current_price < trigger:
            return 0

        if 0.2 <= distance_in_atrs <= 0.8:
            return 100
        elif distance_in_atrs < 0.2:
            return 60  # Very close, might gap through
        elif distance_in_atrs <= 1.5:
            # Linear ramp from 100 at 0.8 to 30 at 1.5
            return max(30, 100 - (distance_in_atrs - 0.8) * 100)
        else:
            return 10  # Far from trigger

    @staticmethod
    def _score_atr(atr_pct: float) -> float:
        """Score ATR%: 2-4% sweet spot = 100, scale down outside."""
        if 2.0 <= atr_pct <= 4.0:
            return 100
        elif 4.0 < atr_pct <= 6.0:
            return 75
        elif 1.0 <= atr_pct < 2.0:
            return 40
        elif atr_pct > 6.0:
            return 50  # Very volatile, still tradable
        return 20  # Below 1% - too quiet
