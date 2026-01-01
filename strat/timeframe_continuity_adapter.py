"""
Adapter utilities for applying STRAT timeframe continuity checks consistently
across equities, crypto, backtests, and paper trading.

The adapter wraps :class:`TimeframeContinuityChecker` and adds deterministic
risk-multiplier and priority mapping so that sizing/alert ordering uses the
same rules everywhere ("single source of truth").
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from strat.timeframe_continuity import TimeframeContinuityChecker


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ContinuityAssessment:
    """Structured continuity result with sizing/prioritization hints."""

    strength: int = 0
    passes_flexible: bool = False
    aligned_timeframes: List[str] = None
    required_timeframes: List[str] = None
    direction: str = ""
    detection_timeframe: str = ""
    risk_multiplier: float = 1.0
    priority_rank: int = 5

    def alignment_label(self) -> str:
        """Return a human readable alignment string (e.g., "3/4 BULLISH")."""

        total = len(self.required_timeframes or []) or 1
        suffix = self.direction.upper() if self.direction else ""
        return f"{self.strength}/{total} {suffix}".strip()


# ---------------------------------------------------------------------------
# Mapping utilities
# ---------------------------------------------------------------------------


def strength_to_risk_multiplier(strength: int) -> float:
    """Map a continuity strength score to a position-size multiplier."""

    if strength >= 4:
        return 1.0
    if strength == 3:
        return 0.5
    return 0.0


def strength_to_priority_rank(strength: int) -> int:
    """Map continuity strength to a sortable priority rank (1 = highest)."""

    if strength >= 4:
        return 1
    if strength == 3:
        return 2
    if strength == 2:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TimeframeContinuityAdapter:
    """Compute STRAT timeframe continuity with consistent risk mapping."""

    def __init__(
        self,
        timeframes: Optional[Iterable[str]] = None,
        min_strength: int = 3,
    ) -> None:
        self.timeframes = list(timeframes or ["1M", "1W", "1D", "4H", "1H"])
        self.min_strength = min_strength
        self.checker = TimeframeContinuityChecker(timeframes=self.timeframes)

    def evaluate(
        self,
        fetcher: Callable[[str], Optional[pd.DataFrame]],
        detection_timeframe: str,
        direction: str,
        timeframe_aliases: Optional[Dict[str, str]] = None,
    ) -> ContinuityAssessment:
        """
        Fetch OHLC data, run continuity, and attach risk/priority metadata.

        Args:
            fetcher: Callable returning a DataFrame with High/Low columns.
            detection_timeframe: Timeframe where the pattern was detected.
            direction: "bullish" or "bearish".
            timeframe_aliases: Optional mapping from canonical timeframe (e.g.,
                "1W") to fetcher-specific values (e.g., "1w" for crypto).
        """

        direction = direction.lower()
        detection_tf = detection_timeframe.upper()

        high_dict: Dict[str, pd.Series] = {}
        low_dict: Dict[str, pd.Series] = {}

        for tf in self.timeframes:
            fetch_tf = timeframe_aliases.get(tf, tf) if timeframe_aliases else tf
            df = fetcher(fetch_tf)
            if df is None or df.empty or "High" not in df.columns or "Low" not in df.columns:
                continue
            high_dict[tf] = df["High"]
            low_dict[tf] = df["Low"]

        if not high_dict or not low_dict:
            return ContinuityAssessment(
                strength=0,
                passes_flexible=False,
                aligned_timeframes=[],
                required_timeframes=self.timeframes,
                direction=direction,
                detection_timeframe=detection_tf,
                risk_multiplier=strength_to_risk_multiplier(0),
                priority_rank=strength_to_priority_rank(0),
            )

        continuity = self.checker.check_flexible_continuity(
            high_dict=high_dict,
            low_dict=low_dict,
            direction=direction,
            min_strength=self.min_strength,
            detection_timeframe=detection_tf,
        )

        strength = continuity.get("strength", 0)
        required = continuity.get("required_timeframes", self.timeframes)

        return ContinuityAssessment(
            strength=strength,
            passes_flexible=continuity.get("passes_flexible", False),
            aligned_timeframes=continuity.get("aligned_timeframes", []),
            required_timeframes=required,
            direction=direction,
            detection_timeframe=detection_tf,
            risk_multiplier=strength_to_risk_multiplier(strength),
            priority_rank=strength_to_priority_rank(strength),
        )

