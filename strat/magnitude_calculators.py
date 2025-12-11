"""
Magnitude Calculation Strategies for 3-2 Patterns.

Session 83K-62: Implements four strategies for comparison backtesting.

Options:
    A: Previous Outside Bar - Current implementation (~1.41 R:R)
    B-N2: N-bar Swing Pivot with N=2 + 1.5x fallback
    B-N3: N-bar Swing Pivot with N=3 + 1.5x fallback
    C: Always 1.5x R:R measured move

Usage:
    from strat.magnitude_calculators import (
        OptionA_PreviousOutsideBar,
        OptionB_SwingPivot,
        OptionC_MeasuredMove,
    )

    calculator = OptionB_SwingPivot(n_bars=2)
    result = calculator.calculate_target(
        entry_price=300.0,
        stop_price=295.0,
        direction=1,
        high=high_array,
        low=low_array,
        classifications=class_array,
        pattern_idx=50
    )
    print(f"Target: {result.target_price}, Method: {result.method_used}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class MagnitudeResult:
    """Result of magnitude calculation."""

    target_price: float
    method_used: str  # 'previous_outside', 'swing_pivot', 'measured_move'
    rr_ratio: float  # Risk-reward ratio
    lookback_distance: Optional[int] = None  # Bars looked back to find reference

    def __repr__(self) -> str:
        return (
            f"MagnitudeResult(target={self.target_price:.2f}, "
            f"method='{self.method_used}', R:R={self.rr_ratio:.2f})"
        )


# -----------------------------------------------------------------------------
# Numba Helper Functions
# -----------------------------------------------------------------------------

@njit
def validate_target_geometry_nb(
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: int
) -> bool:
    """
    Validate that target price creates profitable geometry.

    For bullish patterns: target must be ABOVE entry
    For bearish patterns: target must be BELOW entry

    Parameters
    ----------
    entry_price : float
        Entry price for the trade
    stop_price : float
        Stop loss price
    target_price : float
        Proposed magnitude target price
    direction : int
        1 for bullish, -1 for bearish

    Returns
    -------
    bool
        True if geometry is valid, False if inverted
    """
    if direction == 1:  # Bullish - target must be ABOVE entry
        return target_price > entry_price
    elif direction == -1:  # Bearish - target must be BELOW entry
        return target_price < entry_price
    else:
        return False  # Invalid direction


@njit
def calculate_measured_move_nb(
    entry_price: float,
    stop_price: float,
    direction: int,
    multiplier: float = 1.5
) -> float:
    """
    Calculate measured move target using stop distance projected from entry.

    This is the fallback when previous directional bars create inverted geometry
    or when no valid reference bar is found.

    Parameters
    ----------
    entry_price : float
        Entry price for the trade
    stop_price : float
        Stop loss price
    direction : int
        1 for bullish, -1 for bearish
    multiplier : float
        Risk-reward multiplier (default 1.5 for 1.5:1 R:R)

    Returns
    -------
    float
        Measured move target price

    Examples
    --------
    Bullish (entry $300, stop $295):
        stop_distance = $5.00
        target = $300 + ($5.00 * 1.5) = $307.50

    Bearish (entry $300, stop $305):
        stop_distance = $5.00
        target = $300 - ($5.00 * 1.5) = $292.50
    """
    stop_distance = abs(entry_price - stop_price)

    if direction == 1:  # Bullish - project upward
        return entry_price + (stop_distance * multiplier)
    elif direction == -1:  # Bearish - project downward
        return entry_price - (stop_distance * multiplier)
    else:
        return entry_price  # Fallback for invalid direction


@njit
def calculate_rr_ratio_nb(
    entry_price: float,
    stop_price: float,
    target_price: float
) -> float:
    """Calculate risk-reward ratio."""
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    if risk > 0:
        return reward / risk
    return 0.0


@njit
def find_previous_outside_bar_nb(
    classifications: np.ndarray,
    start_idx: int,
    max_lookback: int = 100
) -> int:
    """
    Find the first previous outside bar (abs(classification) == 3).

    Parameters
    ----------
    classifications : np.ndarray
        Bar classifications array
    start_idx : int
        Index to start searching backwards from
    max_lookback : int
        Maximum bars to search backwards

    Returns
    -------
    int
        Index of previous outside bar, or -1 if not found
    """
    earliest_idx = max(0, start_idx - max_lookback)

    for j in range(start_idx, earliest_idx - 1, -1):
        if abs(classifications[j]) == 3:
            return j

    return -1  # Not found


@njit
def find_swing_high_nb(
    high: np.ndarray,
    start_idx: int,
    threshold: float,
    n_bars: int,
    max_lookback: int
) -> int:
    """
    Find first swing high above threshold, searching backwards.

    A swing high at index i requires:
    - high[i] > high[i-n] for all n in [1, n_bars]
    - high[i] > high[i+n] for all n in [1, n_bars]
    - high[i] > threshold

    Parameters
    ----------
    high : np.ndarray
        Array of bar high prices
    start_idx : int
        Index to start searching backwards from
    threshold : float
        Minimum high price (swing must be above this)
    n_bars : int
        Number of bars required on each side for swing confirmation
    max_lookback : int
        Maximum bars to search backwards

    Returns
    -------
    int
        Index of swing high, or -1 if not found
    """
    # Need at least n_bars on each side for valid swing
    earliest_idx = max(n_bars, start_idx - max_lookback)
    latest_valid = len(high) - n_bars - 1

    for i in range(start_idx - n_bars - 1, earliest_idx - 1, -1):
        # Skip if too close to end of array
        if i > latest_valid:
            continue

        # Check if bar i is a swing high
        is_swing = True

        # Check N bars before (must be lower than swing high)
        for j in range(1, n_bars + 1):
            if i - j < 0 or high[i] <= high[i - j]:
                is_swing = False
                break

        if not is_swing:
            continue

        # Check N bars after (must be lower than swing high)
        for j in range(1, n_bars + 1):
            if i + j >= len(high) or high[i] <= high[i + j]:
                is_swing = False
                break

        if not is_swing:
            continue

        # Check threshold
        if high[i] > threshold:
            return i

    return -1  # Not found


@njit
def find_swing_low_nb(
    low: np.ndarray,
    start_idx: int,
    threshold: float,
    n_bars: int,
    max_lookback: int
) -> int:
    """
    Find first swing low below threshold, searching backwards.

    A swing low at index i requires:
    - low[i] < low[i-n] for all n in [1, n_bars]
    - low[i] < low[i+n] for all n in [1, n_bars]
    - low[i] < threshold

    Parameters
    ----------
    low : np.ndarray
        Array of bar low prices
    start_idx : int
        Index to start searching backwards from
    threshold : float
        Maximum low price (swing must be below this)
    n_bars : int
        Number of bars required on each side for swing confirmation
    max_lookback : int
        Maximum bars to search backwards

    Returns
    -------
    int
        Index of swing low, or -1 if not found
    """
    # Need at least n_bars on each side for valid swing
    earliest_idx = max(n_bars, start_idx - max_lookback)
    latest_valid = len(low) - n_bars - 1

    for i in range(start_idx - n_bars - 1, earliest_idx - 1, -1):
        # Skip if too close to end of array
        if i > latest_valid:
            continue

        # Check if bar i is a swing low
        is_swing = True

        # Check N bars before (must be higher than swing low)
        for j in range(1, n_bars + 1):
            if i - j < 0 or low[i] >= low[i - j]:
                is_swing = False
                break

        if not is_swing:
            continue

        # Check N bars after (must be higher than swing low)
        for j in range(1, n_bars + 1):
            if i + j >= len(low) or low[i] >= low[i + j]:
                is_swing = False
                break

        if not is_swing:
            continue

        # Check threshold
        if low[i] < threshold:
            return i

    return -1  # Not found


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------

class MagnitudeCalculator(ABC):
    """Abstract base class for magnitude calculation strategies."""

    @abstractmethod
    def calculate_target(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        classifications: np.ndarray,
        pattern_idx: int
    ) -> MagnitudeResult:
        """
        Calculate target price for a 3-2 pattern.

        Parameters
        ----------
        entry_price : float
            Entry price (break of outside bar high/low)
        stop_price : float
            Stop loss price (opposite side of outside bar)
        direction : int
            1 for bullish (3-2U), -1 for bearish (3-2D)
        high : np.ndarray
            Array of bar high prices
        low : np.ndarray
            Array of bar low prices
        classifications : np.ndarray
            Bar classifications from bar_classifier.py
        pattern_idx : int
            Index of the trigger bar (the "2" bar in 3-2)

        Returns
        -------
        MagnitudeResult
            Target price, method used, and R:R ratio
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass


# -----------------------------------------------------------------------------
# Option C: Always 1.5x Measured Move (Simplest)
# -----------------------------------------------------------------------------

class OptionC_MeasuredMove(MagnitudeCalculator):
    """
    Option C: Always 1.5x R:R Measured Move.

    Simple, consistent approach:
    - Target = entry + (entry - stop) * multiplier * direction
    - No lookback required
    - Deterministic R:R for every trade
    """

    def __init__(self, multiplier: float = 1.5):
        """
        Initialize with R:R multiplier.

        Parameters
        ----------
        multiplier : float
            Risk-reward multiplier (default 1.5)
        """
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"Option_C_MeasuredMove_{self.multiplier}x"

    def calculate_target(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        classifications: np.ndarray,
        pattern_idx: int
    ) -> MagnitudeResult:
        """Calculate target using measured move."""
        target = calculate_measured_move_nb(
            entry_price, stop_price, direction, self.multiplier
        )
        rr = calculate_rr_ratio_nb(entry_price, stop_price, target)

        return MagnitudeResult(
            target_price=target,
            method_used="measured_move",
            rr_ratio=rr,
            lookback_distance=None
        )


# -----------------------------------------------------------------------------
# Option A: Previous Outside Bar (Current Implementation)
# -----------------------------------------------------------------------------

class OptionA_PreviousOutsideBar(MagnitudeCalculator):
    """
    Option A: Previous Outside Bar Logic (Current Implementation).

    1. Look backwards from pattern_idx-2 for first outside bar (abs(class)==3)
    2. For bullish (3-2U): target = high[prev_outside_idx]
    3. For bearish (3-2D): target = low[prev_outside_idx]
    4. Geometry validation: target must be in profit direction
    5. Fallback: 1.5x measured move if geometry invalid or no prior bar found

    This wraps the existing logic from pattern_detector.py:739-754.
    """

    def __init__(self, max_lookback: int = 100, fallback_multiplier: float = 1.5):
        """
        Initialize with lookback limit and fallback multiplier.

        Parameters
        ----------
        max_lookback : int
            Maximum bars to search backwards for outside bar
        fallback_multiplier : float
            R:R multiplier for measured move fallback
        """
        self.max_lookback = max_lookback
        self.fallback_multiplier = fallback_multiplier

    @property
    def name(self) -> str:
        return "Option_A_PreviousOutsideBar"

    def calculate_target(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        classifications: np.ndarray,
        pattern_idx: int
    ) -> MagnitudeResult:
        """Calculate target using previous outside bar."""
        # Start search from pattern_idx - 2 (skip current pattern bars)
        search_start = pattern_idx - 2

        if search_start < 0:
            # Not enough bars for lookback - use fallback
            target = calculate_measured_move_nb(
                entry_price, stop_price, direction, self.fallback_multiplier
            )
            return MagnitudeResult(
                target_price=target,
                method_used="measured_move",
                rr_ratio=self.fallback_multiplier,
                lookback_distance=None
            )

        # Find previous outside bar
        prev_outside_idx = find_previous_outside_bar_nb(
            classifications, search_start, self.max_lookback
        )

        if prev_outside_idx >= 0:
            # Propose target based on direction
            if direction == 1:  # Bullish - target is high of prev outside bar
                proposed_target = high[prev_outside_idx]
            else:  # Bearish - target is low of prev outside bar
                proposed_target = low[prev_outside_idx]

            # Validate geometry
            if validate_target_geometry_nb(entry_price, stop_price, proposed_target, direction):
                rr = calculate_rr_ratio_nb(entry_price, stop_price, proposed_target)
                lookback = pattern_idx - prev_outside_idx
                return MagnitudeResult(
                    target_price=proposed_target,
                    method_used="previous_outside",
                    rr_ratio=rr,
                    lookback_distance=lookback
                )

        # Fallback: geometry invalid or no previous outside bar found
        target = calculate_measured_move_nb(
            entry_price, stop_price, direction, self.fallback_multiplier
        )
        return MagnitudeResult(
            target_price=target,
            method_used="measured_move",
            rr_ratio=self.fallback_multiplier,
            lookback_distance=None
        )


# -----------------------------------------------------------------------------
# Option B: N-bar Swing Pivot
# -----------------------------------------------------------------------------

class OptionB_SwingPivot(MagnitudeCalculator):
    """
    Option B: N-bar Swing Pivot Target with 1.5x Fallback.

    N-bar swing detection:
    - Swing HIGH: bar whose HIGH > N bars on both sides
    - Swing LOW: bar whose LOW < N bars on both sides

    For 3-2U (bullish): Find first swing HIGH above 3-bar's HIGH
    For 3-2D (bearish): Find first swing LOW below 3-bar's LOW

    Fallback: 1.5x measured move if no valid pivot found
    """

    def __init__(
        self,
        n_bars: int = 2,
        max_lookback: int = 50,
        fallback_multiplier: float = 1.5
    ):
        """
        Initialize with swing detection parameters.

        Parameters
        ----------
        n_bars : int
            Bars required on each side for swing detection (default 2)
        max_lookback : int
            Maximum bars to search for pivot (default 50)
        fallback_multiplier : float
            R:R multiplier for measured move fallback
        """
        self.n_bars = n_bars
        self.max_lookback = max_lookback
        self.fallback_multiplier = fallback_multiplier

    @property
    def name(self) -> str:
        return f"Option_B_SwingPivot_N{self.n_bars}"

    def calculate_target(
        self,
        entry_price: float,
        stop_price: float,
        direction: int,
        high: np.ndarray,
        low: np.ndarray,
        classifications: np.ndarray,
        pattern_idx: int
    ) -> MagnitudeResult:
        """Calculate target using N-bar swing pivot."""
        # For 3-2 patterns, pattern_idx is the "2" bar, pattern_idx-1 is the "3" bar
        # The threshold is the 3-bar's extreme (high for bullish, low for bearish)
        bar3_idx = pattern_idx - 1

        if bar3_idx < 0:
            # Not enough data
            target = calculate_measured_move_nb(
                entry_price, stop_price, direction, self.fallback_multiplier
            )
            return MagnitudeResult(
                target_price=target,
                method_used="measured_move",
                rr_ratio=self.fallback_multiplier,
                lookback_distance=None
            )

        # Start search from bar before the 3-bar
        search_start = bar3_idx - 1

        if search_start < self.n_bars:
            # Not enough bars for swing detection
            target = calculate_measured_move_nb(
                entry_price, stop_price, direction, self.fallback_multiplier
            )
            return MagnitudeResult(
                target_price=target,
                method_used="measured_move",
                rr_ratio=self.fallback_multiplier,
                lookback_distance=None
            )

        if direction == 1:  # Bullish 3-2U - find swing HIGH above 3-bar's HIGH
            threshold = high[bar3_idx]  # Swing must be ABOVE this
            swing_idx = find_swing_high_nb(
                high, search_start, threshold, self.n_bars, self.max_lookback
            )

            if swing_idx >= 0:
                proposed_target = high[swing_idx]
                # Validate geometry (should always pass for swing above threshold)
                if validate_target_geometry_nb(entry_price, stop_price, proposed_target, direction):
                    rr = calculate_rr_ratio_nb(entry_price, stop_price, proposed_target)
                    lookback = pattern_idx - swing_idx
                    return MagnitudeResult(
                        target_price=proposed_target,
                        method_used="swing_pivot",
                        rr_ratio=rr,
                        lookback_distance=lookback
                    )

        else:  # Bearish 3-2D - find swing LOW below 3-bar's LOW
            threshold = low[bar3_idx]  # Swing must be BELOW this
            swing_idx = find_swing_low_nb(
                low, search_start, threshold, self.n_bars, self.max_lookback
            )

            if swing_idx >= 0:
                proposed_target = low[swing_idx]
                # Validate geometry (should always pass for swing below threshold)
                if validate_target_geometry_nb(entry_price, stop_price, proposed_target, direction):
                    rr = calculate_rr_ratio_nb(entry_price, stop_price, proposed_target)
                    lookback = pattern_idx - swing_idx
                    return MagnitudeResult(
                        target_price=proposed_target,
                        method_used="swing_pivot",
                        rr_ratio=rr,
                        lookback_distance=lookback
                    )

        # Fallback: no valid swing pivot found
        target = calculate_measured_move_nb(
            entry_price, stop_price, direction, self.fallback_multiplier
        )
        return MagnitudeResult(
            target_price=target,
            method_used="measured_move",
            rr_ratio=self.fallback_multiplier,
            lookback_distance=None
        )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

def get_all_calculators() -> list[MagnitudeCalculator]:
    """Get all four magnitude calculators for comparison."""
    return [
        OptionA_PreviousOutsideBar(),
        OptionB_SwingPivot(n_bars=2),
        OptionB_SwingPivot(n_bars=3),
        OptionC_MeasuredMove(multiplier=1.5),
    ]
