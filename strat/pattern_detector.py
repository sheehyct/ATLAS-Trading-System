r"""
STRAT Pattern Detection VBT Custom Indicator

Implements 3-1-2, 2-1-2, 2-2, 3-2, and 3-2-2 pattern detection with structural level targets.

Pattern Types:
    3-1-2: Outside-Inside-Directional (reversal pattern)
        - Bar 1: Outside bar (classification = 3 or -3)
        - Bar 2: Inside bar (classification = 1)
        - Bar 3: Directional bar (classification = 2 or -2)

    2-1-2: Directional-Inside-Directional (continuation/reversal pattern)
        - Bar 1: First directional bar (classification = 2 or -2)
        - Bar 2: Inside bar (classification = 1)
        - Bar 3: Second directional bar (classification = 2 or -2)

    2-2: Directional-Directional (rapid reversal pattern)
        - Bar 1: First directional bar (classification = 2 or -2)
        - Bar 2: Opposite directional bar (classification = -2 or 2)
        - NO inside bar - immediate momentum shift

    3-2: Outside-Directional (outside bar reversal pattern)
        - Bar 1: Outside bar (classification = 3 or -3)
        - Bar 2: Directional bar (classification = 2 or -2)
        - NO inside bar - outside bar rejection/reversal

    3-2-2: Outside-Directional-Reversal (outside bar reversal pattern)
        - Bar 1: Outside bar (classification = 3)
        - Bar 2: First directional bar (2D or 2U)
        - Bar 3: Opposite directional bar (reversal confirmed)

Entry/Stop/Target Calculation:
    3-1-2 & 2-1-2: Entry at inside bar high/low
    2-2 & 3-2-2: Entry at trigger bar high/low (no inside bar)
    Stop: Structural level (opposite extreme)
    Target: Structural level (bar extreme per STRAT methodology)

CRITICAL: Uses STRUCTURAL LEVEL targets per STRAT methodology
    - 3-1-2: Target = Outside bar extreme (high[i-2] or low[i-2])
    - 2-1-2: Target = First directional bar extreme (high[i-2] or low[i-2])
    - 2-2, 3-2, 3-2-2: Target = Bar[i-2] extreme (already correct)
"""

import numpy as np
from numba import njit
import vectorbtpro as vbt


@njit
def detect_312_patterns_nb(classifications, high, low):
    """
    Detect 3-1-2 patterns (Outside-Inside-Directional reversals).

    Pattern Structure:
        - Bar at i-2: Outside bar (classification = 3)
        - Bar at i-1: Inside bar (classification = 1)
        - Bar at i: Directional bar (classification = 2 or -2) - TRIGGER

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D array)
    high : np.ndarray
        Array of bar high prices (1D array)
    low : np.ndarray
        Array of bar low prices (1D array)

    Returns:
    --------
    tuple of 4 np.ndarray:
        entries : Boolean array (True at trigger bar index)
        stops : Stop loss prices (np.nan where no pattern)
        targets : Target prices (np.nan where no pattern)
        directions : 1 for bullish, -1 for bearish, 0 for no pattern

    Examples:
    ---------
    3-1-2 Bullish Pattern:
        Bar 0 (idx=2): Outside (H=110, L=90, classification=3)
        Bar 1 (idx=3): Inside (H=105, L=95, classification=1)
        Bar 2 (idx=4): 2U directional (classification=2) - TRIGGER

        Entry: 105 (inside bar high)
        Stop: 90 (outside bar low)
        Target: 110 (outside bar high - structural level)
        Direction: 1 (bullish)
    """
    # Handle both 1D and 2D arrays from VBT
    if classifications.ndim == 1:
        n = len(classifications)
        # Initialize output arrays as 1D
        entries = np.zeros(n, dtype=np.bool_)
        stops = np.full(n, np.nan, dtype=np.float64)
        targets = np.full(n, np.nan, dtype=np.float64)
        directions = np.zeros(n, dtype=np.int8)

        # Pattern requires 3 bars minimum
        for i in range(2, n):
            bar1_class = classifications[i-2]  # Outside bar
            bar2_class = classifications[i-1]  # Inside bar
            bar3_class = classifications[i]    # Directional bar (trigger)

            # Check for 3-1-2 pattern structure
            if bar1_class == 3 and bar2_class == 1:
                # Bullish 3-1-2U pattern (reversal to upside)
                if bar3_class == 2:
                    entries[i] = True
                    directions[i] = 1  # Bullish

                    # Entry trigger: Inside bar high
                    # Stop: Outside bar low (structural level)
                    # Target: Outside bar high (structural level per STRAT methodology)
                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-2]  # Outside bar low
                    targets[i] = high[i-2]  # Outside bar high (structural level)

                # Bearish 3-1-2D pattern (reversal to downside)
                elif bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Outside bar high (structural level)
                    # Target: Outside bar low (structural level per STRAT methodology)
                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-2]  # Outside bar high
                    targets[i] = low[i-2]  # Outside bar low (structural level)

    else:  # 2D arrays (n, 1) from VBT
        n = classifications.shape[0]
        # Initialize output arrays as 2D (n, 1)
        entries = np.zeros((n, 1), dtype=np.bool_)
        stops = np.full((n, 1), np.nan, dtype=np.float64)
        targets = np.full((n, 1), np.nan, dtype=np.float64)
        directions = np.zeros((n, 1), dtype=np.int8)

        # Pattern requires 3 bars minimum
        for i in range(2, n):
            bar1_class = classifications[i-2, 0]  # Outside bar
            bar2_class = classifications[i-1, 0]  # Inside bar
            bar3_class = classifications[i, 0]    # Directional bar (trigger)

            # Check for 3-1-2 pattern structure
            if bar1_class == 3 and bar2_class == 1:
                # Bullish 3-1-2U pattern (reversal to upside)
                if bar3_class == 2:
                    entries[i, 0] = True
                    directions[i, 0] = 1  # Bullish

                    # Entry trigger: Inside bar high
                    # Stop: Outside bar low (structural level)
                    # Target: Outside bar high (structural level per STRAT methodology)
                    trigger_price = high[i-1, 0]  # Inside bar high
                    stops[i, 0] = low[i-2, 0]  # Outside bar low
                    targets[i, 0] = high[i-2, 0]  # Outside bar high (structural level)

                # Bearish 3-1-2D pattern (reversal to downside)
                elif bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Outside bar high (structural level)
                    # Target: Outside bar low (structural level per STRAT methodology)
                    trigger_price = low[i-1, 0]  # Inside bar low
                    stops[i, 0] = high[i-2, 0]  # Outside bar high
                    targets[i, 0] = low[i-2, 0]  # Outside bar low (structural level)

    return (entries, stops, targets, directions)


@njit
def detect_212_patterns_nb(classifications, high, low):
    """
    Detect 2-1-2 patterns (Directional-Inside-Directional continuation/reversals).

    Pattern Structure:
        - Bar at i-2: First directional bar (classification = 2 or -2)
        - Bar at i-1: Inside bar (classification = 1)
        - Bar at i: Second directional bar (classification = 2 or -2) - TRIGGER

    Pattern Types:
        - 2U-1-2U: Bullish continuation
        - 2D-1-2D: Bearish continuation
        - 2D-1-2U: Bullish reversal (failed breakdown)
        - 2U-1-2D: Bearish reversal (failed breakout)

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D array)
    high : np.ndarray
        Array of bar high prices (1D array)
    low : np.ndarray
        Array of bar low prices (1D array)

    Returns:
    --------
    tuple of 4 np.ndarray:
        entries : Boolean array (True at trigger bar index)
        stops : Stop loss prices (np.nan where no pattern)
        targets : Target prices (np.nan where no pattern)
        directions : 1 for bullish, -1 for bearish, 0 for no pattern

    Examples:
    ---------
    2-1-2 Bullish Continuation (2U-1-2U):
        Bar 0 (idx=2): 2U directional (H=105, L=96, classification=2)
        Bar 1 (idx=3): Inside (H=104, L=97, classification=1)
        Bar 2 (idx=4): 2U directional (classification=2) - TRIGGER

        Entry: 104 (inside bar high)
        Stop: 97 (inside bar low)
        Target: 105 (first directional bar high - structural level)
        Direction: 1 (bullish)
    """
    # Handle both 1D and 2D arrays from VBT
    if classifications.ndim == 1:
        n = len(classifications)
        # Initialize output arrays as 1D
        entries = np.zeros(n, dtype=np.bool_)
        stops = np.full(n, np.nan, dtype=np.float64)
        targets = np.full(n, np.nan, dtype=np.float64)
        directions = np.zeros(n, dtype=np.int8)

        # Pattern requires 3 bars minimum
        for i in range(2, n):
            bar1_class = classifications[i-2]  # First directional bar
            bar2_class = classifications[i-1]  # Inside bar
            bar3_class = classifications[i]    # Second directional bar (trigger)

            # Check for 2-1-2 pattern structure (middle bar MUST be inside)
            if bar2_class == 1:
                # Bullish continuation: 2U-1-2U
                if bar1_class == 2 and bar3_class == 2:
                    entries[i] = True
                    directions[i] = 1  # Bullish

                    # Entry trigger: Inside bar high
                    # Stop: FIRST DIRECTIONAL BAR low (per STRAT methodology)
                    # Target: First directional bar high (structural level)
                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-2]  # First directional bar low
                    targets[i] = high[i-2]  # First directional bar high (structural level)

                # Bearish continuation: 2D-1-2D
                elif bar1_class == -2 and bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: FIRST DIRECTIONAL BAR high (per STRAT methodology)
                    # Target: First directional bar low (structural level)
                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-2]  # First directional bar high
                    targets[i] = low[i-2]  # First directional bar low (structural level)

                # Bullish reversal: 2D-1-2U (failed breakdown)
                elif bar1_class == -2 and bar3_class == 2:
                    entries[i] = True
                    directions[i] = 1  # Bullish

                    # Entry trigger: Inside bar high
                    # Stop: FIRST DIRECTIONAL BAR low (per STRAT methodology)
                    # Target: First directional bar high (structural level)
                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-2]  # First directional bar low
                    targets[i] = high[i-2]  # First directional bar high (structural level)

                # Bearish reversal: 2U-1-2D (failed breakout)
                elif bar1_class == 2 and bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: FIRST DIRECTIONAL BAR high (per STRAT methodology)
                    # Target: First directional bar low (structural level)
                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-2]  # First directional bar high
                    targets[i] = low[i-2]  # First directional bar low (structural level)

    else:  # 2D arrays (n, 1) from VBT
        n = classifications.shape[0]
        # Initialize output arrays as 2D (n, 1)
        entries = np.zeros((n, 1), dtype=np.bool_)
        stops = np.full((n, 1), np.nan, dtype=np.float64)
        targets = np.full((n, 1), np.nan, dtype=np.float64)
        directions = np.zeros((n, 1), dtype=np.int8)

        # Pattern requires 3 bars minimum
        for i in range(2, n):
            bar1_class = classifications[i-2, 0]  # First directional bar
            bar2_class = classifications[i-1, 0]  # Inside bar
            bar3_class = classifications[i, 0]    # Second directional bar (trigger)

            # Check for 2-1-2 pattern structure (middle bar MUST be inside)
            if bar2_class == 1:
                # Bullish continuation: 2U-1-2U
                if bar1_class == 2 and bar3_class == 2:
                    entries[i, 0] = True
                    directions[i, 0] = 1  # Bullish

                    # Stop: FIRST DIRECTIONAL BAR low (per STRAT methodology)
                    # Target: First directional bar high (structural level)
                    stops[i, 0] = low[i-2, 0]  # First directional bar low
                    targets[i, 0] = high[i-2, 0]  # First directional bar high (structural level)

                # Bearish continuation: 2D-1-2D
                elif bar1_class == -2 and bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    # Stop: FIRST DIRECTIONAL BAR high (per STRAT methodology)
                    # Target: First directional bar low (structural level)
                    stops[i, 0] = high[i-2, 0]  # First directional bar high
                    targets[i, 0] = low[i-2, 0]  # First directional bar low (structural level)

                # Bullish reversal: 2D-1-2U (failed breakdown)
                elif bar1_class == -2 and bar3_class == 2:
                    entries[i, 0] = True
                    directions[i, 0] = 1  # Bullish

                    # Stop: FIRST DIRECTIONAL BAR low (per STRAT methodology)
                    stops[i, 0] = low[i-2, 0]  # First directional bar low
                    targets[i, 0] = high[i-2, 0]  # First directional bar high (structural level)

                # Bearish reversal: 2U-1-2D (failed breakout)
                elif bar1_class == 2 and bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    # Stop: FIRST DIRECTIONAL BAR high (per STRAT methodology)
                    stops[i, 0] = high[i-2, 0]  # First directional bar high
                    targets[i, 0] = low[i-2, 0]  # First directional bar low (structural level)

    return (entries, stops, targets, directions)


@njit
def find_previous_directional_bar_nb(classifications, current_idx, direction):
    """
    Find the previous directional bar (2U or 2D) in the same direction.

    Used for 2-2 pattern magnitude calculation. Looks backward from current_idx
    to find the most recent bar with the specified classification.

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classification array (1D)
    current_idx : int
        Current bar index (trigger bar)
    direction : float
        Target classification to find (2.0 for 2U, -2.0 for 2D)

    Returns:
    --------
    int
        Index of previous directional bar, or -1 if not found

    Examples:
    ---------
    Pattern: [2D, 2D, 2U] at indices [0, 1, 2]
    find_previous_directional_bar_nb(classifications, 2, -2.0)
    Returns: 0 (first 2D bar, skipping bar 1 which is at i-1)
    """
    # Start from i-2 (skip the bar at i-1 which is part of the reversal)
    # Look backward to find previous bar with same direction
    for i in range(current_idx - 2, -1, -1):
        if classifications[i] == direction:
            return i
    return -1  # Not found


@njit
def validate_target_geometry_nb(entry_price, stop_price, target_price, direction):
    """
    Validate that target price creates profitable geometry.

    For bullish patterns: target must be ABOVE entry
    For bearish patterns: target must be BELOW entry

    This prevents inverted magnitude targets where previous directional bars
    are found at price levels that create geometrically impossible trades.

    Parameters:
    -----------
    entry_price : float
        Entry price for the trade
    stop_price : float
        Stop loss price
    target_price : float
        Proposed magnitude target price
    direction : int
        1 for bullish, -1 for bearish

    Returns:
    --------
    bool
        True if geometry is valid, False if inverted

    Examples:
    ---------
    Bullish pattern (direction=1):
        validate_target_geometry_nb(217.36, 214.86, 193.19, 1) -> False (target below entry)
        validate_target_geometry_nb(217.36, 214.86, 220.00, 1) -> True (target above entry)

    Bearish pattern (direction=-1):
        validate_target_geometry_nb(217.36, 220.00, 214.00, -1) -> True (target below entry)
        validate_target_geometry_nb(217.36, 220.00, 220.50, -1) -> False (target above entry)
    """
    if direction == 1:  # Bullish - target must be ABOVE entry
        return target_price > entry_price
    elif direction == -1:  # Bearish - target must be BELOW entry
        return target_price < entry_price
    else:
        return False  # Invalid direction


@njit
def calculate_measured_move_nb(entry_price, stop_price, direction, multiplier=1.5):
    """
    Calculate measured move target when geometric validation fails.

    Uses stop distance projected from entry in profit direction.
    This is the fallback when previous directional bars create inverted geometry.

    Parameters:
    -----------
    entry_price : float
        Entry price for the trade
    stop_price : float
        Stop loss price
    direction : int
        1 for bullish, -1 for bearish
    multiplier : float
        Risk-reward multiplier (default 1.5 for 1.5:1 R:R)

    Returns:
    --------
    float
        Measured move target price

    Examples:
    ---------
    Bullish (entry $217.36, stop $214.86):
        stop_distance = $217.36 - $214.86 = $2.50
        target = $217.36 + ($2.50 * 1.5) = $221.11

    Bearish (entry $217.36, stop $220.00):
        stop_distance = $220.00 - $217.36 = $2.64
        target = $217.36 - ($2.64 * 1.5) = $213.40
    """
    stop_distance = abs(entry_price - stop_price)

    if direction == 1:  # Bullish - project upward
        return entry_price + (stop_distance * multiplier)
    elif direction == -1:  # Bearish - project downward
        return entry_price - (stop_distance * multiplier)
    else:
        return entry_price  # Fallback for invalid direction


@njit
def detect_22_patterns_nb(classifications, high, low):
    """
    Detect 2-2 reversal patterns (2D-2U, 2U-2D).

    Pattern Structure:
        - Bar at i-2: Any bar (establishes magnitude target)
        - Bar at i-1: First directional bar (classification = 2 or -2)
        - Bar at i: Opposite directional bar (classification = -2 or 2) - TRIGGER
        - NO inside bar - rapid momentum reversal

    2D-2U: Bearish → Bullish reversal (failed breakdown)
    2U-2D: Bullish → Bearish reversal (failed breakout)

    CORRECTED Magnitude Calculation (Session 76):
        Target = Extreme of bar[i-2] (the bar PREVIOUS to the 2-bar reversal)
        - For 2D-2U bullish: Target = high[i-2]
        - For 2U-2D bearish: Target = low[i-2]

        This is per STRAT methodology - the target is the structural level
        established by the bar immediately before the reversal pattern.

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D or 2D array)
    high : np.ndarray
        Array of bar high prices
    low : np.ndarray
        Array of bar low prices

    Returns:
    --------
    tuple of 4 np.ndarray:
        entries : Boolean array (True at trigger bar index)
        stops : Stop loss prices (np.nan where no pattern)
        targets : Target prices (np.nan where no pattern)
        directions : 1 for bullish, -1 for bearish, 0 for no pattern

    Examples:
    ---------
    2D-2U Bullish Reversal (SPY Daily 2025-11-19):
        Bar i-2 (Nov 14): 2D (H=$673.71) ← TARGET
        Bar i-1 (Nov 18): 2D (H=$665.12, L=$655.86)
        Bar i (Nov 19):   2U - TRIGGER

        Entry: $665.12 (2D bar high)
        Stop: $655.86 (2D bar low)
        Target: $673.71 (high of bar i-2, previous to reversal)
        Direction: 1 (bullish)

    2U-2D Bearish Reversal:
        Bar i-2: 2U (L=$100)
        Bar i-1: 2U (H=$110, L=$105)
        Bar i:   2D - TRIGGER

        Entry: $105 (2U bar low)
        Stop: $110 (2U bar high)
        Target: $100 (low of bar i-2, previous to reversal)
        Direction: -1 (bearish)
    """
    # Handle both 1D and 2D arrays from VBT
    if classifications.ndim == 1:
        n = len(classifications)
        # Initialize output arrays as 1D
        entries = np.zeros(n, dtype=np.bool_)
        stops = np.full(n, np.nan, dtype=np.float64)
        targets = np.full(n, np.nan, dtype=np.float64)
        directions = np.zeros(n, dtype=np.int8)

        # Pattern requires 3 bars minimum (i-2, i-1, i)
        for i in range(2, n):
            bar_prev = classifications[i-2]   # Bar before reversal (check for outside bar)
            bar1_class = classifications[i-1]  # First directional bar
            bar2_class = classifications[i]    # Opposite directional bar (trigger)

            # Skip if this is a 3-2-2 pattern (outside bar at i-2)
            # 3-2-2 patterns are detected separately by detect_322_patterns_nb
            if abs(bar_prev) == 3:
                continue

            # 2D-2U: Bearish to Bullish reversal (failed breakdown)
            if bar1_class == -2 and bar2_class == 2:
                entries[i] = True
                directions[i] = 1  # Bullish

                # SESSION 76 CORRECTED Entry/Stop/Target:
                # Entry Price = high[i-1] (2D bar high, the level that broke to create 2U)
                # Stop Price = low[i-1] (2D bar low)
                # Target = high[i-2] (bar PREVIOUS to the 2-bar reversal pattern)
                entry_price = high[i-1]  # 2D bar high
                stops[i] = low[i-1]      # 2D bar low

                # Target = high of bar[i-2] (bar previous to reversal)
                proposed_target = high[i-2]

                # Validate geometry (target must be ABOVE entry for bullish)
                if validate_target_geometry_nb(entry_price, stops[i], proposed_target, 1):
                    targets[i] = proposed_target
                else:
                    # Geometry invalid - use measured move fallback
                    targets[i] = calculate_measured_move_nb(entry_price, stops[i], 1, 1.5)

            # 2U-2D: Bullish to Bearish reversal (failed breakout)
            elif bar1_class == 2 and bar2_class == -2:
                entries[i] = True
                directions[i] = -1  # Bearish

                # SESSION 76 CORRECTED Entry/Stop/Target:
                # Entry Price = low[i-1] (2U bar low, the level that broke to create 2D)
                # Stop Price = high[i-1] (2U bar high)
                # Target = low[i-2] (bar PREVIOUS to the 2-bar reversal pattern)
                entry_price = low[i-1]   # 2U bar low
                stops[i] = high[i-1]     # 2U bar high

                # Target = low of bar[i-2] (bar previous to reversal)
                proposed_target = low[i-2]

                # Validate geometry (target must be BELOW entry for bearish)
                if validate_target_geometry_nb(entry_price, stops[i], proposed_target, -1):
                    targets[i] = proposed_target
                else:
                    # Geometry invalid - use measured move fallback
                    targets[i] = calculate_measured_move_nb(entry_price, stops[i], -1, 1.5)

    else:  # 2D array
        n = classifications.shape[0]
        # Initialize output arrays as 2D (n, 1)
        entries = np.zeros((n, 1), dtype=np.bool_)
        stops = np.full((n, 1), np.nan, dtype=np.float64)
        targets = np.full((n, 1), np.nan, dtype=np.float64)
        directions = np.zeros((n, 1), dtype=np.int8)

        # Pattern requires 3 bars minimum (i-2, i-1, i)
        for i in range(2, n):
            bar_prev = classifications[i-2, 0]   # Bar before reversal (check for outside bar)
            bar1_class = classifications[i-1, 0]  # First directional bar
            bar2_class = classifications[i, 0]    # Opposite directional bar (trigger)

            # Skip if this is a 3-2-2 pattern (outside bar at i-2)
            # 3-2-2 patterns are detected separately by detect_322_patterns_nb
            if abs(bar_prev) == 3:
                continue

            # 2D-2U: Bearish to Bullish reversal
            if bar1_class == -2 and bar2_class == 2:
                entries[i, 0] = True
                directions[i, 0] = 1  # Bullish

                # SESSION 76 CORRECTED Entry/Stop/Target:
                # Entry Price = high[i-1] (2D bar high)
                # Stop Price = low[i-1] (2D bar low)
                # Target = high[i-2] (bar PREVIOUS to the 2-bar reversal pattern)
                entry_price = high[i-1, 0]  # 2D bar high
                stops[i, 0] = low[i-1, 0]   # 2D bar low

                # Target = high of bar[i-2] (bar previous to reversal)
                proposed_target = high[i-2, 0]

                # Validate geometry (target must be ABOVE entry for bullish)
                if validate_target_geometry_nb(entry_price, stops[i, 0], proposed_target, 1):
                    targets[i, 0] = proposed_target
                else:
                    # Geometry invalid - use measured move fallback
                    targets[i, 0] = calculate_measured_move_nb(entry_price, stops[i, 0], 1, 1.5)

            # 2U-2D: Bullish to Bearish reversal
            elif bar1_class == 2 and bar2_class == -2:
                entries[i, 0] = True
                directions[i, 0] = -1  # Bearish

                # SESSION 76 CORRECTED Entry/Stop/Target:
                # Entry Price = low[i-1] (2U bar low)
                # Stop Price = high[i-1] (2U bar high)
                # Target = low[i-2] (bar PREVIOUS to the 2-bar reversal pattern)
                entry_price = low[i-1, 0]   # 2U bar low
                stops[i, 0] = high[i-1, 0]  # 2U bar high

                # Target = low of bar[i-2] (bar previous to reversal)
                proposed_target = low[i-2, 0]

                # Validate geometry (target must be BELOW entry for bearish)
                if validate_target_geometry_nb(entry_price, stops[i, 0], proposed_target, -1):
                    targets[i, 0] = proposed_target
                else:
                    # Geometry invalid - use measured move fallback
                    targets[i, 0] = calculate_measured_move_nb(entry_price, stops[i, 0], -1, 1.5)

    return (entries, stops, targets, directions)


@njit
def detect_32_patterns_nb(classifications, high, low):
    """
    Detect 3-2 reversal patterns (3D-2U, 3U-2D, 3-2U, 3-2D).

    Pattern Structure:
        - Bar at i-1: Outside bar (classification = 3, -3, or abs = 3)
        - Bar at i: Directional bar (classification = 2 or -2) - TRIGGER
        - NO inside bar - outside bar rejection/reversal

    3D-2U / 3-2U: Outside bar down → Bullish reversal (bearish rejection)
    3U-2D / 3-2D: Outside bar up → Bearish reversal (bullish rejection)

    Magnitude Calculation (Option C - 1.5x Measured Move):
        - Always uses 1.5x risk/reward ratio (Session 83K-62 winner)
        - Target = Entry +/- 1.5 * abs(Entry - Stop)
        - For bullish (3D-2U/3-2U): Target = Entry + 1.5 * (Entry - Stop)
        - For bearish (3U-2D/3-2D): Target = Entry - 1.5 * (Stop - Entry)
        - Outperforms previous outside bar lookback in OOS testing

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D or 2D array)
    high : np.ndarray
        Array of bar high prices
    low : np.ndarray
        Array of bar low prices

    Returns:
    --------
    tuple of 4 np.ndarray:
        entries : Boolean array (True at trigger bar index)
        stops : Stop loss prices (np.nan where no pattern)
        targets : Target prices (np.nan where no pattern)
        directions : 1 for bullish, -1 for bearish, 0 for no pattern

    Examples:
    ---------
    3D-2U Bullish Reversal (outside bar rejection):
        Bar 0 (idx=0): 3D (H=105, L=95, classification=-3)
        Bar 1 (idx=1): 2U (H=107, L=96, classification=2) - TRIGGER

        Entry: 105 (3D bar high, level that broke to create 2U)
        Stop: 95 (3D bar low)
        Target: 105 + 1.5 * (105 - 95) = 120 (1.5x measured move)
        Direction: 1 (bullish)

    3U-2D Bearish Reversal (outside bar rejection):
        Bar 0 (idx=0): 3U (H=110, L=100, classification=3)
        Bar 1 (idx=1): 2D (H=105, L=95, classification=-2) - TRIGGER

        Entry: 100 (3U bar low, level that broke to create 2D)
        Stop: 110 (3U bar high)
        Target: 100 - 1.5 * (110 - 100) = 85 (1.5x measured move)
        Direction: -1 (bearish)
    """
    # Handle both 1D and 2D arrays from VBT
    if classifications.ndim == 1:
        n = len(classifications)
        # Initialize output arrays as 1D
        entries = np.zeros(n, dtype=np.bool_)
        stops = np.full(n, np.nan, dtype=np.float64)
        targets = np.full(n, np.nan, dtype=np.float64)
        directions = np.zeros(n, dtype=np.int8)

        # Pattern requires 2 bars minimum
        for i in range(1, n):
            bar1_class = classifications[i-1]  # Outside bar
            bar2_class = classifications[i]    # Directional bar (trigger)

            # 3D-2U or 3-2U: Outside bar down → Bullish reversal
            # Accept both -3 (outside bar down) and 3 (neutral outside bar)
            if (bar1_class == -3 or abs(bar1_class) == 3) and bar2_class == 2:
                entries[i] = True
                directions[i] = 1  # Bullish

                # Entry: LIVE entry when bar i breaks ABOVE bar i-1 HIGH
                # Entry Price = high[i-1] (outside bar high)
                # Stop Price = low[i-1] (outside bar low)
                entry_price = high[i-1]  # Outside bar high
                stops[i] = low[i-1]      # Outside bar low

                # Option C: 1.5x measured move (Session 83K-62 winner)
                # Target = Entry + 1.5 * (Entry - Stop)
                targets[i] = calculate_measured_move_nb(entry_price, stops[i], 1, 1.5)

            # 3U-2D or 3-2D: Outside bar up → Bearish reversal
            # Accept both 3 (outside bar up) and -3 or neutral outside bar
            elif (bar1_class == 3 or abs(bar1_class) == 3) and bar2_class == -2:
                entries[i] = True
                directions[i] = -1  # Bearish

                # Entry: LIVE entry when bar i breaks BELOW bar i-1 LOW
                # Entry Price = low[i-1] (outside bar low)
                # Stop Price = high[i-1] (outside bar high)
                entry_price = low[i-1]   # Outside bar low
                stops[i] = high[i-1]     # Outside bar high

                # Option C: 1.5x measured move (Session 83K-62 winner)
                # Target = Entry - 1.5 * (Stop - Entry)
                targets[i] = calculate_measured_move_nb(entry_price, stops[i], -1, 1.5)

    else:  # 2D array (multi-column from VBT)
        n = classifications.shape[0]
        ncol = classifications.shape[1]

        # Initialize output arrays as 2D
        entries = np.zeros((n, ncol), dtype=np.bool_)
        stops = np.full((n, ncol), np.nan, dtype=np.float64)
        targets = np.full((n, ncol), np.nan, dtype=np.float64)
        directions = np.zeros((n, ncol), dtype=np.int8)

        # Process each column independently
        for col in range(ncol):
            for i in range(1, n):
                bar1_class = classifications[i-1, col]
                bar2_class = classifications[i, col]

                # 3D-2U or 3-2U: Outside bar down → Bullish reversal
                if (bar1_class == -3 or abs(bar1_class) == 3) and bar2_class == 2:
                    entries[i, col] = True
                    directions[i, col] = 1

                    entry_price = high[i-1, col]
                    stops[i, col] = low[i-1, col]

                    # Option C: 1.5x measured move (Session 83K-62 winner)
                    targets[i, col] = calculate_measured_move_nb(entry_price, stops[i, col], 1, 1.5)

                # 3U-2D or 3-2D: Outside bar up → Bearish reversal
                elif (bar1_class == 3 or abs(bar1_class) == 3) and bar2_class == -2:
                    entries[i, col] = True
                    directions[i, col] = -1

                    entry_price = low[i-1, col]
                    stops[i, col] = high[i-1, col]

                    # Option C: 1.5x measured move (Session 83K-62 winner)
                    targets[i, col] = calculate_measured_move_nb(entry_price, stops[i, col], -1, 1.5)

    return (entries, stops, targets, directions)


@njit
def detect_322_patterns_nb(classifications, high, low):
    """
    Detect 3-2-2 reversal patterns (3-2D-2U, 3-2U-2D).

    Pattern Structure (Session 76):
        - Bar at i-2: Outside bar (classification = 3 or abs = 3)
        - Bar at i-1: First directional bar (failed breakdown/breakout from outside bar)
        - Bar at i: Opposite directional bar - TRIGGER (reversal confirmed)

    3-2D-2U: Outside bar → 2D (failed breakdown) → 2U (bullish reversal)
    3-2U-2D: Outside bar → 2U (failed breakout) → 2D (bearish reversal)

    Target Calculation:
        Target = Outside bar extreme (bar i-2)
        - For 3-2D-2U bullish: Target = high[i-2] (outside bar high)
        - For 3-2U-2D bearish: Target = low[i-2] (outside bar low)

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D or 2D array)
    high : np.ndarray
        Array of bar high prices
    low : np.ndarray
        Array of bar low prices

    Returns:
    --------
    tuple of 4 np.ndarray:
        entries : Boolean array (True at trigger bar index)
        stops : Stop loss prices (np.nan where no pattern)
        targets : Target prices (np.nan where no pattern)
        directions : 1 for bullish, -1 for bearish, 0 for no pattern

    Examples:
    ---------
    3-2D-2U Bullish Reversal (SPY Daily 2025-11-24):
        Bar i-2 (Nov 20): Type 3 (Outside), H=$675.56 ← TARGET
        Bar i-1 (Nov 21): Type 2D, H=$664.55, L=$650.85
        Bar i (Nov 24):   Type 2U - TRIGGER

        Entry: $664.55 (2D bar high)
        Stop: $650.85 (2D bar low)
        Target: $675.56 (outside bar high)
        Direction: 1 (bullish)

    3-2U-2D Bearish Reversal:
        Bar i-2: Type 3 (Outside), L=$95
        Bar i-1: Type 2U (failed breakout), H=$105, L=$100
        Bar i:   Type 2D - TRIGGER

        Entry: $100 (2U bar low)
        Stop: $105 (2U bar high)
        Target: $95 (outside bar low)
        Direction: -1 (bearish)
    """
    # Handle both 1D and 2D arrays from VBT
    if classifications.ndim == 1:
        n = len(classifications)
        # Initialize output arrays as 1D
        entries = np.zeros(n, dtype=np.bool_)
        stops = np.full(n, np.nan, dtype=np.float64)
        targets = np.full(n, np.nan, dtype=np.float64)
        directions = np.zeros(n, dtype=np.int8)

        # Pattern requires 3 bars minimum (i-2, i-1, i)
        for i in range(2, n):
            bar_outside = classifications[i-2]  # Outside bar
            bar1_class = classifications[i-1]   # First directional bar
            bar2_class = classifications[i]     # Opposite directional bar (trigger)

            # 3-2D-2U: Outside bar → 2D → 2U (Bullish reversal)
            if abs(bar_outside) == 3 and bar1_class == -2 and bar2_class == 2:
                entries[i] = True
                directions[i] = 1  # Bullish

                # Entry Price = high[i-1] (2D bar high)
                # Stop Price = low[i-1] (2D bar low)
                # Target = high[i-2] (outside bar high)
                entry_price = high[i-1]  # 2D bar high
                stops[i] = low[i-1]      # 2D bar low
                targets[i] = high[i-2]   # Outside bar high (magnitude)

            # 3-2U-2D: Outside bar → 2U → 2D (Bearish reversal)
            elif abs(bar_outside) == 3 and bar1_class == 2 and bar2_class == -2:
                entries[i] = True
                directions[i] = -1  # Bearish

                # Entry Price = low[i-1] (2U bar low)
                # Stop Price = high[i-1] (2U bar high)
                # Target = low[i-2] (outside bar low)
                entry_price = low[i-1]   # 2U bar low
                stops[i] = high[i-1]     # 2U bar high
                targets[i] = low[i-2]    # Outside bar low (magnitude)

    else:  # 2D array
        n = classifications.shape[0]
        # Initialize output arrays as 2D (n, 1)
        entries = np.zeros((n, 1), dtype=np.bool_)
        stops = np.full((n, 1), np.nan, dtype=np.float64)
        targets = np.full((n, 1), np.nan, dtype=np.float64)
        directions = np.zeros((n, 1), dtype=np.int8)

        # Pattern requires 3 bars minimum (i-2, i-1, i)
        for i in range(2, n):
            bar_outside = classifications[i-2, 0]  # Outside bar
            bar1_class = classifications[i-1, 0]   # First directional bar
            bar2_class = classifications[i, 0]     # Opposite directional bar (trigger)

            # 3-2D-2U: Outside bar → 2D → 2U (Bullish reversal)
            if abs(bar_outside) == 3 and bar1_class == -2 and bar2_class == 2:
                entries[i, 0] = True
                directions[i, 0] = 1  # Bullish

                entry_price = high[i-1, 0]  # 2D bar high
                stops[i, 0] = low[i-1, 0]   # 2D bar low
                targets[i, 0] = high[i-2, 0]  # Outside bar high

            # 3-2U-2D: Outside bar → 2U → 2D (Bearish reversal)
            elif abs(bar_outside) == 3 and bar1_class == 2 and bar2_class == -2:
                entries[i, 0] = True
                directions[i, 0] = -1  # Bearish

                entry_price = low[i-1, 0]   # 2U bar low
                stops[i, 0] = high[i-1, 0]  # 2U bar high
                targets[i, 0] = low[i-2, 0]  # Outside bar low

    return (entries, stops, targets, directions)


# =============================================================================
# SETUP DETECTION FUNCTIONS (Session 83K-68)
# =============================================================================
# These functions detect patterns ONE BAR EARLIER - when the SETUP forms,
# not when the pattern completes. This enables live entry when the trigger breaks.
#
# STRAT Methodology: "Where Is The Next 2?"
# - Every bar starts as Type 1 at open
# - We watch for it to break prior bar's high (becomes 2U) or low (becomes 2D)
# - Entry happens LIVE at that break moment
# =============================================================================


@njit
def detect_312_setups_nb(classifications, high, low):
    """
    Detect 3-1 setups (Outside-Inside) awaiting directional break.

    Session 83K-68: Setup-based detection for live trading.

    Setup Structure:
        - Bar at i-1: Outside bar (classification = 3)
        - Bar at i: Inside bar (classification = 1) <-- SETUP detected here

    The NEXT bar will determine if this becomes:
        - 3-1-2U (bullish) if next bar breaks above inside bar high
        - 3-1-2D (bearish) if next bar breaks below inside bar low

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py (1D array)
    high : np.ndarray
        Array of bar high prices (1D array)
    low : np.ndarray
        Array of bar low prices (1D array)

    Returns:
    --------
    tuple of 7 np.ndarray:
        setups : Boolean array (True where 3-1 setup exists)
        bullish_trigger : Price to break for bullish entry (inside bar high)
        bearish_trigger : Price to break for bearish entry (inside bar low)
        stop_long : Stop for bullish trade (outside bar low)
        stop_short : Stop for bearish trade (outside bar high)
        target_long : Target for bullish trade (outside bar high)
        target_short : Target for bearish trade (outside bar low)
    """
    if classifications.ndim == 1:
        n = len(classifications)
        setups = np.zeros(n, dtype=np.bool_)
        bullish_trigger = np.full(n, np.nan, dtype=np.float64)
        bearish_trigger = np.full(n, np.nan, dtype=np.float64)
        stop_long = np.full(n, np.nan, dtype=np.float64)
        stop_short = np.full(n, np.nan, dtype=np.float64)
        target_long = np.full(n, np.nan, dtype=np.float64)
        target_short = np.full(n, np.nan, dtype=np.float64)

        # Setup requires 2 bars minimum (outside + inside)
        for i in range(1, n):
            bar_outside = classifications[i-1]  # Previous bar is outside
            bar_inside = classifications[i]      # Current bar is inside

            # 3-1 setup: Outside bar followed by Inside bar
            if bar_outside == 3 and bar_inside == 1:
                setups[i] = True

                # Bullish trigger: Inside bar high (break above = 2U)
                bullish_trigger[i] = high[i]  # Inside bar high

                # Bearish trigger: Inside bar low (break below = 2D)
                bearish_trigger[i] = low[i]   # Inside bar low

                # Stops and targets from outside bar
                stop_long[i] = low[i-1]    # Outside bar low
                stop_short[i] = high[i-1]  # Outside bar high
                target_long[i] = high[i-1]  # Outside bar high
                target_short[i] = low[i-1]  # Outside bar low

    else:  # 2D arrays
        n = classifications.shape[0]
        setups = np.zeros((n, 1), dtype=np.bool_)
        bullish_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        bearish_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        stop_long = np.full((n, 1), np.nan, dtype=np.float64)
        stop_short = np.full((n, 1), np.nan, dtype=np.float64)
        target_long = np.full((n, 1), np.nan, dtype=np.float64)
        target_short = np.full((n, 1), np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_outside = classifications[i-1, 0]
            bar_inside = classifications[i, 0]

            if bar_outside == 3 and bar_inside == 1:
                setups[i, 0] = True
                bullish_trigger[i, 0] = high[i, 0]
                bearish_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i-1, 0]
                stop_short[i, 0] = high[i-1, 0]
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

    return (setups, bullish_trigger, bearish_trigger,
            stop_long, stop_short, target_long, target_short)


@njit
def detect_212_setups_nb(classifications, high, low):
    """
    Detect 2-1 setups (Directional-Inside) awaiting directional break.

    Session 83K-68: Setup-based detection for live trading.

    Setup Structure:
        - Bar at i-1: First directional bar (classification = 2 or -2)
        - Bar at i: Inside bar (classification = 1) <-- SETUP detected here

    The NEXT bar will determine if this becomes:
        - 2U-1-2U or 2D-1-2U (bullish) if next bar breaks above inside bar high
        - 2U-1-2D or 2D-1-2D (bearish) if next bar breaks below inside bar low

    Returns:
    --------
    tuple of 8 np.ndarray:
        setups : Boolean array (True where 2-1 setup exists)
        first_bar_dir : Direction of first bar (2 for 2U, -2 for 2D)
        bullish_trigger : Price to break for bullish entry (inside bar high)
        bearish_trigger : Price to break for bearish entry (inside bar low)
        stop_long : Stop for bullish trade (inside bar low)
        stop_short : Stop for bearish trade (inside bar high)
        target_long : Target for bullish trade (first directional bar high)
        target_short : Target for bearish trade (first directional bar low)
    """
    if classifications.ndim == 1:
        n = len(classifications)
        setups = np.zeros(n, dtype=np.bool_)
        first_bar_dir = np.zeros(n, dtype=np.int8)
        bullish_trigger = np.full(n, np.nan, dtype=np.float64)
        bearish_trigger = np.full(n, np.nan, dtype=np.float64)
        stop_long = np.full(n, np.nan, dtype=np.float64)
        stop_short = np.full(n, np.nan, dtype=np.float64)
        target_long = np.full(n, np.nan, dtype=np.float64)
        target_short = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_dir = classifications[i-1]  # First directional bar
            bar_inside = classifications[i]  # Current bar is inside

            # 2-1 setup: Directional bar followed by Inside bar
            if (bar_dir == 2 or bar_dir == -2) and bar_inside == 1:
                setups[i] = True
                first_bar_dir[i] = bar_dir

                # Entry triggers from inside bar
                bullish_trigger[i] = high[i]  # Inside bar high
                bearish_trigger[i] = low[i]   # Inside bar low

                # Stops from FIRST DIRECTIONAL BAR (per STRAT methodology)
                # This provides proper structural stop placement
                stop_long[i] = low[i-1]     # First dir bar low
                stop_short[i] = high[i-1]   # First dir bar high

                # Targets from first directional bar
                target_long[i] = high[i-1]   # First dir bar high
                target_short[i] = low[i-1]   # First dir bar low

    else:  # 2D arrays
        n = classifications.shape[0]
        setups = np.zeros((n, 1), dtype=np.bool_)
        first_bar_dir = np.zeros((n, 1), dtype=np.int8)
        bullish_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        bearish_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        stop_long = np.full((n, 1), np.nan, dtype=np.float64)
        stop_short = np.full((n, 1), np.nan, dtype=np.float64)
        target_long = np.full((n, 1), np.nan, dtype=np.float64)
        target_short = np.full((n, 1), np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_dir = classifications[i-1, 0]
            bar_inside = classifications[i, 0]

            if (bar_dir == 2 or bar_dir == -2) and bar_inside == 1:
                setups[i, 0] = True
                first_bar_dir[i, 0] = bar_dir
                bullish_trigger[i, 0] = high[i, 0]
                bearish_trigger[i, 0] = low[i, 0]
                # Stops from FIRST DIRECTIONAL BAR (per STRAT methodology)
                stop_long[i, 0] = low[i-1, 0]     # First dir bar low
                stop_short[i, 0] = high[i-1, 0]   # First dir bar high
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

    return (setups, first_bar_dir, bullish_trigger, bearish_trigger,
            stop_long, stop_short, target_long, target_short)


@njit
def detect_22_setups_nb(classifications, high, low):
    """
    Detect 2D and 2U bars as BIDIRECTIONAL setups.

    Session EQUITY-19: CRITICAL FIX - Return BOTH directions per STRAT methodology.

    Per STRAT: "The forming bar is always treated as '1' until it breaks."

    Setup Structure:
        - Bar at i-1: Any bar (reference for target)
        - Bar at i: Directional bar (2D or 2U) <-- SETUP detected here
        - FORMING bar: Treated as implicit "1" until it breaks

    The NEXT bar (forming bar) will determine direction by which bound breaks FIRST:
        - For 2U bar: Break HIGH = 2U-2U continuation LONG, Break LOW = 2U-2D reversal SHORT
        - For 2D bar: Break HIGH = 2D-2U reversal LONG, Break LOW = 2D-2D continuation SHORT

    Returns:
    --------
    tuple of 8 np.ndarray:
        setups : Boolean array (True where 2D or 2U setup exists)
        setup_dir : Direction of setup bar (2 for 2U, -2 for 2D)
        long_trigger : Price to break for LONG entry (bar HIGH)
        short_trigger : Price to break for SHORT entry (bar LOW)
        stop_long : Stop for LONG trade (bar LOW)
        stop_short : Stop for SHORT trade (bar HIGH)
        target_long : Target for LONG trade (prior bar HIGH)
        target_short : Target for SHORT trade (prior bar LOW)
    """
    if classifications.ndim == 1:
        n = len(classifications)
        setups = np.zeros(n, dtype=np.bool_)
        setup_dir = np.zeros(n, dtype=np.int8)
        long_trigger = np.full(n, np.nan, dtype=np.float64)
        short_trigger = np.full(n, np.nan, dtype=np.float64)
        stop_long = np.full(n, np.nan, dtype=np.float64)
        stop_short = np.full(n, np.nan, dtype=np.float64)
        target_long = np.full(n, np.nan, dtype=np.float64)
        target_short = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_class = classifications[i]

            # 2D bar: Bidirectional setup
            # - LONG (reversal 2D-2U): Break above 2D bar high
            # - SHORT (continuation 2D-2D): Break below 2D bar low
            if bar_class == -2:
                setups[i] = True
                setup_dir[i] = -2

                long_trigger[i] = high[i]   # Break HIGH for LONG
                short_trigger[i] = low[i]   # Break LOW for SHORT
                stop_long[i] = low[i]       # LONG stop at bar LOW
                stop_short[i] = high[i]     # SHORT stop at bar HIGH
                target_long[i] = high[i-1]  # LONG target = prior bar HIGH
                target_short[i] = low[i-1]  # SHORT target = prior bar LOW

            # 2U bar: Bidirectional setup
            # - LONG (continuation 2U-2U): Break above 2U bar high
            # - SHORT (reversal 2U-2D): Break below 2U bar low
            elif bar_class == 2:
                setups[i] = True
                setup_dir[i] = 2

                long_trigger[i] = high[i]   # Break HIGH for LONG
                short_trigger[i] = low[i]   # Break LOW for SHORT
                stop_long[i] = low[i]       # LONG stop at bar LOW
                stop_short[i] = high[i]     # SHORT stop at bar HIGH
                target_long[i] = high[i-1]  # LONG target = prior bar HIGH
                target_short[i] = low[i-1]  # SHORT target = prior bar LOW

    else:  # 2D arrays
        n = classifications.shape[0]
        setups = np.zeros((n, 1), dtype=np.bool_)
        setup_dir = np.zeros((n, 1), dtype=np.int8)
        long_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        short_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        stop_long = np.full((n, 1), np.nan, dtype=np.float64)
        stop_short = np.full((n, 1), np.nan, dtype=np.float64)
        target_long = np.full((n, 1), np.nan, dtype=np.float64)
        target_short = np.full((n, 1), np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_class = classifications[i, 0]

            if bar_class == -2:
                setups[i, 0] = True
                setup_dir[i, 0] = -2
                long_trigger[i, 0] = high[i, 0]
                short_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i, 0]
                stop_short[i, 0] = high[i, 0]
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

            elif bar_class == 2:
                setups[i, 0] = True
                setup_dir[i, 0] = 2
                long_trigger[i, 0] = high[i, 0]
                short_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i, 0]
                stop_short[i, 0] = high[i, 0]
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

    return (setups, setup_dir, long_trigger, short_trigger,
            stop_long, stop_short, target_long, target_short)


@njit
def detect_322_setups_nb(classifications, high, low):
    """
    Detect 3-2 setups (Outside-Directional) as BIDIRECTIONAL setups.

    Session EQUITY-19: CRITICAL FIX - Return BOTH directions per STRAT methodology.

    Per STRAT: "The forming bar is always treated as '1' until it breaks."

    Setup Structure:
        - Bar at i-1: Outside bar (classification = 3)
        - Bar at i: First directional bar (2D or 2U) <-- SETUP detected here
        - FORMING bar: Treated as implicit "1" until it breaks

    The NEXT bar (forming bar) will determine direction by which bound breaks FIRST:
        - For 3-2D: Break HIGH = 3-2D-2U reversal LONG, Break LOW = 3-2D-2D continuation SHORT
        - For 3-2U: Break HIGH = 3-2U-2U continuation LONG, Break LOW = 3-2U-2D reversal SHORT

    Returns:
    --------
    tuple of 8 np.ndarray:
        setups : Boolean array (True where 3-2 setup exists)
        setup_dir : Direction of directional bar (2 for 2U, -2 for 2D)
        long_trigger : Price to break for LONG entry (directional bar HIGH)
        short_trigger : Price to break for SHORT entry (directional bar LOW)
        stop_long : Stop for LONG trade (directional bar LOW)
        stop_short : Stop for SHORT trade (directional bar HIGH)
        target_long : Target for LONG trade (outside bar HIGH)
        target_short : Target for SHORT trade (outside bar LOW)
    """
    if classifications.ndim == 1:
        n = len(classifications)
        setups = np.zeros(n, dtype=np.bool_)
        setup_dir = np.zeros(n, dtype=np.int8)
        long_trigger = np.full(n, np.nan, dtype=np.float64)
        short_trigger = np.full(n, np.nan, dtype=np.float64)
        stop_long = np.full(n, np.nan, dtype=np.float64)
        stop_short = np.full(n, np.nan, dtype=np.float64)
        target_long = np.full(n, np.nan, dtype=np.float64)
        target_short = np.full(n, np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_outside = classifications[i-1]
            bar_dir = classifications[i]

            # 3-2D setup: Outside followed by 2D (bidirectional)
            # - LONG (reversal 3-2D-2U): Break above 2D bar high
            # - SHORT (continuation 3-2D-2D): Break below 2D bar low
            if bar_outside == 3 and bar_dir == -2:
                setups[i] = True
                setup_dir[i] = -2

                long_trigger[i] = high[i]    # Break HIGH for LONG
                short_trigger[i] = low[i]    # Break LOW for SHORT
                stop_long[i] = low[i]        # LONG stop at 2D bar LOW
                stop_short[i] = high[i]      # SHORT stop at 2D bar HIGH
                target_long[i] = high[i-1]   # LONG target = outside bar HIGH
                target_short[i] = low[i-1]   # SHORT target = outside bar LOW

            # 3-2U setup: Outside followed by 2U (bidirectional)
            # - LONG (continuation 3-2U-2U): Break above 2U bar high
            # - SHORT (reversal 3-2U-2D): Break below 2U bar low
            elif bar_outside == 3 and bar_dir == 2:
                setups[i] = True
                setup_dir[i] = 2

                long_trigger[i] = high[i]    # Break HIGH for LONG
                short_trigger[i] = low[i]    # Break LOW for SHORT
                stop_long[i] = low[i]        # LONG stop at 2U bar LOW
                stop_short[i] = high[i]      # SHORT stop at 2U bar HIGH
                target_long[i] = high[i-1]   # LONG target = outside bar HIGH
                target_short[i] = low[i-1]   # SHORT target = outside bar LOW

    else:  # 2D arrays
        n = classifications.shape[0]
        setups = np.zeros((n, 1), dtype=np.bool_)
        setup_dir = np.zeros((n, 1), dtype=np.int8)
        long_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        short_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        stop_long = np.full((n, 1), np.nan, dtype=np.float64)
        stop_short = np.full((n, 1), np.nan, dtype=np.float64)
        target_long = np.full((n, 1), np.nan, dtype=np.float64)
        target_short = np.full((n, 1), np.nan, dtype=np.float64)

        for i in range(1, n):
            bar_outside = classifications[i-1, 0]
            bar_dir = classifications[i, 0]

            if bar_outside == 3 and bar_dir == -2:
                setups[i, 0] = True
                setup_dir[i, 0] = -2
                long_trigger[i, 0] = high[i, 0]
                short_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i, 0]
                stop_short[i, 0] = high[i, 0]
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

            elif bar_outside == 3 and bar_dir == 2:
                setups[i, 0] = True
                setup_dir[i, 0] = 2
                long_trigger[i, 0] = high[i, 0]
                short_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i, 0]
                stop_short[i, 0] = high[i, 0]
                target_long[i, 0] = high[i-1, 0]
                target_short[i, 0] = low[i-1, 0]

    return (setups, setup_dir, long_trigger, short_trigger,
            stop_long, stop_short, target_long, target_short)


@njit
def detect_outside_bar_setups_nb(classifications, high, low):
    """
    Detect outside bars (Type 3) as BIDIRECTIONAL setups.

    Session EQUITY-19: NEW - Detects pure 3-? setups per STRAT methodology.

    Per STRAT: "The forming bar is always treated as '1' until it breaks."

    Setup Structure:
        - Bar at i: Outside bar (classification = 3) <-- SETUP detected here
        - FORMING bar: Treated as implicit "1" until it breaks

    The NEXT bar (forming bar) will determine direction by which bound breaks FIRST:
        - Break HIGH = 3-2U pattern LONG
        - Break LOW = 3-2D pattern SHORT

    This complements detect_322_setups_nb() which requires BOTH outside + directional
    bars to be closed. This function fires on the outside bar itself.

    Returns:
    --------
    tuple of 7 np.ndarray:
        setups : Boolean array (True where outside bar exists)
        long_trigger : Price to break for LONG entry (outside bar HIGH)
        short_trigger : Price to break for SHORT entry (outside bar LOW)
        stop_long : Stop for LONG trade (outside bar LOW)
        stop_short : Stop for SHORT trade (outside bar HIGH)
        target_long : Target for LONG trade (measured move HIGH)
        target_short : Target for SHORT trade (measured move LOW)
    """
    if classifications.ndim == 1:
        n = len(classifications)
        setups = np.zeros(n, dtype=np.bool_)
        long_trigger = np.full(n, np.nan, dtype=np.float64)
        short_trigger = np.full(n, np.nan, dtype=np.float64)
        stop_long = np.full(n, np.nan, dtype=np.float64)
        stop_short = np.full(n, np.nan, dtype=np.float64)
        target_long = np.full(n, np.nan, dtype=np.float64)
        target_short = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            bar_class = classifications[i]

            # Outside bar: Bidirectional setup
            # - LONG (3-2U): Break above outside bar high
            # - SHORT (3-2D): Break below outside bar low
            if bar_class == 3:
                setups[i] = True

                long_trigger[i] = high[i]   # Break HIGH for LONG
                short_trigger[i] = low[i]   # Break LOW for SHORT
                stop_long[i] = low[i]       # LONG stop at outside bar LOW
                stop_short[i] = high[i]     # SHORT stop at outside bar HIGH

                # Target = measured move (bar range projected)
                bar_range = high[i] - low[i]
                target_long[i] = high[i] + bar_range   # 1R above entry
                target_short[i] = low[i] - bar_range   # 1R below entry

    else:  # 2D arrays
        n = classifications.shape[0]
        setups = np.zeros((n, 1), dtype=np.bool_)
        long_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        short_trigger = np.full((n, 1), np.nan, dtype=np.float64)
        stop_long = np.full((n, 1), np.nan, dtype=np.float64)
        stop_short = np.full((n, 1), np.nan, dtype=np.float64)
        target_long = np.full((n, 1), np.nan, dtype=np.float64)
        target_short = np.full((n, 1), np.nan, dtype=np.float64)

        for i in range(n):
            bar_class = classifications[i, 0]

            if bar_class == 3:
                setups[i, 0] = True
                long_trigger[i, 0] = high[i, 0]
                short_trigger[i, 0] = low[i, 0]
                stop_long[i, 0] = low[i, 0]
                stop_short[i, 0] = high[i, 0]

                bar_range = high[i, 0] - low[i, 0]
                target_long[i, 0] = high[i, 0] + bar_range
                target_short[i, 0] = low[i, 0] - bar_range

    return (setups, long_trigger, short_trigger,
            stop_long, stop_short, target_long, target_short)


@njit
def detect_all_patterns_nb(classifications, high, low):
    """
    Detect all pattern types (3-1-2, 2-1-2, 2-2, 3-2, and 3-2-2) in a single pass.

    Combines all pattern detectors for efficiency and returns all outputs.

    Parameters:
    -----------
    classifications : np.ndarray
        Bar classifications from bar_classifier.py
    high : np.ndarray
        Array of bar high prices
    low : np.ndarray
        Array of bar low prices

    Returns:
    --------
    tuple of 20 np.ndarray:
        entries_312 : Boolean array for 3-1-2 patterns
        stops_312 : Stop prices for 3-1-2 patterns
        targets_312 : Target prices for 3-1-2 patterns
        directions_312 : Directions for 3-1-2 patterns
        entries_212 : Boolean array for 2-1-2 patterns
        stops_212 : Stop prices for 2-1-2 patterns
        targets_212 : Target prices for 2-1-2 patterns
        directions_212 : Directions for 2-1-2 patterns
        entries_22 : Boolean array for 2-2 patterns
        stops_22 : Stop prices for 2-2 patterns
        targets_22 : Target prices for 2-2 patterns
        directions_22 : Directions for 2-2 patterns
        entries_32 : Boolean array for 3-2 patterns
        stops_32 : Stop prices for 3-2 patterns
        targets_32 : Target prices for 3-2 patterns
        directions_32 : Directions for 3-2 patterns
        entries_322 : Boolean array for 3-2-2 patterns
        stops_322 : Stop prices for 3-2-2 patterns
        targets_322 : Target prices for 3-2-2 patterns
        directions_322 : Directions for 3-2-2 patterns
    """
    # Detect 3-1-2 patterns
    entries_312, stops_312, targets_312, directions_312 = detect_312_patterns_nb(
        classifications, high, low
    )

    # Detect 2-1-2 patterns
    entries_212, stops_212, targets_212, directions_212 = detect_212_patterns_nb(
        classifications, high, low
    )

    # Detect 2-2 patterns
    entries_22, stops_22, targets_22, directions_22 = detect_22_patterns_nb(
        classifications, high, low
    )

    # Detect 3-2 patterns
    entries_32, stops_32, targets_32, directions_32 = detect_32_patterns_nb(
        classifications, high, low
    )

    # Detect 3-2-2 patterns (Session 76)
    entries_322, stops_322, targets_322, directions_322 = detect_322_patterns_nb(
        classifications, high, low
    )

    # Return all 20 outputs as tuple (order MUST match output_names)
    return (
        entries_312, stops_312, targets_312, directions_312,
        entries_212, stops_212, targets_212, directions_212,
        entries_22, stops_22, targets_22, directions_22,
        entries_32, stops_32, targets_32, directions_32,
        entries_322, stops_322, targets_322, directions_322
    )


# Create VBT custom indicator with 20 outputs (Session 76: Added 3-2-2 patterns)
StratPatternDetector = vbt.IF(
    class_name='StratPatternDetector',
    input_names=['classifications', 'high', 'low'],
    output_names=[
        'entries_312', 'stops_312', 'targets_312', 'directions_312',
        'entries_212', 'stops_212', 'targets_212', 'directions_212',
        'entries_22', 'stops_22', 'targets_22', 'directions_22',
        'entries_32', 'stops_32', 'targets_32', 'directions_32',
        'entries_322', 'stops_322', 'targets_322', 'directions_322'
    ]
).with_apply_func(detect_all_patterns_nb)


# Convenience function for detecting all patterns
def detect_patterns(classifications, high, low):
    """
    Detect STRAT patterns (3-1-2 and 2-1-2) with entry/stop/target prices.

    Convenience wrapper around StratPatternDetector.run() for single-column data.

    Parameters:
    -----------
    classifications : pd.Series or np.ndarray
        Bar classifications from bar_classifier.py
    high : pd.Series or np.ndarray
        Bar high prices
    low : pd.Series or np.ndarray
        Bar low prices

    Returns:
    --------
    StratPatternDetector instance with attributes:
        .entries_312, .stops_312, .targets_312, .directions_312
        .entries_212, .stops_212, .targets_212, .directions_212

    Examples:
    ---------
    >>> from strat.bar_classifier import classify_bars
    >>> from strat.pattern_detector import detect_patterns
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({
    ...     'high': [100, 110, 105, 108, 112],
    ...     'low': [95, 90, 95, 96, 97]
    ... })
    >>>
    >>> # Step 1: Classify bars
    >>> classifications = classify_bars(data['high'], data['low'])
    >>>
    >>> # Step 2: Detect patterns
    >>> patterns = detect_patterns(classifications, data['high'], data['low'])
    >>>
    >>> # Access 3-1-2 patterns
    >>> patterns.entries_312  # Boolean array of entry signals
    >>> patterns.stops_312    # Stop loss prices
    >>> patterns.targets_312  # Target prices
    >>> patterns.directions_312  # 1=bullish, -1=bearish
    >>>
    >>> # Access 2-1-2 patterns
    >>> patterns.entries_212
    >>> patterns.stops_212
    >>> patterns.targets_212
    >>> patterns.directions_212
    """
    result = StratPatternDetector.run(classifications, high, low)
    return result
