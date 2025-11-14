r"""
STRAT Pattern Detection VBT Custom Indicator

Implements 3-1-2 and 2-1-2 pattern detection with measured move targets.

Pattern Types:
    3-1-2: Outside-Inside-Directional (reversal pattern)
        - Bar 1: Outside bar (classification = 3)
        - Bar 2: Inside bar (classification = 1)
        - Bar 3: Directional bar (classification = 2 or -2)

    2-1-2: Directional-Inside-Directional (continuation/reversal pattern)
        - Bar 1: First directional bar (classification = 2 or -2)
        - Bar 2: Inside bar (classification = 1)
        - Bar 3: Second directional bar (classification = 2 or -2)

Entry/Stop/Target Calculation:
    Entry: Inside bar high (bullish) or low (bearish)
    Stop: Structural level (Outside bar or Inside bar opposite extreme)
    Target: Measured move = entry + pattern_height

Algorithm ported from:
    C:\STRAT-Algorithmic-Trading-System-V3\core\analyzer.py lines 521-674
    (verified CORRECT implementation per OLD_STRAT_SYSTEM_ANALYSIS.md)

CRITICAL: Uses MEASURED MOVE targets, NOT structural levels
    (Old system bug: used data['high'].iloc[idx-2] which was incorrect)
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
        Bar 0 (idx=2): Outside (H=110, L=90, classification=3, range=20)
        Bar 1 (idx=3): Inside (H=105, L=95, classification=1)
        Bar 2 (idx=4): 2U directional (classification=2) - TRIGGER

        Entry: 105 (inside bar high)
        Stop: 90 (outside bar low)
        Target: 105 + 20 = 125 (measured move)
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
                    # Target: Measured move (outside bar range projected from entry)
                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-2]  # Outside bar low

                    pattern_height = high[i-2] - low[i-2]  # Outside bar range
                    targets[i] = trigger_price + pattern_height

                # Bearish 3-1-2D pattern (reversal to downside)
                elif bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Outside bar high (structural level)
                    # Target: Measured move (outside bar range projected from entry)
                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-2]  # Outside bar high

                    pattern_height = high[i-2] - low[i-2]  # Outside bar range
                    targets[i] = trigger_price - pattern_height

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
                    # Target: Measured move (outside bar range projected from entry)
                    trigger_price = high[i-1, 0]  # Inside bar high
                    stops[i, 0] = low[i-2, 0]  # Outside bar low

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # Outside bar range
                    targets[i, 0] = trigger_price + pattern_height

                # Bearish 3-1-2D pattern (reversal to downside)
                elif bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Outside bar high (structural level)
                    # Target: Measured move (outside bar range projected from entry)
                    trigger_price = low[i-1, 0]  # Inside bar low
                    stops[i, 0] = high[i-2, 0]  # Outside bar high

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # Outside bar range
                    targets[i, 0] = trigger_price - pattern_height

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
        Bar 0 (idx=2): 2U directional (H=105, L=96, classification=2, range=9)
        Bar 1 (idx=3): Inside (H=104, L=97, classification=1)
        Bar 2 (idx=4): 2U directional (classification=2) - TRIGGER

        Entry: 104 (inside bar high)
        Stop: 97 (inside bar low)
        Target: 104 + 9 = 113 (measured move using first directional bar range)
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
                    # Stop: Inside bar low (tighter stop than 3-1-2)
                    # Target: Measured move (first directional bar range)
                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-1]  # Inside bar low

                    pattern_height = high[i-2] - low[i-2]  # First directional bar range
                    targets[i] = trigger_price + pattern_height

                # Bearish continuation: 2D-1-2D
                elif bar1_class == -2 and bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Inside bar high (tighter stop than 3-1-2)
                    # Target: Measured move (first directional bar range)
                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-1]  # Inside bar high

                    pattern_height = high[i-2] - low[i-2]  # First directional bar range
                    targets[i] = trigger_price - pattern_height

                # Bullish reversal: 2D-1-2U (failed breakdown)
                elif bar1_class == -2 and bar3_class == 2:
                    entries[i] = True
                    directions[i] = 1  # Bullish

                    trigger_price = high[i-1]  # Inside bar high
                    stops[i] = low[i-1]  # Inside bar low

                    pattern_height = high[i-2] - low[i-2]  # First directional bar range
                    targets[i] = trigger_price + pattern_height

                # Bearish reversal: 2U-1-2D (failed breakout)
                elif bar1_class == 2 and bar3_class == -2:
                    entries[i] = True
                    directions[i] = -1  # Bearish

                    trigger_price = low[i-1]  # Inside bar low
                    stops[i] = high[i-1]  # Inside bar high

                    pattern_height = high[i-2] - low[i-2]  # First directional bar range
                    targets[i] = trigger_price - pattern_height

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

                    # Entry trigger: Inside bar high
                    # Stop: Inside bar low (tighter stop than 3-1-2)
                    # Target: Measured move (first directional bar range)
                    trigger_price = high[i-1, 0]  # Inside bar high
                    stops[i, 0] = low[i-1, 0]  # Inside bar low

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # First directional bar range
                    targets[i, 0] = trigger_price + pattern_height

                # Bearish continuation: 2D-1-2D
                elif bar1_class == -2 and bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    # Entry trigger: Inside bar low
                    # Stop: Inside bar high (tighter stop than 3-1-2)
                    # Target: Measured move (first directional bar range)
                    trigger_price = low[i-1, 0]  # Inside bar low
                    stops[i, 0] = high[i-1, 0]  # Inside bar high

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # First directional bar range
                    targets[i, 0] = trigger_price - pattern_height

                # Bullish reversal: 2D-1-2U (failed breakdown)
                elif bar1_class == -2 and bar3_class == 2:
                    entries[i, 0] = True
                    directions[i, 0] = 1  # Bullish

                    trigger_price = high[i-1, 0]  # Inside bar high
                    stops[i, 0] = low[i-1, 0]  # Inside bar low

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # First directional bar range
                    targets[i, 0] = trigger_price + pattern_height

                # Bearish reversal: 2U-1-2D (failed breakout)
                elif bar1_class == 2 and bar3_class == -2:
                    entries[i, 0] = True
                    directions[i, 0] = -1  # Bearish

                    trigger_price = low[i-1, 0]  # Inside bar low
                    stops[i, 0] = high[i-1, 0]  # Inside bar high

                    pattern_height = high[i-2, 0] - low[i-2, 0]  # First directional bar range
                    targets[i, 0] = trigger_price - pattern_height

    return (entries, stops, targets, directions)


@njit
def detect_all_patterns_nb(classifications, high, low):
    """
    Detect all pattern types (3-1-2 and 2-1-2) in a single pass.

    Combines both pattern detectors for efficiency and returns all outputs.

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
    tuple of 8 np.ndarray:
        entries_312 : Boolean array for 3-1-2 patterns
        stops_312 : Stop prices for 3-1-2 patterns
        targets_312 : Target prices for 3-1-2 patterns
        directions_312 : Directions for 3-1-2 patterns
        entries_212 : Boolean array for 2-1-2 patterns
        stops_212 : Stop prices for 2-1-2 patterns
        targets_212 : Target prices for 2-1-2 patterns
        directions_212 : Directions for 2-1-2 patterns
    """
    # Detect 3-1-2 patterns
    entries_312, stops_312, targets_312, directions_312 = detect_312_patterns_nb(
        classifications, high, low
    )

    # Detect 2-1-2 patterns
    entries_212, stops_212, targets_212, directions_212 = detect_212_patterns_nb(
        classifications, high, low
    )

    # Return all 8 outputs as tuple (order MUST match output_names)
    return (
        entries_312, stops_312, targets_312, directions_312,
        entries_212, stops_212, targets_212, directions_212
    )


# Create VBT custom indicator with 8 outputs
StratPatternDetector = vbt.IF(
    class_name='StratPatternDetector',
    input_names=['classifications', 'high', 'low'],
    output_names=[
        'entries_312', 'stops_312', 'targets_312', 'directions_312',
        'entries_212', 'stops_212', 'targets_212', 'directions_212'
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
