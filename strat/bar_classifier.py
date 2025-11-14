"""
STRAT Bar Classification VBT Custom Indicator

Implements Rob Smith's STRAT bar classification with governing range tracking.

Bar Types:
    -999: Reference bar (first bar, not classified)
    1: Inside bar (contained within current governing range)
    2: 2U - Breaks high only (upward directional)
    -2: 2D - Breaks low only (downward directional)
    3: Outside bar (breaks both high and low)

Governing Range:
    Consecutive inside bars reference the SAME governing range until broken
    by a directional (2U/2D) or outside (3) bar.

Algorithm ported from:
    C:\STRAT-Algorithmic-Trading-System-V3\core\analyzer.py lines 137-212
    (verified CORRECT implementation per OLD_STRAT_SYSTEM_ANALYSIS.md)
"""

import numpy as np
from numba import njit
import vectorbtpro as vbt


@njit
def classify_bars_nb(high, low):
    """
    Classify bars using governing range tracking.

    The governing range is established by the first bar and updated only when
    a bar breaks high (2U), breaks low (2D), or breaks both (3). Inside bars (1)
    do NOT update the governing range.

    Parameters:
    -----------
    high : np.ndarray
        Array of bar high prices
    low : np.ndarray
        Array of bar low prices

    Returns:
    --------
    np.ndarray
        Array of classifications:
        -999: Reference bar (first bar)
        1: Inside bar
        2: 2U (breaks high only)
        -2: 2D (breaks low only)
        3: Outside bar (breaks both)

    Examples:
    ---------
    >>> high = np.array([100, 105, 104, 107, 110])
    >>> low = np.array([95, 98, 99, 101, 93])
    >>> classify_bars_nb(high, low)
    array([-999., 2., 1., 2., 3.])

    Explanation:
    - Bar 0 (H=100, L=95): Reference bar (-999)
    - Bar 1 (H=105, L=98): Breaks high only -> 2U (gov_range now 105/95)
    - Bar 2 (H=104, L=99): Inside 105/95 -> 1 (gov_range unchanged)
    - Bar 3 (H=107, L=101): Breaks high -> 2U (gov_range now 107/95)
    - Bar 4 (H=110, L=93): Breaks both -> 3 (gov_range now 110/93)
    """
    n_bars = len(high)
    classifications = np.full(n_bars, np.nan)

    # First bar is reference (not classified)
    classifications[0] = -999
    governing_high = high[0]
    governing_low = low[0]

    # Classify remaining bars against governing range
    for i in range(1, n_bars):
        current_high = high[i]
        current_low = low[i]

        # Check against GOVERNING range (not previous bar)
        is_inside = (current_high <= governing_high) and (current_low >= governing_low)
        breaks_high = current_high > governing_high
        breaks_low = current_low < governing_low

        if is_inside:
            # Inside bar - governing range UNCHANGED
            classifications[i] = 1
        elif breaks_high and breaks_low:
            # Outside bar - breaks both bounds
            classifications[i] = 3
            governing_high = current_high
            governing_low = current_low
        elif breaks_high:
            # 2U - breaks high only (upward directional)
            classifications[i] = 2
            governing_high = current_high
            governing_low = current_low
        elif breaks_low:
            # 2D - breaks low only (downward directional)
            classifications[i] = -2
            governing_high = current_high
            governing_low = current_low

    return classifications


# Create VBT custom indicator
StratBarClassifier = vbt.IF(
    class_name='StratBarClassifier',
    input_names=['high', 'low'],
    output_names=['classification'],
).with_apply_func(classify_bars_nb)


# Convenience function for single-column usage
def classify_bars(high, low):
    """
    Classify bars using STRAT methodology.

    Convenience wrapper around StratBarClassifier.run() for single-column data.

    Parameters:
    -----------
    high : pd.Series or np.ndarray
        Bar high prices
    low : pd.Series or np.ndarray
        Bar low prices

    Returns:
    --------
    pd.Series or np.ndarray
        Bar classifications (same type as input)

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'high': [100, 105, 104, 107, 110],
    ...     'low': [95, 98, 99, 101, 93]
    ... })
    >>> result = classify_bars(data['high'], data['low'])
    >>> result.classification
    0   -999.0
    1      2.0
    2      1.0
    3      2.0
    4      3.0
    Name: classification, dtype: float64
    """
    result = StratBarClassifier.run(high, low)
    return result.classification
