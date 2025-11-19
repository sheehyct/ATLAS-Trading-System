"""
STRAT Bar Classification VBT Custom Indicator

Implements Rob Smith's STRAT bar classification using previous bar comparison.

Bar Types:
    -999: Reference bar (first bar, not classified)
    1: Inside bar (does not break previous bar's high or low)
    2: 2U - Breaks previous bar's high only (upward directional)
    -2: 2D - Breaks previous bar's low only (downward directional)
    3: Outside bar (breaks both previous bar's high and low)

Classification Method:
    Each bar is classified by comparing its high/low to the IMMEDIATELY
    PREVIOUS bar's high/low. This is the standard STRAT methodology.
"""

import numpy as np
from numba import njit
import vectorbtpro as vbt


@njit
def classify_bars_nb(high, low):
    """
    Classify bars using previous bar comparison (standard STRAT methodology).

    Each bar is classified by comparing its high/low to the immediately
    previous bar's high/low.

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
        1: Inside bar (does not break previous bar's high or low)
        2: 2U (breaks previous bar's high only)
        -2: 2D (breaks previous bar's low only)
        3: Outside bar (breaks both previous bar's high and low)

    Examples:
    ---------
    >>> high = np.array([100, 105, 104, 107, 110])
    >>> low = np.array([95, 98, 99, 101, 93])
    >>> classify_bars_nb(high, low)
    array([-999., 2., 1., 2., 3.])

    Explanation:
    - Bar 0 (H=100, L=95): Reference bar (-999)
    - Bar 1 (H=105, L=98): Compare to Bar 0: 105>100 and 98>=95 -> 2U
    - Bar 2 (H=104, L=99): Compare to Bar 1: 104<=105 and 99>=98 -> 1 (inside)
    - Bar 3 (H=107, L=101): Compare to Bar 2: 107>104 and 101>=99 -> 2U
    - Bar 4 (H=110, L=93): Compare to Bar 3: 110>107 and 93<101 -> 3 (outside)
    """
    n_bars = len(high)
    classifications = np.full(n_bars, np.nan)

    # First bar is reference (not classified)
    classifications[0] = -999

    # Classify remaining bars against PREVIOUS bar
    for i in range(1, n_bars):
        current_high = high[i]
        current_low = low[i]
        prev_high = high[i - 1]
        prev_low = low[i - 1]

        # Compare to PREVIOUS bar (not governing range)
        breaks_high = current_high > prev_high
        breaks_low = current_low < prev_low

        if breaks_high and breaks_low:
            # Outside bar - breaks both previous bar's high and low
            classifications[i] = 3
        elif breaks_high:
            # 2U - breaks previous bar's high only (upward directional)
            classifications[i] = 2
        elif breaks_low:
            # 2D - breaks previous bar's low only (downward directional)
            classifications[i] = -2
        else:
            # Inside bar - does not break previous bar's high or low
            classifications[i] = 1

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


def format_bar_classifications(classifications, skip_reference=True):
    """
    Format bar classifications into human-readable strings.

    Parameters:
    -----------
    classifications : pd.Series or np.ndarray
        Numeric bar classifications from classify_bars()
    skip_reference : bool, default True
        If True, skip the reference bar (-999) in output

    Returns:
    --------
    list of str
        List of classification labels: "1", "2U", "2D", "3", or "REF"

    Examples:
    ---------
    >>> classifications = np.array([-999, 2, 1, 2, 3])
    >>> format_bar_classifications(classifications)
    ['2U', '1', '2U', '3']
    >>> format_bar_classifications(classifications, skip_reference=False)
    ['REF', '2U', '1', '2U', '3']
    """
    # Mapping of numeric codes to labels
    label_map = {
        -999: 'REF',
        1: '1',
        2: '2U',
        -2: '2D',
        3: '3'
    }

    # Convert to list, handling both numpy arrays and pandas Series
    if hasattr(classifications, 'values'):
        values = classifications.values
    else:
        values = classifications

    # Format each classification
    labels = []
    for i, val in enumerate(values):
        if np.isnan(val):
            continue
        label = label_map.get(int(val), 'UNK')
        if skip_reference and label == 'REF':
            continue
        labels.append(label)

    return labels


def get_bar_sequence_string(classifications, skip_reference=True):
    """
    Get bar classifications as a comma-separated string (oldest to newest).

    Parameters:
    -----------
    classifications : pd.Series or np.ndarray
        Numeric bar classifications from classify_bars()
    skip_reference : bool, default True
        If True, skip the reference bar (-999) in output

    Returns:
    --------
    str
        Comma-separated bar classification sequence (e.g., "3, 2D, 2D, 2U, 1, 2U")

    Examples:
    ---------
    >>> classifications = np.array([-999, 2, 1, 2, 3])
    >>> get_bar_sequence_string(classifications)
    '2U, 1, 2U, 3'
    """
    labels = format_bar_classifications(classifications, skip_reference)
    return ', '.join(labels)
