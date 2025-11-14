r"""
ATLAS-STRAT Integration Layer

Combines ATLAS Layer 1 (regime detection) with STRAT Layer 2 (pattern recognition)
to generate signal quality ratings for confluence trading.

Architecture:
    - Mode 3: Integrated ATLAS+STRAT (confluence trading)
    - Signal quality matrix: HIGH/MEDIUM/REJECT
    - CRASH regime has veto power over bullish signals

Signal Quality Ratings:
    HIGH: ATLAS regime and STRAT pattern aligned
        - TREND_BULL + bullish pattern (3-1-2 or 2-1-2 bullish)
        - TREND_BEAR + bearish pattern (3-1-2 or 2-1-2 bearish)
        - Position size: 100% (full allocation)

    MEDIUM: Neutral regime or partial alignment
        - TREND_NEUTRAL + any pattern
        - Mixed signals suggest lower probability
        - Position size: 50% (reduced risk)

    REJECT: Counter-trend or CRASH regime veto
        - CRASH + any bullish pattern (veto power)
        - TREND_BULL + bearish pattern (counter-trend)
        - TREND_BEAR + bullish pattern (counter-trend)
        - Position size: 0% (no trade)

Reference:
    docs/SYSTEM_ARCHITECTURE/INTEGRATION_ARCHITECTURE.md
    Implementation follows Mode 3 architecture (lines 103-164)

Design Principle:
    STRAT and ATLAS are peer systems. This integration layer is OPTIONAL.
    STRAT can operate standalone (Mode 2) or integrated (Mode 3).
    Integration does NOT modify STRAT or ATLAS implementations.
"""

import numpy as np
import pandas as pd


def filter_strat_signals(atlas_regime, pattern_direction):
    """
    Filter STRAT pattern signals based on ATLAS regime detection.

    Implements signal quality matrix for confluence trading (Mode 3).

    Parameters:
    -----------
    atlas_regime : str or pd.Series
        ATLAS regime: 'TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH'
        Can be single string or Series for vectorized operation

    pattern_direction : int or np.ndarray or pd.Series
        STRAT pattern direction:
            1 = bullish pattern (3-1-2 up or 2-1-2 bullish reversal)
           -1 = bearish pattern (3-1-2 down or 2-1-2 bearish reversal)
            0 = no pattern
        Can be single int or array for vectorized operation

    Returns:
    --------
    signal_quality : str or pd.Series
        Signal quality rating: 'HIGH', 'MEDIUM', 'REJECT'

    Examples:
    ---------
    >>> # Single signal evaluation
    >>> filter_strat_signals('TREND_BULL', 1)
    'HIGH'

    >>> # CRASH veto on bullish pattern
    >>> filter_strat_signals('CRASH', 1)
    'REJECT'

    >>> # Vectorized operation
    >>> regimes = pd.Series(['TREND_BULL', 'CRASH', 'TREND_NEUTRAL'])
    >>> directions = pd.Series([1, 1, -1])
    >>> filter_strat_signals(regimes, directions)
    0       HIGH
    1     REJECT
    2     MEDIUM
    dtype: object

    Notes:
    ------
    CRASH regime has ABSOLUTE veto power over bullish patterns.
    This is the highest priority rule in the signal quality matrix.

    Historical validation: March 2020 had 77% CRASH regime detection.
    During this period, bullish STRAT patterns would have failed.
    Veto power prevented losses from counter-trend trades.
    """
    # Handle scalar inputs
    if isinstance(atlas_regime, str) and isinstance(pattern_direction, (int, np.integer)):
        return _filter_single_signal(atlas_regime, pattern_direction)

    # Handle vectorized inputs (pandas Series or numpy arrays)
    if isinstance(atlas_regime, pd.Series):
        # Use pandas vectorized operations
        result = pd.Series(index=atlas_regime.index, dtype=object)

        # Convert pattern_direction to Series if needed
        if isinstance(pattern_direction, np.ndarray):
            pattern_direction = pd.Series(pattern_direction, index=atlas_regime.index)

        # Apply filtering logic to each element
        for idx in atlas_regime.index:
            regime = atlas_regime.loc[idx]
            direction = pattern_direction.loc[idx]
            result.loc[idx] = _filter_single_signal(regime, direction)

        return result

    # Handle numpy array inputs
    if isinstance(pattern_direction, np.ndarray):
        # Convert to Series for consistent handling
        if not isinstance(atlas_regime, pd.Series):
            atlas_regime = pd.Series(atlas_regime)
            pattern_direction = pd.Series(pattern_direction)

        return filter_strat_signals(atlas_regime, pattern_direction)

    raise TypeError(
        f"Unsupported types: atlas_regime={type(atlas_regime)}, "
        f"pattern_direction={type(pattern_direction)}"
    )


def _filter_single_signal(atlas_regime, pattern_direction):
    """
    Filter a single signal (internal helper function).

    Parameters:
    -----------
    atlas_regime : str
        ATLAS regime classification

    pattern_direction : int
        Pattern direction (1=bullish, -1=bearish, 0=no pattern)

    Returns:
    --------
    str : Signal quality ('HIGH', 'MEDIUM', 'REJECT')
    """
    # No pattern detected - reject
    if pattern_direction == 0:
        return 'REJECT'

    # CRASH VETO: Absolute veto power over bullish patterns
    # This is the HIGHEST PRIORITY rule
    if atlas_regime == 'CRASH' and pattern_direction > 0:
        return 'REJECT'

    # HIGH QUALITY: Regime and pattern aligned
    if atlas_regime == 'TREND_BULL' and pattern_direction > 0:
        return 'HIGH'  # Bullish regime + bullish pattern

    if atlas_regime == 'TREND_BEAR' and pattern_direction < 0:
        return 'HIGH'  # Bearish regime + bearish pattern

    # MEDIUM QUALITY: Neutral regime (no directional bias)
    if atlas_regime == 'TREND_NEUTRAL':
        return 'MEDIUM'  # Pattern-dependent, reduced size

    # REJECT: Counter-trend patterns
    if atlas_regime == 'TREND_BULL' and pattern_direction < 0:
        return 'REJECT'  # Bearish pattern in bullish regime

    if atlas_regime == 'TREND_BEAR' and pattern_direction > 0:
        return 'REJECT'  # Bullish pattern in bearish regime

    # CRASH regime + bearish pattern: Allow (shorts in crash)
    if atlas_regime == 'CRASH' and pattern_direction < 0:
        return 'MEDIUM'  # Reduce size during extreme volatility

    # Fallback (should not reach here if all cases covered)
    return 'REJECT'


def get_position_size_multiplier(signal_quality):
    """
    Get position size multiplier based on signal quality.

    Parameters:
    -----------
    signal_quality : str or pd.Series
        Signal quality rating: 'HIGH', 'MEDIUM', 'REJECT'

    Returns:
    --------
    float or pd.Series
        Position size multiplier:
            HIGH: 1.0 (100% of target position)
            MEDIUM: 0.5 (50% of target position)
            REJECT: 0.0 (no trade)

    Examples:
    ---------
    >>> get_position_size_multiplier('HIGH')
    1.0

    >>> get_position_size_multiplier('REJECT')
    0.0

    >>> qualities = pd.Series(['HIGH', 'MEDIUM', 'REJECT'])
    >>> get_position_size_multiplier(qualities)
    0    1.0
    1    0.5
    2    0.0
    dtype: float64
    """
    # Define multipliers
    multipliers = {
        'HIGH': 1.0,
        'MEDIUM': 0.5,
        'REJECT': 0.0
    }

    # Handle scalar input
    if isinstance(signal_quality, str):
        return multipliers.get(signal_quality, 0.0)

    # Handle vectorized input (pandas Series)
    if isinstance(signal_quality, pd.Series):
        return signal_quality.map(multipliers).fillna(0.0)

    raise TypeError(f"Unsupported type: {type(signal_quality)}")


def combine_pattern_signals(entries_312, directions_312, entries_212, directions_212):
    """
    Combine 3-1-2 and 2-1-2 pattern signals into unified direction array.

    When both patterns trigger on same bar, prioritize 3-1-2 (stronger reversal signal).

    Parameters:
    -----------
    entries_312 : np.ndarray or pd.Series
        Boolean array of 3-1-2 pattern entry triggers

    directions_312 : np.ndarray or pd.Series
        Direction array for 3-1-2 patterns (1=bullish, -1=bearish, 0=none)

    entries_212 : np.ndarray or pd.Series
        Boolean array of 2-1-2 pattern entry triggers

    directions_212 : np.ndarray or pd.Series
        Direction array for 2-1-2 patterns (1=bullish, -1=bearish, 0=none)

    Returns:
    --------
    combined_direction : np.ndarray or pd.Series
        Combined pattern direction (1=bullish, -1=bearish, 0=no pattern)
        Prioritizes 3-1-2 over 2-1-2 when both trigger

    Examples:
    ---------
    >>> entries_312 = np.array([False, True, False])
    >>> directions_312 = np.array([0, 1, 0])
    >>> entries_212 = np.array([False, False, True])
    >>> directions_212 = np.array([0, 0, -1])
    >>> combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)
    array([0, 1, -1])

    >>> # 3-1-2 takes priority when both trigger
    >>> entries_312 = np.array([True])
    >>> directions_312 = np.array([1])
    >>> entries_212 = np.array([True])
    >>> directions_212 = np.array([-1])
    >>> combine_pattern_signals(entries_312, directions_312, entries_212, directions_212)
    array([1])  # 3-1-2 bullish wins over 2-1-2 bearish
    """
    # Convert to numpy arrays for consistent handling
    if isinstance(entries_312, pd.Series):
        index = entries_312.index
        entries_312 = entries_312.values
        directions_312 = directions_312.values
        entries_212 = entries_212.values
        directions_212 = directions_212.values
        return_series = True
    else:
        return_series = False
        index = None

    # Initialize combined direction array
    combined = np.zeros(len(entries_312), dtype=np.int8)

    # Apply 2-1-2 patterns first (lower priority)
    combined[entries_212] = directions_212[entries_212]

    # Apply 3-1-2 patterns second (higher priority, overwrites 2-1-2 if conflict)
    combined[entries_312] = directions_312[entries_312]

    # Return as Series if input was Series
    if return_series:
        return pd.Series(combined, index=index)

    return combined
