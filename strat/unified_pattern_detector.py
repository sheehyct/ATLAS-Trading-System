"""
Unified STRAT Pattern Detector - Single Source of Truth

This module provides a unified pattern detection function that is used by BOTH
paper trading AND backtesting to ensure consistent results.

Created: Session EQUITY-38 (2025-12-29)
Purpose: Fix pattern ordering bug and missing patterns in backtest

Key Features:
    - Detects ALL 5 STRAT pattern types (2-2, 3-2, 3-2-2, 2-1-2, 3-1-2)
    - Returns patterns in CHRONOLOGICAL order (not grouped by type)
    - Includes 2-2 Down (2U-2D) patterns by default
    - Uses full bar sequence naming per CLAUDE.md Section 13

Pattern Types:
    2-2:   Directional-Directional reversal (2D-2U bullish, 2U-2D bearish)
    3-2:   Outside-Directional (outside bar breakout/rejection)
    3-2-2: Outside-Directional-Reversal (failed breakout)
    2-1-2: Directional-Inside-Directional (continuation or reversal)
    3-1-2: Outside-Inside-Directional (reversal pattern)

Usage:
    from strat.unified_pattern_detector import detect_all_patterns, PatternDetectionConfig

    # Detect all patterns (default config)
    patterns = detect_all_patterns(ohlc_data, timeframe='1D')

    # Detect with custom config
    config = PatternDetectionConfig(include_22_down=False)
    patterns = detect_all_patterns(ohlc_data, config, timeframe='1D')
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import (
    detect_312_patterns_nb,
    detect_212_patterns_nb,
    detect_22_patterns_nb,
    detect_32_patterns_nb,
    detect_322_patterns_nb,
)


def calculate_atr_target(
    entry: float,
    direction: int,
    atr: float,
    multiplier: float = 1.5
) -> float:
    """
    Calculate ATR-based target for 3-2 patterns (EQUITY-52).

    Replaces fixed 1.5% measured move with dynamic ATR-based target that
    scales appropriately across price ranges (COIN $300 vs ACHR $8).

    Formula: target = entry +/- (ATR * multiplier)

    Args:
        entry: Entry price
        direction: 1 for bullish (CALL), -1 for bearish (PUT)
        atr: ATR(14) value at detection time
        multiplier: ATR multiplier for target distance (default 1.5)

    Returns:
        Target price. Returns 0.0 if ATR is invalid (caller should use fallback).
    """
    if atr <= 0:
        return 0.0  # Invalid ATR - caller should fall back to measured move

    target_distance = atr * multiplier

    if direction > 0:  # Bullish (CALL)
        return entry + target_distance
    else:  # Bearish (PUT)
        return entry - target_distance


def _calculate_atr_from_df(
    df: pd.DataFrame,
    current_idx: int,
    period: int = 14
) -> float:
    """
    Calculate ATR up to current index for use in pattern detection (EQUITY-52).

    Uses True Range formula: TR = max(H-L, |H-Cprev|, |L-Cprev|)
    ATR = Simple moving average of TR over period.

    Args:
        df: OHLC DataFrame with High, Low, Close columns
        current_idx: Current bar index (calculates ATR using bars 0 to current_idx)
        period: ATR lookback period (default 14)

    Returns:
        ATR value. Returns 0.0 if insufficient data.
    """
    # Need at least period+1 bars for TR calculation
    if current_idx < period:
        return 0.0

    # Need Close column for ATR
    if 'Close' not in df.columns:
        return 0.0

    try:
        high = df['High'].values[:current_idx + 1]
        low = df['Low'].values[:current_idx + 1]
        close = df['Close'].values[:current_idx + 1]

        # Calculate True Range components
        # TR1 = High - Low (current bar range)
        tr1 = high[1:] - low[1:]
        # TR2 = |High - Previous Close| (gap up capture)
        tr2 = np.abs(high[1:] - close[:-1])
        # TR3 = |Low - Previous Close| (gap down capture)
        tr3 = np.abs(low[1:] - close[:-1])

        # True Range = max of all three components
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        if len(tr) < period:
            return 0.0

        # ATR = Simple moving average of last 'period' TR values
        atr = float(np.mean(tr[-period:]))
        return atr if not np.isnan(atr) else 0.0
    except Exception:
        return 0.0


def apply_timeframe_adjustment(
    target: float,
    entry: float,
    stop: float,
    direction: int,
    timeframe: str,
    pattern_type: str = ''
) -> float:
    """
    Apply timeframe-specific target adjustment (EQUITY-39, updated EQUITY-52).

    This is part of the SINGLE SOURCE OF TRUTH for target calculation.
    Ensures paper trading and backtesting use identical target methodology.

    Rules:
        1. 3-2 patterns: Use ATR-based targets for ALL timeframes (EQUITY-52)
           - Bypasses 1H 1.0x override from EQUITY-36
           - Target already calculated as entry +/- (ATR * 1.5)
        2. Non-3-2 on 1H: Use 1.0x R:R (EQUITY-36 - intraday targets)
        3. 1D/1W/1M with valid geometry: Use structural target unchanged
        4. Invalid geometry: 1.5x fallback (handled earlier in _detect_single_pattern_type)

    Args:
        target: Target price from pattern detector (may be structural, ATR, or fallback)
        entry: Entry trigger price
        stop: Stop loss price
        direction: 1 (bullish/CALL) or -1 (bearish/PUT)
        timeframe: Timeframe string ('1H', '1D', '1W', '1M', '60MIN', '60M')
        pattern_type: Full bar sequence (e.g., '3-2U', '3-2D', '2D-2U') for bypass logic

    Returns:
        Adjusted target price (ATR-based for 3-2, 1.0x R:R for non-3-2 1H, unchanged for others)
    """
    # EQUITY-52: 3-2 patterns use ATR-based targets for ALL timeframes
    # This overrides the EQUITY-36 1.0x rule for 3-2 specifically
    # Pattern type contains '3-2' but not '3-2-2' (which is a different pattern)
    is_32_pattern = '3-2' in pattern_type and '3-2-2' not in pattern_type

    if is_32_pattern:
        # 3-2 patterns bypass timeframe adjustment - keep ATR-based target
        return target

    # Non-3-2 patterns on 1H: Force 1.0x R:R (EQUITY-36)
    # Intraday targets are more conservative due to shorter holding periods
    if timeframe.upper() in ['1H', '60MIN', '60M']:
        risk = abs(entry - stop)
        if direction > 0:  # Bullish
            return entry + risk  # 1.0x R:R
        else:  # Bearish
            return entry - risk  # 1.0x R:R

    # All other timeframes (1D, 1W, 1M): Return structural target unchanged
    return target


# All supported pattern types in detection order
ALL_PATTERN_TYPES = ['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']


@dataclass
class PatternDetectionConfig:
    """
    Configuration for pattern detection.

    Default configuration detects ALL patterns including 2-2 Down (2U-2D).
    This matches paper trading behavior for data collection purposes.

    Attributes:
        include_22: Include 2-2 patterns (2D-2U, 2U-2D)
        include_32: Include 3-2 patterns (3-2U, 3-2D)
        include_322: Include 3-2-2 patterns
        include_212: Include 2-1-2 patterns
        include_312: Include 3-1-2 patterns
        include_22_down: Include 2U-2D patterns (bearish 2-2)
        include_bullish: Include bullish patterns (CALL direction)
        include_bearish: Include bearish patterns (PUT direction)
        min_risk_reward: Minimum R:R ratio to include (0 = no filter)
        max_magnitude_pct: Maximum magnitude % to include (filter outliers)
        sort_chronologically: Sort by timestamp (CRITICAL - must be True)
    """
    include_22: bool = True
    include_32: bool = True
    include_322: bool = True
    include_212: bool = True
    include_312: bool = True
    include_22_down: bool = True  # Changed from False: now matches paper trading
    include_bullish: bool = True
    include_bearish: bool = True
    min_risk_reward: float = 0.0
    max_magnitude_pct: float = 100.0
    sort_chronologically: bool = True

    def get_enabled_pattern_types(self) -> List[str]:
        """Return list of enabled pattern types."""
        types = []
        if self.include_22:
            types.append('2-2')
        if self.include_32:
            types.append('3-2')
        if self.include_322:
            types.append('3-2-2')
        if self.include_212:
            types.append('2-1-2')
        if self.include_312:
            types.append('3-1-2')
        return types


def _bar_to_str(bar_class: int) -> str:
    """
    Convert numeric bar classification to string.

    Per CLAUDE.md Section 13: Every directional bar MUST be classified as 2U or 2D.

    Args:
        bar_class: Bar classification (-999=ref, 1=inside, 2=2U, -2=2D, 3=outside)

    Returns:
        String representation ('1', '2U', '2D', '3')
    """
    if bar_class == 1:
        return "1"
    elif bar_class == 2:
        return "2U"
    elif bar_class == -2:
        return "2D"
    elif abs(bar_class) == 3:
        return "3"
    else:
        return "?"


def _get_full_bar_sequence(
    pattern_type: str,
    classifications: np.ndarray,
    idx: int,
    direction: int
) -> str:
    """
    Get full bar sequence string for pattern.

    Per CLAUDE.md Section 13: Use full bar sequence (2D-2U not "2-2 Up").

    Args:
        pattern_type: Base pattern type ('2-2', '3-2', '3-2-2', '2-1-2', '3-1-2')
        classifications: Array of bar classifications
        idx: Index of trigger bar
        direction: 1 for bullish, -1 for bearish

    Returns:
        Full bar sequence string (e.g., '2D-2U', '3-1-2U', '2U-1-2D')
    """
    # Not enough bars for pattern
    if idx < 2:
        return f"{pattern_type}{'U' if direction > 0 else 'D'}"

    if pattern_type == '2-2':
        # 2-2: bar at i-1 (first 2) + bar at i (trigger 2)
        bar1 = int(classifications[idx - 1])
        bar2 = int(classifications[idx])
        return f"{_bar_to_str(bar1)}-{_bar_to_str(bar2)}"

    elif pattern_type == '3-2':
        # 3-2: outside bar at i-1 + directional bar at i
        bar1 = int(classifications[idx - 1])
        bar2 = int(classifications[idx])
        return f"{_bar_to_str(bar1)}-{_bar_to_str(bar2)}"

    elif pattern_type == '3-2-2':
        # 3-2-2: outside bar at i-2 + directional at i-1 + directional at i
        if idx < 2:
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"
        bar1 = int(classifications[idx - 2])
        bar2 = int(classifications[idx - 1])
        bar3 = int(classifications[idx])
        return f"{_bar_to_str(bar1)}-{_bar_to_str(bar2)}-{_bar_to_str(bar3)}"

    elif pattern_type == '2-1-2':
        # 2-1-2: directional at i-2 + inside at i-1 + directional at i
        if idx < 2:
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"
        bar1 = int(classifications[idx - 2])
        bar2 = int(classifications[idx - 1])
        bar3 = int(classifications[idx])
        return f"{_bar_to_str(bar1)}-{_bar_to_str(bar2)}-{_bar_to_str(bar3)}"

    elif pattern_type == '3-1-2':
        # 3-1-2: outside at i-2 + inside at i-1 + directional at i
        if idx < 2:
            return f"{pattern_type}{'U' if direction > 0 else 'D'}"
        bar1 = int(classifications[idx - 2])
        bar2 = int(classifications[idx - 1])
        bar3 = int(classifications[idx])
        return f"{_bar_to_str(bar1)}-{_bar_to_str(bar2)}-{_bar_to_str(bar3)}"

    else:
        # Unknown pattern type, return simple format
        return f"{pattern_type}{'U' if direction > 0 else 'D'}"


def _detect_single_pattern_type(
    df: pd.DataFrame,
    pattern_type: str,
    classifications: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    config: PatternDetectionConfig
) -> List[Dict]:
    """
    Detect a single pattern type.

    This mirrors the paper_signal_scanner._detect_patterns() logic exactly.

    Args:
        df: OHLC DataFrame with DatetimeIndex
        pattern_type: Pattern type to detect ('2-2', '3-2', etc.)
        classifications: Array of bar classifications
        high: Array of high prices
        low: Array of low prices
        config: Detection configuration

    Returns:
        List of pattern dicts for this pattern type
    """
    patterns = []

    # Get detector outputs based on pattern type
    if pattern_type == '2-2':
        result = detect_22_patterns_nb(classifications, high, low)
    elif pattern_type == '3-2':
        result = detect_32_patterns_nb(classifications, high, low)
    elif pattern_type == '3-2-2':
        result = detect_322_patterns_nb(classifications, high, low)
    elif pattern_type == '2-1-2':
        result = detect_212_patterns_nb(classifications, high, low)
    elif pattern_type == '3-1-2':
        result = detect_312_patterns_nb(classifications, high, low)
    else:
        return []

    entries_mask, stops, targets, directions = result[:4]

    # Extract pattern occurrences
    for i in range(len(entries_mask)):
        if not entries_mask[i]:
            continue

        direction = int(directions[i])
        direction_str = 'CALL' if direction > 0 else 'PUT'

        # Apply direction filters
        if direction > 0 and not config.include_bullish:
            continue
        if direction < 0 and not config.include_bearish:
            continue

        # Special handling for 2-2 Down (2U-2D)
        if pattern_type == '2-2' and direction < 0 and not config.include_22_down:
            continue

        # Entry trigger uses SETUP bar (i-1), not trigger bar (i)
        # Per STRAT methodology: Entry happens LIVE when bar breaks setup bar's high/low
        entry = high[i-1] if direction > 0 else low[i-1]

        # Store setup bar levels
        setup_bar_high = high[i-1] if i > 0 else high[i]
        setup_bar_low = low[i-1] if i > 0 else low[i]

        stop = stops[i]
        target = targets[i]

        # Get full bar sequence for pattern classification
        bar_sequence = _get_full_bar_sequence(
            pattern_type, classifications, i, direction
        )

        # EQUITY-52: Calculate ATR for 3-2 patterns (ATR-based targets)
        atr_at_detection = 0.0
        if pattern_type == '3-2':
            atr_at_detection = _calculate_atr_from_df(df, i, period=14)
            if atr_at_detection > 0:
                # Use ATR-based target: entry +/- (ATR * 1.5)
                atr_target = calculate_atr_target(entry, direction, atr_at_detection, 1.5)
                if atr_target > 0:
                    target = atr_target

        # Validate target geometry and apply measured move fallback (non-3-2 patterns)
        # 3-2 patterns already have ATR-based targets from above
        if pattern_type != '3-2' and not np.isnan(target) and not np.isnan(stop) and stop > 0:
            if direction > 0:  # Bullish - target must be ABOVE entry
                if target <= entry:
                    # Target geometrically invalid - use measured move (1.5 R:R)
                    risk = entry - stop
                    if risk > 0:
                        target = entry + (risk * 1.5)
            elif direction < 0:  # Bearish - target must be BELOW entry
                if target >= entry:
                    # Target geometrically invalid - use measured move (1.5 R:R)
                    risk = stop - entry
                    if risk > 0:
                        target = entry - (risk * 1.5)

        # Skip if invalid prices
        if entry <= 0 or np.isnan(target) or target <= 0:
            continue
        if np.isnan(stop) or stop <= 0:
            continue

        # Calculate magnitude and R:R
        magnitude = abs(target - entry) / entry * 100
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0

        # Apply quality filters
        if rr < config.min_risk_reward:
            continue
        if magnitude > config.max_magnitude_pct:
            continue

        # Get setup bar timestamp
        setup_bar_timestamp = df.index[i-1] if i > 0 else df.index[i]

        patterns.append({
            'index': i,
            'timestamp': df.index[i],
            'base_pattern': pattern_type,
            'pattern_type': bar_sequence,  # Full bar sequence (e.g., '2D-2U')
            'direction': direction,
            'direction_str': direction_str,
            'entry_price': entry,
            'stop_price': stop,
            'target_price': target,
            'magnitude_pct': magnitude,
            'risk_reward': rr,
            'setup_bar_high': setup_bar_high,
            'setup_bar_low': setup_bar_low,
            'setup_bar_timestamp': setup_bar_timestamp,
            'signal_type': 'COMPLETED',
            'atr_at_detection': atr_at_detection,  # EQUITY-52: ATR for 3-2 patterns
        })

    return patterns


def detect_all_patterns(
    data: pd.DataFrame,
    config: Optional[PatternDetectionConfig] = None,
    timeframe: str = '1D'
) -> List[Dict]:
    """
    Detect ALL STRAT patterns with chronological ordering.

    This is the SINGLE SOURCE OF TRUTH for pattern detection.
    Used by both paper trading and backtesting.

    Args:
        data: OHLC DataFrame with columns: Open, High, Low, Close (case-insensitive)
              Index must be DatetimeIndex (timezone-aware preferred)
        config: Configuration options. Defaults to detecting all patterns.
        timeframe: Timeframe string for metadata ('1H', '1D', '1W', '1M')

    Returns:
        List of pattern dicts, sorted chronologically by timestamp.
        Returns empty list if no patterns found.

        Each dict contains:
            - index: Bar index in source data
            - timestamp: Pattern trigger timestamp (DatetimeIndex value)
            - base_pattern: Base pattern type ('2-2', '3-1-2', etc.)
            - pattern_type: Full bar sequence ('2D-2U', '3-1-2U', etc.)
            - direction: 1 (bullish) or -1 (bearish)
            - direction_str: 'CALL' or 'PUT'
            - entry_price: Entry trigger price
            - stop_price: Stop loss price
            - target_price: Target price
            - magnitude_pct: Target distance as % of entry
            - risk_reward: Reward / Risk ratio
            - setup_bar_high: Setup bar high price
            - setup_bar_low: Setup bar low price
            - setup_bar_timestamp: Setup bar close time
            - signal_type: 'COMPLETED' (pattern has triggered)

    Examples:
        >>> from strat.unified_pattern_detector import detect_all_patterns
        >>>
        >>> # Basic usage - all patterns
        >>> patterns = detect_all_patterns(ohlc_data)
        >>>
        >>> # Filter to specific patterns
        >>> config = PatternDetectionConfig(
        ...     include_22=True,
        ...     include_312=True,
        ...     include_212=False,
        ...     include_32=False,
        ...     include_322=False,
        ... )
        >>> patterns = detect_all_patterns(ohlc_data, config=config)

    Notes:
        CRITICAL: Patterns are sorted chronologically by timestamp, NOT grouped by type.
        This ensures consistent behavior between paper trading and backtesting.
    """
    if config is None:
        config = PatternDetectionConfig()

    # Normalize column names to handle case variations
    df = data.copy()
    df.columns = [col.capitalize() for col in df.columns]

    # Validate required columns
    required_cols = ['High', 'Low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Need at least 5 bars for pattern detection
    if len(df) < 5:
        return []

    # Get numpy arrays for numba functions
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)

    # Classify bars (single pass)
    classifications = classify_bars_nb(high, low)

    # Collect patterns from all enabled pattern types
    all_patterns: List[Dict] = []

    for pattern_type in ALL_PATTERN_TYPES:
        # Check if pattern type is enabled
        if pattern_type == '2-2' and not config.include_22:
            continue
        if pattern_type == '3-2' and not config.include_32:
            continue
        if pattern_type == '3-2-2' and not config.include_322:
            continue
        if pattern_type == '2-1-2' and not config.include_212:
            continue
        if pattern_type == '3-1-2' and not config.include_312:
            continue

        # Detect patterns for this type
        patterns = _detect_single_pattern_type(
            df, pattern_type, classifications, high, low, config
        )

        # EQUITY-39/52: Apply timeframe-specific target adjustments
        # This ensures paper trading and backtesting use identical target methodology
        # EQUITY-52: 3-2 patterns bypass 1H override (use ATR-based targets)
        for pattern in patterns:
            pattern['target_price'] = apply_timeframe_adjustment(
                target=pattern['target_price'],
                entry=pattern['entry_price'],
                stop=pattern['stop_price'],
                direction=pattern['direction'],
                timeframe=timeframe,
                pattern_type=pattern['pattern_type']  # EQUITY-52: For 3-2 bypass
            )
            # Recalculate magnitude and R:R after adjustment
            if pattern['entry_price'] > 0:
                pattern['magnitude_pct'] = abs(
                    pattern['target_price'] - pattern['entry_price']
                ) / pattern['entry_price'] * 100
            risk = abs(pattern['entry_price'] - pattern['stop_price'])
            reward = abs(pattern['target_price'] - pattern['entry_price'])
            pattern['risk_reward'] = reward / risk if risk > 0 else 0

        all_patterns.extend(patterns)

    # CRITICAL: Sort chronologically by timestamp
    if config.sort_chronologically:
        all_patterns.sort(key=lambda p: p['timestamp'])

    return all_patterns


def detect_patterns_to_dataframe(
    data: pd.DataFrame,
    config: Optional[PatternDetectionConfig] = None,
    timeframe: str = '1D'
) -> pd.DataFrame:
    """
    Detect patterns and return as DataFrame (convenience wrapper).

    Args:
        data: OHLC DataFrame
        config: Configuration options
        timeframe: Timeframe string for metadata

    Returns:
        DataFrame with pattern data, indexed by timestamp.
        Returns empty DataFrame if no patterns found.
    """
    patterns = detect_all_patterns(data, config, timeframe)

    if not patterns:
        return pd.DataFrame()

    df = pd.DataFrame(patterns)
    return df.set_index('timestamp')


# Pre-configured profiles for common use cases
TIER1_CONFIG = PatternDetectionConfig(
    include_22=True,
    include_32=False,      # Excluded from original Tier 1
    include_322=False,     # Excluded from original Tier 1
    include_212=True,
    include_312=True,
    include_22_down=False,  # 2U-2D has negative expectancy
)

ALL_PATTERNS_CONFIG = PatternDetectionConfig()  # All defaults = all patterns

PAPER_TRADING_CONFIG = PatternDetectionConfig(
    include_22_down=True,  # Include for data collection
)
