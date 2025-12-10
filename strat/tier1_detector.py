"""
Tier 1 Pattern Detector for ATLAS Trading System.

Session 70: Implements validated Tier 1 STRAT patterns for options trading.
Session 83K-8: CRITICAL FIX - Removed look-ahead bias entry filter.

CRITICAL CORRECTION (Session 83K-7):
- Continuation bars are EXIT LOGIC, not ENTRY FILTER
- Using continuation bars for entry requires seeing future bars (look-ahead bias)
- All patterns are now returned; continuation_bars is tracked for analytics only

TIER 1 PATTERNS:
1. 2-1-2 Up/Down @ 1W - Best balance (80.7% win, 563.6% exp, 57 patterns)
2. 2-2 Up (2D-2U only) @ 1W - High frequency (86.2% win, 409.5% exp, 123 patterns)
3. 3-1-2 Up/Down @ 1W - Highest quality (72.7% win, 462.7% exp, 11 patterns)
4. 2-1-2 Up @ 1M - Moonshot (88.2% win, 1,570.9% exp, 17 patterns)

WARNING: Do NOT include 2-2 Down (2U-2D) without proper exit management.

Usage:
    from strat.tier1_detector import Tier1Detector

    detector = Tier1Detector()
    signals = detector.detect_weekly_patterns(data)
    # All patterns returned; continuation_bars field available for analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from strat.pattern_detector import (
    detect_312_patterns_nb,
    detect_212_patterns_nb,
    detect_22_patterns_nb,
)
from strat.bar_classifier import classify_bars_nb


class PatternType(str, Enum):
    """
    Tier 1 pattern types with full bar sequence notation.

    STRAT Bar Classification Rule: Every directional bar MUST be classified as 2U or 2D.
    Pattern naming uses full bar sequences (e.g., "2U-1-2U" not "2-1-2U").

    Session 83K-52: Consolidated to single source of truth. Removed duplicates from
    paper_trading.py and pattern_metrics.py.
    """
    # 3-1-2 patterns (outside bar is neutral, only exit bar direction matters)
    PATTERN_312_UP = "3-1-2U"
    PATTERN_312_DOWN = "3-1-2D"

    # 2-1-2 patterns - all 4 variants with full bar sequence
    # Session 83K-44: Added full bar sequences for proper classification
    PATTERN_212_2U12U = "2U-1-2U"  # Bullish continuation
    PATTERN_212_2D12D = "2D-1-2D"  # Bearish continuation
    PATTERN_212_2D12U = "2D-1-2U"  # Bullish reversal (failed breakdown)
    PATTERN_212_2U12D = "2U-1-2D"  # Bearish reversal (failed breakout)
    # Legacy aliases for backward compatibility
    PATTERN_212_UP = "2-1-2U"      # DEPRECATED: Use specific variants above
    PATTERN_212_DOWN = "2-1-2D"    # DEPRECATED: Use specific variants above

    # 2-2 reversal patterns (already have full bar sequence)
    PATTERN_22_UP = "2D-2U"  # Bullish reversal (2D-2U) - SAFE
    PATTERN_22_DOWN = "2U-2D"  # Bearish reversal (2U-2D) - DANGEROUS without filters

    # 3-2 patterns (outside bar followed by directional)
    PATTERN_32_UP = "3-2U"
    PATTERN_32_DOWN = "3-2D"

    # 3-2-2 patterns - all 4 variants with full bar sequence
    # Session 83K-44: Added full bar sequences for proper classification
    PATTERN_322_32U2U = "3-2U-2U"  # Outside bar, bullish continuation
    PATTERN_322_32D2D = "3-2D-2D"  # Outside bar, bearish continuation
    PATTERN_322_32D2U = "3-2D-2U"  # Outside bar, bullish reversal
    PATTERN_322_32U2D = "3-2U-2D"  # Outside bar, bearish reversal
    # Legacy aliases for backward compatibility
    PATTERN_322_UP = "3-2-2U"      # DEPRECATED: Use specific variants above
    PATTERN_322_DOWN = "3-2-2D"    # DEPRECATED: Use specific variants above

    # Aliases for pattern_metrics.py backward compatibility (Session 83K-52)
    # These match the old naming convention used in pattern_metrics tests
    PATTERN_312U = "3-1-2U"        # Alias for PATTERN_312_UP
    PATTERN_312D = "3-1-2D"        # Alias for PATTERN_312_DOWN
    PATTERN_212U = "2-1-2U"        # Alias for PATTERN_212_UP (legacy simple notation)
    PATTERN_212D = "2-1-2D"        # Alias for PATTERN_212_DOWN (legacy simple notation)
    PATTERN_2D2U = "2D-2U"         # Alias for PATTERN_22_UP
    PATTERN_2U2D = "2U-2D"         # Alias for PATTERN_22_DOWN
    PATTERN_32U = "3-2U"           # Alias for PATTERN_32_UP
    PATTERN_32D = "3-2D"           # Alias for PATTERN_32_DOWN
    PATTERN_32D2U = "3-2D-2U"      # Alias for PATTERN_322_32D2U
    PATTERN_32U2D = "3-2U-2D"      # Alias for PATTERN_322_32U2D

    # Unknown/fallback for parsing
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, pattern_str: str) -> 'PatternType':
        """
        Convert string to PatternType enum with legacy mapping support.

        Handles various string formats including full bar sequences and legacy notation.

        Args:
            pattern_str: Pattern string (e.g., '2U-1-2U', '2-1-2U', '2D-2U')

        Returns:
            Matching PatternType enum member, or UNKNOWN if not found
        """
        # Normalize string (uppercase, strip)
        pattern_str = pattern_str.strip().upper()

        # Map common variations to canonical enum members
        pattern_map = {
            # Full bar sequences (preferred - CLAUDE.md Section 12)
            '2U-1-2U': cls.PATTERN_212_2U12U,
            '2D-1-2D': cls.PATTERN_212_2D12D,
            '2D-1-2U': cls.PATTERN_212_2D12U,
            '2U-1-2D': cls.PATTERN_212_2U12D,
            '3-2U-2U': cls.PATTERN_322_32U2U,
            '3-2D-2D': cls.PATTERN_322_32D2D,
            '3-2D-2U': cls.PATTERN_322_32D2U,
            '3-2U-2D': cls.PATTERN_322_32U2D,
            # 2-2 patterns
            '2D-2U': cls.PATTERN_22_UP,
            '2U-2D': cls.PATTERN_22_DOWN,
            # 3-1-2 patterns
            '3-1-2U': cls.PATTERN_312_UP,
            '3-1-2D': cls.PATTERN_312_DOWN,
            # 3-2 patterns
            '3-2U': cls.PATTERN_32_UP,
            '3-2D': cls.PATTERN_32_DOWN,
            # Legacy mappings (backward compatibility)
            '2-1-2U': cls.PATTERN_212_UP,
            '2-1-2D': cls.PATTERN_212_DOWN,
            '212U': cls.PATTERN_212_UP,
            '212D': cls.PATTERN_212_DOWN,
            '312U': cls.PATTERN_312_UP,
            '312D': cls.PATTERN_312_DOWN,
            '3-2-2U': cls.PATTERN_322_UP,
            '3-2-2D': cls.PATTERN_322_DOWN,
            # Paper trading legacy (incorrect but mapped for compat)
            '2-2U': cls.PATTERN_22_UP,
            '2-2D': cls.PATTERN_22_DOWN,
        }

        return pattern_map.get(pattern_str, cls.UNKNOWN)

    def is_bullish(self) -> bool:
        """Check if pattern is bullish (ends with U)."""
        return self.value.endswith('U')

    def is_bearish(self) -> bool:
        """Check if pattern is bearish (ends with D)."""
        return self.value.endswith('D')

    def base_pattern(self) -> str:
        """
        Get base pattern type without direction (e.g., '3-1-2').

        Returns:
            Base pattern string: '3-1-2', '2-1-2', '3-2-2', '3-2', '2-2', or 'UNKNOWN'
        """
        val = self.value
        if '3-1-2' in val:
            return '3-1-2'
        elif '2U-1-2' in val or '2D-1-2' in val or '2-1-2' in val:
            return '2-1-2'
        elif '3-2U-2' in val or '3-2D-2' in val or '3-2-2' in val:
            return '3-2-2'
        elif '3-2' in val:
            return '3-2'
        elif '2D-2U' in val or '2U-2D' in val:
            return '2-2'
        return 'UNKNOWN'


class Timeframe(Enum):
    """Supported timeframes for Tier 1 patterns."""
    HOURLY = "1H"   # Session 83K-31: Added for intraday patterns
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"


@dataclass
class PatternSignal:
    """
    Represents a detected Tier 1 pattern signal.

    Attributes:
        pattern_type: Type of pattern (312, 212, 22)
        direction: 1 for bullish, -1 for bearish
        entry_price: Entry trigger price
        stop_price: Stop loss price
        target_price: Target price (measured move or magnitude)
        timestamp: Pattern detection timestamp
        timeframe: Pattern timeframe (weekly, monthly)
        continuation_bars: Number of continuation bars detected
        is_filtered: Whether pattern passes continuation bar filter
        risk_reward: Calculated risk/reward ratio
    """
    pattern_type: PatternType
    direction: int
    entry_price: float
    stop_price: float
    target_price: float
    timestamp: pd.Timestamp
    timeframe: Timeframe
    continuation_bars: int = 0
    is_filtered: bool = False
    risk_reward: float = 0.0

    def __post_init__(self):
        """Calculate risk/reward after initialization."""
        if self.stop_price and self.entry_price:
            risk = abs(self.entry_price - self.stop_price)
            reward = abs(self.target_price - self.entry_price)
            if risk > 0:
                self.risk_reward = reward / risk


class Tier1Detector:
    """
    Tier 1 Pattern Detector for STRAT patterns.

    Session 83K-8: Removed look-ahead bias entry filtering.
    Continuation bars are counted for analytics but ALL patterns are returned.
    Continuation bar logic is now EXIT management, not ENTRY filtering.

    IMPORTANT: This detector outputs ALL detected patterns:
    - 2-2 Up patterns only by default (excludes 2-2 Down unless enabled)
    - Weekly or Monthly timeframes
    - continuation_bars field populated for analytics/DTE selection

    Attributes:
        min_continuation_bars: Kept for backward compatibility (no longer filters)
        include_22_down: Whether to include 2-2 Down patterns (default: False)
    """

    def __init__(
        self,
        min_continuation_bars: int = 2,
        include_22_down: bool = False
    ):
        """
        Initialize Tier 1 Detector.

        Args:
            min_continuation_bars: Kept for backward compatibility and analytics.
                                   No longer used for entry filtering (Session 83K-8).
            include_22_down: Include 2U-2D patterns? (default: False - dangerous!)
        """
        # Session 83K-8: min_continuation_bars no longer used for entry filtering
        # Kept for backward compatibility and analytics (is_filtered flag)
        # Continuation bars are EXIT logic, not ENTRY filter (look-ahead bias fix)
        self.min_continuation_bars = min_continuation_bars
        self.include_22_down = include_22_down

        if include_22_down:
            import warnings
            warnings.warn(
                "WARNING: Including 2-2 Down (2U-2D) patterns. "
                "Session 69 found these have NEGATIVE expectancy without filters. "
                "Ensure continuation bar filter is applied.",
                UserWarning
            )

    def detect_patterns(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe = Timeframe.WEEKLY,
        pattern_types: Optional[List[str]] = None
    ) -> List[PatternSignal]:
        """
        Detect all Tier 1 patterns in OHLC data.

        Args:
            data: DataFrame with Open, High, Low, Close columns (or lowercase)
            timeframe: Pattern timeframe (default: WEEKLY)
            pattern_types: Optional list of patterns to detect
                         (default: all Tier 1 patterns)

        Returns:
            List of PatternSignal objects (all patterns; continuation_bars for analytics)

        Raises:
            ValueError: If data doesn't have required columns
        """
        # Normalize column names
        data = self._normalize_columns(data)

        # Classify bars
        classifications = classify_bars_nb(
            data['high'].values,
            data['low'].values
        )

        all_signals = []

        # Detect each pattern type
        if pattern_types is None or '312' in pattern_types:
            signals_312 = self._detect_312(data, classifications, timeframe)
            all_signals.extend(signals_312)

        if pattern_types is None or '212' in pattern_types:
            signals_212 = self._detect_212(data, classifications, timeframe)
            all_signals.extend(signals_212)

        if pattern_types is None or '22' in pattern_types:
            signals_22 = self._detect_22(data, classifications, timeframe)
            all_signals.extend(signals_22)

        # Apply continuation bar filter
        filtered_signals = self._apply_continuation_filter(
            all_signals,
            data,
            classifications
        )

        return filtered_signals

    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        data = data.copy()
        data.columns = [col.lower() for col in data.columns]

        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return data

    def _detect_312(
        self,
        data: pd.DataFrame,
        classifications: np.ndarray,
        timeframe: Timeframe
    ) -> List[PatternSignal]:
        """Detect 3-1-2 patterns."""
        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications,
            data['high'].values,
            data['low'].values
        )

        signals = []
        for i in range(len(entries)):
            if entries[i]:
                direction = int(directions[i])
                pattern_type = (
                    PatternType.PATTERN_312_UP if direction == 1
                    else PatternType.PATTERN_312_DOWN
                )

                signal = PatternSignal(
                    pattern_type=pattern_type,
                    direction=direction,
                    entry_price=float(data['high'].iloc[i-1] if direction == 1
                                     else data['low'].iloc[i-1]),
                    stop_price=float(stops[i]),
                    target_price=float(targets[i]),
                    timestamp=data.index[i],
                    timeframe=timeframe,
                )
                signals.append(signal)

        return signals

    def _detect_212(
        self,
        data: pd.DataFrame,
        classifications: np.ndarray,
        timeframe: Timeframe
    ) -> List[PatternSignal]:
        """
        Detect 2-1-2 patterns with full bar sequence classification.

        Session 83K-44: Now uses full bar sequences (2U-1-2U, 2D-1-2D, etc.)
        instead of simplified notation (2-1-2U, 2-1-2D).
        """
        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications,
            data['high'].values,
            data['low'].values
        )

        signals = []
        for i in range(len(entries)):
            if entries[i]:
                direction = int(directions[i])

                # Session 83K-44: Determine full bar sequence from classifications
                # Bar at i-2 is first directional, bar at i is trigger directional
                bar1_class = int(classifications[i-2])  # First directional bar
                bar3_class = int(classifications[i])    # Trigger bar

                # Determine pattern type based on actual bar sequence
                if bar1_class == 2 and bar3_class == 2:
                    # 2U-1-2U: Bullish continuation
                    pattern_type = PatternType.PATTERN_212_2U12U
                elif bar1_class == -2 and bar3_class == -2:
                    # 2D-1-2D: Bearish continuation
                    pattern_type = PatternType.PATTERN_212_2D12D
                elif bar1_class == -2 and bar3_class == 2:
                    # 2D-1-2U: Bullish reversal (failed breakdown)
                    pattern_type = PatternType.PATTERN_212_2D12U
                elif bar1_class == 2 and bar3_class == -2:
                    # 2U-1-2D: Bearish reversal (failed breakout)
                    pattern_type = PatternType.PATTERN_212_2U12D
                else:
                    # Fallback to legacy types (should not happen)
                    pattern_type = (
                        PatternType.PATTERN_212_UP if direction == 1
                        else PatternType.PATTERN_212_DOWN
                    )

                signal = PatternSignal(
                    pattern_type=pattern_type,
                    direction=direction,
                    entry_price=float(data['high'].iloc[i-1] if direction == 1
                                     else data['low'].iloc[i-1]),
                    stop_price=float(stops[i]),
                    target_price=float(targets[i]),
                    timestamp=data.index[i],
                    timeframe=timeframe,
                )
                signals.append(signal)

        return signals

    def _detect_22(
        self,
        data: pd.DataFrame,
        classifications: np.ndarray,
        timeframe: Timeframe
    ) -> List[PatternSignal]:
        """
        Detect 2-2 patterns.

        CRITICAL: Excludes 2-2 Down (2U-2D) by default per Session 69 findings.
        """
        entries, stops, targets, directions = detect_22_patterns_nb(
            classifications,
            data['high'].values,
            data['low'].values
        )

        signals = []
        for i in range(len(entries)):
            if entries[i]:
                direction = int(directions[i])

                # Determine pattern type
                if direction == 1:
                    pattern_type = PatternType.PATTERN_22_UP
                else:
                    pattern_type = PatternType.PATTERN_22_DOWN

                    # CRITICAL: Skip 2-2 Down unless explicitly enabled
                    if not self.include_22_down:
                        continue

                signal = PatternSignal(
                    pattern_type=pattern_type,
                    direction=direction,
                    entry_price=float(data['high'].iloc[i-1] if direction == 1
                                     else data['low'].iloc[i-1]),
                    stop_price=float(stops[i]),
                    target_price=float(targets[i]),
                    timestamp=data.index[i],
                    timeframe=timeframe,
                )
                signals.append(signal)

        return signals

    def _apply_continuation_filter(
        self,
        signals: List[PatternSignal],
        data: pd.DataFrame,
        classifications: np.ndarray
    ) -> List[PatternSignal]:
        """
        Count continuation bars for signals (analytics only - no filtering).

        Session 83K-8: Continuation bars are now EXIT logic, not ENTRY filter.
        All signals are returned. The continuation_bars field is populated
        for analytics/DTE selection, but no signals are rejected.

        Continuation bars are directional bars (2 or -2) following the pattern
        in the same direction as the trade.

        Args:
            signals: Raw pattern signals
            data: OHLC data
            classifications: Bar classifications

        Returns:
            All signals with continuation bar counts populated
        """
        result = []

        for signal in signals:
            # Find pattern index in data
            try:
                idx = data.index.get_loc(signal.timestamp)
            except KeyError:
                continue

            # Count continuation bars after pattern
            continuation_count = 0
            lookforward = min(5, len(data) - idx - 1)  # Look up to 5 bars forward

            for j in range(1, lookforward + 1):
                bar_class = classifications[idx + j] if idx + j < len(classifications) else 0

                # Count directional bars in pattern direction
                if signal.direction == 1 and bar_class == 2:
                    continuation_count += 1
                elif signal.direction == -1 and bar_class == -2:
                    continuation_count += 1
                # Break on reversal bars (opposite direction)
                elif signal.direction == 1 and bar_class == -2:
                    break  # Reversal (2D) for bullish pattern
                elif signal.direction == -1 and bar_class == 2:
                    break  # Reversal (2U) for bearish pattern
                # Break on outside bars (exhaustion signal)
                elif bar_class == 3:
                    break
                # Inside bars (1) - continue without counting or breaking

            # Update signal with continuation count
            signal.continuation_bars = continuation_count
            # is_filtered indicates whether pattern meets analytics threshold
            # (kept for backward compatibility but no longer used for filtering)
            signal.is_filtered = continuation_count >= self.min_continuation_bars

            # Session 83K-8: Return ALL signals (no filtering)
            # Continuation bar count kept for analytics/DTE selection
            result.append(signal)

        return result

    def get_tier1_weekly(self, data: pd.DataFrame) -> List[PatternSignal]:
        """
        Convenience method for weekly Tier 1 patterns.

        Patterns included:
        - 2-1-2 Up/Down
        - 2-2 Up (2D-2U only)
        - 3-1-2 Up/Down
        """
        return self.detect_patterns(data, timeframe=Timeframe.WEEKLY)

    def get_tier1_monthly(self, data: pd.DataFrame) -> List[PatternSignal]:
        """
        Convenience method for monthly Tier 1 patterns.

        Patterns included:
        - 2-1-2 Up (moonshot pattern)
        """
        return self.detect_patterns(
            data,
            timeframe=Timeframe.MONTHLY,
            pattern_types=['212']
        )

    def signals_to_dataframe(
        self,
        signals: List[PatternSignal]
    ) -> pd.DataFrame:
        """
        Convert signals to DataFrame for analysis.

        Args:
            signals: List of PatternSignal objects

        Returns:
            DataFrame with signal details
        """
        if not signals:
            return pd.DataFrame()

        records = []
        for sig in signals:
            records.append({
                'timestamp': sig.timestamp,
                'pattern_type': sig.pattern_type.value,
                'direction': sig.direction,
                'entry_price': sig.entry_price,
                'stop_price': sig.stop_price,
                'target_price': sig.target_price,
                'risk_reward': sig.risk_reward,
                'continuation_bars': sig.continuation_bars,
                'is_filtered': sig.is_filtered,
                'timeframe': sig.timeframe.value,
            })

        return pd.DataFrame(records)

    def generate_summary(self, signals: List[PatternSignal]) -> Dict:
        """
        Generate summary statistics for detected signals.

        Args:
            signals: List of PatternSignal objects

        Returns:
            Dictionary with summary statistics
        """
        if not signals:
            return {'total_signals': 0}

        df = self.signals_to_dataframe(signals)

        summary = {
            'total_signals': len(signals),
            'by_pattern_type': df['pattern_type'].value_counts().to_dict(),
            'by_direction': {
                'bullish': len(df[df['direction'] == 1]),
                'bearish': len(df[df['direction'] == -1]),
            },
            'avg_risk_reward': df['risk_reward'].mean(),
            'avg_continuation_bars': df['continuation_bars'].mean(),
            'filtered_count': len(df[df['is_filtered']]),
            'filter_rate': len(df[df['is_filtered']]) / len(df) if len(df) > 0 else 0,
        }

        return summary


# Convenience function for quick detection
def detect_tier1_patterns(
    data: pd.DataFrame,
    timeframe: str = '1W',
    min_continuation_bars: int = 2
) -> pd.DataFrame:
    """
    Quick function to detect Tier 1 patterns.

    Args:
        data: OHLC DataFrame
        timeframe: '1W' for weekly, '1M' for monthly
        min_continuation_bars: Minimum continuation bars (default: 2)

    Returns:
        DataFrame with detected signals
    """
    tf = Timeframe.WEEKLY if timeframe == '1W' else Timeframe.MONTHLY
    detector = Tier1Detector(min_continuation_bars=min_continuation_bars)
    signals = detector.detect_patterns(data, timeframe=tf)
    return detector.signals_to_dataframe(signals)


if __name__ == "__main__":
    # Test the detector
    print("=" * 60)
    print("Tier 1 Pattern Detector Test")
    print("=" * 60)

    # Create sample data
    import numpy as np

    dates = pd.date_range('2024-01-01', periods=100, freq='W')
    np.random.seed(42)

    # Generate realistic OHLC data
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.random.rand(100) * 3
    low = close - np.random.rand(100) * 3
    open_ = close + np.random.randn(100)

    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    }, index=dates)

    # Initialize detector
    detector = Tier1Detector(min_continuation_bars=2)

    # Detect patterns
    signals = detector.detect_patterns(data, timeframe=Timeframe.WEEKLY)

    print(f"\nDetected {len(signals)} Tier 1 patterns")

    if signals:
        df = detector.signals_to_dataframe(signals)
        print(f"\nSignals DataFrame:")
        print(df.head(10))

        summary = detector.generate_summary(signals)
        print(f"\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
