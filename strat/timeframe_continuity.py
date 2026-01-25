"""
STRAT Timeframe Continuity Checker

Analyzes multi-timeframe alignment of STRAT bar classifications to identify
high-confidence trading signals.

Timeframe Continuity Concept:
    Full continuity (5/5 aligned): Highest conviction signals
    - Monthly, Weekly, Daily, 4H, and 1H all show directional bars in same direction
    - Example: All timeframes show 2U (bullish) alignment

    Partial continuity (3-4/5): Medium conviction
    - Majority of timeframes aligned

    No continuity (<3/5): Low conviction
    - Mixed or conflicting signals across timeframes

Usage:
    Used to filter STRAT pattern signals (3-1-2, 2-1-2) for highest probability setups.
    Higher continuity = stronger institutional participation = higher win rate.
"""

import numpy as np
import pandas as pd
from numba import njit
import vectorbtpro as vbt
from typing import Dict, Tuple, Optional

from strat.bar_classifier import StratBarClassifier, classify_bars


class TimeframeContinuityChecker:
    """
    Check alignment of STRAT directional bars across multiple timeframes.

    Analyzes whether bullish or bearish directional bars (2U or 2D) are aligned
    across Monthly, Weekly, Daily, 4H, and 1H timeframes.

    Full continuity indicates strong institutional participation and higher
    probability of pattern success.

    Attributes:
    -----------
    timeframes : list
        Timeframes to analyze, from longest to shortest
        Default: ['1M', '1W', '1D', '4H', '1H']

    Examples:
    ---------
    >>> checker = TimeframeContinuityChecker()
    >>> data = vbt.YFData.pull('SPY', start='2024-01-01', timeframe='1H')
    >>> continuity = checker.check_continuity(data, direction='bullish')
    >>> print(f"Continuity strength: {continuity['strength']}/5")
    >>> print(f"Full continuity: {continuity['full_continuity']}")
    """

    def __init__(self, timeframes: Optional[list] = None):
        """
        Initialize timeframe continuity checker.

        Parameters:
        -----------
        timeframes : list, optional
            List of timeframes to check, from longest to shortest
            Default: ['1M', '1W', '1D', '4H', '1H']
        """
        if timeframes is None:
            timeframes = ['1M', '1W', '1D', '4H', '1H']

        self.timeframes = timeframes
        self.n_timeframes = len(timeframes)

    def resample_to_timeframe(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLC data to target timeframe.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLC data with 'High', 'Low', 'Open', 'Close' columns
            Must have datetime index
        timeframe : str
            Target timeframe (e.g., '1D', '4H', '1W', '1M', '2D')
            Session 65: Added '2D' hybrid timeframe support (STRAT Lab optimization)

        Returns:
        --------
        pd.DataFrame
            Resampled OHLC data
        """
        # Session 65: 2D hybrid timeframe optimization (STRAT Lab research)
        # Research shows 2D charts have +8.3 percentage point higher transition
        # probabilities than standard 1D charts (48.6% vs 40.3% for Hammer→2u)
        if timeframe == '2D':
            # 2-day hybrid bars: Resample to 2-day periods
            # STRAT Lab: "2D chart signals are less likely to be random noise"
            resampled = data.resample('2D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
        else:
            # Standard timeframes (1H, 4H, 1D, 1W, 1M)
            resampled = data.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()

        return resampled

    def check_directional_bar(
        self,
        classification: float,
        direction: str,
        open_price: Optional[float] = None,
        close_price: Optional[float] = None
    ) -> bool:
        """
        Check if bar is directional in specified direction.

        Per STRAT methodology (EQUITY-44):
        - Type 1 (inside): Does NOT count toward TFC
        - Type 2U/2D: Counts as directional (bullish/bearish)
        - Type 3 (outside): Counts, direction by candle color (green=bullish, red=bearish)

        Parameters:
        -----------
        classification : float
            Bar classification from classify_bars_nb()
            2.0 = 2U (bullish), -2.0 = 2D (bearish), 3.0 = Type 3 (outside)
        direction : str
            'bullish' or 'bearish'
        open_price : float, optional
            Bar open price (needed for Type 3 candle color determination)
        close_price : float, optional
            Bar close price (needed for Type 3 candle color determination)

        Returns:
        --------
        bool
            True if bar is directional in specified direction
        """
        if np.isnan(classification) or classification == -999:
            return False

        # Type 2 bars - direct direction match
        if direction == 'bullish':
            if classification == 2.0:  # 2U bar
                return True
        elif direction == 'bearish':
            if classification == -2.0:  # 2D bar
                return True

        # Type 3 (outside bar) - direction by candle color (EQUITY-44)
        # Per STRAT methodology: Green = bullish, Red = bearish
        if classification == 3.0:
            if open_price is not None and close_price is not None:
                is_green = close_price > open_price
                if direction == 'bullish' and is_green:
                    return True
                elif direction == 'bearish' and not is_green:
                    return True
            # If no price data, Type 3 doesn't count (conservative approach)

        return False

    def check_continuity(
        self,
        high_dict: Dict[str, pd.Series],
        low_dict: Dict[str, pd.Series],
        direction: str = 'bullish',
        bar_index: int = -1,
        open_dict: Optional[Dict[str, pd.Series]] = None,
        close_dict: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, any]:
        """
        Check timeframe continuity at specific bar index.

        Parameters:
        -----------
        high_dict : dict
            Dictionary mapping timeframes to high price Series
            Example: {'1M': monthly_high, '1W': weekly_high, ...}
        low_dict : dict
            Dictionary mapping timeframes to low price Series
            Example: {'1M': monthly_low, '1W': weekly_low, ...}
        direction : str, default 'bullish'
            Direction to check: 'bullish' or 'bearish'
        bar_index : int, default -1
            Bar index to check (-1 = most recent bar)
        open_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to open price Series
            Required for Type 3 candle color determination
        close_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to close price Series
            Required for Type 3 candle color determination

        Returns:
        --------
        dict
            {
                'strength': int (0-5, number of aligned timeframes),
                'full_continuity': bool (True if all 5 timeframes aligned),
                'aligned_timeframes': list of str (which timeframes are aligned),
                'direction': str ('bullish' or 'bearish')
            }

        Examples:
        ---------
        >>> # Full continuity example (all bullish)
        >>> result = checker.check_continuity(high_dict, low_dict, 'bullish')
        >>> result['strength']
        5
        >>> result['full_continuity']
        True

        >>> # Partial continuity (3/5 aligned)
        >>> result = checker.check_continuity(high_dict, low_dict, 'bearish')
        >>> result['strength']
        3
        >>> result['aligned_timeframes']
        ['1M', '1W', '1D']
        """
        aligned_timeframes = []

        for tf in self.timeframes:
            if tf not in high_dict or tf not in low_dict:
                continue

            high_series = high_dict[tf]
            low_series = low_dict[tf]

            # Classify bars for this timeframe
            classifications = classify_bars(high_series, low_series)

            # Get classification at specified index
            if bar_index < 0:
                # Negative index (from end)
                bar_classification = classifications.iloc[bar_index]
            else:
                # Positive index
                bar_classification = classifications.iloc[bar_index]

            # Get Open/Close for Type 3 candle color (EQUITY-44)
            open_price = None
            close_price = None
            if open_dict and close_dict and tf in open_dict and tf in close_dict:
                try:
                    open_price = open_dict[tf].iloc[bar_index]
                    close_price = close_dict[tf].iloc[bar_index]
                except (IndexError, KeyError):
                    pass  # Fall back to no color data

            # Check if directional in specified direction
            is_aligned = self.check_directional_bar(
                bar_classification, direction, open_price, close_price
            )

            if is_aligned:
                aligned_timeframes.append(tf)

        strength = len(aligned_timeframes)
        full_continuity = (strength == self.n_timeframes)

        return {
            'strength': strength,
            'full_continuity': full_continuity,
            'aligned_timeframes': aligned_timeframes,
            'direction': direction
        }

    def check_flexible_continuity_at_datetime(
        self,
        high_dict: Dict[str, pd.Series],
        low_dict: Dict[str, pd.Series],
        target_datetime: pd.Timestamp,
        direction: str = 'bullish',
        min_strength: int = 3,
        detection_timeframe: str = '1D',
        open_dict: Optional[Dict[str, pd.Series]] = None,
        close_dict: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, any]:
        """
        Check flexible timeframe continuity at specific datetime.

        This method combines check_continuity_at_datetime() with flexible continuity rules.
        Used by validation script to check timeframe-appropriate continuity at pattern datetime.

        Parameters:
        -----------
        high_dict : dict
            Dictionary mapping timeframes to high price Series with datetime index
        low_dict : dict
            Dictionary mapping timeframes to low price Series with datetime index
        target_datetime : pd.Timestamp
            Datetime to check continuity at
        direction : str, default 'bullish'
            Direction to check: 'bullish' or 'bearish'
        min_strength : int, default 3
            Minimum number of aligned timeframes required (will be adjusted for timeframe)
        detection_timeframe : str, default '1D'
            Timeframe where pattern was detected (determines which TFs to check)
        open_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to open price Series
            Required for Type 3 candle color determination
        close_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to close price Series
            Required for Type 3 candle color determination

        Returns:
        --------
        dict
            Same format as check_flexible_continuity()
        """
        # Define timeframe-appropriate continuity requirements
        # Session EQUITY-63: Include 1M for 1H - if all aligned, that's FTFC (no conflict)
        timeframe_requirements = {
            '1H': ['1M', '1W', '1D', '1H'],  # Hourly: All 4 TFs for Full TFC possibility
            '4H': ['1W', '1D', '4H', '1H'],  # EQUITY-89: 4H for crypto (no monthly needed)
            '1D': ['1M', '1W', '1D'],        # Daily: Month, Week, Day, need 2/3
            '1W': ['1M', '1W'],              # Weekly: Just month+week, need 1/2
            '1M': ['1M']                     # Monthly: Just itself, need 1/1
        }

        # Timeframe-appropriate minimum strength
        # Session EQUITY-63: 1H now checks 4 TFs, min 3 means 3/4 to pass
        timeframe_min_strength = {
            '1H': 3,  # Need 3/4 (any 3 of Month, Week, Day, Hour aligned)
            '4H': 2,  # EQUITY-89: Need 2/4 (any 2 of Week, Day, 4H, Hour aligned)
            '1D': 2,  # Need 2/3 (any 2 of Month, Week, Day)
            '1W': 1,  # Need 1/2 (Month OR Week aligned)
            '1M': 1   # Need 1/1 (Monthly bar itself)
        }

        # Get required timeframes for this detection timeframe
        required_tfs = timeframe_requirements.get(detection_timeframe, self.timeframes)

        # Override min_strength with timeframe-appropriate value
        min_strength = timeframe_min_strength.get(detection_timeframe, min_strength)

        aligned_timeframes = []

        for tf in required_tfs:
            if tf not in high_dict or tf not in low_dict:
                continue

            high_series = high_dict[tf]
            low_series = low_dict[tf]

            # Find nearest datetime (forward-fill behavior)
            valid_index = high_series.index[high_series.index <= target_datetime]
            if len(valid_index) == 0:
                continue

            nearest_datetime = valid_index[-1]

            # Classify bars up to this datetime
            high_subset = high_series.loc[:nearest_datetime]
            low_subset = low_series.loc[:nearest_datetime]

            classifications = classify_bars(high_subset, low_subset)

            # Get Open/Close for Type 3 candle color (EQUITY-44)
            open_price = None
            close_price = None
            if open_dict and close_dict and tf in open_dict and tf in close_dict:
                try:
                    open_subset = open_dict[tf].loc[:nearest_datetime]
                    close_subset = close_dict[tf].loc[:nearest_datetime]
                    open_price = open_subset.iloc[-1]
                    close_price = close_subset.iloc[-1]
                except (IndexError, KeyError):
                    pass  # Fall back to no color data

            # Check most recent bar
            bar_classification = classifications.iloc[-1]
            is_aligned = self.check_directional_bar(
                bar_classification, direction, open_price, close_price
            )

            if is_aligned:
                aligned_timeframes.append(tf)

        strength = len(aligned_timeframes)
        passes_flexible = (strength >= min_strength)

        return {
            'strength': strength,
            'passes_flexible': passes_flexible,
            'aligned_timeframes': aligned_timeframes,
            'direction': direction,
            'required_timeframes': required_tfs
        }

    def check_flexible_continuity(
        self,
        high_dict: Dict[str, pd.Series],
        low_dict: Dict[str, pd.Series],
        direction: str = 'bullish',
        bar_index: int = -1,
        min_strength: int = 3,
        detection_timeframe: str = '1D',
        open_dict: Optional[Dict[str, pd.Series]] = None,
        close_dict: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, any]:
        """
        Check timeframe continuity with flexible alignment requirements.

        CRITICAL: Timeframe-appropriate continuity prevents false filtering.

        Problem with full continuity (5/5):
        - Requires ALL timeframes aligned (1M, 1W, 1D, 4H, 1H)
        - Checking lower timeframes for higher TF patterns creates false filtering
        - Example: Daily pattern spans 6.5 hours, individual hourly bars don't need alignment

        Flexible Continuity Rules by Detection Timeframe:
        - HOURLY (1H): Require Week+2D+Day+Hour (3/4 TFs, skip Monthly)
        - DAILY (1D): Require Month+Week+2D+Day (2/4 TFs minimum)
        - WEEKLY (1W): Require Month+Week only
        - MONTHLY (1M): Just monthly bar itself

        Parameters:
        -----------
        high_dict : dict
            Dictionary mapping timeframes to high price Series
        low_dict : dict
            Dictionary mapping timeframes to low price Series
        direction : str, default 'bullish'
            Direction to check: 'bullish' or 'bearish'
        bar_index : int, default -1
            Bar index to check (-1 = most recent bar)
        min_strength : int, default 3
            Minimum number of aligned timeframes required (3/5, 4/5, or 5/5)
        detection_timeframe : str, default '1D'
            Timeframe where pattern was detected (determines which TFs to check)
        open_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to open price Series
            Required for Type 3 candle color determination
        close_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to close price Series
            Required for Type 3 candle color determination

        Returns:
        --------
        dict
            {
                'strength': int (0-5, number of aligned timeframes),
                'passes_flexible': bool (True if strength >= min_strength),
                'aligned_timeframes': list of str (which timeframes are aligned),
                'direction': str ('bullish' or 'bearish'),
                'required_timeframes': list of str (timeframes that should be checked)
            }

        Examples:
        ---------
        >>> # Hourly pattern: Check Week+2D+Day+Hour (3 TFs required)
        >>> result = checker.check_flexible_continuity(
        ...     high_dict, low_dict, 'bullish',
        ...     min_strength=3, detection_timeframe='1H'
        ... )
        >>> # Daily pattern: Check Month+Week+2D+Day (2 TFs required)
        >>> result = checker.check_flexible_continuity(
        ...     high_dict, low_dict, 'bearish',
        ...     min_strength=3, detection_timeframe='1D'
        ... )
        """
        # Define timeframe-appropriate continuity requirements
        # Key insight from Session 55: Don't check lower TFs for higher TF patterns
        # Session EQUITY-63: Include 1M for 1H - if all aligned, that's FTFC (no conflict)
        timeframe_requirements = {
            '1H': ['1M', '1W', '1D', '1H'],  # Hourly: All 4 TFs for Full TFC possibility
            '4H': ['1W', '1D', '4H', '1H'],  # EQUITY-89: 4H for crypto (no monthly needed)
            '1D': ['1M', '1W', '1D'],        # Daily: Month, Week, Day, need 2/3
            '1W': ['1M', '1W'],              # Weekly: Just month+week, need 1/2
            '1M': ['1M']                     # Monthly: Just itself, need 1/1
        }

        # Timeframe-appropriate minimum strength
        # Session EQUITY-63: 1H now checks 4 TFs, min 3 means 3/4 to pass
        timeframe_min_strength = {
            '1H': 3,  # Need 3/4 (any 3 of Month, Week, Day, Hour aligned)
            '4H': 2,  # EQUITY-89: Need 2/4 (any 2 of Week, Day, 4H, Hour aligned)
            '1D': 2,  # Need 2/3 (any 2 of Month, Week, Day)
            '1W': 1,  # Need 1/2 (Month OR Week aligned)
            '1M': 1   # Need 1/1 (Monthly bar itself)
        }

        # Get required timeframes for this detection timeframe
        required_tfs = timeframe_requirements.get(detection_timeframe, self.timeframes)

        # Override min_strength with timeframe-appropriate value
        min_strength = timeframe_min_strength.get(detection_timeframe, min_strength)

        aligned_timeframes = []

        for tf in required_tfs:
            if tf not in high_dict or tf not in low_dict:
                continue

            high_series = high_dict[tf]
            low_series = low_dict[tf]

            # Classify bars for this timeframe
            classifications = classify_bars(high_series, low_series)

            # Get classification at specified index
            if bar_index < 0:
                bar_classification = classifications.iloc[bar_index]
            else:
                bar_classification = classifications.iloc[bar_index]

            # Get Open/Close for Type 3 candle color (EQUITY-44)
            open_price = None
            close_price = None
            if open_dict and close_dict and tf in open_dict and tf in close_dict:
                try:
                    open_price = open_dict[tf].iloc[bar_index]
                    close_price = close_dict[tf].iloc[bar_index]
                except (IndexError, KeyError):
                    pass  # Fall back to no color data

            # Check if directional in specified direction
            is_aligned = self.check_directional_bar(
                bar_classification, direction, open_price, close_price
            )

            if is_aligned:
                aligned_timeframes.append(tf)

        strength = len(aligned_timeframes)
        passes_flexible = (strength >= min_strength)

        return {
            'strength': strength,
            'passes_flexible': passes_flexible,
            'aligned_timeframes': aligned_timeframes,
            'direction': direction,
            'required_timeframes': required_tfs
        }

    def check_continuity_at_datetime(
        self,
        high_dict: Dict[str, pd.Series],
        low_dict: Dict[str, pd.Series],
        target_datetime: pd.Timestamp,
        direction: str = 'bullish',
        open_dict: Optional[Dict[str, pd.Series]] = None,
        close_dict: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, any]:
        """
        Check timeframe continuity at specific datetime.

        Parameters:
        -----------
        high_dict : dict
            Dictionary mapping timeframes to high price Series with datetime index
        low_dict : dict
            Dictionary mapping timeframes to low price Series with datetime index
        target_datetime : pd.Timestamp
            Datetime to check continuity at
        direction : str, default 'bullish'
            Direction to check: 'bullish' or 'bearish'
        open_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to open price Series
            Required for Type 3 candle color determination
        close_dict : dict, optional (EQUITY-44)
            Dictionary mapping timeframes to close price Series
            Required for Type 3 candle color determination

        Returns:
        --------
        dict
            Same format as check_continuity()

        Examples:
        ---------
        >>> target_dt = pd.Timestamp('2024-11-15 14:00:00')
        >>> result = checker.check_continuity_at_datetime(
        ...     high_dict, low_dict, target_dt, 'bullish'
        ... )
        """
        aligned_timeframes = []

        for tf in self.timeframes:
            if tf not in high_dict or tf not in low_dict:
                continue

            high_series = high_dict[tf]
            low_series = low_dict[tf]

            # Find nearest datetime (forward-fill behavior)
            # This matches VBT's timeframe parameter behavior
            valid_index = high_series.index[high_series.index <= target_datetime]
            if len(valid_index) == 0:
                continue

            nearest_datetime = valid_index[-1]

            # Classify bars up to this datetime
            high_subset = high_series.loc[:nearest_datetime]
            low_subset = low_series.loc[:nearest_datetime]

            classifications = classify_bars(high_subset, low_subset)

            # Get Open/Close for Type 3 candle color (EQUITY-44)
            open_price = None
            close_price = None
            if open_dict and close_dict and tf in open_dict and tf in close_dict:
                try:
                    open_subset = open_dict[tf].loc[:nearest_datetime]
                    close_subset = close_dict[tf].loc[:nearest_datetime]
                    open_price = open_subset.iloc[-1]
                    close_price = close_subset.iloc[-1]
                except (IndexError, KeyError):
                    pass  # Fall back to no color data

            # Check most recent bar
            bar_classification = classifications.iloc[-1]
            is_aligned = self.check_directional_bar(
                bar_classification, direction, open_price, close_price
            )

            if is_aligned:
                aligned_timeframes.append(tf)

        strength = len(aligned_timeframes)
        full_continuity = (strength == self.n_timeframes)

        return {
            'strength': strength,
            'full_continuity': full_continuity,
            'aligned_timeframes': aligned_timeframes,
            'direction': direction,
            'target_datetime': target_datetime
        }


# Convenience function for single-call continuity check
def check_full_continuity(
    high_dict: Dict[str, pd.Series],
    low_dict: Dict[str, pd.Series],
    direction: str = 'bullish',
    bar_index: int = -1,
    open_dict: Optional[Dict[str, pd.Series]] = None,
    close_dict: Optional[Dict[str, pd.Series]] = None
) -> bool:
    """
    Quick check for full timeframe continuity (all 5 timeframes aligned).

    Parameters:
    -----------
    high_dict : dict
        Dictionary mapping timeframes to high price Series
    low_dict : dict
        Dictionary mapping timeframes to low price Series
    direction : str, default 'bullish'
        Direction to check: 'bullish' or 'bearish'
    bar_index : int, default -1
        Bar index to check (-1 = most recent bar)
    open_dict : dict, optional (EQUITY-44)
        Dictionary mapping timeframes to open price Series
        Required for Type 3 candle color determination
    close_dict : dict, optional (EQUITY-44)
        Dictionary mapping timeframes to close price Series
        Required for Type 3 candle color determination

    Returns:
    --------
    bool
        True if all 5 timeframes show directional bars in same direction

    Examples:
    ---------
    >>> has_full_continuity = check_full_continuity(
    ...     high_dict, low_dict, 'bullish'
    ... )
    >>> if has_full_continuity:
    ...     print("HIGHEST CONVICTION SIGNAL - All timeframes aligned")
    """
    checker = TimeframeContinuityChecker()
    result = checker.check_continuity(
        high_dict, low_dict, direction, bar_index, open_dict, close_dict
    )
    return result['full_continuity']


def get_continuity_strength(
    high_dict: Dict[str, pd.Series],
    low_dict: Dict[str, pd.Series],
    direction: str = 'bullish',
    bar_index: int = -1,
    open_dict: Optional[Dict[str, pd.Series]] = None,
    close_dict: Optional[Dict[str, pd.Series]] = None
) -> int:
    """
    Get continuity strength score (0-5).

    Parameters:
    -----------
    high_dict : dict
        Dictionary mapping timeframes to high price Series
    low_dict : dict
        Dictionary mapping timeframes to low price Series
    direction : str, default 'bullish'
        Direction to check: 'bullish' or 'bearish'
    bar_index : int, default -1
        Bar index to check (-1 = most recent bar)
    open_dict : dict, optional
        Dictionary mapping timeframes to open price Series (for Type 3 candle color)
    close_dict : dict, optional
        Dictionary mapping timeframes to close price Series (for Type 3 candle color)

    Returns:
    --------
    int
        Number of aligned timeframes (0-5)
        5 = Full continuity (highest conviction)
        3-4 = Partial continuity (medium conviction)
        0-2 = Weak or no continuity (low conviction)

    Examples:
    ---------
    >>> strength = get_continuity_strength(high_dict, low_dict, 'bullish')
    >>> if strength >= 5:
    ...     signal_quality = 'HIGH'
    ... elif strength >= 3:
    ...     signal_quality = 'MEDIUM'
    ... else:
    ...     signal_quality = 'LOW'
    """
    checker = TimeframeContinuityChecker()
    result = checker.check_continuity(
        high_dict, low_dict, direction, bar_index, open_dict, close_dict
    )
    return result['strength']
