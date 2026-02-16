"""
Backtest Signal Generator

Wraps the existing detect_all_patterns() and FilterManager
to generate trading signals from historical bar data.

This is a thin adapter - the heavy lifting is done by the
existing pattern detection infrastructure.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

from strat.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestSignal:
    """
    A detected pattern signal for backtesting.

    Simplified version of DetectedSignal/StoredSignal from the live system,
    containing only the fields needed for backtest entry/exit simulation.
    """
    # Pattern identification
    pattern_type: str           # e.g., '2D-2U', '3-1-2D'
    direction: str              # 'CALL' or 'PUT'
    symbol: str
    timeframe: str

    # Price levels
    entry_trigger: float        # Price that triggers entry
    stop_price: float
    target_price: float

    # Detection metadata
    detected_time: datetime
    detected_bar_index: int     # Index into the bar DataFrame

    # Quality metrics
    magnitude_pct: float = 0.0
    risk_reward: float = 0.0

    # Setup bar bounds (for pattern invalidation)
    setup_bar_high: float = 0.0
    setup_bar_low: float = 0.0

    # Pattern classification
    is_bidirectional: bool = True
    signal_type: str = 'SETUP'  # 'SETUP' or 'COMPLETED'
    num_bars_in_pattern: int = 2  # 2 for 2-bar, 3 for 3-bar patterns

    # ATR at detection (for 3-2 ATR trailing stops)
    atr_at_detection: float = 0.0

    # TFC context at detection
    tfc_score: int = 0
    tfc_alignment: str = ''
    tfc_passes: bool = False
    priority_rank: int = 0


class BacktestSignalGenerator:
    """
    Generates trading signals from historical OHLCV data.

    Reuses the existing pattern detection infrastructure:
    - detect_all_patterns() for pattern recognition
    - FilterManager for quality filtering

    Usage:
        generator = BacktestSignalGenerator(config)
        signals = generator.detect_signals(ohlcv_df, 'SPY', '1D')
    """

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._filter_manager = None

    def _ensure_filter_manager(self):
        """Lazy-load FilterManager to avoid import issues."""
        if self._filter_manager is None:
            try:
                from strat.signal_automation.coordinators.filter_manager import (
                    FilterManager, FilterConfig,
                )
                filter_config = FilterConfig(
                    min_magnitude_pct=self._config.min_magnitude_pct,
                    min_risk_reward=self._config.min_risk_reward,
                )
                self._filter_manager = FilterManager(filter_config)
            except ImportError:
                logger.warning("FilterManager not available, using no filters")

    def detect_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> List[BacktestSignal]:
        """
        Detect all STRAT pattern signals in a DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            symbol: Symbol being scanned
            timeframe: Timeframe string ('1H', '1D', etc.)

        Returns:
            List of BacktestSignal, chronologically ordered
        """
        from strat.unified_pattern_detector import detect_all_patterns

        self._ensure_filter_manager()

        try:
            patterns = detect_all_patterns(
                df,
                timeframe=timeframe,
                patterns=self._config.patterns,
            )
        except Exception as e:
            logger.error("Pattern detection failed for %s %s: %s", symbol, timeframe, e)
            return []

        if patterns is None or patterns.empty:
            return []

        signals = []
        atr_14 = self._calculate_atr(df)

        for idx, row in patterns.iterrows():
            signal = self._convert_pattern_to_signal(
                row, idx, symbol, timeframe, df, atr_14,
            )
            if signal is None:
                continue

            # Apply quality filters
            if self._filter_manager:
                if not self._passes_filters(signal):
                    continue

            signals.append(signal)

        logger.debug("Detected %d signals for %s %s", len(signals), symbol, timeframe)
        return signals

    def _convert_pattern_to_signal(
        self,
        row: pd.Series,
        timestamp: pd.Timestamp,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        atr_14: float,
    ) -> Optional[BacktestSignal]:
        """Convert a pattern detection row to a BacktestSignal."""
        try:
            pattern_type = row.get('pattern_type', row.get('pattern', ''))
            direction_raw = row.get('direction', 0)

            # Convert numeric direction to string
            if isinstance(direction_raw, (int, float)):
                direction = 'CALL' if direction_raw > 0 else 'PUT'
            else:
                direction = str(direction_raw).upper()
                if direction in ('BULL', 'UP', 'LONG'):
                    direction = 'CALL'
                elif direction in ('BEAR', 'DOWN', 'SHORT'):
                    direction = 'PUT'

            entry_trigger = float(row.get('entry_price', row.get('trigger_price', 0)))
            stop_price = float(row.get('stop_price', 0))
            target_price = float(row.get('target_price', 0))

            if entry_trigger <= 0 or stop_price <= 0 or target_price <= 0:
                return None

            # Calculate metrics
            risk = abs(entry_trigger - stop_price)
            reward = abs(target_price - entry_trigger)
            rr = reward / risk if risk > 0 else 0.0
            magnitude_pct = reward / entry_trigger * 100 if entry_trigger > 0 else 0.0

            # Determine bar count in pattern
            pattern_str = str(pattern_type)
            parts = pattern_str.replace('U', '').replace('D', '').split('-')
            num_bars = len(parts)

            # Setup bar bounds
            setup_high = float(row.get('setup_bar_high', row.get('bar_high', 0)))
            setup_low = float(row.get('setup_bar_low', row.get('bar_low', 0)))

            # Find bar index
            try:
                bar_index = df.index.get_loc(timestamp)
            except (KeyError, TypeError):
                bar_index = 0

            # Check bidirectionality
            is_bidir = True
            try:
                from strat.pattern_registry import is_bidirectional_pattern
                is_bidir = is_bidirectional_pattern(pattern_type)
            except ImportError:
                pass

            # Determine if this is a 3-2 pattern (for ATR trailing)
            is_32_pattern = '3-2' in pattern_str and '3-2-2' not in pattern_str

            return BacktestSignal(
                pattern_type=pattern_type,
                direction=direction,
                symbol=symbol,
                timeframe=timeframe,
                entry_trigger=entry_trigger,
                stop_price=stop_price,
                target_price=target_price,
                detected_time=timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp,
                detected_bar_index=bar_index,
                magnitude_pct=magnitude_pct,
                risk_reward=rr,
                setup_bar_high=setup_high,
                setup_bar_low=setup_low,
                is_bidirectional=is_bidir,
                num_bars_in_pattern=num_bars,
                atr_at_detection=atr_14 if is_32_pattern else 0.0,
            )
        except Exception as e:
            logger.debug("Failed to convert pattern row: %s", e)
            return None

    def _passes_filters(self, signal: BacktestSignal) -> bool:
        """Apply quality filters to a signal."""
        if signal.magnitude_pct < self._config.min_magnitude_pct:
            return False
        if signal.risk_reward < self._config.min_risk_reward:
            return False
        return True

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR(14) from OHLCV data."""
        if len(df) < period + 1:
            return 0.0

        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        tr = []
        for i in range(1, len(high)):
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            ))

        if len(tr) < period:
            return sum(tr) / len(tr) if tr else 0.0

        return sum(tr[-period:]) / period
