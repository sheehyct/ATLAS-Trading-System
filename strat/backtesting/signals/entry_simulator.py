"""
Entry Simulator - Backtest equivalent of EntryMonitor

Evaluates whether a detected signal should trigger an entry
on a given bar. Implements:
- Gap-through detection (entry at open, not trigger)
- Time gates for 1H patterns (10:30 for 2-bar, 11:30 for 3-bar)
- Bidirectional logic (both directions for 3-x patterns)
- TFC re-evaluation at entry time
- Hourly daily entry limit
"""

import logging
from datetime import datetime, time
from typing import Optional, Dict

import pandas as pd

from strat.backtesting.config import BacktestConfig
from strat.backtesting.signals.signal_generator import BacktestSignal

logger = logging.getLogger(__name__)


class EntrySimulator:
    """
    Determines if and how a signal triggers entry on a given bar.

    Mirrors the live EntryMonitor logic but evaluates against
    historical bar data instead of real-time prices.

    Usage:
        sim = EntrySimulator(config)
        result = sim.evaluate_entry(signal, bar, bar_time)
    """

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._hourly_entries_today: Dict[str, int] = {}  # date_str -> count
        self._tfc_checker = None

    def evaluate_entry(
        self,
        signal: BacktestSignal,
        bar: pd.Series,
        bar_time: datetime,
    ) -> Optional[dict]:
        """
        Check if a signal triggers entry on this bar.

        Args:
            signal: The detected signal to evaluate
            bar: OHLCV bar data (Open, High, Low, Close)
            bar_time: Timestamp of this bar

        Returns:
            Dict with entry details if triggered, None otherwise.
            Keys: actual_entry_price, gap_through, direction
        """
        # Time gate check for 1H patterns
        if not self._passes_time_gate(signal, bar_time):
            return None

        # Hourly daily limit check
        if not self._passes_hourly_limit(signal, bar_time):
            return None

        # Check trigger price vs bar range
        entry_result = self._check_trigger(signal, bar, bar_time)
        if entry_result is None:
            return None

        # TFC re-evaluation (optional)
        if self._config.tfc_reeval_enabled:
            if not self._passes_tfc_reeval(signal, bar_time):
                return None

        return entry_result

    def _passes_time_gate(self, signal: BacktestSignal, bar_time: datetime) -> bool:
        """
        Check 'let the market breathe' time gates for 1H patterns.

        2-bar patterns: earliest entry at 10:30 ET
        3-bar patterns: earliest entry at 11:30 ET
        Non-hourly patterns: no restriction
        """
        if signal.timeframe != '1H':
            return True

        bar_time_of_day = bar_time.time() if hasattr(bar_time, 'time') else time(0, 0)

        if signal.num_bars_in_pattern <= 2:
            earliest = time(
                self._config.hourly_2bar_earliest_hour,
                self._config.hourly_2bar_earliest_minute,
            )
        else:
            earliest = time(
                self._config.hourly_3bar_earliest_hour,
                self._config.hourly_3bar_earliest_minute,
            )

        if bar_time_of_day < earliest:
            logger.debug(
                "Time gate blocked %s %s: %s < %s",
                signal.symbol, signal.pattern_type, bar_time_of_day, earliest,
            )
            return False

        return True

    def _passes_hourly_limit(self, signal: BacktestSignal, bar_time: datetime) -> bool:
        """Check hourly daily entry limit."""
        if self._config.max_hourly_entries_per_day < 0:
            return True  # Unlimited

        if signal.timeframe != '1H':
            return True

        date_str = bar_time.strftime('%Y-%m-%d')
        count = self._hourly_entries_today.get(date_str, 0)

        if count >= self._config.max_hourly_entries_per_day:
            return False

        return True

    def record_hourly_entry(self, bar_time: datetime) -> None:
        """Record that an hourly entry was made (call after successful entry)."""
        date_str = bar_time.strftime('%Y-%m-%d')
        self._hourly_entries_today[date_str] = self._hourly_entries_today.get(date_str, 0) + 1

    def _check_trigger(
        self,
        signal: BacktestSignal,
        bar: pd.Series,
        bar_time: datetime,
    ) -> Optional[dict]:
        """
        Check if the bar triggers entry for this signal.

        Handles:
        - Normal trigger: bar range passes through entry_trigger
        - Gap-through: bar opens beyond trigger (entry at open price)
        """
        bar_open = bar.get('Open', bar.get('open', 0))
        bar_high = bar.get('High', bar.get('high', 0))
        bar_low = bar.get('Low', bar.get('low', 0))

        trigger = signal.entry_trigger
        is_bullish = signal.direction.upper() in ('CALL', 'BULL', 'UP')

        if is_bullish:
            # Bull entry: price must break ABOVE trigger
            if bar_high < trigger:
                return None  # Bar didn't reach trigger

            # Gap-through: opened above trigger
            gap_through = bar_open >= trigger
            actual_entry = bar_open if gap_through else trigger

            # Validate entry < target (skip if entry exceeds target)
            if actual_entry >= signal.target_price:
                return None

        else:
            # Bear entry: price must break BELOW trigger
            if bar_low > trigger:
                return None  # Bar didn't reach trigger

            # Gap-through: opened below trigger
            gap_through = bar_open <= trigger
            actual_entry = bar_open if gap_through else trigger

            # Validate entry < target (for puts, target < entry)
            if actual_entry <= signal.target_price:
                return None

        return {
            'actual_entry_price': actual_entry,
            'gap_through': gap_through,
            'direction': signal.direction,
        }

    def _passes_tfc_reeval(self, signal: BacktestSignal, bar_time: datetime) -> bool:
        """
        Re-evaluate TFC at entry time.

        TFC can change between pattern detection and entry trigger.
        Block entry if alignment degraded below minimum strength.
        """
        if not self._config.tfc_reeval_enabled:
            return True

        # If signal already has TFC data, use it for quick check
        if signal.tfc_score > 0:
            if signal.tfc_score < self._config.tfc_reeval_min_strength:
                logger.debug(
                    "TFC re-eval blocked %s: score %d < min %d",
                    signal.signal_key if hasattr(signal, 'signal_key') else signal.symbol,
                    signal.tfc_score, self._config.tfc_reeval_min_strength,
                )
                return False

        # Full TFC re-evaluation would require fetching multi-TF data
        # at the entry bar time. For backtesting efficiency, we rely on
        # the TFC score captured at detection time (already filtered).
        # A full re-eval can be added as an optional mode later.
        return True
