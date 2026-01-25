"""
EQUITY-90: TrailingStopManager - Extracted from PositionMonitor

Manages trailing stop logic for tracked option positions.

Supports two trailing stop strategies:
1. ATR-based (EQUITY-52): For 3-2 patterns
   - Activation: 0.75 ATR profit
   - Trail distance: 1.0 ATR from high water mark

2. Percentage-based (EQUITY-36): For all other patterns
   - Activation: 0.5x R:R profit
   - Trail distance: 50% of profit from high water mark

Session EQUITY-90: Phase 4.2 - God class refactoring.
"""

import logging
from typing import Optional

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """
    Manages trailing stop calculations and activation for option positions.

    Extracted from PositionMonitor as part of EQUITY-90 Phase 4.2 refactoring.
    Implements the TrailingStopChecker protocol defined in exit_evaluator.

    Two trailing stop strategies are supported:
    - ATR-based: For 3-2 patterns (more volatile, need wider trail)
    - Percentage-based: For reversal patterns (tighter trail)

    Usage:
        manager = TrailingStopManager(config)
        exit_signal = manager.check(position)

    Args:
        config: MonitoringConfig with trailing stop settings
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize trailing stop manager.

        Args:
            config: MonitoringConfig with trailing stop thresholds
        """
        self._config = config

    def check(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Check and update trailing stop for a position.

        This method:
        1. Updates the high water mark as price moves favorably
        2. Activates trailing stop when profit threshold is reached
        3. Calculates trailing stop level based on strategy
        4. Returns exit signal if trailing stop is hit

        Routes to appropriate strategy:
        - ATR-based for 3-2 patterns (pos.use_atr_trailing = True)
        - Percentage-based for all other patterns

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal if trailing stop is hit, None otherwise
        """
        # Route to appropriate trailing stop method based on pattern
        if pos.use_atr_trailing and pos.atr_trail_distance > 0:
            return self._check_atr_trailing_stop(pos)
        else:
            return self._check_percentage_trailing_stop(pos)

    def _check_atr_trailing_stop(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        ATR-based trailing stop for 3-2 patterns.

        EQUITY-52: 3-2 patterns are breakout trades with higher volatility.
        Uses ATR-based activation and trail distances for more adaptive behavior.

        Logic:
        1. Update high water mark as price moves in our favor
        2. Activate trailing stop once 0.75 ATR profit is reached
        3. Trail stop at 1.0 ATR distance from high water mark
        4. Exit if price retraces to trailing stop level

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal if trailing stop is hit, None otherwise
        """
        is_bullish = pos.direction.upper() in ['CALL', 'BULL', 'UP']
        entry_price = pos.actual_entry_underlying if pos.actual_entry_underlying > 0 else pos.entry_trigger

        # Calculate profit in direction of trade
        if is_bullish:
            current_profit = pos.underlying_price - entry_price
            # Update high water mark
            if pos.underlying_price > pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price
        else:
            current_profit = entry_price - pos.underlying_price
            # Update high water mark (lowest price for puts)
            if pos.underlying_price < pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price

        # Check activation: 0.75 ATR profit
        activation_threshold = pos.atr_at_detection * self._config.atr_trailing_activation_multiple

        if not pos.trailing_stop_active and current_profit >= activation_threshold:
            pos.trailing_stop_active = True
            logger.info(
                f"ATR Trailing stop ACTIVATED for {pos.osi_symbol}: "
                f"profit ${current_profit:.2f} >= 0.75 ATR (${activation_threshold:.2f})"
            )

        # If active, calculate and check trailing stop level
        if pos.trailing_stop_active:
            trail_distance = pos.atr_trail_distance  # Pre-calculated 1.0 ATR

            if is_bullish:
                pos.trailing_stop_price = pos.high_water_mark - trail_distance

                if pos.underlying_price <= pos.trailing_stop_price:
                    # Check minimum profit requirement
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        logger.info(
                            f"ATR TRAILING STOP BLOCKED for {pos.osi_symbol}: "
                            f"option P/L {pos.unrealized_pct:.1%} < min"
                        )
                        return None

                    logger.info(
                        f"ATR TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} <= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f}, 1.0 ATR: ${trail_distance:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"ATR trailing stop at ${pos.trailing_stop_price:.2f} hit (1.0 ATR trail, HWM: ${pos.high_water_mark:.2f})",
                    )
            else:  # Bearish (PUT)
                pos.trailing_stop_price = pos.high_water_mark + trail_distance

                if pos.underlying_price >= pos.trailing_stop_price:
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        logger.info(
                            f"ATR TRAILING STOP BLOCKED for {pos.osi_symbol}: "
                            f"option P/L {pos.unrealized_pct:.1%} < min"
                        )
                        return None

                    logger.info(
                        f"ATR TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} >= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f}, 1.0 ATR: ${trail_distance:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"ATR trailing stop at ${pos.trailing_stop_price:.2f} hit (1.0 ATR trail, HWM: ${pos.high_water_mark:.2f})",
                    )

        return None

    def _check_percentage_trailing_stop(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Percentage-based trailing stop for non-3-2 patterns.

        EQUITY-36: Reversal patterns (2-1-2, 3-1-2) use risk-based
        activation and percentage-based trail.

        Logic:
        1. Update high water mark as price moves in our favor
        2. Activate trailing stop once 0.5x R:R profit is reached
        3. Trail stop at 50% of profit from high water mark
        4. Exit if price retraces to trailing stop level

        CRITICAL FIX (EQUITY-36): Uses actual_entry_underlying for P/L
        calculations, not entry_trigger. For gap-through scenarios,
        entry_trigger differs from actual entry price.

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal if trailing stop is hit, None otherwise
        """
        is_bullish = pos.direction.upper() in ['CALL', 'BULL', 'UP']

        # Use actual entry price, fallback to trigger level
        entry_price = pos.actual_entry_underlying if pos.actual_entry_underlying > 0 else pos.entry_trigger

        risk = abs(entry_price - pos.stop_price)
        activation_threshold = self._config.trailing_stop_activation_rr * risk

        # Calculate profit in direction of trade
        if is_bullish:
            current_profit = pos.underlying_price - entry_price
            # Update high water mark
            if pos.underlying_price > pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price
        else:
            current_profit = entry_price - pos.underlying_price
            # Update high water mark (lowest price for puts)
            if pos.underlying_price < pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price

        # Check if we should activate trailing stop
        if not pos.trailing_stop_active and current_profit >= activation_threshold:
            pos.trailing_stop_active = True
            logger.info(
                f"Trailing stop ACTIVATED for {pos.osi_symbol}: "
                f"profit ${current_profit:.2f} >= activation ${activation_threshold:.2f} (0.5x R:R)"
            )

        # If trailing stop is active, calculate and check trailing stop level
        if pos.trailing_stop_active:
            # Calculate trail amount (50% of max profit from high water mark)
            if is_bullish:
                max_profit_from_hwm = pos.high_water_mark - entry_price
                trail_amount = max_profit_from_hwm * self._config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark - trail_amount

                # Check if trailing stop is hit
                if pos.underlying_price <= pos.trailing_stop_price:
                    # Don't exit at a loss with trailing stop (EQUITY-42)
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        logger.info(
                            f"TRAILING STOP BLOCKED for {pos.osi_symbol}: "
                            f"option P/L {pos.unrealized_pct:.1%} < min {self._config.trailing_stop_min_profit_pct:.1%}"
                        )
                        return None

                    logger.info(
                        f"TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} <= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f}, entry: ${entry_price:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"Trailing stop at ${pos.trailing_stop_price:.2f} hit (HWM: ${pos.high_water_mark:.2f})",
                    )
            else:  # Bearish (PUT)
                max_profit_from_hwm = entry_price - pos.high_water_mark
                trail_amount = max_profit_from_hwm * self._config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark + trail_amount

                # Check if trailing stop is hit
                if pos.underlying_price >= pos.trailing_stop_price:
                    # Don't exit at a loss with trailing stop (EQUITY-42)
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        logger.info(
                            f"TRAILING STOP BLOCKED for {pos.osi_symbol}: "
                            f"option P/L {pos.unrealized_pct:.1%} < min {self._config.trailing_stop_min_profit_pct:.1%}"
                        )
                        return None

                    logger.info(
                        f"TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} >= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f}, entry: ${entry_price:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"Trailing stop at ${pos.trailing_stop_price:.2f} hit (HWM: ${pos.high_water_mark:.2f})",
                    )

        return None

    @property
    def config(self) -> MonitoringConfig:
        """Return current monitoring configuration."""
        return self._config
