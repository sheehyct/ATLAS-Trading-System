"""
Trailing Stop Evaluator - Backtest Port

Two strategies mirroring the live TrailingStopManager:

1. ATR-based (EQUITY-52): For 3-2 patterns
   - Activation: 0.75 ATR profit from entry
   - Trail distance: 1.0 ATR from high water mark

2. Percentage-based (EQUITY-36): For all other patterns
   - Activation: 0.5x R:R profit from entry
   - Trail distance: 50% below high water mark profit

Critical fix (EQUITY-36): Uses actual_entry_underlying for P&L
calculations, not entry_trigger.
"""

import logging
from typing import Optional

from strat.backtesting.config import BacktestConfig
from strat.backtesting.simulation.position_tracker import SimulatedPosition, ExitReason

logger = logging.getLogger(__name__)


class TrailingStopEvaluator:
    """
    Evaluates trailing stop conditions for backtest positions.

    Routes to ATR-based or percentage-based strategy based on
    position's use_atr_trailing flag (set for 3-2 patterns).
    """

    def __init__(self, config: BacktestConfig):
        self._config = config

    def check(self, pos: SimulatedPosition) -> Optional['ExitEvalResult']:
        """
        Check and update trailing stop for a position.

        Updates high water mark, activates trailing stop when
        threshold reached, and checks if stop level is hit.

        Returns:
            ExitEvalResult if trailing stop triggered, None otherwise
        """
        from strat.backtesting.exits.exit_evaluator import ExitEvalResult

        if pos.use_atr_trailing and pos.atr_trail_distance > 0:
            return self._check_atr_trailing(pos)
        else:
            return self._check_pct_trailing(pos)

    def _check_atr_trailing(self, pos: SimulatedPosition) -> Optional['ExitEvalResult']:
        """
        ATR-based trailing stop for 3-2 patterns (EQUITY-52).

        Activation: 0.75 ATR profit
        Trail: 1.0 ATR from high water mark
        """
        from strat.backtesting.exits.exit_evaluator import ExitEvalResult

        entry_price = pos.actual_entry_underlying if pos.actual_entry_underlying > 0 else pos.entry_trigger

        # Calculate profit in trade direction
        if pos.is_bullish:
            current_profit = pos.underlying_price - entry_price
            # Update HWM
            if pos.underlying_price > pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price
        else:
            current_profit = entry_price - pos.underlying_price
            if pos.underlying_price < pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price

        # Check activation
        activation_threshold = pos.atr_at_detection * self._config.atr_trailing_activation_multiple
        if not pos.trailing_stop_active and current_profit >= activation_threshold:
            pos.trailing_stop_active = True
            logger.debug(
                "ATR trailing activated for %s: profit $%.2f >= %.2f ATR ($%.2f)",
                pos.symbol, current_profit,
                self._config.atr_trailing_activation_multiple, activation_threshold,
            )

        # If active, calculate trail and check
        if pos.trailing_stop_active:
            trail_distance = pos.atr_trail_distance

            if pos.is_bullish:
                pos.trailing_stop_price = pos.high_water_mark - trail_distance
                if pos.underlying_price <= pos.trailing_stop_price:
                    # Min profit gate
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        return None
                    return ExitEvalResult(True, ExitReason.TRAILING_STOP,
                        f"ATR trail at ${pos.trailing_stop_price:.2f} hit "
                        f"(HWM: ${pos.high_water_mark:.2f}, 1.0 ATR: ${trail_distance:.2f})")
            else:
                pos.trailing_stop_price = pos.high_water_mark + trail_distance
                if pos.underlying_price >= pos.trailing_stop_price:
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        return None
                    return ExitEvalResult(True, ExitReason.TRAILING_STOP,
                        f"ATR trail at ${pos.trailing_stop_price:.2f} hit "
                        f"(HWM: ${pos.high_water_mark:.2f}, 1.0 ATR: ${trail_distance:.2f})")

        return None

    def _check_pct_trailing(self, pos: SimulatedPosition) -> Optional['ExitEvalResult']:
        """
        Percentage-based trailing stop for non-3-2 patterns (EQUITY-36).

        Activation: 0.5x R:R profit
        Trail: 50% of max profit from high water mark
        """
        from strat.backtesting.exits.exit_evaluator import ExitEvalResult

        entry_price = pos.actual_entry_underlying if pos.actual_entry_underlying > 0 else pos.entry_trigger
        risk = abs(entry_price - pos.stop_price)
        if risk <= 0:
            return None

        activation_threshold = self._config.trailing_stop_activation_rr * risk

        if pos.is_bullish:
            current_profit = pos.underlying_price - entry_price
            if pos.underlying_price > pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price
        else:
            current_profit = entry_price - pos.underlying_price
            if pos.underlying_price < pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price

        # Activate
        if not pos.trailing_stop_active and current_profit >= activation_threshold:
            pos.trailing_stop_active = True
            logger.debug(
                "Pct trailing activated for %s: profit $%.2f >= 0.5x R:R ($%.2f)",
                pos.symbol, current_profit, activation_threshold,
            )

        # Calculate and check
        if pos.trailing_stop_active:
            if pos.is_bullish:
                max_profit = pos.high_water_mark - entry_price
                trail_amount = max_profit * self._config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark - trail_amount

                if pos.underlying_price <= pos.trailing_stop_price:
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        return None
                    return ExitEvalResult(True, ExitReason.TRAILING_STOP,
                        f"Pct trail at ${pos.trailing_stop_price:.2f} hit "
                        f"(HWM: ${pos.high_water_mark:.2f})")
            else:
                max_profit = entry_price - pos.high_water_mark
                trail_amount = max_profit * self._config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark + trail_amount

                if pos.underlying_price >= pos.trailing_stop_price:
                    if pos.unrealized_pct < self._config.trailing_stop_min_profit_pct:
                        return None
                    return ExitEvalResult(True, ExitReason.TRAILING_STOP,
                        f"Pct trail at ${pos.trailing_stop_price:.2f} hit "
                        f"(HWM: ${pos.high_water_mark:.2f})")

        return None
