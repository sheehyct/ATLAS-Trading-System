"""
Backtest Exit Condition Evaluator

Evaluates all 9 exit conditions in strict priority order,
mirroring the live ExitConditionEvaluator exactly.

Priority Order:
1. TARGET_HIT     - Underlying reached target price
2. STOP_HIT       - Underlying reached stop price
3. DTE_EXIT       - DTE below threshold (3 days)
4. MAX_LOSS       - Option unrealized loss exceeds TF-specific threshold
5. EOD_EXIT       - 1H trades: exit at 15:55 ET
6. PATTERN_INVALIDATED - Entry bar evolved to Type 3
7. TRAILING_STOP  - Trailing stop triggered
8. PARTIAL_EXIT   - Multi-contract partial at 1.0x R:R
9. TIME_EXIT      - Max holding bars exceeded
"""

import logging
from datetime import datetime
from typing import Optional

from strat.backtesting.config import BacktestConfig
from strat.backtesting.simulation.position_tracker import SimulatedPosition, ExitReason
from strat.backtesting.exits.trailing_stop import TrailingStopEvaluator
from strat.backtesting.exits.partial_exit import PartialExitEvaluator

logger = logging.getLogger(__name__)


class ExitEvalResult:
    """Result of exit condition evaluation."""

    __slots__ = ('should_exit', 'reason', 'details', 'contracts_to_close')

    def __init__(
        self,
        should_exit: bool = False,
        reason: Optional[ExitReason] = None,
        details: str = "",
        contracts_to_close: Optional[int] = None,
    ):
        self.should_exit = should_exit
        self.reason = reason
        self.details = details
        self.contracts_to_close = contracts_to_close  # None = all


class BacktestExitEvaluator:
    """
    Evaluates all 9 exit conditions for backtest positions.

    Mirrors the live ExitConditionEvaluator's priority ordering.
    First matching condition wins; remaining conditions are skipped.

    Usage:
        evaluator = BacktestExitEvaluator(config)
        result = evaluator.evaluate(position, bar_time, bar_data)
    """

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._trailing = TrailingStopEvaluator(config)
        self._partial = PartialExitEvaluator(config)

    def evaluate(
        self,
        pos: SimulatedPosition,
        bar_time: datetime,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> ExitEvalResult:
        """
        Evaluate all exit conditions for a position against a bar.

        Uses intrabar price path simulation when a bar's range
        encompasses both target and stop to determine which fires first.

        Args:
            pos: The simulated position
            bar_time: Current bar timestamp
            bar_high: Bar high price
            bar_low: Bar low price
            bar_close: Bar close price

        Returns:
            ExitEvalResult with should_exit=True if any condition met
        """
        # Update intrabar extremes
        pos.update_underlying(bar_high)
        pos.update_underlying(bar_low)
        pos.underlying_price = bar_close
        pos.bars_held += 1

        # ── Priority 1: TARGET_HIT ──────────────────────────────────
        target_hit = self._check_target_hit(pos, bar_high, bar_low)

        # ── Priority 2: STOP_HIT ────────────────────────────────────
        stop_hit = self._check_stop_hit(pos, bar_high, bar_low)

        # Same-bar resolution: if both target and stop within bar range,
        # use intrabar price path to determine which fires first
        if target_hit and stop_hit:
            first = self._resolve_same_bar(pos, bar_high, bar_low, bar_close)
            if first == 'target':
                return ExitEvalResult(True, ExitReason.TARGET_HIT,
                    f"Target ${pos.target_price:.2f} hit (same-bar resolution favored target)")
            else:
                return ExitEvalResult(True, ExitReason.STOP_HIT,
                    f"Stop ${pos.stop_price:.2f} hit (same-bar resolution favored stop)")

        if target_hit:
            return ExitEvalResult(True, ExitReason.TARGET_HIT,
                f"Underlying hit target ${pos.target_price:.2f}")

        if stop_hit:
            return ExitEvalResult(True, ExitReason.STOP_HIT,
                f"Underlying hit stop ${pos.stop_price:.2f}")

        # ── Priority 3: DTE_EXIT ────────────────────────────────────
        if pos.expiration:
            dte = self._calculate_dte(pos.expiration, bar_time)
            pos.dte = dte
            if dte <= self._config.exit_dte:
                return ExitEvalResult(True, ExitReason.DTE_EXIT,
                    f"DTE {dte} <= threshold {self._config.exit_dte}")

        # ── Priority 4: MAX_LOSS ────────────────────────────────────
        max_loss = self._config.get_max_loss_pct(pos.timeframe)
        if pos.unrealized_pct <= -max_loss:
            return ExitEvalResult(True, ExitReason.MAX_LOSS,
                f"Loss {pos.unrealized_pct:.1%} >= max {max_loss:.1%} ({pos.timeframe})")

        # ── Priority 5: EOD_EXIT (1H only) ─────────────────────────
        if pos.timeframe == '1H':
            eod_result = self._check_eod_exit(pos, bar_time)
            if eod_result:
                return eod_result

        # ── Priority 6: PATTERN_INVALIDATED ─────────────────────────
        if self._config.pattern_invalidation_enabled:
            invalidation = self._check_pattern_invalidation(pos)
            if invalidation:
                return invalidation

        # ── Priority 7: TRAILING_STOP ───────────────────────────────
        if self._config.use_trailing_stop:
            trail_result = self._trailing.check(pos)
            if trail_result:
                return trail_result

        # ── Priority 8: PARTIAL_EXIT ────────────────────────────────
        if self._config.partial_exit_enabled:
            partial_result = self._partial.check(pos)
            if partial_result:
                return partial_result

        # ── Priority 9: TIME_EXIT ───────────────────────────────────
        max_bars = self._config.get_max_holding(pos.timeframe)
        if pos.bars_held >= max_bars:
            return ExitEvalResult(True, ExitReason.TIME_EXIT,
                f"Held {pos.bars_held} bars >= max {max_bars} ({pos.timeframe})")

        return ExitEvalResult(should_exit=False)

    def _check_target_hit(
        self, pos: SimulatedPosition, bar_high: float, bar_low: float,
    ) -> bool:
        """Check if underlying reached target price within bar."""
        if pos.is_bullish:
            return bar_high >= pos.target_price
        else:
            return bar_low <= pos.target_price

    def _check_stop_hit(
        self, pos: SimulatedPosition, bar_high: float, bar_low: float,
    ) -> bool:
        """Check if underlying reached stop price within bar."""
        if pos.is_bullish:
            return bar_low <= pos.stop_price
        else:
            return bar_high >= pos.stop_price

    def _resolve_same_bar(
        self,
        pos: SimulatedPosition,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> str:
        """
        Resolve same-bar target/stop ambiguity using intrabar price path.

        Uses bar type classification logic:
        - For Type 2U (bullish bar): Open -> Low -> High -> Close
          (favorable first for calls, adverse first for puts)
        - For Type 2D (bearish bar): Open -> High -> Low -> Close
          (adverse first for calls, favorable first for puts)
        - For Type 3: use candle color (green = high first, red = low first)

        Returns: 'target' or 'stop'
        """
        # Determine if bar is green (close > open implies up move first)
        is_green = bar_close > (bar_high + bar_low) / 2  # Approximate open

        if pos.is_bullish:
            # For calls: high = favorable, low = adverse
            # Green bar: price went up first -> hit target first
            if is_green:
                return 'target'
            else:
                return 'stop'
        else:
            # For puts: low = favorable, high = adverse
            # Red bar: price went down first -> hit target first
            if not is_green:
                return 'target'
            else:
                return 'stop'

    def _check_eod_exit(
        self, pos: SimulatedPosition, bar_time: datetime,
    ) -> Optional[ExitEvalResult]:
        """Check EOD exit for 1H trades."""
        if pos.timeframe != '1H':
            return None

        bar_hour = bar_time.hour if hasattr(bar_time, 'hour') else 0
        bar_minute = bar_time.minute if hasattr(bar_time, 'minute') else 0

        if (bar_hour > self._config.eod_exit_hour or
            (bar_hour == self._config.eod_exit_hour and
             bar_minute >= self._config.eod_exit_minute)):
            return ExitEvalResult(True, ExitReason.EOD_EXIT,
                f"1H EOD exit at {bar_hour}:{bar_minute:02d} ET")

        return None

    def _check_pattern_invalidation(
        self, pos: SimulatedPosition,
    ) -> Optional[ExitEvalResult]:
        """
        Check if entry bar evolved to Type 3 (pattern invalidation).

        Type 3 = intrabar price broke BOTH setup bar high AND low.
        """
        if pos.entry_bar_type not in ('2U', '2D'):
            return None

        if pos.entry_bar_high <= 0 or pos.entry_bar_low <= 0:
            return None

        broke_high = pos.intrabar_high > pos.entry_bar_high
        broke_low = pos.intrabar_low < pos.entry_bar_low

        if broke_high and broke_low:
            return ExitEvalResult(True, ExitReason.PATTERN_INVALIDATED,
                f"Type 3 evolution: setup H=${pos.entry_bar_high:.2f} L=${pos.entry_bar_low:.2f}, "
                f"intrabar H=${pos.intrabar_high:.2f} L=${pos.intrabar_low:.2f}")

        return None

    @staticmethod
    def _calculate_dte(expiration: str, current_date: datetime) -> int:
        """Calculate days to expiration."""
        try:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            return max(0, (exp_date - current_date).days)
        except (ValueError, TypeError):
            return 0
