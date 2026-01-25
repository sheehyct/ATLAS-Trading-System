"""
EQUITY-90: ExitConditionEvaluator - Extracted from PositionMonitor

Evaluates exit conditions for tracked option positions in priority order:
1. Minimum hold time (skip all checks if too soon after entry)
2. EOD exit for 1H trades (avoid overnight gap risk)
3. DTE exit (mandatory close before expiration)
4. Stop hit (underlying price check)
5. Max loss (timeframe-specific thresholds)
6. Target hit (underlying reached profit target)
7. Pattern invalidation (Type 3 evolution)
8. Trailing stop (delegate to TrailingStopManager)
9. Partial exit (delegate to PartialExitManager)
10. Max profit (option premium gain threshold)

Session EQUITY-90: Phase 4.1 - God class refactoring.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Protocol

import pytz
import pandas_market_calendars as mcal

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)

logger = logging.getLogger(__name__)


class TrailingStopChecker(Protocol):
    """Protocol for trailing stop evaluation."""

    def check(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """Check if trailing stop should trigger for this position."""
        ...


class PartialExitChecker(Protocol):
    """Protocol for partial exit evaluation."""

    def check(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """Check if partial exit should trigger for this position."""
        ...


class ExitConditionEvaluator:
    """
    Evaluates exit conditions for tracked option positions.

    Extracted from PositionMonitor as part of EQUITY-90 Phase 4.1 refactoring.
    Uses Facade pattern - monitor delegates exit evaluation to this class.

    Exit conditions are checked in strict priority order. The first condition
    that triggers returns an ExitSignal; remaining conditions are skipped.

    Dependencies:
    - MonitoringConfig: Thresholds for DTE, max loss, trailing stops, etc.
    - TrailingStopChecker: Optional delegate for trailing stop evaluation
    - PartialExitChecker: Optional delegate for partial exit evaluation

    Args:
        config: MonitoringConfig with exit thresholds
        trailing_stop_checker: Optional TrailingStopChecker implementation
        partial_exit_checker: Optional PartialExitChecker implementation
        timezone: Timezone for time comparisons (default: America/New_York)
    """

    def __init__(
        self,
        config: MonitoringConfig,
        trailing_stop_checker: Optional[TrailingStopChecker] = None,
        partial_exit_checker: Optional[PartialExitChecker] = None,
        timezone: str = 'America/New_York',
    ):
        self._config = config
        self._trailing_stop_checker = trailing_stop_checker
        self._partial_exit_checker = partial_exit_checker
        self._timezone = pytz.timezone(timezone)
        self._nyse_calendar = mcal.get_calendar('NYSE')

    def evaluate(
        self,
        pos: TrackedPosition,
        underlying_price: Optional[float] = None,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExitSignal]:
        """
        Evaluate exit conditions for a position.

        Args:
            pos: TrackedPosition to evaluate
            underlying_price: Current underlying price (updates pos.underlying_price)
            bar_data: Optional bar OHLCV data for pattern invalidation check

        Returns:
            ExitSignal if exit condition is met, None otherwise
        """
        # Update underlying price if provided
        if underlying_price is not None:
            pos.underlying_price = underlying_price
            # Update intrabar extremes for Type 3 detection
            if underlying_price > pos.intrabar_high:
                pos.intrabar_high = underlying_price
            if underlying_price < pos.intrabar_low:
                pos.intrabar_low = underlying_price

        # Priority 0: Minimum hold time (skip all checks if too soon)
        hold_duration = (datetime.now() - pos.entry_time).total_seconds()
        if hold_duration < self._config.minimum_hold_seconds:
            logger.debug(
                f"{pos.osi_symbol}: Held {hold_duration:.0f}s < "
                f"min {self._config.minimum_hold_seconds}s - skipping exit check"
            )
            return None

        # Priority 0.5: EOD exit for 1H trades
        eod_signal = self._check_eod_exit(pos)
        if eod_signal:
            return eod_signal

        # Update DTE for remaining checks
        pos.dte = self._calculate_dte(pos.expiration)

        # Priority 1: DTE exit
        if pos.dte <= self._config.exit_dte:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.DTE_EXIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"DTE {pos.dte} <= threshold {self._config.exit_dte}",
            )

        # Need underlying price for target/stop checks
        if not pos.underlying_price:
            logger.debug(f"No underlying price for {pos.symbol}")
            return None

        # Priority 2: Stop hit
        if self._check_stop_hit(pos):
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.STOP_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Underlying ${pos.underlying_price:.2f} hit stop ${pos.stop_price:.2f}",
            )

        # Priority 3: Max loss (timeframe-specific thresholds)
        max_loss_threshold = self._config.get_max_loss_pct(pos.timeframe)
        if pos.unrealized_pct <= -max_loss_threshold:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.MAX_LOSS,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Loss {pos.unrealized_pct:.1%} >= max {max_loss_threshold:.1%} ({pos.timeframe})",
            )

        # Priority 4: Target hit (check before trailing stop per EQUITY-42)
        if self._check_target_hit(pos):
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.TARGET_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Underlying ${pos.underlying_price:.2f} hit target ${pos.target_price:.2f}",
            )

        # Priority 4.5: Pattern invalidation (Type 3 evolution)
        invalidation_signal = self._check_pattern_invalidation(pos, bar_data)
        if invalidation_signal:
            return invalidation_signal

        # Priority 5: Trailing stop (delegate to manager)
        if self._trailing_stop_checker:
            trailing_signal = self._trailing_stop_checker.check(pos)
            if trailing_signal:
                return trailing_signal

        # Priority 6: Partial exit (delegate to manager)
        if self._partial_exit_checker:
            partial_signal = self._partial_exit_checker.check(pos)
            if partial_signal:
                return partial_signal

        # Priority 7: Max profit
        if pos.unrealized_pct >= self._config.max_profit_pct:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.TARGET_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Profit {pos.unrealized_pct:.1%} >= target {self._config.max_profit_pct:.1%}",
            )

        return None

    def _check_eod_exit(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Check for end-of-day exit condition for 1H trades.

        Session EQUITY-35: All hourly trades must exit before market close
        to avoid overnight gap risk.

        Session EQUITY-51: Also exit IMMEDIATELY if position is stale
        (entered on a previous trading day).

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal if EOD condition met, None otherwise
        """
        # Only applies to 1H timeframe variants
        if not pos.timeframe or pos.timeframe.upper() not in ['1H', '60MIN', '60M']:
            return None

        now_et = datetime.now(self._timezone)

        # Check for stale 1H position first (entered on previous trading day)
        if self._is_stale_1h_position(pos.entry_time):
            logger.warning(
                f"STALE EOD EXIT: {pos.osi_symbol} (1H) - "
                f"entered {pos.entry_time.strftime('%Y-%m-%d %H:%M')} but still open on {now_et.date()}"
            )
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.EOD_EXIT,
                underlying_price=pos.underlying_price or 0.0,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"STALE 1H trade - entered {pos.entry_time.strftime('%Y-%m-%d')} but not exited, closing immediately",
            )

        # Normal EOD exit check for positions entered today
        eod_exit_time = now_et.replace(
            hour=self._config.eod_exit_hour,
            minute=self._config.eod_exit_minute,
            second=0,
            microsecond=0
        )

        if now_et >= eod_exit_time:
            logger.info(
                f"EOD EXIT: {pos.osi_symbol} (1H) - "
                f"current time {now_et.strftime('%H:%M ET')} >= "
                f"EOD cutoff {self._config.eod_exit_hour}:{self._config.eod_exit_minute:02d} ET"
            )
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.EOD_EXIT,
                underlying_price=pos.underlying_price or 0.0,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"1H trade EOD exit at {now_et.strftime('%H:%M ET')} (cutoff: {self._config.eod_exit_hour}:{self._config.eod_exit_minute:02d} ET)",
            )

        return None

    def _is_stale_1h_position(self, entry_time: datetime) -> bool:
        """
        Check if a 1H position was entered on a previous trading day.

        Session EQUITY-51: 1H positions must exit before market close on the
        SAME day they were entered. If a position survives overnight (daemon
        restart, timing issue), it should be closed immediately.

        Args:
            entry_time: When the position was entered

        Returns:
            True if entry was on a previous trading day (stale position)
            False if entry was today (normal position)
        """
        now_et = datetime.now(self._timezone)

        # Make entry_time timezone-aware if needed
        if entry_time.tzinfo is None:
            entry_time_et = self._timezone.localize(entry_time)
        else:
            entry_time_et = entry_time.astimezone(self._timezone)

        # Same calendar day = not stale
        if entry_time_et.date() == now_et.date():
            return False

        # Different calendar day - check if it's a different TRADING day
        schedule = self._nyse_calendar.schedule(
            start_date=entry_time_et.date(),
            end_date=now_et.date()
        )

        # More than 1 trading day = stale (entry day + today = 2 days)
        if len(schedule) > 1:
            logger.warning(
                f"STALE 1H POSITION: Entered on {entry_time_et.date()}, "
                f"now {now_et.date()} - {len(schedule)} trading days span"
            )
            return True

        return False

    def _check_target_hit(self, pos: TrackedPosition) -> bool:
        """
        Check if underlying has reached target price.

        Args:
            pos: TrackedPosition to check

        Returns:
            True if target hit, False otherwise
        """
        if pos.direction.upper() in ['CALL', 'BULL', 'UP']:
            return pos.underlying_price >= pos.target_price
        else:
            return pos.underlying_price <= pos.target_price

    def _check_stop_hit(self, pos: TrackedPosition) -> bool:
        """
        Check if underlying has reached stop price.

        Args:
            pos: TrackedPosition to check

        Returns:
            True if stop hit, False otherwise
        """
        if pos.direction.upper() in ['CALL', 'BULL', 'UP']:
            return pos.underlying_price <= pos.stop_price
        else:
            return pos.underlying_price >= pos.stop_price

    def _check_pattern_invalidation(
        self,
        pos: TrackedPosition,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExitSignal]:
        """
        Check if entry bar evolved to Type 3 (pattern invalidation).

        Session EQUITY-44: Per STRAT methodology, if the entry bar breaks BOTH
        high and low (Type 3 evolution), exit immediately - the pattern premise
        is invalidated.

        Session EQUITY-48: Enhanced with REAL-TIME detection using intrabar
        high/low tracking. Detects Type 3 evolution as it happens using
        accumulated intrabar_high and intrabar_low since entry.

        Args:
            pos: TrackedPosition to check
            bar_data: Optional bar OHLCV data (fallback if intrabar tracking unavailable)

        Returns:
            ExitSignal if pattern invalidated, None otherwise
        """
        # Skip if not a Type 2 entry or missing setup bar data
        if pos.entry_bar_type not in ['2U', '2D']:
            return None

        if pos.entry_bar_high <= 0 or pos.entry_bar_low <= 0:
            return None

        # Use intrabar extremes for real-time detection
        intrabar_high = pos.intrabar_high
        intrabar_low = pos.intrabar_low

        # Fallback to bar data if intrabar tracking not initialized
        if intrabar_high <= 0 or intrabar_low == float('inf'):
            if bar_data:
                cache_high = bar_data.get('high', 0)
                cache_low = bar_data.get('low', float('inf'))
                if cache_high <= 0 or cache_low == float('inf'):
                    logger.debug(
                        f"Pattern invalidation skipped for {pos.symbol}: "
                        f"incomplete bar data (H={cache_high}, L={cache_low})"
                    )
                    return None
                intrabar_high = cache_high
                intrabar_low = cache_low
            else:
                logger.debug(
                    f"Pattern invalidation skipped for {pos.symbol}: "
                    f"no intrabar tracking and no bar data"
                )
                return None

        # Check for Type 3 evolution: broke BOTH setup bar high AND low
        broke_high = intrabar_high > pos.entry_bar_high
        broke_low = intrabar_low < pos.entry_bar_low

        if broke_high and broke_low:
            details = (
                f"Entry bar evolved to Type 3: "
                f"Setup H=${pos.entry_bar_high:.2f} L=${pos.entry_bar_low:.2f}, "
                f"Intrabar H=${intrabar_high:.2f} L=${intrabar_low:.2f}"
            )

            logger.warning(
                f"PATTERN INVALIDATED: {pos.signal_key} ({pos.osi_symbol}) - {details}"
            )

            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.PATTERN_INVALIDATED,
                underlying_price=pos.underlying_price or 0.0,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=details,
            )

        return None

    def _calculate_dte(self, expiration: str) -> int:
        """
        Calculate days to expiration.

        Args:
            expiration: Expiration date string (YYYY-MM-DD)

        Returns:
            Days until expiration (0 if expired or invalid)
        """
        if not expiration:
            return 0

        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d")
            return max(0, (exp_date - datetime.now()).days)
        except ValueError:
            return 0

    @property
    def config(self) -> MonitoringConfig:
        """Return current monitoring configuration."""
        return self._config

    def set_trailing_stop_checker(self, checker: TrailingStopChecker) -> None:
        """Set the trailing stop checker implementation."""
        self._trailing_stop_checker = checker

    def set_partial_exit_checker(self, checker: PartialExitChecker) -> None:
        """Set the partial exit checker implementation."""
        self._partial_exit_checker = checker
