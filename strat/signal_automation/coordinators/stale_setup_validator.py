"""
EQUITY-89: StaleSetupValidator - Extracted from SignalDaemon

Validates whether SETUP signals are still fresh or have become stale.

A setup becomes stale when the "forming bar" period ends without the
trigger being hit. Once a new bar closes, the pattern structure may
have evolved, making the original setup invalid.

Staleness windows by timeframe:
- 1H: 1.5 hours after setup_bar_timestamp
- 4H: 4 hours after setup_bar_timestamp
- 1D: 2 trading days (setup day + forming day)
- 1W: 2 weeks
- 1M: 2 months
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple
from dataclasses import dataclass

import pytz
import pandas_market_calendars as mcal

from strat.signal_automation.signal_store import StoredSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class StalenessConfig:
    """
    Configuration for staleness thresholds.

    Allows runtime customization of staleness windows per timeframe.
    """
    # Intraday staleness windows
    hourly_window_hours: float = 1.5   # 1H: 1.5 hours
    four_hour_window_hours: float = 4.0  # 4H: 4 hours

    # Higher timeframe staleness (in periods)
    daily_max_trading_days: int = 2    # 1D: 2 trading days
    weekly_max_weeks: int = 1          # 1W: 2 weeks (1 week diff = stale on week 3)
    monthly_max_months: int = 1        # 1M: 2 months (1 month diff = stale on month 3)


class StaleSetupValidator:
    """
    Validates whether SETUP signals are still fresh or have become stale.

    Extracted from SignalDaemon as part of EQUITY-89 Phase 3.2 refactoring.
    Uses Facade pattern - daemon delegates staleness checks to this validator.

    Session EQUITY-46: Fix for stale setup bug where setups from N days ago
    triggered when they should have been invalidated by intervening bars.

    A setup is only valid during the "forming bar" period. Once that bar closes:
    - If trigger was hit -> pattern COMPLETED (entry happened)
    - If trigger NOT hit -> pattern EVOLVED (new bar inserted, setup stale)

    Args:
        config: Optional StalenessConfig for customizing thresholds
        timezone: Timezone for time comparisons (default: America/New_York)
    """

    def __init__(
        self,
        config: StalenessConfig | None = None,
        timezone: str = 'America/New_York',
    ):
        self._config = config or StalenessConfig()
        self._timezone = pytz.timezone(timezone)
        self._nyse_calendar = mcal.get_calendar('NYSE')

    def is_setup_stale(self, signal: StoredSignal) -> Tuple[bool, str]:
        """
        Check if a SETUP signal is stale (new bar has closed since detection).

        Args:
            signal: The stored signal to check

        Returns:
            Tuple of (is_stale: bool, reason: str)
        """
        # Only check SETUP signals (COMPLETED are already validated)
        if signal.signal_type != SignalType.SETUP.value:
            return False, ""

        # Need setup_bar_timestamp to check staleness
        setup_ts = signal.setup_bar_timestamp
        if setup_ts is None:
            logger.warning(
                f"STALE CHECK SKIPPED: {signal.signal_key} has no setup_bar_timestamp"
            )
            return False, ""

        # Ensure timezone-aware comparison
        now = datetime.now(self._timezone)

        # Make setup_ts timezone-aware if it isn't
        if setup_ts.tzinfo is None:
            setup_ts = self._timezone.localize(setup_ts)

        # Delegate to timeframe-specific check
        timeframe = signal.timeframe

        if timeframe == '1H':
            return self._check_hourly_staleness(setup_ts, now)
        elif timeframe == '4H':
            return self._check_4h_staleness(setup_ts, now)
        elif timeframe == '1D':
            return self._check_daily_staleness(setup_ts, now)
        elif timeframe == '1W':
            return self._check_weekly_staleness(setup_ts, now)
        elif timeframe == '1M':
            return self._check_monthly_staleness(setup_ts, now)
        else:
            # Unknown timeframe - don't mark as stale
            logger.debug(f"STALE CHECK: Unknown timeframe {timeframe} - not marking as stale")
            return False, ""

    def _check_hourly_staleness(
        self,
        setup_ts: datetime,
        now: datetime
    ) -> Tuple[bool, str]:
        """
        Check staleness for 1H timeframe.

        Session EQUITY-65: Extended 1H staleness window.
        Allow 1.5 hours: covers the forming bar plus 30min buffer
        for triggers detected early in the next bar.

        Args:
            setup_ts: Setup bar timestamp (timezone-aware)
            now: Current time (timezone-aware)

        Returns:
            Tuple of (is_stale, reason)
        """
        hours = self._config.hourly_window_hours
        valid_until = setup_ts + timedelta(hours=hours)

        if now > valid_until:
            return True, (
                f"Hourly setup expired: detected {setup_ts.strftime('%H:%M')}, "
                f"now {now.strftime('%H:%M')}"
            )

        return False, ""

    def _check_4h_staleness(
        self,
        setup_ts: datetime,
        now: datetime
    ) -> Tuple[bool, str]:
        """
        Check staleness for 4H timeframe.

        Session EQUITY-65: Add 4H staleness handling.
        For 4H: setup is valid for 4 hours after setup_bar_timestamp.

        Args:
            setup_ts: Setup bar timestamp (timezone-aware)
            now: Current time (timezone-aware)

        Returns:
            Tuple of (is_stale, reason)
        """
        hours = self._config.four_hour_window_hours
        valid_until = setup_ts + timedelta(hours=hours)

        if now > valid_until:
            return True, (
                f"4H setup expired: detected {setup_ts.strftime('%H:%M')}, "
                f"now {now.strftime('%H:%M')}"
            )

        return False, ""

    def _check_daily_staleness(
        self,
        setup_ts: datetime,
        now: datetime
    ) -> Tuple[bool, str]:
        """
        Check staleness for 1D timeframe.

        For daily: setup is valid until end of NEXT trading day.
        If setup from Jan 5, valid during Jan 6, stale on Jan 7+.

        Uses NYSE calendar for accurate trading day calculation.

        Args:
            setup_ts: Setup bar timestamp (timezone-aware)
            now: Current time (timezone-aware)

        Returns:
            Tuple of (is_stale, reason)
        """
        # Get trading days from setup date to now
        schedule = self._nyse_calendar.schedule(
            start_date=setup_ts.date(),
            end_date=now.date()
        )

        max_days = self._config.daily_max_trading_days

        if len(schedule) > max_days:
            # More than 2 trading days (setup day + forming day + today)
            return True, (
                f"Daily setup expired: {len(schedule)} trading days since setup "
                f"({setup_ts.date()} to {now.date()})"
            )

        return False, ""

    def _check_weekly_staleness(
        self,
        setup_ts: datetime,
        now: datetime
    ) -> Tuple[bool, str]:
        """
        Check staleness for 1W timeframe.

        For weekly: setup is valid until end of NEXT week.

        Args:
            setup_ts: Setup bar timestamp (timezone-aware)
            now: Current time (timezone-aware)

        Returns:
            Tuple of (is_stale, reason)
        """
        # Calculate weeks difference
        setup_week = setup_ts.isocalendar()[1]
        setup_year = setup_ts.isocalendar()[0]
        now_week = now.isocalendar()[1]
        now_year = now.isocalendar()[0]

        # Simple week difference (handles year boundary)
        weeks_diff = (now_year - setup_year) * 52 + (now_week - setup_week)

        max_weeks = self._config.weekly_max_weeks

        if weeks_diff > max_weeks:
            return True, f"Weekly setup expired: {weeks_diff} weeks since setup"

        return False, ""

    def _check_monthly_staleness(
        self,
        setup_ts: datetime,
        now: datetime
    ) -> Tuple[bool, str]:
        """
        Check staleness for 1M timeframe.

        For monthly: setup is valid until end of NEXT month.

        Args:
            setup_ts: Setup bar timestamp (timezone-aware)
            now: Current time (timezone-aware)

        Returns:
            Tuple of (is_stale, reason)
        """
        # Calculate months difference
        setup_month = setup_ts.month
        setup_year = setup_ts.year
        now_month = now.month
        now_year = now.year

        months_diff = (now_year - setup_year) * 12 + (now_month - setup_month)

        max_months = self._config.monthly_max_months

        if months_diff > max_months:
            return True, f"Monthly setup expired: {months_diff} months since setup"

        return False, ""

    @property
    def config(self) -> StalenessConfig:
        """Return current staleness configuration."""
        return self._config

    @property
    def timezone(self) -> str:
        """Return configured timezone name."""
        return str(self._timezone)
