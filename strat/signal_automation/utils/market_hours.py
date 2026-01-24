# EQUITY-86: MarketHoursValidator - Shared NYSE market hours validation
# Phase 1.3 extraction - Consolidates duplicate market hours logic
"""
Market hours validation utility for NYSE trading hours.

Provides accurate market hours checking including:
- Holiday detection (NYSE holidays)
- Early close handling (Christmas Eve, day after Thanksgiving)
- Weekend detection
- Pre-market and after-hours detection

Session EQUITY-36: Original implementation added pandas_market_calendars
Session EQUITY-86: Extracted to shared utility (Phase 4 refactoring)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Tuple

import pandas_market_calendars as mcal
import pytz

logger = logging.getLogger(__name__)


@dataclass
class MarketSchedule:
    """Market schedule for a single trading day."""

    date: date
    is_trading_day: bool
    market_open: Optional[datetime] = None
    market_close: Optional[datetime] = None
    is_early_close: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date.isoformat(),
            'is_trading_day': self.is_trading_day,
            'market_open': self.market_open.isoformat() if self.market_open else None,
            'market_close': self.market_close.isoformat() if self.market_close else None,
            'is_early_close': self.is_early_close,
        }


@dataclass
class MarketHoursValidator:
    """
    NYSE market hours validator with holiday and early close support.

    Uses pandas_market_calendars for accurate NYSE schedule data.

    Args:
        timezone: Timezone for time comparisons (default: America/New_York)
        calendar_name: Market calendar name (default: NYSE)

    Example:
        validator = MarketHoursValidator()
        if validator.is_market_hours():
            # Execute trade logic
            pass
    """

    timezone: str = 'America/New_York'
    calendar_name: str = 'NYSE'
    _tz: pytz.BaseTzInfo = field(init=False, repr=False)
    _calendar: mcal.MarketCalendar = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize timezone and calendar objects."""
        self._tz = pytz.timezone(self.timezone)
        self._calendar = mcal.get_calendar(self.calendar_name)

    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if given time is within NYSE market hours.

        Args:
            dt: Datetime to check (default: current time)

        Returns:
            True if within market hours, False otherwise
        """
        if dt is None:
            dt = datetime.now(self._tz)
        elif dt.tzinfo is None:
            dt = self._tz.localize(dt)

        schedule = self.get_schedule(dt.date())

        if not schedule.is_trading_day:
            logger.debug(f"Market closed: {dt.date()} is not a trading day")
            return False

        # Type narrowing: when is_trading_day=True, open/close are guaranteed set
        market_open = schedule.market_open
        market_close = schedule.market_close
        assert market_open is not None and market_close is not None

        if dt < market_open:
            logger.debug(
                f"Market not yet open: opens at "
                f"{market_open.strftime('%H:%M')} ET"
            )
            return False

        if dt > market_close:
            logger.debug(
                f"Market closed: closed at "
                f"{market_close.strftime('%H:%M')} ET"
            )
            return False

        return True

    def is_trading_day(self, d: Optional[date] = None) -> bool:
        """
        Check if given date is a trading day.

        Args:
            d: Date to check (default: today)

        Returns:
            True if trading day, False if weekend/holiday
        """
        if d is None:
            d = datetime.now(self._tz).date()

        return self.get_schedule(d).is_trading_day

    def get_schedule(self, d: Optional[date] = None) -> MarketSchedule:
        """
        Get market schedule for a specific date.

        Args:
            d: Date to get schedule for (default: today)

        Returns:
            MarketSchedule with open/close times or is_trading_day=False
        """
        if d is None:
            d = datetime.now(self._tz).date()

        schedule_df = self._calendar.schedule(start_date=d, end_date=d)

        if schedule_df.empty:
            return MarketSchedule(date=d, is_trading_day=False)

        market_open = schedule_df.iloc[0]['market_open']
        market_close = schedule_df.iloc[0]['market_close']

        # Detect early close (normal close is 16:00 ET)
        close_hour = market_close.tz_convert(self._tz).hour
        is_early_close = close_hour < 16

        return MarketSchedule(
            date=d,
            is_trading_day=True,
            market_open=market_open.to_pydatetime(),
            market_close=market_close.to_pydatetime(),
            is_early_close=is_early_close,
        )

    def get_market_hours_today(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Get today's market open and close times.

        Returns:
            Tuple of (market_open, market_close) or None if not a trading day
        """
        schedule = self.get_schedule()
        if not schedule.is_trading_day:
            return None
        # Type narrowing: when is_trading_day=True, open/close are guaranteed set
        assert schedule.market_open is not None and schedule.market_close is not None
        return (schedule.market_open, schedule.market_close)

    def minutes_until_open(self, dt: Optional[datetime] = None) -> Optional[int]:
        """
        Calculate minutes until market opens.

        Args:
            dt: Datetime to calculate from (default: current time)

        Returns:
            Minutes until open, 0 if already open, None if not a trading day
        """
        if dt is None:
            dt = datetime.now(self._tz)
        elif dt.tzinfo is None:
            dt = self._tz.localize(dt)

        schedule = self.get_schedule(dt.date())

        if not schedule.is_trading_day:
            return None

        # Type narrowing: when is_trading_day=True, market_open is guaranteed set
        market_open = schedule.market_open
        assert market_open is not None

        if dt >= market_open:
            return 0

        delta = market_open - dt
        return int(delta.total_seconds() / 60)

    def minutes_until_close(self, dt: Optional[datetime] = None) -> Optional[int]:
        """
        Calculate minutes until market closes.

        Args:
            dt: Datetime to calculate from (default: current time)

        Returns:
            Minutes until close, 0 if already closed, None if not a trading day
        """
        if dt is None:
            dt = datetime.now(self._tz)
        elif dt.tzinfo is None:
            dt = self._tz.localize(dt)

        schedule = self.get_schedule(dt.date())

        if not schedule.is_trading_day:
            return None

        # Type narrowing: when is_trading_day=True, market_close is guaranteed set
        market_close = schedule.market_close
        assert market_close is not None

        if dt >= market_close:
            return 0

        delta = market_close - dt
        return int(delta.total_seconds() / 60)

    def is_near_close(
        self,
        minutes_threshold: int = 5,
        dt: Optional[datetime] = None
    ) -> bool:
        """
        Check if market is within N minutes of closing.

        Args:
            minutes_threshold: Minutes before close to trigger (default: 5)
            dt: Datetime to check (default: current time)

        Returns:
            True if within threshold minutes of close during market hours
        """
        if not self.is_market_hours(dt):
            return False

        minutes_left = self.minutes_until_close(dt)
        return minutes_left is not None and minutes_left <= minutes_threshold


# Module-level convenience function for simple use cases
_default_validator: Optional[MarketHoursValidator] = None


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if given time is within NYSE market hours.

    Convenience function using default validator instance.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if within market hours, False otherwise
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = MarketHoursValidator()
    return _default_validator.is_market_hours(dt)


def get_market_hours_validator(
    timezone: str = 'America/New_York',
    calendar_name: str = 'NYSE'
) -> MarketHoursValidator:
    """
    Factory function to create a MarketHoursValidator instance.

    Args:
        timezone: Timezone for time comparisons
        calendar_name: Market calendar name

    Returns:
        Configured MarketHoursValidator instance
    """
    return MarketHoursValidator(timezone=timezone, calendar_name=calendar_name)
