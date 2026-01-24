"""
EQUITY-86: Tests for MarketHoursValidator utility.

Tests NYSE market hours validation including:
- Holiday detection
- Early close handling
- Weekend detection
- Pre-market/post-market detection
- Timezone handling
"""

import pytest
from datetime import datetime, date
from unittest.mock import patch
import pytz

from strat.signal_automation.utils.market_hours import (
    MarketHoursValidator,
    MarketSchedule,
    is_market_hours,
    get_market_hours_validator,
)


class TestMarketSchedule:
    """Tests for MarketSchedule dataclass."""

    def test_non_trading_day(self):
        """MarketSchedule for non-trading day has correct defaults."""
        schedule = MarketSchedule(date=date(2026, 1, 1), is_trading_day=False)

        assert schedule.date == date(2026, 1, 1)
        assert schedule.is_trading_day is False
        assert schedule.market_open is None
        assert schedule.market_close is None
        assert schedule.is_early_close is False

    def test_trading_day(self):
        """MarketSchedule for trading day has open/close times."""
        et = pytz.timezone('America/New_York')
        market_open = et.localize(datetime(2026, 1, 5, 9, 30))
        market_close = et.localize(datetime(2026, 1, 5, 16, 0))

        schedule = MarketSchedule(
            date=date(2026, 1, 5),
            is_trading_day=True,
            market_open=market_open,
            market_close=market_close,
        )

        assert schedule.is_trading_day is True
        assert schedule.market_open == market_open
        assert schedule.market_close == market_close
        assert schedule.is_early_close is False

    def test_early_close_day(self):
        """MarketSchedule detects early close days."""
        et = pytz.timezone('America/New_York')
        # Christmas Eve typically closes at 1 PM
        market_open = et.localize(datetime(2025, 12, 24, 9, 30))
        market_close = et.localize(datetime(2025, 12, 24, 13, 0))

        schedule = MarketSchedule(
            date=date(2025, 12, 24),
            is_trading_day=True,
            market_open=market_open,
            market_close=market_close,
            is_early_close=True,
        )

        assert schedule.is_early_close is True

    def test_to_dict(self):
        """MarketSchedule serializes to dictionary."""
        schedule = MarketSchedule(date=date(2026, 1, 5), is_trading_day=False)
        result = schedule.to_dict()

        assert result['date'] == '2026-01-05'
        assert result['is_trading_day'] is False
        assert result['market_open'] is None
        assert result['market_close'] is None
        assert result['is_early_close'] is False

    def test_to_dict_with_times(self):
        """MarketSchedule serializes times in ISO format."""
        et = pytz.timezone('America/New_York')
        market_open = et.localize(datetime(2026, 1, 5, 9, 30))
        market_close = et.localize(datetime(2026, 1, 5, 16, 0))

        schedule = MarketSchedule(
            date=date(2026, 1, 5),
            is_trading_day=True,
            market_open=market_open,
            market_close=market_close,
        )
        result = schedule.to_dict()

        assert 'T09:30:00' in result['market_open']
        assert 'T16:00:00' in result['market_close']


class TestMarketHoursValidatorInit:
    """Tests for MarketHoursValidator initialization."""

    def test_default_init(self):
        """MarketHoursValidator initializes with defaults."""
        validator = MarketHoursValidator()

        assert validator.timezone == 'America/New_York'
        assert validator.calendar_name == 'NYSE'

    def test_custom_timezone(self):
        """MarketHoursValidator accepts custom timezone."""
        validator = MarketHoursValidator(timezone='US/Eastern')

        assert validator.timezone == 'US/Eastern'

    def test_invalid_timezone_raises(self):
        """MarketHoursValidator raises for invalid timezone."""
        with pytest.raises(Exception):  # pytz.UnknownTimeZoneError
            MarketHoursValidator(timezone='Invalid/Zone')


class TestIsMarketHours:
    """Tests for is_market_hours() method."""

    def test_during_market_hours(self):
        """is_market_hours returns True during regular hours."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 10:30 AM ET on a Monday (normal trading day)
        test_time = et.localize(datetime(2026, 1, 5, 10, 30))

        result = validator.is_market_hours(test_time)

        assert result is True

    def test_before_market_open(self):
        """is_market_hours returns False before market open."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 8:00 AM ET - before market open
        test_time = et.localize(datetime(2026, 1, 5, 8, 0))

        result = validator.is_market_hours(test_time)

        assert result is False

    def test_after_market_close(self):
        """is_market_hours returns False after market close."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 5:00 PM ET - after market close
        test_time = et.localize(datetime(2026, 1, 5, 17, 0))

        result = validator.is_market_hours(test_time)

        assert result is False

    def test_exactly_at_open(self):
        """is_market_hours returns True exactly at market open."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 9:30 AM ET exactly
        test_time = et.localize(datetime(2026, 1, 5, 9, 30))

        result = validator.is_market_hours(test_time)

        assert result is True

    def test_exactly_at_close(self):
        """is_market_hours returns True at exactly market close."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 4:00 PM ET exactly (16:00)
        test_time = et.localize(datetime(2026, 1, 5, 16, 0))

        result = validator.is_market_hours(test_time)

        assert result is True

    def test_weekend_saturday(self):
        """is_market_hours returns False on Saturday."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # Saturday
        test_time = et.localize(datetime(2026, 1, 3, 10, 30))

        result = validator.is_market_hours(test_time)

        assert result is False

    def test_weekend_sunday(self):
        """is_market_hours returns False on Sunday."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # Sunday
        test_time = et.localize(datetime(2026, 1, 4, 10, 30))

        result = validator.is_market_hours(test_time)

        assert result is False

    def test_naive_datetime_localized(self):
        """is_market_hours localizes naive datetime."""
        validator = MarketHoursValidator()
        # Naive datetime during market hours
        test_time = datetime(2026, 1, 5, 10, 30)

        result = validator.is_market_hours(test_time)

        assert result is True

    def test_none_uses_current_time(self):
        """is_market_hours uses current time when None passed."""
        validator = MarketHoursValidator()
        # Result depends on when test runs, but should not raise
        result = validator.is_market_hours(None)
        assert isinstance(result, bool)


class TestIsTradingDay:
    """Tests for is_trading_day() method."""

    def test_normal_weekday(self):
        """is_trading_day returns True for normal weekday."""
        validator = MarketHoursValidator()
        # Monday
        result = validator.is_trading_day(date(2026, 1, 5))

        assert result is True

    def test_saturday(self):
        """is_trading_day returns False for Saturday."""
        validator = MarketHoursValidator()
        result = validator.is_trading_day(date(2026, 1, 3))

        assert result is False

    def test_sunday(self):
        """is_trading_day returns False for Sunday."""
        validator = MarketHoursValidator()
        result = validator.is_trading_day(date(2026, 1, 4))

        assert result is False

    def test_new_years_day(self):
        """is_trading_day returns False for New Year's Day."""
        validator = MarketHoursValidator()
        # January 1, 2026 is a Thursday
        result = validator.is_trading_day(date(2026, 1, 1))

        assert result is False

    def test_christmas_day(self):
        """is_trading_day returns False for Christmas Day."""
        validator = MarketHoursValidator()
        # December 25, 2025 is a Thursday
        result = validator.is_trading_day(date(2025, 12, 25))

        assert result is False


class TestGetSchedule:
    """Tests for get_schedule() method."""

    def test_trading_day_schedule(self):
        """get_schedule returns full schedule for trading day."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        schedule = validator.get_schedule(date(2026, 1, 5))

        assert schedule.is_trading_day is True
        assert schedule.market_open is not None
        assert schedule.market_close is not None
        # Convert to ET for hour comparison (returned in UTC)
        open_et = schedule.market_open.astimezone(et)
        assert open_et.hour == 9
        assert open_et.minute == 30

    def test_non_trading_day_schedule(self):
        """get_schedule returns minimal schedule for non-trading day."""
        validator = MarketHoursValidator()
        schedule = validator.get_schedule(date(2026, 1, 1))  # New Year's

        assert schedule.is_trading_day is False
        assert schedule.market_open is None
        assert schedule.market_close is None

    def test_early_close_detection(self):
        """get_schedule detects early close days."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # Day after Thanksgiving 2025 - early close at 1 PM
        # Nov 28, 2025 is Friday after Thanksgiving
        schedule = validator.get_schedule(date(2025, 11, 28))

        assert schedule.is_trading_day is True
        assert schedule.is_early_close is True
        # Convert to ET for hour comparison (returned in UTC)
        close_et = schedule.market_close.astimezone(et)
        assert close_et.hour == 13


class TestGetMarketHoursToday:
    """Tests for get_market_hours_today() method."""

    def test_trading_day_returns_tuple(self):
        """get_market_hours_today returns tuple on trading day."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')

        # Mock get_schedule to return a trading day
        with patch.object(validator, 'get_schedule') as mock_schedule:
            mock_schedule.return_value = MarketSchedule(
                date=date(2026, 1, 5),
                is_trading_day=True,
                market_open=et.localize(datetime(2026, 1, 5, 9, 30)),
                market_close=et.localize(datetime(2026, 1, 5, 16, 0)),
            )

            result = validator.get_market_hours_today()

            assert result is not None
            assert len(result) == 2
            assert result[0].hour == 9
            assert result[0].minute == 30
            assert result[1].hour == 16

    def test_non_trading_day_returns_none(self):
        """get_market_hours_today returns None on non-trading day."""
        validator = MarketHoursValidator()

        with patch.object(validator, 'get_schedule') as mock_schedule:
            mock_schedule.return_value = MarketSchedule(
                date=date(2026, 1, 1),
                is_trading_day=False,
            )

            result = validator.get_market_hours_today()

            assert result is None


class TestMinutesUntilOpen:
    """Tests for minutes_until_open() method."""

    def test_before_open(self):
        """minutes_until_open returns positive minutes before market open."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 9:00 AM - 30 minutes before open
        test_time = et.localize(datetime(2026, 1, 5, 9, 0))

        result = validator.minutes_until_open(test_time)

        assert result == 30

    def test_after_open(self):
        """minutes_until_open returns 0 after market is open."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 10:00 AM - 30 minutes after open
        test_time = et.localize(datetime(2026, 1, 5, 10, 0))

        result = validator.minutes_until_open(test_time)

        assert result == 0

    def test_non_trading_day(self):
        """minutes_until_open returns None on non-trading day."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # Saturday
        test_time = et.localize(datetime(2026, 1, 3, 9, 0))

        result = validator.minutes_until_open(test_time)

        assert result is None


class TestMinutesUntilClose:
    """Tests for minutes_until_close() method."""

    def test_during_market_hours(self):
        """minutes_until_close returns positive minutes during market."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 3:00 PM - 60 minutes before close
        test_time = et.localize(datetime(2026, 1, 5, 15, 0))

        result = validator.minutes_until_close(test_time)

        assert result == 60

    def test_after_close(self):
        """minutes_until_close returns 0 after market is closed."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 5:00 PM - after close
        test_time = et.localize(datetime(2026, 1, 5, 17, 0))

        result = validator.minutes_until_close(test_time)

        assert result == 0

    def test_non_trading_day(self):
        """minutes_until_close returns None on non-trading day."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # Sunday
        test_time = et.localize(datetime(2026, 1, 4, 10, 0))

        result = validator.minutes_until_close(test_time)

        assert result is None


class TestIsNearClose:
    """Tests for is_near_close() method."""

    def test_near_close_within_threshold(self):
        """is_near_close returns True when within threshold."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 3:57 PM - 3 minutes before close (within 5 min default)
        test_time = et.localize(datetime(2026, 1, 5, 15, 57))

        result = validator.is_near_close(minutes_threshold=5, dt=test_time)

        assert result is True

    def test_near_close_outside_threshold(self):
        """is_near_close returns False when outside threshold."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 3:00 PM - 60 minutes before close
        test_time = et.localize(datetime(2026, 1, 5, 15, 0))

        result = validator.is_near_close(minutes_threshold=5, dt=test_time)

        assert result is False

    def test_near_close_custom_threshold(self):
        """is_near_close respects custom threshold."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 3:45 PM - 15 minutes before close
        test_time = et.localize(datetime(2026, 1, 5, 15, 45))

        result_5min = validator.is_near_close(minutes_threshold=5, dt=test_time)
        result_30min = validator.is_near_close(minutes_threshold=30, dt=test_time)

        assert result_5min is False
        assert result_30min is True

    def test_near_close_outside_market_hours(self):
        """is_near_close returns False outside market hours."""
        validator = MarketHoursValidator()
        et = pytz.timezone('America/New_York')
        # 8:00 AM - before market open
        test_time = et.localize(datetime(2026, 1, 5, 8, 0))

        result = validator.is_near_close(dt=test_time)

        assert result is False


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_is_market_hours_function(self):
        """Module-level is_market_hours works correctly."""
        et = pytz.timezone('America/New_York')
        # During market hours
        test_time = et.localize(datetime(2026, 1, 5, 10, 30))

        result = is_market_hours(test_time)

        assert result is True

    def test_is_market_hours_function_weekend(self):
        """Module-level is_market_hours returns False on weekend."""
        et = pytz.timezone('America/New_York')
        # Saturday
        test_time = et.localize(datetime(2026, 1, 3, 10, 30))

        result = is_market_hours(test_time)

        assert result is False

    def test_get_market_hours_validator_factory(self):
        """get_market_hours_validator returns configured instance."""
        validator = get_market_hours_validator(
            timezone='US/Eastern',
            calendar_name='NYSE',
        )

        assert isinstance(validator, MarketHoursValidator)
        assert validator.timezone == 'US/Eastern'
        assert validator.calendar_name == 'NYSE'

    def test_get_market_hours_validator_defaults(self):
        """get_market_hours_validator uses defaults."""
        validator = get_market_hours_validator()

        assert validator.timezone == 'America/New_York'
        assert validator.calendar_name == 'NYSE'
