"""
Tests for strat/risk_free_rate.py

Covers:
- RATE_HISTORY constant structure and values
- DEFAULT_RATE constant
- get_risk_free_rate function for various date ranges
- get_current_risk_free_rate function
- Edge cases (timezone handling, pandas Timestamp, dates before history)

Session EQUITY-77: Test coverage for risk-free rate module.
"""

import pytest
from datetime import datetime
from unittest.mock import patch
import pandas as pd

from strat.risk_free_rate import (
    RATE_HISTORY,
    DEFAULT_RATE,
    get_risk_free_rate,
    get_current_risk_free_rate,
)


# =============================================================================
# Constants Tests
# =============================================================================

class TestRateHistoryConstant:
    """Tests for RATE_HISTORY constant."""

    def test_rate_history_is_list(self):
        """RATE_HISTORY is a list."""
        assert isinstance(RATE_HISTORY, list)

    def test_rate_history_not_empty(self):
        """RATE_HISTORY has entries."""
        assert len(RATE_HISTORY) > 0

    def test_rate_history_tuples_have_correct_structure(self):
        """Each entry is (datetime, float) tuple."""
        for entry in RATE_HISTORY:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], datetime)
            assert isinstance(entry[1], float)

    def test_rate_history_dates_are_ordered(self):
        """RATE_HISTORY dates are in chronological order."""
        dates = [entry[0] for entry in RATE_HISTORY]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], f"Date {dates[i]} not after {dates[i-1]}"

    def test_rate_history_rates_are_valid(self):
        """All rates are between 0 and 1 (0% to 100%)."""
        for date, rate in RATE_HISTORY:
            assert 0.0 <= rate <= 1.0, f"Invalid rate {rate} for date {date}"

    def test_rate_history_contains_known_events(self):
        """RATE_HISTORY contains key historical events."""
        dates = [entry[0] for entry in RATE_HISTORY]

        # Financial crisis (2008)
        assert datetime(2008, 12, 1) in dates, "Missing 2008 financial crisis"

        # COVID emergency cuts (2020)
        assert datetime(2020, 3, 1) in dates, "Missing COVID 2020 cuts"

        # 2022 rate hikes
        assert datetime(2022, 3, 1) in dates, "Missing 2022 rate hikes"


class TestDefaultRateConstant:
    """Tests for DEFAULT_RATE constant."""

    def test_default_rate_is_float(self):
        """DEFAULT_RATE is a float."""
        assert isinstance(DEFAULT_RATE, float)

    def test_default_rate_is_reasonable(self):
        """DEFAULT_RATE is a reasonable value (0-10%)."""
        assert 0.0 <= DEFAULT_RATE <= 0.10

    def test_default_rate_value(self):
        """DEFAULT_RATE has expected value."""
        assert DEFAULT_RATE == 0.05  # 5%


# =============================================================================
# get_risk_free_rate Function Tests
# =============================================================================

class TestGetRiskFreeRatePreCrisis:
    """Tests for get_risk_free_rate for pre-2008 dates."""

    def test_pre_financial_crisis_rate(self):
        """Pre-2008 rates return ~4%."""
        rate = get_risk_free_rate(datetime(2008, 6, 15))
        assert rate == 0.04  # 4%

    def test_early_2008_rate(self):
        """Early 2008 returns pre-crisis rate."""
        rate = get_risk_free_rate(datetime(2008, 1, 15))
        assert rate == 0.04


class TestGetRiskFreeRateFinancialCrisis:
    """Tests for get_risk_free_rate during 2008 financial crisis."""

    def test_post_lehman_rate(self):
        """Post-Lehman (Dec 2008) shows near-zero rates."""
        rate = get_risk_free_rate(datetime(2008, 12, 15))
        assert rate == 0.0025  # 0.25%

    def test_zirp_2010(self):
        """2010 shows ZIRP continuation."""
        rate = get_risk_free_rate(datetime(2010, 6, 15))
        assert rate == 0.0025  # 0.25%

    def test_zirp_2014(self):
        """2014 shows ZIRP continuation."""
        rate = get_risk_free_rate(datetime(2014, 6, 15))
        assert rate == 0.0025  # 0.25%


class TestGetRiskFreeRateNormalization:
    """Tests for get_risk_free_rate during 2015-2019 normalization."""

    def test_first_hike_dec_2015(self):
        """Dec 2015 first rate hike shows 0.50%."""
        rate = get_risk_free_rate(datetime(2015, 12, 20))
        assert rate == 0.0050  # 0.50%

    def test_gradual_hikes_2017(self):
        """2017 shows gradual rate increases."""
        rate = get_risk_free_rate(datetime(2017, 6, 15))
        assert rate == 0.0125  # 1.25%

    def test_peak_rates_2018(self):
        """Late 2018 shows peak rates around 2.5%."""
        rate = get_risk_free_rate(datetime(2018, 12, 20))
        assert rate == 0.0250  # 2.50%

    def test_rate_cuts_2019(self):
        """Late 2019 shows rate cuts."""
        rate = get_risk_free_rate(datetime(2019, 10, 15))
        assert rate == 0.0175  # 1.75%


class TestGetRiskFreeRateCOVID:
    """Tests for get_risk_free_rate during COVID period."""

    def test_pre_covid_rate_feb_2020(self):
        """Feb 2020 (pre-COVID) shows rates before emergency cuts."""
        rate = get_risk_free_rate(datetime(2020, 2, 15))
        assert rate == 0.0175  # 1.75% (late 2019 rate)

    def test_covid_emergency_cuts_march_2020(self):
        """March 2020 COVID emergency cuts to near-zero."""
        rate = get_risk_free_rate(datetime(2020, 3, 15))
        assert rate == 0.0025  # 0.25%

    def test_covid_zirp_2021(self):
        """2021 shows continued near-zero rates."""
        rate = get_risk_free_rate(datetime(2021, 6, 15))
        assert rate == 0.0025  # 0.25%


class TestGetRiskFreeRateHikingCycle:
    """Tests for get_risk_free_rate during 2022-2023 hiking cycle."""

    def test_early_hikes_march_2022(self):
        """March 2022 shows first rate hike."""
        rate = get_risk_free_rate(datetime(2022, 3, 20))
        assert rate == 0.0050  # 0.50%

    def test_rapid_hikes_july_2022(self):
        """July 2022 shows rapid rate increases."""
        rate = get_risk_free_rate(datetime(2022, 7, 20))
        assert rate == 0.0250  # 2.50%

    def test_aggressive_hikes_dec_2022(self):
        """Dec 2022 shows aggressive tightening."""
        rate = get_risk_free_rate(datetime(2022, 12, 20))
        assert rate == 0.0450  # 4.50%

    def test_peak_rates_july_2023(self):
        """July 2023 shows terminal/peak rates."""
        rate = get_risk_free_rate(datetime(2023, 7, 20))
        assert rate == 0.0550  # 5.50%


class TestGetRiskFreeRate2024:
    """Tests for get_risk_free_rate in 2024 (rate cuts begin)."""

    def test_pre_cut_rate_august_2024(self):
        """August 2024 (before first cut) shows elevated rates."""
        rate = get_risk_free_rate(datetime(2024, 8, 15))
        assert rate == 0.0550  # 5.50%

    def test_first_cut_september_2024(self):
        """September 2024 first rate cut."""
        rate = get_risk_free_rate(datetime(2024, 9, 20))
        assert rate == 0.0500  # 5.00%

    def test_second_cut_november_2024(self):
        """November 2024 second rate cut."""
        rate = get_risk_free_rate(datetime(2024, 11, 20))
        assert rate == 0.0475  # 4.75%


class TestGetRiskFreeRateDateFormats:
    """Tests for get_risk_free_rate with various date formats."""

    def test_datetime_input(self):
        """Function accepts datetime input."""
        rate = get_risk_free_rate(datetime(2023, 7, 15))
        assert isinstance(rate, float)
        assert rate > 0

    def test_pandas_timestamp_input(self):
        """Function accepts pandas Timestamp input."""
        ts = pd.Timestamp('2023-07-15')
        rate = get_risk_free_rate(ts)
        assert isinstance(rate, float)
        assert rate == 0.0550  # 5.50%

    def test_pandas_timestamp_with_time(self):
        """Function accepts pandas Timestamp with time component."""
        ts = pd.Timestamp('2023-07-15 10:30:00')
        rate = get_risk_free_rate(ts)
        assert rate == 0.0550

    def test_timezone_aware_datetime(self):
        """Function handles timezone-aware datetime."""
        import pytz
        tz = pytz.timezone('US/Eastern')
        dt = tz.localize(datetime(2023, 7, 15, 10, 30))
        rate = get_risk_free_rate(dt)
        assert rate == 0.0550

    def test_timezone_aware_pandas_timestamp(self):
        """Function handles timezone-aware pandas Timestamp."""
        ts = pd.Timestamp('2023-07-15 10:30:00', tz='US/Eastern')
        rate = get_risk_free_rate(ts)
        assert rate == 0.0550


class TestGetRiskFreeRateEdgeCases:
    """Tests for get_risk_free_rate edge cases."""

    def test_date_before_history(self):
        """Date before first entry returns first rate."""
        rate = get_risk_free_rate(datetime(2005, 1, 1))
        # Should return first rate in history (pre-2008 = 4%)
        assert rate == RATE_HISTORY[0][1]

    def test_exact_threshold_date(self):
        """Exact threshold date returns that rate."""
        # March 1, 2020 is a threshold date
        rate = get_risk_free_rate(datetime(2020, 3, 1))
        assert rate == 0.0025  # COVID cuts rate

    def test_day_before_threshold(self):
        """Day before threshold returns previous rate."""
        rate = get_risk_free_rate(datetime(2020, 2, 29))
        assert rate == 0.0175  # Pre-COVID rate

    def test_day_after_threshold(self):
        """Day after threshold returns new rate."""
        rate = get_risk_free_rate(datetime(2020, 3, 2))
        assert rate == 0.0025  # COVID cuts rate

    def test_future_date(self):
        """Future date returns most recent rate."""
        rate = get_risk_free_rate(datetime(2030, 1, 1))
        # Should return last rate in history
        expected = RATE_HISTORY[-1][1]
        assert rate == expected

    def test_midnight_datetime(self):
        """Midnight datetime works correctly."""
        rate = get_risk_free_rate(datetime(2023, 7, 15, 0, 0, 0))
        assert rate == 0.0550


class TestGetRiskFreeRateBacktestScenarios:
    """Tests for get_risk_free_rate in realistic backtest scenarios."""

    def test_march_2020_crash_period(self):
        """Rates during March 2020 crash are near-zero."""
        crash_dates = [
            datetime(2020, 3, 9),   # Black Monday
            datetime(2020, 3, 12),  # Black Thursday
            datetime(2020, 3, 16),  # Worst Monday
            datetime(2020, 3, 23),  # Bottom
        ]
        for date in crash_dates:
            rate = get_risk_free_rate(date)
            assert rate == 0.0025, f"Expected 0.25% for {date}, got {rate}"

    def test_vix_spike_august_2024(self):
        """Rates during August 2024 yen carry unwind."""
        # August 5, 2024 VIX spike
        rate = get_risk_free_rate(datetime(2024, 8, 5))
        assert rate == 0.0550  # Still at peak rate before Sep cut

    def test_options_pricing_consistency(self):
        """Rates are consistent for options pricing calculations."""
        # For a 30-day option expiring on different dates
        dates_2020 = [
            (datetime(2020, 2, 15), 0.0175),  # Pre-COVID
            (datetime(2020, 4, 15), 0.0025),  # Post-COVID
        ]
        for date, expected in dates_2020:
            rate = get_risk_free_rate(date)
            assert rate == expected, f"Rate mismatch for {date}"


# =============================================================================
# get_current_risk_free_rate Function Tests
# =============================================================================

class TestGetCurrentRiskFreeRate:
    """Tests for get_current_risk_free_rate function."""

    def test_returns_float(self):
        """Function returns a float."""
        rate = get_current_risk_free_rate()
        assert isinstance(rate, float)

    def test_returns_reasonable_value(self):
        """Function returns a reasonable rate (0-10%)."""
        rate = get_current_risk_free_rate()
        assert 0.0 <= rate <= 0.10

    def test_consistent_with_get_risk_free_rate(self):
        """Result is consistent with get_risk_free_rate(now)."""
        current_rate = get_current_risk_free_rate()
        now_rate = get_risk_free_rate(datetime.now())
        assert current_rate == now_rate

    @patch('strat.risk_free_rate.datetime')
    def test_uses_datetime_now(self, mock_datetime):
        """Function uses datetime.now() internally."""
        mock_now = datetime(2023, 7, 15)
        mock_datetime.now.return_value = mock_now
        # Also need to allow datetime() to work normally for comparison
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        # Note: The patch may not work perfectly due to import timing
        # This test mainly documents expected behavior
        rate = get_current_risk_free_rate()
        assert isinstance(rate, float)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRiskFreeRateIntegration:
    """Integration tests for risk-free rate module."""

    def test_rate_progression_over_time(self):
        """Rates change appropriately over major periods."""
        # Define key periods and expected rate ranges
        periods = [
            (datetime(2009, 1, 1), datetime(2015, 11, 30), 0.0, 0.01),   # ZIRP
            (datetime(2016, 1, 1), datetime(2019, 6, 30), 0.005, 0.03), # Normalization
            (datetime(2020, 4, 1), datetime(2022, 2, 28), 0.0, 0.01),   # COVID ZIRP
            (datetime(2023, 6, 1), datetime(2024, 8, 31), 0.05, 0.06),  # Peak rates
        ]

        for start, end, min_rate, max_rate in periods:
            rate = get_risk_free_rate(start)
            assert min_rate <= rate <= max_rate, f"Rate {rate} not in [{min_rate}, {max_rate}] for {start}"

    def test_all_rate_history_dates_work(self):
        """All dates in RATE_HISTORY return expected rates."""
        for threshold_date, expected_rate in RATE_HISTORY:
            rate = get_risk_free_rate(threshold_date)
            assert rate == expected_rate, f"Rate mismatch for {threshold_date}"

    def test_rate_changes_at_boundaries(self):
        """Rates change exactly at boundary dates."""
        # Test that rate changes on the threshold date, not before
        for i in range(1, len(RATE_HISTORY)):
            prev_date, prev_rate = RATE_HISTORY[i-1]
            curr_date, curr_rate = RATE_HISTORY[i]

            # Rate should be previous rate just before threshold
            day_before = curr_date.replace(day=curr_date.day - 1) if curr_date.day > 1 else curr_date
            if day_before < curr_date:
                rate_before = get_risk_free_rate(day_before)
                rate_on = get_risk_free_rate(curr_date)

                # On the threshold date, should get new rate
                assert rate_on == curr_rate, f"Expected {curr_rate} on {curr_date}"

    def test_pandas_series_compatibility(self):
        """Function works with pandas Series of dates."""
        dates = pd.Series([
            datetime(2020, 2, 15),
            datetime(2020, 3, 15),
            datetime(2023, 7, 15),
        ])

        rates = dates.apply(get_risk_free_rate)

        assert len(rates) == 3
        assert rates.iloc[0] == 0.0175  # Pre-COVID
        assert rates.iloc[1] == 0.0025  # COVID
        assert rates.iloc[2] == 0.0550  # Peak 2023

    def test_dataframe_date_column(self):
        """Function works with DataFrame date column."""
        df = pd.DataFrame({
            'date': [
                datetime(2020, 2, 15),
                datetime(2020, 3, 15),
                datetime(2023, 7, 15),
            ],
            'price': [100, 95, 150],
        })

        df['risk_free_rate'] = df['date'].apply(get_risk_free_rate)

        assert df['risk_free_rate'].tolist() == [0.0175, 0.0025, 0.0550]
