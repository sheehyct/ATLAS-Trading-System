"""
Tests for STRAT Options Module - Session EQUITY-69

Tests cover:
1. OptionType and OptionStrategy enums
2. OptionContract dataclass and OSI symbol generation
3. OptionTrade dataclass and risk/reward calculation
4. OptionsExecutor helper methods
5. Strike selection and rounding
6. Hourly time filter (STRAT "Let the Market Breathe")
7. Magnitude filter (Session 83K-31)
8. DTE and expiration calculation

Per STRAT methodology:
- Strike within entry-to-target range
- Delta targeting (0.45-0.65 optimal)
- Theta cost verification
- Hourly: 2-bar at 10:30, 3-bar at 11:30, exit by 15:30
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from strat.options_module import (
    OptionType,
    OptionStrategy,
    OptionContract,
    OptionTrade,
    OptionsExecutor,
    _convert_to_naive_et,
)
from strat.tier1_detector import PatternSignal, PatternType, Timeframe


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_call_signal():
    """Create sample bullish pattern signal."""
    return PatternSignal(
        timestamp=datetime(2026, 1, 15, 10, 30),
        pattern_type=PatternType.PATTERN_32_UP,
        timeframe=Timeframe.DAILY,
        direction=1,  # Bullish
        entry_price=500.0,
        stop_price=495.0,
        target_price=510.0,
        risk_reward=2.0,
    )


@pytest.fixture
def sample_put_signal():
    """Create sample bearish pattern signal."""
    return PatternSignal(
        timestamp=datetime(2026, 1, 15, 10, 30),
        pattern_type=PatternType.PATTERN_32_DOWN,
        timeframe=Timeframe.DAILY,
        direction=-1,  # Bearish
        entry_price=500.0,
        stop_price=505.0,
        target_price=490.0,
        risk_reward=2.0,
    )


@pytest.fixture
def hourly_signal():
    """Create hourly pattern signal for time filter tests."""
    return PatternSignal(
        timestamp=datetime(2026, 1, 15, 11, 30),
        pattern_type=PatternType.PATTERN_32_UP,
        timeframe=Timeframe.HOURLY,
        direction=1,
        entry_price=500.0,
        stop_price=498.0,
        target_price=504.0,
        risk_reward=2.0,
    )


@pytest.fixture
def executor():
    """Create OptionsExecutor with mocked credentials."""
    with patch('strat.options_module.get_alpaca_credentials', return_value={}):
        return OptionsExecutor(account='MID')


# =============================================================================
# Test Enums
# =============================================================================

class TestOptionTypeEnum:
    """Tests for OptionType enum."""

    def test_call_value(self):
        """Test CALL enum value."""
        assert OptionType.CALL.value == "C"

    def test_put_value(self):
        """Test PUT enum value."""
        assert OptionType.PUT.value == "P"


class TestOptionStrategyEnum:
    """Tests for OptionStrategy enum."""

    def test_long_call_value(self):
        """Test LONG_CALL strategy."""
        assert OptionStrategy.LONG_CALL.value == "long_call"

    def test_long_put_value(self):
        """Test LONG_PUT strategy."""
        assert OptionStrategy.LONG_PUT.value == "long_put"

    def test_spread_strategies(self):
        """Test debit spread strategies."""
        assert OptionStrategy.CALL_DEBIT_SPREAD.value == "call_debit_spread"
        assert OptionStrategy.PUT_DEBIT_SPREAD.value == "put_debit_spread"


# =============================================================================
# Test OptionContract Dataclass
# =============================================================================

class TestOptionContract:
    """Tests for OptionContract dataclass."""

    def test_basic_creation(self):
        """Test basic contract creation."""
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2026, 2, 20),
            option_type=OptionType.CALL,
            strike=500.0,
        )

        assert contract.underlying == 'SPY'
        assert contract.strike == 500.0
        assert contract.option_type == OptionType.CALL

    def test_osi_symbol_generation_call(self):
        """Test OSI symbol generation for CALL."""
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2026, 12, 20),
            option_type=OptionType.CALL,
            strike=500.0,
        )

        # Format: SYMBOL + YYMMDD + C/P + STRIKE*1000
        assert contract.osi_symbol == 'SPY261220C00500000'

    def test_osi_symbol_generation_put(self):
        """Test OSI symbol generation for PUT."""
        contract = OptionContract(
            underlying='QQQ',
            expiration=datetime(2026, 3, 15),
            option_type=OptionType.PUT,
            strike=420.0,
        )

        assert contract.osi_symbol == 'QQQ260315P00420000'

    def test_osi_symbol_fractional_strike(self):
        """Test OSI symbol with fractional strike price."""
        contract = OptionContract(
            underlying='IWM',
            expiration=datetime(2026, 1, 17),
            option_type=OptionType.CALL,
            strike=230.50,  # Fractional strike
        )

        # 230.50 * 1000 = 230500
        assert contract.osi_symbol == 'IWM260117C00230500'

    def test_osi_symbol_short_underlying(self):
        """Test OSI symbol with short underlying symbol."""
        contract = OptionContract(
            underlying='F',  # Ford - single letter
            expiration=datetime(2026, 6, 20),
            option_type=OptionType.PUT,
            strike=12.0,
        )

        # Short symbols should still work
        assert 'F' in contract.osi_symbol
        assert '260620' in contract.osi_symbol
        assert 'P' in contract.osi_symbol

    def test_generate_osi_symbol_method(self):
        """Test explicit call to generate_osi_symbol."""
        contract = OptionContract(
            underlying='AAPL',
            expiration=datetime(2026, 4, 17),
            option_type=OptionType.CALL,
            strike=185.0,
            osi_symbol='',  # Empty to test generation
        )

        # Should auto-generate in __post_init__
        assert contract.osi_symbol != ''
        assert 'AAPL' in contract.osi_symbol


# =============================================================================
# Test OptionTrade Dataclass
# =============================================================================

class TestOptionTrade:
    """Tests for OptionTrade dataclass."""

    def test_basic_creation(self, sample_call_signal):
        """Test basic trade creation."""
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2026, 2, 20),
            option_type=OptionType.CALL,
            strike=505.0,
        )

        trade = OptionTrade(
            pattern_signal=sample_call_signal,
            contract=contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=500.0,
            target_exit=510.0,
            stop_exit=495.0,
            quantity=2,
        )

        assert trade.entry_trigger == 500.0
        assert trade.target_exit == 510.0
        assert trade.stop_exit == 495.0
        assert trade.quantity == 2

    def test_calculate_risk_reward(self, sample_call_signal):
        """Test risk/reward calculation."""
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2026, 2, 20),
            option_type=OptionType.CALL,
            strike=505.0,
        )

        trade = OptionTrade(
            pattern_signal=sample_call_signal,
            contract=contract,
            strategy=OptionStrategy.LONG_CALL,
            entry_trigger=500.0,
            target_exit=510.0,
            stop_exit=495.0,
            quantity=2,
        )

        # Calculate with $5.00 premium
        trade.calculate_risk_reward(premium=5.0)

        # Max risk = premium * 100 * quantity = 5 * 100 * 2 = $1000
        assert trade.max_risk == 1000.0
        assert trade.option_premium == 5.0
        # Expected reward = max_risk * risk_reward = 1000 * 2.0 = $2000
        assert trade.expected_reward == 2000.0


# =============================================================================
# Test Strike Rounding
# =============================================================================

class TestStrikeRounding:
    """Tests for strike rounding to standard intervals."""

    def test_round_under_100_uses_1_intervals(self, executor):
        """Test that strikes under $100 use $1 intervals."""
        # Raw strike 45.3 should round to 45.0
        result = executor._round_to_standard_strike(45.3, 50.0)
        assert result == 45.0

        result = executor._round_to_standard_strike(45.7, 50.0)
        assert result == 46.0

    def test_round_100_to_500_uses_5_intervals(self, executor):
        """Test that strikes $100-$500 use $5 intervals."""
        # Raw strike 232.3 should round to 230.0
        result = executor._round_to_standard_strike(232.3, 250.0)
        assert result == 230.0

        result = executor._round_to_standard_strike(233.5, 250.0)
        assert result == 235.0

    def test_round_over_500_uses_10_intervals(self, executor):
        """Test that strikes over $500 use $10 intervals."""
        # Raw strike 503.5 should round to 500.0
        result = executor._round_to_standard_strike(503.5, 550.0)
        assert result == 500.0

        result = executor._round_to_standard_strike(508.5, 550.0)
        assert result == 510.0


# =============================================================================
# Test Candidate Strike Generation
# =============================================================================

class TestCandidateStrikeGeneration:
    """Tests for generating candidate strikes."""

    def test_generate_candidates_basic(self, executor):
        """Test basic candidate generation."""
        candidates = executor._generate_candidate_strikes(
            strike_min=495.0,
            strike_max=510.0,
            interval=5.0
        )

        assert 495.0 in candidates
        assert 500.0 in candidates
        assert 505.0 in candidates
        assert 510.0 in candidates
        assert len(candidates) == 4

    def test_generate_candidates_empty_range(self, executor):
        """Test empty result when min >= max."""
        candidates = executor._generate_candidate_strikes(
            strike_min=510.0,
            strike_max=495.0,
            interval=5.0
        )
        assert candidates == []

    def test_generate_candidates_narrow_range(self, executor):
        """Test narrow range with single candidate."""
        candidates = executor._generate_candidate_strikes(
            strike_min=500.0,
            strike_max=502.0,
            interval=5.0
        )
        # Only 500 fits in [500, 502] at $5 intervals
        assert candidates == [500.0]


# =============================================================================
# Test Hourly Time Filter (STRAT "Let the Market Breathe")
# =============================================================================

class TestHourlyTimeFilter:
    """Tests for hourly time filter per STRAT methodology."""

    def test_22_pattern_allowed_after_1030(self, executor):
        """Test 2-2 patterns allowed after 10:30 ET."""
        signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 10, 31),  # 10:31 ET
            pattern_type=PatternType.PATTERN_22_UP,  # 2D-2U
            timeframe=Timeframe.HOURLY,
            direction=1,
            entry_price=500.0,
            stop_price=498.0,
            target_price=504.0,
                        risk_reward=2.0,
        )

        result = executor._check_hourly_time_filter(signal)
        assert result is True

    def test_22_pattern_blocked_before_1030(self, executor):
        """Test 2-2 patterns blocked before 10:30 ET."""
        signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 10, 15),  # 10:15 ET - before 10:30
            pattern_type=PatternType.PATTERN_22_UP,
            timeframe=Timeframe.HOURLY,
            direction=1,
            entry_price=500.0,
            stop_price=498.0,
            target_price=504.0,
                        risk_reward=2.0,
        )

        result = executor._check_hourly_time_filter(signal)
        assert result is False
        assert len(executor.get_skipped_time_filter()) == 1

    def test_3bar_pattern_allowed_after_1130(self, executor):
        """Test 3-bar patterns allowed after 11:30 ET."""
        signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 11, 45),  # 11:45 ET
            pattern_type=PatternType.PATTERN_312_UP,  # 3-1-2U (3-bar)
            timeframe=Timeframe.HOURLY,
            direction=1,
            entry_price=500.0,
            stop_price=495.0,
            target_price=510.0,
                        risk_reward=2.0,
        )

        result = executor._check_hourly_time_filter(signal)
        assert result is True

    def test_3bar_pattern_blocked_before_1130(self, executor):
        """Test 3-bar patterns blocked before 11:30 ET."""
        signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 10, 45),  # 10:45 ET - before 11:30
            pattern_type=PatternType.PATTERN_32_UP,  # 3-2U (uses 3-bar timing)
            timeframe=Timeframe.HOURLY,
            direction=1,
            entry_price=500.0,
            stop_price=495.0,
            target_price=510.0,
                        risk_reward=2.0,
        )

        result = executor._check_hourly_time_filter(signal)
        assert result is False

    def test_pattern_blocked_after_1530(self, executor):
        """Test all patterns blocked after 15:30 ET (exit time)."""
        signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 15, 35),  # 15:35 ET - after 15:30
            pattern_type=PatternType.PATTERN_22_UP,
            timeframe=Timeframe.HOURLY,
            direction=1,
            entry_price=500.0,
            stop_price=498.0,
            target_price=504.0,
                        risk_reward=2.0,
        )

        result = executor._check_hourly_time_filter(signal)
        assert result is False


# =============================================================================
# Test Magnitude Filter
# =============================================================================

class TestMagnitudeFilter:
    """Tests for magnitude filter (Session 83K-31)."""

    def test_low_magnitude_skipped(self, executor, sample_call_signal):
        """Test that low magnitude patterns are skipped."""
        # Create signal with low magnitude
        low_mag_signal = PatternSignal(
            timestamp=datetime(2026, 1, 15, 10, 30),
            pattern_type=PatternType.PATTERN_32_UP,
            timeframe=Timeframe.DAILY,
            direction=1,
            entry_price=500.0,
            stop_price=499.0,
            target_price=501.5,  # 0.3% magnitude (below 0.5% threshold)
                        risk_reward=1.5,
        )

        executor.clear_skipped_patterns()
        trades = executor.generate_option_trades(
            signals=[low_mag_signal],
            underlying='SPY',
            underlying_price=500.0,
        )

        # Trade should be skipped
        assert len(trades) == 0
        skipped = executor.get_skipped_patterns()
        assert len(skipped) == 1
        assert 'magnitude' in skipped[0]['reason']

    def test_sufficient_magnitude_accepted(self, executor, sample_call_signal):
        """Test that sufficient magnitude patterns are accepted."""
        executor.clear_skipped_patterns()

        # sample_call_signal has 2.0% magnitude (above 0.5% threshold)
        with patch.object(executor, '_calculate_expiration', return_value=datetime(2026, 2, 20)):
            trades = executor.generate_option_trades(
                signals=[sample_call_signal],
                underlying='SPY',
                underlying_price=500.0,
            )

        # Trade should be accepted (not in skipped list)
        skipped = executor.get_skipped_patterns()
        # Should not be skipped for magnitude
        assert not any('magnitude' in s.get('reason', '') for s in skipped)


# =============================================================================
# Test Expected Holding Days
# =============================================================================

class TestExpectedHoldingDays:
    """Tests for expected holding period calculation."""

    def test_daily_holding_days(self, executor):
        """Test daily patterns use 3 day holding."""
        days = executor._get_expected_holding_days(Timeframe.DAILY)
        assert days == 3

    def test_weekly_holding_days(self, executor):
        """Test weekly patterns use 7 day holding."""
        days = executor._get_expected_holding_days(Timeframe.WEEKLY)
        assert days == 7

    def test_monthly_holding_days(self, executor):
        """Test monthly patterns use 21 day holding."""
        days = executor._get_expected_holding_days(Timeframe.MONTHLY)
        assert days == 21

    def test_unknown_timeframe_defaults_to_weekly(self, executor):
        """Test unknown timeframe defaults to 7 days."""
        days = executor._get_expected_holding_days(None)
        assert days == 7


# =============================================================================
# Test Strike Interval
# =============================================================================

class TestStrikeInterval:
    """Tests for strike interval calculation."""

    def test_under_100_uses_1_dollar(self, executor):
        """Test prices under $100 use $1 intervals."""
        interval = executor._get_strike_interval(50.0)
        assert interval == 1.0

    def test_100_to_500_uses_5_dollar(self, executor):
        """Test prices $100-$500 use $5 intervals."""
        interval = executor._get_strike_interval(250.0)
        assert interval == 5.0

    def test_over_500_uses_10_dollar(self, executor):
        """Test prices over $500 use $10 intervals."""
        interval = executor._get_strike_interval(600.0)
        assert interval == 10.0


# =============================================================================
# Test Timezone Conversion
# =============================================================================

class TestTimezoneConversion:
    """Tests for _convert_to_naive_et function."""

    def test_naive_datetime_passthrough(self):
        """Test that naive datetimes pass through unchanged."""
        dt = datetime(2026, 1, 15, 10, 30)
        result = _convert_to_naive_et(dt)
        assert result == dt
        assert result.tzinfo is None

    def test_pandas_timestamp_conversion(self):
        """Test pandas Timestamp conversion."""
        ts = pd.Timestamp('2026-01-15 10:30:00', tz='America/New_York')
        result = _convert_to_naive_et(ts)

        assert result.tzinfo is None
        assert result.hour == 10
        assert result.minute == 30


# =============================================================================
# Test Fallback to Geometric Formula
# =============================================================================

class TestFallbackToGeometric:
    """Tests for geometric fallback strike selection."""

    def test_call_geometric_formula(self, executor, sample_call_signal):
        """Test 0.3x geometric formula for CALL."""
        strike, delta, theta = executor._fallback_to_geometric(
            sample_call_signal,
            underlying_price=500.0,
            option_type=OptionType.CALL,
        )

        # For CALL: strike = entry + 0.3 * (target - entry)
        # strike = 500 + 0.3 * (510 - 500) = 500 + 3 = 503
        # Rounded to $10 interval = 500.0
        assert strike == 500.0
        assert delta is None
        assert theta is None

    def test_put_geometric_formula(self, executor, sample_put_signal):
        """Test 0.3x geometric formula for PUT."""
        strike, delta, theta = executor._fallback_to_geometric(
            sample_put_signal,
            underlying_price=500.0,
            option_type=OptionType.PUT,
        )

        # For PUT: strike = entry - 0.3 * (entry - target)
        # strike = 500 - 0.3 * (500 - 490) = 500 - 3 = 497
        # Rounded to $10 interval = 500.0
        assert strike == 500.0
        assert delta is None
        assert theta is None


# =============================================================================
# Test Default DTE Values
# =============================================================================

class TestDefaultDTEValues:
    """Tests for default DTE configuration."""

    def test_default_dte_values(self):
        """Test default DTE values match STRAT methodology."""
        with patch('strat.options_module.get_alpaca_credentials', return_value={}):
            executor = OptionsExecutor()

        assert executor.default_dte_hourly == 7
        assert executor.default_dte_daily == 21
        assert executor.default_dte_weekly == 35
        assert executor.default_dte_monthly == 75

    def test_custom_dte_values(self):
        """Test custom DTE values."""
        with patch('strat.options_module.get_alpaca_credentials', return_value={}):
            executor = OptionsExecutor(
                default_dte_hourly=5,
                default_dte_daily=14,
                default_dte_weekly=30,
                default_dte_monthly=60,
            )

        assert executor.default_dte_hourly == 5
        assert executor.default_dte_daily == 14
        assert executor.default_dte_weekly == 30
        assert executor.default_dte_monthly == 60


# =============================================================================
# Test Hourly Config
# =============================================================================

class TestHourlyConfig:
    """Tests for hourly-specific configuration."""

    def test_default_hourly_config(self):
        """Test default hourly config values."""
        with patch('strat.options_module.get_alpaca_credentials', return_value={}):
            executor = OptionsExecutor()

        assert executor.hourly_config['first_entry_22'] == '10:30'
        assert executor.hourly_config['first_entry_3bar'] == '11:30'
        assert executor.hourly_config['last_exit'] == '15:30'
        assert executor.hourly_config['target_delta'] == 0.45
        assert executor.hourly_config['delta_range'] == (0.35, 0.50)

    def test_custom_hourly_config(self):
        """Test custom hourly config values."""
        custom_config = {
            'first_entry_22': '10:00',
            'last_exit': '15:00',
        }
        with patch('strat.options_module.get_alpaca_credentials', return_value={}):
            executor = OptionsExecutor(hourly_config=custom_config)

        assert executor.hourly_config['first_entry_22'] == '10:00'
        assert executor.hourly_config['last_exit'] == '15:00'
        # Unchanged defaults
        assert executor.hourly_config['first_entry_3bar'] == '11:30'
