"""
Tests for strat/signal_automation/position_monitor.py

Comprehensive tests for position monitoring in the signal automation system.

Session: EQUITY-80
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from strat.signal_automation.position_monitor import (
    ExitReason,
    MonitoringConfig,
    TrackedPosition,
    ExitSignal,
    PositionMonitor,
)


# =============================================================================
# ExitReason Enum Tests
# =============================================================================

class TestExitReason:
    """Tests for ExitReason enum."""

    def test_target_hit_value(self):
        """Test TARGET_HIT value."""
        assert ExitReason.TARGET_HIT.value == "TARGET"

    def test_stop_hit_value(self):
        """Test STOP_HIT value."""
        assert ExitReason.STOP_HIT.value == "STOP"

    def test_dte_exit_value(self):
        """Test DTE_EXIT value."""
        assert ExitReason.DTE_EXIT.value == "DTE"

    def test_max_loss_value(self):
        """Test MAX_LOSS value."""
        assert ExitReason.MAX_LOSS.value == "MAX_LOSS"

    def test_manual_value(self):
        """Test MANUAL value."""
        assert ExitReason.MANUAL.value == "MANUAL"

    def test_time_exit_value(self):
        """Test TIME_EXIT value."""
        assert ExitReason.TIME_EXIT.value == "TIME"

    def test_eod_exit_value(self):
        """Test EOD_EXIT value."""
        assert ExitReason.EOD_EXIT.value == "EOD"

    def test_partial_exit_value(self):
        """Test PARTIAL_EXIT value."""
        assert ExitReason.PARTIAL_EXIT.value == "PARTIAL"

    def test_trailing_stop_value(self):
        """Test TRAILING_STOP value."""
        assert ExitReason.TRAILING_STOP.value == "TRAIL"

    def test_pattern_invalidated_value(self):
        """Test PATTERN_INVALIDATED value."""
        assert ExitReason.PATTERN_INVALIDATED.value == "PATTERN"

    def test_all_reasons_are_strings(self):
        """Test all reasons inherit from str."""
        for reason in ExitReason:
            assert isinstance(reason, str)

    def test_enum_count(self):
        """Test correct number of exit reasons."""
        assert len(ExitReason) == 10


# =============================================================================
# MonitoringConfig Tests
# =============================================================================

class TestMonitoringConfig:
    """Tests for MonitoringConfig dataclass."""

    def test_default_exit_dte(self):
        """Test default exit DTE is 3."""
        config = MonitoringConfig()
        assert config.exit_dte == 3

    def test_default_max_loss_pct(self):
        """Test default max loss is 50%."""
        config = MonitoringConfig()
        assert config.max_loss_pct == 0.50

    def test_default_max_profit_pct(self):
        """Test default max profit is 100%."""
        config = MonitoringConfig()
        assert config.max_profit_pct == 1.00

    def test_default_check_interval(self):
        """Test default check interval is 60 seconds."""
        config = MonitoringConfig()
        assert config.check_interval == 60

    def test_default_minimum_hold_seconds(self):
        """Test default minimum hold is 300 seconds."""
        config = MonitoringConfig()
        assert config.minimum_hold_seconds == 300

    def test_default_use_market_orders(self):
        """Test default use market orders is True."""
        config = MonitoringConfig()
        assert config.use_market_orders is True

    def test_default_eod_exit_time(self):
        """Test default EOD exit time is 15:59."""
        config = MonitoringConfig()
        assert config.eod_exit_hour == 15
        assert config.eod_exit_minute == 59

    def test_default_hourly_target_rr(self):
        """Test default hourly target R:R is 1.0."""
        config = MonitoringConfig()
        assert config.hourly_target_rr == 1.0

    def test_default_trailing_stop_settings(self):
        """Test default trailing stop settings."""
        config = MonitoringConfig()
        assert config.use_trailing_stop is True
        assert config.trailing_stop_activation_rr == 0.5
        assert config.trailing_stop_pct == 0.50

    def test_default_atr_trailing_settings(self):
        """Test default ATR trailing settings for 3-2 patterns."""
        config = MonitoringConfig()
        assert config.use_atr_trailing_for_32 is True
        assert config.atr_trailing_activation_multiple == 0.75
        assert config.atr_trailing_distance_multiple == 1.0

    def test_default_partial_exit_settings(self):
        """Test default partial exit settings."""
        config = MonitoringConfig()
        assert config.partial_exit_enabled is True
        assert config.partial_exit_rr == 1.0
        assert config.partial_exit_pct == 0.50

    def test_post_init_creates_max_loss_by_timeframe(self):
        """Test __post_init__ creates timeframe-specific max loss dict."""
        config = MonitoringConfig()
        assert config.max_loss_pct_by_timeframe is not None
        assert '1M' in config.max_loss_pct_by_timeframe
        assert '1W' in config.max_loss_pct_by_timeframe
        assert '1D' in config.max_loss_pct_by_timeframe
        assert '1H' in config.max_loss_pct_by_timeframe

    def test_max_loss_by_timeframe_values(self):
        """Test timeframe-specific max loss values."""
        config = MonitoringConfig()
        assert config.max_loss_pct_by_timeframe['1M'] == 0.75
        assert config.max_loss_pct_by_timeframe['1W'] == 0.65
        assert config.max_loss_pct_by_timeframe['1D'] == 0.50
        assert config.max_loss_pct_by_timeframe['1H'] == 0.40

    def test_get_max_loss_pct_1m(self):
        """Test get_max_loss_pct returns correct value for 1M."""
        config = MonitoringConfig()
        assert config.get_max_loss_pct('1M') == 0.75

    def test_get_max_loss_pct_1w(self):
        """Test get_max_loss_pct returns correct value for 1W."""
        config = MonitoringConfig()
        assert config.get_max_loss_pct('1W') == 0.65

    def test_get_max_loss_pct_1d(self):
        """Test get_max_loss_pct returns correct value for 1D."""
        config = MonitoringConfig()
        assert config.get_max_loss_pct('1D') == 0.50

    def test_get_max_loss_pct_1h(self):
        """Test get_max_loss_pct returns correct value for 1H."""
        config = MonitoringConfig()
        assert config.get_max_loss_pct('1H') == 0.40

    def test_get_max_loss_pct_unknown_timeframe(self):
        """Test get_max_loss_pct returns default for unknown timeframe."""
        config = MonitoringConfig()
        assert config.get_max_loss_pct('UNKNOWN') == config.max_loss_pct

    def test_custom_max_loss_by_timeframe(self):
        """Test custom max loss by timeframe."""
        custom = {'1H': 0.30, '1D': 0.60}
        config = MonitoringConfig(max_loss_pct_by_timeframe=custom)
        assert config.get_max_loss_pct('1H') == 0.30
        assert config.get_max_loss_pct('1D') == 0.60


# =============================================================================
# TrackedPosition Tests
# =============================================================================

class TestTrackedPosition:
    """Tests for TrackedPosition dataclass."""

    @pytest.fixture
    def sample_position(self):
        """Create a sample tracked position."""
        return TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H_3-1-2U_2024-12-15',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime(2024, 12, 15, 10, 30),
            expiration='2024-12-20',
        )

    def test_creation_basic(self, sample_position):
        """Test basic position creation."""
        assert sample_position.symbol == 'SPY'
        assert sample_position.direction == 'CALL'
        assert sample_position.contracts == 2

    def test_default_values(self, sample_position):
        """Test default values are set correctly."""
        assert sample_position.current_price == 0.0
        assert sample_position.unrealized_pnl == 0.0
        assert sample_position.is_active is True
        assert sample_position.exit_reason is None
        assert sample_position.trailing_stop_active is False

    def test_intrabar_low_default(self, sample_position):
        """Test intrabar_low defaults to infinity."""
        assert sample_position.intrabar_low == float('inf')

    def test_to_dict_basic(self, sample_position):
        """Test to_dict returns correct structure."""
        d = sample_position.to_dict()
        assert d['osi_symbol'] == 'SPY241220C00450000'
        assert d['symbol'] == 'SPY'
        assert d['direction'] == 'CALL'
        assert d['contracts'] == 2

    def test_to_dict_entry_time_format(self, sample_position):
        """Test to_dict formats entry_time as ISO."""
        d = sample_position.to_dict()
        assert 'entry_time' in d
        assert '2024-12-15' in d['entry_time']

    def test_to_dict_intrabar_low_not_inf(self, sample_position):
        """Test to_dict converts inf to 0.0."""
        d = sample_position.to_dict()
        # intrabar_low should be 0.0 instead of inf in dict
        assert d['intrabar_low'] == 0.0

    def test_to_dict_with_exit_time(self, sample_position):
        """Test to_dict with exit time."""
        sample_position.exit_time = datetime(2024, 12, 15, 14, 30)
        d = sample_position.to_dict()
        assert 'exit_time' in d
        assert '2024-12-15' in d['exit_time']

    def test_to_dict_without_exit_time(self, sample_position):
        """Test to_dict with None exit time."""
        d = sample_position.to_dict()
        assert d['exit_time'] is None

    def test_to_dict_includes_all_fields(self, sample_position):
        """Test to_dict includes all important fields."""
        d = sample_position.to_dict()
        required_fields = [
            'osi_symbol', 'signal_key', 'symbol', 'direction',
            'entry_trigger', 'target_price', 'stop_price',
            'pattern_type', 'timeframe', 'entry_price', 'contracts',
            'is_active', 'trailing_stop_active', 'atr_at_detection',
            'entry_bar_type', 'intrabar_high', 'intrabar_low'
        ]
        for field in required_fields:
            assert field in d


# =============================================================================
# ExitSignal Tests
# =============================================================================

class TestExitSignal:
    """Tests for ExitSignal dataclass."""

    def test_creation_basic(self):
        """Test basic ExitSignal creation."""
        signal = ExitSignal(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            reason=ExitReason.TARGET_HIT,
            underlying_price=455.0,
            current_option_price=8.50,
            unrealized_pnl=300.0,
            dte=5,
        )
        assert signal.reason == ExitReason.TARGET_HIT
        assert signal.underlying_price == 455.0

    def test_default_details_empty(self):
        """Test details defaults to empty string."""
        signal = ExitSignal(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            reason=ExitReason.STOP_HIT,
            underlying_price=440.0,
            current_option_price=3.50,
            unrealized_pnl=-200.0,
            dte=5,
        )
        assert signal.details == ""

    def test_timestamp_defaults_to_now(self):
        """Test timestamp defaults to current time."""
        before = datetime.now()
        signal = ExitSignal(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            reason=ExitReason.DTE_EXIT,
            underlying_price=450.0,
            current_option_price=5.00,
            unrealized_pnl=0.0,
            dte=2,
        )
        after = datetime.now()
        assert before <= signal.timestamp <= after

    def test_contracts_to_close_defaults_none(self):
        """Test contracts_to_close defaults to None (close all)."""
        signal = ExitSignal(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            reason=ExitReason.MAX_LOSS,
            underlying_price=435.0,
            current_option_price=2.00,
            unrealized_pnl=-350.0,
            dte=5,
        )
        assert signal.contracts_to_close is None

    def test_partial_exit_with_contracts(self):
        """Test partial exit with specific contract count."""
        signal = ExitSignal(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            reason=ExitReason.PARTIAL_EXIT,
            underlying_price=452.0,
            current_option_price=7.00,
            unrealized_pnl=150.0,
            dte=5,
            contracts_to_close=1,
        )
        assert signal.contracts_to_close == 1


# =============================================================================
# PositionMonitor Initialization Tests
# =============================================================================

class TestPositionMonitorInit:
    """Tests for PositionMonitor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        monitor = PositionMonitor()
        assert monitor.config is not None
        assert isinstance(monitor.config, MonitoringConfig)

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = MonitoringConfig(exit_dte=5, max_loss_pct=0.40)
        monitor = PositionMonitor(config=config)
        assert monitor.config.exit_dte == 5
        assert monitor.config.max_loss_pct == 0.40

    def test_init_internal_state(self):
        """Test initial internal state."""
        monitor = PositionMonitor()
        assert monitor._positions == {}
        assert monitor._underlying_cache == {}
        assert monitor._check_count == 0
        assert monitor._exit_count == 0
        assert monitor._error_count == 0

    def test_init_with_callback(self):
        """Test initialization with exit callback."""
        callback = Mock()
        monitor = PositionMonitor(on_exit_callback=callback)
        assert monitor.on_exit_callback == callback


# =============================================================================
# PositionMonitor._parse_expiration Tests
# =============================================================================

class TestParseExpiration:
    """Tests for _parse_expiration method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    def test_parse_standard_osi_call(self, monitor):
        """Test parsing standard OSI call symbol."""
        result = monitor._parse_expiration('SPY241220C00450000')
        assert result == '2024-12-20'

    def test_parse_standard_osi_put(self, monitor):
        """Test parsing standard OSI put symbol."""
        result = monitor._parse_expiration('AAPL250117P00175000')
        assert result == '2025-01-17'

    def test_parse_different_ticker_lengths(self, monitor):
        """Test parsing with different ticker lengths."""
        # 3-letter ticker
        result1 = monitor._parse_expiration('SPY241220C00450000')
        assert result1 == '2024-12-20'

        # 4-letter ticker
        result2 = monitor._parse_expiration('AAPL241220C00175000')
        assert result2 == '2024-12-20'

        # 5-letter ticker (like GOOGL)
        result3 = monitor._parse_expiration('GOOGL241220C00140000')
        assert result3 == '2024-12-20'

    def test_parse_invalid_symbol(self, monitor):
        """Test parsing invalid symbol returns empty string."""
        result = monitor._parse_expiration('INVALID')
        assert result == ''

    def test_parse_empty_symbol(self, monitor):
        """Test parsing empty symbol returns empty string."""
        result = monitor._parse_expiration('')
        assert result == ''


# =============================================================================
# PositionMonitor._calculate_dte Tests
# =============================================================================

class TestCalculateDte:
    """Tests for _calculate_dte method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    def test_calculate_dte_future_date(self, monitor):
        """Test DTE calculation for future date."""
        future_date = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
        dte = monitor._calculate_dte(future_date)
        assert dte == 10 or dte == 9  # Allow for time-of-day variation

    def test_calculate_dte_today(self, monitor):
        """Test DTE calculation for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        dte = monitor._calculate_dte(today)
        assert dte == 0

    def test_calculate_dte_past_date(self, monitor):
        """Test DTE calculation for past date returns 0."""
        past_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        dte = monitor._calculate_dte(past_date)
        assert dte == 0

    def test_calculate_dte_empty_string(self, monitor):
        """Test DTE calculation for empty string."""
        dte = monitor._calculate_dte('')
        assert dte == 0

    def test_calculate_dte_invalid_format(self, monitor):
        """Test DTE calculation for invalid format."""
        dte = monitor._calculate_dte('invalid-date')
        assert dte == 0


# =============================================================================
# PositionMonitor._check_target_hit Tests
# =============================================================================

class TestCheckTargetHit:
    """Tests for _check_target_hit method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    @pytest.fixture
    def call_position(self):
        return TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )

    @pytest.fixture
    def put_position(self):
        return TrackedPosition(
            osi_symbol='SPY241220P00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='PUT',
            entry_trigger=455.0,
            target_price=445.0,
            stop_price=460.0,
            pattern_type='3-1-2D',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )

    def test_call_target_hit(self, monitor, call_position):
        """Test CALL target hit when price above target."""
        call_position.underlying_price = 456.0
        assert monitor._check_target_hit(call_position) is True

    def test_call_target_not_hit(self, monitor, call_position):
        """Test CALL target not hit when price below target."""
        call_position.underlying_price = 450.0
        assert monitor._check_target_hit(call_position) is False

    def test_call_target_exact(self, monitor, call_position):
        """Test CALL target hit at exact target price."""
        call_position.underlying_price = 455.0
        assert monitor._check_target_hit(call_position) is True

    def test_put_target_hit(self, monitor, put_position):
        """Test PUT target hit when price below target."""
        put_position.underlying_price = 444.0
        assert monitor._check_target_hit(put_position) is True

    def test_put_target_not_hit(self, monitor, put_position):
        """Test PUT target not hit when price above target."""
        put_position.underlying_price = 450.0
        assert monitor._check_target_hit(put_position) is False

    def test_put_target_exact(self, monitor, put_position):
        """Test PUT target hit at exact target price."""
        put_position.underlying_price = 445.0
        assert monitor._check_target_hit(put_position) is True

    def test_bull_direction(self, monitor, call_position):
        """Test BULL direction treated as CALL."""
        call_position.direction = 'BULL'
        call_position.underlying_price = 456.0
        assert monitor._check_target_hit(call_position) is True


# =============================================================================
# PositionMonitor._check_stop_hit Tests
# =============================================================================

class TestCheckStopHit:
    """Tests for _check_stop_hit method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    @pytest.fixture
    def call_position(self):
        return TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )

    @pytest.fixture
    def put_position(self):
        return TrackedPosition(
            osi_symbol='SPY241220P00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='PUT',
            entry_trigger=455.0,
            target_price=445.0,
            stop_price=460.0,
            pattern_type='3-1-2D',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )

    def test_call_stop_hit(self, monitor, call_position):
        """Test CALL stop hit when price below stop."""
        call_position.underlying_price = 438.0
        assert monitor._check_stop_hit(call_position) is True

    def test_call_stop_not_hit(self, monitor, call_position):
        """Test CALL stop not hit when price above stop."""
        call_position.underlying_price = 445.0
        assert monitor._check_stop_hit(call_position) is False

    def test_call_stop_exact(self, monitor, call_position):
        """Test CALL stop hit at exact stop price."""
        call_position.underlying_price = 440.0
        assert monitor._check_stop_hit(call_position) is True

    def test_put_stop_hit(self, monitor, put_position):
        """Test PUT stop hit when price above stop."""
        put_position.underlying_price = 462.0
        assert monitor._check_stop_hit(put_position) is True

    def test_put_stop_not_hit(self, monitor, put_position):
        """Test PUT stop not hit when price below stop."""
        put_position.underlying_price = 455.0
        assert monitor._check_stop_hit(put_position) is False

    def test_put_stop_exact(self, monitor, put_position):
        """Test PUT stop hit at exact stop price."""
        put_position.underlying_price = 460.0
        assert monitor._check_stop_hit(put_position) is True


# =============================================================================
# PositionMonitor._check_partial_exit Tests
# =============================================================================

class TestCheckPartialExit:
    """Tests for _check_partial_exit method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    @pytest.fixture
    def multi_contract_position(self):
        return TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            target_1x=450.0,  # 1.0x R:R target
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=4,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )

    def test_partial_exit_skips_single_contract(self, monitor):
        """Test partial exit skips positions with 1 contract."""
        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            target_1x=450.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=1,  # Only 1 contract
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )
        pos.underlying_price = 451.0  # Above target_1x
        result = monitor._check_partial_exit(pos)
        assert result is None

    def test_partial_exit_skips_if_already_done(self, monitor, multi_contract_position):
        """Test partial exit skips if already done."""
        multi_contract_position.partial_exit_done = True
        multi_contract_position.underlying_price = 451.0
        result = monitor._check_partial_exit(multi_contract_position)
        assert result is None

    def test_partial_exit_call_target_hit(self, monitor, multi_contract_position):
        """Test partial exit triggers when CALL target_1x hit."""
        multi_contract_position.underlying_price = 451.0  # Above target_1x (450)
        result = monitor._check_partial_exit(multi_contract_position)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT
        assert result.contracts_to_close == 2  # 50% of 4

    def test_partial_exit_call_target_not_hit(self, monitor, multi_contract_position):
        """Test partial exit does not trigger when below target_1x."""
        multi_contract_position.underlying_price = 448.0  # Below target_1x
        result = monitor._check_partial_exit(multi_contract_position)
        assert result is None

    def test_partial_exit_put_target_hit(self, monitor):
        """Test partial exit triggers when PUT target_1x hit."""
        pos = TrackedPosition(
            osi_symbol='SPY241220P00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='PUT',
            entry_trigger=455.0,
            target_price=445.0,
            target_1x=450.0,  # 1.0x R:R target
            stop_price=460.0,
            pattern_type='3-1-2D',
            timeframe='1H',
            entry_price=5.50,
            contracts=4,
            entry_time=datetime.now(),
            expiration='2024-12-20',
        )
        pos.underlying_price = 449.0  # Below target_1x (450)
        result = monitor._check_partial_exit(pos)
        assert result is not None
        assert result.reason == ExitReason.PARTIAL_EXIT


# =============================================================================
# PositionMonitor Sync Positions Tests
# =============================================================================

class TestSyncPositions:
    """Tests for sync_positions method."""

    def test_sync_without_trading_client(self):
        """Test sync returns 0 without trading client."""
        monitor = PositionMonitor()
        result = monitor.sync_positions()
        assert result == 0

    def test_sync_handles_alpaca_error(self):
        """Test sync handles Alpaca API errors."""
        trading_client = Mock()
        trading_client.list_option_positions.side_effect = Exception("API Error")

        monitor = PositionMonitor(trading_client=trading_client)
        result = monitor.sync_positions()

        assert result == 0
        assert monitor._error_count == 1


# =============================================================================
# PositionMonitor Check Position Tests
# =============================================================================

class TestCheckPosition:
    """Tests for _check_position method."""

    @pytest.fixture
    def monitor(self):
        return PositionMonitor()

    @pytest.fixture
    def position(self):
        return TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now() - timedelta(minutes=10),  # 10 min ago
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )

    def test_minimum_hold_time_blocks_exit(self, monitor):
        """Test minimum hold time blocks exit checks."""
        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1D',  # Daily to avoid EOD check
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now() - timedelta(seconds=60),  # Only 60s ago
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )
        pos.underlying_price = 456.0  # Above target
        result = monitor._check_position(pos)
        assert result is None  # Blocked by minimum hold

    def test_dte_exit(self, monitor, position):
        """Test DTE exit when below threshold."""
        position.timeframe = '1D'  # Avoid EOD check
        position.dte = 2  # Below threshold of 3
        position.expiration = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        result = monitor._check_position(position)
        assert result is not None
        assert result.reason == ExitReason.DTE_EXIT

    def test_max_loss_exit(self):
        """Test max loss exit when threshold exceeded."""
        # Disable partial exit to test max loss in isolation
        config = MonitoringConfig(partial_exit_enabled=False, minimum_hold_seconds=0)
        monitor = PositionMonitor(config=config)

        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,  # Single contract to avoid partial exit
            entry_time=datetime.now() - timedelta(minutes=10),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )
        pos.underlying_price = 442.0  # Not hitting stop
        pos.unrealized_pct = -0.45  # 45% loss, threshold for 1D is 50%

        result = monitor._check_position(pos)
        assert result is None  # Not yet at threshold

        pos.unrealized_pct = -0.55  # 55% loss, above 50% threshold
        result = monitor._check_position(pos)
        assert result is not None
        assert result.reason == ExitReason.MAX_LOSS

    def test_timeframe_specific_max_loss(self):
        """Test timeframe-specific max loss thresholds.

        Note: Uses 1D timeframe which has 50% max loss threshold.
        Uses 1 contract to avoid partial exit logic.
        """
        # Disable partial exit to test max loss in isolation
        config = MonitoringConfig(partial_exit_enabled=False, minimum_hold_seconds=0)
        monitor = PositionMonitor(config=config)

        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1D',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1D',  # 50% threshold, avoids 1H EOD logic
            entry_price=5.50,
            contracts=1,  # Single contract to avoid partial exit
            entry_time=datetime.now() - timedelta(minutes=10),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )
        pos.underlying_price = 442.0  # Not hitting stop
        pos.unrealized_pct = -0.55  # 55% loss, above 50% threshold for 1D

        result = monitor._check_position(pos)
        assert result is not None
        assert result.reason == ExitReason.MAX_LOSS


# =============================================================================
# PositionMonitor Statistics Tests
# =============================================================================

class TestPositionMonitorStats:
    """Tests for position monitor statistics."""

    def test_check_count_increments(self):
        """Test check count increments on check_positions."""
        monitor = PositionMonitor()
        assert monitor._check_count == 0

        with patch.object(monitor, 'sync_positions'):
            with patch.object(monitor, '_update_underlying_prices'):
                with patch.object(monitor, '_update_bar_data'):
                    monitor.check_positions()

        assert monitor._check_count == 1

    def test_error_count_increments_on_error(self):
        """Test error count increments on Alpaca errors."""
        trading_client = Mock()
        trading_client.list_option_positions.side_effect = Exception("API Error")

        monitor = PositionMonitor(trading_client=trading_client)
        assert monitor._error_count == 0

        monitor.sync_positions()
        assert monitor._error_count == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestPositionMonitorIntegration:
    """Integration tests for PositionMonitor."""

    def test_full_exit_flow_target_hit(self):
        """Test complete exit flow when target is hit."""
        config = MonitoringConfig(minimum_hold_seconds=0)  # Disable hold time
        monitor = PositionMonitor(config=config)

        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now() - timedelta(hours=1),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )
        pos.underlying_price = 456.0  # Above target
        pos.current_price = 8.00
        pos.unrealized_pnl = 500.0
        pos.unrealized_pct = 0.45

        result = monitor._check_position(pos)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT
        assert result.underlying_price == 456.0

    def test_full_exit_flow_stop_hit(self):
        """Test complete exit flow when stop is hit."""
        config = MonitoringConfig(minimum_hold_seconds=0)
        monitor = PositionMonitor(config=config)

        pos = TrackedPosition(
            osi_symbol='SPY241220C00450000',
            signal_key='SPY_1H',
            symbol='SPY',
            direction='CALL',
            entry_trigger=445.0,
            target_price=455.0,
            stop_price=440.0,
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=2,
            entry_time=datetime.now() - timedelta(hours=1),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        )
        pos.underlying_price = 438.0  # Below stop
        pos.current_price = 2.00
        pos.unrealized_pnl = -350.0
        pos.unrealized_pct = -0.35

        result = monitor._check_position(pos)
        assert result is not None
        assert result.reason == ExitReason.STOP_HIT
