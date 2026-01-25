"""
Tests for ExitConditionEvaluator coordinator.

EQUITY-90: Phase 4.1 - Extracted exit condition evaluation from PositionMonitor.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strat.signal_automation.position_monitor import (
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)
from strat.signal_automation.coordinators.exit_evaluator import (
    ExitConditionEvaluator,
    TrailingStopChecker,
    PartialExitChecker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default monitoring config."""
    return MonitoringConfig()


@pytest.fixture
def evaluator(config):
    """ExitConditionEvaluator with default config."""
    return ExitConditionEvaluator(config=config)


@pytest.fixture
def call_position():
    """Basic bullish position for testing."""
    return TrackedPosition(
        osi_symbol='SPY250120C00450000',
        signal_key='SPY_1D_2U_test',
        symbol='SPY',
        direction='CALL',
        entry_trigger=450.0,
        target_price=460.0,
        stop_price=445.0,
        pattern_type='2-1-2U',
        timeframe='1D',
        entry_price=5.50,
        contracts=2,
        entry_time=datetime.now() - timedelta(hours=1),  # 1 hour ago
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        current_price=6.00,
        unrealized_pnl=100.0,
        unrealized_pct=0.10,  # 10% profit
        underlying_price=455.0,
        dte=10,
        target_1x=455.0,
        original_target=460.0,
        actual_entry_underlying=450.0,
        contracts_remaining=2,
        entry_bar_type='2U',
        entry_bar_high=452.0,
        entry_bar_low=448.0,
        intrabar_high=455.0,
        intrabar_low=449.0,
    )


@pytest.fixture
def put_position():
    """Basic bearish position for testing."""
    return TrackedPosition(
        osi_symbol='SPY250120P00450000',
        signal_key='SPY_1D_2D_test',
        symbol='SPY',
        direction='PUT',
        entry_trigger=450.0,
        target_price=440.0,
        stop_price=455.0,
        pattern_type='2-1-2D',
        timeframe='1D',
        entry_price=5.50,
        contracts=2,
        entry_time=datetime.now() - timedelta(hours=1),
        expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
        current_price=6.00,
        unrealized_pnl=100.0,
        unrealized_pct=0.10,
        underlying_price=445.0,
        dte=10,
        target_1x=445.0,
        original_target=440.0,
        actual_entry_underlying=450.0,
        contracts_remaining=2,
        entry_bar_type='2D',
        entry_bar_high=452.0,
        entry_bar_low=448.0,
        intrabar_high=451.0,
        intrabar_low=445.0,
    )


# =============================================================================
# Test ExitConditionEvaluator Initialization
# =============================================================================


class TestExitConditionEvaluatorInit:
    """Tests for ExitConditionEvaluator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        evaluator = ExitConditionEvaluator(config=MonitoringConfig())
        assert evaluator.config is not None
        assert evaluator._trailing_stop_checker is None
        assert evaluator._partial_exit_checker is None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = MonitoringConfig(exit_dte=5, max_loss_pct=0.60)
        evaluator = ExitConditionEvaluator(config=config)
        assert evaluator.config.exit_dte == 5
        assert evaluator.config.max_loss_pct == 0.60

    def test_init_with_trailing_stop_checker(self):
        """Test initialization with trailing stop checker."""
        mock_checker = Mock(spec=TrailingStopChecker)
        evaluator = ExitConditionEvaluator(
            config=MonitoringConfig(),
            trailing_stop_checker=mock_checker,
        )
        assert evaluator._trailing_stop_checker is mock_checker

    def test_init_with_partial_exit_checker(self):
        """Test initialization with partial exit checker."""
        mock_checker = Mock(spec=PartialExitChecker)
        evaluator = ExitConditionEvaluator(
            config=MonitoringConfig(),
            partial_exit_checker=mock_checker,
        )
        assert evaluator._partial_exit_checker is mock_checker

    def test_set_trailing_stop_checker(self, evaluator):
        """Test setting trailing stop checker after init."""
        mock_checker = Mock(spec=TrailingStopChecker)
        evaluator.set_trailing_stop_checker(mock_checker)
        assert evaluator._trailing_stop_checker is mock_checker

    def test_set_partial_exit_checker(self, evaluator):
        """Test setting partial exit checker after init."""
        mock_checker = Mock(spec=PartialExitChecker)
        evaluator.set_partial_exit_checker(mock_checker)
        assert evaluator._partial_exit_checker is mock_checker


# =============================================================================
# Test Minimum Hold Time
# =============================================================================


class TestMinimumHoldTime:
    """Tests for minimum hold time check."""

    def test_blocks_exit_if_held_too_short(self, evaluator, call_position):
        """Test that positions held less than minimum are blocked."""
        call_position.entry_time = datetime.now() - timedelta(seconds=60)  # Only 60s
        result = evaluator.evaluate(call_position)
        assert result is None

    def test_allows_exit_if_held_long_enough(self, evaluator, call_position):
        """Test that positions held long enough can exit."""
        call_position.entry_time = datetime.now() - timedelta(seconds=400)  # 400s > 300s
        call_position.underlying_price = 460.0  # Above target
        result = evaluator.evaluate(call_position)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT

    def test_respects_custom_hold_time(self, call_position):
        """Test custom minimum hold seconds."""
        config = MonitoringConfig(minimum_hold_seconds=600)  # 10 minutes
        evaluator = ExitConditionEvaluator(config=config)
        call_position.entry_time = datetime.now() - timedelta(seconds=400)  # 6.7 min
        result = evaluator.evaluate(call_position)
        assert result is None  # Still blocked


# =============================================================================
# Test DTE Exit
# =============================================================================


class TestDTEExit:
    """Tests for DTE exit condition."""

    def test_dte_exit_when_at_threshold(self, evaluator, call_position):
        """Test exit when DTE equals threshold."""
        call_position.expiration = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
        result = evaluator.evaluate(call_position)
        assert result is not None
        assert result.reason == ExitReason.DTE_EXIT
        assert 'DTE' in result.details

    def test_dte_exit_when_below_threshold(self, evaluator, call_position):
        """Test exit when DTE below threshold."""
        call_position.expiration = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        result = evaluator.evaluate(call_position)
        assert result is not None
        assert result.reason == ExitReason.DTE_EXIT

    def test_no_dte_exit_when_above_threshold(self, evaluator, call_position):
        """Test no exit when DTE above threshold."""
        call_position.expiration = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
        call_position.underlying_price = 455.0  # Not at target
        result = evaluator.evaluate(call_position)
        # No DTE exit, but could be other conditions
        if result:
            assert result.reason != ExitReason.DTE_EXIT

    def test_custom_dte_threshold(self, call_position):
        """Test custom DTE threshold."""
        config = MonitoringConfig(exit_dte=5)
        evaluator = ExitConditionEvaluator(config=config)
        call_position.expiration = (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')
        result = evaluator.evaluate(call_position)
        assert result is not None
        assert result.reason == ExitReason.DTE_EXIT


# =============================================================================
# Test Target Hit
# =============================================================================


class TestTargetHit:
    """Tests for target hit condition."""

    def test_call_target_hit(self, evaluator, call_position):
        """Test call target hit when price above target."""
        call_position.underlying_price = 461.0  # Above 460 target
        result = evaluator.evaluate(call_position, underlying_price=461.0)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT
        assert '461.00' in result.details

    def test_call_target_exact(self, evaluator, call_position):
        """Test call target hit at exact target price."""
        call_position.underlying_price = 460.0
        result = evaluator.evaluate(call_position, underlying_price=460.0)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT

    def test_call_target_not_hit(self, evaluator, call_position):
        """Test no exit when below target."""
        call_position.underlying_price = 458.0
        result = evaluator.evaluate(call_position, underlying_price=458.0)
        if result:
            assert result.reason != ExitReason.TARGET_HIT

    def test_put_target_hit(self, evaluator, put_position):
        """Test put target hit when price below target."""
        put_position.underlying_price = 439.0  # Below 440 target
        result = evaluator.evaluate(put_position, underlying_price=439.0)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT

    def test_put_target_not_hit(self, evaluator, put_position):
        """Test no exit when above target."""
        put_position.underlying_price = 442.0
        result = evaluator.evaluate(put_position, underlying_price=442.0)
        if result:
            assert result.reason != ExitReason.TARGET_HIT


# =============================================================================
# Test Stop Hit
# =============================================================================


class TestStopHit:
    """Tests for stop hit condition."""

    def test_call_stop_hit(self, evaluator, call_position):
        """Test call stop hit when price below stop."""
        call_position.underlying_price = 444.0  # Below 445 stop
        result = evaluator.evaluate(call_position, underlying_price=444.0)
        assert result is not None
        assert result.reason == ExitReason.STOP_HIT
        assert '444.00' in result.details

    def test_call_stop_exact(self, evaluator, call_position):
        """Test call stop hit at exact stop price."""
        call_position.underlying_price = 445.0
        result = evaluator.evaluate(call_position, underlying_price=445.0)
        assert result is not None
        assert result.reason == ExitReason.STOP_HIT

    def test_call_stop_not_hit(self, evaluator, call_position):
        """Test no exit when above stop."""
        call_position.underlying_price = 450.0
        result = evaluator.evaluate(call_position, underlying_price=450.0)
        if result:
            assert result.reason != ExitReason.STOP_HIT

    def test_put_stop_hit(self, evaluator, put_position):
        """Test put stop hit when price above stop."""
        put_position.underlying_price = 456.0  # Above 455 stop
        result = evaluator.evaluate(put_position, underlying_price=456.0)
        assert result is not None
        assert result.reason == ExitReason.STOP_HIT

    def test_put_stop_not_hit(self, evaluator, put_position):
        """Test no exit when below stop."""
        put_position.underlying_price = 450.0
        result = evaluator.evaluate(put_position, underlying_price=450.0)
        if result:
            assert result.reason != ExitReason.STOP_HIT


# =============================================================================
# Test Max Loss
# =============================================================================


class TestMaxLoss:
    """Tests for max loss condition."""

    def test_max_loss_exceeded(self, evaluator, call_position):
        """Test exit when max loss exceeded."""
        call_position.unrealized_pct = -0.55  # 55% loss, default threshold is 50%
        call_position.underlying_price = 448.0  # Not at stop
        result = evaluator.evaluate(call_position, underlying_price=448.0)
        assert result is not None
        assert result.reason == ExitReason.MAX_LOSS
        assert '-55.0%' in result.details

    def test_max_loss_exact(self, evaluator, call_position):
        """Test exit at exact max loss threshold."""
        call_position.unrealized_pct = -0.50
        call_position.underlying_price = 448.0
        result = evaluator.evaluate(call_position, underlying_price=448.0)
        assert result is not None
        assert result.reason == ExitReason.MAX_LOSS

    def test_max_loss_not_exceeded(self, evaluator, call_position):
        """Test no exit when loss below threshold."""
        call_position.unrealized_pct = -0.30  # 30% loss
        call_position.underlying_price = 448.0
        result = evaluator.evaluate(call_position, underlying_price=448.0)
        if result:
            assert result.reason != ExitReason.MAX_LOSS

    def test_timeframe_specific_max_loss_1h(self, call_position):
        """Test 1H uses tighter max loss threshold."""
        config = MonitoringConfig()  # Default has 40% for 1H
        evaluator = ExitConditionEvaluator(config=config)
        call_position.timeframe = '1H'
        call_position.unrealized_pct = -0.45  # 45% loss, 1H threshold is 40%
        call_position.underlying_price = 448.0
        # Need to patch is_stale to avoid EOD check issues
        with patch.object(evaluator, '_is_stale_1h_position', return_value=False):
            result = evaluator.evaluate(call_position, underlying_price=448.0)
        assert result is not None
        assert result.reason == ExitReason.MAX_LOSS

    def test_timeframe_specific_max_loss_1m(self, call_position):
        """Test 1M uses wider max loss threshold."""
        config = MonitoringConfig()  # Default has 75% for 1M
        evaluator = ExitConditionEvaluator(config=config)
        call_position.timeframe = '1M'
        call_position.unrealized_pct = -0.60  # 60% loss, 1M threshold is 75%
        call_position.underlying_price = 448.0
        result = evaluator.evaluate(call_position, underlying_price=448.0)
        # Should NOT exit - loss below 75% threshold
        if result:
            assert result.reason != ExitReason.MAX_LOSS


# =============================================================================
# Test Max Profit
# =============================================================================


class TestMaxProfit:
    """Tests for max profit condition."""

    def test_max_profit_exceeded(self, evaluator, call_position):
        """Test exit when max profit exceeded."""
        call_position.unrealized_pct = 1.05  # 105% profit, default threshold is 100%
        call_position.underlying_price = 455.0  # Not at target
        result = evaluator.evaluate(call_position, underlying_price=455.0)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT
        assert '105.0%' in result.details

    def test_max_profit_exact(self, evaluator, call_position):
        """Test exit at exact max profit threshold."""
        call_position.unrealized_pct = 1.00
        call_position.underlying_price = 455.0
        result = evaluator.evaluate(call_position, underlying_price=455.0)
        assert result is not None
        assert result.reason == ExitReason.TARGET_HIT

    def test_max_profit_not_reached(self, evaluator, call_position):
        """Test no exit when profit below threshold."""
        call_position.unrealized_pct = 0.50  # 50% profit
        call_position.underlying_price = 455.0
        result = evaluator.evaluate(call_position, underlying_price=455.0)
        if result:
            assert 'target' not in result.details.lower() or 'profit' not in result.details.lower()


# =============================================================================
# Test Pattern Invalidation
# =============================================================================


class TestPatternInvalidation:
    """Tests for pattern invalidation (Type 3 evolution)."""

    def test_pattern_invalidated_call(self, evaluator, call_position):
        """Test pattern invalidation when call breaks both bounds."""
        # Setup: entry bar high=452, low=448
        # Intrabar breaks both: high=453, low=447
        call_position.intrabar_high = 453.0
        call_position.intrabar_low = 447.0
        result = evaluator.evaluate(call_position)
        assert result is not None
        assert result.reason == ExitReason.PATTERN_INVALIDATED
        assert 'Type 3' in result.details

    def test_pattern_not_invalidated_only_high_broken(self, evaluator, call_position):
        """Test no invalidation when only high broken."""
        call_position.intrabar_high = 453.0
        call_position.intrabar_low = 449.0  # Above entry_bar_low 448
        result = evaluator.evaluate(call_position)
        if result:
            assert result.reason != ExitReason.PATTERN_INVALIDATED

    def test_pattern_not_invalidated_only_low_broken(self, evaluator, call_position):
        """Test no invalidation when only low broken."""
        call_position.intrabar_high = 451.0  # Below entry_bar_high 452
        call_position.intrabar_low = 447.0
        result = evaluator.evaluate(call_position)
        if result:
            assert result.reason != ExitReason.PATTERN_INVALIDATED

    def test_pattern_invalidation_skipped_non_type2(self, evaluator, call_position):
        """Test pattern invalidation skipped for non-Type 2 entries."""
        call_position.entry_bar_type = '3'  # Not 2U or 2D
        call_position.intrabar_high = 453.0
        call_position.intrabar_low = 447.0
        result = evaluator.evaluate(call_position)
        if result:
            assert result.reason != ExitReason.PATTERN_INVALIDATED

    def test_pattern_invalidation_uses_bar_data_fallback(self, evaluator, call_position):
        """Test pattern invalidation uses bar_data when intrabar not available."""
        call_position.intrabar_high = 0.0  # Not initialized
        call_position.intrabar_low = float('inf')
        bar_data = {'high': 453.0, 'low': 447.0}
        result = evaluator.evaluate(call_position, bar_data=bar_data)
        assert result is not None
        assert result.reason == ExitReason.PATTERN_INVALIDATED


# =============================================================================
# Test Trailing Stop Checker Integration
# =============================================================================


class TestTrailingStopCheckerIntegration:
    """Tests for trailing stop checker delegation."""

    def test_calls_trailing_stop_checker(self, config, call_position):
        """Test that trailing stop checker is called."""
        mock_checker = Mock(spec=TrailingStopChecker)
        mock_checker.check.return_value = None
        evaluator = ExitConditionEvaluator(
            config=config,
            trailing_stop_checker=mock_checker,
        )
        evaluator.evaluate(call_position)
        mock_checker.check.assert_called_once_with(call_position)

    def test_returns_trailing_stop_signal(self, config, call_position):
        """Test that trailing stop signal is returned."""
        mock_signal = ExitSignal(
            osi_symbol=call_position.osi_symbol,
            signal_key=call_position.signal_key,
            reason=ExitReason.TRAILING_STOP,
            underlying_price=455.0,
            current_option_price=6.0,
            unrealized_pnl=100.0,
            dte=10,
            details='Trailing stop hit',
        )
        mock_checker = Mock(spec=TrailingStopChecker)
        mock_checker.check.return_value = mock_signal
        evaluator = ExitConditionEvaluator(
            config=config,
            trailing_stop_checker=mock_checker,
        )
        result = evaluator.evaluate(call_position)
        assert result is mock_signal

    def test_trailing_stop_not_called_if_no_checker(self, evaluator, call_position):
        """Test no error if trailing stop checker not set."""
        # Evaluator has no checker by default
        result = evaluator.evaluate(call_position)
        # Should not crash, just continue to other checks


# =============================================================================
# Test Partial Exit Checker Integration
# =============================================================================


class TestPartialExitCheckerIntegration:
    """Tests for partial exit checker delegation."""

    def test_calls_partial_exit_checker(self, config, call_position):
        """Test that partial exit checker is called."""
        mock_checker = Mock(spec=PartialExitChecker)
        mock_checker.check.return_value = None
        evaluator = ExitConditionEvaluator(
            config=config,
            partial_exit_checker=mock_checker,
        )
        evaluator.evaluate(call_position)
        mock_checker.check.assert_called_once_with(call_position)

    def test_returns_partial_exit_signal(self, config, call_position):
        """Test that partial exit signal is returned."""
        mock_signal = ExitSignal(
            osi_symbol=call_position.osi_symbol,
            signal_key=call_position.signal_key,
            reason=ExitReason.PARTIAL_EXIT,
            underlying_price=455.0,
            current_option_price=6.0,
            unrealized_pnl=100.0,
            dte=10,
            details='Partial exit',
            contracts_to_close=1,
        )
        mock_checker = Mock(spec=PartialExitChecker)
        mock_checker.check.return_value = mock_signal
        evaluator = ExitConditionEvaluator(
            config=config,
            partial_exit_checker=mock_checker,
        )
        result = evaluator.evaluate(call_position)
        assert result is mock_signal


# =============================================================================
# Test Exit Priority Order
# =============================================================================


class TestExitPriorityOrder:
    """Tests for correct exit condition priority order."""

    def test_dte_before_stop(self, evaluator, call_position):
        """Test DTE exit takes priority over stop hit."""
        call_position.expiration = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        call_position.underlying_price = 444.0  # At stop
        result = evaluator.evaluate(call_position, underlying_price=444.0)
        assert result.reason == ExitReason.DTE_EXIT

    def test_stop_before_max_loss(self, evaluator, call_position):
        """Test stop hit takes priority over max loss."""
        call_position.underlying_price = 444.0  # At stop
        call_position.unrealized_pct = -0.60  # Also max loss exceeded
        result = evaluator.evaluate(call_position, underlying_price=444.0)
        assert result.reason == ExitReason.STOP_HIT

    def test_max_loss_before_target(self, evaluator, call_position):
        """Test max loss takes priority over target."""
        call_position.underlying_price = 461.0  # At target
        call_position.unrealized_pct = -0.60  # Also max loss exceeded
        result = evaluator.evaluate(call_position, underlying_price=461.0)
        assert result.reason == ExitReason.MAX_LOSS

    def test_target_before_pattern_invalidation(self, evaluator, call_position):
        """Test target hit takes priority over pattern invalidation."""
        call_position.underlying_price = 461.0  # At target
        call_position.intrabar_high = 453.0  # Also pattern invalidated
        call_position.intrabar_low = 447.0
        result = evaluator.evaluate(call_position, underlying_price=461.0)
        assert result.reason == ExitReason.TARGET_HIT


# =============================================================================
# Test EOD Exit for 1H
# =============================================================================


class TestEODExit:
    """Tests for end-of-day exit for 1H trades."""

    def test_eod_exit_1h_after_cutoff(self, evaluator, call_position):
        """Test 1H position exits after EOD cutoff."""
        call_position.timeframe = '1H'
        # Mock time to be after 15:59 ET
        import pytz
        et = pytz.timezone('America/New_York')
        fake_now = datetime.now(et).replace(hour=16, minute=0)
        with patch('strat.signal_automation.coordinators.exit_evaluator.datetime') as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.strptime = datetime.strptime
            with patch.object(evaluator, '_is_stale_1h_position', return_value=False):
                result = evaluator._check_eod_exit(call_position)
        assert result is not None
        assert result.reason == ExitReason.EOD_EXIT

    def test_eod_exit_skipped_for_daily(self, evaluator, call_position):
        """Test EOD exit skipped for non-1H timeframes."""
        call_position.timeframe = '1D'
        result = evaluator._check_eod_exit(call_position)
        assert result is None

    def test_stale_1h_position_exits_immediately(self, evaluator, call_position):
        """Test stale 1H position (from previous day) exits immediately."""
        call_position.timeframe = '1H'
        # Use a weekday far enough back to guarantee multiple trading days
        call_position.entry_time = datetime.now() - timedelta(days=7)
        result = evaluator._check_eod_exit(call_position)
        assert result is not None
        assert result.reason == ExitReason.EOD_EXIT
        assert 'STALE' in result.details


# =============================================================================
# Test Calculate DTE
# =============================================================================


class TestCalculateDTE:
    """Tests for DTE calculation."""

    def test_calculate_dte_future(self, evaluator):
        """Test DTE calculation for future date."""
        future = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
        dte = evaluator._calculate_dte(future)
        # DTE can be 9 or 10 depending on time of day (date boundary)
        assert dte in [9, 10]

    def test_calculate_dte_today(self, evaluator):
        """Test DTE calculation for today."""
        today = datetime.now().strftime('%Y-%m-%d')
        dte = evaluator._calculate_dte(today)
        assert dte == 0

    def test_calculate_dte_past(self, evaluator):
        """Test DTE calculation for past date."""
        past = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        dte = evaluator._calculate_dte(past)
        assert dte == 0  # Should return 0, not negative

    def test_calculate_dte_empty(self, evaluator):
        """Test DTE calculation for empty string."""
        dte = evaluator._calculate_dte('')
        assert dte == 0

    def test_calculate_dte_invalid(self, evaluator):
        """Test DTE calculation for invalid format."""
        dte = evaluator._calculate_dte('not-a-date')
        assert dte == 0


# =============================================================================
# Test Underlying Price Update
# =============================================================================


class TestUnderlyingPriceUpdate:
    """Tests for underlying price updates during evaluation."""

    def test_updates_underlying_price(self, evaluator, call_position):
        """Test that underlying price is updated from parameter."""
        assert call_position.underlying_price == 455.0
        evaluator.evaluate(call_position, underlying_price=460.0)
        assert call_position.underlying_price == 460.0

    def test_updates_intrabar_high(self, evaluator, call_position):
        """Test that intrabar high is updated."""
        call_position.intrabar_high = 455.0
        evaluator.evaluate(call_position, underlying_price=458.0)
        assert call_position.intrabar_high == 458.0

    def test_updates_intrabar_low(self, evaluator, call_position):
        """Test that intrabar low is updated."""
        call_position.intrabar_low = 449.0
        evaluator.evaluate(call_position, underlying_price=447.0)
        assert call_position.intrabar_low == 447.0

    def test_does_not_update_if_not_extreme(self, evaluator, call_position):
        """Test that non-extreme prices don't update intrabar."""
        call_position.intrabar_high = 455.0
        call_position.intrabar_low = 449.0
        evaluator.evaluate(call_position, underlying_price=452.0)
        assert call_position.intrabar_high == 455.0  # Unchanged
        assert call_position.intrabar_low == 449.0   # Unchanged
