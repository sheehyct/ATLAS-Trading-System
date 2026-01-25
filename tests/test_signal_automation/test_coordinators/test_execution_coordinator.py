"""
EQUITY-88: Tests for ExecutionCoordinator.

Tests signal execution logic extracted from SignalDaemon.
Covers triggered pattern execution, TFC re-evaluation, and intraday timing filters.
"""

import pytest
from datetime import datetime, time as dt_time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from strat.signal_automation.coordinators.execution_coordinator import ExecutionCoordinator
from strat.signal_automation.executor import ExecutionResult, ExecutionState


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockExecutionConfig:
    """Mock execution config for testing."""
    enabled: bool = True
    tfc_reeval_enabled: bool = True
    tfc_reeval_min_strength: int = 3
    tfc_reeval_block_on_flip: bool = True
    tfc_reeval_log_always: bool = True


@dataclass
class MockStoredSignal:
    """Mock stored signal for testing."""
    signal_key: str = "SPY_1H_3-1-2U_CALL"
    symbol: str = "SPY"
    timeframe: str = "1H"
    pattern_type: str = "3-1-2U"
    direction: str = "CALL"
    status: str = "ALERTED"
    signal_type: str = "COMPLETED"
    entry_trigger: float = 580.0
    stop_price: float = 575.0
    target_price: float = 590.0
    tfc_score: Optional[int] = 3
    tfc_alignment: Optional[str] = "3/4 BULLISH"
    passes_flexible: Optional[bool] = True
    priority: int = 1


@dataclass
class MockTFCAssessment:
    """Mock TFC assessment for testing."""
    strength: int = 3
    direction: str = "bullish"
    passes_flexible: bool = True

    def alignment_label(self) -> str:
        return f"{self.strength}/4 {self.direction.upper()}"


@pytest.fixture
def mock_config():
    """Create mock execution config."""
    return MockExecutionConfig()


@pytest.fixture
def mock_executor():
    """Create mock executor."""
    executor = Mock()
    executor._trading_client = Mock()
    executor._trading_client.get_stock_quotes.return_value = {
        'SPY': {'mid': 585.0}
    }
    executor.execute_signal.return_value = ExecutionResult(
        signal_key="SPY_1H_3-1-2U_CALL",
        state=ExecutionState.ORDER_SUBMITTED,
        osi_symbol="SPY250131C00580000"
    )
    return executor


@pytest.fixture
def mock_signal_store():
    """Create mock signal store."""
    store = Mock()
    return store


@pytest.fixture
def mock_tfc_evaluator():
    """Create mock TFC evaluator."""
    evaluator = Mock()
    evaluator.evaluate_tfc.return_value = MockTFCAssessment(
        strength=3,
        direction="bullish",
        passes_flexible=True
    )
    return evaluator


@pytest.fixture
def mock_alerters():
    """Create mock alerters list."""
    discord_alerter = Mock()
    discord_alerter.__class__.__name__ = 'DiscordAlerter'
    logging_alerter = Mock()
    logging_alerter.__class__.__name__ = 'LoggingAlerter'
    return [discord_alerter, logging_alerter]


@pytest.fixture
def coordinator(mock_config, mock_executor, mock_signal_store, mock_tfc_evaluator, mock_alerters):
    """Create ExecutionCoordinator with mock dependencies."""
    return ExecutionCoordinator(
        config=mock_config,
        executor=mock_executor,
        signal_store=mock_signal_store,
        tfc_evaluator=mock_tfc_evaluator,
        alerters=mock_alerters,
        on_execution=Mock(),
        on_error=Mock(),
    )


@pytest.fixture
def basic_signal():
    """Create a basic signal for testing."""
    return MockStoredSignal()


# =============================================================================
# ExecutionCoordinator Initialization Tests
# =============================================================================


class TestExecutionCoordinatorInit:
    """Tests for ExecutionCoordinator initialization."""

    def test_init_with_all_dependencies(self, mock_config, mock_executor):
        """Test initialization with all dependencies."""
        coord = ExecutionCoordinator(
            config=mock_config,
            executor=mock_executor,
        )
        assert coord.config == mock_config
        assert coord.executor == mock_executor

    def test_init_with_minimal_dependencies(self, mock_config):
        """Test initialization with minimal dependencies."""
        coord = ExecutionCoordinator(config=mock_config)
        assert coord.config == mock_config
        assert coord.executor is None
        assert coord._alerters == []

    def test_set_executor(self, mock_config, mock_executor):
        """Test setting executor after initialization."""
        coord = ExecutionCoordinator(config=mock_config)
        assert coord.executor is None
        coord.set_executor(mock_executor)
        assert coord.executor == mock_executor

    def test_set_tfc_evaluator(self, mock_config, mock_tfc_evaluator):
        """Test setting TFC evaluator after initialization."""
        coord = ExecutionCoordinator(config=mock_config)
        coord.set_tfc_evaluator(mock_tfc_evaluator)
        assert coord._tfc_evaluator == mock_tfc_evaluator


# =============================================================================
# Get Current Price Tests
# =============================================================================


class TestGetCurrentPrice:
    """Tests for _get_current_price method."""

    def test_get_price_success(self, coordinator):
        """Test successful price fetch."""
        price = coordinator._get_current_price('SPY')
        assert price == 585.0

    def test_get_price_no_executor(self, mock_config):
        """Test price fetch with no executor."""
        coord = ExecutionCoordinator(config=mock_config)
        price = coord._get_current_price('SPY')
        assert price is None

    def test_get_price_dict_format(self, coordinator, mock_executor):
        """Test price fetch with dict response format."""
        mock_executor._trading_client.get_stock_quotes.return_value = {
            'AAPL': {'mid': 195.50}
        }
        price = coordinator._get_current_price('AAPL')
        assert price == 195.50

    def test_get_price_numeric_format(self, coordinator, mock_executor):
        """Test price fetch with numeric response format."""
        mock_executor._trading_client.get_stock_quotes.return_value = {
            'NVDA': 950.25
        }
        price = coordinator._get_current_price('NVDA')
        assert price == 950.25

    def test_get_price_symbol_not_found(self, coordinator, mock_executor):
        """Test price fetch for missing symbol."""
        mock_executor._trading_client.get_stock_quotes.return_value = {}
        price = coordinator._get_current_price('UNKNOWN')
        assert price is None

    def test_get_price_exception(self, coordinator, mock_executor):
        """Test price fetch with exception."""
        mock_executor._trading_client.get_stock_quotes.side_effect = Exception("API error")
        price = coordinator._get_current_price('SPY')
        assert price is None


# =============================================================================
# Intraday Entry Allowed Tests
# =============================================================================


class TestIsIntradayEntryAllowed:
    """Tests for is_intraday_entry_allowed method."""

    def test_daily_pattern_always_allowed(self, coordinator, basic_signal):
        """Test daily patterns have no time restriction."""
        basic_signal.timeframe = '1D'
        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    def test_weekly_pattern_always_allowed(self, coordinator, basic_signal):
        """Test weekly patterns have no time restriction."""
        basic_signal.timeframe = '1W'
        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    def test_monthly_pattern_always_allowed(self, coordinator, basic_signal):
        """Test monthly patterns have no time restriction."""
        basic_signal.timeframe = '1M'
        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_1h_2bar_before_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 1H 2-bar pattern blocked before 10:30."""
        # Set current time to 10:00 ET (before 10:30 threshold)
        mock_now = Mock()
        mock_now.time.return_value = dt_time(10, 0)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '1H'
        basic_signal.pattern_type = '3-2U'  # 2-bar pattern

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is False

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_1h_2bar_after_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 1H 2-bar pattern allowed after 10:30."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(11, 0)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '1H'
        basic_signal.pattern_type = '3-2U'

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_1h_3bar_before_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 1H 3-bar pattern blocked before 11:30."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(11, 0)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '1H'
        basic_signal.pattern_type = '3-1-2U'  # 3-bar pattern

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is False

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_1h_3bar_after_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 1H 3-bar pattern allowed after 11:30."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(12, 0)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '1H'
        basic_signal.pattern_type = '3-1-2U'

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_30m_2bar_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 30m 2-bar pattern threshold at 10:00."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(10, 15)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '30m'
        basic_signal.pattern_type = '2D-2U'

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_15m_3bar_threshold(self, mock_datetime, coordinator, basic_signal):
        """Test 15m 3-bar pattern threshold at 10:00."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(10, 15)
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '15m'
        basic_signal.pattern_type = '2D-1-2U'

        result = coordinator.is_intraday_entry_allowed(basic_signal)
        assert result is True


# =============================================================================
# TFC Re-evaluation Tests
# =============================================================================


class TestReevaluateTFCAtEntry:
    """Tests for reevaluate_tfc_at_entry method."""

    def test_tfc_disabled_returns_false(self, coordinator, basic_signal):
        """Test that disabled TFC re-eval always returns False (don't block)."""
        coordinator._config.tfc_reeval_enabled = False
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False
        assert reason == ""

    def test_tfc_no_evaluator_returns_false(self, mock_config, basic_signal):
        """Test that missing TFC evaluator returns False (don't block)."""
        coord = ExecutionCoordinator(config=mock_config)
        blocked, reason = coord.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_unchanged_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test signal not blocked when TFC unchanged."""
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=3, direction="bullish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_improved_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test signal not blocked when TFC improved."""
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=4, direction="bullish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_degraded_below_min_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test signal blocked when TFC drops below minimum."""
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=2, direction="bullish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is True
        assert "TFC strength 2 < min threshold 3" in reason

    def test_tfc_direction_flip_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test signal blocked when TFC direction flips."""
        basic_signal.tfc_alignment = "3/4 BULLISH"
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=3, direction="bearish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is True
        assert "direction flipped" in reason

    def test_tfc_flip_allowed_when_disabled(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test direction flip not blocked when flip blocking disabled."""
        coordinator._config.tfc_reeval_block_on_flip = False
        basic_signal.tfc_alignment = "3/4 BULLISH"
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=3, direction="bearish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_connection_error_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test connection error fails open (doesn't block)."""
        mock_tfc_evaluator.evaluate_tfc.side_effect = ConnectionError("Network error")
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False
        coordinator._on_error.assert_called_once()

    def test_tfc_timeout_error_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test timeout error fails open (doesn't block)."""
        mock_tfc_evaluator.evaluate_tfc.side_effect = TimeoutError("Timeout")
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_unexpected_error_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test unexpected error fails open (doesn't block)."""
        mock_tfc_evaluator.evaluate_tfc.side_effect = RuntimeError("Unexpected")
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_invalid_assessment_not_blocked(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test invalid assessment fails open (doesn't block)."""
        mock_tfc_evaluator.evaluate_tfc.return_value = None
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False

    def test_tfc_missing_original_alignment(self, coordinator, basic_signal, mock_tfc_evaluator):
        """Test handling of missing original TFC alignment."""
        basic_signal.tfc_alignment = None
        mock_tfc_evaluator.evaluate_tfc.return_value = MockTFCAssessment(
            strength=3, direction="bullish"
        )
        blocked, reason = coordinator.reevaluate_tfc_at_entry(basic_signal)
        assert blocked is False  # Can't detect flip, shouldn't block


# =============================================================================
# Execute Triggered Pattern Tests
# =============================================================================


class TestExecuteTriggeredPattern:
    """Tests for execute_triggered_pattern method."""

    def test_no_executor_returns_none(self, mock_config, basic_signal):
        """Test that missing executor returns None."""
        coord = ExecutionCoordinator(config=mock_config)
        result = coord.execute_triggered_pattern(basic_signal)
        assert result is None

    def test_no_price_returns_none(self, coordinator, basic_signal, mock_executor):
        """Test that missing price returns None."""
        mock_executor._trading_client.get_stock_quotes.return_value = {}
        result = coordinator.execute_triggered_pattern(basic_signal)
        assert result is None

    def test_call_past_target_skipped(self, coordinator, basic_signal, mock_executor):
        """Test CALL skipped when price past target."""
        mock_executor._trading_client.get_stock_quotes.return_value = {
            'SPY': {'mid': 595.0}  # Past target of 590
        }
        basic_signal.direction = "CALL"
        basic_signal.target_price = 590.0
        result = coordinator.execute_triggered_pattern(basic_signal)
        assert result is None

    def test_put_past_target_skipped(self, coordinator, basic_signal, mock_executor):
        """Test PUT skipped when price past target."""
        mock_executor._trading_client.get_stock_quotes.return_value = {
            'SPY': {'mid': 560.0}  # Past target of 570
        }
        basic_signal.direction = "PUT"
        basic_signal.target_price = 570.0
        result = coordinator.execute_triggered_pattern(basic_signal)
        assert result is None

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_successful_execution(self, mock_datetime, coordinator, basic_signal, mock_executor):
        """Test successful triggered pattern execution."""
        # Set time after threshold
        mock_now = Mock()
        mock_now.time.return_value = dt_time(14, 0)
        mock_datetime.now.return_value = mock_now

        result = coordinator.execute_triggered_pattern(basic_signal)

        assert result is not None
        assert result.state == ExecutionState.ORDER_SUBMITTED
        coordinator._on_execution.assert_called_once()

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_signal_store_updated(self, mock_datetime, coordinator, basic_signal, mock_signal_store):
        """Test signal store is updated on successful execution."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(14, 0)
        mock_datetime.now.return_value = mock_now

        result = coordinator.execute_triggered_pattern(basic_signal)

        mock_signal_store.mark_triggered.assert_called_once_with(basic_signal.signal_key)
        mock_signal_store.set_executed_osi_symbol.assert_called_once()

    @patch('strat.signal_automation.coordinators.execution_coordinator.datetime')
    def test_intraday_timing_blocked(self, mock_datetime, coordinator, basic_signal):
        """Test intraday pattern blocked before threshold."""
        mock_now = Mock()
        mock_now.time.return_value = dt_time(9, 45)  # Before 11:30 for 3-bar
        mock_datetime.now.return_value = mock_now

        basic_signal.timeframe = '1H'
        basic_signal.pattern_type = '3-1-2U'

        result = coordinator.execute_triggered_pattern(basic_signal)
        assert result is None


# =============================================================================
# Execute Signals Tests
# =============================================================================


class TestExecuteSignals:
    """Tests for execute_signals method."""

    def test_no_executor_returns_empty(self, mock_config):
        """Test that missing executor returns empty list."""
        coord = ExecutionCoordinator(config=mock_config)
        result = coord.execute_signals([MockStoredSignal()])
        assert result == []

    def test_setup_signals_skipped(self, coordinator, basic_signal):
        """Test SETUP signals are skipped (wait for entry monitor)."""
        basic_signal.signal_type = 'SETUP'
        results = coordinator.execute_signals([basic_signal])
        assert len(results) == 0

    def test_completed_signals_skipped(self, coordinator, basic_signal):
        """Test COMPLETED signals are skipped (handled by execute_triggered_pattern)."""
        basic_signal.signal_type = 'COMPLETED'
        results = coordinator.execute_signals([basic_signal])
        assert len(results) == 0

    def test_execution_count_incremented(self, coordinator, basic_signal, mock_executor):
        """Test execution count callback called on success."""
        basic_signal.signal_type = 'OTHER'  # Not SETUP or COMPLETED
        mock_executor.execute_signal.return_value = ExecutionResult(
            signal_key=basic_signal.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
            osi_symbol="SPY250131C00580000"
        )

        with patch.object(coordinator, 'is_intraday_entry_allowed', return_value=True):
            results = coordinator.execute_signals([basic_signal])

        assert len(results) == 1
        coordinator._on_execution.assert_called()

    def test_error_count_incremented_on_failure(self, coordinator, basic_signal, mock_executor):
        """Test error count callback called on failure."""
        basic_signal.signal_type = 'OTHER'
        mock_executor.execute_signal.return_value = ExecutionResult(
            signal_key=basic_signal.signal_key,
            state=ExecutionState.FAILED,
            error="Order rejected"
        )

        with patch.object(coordinator, 'is_intraday_entry_allowed', return_value=True):
            results = coordinator.execute_signals([basic_signal])

        assert len(results) == 1
        coordinator._on_error.assert_called()

    def test_exception_handled(self, coordinator, basic_signal, mock_executor):
        """Test exception is caught and result returned."""
        basic_signal.signal_type = 'OTHER'
        mock_executor.execute_signal.side_effect = Exception("API Error")

        with patch.object(coordinator, 'is_intraday_entry_allowed', return_value=True):
            results = coordinator.execute_signals([basic_signal])

        assert len(results) == 1
        assert results[0].state == ExecutionState.FAILED
        assert "API Error" in results[0].error


# =============================================================================
# Send Entry Alerts Tests
# =============================================================================


class TestSendEntryAlerts:
    """Tests for _send_entry_alerts method."""

    def test_alerts_sent_to_all_alerters(self, coordinator, basic_signal, mock_alerters):
        """Test alerts are sent to all alerters."""
        result = ExecutionResult(
            signal_key=basic_signal.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
            osi_symbol="SPY250131C00580000"
        )

        # Mock isinstance checks
        with patch('strat.signal_automation.coordinators.execution_coordinator.isinstance') as mock_isinstance:
            # First call checks DiscordAlerter, second checks LoggingAlerter
            mock_isinstance.side_effect = lambda obj, cls: cls.__name__ in str(type(obj))
            coordinator._send_entry_alerts(basic_signal, result)

    def test_alert_error_caught(self, coordinator, basic_signal, mock_alerters):
        """Test alert errors are caught and logged."""
        result = ExecutionResult(
            signal_key=basic_signal.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
            osi_symbol="SPY250131C00580000"
        )

        # Make first alerter raise exception
        mock_alerters[0].send_entry_alert.side_effect = Exception("Discord error")

        # Should not raise
        coordinator._send_entry_alerts(basic_signal, result)


# =============================================================================
# Config Property Tests
# =============================================================================


class TestConfigProperty:
    """Tests for config property."""

    def test_config_property_returns_config(self, coordinator, mock_config):
        """Test config property returns the config."""
        assert coordinator.config == mock_config

    def test_executor_property(self, coordinator, mock_executor):
        """Test executor property returns the executor."""
        assert coordinator.executor == mock_executor
