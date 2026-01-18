"""
Tests for SignalExecutor - Session EQUITY-71

Comprehensive test coverage for strat/signal_automation/executor.py including:
- ExecutionState enum
- ExecutionResult dataclass (to_dict, from_dict)
- ExecutorConfig defaults
- SignalExecutor initialization, connection, persistence
- Signal execution flow (filters, limits, orders)
- Position sizing with TFC risk multipliers
- Helper methods
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from strat.signal_automation.executor import (
    ExecutionState,
    ExecutionResult,
    ExecutorConfig,
    SignalExecutor,
    create_paper_executor,
)
from strat.signal_automation.signal_store import StoredSignal, SignalStatus, SignalType
from strat.options_module import OptionContract, OptionType


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for persistence tests."""
    data_dir = tmp_path / 'executions'
    data_dir.mkdir(parents=True)
    return str(data_dir)


@pytest.fixture
def mock_trading_client():
    """Create a mock Alpaca trading client."""
    client = Mock()
    client.connect.return_value = True
    client.get_account.return_value = {
        'id': 'test_account',
        'equity': 10000.0,
        'buying_power': 10000.0,
    }
    client.list_option_positions.return_value = []
    client.get_stock_price.return_value = 600.0
    client.get_position.return_value = {'current_price': 600.0}
    client.get_option_contracts.return_value = [
        {
            'symbol': 'SPY241220C00600000',
            'strike': 600.0,
            'expiration': '2024-12-20',
        }
    ]
    client.submit_option_limit_order.return_value = {'id': 'order-123'}
    client.submit_option_market_order.return_value = {'id': 'order-456'}
    client.get_order.return_value = {'id': 'order-123', 'status': 'filled'}
    client.close_option_position.return_value = {'id': 'close-order-789'}
    return client


@pytest.fixture
def sample_stored_signal():
    """Create a sample StoredSignal for testing."""
    return StoredSignal(
        signal_key='SPY_1D_2-1-2U_20241206',
        pattern_type='2-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime(2024, 12, 6, 10, 0),
        entry_trigger=600.50,
        stop_price=595.00,
        target_price=610.00,
        magnitude_pct=1.58,
        risk_reward=1.73,
        vix=15.5,
        status=SignalStatus.DETECTED.value,
        signal_type=SignalType.COMPLETED.value,
        passes_flexible=True,
        risk_multiplier=1.0,
        setup_bar_high=602.0,
        setup_bar_low=598.0,
    )


@pytest.fixture
def sample_setup_signal():
    """Create a SETUP signal (awaiting break) for testing."""
    return StoredSignal(
        signal_key='AAPL_1H_3-1-?_20241206',
        pattern_type='3-1-?',
        direction='CALL',
        symbol='AAPL',
        timeframe='1H',
        detected_time=datetime(2024, 12, 6, 14, 0),
        entry_trigger=195.00,
        stop_price=193.50,
        target_price=197.50,
        magnitude_pct=0.15,  # Low magnitude typical for SETUP
        risk_reward=0.5,     # Low R:R typical for SETUP
        status=SignalStatus.DETECTED.value,
        signal_type=SignalType.SETUP.value,
        passes_flexible=True,
    )


@pytest.fixture
def executor_config(temp_data_dir):
    """Create executor config with temp directory."""
    return ExecutorConfig(
        account='SMALL',
        max_capital_per_trade=300.0,
        max_concurrent_positions=5,
        persistence_path=temp_data_dir,
        paper_mode=True,
    )


# =============================================================================
# TEST EXECUTION STATE ENUM
# =============================================================================


class TestExecutionState:
    """Tests for ExecutionState enum."""

    def test_all_states_exist(self):
        """Verify all expected states are defined."""
        expected_states = [
            'PENDING', 'ORDER_SUBMITTED', 'ORDER_FILLED',
            'MONITORING', 'CLOSED', 'FAILED', 'SKIPPED'
        ]
        for state_name in expected_states:
            assert hasattr(ExecutionState, state_name)

    def test_state_values(self):
        """Verify state values are lowercase strings."""
        assert ExecutionState.PENDING.value == 'pending'
        assert ExecutionState.ORDER_SUBMITTED.value == 'submitted'
        assert ExecutionState.ORDER_FILLED.value == 'filled'
        assert ExecutionState.MONITORING.value == 'monitoring'
        assert ExecutionState.CLOSED.value == 'closed'
        assert ExecutionState.FAILED.value == 'failed'
        assert ExecutionState.SKIPPED.value == 'skipped'


# =============================================================================
# TEST EXECUTION RESULT DATACLASS
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_default_values(self):
        """Test default field values."""
        result = ExecutionResult(
            signal_key='test_key',
            state=ExecutionState.PENDING,
        )
        assert result.signal_key == 'test_key'
        assert result.state == ExecutionState.PENDING
        assert result.order_id is None
        assert result.osi_symbol is None
        assert result.strike is None
        assert result.expiration is None
        assert result.contracts == 0
        assert result.premium == 0.0
        assert result.side == ''
        assert result.error is None
        assert result.underlying_entry_price is None
        assert result.entry_bar_type == ''
        assert result.entry_bar_high == 0.0
        assert result.entry_bar_low == 0.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = ExecutionResult(
            signal_key='SPY_1D_2-1-2U_20241206',
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='order-123',
            osi_symbol='SPY241220C00600000',
            strike=600.0,
            expiration='2024-12-20',
            contracts=2,
            premium=5.50,
            side='buy',
            timestamp=datetime(2024, 12, 6, 10, 30),
            underlying_entry_price=601.25,
            entry_bar_type='2U',
            entry_bar_high=602.0,
            entry_bar_low=598.0,
        )
        d = result.to_dict()

        assert d['signal_key'] == 'SPY_1D_2-1-2U_20241206'
        assert d['state'] == 'submitted'
        assert d['order_id'] == 'order-123'
        assert d['osi_symbol'] == 'SPY241220C00600000'
        assert d['strike'] == 600.0
        assert d['expiration'] == '2024-12-20'
        assert d['contracts'] == 2
        assert d['premium'] == 5.50
        assert d['side'] == 'buy'
        assert d['timestamp'] == '2024-12-06T10:30:00'
        assert d['underlying_entry_price'] == 601.25
        assert d['entry_bar_type'] == '2U'
        assert d['entry_bar_high'] == 602.0
        assert d['entry_bar_low'] == 598.0

    def test_from_dict(self):
        """Test from_dict deserialization."""
        d = {
            'signal_key': 'SPY_1D_2-1-2U_20241206',
            'state': 'submitted',
            'order_id': 'order-123',
            'osi_symbol': 'SPY241220C00600000',
            'strike': 600.0,
            'expiration': '2024-12-20',
            'contracts': 2,
            'premium': 5.50,
            'side': 'buy',
            'timestamp': '2024-12-06T10:30:00',
            'underlying_entry_price': 601.25,
            'entry_bar_type': '2U',
            'entry_bar_high': 602.0,
            'entry_bar_low': 598.0,
        }
        result = ExecutionResult.from_dict(d)

        assert result.signal_key == 'SPY_1D_2-1-2U_20241206'
        assert result.state == ExecutionState.ORDER_SUBMITTED
        assert result.order_id == 'order-123'
        assert result.osi_symbol == 'SPY241220C00600000'
        assert result.strike == 600.0
        assert result.contracts == 2
        assert result.premium == 5.50
        assert result.timestamp == datetime(2024, 12, 6, 10, 30)
        assert result.underlying_entry_price == 601.25
        assert result.entry_bar_type == '2U'
        assert result.entry_bar_high == 602.0
        assert result.entry_bar_low == 598.0

    def test_from_dict_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        d = {
            'signal_key': 'test_key',
            'state': 'pending',
        }
        result = ExecutionResult.from_dict(d)

        assert result.signal_key == 'test_key'
        assert result.state == ExecutionState.PENDING
        assert result.order_id is None
        assert result.contracts == 0
        assert result.premium == 0.0
        assert result.entry_bar_type == ''

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict preserves data."""
        original = ExecutionResult(
            signal_key='AAPL_1H_3-1-2U_20241206',
            state=ExecutionState.ORDER_FILLED,
            order_id='order-abc',
            osi_symbol='AAPL241220C00195000',
            strike=195.0,
            expiration='2024-12-20',
            contracts=3,
            premium=4.25,
            side='buy',
            underlying_entry_price=195.50,
            entry_bar_type='2U',
            entry_bar_high=196.0,
            entry_bar_low=194.0,
        )
        d = original.to_dict()
        restored = ExecutionResult.from_dict(d)

        assert restored.signal_key == original.signal_key
        assert restored.state == original.state
        assert restored.order_id == original.order_id
        assert restored.osi_symbol == original.osi_symbol
        assert restored.strike == original.strike
        assert restored.contracts == original.contracts
        assert restored.premium == original.premium
        assert restored.underlying_entry_price == original.underlying_entry_price
        assert restored.entry_bar_type == original.entry_bar_type


# =============================================================================
# TEST EXECUTOR CONFIG
# =============================================================================


class TestExecutorConfig:
    """Tests for ExecutorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExecutorConfig()

        assert config.account == 'SMALL'
        assert config.max_capital_per_trade == 300.0
        assert config.max_concurrent_positions == 5
        assert config.target_delta == 0.55
        assert config.delta_range_min == 0.45
        assert config.delta_range_max == 0.65
        assert config.min_dte == 7
        assert config.max_dte == 21
        assert config.target_dte == 14
        assert config.min_magnitude_pct == 0.5
        assert config.min_risk_reward == 1.0
        assert config.max_bid_ask_spread_pct == 0.10
        assert config.use_limit_orders is True
        assert config.limit_price_buffer == 0.02
        assert config.paper_mode is True
        assert config.persistence_path == 'data/executions'

    def test_custom_values(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            account='LARGE',
            max_capital_per_trade=500.0,
            max_concurrent_positions=10,
            min_dte=14,
            max_dte=30,
        )

        assert config.account == 'LARGE'
        assert config.max_capital_per_trade == 500.0
        assert config.max_concurrent_positions == 10
        assert config.min_dte == 14
        assert config.max_dte == 30


# =============================================================================
# TEST SIGNAL EXECUTOR INITIALIZATION
# =============================================================================


class TestSignalExecutorInit:
    """Tests for SignalExecutor initialization."""

    def test_init_with_defaults(self, temp_data_dir):
        """Test initialization with default config."""
        config = ExecutorConfig(persistence_path=temp_data_dir)
        executor = SignalExecutor(config=config)

        assert executor.config.account == 'SMALL'
        assert executor._connected is False
        assert executor._executions == {}

    def test_init_with_custom_config(self, temp_data_dir):
        """Test initialization with custom config."""
        config = ExecutorConfig(
            account='LARGE',
            max_concurrent_positions=10,
            persistence_path=temp_data_dir,
        )
        executor = SignalExecutor(config=config)

        assert executor.config.account == 'LARGE'
        assert executor.config.max_concurrent_positions == 10

    def test_init_with_injected_client(self, temp_data_dir, mock_trading_client):
        """Test initialization with injected trading client."""
        config = ExecutorConfig(persistence_path=temp_data_dir)
        executor = SignalExecutor(
            config=config,
            trading_client=mock_trading_client,
        )

        assert executor._trading_client is mock_trading_client

    def test_init_creates_persistence_directory(self, tmp_path):
        """Test that init creates persistence directory if missing."""
        persist_dir = tmp_path / 'new_executions'
        config = ExecutorConfig(persistence_path=str(persist_dir))

        executor = SignalExecutor(config=config)

        assert persist_dir.exists()


# =============================================================================
# TEST SIGNAL EXECUTOR CONNECTION
# =============================================================================


class TestSignalExecutorConnect:
    """Tests for SignalExecutor.connect()."""

    def test_connect_success(self, executor_config, mock_trading_client):
        """Test successful connection."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        result = executor.connect()

        assert result is True
        assert executor.is_connected is True

    def test_connect_failure(self, executor_config, mock_trading_client):
        """Test connection failure."""
        mock_trading_client.connect.return_value = False
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        result = executor.connect()

        assert result is False
        assert executor.is_connected is False

    def test_connect_creates_client_if_none(self, executor_config):
        """Test that connect creates client if not injected."""
        executor = SignalExecutor(config=executor_config)

        with patch('strat.signal_automation.executor.AlpacaTradingClient') as MockClient:
            mock_instance = Mock()
            mock_instance.connect.return_value = True
            MockClient.return_value = mock_instance

            result = executor.connect()

            MockClient.assert_called_once_with(account='SMALL')
            assert result is True


# =============================================================================
# TEST PERSISTENCE
# =============================================================================


class TestSignalExecutorPersistence:
    """Tests for SignalExecutor persistence (load/save)."""

    def test_save_and_load(self, executor_config, mock_trading_client):
        """Test saving and loading executions."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        # Add execution
        result = ExecutionResult(
            signal_key='test_key',
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='order-123',
        )
        executor._executions['test_key'] = result
        executor._save()

        # Create new executor and verify load
        executor2 = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert 'test_key' in executor2._executions
        loaded = executor2._executions['test_key']
        assert loaded.state == ExecutionState.ORDER_SUBMITTED
        assert loaded.order_id == 'order-123'

    def test_load_empty_file(self, executor_config):
        """Test loading when no file exists."""
        executor = SignalExecutor(config=executor_config)
        assert executor._executions == {}

    def test_load_corrupted_file(self, executor_config):
        """Test loading corrupted JSON file."""
        # Write corrupted data
        executions_file = Path(executor_config.persistence_path) / 'executions.json'
        with open(executions_file, 'w') as f:
            f.write('not valid json')

        executor = SignalExecutor(config=executor_config)
        assert executor._executions == {}

    def test_save_multiple_executions(self, executor_config, mock_trading_client):
        """Test saving multiple executions."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        # Add multiple executions
        for i in range(3):
            result = ExecutionResult(
                signal_key=f'key_{i}',
                state=ExecutionState.PENDING,
            )
            executor._executions[f'key_{i}'] = result

        executor._save()

        # Verify file content
        executions_file = Path(executor_config.persistence_path) / 'executions.json'
        with open(executions_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 3
        assert 'key_0' in data
        assert 'key_1' in data
        assert 'key_2' in data


# =============================================================================
# TEST EXECUTE SIGNAL
# =============================================================================


class TestExecuteSignal:
    """Tests for SignalExecutor.execute_signal()."""

    def test_execute_not_connected(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution fails when not connected."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        # Don't call connect()

        result = executor.execute_signal(sample_stored_signal)

        assert result.state == ExecutionState.FAILED
        assert 'Not connected' in result.error

    def test_execute_already_executed(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test duplicate execution is prevented."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        # Execute once
        executor._executions[sample_stored_signal.signal_key] = ExecutionResult(
            signal_key=sample_stored_signal.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
        )

        # Try to execute again
        result = executor.execute_signal(sample_stored_signal)

        assert result.state == ExecutionState.ORDER_SUBMITTED  # Returns existing

    def test_execute_historical_triggered_skipped(self, executor_config, mock_trading_client):
        """Test HISTORICAL_TRIGGERED signals are skipped."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        signal = StoredSignal(
            signal_key='SPY_1D_2-1-2U_20241206',
            pattern_type='2-1-2U',
            direction='CALL',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=600.0,
            stop_price=595.0,
            target_price=610.0,
            magnitude_pct=1.5,
            risk_reward=1.7,
            status='HISTORICAL_TRIGGERED',  # Already completed
        )

        result = executor.execute_signal(signal)

        assert result.state == ExecutionState.SKIPPED
        assert 'HISTORICAL_TRIGGERED' in result.error

    def test_execute_fails_flexible_check(self, executor_config, mock_trading_client):
        """Test signals failing flexible continuity are skipped."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        signal = StoredSignal(
            signal_key='SPY_1D_2-1-2U_20241206',
            pattern_type='2-1-2U',
            direction='CALL',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=600.0,
            stop_price=595.0,
            target_price=610.0,
            magnitude_pct=1.5,
            risk_reward=1.7,
            passes_flexible=False,  # Fails TFC check
        )

        result = executor.execute_signal(signal)

        assert result.state == ExecutionState.SKIPPED
        assert 'flexible continuity' in result.error

    def test_execute_max_positions_reached(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution skipped when max positions reached."""
        mock_trading_client.list_option_positions.return_value = [
            {'symbol': f'POS{i}'} for i in range(5)
        ]
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(sample_stored_signal)

        assert result.state == ExecutionState.SKIPPED
        assert 'Max positions' in result.error

    def test_execute_price_fetch_failure(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution fails when price fetch fails."""
        mock_trading_client.get_stock_price.return_value = None
        mock_trading_client.get_position.return_value = None
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(sample_stored_signal)

        assert result.state == ExecutionState.FAILED
        assert 'Could not get price' in result.error

    def test_execute_no_contract_found(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution fails when no contract found."""
        mock_trading_client.get_option_contracts.return_value = []
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(sample_stored_signal)

        assert result.state == ExecutionState.FAILED
        assert 'No suitable contract' in result.error

    def test_execute_success_limit_order(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test successful execution with limit order."""
        # Mock _get_option_price to return a price so limit order is used
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()
        executor._get_option_price = Mock(return_value=5.50)  # Return option price

        result = executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        assert result.state == ExecutionState.ORDER_SUBMITTED
        assert result.order_id == 'order-123'
        assert result.osi_symbol == 'SPY241220C00600000'
        assert result.strike == 600.0
        assert result.contracts >= 1
        assert result.side == 'buy'
        assert result.underlying_entry_price == 600.0
        mock_trading_client.submit_option_limit_order.assert_called_once()

    def test_execute_success_market_order(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test successful execution with market order when price unavailable."""
        executor_config.use_limit_orders = False
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        assert result.state == ExecutionState.ORDER_SUBMITTED
        assert result.order_id == 'order-456'
        mock_trading_client.submit_option_market_order.assert_called_once()

    def test_execute_captures_entry_bar_data(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution captures setup bar data for pattern invalidation."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        assert result.entry_bar_type == '2U'  # CALL direction
        assert result.entry_bar_high == 602.0
        assert result.entry_bar_low == 598.0

    def test_execute_put_direction(self, executor_config, mock_trading_client):
        """Test PUT direction sets entry_bar_type to 2D."""
        put_signal = StoredSignal(
            signal_key='SPY_1D_2-1-2D_20241206',
            pattern_type='2-1-2D',
            direction='PUT',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=595.0,
            stop_price=600.0,
            target_price=585.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
            passes_flexible=True,
            setup_bar_high=598.0,
            setup_bar_low=594.0,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.execute_signal(put_signal, underlying_price=595.0)

        assert result.entry_bar_type == '2D'

    def test_execute_persists_result(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test execution result is persisted to disk."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()
        executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        # Check file was written
        executions_file = Path(executor_config.persistence_path) / 'executions.json'
        assert executions_file.exists()

        with open(executions_file, 'r') as f:
            data = json.load(f)

        assert sample_stored_signal.signal_key in data


# =============================================================================
# TEST FILTERS
# =============================================================================


class TestPassesFilters:
    """Tests for SignalExecutor._passes_filters()."""

    def test_completed_signal_passes(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test COMPLETED signal passes with good magnitude and R:R."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor._passes_filters(sample_stored_signal) is True

    def test_completed_signal_fails_magnitude(self, executor_config, mock_trading_client):
        """Test COMPLETED signal fails low magnitude."""
        signal = StoredSignal(
            signal_key='test',
            pattern_type='2-1-2U',
            direction='CALL',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=600.0,
            stop_price=595.0,
            target_price=610.0,
            magnitude_pct=0.1,  # Below 0.5% threshold
            risk_reward=1.7,
            signal_type=SignalType.COMPLETED.value,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor._passes_filters(signal) is False

    def test_completed_signal_fails_risk_reward(self, executor_config, mock_trading_client):
        """Test COMPLETED signal fails low R:R."""
        signal = StoredSignal(
            signal_key='test',
            pattern_type='2-1-2U',
            direction='CALL',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=600.0,
            stop_price=595.0,
            target_price=610.0,
            magnitude_pct=1.5,
            risk_reward=0.5,  # Below 1.0 threshold
            signal_type=SignalType.COMPLETED.value,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor._passes_filters(signal) is False

    def test_setup_signal_relaxed_thresholds(self, executor_config, mock_trading_client, sample_setup_signal):
        """Test SETUP signals use relaxed thresholds."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        # Sample setup has 0.15% magnitude and 0.5 R:R
        # Default SETUP thresholds: 0.1% magnitude, 0.3 R:R
        assert executor._passes_filters(sample_setup_signal) is True

    def test_setup_signal_fails_very_low_magnitude(self, executor_config, mock_trading_client):
        """Test SETUP signal fails very low magnitude."""
        signal = StoredSignal(
            signal_key='test',
            pattern_type='3-1-?',
            direction='CALL',
            symbol='AAPL',
            timeframe='1H',
            detected_time=datetime.now(),
            entry_trigger=195.0,
            stop_price=193.5,
            target_price=197.5,
            magnitude_pct=0.05,  # Below 0.1% relaxed threshold
            risk_reward=0.5,
            signal_type=SignalType.SETUP.value,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor._passes_filters(signal) is False

    def test_setup_thresholds_from_env(self, executor_config, mock_trading_client, monkeypatch):
        """Test SETUP thresholds can be configured via environment."""
        monkeypatch.setenv('EXECUTOR_SETUP_MIN_MAGNITUDE', '0.5')
        monkeypatch.setenv('EXECUTOR_SETUP_MIN_RR', '1.0')

        signal = StoredSignal(
            signal_key='test',
            pattern_type='3-1-?',
            direction='CALL',
            symbol='AAPL',
            timeframe='1H',
            detected_time=datetime.now(),
            entry_trigger=195.0,
            stop_price=193.5,
            target_price=197.5,
            magnitude_pct=0.3,  # Below 0.5% env threshold
            risk_reward=0.8,    # Below 1.0 env threshold
            signal_type=SignalType.SETUP.value,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor._passes_filters(signal) is False


# =============================================================================
# TEST POSITION SIZE CALCULATION
# =============================================================================


class TestCalculatePositionSize:
    """Tests for SignalExecutor._calculate_position_size()."""

    def test_basic_position_size(self, executor_config, mock_trading_client):
        """Test basic position size calculation."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=600.0,
            osi_symbol='SPY241220C00600000',
        )

        contracts = executor._calculate_position_size(contract, 600.0)

        # With $300 max capital and ~3% premium estimate ($18), max ~1-2 contracts
        assert 1 <= contracts <= 5

    def test_position_size_with_risk_multiplier(self, executor_config, mock_trading_client):
        """Test TFC risk multiplier affects position size."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=600.0,
            osi_symbol='SPY241220C00600000',
        )

        # Signal with low risk multiplier
        low_tfc_signal = Mock()
        low_tfc_signal.risk_multiplier = 0.5

        contracts_low = executor._calculate_position_size(contract, 600.0, low_tfc_signal)

        # Signal with high risk multiplier
        high_tfc_signal = Mock()
        high_tfc_signal.risk_multiplier = 1.0

        contracts_high = executor._calculate_position_size(contract, 600.0, high_tfc_signal)

        # Low TFC should result in fewer or equal contracts
        assert contracts_low <= contracts_high

    def test_position_size_minimum_one(self, executor_config, mock_trading_client):
        """Test position size is at least 1 contract."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=600.0,
            osi_symbol='SPY241220C00600000',
        )

        # Signal with zero risk multiplier
        signal = Mock()
        signal.risk_multiplier = 0.0

        contracts = executor._calculate_position_size(contract, 600.0, signal)

        assert contracts >= 1

    def test_position_size_maximum_five(self, executor_config, mock_trading_client):
        """Test position size capped at 5 contracts."""
        executor_config.max_capital_per_trade = 10000.0  # Very high capital
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=600.0,
            osi_symbol='SPY241220C00600000',
        )

        contracts = executor._calculate_position_size(contract, 600.0)

        assert contracts <= 5


# =============================================================================
# TEST CONTRACT SELECTION
# =============================================================================


class TestSelectContract:
    """Tests for SignalExecutor._select_contract()."""

    def test_select_call_contract(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test CALL contract selection."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = executor._select_contract(sample_stored_signal, 600.0)

        assert contract is not None
        assert contract.option_type == OptionType.CALL
        assert contract.strike == 600.0

    def test_select_put_contract(self, executor_config, mock_trading_client):
        """Test PUT contract selection."""
        mock_trading_client.get_option_contracts.return_value = [
            {
                'symbol': 'SPY241220P00595000',
                'strike': 595.0,
                'expiration': '2024-12-20',
            }
        ]
        signal = StoredSignal(
            signal_key='test',
            pattern_type='2-1-2D',
            direction='PUT',
            symbol='SPY',
            timeframe='1D',
            detected_time=datetime.now(),
            entry_trigger=595.0,
            stop_price=600.0,
            target_price=585.0,
            magnitude_pct=1.5,
            risk_reward=1.5,
        )
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = executor._select_contract(signal, 595.0)

        assert contract is not None
        assert contract.option_type == OptionType.PUT

    def test_select_contract_none_available(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test returns None when no contracts available."""
        mock_trading_client.get_option_contracts.return_value = []
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = executor._select_contract(sample_stored_signal, 600.0)

        assert contract is None

    def test_select_contract_closest_strike(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test selects strike closest to underlying price."""
        mock_trading_client.get_option_contracts.return_value = [
            {'symbol': 'SPY241220C00590000', 'strike': 590.0, 'expiration': '2024-12-20'},
            {'symbol': 'SPY241220C00600000', 'strike': 600.0, 'expiration': '2024-12-20'},
            {'symbol': 'SPY241220C00610000', 'strike': 610.0, 'expiration': '2024-12-20'},
        ]
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        contract = executor._select_contract(sample_stored_signal, 598.0)

        assert contract.strike == 600.0  # Closest to 598


# =============================================================================
# TEST HELPER METHODS
# =============================================================================


class TestHelperMethods:
    """Tests for SignalExecutor helper methods."""

    def test_get_positions(self, executor_config, mock_trading_client):
        """Test get_positions returns positions."""
        mock_trading_client.list_option_positions.return_value = [
            {'symbol': 'SPY241220C00600000', 'qty': 2}
        ]
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        positions = executor.get_positions()

        assert len(positions) == 1
        assert positions[0]['symbol'] == 'SPY241220C00600000'

    def test_get_positions_not_connected(self, executor_config, mock_trading_client):
        """Test get_positions returns empty when not connected."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        positions = executor.get_positions()

        assert positions == []

    def test_get_execution(self, executor_config, mock_trading_client):
        """Test get_execution retrieves specific execution."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        result = ExecutionResult(
            signal_key='test_key',
            state=ExecutionState.ORDER_SUBMITTED,
        )
        executor._executions['test_key'] = result

        retrieved = executor.get_execution('test_key')

        assert retrieved is not None
        assert retrieved.signal_key == 'test_key'

    def test_get_execution_not_found(self, executor_config, mock_trading_client):
        """Test get_execution returns None for unknown key."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        retrieved = executor.get_execution('unknown_key')

        assert retrieved is None

    def test_get_all_executions(self, executor_config, mock_trading_client):
        """Test get_all_executions returns copy of all executions."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        for i in range(3):
            executor._executions[f'key_{i}'] = ExecutionResult(
                signal_key=f'key_{i}',
                state=ExecutionState.PENDING,
            )

        all_executions = executor.get_all_executions()

        assert len(all_executions) == 3
        # Verify it's a copy
        all_executions['new_key'] = ExecutionResult(
            signal_key='new_key',
            state=ExecutionState.PENDING,
        )
        assert 'new_key' not in executor._executions

    def test_check_order_status(self, executor_config, mock_trading_client):
        """Test check_order_status retrieves order."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        status = executor.check_order_status('order-123')

        assert status is not None
        assert status['id'] == 'order-123'

    def test_check_order_status_not_connected(self, executor_config, mock_trading_client):
        """Test check_order_status returns None when not connected."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        status = executor.check_order_status('order-123')

        assert status is None

    def test_close_position(self, executor_config, mock_trading_client):
        """Test close_position closes position."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.close_position('SPY241220C00600000')

        assert result is not None
        mock_trading_client.close_option_position.assert_called_once_with('SPY241220C00600000')

    def test_close_position_not_connected(self, executor_config, mock_trading_client):
        """Test close_position returns None when not connected."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        result = executor.close_position('SPY241220C00600000')

        assert result is None

    def test_get_account_info(self, executor_config, mock_trading_client):
        """Test get_account_info retrieves account."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        account = executor.get_account_info()

        assert account is not None

    def test_get_account_info_not_connected(self, executor_config, mock_trading_client):
        """Test get_account_info returns None when not connected."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        account = executor.get_account_info()

        assert account is None

    def test_is_connected_property(self, executor_config, mock_trading_client):
        """Test is_connected property."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        assert executor.is_connected is False

        executor.connect()

        assert executor.is_connected is True


# =============================================================================
# TEST CONVENIENCE FUNCTION
# =============================================================================


class TestCreatePaperExecutor:
    """Tests for create_paper_executor convenience function."""

    def test_creates_executor_with_defaults(self):
        """Test create_paper_executor returns properly configured executor."""
        with patch.object(SignalExecutor, '_load'):  # Skip loading
            executor = create_paper_executor()

        assert executor.config.account == 'SMALL'
        assert executor.config.paper_mode is True
        assert executor.config.max_capital_per_trade == 300.0
        assert executor.config.max_concurrent_positions == 5


# =============================================================================
# TEST THREAD SAFETY
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety in SignalExecutor."""

    def test_execution_lock_exists(self, executor_config, mock_trading_client):
        """Test that executor has a lock for thread safety."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )

        # Verify lock exists
        assert hasattr(executor, '_execution_lock')
        assert executor._execution_lock is not None

    def test_duplicate_execution_returns_existing(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test that re-executing same signal returns existing result."""
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        # Execute first time
        result1 = executor.execute_signal(sample_stored_signal, underlying_price=600.0)
        assert result1.state == ExecutionState.ORDER_SUBMITTED

        # Execute second time - should return existing
        result2 = executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        # Second result should be the same as first (returns existing)
        assert result2.signal_key == result1.signal_key
        assert result2.state == ExecutionState.ORDER_SUBMITTED


# =============================================================================
# TEST ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in SignalExecutor."""

    def test_execute_handles_order_submission_error(self, executor_config, mock_trading_client, sample_stored_signal):
        """Test graceful handling of order submission error."""
        # Need to make both limit and market orders fail
        mock_trading_client.submit_option_limit_order.side_effect = Exception('Network error')
        mock_trading_client.submit_option_market_order.side_effect = Exception('Network error')
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()
        executor._get_option_price = Mock(return_value=5.50)  # Ensure limit order path is taken

        result = executor.execute_signal(sample_stored_signal, underlying_price=600.0)

        assert result.state == ExecutionState.FAILED
        assert 'Network error' in result.error

    def test_check_order_handles_error(self, executor_config, mock_trading_client):
        """Test check_order_status handles API errors."""
        mock_trading_client.get_order.side_effect = Exception('API error')
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.check_order_status('order-123')

        assert result is None

    def test_close_position_handles_error(self, executor_config, mock_trading_client):
        """Test close_position handles API errors."""
        mock_trading_client.close_option_position.side_effect = Exception('API error')
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        result = executor.close_position('SPY241220C00600000')

        assert result is None

    def test_can_open_position_handles_error(self, executor_config, mock_trading_client):
        """Test _can_open_position handles API errors."""
        mock_trading_client.list_option_positions.side_effect = Exception('API error')
        executor = SignalExecutor(
            config=executor_config,
            trading_client=mock_trading_client,
        )
        executor.connect()

        can_open = executor._can_open_position()

        assert can_open is False
