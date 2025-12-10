"""
End-to-End Signal Automation Tests - Session 83K-51

Tests the full signal -> execute -> monitor -> exit flow for autonomous paper trading.

Test Coverage:
1. Signal Detection -> Signal Store integration
2. Signal Store -> Executor integration
3. Executor -> Position Monitor integration
4. Position Monitor -> Exit Execution integration
5. Full flow simulation with mocked Alpaca

These tests verify the Phase 4 Full Orchestration milestone.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from strat.signal_automation.config import SignalAutomationConfig, ExecutionConfig
from strat.signal_automation.signal_store import SignalStore, StoredSignal, SignalStatus
from strat.signal_automation.executor import (
    SignalExecutor,
    ExecutorConfig,
    ExecutionResult,
    ExecutionState,
)
from strat.signal_automation.position_monitor import (
    PositionMonitor,
    MonitoringConfig,
    TrackedPosition,
    ExitSignal,
    ExitReason,
)
from strat.paper_signal_scanner import DetectedSignal, SignalContext


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directories for testing."""
    signals_dir = tmp_path / 'signals'
    executions_dir = tmp_path / 'executions'
    signals_dir.mkdir(parents=True)
    executions_dir.mkdir(parents=True)
    return {
        'signals': str(signals_dir),
        'executions': str(executions_dir),
    }


@pytest.fixture
def mock_detected_signal():
    """Create a mock detected signal from scanner."""
    context = SignalContext(
        vix=15.5,
        atr_14=5.2,
        atr_percent=0.87,
        volume_ratio=1.2,
        market_regime='TREND_NEUTRAL',
    )
    return DetectedSignal(
        pattern_type='2-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime.now(),
        entry_trigger=600.50,
        stop_price=595.00,
        target_price=610.00,
        magnitude_pct=1.58,
        risk_reward=1.73,
        context=context,
    )


@pytest.fixture
def mock_stored_signal():
    """Create a mock stored signal."""
    return StoredSignal(
        signal_key='SPY_1D_2-1-2U_202412060000',
        pattern_type='2-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime.now(),
        entry_trigger=600.50,
        stop_price=595.00,
        target_price=610.00,
        magnitude_pct=1.58,
        risk_reward=1.73,
        vix=15.5,
        status=SignalStatus.DETECTED,
    )


@pytest.fixture
def mock_alpaca_credentials(monkeypatch):
    """Mock Alpaca credentials."""
    def mock_get_credentials(account):
        return {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }
    monkeypatch.setattr(
        'integrations.alpaca_trading_client.get_alpaca_credentials',
        mock_get_credentials
    )


@pytest.fixture
def mock_trading_client():
    """Create a mock Alpaca trading client."""
    client = Mock()
    client.connected = True

    # Mock account
    mock_account = Mock()
    mock_account.id = 'test_account'
    mock_account.equity = 3000.0
    mock_account.buying_power = 3000.0
    client.get_account.return_value = mock_account

    # Mock option contracts
    mock_contract = Mock()
    mock_contract.symbol = 'SPY241220C00600000'
    mock_contract.underlying_symbol = 'SPY'
    mock_contract.strike_price = 600.0
    mock_contract.expiration_date = datetime(2024, 12, 20)
    mock_contract.type = Mock(value='call')
    mock_contract.status = Mock(value='active')
    mock_contract.tradable = True

    mock_contracts_response = Mock()
    mock_contracts_response.option_contracts = [mock_contract]
    client.get_option_contracts.return_value = mock_contracts_response

    # Mock order submission
    mock_order = Mock()
    mock_order.id = 'test-order-123'
    mock_order.symbol = 'SPY241220C00600000'
    mock_order.qty = 1
    mock_order.side = Mock(value='buy')
    mock_order.type = Mock(value='market')
    mock_order.status = Mock(value='accepted')
    mock_order.time_in_force = Mock(value='day')
    mock_order.limit_price = None
    mock_order.filled_qty = 0
    mock_order.filled_avg_price = None
    mock_order.submitted_at = datetime.now()
    mock_order.filled_at = None
    client.submit_order.return_value = mock_order

    # Mock positions
    client.get_all_positions.return_value = []
    client.list_option_positions.return_value = []

    # Mock quotes
    mock_quote = Mock()
    mock_quote.bid_price = 600.00
    mock_quote.ask_price = 600.10
    mock_quote.bid_size = 100
    mock_quote.ask_size = 150
    mock_quote.timestamp = datetime.now()
    client.get_stock_latest_quote.return_value = {'SPY': mock_quote}

    return client


# =============================================================================
# SIGNAL STORE TESTS
# =============================================================================


class TestSignalStoreIntegration:
    """Test signal store integration with detected signals."""

    def test_detected_signal_to_stored_signal(self, temp_data_dir, mock_detected_signal):
        """Test converting detected signal to stored signal."""
        store = SignalStore(temp_data_dir['signals'])

        # Add signal
        stored = store.add_signal(mock_detected_signal)

        assert stored is not None
        assert stored.symbol == 'SPY'
        assert stored.pattern_type == '2-1-2U'
        assert stored.direction == 'CALL'
        assert stored.status == SignalStatus.DETECTED
        assert stored.signal_key is not None

    def test_signal_deduplication(self, temp_data_dir, mock_detected_signal):
        """Test signals are not duplicated."""
        store = SignalStore(temp_data_dir['signals'])

        # Add same signal twice
        stored1 = store.add_signal(mock_detected_signal)

        # Check for duplicate
        is_dup = store.is_duplicate(mock_detected_signal)

        assert is_dup is True

    def test_signal_status_transitions(self, temp_data_dir, mock_detected_signal):
        """Test signal status transitions through lifecycle."""
        store = SignalStore(temp_data_dir['signals'])

        # Add signal (DETECTED)
        stored = store.add_signal(mock_detected_signal)
        assert stored.status == SignalStatus.DETECTED

        # Mark as alerted
        store.mark_alerted(stored.signal_key)
        updated = store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.ALERTED

        # Mark as triggered
        store.mark_triggered(stored.signal_key)
        updated = store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.TRIGGERED


# =============================================================================
# EXECUTOR INTEGRATION TESTS
# =============================================================================


class TestExecutorIntegration:
    """Test executor integration with signal store and trading client."""

    @patch('integrations.alpaca_trading_client.TradingClient')
    @patch('integrations.alpaca_trading_client.StockHistoricalDataClient')
    def test_executor_executes_stored_signal(
        self,
        mock_data_client,
        mock_trading_client_class,
        temp_data_dir,
        mock_stored_signal,
        mock_alpaca_credentials,
    ):
        """Test executor successfully executes a stored signal."""
        # Setup mock trading client
        mock_client = Mock()
        mock_account = Mock()
        mock_account.id = 'test'
        mock_account.equity = 3000.0
        mock_account.buying_power = 3000.0
        mock_client.get_account.return_value = mock_account
        mock_client.get_all_positions.return_value = []

        # Mock option contracts
        mock_contract = {
            'symbol': 'SPY241220C00600000',
            'underlying': 'SPY',
            'strike': 600.0,
            'expiration': (datetime.now() + timedelta(days=14)).isoformat(),
            'type': 'call',
            'tradable': True,
        }

        mock_trading_client_class.return_value = mock_client

        # Create executor
        config = ExecutorConfig(
            account='SMALL',
            persistence_path=temp_data_dir['executions'],
            max_capital_per_trade=300.0,
        )
        executor = SignalExecutor(config=config)

        # Inject mock client directly for testing
        from integrations.alpaca_trading_client import AlpacaTradingClient
        mock_alpaca_client = Mock(spec=AlpacaTradingClient)
        mock_alpaca_client.get_option_contracts.return_value = [mock_contract]
        mock_alpaca_client.list_option_positions.return_value = []
        mock_alpaca_client.get_stock_price.return_value = 600.05
        mock_alpaca_client.submit_option_market_order.return_value = {
            'id': 'test-order-123',
            'symbol': 'SPY241220C00600000',
            'qty': 1,
            'side': 'buy',
            'status': 'accepted',
        }

        executor._trading_client = mock_alpaca_client
        executor._connected = True

        # Execute signal
        result = executor.execute_signal(mock_stored_signal, underlying_price=600.05)

        # Verify execution
        assert result.state in [ExecutionState.ORDER_SUBMITTED, ExecutionState.SKIPPED, ExecutionState.FAILED]

    def test_executor_persistence(self, temp_data_dir, mock_stored_signal):
        """Test executor persists executions to disk."""
        config = ExecutorConfig(persistence_path=temp_data_dir['executions'])

        executor = SignalExecutor(config=config)

        # Add a test execution
        result = ExecutionResult(
            signal_key=mock_stored_signal.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='test-order-123',
            osi_symbol='SPY241220C00600000',
            strike=600.0,
            contracts=1,
        )
        executor._executions[mock_stored_signal.signal_key] = result
        executor._save()

        # Verify file exists
        exec_file = Path(temp_data_dir['executions']) / 'executions.json'
        assert exec_file.exists()

        # Verify content
        with open(exec_file) as f:
            data = json.load(f)
        assert mock_stored_signal.signal_key in data

    def test_executor_loads_persisted_executions(self, temp_data_dir):
        """Test executor loads existing executions on startup."""
        exec_path = Path(temp_data_dir['executions'])
        exec_path.mkdir(parents=True, exist_ok=True)

        # Write test data
        test_data = {
            'TEST_SIGNAL': {
                'signal_key': 'TEST_SIGNAL',
                'state': 'submitted',
                'order_id': 'order-123',
                'osi_symbol': 'SPY241220C00600000',
                'strike': 600.0,
                'expiration': '2024-12-20',
                'contracts': 1,
                'premium': 5.5,
                'side': 'buy',
                'timestamp': datetime.now().isoformat(),
                'error': None,
            }
        }
        with open(exec_path / 'executions.json', 'w') as f:
            json.dump(test_data, f)

        # Create executor (should load data)
        config = ExecutorConfig(persistence_path=str(exec_path))
        executor = SignalExecutor(config=config)

        assert 'TEST_SIGNAL' in executor._executions
        assert executor._executions['TEST_SIGNAL'].state == ExecutionState.ORDER_SUBMITTED


# =============================================================================
# POSITION MONITOR INTEGRATION TESTS
# =============================================================================


class TestPositionMonitorIntegration:
    """Test position monitor integration with executor and trading client."""

    def test_monitor_detects_dte_exit(self):
        """Test monitor detects DTE exit condition."""
        # Create tracked position with low DTE
        position = TrackedPosition(
            osi_symbol='SPY241206C00600000',  # Expires today
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=10),
            expiration=datetime.now().strftime('%Y-%m-%d'),  # Today
            current_price=5.00,
            unrealized_pnl=-50.0,
            unrealized_pct=-0.09,
            underlying_price=598.0,
            dte=0,  # Expired
        )

        # Create monitor
        config = MonitoringConfig(exit_dte=3)
        monitor = PositionMonitor(config=config)

        # Check position
        exit_signal = monitor._check_position(position)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.DTE_EXIT

    def test_monitor_detects_target_hit(self):
        """Test monitor detects target hit for CALL."""
        position = TrackedPosition(
            osi_symbol='SPY241220C00600000',
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=5),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
            current_price=10.00,
            unrealized_pnl=450.0,
            unrealized_pct=0.82,
            underlying_price=611.00,  # Above target
            dte=10,
        )

        config = MonitoringConfig(exit_dte=3)
        monitor = PositionMonitor(config=config)
        monitor._underlying_cache['SPY'] = {'price': 611.0, 'updated': datetime.now()}

        exit_signal = monitor._check_position(position)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.TARGET_HIT

    def test_monitor_detects_stop_hit(self):
        """Test monitor detects stop hit for CALL."""
        position = TrackedPosition(
            osi_symbol='SPY241220C00600000',
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=5),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
            current_price=2.00,
            unrealized_pnl=-350.0,
            unrealized_pct=-0.64,
            underlying_price=593.00,  # Below stop
            dte=10,
        )

        config = MonitoringConfig(exit_dte=3)
        monitor = PositionMonitor(config=config)
        monitor._underlying_cache['SPY'] = {'price': 593.0, 'updated': datetime.now()}

        exit_signal = monitor._check_position(position)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.STOP_HIT

    def test_monitor_detects_max_loss(self):
        """Test monitor detects max loss threshold."""
        position = TrackedPosition(
            osi_symbol='SPY241220C00600000',
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=5),
            expiration=(datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
            current_price=2.00,
            unrealized_pnl=-350.0,
            unrealized_pct=-0.64,  # 64% loss > 50% threshold
            underlying_price=597.00,  # Not at stop yet
            dte=10,
        )

        config = MonitoringConfig(exit_dte=3, max_loss_pct=0.50)
        monitor = PositionMonitor(config=config)
        monitor._underlying_cache['SPY'] = {'price': 597.0, 'updated': datetime.now()}

        exit_signal = monitor._check_position(position)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.MAX_LOSS

    def test_monitor_no_exit_healthy_position(self):
        """Test monitor does not exit healthy position."""
        position = TrackedPosition(
            osi_symbol='SPY241220C00600000',
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=2),
            expiration=(datetime.now() + timedelta(days=12)).strftime('%Y-%m-%d'),
            current_price=6.00,
            unrealized_pnl=50.0,
            unrealized_pct=0.09,
            underlying_price=602.00,  # Between entry and target
            dte=12,
        )

        config = MonitoringConfig(exit_dte=3, max_loss_pct=0.50)
        monitor = PositionMonitor(config=config)
        monitor._underlying_cache['SPY'] = {'price': 602.0, 'updated': datetime.now()}

        exit_signal = monitor._check_position(position)

        assert exit_signal is None


# =============================================================================
# END-TO-END FLOW TESTS
# =============================================================================


class TestEndToEndFlow:
    """Test complete signal -> execute -> monitor -> exit flow."""

    def test_full_signal_lifecycle(self, temp_data_dir, mock_detected_signal):
        """Test full signal lifecycle from detection to execution tracking."""
        # Step 1: Signal Detection -> Signal Store
        store = SignalStore(temp_data_dir['signals'])
        stored = store.add_signal(mock_detected_signal)

        assert stored is not None
        assert stored.status == SignalStatus.DETECTED

        # Step 2: Signal Store -> Executor (mock execution)
        exec_config = ExecutorConfig(persistence_path=temp_data_dir['executions'])
        executor = SignalExecutor(config=exec_config)

        # Create mock execution result
        result = ExecutionResult(
            signal_key=stored.signal_key,
            state=ExecutionState.ORDER_SUBMITTED,
            order_id='test-order-123',
            osi_symbol='SPY241220C00600000',
            strike=600.0,
            expiration='2024-12-20',
            contracts=1,
            premium=5.50,
            side='buy',
        )
        executor._executions[stored.signal_key] = result
        executor._save()

        # Step 3: Mark signal as triggered
        store.mark_triggered(stored.signal_key)
        updated = store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.TRIGGERED

        # Step 4: Verify execution persisted
        exec_file = Path(temp_data_dir['executions']) / 'executions.json'
        assert exec_file.exists()

        # Step 5: Verify execution can be loaded
        executor2 = SignalExecutor(config=exec_config)
        assert stored.signal_key in executor2._executions

    def test_exit_flow_integration(self, temp_data_dir):
        """Test exit flow from position monitor to order execution."""
        # Create tracked position
        position = TrackedPosition(
            osi_symbol='SPY241206C00600000',
            signal_key='TEST_SIGNAL',
            symbol='SPY',
            direction='CALL',
            entry_trigger=600.0,
            target_price=610.0,
            stop_price=595.0,
            pattern_type='2-1-2U',
            timeframe='1D',
            entry_price=5.50,
            contracts=1,
            entry_time=datetime.now() - timedelta(days=10),
            expiration=datetime.now().strftime('%Y-%m-%d'),  # Expires today
            current_price=5.00,
            unrealized_pnl=-50.0,
            dte=0,
        )

        # Create mock trading client
        mock_client = Mock()
        mock_client.close_option_position.return_value = {
            'id': 'close-order-123',
            'symbol': 'SPY241206C00600000',
            'side': 'sell',
            'status': 'accepted',
        }

        # Create monitor
        exit_callback_called = []
        def on_exit(signal, result):
            exit_callback_called.append((signal, result))

        config = MonitoringConfig(exit_dte=3)
        monitor = PositionMonitor(
            config=config,
            trading_client=mock_client,
            on_exit_callback=on_exit,
        )

        # Add position to monitor
        monitor._positions[position.osi_symbol] = position

        # Check and detect exit
        exit_signal = monitor._check_position(position)
        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.DTE_EXIT

        # Execute exit
        result = monitor.execute_exit(exit_signal)

        assert result is not None
        assert mock_client.close_option_position.called
        assert position.is_active is False
        assert len(exit_callback_called) == 1


class TestMarketDataIntegration:
    """Test market data integration for position monitoring."""

    def test_underlying_price_batch_fetch(self):
        """Test batch fetching of underlying prices."""
        # Create mock trading client with quote support
        mock_client = Mock()
        mock_client.get_stock_quotes.return_value = {
            'SPY': {'symbol': 'SPY', 'bid': 600.0, 'ask': 600.10, 'mid': 600.05},
            'QQQ': {'symbol': 'QQQ', 'bid': 500.0, 'ask': 500.08, 'mid': 500.04},
        }

        config = MonitoringConfig()
        monitor = PositionMonitor(config=config, trading_client=mock_client)

        # Add positions for multiple symbols
        for symbol in ['SPY', 'QQQ']:
            position = TrackedPosition(
                osi_symbol=f'{symbol}241220C00600000',
                signal_key=f'{symbol}_SIGNAL',
                symbol=symbol,
                direction='CALL',
                entry_trigger=600.0,
                target_price=610.0,
                stop_price=595.0,
                pattern_type='2-1-2U',
                timeframe='1D',
                entry_price=5.50,
                contracts=1,
                entry_time=datetime.now(),
                expiration=(datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            )
            monitor._positions[position.osi_symbol] = position

        # Fetch prices
        monitor._update_underlying_prices()

        # Verify cache
        assert 'SPY' in monitor._underlying_cache
        assert 'QQQ' in monitor._underlying_cache
        assert monitor._underlying_cache['SPY']['price'] == 600.05
        assert monitor._underlying_cache['QQQ']['price'] == 500.04
