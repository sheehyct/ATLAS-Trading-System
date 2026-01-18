"""
Tests for SignalStore - Session EQUITY-71

Comprehensive test coverage for strat/signal_automation/signal_store.py including:
- SignalStatus and SignalType enums
- TIMEFRAME_PRIORITY constant
- StoredSignal dataclass (properties, serialization, factory methods)
- SignalStore initialization and persistence
- Signal lifecycle management (add, mark, query)
- Deduplication logic
- OSI symbol reverse index
- Query methods and statistics
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from strat.signal_automation.signal_store import (
    SignalStatus,
    SignalType,
    TIMEFRAME_PRIORITY,
    StoredSignal,
    SignalStore,
)
from strat.paper_signal_scanner import DetectedSignal, SignalContext


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_store_dir(tmp_path):
    """Create temporary directory for signal store tests."""
    store_dir = tmp_path / 'signals'
    store_dir.mkdir(parents=True)
    return str(store_dir)


@pytest.fixture
def mock_signal_context():
    """Create a mock signal context."""
    return SignalContext(
        vix=15.5,
        atr_14=5.2,
        atr_percent=0.87,
        volume_ratio=1.2,
        market_regime='TREND_NEUTRAL',
        tfc_score=3,
        tfc_alignment='1M:2U, 1W:2U, 1D:2U',
    )


@pytest.fixture
def mock_detected_signal(mock_signal_context):
    """Create a mock detected signal."""
    return DetectedSignal(
        pattern_type='2-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime(2024, 12, 6, 10, 30),
        entry_trigger=600.50,
        stop_price=595.00,
        target_price=610.00,
        magnitude_pct=1.58,
        risk_reward=1.73,
        context=mock_signal_context,
    )


@pytest.fixture
def sample_stored_signal():
    """Create a sample StoredSignal for testing."""
    return StoredSignal(
        signal_key='SPY_1D_2-1-2U_CALL_202412060000',
        pattern_type='2-1-2U',
        direction='CALL',
        symbol='SPY',
        timeframe='1D',
        detected_time=datetime(2024, 12, 6, 10, 30),
        entry_trigger=600.50,
        stop_price=595.00,
        target_price=610.00,
        magnitude_pct=1.58,
        risk_reward=1.73,
        vix=15.5,
        status=SignalStatus.DETECTED.value,
        first_seen_at=datetime(2024, 12, 6, 10, 30),
        last_seen_at=datetime(2024, 12, 6, 10, 30),
    )


@pytest.fixture
def signal_store(temp_store_dir):
    """Create a signal store with temp directory."""
    return SignalStore(store_path=temp_store_dir)


# =============================================================================
# TEST ENUMS
# =============================================================================


class TestSignalStatus:
    """Tests for SignalStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses are defined."""
        expected = ['DETECTED', 'ALERTED', 'TRIGGERED', 'HISTORICAL_TRIGGERED', 'EXPIRED', 'CONVERTED']
        for status in expected:
            assert hasattr(SignalStatus, status)

    def test_status_values(self):
        """Verify status values match enum names."""
        assert SignalStatus.DETECTED.value == 'DETECTED'
        assert SignalStatus.ALERTED.value == 'ALERTED'
        assert SignalStatus.TRIGGERED.value == 'TRIGGERED'
        assert SignalStatus.HISTORICAL_TRIGGERED.value == 'HISTORICAL_TRIGGERED'
        assert SignalStatus.EXPIRED.value == 'EXPIRED'
        assert SignalStatus.CONVERTED.value == 'CONVERTED'

    def test_status_is_string_enum(self):
        """Verify SignalStatus inherits from str."""
        assert isinstance(SignalStatus.DETECTED, str)
        assert SignalStatus.DETECTED == 'DETECTED'


class TestSignalType:
    """Tests for SignalType enum."""

    def test_all_types_exist(self):
        """Verify all expected types are defined."""
        assert hasattr(SignalType, 'SETUP')
        assert hasattr(SignalType, 'COMPLETED')

    def test_type_values(self):
        """Verify type values."""
        assert SignalType.SETUP.value == 'SETUP'
        assert SignalType.COMPLETED.value == 'COMPLETED'


class TestTimeframePriority:
    """Tests for TIMEFRAME_PRIORITY constant."""

    def test_all_timeframes_defined(self):
        """Verify all timeframes have priority."""
        assert '1M' in TIMEFRAME_PRIORITY
        assert '1W' in TIMEFRAME_PRIORITY
        assert '1D' in TIMEFRAME_PRIORITY
        assert '1H' in TIMEFRAME_PRIORITY

    def test_priority_ordering(self):
        """Verify monthly > weekly > daily > hourly."""
        assert TIMEFRAME_PRIORITY['1M'] > TIMEFRAME_PRIORITY['1W']
        assert TIMEFRAME_PRIORITY['1W'] > TIMEFRAME_PRIORITY['1D']
        assert TIMEFRAME_PRIORITY['1D'] > TIMEFRAME_PRIORITY['1H']


# =============================================================================
# TEST STORED SIGNAL DATACLASS
# =============================================================================


class TestStoredSignalProperties:
    """Tests for StoredSignal properties."""

    def test_priority_property(self, sample_stored_signal):
        """Test priority returns timeframe priority."""
        sample_stored_signal.timeframe = '1D'
        assert sample_stored_signal.priority == 2  # Daily priority

        sample_stored_signal.timeframe = '1W'
        assert sample_stored_signal.priority == 3  # Weekly priority

    def test_priority_unknown_timeframe(self, sample_stored_signal):
        """Test priority returns 0 for unknown timeframe."""
        sample_stored_signal.timeframe = '4H'
        assert sample_stored_signal.priority == 0

    def test_priority_rank_property(self, sample_stored_signal):
        """Test priority_rank returns TFC-based rank."""
        sample_stored_signal.tfc_priority_rank = 3
        assert sample_stored_signal.priority_rank == 3

    def test_signal_id_alias(self, sample_stored_signal):
        """Test signal_id is alias for signal_key."""
        assert sample_stored_signal.signal_id == sample_stored_signal.signal_key


class TestStoredSignalSerialization:
    """Tests for StoredSignal serialization."""

    def test_to_dict(self, sample_stored_signal):
        """Test to_dict converts to dictionary."""
        d = sample_stored_signal.to_dict()

        assert d['signal_key'] == 'SPY_1D_2-1-2U_CALL_202412060000'
        assert d['pattern_type'] == '2-1-2U'
        assert d['direction'] == 'CALL'
        assert d['symbol'] == 'SPY'
        assert d['timeframe'] == '1D'
        assert d['entry_trigger'] == 600.50
        assert d['stop_price'] == 595.00
        assert d['target_price'] == 610.00
        assert d['status'] == 'DETECTED'
        # Datetime should be ISO string
        assert isinstance(d['detected_time'], str)
        assert '2024-12-06' in d['detected_time']

    def test_from_dict(self):
        """Test from_dict creates StoredSignal."""
        d = {
            'signal_key': 'AAPL_1H_3-1-2U_CALL_202412061000',
            'pattern_type': '3-1-2U',
            'direction': 'CALL',
            'symbol': 'AAPL',
            'timeframe': '1H',
            'detected_time': '2024-12-06T10:00:00',
            'entry_trigger': 195.00,
            'stop_price': 193.50,
            'target_price': 197.50,
            'magnitude_pct': 0.75,
            'risk_reward': 1.5,
            'status': 'ALERTED',
            'first_seen_at': '2024-12-06T10:00:00',
            'last_seen_at': '2024-12-06T10:30:00',
        }
        signal = StoredSignal.from_dict(d)

        assert signal.signal_key == 'AAPL_1H_3-1-2U_CALL_202412061000'
        assert signal.pattern_type == '3-1-2U'
        assert signal.symbol == 'AAPL'
        assert signal.detected_time == datetime(2024, 12, 6, 10, 0)
        assert signal.status == 'ALERTED'

    def test_roundtrip_serialization(self, sample_stored_signal):
        """Test to_dict -> from_dict preserves data."""
        d = sample_stored_signal.to_dict()
        restored = StoredSignal.from_dict(d)

        assert restored.signal_key == sample_stored_signal.signal_key
        assert restored.pattern_type == sample_stored_signal.pattern_type
        assert restored.direction == sample_stored_signal.direction
        assert restored.symbol == sample_stored_signal.symbol
        assert restored.entry_trigger == sample_stored_signal.entry_trigger
        assert restored.status == sample_stored_signal.status

    def test_from_dict_handles_none_datetimes(self):
        """Test from_dict handles None datetime fields."""
        d = {
            'signal_key': 'test',
            'pattern_type': '2-1-2U',
            'direction': 'CALL',
            'symbol': 'SPY',
            'timeframe': '1D',
            'detected_time': '2024-12-06T10:00:00',
            'entry_trigger': 600.0,
            'stop_price': 595.0,
            'target_price': 610.0,
            'magnitude_pct': 1.0,
            'risk_reward': 1.5,
            'first_seen_at': '2024-12-06T10:00:00',
            'last_seen_at': '2024-12-06T10:00:00',
            'alerted_at': None,
            'triggered_at': None,
            'expired_at': None,
        }
        signal = StoredSignal.from_dict(d)

        assert signal.alerted_at is None
        assert signal.triggered_at is None
        assert signal.expired_at is None


class TestStoredSignalGenerateKey:
    """Tests for StoredSignal.generate_key()."""

    def test_generate_key_daily(self, mock_detected_signal):
        """Test key generation for daily timeframe."""
        mock_detected_signal.timeframe = '1D'
        mock_detected_signal.detected_time = datetime(2024, 12, 6, 10, 30)

        key = StoredSignal.generate_key(mock_detected_signal)

        # Daily truncates to start of day
        assert key == 'SPY_1D_2-1-2U_CALL_202412060000'

    def test_generate_key_hourly(self, mock_detected_signal):
        """Test key generation for hourly timeframe."""
        mock_detected_signal.timeframe = '1H'
        mock_detected_signal.detected_time = datetime(2024, 12, 6, 10, 45)

        key = StoredSignal.generate_key(mock_detected_signal)

        # Hourly truncates to start of hour
        assert key == 'SPY_1H_2-1-2U_CALL_202412061000'

    def test_generate_key_weekly(self, mock_detected_signal):
        """Test key generation for weekly timeframe."""
        mock_detected_signal.timeframe = '1W'
        # December 6, 2024 is a Friday
        mock_detected_signal.detected_time = datetime(2024, 12, 6, 10, 30)

        key = StoredSignal.generate_key(mock_detected_signal)

        # Weekly truncates to Monday of the week (Dec 2)
        assert 'SPY_1W_2-1-2U_CALL_' in key
        assert '202412020000' in key  # Monday Dec 2

    def test_generate_key_monthly(self, mock_detected_signal):
        """Test key generation for monthly timeframe."""
        mock_detected_signal.timeframe = '1M'
        mock_detected_signal.detected_time = datetime(2024, 12, 15, 10, 30)

        key = StoredSignal.generate_key(mock_detected_signal)

        # Monthly truncates to 1st of month
        assert key == 'SPY_1M_2-1-2U_CALL_202412010000'

    def test_generate_key_includes_direction(self, mock_detected_signal):
        """Test key includes direction for bidirectional patterns."""
        mock_detected_signal.direction = 'PUT'

        key = StoredSignal.generate_key(mock_detected_signal)

        assert '_PUT_' in key


class TestStoredSignalFromDetectedSignal:
    """Tests for StoredSignal.from_detected_signal()."""

    def test_from_detected_signal_basic(self, mock_detected_signal):
        """Test basic conversion from DetectedSignal."""
        stored = StoredSignal.from_detected_signal(mock_detected_signal)

        assert stored.pattern_type == '2-1-2U'
        assert stored.direction == 'CALL'
        assert stored.symbol == 'SPY'
        assert stored.timeframe == '1D'
        assert stored.entry_trigger == 600.50
        assert stored.stop_price == 595.00
        assert stored.target_price == 610.00
        assert stored.magnitude_pct == 1.58
        assert stored.risk_reward == 1.73

    def test_from_detected_signal_extracts_context(self, mock_detected_signal, mock_signal_context):
        """Test context fields are extracted."""
        stored = StoredSignal.from_detected_signal(mock_detected_signal)

        assert stored.vix == 15.5
        assert stored.atr_14 == 5.2
        assert stored.volume_ratio == 1.2
        assert stored.market_regime == 'TREND_NEUTRAL'
        assert stored.tfc_score == 3

    def test_from_detected_signal_no_context(self, mock_detected_signal):
        """Test handles missing context."""
        mock_detected_signal.context = None

        stored = StoredSignal.from_detected_signal(mock_detected_signal)

        assert stored.vix == 0.0
        assert stored.atr_14 == 0.0
        assert stored.tfc_score == 0

    def test_from_detected_signal_extracts_setup_fields(self, mock_detected_signal):
        """Test setup fields are extracted."""
        mock_detected_signal.signal_type = 'SETUP'
        mock_detected_signal.setup_bar_high = 602.0
        mock_detected_signal.setup_bar_low = 598.0

        stored = StoredSignal.from_detected_signal(mock_detected_signal)

        assert stored.signal_type == 'SETUP'
        assert stored.setup_bar_high == 602.0
        assert stored.setup_bar_low == 598.0


# =============================================================================
# TEST SIGNAL STORE INITIALIZATION
# =============================================================================


class TestSignalStoreInit:
    """Tests for SignalStore initialization."""

    def test_init_creates_directory(self, tmp_path):
        """Test init creates store directory."""
        store_dir = tmp_path / 'new_signals'
        store = SignalStore(store_path=str(store_dir))

        assert store_dir.exists()

    def test_init_empty_store(self, signal_store):
        """Test init with empty store."""
        assert len(signal_store) == 0

    def test_init_loads_existing_signals(self, temp_store_dir):
        """Test init loads existing signals from disk."""
        # Create a signals file
        signals_file = Path(temp_store_dir) / 'signals.json'
        data = {
            'test_key': {
                'signal_key': 'test_key',
                'pattern_type': '2-1-2U',
                'direction': 'CALL',
                'symbol': 'SPY',
                'timeframe': '1D',
                'detected_time': '2024-12-06T10:00:00',
                'entry_trigger': 600.0,
                'stop_price': 595.0,
                'target_price': 610.0,
                'magnitude_pct': 1.5,
                'risk_reward': 1.7,
                'status': 'DETECTED',
                'first_seen_at': '2024-12-06T10:00:00',
                'last_seen_at': '2024-12-06T10:00:00',
            }
        }
        with open(signals_file, 'w') as f:
            json.dump(data, f)

        store = SignalStore(store_path=temp_store_dir)

        assert len(store) == 1
        assert 'test_key' in store

    def test_init_handles_corrupted_file(self, temp_store_dir):
        """Test init handles corrupted JSON file."""
        signals_file = Path(temp_store_dir) / 'signals.json'
        with open(signals_file, 'w') as f:
            f.write('not valid json')

        store = SignalStore(store_path=temp_store_dir)

        assert len(store) == 0


# =============================================================================
# TEST SIGNAL STORE PERSISTENCE
# =============================================================================


class TestSignalStorePersistence:
    """Tests for SignalStore persistence."""

    def test_save_and_load(self, temp_store_dir, mock_detected_signal):
        """Test signals are saved and loaded correctly."""
        store1 = SignalStore(store_path=temp_store_dir)
        stored = store1.add_signal(mock_detected_signal)

        # Create new store instance
        store2 = SignalStore(store_path=temp_store_dir)

        assert len(store2) == 1
        loaded = store2.get_signal(stored.signal_key)
        assert loaded is not None
        assert loaded.symbol == 'SPY'


# =============================================================================
# TEST ADD SIGNAL
# =============================================================================


class TestAddSignal:
    """Tests for SignalStore.add_signal()."""

    def test_add_signal(self, signal_store, mock_detected_signal):
        """Test adding a signal."""
        stored = signal_store.add_signal(mock_detected_signal)

        assert stored.signal_key in signal_store
        assert stored.symbol == 'SPY'
        assert stored.status == SignalStatus.DETECTED.value

    def test_add_signal_persists(self, signal_store, mock_detected_signal):
        """Test added signal is persisted to disk."""
        stored = signal_store.add_signal(mock_detected_signal)

        # Check file exists
        assert signal_store.signals_file.exists()

        # Read file and verify
        with open(signal_store.signals_file, 'r') as f:
            data = json.load(f)

        assert stored.signal_key in data


# =============================================================================
# TEST DEDUPLICATION
# =============================================================================


class TestIsDuplicate:
    """Tests for SignalStore.is_duplicate()."""

    def test_is_duplicate_new_signal(self, signal_store, mock_detected_signal):
        """Test new signal is not duplicate."""
        assert signal_store.is_duplicate(mock_detected_signal) is False

    def test_is_duplicate_existing_signal(self, signal_store, mock_detected_signal):
        """Test existing signal is duplicate."""
        signal_store.add_signal(mock_detected_signal)

        # Same signal should be duplicate
        assert signal_store.is_duplicate(mock_detected_signal) is True

    def test_is_duplicate_increments_count(self, signal_store, mock_detected_signal):
        """Test duplicate detection increments occurrence count."""
        stored = signal_store.add_signal(mock_detected_signal)
        initial_count = stored.occurrence_count

        signal_store.is_duplicate(mock_detected_signal)

        updated = signal_store.get_signal(stored.signal_key)
        assert updated.occurrence_count == initial_count + 1

    def test_is_duplicate_updates_last_seen(self, signal_store, mock_detected_signal):
        """Test duplicate detection updates last_seen_at."""
        stored = signal_store.add_signal(mock_detected_signal)
        original_last_seen = stored.last_seen_at

        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)

        signal_store.is_duplicate(mock_detected_signal)

        updated = signal_store.get_signal(stored.signal_key)
        assert updated.last_seen_at >= original_last_seen


# =============================================================================
# TEST SIGNAL LIFECYCLE
# =============================================================================


class TestMarkAlerted:
    """Tests for SignalStore.mark_alerted()."""

    def test_mark_alerted(self, signal_store, mock_detected_signal):
        """Test marking signal as alerted."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.mark_alerted(stored.signal_key)

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.ALERTED.value
        assert updated.alerted_at is not None

    def test_mark_alerted_not_found(self, signal_store):
        """Test marking non-existent signal."""
        result = signal_store.mark_alerted('nonexistent_key')
        assert result is False


class TestMarkTriggered:
    """Tests for SignalStore.mark_triggered()."""

    def test_mark_triggered(self, signal_store, mock_detected_signal):
        """Test marking signal as triggered."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.mark_triggered(stored.signal_key)

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.TRIGGERED.value
        assert updated.triggered_at is not None

    def test_mark_triggered_not_found(self, signal_store):
        """Test marking non-existent signal."""
        result = signal_store.mark_triggered('nonexistent_key')
        assert result is False


class TestMarkHistoricalTriggered:
    """Tests for SignalStore.mark_historical_triggered()."""

    def test_mark_historical_triggered(self, signal_store, mock_detected_signal):
        """Test marking signal as historical triggered."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.mark_historical_triggered(stored.signal_key)

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.HISTORICAL_TRIGGERED.value

    def test_mark_historical_triggered_not_found(self, signal_store):
        """Test marking non-existent signal."""
        result = signal_store.mark_historical_triggered('nonexistent_key')
        assert result is False


class TestMarkExpired:
    """Tests for SignalStore.mark_expired()."""

    def test_mark_expired(self, signal_store, mock_detected_signal):
        """Test marking signal as expired."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.mark_expired(stored.signal_key)

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.EXPIRED.value
        assert updated.expired_at is not None

    def test_mark_expired_not_found(self, signal_store):
        """Test marking non-existent signal."""
        result = signal_store.mark_expired('nonexistent_key')
        assert result is False


class TestMarkConverted:
    """Tests for SignalStore.mark_converted()."""

    def test_mark_converted(self, signal_store, mock_detected_signal):
        """Test marking signal as converted."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.mark_converted(stored.signal_key, 'trade_123')

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.status == SignalStatus.CONVERTED.value
        assert updated.converted_at is not None
        assert updated.paper_trade_id == 'trade_123'

    def test_mark_converted_not_found(self, signal_store):
        """Test marking non-existent signal."""
        result = signal_store.mark_converted('nonexistent_key', 'trade_123')
        assert result is False


# =============================================================================
# TEST OSI SYMBOL INDEX
# =============================================================================


class TestOsiSymbolIndex:
    """Tests for OSI symbol reverse index."""

    def test_set_executed_osi_symbol(self, signal_store, mock_detected_signal):
        """Test setting executed OSI symbol."""
        stored = signal_store.add_signal(mock_detected_signal)

        result = signal_store.set_executed_osi_symbol(stored.signal_key, 'SPY241220C00600000')

        assert result is True
        updated = signal_store.get_signal(stored.signal_key)
        assert updated.executed_osi_symbol == 'SPY241220C00600000'

    def test_set_executed_osi_symbol_not_found(self, signal_store):
        """Test setting OSI symbol for non-existent signal."""
        result = signal_store.set_executed_osi_symbol('nonexistent_key', 'SPY241220C00600000')
        assert result is False

    def test_get_signal_by_osi_symbol(self, signal_store, mock_detected_signal):
        """Test looking up signal by OSI symbol."""
        stored = signal_store.add_signal(mock_detected_signal)
        signal_store.set_executed_osi_symbol(stored.signal_key, 'SPY241220C00600000')

        found = signal_store.get_signal_by_osi_symbol('SPY241220C00600000')

        assert found is not None
        assert found.signal_key == stored.signal_key

    def test_get_signal_by_osi_symbol_not_found(self, signal_store):
        """Test looking up non-existent OSI symbol."""
        found = signal_store.get_signal_by_osi_symbol('NONEXISTENT')
        assert found is None

    def test_osi_index_rebuilt_on_load(self, temp_store_dir, mock_detected_signal):
        """Test OSI index is rebuilt when loading from disk."""
        # Create store and add signal with OSI
        store1 = SignalStore(store_path=temp_store_dir)
        stored = store1.add_signal(mock_detected_signal)
        store1.set_executed_osi_symbol(stored.signal_key, 'SPY241220C00600000')

        # Create new store instance
        store2 = SignalStore(store_path=temp_store_dir)

        # Should be able to lookup by OSI
        found = store2.get_signal_by_osi_symbol('SPY241220C00600000')
        assert found is not None


# =============================================================================
# TEST QUERY METHODS
# =============================================================================


class TestGetPendingSignals:
    """Tests for SignalStore.get_pending_signals()."""

    def test_get_pending_signals(self, signal_store, mock_detected_signal):
        """Test getting pending signals."""
        stored = signal_store.add_signal(mock_detected_signal)

        pending = signal_store.get_pending_signals()

        assert len(pending) == 1
        assert pending[0].signal_key == stored.signal_key

    def test_get_pending_signals_excludes_alerted(self, signal_store, mock_detected_signal):
        """Test pending excludes alerted signals."""
        stored = signal_store.add_signal(mock_detected_signal)
        signal_store.mark_alerted(stored.signal_key)

        pending = signal_store.get_pending_signals()

        assert len(pending) == 0


class TestGetAlertedSignals:
    """Tests for SignalStore.get_alerted_signals()."""

    def test_get_alerted_signals(self, signal_store, mock_detected_signal):
        """Test getting alerted signals."""
        stored = signal_store.add_signal(mock_detected_signal)
        signal_store.mark_alerted(stored.signal_key)

        alerted = signal_store.get_alerted_signals()

        assert len(alerted) == 1
        assert alerted[0].signal_key == stored.signal_key

    def test_get_alerted_signals_excludes_detected(self, signal_store, mock_detected_signal):
        """Test alerted excludes detected signals."""
        signal_store.add_signal(mock_detected_signal)

        alerted = signal_store.get_alerted_signals()

        assert len(alerted) == 0


class TestGetSetupSignalsForMonitoring:
    """Tests for SignalStore.get_setup_signals_for_monitoring()."""

    def test_get_setup_signals(self, signal_store, mock_detected_signal):
        """Test getting SETUP signals for monitoring."""
        mock_detected_signal.signal_type = SignalType.SETUP.value
        stored = signal_store.add_signal(mock_detected_signal)
        # Manually set signal_type since from_detected_signal uses getattr
        signal_store._signals[stored.signal_key].signal_type = SignalType.SETUP.value

        setup_signals = signal_store.get_setup_signals_for_monitoring()

        assert len(setup_signals) == 1

    def test_get_setup_signals_excludes_triggered(self, signal_store, mock_detected_signal):
        """Test setup excludes triggered signals."""
        mock_detected_signal.signal_type = SignalType.SETUP.value
        stored = signal_store.add_signal(mock_detected_signal)
        signal_store._signals[stored.signal_key].signal_type = SignalType.SETUP.value
        signal_store.mark_triggered(stored.signal_key)

        setup_signals = signal_store.get_setup_signals_for_monitoring()

        assert len(setup_signals) == 0


class TestGetHistoricalTriggeredSignals:
    """Tests for SignalStore.get_historical_triggered_signals()."""

    def test_get_historical_triggered_signals(self, signal_store, mock_detected_signal):
        """Test getting historical triggered signals."""
        stored = signal_store.add_signal(mock_detected_signal)
        signal_store.mark_historical_triggered(stored.signal_key)

        historical = signal_store.get_historical_triggered_signals()

        assert len(historical) == 1
        assert historical[0].signal_key == stored.signal_key


class TestGetSignalsBySymbol:
    """Tests for SignalStore.get_signals_by_symbol()."""

    def test_get_signals_by_symbol(self, signal_store, mock_detected_signal):
        """Test filtering by symbol."""
        signal_store.add_signal(mock_detected_signal)

        spy_signals = signal_store.get_signals_by_symbol('SPY')
        aapl_signals = signal_store.get_signals_by_symbol('AAPL')

        assert len(spy_signals) == 1
        assert len(aapl_signals) == 0


class TestGetSignalsByTimeframe:
    """Tests for SignalStore.get_signals_by_timeframe()."""

    def test_get_signals_by_timeframe(self, signal_store, mock_detected_signal):
        """Test filtering by timeframe."""
        signal_store.add_signal(mock_detected_signal)

        daily_signals = signal_store.get_signals_by_timeframe('1D')
        hourly_signals = signal_store.get_signals_by_timeframe('1H')

        assert len(daily_signals) == 1
        assert len(hourly_signals) == 0


class TestGetSignalsByPattern:
    """Tests for SignalStore.get_signals_by_pattern()."""

    def test_get_signals_by_pattern(self, signal_store, mock_detected_signal):
        """Test filtering by pattern."""
        signal_store.add_signal(mock_detected_signal)

        pattern_212 = signal_store.get_signals_by_pattern('2-1-2U')
        pattern_312 = signal_store.get_signals_by_pattern('3-1-2U')

        assert len(pattern_212) == 1
        assert len(pattern_312) == 0


class TestGetRecentSignals:
    """Tests for SignalStore.get_recent_signals()."""

    def test_get_recent_signals(self, signal_store, mock_detected_signal):
        """Test getting recent signals."""
        signal_store.add_signal(mock_detected_signal)

        recent = signal_store.get_recent_signals(hours=24)

        assert len(recent) == 1

    def test_get_recent_signals_excludes_old(self, signal_store, sample_stored_signal):
        """Test recent excludes old signals."""
        # Add signal with old first_seen_at
        sample_stored_signal.first_seen_at = datetime.now() - timedelta(hours=48)
        signal_store._signals[sample_stored_signal.signal_key] = sample_stored_signal

        recent = signal_store.get_recent_signals(hours=24)

        assert len(recent) == 0


# =============================================================================
# TEST CLEANUP
# =============================================================================


class TestCleanupOldSignals:
    """Tests for SignalStore.cleanup_old_signals()."""

    def test_cleanup_old_signals(self, signal_store, sample_stored_signal):
        """Test cleaning up old signals."""
        # Add old signal
        sample_stored_signal.first_seen_at = datetime.now() - timedelta(days=60)
        signal_store._signals[sample_stored_signal.signal_key] = sample_stored_signal
        signal_store._save()

        removed = signal_store.cleanup_old_signals(days=30)

        assert removed == 1
        assert len(signal_store) == 0

    def test_cleanup_preserves_recent(self, signal_store, mock_detected_signal):
        """Test cleanup preserves recent signals."""
        signal_store.add_signal(mock_detected_signal)

        removed = signal_store.cleanup_old_signals(days=30)

        assert removed == 0
        assert len(signal_store) == 1


# =============================================================================
# TEST STATISTICS
# =============================================================================


class TestGetStats:
    """Tests for SignalStore.get_stats()."""

    def test_get_stats_empty(self, signal_store):
        """Test stats for empty store."""
        stats = signal_store.get_stats()

        assert stats['total'] == 0
        assert stats['by_status'] == {}
        assert stats['by_symbol'] == {}

    def test_get_stats_with_signals(self, signal_store, mock_detected_signal):
        """Test stats with signals."""
        signal_store.add_signal(mock_detected_signal)

        stats = signal_store.get_stats()

        assert stats['total'] == 1
        assert stats['by_status'] == {'DETECTED': 1}
        assert stats['by_symbol'] == {'SPY': 1}
        assert stats['by_timeframe'] == {'1D': 1}
        assert stats['by_pattern'] == {'2-1-2U': 1}
        assert 'oldest' in stats
        assert 'newest' in stats


# =============================================================================
# TEST DUNDER METHODS
# =============================================================================


class TestDunderMethods:
    """Tests for SignalStore dunder methods."""

    def test_len(self, signal_store, mock_detected_signal):
        """Test __len__ returns signal count."""
        assert len(signal_store) == 0

        signal_store.add_signal(mock_detected_signal)

        assert len(signal_store) == 1

    def test_contains(self, signal_store, mock_detected_signal):
        """Test __contains__ checks key existence."""
        stored = signal_store.add_signal(mock_detected_signal)

        assert stored.signal_key in signal_store
        assert 'nonexistent_key' not in signal_store


# =============================================================================
# TEST LOAD SIGNALS METHOD
# =============================================================================


class TestLoadSignals:
    """Tests for SignalStore.load_signals()."""

    def test_load_signals_returns_copy(self, signal_store, mock_detected_signal):
        """Test load_signals returns a copy."""
        signal_store.add_signal(mock_detected_signal)

        signals = signal_store.load_signals()

        # Modify the copy
        signals['new_key'] = 'test'

        # Original should be unchanged
        assert 'new_key' not in signal_store._signals

    def test_load_signals_empty(self, signal_store):
        """Test load_signals with empty store."""
        signals = signal_store.load_signals()
        assert signals == {}
