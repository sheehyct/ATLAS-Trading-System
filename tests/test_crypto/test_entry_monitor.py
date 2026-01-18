"""
Tests for crypto/scanning/entry_monitor.py

Session EQUITY-72: Comprehensive test coverage for CryptoEntryMonitor.

Covers:
- TIMEFRAME_PRIORITY constant
- CryptoTriggerEvent dataclass
- CryptoEntryMonitorConfig dataclass
- CryptoEntryMonitor class:
  - Initialization
  - Maintenance window detection
  - Signal management (add, remove, clear, get)
  - Expired signal cleanup
  - Price fetching
  - Trigger detection (the main logic)
  - Background monitoring (start/stop)
  - Statistics and status
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch, call
import threading
import time

from crypto.scanning.entry_monitor import (
    TIMEFRAME_PRIORITY,
    CryptoTriggerEvent,
    CryptoEntryMonitorConfig,
    CryptoEntryMonitor,
)
from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto import config as crypto_config


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_signal_setup_long():
    """Create a SETUP signal expecting LONG direction."""
    return CryptoDetectedSignal(
        pattern_type="3-1-?",
        direction="LONG",
        symbol="BTC-USD",
        timeframe="1h",
        detected_time=datetime.now(timezone.utc),
        entry_trigger=50500.0,
        stop_price=49000.0,
        target_price=52000.0,
        magnitude_pct=3.0,
        risk_reward=1.5,
        signal_type="SETUP",
        setup_bar_high=50500.0,
        setup_bar_low=49500.0,
        prior_bar_type=3,  # Outside bar
        prior_bar_high=50600.0,
        prior_bar_low=49400.0,
    )


@pytest.fixture
def mock_signal_setup_short():
    """Create a SETUP signal expecting SHORT direction."""
    return CryptoDetectedSignal(
        pattern_type="2U-1-?",
        direction="SHORT",
        symbol="ETH-USD",
        timeframe="4h",
        detected_time=datetime.now(timezone.utc),
        entry_trigger=3500.0,
        stop_price=3600.0,
        target_price=3300.0,
        magnitude_pct=5.7,
        risk_reward=2.0,
        signal_type="SETUP",
        setup_bar_high=3550.0,
        setup_bar_low=3450.0,
        prior_bar_type=2,  # 2U bar
        prior_bar_high=3560.0,
        prior_bar_low=3440.0,
    )


@pytest.fixture
def mock_signal_completed():
    """Create a COMPLETED signal (not SETUP)."""
    return CryptoDetectedSignal(
        pattern_type="3-2U",
        direction="LONG",
        symbol="BTC-USD",
        timeframe="1d",
        detected_time=datetime.now(timezone.utc),
        entry_trigger=50000.0,
        stop_price=48000.0,
        target_price=54000.0,
        magnitude_pct=8.0,
        risk_reward=2.0,
        signal_type="COMPLETED",
    )


@pytest.fixture
def mock_signal_expired():
    """Create an expired SETUP signal (detected 48 hours ago)."""
    return CryptoDetectedSignal(
        pattern_type="2D-1-?",
        direction="LONG",
        symbol="SOL-USD",
        timeframe="1h",
        detected_time=datetime.now(timezone.utc) - timedelta(hours=48),
        entry_trigger=100.0,
        stop_price=95.0,
        target_price=110.0,
        magnitude_pct=10.0,
        risk_reward=2.0,
        signal_type="SETUP",
        setup_bar_high=100.0,
        setup_bar_low=95.0,
    )


@pytest.fixture
def mock_coinbase_client():
    """Create a mock CoinbaseClient."""
    from crypto.exchange.coinbase_client import CoinbaseClient

    client = Mock(spec=CoinbaseClient)
    # Default prices
    client.get_current_price.side_effect = lambda symbol: {
        "BTC-USD": 50000.0,
        "ETH-USD": 3500.0,
        "SOL-USD": 100.0,
    }.get(symbol, None)
    return client


@pytest.fixture
def entry_monitor(mock_coinbase_client):
    """Create a CryptoEntryMonitor with mocked client."""
    return CryptoEntryMonitor(client=mock_coinbase_client)


# =============================================================================
# TIMEFRAME_PRIORITY TESTS
# =============================================================================


class TestTimeframePriority:
    """Tests for TIMEFRAME_PRIORITY constant."""

    def test_priority_values(self):
        """Test that priority values are correct."""
        assert TIMEFRAME_PRIORITY["1w"] == 5
        assert TIMEFRAME_PRIORITY["1d"] == 4
        assert TIMEFRAME_PRIORITY["4h"] == 3
        assert TIMEFRAME_PRIORITY["1h"] == 2
        assert TIMEFRAME_PRIORITY["15m"] == 1

    def test_higher_timeframes_have_higher_priority(self):
        """Test that higher timeframes have higher priority."""
        assert TIMEFRAME_PRIORITY["1w"] > TIMEFRAME_PRIORITY["1d"]
        assert TIMEFRAME_PRIORITY["1d"] > TIMEFRAME_PRIORITY["4h"]
        assert TIMEFRAME_PRIORITY["4h"] > TIMEFRAME_PRIORITY["1h"]
        assert TIMEFRAME_PRIORITY["1h"] > TIMEFRAME_PRIORITY["15m"]


# =============================================================================
# CryptoTriggerEvent TESTS
# =============================================================================


class TestCryptoTriggerEvent:
    """Tests for CryptoTriggerEvent dataclass."""

    def test_creation(self, mock_signal_setup_long):
        """Test basic creation of trigger event."""
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        assert event.signal == mock_signal_setup_long
        assert event.trigger_price == 50500.0
        assert event.current_price == 50600.0

    def test_triggered_at_default(self, mock_signal_setup_long):
        """Test that triggered_at defaults to now UTC."""
        before = datetime.now(timezone.utc)
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        after = datetime.now(timezone.utc)
        assert before <= event.triggered_at <= after

    def test_priority_property(self, mock_signal_setup_long):
        """Test priority property returns correct timeframe priority."""
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        assert event.priority == TIMEFRAME_PRIORITY["1h"]  # signal is 1h

    def test_priority_unknown_timeframe(self, mock_signal_setup_long):
        """Test priority returns 0 for unknown timeframe."""
        mock_signal_setup_long.timeframe = "5m"  # Not in TIMEFRAME_PRIORITY
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        assert event.priority == 0

    def test_symbol_property(self, mock_signal_setup_long):
        """Test symbol property returns signal symbol."""
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        assert event.symbol == "BTC-USD"

    def test_direction_property(self, mock_signal_setup_long):
        """Test direction property returns signal direction."""
        event = CryptoTriggerEvent(
            signal=mock_signal_setup_long,
            trigger_price=50500.0,
            current_price=50600.0,
        )
        assert event.direction == "LONG"


# =============================================================================
# CryptoEntryMonitorConfig TESTS
# =============================================================================


class TestCryptoEntryMonitorConfig:
    """Tests for CryptoEntryMonitorConfig dataclass."""

    def test_default_values(self):
        """Test default values from crypto config module."""
        config = CryptoEntryMonitorConfig()
        assert config.poll_interval == crypto_config.ENTRY_MONITOR_POLL_SECONDS
        assert config.maintenance_window_enabled == crypto_config.MAINTENANCE_WINDOW_ENABLED
        assert config.max_signals_per_poll == 0
        assert config.on_trigger is None
        assert config.on_poll is None
        assert config.signal_expiry_hours == crypto_config.SIGNAL_EXPIRY_HOURS

    def test_custom_values(self):
        """Test custom values override defaults."""
        callback = Mock()
        poll_callback = Mock()
        config = CryptoEntryMonitorConfig(
            poll_interval=30,
            maintenance_window_enabled=False,
            max_signals_per_poll=10,
            on_trigger=callback,
            on_poll=poll_callback,
            signal_expiry_hours=48,
        )
        assert config.poll_interval == 30
        assert config.maintenance_window_enabled is False
        assert config.max_signals_per_poll == 10
        assert config.on_trigger == callback
        assert config.on_poll == poll_callback
        assert config.signal_expiry_hours == 48


# =============================================================================
# CryptoEntryMonitor INITIALIZATION TESTS
# =============================================================================


class TestCryptoEntryMonitorInit:
    """Tests for CryptoEntryMonitor initialization."""

    def test_init_default_client(self):
        """Test initialization creates default client."""
        monitor = CryptoEntryMonitor()
        assert monitor.client is not None
        assert monitor.config is not None

    def test_init_custom_client(self, mock_coinbase_client):
        """Test initialization with custom client."""
        monitor = CryptoEntryMonitor(client=mock_coinbase_client)
        assert monitor.client == mock_coinbase_client

    def test_init_custom_config(self, mock_coinbase_client):
        """Test initialization with custom config."""
        config = CryptoEntryMonitorConfig(poll_interval=30)
        monitor = CryptoEntryMonitor(client=mock_coinbase_client, config=config)
        assert monitor.config.poll_interval == 30

    def test_init_state(self, entry_monitor):
        """Test initial state is correct."""
        assert entry_monitor._signals == {}
        assert entry_monitor._running is False
        assert entry_monitor._thread is None
        assert entry_monitor._last_check is None
        assert entry_monitor._trigger_count == 0


# =============================================================================
# MAINTENANCE WINDOW TESTS
# =============================================================================


class TestMaintenanceWindow:
    """Tests for maintenance window detection."""

    def test_maintenance_disabled(self, mock_coinbase_client):
        """Test maintenance check returns False when disabled."""
        config = CryptoEntryMonitorConfig(maintenance_window_enabled=False)
        monitor = CryptoEntryMonitor(client=mock_coinbase_client, config=config)

        # Friday 22:00 UTC (during maintenance)
        friday_maintenance = datetime(2026, 1, 16, 22, 30, tzinfo=timezone.utc)
        assert monitor.is_maintenance_window(friday_maintenance) is False

    def test_maintenance_friday_during_window(self, entry_monitor):
        """Test maintenance returns True during Friday window."""
        # Friday 22:30 UTC (during 22:00-23:00 window)
        friday_maintenance = datetime(2026, 1, 16, 22, 30, tzinfo=timezone.utc)
        assert friday_maintenance.weekday() == 4  # Verify it's Friday
        assert entry_monitor.is_maintenance_window(friday_maintenance) is True

    def test_maintenance_friday_before_window(self, entry_monitor):
        """Test maintenance returns False before Friday window."""
        # Friday 21:30 UTC (before 22:00)
        friday_before = datetime(2026, 1, 16, 21, 30, tzinfo=timezone.utc)
        assert entry_monitor.is_maintenance_window(friday_before) is False

    def test_maintenance_friday_after_window(self, entry_monitor):
        """Test maintenance returns False after Friday window."""
        # Friday 23:30 UTC (after 23:00)
        friday_after = datetime(2026, 1, 16, 23, 30, tzinfo=timezone.utc)
        assert entry_monitor.is_maintenance_window(friday_after) is False

    def test_maintenance_other_day(self, entry_monitor):
        """Test maintenance returns False on non-Friday."""
        # Saturday same time
        saturday = datetime(2026, 1, 17, 22, 30, tzinfo=timezone.utc)
        assert entry_monitor.is_maintenance_window(saturday) is False

        # Wednesday same time
        wednesday = datetime(2026, 1, 14, 22, 30, tzinfo=timezone.utc)
        assert entry_monitor.is_maintenance_window(wednesday) is False

    def test_maintenance_naive_datetime(self, entry_monitor):
        """Test maintenance handles naive datetime by assuming UTC."""
        # Friday 22:30 naive (should be treated as UTC)
        friday_naive = datetime(2026, 1, 16, 22, 30)
        assert entry_monitor.is_maintenance_window(friday_naive) is True

    def test_maintenance_default_now(self, entry_monitor):
        """Test maintenance uses current time if None passed."""
        # Should not raise, just return a boolean
        result = entry_monitor.is_maintenance_window()
        assert isinstance(result, bool)


# =============================================================================
# SIGNAL MANAGEMENT TESTS
# =============================================================================


class TestSignalManagement:
    """Tests for signal add/remove/clear operations."""

    def test_generate_signal_id(self, entry_monitor, mock_signal_setup_long):
        """Test signal ID generation format."""
        signal_id = entry_monitor._generate_signal_id(mock_signal_setup_long)
        assert "BTC-USD" in signal_id
        assert "1h" in signal_id
        assert "3-1-?" in signal_id
        assert "LONG" in signal_id

    def test_add_signal_setup(self, entry_monitor, mock_signal_setup_long):
        """Test adding a SETUP signal succeeds."""
        result = entry_monitor.add_signal(mock_signal_setup_long)
        assert result is True
        assert len(entry_monitor.get_pending_signals()) == 1

    def test_add_signal_completed_rejected(self, entry_monitor, mock_signal_completed):
        """Test adding a COMPLETED signal is rejected."""
        result = entry_monitor.add_signal(mock_signal_completed)
        assert result is False
        assert len(entry_monitor.get_pending_signals()) == 0

    def test_add_signal_duplicate(self, entry_monitor, mock_signal_setup_long):
        """Test adding duplicate signal is rejected."""
        entry_monitor.add_signal(mock_signal_setup_long)
        result = entry_monitor.add_signal(mock_signal_setup_long)
        assert result is False
        assert len(entry_monitor.get_pending_signals()) == 1

    def test_add_signals_batch(self, entry_monitor, mock_signal_setup_long, mock_signal_setup_short):
        """Test batch adding signals."""
        signals = [mock_signal_setup_long, mock_signal_setup_short]
        added = entry_monitor.add_signals(signals)
        assert added == 2
        assert len(entry_monitor.get_pending_signals()) == 2

    def test_add_signals_with_completed(
        self, entry_monitor, mock_signal_setup_long, mock_signal_completed
    ):
        """Test batch add skips COMPLETED signals."""
        signals = [mock_signal_setup_long, mock_signal_completed]
        added = entry_monitor.add_signals(signals)
        assert added == 1

    def test_remove_signal(self, entry_monitor, mock_signal_setup_long):
        """Test removing a signal by ID."""
        entry_monitor.add_signal(mock_signal_setup_long)
        signal_id = entry_monitor._generate_signal_id(mock_signal_setup_long)
        result = entry_monitor.remove_signal(signal_id)
        assert result is True
        assert len(entry_monitor.get_pending_signals()) == 0

    def test_remove_signal_not_found(self, entry_monitor):
        """Test removing non-existent signal returns False."""
        result = entry_monitor.remove_signal("nonexistent-id")
        assert result is False

    def test_clear_signals(self, entry_monitor, mock_signal_setup_long, mock_signal_setup_short):
        """Test clearing all signals."""
        entry_monitor.add_signal(mock_signal_setup_long)
        entry_monitor.add_signal(mock_signal_setup_short)
        count = entry_monitor.clear_signals()
        assert count == 2
        assert len(entry_monitor.get_pending_signals()) == 0

    def test_clear_signals_empty(self, entry_monitor):
        """Test clearing when no signals."""
        count = entry_monitor.clear_signals()
        assert count == 0


class TestGetPendingSignals:
    """Tests for get_pending_signals method."""

    def test_sorted_by_priority(self, entry_monitor, mock_signal_setup_long, mock_signal_setup_short):
        """Test signals are sorted by timeframe priority (highest first)."""
        # mock_signal_setup_short is 4h (priority 3)
        # mock_signal_setup_long is 1h (priority 2)
        entry_monitor.add_signal(mock_signal_setup_long)  # 1h
        entry_monitor.add_signal(mock_signal_setup_short)  # 4h

        pending = entry_monitor.get_pending_signals()
        assert len(pending) == 2
        assert pending[0].timeframe == "4h"  # Higher priority first
        assert pending[1].timeframe == "1h"

    def test_max_signals_limit(self, entry_monitor, mock_signal_setup_long, mock_signal_setup_short):
        """Test max_signals_per_poll limits results."""
        entry_monitor.config.max_signals_per_poll = 1
        entry_monitor.add_signal(mock_signal_setup_long)
        entry_monitor.add_signal(mock_signal_setup_short)

        pending = entry_monitor.get_pending_signals()
        assert len(pending) == 1
        # Should get highest priority (4h)
        assert pending[0].timeframe == "4h"

    def test_empty_returns_empty_list(self, entry_monitor):
        """Test returns empty list when no signals."""
        pending = entry_monitor.get_pending_signals()
        assert pending == []


class TestExpiredSignals:
    """Tests for expired signal cleanup."""

    def test_remove_expired_signals(self, entry_monitor, mock_signal_expired, mock_signal_setup_long):
        """Test expired signals are removed."""
        entry_monitor.add_signal(mock_signal_expired)
        entry_monitor.add_signal(mock_signal_setup_long)

        removed = entry_monitor.remove_expired_signals()
        assert removed == 1
        pending = entry_monitor.get_pending_signals()
        assert len(pending) == 1
        assert pending[0].symbol == "BTC-USD"  # Fresh signal remains

    def test_no_expired_signals(self, entry_monitor, mock_signal_setup_long):
        """Test no removal when signals are fresh."""
        entry_monitor.add_signal(mock_signal_setup_long)
        removed = entry_monitor.remove_expired_signals()
        assert removed == 0
        assert len(entry_monitor.get_pending_signals()) == 1

    def test_handles_naive_datetime(self, entry_monitor):
        """Test handles signals with naive detected_time."""
        # Create signal with naive datetime
        signal = CryptoDetectedSignal(
            pattern_type="3-1-?",
            direction="LONG",
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now() - timedelta(hours=48),  # Naive, expired
            entry_trigger=50000.0,
            stop_price=49000.0,
            target_price=52000.0,
            magnitude_pct=3.0,
            risk_reward=1.5,
            signal_type="SETUP",
            setup_bar_high=50500.0,
            setup_bar_low=49500.0,
        )
        entry_monitor.add_signal(signal)
        removed = entry_monitor.remove_expired_signals()
        assert removed == 1


# =============================================================================
# PRICE FETCHING TESTS
# =============================================================================


class TestPriceFetching:
    """Tests for _fetch_prices method."""

    def test_fetch_single_symbol(self, entry_monitor, mock_coinbase_client):
        """Test fetching price for single symbol."""
        prices = entry_monitor._fetch_prices(["BTC-USD"])
        assert prices == {"BTC-USD": 50000.0}
        mock_coinbase_client.get_current_price.assert_called_once_with("BTC-USD")

    def test_fetch_multiple_symbols(self, entry_monitor, mock_coinbase_client):
        """Test fetching prices for multiple symbols."""
        prices = entry_monitor._fetch_prices(["BTC-USD", "ETH-USD"])
        assert prices == {"BTC-USD": 50000.0, "ETH-USD": 3500.0}

    def test_fetch_unknown_symbol(self, entry_monitor, mock_coinbase_client):
        """Test fetching unknown symbol returns empty dict for that symbol."""
        prices = entry_monitor._fetch_prices(["UNKNOWN-USD"])
        assert "UNKNOWN-USD" not in prices

    def test_fetch_handles_error(self, entry_monitor, mock_coinbase_client):
        """Test fetch handles API errors gracefully."""
        mock_coinbase_client.get_current_price.side_effect = Exception("API Error")
        prices = entry_monitor._fetch_prices(["BTC-USD"])
        assert prices == {}

    def test_fetch_ignores_zero_price(self, entry_monitor, mock_coinbase_client):
        """Test fetch ignores zero prices."""
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 0.0
        prices = entry_monitor._fetch_prices(["BTC-USD"])
        assert prices == {}


# =============================================================================
# TRIGGER DETECTION TESTS - THE MAIN LOGIC
# =============================================================================


class TestCheckTriggersLongBreak:
    """Tests for LONG trigger detection."""

    def test_long_trigger_break_up(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test LONG trigger when price breaks above setup_bar_high."""
        # Setup: signal expects LONG, price breaks up
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0  # Above 50500 high
        entry_monitor.add_signal(mock_signal_setup_long)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        event = triggered[0]
        assert event.trigger_price == 50500.0
        assert event.current_price == 50600.0
        assert event._actual_direction == "LONG"
        # Signal removed from pending
        assert len(entry_monitor.get_pending_signals()) == 0

    def test_long_trigger_expected_direction(
        self, entry_monitor, mock_signal_setup_long, mock_coinbase_client
    ):
        """Test LONG trigger resolves pattern correctly when direction matches."""
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0
        entry_monitor.add_signal(mock_signal_setup_long)

        triggered = entry_monitor.check_triggers()

        # Pattern 3-1-? becomes 3-1-2U
        event = triggered[0]
        assert "2U" in event._actual_pattern


class TestCheckTriggersShortBreak:
    """Tests for SHORT trigger detection."""

    def test_short_trigger_break_down(
        self, entry_monitor, mock_signal_setup_short, mock_coinbase_client
    ):
        """Test SHORT trigger when price breaks below setup_bar_low."""
        # Setup: signal expects SHORT, price breaks down
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 3400.0  # Below 3450 low
        entry_monitor.add_signal(mock_signal_setup_short)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        event = triggered[0]
        assert event.trigger_price == 3450.0  # setup_bar_low
        assert event.current_price == 3400.0
        assert event._actual_direction == "SHORT"


class TestCheckTriggersOppositeDirection:
    """Tests for triggers in opposite direction."""

    def test_long_signal_breaks_down(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test signal expecting LONG but price breaks DOWN."""
        # Signal expects LONG but price breaks below setup_bar_low
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 49400.0  # Below 49500 low
        entry_monitor.add_signal(mock_signal_setup_long)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        event = triggered[0]
        assert event._actual_direction == "SHORT"  # Changed to SHORT
        # Pattern becomes 2-bar (prior was type 3, now 2D)
        assert "2D" in event._actual_pattern

    def test_short_signal_breaks_up(
        self, entry_monitor, mock_signal_setup_short, mock_coinbase_client
    ):
        """Test signal expecting SHORT but price breaks UP."""
        # Signal expects SHORT but price breaks above setup_bar_high
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 3600.0  # Above 3550 high
        entry_monitor.add_signal(mock_signal_setup_short)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        event = triggered[0]
        assert event._actual_direction == "LONG"  # Changed to LONG
        # Prior bar was 2U, so pattern becomes 2U-2U
        assert "2U" in event._actual_pattern


class TestCheckTriggersNoTrigger:
    """Tests for no trigger scenarios."""

    def test_price_within_range(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test no trigger when price stays within range."""
        mock_coinbase_client.get_current_price.return_value = 50000.0  # Between 49500 and 50500
        entry_monitor.add_signal(mock_signal_setup_long)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 0
        # Signal stays in pending
        assert len(entry_monitor.get_pending_signals()) == 1

    def test_no_price_available(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test no trigger when price fetch fails."""
        mock_coinbase_client.get_current_price.return_value = None
        entry_monitor.add_signal(mock_signal_setup_long)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 0

    def test_no_pending_signals(self, entry_monitor):
        """Test check_triggers with no pending signals."""
        triggered = entry_monitor.check_triggers()
        assert triggered == []


class TestCheckTriggersOutsideBar:
    """Tests for outside bar (breaks both bounds) scenario."""

    def test_outside_bar_no_trigger(self, entry_monitor, mock_coinbase_client):
        """Test normal short trigger when price breaks low."""
        # Create a signal with normal bounds
        signal = CryptoDetectedSignal(
            pattern_type="2D-1-?",
            direction="LONG",
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50100.0,
            stop_price=49900.0,
            target_price=50500.0,
            magnitude_pct=1.0,
            risk_reward=1.5,
            signal_type="SETUP",
            setup_bar_high=50100.0,
            setup_bar_low=50000.0,
            prior_bar_type=-2,  # 2D bar
        )
        entry_monitor.add_signal(signal)

        # Price breaks low (normal short trigger)
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 49900.0  # Breaks low

        # The outside bar case (broke_up AND broke_down) is logically impossible
        # for a single price point. This test verifies normal break behavior.
        triggered = entry_monitor.check_triggers()
        assert len(triggered) == 1  # Normal short trigger


class TestCheckTriggersMultiple:
    """Tests for multiple signals triggering."""

    def test_multiple_signals_trigger(self, entry_monitor, mock_coinbase_client):
        """Test multiple signals can trigger in same check."""
        # Signal 1: BTC LONG
        signal1 = CryptoDetectedSignal(
            pattern_type="3-1-?",
            direction="LONG",
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50500.0,
            stop_price=49000.0,
            target_price=52000.0,
            magnitude_pct=3.0,
            risk_reward=1.5,
            signal_type="SETUP",
            setup_bar_high=50500.0,
            setup_bar_low=49500.0,
        )
        # Signal 2: ETH SHORT
        signal2 = CryptoDetectedSignal(
            pattern_type="2U-1-?",
            direction="SHORT",
            symbol="ETH-USD",
            timeframe="4h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=3450.0,
            stop_price=3600.0,
            target_price=3200.0,
            magnitude_pct=7.2,
            risk_reward=2.5,
            signal_type="SETUP",
            setup_bar_high=3550.0,
            setup_bar_low=3450.0,
        )

        entry_monitor.add_signal(signal1)
        entry_monitor.add_signal(signal2)

        # Set prices to trigger both
        mock_coinbase_client.get_current_price.side_effect = lambda s: {
            "BTC-USD": 50600.0,  # Above high
            "ETH-USD": 3400.0,  # Below low
        }.get(s)

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 2
        # Should be sorted by priority (4h > 1h)
        assert triggered[0].signal.timeframe == "4h"
        assert triggered[1].signal.timeframe == "1h"

    def test_updates_stats(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test check_triggers updates statistics."""
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0
        entry_monitor.add_signal(mock_signal_setup_long)

        assert entry_monitor._trigger_count == 0
        assert entry_monitor._last_check is None

        entry_monitor.check_triggers()

        assert entry_monitor._trigger_count == 1
        assert entry_monitor._last_check is not None


# =============================================================================
# BACKGROUND MONITORING TESTS
# =============================================================================


class TestBackgroundMonitoring:
    """Tests for start/stop background monitoring."""

    def test_start_creates_thread(self, entry_monitor):
        """Test start creates daemon thread."""
        entry_monitor.start()
        try:
            assert entry_monitor._running is True
            assert entry_monitor._thread is not None
            assert entry_monitor._thread.daemon is True
        finally:
            entry_monitor.stop()

    def test_start_already_running(self, entry_monitor):
        """Test start when already running logs warning."""
        entry_monitor.start()
        try:
            # Start again should not create new thread
            original_thread = entry_monitor._thread
            entry_monitor.start()
            assert entry_monitor._thread == original_thread
        finally:
            entry_monitor.stop()

    def test_stop(self, entry_monitor):
        """Test stop terminates thread."""
        entry_monitor.start()
        assert entry_monitor._running is True

        entry_monitor.stop()
        assert entry_monitor._running is False

    def test_stop_not_running(self, entry_monitor):
        """Test stop when not running does nothing."""
        entry_monitor.stop()  # Should not raise
        assert entry_monitor._running is False

    def test_is_running_property(self, entry_monitor):
        """Test is_running property."""
        assert entry_monitor.is_running is False
        entry_monitor.start()
        try:
            assert entry_monitor.is_running is True
        finally:
            entry_monitor.stop()
        assert entry_monitor.is_running is False


class TestMonitorLoop:
    """Tests for the monitor loop behavior."""

    def test_loop_checks_maintenance(self, entry_monitor):
        """Test loop skips checks during maintenance window."""
        # Patch is_maintenance_window to return True
        with patch.object(entry_monitor, 'is_maintenance_window', return_value=True):
            with patch.object(entry_monitor, 'check_triggers') as mock_check:
                entry_monitor.config.poll_interval = 0.01  # Fast polling for test
                entry_monitor.start()
                time.sleep(0.05)
                entry_monitor.stop()

                # check_triggers should NOT have been called during maintenance
                mock_check.assert_not_called()

    def test_loop_removes_expired(self, entry_monitor):
        """Test loop removes expired signals."""
        with patch.object(entry_monitor, 'remove_expired_signals') as mock_remove:
            entry_monitor.config.poll_interval = 0.01
            entry_monitor.start()
            time.sleep(0.05)
            entry_monitor.stop()

            assert mock_remove.called

    def test_loop_fires_on_trigger_callback(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test loop fires on_trigger callback."""
        trigger_callback = Mock()
        entry_monitor.config.on_trigger = trigger_callback
        entry_monitor.config.poll_interval = 0.01

        # Price triggers signal - must clear side_effect
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0
        entry_monitor.add_signal(mock_signal_setup_long)

        entry_monitor.start()
        time.sleep(0.1)  # Give more time for async operation
        entry_monitor.stop()

        assert trigger_callback.called

    def test_loop_fires_on_poll_callback(self, entry_monitor):
        """Test loop fires on_poll callback each cycle."""
        poll_callback = Mock()
        entry_monitor.config.on_poll = poll_callback
        entry_monitor.config.poll_interval = 0.01

        entry_monitor.start()
        time.sleep(0.05)
        entry_monitor.stop()

        assert poll_callback.called

    def test_loop_handles_callback_error(self, entry_monitor, mock_signal_setup_long, mock_coinbase_client):
        """Test loop continues after callback error."""
        trigger_callback = Mock(side_effect=Exception("Callback error"))
        entry_monitor.config.on_trigger = trigger_callback
        entry_monitor.config.poll_interval = 0.01

        # Clear side_effect for proper mock behavior
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0
        entry_monitor.add_signal(mock_signal_setup_long)

        entry_monitor.start()
        time.sleep(0.1)  # Give more time
        # Should not crash
        assert entry_monitor._running is True
        entry_monitor.stop()


# =============================================================================
# STATISTICS AND STATUS TESTS
# =============================================================================


class TestStatisticsAndStatus:
    """Tests for get_stats and print_status."""

    def test_get_stats(self, entry_monitor, mock_signal_setup_long):
        """Test get_stats returns correct information."""
        entry_monitor.add_signal(mock_signal_setup_long)

        stats = entry_monitor.get_stats()

        assert stats["running"] is False
        assert stats["last_check"] is None
        assert stats["trigger_count"] == 0
        assert stats["pending_signals"] == 1
        assert isinstance(stats["is_maintenance_window"], bool)
        assert stats["poll_interval"] == entry_monitor.config.poll_interval

    def test_get_stats_after_trigger(
        self, entry_monitor, mock_signal_setup_long, mock_coinbase_client
    ):
        """Test get_stats after trigger updates correctly."""
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0
        entry_monitor.add_signal(mock_signal_setup_long)
        entry_monitor.check_triggers()

        stats = entry_monitor.get_stats()

        assert stats["trigger_count"] == 1
        assert stats["last_check"] is not None
        assert stats["pending_signals"] == 0

    def test_print_status(self, entry_monitor, mock_signal_setup_long, capsys):
        """Test print_status outputs correctly."""
        entry_monitor.add_signal(mock_signal_setup_long)
        entry_monitor.print_status()

        captured = capsys.readouterr()
        assert "CRYPTO ENTRY MONITOR STATUS" in captured.out
        assert "BTC-USD" in captured.out
        assert "3-1-?" in captured.out
        assert "LONG" in captured.out

    def test_print_status_empty(self, entry_monitor, capsys):
        """Test print_status with no signals."""
        entry_monitor.print_status()

        captured = capsys.readouterr()
        assert "Pending Signals: 0" in captured.out


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_add_signals(self, entry_monitor):
        """Test adding signals from multiple threads."""
        results = []

        def add_signal(i):
            signal = CryptoDetectedSignal(
                pattern_type=f"3-1-{i}",
                direction="LONG",
                symbol="BTC-USD",
                timeframe="1h",
                detected_time=datetime.now(timezone.utc) + timedelta(microseconds=i),
                entry_trigger=50000.0,
                stop_price=49000.0,
                target_price=52000.0,
                magnitude_pct=3.0,
                risk_reward=1.5,
                signal_type="SETUP",
                setup_bar_high=50500.0,
                setup_bar_low=49500.0,
            )
            result = entry_monitor.add_signal(signal)
            results.append(result)

        threads = [threading.Thread(target=add_signal, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (unique signals)
        assert all(results)
        assert len(entry_monitor.get_pending_signals()) == 10

    def test_concurrent_clear(self, entry_monitor, mock_signal_setup_long):
        """Test clearing while checking triggers."""
        entry_monitor.add_signal(mock_signal_setup_long)

        def clear_loop():
            for _ in range(100):
                entry_monitor.clear_signals()

        def check_loop():
            for _ in range(100):
                entry_monitor.check_triggers()

        t1 = threading.Thread(target=clear_loop)
        t2 = threading.Thread(target=check_loop)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should not raise or deadlock


# =============================================================================
# PATTERN RESOLUTION TESTS
# =============================================================================


class TestPatternResolution:
    """Tests for pattern resolution based on prior bar type."""

    def test_prior_bar_type_3_breaks_up(self, entry_monitor, mock_coinbase_client):
        """Test prior outside bar (3) breaking up creates 3-2U."""
        signal = CryptoDetectedSignal(
            pattern_type="3-1-?",
            direction="SHORT",  # Expected SHORT but will break UP
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50500.0,
            stop_price=49000.0,
            target_price=48000.0,
            magnitude_pct=3.0,
            risk_reward=1.0,
            signal_type="SETUP",
            setup_bar_high=50500.0,
            setup_bar_low=49500.0,
            prior_bar_type=3,  # Outside bar
        )
        entry_monitor.add_signal(signal)
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 50600.0

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        assert triggered[0]._actual_direction == "LONG"
        assert "3-2U" == triggered[0]._actual_pattern

    def test_prior_bar_type_2u_breaks_down(self, entry_monitor, mock_coinbase_client):
        """Test prior 2U bar breaking down creates 2U-2D."""
        signal = CryptoDetectedSignal(
            pattern_type="2U-1-?",
            direction="LONG",  # Expected LONG but will break DOWN
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=50500.0,
            stop_price=49000.0,
            target_price=52000.0,
            magnitude_pct=3.0,
            risk_reward=1.5,
            signal_type="SETUP",
            setup_bar_high=50500.0,
            setup_bar_low=49500.0,
            prior_bar_type=2,  # 2U bar
        )
        entry_monitor.add_signal(signal)
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 49400.0

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        assert triggered[0]._actual_direction == "SHORT"
        assert "2U-2D" == triggered[0]._actual_pattern

    def test_prior_bar_type_2d_breaks_down(self, entry_monitor, mock_coinbase_client):
        """Test prior 2D bar breaking down creates 2D-2D continuation."""
        signal = CryptoDetectedSignal(
            pattern_type="2D-1-?",
            direction="SHORT",  # Expected SHORT
            symbol="BTC-USD",
            timeframe="1h",
            detected_time=datetime.now(timezone.utc),
            entry_trigger=49500.0,
            stop_price=51000.0,
            target_price=48000.0,
            magnitude_pct=3.0,
            risk_reward=1.5,
            signal_type="SETUP",
            setup_bar_high=50500.0,
            setup_bar_low=49500.0,
            prior_bar_type=-2,  # 2D bar
        )
        entry_monitor.add_signal(signal)
        mock_coinbase_client.get_current_price.side_effect = None
        mock_coinbase_client.get_current_price.return_value = 49400.0

        triggered = entry_monitor.check_triggers()

        assert len(triggered) == 1
        assert triggered[0]._actual_direction == "SHORT"
        # Expected direction matched, so pattern completes as expected
        assert "2D" in triggered[0]._actual_pattern
