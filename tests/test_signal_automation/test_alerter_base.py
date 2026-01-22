"""
Tests for strat/signal_automation/alerters/base.py

Covers:
- BaseAlerter abstract class
- Throttling mechanism
- Default batch alert implementation
- Signal message formatting

Session EQUITY-77: Test coverage for alerter base module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from strat.signal_automation.alerters.base import BaseAlerter
from strat.signal_automation.signal_store import StoredSignal


# =============================================================================
# Concrete Implementation for Testing
# =============================================================================

class TestAlerter(BaseAlerter):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, name: str = 'test'):
        super().__init__(name)
        self.alerts_sent = []
        self.connection_test_result = True

    def send_alert(self, signal: StoredSignal) -> bool:
        """Track sent alerts and return success."""
        self.alerts_sent.append(signal)
        return True

    def test_connection(self) -> bool:
        """Return configured test result."""
        return self.connection_test_result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def alerter():
    """Create test alerter instance."""
    return TestAlerter('test_alerter')


@pytest.fixture
def mock_signal():
    """Create a mock StoredSignal for testing."""
    signal = MagicMock(spec=StoredSignal)
    signal.symbol = 'SPY'
    signal.pattern_type = '3-1-2U'
    signal.direction = 'CALL'
    signal.timeframe = '1H'
    signal.entry_trigger = 480.50
    signal.target_price = 487.50
    signal.stop_price = 477.00
    signal.risk_reward = 2.0
    signal.magnitude_pct = 1.5
    signal.vix = 18.5
    signal.market_regime = 'TREND_BULL'
    signal.detected_time = datetime(2024, 1, 15, 10, 30)
    return signal


# =============================================================================
# BaseAlerter Initialization Tests
# =============================================================================

class TestBaseAlerterInit:
    """Tests for BaseAlerter initialization."""

    def test_init_sets_name(self, alerter):
        """Alerter stores provided name."""
        assert alerter.name == 'test_alerter'

    def test_init_empty_last_alert_time(self, alerter):
        """Alerter initializes with empty last alert time dict."""
        assert alerter._last_alert_time == {}

    def test_init_default_throttle_interval(self, alerter):
        """Alerter has default 60 second throttle interval."""
        assert alerter._min_interval_seconds == 60


# =============================================================================
# Throttle Interval Tests
# =============================================================================

class TestThrottleInterval:
    """Tests for throttle interval configuration."""

    def test_set_throttle_interval(self, alerter):
        """set_throttle_interval updates interval."""
        alerter.set_throttle_interval(120)
        assert alerter._min_interval_seconds == 120

    def test_set_throttle_interval_zero(self, alerter):
        """Throttle interval can be set to zero (disabled)."""
        alerter.set_throttle_interval(0)
        assert alerter._min_interval_seconds == 0

    def test_set_throttle_interval_large(self, alerter):
        """Throttle interval can be set to large values."""
        alerter.set_throttle_interval(3600)  # 1 hour
        assert alerter._min_interval_seconds == 3600


# =============================================================================
# Throttling Tests
# =============================================================================

class TestIsThrottled:
    """Tests for is_throttled method."""

    def test_is_throttled_unknown_key(self, alerter):
        """Unknown signal key is not throttled."""
        assert alerter.is_throttled('unknown_key') is False

    def test_is_throttled_after_record(self, alerter):
        """Signal is throttled immediately after recording."""
        alerter.record_alert('test_key')
        assert alerter.is_throttled('test_key') is True

    def test_is_throttled_after_interval_passed(self, alerter):
        """Signal is not throttled after interval passes."""
        alerter.set_throttle_interval(1)  # 1 second
        alerter._last_alert_time['test_key'] = datetime.now() - timedelta(seconds=2)
        assert alerter.is_throttled('test_key') is False

    def test_is_throttled_within_interval(self, alerter):
        """Signal is throttled within interval."""
        alerter.set_throttle_interval(60)
        alerter._last_alert_time['test_key'] = datetime.now() - timedelta(seconds=30)
        assert alerter.is_throttled('test_key') is True

    def test_is_throttled_exact_boundary(self, alerter):
        """Signal at exact boundary is still throttled."""
        alerter.set_throttle_interval(60)
        # Set to exactly 59 seconds ago - should still be throttled
        alerter._last_alert_time['test_key'] = datetime.now() - timedelta(seconds=59)
        assert alerter.is_throttled('test_key') is True

    def test_is_throttled_independent_keys(self, alerter):
        """Different signal keys have independent throttling."""
        alerter.record_alert('key_1')
        assert alerter.is_throttled('key_1') is True
        assert alerter.is_throttled('key_2') is False


# =============================================================================
# Record Alert Tests
# =============================================================================

class TestRecordAlert:
    """Tests for record_alert method."""

    def test_record_alert_stores_time(self, alerter):
        """record_alert stores current time for key."""
        before = datetime.now()
        alerter.record_alert('test_key')
        after = datetime.now()

        assert 'test_key' in alerter._last_alert_time
        assert before <= alerter._last_alert_time['test_key'] <= after

    def test_record_alert_updates_existing(self, alerter):
        """record_alert updates existing key with new time."""
        old_time = datetime.now() - timedelta(hours=1)
        alerter._last_alert_time['test_key'] = old_time

        alerter.record_alert('test_key')

        assert alerter._last_alert_time['test_key'] > old_time

    def test_record_alert_multiple_keys(self, alerter):
        """Multiple keys can be recorded independently."""
        alerter.record_alert('key_1')
        alerter.record_alert('key_2')
        alerter.record_alert('key_3')

        assert len(alerter._last_alert_time) == 3


# =============================================================================
# Send Batch Alert Tests
# =============================================================================

class TestSendBatchAlert:
    """Tests for send_batch_alert default implementation."""

    def test_send_batch_alert_empty_list(self, alerter):
        """Empty list returns True."""
        result = alerter.send_batch_alert([])
        assert result is True
        assert alerter.alerts_sent == []

    def test_send_batch_alert_single_signal(self, alerter, mock_signal):
        """Single signal batch works correctly."""
        result = alerter.send_batch_alert([mock_signal])
        assert result is True
        assert len(alerter.alerts_sent) == 1
        assert alerter.alerts_sent[0] == mock_signal

    def test_send_batch_alert_multiple_signals(self, alerter, mock_signal):
        """Multiple signals sent individually."""
        signals = [mock_signal, mock_signal, mock_signal]
        result = alerter.send_batch_alert(signals)
        assert result is True
        assert len(alerter.alerts_sent) == 3

    def test_send_batch_alert_partial_failure(self, alerter, mock_signal):
        """Partial failure returns False but continues sending."""
        # Create alerter that fails on second signal
        call_count = [0]
        original_send = alerter.send_alert

        def failing_send(signal):
            call_count[0] += 1
            if call_count[0] == 2:
                return False
            return original_send(signal)

        alerter.send_alert = failing_send

        signals = [mock_signal, mock_signal, mock_signal]
        result = alerter.send_batch_alert(signals)

        assert result is False
        assert call_count[0] == 3  # All signals attempted


# =============================================================================
# Format Signal Message Tests
# =============================================================================

class TestFormatSignalMessage:
    """Tests for format_signal_message method."""

    def test_format_includes_symbol(self, alerter, mock_signal):
        """Formatted message includes symbol."""
        msg = alerter.format_signal_message(mock_signal)
        assert 'SPY' in msg

    def test_format_includes_pattern(self, alerter, mock_signal):
        """Formatted message includes pattern type."""
        msg = alerter.format_signal_message(mock_signal)
        assert '3-1-2U' in msg

    def test_format_includes_direction(self, alerter, mock_signal):
        """Formatted message includes direction."""
        msg = alerter.format_signal_message(mock_signal)
        assert 'CALL' in msg

    def test_format_includes_timeframe(self, alerter, mock_signal):
        """Formatted message includes timeframe."""
        msg = alerter.format_signal_message(mock_signal)
        assert '1H' in msg

    def test_format_includes_prices(self, alerter, mock_signal):
        """Formatted message includes entry/target/stop prices."""
        msg = alerter.format_signal_message(mock_signal)
        assert '$480.50' in msg  # Entry
        assert '$487.50' in msg  # Target
        assert '$477.00' in msg  # Stop

    def test_format_includes_risk_reward(self, alerter, mock_signal):
        """Formatted message includes R:R ratio."""
        msg = alerter.format_signal_message(mock_signal)
        assert '2.00:1' in msg

    def test_format_includes_magnitude(self, alerter, mock_signal):
        """Formatted message includes magnitude percentage."""
        msg = alerter.format_signal_message(mock_signal)
        assert '1.50%' in msg

    def test_format_includes_vix(self, alerter, mock_signal):
        """Formatted message includes VIX level."""
        msg = alerter.format_signal_message(mock_signal)
        assert '18.5' in msg

    def test_format_includes_regime(self, alerter, mock_signal):
        """Formatted message includes market regime."""
        msg = alerter.format_signal_message(mock_signal)
        assert 'TREND_BULL' in msg

    def test_format_includes_detected_time(self, alerter, mock_signal):
        """Formatted message includes detection timestamp."""
        msg = alerter.format_signal_message(mock_signal)
        assert '2024-01-15' in msg
        assert '10:30:00' in msg

    def test_format_message_is_string(self, alerter, mock_signal):
        """Formatted message is a string."""
        msg = alerter.format_signal_message(mock_signal)
        assert isinstance(msg, str)

    def test_format_message_not_empty(self, alerter, mock_signal):
        """Formatted message is not empty."""
        msg = alerter.format_signal_message(mock_signal)
        assert len(msg) > 0


# =============================================================================
# Abstract Method Tests
# =============================================================================

class TestAbstractMethods:
    """Tests for abstract method requirements."""

    def test_send_alert_is_abstract(self):
        """send_alert must be implemented by subclass."""
        with pytest.raises(TypeError):
            # Cannot instantiate ABC without implementing abstract methods
            BaseAlerter('test')

    def test_concrete_implementation_works(self, alerter, mock_signal):
        """Concrete implementation can be instantiated and used."""
        assert alerter.send_alert(mock_signal) is True
        assert alerter.test_connection() is True
