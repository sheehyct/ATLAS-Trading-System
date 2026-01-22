"""
Tests for Discord Alerter - Session EQUITY-76

Comprehensive test coverage for strat/signal_automation/alerters/discord_alerter.py
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import Optional

# Import the module under test
from strat.signal_automation.alerters.discord_alerter import (
    DiscordAlerter,
    COLORS,
    NOTIFY_USER_ID,
)
from strat.signal_automation.alerters.base import BaseAlerter


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def valid_webhook_url():
    """Valid Discord webhook URL for testing."""
    return "https://discord.com/api/webhooks/123456789/abcdefghijklmnop"


@pytest.fixture
def alerter(valid_webhook_url):
    """Create a DiscordAlerter instance for testing."""
    return DiscordAlerter(webhook_url=valid_webhook_url)


@pytest.fixture
def mock_signal():
    """Create a mock StoredSignal for testing."""
    signal = Mock()
    signal.signal_key = "SPY_3-1-2U_1D_2026-01-21"
    signal.symbol = "SPY"
    signal.pattern_type = "3-1-2U"
    signal.direction = "CALL"
    signal.timeframe = "1D"
    signal.entry_trigger = 600.00
    signal.target_price = 609.00
    signal.stop_price = 595.00
    signal.magnitude_pct = 1.5
    signal.risk_reward = 1.8
    signal.vix = 15.5
    signal.market_regime = "TREND_BULL"
    signal.detected_time = datetime(2026, 1, 21, 10, 30, 0)
    signal.continuity_strength = 4
    signal.tfc_score = 4
    signal.setup_bar_high = 600.50
    signal.setup_bar_low = 594.00
    signal.signal_type = "SETUP"
    return signal


@pytest.fixture
def mock_exit_signal():
    """Create a mock ExitSignal for testing."""
    exit_signal = Mock()
    exit_signal.osi_symbol = "SPY260121C00600000"
    exit_signal.signal_key = "SPY_3-1-2U_1D_2026-01-21"
    exit_signal.unrealized_pnl = 150.00
    exit_signal.dte = 5
    exit_signal.underlying_price = 608.50
    exit_signal.current_option_price = 12.50
    exit_signal.reason = Mock(value="TARGET")
    return exit_signal


@pytest.fixture
def mock_execution_result():
    """Create a mock ExecutionResult for testing."""
    result = Mock()
    result.order_id = "order_12345678901234567890"
    result.status = "filled"
    return result


# ============================================================================
# Test COLORS Constant
# ============================================================================

class TestColorsConstant:
    """Tests for the COLORS constant definition."""

    def test_colors_has_call(self):
        """COLORS should have CALL color defined."""
        assert 'CALL' in COLORS
        assert COLORS['CALL'] == 0x00FF00  # Green

    def test_colors_has_put(self):
        """COLORS should have PUT color defined."""
        assert 'PUT' in COLORS
        assert COLORS['PUT'] == 0xFF0000  # Red

    def test_colors_has_info(self):
        """COLORS should have INFO color defined."""
        assert 'INFO' in COLORS
        assert COLORS['INFO'] == 0x0099FF  # Blue

    def test_colors_has_warning(self):
        """COLORS should have WARNING color defined."""
        assert 'WARNING' in COLORS
        assert COLORS['WARNING'] == 0xFFAA00  # Orange

    def test_colors_has_error(self):
        """COLORS should have ERROR color defined."""
        assert 'ERROR' in COLORS
        assert COLORS['ERROR'] == 0xFF0000  # Red

    def test_notify_user_id_defined(self):
        """NOTIFY_USER_ID should be defined."""
        assert NOTIFY_USER_ID is not None
        assert isinstance(NOTIFY_USER_ID, str)


# ============================================================================
# Test DiscordAlerter Initialization
# ============================================================================

class TestDiscordAlerterInit:
    """Tests for DiscordAlerter initialization."""

    def test_init_with_valid_url(self, valid_webhook_url):
        """Should initialize with valid webhook URL."""
        alerter = DiscordAlerter(webhook_url=valid_webhook_url)
        assert alerter.webhook_url == valid_webhook_url
        assert alerter.username == 'STRAT Signal Bot'
        assert alerter.avatar_url is None
        assert alerter.retry_attempts == 3
        assert alerter.retry_delay == 1.0

    def test_init_with_custom_username(self, valid_webhook_url):
        """Should accept custom username."""
        alerter = DiscordAlerter(
            webhook_url=valid_webhook_url,
            username='Custom Bot'
        )
        assert alerter.username == 'Custom Bot'

    def test_init_with_avatar_url(self, valid_webhook_url):
        """Should accept avatar URL."""
        avatar = "https://example.com/avatar.png"
        alerter = DiscordAlerter(
            webhook_url=valid_webhook_url,
            avatar_url=avatar
        )
        assert alerter.avatar_url == avatar

    def test_init_with_custom_retry_settings(self, valid_webhook_url):
        """Should accept custom retry settings."""
        alerter = DiscordAlerter(
            webhook_url=valid_webhook_url,
            retry_attempts=5,
            retry_delay=2.0
        )
        assert alerter.retry_attempts == 5
        assert alerter.retry_delay == 2.0

    def test_init_empty_url_raises(self):
        """Should raise ValueError for empty URL."""
        with pytest.raises(ValueError, match="Discord webhook URL is required"):
            DiscordAlerter(webhook_url='')

    def test_init_none_url_raises(self):
        """Should raise ValueError for None URL."""
        with pytest.raises(ValueError, match="Discord webhook URL is required"):
            DiscordAlerter(webhook_url=None)

    def test_init_invalid_url_format_raises(self):
        """Should raise ValueError for invalid URL format."""
        with pytest.raises(ValueError, match="Invalid Discord webhook URL format"):
            DiscordAlerter(webhook_url='https://invalid.com/webhook')

    def test_init_http_url_raises(self):
        """Should raise ValueError for non-HTTPS URL."""
        with pytest.raises(ValueError, match="Invalid Discord webhook URL format"):
            DiscordAlerter(webhook_url='http://discord.com/api/webhooks/123/abc')

    def test_inherits_from_base_alerter(self, alerter):
        """Should inherit from BaseAlerter."""
        assert isinstance(alerter, BaseAlerter)
        assert alerter.name == 'discord'

    def test_rate_limit_constants(self, alerter):
        """Should have rate limit constants defined."""
        assert alerter.RATE_LIMIT_WINDOW == 60
        assert alerter.RATE_LIMIT_MAX == 25

    def test_request_times_initialized_empty(self, alerter):
        """Should initialize with empty request times list."""
        assert alerter._request_times == []


# ============================================================================
# Test Rate Limiting
# ============================================================================

class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_check_rate_limit_empty_list(self, alerter):
        """Should allow request when no previous requests."""
        assert alerter._check_rate_limit() is True

    def test_check_rate_limit_under_limit(self, alerter):
        """Should allow request when under rate limit."""
        # Add some requests
        now = time.time()
        alerter._request_times = [now - 10, now - 20, now - 30]
        assert alerter._check_rate_limit() is True

    def test_check_rate_limit_at_limit(self, alerter):
        """Should deny request when at rate limit."""
        now = time.time()
        # Add 25 requests (at limit)
        alerter._request_times = [now - i for i in range(25)]
        assert alerter._check_rate_limit() is False

    def test_check_rate_limit_clears_old_requests(self, alerter):
        """Should clear requests older than window."""
        now = time.time()
        # Add old requests (older than 60 seconds)
        alerter._request_times = [now - 100, now - 120, now - 150]
        # Should clear old requests and allow
        assert alerter._check_rate_limit() is True
        assert len(alerter._request_times) == 0

    def test_check_rate_limit_mixed_old_new(self, alerter):
        """Should keep new requests, clear old ones."""
        now = time.time()
        # Mix of old and new requests
        alerter._request_times = [now - 100, now - 10, now - 5]
        assert alerter._check_rate_limit() is True
        assert len(alerter._request_times) == 2  # Only 2 recent ones

    def test_record_request(self, alerter):
        """Should record request timestamp."""
        assert len(alerter._request_times) == 0
        alerter._record_request()
        assert len(alerter._request_times) == 1
        assert time.time() - alerter._request_times[0] < 1  # Within 1 second


# ============================================================================
# Test Webhook Sending
# ============================================================================

class TestSendWebhook:
    """Tests for _send_webhook method."""

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_success(self, mock_post, alerter):
        """Should return True on successful send (204)."""
        mock_post.return_value = Mock(status_code=204)

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        mock_post.assert_called_once()

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_records_request(self, mock_post, alerter):
        """Should record request after successful send."""
        mock_post.return_value = Mock(status_code=204)

        alerter._send_webhook({'content': 'test'})

        assert len(alerter._request_times) == 1

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_client_error(self, mock_post, alerter):
        """Should return False on 4xx error."""
        mock_post.return_value = Mock(status_code=400, text='Bad Request')

        result = alerter._send_webhook({'content': 'test'})

        assert result is False

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_rate_limited_by_discord(self, mock_post, alerter):
        """Should retry after Discord rate limit (429)."""
        # First call returns 429, second returns 204
        mock_post.side_effect = [
            Mock(status_code=429, json=lambda: {'retry_after': 0.1}),
            Mock(status_code=204)
        ]

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        assert mock_post.call_count == 2

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_timeout_retry(self, mock_post, alerter):
        """Should retry on timeout."""
        import requests as req
        # First call times out, second succeeds
        mock_post.side_effect = [
            req.exceptions.Timeout(),
            Mock(status_code=204)
        ]
        alerter.retry_delay = 0.01  # Fast retry for test

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        assert mock_post.call_count == 2

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_all_retries_fail(self, mock_post, alerter):
        """Should return False after all retries fail."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()
        alerter.retry_delay = 0.01
        alerter.retry_attempts = 2

        result = alerter._send_webhook({'content': 'test'})

        assert result is False
        assert mock_post.call_count == 2

    @patch('strat.signal_automation.alerters.discord_alerter.requests.post')
    def test_send_webhook_request_exception(self, mock_post, alerter):
        """Should handle RequestException."""
        import requests as req
        mock_post.side_effect = req.exceptions.RequestException("Connection error")
        alerter.retry_delay = 0.01
        alerter.retry_attempts = 2

        result = alerter._send_webhook({'content': 'test'})

        assert result is False


# ============================================================================
# Test Entry Trigger Display
# ============================================================================

class TestGetEntryTriggerDisplay:
    """Tests for _get_entry_trigger_display method."""

    def test_normal_signal_uses_entry_trigger(self, alerter, mock_signal):
        """Should use entry_trigger for normal signals."""
        mock_signal.entry_trigger = 600.00
        mock_signal.signal_type = "COMPLETED"

        result = alerter._get_entry_trigger_display(mock_signal)

        assert result == "$600.00"

    def test_setup_signal_call_uses_setup_bar_high(self, alerter, mock_signal):
        """Should use setup_bar_high for SETUP CALL signals with 0.0 entry."""
        mock_signal.entry_trigger = 0.0
        mock_signal.signal_type = "SETUP"
        mock_signal.direction = "CALL"
        mock_signal.setup_bar_high = 605.50

        result = alerter._get_entry_trigger_display(mock_signal)

        assert result == "$605.50"

    def test_setup_signal_put_uses_setup_bar_low(self, alerter, mock_signal):
        """Should use setup_bar_low for SETUP PUT signals with 0.0 entry."""
        mock_signal.entry_trigger = 0.0
        mock_signal.signal_type = "SETUP"
        mock_signal.direction = "PUT"
        mock_signal.setup_bar_low = 594.25

        result = alerter._get_entry_trigger_display(mock_signal)

        assert result == "$594.25"

    def test_setup_with_valid_entry_trigger_still_uses_setup_bar(self, alerter, mock_signal):
        """SETUP signals always use setup_bar levels, even if entry_trigger is set."""
        mock_signal.entry_trigger = 600.00
        mock_signal.signal_type = "SETUP"
        mock_signal.direction = "CALL"
        mock_signal.setup_bar_high = 605.50

        result = alerter._get_entry_trigger_display(mock_signal)

        # SETUP signals use setup_bar_high for CALL (the level that needs to break)
        assert result == "$605.50"


# ============================================================================
# Test Signal Embed Creation
# ============================================================================

class TestCreateSignalEmbed:
    """Tests for _create_signal_embed method."""

    def test_embed_has_title(self, alerter, mock_signal):
        """Embed should have correct title format."""
        embed = alerter._create_signal_embed(mock_signal)

        assert 'title' in embed
        assert 'SPY' in embed['title']
        assert '3-1-2U' in embed['title']
        assert '1D' in embed['title']

    def test_embed_call_color(self, alerter, mock_signal):
        """CALL signals should have green color."""
        mock_signal.direction = "CALL"
        embed = alerter._create_signal_embed(mock_signal)

        assert embed['color'] == COLORS['CALL']

    def test_embed_put_color(self, alerter, mock_signal):
        """PUT signals should have red color."""
        mock_signal.direction = "PUT"
        embed = alerter._create_signal_embed(mock_signal)

        assert embed['color'] == COLORS['PUT']

    def test_embed_has_fields(self, alerter, mock_signal):
        """Embed should have required fields."""
        embed = alerter._create_signal_embed(mock_signal)

        field_names = [f['name'] for f in embed['fields']]
        assert 'Entry Trigger' in field_names
        assert 'Target' in field_names
        assert 'Stop' in field_names
        assert 'R:R Ratio' in field_names
        assert 'Magnitude' in field_names
        assert 'VIX' in field_names

    def test_embed_has_footer(self, alerter, mock_signal):
        """Embed should have footer with context."""
        embed = alerter._create_signal_embed(mock_signal)

        assert 'footer' in embed
        assert 'text' in embed['footer']

    def test_embed_has_timestamp(self, alerter, mock_signal):
        """Embed should have timestamp."""
        embed = alerter._create_signal_embed(mock_signal)

        assert 'timestamp' in embed


# ============================================================================
# Test Send Alert
# ============================================================================

class TestSendAlert:
    """Tests for send_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_send_alert_success(self, mock_webhook, alerter, mock_signal):
        """Should return True on success."""
        mock_webhook.return_value = True

        result = alerter.send_alert(mock_signal)

        assert result is True
        mock_webhook.assert_called_once()

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_send_alert_records_alert(self, mock_webhook, alerter, mock_signal):
        """Should record alert on success."""
        mock_webhook.return_value = True

        alerter.send_alert(mock_signal)

        assert mock_signal.signal_key in alerter._last_alert_time

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_send_alert_throttled(self, mock_webhook, alerter, mock_signal):
        """Should return True (not failure) when throttled."""
        # Pre-record an alert
        alerter.record_alert(mock_signal.signal_key)

        result = alerter.send_alert(mock_signal)

        assert result is True
        mock_webhook.assert_not_called()

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_send_alert_includes_avatar_if_set(self, mock_webhook, valid_webhook_url, mock_signal):
        """Should include avatar URL in payload if set."""
        avatar = "https://example.com/avatar.png"
        alerter = DiscordAlerter(
            webhook_url=valid_webhook_url,
            avatar_url=avatar
        )
        mock_webhook.return_value = True

        alerter.send_alert(mock_signal)

        call_args = mock_webhook.call_args[0][0]
        assert 'avatar_url' in call_args
        assert call_args['avatar_url'] == avatar


# ============================================================================
# Test Batch Alert
# ============================================================================

class TestSendBatchAlert:
    """Tests for send_batch_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_empty_list(self, mock_webhook, alerter):
        """Should return True for empty list."""
        result = alerter.send_batch_alert([])

        assert result is True
        mock_webhook.assert_not_called()

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_single_signal(self, mock_webhook, alerter, mock_signal):
        """Should send single signal in batch."""
        mock_webhook.return_value = True

        result = alerter.send_batch_alert([mock_signal])

        assert result is True
        mock_webhook.assert_called_once()

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_multiple_signals(self, mock_webhook, alerter, mock_signal):
        """Should send multiple signals in one batch."""
        mock_webhook.return_value = True
        signals = [mock_signal] * 5

        result = alerter.send_batch_alert(signals)

        assert result is True
        # All 5 should be in one message
        mock_webhook.assert_called_once()
        call_args = mock_webhook.call_args[0][0]
        assert len(call_args['embeds']) == 5

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_chunks_over_10(self, mock_webhook, alerter, mock_signal):
        """Should split batches of more than 10 signals."""
        mock_webhook.return_value = True
        signals = [mock_signal] * 15
        # Make unique keys to avoid throttling
        for i, s in enumerate(signals):
            s.signal_key = f"key_{i}"

        result = alerter.send_batch_alert(signals)

        assert result is True
        # Should be 2 calls: 10 + 5
        assert mock_webhook.call_count == 2

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_filters_throttled(self, mock_webhook, alerter, mock_signal):
        """Should filter out throttled signals."""
        mock_webhook.return_value = True
        # Pre-throttle the signal
        alerter.record_alert(mock_signal.signal_key)

        result = alerter.send_batch_alert([mock_signal])

        assert result is True
        mock_webhook.assert_not_called()  # Filtered out

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_batch_alert_partial_failure(self, mock_webhook, alerter, mock_signal):
        """Should return False if any batch fails."""
        # First chunk succeeds, second fails
        mock_webhook.side_effect = [True, False]
        signals = []
        for i in range(15):
            s = Mock()
            s.signal_key = f"key_{i}"
            s.symbol = mock_signal.symbol
            s.pattern_type = mock_signal.pattern_type
            s.direction = mock_signal.direction
            s.timeframe = mock_signal.timeframe
            s.entry_trigger = mock_signal.entry_trigger
            s.target_price = mock_signal.target_price
            s.stop_price = mock_signal.stop_price
            s.magnitude_pct = mock_signal.magnitude_pct
            s.risk_reward = mock_signal.risk_reward
            s.vix = mock_signal.vix
            s.market_regime = mock_signal.market_regime
            s.detected_time = mock_signal.detected_time
            s.continuity_strength = mock_signal.continuity_strength
            signals.append(s)

        result = alerter.send_batch_alert(signals)

        assert result is False


# ============================================================================
# Test Scan Summary
# ============================================================================

class TestSendScanSummary:
    """Tests for send_scan_summary method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_scan_summary_success(self, mock_webhook, alerter):
        """Should send scan summary successfully."""
        mock_webhook.return_value = True

        result = alerter.send_scan_summary(
            timeframe='1D',
            signals_found=5,
            symbols_scanned=100,
            duration_seconds=12.5
        )

        assert result is True
        mock_webhook.assert_called_once()

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_scan_summary_no_signals_info_color(self, mock_webhook, alerter):
        """Should use INFO color when no signals found."""
        mock_webhook.return_value = True

        alerter.send_scan_summary(
            timeframe='1D',
            signals_found=0,
            symbols_scanned=100,
            duration_seconds=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['INFO']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_scan_summary_with_signals_warning_color(self, mock_webhook, alerter):
        """Should use WARNING color when signals found."""
        mock_webhook.return_value = True

        alerter.send_scan_summary(
            timeframe='1D',
            signals_found=3,
            symbols_scanned=100,
            duration_seconds=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['WARNING']


# ============================================================================
# Test Daemon Status
# ============================================================================

class TestSendDaemonStatus:
    """Tests for send_daemon_status method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daemon_status_started(self, mock_webhook, alerter):
        """Should send started status."""
        mock_webhook.return_value = True

        result = alerter.send_daemon_status('Started', 'Daemon initialized')

        assert result is True
        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['INFO']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daemon_status_stopped_warning_color(self, mock_webhook, alerter):
        """Should use WARNING color for stopped status."""
        mock_webhook.return_value = True

        alerter.send_daemon_status('Stopped', 'Shutdown complete')

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['WARNING']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daemon_status_error_color(self, mock_webhook, alerter):
        """Should use ERROR color for error status."""
        mock_webhook.return_value = True

        alerter.send_daemon_status('Error', 'Connection failed')

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['ERROR']


# ============================================================================
# Test Error Alert
# ============================================================================

class TestSendErrorAlert:
    """Tests for send_error_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_error_alert_success(self, mock_webhook, alerter):
        """Should send error alert."""
        mock_webhook.return_value = True

        result = alerter.send_error_alert('ScanError', 'Failed to scan SPY')

        assert result is True

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_error_alert_uses_error_color(self, mock_webhook, alerter):
        """Should use ERROR color."""
        mock_webhook.return_value = True

        alerter.send_error_alert('ScanError', 'Failed')

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['ERROR']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_error_alert_truncates_long_message(self, mock_webhook, alerter):
        """Should truncate messages over 2000 chars."""
        mock_webhook.return_value = True
        long_message = 'x' * 3000

        alerter.send_error_alert('Error', long_message)

        call_args = mock_webhook.call_args[0][0]
        assert len(call_args['embeds'][0]['description']) <= 2000


# ============================================================================
# Test Exit Alert
# ============================================================================

class TestSendExitAlert:
    """Tests for send_exit_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_exit_alert_profit(self, mock_webhook, alerter, mock_exit_signal):
        """Should send profit exit alert with green color."""
        mock_webhook.return_value = True
        mock_exit_signal.unrealized_pnl = 150.00

        result = alerter.send_exit_alert(mock_exit_signal, {'id': 'order123'})

        assert result is True
        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['CALL']  # Green

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_exit_alert_loss(self, mock_webhook, alerter, mock_exit_signal):
        """Should send loss exit alert with red color."""
        mock_webhook.return_value = True
        mock_exit_signal.unrealized_pnl = -50.00

        alerter.send_exit_alert(mock_exit_signal, {'id': 'order123'})

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['PUT']  # Red

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_exit_alert_includes_order_id(self, mock_webhook, alerter, mock_exit_signal):
        """Should include truncated order ID."""
        mock_webhook.return_value = True

        alerter.send_exit_alert(mock_exit_signal, {'id': 'order_very_long_id_12345'})

        call_args = mock_webhook.call_args[0][0]
        fields = call_args['embeds'][0]['fields']
        order_field = next((f for f in fields if f['name'] == 'Order ID'), None)
        assert order_field is not None
        assert len(order_field['value']) <= 20  # Truncated


# ============================================================================
# Test Entry Alert
# ============================================================================

class TestSendEntryAlert:
    """Tests for send_entry_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_entry_alert_success(self, mock_webhook, alerter, mock_signal, mock_execution_result):
        """Should send entry alert successfully."""
        mock_webhook.return_value = True

        result = alerter.send_entry_alert(mock_signal, mock_execution_result)

        assert result is True

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_entry_alert_format(self, mock_webhook, alerter, mock_signal, mock_execution_result):
        """Should format entry alert for mobile."""
        mock_webhook.return_value = True
        mock_signal.tfc_score = 4

        alerter.send_entry_alert(mock_signal, mock_execution_result)

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        assert 'Entry:' in content
        assert mock_signal.symbol in content
        assert 'Target:' in content
        assert 'Stop:' in content
        assert 'TFC: 4/4' in content

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_entry_alert_mentions_user(self, mock_webhook, alerter, mock_signal, mock_execution_result):
        """Should mention user for notification."""
        mock_webhook.return_value = True

        alerter.send_entry_alert(mock_signal, mock_execution_result)

        call_args = mock_webhook.call_args[0][0]
        assert NOTIFY_USER_ID in call_args['content']


# ============================================================================
# Test Simple Exit Alert
# ============================================================================

class TestSendSimpleExitAlert:
    """Tests for send_simple_exit_alert method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_simple_exit_alert_success(self, mock_webhook, alerter):
        """Should send simple exit alert."""
        mock_webhook.return_value = True

        result = alerter.send_simple_exit_alert(
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            direction='CALL',
            exit_reason='Target Hit',
            pnl=150.00
        )

        assert result is True

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_simple_exit_alert_profit_format(self, mock_webhook, alerter):
        """Should format profit with plus sign."""
        mock_webhook.return_value = True

        alerter.send_simple_exit_alert(
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            direction='CALL',
            exit_reason='Target Hit',
            pnl=150.00
        )

        call_args = mock_webhook.call_args[0][0]
        assert '+$150.00' in call_args['content']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_simple_exit_alert_loss_format(self, mock_webhook, alerter):
        """Should format loss with minus sign."""
        mock_webhook.return_value = True

        alerter.send_simple_exit_alert(
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            direction='CALL',
            exit_reason='Stop',
            pnl=-75.00
        )

        call_args = mock_webhook.call_args[0][0]
        assert '-$75.00' in call_args['content']


# ============================================================================
# Test Daily Audit
# ============================================================================

class TestSendDailyAudit:
    """Tests for send_daily_audit method."""

    @pytest.fixture
    def audit_data(self):
        """Create sample audit data."""
        return {
            'date': '2026-01-21',
            'trades_today': 5,
            'wins': 3,
            'losses': 2,
            'total_pnl': 250.00,
            'profit_factor': 1.5,
            'open_positions': [
                {'symbol': 'SPY', 'pattern_type': '3-1-2U', 'timeframe': '1D',
                 'unrealized_pnl': 100, 'unrealized_pct': 0.1},
            ],
            'anomalies': []
        }

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daily_audit_success(self, mock_webhook, alerter, audit_data):
        """Should send daily audit successfully."""
        mock_webhook.return_value = True

        result = alerter.send_daily_audit(audit_data)

        assert result is True

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daily_audit_profit_green(self, mock_webhook, alerter, audit_data):
        """Should use green color for profitable day."""
        mock_webhook.return_value = True
        audit_data['total_pnl'] = 250.00

        alerter.send_daily_audit(audit_data)

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['CALL']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daily_audit_loss_red(self, mock_webhook, alerter, audit_data):
        """Should use red color for losing day."""
        mock_webhook.return_value = True
        audit_data['total_pnl'] = -150.00

        alerter.send_daily_audit(audit_data)

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['PUT']

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daily_audit_with_anomalies(self, mock_webhook, alerter, audit_data):
        """Should include anomalies in report."""
        mock_webhook.return_value = True
        audit_data['anomalies'] = ['Missed entry on SPY', 'Unusual slippage']

        alerter.send_daily_audit(audit_data)

        call_args = mock_webhook.call_args[0][0]
        fields = call_args['embeds'][0]['fields']
        anomaly_field = next((f for f in fields if 'Anomalies' in f['name']), None)
        assert anomaly_field is not None
        assert '2' in anomaly_field['name']  # Count in title

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_daily_audit_limits_positions(self, mock_webhook, alerter, audit_data):
        """Should limit open positions to 5."""
        mock_webhook.return_value = True
        audit_data['open_positions'] = [
            {'symbol': f'SYM{i}', 'pattern_type': '3-1-2U', 'timeframe': '1D',
             'unrealized_pnl': 10, 'unrealized_pct': 0.01}
            for i in range(10)
        ]

        alerter.send_daily_audit(audit_data)

        call_args = mock_webhook.call_args[0][0]
        fields = call_args['embeds'][0]['fields']
        pos_field = next((f for f in fields if 'Open Positions' in f['name']), None)
        assert '... +5 more' in pos_field['value']


# ============================================================================
# Test Connection Test
# ============================================================================

class TestConnectionTest:
    """Tests for test_connection method."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_connection_success(self, mock_webhook, alerter):
        """Should return True on successful connection."""
        mock_webhook.return_value = True

        result = alerter.test_connection()

        assert result is True

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_connection_failure(self, mock_webhook, alerter):
        """Should return False on failed connection."""
        mock_webhook.return_value = False

        result = alerter.test_connection()

        assert result is False

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_connection_sends_test_message(self, mock_webhook, alerter):
        """Should send test message content."""
        mock_webhook.return_value = True

        alerter.test_connection()

        call_args = mock_webhook.call_args[0][0]
        assert 'connected successfully' in call_args['content']


# ============================================================================
# Test Throttling (Inherited from BaseAlerter)
# ============================================================================

class TestThrottling:
    """Tests for throttling functionality inherited from BaseAlerter."""

    def test_set_throttle_interval(self, alerter):
        """Should allow setting throttle interval."""
        alerter.set_throttle_interval(120)
        assert alerter._min_interval_seconds == 120

    def test_is_throttled_false_initially(self, alerter, mock_signal):
        """Should not be throttled initially."""
        assert alerter.is_throttled(mock_signal.signal_key) is False

    def test_is_throttled_true_after_record(self, alerter, mock_signal):
        """Should be throttled after recording alert."""
        alerter.record_alert(mock_signal.signal_key)
        assert alerter.is_throttled(mock_signal.signal_key) is True

    def test_throttle_expires(self, alerter, mock_signal):
        """Should expire throttle after interval."""
        alerter._min_interval_seconds = 0  # Immediate expiry
        alerter.record_alert(mock_signal.signal_key)
        # Should expire immediately
        assert alerter.is_throttled(mock_signal.signal_key) is False


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_full_alert_workflow(self, mock_webhook, alerter, mock_signal, mock_exit_signal):
        """Test full workflow: entry -> exit alerts."""
        mock_webhook.return_value = True

        # Send entry alert
        entry_result = alerter.send_entry_alert(mock_signal, Mock())
        assert entry_result is True

        # Send exit alert
        exit_result = alerter.send_exit_alert(mock_exit_signal, {'id': 'order123'})
        assert exit_result is True

        assert mock_webhook.call_count == 2

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_high_volume_batch(self, mock_webhook, alerter, mock_signal):
        """Test handling many signals at once."""
        mock_webhook.return_value = True

        # Create 25 unique signals
        signals = []
        for i in range(25):
            s = Mock()
            s.signal_key = f"signal_{i}"
            s.symbol = mock_signal.symbol
            s.pattern_type = mock_signal.pattern_type
            s.direction = mock_signal.direction
            s.timeframe = mock_signal.timeframe
            s.entry_trigger = mock_signal.entry_trigger
            s.target_price = mock_signal.target_price
            s.stop_price = mock_signal.stop_price
            s.magnitude_pct = mock_signal.magnitude_pct
            s.risk_reward = mock_signal.risk_reward
            s.vix = mock_signal.vix
            s.market_regime = mock_signal.market_regime
            s.detected_time = mock_signal.detected_time
            s.continuity_strength = mock_signal.continuity_strength
            signals.append(s)

        result = alerter.send_batch_alert(signals)

        assert result is True
        # Should be 3 batches: 10 + 10 + 5
        assert mock_webhook.call_count == 3

    @patch.object(DiscordAlerter, '_send_webhook')
    def test_mixed_success_failure(self, mock_webhook, alerter, mock_signal):
        """Test mixed success and failure responses."""
        # Alternate success/failure
        mock_webhook.side_effect = [True, False, True]

        # Individual alerts
        r1 = alerter.send_alert(mock_signal)
        mock_signal.signal_key = "key2"
        alerter._last_alert_time.clear()  # Clear throttle
        r2 = alerter.send_alert(mock_signal)
        mock_signal.signal_key = "key3"
        r3 = alerter.send_alert(mock_signal)

        assert r1 is True
        assert r2 is False
        assert r3 is True
