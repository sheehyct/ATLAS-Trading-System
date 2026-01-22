"""
Tests for Crypto Discord Alerter - Session EQUITY-76

Comprehensive test coverage for crypto/alerters/discord_alerter.py
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from crypto.alerters.discord_alerter import (
    CryptoDiscordAlerter,
    COLORS,
    NOTIFY_USER_ID,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def valid_webhook_url():
    """Valid Discord webhook URL for testing."""
    return "https://discord.com/api/webhooks/123456789/abcdefghijklmnop"


@pytest.fixture
def alerter(valid_webhook_url):
    """Create a CryptoDiscordAlerter instance for testing."""
    return CryptoDiscordAlerter(webhook_url=valid_webhook_url)


@pytest.fixture
def mock_signal():
    """Create a mock CryptoDetectedSignal for testing."""
    signal = Mock()
    signal.symbol = "BTC-USD"
    signal.pattern_type = "3-1-2U"
    signal.direction = "LONG"
    signal.timeframe = "1H"
    signal.entry_trigger = 95000.00
    signal.target_price = 97500.00
    signal.stop_price = 93000.00
    signal.magnitude_pct = 2.5
    signal.risk_reward = 1.25
    signal.signal_type = "SETUP"
    signal.detected_time = datetime(2026, 1, 21, 10, 30, 0, tzinfo=timezone.utc)

    # Context mock
    signal.context = Mock()
    signal.context.tfc_score = 3
    signal.context.tfc_alignment = "3/4 Bullish"

    return signal


@pytest.fixture
def mock_trigger_event(mock_signal):
    """Create a mock CryptoTriggerEvent for testing."""
    event = Mock()
    event.signal = mock_signal
    event.trigger_price = 95100.00
    event.current_price = 95250.00
    event._actual_direction = "LONG"
    return event


# ============================================================================
# Test COLORS Constant
# ============================================================================

class TestColorsConstant:
    """Tests for the COLORS constant definition."""

    def test_colors_has_long(self):
        """COLORS should have LONG color defined."""
        assert 'LONG' in COLORS
        assert COLORS['LONG'] == 0x00FF00  # Green

    def test_colors_has_short(self):
        """COLORS should have SHORT color defined."""
        assert 'SHORT' in COLORS
        assert COLORS['SHORT'] == 0xFF0000  # Red

    def test_colors_has_info(self):
        """COLORS should have INFO color defined."""
        assert 'INFO' in COLORS
        assert COLORS['INFO'] == 0x0099FF  # Blue

    def test_colors_has_warning(self):
        """COLORS should have WARNING color defined."""
        assert 'WARNING' in COLORS
        assert COLORS['WARNING'] == 0xFFAA00  # Orange

    def test_colors_has_profit(self):
        """COLORS should have PROFIT color defined."""
        assert 'PROFIT' in COLORS
        assert COLORS['PROFIT'] == 0x00FF00  # Green

    def test_colors_has_loss(self):
        """COLORS should have LOSS color defined."""
        assert 'LOSS' in COLORS
        assert COLORS['LOSS'] == 0xFF0000  # Red

    def test_notify_user_id_defined(self):
        """NOTIFY_USER_ID should be defined."""
        assert NOTIFY_USER_ID is not None
        assert isinstance(NOTIFY_USER_ID, str)


# ============================================================================
# Test CryptoDiscordAlerter Initialization
# ============================================================================

class TestCryptoDiscordAlerterInit:
    """Tests for CryptoDiscordAlerter initialization."""

    def test_init_with_valid_url(self, valid_webhook_url):
        """Should initialize with valid webhook URL."""
        alerter = CryptoDiscordAlerter(webhook_url=valid_webhook_url)
        assert alerter.webhook_url == valid_webhook_url
        assert alerter.username == 'ATLAS Crypto Bot'
        assert alerter.avatar_url is None
        assert alerter.retry_attempts == 3
        assert alerter.retry_delay == 1.0

    def test_init_with_custom_username(self, valid_webhook_url):
        """Should accept custom username."""
        alerter = CryptoDiscordAlerter(
            webhook_url=valid_webhook_url,
            username='Custom Crypto Bot'
        )
        assert alerter.username == 'Custom Crypto Bot'

    def test_init_with_avatar_url(self, valid_webhook_url):
        """Should accept avatar URL."""
        avatar = "https://example.com/avatar.png"
        alerter = CryptoDiscordAlerter(
            webhook_url=valid_webhook_url,
            avatar_url=avatar
        )
        assert alerter.avatar_url == avatar

    def test_init_with_custom_retry_settings(self, valid_webhook_url):
        """Should accept custom retry settings."""
        alerter = CryptoDiscordAlerter(
            webhook_url=valid_webhook_url,
            retry_attempts=5,
            retry_delay=2.0
        )
        assert alerter.retry_attempts == 5
        assert alerter.retry_delay == 2.0

    def test_init_empty_url_raises(self):
        """Should raise ValueError for empty URL."""
        with pytest.raises(ValueError, match="Discord webhook URL is required"):
            CryptoDiscordAlerter(webhook_url='')

    def test_init_none_url_raises(self):
        """Should raise ValueError for None URL."""
        with pytest.raises(ValueError, match="Discord webhook URL is required"):
            CryptoDiscordAlerter(webhook_url=None)

    def test_init_invalid_url_format_raises(self):
        """Should raise ValueError for invalid URL format."""
        with pytest.raises(ValueError, match="Invalid Discord webhook URL format"):
            CryptoDiscordAlerter(webhook_url='https://invalid.com/webhook')

    def test_rate_limit_constants(self, alerter):
        """Should have rate limit constants defined."""
        assert alerter.RATE_LIMIT_WINDOW == 60
        assert alerter.RATE_LIMIT_MAX == 25

    def test_throttle_seconds_default(self, alerter):
        """Should have default throttle of 5 minutes."""
        assert alerter._throttle_seconds == 300


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
        now = time.time()
        alerter._request_times = [now - 10, now - 20, now - 30]
        assert alerter._check_rate_limit() is True

    def test_check_rate_limit_at_limit(self, alerter):
        """Should deny request when at rate limit."""
        now = time.time()
        alerter._request_times = [now - i for i in range(25)]
        assert alerter._check_rate_limit() is False

    def test_check_rate_limit_clears_old_requests(self, alerter):
        """Should clear requests older than window."""
        now = time.time()
        alerter._request_times = [now - 100, now - 120, now - 150]
        assert alerter._check_rate_limit() is True
        assert len(alerter._request_times) == 0

    def test_record_request(self, alerter):
        """Should record request timestamp."""
        assert len(alerter._request_times) == 0
        alerter._record_request()
        assert len(alerter._request_times) == 1


# ============================================================================
# Test Throttling
# ============================================================================

class TestThrottling:
    """Tests for signal throttling functionality."""

    def test_is_throttled_false_initially(self, alerter):
        """Should not be throttled initially."""
        assert alerter._is_throttled("BTC_1H_3-1-2U_LONG") is False

    def test_is_throttled_true_after_record(self, alerter):
        """Should be throttled after recording alert."""
        signal_key = "BTC_1H_3-1-2U_LONG"
        alerter._record_alert(signal_key)
        assert alerter._is_throttled(signal_key) is True

    def test_throttle_expires(self, alerter):
        """Should expire throttle after interval."""
        signal_key = "BTC_1H_3-1-2U_LONG"
        alerter._throttle_seconds = 0  # Immediate expiry
        alerter._record_alert(signal_key)
        assert alerter._is_throttled(signal_key) is False

    def test_get_signal_key(self, alerter, mock_signal):
        """Should generate correct signal key."""
        key = alerter._get_signal_key(mock_signal)
        assert key == "BTC-USD_1H_3-1-2U_LONG"


# ============================================================================
# Test Webhook Sending
# ============================================================================

class TestSendWebhook:
    """Tests for _send_webhook method."""

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_success(self, mock_post, alerter):
        """Should return True on successful send (204)."""
        mock_post.return_value = Mock(status_code=204)

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        mock_post.assert_called_once()

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_records_request(self, mock_post, alerter):
        """Should record request after successful send."""
        mock_post.return_value = Mock(status_code=204)

        alerter._send_webhook({'content': 'test'})

        assert len(alerter._request_times) == 1

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_client_error(self, mock_post, alerter):
        """Should return False on 4xx error."""
        mock_post.return_value = Mock(status_code=400, text='Bad Request')

        result = alerter._send_webhook({'content': 'test'})

        assert result is False

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_rate_limited_by_discord(self, mock_post, alerter):
        """Should retry after Discord rate limit (429)."""
        mock_post.side_effect = [
            Mock(status_code=429, json=lambda: {'retry_after': 0.1}),
            Mock(status_code=204)
        ]

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        assert mock_post.call_count == 2

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_timeout_retry(self, mock_post, alerter):
        """Should retry on timeout."""
        import requests as req
        mock_post.side_effect = [
            req.exceptions.Timeout(),
            Mock(status_code=204)
        ]
        alerter.retry_delay = 0.01

        result = alerter._send_webhook({'content': 'test'})

        assert result is True
        assert mock_post.call_count == 2

    @patch('crypto.alerters.discord_alerter.requests.post')
    def test_send_webhook_all_retries_fail(self, mock_post, alerter):
        """Should return False after all retries fail."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()
        alerter.retry_delay = 0.01
        alerter.retry_attempts = 2

        result = alerter._send_webhook({'content': 'test'})

        assert result is False
        assert mock_post.call_count == 2


# ============================================================================
# Test Signal Embed Creation
# ============================================================================

class TestCreateSignalEmbed:
    """Tests for _create_signal_embed method."""

    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_embed_has_title(self, mock_intraday, mock_leverage, alerter, mock_signal):
        """Embed should have correct title format."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True

        embed = alerter._create_signal_embed(mock_signal)

        assert 'title' in embed
        assert 'BTC-USD' in embed['title']
        assert '3-1-2U' in embed['title']
        assert '1H' in embed['title']

    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_embed_long_color(self, mock_intraday, mock_leverage, alerter, mock_signal):
        """LONG signals should have green color."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True
        mock_signal.direction = "LONG"

        embed = alerter._create_signal_embed(mock_signal)

        assert embed['color'] == COLORS['LONG']

    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_embed_short_color(self, mock_intraday, mock_leverage, alerter, mock_signal):
        """SHORT signals should have red color."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True
        mock_signal.direction = "SHORT"

        embed = alerter._create_signal_embed(mock_signal)

        assert embed['color'] == COLORS['SHORT']

    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_embed_has_fields(self, mock_intraday, mock_leverage, alerter, mock_signal):
        """Embed should have required fields."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True

        embed = alerter._create_signal_embed(mock_signal)

        field_names = [f['name'] for f in embed['fields']]
        assert 'Entry Trigger' in field_names
        assert 'Target' in field_names
        assert 'Stop' in field_names
        assert 'R:R Ratio' in field_names
        assert 'Magnitude' in field_names
        assert 'Leverage' in field_names

    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_embed_has_tfc(self, mock_intraday, mock_leverage, alerter, mock_signal):
        """Embed should include TFC field when available."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True

        embed = alerter._create_signal_embed(mock_signal)

        field_names = [f['name'] for f in embed['fields']]
        assert 'TFC' in field_names


# ============================================================================
# Test Send Signal Alert
# ============================================================================

class TestSendSignalAlert:
    """Tests for send_signal_alert method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_send_signal_alert_success(self, mock_intraday, mock_leverage, mock_webhook, alerter, mock_signal):
        """Should return True on success."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True
        mock_webhook.return_value = True

        result = alerter.send_signal_alert(mock_signal)

        assert result is True
        mock_webhook.assert_called_once()

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_send_signal_alert_throttled(self, mock_intraday, mock_leverage, mock_webhook, alerter, mock_signal):
        """Should return True (not failure) when throttled."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True

        signal_key = alerter._get_signal_key(mock_signal)
        alerter._record_alert(signal_key)

        result = alerter.send_signal_alert(mock_signal)

        assert result is True
        mock_webhook.assert_not_called()


# ============================================================================
# Test Send Trigger Alert
# ============================================================================

class TestSendTriggerAlert:
    """Tests for send_trigger_alert method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_trigger_alert_success(self, mock_webhook, alerter, mock_trigger_event):
        """Should return True on success."""
        mock_webhook.return_value = True

        result = alerter.send_trigger_alert(mock_trigger_event)

        assert result is True

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_trigger_alert_format(self, mock_webhook, alerter, mock_trigger_event):
        """Should format trigger alert for mobile."""
        mock_webhook.return_value = True

        alerter.send_trigger_alert(mock_trigger_event)

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        assert 'TRIGGERED:' in content
        assert 'BTC-USD' in content
        assert 'Break @' in content


# ============================================================================
# Test Send Entry Alert
# ============================================================================

class TestSendEntryAlert:
    """Tests for send_entry_alert method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_entry_alert_success(self, mock_webhook, alerter, mock_signal):
        """Should return True on success."""
        mock_webhook.return_value = True

        result = alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=95100.00,
            quantity=0.1,
            leverage=10.0
        )

        assert result is True

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_entry_alert_format(self, mock_webhook, alerter, mock_signal):
        """Should format entry alert for mobile."""
        mock_webhook.return_value = True

        alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=95100.00,
            quantity=0.1,
            leverage=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        assert 'ENTRY:' in content
        assert 'BTC-USD' in content
        assert 'Pattern:' in content
        assert 'TFC:' in content

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_entry_alert_with_overrides(self, mock_webhook, alerter, mock_signal):
        """Should use override values when provided."""
        mock_webhook.return_value = True

        alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=95100.00,
            quantity=0.1,
            leverage=10.0,
            pattern_override='3-2D',
            direction_override='SHORT',
            stop_override=96000.00,
            target_override=93000.00
        )

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        assert 'SHORT' in content
        assert '3-2D' in content

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_entry_alert_resolves_setup_pattern(self, mock_webhook, alerter, mock_signal):
        """Should resolve ? to 2U/2D based on direction."""
        mock_webhook.return_value = True
        mock_signal.pattern_type = "3-?"

        alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=95100.00,
            quantity=0.1,
            leverage=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        # LONG should resolve ? to 2U
        assert '3-2U' in content


# ============================================================================
# Test Send Exit Alert
# ============================================================================

class TestSendExitAlert:
    """Tests for send_exit_alert method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_exit_alert_profit(self, mock_webhook, alerter):
        """Should send profit exit alert."""
        mock_webhook.return_value = True

        result = alerter.send_exit_alert(
            symbol='BTC-USD',
            direction='LONG',
            exit_reason='Target',
            entry_price=95000.00,
            exit_price=97500.00,
            pnl=250.00,
            pnl_pct=2.63
        )

        assert result is True
        call_args = mock_webhook.call_args[0][0]
        assert '+$250.00' in call_args['content']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_exit_alert_loss(self, mock_webhook, alerter):
        """Should send loss exit alert."""
        mock_webhook.return_value = True

        alerter.send_exit_alert(
            symbol='BTC-USD',
            direction='LONG',
            exit_reason='Stop',
            entry_price=95000.00,
            exit_price=93000.00,
            pnl=-200.00,
            pnl_pct=-2.11
        )

        call_args = mock_webhook.call_args[0][0]
        assert '-$200.00' in call_args['content']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_exit_alert_with_pattern(self, mock_webhook, alerter):
        """Should include pattern and timeframe when provided."""
        mock_webhook.return_value = True

        alerter.send_exit_alert(
            symbol='BTC-USD',
            direction='LONG',
            exit_reason='Target',
            entry_price=95000.00,
            exit_price=97500.00,
            pnl=250.00,
            pnl_pct=2.63,
            pattern_type='3-2U',
            timeframe='1H'
        )

        call_args = mock_webhook.call_args[0][0]
        assert '3-2U' in call_args['content']
        assert '1H' in call_args['content']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_send_exit_alert_with_duration(self, mock_webhook, alerter):
        """Should include duration when times and pattern provided."""
        mock_webhook.return_value = True
        entry_time = datetime(2026, 1, 21, 10, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2026, 1, 21, 14, 30, 0, tzinfo=timezone.utc)

        alerter.send_exit_alert(
            symbol='BTC-USD',
            direction='LONG',
            exit_reason='Target',
            entry_price=95000.00,
            exit_price=97500.00,
            pnl=250.00,
            pnl_pct=2.63,
            pattern_type='3-2U',  # Duration only shown with pattern
            timeframe='1H',
            entry_time=entry_time,
            exit_time=exit_time
        )

        call_args = mock_webhook.call_args[0][0]
        assert '4.5h' in call_args['content']


# ============================================================================
# Test Send Scan Summary
# ============================================================================

class TestSendScanSummary:
    """Tests for send_scan_summary method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_scan_summary_success(self, mock_webhook, alerter):
        """Should send scan summary successfully."""
        mock_webhook.return_value = True

        result = alerter.send_scan_summary(
            symbols_scanned=20,
            signals_found=5,
            duration_seconds=12.5
        )

        assert result is True
        mock_webhook.assert_called_once()

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_scan_summary_no_signals_info_color(self, mock_webhook, alerter):
        """Should use INFO color when no signals found."""
        mock_webhook.return_value = True

        alerter.send_scan_summary(
            symbols_scanned=20,
            signals_found=0,
            duration_seconds=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['INFO']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_scan_summary_with_signals_warning_color(self, mock_webhook, alerter):
        """Should use WARNING color when signals found."""
        mock_webhook.return_value = True

        alerter.send_scan_summary(
            symbols_scanned=20,
            signals_found=3,
            duration_seconds=10.0
        )

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['WARNING']


# ============================================================================
# Test Send Daemon Status
# ============================================================================

class TestSendDaemonStatus:
    """Tests for send_daemon_status method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_daemon_status_started(self, mock_webhook, alerter):
        """Should send started status."""
        mock_webhook.return_value = True

        result = alerter.send_daemon_status('Started', 'Daemon initialized')

        assert result is True
        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['INFO']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_daemon_status_stopped_warning_color(self, mock_webhook, alerter):
        """Should use WARNING color for stopped status."""
        mock_webhook.return_value = True

        alerter.send_daemon_status('Stopped', 'Shutdown complete')

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['WARNING']

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_daemon_status_error_color(self, mock_webhook, alerter):
        """Should use ERROR color for error status."""
        mock_webhook.return_value = True

        alerter.send_daemon_status('Error', 'Connection failed')

        call_args = mock_webhook.call_args[0][0]
        assert call_args['embeds'][0]['color'] == COLORS['ERROR']


# ============================================================================
# Test Connection Test
# ============================================================================

class TestConnectionTest:
    """Tests for test_connection method."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_connection_success(self, mock_webhook, alerter):
        """Should return True on successful connection."""
        mock_webhook.return_value = True

        result = alerter.test_connection()

        assert result is True

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_connection_failure(self, mock_webhook, alerter):
        """Should return False on failed connection."""
        mock_webhook.return_value = False

        result = alerter.test_connection()

        assert result is False

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_connection_sends_test_message(self, mock_webhook, alerter):
        """Should send test message content."""
        mock_webhook.return_value = True

        alerter.test_connection()

        call_args = mock_webhook.call_args[0][0]
        assert 'connected successfully' in call_args['content']


# ============================================================================
# Test Get Now ET
# ============================================================================

class TestGetNowET:
    """Tests for _get_now_et method."""

    def test_get_now_et_returns_datetime(self, alerter):
        """Should return datetime object."""
        result = alerter._get_now_et()
        assert isinstance(result, datetime)

    def test_get_now_et_reasonable_time(self, alerter):
        """Should return reasonable current time."""
        result = alerter._get_now_et()
        now_utc = datetime.now(timezone.utc)
        # Should be within 24 hours of UTC
        delta = abs((now_utc.replace(tzinfo=None) - result.replace(tzinfo=None)).total_seconds())
        assert delta < 24 * 3600


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    @patch('crypto.alerters.discord_alerter.get_max_leverage_for_symbol')
    @patch('crypto.alerters.discord_alerter.is_intraday_window')
    def test_full_trade_lifecycle(
        self, mock_intraday, mock_leverage, mock_webhook, alerter, mock_signal
    ):
        """Test full workflow: signal -> trigger -> entry -> exit."""
        mock_leverage.return_value = 10
        mock_intraday.return_value = True
        mock_webhook.return_value = True

        # 1. Signal alert
        signal_result = alerter.send_signal_alert(mock_signal)
        assert signal_result is True

        # 2. Trigger alert
        trigger_event = Mock()
        trigger_event.signal = mock_signal
        trigger_event.trigger_price = 95100.00
        trigger_event.current_price = 95200.00
        trigger_event._actual_direction = "LONG"
        trigger_result = alerter.send_trigger_alert(trigger_event)
        assert trigger_result is True

        # 3. Entry alert
        entry_result = alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=95200.00,
            quantity=0.1,
            leverage=10.0
        )
        assert entry_result is True

        # 4. Exit alert
        exit_result = alerter.send_exit_alert(
            symbol='BTC-USD',
            direction='LONG',
            exit_reason='Target',
            entry_price=95200.00,
            exit_price=97500.00,
            pnl=230.00,
            pnl_pct=2.42
        )
        assert exit_result is True

        assert mock_webhook.call_count == 4

    @patch.object(CryptoDiscordAlerter, '_send_webhook')
    def test_direction_flip_scenario(self, mock_webhook, alerter, mock_signal):
        """Test handling direction flip on SETUP signals."""
        mock_webhook.return_value = True
        mock_signal.pattern_type = "3-?"
        mock_signal.direction = "LONG"

        # Entry with direction flip (original LONG, actual SHORT)
        alerter.send_entry_alert(
            signal=mock_signal,
            entry_price=93000.00,
            quantity=0.1,
            leverage=10.0,
            direction_override='SHORT',
            stop_override=95000.00,
            target_override=91000.00
        )

        call_args = mock_webhook.call_args[0][0]
        content = call_args['content']
        # Should show SHORT (override) not LONG (original)
        assert 'SHORT' in content
        # Pattern should resolve to 2D for SHORT
        assert '3-2D' in content
