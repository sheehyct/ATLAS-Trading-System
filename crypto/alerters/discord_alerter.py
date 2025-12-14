"""
Crypto Discord Alerter - Session CRYPTO-5

Discord webhook alerter for crypto STRAT signals on perpetual futures.
Adapted from equities DiscordAlerter with crypto-specific formatting.

Features:
- Rich embeds with color-coded signals (green=LONG, red=SHORT)
- Crypto-specific fields (TFC score, funding rate, leverage tier)
- Rate limiting and retry logic
- Entry and exit alerts
"""

import requests
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from crypto.scanning.models import CryptoDetectedSignal, CryptoSignalContext
from crypto.scanning.entry_monitor import CryptoTriggerEvent
from crypto.config import is_intraday_window, get_max_leverage_for_symbol

# Timezone support
try:
    import pytz
    ET_TIMEZONE = pytz.timezone("America/New_York")
except ImportError:
    ET_TIMEZONE = None

logger = logging.getLogger(__name__)


# Discord embed color codes
COLORS = {
    'LONG': 0x00FF00,    # Green for bullish
    'SHORT': 0xFF0000,   # Red for bearish
    'INFO': 0x0099FF,    # Blue for info
    'WARNING': 0xFFAA00, # Orange for warnings
    'ERROR': 0xFF0000,   # Red for errors
    'PROFIT': 0x00FF00,  # Green for profit
    'LOSS': 0xFF0000,    # Red for loss
}


class CryptoDiscordAlerter:
    """
    Discord webhook alerter for crypto STRAT signals.

    Features:
    - Rich embeds with color-coded signals
    - Retry logic with exponential backoff
    - Rate limit handling (Discord 30 req/min)
    - Crypto-specific formatting (leverage, funding, TFC)

    Usage:
        alerter = CryptoDiscordAlerter(webhook_url)
        if alerter.test_connection():
            alerter.send_signal_alert(signal)
    """

    # Discord rate limit: 30 requests per 60 seconds
    RATE_LIMIT_WINDOW = 60
    RATE_LIMIT_MAX = 25  # Stay under limit

    def __init__(
        self,
        webhook_url: str,
        username: str = 'ATLAS Crypto Bot',
        avatar_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize crypto Discord alerter.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username to display
            avatar_url: Optional avatar URL for bot
            retry_attempts: Number of retry attempts on failure
            retry_delay: Base delay between retries (exponential backoff)
        """
        if not webhook_url:
            raise ValueError("Discord webhook URL is required")

        if not webhook_url.startswith('https://discord.com/api/webhooks/'):
            raise ValueError("Invalid Discord webhook URL format")

        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Rate limiting
        self._request_times: List[float] = []

        # Alert tracking for throttling
        self._sent_alerts: Dict[str, float] = {}
        self._throttle_seconds: int = 300  # 5 min throttle per signal

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        self._request_times = [
            t for t in self._request_times
            if now - t < self.RATE_LIMIT_WINDOW
        ]
        return len(self._request_times) < self.RATE_LIMIT_MAX

    def _record_request(self) -> None:
        """Record a request for rate limiting."""
        self._request_times.append(time.time())

    def _is_throttled(self, signal_key: str) -> bool:
        """Check if a signal is throttled."""
        if signal_key in self._sent_alerts:
            elapsed = time.time() - self._sent_alerts[signal_key]
            return elapsed < self._throttle_seconds
        return False

    def _record_alert(self, signal_key: str) -> None:
        """Record an alert was sent."""
        self._sent_alerts[signal_key] = time.time()

    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """Send payload to Discord webhook with retry logic."""
        if not self._check_rate_limit():
            wait_time = self.RATE_LIMIT_WINDOW - (
                time.time() - min(self._request_times)
            )
            logger.info(f"Rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10
                )

                self._record_request()

                if response.status_code == 204:
                    return True

                if response.status_code == 429:
                    retry_after = response.json().get('retry_after', 5)
                    logger.warning(f"Discord rate limited, retry after {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 400:
                    logger.error(
                        f"Discord webhook error: {response.status_code} - "
                        f"{response.text[:200]}"
                    )
                    return False

                return True

            except requests.exceptions.Timeout:
                logger.warning(f"Discord webhook timeout (attempt {attempt + 1})")
                time.sleep(self.retry_delay * (2 ** attempt))

            except requests.exceptions.RequestException as e:
                logger.error(f"Discord webhook request error: {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

        logger.error(f"Discord webhook failed after {self.retry_attempts} attempts")
        return False

    def _get_signal_key(self, signal: CryptoDetectedSignal) -> str:
        """Generate unique key for a signal."""
        return f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{signal.direction}"

    def _get_now_et(self) -> datetime:
        """Get current time in Eastern timezone."""
        now_utc = datetime.now(timezone.utc)
        if ET_TIMEZONE is not None:
            return now_utc.astimezone(ET_TIMEZONE)
        # Fallback: UTC-5 approximation
        from datetime import timedelta
        return (now_utc - timedelta(hours=5)).replace(tzinfo=None)

    def _create_signal_embed(self, signal: CryptoDetectedSignal) -> Dict[str, Any]:
        """Create Discord embed for a crypto signal."""
        color = COLORS.get(signal.direction, COLORS['INFO'])

        # Direction indicator
        direction_indicator = "[LONG]" if signal.direction == 'LONG' else "[SHORT]"

        # Title
        title = f"{direction_indicator} {signal.symbol} {signal.pattern_type} | {signal.timeframe}"

        # Description
        signal_type_str = "Setup waiting for trigger" if signal.signal_type == "SETUP" else "Pattern completed"
        description = f"**{signal.pattern_type}** {signal_type_str} on {signal.timeframe}"

        # Leverage tier info
        now_et = self._get_now_et()
        leverage = get_max_leverage_for_symbol(signal.symbol, now_et)
        tier = "INTRADAY" if is_intraday_window(now_et) else "SWING"

        # Fields
        fields = [
            {
                'name': 'Entry Trigger',
                'value': f'${signal.entry_trigger:,.2f}',
                'inline': True
            },
            {
                'name': 'Target',
                'value': f'${signal.target_price:,.2f}',
                'inline': True
            },
            {
                'name': 'Stop',
                'value': f'${signal.stop_price:,.2f}',
                'inline': True
            },
            {
                'name': 'R:R Ratio',
                'value': f'{signal.risk_reward:.2f}:1',
                'inline': True
            },
            {
                'name': 'Magnitude',
                'value': f'{signal.magnitude_pct:.2f}%',
                'inline': True
            },
            {
                'name': 'Leverage',
                'value': f'{leverage}x ({tier})',
                'inline': True
            },
        ]

        # Add TFC if available
        if signal.context.tfc_score > 0:
            fields.append({
                'name': 'TFC',
                'value': signal.context.tfc_alignment or f'{signal.context.tfc_score}/4',
                'inline': True
            })

        # Footer
        footer_parts = [
            f"Type: {signal.signal_type}",
            f"Detected: {signal.detected_time.strftime('%Y-%m-%d %H:%M UTC')}"
        ]

        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields,
            'footer': {
                'text': ' | '.join(footer_parts)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        return embed

    def send_signal_alert(self, signal: CryptoDetectedSignal) -> bool:
        """
        Send alert for a crypto signal.

        Args:
            signal: Signal to alert

        Returns:
            True if alert was sent successfully
        """
        signal_key = self._get_signal_key(signal)

        if self._is_throttled(signal_key):
            logger.debug(f"Discord alert throttled: {signal_key}")
            return True

        embed = self._create_signal_embed(signal)

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            self._record_alert(signal_key)
            logger.info(f"Discord crypto alert sent: {signal_key}")

        return success

    def send_trigger_alert(self, event: CryptoTriggerEvent) -> bool:
        """
        Send alert when a SETUP signal triggers.

        Args:
            event: Trigger event with signal and prices

        Returns:
            True if sent successfully
        """
        signal = event.signal
        color = COLORS.get(signal.direction, COLORS['INFO'])

        direction_str = "LONG" if signal.direction == "LONG" else "SHORT"

        # Clean, mobile-friendly message
        message = (
            f"**TRIGGERED: {signal.symbol} {signal.pattern_type} {signal.timeframe} {direction_str}**\n"
            f"Break @ ${event.trigger_price:,.2f} | "
            f"Current: ${event.current_price:,.2f} | "
            f"Target: ${signal.target_price:,.2f} | "
            f"Stop: ${signal.stop_price:,.2f}"
        )

        payload = {
            'username': self.username,
            'content': message,
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord trigger alert sent: {signal.symbol} {signal.pattern_type}")

        return success

    def send_entry_alert(
        self,
        signal: CryptoDetectedSignal,
        entry_price: float,
        quantity: float,
        leverage: float
    ) -> bool:
        """
        Send clean entry alert for mobile notifications.

        Args:
            signal: The signal being traded
            entry_price: Actual entry price
            quantity: Position quantity
            leverage: Leverage used

        Returns:
            True if sent successfully
        """
        direction_str = "LONG" if signal.direction == "LONG" else "SHORT"

        message = (
            f"**Entry: {signal.symbol} {signal.pattern_type} {signal.timeframe} {direction_str}**\n"
            f"@ ${entry_price:,.2f} | Qty: {quantity:.4f} | Leverage: {leverage}x\n"
            f"Target: ${signal.target_price:,.2f} | Stop: ${signal.stop_price:,.2f}"
        )

        payload = {
            'username': self.username,
            'content': message,
        }

        return self._send_webhook(payload)

    def send_exit_alert(
        self,
        symbol: str,
        direction: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float
    ) -> bool:
        """
        Send clean exit alert for mobile notifications.

        Args:
            symbol: Symbol that was closed
            direction: LONG or SHORT
            exit_reason: Why position closed (Target, Stop, Manual)
            entry_price: Original entry price
            exit_price: Exit price
            pnl: Dollar P&L
            pnl_pct: Percentage P&L

        Returns:
            True if sent successfully
        """
        pnl_indicator = "PROFIT" if pnl >= 0 else "LOSS"
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pnl_pct_str = f"+{pnl_pct:.2f}%" if pnl_pct >= 0 else f"{pnl_pct:.2f}%"

        message = (
            f"**Exit [{pnl_indicator}]: {symbol} {direction}**\n"
            f"{exit_reason} | Entry: ${entry_price:,.2f} -> Exit: ${exit_price:,.2f}\n"
            f"P/L: {pnl_str} ({pnl_pct_str})"
        )

        payload = {
            'username': self.username,
            'content': message,
        }

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord exit alert sent: {symbol} {exit_reason} P/L: {pnl_str}")

        return success

    def send_scan_summary(
        self,
        symbols_scanned: int,
        signals_found: int,
        duration_seconds: float
    ) -> bool:
        """
        Send scan completion summary.

        Args:
            symbols_scanned: Number of symbols scanned
            signals_found: Number of signals found
            duration_seconds: Scan duration

        Returns:
            True if sent successfully
        """
        color = COLORS['INFO'] if signals_found == 0 else COLORS['WARNING']

        embed = {
            'title': 'Crypto Scan Complete',
            'description': f'Scanned {symbols_scanned} symbols in {duration_seconds:.1f}s',
            'color': color,
            'fields': [
                {
                    'name': 'Signals Found',
                    'value': str(signals_found),
                    'inline': True
                },
                {
                    'name': 'Symbols',
                    'value': str(symbols_scanned),
                    'inline': True
                },
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        return self._send_webhook(payload)

    def send_daemon_status(self, status: str, details: str = '') -> bool:
        """
        Send daemon status update.

        Args:
            status: Status message (e.g., 'Started', 'Stopped', 'Error')
            details: Additional details

        Returns:
            True if sent successfully
        """
        if 'error' in status.lower():
            color = COLORS['ERROR']
        elif 'stopped' in status.lower():
            color = COLORS['WARNING']
        else:
            color = COLORS['INFO']

        embed = {
            'title': f'Crypto Daemon: {status}',
            'description': details or 'No additional details',
            'color': color,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        return self._send_webhook(payload)

    def test_connection(self) -> bool:
        """
        Test Discord webhook connection.

        Returns:
            True if connection test passed
        """
        payload = {
            'username': self.username,
            'content': 'ATLAS Crypto Bot connected successfully!',
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            logger.info("Discord crypto connection test passed")
        else:
            logger.error("Discord crypto connection test failed")

        return success
