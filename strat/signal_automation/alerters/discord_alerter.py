"""
Discord Alerter - Session 83K-46

Discord webhook alerter with rich embed formatting for STRAT signals.
Supports throttling, retry logic, and rate limit handling.

Discord Embed Format:
- Color-coded by direction (green=CALL, red=PUT)
- Structured fields for entry, target, stop
- Footer with market context (VIX, regime)
"""

import requests
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from strat.signal_automation.alerters.base import BaseAlerter
from strat.signal_automation.signal_store import StoredSignal

logger = logging.getLogger(__name__)


# Discord embed color codes
COLORS = {
    'CALL': 0x00FF00,    # Green for bullish
    'PUT': 0xFF0000,     # Red for bearish
    'INFO': 0x0099FF,    # Blue for info
    'WARNING': 0xFFAA00, # Orange for warnings
    'ERROR': 0xFF0000,   # Red for errors
}


class DiscordAlerter(BaseAlerter):
    """
    Discord webhook alerter with rich embed formatting.

    Features:
    - Rich embeds with color-coded signals
    - Retry logic with exponential backoff
    - Rate limit handling (Discord 30 req/min)
    - Throttling to prevent spam
    - Batch alerts for scan summaries

    Usage:
        alerter = DiscordAlerter(webhook_url)
        if alerter.test_connection():
            alerter.send_alert(signal)
    """

    # Discord rate limit: 30 requests per 60 seconds
    RATE_LIMIT_WINDOW = 60
    RATE_LIMIT_MAX = 25  # Stay under limit

    def __init__(
        self,
        webhook_url: str,
        username: str = 'STRAT Signal Bot',
        avatar_url: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Discord alerter.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username to display
            avatar_url: Optional avatar URL for bot
            retry_attempts: Number of retry attempts on failure
            retry_delay: Base delay between retries (exponential backoff)
        """
        super().__init__('discord')

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

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if OK to send, False if rate limited
        """
        now = time.time()

        # Remove requests older than window
        self._request_times = [
            t for t in self._request_times
            if now - t < self.RATE_LIMIT_WINDOW
        ]

        if len(self._request_times) >= self.RATE_LIMIT_MAX:
            logger.warning(
                f"Discord rate limit reached ({len(self._request_times)} requests in window)"
            )
            return False

        return True

    def _record_request(self) -> None:
        """Record a request for rate limiting."""
        self._request_times.append(time.time())

    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Send payload to Discord webhook with retry logic.

        Args:
            payload: Discord webhook payload

        Returns:
            True if sent successfully
        """
        if not self._check_rate_limit():
            # Wait for rate limit window to clear
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
                    # Success (no content)
                    return True

                if response.status_code == 429:
                    # Rate limited by Discord
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

    def _create_signal_embed(self, signal: StoredSignal) -> Dict[str, Any]:
        """
        Create Discord embed for a signal.

        Args:
            signal: Signal to format

        Returns:
            Discord embed dictionary
        """
        # Determine color based on direction
        color = COLORS.get(signal.direction, COLORS['INFO'])

        # Direction emoji (ASCII-safe)
        direction_indicator = "[CALL]" if signal.direction == 'CALL' else "[PUT]"

        # Build title
        title = f"{direction_indicator} {signal.symbol} {signal.pattern_type} | {signal.timeframe}"

        # Build description
        description = f"**{signal.pattern_type}** pattern detected on {signal.timeframe} timeframe"

        # Price fields
        fields = [
            {
                'name': 'Entry Trigger',
                'value': f'${signal.entry_trigger:.2f}',
                'inline': True
            },
            {
                'name': 'Target',
                'value': f'${signal.target_price:.2f}',
                'inline': True
            },
            {
                'name': 'Stop',
                'value': f'${signal.stop_price:.2f}',
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
                'name': 'VIX',
                'value': f'{signal.vix:.1f}',
                'inline': True
            },
        ]

        # Footer with context
        footer_parts = []
        if signal.market_regime:
            footer_parts.append(f"Regime: {signal.market_regime}")
        footer_parts.append(
            f"Detected: {signal.detected_time.strftime('%Y-%m-%d %H:%M ET')}"
        )

        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields,
            'footer': {
                'text': ' | '.join(footer_parts)
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        return embed

    def send_alert(self, signal: StoredSignal) -> bool:
        """
        Send alert for a single signal.

        Args:
            signal: Signal to alert

        Returns:
            True if alert was sent successfully
        """
        # Check throttling
        if self.is_throttled(signal.signal_key):
            logger.debug(f"Discord alert throttled: {signal.signal_key}")
            return True  # Throttled is not a failure

        embed = self._create_signal_embed(signal)

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            self.record_alert(signal.signal_key)
            logger.info(f"Discord alert sent: {signal.signal_key}")

        return success

    def send_batch_alert(self, signals: List[StoredSignal]) -> bool:
        """
        Send batch alert with multiple signals.

        Discord supports up to 10 embeds per message.

        Args:
            signals: List of signals to alert

        Returns:
            True if all alerts sent successfully
        """
        if not signals:
            return True

        # Filter out throttled signals
        signals_to_send = [
            s for s in signals
            if not self.is_throttled(s.signal_key)
        ]

        if not signals_to_send:
            logger.debug("All signals throttled in batch")
            return True

        # Split into chunks of 10 (Discord limit)
        success = True
        for i in range(0, len(signals_to_send), 10):
            chunk = signals_to_send[i:i + 10]
            embeds = [self._create_signal_embed(s) for s in chunk]

            payload = {
                'username': self.username,
                'content': f'**{len(chunk)} STRAT Signal(s) Detected**',
                'embeds': embeds
            }

            if self.avatar_url:
                payload['avatar_url'] = self.avatar_url

            if self._send_webhook(payload):
                # Record all as alerted
                for signal in chunk:
                    self.record_alert(signal.signal_key)
            else:
                success = False

        return success

    def send_scan_summary(
        self,
        timeframe: str,
        signals_found: int,
        symbols_scanned: int,
        duration_seconds: float
    ) -> bool:
        """
        Send scan completion summary.

        Args:
            timeframe: Timeframe that was scanned
            signals_found: Number of signals found
            symbols_scanned: Number of symbols scanned
            duration_seconds: Scan duration

        Returns:
            True if sent successfully
        """
        # Color based on results
        color = COLORS['INFO'] if signals_found == 0 else COLORS['WARNING']

        embed = {
            'title': f'Scan Complete: {timeframe}',
            'description': f'Scanned {symbols_scanned} symbols in {duration_seconds:.1f}s',
            'color': color,
            'fields': [
                {
                    'name': 'Signals Found',
                    'value': str(signals_found),
                    'inline': True
                },
                {
                    'name': 'Timeframe',
                    'value': timeframe,
                    'inline': True
                },
            ],
            'timestamp': datetime.utcnow().isoformat()
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
        # Color based on status
        if 'error' in status.lower():
            color = COLORS['ERROR']
        elif 'stopped' in status.lower():
            color = COLORS['WARNING']
        else:
            color = COLORS['INFO']

        embed = {
            'title': f'Signal Daemon: {status}',
            'description': details or 'No additional details',
            'color': color,
            'timestamp': datetime.utcnow().isoformat()
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        return self._send_webhook(payload)

    def send_error_alert(self, error_type: str, error_message: str) -> bool:
        """
        Send error alert.

        Args:
            error_type: Type of error
            error_message: Error message

        Returns:
            True if sent successfully
        """
        embed = {
            'title': f'[ERROR] {error_type}',
            'description': error_message[:2000],  # Discord limit
            'color': COLORS['ERROR'],
            'timestamp': datetime.utcnow().isoformat()
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        return self._send_webhook(payload)

    def send_exit_alert(
        self,
        exit_signal,  # ExitSignal from position_monitor
        order_result: Dict[str, Any]
    ) -> bool:
        """
        Send alert when a position is closed (Session 83K-49).

        Args:
            exit_signal: ExitSignal with exit details
            order_result: Order result from Alpaca

        Returns:
            True if sent successfully
        """
        # Color based on P&L
        pnl = exit_signal.unrealized_pnl
        if pnl >= 0:
            color = COLORS['CALL']  # Green for profit
            pnl_indicator = "[PROFIT]"
        else:
            color = COLORS['PUT']   # Red for loss
            pnl_indicator = "[LOSS]"

        # Build title
        title = f"{pnl_indicator} Position Closed: {exit_signal.osi_symbol}"

        # Build description based on exit reason
        reason_descriptions = {
            'TARGET': 'Underlying reached target price',
            'STOP': 'Underlying hit stop price',
            'DTE': 'Days to expiration reached threshold',
            'MAX_LOSS': 'Maximum loss threshold reached',
            'MANUAL': 'Manually closed by user',
            'TIME': 'Maximum hold time exceeded',
        }
        reason_str = exit_signal.reason.value if hasattr(exit_signal.reason, 'value') else str(exit_signal.reason)
        description = reason_descriptions.get(
            reason_str,
            f'Exit reason: {reason_str}'
        )

        # Fields
        fields = [
            {
                'name': 'Exit Reason',
                'value': reason_str,
                'inline': True
            },
            {
                'name': 'P&L',
                'value': f'${pnl:+.2f}',
                'inline': True
            },
            {
                'name': 'DTE',
                'value': str(exit_signal.dte),
                'inline': True
            },
            {
                'name': 'Underlying Price',
                'value': f'${exit_signal.underlying_price:.2f}',
                'inline': True
            },
            {
                'name': 'Option Price',
                'value': f'${exit_signal.current_option_price:.2f}',
                'inline': True
            },
        ]

        # Add order ID if available
        if order_result and 'id' in order_result:
            fields.append({
                'name': 'Order ID',
                'value': order_result['id'][:16] + '...',
                'inline': True
            })

        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields,
            'footer': {
                'text': f"Signal: {exit_signal.signal_key}" if exit_signal.signal_key else "Position Closed"
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord exit alert sent: {exit_signal.osi_symbol}")

        return success

    def send_entry_alert(
        self,
        signal: StoredSignal,
        execution_result,  # ExecutionResult from executor
    ) -> bool:
        """
        Send clean entry alert for mobile notifications (Session 83K-77).

        Format: Entry: SPY 3-1-2U 1D Call @ $670 | Target: $690 | Stop: $665

        Args:
            signal: The signal that was executed
            execution_result: Execution result with order details

        Returns:
            True if sent successfully
        """
        # Color based on direction
        color = COLORS.get(signal.direction, COLORS['INFO'])

        # Option type
        option_type = "Call" if signal.direction == "CALL" else "Put"

        # Clean, mobile-friendly message
        message = (
            f"**Entry: {signal.symbol} {signal.pattern_type} {signal.timeframe} {option_type}**\n"
            f"@ ${signal.entry_trigger:.2f} | "
            f"Target: ${signal.target_price:.2f} | "
            f"Stop: ${signal.stop_price:.2f}"
        )

        payload = {
            'username': self.username,
            'content': message,
        }

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord entry alert sent: {signal.symbol} {signal.pattern_type}")

        return success

    def send_simple_exit_alert(
        self,
        symbol: str,
        pattern_type: str,
        timeframe: str,
        direction: str,
        exit_reason: str,
        pnl: float,
    ) -> bool:
        """
        Send clean exit alert for mobile notifications (Session 83K-77).

        Format: Exit: SPY 3-1-2U 1D Call | Target Hit | P/L: +$325

        Args:
            symbol: Underlying symbol
            pattern_type: STRAT pattern (e.g., 3-1-2U)
            timeframe: Signal timeframe
            direction: CALL or PUT
            exit_reason: Reason for exit (Target Hit, Stop, DTE, etc.)
            pnl: Profit/loss in dollars

        Returns:
            True if sent successfully
        """
        # Option type
        option_type = "Call" if direction == "CALL" else "Put"

        # P/L formatting
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        # Clean, mobile-friendly message
        message = (
            f"**Exit: {symbol} {pattern_type} {timeframe} {option_type}**\n"
            f"{exit_reason} | P/L: {pnl_str}"
        )

        payload = {
            'username': self.username,
            'content': message,
        }

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord exit alert sent: {symbol} {exit_reason} P/L: {pnl_str}")

        return success

    def test_connection(self) -> bool:
        """
        Test Discord webhook connection.

        Sends a test message to verify webhook is working.

        Returns:
            True if connection test passed
        """
        payload = {
            'username': self.username,
            'content': 'ATLAS Signal Bot connected successfully!',
        }

        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url

        success = self._send_webhook(payload)

        if success:
            logger.info("Discord connection test passed")
        else:
            logger.error("Discord connection test failed")

        return success
