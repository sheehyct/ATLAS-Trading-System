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

# User ID for notification mentions (test enhancement)
NOTIFY_USER_ID = "1458483340310085727"


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

    # Max setups to display in morning report embed
    _MORNING_MAX_SETUPS = 8

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

    def _get_entry_trigger_display(self, signal: StoredSignal) -> str:
        """
        Get entry trigger price for display.

        Session EQUITY-34: For SETUP signals (incomplete patterns like "3-?"),
        entry_trigger is 0.0 since the pattern is incomplete. In this case,
        display the setup bar level that needs to break:
        - CALL: setup_bar_high (bullish break level)
        - PUT: setup_bar_low (bearish break level)

        Args:
            signal: The signal to get entry trigger for

        Returns:
            Formatted price string like "$123.45"
        """
        # Check if this is a SETUP with 0.0 entry trigger
        if signal.entry_trigger == 0.0 or (
            hasattr(signal, 'signal_type') and signal.signal_type == 'SETUP'
        ):
            # Use setup bar levels based on direction
            if signal.direction == 'CALL' and hasattr(signal, 'setup_bar_high') and signal.setup_bar_high > 0:
                return f"${signal.setup_bar_high:.2f}"
            elif signal.direction == 'PUT' and hasattr(signal, 'setup_bar_low') and signal.setup_bar_low > 0:
                return f"${signal.setup_bar_low:.2f}"

        # Default: use entry_trigger
        return f"${signal.entry_trigger:.2f}"

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
        # Session EQUITY-34: Use helper for SETUP signals with 0.0 entry_trigger
        fields = [
            {
                'name': 'Entry Trigger',
                'value': self._get_entry_trigger_display(signal),
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
                'name': 'Continuity',
                'value': f"{getattr(signal, 'continuity_strength', 0)}/5",
                'inline': True
            },
            {
                'name': 'Priority',
                'value': f"{getattr(signal, 'priority', 0)}",
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

        # Session EQUITY-35: Convert detected_time to ET for display
        # The datetime may be naive (in server timezone) or UTC - convert properly
        import pytz
        et = pytz.timezone('America/New_York')
        dt = signal.detected_time
        if dt.tzinfo is None:
            # Assume naive datetime is UTC (server runs in UTC)
            dt = pytz.utc.localize(dt)
        dt_et = dt.astimezone(et)
        footer_parts.append(
            f"Detected: {dt_et.strftime('%Y-%m-%d %H:%M ET')}"
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
        # Session EQUITY-33: Added magnitude and TFC score
        # Session EQUITY-34: Fix truthiness check - 0 is valid TFC score
        # Session EQUITY-34: Use helper for SETUP signals with 0.0 entry_trigger
        tfc_display = f"{signal.tfc_score}/4" if hasattr(signal, 'tfc_score') and signal.tfc_score is not None else "N/A"
        entry_display = self._get_entry_trigger_display(signal)
        message = (
            f"<@{NOTIFY_USER_ID}> "
            f"**Entry: {signal.symbol} {signal.pattern_type} {signal.timeframe} {option_type}**\n"
            f"@ {entry_display} | "
            f"Target: ${signal.target_price:.2f} | "
            f"Stop: ${signal.stop_price:.2f}\n"
            f"Mag: {signal.magnitude_pct:.2f}% | TFC: {tfc_display}"
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
            f"<@{NOTIFY_USER_ID}> "
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

    def send_daily_audit(self, audit_data: Dict[str, Any]) -> bool:
        """
        EQUITY-52: Send daily trade audit report via webhook.

        Args:
            audit_data: Dictionary containing:
                - date: Audit date string (YYYY-MM-DD)
                - trades_today: Number of trades executed today
                - wins: Number of winning trades
                - losses: Number of losing trades
                - total_pnl: Total P&L for the day
                - profit_factor: Profit factor (gross profit / gross loss)
                - open_positions: List of open position summaries
                - anomalies: List of anomaly descriptions

        Returns:
            True if sent successfully
        """
        date = audit_data.get('date', 'Unknown')
        trades_today = audit_data.get('trades_today', 0)
        wins = audit_data.get('wins', 0)
        losses = audit_data.get('losses', 0)
        total_pnl = audit_data.get('total_pnl', 0.0)
        profit_factor = audit_data.get('profit_factor', 0.0)
        open_positions = audit_data.get('open_positions', [])
        anomalies = audit_data.get('anomalies', [])

        # Calculate win rate
        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0

        # Color based on P&L
        if total_pnl >= 0:
            color = COLORS['CALL']  # Green for profit
        else:
            color = COLORS['PUT']   # Red for loss

        # Format P&L with sign
        pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"

        # Build description
        description = f"Trades: {trades_today} | Win Rate: {win_rate:.0f}%\nP/L: {pnl_str}"
        if profit_factor > 0:
            description += f" | PF: {profit_factor:.2f}"

        # Build fields
        fields = []

        # Open positions field
        if open_positions:
            positions_text = ""
            for pos in open_positions[:5]:  # Limit to 5
                symbol = pos.get('symbol', 'N/A')
                pattern = pos.get('pattern_type', 'N/A')
                timeframe = pos.get('timeframe', 'N/A')
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('unrealized_pct', 0) * 100
                pnl_sign = '+' if pnl >= 0 else ''
                positions_text += f"- {symbol} {pattern} {timeframe}: {pnl_sign}${pnl:.0f} ({pnl_sign}{pnl_pct:.0f}%)\n"
            if len(open_positions) > 5:
                positions_text += f"... +{len(open_positions) - 5} more"
            fields.append({
                'name': f'Open Positions ({len(open_positions)})',
                'value': positions_text or 'None',
                'inline': False
            })

        # EQUITY-112: MFE/MAE Excursion Analysis field
        excursion = audit_data.get('excursion')
        if excursion and excursion.get('trades_with_excursion', 0) > 0:
            avg_eff = excursion.get('avg_exit_efficiency', 0)
            avg_mfe = excursion.get('avg_mfe', 0)
            avg_mae = excursion.get('avg_mae', 0)
            lwg = excursion.get('losers_went_green', 0)
            total_losers = excursion.get('total_losers', 0)
            n_exc = excursion.get('trades_with_excursion', 0)

            excursion_text = (
                f"Avg MFE: ${avg_mfe:+.2f} | Avg MAE: ${avg_mae:+.2f}\n"
                f"Exit Efficiency: {avg_eff:.0%}"
            )
            if total_losers > 0:
                excursion_text += f"\nLosers went green: {lwg}/{total_losers}"
            fields.append({
                'name': f'MFE/MAE Analysis ({n_exc} trades)',
                'value': excursion_text,
                'inline': False
            })

        # EQUITY-112: Capital Status field (data from daemon since EQUITY-107)
        capital = audit_data.get('capital')
        if capital:
            avail = capital.get('available_capital', 0)
            deployed = capital.get('deployed_capital', 0)
            heat = capital.get('portfolio_heat_pct', 0)
            cap_text = (
                f"Available: ${avail:,.0f} | Deployed: ${deployed:,.0f}\n"
                f"Heat: {heat:.1f}%"
            )
            fields.append({
                'name': 'Capital Status',
                'value': cap_text,
                'inline': False
            })

        # Anomalies field
        anomaly_count = len(anomalies)
        if anomalies:
            anomalies_text = "\n".join([f"- {a}" for a in anomalies[:3]])
            if len(anomalies) > 3:
                anomalies_text += f"\n... +{len(anomalies) - 3} more"
            fields.append({
                'name': f'Anomalies ({anomaly_count})',
                'value': anomalies_text,
                'inline': False
            })
        else:
            fields.append({
                'name': 'Anomalies',
                'value': 'None detected',
                'inline': False
            })

        embed = {
            'title': f'DAILY TRADE AUDIT - {date}',
            'description': description,
            'color': color,
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ATLAS Signal Automation'}
        }

        payload = {
            'username': self.username,
            'embeds': [embed]
        }

        success = self._send_webhook(payload)

        if success:
            logger.info(f"Discord daily audit sent for {date}: {trades_today} trades, P/L: {pnl_str}")

        return success

    def send_morning_report(self, report_data: Dict[str, Any]) -> bool:
        """
        EQUITY-112: Send pre-market morning report via webhook.

        Args:
            report_data: Dict with keys: date, setups, gaps, open_positions,
                yesterday, capital, pipeline_stats, duration_seconds.

        Returns:
            True if sent successfully.
        """
        report_date = report_data.get('date', 'Unknown')
        setups = report_data.get('setups', [])
        gaps = report_data.get('gaps', [])
        positions = report_data.get('open_positions', [])
        yesterday = report_data.get('yesterday', {})
        capital = report_data.get('capital', {})
        pipeline_stats = report_data.get('pipeline_stats', {})
        duration = report_data.get('duration_seconds', 0)

        # Summary line
        n_setups = len(setups)
        n_positions = len(positions)
        yday_pnl = yesterday.get('total_pnl', 0)
        yday_str = f"+${yday_pnl:.0f}" if yday_pnl >= 0 else f"-${abs(yday_pnl):.0f}"
        description = (
            f"Pipeline: {pipeline_stats.get('final_candidates', n_setups)} candidates "
            f"| {n_positions} positions open "
            f"| Yesterday: {yday_str}"
        )

        fields = []

        # Tiered setup display (v2.0) or legacy flat display
        tier1_setups = report_data.get('tier1_setups', [])
        tier2_setups = report_data.get('tier2_setups', [])
        tier3_context = report_data.get('tier3_context', [])
        has_tiers = bool(tier1_setups or tier3_context)

        if has_tiers:
            # Tier 1: Convergence setups (detailed)
            if tier1_setups:
                fields.append(self._format_tier1_field(tier1_setups))

            # Tier 2: Standard setups (existing format)
            if tier2_setups:
                lines = []
                for c in tier2_setups[:self._MORNING_MAX_SETUPS]:
                    lines.append(self._format_setup_line(c))
                if len(tier2_setups) > self._MORNING_MAX_SETUPS:
                    lines.append(f"... +{len(tier2_setups) - self._MORNING_MAX_SETUPS} more")
                fields.append({
                    'name': f'TIER 2: STANDARD ({len(tier2_setups)})',
                    'value': '\n'.join(lines) or 'None',
                    'inline': False,
                })

            # Tier 3: Directional context (compact)
            if tier3_context:
                fields.append(self._format_tier3_field(tier3_context))
        elif setups:
            # Legacy flat display (v1.0 candidates without tier data)
            lines = []
            for c in setups[:self._MORNING_MAX_SETUPS]:
                lines.append(self._format_setup_line(c))
            if len(setups) > self._MORNING_MAX_SETUPS:
                lines.append(f"... +{len(setups) - self._MORNING_MAX_SETUPS} more")
            fields.append({
                'name': f'STRAT Setups ({n_setups})',
                'value': '\n'.join(lines) or 'None',
                'inline': False,
            })

        # Field 2: Pre-Market Gaps
        if gaps:
            gap_lines = []
            for g in gaps[:8]:
                sign = '+' if g['gap_pct'] >= 0 else ''
                gap_lines.append(
                    f"{g['symbol']}: {sign}{g['gap_pct']:.1f}% "
                    f"(${g['prev_close']:.2f} -> ${g['premarket_price']:.2f})"
                )
            fields.append({
                'name': f'Pre-Market Gaps ({len(gaps)})',
                'value': '\n'.join(gap_lines),
                'inline': False,
            })

        # Field 3: Open Positions
        if positions:
            pos_lines = []
            for pos in positions[:5]:
                symbol = pos.get('symbol', 'N/A')
                pattern = pos.get('pattern_type', 'N/A')
                tf = pos.get('timeframe', 'N/A')
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('unrealized_pct', 0) * 100
                sign = '+' if pnl >= 0 else ''
                pos_lines.append(
                    f"- {symbol} {pattern} {tf}: {sign}${pnl:.0f} ({sign}{pnl_pct:.0f}%)"
                )
            if len(positions) > 5:
                pos_lines.append(f"... +{len(positions) - 5} more")
            fields.append({
                'name': f'Open Positions ({n_positions})',
                'value': '\n'.join(pos_lines),
                'inline': False,
            })

        # Field 4: Yesterday's Performance (inline)
        if yesterday and yesterday.get('trades', 0) > 0:
            wr = yesterday.get('win_rate', 0)
            pf = yesterday.get('profit_factor', 0)
            fields.append({
                'name': 'Yesterday',
                'value': (
                    f"Trades: {yesterday['trades']} | WR: {wr:.0f}%\n"
                    f"P/L: {yday_str} | PF: {pf:.2f}"
                ),
                'inline': True,
            })

        # Field 5: Capital Status (inline)
        if capital:
            avail = capital.get('available_capital', 0)
            heat = capital.get('portfolio_heat_pct', 0)
            fields.append({
                'name': 'Capital',
                'value': f"Available: ${avail:,.0f}\nHeat: {heat:.1f}%",
                'inline': True,
            })

        embed = {
            'title': f'PRE-MARKET MORNING REPORT - {report_date}',
            'description': description,
            'color': COLORS['INFO'],
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': f'ATLAS Pre-Market | Pipeline: {duration:.0f}s'},
        }

        payload = {
            'username': self.username,
            'embeds': [embed],
        }

        success = self._send_webhook(payload)

        if success:
            logger.info(
                f"Discord morning report sent for {report_date}: "
                f"{n_setups} setups, {len(gaps)} gaps"
            )

        return success

    # ------------------------------------------------------------------
    # Morning report helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_setup_line(c: Dict[str, Any]) -> str:
        """Format a single setup candidate as a one-line string."""
        pat = c.get('pattern', {})
        lvl = c.get('levels', {})
        tfc = c.get('tfc', {})
        direction = pat.get('direction', '?')
        tag = f"[{direction}]"
        pat_type = pat.get('type', '?')
        tf = pat.get('timeframe', '?')
        tfc_str = tfc.get('alignment', '?')
        entry = lvl.get('entry_trigger', 0)
        stop = lvl.get('stop_price', 0)
        target = lvl.get('target_price', 0)
        return (
            f"{tag} {c.get('symbol', '?')} {pat_type} {tf} "
            f"| TFC: {tfc_str} "
            f"| E: ${entry:.2f} S: ${stop:.2f} T: ${target:.2f}"
        )

    @staticmethod
    def _format_tier1_field(tier1_setups: List[Dict[str, Any]]) -> Dict:
        """
        Format Tier 1 convergence setups with trigger level detail.

        Example output:
            [PUT] CRWD  3-inside (1M/1W/1D)
              Break $371.50 -> cascades 3 TFs bearish
              Prior: 2D(1M) 2D(1W) 2U(1D) | Spread: 1.8%
              Score: 87 | ATR: 3.2%
        """
        lines = []
        for c in tier1_setups:
            conv = c.get('convergence', {})
            pat = c.get('pattern', {})
            metrics = c.get('metrics', {})
            direction = pat.get('direction', '?')
            tag = f"[{direction}]"
            symbol = c.get('symbol', '?')

            inside_count = conv.get('inside_bar_count', 0)
            inside_tfs = conv.get('inside_bar_timeframes', [])
            tfs_str = '/'.join(inside_tfs) if inside_tfs else '?'

            lines.append(f"{tag} **{symbol}**  {inside_count}-inside ({tfs_str})")

            # Trigger level line
            alignment = conv.get('prior_direction_alignment', 'mixed')
            if direction == 'PUT':
                trigger = conv.get('bearish_trigger')
                cascade_dir = 'bearish'
            else:
                trigger = conv.get('bullish_trigger')
                cascade_dir = 'bullish'
            if trigger is not None:
                lines.append(
                    f"  Break ${trigger:.2f} -> cascades {inside_count} TFs {cascade_dir}"
                )

            # Spread and score
            spread = conv.get('trigger_spread_pct', 0)
            score = conv.get('convergence_score', 0)
            atr = metrics.get('atr_percent', 0)
            lines.append(f"  Spread: {spread:.1f}% | Score: {score:.0f} | ATR: {atr:.1f}%")

        return {
            'name': f'TIER 1: CONVERGENCE ({len(tier1_setups)})',
            'value': '\n'.join(lines) or 'None',
            'inline': False,
        }

    @staticmethod
    def _format_tier3_field(tier3_context: List[Dict[str, Any]]) -> Dict:
        """
        Format Tier 3 continuation context as compact directional summary.

        Example output:
            Bearish: AAPL(1W 2D-2D), NFLX(1D 2D-2D)
            Bullish: META(1W 2U-2U)
        """
        bearish = []
        bullish = []
        for c in tier3_context:
            pat = c.get('pattern', {})
            symbol = c.get('symbol', '?')
            tf = pat.get('timeframe', '?')
            pat_type = pat.get('type', '?')
            direction = pat.get('direction', '?')
            label = f"{symbol}({tf} {pat_type})"
            if direction == 'PUT':
                bearish.append(label)
            else:
                bullish.append(label)

        lines = []
        if bearish:
            lines.append(f"Bearish: {', '.join(bearish)}")
        if bullish:
            lines.append(f"Bullish: {', '.join(bullish)}")

        return {
            'name': 'DIRECTIONAL CONTEXT',
            'value': '\n'.join(lines) or 'None',
            'inline': False,
        }

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
