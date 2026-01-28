"""
EQUITY-94: CryptoAlertManager - Extracted from CryptoSignalDaemon

Manages Discord alerting for the crypto signal daemon:
- Signal detection alerts
- Trigger alerts
- Trade entry alerts (with direction/stop/target overrides)
- Trade exit alerts with P&L

Extracted as part of Phase 6.4 coordinator extraction.
"""

import logging
from typing import Any, List, Optional

from crypto.scanning.entry_monitor import CryptoTriggerEvent
from crypto.scanning.models import CryptoDetectedSignal

logger = logging.getLogger(__name__)

# Optional Discord alerter import
try:
    from crypto.alerters.discord_alerter import CryptoDiscordAlerter
except Exception as _discord_import_err:
    CryptoDiscordAlerter = None  # type: ignore
    _discord_import_error_msg = str(_discord_import_err)


class CryptoAlertManager:
    """
    Discord alerting coordinator for crypto signal daemon.

    Wraps CryptoDiscordAlerter with configuration-based routing
    and error handling.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        alert_on_signal_detection: bool = False,
        alert_on_trigger: bool = False,
        alert_on_trade_entry: bool = True,
        alert_on_trade_exit: bool = True,
    ):
        """
        Initialize alert manager.

        Args:
            webhook_url: Discord webhook URL (None = alerting disabled)
            alert_on_signal_detection: Alert on pattern detection
            alert_on_trigger: Alert when setup price hit
            alert_on_trade_entry: Alert when trade executes
            alert_on_trade_exit: Alert when trade closes
        """
        self._alert_on_signal_detection = alert_on_signal_detection
        self._alert_on_trigger = alert_on_trigger
        self._alert_on_trade_entry = alert_on_trade_entry
        self._alert_on_trade_exit = alert_on_trade_exit
        self._alerter: Optional[Any] = None

        if webhook_url:
            self._init_alerter(webhook_url)

    def _init_alerter(self, webhook_url: str) -> None:
        """Initialize Discord alerter."""
        if CryptoDiscordAlerter is None:
            error_msg = globals().get('_discord_import_error_msg', 'unknown error')
            logger.warning(f"Discord alerter not available (import failed: {error_msg})")
            return

        try:
            self._alerter = CryptoDiscordAlerter(
                webhook_url=webhook_url,
                username='ATLAS Crypto Bot',
            )
            logger.info("Discord alerter initialized")
            self._alerter.test_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Discord alerter: {e}")

    @property
    def alerter(self) -> Optional[Any]:
        """Get underlying alerter (for StatArb executor access)."""
        return self._alerter

    @property
    def is_configured(self) -> bool:
        """Check if alerter is configured and available."""
        return self._alerter is not None

    def send_signal_alert(self, signal: CryptoDetectedSignal) -> None:
        """Send alert for newly detected signal."""
        if not self._alerter or not self._alert_on_signal_detection:
            return
        try:
            self._alerter.send_signal_alert(signal)
        except Exception as e:
            logger.error(f"Discord signal alert error: {e}")

    def send_trigger_alert(self, event: CryptoTriggerEvent) -> None:
        """Send alert when trigger fires."""
        if not self._alerter or not self._alert_on_trigger:
            return
        try:
            self._alerter.send_trigger_alert(event)
        except Exception as e:
            logger.error(f"Discord trigger alert error: {e}")

    def send_entry_alert(
        self,
        signal: CryptoDetectedSignal,
        entry_price: float,
        quantity: float,
        leverage: float,
        pattern_override: Optional[str] = None,
        direction_override: Optional[str] = None,
        stop_override: Optional[float] = None,
        target_override: Optional[float] = None,
    ) -> None:
        """Send alert when trade executes."""
        if not self._alerter or not self._alert_on_trade_entry:
            return
        try:
            kwargs: dict = {
                'signal': signal,
                'entry_price': entry_price,
                'quantity': quantity,
                'leverage': leverage,
            }
            if pattern_override:
                kwargs['pattern_override'] = pattern_override
            if direction_override:
                kwargs['direction_override'] = direction_override
            if stop_override:
                kwargs['stop_override'] = stop_override
            if target_override:
                kwargs['target_override'] = target_override
            self._alerter.send_entry_alert(**kwargs)
        except Exception as alert_err:
            logger.warning(f"Failed to send entry alert: {alert_err}")

    def send_exit_alert(self, trade: Any) -> None:
        """
        Send alert when trade closes with P&L.

        Args:
            trade: Closed trade object with entry/exit prices
        """
        if not self._alerter or not self._alert_on_trade_exit:
            return
        try:
            # Calculate P&L
            if trade.side == "BUY":  # LONG
                pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                pnl_pct = ((trade.exit_price / trade.entry_price) - 1) * 100
            else:  # SHORT
                pnl = (trade.entry_price - trade.exit_price) * trade.quantity
                pnl_pct = ((trade.entry_price / trade.exit_price) - 1) * 100

            self._alerter.send_exit_alert(
                symbol=trade.symbol,
                direction="LONG" if trade.side == "BUY" else "SHORT",
                exit_reason=trade.exit_reason or "Unknown",
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                pattern_type=trade.pattern_type,
                timeframe=trade.timeframe,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
            )
        except Exception as alert_err:
            logger.warning(f"Failed to send exit alert: {alert_err}")

    def send_exit_alerts(self, trades: List[Any]) -> None:
        """Send exit alerts for multiple closed trades."""
        for trade in trades:
            self.send_exit_alert(trade)
