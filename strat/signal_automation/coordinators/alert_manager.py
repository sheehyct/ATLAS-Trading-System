"""
EQUITY-84: AlertManager - Extracted from SignalDaemon

Manages all alert delivery across multiple channels (Discord, logging).
Handles signal discovery alerts, trade entry alerts, and position exit alerts.

Responsibilities:
- Route alerts to appropriate channels (Discord vs logging)
- Respect market hours for alert delivery
- Sort signals by priority before alerting
- Track alerted status in signal store
- Handle alerter errors gracefully
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

from strat.signal_automation.signal_store import SignalStore, StoredSignal, SignalStatus
from strat.signal_automation.alerters import BaseAlerter, LoggingAlerter, DiscordAlerter
from strat.signal_automation.executor import ExecutionResult
from strat.signal_automation.position_monitor import ExitSignal

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert delivery for signal automation.

    Extracted from SignalDaemon as part of EQUITY-84 Phase 4 refactoring.
    Uses Facade pattern - daemon delegates alerting to this coordinator.

    Args:
        alerters: List of alerter instances (Discord, logging, etc.)
        signal_store: SignalStore for marking signals as alerted
        config_alerts: Alert configuration (from SignalAutomationConfig.alerts)
        is_market_hours_fn: Callable that returns True if within market hours
        on_error: Optional callback for error counting
    """

    def __init__(
        self,
        alerters: List[BaseAlerter],
        signal_store: SignalStore,
        config_alerts: Any,  # AlertConfig from config.py
        is_market_hours_fn: Callable[[], bool],
        on_error: Optional[Callable[[], None]] = None,
    ):
        self._alerters = alerters
        self._signal_store = signal_store
        self._config = config_alerts
        self._is_market_hours = is_market_hours_fn
        self._on_error = on_error or (lambda: None)

    def send_signal_alerts(self, signals: List[StoredSignal]) -> None:
        """
        Send alerts for new signals.

        Session EQUITY-34: Uses explicit config flags for Discord alert control.
        Discord only receives alerts based on alert_on_signal_detection config.
        Trade entry/exit alerts are controlled by alert_on_trade_entry/exit flags.
        Logging alerter still logs all signals.

        Args:
            signals: Signals to alert
        """
        import pytz
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)

        logger.info(
            f"AlertManager.send_signal_alerts: {len(signals)} signals at {now_et.strftime('%H:%M:%S ET')}, "
            f"is_market_hours={self._is_market_hours()}, "
            f"alert_on_signal_detection={self._config.alert_on_signal_detection}"
        )

        # Session EQUITY-40: Sort signals by priority and continuity strength
        signals = sorted(
            signals,
            key=lambda s: (
                getattr(s, 'priority', 0),
                getattr(s, 'continuity_strength', 0),
                getattr(s, 'magnitude_pct', 0),
            ),
            reverse=True,
        )

        # Session EQUITY-33: Skip ALL alerting during premarket/afterhours
        if not self._is_market_hours():
            logger.info(
                f"BLOCKED (outside market hours): {len(signals)} signals at {now_et.strftime('%H:%M:%S ET')}"
            )
            # Still mark as alerted for internal tracking
            for signal in signals:
                if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                    self._signal_store.mark_alerted(signal.signal_key)
            return

        for alerter in self._alerters:
            try:
                # Session EQUITY-34: Use explicit config flag for Discord signal detection
                if isinstance(alerter, DiscordAlerter):
                    if not self._config.alert_on_signal_detection:
                        logger.info(
                            f"BLOCKED Discord pattern alerts: alert_on_signal_detection=False, "
                            f"{len(signals)} signals"
                        )
                        # Mark as alerted without sending Discord message
                        for signal in signals:
                            if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                                self._signal_store.mark_alerted(signal.signal_key)
                        continue

                # Send alerts via alerter
                if len(signals) > 1 and hasattr(alerter, 'send_batch_alert'):
                    success = alerter.send_batch_alert(signals)
                else:
                    success = True
                    for signal in signals:
                        if not alerter.send_alert(signal):
                            success = False

                if success:
                    # Mark signals as alerted (skip if already HISTORICAL_TRIGGERED)
                    for signal in signals:
                        if signal.status != SignalStatus.HISTORICAL_TRIGGERED.value:
                            self._signal_store.mark_alerted(signal.signal_key)

            except Exception as e:
                logger.error(f"Alert error ({alerter.name}): {e}")
                self._on_error()

    def send_entry_alert(
        self,
        signal: StoredSignal,
        result: ExecutionResult,
    ) -> None:
        """
        Send alert for trade entry (execution).

        Session 83K-77: Simplified Discord alerts for entries.

        Args:
            signal: The signal that was executed
            result: Execution result with order details
        """
        for alerter in self._alerters:
            try:
                if isinstance(alerter, DiscordAlerter):
                    alerter.send_entry_alert(signal, result)
                elif isinstance(alerter, LoggingAlerter):
                    # Log entry execution if method exists
                    if hasattr(alerter, 'log_execution'):
                        alerter.log_execution(result)
                    else:
                        # Fallback: log basic info
                        logger.info(f"Entry executed: {signal.symbol} {signal.direction}")
            except Exception as e:
                logger.error(f"Entry alert error ({alerter.name}): {e}")
                self._on_error()

    def send_exit_alert(
        self,
        exit_signal: ExitSignal,
        order_result: Dict[str, Any],
        signal: Optional[StoredSignal] = None,
    ) -> None:
        """
        Send alert for position exit.

        Session 83K-77: Simplified Discord alerts for exits.

        Args:
            exit_signal: Exit signal with reason and P/L
            order_result: Order execution result
            signal: Optional original signal for context
        """
        for alerter in self._alerters:
            try:
                if isinstance(alerter, DiscordAlerter):
                    # Use simplified exit alert
                    reason_str = (
                        exit_signal.reason.value
                        if hasattr(exit_signal.reason, 'value')
                        else str(exit_signal.reason)
                    )
                    alerter.send_simple_exit_alert(
                        symbol=signal.symbol if signal else exit_signal.osi_symbol[:3],
                        pattern_type=signal.pattern_type if signal else "Unknown",
                        timeframe=signal.timeframe if signal else "Unknown",
                        direction=signal.direction if signal else "CALL",
                        exit_reason=reason_str,
                        pnl=exit_signal.unrealized_pnl,
                    )
                elif isinstance(alerter, LoggingAlerter):
                    alerter.log_position_exit(exit_signal, order_result)
            except Exception as e:
                logger.error(f"Exit alert error ({alerter.name}): {e}")
                self._on_error()

    def test_alerters(self) -> Dict[str, bool]:
        """
        Test all alerters for connectivity.

        Returns:
            Dict mapping alerter name to success status
        """
        results = {}
        for alerter in self._alerters:
            try:
                success = alerter.test_connection()
                results[alerter.name] = success
            except Exception as e:
                logger.error(f"Alerter test failed ({alerter.name}): {e}")
                results[alerter.name] = False
        return results

    @property
    def alerters(self) -> List[BaseAlerter]:
        """Return list of alerters."""
        return self._alerters
