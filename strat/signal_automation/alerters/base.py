"""
Base Alerter Interface - Session 83K-45

Abstract base class for signal alerters.
Defines the interface that all alerters must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from strat.signal_automation.signal_store import StoredSignal


class BaseAlerter(ABC):
    """
    Abstract base class for signal alerters.

    All alerters must implement:
    - send_alert(): Send a single signal alert
    - send_batch_alert(): Send multiple signals in one alert (optional)
    - test_connection(): Verify the alerter is working

    Alerters handle:
    - Message formatting for their channel
    - Delivery with retry logic
    - Throttling to prevent spam
    """

    def __init__(self, name: str):
        """
        Initialize alerter.

        Args:
            name: Unique name for this alerter (e.g., 'discord', 'email')
        """
        self.name = name
        self._last_alert_time: Dict[str, datetime] = {}  # Key -> last alert time
        self._min_interval_seconds: int = 60

    def set_throttle_interval(self, seconds: int) -> None:
        """Set minimum interval between alerts for the same signal."""
        self._min_interval_seconds = seconds

    def is_throttled(self, signal_key: str) -> bool:
        """
        Check if a signal is currently throttled.

        Args:
            signal_key: Signal key to check

        Returns:
            True if recently alerted (throttled), False otherwise
        """
        if signal_key not in self._last_alert_time:
            return False

        last_time = self._last_alert_time[signal_key]
        elapsed = (datetime.now() - last_time).total_seconds()

        return elapsed < self._min_interval_seconds

    def record_alert(self, signal_key: str) -> None:
        """Record that an alert was sent for a signal."""
        self._last_alert_time[signal_key] = datetime.now()

    @abstractmethod
    def send_alert(self, signal: StoredSignal) -> bool:
        """
        Send alert for a single signal.

        Args:
            signal: Signal to alert

        Returns:
            True if alert was sent successfully
        """
        pass

    def send_batch_alert(self, signals: list) -> bool:
        """
        Send alert for multiple signals.

        Default implementation sends individual alerts.
        Override for channels that support batching (e.g., email digest).

        Args:
            signals: List of StoredSignal to alert

        Returns:
            True if all alerts were sent successfully
        """
        success = True
        for signal in signals:
            if not self.send_alert(signal):
                success = False
        return success

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test that the alerter is properly configured and working.

        Returns:
            True if connection test passed
        """
        pass

    def format_signal_message(self, signal: StoredSignal) -> str:
        """
        Format signal for display.

        Default plain-text format. Override for channel-specific formatting.

        Args:
            signal: Signal to format

        Returns:
            Formatted message string
        """
        return f"""
STRAT Signal Detected
Symbol: {signal.symbol} | Pattern: {signal.pattern_type} | Direction: {signal.direction}
Timeframe: {signal.timeframe}

Entry: ${signal.entry_trigger:.2f} | Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}
R:R: {signal.risk_reward:.2f}:1 | Magnitude: {signal.magnitude_pct:.2f}%
VIX: {signal.vix:.1f} | Regime: {signal.market_regime}

Detected: {signal.detected_time.strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
