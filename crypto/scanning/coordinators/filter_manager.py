"""
EQUITY-94: CryptoFilterManager - Extracted from CryptoSignalDaemon

Manages signal quality filtering, deduplication, and expiry cleanup:
- Magnitude and risk:reward filtering
- Maintenance gap filtering
- Signal ID generation for deduplication
- Duplicate detection
- Expired signal cleanup

Extracted as part of Phase 6.4 coordinator extraction.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List

from crypto.scanning.models import CryptoDetectedSignal

logger = logging.getLogger(__name__)


class CryptoFilterManager:
    """
    Signal quality filtering and deduplication for crypto daemon.

    Manages the signal store (detected signals dict) and provides
    filtering, dedup, and expiry cleanup.
    """

    def __init__(
        self,
        min_magnitude_pct: float = 0.5,
        min_risk_reward: float = 1.0,
        signal_expiry_hours: int = 24,
    ):
        """
        Initialize filter manager.

        Args:
            min_magnitude_pct: Minimum setup magnitude percentage
            min_risk_reward: Minimum risk:reward ratio
            signal_expiry_hours: Hours before signals expire from store
        """
        self._min_magnitude_pct = min_magnitude_pct
        self._min_risk_reward = min_risk_reward
        self._signal_expiry_hours = signal_expiry_hours

        # Signal store
        self._detected_signals: Dict[str, CryptoDetectedSignal] = {}
        self._signals_lock = threading.Lock()

    @property
    def signals_in_store(self) -> int:
        """Number of signals currently in store."""
        with self._signals_lock:
            return len(self._detected_signals)

    def get_detected_signals(self) -> List[CryptoDetectedSignal]:
        """Get all detected signals currently in store."""
        with self._signals_lock:
            return list(self._detected_signals.values())

    def passes_filters(self, signal: CryptoDetectedSignal) -> bool:
        """
        Check if signal passes quality filters.

        Args:
            signal: Signal to check

        Returns:
            True if passes all filters
        """
        # Magnitude filter
        if signal.magnitude_pct < self._min_magnitude_pct:
            return False

        # Risk:Reward filter
        if signal.risk_reward < self._min_risk_reward:
            return False

        # Skip signals with maintenance gaps
        if signal.has_maintenance_gap:
            logger.debug(f"Skipping signal with maintenance gap: {signal.symbol}")
            return False

        return True

    def generate_signal_id(self, signal: CryptoDetectedSignal) -> str:
        """
        Generate unique ID for deduplication.

        CRYPTO-MONITOR-1 FIX: Use setup_bar_timestamp instead of detected_time.
        This ensures:
        - Same bar across scans -> same ID -> deduplicated
        - Different bars -> different IDs -> kept as separate setups
        """
        bar_ts = signal.setup_bar_timestamp
        if bar_ts is not None and hasattr(bar_ts, 'isoformat'):
            ts_str = bar_ts.isoformat()
        elif bar_ts is not None:
            ts_str = str(bar_ts)
        else:
            ts_str = signal.detected_time.isoformat()

        return (
            f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_"
            f"{signal.direction}_{ts_str}"
        )

    def is_duplicate(self, signal: CryptoDetectedSignal) -> bool:
        """Check if signal is a duplicate."""
        signal_id = self.generate_signal_id(signal)
        with self._signals_lock:
            return signal_id in self._detected_signals

    def store_signal(self, signal: CryptoDetectedSignal) -> str:
        """
        Store a signal in the signal store.

        Args:
            signal: Signal to store

        Returns:
            Signal ID
        """
        signal_id = self.generate_signal_id(signal)
        with self._signals_lock:
            self._detected_signals[signal_id] = signal
        return signal_id

    def cleanup_expired_signals(self) -> int:
        """
        Remove signals older than expiry threshold.

        Returns:
            Number of expired signals removed
        """
        now = datetime.now(timezone.utc)
        expired_ids = []

        with self._signals_lock:
            for signal_id, signal in self._detected_signals.items():
                detected_time = signal.detected_time
                if detected_time.tzinfo is None:
                    detected_time = detected_time.replace(tzinfo=timezone.utc)

                age_hours = (now - detected_time).total_seconds() / 3600
                if age_hours > self._signal_expiry_hours:
                    expired_ids.append(signal_id)

            for signal_id in expired_ids:
                del self._detected_signals[signal_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired signals")

        return len(expired_ids)
