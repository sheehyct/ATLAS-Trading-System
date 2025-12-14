"""
Crypto STRAT Entry Trigger Monitor - Session CRYPTO-3

Real-time monitoring for signal entry triggers on crypto perpetual futures.
Monitors price vs setup_bar_high/low and executes when breached.

Key Differences from Equities:
- 24/7 operation (no market hours filter)
- Friday maintenance window handling (5-6 PM ET)
- Uses CoinbaseClient for price fetching
- LONG triggers when price > setup_bar_high
- SHORT triggers when price < setup_bar_low

Usage:
    from crypto.scanning.entry_monitor import CryptoEntryMonitor

    monitor = CryptoEntryMonitor()
    monitor.start()  # Start background monitoring
    ...
    monitor.stop()   # Stop monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
import logging
import threading
import time as time_module

from crypto import config
from crypto.exchange.coinbase_client import CoinbaseClient
from crypto.scanning.models import CryptoDetectedSignal

logger = logging.getLogger(__name__)


# Timeframe priority (higher = more important)
TIMEFRAME_PRIORITY: Dict[str, int] = {
    "1w": 5,
    "1d": 4,
    "4h": 3,
    "1h": 2,
    "15m": 1,
}


@dataclass
class CryptoTriggerEvent:
    """Represents a triggered crypto entry signal."""

    signal: CryptoDetectedSignal
    trigger_price: float
    current_price: float
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def priority(self) -> int:
        """Get priority from signal timeframe."""
        return TIMEFRAME_PRIORITY.get(self.signal.timeframe, 0)

    @property
    def symbol(self) -> str:
        """Get symbol from underlying signal."""
        return self.signal.symbol

    @property
    def direction(self) -> str:
        """Get direction from underlying signal."""
        return self.signal.direction


@dataclass
class CryptoEntryMonitorConfig:
    """Configuration for crypto entry trigger monitoring."""

    # Polling interval in seconds (default from config)
    poll_interval: int = config.ENTRY_MONITOR_POLL_SECONDS

    # Enable maintenance window pause
    maintenance_window_enabled: bool = config.MAINTENANCE_WINDOW_ENABLED

    # Max signals to check per poll (0 = unlimited)
    max_signals_per_poll: int = 0

    # Callback when trigger fires
    on_trigger: Optional[Callable[[CryptoTriggerEvent], None]] = None

    # Callback on each poll cycle (for position monitoring) - Session CRYPTO-5
    on_poll: Optional[Callable[[], None]] = None

    # Signal expiry in hours
    signal_expiry_hours: int = config.SIGNAL_EXPIRY_HOURS


class CryptoEntryMonitor:
    """
    Monitors crypto prices and detects entry trigger breaches.

    Handles 24/7 crypto markets with Friday maintenance window.

    Usage:
        monitor = CryptoEntryMonitor()
        monitor.add_signal(signal)  # Add signals to monitor
        monitor.start()  # Start background monitoring
        ...
        triggered = monitor.check_triggers()  # Or manual check
        ...
        monitor.stop()   # Stop monitoring
    """

    def __init__(
        self,
        client: Optional[CoinbaseClient] = None,
        config: Optional[CryptoEntryMonitorConfig] = None,
    ):
        """
        Initialize crypto entry monitor.

        Args:
            client: CoinbaseClient instance (creates new if None)
            config: Monitor configuration
        """
        self.client = client or CoinbaseClient(simulation_mode=True)
        self.config = config or CryptoEntryMonitorConfig()

        # Pending signals to monitor
        self._signals: Dict[str, CryptoDetectedSignal] = {}
        self._signals_lock = threading.Lock()

        # Monitor state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check: Optional[datetime] = None
        self._trigger_count = 0

    # =========================================================================
    # MAINTENANCE WINDOW
    # =========================================================================

    def is_maintenance_window(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if within Coinbase maintenance window.

        Maintenance: Friday 5-6 PM ET (22:00-23:00 UTC)

        Args:
            dt: Datetime to check (defaults to now UTC)

        Returns:
            True if within maintenance window
        """
        if not self.config.maintenance_window_enabled:
            return False

        if dt is None:
            dt = datetime.now(timezone.utc)

        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Check if Friday (weekday 4)
        if dt.weekday() != config.MAINTENANCE_DAY:
            return False

        # Check if within maintenance hours
        hour = dt.hour
        return config.MAINTENANCE_START_HOUR_UTC <= hour < config.MAINTENANCE_END_HOUR_UTC

    # =========================================================================
    # SIGNAL MANAGEMENT
    # =========================================================================

    def _generate_signal_id(self, signal: CryptoDetectedSignal) -> str:
        """Generate unique ID for a signal."""
        return f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{signal.direction}_{signal.detected_time.isoformat()}"

    def add_signal(self, signal: CryptoDetectedSignal) -> bool:
        """
        Add a signal to monitor for trigger.

        Only SETUP signals are monitored (COMPLETED already triggered).

        Args:
            signal: CryptoDetectedSignal to monitor

        Returns:
            True if added, False if not (wrong type or duplicate)
        """
        # Only monitor SETUP signals
        if signal.signal_type != "SETUP":
            logger.debug(
                f"Skipping non-SETUP signal: {signal.symbol} {signal.pattern_type} "
                f"(type={signal.signal_type})"
            )
            return False

        signal_id = self._generate_signal_id(signal)

        with self._signals_lock:
            if signal_id in self._signals:
                return False  # Duplicate
            self._signals[signal_id] = signal

        logger.info(
            f"Added signal to monitor: {signal.symbol} {signal.pattern_type} "
            f"{signal.direction} ({signal.timeframe})"
        )
        return True

    def add_signals(self, signals: List[CryptoDetectedSignal]) -> int:
        """
        Add multiple signals to monitor.

        Args:
            signals: List of signals to add

        Returns:
            Number of signals added
        """
        added = 0
        for signal in signals:
            if self.add_signal(signal):
                added += 1
        return added

    def remove_signal(self, signal_id: str) -> bool:
        """
        Remove a signal from monitoring.

        Args:
            signal_id: Signal ID to remove

        Returns:
            True if removed, False if not found
        """
        with self._signals_lock:
            if signal_id in self._signals:
                del self._signals[signal_id]
                return True
        return False

    def clear_signals(self) -> int:
        """
        Clear all monitored signals.

        Returns:
            Number of signals cleared
        """
        with self._signals_lock:
            count = len(self._signals)
            self._signals.clear()
        return count

    def get_pending_signals(self) -> List[CryptoDetectedSignal]:
        """
        Get all signals currently being monitored.

        Returns:
            List of signals sorted by priority (highest first)
        """
        with self._signals_lock:
            signals = list(self._signals.values())

        # Sort by timeframe priority (highest first)
        signals.sort(
            key=lambda s: TIMEFRAME_PRIORITY.get(s.timeframe, 0), reverse=True
        )

        # Apply limit if configured
        if self.config.max_signals_per_poll > 0:
            signals = signals[: self.config.max_signals_per_poll]

        return signals

    def remove_expired_signals(self) -> int:
        """
        Remove signals older than expiry threshold.

        Returns:
            Number of signals removed
        """
        now = datetime.now(timezone.utc)
        expired_ids = []

        with self._signals_lock:
            for signal_id, signal in self._signals.items():
                # Ensure timezone-aware comparison
                detected_time = signal.detected_time
                if detected_time.tzinfo is None:
                    detected_time = detected_time.replace(tzinfo=timezone.utc)

                age_hours = (now - detected_time).total_seconds() / 3600
                if age_hours > self.config.signal_expiry_hours:
                    expired_ids.append(signal_id)

            for signal_id in expired_ids:
                del self._signals[signal_id]

        if expired_ids:
            logger.info(f"Removed {len(expired_ids)} expired signals")

        return len(expired_ids)

    # =========================================================================
    # PRICE FETCHING
    # =========================================================================

    def _fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for symbols.

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dict mapping symbol to current price
        """
        prices = {}

        for symbol in symbols:
            try:
                price = self.client.get_current_price(symbol)
                if price is not None and price > 0:
                    prices[symbol] = price
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")

        return prices

    # =========================================================================
    # TRIGGER DETECTION
    # =========================================================================

    def check_triggers(self) -> List[CryptoTriggerEvent]:
        """
        Check all pending signals for entry trigger breaches.

        Per STRAT methodology (from strat-methodology skill):
        - LONG: price > setup_bar_high (bullish break)
        - SHORT: price < setup_bar_low (bearish break)

        Entry is LIVE when current bar breaks inside bar bound.

        Returns:
            List of CryptoTriggerEvent for signals that triggered
        """
        pending = self.get_pending_signals()
        if not pending:
            return []

        # Get unique symbols
        symbols = list(set(s.symbol for s in pending))

        # Fetch current prices
        prices = self._fetch_prices(symbols)
        if not prices:
            logger.warning("No prices fetched - skipping trigger check")
            return []

        triggered: List[CryptoTriggerEvent] = []
        triggered_ids: List[str] = []

        for signal in pending:
            current_price = prices.get(signal.symbol)
            if current_price is None:
                continue

            # Determine trigger level based on direction
            # Per STRAT: X-1-2 patterns trigger when price breaks inside bar
            is_triggered = False
            trigger_level = 0.0

            if signal.direction == "LONG":
                # Use setup_bar_high for bullish break
                trigger_level = signal.setup_bar_high
                if trigger_level > 0:
                    # LONG triggers when price > setup_bar_high (strict break above)
                    is_triggered = current_price > trigger_level
            else:  # SHORT
                # Use setup_bar_low for bearish break
                trigger_level = signal.setup_bar_low
                if trigger_level > 0:
                    # SHORT triggers when price < setup_bar_low (strict break below)
                    is_triggered = current_price < trigger_level

            if is_triggered:
                event = CryptoTriggerEvent(
                    signal=signal,
                    trigger_price=trigger_level,
                    current_price=current_price,
                )
                triggered.append(event)
                triggered_ids.append(self._generate_signal_id(signal))

                logger.info(
                    f"TRIGGER: {signal.symbol} {signal.pattern_type} {signal.direction} "
                    f"@ ${trigger_level:,.2f} (current: ${current_price:,.2f}, "
                    f"timeframe: {signal.timeframe})"
                )

        # Remove triggered signals from monitoring
        with self._signals_lock:
            for signal_id in triggered_ids:
                if signal_id in self._signals:
                    del self._signals[signal_id]

        # Sort by priority (highest first)
        triggered.sort(key=lambda e: e.priority, reverse=True)

        self._last_check = datetime.now(timezone.utc)
        self._trigger_count += len(triggered)

        return triggered

    # =========================================================================
    # BACKGROUND MONITORING
    # =========================================================================

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("Crypto entry monitor started (24/7 mode)")

        while self._running:
            try:
                # Check maintenance window
                if self.is_maintenance_window():
                    logger.debug("Maintenance window active - skipping check")
                    time_module.sleep(self.config.poll_interval)
                    continue

                # Remove expired signals
                self.remove_expired_signals()

                # Check for triggers
                triggered = self.check_triggers()

                # Fire callbacks for triggered signals
                if triggered and self.config.on_trigger:
                    for event in triggered:
                        try:
                            self.config.on_trigger(event)
                        except Exception as e:
                            logger.error(f"Trigger callback error: {e}")

                # Fire on_poll callback (for position monitoring) - Session CRYPTO-5
                if self.config.on_poll:
                    try:
                        self.config.on_poll()
                    except Exception as e:
                        logger.error(f"Poll callback error: {e}")

                # Sleep until next poll
                time_module.sleep(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time_module.sleep(self.config.poll_interval)

        logger.info("Crypto entry monitor stopped")

    def start(self):
        """Start background monitoring."""
        if self._running:
            logger.warning("Crypto entry monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Crypto entry monitor started (poll_interval={self.config.poll_interval}s, "
            f"maintenance_window={self.config.maintenance_window_enabled})"
        )

    def stop(self):
        """Stop background monitoring."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Crypto entry monitor stopped")

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    # =========================================================================
    # STATUS AND STATS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get monitor statistics."""
        return {
            "running": self._running,
            "last_check": (
                self._last_check.isoformat() if self._last_check else None
            ),
            "trigger_count": self._trigger_count,
            "pending_signals": len(self.get_pending_signals()),
            "is_maintenance_window": self.is_maintenance_window(),
            "poll_interval": self.config.poll_interval,
        }

    def print_status(self) -> None:
        """Print current monitor status."""
        stats = self.get_stats()
        pending = self.get_pending_signals()

        print("\n" + "=" * 60)
        print("CRYPTO ENTRY MONITOR STATUS")
        print("=" * 60)
        print(f"Running: {stats['running']}")
        print(f"Last Check: {stats['last_check'] or 'Never'}")
        print(f"Total Triggers: {stats['trigger_count']}")
        print(f"Maintenance Window: {stats['is_maintenance_window']}")
        print(f"Poll Interval: {stats['poll_interval']}s")
        print(f"\nPending Signals: {len(pending)}")

        if pending:
            print("-" * 60)
            for i, s in enumerate(pending, 1):
                print(
                    f"  [{i}] {s.symbol} {s.pattern_type} {s.direction} "
                    f"({s.timeframe})"
                )
                if s.direction == "LONG":
                    print(f"      Trigger: > ${s.setup_bar_high:,.2f}")
                else:
                    print(f"      Trigger: < ${s.setup_bar_low:,.2f}")

        print("=" * 60)
