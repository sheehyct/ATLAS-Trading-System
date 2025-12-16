"""
STRAT Entry Trigger Monitor - Session 83K-66

Real-time monitoring for signal entry triggers across all timeframes.
Monitors price vs entry_trigger and executes when breached.

Key Design:
- Polls prices every 1 minute during market hours (configurable)
- Checks ALL pending signals regardless of timeframe
- Prioritizes execution by timeframe (1M > 1W > 1D > 1H)
- Respects max_concurrent_positions limit
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Callable
import logging
import threading
import time as time_module

from strat.signal_automation.signal_store import (
    SignalStore,
    StoredSignal,
    SignalStatus,
    SignalType,
    TIMEFRAME_PRIORITY,
)

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvent:
    """Represents a triggered entry signal."""
    signal: StoredSignal
    trigger_price: float
    current_price: float
    triggered_at: datetime = field(default_factory=datetime.now)

    @property
    def priority(self) -> int:
        """Get priority from underlying signal."""
        return self.signal.priority


@dataclass
class EntryMonitorConfig:
    """Configuration for entry trigger monitoring."""
    # Polling interval in seconds
    poll_interval: int = 60  # 1 minute default

    # Market hours (Eastern Time)
    market_open: time = field(default_factory=lambda: time(9, 30))
    market_close: time = field(default_factory=lambda: time(16, 0))

    # Only monitor during market hours
    market_hours_only: bool = True

    # Max signals to check per poll (0 = unlimited)
    max_signals_per_poll: int = 0

    # Callback when trigger fires
    on_trigger: Optional[Callable[[TriggerEvent], None]] = None


class EntryMonitor:
    """
    Monitors prices and detects entry trigger breaches.

    Usage:
        monitor = EntryMonitor(signal_store, price_fetcher, config)
        monitor.start()  # Start background monitoring
        ...
        monitor.stop()   # Stop monitoring

    Or for single checks:
        triggered = monitor.check_triggers()
    """

    def __init__(
        self,
        signal_store: SignalStore,
        price_fetcher: Callable[[List[str]], Dict[str, float]],
        config: Optional[EntryMonitorConfig] = None,
    ):
        """
        Initialize entry monitor.

        Args:
            signal_store: Signal store with pending signals
            price_fetcher: Function that takes list of symbols, returns {symbol: price}
            config: Monitor configuration
        """
        self.signal_store = signal_store
        self.price_fetcher = price_fetcher
        self.config = config or EntryMonitorConfig()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check: Optional[datetime] = None
        self._trigger_count = 0

    def is_market_hours(self) -> bool:
        """Check if currently within market hours (Eastern Time)."""
        now = datetime.now()
        current_time = now.time()

        # Check day of week (0=Monday, 6=Sunday)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        return self.config.market_open <= current_time <= self.config.market_close

    def get_pending_signals(self) -> List[StoredSignal]:
        """Get all signals eligible for trigger monitoring."""
        # Get signals in ALERTED status (ready for trigger check)
        signals = [
            s for s in self.signal_store._signals.values()
            if s.status in (SignalStatus.DETECTED.value, SignalStatus.ALERTED.value)
        ]

        # Sort by priority (highest first)
        signals.sort(key=lambda s: s.priority, reverse=True)

        # Apply limit if configured
        if self.config.max_signals_per_poll > 0:
            signals = signals[:self.config.max_signals_per_poll]

        return signals

    def check_triggers(self) -> List[TriggerEvent]:
        """
        Check all pending signals for entry trigger breaches.

        Session 83K-68: Updated to handle setup-based signals.
        - SETUP signals: Monitor for setup bar break (live trading)
        - COMPLETED signals: Skip monitoring (already triggered historically)

        Returns:
            List of TriggerEvent for signals that triggered, sorted by priority
        """
        pending = self.get_pending_signals()
        if not pending:
            return []

        # Get unique symbols
        symbols = list(set(s.symbol for s in pending))

        # Fetch current prices
        try:
            prices = self.price_fetcher(symbols)
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return []

        triggered: List[TriggerEvent] = []

        for signal in pending:
            # Session 83K-68: Skip COMPLETED signals (historical, already triggered)
            if signal.signal_type == SignalType.COMPLETED.value:
                logger.debug(
                    f"Skipping COMPLETED signal: {signal.symbol} {signal.pattern_type} "
                    "(entry already happened historically)"
                )
                continue

            current_price = prices.get(signal.symbol)
            if current_price is None:
                continue

            # Session CRYPTO-11: BIDIRECTIONAL trigger checking for SETUP signals
            # Per STRAT: "Where is the next 2?" - whichever direction breaks first
            # This fixes the bug where we only checked the signal's declared direction
            is_triggered = False
            trigger_level = 0.0
            actual_direction = signal.direction  # May change if opposite break happens

            # Get setup bar levels
            setup_high = signal.setup_bar_high if signal.setup_bar_high > 0 else 0
            setup_low = signal.setup_bar_low if signal.setup_bar_low > 0 else 0

            # Check for breaks in both directions
            broke_up = setup_high > 0 and current_price > setup_high
            broke_down = setup_low > 0 and current_price < setup_low

            if broke_up and not broke_down:
                # Inside bar broke UP -> X-1-2U -> CALL
                trigger_level = setup_high
                actual_direction = 'CALL'
                is_triggered = True
            elif broke_down and not broke_up:
                # Inside bar broke DOWN -> X-2D or X-1-2D -> PUT
                trigger_level = setup_low
                actual_direction = 'PUT'
                is_triggered = True
            elif broke_up and broke_down:
                # Both bounds broken -> outside bar (type 3) - unusual during monitoring
                logger.warning(
                    f"OUTSIDE BAR: {signal.symbol} {signal.pattern_type} broke both bounds "
                    f"(high: ${setup_high:.2f}, low: ${setup_low:.2f}, price: ${current_price:.2f})"
                )
                # Skip this signal - let next scan reclassify it
                continue
            # else: Neither broken - is_triggered remains False

            if is_triggered:
                event = TriggerEvent(
                    signal=signal,
                    trigger_price=trigger_level,
                    current_price=current_price,
                )
                # Store actual direction for executor (may differ from signal.direction)
                event._actual_direction = actual_direction
                triggered.append(event)

                # Log with direction change indicator if applicable
                direction_info = actual_direction
                if actual_direction != signal.direction:
                    direction_info = f"{actual_direction} (was {signal.direction})"
                logger.info(
                    f"TRIGGER: {signal.symbol} {signal.pattern_type} {direction_info} "
                    f"@ ${trigger_level:.2f} (current: ${current_price:.2f}, "
                    f"priority: {signal.priority}, type: {signal.signal_type})"
                )

        # Sort by priority (already sorted, but ensure after filtering)
        triggered.sort(key=lambda e: e.priority, reverse=True)

        self._last_check = datetime.now()
        self._trigger_count += len(triggered)

        return triggered

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("Entry monitor started")

        while self._running:
            try:
                # Check if we should poll
                should_check = True
                if self.config.market_hours_only and not self.is_market_hours():
                    should_check = False

                if should_check:
                    triggered = self.check_triggers()

                    # Fire callbacks for triggered signals
                    if triggered and self.config.on_trigger:
                        for event in triggered:
                            try:
                                self.config.on_trigger(event)
                            except Exception as e:
                                logger.error(f"Trigger callback error: {e}")

                # Sleep until next poll
                time_module.sleep(self.config.poll_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time_module.sleep(self.config.poll_interval)

        logger.info("Entry monitor stopped")

    def start(self):
        """Start background monitoring."""
        if self._running:
            logger.warning("Entry monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Entry monitor started (poll_interval={self.config.poll_interval}s, "
            f"market_hours_only={self.config.market_hours_only})"
        )

    def stop(self):
        """Stop background monitoring."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Entry monitor stopped")

    def get_stats(self) -> Dict:
        """Get monitor statistics."""
        return {
            'running': self._running,
            'last_check': self._last_check.isoformat() if self._last_check else None,
            'trigger_count': self._trigger_count,
            'pending_signals': len(self.get_pending_signals()),
            'is_market_hours': self.is_market_hours(),
        }
