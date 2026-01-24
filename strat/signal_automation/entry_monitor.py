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
from strat.pattern_registry import is_bidirectional_pattern
from strat.signal_automation.utils import MarketHoursValidator

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

    # =========================================================================
    # "Let the Market Breathe" - Hourly Pattern Time Restrictions
    # =========================================================================
    # For hourly (1H) patterns, we must wait for bars to close before entry:
    # - 2-bar patterns (2U-2D, 2D-2U, etc.): First bar must close → 10:30 AM EST
    # - 3-bar patterns (2D-1-2U, 3-1-2D, etc.): First two bars must close → 11:30 AM EST
    # Daily/Weekly/Monthly patterns have no time restriction (larger TFs = more significance)
    hourly_2bar_earliest: time = field(default_factory=lambda: time(10, 30))
    hourly_3bar_earliest: time = field(default_factory=lambda: time(11, 30))


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
        self._market_hours_validator = MarketHoursValidator()  # EQUITY-86: Shared utility

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check: Optional[datetime] = None
        self._trigger_count = 0

    def is_market_hours(self) -> bool:
        """Check if currently within market hours (Eastern Time).

        Uses pandas_market_calendars for accurate holiday/early close handling.
        Session EQUITY-36: Fixed to properly handle NYSE holidays.
        Session EQUITY-86: Delegates to shared MarketHoursValidator utility.
        """
        return self._market_hours_validator.is_market_hours()

    def is_hourly_entry_allowed(self, signal: StoredSignal) -> bool:
        """
        Check if entry is allowed based on "let the market breathe" rules.

        For hourly (1H) patterns, we must wait for sufficient bars to close:
        - 2-bar patterns: Earliest entry at 10:30 AM EST (after first 1H bar closes)
        - 3-bar patterns: Earliest entry at 11:30 AM EST (after first two 1H bars close)

        Daily, Weekly, Monthly patterns have no time restriction because
        larger timeframes carry more significance.

        Args:
            signal: The signal to check

        Returns:
            True if entry is allowed, False if too early
        """
        # Only apply time restriction to hourly patterns
        if signal.timeframe != '1H':
            return True

        import pytz
        et = pytz.timezone('America/New_York')
        current_time = datetime.now(et).time()

        # Determine if this is a 2-bar or 3-bar pattern by counting components
        # 3-bar patterns have 3 components: X-Y-Z (e.g., 3-2D-2U, 2D-1-2U, 2U-1-?)
        # 2-bar patterns have 2 components: X-Y (e.g., 2D-2U, 3-2D)
        pattern = signal.pattern_type
        pattern_parts = pattern.split('-')
        is_3bar_pattern = len(pattern_parts) >= 3

        if is_3bar_pattern:
            # 3-bar pattern: need two closed bars before entry
            earliest = self.config.hourly_3bar_earliest
            if current_time < earliest:
                logger.debug(
                    f"Hourly 3-bar pattern {signal.symbol} {pattern} blocked: "
                    f"current time {current_time.strftime('%H:%M')} < {earliest.strftime('%H:%M')}"
                )
                return False
        else:
            # 2-bar pattern: need one closed bar before entry
            earliest = self.config.hourly_2bar_earliest
            if current_time < earliest:
                logger.debug(
                    f"Hourly 2-bar pattern {signal.symbol} {pattern} blocked: "
                    f"current time {current_time.strftime('%H:%M')} < {earliest.strftime('%H:%M')}"
                )
                return False

        return True

    def get_pending_signals(self) -> List[StoredSignal]:
        """Get all signals eligible for trigger monitoring."""
        # Get signals in ALERTED status (ready for trigger check)
        signals = [
            s for s in self.signal_store._signals.values()
            if s.status in (SignalStatus.DETECTED.value, SignalStatus.ALERTED.value)
        ]

        # Sort by priority: TFC rank first (higher = better), timeframe second (higher = better)
        # Session EQUITY-57: Hybrid priority - execute high-TFC signals before low-TFC
        signals.sort(key=lambda s: (s.priority_rank, s.priority), reverse=True)

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

            # "Let the Market Breathe" - Skip hourly patterns if too early in session
            if not self.is_hourly_entry_allowed(signal):
                continue

            current_price = prices.get(signal.symbol)
            if current_price is None:
                continue

            # Session EQUITY-41: Respect pattern bidirectionality from registry
            # BIDIRECTIONAL patterns (3-?, 3-1-?, X-1-?): Check both directions
            # UNIDIRECTIONAL patterns (3-2D-?, 3-2U-?, X-2D-?, X-2U-?): Only declared direction
            is_triggered = False
            trigger_level = 0.0
            actual_direction = signal.direction

            # Get setup bar levels
            setup_high = signal.setup_bar_high if signal.setup_bar_high > 0 else 0
            setup_low = signal.setup_bar_low if signal.setup_bar_low > 0 else 0

            # Session EQUITY-41: Use stored is_bidirectional flag, fall back to registry
            # Prefer the flag stored on the signal (set by scanner using pattern_registry)
            is_bidirectional = getattr(signal, 'is_bidirectional', None)
            if is_bidirectional is None:
                # Fallback: check registry directly
                is_bidirectional = is_bidirectional_pattern(signal.pattern_type)

            if is_bidirectional:
                # BIDIRECTIONAL: Check both directions (X-1-?, 3-?, etc.)
                # "Where is the next 2?" - whichever direction breaks first
                broke_up = setup_high > 0 and current_price > setup_high
                broke_down = setup_low > 0 and current_price < setup_low

                if broke_up and not broke_down:
                    trigger_level = setup_high
                    actual_direction = 'CALL'
                    is_triggered = True
                elif broke_down and not broke_up:
                    trigger_level = setup_low
                    actual_direction = 'PUT'
                    is_triggered = True
                elif broke_up and broke_down:
                    # Both bounds broken -> outside bar (type 3) - unusual
                    logger.warning(
                        f"OUTSIDE BAR: {signal.symbol} {signal.pattern_type} broke both bounds "
                        f"(high: ${setup_high:.2f}, low: ${setup_low:.2f}, price: ${current_price:.2f})"
                    )
                    continue
            else:
                # UNIDIRECTIONAL: Only check declared direction (3-2D-?, 3-2U-?, etc.)
                # Opposite break INVALIDATES the setup, does NOT reverse it
                if signal.direction == 'CALL':
                    # Only trigger on break UP
                    if setup_high > 0 and current_price > setup_high:
                        trigger_level = setup_high
                        actual_direction = 'CALL'
                        is_triggered = True
                    elif setup_low > 0 and current_price < setup_low:
                        # Opposite break - invalidate this setup
                        # Session EQUITY-48: Include signal_key for lifecycle tracing
                        logger.info(
                            f"INVALIDATED: {signal.signal_key} - CALL broke DOWN "
                            f"(${current_price:.2f} < ${setup_low:.2f})"
                        )
                        # Session EQUITY-41: Use signal store's mark_expired to persist change
                        self.signal_store.mark_expired(signal.signal_key)
                        continue
                elif signal.direction == 'PUT':
                    # Only trigger on break DOWN
                    if setup_low > 0 and current_price < setup_low:
                        trigger_level = setup_low
                        actual_direction = 'PUT'
                        is_triggered = True
                    elif setup_high > 0 and current_price > setup_high:
                        # Opposite break - invalidate this setup
                        # Session EQUITY-48: Include signal_key for lifecycle tracing
                        logger.info(
                            f"INVALIDATED: {signal.signal_key} - PUT broke UP "
                            f"(${current_price:.2f} > ${setup_high:.2f})"
                        )
                        # Session EQUITY-41: Use signal store's mark_expired to persist change
                        self.signal_store.mark_expired(signal.signal_key)
                        continue

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
                # Session EQUITY-48: Include signal_key for lifecycle tracing
                direction_info = actual_direction
                if actual_direction != signal.direction:
                    direction_info = f"{actual_direction} (was {signal.direction})"
                logger.info(
                    f"TRIGGER: {signal.signal_key} ({signal.symbol}) {signal.pattern_type} "
                    f"{direction_info} @ ${trigger_level:.2f}"
                )

        # Sort by priority: TFC rank first, timeframe second
        # Session EQUITY-57: Hybrid priority for triggered events
        triggered.sort(key=lambda e: (e.signal.priority_rank, e.priority), reverse=True)

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
