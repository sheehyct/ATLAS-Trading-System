"""
STRAT Signal Store - Session 83K-45

Persistent storage for detected signals with deduplication.
Follows established patterns from paper_trading.py (PaperTradeLog).

Key Features:
1. Signal persistence to JSON (matching existing patterns)
2. Deduplication based on composite key
3. Signal lifecycle tracking (DETECTED -> ALERTED -> TRIGGERED -> etc.)
4. Query methods for filtering and analysis
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json
import logging
import re

from strat.paper_signal_scanner import DetectedSignal, SignalContext

logger = logging.getLogger(__name__)


class SignalStatus(str, Enum):
    """Signal lifecycle status."""
    DETECTED = 'DETECTED'       # Signal first seen (setup awaiting break)
    ALERTED = 'ALERTED'         # Alert sent to user
    TRIGGERED = 'TRIGGERED'     # Price hit entry trigger (live entry)
    HISTORICAL_TRIGGERED = 'HISTORICAL_TRIGGERED'  # Pattern already complete at detection
    EXPIRED = 'EXPIRED'         # Signal no longer valid (too old, price moved)
    CONVERTED = 'CONVERTED'     # User created paper trade from signal


class SignalType(str, Enum):
    """Signal type based on pattern completion state."""
    SETUP = 'SETUP'             # Pattern awaiting break (3-1-?, 2D-?, etc.)
    COMPLETED = 'COMPLETED'     # Pattern already formed (3-1-2U, 2D-2U, etc.)


# Timeframe priority for execution order (higher = execute first)
# Based on historical win rates: Monthly/Weekly outperform Daily/Hourly
TIMEFRAME_PRIORITY = {
    '1M': 4,  # Monthly - highest priority
    '1W': 3,  # Weekly - high priority
    '1D': 2,  # Daily - medium priority
    '1H': 1,  # Hourly - lowest priority (36% win rate vs 70%+ for others)
}


@dataclass
class StoredSignal:
    """
    A signal stored in the signal store with metadata.

    Contains the original DetectedSignal plus storage metadata
    for deduplication and lifecycle tracking.
    """
    # Unique key for deduplication
    signal_key: str

    # Original signal data
    pattern_type: str
    direction: str
    symbol: str
    timeframe: str
    detected_time: datetime
    entry_trigger: float
    stop_price: float
    target_price: float
    magnitude_pct: float
    risk_reward: float

    # Context at detection
    vix: float = 0.0
    atr_14: float = 0.0
    volume_ratio: float = 0.0
    market_regime: str = ''

    # Session EQUITY-33: TFC (Timeframe Continuity) data
    tfc_score: int = 0
    tfc_alignment: str = ''
    # Session EQUITY-40/42: TFC integration for sizing/prioritization
    continuity_strength: int = 0
    passes_flexible: bool = True
    risk_multiplier: float = 1.0
    # Session EQUITY-42: Store TFC-based priority separately from timeframe priority
    tfc_priority_rank: int = 5  # TFC-based priority (1=lowest, 5=highest)

    # Setup-specific fields (Session 83K-68: Setup-based detection)
    signal_type: str = SignalType.SETUP.value  # SETUP or COMPLETED
    setup_bar_high: float = 0.0     # Level to monitor for bullish break
    setup_bar_low: float = 0.0      # Level to monitor for bearish break
    setup_bar_timestamp: Optional[datetime] = None  # When setup bar closed

    # Session EQUITY-41: Pattern bidirectionality (from pattern_registry)
    # BIDIRECTIONAL (3-?, 3-1-?, X-1-?): Break determines direction
    # UNIDIRECTIONAL (3-2D-?, 3-2U-?, X-2D-?, X-2U-?): Only reversal direction triggers
    is_bidirectional: bool = True  # Default True for safety (check both directions)

    # Storage metadata
    status: str = SignalStatus.DETECTED.value
    first_seen_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime = field(default_factory=datetime.now)
    alerted_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    converted_at: Optional[datetime] = None
    paper_trade_id: Optional[str] = None

    # Execution tracking (for correlating closed trades with patterns)
    executed_osi_symbol: Optional[str] = None  # OCC symbol of executed option contract

    # Occurrence tracking
    occurrence_count: int = 1  # How many times this signal was detected

    @property
    def priority(self) -> int:
        """
        Get execution priority based on timeframe.
        Higher priority signals should be executed first when multiple trigger.
        """
        return TIMEFRAME_PRIORITY.get(self.timeframe, 0)

    @property
    def priority_rank(self) -> int:
        """
        Get TFC-based priority rank for execution queueing.

        Session EQUITY-42: Now returns TFC-based priority instead of timeframe priority.
        Higher TFC score = higher priority = execute first when multiple signals trigger.
        """
        return self.tfc_priority_rank

    @property
    def signal_id(self) -> str:
        """Alias for signal_key for compatibility."""
        return self.signal_key

    @classmethod
    def from_detected_signal(cls, signal: DetectedSignal) -> 'StoredSignal':
        """Create StoredSignal from DetectedSignal."""
        signal_key = cls.generate_key(signal)

        return cls(
            signal_key=signal_key,
            pattern_type=signal.pattern_type,
            direction=signal.direction,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            detected_time=signal.detected_time,
            entry_trigger=signal.entry_trigger,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            magnitude_pct=signal.magnitude_pct,
            risk_reward=signal.risk_reward,
            vix=signal.context.vix if signal.context else 0.0,
            atr_14=signal.context.atr_14 if signal.context else 0.0,
            volume_ratio=signal.context.volume_ratio if signal.context else 0.0,
            market_regime=signal.context.market_regime if signal.context else '',
            # Session EQUITY-33: TFC data
            tfc_score=signal.context.tfc_score if signal.context else 0,
            tfc_alignment=signal.context.tfc_alignment if signal.context else '',
            # Session EQUITY-40/42: TFC integration for sizing/prioritization
            continuity_strength=getattr(signal, 'continuity_strength', getattr(signal.context, 'continuity_strength', 0) if signal.context else 0),
            passes_flexible=getattr(signal, 'passes_flexible', getattr(signal.context, 'passes_flexible', True) if signal.context else True),
            # Session EQUITY-42 Bug A Fix: Extract risk_multiplier from context, not just signal
            risk_multiplier=getattr(signal, 'risk_multiplier', getattr(signal.context, 'risk_multiplier', 1.0) if signal.context else 1.0),
            # Session EQUITY-42: Store TFC-based priority rank
            tfc_priority_rank=getattr(signal, 'priority_rank', getattr(signal.context, 'priority_rank', 5) if signal.context else 5),
            # Session 83K-68: Setup-based detection fields
            signal_type=getattr(signal, 'signal_type', SignalType.COMPLETED.value),
            setup_bar_high=getattr(signal, 'setup_bar_high', 0.0),
            setup_bar_low=getattr(signal, 'setup_bar_low', 0.0),
            setup_bar_timestamp=getattr(signal, 'setup_bar_timestamp', None),
            # Session EQUITY-41: Pattern bidirectionality
            is_bidirectional=getattr(signal, 'is_bidirectional', True),
        )

    @staticmethod
    def generate_key(signal: DetectedSignal) -> str:
        """
        Generate unique key for signal deduplication.

        Key format: {symbol}_{timeframe}_{pattern_type}_{detected_timestamp}

        The timestamp is truncated to the bar level (hourly for 1H, daily for 1D, etc.)
        to ensure signals from the same bar are considered duplicates.
        """
        # Truncate timestamp based on timeframe
        dt = signal.detected_time
        if signal.timeframe == '1H':
            # Truncate to hour
            truncated = dt.replace(minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1D':
            # Truncate to day
            truncated = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1W':
            # Truncate to week start (Monday)
            truncated = dt - timedelta(days=dt.weekday())
            truncated = truncated.replace(hour=0, minute=0, second=0, microsecond=0)
        elif signal.timeframe == '1M':
            # Truncate to month start
            truncated = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            truncated = dt

        timestamp_str = truncated.strftime('%Y%m%d%H%M')

        # Session 83K-72: Include direction to allow both CALL and PUT for same pattern
        return f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{signal.direction}_{timestamp_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredSignal':
        """Create from dictionary (JSON deserialization)."""
        # Convert ISO strings back to datetime
        datetime_fields = [
            'detected_time', 'first_seen_at', 'last_seen_at',
            'alerted_at', 'triggered_at', 'expired_at', 'converted_at',
            'setup_bar_timestamp'  # Session 83K-68: Setup-based detection
        ]
        for field_name in datetime_fields:
            if data.get(field_name):
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)


class SignalStore:
    """
    Persistent storage for detected signals with deduplication.

    Follows PaperTradeLog pattern from paper_trading.py.

    Features:
    - JSON persistence to disk
    - Deduplication based on composite key
    - Signal lifecycle tracking
    - Query methods for filtering

    Usage:
        store = SignalStore('data/signals')
        signal = DetectedSignal(...)

        if not store.is_duplicate(signal):
            stored = store.add_signal(signal)
            store.mark_alerted(stored.signal_key)
    """

    def __init__(self, store_path: str = 'data/signals'):
        """
        Initialize signal store.

        Args:
            store_path: Directory for signal storage files
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.signals_file = self.store_path / 'signals.json'
        self._signals: Dict[str, StoredSignal] = {}

        # Session EQUITY-52: Reverse index for O(1) OSI symbol lookup
        # Maps executed_osi_symbol -> signal_key
        self._osi_symbol_index: Dict[str, str] = {}

        # Load existing signals
        self._load()

    def _load(self) -> None:
        """Load signals from disk."""
        if self.signals_file.exists():
            try:
                with open(self.signals_file, 'r') as f:
                    data = json.load(f)
                    for key, signal_data in data.items():
                        self._signals[key] = StoredSignal.from_dict(signal_data)

                # Session EQUITY-52: Rebuild OSI symbol reverse index
                self._rebuild_osi_index()

                logger.info(f"Loaded {len(self._signals)} signals from {self.signals_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error loading signals: {e}")
                self._signals = {}
                self._osi_symbol_index = {}
        else:
            logger.info(f"No existing signals file at {self.signals_file}")

    def _rebuild_osi_index(self) -> None:
        """
        Rebuild the OSI symbol reverse index from signals.

        Session EQUITY-52: Called after loading signals to enable O(1) lookups.
        """
        self._osi_symbol_index.clear()
        for signal_key, signal in self._signals.items():
            if signal.executed_osi_symbol:
                self._osi_symbol_index[signal.executed_osi_symbol] = signal_key
        logger.debug(f"Rebuilt OSI symbol index with {len(self._osi_symbol_index)} entries")

    def load_signals(self) -> Dict[str, StoredSignal]:
        """
        Get all signals from the store.

        Returns:
            Dictionary mapping signal_key to StoredSignal
        """
        return self._signals.copy()

    def _save(self) -> None:
        """Save signals to disk."""
        try:
            data = {key: signal.to_dict() for key, signal in self._signals.items()}
            with open(self.signals_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._signals)} signals to {self.signals_file}")
        except IOError as e:
            logger.error(f"Error saving signals: {e}")

    def is_duplicate(
        self,
        signal: DetectedSignal,
        lookback_bars: int = 3
    ) -> bool:
        """
        Check if signal is a duplicate of a recent detection.

        A signal is considered a duplicate if:
        1. Same composite key exists in store
        2. The existing signal was detected within lookback_bars

        Args:
            signal: Signal to check
            lookback_bars: Number of bars within which to consider duplicate

        Returns:
            True if signal is duplicate, False otherwise
        """
        key = StoredSignal.generate_key(signal)

        if key not in self._signals:
            # EQUITY-105: Cross-pattern dedup for bidirectional patterns
            # A resolved "3-2U" is a duplicate if "3-?" already triggered/alerted
            # for the same symbol, timeframe, and bar window.
            # Mappings: "3-2U"/"3-2D" -> "3-?", "3-1-2U"/"3-1-2D" -> "3-1-?"
            wildcard_pattern = re.sub(r'2[UD]$', '?', signal.pattern_type)
            if wildcard_pattern != signal.pattern_type:
                # Use same time-window as regular dedup to avoid permanent blocking
                if signal.timeframe == '1H':
                    lookback_delta = timedelta(hours=lookback_bars)
                elif signal.timeframe == '1D':
                    lookback_delta = timedelta(days=lookback_bars)
                elif signal.timeframe == '1W':
                    lookback_delta = timedelta(weeks=lookback_bars)
                elif signal.timeframe == '1M':
                    lookback_delta = timedelta(days=lookback_bars * 30)
                else:
                    lookback_delta = timedelta(days=lookback_bars)
                cutoff = datetime.now() - lookback_delta

                for existing in self._signals.values():
                    if (existing.symbol == signal.symbol and
                            existing.timeframe == signal.timeframe and
                            existing.pattern_type == wildcard_pattern and
                            existing.status in (
                                SignalStatus.TRIGGERED.value,
                                SignalStatus.ALERTED.value,
                                SignalStatus.CONVERTED.value,
                            ) and
                            existing.last_seen_at >= cutoff):
                        logger.info(
                            f"Cross-pattern duplicate: {signal.symbol} {signal.pattern_type} "
                            f"blocked by existing {existing.pattern_type} "
                            f"(status: {existing.status})"
                        )
                        return True
            return False

        existing = self._signals[key]

        # Calculate lookback window based on timeframe
        if signal.timeframe == '1H':
            lookback_delta = timedelta(hours=lookback_bars)
        elif signal.timeframe == '1D':
            lookback_delta = timedelta(days=lookback_bars)
        elif signal.timeframe == '1W':
            lookback_delta = timedelta(weeks=lookback_bars)
        elif signal.timeframe == '1M':
            lookback_delta = timedelta(days=lookback_bars * 30)  # Approximate
        else:
            lookback_delta = timedelta(days=lookback_bars)

        # Check if within lookback window
        now = datetime.now()
        cutoff = now - lookback_delta

        if existing.last_seen_at >= cutoff:
            # Update occurrence count and last_seen
            existing.occurrence_count += 1
            existing.last_seen_at = now
            self._save()
            logger.debug(f"Duplicate signal: {key} (occurrence #{existing.occurrence_count})")
            return True

        return False

    def add_signal(self, signal: DetectedSignal) -> StoredSignal:
        """
        Add a new signal to the store.

        Args:
            signal: DetectedSignal to add

        Returns:
            StoredSignal with metadata
        """
        stored = StoredSignal.from_detected_signal(signal)
        self._signals[stored.signal_key] = stored
        self._save()
        logger.info(f"Added signal: {stored.signal_key}")
        return stored

    def get_signal(self, signal_key: str) -> Optional[StoredSignal]:
        """Get signal by key."""
        return self._signals.get(signal_key)

    def rekey_signal(self, old_key: str, new_key: str) -> bool:
        """
        Re-key a signal after pattern/direction resolution changes its identity.

        Moves the signal from old_key to new_key in the internal store and
        updates the OSI symbol reverse index if applicable.

        Args:
            old_key: The current signal key
            new_key: The new signal key after pattern/direction resolution

        Returns:
            True if rekeyed successfully, False if old_key not found
        """
        if old_key not in self._signals:
            logger.warning(f"Cannot rekey signal - old key not found: {old_key}")
            return False

        signal = self._signals.pop(old_key)
        signal.signal_key = new_key
        self._signals[new_key] = signal

        # Update OSI symbol reverse index if this signal had an executed option
        for osi_symbol, mapped_key in list(self._osi_symbol_index.items()):
            if mapped_key == old_key:
                self._osi_symbol_index[osi_symbol] = new_key

        self._save()
        logger.info(f"Rekeyed signal: {old_key} -> {new_key}")
        return True

    def mark_alerted(self, signal_key: str) -> bool:
        """
        Mark signal as alerted.

        Args:
            signal_key: Signal key to update

        Returns:
            True if updated, False if not found
        """
        if signal_key not in self._signals:
            return False

        signal = self._signals[signal_key]
        signal.status = SignalStatus.ALERTED.value
        signal.alerted_at = datetime.now()
        self._save()
        logger.debug(f"Marked as alerted: {signal_key}")
        return True

    def mark_triggered(self, signal_key: str) -> bool:
        """Mark signal as triggered (price hit entry)."""
        if signal_key not in self._signals:
            return False

        signal = self._signals[signal_key]
        signal.status = SignalStatus.TRIGGERED.value
        signal.triggered_at = datetime.now()
        self._save()
        return True

    def mark_historical_triggered(self, signal_key: str) -> bool:
        """
        Mark signal as historical triggered (Session 83K-68).

        Used for completed patterns where entry already happened
        at the time of detection (pattern bar already closed).
        """
        if signal_key not in self._signals:
            return False

        signal = self._signals[signal_key]
        signal.status = SignalStatus.HISTORICAL_TRIGGERED.value
        signal.triggered_at = datetime.now()
        self._save()
        logger.debug(f"Marked as historical triggered: {signal_key}")
        return True

    def mark_expired(self, signal_key: str) -> bool:
        """Mark signal as expired (no longer valid)."""
        if signal_key not in self._signals:
            return False

        signal = self._signals[signal_key]
        signal.status = SignalStatus.EXPIRED.value
        signal.expired_at = datetime.now()
        self._save()
        return True

    def mark_converted(self, signal_key: str, paper_trade_id: str) -> bool:
        """Mark signal as converted to paper trade."""
        if signal_key not in self._signals:
            return False

        signal = self._signals[signal_key]
        signal.status = SignalStatus.CONVERTED.value
        signal.converted_at = datetime.now()
        signal.paper_trade_id = paper_trade_id
        self._save()
        return True

    def set_executed_osi_symbol(self, signal_key: str, osi_symbol: str) -> bool:
        """
        Store the OSI symbol of the executed option contract.

        Used to correlate closed trades back to their originating signals
        for pattern attribution in the dashboard.

        Args:
            signal_key: Unique signal identifier
            osi_symbol: OCC option symbol that was executed (e.g., SPY241220C00600000)

        Returns:
            True if updated successfully
        """
        if signal_key not in self._signals:
            logger.warning(f"Cannot set OSI symbol - signal not found: {signal_key}")
            return False

        signal = self._signals[signal_key]
        signal.executed_osi_symbol = osi_symbol

        # Session EQUITY-52: Update reverse index for O(1) lookup
        self._osi_symbol_index[osi_symbol] = signal_key

        self._save()
        logger.debug(f"Set executed_osi_symbol for {signal_key}: {osi_symbol}")
        return True

    def update_tfc(self, signal_key: str, tfc_score: int, tfc_alignment: str) -> bool:
        """
        Update TFC score and alignment for a signal after re-evaluation.

        EQUITY-102: Write TFC data back to signal store after re-evaluation
        at entry time, so trade_metadata and downstream consumers get correct
        TFC values instead of the stale tfc_score=0 from initial detection.

        Args:
            signal_key: Signal key to update
            tfc_score: Re-evaluated TFC strength score
            tfc_alignment: Re-evaluated TFC alignment label

        Returns:
            True if updated, False if signal not found
        """
        if signal_key not in self._signals:
            logger.warning(f"Cannot update TFC - signal not found: {signal_key}")
            return False

        signal = self._signals[signal_key]
        signal.tfc_score = tfc_score
        signal.tfc_alignment = tfc_alignment
        self._save()
        logger.debug(f"Updated TFC for {signal_key}: score={tfc_score}, alignment={tfc_alignment}")
        return True

    def get_signal_by_osi_symbol(self, osi_symbol: str) -> Optional[StoredSignal]:
        """
        Look up a signal by the OSI symbol of the executed option.

        Used to correlate closed trades with their originating patterns.

        Session EQUITY-52: Now uses O(1) reverse index lookup instead of O(n) scan.

        Args:
            osi_symbol: OCC option symbol to look up

        Returns:
            StoredSignal if found, None otherwise
        """
        # O(1) lookup using reverse index
        signal_key = self._osi_symbol_index.get(osi_symbol)
        if signal_key:
            return self._signals.get(signal_key)
        return None

    def get_pending_signals(self) -> List[StoredSignal]:
        """Get signals that are detected but not yet alerted."""
        return [
            s for s in self._signals.values()
            if s.status == SignalStatus.DETECTED.value
        ]

    def get_alerted_signals(self) -> List[StoredSignal]:
        """Get signals that have been alerted."""
        return [
            s for s in self._signals.values()
            if s.status == SignalStatus.ALERTED.value
        ]

    def get_setup_signals_for_monitoring(self) -> List[StoredSignal]:
        """
        Get SETUP signals that should be monitored for entry trigger breaks.
        Session 83K-68: Setup-based detection.

        Returns signals that:
        - Are SETUP type (not COMPLETED)
        - Are in DETECTED or ALERTED status
        - Have not yet triggered

        These are the signals the entry monitor should actively watch.
        """
        return [
            s for s in self._signals.values()
            if s.signal_type == SignalType.SETUP.value
            and s.status in (SignalStatus.DETECTED.value, SignalStatus.ALERTED.value)
        ]

    def get_historical_triggered_signals(self) -> List[StoredSignal]:
        """Get signals that were historical (already triggered at detection)."""
        return [
            s for s in self._signals.values()
            if s.status == SignalStatus.HISTORICAL_TRIGGERED.value
        ]

    def get_signals_by_symbol(self, symbol: str) -> List[StoredSignal]:
        """Get all signals for a symbol."""
        return [s for s in self._signals.values() if s.symbol == symbol]

    def get_signals_by_timeframe(self, timeframe: str) -> List[StoredSignal]:
        """Get all signals for a timeframe."""
        return [s for s in self._signals.values() if s.timeframe == timeframe]

    def get_signals_by_pattern(self, pattern_type: str) -> List[StoredSignal]:
        """Get all signals for a pattern type."""
        return [s for s in self._signals.values() if s.pattern_type == pattern_type]

    def get_recent_signals(
        self,
        hours: int = 24
    ) -> List[StoredSignal]:
        """Get signals from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            s for s in self._signals.values()
            if s.first_seen_at >= cutoff
        ]

    def cleanup_old_signals(self, days: int = 30) -> int:
        """
        Remove signals older than N days.

        Args:
            days: Age threshold for cleanup

        Returns:
            Number of signals removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        old_keys = [
            key for key, signal in self._signals.items()
            if signal.first_seen_at < cutoff
        ]

        for key in old_keys:
            del self._signals[key]

        if old_keys:
            self._save()
            logger.info(f"Cleaned up {len(old_keys)} old signals")

        return len(old_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get signal store statistics."""
        signals = list(self._signals.values())

        if not signals:
            return {
                'total': 0,
                'by_status': {},
                'by_symbol': {},
                'by_timeframe': {},
                'by_pattern': {},
            }

        return {
            'total': len(signals),
            'by_status': self._count_by_field(signals, 'status'),
            'by_symbol': self._count_by_field(signals, 'symbol'),
            'by_timeframe': self._count_by_field(signals, 'timeframe'),
            'by_pattern': self._count_by_field(signals, 'pattern_type'),
            'oldest': min(s.first_seen_at for s in signals).isoformat(),
            'newest': max(s.first_seen_at for s in signals).isoformat(),
        }

    @staticmethod
    def _count_by_field(signals: List[StoredSignal], field: str) -> Dict[str, int]:
        """Count signals by a field value."""
        counts: Dict[str, int] = {}
        for signal in signals:
            value = getattr(signal, field)
            counts[value] = counts.get(value, 0) + 1
        return counts

    def __len__(self) -> int:
        """Return number of signals in store."""
        return len(self._signals)

    def __contains__(self, key: str) -> bool:
        """Check if signal key exists."""
        return key in self._signals
