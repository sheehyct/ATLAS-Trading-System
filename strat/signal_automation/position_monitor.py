"""
Position Monitor - Session 83K-49

Monitors open option positions for exit conditions:
- Target price reached (underlying hits signal target)
- Stop price reached (underlying hits signal stop)
- DTE threshold reached (close before theta decay accelerates)
- Max loss threshold reached (risk management)

Integrates with SignalDaemon for autonomous position management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from integrations.alpaca_trading_client import AlpacaTradingClient
from strat.signal_automation.executor import (
    SignalExecutor,
    ExecutionResult,
    ExecutionState,
)
from strat.signal_automation.signal_store import SignalStore, StoredSignal
from strat.signal_automation.utils import MarketHoursValidator

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """Reason for position exit."""
    TARGET_HIT = "TARGET"       # Underlying reached target price
    STOP_HIT = "STOP"           # Underlying reached stop price
    DTE_EXIT = "DTE"            # DTE below threshold
    MAX_LOSS = "MAX_LOSS"       # Unrealized loss exceeded threshold
    MANUAL = "MANUAL"           # Manually closed
    TIME_EXIT = "TIME"          # Max hold time exceeded
    EOD_EXIT = "EOD"            # Session EQUITY-35: End of day exit for 1H trades
    # Session EQUITY-36: Optimal exit strategy
    PARTIAL_EXIT = "PARTIAL"    # Partial exit at 1.0x R:R (multi-contract)
    TRAILING_STOP = "TRAIL"     # Trailing stop hit after profit
    # Session EQUITY-44: Type 3 pattern invalidation
    PATTERN_INVALIDATED = "PATTERN"  # Entry bar evolved to Type 3 (exit immediately)


@dataclass
class MonitoringConfig:
    """Configuration for position monitoring."""
    # Exit thresholds
    exit_dte: int = 3                    # Close at or below this DTE
    max_loss_pct: float = 0.50           # Max loss as % of premium (50%)
    max_profit_pct: float = 1.00         # Take profit at 100% gain

    # Session EQUITY-42: Timeframe-specific max loss thresholds
    # Monthly patterns need more room due to longer time horizon and theta decay
    # Option premium can drop significantly while underlying pattern is still valid
    max_loss_pct_by_timeframe: Optional[Dict[str, float]] = None  # Set in __post_init__

    # Monitoring intervals (seconds)
    check_interval: int = 60             # Check positions every N seconds
    underlying_fetch_interval: int = 30  # Fetch underlying prices every N seconds
    minimum_hold_seconds: int = 300      # 5 min before exit checks (Session 83K-77)

    # Exit behavior
    use_market_orders: bool = True       # Use market orders for exits
    close_partial_on_profit: bool = False  # Close half at 50% gain (future)

    # Alerting
    alert_on_exit: bool = True           # Send alerts for exits
    alert_on_profit_target: bool = True  # Alert when approaching profit target
    alert_pct_to_target: float = 0.80    # Alert when 80% to target

    # Session EQUITY-35: EOD exit for hourly trades
    # All 1H timeframe trades must exit before market close to avoid overnight gap risk
    # Session EQUITY-95: Changed from 15:59 back to 15:55 for reliable execution
    # 5-minute buffer accounts for: position check cycle (60s), order latency, fill time
    eod_exit_hour: int = 15              # Hour in ET for EOD exit
    eod_exit_minute: int = 55            # Minute in ET for EOD exit (15:55 = 5 min buffer)

    # Session EQUITY-36: Optimal exit strategy
    # 1H patterns get reduced target (1.0x R:R instead of 1.5x)
    hourly_target_rr: float = 1.0        # R:R target for 1H patterns (was 1.5x)

    # Trailing stop - activate once in profit, trail at percentage of max profit
    use_trailing_stop: bool = True       # Enable trailing stop for single-contract or remainder
    trailing_stop_activation_rr: float = 0.5  # Activate trailing stop at 0.5x R:R profit
    trailing_stop_pct: float = 0.50      # Trail 50% below high water mark

    # Session EQUITY-42: Trailing stop must be in profit to exit
    # Prevents confusing exits where trailing stop triggers but option P/L is negative
    trailing_stop_min_profit_pct: float = 0.0  # Minimum option profit % to allow trail exit (0 = breakeven)

    # EQUITY-52: ATR-based trailing stop for 3-2 patterns
    # 3-2 patterns use ATR-based stops instead of percentage-based
    # Activation: 0.75 ATR profit, Trail distance: 1.0 ATR from high water mark
    use_atr_trailing_for_32: bool = True       # Enable ATR-based trailing for 3-2 patterns
    atr_trailing_activation_multiple: float = 0.75  # Activate at 0.75 ATR profit
    atr_trailing_distance_multiple: float = 1.0     # Trail at 1.0 ATR distance from HWM

    # Partial exits - for multi-contract positions
    partial_exit_enabled: bool = True    # Enable partial exits at 1.0x R:R
    partial_exit_rr: float = 1.0         # Take partial profit at 1.0x R:R
    partial_exit_pct: float = 0.50       # Exit 50% of contracts at partial target

    def __post_init__(self):
        """Initialize timeframe-specific settings."""
        if self.max_loss_pct_by_timeframe is None:
            # Session EQUITY-42: Wider stops for longer timeframes
            # Monthly patterns shouldn't exit on 50% option loss if underlying is valid
            self.max_loss_pct_by_timeframe = {
                '1M': 0.75,  # 75% max loss for monthly (more time to recover)
                '1W': 0.65,  # 65% for weekly
                '1D': 0.50,  # 50% for daily (current default)
                '1H': 0.40,  # 40% for hourly (tighter risk control)
            }

    def get_max_loss_pct(self, timeframe: str) -> float:
        """Get timeframe-specific max loss percentage."""
        if self.max_loss_pct_by_timeframe and timeframe in self.max_loss_pct_by_timeframe:
            return self.max_loss_pct_by_timeframe[timeframe]
        return self.max_loss_pct  # Default fallback


@dataclass
class TrackedPosition:
    """A position being monitored with its original signal data."""
    # Position identifiers
    osi_symbol: str
    signal_key: str

    # Original signal data
    symbol: str                  # Underlying symbol
    direction: str               # CALL or PUT
    entry_trigger: float         # Signal entry price
    target_price: float          # Signal target price
    stop_price: float            # Signal stop price
    pattern_type: str
    timeframe: str

    # Execution data
    entry_price: float           # Actual option entry price
    contracts: int               # Number of contracts
    entry_time: datetime
    expiration: str              # Option expiration date (YYYY-MM-DD)

    # Current state (updated during monitoring)
    current_price: float = 0.0   # Current option price
    unrealized_pnl: float = 0.0  # Unrealized P&L in dollars
    unrealized_pct: float = 0.0  # Unrealized P&L as percentage
    underlying_price: float = 0.0  # Current underlying price
    dte: int = 0                 # Days to expiration

    # Status
    is_active: bool = True
    exit_reason: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    # Session EQUITY-36: Optimal exit strategy fields
    # Target levels
    target_1x: float = 0.0               # 1.0x R:R target (for partial exits and 1H patterns)
    original_target: float = 0.0         # Original target before adjustment

    # Session EQUITY-36: Track actual underlying price at execution
    # For gap-through scenarios, this differs from entry_trigger (the trigger level)
    # Used for accurate P/L and trailing stop calculations
    actual_entry_underlying: float = 0.0

    # Trailing stop state
    trailing_stop_active: bool = False   # Whether trailing stop is activated
    trailing_stop_price: float = 0.0     # Current trailing stop level (underlying price)
    high_water_mark: float = 0.0         # Best underlying price seen (for trailing calc)

    # EQUITY-52: ATR-based trailing stop for 3-2 patterns
    atr_at_detection: float = 0.0        # ATR(14) captured at signal detection time
    use_atr_trailing: bool = False       # Whether this position uses ATR trailing (3-2 only)
    atr_trail_distance: float = 0.0      # Pre-calculated 1.0 ATR trail distance
    atr_activation_level: float = 0.0    # Pre-calculated 0.75 ATR profit level

    # Partial exit state
    partial_exit_done: bool = False      # Whether partial exit has been executed
    contracts_remaining: int = 0         # Contracts remaining after partial exit

    # Session EQUITY-44: Pattern invalidation tracking
    # Per STRAT methodology: exit immediately if entry bar evolves to Type 3
    entry_bar_type: str = ""             # Entry bar type: "2U", "2D", or "3"
    entry_bar_high: float = 0.0          # Setup bar high (inside bar high for X-1-2 patterns)
    entry_bar_low: float = 0.0           # Setup bar low (inside bar low for X-1-2 patterns)

    # Session EQUITY-48: Real-time Type 3 evolution detection
    # Track intrabar extremes since entry to detect Type 3 evolution in real-time
    # (not just at bar close). Compare against setup bar bounds.
    intrabar_high: float = 0.0           # Highest underlying price since entry
    intrabar_low: float = float('inf')   # Lowest underlying price since entry

    # Tracking timestamps
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'osi_symbol': self.osi_symbol,
            'signal_key': self.signal_key,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_trigger': self.entry_trigger,
            'target_price': self.target_price,
            'stop_price': self.stop_price,
            'pattern_type': self.pattern_type,
            'timeframe': self.timeframe,
            'entry_price': self.entry_price,
            'contracts': self.contracts,
            'entry_time': self.entry_time.isoformat(),
            'expiration': self.expiration,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pct': self.unrealized_pct,
            'underlying_price': self.underlying_price,
            'dte': self.dte,
            'is_active': self.is_active,
            'exit_reason': self.exit_reason,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'realized_pnl': self.realized_pnl,
            # Session EQUITY-36: Optimal exit strategy fields
            'target_1x': self.target_1x,
            'original_target': self.original_target,
            'actual_entry_underlying': self.actual_entry_underlying,  # Gap-through fix
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            'high_water_mark': self.high_water_mark,
            # EQUITY-52: ATR-based trailing stop fields
            'atr_at_detection': self.atr_at_detection,
            'use_atr_trailing': self.use_atr_trailing,
            'atr_trail_distance': self.atr_trail_distance,
            'atr_activation_level': self.atr_activation_level,
            'partial_exit_done': self.partial_exit_done,
            'contracts_remaining': self.contracts_remaining,
            # Session EQUITY-44: Pattern invalidation tracking
            'entry_bar_type': self.entry_bar_type,
            'entry_bar_high': self.entry_bar_high,
            'entry_bar_low': self.entry_bar_low,
            # Session EQUITY-48: Real-time Type 3 detection
            'intrabar_high': self.intrabar_high,
            'intrabar_low': self.intrabar_low if self.intrabar_low != float('inf') else 0.0,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    osi_symbol: str
    signal_key: str
    reason: ExitReason
    underlying_price: float
    current_option_price: float
    unrealized_pnl: float
    dte: int
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    # Session EQUITY-36: Partial exit support
    contracts_to_close: Optional[int] = None  # None = close all, int = close specific amount


class PositionMonitor:
    """
    Monitors open option positions for exit conditions.

    Integrates with:
    - SignalExecutor for position data
    - AlpacaTradingClient for current prices and exits
    - SignalStore for original signal data

    Usage:
        monitor = PositionMonitor(config, executor, trading_client, signal_store)

        # Check all positions for exit conditions
        exit_signals = monitor.check_positions()

        # Execute exits
        for signal in exit_signals:
            result = monitor.execute_exit(signal)
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        executor: Optional[SignalExecutor] = None,
        trading_client: Optional[AlpacaTradingClient] = None,
        signal_store: Optional[SignalStore] = None,
        on_exit_callback: Optional[Callable[[ExitSignal, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize position monitor.

        Args:
            config: Monitoring configuration
            executor: SignalExecutor for execution data
            trading_client: Alpaca client for prices and orders
            signal_store: Signal store for original signal data
            on_exit_callback: Called when position is exited (for alerting)
        """
        self.config = config or MonitoringConfig()
        self.executor = executor
        self.trading_client = trading_client
        self.signal_store = signal_store
        self.on_exit_callback = on_exit_callback
        self._market_hours_validator = MarketHoursValidator()  # EQUITY-86: Shared utility

        # EQUITY-90: Exit condition evaluator (Phase 4.1) with managers (Phase 4.2/4.3)
        # Import here to avoid circular import (position_monitor <-> coordinators)
        from strat.signal_automation.coordinators.exit_evaluator import ExitConditionEvaluator
        from strat.signal_automation.coordinators.trailing_stop_manager import TrailingStopManager
        from strat.signal_automation.coordinators.partial_exit_manager import PartialExitManager

        # EQUITY-90 Phase 4.2: TrailingStopManager
        self._trailing_stop_manager = TrailingStopManager(config=self.config)

        # EQUITY-90 Phase 4.3: PartialExitManager
        self._partial_exit_manager = PartialExitManager(config=self.config)

        self._exit_evaluator = ExitConditionEvaluator(
            config=self.config,
            trailing_stop_checker=self._trailing_stop_manager,
            partial_exit_checker=self._partial_exit_manager,
        )

        # Tracked positions (osi_symbol -> TrackedPosition)
        self._positions: Dict[str, TrackedPosition] = {}

        # Cache for underlying prices
        self._underlying_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_updated: Optional[datetime] = None

        # Session EQUITY-44: Bar cache for pattern invalidation detection
        self._bar_cache: Dict[str, Dict[str, Any]] = {}
        self._bar_cache_updated: Optional[datetime] = None

        # Statistics
        self._check_count = 0
        self._exit_count = 0
        self._error_count = 0

    def sync_positions(self) -> int:
        """
        Sync tracked positions with Alpaca.

        Adds new positions from Alpaca that we're not tracking.
        Removes positions that are no longer open.

        Returns:
            Number of positions added
        """
        if self.trading_client is None:
            logger.warning("No trading client - cannot sync positions")
            return 0

        try:
            alpaca_positions = self.trading_client.list_option_positions()
        except Exception as e:
            logger.error(f"Error fetching positions from Alpaca: {e}")
            self._error_count += 1
            return 0

        alpaca_symbols = {p['symbol'] for p in alpaca_positions}

        # Remove positions no longer on Alpaca
        for osi_symbol in list(self._positions.keys()):
            if osi_symbol not in alpaca_symbols:
                pos = self._positions.pop(osi_symbol)
                logger.info(f"Position closed externally: {osi_symbol}")
                pos.is_active = False

        # Add new positions from Alpaca
        added = 0
        for alpaca_pos in alpaca_positions:
            osi_symbol = alpaca_pos['symbol']

            if osi_symbol in self._positions:
                # Update existing position
                self._update_position_from_alpaca(osi_symbol, alpaca_pos)
            else:
                # Try to find matching execution
                tracked = self._create_tracked_position(alpaca_pos)
                if tracked:
                    self._positions[osi_symbol] = tracked
                    added += 1
                    logger.info(f"Tracking new position: {osi_symbol}")

        return added

    def _update_position_from_alpaca(
        self,
        osi_symbol: str,
        alpaca_pos: Dict[str, Any]
    ) -> None:
        """Update tracked position with Alpaca data."""
        pos = self._positions.get(osi_symbol)
        if not pos:
            return

        pos.current_price = alpaca_pos.get('current_price', 0.0)
        pos.unrealized_pnl = alpaca_pos.get('unrealized_pl', 0.0)

        # Calculate percentage P&L
        cost_basis = pos.entry_price * pos.contracts * 100
        if cost_basis > 0:
            pos.unrealized_pct = pos.unrealized_pnl / cost_basis

        pos.last_updated = datetime.now()

    def _create_tracked_position(
        self,
        alpaca_pos: Dict[str, Any]
    ) -> Optional[TrackedPosition]:
        """
        Create TrackedPosition from Alpaca position.

        Tries to match against executor's executions to get signal data.
        """
        osi_symbol = alpaca_pos['symbol']

        # Find matching execution from executor
        execution: Optional[ExecutionResult] = None
        signal: Optional[StoredSignal] = None

        if self.executor:
            for exec_result in self.executor.get_all_executions().values():
                if exec_result.osi_symbol == osi_symbol:
                    execution = exec_result
                    break

        # Get signal data if we have a matching execution
        if execution and self.signal_store:
            signal = self.signal_store.get_signal(execution.signal_key)

        if not signal:
            # Cannot track without signal data
            logger.warning(
                f"Cannot track {osi_symbol}: no matching signal found"
            )
            return None

        # Parse expiration from OSI symbol
        # Format: SPY241220C00450000 -> 241220 = Dec 20, 2024
        expiration = self._parse_expiration(osi_symbol)
        dte = self._calculate_dte(expiration)

        # Determine direction from signal or OSI symbol
        direction = signal.direction
        if not direction:
            direction = 'CALL' if 'C' in osi_symbol[-9:] else 'PUT'

        # Session EQUITY-36: Calculate optimal exit targets
        entry_trigger = signal.entry_trigger
        stop_price = signal.stop_price
        original_target = signal.target_price
        contracts = alpaca_pos.get('qty', 0)

        # Session EQUITY-36: Get actual underlying entry price from execution
        # For gap-through scenarios, this differs from entry_trigger
        # Fallback to entry_trigger if not available (legacy executions)
        actual_entry_underlying = entry_trigger  # Default to trigger level
        if execution and hasattr(execution, 'underlying_entry_price'):
            if execution.underlying_entry_price and execution.underlying_entry_price > 0:
                actual_entry_underlying = execution.underlying_entry_price
                if abs(actual_entry_underlying - entry_trigger) > 0.10:
                    # Log significant gap-through scenarios
                    logger.info(
                        f"Gap-through entry for {osi_symbol}: "
                        f"trigger=${entry_trigger:.2f}, actual=${actual_entry_underlying:.2f}"
                    )

        # Calculate 1.0x R:R target using ACTUAL entry, not trigger
        # This ensures accurate P/L calculations for gap scenarios
        risk = abs(actual_entry_underlying - stop_price)
        is_bullish = direction.upper() in ['CALL', 'BULL', 'UP']

        if is_bullish:
            target_1x = actual_entry_underlying + risk  # 1.0x R:R target for bullish
        else:
            target_1x = actual_entry_underlying - risk  # 1.0x R:R target for bearish

        # For 1H patterns, use 1.0x target instead of 1.5x
        # (Session EQUITY-36: TSLA analysis showed 1.5x too aggressive for intraday)
        # EQUITY-52: 3-2 patterns use ATR-based targets and bypass 1H override
        effective_target = original_target
        timeframe = signal.timeframe
        pattern_type = signal.pattern_type if hasattr(signal, 'pattern_type') else ''

        # Check if this is a 3-2 pattern (not 3-2-2)
        # Session EQUITY-78: Also include 3-? patterns - when triggered, they ARE 3-2 trades
        # Pattern "3-?" means Type 3 waiting for break, which becomes 3-2U or 3-2D on trigger
        is_32_pattern = (
            ('3-2' in pattern_type and '3-2-2' not in pattern_type) or
            pattern_type.startswith('3-?')
        )

        if is_32_pattern:
            # Session EQUITY-78: 3-2 patterns use simple 1.5% target per strat-methodology
            # This overrides the R:R-based original_target for 3-? setups that triggered
            # Bullish: entry * 1.015, Bearish: entry * 0.985
            if direction.upper() in ['CALL', 'BULL', 'UP']:
                effective_target = actual_entry_underlying * 1.015
            else:
                effective_target = actual_entry_underlying * 0.985
            logger.info(
                f"3-2 pattern {osi_symbol}: Using 1.5% target ${effective_target:.2f} "
                f"(entry: ${actual_entry_underlying:.2f}, pattern: {pattern_type})"
            )
        elif timeframe and timeframe.upper() in ['1H', '60MIN', '60M']:
            # Non-3-2 patterns on 1H: Use 1.0x R:R (EQUITY-36)
            effective_target = target_1x
            logger.info(
                f"1H pattern {osi_symbol}: Adjusted target from ${original_target:.2f} "
                f"(1.5x) to ${target_1x:.2f} (1.0x R:R)"
            )

        # Session EQUITY-44: Get entry bar data from execution for pattern invalidation
        # Note: entry_bar_high/low are actually the SETUP bar bounds (inside bar for X-1-2)
        entry_bar_type = ""
        entry_bar_high = 0.0
        entry_bar_low = 0.0
        if execution:
            entry_bar_type = getattr(execution, 'entry_bar_type', '')
            entry_bar_high = getattr(execution, 'entry_bar_high', 0.0)
            entry_bar_low = getattr(execution, 'entry_bar_low', 0.0)

        # Session EQUITY-48: Initialize intrabar tracking with actual underlying entry price
        # This enables real-time Type 3 detection (not just at bar close)
        # EQUITY-61 FIX: Use underlying price, NOT option price!
        # BUG: alpaca_pos.get('current_price') returns OPTION price (e.g., $1.75),
        # not underlying stock price (e.g., $29). This caused false Type 3 invalidations
        # because intrabar_low was initialized to the option price.
        intrabar_high = actual_entry_underlying if actual_entry_underlying > 0 else 0.0
        intrabar_low = actual_entry_underlying if actual_entry_underlying > 0 else float('inf')

        # Session EQUITY-51: Use actual execution timestamp, not datetime.now()
        # This is critical for stale 1H position detection - if we use now(),
        # a position from yesterday would appear as entered today after daemon restart
        entry_time = datetime.now()
        if execution and hasattr(execution, 'timestamp') and execution.timestamp:
            entry_time = execution.timestamp
            logger.debug(f"Using execution timestamp for {osi_symbol}: {entry_time}")

        # EQUITY-52: Initialize ATR-based trailing stop for 3-2 patterns
        atr_at_detection = getattr(signal, 'atr_14', 0.0) or 0.0
        use_atr_trailing = (
            is_32_pattern and
            self.config.use_atr_trailing_for_32 and
            atr_at_detection > 0
        )
        atr_trail_distance = 0.0
        atr_activation_level = 0.0

        if use_atr_trailing:
            # Pre-calculate ATR trail distance (1.0 ATR)
            atr_trail_distance = atr_at_detection * self.config.atr_trailing_distance_multiple
            # Pre-calculate activation level (0.75 ATR profit from entry)
            atr_activation_profit = atr_at_detection * self.config.atr_trailing_activation_multiple
            if is_bullish:
                atr_activation_level = actual_entry_underlying + atr_activation_profit
            else:
                atr_activation_level = actual_entry_underlying - atr_activation_profit

            logger.info(
                f"3-2 ATR trailing initialized for {osi_symbol}: "
                f"ATR=${atr_at_detection:.2f}, activation=${atr_activation_level:.2f}, "
                f"trail_distance=${atr_trail_distance:.2f}"
            )

        return TrackedPosition(
            osi_symbol=osi_symbol,
            signal_key=execution.signal_key if execution else "",
            symbol=signal.symbol,
            direction=direction,
            entry_trigger=entry_trigger,
            target_price=effective_target,
            stop_price=stop_price,
            pattern_type=signal.pattern_type,
            timeframe=timeframe,
            entry_price=alpaca_pos.get('avg_entry_price', 0.0),
            contracts=contracts,
            entry_time=entry_time,
            expiration=expiration,
            current_price=alpaca_pos.get('current_price', 0.0),
            unrealized_pnl=alpaca_pos.get('unrealized_pl', 0.0),
            dte=dte,
            # Session EQUITY-36: Optimal exit fields
            target_1x=target_1x,
            original_target=original_target,
            actual_entry_underlying=actual_entry_underlying,  # Gap-through fix
            contracts_remaining=contracts,
            # Session EQUITY-44: Pattern invalidation tracking
            entry_bar_type=entry_bar_type,
            entry_bar_high=entry_bar_high,
            entry_bar_low=entry_bar_low,
            # Session EQUITY-48: Real-time Type 3 detection
            intrabar_high=intrabar_high,
            intrabar_low=intrabar_low,
            # EQUITY-52: ATR-based trailing stop for 3-2 patterns
            atr_at_detection=atr_at_detection,
            use_atr_trailing=use_atr_trailing,
            atr_trail_distance=atr_trail_distance,
            atr_activation_level=atr_activation_level,
        )

    def _parse_expiration(self, osi_symbol: str) -> str:
        """Parse expiration date from OSI symbol."""
        # OSI format: SPY241220C00450000
        # Extract YYMMDD from positions 3-9 (after ticker)
        try:
            # Find where the date starts (after ticker, before C/P)
            for i, char in enumerate(osi_symbol):
                if char.isdigit():
                    date_str = osi_symbol[i:i+6]
                    year = 2000 + int(date_str[0:2])
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    return f"{year}-{month:02d}-{day:02d}"
        except (ValueError, IndexError):
            pass

        return ""

    def _calculate_dte(self, expiration: str) -> int:
        """Calculate days to expiration."""
        if not expiration:
            return 0

        try:
            exp_date = datetime.strptime(expiration, "%Y-%m-%d")
            return max(0, (exp_date - datetime.now()).days)
        except ValueError:
            return 0

    def check_positions(self) -> List[ExitSignal]:
        """
        Check all positions for exit conditions.

        Returns:
            List of ExitSignal for positions that should be closed
        """
        self._check_count += 1
        exit_signals: List[ExitSignal] = []

        # Sync with Alpaca first
        self.sync_positions()

        # Update underlying prices
        self._update_underlying_prices()

        # Session EQUITY-44: Update bar data for pattern invalidation detection
        self._update_bar_data()

        for osi_symbol, pos in self._positions.items():
            if not pos.is_active:
                continue

            exit_signal = self._check_position(pos)
            if exit_signal:
                exit_signals.append(exit_signal)

        return exit_signals

    def _check_position(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Check single position for exit conditions.

        EQUITY-90: Delegates to ExitConditionEvaluator for main logic.
        Trailing stop and partial exit are still handled locally until
        Phase 4.2 and 4.3 complete extraction.

        Exit conditions (in priority order):
        0. Minimum hold time check (Session 83K-77 - prevent rapid exit)
        0.5. EOD exit for 1H trades (Session EQUITY-35 - avoid overnight gap)
        1. DTE exit (mandatory - theta decay risk)
        2. Stop hit (underlying price check)
        3. Max loss exceeded (timeframe-specific % thresholds - Session EQUITY-42)
        4. Target hit (MOVED BEFORE trailing stop - Session EQUITY-42)
        5. Trailing stop (only if option P/L >= min threshold - Session EQUITY-42)
        6. Partial exit (multi-contract positions)
        7. Max profit exceeded (take profits)
        """
        # EQUITY-90: Check minimum hold time FIRST to block all exit checks
        # This ensures local trailing/partial exit checks also respect minimum hold
        hold_duration = (datetime.now() - pos.entry_time).total_seconds()
        if hold_duration < self.config.minimum_hold_seconds:
            logger.debug(
                f"{pos.osi_symbol}: Held {hold_duration:.0f}s < "
                f"min {self.config.minimum_hold_seconds}s - skipping exit check"
            )
            return None

        # Get underlying price for the evaluator
        underlying_price = self._get_underlying_price(pos.symbol)

        # Get bar data for pattern invalidation check
        bar_data = self._bar_cache.get(pos.symbol)

        # EQUITY-90: Delegate to ExitConditionEvaluator
        # Note: Trailing stop and partial exit checkers not yet wired (Phase 4.2/4.3)
        # Those checks fall through to local methods below
        exit_signal = self._exit_evaluator.evaluate(
            pos=pos,
            underlying_price=underlying_price,
            bar_data=bar_data,
        )

        # EQUITY-90: All exit checks now handled by evaluator + managers
        # - TrailingStopManager (Phase 4.2)
        # - PartialExitManager (Phase 4.3)
        return exit_signal

    def _check_target_hit(self, pos: TrackedPosition) -> bool:
        """Check if underlying has reached target price."""
        if pos.direction.upper() in ['CALL', 'BULL', 'UP']:
            # For calls, target is above entry
            return pos.underlying_price >= pos.target_price
        else:
            # For puts, target is below entry
            return pos.underlying_price <= pos.target_price

    def _check_stop_hit(self, pos: TrackedPosition) -> bool:
        """Check if underlying has reached stop price."""
        if pos.direction.upper() in ['CALL', 'BULL', 'UP']:
            # For calls, stop is below entry
            return pos.underlying_price <= pos.stop_price
        else:
            # For puts, stop is above entry
            return pos.underlying_price >= pos.stop_price

    # EQUITY-90: Trailing stop and partial exit methods removed
    # Now handled by TrailingStopManager and PartialExitManager
    # See strat/signal_automation/coordinators/

    def _update_underlying_prices(self) -> None:
        """
        Update underlying price cache from Alpaca market data.

        Session 83K-50: Uses new get_stock_price() method for real-time quotes.
        This works regardless of whether we hold an equity position.
        """
        if not self.trading_client:
            return

        # Get unique underlying symbols
        symbols = list({pos.symbol for pos in self._positions.values() if pos.is_active})

        if not symbols:
            return

        # Batch fetch quotes for all symbols (more efficient)
        try:
            quotes = self.trading_client.get_stock_quotes(symbols)

            for symbol in symbols:
                quote = quotes.get(symbol.upper())
                if quote:
                    self._underlying_cache[symbol] = {
                        'price': quote['mid'],
                        'updated': datetime.now(),
                    }
                    logger.debug(f"Updated {symbol} price: ${quote['mid']:.2f}")
                else:
                    # Quote not available - keep previous value if exists
                    if symbol not in self._underlying_cache:
                        self._underlying_cache[symbol] = {
                            'price': None,
                            'updated': datetime.now(),
                        }

        except Exception as e:
            logger.warning(f"Error fetching underlying prices: {e}")

    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get underlying price from cache."""
        cached = self._underlying_cache.get(symbol)
        if cached:
            return cached.get('price')
        return None

    def _update_bar_data(self) -> None:
        """
        Update bar data cache from Alpaca market data.

        Session EQUITY-44: Used for pattern invalidation detection.
        Fetches latest bar OHLCV to check if entry bar evolved to Type 3.
        """
        if not self.trading_client:
            return

        # Get unique underlying symbols from positions with entry bar data
        symbols = list({
            pos.symbol for pos in self._positions.values()
            if pos.is_active and pos.entry_bar_high > 0 and pos.entry_bar_low > 0
        })

        if not symbols:
            return

        try:
            bars = self.trading_client.get_latest_bars(symbols)

            for symbol in symbols:
                bar = bars.get(symbol.upper())
                if bar:
                    self._bar_cache[symbol] = bar
                    logger.debug(
                        f"Updated {symbol} bar: H=${bar['high']:.2f} L=${bar['low']:.2f}"
                    )

            self._bar_cache_updated = datetime.now()

        except Exception as e:
            logger.warning(f"Error fetching bar data: {e}")

    def _check_pattern_invalidation(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Check if entry bar evolved to Type 3 (pattern invalidation).

        Session EQUITY-44: Per STRAT methodology EXECUTION.md Section 8,
        if the entry bar breaks BOTH high and low (Type 3 evolution),
        exit immediately - the pattern premise is invalidated.

        Session EQUITY-48: Enhanced with REAL-TIME detection using intrabar
        high/low tracking. Previously only detected Type 3 at bar close via
        bar cache. Now detects Type 3 evolution as it happens using accumulated
        intrabar_high and intrabar_low since entry.

        Exit Priority:
        1. Target Hit
        2. Pattern Invalidated (Type 3 evolution) <- This check
        3. Traditional Stop

        Args:
            pos: TrackedPosition to check

        Returns:
            ExitSignal if pattern invalidated, None otherwise
        """
        # Skip if not a Type 2 entry or missing setup bar data
        # Note: entry_bar_high/low are actually setup bar (inside bar) bounds
        if pos.entry_bar_type not in ['2U', '2D']:
            return None

        if pos.entry_bar_high <= 0 or pos.entry_bar_low <= 0:
            return None

        # Session EQUITY-48: Use intrabar extremes for real-time detection
        # Compare accumulated high/low since entry against setup bar bounds
        # This detects Type 3 evolution as it happens, not just at bar close
        intrabar_high = pos.intrabar_high
        intrabar_low = pos.intrabar_low

        # Fallback to bar cache if intrabar tracking not initialized
        if intrabar_high <= 0 or intrabar_low == float('inf'):
            bar_data = self._bar_cache.get(pos.symbol)
            if bar_data:
                # Validate bar cache has required keys
                cache_high = bar_data.get('high', 0)
                cache_low = bar_data.get('low', float('inf'))
                if cache_high <= 0 or cache_low == float('inf'):
                    logger.debug(
                        f"Pattern invalidation skipped for {pos.symbol}: "
                        f"incomplete bar cache (H={cache_high}, L={cache_low})"
                    )
                    return None
                intrabar_high = cache_high
                intrabar_low = cache_low
            else:
                logger.debug(
                    f"Pattern invalidation skipped for {pos.symbol}: "
                    f"no intrabar tracking and no bar cache"
                )
                return None

        # Check for Type 3 evolution: broke BOTH the setup bar high AND low
        broke_high = intrabar_high > pos.entry_bar_high
        broke_low = intrabar_low < pos.entry_bar_low

        if broke_high and broke_low:
            # Pattern invalidated! Entry bar evolved to Type 3
            details = (
                f"Entry bar evolved to Type 3: "
                f"Setup H=${pos.entry_bar_high:.2f} L=${pos.entry_bar_low:.2f}, "
                f"Intrabar H=${intrabar_high:.2f} L=${intrabar_low:.2f}"
            )

            # Session EQUITY-48: Include signal_key for lifecycle tracing
            logger.warning(
                f"PATTERN INVALIDATED: {pos.signal_key} ({pos.osi_symbol}) - {details}"
            )

            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.PATTERN_INVALIDATED,
                underlying_price=self._get_underlying_price(pos.symbol) or 0.0,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=details,
            )

        return None

    def _is_market_hours(self) -> bool:
        """
        Check if within NYSE market hours.

        Session 83K-77: Prevent exit attempts outside market hours.
        Session EQUITY-36: Uses pandas_market_calendars for accurate holiday
        and early close handling (e.g., Christmas Eve 1PM, day after Christmas).
        Session EQUITY-86: Delegates to shared MarketHoursValidator utility.
        """
        return self._market_hours_validator.is_market_hours()

    def _is_stale_1h_position(self, entry_time: datetime) -> bool:
        """
        Check if a 1H position was entered on a previous trading day.

        Session EQUITY-51: 1H positions must exit before market close on the
        SAME day they were entered. If a position somehow survives overnight
        (daemon restart, timing issue, etc.), it should be closed immediately
        at the next opportunity - not wait until today's 15:59.

        Returns:
            True if entry was on a previous trading day (stale position)
            False if entry was today (normal position)
        """
        import pytz
        import pandas_market_calendars as mcal

        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)

        # Make entry_time timezone-aware if needed
        if entry_time.tzinfo is None:
            entry_time_et = et.localize(entry_time)
        else:
            entry_time_et = entry_time.astimezone(et)

        # Same calendar day = not stale
        if entry_time_et.date() == now_et.date():
            return False

        # Different calendar day - check if it's a different TRADING day
        # (handles weekend positions that might span Fri->Mon)
        nyse = mcal.get_calendar('NYSE')

        # Get trading days between entry date and today
        schedule = nyse.schedule(
            start_date=entry_time_et.date(),
            end_date=now_et.date()
        )

        # If there's more than 1 trading day in the range, it's stale
        # (entry day + today = 2 days minimum for overnight hold)
        if len(schedule) > 1:
            logger.warning(
                f"STALE 1H POSITION: Entered on {entry_time_et.date()}, "
                f"now {now_et.date()} - {len(schedule)} trading days span"
            )
            return True

        return False

    def execute_exit(self, exit_signal: ExitSignal) -> Optional[Dict[str, Any]]:
        """
        Execute an exit for a position.

        Args:
            exit_signal: ExitSignal with exit details

        Returns:
            Order result or None if failed
        """
        # Session 83K-77: Skip exits outside market hours
        # Session EQUITY-95: All exits respect market hours, including EOD.
        # EOD fires at 15:59 while market is still open (< 16:00). After close,
        # exits are blocked here; stale 1H positions exit next morning via
        # _is_stale_1h_position() check at market open.
        if not self._is_market_hours():
            logger.debug(
                f"Skipping exit for {exit_signal.osi_symbol}: "
                f"outside market hours ({exit_signal.reason.value})"
            )
            return None

        if not self.trading_client:
            logger.error("No trading client - cannot execute exit")
            return None

        pos = self._positions.get(exit_signal.osi_symbol)
        if not pos:
            logger.warning(f"Position not found: {exit_signal.osi_symbol}")
            return None

        try:
            # Session EQUITY-36: Handle partial exits
            is_partial = exit_signal.contracts_to_close is not None
            qty_to_close = exit_signal.contracts_to_close

            # Session EQUITY-48: Include signal_key for lifecycle tracing
            logger.info(
                f"Executing {'partial ' if is_partial else ''}exit: {exit_signal.signal_key} "
                f"({exit_signal.osi_symbol}) - {exit_signal.reason.value}"
            )

            result = self.trading_client.close_option_position(
                exit_signal.osi_symbol,
                qty=qty_to_close  # None = close all, int = partial close
            )

            if result:
                if is_partial:
                    # Partial exit - update position state but keep active
                    pos.partial_exit_done = True
                    pos.contracts_remaining = pos.contracts - qty_to_close

                    logger.info(
                        f"Partial exit executed: {pos.signal_key} ({exit_signal.osi_symbol}) - "
                        f"Closed {qty_to_close}, remaining {pos.contracts_remaining}"
                    )
                else:
                    # Full exit - mark position as inactive
                    pos.is_active = False
                    pos.exit_reason = exit_signal.reason.value
                    pos.exit_time = datetime.now()

                    # Session EQUITY-41: Use actual fill price from order, not cached quote
                    # pos.current_price can be $0 when Alpaca fails to provide quote
                    filled_price = result.get('filled_avg_price') if isinstance(result, dict) else None
                    if filled_price and filled_price > 0:
                        pos.exit_price = filled_price
                    else:
                        pos.exit_price = pos.current_price
                        if pos.current_price <= 0:
                            logger.warning(
                                f"Invalid exit price for {exit_signal.osi_symbol}: "
                                f"fill={filled_price}, quote={pos.current_price}"
                            )

                    pos.realized_pnl = pos.unrealized_pnl

                    logger.info(
                        f"Position closed: {pos.signal_key} ({exit_signal.osi_symbol}) - "
                        f"P&L: ${pos.realized_pnl:.2f} ({exit_signal.reason.value})"
                    )

                self._exit_count += 1

                # Callback for alerting
                if self.on_exit_callback:
                    try:
                        self.on_exit_callback(exit_signal, result)
                    except Exception as e:
                        logger.error(f"Exit callback error: {e}")

                return result
            else:
                # Session EQUITY-45: Log when close_option_position returns falsy
                # This was causing silent failures and infinite partial exit loops
                # Session EQUITY-48: Include signal_key for lifecycle tracing
                logger.error(
                    f"Exit failed: {exit_signal.signal_key} ({exit_signal.osi_symbol}) - "
                    f"close_option_position returned falsy ({exit_signal.reason.value})"
                )
                self._error_count += 1

        except Exception as e:
            logger.error(f"Exit execution error: {exit_signal.signal_key} ({exit_signal.osi_symbol}) - {e}")
            self._error_count += 1

        return None

    def execute_all_exits(
        self,
        exit_signals: List[ExitSignal]
    ) -> List[Dict[str, Any]]:
        """
        Execute all pending exits.

        Args:
            exit_signals: List of ExitSignal to execute

        Returns:
            List of order results
        """
        results = []
        for signal in exit_signals:
            result = self.execute_exit(signal)
            if result:
                results.append(result)
        return results

    def get_tracked_positions(self) -> List[TrackedPosition]:
        """Get all tracked positions."""
        return list(self._positions.values())

    def get_active_positions(self) -> List[TrackedPosition]:
        """Get active (not exited) positions."""
        return [p for p in self._positions.values() if p.is_active]

    def get_position(self, osi_symbol: str) -> Optional[TrackedPosition]:
        """Get a specific tracked position."""
        return self._positions.get(osi_symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        active = len([p for p in self._positions.values() if p.is_active])
        closed = len([p for p in self._positions.values() if not p.is_active])

        total_pnl = sum(
            p.unrealized_pnl for p in self._positions.values() if p.is_active
        )

        return {
            'active_positions': active,
            'closed_positions': closed,
            'total_unrealized_pnl': total_pnl,
            'check_count': self._check_count,
            'exit_count': self._exit_count,
            'error_count': self._error_count,
        }


# Convenience function
def create_position_monitor(
    executor: SignalExecutor,
    trading_client: AlpacaTradingClient,
    signal_store: SignalStore,
) -> PositionMonitor:
    """Create a position monitor with default config."""
    return PositionMonitor(
        config=MonitoringConfig(),
        executor=executor,
        trading_client=trading_client,
        signal_store=signal_store,
    )
