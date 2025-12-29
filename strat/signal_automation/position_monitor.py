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


@dataclass
class MonitoringConfig:
    """Configuration for position monitoring."""
    # Exit thresholds
    exit_dte: int = 3                    # Close at or below this DTE
    max_loss_pct: float = 0.50           # Max loss as % of premium (50%)
    max_profit_pct: float = 1.00         # Take profit at 100% gain

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
    eod_exit_hour: int = 15              # Hour in ET for EOD exit
    eod_exit_minute: int = 55            # Minute in ET for EOD exit (15:55 = 5 min buffer)

    # Session EQUITY-36: Optimal exit strategy
    # 1H patterns get reduced target (1.0x R:R instead of 1.5x)
    hourly_target_rr: float = 1.0        # R:R target for 1H patterns (was 1.5x)

    # Trailing stop - activate once in profit, trail at percentage of max profit
    use_trailing_stop: bool = True       # Enable trailing stop for single-contract or remainder
    trailing_stop_activation_rr: float = 0.5  # Activate trailing stop at 0.5x R:R profit
    trailing_stop_pct: float = 0.50      # Trail 50% below high water mark

    # Partial exits - for multi-contract positions
    partial_exit_enabled: bool = True    # Enable partial exits at 1.0x R:R
    partial_exit_rr: float = 1.0         # Take partial profit at 1.0x R:R
    partial_exit_pct: float = 0.50       # Exit 50% of contracts at partial target


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

    # Trailing stop state
    trailing_stop_active: bool = False   # Whether trailing stop is activated
    trailing_stop_price: float = 0.0     # Current trailing stop level (underlying price)
    high_water_mark: float = 0.0         # Best underlying price seen (for trailing calc)

    # Partial exit state
    partial_exit_done: bool = False      # Whether partial exit has been executed
    contracts_remaining: int = 0         # Contracts remaining after partial exit

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
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            'high_water_mark': self.high_water_mark,
            'partial_exit_done': self.partial_exit_done,
            'contracts_remaining': self.contracts_remaining,
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

        # Tracked positions (osi_symbol -> TrackedPosition)
        self._positions: Dict[str, TrackedPosition] = {}

        # Cache for underlying prices
        self._underlying_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_updated: Optional[datetime] = None

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

        # Calculate 1.0x R:R target
        risk = abs(entry_trigger - stop_price)
        is_bullish = direction.upper() in ['CALL', 'BULL', 'UP']

        if is_bullish:
            target_1x = entry_trigger + risk  # 1.0x R:R target for bullish
        else:
            target_1x = entry_trigger - risk  # 1.0x R:R target for bearish

        # For 1H patterns, use 1.0x target instead of 1.5x
        # (Session EQUITY-36: TSLA analysis showed 1.5x too aggressive for intraday)
        effective_target = original_target
        timeframe = signal.timeframe
        if timeframe and timeframe.upper() in ['1H', '60MIN', '60M']:
            effective_target = target_1x
            logger.info(
                f"1H pattern {osi_symbol}: Adjusted target from ${original_target:.2f} "
                f"(1.5x) to ${target_1x:.2f} (1.0x R:R)"
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
            entry_time=datetime.now(),  # Approximate
            expiration=expiration,
            current_price=alpaca_pos.get('current_price', 0.0),
            unrealized_pnl=alpaca_pos.get('unrealized_pl', 0.0),
            dte=dte,
            # Session EQUITY-36: Optimal exit fields
            target_1x=target_1x,
            original_target=original_target,
            contracts_remaining=contracts,
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

        Exit conditions (in priority order):
        0. Minimum hold time check (Session 83K-77 - prevent rapid exit)
        0.5. EOD exit for 1H trades (Session EQUITY-35 - avoid overnight gap)
        1. DTE exit (mandatory - theta decay risk)
        2. Stop hit (loss management)
        3. Max loss exceeded (risk management)
        4. Target hit (profit taking)
        5. Max profit exceeded (take profits)
        """
        # 0. Check minimum hold time before any exit condition (Session 83K-77)
        hold_duration = (datetime.now() - pos.entry_time).total_seconds()
        if hold_duration < self.config.minimum_hold_seconds:
            logger.debug(
                f"{pos.osi_symbol}: Held {hold_duration:.0f}s < "
                f"min {self.config.minimum_hold_seconds}s - skipping exit check"
            )
            return None

        # 0.5. Session EQUITY-35: EOD exit for 1H trades
        # All hourly trades must exit before market close to avoid overnight gap risk
        if pos.timeframe == '1H':
            import pytz
            et = pytz.timezone('America/New_York')
            now_et = datetime.now(et)
            eod_exit_time = now_et.replace(
                hour=self.config.eod_exit_hour,
                minute=self.config.eod_exit_minute,
                second=0,
                microsecond=0
            )
            if now_et >= eod_exit_time:
                logger.info(
                    f"EOD EXIT: {pos.osi_symbol} (1H) - "
                    f"current time {now_et.strftime('%H:%M ET')} >= "
                    f"EOD cutoff {self.config.eod_exit_hour}:{self.config.eod_exit_minute:02d} ET"
                )
                return ExitSignal(
                    osi_symbol=pos.osi_symbol,
                    signal_key=pos.signal_key,
                    reason=ExitReason.EOD_EXIT,
                    underlying_price=pos.underlying_price or 0.0,
                    current_option_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    dte=pos.dte,
                    details=f"1H trade EOD exit at {now_et.strftime('%H:%M ET')} (cutoff: {self.config.eod_exit_hour}:{self.config.eod_exit_minute:02d} ET)",
                )

        # Update DTE
        pos.dte = self._calculate_dte(pos.expiration)

        # Get underlying price
        underlying_price = self._get_underlying_price(pos.symbol)
        if underlying_price:
            pos.underlying_price = underlying_price

        # 1. DTE Exit - mandatory close before expiration
        if pos.dte <= self.config.exit_dte:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.DTE_EXIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"DTE {pos.dte} <= threshold {self.config.exit_dte}",
            )

        # Need underlying price for target/stop checks
        if not pos.underlying_price:
            logger.debug(f"No underlying price for {pos.symbol}")
            return None

        # 2. Check stop hit
        if self._check_stop_hit(pos):
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.STOP_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Underlying ${pos.underlying_price:.2f} hit stop ${pos.stop_price:.2f}",
            )

        # 3. Max loss check
        if pos.unrealized_pct <= -self.config.max_loss_pct:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.MAX_LOSS,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Loss {pos.unrealized_pct:.1%} >= max {self.config.max_loss_pct:.1%}",
            )

        # Session EQUITY-36: Optimal exit strategy
        # 3.5. Update trailing stop state and check conditions
        if self.config.use_trailing_stop:
            trailing_signal = self._check_trailing_stop(pos)
            if trailing_signal:
                return trailing_signal

        # 3.6. Check partial exit for multi-contract positions
        if self.config.partial_exit_enabled:
            partial_signal = self._check_partial_exit(pos)
            if partial_signal:
                return partial_signal

        # 4. Check target hit
        if self._check_target_hit(pos):
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.TARGET_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Underlying ${pos.underlying_price:.2f} hit target ${pos.target_price:.2f}",
            )

        # 5. Max profit check
        if pos.unrealized_pct >= self.config.max_profit_pct:
            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.TARGET_HIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Profit {pos.unrealized_pct:.1%} >= target {self.config.max_profit_pct:.1%}",
            )

        return None

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

    def _check_trailing_stop(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Session EQUITY-36: Check and update trailing stop logic.

        Logic:
        1. Update high water mark as price moves in our favor
        2. Activate trailing stop once 0.5x R:R profit is reached
        3. Trail stop at 50% of profit from high water mark
        4. Exit if price retraces to trailing stop level

        Returns ExitSignal if trailing stop is hit, None otherwise.
        """
        is_bullish = pos.direction.upper() in ['CALL', 'BULL', 'UP']
        risk = abs(pos.entry_trigger - pos.stop_price)
        activation_threshold = self.config.trailing_stop_activation_rr * risk

        # Calculate profit in direction of trade
        if is_bullish:
            current_profit = pos.underlying_price - pos.entry_trigger
            # Update high water mark
            if pos.underlying_price > pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price
        else:
            current_profit = pos.entry_trigger - pos.underlying_price
            # Update high water mark (lowest price for puts)
            if pos.underlying_price < pos.high_water_mark or pos.high_water_mark == 0:
                pos.high_water_mark = pos.underlying_price

        # Check if we should activate trailing stop
        if not pos.trailing_stop_active and current_profit >= activation_threshold:
            pos.trailing_stop_active = True
            logger.info(
                f"Trailing stop ACTIVATED for {pos.osi_symbol}: "
                f"profit ${current_profit:.2f} >= activation ${activation_threshold:.2f} (0.5x R:R)"
            )

        # If trailing stop is active, calculate and check trailing stop level
        if pos.trailing_stop_active:
            # Calculate trail amount (50% of max profit from high water mark)
            if is_bullish:
                max_profit_from_hwm = pos.high_water_mark - pos.entry_trigger
                trail_amount = max_profit_from_hwm * self.config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark - trail_amount

                # Check if trailing stop is hit
                if pos.underlying_price <= pos.trailing_stop_price:
                    logger.info(
                        f"TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} <= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"Trailing stop at ${pos.trailing_stop_price:.2f} hit (HWM: ${pos.high_water_mark:.2f})",
                    )
            else:  # Bearish (PUT)
                max_profit_from_hwm = pos.entry_trigger - pos.high_water_mark
                trail_amount = max_profit_from_hwm * self.config.trailing_stop_pct
                pos.trailing_stop_price = pos.high_water_mark + trail_amount

                # Check if trailing stop is hit
                if pos.underlying_price >= pos.trailing_stop_price:
                    logger.info(
                        f"TRAILING STOP HIT for {pos.osi_symbol}: "
                        f"${pos.underlying_price:.2f} >= trail ${pos.trailing_stop_price:.2f} "
                        f"(HWM: ${pos.high_water_mark:.2f})"
                    )
                    return ExitSignal(
                        osi_symbol=pos.osi_symbol,
                        signal_key=pos.signal_key,
                        reason=ExitReason.TRAILING_STOP,
                        underlying_price=pos.underlying_price,
                        current_option_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        dte=pos.dte,
                        details=f"Trailing stop at ${pos.trailing_stop_price:.2f} hit (HWM: ${pos.high_water_mark:.2f})",
                    )

        return None

    def _check_partial_exit(self, pos: TrackedPosition) -> Optional[ExitSignal]:
        """
        Session EQUITY-36: Check for partial exit at 1.0x R:R.

        Logic:
        1. Only for positions with contracts > 1
        2. Only if partial exit not already done
        3. Exit 50% of contracts when 1.0x R:R target is hit

        Returns ExitSignal for partial exit if conditions met, None otherwise.
        """
        # Skip if only 1 contract (use trailing stop instead)
        if pos.contracts <= 1:
            return None

        # Skip if partial exit already done
        if pos.partial_exit_done:
            return None

        # Check if 1.0x R:R target reached
        is_bullish = pos.direction.upper() in ['CALL', 'BULL', 'UP']

        if is_bullish:
            target_1x_reached = pos.underlying_price >= pos.target_1x
        else:
            target_1x_reached = pos.underlying_price <= pos.target_1x

        if target_1x_reached:
            # Calculate contracts to close (50% rounded up)
            contracts_to_close = max(1, int(pos.contracts * self.config.partial_exit_pct + 0.5))

            logger.info(
                f"PARTIAL EXIT for {pos.osi_symbol}: "
                f"Closing {contracts_to_close} of {pos.contracts} contracts at 1.0x R:R "
                f"(underlying ${pos.underlying_price:.2f} hit target_1x ${pos.target_1x:.2f})"
            )

            return ExitSignal(
                osi_symbol=pos.osi_symbol,
                signal_key=pos.signal_key,
                reason=ExitReason.PARTIAL_EXIT,
                underlying_price=pos.underlying_price,
                current_option_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                dte=pos.dte,
                details=f"Partial exit: {contracts_to_close}/{pos.contracts} contracts at 1.0x R:R (${pos.target_1x:.2f})",
                contracts_to_close=contracts_to_close,
            )

        return None

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

    def _is_market_hours(self) -> bool:
        """
        Check if within options market hours (9:30 AM - 4:00 PM ET, Mon-Fri).

        Session 83K-77: Prevent exit attempts outside market hours.
        """
        import pytz
        et = pytz.timezone('America/New_York')
        now = datetime.now(et)
        # Weekend check
        if now.weekday() >= 5:
            return False
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def execute_exit(self, exit_signal: ExitSignal) -> Optional[Dict[str, Any]]:
        """
        Execute an exit for a position.

        Args:
            exit_signal: ExitSignal with exit details

        Returns:
            Order result or None if failed
        """
        # Session 83K-77: Skip exits outside market hours
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

            logger.info(
                f"Executing {'partial ' if is_partial else ''}exit for {exit_signal.osi_symbol}: "
                f"{exit_signal.reason.value} - {exit_signal.details}"
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
                        f"Partial exit executed: {exit_signal.osi_symbol} - "
                        f"Closed {qty_to_close}, remaining {pos.contracts_remaining}"
                    )
                else:
                    # Full exit - mark position as inactive
                    pos.is_active = False
                    pos.exit_reason = exit_signal.reason.value
                    pos.exit_time = datetime.now()
                    pos.exit_price = pos.current_price
                    pos.realized_pnl = pos.unrealized_pnl

                    logger.info(
                        f"Position closed: {exit_signal.osi_symbol} - "
                        f"P&L: ${pos.realized_pnl:.2f}"
                    )

                self._exit_count += 1

                # Callback for alerting
                if self.on_exit_callback:
                    try:
                        self.on_exit_callback(exit_signal, result)
                    except Exception as e:
                        logger.error(f"Exit callback error: {e}")

                return result

        except Exception as e:
            logger.error(f"Exit execution error for {exit_signal.osi_symbol}: {e}")
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
