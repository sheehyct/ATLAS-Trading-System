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

    # Exit behavior
    use_market_orders: bool = True       # Use market orders for exits
    close_partial_on_profit: bool = False  # Close half at 50% gain (future)

    # Alerting
    alert_on_exit: bool = True           # Send alerts for exits
    alert_on_profit_target: bool = True  # Alert when approaching profit target
    alert_pct_to_target: float = 0.80    # Alert when 80% to target


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

        return TrackedPosition(
            osi_symbol=osi_symbol,
            signal_key=execution.signal_key if execution else "",
            symbol=signal.symbol,
            direction=direction,
            entry_trigger=signal.entry_trigger,
            target_price=signal.target_price,
            stop_price=signal.stop_price,
            pattern_type=signal.pattern_type,
            timeframe=signal.timeframe,
            entry_price=alpaca_pos.get('avg_entry_price', 0.0),
            contracts=alpaca_pos.get('qty', 0),
            entry_time=datetime.now(),  # Approximate
            expiration=expiration,
            current_price=alpaca_pos.get('current_price', 0.0),
            unrealized_pnl=alpaca_pos.get('unrealized_pl', 0.0),
            dte=dte,
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
        1. DTE exit (mandatory - theta decay risk)
        2. Stop hit (loss management)
        3. Max loss exceeded (risk management)
        4. Target hit (profit taking)
        5. Max profit exceeded (take profits)
        """
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

    def execute_exit(self, exit_signal: ExitSignal) -> Optional[Dict[str, Any]]:
        """
        Execute an exit for a position.

        Args:
            exit_signal: ExitSignal with exit details

        Returns:
            Order result or None if failed
        """
        if not self.trading_client:
            logger.error("No trading client - cannot execute exit")
            return None

        pos = self._positions.get(exit_signal.osi_symbol)
        if not pos:
            logger.warning(f"Position not found: {exit_signal.osi_symbol}")
            return None

        try:
            logger.info(
                f"Executing exit for {exit_signal.osi_symbol}: "
                f"{exit_signal.reason.value} - {exit_signal.details}"
            )

            result = self.trading_client.close_option_position(exit_signal.osi_symbol)

            if result:
                # Update position state
                pos.is_active = False
                pos.exit_reason = exit_signal.reason.value
                pos.exit_time = datetime.now()
                pos.exit_price = pos.current_price
                pos.realized_pnl = pos.unrealized_pnl

                self._exit_count += 1

                logger.info(
                    f"Position closed: {exit_signal.osi_symbol} - "
                    f"P&L: ${pos.realized_pnl:.2f}"
                )

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
