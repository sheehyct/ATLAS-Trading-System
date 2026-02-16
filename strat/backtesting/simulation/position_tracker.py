"""
SimulatedPosition - Backtest equivalent of TrackedPosition

Pure data container mirroring the live TrackedPosition's 30+ fields
without any Alpaca dependency, threading, or I/O side effects.

Used by the bar simulator and exit evaluator to track open positions
during backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class ExitReason(str, Enum):
    """
    Reason for position exit.

    Local copy of the live ExitReason to avoid importing from
    position_monitor.py (which pulls in AlpacaTradingClient).
    Must be kept in sync with the live enum values.
    """
    TARGET_HIT = "TARGET"
    STOP_HIT = "STOP"
    DTE_EXIT = "DTE"
    MAX_LOSS = "MAX_LOSS"
    TIME_EXIT = "TIME"
    EOD_EXIT = "EOD"
    PARTIAL_EXIT = "PARTIAL"
    TRAILING_STOP = "TRAIL"
    PATTERN_INVALIDATED = "PATTERN"
    MANUAL = "MANUAL"


@dataclass
class SimulatedPosition:
    """
    A simulated position being tracked during backtesting.

    Mirrors TrackedPosition from position_monitor.py but without
    Alpaca dependencies. All fields that the exit evaluator,
    trailing stop, and partial exit logic need are present.
    """

    # ── Position Identifiers ────────────────────────────────────────
    symbol: str                      # Underlying symbol (e.g., 'SPY')
    signal_key: str                  # Unique signal identifier
    osi_symbol: str = ""             # Option symbol (constructed during entry)

    # ── Original Signal Data ────────────────────────────────────────
    direction: str = ""              # 'CALL' or 'PUT'
    entry_trigger: float = 0.0      # Signal entry trigger price
    target_price: float = 0.0       # Signal target price
    stop_price: float = 0.0         # Signal stop price
    pattern_type: str = ""           # e.g., '2-1-2', '3-2'
    timeframe: str = ""              # e.g., '1H', '1D', '1W', '1M'

    # ── Execution Data ──────────────────────────────────────────────
    entry_price: float = 0.0        # Option premium paid (ask at entry)
    contracts: int = 1              # Number of contracts
    entry_time: Optional[datetime] = None
    expiration: str = ""            # Option expiration date (YYYY-MM-DD)

    # Actual underlying price at execution (may differ from trigger for gap-through)
    actual_entry_underlying: float = 0.0

    # ── Current State (updated each bar) ────────────────────────────
    current_price: float = 0.0      # Current option price
    unrealized_pnl: float = 0.0     # Unrealized P&L in dollars
    unrealized_pct: float = 0.0     # Unrealized P&L as percentage
    underlying_price: float = 0.0   # Current underlying price
    dte: int = 0                    # Days to expiration
    bars_held: int = 0              # Bars since entry

    # ── Target Levels ───────────────────────────────────────────────
    target_1x: float = 0.0          # 1.0x R:R target (for partial exits & 1H)
    original_target: float = 0.0    # Original target before adjustment

    # ── Trailing Stop State ─────────────────────────────────────────
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    high_water_mark: float = 0.0

    # ATR-based trailing for 3-2 patterns (EQUITY-52)
    atr_at_detection: float = 0.0
    use_atr_trailing: bool = False
    atr_trail_distance: float = 0.0
    atr_activation_level: float = 0.0

    # ── Partial Exit State ──────────────────────────────────────────
    partial_exit_done: bool = False
    contracts_remaining: int = 0

    # ── Pattern Invalidation (EQUITY-44/48) ─────────────────────────
    entry_bar_type: str = ""         # '2U', '2D', or '3'
    entry_bar_high: float = 0.0     # Setup bar high
    entry_bar_low: float = 0.0      # Setup bar low
    intrabar_high: float = 0.0      # Highest price since entry
    intrabar_low: float = float('inf')  # Lowest price since entry

    # ── Position Status ─────────────────────────────────────────────
    is_active: bool = True
    exit_reason: Optional[ExitReason] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_underlying_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    # ── Risk Metrics ────────────────────────────────────────────────
    risk_per_contract: float = 0.0  # Dollar risk per contract at entry
    cost_basis: float = 0.0         # Total cost (entry_price * contracts * 100)

    # Strike selection info
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'C' or 'P'

    # ── Metadata ────────────────────────────────────────────────────
    entry_bar_index: int = 0        # Index into bar array at entry

    def __post_init__(self):
        """Initialize computed fields."""
        if self.contracts_remaining == 0:
            self.contracts_remaining = self.contracts
        if self.cost_basis == 0.0 and self.entry_price > 0:
            self.cost_basis = self.entry_price * self.contracts * 100

    @property
    def is_bullish(self) -> bool:
        """Whether this is a bullish (CALL) position."""
        return self.direction.upper() in ('CALL', 'BULL', 'UP')

    def update_underlying(self, price: float) -> None:
        """
        Update underlying price and intrabar extremes.

        Called each bar (or intrabar price point) during simulation.
        """
        self.underlying_price = price
        if price > self.intrabar_high:
            self.intrabar_high = price
        if price < self.intrabar_low:
            self.intrabar_low = price

    def update_option_price(self, price: float) -> None:
        """
        Update current option price and unrealized P&L.

        Args:
            price: Current option price (mid or bid depending on config)
        """
        self.current_price = price
        if self.entry_price > 0 and self.contracts > 0:
            self.unrealized_pnl = (price - self.entry_price) * self.contracts_remaining * 100
            self.unrealized_pct = (price - self.entry_price) / self.entry_price

    def close(
        self,
        reason: ExitReason,
        exit_time: datetime,
        exit_option_price: float,
        exit_underlying_price: float,
        contracts_closed: Optional[int] = None,
    ) -> float:
        """
        Close position (fully or partially).

        Args:
            reason: Exit reason
            exit_time: When position was closed
            exit_option_price: Option price at exit (bid)
            exit_underlying_price: Underlying price at exit
            contracts_closed: Number of contracts to close (None = all)

        Returns:
            Realized P&L in dollars for the closed portion
        """
        if contracts_closed is None:
            contracts_closed = self.contracts_remaining

        pnl = (exit_option_price - self.entry_price) * contracts_closed * 100

        if contracts_closed >= self.contracts_remaining:
            # Full close
            self.is_active = False
            self.exit_reason = reason
            self.exit_time = exit_time
            self.exit_price = exit_option_price
            self.exit_underlying_price = exit_underlying_price
            self.realized_pnl = pnl
            self.contracts_remaining = 0
        else:
            # Partial close
            self.contracts_remaining -= contracts_closed
            self.partial_exit_done = True
            # Accumulate realized P&L from partials
            if self.realized_pnl is None:
                self.realized_pnl = pnl
            else:
                self.realized_pnl += pnl

        return pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal_key': self.signal_key,
            'osi_symbol': self.osi_symbol,
            'direction': self.direction,
            'pattern_type': self.pattern_type,
            'timeframe': self.timeframe,
            'entry_trigger': self.entry_trigger,
            'target_price': self.target_price,
            'stop_price': self.stop_price,
            'entry_price': self.entry_price,
            'contracts': self.contracts,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'expiration': self.expiration,
            'actual_entry_underlying': self.actual_entry_underlying,
            'target_1x': self.target_1x,
            'original_target': self.original_target,
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            'high_water_mark': self.high_water_mark,
            'atr_at_detection': self.atr_at_detection,
            'use_atr_trailing': self.use_atr_trailing,
            'partial_exit_done': self.partial_exit_done,
            'contracts_remaining': self.contracts_remaining,
            'entry_bar_type': self.entry_bar_type,
            'entry_bar_high': self.entry_bar_high,
            'entry_bar_low': self.entry_bar_low,
            'is_active': self.is_active,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'realized_pnl': self.realized_pnl,
            'bars_held': self.bars_held,
            'cost_basis': self.cost_basis,
            'strike': self.strike,
            'option_type': self.option_type,
        }
