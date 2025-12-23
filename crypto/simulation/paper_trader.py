"""
Paper trading simulation system for crypto derivatives.

Provides comprehensive paper trading since Coinbase doesn't offer native paper trading.
Tracks simulated positions, orders, P&L, and trade history with realistic fills.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    """Represents a completed simulated trade."""

    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED
    # Position monitoring fields (Session CRYPTO-4)
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    timeframe: Optional[str] = None
    pattern_type: Optional[str] = None
    exit_reason: Optional[str] = None  # 'STOP', 'TARGET', 'MANUAL'
    tfc_score: Optional[int] = None  # Timeframe Continuity score (0-4)
    # Session EQUITY-34: Margin tracking
    margin_reserved: float = 0.0  # Margin reserved for this position

    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """
        Close the trade and calculate P&L.

        Args:
            exit_price: Price at which trade was closed
            exit_time: Time of exit (defaults to now)
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.utcnow()
        self.status = "CLOSED"

        # Calculate P&L
        if self.side == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity

        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "status": self.status,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "timeframe": self.timeframe,
            "pattern_type": self.pattern_type,
            "exit_reason": self.exit_reason,
            "tfc_score": self.tfc_score,
            "margin_reserved": self.margin_reserved,  # Session EQUITY-34
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedTrade":
        """Create from dictionary."""
        trade = cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            side=data["side"],
            quantity=data["quantity"],
            entry_price=data["entry_price"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            status=data.get("status", "OPEN"),
            stop_price=data.get("stop_price"),
            target_price=data.get("target_price"),
            timeframe=data.get("timeframe"),
            pattern_type=data.get("pattern_type"),
            exit_reason=data.get("exit_reason"),
            tfc_score=data.get("tfc_score"),
            margin_reserved=data.get("margin_reserved", 0.0),  # Session EQUITY-34
        )
        if data.get("exit_price"):
            trade.exit_price = data["exit_price"]
            trade.exit_time = datetime.fromisoformat(data["exit_time"])
            trade.pnl = data.get("pnl")
            trade.pnl_percent = data.get("pnl_percent")
        return trade


@dataclass
class PaperTradingAccount:
    """Simulated trading account state."""

    starting_balance: float = 1000.0
    current_balance: float = 1000.0
    realized_pnl: float = 0.0
    open_trades: List[SimulatedTrade] = field(default_factory=list)
    closed_trades: List[SimulatedTrade] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Session EQUITY-34: Track margin in use for open positions
    reserved_margin: float = 0.0


class PaperTrader:
    """
    Paper trading simulator for crypto derivatives.

    Features:
    - Simulated order execution with realistic fills
    - Position tracking and P&L calculation
    - Trade history with FIFO matching
    - Persistence to JSON for session continuity
    - Performance metrics calculation
    """

    def __init__(
        self,
        starting_balance: float = 1000.0,
        data_dir: Optional[Path] = None,
        account_name: str = "default",
    ) -> None:
        """
        Initialize paper trader.

        Args:
            starting_balance: Starting account balance in USD
            data_dir: Directory for persisting state (optional)
            account_name: Name for this paper trading account
        """
        self.account_name = account_name
        self.data_dir = data_dir or Path("crypto/simulation/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.account = PaperTradingAccount(
            starting_balance=starting_balance,
            current_balance=starting_balance,
        )

        self._trade_counter = 0

        # Try to load existing state
        self._load_state()

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"SIM-{self.account_name}-{self._trade_counter:05d}"

    # =========================================================================
    # TRADING OPERATIONS
    # =========================================================================

    def open_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_time: Optional[datetime] = None,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        timeframe: Optional[str] = None,
        pattern_type: Optional[str] = None,
        tfc_score: Optional[int] = None,
        leverage: float = 4.0,
    ) -> Optional[SimulatedTrade]:
        """
        Open a new simulated trade with margin reservation.

        Session EQUITY-34: Now tracks margin requirements and prevents
        over-leveraging by checking available balance before opening.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            quantity: Position size in base currency
            entry_price: Entry price
            entry_time: Entry timestamp (defaults to now)
            stop_price: Stop loss price for position monitoring
            target_price: Take profit price for position monitoring
            timeframe: Signal timeframe (e.g., '1d', '4h')
            pattern_type: STRAT pattern (e.g., '3-2U', '2D-1-2U')
            tfc_score: Timeframe Continuity score (0-4)
            leverage: Leverage used for margin calculation (default: 4x swing)

        Returns:
            SimulatedTrade object if opened, None if insufficient margin
        """
        # Calculate margin required for this position
        # Session EQUITY-34: Use max(1.0, leverage) to avoid margin > notional when leverage < 1
        position_notional = quantity * entry_price
        effective_leverage = max(1.0, leverage)
        margin_required = position_notional / effective_leverage

        # Check available balance (Session EQUITY-34)
        available = self.get_available_balance()
        if margin_required > available:
            logger.warning(
                "Insufficient margin for trade: need $%.2f, have $%.2f available",
                margin_required,
                available
            )
            return None

        trade = SimulatedTrade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time or datetime.utcnow(),
            stop_price=stop_price,
            target_price=target_price,
            timeframe=timeframe,
            pattern_type=pattern_type,
            tfc_score=tfc_score,
            margin_reserved=margin_required,
        )

        # Reserve margin (Session EQUITY-34)
        self.account.reserved_margin += margin_required
        self.account.open_trades.append(trade)

        logger.info(
            "Opened simulated trade: %s %s %s @ %.2f (margin=$%.2f, stop=%.2f, target=%.2f)",
            trade.trade_id,
            side,
            symbol,
            entry_price,
            margin_required,
            stop_price or 0,
            target_price or 0,
        )

        self._save_state()
        return trade

    def get_available_balance(self) -> float:
        """
        Get available balance after margin reservations.

        Session EQUITY-34: Returns current_balance minus reserved_margin.

        Returns:
            Available balance in USD
        """
        return max(0, self.account.current_balance - self.account.reserved_margin)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[SimulatedTrade]:
        """
        Close an open trade and release margin.

        Session EQUITY-34: Now releases reserved margin when trade closes.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_time: Exit timestamp (defaults to now)

        Returns:
            Closed SimulatedTrade or None if not found
        """
        trade = self._find_open_trade(trade_id)
        if not trade:
            logger.warning("Trade not found: %s", trade_id)
            return None

        trade.close(exit_price, exit_time)

        # Move to closed trades
        self.account.open_trades.remove(trade)
        self.account.closed_trades.append(trade)

        # Release margin (Session EQUITY-34)
        self.account.reserved_margin -= trade.margin_reserved
        self.account.reserved_margin = max(0, self.account.reserved_margin)

        # Update account
        self.account.realized_pnl += trade.pnl or 0
        self.account.current_balance += trade.pnl or 0

        logger.info(
            "Closed trade %s: P&L = $%.2f (%.2f%%), margin released: $%.2f",
            trade_id,
            trade.pnl or 0,
            trade.pnl_percent or 0,
            trade.margin_reserved,
        )

        self._save_state()
        return trade

    def close_all_trades(
        self,
        symbol: Optional[str] = None,
        exit_price: Optional[float] = None,
    ) -> List[SimulatedTrade]:
        """
        Close all open trades (optionally filtered by symbol).

        Args:
            symbol: Only close trades for this symbol (optional)
            exit_price: Exit price (required if closing)

        Returns:
            List of closed trades
        """
        if not exit_price:
            logger.warning("Cannot close trades without exit price")
            return []

        trades_to_close = [
            t
            for t in self.account.open_trades
            if symbol is None or t.symbol == symbol
        ]

        closed = []
        for trade in trades_to_close:
            result = self.close_trade(trade.trade_id, exit_price)
            if result:
                closed.append(result)

        return closed

    def _find_open_trade(self, trade_id: str) -> Optional[SimulatedTrade]:
        """Find an open trade by ID."""
        for trade in self.account.open_trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated open position for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position dict or None
        """
        trades = [t for t in self.account.open_trades if t.symbol == symbol]
        if not trades:
            return None

        # Aggregate position
        total_qty = sum(t.quantity if t.side == "BUY" else -t.quantity for t in trades)
        if total_qty == 0:
            return None

        side = "BUY" if total_qty > 0 else "SELL"
        avg_price = sum(t.entry_price * t.quantity for t in trades) / sum(
            t.quantity for t in trades
        )

        return {
            "symbol": symbol,
            "side": side,
            "quantity": abs(total_qty),
            "avg_entry_price": avg_price,
            "trade_count": len(trades),
        }

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary with all metrics.

        Returns:
            Dict with account summary
        """
        return {
            "account_name": self.account_name,
            "starting_balance": self.account.starting_balance,
            "current_balance": self.account.current_balance,
            "realized_pnl": self.account.realized_pnl,
            "return_percent": (
                (self.account.current_balance - self.account.starting_balance)
                / self.account.starting_balance
                * 100
            ),
            "open_trades": len(self.account.open_trades),
            "closed_trades": len(self.account.closed_trades),
            "created_at": self.account.created_at.isoformat(),
        }

    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dict with performance metrics
        """
        closed = self.account.closed_trades
        if not closed:
            return {"message": "No closed trades yet"}

        winners = [t for t in closed if (t.pnl or 0) > 0]
        losers = [t for t in closed if (t.pnl or 0) < 0]

        total_pnl = sum(t.pnl or 0 for t in closed)
        gross_profit = sum(t.pnl or 0 for t in winners)
        gross_loss = abs(sum(t.pnl or 0 for t in losers))

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        return {
            "total_trades": len(closed),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(closed) * 100 if closed else 0,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": (
                (len(winners) / len(closed) * avg_win)
                - (len(losers) / len(closed) * avg_loss)
                if closed
                else 0
            ),
            "largest_win": max((t.pnl or 0 for t in closed), default=0),
            "largest_loss": min((t.pnl or 0 for t in closed), default=0),
        }

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent trade history.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts (most recent first)
        """
        all_trades = self.account.closed_trades + self.account.open_trades
        all_trades.sort(key=lambda t: t.entry_time, reverse=True)
        return [t.to_dict() for t in all_trades[:limit]]

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _get_state_file(self) -> Path:
        """Get path to state file."""
        return self.data_dir / f"paper_trader_{self.account_name}.json"

    def _save_state(self) -> None:
        """Save current state to file."""
        state = {
            "account_name": self.account_name,
            "starting_balance": self.account.starting_balance,
            "current_balance": self.account.current_balance,
            "realized_pnl": self.account.realized_pnl,
            "reserved_margin": self.account.reserved_margin,  # Session EQUITY-34
            "created_at": self.account.created_at.isoformat(),
            "trade_counter": self._trade_counter,
            "open_trades": [t.to_dict() for t in self.account.open_trades],
            "closed_trades": [t.to_dict() for t in self.account.closed_trades],
        }

        try:
            with open(self._get_state_file(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error("Failed to save paper trading state: %s", e)

    def _load_state(self) -> bool:
        """
        Load state from file.

        Session EQUITY-34: Now loads reserved_margin and recalculates
        from open trades if not present for backward compatibility.

        Returns:
            True if state was loaded, False otherwise
        """
        state_file = self._get_state_file()
        if not state_file.exists():
            return False

        try:
            with open(state_file) as f:
                state = json.load(f)

            self.account.starting_balance = state["starting_balance"]
            self.account.current_balance = state["current_balance"]
            self.account.realized_pnl = state["realized_pnl"]
            self.account.created_at = datetime.fromisoformat(state["created_at"])
            self._trade_counter = state.get("trade_counter", 0)

            self.account.open_trades = [
                SimulatedTrade.from_dict(t) for t in state.get("open_trades", [])
            ]
            self.account.closed_trades = [
                SimulatedTrade.from_dict(t) for t in state.get("closed_trades", [])
            ]

            # Session EQUITY-34: Load or recalculate reserved_margin
            if "reserved_margin" in state:
                self.account.reserved_margin = state["reserved_margin"]
            else:
                # Backward compat: recalculate from open trades
                self.account.reserved_margin = sum(
                    t.margin_reserved for t in self.account.open_trades
                )

            logger.info(
                "Loaded paper trading state: %d open, %d closed trades, $%.2f reserved",
                len(self.account.open_trades),
                len(self.account.closed_trades),
                self.account.reserved_margin,
            )
            return True

        except Exception as e:
            logger.error("Failed to load paper trading state: %s", e)
            return False

    def reset(self, starting_balance: Optional[float] = None) -> None:
        """
        Reset account to initial state.

        Args:
            starting_balance: New starting balance (optional)
        """
        balance = starting_balance or self.account.starting_balance
        self.account = PaperTradingAccount(
            starting_balance=balance,
            current_balance=balance,
        )
        self._trade_counter = 0
        self._save_state()
        logger.info("Paper trading account reset with balance: $%.2f", balance)
