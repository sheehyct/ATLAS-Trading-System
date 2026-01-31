"""
Paper trading simulation system for crypto derivatives.

Provides comprehensive paper trading since Coinbase doesn't offer native paper trading.
Tracks simulated positions, orders, P&L, and trade history with realistic fills.

Session Jan 24, 2026: Enhanced with Coinbase CFM fee modeling:
- Taker fee: 0.07% + $0.15/contract
- Maker fee: 0.065% + $0.15/contract  
- Slippage simulation (configurable)
- Funding rate accumulation (8-hour intervals)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# FEE CONFIGURATION - VERIFIED Jan 24, 2026
# =============================================================================

# Coinbase CFM fee structure
TAKER_FEE_RATE: float = 0.0007     # 0.07%
MAKER_FEE_RATE: float = 0.00065   # 0.065%
FIXED_FEE_PER_CONTRACT: float = 0.15  # $0.15 per contract

# Default slippage (0.05% = 5 bps)
DEFAULT_SLIPPAGE_RATE: float = 0.0005

# Funding rate (annualized, typical bull market)
ANNUAL_FUNDING_RATE: float = 0.10  # 10% APR
FUNDING_PERIODS_PER_DAY: int = 3   # Every 8 hours


def calculate_trade_fee(
    notional: float,
    num_contracts: int = 1,
    is_maker: bool = False,
) -> float:
    """
    Calculate Coinbase CFM trade fee.
    
    Formula: (Notional × Rate) + (Fixed × Contracts)
    """
    rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
    return (notional * rate) + (FIXED_FEE_PER_CONTRACT * num_contracts)


def calculate_slippage(
    price: float,
    side: str,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> float:
    """
    Calculate slippage-adjusted fill price.
    
    Buys get filled higher, sells get filled lower.
    """
    if side == "BUY":
        return price * (1 + slippage_rate)
    else:
        return price * (1 - slippage_rate)


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
    # Session EQUITY-91: Strategy attribution for separate P/L tracking
    strategy: str = "strat"  # "strat" or "statarb"
    # Position monitoring fields (Session CRYPTO-4)
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    timeframe: Optional[str] = None
    pattern_type: Optional[str] = None
    exit_reason: Optional[str] = None  # 'STOP', 'TARGET', 'MANUAL'
    tfc_score: Optional[int] = None  # Timeframe Continuity score (0-4)
    risk_multiplier: float = 1.0
    priority_rank: int = 0  # Session EQUITY-40: Priority for trade queueing
    # Session EQUITY-34: Margin tracking
    margin_reserved: float = 0.0  # Margin reserved for this position
    # Session EQUITY-67: Pattern invalidation tracking (ported from equity)
    entry_bar_type: str = ""  # '2U', '2D', or '3' - entry bar classification
    entry_bar_high: float = 0.0  # Setup bar high for Type 3 detection
    entry_bar_low: float = 0.0  # Setup bar low for Type 3 detection
    intrabar_high: float = 0.0  # Highest price since entry
    intrabar_low: float = float("inf")  # Lowest price since entry
    # Session EQUITY-99: Leverage tier tracking for deadline enforcement
    leverage_tier: str = "swing"  # "intraday" (10x, 4PM deadline) or "swing" (4x, no deadline)
    # Session Jan 24, 2026: Fee and cost tracking
    entry_fee: float = 0.0  # Fee paid on entry
    exit_fee: float = 0.0  # Fee paid on exit
    entry_slippage: float = 0.0  # Slippage cost on entry
    exit_slippage: float = 0.0  # Slippage cost on exit
    accumulated_funding: float = 0.0  # Funding payments (positive = paid, negative = received)
    gross_pnl: Optional[float] = None  # P&L before fees/costs
    total_costs: float = 0.0  # Total fees + funding

    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """
        Close the trade and calculate P&L including fees and costs.

        Session Jan 24, 2026: Now calculates gross P&L, exit fees, and net P&L.

        Args:
            exit_price: Price at which trade was closed
            exit_time: Time of exit (defaults to now)
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.utcnow()
        self.status = "CLOSED"

        # Calculate gross P&L (before fees)
        notional = self.entry_price * self.quantity
        if self.side == "BUY":
            self.gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.quantity

        # Calculate exit fee
        exit_notional = exit_price * self.quantity
        self.exit_fee = calculate_trade_fee(exit_notional, num_contracts=1, is_maker=False)

        # Calculate total costs
        self.total_costs = self.entry_fee + self.exit_fee + self.accumulated_funding

        # Net P&L (after all costs)
        self.pnl = self.gross_pnl - self.total_costs
        self.pnl_percent = (self.pnl / notional) * 100 if notional > 0 else 0

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
            "strategy": self.strategy,  # Session EQUITY-91
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "timeframe": self.timeframe,
            "pattern_type": self.pattern_type,
            "exit_reason": self.exit_reason,
            "tfc_score": self.tfc_score,
            "risk_multiplier": self.risk_multiplier,
            "priority_rank": self.priority_rank,
            "margin_reserved": self.margin_reserved,
            "entry_bar_type": self.entry_bar_type,
            "entry_bar_high": self.entry_bar_high,
            "entry_bar_low": self.entry_bar_low,
            "intrabar_high": self.intrabar_high,
            "intrabar_low": self.intrabar_low if self.intrabar_low != float("inf") else 0.0,
            # Session Jan 24, 2026: Fee and cost tracking
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "entry_slippage": self.entry_slippage,
            "exit_slippage": self.exit_slippage,
            "accumulated_funding": self.accumulated_funding,
            "gross_pnl": self.gross_pnl,
            "total_costs": self.total_costs,
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
            strategy=data.get("strategy", "strat"),  # Session EQUITY-91
            stop_price=data.get("stop_price"),
            target_price=data.get("target_price"),
            timeframe=data.get("timeframe"),
            pattern_type=data.get("pattern_type"),
            exit_reason=data.get("exit_reason"),
            tfc_score=data.get("tfc_score"),
            risk_multiplier=data.get("risk_multiplier", 1.0),
            priority_rank=data.get("priority_rank", 0),
            margin_reserved=data.get("margin_reserved", 0.0),
            entry_bar_type=data.get("entry_bar_type", ""),
            entry_bar_high=data.get("entry_bar_high", 0.0),
            entry_bar_low=data.get("entry_bar_low", 0.0),
            intrabar_high=data.get("intrabar_high", 0.0),
            intrabar_low=data.get("intrabar_low") if data.get("intrabar_low", 0.0) > 0 else float("inf"),
            # Session Jan 24, 2026: Fee and cost tracking
            entry_fee=data.get("entry_fee", 0.0),
            exit_fee=data.get("exit_fee", 0.0),
            entry_slippage=data.get("entry_slippage", 0.0),
            exit_slippage=data.get("exit_slippage", 0.0),
            accumulated_funding=data.get("accumulated_funding", 0.0),
            gross_pnl=data.get("gross_pnl"),
            total_costs=data.get("total_costs", 0.0),
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
        risk_multiplier: float = 1.0,
        priority_rank: int = 0,
        leverage: float = 4.0,
        # Session EQUITY-67: Pattern invalidation tracking
        entry_bar_type: str = "",
        entry_bar_high: float = 0.0,
        entry_bar_low: float = 0.0,
        # Session Jan 24, 2026: Realistic execution modeling
        apply_slippage: bool = True,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        # Session EQUITY-91: Strategy attribution
        strategy: str = "strat",
        # Session EQUITY-99: Leverage tier for deadline enforcement
        leverage_tier: str = "swing",
    ) -> Optional[SimulatedTrade]:
        """
        Open a new simulated trade with margin reservation, fees, and slippage.

        Session Jan 24, 2026: Enhanced with realistic execution modeling:
        - Slippage applied to entry price (buys fill higher, sells fill lower)
        - Entry fees calculated and tracked
        - Fee deducted from account balance immediately

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            side: 'BUY' or 'SELL'
            quantity: Position size in base currency
            entry_price: Entry price (before slippage)
            entry_time: Entry timestamp (defaults to now)
            stop_price: Stop loss price for position monitoring
            target_price: Take profit price for position monitoring
            timeframe: Signal timeframe (e.g., '1d', '4h')
            pattern_type: STRAT pattern (e.g., '3-2U', '2D-1-2U')
            tfc_score: Timeframe Continuity score (0-4)
            priority_rank: Priority rank for trade queueing
            leverage: Leverage used for margin calculation (default: 4x swing)
            apply_slippage: Whether to apply slippage to entry (default: True)
            slippage_rate: Slippage rate to apply (default: 0.05%)
            strategy: Strategy name for P/L attribution ("strat" or "statarb")

        Returns:
            SimulatedTrade object if opened, None if insufficient margin
        """
        # Session EQUITY-40: Apply risk_multiplier to adjust quantity
        adjusted_quantity = quantity * max(0.0, risk_multiplier)
        if adjusted_quantity <= 0:
            logger.info(
                "Skipping trade: risk_multiplier %.2f reduced quantity to 0 for %s",
                risk_multiplier,
                symbol
            )
            return None
        quantity = adjusted_quantity

        # Session Jan 24, 2026: Apply slippage to get actual fill price
        if apply_slippage:
            fill_price = calculate_slippage(entry_price, side, slippage_rate)
            entry_slippage_cost = abs(fill_price - entry_price) * quantity
        else:
            fill_price = entry_price
            entry_slippage_cost = 0.0

        # Calculate entry fee
        entry_notional = fill_price * quantity
        entry_fee = calculate_trade_fee(entry_notional, num_contracts=1, is_maker=False)

        # Calculate margin required for this position
        effective_leverage = max(1.0, leverage)
        margin_required = entry_notional / effective_leverage

        # Check available balance (margin + entry fee)
        total_required = margin_required + entry_fee
        available = self.get_available_balance()
        if total_required > available:
            logger.warning(
                "Insufficient funds: need $%.2f (margin $%.2f + fee $%.2f), have $%.2f",
                total_required,
                margin_required,
                entry_fee,
                available
            )
            return None

        trade = SimulatedTrade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,  # Use slippage-adjusted price
            entry_time=entry_time or datetime.utcnow(),
            stop_price=stop_price,
            target_price=target_price,
            timeframe=timeframe,
            pattern_type=pattern_type,
            tfc_score=tfc_score,
            risk_multiplier=risk_multiplier,
            priority_rank=priority_rank,
            margin_reserved=margin_required,
            strategy=strategy,  # Session EQUITY-91
            leverage_tier=leverage_tier,  # Session EQUITY-99
            entry_bar_type=entry_bar_type,
            entry_bar_high=entry_bar_high,
            entry_bar_low=entry_bar_low,
            intrabar_high=fill_price,
            intrabar_low=fill_price,
            # Session Jan 24, 2026: Track costs
            entry_fee=entry_fee,
            entry_slippage=entry_slippage_cost,
        )

        # Reserve margin and deduct entry fee
        self.account.reserved_margin += margin_required
        self.account.current_balance -= entry_fee  # Fee deducted immediately
        self.account.open_trades.append(trade)

        logger.info(
            "Opened trade: %s %s %s @ %.4f (slip: %.4f, fee: $%.2f, margin: $%.2f)",
            trade.trade_id,
            side,
            symbol,
            fill_price,
            entry_slippage_cost,
            entry_fee,
            margin_required,
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
        apply_slippage: bool = True,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ) -> Optional[SimulatedTrade]:
        """
        Close an open trade and release margin.

        Session Jan 24, 2026: Enhanced with exit slippage and fee tracking.
        Exit fee is deducted from balance along with P&L.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price (before slippage)
            exit_time: Exit timestamp (defaults to now)
            apply_slippage: Whether to apply slippage to exit
            slippage_rate: Slippage rate to apply

        Returns:
            Closed SimulatedTrade or None if not found
        """
        trade = self._find_open_trade(trade_id)
        if not trade:
            logger.warning("Trade not found: %s", trade_id)
            return None

        # Apply slippage to exit (opposite direction of entry)
        # For a BUY trade closing, we're selling - so price slips DOWN
        # For a SELL trade closing, we're buying back - so price slips UP
        if apply_slippage:
            close_side = "SELL" if trade.side == "BUY" else "BUY"
            fill_price = calculate_slippage(exit_price, close_side, slippage_rate)
            trade.exit_slippage = abs(fill_price - exit_price) * trade.quantity
        else:
            fill_price = exit_price
            trade.exit_slippage = 0.0

        # Calculate accumulated funding before closing
        self._accrue_funding_for_trade(trade, exit_time or datetime.utcnow())

        # Close the trade (calculates P&L internally)
        trade.close(fill_price, exit_time)

        # Move to closed trades
        self.account.open_trades.remove(trade)
        self.account.closed_trades.append(trade)

        # Release margin
        self.account.reserved_margin -= trade.margin_reserved
        self.account.reserved_margin = max(0, self.account.reserved_margin)

        # Update account: Net P&L already accounts for exit fee
        # But exit fee was calculated in close(), so we just apply net P&L
        self.account.realized_pnl += trade.pnl or 0
        self.account.current_balance += (trade.gross_pnl or 0) - trade.exit_fee - trade.accumulated_funding

        logger.info(
            "Closed %s: Gross $%.2f, Fees $%.2f, Funding $%.2f, Net $%.2f (%.1f%%)",
            trade_id,
            trade.gross_pnl or 0,
            trade.entry_fee + trade.exit_fee,
            trade.accumulated_funding,
            trade.pnl or 0,
            trade.pnl_percent or 0,
        )

        self._save_state()
        return trade

    def _accrue_funding_for_trade(self, trade: SimulatedTrade, current_time: datetime) -> None:
        """
        Calculate and accrue funding rate charges for a trade.

        Funding is charged every 8 hours. Longs typically pay shorts in bull markets.

        Args:
            trade: Trade to calculate funding for
            current_time: Current timestamp
        """
        if trade.entry_time is None:
            return

        # Calculate hold duration in days
        hold_duration = current_time - trade.entry_time
        hold_days = hold_duration.total_seconds() / 86400

        # Calculate funding periods (every 8 hours = 3 per day)
        funding_periods = int(hold_days * FUNDING_PERIODS_PER_DAY)

        if funding_periods <= 0:
            return

        # Calculate funding cost
        # Rate per period = annual_rate / (365 * 3)
        rate_per_period = ANNUAL_FUNDING_RATE / (365 * FUNDING_PERIODS_PER_DAY)
        notional = trade.entry_price * trade.quantity

        # Longs pay funding in bull market (positive rate)
        # Shorts receive funding
        if trade.side == "BUY":
            funding_cost = notional * rate_per_period * funding_periods
        else:
            funding_cost = -notional * rate_per_period * funding_periods  # Shorts receive

        trade.accumulated_funding = funding_cost

    def accrue_funding_all_positions(self) -> float:
        """
        Accrue funding charges for all open positions.

        Call this periodically (e.g., every 8 hours) to track funding costs.

        Returns:
            Total funding accrued this call
        """
        total_funding = 0.0
        current_time = datetime.utcnow()

        for trade in self.account.open_trades:
            old_funding = trade.accumulated_funding
            self._accrue_funding_for_trade(trade, current_time)
            new_funding = trade.accumulated_funding - old_funding
            total_funding += new_funding

        if total_funding != 0:
            logger.info("Accrued funding: $%.4f across %d positions",
                       total_funding, len(self.account.open_trades))
            self._save_state()

        return total_funding

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

        Session EQUITY-34: Added reserved_margin and available_balance.

        Returns:
            Dict with account summary
        """
        return {
            "account_name": self.account_name,
            "starting_balance": self.account.starting_balance,
            "current_balance": self.account.current_balance,
            "reserved_margin": self.account.reserved_margin,
            "available_balance": self.get_available_balance(),
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

    # =========================================================================
    # STRATEGY ATTRIBUTION (Session EQUITY-91)
    # =========================================================================

    def get_trades_by_strategy(self, strategy: str) -> List[SimulatedTrade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy: Strategy name ("strat" or "statarb")

        Returns:
            List of trades for that strategy (open and closed)
        """
        open_trades = [t for t in self.account.open_trades if t.strategy == strategy]
        closed_trades = [t for t in self.account.closed_trades if t.strategy == strategy]
        return open_trades + closed_trades

    def get_pnl_by_strategy(self) -> Dict[str, Dict[str, float]]:
        """
        Get P/L breakdown by strategy.

        Returns:
            Dictionary with strategy -> {gross, fees, funding, net, trades}

        Example:
            {
                "strat": {"gross": 150.0, "fees": -12.50, "funding": -2.0, "net": 135.50, "trades": 5},
                "statarb": {"gross": 45.0, "fees": -8.20, "funding": -1.5, "net": 35.30, "trades": 2},
                "combined": {"gross": 195.0, "fees": -20.70, "funding": -3.5, "net": 170.80, "trades": 7}
            }
        """
        strategies = set()
        for t in self.account.closed_trades:
            strategies.add(t.strategy)

        result: Dict[str, Dict[str, float]] = {}

        for strategy in strategies:
            trades = [t for t in self.account.closed_trades if t.strategy == strategy]
            gross = sum(t.gross_pnl or 0 for t in trades)
            fees = sum(t.entry_fee + t.exit_fee for t in trades)
            funding = sum(t.accumulated_funding for t in trades)
            net = sum(t.pnl or 0 for t in trades)

            result[strategy] = {
                "gross": gross,
                "fees": -fees,  # Negative to show as cost
                "funding": -funding,  # Negative to show as cost
                "net": net,
                "trades": len(trades),
            }

        # Add combined totals
        if result:
            result["combined"] = {
                "gross": sum(s["gross"] for s in result.values()),
                "fees": sum(s["fees"] for s in result.values()),
                "funding": sum(s["funding"] for s in result.values()),
                "net": sum(s["net"] for s in result.values()),
                "trades": sum(int(s["trades"]) for s in result.values()),
            }

        return result

    def get_performance_by_strategy(self, strategy: str) -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific strategy.

        Args:
            strategy: Strategy name ("strat" or "statarb")

        Returns:
            Dict with performance metrics for that strategy
        """
        closed = [t for t in self.account.closed_trades if t.strategy == strategy]
        if not closed:
            return {"message": f"No closed trades for strategy: {strategy}"}

        winners = [t for t in closed if (t.pnl or 0) > 0]
        losers = [t for t in closed if (t.pnl or 0) < 0]

        total_pnl = sum(t.pnl or 0 for t in closed)
        gross_profit = sum(t.pnl or 0 for t in winners)
        gross_loss = abs(sum(t.pnl or 0 for t in losers))

        avg_win = gross_profit / len(winners) if winners else 0
        avg_loss = gross_loss / len(losers) if losers else 0

        return {
            "strategy": strategy,
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
