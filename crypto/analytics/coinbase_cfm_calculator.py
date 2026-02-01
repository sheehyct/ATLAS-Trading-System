"""
Coinbase CFM P/L Calculator.

Calculates realized and unrealized P/L from actual Coinbase CFM fills using FIFO matching.

Supports:
- Crypto perpetuals: BIP, ETP, SOP, ADP, XRP
- Commodity futures: SLRH, GOLJ

Fee Structure (Coinbase CFM - Verified Jan 24, 2026):
- Taker fee: 0.07% + $0.15/contract
- Maker fee: 0.065% + $0.15/contract

Leverage Tiers:
- Intraday (6PM-4PM ET): Higher leverage available
- Overnight/Swing (4PM-6PM ET + weekends): Lower leverage required
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Eastern timezone for leverage window calculations
ET = ZoneInfo("America/New_York")


# =============================================================================
# FEE CONFIGURATION - VERIFIED Jan 24, 2026
# =============================================================================

TAKER_FEE_RATE: float = 0.0007     # 0.07%
MAKER_FEE_RATE: float = 0.00065   # 0.065%
FIXED_FEE_PER_CONTRACT: float = 0.15  # $0.15 per contract

# Product classification
CRYPTO_PERPS = ["BIP", "ETP", "SOP", "ADP", "XRP"]
COMMODITY_FUTURES = ["SLR", "SLRH", "GOL", "GOLJ"]  # Include base and dated symbols

# Symbol to asset mapping
CFM_SYMBOL_MAP = {
    "BIP": "Bitcoin",
    "ETP": "Ethereum",
    "SOP": "Solana",
    "ADP": "Cardano",
    "XRP": "XRP",
    "SLR": "Silver",
    "SLRH": "Silver",
    "GOL": "Gold",
    "GOLJ": "Gold",
}

# =============================================================================
# CONTRACT MULTIPLIERS - CRITICAL FOR P/L CALCULATION
# =============================================================================
# CFM "nano" contracts represent a fraction of the underlying asset.
# P/L = price_diff * num_contracts * contract_size
#
# VERIFIED from crypto/statarb/RESEARCH_HANDOFF.md (January 24, 2026)
# and crypto/config.py
#
# Sources:
# - https://www.metrotrade.com/what-are-nano-bitcoin-futures/
# - https://www.marketswiki.com/wiki/Nano_Ether_Perpetual_Futures
# - https://info.tradovate.com/coinbase-derivatives-nano-bitcoin

CFM_CONTRACT_MULTIPLIERS: Dict[str, float] = {
    # Crypto Perpetuals - VERIFIED Jan 24, 2026
    "BIP": 0.01,      # 0.01 BTC per contract (Nano Bitcoin)
    "ETP": 0.1,       # 0.1 ETH per contract (Nano Ether)
    "SOP": 5.0,       # 5.0 SOL per contract
    "ADP": 1000.0,    # 1000 ADA per contract
    "XRP": 500.0,     # 500 XRP per contract
    # Commodity Futures - VERIFIED Feb 1, 2026
    # Source: https://www.coinbase.com/blog/coinbase-derivatives-expands-futures-offering-to-include-oil-and-gold
    "SLR": 50.0,      # 50 troy ounces of silver per contract
    "SLRH": 50.0,     # Silver March expiry (H = March)
    "GOL": 1.0,       # 1 troy ounce of gold per contract
    "GOLJ": 1.0,      # Gold April expiry (J = April)
}


def get_contract_multiplier(product_id: str) -> float:
    """
    Get the contract multiplier for a CFM product.

    Args:
        product_id: CFM product ID (e.g., 'BIP-20DEC30-CDE')

    Returns:
        Contract multiplier (e.g., 0.01 for BIP = 1/100 BTC per contract)
    """
    base = extract_base_symbol(product_id)
    return CFM_CONTRACT_MULTIPLIERS.get(base, 1.0)


# =============================================================================
# LEVERAGE TIERS - VERIFIED Feb 1, 2026
# =============================================================================
# Intraday window: 6PM ET to 4PM ET next day (22 hours)
# Overnight/Swing: 4PM ET to 6PM ET (2 hours) + weekends

INTRADAY_START_HOUR = 18  # 6PM ET
INTRADAY_END_HOUR = 16    # 4PM ET

# Leverage by tier and product - VERIFIED from Coinbase CFM platform
LEVERAGE_TIERS: Dict[str, Dict[str, float]] = {
    "intraday": {
        # Crypto - 10x intraday for majors, 5x for altcoins
        "BIP": 10.0, "ETP": 10.0, "SOP": 5.0, "ADP": 5.0, "XRP": 5.0,
        # Commodities
        "SLR": 8.9, "SLRH": 8.9,
        "GOL": 19.7, "GOLJ": 19.7,
    },
    "overnight": {
        # Crypto - VERIFIED Jan 24, 2026
        "BIP": 4.1, "ETP": 4.0, "SOP": 2.7, "ADP": 3.4, "XRP": 2.6,
        # Commodities - VERIFIED Feb 1, 2026
        "SLR": 8.9, "SLRH": 8.9,      # Same as intraday
        "GOL": 19.7, "GOLJ": 19.7,    # Same as intraday (verify when market open)
    },
}

DEFAULT_LEVERAGE = 4.0  # Conservative fallback


def is_intraday_window(timestamp: datetime) -> bool:
    """
    Check if timestamp is within intraday leverage window.

    Intraday: 6PM ET to 4PM ET next day (22 hours).
    Overnight: 4PM ET to 6PM ET (2 hours).
    Weekend: Friday 4PM ET to Sunday 6PM ET = overnight tier.

    Args:
        timestamp: Trade timestamp (UTC or timezone-aware)

    Returns:
        True if intraday leverage available at this time
    """
    # Convert to ET
    if timestamp.tzinfo is None:
        # Assume UTC if naive
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    ts_et = timestamp.astimezone(ET)

    hour = ts_et.hour
    weekday = ts_et.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    # Weekend check: Friday after 4PM through Sunday before 6PM
    if weekday == 4 and hour >= INTRADAY_END_HOUR:  # Friday 4PM+
        return False
    if weekday == 5:  # Saturday
        return False
    if weekday == 6 and hour < INTRADAY_START_HOUR:  # Sunday before 6PM
        return False

    # Intraday window: hour >= 18 OR hour < 16
    return hour >= INTRADAY_START_HOUR or hour < INTRADAY_END_HOUR


def get_leverage_tier(timestamp: datetime) -> str:
    """
    Get leverage tier for a given timestamp.

    Args:
        timestamp: Trade timestamp

    Returns:
        "intraday" or "overnight"
    """
    return "intraday" if is_intraday_window(timestamp) else "overnight"


def get_leverage_for_product(product_id: str, timestamp: datetime) -> float:
    """
    Get max leverage for a product at a specific time.

    Args:
        product_id: CFM product ID (e.g., 'BIP-20DEC30-CDE')
        timestamp: Trade timestamp

    Returns:
        Maximum leverage available
    """
    base = extract_base_symbol(product_id)
    tier = get_leverage_tier(timestamp)
    return LEVERAGE_TIERS.get(tier, {}).get(base, DEFAULT_LEVERAGE)


def calculate_margin_required(notional: float, leverage: float) -> float:
    """
    Calculate margin required for a position.

    Args:
        notional: Total notional value in USD
        leverage: Leverage used

    Returns:
        Margin required in USD
    """
    if leverage <= 0:
        return notional
    return notional / leverage


def extract_base_symbol(product_id: str) -> str:
    """
    Extract base symbol from CFM product ID.

    Examples:
        'BIP-20DEC30-CDE' -> 'BIP'
        'SLRH-20MAR26-CDE' -> 'SLRH'
    """
    if not product_id:
        return ""
    return product_id.split("-")[0].upper()


def classify_product(product_id: str) -> str:
    """
    Classify product as crypto_perp or commodity_future.

    Args:
        product_id: CFM product ID (e.g., 'BIP-20DEC30-CDE')

    Returns:
        'crypto_perp' or 'commodity_future'
    """
    base = extract_base_symbol(product_id)
    if base in CRYPTO_PERPS:
        return "crypto_perp"
    elif base in COMMODITY_FUTURES:
        return "commodity_future"
    return "unknown"


def calculate_fee(
    notional: float,
    is_maker: bool = False,
    num_contracts: int = 1,
) -> float:
    """
    Calculate Coinbase CFM trade fee.

    Formula: (Notional × Rate) + (Fixed × Contracts)
    """
    rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
    return (notional * rate) + (FIXED_FEE_PER_CONTRACT * num_contracts)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CFMTransaction:
    """
    Single Coinbase CFM fill/transaction.

    Represents a single fill from the Coinbase get_fills() API.
    """
    fill_id: str
    order_id: str
    product_id: str           # e.g., 'BIP-20DEC30-CDE'
    side: str                 # 'BUY' or 'SELL'
    size: float               # Quantity
    price: float              # Fill price
    fee: float                # Fee paid
    timestamp: datetime
    is_maker: bool = False    # Maker vs taker fill
    trade_type: str = ""      # Trade type from API

    @property
    def base_symbol(self) -> str:
        """Extract base symbol (BIP, ETP, etc.)."""
        return extract_base_symbol(self.product_id)

    @property
    def product_type(self) -> str:
        """Get product type (crypto_perp or commodity_future)."""
        return classify_product(self.product_id)

    @property
    def notional(self) -> float:
        """Calculate notional value."""
        return self.size * self.price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "product_id": self.product_id,
            "base_symbol": self.base_symbol,
            "product_type": self.product_type,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "fee": self.fee,
            "notional": self.notional,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_maker": self.is_maker,
            "trade_type": self.trade_type,
        }

    @classmethod
    def from_coinbase_fill(cls, fill: Dict[str, Any]) -> "CFMTransaction":
        """
        Create CFMTransaction from Coinbase fill dict.

        Args:
            fill: Dictionary from CoinbaseClient.get_fills_live()
        """
        # Parse timestamp
        timestamp_str = fill.get("trade_time") or fill.get("sequence_timestamp")
        if isinstance(timestamp_str, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.utcnow()
        elif isinstance(timestamp_str, datetime):
            timestamp = timestamp_str
        else:
            timestamp = datetime.utcnow()

        # Parse fee
        commission = fill.get("commission", "0")
        if isinstance(commission, str):
            fee = float(commission) if commission else 0.0
        else:
            fee = float(commission or 0)

        # Determine if maker
        is_maker = fill.get("liquidity_indicator", "").upper() == "MAKER"

        return cls(
            fill_id=fill.get("entry_id") or fill.get("trade_id", ""),
            order_id=fill.get("order_id", ""),
            product_id=fill.get("product_id", ""),
            side=fill.get("side", "").upper(),
            size=float(fill.get("size", 0)),
            price=float(fill.get("price", 0)),
            fee=fee,
            timestamp=timestamp,
            is_maker=is_maker,
            trade_type=fill.get("trade_type", ""),
        )


@dataclass
class CFMLot:
    """
    Cost basis lot for FIFO matching.

    Represents an open position lot that can be matched against closing trades.
    """
    fill_id: str              # Original fill that created this lot
    product_id: str
    side: str                 # 'BUY' or 'SELL' (direction of original fill)
    quantity: float           # Original quantity
    remaining_qty: float      # Remaining unmatched quantity
    cost_basis: float         # Price including fees (per unit)
    fee: float                # Fee paid on this lot
    acquired_at: datetime

    @property
    def base_symbol(self) -> str:
        return extract_base_symbol(self.product_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "product_id": self.product_id,
            "base_symbol": self.base_symbol,
            "side": self.side,
            "quantity": self.quantity,
            "remaining_qty": self.remaining_qty,
            "cost_basis": self.cost_basis,
            "fee": self.fee,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
        }


@dataclass
class CFMRealizedPL:
    """
    Realized P/L from a closed position.

    Created when a closing trade is matched against open lots via FIFO.
    Includes leverage tier tracking for capital efficiency analysis.
    """
    product_id: str
    base_symbol: str
    open_fill_id: str         # Fill that opened the position
    close_fill_id: str        # Fill that closed the position
    quantity: float
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    gross_pnl: float          # P/L before fees
    net_pnl: float            # P/L after fees
    pnl_percent: float        # % return on notional
    hold_duration: timedelta
    entry_time: datetime
    exit_time: datetime
    side: str                 # Direction of original position
    product_type: str         # crypto_perp or commodity_future
    # Leverage tier tracking
    leverage_tier: str = ""           # "intraday" or "overnight"
    leverage_used: float = 1.0        # Actual leverage for this trade
    notional_value: float = 0.0       # Total notional in USD
    margin_required: float = 0.0      # Margin needed at entry
    roi_on_margin: float = 0.0        # % return on margin (capital efficiency)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "base_symbol": self.base_symbol,
            "asset_name": CFM_SYMBOL_MAP.get(self.base_symbol, self.base_symbol),
            "open_fill_id": self.open_fill_id,
            "close_fill_id": self.close_fill_id,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "pnl_percent": self.pnl_percent,
            "hold_duration_seconds": self.hold_duration.total_seconds(),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "side": self.side,
            "product_type": self.product_type,
            # Leverage tracking
            "leverage_tier": self.leverage_tier,
            "leverage_used": self.leverage_used,
            "notional_value": self.notional_value,
            "margin_required": self.margin_required,
            "roi_on_margin": self.roi_on_margin,
        }


@dataclass
class CFMOpenPosition:
    """
    Aggregated open position for a product.

    Combines multiple lots into a single position view.
    Includes leverage tier tracking for margin analysis.
    """
    product_id: str
    base_symbol: str
    side: str                 # NET position side
    quantity: float           # Total quantity
    avg_entry_price: float    # Weighted average entry
    total_fees: float         # Total fees paid
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    current_price: Optional[float] = None
    product_type: str = ""
    lots: List[CFMLot] = field(default_factory=list)
    # Leverage tier tracking
    leverage_tier: str = ""           # Based on current time
    leverage_available: float = 1.0   # Max leverage right now
    notional_value: float = 0.0       # Total notional in USD
    margin_required: float = 0.0      # Current margin requirement
    unrealized_roi: float = 0.0       # % return on margin

    def update_unrealized_pnl(self, current_price: float, now: Optional[datetime] = None) -> None:
        """
        Update unrealized P/L with current market price.

        Args:
            current_price: Current market price
            now: Current timestamp for leverage tier calculation (default: now)
        """
        self.current_price = current_price

        # Get contract multiplier for this product
        contract_multiplier = get_contract_multiplier(self.product_id)

        # Calculate unrealized P/L with contract multiplier
        # P/L = price_diff * contracts * contract_size
        if self.side == "BUY":
            price_diff = current_price - self.avg_entry_price
        else:
            price_diff = self.avg_entry_price - current_price

        self.unrealized_pnl = price_diff * self.quantity * contract_multiplier

        # Subtract fees for accurate unrealized P/L
        self.unrealized_pnl -= self.total_fees

        # Notional = price * contracts * contract_multiplier
        self.notional_value = current_price * self.quantity * contract_multiplier
        self.unrealized_pnl_percent = (self.unrealized_pnl / self.notional_value * 100) if self.notional_value > 0 else 0.0

        # Calculate leverage tier based on current time
        if now is None:
            now = datetime.now(timezone.utc)
        self.leverage_tier = get_leverage_tier(now)
        self.leverage_available = get_leverage_for_product(self.product_id, now)

        # Calculate margin required and ROI
        self.margin_required = calculate_margin_required(self.notional_value, self.leverage_available)
        self.unrealized_roi = (self.unrealized_pnl / self.margin_required * 100) if self.margin_required > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "base_symbol": self.base_symbol,
            "asset_name": CFM_SYMBOL_MAP.get(self.base_symbol, self.base_symbol),
            "side": self.side,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "total_fees": self.total_fees,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "product_type": self.product_type,
            "num_lots": len(self.lots),
            # Leverage tracking
            "leverage_tier": self.leverage_tier,
            "leverage_available": self.leverage_available,
            "notional_value": self.notional_value,
            "margin_required": self.margin_required,
            "unrealized_roi": self.unrealized_roi,
        }


# =============================================================================
# CALCULATOR
# =============================================================================

class CoinbaseCFMCalculator:
    """
    FIFO-based P/L calculator for Coinbase CFM derivatives.

    Processes raw fills to calculate:
    - Realized P/L from closed positions
    - Open positions with cost basis
    - Performance metrics (win rate, profit factor)
    """

    def __init__(self):
        """Initialize calculator with empty state."""
        # Open lots by product_id, keyed by (product_id, side)
        self._lots: Dict[Tuple[str, str], List[CFMLot]] = {}
        # Realized P/L records
        self._realized_pnl: List[CFMRealizedPL] = []
        # All processed transactions
        self._transactions: List[CFMTransaction] = []
        # Funding payments tracked separately
        self._funding_payments: List[Dict[str, Any]] = []

    def process_fills(self, fills: List[Dict[str, Any]]) -> None:
        """
        Process raw fills from Coinbase API.

        Uses FIFO matching to calculate realized P/L and track open positions.

        Args:
            fills: List of fill dicts from CoinbaseClient.get_fills_live()
        """
        # Convert to transactions and sort by timestamp
        transactions = [CFMTransaction.from_coinbase_fill(f) for f in fills]
        transactions.sort(key=lambda t: t.timestamp)

        self._transactions = transactions
        self._lots = {}
        self._realized_pnl = []

        for txn in transactions:
            self._process_transaction(txn)

        logger.info(
            "Processed %d fills: %d realized trades, %d open lots",
            len(transactions),
            len(self._realized_pnl),
            sum(len(lots) for lots in self._lots.values()),
        )

    def _process_transaction(self, txn: CFMTransaction) -> None:
        """Process a single transaction."""
        if not txn.product_id or not txn.side or txn.size <= 0:
            return

        # Check if this is opening or closing a position
        opposite_side = "SELL" if txn.side == "BUY" else "BUY"
        lot_key = (txn.product_id, opposite_side)

        # If there are lots on the opposite side, this closes them (FIFO)
        if lot_key in self._lots and self._lots[lot_key]:
            remaining_qty = txn.size
            remaining_fee = txn.fee

            while remaining_qty > 0 and self._lots[lot_key]:
                lot = self._lots[lot_key][0]

                if lot.remaining_qty <= remaining_qty:
                    # Close entire lot
                    close_qty = lot.remaining_qty
                    close_fee_portion = (close_qty / txn.size) * txn.fee if txn.size > 0 else remaining_fee

                    self._record_realized_pnl(lot, txn, close_qty, close_fee_portion)

                    remaining_qty -= close_qty
                    remaining_fee -= close_fee_portion
                    self._lots[lot_key].pop(0)
                else:
                    # Partial close
                    close_qty = remaining_qty
                    close_fee_portion = remaining_fee

                    self._record_realized_pnl(lot, txn, close_qty, close_fee_portion)

                    lot.remaining_qty -= close_qty
                    remaining_qty = 0
                    remaining_fee = 0

            # If there's remaining quantity, it opens a new position
            if remaining_qty > 0:
                self._add_lot(txn, remaining_qty, remaining_fee)
        else:
            # Opening a new position
            self._add_lot(txn, txn.size, txn.fee)

    def _add_lot(self, txn: CFMTransaction, qty: float, fee: float) -> None:
        """Add a new lot for an opening transaction."""
        lot_key = (txn.product_id, txn.side)

        if lot_key not in self._lots:
            self._lots[lot_key] = []

        # Calculate cost basis per unit (price + fee per unit)
        fee_per_unit = fee / qty if qty > 0 else 0
        cost_basis = txn.price + fee_per_unit

        lot = CFMLot(
            fill_id=txn.fill_id,
            product_id=txn.product_id,
            side=txn.side,
            quantity=qty,
            remaining_qty=qty,
            cost_basis=cost_basis,
            fee=fee,
            acquired_at=txn.timestamp,
        )

        self._lots[lot_key].append(lot)

    def _record_realized_pnl(
        self,
        lot: CFMLot,
        close_txn: CFMTransaction,
        qty: float,
        exit_fee: float,
    ) -> None:
        """Record realized P/L when a lot is closed."""
        # Calculate fees proportionally
        entry_fee_portion = (qty / lot.quantity) * lot.fee if lot.quantity > 0 else 0

        # Get contract multiplier for this product
        # CFM contracts represent fractions of the underlying asset
        # e.g., BIP = 0.01 BTC per contract, so 100 contracts = 1 BTC exposure
        contract_multiplier = get_contract_multiplier(lot.product_id)

        # Calculate gross P/L with contract multiplier
        # P/L = price_diff * num_contracts * contract_size
        # For LONG (BUY): profit when price goes UP
        # For SHORT (SELL): profit when price goes DOWN
        price_diff = close_txn.price - lot.cost_basis
        if lot.side == "SELL":
            # Short position: profit = entry_price - exit_price
            price_diff = lot.cost_basis - close_txn.price

        # Gross P/L = price movement * contracts * contract multiplier
        gross_pnl = price_diff * qty * contract_multiplier

        # Net P/L after fees (entry fee already in cost_basis, subtract exit fee)
        net_pnl = gross_pnl - exit_fee - entry_fee_portion

        # Calculate percentage based on notional value
        # Notional = price * contracts * contract_multiplier
        notional = lot.cost_basis * qty * contract_multiplier
        pnl_percent = (net_pnl / notional * 100) if notional > 0 else 0.0

        # Hold duration
        hold_duration = close_txn.timestamp - lot.acquired_at

        # Determine leverage tier at entry time
        leverage_tier = get_leverage_tier(lot.acquired_at)
        leverage_used = get_leverage_for_product(lot.product_id, lot.acquired_at)

        # Calculate margin required and ROI on margin
        margin_required = calculate_margin_required(notional, leverage_used)
        roi_on_margin = (net_pnl / margin_required * 100) if margin_required > 0 else 0.0

        realized = CFMRealizedPL(
            product_id=lot.product_id,
            base_symbol=lot.base_symbol,
            open_fill_id=lot.fill_id,
            close_fill_id=close_txn.fill_id,
            quantity=qty,
            entry_price=lot.cost_basis - (entry_fee_portion / qty if qty > 0 else 0),
            exit_price=close_txn.price,
            entry_fee=entry_fee_portion,
            exit_fee=exit_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_percent=pnl_percent,
            hold_duration=hold_duration,
            entry_time=lot.acquired_at,
            exit_time=close_txn.timestamp,
            side=lot.side,
            product_type=classify_product(lot.product_id),
            # Leverage tracking
            leverage_tier=leverage_tier,
            leverage_used=leverage_used,
            notional_value=notional,
            margin_required=margin_required,
            roi_on_margin=roi_on_margin,
        )

        self._realized_pnl.append(realized)

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def get_realized_pnl(self) -> List[CFMRealizedPL]:
        """Get all realized P/L records."""
        return self._realized_pnl.copy()

    def get_realized_pnl_total(self) -> Dict[str, float]:
        """
        Get total realized P/L summary.

        Returns:
            Dict with gross_pnl, total_fees, net_pnl
        """
        gross = sum(r.gross_pnl for r in self._realized_pnl)
        fees = sum(r.entry_fee + r.exit_fee for r in self._realized_pnl)
        net = sum(r.net_pnl for r in self._realized_pnl)

        return {
            "gross_pnl": gross,
            "total_fees": fees,
            "net_pnl": net,
            "trade_count": len(self._realized_pnl),
        }

    def get_open_positions(self) -> List[CFMOpenPosition]:
        """
        Get aggregated open positions.

        Returns:
            List of CFMOpenPosition for each product/side with open lots
        """
        positions = []

        for (product_id, side), lots in self._lots.items():
            if not lots:
                continue

            total_qty = sum(lot.remaining_qty for lot in lots)
            total_cost = sum(lot.remaining_qty * lot.cost_basis for lot in lots)
            total_fees = sum((lot.remaining_qty / lot.quantity) * lot.fee if lot.quantity > 0 else 0 for lot in lots)
            avg_price = total_cost / total_qty if total_qty > 0 else 0

            pos = CFMOpenPosition(
                product_id=product_id,
                base_symbol=extract_base_symbol(product_id),
                side=side,
                quantity=total_qty,
                avg_entry_price=avg_price,
                total_fees=total_fees,
                product_type=classify_product(product_id),
                lots=lots.copy(),
            )
            positions.append(pos)

        return positions

    def get_pnl_by_product(self) -> Dict[str, Dict[str, float]]:
        """
        Get P/L breakdown by product/symbol.

        Returns:
            Dict mapping base_symbol to {gross_pnl, net_pnl, trade_count, win_rate}
        """
        by_product: Dict[str, List[CFMRealizedPL]] = {}

        for r in self._realized_pnl:
            symbol = r.base_symbol
            if symbol not in by_product:
                by_product[symbol] = []
            by_product[symbol].append(r)

        result = {}
        for symbol, trades in by_product.items():
            winners = [t for t in trades if t.net_pnl > 0]
            losers = [t for t in trades if t.net_pnl <= 0]

            result[symbol] = {
                "asset_name": CFM_SYMBOL_MAP.get(symbol, symbol),
                "product_type": classify_product(trades[0].product_id) if trades else "unknown",
                "gross_pnl": sum(t.gross_pnl for t in trades),
                "net_pnl": sum(t.net_pnl for t in trades),
                "total_fees": sum(t.entry_fee + t.exit_fee for t in trades),
                "trade_count": len(trades),
                "win_count": len(winners),
                "loss_count": len(losers),
                "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
                "avg_winner": sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0,
                "avg_loser": sum(t.net_pnl for t in losers) / len(losers) if losers else 0.0,
            }

        return result

    def get_pnl_by_product_type(self) -> Dict[str, Dict[str, float]]:
        """
        Get P/L breakdown by product type (crypto_perp vs commodity_future).

        Returns:
            Dict with 'crypto_perp' and 'commodity_future' keys
        """
        crypto_trades = [r for r in self._realized_pnl if r.product_type == "crypto_perp"]
        commodity_trades = [r for r in self._realized_pnl if r.product_type == "commodity_future"]

        def summarize(trades: List[CFMRealizedPL]) -> Dict[str, float]:
            winners = [t for t in trades if t.net_pnl > 0]
            return {
                "gross_pnl": sum(t.gross_pnl for t in trades),
                "net_pnl": sum(t.net_pnl for t in trades),
                "total_fees": sum(t.entry_fee + t.exit_fee for t in trades),
                "trade_count": len(trades),
                "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
            }

        return {
            "crypto_perp": summarize(crypto_trades),
            "commodity_future": summarize(commodity_trades),
        }

    def get_pnl_by_leverage_tier(self) -> Dict[str, Dict[str, Any]]:
        """
        Get P/L breakdown by leverage tier (intraday vs overnight).

        Returns:
            Dict with 'intraday' and 'overnight' keys containing:
            - gross_pnl, net_pnl, total_fees, trade_count, win_rate
            - avg_leverage, total_notional, total_margin, avg_roi
        """
        intraday_trades = [r for r in self._realized_pnl if r.leverage_tier == "intraday"]
        overnight_trades = [r for r in self._realized_pnl if r.leverage_tier == "overnight"]

        def summarize_tier(trades: List[CFMRealizedPL]) -> Dict[str, Any]:
            if not trades:
                return {
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "total_fees": 0.0,
                    "trade_count": 0,
                    "win_rate": 0.0,
                    "avg_leverage": 0.0,
                    "total_notional": 0.0,
                    "total_margin": 0.0,
                    "avg_roi": 0.0,
                }

            winners = [t for t in trades if t.net_pnl > 0]
            total_notional = sum(t.notional_value for t in trades)
            total_margin = sum(t.margin_required for t in trades)

            return {
                "gross_pnl": sum(t.gross_pnl for t in trades),
                "net_pnl": sum(t.net_pnl for t in trades),
                "total_fees": sum(t.entry_fee + t.exit_fee for t in trades),
                "trade_count": len(trades),
                "win_rate": len(winners) / len(trades) * 100 if trades else 0.0,
                "avg_leverage": sum(t.leverage_used for t in trades) / len(trades),
                "total_notional": total_notional,
                "total_margin": total_margin,
                "avg_roi": sum(t.roi_on_margin for t in trades) / len(trades) if trades else 0.0,
            }

        return {
            "intraday": summarize_tier(intraday_trades),
            "overnight": summarize_tier(overnight_trades),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall performance metrics.

        Returns:
            Dict with win_rate, profit_factor, expectancy, avg_hold_time, etc.
        """
        if not self._realized_pnl:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_hold_hours": 0.0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "total_fees": 0.0,
            }

        winners = [r for r in self._realized_pnl if r.net_pnl > 0]
        losers = [r for r in self._realized_pnl if r.net_pnl <= 0]

        gross_profit = sum(r.net_pnl for r in winners)
        gross_loss = abs(sum(r.net_pnl for r in losers))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_winner = gross_profit / len(winners) if winners else 0.0
        avg_loser = gross_loss / len(losers) if losers else 0.0

        win_rate = len(winners) / len(self._realized_pnl) * 100

        # Expectancy = (Win% × Avg Winner) - (Loss% × Avg Loser)
        expectancy = (win_rate / 100 * avg_winner) - ((100 - win_rate) / 100 * avg_loser)

        avg_hold_hours = sum(
            r.hold_duration.total_seconds() / 3600 for r in self._realized_pnl
        ) / len(self._realized_pnl)

        return {
            "trade_count": len(self._realized_pnl),
            "win_count": len(winners),
            "loss_count": len(losers),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "avg_hold_hours": avg_hold_hours,
            "gross_pnl": sum(r.gross_pnl for r in self._realized_pnl),
            "net_pnl": sum(r.net_pnl for r in self._realized_pnl),
            "total_fees": sum(r.entry_fee + r.exit_fee for r in self._realized_pnl),
        }

    def get_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent closed trades as dictionaries.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dicts sorted by exit_time descending
        """
        sorted_trades = sorted(
            self._realized_pnl,
            key=lambda r: r.exit_time,
            reverse=True,
        )
        return [t.to_dict() for t in sorted_trades[:limit]]

    def add_funding_payments(self, payments: List[Dict[str, Any]]) -> None:
        """
        Add funding payment records for perpetuals.

        Args:
            payments: List of funding payment dicts from get_cfm_funding_payments()
        """
        self._funding_payments = payments

    def get_funding_summary(self) -> Dict[str, float]:
        """
        Get funding payment summary.

        Returns:
            Dict with total_paid, total_received, net_funding
        """
        paid = 0.0
        received = 0.0

        for payment in self._funding_payments:
            amount = float(payment.get("amount", 0))
            if amount > 0:
                paid += amount
            else:
                received += abs(amount)

        return {
            "total_paid": paid,
            "total_received": received,
            "net_funding": received - paid,
            "payment_count": len(self._funding_payments),
        }

    def get_cumulative_pnl_series(self) -> List[Dict[str, Any]]:
        """
        Generate cumulative P/L time series for charting.

        Returns:
            List of dicts with timestamp, daily_pnl, cumulative_pnl, trade_count
            sorted by timestamp ascending.
        """
        if not self._realized_pnl:
            return []

        # Sort trades by exit time
        sorted_trades = sorted(self._realized_pnl, key=lambda r: r.exit_time)

        # Group by date and calculate daily/cumulative P/L
        from collections import defaultdict
        daily_pnl: Dict[str, float] = defaultdict(float)
        daily_trades: Dict[str, int] = defaultdict(int)

        for trade in sorted_trades:
            date_key = trade.exit_time.strftime("%Y-%m-%d")
            daily_pnl[date_key] += trade.net_pnl
            daily_trades[date_key] += 1

        # Build cumulative series
        series = []
        cumulative = 0.0
        total_trades = 0

        for date_key in sorted(daily_pnl.keys()):
            cumulative += daily_pnl[date_key]
            total_trades += daily_trades[date_key]
            series.append({
                "date": date_key,
                "daily_pnl": round(daily_pnl[date_key], 2),
                "cumulative_pnl": round(cumulative, 2),
                "trade_count": daily_trades[date_key],
                "total_trades": total_trades,
            })

        return series

    def get_pnl_by_product_series(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate cumulative P/L time series per product for charting.

        Returns:
            Dict mapping product symbol to list of cumulative P/L data points.
        """
        if not self._realized_pnl:
            return {}

        from collections import defaultdict

        # Group trades by product and date
        product_daily: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for trade in self._realized_pnl:
            date_key = trade.exit_time.strftime("%Y-%m-%d")
            product_daily[trade.base_symbol][date_key] += trade.net_pnl

        # Build cumulative series per product
        result = {}
        for symbol, daily_data in product_daily.items():
            series = []
            cumulative = 0.0
            for date_key in sorted(daily_data.keys()):
                cumulative += daily_data[date_key]
                series.append({
                    "date": date_key,
                    "daily_pnl": round(daily_data[date_key], 2),
                    "cumulative_pnl": round(cumulative, 2),
                })
            result[symbol] = series

        return result
