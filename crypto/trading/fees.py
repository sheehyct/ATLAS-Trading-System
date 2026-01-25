"""
Fee calculations for Coinbase CFM crypto derivatives trading.

Implements accurate fee modeling including:
- Percentage-based taker/maker fees
- Fixed fee per contract
- Round-trip cost calculations
- VectorBT Pro integration for backtesting

Coinbase CFM Fee Structure - VERIFIED Jan 24, 2026:
- Taker fee: 0.07% (0.0007) based on trailing 30-day volume
- Maker fee: 0.065% (0.00065) for limit orders
- Fixed fee: $0.15 per contract (covers NFA, exchange, vendor costs)
- Formula: (Notional × Rate) + (Fixed × Contracts)

Reference: Coinbase CFM platform fee tier display
"""

from typing import Callable, Dict, Optional


# =============================================================================
# FEE CONSTANTS - VERIFIED Jan 24, 2026
# =============================================================================

# Coinbase CFM fee structure
# Note: Fee tiers are based on trailing 30-day derivatives volume
# These are default tier rates - may be lower with higher volume
TAKER_FEE_RATE: float = 0.0007   # 0.07%
MAKER_FEE_RATE: float = 0.00065  # 0.065% for limit orders

# Fixed per-contract fee (covers NFA, exchange, vendor costs - ALL trades)
FIXED_FEE_PER_CONTRACT: float = 0.15  # $0.15 per contract

# Legacy alias for backward compatibility
MIN_FEE_PER_CONTRACT: float = FIXED_FEE_PER_CONTRACT

# Contract specifications for fee calculations
# Maps symbol prefix to contract multiplier (units of underlying per contract)
CONTRACT_MULTIPLIERS: Dict[str, float] = {
    "BTC": 0.01,      # Nano BTC = 0.01 BTC per contract
    "BIP": 0.01,      # CFM Nano Bitcoin
    "ETH": 0.10,      # Nano ETH = 0.10 ETH per contract
    "ETP": 0.10,      # CFM Nano Ether
    "SOL": 5.0,       # Nano SOL = 5 SOL per contract
    "XRP": 500.0,     # Nano XRP = 500 XRP per contract
    "ADA": 1000.0,    # Nano ADA = 1000 ADA per contract
}


# =============================================================================
# FEE CALCULATION FUNCTIONS
# =============================================================================


def calculate_fee(
    notional_value: float,
    num_contracts: int = 1,
    is_maker: bool = False,
) -> float:
    """
    Calculate trading fee with fixed per-contract component.

    Formula: (Notional × Rate) + (Fixed × Contracts)
    
    VERIFIED Jan 24, 2026 from Coinbase CFM platform.

    Args:
        notional_value: Total notional value of the trade in USD
        num_contracts: Number of contracts traded
        is_maker: True for limit orders (maker), False for market orders (taker)

    Returns:
        Fee amount in USD

    Example:
        >>> calculate_fee(notional_value=1000, num_contracts=1, is_maker=False)
        0.85  # $1000 × 0.07% + $0.15 = $0.70 + $0.15 = $0.85

        >>> calculate_fee(notional_value=50000, num_contracts=5, is_maker=False)
        35.75  # $50,000 × 0.07% + 5 × $0.15 = $35 + $0.75 = $35.75
    """
    if notional_value <= 0 or num_contracts <= 0:
        return 0.0

    fee_rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
    percentage_fee = notional_value * fee_rate
    fixed_fee = FIXED_FEE_PER_CONTRACT * num_contracts

    return percentage_fee + fixed_fee


def calculate_round_trip_fee(
    notional_value: float,
    num_contracts: int = 1,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
) -> float:
    """
    Calculate total fees for a round-trip trade (entry + exit).

    Args:
        notional_value: Notional value of the position in USD
        num_contracts: Number of contracts
        entry_is_maker: True if entry order is a limit order
        exit_is_maker: True if exit order is a limit order

    Returns:
        Total round-trip fee in USD

    Example:
        >>> calculate_round_trip_fee(5000, num_contracts=12, entry_is_maker=False, exit_is_maker=False)
        3.60  # 12 contracts × $0.15 × 2 = $3.60 (minimum dominates)
    """
    entry_fee = calculate_fee(notional_value, num_contracts, entry_is_maker)
    exit_fee = calculate_fee(notional_value, num_contracts, exit_is_maker)
    return entry_fee + exit_fee


def calculate_breakeven_move(
    notional_value: float,
    num_contracts: int = 1,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
) -> float:
    """
    Calculate minimum price move percentage to breakeven after fees.

    Args:
        notional_value: Notional value of the position in USD
        num_contracts: Number of contracts
        entry_is_maker: True if entry order is a limit order
        exit_is_maker: True if exit order is a limit order

    Returns:
        Breakeven move as decimal (0.001 = 0.1%)

    Example:
        >>> calculate_breakeven_move(5000, num_contracts=12)
        0.00072  # Need 0.072% move to cover $3.60 in fees on $5000 notional
    """
    if notional_value <= 0:
        return 0.0

    total_fees = calculate_round_trip_fee(
        notional_value, num_contracts, entry_is_maker, exit_is_maker
    )
    return total_fees / notional_value


def calculate_num_contracts(
    notional_value: float,
    price: float,
    symbol: str,
) -> int:
    """
    Calculate number of contracts for a given notional value.

    Args:
        notional_value: Desired notional exposure in USD
        price: Current price of the underlying asset
        symbol: Symbol or symbol prefix (e.g., "BTC", "ADA", "BIP")

    Returns:
        Number of whole contracts (rounded down)

    Example:
        >>> calculate_num_contracts(5000, price=0.35, symbol="ADA")
        14  # $5000 / ($0.35 × 1000 ADA/contract) = 14.28 → 14 contracts
    """
    # Extract symbol prefix if full symbol provided
    symbol_prefix = symbol.split("-")[0].upper()

    multiplier = CONTRACT_MULTIPLIERS.get(symbol_prefix)
    if multiplier is None:
        raise ValueError(f"Unknown symbol: {symbol}. Known: {list(CONTRACT_MULTIPLIERS.keys())}")

    notional_per_contract = price * multiplier
    if notional_per_contract <= 0:
        return 0

    return int(notional_value / notional_per_contract)


def calculate_notional_from_contracts(
    num_contracts: int,
    price: float,
    symbol: str,
) -> float:
    """
    Calculate notional value from number of contracts.

    Args:
        num_contracts: Number of contracts
        price: Current price of the underlying asset
        symbol: Symbol or symbol prefix (e.g., "BTC", "ADA", "BIP")

    Returns:
        Notional value in USD

    Example:
        >>> calculate_notional_from_contracts(12, price=0.3415, symbol="ADA")
        4098.0  # 12 contracts × 1000 ADA × $0.3415 = $4,098
    """
    symbol_prefix = symbol.split("-")[0].upper()

    multiplier = CONTRACT_MULTIPLIERS.get(symbol_prefix)
    if multiplier is None:
        raise ValueError(f"Unknown symbol: {symbol}. Known: {list(CONTRACT_MULTIPLIERS.keys())}")

    return num_contracts * multiplier * price


# =============================================================================
# VECTORBT PRO INTEGRATION
# =============================================================================


def create_coinbase_fee_func(
    symbol: str,
    is_maker: bool = False,
) -> Callable:
    """
    Create a fee function compatible with VectorBT Pro backtesting.

    VectorBT expects a function with signature: func(col, i, val) -> fee
    where val is the trade value (positive for buys, negative for sells).
    
    Fee formula: (Notional × Rate) + (Fixed × Contracts)

    Args:
        symbol: Trading symbol for contract size lookup
        is_maker: Whether to use maker fee rate

    Returns:
        Fee function for VectorBT Pro

    Example:
        >>> fee_func = create_coinbase_fee_func("ADA", is_maker=False)
        >>> # Use in VectorBT:
        >>> pf = vbt.Portfolio.from_signals(
        ...     close=price_data,
        ...     entries=entries,
        ...     exits=exits,
        ...     fees=fee_func,
        ... )
    """
    symbol_prefix = symbol.split("-")[0].upper()
    multiplier = CONTRACT_MULTIPLIERS.get(symbol_prefix, 1.0)

    def coinbase_cfm_fee(col, i, val):
        """VectorBT-compatible fee function."""
        notional = abs(val)
        if notional <= 0:
            return 0.0

        # Estimate number of contracts from notional
        # This is approximate since we don't have price at this point
        # For accurate backtesting, price should be passed separately
        num_contracts = max(1, int(notional / (multiplier * 100)))  # Rough estimate

        fee_rate = MAKER_FEE_RATE if is_maker else TAKER_FEE_RATE
        percentage_fee = notional * fee_rate
        fixed_fee = FIXED_FEE_PER_CONTRACT * num_contracts

        return percentage_fee + fixed_fee

    return coinbase_cfm_fee


def create_fixed_pct_fee_func(fee_pct: float = 0.0007) -> Callable:
    """
    Create a simple percentage-based fee function for VectorBT.

    Use this for quick backtesting when contract-level precision isn't needed.
    Default is taker rate (0.07%). Does NOT include fixed per-contract fee.

    Args:
        fee_pct: Fee as decimal (0.0007 = 0.07%)

    Returns:
        Fee function for VectorBT Pro
    """
    def fixed_fee(col, i, val):
        return abs(val) * fee_pct

    return fixed_fee


# =============================================================================
# FEE IMPACT ANALYSIS
# =============================================================================


def analyze_fee_impact(
    account_value: float,
    leverage: float,
    price: float,
    symbol: str,
    stop_percent: float,
    target_percent: float,
) -> Dict[str, float]:
    """
    Analyze fee impact on a hypothetical trade.

    Useful for understanding how fees affect expected returns.

    Args:
        account_value: Account equity in USD
        leverage: Leverage multiplier
        price: Current asset price
        symbol: Trading symbol
        stop_percent: Stop loss as decimal (0.02 = 2%)
        target_percent: Target as decimal (0.04 = 4%)

    Returns:
        Dictionary with fee analysis metrics

    Example:
        >>> analyze_fee_impact(
        ...     account_value=1000, leverage=5, price=0.35,
        ...     symbol="ADA", stop_percent=0.02, target_percent=0.04
        ... )
        {
            'notional': 5000.0,
            'num_contracts': 14,
            'round_trip_fee': 4.20,
            'breakeven_move': 0.00084,
            'fee_as_pct_of_target': 0.021,
            'fee_as_pct_of_stop': 0.042,
            'net_target_pct': 0.03916,
            'net_rr_ratio': 1.958,
        }
    """
    notional = account_value * leverage
    num_contracts = calculate_num_contracts(notional, price, symbol)

    # Recalculate actual notional based on whole contracts
    actual_notional = calculate_notional_from_contracts(num_contracts, price, symbol)

    rt_fee = calculate_round_trip_fee(actual_notional, num_contracts)
    breakeven = calculate_breakeven_move(actual_notional, num_contracts)

    # Expected P/L before fees
    gross_target_pnl = actual_notional * target_percent
    gross_stop_loss = actual_notional * stop_percent

    # Net P/L after fees
    net_target_pnl = gross_target_pnl - rt_fee
    net_stop_loss = gross_stop_loss + rt_fee  # Fees add to loss

    # Fee impact metrics
    fee_pct_of_target = rt_fee / gross_target_pnl if gross_target_pnl > 0 else 0
    fee_pct_of_stop = rt_fee / gross_stop_loss if gross_stop_loss > 0 else 0

    # Net R:R ratio
    net_rr = net_target_pnl / net_stop_loss if net_stop_loss > 0 else 0

    return {
        "notional": actual_notional,
        "num_contracts": num_contracts,
        "round_trip_fee": rt_fee,
        "breakeven_move": breakeven,
        "fee_as_pct_of_target": fee_pct_of_target,
        "fee_as_pct_of_stop": fee_pct_of_stop,
        "net_target_pct": target_percent - breakeven,
        "net_rr_ratio": net_rr,
    }
