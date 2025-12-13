"""
Derivatives-specific utilities for perpetual futures trading.

Handles:
- Funding rate calculations
- Margin requirements
- Leverage tier selection
- Liquidation price calculation
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from crypto.config import (
    FUNDING_INTERVAL_HOURS,
    FUNDING_RATE_PER_PERIOD,
    FUNDING_TIMES_UTC,
    INCLUDE_FUNDING_IN_PNL,
    INITIAL_MARGIN_PERCENT,
    LEVERAGE_TIERS,
    MAINTENANCE_MARGIN_PERCENT,
)


def get_leverage_for_tier(
    symbol: str,
    tier: str = "swing",
) -> float:
    """
    Get maximum leverage for a symbol and tier.

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        tier: 'intraday' (10x) or 'swing' (4x)

    Returns:
        Maximum leverage for the tier
    """
    # Extract base asset (BTC from BTC-USD)
    base = symbol.split("-")[0] if "-" in symbol else symbol

    tier_config = LEVERAGE_TIERS.get(tier, LEVERAGE_TIERS["swing"])
    return tier_config.get(base, 4.0)


def calculate_funding_cost(
    position_size_usd: float,
    side: str,
    holding_hours: float,
    funding_rate: Optional[float] = None,
) -> float:
    """
    Calculate funding cost/income for a position.

    Funding is paid/received every 8 hours:
    - Long pays short when funding rate is positive
    - Short pays long when funding rate is negative

    Args:
        position_size_usd: Position notional value in USD
        side: 'BUY' (long) or 'SELL' (short)
        holding_hours: How long position was held
        funding_rate: Per-period funding rate (default from config)

    Returns:
        Funding cost (negative = cost, positive = income)
    """
    if not INCLUDE_FUNDING_IN_PNL:
        return 0.0

    rate = funding_rate if funding_rate is not None else FUNDING_RATE_PER_PERIOD

    # Number of funding periods during holding time
    funding_periods = holding_hours / FUNDING_INTERVAL_HOURS

    # Calculate funding payment
    funding_payment = position_size_usd * rate * funding_periods

    # Long pays when rate is positive, short receives
    if side == "BUY":
        return -funding_payment  # Cost for longs
    else:
        return funding_payment  # Income for shorts


def get_next_funding_time() -> datetime:
    """
    Get the next funding time in UTC.

    Returns:
        datetime of next funding period
    """
    now = datetime.utcnow()

    for time_str in FUNDING_TIMES_UTC:
        hour, minute = map(int, time_str.split(":"))
        funding_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if funding_time > now:
            return funding_time

    # Next funding is tomorrow at first time
    hour, minute = map(int, FUNDING_TIMES_UTC[0].split(":"))
    return (now + timedelta(days=1)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


def time_to_funding() -> Tuple[float, str]:
    """
    Calculate time until next funding.

    Returns:
        Tuple of (hours_remaining, formatted_string)
    """
    next_funding = get_next_funding_time()
    delta = next_funding - datetime.utcnow()
    hours = delta.total_seconds() / 3600

    if hours < 1:
        minutes = int(delta.total_seconds() / 60)
        return hours, f"{minutes}m to funding"
    else:
        return hours, f"{hours:.1f}h to funding"


def calculate_initial_margin(
    position_size_usd: float,
    symbol: str,
) -> float:
    """
    Calculate initial margin required for a position.

    Args:
        position_size_usd: Position notional value
        symbol: Trading pair

    Returns:
        Initial margin in USD
    """
    base = symbol.split("-")[0] if "-" in symbol else symbol
    margin_pct = INITIAL_MARGIN_PERCENT.get(base, 0.10)
    return position_size_usd * margin_pct


def calculate_maintenance_margin(
    position_size_usd: float,
    symbol: str,
) -> float:
    """
    Calculate maintenance margin for a position.

    Args:
        position_size_usd: Position notional value
        symbol: Trading pair

    Returns:
        Maintenance margin in USD
    """
    base = symbol.split("-")[0] if "-" in symbol else symbol
    margin_pct = MAINTENANCE_MARGIN_PERCENT.get(base, 0.05)
    return position_size_usd * margin_pct


def calculate_liquidation_price(
    entry_price: float,
    side: str,
    leverage: float,
    symbol: str,
) -> float:
    """
    Calculate liquidation price for a leveraged position.

    Liquidation occurs when unrealized loss equals margin minus maintenance.

    Args:
        entry_price: Position entry price
        side: 'BUY' (long) or 'SELL' (short)
        leverage: Position leverage
        symbol: Trading pair

    Returns:
        Liquidation price
    """
    base = symbol.split("-")[0] if "-" in symbol else symbol
    maint_margin_pct = MAINTENANCE_MARGIN_PERCENT.get(base, 0.05)

    # Distance to liquidation as percentage
    # For a long: liq_price = entry * (1 - (1/leverage) + maint_margin)
    # For a short: liq_price = entry * (1 + (1/leverage) - maint_margin)

    if side == "BUY":
        liq_distance = (1 / leverage) - maint_margin_pct
        liq_price = entry_price * (1 - liq_distance)
    else:
        liq_distance = (1 / leverage) - maint_margin_pct
        liq_price = entry_price * (1 + liq_distance)

    return liq_price


def should_close_before_funding(
    entry_time: datetime,
    side: str,
    expected_funding_rate: float = FUNDING_RATE_PER_PERIOD,
) -> Tuple[bool, str]:
    """
    Determine if position should be closed before funding.

    Useful for intraday trading to avoid funding costs.

    Args:
        entry_time: When position was opened
        side: 'BUY' or 'SELL'
        expected_funding_rate: Expected funding rate

    Returns:
        Tuple of (should_close, reason)
    """
    hours_to_funding, time_str = time_to_funding()

    # If funding is soon (< 30 min) and rate is unfavorable
    if hours_to_funding < 0.5:
        if side == "BUY" and expected_funding_rate > 0:
            return True, f"Close long before funding ({time_str}, rate positive)"
        elif side == "SELL" and expected_funding_rate < 0:
            return True, f"Close short before funding ({time_str}, rate negative)"

    return False, ""


def calculate_effective_leverage(
    account_equity: float,
    position_size_usd: float,
) -> float:
    """
    Calculate effective leverage of current position.

    Args:
        account_equity: Total account equity
        position_size_usd: Position notional value

    Returns:
        Effective leverage ratio
    """
    if account_equity <= 0:
        return 0.0
    return position_size_usd / account_equity


def is_leverage_safe(
    account_equity: float,
    position_size_usd: float,
    tier: str = "swing",
    symbol: str = "BTC-USD",
) -> Tuple[bool, str]:
    """
    Check if current leverage is within safe limits for tier.

    Args:
        account_equity: Total account equity
        position_size_usd: Position notional value
        tier: 'intraday' or 'swing'
        symbol: Trading pair

    Returns:
        Tuple of (is_safe, warning_message)
    """
    effective_lev = calculate_effective_leverage(account_equity, position_size_usd)
    max_lev = get_leverage_for_tier(symbol, tier)

    if effective_lev > max_lev:
        return False, (
            f"Leverage {effective_lev:.1f}x exceeds {tier} max {max_lev:.1f}x"
        )
    elif effective_lev > max_lev * 0.8:
        return True, (
            f"Warning: Leverage {effective_lev:.1f}x approaching {tier} max {max_lev:.1f}x"
        )

    return True, ""
