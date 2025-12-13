"""
Position sizing for crypto derivatives trading.

Implements ATR-based position sizing with leverage limits.
"""

from typing import Tuple


def calculate_position_size(
    account_value: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    max_leverage: float = 8.0,
) -> Tuple[float, float, float]:
    """
    Calculate position size based on risk and stop distance.

    Uses ATR-derived stop distance to determine position size while
    respecting maximum leverage limits.

    Args:
        account_value: Current account equity in USD
        risk_percent: Risk per trade as decimal (0.02 = 2%)
        entry_price: Intended entry price
        stop_price: Stop loss price
        max_leverage: Maximum allowed leverage (default: 8x)

    Returns:
        Tuple of (position_size_base, implied_leverage, actual_risk_usd)
        - position_size_base: Quantity in base currency (e.g., BTC)
        - implied_leverage: Leverage used for this position
        - actual_risk_usd: Actual dollar risk (may be less if leverage-capped)

    Example:
        >>> size, lev, risk = calculate_position_size(400, 0.02, 100000, 98500, 8.0)
        >>> # With $400, 2% risk ($8), BTC at $100k, stop at $98.5k (1.5% stop)
        >>> # Position would be ~$533 notional, 1.33x leverage, $8 risk
    """
    # Validate inputs
    if entry_price <= 0 or stop_price <= 0 or account_value <= 0:
        return 0.0, 0.0, 0.0

    # Calculate stop distance
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return 0.0, 0.0, 0.0

    stop_percent = stop_distance / entry_price

    # Calculate target risk in USD
    risk_usd = account_value * risk_percent

    # Position size in USD notional to achieve target risk
    position_notional = risk_usd / stop_percent

    # Calculate implied leverage
    implied_leverage = position_notional / account_value

    # Cap at max leverage if needed
    if implied_leverage > max_leverage:
        position_notional = account_value * max_leverage
        implied_leverage = max_leverage
        # Recalculate actual risk (will be less than target)
        actual_risk = position_notional * stop_percent
    else:
        actual_risk = risk_usd

    # Convert to base currency quantity
    position_size_base = position_notional / entry_price

    return position_size_base, implied_leverage, actual_risk


def should_skip_trade(
    account_value: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    max_leverage: float = 8.0,
) -> Tuple[bool, str]:
    """
    Determine if a trade should be skipped due to leverage constraints.

    If the stop distance requires more than max_leverage to maintain
    the target risk, the trade should be skipped entirely rather than
    reducing the risk amount.

    Args:
        account_value: Current account equity
        risk_percent: Risk per trade as decimal
        entry_price: Intended entry price
        stop_price: Stop loss price
        max_leverage: Maximum allowed leverage

    Returns:
        Tuple of (should_skip, reason)
    """
    if entry_price <= 0 or stop_price <= 0:
        return True, "Invalid entry or stop price"

    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0:
        return True, "Stop distance is zero"

    stop_percent = stop_distance / entry_price
    risk_usd = account_value * risk_percent
    position_notional = risk_usd / stop_percent
    implied_leverage = position_notional / account_value

    if implied_leverage > max_leverage:
        return True, (
            f"Setup requires {implied_leverage:.1f}x leverage to maintain "
            f"{risk_percent*100:.1f}% risk. Max allowed: {max_leverage}x. "
            "Wait for setup with tighter stop."
        )

    return False, "Trade acceptable"


def calculate_stop_distance_for_leverage(
    account_value: float,
    risk_percent: float,
    entry_price: float,
    target_leverage: float,
) -> float:
    """
    Calculate required stop distance for a target leverage.

    Useful for determining minimum stop distance to stay within leverage limits.

    Args:
        account_value: Current account equity
        risk_percent: Risk per trade as decimal
        entry_price: Intended entry price
        target_leverage: Desired leverage

    Returns:
        Required stop distance in price units
    """
    risk_usd = account_value * risk_percent
    position_notional = account_value * target_leverage
    stop_percent = risk_usd / position_notional
    return entry_price * stop_percent
