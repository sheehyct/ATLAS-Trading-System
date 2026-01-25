"""
Beta and capital efficiency analysis for crypto derivatives trading.

Implements:
- Beta calculation (asset volatility relative to BTC)
- Capital efficiency metrics accounting for leverage tiers
- Instrument selection/ranking for optimal capital deployment
- Rolling beta calculations for dynamic adjustment

Key Insight (Session Jan 23, 2026):
Lower leverage altcoins can outperform higher leverage BTC/ETH
when their beta exceeds the leverage differential.

Example: ADA with 5x leverage and 2.2x beta produces higher returns
than BTC with 10x leverage and 1.0x beta on the same market move.

Effective Multiplier = Leverage × Beta
- BTC:  10x × 1.00 = 10.0
- ETH:  10x × 1.98 = 19.8
- ADA:  5x  × 2.20 = 11.0
- XRP:  5x  × 1.77 = 8.85
- SOL:  5x  × 1.55 = 7.75
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


# =============================================================================
# EMPIRICAL BETA VALUES
# =============================================================================

# Beta values calculated from Day Down/Day Up ranges (Jan 23, 2026 snapshot)
# These represent how much each asset moves relative to BTC on the same market structure
# Should be recalculated periodically as market dynamics change

CRYPTO_BETA_TO_BTC: Dict[str, float] = {
    "BTC": 1.00,
    "ETH": 1.98,
    "SOL": 1.55,
    "XRP": 1.77,
    "ADA": 2.20,
}

# Intraday leverage tiers (confirmed from Coinbase CFM trading)
INTRADAY_LEVERAGE: Dict[str, float] = {
    "BTC": 10.0,
    "ETH": 10.0,
    "SOL": 5.0,
    "XRP": 5.0,
    "ADA": 5.0,
}

# Swing/Overnight leverage tiers (4PM ET to 6PM ET, weekends)
# VERIFIED Jan 24, 2026 from Coinbase CFM platform
OVERNIGHT_LEVERAGE: Dict[str, float] = {
    "BTC": 4.1,
    "ETH": 4.0,
    "SOL": 2.7,
    "XRP": 2.6,
    "ADA": 3.4,
}

# Legacy alias for backward compatibility
SWING_LEVERAGE: Dict[str, float] = OVERNIGHT_LEVERAGE


# =============================================================================
# EFFECTIVE MULTIPLIER CALCULATIONS
# =============================================================================


def calculate_effective_multiplier(
    symbol: str,
    leverage_tier: str = "intraday",
) -> float:
    """
    Calculate effective multiplier (leverage × beta) for a symbol.

    The effective multiplier represents total capital efficiency,
    accounting for both leverage and volatility characteristics.

    Args:
        symbol: Asset symbol (BTC, ETH, SOL, XRP, ADA)
        leverage_tier: "intraday" or "swing"

    Returns:
        Effective multiplier

    Example:
        >>> calculate_effective_multiplier("ADA", "intraday")
        11.0  # 5x leverage × 2.2 beta
    """
    symbol = symbol.upper().split("-")[0]

    beta = CRYPTO_BETA_TO_BTC.get(symbol, 1.0)

    if leverage_tier == "intraday":
        leverage = INTRADAY_LEVERAGE.get(symbol, 5.0)
    else:
        leverage = SWING_LEVERAGE.get(symbol, 3.0)

    return leverage * beta


def get_effective_multipliers(
    leverage_tier: str = "intraday",
) -> Dict[str, float]:
    """
    Get effective multipliers for all tracked symbols.

    Args:
        leverage_tier: "intraday" or "swing"

    Returns:
        Dictionary of symbol -> effective multiplier

    Example:
        >>> get_effective_multipliers("intraday")
        {'BTC': 10.0, 'ETH': 19.8, 'SOL': 7.75, 'XRP': 8.85, 'ADA': 11.0}
    """
    return {
        symbol: calculate_effective_multiplier(symbol, leverage_tier)
        for symbol in CRYPTO_BETA_TO_BTC.keys()
    }


def rank_by_capital_efficiency(
    leverage_tier: str = "intraday",
) -> List[Tuple[str, float]]:
    """
    Rank symbols by capital efficiency (effective multiplier).

    Use this to prioritize instruments when multiple setups are available.

    Args:
        leverage_tier: "intraday" or "swing"

    Returns:
        List of (symbol, effective_multiplier) sorted descending

    Example:
        >>> rank_by_capital_efficiency("intraday")
        [('ETH', 19.8), ('ADA', 11.0), ('BTC', 10.0), ('XRP', 8.85), ('SOL', 7.75)]
    """
    multipliers = get_effective_multipliers(leverage_tier)
    return sorted(multipliers.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# P/L PROJECTION
# =============================================================================


def project_pnl_on_btc_move(
    btc_move_percent: float,
    account_value: float,
    symbol: str,
    leverage_tier: str = "intraday",
) -> Dict[str, float]:
    """
    Project P/L for a symbol based on an expected BTC move.

    Uses beta to estimate how much the symbol will move when BTC moves.

    Args:
        btc_move_percent: Expected BTC move as decimal (0.03 = 3%)
        account_value: Account equity in USD
        symbol: Asset symbol
        leverage_tier: "intraday" or "swing"

    Returns:
        Dictionary with projection details

    Example:
        >>> project_pnl_on_btc_move(0.03, 1000, "ADA", "intraday")
        {
            'symbol': 'ADA',
            'btc_move': 0.03,
            'beta': 2.2,
            'expected_move': 0.066,
            'leverage': 5.0,
            'notional': 5000.0,
            'expected_pnl': 330.0,
            'pnl_percent': 0.33,
        }
    """
    symbol = symbol.upper().split("-")[0]

    beta = CRYPTO_BETA_TO_BTC.get(symbol, 1.0)
    expected_move = btc_move_percent * beta

    if leverage_tier == "intraday":
        leverage = INTRADAY_LEVERAGE.get(symbol, 5.0)
    else:
        leverage = SWING_LEVERAGE.get(symbol, 3.0)

    notional = account_value * leverage
    expected_pnl = notional * expected_move

    return {
        "symbol": symbol,
        "btc_move": btc_move_percent,
        "beta": beta,
        "expected_move": expected_move,
        "leverage": leverage,
        "notional": notional,
        "expected_pnl": expected_pnl,
        "pnl_percent": expected_pnl / account_value,
    }


def compare_instruments_on_btc_move(
    btc_move_percent: float,
    account_value: float,
    leverage_tier: str = "intraday",
) -> pd.DataFrame:
    """
    Compare all instruments' projected P/L on a given BTC move.

    Useful for instrument selection decisions.

    Args:
        btc_move_percent: Expected BTC move as decimal
        account_value: Account equity in USD
        leverage_tier: "intraday" or "swing"

    Returns:
        DataFrame with projections for all symbols, sorted by P/L

    Example:
        >>> df = compare_instruments_on_btc_move(0.0358, 1000, "intraday")
        >>> print(df[['symbol', 'expected_pnl', 'pnl_percent']])
    """
    projections = []
    for symbol in CRYPTO_BETA_TO_BTC.keys():
        proj = project_pnl_on_btc_move(
            btc_move_percent, account_value, symbol, leverage_tier
        )
        projections.append(proj)

    df = pd.DataFrame(projections)
    df = df.sort_values("expected_pnl", ascending=False)
    return df.reset_index(drop=True)


# =============================================================================
# ROLLING BETA CALCULATION
# =============================================================================


def calculate_rolling_beta(
    asset_prices: pd.Series,
    btc_prices: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Calculate rolling beta of an asset versus BTC.

    Uses covariance/variance method on returns.

    Args:
        asset_prices: Price series for the asset
        btc_prices: Price series for BTC
        window: Rolling window size in periods

    Returns:
        Series of rolling beta values

    Example:
        >>> ada_beta = calculate_rolling_beta(ada_close, btc_close, window=20)
        >>> print(f"Current ADA beta: {ada_beta.iloc[-1]:.2f}")
    """
    # Calculate returns
    asset_returns = asset_prices.pct_change()
    btc_returns = btc_prices.pct_change()

    # Rolling covariance and variance
    covariance = asset_returns.rolling(window).cov(btc_returns)
    btc_variance = btc_returns.rolling(window).var()

    # Beta = Cov(asset, btc) / Var(btc)
    beta = covariance / btc_variance

    return beta


def calculate_beta_from_ranges(
    asset_high: float,
    asset_low: float,
    btc_high: float,
    btc_low: float,
) -> float:
    """
    Calculate beta from Day Up/Day Down price ranges.

    This is the method used in the original analysis - comparing
    the percentage range of each asset.

    Args:
        asset_high: Asset's Day Up price
        asset_low: Asset's Day Down price
        btc_high: BTC's Day Up price
        btc_low: BTC's Day Down price

    Returns:
        Beta value

    Example:
        >>> beta = calculate_beta_from_ranges(
        ...     asset_high=0.3737, asset_low=0.3464,  # ADA
        ...     btc_high=90273, btc_low=87156,        # BTC
        ... )
        >>> print(f"ADA beta: {beta:.2f}")  # 2.20
    """
    asset_range_pct = (asset_high - asset_low) / asset_low
    btc_range_pct = (btc_high - btc_low) / btc_low

    if btc_range_pct == 0:
        return 1.0

    return asset_range_pct / btc_range_pct


def update_beta_from_current_levels(
    levels: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Update beta values from current Day Up/Day Down levels.

    Args:
        levels: Dictionary of symbol -> {'day_up': float, 'day_down': float}
                Must include 'BTC' as baseline.

    Returns:
        Dictionary of symbol -> beta

    Example:
        >>> levels = {
        ...     'BTC': {'day_up': 90273, 'day_down': 87156},
        ...     'ADA': {'day_up': 0.3737, 'day_down': 0.3464},
        ...     'ETH': {'day_up': 3066.65, 'day_down': 2863.40},
        ... }
        >>> betas = update_beta_from_current_levels(levels)
    """
    if "BTC" not in levels:
        raise ValueError("BTC levels required as baseline")

    btc = levels["BTC"]
    betas = {"BTC": 1.0}

    for symbol, data in levels.items():
        if symbol == "BTC":
            continue

        betas[symbol] = calculate_beta_from_ranges(
            asset_high=data["day_up"],
            asset_low=data["day_down"],
            btc_high=btc["day_up"],
            btc_low=btc["day_down"],
        )

    return betas


# =============================================================================
# BETA-ADJUSTED POSITION SIZING
# =============================================================================


def calculate_beta_adjusted_size(
    account_value: float,
    target_risk_usd: float,
    entry_price: float,
    stop_price: float,
    symbol: str,
    leverage_tier: str = "intraday",
    normalize_to_btc: bool = True,
) -> Dict[str, float]:
    """
    Calculate position size with beta adjustment.

    When normalize_to_btc=True, sizes the position so that dollar risk
    is equivalent to what it would be trading BTC with the same setup.

    This prevents high-beta assets from creating outsized risk.

    Args:
        account_value: Account equity in USD
        target_risk_usd: Target risk in USD
        entry_price: Entry price
        stop_price: Stop loss price
        symbol: Asset symbol
        leverage_tier: "intraday" or "swing"
        normalize_to_btc: If True, adjust size to normalize risk to BTC equivalent

    Returns:
        Dictionary with sizing details

    Example:
        >>> sizing = calculate_beta_adjusted_size(
        ...     account_value=1000, target_risk_usd=50,
        ...     entry_price=0.35, stop_price=0.34,
        ...     symbol="ADA", normalize_to_btc=True
        ... )
    """
    symbol = symbol.upper().split("-")[0]
    beta = CRYPTO_BETA_TO_BTC.get(symbol, 1.0)

    if leverage_tier == "intraday":
        max_leverage = INTRADAY_LEVERAGE.get(symbol, 5.0)
    else:
        max_leverage = SWING_LEVERAGE.get(symbol, 3.0)

    stop_distance = abs(entry_price - stop_price)
    stop_percent = stop_distance / entry_price

    # Standard position sizing (risk-based)
    position_notional = target_risk_usd / stop_percent

    # Beta adjustment: High-beta assets get smaller positions
    if normalize_to_btc and beta > 0:
        adjusted_notional = position_notional / beta
    else:
        adjusted_notional = position_notional

    # Cap at max leverage
    max_notional = account_value * max_leverage
    final_notional = min(adjusted_notional, max_notional)

    # Calculate actual values
    implied_leverage = final_notional / account_value
    actual_risk = final_notional * stop_percent
    beta_adjusted_risk = actual_risk * beta  # True risk accounting for beta

    return {
        "symbol": symbol,
        "beta": beta,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_percent": stop_percent,
        "target_risk_usd": target_risk_usd,
        "raw_notional": position_notional,
        "beta_adjusted_notional": adjusted_notional,
        "final_notional": final_notional,
        "leverage_used": implied_leverage,
        "max_leverage": max_leverage,
        "actual_risk_usd": actual_risk,
        "beta_adjusted_risk_usd": beta_adjusted_risk,
        "leverage_capped": final_notional < adjusted_notional,
    }


# =============================================================================
# INSTRUMENT SELECTION
# =============================================================================


def select_best_instrument(
    signals: List[Dict],
    account_value: float,
    leverage_tier: str = "intraday",
    min_rr_ratio: float = 1.5,
) -> Optional[Dict]:
    """
    Select the best instrument from multiple available signals.

    Ranks by effective multiplier (capital efficiency) when multiple
    valid setups are available simultaneously.

    Args:
        signals: List of signal dictionaries with 'symbol', 'entry', 'stop', 'target'
        account_value: Account equity in USD
        leverage_tier: "intraday" or "swing"
        min_rr_ratio: Minimum reward:risk ratio to consider

    Returns:
        Best signal dictionary with added efficiency metrics, or None if no valid signals

    Example:
        >>> signals = [
        ...     {'symbol': 'BTC', 'entry': 90000, 'stop': 89000, 'target': 92000},
        ...     {'symbol': 'ADA', 'entry': 0.36, 'stop': 0.35, 'target': 0.40},
        ... ]
        >>> best = select_best_instrument(signals, 1000, "intraday")
    """
    valid_signals = []

    for signal in signals:
        symbol = signal["symbol"].upper().split("-")[0]
        entry = signal["entry"]
        stop = signal["stop"]
        target = signal["target"]

        # Calculate R:R
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < min_rr_ratio:
            continue

        # Add efficiency metrics
        eff_mult = calculate_effective_multiplier(symbol, leverage_tier)

        signal_copy = signal.copy()
        signal_copy["effective_multiplier"] = eff_mult
        signal_copy["rr_ratio"] = rr_ratio
        signal_copy["beta"] = CRYPTO_BETA_TO_BTC.get(symbol, 1.0)

        valid_signals.append(signal_copy)

    if not valid_signals:
        return None

    # Sort by effective multiplier (highest first)
    valid_signals.sort(key=lambda x: x["effective_multiplier"], reverse=True)

    return valid_signals[0]
