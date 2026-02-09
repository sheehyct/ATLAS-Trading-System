"""
Spread and Z-score calculations for pairs trading.

Implements:
- Spread calculation with hedge ratio
- Rolling Z-score calculation
- Signal generation based on Z-score thresholds
- Beta-adjusted position sizing for Coinbase CFM
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import beta values from config
try:
    from crypto.config import CRYPTO_BETA_TO_BTC, LEVERAGE_TIERS
except ImportError:
    # Fallback values if config not available
    CRYPTO_BETA_TO_BTC = {
        "BTC": 1.00,
        "ETH": 1.98,
        "SOL": 1.55,
        "XRP": 1.77,
        "ADA": 2.20,
    }
    LEVERAGE_TIERS = {
        "intraday": {"BTC": 10.0, "ETH": 10.0, "SOL": 5.0, "XRP": 5.0, "ADA": 5.0},
        "swing": {"BTC": 4.0, "ETH": 4.0, "SOL": 3.0, "XRP": 3.0, "ADA": 3.0},
    }


def calculate_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    window: Optional[int] = None,
    method: str = "ols",
) -> pd.Series:
    """
    Calculate hedge ratio between two price series.
    
    The hedge ratio determines position sizes:
    - Long 1 unit of asset1, Short hedge_ratio units of asset2
    
    Args:
        price1: First price series (Y in regression)
        price2: Second price series (X in regression)
        window: Rolling window (None for expanding/full-sample)
        method: "ols" for ordinary least squares
        
    Returns:
        Series of hedge ratios (single value if window=None)
    """
    # Use log prices for stability
    log_p1 = np.log(price1)
    log_p2 = np.log(price2)
    
    if window is None:
        # Full-sample OLS
        hedge_ratio = np.polyfit(log_p2, log_p1, 1)[0]
        return pd.Series(hedge_ratio, index=price1.index)
    
    # Rolling OLS using vectorized approach
    def rolling_ols_coef(y, x, w):
        """Rolling OLS coefficient calculation."""
        result = np.full(len(y), np.nan)
        
        for i in range(w - 1, len(y)):
            y_window = y[i - w + 1:i + 1]
            x_window = x[i - w + 1:i + 1]
            
            if np.any(np.isnan(y_window)) or np.any(np.isnan(x_window)):
                continue
                
            # OLS: y = beta * x + alpha
            # beta = cov(x,y) / var(x)
            cov_xy = np.cov(x_window, y_window)[0, 1]
            var_x = np.var(x_window)
            
            if var_x > 0:
                result[i] = cov_xy / var_x
        
        return result
    
    hedge_ratios = rolling_ols_coef(log_p1.values, log_p2.values, window)
    return pd.Series(hedge_ratios, index=price1.index, name="hedge_ratio")


def calculate_spread(
    price1: pd.Series,
    price2: pd.Series,
    hedge_ratio: Optional[float] = None,
    rolling_window: Optional[int] = None,
    use_log: bool = True,
) -> pd.Series:
    """
    Calculate spread between two price series.
    
    Spread = log(P1) - hedge_ratio * log(P2)
    or
    Spread = P1 - hedge_ratio * P2  (if use_log=False)
    
    Args:
        price1: First price series
        price2: Second price series
        hedge_ratio: Fixed hedge ratio (None to calculate)
        rolling_window: Window for rolling hedge ratio (None for fixed)
        use_log: Use log prices (recommended)
        
    Returns:
        Spread series
    """
    if use_log:
        p1 = np.log(price1)
        p2 = np.log(price2)
    else:
        p1 = price1
        p2 = price2
    
    if hedge_ratio is not None:
        # Use provided fixed hedge ratio
        spread = p1 - hedge_ratio * p2
    elif rolling_window is not None:
        # Calculate rolling hedge ratio
        hr = calculate_hedge_ratio(price1, price2, window=rolling_window)
        spread = p1 - hr * p2
    else:
        # Calculate full-sample hedge ratio
        hr = calculate_hedge_ratio(price1, price2, window=None)
        spread = p1 - hr.iloc[0] * p2
    
    spread.name = "spread"
    return spread


def calculate_zscore(
    spread: pd.Series,
    window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Calculate rolling Z-score of spread.
    
    Z = (spread - mean) / std
    
    Args:
        spread: Spread series
        window: Rolling window for mean/std
        min_periods: Minimum periods for valid calculation
        
    Returns:
        Z-score series
    """
    if min_periods is None:
        min_periods = window
    
    rolling_mean = spread.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = spread.rolling(window=window, min_periods=min_periods).std()
    
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals from Z-score.
    
    Strategy logic:
    - Long spread when Z < -entry_threshold (spread undervalued)
    - Short spread when Z > entry_threshold (spread overvalued)
    - Exit when Z crosses exit_threshold toward zero
    - Stop out when Z exceeds stop_threshold
    
    Args:
        zscore: Z-score series
        entry_threshold: Entry threshold (absolute value)
        exit_threshold: Exit threshold (typically 0 or small)
        stop_threshold: Stop-loss threshold
        
    Returns:
        Tuple of (long_spread_entries, short_spread_entries)
    """
    # Long spread: Z crosses below -entry_threshold
    long_entries = (zscore.shift(1) > -entry_threshold) & (zscore <= -entry_threshold)
    
    # Short spread: Z crosses above entry_threshold
    short_entries = (zscore.shift(1) < entry_threshold) & (zscore >= entry_threshold)
    
    return long_entries, short_entries


def calculate_position_sizes(
    symbol1: str,
    symbol2: str,
    account_value: float,
    leverage_tier: str = "swing",
    hedge_ratio: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate beta-adjusted position sizes for pairs trade.
    
    Ensures dollar-neutral exposure when accounting for beta differences.
    
    Example: BTC-ETH pair with $1000 account, 4x swing leverage
    - ETH beta = 1.98, BTC beta = 1.0
    - Need ~2x more BTC notional to match ETH volatility
    
    Args:
        symbol1: First symbol (long side of spread)
        symbol2: Second symbol (short side of spread)
        account_value: Total account equity
        leverage_tier: "intraday" or "swing"
        hedge_ratio: Cointegration hedge ratio
        
    Returns:
        Dict with position sizes and notional values
    """
    # Extract base asset from symbol
    def get_base(sym):
        return sym.split("-")[0].upper()
    
    base1 = get_base(symbol1)
    base2 = get_base(symbol2)
    
    # Get beta values
    beta1 = CRYPTO_BETA_TO_BTC.get(base1, 1.0)
    beta2 = CRYPTO_BETA_TO_BTC.get(base2, 1.0)
    
    # Get leverage limits
    lev1 = LEVERAGE_TIERS.get(leverage_tier, {}).get(base1, 4.0)
    lev2 = LEVERAGE_TIERS.get(leverage_tier, {}).get(base2, 4.0)
    
    # Use minimum leverage of the two
    effective_leverage = min(lev1, lev2)
    
    # Total notional exposure
    total_notional = account_value * effective_leverage
    
    # Beta-adjusted split
    # To be market-neutral: notional1 * beta1 = notional2 * beta2
    # Combined: notional1 + notional2 = total_notional
    # Solve: notional1 = total_notional * beta2 / (beta1 + beta2)
    beta_sum = beta1 + beta2
    
    notional1 = total_notional * beta2 / beta_sum
    notional2 = total_notional * beta1 / beta_sum
    
    # Apply hedge ratio adjustment
    # hedge_ratio tells us how many units of asset2 per unit of asset1
    # Adjust notional2 by hedge_ratio
    notional2_adjusted = notional1 * abs(hedge_ratio) * beta1 / beta2
    
    # Rebalance to stay within leverage
    if notional1 + notional2_adjusted > total_notional:
        scale = total_notional / (notional1 + notional2_adjusted)
        notional1 *= scale
        notional2_adjusted *= scale
    
    return {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "notional1": notional1,
        "notional2": notional2_adjusted,
        "total_notional": notional1 + notional2_adjusted,
        "effective_leverage": effective_leverage,
        "beta1": beta1,
        "beta2": beta2,
        "beta_ratio": beta1 / beta2,
        "hedge_ratio": hedge_ratio,
    }


def calculate_spread_returns(
    price1: pd.Series,
    price2: pd.Series,
    hedge_ratio: float,
    signals: pd.Series,
) -> pd.Series:
    """
    Calculate returns from spread trading.
    
    When long spread: +1 * return1 - hedge_ratio * return2
    When short spread: -1 * return1 + hedge_ratio * return2
    
    Args:
        price1: First asset prices
        price2: Second asset prices
        hedge_ratio: Position ratio
        signals: Position series (+1 long spread, -1 short spread, 0 flat)
        
    Returns:
        Spread returns series
    """
    ret1 = price1.pct_change()
    ret2 = price2.pct_change()
    
    # Spread return = position1_return - hedge_ratio * position2_return
    # When long spread: long asset1, short asset2
    spread_returns = signals.shift(1) * (ret1 - hedge_ratio * ret2)
    
    return spread_returns


def calculate_breakeven_zscore(
    avg_spread_std: float,
    round_trip_fee_pct: float,
) -> float:
    """
    Calculate minimum Z-score move to breakeven after fees.
    
    Used to set realistic entry thresholds.
    
    Args:
        avg_spread_std: Average standard deviation of spread (in log terms)
        round_trip_fee_pct: Total round-trip fees as decimal
        
    Returns:
        Z-score threshold needed to cover fees
    """
    # Fee as percentage of position must be covered by spread move
    # spread_move = zscore * spread_std
    # breakeven: zscore * spread_std >= fee_pct
    # zscore >= fee_pct / spread_std
    
    if avg_spread_std <= 0:
        return np.inf
    
    return round_trip_fee_pct / avg_spread_std
