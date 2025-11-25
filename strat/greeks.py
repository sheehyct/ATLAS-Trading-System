"""
Options Greeks Module for ATLAS Trading System.

Session 71: Implements Black-Scholes Greeks calculation for accurate
options P/L modeling with time decay, volatility sensitivity, and
dynamic delta tracking.

Features:
- Black-Scholes option pricing
- Full Greeks calculation (delta, gamma, theta, vega, rho)
- Implied volatility estimation from historical data
- Delta range validation (0.50-0.80 per STRAT methodology)
- Time decay modeling for accurate P/L calculation

STRAT Options Requirements:
- Delta range: 0.50-0.80 (optimal balance of probability and leverage)
- Theta: -$0.02 to -$0.50/day depending on DTE
- IV percentile: Track current vs historical for vega risk

Usage:
    from strat.greeks import calculate_greeks, Greeks, estimate_iv

    greeks = calculate_greeks(
        S=450.0,      # Stock price
        K=455.0,      # Strike
        T=35/365,     # Time to expiration (years)
        r=0.05,       # Risk-free rate
        sigma=0.20,   # Implied volatility
        option_type='call'
    )
    print(f"Delta: {greeks.delta:.3f}")
    print(f"Theta: ${greeks.theta:.2f}/day")
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from scipy.stats import norm
import pandas as pd


@dataclass
class Greeks:
    """
    Container for option Greeks values.

    Attributes:
        delta: Rate of change of option price vs stock price (-1 to 1)
        gamma: Rate of change of delta vs stock price
        theta: Time decay per day (negative for long options)
        vega: Sensitivity to 1% change in IV
        rho: Sensitivity to 1% change in interest rate
        option_price: Theoretical option price
    """
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% IV change
    rho: float    # Per 1% rate change
    option_price: float

    def validate_delta_range(self, min_delta: float = 0.50, max_delta: float = 0.80) -> Tuple[bool, str]:
        """
        Validate delta is within STRAT optimal range.

        Per OPTIONS.md: Optimal delta range is 0.50-0.80 for
        balance of probability and leverage.

        Returns:
            (is_valid, message)
        """
        delta_abs = abs(self.delta)

        if delta_abs < 0.30:
            return False, f"Delta {delta_abs:.2f} too low - strike too far OTM"
        elif delta_abs < min_delta:
            return False, f"Delta {delta_abs:.2f} below optimal - consider closer strike"
        elif delta_abs <= max_delta:
            return True, f"Delta {delta_abs:.2f} in optimal range"
        elif delta_abs <= 0.90:
            return False, f"Delta {delta_abs:.2f} high - expensive, low leverage"
        else:
            return False, f"Delta {delta_abs:.2f} too high - essentially stock"


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years (e.g., 35/365 for 35 days)
        r: Risk-free interest rate (e.g., 0.05 for 5%)
        sigma: Implied volatility (e.g., 0.20 for 20%)
        option_type: 'call' or 'put'

    Returns:
        Theoretical option price
    """
    if T <= 0:
        # At expiration, return intrinsic value
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(price, 0)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call'
) -> Greeks:
    """
    Calculate all Greeks using Black-Scholes model.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annual)
        sigma: Implied volatility (annual)
        option_type: 'call' or 'put'

    Returns:
        Greeks object with all values

    Example:
        >>> greeks = calculate_greeks(450, 455, 35/365, 0.05, 0.20, 'call')
        >>> print(f"Delta: {greeks.delta:.3f}")
        Delta: 0.482
    """
    # Handle edge cases
    if T <= 0:
        # At expiration
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
            intrinsic = max(S - K, 0)
        else:
            delta = -1.0 if S < K else 0.0
            intrinsic = max(K - S, 0)
        return Greeks(
            delta=delta,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            option_price=intrinsic
        )

    if sigma <= 0:
        sigma = 0.001  # Minimum volatility

    # Calculate d1 and d2
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    # Standard normal PDF and CDF
    n_d1 = norm.pdf(d1)  # PDF for gamma/vega/theta
    N_d1 = norm.cdf(d1)  # CDF for delta
    N_d2 = norm.cdf(d2)  # CDF for delta/theta

    sqrt_T = np.sqrt(T)
    discount = np.exp(-r * T)

    # Calculate option price
    if option_type == 'call':
        price = S * N_d1 - K * discount * N_d2
    else:
        price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Delta: dV/dS
    if option_type == 'call':
        delta = N_d1
    else:
        delta = N_d1 - 1  # Negative for puts

    # Gamma: d2V/dS2 (same for calls and puts)
    gamma = n_d1 / (S * sigma * sqrt_T)

    # Theta: dV/dT (per day)
    # Note: Standard theta is annual, we convert to daily
    if option_type == 'call':
        theta_annual = (-S * n_d1 * sigma / (2 * sqrt_T)
                       - r * K * discount * N_d2)
    else:
        theta_annual = (-S * n_d1 * sigma / (2 * sqrt_T)
                       + r * K * discount * norm.cdf(-d2))
    theta_daily = theta_annual / 365  # Convert to per day

    # Vega: dV/dsigma (per 1% IV change)
    # Standard vega is per 1 unit (100%) change, we want per 1% change
    vega = S * sqrt_T * n_d1 / 100  # Divide by 100 for per 1% change

    # Rho: dV/dr (per 1% rate change)
    if option_type == 'call':
        rho = K * T * discount * N_d2 / 100  # Per 1% change
    else:
        rho = -K * T * discount * norm.cdf(-d2) / 100

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta_daily,
        vega=vega,
        rho=rho,
        option_price=max(price, 0)
    )


def estimate_iv_from_history(
    price_data: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> float:
    """
    Estimate implied volatility from historical price data.

    Uses historical volatility as a proxy for IV.
    This is a simplification - real IV comes from option prices.

    Args:
        price_data: Series of closing prices
        window: Rolling window for volatility calculation
        annualize: If True, annualize the volatility

    Returns:
        Estimated IV (e.g., 0.20 for 20% volatility)
    """
    if len(price_data) < window:
        window = len(price_data) - 1

    if window < 2:
        return 0.20  # Default 20% if insufficient data

    # Calculate log returns
    returns = np.log(price_data / price_data.shift(1)).dropna()

    # Calculate standard deviation of returns
    std = returns.rolling(window=window).std().iloc[-1]

    if np.isnan(std) or std <= 0:
        return 0.20  # Default

    # Annualize (assuming daily data, 252 trading days)
    if annualize:
        std = std * np.sqrt(252)

    return std


def validate_delta_range(
    delta: float,
    min_delta: float = 0.50,
    max_delta: float = 0.80
) -> tuple:
    """
    Validate delta is within STRAT optimal range.

    Per OPTIONS.md: Optimal delta range is 0.50-0.80 for
    balance of probability and leverage.

    Args:
        delta: Delta value to validate (can be negative for puts)
        min_delta: Minimum acceptable delta (default 0.50)
        max_delta: Maximum acceptable delta (default 0.80)

    Returns:
        (is_valid, message)
    """
    delta_abs = abs(delta)

    if delta_abs < 0.30:
        return False, f"Delta {delta_abs:.2f} too far OTM - low probability of profit"
    elif delta_abs < min_delta:
        return False, f"Delta {delta_abs:.2f} below optimal range - consider closer strike"
    elif delta_abs <= max_delta:
        return True, f"Delta {delta_abs:.2f} in optimal range (0.50-0.80)"
    elif delta_abs <= 0.90:
        return False, f"Delta {delta_abs:.2f} above optimal - expensive, lower leverage"
    else:
        return False, f"Delta {delta_abs:.2f} too high - essentially stock replacement"


def calculate_iv_percentile(
    current_iv: float,
    historical_iv: pd.Series,
    lookback_days: int = 252
) -> float:
    """
    Calculate IV percentile (rank) vs historical.

    Used to assess if IV is elevated (vega risk) or low (opportunity).

    Args:
        current_iv: Current implied volatility
        historical_iv: Series of historical IV values
        lookback_days: Number of days to look back

    Returns:
        IV percentile (0-100)
    """
    if len(historical_iv) < 2:
        return 50.0  # Default to median

    # Use most recent lookback_days
    recent_iv = historical_iv.tail(lookback_days)

    # Calculate percentile rank
    below_current = (recent_iv < current_iv).sum()
    percentile = (below_current / len(recent_iv)) * 100

    return percentile


def calculate_pnl_with_greeks(
    entry_greeks: Greeks,
    exit_greeks: Greeks,
    price_move: float,
    days_held: float,
    contracts: int = 1,
    entry_premium: float = 0.0
) -> dict:
    """
    Calculate P/L using Greeks for more accurate modeling.

    Accounts for:
    - Delta: Linear price movement
    - Gamma: Non-linear price movement (large moves)
    - Theta: Time decay

    Args:
        entry_greeks: Greeks at entry
        exit_greeks: Greeks at exit (or current)
        price_move: Underlying price change (positive = up)
        days_held: Number of days position held
        contracts: Number of option contracts
        entry_premium: Premium paid per contract

    Returns:
        Dict with P/L breakdown
    """
    # Delta P/L: delta * price_move
    delta_pnl = entry_greeks.delta * price_move

    # Gamma P/L: 0.5 * gamma * price_move^2 (convexity)
    gamma_pnl = 0.5 * entry_greeks.gamma * (price_move ** 2)

    # Theta P/L: theta * days_held (time decay, usually negative)
    theta_pnl = entry_greeks.theta * days_held

    # Total P/L per contract (in dollars, * 100 for shares per contract)
    gross_pnl = (delta_pnl + gamma_pnl + theta_pnl) * 100

    # Subtract entry premium if provided
    if entry_premium > 0:
        net_pnl = gross_pnl - entry_premium * 100
    else:
        net_pnl = gross_pnl

    # Scale by number of contracts
    total_pnl = net_pnl * contracts

    return {
        'delta_pnl': delta_pnl * 100 * contracts,
        'gamma_pnl': gamma_pnl * 100 * contracts,
        'theta_pnl': theta_pnl * 100 * contracts,
        'gross_pnl': gross_pnl * contracts,
        'net_pnl': total_pnl,
        'days_held': days_held,
        'entry_premium': entry_premium * 100 * contracts,
    }


def evaluate_trade_quality(
    greeks: Greeks,
    entry_premium: float,
    target_move: float,
    max_days: int = 30
) -> dict:
    """
    Evaluate trade quality based on Greeks.

    Checks:
    - Delta in optimal range (0.50-0.80)
    - Expected ROI meets threshold
    - Theta decay manageable

    Args:
        greeks: Current Greeks
        entry_premium: Premium per contract
        target_move: Expected price move to target
        max_days: Maximum days to hold

    Returns:
        Dict with quality assessment
    """
    # Delta validation
    delta_valid, delta_msg = greeks.validate_delta_range()

    # Expected P/L at target
    expected_pnl = greeks.delta * target_move * 100 - entry_premium * 100

    # Theta cost over holding period
    theta_cost = abs(greeks.theta) * max_days * 100

    # ROI calculation
    if entry_premium > 0:
        roi = (expected_pnl / (entry_premium * 100)) * 100
    else:
        roi = 0

    # Quality assessment
    quality_score = 0
    issues = []

    if delta_valid:
        quality_score += 1
    else:
        issues.append(delta_msg)

    if roi >= 100:
        quality_score += 1
    else:
        issues.append(f"ROI {roi:.0f}% below 100% threshold")

    if theta_cost < expected_pnl * 0.3:  # Theta < 30% of expected profit
        quality_score += 1
    else:
        issues.append(f"Theta decay ${theta_cost:.2f} may erode profits")

    return {
        'delta': greeks.delta,
        'delta_valid': delta_valid,
        'delta_msg': delta_msg,
        'expected_pnl': expected_pnl,
        'theta_cost': theta_cost,
        'roi_pct': roi,
        'quality_score': quality_score,  # 0-3
        'issues': issues,
        'recommendation': 'ACCEPT' if quality_score >= 2 else 'REVIEW' if quality_score == 1 else 'REJECT'
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Greeks Module Test")
    print("=" * 60)

    # Test 1: Basic Greeks calculation
    print("\n[TEST 1] Calculate Greeks for ATM Call...")
    S, K, T, r, sigma = 450.0, 450.0, 35/365, 0.05, 0.20
    greeks = calculate_greeks(S, K, T, r, sigma, 'call')

    print(f"  Stock: ${S}")
    print(f"  Strike: ${K}")
    print(f"  DTE: {int(T*365)} days")
    print(f"  IV: {sigma*100:.0f}%")
    print(f"  ---")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.6f}")
    print(f"  Theta: ${greeks.theta:.4f}/day")
    print(f"  Vega: ${greeks.vega:.4f}/1% IV")
    print(f"  Option Price: ${greeks.option_price:.2f}")

    # Test 2: Delta validation
    print("\n[TEST 2] Delta Range Validation...")
    valid, msg = greeks.validate_delta_range()
    print(f"  {msg} - {'PASS' if valid else 'FAIL'}")

    # Test 3: OTM Put
    print("\n[TEST 3] OTM Put Greeks...")
    greeks_put = calculate_greeks(450.0, 440.0, 35/365, 0.05, 0.20, 'put')
    print(f"  Delta: {greeks_put.delta:.4f}")
    print(f"  Theta: ${greeks_put.theta:.4f}/day")
    print(f"  Option Price: ${greeks_put.option_price:.2f}")

    # Test 4: Trade quality evaluation
    print("\n[TEST 4] Trade Quality Evaluation...")
    quality = evaluate_trade_quality(greeks, entry_premium=5.0, target_move=8.0, max_days=30)
    print(f"  Expected P/L: ${quality['expected_pnl']:.2f}")
    print(f"  Theta Cost: ${quality['theta_cost']:.2f}")
    print(f"  ROI: {quality['roi_pct']:.0f}%")
    print(f"  Quality Score: {quality['quality_score']}/3")
    print(f"  Recommendation: {quality['recommendation']}")

    # Test 5: P/L calculation with Greeks
    print("\n[TEST 5] P/L Calculation...")
    entry_greeks = calculate_greeks(450.0, 455.0, 35/365, 0.05, 0.20, 'call')
    exit_greeks = calculate_greeks(458.0, 455.0, 28/365, 0.05, 0.20, 'call')
    pnl = calculate_pnl_with_greeks(
        entry_greeks, exit_greeks,
        price_move=8.0, days_held=7, contracts=2, entry_premium=3.50
    )
    print(f"  Delta P/L: ${pnl['delta_pnl']:.2f}")
    print(f"  Gamma P/L: ${pnl['gamma_pnl']:.2f}")
    print(f"  Theta P/L: ${pnl['theta_pnl']:.2f}")
    print(f"  Entry Premium: ${pnl['entry_premium']:.2f}")
    print(f"  Net P/L: ${pnl['net_pnl']:.2f}")

    print("\n" + "=" * 60)
    print("Greeks Module Test Complete")
    print("=" * 60)
