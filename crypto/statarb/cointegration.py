"""
Cointegration testing for crypto pairs trading.

Implements:
- Engle-Granger two-step test
- Johansen test (for multi-asset analysis)
- Half-life calculation for mean reversion speed
- Batch testing across symbol pairs

References:
- Engle & Granger (1987): "Co-integration and Error Correction"
- Johansen (1991): "Estimation and Hypothesis Testing of Cointegration Vectors"
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Results from cointegration test."""
    
    symbol1: str
    symbol2: str
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]  # 1%, 5%, 10%
    hedge_ratio: float  # OLS coefficient
    half_life: float  # Mean reversion half-life in bars
    method: str  # "engle_granger" or "johansen"
    
    @property
    def strength(self) -> str:
        """Cointegration strength based on p-value."""
        if self.p_value < 0.01:
            return "strong"
        elif self.p_value < 0.05:
            return "moderate"
        elif self.p_value < 0.10:
            return "weak"
        return "none"


def engle_granger_test(
    price1: pd.Series,
    price2: pd.Series,
    significance_level: float = 0.05,
    use_log_prices: bool = True,
) -> CointegrationResult:
    """
    Engle-Granger two-step cointegration test.
    
    Step 1: Regress price1 on price2 to get hedge ratio (OLS)
    Step 2: Test residuals for stationarity (ADF test)
    
    Args:
        price1: First price series (will be Y in regression)
        price2: Second price series (will be X in regression)
        significance_level: P-value threshold for cointegration
        use_log_prices: Use log prices for better stability
        
    Returns:
        CointegrationResult with test statistics and hedge ratio
    """
    if len(price1) != len(price2):
        raise ValueError("Price series must have same length")
    
    if len(price1) < 50:
        raise ValueError("Need at least 50 observations for reliable test")
    
    # Use log prices for better statistical properties
    if use_log_prices:
        p1 = np.log(price1)
        p2 = np.log(price2)
    else:
        p1 = price1.values
        p2 = price2.values
    
    # Run cointegration test (statsmodels handles both steps)
    test_stat, p_value, crit_values = coint(p1, p2)
    
    # Calculate hedge ratio from OLS regression
    # p1 = beta * p2 + epsilon
    hedge_ratio = np.polyfit(p2, p1, 1)[0]
    
    # Calculate spread and half-life
    spread = p1 - hedge_ratio * p2
    half_life = calculate_half_life(spread)
    
    # Extract critical values (1%, 5%, 10%)
    crit_dict = {
        "1%": crit_values[0],
        "5%": crit_values[1],
        "10%": crit_values[2],
    }
    
    is_cointegrated = p_value < significance_level
    
    return CointegrationResult(
        symbol1=price1.name if hasattr(price1, 'name') else "asset1",
        symbol2=price2.name if hasattr(price2, 'name') else "asset2",
        is_cointegrated=is_cointegrated,
        p_value=p_value,
        test_statistic=test_stat,
        critical_values=crit_dict,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
        method="engle_granger",
    )


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Dict[str, any]:
    """
    Johansen cointegration test for multiple time series.
    
    More powerful than Engle-Granger for multi-asset analysis.
    Returns optimal hedge ratios as eigenvectors.
    
    Args:
        prices: DataFrame with price series as columns
        det_order: Deterministic trend order (-1: no constant, 0: constant)
        k_ar_diff: Number of lagged differences in VAR model
        
    Returns:
        Dictionary with test results and optimal hedge ratios
    """
    if prices.shape[1] < 2:
        raise ValueError("Need at least 2 price series")
    
    # Use log prices
    log_prices = np.log(prices)
    
    result = coint_johansen(log_prices, det_order, k_ar_diff)
    
    # Trace statistic test
    trace_stat = result.lr1
    trace_crit = result.cvt  # Critical values at 90%, 95%, 99%
    
    # Maximum eigenvalue test  
    max_eig_stat = result.lr2
    max_eig_crit = result.cvm
    
    # Cointegrating vectors (eigenvectors)
    eigenvecs = result.evec
    
    # Eigenvalues
    eigenvals = result.eig
    
    # Determine rank (number of cointegrating relationships)
    # Compare trace statistics to 95% critical values
    n_coint = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i, 1]:  # 95% level is column 1
            n_coint = len(trace_stat) - i
            break
    
    return {
        "n_cointegrating_relations": n_coint,
        "trace_statistics": trace_stat,
        "trace_critical_values": trace_crit,
        "max_eigenvalue_statistics": max_eig_stat,
        "max_eigenvalue_critical_values": max_eig_crit,
        "eigenvectors": eigenvecs,
        "eigenvalues": eigenvals,
        "optimal_hedge_ratios": eigenvecs[:, 0] if n_coint > 0 else None,
    }


def calculate_half_life(spread: np.ndarray) -> float:
    """
    Calculate mean-reversion half-life using Ornstein-Uhlenbeck model.
    
    The spread is modeled as: dS = theta * (mu - S) * dt + sigma * dW
    Half-life = ln(2) / theta
    
    Estimated via OLS on lagged spread:
    S[t] - S[t-1] = alpha + beta * S[t-1] + epsilon
    theta = -ln(1 + beta)
    
    Args:
        spread: Spread series (can be numpy array or pandas Series)
        
    Returns:
        Half-life in bars (same timeframe as input data)
    """
    spread = np.asarray(spread)
    
    # Remove NaN values
    spread = spread[~np.isnan(spread)]
    
    if len(spread) < 10:
        return np.nan
    
    # Lagged regression: delta_S = alpha + beta * S_lag
    spread_lag = spread[:-1]
    spread_delta = spread[1:] - spread[:-1]
    
    # OLS regression
    # Add constant term
    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    
    try:
        beta = np.linalg.lstsq(X, spread_delta, rcond=None)[0]
        theta = -np.log(1 + beta[1])
        
        if theta <= 0:
            return np.inf  # No mean reversion
        
        half_life = np.log(2) / theta
        return half_life
        
    except Exception as e:
        logger.warning("Half-life calculation failed: %s", e)
        return np.nan


def calculate_half_life_alt(spread: np.ndarray) -> float:
    """
    Alternative half-life calculation using direct formula.
    
    More numerically stable for some datasets.
    """
    spread = np.asarray(spread)
    spread = spread[~np.isnan(spread)]
    
    if len(spread) < 10:
        return np.nan
    
    spread_lag = spread[:-1]
    spread_ret = spread[1:] - spread_lag
    
    # Direct regression coefficient
    try:
        import statsmodels.api as sm
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, X).fit()
        theta = -model.params[1]
        
        if theta <= 0:
            return np.inf
            
        return np.log(2) / theta
        
    except Exception:
        return calculate_half_life(spread)


def test_all_pairs(
    prices: pd.DataFrame,
    significance_level: float = 0.05,
    min_half_life: float = 1.0,
    max_half_life: float = 42.0,
) -> List[CointegrationResult]:
    """
    Test cointegration for all pairs in a price DataFrame.
    
    Args:
        prices: DataFrame with symbols as columns, datetime index
        significance_level: P-value threshold
        min_half_life: Minimum half-life in bars (filter too fast)
        max_half_life: Maximum half-life in bars (filter too slow)
        
    Returns:
        List of CointegrationResult, sorted by p-value (best first)
    """
    symbols = prices.columns.tolist()
    results = []
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            try:
                result = engle_granger_test(
                    prices[sym1],
                    prices[sym2],
                    significance_level=1.0,  # Get all, filter later
                    use_log_prices=True,
                )
                
                # Apply filters
                if result.p_value <= significance_level:
                    if min_half_life <= result.half_life <= max_half_life:
                        results.append(result)
                        
            except Exception as e:
                logger.warning("Failed to test %s-%s: %s", sym1, sym2, e)
                continue
    
    # Sort by p-value (lowest = strongest cointegration)
    results.sort(key=lambda x: x.p_value)
    
    return results


def validate_cointegration_stability(
    price1: pd.Series,
    price2: pd.Series,
    window_size: int = 252,
    step_size: int = 21,
) -> pd.DataFrame:
    """
    Rolling cointegration test to validate stability over time.
    
    Critical for production - cointegration can break down.
    
    Args:
        price1: First price series
        price2: Second price series
        window_size: Rolling window in bars
        step_size: Step between windows (for efficiency)
        
    Returns:
        DataFrame with rolling p-values, hedge ratios, half-lives
    """
    results = []
    n = len(price1)
    
    for start in range(0, n - window_size, step_size):
        end = start + window_size
        
        try:
            result = engle_granger_test(
                price1.iloc[start:end],
                price2.iloc[start:end],
                significance_level=1.0,  # Get raw p-value
            )
            
            results.append({
                "end_date": price1.index[end - 1],
                "p_value": result.p_value,
                "hedge_ratio": result.hedge_ratio,
                "half_life": result.half_life,
                "is_cointegrated_5pct": result.p_value < 0.05,
            })
            
        except Exception as e:
            logger.warning("Rolling test failed at %d: %s", start, e)
            continue
    
    return pd.DataFrame(results).set_index("end_date")
