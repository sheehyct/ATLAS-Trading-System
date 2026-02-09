"""
Ensemble Beta Calculator with Confidence Weighting.

Priority 2 component of the Beta Monitoring System.

Purpose:
- Maintain multiple beta estimates across different time windows
- Weight estimates by recent prediction accuracy
- Provide confidence intervals for position sizing decisions

Philosophy:
- Multiple time windows capture different market dynamics
- Short windows respond faster but are noisier
- Long windows are stable but lag structural changes
- Weight by ERROR, not arbitrary preference
- Wider confidence interval = reduce position size

Key Innovation:
We track prediction errors in real-time and adjust weights accordingly.
When short-term beta accurately predicted yesterday's move, it gets more weight.
When it missed badly, long-term beta gets more influence.

Session Origin: February 2, 2026
Catalyst: Feb 1-2 selloff showed static betas were ~5% off for most assets
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Time windows for beta calculation (in days)
BETA_WINDOWS: List[int] = [7, 30, 90, 365]

# Base weights for each window (before error adjustment)
# These serve as priors - longer windows get more base weight
BASE_WEIGHTS: Dict[int, float] = {
    7: 0.10,    # Very responsive, but noisy
    30: 0.25,   # Balanced responsiveness
    90: 0.30,   # Stable medium-term
    365: 0.35,  # Most stable, but may lag
}

# Error decay rate (exponential smoothing factor)
# Higher = more weight to recent errors
ERROR_DECAY: float = 0.94  # ~16-day half-life

# Confidence interval scaling
# Used to compute 95% confidence bounds
CONFIDENCE_Z_SCORE: float = 1.96


@dataclass
class BetaEstimate:
    """Single beta estimate from one time window."""
    
    window_days: int
    beta_value: float
    recent_error: float      # Most recent prediction error
    avg_error: float         # Exponentially weighted average error
    weight: float            # Current weight in ensemble
    observations: int        # Number of data points used
    last_updated: datetime


@dataclass 
class EnsembleBetaResult:
    """Combined result from ensemble beta calculation."""
    
    symbol: str
    timestamp: datetime
    
    # Ensemble beta (weighted average)
    beta_ensemble: float
    
    # Confidence interval
    beta_lower: float       # 95% CI lower bound
    beta_upper: float       # 95% CI upper bound
    confidence_width: float # Width of CI
    
    # Individual estimates
    estimates: Dict[int, BetaEstimate]  # window -> estimate
    
    # Comparison to static
    static_beta: float
    deviation_from_static: float  # Percentage deviation
    
    # Quality metrics
    ensemble_confidence: float  # 0-1, how confident are we in this estimate
    
    @property
    def is_significantly_different(self) -> bool:
        """True if ensemble differs significantly from static beta."""
        return abs(self.deviation_from_static) > 0.10  # >10% difference
    
    @property
    def position_size_multiplier(self) -> float:
        """Suggested position size adjustment based on confidence."""
        # Wide confidence interval = reduce size
        relative_width = self.confidence_width / self.beta_ensemble if self.beta_ensemble > 0 else 1.0
        
        if relative_width > 0.50:  # >50% relative width
            return 0.50
        elif relative_width > 0.25:  # >25% relative width
            return 0.75
        else:
            return 1.00


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================


def calculate_beta(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    window: Optional[int] = None,
) -> float:
    """
    Calculate beta using covariance/variance method.
    
    Beta = Cov(asset, BTC) / Var(BTC)
    
    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        window: Use last N observations (None for all)
        
    Returns:
        Beta value
    """
    if window is not None:
        asset_returns = asset_returns.iloc[-window:]
        btc_returns = btc_returns.iloc[-window:]
    
    # Align indices
    common_idx = asset_returns.index.intersection(btc_returns.index)
    asset_returns = asset_returns.loc[common_idx].dropna()
    btc_returns = btc_returns.loc[common_idx].dropna()
    
    if len(asset_returns) < 5:
        return np.nan
    
    covariance = np.cov(asset_returns.values, btc_returns.values)[0, 1]
    btc_variance = np.var(btc_returns.values)
    
    if btc_variance == 0:
        return 1.0
    
    return covariance / btc_variance


def calculate_rolling_beta(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate rolling beta over time.
    
    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        window: Rolling window size
        
    Returns:
        Series of beta values
    """
    # Align indices
    common_idx = asset_returns.index.intersection(btc_returns.index)
    asset_returns = asset_returns.loc[common_idx]
    btc_returns = btc_returns.loc[common_idx]
    
    covariance = asset_returns.rolling(window).cov(btc_returns)
    variance = btc_returns.rolling(window).var()
    
    beta_series = covariance / variance
    beta_series.name = f'beta_{window}d'
    
    return beta_series


def calculate_prediction_error(
    actual_asset_return: float,
    btc_return: float,
    predicted_beta: float,
) -> float:
    """
    Calculate prediction error for a beta estimate.
    
    Error = actual_move - predicted_move
    Where predicted_move = btc_return * beta
    
    Args:
        actual_asset_return: Actual asset return
        btc_return: BTC return
        predicted_beta: Beta used for prediction
        
    Returns:
        Absolute prediction error
    """
    if btc_return == 0:
        return 0.0
    
    predicted_return = btc_return * predicted_beta
    error = abs(actual_asset_return - predicted_return)
    
    return error


def update_error_ewma(
    current_ewma: float,
    new_error: float,
    decay: float = ERROR_DECAY,
) -> float:
    """
    Update exponentially weighted moving average of errors.
    
    Args:
        current_ewma: Current EWMA value
        new_error: New error observation
        decay: Decay factor (0-1)
        
    Returns:
        Updated EWMA
    """
    if np.isnan(current_ewma):
        return new_error
    
    return decay * current_ewma + (1 - decay) * new_error


def calculate_weights_from_errors(
    errors: Dict[int, float],
    base_weights: Dict[int, float] = BASE_WEIGHTS,
) -> Dict[int, float]:
    """
    Calculate weights inversely proportional to errors.
    
    Lower error = higher weight.
    Uses inverse-error weighting with base weight prior.
    
    Args:
        errors: Dictionary of window -> EWMA error
        base_weights: Base/prior weights
        
    Returns:
        Dictionary of window -> adjusted weight
    """
    # Handle case where all errors are 0 or NaN
    valid_errors = {w: e for w, e in errors.items() if not np.isnan(e) and e > 0}
    
    if not valid_errors:
        return base_weights.copy()
    
    # Calculate inverse errors (add small epsilon to avoid division by zero)
    epsilon = 0.001
    inverse_errors = {w: 1.0 / (e + epsilon) for w, e in valid_errors.items()}
    
    # Combine with base weights
    combined = {}
    for window in base_weights:
        if window in inverse_errors:
            # Product of base weight and inverse error
            combined[window] = base_weights[window] * inverse_errors[window]
        else:
            # Use base weight if no error data
            combined[window] = base_weights[window]
    
    # Normalize to sum to 1
    total = sum(combined.values())
    if total > 0:
        return {w: v / total for w, v in combined.items()}
    
    return base_weights.copy()


def calculate_ensemble_beta(
    betas: Dict[int, float],
    weights: Dict[int, float],
) -> float:
    """
    Calculate weighted average beta.
    
    Args:
        betas: Dictionary of window -> beta value
        weights: Dictionary of window -> weight
        
    Returns:
        Ensemble beta estimate
    """
    valid_betas = {w: b for w, b in betas.items() if not np.isnan(b)}
    
    if not valid_betas:
        return np.nan
    
    # Renormalize weights for valid betas only
    valid_weights = {w: weights.get(w, 0) for w in valid_betas}
    total_weight = sum(valid_weights.values())
    
    if total_weight == 0:
        return np.mean(list(valid_betas.values()))
    
    ensemble = sum(b * valid_weights[w] / total_weight for w, b in valid_betas.items())
    
    return ensemble


def calculate_confidence_interval(
    betas: Dict[int, float],
    weights: Dict[int, float],
    z_score: float = CONFIDENCE_Z_SCORE,
) -> Tuple[float, float]:
    """
    Calculate confidence interval for ensemble beta.
    
    Uses weighted standard deviation to estimate uncertainty.
    
    Args:
        betas: Dictionary of window -> beta value
        weights: Dictionary of window -> weight
        z_score: Z-score for confidence level (1.96 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    valid_betas = {w: b for w, b in betas.items() if not np.isnan(b)}
    
    if len(valid_betas) < 2:
        # Not enough data for CI
        mean_beta = np.mean(list(valid_betas.values())) if valid_betas else 1.0
        return (mean_beta * 0.8, mean_beta * 1.2)  # Default 20% bands
    
    ensemble = calculate_ensemble_beta(betas, weights)
    
    # Weighted variance
    beta_values = np.array(list(valid_betas.values()))
    weight_values = np.array([weights.get(w, 1.0/len(valid_betas)) for w in valid_betas])
    weight_values = weight_values / weight_values.sum()
    
    weighted_variance = np.sum(weight_values * (beta_values - ensemble) ** 2)
    weighted_std = np.sqrt(weighted_variance)
    
    # Add error-based uncertainty
    margin = z_score * weighted_std
    
    return (ensemble - margin, ensemble + margin)


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================


class EnsembleBetaTracker:
    """
    Track ensemble beta estimates for crypto assets.
    
    Maintains multiple beta calculations across time windows,
    tracks prediction errors, and produces confidence-weighted estimates.
    
    Usage:
        tracker = EnsembleBetaTracker()
        tracker.load_data(asset_prices, btc_prices)
        result = tracker.get_ensemble_beta('ADA')
        
        print(f"ADA Beta: {result.beta_ensemble:.2f} +/- {result.confidence_width/2:.2f}")
        if result.is_significantly_different:
            print(f"Warning: Differs from static by {result.deviation_from_static:.1%}")
    """
    
    def __init__(
        self,
        windows: List[int] = BETA_WINDOWS,
        base_weights: Dict[int, float] = BASE_WEIGHTS,
        static_betas: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the tracker.
        
        Args:
            windows: List of window sizes in days
            base_weights: Base/prior weights for each window
            static_betas: Static beta values for comparison
        """
        self.windows = windows
        self.base_weights = base_weights
        
        # Default static betas from config if not provided
        if static_betas is None:
            try:
                from crypto.config import CRYPTO_BETA_TO_BTC
                static_betas = CRYPTO_BETA_TO_BTC
            except ImportError:
                static_betas = {"BTC": 1.0, "ETH": 2.0, "ADA": 2.2, "XRP": 1.8, "SOL": 1.5}
        
        self.static_betas = static_betas
        
        # Internal storage
        self._prices: Dict[str, pd.Series] = {}
        self._returns: Dict[str, pd.Series] = {}
        self._error_ewmas: Dict[str, Dict[int, float]] = {}  # symbol -> window -> EWMA error
        self._last_betas: Dict[str, Dict[int, float]] = {}   # symbol -> window -> last beta
    
    def load_data(
        self,
        asset_prices: pd.Series,
        btc_prices: pd.Series,
        symbol: str,
    ) -> None:
        """
        Load price data for an asset.
        
        Args:
            asset_prices: Asset price series
            btc_prices: BTC price series
            symbol: Asset symbol
        """
        symbol = symbol.upper().split('-')[0]
        
        # Align indices
        common_idx = asset_prices.index.intersection(btc_prices.index)
        asset_prices = asset_prices.loc[common_idx]
        btc_prices = btc_prices.loc[common_idx]
        
        self._prices[symbol] = asset_prices
        self._prices['BTC'] = btc_prices
        
        self._returns[symbol] = asset_prices.pct_change()
        self._returns['BTC'] = btc_prices.pct_change()
        
        # Initialize error tracking
        if symbol not in self._error_ewmas:
            self._error_ewmas[symbol] = {w: np.nan for w in self.windows}
            self._last_betas[symbol] = {w: np.nan for w in self.windows}
        
        logger.info(f"Loaded data for {symbol}: {len(asset_prices)} observations")
    
    def update_errors(self, symbol: str) -> None:
        """
        Update prediction error tracking for a symbol.
        
        Call this daily (or after each new observation) to track
        how well each beta window predicted the latest move.
        
        Args:
            symbol: Asset symbol
        """
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._returns:
            logger.warning(f"No data for {symbol}")
            return
        
        asset_returns = self._returns[symbol]
        btc_returns = self._returns['BTC']
        
        # Get latest returns
        if len(asset_returns) < 2 or len(btc_returns) < 2:
            return
        
        latest_asset_return = asset_returns.iloc[-1]
        latest_btc_return = btc_returns.iloc[-1]
        
        # Calculate and update errors for each window
        for window in self.windows:
            if window not in self._last_betas[symbol]:
                continue
            
            last_beta = self._last_betas[symbol][window]
            
            if np.isnan(last_beta):
                # Calculate beta now if we don't have a previous one
                last_beta = calculate_beta(
                    asset_returns.iloc[:-1],  # Exclude latest
                    btc_returns.iloc[:-1],
                    window=window
                )
                self._last_betas[symbol][window] = last_beta
                continue
            
            # Calculate prediction error
            error = calculate_prediction_error(
                latest_asset_return,
                latest_btc_return,
                last_beta
            )
            
            # Update EWMA
            current_ewma = self._error_ewmas[symbol].get(window, np.nan)
            self._error_ewmas[symbol][window] = update_error_ewma(current_ewma, error)
            
            # Update beta for next prediction
            new_beta = calculate_beta(asset_returns, btc_returns, window=window)
            self._last_betas[symbol][window] = new_beta
    
    def get_ensemble_beta(self, symbol: str) -> EnsembleBetaResult:
        """
        Get ensemble beta estimate with confidence interval.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            EnsembleBetaResult with ensemble estimate and metrics
        """
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._returns:
            raise ValueError(f"No data for {symbol}. Call load_data() first.")
        
        asset_returns = self._returns[symbol]
        btc_returns = self._returns['BTC']
        
        # Calculate betas for each window
        betas: Dict[int, float] = {}
        estimates: Dict[int, BetaEstimate] = {}
        
        for window in self.windows:
            beta = calculate_beta(asset_returns, btc_returns, window=window)
            betas[window] = beta
            
            # Get error metrics
            error_ewma = self._error_ewmas.get(symbol, {}).get(window, np.nan)
            recent_error = 0.0  # Would need to track most recent separately
            
            # Count observations
            obs = min(len(asset_returns), window)
            
            estimates[window] = BetaEstimate(
                window_days=window,
                beta_value=beta,
                recent_error=recent_error,
                avg_error=error_ewma if not np.isnan(error_ewma) else 0.0,
                weight=0.0,  # Will be set below
                observations=obs,
                last_updated=datetime.now(),
            )
        
        # Calculate weights from errors
        errors = self._error_ewmas.get(symbol, {})
        weights = calculate_weights_from_errors(errors, self.base_weights)
        
        # Update weights in estimates
        for window in estimates:
            estimates[window].weight = weights.get(window, 0.0)
        
        # Calculate ensemble
        ensemble_beta = calculate_ensemble_beta(betas, weights)
        
        # Calculate confidence interval
        ci_lower, ci_upper = calculate_confidence_interval(betas, weights)
        ci_width = ci_upper - ci_lower
        
        # Compare to static
        static_beta = self.static_betas.get(symbol, 1.0)
        deviation = (ensemble_beta - static_beta) / static_beta if static_beta > 0 else 0.0
        
        # Calculate confidence metric (inverse of relative CI width)
        relative_ci = ci_width / ensemble_beta if ensemble_beta > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - relative_ci))
        
        return EnsembleBetaResult(
            symbol=symbol,
            timestamp=datetime.now(),
            beta_ensemble=ensemble_beta,
            beta_lower=ci_lower,
            beta_upper=ci_upper,
            confidence_width=ci_width,
            estimates=estimates,
            static_beta=static_beta,
            deviation_from_static=deviation,
            ensemble_confidence=confidence,
        )
    
    def get_all_betas(self) -> Dict[str, EnsembleBetaResult]:
        """
        Get ensemble betas for all loaded symbols.
        
        Returns:
            Dictionary of symbol -> EnsembleBetaResult
        """
        results = {}
        for symbol in self._prices:
            if symbol == 'BTC':
                continue
            try:
                results[symbol] = self.get_ensemble_beta(symbol)
            except Exception as e:
                logger.warning(f"Failed to calculate beta for {symbol}: {e}")
        
        return results
    
    def format_beta_report(self) -> str:
        """
        Format all beta estimates as readable report.
        
        Returns:
            Formatted string report
        """
        lines = [
            "=" * 70,
            "ENSEMBLE BETA REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            f"{'Symbol':<8} {'Static':>8} {'Ensemble':>10} {'95% CI':>16} {'Deviation':>10}",
            "-" * 70,
        ]
        
        for symbol, result in self.get_all_betas().items():
            ci_str = f"[{result.beta_lower:.2f}, {result.beta_upper:.2f}]"
            dev_str = f"{result.deviation_from_static:+.1%}"
            flag = " !!" if result.is_significantly_different else ""
            
            lines.append(
                f"{symbol:<8} {result.static_beta:>8.2f} {result.beta_ensemble:>10.2f} "
                f"{ci_str:>16} {dev_str:>10}{flag}"
            )
        
        lines.append("")
        lines.append("!! = Significant deviation from static beta (>10%)")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_ensemble_beta(
    asset_prices: pd.Series,
    btc_prices: pd.Series,
    symbol: str = "ASSET",
    static_beta: float = 1.0,
) -> EnsembleBetaResult:
    """
    Quick one-shot ensemble beta calculation.
    
    Args:
        asset_prices: Asset price series
        btc_prices: BTC price series
        symbol: Asset symbol
        static_beta: Static beta for comparison
        
    Returns:
        EnsembleBetaResult
    """
    tracker = EnsembleBetaTracker(static_betas={symbol: static_beta})
    tracker.load_data(asset_prices, btc_prices, symbol)
    return tracker.get_ensemble_beta(symbol)


def get_beta_adjustment_factor(
    ensemble_result: EnsembleBetaResult,
) -> float:
    """
    Get multiplicative adjustment to apply to static beta.
    
    Use this to convert static beta to dynamic beta for position sizing.
    
    Args:
        ensemble_result: Result from ensemble calculation
        
    Returns:
        Adjustment factor (multiply static beta by this)
        
    Example:
        >>> factor = get_beta_adjustment_factor(result)
        >>> adjusted_beta = static_beta * factor
    """
    if ensemble_result.static_beta == 0:
        return 1.0
    
    return ensemble_result.beta_ensemble / ensemble_result.static_beta
