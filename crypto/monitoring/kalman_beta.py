"""
Kalman Filter Beta Estimator - Priority 4.

Implements a state-space model for time-varying beta estimation:
- State equation: Beta_t = Beta_{t-1} + w_t (random walk)
- Observation equation: R_asset = Beta_t * R_BTC + v_t

Provides:
- Real-time beta estimate
- Confidence interval that widens during uncertainty
- Adapts faster during high-volatility periods

Session Origin: February 3, 2026
Philosophy: Adaptive measurement, not predictive modeling
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class KalmanBetaEstimate:
    """Result from a single Kalman filter update."""
    
    timestamp: datetime
    symbol: str
    
    # Beta estimates
    beta: float                    # Current filtered beta estimate
    beta_prior: float              # Prior (predicted) beta before update
    
    # Uncertainty
    variance: float                # Posterior variance of beta estimate
    std: float                     # Posterior standard deviation
    confidence_lower: float        # 95% CI lower bound
    confidence_upper: float        # 95% CI upper bound
    
    # Kalman diagnostics
    kalman_gain: float             # How much we trusted the observation
    innovation: float              # Prediction error (actual - expected)
    innovation_variance: float     # Variance of innovation
    
    # Comparison to static
    static_beta: Optional[float]
    deviation_from_static: Optional[float]
    
    @property
    def confidence_width(self) -> float:
        """Width of 95% confidence interval."""
        return self.confidence_upper - self.confidence_lower
    
    @property
    def is_uncertain(self) -> bool:
        """True if confidence interval is wide (>30% of beta)."""
        if self.beta == 0:
            return True
        return self.confidence_width / abs(self.beta) > 0.30
    
    @property
    def position_size_multiplier(self) -> float:
        """
        Suggested position size multiplier based on uncertainty.
        
        Full size (1.0) when confident, reduced when uncertain.
        """
        if self.beta == 0:
            return 0.5
        
        relative_width = self.confidence_width / abs(self.beta)
        
        if relative_width > 0.50:
            return 0.25  # Very uncertain
        elif relative_width > 0.30:
            return 0.50  # Uncertain
        elif relative_width > 0.20:
            return 0.75  # Somewhat uncertain
        else:
            return 1.00  # Confident


@dataclass
class KalmanBetaHistory:
    """Full history of Kalman filter estimates for an asset."""
    
    symbol: str
    estimates: List[KalmanBetaEstimate]
    
    @property
    def latest(self) -> Optional[KalmanBetaEstimate]:
        """Most recent estimate."""
        return self.estimates[-1] if self.estimates else None
    
    @property
    def beta_series(self) -> pd.Series:
        """Time series of beta estimates."""
        if not self.estimates:
            return pd.Series(dtype=float)
        return pd.Series(
            [e.beta for e in self.estimates],
            index=[e.timestamp for e in self.estimates],
            name=f"{self.symbol}_kalman_beta"
        )
    
    @property
    def confidence_width_series(self) -> pd.Series:
        """Time series of confidence interval widths."""
        if not self.estimates:
            return pd.Series(dtype=float)
        return pd.Series(
            [e.confidence_width for e in self.estimates],
            index=[e.timestamp for e in self.estimates],
            name=f"{self.symbol}_ci_width"
        )


# =============================================================================
# KALMAN FILTER IMPLEMENTATION
# =============================================================================


class KalmanBetaFilter:
    """
    Kalman filter for time-varying beta estimation.
    
    State-space model:
    - State: beta_t (scalar)
    - State transition: beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
    - Observation: r_asset,t = beta_t * r_btc,t + v_t, v_t ~ N(0, R)
    
    Parameters:
    - Q (process_variance): How much beta can change each period
    - R (observation_variance): Noise in the return relationship
    
    Usage:
        kf = KalmanBetaFilter(symbol="ETH", initial_beta=1.98)
        
        for t in range(len(returns)):
            result = kf.update(
                asset_return=eth_returns[t],
                btc_return=btc_returns[t],
                timestamp=timestamps[t]
            )
            print(f"Beta: {result.beta:.3f} [{result.confidence_lower:.2f}, {result.confidence_upper:.2f}]")
    """
    
    def __init__(
        self,
        symbol: str,
        initial_beta: float = 1.0,
        initial_variance: float = 0.25,
        process_variance: float = 0.001,
        observation_variance: float = 0.0001,
        static_beta: Optional[float] = None,
    ):
        """
        Initialize Kalman filter for beta estimation.
        
        Args:
            symbol: Asset symbol (e.g., "ETH")
            initial_beta: Starting beta estimate
            initial_variance: Initial uncertainty in beta (P_0)
            process_variance: Q - how much beta can drift per period
            observation_variance: R - noise in return relationship
            static_beta: Reference static beta for comparison
        """
        self.symbol = symbol.upper()
        self.static_beta = static_beta or initial_beta
        
        # Kalman filter state
        self._beta = initial_beta        # x̂ (state estimate)
        self._variance = initial_variance  # P (state covariance)
        
        # Filter parameters
        self._Q = process_variance        # Process noise variance
        self._R = observation_variance    # Observation noise variance
        
        # History
        self._history: List[KalmanBetaEstimate] = []
        self._n_updates = 0
    
    @property
    def beta(self) -> float:
        """Current beta estimate."""
        return self._beta
    
    @property
    def variance(self) -> float:
        """Current variance (uncertainty) in beta estimate."""
        return self._variance
    
    @property
    def std(self) -> float:
        """Current standard deviation of beta estimate."""
        return np.sqrt(self._variance)
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval for beta."""
        margin = 1.96 * self.std
        return (self._beta - margin, self._beta + margin)
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict step: propagate state forward.
        
        Returns:
            (predicted_beta, predicted_variance)
        """
        # State transition: beta_t = beta_{t-1} (random walk)
        beta_prior = self._beta
        
        # Variance grows by process noise
        variance_prior = self._variance + self._Q
        
        return beta_prior, variance_prior
    
    def update(
        self,
        asset_return: float,
        btc_return: float,
        timestamp: Optional[datetime] = None,
    ) -> KalmanBetaEstimate:
        """
        Update step: incorporate new observation.
        
        Args:
            asset_return: Asset return for this period
            btc_return: BTC return for this period
            timestamp: Optional timestamp
            
        Returns:
            KalmanBetaEstimate with updated values
        """
        timestamp = timestamp or datetime.now()
        
        # Skip update if BTC return is too small (division issues)
        if abs(btc_return) < 1e-8:
            # Return current state unchanged
            ci = self.confidence_interval
            estimate = KalmanBetaEstimate(
                timestamp=timestamp,
                symbol=self.symbol,
                beta=self._beta,
                beta_prior=self._beta,
                variance=self._variance,
                std=self.std,
                confidence_lower=ci[0],
                confidence_upper=ci[1],
                kalman_gain=0.0,
                innovation=0.0,
                innovation_variance=0.0,
                static_beta=self.static_beta,
                deviation_from_static=(self._beta - self.static_beta) / self.static_beta if self.static_beta else None,
            )
            self._history.append(estimate)
            return estimate
        
        # PREDICT STEP
        beta_prior, variance_prior = self.predict()
        
        # OBSERVATION MODEL
        # H_t = btc_return (time-varying observation matrix)
        # Expected observation: y_expected = H_t * beta_prior
        H = btc_return
        y_expected = H * beta_prior
        
        # Innovation (prediction error)
        innovation = asset_return - y_expected
        
        # Innovation variance: S = H * P * H' + R
        innovation_variance = H * variance_prior * H + self._R
        
        # Kalman gain: K = P * H' * S^(-1)
        if abs(innovation_variance) < 1e-12:
            kalman_gain = 0.0
        else:
            kalman_gain = variance_prior * H / innovation_variance
        
        # UPDATE STEP
        # New state estimate: x̂ = x̂_prior + K * innovation
        self._beta = beta_prior + kalman_gain * innovation
        
        # New variance: P = (1 - K * H) * P_prior
        self._variance = (1 - kalman_gain * H) * variance_prior
        
        # Ensure variance stays positive
        self._variance = max(self._variance, 1e-10)
        
        self._n_updates += 1
        
        # Build result
        ci = self.confidence_interval
        estimate = KalmanBetaEstimate(
            timestamp=timestamp,
            symbol=self.symbol,
            beta=self._beta,
            beta_prior=beta_prior,
            variance=self._variance,
            std=self.std,
            confidence_lower=ci[0],
            confidence_upper=ci[1],
            kalman_gain=kalman_gain,
            innovation=innovation,
            innovation_variance=innovation_variance,
            static_beta=self.static_beta,
            deviation_from_static=(self._beta - self.static_beta) / self.static_beta if self.static_beta else None,
        )
        
        self._history.append(estimate)
        return estimate
    
    def run_filter(
        self,
        asset_returns: pd.Series,
        btc_returns: pd.Series,
    ) -> KalmanBetaHistory:
        """
        Run filter over entire return series.
        
        Args:
            asset_returns: Asset return series
            btc_returns: BTC return series (must align)
            
        Returns:
            KalmanBetaHistory with all estimates
        """
        # Align indices
        common_idx = asset_returns.index.intersection(btc_returns.index)
        asset_ret = asset_returns.loc[common_idx]
        btc_ret = btc_returns.loc[common_idx]
        
        for i, (ts, asset_r) in enumerate(asset_ret.items()):
            btc_r = btc_ret.iloc[i]
            self.update(asset_r, btc_r, timestamp=ts)
        
        return self.get_history()
    
    def get_history(self) -> KalmanBetaHistory:
        """Get full history of estimates."""
        return KalmanBetaHistory(
            symbol=self.symbol,
            estimates=self._history.copy()
        )
    
    def reset(
        self,
        initial_beta: Optional[float] = None,
        initial_variance: Optional[float] = None,
    ):
        """Reset filter state."""
        if initial_beta is not None:
            self._beta = initial_beta
        if initial_variance is not None:
            self._variance = initial_variance
        self._history = []
        self._n_updates = 0


# =============================================================================
# MULTI-ASSET TRACKER
# =============================================================================


class KalmanBetaTracker:
    """
    Track Kalman-filtered betas for multiple assets.
    
    Maintains separate Kalman filters for each asset, all referenced to BTC.
    
    Usage:
        tracker = KalmanBetaTracker()
        tracker.load_data(prices_df)  # Must include 'BTC'
        
        # Get current estimate for single asset
        eth_estimate = tracker.get_estimate("ETH")
        print(f"ETH Kalman Beta: {eth_estimate.beta:.3f}")
        
        # Get all estimates
        all_estimates = tracker.get_all_estimates()
    """
    
    def __init__(
        self,
        static_betas: Optional[dict] = None,
        process_variance: float = 0.001,
        observation_variance: float = 0.0001,
        initial_variance: float = 0.25,
    ):
        """
        Initialize tracker.
        
        Args:
            static_betas: Dict of symbol -> static beta (for comparison)
            process_variance: Q parameter for all filters
            observation_variance: R parameter for all filters
            initial_variance: Initial P_0 for all filters
        """
        # Load static betas from config if not provided
        if static_betas is None:
            try:
                from crypto.config import CRYPTO_BETA_TO_BTC
                static_betas = CRYPTO_BETA_TO_BTC
            except ImportError:
                static_betas = {
                    "BTC": 1.00,
                    "ETH": 1.98,
                    "SOL": 1.55,
                    "XRP": 1.77,
                    "ADA": 2.20,
                }
        
        self.static_betas = static_betas
        self._process_variance = process_variance
        self._observation_variance = observation_variance
        self._initial_variance = initial_variance
        
        # Kalman filters per asset
        self._filters: dict[str, KalmanBetaFilter] = {}
        
        # Data
        self._returns: Optional[pd.DataFrame] = None
        self._btc_returns: Optional[pd.Series] = None
    
    def _get_or_create_filter(self, symbol: str) -> KalmanBetaFilter:
        """Get existing filter or create new one."""
        symbol = symbol.upper()
        
        if symbol not in self._filters:
            static_beta = self.static_betas.get(symbol, 1.0)
            self._filters[symbol] = KalmanBetaFilter(
                symbol=symbol,
                initial_beta=static_beta,
                initial_variance=self._initial_variance,
                process_variance=self._process_variance,
                observation_variance=self._observation_variance,
                static_beta=static_beta,
            )
        
        return self._filters[symbol]
    
    def load_data(self, prices: pd.DataFrame) -> None:
        """
        Load price data and run filters.
        
        Args:
            prices: DataFrame with symbols as columns, datetime index
                   Must include 'BTC' column
        """
        if 'BTC' not in prices.columns:
            raise ValueError("Prices DataFrame must include 'BTC' column")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        self._returns = returns
        self._btc_returns = returns['BTC']
        
        # Run filter for each non-BTC asset
        for col in returns.columns:
            symbol = col.upper().split('-')[0]
            if symbol == 'BTC':
                continue
            
            kf = self._get_or_create_filter(symbol)
            kf.reset(initial_beta=self.static_betas.get(symbol, 1.0))
            kf.run_filter(returns[col], self._btc_returns)
        
        logger.info(f"Loaded data for {len(self._filters)} assets, {len(returns)} observations")
    
    def get_estimate(self, symbol: str) -> Optional[KalmanBetaEstimate]:
        """Get latest Kalman beta estimate for an asset."""
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._filters:
            logger.warning(f"No filter for {symbol}")
            return None
        
        history = self._filters[symbol].get_history()
        return history.latest
    
    def get_history(self, symbol: str) -> Optional[KalmanBetaHistory]:
        """Get full history for an asset."""
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._filters:
            return None
        
        return self._filters[symbol].get_history()
    
    def get_all_estimates(self) -> dict[str, KalmanBetaEstimate]:
        """Get latest estimates for all tracked assets."""
        results = {}
        for symbol, kf in self._filters.items():
            history = kf.get_history()
            if history.latest:
                results[symbol] = history.latest
        return results
    
    def compare_to_ensemble(
        self,
        ensemble_betas: dict[str, float],
    ) -> pd.DataFrame:
        """
        Compare Kalman betas to ensemble betas.
        
        Args:
            ensemble_betas: Dict of symbol -> ensemble beta from Priority 2
            
        Returns:
            DataFrame comparing Kalman, Ensemble, and Static betas
        """
        rows = []
        
        estimates = self.get_all_estimates()
        
        for symbol in set(estimates.keys()) | set(ensemble_betas.keys()):
            row = {"symbol": symbol}
            
            # Static
            row["static_beta"] = self.static_betas.get(symbol, np.nan)
            
            # Ensemble
            row["ensemble_beta"] = ensemble_betas.get(symbol, np.nan)
            
            # Kalman
            if symbol in estimates:
                est = estimates[symbol]
                row["kalman_beta"] = est.beta
                row["kalman_std"] = est.std
                row["kalman_ci_lower"] = est.confidence_lower
                row["kalman_ci_upper"] = est.confidence_upper
                row["is_uncertain"] = est.is_uncertain
                row["position_multiplier"] = est.position_size_multiplier
            else:
                row["kalman_beta"] = np.nan
                row["kalman_std"] = np.nan
                row["kalman_ci_lower"] = np.nan
                row["kalman_ci_upper"] = np.nan
                row["is_uncertain"] = True
                row["position_multiplier"] = 0.5
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("symbol").reset_index(drop=True)
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_kalman_beta(
    asset_prices: pd.Series,
    btc_prices: pd.Series,
    symbol: str,
    static_beta: Optional[float] = None,
) -> KalmanBetaEstimate:
    """
    Quick Kalman beta estimation for a single asset.
    
    Args:
        asset_prices: Asset price series
        btc_prices: BTC price series
        symbol: Asset symbol
        static_beta: Optional reference static beta
        
    Returns:
        Latest KalmanBetaEstimate
    """
    # Get static beta from config if not provided
    if static_beta is None:
        try:
            from crypto.config import CRYPTO_BETA_TO_BTC
            static_beta = CRYPTO_BETA_TO_BTC.get(symbol.upper(), 1.0)
        except ImportError:
            static_beta = 1.0
    
    # Calculate returns
    asset_returns = asset_prices.pct_change().dropna()
    btc_returns = btc_prices.pct_change().dropna()
    
    # Create and run filter
    kf = KalmanBetaFilter(
        symbol=symbol,
        initial_beta=static_beta,
        static_beta=static_beta,
    )
    history = kf.run_filter(asset_returns, btc_returns)
    
    return history.latest


def estimate_kalman_parameters(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    beta_guess: float = 1.0,
) -> dict:
    """
    Estimate reasonable Q and R parameters from data.
    
    Uses method of moments on residuals to estimate observation noise,
    and assumes process noise is a fraction of observation noise.
    
    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        beta_guess: Initial beta estimate
        
    Returns:
        Dict with 'process_variance' (Q) and 'observation_variance' (R)
    """
    # Align
    common_idx = asset_returns.index.intersection(btc_returns.index)
    asset_ret = asset_returns.loc[common_idx]
    btc_ret = btc_returns.loc[common_idx]
    
    # Estimate residuals using simple OLS beta
    x = btc_ret.values
    y = asset_ret.values
    
    # OLS beta
    if np.var(x) > 0:
        beta_ols = np.cov(x, y)[0, 1] / np.var(x)
    else:
        beta_ols = beta_guess
    
    # Residuals
    residuals = y - beta_ols * x
    
    # Observation variance from residual variance
    R = np.var(residuals)
    
    # Process variance: assume beta can drift by ~1% per period
    # Q = (0.01)^2 = 0.0001, but scale with R
    Q = 0.01 * R
    
    return {
        "process_variance": Q,
        "observation_variance": R,
        "ols_beta": beta_ols,
        "residual_std": np.std(residuals),
    }


def validate_kalman_vs_static(
    prices: pd.DataFrame,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Validate Kalman beta accuracy vs static beta on holdout data.
    
    Trains Kalman filter on data excluding last N days,
    then compares prediction errors on holdout.
    
    Args:
        prices: Price DataFrame with 'BTC' and other assets
        lookback_days: Number of days to holdout for validation
        
    Returns:
        DataFrame with prediction errors for Kalman vs Static
    """
    # Split data
    train = prices.iloc[:-lookback_days]
    test = prices.iloc[-lookback_days:]
    
    # Load static betas
    try:
        from crypto.config import CRYPTO_BETA_TO_BTC
        static_betas = CRYPTO_BETA_TO_BTC
    except ImportError:
        static_betas = {"ETH": 1.98, "SOL": 1.55, "XRP": 1.77, "ADA": 2.20}
    
    # Calculate returns
    test_returns = test.pct_change().dropna()
    btc_test_returns = test_returns['BTC']
    
    results = []
    
    for col in prices.columns:
        symbol = col.upper().split('-')[0]
        if symbol == 'BTC':
            continue
        
        static_beta = static_betas.get(symbol, 1.0)
        
        # Train Kalman filter
        tracker = KalmanBetaTracker(static_betas=static_betas)
        tracker.load_data(train)
        kalman_estimate = tracker.get_estimate(symbol)
        
        if kalman_estimate is None:
            continue
        
        kalman_beta = kalman_estimate.beta
        
        # Predict returns on test set
        asset_test_returns = test_returns[col]
        
        # Prediction errors
        static_predictions = static_beta * btc_test_returns
        kalman_predictions = kalman_beta * btc_test_returns
        
        static_errors = asset_test_returns - static_predictions
        kalman_errors = asset_test_returns - kalman_predictions
        
        results.append({
            "symbol": symbol,
            "static_beta": static_beta,
            "kalman_beta": kalman_beta,
            "static_mae": np.abs(static_errors).mean(),
            "kalman_mae": np.abs(kalman_errors).mean(),
            "static_rmse": np.sqrt((static_errors ** 2).mean()),
            "kalman_rmse": np.sqrt((kalman_errors ** 2).mean()),
            "improvement_mae": (np.abs(static_errors).mean() - np.abs(kalman_errors).mean()) / np.abs(static_errors).mean(),
            "improvement_rmse": (np.sqrt((static_errors ** 2).mean()) - np.sqrt((kalman_errors ** 2).mean())) / np.sqrt((static_errors ** 2).mean()),
        })
    
    return pd.DataFrame(results)
