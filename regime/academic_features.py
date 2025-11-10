"""
Academic Statistical Jump Model - Feature Calculation Module

This module implements feature calculations for the Academic Statistical Jump Model
as specified in "Downside Risk Reduction Using Regime-Switching Signals: A Statistical
Jump Model Approach" (Shu et al., Princeton, 2024).

Features:
    1. Downside Deviation (EWM, 10-day halflife)
    2. Sortino Ratio (EWM, 20-day halflife)
    3. Sortino Ratio (EWM, 60-day halflife)

Academic Foundation:
    - 33 years empirical validation (1990-2023)
    - Proven Sharpe improvements: +42% to +158%
    - MaxDD reduction: ~50% across S&P 500/DAX/Nikkei

Implementation Reference:
    Section 3.4.1 "Features" in academic paper
    Table 2: Feature specifications
"""

from typing import Optional
import pandas as pd
import numpy as np


def calculate_excess_returns(
    close: pd.Series,
    risk_free_rate: Optional[float] = None
) -> pd.Series:
    """
    Calculate excess returns (returns above risk-free rate).

    The academic paper uses excess returns for all calculations.
    If risk_free_rate is None, uses simple returns.

    Args:
        close: Close prices (pd.Series)
        risk_free_rate: Annual risk-free rate (e.g., 0.03 for 3%)
                       If None, uses 0.0 (simple returns)

    Returns:
        Excess returns (pd.Series)

    Example:
        >>> excess_ret = calculate_excess_returns(spy['Close'], risk_free_rate=0.03)
        >>> print(f"Mean excess return: {excess_ret.mean():.4f}")

    Note:
        Risk-free rate is converted from annual to daily: rf_daily = (1 + rf_annual)^(1/252) - 1
    """
    # Calculate simple returns (specify fill_method=None for Pandas 2.0+)
    returns = close.pct_change(fill_method=None)

    if risk_free_rate is None or risk_free_rate == 0.0:
        return returns

    # Convert annual risk-free rate to daily
    # Formula: (1 + r_annual)^(1/252) - 1
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1

    # Excess returns = returns - risk_free_rate
    excess_returns = returns - rf_daily

    return excess_returns


def calculate_downside_deviation(
    returns: pd.Series,
    halflife: int = 10,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate Exponentially Weighted Downside Deviation.

    Downside Deviation (DD) measures volatility of negative returns only,
    focusing on downside risk that investors care about most.

    Formula:
        DD_t = sqrt(EWM[R^2 * 1_{R<0}])

    Where:
        - R = excess returns
        - 1_{R<0} = indicator function (1 if R<0, else 0)
        - EWM = Exponentially Weighted Moving average with specified halflife

    Args:
        returns: Excess returns (pd.Series)
        halflife: Halflife in days for exponential weighting (default: 10)
                 Values lose half their weight after this many days
        min_periods: Minimum observations required (default: halflife)

    Returns:
        Downside deviation (pd.Series)

    Example:
        >>> dd = calculate_downside_deviation(excess_returns, halflife=10)
        >>> print(f"Current DD: {dd.iloc[-1]:.4f}")

    Academic Reference:
        Section 3.4.1, Table 2, Row 1
        Paper: "investors are more concerned with downside losses than uncertainty of upside gains"
    """
    if min_periods is None:
        min_periods = halflife

    # Isolate negative returns (downside)
    # Set positive returns to 0, keep negative returns
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0

    # Square the downside returns: R^2 * 1_{R<0}
    squared_downside = downside_returns ** 2

    # Calculate EWM of squared downside returns
    ewm_squared = squared_downside.ewm(
        halflife=halflife,
        min_periods=min_periods,
        adjust=False
    ).mean()

    # Take square root to get deviation
    downside_deviation = np.sqrt(ewm_squared)

    return downside_deviation


def calculate_sortino_ratio(
    returns: pd.Series,
    halflife: int = 20,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate Exponentially Weighted Sortino Ratio.

    Sortino Ratio measures risk-adjusted returns using downside deviation
    instead of total volatility, focusing on harmful volatility.

    Formula:
        Sortino_t = EWM_mean(R) / EWM_DD(R)

    Where:
        - EWM_mean(R) = Exponentially weighted average of returns
        - EWM_DD(R) = Exponentially weighted downside deviation
        - Both use the same halflife for consistency

    Args:
        returns: Excess returns (pd.Series)
        halflife: Halflife in days for exponential weighting (default: 20)
                 Common values: 20d (shorter-term), 60d (longer-term)
        min_periods: Minimum observations required (default: halflife)

    Returns:
        Sortino ratio (pd.Series)

    Example:
        >>> sortino_20 = calculate_sortino_ratio(excess_returns, halflife=20)
        >>> sortino_60 = calculate_sortino_ratio(excess_returns, halflife=60)
        >>> print(f"Sortino 20d: {sortino_20.iloc[-1]:.4f}")
        >>> print(f"Sortino 60d: {sortino_60.iloc[-1]:.4f}")

    Academic Reference:
        Section 3.4.1, Table 2, Rows 2-3
        Paper uses both 20d and 60d halflives to capture different time scales

    Note:
        - Higher Sortino = Better risk-adjusted returns
        - Negative Sortino = Negative average returns (bear market)
        - Very high Sortino = Strong bull market
    """
    if min_periods is None:
        min_periods = halflife

    # Calculate EWM mean of returns
    ewm_mean = returns.ewm(
        halflife=halflife,
        min_periods=min_periods,
        adjust=False
    ).mean()

    # Calculate EWM downside deviation (using same halflife)
    ewm_dd = calculate_downside_deviation(
        returns,
        halflife=halflife,
        min_periods=min_periods
    )

    # Sortino ratio = mean / downside_deviation
    # Handle division by zero: replace inf/nan with 0
    sortino_ratio = ewm_mean / ewm_dd
    sortino_ratio = sortino_ratio.replace([np.inf, -np.inf], np.nan)

    return sortino_ratio


def calculate_features(
    close: pd.Series,
    risk_free_rate: Optional[float] = None,
    halflife_dd: int = 10,
    halflife_sortino_1: int = 20,
    halflife_sortino_2: int = 60,
    standardize: bool = False
) -> pd.DataFrame:
    """
    Calculate all three features for Statistical Jump Model.

    This is the main feature calculation function that computes all features
    specified in the academic paper (Table 2).

    Features:
        1. Downside Deviation (10-day halflife)
        2. Sortino Ratio (20-day halflife)
        3. Sortino Ratio (60-day halflife)

    Args:
        close: Close prices (pd.Series)
        risk_free_rate: Annual risk-free rate (default: None = 0%)
        halflife_dd: Halflife for downside deviation (default: 10 days)
        halflife_sortino_1: Halflife for first Sortino ratio (default: 20 days)
        halflife_sortino_2: Halflife for second Sortino ratio (default: 60 days)
        standardize: Whether to z-score standardize features (default: False)
                    NOT used in reference implementation - features are used raw

    Returns:
        DataFrame with columns:
            - 'downside_dev': Downside deviation
            - 'sortino_20': Sortino ratio (20-day)
            - 'sortino_60': Sortino ratio (60-day)

        Returns raw features by default (matching reference implementation)

    Example:
        >>> features = calculate_features(spy['Close'], risk_free_rate=0.03)
        >>> print(features.tail())
        >>> print(f"Feature means: {features.mean()}")
        >>> print(f"Feature stds: {features.std()}")

    Academic Reference:
        Section 3.4.1, Table 2
        Paper: "We use an exponentially weighted moving (EWM) downside deviation
               with a halflife of 10 trading days, and EWM Sortino ratios with
               halflives of 20 and 60 days."

        Reference implementation (Yizhan-Oliver-Shu/jump-models) does NOT
        standardize features internally - they are used in raw form.

    Note:
        Features are already on comparable scales:
        - Downside Deviation: ~0.0001-0.05 (daily return scale)
        - Sortino Ratios: ~-2 to +3 (dimensionless)

        Optional standardization provided for experimentation but not used by default.
    """
    # Step 1: Calculate excess returns
    excess_returns = calculate_excess_returns(close, risk_free_rate)

    # Step 2: Calculate downside deviation (10-day halflife)
    dd = calculate_downside_deviation(excess_returns, halflife=halflife_dd)

    # Step 3: Calculate Sortino ratios (20-day and 60-day halflives)
    sortino_20 = calculate_sortino_ratio(excess_returns, halflife=halflife_sortino_1)
    sortino_60 = calculate_sortino_ratio(excess_returns, halflife=halflife_sortino_2)

    # Step 4: Combine into DataFrame
    features = pd.DataFrame({
        'downside_dev': dd,
        'sortino_20': sortino_20,
        'sortino_60': sortino_60
    }, index=close.index)

    # Step 5: Optional standardization (z-score normalization)
    if standardize:
        # Z-score: (x - mean) / std using GLOBAL statistics
        # Matches reference implementation (StandardScalerPD pattern)
        # Produces proper z-scores with mean=0, std=1
        features_standardized = features.copy()
        for col in features.columns:
            # Use global mean/std from full dataset (fit/transform pattern)
            mean = features[col].mean()
            std = features[col].std()

            # Handle edge case where std=0 (constant feature)
            if std > 0:
                features_standardized[col] = (features[col] - mean) / std
            else:
                features_standardized[col] = 0.0

        return features_standardized

    return features


def validate_features(features: pd.DataFrame) -> dict:
    """
    Validate calculated features for sanity checks.

    Checks:
        1. No NaN/Inf values in the last 252 days
        2. Downside deviation > 0 (strictly positive)
        3. Sortino ratios in reasonable range (-5 to 5 after standardization)
        4. Feature means close to 0 (if standardized)
        5. Feature stds close to 1 (if standardized)

    Args:
        features: DataFrame from calculate_features()

    Returns:
        Dictionary with validation results:
            - 'valid': bool (True if all checks pass)
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'stats': feature statistics

    Example:
        >>> features = calculate_features(spy['Close'])
        >>> validation = validate_features(features)
        >>> if not validation['valid']:
        >>>     print("Errors:", validation['errors'])
    """
    errors = []
    warnings = []

    # Get last 252 days for validation (1 trading year)
    recent = features.tail(252)

    # Check 1: No NaN/Inf in recent data
    if recent.isna().any().any():
        nan_cols = recent.columns[recent.isna().any()].tolist()
        errors.append(f"NaN values found in recent data: {nan_cols}")

    if np.isinf(recent.values).any():
        errors.append("Inf values found in recent data")

    # Check 2: Downside deviation should be positive
    if (recent['downside_dev'] <= 0).any():
        errors.append("Downside deviation has non-positive values")

    # Check 3: Sortino ratios in reasonable range
    for col in ['sortino_20', 'sortino_60']:
        if (recent[col].abs() > 10).any():
            warnings.append(f"{col} has extreme values (>10 std devs)")

    # Check 4: If standardized, means should be close to 0
    means = recent.mean()
    if (means.abs() > 0.5).any():
        warnings.append(f"Feature means not close to 0: {means.to_dict()}")

    # Check 5: If standardized, stds should be close to 1
    stds = recent.std()
    if ((stds < 0.5) | (stds > 2.0)).any():
        warnings.append(f"Feature stds not close to 1: {stds.to_dict()}")

    # Compile results
    validation = {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': {
            'mean': means.to_dict(),
            'std': stds.to_dict(),
            'min': recent.min().to_dict(),
            'max': recent.max().to_dict()
        }
    }

    return validation
