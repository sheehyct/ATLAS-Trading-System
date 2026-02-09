"""
Residual Anomaly Detection for Crypto Beta Monitoring.

Priority 3 component of the Beta Monitoring System.

Purpose:
- Detect moves that deviate significantly from beta-implied expectations
- Identify idiosyncratic risk events (e.g., SOL's -23% vs expected -20.2%)
- Flag assets that may be experiencing structural changes

Philosophy:
- If an asset consistently moves as beta predicts, relationship is healthy
- Large residuals indicate either:
  1. Idiosyncratic risk (news, liquidations, DeFi issues)
  2. Beta regime change (needs recalculation)
  3. Correlation breakdown (relationship failing)
- We MEASURE anomalies, we don't PREDICT them

Key Insight from Feb 1-2, 2026 Selloff:
- ETH residual: +0.7% (tracked well)
- XRP residual: +1.0% (tracked well)  
- SOL residual: -2.8% (ANOMALY - underperformed beta by 14% relative)

This would have triggered an alert for SOL, prompting investigation.

Session Origin: February 2, 2026
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Z-score thresholds for anomaly classification
ANOMALY_THRESHOLD_WARNING: float = 2.0    # 2 std deviations
ANOMALY_THRESHOLD_CRITICAL: float = 3.0   # 3 std deviations

# Window for historical residual statistics
RESIDUAL_LOOKBACK_DAYS: int = 60

# Minimum observations for reliable statistics
MIN_OBSERVATIONS: int = 20

# Rolling window for residual calculations
RESIDUAL_WINDOW: int = 30


@dataclass
class ResidualAnomaly:
    """Single anomaly detection result."""
    
    symbol: str
    timestamp: datetime
    
    # The move
    btc_return: float          # BTC's move
    asset_return: float        # Actual asset move
    expected_return: float     # Beta-implied expected move
    
    # Residual analysis
    residual: float            # actual - expected
    residual_pct: float        # residual as % of expected
    residual_zscore: float     # Standardized residual
    
    # Classification
    severity: str              # 'NORMAL', 'WARNING', 'CRITICAL'
    direction: str             # 'OUTPERFORMED', 'UNDERPERFORMED', 'INLINE'
    
    # Context
    beta_used: float           # Beta value used for prediction
    historical_std: float      # Historical residual std for context


@dataclass
class AnomalyReport:
    """Summary report for multiple assets."""
    
    timestamp: datetime
    btc_return: float
    
    # Per-asset results
    anomalies: Dict[str, ResidualAnomaly]
    
    # Summary stats
    total_assets: int
    normal_count: int
    warning_count: int
    critical_count: int
    
    @property
    def has_anomalies(self) -> bool:
        """True if any warnings or critical anomalies detected."""
        return self.warning_count > 0 or self.critical_count > 0
    
    @property
    def worst_anomaly(self) -> Optional[ResidualAnomaly]:
        """Return the most severe anomaly, if any."""
        criticals = [a for a in self.anomalies.values() if a.severity == 'CRITICAL']
        if criticals:
            return max(criticals, key=lambda x: abs(x.residual_zscore))
        
        warnings = [a for a in self.anomalies.values() if a.severity == 'WARNING']
        if warnings:
            return max(warnings, key=lambda x: abs(x.residual_zscore))
        
        return None


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================


def calculate_residual(
    actual_return: float,
    btc_return: float,
    beta: float,
) -> float:
    """
    Calculate residual from beta-implied prediction.
    
    Residual = actual - (beta * btc_return)
    
    Positive residual = asset outperformed beta expectation
    Negative residual = asset underperformed beta expectation
    
    Args:
        actual_return: Actual asset return
        btc_return: BTC return
        beta: Beta coefficient
        
    Returns:
        Residual value
    """
    expected = beta * btc_return
    return actual_return - expected


def calculate_residual_series(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    beta: float,
) -> pd.Series:
    """
    Calculate residual series over time.
    
    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        beta: Beta coefficient (can be static or rolling)
        
    Returns:
        Series of residuals
    """
    # Align indices
    common_idx = asset_returns.index.intersection(btc_returns.index)
    asset_returns = asset_returns.loc[common_idx]
    btc_returns = btc_returns.loc[common_idx]
    
    expected = beta * btc_returns
    residuals = asset_returns - expected
    residuals.name = 'residual'
    
    return residuals


def calculate_rolling_residual_stats(
    residuals: pd.Series,
    window: int = RESIDUAL_WINDOW,
) -> pd.DataFrame:
    """
    Calculate rolling statistics for residual series.
    
    Args:
        residuals: Residual series
        window: Rolling window size
        
    Returns:
        DataFrame with rolling mean, std, z-scores
    """
    df = pd.DataFrame({'residual': residuals})
    
    df['residual_mean'] = residuals.rolling(window).mean()
    df['residual_std'] = residuals.rolling(window).std()
    df['residual_zscore'] = (residuals - df['residual_mean']) / df['residual_std']
    
    # Replace inf values
    df['residual_zscore'] = df['residual_zscore'].replace([np.inf, -np.inf], np.nan)
    
    return df


def classify_anomaly(
    residual_zscore: float,
    warning_threshold: float = ANOMALY_THRESHOLD_WARNING,
    critical_threshold: float = ANOMALY_THRESHOLD_CRITICAL,
) -> Tuple[str, str]:
    """
    Classify anomaly severity and direction.
    
    Args:
        residual_zscore: Standardized residual
        warning_threshold: Z-score for WARNING level
        critical_threshold: Z-score for CRITICAL level
        
    Returns:
        Tuple of (severity, direction)
    """
    if np.isnan(residual_zscore):
        return ('UNKNOWN', 'UNKNOWN')
    
    abs_z = abs(residual_zscore)
    
    # Severity
    if abs_z >= critical_threshold:
        severity = 'CRITICAL'
    elif abs_z >= warning_threshold:
        severity = 'WARNING'
    else:
        severity = 'NORMAL'
    
    # Direction
    if abs_z < 0.5:
        direction = 'INLINE'
    elif residual_zscore > 0:
        direction = 'OUTPERFORMED'
    else:
        direction = 'UNDERPERFORMED'
    
    return severity, direction


def calculate_historical_residual_std(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    beta: float,
    lookback_days: int = RESIDUAL_LOOKBACK_DAYS,
) -> float:
    """
    Calculate historical standard deviation of residuals.
    
    Used for standardizing current residuals.
    
    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        beta: Beta coefficient
        lookback_days: Days of history to use
        
    Returns:
        Standard deviation of historical residuals
    """
    residuals = calculate_residual_series(asset_returns, btc_returns, beta)
    
    # Use recent history only
    if len(residuals) > lookback_days:
        residuals = residuals.iloc[-lookback_days:]
    
    return residuals.std()


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================


class ResidualAnomalyDetector:
    """
    Detect anomalous moves that deviate from beta expectations.
    
    Monitors residuals (actual - expected) and flags significant deviations
    that indicate idiosyncratic risk or relationship breakdown.
    
    Usage:
        detector = ResidualAnomalyDetector()
        detector.load_data(prices_df)
        
        # Check single asset
        anomaly = detector.check_latest('SOL')
        if anomaly.severity != 'NORMAL':
            print(f"Alert: {anomaly.symbol} {anomaly.direction} by {anomaly.residual:.2%}")
        
        # Check all assets
        report = detector.check_all()
        if report.has_anomalies:
            print(f"Warning: {report.warning_count} warnings, {report.critical_count} critical")
    """
    
    def __init__(
        self,
        betas: Optional[Dict[str, float]] = None,
        warning_threshold: float = ANOMALY_THRESHOLD_WARNING,
        critical_threshold: float = ANOMALY_THRESHOLD_CRITICAL,
        lookback_days: int = RESIDUAL_LOOKBACK_DAYS,
    ):
        """
        Initialize the detector.
        
        Args:
            betas: Dictionary of symbol -> beta. If None, loads from config.
            warning_threshold: Z-score for WARNING level
            critical_threshold: Z-score for CRITICAL level
            lookback_days: Days of history for statistics
        """
        # Load default betas if not provided
        if betas is None:
            try:
                from crypto.config import CRYPTO_BETA_TO_BTC
                betas = CRYPTO_BETA_TO_BTC
            except ImportError:
                betas = {"BTC": 1.0, "ETH": 2.0, "ADA": 2.2, "XRP": 1.8, "SOL": 1.5}
        
        self.betas = betas
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.lookback_days = lookback_days
        
        # Internal storage
        self._prices: Dict[str, pd.Series] = {}
        self._returns: Dict[str, pd.Series] = {}
        self._residual_history: Dict[str, pd.Series] = {}
        self._historical_stds: Dict[str, float] = {}
    
    def load_data(
        self,
        prices: pd.DataFrame,
    ) -> None:
        """
        Load price data for all assets.
        
        Args:
            prices: DataFrame with symbols as columns, datetime index
                   Must include 'BTC' column
        """
        if 'BTC' not in prices.columns:
            raise ValueError("Prices DataFrame must include 'BTC' column")
        
        for col in prices.columns:
            symbol = col.upper().split('-')[0]
            self._prices[symbol] = prices[col]
            self._returns[symbol] = prices[col].pct_change()
        
        # Pre-calculate residual history and stats
        self._calculate_all_residual_history()
        
        logger.info(f"Loaded data for {len(prices.columns)} assets, "
                   f"{len(prices)} observations each")
    
    def _calculate_all_residual_history(self) -> None:
        """Calculate and store residual history for all assets."""
        btc_returns = self._returns.get('BTC')
        if btc_returns is None:
            return
        
        for symbol in self._returns:
            if symbol == 'BTC':
                continue
            
            beta = self.betas.get(symbol, 1.0)
            asset_returns = self._returns[symbol]
            
            residuals = calculate_residual_series(asset_returns, btc_returns, beta)
            self._residual_history[symbol] = residuals
            
            # Calculate historical std
            if len(residuals.dropna()) >= MIN_OBSERVATIONS:
                self._historical_stds[symbol] = residuals.iloc[-self.lookback_days:].std()
            else:
                self._historical_stds[symbol] = np.nan
    
    def check_latest(
        self,
        symbol: str,
        beta_override: Optional[float] = None,
    ) -> ResidualAnomaly:
        """
        Check the latest observation for anomaly.
        
        Args:
            symbol: Asset symbol
            beta_override: Optional beta to use instead of stored value
            
        Returns:
            ResidualAnomaly result
        """
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._returns:
            raise ValueError(f"No data for {symbol}. Call load_data() first.")
        
        beta = beta_override if beta_override is not None else self.betas.get(symbol, 1.0)
        
        asset_returns = self._returns[symbol]
        btc_returns = self._returns['BTC']
        
        # Get latest values
        latest_asset = asset_returns.iloc[-1]
        latest_btc = btc_returns.iloc[-1]
        
        # Calculate residual
        expected = beta * latest_btc
        residual = latest_asset - expected
        
        # Calculate residual percentage
        if expected != 0:
            residual_pct = residual / abs(expected)
        else:
            residual_pct = 0.0
        
        # Get historical std for z-score
        hist_std = self._historical_stds.get(symbol, np.nan)
        
        if hist_std > 0 and not np.isnan(hist_std):
            residual_zscore = residual / hist_std
        else:
            residual_zscore = np.nan
        
        # Classify
        severity, direction = classify_anomaly(
            residual_zscore, 
            self.warning_threshold, 
            self.critical_threshold
        )
        
        return ResidualAnomaly(
            symbol=symbol,
            timestamp=asset_returns.index[-1] if hasattr(asset_returns.index, '__getitem__') else datetime.now(),
            btc_return=latest_btc,
            asset_return=latest_asset,
            expected_return=expected,
            residual=residual,
            residual_pct=residual_pct,
            residual_zscore=residual_zscore,
            severity=severity,
            direction=direction,
            beta_used=beta,
            historical_std=hist_std,
        )
    
    def check_all(self) -> AnomalyReport:
        """
        Check all assets for anomalies.
        
        Returns:
            AnomalyReport with results for all assets
        """
        anomalies = {}
        
        for symbol in self._returns:
            if symbol == 'BTC':
                continue
            
            try:
                anomaly = self.check_latest(symbol)
                anomalies[symbol] = anomaly
            except Exception as e:
                logger.warning(f"Failed to check {symbol}: {e}")
        
        # Count severities
        normal = sum(1 for a in anomalies.values() if a.severity == 'NORMAL')
        warning = sum(1 for a in anomalies.values() if a.severity == 'WARNING')
        critical = sum(1 for a in anomalies.values() if a.severity == 'CRITICAL')
        
        # Get BTC return
        btc_return = self._returns['BTC'].iloc[-1] if 'BTC' in self._returns else 0.0
        
        return AnomalyReport(
            timestamp=datetime.now(),
            btc_return=btc_return,
            anomalies=anomalies,
            total_assets=len(anomalies),
            normal_count=normal,
            warning_count=warning,
            critical_count=critical,
        )
    
    def get_residual_history(
        self,
        symbol: str,
        include_stats: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical residuals with optional rolling statistics.
        
        Args:
            symbol: Asset symbol
            include_stats: Include rolling mean, std, z-score
            
        Returns:
            DataFrame with residual history
        """
        symbol = symbol.upper().split('-')[0]
        
        if symbol not in self._residual_history:
            raise ValueError(f"No residual history for {symbol}")
        
        residuals = self._residual_history[symbol]
        
        if include_stats:
            return calculate_rolling_residual_stats(residuals, RESIDUAL_WINDOW)
        else:
            return pd.DataFrame({'residual': residuals})
    
    def format_anomaly_report(self, report: AnomalyReport) -> str:
        """
        Format anomaly report as readable string.
        
        Args:
            report: AnomalyReport to format
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 70,
            "RESIDUAL ANOMALY REPORT",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"BTC Return: {report.btc_return:+.2%}",
            "=" * 70,
            "",
        ]
        
        # Summary
        lines.append(f"Assets Checked: {report.total_assets}")
        lines.append(f"  Normal: {report.normal_count}")
        lines.append(f"  Warning: {report.warning_count}")
        lines.append(f"  Critical: {report.critical_count}")
        lines.append("")
        
        # Header
        lines.append(
            f"{'Symbol':<8} {'Expected':>10} {'Actual':>10} {'Residual':>10} "
            f"{'Z-Score':>8} {'Status':>10}"
        )
        lines.append("-" * 70)
        
        # Sort by severity then z-score
        sorted_anomalies = sorted(
            report.anomalies.values(),
            key=lambda x: (
                0 if x.severity == 'CRITICAL' else 1 if x.severity == 'WARNING' else 2,
                -abs(x.residual_zscore) if not np.isnan(x.residual_zscore) else 999
            )
        )
        
        for a in sorted_anomalies:
            status_icon = (
                "!!!" if a.severity == 'CRITICAL' 
                else "!!" if a.severity == 'WARNING' 
                else "OK"
            )
            
            z_str = f"{a.residual_zscore:+.2f}" if not np.isnan(a.residual_zscore) else "N/A"
            
            lines.append(
                f"{a.symbol:<8} {a.expected_return:>+10.2%} {a.asset_return:>+10.2%} "
                f"{a.residual:>+10.2%} {z_str:>8} {status_icon:>10}"
            )
        
        # Recommendations
        if report.has_anomalies:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            
            for a in sorted_anomalies:
                if a.severity == 'CRITICAL':
                    lines.append(f"  - {a.symbol}: INVESTIGATE - "
                               f"{a.direction.lower()} beta expectation by {abs(a.residual_pct):.1%}")
                elif a.severity == 'WARNING':
                    lines.append(f"  - {a.symbol}: MONITOR - "
                               f"{a.direction.lower()}, consider reducing exposure")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_anomaly_check(
    asset_prices: pd.Series,
    btc_prices: pd.Series,
    beta: float,
    symbol: str = "ASSET",
) -> ResidualAnomaly:
    """
    Quick one-shot anomaly check.
    
    Args:
        asset_prices: Asset price series
        btc_prices: BTC price series
        beta: Beta coefficient
        symbol: Asset symbol
        
    Returns:
        ResidualAnomaly result
    """
    prices_df = pd.DataFrame({
        'BTC': btc_prices,
        symbol: asset_prices,
    })
    
    detector = ResidualAnomalyDetector(betas={symbol: beta, 'BTC': 1.0})
    detector.load_data(prices_df)
    
    return detector.check_latest(symbol)


def validate_february_selloff(
    btc_move: float = -0.13,
    actual_moves: Optional[Dict[str, float]] = None,
    betas: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict]:
    """
    Validate residual detection against Feb 1-2, 2026 selloff.
    
    This is the catalyst event that motivated this monitoring system.
    
    Args:
        btc_move: BTC's move (default: -13%)
        actual_moves: Dict of symbol -> actual move
        betas: Dict of symbol -> beta
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> results = validate_february_selloff()
        >>> for symbol, data in results.items():
        ...     print(f"{symbol}: error = {data['prediction_error']:.1%}")
    """
    if actual_moves is None:
        actual_moves = {
            'ETH': -0.25,   # Dropped 25%
            'XRP': -0.22,   # Dropped 22%
            'SOL': -0.23,   # Dropped 23%
        }
    
    if betas is None:
        betas = {
            'ETH': 1.98,
            'XRP': 1.77,
            'SOL': 1.55,
            'ADA': 2.20,
        }
    
    results = {}
    
    for symbol in actual_moves:
        if symbol not in betas:
            continue
        
        beta = betas[symbol]
        actual = actual_moves[symbol]
        expected = btc_move * beta
        residual = actual - expected
        error_pct = residual / abs(expected) if expected != 0 else 0
        
        results[symbol] = {
            'beta': beta,
            'btc_move': btc_move,
            'expected_move': expected,
            'actual_move': actual,
            'residual': residual,
            'prediction_error': error_pct,
            'tracked_well': abs(error_pct) < 0.10,  # Within 10%
        }
    
    return results
