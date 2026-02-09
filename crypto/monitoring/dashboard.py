"""
Beta Monitoring Dashboard - Unified Interface.

Combines Priority 1-3 components into a single monitoring interface:
- Correlation Stability (Priority 1)
- Ensemble Beta (Priority 2)
- Residual Anomaly Detection (Priority 3)

Provides consolidated dashboard output and actionable recommendations.

Session Origin: February 2, 2026
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .correlation_stability import (
    CorrelationStabilityTracker,
    CorrelationStabilityResult,
    calculate_position_size_multiplier as stability_multiplier,
)
from .ensemble_beta import (
    EnsembleBetaTracker,
    EnsembleBetaResult,
)
from .residual_anomaly import (
    ResidualAnomalyDetector,
    ResidualAnomaly,
    AnomalyReport,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMBINED RESULTS
# =============================================================================


@dataclass
class AssetMonitoringResult:
    """Combined monitoring result for a single asset."""
    
    symbol: str
    timestamp: datetime
    
    # From Ensemble Beta (Priority 2)
    static_beta: float
    ensemble_beta: float
    beta_confidence_interval: Tuple[float, float]
    beta_deviation: float  # From static
    
    # From Residual Anomaly (Priority 3)
    latest_residual: float
    residual_zscore: float
    anomaly_severity: str
    
    # Consolidated recommendations
    overall_health: str              # 'HEALTHY', 'CAUTION', 'WARNING', 'CRITICAL'
    position_size_multiplier: float  # 0-1
    recommended_beta: float          # Which beta to use for sizing
    
    @property
    def needs_attention(self) -> bool:
        """True if any component signals concern."""
        return self.overall_health in ('WARNING', 'CRITICAL')


@dataclass
class PairMonitoringResult:
    """Combined monitoring result for a trading pair."""
    
    pair_name: str
    symbol1: str
    symbol2: str
    timestamp: datetime
    
    # From Correlation Stability (Priority 1)
    correlation_short: float
    correlation_long: float
    stability_score: float
    is_stable: bool
    
    # Per-asset results
    asset1_result: AssetMonitoringResult
    asset2_result: AssetMonitoringResult
    
    # Consolidated
    pair_health: str
    position_size_multiplier: float
    
    @property
    def needs_attention(self) -> bool:
        """True if pair or either asset needs attention."""
        return (not self.is_stable or 
                self.asset1_result.needs_attention or 
                self.asset2_result.needs_attention)


@dataclass
class DashboardSnapshot:
    """Full dashboard snapshot at a point in time."""
    
    timestamp: datetime
    btc_price: Optional[float]
    btc_return_24h: Optional[float]
    
    # Per-asset results
    asset_results: Dict[str, AssetMonitoringResult]
    
    # Per-pair results (for stat arb)
    pair_results: Dict[str, PairMonitoringResult]
    
    # Summary counts
    total_assets: int
    healthy_assets: int
    warning_assets: int
    critical_assets: int
    
    # Recommendations
    top_recommendation: str


# =============================================================================
# MAIN DASHBOARD CLASS
# =============================================================================


class BetaMonitoringDashboard:
    """
    Unified Beta Monitoring Dashboard.
    
    Integrates correlation stability, ensemble beta, and anomaly detection
    into a single interface for crypto trading decisions.
    
    Usage:
        dashboard = BetaMonitoringDashboard()
        dashboard.load_prices(prices_df)
        
        snapshot = dashboard.get_snapshot()
        print(snapshot.top_recommendation)
        
        # Check specific pair for stat arb
        pair_result = dashboard.check_pair('ADA', 'XRP')
        if pair_result.needs_attention:
            print(f"Warning: {pair_result.pair_name} requires attention")
    """
    
    def __init__(
        self,
        static_betas: Optional[Dict[str, float]] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Initialize the dashboard.
        
        Args:
            static_betas: Dictionary of symbol -> static beta
            pairs: List of (symbol1, symbol2) pairs to monitor for stat arb
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
        self.pairs = pairs or [('ADA', 'XRP')]  # Default stat arb pair
        
        # Initialize component trackers
        self._correlation_tracker = CorrelationStabilityTracker()
        self._beta_tracker = EnsembleBetaTracker(static_betas=static_betas)
        self._anomaly_detector = ResidualAnomalyDetector(betas=static_betas)
        
        # Data storage
        self._prices: Optional[pd.DataFrame] = None
        self._btc_price: Optional[float] = None
        self._btc_return_24h: Optional[float] = None
    
    def load_prices(
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
        
        self._prices = prices
        
        # Store BTC info
        self._btc_price = prices['BTC'].iloc[-1]
        btc_returns = prices['BTC'].pct_change()
        self._btc_return_24h = btc_returns.iloc[-1]
        
        # Load into component trackers
        for col in prices.columns:
            symbol = col.upper().split('-')[0]
            
            # Correlation tracker
            self._correlation_tracker.add_prices(symbol, prices[col])
            
            # Beta tracker (needs BTC as reference)
            if symbol != 'BTC':
                self._beta_tracker.load_data(prices[col], prices['BTC'], symbol)
        
        # Anomaly detector
        self._anomaly_detector.load_data(prices)
        
        logger.info(f"Loaded {len(prices.columns)} assets, {len(prices)} observations")
    
    def _get_asset_result(self, symbol: str) -> AssetMonitoringResult:
        """Get combined monitoring result for a single asset."""
        symbol = symbol.upper().split('-')[0]
        
        # Get ensemble beta
        try:
            beta_result = self._beta_tracker.get_ensemble_beta(symbol)
        except Exception as e:
            logger.warning(f"Failed to get ensemble beta for {symbol}: {e}")
            beta_result = None
        
        # Get anomaly check
        try:
            anomaly_result = self._anomaly_detector.check_latest(symbol)
        except Exception as e:
            logger.warning(f"Failed to check anomaly for {symbol}: {e}")
            anomaly_result = None
        
        # Extract values
        static_beta = self.static_betas.get(symbol, 1.0)
        
        if beta_result:
            ensemble_beta = beta_result.beta_ensemble
            beta_ci = (beta_result.beta_lower, beta_result.beta_upper)
            beta_deviation = beta_result.deviation_from_static
        else:
            ensemble_beta = static_beta
            beta_ci = (static_beta * 0.8, static_beta * 1.2)
            beta_deviation = 0.0
        
        if anomaly_result:
            residual = anomaly_result.residual
            residual_z = anomaly_result.residual_zscore
            severity = anomaly_result.severity
        else:
            residual = 0.0
            residual_z = 0.0
            severity = 'UNKNOWN'
        
        # Determine overall health
        health = self._assess_asset_health(beta_deviation, severity, beta_ci)
        
        # Calculate position multiplier
        multiplier = self._calculate_asset_multiplier(beta_deviation, severity, beta_ci)
        
        # Determine recommended beta
        recommended_beta = ensemble_beta if abs(beta_deviation) < 0.20 else static_beta
        
        return AssetMonitoringResult(
            symbol=symbol,
            timestamp=datetime.now(),
            static_beta=static_beta,
            ensemble_beta=ensemble_beta,
            beta_confidence_interval=beta_ci,
            beta_deviation=beta_deviation,
            latest_residual=residual,
            residual_zscore=residual_z if not np.isnan(residual_z) else 0.0,
            anomaly_severity=severity,
            overall_health=health,
            position_size_multiplier=multiplier,
            recommended_beta=recommended_beta,
        )
    
    def _assess_asset_health(
        self,
        beta_deviation: float,
        anomaly_severity: str,
        beta_ci: Tuple[float, float],
    ) -> str:
        """Assess overall health based on multiple factors."""
        # Critical conditions
        if anomaly_severity == 'CRITICAL':
            return 'CRITICAL'
        if abs(beta_deviation) > 0.30:  # >30% beta deviation
            return 'CRITICAL'
        
        # Warning conditions
        if anomaly_severity == 'WARNING':
            return 'WARNING'
        if abs(beta_deviation) > 0.15:  # >15% beta deviation
            return 'WARNING'
        
        ci_width = beta_ci[1] - beta_ci[0]
        mean_beta = (beta_ci[0] + beta_ci[1]) / 2
        if ci_width / mean_beta > 0.30:  # >30% relative CI width
            return 'CAUTION'
        
        return 'HEALTHY'
    
    def _calculate_asset_multiplier(
        self,
        beta_deviation: float,
        anomaly_severity: str,
        beta_ci: Tuple[float, float],
    ) -> float:
        """Calculate position size multiplier for an asset."""
        multiplier = 1.0
        
        # Reduce for beta deviation
        if abs(beta_deviation) > 0.20:
            multiplier *= 0.5
        elif abs(beta_deviation) > 0.10:
            multiplier *= 0.75
        
        # Reduce for anomalies
        if anomaly_severity == 'CRITICAL':
            multiplier *= 0.0  # No position
        elif anomaly_severity == 'WARNING':
            multiplier *= 0.5
        
        # Reduce for wide confidence interval
        ci_width = beta_ci[1] - beta_ci[0]
        mean_beta = (beta_ci[0] + beta_ci[1]) / 2
        relative_width = ci_width / mean_beta if mean_beta > 0 else 1.0
        
        if relative_width > 0.30:
            multiplier *= 0.5
        elif relative_width > 0.20:
            multiplier *= 0.75
        
        return max(0.0, min(1.0, multiplier))
    
    def check_pair(
        self,
        symbol1: str,
        symbol2: str,
    ) -> PairMonitoringResult:
        """
        Get combined monitoring result for a trading pair.
        
        Args:
            symbol1: First symbol (long side of spread)
            symbol2: Second symbol (short side of spread)
            
        Returns:
            PairMonitoringResult
        """
        symbol1 = symbol1.upper().split('-')[0]
        symbol2 = symbol2.upper().split('-')[0]
        pair_name = f"{symbol1}/{symbol2}"
        
        # Get correlation stability
        try:
            stability = self._correlation_tracker.get_current_stability(symbol1, symbol2)
        except Exception as e:
            logger.warning(f"Failed to get stability for {pair_name}: {e}")
            stability = None
        
        # Get asset results
        asset1 = self._get_asset_result(symbol1)
        asset2 = self._get_asset_result(symbol2)
        
        # Extract stability values
        if stability:
            corr_short = stability.correlation_short
            corr_long = stability.correlation_long
            stab_score = stability.stability_score
            is_stable = stability.is_stable
        else:
            corr_short = np.nan
            corr_long = np.nan
            stab_score = np.nan
            is_stable = False
        
        # Determine pair health
        pair_health = self._assess_pair_health(is_stable, stab_score, asset1, asset2)
        
        # Calculate pair multiplier (product of components)
        stability_mult = stability_multiplier(stab_score) if not np.isnan(stab_score) else 0.5
        pair_multiplier = (
            stability_mult * 
            asset1.position_size_multiplier * 
            asset2.position_size_multiplier
        )
        
        return PairMonitoringResult(
            pair_name=pair_name,
            symbol1=symbol1,
            symbol2=symbol2,
            timestamp=datetime.now(),
            correlation_short=corr_short,
            correlation_long=corr_long,
            stability_score=stab_score,
            is_stable=is_stable,
            asset1_result=asset1,
            asset2_result=asset2,
            pair_health=pair_health,
            position_size_multiplier=pair_multiplier,
        )
    
    def _assess_pair_health(
        self,
        is_stable: bool,
        stability_score: float,
        asset1: AssetMonitoringResult,
        asset2: AssetMonitoringResult,
    ) -> str:
        """Assess overall health of a trading pair."""
        # Critical if either asset is critical or stability is very low
        if asset1.overall_health == 'CRITICAL' or asset2.overall_health == 'CRITICAL':
            return 'CRITICAL'
        if not np.isnan(stability_score) and stability_score < 0.50:
            return 'CRITICAL'
        
        # Warning if either asset has warning or stability is low
        if asset1.overall_health == 'WARNING' or asset2.overall_health == 'WARNING':
            return 'WARNING'
        if not is_stable:
            return 'WARNING'
        
        # Caution if either has caution
        if asset1.overall_health == 'CAUTION' or asset2.overall_health == 'CAUTION':
            return 'CAUTION'
        
        return 'HEALTHY'
    
    def get_snapshot(self) -> DashboardSnapshot:
        """
        Get full dashboard snapshot.
        
        Returns:
            DashboardSnapshot with all monitoring data
        """
        # Get all asset results
        asset_results = {}
        for symbol in self.static_betas:
            if symbol == 'BTC':
                continue
            try:
                asset_results[symbol] = self._get_asset_result(symbol)
            except Exception as e:
                logger.warning(f"Failed to get result for {symbol}: {e}")
        
        # Get pair results
        pair_results = {}
        for s1, s2 in self.pairs:
            try:
                pair_name = f"{s1}/{s2}"
                pair_results[pair_name] = self.check_pair(s1, s2)
            except Exception as e:
                logger.warning(f"Failed to get result for {s1}/{s2}: {e}")
        
        # Count health statuses
        healths = [r.overall_health for r in asset_results.values()]
        healthy = healths.count('HEALTHY')
        warning = healths.count('WARNING') + healths.count('CAUTION')
        critical = healths.count('CRITICAL')
        
        # Generate top recommendation
        recommendation = self._generate_top_recommendation(asset_results, pair_results)
        
        return DashboardSnapshot(
            timestamp=datetime.now(),
            btc_price=self._btc_price,
            btc_return_24h=self._btc_return_24h,
            asset_results=asset_results,
            pair_results=pair_results,
            total_assets=len(asset_results),
            healthy_assets=healthy,
            warning_assets=warning,
            critical_assets=critical,
            top_recommendation=recommendation,
        )
    
    def _generate_top_recommendation(
        self,
        asset_results: Dict[str, AssetMonitoringResult],
        pair_results: Dict[str, PairMonitoringResult],
    ) -> str:
        """Generate the most important recommendation."""
        # Check for critical issues first
        for symbol, result in asset_results.items():
            if result.overall_health == 'CRITICAL':
                return f"CRITICAL: {symbol} - Halt trading, investigate immediately"
        
        for pair_name, result in pair_results.items():
            if result.pair_health == 'CRITICAL':
                return f"CRITICAL: {pair_name} pair - Reduce or close positions"
        
        # Check for warnings
        warnings = [r for r in asset_results.values() if r.overall_health == 'WARNING']
        if warnings:
            symbols = [r.symbol for r in warnings]
            return f"WARNING: {', '.join(symbols)} - Consider reducing exposure"
        
        for pair_name, result in pair_results.items():
            if result.pair_health == 'WARNING':
                return f"WARNING: {pair_name} - Scale to {result.position_size_multiplier:.0%} size"
        
        return "All systems nominal - Normal position sizing OK"
    
    def format_dashboard(self) -> str:
        """
        Format dashboard as ASCII text.
        
        Returns:
            Formatted dashboard string
        """
        snapshot = self.get_snapshot()
        
        lines = []
        
        # Header
        lines.append("+" + "=" * 68 + "+")
        lines.append("|" + "CRYPTO BETA MONITORING DASHBOARD".center(68) + "|")
        lines.append("+" + "=" * 68 + "+")
        lines.append(f"| Timestamp: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<55}|")
        
        if snapshot.btc_price:
            lines.append(f"| BTC Price: ${snapshot.btc_price:,.0f}  "
                        f"24h Return: {(snapshot.btc_return_24h or 0) * 100:+.1f}%".ljust(67) + "|")
        
        lines.append("+" + "-" * 68 + "+")
        
        # Asset table
        lines.append("| ASSET MONITORING".ljust(68) + "|")
        lines.append("+" + "-" * 68 + "+")
        lines.append(f"| {'Symbol':<6} {'Static':>7} {'Ensemble':>9} {'Dev':>7} "
                    f"{'Residual':>10} {'Health':>10} {'Size':>7} |")
        lines.append("+" + "-" * 68 + "+")
        
        for symbol, result in sorted(snapshot.asset_results.items()):
            dev_str = f"{result.beta_deviation:+.1%}"
            res_str = f"{result.residual_zscore:+.1f}s" if result.residual_zscore != 0 else "0.0s"
            
            lines.append(
                f"| {symbol:<6} {result.static_beta:>7.2f} {result.ensemble_beta:>9.2f} "
                f"{dev_str:>7} {res_str:>10} {result.overall_health:>10} "
                f"{result.position_size_multiplier:>6.0%} |"
            )
        
        lines.append("+" + "-" * 68 + "+")
        
        # Pair table
        if snapshot.pair_results:
            lines.append("| PAIR MONITORING (Stat Arb)".ljust(68) + "|")
            lines.append("+" + "-" * 68 + "+")
            lines.append(f"| {'Pair':<10} {'Corr 30d':>9} {'Corr 90d':>9} "
                        f"{'Stability':>10} {'Health':>10} {'Size':>8} |")
            lines.append("+" + "-" * 68 + "+")
            
            for pair_name, result in snapshot.pair_results.items():
                corr_s = f"{result.correlation_short:.2f}" if not np.isnan(result.correlation_short) else "N/A"
                corr_l = f"{result.correlation_long:.2f}" if not np.isnan(result.correlation_long) else "N/A"
                stab = f"{result.stability_score:.2f}" if not np.isnan(result.stability_score) else "N/A"
                
                lines.append(
                    f"| {pair_name:<10} {corr_s:>9} {corr_l:>9} "
                    f"{stab:>10} {result.pair_health:>10} "
                    f"{result.position_size_multiplier:>7.0%} |"
                )
            
            lines.append("+" + "-" * 68 + "+")
        
        # Summary
        lines.append("| SUMMARY".ljust(68) + "|")
        lines.append(f"| Healthy: {snapshot.healthy_assets}  "
                    f"Warning: {snapshot.warning_assets}  "
                    f"Critical: {snapshot.critical_assets}".ljust(67) + "|")
        lines.append("+" + "-" * 68 + "+")
        
        # Recommendation
        lines.append("| RECOMMENDATION".ljust(68) + "|")
        
        # Word wrap recommendation
        rec = snapshot.top_recommendation
        while rec:
            chunk = rec[:66]
            rec = rec[66:]
            lines.append(f"| {chunk:<66} |")
        
        lines.append("+" + "=" * 68 + "+")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_dashboard(
    prices: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> BetaMonitoringDashboard:
    """
    Create and initialize a dashboard.
    
    Args:
        prices: Price DataFrame with BTC and other assets
        pairs: Optional list of pairs to monitor
        
    Returns:
        Initialized BetaMonitoringDashboard
    """
    dashboard = BetaMonitoringDashboard(pairs=pairs)
    dashboard.load_prices(prices)
    return dashboard


def quick_check(prices: pd.DataFrame) -> str:
    """
    Quick health check - returns formatted dashboard.
    
    Args:
        prices: Price DataFrame
        
    Returns:
        Formatted dashboard string
    """
    dashboard = create_dashboard(prices)
    return dashboard.format_dashboard()
