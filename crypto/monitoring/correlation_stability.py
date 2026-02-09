"""
Correlation Stability Tracker for Crypto Pairs Trading.

Priority 1 component of the Beta Monitoring System.

Purpose:
- Track whether pair correlations are stable across time windows
- Detect relationship breakdown BEFORE P&L impact
- Provide early warning for position sizing adjustments

Philosophy:
- This is MEASUREMENT, not PREDICTION
- We're validating our static assumptions, not replacing economic rationale
- Stable correlation = confidence in hedge ratio, unstable = reduce exposure

Key Metrics:
- Short-term correlation (30d rolling)
- Long-term correlation (90d rolling) 
- Stability score = 1 - abs(short - long)
- Alert when stability drops below threshold

Session Origin: February 2, 2026
Catalyst: Feb 1-2 selloff showed ETH/XRP tracked betas well, SOL diverged
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

# Default window sizes for correlation calculation
SHORT_WINDOW: int = 30   # 30-day rolling correlation
LONG_WINDOW: int = 90    # 90-day rolling correlation

# Stability threshold - below this triggers alert
STABILITY_THRESHOLD: float = 0.70

# Alert levels
ALERT_CRITICAL: float = 0.50   # Major relationship breakdown
ALERT_WARNING: float = 0.70    # Elevated instability
ALERT_NORMAL: float = 0.85     # Minor deviation


@dataclass
class CorrelationStabilityResult:
    """Result from correlation stability analysis."""
    
    symbol1: str
    symbol2: str
    pair_name: str
    timestamp: datetime
    
    # Correlation values
    correlation_short: float   # Short window (30d)
    correlation_long: float    # Long window (90d)
    
    # Stability metrics
    stability_score: float     # 0-1, higher = more stable
    divergence: float          # Absolute difference between windows
    
    # Alert status
    alert_level: str           # 'NORMAL', 'WARNING', 'CRITICAL'
    is_stable: bool            # Convenience flag
    
    # Trend info
    correlation_trend: str     # 'STRENGTHENING', 'WEAKENING', 'STABLE'
    days_at_current_level: int # How long at current alert level
    
    @property
    def action_recommendation(self) -> str:
        """Position sizing recommendation based on stability."""
        if self.alert_level == 'CRITICAL':
            return "HALT - Reduce or close positions"
        elif self.alert_level == 'WARNING':
            return "REDUCE - Scale to 50% normal size"
        else:
            return "NORMAL - Full position sizing OK"


# =============================================================================
# CORE CALCULATION FUNCTIONS
# =============================================================================


def calculate_rolling_correlation(
    returns1: pd.Series,
    returns2: pd.Series,
    window: int = 30,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Calculate rolling correlation between two return series.
    
    Args:
        returns1: First return series
        returns2: Second return series
        window: Rolling window size
        min_periods: Minimum periods for valid calculation (default: window)
        
    Returns:
        Series of rolling correlation values
    """
    if min_periods is None:
        min_periods = window
    
    return returns1.rolling(window=window, min_periods=min_periods).corr(returns2)


def calculate_stability_score(
    correlation_short: float,
    correlation_long: float,
) -> float:
    """
    Calculate stability score from short and long-term correlations.
    
    Stability = 1 - abs(short - long)
    
    A score of 1.0 means perfect stability (short == long).
    A score near 0 means significant divergence between timeframes.
    
    Args:
        correlation_short: Short-window correlation
        correlation_long: Long-window correlation
        
    Returns:
        Stability score between 0 and 1
    """
    if np.isnan(correlation_short) or np.isnan(correlation_long):
        return np.nan
    
    divergence = abs(correlation_short - correlation_long)
    stability = 1.0 - divergence
    
    return max(0.0, min(1.0, stability))  # Clamp to [0, 1]


def determine_alert_level(stability_score: float) -> str:
    """
    Determine alert level from stability score.
    
    Args:
        stability_score: Stability metric (0-1)
        
    Returns:
        Alert level string: 'NORMAL', 'WARNING', or 'CRITICAL'
    """
    if np.isnan(stability_score):
        return 'UNKNOWN'
    
    if stability_score < ALERT_CRITICAL:
        return 'CRITICAL'
    elif stability_score < ALERT_WARNING:
        return 'WARNING'
    else:
        return 'NORMAL'


def detect_correlation_trend(
    correlation_series: pd.Series,
    lookback: int = 7,
    threshold: float = 0.05,
) -> str:
    """
    Detect if correlation is trending up, down, or stable.
    
    Args:
        correlation_series: Series of correlation values
        lookback: Number of periods to examine
        threshold: Minimum change to consider a trend
        
    Returns:
        Trend string: 'STRENGTHENING', 'WEAKENING', or 'STABLE'
    """
    if len(correlation_series) < lookback:
        return 'UNKNOWN'
    
    recent = correlation_series.iloc[-lookback:]
    
    if recent.isna().all():
        return 'UNKNOWN'
    
    # Linear regression slope
    valid = recent.dropna()
    if len(valid) < 3:
        return 'UNKNOWN'
    
    x = np.arange(len(valid))
    slope = np.polyfit(x, valid.values, 1)[0]
    
    if slope > threshold / lookback:
        return 'STRENGTHENING'
    elif slope < -threshold / lookback:
        return 'WEAKENING'
    else:
        return 'STABLE'


# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================


class CorrelationStabilityTracker:
    """
    Track correlation stability for crypto pairs.
    
    Monitors the relationship between asset pairs across multiple
    time windows to detect potential breakdown before P&L impact.
    
    Usage:
        tracker = CorrelationStabilityTracker()
        tracker.add_data(ada_prices, xrp_prices)
        result = tracker.get_current_stability('ADA', 'XRP')
        
        if not result.is_stable:
            print(f"Warning: {result.pair_name} unstable - {result.action_recommendation}")
    """
    
    def __init__(
        self,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
        stability_threshold: float = STABILITY_THRESHOLD,
    ):
        """
        Initialize the tracker.
        
        Args:
            short_window: Window for short-term correlation (default: 30)
            long_window: Window for long-term correlation (default: 90)
            stability_threshold: Alert threshold (default: 0.70)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.stability_threshold = stability_threshold
        
        # Internal storage
        self._prices: Dict[str, pd.Series] = {}
        self._returns: Dict[str, pd.Series] = {}
        self._alert_history: Dict[str, List[Tuple[datetime, str]]] = {}
    
    def add_prices(
        self,
        symbol: str,
        prices: pd.Series,
    ) -> None:
        """
        Add price series for a symbol.
        
        Args:
            symbol: Asset symbol (e.g., 'ADA', 'XRP')
            prices: Price series with datetime index
        """
        symbol = symbol.upper().split('-')[0]
        self._prices[symbol] = prices
        self._returns[symbol] = prices.pct_change()
        logger.info(f"Added price data for {symbol}: {len(prices)} observations")
    
    def add_prices_dict(
        self,
        prices_dict: Dict[str, pd.Series],
    ) -> None:
        """
        Add multiple price series at once.
        
        Args:
            prices_dict: Dictionary of symbol -> price series
        """
        for symbol, prices in prices_dict.items():
            self.add_prices(symbol, prices)
    
    def get_correlation_series(
        self,
        symbol1: str,
        symbol2: str,
        window: int,
    ) -> pd.Series:
        """
        Get rolling correlation series between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            window: Rolling window size
            
        Returns:
            Series of rolling correlations
        """
        symbol1 = symbol1.upper().split('-')[0]
        symbol2 = symbol2.upper().split('-')[0]
        
        if symbol1 not in self._returns:
            raise ValueError(f"No data for {symbol1}. Call add_prices() first.")
        if symbol2 not in self._returns:
            raise ValueError(f"No data for {symbol2}. Call add_prices() first.")
        
        ret1 = self._returns[symbol1]
        ret2 = self._returns[symbol2]
        
        # Align indices
        common_idx = ret1.index.intersection(ret2.index)
        ret1 = ret1.loc[common_idx]
        ret2 = ret2.loc[common_idx]
        
        return calculate_rolling_correlation(ret1, ret2, window)
    
    def get_stability_series(
        self,
        symbol1: str,
        symbol2: str,
    ) -> pd.DataFrame:
        """
        Get full stability analysis series.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            DataFrame with correlation and stability metrics over time
        """
        corr_short = self.get_correlation_series(symbol1, symbol2, self.short_window)
        corr_long = self.get_correlation_series(symbol1, symbol2, self.long_window)
        
        df = pd.DataFrame({
            'correlation_short': corr_short,
            'correlation_long': corr_long,
        })
        
        df['stability_score'] = df.apply(
            lambda row: calculate_stability_score(row['correlation_short'], row['correlation_long']),
            axis=1
        )
        
        df['divergence'] = (df['correlation_short'] - df['correlation_long']).abs()
        df['is_stable'] = df['stability_score'] >= self.stability_threshold
        df['alert_level'] = df['stability_score'].apply(determine_alert_level)
        
        return df
    
    def get_current_stability(
        self,
        symbol1: str,
        symbol2: str,
    ) -> CorrelationStabilityResult:
        """
        Get current stability assessment for a pair.
        
        This is the main interface for real-time monitoring.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            CorrelationStabilityResult with current metrics
        """
        symbol1 = symbol1.upper().split('-')[0]
        symbol2 = symbol2.upper().split('-')[0]
        pair_name = f"{symbol1}/{symbol2}"
        
        stability_df = self.get_stability_series(symbol1, symbol2)
        
        if stability_df.empty or stability_df['stability_score'].isna().all():
            logger.warning(f"Insufficient data for {pair_name} stability calculation")
            return CorrelationStabilityResult(
                symbol1=symbol1,
                symbol2=symbol2,
                pair_name=pair_name,
                timestamp=datetime.now(),
                correlation_short=np.nan,
                correlation_long=np.nan,
                stability_score=np.nan,
                divergence=np.nan,
                alert_level='UNKNOWN',
                is_stable=False,
                correlation_trend='UNKNOWN',
                days_at_current_level=0,
            )
        
        # Get current values (most recent non-NaN)
        current = stability_df.dropna().iloc[-1]
        corr_short = current['correlation_short']
        corr_long = current['correlation_long']
        stability = current['stability_score']
        divergence = current['divergence']
        alert_level = current['alert_level']
        
        # Calculate trend from short-term correlation
        corr_series = stability_df['correlation_short'].dropna()
        trend = detect_correlation_trend(corr_series)
        
        # Count days at current alert level
        alert_series = stability_df['alert_level'].dropna()
        days_at_level = 1
        for i in range(len(alert_series) - 2, -1, -1):
            if alert_series.iloc[i] == alert_level:
                days_at_level += 1
            else:
                break
        
        # Track alert history
        self._update_alert_history(pair_name, alert_level)
        
        return CorrelationStabilityResult(
            symbol1=symbol1,
            symbol2=symbol2,
            pair_name=pair_name,
            timestamp=datetime.now(),
            correlation_short=corr_short,
            correlation_long=corr_long,
            stability_score=stability,
            divergence=divergence,
            alert_level=alert_level,
            is_stable=stability >= self.stability_threshold,
            correlation_trend=trend,
            days_at_current_level=days_at_level,
        )
    
    def _update_alert_history(self, pair_name: str, alert_level: str) -> None:
        """Update internal alert history tracking."""
        if pair_name not in self._alert_history:
            self._alert_history[pair_name] = []
        
        history = self._alert_history[pair_name]
        
        # Only add if different from last entry
        if not history or history[-1][1] != alert_level:
            history.append((datetime.now(), alert_level))
        
        # Keep last 100 entries
        self._alert_history[pair_name] = history[-100:]
    
    def get_all_pairs_stability(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[CorrelationStabilityResult]:
        """
        Get stability for multiple pairs at once.
        
        Args:
            pairs: List of (symbol1, symbol2) tuples. If None, uses all available combinations.
            
        Returns:
            List of CorrelationStabilityResult objects
        """
        if pairs is None:
            # Generate all combinations from available symbols
            symbols = list(self._prices.keys())
            pairs = [(symbols[i], symbols[j]) 
                     for i in range(len(symbols)) 
                     for j in range(i + 1, len(symbols))]
        
        results = []
        for s1, s2 in pairs:
            try:
                result = self.get_current_stability(s1, s2)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to calculate stability for {s1}/{s2}: {e}")
        
        return results
    
    def format_stability_report(
        self,
        results: List[CorrelationStabilityResult],
    ) -> str:
        """
        Format stability results as readable report.
        
        Args:
            results: List of stability results
            
        Returns:
            Formatted string report
        """
        lines = [
            "=" * 60,
            "CORRELATION STABILITY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]
        
        # Sort by stability score (lowest first = most concerning)
        sorted_results = sorted(results, key=lambda r: r.stability_score if not np.isnan(r.stability_score) else 999)
        
        for r in sorted_results:
            status = "STABLE" if r.is_stable else "UNSTABLE"
            icon = "OK" if r.is_stable else "!!"
            
            lines.append(f"[{icon}] {r.pair_name}")
            lines.append(f"    Correlation: {r.correlation_short:.3f} (30d) vs {r.correlation_long:.3f} (90d)")
            lines.append(f"    Stability Score: {r.stability_score:.2f} ({status})")
            lines.append(f"    Alert Level: {r.alert_level}")
            lines.append(f"    Trend: {r.correlation_trend}")
            lines.append(f"    Recommendation: {r.action_recommendation}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_stability_check(
    prices1: pd.Series,
    prices2: pd.Series,
    symbol1: str = "ASSET1",
    symbol2: str = "ASSET2",
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
) -> CorrelationStabilityResult:
    """
    Quick one-shot stability check for a pair.
    
    Convenience function when you don't need persistent tracking.
    
    Args:
        prices1: First asset prices
        prices2: Second asset prices
        symbol1: Name for first asset
        symbol2: Name for second asset
        short_window: Short correlation window
        long_window: Long correlation window
        
    Returns:
        CorrelationStabilityResult
        
    Example:
        >>> result = quick_stability_check(ada_close, xrp_close, 'ADA', 'XRP')
        >>> print(f"{result.pair_name}: {result.stability_score:.2f} - {result.alert_level}")
    """
    tracker = CorrelationStabilityTracker(short_window, long_window)
    tracker.add_prices(symbol1, prices1)
    tracker.add_prices(symbol2, prices2)
    return tracker.get_current_stability(symbol1, symbol2)


def calculate_position_size_multiplier(stability_score: float) -> float:
    """
    Calculate position size multiplier based on stability.
    
    Maps stability score to position sizing:
    - 1.0 stability -> 1.0 multiplier (full size)
    - 0.7 stability -> 0.7 multiplier (reduced)
    - 0.5 stability -> 0.3 multiplier (heavily reduced)
    - <0.5 stability -> 0.0 multiplier (no position)
    
    Args:
        stability_score: Stability metric (0-1)
        
    Returns:
        Position size multiplier (0-1)
    """
    if np.isnan(stability_score) or stability_score < ALERT_CRITICAL:
        return 0.0
    elif stability_score < ALERT_WARNING:
        # Linear interpolation from 0.3 to 0.7
        return 0.3 + (stability_score - ALERT_CRITICAL) / (ALERT_WARNING - ALERT_CRITICAL) * 0.4
    else:
        # Linear interpolation from 0.7 to 1.0
        return 0.7 + (stability_score - ALERT_WARNING) / (1.0 - ALERT_WARNING) * 0.3
