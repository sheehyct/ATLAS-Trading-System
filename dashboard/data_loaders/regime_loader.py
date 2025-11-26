"""
Regime Data Loader

Loads regime detection data from ATLAS Layer 1 (academic_jump_model.py).
Provides daily-cached regime data for dashboard visualization.

IMPORTANT: ATLAS regime detection is inherently DAILY:
- Input: Daily OHLCV data
- Output: Regime classification for the current day
- Tradeable: Next trading day (regime changes can't be acted on intraday)
- Updates: Once per day after market close (4 PM ET)

The loader caches results daily to avoid expensive re-computation on every page load.

Note: This module uses yfinance directly for data fetching to avoid VectorBT Pro
dependency on Railway. Backtesting code continues to use VBT Pro for full features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def _fetch_yf_data(symbol: str, start: str, end: str, tz: str = 'America/New_York') -> pd.DataFrame:
    """
    Fetch OHLCV data using yfinance directly.

    This is a lightweight alternative to vbt.YFData.pull() for dashboard use.
    Returns DataFrame with same structure as VBT for compatibility.

    Args:
        symbol: Ticker symbol (e.g., 'SPY', '^VIX')
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        tz: Timezone to localize to (default: America/New_York)

    Returns:
        DataFrame with OHLCV columns, timezone-aware index
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        logger.warning(f"No data returned for {symbol} from {start} to {end}")
        return pd.DataFrame()

    # Ensure timezone-aware index (yfinance returns UTC, convert to requested tz)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(tz)

    # Keep only OHLCV columns (matching VBT structure)
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in columns_to_keep if c in df.columns]]

    return df


class RegimeDataLoader:
    """
    Load and format regime detection data for dashboard visualization.

    Uses daily caching - regime is computed once per day and cached.
    """

    # Class-level cache shared across instances
    _daily_cache: Dict[str, Tuple[date, pd.DataFrame]] = {}
    _current_regime_cache: Dict[str, Tuple[date, str, datetime]] = {}

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize RegimeDataLoader.

        Args:
            data_dir: Optional path to data directory
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / 'data'

        # Initialize ATLAS regime detection model
        self.atlas_model = None
        try:
            from regime.academic_jump_model import AcademicJumpModel
            self.atlas_model = AcademicJumpModel()
            logger.info("RegimeDataLoader initialized with AcademicJumpModel")
        except Exception as e:
            logger.error(f"Failed to initialize AcademicJumpModel: {e}")

    def get_current_regime(self) -> Dict:
        """
        Get the current regime classification with metadata.

        Uses daily caching - only recomputes if date has changed.

        Returns:
            Dictionary with:
                - regime: Current regime label
                - as_of_date: Date the regime was calculated for
                - calculated_at: Timestamp when calculation was performed
                - spy_close: SPY close price for that date
                - allocation_pct: Recommended allocation percentage
        """
        today = date.today()
        cache_key = 'current_regime'

        # Check if we have a valid cache for today
        if cache_key in self._current_regime_cache:
            cached_date, cached_regime, cached_time = self._current_regime_cache[cache_key]
            if cached_date == today:
                logger.info(f"Using cached regime: {cached_regime} (calculated at {cached_time})")
                return {
                    'regime': cached_regime,
                    'as_of_date': cached_date.strftime('%Y-%m-%d'),
                    'calculated_at': cached_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'allocation_pct': self._get_allocation_pct(cached_regime),
                    'cached': True
                }

        # Need to compute fresh regime
        try:
            if self.atlas_model is None:
                logger.warning("AcademicJumpModel not initialized")
                return self._error_response("ATLAS model not available")

            # Fetch data - need 1000+ days for lookback
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = '2020-01-01'  # Ensures enough history

            logger.info(f"Computing regime detection (this may take 10-15 seconds)...")

            # Use yfinance directly (no VBT Pro dependency for dashboard)
            spy_df = _fetch_yf_data('SPY', start_date, end_date, tz='America/New_York')
            if spy_df.empty:
                return self._error_response("Failed to fetch SPY data")

            vix_df = _fetch_yf_data('^VIX', start_date, end_date, tz='America/New_York')
            if vix_df.empty:
                return self._error_response("Failed to fetch VIX data")
            vix_close = vix_df['Close']

            # Run ATLAS regime detection
            regimes, lambdas, thetas = self.atlas_model.online_inference(
                spy_df,
                lookback=1000,
                default_lambda=1.5,
                vix_data=vix_close
            )

            if regimes.empty:
                return self._error_response("No regime data returned")

            # Get the most recent regime
            current_regime = regimes.iloc[-1]
            regime_date = regimes.index[-1]
            spy_close = spy_df['Close'].iloc[-1]
            calculated_at = datetime.now()

            # Cache the result
            self._current_regime_cache[cache_key] = (today, current_regime, calculated_at)

            logger.info(f"Regime computed: {current_regime} as of {regime_date}")

            return {
                'regime': current_regime,
                'as_of_date': regime_date.strftime('%Y-%m-%d'),
                'calculated_at': calculated_at.strftime('%Y-%m-%d %H:%M:%S'),
                'spy_close': float(spy_close),
                'allocation_pct': self._get_allocation_pct(current_regime),
                'cached': False
            }

        except Exception as e:
            logger.error(f"Error computing regime: {e}", exc_info=True)
            return self._error_response(str(e))

    def get_regime_timeline(
        self,
        start_date: str,
        end_date: str,
        symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Get regime classification timeline.

        Uses daily caching - full timeline is computed once per day.

        Args:
            start_date: Start date (YYYY-MM-DD) for display filtering
            end_date: End date (YYYY-MM-DD)
            symbol: Stock symbol (default: SPY)

        Returns:
            DataFrame with columns: date, regime, price
        """
        today = date.today()
        cache_key = f"timeline_{symbol}"

        # Check daily cache
        if cache_key in self._daily_cache:
            cached_date, cached_df = self._daily_cache[cache_key]
            if cached_date == today and not cached_df.empty:
                logger.info("Using cached regime timeline")
                # Filter to requested date range
                mask = (cached_df['date'] >= pd.Timestamp(start_date)) & \
                       (cached_df['date'] <= pd.Timestamp(end_date))
                return cached_df[mask].reset_index(drop=True)

        # Compute fresh timeline
        try:
            if self.atlas_model is None:
                logger.warning("AcademicJumpModel not initialized")
                return pd.DataFrame(columns=['date', 'regime', 'price'])

            # Always fetch full history for ATLAS lookback requirement
            lookback_start = '2020-01-01'
            logger.info(f"Computing regime timeline from {lookback_start}...")

            # Use yfinance directly (no VBT Pro dependency for dashboard)
            spy_df = _fetch_yf_data(symbol, lookback_start, end_date, tz='America/New_York')
            if spy_df.empty:
                logger.error(f"Failed to fetch {symbol} data")
                return pd.DataFrame(columns=['date', 'regime', 'price'])

            vix_df = _fetch_yf_data('^VIX', lookback_start, end_date, tz='America/New_York')
            if vix_df.empty:
                logger.error("Failed to fetch VIX data")
                return pd.DataFrame(columns=['date', 'regime', 'price'])
            vix_close = vix_df['Close']

            # Run ATLAS regime detection
            regimes, lambdas, thetas = self.atlas_model.online_inference(
                spy_df,
                lookback=1000,
                default_lambda=1.5,
                vix_data=vix_close
            )

            # Format as DataFrame
            df = pd.DataFrame({
                'date': regimes.index,
                'regime': regimes.values,
                'price': spy_df['Close'].loc[regimes.index].values
            })

            # Cache full timeline
            self._daily_cache[cache_key] = (today, df)
            logger.info(f"Cached {len(df)} regime observations")

            # Return filtered to requested range
            mask = (df['date'] >= pd.Timestamp(start_date)) & \
                   (df['date'] <= pd.Timestamp(end_date))
            return df[mask].reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error loading regime timeline: {e}", exc_info=True)
            return pd.DataFrame(columns=['date', 'regime', 'price'])

    def get_regime_features(
        self,
        start_date: str,
        end_date: str,
        symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Get regime detection feature values over time.

        Calculates actual features from academic_features module.
        """
        try:
            from regime.academic_features import calculate_features

            # Use yfinance directly (no VBT Pro dependency for dashboard)
            spy_df = _fetch_yf_data(symbol, start_date, end_date, tz='America/New_York')
            if spy_df.empty:
                logger.error(f"Failed to fetch {symbol} data for features")
                return pd.DataFrame(columns=['date', 'downside_dev', 'sortino_20d', 'sortino_60d'])

            # Calculate features
            features = calculate_features(spy_df)

            df = pd.DataFrame({
                'date': features.index,
                'downside_dev': features['downside_dev'].values,
                'sortino_20d': features['sortino_20d'].values,
                'sortino_60d': features['sortino_60d'].values
            })

            logger.info(f"Loaded {len(df)} feature observations")
            return df

        except Exception as e:
            logger.error(f"Error loading regime features: {e}")
            # Return empty DataFrame instead of mock data
            return pd.DataFrame(columns=['date', 'downside_dev', 'sortino_20d', 'sortino_60d'])

    def get_regime_statistics(self, symbol: str = 'SPY') -> pd.DataFrame:
        """
        Calculate aggregate statistics per regime from cached timeline.
        """
        try:
            # Use cached timeline if available
            today = date.today()
            cache_key = f"timeline_{symbol}"

            if cache_key in self._daily_cache:
                cached_date, timeline_df = self._daily_cache[cache_key]
                if not timeline_df.empty:
                    # Calculate actual statistics from timeline
                    stats = timeline_df.groupby('regime').agg({
                        'date': 'count',
                        'price': ['mean', 'std']
                    }).round(2)

                    return timeline_df[['regime']].copy()

            logger.warning("No cached timeline for statistics")
            return pd.DataFrame(columns=['regime', 'returns'])

        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return pd.DataFrame(columns=['regime', 'returns'])

    def get_vix_status(self) -> Dict:
        """
        Get current VIX crash detection status.
        """
        try:
            from regime.vix_spike_detector import VIXSpikeDetector

            detector = VIXSpikeDetector()
            details = detector.get_details()

            logger.info(f"VIX status: {details['vix_current']:.2f}, Crash: {details['is_crash']}")
            return details

        except Exception as e:
            logger.error(f"Error getting VIX status: {e}", exc_info=True)
            return {
                'vix_current': 0.0,
                'intraday_change_pct': 0.0,
                'one_day_change_pct': 0.0,
                'three_day_change_pct': 0.0,
                'is_crash': False,
                'triggers': []
            }

    def _get_allocation_pct(self, regime: str) -> int:
        """Get allocation percentage for regime."""
        allocations = {
            'TREND_BULL': 100,
            'TREND_NEUTRAL': 70,
            'TREND_BEAR': 30,
            'CRASH': 0
        }
        return allocations.get(regime, 0)

    def _error_response(self, error_msg: str) -> Dict:
        """Return error response dict."""
        return {
            'regime': 'UNKNOWN',
            'as_of_date': None,
            'calculated_at': None,
            'error': error_msg,
            'allocation_pct': 0,
            'cached': False
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._daily_cache.clear()
        self._current_regime_cache.clear()
        logger.info("Cache cleared")
