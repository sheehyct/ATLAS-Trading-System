"""
Regime Data Loader

Loads regime detection data from ATLAS Layer 1 (academic_jump_model.py).
Provides data for regime timeline, feature evolution, and statistics visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RegimeDataLoader:
    """
    Load and format regime detection data for dashboard visualization.

    This loader interfaces with the academic jump model from regime/academic_jump_model.py
    and provides formatted data for the dashboard visualizations.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize RegimeDataLoader.

        Args:
            data_dir: Optional path to data directory
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / 'data'
        self.cache = {}

        # Initialize ATLAS regime detection model
        try:
            from regime.academic_jump_model import AcademicJumpModel
            self.atlas_model = AcademicJumpModel()
            logger.info("RegimeDataLoader initialized with AcademicJumpModel")
        except Exception as e:
            logger.error(f"Failed to initialize AcademicJumpModel: {e}")
            self.atlas_model = None

    def get_regime_timeline(
        self,
        start_date: str,
        end_date: str,
        symbol: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Get regime classification timeline using ATLAS AcademicJumpModel.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Stock symbol (default: SPY)

        Returns:
            DataFrame with columns:
                - date: DatetimeIndex
                - regime: Regime label (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH)
                - price: Close price
        """

        try:
            if self.atlas_model is None:
                logger.warning("AcademicJumpModel not initialized, returning empty data")
                return pd.DataFrame(columns=['date', 'regime', 'price'])

            logger.info(f"Loading regime data for {symbol} from {start_date} to {end_date}")

            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.cache:
                logger.info("Using cached regime data")
                return self.cache[cache_key]

            # Fetch SPY market data using VectorBT
            import vectorbtpro as vbt

            spy_data = vbt.YFData.pull(
                symbol,
                start=start_date,
                end=end_date,
                tz='America/New_York'
            )
            spy_df = spy_data.get()

            # Fetch VIX data for crash detection
            vix_data = vbt.YFData.pull(
                '^VIX',
                start=start_date,
                end=end_date,
                tz='America/New_York'
            )
            vix_close = vix_data.get()['Close']

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

            # Cache the result
            self.cache[cache_key] = df

            logger.info(f"Loaded {len(df)} regime observations from ATLAS model")
            return df

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

        Features from academic_features.py:
        - downside_dev: 10-day EWMA of downside deviation
        - sortino_20d: 20-day Sortino ratio
        - sortino_60d: 60-day Sortino ratio

        Args:
            start_date: Start date
            end_date: End date
            symbol: Stock symbol

        Returns:
            DataFrame with columns:
                - date: DatetimeIndex
                - downside_dev: Downside deviation
                - sortino_20d: 20-day Sortino ratio
                - sortino_60d: 60-day Sortino ratio
        """

        try:
            logger.info(f"Loading regime features for {symbol} from {start_date} to {end_date}")

            # PLACEHOLDER: Generate sample features
            # In production, load from academic_features.calculate_features()

            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Simulate feature evolution
            np.random.seed(42)
            downside_dev = np.abs(np.random.randn(len(dates)) * 0.01 + 0.015)
            sortino_20d = np.random.randn(len(dates)) * 0.5
            sortino_60d = np.random.randn(len(dates)) * 0.3 + 0.2

            df = pd.DataFrame({
                'date': dates,
                'downside_dev': downside_dev,
                'sortino_20d': sortino_20d,
                'sortino_60d': sortino_60d
            })

            logger.info(f"Loaded {len(df)} feature observations")
            return df

        except Exception as e:
            logger.error(f"Error loading regime features: {e}")
            return pd.DataFrame(columns=['date', 'downside_dev', 'sortino_20d', 'sortino_60d'])

    def get_regime_statistics(self, symbol: str = 'SPY') -> pd.DataFrame:
        """
        Calculate aggregate statistics per regime.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with regime-level summary stats
        """

        try:
            # PLACEHOLDER: Load actual regime-classified returns
            # For now, generate sample data

            logger.info(f"Calculating regime statistics for {symbol}")

            # Sample data
            np.random.seed(42)
            data = []

            for regime in ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']:
                n_days = np.random.randint(100, 500)
                returns = np.random.randn(n_days) * 0.01

                # Adjust mean based on regime
                if regime == 'TREND_BULL':
                    returns += 0.001
                elif regime == 'TREND_BEAR':
                    returns -= 0.0005
                elif regime == 'CRASH':
                    returns -= 0.003

                data.extend([{'regime': regime, 'returns': r} for r in returns])

            df = pd.DataFrame(data)
            logger.info(f"Calculated statistics for {len(df)} observations")

            return df

        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return pd.DataFrame(columns=['regime', 'returns'])

    def load_from_model(self, model_path: Path) -> Dict:
        """
        Load regime data from saved model file.

        Args:
            model_path: Path to saved regime model

        Returns:
            Dictionary with regime data and metadata
        """

        # PLACEHOLDER: Implement model loading
        # This would load pickled or JSON model results

        logger.warning("load_from_model not yet implemented")
        return {}

    def get_current_regime(self) -> str:
        """
        Get the current regime classification.

        Returns:
            Current regime: 'TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', or 'CRASH'
        """

        try:
            # Get regime timeline for recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = '2020-01-01'  # Get enough history for lookback

            timeline = self.get_regime_timeline(start_date, end_date)

            if timeline.empty:
                logger.warning("No regime data available")
                return 'UNKNOWN'

            # Return the most recent regime
            current = timeline.iloc[-1]['regime']
            logger.info(f"Current regime: {current}")

            return current

        except Exception as e:
            logger.error(f"Error getting current regime: {e}", exc_info=True)
            return 'UNKNOWN'

    def get_vix_status(self) -> Dict:
        """
        Get current VIX crash detection status.

        Returns:
            Dictionary with:
                - vix_current: Current VIX level
                - intraday_change_pct: Intraday percentage change
                - one_day_change_pct: 1-day percentage change
                - three_day_change_pct: 3-day percentage change
                - is_crash: Boolean indicating if crash detected
                - triggers: List of triggered thresholds
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

    def clear_cache(self):
        """Clear cached data."""
        self.cache = {}
        logger.info("Cache cleared")
