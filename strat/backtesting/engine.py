"""
Backtest Engine - Top-level Orchestrator

Coordinates the full backtesting pipeline:
1. Load/fetch OHLCV data for each symbol/timeframe
2. Initialize price provider and capital simulator
3. Run bar simulation for each symbol/timeframe
4. Collect results and produce analytics

Usage:
    from strat.backtesting.engine import BacktestEngine
    from strat.backtesting.config import BacktestConfig

    config = BacktestConfig(symbols=['SPY'], timeframes=['1D'])
    engine = BacktestEngine(config)
    results = engine.run()
    print(results.summary())
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd

from strat.backtesting.config import BacktestConfig
from strat.backtesting.simulation.bar_simulator import BarSimulator
from strat.backtesting.simulation.position_tracker import SimulatedPosition
from strat.backtesting.simulation.capital_simulator import CapitalSimulator
from strat.backtesting.data_providers.base import OptionsPriceProvider
from strat.backtesting.analytics.trade_recorder import TradeRecorder
from strat.backtesting.analytics.results_formatter import BacktestResults, ResultsFormatter

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Top-level backtest orchestrator.

    Manages data loading, simulation execution, and results collection
    across multiple symbols and timeframes.
    """

    def __init__(
        self,
        config: BacktestConfig,
        price_provider: Optional[OptionsPriceProvider] = None,
    ):
        self._config = config
        self._price_provider = price_provider
        self._capital_sim = None
        self._trade_recorder = TradeRecorder()

    def run(self) -> 'BacktestResults':
        """
        Execute the full backtest across all symbols and timeframes.

        Returns:
            BacktestResults with all trades and summary statistics
        """
        # Validate config
        issues = self._config.validate()
        if issues:
            for issue in issues:
                logger.warning("Config issue: %s", issue)

        # Initialize capital simulator
        if self._config.capital_tracking_enabled:
            self._capital_sim = CapitalSimulator(self._config)

        all_trades: List[SimulatedPosition] = []

        for symbol in self._config.symbols:
            for timeframe in self._config.timeframes:
                logger.info("Running backtest: %s %s (%s to %s)",
                            symbol, timeframe,
                            self._config.start_date, self._config.end_date)

                # Load data
                df = self._load_data(symbol, timeframe)
                if df is None or df.empty:
                    logger.warning("No data for %s %s, skipping", symbol, timeframe)
                    continue

                # Run simulation
                sim = BarSimulator(
                    config=self._config,
                    price_provider=self._price_provider,
                    capital_sim=self._capital_sim,
                )
                trades = sim.run(df, symbol, timeframe)
                all_trades.extend(trades)

                logger.info("%s %s: %d trades", symbol, timeframe, len(trades))

        # Record all trades
        for trade in all_trades:
            self._trade_recorder.record(trade)

        # Format results
        results = ResultsFormatter.format(
            all_trades,
            self._config,
            capital_summary=self._capital_sim.get_summary() if self._capital_sim else None,
        )

        logger.info("Backtest complete: %d total trades", len(all_trades))
        return results

    def _load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a symbol/timeframe.

        Uses VBT Pro's AlpacaData with split adjustment (matching the
        existing backtest pipeline's data source).
        """
        try:
            import vectorbtpro as vbt

            # Map timeframe to Alpaca format
            tf_map = {
                '1H': '1 hour',
                '1D': '1 day',
                '1W': '1 week',
                '1M': '1 month',
            }
            alpaca_tf = tf_map.get(timeframe, '1 day')

            data = vbt.AlpacaData.pull(
                symbol,
                start=self._config.start_date,
                end=self._config.end_date,
                timeframe=alpaca_tf,
                adjustment='split',
            )

            df = data.get()

            # For 1H: need market-aligned bars (EQUITY-34 fix)
            if timeframe == '1H':
                df = self._align_hourly_bars(symbol, df)

            # Filter market hours
            if timeframe == '1H':
                df = self._filter_market_hours(df)

            logger.info("Loaded %d bars for %s %s", len(df), symbol, timeframe)
            return df

        except Exception as e:
            logger.error("Data load failed for %s %s: %s", symbol, timeframe, e)
            return None

    def _align_hourly_bars(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align hourly bars to market open (9:30, 10:30, ...).

        EQUITY-34 critical fix: Alpaca returns clock-aligned (10:00, 11:00)
        but STRAT patterns need market-aligned (9:30, 10:30).

        If df already has minute data, resample. Otherwise attempt
        to fetch minute data and resample.
        """
        try:
            import vectorbtpro as vbt

            # Fetch minute data for proper resampling
            minute_data = vbt.AlpacaData.pull(
                symbol,
                start=self._config.start_date,
                end=self._config.end_date,
                timeframe='1 minute',
                adjustment='split',
            )
            minute_df = minute_data.get()

            if minute_df is not None and len(minute_df) > 0:
                # Resample with 30-minute offset for market alignment
                resampled = minute_df.resample('1h', offset='30min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                }).dropna()
                return resampled

        except Exception as e:
            logger.warning("Hourly alignment fallback to raw data: %s", e)

        return df

    @staticmethod
    def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Filter to market hours only (9:30-16:00 ET)."""
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize('America/New_York')
            except Exception:
                return df

        mask = (df.index.hour >= 9) & (
            (df.index.hour > 9) | (df.index.minute >= 30)
        ) & (df.index.hour < 16)
        return df[mask]
