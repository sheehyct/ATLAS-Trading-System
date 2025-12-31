# IMPLEMENTATION STATUS: ACTIVE - Session EQUITY-35
# PRIORITY: Phase 1 - All-weather strategy for ATLAS portfolio
# Data Source: Alpha Vantage (fundamental data with 90-day caching)
#
"""
Quality-Momentum Combination Strategy for ATLAS Trading System v2.0

This module implements the Quality-Momentum strategy, combining quality factor
screening with momentum ranking for all-weather performance.

Academic Foundation:
- Asness, Frazzini, Pedersen (2018): "Quality Minus Junk"
- Jegadeesh & Titman (1993): Momentum effect documentation
- Documented Sharpe ratio: 1.55 (validated in academic research)

Strategy Logic (from architecture spec):
- Quality Filter: Remove bottom 50% by quality score
  - 40% ROE rank
  - 30% Earnings quality (accruals ratio)
  - 30% Inverse leverage rank
- Momentum Score: 12-month return with 1-month lag (12-1 momentum)
- Entry: Top 50% of quality-filtered stocks by momentum
- Exit: Quarterly rebalance OR quality/momentum rank drops below 40% (buffer)
- Position Count: 20-30 stocks (equal weight or volatility-scaled)

Performance Targets (per architecture):
- Sharpe Ratio: 1.3-1.7 (validated 1.55 in research)
- Turnover: 50-80% quarterly
- Win Rate: 55-65%
- CAGR: 15-22%
- Max Drawdown: -18% to -22%

Regime Compatibility (UNIQUE - works in ALL regimes):
- TREND_BULL: 25-30% allocation (momentum enhances returns)
- TREND_NEUTRAL: 30-35% allocation (quality protects)
- TREND_BEAR: 20-30% allocation (DEFENSIVE - quality prevents blow-up)
- CRASH: 10-15% allocation (only highest quality if any)

CAPITAL REQUIREMENTS:
- Minimum Viable: $10,000 (for 20+ stock diversification)
- Optimal: $25,000+ (full 30-stock portfolio)

Implementation Status: SKELETON - Ready for implementation
Implementation Priority: PHASE 1 (highest priority - all-weather strategy)

Reference:
- docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md (lines 210-268)
"""

from typing import Dict, Optional, List, Union
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig
from utils.position_sizing import calculate_position_size_atr


class QualityMomentum(BaseStrategy):
    """
    Quality-Momentum Combination Strategy Implementation.

    Combines quality factor screening with momentum ranking to create an
    all-weather strategy that works across ALL market regimes.

    Key Advantage: Quality filter reduces left-tail risk in bear markets
    while momentum enhances returns in bull markets.

    Quality Metrics (weighted composite):
    - ROE (Return on Equity): 40% weight
    - Earnings Quality (low accruals): 30% weight
    - Low Leverage: 30% weight

    Entry Conditions:
    1. Pass quality filter (top 50% by quality score)
    2. Rank in top 50% by 12-1 momentum among quality stocks
    3. Quarterly rebalance date

    Exit Conditions:
    1. Quarterly rebalance (refresh rankings)
    2. Quality or momentum rank drops below 40% (buffer to reduce turnover)

    Position Sizing:
    - Equal weight OR inverse-volatility weighted (configurable)
    - No leverage allowed (quality premium sufficient)

    Example Usage:
        >>> config = StrategyConfig(
        ...     name="Quality-Momentum",
        ...     universe="sp500",
        ...     rebalance_frequency="quarterly",
        ...     regime_compatibility={
        ...         'TREND_BULL': True,
        ...         'TREND_NEUTRAL': True,
        ...         'TREND_BEAR': True,  # UNIQUE: Works in bear markets
        ...         'CRASH': True        # Reduced allocation but still active
        ...     },
        ...     risk_per_trade=0.02,
        ...     max_positions=20
        ... )
        >>> strategy = QualityMomentum(config)
        >>> pf = strategy.backtest(data, initial_capital=25000)
    """

    def __init__(
        self,
        config: StrategyConfig,
        momentum_lookback: int = 252,
        momentum_lag: int = 21,
        quality_threshold: float = 0.50,
        momentum_threshold: float = 0.50,
        exit_buffer: float = 0.40,
        atr_multiplier: float = 2.5,
        position_weighting: str = 'equal'
    ):
        """
        Initialize Quality-Momentum strategy.

        Args:
            config: StrategyConfig with validated parameters
            momentum_lookback: Days for momentum calculation (default: 252 = 1 year)
            momentum_lag: Days to skip recent performance (default: 21 = 1 month)
                This avoids short-term reversal effect
            quality_threshold: Minimum quality percentile for entry (default: 0.50)
            momentum_threshold: Minimum momentum percentile among quality stocks (default: 0.50)
            exit_buffer: Exit threshold buffer to reduce turnover (default: 0.40)
            atr_multiplier: Stop loss distance multiplier (default: 2.5)
            position_weighting: 'equal' or 'inverse_vol' (default: 'equal')

        Raises:
            ValueError: If parameters outside reasonable ranges
        """
        # Validate strategy-specific parameters
        if not 126 <= momentum_lookback <= 504:
            raise ValueError(
                f"momentum_lookback {momentum_lookback} outside range [126, 504]. "
                f"252 (1 year) is standard."
            )

        if not 0.0 < quality_threshold <= 1.0:
            raise ValueError(
                f"quality_threshold must be between 0 and 1, got {quality_threshold}"
            )

        if position_weighting not in ['equal', 'inverse_vol']:
            raise ValueError(
                f"position_weighting must be 'equal' or 'inverse_vol', got {position_weighting}"
            )

        self.momentum_lookback = momentum_lookback
        self.momentum_lag = momentum_lag
        self.quality_threshold = quality_threshold
        self.momentum_threshold = momentum_threshold
        self.exit_buffer = exit_buffer
        self.atr_multiplier = atr_multiplier
        self.position_weighting = position_weighting

        # Call parent constructor (validates config)
        super().__init__(config)

    def validate_parameters(self) -> bool:
        """
        Validate strategy-specific parameters.

        Returns:
            True if all parameters valid

        Raises:
            AssertionError: If validation fails
        """
        assert 126 <= self.momentum_lookback <= 504, \
            f"momentum_lookback {self.momentum_lookback} outside range [126, 504]"
        assert 0.0 < self.quality_threshold <= 1.0, \
            f"quality_threshold must be between 0 and 1"
        assert self.exit_buffer < self.quality_threshold, \
            f"exit_buffer {self.exit_buffer} must be < quality_threshold {self.quality_threshold}"
        return True

    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: Optional[str] = None,
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
        fundamental_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals for Quality-Momentum strategy.

        Signal Logic:
        1. Calculate quality score (ROE, earnings quality, leverage)
        2. Filter to top 50% by quality
        3. Calculate 12-1 momentum (12-month return, 1-month lag)
        4. Rank by momentum among quality stocks
        5. Entry: Top 50% momentum among quality stocks
        6. Exit: Falls below 40% threshold (buffer)

        Args:
            data: OHLCV DataFrame with DatetimeIndex
                For single-stock backtesting, uses price-based quality proxies
                For multi-stock portfolio, this is the index-aligned combined data
            regime: Optional market regime for allocation adjustment
                ('TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH')
            universe_data: Optional dict of symbol -> OHLCV DataFrames
                When provided, enables portfolio mode with cross-sectional ranking
            fundamental_data: Optional DataFrame with fundamental metrics
                When provided, uses actual quality scores instead of proxies

        Returns:
            Dictionary with v2.0 format signals:
            - 'entry_signal': Boolean Series for entry events
            - 'exit_signal': Boolean Series for exit events
            - 'stop_distance': Float Series for stop losses
            - 'quality_score': Float Series (for debugging)
            - 'momentum_score': Float Series (for debugging)
        """
        # Calculate ATR for stop loss
        atr = self._calculate_atr(data, period=14)
        stop_distance = atr * self.atr_multiplier

        # CRASH regime: Only trade highest quality (top 10%)
        # Other regimes: Strategy works in all with different allocations
        crash_mode = regime == 'CRASH'
        quality_threshold = 0.90 if crash_mode else self.quality_threshold

        # Portfolio mode: Use actual fundamental data and cross-sectional ranking
        if universe_data is not None and fundamental_data is not None:
            return self._generate_portfolio_signals(
                data=data,
                universe_data=universe_data,
                fundamental_data=fundamental_data,
                stop_distance=stop_distance,
                quality_threshold=quality_threshold
            )

        # Single-stock mode: Use price-based quality proxies
        return self._generate_single_stock_signals(
            data=data,
            stop_distance=stop_distance,
            quality_threshold=quality_threshold
        )

    def _generate_single_stock_signals(
        self,
        data: pd.DataFrame,
        stop_distance: pd.Series,
        quality_threshold: float
    ) -> Dict[str, pd.Series]:
        """
        Generate signals for single-stock backtesting.

        Uses price-based quality proxies:
        - Low volatility as proxy for quality (stable companies)
        - 12-1 momentum for momentum score

        Args:
            data: OHLCV DataFrame
            stop_distance: ATR-based stop distances
            quality_threshold: Minimum quality percentile for entry

        Returns:
            Signal dictionary with v2.0 format
        """
        # Quality proxy: Inverse volatility (lower vol = higher quality)
        volatility = data['Close'].pct_change().rolling(60).std()
        quality_score = 1 - volatility.rank(pct=True)

        # Momentum: 12-month return with 1-month lag
        if len(data) > self.momentum_lookback + self.momentum_lag:
            momentum_score = data['Close'].pct_change(self.momentum_lookback).shift(self.momentum_lag)
        else:
            momentum_score = pd.Series(0.0, index=data.index)

        # Calculate percentile ranks for filtering
        quality_rank = quality_score.rank(pct=True)
        momentum_rank = momentum_score.rank(pct=True)

        # Entry zone: Both quality and momentum above thresholds
        in_entry_zone = (
            (quality_rank >= quality_threshold) &
            (momentum_rank >= self.momentum_threshold) &
            quality_rank.notna() &
            momentum_rank.notna()
        )

        # Exit zone: Either quality or momentum falls below exit buffer (40%)
        in_exit_zone = (
            (quality_rank < self.exit_buffer) |
            (momentum_rank < self.exit_buffer)
        ) & quality_rank.notna()

        # Convert states to events (state transitions) per reference implementation
        entry_signal = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
        exit_signal = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

        return {
            'entry_signal': entry_signal.fillna(False),
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance.fillna(0.0),
            'quality_score': quality_score.fillna(0.0),
            'momentum_score': momentum_score.fillna(0.0)
        }

    def _generate_portfolio_signals(
        self,
        data: pd.DataFrame,
        universe_data: Dict[str, pd.DataFrame],
        fundamental_data: pd.DataFrame,
        stop_distance: pd.Series,
        quality_threshold: float
    ) -> Dict[str, pd.Series]:
        """
        Generate signals for portfolio mode with cross-sectional ranking.

        Uses actual fundamental data for quality scores and ranks
        across the universe for both quality and momentum.

        Args:
            data: Index-aligned OHLCV DataFrame (for signal index)
            universe_data: Dict of symbol -> OHLCV DataFrames
            fundamental_data: DataFrame with quality metrics
            stop_distance: ATR-based stop distances
            quality_threshold: Minimum quality percentile for entry

        Returns:
            Signal dictionary with v2.0 format
        """
        # Calculate quality scores from fundamental data
        quality_df = self.calculate_quality_scores_from_data(fundamental_data)

        # Filter to top stocks by quality
        quality_filtered_symbols = self.filter_by_quality(quality_df, quality_threshold)

        if not quality_filtered_symbols:
            # No stocks pass quality filter
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': stop_distance.fillna(0.0),
                'quality_score': pd.Series(0.0, index=data.index),
                'momentum_score': pd.Series(0.0, index=data.index)
            }

        # Calculate momentum for quality-filtered symbols
        momentum_df = self.calculate_momentum_scores(quality_filtered_symbols, universe_data)

        if momentum_df.empty:
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': stop_distance.fillna(0.0),
                'quality_score': pd.Series(0.0, index=data.index),
                'momentum_score': pd.Series(0.0, index=data.index)
            }

        # Filter to top momentum stocks
        momentum_threshold_value = momentum_df['momentum_rank'].quantile(1 - self.momentum_threshold)
        selected_symbols = momentum_df[
            momentum_df['momentum_rank'] >= momentum_threshold_value
        ]['symbol'].tolist()

        # Generate entry signals for selected symbols on rebalance days
        entry_signal = pd.Series(False, index=data.index)
        exit_signal = pd.Series(False, index=data.index)

        for date in data.index:
            if self.is_rebalance_day(date):
                # Entry on rebalance day if symbols are selected
                if selected_symbols:
                    entry_signal.loc[date] = True

        # Get average quality and momentum scores for debugging
        avg_quality = quality_df[quality_df['symbol'].isin(selected_symbols)]['quality_score'].mean()
        avg_momentum = momentum_df[momentum_df['symbol'].isin(selected_symbols)]['momentum'].mean()

        quality_score = pd.Series(avg_quality if pd.notna(avg_quality) else 0.0, index=data.index)
        momentum_score = pd.Series(avg_momentum if pd.notna(avg_momentum) else 0.0, index=data.index)

        return {
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
            'stop_distance': stop_distance.fillna(0.0),
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'selected_symbols': selected_symbols  # Bonus: which symbols were selected
        }

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes for Quality-Momentum strategy.

        Uses either equal-weight or inverse-volatility weighting
        based on configuration.

        Args:
            data: OHLCV DataFrame
            capital: Current account capital
            stop_distance: Stop loss distances from generate_signals()

        Returns:
            Position sizes as pd.Series of share counts

        TODO: Implementation details
        - For multi-stock portfolio: Distribute across max_positions
        - For inverse_vol: Weight by 1/volatility
        """
        atr = self._calculate_atr(data, period=14)

        # Use ATR-based position sizing
        position_sizes, actual_risks, constrained = calculate_position_size_atr(
            init_cash=capital,
            close=data['Close'],
            atr=atr,
            atr_multiplier=self.atr_multiplier,
            risk_pct=self.config.risk_per_trade
        )

        # Scale by max_positions for portfolio allocation
        # In multi-stock mode, each position gets (1/max_positions) of capital
        position_sizes = (position_sizes / self.config.max_positions).astype(int)

        return position_sizes

    def get_strategy_name(self) -> str:
        """Return strategy name for logging and reporting."""
        return "Quality-Momentum Combination"

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            data: OHLCV DataFrame
            period: Lookback period (default: 14)

        Returns:
            ATR as pd.Series
        """
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()

        return atr

    # ================================================================
    # FUNDAMENTAL DATA INTEGRATION - Session EQUITY-35
    # ================================================================

    def calculate_quality_scores(
        self,
        symbols: List[str],
        max_new_fetches: int = 6
    ) -> pd.DataFrame:
        """
        Calculate quality scores for a universe of stocks.

        Uses Alpha Vantage to fetch fundamental data (ROE, accruals, leverage)
        and calculates composite quality scores per architecture spec:

        Quality Score = 0.40 * ROE_rank + 0.30 * Earnings_quality + 0.30 * Inverse_leverage

        Args:
            symbols: List of stock symbols to evaluate
            max_new_fetches: Maximum new symbols to fetch from API per call.
                            Each symbol requires 4 API calls.
                            With 25 calls/day limit, default is 6 symbols.

        Returns:
            DataFrame with columns: symbol, roe, accruals_ratio, debt_to_equity,
                                   roe_rank, earnings_quality, inverse_leverage, quality_score

        Note:
            Cached data is used when available (90-day cache expiration).
            New API calls are rate-limited to avoid exceeding daily quota.
        """
        from integrations.alphavantage_fundamentals import AlphaVantageFundamentals

        fetcher = AlphaVantageFundamentals()

        # Fetch metrics for all symbols (uses caching)
        metrics_df = fetcher.get_quality_metrics_batch(symbols, max_new_fetches)

        # Calculate composite quality scores
        quality_df = fetcher.calculate_quality_scores(metrics_df)

        return quality_df

    def calculate_quality_scores_from_data(
        self,
        fundamental_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate quality scores from pre-fetched fundamental data.

        Useful for testing with mock data or pre-cached fundamentals.

        Args:
            fundamental_data: DataFrame with columns: symbol, roe, accruals_ratio, debt_to_equity

        Returns:
            DataFrame with additional columns: roe_rank, earnings_quality,
                                              inverse_leverage, quality_score
        """
        df = fundamental_data.copy()

        # ROE rank: higher ROE = higher rank (0-1 percentile)
        df['roe_rank'] = df['roe'].rank(pct=True, na_option='bottom')

        # Earnings quality: lower accruals ratio = higher quality
        df['accruals_rank'] = df['accruals_ratio'].rank(pct=True, na_option='top')
        df['earnings_quality'] = 1 - df['accruals_rank']

        # Leverage rank: lower debt/equity = higher rank
        df['leverage_rank'] = df['debt_to_equity'].rank(pct=True, na_option='top')
        df['inverse_leverage'] = 1 - df['leverage_rank']

        # Composite quality score (per architecture spec)
        df['quality_score'] = (
            0.40 * df['roe_rank'] +
            0.30 * df['earnings_quality'] +
            0.30 * df['inverse_leverage']
        )

        return df

    def filter_by_quality(
        self,
        quality_df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Filter to top stocks by quality score.

        Args:
            quality_df: DataFrame from calculate_quality_scores()
            threshold: Minimum quality percentile (default: self.quality_threshold = 0.50)

        Returns:
            List of symbols that pass quality filter
        """
        threshold = threshold or self.quality_threshold
        quality_percentile = quality_df['quality_score'].quantile(1 - threshold)
        passing = quality_df[quality_df['quality_score'] >= quality_percentile]
        return passing['symbol'].tolist()

    def calculate_momentum_scores(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate 12-1 momentum for multiple symbols.

        12-1 momentum = 12-month return with 1-month lag
        This avoids short-term reversal effect at monthly frequency.

        Args:
            symbols: List of quality-filtered symbols
            price_data: Dict mapping symbol to OHLCV DataFrame

        Returns:
            DataFrame with columns: symbol, momentum, momentum_rank
        """
        momentum_scores = []

        for symbol in symbols:
            if symbol not in price_data:
                continue

            df = price_data[symbol]

            if len(df) < self.momentum_lookback + self.momentum_lag:
                continue

            # 12-month return with 1-month lag
            momentum = df['Close'].pct_change(self.momentum_lookback).shift(self.momentum_lag).iloc[-1]

            if pd.notna(momentum):
                momentum_scores.append({
                    'symbol': symbol,
                    'momentum': momentum
                })

        if not momentum_scores:
            return pd.DataFrame(columns=['symbol', 'momentum', 'momentum_rank'])

        result = pd.DataFrame(momentum_scores)

        # Rank momentum (higher is better)
        result['momentum_rank'] = result['momentum'].rank(pct=True, na_option='bottom')

        return result

    def is_rebalance_day(self, date: pd.Timestamp) -> bool:
        """
        Check if date is a quarterly rebalance day.

        Rebalance months: January, April, July, October
        Rebalance happens on first trading day of the month.

        Args:
            date: Date to check

        Returns:
            True if this is a rebalance day
        """
        rebalance_months = [1, 4, 7, 10]  # Q1, Q2, Q3, Q4

        if date.month in rebalance_months and date.day <= 5:
            # First trading day of rebalance month (within first 5 calendar days)
            return True

        return False

    def get_next_rebalance_date(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """
        Get the next quarterly rebalance date.

        Args:
            current_date: Current date

        Returns:
            Next rebalance date (first of next rebalance month)
        """
        rebalance_months = [1, 4, 7, 10]

        current_month = current_date.month
        current_year = current_date.year

        # Find next rebalance month
        for month in rebalance_months:
            if month > current_month:
                return pd.Timestamp(year=current_year, month=month, day=1)

        # Next year
        return pd.Timestamp(year=current_year + 1, month=1, day=1)
