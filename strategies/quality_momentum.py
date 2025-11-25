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

from typing import Dict, Optional, List
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
        regime: Optional[str] = None
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
                For multi-stock portfolio, requires fundamental data
            regime: Optional market regime for allocation adjustment
                ('TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH')

        Returns:
            Dictionary with v2.0 format signals:
            - 'entry_signal': Boolean Series for entry events
            - 'exit_signal': Boolean Series for exit events
            - 'stop_distance': Float Series for stop losses
            - 'quality_score': Float Series (for debugging)
            - 'momentum_score': Float Series (for debugging)

        TODO: Implementation required
        - Integrate fundamental data source for quality metrics
        - Implement quarterly rebalance logic
        - Add multi-stock portfolio support
        """
        # Regime filter: Strategy works in all regimes but may adjust sizing
        # For now, return empty signals if regime is CRASH (most conservative)
        if regime == 'CRASH':
            # In CRASH, only trade highest quality - requires special handling
            # TODO: Implement CRASH-specific filtering (only top 10% quality)
            pass

        # Calculate ATR for stop loss
        atr = self._calculate_atr(data, period=14)
        stop_distance = atr * self.atr_multiplier

        # ============================================================
        # TODO: IMPLEMENT QUALITY SCORE CALCULATION
        # ============================================================
        # For single-stock backtesting, use price-based quality proxies:
        # - Price stability (low volatility) as proxy for quality
        # - Trend consistency as proxy for earnings quality
        #
        # For multi-stock portfolio, integrate fundamental data:
        # quality_score = (
        #     0.40 * roe_rank +           # Return on Equity
        #     0.30 * earnings_quality +    # Accruals ratio (lower = better)
        #     0.30 * (1 / leverage_rank)   # Low leverage preferred
        # )
        # ============================================================

        # Placeholder: Use volatility as quality proxy for single stock
        volatility = data['Close'].pct_change().rolling(60).std()
        quality_score = 1 - volatility.rank(pct=True)  # Lower vol = higher quality

        # ============================================================
        # TODO: IMPLEMENT MOMENTUM SCORE CALCULATION
        # ============================================================
        # momentum_score = price.pct_change(252).shift(21)  # 12-month return, 1-month lag
        # ============================================================

        # Placeholder: Calculate 12-1 momentum
        if len(data) > self.momentum_lookback + self.momentum_lag:
            momentum_score = data['Close'].pct_change(self.momentum_lookback).shift(self.momentum_lag)
        else:
            momentum_score = pd.Series(0.0, index=data.index)

        # ============================================================
        # TODO: IMPLEMENT QUALITY-MOMENTUM FILTER
        # ============================================================
        # For single stock: Compare against historical percentiles
        # For multi-stock: Rank across universe
        #
        # quality_filter = quality_score.rank(pct=True) >= self.quality_threshold
        # momentum_rank = momentum_score[quality_filter].rank(pct=True)
        # entry_signal = momentum_rank >= self.momentum_threshold
        # exit_signal = ~(quality_filter & (momentum_rank >= self.exit_buffer))
        # ============================================================

        # Placeholder signals for single-stock testing
        quality_rank = quality_score.rank(pct=True)
        momentum_rank = momentum_score.rank(pct=True)

        # Entry: Both quality and momentum above thresholds
        in_entry_zone = (
            (quality_rank >= self.quality_threshold) &
            (momentum_rank >= self.momentum_threshold) &
            quality_rank.notna() &
            momentum_rank.notna()
        )

        # Exit: Either quality or momentum falls below exit buffer
        in_exit_zone = (
            (quality_rank < self.exit_buffer) |
            (momentum_rank < self.exit_buffer)
        ) & quality_rank.notna()

        # Convert states to events (state transitions)
        entry_signal = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
        exit_signal = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

        return {
            'entry_signal': entry_signal.fillna(False),
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance.fillna(0.0),
            'quality_score': quality_score.fillna(0.0),
            'momentum_score': momentum_score.fillna(0.0)
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
    # TODO: IMPLEMENT FUNDAMENTAL DATA INTEGRATION
    # ================================================================
    # The following methods need fundamental data source integration:
    #
    # def calculate_roe(self, symbol: str) -> float:
    #     """Calculate Return on Equity from financial statements."""
    #     pass
    #
    # def calculate_accruals_ratio(self, symbol: str) -> float:
    #     """Calculate earnings quality via accruals ratio."""
    #     pass
    #
    # def calculate_leverage(self, symbol: str) -> float:
    #     """Calculate debt-to-equity ratio."""
    #     pass
    #
    # def get_fundamental_data(self, symbols: List[str]) -> pd.DataFrame:
    #     """Fetch fundamental data for universe of stocks."""
    #     # Options:
    #     # - Alpha Vantage (free tier limited)
    #     # - Financial Modeling Prep API
    #     # - Yahoo Finance (yfinance library)
    #     # - Tiingo fundamentals
    #     pass
    # ================================================================
