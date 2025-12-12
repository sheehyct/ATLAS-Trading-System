# IMPLEMENTATION STATUS: SKELETON - Roadmap for future development
# PRIORITY: Deferred - Focus on STRAT options execution first
# See docs/HANDOFF.md for current priorities
#
"""
IBS Mean Reversion Strategy for ATLAS Trading System v2.0

This module implements the Internal Bar Strength (IBS) Mean Reversion strategy,
which captures short-term reversals after oversold conditions.

Academic Foundation:
- Connors Research: Validated across 20+ years of data
- Superior documented Sharpe ratio: 1.5-2.0
- Daily signals create more trading opportunities than weekly patterns

Strategy Logic (from architecture spec):
- Entry: IBS < 0.20 (closed in bottom 20% of daily range)
         + Price > 200-day SMA (uptrend filter)
         + Volume > 2.0x 20-day average (MANDATORY confirmation)
- Exit: IBS > 0.80 (closed in top 80% of range)
        OR 3-day time stop (max holding period)
        OR ATR-based stop loss triggered

IBS Formula:
    IBS = (Close - Low) / (High - Low)
    Range: 0.0 (closed at low) to 1.0 (closed at high)

Performance Targets (per architecture):
- Sharpe Ratio: 1.5-2.0 (superior to alternatives)
- Turnover: High (daily signals)
- Win Rate: 65-75%
- Average Hold: 1-3 days
- CAGR: 8-12%
- Max Drawdown: -10% to -12%

Regime Compatibility:
- TREND_BULL: 5-10% allocation (works but momentum preferred)
- TREND_NEUTRAL/CHOP: 15-20% allocation (THRIVES in chop)
- TREND_BEAR: 0% (mean reversion fails in crashes)
- CRASH: 0% (risk-off)

Position Limits:
- Max 3 concurrent positions (reduce correlation risk)
- Only stocks > $50M daily volume

Implementation Status: SKELETON - Ready for implementation
Implementation Priority: PHASE 2 (after foundation strategies)

Why IBS vs 5-Day Washout (per architecture):
- Superior documented Sharpe ratio (1.5-2.0 vs unknown)
- Daily signals vs weekly = more opportunities
- Simpler logic = less parameter overfitting risk
- Better academic validation

Reference:
- docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md (lines 330-389)
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig
from utils.position_sizing import calculate_position_size_atr


class IBSMeanReversion(BaseStrategy):
    """
    IBS (Internal Bar Strength) Mean Reversion Strategy Implementation.

    Captures short-term reversals after oversold conditions. Uses the
    Internal Bar Strength indicator to identify high-probability reversal points.

    Key Advantage: Works best in TREND_NEUTRAL/CHOP markets where momentum
    strategies struggle. Provides negative correlation to momentum strategies.

    IBS Calculation:
        IBS = (Close - Low) / (High - Low)
        - IBS < 0.20: Oversold (closed near daily low)
        - IBS > 0.80: Overbought (closed near daily high)

    Entry Conditions:
    1. IBS < 0.20 (oversold condition)
    2. Price > 200-day SMA (uptrend filter - avoid catching falling knives)
    3. Volume > 2.0x 20-day average (MANDATORY - confirms institutional interest)

    Exit Conditions (any of):
    1. IBS > 0.80 (profit target - closed strong)
    2. Days held >= 3 (time stop - prevent dead money)
    3. ATR-based stop loss triggered

    Position Management:
    - Max 3 concurrent positions (correlation management)
    - Short holding period (1-3 days)
    - Quick profit-taking on IBS > 0.80

    Example Usage:
        >>> config = StrategyConfig(
        ...     name="IBS Mean Reversion",
        ...     universe="sp500",
        ...     rebalance_frequency="daily",
        ...     regime_compatibility={
        ...         'TREND_BULL': True,
        ...         'TREND_NEUTRAL': True,  # THRIVES here
        ...         'TREND_BEAR': False,
        ...         'CRASH': False
        ...     },
        ...     risk_per_trade=0.02,
        ...     max_positions=3  # Critical: limit concurrent positions
        ... )
        >>> strategy = IBSMeanReversion(config)
        >>> pf = strategy.backtest(data, initial_capital=10000)
    """

    def __init__(
        self,
        config: StrategyConfig,
        ibs_entry_threshold: float = 0.20,
        ibs_exit_threshold: float = 0.80,
        sma_period: int = 200,
        volume_multiplier: float = 2.0,
        max_hold_days: int = 3,
        atr_multiplier: float = 2.5,
        min_daily_volume: float = 50_000_000
    ):
        """
        Initialize IBS Mean Reversion strategy.

        Args:
            config: StrategyConfig with validated parameters
            ibs_entry_threshold: IBS level for oversold entry (default: 0.20)
            ibs_exit_threshold: IBS level for profit exit (default: 0.80)
            sma_period: Moving average period for trend filter (default: 200)
            volume_multiplier: Volume confirmation threshold (default: 2.0)
            max_hold_days: Maximum holding period (default: 3 days)
            atr_multiplier: Stop loss distance multiplier (default: 2.5)
            min_daily_volume: Minimum daily dollar volume (default: $50M)

        Raises:
            ValueError: If parameters outside reasonable ranges
        """
        # Validate strategy-specific parameters
        if not 0.05 <= ibs_entry_threshold <= 0.30:
            raise ValueError(
                f"ibs_entry_threshold {ibs_entry_threshold} outside range [0.05, 0.30]. "
                f"0.20 is standard."
            )

        if not 0.70 <= ibs_exit_threshold <= 0.95:
            raise ValueError(
                f"ibs_exit_threshold {ibs_exit_threshold} outside range [0.70, 0.95]. "
                f"0.80 is standard."
            )

        if ibs_entry_threshold >= ibs_exit_threshold:
            raise ValueError(
                f"ibs_entry_threshold {ibs_entry_threshold} must be < ibs_exit_threshold {ibs_exit_threshold}"
            )

        if not 1 <= max_hold_days <= 10:
            raise ValueError(
                f"max_hold_days {max_hold_days} outside range [1, 10]. "
                f"3 days is standard for IBS."
            )

        self.ibs_entry_threshold = ibs_entry_threshold
        self.ibs_exit_threshold = ibs_exit_threshold
        self.sma_period = sma_period
        self.volume_multiplier = volume_multiplier
        self.max_hold_days = max_hold_days
        self.atr_multiplier = atr_multiplier
        self.min_daily_volume = min_daily_volume

        # Call parent constructor
        super().__init__(config)

    def validate_parameters(self) -> bool:
        """
        Validate strategy-specific parameters.

        Returns:
            True if all parameters valid

        Raises:
            AssertionError: If validation fails
        """
        assert 0.05 <= self.ibs_entry_threshold <= 0.30, \
            f"ibs_entry_threshold {self.ibs_entry_threshold} outside range [0.05, 0.30]"
        assert 0.70 <= self.ibs_exit_threshold <= 0.95, \
            f"ibs_exit_threshold {self.ibs_exit_threshold} outside range [0.70, 0.95]"
        assert self.ibs_entry_threshold < self.ibs_exit_threshold, \
            "ibs_entry_threshold must be < ibs_exit_threshold"
        return True

    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals for IBS Mean Reversion strategy.

        Signal Logic:
        1. Calculate IBS: (Close - Low) / (High - Low)
        2. Calculate 200-day SMA for trend filter
        3. Calculate 20-day volume average for confirmation
        4. Entry: IBS < 0.20 AND Close > SMA200 AND Volume > 2x avg
        5. Exit: IBS > 0.80 OR time_stop (3 days) OR ATR stop

        Args:
            data: OHLCV DataFrame with DatetimeIndex
                Required columns: Open, High, Low, Close, Volume
            regime: Optional market regime for filtering
                ('TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH')

        Returns:
            Dictionary with v2.0 format signals:
            - 'entry_signal': Boolean Series for entry events
            - 'exit_signal': Boolean Series for exit events
            - 'stop_distance': Float Series for stop losses
            - 'ibs': Float Series (for debugging/analysis)
            - 'volume_confirmed': Boolean Series (for debugging)
            - 'above_sma': Boolean Series (for debugging)

        Note:
            Time-based exit (max_hold_days) must be handled by the backtester
            or position manager, as VBT from_signals doesn't natively support it.
        """
        # Regime filter: Don't trade in TREND_BEAR or CRASH
        if regime and not self.should_trade_in_regime(regime):
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': pd.Series(0.0, index=data.index),
                'ibs': pd.Series(0.0, index=data.index),
                'volume_confirmed': pd.Series(False, index=data.index),
                'above_sma': pd.Series(False, index=data.index)
            }

        # ============================================================
        # CALCULATE IBS (Internal Bar Strength)
        # ============================================================
        # IBS = (Close - Low) / (High - Low)
        # Handle division by zero when High == Low (doji bars)
        bar_range = data['High'] - data['Low']
        bar_range = bar_range.replace(0, np.nan)  # Avoid division by zero

        ibs = (data['Close'] - data['Low']) / bar_range
        ibs = ibs.fillna(0.5)  # Doji bars get neutral IBS

        # ============================================================
        # CALCULATE TREND FILTER (200-day SMA)
        # ============================================================
        sma_200 = data['Close'].rolling(window=self.sma_period, min_periods=self.sma_period).mean()
        above_sma = data['Close'] > sma_200

        # ============================================================
        # CALCULATE VOLUME CONFIRMATION
        # ============================================================
        volume_ma_20 = data['Volume'].rolling(window=20, min_periods=20).mean()
        volume_confirmed = (
            (data['Volume'] > (volume_ma_20 * self.volume_multiplier)) &
            volume_ma_20.notna()
        )

        # ============================================================
        # CALCULATE ATR FOR STOP LOSS
        # ============================================================
        atr = self._calculate_atr(data, period=14)
        stop_distance = atr * self.atr_multiplier

        # ============================================================
        # ENTRY CONDITIONS
        # ============================================================
        # 1. IBS < 0.20 (oversold)
        # 2. Price > 200 SMA (uptrend)
        # 3. Volume > 2x average (MANDATORY)
        in_entry_zone = (
            (ibs < self.ibs_entry_threshold) &
            above_sma &
            volume_confirmed &
            sma_200.notna() &
            atr.notna()
        )

        # ============================================================
        # EXIT CONDITIONS
        # ============================================================
        # 1. IBS > 0.80 (profit target - closed strong)
        # 2. ATR stop handled separately by VBT
        # Note: Time stop (max_hold_days) requires custom implementation
        in_exit_zone = (ibs > self.ibs_exit_threshold) & ibs.notna()

        # Convert states to events (state transitions)
        entry_signal = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
        exit_signal = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

        return {
            'entry_signal': entry_signal.fillna(False),
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance.fillna(0.0),
            'ibs': ibs.fillna(0.5),
            'volume_confirmed': volume_confirmed.fillna(False),
            'above_sma': above_sma.fillna(False)
        }

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes for IBS Mean Reversion.

        Uses ATR-based position sizing with capital constraints.
        Position sizes are intentionally smaller (max 3 positions)
        to allow for multiple concurrent mean reversion trades.

        Args:
            data: OHLCV DataFrame
            capital: Current account capital
            stop_distance: Stop loss distances from generate_signals()

        Returns:
            Position sizes as pd.Series of share counts
        """
        atr = self._calculate_atr(data, period=14)

        # Calculate base position sizes
        position_sizes, actual_risks, constrained = calculate_position_size_atr(
            init_cash=capital,
            close=data['Close'],
            atr=atr,
            atr_multiplier=self.atr_multiplier,
            risk_pct=self.config.risk_per_trade
        )

        # Scale down for multiple concurrent positions
        # With max_positions=3, each position gets ~33% of allocated capital
        position_sizes = (position_sizes / self.config.max_positions).astype(int)

        return position_sizes

    def get_strategy_name(self) -> str:
        """Return strategy name for logging and reporting."""
        return "IBS Mean Reversion"

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
    # HELPER METHODS FOR ANALYSIS
    # ================================================================

    def get_current_ibs(self, data: pd.DataFrame) -> float:
        """
        Get current IBS value for real-time monitoring.

        Args:
            data: Recent OHLCV data

        Returns:
            Current IBS value (0.0 to 1.0)
        """
        if len(data) == 0:
            return 0.5

        last_row = data.iloc[-1]
        bar_range = last_row['High'] - last_row['Low']

        if bar_range == 0:
            return 0.5  # Doji bar

        return (last_row['Close'] - last_row['Low']) / bar_range

    def is_oversold(self, data: pd.DataFrame) -> bool:
        """
        Check if current bar is in oversold territory.

        Args:
            data: Recent OHLCV data

        Returns:
            True if IBS < entry threshold
        """
        return self.get_current_ibs(data) < self.ibs_entry_threshold

    def is_overbought(self, data: pd.DataFrame) -> bool:
        """
        Check if current bar is in overbought territory.

        Args:
            data: Recent OHLCV data

        Returns:
            True if IBS > exit threshold
        """
        return self.get_current_ibs(data) > self.ibs_exit_threshold

    # ================================================================
    # TODO: IMPLEMENT TIME-BASED EXIT
    # ================================================================
    # VBT from_signals doesn't natively support time-based exits.
    # Options for implementation:
    #
    # 1. Custom VBT callback (preferred):
    #    def time_exit_callback(entries, exits, i, col, ...):
    #        if bars_since_entry >= max_hold_days:
    #            return True
    #        return exits[i, col]
    #
    # 2. Post-process trades:
    #    for trade in pf.trades.records:
    #        if trade.duration >= max_hold_days:
    #            force_exit(trade)
    #
    # 3. Generate time exits in signal generation:
    #    Track entry bars and generate exit signal after max_hold_days
    #    This requires stateful signal generation (more complex)
    # ================================================================
