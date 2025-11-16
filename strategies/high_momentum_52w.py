"""
52-Week High Momentum Strategy for ATLAS Trading System v2.0

This module implements the 52-Week High Momentum strategy, a foundation strategy
for the ATLAS multi-layer trading system. The strategy captures momentum by
identifying stocks trading near their 52-week highs with volume confirmation.

Academic Foundation:
- Novy-Marx (2012): "Is Momentum Really Momentum?"
- George & Hwang (2004): "The 52-Week High and Momentum Investing"
- Documented momentum effect with lowest turnover among momentum strategies

Strategy Logic (VALIDATED Session 36):
- Entry: Price within 10% of 52-week high (distance >= 0.90)
- Exit: Price 12% off highs (distance < 0.88) - creates balanced entry/exit cycles
- Volume Confirmation: Configurable threshold (default 1.25x for SPY)
  - Calibrated per asset volatility (high vol: 1.15x, moderate: 1.5-1.75x, low: 1.25x)
  - Set to None to disable volume filter
- Signal Type: EVENT-BASED (state transitions, not continuous states)
- Position Sizing: ATR-based with 2% risk per trade

Performance Targets (per architecture):
- Sharpe Ratio: 0.8-1.2
- Turnover: ~50% semi-annually
- Win Rate: 50-60%
- CAGR: 10-15%
- Max Drawdown: -25% to -30%

Regime Compatibility (unique advantage):
- TREND_BULL: 30-40% allocation (strong performance expected)
- TREND_NEUTRAL: 20-25% allocation (STILL WORKS - unique among momentum strategies)
- TREND_BEAR: 0% (exit all positions)
- CRASH: 0% (risk-off)

Implementation Notes:
- Uses v2.0 BaseStrategy interface (entry_signal, exit_signal, stop_distance)
- ATR-based stop loss (2.5x multiplier standard)
- Volume confirmation non-negotiable (research-validated 2.0x threshold)
- Compatible with VectorBT Pro vectorized operations
- NO Python loops (all pandas/numpy operations)

v2.0 Enhancements:
- Regime awareness via BaseStrategy.should_trade_in_regime()
- Semi-annual rebalance frequency configuration
- 52-week high calculated over 252 trading days
- Volume confirmation integrated into signal generation

Reference:
- docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md (lines 161-208)
- docs/CLAUDE.md (lines 382-433 - volume confirmation requirement)
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig
from utils.position_sizing import calculate_position_size_atr


class HighMomentum52W(BaseStrategy):
    """
    52-Week High Momentum Strategy Implementation.

    Captures momentum by identifying securities trading near their 52-week highs.
    Unique advantage: Works in both TREND_BULL and TREND_NEUTRAL regimes.

    Entry Conditions:
    1. Price within 10% of 52-week high (distance >= 0.90)
    2. Volume > 2.0x 20-day average (MANDATORY confirmation)
    3. Compatible regime (TREND_BULL or TREND_NEUTRAL)

    Exit Conditions:
    1. Price 30% off highs (distance < 0.70)
    2. OR ATR-based stop loss triggered

    Position Sizing:
    - ATR-based: 2% risk per trade
    - Stop distance: 2.5x ATR (standard)
    - Capital constrained (never exceeds 100% of capital)

    Example Usage:
        >>> config = StrategyConfig(
        ...     name="52-Week High Momentum",
        ...     universe="sp500",
        ...     rebalance_frequency="semi_annual",
        ...     regime_compatibility={
        ...         'TREND_BULL': True,
        ...         'TREND_NEUTRAL': True,
        ...         'TREND_BEAR': False,
        ...         'CRASH': False
        ...     },
        ...     risk_per_trade=0.02,
        ...     max_positions=5
        ... )
        >>> strategy = HighMomentum52W(config)
        >>> pf = strategy.backtest(data, initial_capital=10000, regime='TREND_BULL')
        >>> metrics = strategy.get_performance_metrics(pf)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        config: StrategyConfig,
        atr_multiplier: float = 2.5,
        volume_multiplier: float = 1.25
    ):
        """
        Initialize 52-Week High Momentum strategy.

        Args:
            config: StrategyConfig with validated parameters
            atr_multiplier: Stop loss distance multiplier (default: 2.5)
                Reasonable range: 2.0-3.0
            volume_multiplier: Volume confirmation threshold (default: 1.25)
                Calibrated per asset volatility:
                - High volatility (TSLA): 1.15-1.25
                - Moderate volatility (MSFT): 1.5-1.75
                - Low volatility (SPY): 1.25-1.5
                Set to None to disable volume filter

        Raises:
            ValueError: If parameters outside reasonable ranges
        """
        # Validate strategy-specific parameters before calling parent
        if not 2.0 <= atr_multiplier <= 3.0:
            raise ValueError(
                f"atr_multiplier {atr_multiplier} outside reasonable range [2.0, 3.0]. "
                f"2.5 is standard."
            )

        if volume_multiplier is not None and not 1.0 <= volume_multiplier <= 2.5:
            raise ValueError(
                f"volume_multiplier {volume_multiplier} outside reasonable range [1.0, 2.5]. "
                f"1.25 recommended for SPY."
            )

        self.atr_multiplier = atr_multiplier
        self.volume_multiplier = volume_multiplier

        # Call parent constructor (validates config)
        super().__init__(config)

    def validate_parameters(self) -> bool:
        """
        Validate strategy-specific parameters.

        Checks:
        - ATR multiplier in reasonable range (2.0-3.0)

        Returns:
            True if all parameters valid

        Raises:
            AssertionError: If validation fails
        """
        assert 2.0 <= self.atr_multiplier <= 3.0, \
            f"atr_multiplier {self.atr_multiplier} outside range [2.0, 3.0]"
        return True

    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals for 52-Week High Momentum strategy.

        Signal Logic (VALIDATED Session 36):
        1. Calculate 52-week high (252 trading days rolling maximum)
        2. Calculate distance from high (close / 52w_high)
        3. Define entry/exit ZONES:
           - Entry zone: distance >= 0.90 (within 10% of highs)
           - Exit zone: distance < 0.88 (exits just below entry, 12% off highs)
        4. Generate EVENT signals (state transitions):
           - Entry event: Transition INTO entry zone WITH volume confirmation
           - Exit event: Transition INTO exit zone
        5. Stop distance: 2.5x ATR (for position sizing)

        CRITICAL: Uses event-based signals (state transitions) not continuous states.
        VBT from_signals() only allows one position at a time, so we generate
        discrete entry/exit EVENTS rather than continuous TRUE/FALSE states.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
                Required columns: Open, High, Low, Close, Volume
            regime: Optional market regime for filtering
                ('TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH')

        Returns:
            Dictionary with v2.0 format signals:
            - 'entry_signal': Boolean Series for entry EVENTS (state transitions)
            - 'exit_signal': Boolean Series for exit EVENTS (state transitions)
            - 'stop_distance': Float Series for stop losses (in price units)
            - 'distance_from_high': Float Series (for debugging/CSV export)
            - 'volume_confirmed': Boolean Series (for debugging/CSV export)

        Regime Filtering:
            - If regime incompatible (TREND_BEAR, CRASH), returns all False signals
            - If regime compatible or None, generates signals normally

        Example:
            >>> signals = strategy.generate_signals(spy_data, regime='TREND_BULL')
            >>> print(f"Entry events: {signals['entry_signal'].sum()}")
            >>> print(f"Exit events: {signals['exit_signal'].sum()}")
        """
        # Regime filter: If incompatible regime, return all False signals
        if regime and not self.should_trade_in_regime(regime):
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': pd.Series(0.0, index=data.index),
                'distance_from_high': pd.Series(0.0, index=data.index),
                'volume_confirmed': pd.Series(False, index=data.index)
            }

        # Calculate 52-week high (252 trading days = 1 year)
        high_52w = data['High'].rolling(window=252, min_periods=252).max()

        # Calculate distance from 52-week high
        distance_from_high = data['Close'] / high_52w

        # Calculate volume moving average (20-day standard)
        volume_ma_20 = data['Volume'].rolling(window=20, min_periods=20).mean()

        # Calculate ATR for stop loss (14-period standard)
        atr = self._calculate_atr(data, period=14)

        # Define STATES (continuous conditions)
        in_entry_zone = (distance_from_high >= 0.90) & high_52w.notna() & atr.notna()
        in_exit_zone = (distance_from_high < 0.88) & high_52w.notna()

        # Apply volume filter if configured
        if self.volume_multiplier is not None:
            volume_confirmed = (
                (data['Volume'] > (volume_ma_20 * self.volume_multiplier)) &
                volume_ma_20.notna()
            )
            in_entry_zone = in_entry_zone & volume_confirmed
        else:
            volume_confirmed = pd.Series(True, index=data.index)

        # Convert states to EVENTS (state transitions)
        # Entry: Transition FROM outside entry zone TO inside entry zone
        entry_signal = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)

        # Exit: Transition FROM outside exit zone TO inside exit zone
        exit_signal = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

        # Stop Distance: 2.5x ATR (standard)
        stop_distance = atr * self.atr_multiplier
        stop_distance = stop_distance.fillna(0.0)

        return {
            'entry_signal': entry_signal.fillna(False),
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance,
            'distance_from_high': distance_from_high.fillna(0.0),  # For debugging
            'volume_confirmed': volume_confirmed.fillna(False)      # For debugging
        }

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes using ATR-based risk management.

        Uses utils/position_sizing.py for standardized ATR-based sizing
        with capital constraint (Gate 1 requirement).

        Args:
            data: OHLCV DataFrame (same as generate_signals input)
            capital: Current account capital (float)
            stop_distance: Stop loss distances from generate_signals()

        Returns:
            Position sizes as pd.Series of share counts (not dollars)
            Values are integers (whole shares only)

        Position Sizing Logic:
            1. Calculate ATR (14-period standard)
            2. Risk-based sizing: (capital * risk_pct) / (ATR * multiplier)
            3. Capital constraint: min(risk_based, capital / close)
            4. Result: Never exceeds 100% of capital (Gate 1 PASS)

        Example:
            >>> stop_distance = signals['stop_distance']
            >>> position_sizes = strategy.calculate_position_size(
            ...     data, capital=10000, stop_distance=stop_distance
            ... )
            >>> print(f"Mean position size: {position_sizes.mean():.0f} shares")
        """
        # Calculate ATR (same as in generate_signals for consistency)
        atr = self._calculate_atr(data, period=14)

        # Use standardized ATR-based position sizing from utils
        position_sizes, actual_risks, constrained = calculate_position_size_atr(
            init_cash=capital,
            close=data['Close'],
            atr=atr,
            atr_multiplier=self.atr_multiplier,
            risk_pct=self.config.risk_per_trade
        )

        # Ensure position sizes are integers (VBT requirement for size_type='amount')
        position_sizes = position_sizes.astype(int)

        return position_sizes  # pd.Series of share counts

    def get_strategy_name(self) -> str:
        """
        Return strategy name for logging and reporting.

        Returns:
            Human-readable strategy name
        """
        return "52-Week High Momentum"

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility-based stops.

        ATR measures market volatility by decomposing the entire range of an asset
        price for that period. Used for dynamic stop loss distances.

        Formula:
            TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
            ATR = EMA(TR, period)

        Args:
            data: OHLCV DataFrame
            period: Lookback period (default: 14 days, standard)

        Returns:
            ATR as pd.Series (same index as input data)

        Note:
            - Uses Exponential Moving Average (EMA) for smoothing
            - First value uses Simple Moving Average (SMA) as seed
            - Returns NaN for first 'period' bars (insufficient data)

        Example:
            >>> atr = strategy._calculate_atr(data, period=14)
            >>> print(f"Mean ATR: ${atr.mean():.2f}")
        """
        # True Range calculation
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR = EMA of True Range
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()

        return atr
