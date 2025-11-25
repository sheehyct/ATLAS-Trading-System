"""
Semi-Volatility Momentum Strategy for ATLAS Trading System v2.0

This module implements the Semi-Volatility Momentum strategy, which scales
momentum positions based on realized volatility to improve risk-adjusted returns.

Academic Foundation:
- Moreira & Muir (2017): "Volatility-Managed Portfolios"
- Documented Sharpe improvement from 0.8 (base momentum) to 1.6-1.7 (vol-managed)
- Key insight: Scale position sizes inversely with volatility

Strategy Logic (from architecture spec):
- Base Signal: Standard 12-1 momentum (12-month return, 1-month lag)
- Volatility Scaling: Position size = base_size * (target_vol / realized_vol)
- Scaling Limits: Clipped to 0.5x - 2.0x of base position
- Target Volatility: 15% annualized (configurable)
- Regime Filter: Only trades in TREND_BULL with low/moderate volatility

Performance Targets (per architecture):
- Sharpe Ratio: 1.4-1.8 (significant improvement from base momentum)
- Turnover: ~100% annually (monthly rebalance)
- Win Rate: 50-60%
- CAGR: 15-20%
- Max Drawdown: -15% to -20%

Regime Compatibility:
- TREND_BULL + Low Vol: 15-20% allocation (ideal conditions)
- TREND_BULL + High Vol: 5-10% allocation (reduced due to scaling)
- TREND_NEUTRAL: 0% (sit out)
- TREND_BEAR: 0% (sit out)
- CRASH: 0% (risk-off)

Circuit Breakers:
- Exit all positions if portfolio volatility > 22%
- Only trade when market vol < 18%

Implementation Status: SKELETON - Ready for implementation
Implementation Priority: PHASE 2 (after foundation strategies proven)

Reference:
- docs/SYSTEM_ARCHITECTURE/1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md (lines 270-328)
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy, StrategyConfig
from utils.position_sizing import calculate_position_size_atr


class SemiVolMomentum(BaseStrategy):
    """
    Semi-Volatility Momentum Strategy Implementation.

    Scales momentum positions based on realized volatility to improve
    risk-adjusted returns. Based on academic research showing Sharpe
    improvement from 0.8 to 1.6-1.7.

    Core Mechanism:
    - When volatility is LOW: Increase position size (up to 2.0x)
    - When volatility is HIGH: Decrease position size (down to 0.5x)
    - Result: Smoother equity curve, better risk-adjusted returns

    Entry Conditions:
    1. Positive 12-1 momentum (12-month return > 0, with 1-month lag)
    2. Realized volatility < 18% (market stability check)
    3. TREND_BULL regime only

    Exit Conditions:
    1. Negative momentum (12-month return < 0)
    2. Volatility spike > 22% (circuit breaker)
    3. Regime change (exit on TREND_NEUTRAL, TREND_BEAR, CRASH)

    Position Sizing:
    - Base size from ATR calculation
    - Scaled by (target_vol / realized_vol)
    - Clipped to 0.5x - 2.0x range

    Example Usage:
        >>> config = StrategyConfig(
        ...     name="Semi-Volatility Momentum",
        ...     universe="sp500",
        ...     rebalance_frequency="monthly",
        ...     regime_compatibility={
        ...         'TREND_BULL': True,
        ...         'TREND_NEUTRAL': False,
        ...         'TREND_BEAR': False,
        ...         'CRASH': False
        ...     },
        ...     risk_per_trade=0.02,
        ...     max_positions=5
        ... )
        >>> strategy = SemiVolMomentum(config)
        >>> pf = strategy.backtest(data, initial_capital=10000, regime='TREND_BULL')
    """

    def __init__(
        self,
        config: StrategyConfig,
        target_volatility: float = 0.15,
        volatility_lookback: int = 60,
        momentum_lookback: int = 252,
        momentum_lag: int = 21,
        min_vol_scalar: float = 0.5,
        max_vol_scalar: float = 2.0,
        vol_ceiling: float = 0.18,
        vol_circuit_breaker: float = 0.22,
        atr_multiplier: float = 2.5
    ):
        """
        Initialize Semi-Volatility Momentum strategy.

        Args:
            config: StrategyConfig with validated parameters
            target_volatility: Target portfolio volatility (default: 15% annualized)
            volatility_lookback: Days for volatility calculation (default: 60)
            momentum_lookback: Days for momentum calculation (default: 252)
            momentum_lag: Days to skip recent performance (default: 21)
            min_vol_scalar: Minimum position scaling factor (default: 0.5)
            max_vol_scalar: Maximum position scaling factor (default: 2.0)
            vol_ceiling: Maximum volatility for entry (default: 18%)
            vol_circuit_breaker: Volatility level triggering full exit (default: 22%)
            atr_multiplier: Stop loss distance multiplier (default: 2.5)

        Raises:
            ValueError: If parameters outside reasonable ranges
        """
        # Validate strategy-specific parameters
        if not 0.10 <= target_volatility <= 0.25:
            raise ValueError(
                f"target_volatility {target_volatility} outside range [0.10, 0.25]. "
                f"0.15 (15%) is standard."
            )

        if not 0.25 <= min_vol_scalar <= 1.0:
            raise ValueError(
                f"min_vol_scalar {min_vol_scalar} outside range [0.25, 1.0]"
            )

        if not 1.0 <= max_vol_scalar <= 3.0:
            raise ValueError(
                f"max_vol_scalar {max_vol_scalar} outside range [1.0, 3.0]"
            )

        if vol_ceiling >= vol_circuit_breaker:
            raise ValueError(
                f"vol_ceiling {vol_ceiling} must be < vol_circuit_breaker {vol_circuit_breaker}"
            )

        self.target_volatility = target_volatility
        self.volatility_lookback = volatility_lookback
        self.momentum_lookback = momentum_lookback
        self.momentum_lag = momentum_lag
        self.min_vol_scalar = min_vol_scalar
        self.max_vol_scalar = max_vol_scalar
        self.vol_ceiling = vol_ceiling
        self.vol_circuit_breaker = vol_circuit_breaker
        self.atr_multiplier = atr_multiplier

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
        assert 0.10 <= self.target_volatility <= 0.25, \
            f"target_volatility {self.target_volatility} outside range [0.10, 0.25]"
        assert self.min_vol_scalar < self.max_vol_scalar, \
            f"min_vol_scalar must be < max_vol_scalar"
        assert self.vol_ceiling < self.vol_circuit_breaker, \
            f"vol_ceiling must be < vol_circuit_breaker"
        return True

    def generate_signals(
        self,
        data: pd.DataFrame,
        regime: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate entry/exit signals for Semi-Volatility Momentum strategy.

        Signal Logic:
        1. Calculate realized volatility (60-day annualized)
        2. Calculate 12-1 momentum
        3. Check volatility conditions (< ceiling for entry)
        4. Check momentum condition (positive for entry)
        5. Apply regime filter (TREND_BULL only)
        6. Generate volatility scaling factor for position sizing

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
            - 'vol_scalar': Float Series for position scaling (0.5-2.0)
            - 'realized_vol': Float Series (for debugging)
            - 'momentum': Float Series (for debugging)

        Circuit Breaker:
            If realized_vol > vol_circuit_breaker (22%), generates exit signal
            regardless of other conditions.
        """
        # Regime filter: Only trade in TREND_BULL
        if regime and not self.should_trade_in_regime(regime):
            return {
                'entry_signal': pd.Series(False, index=data.index),
                'exit_signal': pd.Series(False, index=data.index),
                'stop_distance': pd.Series(0.0, index=data.index),
                'vol_scalar': pd.Series(1.0, index=data.index),
                'realized_vol': pd.Series(0.0, index=data.index),
                'momentum': pd.Series(0.0, index=data.index)
            }

        # Calculate returns
        returns = data['Close'].pct_change()

        # Calculate realized volatility (annualized)
        realized_vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)

        # Calculate volatility scalar
        vol_scalar = self.target_volatility / realized_vol
        vol_scalar = vol_scalar.clip(self.min_vol_scalar, self.max_vol_scalar)

        # Calculate 12-1 momentum
        if len(data) > self.momentum_lookback + self.momentum_lag:
            momentum = data['Close'].pct_change(self.momentum_lookback).shift(self.momentum_lag)
        else:
            momentum = pd.Series(0.0, index=data.index)

        # Calculate ATR for stop loss
        atr = self._calculate_atr(data, period=14)
        stop_distance = atr * self.atr_multiplier

        # ============================================================
        # ENTRY CONDITIONS
        # ============================================================
        # 1. Positive momentum (12-month return > 0)
        # 2. Volatility below ceiling (< 18%)
        # 3. Sufficient data
        in_entry_zone = (
            (momentum > 0) &
            (realized_vol < self.vol_ceiling) &
            realized_vol.notna() &
            momentum.notna()
        )

        # ============================================================
        # EXIT CONDITIONS
        # ============================================================
        # 1. Negative momentum (12-month return < 0)
        # 2. OR Volatility circuit breaker triggered (> 22%)
        circuit_breaker_triggered = realized_vol > self.vol_circuit_breaker

        in_exit_zone = (
            (momentum < 0) |
            circuit_breaker_triggered
        ) & realized_vol.notna()

        # Convert states to events (state transitions)
        entry_signal = in_entry_zone & ~in_entry_zone.shift(1).fillna(False)
        exit_signal = in_exit_zone & ~in_exit_zone.shift(1).fillna(False)

        # Also exit on circuit breaker even if already in exit zone
        # This ensures we exit immediately on volatility spike
        circuit_breaker_exit = circuit_breaker_triggered & ~circuit_breaker_triggered.shift(1).fillna(False)
        exit_signal = exit_signal | circuit_breaker_exit

        return {
            'entry_signal': entry_signal.fillna(False),
            'exit_signal': exit_signal.fillna(False),
            'stop_distance': stop_distance.fillna(0.0),
            'vol_scalar': vol_scalar.fillna(1.0),
            'realized_vol': realized_vol.fillna(0.0),
            'momentum': momentum.fillna(0.0)
        }

    def calculate_position_size(
        self,
        data: pd.DataFrame,
        capital: float,
        stop_distance: pd.Series
    ) -> pd.Series:
        """
        Calculate position sizes with volatility scaling.

        This is the CORE mechanism of the strategy:
        - Base position size from ATR calculation
        - Scaled by vol_scalar (target_vol / realized_vol)
        - Result: Larger positions in low vol, smaller in high vol

        Args:
            data: OHLCV DataFrame
            capital: Current account capital
            stop_distance: Stop loss distances from generate_signals()

        Returns:
            Position sizes as pd.Series of share counts
            Sizes are adjusted by volatility scalar
        """
        atr = self._calculate_atr(data, period=14)

        # Calculate base position sizes
        base_positions, actual_risks, constrained = calculate_position_size_atr(
            init_cash=capital,
            close=data['Close'],
            atr=atr,
            atr_multiplier=self.atr_multiplier,
            risk_pct=self.config.risk_per_trade
        )

        # Calculate volatility scalar
        returns = data['Close'].pct_change()
        realized_vol = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        vol_scalar = self.target_volatility / realized_vol
        vol_scalar = vol_scalar.clip(self.min_vol_scalar, self.max_vol_scalar)

        # Apply volatility scaling
        # CRITICAL: This is the key innovation from Moreira & Muir
        position_sizes = base_positions * vol_scalar

        # Capital constraint: Never exceed 100% of capital (even with 2x scaling)
        max_shares = capital / data['Close']
        position_sizes = position_sizes.clip(upper=max_shares)

        return position_sizes.fillna(0).astype(int)

    def get_strategy_name(self) -> str:
        """Return strategy name for logging and reporting."""
        return "Semi-Volatility Momentum"

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
    # ADDITIONAL METHODS FOR PRODUCTION USE
    # ================================================================

    def get_current_vol_scalar(self, data: pd.DataFrame) -> float:
        """
        Get current volatility scalar for position sizing decisions.

        Useful for real-time monitoring and position adjustment.

        Args:
            data: Recent OHLCV data (at least volatility_lookback days)

        Returns:
            Current volatility scalar (0.5 to 2.0)
        """
        returns = data['Close'].pct_change()
        current_vol = returns.iloc[-self.volatility_lookback:].std() * np.sqrt(252)

        if pd.isna(current_vol) or current_vol == 0:
            return 1.0

        scalar = self.target_volatility / current_vol
        return np.clip(scalar, self.min_vol_scalar, self.max_vol_scalar)

    def is_circuit_breaker_active(self, data: pd.DataFrame) -> bool:
        """
        Check if volatility circuit breaker is currently active.

        Args:
            data: Recent OHLCV data

        Returns:
            True if current volatility > circuit breaker threshold
        """
        returns = data['Close'].pct_change()
        current_vol = returns.iloc[-self.volatility_lookback:].std() * np.sqrt(252)
        return current_vol > self.vol_circuit_breaker
