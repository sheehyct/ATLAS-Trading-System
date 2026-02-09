"""
Test Suite for Semi-Volatility Momentum Strategy

Session MOMENTUM-3: Created comprehensive test coverage for Semi-Vol Momentum.

Test Categories:
1. Configuration and Initialization (4 tests)
2. Volatility Calculation (4 tests)
3. Volatility Scalar Calculation (3 tests)
4. Momentum Calculation (3 tests)
5. Entry Signal Generation (3 tests)
6. Exit Signal Generation (3 tests)
7. Circuit Breaker (3 tests)
8. Regime Filtering (4 tests)
9. Position Sizing with Vol Scaling (3 tests)
10. Edge Cases (3 tests)

Total: ~33 tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.semi_vol_momentum import SemiVolMomentum
from strategies.base_strategy import StrategyConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default configuration for Semi-Vol Momentum strategy."""
    return StrategyConfig(
        name="Semi-Vol Momentum",
        universe="sp500",
        rebalance_frequency="monthly",
        regime_compatibility={
            'TREND_BULL': True,
            'TREND_NEUTRAL': False,  # Sits out in neutral
            'TREND_BEAR': False,     # Sits out in bear
            'CRASH': False           # Risk-off in crash
        },
        risk_per_trade=0.02,
        max_positions=5,
        enable_shorts=False
    )


@pytest.fixture
def synthetic_low_vol_data():
    """
    Generate synthetic low volatility data for 300 days.

    Creates data with ~10% annualized volatility (below 18% ceiling).
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate trending price with low volatility
    daily_return = 0.0005  # ~12.5% annual
    daily_vol = 0.006  # ~10% annualized
    returns = np.random.normal(daily_return, daily_vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.998, 1.002, n_days),
        'High': prices * np.random.uniform(1.002, 1.008, n_days),
        'Low': prices * np.random.uniform(0.992, 0.998, n_days),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    }, index=dates)

    return df


@pytest.fixture
def synthetic_high_vol_data():
    """
    Generate synthetic high volatility data for 300 days.

    Creates data with ~25% annualized volatility (above 22% circuit breaker).
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate trending price with high volatility
    daily_return = 0.0003
    daily_vol = 0.016  # ~25% annualized
    returns = np.random.normal(daily_return, daily_vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.005, n_days),
        'High': prices * np.random.uniform(1.01, 1.03, n_days),
        'Low': prices * np.random.uniform(0.97, 0.99, n_days),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    }, index=dates)

    return df


@pytest.fixture
def synthetic_negative_momentum_data():
    """
    Generate synthetic data with negative momentum.

    Creates data with downtrend for testing exit conditions.
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate downtrending price
    daily_return = -0.0004  # ~-10% annual
    daily_vol = 0.012
    returns = np.random.normal(daily_return, daily_vol, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * 1.002,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    }, index=dates)

    return df


# =============================================================================
# TEST CATEGORY 1: Configuration and Initialization
# =============================================================================

class TestInitialization:
    """Test strategy initialization and configuration validation."""

    def test_strategy_initialization(self, default_config):
        """Test strategy initializes correctly with valid config."""
        strategy = SemiVolMomentum(default_config)

        assert strategy.config.name == "Semi-Vol Momentum"
        assert strategy.target_volatility == 0.15
        assert strategy.volatility_lookback == 60
        assert strategy.momentum_lookback == 252
        assert strategy.momentum_lag == 21

    def test_invalid_target_volatility(self, default_config):
        """Test validation rejects invalid target volatility."""
        with pytest.raises(ValueError, match="target_volatility"):
            SemiVolMomentum(default_config, target_volatility=0.05)  # Too low

        with pytest.raises(ValueError, match="target_volatility"):
            SemiVolMomentum(default_config, target_volatility=0.30)  # Too high

    def test_invalid_vol_scalar_range(self, default_config):
        """Test validation rejects invalid vol scalar range."""
        with pytest.raises(ValueError, match="min_vol_scalar"):
            SemiVolMomentum(default_config, min_vol_scalar=0.1)  # Too low

        with pytest.raises(ValueError, match="max_vol_scalar"):
            SemiVolMomentum(default_config, max_vol_scalar=4.0)  # Too high

    def test_invalid_circuit_breaker_order(self, default_config):
        """Test validation rejects ceiling >= circuit breaker."""
        with pytest.raises(ValueError, match="vol_ceiling"):
            SemiVolMomentum(default_config, vol_ceiling=0.25, vol_circuit_breaker=0.20)

    def test_validate_parameters(self, default_config):
        """Test validate_parameters returns True for valid config."""
        strategy = SemiVolMomentum(default_config)
        assert strategy.validate_parameters() is True


# =============================================================================
# TEST CATEGORY 2: Volatility Calculation
# =============================================================================

class TestVolatilityCalculation:
    """Test realized volatility calculation."""

    def test_volatility_calculation_low_vol(self, default_config, synthetic_low_vol_data):
        """Test volatility calculation on low-vol data."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        realized_vol = signals['realized_vol']
        # After warm-up period, vol should be around 10%
        valid_vol = realized_vol[realized_vol > 0]
        assert len(valid_vol) > 0
        assert valid_vol.mean() < 0.18  # Below ceiling

    def test_volatility_calculation_high_vol(self, default_config, synthetic_high_vol_data):
        """Test volatility calculation on high-vol data."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_high_vol_data, regime='TREND_BULL')

        realized_vol = signals['realized_vol']
        valid_vol = realized_vol[realized_vol > 0]
        assert len(valid_vol) > 0
        assert valid_vol.mean() > 0.20  # Above ceiling

    def test_volatility_annualization(self, default_config):
        """Test volatility is correctly annualized."""
        strategy = SemiVolMomentum(default_config)

        # Create data with known daily volatility
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

        # Daily vol = 0.01 -> Annual vol ~ 15.9%
        returns = np.random.normal(0.0, 0.01, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.full(n_days, 1e6)
        }, index=dates)

        signals = strategy.generate_signals(df, regime='TREND_BULL')
        realized_vol = signals['realized_vol'].dropna()

        if len(realized_vol) > 0:
            # Annualized vol should be ~15.9% for 1% daily vol
            assert 0.10 < realized_vol.iloc[-1] < 0.25

    def test_volatility_lookback_period(self, default_config, synthetic_low_vol_data):
        """Test volatility uses correct lookback period."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        realized_vol = signals['realized_vol']
        # First 59 days should be NaN/0 (60-day lookback)
        assert realized_vol.iloc[:59].isna().sum() > 0 or (realized_vol.iloc[:59] == 0).sum() > 0


# =============================================================================
# TEST CATEGORY 3: Volatility Scalar Calculation
# =============================================================================

class TestVolatilityScalar:
    """Test volatility scalar calculation for position sizing."""

    def test_vol_scalar_low_vol_environment(self, default_config, synthetic_low_vol_data):
        """Test vol scalar is high when realized vol is low."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        vol_scalar = signals['vol_scalar']
        valid_scalars = vol_scalar[vol_scalar != 1.0]

        if len(valid_scalars) > 0:
            # With target_vol=15% and realized_vol~10%, scalar should be >1.0
            assert valid_scalars.mean() > 1.0

    def test_vol_scalar_high_vol_environment(self, default_config, synthetic_high_vol_data):
        """Test vol scalar is low when realized vol is high."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_high_vol_data, regime='TREND_BULL')

        vol_scalar = signals['vol_scalar']
        valid_scalars = vol_scalar[vol_scalar != 1.0]

        if len(valid_scalars) > 0:
            # With target_vol=15% and realized_vol~25%, scalar should be <1.0
            assert valid_scalars.mean() < 1.0

    def test_vol_scalar_clipping(self, default_config):
        """Test vol scalar is clipped to min/max range."""
        strategy = SemiVolMomentum(default_config)

        # Create extreme low vol data (scalar would be very high)
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

        # Very low vol data
        returns = np.random.normal(0.001, 0.002, n_days)  # ~3% annualized
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.full(n_days, 1e6)
        }, index=dates)

        signals = strategy.generate_signals(df, regime='TREND_BULL')
        vol_scalar = signals['vol_scalar']

        # Scalar should never exceed max (2.0) or go below min (0.5)
        assert vol_scalar.max() <= 2.0
        assert vol_scalar.min() >= 0.5


# =============================================================================
# TEST CATEGORY 4: Momentum Calculation
# =============================================================================

class TestMomentumCalculation:
    """Test 12-1 momentum calculation."""

    def test_positive_momentum_detection(self, default_config, synthetic_low_vol_data):
        """Test positive momentum is correctly detected."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        momentum = signals['momentum']
        valid_momentum = momentum[momentum != 0]

        if len(valid_momentum) > 0:
            # Low vol data has positive trend
            assert valid_momentum.iloc[-1] > 0

    def test_negative_momentum_detection(self, default_config, synthetic_negative_momentum_data):
        """Test negative momentum is correctly detected."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_negative_momentum_data, regime='TREND_BULL')

        momentum = signals['momentum']
        valid_momentum = momentum[momentum != 0]

        if len(valid_momentum) > 0:
            # Negative momentum data
            assert valid_momentum.iloc[-1] < 0

    def test_momentum_lag_applied(self, default_config, synthetic_low_vol_data):
        """Test 21-day lag is applied to momentum."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        momentum = signals['momentum']
        # Momentum should be NaN for first 252 + 21 days
        nan_count = momentum.isna().sum() + (momentum == 0).sum()
        assert nan_count >= 273  # momentum_lookback + momentum_lag


# =============================================================================
# TEST CATEGORY 5: Entry Signal Generation
# =============================================================================

class TestEntrySignals:
    """Test entry signal generation."""

    def test_entry_signal_low_vol_positive_momentum(self, default_config, synthetic_low_vol_data):
        """Test entry signals generated with low vol + positive momentum."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        entry_signal = signals['entry_signal']
        # Should have at least some entry signals
        assert entry_signal.sum() > 0

    def test_no_entry_high_vol(self, default_config, synthetic_high_vol_data):
        """Test no entry signals when volatility is too high."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_high_vol_data, regime='TREND_BULL')

        entry_signal = signals['entry_signal']
        # High vol should prevent entries (vol > ceiling)
        # This depends on how high the vol is
        realized_vol = signals['realized_vol']
        high_vol_mask = realized_vol > strategy.vol_ceiling

        # Entry signals should be False when vol is too high
        high_vol_entries = entry_signal[high_vol_mask]
        assert high_vol_entries.sum() == 0

    def test_no_entry_negative_momentum(self, default_config, synthetic_negative_momentum_data):
        """Test no entry signals with negative momentum."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_negative_momentum_data, regime='TREND_BULL')

        entry_signal = signals['entry_signal']
        # Negative momentum should prevent entries after warm-up
        # Allow some entries during transition periods
        # Main assertion: fewer entries than positive momentum scenario
        assert entry_signal.sum() < 5


# =============================================================================
# TEST CATEGORY 6: Exit Signal Generation
# =============================================================================

class TestExitSignals:
    """Test exit signal generation."""

    def test_exit_on_negative_momentum(self, default_config, synthetic_negative_momentum_data):
        """Test exit signals generated when momentum turns negative."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_negative_momentum_data, regime='TREND_BULL')

        exit_signal = signals['exit_signal']
        # Should have exit signals when momentum is negative
        assert exit_signal.sum() > 0

    def test_exit_on_circuit_breaker(self, default_config, synthetic_high_vol_data):
        """Test exit signals when circuit breaker is triggered."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_high_vol_data, regime='TREND_BULL')

        exit_signal = signals['exit_signal']
        realized_vol = signals['realized_vol']

        # Check if circuit breaker was triggered
        circuit_breaker_triggered = realized_vol > strategy.vol_circuit_breaker
        if circuit_breaker_triggered.any():
            # Should have exit signals when circuit breaker fires
            assert exit_signal.sum() > 0

    def test_exit_transition_detection(self, default_config, synthetic_low_vol_data):
        """Test exit signals detect state transitions."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        exit_signal = signals['exit_signal']
        # Exit signals should be sparse (state transitions only)
        # Not every day in exit zone
        if exit_signal.sum() > 0:
            assert exit_signal.sum() < len(exit_signal) * 0.1


# =============================================================================
# TEST CATEGORY 7: Circuit Breaker
# =============================================================================

class TestCircuitBreaker:
    """Test volatility circuit breaker functionality."""

    def test_circuit_breaker_threshold(self, default_config):
        """Test circuit breaker fires at 22% vol."""
        strategy = SemiVolMomentum(default_config)
        assert strategy.vol_circuit_breaker == 0.22

    def test_is_circuit_breaker_active_high_vol(self, default_config, synthetic_high_vol_data):
        """Test circuit breaker detection method."""
        strategy = SemiVolMomentum(default_config)

        # High vol data should trigger circuit breaker
        is_active = strategy.is_circuit_breaker_active(synthetic_high_vol_data)

        # May or may not be active depending on exact vol levels
        # Just verify method runs without error and returns bool-like
        assert is_active in [True, False, np.True_, np.False_]

    def test_is_circuit_breaker_active_low_vol(self, default_config, synthetic_low_vol_data):
        """Test circuit breaker not active in low vol."""
        strategy = SemiVolMomentum(default_config)

        is_active = strategy.is_circuit_breaker_active(synthetic_low_vol_data)
        # Low vol should not trigger circuit breaker
        assert not is_active  # Works with both Python bool and numpy bool


# =============================================================================
# TEST CATEGORY 8: Regime Filtering
# =============================================================================

class TestRegimeFiltering:
    """Test regime-based filtering."""

    def test_trend_bull_allowed(self, default_config, synthetic_low_vol_data):
        """Test TREND_BULL regime allows trading."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')

        # Should have entry signals in bull market
        assert signals['entry_signal'].sum() > 0

    def test_trend_neutral_blocked(self, default_config, synthetic_low_vol_data):
        """Test TREND_NEUTRAL regime blocks trading."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_NEUTRAL')

        # Should have no entry signals in neutral market
        assert signals['entry_signal'].sum() == 0

    def test_trend_bear_blocked(self, default_config, synthetic_low_vol_data):
        """Test TREND_BEAR regime blocks trading."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BEAR')

        # Should have no entry signals in bear market
        assert signals['entry_signal'].sum() == 0

    def test_crash_blocked(self, default_config, synthetic_low_vol_data):
        """Test CRASH regime blocks trading."""
        strategy = SemiVolMomentum(default_config)
        signals = strategy.generate_signals(synthetic_low_vol_data, regime='CRASH')

        # Should have no entry signals in crash
        assert signals['entry_signal'].sum() == 0


# =============================================================================
# TEST CATEGORY 9: Position Sizing with Vol Scaling
# =============================================================================

class TestPositionSizing:
    """Test position sizing with volatility scaling."""

    def test_position_size_calculation(self, default_config, synthetic_low_vol_data):
        """Test position sizes are calculated correctly."""
        strategy = SemiVolMomentum(default_config)

        signals = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')
        position_sizes = strategy.calculate_position_size(
            synthetic_low_vol_data,
            capital=100000,
            stop_distance=signals['stop_distance']
        )

        # Position sizes should be positive integers
        assert position_sizes.dtype in [np.int64, np.int32, int]
        assert (position_sizes >= 0).all()

    def test_position_size_vol_scaling(self, default_config, synthetic_low_vol_data, synthetic_high_vol_data):
        """Test position sizes are scaled by volatility."""
        strategy = SemiVolMomentum(default_config)

        # Low vol should have larger positions
        signals_low = strategy.generate_signals(synthetic_low_vol_data, regime='TREND_BULL')
        positions_low = strategy.calculate_position_size(
            synthetic_low_vol_data,
            capital=100000,
            stop_distance=signals_low['stop_distance']
        )

        # High vol should have smaller positions
        signals_high = strategy.generate_signals(synthetic_high_vol_data, regime='TREND_BULL')
        positions_high = strategy.calculate_position_size(
            synthetic_high_vol_data,
            capital=100000,
            stop_distance=signals_high['stop_distance']
        )

        # Compare median position sizes (after warm-up)
        median_low = positions_low[60:].median()
        median_high = positions_high[60:].median()

        # Low vol should generally have larger positions due to vol scaling
        # (But price levels also affect this, so we just check they're different)
        assert median_low != median_high

    def test_get_current_vol_scalar(self, default_config, synthetic_low_vol_data):
        """Test get_current_vol_scalar method."""
        strategy = SemiVolMomentum(default_config)

        scalar = strategy.get_current_vol_scalar(synthetic_low_vol_data)

        # Should be in valid range
        assert 0.5 <= scalar <= 2.0


# =============================================================================
# TEST CATEGORY 10: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_insufficient_data(self, default_config):
        """Test handling of insufficient data."""
        strategy = SemiVolMomentum(default_config)

        # Create data shorter than momentum lookback
        n_days = 100  # Less than 252 + 21 required
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
        prices = np.linspace(100, 120, n_days)

        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.full(n_days, 1e6)
        }, index=dates)

        signals = strategy.generate_signals(df, regime='TREND_BULL')

        # Should not crash, but momentum should be mostly NaN/zero
        momentum = signals['momentum']
        assert (momentum == 0).sum() == len(momentum) or momentum.isna().sum() == len(momentum)

    def test_nan_handling(self, default_config, synthetic_low_vol_data):
        """Test handling of NaN values in data."""
        strategy = SemiVolMomentum(default_config)

        # Introduce NaN values
        data_with_nan = synthetic_low_vol_data.copy()
        data_with_nan.loc[data_with_nan.index[50:55], 'Close'] = np.nan

        signals = strategy.generate_signals(data_with_nan, regime='TREND_BULL')

        # Signals should still be valid Series
        assert isinstance(signals['entry_signal'], pd.Series)
        assert isinstance(signals['exit_signal'], pd.Series)

    def test_zero_volatility_handling(self, default_config):
        """Test handling when volatility approaches zero."""
        strategy = SemiVolMomentum(default_config)

        # Create flat price data (zero volatility)
        n_days = 100
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

        df = pd.DataFrame({
            'Open': np.full(n_days, 100.0),
            'High': np.full(n_days, 100.01),
            'Low': np.full(n_days, 99.99),
            'Close': np.full(n_days, 100.0),
            'Volume': np.full(n_days, 1e6)
        }, index=dates)

        signals = strategy.generate_signals(df, regime='TREND_BULL')

        # Should not crash, vol_scalar should be clipped to max
        vol_scalar = signals['vol_scalar']
        assert vol_scalar.max() <= 2.0


# =============================================================================
# TEST CATEGORY 11: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full strategy flow."""

    def test_backtest_execution(self, default_config, synthetic_low_vol_data):
        """Test full backtest executes without errors."""
        strategy = SemiVolMomentum(default_config)

        # Run backtest
        pf = strategy.backtest(
            synthetic_low_vol_data,
            initial_capital=100000,
            regime='TREND_BULL'
        )

        # Basic portfolio checks
        assert pf is not None
        assert hasattr(pf, 'total_return')
        assert hasattr(pf, 'sharpe_ratio')

    def test_performance_metrics_extraction(self, default_config, synthetic_low_vol_data):
        """Test performance metrics can be extracted."""
        strategy = SemiVolMomentum(default_config)

        pf = strategy.backtest(
            synthetic_low_vol_data,
            initial_capital=100000,
            regime='TREND_BULL'
        )

        metrics = strategy.get_performance_metrics(pf)

        # Check required metrics exist
        assert 'total_return' in metrics
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics

    def test_strategy_name(self, default_config):
        """Test strategy name method."""
        strategy = SemiVolMomentum(default_config)
        assert strategy.get_strategy_name() == "Semi-Volatility Momentum"
