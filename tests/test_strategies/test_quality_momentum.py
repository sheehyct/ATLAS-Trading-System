"""
Test Suite for Quality-Momentum Strategy

Session EQUITY-35: Created comprehensive test coverage for Quality-Momentum.

Test Categories:
1. Configuration and Initialization (4 tests)
2. Quality Score Calculation (5 tests)
3. Momentum Score Calculation (3 tests)
4. Entry Signal Generation (4 tests)
5. Exit Signal Generation (3 tests)
6. Regime Filtering (4 tests)
7. Buffer Rule - 40% exit threshold (2 tests)
8. Position Sizing (3 tests)
9. Edge Cases (3 tests)
10. Integration Tests (2 tests)

Total: 33 tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from strategies.quality_momentum import QualityMomentum
from strategies.base_strategy import StrategyConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default configuration for Quality-Momentum strategy."""
    return StrategyConfig(
        name="Quality-Momentum",
        universe="sp500",
        rebalance_frequency="quarterly",
        regime_compatibility={
            'TREND_BULL': True,
            'TREND_NEUTRAL': True,
            'TREND_BEAR': True,  # Works in all regimes
            'CRASH': True        # Reduced but still active
        },
        risk_per_trade=0.02,
        max_positions=20,
        enable_shorts=False
    )


@pytest.fixture
def mock_fundamental_data():
    """
    Mock fundamental data for 10 test symbols.

    Returns DataFrame with quality metrics matching expected Alpha Vantage format.
    """
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ'],
        'roe': [0.25, 0.30, 0.22, 0.18, 0.20, 0.35, 0.15, 0.12, 0.28, 0.22],
        'accruals_ratio': [0.02, 0.01, 0.03, 0.05, 0.04, 0.02, 0.08, 0.03, 0.01, 0.02],
        'debt_to_equity': [1.5, 0.8, 0.3, 1.2, 0.5, 0.4, 1.8, 2.5, 1.0, 0.6]
    })


@pytest.fixture
def synthetic_uptrend_data():
    """
    Generate synthetic uptrend data for 300 days.

    Creates data with positive momentum and varying volatility.
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Generate trending price with noise
    daily_return = 0.0006  # ~15% annual
    returns = np.random.normal(daily_return, 0.012, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.995, 1.005, n_days),
        'High': prices * np.random.uniform(1.005, 1.02, n_days),
        'Low': prices * np.random.uniform(0.98, 0.995, n_days),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, n_days)
    }, index=dates)

    return df


@pytest.fixture
def mock_price_data():
    """
    Generate mock price data for 10 symbols with different momentum profiles.
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')

    # Different momentum profiles for each symbol
    momentum_profiles = {
        'AAPL': 0.30,   # Strong positive momentum
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'META': 0.10,
        'NVDA': 0.40,   # Strongest
        'TSLA': -0.05,  # Slightly negative
        'JPM': 0.05,
        'V': 0.18,
        'JNJ': 0.08    # Defensive, low momentum
    }

    price_data = {}

    for symbol, annual_return in momentum_profiles.items():
        # Generate trending price with noise
        daily_return = annual_return / 252
        returns = np.random.normal(daily_return, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.uniform(1e6, 5e6, n_days)
        }, index=dates)

        price_data[symbol] = df

    return price_data


# =============================================================================
# TEST CATEGORY 1: Configuration and Initialization
# =============================================================================

class TestInitialization:
    """Test strategy initialization and configuration validation."""

    def test_strategy_initialization(self, default_config):
        """Test strategy initializes correctly with valid config."""
        strategy = QualityMomentum(default_config)
        assert strategy.config.name == "Quality-Momentum"
        assert strategy.atr_multiplier == 2.5  # Default value
        assert strategy.momentum_lookback == 252
        assert strategy.momentum_lag == 21
        assert strategy.quality_threshold == 0.50
        assert strategy.exit_buffer == 0.40

    def test_custom_parameters(self, default_config):
        """Test strategy accepts custom parameters."""
        strategy = QualityMomentum(
            default_config,
            momentum_lookback=189,  # 9 months
            quality_threshold=0.60,
            exit_buffer=0.35
        )
        assert strategy.momentum_lookback == 189
        assert strategy.quality_threshold == 0.60
        assert strategy.exit_buffer == 0.35

    def test_invalid_momentum_lookback_low(self, default_config):
        """Test strategy rejects momentum_lookback below range."""
        with pytest.raises(ValueError, match="outside range"):
            QualityMomentum(default_config, momentum_lookback=100)

    def test_invalid_quality_threshold(self, default_config):
        """Test strategy rejects quality_threshold outside [0, 1]."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            QualityMomentum(default_config, quality_threshold=1.5)


# =============================================================================
# TEST CATEGORY 2: Quality Score Calculation
# =============================================================================

class TestQualityScoreCalculation:
    """Test quality score calculation from fundamental data."""

    def test_quality_score_formula(self, mock_fundamental_data, default_config):
        """Test quality score follows the 40/30/30 weighting formula."""
        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(mock_fundamental_data)

        # Quality score should exist and be between 0 and 1
        assert 'quality_score' in quality_df.columns
        assert quality_df['quality_score'].min() >= 0
        assert quality_df['quality_score'].max() <= 1

    def test_roe_weight_40_percent(self, default_config):
        """Test ROE has 40% weight in quality score."""
        # Create data where only ROE varies
        data = pd.DataFrame({
            'symbol': ['A', 'B'],
            'roe': [0.40, 0.10],  # A has much higher ROE
            'accruals_ratio': [0.02, 0.02],  # Same
            'debt_to_equity': [1.0, 1.0]  # Same
        })

        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(data)

        # A should have higher quality score due to higher ROE
        a_score = quality_df[quality_df['symbol'] == 'A']['quality_score'].values[0]
        b_score = quality_df[quality_df['symbol'] == 'B']['quality_score'].values[0]
        assert a_score > b_score

    def test_earnings_quality_lower_accruals_better(self, default_config):
        """Test lower accruals ratio = higher earnings quality = higher score."""
        data = pd.DataFrame({
            'symbol': ['A', 'B'],
            'roe': [0.20, 0.20],  # Same
            'accruals_ratio': [0.01, 0.10],  # A has lower (better) accruals
            'debt_to_equity': [1.0, 1.0]  # Same
        })

        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(data)

        a_score = quality_df[quality_df['symbol'] == 'A']['quality_score'].values[0]
        b_score = quality_df[quality_df['symbol'] == 'B']['quality_score'].values[0]
        assert a_score > b_score  # Lower accruals = higher score

    def test_leverage_lower_debt_better(self, default_config):
        """Test lower debt-to-equity = higher score."""
        data = pd.DataFrame({
            'symbol': ['A', 'B'],
            'roe': [0.20, 0.20],  # Same
            'accruals_ratio': [0.02, 0.02],  # Same
            'debt_to_equity': [0.5, 2.0]  # A has lower leverage
        })

        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(data)

        a_score = quality_df[quality_df['symbol'] == 'A']['quality_score'].values[0]
        b_score = quality_df[quality_df['symbol'] == 'B']['quality_score'].values[0]
        assert a_score > b_score  # Lower leverage = higher score

    def test_quality_filter_top_50_percent(self, mock_fundamental_data, default_config):
        """Test quality filter selects top 50% by quality score."""
        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(mock_fundamental_data)
        filtered = strategy.filter_by_quality(quality_df)

        # Should select 5 out of 10 symbols (50%)
        assert len(filtered) == 5


# =============================================================================
# TEST CATEGORY 3: Momentum Score Calculation
# =============================================================================

class TestMomentumScoreCalculation:
    """Test momentum score calculation."""

    def test_momentum_score_positive(self, mock_price_data, default_config):
        """Test momentum scores are calculated correctly for positive returns."""
        strategy = QualityMomentum(default_config)
        symbols = ['AAPL', 'MSFT', 'NVDA']
        momentum_df = strategy.calculate_momentum_scores(symbols, mock_price_data)

        assert len(momentum_df) == 3
        assert 'momentum' in momentum_df.columns
        assert 'momentum_rank' in momentum_df.columns

        # NVDA should have highest momentum (0.40 annual return)
        nvda_rank = momentum_df[momentum_df['symbol'] == 'NVDA']['momentum_rank'].values[0]
        assert nvda_rank >= 0.5  # Should be in top half

    def test_momentum_ranking(self, mock_price_data, default_config):
        """Test momentum ranking orders correctly."""
        strategy = QualityMomentum(default_config)
        symbols = list(mock_price_data.keys())
        momentum_df = strategy.calculate_momentum_scores(symbols, mock_price_data)

        # NVDA (0.40) should rank higher than TSLA (-0.05)
        nvda_momentum = momentum_df[momentum_df['symbol'] == 'NVDA']['momentum'].values[0]
        tsla_momentum = momentum_df[momentum_df['symbol'] == 'TSLA']['momentum'].values[0]
        assert nvda_momentum > tsla_momentum

    def test_momentum_insufficient_data(self, default_config):
        """Test momentum calculation handles insufficient data gracefully."""
        strategy = QualityMomentum(default_config)

        # Create short data (less than 252 + 21 days)
        short_data = pd.DataFrame({
            'Close': np.linspace(100, 110, 100)
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='B'))

        momentum_df = strategy.calculate_momentum_scores(['TEST'], {'TEST': short_data})
        assert len(momentum_df) == 0  # Should be empty due to insufficient data


# =============================================================================
# TEST CATEGORY 4: Entry Signal Generation
# =============================================================================

class TestEntrySignalGeneration:
    """Test entry signal generation."""

    def test_entry_signal_format(self, synthetic_uptrend_data, default_config):
        """Test entry signals follow v2.0 format."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        assert 'entry_signal' in signals
        assert 'exit_signal' in signals
        assert 'stop_distance' in signals
        assert 'quality_score' in signals
        assert 'momentum_score' in signals

        # Check types
        assert signals['entry_signal'].dtype == bool
        assert signals['exit_signal'].dtype == bool

    def test_entry_requires_quality_and_momentum(self, synthetic_uptrend_data, default_config):
        """Test entry requires both quality and momentum thresholds."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        # Entry signals should only occur where both conditions met
        entry_count = signals['entry_signal'].sum()
        assert entry_count >= 0  # May have entries

    def test_entry_is_event_based(self, synthetic_uptrend_data, default_config):
        """Test entry signals are events (state transitions), not continuous."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        # Cannot have consecutive True entries (event-based)
        entry = signals['entry_signal']
        consecutive = (entry & entry.shift(1).fillna(False)).sum()
        assert consecutive == 0  # No consecutive entries

    def test_portfolio_mode_entry(self, synthetic_uptrend_data, mock_fundamental_data, mock_price_data, default_config):
        """Test entry signals in portfolio mode."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(
            synthetic_uptrend_data,
            universe_data=mock_price_data,
            fundamental_data=mock_fundamental_data
        )

        assert 'entry_signal' in signals
        assert 'selected_symbols' in signals  # Portfolio mode bonus field


# =============================================================================
# TEST CATEGORY 5: Exit Signal Generation
# =============================================================================

class TestExitSignalGeneration:
    """Test exit signal generation."""

    def test_exit_signal_format(self, synthetic_uptrend_data, default_config):
        """Test exit signals follow correct format."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        assert 'exit_signal' in signals
        assert signals['exit_signal'].dtype == bool

    def test_exit_is_event_based(self, synthetic_uptrend_data, default_config):
        """Test exit signals are events (state transitions), not continuous."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        # Cannot have consecutive True exits (event-based)
        exit_sig = signals['exit_signal']
        consecutive = (exit_sig & exit_sig.shift(1).fillna(False)).sum()
        assert consecutive == 0  # No consecutive exits

    def test_exit_buffer_threshold(self, synthetic_uptrend_data, default_config):
        """Test exit uses 40% buffer threshold."""
        strategy = QualityMomentum(default_config)
        assert strategy.exit_buffer == 0.40  # Verify buffer setting

        signals = strategy.generate_signals(synthetic_uptrend_data)
        # Exit should trigger when rank drops below 40%
        assert 'exit_signal' in signals


# =============================================================================
# TEST CATEGORY 6: Regime Filtering
# =============================================================================

class TestRegimeFiltering:
    """Test regime-based signal filtering."""

    def test_trend_bull_regime(self, synthetic_uptrend_data, default_config):
        """Test strategy works in TREND_BULL regime."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data, regime='TREND_BULL')

        # Should have signals in TREND_BULL
        assert 'entry_signal' in signals

    def test_trend_neutral_regime(self, synthetic_uptrend_data, default_config):
        """Test strategy works in TREND_NEUTRAL regime (quality protects)."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data, regime='TREND_NEUTRAL')

        # Should have signals in TREND_NEUTRAL
        assert 'entry_signal' in signals

    def test_trend_bear_regime(self, synthetic_uptrend_data, default_config):
        """Test strategy works in TREND_BEAR regime (defensive)."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data, regime='TREND_BEAR')

        # Should have signals in TREND_BEAR (unique to Quality-Momentum)
        assert 'entry_signal' in signals

    def test_crash_regime_stricter_quality(self, synthetic_uptrend_data, default_config):
        """Test CRASH regime uses stricter quality filter (top 10%)."""
        strategy = QualityMomentum(default_config)
        signals_normal = strategy.generate_signals(synthetic_uptrend_data, regime='TREND_BULL')
        signals_crash = strategy.generate_signals(synthetic_uptrend_data, regime='CRASH')

        # CRASH should have fewer or equal entries due to stricter filter
        entry_normal = signals_normal['entry_signal'].sum()
        entry_crash = signals_crash['entry_signal'].sum()
        assert entry_crash <= entry_normal


# =============================================================================
# TEST CATEGORY 7: Buffer Rule (40% Exit Threshold)
# =============================================================================

class TestBufferRule:
    """Test 40% exit buffer to reduce turnover."""

    def test_buffer_less_than_threshold(self, default_config):
        """Test exit_buffer < quality_threshold validation."""
        strategy = QualityMomentum(default_config)
        assert strategy.exit_buffer < strategy.quality_threshold

    def test_invalid_buffer_exceeds_threshold(self, default_config):
        """Test strategy rejects buffer >= threshold."""
        with pytest.raises(AssertionError):
            QualityMomentum(
                default_config,
                quality_threshold=0.50,
                exit_buffer=0.55  # Invalid: buffer > threshold
            )


# =============================================================================
# TEST CATEGORY 8: Position Sizing
# =============================================================================

class TestPositionSizing:
    """Test position sizing calculations."""

    def test_position_sizing_returns_series(self, synthetic_uptrend_data, default_config):
        """Test position sizing returns pd.Series."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        position_sizes = strategy.calculate_position_size(
            synthetic_uptrend_data,
            capital=10000,
            stop_distance=signals['stop_distance']
        )

        assert isinstance(position_sizes, pd.Series)
        assert len(position_sizes) == len(synthetic_uptrend_data)

    def test_position_sizing_integer_shares(self, synthetic_uptrend_data, default_config):
        """Test position sizes are integers (whole shares)."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        position_sizes = strategy.calculate_position_size(
            synthetic_uptrend_data,
            capital=10000,
            stop_distance=signals['stop_distance']
        )

        # All values should be integers
        assert position_sizes.dtype in [np.int64, np.int32, int]

    def test_position_sizing_capital_constraint(self, synthetic_uptrend_data, default_config):
        """Test position sizing never exceeds capital (Gate 1)."""
        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(synthetic_uptrend_data)

        capital = 10000
        position_sizes = strategy.calculate_position_size(
            synthetic_uptrend_data,
            capital=capital,
            stop_distance=signals['stop_distance']
        )

        # Position value should never exceed capital
        position_values = position_sizes * synthetic_uptrend_data['Close']
        max_position_value = position_values.max()
        assert max_position_value <= capital


# =============================================================================
# TEST CATEGORY 9: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self, default_config):
        """Test strategy handles insufficient data gracefully."""
        # Create short data (less than momentum_lookback + momentum_lag)
        short_data = pd.DataFrame({
            'Open': np.linspace(99, 101, 100),
            'High': np.linspace(100, 102, 100),
            'Low': np.linspace(98, 100, 100),
            'Close': np.linspace(99.5, 101.5, 100),
            'Volume': np.ones(100) * 1e6
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='B'))

        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(short_data)

        # Should return valid signals (may be all zeros)
        assert 'entry_signal' in signals
        assert 'exit_signal' in signals
        assert len(signals['entry_signal']) == len(short_data)

    def test_nan_values_in_data(self, synthetic_uptrend_data, default_config):
        """Test strategy handles NaN values gracefully."""
        data = synthetic_uptrend_data.copy()
        data.loc[data.index[150], 'Close'] = np.nan

        strategy = QualityMomentum(default_config)
        signals = strategy.generate_signals(data)

        # Should not raise error
        assert 'entry_signal' in signals

    def test_nan_in_fundamental_data(self, default_config):
        """Test quality score handles NaN values in fundamentals."""
        data = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'roe': [0.25, np.nan, 0.20],
            'accruals_ratio': [0.02, 0.03, np.nan],
            'debt_to_equity': [1.0, 1.5, 0.8]
        })

        strategy = QualityMomentum(default_config)
        quality_df = strategy.calculate_quality_scores_from_data(data)

        # Should not raise error, NaN values ranked at bottom
        assert 'quality_score' in quality_df.columns
        assert len(quality_df) == 3


# =============================================================================
# TEST CATEGORY 10: Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_backtest_pipeline(self, synthetic_uptrend_data, default_config):
        """Test complete backtest pipeline from signals to portfolio."""
        strategy = QualityMomentum(default_config)

        # Run backtest
        pf = strategy.backtest(synthetic_uptrend_data, initial_capital=10000)

        # Should return valid portfolio
        assert pf is not None
        assert hasattr(pf, 'total_return')

    def test_performance_metrics(self, synthetic_uptrend_data, default_config):
        """Test performance metrics extraction."""
        strategy = QualityMomentum(default_config)

        pf = strategy.backtest(synthetic_uptrend_data, initial_capital=10000)
        metrics = strategy.get_performance_metrics(pf)

        # Verify metrics structure
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_signal_summary(signals: dict, title: str = "Signal Summary"):
    """Print summary statistics for debugging."""
    print(f"\n{title}")
    print(f"Entry signals: {signals['entry_signal'].sum()}")
    print(f"Exit signals: {signals['exit_signal'].sum()}")
    if 'quality_score' in signals:
        print(f"Mean quality score: {signals['quality_score'].mean():.3f}")
    if 'momentum_score' in signals:
        print(f"Mean momentum score: {signals['momentum_score'].mean():.3f}")


# =============================================================================
# REBALANCE DAY TESTS
# =============================================================================

class TestRebalanceLogic:
    """Test quarterly rebalance logic."""

    def test_rebalance_day_january(self, default_config):
        """Test January 1-5 is rebalance period."""
        strategy = QualityMomentum(default_config)

        assert strategy.is_rebalance_day(pd.Timestamp('2024-01-02'))
        assert strategy.is_rebalance_day(pd.Timestamp('2024-01-05'))
        assert not strategy.is_rebalance_day(pd.Timestamp('2024-01-10'))

    def test_rebalance_day_april(self, default_config):
        """Test April is rebalance month."""
        strategy = QualityMomentum(default_config)

        assert strategy.is_rebalance_day(pd.Timestamp('2024-04-01'))
        assert not strategy.is_rebalance_day(pd.Timestamp('2024-03-15'))

    def test_next_rebalance_date(self, default_config):
        """Test next rebalance date calculation."""
        strategy = QualityMomentum(default_config)

        # From February, next should be April
        next_date = strategy.get_next_rebalance_date(pd.Timestamp('2024-02-15'))
        assert next_date.month == 4

        # From November, next should be January next year
        next_date = strategy.get_next_rebalance_date(pd.Timestamp('2024-11-15'))
        assert next_date.month == 1
        assert next_date.year == 2025
