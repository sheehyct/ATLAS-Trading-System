"""
Tests for crypto/statarb/signal_generator.py

Session EQUITY-91: StatArb signal generation for pairs trading.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict

from crypto.statarb.signal_generator import (
    StatArbSignalGenerator,
    StatArbSignal,
    StatArbSignalType,
    StatArbConfig,
    StatArbPosition,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def config():
    """Default StatArb configuration."""
    return StatArbConfig(
        zscore_window=15,
        entry_threshold=2.0,
        exit_threshold=0.0,
        stop_threshold=3.0,
        min_history_bars=15,
    )


@pytest.fixture
def generator(config):
    """StatArb signal generator for ADA/XRP pair."""
    return StatArbSignalGenerator(
        pairs=[("ADA-USD", "XRP-USD")],
        config=config,
        account_value=1000.0,
    )


def generate_price_history(
    base_price: float,
    num_bars: int,
    volatility: float = 0.01,
) -> list:
    """Generate synthetic price history."""
    import numpy as np
    np.random.seed(42)

    prices = []
    price = base_price
    base_time = datetime(2026, 1, 1, 0, 0)

    for i in range(num_bars):
        prices.append((base_time + timedelta(hours=i), price))
        price = price * (1 + np.random.normal(0, volatility))

    return prices


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestStatArbConfig:
    """Tests for StatArbConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StatArbConfig()

        assert config.zscore_window == 15
        assert config.entry_threshold == 2.0
        assert config.exit_threshold == 0.0
        assert config.stop_threshold == 3.0
        assert config.max_position_pct == 0.20
        assert config.leverage_tier == "swing"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StatArbConfig(
            zscore_window=20,
            entry_threshold=2.5,
            exit_threshold=0.5,
        )

        assert config.zscore_window == 20
        assert config.entry_threshold == 2.5
        assert config.exit_threshold == 0.5


# =============================================================================
# GENERATOR INITIALIZATION TESTS
# =============================================================================


class TestGeneratorInit:
    """Tests for StatArbSignalGenerator initialization."""

    def test_init_with_pairs(self, config):
        """Test initialization with pairs."""
        gen = StatArbSignalGenerator(
            pairs=[("ADA-USD", "XRP-USD")],
            config=config,
        )

        assert len(gen.pairs) == 1
        assert gen.pairs[0] == ("ADA-USD", "XRP-USD")

    def test_init_with_multiple_pairs(self, config):
        """Test initialization with multiple pairs."""
        gen = StatArbSignalGenerator(
            pairs=[("ADA-USD", "XRP-USD"), ("BTC-USD", "ETH-USD")],
            config=config,
        )

        assert len(gen.pairs) == 2

    def test_init_creates_price_history(self, generator):
        """Test that initialization creates price history containers."""
        assert "ADA-USD" in generator._price_history
        assert "XRP-USD" in generator._price_history

    def test_update_account_value(self, generator):
        """Test updating account value."""
        generator.update_account_value(5000.0)
        assert generator.account_value == 5000.0


# =============================================================================
# PRICE UPDATE TESTS
# =============================================================================


class TestPriceUpdates:
    """Tests for price update functionality."""

    def test_update_prices(self, generator):
        """Test updating prices adds to history."""
        prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
        generator.update_prices(prices)

        assert len(generator._price_history["ADA-USD"]) == 1
        assert len(generator._price_history["XRP-USD"]) == 1

    def test_update_prices_multiple_times(self, generator):
        """Test multiple price updates accumulate."""
        for i in range(5):
            prices = {"ADA-USD": 0.35 + i * 0.01, "XRP-USD": 0.50 + i * 0.01}
            generator.update_prices(prices)

        assert len(generator._price_history["ADA-USD"]) == 5
        assert len(generator._price_history["XRP-USD"]) == 5

    def test_update_prices_with_timestamp(self, generator):
        """Test updating prices with custom timestamp."""
        ts = datetime(2026, 1, 15, 12, 0)
        prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
        generator.update_prices(prices, timestamp=ts)

        assert generator._price_history["ADA-USD"][0][0] == ts

    def test_price_history_trimmed(self, generator):
        """Test price history is trimmed to max length."""
        max_history = generator.config.zscore_window * 3

        for i in range(max_history + 10):
            prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
            generator.update_prices(prices)

        assert len(generator._price_history["ADA-USD"]) <= max_history


# =============================================================================
# Z-SCORE CALCULATION TESTS
# =============================================================================


class TestZScoreCalculation:
    """Tests for Z-score calculation."""

    def test_zscore_insufficient_data(self, generator):
        """Test Z-score returns None with insufficient data."""
        # Only add a few bars
        for i in range(5):
            prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
            generator.update_prices(prices)

        zscore = generator._calculate_zscore(("ADA-USD", "XRP-USD"))
        assert zscore is None

    def test_zscore_with_sufficient_data(self):
        """Test Z-score calculation with sufficient data."""
        import numpy as np
        np.random.seed(42)

        # Create fresh generator with low min_history
        config = StatArbConfig(
            zscore_window=10,
            min_history_bars=10,
        )
        gen = StatArbSignalGenerator(
            pairs=[("ADA-USD", "XRP-USD")],
            config=config,
        )

        base_time = datetime(2026, 1, 1, 0, 0)

        # Add enough bars with realistic price movement
        # Use explicit timestamps so both symbols have aligned indices
        for i in range(20):
            ts = base_time + timedelta(hours=i)
            # Create prices with variance so spread has non-zero std
            ada_noise = np.random.normal(0, 0.02)
            xrp_noise = np.random.normal(0, 0.015)
            prices = {
                "ADA-USD": 0.35 * (1 + ada_noise),
                "XRP-USD": 0.50 * (1 + xrp_noise),
            }
            gen.update_prices(prices, timestamp=ts)

        zscore = gen._calculate_zscore(("ADA-USD", "XRP-USD"))
        assert zscore is not None
        assert isinstance(zscore, float)

    def test_get_current_zscore(self, generator):
        """Test getting current Z-score."""
        # Build history
        for i in range(30):
            prices = {
                "ADA-USD": 0.35 + 0.001 * (i % 5 - 2),
                "XRP-USD": 0.50 + 0.001 * (i % 5 - 2),
            }
            generator.update_prices(prices)

        # Calculate Z-score
        generator._calculate_zscore(("ADA-USD", "XRP-USD"))

        zscore = generator.get_current_zscore(("ADA-USD", "XRP-USD"))
        assert zscore is not None


# =============================================================================
# SIGNAL GENERATION TESTS
# =============================================================================


class TestSignalGeneration:
    """Tests for signal generation."""

    def test_no_signals_insufficient_data(self, generator):
        """Test no signals with insufficient data."""
        signals = generator.check_for_signals({"ADA-USD": 0.35, "XRP-USD": 0.50})
        assert signals == []

    def test_check_for_signals_updates_prices(self, generator):
        """Test check_for_signals updates prices."""
        prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
        generator.check_for_signals(prices)

        assert len(generator._price_history["ADA-USD"]) == 1

    def test_long_spread_signal_generation(self):
        """Test long spread signal when Z-score drops below threshold."""
        config = StatArbConfig(
            zscore_window=10,
            entry_threshold=2.0,
            min_history_bars=10,
        )
        gen = StatArbSignalGenerator(
            pairs=[("ADA-USD", "XRP-USD")],
            config=config,
            account_value=1000.0,
        )

        # Build stable history
        for i in range(15):
            prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
            gen.update_prices(prices)

        # Simulate Z-score crossing below -2.0
        # Force previous Z-score to be above threshold
        gen._last_zscore[("ADA-USD", "XRP-USD")] = -1.5

        # Drop ADA price significantly to push Z-score below -2
        prices = {"ADA-USD": 0.30, "XRP-USD": 0.50}
        gen.update_prices(prices)

        # The signal generator needs a large move to trigger
        # For testing, we can check the entry logic directly
        signal = gen._check_entry(
            ("ADA-USD", "XRP-USD"),
            zscore=-2.5,
            prev_zscore=-1.5,
        )

        assert signal is not None
        assert signal.signal_type == StatArbSignalType.LONG_SPREAD

    def test_short_spread_signal_generation(self, generator):
        """Test short spread signal when Z-score rises above threshold."""
        # Force previous Z-score to be below threshold
        generator._last_zscore[("ADA-USD", "XRP-USD")] = 1.5

        signal = generator._check_entry(
            ("ADA-USD", "XRP-USD"),
            zscore=2.5,
            prev_zscore=1.5,
        )

        assert signal is not None
        assert signal.signal_type == StatArbSignalType.SHORT_SPREAD


# =============================================================================
# POSITION TRACKING TESTS
# =============================================================================


class TestPositionTracking:
    """Tests for position tracking."""

    def test_no_position_initially(self, generator):
        """Test no position exists initially."""
        assert generator.has_position(("ADA-USD", "XRP-USD")) is False

    def test_get_position_none(self, generator):
        """Test get_position returns None when no position."""
        pos = generator.get_position(("ADA-USD", "XRP-USD"))
        assert pos is None

    def test_get_all_positions_empty(self, generator):
        """Test get_all_positions returns empty dict."""
        positions = generator.get_all_positions()
        assert positions == {}

    def test_get_active_symbols_empty(self, generator):
        """Test get_active_symbols returns empty set."""
        symbols = generator.get_active_symbols()
        assert symbols == set()

    def test_close_position_none(self, generator):
        """Test close_position returns None for non-existent position."""
        result = generator.close_position(("ADA-USD", "XRP-USD"))
        assert result is None


# =============================================================================
# EXIT SIGNAL TESTS
# =============================================================================


class TestExitSignals:
    """Tests for exit signal generation."""

    def test_exit_on_mean_reversion_long(self, generator):
        """Test exit signal on mean reversion for long spread."""
        position = StatArbPosition(
            pair=("ADA-USD", "XRP-USD"),
            direction="long_spread",
            entry_zscore=-2.5,
            entry_spread=0.0,
            entry_time=datetime.utcnow(),
            hedge_ratio=1.0,
            long_notional=100.0,
            short_notional=100.0,
            bars_held=5,
        )

        # Z-score crosses above exit threshold (0.0)
        signal = generator._check_exit(
            ("ADA-USD", "XRP-USD"),
            position,
            zscore=0.5,  # Above exit threshold of 0.0
        )

        assert signal is not None
        assert signal.signal_type == StatArbSignalType.EXIT

    def test_exit_on_mean_reversion_short(self, generator):
        """Test exit signal on mean reversion for short spread."""
        position = StatArbPosition(
            pair=("ADA-USD", "XRP-USD"),
            direction="short_spread",
            entry_zscore=2.5,
            entry_spread=0.0,
            entry_time=datetime.utcnow(),
            hedge_ratio=1.0,
            long_notional=100.0,
            short_notional=100.0,
            bars_held=5,
        )

        # Z-score crosses below exit threshold (0.0)
        signal = generator._check_exit(
            ("ADA-USD", "XRP-USD"),
            position,
            zscore=-0.5,
        )

        assert signal is not None
        assert signal.signal_type == StatArbSignalType.EXIT

    def test_no_exit_during_min_hold(self, generator):
        """Test no exit signal during minimum hold period."""
        position = StatArbPosition(
            pair=("ADA-USD", "XRP-USD"),
            direction="long_spread",
            entry_zscore=-2.5,
            entry_spread=0.0,
            entry_time=datetime.utcnow(),
            hedge_ratio=1.0,
            long_notional=100.0,
            short_notional=100.0,
            bars_held=0,  # Just entered
        )

        signal = generator._check_exit(
            ("ADA-USD", "XRP-USD"),
            position,
            zscore=0.5,
        )

        assert signal is None

    def test_stop_out_on_extreme_zscore(self, generator):
        """Test stop out when Z-score exceeds threshold."""
        position = StatArbPosition(
            pair=("ADA-USD", "XRP-USD"),
            direction="long_spread",
            entry_zscore=-2.5,
            entry_spread=0.0,
            entry_time=datetime.utcnow(),
            hedge_ratio=1.0,
            long_notional=100.0,
            short_notional=100.0,
            bars_held=5,
        )

        # Z-score goes extreme
        signal = generator._check_exit(
            ("ADA-USD", "XRP-USD"),
            position,
            zscore=-4.0,  # Exceeds stop_threshold of 3.0
        )

        assert signal is not None
        assert signal.signal_type == StatArbSignalType.EXIT


# =============================================================================
# STATUS TESTS
# =============================================================================


class TestStatus:
    """Tests for status reporting."""

    def test_get_status(self, generator):
        """Test get_status returns expected format."""
        status = generator.get_status()

        assert "pairs" in status
        assert "positions" in status
        assert "zscores" in status
        assert "history_bars" in status
        assert status["positions"] == 0

    def test_get_status_with_data(self, generator):
        """Test get_status with price data."""
        for i in range(20):
            prices = {"ADA-USD": 0.35, "XRP-USD": 0.50}
            generator.update_prices(prices)

        status = generator.get_status()

        assert status["history_bars"]["ADA-USD/XRP-USD"] == 20
