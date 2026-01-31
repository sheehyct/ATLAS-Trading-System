"""
Tests for fee profitability filter - Session EQUITY-99

Tests the fee filter that rejects trades where fees exceed a threshold
percentage of the expected profit target.
"""

import pytest
from unittest.mock import MagicMock, patch

from crypto import config
from crypto.trading.fees import analyze_fee_impact


class TestFeeProfitabilityConfig:
    """Test fee profitability config constants."""

    def test_filter_enabled_by_default(self):
        """Fee filter should be enabled by default."""
        assert config.FEE_PROFITABILITY_FILTER_ENABLED is True

    def test_max_fee_pct_default(self):
        """Max fee percentage of target should be 20% by default."""
        assert config.MAX_FEE_PCT_OF_TARGET == 0.20


class TestFeeImpactAnalysis:
    """Test the fee impact analysis function used by the filter."""

    def test_small_trade_high_fee_ratio(self):
        """Trade with tight target should have high fee ratio."""
        # $4000 notional with 0.5% target = $20 expected profit
        # Round-trip fees ~$8 = 40% of target
        analysis = analyze_fee_impact(
            account_value=1000.0,
            leverage=4.0,
            price=50000.0,  # BTC price
            symbol="BTC",
            stop_percent=0.02,  # 2% stop
            target_percent=0.005,  # 0.5% target (very tight)
        )

        # Fees should be high relative to target (>20%)
        fee_pct = analysis["fee_as_pct_of_target"]
        assert fee_pct > 0.20  # > 20% of target

    def test_large_trade_low_fee_ratio(self):
        """Large trade with wide target should have low fee ratio."""
        # Large position ($4000) with 4% target = $160 expected profit
        analysis = analyze_fee_impact(
            account_value=1000.0,
            leverage=4.0,
            price=50000.0,  # BTC price
            symbol="BTC",
            stop_percent=0.02,  # 2% stop
            target_percent=0.04,  # 4% target
        )

        # Fees should be low relative to target
        fee_pct = analysis["fee_as_pct_of_target"]
        assert fee_pct < 0.20  # < 20% of target

    def test_fee_analysis_returns_all_metrics(self):
        """Fee analysis should return all expected metrics."""
        analysis = analyze_fee_impact(
            account_value=1000.0,
            leverage=4.0,
            price=50000.0,
            symbol="BTC",
            stop_percent=0.02,
            target_percent=0.04,
        )

        assert "notional" in analysis
        assert "num_contracts" in analysis
        assert "round_trip_fee" in analysis
        assert "breakeven_move" in analysis
        assert "fee_as_pct_of_target" in analysis
        assert "fee_as_pct_of_stop" in analysis
        assert "net_target_pct" in analysis
        assert "net_rr_ratio" in analysis


class TestFeeFilterThreshold:
    """Test the 20% threshold logic."""

    def test_below_threshold_should_pass(self):
        """Trades with fee ratio below 20% should pass filter."""
        analysis = analyze_fee_impact(
            account_value=1000.0,
            leverage=4.0,
            price=50000.0,
            symbol="BTC",
            stop_percent=0.02,
            target_percent=0.04,
        )

        fee_pct = analysis["fee_as_pct_of_target"]
        should_skip = fee_pct > config.MAX_FEE_PCT_OF_TARGET

        assert should_skip is False

    def test_above_threshold_should_fail(self):
        """Trades with fee ratio above 20% should fail filter."""
        # $4000 notional with 0.5% target = fees ~40% of target
        analysis = analyze_fee_impact(
            account_value=1000.0,
            leverage=4.0,
            price=50000.0,  # BTC
            symbol="BTC",
            stop_percent=0.02,
            target_percent=0.005,  # Very tight 0.5% target
        )

        fee_pct = analysis["fee_as_pct_of_target"]
        should_skip = fee_pct > config.MAX_FEE_PCT_OF_TARGET

        assert should_skip is True


class TestStatArbFeeCheck:
    """Test StatArb executor fee check."""

    def test_statarb_fee_check_in_executor(self):
        """StatArb executor should have fee check in _execute_entry."""
        from crypto.scanning.coordinators.statarb_executor import CryptoStatArbExecutor
        import inspect

        # Verify the fee check code is present in the executor
        source = inspect.getsource(CryptoStatArbExecutor._execute_entry)

        assert "FEE_PROFITABILITY_FILTER_ENABLED" in source
        assert "calculate_round_trip_fee" in source
        assert "STATARB SKIPPED (FEES)" in source


class TestDaemonFeeCheckMethod:
    """Test the daemon's _check_fee_profitability method exists and has correct signature."""

    def test_fee_check_method_exists(self):
        """Daemon should have _check_fee_profitability method."""
        from crypto.scanning.daemon import CryptoSignalDaemon
        import inspect

        assert hasattr(CryptoSignalDaemon, "_check_fee_profitability")

        # Check method signature
        sig = inspect.signature(CryptoSignalDaemon._check_fee_profitability)
        params = list(sig.parameters.keys())

        assert "symbol" in params
        assert "entry_price" in params
        assert "stop_price" in params
        assert "target_price" in params
        assert "available_balance" in params
        assert "leverage" in params

    def test_fee_check_in_execute_trade(self):
        """Fee check should be called in _execute_trade method."""
        from crypto.scanning.daemon import CryptoSignalDaemon
        import inspect

        # Verify the fee check is integrated in _execute_trade
        source = inspect.getsource(CryptoSignalDaemon._execute_trade)

        assert "_check_fee_profitability" in source
        assert "SKIPPING TRADE (FEES)" in source
