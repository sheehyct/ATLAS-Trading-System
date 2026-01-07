"""
Tests for Type 3 Pattern Invalidation Exit (EQUITY-44).

Per STRAT methodology EXECUTION.md Section 8:
- If entry bar evolves from Type 2 to Type 3, exit immediately
- Type 3 = bar breaks BOTH the entry bar high AND low
- Exit Priority: Target > Pattern Invalidated > Traditional Stop
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from strat.signal_automation.position_monitor import (
    PositionMonitor,
    TrackedPosition,
    ExitSignal,
    ExitReason,
    MonitoringConfig,
)


class TestPatternInvalidationDetection:
    """Test _check_pattern_invalidation method."""

    def setup_method(self):
        """Create test monitor with mocked dependencies."""
        self.config = MonitoringConfig(minimum_hold_seconds=0)
        self.monitor = PositionMonitor(config=self.config)

    def _create_position(
        self,
        entry_bar_type: str = "2U",
        entry_bar_high: float = 100.0,
        entry_bar_low: float = 90.0,
        direction: str = "CALL",
    ) -> TrackedPosition:
        """Create a test position with entry bar data."""
        return TrackedPosition(
            osi_symbol="SPY240101C00450000",
            signal_key="test_signal",
            symbol="SPY",
            direction=direction,
            entry_trigger=95.0,
            target_price=105.0,
            stop_price=88.0,
            pattern_type="2-1-2U",
            timeframe="1H",
            entry_price=5.0,
            contracts=1,
            entry_time=datetime.now() - timedelta(hours=1),
            expiration="2024-01-01",
            current_price=5.50,
            underlying_price=96.0,
            dte=14,
            entry_bar_type=entry_bar_type,
            entry_bar_high=entry_bar_high,
            entry_bar_low=entry_bar_low,
        )

    def test_2u_to_type3_triggers_exit(self):
        """Entry bar 2U evolving to Type 3 should trigger PATTERN_INVALIDATED exit."""
        pos = self._create_position(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        # Current bar breaks BOTH high and low -> Type 3
        self.monitor._bar_cache["SPY"] = {
            "high": 101.0,  # > entry bar high (100)
            "low": 89.0,    # < entry bar low (90)
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.PATTERN_INVALIDATED
        assert "Type 3" in exit_signal.details

    def test_2d_to_type3_triggers_exit(self):
        """Entry bar 2D evolving to Type 3 should trigger PATTERN_INVALIDATED exit."""
        pos = self._create_position(
            entry_bar_type="2D",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            direction="PUT",
        )

        # Current bar breaks BOTH high and low -> Type 3
        self.monitor._bar_cache["SPY"] = {
            "high": 101.0,  # > entry bar high (100)
            "low": 89.0,    # < entry bar low (90)
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.PATTERN_INVALIDATED

    def test_only_high_broken_no_exit(self):
        """Breaking only the high should NOT trigger pattern invalidation."""
        pos = self._create_position(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        # Current bar breaks high only -> still Type 2U, not invalidated
        self.monitor._bar_cache["SPY"] = {
            "high": 101.0,  # > entry bar high (100)
            "low": 91.0,    # > entry bar low (90) - NOT broken
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is None

    def test_only_low_broken_no_exit(self):
        """Breaking only the low should NOT trigger pattern invalidation."""
        pos = self._create_position(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        # Current bar breaks low only -> still Type 2D variant, not Type 3
        self.monitor._bar_cache["SPY"] = {
            "high": 99.0,   # < entry bar high (100) - NOT broken
            "low": 89.0,    # < entry bar low (90)
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is None

    def test_type3_entry_skips_check(self):
        """Entry bar that's already Type 3 should skip invalidation check."""
        pos = self._create_position(
            entry_bar_type="3",  # Already Type 3
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        # Even if bar extends, Type 3 can't evolve further
        self.monitor._bar_cache["SPY"] = {
            "high": 110.0,
            "low": 80.0,
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        # Should return None because Type 3 entry doesn't need invalidation check
        assert exit_signal is None

    def test_no_entry_bar_data_skips_check(self):
        """Missing entry bar data should skip invalidation check."""
        pos = self._create_position(
            entry_bar_type="2U",
            entry_bar_high=0.0,  # No data
            entry_bar_low=0.0,   # No data
        )

        self.monitor._bar_cache["SPY"] = {
            "high": 110.0,
            "low": 80.0,
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is None

    def test_no_bar_cache_skips_check(self):
        """Missing bar cache data should skip invalidation check."""
        pos = self._create_position(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        # No cache entry for SPY
        self.monitor._bar_cache = {}

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is None

    def test_empty_entry_bar_type_skips_check(self):
        """Empty entry bar type should skip invalidation check."""
        pos = self._create_position(
            entry_bar_type="",  # Empty
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        self.monitor._bar_cache["SPY"] = {
            "high": 110.0,
            "low": 80.0,
        }

        exit_signal = self.monitor._check_pattern_invalidation(pos)

        assert exit_signal is None


class TestPatternInvalidationExitPriority:
    """Test exit priority with pattern invalidation."""

    def setup_method(self):
        """Create test monitor with mocked dependencies."""
        self.config = MonitoringConfig(
            minimum_hold_seconds=0,
            exit_dte=3,
        )
        self.monitor = PositionMonitor(config=self.config)
        # Mock _calculate_dte to return a fixed value (avoid date issues)
        self.monitor._calculate_dte = lambda exp: 14

    def _create_position(self, **kwargs) -> TrackedPosition:
        """Create a test position with sensible defaults."""
        # Use a future expiration date
        future_exp = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
        defaults = {
            "osi_symbol": "SPY260101C00450000",
            "signal_key": "test_signal",
            "symbol": "SPY",
            "direction": "CALL",
            "entry_trigger": 450.0,
            "target_price": 460.0,
            "stop_price": 445.0,
            "pattern_type": "2-1-2U",
            "timeframe": "1D",
            "entry_price": 5.0,
            "contracts": 1,
            "entry_time": datetime.now() - timedelta(hours=1),
            "expiration": future_exp,
            "current_price": 5.50,
            "underlying_price": 452.0,
            "dte": 14,
            "entry_bar_type": "2U",
            "entry_bar_high": 451.0,
            "entry_bar_low": 449.0,
        }
        defaults.update(kwargs)
        return TrackedPosition(**defaults)

    def test_target_hit_before_pattern_invalidation(self):
        """Target hit should take priority over pattern invalidation."""
        pos = self._create_position(
            target_price=455.0,
            underlying_price=456.0,  # Hit target
        )
        self.monitor._positions[pos.osi_symbol] = pos

        # Set up cache for underlying price
        self.monitor._underlying_cache["SPY"] = {"price": 456.0}

        # Set up bar cache showing Type 3 evolution
        self.monitor._bar_cache["SPY"] = {
            "high": 460.0,  # > entry bar high
            "low": 448.0,   # < entry bar low
        }

        exit_signal = self.monitor._check_position(pos)

        # Target hit should win
        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.TARGET_HIT

    def test_pattern_invalidation_before_trailing_stop(self):
        """Pattern invalidation should take priority over trailing stop."""
        pos = self._create_position(
            target_price=460.0,
            underlying_price=453.0,  # Not at target
            trailing_stop_active=True,
            trailing_stop_price=451.0,
        )
        self.monitor._positions[pos.osi_symbol] = pos

        # Set up cache
        self.monitor._underlying_cache["SPY"] = {"price": 453.0}

        # Set up bar cache showing Type 3 evolution
        self.monitor._bar_cache["SPY"] = {
            "high": 455.0,  # > entry bar high (451)
            "low": 448.0,   # < entry bar low (449)
        }

        # Enable trailing stop
        self.monitor.config.use_trailing_stop = True

        exit_signal = self.monitor._check_position(pos)

        # Pattern invalidation should win over trailing stop
        assert exit_signal is not None
        assert exit_signal.reason == ExitReason.PATTERN_INVALIDATED


class TestExecutionResultEntryBarData:
    """Test ExecutionResult captures entry bar data."""

    def test_execution_result_has_entry_bar_fields(self):
        """ExecutionResult should have entry bar fields."""
        from strat.signal_automation.executor import ExecutionResult, ExecutionState

        result = ExecutionResult(
            signal_key="test",
            state=ExecutionState.ORDER_FILLED,
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
        )

        assert result.entry_bar_type == "2U"
        assert result.entry_bar_high == 100.0
        assert result.entry_bar_low == 90.0

    def test_execution_result_to_dict_includes_entry_bar(self):
        """ExecutionResult.to_dict() should include entry bar fields."""
        from strat.signal_automation.executor import ExecutionResult, ExecutionState

        result = ExecutionResult(
            signal_key="test",
            state=ExecutionState.ORDER_FILLED,
            entry_bar_type="2D",
            entry_bar_high=50.0,
            entry_bar_low=45.0,
        )

        data = result.to_dict()

        assert data["entry_bar_type"] == "2D"
        assert data["entry_bar_high"] == 50.0
        assert data["entry_bar_low"] == 45.0

    def test_execution_result_from_dict_loads_entry_bar(self):
        """ExecutionResult.from_dict() should load entry bar fields."""
        from strat.signal_automation.executor import ExecutionResult, ExecutionState

        data = {
            "signal_key": "test",
            "state": "filled",
            "entry_bar_type": "2U",
            "entry_bar_high": 75.0,
            "entry_bar_low": 70.0,
        }

        result = ExecutionResult.from_dict(data)

        assert result.entry_bar_type == "2U"
        assert result.entry_bar_high == 75.0
        assert result.entry_bar_low == 70.0


class TestTrackedPositionEntryBarData:
    """Test TrackedPosition has entry bar fields."""

    def test_tracked_position_has_entry_bar_fields(self):
        """TrackedPosition should have entry bar fields."""
        pos = TrackedPosition(
            osi_symbol="SPY240101C00450000",
            signal_key="test",
            symbol="SPY",
            direction="CALL",
            entry_trigger=450.0,
            target_price=460.0,
            stop_price=445.0,
            pattern_type="2-1-2U",
            timeframe="1H",
            entry_price=5.0,
            contracts=1,
            entry_time=datetime.now(),
            expiration="2024-01-01",
            entry_bar_type="2U",
            entry_bar_high=451.0,
            entry_bar_low=449.0,
        )

        assert pos.entry_bar_type == "2U"
        assert pos.entry_bar_high == 451.0
        assert pos.entry_bar_low == 449.0

    def test_tracked_position_to_dict_includes_entry_bar(self):
        """TrackedPosition.to_dict() should include entry bar fields."""
        pos = TrackedPosition(
            osi_symbol="SPY240101C00450000",
            signal_key="test",
            symbol="SPY",
            direction="CALL",
            entry_trigger=450.0,
            target_price=460.0,
            stop_price=445.0,
            pattern_type="2-1-2U",
            timeframe="1H",
            entry_price=5.0,
            contracts=1,
            entry_time=datetime.now(),
            expiration="2024-01-01",
            entry_bar_type="2D",
            entry_bar_high=100.0,
            entry_bar_low=95.0,
        )

        data = pos.to_dict()

        assert data["entry_bar_type"] == "2D"
        assert data["entry_bar_high"] == 100.0
        assert data["entry_bar_low"] == 95.0
