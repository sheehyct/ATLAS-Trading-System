"""
Tests for CryptoPositionMonitor - Pattern Invalidation and Exit Priority.

Session EQUITY-68: Tests validate EQUITY-67 changes:
- Type 3 pattern invalidation detection
- Exit priority: Target > Pattern > Stop
- Intrabar high/low tracking

Per STRAT methodology EXECUTION.md Section 8:
- If entry bar evolves from Type 2 to Type 3, exit immediately
- Type 3 = bar breaks BOTH the setup bar high AND low
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from crypto.simulation.paper_trader import SimulatedTrade
from crypto.simulation.position_monitor import CryptoPositionMonitor, ExitSignal


class TestPatternInvalidationDetection:
    """Test _check_pattern_invalidation method."""

    def _create_trade(
        self,
        entry_bar_type: str = "2U",
        entry_bar_high: float = 100.0,
        entry_bar_low: float = 90.0,
        intrabar_high: float = 95.0,
        intrabar_low: float = 92.0,
        side: str = "BUY",
    ) -> SimulatedTrade:
        """Create a test trade with pattern invalidation fields."""
        return SimulatedTrade(
            trade_id="TEST-001",
            symbol="BTC-USD",
            side=side,
            quantity=0.01,
            entry_price=95.0,
            entry_time=datetime.utcnow(),
            stop_price=88.0 if side == "BUY" else 105.0,
            target_price=105.0 if side == "BUY" else 88.0,
            timeframe="1H",
            pattern_type="2-1-2U" if side == "BUY" else "2-1-2D",
            entry_bar_type=entry_bar_type,
            entry_bar_high=entry_bar_high,
            entry_bar_low=entry_bar_low,
            intrabar_high=intrabar_high,
            intrabar_low=intrabar_low,
        )

    def test_2u_to_type3_triggers_exit(self, position_monitor):
        """Entry bar 2U evolving to Type 3 should trigger PATTERN exit."""
        trade = self._create_trade(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=101.0,  # Broke high
            intrabar_low=89.0,    # Broke low -> Type 3!
        )

        current_price = 95.0  # Current price doesn't matter for detection
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "PATTERN"

    def test_2d_to_type3_triggers_exit(self, position_monitor):
        """Entry bar 2D evolving to Type 3 should trigger PATTERN exit."""
        trade = self._create_trade(
            entry_bar_type="2D",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=101.0,  # Broke high
            intrabar_low=89.0,    # Broke low -> Type 3!
            side="SELL",
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "PATTERN"

    def test_only_high_broken_no_exit(self, position_monitor):
        """Breaking only the high should NOT trigger pattern invalidation."""
        trade = self._create_trade(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=101.0,  # Broke high
            intrabar_low=91.0,    # Did NOT break low
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None

    def test_only_low_broken_no_exit(self, position_monitor):
        """Breaking only the low should NOT trigger pattern invalidation."""
        trade = self._create_trade(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=99.0,   # Did NOT break high
            intrabar_low=89.0,    # Broke low
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None

    def test_type3_entry_skips_check(self, position_monitor):
        """Entry bar that's already Type 3 should skip invalidation check."""
        trade = self._create_trade(
            entry_bar_type="3",  # Already Type 3
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=110.0,  # Even if extends further
            intrabar_low=80.0,
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        # Should return None because Type 3 entry doesn't need invalidation check
        assert exit_signal is None

    def test_type1_entry_skips_check(self, position_monitor):
        """Entry bar that's Type 1 (inside) should skip invalidation check."""
        trade = self._create_trade(
            entry_bar_type="1",  # Inside bar
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=101.0,
            intrabar_low=89.0,
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None

    def test_no_entry_bar_data_skips_check(self, position_monitor):
        """Missing entry bar data (high/low = 0) should skip invalidation check."""
        trade = self._create_trade(
            entry_bar_type="2U",
            entry_bar_high=0.0,  # No data
            entry_bar_low=0.0,   # No data
            intrabar_high=110.0,
            intrabar_low=80.0,
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None

    def test_empty_entry_bar_type_skips_check(self, position_monitor):
        """Empty entry bar type should skip invalidation check."""
        trade = self._create_trade(
            entry_bar_type="",  # Empty
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=110.0,
            intrabar_low=80.0,
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None

    def test_exact_boundary_no_exit(self, position_monitor):
        """Exact boundary hits (equal, not exceed) should NOT trigger exit."""
        trade = self._create_trade(
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=100.0,  # Equals, not exceeds
            intrabar_low=90.0,    # Equals, not exceeds
        )

        current_price = 95.0
        exit_signal = position_monitor._check_pattern_invalidation(trade, current_price)

        assert exit_signal is None


class TestIntrabarTracking:
    """Test intrabar high/low tracking updates."""

    def _create_trade(
        self,
        intrabar_high: float = 95.0,
        intrabar_low: float = 95.0,
    ) -> SimulatedTrade:
        """Create a test trade with specified intrabar values."""
        return SimulatedTrade(
            trade_id="TEST-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=95.0,
            entry_time=datetime.utcnow(),
            stop_price=88.0,
            target_price=105.0,
            timeframe="1H",
            pattern_type="2-1-2U",
            entry_bar_type="2U",
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=intrabar_high,
            intrabar_low=intrabar_low,
        )

    def test_intrabar_high_updates(self, position_monitor):
        """Intrabar high should update when price exceeds it."""
        trade = self._create_trade(intrabar_high=95.0, intrabar_low=95.0)

        # Current price higher than intrabar_high
        current_price = 98.0
        position_monitor._check_pattern_invalidation(trade, current_price)

        assert trade.intrabar_high == 98.0

    def test_intrabar_low_updates(self, position_monitor):
        """Intrabar low should update when price goes below it."""
        trade = self._create_trade(intrabar_high=95.0, intrabar_low=95.0)

        # Current price lower than intrabar_low
        current_price = 92.0
        position_monitor._check_pattern_invalidation(trade, current_price)

        assert trade.intrabar_low == 92.0

    def test_intrabar_tracks_extremes(self, position_monitor):
        """Intrabar should track maximum high and minimum low over time."""
        trade = self._create_trade(intrabar_high=95.0, intrabar_low=95.0)

        # First price update - goes up
        position_monitor._check_pattern_invalidation(trade, 98.0)
        assert trade.intrabar_high == 98.0
        assert trade.intrabar_low == 95.0

        # Second price update - goes down
        position_monitor._check_pattern_invalidation(trade, 91.0)
        assert trade.intrabar_high == 98.0  # Still tracks peak
        assert trade.intrabar_low == 91.0

        # Third price update - back up but not new high
        position_monitor._check_pattern_invalidation(trade, 96.0)
        assert trade.intrabar_high == 98.0  # Unchanged
        assert trade.intrabar_low == 91.0   # Unchanged

    def test_type3_detection_after_accumulation(self, position_monitor):
        """Type 3 should be detected after accumulating breaks over multiple checks."""
        trade = self._create_trade(intrabar_high=95.0, intrabar_low=95.0)

        # First check - breaks only high
        result1 = position_monitor._check_pattern_invalidation(trade, 101.0)
        assert result1 is None  # Not Type 3 yet

        # Second check - now breaks low too -> Type 3
        result2 = position_monitor._check_pattern_invalidation(trade, 89.0)
        assert result2 is not None
        assert result2.reason == "PATTERN"


class TestExitPriority:
    """Test exit priority ordering: Target > Pattern > Stop."""

    def _create_trade(
        self,
        entry_bar_type: str = "2U",
        intrabar_high: float = 95.0,
        intrabar_low: float = 95.0,
        target_price: float = 105.0,
        stop_price: float = 88.0,
    ) -> SimulatedTrade:
        """Create a test trade for priority testing."""
        return SimulatedTrade(
            trade_id="TEST-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=95.0,
            entry_time=datetime.utcnow(),
            stop_price=stop_price,
            target_price=target_price,
            timeframe="1H",
            pattern_type="2-1-2U",
            entry_bar_type=entry_bar_type,
            entry_bar_high=100.0,
            entry_bar_low=90.0,
            intrabar_high=intrabar_high,
            intrabar_low=intrabar_low,
        )

    def test_target_beats_pattern_invalidation(self, position_monitor):
        """Target hit should take priority over pattern invalidation."""
        trade = self._create_trade(
            entry_bar_type="2U",
            target_price=105.0,
            intrabar_high=101.0,  # Broke high
            intrabar_low=89.0,    # Broke low -> Type 3
        )

        # Current price hits target
        current_price = 106.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        # Target should win
        assert exit_signal is not None
        assert exit_signal.reason == "TARGET"

    def test_pattern_beats_stop(self, position_monitor):
        """Pattern invalidation should take priority over stop loss."""
        trade = self._create_trade(
            entry_bar_type="2U",
            target_price=105.0,
            stop_price=88.0,
            intrabar_high=101.0,  # Broke high
            intrabar_low=89.0,    # Broke low -> Type 3
        )

        # Current price not at target, not at stop, but pattern invalidated
        current_price = 92.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        # Pattern should win over stop (which would be at 88)
        assert exit_signal is not None
        assert exit_signal.reason == "PATTERN"

    def test_stop_when_no_invalidation(self, position_monitor):
        """Stop should trigger when no target hit and no pattern invalidation."""
        trade = self._create_trade(
            entry_bar_type="2U",
            target_price=105.0,
            stop_price=88.0,
            intrabar_high=99.0,   # Did NOT break high
            intrabar_low=92.0,    # Did NOT break low
        )

        # Current price hits stop
        current_price = 87.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "STOP"

    def test_no_exit_when_no_conditions_met(self, position_monitor):
        """No exit should trigger when no conditions are met."""
        trade = self._create_trade(
            entry_bar_type="2U",
            target_price=105.0,
            stop_price=88.0,
            intrabar_high=95.0,
            intrabar_low=95.0,
        )

        # Current price in middle - no exit conditions
        current_price = 95.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        assert exit_signal is None


class TestExitExecution:
    """Test exit signal execution."""

    def test_execute_exit_closes_trade(self, position_monitor_with_trade):
        """execute_exit should close the trade via paper_trader."""
        # Get the open trade
        trade = position_monitor_with_trade.paper_trader.account.open_trades[0]

        # Create an exit signal
        exit_signal = ExitSignal(
            trade=trade,
            reason="TARGET",
            current_price=52000.0,
            trigger_price=52000.0,
            unrealized_pnl=20.0,
            unrealized_pnl_percent=4.0,
        )

        # Execute the exit
        closed = position_monitor_with_trade.execute_exit(exit_signal)

        # Verify trade was closed
        assert closed is not None
        assert closed.status == "CLOSED"
        assert closed.exit_price == 52000.0

    def test_execute_exit_updates_exit_reason(self, position_monitor_with_trade):
        """execute_exit should set exit_reason on trade."""
        trade = position_monitor_with_trade.paper_trader.account.open_trades[0]

        exit_signal = ExitSignal(
            trade=trade,
            reason="PATTERN",
            current_price=50000.0,
            trigger_price=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
        )

        position_monitor_with_trade.execute_exit(exit_signal)

        # exit_reason should be set
        assert trade.exit_reason == "PATTERN"


class TestCheckExitsIntegration:
    """Integration tests for check_exits method."""

    def test_check_exits_returns_signals(self, mock_client, mock_paper_trader):
        """check_exits should return exit signals for qualifying trades."""
        # Create a trade that will hit target
        trade = SimulatedTrade(
            trade_id="TEST-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            stop_price=49000.0,
            target_price=52000.0,
            timeframe="1H",
            pattern_type="2-1-2U",
            entry_bar_type="2U",
            entry_bar_high=50500.0,
            entry_bar_low=49500.0,
            intrabar_high=50000.0,
            intrabar_low=50000.0,
        )

        mock_paper_trader.account.open_trades = [trade]

        # Price exceeds target
        mock_client.get_current_price.return_value = 53000.0

        monitor = CryptoPositionMonitor(
            client=mock_client,
            paper_trader=mock_paper_trader,
        )

        signals = monitor.check_exits()

        assert len(signals) == 1
        assert signals[0].reason == "TARGET"
        assert signals[0].trade.trade_id == "TEST-001"

    def test_check_exits_multiple_trades(self, mock_client, mock_paper_trader):
        """check_exits should check all open trades."""
        # Create multiple trades
        trade1 = SimulatedTrade(
            trade_id="TEST-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            stop_price=49000.0,
            target_price=55000.0,  # Not hit
            timeframe="1H",
            pattern_type="2-1-2U",
            entry_bar_type="2U",
            entry_bar_high=50500.0,
            entry_bar_low=49500.0,
            intrabar_high=50000.0,
            intrabar_low=50000.0,
        )

        trade2 = SimulatedTrade(
            trade_id="TEST-002",
            symbol="ETH-USD",
            side="BUY",
            quantity=0.1,
            entry_price=3000.0,
            entry_time=datetime.utcnow(),
            stop_price=2900.0,
            target_price=3200.0,  # Will be hit
            timeframe="1H",
            pattern_type="3-2U",
            entry_bar_type="2U",
            entry_bar_high=3050.0,
            entry_bar_low=2950.0,
            intrabar_high=3000.0,
            intrabar_low=3000.0,
        )

        mock_paper_trader.account.open_trades = [trade1, trade2]

        # Set up prices - BTC in middle, ETH hits target
        def get_price(symbol):
            if symbol == "BTC-USD":
                return 51000.0  # No exit
            elif symbol == "ETH-USD":
                return 3300.0  # Target hit
            return None

        mock_client.get_current_price.side_effect = get_price

        monitor = CryptoPositionMonitor(
            client=mock_client,
            paper_trader=mock_paper_trader,
        )

        signals = monitor.check_exits()

        # Only ETH should have exit signal
        assert len(signals) == 1
        assert signals[0].trade.trade_id == "TEST-002"

    def test_check_exits_empty_trades(self, mock_client, mock_paper_trader):
        """check_exits should return empty list when no open trades."""
        mock_paper_trader.account.open_trades = []

        monitor = CryptoPositionMonitor(
            client=mock_client,
            paper_trader=mock_paper_trader,
        )

        signals = monitor.check_exits()

        assert signals == []


class TestSellSideExits:
    """Test exit conditions for SELL (short) positions."""

    def _create_sell_trade(
        self,
        target_price: float = 48000.0,
        stop_price: float = 52000.0,
        entry_bar_type: str = "2D",
        intrabar_high: float = 50000.0,
        intrabar_low: float = 50000.0,
    ) -> SimulatedTrade:
        """Create a test SELL trade."""
        return SimulatedTrade(
            trade_id="TEST-SELL-001",
            symbol="BTC-USD",
            side="SELL",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            stop_price=stop_price,
            target_price=target_price,
            timeframe="1H",
            pattern_type="3-2D",
            entry_bar_type=entry_bar_type,
            entry_bar_high=50500.0,
            entry_bar_low=49500.0,
            intrabar_high=intrabar_high,
            intrabar_low=intrabar_low,
        )

    def test_sell_target_hit(self, position_monitor):
        """SELL target hit when price <= target_price."""
        trade = self._create_sell_trade(target_price=48000.0)

        # Price drops to target
        current_price = 47500.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "TARGET"

    def test_sell_stop_hit(self, position_monitor):
        """SELL stop hit when price >= stop_price."""
        trade = self._create_sell_trade(stop_price=52000.0)

        # Price rises to stop
        current_price = 52500.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "STOP"

    def test_sell_pattern_invalidation(self, position_monitor):
        """SELL position pattern invalidation works correctly."""
        trade = self._create_sell_trade(
            entry_bar_type="2D",
            intrabar_high=50600.0,  # Broke high
            intrabar_low=49400.0,   # Broke low -> Type 3
        )

        # Price in middle
        current_price = 50000.0
        exit_signal = position_monitor._check_trade_exit(trade, current_price)

        assert exit_signal is not None
        assert exit_signal.reason == "PATTERN"
