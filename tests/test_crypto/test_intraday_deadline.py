"""
Tests for intraday position deadline handling - Session EQUITY-99

Tests the 4PM ET deadline enforcement for positions opened with
intraday leverage (10x).
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz

from crypto.simulation.paper_trader import SimulatedTrade
from crypto.simulation.position_monitor import CryptoPositionMonitor, ExitSignal

ET_TIMEZONE = pytz.timezone("America/New_York")


class TestLeverageTierField:
    """Test leverage_tier field in SimulatedTrade."""

    def test_default_leverage_tier_is_swing(self):
        """Default leverage tier should be 'swing'."""
        trade = SimulatedTrade(
            trade_id="TEST001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        assert trade.leverage_tier == "swing"

    def test_leverage_tier_can_be_intraday(self):
        """Leverage tier can be set to 'intraday'."""
        trade = SimulatedTrade(
            trade_id="TEST001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage_tier="intraday",
        )
        assert trade.leverage_tier == "intraday"


class TestIntradayDeadlineCheck:
    """Test intraday deadline check in position monitor."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock position monitor."""
        client = MagicMock()
        paper_trader = MagicMock()
        paper_trader.account.open_trades = []
        return CryptoPositionMonitor(client, paper_trader)

    @pytest.fixture
    def intraday_trade(self):
        """Create an intraday leverage trade."""
        return SimulatedTrade(
            trade_id="INTRA001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage_tier="intraday",
        )

    @pytest.fixture
    def swing_trade(self):
        """Create a swing leverage trade."""
        return SimulatedTrade(
            trade_id="SWING001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage_tier="swing",
        )

    def test_swing_position_no_deadline(self, mock_monitor, swing_trade):
        """Swing leverage position should not trigger deadline exit."""
        # Even near 4PM ET, swing positions should not have deadline
        with patch("crypto.simulation.position_monitor.datetime") as mock_dt:
            # Mock time at 3:58 PM ET
            mock_time = datetime(2026, 1, 31, 15, 58, 0, tzinfo=ET_TIMEZONE)
            mock_dt.now.return_value = mock_time

            signal = mock_monitor._check_intraday_deadline(swing_trade, 51000.0)

            assert signal is None

    def test_intraday_position_triggers_at_deadline(self, mock_monitor, intraday_trade):
        """Intraday position should trigger exit 5 min before 4PM ET."""
        # Mock time_until_intraday_close_et to return 3 minutes
        with patch(
            "crypto.simulation.position_monitor.time_until_intraday_close_et"
        ) as mock_time:
            mock_time.return_value = timedelta(minutes=3)

            with patch("crypto.simulation.position_monitor.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(
                    2026, 1, 31, 15, 57, 0, tzinfo=ET_TIMEZONE
                )

                signal = mock_monitor._check_intraday_deadline(intraday_trade, 51000.0)

                assert signal is not None
                assert signal.reason == "INTRADAY_DEADLINE"
                assert signal.trade == intraday_trade

    def test_intraday_position_no_trigger_outside_window(
        self, mock_monitor, intraday_trade
    ):
        """Intraday position should not trigger if more than 5 min to deadline."""
        # Mock time_until_intraday_close_et to return 30 minutes
        with patch(
            "crypto.simulation.position_monitor.time_until_intraday_close_et"
        ) as mock_time:
            mock_time.return_value = timedelta(minutes=30)

            with patch("crypto.simulation.position_monitor.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(
                    2026, 1, 31, 15, 30, 0, tzinfo=ET_TIMEZONE
                )

                signal = mock_monitor._check_intraday_deadline(intraday_trade, 51000.0)

                assert signal is None

    def test_deadline_exit_calculates_pnl_correctly(self, mock_monitor, intraday_trade):
        """Deadline exit should calculate unrealized P&L correctly."""
        with patch(
            "crypto.simulation.position_monitor.time_until_intraday_close_et"
        ) as mock_time:
            mock_time.return_value = timedelta(minutes=4)

            with patch("crypto.simulation.position_monitor.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(
                    2026, 1, 31, 15, 56, 0, tzinfo=ET_TIMEZONE
                )

                # Entry at 50000, current at 51000 = +$100 P&L (0.1 qty)
                signal = mock_monitor._check_intraday_deadline(intraday_trade, 51000.0)

                assert signal is not None
                assert signal.unrealized_pnl == pytest.approx(100.0, rel=0.01)
                assert signal.unrealized_pnl_percent == pytest.approx(2.0, rel=0.01)


class TestExitPriorityWithDeadline:
    """Test that deadline has highest priority in exit checks."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock position monitor."""
        client = MagicMock()
        paper_trader = MagicMock()
        paper_trader.account.open_trades = []
        return CryptoPositionMonitor(client, paper_trader)

    def test_deadline_beats_target(self, mock_monitor):
        """Deadline exit should have higher priority than target hit."""
        trade = SimulatedTrade(
            trade_id="TEST001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage_tier="intraday",
            target_price=50500.0,  # Target would also be hit
        )

        with patch(
            "crypto.simulation.position_monitor.time_until_intraday_close_et"
        ) as mock_time:
            mock_time.return_value = timedelta(minutes=3)

            with patch("crypto.simulation.position_monitor.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(
                    2026, 1, 31, 15, 57, 0, tzinfo=ET_TIMEZONE
                )

                # Current price would hit target (51000 > 50500)
                signal = mock_monitor._check_trade_exit(trade, 51000.0)

                # But deadline should be checked first
                assert signal is not None
                assert signal.reason == "INTRADAY_DEADLINE"
