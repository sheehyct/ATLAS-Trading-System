"""
Tests for crypto/simulation/paper_trader.py

Session EQUITY-72: Comprehensive test coverage for PaperTrader.

Covers:
- SimulatedTrade dataclass (creation, close, serialization)
- PaperTradingAccount dataclass
- PaperTrader class:
  - Initialization and trade ID generation
  - Trading operations (open, close, close_all)
  - Margin tracking and balance management
  - Account info and position aggregation
  - Performance metrics calculation
  - Trade history
  - Persistence (save/load state)
  - Reset functionality
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import json
import tempfile
import os

from crypto.simulation.paper_trader import (
    SimulatedTrade,
    PaperTradingAccount,
    PaperTrader,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for paper trading data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def paper_trader(temp_data_dir):
    """Create a PaperTrader with temporary data directory."""
    return PaperTrader(
        starting_balance=1000.0,
        data_dir=temp_data_dir,
        account_name="test",
    )


@pytest.fixture
def sample_trade():
    """Create a sample SimulatedTrade."""
    return SimulatedTrade(
        trade_id="SIM-test-00001",
        symbol="BTC-USD",
        side="BUY",
        quantity=0.01,
        entry_price=50000.0,
        entry_time=datetime(2026, 1, 15, 12, 0),
        stop_price=49000.0,
        target_price=52000.0,
        timeframe="1h",
        pattern_type="3-2U",
        tfc_score=3,
        risk_multiplier=1.0,
        priority_rank=1,
        entry_bar_type="2U",
        entry_bar_high=50500.0,
        entry_bar_low=49500.0,
    )


# =============================================================================
# SIMULATED TRADE TESTS
# =============================================================================


class TestSimulatedTradeCreation:
    """Tests for SimulatedTrade creation."""

    def test_create_basic_trade(self):
        """Test creating a basic trade."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        assert trade.trade_id == "SIM-001"
        assert trade.symbol == "BTC-USD"
        assert trade.side == "BUY"
        assert trade.status == "OPEN"
        assert trade.pnl is None

    def test_trade_default_values(self):
        """Test trade default values."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        assert trade.exit_price is None
        assert trade.stop_price is None
        assert trade.target_price is None
        assert trade.risk_multiplier == 1.0
        assert trade.margin_reserved == 0.0
        assert trade.intrabar_low == float("inf")

    def test_trade_with_all_fields(self, sample_trade):
        """Test trade with all fields populated."""
        assert sample_trade.stop_price == 49000.0
        assert sample_trade.target_price == 52000.0
        assert sample_trade.timeframe == "1h"
        assert sample_trade.pattern_type == "3-2U"
        assert sample_trade.entry_bar_type == "2U"


class TestSimulatedTradeClose:
    """Tests for SimulatedTrade.close() method."""

    def test_close_winning_buy(self):
        """Test closing a winning BUY trade."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        trade.close(51000.0)  # Exit at $51000

        assert trade.status == "CLOSED"
        assert trade.exit_price == 51000.0
        assert trade.pnl == 10.0  # (51000 - 50000) * 0.01
        assert trade.pnl_percent == pytest.approx(2.0)  # 2% gain

    def test_close_losing_buy(self):
        """Test closing a losing BUY trade."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        trade.close(49000.0)  # Exit at $49000

        assert trade.pnl == -10.0  # (49000 - 50000) * 0.01
        assert trade.pnl_percent == pytest.approx(-2.0)

    def test_close_winning_sell(self):
        """Test closing a winning SELL trade."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="SELL",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        trade.close(49000.0)  # Exit at $49000 (price went down = profit)

        assert trade.pnl == 10.0  # (50000 - 49000) * 0.01
        assert trade.pnl_percent == pytest.approx(2.0)

    def test_close_losing_sell(self):
        """Test closing a losing SELL trade."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="SELL",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        trade.close(51000.0)  # Exit at $51000 (price went up = loss)

        assert trade.pnl == -10.0
        assert trade.pnl_percent == pytest.approx(-2.0)

    def test_close_sets_exit_time(self):
        """Test close sets exit_time."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        custom_time = datetime(2026, 1, 16, 12, 0)
        trade.close(51000.0, exit_time=custom_time)

        assert trade.exit_time == custom_time

    def test_close_defaults_exit_time_to_now(self):
        """Test close defaults exit_time to now."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        before = datetime.utcnow()
        trade.close(51000.0)
        after = datetime.utcnow()

        assert before <= trade.exit_time <= after


class TestSimulatedTradeSerialization:
    """Tests for SimulatedTrade serialization."""

    def test_to_dict(self, sample_trade):
        """Test to_dict serialization."""
        d = sample_trade.to_dict()

        assert d["trade_id"] == "SIM-test-00001"
        assert d["symbol"] == "BTC-USD"
        assert d["side"] == "BUY"
        assert d["quantity"] == 0.01
        assert d["entry_price"] == 50000.0
        assert d["stop_price"] == 49000.0
        assert d["pattern_type"] == "3-2U"
        assert d["entry_bar_type"] == "2U"

    def test_to_dict_handles_infinity(self):
        """Test to_dict converts infinity to 0."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            intrabar_low=float("inf"),
        )
        d = trade.to_dict()
        assert d["intrabar_low"] == 0.0

    def test_from_dict(self, sample_trade):
        """Test from_dict deserialization."""
        d = sample_trade.to_dict()
        restored = SimulatedTrade.from_dict(d)

        assert restored.trade_id == sample_trade.trade_id
        assert restored.symbol == sample_trade.symbol
        assert restored.entry_price == sample_trade.entry_price
        assert restored.pattern_type == sample_trade.pattern_type

    def test_from_dict_with_exit(self, sample_trade):
        """Test from_dict with closed trade."""
        sample_trade.close(52000.0)
        d = sample_trade.to_dict()
        restored = SimulatedTrade.from_dict(d)

        assert restored.status == "CLOSED"
        assert restored.exit_price == 52000.0
        assert restored.pnl == sample_trade.pnl

    def test_roundtrip_serialization(self, sample_trade):
        """Test roundtrip serialization preserves all fields."""
        sample_trade.close(52000.0)
        d = sample_trade.to_dict()
        restored = SimulatedTrade.from_dict(d)

        # Compare all serializable fields
        assert restored.to_dict() == sample_trade.to_dict()


# =============================================================================
# PAPER TRADING ACCOUNT TESTS
# =============================================================================


class TestPaperTradingAccount:
    """Tests for PaperTradingAccount dataclass."""

    def test_default_values(self):
        """Test default values."""
        account = PaperTradingAccount()
        assert account.starting_balance == 1000.0
        assert account.current_balance == 1000.0
        assert account.realized_pnl == 0.0
        assert account.open_trades == []
        assert account.closed_trades == []
        assert account.reserved_margin == 0.0

    def test_custom_balance(self):
        """Test custom starting balance."""
        account = PaperTradingAccount(
            starting_balance=5000.0,
            current_balance=5000.0,
        )
        assert account.starting_balance == 5000.0


# =============================================================================
# PAPER TRADER INITIALIZATION TESTS
# =============================================================================


class TestPaperTraderInit:
    """Tests for PaperTrader initialization."""

    def test_init_creates_data_dir(self, temp_data_dir):
        """Test init creates data directory."""
        new_dir = temp_data_dir / "new_subdir"
        trader = PaperTrader(data_dir=new_dir)
        assert new_dir.exists()

    def test_init_default_balance(self, paper_trader):
        """Test init with default balance."""
        assert paper_trader.account.starting_balance == 1000.0
        assert paper_trader.account.current_balance == 1000.0

    def test_init_custom_balance(self, temp_data_dir):
        """Test init with custom balance."""
        trader = PaperTrader(
            starting_balance=5000.0,
            data_dir=temp_data_dir,
        )
        assert trader.account.starting_balance == 5000.0

    def test_init_loads_existing_state(self, temp_data_dir):
        """Test init loads existing state from file."""
        # Create initial trader and make a trade
        trader1 = PaperTrader(
            starting_balance=1000.0,
            data_dir=temp_data_dir,
            account_name="persist_test",
        )
        trader1.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        # Create new trader - should load existing state
        trader2 = PaperTrader(
            data_dir=temp_data_dir,
            account_name="persist_test",
        )
        assert len(trader2.account.open_trades) == 1

    def test_generate_trade_id(self, paper_trader):
        """Test trade ID generation."""
        id1 = paper_trader._generate_trade_id()
        id2 = paper_trader._generate_trade_id()

        assert id1.startswith("SIM-test-")
        assert id2.startswith("SIM-test-")
        assert id1 != id2


# =============================================================================
# TRADING OPERATIONS TESTS
# =============================================================================


class TestOpenTrade:
    """Tests for open_trade method."""

    def test_open_basic_trade(self, paper_trader):
        """Test opening a basic trade."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        assert trade is not None
        assert trade.symbol == "BTC-USD"
        assert trade.side == "BUY"
        assert len(paper_trader.account.open_trades) == 1

    def test_open_trade_reserves_margin(self, paper_trader):
        """Test opening trade reserves margin."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            leverage=4.0,
        )
        # Margin = 0.01 * 50000 / 4 = 125
        assert paper_trader.account.reserved_margin == 125.0

    def test_open_trade_insufficient_margin(self, paper_trader):
        """Test opening trade fails with insufficient margin."""
        # Try to open trade requiring more margin than available
        # $1000 balance, 4x leverage -> max $4000 notional -> $80k position
        # But 0.1 BTC * 50000 / 4 = 1250 > 1000
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,  # $5000 notional / 4 = $1250 margin needed
            entry_price=50000.0,
            leverage=4.0,
        )
        assert trade is None

    def test_open_trade_with_risk_multiplier(self, paper_trader):
        """Test risk_multiplier adjusts quantity."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            risk_multiplier=0.5,
        )

        assert trade is not None
        assert trade.quantity == 0.005  # Reduced by multiplier

    def test_open_trade_zero_risk_multiplier(self, paper_trader):
        """Test zero risk_multiplier skips trade."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            risk_multiplier=0.0,
        )
        assert trade is None

    def test_open_trade_with_pattern_fields(self, paper_trader):
        """Test opening trade with STRAT pattern fields."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            timeframe="1h",
            pattern_type="3-2U",
            tfc_score=3,
            entry_bar_type="2U",
            entry_bar_high=50500.0,
            entry_bar_low=49500.0,
        )

        assert trade.timeframe == "1h"
        assert trade.pattern_type == "3-2U"
        assert trade.entry_bar_type == "2U"

    def test_open_trade_initializes_intrabar(self, paper_trader):
        """Test intrabar high/low initialized to entry price."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        assert trade.intrabar_high == 50000.0
        assert trade.intrabar_low == 50000.0


class TestCloseTrade:
    """Tests for close_trade method."""

    def test_close_trade(self, paper_trader):
        """Test closing a trade."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        trade_id = trade.trade_id

        closed = paper_trader.close_trade(trade_id, exit_price=51000.0)

        assert closed is not None
        assert closed.status == "CLOSED"
        assert len(paper_trader.account.open_trades) == 0
        assert len(paper_trader.account.closed_trades) == 1

    def test_close_trade_releases_margin(self, paper_trader):
        """Test closing trade releases margin."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            leverage=4.0,
        )
        initial_margin = paper_trader.account.reserved_margin

        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        assert paper_trader.account.reserved_margin == 0.0
        assert initial_margin > 0

    def test_close_trade_updates_balance(self, paper_trader):
        """Test closing trade updates account balance."""
        initial_balance = paper_trader.account.current_balance

        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        # P&L = (51000 - 50000) * 0.01 = $10
        assert paper_trader.account.current_balance == initial_balance + 10.0
        assert paper_trader.account.realized_pnl == 10.0

    def test_close_trade_not_found(self, paper_trader):
        """Test closing non-existent trade."""
        result = paper_trader.close_trade("nonexistent-id", exit_price=50000.0)
        assert result is None


class TestCloseAllTrades:
    """Tests for close_all_trades method."""

    def test_close_all_trades(self, paper_trader):
        """Test closing all trades."""
        paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01, entry_price=50000.0
        )
        paper_trader.open_trade(
            symbol="ETH-USD", side="BUY", quantity=0.1, entry_price=3500.0
        )

        closed = paper_trader.close_all_trades(exit_price=50000.0)

        assert len(closed) == 2
        assert len(paper_trader.account.open_trades) == 0

    def test_close_all_trades_by_symbol(self, paper_trader):
        """Test closing trades filtered by symbol."""
        paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01, entry_price=50000.0
        )
        paper_trader.open_trade(
            symbol="ETH-USD", side="BUY", quantity=0.1, entry_price=3500.0
        )

        closed = paper_trader.close_all_trades(symbol="BTC-USD", exit_price=51000.0)

        assert len(closed) == 1
        assert closed[0].symbol == "BTC-USD"
        assert len(paper_trader.account.open_trades) == 1

    def test_close_all_trades_no_price(self, paper_trader):
        """Test close_all_trades without price returns empty."""
        paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01, entry_price=50000.0
        )

        closed = paper_trader.close_all_trades()  # No exit_price

        assert closed == []
        assert len(paper_trader.account.open_trades) == 1


class TestGetAvailableBalance:
    """Tests for get_available_balance method."""

    def test_available_balance_no_trades(self, paper_trader):
        """Test available balance with no trades."""
        assert paper_trader.get_available_balance() == 1000.0

    def test_available_balance_with_reserved_margin(self, paper_trader):
        """Test available balance after opening trade."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            leverage=4.0,
        )
        # Margin = 500 / 4 = 125
        assert paper_trader.get_available_balance() == 875.0


# =============================================================================
# ACCOUNT INFO TESTS
# =============================================================================


class TestGetOpenPosition:
    """Tests for get_open_position method."""

    def test_get_open_position(self, paper_trader):
        """Test getting aggregated position."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        position = paper_trader.get_open_position("BTC-USD")

        assert position is not None
        assert position["symbol"] == "BTC-USD"
        assert position["side"] == "BUY"
        assert position["quantity"] == 0.01

    def test_get_open_position_multiple_trades(self, paper_trader):
        """Test aggregated position with multiple trades."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.02,
            entry_price=51000.0,
        )

        position = paper_trader.get_open_position("BTC-USD")

        assert position["quantity"] == 0.03
        # Avg price = (0.01*50000 + 0.02*51000) / 0.03 = 50666.67
        assert position["avg_entry_price"] == pytest.approx(50666.67, rel=0.01)

    def test_get_open_position_no_position(self, paper_trader):
        """Test getting position for symbol with no trades."""
        position = paper_trader.get_open_position("BTC-USD")
        assert position is None

    def test_get_open_position_different_symbol(self, paper_trader):
        """Test getting position for wrong symbol."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        position = paper_trader.get_open_position("ETH-USD")
        assert position is None


class TestGetAccountSummary:
    """Tests for get_account_summary method."""

    def test_account_summary_initial(self, paper_trader):
        """Test account summary with no trades."""
        summary = paper_trader.get_account_summary()

        assert summary["account_name"] == "test"
        assert summary["starting_balance"] == 1000.0
        assert summary["current_balance"] == 1000.0
        assert summary["reserved_margin"] == 0.0
        assert summary["available_balance"] == 1000.0
        assert summary["realized_pnl"] == 0.0
        assert summary["return_percent"] == 0.0

    def test_account_summary_after_trades(self, paper_trader):
        """Test account summary after trading."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            leverage=4.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=52000.0)

        summary = paper_trader.get_account_summary()

        assert summary["realized_pnl"] == 20.0  # (52000-50000)*0.01
        assert summary["current_balance"] == 1020.0
        assert summary["return_percent"] == 2.0
        assert summary["closed_trades"] == 1


# =============================================================================
# PERFORMANCE METRICS TESTS
# =============================================================================


class TestPerformanceMetrics:
    """Tests for get_performance_metrics method."""

    def test_metrics_no_trades(self, paper_trader):
        """Test metrics with no closed trades."""
        metrics = paper_trader.get_performance_metrics()
        assert "message" in metrics

    def test_metrics_all_winners(self, paper_trader):
        """Test metrics with all winning trades."""
        # Create and close winning trades
        for i in range(3):
            trade = paper_trader.open_trade(
                symbol="BTC-USD",
                side="BUY",
                quantity=0.01,
                entry_price=50000.0,
            )
            paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        metrics = paper_trader.get_performance_metrics()

        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 3
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 100.0
        assert metrics["profit_factor"] == float("inf")

    def test_metrics_mixed(self, paper_trader):
        """Test metrics with mixed wins and losses."""
        # 2 winners
        for i in range(2):
            trade = paper_trader.open_trade(
                symbol="BTC-USD",
                side="BUY",
                quantity=0.01,
                entry_price=50000.0,
            )
            paper_trader.close_trade(trade.trade_id, exit_price=51000.0)  # +$10

        # 1 loser
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=49000.0)  # -$10

        metrics = paper_trader.get_performance_metrics()

        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == pytest.approx(66.67, rel=0.01)
        assert metrics["gross_profit"] == 20.0
        assert metrics["gross_loss"] == 10.0
        assert metrics["profit_factor"] == 2.0

    def test_metrics_expectancy(self, paper_trader):
        """Test expectancy calculation."""
        # Create predictable trades for expectancy check
        # 2 winners at $10 each, 1 loser at $10
        for i in range(2):
            trade = paper_trader.open_trade(
                symbol="BTC-USD",
                side="BUY",
                quantity=0.01,
                entry_price=50000.0,
            )
            paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=49000.0)

        metrics = paper_trader.get_performance_metrics()

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        # = (0.67 * 10) - (0.33 * 10) = 3.33
        assert metrics["expectancy"] == pytest.approx(3.33, rel=0.1)


class TestTradeHistory:
    """Tests for get_trade_history method."""

    def test_trade_history(self, paper_trader):
        """Test getting trade history."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        history = paper_trader.get_trade_history()

        assert len(history) == 1
        assert history[0]["symbol"] == "BTC-USD"

    def test_trade_history_limit(self, paper_trader):
        """Test trade history respects limit."""
        for i in range(10):
            trade = paper_trader.open_trade(
                symbol="BTC-USD",
                side="BUY",
                quantity=0.001,
                entry_price=50000.0,
            )
            paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        history = paper_trader.get_trade_history(limit=5)

        assert len(history) == 5

    def test_trade_history_includes_open(self, paper_trader):
        """Test trade history includes open trades."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        history = paper_trader.get_trade_history()

        assert len(history) == 1
        assert history[0]["status"] == "OPEN"


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================


class TestPersistence:
    """Tests for save/load state."""

    def test_save_state(self, paper_trader):
        """Test saving state creates file."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        state_file = paper_trader._get_state_file()
        assert state_file.exists()

    def test_load_state(self, temp_data_dir):
        """Test loading state restores trades."""
        # Create trader and make trades
        trader1 = PaperTrader(
            starting_balance=2000.0,
            data_dir=temp_data_dir,
            account_name="load_test",
        )
        trader1.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        # Create new trader instance
        trader2 = PaperTrader(
            data_dir=temp_data_dir,
            account_name="load_test",
        )

        assert len(trader2.account.open_trades) == 1
        assert trader2.account.starting_balance == 2000.0

    def test_persistence_preserves_margin(self, temp_data_dir):
        """Test persistence preserves reserved margin."""
        trader1 = PaperTrader(
            starting_balance=1000.0,
            data_dir=temp_data_dir,
            account_name="margin_test",
        )
        trader1.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            leverage=4.0,
        )
        expected_margin = trader1.account.reserved_margin

        trader2 = PaperTrader(
            data_dir=temp_data_dir,
            account_name="margin_test",
        )

        assert trader2.account.reserved_margin == expected_margin

    def test_state_file_path(self, paper_trader):
        """Test state file path format."""
        path = paper_trader._get_state_file()
        assert "paper_trader_test.json" in str(path)


# =============================================================================
# RESET TESTS
# =============================================================================


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_trades(self, paper_trader):
        """Test reset clears all trades."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        paper_trader.reset()

        assert len(paper_trader.account.open_trades) == 0
        assert len(paper_trader.account.closed_trades) == 0

    def test_reset_restores_balance(self, paper_trader):
        """Test reset restores starting balance."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.close_trade(trade.trade_id, exit_price=49000.0)

        paper_trader.reset()

        assert paper_trader.account.current_balance == 1000.0
        assert paper_trader.account.realized_pnl == 0.0

    def test_reset_clears_margin(self, paper_trader):
        """Test reset clears reserved margin."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        paper_trader.reset()

        assert paper_trader.account.reserved_margin == 0.0

    def test_reset_custom_balance(self, paper_trader):
        """Test reset with new starting balance."""
        paper_trader.reset(starting_balance=5000.0)

        assert paper_trader.account.starting_balance == 5000.0
        assert paper_trader.account.current_balance == 5000.0

    def test_reset_counter(self, paper_trader):
        """Test reset resets trade counter."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )

        paper_trader.reset()

        assert paper_trader._trade_counter == 0


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_positions_same_symbol(self, paper_trader):
        """Test multiple positions in same symbol."""
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.02,
            entry_price=52000.0,
        )

        assert len(paper_trader.account.open_trades) == 2

    def test_close_partial_position(self, paper_trader):
        """Test closing one trade of multiple."""
        trade1 = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.02,
            entry_price=52000.0,
        )

        paper_trader.close_trade(trade1.trade_id, exit_price=53000.0)

        assert len(paper_trader.account.open_trades) == 1
        assert len(paper_trader.account.closed_trades) == 1

    def test_leverage_floor(self, paper_trader):
        """Test leverage < 1 is floored to 1."""
        # With leverage 0.5, margin should be capped at notional
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.001,
            entry_price=50000.0,
            leverage=0.5,  # Would make margin > notional
        )

        # Margin = 50 / max(1, 0.5) = 50 (not 100)
        assert trade.margin_reserved == 50.0

    def test_negative_risk_multiplier(self, paper_trader):
        """Test negative risk multiplier is treated as zero."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            risk_multiplier=-0.5,
        )
        assert trade is None

    def test_from_dict_backward_compat(self):
        """Test from_dict handles missing fields for backward compat."""
        minimal_data = {
            "trade_id": "SIM-001",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "entry_time": "2026-01-15T12:00:00",
        }
        trade = SimulatedTrade.from_dict(minimal_data)

        assert trade.trade_id == "SIM-001"
        assert trade.pattern_type is None
        assert trade.entry_bar_type == ""
        assert trade.priority_rank == 0


# =============================================================================
# STRATEGY ATTRIBUTION TESTS (Session EQUITY-91)
# =============================================================================


class TestStrategyField:
    """Tests for strategy field on SimulatedTrade."""

    def test_default_strategy_is_strat(self):
        """Test default strategy is 'strat'."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
        )
        assert trade.strategy == "strat"

    def test_strategy_serialization(self):
        """Test strategy is serialized in to_dict."""
        trade = SimulatedTrade(
            trade_id="SIM-001",
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            strategy="statarb",
        )
        d = trade.to_dict()
        assert d["strategy"] == "statarb"

    def test_strategy_deserialization(self):
        """Test strategy is deserialized from dict."""
        data = {
            "trade_id": "SIM-001",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "entry_time": "2026-01-15T12:00:00",
            "strategy": "statarb",
        }
        trade = SimulatedTrade.from_dict(data)
        assert trade.strategy == "statarb"

    def test_strategy_deserialization_backward_compat(self):
        """Test missing strategy defaults to 'strat'."""
        data = {
            "trade_id": "SIM-001",
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 0.01,
            "entry_price": 50000.0,
            "entry_time": "2026-01-15T12:00:00",
        }
        trade = SimulatedTrade.from_dict(data)
        assert trade.strategy == "strat"


class TestOpenTradeWithStrategy:
    """Tests for open_trade with strategy parameter."""

    def test_open_trade_default_strategy(self, paper_trader):
        """Test open_trade defaults to 'strat' strategy."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.01,
            entry_price=50000.0,
        )
        assert trade.strategy == "strat"

    def test_open_trade_custom_strategy(self, paper_trader):
        """Test open_trade with custom strategy."""
        trade = paper_trader.open_trade(
            symbol="ADA-USD",
            side="BUY",
            quantity=1000.0,
            entry_price=0.35,
            strategy="statarb",
        )
        assert trade.strategy == "statarb"


class TestGetTradesByStrategy:
    """Tests for get_trades_by_strategy method."""

    def test_get_trades_by_strategy_empty(self, paper_trader):
        """Test get_trades_by_strategy with no trades."""
        trades = paper_trader.get_trades_by_strategy("strat")
        assert trades == []

    def test_get_trades_by_strategy_filters(self, paper_trader):
        """Test get_trades_by_strategy filters correctly."""
        # Create trades for different strategies
        paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.open_trade(
            symbol="ADA-USD", side="BUY", quantity=1000.0,
            entry_price=0.35, strategy="statarb"
        )
        paper_trader.open_trade(
            symbol="XRP-USD", side="SELL", quantity=500.0,
            entry_price=0.50, strategy="statarb"
        )

        strat_trades = paper_trader.get_trades_by_strategy("strat")
        statarb_trades = paper_trader.get_trades_by_strategy("statarb")

        assert len(strat_trades) == 1
        assert len(statarb_trades) == 2
        assert strat_trades[0].symbol == "BTC-USD"

    def test_get_trades_by_strategy_includes_closed(self, paper_trader):
        """Test get_trades_by_strategy includes closed trades."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        trades = paper_trader.get_trades_by_strategy("strat")
        assert len(trades) == 1
        assert trades[0].status == "CLOSED"


class TestGetPnlByStrategy:
    """Tests for get_pnl_by_strategy method."""

    def test_pnl_by_strategy_empty(self, paper_trader):
        """Test get_pnl_by_strategy with no closed trades."""
        pnl = paper_trader.get_pnl_by_strategy()
        assert pnl == {}

    def test_pnl_by_strategy_single_strategy(self, paper_trader):
        """Test get_pnl_by_strategy with single strategy."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        pnl = paper_trader.get_pnl_by_strategy()

        assert "strat" in pnl
        assert "combined" in pnl
        assert pnl["strat"]["trades"] == 1

    def test_pnl_by_strategy_multiple_strategies(self, paper_trader):
        """Test get_pnl_by_strategy with multiple strategies."""
        # STRAT trade
        t1 = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(t1.trade_id, exit_price=51000.0)  # +$10 gross

        # StatArb trade
        t2 = paper_trader.open_trade(
            symbol="ADA-USD", side="BUY", quantity=100.0,
            entry_price=0.35, strategy="statarb"
        )
        paper_trader.close_trade(t2.trade_id, exit_price=0.36)  # +$1 gross

        pnl = paper_trader.get_pnl_by_strategy()

        assert "strat" in pnl
        assert "statarb" in pnl
        assert "combined" in pnl
        assert pnl["strat"]["trades"] == 1
        assert pnl["statarb"]["trades"] == 1
        assert pnl["combined"]["trades"] == 2

    def test_pnl_by_strategy_includes_fees(self, paper_trader):
        """Test get_pnl_by_strategy includes fee breakdown."""
        trade = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(trade.trade_id, exit_price=51000.0)

        pnl = paper_trader.get_pnl_by_strategy()

        # Fees should be negative (cost)
        assert pnl["strat"]["fees"] < 0


class TestGetPerformanceByStrategy:
    """Tests for get_performance_by_strategy method."""

    def test_performance_by_strategy_no_trades(self, paper_trader):
        """Test performance with no trades for strategy."""
        perf = paper_trader.get_performance_by_strategy("strat")
        assert "message" in perf

    def test_performance_by_strategy_basic(self, paper_trader):
        """Test performance metrics for specific strategy."""
        # Create winning trade
        t1 = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(t1.trade_id, exit_price=51000.0)

        # Create losing trade
        t2 = paper_trader.open_trade(
            symbol="ETH-USD", side="BUY", quantity=0.01,
            entry_price=3000.0, strategy="strat"
        )
        paper_trader.close_trade(t2.trade_id, exit_price=2900.0)

        perf = paper_trader.get_performance_by_strategy("strat")

        assert perf["strategy"] == "strat"
        assert perf["total_trades"] == 2
        assert perf["winning_trades"] == 1
        assert perf["losing_trades"] == 1
        assert perf["win_rate"] == 50.0

    def test_performance_by_strategy_isolated(self, paper_trader):
        """Test strategy performance is isolated from other strategies."""
        # STRAT trades (1 winner)
        t1 = paper_trader.open_trade(
            symbol="BTC-USD", side="BUY", quantity=0.01,
            entry_price=50000.0, strategy="strat"
        )
        paper_trader.close_trade(t1.trade_id, exit_price=51000.0)

        # StatArb trades (2 losers)
        for _ in range(2):
            t = paper_trader.open_trade(
                symbol="ADA-USD", side="BUY", quantity=100.0,
                entry_price=0.35, strategy="statarb"
            )
            paper_trader.close_trade(t.trade_id, exit_price=0.34)

        strat_perf = paper_trader.get_performance_by_strategy("strat")
        statarb_perf = paper_trader.get_performance_by_strategy("statarb")

        # STRAT should show 100% win rate (1/1)
        assert strat_perf["win_rate"] == 100.0
        assert strat_perf["total_trades"] == 1

        # StatArb should show 0% win rate (0/2)
        assert statarb_perf["win_rate"] == 0.0
        assert statarb_perf["total_trades"] == 2
