"""
Tests for STRAT Paper Trading Infrastructure - Session 83K-41

Tests cover:
1. PaperTrade dataclass creation and field calculation
2. PaperTradeLog CRUD operations and persistence
3. Trade state management (open/close)
4. Summary statistics and breakdowns
5. Backtest comparison framework
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date, timedelta

from strat.paper_trading import (
    PaperTrade,
    PaperTradeLog,
    create_paper_trade,
    TradeDirection,
    ExitReason,
    MarketRegime,
    VixBucket,
    PatternType,
    Timeframe,
    BASELINE_BACKTEST_STATS,
)


class TestPaperTradeDataclass:
    """Tests for PaperTrade dataclass."""

    def test_create_paper_trade_basic(self):
        """Test basic paper trade creation."""
        trade = PaperTrade(
            trade_id="PT_TEST_001",
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
        )

        assert trade.trade_id == "PT_TEST_001"
        assert trade.pattern_type == "3-2U"
        assert trade.timeframe == "1D"
        assert trade.symbol == "SPY"
        assert trade.direction == "CALL"
        assert trade.status == "PENDING"

    def test_create_paper_trade_with_prices(self):
        """Test paper trade with entry/target/stop prices."""
        trade = PaperTrade(
            trade_id="PT_TEST_002",
            pattern_type="2-2D",
            timeframe="1W",
            symbol="QQQ",
            direction="PUT",
            pattern_detected_time=datetime.now(),
            entry_trigger=400.0,
            target_price=392.0,
            stop_price=405.0,
        )

        # Should auto-calculate magnitude
        assert trade.magnitude_pct == pytest.approx(2.0, rel=0.01)
        # Should auto-calculate R:R
        assert trade.risk_reward == pytest.approx(1.6, rel=0.01)

    def test_vix_bucket_calculation(self):
        """Test VIX bucket auto-calculation."""
        # Low VIX
        trade = PaperTrade(
            trade_id="PT_TEST_003",
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            vix_at_entry=12.5,
        )
        assert trade.vix_bucket == VixBucket.LOW.value

        # Medium VIX
        trade2 = PaperTrade(
            trade_id="PT_TEST_004",
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            vix_at_entry=17.0,
        )
        assert trade2.vix_bucket == VixBucket.MEDIUM.value

        # Extreme VIX
        trade3 = PaperTrade(
            trade_id="PT_TEST_005",
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            vix_at_entry=45.0,
        )
        assert trade3.vix_bucket == VixBucket.EXTREME.value

    def test_generate_trade_id(self):
        """Test trade ID generation."""
        trade_id = PaperTrade.generate_trade_id()

        assert trade_id.startswith("PT_")
        assert len(trade_id) > 10  # Has timestamp

    def test_factory_function(self):
        """Test create_paper_trade factory function."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
            vix=15.0,
            atr=8.5,
            market_regime="TREND_BULL",
        )

        assert trade.trade_id.startswith("PT_")
        assert trade.pattern_type == "3-2U"
        assert trade.vix_at_entry == 15.0
        assert trade.atr_14 == 8.5
        assert trade.market_regime == "TREND_BULL"
        assert trade.status == "PENDING"


class TestPaperTradeStateManagement:
    """Tests for trade state management (open/close)."""

    def test_open_trade(self):
        """Test opening a pending trade."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )

        assert trade.status == "PENDING"
        assert not trade.is_open

        # Open the trade
        trade.open_trade(
            entry_time=datetime.now(),
            entry_price=595.50,
            option_price=8.50,
            delta=0.52,
        )

        assert trade.status == "OPEN"
        assert trade.is_open
        assert trade.entry_price == 595.50
        assert trade.option_price_entry == 8.50
        assert trade.delta_at_entry == 0.52

    def test_close_trade_winner(self):
        """Test closing a trade as winner."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now() - timedelta(days=2),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )

        # Open the trade
        trade.open_trade(
            entry_time=datetime.now() - timedelta(days=1),
            entry_price=595.50,
            option_price=8.50,
            delta=0.52,
        )

        # Close the trade as winner
        trade.close_trade(
            exit_time=datetime.now(),
            exit_price=602.00,
            option_price=12.30,
            exit_reason=ExitReason.TARGET_HIT.value,
            delta=0.68,
        )

        assert trade.status == "CLOSED"
        assert trade.is_closed
        assert trade.is_winner
        assert trade.hit_target
        assert trade.pnl_dollars == pytest.approx(380.0, rel=0.01)  # (12.30 - 8.50) * 100
        assert trade.days_held >= 1

    def test_close_trade_loser(self):
        """Test closing a trade as loser."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now() - timedelta(days=2),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )

        # Open the trade
        trade.open_trade(
            entry_time=datetime.now() - timedelta(days=1),
            entry_price=595.50,
            option_price=8.50,
            delta=0.52,
        )

        # Close the trade as loser
        trade.close_trade(
            exit_time=datetime.now(),
            exit_price=590.00,
            option_price=3.50,
            exit_reason=ExitReason.STOP_HIT.value,
            delta=0.35,
        )

        assert trade.status == "CLOSED"
        assert not trade.is_winner
        assert trade.hit_stop
        assert trade.pnl_dollars == pytest.approx(-500.0, rel=0.01)  # (3.50 - 8.50) * 100


class TestPaperTradeLog:
    """Tests for PaperTradeLog persistence and CRUD."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_create_log(self, temp_log_dir):
        """Test creating a new log."""
        log = PaperTradeLog(storage_dir=temp_log_dir)
        assert len(log) == 0

    def test_add_trade(self, temp_log_dir):
        """Test adding a trade."""
        log = PaperTradeLog(storage_dir=temp_log_dir)

        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )

        log.add_trade(trade)
        assert len(log) == 1

    def test_persistence(self, temp_log_dir):
        """Test that trades persist across log instances."""
        # Create log and add trade
        log1 = PaperTradeLog(storage_dir=temp_log_dir)
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )
        log1.add_trade(trade)
        trade_id = trade.trade_id

        # Create new log instance and verify trade exists
        log2 = PaperTradeLog(storage_dir=temp_log_dir)
        assert len(log2) == 1
        assert log2.get_trade(trade_id) is not None

    def test_get_trade(self, temp_log_dir):
        """Test getting a specific trade."""
        log = PaperTradeLog(storage_dir=temp_log_dir)

        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )
        log.add_trade(trade)

        retrieved = log.get_trade(trade.trade_id)
        assert retrieved is not None
        assert retrieved.trade_id == trade.trade_id

        # Test non-existent trade
        assert log.get_trade("NONEXISTENT") is None

    def test_delete_trade(self, temp_log_dir):
        """Test deleting a trade."""
        log = PaperTradeLog(storage_dir=temp_log_dir)

        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime.now(),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )
        log.add_trade(trade)
        assert len(log) == 1

        result = log.delete_trade(trade.trade_id)
        assert result is True
        assert len(log) == 0

    def test_filter_by_pattern(self, temp_log_dir):
        """Test filtering trades by pattern type."""
        log = PaperTradeLog(storage_dir=temp_log_dir)

        # Add different pattern types
        for pattern in ["3-2U", "3-2U", "2-2D", "3-1-2U"]:
            trade = create_paper_trade(
                pattern_type=pattern,
                timeframe="1D",
                symbol="SPY",
                direction="CALL" if "U" in pattern else "PUT",
                pattern_detected_time=datetime.now(),
                entry_trigger=595.0,
                target_price=602.0,
                stop_price=591.0,
            )
            log.add_trade(trade)

        assert len(log.get_trades_by_pattern("3-2U")) == 2
        assert len(log.get_trades_by_pattern("2-2D")) == 1
        assert len(log.get_trades_by_pattern("3-1-2U")) == 1


class TestPaperTradeLogStatistics:
    """Tests for trade log statistics and analysis."""

    @pytest.fixture
    def populated_log(self):
        """Create a log with sample trades."""
        temp_dir = tempfile.mkdtemp()
        log = PaperTradeLog(storage_dir=temp_dir)

        # Add winning trades
        for i in range(3):
            trade = create_paper_trade(
                pattern_type="3-2U",
                timeframe="1D",
                symbol="SPY",
                direction="CALL",
                pattern_detected_time=datetime.now() - timedelta(days=10-i),
                entry_trigger=595.0,
                target_price=602.0,
                stop_price=591.0,
            )
            trade.open_trade(
                entry_time=datetime.now() - timedelta(days=9-i),
                entry_price=595.50,
                option_price=8.50,
                delta=0.52,
            )
            trade.close_trade(
                exit_time=datetime.now() - timedelta(days=8-i),
                exit_price=602.00,
                option_price=12.00,
                exit_reason=ExitReason.TARGET_HIT.value,
            )
            log.add_trade(trade)

        # Add losing trades
        for i in range(2):
            trade = create_paper_trade(
                pattern_type="2-2D",
                timeframe="1W",
                symbol="QQQ",
                direction="PUT",
                pattern_detected_time=datetime.now() - timedelta(days=5-i),
                entry_trigger=400.0,
                target_price=392.0,
                stop_price=405.0,
            )
            trade.open_trade(
                entry_time=datetime.now() - timedelta(days=4-i),
                entry_price=400.0,
                option_price=7.00,
                delta=-0.45,
            )
            trade.close_trade(
                exit_time=datetime.now() - timedelta(days=3-i),
                exit_price=406.00,
                option_price=3.00,
                exit_reason=ExitReason.STOP_HIT.value,
            )
            log.add_trade(trade)

        yield log
        shutil.rmtree(temp_dir)

    def test_summary_stats(self, populated_log):
        """Test summary statistics calculation."""
        stats = populated_log.get_summary_stats()

        assert stats['total_trades'] == 5
        assert stats['winners'] == 3
        assert stats['losers'] == 2
        assert stats['win_rate'] == 60.0  # 3/5
        assert stats['target_hits'] == 3
        assert stats['stop_hits'] == 2

    def test_pattern_breakdown(self, populated_log):
        """Test pattern breakdown."""
        breakdown = populated_log.get_pattern_breakdown()

        assert "3-2U" in breakdown
        assert breakdown["3-2U"]['trades'] == 3
        assert breakdown["3-2U"]['win_rate'] == 100.0

        assert "2-2D" in breakdown
        assert breakdown["2-2D"]['trades'] == 2
        assert breakdown["2-2D"]['win_rate'] == 0.0

    def test_timeframe_breakdown(self, populated_log):
        """Test timeframe breakdown."""
        breakdown = populated_log.get_timeframe_breakdown()

        assert "1D" in breakdown
        assert breakdown["1D"]['trades'] == 3

        assert "1W" in breakdown
        assert breakdown["1W"]['trades'] == 2

    def test_compare_to_backtest(self, populated_log):
        """Test backtest comparison."""
        comparison = populated_log.compare_to_backtest(BASELINE_BACKTEST_STATS)

        assert 'win_rate' in comparison
        assert comparison['win_rate']['backtest'] == BASELINE_BACKTEST_STATS['win_rate']
        assert comparison['win_rate']['paper'] == 60.0
        assert 'difference' in comparison['win_rate']


class TestPaperTradeSerialization:
    """Tests for trade serialization (to/from dict)."""

    def test_to_dict(self):
        """Test converting trade to dictionary."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime(2025, 12, 4, 10, 30),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
            vix=15.0,
        )

        data = trade.to_dict()

        assert data['trade_id'] == trade.trade_id
        assert data['pattern_type'] == "3-2U"
        assert data['symbol'] == "SPY"
        assert '2025-12-04' in data['pattern_detected_time']

    def test_from_dict(self):
        """Test creating trade from dictionary."""
        trade = create_paper_trade(
            pattern_type="3-2U",
            timeframe="1D",
            symbol="SPY",
            direction="CALL",
            pattern_detected_time=datetime(2025, 12, 4, 10, 30),
            entry_trigger=595.0,
            target_price=602.0,
            stop_price=591.0,
        )

        data = trade.to_dict()
        restored = PaperTrade.from_dict(data)

        assert restored.trade_id == trade.trade_id
        assert restored.pattern_type == trade.pattern_type
        assert restored.symbol == trade.symbol


class TestEnumerations:
    """Tests for enum definitions."""

    def test_trade_direction(self):
        """Test trade direction enum."""
        assert TradeDirection.CALL.value == 'CALL'
        assert TradeDirection.PUT.value == 'PUT'

    def test_exit_reason(self):
        """Test exit reason enum."""
        assert ExitReason.TARGET_HIT.value == 'TARGET_HIT'
        assert ExitReason.STOP_HIT.value == 'STOP_HIT'
        assert ExitReason.TIME_EXIT.value == 'TIME_EXIT'

    def test_market_regime(self):
        """Test market regime enum."""
        assert MarketRegime.TREND_BULL.value == 'TREND_BULL'
        assert MarketRegime.CRASH.value == 'CRASH'

    def test_vix_bucket(self):
        """Test VIX bucket enum."""
        assert VixBucket.LOW.value == 1
        assert VixBucket.EXTREME.value == 5

    def test_pattern_type(self):
        """Test pattern type enum."""
        # Session 83K-52: Updated to use full bar sequence notation per CLAUDE.md Section 12
        assert PatternType.PATTERN_22_UP.value == '2D-2U'
        assert PatternType.PATTERN_32_DOWN.value == '3-2D'

    def test_timeframe(self):
        """Test timeframe enum."""
        assert Timeframe.HOURLY.value == '1H'
        assert Timeframe.MONTHLY.value == '1M'
