"""
Tests for VirtualBalanceTracker - Session EQUITY-107

Covers:
- Initial state and defaults
- Capital reservation and release (full + partial)
- Settlement lifecycle
- Portfolio heat gating
- Trade budget (fixed_dollar and pct_capital modes)
- Save/load persistence roundtrip
- Corrupt and missing state file recovery
- Alpaca position sync
- Thread safety under concurrent access
- Negative capital safety
- Multi-position tracking
- get_summary completeness
"""

import json
import threading
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from strat.signal_automation.capital_tracker import (
    VirtualBalanceTracker,
    _next_trading_day,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def state_file(tmp_path):
    """Return a temp state file path."""
    return str(tmp_path / 'capital_state.json')


@pytest.fixture
def tracker(state_file):
    """Default tracker with tmp state file."""
    return VirtualBalanceTracker(
        virtual_capital=3000.0,
        sizing_mode='fixed_dollar',
        fixed_dollar_amount=300.0,
        pct_of_capital=0.10,
        max_portfolio_heat=0.08,
        settlement_days=1,
        state_file=state_file,
    )


# Deterministic settlement date for tests
MOCK_SETTLEMENT_DATE = '2099-12-31'
MOCK_PAST_DATE = '2000-01-01'


@pytest.fixture(autouse=True)
def mock_next_trading_day():
    """Mock _next_trading_day to return a far-future date by default."""
    with patch(
        'strat.signal_automation.capital_tracker._next_trading_day',
        return_value=MOCK_SETTLEMENT_DATE,
    ) as m:
        yield m


# =============================================================================
# 1. INITIAL STATE
# =============================================================================


class TestInitialState:
    def test_defaults(self, tracker):
        assert tracker._virtual_capital == 3000.0
        assert tracker._deployed_capital == 0.0
        assert tracker._realized_pnl == 0.0
        assert tracker._positions == {}
        assert tracker._pending_settlements == []

    def test_available_equals_virtual_at_start(self, tracker):
        assert tracker.available_capital == 3000.0


# =============================================================================
# 2. RESERVE CAPITAL
# =============================================================================


class TestReserveCapital:
    def test_reserve_updates_deployed(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)

        assert tracker._deployed_capital == 300.0
        assert tracker._positions == {'OPT_A': 300.0}
        assert tracker.available_capital == 2700.0

    def test_reserve_multiple(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.reserve_capital('OPT_B', 200.0)

        assert tracker._deployed_capital == 500.0
        assert tracker.available_capital == 2500.0


# =============================================================================
# 3. RELEASE CAPITAL (FULL EXIT)
# =============================================================================


class TestReleaseCapital:
    def test_release_with_profit(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.release_capital('OPT_A', proceeds=450.0)

        assert 'OPT_A' not in tracker._positions
        assert tracker._deployed_capital == 0.0
        assert tracker._realized_pnl == 150.0
        # virtual_capital increased by pnl
        assert tracker._virtual_capital == 3150.0

    def test_release_with_loss(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.release_capital('OPT_A', proceeds=100.0)

        assert tracker._realized_pnl == -200.0
        assert tracker._virtual_capital == 2800.0

    def test_release_creates_pending_settlement(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.release_capital('OPT_A', proceeds=300.0)

        assert len(tracker._pending_settlements) == 1
        s = tracker._pending_settlements[0]
        assert s['amount'] == 300.0
        assert s['osi_symbol'] == 'OPT_A'
        assert s['settlement_date'] == MOCK_SETTLEMENT_DATE


# =============================================================================
# 4. RELEASE CAPITAL PARTIAL
# =============================================================================


class TestReleaseCapitalPartial:
    def test_partial_release_half(self, tracker):
        tracker.reserve_capital('OPT_A', 400.0)
        tracker.release_capital_partial('OPT_A', fraction=0.5, proceeds=250.0)

        # Remaining cost basis
        assert tracker._positions['OPT_A'] == 200.0
        # Deployed reduced by partial cost
        assert tracker._deployed_capital == 200.0
        # P&L = 250 - 200 = 50
        assert tracker._realized_pnl == 50.0
        assert tracker._virtual_capital == 3050.0

    def test_partial_then_full_exit(self, tracker):
        tracker.reserve_capital('OPT_A', 400.0)
        # Close half
        tracker.release_capital_partial('OPT_A', fraction=0.5, proceeds=250.0)
        # Close remaining
        tracker.release_capital('OPT_A', proceeds=300.0)

        assert 'OPT_A' not in tracker._positions
        assert tracker._deployed_capital == 0.0
        # Total P&L: (250-200) + (300-200) = 50 + 100 = 150
        assert tracker._realized_pnl == 150.0
        assert tracker._virtual_capital == 3150.0


# =============================================================================
# 5. AVAILABLE CAPITAL EXCLUDES UNSETTLED
# =============================================================================


class TestAvailableCapitalUnsettled:
    def test_unsettled_reduces_available(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.release_capital('OPT_A', proceeds=300.0)

        # Proceeds are unsettled (settlement_date in the future)
        # virtual_capital unchanged (pnl=0), deployed=0, but 300 unsettled
        assert tracker.available_capital == 3000.0 - 300.0


# =============================================================================
# 6. SETTLE PENDING
# =============================================================================


class TestSettlePending:
    def test_settle_clears_past_settlements(self, tracker, mock_next_trading_day):
        # Set up a settlement in the past
        tracker._pending_settlements = [
            {'amount': 300.0, 'settlement_date': MOCK_PAST_DATE, 'osi_symbol': 'OPT_A'},
        ]
        tracker.settle_pending()

        assert len(tracker._pending_settlements) == 0

    def test_settle_keeps_future_settlements(self, tracker):
        tracker._pending_settlements = [
            {'amount': 300.0, 'settlement_date': MOCK_SETTLEMENT_DATE, 'osi_symbol': 'OPT_A'},
        ]
        tracker.settle_pending()

        assert len(tracker._pending_settlements) == 1


# =============================================================================
# 7. CAN_OPEN_TRADE - INSUFFICIENT CAPITAL
# =============================================================================


class TestCanOpenTrade:
    def test_returns_false_when_insufficient(self, tracker):
        assert tracker.can_open_trade(5000.0) is False

    def test_returns_true_when_sufficient(self, tracker):
        # 200 / 3000 = 6.7% heat, under 8% limit
        assert tracker.can_open_trade(200.0) is True

    def test_zero_cost_rejected(self, tracker):
        assert tracker.can_open_trade(0.0) is False

    def test_negative_cost_rejected(self, tracker):
        assert tracker.can_open_trade(-100.0) is False


# =============================================================================
# 8. CAN_OPEN_TRADE - HEAT LIMIT
# =============================================================================


class TestCanOpenTradeHeat:
    def test_heat_limit_blocks_trade(self, tracker):
        # Fill up to near heat limit: 8% of 3000 = 240
        # Add positions totaling 240
        tracker._positions = {'OPT_A': 120.0, 'OPT_B': 120.0}
        tracker._deployed_capital = 240.0

        # Another 100 would push heat to (240+100)/3000 = 11.3% > 8%
        assert tracker.can_open_trade(100.0) is False

    def test_heat_allows_small_trade(self, tracker):
        # One position at 100 -> heat = 100/3000 = 3.3%
        tracker._positions = {'OPT_A': 100.0}
        tracker._deployed_capital = 100.0

        # Adding 50 -> (100+50)/3000 = 5% < 8%
        assert tracker.can_open_trade(50.0) is True


# =============================================================================
# 9. GET TRADE BUDGET - FIXED DOLLAR
# =============================================================================


class TestGetTradeBudgetFixed:
    def test_returns_fixed_amount(self, tracker):
        assert tracker.get_trade_budget() == 300.0

    def test_capped_by_available(self, tracker):
        tracker._virtual_capital = 200.0
        assert tracker.get_trade_budget() == 200.0


# =============================================================================
# 10. GET TRADE BUDGET - PCT CAPITAL
# =============================================================================


class TestGetTradeBudgetPct:
    def test_returns_pct_of_virtual(self, state_file):
        tracker = VirtualBalanceTracker(
            virtual_capital=3000.0,
            sizing_mode='pct_capital',
            pct_of_capital=0.10,
            state_file=state_file,
        )
        # 10% of 3000 = 300
        assert tracker.get_trade_budget() == 300.0

    def test_pct_capped_by_available(self, state_file):
        tracker = VirtualBalanceTracker(
            virtual_capital=500.0,
            sizing_mode='pct_capital',
            pct_of_capital=0.50,
            state_file=state_file,
        )
        tracker._deployed_capital = 400.0
        # 50% of 500 = 250, but available = 100
        assert tracker.get_trade_budget() == 100.0


# =============================================================================
# 11. SAVE / LOAD ROUNDTRIP
# =============================================================================


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, state_file):
        tracker = VirtualBalanceTracker(
            virtual_capital=3000.0, state_file=state_file,
        )
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.release_capital('OPT_A', proceeds=450.0)

        # Load into fresh tracker
        tracker2 = VirtualBalanceTracker(
            virtual_capital=3000.0, state_file=state_file,
        )
        tracker2.load()

        assert tracker2._virtual_capital == tracker._virtual_capital
        assert tracker2._deployed_capital == tracker._deployed_capital
        assert tracker2._realized_pnl == tracker._realized_pnl
        assert tracker2._positions == tracker._positions
        assert len(tracker2._pending_settlements) == len(tracker._pending_settlements)


# =============================================================================
# 12. CORRUPTED STATE FILE
# =============================================================================


class TestCorruptedStateFile:
    def test_corrupt_json_recovers(self, state_file):
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            f.write('NOT VALID JSON {{{{')

        tracker = VirtualBalanceTracker(
            virtual_capital=3000.0, state_file=state_file,
        )
        tracker.load()

        # Should fall back to defaults
        assert tracker._virtual_capital == 3000.0
        assert tracker._deployed_capital == 0.0
        assert tracker._positions == {}


# =============================================================================
# 13. SYNC WITH POSITIONS
# =============================================================================


class TestSyncWithPositions:
    def test_adds_unknown_positions(self, tracker):
        alpaca = [
            {'symbol': 'OPT_X', 'avg_entry_price': '1.50', 'qty': '2'},
        ]
        tracker.sync_with_positions(alpaca)

        # cost = 1.50 * 2 * 100 = 300
        assert 'OPT_X' in tracker._positions
        assert tracker._positions['OPT_X'] == 300.0
        assert tracker._deployed_capital == 300.0

    def test_removes_stale_positions(self, tracker):
        tracker._positions = {'STALE_OPT': 200.0}
        tracker._deployed_capital = 200.0

        tracker.sync_with_positions([])  # Alpaca has nothing

        assert 'STALE_OPT' not in tracker._positions
        assert tracker._deployed_capital == 0.0

    def test_sync_no_change_when_matching(self, tracker):
        tracker._positions = {'OPT_A': 300.0}
        tracker._deployed_capital = 300.0

        alpaca = [
            {'symbol': 'OPT_A', 'avg_entry_price': '1.50', 'qty': '2'},
        ]
        tracker.sync_with_positions(alpaca)

        # OPT_A already existed so cost should stay at tracker value (not overwritten)
        assert tracker._positions['OPT_A'] == 300.0


# =============================================================================
# 14. THREAD SAFETY
# =============================================================================


class TestThreadSafety:
    def test_concurrent_reserve_release(self, state_file):
        tracker = VirtualBalanceTracker(
            virtual_capital=10000.0, state_file=state_file,
        )
        errors = []

        def reserve_and_release(i):
            try:
                sym = f'OPT_{i}'
                tracker.reserve_capital(sym, 100.0)
                tracker.release_capital(sym, 100.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reserve_and_release, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # All positions should be released -> deployed = 0
        assert tracker._deployed_capital == 0.0
        assert len(tracker._positions) == 0


# =============================================================================
# 15. NEGATIVE CAPITAL SAFETY
# =============================================================================


class TestNegativeCapitalSafety:
    def test_cannot_open_when_would_go_negative(self, tracker):
        tracker._deployed_capital = 2900.0
        # Only 100 available
        assert tracker.can_open_trade(200.0) is False

    def test_deployed_floors_at_zero(self, tracker):
        # Release a position that somehow isn't tracked (cost=0)
        tracker.release_capital('UNKNOWN', proceeds=100.0)
        assert tracker._deployed_capital == 0.0


# =============================================================================
# 16. SETTLEMENT ACCOUNTING FULL FLOW
# =============================================================================


class TestSettlementAccounting:
    def test_buy_sell_settle(self, tracker, mock_next_trading_day):
        # Buy
        tracker.reserve_capital('OPT_A', 300.0)
        assert tracker.available_capital == 2700.0

        # Sell
        tracker.release_capital('OPT_A', proceeds=400.0)
        # virtual_capital = 3100, deployed = 0, unsettled = 400
        assert tracker._virtual_capital == 3100.0
        assert tracker.available_capital == 3100.0 - 400.0  # 2700

        # Settle (mock past date)
        mock_next_trading_day.return_value = MOCK_PAST_DATE
        # Replace settlement date with past to simulate next day
        tracker._pending_settlements[0]['settlement_date'] = MOCK_PAST_DATE
        tracker.settle_pending()

        assert len(tracker._pending_settlements) == 0
        assert tracker.available_capital == 3100.0


# =============================================================================
# 17. MULTIPLE POSITIONS
# =============================================================================


class TestMultiplePositions:
    def test_three_positions(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        tracker.reserve_capital('OPT_B', 200.0)
        tracker.reserve_capital('OPT_C', 100.0)

        assert len(tracker._positions) == 3
        assert tracker._deployed_capital == 600.0
        assert tracker.available_capital == 2400.0

        # Close one
        tracker.release_capital('OPT_B', proceeds=250.0)

        assert len(tracker._positions) == 2
        assert tracker._deployed_capital == 400.0
        assert tracker._realized_pnl == 50.0


# =============================================================================
# 18. GET SUMMARY
# =============================================================================


class TestGetSummary:
    def test_all_fields_present(self, tracker):
        tracker.reserve_capital('OPT_A', 300.0)
        summary = tracker.get_summary()

        expected_keys = {
            'virtual_capital', 'deployed_capital', 'available_capital',
            'realized_pnl', 'position_count', 'unsettled_amount', 'positions',
        }
        assert set(summary.keys()) == expected_keys
        assert summary['position_count'] == 1
        assert summary['deployed_capital'] == 300.0
        assert isinstance(summary['positions'], dict)


# =============================================================================
# 19. MISSING STATE FILE
# =============================================================================


class TestMissingStateFile:
    def test_load_returns_defaults(self, tmp_path):
        missing = str(tmp_path / 'nonexistent' / 'state.json')
        tracker = VirtualBalanceTracker(
            virtual_capital=5000.0, state_file=missing,
        )
        result = tracker.load()

        assert result is tracker  # chaining
        assert tracker._virtual_capital == 5000.0
        assert tracker._deployed_capital == 0.0


# =============================================================================
# 20. PARTIAL EXIT THEN FULL EXIT
# =============================================================================


class TestPartialThenFullExit:
    def test_correct_accounting(self, tracker):
        tracker.reserve_capital('OPT_A', 400.0)

        # Partial: close 25% for $120 (cost basis = 100, pnl = +20)
        tracker.release_capital_partial('OPT_A', fraction=0.25, proceeds=120.0)
        assert tracker._positions['OPT_A'] == 300.0
        assert tracker._realized_pnl == 20.0

        # Full exit of remaining for $350 (cost basis = 300, pnl = +50)
        tracker.release_capital('OPT_A', proceeds=350.0)
        assert 'OPT_A' not in tracker._positions
        assert tracker._deployed_capital == 0.0
        assert tracker._realized_pnl == 70.0  # 20 + 50
        # virtual = 3000 + 70 = 3070
        assert tracker._virtual_capital == 3070.0
