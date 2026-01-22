"""
Tests for strat/pattern_metrics.py

Covers:
- PatternTradeResult dataclass fields, properties, and methods
- create_trade_from_backtest_row factory function
- create_trades_from_dataframe factory function

Session EQUITY-77: Test coverage for pattern metrics module.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

from strat.pattern_metrics import (
    PatternTradeResult,
    create_trade_from_backtest_row,
    create_trades_from_dataframe,
)
from strat.tier1_detector import PatternType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dates():
    """Sample entry/exit dates for testing."""
    return {
        'entry': datetime(2024, 1, 15, 10, 30),
        'exit': datetime(2024, 1, 18, 14, 30),
    }


@pytest.fixture
def bullish_trade(sample_dates):
    """Create a sample bullish winning trade."""
    return PatternTradeResult(
        trade_id=1,
        symbol='SPY',
        pattern_type='3-1-2U',
        timeframe='1D',
        entry_date=sample_dates['entry'],
        exit_date=sample_dates['exit'],
        entry_price=480.0,
        exit_price=490.0,
        stop_price=475.0,
        target_price=495.0,
        pnl=1000.0,
        pnl_pct=0.0208,
        is_winner=True,
        regime='TREND_BULL',
        hit_target=True,
    )


@pytest.fixture
def bearish_trade(sample_dates):
    """Create a sample bearish losing trade."""
    return PatternTradeResult(
        trade_id=2,
        symbol='QQQ',
        pattern_type='3-1-2D',
        timeframe='1D',
        entry_date=sample_dates['entry'],
        exit_date=sample_dates['exit'],
        entry_price=400.0,
        exit_price=405.0,
        stop_price=405.0,  # Hit stop
        target_price=390.0,
        pnl=-500.0,
        pnl_pct=-0.0125,
        is_winner=False,
        regime='TREND_BEAR',
        hit_stop=True,
    )


@pytest.fixture
def options_trade(sample_dates):
    """Create a sample options trade."""
    return PatternTradeResult(
        trade_id=3,
        symbol='AAPL',
        pattern_type='2U-1-2U',
        timeframe='1H',
        entry_date=sample_dates['entry'],
        exit_date=sample_dates['exit'],
        entry_price=180.0,
        exit_price=185.0,
        stop_price=177.0,
        target_price=188.0,
        pnl=750.0,
        pnl_pct=0.15,
        is_winner=True,
        is_options_trade=True,
        data_source='ThetaData',
        entry_delta=0.55,
        exit_delta=0.65,
        entry_theta=-0.15,
        theta_cost=0.45,
        entry_iv=0.25,
        exit_iv=0.22,
    )


@pytest.fixture
def sample_backtest_row():
    """Sample row from backtest DataFrame."""
    return {
        'symbol': 'SPY',
        'pattern_type': '3-1-2U',
        'timeframe': '1D',
        'regime': 'TREND_BULL',
        'entry_date': datetime(2024, 1, 15),
        'exit_date': datetime(2024, 1, 18),
        'entry_price': 480.0,
        'exit_price': 490.0,
        'stop_price': 475.0,
        'target_price': 495.0,
        'pnl': 1000.0,
        'pnl_pct': 0.0208,
        'days_held': 3,
        'hit_target': True,
        'hit_stop': False,
    }


@pytest.fixture
def sample_backtest_df(sample_backtest_row):
    """Sample backtest DataFrame with multiple trades."""
    rows = [
        sample_backtest_row,
        {
            'symbol': 'QQQ',
            'pattern_type': '2D-1-2D',
            'timeframe': '1D',
            'regime': 'TREND_BEAR',
            'entry_date': datetime(2024, 1, 20),
            'exit_date': datetime(2024, 1, 23),
            'entry_price': 400.0,
            'exit_price': 395.0,
            'stop_price': 405.0,
            'target_price': 390.0,
            'pnl': 500.0,
            'pnl_pct': 0.0125,
            'days_held': 3,
            'hit_target': True,
            'hit_stop': False,
        },
        {
            'symbol': 'AAPL',
            'pattern_type': '2U-2D',
            'timeframe': '1H',
            'entry_date': '2024-01-25T10:30:00',  # String date
            'exit_date': '2024-01-25T15:30:00',
            'entry_price': 185.0,
            'exit_price': 182.0,
            'stop_price': 188.0,
            'target_price': 180.0,
            'pnl': -300.0,
            'pnl_pct': -0.0162,
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# PatternTradeResult Creation Tests
# =============================================================================

class TestPatternTradeResultCreation:
    """Tests for PatternTradeResult creation and initialization."""

    def test_create_with_required_fields(self, sample_dates):
        """Trade can be created with required fields."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
        )
        assert trade.trade_id == 1
        assert trade.symbol == 'SPY'
        assert trade.pattern_type == '3-1-2U'

    def test_default_values(self, sample_dates):
        """Trade has correct default values."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
        )
        assert trade.regime == 'UNKNOWN'
        assert trade.days_held == 3  # Calculated from dates
        assert trade.hit_target is False
        assert trade.hit_stop is False
        assert trade.is_options_trade is False
        assert trade.data_source == 'Synthetic'
        assert trade.continuation_bars == 0
        assert trade.mtf_alignment == 0

    def test_options_trade_fields(self, options_trade):
        """Options trade stores all options-specific fields."""
        assert options_trade.is_options_trade is True
        assert options_trade.data_source == 'ThetaData'
        assert options_trade.entry_delta == 0.55
        assert options_trade.exit_delta == 0.65
        assert options_trade.entry_theta == -0.15
        assert options_trade.theta_cost == 0.45
        assert options_trade.entry_iv == 0.25
        assert options_trade.exit_iv == 0.22

    def test_metadata_dict(self, sample_dates):
        """Trade stores custom metadata."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
            metadata={'tfc_score': 4, 'notes': 'Clean breakout'},
        )
        assert trade.metadata['tfc_score'] == 4
        assert trade.metadata['notes'] == 'Clean breakout'


class TestPatternTradeResultPostInit:
    """Tests for PatternTradeResult __post_init__ calculations."""

    def test_days_held_calculated(self, sample_dates):
        """days_held is calculated from entry/exit dates."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],  # 3 days later
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
        )
        assert trade.days_held == 3

    def test_days_held_minimum_one(self, sample_dates):
        """days_held is at least 1 for same-day trades."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['entry'] + timedelta(hours=2),
            entry_price=480.0,
            exit_price=485.0,
            stop_price=478.0,
            target_price=487.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade.days_held == 1

    def test_days_held_not_overwritten_if_provided(self, sample_dates):
        """days_held is not recalculated if already provided."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
            days_held=5,  # Explicitly provided
        )
        # Should NOT recalculate because days_held != 0
        assert trade.days_held == 5

    def test_is_winner_updated_from_pnl(self, sample_dates):
        """is_winner is updated to match pnl sign."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=False,  # Wrong - should be corrected
        )
        assert trade.is_winner is True  # Corrected from pnl > 0

    def test_is_winner_false_when_pnl_negative(self, sample_dates):
        """is_winner is set to False for negative pnl."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=475.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=-500.0,
            pnl_pct=-0.0104,
            is_winner=True,  # Wrong - should be corrected
        )
        assert trade.is_winner is False  # Corrected from pnl < 0


# =============================================================================
# PatternTradeResult Properties Tests
# =============================================================================

class TestPatternTradeResultPatternProperties:
    """Tests for PatternTradeResult pattern-related properties."""

    def test_pattern_enum_312u(self, bullish_trade):
        """pattern_enum returns correct PatternType for 3-1-2U."""
        assert bullish_trade.pattern_enum == PatternType.PATTERN_312_UP

    def test_pattern_enum_312d(self, bearish_trade):
        """pattern_enum returns correct PatternType for 3-1-2D."""
        assert bearish_trade.pattern_enum == PatternType.PATTERN_312_DOWN

    def test_pattern_enum_2u12u(self, options_trade):
        """pattern_enum returns correct PatternType for 2U-1-2U."""
        assert options_trade.pattern_enum == PatternType.PATTERN_212_2U12U

    def test_is_bullish_true(self, bullish_trade):
        """is_bullish returns True for bullish pattern."""
        assert bullish_trade.is_bullish is True

    def test_is_bullish_false(self, bearish_trade):
        """is_bullish returns False for bearish pattern."""
        assert bearish_trade.is_bullish is False

    def test_is_bearish_true(self, bearish_trade):
        """is_bearish returns True for bearish pattern."""
        assert bearish_trade.is_bearish is True

    def test_is_bearish_false(self, bullish_trade):
        """is_bearish returns False for bullish pattern."""
        assert bullish_trade.is_bearish is False

    def test_base_pattern_312(self, bullish_trade):
        """base_pattern returns '3-1-2' for 3-1-2U."""
        assert bullish_trade.base_pattern == '3-1-2'

    def test_base_pattern_212(self, options_trade):
        """base_pattern returns '2-1-2' for 2U-1-2U."""
        assert options_trade.base_pattern == '2-1-2'


class TestPatternTradeResultRiskRewardProperties:
    """Tests for PatternTradeResult risk/reward properties."""

    def test_risk_amount_bullish(self, bullish_trade):
        """risk_amount returns entry to stop distance for bullish trade."""
        # Bullish: entry=480, stop=475, risk=5
        assert bullish_trade.risk_amount == 5.0

    def test_risk_amount_bearish(self, bearish_trade):
        """risk_amount returns stop to entry distance for bearish trade."""
        # Bearish: entry=400, stop=405, risk=5
        assert bearish_trade.risk_amount == 5.0

    def test_reward_amount_bullish(self, bullish_trade):
        """reward_amount returns target to entry distance for bullish trade."""
        # Bullish: entry=480, target=495, reward=15
        assert bullish_trade.reward_amount == 15.0

    def test_reward_amount_bearish(self, bearish_trade):
        """reward_amount returns entry to target distance for bearish trade."""
        # Bearish: entry=400, target=390, reward=10
        assert bearish_trade.reward_amount == 10.0

    def test_planned_risk_reward_bullish(self, bullish_trade):
        """planned_risk_reward calculates reward/risk ratio."""
        # reward=15, risk=5, R:R=3.0
        assert bullish_trade.planned_risk_reward == 3.0

    def test_planned_risk_reward_bearish(self, bearish_trade):
        """planned_risk_reward calculates reward/risk ratio for bearish."""
        # reward=10, risk=5, R:R=2.0
        assert bearish_trade.planned_risk_reward == 2.0

    def test_planned_risk_reward_zero_risk(self, sample_dates):
        """planned_risk_reward returns 0 when risk is 0."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=485.0,
            stop_price=480.0,  # No risk
            target_price=490.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade.planned_risk_reward == 0.0

    def test_actual_risk_reward_winning_trade(self, bullish_trade):
        """actual_risk_reward calculates realized R:R for winners."""
        # entry=480, exit=490, move=10, risk=5
        # R:R = 10/5 = 2.0
        assert bullish_trade.actual_risk_reward == 2.0

    def test_actual_risk_reward_losing_trade(self, bearish_trade):
        """actual_risk_reward returns 0 for losing trade."""
        assert bearish_trade.pnl < 0
        assert bearish_trade.actual_risk_reward == 0.0

    def test_actual_risk_reward_zero_risk(self, sample_dates):
        """actual_risk_reward returns 0 when risk is 0."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=485.0,
            stop_price=480.0,  # No risk
            target_price=490.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade.actual_risk_reward == 0.0


class TestPatternTradeResultToDict:
    """Tests for PatternTradeResult to_dict method."""

    def test_to_dict_contains_all_required_fields(self, bullish_trade):
        """to_dict contains all required fields."""
        result = bullish_trade.to_dict()

        assert result['trade_id'] == 1
        assert result['symbol'] == 'SPY'
        assert result['pattern_type'] == '3-1-2U'
        assert result['timeframe'] == '1D'
        assert result['entry_price'] == 480.0
        assert result['exit_price'] == 490.0
        assert result['stop_price'] == 475.0
        assert result['target_price'] == 495.0
        assert result['pnl'] == 1000.0
        assert result['pnl_pct'] == 0.0208
        assert result['is_winner'] is True

    def test_to_dict_contains_computed_fields(self, bullish_trade):
        """to_dict includes computed properties."""
        result = bullish_trade.to_dict()

        assert result['base_pattern'] == '3-1-2'
        assert result['planned_risk_reward'] == 3.0

    def test_to_dict_dates_as_isoformat(self, bullish_trade):
        """to_dict converts dates to ISO format strings."""
        result = bullish_trade.to_dict()

        assert isinstance(result['entry_date'], str)
        assert isinstance(result['exit_date'], str)
        assert '2024-01-15' in result['entry_date']
        assert '2024-01-18' in result['exit_date']

    def test_to_dict_options_fields(self, options_trade):
        """to_dict includes options-specific fields."""
        result = options_trade.to_dict()

        assert result['is_options_trade'] is True
        assert result['data_source'] == 'ThetaData'
        assert result['entry_delta'] == 0.55
        assert result['exit_delta'] == 0.65
        assert result['entry_theta'] == -0.15
        assert result['theta_cost'] == 0.45
        assert result['entry_iv'] == 0.25
        assert result['exit_iv'] == 0.22

    def test_to_dict_metadata_included(self, sample_dates):
        """to_dict includes metadata dictionary."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
            metadata={'custom_field': 'value'},
        )
        result = trade.to_dict()

        assert result['metadata'] == {'custom_field': 'value'}


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateTradeFromBacktestRow:
    """Tests for create_trade_from_backtest_row factory function."""

    def test_creates_trade_from_dict(self, sample_backtest_row):
        """Factory creates PatternTradeResult from dict."""
        trade = create_trade_from_backtest_row(sample_backtest_row, trade_id=1)

        assert isinstance(trade, PatternTradeResult)
        assert trade.trade_id == 1
        assert trade.symbol == 'SPY'
        assert trade.pattern_type == '3-1-2U'

    def test_uses_provided_trade_id(self, sample_backtest_row):
        """Factory uses provided trade_id."""
        trade = create_trade_from_backtest_row(sample_backtest_row, trade_id=42)
        assert trade.trade_id == 42

    def test_extracts_all_fields(self, sample_backtest_row):
        """Factory extracts all fields from row."""
        trade = create_trade_from_backtest_row(sample_backtest_row, trade_id=1)

        assert trade.timeframe == '1D'
        assert trade.regime == 'TREND_BULL'
        assert trade.entry_price == 480.0
        assert trade.exit_price == 490.0
        assert trade.stop_price == 475.0
        assert trade.target_price == 495.0
        assert trade.pnl == 1000.0
        assert trade.days_held == 3
        assert trade.hit_target is True
        assert trade.hit_stop is False

    def test_calculates_pnl_from_prices(self):
        """Factory calculates pnl from prices if not provided."""
        row = {
            'symbol': 'SPY',
            'pattern_type': '3-1-2U',
            'entry_date': datetime(2024, 1, 15),
            'exit_date': datetime(2024, 1, 18),
            'entry_price': 480.0,
            'exit_price': 490.0,
            'stop_price': 475.0,
            'target_price': 495.0,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        # pnl = exit - entry = 490 - 480 = 10
        assert trade.pnl == 10.0

    def test_calculates_pnl_pct_from_prices(self):
        """Factory calculates pnl_pct from prices if not provided."""
        row = {
            'symbol': 'SPY',
            'pattern_type': '3-1-2U',
            'entry_date': datetime(2024, 1, 15),
            'exit_date': datetime(2024, 1, 18),
            'entry_price': 480.0,
            'exit_price': 490.0,
            'stop_price': 475.0,
            'target_price': 495.0,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        # pnl_pct = 10 / 480 = 0.0208...
        assert abs(trade.pnl_pct - 0.0208) < 0.001

    def test_parses_string_dates(self):
        """Factory parses ISO format string dates."""
        row = {
            'symbol': 'SPY',
            'pattern_type': '3-1-2U',
            'entry_date': '2024-01-15T10:30:00',
            'exit_date': '2024-01-18T14:30:00',
            'entry_price': 480.0,
            'exit_price': 490.0,
            'stop_price': 475.0,
            'target_price': 495.0,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        assert trade.entry_date == datetime(2024, 1, 15, 10, 30)
        assert trade.exit_date == datetime(2024, 1, 18, 14, 30)

    def test_is_options_flag(self, sample_backtest_row):
        """Factory sets is_options_trade when is_options=True."""
        trade = create_trade_from_backtest_row(
            sample_backtest_row, trade_id=1, is_options=True
        )

        assert trade.is_options_trade is True
        assert trade.data_source == 'ThetaData'

    def test_defaults_for_missing_fields(self):
        """Factory provides defaults for missing fields."""
        row = {
            'entry_date': datetime(2024, 1, 15),
            'exit_date': datetime(2024, 1, 18),
            'entry_price': 480.0,
            'exit_price': 490.0,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        assert trade.symbol == 'UNKNOWN'
        assert trade.pattern_type == 'UNKNOWN'
        assert trade.timeframe == '1D'
        assert trade.regime == 'UNKNOWN'
        assert trade.stop_price == 0
        assert trade.target_price == 0

    def test_extracts_options_greeks(self):
        """Factory extracts options Greek fields."""
        row = {
            'symbol': 'AAPL',
            'pattern_type': '2U-1-2U',
            'entry_date': datetime(2024, 1, 15),
            'exit_date': datetime(2024, 1, 18),
            'entry_price': 180.0,
            'exit_price': 185.0,
            'stop_price': 177.0,
            'target_price': 188.0,
            'entry_delta': 0.55,
            'exit_delta': 0.65,
            'entry_theta': -0.15,
            'theta_cost': 0.45,
            'entry_iv': 0.25,
            'exit_iv': 0.22,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1, is_options=True)

        assert trade.entry_delta == 0.55
        assert trade.exit_delta == 0.65
        assert trade.entry_theta == -0.15
        assert trade.theta_cost == 0.45
        assert trade.entry_iv == 0.25
        assert trade.exit_iv == 0.22


class TestCreateTradesFromDataFrame:
    """Tests for create_trades_from_dataframe factory function."""

    def test_creates_list_of_trades(self, sample_backtest_df):
        """Factory creates list of PatternTradeResult from DataFrame."""
        trades = create_trades_from_dataframe(sample_backtest_df)

        assert isinstance(trades, list)
        assert len(trades) == 3
        assert all(isinstance(t, PatternTradeResult) for t in trades)

    def test_assigns_sequential_trade_ids(self, sample_backtest_df):
        """Factory assigns sequential trade_ids starting from 1."""
        trades = create_trades_from_dataframe(sample_backtest_df)

        assert trades[0].trade_id == 1
        assert trades[1].trade_id == 2
        assert trades[2].trade_id == 3

    def test_extracts_all_rows(self, sample_backtest_df):
        """Factory extracts data from all rows."""
        trades = create_trades_from_dataframe(sample_backtest_df)

        assert trades[0].symbol == 'SPY'
        assert trades[1].symbol == 'QQQ'
        assert trades[2].symbol == 'AAPL'

    def test_is_options_flag_applied_to_all(self, sample_backtest_df):
        """Factory applies is_options flag to all trades."""
        trades = create_trades_from_dataframe(sample_backtest_df, is_options=True)

        assert all(t.is_options_trade for t in trades)
        assert all(t.data_source == 'ThetaData' for t in trades)

    def test_empty_dataframe_returns_empty_list(self):
        """Factory returns empty list for empty DataFrame."""
        empty_df = pd.DataFrame(columns=['symbol', 'pattern_type', 'entry_price'])
        trades = create_trades_from_dataframe(empty_df)

        assert trades == []

    def test_parses_string_dates_in_df(self, sample_backtest_df):
        """Factory parses string dates in DataFrame."""
        trades = create_trades_from_dataframe(sample_backtest_df)

        # Third row has string dates
        assert trades[2].entry_date == datetime(2024, 1, 25, 10, 30)
        assert trades[2].exit_date == datetime(2024, 1, 25, 15, 30)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestPatternTradeResultEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_unknown_pattern_type(self, sample_dates):
        """Trade handles unknown pattern type gracefully."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='UNKNOWN-PATTERN',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
        )
        # Should not raise, just return UNKNOWN enum
        assert trade.pattern_enum == PatternType.UNKNOWN
        assert trade.base_pattern == 'UNKNOWN'

    def test_zero_pnl_trade(self, sample_dates):
        """Trade handles zero P/L correctly."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=480.0,
            stop_price=475.0,
            target_price=490.0,
            pnl=0.0,
            pnl_pct=0.0,
            is_winner=False,
        )
        # Zero pnl doesn't change is_winner (stays as provided)
        assert trade.is_winner is False
        assert trade.pnl == 0.0

    def test_very_large_pnl(self, sample_dates):
        """Trade handles very large P/L values."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=100.0,
            exit_price=200.0,
            stop_price=90.0,
            target_price=200.0,
            pnl=10_000_000.0,
            pnl_pct=1.0,  # 100% gain
            is_winner=True,
        )
        assert trade.pnl == 10_000_000.0
        assert trade.is_winner is True

    def test_22_pattern_types(self, sample_dates):
        """Trade handles 2-2 reversal patterns correctly."""
        # 2D-2U (bullish reversal)
        trade1 = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='2D-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=488.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade1.is_bullish is True
        assert trade1.is_bearish is False
        assert trade1.base_pattern == '2-2'

        # 2U-2D (bearish reversal)
        trade2 = PatternTradeResult(
            trade_id=2,
            symbol='SPY',
            pattern_type='2U-2D',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=475.0,
            stop_price=483.0,
            target_price=472.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade2.is_bullish is False
        assert trade2.is_bearish is True
        assert trade2.base_pattern == '2-2'

    def test_32_pattern_types(self, sample_dates):
        """Trade handles 3-2 patterns correctly."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=487.0,
            stop_price=477.0,
            target_price=487.2,  # 1.5% target
            pnl=720.0,
            pnl_pct=0.015,
            is_winner=True,
        )
        assert trade.is_bullish is True
        assert trade.base_pattern == '3-2'

    def test_legacy_212_patterns(self, sample_dates):
        """Trade handles legacy 2-1-2U/2-1-2D patterns."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='2-1-2U',  # Legacy format
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=488.0,
            pnl=500.0,
            pnl_pct=0.0104,
            is_winner=True,
        )
        assert trade.is_bullish is True
        assert trade.base_pattern == '2-1-2'

    def test_all_212_variants(self, sample_dates):
        """Trade handles all 4 variants of 2-1-2 patterns."""
        patterns = ['2U-1-2U', '2D-1-2D', '2D-1-2U', '2U-1-2D']
        is_bullish = [True, False, True, False]

        for pattern, expected_bullish in zip(patterns, is_bullish):
            trade = PatternTradeResult(
                trade_id=1,
                symbol='SPY',
                pattern_type=pattern,
                timeframe='1D',
                entry_date=sample_dates['entry'],
                exit_date=sample_dates['exit'],
                entry_price=480.0,
                exit_price=485.0 if expected_bullish else 475.0,
                stop_price=475.0 if expected_bullish else 485.0,
                target_price=490.0 if expected_bullish else 470.0,
                pnl=500.0,
                pnl_pct=0.0104,
                is_winner=True,
            )
            assert trade.is_bullish == expected_bullish, f"Pattern {pattern} bullish check failed"
            assert trade.is_bearish == (not expected_bullish), f"Pattern {pattern} bearish check failed"
            assert trade.base_pattern == '2-1-2', f"Pattern {pattern} base_pattern failed"

    def test_to_dict_none_values(self, sample_dates):
        """to_dict handles None values for optional fields."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=sample_dates['entry'],
            exit_date=sample_dates['exit'],
            entry_price=480.0,
            exit_price=490.0,
            stop_price=475.0,
            target_price=495.0,
            pnl=1000.0,
            pnl_pct=0.0208,
            is_winner=True,
            entry_delta=None,
            exit_delta=None,
        )
        result = trade.to_dict()

        assert result['entry_delta'] is None
        assert result['exit_delta'] is None
