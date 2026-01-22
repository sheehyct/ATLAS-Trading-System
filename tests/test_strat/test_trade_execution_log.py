"""
Tests for strat/trade_execution_log.py

Covers:
- ExitReason enum values and string conversion
- TradeExecutionRecord dataclass fields, properties, and methods
- TradeExecutionLog collection class methods and analysis functions

Session EQUITY-77: Test coverage for trade execution logging module.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from strat.trade_execution_log import (
    ExitReason,
    TradeExecutionRecord,
    TradeExecutionLog,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_timestamps():
    """Sample timestamps for testing."""
    return {
        'pattern': datetime(2024, 1, 15, 10, 30),
        'entry': datetime(2024, 1, 15, 11, 0),
        'exit': datetime(2024, 1, 18, 14, 30),
    }


@pytest.fixture
def sample_record(sample_timestamps):
    """Create a sample TradeExecutionRecord."""
    return TradeExecutionRecord(
        trade_id=1,
        symbol='SPY',
        pattern_type='3-1-2U',
        timeframe='1H',
        pattern_timestamp=sample_timestamps['pattern'],
        entry_timestamp=sample_timestamps['entry'],
        exit_timestamp=sample_timestamps['exit'],
        exit_reason='TARGET',
        entry_price=480.0,
        exit_price=485.0,
        stop_price=477.0,
        target_price=485.0,
        strike=480.0,
        option_type='CALL',
        osi_symbol='SPY240119C00480000',
        option_entry_price=5.50,
        option_exit_price=8.20,
        pnl=270.0,
        pnl_pct=0.049,
    )


@pytest.fixture
def sample_losing_record(sample_timestamps):
    """Create a sample losing trade record."""
    return TradeExecutionRecord(
        trade_id=2,
        symbol='QQQ',
        pattern_type='2-1-2D',
        timeframe='1D',
        pattern_timestamp=sample_timestamps['pattern'],
        entry_timestamp=sample_timestamps['entry'],
        exit_timestamp=sample_timestamps['exit'],
        exit_reason='STOP',
        entry_price=400.0,
        exit_price=395.0,
        stop_price=395.0,
        target_price=410.0,
        strike=400.0,
        option_type='PUT',
        option_entry_price=4.00,
        option_exit_price=1.50,
        pnl=-250.0,
        pnl_pct=-0.0625,
    )


@pytest.fixture
def sample_rejected_record(sample_timestamps):
    """Create a sample rejected trade record."""
    return TradeExecutionRecord(
        trade_id=3,
        symbol='AAPL',
        pattern_type='3-2U',
        timeframe='1H',
        pattern_timestamp=sample_timestamps['pattern'],
        entry_timestamp=sample_timestamps['entry'],
        exit_timestamp=sample_timestamps['entry'],  # Same time - rejected before entry
        exit_reason='REJECTED',
        entry_price=180.0,
        exit_price=180.0,
        stop_price=178.0,
        target_price=183.0,
        validation_passed=False,
        validation_reason='TFC below threshold',
    )


@pytest.fixture
def populated_log(sample_record, sample_losing_record, sample_rejected_record):
    """Create a TradeExecutionLog with multiple records."""
    log = TradeExecutionLog()
    log.add_records([sample_record, sample_losing_record, sample_rejected_record])
    return log


# =============================================================================
# ExitReason Enum Tests
# =============================================================================

class TestExitReasonEnum:
    """Tests for ExitReason enumeration."""

    def test_exit_reason_target_value(self):
        """ExitReason.TARGET has correct value."""
        assert ExitReason.TARGET.value == 'TARGET'

    def test_exit_reason_stop_value(self):
        """ExitReason.STOP has correct value."""
        assert ExitReason.STOP.value == 'STOP'

    def test_exit_reason_expiration_value(self):
        """ExitReason.EXPIRATION has correct value."""
        assert ExitReason.EXPIRATION.value == 'EXPIRATION'

    def test_exit_reason_time_exit_value(self):
        """ExitReason.TIME_EXIT has correct value."""
        assert ExitReason.TIME_EXIT.value == 'TIME_EXIT'

    def test_exit_reason_rejected_value(self):
        """ExitReason.REJECTED has correct value."""
        assert ExitReason.REJECTED.value == 'REJECTED'

    def test_exit_reason_manual_value(self):
        """ExitReason.MANUAL has correct value."""
        assert ExitReason.MANUAL.value == 'MANUAL'

    def test_exit_reason_unknown_value(self):
        """ExitReason.UNKNOWN has correct value."""
        assert ExitReason.UNKNOWN.value == 'UNKNOWN'

    def test_exit_reason_is_str_enum(self):
        """ExitReason values can be used as strings."""
        assert ExitReason.TARGET == 'TARGET'
        assert ExitReason.STOP == 'STOP'

    def test_exit_reason_all_values_exist(self):
        """All expected exit reasons exist."""
        expected = {'TARGET', 'STOP', 'EXPIRATION', 'TIME_EXIT', 'REJECTED', 'MANUAL', 'UNKNOWN'}
        actual = {e.value for e in ExitReason}
        assert actual == expected


# =============================================================================
# TradeExecutionRecord Dataclass Tests
# =============================================================================

class TestTradeExecutionRecordCreation:
    """Tests for TradeExecutionRecord creation and initialization."""

    def test_record_creation_with_required_fields(self, sample_timestamps):
        """Record can be created with required fields only."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.trade_id == 1
        assert record.symbol == 'SPY'
        assert record.pattern_type == '3-1-2U'

    def test_record_default_values(self, sample_timestamps):
        """Record has correct default values."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.strike == 0.0
        assert record.option_type == ''
        assert record.osi_symbol == ''
        assert record.pnl == 0.0
        assert record.validation_passed is True
        assert record.circuit_state == 'NORMAL'
        assert record.data_source == 'BlackScholes'
        assert record.direction == 1

    def test_record_with_all_optional_fields(self, sample_record):
        """Record stores all optional fields correctly."""
        assert sample_record.strike == 480.0
        assert sample_record.option_type == 'CALL'
        assert sample_record.osi_symbol == 'SPY240119C00480000'
        assert sample_record.option_entry_price == 5.50
        assert sample_record.option_exit_price == 8.20
        assert sample_record.pnl == 270.0

    def test_record_code_version_tracking(self, sample_timestamps):
        """Record stores code version tracking fields."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
            code_version='abc123',
            code_session='EQUITY-77',
            code_branch='main',
        )
        assert record.code_version == 'abc123'
        assert record.code_session == 'EQUITY-77'
        assert record.code_branch == 'main'

    def test_record_metadata_dict(self, sample_timestamps):
        """Record stores custom metadata."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
            metadata={'tfc_score': 4, 'magnitude': 0.015},
        )
        assert record.metadata['tfc_score'] == 4
        assert record.metadata['magnitude'] == 0.015


class TestTradeExecutionRecordPostInit:
    """Tests for TradeExecutionRecord __post_init__ calculations."""

    def test_days_held_calculated_from_timestamps(self, sample_timestamps):
        """days_held is calculated from entry/exit timestamps."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],  # 3 days later
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.days_held == 3

    def test_days_held_minimum_one(self, sample_timestamps):
        """days_held is at least 1 for same-day trades."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['entry'] + timedelta(hours=2),
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.days_held == 1

    def test_days_held_not_overwritten_if_provided(self, sample_timestamps):
        """days_held is not recalculated if already provided."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
            days_held=5,  # Explicitly provided
        )
        # Should NOT recalculate because days_held != 0
        assert record.days_held == 5

    def test_pnl_pct_calculated_from_pnl(self, sample_timestamps):
        """pnl_pct is calculated from pnl and option_entry_price."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
            pnl=100.0,
            option_entry_price=5.0,  # 100 shares * 5.0 = $500 cost
        )
        # pnl_pct = pnl / (option_entry_price * 100) = 100 / 500 = 0.2
        assert record.pnl_pct == 0.2


class TestTradeExecutionRecordProperties:
    """Tests for TradeExecutionRecord property accessors."""

    def test_is_winner_true_for_positive_pnl(self, sample_record):
        """is_winner returns True for positive P/L."""
        assert sample_record.pnl > 0
        assert sample_record.is_winner is True

    def test_is_winner_false_for_negative_pnl(self, sample_losing_record):
        """is_winner returns False for negative P/L."""
        assert sample_losing_record.pnl < 0
        assert sample_losing_record.is_winner is False

    def test_is_winner_false_for_zero_pnl(self, sample_timestamps):
        """is_winner returns False for zero P/L."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TIME_EXIT',
            entry_price=480.0,
            exit_price=480.0,
            stop_price=477.0,
            target_price=485.0,
            pnl=0.0,
        )
        assert record.is_winner is False

    def test_hit_target_true(self, sample_record):
        """hit_target returns True for TARGET exit."""
        assert sample_record.exit_reason == 'TARGET'
        assert sample_record.hit_target is True

    def test_hit_target_false(self, sample_losing_record):
        """hit_target returns False for non-TARGET exit."""
        assert sample_losing_record.exit_reason == 'STOP'
        assert sample_losing_record.hit_target is False

    def test_hit_stop_true(self, sample_losing_record):
        """hit_stop returns True for STOP exit."""
        assert sample_losing_record.exit_reason == 'STOP'
        assert sample_losing_record.hit_stop is True

    def test_hit_stop_false(self, sample_record):
        """hit_stop returns False for non-STOP exit."""
        assert sample_record.exit_reason == 'TARGET'
        assert sample_record.hit_stop is False

    def test_was_rejected_true(self, sample_rejected_record):
        """was_rejected returns True for REJECTED exit."""
        assert sample_rejected_record.exit_reason == 'REJECTED'
        assert sample_rejected_record.was_rejected is True

    def test_was_rejected_false(self, sample_record):
        """was_rejected returns False for non-REJECTED exit."""
        assert sample_record.exit_reason == 'TARGET'
        assert sample_record.was_rejected is False

    def test_time_to_entry_hours(self, sample_record):
        """time_to_entry returns hours from pattern to entry."""
        # Pattern at 10:30, entry at 11:00 = 0.5 hours
        assert sample_record.time_to_entry == 0.5

    def test_time_to_entry_none_if_missing_timestamps(self, sample_timestamps):
        """time_to_entry returns None if timestamps missing."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=None,
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.time_to_entry is None

    def test_time_in_trade_hours(self, sample_record, sample_timestamps):
        """time_in_trade returns hours from entry to exit."""
        # Entry Jan 15 11:00, Exit Jan 18 14:30 = 3 days + 3.5 hours
        expected_hours = 3 * 24 + 3.5  # 75.5 hours
        assert sample_record.time_in_trade == expected_hours

    def test_time_in_trade_none_if_missing_timestamps(self, sample_timestamps):
        """time_in_trade returns None if timestamps missing."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=None,
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
        )
        assert record.time_in_trade is None


class TestTradeExecutionRecordMethods:
    """Tests for TradeExecutionRecord methods."""

    def test_to_dict_contains_all_fields(self, sample_record):
        """to_dict returns dictionary with all fields."""
        result = sample_record.to_dict()

        assert result['trade_id'] == 1
        assert result['symbol'] == 'SPY'
        assert result['pattern_type'] == '3-1-2U'
        assert result['timeframe'] == '1H'
        assert result['exit_reason'] == 'TARGET'
        assert result['entry_price'] == 480.0
        assert result['exit_price'] == 485.0
        assert result['stop_price'] == 477.0
        assert result['target_price'] == 485.0
        assert result['strike'] == 480.0
        assert result['option_type'] == 'CALL'
        assert result['pnl'] == 270.0
        assert result['validation_passed'] is True
        assert result['data_source'] == 'BlackScholes'

    def test_to_dict_includes_timestamps(self, sample_record, sample_timestamps):
        """to_dict includes all timestamp fields."""
        result = sample_record.to_dict()

        assert result['pattern_timestamp'] == sample_timestamps['pattern']
        assert result['entry_timestamp'] == sample_timestamps['entry']
        assert result['exit_timestamp'] == sample_timestamps['exit']

    def test_to_dict_includes_code_tracking(self, sample_timestamps):
        """to_dict includes code version tracking fields."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=485.0,
            stop_price=477.0,
            target_price=485.0,
            code_version='abc123',
            code_session='EQUITY-77',
            code_branch='main',
        )
        result = record.to_dict()

        assert result['code_version'] == 'abc123'
        assert result['code_session'] == 'EQUITY-77'
        assert result['code_branch'] == 'main'

    def test_to_backtest_row_format(self, sample_record, sample_timestamps):
        """to_backtest_row returns BacktestResult-compatible format."""
        result = sample_record.to_backtest_row()

        # Required BacktestResult columns
        assert result['pnl'] == 270.0
        assert result['pnl_pct'] == 0.049
        assert result['entry_date'] == sample_timestamps['entry']
        assert result['exit_date'] == sample_timestamps['exit']
        assert result['days_held'] == 3

        # Pattern columns
        assert result['pattern_type'] == '3-1-2U'
        assert result['exit_type'] == 'TARGET'
        assert result['symbol'] == 'SPY'
        assert result['direction'] == 1

    def test_to_backtest_row_extended_columns(self, sample_record):
        """to_backtest_row includes extended Session 83K columns."""
        result = sample_record.to_backtest_row()

        assert 'pattern_timestamp' in result
        assert result['entry_price'] == 480.0
        assert result['exit_price'] == 485.0
        assert result['stop_price'] == 477.0
        assert result['target_price'] == 485.0
        assert result['strike'] == 480.0
        assert result['option_type'] == 'CALL'


# =============================================================================
# TradeExecutionLog Collection Tests
# =============================================================================

class TestTradeExecutionLogBasics:
    """Tests for TradeExecutionLog basic operations."""

    def test_empty_log_creation(self):
        """Empty log can be created."""
        log = TradeExecutionLog()
        assert len(log) == 0
        assert log.records == []

    def test_add_single_record(self, sample_record):
        """Single record can be added."""
        log = TradeExecutionLog()
        log.add_record(sample_record)
        assert len(log) == 1
        assert log.records[0] == sample_record

    def test_add_multiple_records(self, sample_record, sample_losing_record):
        """Multiple records can be added at once."""
        log = TradeExecutionLog()
        log.add_records([sample_record, sample_losing_record])
        assert len(log) == 2

    def test_len_returns_record_count(self, populated_log):
        """__len__ returns number of records."""
        assert len(populated_log) == 3

    def test_iter_yields_records(self, populated_log):
        """__iter__ yields all records."""
        records = list(populated_log)
        assert len(records) == 3


class TestTradeExecutionLogDataFrame:
    """Tests for TradeExecutionLog DataFrame conversion."""

    def test_to_dataframe_empty_log(self):
        """to_dataframe returns empty DataFrame with correct columns for empty log."""
        log = TradeExecutionLog()
        df = log.to_dataframe()

        assert len(df) == 0
        assert 'pnl' in df.columns
        assert 'entry_date' in df.columns
        assert 'pattern_type' in df.columns

    def test_to_dataframe_with_records(self, populated_log):
        """to_dataframe converts records to DataFrame."""
        df = populated_log.to_dataframe()

        assert len(df) == 3
        assert df['symbol'].tolist() == ['SPY', 'QQQ', 'AAPL']
        assert df['pattern_type'].tolist() == ['3-1-2U', '2-1-2D', '3-2U']

    def test_to_dataframe_backtest_format(self, populated_log):
        """to_dataframe returns BacktestResult-compatible format."""
        df = populated_log.to_dataframe()

        # Required columns for BacktestResult.trades
        required_cols = ['pnl', 'pnl_pct', 'entry_date', 'exit_date', 'days_held']
        for col in required_cols:
            assert col in df.columns

    def test_to_full_dataframe_empty_log(self):
        """to_full_dataframe returns empty DataFrame for empty log."""
        log = TradeExecutionLog()
        df = log.to_full_dataframe()
        assert len(df) == 0

    def test_to_full_dataframe_with_records(self, populated_log):
        """to_full_dataframe includes all fields."""
        df = populated_log.to_full_dataframe()

        assert len(df) == 3
        # Full DataFrame should have more columns than to_dataframe
        assert 'osi_symbol' in df.columns
        assert 'entry_delta' in df.columns
        assert 'circuit_state' in df.columns
        assert 'code_version' in df.columns


class TestTradeExecutionLogBreakdowns:
    """Tests for TradeExecutionLog breakdown analysis methods."""

    def test_get_exit_reason_breakdown(self, populated_log):
        """get_exit_reason_breakdown counts by exit reason."""
        breakdown = populated_log.get_exit_reason_breakdown()

        assert breakdown['TARGET'] == 1
        assert breakdown['STOP'] == 1
        assert breakdown['REJECTED'] == 1

    def test_get_exit_reason_breakdown_empty(self):
        """get_exit_reason_breakdown returns empty dict for empty log."""
        log = TradeExecutionLog()
        breakdown = log.get_exit_reason_breakdown()
        assert breakdown == {}

    def test_get_pattern_breakdown(self, populated_log):
        """get_pattern_breakdown counts by pattern type."""
        breakdown = populated_log.get_pattern_breakdown()

        assert breakdown['3-1-2U'] == 1
        assert breakdown['2-1-2D'] == 1
        assert breakdown['3-2U'] == 1

    def test_get_pattern_breakdown_with_duplicates(self, sample_record):
        """get_pattern_breakdown handles duplicate patterns."""
        log = TradeExecutionLog()
        # Add same pattern twice
        log.add_record(sample_record)
        record2 = TradeExecutionRecord(
            trade_id=2,
            symbol='QQQ',
            pattern_type='3-1-2U',  # Same pattern
            timeframe='1H',
            pattern_timestamp=datetime.now(),
            entry_timestamp=datetime.now(),
            exit_timestamp=datetime.now(),
            exit_reason='STOP',
            entry_price=400.0,
            exit_price=395.0,
            stop_price=395.0,
            target_price=410.0,
        )
        log.add_record(record2)

        breakdown = log.get_pattern_breakdown()
        assert breakdown['3-1-2U'] == 2

    def test_get_symbol_breakdown(self, populated_log):
        """get_symbol_breakdown counts by symbol."""
        breakdown = populated_log.get_symbol_breakdown()

        assert breakdown['SPY'] == 1
        assert breakdown['QQQ'] == 1
        assert breakdown['AAPL'] == 1

    def test_get_data_source_breakdown(self, populated_log):
        """get_data_source_breakdown counts by data source."""
        breakdown = populated_log.get_data_source_breakdown()

        # All sample records use default 'BlackScholes'
        assert breakdown['BlackScholes'] == 3


class TestTradeExecutionLogTimestampAnalysis:
    """Tests for TradeExecutionLog timestamp analysis."""

    def test_get_timestamp_analysis_empty(self):
        """get_timestamp_analysis returns empty dict for empty log."""
        log = TradeExecutionLog()
        analysis = log.get_timestamp_analysis()
        assert analysis == {}

    def test_get_timestamp_analysis_fields(self, populated_log):
        """get_timestamp_analysis returns expected fields."""
        analysis = populated_log.get_timestamp_analysis()

        assert 'avg_time_to_entry_hours' in analysis
        assert 'avg_time_in_trade_hours' in analysis
        assert 'min_time_to_entry_hours' in analysis
        assert 'max_time_to_entry_hours' in analysis
        assert 'min_time_in_trade_hours' in analysis
        assert 'max_time_in_trade_hours' in analysis

    def test_get_timestamp_analysis_values(self, sample_record):
        """get_timestamp_analysis calculates correct values."""
        log = TradeExecutionLog()
        log.add_record(sample_record)

        analysis = log.get_timestamp_analysis()

        # time_to_entry = 0.5 hours (10:30 to 11:00)
        assert analysis['avg_time_to_entry_hours'] == 0.5
        assert analysis['min_time_to_entry_hours'] == 0.5
        assert analysis['max_time_to_entry_hours'] == 0.5


class TestTradeExecutionLogValidationStats:
    """Tests for TradeExecutionLog validation statistics."""

    def test_get_validation_stats_empty(self):
        """get_validation_stats returns empty dict for empty log."""
        log = TradeExecutionLog()
        stats = log.get_validation_stats()
        assert stats == {}

    def test_get_validation_stats_fields(self, populated_log):
        """get_validation_stats returns expected fields."""
        stats = populated_log.get_validation_stats()

        assert 'total_trades' in stats
        assert 'validation_passed' in stats
        assert 'validation_rejected' in stats
        assert 'pass_rate' in stats

    def test_get_validation_stats_values(self, populated_log):
        """get_validation_stats calculates correct values."""
        stats = populated_log.get_validation_stats()

        assert stats['total_trades'] == 3
        assert stats['validation_passed'] == 2  # sample_record and sample_losing_record
        assert stats['validation_rejected'] == 1  # sample_rejected_record
        # pass_rate = 2/3 = 0.666...
        assert abs(stats['pass_rate'] - 2/3) < 0.001


class TestTradeExecutionLogFiltering:
    """Tests for TradeExecutionLog filtering methods."""

    def test_filter_by_pattern(self, populated_log):
        """filter_by_pattern returns filtered log."""
        filtered = populated_log.filter_by_pattern('3-1-2U')

        assert len(filtered) == 1
        assert filtered.records[0].pattern_type == '3-1-2U'

    def test_filter_by_pattern_no_match(self, populated_log):
        """filter_by_pattern returns empty log for no matches."""
        filtered = populated_log.filter_by_pattern('NONEXISTENT')
        assert len(filtered) == 0

    def test_filter_by_pattern_returns_new_log(self, populated_log):
        """filter_by_pattern returns new TradeExecutionLog instance."""
        filtered = populated_log.filter_by_pattern('3-1-2U')

        assert isinstance(filtered, TradeExecutionLog)
        assert filtered is not populated_log

    def test_filter_by_symbol(self, populated_log):
        """filter_by_symbol returns filtered log."""
        filtered = populated_log.filter_by_symbol('SPY')

        assert len(filtered) == 1
        assert filtered.records[0].symbol == 'SPY'

    def test_filter_by_symbol_no_match(self, populated_log):
        """filter_by_symbol returns empty log for no matches."""
        filtered = populated_log.filter_by_symbol('MSFT')
        assert len(filtered) == 0

    def test_filter_by_exit_reason(self, populated_log):
        """filter_by_exit_reason returns filtered log."""
        filtered = populated_log.filter_by_exit_reason('TARGET')

        assert len(filtered) == 1
        assert filtered.records[0].exit_reason == 'TARGET'

    def test_filter_by_exit_reason_stop(self, populated_log):
        """filter_by_exit_reason works with STOP."""
        filtered = populated_log.filter_by_exit_reason('STOP')

        assert len(filtered) == 1
        assert filtered.records[0].exit_reason == 'STOP'

    def test_filter_chaining(self, sample_timestamps):
        """Multiple filters can be chained."""
        log = TradeExecutionLog()

        # Add multiple SPY trades with different patterns
        record1 = TradeExecutionRecord(
            trade_id=1, symbol='SPY', pattern_type='3-1-2U', timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0, exit_price=485.0, stop_price=477.0, target_price=485.0,
        )
        record2 = TradeExecutionRecord(
            trade_id=2, symbol='SPY', pattern_type='2-1-2D', timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='STOP',
            entry_price=480.0, exit_price=475.0, stop_price=475.0, target_price=490.0,
        )
        log.add_records([record1, record2])

        # Filter by symbol then by exit reason
        filtered = log.filter_by_symbol('SPY').filter_by_exit_reason('TARGET')

        assert len(filtered) == 1
        assert filtered.records[0].pattern_type == '3-1-2U'


class TestTradeExecutionLogSummary:
    """Tests for TradeExecutionLog summary method."""

    def test_summary_empty_log(self):
        """summary returns message for empty log."""
        log = TradeExecutionLog()
        summary = log.summary()
        assert 'No trades recorded' in summary

    def test_summary_contains_total(self, populated_log):
        """summary contains total trade count."""
        summary = populated_log.summary()
        assert 'Total Trades: 3' in summary

    def test_summary_contains_exit_reasons(self, populated_log):
        """summary contains exit reason breakdown."""
        summary = populated_log.summary()

        assert 'Exit Reasons:' in summary
        assert 'TARGET' in summary
        assert 'STOP' in summary
        assert 'REJECTED' in summary

    def test_summary_contains_validation_rate(self, populated_log):
        """summary contains validation pass rate."""
        summary = populated_log.summary()
        assert 'Validation Pass Rate' in summary

    def test_summary_contains_timing_info(self, populated_log):
        """summary contains timing information."""
        summary = populated_log.summary()

        assert 'Avg Time to Entry' in summary
        assert 'Avg Time in Trade' in summary

    def test_summary_is_string(self, populated_log):
        """summary returns a string."""
        summary = populated_log.summary()
        assert isinstance(summary, str)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestTradeExecutionLogEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_record_with_none_timestamps(self):
        """Record handles None timestamps gracefully."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=None,
            entry_timestamp=None,
            exit_timestamp=None,
            exit_reason='REJECTED',
            entry_price=480.0,
            exit_price=480.0,
            stop_price=477.0,
            target_price=485.0,
        )
        # Should not raise, just return 0 or None
        assert record.days_held == 0
        assert record.time_to_entry is None
        assert record.time_in_trade is None

    def test_large_pnl_values(self, sample_timestamps):
        """Record handles large P/L values."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=500.0,
            stop_price=460.0,
            target_price=500.0,
            pnl=1_000_000.0,
            pnl_pct=2.0,
        )
        assert record.pnl == 1_000_000.0
        assert record.is_winner is True

    def test_negative_direction(self, sample_timestamps):
        """Record handles short direction correctly."""
        record = TradeExecutionRecord(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2D',
            timeframe='1H',
            pattern_timestamp=sample_timestamps['pattern'],
            entry_timestamp=sample_timestamps['entry'],
            exit_timestamp=sample_timestamps['exit'],
            exit_reason='TARGET',
            entry_price=480.0,
            exit_price=470.0,
            stop_price=485.0,
            target_price=470.0,
            direction=-1,
        )
        assert record.direction == -1

    def test_log_preserves_record_order(self, sample_timestamps):
        """Log preserves insertion order of records."""
        log = TradeExecutionLog()

        for i in range(5):
            record = TradeExecutionRecord(
                trade_id=i,
                symbol=f'SYM{i}',
                pattern_type='3-1-2U',
                timeframe='1H',
                pattern_timestamp=sample_timestamps['pattern'],
                entry_timestamp=sample_timestamps['entry'],
                exit_timestamp=sample_timestamps['exit'],
                exit_reason='TARGET',
                entry_price=100.0,
                exit_price=105.0,
                stop_price=97.0,
                target_price=105.0,
            )
            log.add_record(record)

        symbols = [r.symbol for r in log]
        assert symbols == ['SYM0', 'SYM1', 'SYM2', 'SYM3', 'SYM4']
