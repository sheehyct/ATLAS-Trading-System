"""
Tests for trade analytics functionality.

Session EQUITY-51: P6 Trade Analytics Dashboard test coverage.

Tests calculate_trade_analytics() function which provides:
- Pattern breakdown (win rate, avg P&L by pattern type)
- TFC breakdown (win rate, avg P&L by TFC score)
- Timeframe breakdown (win rate, avg P&L by timeframe)

NOTE: These tests verify calculation correctness only. TFC score does not
guarantee performance - it indicates timeframe alignment, not trade outcome.
"""
import pytest
from dashboard.components.options_panel import calculate_trade_analytics


class TestCalculateTradeAnalyticsEmpty:
    """Tests for empty and edge cases."""

    def test_empty_trades_returns_empty_breakdowns(self):
        """Empty trades list returns empty breakdown dicts."""
        result = calculate_trade_analytics([])

        assert result == {
            'pattern_breakdown': {},
            'tfc_breakdown': {},
            'timeframe_breakdown': {}
        }

    def test_none_trades_returns_empty_breakdowns(self):
        """None input treated as empty (returns empty breakdowns)."""
        result = calculate_trade_analytics(None)

        assert result == {
            'pattern_breakdown': {},
            'tfc_breakdown': {},
            'timeframe_breakdown': {}
        }


class TestCalculateTradeAnalyticsSingleTrade:
    """Tests for single trade scenarios."""

    def test_single_winning_trade(self):
        """Single winning trade calculates correctly."""
        trades = [{
            'pattern': '3-1-2U',
            'tfc_score': 3,
            'timeframe': '1H',
            'realized_pnl': 150.00
        }]

        result = calculate_trade_analytics(trades)

        # Pattern breakdown
        assert '3-1-2U' in result['pattern_breakdown']
        pattern_data = result['pattern_breakdown']['3-1-2U']
        assert pattern_data['trades'] == 1
        assert pattern_data['winners'] == 1
        assert pattern_data['win_rate'] == 100.0
        assert pattern_data['total_pnl'] == 150.00
        assert pattern_data['avg_pnl'] == 150.00

        # TFC breakdown
        assert 3 in result['tfc_breakdown']
        tfc_data = result['tfc_breakdown'][3]
        assert tfc_data['trades'] == 1
        assert tfc_data['win_rate'] == 100.0

        # Timeframe breakdown
        assert '1H' in result['timeframe_breakdown']
        tf_data = result['timeframe_breakdown']['1H']
        assert tf_data['trades'] == 1

    def test_single_losing_trade(self):
        """Single losing trade calculates correctly."""
        trades = [{
            'pattern': '2D-2U',
            'tfc_score': 2,
            'timeframe': '1D',
            'realized_pnl': -75.50
        }]

        result = calculate_trade_analytics(trades)

        pattern_data = result['pattern_breakdown']['2D-2U']
        assert pattern_data['trades'] == 1
        assert pattern_data['winners'] == 0
        assert pattern_data['win_rate'] == 0.0
        assert pattern_data['total_pnl'] == -75.50
        assert pattern_data['avg_pnl'] == -75.50

    def test_single_breakeven_trade(self):
        """Trade with zero P&L counts as loser (not > 0)."""
        trades = [{
            'pattern': '3-2U',
            'tfc_score': 4,
            'timeframe': '1W',
            'realized_pnl': 0.0
        }]

        result = calculate_trade_analytics(trades)

        pattern_data = result['pattern_breakdown']['3-2U']
        assert pattern_data['winners'] == 0  # Zero P&L is not a winner
        assert pattern_data['win_rate'] == 0.0


class TestCalculateTradeAnalyticsMultipleTrades:
    """Tests for multiple trade aggregations."""

    def test_multiple_trades_same_pattern(self):
        """Multiple trades with same pattern aggregate correctly."""
        trades = [
            {'pattern': '3-1-2U', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
            {'pattern': '3-1-2U', 'tfc_score': 4, 'timeframe': '1H', 'realized_pnl': -50},
            {'pattern': '3-1-2U', 'tfc_score': 3, 'timeframe': '1D', 'realized_pnl': 200},
        ]

        result = calculate_trade_analytics(trades)

        pattern_data = result['pattern_breakdown']['3-1-2U']
        assert pattern_data['trades'] == 3
        assert pattern_data['winners'] == 2  # 100 and 200 are positive
        assert pattern_data['win_rate'] == pytest.approx(66.67, rel=0.01)
        assert pattern_data['total_pnl'] == 250  # 100 - 50 + 200
        assert pattern_data['avg_pnl'] == pytest.approx(83.33, rel=0.01)

    def test_multiple_patterns_sorted_by_count(self):
        """Breakdowns are sorted by trade count descending."""
        trades = [
            {'pattern': 'A', 'tfc_score': 1, 'timeframe': '1H', 'realized_pnl': 10},
            {'pattern': 'B', 'tfc_score': 2, 'timeframe': '1D', 'realized_pnl': 20},
            {'pattern': 'B', 'tfc_score': 2, 'timeframe': '1D', 'realized_pnl': 30},
            {'pattern': 'C', 'tfc_score': 3, 'timeframe': '1W', 'realized_pnl': 40},
            {'pattern': 'C', 'tfc_score': 3, 'timeframe': '1W', 'realized_pnl': 50},
            {'pattern': 'C', 'tfc_score': 3, 'timeframe': '1W', 'realized_pnl': 60},
        ]

        result = calculate_trade_analytics(trades)

        # Pattern breakdown should be sorted: C (3), B (2), A (1)
        pattern_keys = list(result['pattern_breakdown'].keys())
        assert pattern_keys == ['C', 'B', 'A']

        # Verify counts
        assert result['pattern_breakdown']['C']['trades'] == 3
        assert result['pattern_breakdown']['B']['trades'] == 2
        assert result['pattern_breakdown']['A']['trades'] == 1

    def test_tfc_breakdown_aggregates_by_score(self):
        """TFC breakdown groups trades by score value."""
        trades = [
            {'pattern': 'X', 'tfc_score': 0, 'timeframe': '1H', 'realized_pnl': -100},
            {'pattern': 'X', 'tfc_score': 2, 'timeframe': '1H', 'realized_pnl': 50},
            {'pattern': 'X', 'tfc_score': 4, 'timeframe': '1H', 'realized_pnl': 200},
            {'pattern': 'X', 'tfc_score': 4, 'timeframe': '1H', 'realized_pnl': 150},
        ]

        result = calculate_trade_analytics(trades)

        # TFC 4 has highest count, should be first
        tfc_keys = list(result['tfc_breakdown'].keys())
        assert tfc_keys[0] == 4  # Most trades

        # Verify correct aggregation
        assert result['tfc_breakdown'][0]['trades'] == 1
        assert result['tfc_breakdown'][0]['avg_pnl'] == -100

        assert result['tfc_breakdown'][4]['trades'] == 2
        assert result['tfc_breakdown'][4]['avg_pnl'] == 175  # (200 + 150) / 2

    def test_timeframe_breakdown(self):
        """Timeframe breakdown aggregates correctly."""
        trades = [
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 50},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1D', 'realized_pnl': 200},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1W', 'realized_pnl': -50},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1M', 'realized_pnl': 300},
        ]

        result = calculate_trade_analytics(trades)

        # 1H has most trades
        tf_keys = list(result['timeframe_breakdown'].keys())
        assert tf_keys[0] == '1H'

        assert result['timeframe_breakdown']['1H']['trades'] == 2
        assert result['timeframe_breakdown']['1H']['win_rate'] == 100.0
        assert result['timeframe_breakdown']['1H']['avg_pnl'] == 75  # (50 + 100) / 2


class TestCalculateTradeAnalyticsMissingFields:
    """Tests for handling missing or invalid field values."""

    def test_missing_pattern_becomes_unknown(self):
        """Missing pattern field becomes 'Unknown'."""
        trades = [
            {'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
        ]

        result = calculate_trade_analytics(trades)

        assert 'Unknown' in result['pattern_breakdown']
        assert result['pattern_breakdown']['Unknown']['trades'] == 1

    def test_none_pattern_becomes_unknown(self):
        """None pattern value becomes 'Unknown'."""
        trades = [
            {'pattern': None, 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
        ]

        result = calculate_trade_analytics(trades)

        assert 'Unknown' in result['pattern_breakdown']

    def test_dash_pattern_becomes_unknown(self):
        """Dash '-' pattern value becomes 'Unknown'."""
        trades = [
            {'pattern': '-', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
        ]

        result = calculate_trade_analytics(trades)

        assert 'Unknown' in result['pattern_breakdown']

    def test_empty_string_pattern_becomes_unknown(self):
        """Empty string pattern becomes 'Unknown'."""
        trades = [
            {'pattern': '', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
        ]

        result = calculate_trade_analytics(trades)

        assert 'Unknown' in result['pattern_breakdown']

    def test_missing_realized_pnl_defaults_to_zero(self):
        """Missing realized_pnl defaults to 0."""
        trades = [
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H'},  # No realized_pnl
        ]

        result = calculate_trade_analytics(trades)

        assert result['pattern_breakdown']['X']['total_pnl'] == 0
        assert result['pattern_breakdown']['X']['avg_pnl'] == 0
        assert result['pattern_breakdown']['X']['winners'] == 0

    def test_none_tfc_score_becomes_unknown(self):
        """None TFC score becomes 'Unknown'."""
        trades = [
            {'pattern': 'X', 'tfc_score': None, 'timeframe': '1H', 'realized_pnl': 100},
        ]

        result = calculate_trade_analytics(trades)

        assert 'Unknown' in result['tfc_breakdown']


class TestCalculateTradeAnalyticsCalculationAccuracy:
    """Tests verifying calculation accuracy with known values."""

    def test_win_rate_calculation_various_ratios(self):
        """Win rate calculated correctly for various winner/loser ratios."""
        # 3 winners, 2 losers = 60% WR
        trades = [
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 100},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 50},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 25},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': -75},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': -100},
        ]

        result = calculate_trade_analytics(trades)

        assert result['pattern_breakdown']['X']['winners'] == 3
        assert result['pattern_breakdown']['X']['win_rate'] == 60.0

    def test_avg_pnl_calculation_with_decimals(self):
        """Avg P&L handles decimal values correctly."""
        trades = [
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 123.45},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': -67.89},
            {'pattern': 'X', 'tfc_score': 3, 'timeframe': '1H', 'realized_pnl': 200.00},
        ]

        result = calculate_trade_analytics(trades)

        expected_total = 123.45 - 67.89 + 200.00  # 255.56
        expected_avg = expected_total / 3  # 85.1867

        assert result['pattern_breakdown']['X']['total_pnl'] == pytest.approx(expected_total, rel=0.001)
        assert result['pattern_breakdown']['X']['avg_pnl'] == pytest.approx(expected_avg, rel=0.01)

    def test_mixed_data_calculates_independently(self):
        """Different dimensions calculate independently from same trade set."""
        trades = [
            {'pattern': 'A', 'tfc_score': 2, 'timeframe': '1H', 'realized_pnl': 100},
            {'pattern': 'A', 'tfc_score': 4, 'timeframe': '1D', 'realized_pnl': -50},
            {'pattern': 'B', 'tfc_score': 2, 'timeframe': '1H', 'realized_pnl': 75},
        ]

        result = calculate_trade_analytics(trades)

        # Pattern breakdown: A has 2 trades, B has 1
        assert result['pattern_breakdown']['A']['trades'] == 2
        assert result['pattern_breakdown']['A']['total_pnl'] == 50  # 100 - 50
        assert result['pattern_breakdown']['B']['trades'] == 1
        assert result['pattern_breakdown']['B']['total_pnl'] == 75

        # TFC breakdown: 2 has 2 trades, 4 has 1
        assert result['tfc_breakdown'][2]['trades'] == 2
        assert result['tfc_breakdown'][2]['total_pnl'] == 175  # 100 + 75
        assert result['tfc_breakdown'][4]['trades'] == 1
        assert result['tfc_breakdown'][4]['total_pnl'] == -50

        # Timeframe breakdown: 1H has 2 trades, 1D has 1
        assert result['timeframe_breakdown']['1H']['trades'] == 2
        assert result['timeframe_breakdown']['1H']['total_pnl'] == 175  # 100 + 75
        assert result['timeframe_breakdown']['1D']['trades'] == 1
        assert result['timeframe_breakdown']['1D']['total_pnl'] == -50
