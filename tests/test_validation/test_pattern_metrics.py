"""
Tests for Pattern Metrics Analysis

Session 83G: Pattern metrics implementation per ATLAS Checklist Section 9.2.

Tests cover:
- PatternTradeResult dataclass functionality
- PatternType enum conversion
- PatternMetricsAnalyzer breakdown calculations
- Options accuracy metrics
- Helper functions and report generation
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from strat.pattern_metrics import (
    PatternTradeResult,
    PatternType,
    create_trade_from_backtest_row,
    create_trades_from_dataframe,
)
from validation.pattern_metrics import (
    PatternMetricsAnalyzer,
    analyze_pattern_metrics,
    get_best_patterns_by_metric,
    get_regime_pattern_compatibility,
    generate_pattern_report,
)
from validation.config import PatternMetricsConfig
from validation.results import PatternStats, OptionsAccuracyMetrics, PatternMetricsResults


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bullish_trade() -> PatternTradeResult:
    """Create a sample winning bullish trade."""
    return PatternTradeResult(
        trade_id=1,
        symbol='SPY',
        pattern_type='3-1-2U',
        timeframe='1D',
        regime='TREND_BULL',
        entry_date=datetime(2024, 1, 15),
        exit_date=datetime(2024, 1, 18),
        entry_price=100.0,
        exit_price=105.0,
        stop_price=97.0,
        target_price=108.0,
        pnl=500.0,
        pnl_pct=0.05,
        is_winner=True,
    )


@pytest.fixture
def sample_bearish_trade() -> PatternTradeResult:
    """Create a sample winning bearish trade."""
    return PatternTradeResult(
        trade_id=2,
        symbol='QQQ',
        pattern_type='3-1-2D',
        timeframe='1D',
        regime='TREND_BEAR',
        entry_date=datetime(2024, 2, 1),
        exit_date=datetime(2024, 2, 5),
        entry_price=380.0,
        exit_price=370.0,
        stop_price=390.0,
        target_price=360.0,
        pnl=1000.0,
        pnl_pct=0.0263,
        is_winner=True,
    )


@pytest.fixture
def sample_losing_trade() -> PatternTradeResult:
    """Create a sample losing trade."""
    return PatternTradeResult(
        trade_id=3,
        symbol='SPY',
        pattern_type='2-1-2U',
        timeframe='1D',
        regime='TREND_NEUTRAL',
        entry_date=datetime(2024, 3, 1),
        exit_date=datetime(2024, 3, 3),
        entry_price=450.0,
        exit_price=445.0,
        stop_price=440.0,
        target_price=460.0,
        pnl=-500.0,
        pnl_pct=-0.0111,
        is_winner=False,
        hit_stop=True,
    )


@pytest.fixture
def sample_options_trade() -> PatternTradeResult:
    """Create a sample options trade."""
    return PatternTradeResult(
        trade_id=4,
        symbol='SPY',
        pattern_type='3-1-2U',
        timeframe='1D',
        regime='TREND_BULL',
        entry_date=datetime(2024, 4, 1),
        exit_date=datetime(2024, 4, 5),
        entry_price=5.50,
        exit_price=7.20,
        stop_price=4.00,
        target_price=8.00,
        pnl=170.0,
        pnl_pct=0.309,
        is_winner=True,
        is_options_trade=True,
        data_source='ThetaData',
        entry_delta=0.55,
        exit_delta=0.62,
        entry_theta=-0.15,
        theta_cost=0.60,
        entry_iv=0.18,
        exit_iv=0.16,
    )


@pytest.fixture
def mixed_trades() -> List[PatternTradeResult]:
    """Create a mixed set of trades for analysis."""
    base_date = datetime(2024, 1, 1)
    trades = []

    # 3-1-2U trades (3 winners, 1 loser)
    for i in range(4):
        trade = PatternTradeResult(
            trade_id=i + 1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            regime='TREND_BULL',
            entry_date=base_date + timedelta(days=i * 7),
            exit_date=base_date + timedelta(days=i * 7 + 3),
            entry_price=100.0,
            exit_price=105.0 if i < 3 else 98.0,
            stop_price=97.0,
            target_price=108.0,
            pnl=500.0 if i < 3 else -200.0,
            pnl_pct=0.05 if i < 3 else -0.02,
            is_winner=i < 3,
        )
        trades.append(trade)

    # 2-1-2U trades (2 winners, 2 losers) - different timeframe
    for i in range(4):
        trade = PatternTradeResult(
            trade_id=i + 5,
            symbol='QQQ',
            pattern_type='2-1-2U',
            timeframe='1W',
            regime='TREND_NEUTRAL',
            entry_date=base_date + timedelta(days=i * 7 + 30),
            exit_date=base_date + timedelta(days=i * 7 + 35),
            entry_price=350.0,
            exit_price=360.0 if i < 2 else 345.0,
            stop_price=340.0,
            target_price=370.0,
            pnl=1000.0 if i < 2 else -500.0,
            pnl_pct=0.0286 if i < 2 else -0.0143,
            is_winner=i < 2,
        )
        trades.append(trade)

    # 2D-2U trades (1 winner, 1 loser) - different regime
    for i in range(2):
        trade = PatternTradeResult(
            trade_id=i + 9,
            symbol='IWM',
            pattern_type='2D-2U',
            timeframe='1D',
            regime='CRASH',
            entry_date=base_date + timedelta(days=i * 7 + 60),
            exit_date=base_date + timedelta(days=i * 7 + 65),
            entry_price=200.0,
            exit_price=210.0 if i == 0 else 195.0,
            stop_price=190.0,
            target_price=220.0,
            pnl=1000.0 if i == 0 else -500.0,
            pnl_pct=0.05 if i == 0 else -0.025,
            is_winner=i == 0,
        )
        trades.append(trade)

    return trades


@pytest.fixture
def options_trades() -> List[PatternTradeResult]:
    """Create a set of options trades for accuracy testing."""
    base_date = datetime(2024, 1, 1)
    trades = []

    data_sources = ['ThetaData', 'ThetaData', 'BlackScholes', 'Mixed']
    deltas = [0.55, 0.65, 0.45, 0.72]

    for i, (source, delta) in enumerate(zip(data_sources, deltas)):
        trade = PatternTradeResult(
            trade_id=i + 1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            regime='TREND_BULL',
            entry_date=base_date + timedelta(days=i * 7),
            exit_date=base_date + timedelta(days=i * 7 + 3),
            entry_price=5.0,
            exit_price=6.5 if i < 3 else 4.0,
            stop_price=3.5,
            target_price=7.0,
            pnl=150.0 if i < 3 else -100.0,
            pnl_pct=0.30 if i < 3 else -0.20,
            is_winner=i < 3,
            is_options_trade=True,
            data_source=source,
            entry_delta=delta,
            exit_delta=delta + 0.05,
            entry_theta=-0.12,
            theta_cost=0.48,
        )
        trades.append(trade)

    return trades


# =============================================================================
# PatternType Enum Tests
# =============================================================================

class TestPatternType:
    """Tests for PatternType enum."""

    def test_from_string_312u(self):
        """Test 3-1-2U pattern conversion."""
        assert PatternType.from_string('3-1-2U') == PatternType.PATTERN_312U
        assert PatternType.from_string('312U') == PatternType.PATTERN_312U
        assert PatternType.from_string('3-1-2u') == PatternType.PATTERN_312U

    def test_from_string_22_reversal(self):
        """Test 2-2 reversal pattern conversion."""
        assert PatternType.from_string('2D-2U') == PatternType.PATTERN_2D2U
        assert PatternType.from_string('2U-2D') == PatternType.PATTERN_2U2D

    def test_from_string_unknown(self):
        """Test unknown pattern returns UNKNOWN."""
        assert PatternType.from_string('invalid') == PatternType.UNKNOWN
        assert PatternType.from_string('') == PatternType.UNKNOWN

    def test_is_bullish(self):
        """Test bullish pattern detection."""
        assert PatternType.PATTERN_312U.is_bullish()
        assert PatternType.PATTERN_2D2U.is_bullish()
        assert not PatternType.PATTERN_312D.is_bullish()

    def test_is_bearish(self):
        """Test bearish pattern detection."""
        assert PatternType.PATTERN_312D.is_bearish()
        assert PatternType.PATTERN_2U2D.is_bearish()
        assert not PatternType.PATTERN_312U.is_bearish()

    def test_base_pattern(self):
        """Test base pattern extraction."""
        assert PatternType.PATTERN_312U.base_pattern() == '3-1-2'
        assert PatternType.PATTERN_312D.base_pattern() == '3-1-2'
        assert PatternType.PATTERN_212U.base_pattern() == '2-1-2'
        assert PatternType.PATTERN_2D2U.base_pattern() == '2-2'
        assert PatternType.PATTERN_32D2U.base_pattern() == '3-2-2'


# =============================================================================
# PatternTradeResult Tests
# =============================================================================

class TestPatternTradeResult:
    """Tests for PatternTradeResult dataclass."""

    def test_creation(self, sample_bullish_trade):
        """Test basic trade creation."""
        assert sample_bullish_trade.trade_id == 1
        assert sample_bullish_trade.symbol == 'SPY'
        assert sample_bullish_trade.pattern_type == '3-1-2U'
        assert sample_bullish_trade.is_winner

    def test_days_held_calculation(self):
        """Test days_held auto-calculation."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 5),
            entry_price=100.0,
            exit_price=105.0,
            stop_price=97.0,
            target_price=108.0,
            pnl=500.0,
            pnl_pct=0.05,
            is_winner=True,
        )
        assert trade.days_held == 4

    def test_pattern_enum_property(self, sample_bullish_trade):
        """Test pattern_enum property."""
        assert sample_bullish_trade.pattern_enum == PatternType.PATTERN_312U

    def test_is_bullish_property(self, sample_bullish_trade, sample_bearish_trade):
        """Test is_bullish property."""
        assert sample_bullish_trade.is_bullish
        assert not sample_bearish_trade.is_bullish

    def test_is_bearish_property(self, sample_bullish_trade, sample_bearish_trade):
        """Test is_bearish property."""
        assert not sample_bullish_trade.is_bearish
        assert sample_bearish_trade.is_bearish

    def test_risk_amount(self, sample_bullish_trade):
        """Test risk_amount calculation."""
        # Bullish: entry=100, stop=97 -> risk=3
        assert sample_bullish_trade.risk_amount == 3.0

    def test_reward_amount(self, sample_bullish_trade):
        """Test reward_amount calculation."""
        # Bullish: target=108, entry=100 -> reward=8
        assert sample_bullish_trade.reward_amount == 8.0

    def test_planned_risk_reward(self, sample_bullish_trade):
        """Test planned_risk_reward calculation."""
        # reward=8, risk=3 -> R:R = 8/3 = 2.67
        assert abs(sample_bullish_trade.planned_risk_reward - 2.67) < 0.01

    def test_to_dict(self, sample_bullish_trade):
        """Test to_dict conversion."""
        d = sample_bullish_trade.to_dict()
        assert d['trade_id'] == 1
        assert d['symbol'] == 'SPY'
        assert d['pattern_type'] == '3-1-2U'
        assert d['base_pattern'] == '3-1-2'
        assert d['pnl'] == 500.0


class TestTradeFactoryFunctions:
    """Tests for trade factory functions."""

    def test_create_trade_from_backtest_row(self):
        """Test creating trade from dictionary row."""
        row = {
            'symbol': 'AAPL',
            'pattern_type': '2-1-2U',
            'timeframe': '1D',
            'regime': 'TREND_BULL',
            'entry_date': datetime(2024, 1, 1),
            'exit_date': datetime(2024, 1, 3),
            'entry_price': 180.0,
            'exit_price': 185.0,
            'stop_price': 175.0,
            'target_price': 190.0,
            'pnl': 500.0,
            'pnl_pct': 0.0278,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        assert trade.trade_id == 1
        assert trade.symbol == 'AAPL'
        assert trade.pattern_type == '2-1-2U'
        assert trade.is_winner

    def test_create_trade_with_string_dates(self):
        """Test creating trade with ISO date strings."""
        row = {
            'symbol': 'AAPL',
            'pattern_type': '2-1-2U',
            'timeframe': '1D',
            'entry_date': '2024-01-01T00:00:00',
            'exit_date': '2024-01-03T00:00:00',
            'entry_price': 180.0,
            'exit_price': 185.0,
            'stop_price': 175.0,
            'target_price': 190.0,
            'pnl': 500.0,
        }
        trade = create_trade_from_backtest_row(row, trade_id=1)

        assert trade.entry_date == datetime(2024, 1, 1)
        assert trade.exit_date == datetime(2024, 1, 3)


# =============================================================================
# PatternMetricsAnalyzer Tests
# =============================================================================

class TestPatternMetricsAnalyzer:
    """Tests for PatternMetricsAnalyzer class."""

    def test_analyze_empty_trades(self):
        """Test analyzing empty trade list."""
        analyzer = PatternMetricsAnalyzer()
        results = analyzer.analyze([])

        assert results.total_trades == 0
        assert results.overall_win_rate == 0.0
        assert results.by_pattern == {}

    def test_analyze_single_trade(self, sample_bullish_trade):
        """Test analyzing single trade."""
        # Use min_pattern_trades=1 to include single trade
        config = PatternMetricsConfig(min_pattern_trades=1)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze([sample_bullish_trade])

        assert results.total_trades == 1
        assert results.overall_win_rate == 1.0

    def test_analyze_by_pattern(self, mixed_trades):
        """Test pattern breakdown analysis."""
        # Use min_pattern_trades=2 to include patterns with 2+ trades
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # Should have 3 pattern types
        assert '3-1-2U' in results.by_pattern
        assert '2-1-2U' in results.by_pattern
        assert '2D-2U' in results.by_pattern

        # 3-1-2U: 4 trades, 3 winners
        stats_312u = results.by_pattern['3-1-2U']
        assert stats_312u.trade_count == 4
        assert stats_312u.win_count == 3
        assert abs(stats_312u.win_rate - 0.75) < 0.001

    def test_analyze_by_timeframe(self, mixed_trades):
        """Test timeframe breakdown analysis."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # Should have 2 timeframes
        assert '1D' in results.by_timeframe
        assert '1W' in results.by_timeframe

        # Daily should have 3-1-2U and 2D-2U
        assert '3-1-2U' in results.by_timeframe['1D']
        assert '2D-2U' in results.by_timeframe['1D']

        # Weekly should have 2-1-2U
        assert '2-1-2U' in results.by_timeframe['1W']

    def test_analyze_by_regime(self, mixed_trades):
        """Test regime breakdown analysis."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # Should have 3 regimes
        assert 'TREND_BULL' in results.by_regime
        assert 'TREND_NEUTRAL' in results.by_regime
        assert 'CRASH' in results.by_regime

    def test_best_worst_pattern(self, mixed_trades):
        """Test best/worst pattern identification."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # Best pattern should have highest expectancy
        assert results.best_pattern is not None
        assert results.worst_pattern is not None
        assert results.best_pattern != results.worst_pattern

    def test_min_pattern_trades_filter(self, mixed_trades):
        """Test minimum trades filter."""
        # Config with high minimum trades
        config = PatternMetricsConfig(min_pattern_trades=10)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # No patterns should meet 10 trade minimum
        assert len(results.by_pattern) == 0


class TestPatternStatsCalculation:
    """Tests for PatternStats calculation accuracy."""

    def test_win_rate_calculation(self, mixed_trades):
        """Test win rate calculation."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # 2-1-2U: 2 winners out of 4
        stats = results.by_pattern['2-1-2U']
        assert abs(stats.win_rate - 0.50) < 0.001

    def test_pnl_calculations(self, mixed_trades):
        """Test P/L calculation accuracy."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # 3-1-2U: 3 winners at 500, 1 loser at -200 = 1300 total
        stats = results.by_pattern['3-1-2U']
        assert stats.total_pnl == 1300.0
        assert abs(stats.avg_pnl - 325.0) < 0.001

    def test_avg_winner_loser_calculation(self, mixed_trades):
        """Test average winner/loser calculation."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # 3-1-2U: 3 winners at 500 each, 1 loser at -200
        stats = results.by_pattern['3-1-2U']
        assert stats.avg_winner_pnl == 500.0
        assert stats.avg_loser_pnl == -200.0

    def test_profit_factor_calculation(self, mixed_trades):
        """Test profit factor calculation."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # 3-1-2U: gross profit=1500, gross loss=200 -> PF=7.5
        stats = results.by_pattern['3-1-2U']
        assert abs(stats.profit_factor - 7.5) < 0.01

    def test_expectancy_calculation(self, mixed_trades):
        """Test expectancy calculation."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        analyzer = PatternMetricsAnalyzer(config=config)
        results = analyzer.analyze(mixed_trades)

        # 3-1-2U: (0.75 * 500) + (0.25 * -200) = 375 - 50 = 325
        stats = results.by_pattern['3-1-2U']
        assert abs(stats.expectancy - 325.0) < 0.01


class TestOptionsAccuracyMetrics:
    """Tests for options accuracy metrics."""

    def test_data_source_breakdown(self, options_trades):
        """Test data source counting."""
        analyzer = PatternMetricsAnalyzer()
        results = analyzer.analyze(options_trades)

        assert results.options_accuracy is not None
        oa = results.options_accuracy

        assert oa.thetadata_trades == 2
        assert oa.black_scholes_trades == 1
        assert oa.mixed_trades == 1

    def test_thetadata_coverage(self, options_trades):
        """Test ThetaData coverage percentage."""
        analyzer = PatternMetricsAnalyzer()
        results = analyzer.analyze(options_trades)

        oa = results.options_accuracy
        # 2 ThetaData out of 4 = 50%
        assert abs(oa.thetadata_coverage_pct - 0.50) < 0.001

    def test_delta_calculations(self, options_trades):
        """Test delta accuracy calculations."""
        analyzer = PatternMetricsAnalyzer()
        results = analyzer.analyze(options_trades)

        oa = results.options_accuracy
        # Deltas: 0.55, 0.65, 0.45, 0.72 -> avg = 0.5925
        assert abs(oa.avg_entry_delta - 0.5925) < 0.001

    def test_delta_in_optimal_range(self, options_trades):
        """Test delta in optimal range percentage."""
        analyzer = PatternMetricsAnalyzer()
        results = analyzer.analyze(options_trades)

        oa = results.options_accuracy
        # Deltas: 0.55 (yes), 0.65 (yes), 0.45 (no), 0.72 (yes) = 75%
        assert abs(oa.delta_in_optimal_range_pct - 0.75) < 0.001


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_analyze_pattern_metrics_convenience(self, mixed_trades):
        """Test convenience function."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        assert results.total_trades == 10

    def test_get_best_patterns_by_expectancy(self, mixed_trades):
        """Test getting best patterns by expectancy."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        best = get_best_patterns_by_metric(results, metric='expectancy', top_n=2)

        assert len(best) == 2
        # Should be sorted descending
        assert best[0][1] >= best[1][1]

    def test_get_best_patterns_by_win_rate(self, mixed_trades):
        """Test getting best patterns by win rate."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        best = get_best_patterns_by_metric(results, metric='win_rate', top_n=3)

        assert len(best) == 3
        # All values should be <= 1.0
        for pattern, win_rate in best:
            assert win_rate <= 1.0

    def test_get_regime_pattern_compatibility(self, mixed_trades):
        """Test regime-pattern compatibility analysis."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        compat = get_regime_pattern_compatibility(
            results,
            min_win_rate=0.50,
            min_expectancy=0.0
        )

        assert 'TREND_BULL' in compat
        assert 'TREND_NEUTRAL' in compat
        assert 'CRASH' in compat

    def test_generate_pattern_report(self, mixed_trades):
        """Test report generation."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        report = generate_pattern_report(results)

        assert 'STRAT PATTERN METRICS REPORT' in report
        assert 'Total Trades' in report
        assert '3-1-2U' in report
        assert 'BY TIMEFRAME' in report
        assert 'BY REGIME' in report


class TestReportGeneration:
    """Tests for report generation."""

    def test_pattern_metrics_results_summary(self, mixed_trades):
        """Test PatternMetricsResults.summary() method."""
        results = analyze_pattern_metrics(mixed_trades)
        summary = results.summary()

        assert 'PATTERN METRICS ANALYSIS' in summary
        assert 'Total Trades' in summary
        assert 'Win Rate' in summary

    def test_report_with_options(self, options_trades):
        """Test report with options accuracy section."""
        results = analyze_pattern_metrics(options_trades)
        report = generate_pattern_report(results, include_details=True)

        assert 'OPTIONS ACCURACY' in report
        assert 'ThetaData' in report
        assert 'Delta' in report

    def test_report_without_details(self, mixed_trades):
        """Test report without detailed breakdowns."""
        results = analyze_pattern_metrics(mixed_trades)
        report = generate_pattern_report(results, include_details=False)

        # Should have pattern summary but not timeframe/regime details
        assert 'BY PATTERN TYPE' in report
        # Timeframe section should be minimal when include_details=False


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_winners(self):
        """Test analysis with all winning trades."""
        trades = []
        for i in range(5):
            trades.append(PatternTradeResult(
                trade_id=i + 1,
                symbol='SPY',
                pattern_type='3-1-2U',
                timeframe='1D',
                regime='TREND_BULL',
                entry_date=datetime(2024, 1, i + 1),
                exit_date=datetime(2024, 1, i + 3),
                entry_price=100.0,
                exit_price=105.0,
                stop_price=97.0,
                target_price=108.0,
                pnl=500.0,
                pnl_pct=0.05,
                is_winner=True,
            ))

        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(trades, config=config)
        assert results.overall_win_rate == 1.0
        assert results.by_pattern['3-1-2U'].profit_factor == 999.99  # Capped inf

    def test_all_losers(self):
        """Test analysis with all losing trades."""
        trades = []
        for i in range(5):
            trades.append(PatternTradeResult(
                trade_id=i + 1,
                symbol='SPY',
                pattern_type='3-1-2U',
                timeframe='1D',
                regime='CRASH',
                entry_date=datetime(2024, 1, i + 1),
                exit_date=datetime(2024, 1, i + 3),
                entry_price=100.0,
                exit_price=95.0,
                stop_price=97.0,
                target_price=108.0,
                pnl=-500.0,
                pnl_pct=-0.05,
                is_winner=False,
            ))

        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(trades, config=config)
        assert results.overall_win_rate == 0.0
        assert results.by_pattern['3-1-2U'].profit_factor == 0.0

    def test_single_pattern_type(self):
        """Test analysis with single pattern type."""
        trades = []
        for i in range(3):
            trades.append(PatternTradeResult(
                trade_id=i + 1,
                symbol='SPY',
                pattern_type='2D-2U',
                timeframe='1D',
                regime='TREND_NEUTRAL',
                entry_date=datetime(2024, 1, i + 1),
                exit_date=datetime(2024, 1, i + 3),
                entry_price=100.0,
                exit_price=103.0 if i < 2 else 97.0,
                stop_price=97.0,
                target_price=106.0,
                pnl=300.0 if i < 2 else -300.0,
                pnl_pct=0.03 if i < 2 else -0.03,
                is_winner=i < 2,
            ))

        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(trades, config=config)
        assert len(results.by_pattern) == 1
        assert '2D-2U' in results.by_pattern
        assert results.best_pattern == results.worst_pattern  # Only one pattern

    def test_zero_risk_trade(self):
        """Test trade with zero risk (entry == stop)."""
        trade = PatternTradeResult(
            trade_id=1,
            symbol='SPY',
            pattern_type='3-1-2U',
            timeframe='1D',
            entry_date=datetime(2024, 1, 1),
            exit_date=datetime(2024, 1, 3),
            entry_price=100.0,
            exit_price=105.0,
            stop_price=100.0,  # Same as entry
            target_price=108.0,
            pnl=500.0,
            pnl_pct=0.05,
            is_winner=True,
        )

        assert trade.risk_amount == 0.0
        assert trade.planned_risk_reward == 0.0  # Division by zero handled


class TestPatternMetricsResultsSerialization:
    """Tests for results serialization."""

    def test_to_dict(self, mixed_trades):
        """Test results to_dict conversion."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        d = results.to_dict()

        assert 'by_pattern' in d
        assert 'by_timeframe' in d
        assert 'by_regime' in d
        assert 'total_trades' in d
        assert d['total_trades'] == 10

    def test_pattern_stats_to_dict(self, mixed_trades):
        """Test PatternStats to_dict conversion."""
        config = PatternMetricsConfig(min_pattern_trades=2)
        results = analyze_pattern_metrics(mixed_trades, config=config)
        stats = results.by_pattern['3-1-2U']
        d = stats.to_dict()

        assert d['pattern_type'] == '3-1-2U'
        assert d['trade_count'] == 4
        assert 'win_rate' in d
        assert 'expectancy' in d
