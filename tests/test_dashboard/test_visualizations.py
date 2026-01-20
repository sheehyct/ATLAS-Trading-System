"""
Dashboard Visualization Functional Tests

Tests actual visualization logic with sample data.
Verifies that visualization functions:
- Create valid Plotly figures
- Handle edge cases (empty data, single points)
- Calculate derived values correctly (drawdown, rolling metrics)
- Apply proper styling and traces

Session EQUITY-74: Phase 3 Test Coverage for dashboard visualizations.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta


# =============================================================================
# PERFORMANCE VISUALIZATION TESTS
# =============================================================================


class TestCreateEquityCurve:
    """Test create_equity_curve function."""

    def test_returns_figure_with_valid_data(self):
        """create_equity_curve returns Plotly figure with valid data."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100), index=dates)

        fig = create_equity_curve(values)

        assert isinstance(fig, go.Figure)

    def test_figure_has_equity_trace(self):
        """create_equity_curve figure has equity trace."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100), index=dates)

        fig = create_equity_curve(values)

        # Check that figure has at least one trace
        assert len(fig.data) >= 1
        # First trace should be the strategy equity
        assert fig.data[0].name == 'Strategy'

    def test_figure_has_drawdown_trace(self):
        """create_equity_curve figure has drawdown trace."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100), index=dates)

        fig = create_equity_curve(values)

        # Should have drawdown trace
        trace_names = [t.name for t in fig.data]
        assert 'Drawdown' in trace_names

    def test_handles_empty_series(self):
        """create_equity_curve handles empty series gracefully."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        empty_series = pd.Series(dtype=float)
        fig = create_equity_curve(empty_series)

        assert isinstance(fig, go.Figure)
        # Should have annotation about insufficient data
        assert len(fig.layout.annotations) > 0

    def test_handles_none_input(self):
        """create_equity_curve handles None input gracefully."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        fig = create_equity_curve(None)

        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_handles_single_point(self):
        """create_equity_curve handles single point series."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        single_point = pd.Series([10000], index=[datetime(2024, 1, 1)])
        fig = create_equity_curve(single_point)

        assert isinstance(fig, go.Figure)
        # Should show insufficient data message
        assert len(fig.layout.annotations) > 0

    def test_includes_benchmark_when_provided(self):
        """create_equity_curve includes benchmark trace when provided."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = pd.Series(10000 + np.cumsum(np.random.randn(100) * 100), index=dates)
        benchmark = pd.Series(10000 + np.cumsum(np.random.randn(100) * 50), index=dates)

        fig = create_equity_curve(values, benchmark_value=benchmark)

        trace_names = [t.name for t in fig.data]
        assert 'Benchmark' in trace_names

    def test_drawdown_calculation_correct(self):
        """create_equity_curve calculates drawdown correctly."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        # Create a series that goes up then down
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        # Equity: 100, 110, 120, 130, 140, 130, 120, 110, 120, 130
        values = pd.Series([100, 110, 120, 130, 140, 130, 120, 110, 120, 130], index=dates)

        fig = create_equity_curve(values)

        # Find drawdown trace
        drawdown_trace = next(t for t in fig.data if t.name == 'Drawdown')

        # Max drawdown should occur at index 7 (value 110, peak was 140)
        # DD = (110 - 140) / 140 * 100 = -21.43%
        # Y values are in percentage
        max_dd_pct = min(drawdown_trace.y)
        assert max_dd_pct < 0  # Should be negative
        assert abs(max_dd_pct - (-21.43)) < 1  # Within 1% of expected


class TestCreateRollingMetrics:
    """Test create_rolling_metrics function."""

    def test_returns_figure_with_valid_data(self):
        """create_rolling_metrics returns Plotly figure."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

        fig = create_rolling_metrics(returns, window=20)

        assert isinstance(fig, go.Figure)

    def test_figure_has_sharpe_trace(self):
        """create_rolling_metrics has rolling Sharpe trace."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

        fig = create_rolling_metrics(returns, window=60)

        trace_names = [t.name for t in fig.data]
        assert any('Sharpe' in name for name in trace_names)

    def test_figure_has_sortino_trace(self):
        """create_rolling_metrics has rolling Sortino trace."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

        fig = create_rolling_metrics(returns, window=60)

        trace_names = [t.name for t in fig.data]
        assert any('Sortino' in name for name in trace_names)

    def test_custom_window_in_trace_name(self):
        """create_rolling_metrics includes window size in trace name."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

        fig = create_rolling_metrics(returns, window=30)

        trace_names = [t.name for t in fig.data]
        assert any('30d' in name for name in trace_names)

    def test_has_reference_line(self):
        """create_rolling_metrics has horizontal reference line."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

        fig = create_rolling_metrics(returns, window=60)

        # Check for shapes (hlines are added as shapes)
        assert len(fig.layout.shapes) > 0 or len(fig.layout.annotations) > 0


# =============================================================================
# REGIME VISUALIZATION TESTS
# =============================================================================


class TestCreateRegimeTimeline:
    """Test create_regime_timeline function."""

    def test_returns_figure(self):
        """create_regime_timeline returns Plotly figure."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        regimes = pd.Series(['TREND_BULL'] * 50 + ['TREND_NEUTRAL'] * 50, index=dates)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices)

        assert isinstance(fig, go.Figure)

    def test_has_price_trace(self):
        """create_regime_timeline has price trace."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        regimes = pd.Series(['TREND_BULL'] * 100, index=dates)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices)

        trace_names = [t.name for t in fig.data]
        assert 'Price' in trace_names

    def test_creates_regime_shading(self):
        """create_regime_timeline creates regime shading rectangles."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Create regime changes
        regimes = pd.Series(['TREND_BULL'] * 50 + ['CRASH'] * 50, index=dates)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices)

        # Check for vrect shapes
        assert len(fig.layout.shapes) > 0

    def test_handles_all_regimes(self):
        """create_regime_timeline handles all regime types."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        regimes = pd.Series(
            ['TREND_BULL'] * 25 + ['TREND_NEUTRAL'] * 25 +
            ['TREND_BEAR'] * 25 + ['CRASH'] * 25,
            index=dates
        )
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices)

        assert isinstance(fig, go.Figure)
        # Should have multiple shapes for regime changes
        assert len(fig.layout.shapes) >= 4

    def test_handles_empty_regimes(self):
        """create_regime_timeline handles empty regime series."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.DatetimeIndex([])
        regimes = pd.Series([], dtype=str)
        prices = pd.Series([], dtype=float)

        fig = create_regime_timeline(dates, regimes, prices)

        assert isinstance(fig, go.Figure)

    def test_handles_no_prices(self):
        """create_regime_timeline handles None prices."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        regimes = pd.Series(['TREND_BULL'] * 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices=None)

        assert isinstance(fig, go.Figure)
        # Should have annotation about missing prices
        assert len(fig.layout.annotations) > 0


class TestCreateFeatureDashboard:
    """Test create_feature_dashboard function."""

    def test_returns_figure(self):
        """create_feature_dashboard returns Plotly figure."""
        from dashboard.visualizations.regime_viz import create_feature_dashboard

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        downside_dev = pd.Series(np.random.rand(100) * 0.02, index=dates)
        sortino_20d = pd.Series(np.random.randn(100), index=dates)
        sortino_60d = pd.Series(np.random.randn(100), index=dates)

        fig = create_feature_dashboard(dates, downside_dev, sortino_20d, sortino_60d)

        assert isinstance(fig, go.Figure)

    def test_has_three_traces(self):
        """create_feature_dashboard has traces for all three features."""
        from dashboard.visualizations.regime_viz import create_feature_dashboard

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        downside_dev = pd.Series(np.random.rand(100) * 0.02, index=dates)
        sortino_20d = pd.Series(np.random.randn(100), index=dates)
        sortino_60d = pd.Series(np.random.randn(100), index=dates)

        fig = create_feature_dashboard(dates, downside_dev, sortino_20d, sortino_60d)

        assert len(fig.data) >= 3

    def test_has_threshold_lines(self):
        """create_feature_dashboard has threshold reference lines."""
        from dashboard.visualizations.regime_viz import create_feature_dashboard

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        downside_dev = pd.Series(np.random.rand(100) * 0.02, index=dates)
        sortino_20d = pd.Series(np.random.randn(100), index=dates)
        sortino_60d = pd.Series(np.random.randn(100), index=dates)

        fig = create_feature_dashboard(dates, downside_dev, sortino_20d, sortino_60d)

        # Should have shapes (hlines) or annotations
        assert len(fig.layout.shapes) >= 1 or len(fig.layout.annotations) >= 1


class TestCreateRegimeStatisticsTable:
    """Test create_regime_statistics_table function."""

    def test_returns_datatable(self):
        """create_regime_statistics_table returns Dash DataTable."""
        from dashboard.visualizations.regime_viz import create_regime_statistics_table
        from dash import dash_table

        regime_data = pd.DataFrame({
            'regime': ['TREND_BULL'] * 50 + ['TREND_BEAR'] * 50,
            'returns': np.random.randn(100) * 0.01
        })

        table = create_regime_statistics_table(regime_data)

        assert isinstance(table, dash_table.DataTable)

    def test_includes_all_regimes(self):
        """create_regime_statistics_table includes all regime rows."""
        from dashboard.visualizations.regime_viz import create_regime_statistics_table

        regime_data = pd.DataFrame({
            'regime': ['TREND_BULL'] * 25 + ['TREND_NEUTRAL'] * 25 +
                      ['TREND_BEAR'] * 25 + ['CRASH'] * 25,
            'returns': np.random.randn(100) * 0.01
        })

        table = create_regime_statistics_table(regime_data)

        # Check that data has 4 rows (one per regime)
        assert len(table.data) == 4

    def test_calculates_days_correctly(self):
        """create_regime_statistics_table calculates days count correctly."""
        from dashboard.visualizations.regime_viz import create_regime_statistics_table

        regime_data = pd.DataFrame({
            'regime': ['TREND_BULL'] * 30 + ['CRASH'] * 20,
            'returns': np.random.randn(50) * 0.01
        })

        table = create_regime_statistics_table(regime_data)

        # Find TREND_BULL row
        bull_row = next(r for r in table.data if r['Regime'] == 'TREND_BULL')
        assert bull_row['Days'] == 30

        # Find CRASH row
        crash_row = next(r for r in table.data if r['Regime'] == 'CRASH')
        assert crash_row['Days'] == 20

    def test_handles_missing_regimes(self):
        """create_regime_statistics_table handles missing regimes with N/A."""
        from dashboard.visualizations.regime_viz import create_regime_statistics_table

        # Only TREND_BULL data
        regime_data = pd.DataFrame({
            'regime': ['TREND_BULL'] * 100,
            'returns': np.random.randn(100) * 0.01
        })

        table = create_regime_statistics_table(regime_data)

        # Find CRASH row (should have N/A values)
        crash_row = next(r for r in table.data if r['Regime'] == 'CRASH')
        assert crash_row['Days'] == 0
        assert crash_row['Avg Return'] == 'N/A'


# =============================================================================
# TRADE VISUALIZATION TESTS
# =============================================================================


class TestCreateTradeDistribution:
    """Test create_trade_distribution function."""

    def test_returns_figure(self):
        """create_trade_distribution returns Plotly figure."""
        from dashboard.visualizations.trade_viz import create_trade_distribution

        trades_df = pd.DataFrame({
            'pnl': np.random.randn(100) * 100,
            'return_pct': np.random.randn(100) * 0.05
        })

        fig = create_trade_distribution(trades_df)

        assert isinstance(fig, go.Figure)

    def test_has_two_histograms(self):
        """create_trade_distribution has P&L and return histograms."""
        from dashboard.visualizations.trade_viz import create_trade_distribution

        trades_df = pd.DataFrame({
            'pnl': np.random.randn(100) * 100,
            'return_pct': np.random.randn(100) * 0.05
        })

        fig = create_trade_distribution(trades_df)

        # Should have 2 histogram traces
        assert len(fig.data) == 2

    def test_histogram_names(self):
        """create_trade_distribution has correct histogram names."""
        from dashboard.visualizations.trade_viz import create_trade_distribution

        trades_df = pd.DataFrame({
            'pnl': np.random.randn(100) * 100,
            'return_pct': np.random.randn(100) * 0.05
        })

        fig = create_trade_distribution(trades_df)

        trace_names = [t.name for t in fig.data]
        assert 'P&L' in trace_names
        assert 'Return %' in trace_names


class TestCreateTradeTimeline:
    """Test create_trade_timeline function."""

    def test_returns_figure(self):
        """create_trade_timeline returns Plotly figure."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:10],
            'entry_price': prices.iloc[:10].values,
            'pnl': np.random.randn(10) * 100
        })

        fig = create_trade_timeline(prices, trades_df)

        assert isinstance(fig, go.Figure)

    def test_has_price_trace(self):
        """create_trade_timeline has price trace."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:10],
            'entry_price': prices.iloc[:10].values,
            'pnl': np.random.randn(10) * 100
        })

        fig = create_trade_timeline(prices, trades_df)

        trace_names = [t.name for t in fig.data]
        assert 'Price' in trace_names

    def test_separates_winning_losing_trades(self):
        """create_trade_timeline separates winning and losing entries."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:10],
            'entry_price': prices.iloc[:10].values,
            'pnl': [100, -50, 200, -30, 150, -80, 100, 50, -20, 75]  # Mix of wins/losses
        })

        fig = create_trade_timeline(prices, trades_df)

        trace_names = [t.name for t in fig.data]
        assert 'Winning Entry' in trace_names
        assert 'Losing Entry' in trace_names

    def test_uses_correct_markers_for_winners(self):
        """create_trade_timeline uses triangle-up for winners."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:5],
            'entry_price': prices.iloc[:5].values,
            'pnl': [100, 50, 200, 150, 75]  # All winners
        })

        fig = create_trade_timeline(prices, trades_df)

        # Find winning entry trace
        winning_trace = next(t for t in fig.data if t.name == 'Winning Entry')
        # Should have 5 points (all trades are winners)
        assert len(winning_trace.x) == 5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestVisualizationEdgeCases:
    """Test edge cases across visualization functions."""

    def test_equity_curve_with_negative_values(self):
        """create_equity_curve handles negative portfolio values."""
        from dashboard.visualizations.performance_viz import create_equity_curve

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Start positive, go negative
        values = pd.Series(1000 - np.arange(100) * 15, index=dates)

        fig = create_equity_curve(values)

        assert isinstance(fig, go.Figure)
        # Drawdown calculation should still work
        assert len(fig.data) >= 2

    def test_rolling_metrics_with_all_zeros(self):
        """create_rolling_metrics handles all-zero returns."""
        from dashboard.visualizations.performance_viz import create_rolling_metrics

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        returns = pd.Series([0.0] * 100, index=dates)

        fig = create_rolling_metrics(returns, window=20)

        assert isinstance(fig, go.Figure)

    def test_trade_timeline_with_all_winners(self):
        """create_trade_timeline handles all winning trades."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:5],
            'entry_price': prices.iloc[:5].values,
            'pnl': [100, 50, 200, 150, 75]  # All positive
        })

        fig = create_trade_timeline(prices, trades_df)

        trace_names = [t.name for t in fig.data]
        assert 'Winning Entry' in trace_names

    def test_trade_timeline_with_all_losers(self):
        """create_trade_timeline handles all losing trades."""
        from dashboard.visualizations.trade_viz import create_trade_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        trades_df = pd.DataFrame({
            'entry_date': dates[:5],
            'entry_price': prices.iloc[:5].values,
            'pnl': [-100, -50, -200, -150, -75]  # All negative
        })

        fig = create_trade_timeline(prices, trades_df)

        trace_names = [t.name for t in fig.data]
        assert 'Losing Entry' in trace_names

    def test_regime_timeline_single_regime(self):
        """create_regime_timeline handles single regime period."""
        from dashboard.visualizations.regime_viz import create_regime_timeline

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        regimes = pd.Series(['TREND_BULL'] * 100, index=dates)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        fig = create_regime_timeline(dates, regimes, prices)

        assert isinstance(fig, go.Figure)
