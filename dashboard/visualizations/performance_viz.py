"""
Strategy Performance Visualizations

Equity curves, drawdowns, rolling metrics, and strategy comparisons.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from dashboard.config import COLORS, CHART_HEIGHT, PERFORMANCE_THRESHOLDS, FONTS


def create_equity_curve(
    portfolio_value: pd.Series,
    benchmark_value: Optional[pd.Series] = None
) -> go.Figure:
    """
    Create equity curve with drawdown subplot.

    Args:
        portfolio_value: Portfolio value over time
        benchmark_value: Optional benchmark for comparison

    Returns:
        Plotly figure with equity and drawdown
    """

    # Calculate drawdown
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Portfolio equity
    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value,
            mode='lines',
            name='Strategy',
            line=dict(color=COLORS['bull_primary'], width=2)
        ),
        row=1, col=1
    )

    # Benchmark (if provided)
    if benchmark_value is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_value.index,
                y=benchmark_value,
                mode='lines',
                name='Benchmark',
                line=dict(color=COLORS['text_secondary'], width=1, dash='dash')
            ),
            row=1, col=1
        )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color=COLORS['bear_primary'], width=1),
            fillcolor=COLORS['bear_fill']
        ),
        row=2, col=1
    )

    # Layout (Premium Luxury)
    fig.update_layout(
        height=CHART_HEIGHT['equity_curve'],
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font=dict(
            family=FONTS['body'],
            size=12,
            color=COLORS['text_primary'],
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=COLORS['bg_card'],
            bordercolor=COLORS['border_default'],
            font=dict(family=FONTS['body'], size=12),
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        ),
    )

    fig.update_yaxes(
        title_text='Portfolio Value ($)',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        gridcolor=COLORS['grid'],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Drawdown (%)',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        gridcolor=COLORS['grid'],
        row=2, col=1
    )
    fig.update_xaxes(
        title_text='Date',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        gridcolor=COLORS['grid'],
        row=2, col=1
    )

    return fig


def create_rolling_metrics(
    returns: pd.Series,
    window: int = 60
) -> go.Figure:
    """
    Create rolling performance metrics chart.

    Args:
        returns: Daily returns
        window: Rolling window size

    Returns:
        Plotly figure
    """

    # Calculate rolling metrics
    rolling_sharpe = (
        returns.rolling(window).mean() /
        returns.rolling(window).std() *
        np.sqrt(252)
    )

    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_sortino = (
        returns.rolling(window).mean() /
        downside_returns.rolling(window).std() *
        np.sqrt(252)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=rolling_sharpe,
            mode='lines',
            name=f'Rolling Sharpe ({window}d)',
            line=dict(color=COLORS['info'], width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=rolling_sortino,
            mode='lines',
            name=f'Rolling Sortino ({window}d)',
            line=dict(color=COLORS['bull_secondary'], width=2)
        )
    )

    # Reference lines
    fig.add_hline(
        y=PERFORMANCE_THRESHOLDS['sharpe_good'],
        line_dash='dash',
        line_color=COLORS['text_tertiary'],
        annotation_text='Good (1.0)',
        annotation_font=dict(family=FONTS['body'], size=10, color=COLORS['text_tertiary']),
    )

    # Layout (Premium Luxury)
    fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font=dict(
            family=FONTS['body'],
            size=12,
            color=COLORS['text_primary'],
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=COLORS['bg_card'],
            bordercolor=COLORS['border_default'],
            font=dict(family=FONTS['body'], size=12),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        ),
    )

    fig.update_yaxes(
        title_text='Ratio',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        gridcolor=COLORS['grid'],
    )
    fig.update_xaxes(
        title_text='Date',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        gridcolor=COLORS['grid'],
    )

    return fig
