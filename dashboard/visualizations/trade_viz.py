"""
Trade Analysis Visualizations

Trade distribution, timeline, and analysis charts.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from dashboard.config import COLORS, TRADE_MARKERS


def create_trade_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """
    Create trade P&L distribution histograms.

    Args:
        trades_df: DataFrame with columns ['pnl', 'return_pct']

    Returns:
        Plotly figure
    """

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P&L Distribution', 'Return % Distribution')
    )

    # P&L histogram
    fig.add_trace(
        go.Histogram(
            x=trades_df['pnl'],
            nbinsx=30,
            name='P&L',
            marker_color=COLORS['info'],
            opacity=0.7
        ),
        row=1, col=1
    )

    # Return % histogram
    fig.add_trace(
        go.Histogram(
            x=trades_df['return_pct'] * 100,
            nbinsx=30,
            name='Return %',
            marker_color=COLORS['bull_secondary'],
            opacity=0.7
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=500,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        showlegend=False
    )

    fig.update_xaxes(title_text='P&L ($)', row=1, col=1, gridcolor=COLORS['grid'])
    fig.update_xaxes(title_text='Return (%)', row=1, col=2, gridcolor=COLORS['grid'])

    return fig


def create_trade_timeline(
    price_data: pd.Series,
    trades_df: pd.DataFrame
) -> go.Figure:
    """
    Show trades overlaid on price chart.

    Args:
        price_data: Price series
        trades_df: DataFrame with trade details

    Returns:
        Plotly figure
    """

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data,
            mode='lines',
            name='Price',
            line=dict(color=COLORS['price_line'], width=1)
        )
    )

    # Winning/losing trades
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    # Winning entries
    fig.add_trace(
        go.Scatter(
            x=winning_trades['entry_date'],
            y=winning_trades['entry_price'],
            mode='markers',
            name='Winning Entry',
            marker=TRADE_MARKERS['long_entry']
        )
    )

    # Losing entries
    fig.add_trace(
        go.Scatter(
            x=losing_trades['entry_date'],
            y=losing_trades['entry_price'],
            mode='markers',
            name='Losing Entry',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color=COLORS['bear_primary'],
                line=dict(color='#FFFFFF', width=2),
                opacity=0.6
            )
        )
    )

    fig.update_layout(
        height=600,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        hovermode='closest'
    )

    fig.update_yaxes(title_text='Price ($)', gridcolor=COLORS['grid'])
    fig.update_xaxes(title_text='Date', gridcolor=COLORS['grid'])

    return fig
