"""
Risk Management Visualizations

Portfolio heat gauges, risk metrics, and position allocation charts.
"""

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table
from typing import Optional

from dashboard.config import COLORS, PERFORMANCE_THRESHOLDS


def create_portfolio_heat_gauge(
    current_heat: float,
    max_heat: float = 0.08
) -> go.Figure:
    """
    Create gauge showing portfolio heat vs limit.

    Args:
        current_heat: Current portfolio heat (0-1 scale)
        max_heat: Maximum allowed heat

    Returns:
        Plotly gauge figure
    """

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=current_heat * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Portfolio Heat', 'font': {'size': 24, 'color': COLORS['text_primary']}},
        delta={'reference': max_heat * 100, 'increasing': {'color': COLORS['danger']}},
        gauge={
            'axis': {
                'range': [None, max_heat * 100],
                'ticksuffix': '%',
                'tickcolor': COLORS['text_primary']
            },
            'bar': {'color': COLORS['info']},
            'steps': [
                {'range': [0, max_heat * 0.5 * 100], 'color': COLORS['success']},
                {'range': [max_heat * 0.5 * 100, max_heat * 0.75 * 100], 'color': COLORS['warning']},
                {'range': [max_heat * 0.75 * 100, max_heat * 100], 'color': COLORS['bear_secondary']}
            ],
            'threshold': {
                'line': {'color': COLORS['danger'], 'width': 4},
                'thickness': 0.75,
                'value': max_heat * 100
            }
        }
    ))

    fig.update_layout(
        height=400,
        paper_bgcolor=COLORS['background_dark'],
        font={'size': 18, 'color': COLORS['text_primary']}
    )

    return fig


def create_risk_metrics_table() -> dash_table.DataTable:
    """
    Create risk metrics summary table.

    Returns:
        Dash DataTable
    """

    metrics = [
        {'Metric': 'Portfolio Heat', 'Current': '4.2%', 'Limit': '8.0%', 'Status': '✓'},
        {'Metric': 'Max Position Size', 'Current': '2.1%', 'Limit': '5.0%', 'Status': '✓'},
        {'Metric': 'Daily Drawdown', 'Current': '-0.8%', 'Limit': '-3.0%', 'Status': '✓'},
        {'Metric': 'VaR (95%)', 'Current': '-2.1%', 'Limit': '-5.0%', 'Status': '✓'},
        {'Metric': 'Open Positions', 'Current': '3', 'Limit': '5', 'Status': '✓'}
    ]

    return dash_table.DataTable(
        data=metrics,
        columns=[{'name': i, 'id': i} for i in metrics[0].keys()],
        style_data_conditional=[
            {
                'if': {'filter_query': '{Status} = "✗"', 'column_id': 'Status'},
                'backgroundColor': 'rgba(255, 0, 0, 0.3)',
                'color': COLORS['danger']
            },
            {
                'if': {'filter_query': '{Status} = "✓"', 'column_id': 'Status'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'color': COLORS['success']
            }
        ],
        style_header={
            'backgroundColor': COLORS['background_medium'],
            'color': COLORS['text_primary'],
            'fontWeight': 'bold',
            'border': f'1px solid {COLORS["grid"]}'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'backgroundColor': COLORS['background_dark'],
            'color': COLORS['text_primary'],
            'border': f'1px solid {COLORS["grid"]}'
        }
    )
