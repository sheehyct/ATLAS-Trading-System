"""
Dashboard Header Component

Top navigation bar with branding, status indicators, and current regime display.
"""

import dash_bootstrap_components as dbc
from dash import html
from dashboard.config import COLORS


def create_header():
    """
    Create dashboard header with title and status indicators.

    Returns:
        Bootstrap navbar component
    """

    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                # Logo and Title
                dbc.Col([
                    html.H3([
                        html.I(className='fas fa-chart-line me-2', style={'color': COLORS['bull_primary']}),
                        'ATLAS Trading Dashboard'
                    ], className='mb-0 text-white')
                ], width='auto'),

                # Status Indicators
                dbc.Col([
                    html.Div([
                        html.Span('LIVE', className='badge bg-success me-2', id='live-status-badge'),
                        html.Span('CONNECTED', className='badge bg-info me-2', id='connection-badge'),
                        html.Span('NEUTRAL', className='badge bg-secondary me-2', id='current-regime-badge'),
                        html.Span(id='market-time-badge', className='badge bg-dark')
                    ], className='d-flex align-items-center')
                ], width='auto', className='ms-auto')
            ], align='center', className='g-0 w-100', justify='between')
        ], fluid=True),
        color='dark',
        dark=True,
        className='mb-4 shadow',
        style={'backgroundColor': COLORS['background_medium']}
    )
