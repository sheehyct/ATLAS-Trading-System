"""
Regime Detection Panel Component

Layer 1 (Regime Detection) visualization panel with timeline, features, and statistics.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from dashboard.config import COLORS


def create_regime_panel():
    """
    Create regime detection visualization panel.

    Layout:
    - Full-width regime timeline chart
    - 2-column layout: feature dashboard (left) + statistics table (right)

    Returns:
        Bootstrap container with regime visualizations
    """

    return dbc.Container([

        # Main Regime Timeline
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-area me-2'),
                        'Regime Timeline'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='regime-timeline-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12)
        ], className='mb-4'),

        # Feature Dashboard and Statistics
        dbc.Row([
            # Feature Evolution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-wave-square me-2'),
                        'Feature Evolution'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='feature-dashboard-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=8),

            # Regime Statistics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-table me-2'),
                        'Regime Statistics'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        html.Div(id='regime-stats-table')
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=4)
        ], className='mb-4'),

        # Info Card
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className='fas fa-info-circle me-2'),
                    'Layer 1 (Regime Detection) uses academic statistical jump models from Princeton University. ',
                    'Regimes: TREND_BULL (green), TREND_NEUTRAL (gray), TREND_BEAR (orange), CRASH (red).'
                ], color='info', className='mb-0')
            ], width=12)
        ])

    ], fluid=True)
