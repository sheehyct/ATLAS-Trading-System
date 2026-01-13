"""
Risk Management Panel Component

Risk monitoring with heat gauges, metrics tables, and limit tracking.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from dashboard.config import COLORS


def create_risk_panel():
    """
    Create risk management visualization panel.

    Layout:
    - Portfolio heat gauge
    - Risk metrics table
    - Position allocation chart
    - Risk limit indicators

    Returns:
        Bootstrap container with risk visualizations
    """

    return dbc.Container([

        # Risk Summary Cards
        dbc.Row([
            # Portfolio Heat Gauge
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-thermometer-half me-2'),
                        'Portfolio Heat Monitor'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='risk-heat-gauge',
                            config={'displayModeBar': False, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=6, className='mb-3'),

            # Risk Metrics Table
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-shield-alt me-2'),
                        'Risk Metrics'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        html.Div(id='risk-metrics-table')
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=6, className='mb-3')
        ], className='mb-4'),

        # Position Allocation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-pie me-2'),
                        'Position Allocation'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='position-allocation-chart',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12)
        ], className='mb-4'),

        # Risk Alerts - Thresholds from config.PERFORMANCE_THRESHOLDS
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className='fas fa-exclamation-triangle me-2'),
                    html.Span([
                        html.Strong('Risk Limits: '),
                        'Portfolio Heat < 8% (2R total) | ',
                        'Max Position < 12% | ',
                        'Daily Loss < 3%'
                    ])
                ], color='warning', className='mb-0'),
                html.Small(
                    'Note: Portfolio heat is approximated using max position concentration. '
                    'True heat requires stop loss data from signal store.',
                    className='text-muted d-block mt-1',
                    style={'fontSize': '0.75rem'}
                )
            ], width=12)
        ])

    ], fluid=True)
