"""
Strategy Performance Panel Component

Strategy backtest analysis with equity curves, metrics, and trade distribution.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from dashboard.config import COLORS, AVAILABLE_STRATEGIES


def create_strategy_panel():
    """
    Create strategy performance visualization panel.

    Layout:
    - Strategy selector dropdown
    - Equity curve with drawdown
    - Rolling metrics and regime comparison
    - Trade distribution analysis

    Returns:
        Bootstrap container with strategy visualizations
    """

    # Strategy options for dbc.Select (uses label/value format)
    strategy_options = [
        {'label': strategy['name'], 'value': key}
        for key, strategy in AVAILABLE_STRATEGIES.items()
    ]

    return dbc.Container([

        # Strategy Selector - DB-3: Switched from dcc.Dropdown to dbc.Select
        # to fix dropdown clipping and dark theme rendering issues
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Label('Select Strategy:', className='fw-bold mb-2',
                                   style={'color': COLORS['text_primary']}),
                        dbc.Select(
                            id='strategy-selector',
                            options=strategy_options,
                            value='strat_options',
                            style={
                                'backgroundColor': COLORS['bg_surface'],
                                'color': COLORS['text_primary'],
                                'border': f'1px solid {COLORS["border_default"]}',
                            }
                        )
                    ])
                ], className='shadow-sm', style={
                    'backgroundColor': COLORS['bg_card'],
                    'border': f'1px solid {COLORS["border_subtle"]}'
                })
            ], width=12, lg=6)
        ], className='mb-4'),

        # Equity Curve
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-line me-2'),
                        'Equity Curve & Drawdown'
                    ], className='fw-bold', style={'backgroundColor': COLORS['bg_card']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='equity-curve-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['bg_void']})
                ], className='shadow-sm', style={'borderColor': COLORS['border_subtle']})
            ], width=12)
        ], className='mb-4'),

        # Rolling Metrics and Regime Comparison
        dbc.Row([
            # Rolling Metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-bar me-2'),
                        'Rolling Performance Metrics'
                    ], className='fw-bold', style={'backgroundColor': COLORS['bg_card']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='rolling-metrics-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['bg_void']})
                ], className='shadow-sm', style={'borderColor': COLORS['border_subtle']})
            ], width=12, lg=6),

            # Regime Comparison
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-layer-group me-2'),
                        'Performance by Regime'
                    ], className='fw-bold', style={'backgroundColor': COLORS['bg_card']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='regime-comparison-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['bg_void']})
                ], className='shadow-sm', style={'borderColor': COLORS['border_subtle']})
            ], width=12, lg=6)
        ], className='mb-4'),

        # Trade Distribution
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-pie me-2'),
                        'Trade Distribution'
                    ], className='fw-bold', style={'backgroundColor': COLORS['bg_card']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='trade-distribution-graph',
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['bg_void']})
                ], className='shadow-sm', style={'borderColor': COLORS['border_subtle']})
            ], width=12)
        ])

    ], fluid=True)
