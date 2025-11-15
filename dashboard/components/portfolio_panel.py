"""
Live Portfolio Panel Component

Real-time portfolio monitoring with positions, P&L, and account status.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dashboard.config import COLORS


def create_portfolio_panel():
    """
    Create live portfolio monitoring panel.

    Layout:
    - Portfolio value card and P&L
    - Current positions table
    - Portfolio heat gauge
    - Account metrics

    Returns:
        Bootstrap container with portfolio visualizations
    """

    return dbc.Container([

        # Portfolio Summary Cards
        dbc.Row([
            # Portfolio Value
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-wallet me-2'),
                        'Portfolio Value'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody(id='portfolio-value-card', style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=4, className='mb-3'),

            # Portfolio Heat Gauge
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-thermometer-half me-2'),
                        'Portfolio Heat'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dcc.Graph(
                            id='portfolio-heat-gauge',
                            config={'displayModeBar': False, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12, lg=8, className='mb-3')
        ], className='mb-4'),

        # Current Positions Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-list me-2'),
                        'Current Positions'
                    ], className='fw-bold', style={'backgroundColor': COLORS['background_medium']}),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id='positions-table',
                            columns=[
                                {'name': 'Symbol', 'id': 'symbol'},
                                {'name': 'Qty', 'id': 'qty'},
                                {'name': 'Market Value', 'id': 'market_value', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                {'name': 'Unrealized P&L', 'id': 'unrealized_pl', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                                {'name': 'Unrealized P&L %', 'id': 'unrealized_plpc', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                {'name': 'Avg Entry', 'id': 'avg_entry_price', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                            ],
                            data=[],
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
                            },
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{unrealized_pl} > 0', 'column_id': 'unrealized_pl'},
                                    'color': COLORS['bull_primary']
                                },
                                {
                                    'if': {'filter_query': '{unrealized_pl} < 0', 'column_id': 'unrealized_pl'},
                                    'color': COLORS['bear_primary']
                                },
                                {
                                    'if': {'filter_query': '{unrealized_plpc} > 0', 'column_id': 'unrealized_plpc'},
                                    'color': COLORS['bull_primary']
                                },
                                {
                                    'if': {'filter_query': '{unrealized_plpc} < 0', 'column_id': 'unrealized_plpc'},
                                    'color': COLORS['bear_primary']
                                }
                            ]
                        )
                    ], style={'backgroundColor': COLORS['background_dark']})
                ], className='shadow-sm', style={'borderColor': COLORS['grid']})
            ], width=12)
        ], className='mb-4'),

        # Live Update Status
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className='fas fa-sync-alt me-2'),
                    'Live data updates every 30 seconds. ',
                    html.Span('Last update: ', className='text-muted'),
                    html.Span(id='last-update-time', className='fw-bold')
                ], color='info', className='mb-0')
            ], width=12)
        ])

    ], fluid=True)
