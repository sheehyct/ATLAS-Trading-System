"""
Options Trading Panel Component

Professional dark-themed options trading panel with STRAT integration featuring:
- Active options trades with P&L tracking
- Progress bars to profit targets (measured move)
- Order execution interface via Alpaca Options API
- Pattern-based trade signals from STRAT Tier 1

Session 76: Full options trading tab with progress visualization.

BROKER SUPPORT:
- Alpaca: Full options support in both Paper and Live environments
  - Paper trading options enabled by default
  - Uses alpaca-py SDK for order execution
  - Supports market, limit, and stop orders for options
- See: https://docs.alpaca.markets/docs/options-trading

"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from dashboard.config import COLORS, CHART_HEIGHT


# ============================================
# DARK THEME CONFIGURATION
# ============================================

DARK_THEME = {
    'background': '#090008',           # Near black
    'card_bg': '#1a1a2e',              # Slightly lighter for cards
    'card_header': '#16213e',          # Card headers
    'input_bg': '#0f0f1a',             # Input fields
    'border': '#333344',               # Borders
    'text_primary': '#e0e0e0',         # Primary text
    'text_secondary': '#a0a0a0',       # Secondary text
    'text_muted': '#666677',           # Muted text
    'accent_green': '#00ff55',         # Bull/profit
    'accent_red': '#ed4807',           # Bear/loss
    'accent_yellow': '#ffd700',        # Warning/outside bar
    'accent_blue': '#2196f3',          # Info/neutral
    'progress_bg': '#2a2a3e',          # Progress bar background
}


# ============================================
# OPTIONS TRADING PANEL
# ============================================

def create_options_panel():
    """
    Create full options trading panel with dark professional theme.

    Layout:
    - Row 1: Trade Entry Form | Current Regime Signal
    - Row 2: Active Options Trades Table with Progress Bars
    - Row 3: Trade Progress Visualization | P&L Summary

    Returns:
        Bootstrap container with options trading interface
    """

    return dbc.Container([

        # ============================================
        # ROW 1: Trade Entry + Regime Signal
        # ============================================
        dbc.Row([

            # Left: Trade Entry Form
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-edit me-2', style={'color': DARK_THEME['accent_green']}),
                        'New Options Trade'
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([

                        # Symbol Input
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Symbol', className='text-light mb-1'),
                                dbc.Input(
                                    id='options-symbol-input',
                                    type='text',
                                    placeholder='SPY',
                                    value='SPY',
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),

                            dbc.Col([
                                dbc.Label('Direction', className='text-light mb-1'),
                                dbc.Select(
                                    id='options-direction-select',
                                    options=[
                                        {'label': '📈 CALL (Bullish)', 'value': 'call'},
                                        {'label': '📉 PUT (Bearish)', 'value': 'put'}
                                    ],
                                    value='call',
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),

                            dbc.Col([
                                dbc.Label('Contracts', className='text-light mb-1'),
                                dbc.Input(
                                    id='options-quantity-input',
                                    type='number',
                                    min=1,
                                    max=10,
                                    value=1,
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),
                        ], className='mb-3'),

                        # Strike and Expiration
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Strike Price', className='text-light mb-1'),
                                dbc.Input(
                                    id='options-strike-input',
                                    type='number',
                                    placeholder='Auto from STRAT',
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),

                            dbc.Col([
                                dbc.Label('Expiration', className='text-light mb-1'),
                                dbc.Input(
                                    id='options-expiry-input',
                                    type='date',
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),

                            dbc.Col([
                                dbc.Label('Entry Price', className='text-light mb-1'),
                                dbc.Input(
                                    id='options-entry-input',
                                    type='number',
                                    step=0.01,
                                    placeholder='Premium $',
                                    style={
                                        'backgroundColor': DARK_THEME['input_bg'],
                                        'color': DARK_THEME['text_primary'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }
                                )
                            ], width=4),
                        ], className='mb-3'),

                        # Target and Stop from STRAT Pattern
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('Pattern Target', className='text-light mb-1'),
                                dbc.InputGroup([
                                    dbc.InputGroupText('$', style={
                                        'backgroundColor': DARK_THEME['card_header'],
                                        'color': DARK_THEME['accent_green'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }),
                                    dbc.Input(
                                        id='options-target-input',
                                        type='number',
                                        step=0.01,
                                        placeholder='Measured Move',
                                        style={
                                            'backgroundColor': DARK_THEME['input_bg'],
                                            'color': DARK_THEME['accent_green'],
                                            'border': f'1px solid {DARK_THEME["border"]}'
                                        }
                                    )
                                ])
                            ], width=6),

                            dbc.Col([
                                dbc.Label('Stop Loss', className='text-light mb-1'),
                                dbc.InputGroup([
                                    dbc.InputGroupText('$', style={
                                        'backgroundColor': DARK_THEME['card_header'],
                                        'color': DARK_THEME['accent_red'],
                                        'border': f'1px solid {DARK_THEME["border"]}'
                                    }),
                                    dbc.Input(
                                        id='options-stop-input',
                                        type='number',
                                        step=0.01,
                                        placeholder='Pattern Stop',
                                        style={
                                            'backgroundColor': DARK_THEME['input_bg'],
                                            'color': DARK_THEME['accent_red'],
                                            'border': f'1px solid {DARK_THEME["border"]}'
                                        }
                                    )
                                ])
                            ], width=6),
                        ], className='mb-3'),

                        # Action Buttons
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.I(className='fas fa-magic me-2'),
                                    'Load STRAT Signal'
                                ],
                                    id='load-strat-signal-btn',
                                    color='info',
                                    className='w-100',
                                    outline=True
                                )
                            ], width=6),

                            dbc.Col([
                                dbc.Button([
                                    html.I(className='fas fa-paper-plane me-2'),
                                    'Submit Trade'
                                ],
                                    id='submit-options-trade-btn',
                                    color='success',
                                    className='w-100'
                                )
                            ], width=6),
                        ]),

                        # Alpaca Options Info
                        dbc.Alert([
                            html.I(className='fas fa-check-circle me-2'),
                            html.Strong('Alpaca Paper Trading: '),
                            'Options trading enabled by default. ',
                            'Orders execute via Alpaca Options API. ',
                            html.A('Docs', href='https://docs.alpaca.markets/docs/options-trading',
                                   target='_blank', className='alert-link')
                        ],
                            color='success',
                            className='mt-3 mb-0',
                            style={'fontSize': '0.8rem'}
                        )

                    ], style={'backgroundColor': DARK_THEME['card_bg']})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}'
                }, className='shadow')
            ], width=12, lg=7, className='mb-3'),

            # Right: Current STRAT Signal
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-crosshairs me-2', style={'color': DARK_THEME['accent_yellow']}),
                        'Active STRAT Signal'
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        html.Div(id='strat-signal-display', children=[
                            _create_strat_signal_placeholder()
                        ])
                    ], style={'backgroundColor': DARK_THEME['card_bg']})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}',
                    'height': '100%'
                }, className='shadow')
            ], width=12, lg=5, className='mb-3'),

        ], className='mb-4'),

        # ============================================
        # ROW 2: Active Options Trades with Progress
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-list-ol me-2', style={'color': DARK_THEME['accent_blue']}),
                        'Active Options Trades',
                        dbc.Badge('3', color='success', className='ms-2')  # Trade count
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        html.Div(id='active-options-trades', children=[
                            _create_sample_trades_table()
                        ])
                    ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '0'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}'
                }, className='shadow')
            ], width=12)
        ], className='mb-4'),

        # ============================================
        # ROW 3: Trade Progress Chart + P&L Summary
        # ============================================
        dbc.Row([

            # Left: Trade Progress Visualization
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-line me-2', style={'color': DARK_THEME['accent_green']}),
                        'Trade Progress to Target'
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='trade-progress-chart',
                            figure=create_trade_progress_chart(),
                            config={'displayModeBar': False, 'displaylogo': False}
                        )
                    ], style={'backgroundColor': DARK_THEME['card_bg']})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}'
                }, className='shadow')
            ], width=12, lg=8, className='mb-3'),

            # Right: P&L Summary
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-dollar-sign me-2', style={'color': DARK_THEME['accent_green']}),
                        'Options P&L Summary'
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        _create_pnl_summary_cards()
                    ], style={'backgroundColor': DARK_THEME['card_bg']})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}',
                    'height': '100%'
                }, className='shadow')
            ], width=12, lg=4, className='mb-3'),

        ]),

        # Hidden storage for trade data
        dcc.Store(id='options-trades-store', data=[]),

    ], fluid=True, style={'backgroundColor': DARK_THEME['background']})


# ============================================
# HELPER COMPONENTS
# ============================================

def _create_strat_signal_placeholder():
    """Create placeholder for STRAT signal display."""
    return html.Div([
        # Pattern Type Badge
        html.Div([
            dbc.Badge('2-1-2 Up', color='success', className='me-2 mb-2', style={'fontSize': '1rem'}),
            dbc.Badge('WEEKLY', color='info', className='mb-2', style={'fontSize': '0.8rem'}),
        ]),

        # Signal Details
        html.Table([
            html.Tr([
                html.Td('Symbol:', style={'color': DARK_THEME['text_secondary'], 'paddingRight': '1rem'}),
                html.Td('SPY', style={'color': DARK_THEME['text_primary'], 'fontWeight': 'bold'})
            ]),
            html.Tr([
                html.Td('Entry:', style={'color': DARK_THEME['text_secondary']}),
                html.Td('$598.50', style={'color': DARK_THEME['text_primary']})
            ]),
            html.Tr([
                html.Td('Target:', style={'color': DARK_THEME['text_secondary']}),
                html.Td('$612.00', style={'color': DARK_THEME['accent_green'], 'fontWeight': 'bold'})
            ]),
            html.Tr([
                html.Td('Stop:', style={'color': DARK_THEME['text_secondary']}),
                html.Td('$592.00', style={'color': DARK_THEME['accent_red']})
            ]),
            html.Tr([
                html.Td('R:R:', style={'color': DARK_THEME['text_secondary']}),
                html.Td('2.08', style={'color': DARK_THEME['accent_blue']})
            ]),
            html.Tr([
                html.Td('Regime:', style={'color': DARK_THEME['text_secondary']}),
                html.Td([
                    html.Span('●', style={'color': DARK_THEME['accent_green'], 'marginRight': '0.5rem'}),
                    'TREND_BULL'
                ], style={'color': DARK_THEME['accent_green']})
            ]),
        ], style={'width': '100%'}),

        # Suggested Option
        html.Hr(style={'borderColor': DARK_THEME['border']}),
        html.P([
            html.I(className='fas fa-lightbulb me-2', style={'color': DARK_THEME['accent_yellow']}),
            html.Strong('Suggested: '),
            'SPY Jan 17 $600 CALL @ ~$8.50'
        ], style={'color': DARK_THEME['text_secondary'], 'fontSize': '0.9rem', 'marginBottom': '0'}),

    ], style={'padding': '0.5rem'})


def _create_sample_trades_table():
    """Create sample options trades table with progress bars."""

    # Sample trade data
    trades = [
        {
            'symbol': 'SPY',
            'contract': 'SPY 12/20 $600C',
            'qty': 2,
            'entry': 5.50,
            'current': 7.20,
            'target_underlying': 612.00,
            'current_underlying': 605.30,
            'entry_underlying': 598.50,
            'stop_underlying': 592.00,
            'pnl': 340.00,
            'pnl_pct': 30.9,
        },
        {
            'symbol': 'QQQ',
            'contract': 'QQQ 12/27 $520C',
            'qty': 1,
            'entry': 8.20,
            'current': 6.80,
            'target_underlying': 535.00,
            'current_underlying': 515.40,
            'entry_underlying': 512.00,
            'stop_underlying': 505.00,
            'pnl': -140.00,
            'pnl_pct': -17.1,
        },
        {
            'symbol': 'AAPL',
            'contract': 'AAPL 01/17 $230C',
            'qty': 3,
            'entry': 4.30,
            'current': 5.85,
            'target_underlying': 245.00,
            'current_underlying': 235.80,
            'entry_underlying': 228.00,
            'stop_underlying': 222.00,
            'pnl': 465.00,
            'pnl_pct': 36.0,
        },
    ]

    rows = []
    for trade in trades:
        # Calculate progress percentage to target
        total_move = trade['target_underlying'] - trade['entry_underlying']
        current_move = trade['current_underlying'] - trade['entry_underlying']
        progress_pct = min(100, max(0, (current_move / total_move) * 100)) if total_move != 0 else 0

        # P&L color
        pnl_color = DARK_THEME['accent_green'] if trade['pnl'] >= 0 else DARK_THEME['accent_red']

        # Progress bar color based on position
        if progress_pct >= 75:
            progress_color = 'success'
        elif progress_pct >= 25:
            progress_color = 'info'
        else:
            progress_color = 'warning'

        rows.append(
            html.Tr([
                # Contract
                html.Td([
                    html.Div(trade['contract'], style={'fontWeight': 'bold', 'color': DARK_THEME['text_primary']}),
                    html.Small(f"Qty: {trade['qty']}", style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '0.75rem'}),

                # Entry/Current
                html.Td([
                    html.Div(f"${trade['entry']:.2f} → ${trade['current']:.2f}",
                             style={'color': DARK_THEME['text_primary']}),
                    html.Small(f"Underlying: ${trade['current_underlying']:.2f}",
                               style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '0.75rem'}),

                # Progress to Target
                html.Td([
                    html.Div([
                        html.Span(f"{progress_pct:.0f}% to target",
                                  style={'color': DARK_THEME['text_primary'], 'fontSize': '0.9rem'}),
                    ]),
                    dbc.Progress(
                        value=progress_pct,
                        color=progress_color,
                        className='mt-1',
                        style={
                            'height': '8px',
                            'backgroundColor': DARK_THEME['progress_bg']
                        }
                    ),
                    html.Small(
                        f"${trade['entry_underlying']:.0f} → ${trade['target_underlying']:.0f}",
                        style={'color': DARK_THEME['text_muted']}
                    )
                ], style={'padding': '0.75rem', 'minWidth': '180px'}),

                # P&L
                html.Td([
                    html.Div(f"${trade['pnl']:+,.2f}",
                             style={'color': pnl_color, 'fontWeight': 'bold', 'fontSize': '1.1rem'}),
                    html.Small(f"({trade['pnl_pct']:+.1f}%)", style={'color': pnl_color})
                ], style={'padding': '0.75rem', 'textAlign': 'right'}),

                # Actions
                html.Td([
                    dbc.ButtonGroup([
                        dbc.Button(
                            html.I(className='fas fa-times'),
                            color='danger',
                            size='sm',
                            outline=True,
                            title='Close Trade'
                        ),
                        dbc.Button(
                            html.I(className='fas fa-edit'),
                            color='info',
                            size='sm',
                            outline=True,
                            title='Modify'
                        ),
                    ], size='sm')
                ], style={'padding': '0.75rem', 'textAlign': 'center'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th('Contract', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Entry → Current', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Progress to Target', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('P&L', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem', 'textAlign': 'right'}),
                html.Th('Actions', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem', 'textAlign': 'center'}),
            ], style={'backgroundColor': DARK_THEME['card_header']})
        ]),
        html.Tbody(rows)
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_pnl_summary_cards():
    """Create P&L summary cards."""
    return html.Div([

        # Total P&L
        html.Div([
            html.P('Total P&L', style={
                'color': DARK_THEME['text_secondary'],
                'marginBottom': '0.25rem',
                'fontSize': '0.9rem'
            }),
            html.H3('$665.00', style={
                'color': DARK_THEME['accent_green'],
                'marginBottom': '0',
                'fontWeight': 'bold'
            }),
            html.Small('+18.3%', style={'color': DARK_THEME['accent_green']})
        ], style={
            'backgroundColor': DARK_THEME['card_header'],
            'padding': '1rem',
            'borderRadius': '8px',
            'marginBottom': '1rem',
            'border': f'1px solid {DARK_THEME["border"]}'
        }),

        # Win/Loss Stats
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P('Winning', style={
                        'color': DARK_THEME['text_secondary'],
                        'marginBottom': '0.25rem',
                        'fontSize': '0.8rem'
                    }),
                    html.H5('2', style={
                        'color': DARK_THEME['accent_green'],
                        'marginBottom': '0'
                    }),
                ], style={
                    'backgroundColor': DARK_THEME['card_header'],
                    'padding': '0.75rem',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'border': f'1px solid {DARK_THEME["border"]}'
                })
            ], width=6),

            dbc.Col([
                html.Div([
                    html.P('Losing', style={
                        'color': DARK_THEME['text_secondary'],
                        'marginBottom': '0.25rem',
                        'fontSize': '0.8rem'
                    }),
                    html.H5('1', style={
                        'color': DARK_THEME['accent_red'],
                        'marginBottom': '0'
                    }),
                ], style={
                    'backgroundColor': DARK_THEME['card_header'],
                    'padding': '0.75rem',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'border': f'1px solid {DARK_THEME["border"]}'
                })
            ], width=6),
        ], className='mb-3'),

        # Largest Win/Loss
        html.Div([
            html.Div([
                html.Span('Largest Win: ', style={'color': DARK_THEME['text_secondary']}),
                html.Span('$465.00', style={'color': DARK_THEME['accent_green'], 'fontWeight': 'bold'}),
            ], style={'marginBottom': '0.5rem'}),
            html.Div([
                html.Span('Largest Loss: ', style={'color': DARK_THEME['text_secondary']}),
                html.Span('-$140.00', style={'color': DARK_THEME['accent_red'], 'fontWeight': 'bold'}),
            ]),
        ], style={
            'backgroundColor': DARK_THEME['card_header'],
            'padding': '0.75rem',
            'borderRadius': '8px',
            'fontSize': '0.9rem',
            'border': f'1px solid {DARK_THEME["border"]}'
        }),

    ])


# ============================================
# TRADE PROGRESS CHART
# ============================================

def create_trade_progress_chart(trades: Optional[List[Dict]] = None) -> go.Figure:
    """
    Create visual chart showing all trades' progress to their targets.

    Features:
    - Bullet chart style for each trade
    - Entry → Current → Target visualization
    - Stop loss indicator
    - Color-coded by profit/loss status

    Args:
        trades: List of trade dictionaries

    Returns:
        Plotly figure with trade progress visualization
    """

    # Sample data if none provided
    if trades is None:
        trades = [
            {
                'name': 'SPY $600C',
                'entry': 598.50,
                'current': 605.30,
                'target': 612.00,
                'stop': 592.00,
            },
            {
                'name': 'QQQ $520C',
                'entry': 512.00,
                'current': 515.40,
                'target': 535.00,
                'stop': 505.00,
            },
            {
                'name': 'AAPL $230C',
                'entry': 228.00,
                'current': 235.80,
                'target': 245.00,
                'stop': 222.00,
            },
        ]

    fig = go.Figure()

    for i, trade in enumerate(trades):
        y_pos = len(trades) - i  # Reverse order (newest at top)

        # Calculate progress percentage
        total_range = trade['target'] - trade['entry']
        current_progress = trade['current'] - trade['entry']
        progress_pct = (current_progress / total_range * 100) if total_range != 0 else 0

        # Determine color based on profit
        is_profit = trade['current'] > trade['entry']
        bar_color = DARK_THEME['accent_green'] if is_profit else DARK_THEME['accent_red']

        # Background bar (entry to target range)
        fig.add_trace(go.Bar(
            x=[total_range],
            y=[y_pos],
            orientation='h',
            marker=dict(color=DARK_THEME['progress_bg']),
            base=trade['entry'],
            width=0.4,
            showlegend=False,
            hoverinfo='skip'
        ))

        # Progress bar (entry to current)
        fig.add_trace(go.Bar(
            x=[current_progress if current_progress > 0 else 0],
            y=[y_pos],
            orientation='h',
            marker=dict(color=bar_color, opacity=0.8),
            base=trade['entry'],
            width=0.4,
            showlegend=False,
            hovertemplate=(
                f"<b>{trade['name']}</b><br>"
                f"Entry: ${trade['entry']:.2f}<br>"
                f"Current: ${trade['current']:.2f}<br>"
                f"Target: ${trade['target']:.2f}<br>"
                f"Progress: {progress_pct:.1f}%<extra></extra>"
            )
        ))

        # Stop loss marker (vertical line)
        fig.add_shape(
            type='line',
            x0=trade['stop'],
            x1=trade['stop'],
            y0=y_pos - 0.25,
            y1=y_pos + 0.25,
            line=dict(color=DARK_THEME['accent_red'], width=2, dash='dot')
        )

        # Target marker (vertical line)
        fig.add_shape(
            type='line',
            x0=trade['target'],
            x1=trade['target'],
            y0=y_pos - 0.25,
            y1=y_pos + 0.25,
            line=dict(color=DARK_THEME['accent_green'], width=2)
        )

        # Current price marker (diamond)
        fig.add_trace(go.Scatter(
            x=[trade['current']],
            y=[y_pos],
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color=bar_color,
                line=dict(color='white', width=1)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Trade name annotation
        fig.add_annotation(
            x=trade['entry'] - (total_range * 0.15),
            y=y_pos,
            text=trade['name'],
            showarrow=False,
            font=dict(color=DARK_THEME['text_primary'], size=11),
            xanchor='right'
        )

        # Progress percentage annotation
        fig.add_annotation(
            x=trade['target'] + (total_range * 0.08),
            y=y_pos,
            text=f"{progress_pct:.0f}%",
            showarrow=False,
            font=dict(color=bar_color, size=12, weight='bold'),
            xanchor='left'
        )

    # Layout
    fig.update_layout(
        height=250,
        template='plotly_dark',
        paper_bgcolor=DARK_THEME['card_bg'],
        plot_bgcolor=DARK_THEME['card_bg'],
        margin=dict(l=100, r=60, t=20, b=40),
        barmode='overlay',
        showlegend=False,
        xaxis=dict(
            title='Underlying Price ($)',
            showgrid=True,
            gridcolor=DARK_THEME['border'],
            zeroline=False,
            tickformat='$,.0f'
        ),
        yaxis=dict(
            visible=False,
            range=[0.5, len(trades) + 0.5]
        ),
        font=dict(color=DARK_THEME['text_primary'])
    )

    # Add legend annotations
    fig.add_annotation(
        x=0.02, y=-0.15,
        xref='paper', yref='paper',
        text='◆ Current  |  ━ Target  |  ╌ Stop',
        showarrow=False,
        font=dict(color=DARK_THEME['text_secondary'], size=10),
        xanchor='left'
    )

    return fig


# ============================================
# INDIVIDUAL TRADE PROGRESS BAR COMPONENT
# ============================================

def create_trade_progress_bar(
    entry_price: float,
    current_price: float,
    target_price: float,
    stop_price: float,
    show_labels: bool = True
) -> html.Div:
    """
    Create an individual trade progress bar showing position relative to target.

    Args:
        entry_price: Trade entry price
        current_price: Current underlying price
        target_price: Pattern target (measured move)
        stop_price: Stop loss price
        show_labels: Whether to show price labels

    Returns:
        HTML div with progress bar visualization
    """

    # Calculate percentages
    total_range = target_price - stop_price
    entry_pct = ((entry_price - stop_price) / total_range * 100) if total_range != 0 else 50
    current_pct = ((current_price - stop_price) / total_range * 100) if total_range != 0 else 50
    target_pct = 100  # Always at end

    # Clamp values
    current_pct = max(0, min(100, current_pct))

    # Determine status
    if current_price >= target_price:
        status_color = DARK_THEME['accent_green']
        status_text = 'TARGET HIT!'
    elif current_price <= stop_price:
        status_color = DARK_THEME['accent_red']
        status_text = 'STOPPED OUT'
    elif current_price >= entry_price:
        status_color = DARK_THEME['accent_green']
        status_text = 'In Profit'
    else:
        status_color = DARK_THEME['accent_red']
        status_text = 'In Loss'

    # Progress to target percentage
    progress_to_target = ((current_price - entry_price) / (target_price - entry_price) * 100) \
        if target_price != entry_price else 0
    progress_to_target = max(-100, min(100, progress_to_target))

    return html.Div([
        # Status header
        html.Div([
            html.Span(status_text, style={'color': status_color, 'fontWeight': 'bold'}),
            html.Span(f' ({progress_to_target:+.1f}% to target)',
                      style={'color': DARK_THEME['text_secondary'], 'marginLeft': '0.5rem'})
        ], style={'marginBottom': '0.5rem'}),

        # Progress bar container
        html.Div([
            # Background
            html.Div(style={
                'position': 'absolute',
                'width': '100%',
                'height': '100%',
                'backgroundColor': DARK_THEME['progress_bg'],
                'borderRadius': '4px'
            }),

            # Stop zone (left portion in red)
            html.Div(style={
                'position': 'absolute',
                'left': '0',
                'width': f'{entry_pct}%',
                'height': '100%',
                'backgroundColor': 'rgba(237, 72, 7, 0.2)',
                'borderRadius': '4px 0 0 4px'
            }),

            # Profit zone (entry to target in green)
            html.Div(style={
                'position': 'absolute',
                'left': f'{entry_pct}%',
                'width': f'{100 - entry_pct}%',
                'height': '100%',
                'backgroundColor': 'rgba(0, 255, 85, 0.2)',
                'borderRadius': '0 4px 4px 0'
            }),

            # Entry marker
            html.Div([
                html.Div(style={
                    'width': '2px',
                    'height': '100%',
                    'backgroundColor': DARK_THEME['text_primary']
                })
            ], style={
                'position': 'absolute',
                'left': f'{entry_pct}%',
                'top': '0',
                'height': '100%',
                'transform': 'translateX(-50%)'
            }),

            # Current price marker (larger, prominent)
            html.Div([
                html.Div(style={
                    'width': '12px',
                    'height': '12px',
                    'backgroundColor': status_color,
                    'borderRadius': '50%',
                    'border': '2px solid white',
                    'boxShadow': '0 0 4px rgba(0,0,0,0.5)'
                })
            ], style={
                'position': 'absolute',
                'left': f'{current_pct}%',
                'top': '50%',
                'transform': 'translate(-50%, -50%)'
            }),

        ], style={
            'position': 'relative',
            'width': '100%',
            'height': '20px',
            'marginBottom': '0.5rem'
        }),

        # Price labels
        html.Div([
            html.Span(f'Stop: ${stop_price:.2f}', style={
                'color': DARK_THEME['accent_red'],
                'fontSize': '0.75rem'
            }),
            html.Span(f'Entry: ${entry_price:.2f}', style={
                'color': DARK_THEME['text_secondary'],
                'fontSize': '0.75rem',
                'position': 'absolute',
                'left': f'{entry_pct}%',
                'transform': 'translateX(-50%)'
            }),
            html.Span(f'Current: ${current_price:.2f}', style={
                'color': status_color,
                'fontSize': '0.75rem',
                'fontWeight': 'bold',
                'position': 'absolute',
                'left': f'{current_pct}%',
                'transform': 'translateX(-50%)',
                'top': '-20px'
            }),
            html.Span(f'Target: ${target_price:.2f}', style={
                'color': DARK_THEME['accent_green'],
                'fontSize': '0.75rem',
                'position': 'absolute',
                'right': '0'
            }),
        ], style={
            'position': 'relative',
            'display': 'flex',
            'justifyContent': 'space-between',
            'paddingTop': '0.25rem'
        }) if show_labels else None,

    ], style={'marginBottom': '1rem'})


# ============================================
# EXPORT MAIN FUNCTION
# ============================================

__all__ = [
    'create_options_panel',
    'create_trade_progress_chart',
    'create_trade_progress_bar',
    'DARK_THEME'
]
