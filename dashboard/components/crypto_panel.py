"""
Crypto Trading Panel Component - Session CRYPTO-6

Professional dark-themed crypto trading panel for dashboard integration:
- Account summary (balance, P&L, return %)
- Daemon status (running, scans, leverage tier)
- Open positions with unrealized P&L
- Pending SETUP signals
- Closed trades history
- Performance metrics

Pattern: Based on options_panel.py
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from typing import List, Dict, Any

from dashboard.config import COLORS, REFRESH_INTERVALS


# ============================================
# DARK THEME CONFIGURATION
# ============================================

DARK_THEME = {
    'background': '#090008',
    'card_bg': '#1a1a2e',
    'card_header': '#16213e',
    'input_bg': '#0f0f1a',
    'border': '#333344',
    'text_primary': '#e0e0e0',
    'text_secondary': '#a0a0a0',
    'text_muted': '#666677',
    'accent_green': '#00ff55',
    'accent_red': '#ed4807',
    'accent_yellow': '#ffd700',
    'accent_blue': '#2196f3',
    'accent_cyan': '#22D3EE',
    'progress_bg': '#2a2a3e',
}


# ============================================
# MAIN PANEL
# ============================================

def create_crypto_panel():
    """
    Create full crypto trading panel with dark professional theme.

    Layout:
    - Row 1: Account Summary | Daemon Status
    - Row 2: Open Positions
    - Row 3: Tabs (Pending Signals | Closed Trades | Performance)

    Returns:
        Bootstrap container with crypto trading interface
    """
    return dbc.Container([

        # ============================================
        # ROW 1: Account Summary + Daemon Status
        # ============================================
        dbc.Row([

            # Left: Account Summary
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-wallet me-2',
                               style={'color': DARK_THEME['accent_cyan']}),
                        'Paper Trading Account'
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='crypto-account-summary',
                        children=[_create_loading_placeholder('Loading account...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow h-100')
            ], width=12, lg=6, className='mb-3'),

            # Right: Daemon Status
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-server me-2',
                               style={'color': DARK_THEME['accent_blue']}),
                        'Daemon Status'
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='crypto-daemon-status',
                        children=[_create_loading_placeholder('Connecting...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow h-100')
            ], width=12, lg=6, className='mb-3'),

        ], className='mb-4'),

        # ============================================
        # ROW 2: Open Positions
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-bar me-2',
                               style={'color': DARK_THEME['accent_green']}),
                        'Open Positions',
                        html.Span(id='crypto-positions-count', className='ms-2')
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='crypto-positions-container',
                        children=[_create_no_positions_placeholder()],
                        style={**_body_style(), 'padding': '0'}
                    )
                ], style=_card_style(), className='shadow')
            ], width=12)
        ], className='mb-4'),

        # ============================================
        # ROW 3: Tabs (Signals | Closed Trades | Performance)
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-crosshairs me-2',
                               style={'color': DARK_THEME['accent_yellow']}),
                        'Trading Activity'
                    ], style=_header_style()),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(
                                label='Pending Signals',
                                tab_id='crypto-tab-signals',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_cyan'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='crypto-signals-container',
                                        children=[_create_no_signals_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                            dbc.Tab(
                                label='Closed Trades',
                                tab_id='crypto-tab-closed',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_cyan'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='crypto-closed-container',
                                        children=[_create_no_closed_trades_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                            dbc.Tab(
                                label='Performance',
                                tab_id='crypto-tab-performance',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_cyan'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='crypto-performance-container',
                                        children=[_create_loading_placeholder('Loading metrics...')],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                        ], id='crypto-tabs', active_tab='crypto-tab-signals', style={
                            'borderBottom': f'1px solid {DARK_THEME["border"]}'
                        }),
                    ], style=_body_style())
                ], style=_card_style(), className='shadow')
            ], width=12)
        ], className='mb-4'),

        # Auto-refresh interval (30 seconds)
        dcc.Interval(
            id='crypto-refresh-interval',
            interval=REFRESH_INTERVALS.get('live_positions', 30000),
            n_intervals=0,
            disabled=False
        ),

    ], fluid=True, style={'backgroundColor': DARK_THEME['background']})


# ============================================
# STYLE HELPERS
# ============================================

def _header_style():
    """Card header style."""
    return {
        'backgroundColor': DARK_THEME['card_header'],
        'color': DARK_THEME['text_primary'],
        'fontWeight': 'bold',
        'borderBottom': f'1px solid {DARK_THEME["border"]}'
    }


def _body_style():
    """Card body style."""
    return {'backgroundColor': DARK_THEME['card_bg']}


def _card_style():
    """Card container style."""
    return {
        'backgroundColor': DARK_THEME['card_bg'],
        'border': f'1px solid {DARK_THEME["border"]}'
    }


# ============================================
# PLACEHOLDER COMPONENTS
# ============================================

def _create_loading_placeholder(message: str = 'Loading...'):
    """Create loading placeholder."""
    return html.Div([
        html.I(className='fas fa-spinner fa-spin fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P(message, style={'color': DARK_THEME['text_muted']})
    ], style={'textAlign': 'center', 'padding': '2rem'})


def _create_no_positions_placeholder():
    """Create placeholder when no positions are held."""
    return html.Div([
        html.I(className='fas fa-inbox fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No open positions', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Crypto positions will appear when trades are opened',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_no_signals_placeholder():
    """Create placeholder when no signals are available."""
    return html.Div([
        html.I(className='fas fa-satellite-dish fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No pending signals', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('SETUP signals awaiting trigger will appear here',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_no_closed_trades_placeholder():
    """Create placeholder when no closed trades are available."""
    return html.Div([
        html.I(className='fas fa-history fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No closed trades', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Completed trades with realized P&L will appear here',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_api_error_placeholder(error: str = 'API not available'):
    """Create placeholder when API is not available."""
    return html.Div([
        html.I(className='fas fa-exclamation-triangle fa-2x mb-3',
               style={'color': DARK_THEME['accent_red']}),
        html.P(error, style={
            'color': DARK_THEME['accent_red'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Check VPS daemon status',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


# ============================================
# DISPLAY COMPONENT BUILDERS
# ============================================

def create_account_summary_display(summary: Dict) -> html.Div:
    """
    Create account summary display with balance, P&L, return %.

    Args:
        summary: Dict from crypto_loader.get_account_summary()

    Returns:
        HTML Div with account cards
    """
    if not summary:
        return _create_api_error_placeholder('Account data not available')

    balance = summary.get('current_balance', 0)
    starting = summary.get('starting_balance', 1000)
    realized_pnl = summary.get('realized_pnl', 0)
    return_pct = summary.get('return_percent', 0)
    open_trades = summary.get('open_trades', 0)
    closed_trades = summary.get('closed_trades', 0)

    # P&L color
    pnl_color = DARK_THEME['accent_green'] if realized_pnl >= 0 else DARK_THEME['accent_red']
    return_color = DARK_THEME['accent_green'] if return_pct >= 0 else DARK_THEME['accent_red']

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Current Balance', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${balance:,.2f}', style={
                        'color': DARK_THEME['text_primary'],
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Starting Balance', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${starting:,.2f}', style={
                        'color': DARK_THEME['text_muted'],
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Realized P&L', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${realized_pnl:+,.2f}', style={
                        'color': pnl_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Return', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'{return_pct:+.2f}%', style={
                        'color': return_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                dbc.Badge(f'{open_trades} Open', color='info', className='me-2'),
                dbc.Badge(f'{closed_trades} Closed', color='secondary'),
            ])
        ])
    ], style={'padding': '1rem'})


def create_daemon_status_display(status: Dict) -> html.Div:
    """
    Create daemon status display with running state, stats, leverage tier.

    Args:
        status: Dict from crypto_loader.get_daemon_status()

    Returns:
        HTML Div with daemon status cards
    """
    if not status:
        return _create_api_error_placeholder('Daemon not reachable')

    running = status.get('running', False)
    uptime = status.get('uptime_seconds', 0)
    scan_count = status.get('scan_count', 0)
    signal_count = status.get('signal_count', 0)
    trigger_count = status.get('trigger_count', 0)
    execution_count = status.get('execution_count', 0)
    error_count = status.get('error_count', 0)
    leverage_tier = status.get('leverage_tier', 'swing')
    maintenance = status.get('maintenance_window', False)

    # Format uptime
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    uptime_str = f'{hours}h {minutes}m' if hours > 0 else f'{minutes}m'

    # Status color
    status_color = DARK_THEME['accent_green'] if running else DARK_THEME['accent_red']
    status_text = 'Running' if running else 'Stopped'

    # Leverage color
    tier_color = DARK_THEME['accent_cyan'] if leverage_tier == 'intraday' else DARK_THEME['accent_yellow']
    tier_leverage = '10x' if leverage_tier == 'intraday' else '4x'

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Status', style={'color': DARK_THEME['text_secondary']}),
                    html.Div([
                        html.Span(status_text, style={
                            'color': status_color,
                            'fontWeight': 'bold',
                            'fontSize': '1.2rem'
                        }),
                        html.I(className='fas fa-circle ms-2', style={
                            'color': status_color,
                            'fontSize': '0.6rem'
                        })
                    ])
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Uptime', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(uptime_str, style={
                        'color': DARK_THEME['text_primary'],
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Leverage Tier', style={'color': DARK_THEME['text_secondary']}),
                    html.Div([
                        dbc.Badge(f'{leverage_tier.upper()} ({tier_leverage})',
                                  style={'backgroundColor': tier_color})
                    ])
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Maintenance', style={'color': DARK_THEME['text_secondary']}),
                    dbc.Badge('Active' if maintenance else 'Normal',
                              color='warning' if maintenance else 'secondary')
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.Small([
                    f'Scans: {scan_count} | ',
                    f'Signals: {signal_count} | ',
                    f'Triggers: {trigger_count} | ',
                    f'Executed: {execution_count}',
                    html.Span(f' | Errors: {error_count}',
                              style={'color': DARK_THEME['accent_red']}) if error_count > 0 else ''
                ], style={'color': DARK_THEME['text_secondary']})
            ])
        ])
    ], style={'padding': '1rem'})


def create_positions_table(positions: List[Dict]) -> html.Table:
    """
    Create table displaying open positions with P&L.

    Args:
        positions: List of position dicts from crypto_loader.get_open_positions()

    Returns:
        HTML table component
    """
    if not positions:
        return _create_no_positions_placeholder()

    rows = []
    for pos in positions:
        symbol = pos.get('symbol', '')
        side = pos.get('side', '')
        quantity = pos.get('quantity', 0)
        entry_price = pos.get('entry_price', 0)
        entry_time = pos.get('entry_time', '')
        current_price = pos.get('current_price', 0)
        unrealized_pnl = pos.get('unrealized_pnl', 0)
        unrealized_pnl_pct = pos.get('unrealized_pnl_percent', 0)
        stop_price = pos.get('stop_price', 0)
        target_price = pos.get('target_price', 0)
        pattern = pos.get('pattern_type', '')
        timeframe = pos.get('timeframe', '')

        # Format entry time (Session CRYPTO-8)
        entry_time_display = ''
        if entry_time:
            try:
                from datetime import datetime
                if isinstance(entry_time, str):
                    # Parse ISO format and format for display
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    entry_time_display = dt.strftime('%m/%d %H:%M')
            except Exception:
                entry_time_display = str(entry_time)[:16]

        # P&L color
        pnl_color = DARK_THEME['accent_green'] if unrealized_pnl >= 0 else DARK_THEME['accent_red']

        # Side color
        side_color = DARK_THEME['accent_green'] if side == 'BUY' else DARK_THEME['accent_red']
        side_label = 'LONG' if side == 'BUY' else 'SHORT'

        rows.append(
            html.Tr([
                # Symbol + Pattern
                html.Td([
                    html.Div(symbol, style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(f'{pattern} ({timeframe})', style={
                        'color': DARK_THEME['text_secondary']
                    })
                ], style={'padding': '0.75rem'}),

                # Side
                html.Td([
                    html.Span(side_label, style={
                        'color': side_color,
                        'fontWeight': 'bold'
                    })
                ], style={'padding': '0.75rem'}),

                # Quantity
                html.Td([
                    f'{quantity:.6f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_primary']}),

                # Entry Price + Time
                html.Td([
                    html.Div(f'${entry_price:,.2f}', style={
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(entry_time_display, style={
                        'color': DARK_THEME['text_secondary']
                    }) if entry_time_display else None
                ], style={'padding': '0.75rem'}),

                # Current Price
                html.Td([
                    f'${current_price:,.2f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_primary']}),

                # Unrealized P&L
                html.Td([
                    html.Div(f'${unrealized_pnl:+,.2f}', style={
                        'color': pnl_color,
                        'fontWeight': 'bold'
                    }),
                    html.Small(f'({unrealized_pnl_pct:+.2f}%)', style={
                        'color': pnl_color
                    })
                ], style={'padding': '0.75rem'}),

                # Stop / Target
                html.Td([
                    html.Span(f'${stop_price:,.2f}',
                              style={'color': DARK_THEME['accent_red']}),
                    html.Span(' / ', style={'color': DARK_THEME['text_muted']}),
                    html.Span(f'${target_price:,.2f}',
                              style={'color': DARK_THEME['accent_green']})
                ], style={'padding': '0.75rem'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th('Symbol', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Side', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Qty', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Entry', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Current', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('P&L', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Stop / Target', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
            ], style={'backgroundColor': DARK_THEME['card_header']})
        ]),
        html.Tbody(rows)
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': DARK_THEME['card_bg']
    })


def create_signals_table(signals: List[Dict]) -> html.Table:
    """
    Create table displaying pending SETUP signals.

    Args:
        signals: List of signal dicts from crypto_loader.get_pending_signals()

    Returns:
        HTML table component
    """
    if not signals:
        return _create_no_signals_placeholder()

    rows = []
    for signal in signals:
        symbol = signal.get('symbol', '')
        timeframe = signal.get('timeframe', '')
        pattern = signal.get('pattern', signal.get('pattern_type', ''))
        direction = signal.get('direction', '')
        entry = signal.get('entry', signal.get('entry_trigger', 0))
        target = signal.get('target', signal.get('target_price', 0))
        stop = signal.get('stop', signal.get('stop_price', 0))
        magnitude = signal.get('magnitude_pct', 0)
        risk_reward = signal.get('risk_reward', 0)
        detected = signal.get('detected_time_display', signal.get('detected_time', ''))

        # Direction color
        dir_color = DARK_THEME['accent_green'] if direction == 'LONG' else DARK_THEME['accent_red']

        rows.append(
            html.Tr([
                # Symbol + Timeframe
                html.Td([
                    html.Div(symbol, style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(timeframe, style={
                        'color': DARK_THEME['text_secondary']
                    })
                ], style={'padding': '0.75rem'}),

                # Pattern
                html.Td([
                    dbc.Badge(pattern, color='dark', style={'fontSize': '0.85rem'})
                ], style={'padding': '0.75rem'}),

                # Direction
                html.Td([
                    html.Span(direction, style={
                        'color': dir_color,
                        'fontWeight': 'bold'
                    })
                ], style={'padding': '0.75rem'}),

                # Entry Trigger
                html.Td([
                    f'${entry:,.2f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_primary']}),

                # Target / Stop
                html.Td([
                    html.Span(f'${target:,.2f}',
                              style={'color': DARK_THEME['accent_green']}),
                    html.Span(' / ', style={'color': DARK_THEME['text_muted']}),
                    html.Span(f'${stop:,.2f}',
                              style={'color': DARK_THEME['accent_red']})
                ], style={'padding': '0.75rem'}),

                # Magnitude
                html.Td([
                    f'{magnitude:.2f}%'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_secondary']}),

                # R:R
                html.Td([
                    f'{risk_reward:.1f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['accent_blue']}),

                # Detected
                html.Td([
                    html.Small(str(detected), style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '0.75rem'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th('Symbol', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Pattern', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Dir', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Entry', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Target / Stop', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Mag', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('R:R', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Detected', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
            ], style={'backgroundColor': DARK_THEME['card_header']})
        ]),
        html.Tbody(rows)
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': DARK_THEME['card_bg']
    })


def create_closed_trades_table(trades: List[Dict]) -> html.Div:
    """
    Create table displaying closed trades with realized P&L.

    Args:
        trades: List of closed trade dicts from crypto_loader.get_closed_trades()

    Returns:
        HTML Div containing summary and table
    """
    if not trades:
        return _create_no_closed_trades_placeholder()

    # Calculate summary stats
    total_pnl = sum(t.get('pnl', 0) or 0 for t in trades)
    winners = [t for t in trades if (t.get('pnl') or 0) > 0]
    losers = [t for t in trades if (t.get('pnl') or 0) <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    # P&L color
    pnl_color = DARK_THEME['accent_green'] if total_pnl >= 0 else DARK_THEME['accent_red']

    # Summary row
    summary = html.Div([
        dbc.Row([
            dbc.Col([
                html.Span('Total P&L: ', style={'color': DARK_THEME['text_secondary']}),
                html.Span(f'${total_pnl:+,.2f}', style={
                    'color': pnl_color,
                    'fontWeight': 'bold'
                })
            ], width='auto'),
            dbc.Col([
                html.Span('Win Rate: ', style={'color': DARK_THEME['text_secondary']}),
                html.Span(f'{win_rate:.1f}%', style={
                    'color': DARK_THEME['text_primary'],
                    'fontWeight': 'bold'
                })
            ], width='auto'),
            dbc.Col([
                dbc.Badge(f'{len(winners)}W', color='success', className='me-1'),
                dbc.Badge(f'{len(losers)}L', color='danger'),
            ], width='auto'),
        ], className='mb-3', justify='start')
    ], style={'padding': '0.5rem 0.75rem', 'borderBottom': f'1px solid {DARK_THEME["border"]}'})

    # Build table rows
    rows = []
    for trade in trades:
        symbol = trade.get('symbol', '')
        side = trade.get('side', '')
        quantity = trade.get('quantity', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('pnl', 0) or 0
        pnl_pct = trade.get('pnl_percent', 0) or 0
        exit_reason = trade.get('exit_reason', '')
        exit_time = trade.get('exit_time', '')

        # P&L color
        row_pnl_color = DARK_THEME['accent_green'] if pnl >= 0 else DARK_THEME['accent_red']

        # Side label
        side_label = 'LONG' if side == 'BUY' else 'SHORT'
        side_color = DARK_THEME['accent_green'] if side == 'BUY' else DARK_THEME['accent_red']

        # Exit reason badge color
        reason_color = 'success' if exit_reason == 'TARGET' else 'danger' if exit_reason == 'STOP' else 'secondary'

        rows.append(
            html.Tr([
                # Symbol
                html.Td([
                    html.Span(symbol, style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    })
                ], style={'padding': '0.75rem'}),

                # Side
                html.Td([
                    html.Span(side_label, style={'color': side_color})
                ], style={'padding': '0.75rem'}),

                # Qty
                html.Td([
                    f'{quantity:.6f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_secondary']}),

                # Entry
                html.Td([
                    f'${entry_price:,.2f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_secondary']}),

                # Exit
                html.Td([
                    f'${exit_price:,.2f}'
                ], style={'padding': '0.75rem', 'color': DARK_THEME['text_secondary']}),

                # Realized P&L
                html.Td([
                    html.Div(f'${pnl:+,.2f}', style={
                        'color': row_pnl_color,
                        'fontWeight': 'bold'
                    }),
                    html.Small(f'({pnl_pct:+.2f}%)', style={'color': row_pnl_color})
                ], style={'padding': '0.75rem'}),

                # Exit Reason
                html.Td([
                    dbc.Badge(exit_reason or 'MANUAL', color=reason_color)
                ], style={'padding': '0.75rem'}),

                # Exit Time
                html.Td([
                    html.Small(str(exit_time)[:16] if exit_time else '',
                               style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '0.75rem'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Symbol', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Side', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Qty', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Entry', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Exit', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('P&L', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Reason', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Closed', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
            ], style={'backgroundColor': DARK_THEME['card_header']})
        ]),
        html.Tbody(rows)
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': DARK_THEME['card_bg']
    })

    return html.Div([summary, table])


def create_performance_metrics(metrics: Dict) -> html.Div:
    """
    Create performance metrics display.

    Args:
        metrics: Dict from crypto_loader.get_performance_metrics()

    Returns:
        HTML Div with performance cards
    """
    if not metrics:
        return _create_api_error_placeholder('Performance data not available')

    total_trades = metrics.get('total_trades', 0)
    winning_trades = metrics.get('winning_trades', 0)
    losing_trades = metrics.get('losing_trades', 0)
    win_rate = metrics.get('win_rate', 0)
    total_pnl = metrics.get('total_pnl', 0)
    profit_factor = metrics.get('profit_factor', 0)
    expectancy = metrics.get('expectancy', 0)
    avg_win = metrics.get('avg_win', 0)
    avg_loss = metrics.get('avg_loss', 0)
    largest_win = metrics.get('largest_win', 0)
    largest_loss = metrics.get('largest_loss', 0)

    # P&L color
    pnl_color = DARK_THEME['accent_green'] if total_pnl >= 0 else DARK_THEME['accent_red']
    exp_color = DARK_THEME['accent_green'] if expectancy >= 0 else DARK_THEME['accent_red']

    return html.Div([
        # Row 1: Key metrics
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Total P&L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${total_pnl:+,.2f}', style={
                        'color': pnl_color,
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Win Rate', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'{win_rate:.1f}%', style={
                        'color': DARK_THEME['text_primary'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Profit Factor', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'{profit_factor:.2f}', style={
                        'color': DARK_THEME['accent_blue'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Expectancy', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${expectancy:+,.2f}', style={
                        'color': exp_color,
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
        ], className='mb-4'),

        html.Hr(style={'borderColor': DARK_THEME['border']}),

        # Row 2: Trade counts
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Total Trades', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(str(total_trades), style={
                        'color': DARK_THEME['text_primary'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small('Winners', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(str(winning_trades), style={
                        'color': DARK_THEME['accent_green'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small('Losers', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(str(losing_trades), style={
                        'color': DARK_THEME['accent_red'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=4),
        ], className='mb-4'),

        # Row 3: Averages
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Avg Win', style={'color': DARK_THEME['text_secondary']}),
                    html.H6(f'${avg_win:,.2f}', style={
                        'color': DARK_THEME['accent_green'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Avg Loss', style={'color': DARK_THEME['text_secondary']}),
                    html.H6(f'${avg_loss:,.2f}', style={
                        'color': DARK_THEME['accent_red'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Largest Win', style={'color': DARK_THEME['text_secondary']}),
                    html.H6(f'${largest_win:,.2f}', style={
                        'color': DARK_THEME['accent_green'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
            dbc.Col([
                html.Div([
                    html.Small('Largest Loss', style={'color': DARK_THEME['text_secondary']}),
                    html.H6(f'${largest_loss:,.2f}', style={
                        'color': DARK_THEME['accent_red'],
                        'marginBottom': '0'
                    })
                ], className='text-center')
            ], width=6, md=3),
        ]),
    ], style={'padding': '1rem'})


# ============================================
# EXPORTS
# ============================================

__all__ = [
    'create_crypto_panel',
    'create_account_summary_display',
    'create_daemon_status_display',
    'create_positions_table',
    'create_signals_table',
    'create_closed_trades_table',
    'create_performance_metrics',
]
