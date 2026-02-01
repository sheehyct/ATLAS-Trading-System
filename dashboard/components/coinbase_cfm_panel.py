"""
Coinbase CFM Trading Panel Component.

Professional dark-themed panel for displaying Coinbase CFM derivatives P/L:
- Account summary (realized/unrealized P/L, fees)
- Crypto perpetuals section (BIP, ETP, SOP, ADP, XRP)
- Commodity futures section (SLRH, GOLJ)
- Open positions with current P/L
- Closed trades history
- Funding payments (perpetuals only)
- Performance metrics

Requires read-only Coinbase API access via COINBASE_READONLY_API_KEY.

Pattern: Based on crypto_panel.py
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from typing import List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from dashboard.config import COLORS


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
    'accent_gold': '#FFD700',
    'accent_silver': '#C0C0C0',
    'progress_bg': '#2a2a3e',
}

# Product colors
PRODUCT_COLORS = {
    'BIP': '#F7931A',   # Bitcoin orange
    'ETP': '#627EEA',   # Ethereum blue
    'SOP': '#9945FF',   # Solana purple
    'ADP': '#0033AD',   # Cardano blue
    'XRP': '#23292F',   # XRP dark
    'SLRH': '#C0C0C0',  # Silver
    'GOLJ': '#FFD700',  # Gold
}


# ============================================
# TIMEZONE HELPER
# ============================================

def _format_time_est(time_str: str) -> str:
    """Convert ISO timestamp to EST display format."""
    if not time_str:
        return ''
    try:
        if isinstance(time_str, str):
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            return str(time_str)[:16]
        est = ZoneInfo('America/New_York')
        dt_est = dt.astimezone(est)
        return dt_est.strftime('%m/%d %H:%M')
    except Exception:
        return str(time_str)[:16]


# ============================================
# STYLE HELPERS
# ============================================

def _card_style():
    """Base card style."""
    return {
        'backgroundColor': DARK_THEME['card_bg'],
        'border': f'1px solid {DARK_THEME["border"]}',
        'borderRadius': '8px',
    }


def _header_style():
    """Card header style."""
    return {
        'backgroundColor': DARK_THEME['card_header'],
        'color': DARK_THEME['text_primary'],
        'borderBottom': f'1px solid {DARK_THEME["border"]}',
        'fontWeight': '600',
    }


def _body_style():
    """Card body style."""
    return {
        'backgroundColor': DARK_THEME['card_bg'],
        'color': DARK_THEME['text_primary'],
        'padding': '1rem',
    }


def _table_style():
    """Table container style."""
    return {
        'backgroundColor': DARK_THEME['card_bg'],
        'color': DARK_THEME['text_primary'],
        'fontSize': '0.85rem',
    }


def _th_style():
    """Table header cell style."""
    return {
        'backgroundColor': DARK_THEME['card_header'],
        'color': DARK_THEME['text_secondary'],
        'padding': '0.5rem',
        'borderBottom': f'1px solid {DARK_THEME["border"]}',
        'fontWeight': '500',
        'fontSize': '0.75rem',
        'textTransform': 'uppercase',
    }


def _td_style():
    """Table data cell style."""
    return {
        'padding': '0.5rem',
        'borderBottom': f'1px solid {DARK_THEME["border"]}',
        'verticalAlign': 'middle',
    }


# ============================================
# PLACEHOLDERS
# ============================================

def _create_loading_placeholder(message: str = 'Loading...'):
    """Create loading placeholder."""
    return html.Div([
        dbc.Spinner(size='sm', color='info', spinner_class_name='me-2'),
        html.Span(message, style={'color': DARK_THEME['text_secondary']})
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
        html.Small('Configure COINBASE_READONLY_API_KEY/SECRET',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_no_data_placeholder(message: str = 'No data available'):
    """Create placeholder when no data."""
    return html.Div([
        html.I(className='fas fa-inbox fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P(message, style={'color': DARK_THEME['text_secondary']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


# ============================================
# MAIN PANEL
# ============================================

def create_coinbase_cfm_panel():
    """
    Create full Coinbase CFM trading panel.

    Layout:
    - Row 1: Crypto Perps Summary | Commodity Futures Summary
    - Row 2: Open Positions
    - Row 3: Tabs (Closed Trades | Funding | Performance | By Product)

    Returns:
        Bootstrap container with CFM trading interface
    """
    return dbc.Container([

        # ============================================
        # ROW 1: Account Summaries
        # ============================================
        dbc.Row([

            # Left: Crypto Perpetuals
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fab fa-bitcoin me-2',
                               style={'color': PRODUCT_COLORS['BIP']}),
                        'Crypto Perpetuals',
                        html.Span(' (BIP / ETP / SOP / ADP / XRP)',
                                  style={'fontSize': '0.8rem', 'color': DARK_THEME['text_muted'],
                                         'marginLeft': '8px'})
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='cfm-crypto-summary',
                        children=[_create_loading_placeholder('Loading crypto P/L...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow h-100')
            ], width=12, lg=6, className='mb-3'),

            # Right: Commodity Futures
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-coins me-2',
                               style={'color': PRODUCT_COLORS['GOLJ']}),
                        'Commodity Futures',
                        html.Span(' (SLRH / GOLJ)',
                                  style={'fontSize': '0.8rem', 'color': DARK_THEME['text_muted'],
                                         'marginLeft': '8px'})
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='cfm-commodity-summary',
                        children=[_create_loading_placeholder('Loading commodity P/L...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow h-100')
            ], width=12, lg=6, className='mb-3'),

        ], className='mb-3'),

        # ============================================
        # ROW 2: Open Positions
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-line me-2',
                               style={'color': DARK_THEME['accent_cyan']}),
                        'Open Positions',
                        html.Span(id='cfm-positions-count',
                                  className='ms-2',
                                  style={'color': DARK_THEME['text_muted']})
                    ], style=_header_style()),
                    dbc.CardBody(
                        id='cfm-positions-table',
                        children=[_create_loading_placeholder('Loading positions...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow')
            ], width=12)
        ], className='mb-3'),

        # ============================================
        # ROW 3: Tabs
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Tabs([
                            dbc.Tab(label='Closed Trades', tab_id='cfm-closed-tab'),
                            dbc.Tab(label='Equity Curve', tab_id='cfm-equity-tab'),
                            dbc.Tab(label='Funding', tab_id='cfm-funding-tab'),
                            dbc.Tab(label='Performance', tab_id='cfm-performance-tab'),
                            dbc.Tab(label='By Product', tab_id='cfm-by-product-tab'),
                        ], id='cfm-tabs', active_tab='cfm-closed-tab',
                        style={'borderBottom': 'none'})
                    ], style={
                        **_header_style(),
                        'padding': '0.5rem 1rem',
                    }),
                    dbc.CardBody(
                        id='cfm-tab-content',
                        children=[_create_loading_placeholder('Loading...')],
                        style=_body_style()
                    )
                ], style=_card_style(), className='shadow')
            ], width=12)
        ]),

        # ============================================
        # REFRESH INTERVAL
        # ============================================
        dcc.Interval(
            id='cfm-refresh-interval',
            interval=300000,  # 5 minutes
            n_intervals=0
        ),

    ], fluid=True, style={
        'backgroundColor': DARK_THEME['background'],
        'padding': '1rem',
        'minHeight': '100vh',
    })


# ============================================
# SUMMARY DISPLAYS
# ============================================

def create_crypto_summary_display(summary: Dict) -> html.Div:
    """
    Create crypto perpetuals summary display.

    Args:
        summary: Dict with realized_pnl, unrealized_pnl, funding, etc.

    Returns:
        HTML Div with crypto summary cards
    """
    if not summary:
        return _create_api_error_placeholder('Crypto data not available')

    realized = summary.get('realized_pnl', 0)
    unrealized = summary.get('unrealized_pnl', 0)
    total_fees = summary.get('total_fees', 0)
    trade_count = summary.get('trade_count', 0)
    win_rate = summary.get('win_rate', 0)
    funding_paid = summary.get('funding_paid', 0)
    funding_received = summary.get('funding_received', 0)
    net_funding = summary.get('net_funding', 0)
    open_positions = summary.get('open_positions', 0)

    # P&L colors
    realized_color = DARK_THEME['accent_green'] if realized >= 0 else DARK_THEME['accent_red']
    unrealized_color = DARK_THEME['accent_green'] if unrealized >= 0 else DARK_THEME['accent_red']
    funding_color = DARK_THEME['accent_green'] if net_funding >= 0 else DARK_THEME['accent_red']

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Realized P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${realized:+,.2f}', style={
                        'color': realized_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Unrealized P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${unrealized:+,.2f}', style={
                        'color': unrealized_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Net Funding', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${net_funding:+,.2f}', style={
                        'color': funding_color,
                        'marginBottom': '0'
                    }),
                    html.Small(f'Paid: ${funding_paid:.2f} / Recv: ${funding_received:.2f}',
                               style={'color': DARK_THEME['text_muted'], 'fontSize': '0.7rem'})
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Fees Paid', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${total_fees:,.2f}', style={
                        'color': DARK_THEME['accent_yellow'],
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                dbc.Badge(f'{open_positions} Open', color='info', className='me-2'),
                dbc.Badge(f'{trade_count} Trades', color='secondary', className='me-2'),
                dbc.Badge(f'{win_rate:.1f}% Win', color='success' if win_rate >= 50 else 'danger'),
            ])
        ])
    ], style={'padding': '0.5rem'})


def create_commodity_summary_display(summary: Dict) -> html.Div:
    """
    Create commodity futures summary display.

    Args:
        summary: Dict with realized_pnl, unrealized_pnl, etc.

    Returns:
        HTML Div with commodity summary cards
    """
    if not summary:
        return _create_api_error_placeholder('Commodity data not available')

    realized = summary.get('realized_pnl', 0)
    unrealized = summary.get('unrealized_pnl', 0)
    total_fees = summary.get('total_fees', 0)
    trade_count = summary.get('trade_count', 0)
    win_rate = summary.get('win_rate', 0)
    open_positions = summary.get('open_positions', 0)

    # P&L colors
    realized_color = DARK_THEME['accent_green'] if realized >= 0 else DARK_THEME['accent_red']
    unrealized_color = DARK_THEME['accent_green'] if unrealized >= 0 else DARK_THEME['accent_red']

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Realized P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${realized:+,.2f}', style={
                        'color': realized_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('Unrealized P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${unrealized:+,.2f}', style={
                        'color': unrealized_color,
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Fees Paid', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${total_fees:,.2f}', style={
                        'color': DARK_THEME['accent_yellow'],
                        'marginBottom': '0'
                    })
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    html.Small('No Funding', style={'color': DARK_THEME['text_muted']}),
                    html.P('Futures settle on expiry',
                           style={'color': DARK_THEME['text_muted'], 'fontSize': '0.75rem',
                                  'marginBottom': '0'})
                ])
            ], width=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col([
                dbc.Badge(f'{open_positions} Open', color='info', className='me-2'),
                dbc.Badge(f'{trade_count} Trades', color='secondary', className='me-2'),
                dbc.Badge(f'{win_rate:.1f}% Win', color='success' if win_rate >= 50 else 'danger'),
            ])
        ])
    ], style={'padding': '0.5rem'})


# ============================================
# POSITIONS TABLE
# ============================================

def create_positions_table(positions: List[Dict]) -> html.Table:
    """
    Create open positions table.

    Args:
        positions: List of position dicts from CFM loader

    Returns:
        HTML Table with open positions
    """
    if not positions:
        return _create_no_data_placeholder('No open positions')

    header = html.Thead(html.Tr([
        html.Th('Symbol', style=_th_style()),
        html.Th('Side', style=_th_style()),
        html.Th('Qty', style=_th_style()),
        html.Th('Entry', style=_th_style()),
        html.Th('Current', style=_th_style()),
        html.Th('Unreal P/L', style=_th_style()),
        html.Th('Type', style=_th_style()),
    ]))

    rows = []
    for pos in positions:
        symbol = pos.get('base_symbol', pos.get('product_id', ''))
        side = pos.get('side', '')
        qty = pos.get('quantity', 0)
        entry = pos.get('avg_entry_price', 0)
        current = pos.get('current_price', 0)
        unrealized = pos.get('unrealized_pnl', 0)
        product_type = pos.get('product_type', '')

        # Colors
        side_color = DARK_THEME['accent_green'] if side == 'BUY' else DARK_THEME['accent_red']
        pnl_color = DARK_THEME['accent_green'] if unrealized >= 0 else DARK_THEME['accent_red']
        symbol_color = PRODUCT_COLORS.get(symbol, DARK_THEME['text_primary'])

        rows.append(html.Tr([
            html.Td([
                dbc.Badge(symbol, style={'backgroundColor': symbol_color, 'color': '#fff'})
            ], style=_td_style()),
            html.Td(side, style={**_td_style(), 'color': side_color, 'fontWeight': 'bold'}),
            html.Td(f'{qty:.4f}', style=_td_style()),
            html.Td(f'${entry:,.2f}', style=_td_style()),
            html.Td(f'${current:,.2f}', style=_td_style()),
            html.Td(f'${unrealized:+,.2f}', style={**_td_style(), 'color': pnl_color, 'fontWeight': 'bold'}),
            html.Td(
                dbc.Badge('Perp' if product_type == 'crypto_perp' else 'Future',
                          color='info' if product_type == 'crypto_perp' else 'warning',
                          style={'fontSize': '0.7rem'}),
                style=_td_style()
            ),
        ]))

    return html.Table([header, html.Tbody(rows)], style={
        **_table_style(),
        'width': '100%',
        'borderCollapse': 'collapse',
    })


# ============================================
# CLOSED TRADES TABLE
# ============================================

def create_closed_trades_table(trades: List[Dict]) -> html.Table:
    """
    Create closed trades table.

    Args:
        trades: List of trade dicts from CFM loader

    Returns:
        HTML Table with closed trades
    """
    if not trades:
        return _create_no_data_placeholder('No closed trades')

    header = html.Thead(html.Tr([
        html.Th('Symbol', style=_th_style()),
        html.Th('Side', style=_th_style()),
        html.Th('Entry', style=_th_style()),
        html.Th('Exit', style=_th_style()),
        html.Th('Net P/L', style=_th_style()),
        html.Th('%', style=_th_style()),
        html.Th('Exit Time', style=_th_style()),
    ]))

    rows = []
    for trade in trades[:25]:  # Limit to 25 rows
        symbol = trade.get('base_symbol', '')
        side = trade.get('side', '')
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        net_pnl = trade.get('net_pnl', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        exit_time = trade.get('exit_time', '')

        # Colors
        side_color = DARK_THEME['accent_green'] if side == 'BUY' else DARK_THEME['accent_red']
        pnl_color = DARK_THEME['accent_green'] if net_pnl >= 0 else DARK_THEME['accent_red']
        symbol_color = PRODUCT_COLORS.get(symbol, DARK_THEME['text_primary'])

        rows.append(html.Tr([
            html.Td([
                dbc.Badge(symbol, style={'backgroundColor': symbol_color, 'color': '#fff'})
            ], style=_td_style()),
            html.Td(side, style={**_td_style(), 'color': side_color}),
            html.Td(f'${entry:,.2f}', style=_td_style()),
            html.Td(f'${exit_price:,.2f}', style=_td_style()),
            html.Td(f'${net_pnl:+,.2f}', style={**_td_style(), 'color': pnl_color, 'fontWeight': 'bold'}),
            html.Td(f'{pnl_pct:+.2f}%', style={**_td_style(), 'color': pnl_color}),
            html.Td(_format_time_est(exit_time), style={**_td_style(), 'color': DARK_THEME['text_muted']}),
        ]))

    return html.Table([header, html.Tbody(rows)], style={
        **_table_style(),
        'width': '100%',
        'borderCollapse': 'collapse',
    })


# ============================================
# FUNDING TABLE
# ============================================

def create_funding_table(payments: List[Dict]) -> html.Div:
    """
    Create funding payments display.

    Args:
        payments: List of funding payment dicts

    Returns:
        HTML Div with funding summary and table
    """
    if not payments:
        return _create_no_data_placeholder('No funding payments recorded')

    # Summary
    total_paid = sum(float(p.get('amount', 0)) for p in payments if float(p.get('amount', 0)) > 0)
    total_received = abs(sum(float(p.get('amount', 0)) for p in payments if float(p.get('amount', 0)) < 0))
    net = total_received - total_paid

    summary = html.Div([
        dbc.Row([
            dbc.Col([
                html.Small('Paid', style={'color': DARK_THEME['text_muted']}),
                html.H5(f'${total_paid:,.2f}', style={'color': DARK_THEME['accent_red']})
            ], width=4),
            dbc.Col([
                html.Small('Received', style={'color': DARK_THEME['text_muted']}),
                html.H5(f'${total_received:,.2f}', style={'color': DARK_THEME['accent_green']})
            ], width=4),
            dbc.Col([
                html.Small('Net', style={'color': DARK_THEME['text_muted']}),
                html.H5(f'${net:+,.2f}', style={
                    'color': DARK_THEME['accent_green'] if net >= 0 else DARK_THEME['accent_red']
                })
            ], width=4),
        ], className='mb-3', style={'textAlign': 'center'})
    ])

    # Table (recent payments)
    header = html.Thead(html.Tr([
        html.Th('Product', style=_th_style()),
        html.Th('Amount', style=_th_style()),
        html.Th('Time', style=_th_style()),
    ]))

    rows = []
    for payment in payments[:20]:
        product = payment.get('product_id', '')
        amount = float(payment.get('amount', 0))
        timestamp = payment.get('timestamp', '')

        amount_color = DARK_THEME['accent_green'] if amount < 0 else DARK_THEME['accent_red']
        display_amount = abs(amount)
        direction = 'Received' if amount < 0 else 'Paid'

        rows.append(html.Tr([
            html.Td(product, style=_td_style()),
            html.Td(f'{direction}: ${display_amount:.4f}', style={**_td_style(), 'color': amount_color}),
            html.Td(_format_time_est(timestamp), style={**_td_style(), 'color': DARK_THEME['text_muted']}),
        ]))

    table = html.Table([header, html.Tbody(rows)], style={
        **_table_style(),
        'width': '100%',
        'borderCollapse': 'collapse',
    }) if rows else html.P('No recent payments', style={'color': DARK_THEME['text_muted']})

    return html.Div([summary, table])


# ============================================
# PERFORMANCE DISPLAY
# ============================================

def create_performance_display(metrics: Dict) -> html.Div:
    """
    Create performance metrics display.

    Args:
        metrics: Dict from CFM loader get_performance_metrics()

    Returns:
        HTML Div with performance cards
    """
    if not metrics or metrics.get('trade_count', 0) == 0:
        return _create_no_data_placeholder('No trades to analyze')

    trade_count = metrics.get('trade_count', 0)
    win_count = metrics.get('win_count', 0)
    loss_count = metrics.get('loss_count', 0)
    win_rate = metrics.get('win_rate', 0)
    profit_factor = metrics.get('profit_factor', 0)
    expectancy = metrics.get('expectancy', 0)
    avg_winner = metrics.get('avg_winner', 0)
    avg_loser = metrics.get('avg_loser', 0)
    avg_hold_hours = metrics.get('avg_hold_hours', 0)
    gross_pnl = metrics.get('gross_pnl', 0)
    net_pnl = metrics.get('net_pnl', 0)
    total_fees = metrics.get('total_fees', 0)

    # Colors
    pnl_color = DARK_THEME['accent_green'] if net_pnl >= 0 else DARK_THEME['accent_red']
    pf_color = DARK_THEME['accent_green'] if profit_factor >= 1 else DARK_THEME['accent_red']
    wr_color = DARK_THEME['accent_green'] if win_rate >= 50 else DARK_THEME['accent_yellow']

    return html.Div([
        # Row 1: P&L Summary
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Net P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H3(f'${net_pnl:+,.2f}', style={'color': pnl_color})
                ], style={'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small('Gross P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${gross_pnl:+,.2f}', style={'color': DARK_THEME['text_primary']})
                ], style={'textAlign': 'center'})
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Small('Total Fees', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${total_fees:,.2f}', style={'color': DARK_THEME['accent_yellow']})
                ], style={'textAlign': 'center'})
            ], width=4),
        ], className='mb-4'),

        # Row 2: Win/Loss
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Win Rate', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'{win_rate:.1f}%', style={'color': wr_color})
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Profit Factor', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'{profit_factor:.2f}', style={'color': pf_color})
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Expectancy', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${expectancy:+,.2f}', style={
                        'color': DARK_THEME['accent_green'] if expectancy > 0 else DARK_THEME['accent_red']
                    })
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Avg Hold', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'{avg_hold_hours:.1f}h', style={'color': DARK_THEME['text_primary']})
                ], style={'textAlign': 'center'})
            ], width=3),
        ], className='mb-4'),

        # Row 3: Trade counts
        dbc.Row([
            dbc.Col([
                dbc.Badge(f'{trade_count} Total Trades', color='secondary', className='me-2'),
                dbc.Badge(f'{win_count} Winners', color='success', className='me-2'),
                dbc.Badge(f'{loss_count} Losers', color='danger', className='me-2'),
            ], style={'textAlign': 'center'})
        ], className='mb-3'),

        # Row 4: Avg Winner/Loser
        dbc.Row([
            dbc.Col([
                html.Small('Avg Winner', style={'color': DARK_THEME['text_muted']}),
                html.P(f'${avg_winner:+,.2f}', style={'color': DARK_THEME['accent_green'], 'marginBottom': '0'})
            ], width=6, style={'textAlign': 'center'}),
            dbc.Col([
                html.Small('Avg Loser', style={'color': DARK_THEME['text_muted']}),
                html.P(f'${avg_loser:,.2f}', style={'color': DARK_THEME['accent_red'], 'marginBottom': '0'})
            ], width=6, style={'textAlign': 'center'}),
        ])
    ], style={'padding': '1rem'})


# ============================================
# BY PRODUCT TABLE
# ============================================

def create_by_product_table(pnl_by_product: Dict) -> html.Table:
    """
    Create P/L breakdown by product table.

    Args:
        pnl_by_product: Dict mapping symbol to P/L metrics

    Returns:
        HTML Table with per-product breakdown
    """
    if not pnl_by_product:
        return _create_no_data_placeholder('No trades to analyze')

    header = html.Thead(html.Tr([
        html.Th('Product', style=_th_style()),
        html.Th('Asset', style=_th_style()),
        html.Th('Net P/L', style=_th_style()),
        html.Th('Trades', style=_th_style()),
        html.Th('Win Rate', style=_th_style()),
        html.Th('Fees', style=_th_style()),
    ]))

    rows = []
    for symbol, data in sorted(pnl_by_product.items(), key=lambda x: x[1].get('net_pnl', 0), reverse=True):
        asset_name = data.get('asset_name', symbol)
        net_pnl = data.get('net_pnl', 0)
        trade_count = data.get('trade_count', 0)
        win_rate = data.get('win_rate', 0)
        total_fees = data.get('total_fees', 0)

        pnl_color = DARK_THEME['accent_green'] if net_pnl >= 0 else DARK_THEME['accent_red']
        symbol_color = PRODUCT_COLORS.get(symbol, DARK_THEME['text_primary'])

        rows.append(html.Tr([
            html.Td([
                dbc.Badge(symbol, style={'backgroundColor': symbol_color, 'color': '#fff'})
            ], style=_td_style()),
            html.Td(asset_name, style=_td_style()),
            html.Td(f'${net_pnl:+,.2f}', style={**_td_style(), 'color': pnl_color, 'fontWeight': 'bold'}),
            html.Td(str(trade_count), style=_td_style()),
            html.Td(f'{win_rate:.1f}%', style={
                **_td_style(),
                'color': DARK_THEME['accent_green'] if win_rate >= 50 else DARK_THEME['accent_yellow']
            }),
            html.Td(f'${total_fees:,.2f}', style={**_td_style(), 'color': DARK_THEME['accent_yellow']}),
        ]))

    return html.Table([header, html.Tbody(rows)], style={
        **_table_style(),
        'width': '100%',
        'borderCollapse': 'collapse',
    })


# ============================================
# EQUITY CURVE CHART
# ============================================

def create_equity_curve_display(
    cumulative_series: List[Dict],
    product_series: Dict[str, List[Dict]]
) -> html.Div:
    """
    Create equity curve chart with cumulative P/L and per-product breakdown.

    Args:
        cumulative_series: List of {date, daily_pnl, cumulative_pnl} from loader
        product_series: Dict mapping product to list of {date, cumulative_pnl}

    Returns:
        HTML Div with equity curve chart and summary stats
    """
    if not cumulative_series:
        return _create_no_data_placeholder('No trade history for equity curve')

    # Extract data for main cumulative chart
    dates = [d['date'] for d in cumulative_series]
    cumulative_pnl = [d['cumulative_pnl'] for d in cumulative_series]

    # Final P/L value
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    pnl_color = DARK_THEME['accent_green'] if final_pnl >= 0 else DARK_THEME['accent_red']

    # Create main equity curve figure
    fig = go.Figure()

    # Add cumulative P/L line
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines',
        name='Cumulative P/L',
        line=dict(color=DARK_THEME['accent_cyan'], width=2),
        fill='tozeroy',
        fillcolor='rgba(34, 211, 238, 0.1)',
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=DARK_THEME['text_muted'], opacity=0.5)

    # Style the chart
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=DARK_THEME['card_bg'],
        plot_bgcolor=DARK_THEME['card_bg'],
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
        title=dict(
            text='Cumulative Realized P/L',
            font=dict(size=14, color=DARK_THEME['text_primary']),
            x=0.5,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=DARK_THEME['border'],
            tickfont=dict(color=DARK_THEME['text_secondary'], size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=DARK_THEME['border'],
            tickfont=dict(color=DARK_THEME['text_secondary'], size=10),
            tickprefix='$',
        ),
        showlegend=False,
        hovermode='x unified',
    )

    # Create per-product chart if data available
    product_fig = None
    if product_series:
        product_fig = go.Figure()

        for symbol, series in product_series.items():
            if series:
                p_dates = [d['date'] for d in series]
                p_cumulative = [d['cumulative_pnl'] for d in series]
                color = PRODUCT_COLORS.get(symbol, DARK_THEME['accent_cyan'])

                product_fig.add_trace(go.Scatter(
                    x=p_dates,
                    y=p_cumulative,
                    mode='lines',
                    name=symbol,
                    line=dict(color=color, width=2),
                ))

        product_fig.add_hline(y=0, line_dash="dash", line_color=DARK_THEME['text_muted'], opacity=0.5)

        product_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=DARK_THEME['card_bg'],
            plot_bgcolor=DARK_THEME['card_bg'],
            margin=dict(l=50, r=20, t=40, b=40),
            height=250,
            title=dict(
                text='P/L by Product',
                font=dict(size=14, color=DARK_THEME['text_primary']),
                x=0.5,
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=DARK_THEME['border'],
                tickfont=dict(color=DARK_THEME['text_secondary'], size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=DARK_THEME['border'],
                tickfont=dict(color=DARK_THEME['text_secondary'], size=10),
                tickprefix='$',
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=10, color=DARK_THEME['text_secondary']),
            ),
            hovermode='x unified',
        )

    # Calculate summary stats
    total_trades = cumulative_series[-1].get('total_trades', len(cumulative_series)) if cumulative_series else 0
    max_pnl = max(cumulative_pnl) if cumulative_pnl else 0
    min_pnl = min(cumulative_pnl) if cumulative_pnl else 0
    max_drawdown = min_pnl if min_pnl < 0 else 0

    # Build the display
    children = [
        # Summary row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Small('Total P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H4(f'${final_pnl:+,.2f}', style={'color': pnl_color, 'marginBottom': '0'})
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Peak P/L', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${max_pnl:+,.2f}', style={
                        'color': DARK_THEME['accent_green'] if max_pnl > 0 else DARK_THEME['text_muted'],
                        'marginBottom': '0'
                    })
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Max Drawdown', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'${max_drawdown:,.2f}', style={
                        'color': DARK_THEME['accent_red'] if max_drawdown < 0 else DARK_THEME['text_muted'],
                        'marginBottom': '0'
                    })
                ], style={'textAlign': 'center'})
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Small('Total Trades', style={'color': DARK_THEME['text_secondary']}),
                    html.H5(f'{total_trades}', style={'color': DARK_THEME['text_primary'], 'marginBottom': '0'})
                ], style={'textAlign': 'center'})
            ], width=3),
        ], className='mb-3'),

        # Main equity curve
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ]

    # Add per-product chart if available
    if product_fig:
        children.append(dcc.Graph(figure=product_fig, config={'displayModeBar': False}))

    return html.Div(children, style={'padding': '0.5rem'})
