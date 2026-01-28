"""
Unified STRAT Analytics Panel - Session EQUITY-52

Unified STRAT pattern analytics dashboard that can switch between:
- Equity Options (Alpaca SMALL account)
- Crypto (VPS crypto trading)

Features:
- Market selector dropdown (Equity Options / Crypto)
- 6 sub-tabs: Overview, Patterns, TFC Comparison, Closed Trades, Pending, Equity Curve
- TFC comparison (WITH vs WITHOUT, threshold >= 4)
- No emojis (CLAUDE.md compliance)

Tab Structure (Reference: strat-analytics-dashboard.html):
1. Overview: 4 metrics cards + Win Rate by Pattern chart + Avg P&L by Pattern chart
2. Patterns: Best/Worst performers + Pattern breakdown table with ranking
3. TFC Comparison: WITH vs WITHOUT TFC comparison boxes + charts
4. Closed Trades: Trade table with Pattern, Entry, Exit, P&L, TFC
5. Pending Patterns: Symbol, Pattern, Status, Entry/Target/Stop
6. Equity Curve: 90-day account balance chart
"""

import dash_bootstrap_components as dbc
import dash_tvlwc
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from dashboard.config import COLORS, FONTS, REFRESH_INTERVALS, TVLWC_CHART_OPTIONS

# Dashboard Overhaul: Import progress bar function for Open Positions tab
from dashboard.components.options_panel import create_trade_progress_display

logger = logging.getLogger(__name__)


# ============================================
# THEME CONFIGURATION (Matching options_panel.py)
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
    'progress_bg': '#2a2a3e',
}

# Light theme for charts (matching reference design)
LIGHT_THEME = {
    'background': '#f8f9fa',
    'card_bg': '#ffffff',
    'border': '#e5e5e5',
    'text_primary': '#1a1a1a',
    'text_secondary': '#666666',
    'accent_green': '#059669',
    'accent_red': '#dc2626',
    'accent_blue': '#0066cc',
}

# TFC threshold: >= 4 means "WITH TFC"
TFC_THRESHOLD = 4

# Dashboard Overhaul Phase 4: Table display settings
MAX_TABLE_ROWS = 15  # Max rows before scroll
TABLE_HEADER_STYLE = {
    'position': 'sticky',
    'top': '0',
    'zIndex': '10',
    'backgroundColor': DARK_THEME['card_header'],
    'borderBottom': f'2px solid {DARK_THEME["border"]}',
    'color': DARK_THEME['text_primary']
}


# ============================================
# DATE FORMATTING HELPERS (DB-3)
# ============================================

def _format_open_date(pos: Dict) -> str:
    """
    Format the open date for a position from available fields.

    Options positions: entry_time_et (from signal store, already formatted as MM/DD HH:MM)
    Crypto positions: entry_time (ISO string from VPS API)

    Args:
        pos: Position dictionary

    Returns:
        Formatted date string or '-' if unavailable
    """
    # Options: entry_time_et is pre-formatted by options_loader
    entry_time = pos.get('entry_time_et') or ''
    if entry_time:
        return str(entry_time)

    # Crypto: entry_time is ISO string from VPS API
    entry_time_raw = pos.get('entry_time') or ''
    if entry_time_raw and isinstance(entry_time_raw, str):
        try:
            dt = datetime.fromisoformat(entry_time_raw.replace('Z', '+00:00'))
            return dt.strftime('%m/%d %H:%M')
        except (ValueError, TypeError):
            return str(entry_time_raw)[:16]

    return '-'


# ============================================
# MAIN PANEL COMPONENT
# ============================================

def create_strat_analytics_panel():
    """
    Create unified STRAT analytics panel with market selector and 6 tabs.

    Returns:
        Bootstrap container with unified analytics interface
    """
    return dbc.Container([
        # Header with market selector
        dbc.Row([
            dbc.Col([
                html.H3('STRAT Pattern Analytics', className='mb-0',
                        style={'color': DARK_THEME['text_primary']}),
                html.P('Pattern detection on underlying/spot -- execution via options or derivatives',
                       className='text-muted mb-0', style={'fontSize': '0.9rem'})
            ], width=8),
            dbc.Col([
                # Account selector (for Equity Options only - hidden for Crypto)
                dbc.Select(
                    id='strat-account-selector',
                    options=[
                        {'label': 'SMALL ($3k)', 'value': 'SMALL'},
                        {'label': 'MID ($5k)', 'value': 'MID'},
                        {'label': 'LARGE ($10k)', 'value': 'LARGE'},
                    ],
                    value='SMALL',
                    style={
                        'backgroundColor': DARK_THEME['input_bg'],
                        'color': DARK_THEME['text_primary'],
                        'border': f'1px solid {DARK_THEME["border"]}',
                        'width': '140px',
                        'marginRight': '10px',
                    }
                ),
                # Strategy selector (EQUITY-93B: Filter by trading strategy)
                dbc.Select(
                    id='strat-strategy-selector',
                    options=[
                        {'label': 'All Strategies', 'value': 'all'},
                        {'label': 'STRAT Patterns', 'value': 'strat'},
                        {'label': 'StatArb Pairs', 'value': 'statarb'},
                    ],
                    value='all',
                    style={
                        'backgroundColor': DARK_THEME['input_bg'],
                        'color': DARK_THEME['text_primary'],
                        'border': f'1px solid {DARK_THEME["border"]}',
                        'width': '140px',
                        'marginRight': '10px',
                    }
                ),
                # Market selector
                dbc.Select(
                    id='strat-market-selector',
                    options=[
                        {'label': 'Equity Options', 'value': 'options'},
                        {'label': 'Crypto (Spot/Derivs)', 'value': 'crypto'},
                    ],
                    value='options',
                    style={
                        'backgroundColor': DARK_THEME['input_bg'],
                        'color': DARK_THEME['text_primary'],
                        'border': f'1px solid {DARK_THEME["border"]}',
                        'width': '160px',
                    }
                )
            ], width=4, className='d-flex align-items-center justify-content-end')
        ], className='mb-3 p-3', style={
            'backgroundColor': DARK_THEME['card_bg'],
            'borderRadius': '8px',
            'border': f'1px solid {DARK_THEME["border"]}'
        }),

        # Tab navigation
        dbc.Tabs([
            dbc.Tab(label='Overview', tab_id='tab-overview'),
            dbc.Tab(label='Open Positions', tab_id='tab-positions'),  # Dashboard Overhaul: Restored
            dbc.Tab(label='Patterns', tab_id='tab-patterns'),
            dbc.Tab(label='Timeframe Continuity', tab_id='tab-tfc'),
            dbc.Tab(label='Closed Trades', tab_id='tab-closed'),
            dbc.Tab(label='Pending Patterns', tab_id='tab-pending'),
            dbc.Tab(label='Equity Curve', tab_id='tab-equity'),
        ], id='strat-analytics-tabs', active_tab='tab-overview', className='mb-3'),

        # Tab content area
        html.Div(id='strat-analytics-tab-content'),

        # Auto-refresh interval
        dcc.Interval(
            id='strat-analytics-refresh',
            interval=30 * 1000,  # 30 seconds
            n_intervals=0
        ),

        # Store for current market selection
        dcc.Store(id='strat-current-market', data='options'),

    ], fluid=True, style={'backgroundColor': DARK_THEME['background'], 'minHeight': '100vh'})


# ============================================
# OVERVIEW TAB
# ============================================

def create_overview_tab(metrics: Dict, pattern_stats: Dict) -> html.Div:
    """
    Create Overview tab with metrics cards and charts.

    Args:
        metrics: Dictionary with total_trades, win_rate, total_pnl, avg_pnl, etc.
        pattern_stats: Dictionary mapping pattern -> stats

    Returns:
        Overview tab content
    """
    return html.Div([
        # Metrics cards row
        dbc.Row([
            _create_metric_card(
                'TOTAL TRADES',
                str(metrics.get('total_trades', 0)),
                f"{metrics.get('winning_trades', 0)}W / {metrics.get('losing_trades', 0)}L"
            ),
            _create_metric_card(
                'WIN RATE',
                f"{metrics.get('win_rate', 0):.1f}%",
                f"{metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)} trades",
                value_color=DARK_THEME['accent_green'] if metrics.get('win_rate', 0) >= 50 else DARK_THEME['accent_red']
            ),
            _create_metric_card(
                'TOTAL P&L',
                f"${metrics.get('total_pnl', 0):,.2f}",
                f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
                value_color=DARK_THEME['accent_green'] if metrics.get('total_pnl', 0) >= 0 else DARK_THEME['accent_red']
            ),
            _create_metric_card(
                'AVG TRADE',
                f"${metrics.get('avg_pnl', 0):,.2f}",
                f"W: ${metrics.get('avg_win', 0):.2f} | L: ${metrics.get('avg_loss', 0):.2f}",
                value_color=DARK_THEME['accent_green'] if metrics.get('avg_pnl', 0) >= 0 else DARK_THEME['accent_red']
            ),
        ], className='mb-4'),

        # Win Rate by Pattern - Dashboard Overhaul: Replaced Plotly with progress bars
        dbc.Card([
            dbc.CardHeader('Win Rate by Pattern', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_win_rate_bars(pattern_stats)
            ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '12px 16px'})
        ], className='mb-3', style={'border': f'1px solid {DARK_THEME["border"]}'}),

        # Avg P&L by Pattern - Dashboard Overhaul: Replaced Plotly with progress bars
        dbc.Card([
            dbc.CardHeader('Average P&L by Pattern', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_pnl_bars(pattern_stats)
            ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '12px 16px'})
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


def _create_metric_card(label: str, value: str, subtext: str,
                        value_color: str = None) -> dbc.Col:
    """Create a metric card."""
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div(label, style={
                    'fontSize': '0.85rem',
                    'color': DARK_THEME['text_secondary'],
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'fontWeight': '500',
                    'marginBottom': '8px'
                }),
                html.Div(value, style={
                    'fontSize': '1.8rem',
                    'fontWeight': '600',
                    'color': value_color or DARK_THEME['text_primary'],
                    'marginBottom': '4px'
                }),
                html.Div(subtext, style={
                    'fontSize': '0.85rem',
                    'color': DARK_THEME['text_secondary']
                })
            ], style={'padding': '20px'})
        ], style={
            'backgroundColor': DARK_THEME['card_bg'],
            'border': f'1px solid {DARK_THEME["border"]}',
            'borderRadius': '8px'
        })
    ], width=3)


def _create_win_rate_bars(pattern_stats: Dict) -> html.Div:
    """
    Create win rate by pattern with horizontal progress bars.

    Dashboard Overhaul: Replaced Plotly bar chart with clean HTML progress bars.
    """
    if not pattern_stats:
        return html.Div('No pattern data available', style={
            'textAlign': 'center',
            'padding': '40px',
            'color': DARK_THEME['text_secondary']
        })

    # Sort by win rate descending
    sorted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1].get('win_rate', 0),
        reverse=True
    )

    rows = []
    for pattern, stats in sorted_patterns:
        win_rate = stats.get('win_rate', 0)
        trades = stats.get('total_trades', 0)  # Fix: was 'trades', should be 'total_trades'

        # Color based on win rate
        bar_color = DARK_THEME['accent_green'] if win_rate >= 50 else DARK_THEME['accent_red']

        rows.append(
            html.Div([
                # Pattern name (left)
                html.Div(pattern, style={
                    'flex': '0 0 100px',
                    'fontWeight': '500',
                    'color': DARK_THEME['text_primary'],
                    'fontSize': '0.9rem'
                }),

                # Progress bar (middle)
                html.Div([
                    html.Div([
                        # Background
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'right': '0',
                            'bottom': '0',
                            'backgroundColor': '#e5e5e5',
                            'borderRadius': '4px'
                        }),
                        # Fill
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'bottom': '0',
                            'width': f'{min(100, win_rate)}%',
                            'backgroundColor': bar_color,
                            'borderRadius': '4px',
                            'transition': 'width 0.3s ease'
                        }),
                    ], style={
                        'position': 'relative',
                        'height': '10px',
                        'borderRadius': '4px'
                    })
                ], style={'flex': '1', 'padding': '0 1rem'}),

                # Win rate value (right)
                html.Div([
                    html.Span(f"{win_rate:.1f}%", style={
                        'fontWeight': '600',
                        'color': bar_color,
                        'fontSize': '0.9rem'
                    }),
                    html.Span(f" ({trades})", style={
                        'color': DARK_THEME['text_secondary'],
                        'fontSize': '0.8rem',
                        'marginLeft': '4px'
                    }),
                ], style={'flex': '0 0 80px', 'textAlign': 'right'}),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '8px 0',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            })
        )

    return html.Div(rows, style={'padding': '8px 0'})


def _create_pnl_bars(pattern_stats: Dict) -> html.Div:
    """
    Create avg P&L by pattern with horizontal progress bars.

    Dashboard Overhaul: Replaced Plotly bar chart with clean HTML progress bars.
    """
    if not pattern_stats:
        return html.Div('No pattern data available', style={
            'textAlign': 'center',
            'padding': '40px',
            'color': DARK_THEME['text_secondary']
        })

    # Sort by avg P&L descending
    sorted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1].get('avg_pnl', 0),
        reverse=True
    )

    # Calculate max absolute P&L for scaling
    max_abs_pnl = max(abs(stats.get('avg_pnl', 0)) for _, stats in sorted_patterns) or 1

    rows = []
    for pattern, stats in sorted_patterns:
        avg_pnl = stats.get('avg_pnl', 0)

        # Color based on P&L
        bar_color = DARK_THEME['accent_green'] if avg_pnl >= 0 else DARK_THEME['accent_red']

        # Scale bar width relative to max
        bar_width = abs(avg_pnl) / max_abs_pnl * 100

        rows.append(
            html.Div([
                # Pattern name (left)
                html.Div(pattern, style={
                    'flex': '0 0 100px',
                    'fontWeight': '500',
                    'color': DARK_THEME['text_primary'],
                    'fontSize': '0.9rem'
                }),

                # Progress bar (middle)
                html.Div([
                    html.Div([
                        # Background
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'right': '0',
                            'bottom': '0',
                            'backgroundColor': '#e5e5e5',
                            'borderRadius': '4px'
                        }),
                        # Fill
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'bottom': '0',
                            'width': f'{bar_width}%',
                            'backgroundColor': bar_color,
                            'borderRadius': '4px',
                            'transition': 'width 0.3s ease'
                        }),
                    ], style={
                        'position': 'relative',
                        'height': '10px',
                        'borderRadius': '4px'
                    })
                ], style={'flex': '1', 'padding': '0 1rem'}),

                # P&L value (right)
                html.Div(f"${avg_pnl:,.2f}", style={
                    'flex': '0 0 80px',
                    'textAlign': 'right',
                    'fontWeight': '600',
                    'color': bar_color,
                    'fontSize': '0.9rem'
                }),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '8px 0',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            })
        )

    return html.Div(rows, style={'padding': '8px 0'})


# Legacy chart functions (kept for backward compatibility)
def _create_win_rate_chart(pattern_stats: Dict) -> go.Figure:
    """Create win rate by pattern bar chart. DEPRECATED - use _create_win_rate_bars()."""
    patterns = list(pattern_stats.keys())
    win_rates = [pattern_stats[p].get('win_rate', 0) for p in patterns]

    fig = go.Figure(data=[
        go.Bar(
            x=patterns,
            y=win_rates,
            marker_color=DARK_THEME['accent_blue'],
            text=[f'{wr:.1f}%' for wr in win_rates],
            textposition='auto'
        )
    ])

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=60),
        height=300,
        yaxis=dict(range=[0, 100], title='Win Rate (%)'),
        xaxis=dict(title=''),
        plot_bgcolor=DARK_THEME['background'],
        paper_bgcolor=DARK_THEME['card_bg'],
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color=DARK_THEME['text_primary'])
    )

    return fig


def _create_pnl_chart(pattern_stats: Dict) -> go.Figure:
    """Create avg P&L by pattern bar chart. DEPRECATED - use _create_pnl_bars()."""
    patterns = list(pattern_stats.keys())
    avg_pnls = [pattern_stats[p].get('avg_pnl', 0) for p in patterns]
    colors = [DARK_THEME['accent_green'] if p >= 0 else DARK_THEME['accent_red'] for p in avg_pnls]

    fig = go.Figure(data=[
        go.Bar(
            x=patterns,
            y=avg_pnls,
            marker_color=colors,
            text=[f'${p:,.2f}' for p in avg_pnls],
            textposition='auto'
        )
    ])

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=60),
        height=300,
        yaxis=dict(title='Average P&L ($)'),
        xaxis=dict(title=''),
        plot_bgcolor=DARK_THEME['background'],
        paper_bgcolor=DARK_THEME['card_bg'],
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color=DARK_THEME['text_primary'])
    )

    return fig


# ============================================
# OPEN POSITIONS TAB (Dashboard Overhaul: Restored)
# ============================================

def create_open_positions_tab(
    positions: List[Dict],
    progress_data: List[Dict],
    account_info: Dict
) -> html.Div:
    """
    Create Open Positions tab with account summary and progress bars.

    Dashboard Overhaul: Restored functionality lost during EQUITY-52 consolidation.

    Args:
        positions: List of open position dicts from Alpaca
        progress_data: List of trade progress data for progress bars
        account_info: Account summary dict with equity, cash, buying_power

    Returns:
        Open Positions tab content
    """
    # Account Summary Section
    equity = account_info.get('equity', 0) if account_info else 0
    cash = account_info.get('cash', 0) if account_info else 0
    buying_power = account_info.get('buying_power', 0) if account_info else 0

    account_section = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div('ACCOUNT BALANCE', style={
                        'fontSize': '0.75rem',
                        'color': DARK_THEME['text_secondary'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '1px',
                        'marginBottom': '8px'
                    }),
                    html.H3(f"${equity:,.2f}", style={
                        'fontWeight': '600',
                        'color': DARK_THEME['text_primary'],
                        'marginBottom': '8px'
                    }),
                    html.Small([
                        f"Cash: ${cash:,.2f}  |  ",
                        f"Buying Power: ${buying_power:,.2f}"
                    ], style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '20px'})
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'border': f'1px solid {DARK_THEME["border"]}',
                'borderRadius': '8px'
            })
        ], width=12)
    ], className='mb-3') if account_info and not account_info.get('error') else html.Div()

    # Positions Table Section - Dashboard Overhaul Phase 4: Sticky headers
    if positions:
        positions_table_content = html.Table([
            html.Thead([
                html.Tr([
                    html.Th('Symbol', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('Pattern', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('Opened', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('Qty', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('Entry', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('Current', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('P&L', style={'padding': '12px 16px', 'fontWeight': '600'}),
                    html.Th('%', style={'padding': '12px 16px', 'fontWeight': '600'}),
                ], style=TABLE_HEADER_STYLE)
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(
                        html.Strong((pos.get('display_contract') or pos.get('symbol', '-'))[:20]),
                        style={'padding': '12px 16px'}
                    ),
                    html.Td(
                        html.Span(pos.get('pattern', '-'), style={
                            'backgroundColor': '#1e3a5f',
                            'color': '#93c5fd',
                            'padding': '3px 8px',
                            'borderRadius': '4px',
                            'fontSize': '0.85rem',
                            'fontWeight': '500'
                        }),
                        style={'padding': '12px 16px'}
                    ),
                    # DB-3: Open date/time column
                    # Options: entry_time_et from signal store lookup
                    # Crypto: entry_time from VPS API
                    html.Td(
                        _format_open_date(pos),
                        style={
                            'padding': '12px 16px',
                            'color': DARK_THEME['text_secondary'],
                            'fontSize': '0.9rem'
                        }
                    ),
                    html.Td(str(pos.get('qty', 0)), style={'padding': '12px 16px'}),
                    html.Td(
                        f"${float(pos.get('avg_entry_price', 0)):.2f}",
                        style={'padding': '12px 16px'}
                    ),
                    html.Td(
                        f"${float(pos.get('current_price', 0)):.2f}",
                        style={'padding': '12px 16px'}
                    ),
                    html.Td(
                        f"${float(pos.get('unrealized_pl', 0)):,.2f}",
                        style={
                            'padding': '12px 16px',
                            'color': DARK_THEME['accent_green']
                            if float(pos.get('unrealized_pl', 0)) >= 0
                            else DARK_THEME['accent_red']
                        }
                    ),
                    html.Td(
                        f"{float(pos.get('unrealized_plpc', 0)) * 100:.1f}%",
                        style={
                            'padding': '12px 16px',
                            'color': DARK_THEME['accent_green']
                            if float(pos.get('unrealized_plpc', 0)) >= 0
                            else DARK_THEME['accent_red']
                        }
                    ),
                ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
                for pos in positions
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'fontSize': '0.95rem',
            'color': DARK_THEME['text_primary']
        })
        positions_table = html.Div(positions_table_content, style={
            'maxHeight': f'{MAX_TABLE_ROWS * 48}px',
            'overflowY': 'auto'
        })
    else:
        positions_table = html.Div(
            'No open positions',
            style={
                'textAlign': 'center',
                'padding': '40px 20px',
                'color': DARK_THEME['text_secondary']
            }
        )

    positions_section = dbc.Card([
        dbc.CardHeader('Open Positions', style={
            'backgroundColor': DARK_THEME['card_bg'],
            'fontWeight': '600',
            'borderBottom': f'1px solid {DARK_THEME["border"]}'
        }),
        dbc.CardBody([
            positions_table
        ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': 0})
    ], className='mb-3', style={'border': f'1px solid {DARK_THEME["border"]}'})

    # Progress Bars Section (using restored function from options_panel)
    progress_section = dbc.Card([
        dbc.CardHeader('Trade Progress to Target', style={
            'backgroundColor': DARK_THEME['card_bg'],
            'fontWeight': '600',
            'color': DARK_THEME['text_primary'],
            'borderBottom': f'1px solid {DARK_THEME["border"]}'
        }),
        dbc.CardBody([
            create_trade_progress_display(progress_data)
        ], style={
            'backgroundColor': DARK_THEME['card_bg'],
            'padding': '16px'
        })
    ], style={
        'border': f'1px solid {DARK_THEME["border"]}',
        'borderRadius': '8px'
    })

    return html.Div([
        account_section,
        positions_section,
        progress_section,
    ])


# ============================================
# PATTERNS TAB
# ============================================

def create_patterns_tab(pattern_stats: Dict) -> html.Div:
    """
    Create Patterns tab with best/worst performers and breakdown table.

    Args:
        pattern_stats: Dictionary mapping pattern -> stats

    Returns:
        Patterns tab content
    """
    # Sort patterns by win rate
    sorted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1].get('win_rate', 0),
        reverse=True
    )

    best = sorted_patterns[0] if sorted_patterns else ('N/A', {})
    worst = sorted_patterns[-1] if sorted_patterns else ('N/A', {})

    # Find highest avg P&L
    by_pnl = sorted(
        pattern_stats.items(),
        key=lambda x: x[1].get('avg_pnl', 0),
        reverse=True
    )
    highest_pnl = by_pnl[0] if by_pnl else ('N/A', {})

    return html.Div([
        # Top performers row
        dbc.Row([
            _create_metric_card(
                'BEST PERFORMER',
                best[0],
                f"{best[1].get('win_rate', 0):.1f}% WR | {best[1].get('total_trades', 0)} trades"
            ),
            _create_metric_card(
                'HIGHEST AVG P&L',
                f"${highest_pnl[1].get('avg_pnl', 0):,.2f}",
                highest_pnl[0],
                value_color=DARK_THEME['accent_green']
            ),
            _create_metric_card(
                'TOTAL PATTERNS',
                str(len(pattern_stats)),
                'Across all trades'
            ),
            _create_metric_card(
                'WEAKEST PERFORMER',
                worst[0],
                f"{worst[1].get('win_rate', 0):.1f}% WR | {worst[1].get('total_trades', 0)} trades"
            ),
        ], className='mb-4'),

        # Pattern breakdown table
        dbc.Card([
            dbc.CardHeader('Pattern Performance Breakdown', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_pattern_table(sorted_patterns)
            ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': 0})
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


def _create_pattern_table(sorted_patterns: List[Tuple]) -> html.Div:
    """Create pattern breakdown table with sticky headers.

    Dashboard Overhaul Phase 4: Added sticky headers.
    """
    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Pattern', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Trades', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Win Rate', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Avg P&L', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Rank', style={'padding': '12px 16px', 'fontWeight': '600'}),
            ], style=TABLE_HEADER_STYLE)
        ]),
        html.Tbody([
            html.Tr([
                html.Td(
                    html.Span(pattern, style={
                        'backgroundColor': '#eff6ff',
                        'color': '#0c4a6e',
                        'padding': '3px 8px',
                        'borderRadius': '4px',
                        'fontSize': '0.85rem',
                        'fontWeight': '500'
                    }),
                    style={'padding': '12px 16px'}
                ),
                html.Td(str(stats.get('total_trades', 0)), style={'padding': '12px 16px'}),
                html.Td(
                    f"{stats.get('win_rate', 0):.1f}%",
                    style={'padding': '12px 16px', 'color': DARK_THEME['accent_green']}
                ),
                html.Td(
                    f"${stats.get('avg_pnl', 0):,.2f}",
                    style={
                        'padding': '12px 16px',
                        'color': DARK_THEME['accent_green'] if stats.get('avg_pnl', 0) >= 0 else DARK_THEME['accent_red']
                    }
                ),
                html.Td(f"#{rank + 1}", style={'padding': '12px 16px'}),
            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
            for rank, (pattern, stats) in enumerate(sorted_patterns)
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'fontSize': '0.95rem'
    })

    return html.Div(table, style={
        'maxHeight': f'{MAX_TABLE_ROWS * 48}px',
        'overflowY': 'auto'
    })


# ============================================
# TFC COMPARISON TAB
# ============================================

def create_tfc_tab(trades: List[Dict]) -> html.Div:
    """
    Create TFC comparison tab.

    TFC threshold: >= 4 = WITH TFC, < 4 = WITHOUT TFC

    Args:
        trades: List of closed trades with tfc_score field

    Returns:
        TFC comparison tab content
    """
    with_tfc, without_tfc = _calculate_tfc_comparison(trades)

    return html.Div([
        # Comparison boxes
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div('WITH Timeframe Continuity (TFC >= 4)', style={
                            'fontWeight': '600',
                            'marginBottom': '15px',
                            'color': DARK_THEME['text_primary']
                        }),
                        _tfc_stat_item('Total Trades', str(with_tfc['total_trades'])),
                        _tfc_stat_item('Win Rate', f"{with_tfc['win_rate']:.1f}%",
                                       color=DARK_THEME['accent_green']),
                        _tfc_stat_item('Avg P&L', f"${with_tfc['avg_pnl']:.2f}",
                                       color=DARK_THEME['accent_green'] if with_tfc['avg_pnl'] >= 0 else DARK_THEME['accent_red']),
                    ], style={'padding': '20px'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}',
                    'borderRadius': '8px'
                })
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div('WITHOUT Timeframe Continuity (TFC < 4)', style={
                            'fontWeight': '600',
                            'marginBottom': '15px',
                            'color': DARK_THEME['text_primary']
                        }),
                        _tfc_stat_item('Total Trades', str(without_tfc['total_trades'])),
                        _tfc_stat_item('Win Rate', f"{without_tfc['win_rate']:.1f}%",
                                       color=DARK_THEME['accent_red']),
                        _tfc_stat_item('Avg P&L', f"${without_tfc['avg_pnl']:.2f}",
                                       color=DARK_THEME['accent_green'] if without_tfc['avg_pnl'] >= 0 else DARK_THEME['accent_red']),
                    ], style={'padding': '20px'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}',
                    'borderRadius': '8px'
                })
            ], width=6),
        ], className='mb-4'),

        # Win Rate Comparison Chart
        dbc.Card([
            dbc.CardHeader('Win Rate Comparison', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                dcc.Graph(
                    figure=_create_tfc_comparison_chart(
                        with_tfc['win_rate'], without_tfc['win_rate'],
                        'Win Rate (%)', max_y=100
                    ),
                    config={'displayModeBar': False}
                )
            ], style={'backgroundColor': DARK_THEME['card_bg']})
        ], className='mb-3', style={'border': f'1px solid {DARK_THEME["border"]}'}),

        # P&L Comparison Chart
        dbc.Card([
            dbc.CardHeader('Average P&L Comparison', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                dcc.Graph(
                    figure=_create_tfc_pnl_comparison_chart(
                        with_tfc['avg_pnl'], without_tfc['avg_pnl']
                    ),
                    config={'displayModeBar': False}
                )
            ], style={'backgroundColor': DARK_THEME['card_bg']})
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


def _tfc_stat_item(label: str, value: str, color: str = None) -> html.Div:
    """Create a TFC stat item."""
    return html.Div([
        html.Div(label, style={
            'color': DARK_THEME['text_secondary'],
            'fontSize': '0.9rem',
            'marginBottom': '3px'
        }),
        html.Div(value, style={
            'fontSize': '1.4rem',
            'fontWeight': '600',
            'color': color or DARK_THEME['text_primary']
        })
    ], style={'marginBottom': '10px'})


def _calculate_tfc_comparison(trades: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Calculate TFC comparison metrics.

    Args:
        trades: List of closed trades

    Returns:
        Tuple of (with_tfc_metrics, without_tfc_metrics)
    """
    with_tfc_trades = [t for t in trades if (t.get('tfc_score') or 0) >= TFC_THRESHOLD]
    without_tfc_trades = [t for t in trades if (t.get('tfc_score') or 0) < TFC_THRESHOLD]

    def calc_metrics(trade_list: List[Dict]) -> Dict:
        if not trade_list:
            return {'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0}

        wins = sum(1 for t in trade_list if (t.get('pnl') or 0) > 0)
        total_pnl = sum(t.get('pnl') or 0 for t in trade_list)

        return {
            'total_trades': len(trade_list),
            'win_rate': (wins / len(trade_list)) * 100 if trade_list else 0,
            'avg_pnl': total_pnl / len(trade_list) if trade_list else 0
        }

    return calc_metrics(with_tfc_trades), calc_metrics(without_tfc_trades)


def _create_tfc_comparison_chart(with_val: float, without_val: float,
                                  y_label: str, max_y: float = None) -> go.Figure:
    """Create TFC comparison bar chart."""
    # Handle empty/zero data state
    if with_val == 0 and without_val == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No TFC data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=DARK_THEME['text_secondary'], size=14)
        )
        fig.update_layout(
            height=300,
            plot_bgcolor=DARK_THEME['background'],
            paper_bgcolor=DARK_THEME['card_bg'],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    fig = go.Figure(data=[
        go.Bar(
            x=['With Continuity', 'Without Continuity'],
            y=[with_val, without_val],
            marker_color=[DARK_THEME['accent_green'], DARK_THEME['accent_red']],
            text=[f'{with_val:.1f}%', f'{without_val:.1f}%'],
            textposition='auto'
        )
    ])

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        yaxis=dict(title=y_label, range=[0, max_y] if max_y else None),
        plot_bgcolor=DARK_THEME['background'],
        paper_bgcolor=DARK_THEME['card_bg'],
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color=DARK_THEME['text_primary'])
    )

    return fig


def _create_tfc_pnl_comparison_chart(with_pnl: float, without_pnl: float) -> go.Figure:
    """Create TFC P&L comparison bar chart."""
    # Handle empty data state
    if with_pnl == 0 and without_pnl == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No P&L data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=DARK_THEME['text_secondary'], size=14)
        )
        fig.update_layout(
            height=300,
            plot_bgcolor=DARK_THEME['background'],
            paper_bgcolor=DARK_THEME['card_bg'],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    colors = [
        DARK_THEME['accent_green'] if with_pnl >= 0 else DARK_THEME['accent_red'],
        DARK_THEME['accent_green'] if without_pnl >= 0 else DARK_THEME['accent_red']
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=['With Continuity', 'Without Continuity'],
            y=[with_pnl, without_pnl],
            marker_color=colors,
            text=[f'${with_pnl:,.2f}', f'${without_pnl:,.2f}'],
            textposition='auto'
        )
    ])

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        yaxis=dict(title='Average P&L ($)'),
        plot_bgcolor=DARK_THEME['background'],
        paper_bgcolor=DARK_THEME['card_bg'],
        font=dict(family='-apple-system, BlinkMacSystemFont, sans-serif', color=DARK_THEME['text_primary'])
    )

    return fig


# ============================================
# CLOSED TRADES TAB
# ============================================

def create_closed_trades_tab(trades: List[Dict]) -> html.Div:
    """
    Create Closed Trades tab with trade table.

    Args:
        trades: List of closed trades

    Returns:
        Closed trades tab content
    """
    # Sort by date descending, limit to 20
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get('sell_time_dt') or datetime.min,
        reverse=True
    )[:20]

    return html.Div([
        dbc.Card([
            dbc.CardHeader('Recent Closed Trades', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_trades_table(sorted_trades)
            ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': 0})
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


def _create_trades_table(trades: List[Dict]) -> html.Div:
    """Create closed trades table with sticky headers and scroll.

    Dashboard Overhaul Phase 4: Added sticky headers and max height scroll.
    """
    if not trades:
        return html.Div('No closed trades found', style={
            'textAlign': 'center',
            'padding': '40px 20px',
            'color': DARK_THEME['text_secondary']
        })

    total_trades = len(trades)
    display_trades = trades[:MAX_TABLE_ROWS * 2]  # Show up to 2 pages worth

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Symbol', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Pattern', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Closed', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Entry', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Exit', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('P&L', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('%', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Continuity', style={'padding': '12px 16px', 'fontWeight': '600'}),
            ], style=TABLE_HEADER_STYLE)
        ]),
        html.Tbody([
            _create_trade_row(trade) for trade in display_trades
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'fontSize': '0.95rem'
    })

    # Wrap in scrollable container
    return html.Div([
        html.Div(table, style={
            'maxHeight': f'{MAX_TABLE_ROWS * 48}px',  # ~48px per row
            'overflowY': 'auto',
            'overflowX': 'auto'
        }),
        # Row count indicator
        html.Div(
            f"Showing {len(display_trades)} of {total_trades} trades",
            style={
                'padding': '8px 16px',
                'fontSize': '0.85rem',
                'color': DARK_THEME['text_secondary'],
                'borderTop': f'1px solid {DARK_THEME["border"]}',
                'backgroundColor': DARK_THEME['card_bg']
            }
        ) if total_trades > MAX_TABLE_ROWS else None
    ])


def _create_trade_row(trade: Dict) -> html.Tr:
    """Create a trade table row."""
    pnl = trade.get('pnl') or 0
    pnl_pct = trade.get('pnl_pct') or 0
    tfc_score = trade.get('tfc_score') or 0
    has_tfc = tfc_score >= TFC_THRESHOLD

    # Get best available display symbol - prefer display_contract for options
    symbol_display = trade.get('display_contract') or trade.get('display_symbol') or trade.get('symbol', '-')

    # DB-3: Get close date/time from available fields
    # Options: sell_time_display (formatted by options_loader)
    # Crypto: exit_time (ISO string from API)
    close_date = trade.get('sell_time_display') or ''
    if not close_date:
        exit_time = trade.get('exit_time') or ''
        if exit_time and isinstance(exit_time, str):
            try:
                dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                close_date = dt.strftime('%m/%d %H:%M')
            except (ValueError, TypeError):
                close_date = str(exit_time)[:16]
        elif hasattr(exit_time, 'strftime'):
            close_date = exit_time.strftime('%m/%d %H:%M')

    return html.Tr([
        html.Td(
            html.Strong(symbol_display[:20]),
            style={'padding': '12px 16px', 'color': DARK_THEME['text_primary']}
        ),
        html.Td(
            html.Span(trade.get('pattern', '-'), style={
                'backgroundColor': '#1e3a5f',
                'color': '#93c5fd',
                'padding': '3px 8px',
                'borderRadius': '4px',
                'fontSize': '0.85rem',
                'fontWeight': '500'
            }),
            style={'padding': '12px 16px'}
        ),
        html.Td(
            close_date or '-',
            style={
                'padding': '12px 16px',
                'color': DARK_THEME['text_secondary'],
                'fontSize': '0.9rem'
            }
        ),
        html.Td(
            f"${trade.get('entry_price', 0):.2f}",
            style={'padding': '12px 16px', 'color': DARK_THEME['text_primary']}
        ),
        html.Td(
            f"${trade.get('exit_price', 0):.2f}",
            style={'padding': '12px 16px', 'color': DARK_THEME['text_primary']}
        ),
        html.Td(
            f"${pnl:,.2f}",
            style={
                'padding': '12px 16px',
                'color': DARK_THEME['accent_green'] if pnl >= 0 else DARK_THEME['accent_red']
            }
        ),
        html.Td(
            f"{pnl_pct:.1f}%",
            style={
                'padding': '12px 16px',
                'color': DARK_THEME['accent_green'] if pnl_pct >= 0 else DARK_THEME['accent_red']
            }
        ),
        html.Td(
            html.Span(
                'Yes' if has_tfc else 'No',
                style={
                    'color': DARK_THEME['accent_green'] if has_tfc else DARK_THEME['accent_red'],
                    'fontWeight': '600'
                }
            ),
            style={'padding': '12px 16px'}
        ),
    ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})


# ============================================
# PENDING PATTERNS TAB
# ============================================

def create_pending_tab(signals: List[Dict]) -> html.Div:
    """
    Create Pending Patterns tab.

    Args:
        signals: List of pending signals

    Returns:
        Pending patterns tab content
    """
    return html.Div([
        dbc.Card([
            dbc.CardHeader('Pending Pattern Confirmation', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_pending_table(signals)
            ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': 0})
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


def _create_pending_table(signals: List[Dict]) -> html.Div:
    """Create pending patterns table with sticky headers.

    Dashboard Overhaul Phase 4: Added sticky headers and max height scroll.
    """
    if not signals:
        return html.Div('No pending patterns', style={
            'textAlign': 'center',
            'padding': '40px 20px',
            'color': DARK_THEME['text_secondary']
        })

    total_signals = len(signals)
    display_signals = signals[:MAX_TABLE_ROWS * 2]

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Symbol', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Pattern', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Timeframe', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Entry', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Target', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Stop', style={'padding': '12px 16px', 'fontWeight': '600'}),
                html.Th('Status', style={'padding': '12px 16px', 'fontWeight': '600'}),
            ], style=TABLE_HEADER_STYLE)
        ]),
        html.Tbody([
            _create_pending_row(signal) for signal in display_signals
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'fontSize': '0.95rem'
    })

    return html.Div([
        html.Div(table, style={
            'maxHeight': f'{MAX_TABLE_ROWS * 48}px',
            'overflowY': 'auto',
            'overflowX': 'auto'
        }),
        html.Div(
            f"Showing {len(display_signals)} of {total_signals} signals",
            style={
                'padding': '8px 16px',
                'fontSize': '0.85rem',
                'color': DARK_THEME['text_secondary'],
                'borderTop': f'1px solid {DARK_THEME["border"]}',
                'backgroundColor': DARK_THEME['card_bg']
            }
        ) if total_signals > MAX_TABLE_ROWS else None
    ])


def _create_pending_row(signal: Dict) -> html.Tr:
    """Create a pending pattern table row."""
    status = signal.get('status', 'PENDING')
    status_styles = {
        'DETECTED': {'backgroundColor': '#fef3c7', 'color': '#92400e'},
        'ALERTED': {'backgroundColor': '#dbeafe', 'color': '#1e40af'},
        'TRIGGERED': {'backgroundColor': '#d1fae5', 'color': '#065f46'},
        'PENDING': {'backgroundColor': '#f3f4f6', 'color': '#374151'},
    }
    style = status_styles.get(status.upper(), status_styles['PENDING'])

    return html.Tr([
        html.Td(
            html.Strong(signal.get('symbol', '-')),
            style={'padding': '12px 16px'}
        ),
        html.Td(
            html.Span(signal.get('pattern_type', '-'), style={
                'backgroundColor': '#eff6ff',
                'color': '#0c4a6e',
                'padding': '3px 8px',
                'borderRadius': '4px',
                'fontSize': '0.85rem',
                'fontWeight': '500'
            }),
            style={'padding': '12px 16px'}
        ),
        html.Td(signal.get('timeframe', '-'), style={'padding': '12px 16px'}),
        html.Td(f"${signal.get('entry_trigger', 0):.2f}", style={'padding': '12px 16px'}),
        html.Td(f"${signal.get('target_price', 0):.2f}", style={'padding': '12px 16px'}),
        html.Td(f"${signal.get('stop_price', 0):.2f}", style={'padding': '12px 16px'}),
        html.Td(
            html.Span(status.upper(), style={
                **style,
                'padding': '4px 10px',
                'borderRadius': '4px',
                'fontSize': '0.85rem',
                'fontWeight': '500',
                'display': 'inline-block'
            }),
            style={'padding': '12px 16px'}
        ),
    ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})


# ============================================
# EQUITY CURVE TAB
# ============================================

def _calculate_equity_stats(portfolio_history: List[Dict]) -> Dict:
    """
    Calculate summary statistics from portfolio history.

    Args:
        portfolio_history: List of portfolio snapshots with timestamp/date and equity/balance

    Returns:
        Dict with total_return_pct, total_return_dollar, max_drawdown_pct,
        start_equity, end_equity, days
    """
    if not portfolio_history:
        return {
            'total_return_pct': 0.0,
            'total_return_dollar': 0.0,
            'max_drawdown_pct': 0.0,
            'start_equity': 0.0,
            'end_equity': 0.0,
            'days': 0,
        }

    equities = [h.get('equity', h.get('balance', 0)) for h in portfolio_history]
    start_equity = equities[0] if equities else 0
    end_equity = equities[-1] if equities else 0

    total_return_dollar = end_equity - start_equity
    total_return_pct = (total_return_dollar / start_equity * 100) if start_equity else 0.0

    # Peak-to-trough max drawdown
    max_drawdown_pct = 0.0
    peak = equities[0] if equities else 0
    for eq in equities:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

    return {
        'total_return_pct': total_return_pct,
        'total_return_dollar': total_return_dollar,
        'max_drawdown_pct': max_drawdown_pct,
        'start_equity': start_equity,
        'end_equity': end_equity,
        'days': len(equities),
    }


def _create_stats_cards(stats: Dict) -> dbc.Row:
    """
    Create summary stats cards row for equity curve tab.

    Args:
        stats: Dict from _calculate_equity_stats()

    Returns:
        dbc.Row with 4 metric cards
    """
    ret_pct = stats['total_return_pct']
    ret_dollar = stats['total_return_dollar']
    dd_pct = stats['max_drawdown_pct']

    ret_color = COLORS['accent_emerald'] if ret_pct >= 0 else COLORS['accent_crimson']
    dd_color = COLORS['accent_crimson'] if dd_pct > 0 else COLORS['text_secondary']

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div('TOTAL RETURN', style={
                        'fontSize': '0.75rem',
                        'color': DARK_THEME['text_secondary'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '6px',
                    }),
                    html.Div(f"{ret_pct:+.2f}%", style={
                        'fontSize': '1.6rem',
                        'fontWeight': '600',
                        'color': ret_color,
                        'fontFamily': FONTS['mono'],
                    }),
                    html.Div(f"${ret_dollar:+,.2f}", style={
                        'fontSize': '0.85rem',
                        'color': ret_color,
                    }),
                ], style={'padding': '16px'})
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'border': f'1px solid {DARK_THEME["border"]}',
                'borderRadius': '8px',
            })
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div('MAX DRAWDOWN', style={
                        'fontSize': '0.75rem',
                        'color': DARK_THEME['text_secondary'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '6px',
                    }),
                    html.Div(f"-{dd_pct:.2f}%", style={
                        'fontSize': '1.6rem',
                        'fontWeight': '600',
                        'color': dd_color,
                        'fontFamily': FONTS['mono'],
                    }),
                    html.Div('Peak to trough', style={
                        'fontSize': '0.85rem',
                        'color': DARK_THEME['text_secondary'],
                    }),
                ], style={'padding': '16px'})
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'border': f'1px solid {DARK_THEME["border"]}',
                'borderRadius': '8px',
            })
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div('STARTING EQUITY', style={
                        'fontSize': '0.75rem',
                        'color': DARK_THEME['text_secondary'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '6px',
                    }),
                    html.Div(f"${stats['start_equity']:,.2f}", style={
                        'fontSize': '1.6rem',
                        'fontWeight': '600',
                        'color': DARK_THEME['text_primary'],
                        'fontFamily': FONTS['mono'],
                    }),
                    html.Div(f"{stats['days']} days tracked", style={
                        'fontSize': '0.85rem',
                        'color': DARK_THEME['text_secondary'],
                    }),
                ], style={'padding': '16px'})
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'border': f'1px solid {DARK_THEME["border"]}',
                'borderRadius': '8px',
            })
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div('CURRENT EQUITY', style={
                        'fontSize': '0.75rem',
                        'color': DARK_THEME['text_secondary'],
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px',
                        'marginBottom': '6px',
                    }),
                    html.Div(f"${stats['end_equity']:,.2f}", style={
                        'fontSize': '1.6rem',
                        'fontWeight': '600',
                        'color': DARK_THEME['text_primary'],
                        'fontFamily': FONTS['mono'],
                    }),
                    html.Div('Latest snapshot', style={
                        'fontSize': '0.85rem',
                        'color': DARK_THEME['text_secondary'],
                    }),
                ], style={'padding': '16px'})
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'border': f'1px solid {DARK_THEME["border"]}',
                'borderRadius': '8px',
            })
        ], width=3),
    ], className='mb-3')


def _create_equity_chart(history: List[Dict]):
    """
    Create TradingView Lightweight Charts equity curve.

    Args:
        history: List of portfolio snapshots with timestamp/date and equity/balance

    Returns:
        dash_tvlwc.Tvlwc component (or html.Div placeholder if no data)
    """
    if not history:
        return html.Div(
            'No portfolio history available',
            style={
                'textAlign': 'center',
                'padding': '80px 20px',
                'color': DARK_THEME['text_secondary'],
                'fontSize': '1.1rem',
                'height': '500px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
            }
        )

    # Transform data: {'timestamp'/'date': t, 'equity'/'balance': v} -> {'time': t, 'value': v}
    series_data = []
    for h in history:
        t = h.get('timestamp') or h.get('date', '')
        v = h.get('equity', h.get('balance', 0))
        if t:
            series_data.append({'time': str(t), 'value': float(v)})

    if not series_data:
        return html.Div(
            'No valid data points',
            style={
                'textAlign': 'center',
                'padding': '80px 20px',
                'color': DARK_THEME['text_secondary'],
            }
        )

    # Determine line color based on overall return
    start_val = series_data[0]['value']
    end_val = series_data[-1]['value']
    line_color = COLORS['accent_emerald'] if end_val >= start_val else COLORS['accent_crimson']

    # Area fill with low opacity matching the line color
    if end_val >= start_val:
        top_color = 'rgba(0, 220, 130, 0.25)'
        bottom_color = 'rgba(0, 220, 130, 0.02)'
    else:
        top_color = 'rgba(255, 59, 92, 0.25)'
        bottom_color = 'rgba(255, 59, 92, 0.02)'

    return dash_tvlwc.Tvlwc(
        id='equity-curve-tvlwc',
        seriesTypes=['Area'],
        seriesData=[series_data],
        seriesOptions=[{
            'lineColor': line_color,
            'topColor': top_color,
            'bottomColor': bottom_color,
            'lineWidth': 2,
            'priceFormat': {
                'type': 'price',
                'precision': 2,
                'minMove': 0.01,
            },
        }],
        chartOptions=TVLWC_CHART_OPTIONS,
        width='100%',
        height=500,
    )


def create_equity_tab(portfolio_history: List[Dict]) -> html.Div:
    """
    Create Equity Curve tab with summary stats and TradingView chart.

    Args:
        portfolio_history: List of portfolio snapshots with date, equity

    Returns:
        Equity curve tab content
    """
    stats = _calculate_equity_stats(portfolio_history)

    return html.Div([
        # Summary stats cards
        _create_stats_cards(stats),

        # TradingView Lightweight Chart
        dbc.Card([
            dbc.CardHeader('90-Day Account Equity Curve', style={
                'backgroundColor': DARK_THEME['card_bg'],
                'fontWeight': '600',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            }),
            dbc.CardBody([
                _create_equity_chart(portfolio_history),
            ], style={
                'backgroundColor': DARK_THEME['card_bg'],
                'padding': '8px',
            })
        ], style={'border': f'1px solid {DARK_THEME["border"]}'}),
    ])


# ============================================
# DATA CALCULATION HELPERS
# ============================================

def calculate_metrics(trades: List[Dict], strategy: str = 'all') -> Dict:
    """
    Calculate overview metrics from trades.

    Args:
        trades: List of closed trades
        strategy: Strategy filter ('all', 'strat', or 'statarb')

    Returns:
        Dictionary with metrics
    """
    # EQUITY-93B: Filter trades by strategy if specified
    if strategy and strategy != 'all':
        trades = [t for t in trades if t.get('strategy') == strategy]

    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
        }

    winners = [t for t in trades if (t.get('pnl') or 0) > 0]
    losers = [t for t in trades if (t.get('pnl') or 0) <= 0]

    total_pnl = sum(t.get('pnl') or 0 for t in trades)
    total_wins = sum(t.get('pnl') or 0 for t in winners)
    total_losses = sum(abs(t.get('pnl') or 0) for t in losers)

    return {
        'total_trades': len(trades),
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': (len(winners) / len(trades)) * 100 if trades else 0,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades) if trades else 0,
        'avg_win': total_wins / len(winners) if winners else 0,
        'avg_loss': -total_losses / len(losers) if losers else 0,
        'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
    }


def calculate_pattern_stats(trades: List[Dict], strategy: str = 'all') -> Dict:
    """
    Calculate statistics grouped by pattern.

    Args:
        trades: List of closed trades
        strategy: Strategy filter ('all', 'strat', or 'statarb')

    Returns:
        Dictionary mapping pattern -> stats
    """
    # EQUITY-93B: Filter trades by strategy if specified
    if strategy and strategy != 'all':
        trades = [t for t in trades if t.get('strategy') == strategy]

    patterns: Dict[str, Dict] = {}

    for trade in trades:
        # Try multiple field names for pattern data
        pattern = trade.get('pattern') or trade.get('pattern_type') or ''
        if not pattern or pattern in ['-', 'None', 'null', '']:
            pattern = 'Unclassified'

        if pattern not in patterns:
            patterns[pattern] = {'trades': [], 'wins': 0, 'total_pnl': 0}

        patterns[pattern]['trades'].append(trade)
        pnl = trade.get('pnl') or 0
        patterns[pattern]['total_pnl'] += pnl
        if pnl > 0:
            patterns[pattern]['wins'] += 1

    # Calculate stats
    stats = {}
    for pattern, data in patterns.items():
        total = len(data['trades'])
        stats[pattern] = {
            'total_trades': total,
            'win_rate': (data['wins'] / total) * 100 if total > 0 else 0,
            'avg_pnl': data['total_pnl'] / total if total > 0 else 0,
        }

    return stats
