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

from dashboard.config import COLORS, CHART_HEIGHT, REFRESH_INTERVALS


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

    Layout (Session 83K-78 Redesign):
    - Row 1: P&L Summary | Live Option Positions
    - Row 2: STRAT Signals with Tabs (Active Setups | Triggered | Low Magnitude)
    - Row 3: Trade Progress Visualization

    Returns:
        Bootstrap container with options trading interface
    """

    return dbc.Container([

        # ============================================
        # ROW 1: P&L Summary + Live Positions (Moved to Top)
        # ============================================
        dbc.Row([

            # Left: P&L Summary
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

            # Right: Live Option Positions
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-list-ol me-2', style={'color': DARK_THEME['accent_blue']}),
                        'Live Option Positions',
                        html.Span(id='options-positions-count', className='ms-2')
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        html.Div(id='options-positions-container', children=[
                            _create_no_positions_placeholder()
                        ])
                    ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '0'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}',
                    'height': '100%'
                }, className='shadow')
            ], width=12, lg=8, className='mb-3'),

        ], className='mb-4'),

        # ============================================
        # ROW 2: STRAT Signals with Tabs
        # ============================================
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-crosshairs me-2', style={'color': DARK_THEME['accent_yellow']}),
                        'STRAT Signals',
                        html.Span(id='options-signals-count', className='ms-2')
                    ], style={
                        'backgroundColor': DARK_THEME['card_header'],
                        'color': DARK_THEME['text_primary'],
                        'fontWeight': 'bold',
                        'borderBottom': f'1px solid {DARK_THEME["border"]}'
                    }),
                    dbc.CardBody([
                        # Signal Tabs
                        # Session EQUITY-33: Renamed tabs for clarity
                        dbc.Tabs([
                            dbc.Tab(
                                label='Pending Entries',  # Renamed from "Active Setups"
                                tab_id='tab-setups',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_blue'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='signals-setups-container',
                                        children=[_create_no_signals_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                            dbc.Tab(
                                label='Triggered Signals',  # Renamed from "Triggered"
                                tab_id='tab-triggered',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_green'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='signals-triggered-container',
                                        children=[_create_no_signals_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                            dbc.Tab(
                                label='Low Magnitude',
                                tab_id='tab-low-mag',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['accent_red'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='signals-lowmag-container',
                                        children=[_create_no_signals_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                            dbc.Tab(
                                label='Closed Trades',
                                tab_id='tab-closed',
                                label_style={'color': DARK_THEME['text_secondary']},
                                active_label_style={
                                    'color': DARK_THEME['text_muted'],
                                    'fontWeight': 'bold'
                                },
                                children=[
                                    html.Div(
                                        id='signals-closed-container',
                                        children=[_create_no_closed_trades_placeholder()],
                                        style={'marginTop': '1rem'}
                                    )
                                ]
                            ),
                        ], id='signals-tabs', active_tab='tab-setups', style={
                            'borderBottom': f'1px solid {DARK_THEME["border"]}'
                        }),
                    ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '1rem'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}'
                }, className='shadow')
            ], width=12)
        ], className='mb-4'),

        # ============================================
        # ROW 3: Trade Progress Display (Simplified - Session EQUITY-34)
        # ============================================
        dbc.Row([
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
                        html.Div(
                            id='trade-progress-container',
                            children=[_create_no_positions_progress_placeholder()]
                        )
                    ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '1rem'})
                ], style={
                    'backgroundColor': DARK_THEME['card_bg'],
                    'border': f'1px solid {DARK_THEME["border"]}'
                }, className='shadow')
            ], width=12)
        ], className='mb-4'),

        # Hidden storage for trade data
        dcc.Store(id='options-trades-store', data=[]),

        # Hidden container for backward compatibility (old signals container)
        html.Div(id='options-signals-container', style={'display': 'none'}),

        # Auto-refresh interval for live data (30 seconds)
        dcc.Interval(
            id='options-refresh-interval',
            interval=REFRESH_INTERVALS.get('live_positions', 30000),
            n_intervals=0,
            disabled=False
        ),

    ], fluid=True, style={'backgroundColor': DARK_THEME['background']})


# ============================================
# HELPER COMPONENTS
# ============================================

def _create_strat_signal_placeholder():
    """Create placeholder when no active signal."""
    return html.Div([
        html.I(className='fas fa-crosshairs fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No active signal', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Signals from the daemon will appear here',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem'
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
        html.Small('Run signal_daemon.py scan-all to detect patterns',
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


def _create_no_positions_placeholder():
    """Create placeholder when no positions are held."""
    return html.Div([
        html.I(className='fas fa-inbox fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No open positions', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Options positions from Alpaca will appear here',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def _create_no_positions_progress_placeholder():
    """Create placeholder when no positions for progress tracking - Session EQUITY-34."""
    return html.Div([
        html.I(className='fas fa-chart-line fa-2x mb-3',
               style={'color': DARK_THEME['text_muted']}),
        html.P('No active positions to track', style={
            'color': DARK_THEME['text_muted'],
            'marginBottom': '0.25rem'
        }),
        html.Small('Open positions will show progress toward targets',
                   style={'color': DARK_THEME['text_muted']})
    ], style={
        'textAlign': 'center',
        'padding': '2rem',
        'backgroundColor': DARK_THEME['card_bg']
    })


def create_trade_progress_display(trades: Optional[List[Dict]] = None) -> html.Div:
    """
    Create simplified trade progress display - Session EQUITY-34.

    Replaced complex Plotly bullet chart with clean HTML progress bars.
    Each trade shows: Name | Progress Bar | Percentage

    Args:
        trades: List of trade dicts with entry/current/target/stop/name/direction/pnl_pct

    Returns:
        HTML Div with progress display
    """
    if not trades:
        return _create_no_positions_progress_placeholder()

    rows = []
    for trade in trades:
        # Calculate progress percentage (0-100, capped)
        entry = trade.get('entry', 0)
        current = trade.get('current', 0)
        target = trade.get('target', 0)
        stop = trade.get('stop', 0)
        direction = trade.get('direction', 'CALL')
        pnl_pct = trade.get('pnl_pct', 0) * 100  # Convert to percentage

        # Calculate progress based on direction
        if direction == 'CALL':
            # CALL: profit when price goes up from entry to target
            total_range = target - entry if target != entry else 1
            progress = (current - entry) / total_range * 100
        else:
            # PUT: profit when price goes down from entry to target
            total_range = entry - target if entry != target else 1
            progress = (entry - current) / total_range * 100

        # Clamp progress to -100% to 200% for display
        progress = max(-100, min(200, progress))

        # Color based on profit
        is_profit = progress > 0
        bar_color = DARK_THEME['accent_green'] if is_profit else DARK_THEME['accent_red']
        pnl_color = DARK_THEME['accent_green'] if pnl_pct >= 0 else DARK_THEME['accent_red']

        # Create progress bar (clamped to 0-100 for display)
        display_progress = max(0, min(100, progress))

        rows.append(
            html.Div([
                # Trade name (left)
                html.Div([
                    html.Span(trade.get('name', 'Unknown'), style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary'],
                        'fontSize': '0.9rem'
                    }),
                ], style={'flex': '0 0 180px', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}),

                # Progress bar (middle)
                html.Div([
                    html.Div([
                        # Background bar
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'right': '0',
                            'bottom': '0',
                            'backgroundColor': DARK_THEME['progress_bg'],
                            'borderRadius': '4px'
                        }),
                        # Progress fill
                        html.Div(style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'bottom': '0',
                            'width': f'{display_progress}%',
                            'backgroundColor': bar_color,
                            'borderRadius': '4px',
                            'transition': 'width 0.3s ease'
                        }),
                    ], style={
                        'position': 'relative',
                        'height': '12px',
                        'borderRadius': '4px',
                        'overflow': 'hidden'
                    })
                ], style={'flex': '1', 'padding': '0 1rem'}),

                # Progress percentage (right)
                html.Div([
                    html.Span(f"{progress:.0f}%", style={
                        'fontWeight': 'bold',
                        'color': bar_color,
                        'fontSize': '0.9rem',
                        'minWidth': '50px',
                        'textAlign': 'right'
                    }),
                    html.Span(f" ({pnl_pct:+.1f}%)", style={
                        'color': pnl_color,
                        'fontSize': '0.8rem',
                        'marginLeft': '0.25rem'
                    }),
                ], style={'flex': '0 0 100px', 'textAlign': 'right'}),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'padding': '0.75rem 0',
                'borderBottom': f'1px solid {DARK_THEME["border"]}'
            })
        )

    return html.Div(rows)


def create_signals_table(signals: List[Dict], show_triggered_time: bool = False) -> html.Table:
    """
    Create table displaying STRAT signals.

    Args:
        signals: List of signal dictionaries from OptionsDataLoader
        show_triggered_time: If True, show triggered_at column instead of detected_time

    Returns:
        HTML table component
    """
    if not signals:
        return _create_no_signals_placeholder()

    rows = []
    for signal in signals:
        # Direction color
        dir_color = DARK_THEME['accent_green'] if signal.get('direction') == 'CALL' else DARK_THEME['accent_red']

        # Status badge color
        status = signal.get('status', 'UNKNOWN')
        if status in ('ALERTED', 'DETECTED'):
            status_color = 'info'
        elif status == 'TRIGGERED':
            status_color = 'success'
        elif status == 'HISTORICAL_TRIGGERED':
            status_color = 'secondary'
        else:
            status_color = 'warning'

        # Magnitude color - highlight low magnitude
        magnitude = signal.get('magnitude_pct', 0)
        mag_color = DARK_THEME['text_secondary'] if magnitude >= 0.5 else DARK_THEME['accent_red']

        # Time display - triggered_at or detected_time
        time_value = signal.get('triggered_at') if show_triggered_time else signal.get('detected_time')
        time_display = time_value if time_value else '-'

        rows.append(
            html.Tr([
                # Symbol + Timeframe
                html.Td([
                    html.Div(signal.get('symbol', ''), style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(signal.get('timeframe', ''), style={
                        'color': DARK_THEME['text_secondary']
                    })
                ], style={'padding': '0.75rem'}),

                # Pattern
                html.Td([
                    dbc.Badge(signal.get('pattern', ''), color='dark',
                              style={'fontSize': '0.85rem'})
                ], style={'padding': '0.75rem'}),

                # Direction
                html.Td([
                    html.Span(signal.get('direction', ''), style={
                        'color': dir_color,
                        'fontWeight': 'bold'
                    })
                ], style={'padding': '0.75rem'}),

                # Entry Trigger
                html.Td([
                    html.Div(f"${signal.get('entry_trigger', 0):.2f}", style={
                        'color': DARK_THEME['text_primary']
                    })
                ], style={'padding': '0.75rem'}),

                # Target / Stop
                html.Td([
                    html.Span(f"${signal.get('target', 0):.2f}",
                              style={'color': DARK_THEME['accent_green']}),
                    html.Span(' / ', style={'color': DARK_THEME['text_muted']}),
                    html.Span(f"${signal.get('stop', 0):.2f}",
                              style={'color': DARK_THEME['accent_red']})
                ], style={'padding': '0.75rem'}),

                # Magnitude
                html.Td([
                    f"{magnitude:.2f}%"
                ], style={
                    'padding': '0.75rem',
                    'color': mag_color,
                    'fontWeight': 'bold' if magnitude < 0.5 else 'normal'
                }),

                # R:R
                html.Td([
                    f"{signal.get('risk_reward', 0):.1f}"
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['accent_blue']
                }),

                # Time (Detected or Triggered)
                html.Td([
                    html.Small(time_display, style={'color': DARK_THEME['text_secondary']})
                ], style={'padding': '0.75rem'}),

                # Status
                html.Td([
                    dbc.Badge(status, color=status_color)
                ], style={'padding': '0.75rem'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    # Dynamic header based on time column type
    time_header = 'Triggered' if show_triggered_time else 'Detected'

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
                html.Th(time_header, style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Status', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
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
        trades: List of closed trade dictionaries from OptionsDataLoader

    Returns:
        HTML Div containing summary and table
    """
    if not trades:
        return _create_no_closed_trades_placeholder()

    # Calculate summary stats
    total_pnl = sum(t.get('realized_pnl', 0) for t in trades)
    winners = [t for t in trades if t.get('realized_pnl', 0) > 0]
    losers = [t for t in trades if t.get('realized_pnl', 0) <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    # Summary row
    summary = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span('Total P&L: ', style={'color': DARK_THEME['text_secondary']}),
                    html.Span(
                        f"${total_pnl:,.2f}",
                        style={
                            'color': DARK_THEME['accent_green'] if total_pnl >= 0 else DARK_THEME['accent_red'],
                            'fontWeight': 'bold',
                            'fontSize': '1.1rem'
                        }
                    ),
                ])
            ], width='auto'),
            dbc.Col([
                html.Div([
                    html.Span('Trades: ', style={'color': DARK_THEME['text_secondary']}),
                    html.Span(str(len(trades)), style={'color': DARK_THEME['text_primary']}),
                ])
            ], width='auto'),
            dbc.Col([
                html.Div([
                    html.Span('Win Rate: ', style={'color': DARK_THEME['text_secondary']}),
                    html.Span(
                        f"{win_rate:.1f}%",
                        style={
                            'color': DARK_THEME['accent_green'] if win_rate >= 50 else DARK_THEME['accent_red']
                        }
                    ),
                ])
            ], width='auto'),
            dbc.Col([
                html.Div([
                    html.Span(f"W: {len(winners)} ", style={'color': DARK_THEME['accent_green']}),
                    html.Span(' / ', style={'color': DARK_THEME['text_muted']}),
                    html.Span(f" L: {len(losers)}", style={'color': DARK_THEME['accent_red']}),
                ])
            ], width='auto'),
        ], className='mb-3', justify='start', style={
            'backgroundColor': DARK_THEME['card_header'],
            'padding': '0.75rem',
            'borderRadius': '4px'
        })
    ])

    # Build table rows
    rows = []
    for trade in trades:
        pnl = trade.get('realized_pnl', 0)
        roi = trade.get('roi_percent', 0)
        pnl_color = DARK_THEME['accent_green'] if pnl >= 0 else DARK_THEME['accent_red']

        # Pattern info from signal correlation (via OSI symbol lookup)
        pattern = trade.get('pattern', '-')
        timeframe = trade.get('timeframe', '-')

        rows.append(
            html.Tr([
                # Contract
                html.Td([
                    html.Div(trade.get('display_contract', trade.get('symbol', '')), style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    })
                ], style={'padding': '0.75rem'}),

                # Pattern + Timeframe
                html.Td([
                    html.Div(pattern, style={
                        'color': DARK_THEME['accent_blue'] if pattern != '-' else DARK_THEME['text_muted']
                    }),
                    html.Small(timeframe, style={
                        'color': DARK_THEME['text_secondary']
                    }) if timeframe != '-' else None
                ], style={'padding': '0.75rem'}),

                # Qty
                html.Td([
                    str(trade.get('qty', 0))
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['text_primary']
                }),

                # Entry (price + time)
                html.Td([
                    html.Div(f"${trade.get('buy_price', 0):.2f}", style={
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(trade.get('buy_time_display', '-'), style={
                        'color': DARK_THEME['text_muted']
                    })
                ], style={'padding': '0.75rem'}),

                # Exit (price + time)
                html.Td([
                    html.Div(f"${trade.get('sell_price', 0):.2f}", style={
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(trade.get('sell_time_display', '-'), style={
                        'color': DARK_THEME['text_muted']
                    })
                ], style={'padding': '0.75rem'}),

                # Realized P&L
                html.Td([
                    html.Div([
                        html.Span(f"${pnl:+,.2f}", style={
                            'fontWeight': 'bold',
                            'color': pnl_color
                        }),
                        html.Small(f" ({roi:+.1f}%)", style={
                            'color': pnl_color,
                            'marginLeft': '0.25rem'
                        })
                    ])
                ], style={'padding': '0.75rem'}),

                # Duration
                html.Td([
                    trade.get('duration', '-')
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['text_secondary']
                }),

                # Pattern (if available)
                html.Td([
                    dbc.Badge(pattern, color='dark', style={'fontSize': '0.8rem'}) if pattern else html.Small('-', style={'color': DARK_THEME['text_muted']})
                ], style={'padding': '0.75rem'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    table = html.Table([
        html.Thead([
            html.Tr([
                html.Th('Contract', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Pattern', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Qty', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Entry', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Exit', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Realized P&L', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Duration', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Pattern', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
            ], style={'backgroundColor': DARK_THEME['card_header']})
        ]),
        html.Tbody(rows)
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': DARK_THEME['card_bg']
    })

    return html.Div([summary, table])


def create_positions_table(positions: List[Dict]) -> html.Table:
    """
    Create table displaying live option positions.

    Session EQUITY-34: Added pattern + timeframe column.

    Args:
        positions: List of position dictionaries from Alpaca with pattern/timeframe

    Returns:
        HTML table component
    """
    if not positions:
        return _create_no_positions_placeholder()

    rows = []
    for pos in positions:
        # P&L color
        pnl = pos.get('unrealized_pl', 0)
        pnl_pct = pos.get('unrealized_plpc', 0)
        pnl_color = DARK_THEME['accent_green'] if pnl >= 0 else DARK_THEME['accent_red']

        # Get display contract or parse OCC symbol
        contract = pos.get('display_contract', pos.get('symbol', ''))

        # Session EQUITY-34: Get pattern and timeframe from signal linkage
        pattern = pos.get('pattern', '-')
        timeframe = pos.get('timeframe', '-')

        rows.append(
            html.Tr([
                # Contract
                html.Td([
                    html.Div(contract, style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['text_primary']
                    }),
                    html.Small(f"Qty: {pos.get('qty', 0)}", style={
                        'color': DARK_THEME['text_secondary']
                    })
                ], style={'padding': '0.75rem'}),

                # Pattern + Timeframe (Session EQUITY-34)
                html.Td([
                    html.Div(pattern, style={
                        'fontWeight': 'bold',
                        'color': DARK_THEME['accent_blue'] if pattern != '-' else DARK_THEME['text_muted']
                    }),
                    html.Small(timeframe, style={
                        'color': DARK_THEME['text_secondary']
                    }) if timeframe != '-' else None
                ], style={'padding': '0.75rem'}),

                # Entry Price
                html.Td([
                    f"${pos.get('avg_entry_price', 0):.2f}"
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['text_primary']
                }),

                # Current Price
                html.Td([
                    f"${pos.get('current_price', 0):.2f}"
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['text_primary']
                }),

                # Market Value
                html.Td([
                    f"${pos.get('market_value', 0):,.2f}"
                ], style={
                    'padding': '0.75rem',
                    'color': DARK_THEME['text_secondary']
                }),

                # P&L
                html.Td([
                    html.Div(f"${pnl:+,.2f}", style={
                        'color': pnl_color,
                        'fontWeight': 'bold'
                    }),
                    html.Small(f"({pnl_pct:+.1f}%)", style={'color': pnl_color})
                ], style={'padding': '0.75rem', 'textAlign': 'right'}),

                # Actions
                html.Td([
                    dbc.Button(
                        html.I(className='fas fa-times'),
                        color='danger',
                        size='sm',
                        outline=True,
                        title='Close Position',
                        id={'type': 'close-position-btn', 'index': pos.get('symbol', '')}
                    )
                ], style={'padding': '0.75rem', 'textAlign': 'center'}),

            ], style={'borderBottom': f'1px solid {DARK_THEME["border"]}'})
        )

    return html.Table([
        html.Thead([
            html.Tr([
                html.Th('Contract', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Pattern', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),  # Session EQUITY-34
                html.Th('Entry', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Current', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
                html.Th('Value', style={'color': DARK_THEME['text_secondary'], 'padding': '0.75rem'}),
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
    """Create placeholder P&L summary cards - will be updated by callback."""
    return html.Div(id='pnl-summary-content', children=[
        html.Div([
            html.I(className='fas fa-chart-pie fa-2x mb-3',
                   style={'color': DARK_THEME['text_muted']}),
            html.P('Loading P&L data...', style={
                'color': DARK_THEME['text_muted'],
                'marginBottom': '0'
            })
        ], style={'textAlign': 'center', 'padding': '2rem'})
    ])


def create_pnl_summary(positions: List[Dict]) -> html.Div:
    """
    Create P&L summary cards from live position data.

    Args:
        positions: List of position dictionaries from Alpaca

    Returns:
        HTML div with P&L summary
    """
    if not positions:
        return html.Div([
            html.I(className='fas fa-inbox fa-2x mb-3',
                   style={'color': DARK_THEME['text_muted']}),
            html.P('No positions', style={
                'color': DARK_THEME['text_muted'],
                'marginBottom': '0.25rem'
            }),
            html.Small('P&L summary will appear when positions are held',
                       style={'color': DARK_THEME['text_muted']})
        ], style={'textAlign': 'center', 'padding': '2rem'})

    # Calculate totals from positions
    total_pnl = sum(p.get('unrealized_pl', 0) for p in positions)
    total_cost = sum(p.get('cost_basis', 0) for p in positions)
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    winning = sum(1 for p in positions if p.get('unrealized_pl', 0) >= 0)
    losing = len(positions) - winning

    pnls = [p.get('unrealized_pl', 0) for p in positions]
    largest_win = max(pnls) if pnls else 0
    largest_loss = min(pnls) if pnls else 0

    pnl_color = DARK_THEME['accent_green'] if total_pnl >= 0 else DARK_THEME['accent_red']

    return html.Div([
        # Total P&L
        html.Div([
            html.P('Total P&L', style={
                'color': DARK_THEME['text_secondary'],
                'marginBottom': '0.25rem',
                'fontSize': '0.9rem'
            }),
            html.H3(f'${total_pnl:,.2f}', style={
                'color': pnl_color,
                'marginBottom': '0',
                'fontWeight': 'bold'
            }),
            html.Small(f'{total_pnl_pct:+.1f}%', style={'color': pnl_color})
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
                    html.H5(str(winning), style={
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
                    html.H5(str(losing), style={
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
                html.Span(f'${largest_win:,.2f}', style={
                    'color': DARK_THEME['accent_green'],
                    'fontWeight': 'bold'
                }),
            ], style={'marginBottom': '0.5rem'}),
            html.Div([
                html.Span('Largest Loss: ', style={'color': DARK_THEME['text_secondary']}),
                html.Span(f'${largest_loss:,.2f}', style={
                    'color': DARK_THEME['accent_red'],
                    'fontWeight': 'bold'
                }),
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
    - Entry -> Current -> Target visualization
    - Stop loss indicator
    - Color-coded by profit/loss status

    Args:
        trades: List of trade dictionaries with entry/current/target/stop

    Returns:
        Plotly figure with trade progress visualization
    """

    # Return empty chart if no trades
    if not trades:
        fig = go.Figure()
        fig.update_layout(
            height=250,
            template='plotly_dark',
            paper_bgcolor=DARK_THEME['card_bg'],
            plot_bgcolor=DARK_THEME['card_bg'],
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[{
                'text': 'No active positions to track',
                'x': 0.5,
                'y': 0.5,
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'color': DARK_THEME['text_muted'], 'size': 14}
            }],
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig

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
    'create_signals_table',
    'create_positions_table',
    'create_pnl_summary',
    'DARK_THEME'
]
