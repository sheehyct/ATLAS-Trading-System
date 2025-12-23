"""
ATLAS Trading Dashboard - Main Application

This is the main Dash application for the ATLAS (Adaptive Trading with Layered Asset System)
algorithmic trading dashboard. It provides real-time monitoring, historical analysis, and
risk management visualization for the multi-layer trading architecture.

Features:
- Regime Detection (Layer 1) visualization with HMM regime shading
- Strategy Performance analysis with VectorBT Pro integration
- Live Portfolio monitoring via Alpaca API
- Risk Management dashboard with portfolio heat gauges
- Mobile-friendly responsive design

Technologies:
- Plotly Dash for interactive web application
- dash-bootstrap-components for professional UI
- lightweight-charts-python for TradingView-quality charts
- VectorBT Pro for backtesting data
- Alpaca API for live trading data

Usage:
    python dashboard/app.py
    # or
    uv run python dashboard/app.py

Access at: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import configuration
from dashboard.config import (
    DASHBOARD_CONFIG,
    COLORS,
    REGIME_COLORS,
    REFRESH_INTERVALS,
    AVAILABLE_STRATEGIES,
)

# Import components
from dashboard.components.header import create_header
from dashboard.components.regime_panel import create_regime_panel
from dashboard.components.strategy_panel import create_strategy_panel
from dashboard.components.portfolio_panel import create_portfolio_panel
from dashboard.components.risk_panel import create_risk_panel
from dashboard.components.options_panel import (
    create_options_panel,
    create_signals_table,
    create_positions_table,
    create_pnl_summary,
    create_trade_progress_chart,
    create_trade_progress_display,
    create_closed_trades_table,
)
from dashboard.components.crypto_panel import (
    create_crypto_panel,
    create_account_summary_display,
    create_daemon_status_display,
    create_positions_table as create_crypto_positions_table,
    create_signals_table as create_crypto_signals_table,
    create_closed_trades_table as create_crypto_closed_table,
    create_performance_metrics,
)

# Import data loaders
from dashboard.data_loaders.regime_loader import RegimeDataLoader
from dashboard.data_loaders.backtest_loader import BacktestDataLoader
from dashboard.data_loaders.live_loader import LiveDataLoader
from dashboard.data_loaders.orders_loader import OrdersDataLoader
from dashboard.data_loaders.options_loader import OptionsDataLoader
from dashboard.data_loaders.crypto_loader import CryptoDataLoader

# Import visualizations
from dashboard.visualizations.regime_viz import (
    create_regime_timeline,
    create_feature_dashboard,
    create_regime_statistics_table,
)
from dashboard.visualizations.performance_viz import (
    create_equity_curve,
    create_rolling_metrics,
)
from dashboard.visualizations.trade_viz import (
    create_trade_distribution,
    create_trade_timeline,
)
from dashboard.visualizations.risk_viz import (
    create_portfolio_heat_gauge,
    create_risk_metrics_table,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZE DASH APP
# ============================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        # IBM Plex - Professional, corporate typography (Google Fonts)
        'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap',
        'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='ATLAS Trading Dashboard',
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        },
        {
            'name': 'theme-color',
            'content': '#000000'
        }
    ],
)

# ============================================
# INITIALIZE DATA LOADERS
# ============================================

# Log environment info for Railway debugging
import os
logger.info("=" * 60)
logger.info("ATLAS Dashboard Starting - Environment Check")
logger.info("=" * 60)
logger.info(f"RAILWAY_ENVIRONMENT: {os.getenv('RAILWAY_ENVIRONMENT', 'not set')}")
logger.info(f"DEFAULT_ACCOUNT: {os.getenv('DEFAULT_ACCOUNT', 'MID (default)')}")
logger.info(f"ALPACA_API_KEY present: {bool(os.getenv('ALPACA_API_KEY'))}")
logger.info(f"ALPACA_SECRET_KEY present: {bool(os.getenv('ALPACA_SECRET_KEY'))}")
logger.info(f"ALPACA_MID_KEY present: {bool(os.getenv('ALPACA_MID_KEY'))}")
logger.info(f"ALPACA_LARGE_KEY present: {bool(os.getenv('ALPACA_LARGE_KEY'))}")
logger.info("=" * 60)

try:
    regime_loader = RegimeDataLoader()
    logger.info("RegimeDataLoader initialized successfully")
except Exception as e:
    logger.warning(f"RegimeDataLoader initialization failed: {e}")
    regime_loader = None

try:
    backtest_loader = BacktestDataLoader()
    logger.info("BacktestDataLoader initialized successfully")
except Exception as e:
    logger.warning(f"BacktestDataLoader initialization failed: {e}")
    backtest_loader = None

try:
    live_loader = LiveDataLoader()
    if live_loader.client is not None:
        logger.info("LiveDataLoader initialized successfully with active Alpaca connection")
    else:
        logger.warning(f"LiveDataLoader initialized but Alpaca client is None. Error: {live_loader.init_error}")
except Exception as e:
    logger.warning(f"LiveDataLoader initialization failed: {e}")
    live_loader = None

try:
    orders_loader = OrdersDataLoader()
    logger.info("OrdersDataLoader initialized successfully")
except Exception as e:
    logger.warning(f"OrdersDataLoader initialization failed: {e}")
    orders_loader = None

try:
    options_loader = OptionsDataLoader(account='SMALL')
    if options_loader._connected:
        logger.info("OptionsDataLoader initialized successfully with Alpaca connection")
    else:
        logger.warning(f"OptionsDataLoader initialized but not connected: {options_loader.init_error}")
except Exception as e:
    logger.warning(f"OptionsDataLoader initialization failed: {e}")
    options_loader = None

try:
    crypto_loader = CryptoDataLoader()
    if crypto_loader._connected:
        logger.info("CryptoDataLoader initialized successfully with VPS connection")
    else:
        logger.warning(f"CryptoDataLoader initialized but not connected: {crypto_loader.init_error}")
except Exception as e:
    logger.warning(f"CryptoDataLoader initialization failed: {e}")
    crypto_loader = None

# ============================================
# APP LAYOUT
# ============================================

app.layout = dbc.Container([

    # Header with branding and status
    create_header(),

    # Control Panel Row
    dbc.Row([
        # Date Range Selector
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label('Date Range:', className='fw-bold mb-2'),
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        start_date=(datetime.now() - timedelta(days=365)).date(),
                        end_date=datetime.now().date(),
                        display_format='YYYY-MM-DD',
                        style={'width': '100%'}
                    )
                ])
            ], className='shadow-sm')
        ], width=12, lg=6, className='mb-3'),

        # Auto-refresh Control
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label('Auto-Refresh:', className='fw-bold mb-2'),
                    dbc.Switch(
                        id='auto-refresh-toggle',
                        label='Enable Live Updates',
                        value=True,
                        className='mt-2'
                    )
                ])
            ], className='shadow-sm')
        ], width=12, lg=6, className='mb-3'),
    ], className='mb-4'),

    # Live Update Interval Component (Hidden)
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVALS['live_positions'],  # 30 seconds
        n_intervals=0,
        disabled=False  # Enabled by default
    ),

    # Tab Navigation
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs(
                id='tabs',
                active_tab='regime-tab',
                children=[
                    # Tab 1: Regime Detection (Layer 1)
                    dbc.Tab(
                        label='Regime Detection',
                        tab_id='regime-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': COLORS['bull_primary'],
                            'font-weight': 'bold'
                        }
                    ),

                    # Tab 2: Strategy Performance
                    dbc.Tab(
                        label='Strategy Performance',
                        tab_id='strategy-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': COLORS['bull_primary'],
                            'font-weight': 'bold'
                        }
                    ),

                    # Tab 3: Live Portfolio
                    dbc.Tab(
                        label='Live Portfolio',
                        tab_id='portfolio-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': COLORS['bull_primary'],
                            'font-weight': 'bold'
                        }
                    ),

                    # Tab 4: Risk Management
                    dbc.Tab(
                        label='Risk Management',
                        tab_id='risk-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': COLORS['bull_primary'],
                            'font-weight': 'bold'
                        }
                    ),

                    # Tab 5: Options Trading (STRAT Integration)
                    dbc.Tab(
                        label='Options Trading',
                        tab_id='options-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': '#ffd700',  # Gold for options
                            'font-weight': 'bold'
                        }
                    ),

                    # Tab 6: Crypto Trading (Session CRYPTO-6)
                    dbc.Tab(
                        label='Crypto Trading',
                        tab_id='crypto-tab',
                        label_style={'color': COLORS['text_primary']},
                        active_label_style={
                            'color': '#22D3EE',  # Cyan for crypto
                            'font-weight': 'bold'
                        }
                    ),
                ],
            ),
        ], className='p-0')
    ], className='shadow-sm mb-4'),

    # Tab Content Container
    html.Div(id='tab-content', className='mt-4'),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                'ATLAS Trading Dashboard v1.0 | ',
                html.A('Documentation', href='#', className='text-decoration-none'),
                ' | ',
                html.A('GitHub', href='#', className='text-decoration-none'),
            ], className='text-center text-muted small')
        ], width=12)
    ], className='mt-5'),

], fluid=True, className='p-4', style={'backgroundColor': COLORS['background_dark']})


# ============================================
# CALLBACKS
# ============================================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(active_tab):
    """
    Render content based on selected tab.

    Args:
        active_tab: ID of the active tab

    Returns:
        Component tree for the selected tab
    """
    if active_tab == 'regime-tab':
        return create_regime_panel()
    elif active_tab == 'strategy-tab':
        return create_strategy_panel()
    elif active_tab == 'portfolio-tab':
        return create_portfolio_panel()
    elif active_tab == 'risk-tab':
        return create_risk_panel()
    elif active_tab == 'options-tab':
        return create_options_panel()
    elif active_tab == 'crypto-tab':
        return create_crypto_panel()
    else:
        return html.Div('Tab content not found')


@app.callback(
    Output('interval-component', 'disabled'),
    Input('auto-refresh-toggle', 'value')
)
def toggle_auto_refresh(enabled):
    """
    Enable/disable auto-refresh based on toggle switch.

    Args:
        enabled: Boolean indicating if auto-refresh is enabled

    Returns:
        Boolean to disable/enable interval component
    """
    return not enabled


@app.callback(
    Output('regime-timeline-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_regime_timeline(start_date, end_date):
    """
    Update regime detection timeline visualization.

    Args:
        start_date: Start date for data range
        end_date: End date for data range

    Returns:
        Plotly figure with regime timeline
    """
    try:
        logger.info(f"update_regime_timeline called: start={start_date}, end={end_date}")

        if regime_loader is None:
            logger.warning("regime_loader is None")
            return create_error_figure("Regime data loader not available")

        if regime_loader.atlas_model is None:
            logger.warning("ATLAS model not initialized")
            return create_error_figure("ATLAS regime model not available")

        # Load regime data (this can take 10-15 seconds)
        logger.info("Loading regime timeline data...")
        regime_data = regime_loader.get_regime_timeline(start_date, end_date)

        if regime_data.empty:
            logger.warning("Regime data returned empty")
            return create_error_figure("No regime data available for selected range")

        logger.info(f"Regime data loaded: {len(regime_data)} rows")

        # Extract data
        dates = regime_data['date']
        regimes = regime_data['regime']
        prices = regime_data.get('price', pd.Series(dtype=float))

        return create_regime_timeline(dates, regimes, prices)

    except Exception as e:
        logger.error(f"Error updating regime timeline: {e}", exc_info=True)
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    Output('feature-dashboard-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_feature_dashboard(start_date, end_date):
    """
    Update regime feature visualization.

    Args:
        start_date: Start date for data range
        end_date: End date for data range

    Returns:
        Plotly figure with feature evolution
    """
    try:
        if regime_loader is None:
            return create_error_figure("Regime data loader not available")

        features_data = regime_loader.get_regime_features(start_date, end_date)

        if features_data.empty:
            return create_error_figure("No feature data available")

        return create_feature_dashboard(
            dates=features_data['date'],
            downside_dev=features_data.get('downside_dev', pd.Series(dtype=float)),
            sortino_20d=features_data.get('sortino_20d', pd.Series(dtype=float)),
            sortino_60d=features_data.get('sortino_60d', pd.Series(dtype=float))
        )

    except Exception as e:
        logger.error(f"Error updating feature dashboard: {e}")
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    Output('regime-stats-table', 'children'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_regime_statistics(start_date, end_date):
    """
    Update regime statistics panel with VIX data and regime metrics.

    Shows:
    - Current VIX level and changes (intraday, 1-day, 3-day)
    - Current regime and allocation
    - Regime duration statistics
    """
    try:
        if regime_loader is None:
            return html.P("Regime loader not available", className='text-muted')

        # Get VIX status
        vix_status = regime_loader.get_vix_status()

        # Get current regime
        current_regime = regime_loader.get_current_regime()

        # Get regime timeline for statistics
        regime_data = regime_loader.get_regime_timeline(start_date, end_date)

        # Calculate regime statistics
        regime_counts = {}
        if not regime_data.empty:
            regime_counts = regime_data['regime'].value_counts().to_dict()
            total_days = len(regime_data)
        else:
            total_days = 0

        # Build statistics cards
        cards = []

        # VIX Status Card
        vix_current = vix_status.get('vix_current', 0) or 0
        intraday_change = vix_status.get('intraday_change_pct', 0) or 0
        one_day_change = vix_status.get('one_day_change_pct', 0) or 0
        three_day_change = vix_status.get('three_day_change_pct', 0) or 0
        is_crash = vix_status.get('is_crash', False)

        # VIX color based on level
        if vix_current >= 35:
            vix_color = COLORS['danger']
        elif vix_current >= 25:
            vix_color = COLORS['warning']
        else:
            vix_color = COLORS['bull_primary']

        cards.append(
            dbc.Card([
                dbc.CardHeader("VIX Status", className='fw-bold small'),
                dbc.CardBody([
                    html.H4(f"{vix_current:.1f}", style={'color': vix_color}),
                    html.Div([
                        html.Span("Intraday: ", className='text-muted small'),
                        html.Span(
                            f"{intraday_change:+.1f}%",
                            className='small',
                            style={'color': COLORS['danger'] if intraday_change > 0 else COLORS['bull_primary']}
                        )
                    ]),
                    html.Div([
                        html.Span("1-Day: ", className='text-muted small'),
                        html.Span(
                            f"{one_day_change:+.1f}%",
                            className='small',
                            style={'color': COLORS['danger'] if one_day_change > 0 else COLORS['bull_primary']}
                        )
                    ]),
                    html.Div([
                        html.Span("3-Day: ", className='text-muted small'),
                        html.Span(
                            f"{three_day_change:+.1f}%",
                            className='small',
                            style={'color': COLORS['danger'] if three_day_change > 0 else COLORS['bull_primary']}
                        )
                    ]),
                    html.Hr(className='my-2'),
                    html.Div([
                        html.I(className=f'fas fa-{"exclamation-triangle text-danger" if is_crash else "check-circle text-success"} me-1'),
                        html.Span("CRASH" if is_crash else "Normal", className='small fw-bold')
                    ])
                ], className='p-2')
            ], className='mb-2', style={'backgroundColor': COLORS['background_medium']})
        )

        # Current Regime Card
        regime = current_regime.get('regime', 'UNKNOWN')
        allocation = current_regime.get('allocation_pct', 0)
        regime_colors_map = {
            'TREND_BULL': COLORS['bull_primary'],
            'TREND_NEUTRAL': COLORS['text_secondary'],
            'TREND_BEAR': COLORS['warning'],
            'CRASH': COLORS['danger'],
            'UNKNOWN': COLORS['text_secondary']
        }

        cards.append(
            dbc.Card([
                dbc.CardHeader("Current Regime", className='fw-bold small'),
                dbc.CardBody([
                    html.H5(regime.replace('TREND_', ''), style={'color': regime_colors_map.get(regime, COLORS['text_secondary'])}),
                    html.Div([
                        html.Span("Allocation: ", className='text-muted small'),
                        html.Span(f"{allocation}%", className='small fw-bold')
                    ]),
                    html.Div([
                        html.Span("As of: ", className='text-muted small'),
                        html.Span(current_regime.get('as_of_date', 'N/A'), className='small')
                    ])
                ], className='p-2')
            ], className='mb-2', style={'backgroundColor': COLORS['background_medium']})
        )

        # Regime Distribution Card
        if total_days > 0:
            dist_items = []
            for regime_name in ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']:
                count = regime_counts.get(regime_name, 0)
                pct = (count / total_days) * 100
                color = regime_colors_map.get(regime_name, COLORS['text_secondary'])
                display_name = regime_name.replace('TREND_', '')
                dist_items.append(
                    html.Div([
                        html.Span(f"{display_name}: ", className='text-muted small'),
                        html.Span(f"{pct:.0f}%", className='small', style={'color': color}),
                        html.Span(f" ({count}d)", className='text-muted small')
                    ])
                )

            cards.append(
                dbc.Card([
                    dbc.CardHeader(f"Distribution ({total_days}d)", className='fw-bold small'),
                    dbc.CardBody(dist_items, className='p-2')
                ], className='mb-2', style={'backgroundColor': COLORS['background_medium']})
            )

        return html.Div(cards)

    except Exception as e:
        logger.error(f"Error updating regime statistics: {e}", exc_info=True)
        return html.P(f"Error: {str(e)}", className='text-danger small')


@app.callback(
    Output('equity-curve-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('interval-component', 'n_intervals')]
)
def update_equity_curve(strategy_name, start_date, end_date, n):
    """
    Update strategy equity curve visualization.

    For STRAT Options: Shows realized P&L summary from closed trades.
    For other strategies: Shows current portfolio P&L from Alpaca.
    """
    import plotly.graph_objects as go

    try:
        # Handle STRAT Options strategy - show closed trades performance
        if strategy_name == 'strat_options':
            if options_loader is None:
                return create_info_figure("Options Loader Not Available", "Cannot fetch closed trades")

            summary = options_loader.get_closed_trades_summary(days=30)

            fig = go.Figure()

            # Title
            fig.add_annotation(
                text=f"<b>STRAT Options Performance (30 Days)</b>",
                xref='paper', yref='paper',
                x=0.5, y=0.95,
                showarrow=False,
                font=dict(size=20, color=COLORS['text_primary'])
            )

            # Total Realized P&L
            total_pnl = summary.get('total_pnl', 0)
            pnl_color = COLORS['bull_primary'] if total_pnl >= 0 else COLORS['danger']
            fig.add_annotation(
                text=f"<span style='font-size:36px;color:{pnl_color}'>${total_pnl:+,.2f}</span>",
                xref='paper', yref='paper',
                x=0.5, y=0.75,
                showarrow=False,
                font=dict(size=24, color=COLORS['text_primary'])
            )

            # Win Rate
            win_rate = summary.get('win_rate', 0)
            win_color = COLORS['bull_primary'] if win_rate >= 50 else COLORS['danger']
            fig.add_annotation(
                text=f"Win Rate: <span style='color:{win_color}'>{win_rate:.1f}%</span>",
                xref='paper', yref='paper',
                x=0.5, y=0.55,
                showarrow=False,
                font=dict(size=16, color=COLORS['text_secondary'])
            )

            # Trade counts
            fig.add_annotation(
                text=f"Trades: {summary.get('total_trades', 0)} | Wins: {summary.get('win_count', 0)} | Losses: {summary.get('loss_count', 0)}",
                xref='paper', yref='paper',
                x=0.5, y=0.40,
                showarrow=False,
                font=dict(size=14, color=COLORS['text_secondary'])
            )

            # Average P&L
            avg_pnl = summary.get('avg_pnl', 0)
            avg_color = COLORS['bull_primary'] if avg_pnl >= 0 else COLORS['danger']
            fig.add_annotation(
                text=f"Avg P&L: <span style='color:{avg_color}'>${avg_pnl:+,.2f}</span>",
                xref='paper', yref='paper',
                x=0.5, y=0.25,
                showarrow=False,
                font=dict(size=14, color=COLORS['text_secondary'])
            )

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor=COLORS['background_dark'],
                plot_bgcolor=COLORS['background_dark'],
                xaxis={'visible': False},
                yaxis={'visible': False},
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            return fig

        # Default: Show portfolio status for other strategies
        if live_loader is None or live_loader.client is None:
            return create_info_figure(
                "Live Data Not Available",
                f"Alpaca connection not established.\n{live_loader.init_error if live_loader else 'Loader not initialized'}"
            )

        # Get account and positions data
        account = live_loader.get_account_status()
        positions = live_loader.get_current_positions()

        if not account:
            return create_info_figure("No Account Data", "Unable to fetch account status from Alpaca")

        # Extract key metrics
        equity = account.get('equity', 0)
        last_equity = account.get('last_equity', equity)
        cash = account.get('cash', 0)
        portfolio_value = account.get('portfolio_value', equity)

        # Calculate daily P&L
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        # Create summary figure with portfolio metrics
        fig = go.Figure()

        # Add portfolio summary as annotations
        fig.add_annotation(
            text=f"<b>Paper Trading Portfolio</b>",
            xref='paper', yref='paper',
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=20, color=COLORS['text_primary'])
        )

        fig.add_annotation(
            text=f"<span style='font-size:36px;color:{COLORS['bull_primary'] if daily_pnl >= 0 else COLORS['danger']}'>${equity:,.2f}</span>",
            xref='paper', yref='paper',
            x=0.5, y=0.75,
            showarrow=False,
            font=dict(size=24, color=COLORS['text_primary'])
        )

        pnl_color = COLORS['bull_primary'] if daily_pnl >= 0 else COLORS['danger']
        fig.add_annotation(
            text=f"Today's P&L: <span style='color:{pnl_color}'>${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)</span>",
            xref='paper', yref='paper',
            x=0.5, y=0.55,
            showarrow=False,
            font=dict(size=16, color=COLORS['text_secondary'])
        )

        fig.add_annotation(
            text=f"Cash: ${cash:,.2f} | Positions: {len(positions) if not positions.empty else 0}",
            xref='paper', yref='paper',
            x=0.5, y=0.40,
            showarrow=False,
            font=dict(size=14, color=COLORS['text_secondary'])
        )

        # Add positions summary if available
        if not positions.empty:
            total_unrealized = positions['unrealized_pl'].sum() if 'unrealized_pl' in positions.columns else 0
            unrealized_color = COLORS['bull_primary'] if total_unrealized >= 0 else COLORS['danger']
            fig.add_annotation(
                text=f"Unrealized P&L: <span style='color:{unrealized_color}'>${total_unrealized:+,.2f}</span>",
                xref='paper', yref='paper',
                x=0.5, y=0.25,
                showarrow=False,
                font=dict(size=14, color=COLORS['text_secondary'])
            )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=COLORS['background_dark'],
            plot_bgcolor=COLORS['background_dark'],
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating equity curve: {e}", exc_info=True)
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    Output('rolling-metrics-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_rolling_metrics(strategy_name, n):
    """Show P&L breakdown - closed trades for STRAT Options, positions for others."""
    import plotly.graph_objects as go

    try:
        # Handle STRAT Options - show closed trades P&L breakdown
        if strategy_name == 'strat_options':
            if options_loader is None:
                return create_info_figure("Options Loader Not Available", "Cannot fetch closed trades")

            closed_trades = options_loader.get_closed_trades(days=30)

            if not closed_trades:
                return create_info_figure("No Closed Trades", "No trades in the last 30 days")

            # Create bar chart of trade P&L (limit to last 10)
            trades_to_show = closed_trades[:10]
            contracts = [t.get('display_contract', t.get('symbol', '')[:10]) for t in trades_to_show]
            pnls = [t.get('realized_pnl', 0) for t in trades_to_show]
            rois = [t.get('roi_percent', 0) for t in trades_to_show]

            colors = [COLORS['bull_primary'] if pl >= 0 else COLORS['danger'] for pl in pnls]

            fig = go.Figure(data=[
                go.Bar(
                    x=contracts,
                    y=pnls,
                    marker_color=colors,
                    text=[f"${pl:+,.0f}" for pl in pnls],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white', size=10),
                    hovertemplate='<b>%{x}</b><br>P&L: $%{y:+,.2f}<br>ROI: %{customdata:+.1f}%<extra></extra>',
                    customdata=rois
                )
            ])

            # Calculate y-axis range with padding
            max_val = max(pnls) if pnls else 0
            min_val = min(pnls) if pnls else 0
            y_padding = max(abs(max_val), abs(min_val), 10) * 0.2

            fig.update_layout(
                title={'text': 'Closed Trades P&L (Last 10)', 'font': {'color': COLORS['text_primary']}},
                template='plotly_dark',
                paper_bgcolor=COLORS['background_dark'],
                plot_bgcolor=COLORS['background_dark'],
                font={'color': COLORS['text_primary']},
                xaxis={'title': 'Contract', 'gridcolor': COLORS['grid'], 'tickangle': -45},
                yaxis={
                    'title': 'Realized P&L ($)',
                    'gridcolor': COLORS['grid'],
                    'range': [min_val - y_padding, max_val + y_padding]
                },
                height=350,
                margin=dict(t=50, b=100, l=60, r=20),
                showlegend=False
            )

            return fig

        # Default: Show current positions P&L
        if live_loader is None or live_loader.client is None:
            return create_info_figure("Live Data Not Available", "Alpaca connection not established")

        positions = live_loader.get_current_positions()

        if positions.empty:
            return create_info_figure("No Open Positions", "No positions to display")

        # Create bar chart of position P&L
        symbols = positions['symbol'].tolist()
        unrealized_pls = positions['unrealized_pl'].tolist() if 'unrealized_pl' in positions.columns else [0] * len(symbols)
        market_values = positions['market_value'].tolist() if 'market_value' in positions.columns else [0] * len(symbols)

        colors = [COLORS['bull_primary'] if pl >= 0 else COLORS['danger'] for pl in unrealized_pls]

        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=unrealized_pls,
                marker_color=colors,
                text=[f"${pl:+,.2f}" for pl in unrealized_pls],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='white', size=10),
                hovertemplate='<b>%{x}</b><br>P&L: $%{y:+,.2f}<br>Value: $%{customdata:,.2f}<extra></extra>',
                customdata=market_values
            )
        ])

        # Calculate y-axis range with padding
        max_val = max(unrealized_pls) if unrealized_pls else 0
        min_val = min(unrealized_pls) if unrealized_pls else 0
        y_padding = max(abs(max_val), abs(min_val), 10) * 0.2

        fig.update_layout(
            title={'text': 'Position P&L ($)', 'font': {'color': COLORS['text_primary']}},
            template='plotly_dark',
            paper_bgcolor=COLORS['background_dark'],
            plot_bgcolor=COLORS['background_dark'],
            font={'color': COLORS['text_primary']},
            xaxis={'title': 'Symbol', 'gridcolor': COLORS['grid']},
            yaxis={
                'title': 'Unrealized P&L ($)',
                'gridcolor': COLORS['grid'],
                'range': [min_val - y_padding, max_val + y_padding]
            },
            height=350,
            margin=dict(t=50, b=50, l=60, r=20),
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating rolling metrics: {e}")
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    Output('regime-comparison-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('interval-component', 'n_intervals'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_regime_comparison(strategy_name, n, start_date, end_date):
    """Show current regime status with P&L context."""
    import plotly.graph_objects as go

    try:
        # Get current regime
        current_regime_data = None
        if regime_loader is not None:
            current_regime_data = regime_loader.get_current_regime()

        # Get positions P&L
        total_pnl = 0
        if live_loader is not None and live_loader.client is not None:
            positions = live_loader.get_current_positions()
            if not positions.empty and 'unrealized_pl' in positions.columns:
                total_pnl = positions['unrealized_pl'].sum()

        # Determine regime info
        regime = current_regime_data.get('regime', 'UNKNOWN') if current_regime_data else 'UNKNOWN'
        allocation = current_regime_data.get('allocation_pct', 0) if current_regime_data else 0

        # Map regime to numeric value for gauge
        regime_values = {'CRASH': 0, 'TREND_BEAR': 1, 'TREND_NEUTRAL': 2, 'TREND_BULL': 3}
        regime_value = regime_values.get(regime, 2)

        regime_colors_map = {
            'TREND_BULL': COLORS['bull_primary'],
            'TREND_NEUTRAL': COLORS['text_secondary'],
            'TREND_BEAR': COLORS['warning'],
            'CRASH': COLORS['danger']
        }
        regime_color = regime_colors_map.get(regime, COLORS['text_secondary'])

        # Create indicator figure
        fig = go.Figure()

        # Regime gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=regime_value,
            number={'suffix': '', 'font': {'size': 1, 'color': COLORS['background_dark']}},  # Hide number
            gauge={
                'axis': {'range': [0, 3], 'tickvals': [0, 1, 2, 3], 'ticktext': ['CRASH', 'BEAR', 'NEUTRAL', 'BULL'],
                        'tickfont': {'color': COLORS['text_secondary'], 'size': 10}},
                'bar': {'color': regime_color, 'thickness': 0.75},
                'bgcolor': COLORS['background_medium'],
                'borderwidth': 2,
                'bordercolor': COLORS['grid'],
                'steps': [
                    {'range': [0, 0.75], 'color': 'rgba(255, 82, 82, 0.2)'},
                    {'range': [0.75, 1.5], 'color': 'rgba(255, 193, 7, 0.2)'},
                    {'range': [1.5, 2.25], 'color': 'rgba(128, 128, 128, 0.2)'},
                    {'range': [2.25, 3], 'color': 'rgba(0, 200, 83, 0.2)'}
                ],
            },
            domain={'x': [0.1, 0.9], 'y': [0.3, 1]}
        ))

        # Add regime label
        fig.add_annotation(
            text=f"<b>{regime.replace('TREND_', '')}</b>",
            x=0.5, y=0.45,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=24, color=regime_color)
        )

        # Add allocation and P&L info
        pnl_color = COLORS['bull_primary'] if total_pnl >= 0 else COLORS['danger']
        fig.add_annotation(
            text=f"Allocation: {allocation}% | Total P&L: <span style='color:{pnl_color}'>${total_pnl:+,.2f}</span>",
            x=0.5, y=0.15,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=12, color=COLORS['text_secondary'])
        )

        fig.update_layout(
            title={'text': 'Current Regime', 'font': {'color': COLORS['text_primary']}},
            template='plotly_dark',
            paper_bgcolor=COLORS['background_dark'],
            plot_bgcolor=COLORS['background_dark'],
            font={'color': COLORS['text_primary']},
            height=350,
            margin=dict(t=50, b=30, l=30, r=30)
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating regime comparison: {e}")
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    Output('trade-distribution-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_trade_distribution(strategy_name, n):
    """Show trade distribution - wins/losses for STRAT Options, positions for others."""
    import plotly.graph_objects as go

    try:
        # Handle STRAT Options - show win/loss distribution
        if strategy_name == 'strat_options':
            if options_loader is None:
                return create_info_figure("Options Loader Not Available", "Cannot fetch closed trades")

            summary = options_loader.get_closed_trades_summary(days=30)

            if summary.get('total_trades', 0) == 0:
                return create_info_figure("No Closed Trades", "No trades in the last 30 days")

            wins = summary.get('win_count', 0)
            losses = summary.get('loss_count', 0)
            avg_win = summary.get('avg_win', 0)
            avg_loss = abs(summary.get('avg_loss', 0))

            # Create pie chart for wins vs losses
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    marker_colors=[COLORS['bull_primary'], COLORS['danger']],
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
                    hole=0.4
                )
            ])

            # Add center text
            fig.add_annotation(
                text=f"<b>{summary.get('total_trades', 0)}</b><br>Trades",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=18, color=COLORS['text_primary'])
            )

            # Add subtitle with average win/loss
            fig.add_annotation(
                text=f"Avg Win: ${avg_win:,.2f} | Avg Loss: ${avg_loss:,.2f}",
                x=0.5, y=-0.1,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=12, color=COLORS['text_secondary'])
            )

            fig.update_layout(
                title={'text': 'Win/Loss Distribution', 'font': {'color': COLORS['text_primary']}},
                template='plotly_dark',
                paper_bgcolor=COLORS['background_dark'],
                plot_bgcolor=COLORS['background_dark'],
                font={'color': COLORS['text_primary']},
                height=350,
                margin=dict(l=30, r=30, t=50, b=50),
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.2,
                    xanchor='center',
                    x=0.5
                )
            )

            return fig

        # Default: Show current positions
        if live_loader is None or live_loader.client is None:
            return create_info_figure("Live Data Not Available", "Alpaca connection not established")

        positions = live_loader.get_current_positions()

        if positions.empty:
            return create_info_figure("No Open Positions", "No positions to display")

        # Create horizontal bar chart showing position sizes
        symbols = positions['symbol'].tolist()
        quantities = positions['qty'].tolist() if 'qty' in positions.columns else [0] * len(symbols)
        avg_prices = positions['avg_entry_price'].tolist() if 'avg_entry_price' in positions.columns else [0] * len(symbols)
        current_prices = positions['current_price'].tolist() if 'current_price' in positions.columns else [0] * len(symbols)
        unrealized_pls = positions['unrealized_pl'].tolist() if 'unrealized_pl' in positions.columns else [0] * len(symbols)

        # Calculate P&L percentage for coloring
        pnl_pcts = []
        for i in range(len(symbols)):
            if avg_prices[i] > 0:
                pnl_pct = ((current_prices[i] - avg_prices[i]) / avg_prices[i]) * 100
            else:
                pnl_pct = 0
            pnl_pcts.append(pnl_pct)

        colors = [COLORS['bull_primary'] if pct >= 0 else COLORS['danger'] for pct in pnl_pcts]

        # Prepare custom data for hover
        custom_data = list(zip(quantities, avg_prices, current_prices, unrealized_pls))

        fig = go.Figure(data=[
            go.Bar(
                y=symbols,
                x=pnl_pcts,
                orientation='h',
                marker_color=colors,
                text=[f"{pct:+.1f}%" for pct in pnl_pcts],
                textposition='outside',
                textfont=dict(size=11),
                customdata=custom_data,
                hovertemplate='<b>%{y}</b><br>' +
                              'P&L: %{x:+.2f}%<br>' +
                              'Qty: %{customdata[0]}<br>' +
                              'Avg Entry: $%{customdata[1]:.2f}<br>' +
                              'Current: $%{customdata[2]:.2f}<br>' +
                              'P&L $: $%{customdata[3]:+,.2f}<extra></extra>'
            )
        ])

        # Calculate x-axis range with padding
        max_pct = max(pnl_pcts) if pnl_pcts else 0
        min_pct = min(pnl_pcts) if pnl_pcts else 0
        x_padding = max(abs(max_pct), abs(min_pct), 1) * 0.25

        fig.update_layout(
            title={'text': 'Position Performance (%)', 'font': {'color': COLORS['text_primary']}},
            template='plotly_dark',
            paper_bgcolor=COLORS['background_dark'],
            plot_bgcolor=COLORS['background_dark'],
            font={'color': COLORS['text_primary']},
            xaxis={
                'title': 'P&L %',
                'gridcolor': COLORS['grid'],
                'range': [min_pct - x_padding, max_pct + x_padding]
            },
            yaxis={'title': '', 'gridcolor': COLORS['grid']},
            height=350,
            margin=dict(l=80, r=60, t=50, b=50),
            showlegend=False
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating trade distribution: {e}")
        return create_error_figure(f"Error: {str(e)}")


@app.callback(
    [Output('portfolio-value-card', 'children'),
     Output('positions-table', 'data'),
     Output('portfolio-heat-gauge', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_live_portfolio(n):
    """
    Update live portfolio metrics (auto-refreshes every 30 seconds).

    Args:
        n: Number of intervals elapsed

    Returns:
        Tuple of (portfolio card, positions data, heat gauge figure)
    """
    try:
        if live_loader is None or live_loader.client is None:
            error_msg = "Live data not available"
            if live_loader and live_loader.init_error:
                error_msg = f"Alpaca connection failed: {live_loader.init_error}"
            logger.warning(f"update_live_portfolio: {error_msg}")
            error_card = create_error_card(error_msg)
            return error_card, [], create_error_figure(error_msg)

        # Get account status
        account = live_loader.get_account_status()

        # Get current positions
        positions = live_loader.get_current_positions()

        # Calculate portfolio heat (max position concentration as proxy)
        # True heat would require stop loss data; using concentration instead
        equity = account.get('equity', 0)
        if not positions.empty and equity > 0:
            max_position_pct = positions['market_value'].max() / equity
            current_heat = max_position_pct  # 0-1 scale for gauge function
        else:
            current_heat = 0

        # Create portfolio value card
        portfolio_card = dbc.Card([
            dbc.CardBody([
                html.H3(
                    f"${account.get('portfolio_value', 0):,.2f}",
                    className='text-primary mb-2'
                ),
                html.P([
                    html.Span('P&L Today: ', className='text-muted'),
                    html.Span(
                        f"${account.get('equity', 0) - account.get('last_equity', 0):+,.2f}",
                        className='text-success' if account.get('equity', 0) > account.get('last_equity', 0) else 'text-danger'
                    )
                ], className='mb-0')
            ])
        ], className='shadow-sm')

        # Format positions for table
        positions_data = positions.to_dict('records') if not positions.empty else []

        # Create heat gauge
        heat_gauge = create_portfolio_heat_gauge(current_heat)

        return portfolio_card, positions_data, heat_gauge

    except Exception as e:
        logger.error(f"Error updating live portfolio: {e}")
        error_card = create_error_card(f"Error: {str(e)}")
        return error_card, [], create_error_figure(f"Error: {str(e)}")


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_error_figure(message: str):
    """
    Create an error figure with message.

    Args:
        message: Error message to display

    Returns:
        Plotly figure with error message
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS['danger'])
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        xaxis={'visible': False},
        yaxis={'visible': False},
        height=400,
    )
    return fig


def create_info_figure(title: str, message: str):
    """
    Create an informational figure with title and message.

    Args:
        title: Title text
        message: Informational message

    Returns:
        Plotly figure with info message (not an error)
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{title}</b><br><br>{message}",
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS['text_secondary']),
        align='center'
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        xaxis={'visible': False},
        yaxis={'visible': False},
        height=350,
    )
    return fig


def create_error_card(message: str):
    """
    Create an error card with message.

    Args:
        message: Error message to display

    Returns:
        Bootstrap card with error message
    """
    return dbc.Card([
        dbc.CardBody([
            html.I(className='fas fa-exclamation-triangle text-warning me-2'),
            html.Span(message, className='text-muted')
        ])
    ], className='shadow-sm')


# ============================================
# RISK MANAGEMENT CALLBACKS
# ============================================

@app.callback(
    [Output('risk-heat-gauge', 'figure'),
     Output('risk-metrics-table', 'children'),
     Output('position-allocation-chart', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_risk_management(n):
    """
    Update risk management visualizations.

    Calculates portfolio heat and risk metrics from live positions.
    """
    import plotly.graph_objects as go

    try:
        if live_loader is None or live_loader.client is None:
            error_fig = create_error_figure("Live data not available")
            error_table = html.P("No data available", className='text-muted')
            return error_fig, error_table, error_fig

        # Get positions and account data
        positions = live_loader.get_current_positions()
        account = live_loader.get_account_status()

        equity = account.get('equity', 0)

        # Calculate portfolio heat (sum of position risks)
        # For now, use position concentration as proxy for heat
        if not positions.empty and equity > 0:
            total_market_value = positions['market_value'].sum()
            max_position_pct = positions['market_value'].max() / equity if equity > 0 else 0
            position_count = len(positions)

            # Simple heat calculation: higher concentration = higher heat
            # Target: <5% per position, <8% total portfolio heat
            avg_position_pct = (total_market_value / equity) / position_count if position_count > 0 else 0
            portfolio_heat = max_position_pct * 100  # Use max position % as heat proxy
        else:
            portfolio_heat = 0
            max_position_pct = 0
            position_count = 0

        # Create heat gauge
        heat_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=portfolio_heat,
            title={'text': "Portfolio Heat %", 'font': {'color': COLORS['text_primary']}},
            delta={'reference': 5, 'increasing': {'color': COLORS['bear_primary']}, 'decreasing': {'color': COLORS['bull_primary']}},
            gauge={
                'axis': {'range': [0, 15], 'tickcolor': COLORS['text_secondary']},
                'bar': {'color': COLORS['bull_primary'] if portfolio_heat < 5 else COLORS['warning'] if portfolio_heat < 8 else COLORS['bear_primary']},
                'bgcolor': COLORS['background_medium'],
                'borderwidth': 2,
                'bordercolor': COLORS['grid'],
                'steps': [
                    {'range': [0, 5], 'color': 'rgba(0, 200, 83, 0.2)'},
                    {'range': [5, 8], 'color': 'rgba(255, 193, 7, 0.2)'},
                    {'range': [8, 15], 'color': 'rgba(255, 82, 82, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': COLORS['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ))
        heat_gauge.update_layout(
            paper_bgcolor=COLORS['background_dark'],
            font={'color': COLORS['text_primary']},
            height=250
        )

        # Create risk metrics table
        metrics_table = html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("Portfolio Value", className='text-muted'),
                    html.Td(f"${equity:,.2f}", className='text-end fw-bold')
                ]),
                html.Tr([
                    html.Td("Open Positions", className='text-muted'),
                    html.Td(f"{position_count}", className='text-end')
                ]),
                html.Tr([
                    html.Td("Max Position %", className='text-muted'),
                    html.Td(f"{max_position_pct*100:.1f}%", className='text-end',
                           style={'color': COLORS['bull_primary'] if max_position_pct < 0.15 else COLORS['warning']})
                ]),
                html.Tr([
                    html.Td("Cash Available", className='text-muted'),
                    html.Td(f"${account.get('cash', 0):,.2f}", className='text-end')
                ]),
            ])
        ], className='table table-sm table-borderless', style={'color': COLORS['text_primary']})

        # Create position allocation pie chart
        if not positions.empty:
            allocation_fig = go.Figure(data=[go.Pie(
                labels=positions['symbol'].tolist(),
                values=positions['market_value'].tolist(),
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                marker={'colors': [COLORS['bull_primary'], COLORS['bull_secondary'],
                                  COLORS['warning'], COLORS['info'],
                                  COLORS['bear_secondary'], COLORS['bear_primary']]}
            )])
            allocation_fig.update_layout(
                paper_bgcolor=COLORS['background_dark'],
                plot_bgcolor=COLORS['background_dark'],
                font={'color': COLORS['text_primary']},
                showlegend=True,
                legend={'orientation': 'h', 'y': -0.1},
                height=400,
                title={'text': 'Position Allocation', 'font': {'color': COLORS['text_primary']}}
            )
        else:
            allocation_fig = create_error_figure("No positions to display")

        return heat_gauge, metrics_table, allocation_fig

    except Exception as e:
        logger.error(f"Error updating risk management: {e}", exc_info=True)
        error_fig = create_error_figure(f"Error: {str(e)}")
        error_table = html.P(f"Error: {str(e)}", className='text-danger')
        return error_fig, error_table, error_fig


# ============================================
# OPTIONS TRADING CALLBACKS
# ============================================

@app.callback(
    [Output('options-signals-container', 'children'),
     Output('options-signals-count', 'children'),
     Output('signals-setups-container', 'children'),
     Output('signals-triggered-container', 'children'),
     Output('signals-lowmag-container', 'children'),
     Output('signals-closed-container', 'children')],
    [Input('options-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_options_signals(n_intervals, active_tab):
    """
    Update STRAT signals across all tabs.

    Only updates when options tab is active to save resources.
    Populates four tabs: Active Setups, Triggered, Low Magnitude, Closed Trades.
    """
    try:
        # Skip update if not on options tab
        if active_tab != 'options-tab':
            from dash import no_update
            return no_update, no_update, no_update, no_update, no_update, no_update

        if options_loader is None:
            empty_table = create_signals_table([])
            empty_closed = create_closed_trades_table([])
            return (
                empty_table,
                dbc.Badge('0', color='secondary'),
                empty_table,
                empty_table,
                empty_table,
                empty_closed
            )

        # Get signals by category
        categories = options_loader.get_signals_by_category()
        setups = categories.get('setups', [])
        triggered = categories.get('triggered', [])
        low_mag = categories.get('low_magnitude', [])

        # Get closed trades (30-day default)
        closed_trades = options_loader.get_closed_trades(days=30)

        # Total count for badge
        total_count = len(setups) + len(triggered)
        badge_color = 'success' if total_count > 0 else 'secondary'

        # Create tables for each tab
        # Setups: show detected time
        setups_table = create_signals_table(setups, show_triggered_time=False)
        # Triggered: show triggered time
        triggered_table = create_signals_table(triggered, show_triggered_time=True)
        # Low magnitude: show detected time
        lowmag_table = create_signals_table(low_mag, show_triggered_time=False)
        # Closed trades: show realized P&L
        closed_table = create_closed_trades_table(closed_trades)

        # Hidden container for backward compatibility
        all_signals = options_loader.get_active_signals()
        hidden_table = create_signals_table(all_signals)

        return (
            hidden_table,
            dbc.Badge(str(total_count), color=badge_color, className='ms-2'),
            setups_table,
            triggered_table,
            lowmag_table,
            closed_table
        )

    except Exception as e:
        logger.error(f"Error updating options signals: {e}")
        empty_table = create_signals_table([])
        empty_closed = create_closed_trades_table([])
        return (
            empty_table,
            dbc.Badge('!', color='danger'),
            empty_table,
            empty_table,
            empty_table,
            empty_closed
        )


@app.callback(
    [Output('options-positions-container', 'children'),
     Output('options-positions-count', 'children')],
    [Input('options-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_options_positions(n_intervals, active_tab):
    """
    Update live options positions table.

    Only updates when options tab is active to save resources.
    """
    try:
        # Skip update if not on options tab
        if active_tab != 'options-tab':
            from dash import no_update
            return no_update, no_update

        if options_loader is None:
            return create_positions_table([]), dbc.Badge('0', color='secondary')

        positions = options_loader.get_option_positions()
        count = len(positions)

        # Badge color based on count
        badge_color = 'info' if count > 0 else 'secondary'

        return (
            create_positions_table(positions),
            dbc.Badge(str(count), color=badge_color, className='ms-2')
        )

    except Exception as e:
        logger.error(f"Error updating options positions: {e}")
        return create_positions_table([]), dbc.Badge('!', color='danger')


@app.callback(
    Output('pnl-summary-content', 'children'),
    [Input('options-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_pnl_summary(n_intervals, active_tab):
    """
    Update P&L summary with live position data.

    Only updates when options tab is active.
    """
    try:
        if active_tab != 'options-tab':
            from dash import no_update
            return no_update

        if options_loader is None:
            return create_pnl_summary([])

        positions = options_loader.get_option_positions()
        return create_pnl_summary(positions)

    except Exception as e:
        logger.error(f"Error updating P&L summary: {e}")
        return create_pnl_summary([])


@app.callback(
    Output('trade-progress-container', 'children'),
    [Input('options-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_trade_progress(n_intervals, active_tab):
    """
    Update trade progress display with live position data.

    Session EQUITY-33: Links positions to their original signals
    to show entry -> current -> target progress.

    Session EQUITY-34: Simplified from Plotly chart to HTML progress bars.

    Only updates when options tab is active.
    """
    try:
        if active_tab != 'options-tab':
            from dash import no_update
            return no_update

        # Get positions linked to their signals
        trades = options_loader.get_positions_with_signals()
        return create_trade_progress_display(trades)

    except Exception as e:
        logger.error(f"Error updating trade progress: {e}")
        return create_trade_progress_display([])


# ============================================
# CRYPTO TRADING CALLBACKS (Session CRYPTO-6)
# ============================================

@app.callback(
    [Output('crypto-account-summary', 'children'),
     Output('crypto-daemon-status', 'children'),
     Output('crypto-positions-container', 'children'),
     Output('crypto-positions-count', 'children')],
    [Input('crypto-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_crypto_overview(n_intervals, active_tab):
    """
    Update crypto overview: account summary, daemon status, positions.

    Only updates when crypto tab is active to save resources.
    """
    try:
        # Skip update if not on crypto tab
        if active_tab != 'crypto-tab':
            from dash import no_update
            return no_update, no_update, no_update, no_update

        if crypto_loader is None:
            from dashboard.components.crypto_panel import _create_api_error_placeholder
            error_placeholder = _create_api_error_placeholder('Crypto loader not available')
            return (
                error_placeholder,
                error_placeholder,
                error_placeholder,
                dbc.Badge('!', color='danger')
            )

        # Fetch data
        account = crypto_loader.get_account_summary()
        status = crypto_loader.get_daemon_status()
        positions = crypto_loader.get_open_positions()

        # Position count badge
        pos_count = len(positions)
        badge_color = 'info' if pos_count > 0 else 'secondary'

        return (
            create_account_summary_display(account),
            create_daemon_status_display(status),
            create_crypto_positions_table(positions),
            dbc.Badge(str(pos_count), color=badge_color, className='ms-2')
        )

    except Exception as e:
        logger.error(f"Error updating crypto overview: {e}")
        from dashboard.components.crypto_panel import _create_api_error_placeholder
        error_placeholder = _create_api_error_placeholder(f'Error: {str(e)[:50]}')
        return (
            error_placeholder,
            error_placeholder,
            error_placeholder,
            dbc.Badge('!', color='danger')
        )


@app.callback(
    [Output('crypto-signals-container', 'children'),
     Output('crypto-closed-container', 'children'),
     Output('crypto-performance-container', 'children')],
    [Input('crypto-refresh-interval', 'n_intervals'),
     Input('tabs', 'active_tab')]
)
def update_crypto_tabs(n_intervals, active_tab):
    """
    Update crypto tab content: signals, closed trades, performance.

    Only updates when crypto tab is active to save resources.
    """
    try:
        # Skip update if not on crypto tab
        if active_tab != 'crypto-tab':
            from dash import no_update
            return no_update, no_update, no_update

        if crypto_loader is None:
            from dashboard.components.crypto_panel import (
                _create_no_signals_placeholder,
                _create_no_closed_trades_placeholder,
                _create_api_error_placeholder,
            )
            return (
                _create_no_signals_placeholder(),
                _create_no_closed_trades_placeholder(),
                _create_api_error_placeholder('Crypto loader not available')
            )

        # Fetch data
        signals = crypto_loader.get_pending_signals()
        closed = crypto_loader.get_closed_trades(limit=50)
        metrics = crypto_loader.get_performance_metrics()

        return (
            create_crypto_signals_table(signals),
            create_crypto_closed_table(closed),
            create_performance_metrics(metrics)
        )

    except Exception as e:
        logger.error(f"Error updating crypto tabs: {e}")
        from dashboard.components.crypto_panel import (
            _create_no_signals_placeholder,
            _create_no_closed_trades_placeholder,
            _create_api_error_placeholder,
        )
        return (
            _create_no_signals_placeholder(),
            _create_no_closed_trades_placeholder(),
            _create_api_error_placeholder(f'Error: {str(e)[:50]}')
        )


# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    logger.info(f"Starting ATLAS Dashboard on {DASHBOARD_CONFIG['host']}:{DASHBOARD_CONFIG['port']}")
    app.run(
        debug=DASHBOARD_CONFIG['debug'],
        host=DASHBOARD_CONFIG['host'],
        port=DASHBOARD_CONFIG['port']
    )
