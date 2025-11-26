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
from dashboard.components.options_panel import create_options_panel

# Import data loaders
from dashboard.data_loaders.regime_loader import RegimeDataLoader
from dashboard.data_loaders.backtest_loader import BacktestDataLoader
from dashboard.data_loaders.live_loader import LiveDataLoader
from dashboard.data_loaders.orders_loader import OrdersDataLoader

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
        'https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='ATLAS Trading Dashboard',
    meta_tags=[
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        }
    ],
)

# ============================================
# INITIALIZE DATA LOADERS
# ============================================

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
    logger.info("LiveDataLoader initialized successfully")
except Exception as e:
    logger.warning(f"LiveDataLoader initialization failed: {e}")
    live_loader = None

try:
    orders_loader = OrdersDataLoader()
    logger.info("OrdersDataLoader initialized successfully")
except Exception as e:
    logger.warning(f"OrdersDataLoader initialization failed: {e}")
    orders_loader = None

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
        if regime_loader is None:
            return create_error_figure("Regime data loader not available")

        # Load regime data
        regime_data = regime_loader.get_regime_timeline(start_date, end_date)

        if regime_data.empty:
            return create_error_figure("No regime data available for selected range")

        # Extract data
        dates = regime_data['date']
        regimes = regime_data['regime']
        prices = regime_data.get('price', pd.Series(dtype=float))

        return create_regime_timeline(dates, regimes, prices)

    except Exception as e:
        logger.error(f"Error updating regime timeline: {e}")
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
    Output('equity-curve-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_equity_curve(strategy_name, start_date, end_date):
    """
    Update strategy equity curve visualization.

    Args:
        strategy_name: Selected strategy ID
        start_date: Start date for data range
        end_date: End date for data range

    Returns:
        Plotly figure with equity curve
    """
    try:
        if backtest_loader is None:
            return create_error_figure("Backtest data loader not available")

        # Load backtest for selected strategy
        backtest_loader.load_backtest(strategy_name)
        portfolio_value = backtest_loader.get_equity_curve()

        if portfolio_value.empty:
            return create_error_figure("No backtest data available")

        return create_equity_curve(portfolio_value)

    except Exception as e:
        logger.error(f"Error updating equity curve: {e}")
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
        if live_loader is None:
            error_card = create_error_card("Live data not available")
            return error_card, [], create_error_figure("Live data not available")

        # Get account status
        account = live_loader.get_account_status()

        # Get current positions
        positions = live_loader.get_current_positions()

        # Calculate portfolio heat (placeholder - implement actual calculation)
        current_heat = 0.042  # Example: 4.2%

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
# RUN SERVER
# ============================================

if __name__ == '__main__':
    logger.info(f"Starting ATLAS Dashboard on {DASHBOARD_CONFIG['host']}:{DASHBOARD_CONFIG['port']}")
    app.run(
        debug=DASHBOARD_CONFIG['debug'],
        host=DASHBOARD_CONFIG['host'],
        port=DASHBOARD_CONFIG['port']
    )
