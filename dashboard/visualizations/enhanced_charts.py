"""
Enhanced Chart Visualizations for ATLAS Dashboard

Provides advanced, TradingView-quality chart components with:
- STRAT pattern overlays with measured move targets
- Candlestick charts with regime background shading
- Multi-timeframe confluence indicators
- Options flow visualization
- Enhanced performance attribution charts

Visual Improvements:
- Gradient fills for better depth perception
- Animated transitions for real-time updates
- Sparkline mini-charts for quick metrics
- Heatmaps for correlation analysis
- Sankey diagrams for P&L attribution

Session 76: Enhanced visualizations for improved dashboard UX.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta

from dashboard.config import (
    COLORS,
    REGIME_COLORS,
    STRAT_COLORS,
    STRAT_BORDER_COLORS,
    TRADE_MARKERS,
    PERFORMANCE_THRESHOLDS,
    CHART_HEIGHT,
    FONTS,
)


# ============================================
# ENHANCED CANDLESTICK WITH STRAT PATTERNS
# ============================================

def create_strat_candlestick_chart(
    ohlc_data: pd.DataFrame,
    bar_types: Optional[pd.Series] = None,
    patterns: Optional[List[Dict]] = None,
    regimes: Optional[pd.Series] = None,
    height: int = 700
) -> go.Figure:
    """
    Create professional candlestick chart with STRAT pattern visualization.

    Features:
    - Color-coded candlesticks by bar type (1, 2U, 2D, 3)
    - Pattern entry/exit zones with measured move targets
    - Regime background shading
    - Volume profile on secondary axis

    Args:
        ohlc_data: DataFrame with columns [Open, High, Low, Close, Volume]
        bar_types: Series with bar classifications (1, 2, -2, 3)
        patterns: List of detected patterns with entries, targets, stops
        regimes: Series with regime labels

    Returns:
        Plotly Figure with STRAT-annotated candlestick chart
    """
    # Create subplot with volume
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price Action with STRAT Patterns', 'Volume')
    )

    # ============================================
    # LAYER 1: Regime Background Shading (Lowest)
    # ============================================
    if regimes is not None and len(regimes) > 0:
        _add_regime_shading(fig, ohlc_data.index, regimes, row=1)

    # ============================================
    # LAYER 2: Candlesticks with STRAT Coloring
    # ============================================

    # Determine candle colors based on bar types
    if bar_types is not None:
        colors_up = []
        colors_down = []

        for bt in bar_types:
            if bt == 1:  # Inside bar - Gray
                colors_up.append('rgba(128, 128, 128, 0.8)')
                colors_down.append('rgba(128, 128, 128, 0.8)')
            elif bt == 2:  # 2U - Green
                colors_up.append(COLORS['bull_primary'])
                colors_down.append(COLORS['bull_secondary'])
            elif bt == -2:  # 2D - Red
                colors_up.append(COLORS['bear_secondary'])
                colors_down.append(COLORS['bear_primary'])
            elif bt == 3 or bt == -3:  # Outside bar - Yellow
                colors_up.append('#FFD700')
                colors_down.append('#FFA500')
            else:  # Default
                colors_up.append(COLORS['bull_secondary'])
                colors_down.append(COLORS['bear_secondary'])
    else:
        colors_up = [COLORS['bull_secondary']] * len(ohlc_data)
        colors_down = [COLORS['bear_secondary']] * len(ohlc_data)

    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close'],
            name='Price',
            increasing=dict(
                line=dict(color=COLORS['bull_secondary']),
                fillcolor=COLORS['bull_secondary']
            ),
            decreasing=dict(
                line=dict(color=COLORS['bear_secondary']),
                fillcolor=COLORS['bear_secondary']
            ),
            hoverinfo='x+y'
        ),
        row=1, col=1
    )

    # ============================================
    # LAYER 3: STRAT Pattern Annotations
    # ============================================
    if patterns:
        _add_pattern_annotations(fig, patterns, ohlc_data, row=1)

    # ============================================
    # LAYER 4: Volume Bars
    # ============================================
    if 'Volume' in ohlc_data.columns:
        volume_colors = [
            COLORS['volume_up'] if ohlc_data['Close'].iloc[i] >= ohlc_data['Open'].iloc[i]
            else COLORS['volume_down']
            for i in range(len(ohlc_data))
        ]

        fig.add_trace(
            go.Bar(
                x=ohlc_data.index,
                y=ohlc_data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ),
            row=2, col=1
        )

    # ============================================
    # Layout Configuration (Premium Luxury)
    # ============================================
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font=dict(
            family=FONTS['body'],
            size=12,
            color=COLORS['text_primary']
        ),
        title=dict(
            font=dict(
                family=FONTS['display'],
                size=16,
                color=COLORS['text_primary'],
            ),
            x=0,
            xanchor='left',
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(
                family=FONTS['body'],
                size=11,
                color=COLORS['text_secondary'],
            ),
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=COLORS['bg_card'],
            bordercolor=COLORS['border_default'],
            font=dict(
                family=FONTS['body'],
                size=12,
                color=COLORS['text_primary'],
            ),
        ),
        margin=dict(l=60, r=30, t=60, b=40)
    )

    # Update axes with premium styling
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        linecolor=COLORS['border_subtle'],
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
    )

    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        linecolor=COLORS['border_subtle'],
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        title_text='Price ($)',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        row=1, col=1
    )

    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        gridwidth=1,
        showgrid=True,
        linecolor=COLORS['border_subtle'],
        tickfont=dict(family=FONTS['mono'], size=10, color=COLORS['text_tertiary']),
        title_text='Volume',
        title_font=dict(family=FONTS['body'], size=11, color=COLORS['text_secondary']),
        row=2, col=1
    )

    return fig


def _add_regime_shading(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    regimes: pd.Series,
    row: int = 1
):
    """Add regime background shading to figure."""
    if len(regimes) == 0:
        return

    current_regime = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[min(i, len(dates) - 1)],
                fillcolor=REGIME_COLORS.get(current_regime, 'rgba(128,128,128,0.1)'),
                layer='below',
                line_width=0,
                row=row, col=1
            )
            current_regime = regimes.iloc[i]
            start_idx = i


def _add_pattern_annotations(
    fig: go.Figure,
    patterns: List[Dict],
    ohlc_data: pd.DataFrame,
    row: int = 1
):
    """Add STRAT pattern annotations with entry, target, stop levels."""
    for pattern in patterns:
        pattern_type = pattern.get('type', 'Unknown')
        entry_date = pattern.get('entry_date')
        entry_price = pattern.get('entry_price', 0)
        target_price = pattern.get('target_price', 0)
        stop_price = pattern.get('stop_price', 0)
        direction = pattern.get('direction', 'long')

        if entry_date is None:
            continue

        # Pattern entry marker
        marker_color = COLORS['bull_primary'] if direction == 'long' else COLORS['bear_primary']

        fig.add_trace(
            go.Scatter(
                x=[entry_date],
                y=[entry_price],
                mode='markers+text',
                name=f'{pattern_type}',
                marker=dict(
                    symbol='triangle-up' if direction == 'long' else 'triangle-down',
                    size=14,
                    color=marker_color,
                    line=dict(color='white', width=2)
                ),
                text=[pattern_type],
                textposition='top center' if direction == 'long' else 'bottom center',
                textfont=dict(size=10, color=marker_color),
                showlegend=False,
                hovertemplate=(
                    f'<b>{pattern_type}</b><br>'
                    f'Entry: ${entry_price:.2f}<br>'
                    f'Target: ${target_price:.2f}<br>'
                    f'Stop: ${stop_price:.2f}<extra></extra>'
                )
            ),
            row=row, col=1
        )

        # Target level line (dashed green/red)
        if target_price > 0:
            target_color = COLORS['bull_primary'] if direction == 'long' else COLORS['bear_primary']

            fig.add_shape(
                type='line',
                x0=entry_date,
                x1=entry_date + timedelta(days=5),  # Extend 5 bars
                y0=target_price,
                y1=target_price,
                line=dict(color=target_color, width=1, dash='dash'),
                row=row, col=1
            )

            fig.add_annotation(
                x=entry_date + timedelta(days=2),
                y=target_price,
                text=f'T: ${target_price:.2f}',
                showarrow=False,
                font=dict(size=9, color=target_color),
                bgcolor='rgba(0,0,0,0.7)',
                row=row, col=1
            )

        # Stop level line (dashed red/green)
        if stop_price > 0:
            stop_color = COLORS['bear_primary'] if direction == 'long' else COLORS['bull_primary']

            fig.add_shape(
                type='line',
                x0=entry_date,
                x1=entry_date + timedelta(days=5),
                y0=stop_price,
                y1=stop_price,
                line=dict(color=stop_color, width=1, dash='dot'),
                row=row, col=1
            )

            fig.add_annotation(
                x=entry_date + timedelta(days=2),
                y=stop_price,
                text=f'S: ${stop_price:.2f}',
                showarrow=False,
                font=dict(size=9, color=stop_color),
                bgcolor='rgba(0,0,0,0.7)',
                row=row, col=1
            )


# ============================================
# ENHANCED PORTFOLIO METRICS DISPLAY
# ============================================

def create_portfolio_metrics_cards(
    equity: float,
    pnl_today: float,
    pnl_pct: float,
    positions_count: int,
    buying_power: float,
    portfolio_heat: float
) -> go.Figure:
    """
    Create visual metrics cards using Plotly indicators.

    Shows key portfolio metrics in a compact, scannable format.

    Args:
        equity: Total account equity
        pnl_today: Today's P&L in dollars
        pnl_pct: Today's P&L as percentage
        positions_count: Number of open positions
        buying_power: Available buying power
        portfolio_heat: Current portfolio heat (risk)

    Returns:
        Plotly Figure with indicator cards
    """
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.2
    )

    # Equity
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=equity,
            title={'text': 'Portfolio Value', 'font': {'size': 14}},
            number={'prefix': '$', 'valueformat': ',.0f', 'font': {'size': 28}},
            domain={'row': 0, 'column': 0}
        ),
        row=1, col=1
    )

    # Daily P&L
    pnl_color = COLORS['bull_primary'] if pnl_today >= 0 else COLORS['bear_primary']
    fig.add_trace(
        go.Indicator(
            mode='number+delta',
            value=pnl_today,
            title={'text': "Today's P&L", 'font': {'size': 14}},
            number={'prefix': '$', 'valueformat': '+,.0f', 'font': {'size': 28, 'color': pnl_color}},
            delta={'reference': 0, 'valueformat': '.2%', 'relative': False},
            domain={'row': 0, 'column': 1}
        ),
        row=1, col=2
    )

    # Positions Count
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=positions_count,
            title={'text': 'Open Positions', 'font': {'size': 14}},
            number={'font': {'size': 28}},
            domain={'row': 0, 'column': 2}
        ),
        row=1, col=3
    )

    # Buying Power
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=buying_power,
            title={'text': 'Buying Power', 'font': {'size': 14}},
            number={'prefix': '$', 'valueformat': ',.0f', 'font': {'size': 28}},
            domain={'row': 1, 'column': 0}
        ),
        row=2, col=1
    )

    # Portfolio Heat (Gauge)
    heat_color = (
        COLORS['success'] if portfolio_heat < 0.04 else
        COLORS['warning'] if portfolio_heat < 0.06 else
        COLORS['danger']
    )
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=portfolio_heat * 100,
            title={'text': 'Portfolio Heat', 'font': {'size': 14}},
            number={'suffix': '%', 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 10], 'ticksuffix': '%'},
                'bar': {'color': heat_color},
                'steps': [
                    {'range': [0, 4], 'color': 'rgba(0, 200, 83, 0.3)'},
                    {'range': [4, 6], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [6, 10], 'color': 'rgba(255, 23, 68, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': COLORS['danger'], 'width': 2},
                    'thickness': 0.75,
                    'value': 8  # 8% max heat limit
                }
            },
            domain={'row': 1, 'column': 1}
        ),
        row=2, col=2
    )

    # Market Status (placeholder)
    fig.add_trace(
        go.Indicator(
            mode='number',
            value=1,  # 1 = Open, 0 = Closed
            title={'text': 'Market Status', 'font': {'size': 14}},
            number={'font': {'size': 0}},  # Hide number, show just title
            domain={'row': 1, 'column': 2}
        ),
        row=2, col=3
    )

    # Add market status annotation
    fig.add_annotation(
        x=0.92, y=0.15,
        text='OPEN',
        showarrow=False,
        font=dict(size=20, color=COLORS['success']),
        xref='paper', yref='paper'
    )

    fig.update_layout(
        height=300,
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font={'color': COLORS['text_primary']},
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================
# MULTI-TIMEFRAME REGIME HEATMAP
# ============================================

def create_regime_heatmap(
    regime_data: Dict[str, pd.Series],
    symbols: List[str]
) -> go.Figure:
    """
    Create heatmap showing regime states across timeframes.

    Useful for multi-timeframe analysis and confluence detection.

    Args:
        regime_data: Dict mapping timeframe to regime Series
        symbols: List of symbols for y-axis

    Returns:
        Plotly heatmap figure
    """
    # Map regimes to numeric values
    regime_map = {
        'TREND_BULL': 3,
        'TREND_NEUTRAL': 2,
        'TREND_BEAR': 1,
        'CRASH': 0
    }

    timeframes = list(regime_data.keys())
    z_data = []

    for symbol in symbols:
        row = []
        for tf in timeframes:
            regime = regime_data[tf].get(symbol, 'TREND_NEUTRAL')
            row.append(regime_map.get(regime, 2))
        z_data.append(row)

    # Custom colorscale: CRASH (red) -> BEAR (orange) -> NEUTRAL (gray) -> BULL (green)
    colorscale = [
        [0.0, COLORS['bear_primary']],      # CRASH
        [0.33, '#FFA500'],                   # BEAR (orange)
        [0.66, '#808080'],                   # NEUTRAL (gray)
        [1.0, COLORS['bull_primary']]        # BULL
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=timeframes,
        y=symbols,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title='Regime',
            tickvals=[0, 1, 2, 3],
            ticktext=['CRASH', 'BEAR', 'NEUTRAL', 'BULL'],
            titleside='right'
        ),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Timeframe: %{x}<br>'
            'Regime: %{z}<extra></extra>'
        )
    ))

    fig.update_layout(
        title='Multi-Timeframe Regime Analysis',
        height=400,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font={'color': COLORS['text_primary']},
        xaxis_title='Timeframe',
        yaxis_title='Symbol'
    )

    return fig


# ============================================
# P&L ATTRIBUTION WATERFALL CHART
# ============================================

def create_pnl_waterfall(
    contributions: Dict[str, float],
    total_pnl: float
) -> go.Figure:
    """
    Create waterfall chart showing P&L attribution by source.

    Args:
        contributions: Dict mapping source name to P&L contribution
        total_pnl: Total P&L (should equal sum of contributions)

    Returns:
        Plotly waterfall figure
    """
    sources = list(contributions.keys())
    values = list(contributions.values())

    # Determine colors based on positive/negative
    colors = [
        COLORS['bull_primary'] if v >= 0 else COLORS['bear_primary']
        for v in values
    ]

    fig = go.Figure(go.Waterfall(
        name='P&L',
        orientation='v',
        measure=['relative'] * len(sources) + ['total'],
        x=sources + ['Total'],
        y=values + [total_pnl],
        textposition='outside',
        text=[f'${v:+,.0f}' for v in values] + [f'${total_pnl:+,.0f}'],
        connector={'line': {'color': COLORS['grid']}},
        increasing={'marker': {'color': COLORS['bull_secondary']}},
        decreasing={'marker': {'color': COLORS['bear_secondary']}},
        totals={'marker': {'color': COLORS['info']}}
    ))

    fig.update_layout(
        title='P&L Attribution',
        height=400,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font={'color': COLORS['text_primary']},
        yaxis_title='P&L ($)',
        showlegend=False
    )

    return fig


# ============================================
# SPARKLINE MINI-CHARTS
# ============================================

def create_sparkline(
    data: pd.Series,
    positive_color: str = None,
    negative_color: str = None,
    width: int = 150,
    height: int = 40
) -> go.Figure:
    """
    Create compact sparkline chart for inline metrics.

    Args:
        data: Time series data
        positive_color: Color for upward trend
        negative_color: Color for downward trend
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        Minimal Plotly figure
    """
    positive_color = positive_color or COLORS['bull_primary']
    negative_color = negative_color or COLORS['bear_primary']

    # Determine overall trend
    trend_color = positive_color if data.iloc[-1] >= data.iloc[0] else negative_color

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data.values,
        mode='lines',
        line=dict(color=trend_color, width=1.5),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(list(int(trend_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
        hoverinfo='skip'
    ))

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    return fig


# ============================================
# OPTIONS FLOW VISUALIZATION
# ============================================

def create_options_flow_chart(
    flow_data: pd.DataFrame,
    height: int = 500
) -> go.Figure:
    """
    Create options flow visualization showing unusual activity.

    Args:
        flow_data: DataFrame with columns [symbol, strike, expiry, type, premium, volume]
        height: Chart height

    Returns:
        Plotly figure with options flow
    """
    if flow_data.empty:
        # Return placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No options flow data available",
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=COLORS['text_secondary'])
        )
        fig.update_layout(
            height=height,
            paper_bgcolor=COLORS['background_dark'],
            plot_bgcolor=COLORS['background_dark']
        )
        return fig

    # Bubble chart: x=strike, y=premium, size=volume, color=call/put
    fig = go.Figure()

    # Calls
    calls = flow_data[flow_data['type'] == 'CALL']
    if not calls.empty:
        fig.add_trace(go.Scatter(
            x=calls['strike'],
            y=calls['premium'],
            mode='markers',
            name='Calls',
            marker=dict(
                size=calls['volume'] / calls['volume'].max() * 50,
                color=COLORS['bull_primary'],
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            text=calls['symbol'],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Strike: $%{x}<br>'
                'Premium: $%{y:,.0f}<br>'
                '<extra>CALL</extra>'
            )
        ))

    # Puts
    puts = flow_data[flow_data['type'] == 'PUT']
    if not puts.empty:
        fig.add_trace(go.Scatter(
            x=puts['strike'],
            y=puts['premium'],
            mode='markers',
            name='Puts',
            marker=dict(
                size=puts['volume'] / puts['volume'].max() * 50,
                color=COLORS['bear_primary'],
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            text=puts['symbol'],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Strike: $%{x}<br>'
                'Premium: $%{y:,.0f}<br>'
                '<extra>PUT</extra>'
            )
        ))

    fig.update_layout(
        title='Options Flow (Unusual Activity)',
        height=height,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font={'color': COLORS['text_primary']},
        xaxis_title='Strike Price',
        yaxis_title='Premium ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    return fig


# ============================================
# CORRELATION MATRIX HEATMAP
# ============================================

def create_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = 'Asset Correlation Matrix'
) -> go.Figure:
    """
    Create correlation matrix heatmap for portfolio analysis.

    Args:
        returns: DataFrame with asset returns (columns = assets)
        title: Chart title

    Returns:
        Plotly heatmap figure
    """
    corr = returns.corr()

    # Custom diverging colorscale: red (negative) -> white (zero) -> green (positive)
    colorscale = [
        [0.0, COLORS['bear_primary']],
        [0.5, '#FFFFFF'],
        [1.0, COLORS['bull_primary']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        height=500,
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font={'color': COLORS['text_primary']},
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    return fig


# ============================================
# REAL-TIME POSITION MONITORING CARD
# ============================================

def create_position_card(
    symbol: str,
    qty: int,
    avg_entry: float,
    current_price: float,
    unrealized_pnl: float,
    unrealized_pnl_pct: float,
    sparkline_data: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Create position card data for dashboard display.

    Args:
        symbol: Stock symbol
        qty: Position quantity
        avg_entry: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized P&L in dollars
        unrealized_pnl_pct: Unrealized P&L as percentage
        sparkline_data: Optional price series for mini-chart

    Returns:
        Dict with formatted position data and sparkline figure
    """
    is_profit = unrealized_pnl >= 0
    pnl_color = COLORS['bull_primary'] if is_profit else COLORS['bear_primary']

    card_data = {
        'symbol': symbol,
        'qty': qty,
        'side': 'LONG' if qty > 0 else 'SHORT',
        'avg_entry': f'${avg_entry:.2f}',
        'current_price': f'${current_price:.2f}',
        'unrealized_pnl': f'${unrealized_pnl:+,.2f}',
        'unrealized_pnl_pct': f'{unrealized_pnl_pct:+.2%}',
        'pnl_color': pnl_color,
        'market_value': f'${abs(qty * current_price):,.2f}'
    }

    # Add sparkline if data provided
    if sparkline_data is not None and len(sparkline_data) > 0:
        card_data['sparkline'] = create_sparkline(sparkline_data)

    return card_data
