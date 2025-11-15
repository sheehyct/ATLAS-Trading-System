"""
Regime Detection Visualizations

This module provides visualization functions for ATLAS Layer 1 (Regime Detection)
using both Plotly (for web dashboard) and lightweight-charts-python (for TradingView-quality charts).

Visualizations include:
- Regime timeline with HMM background shading
- Feature evolution (downside deviation, Sortino ratios)
- Regime statistics tables
- Multi-layer z-ordering for clean, professional charts

Best Practices Applied:
- Regime shading at alpha=0.2 (lowest z-order)
- Grid lines above shading
- Price data on top layer
- Proper color scheme from config.py
- Mobile-friendly responsive sizing
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List
from dash import dash_table

from dashboard.config import (
    REGIME_COLORS,
    REGIME_BORDER_COLORS,
    REGIME_TEXT_COLORS,
    COLORS,
    PLOTLY_TEMPLATE,
    PLOTLY_LAYERS,
    REGIME_THRESHOLDS,
    CHART_HEIGHT,
)


def create_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: pd.Series,
    prices: Optional[pd.Series] = None
) -> go.Figure:
    """
    Create regime timeline with price overlay and background shading.

    This visualization shows market regime changes over time with:
    - Price chart with regime background colors (using add_vrect for clean shading)
    - Regime state indicator subplot
    - Proper z-ordering: regime shading → grid → price line

    Args:
        dates: DatetimeIndex for x-axis
        regimes: Series with regime labels (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH)
        prices: Optional Series with price data

    Returns:
        Plotly Figure with regime timeline

    Z-Ordering (bottom to top):
        1. Regime background shading (alpha=0.2, layer='below')
        2. Grid lines (automatic)
        3. Price line (main trace)
    """

    # Create subplot with 70/30 split
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price with Regime Overlay', 'Regime State'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # ============================================
    # LAYER 1: Regime Background Shading (Lowest)
    # ============================================

    # Track regime changes for vertical rectangles
    if len(regimes) > 0:
        current_regime = regimes.iloc[0]
        start_idx = 0

        for i in range(1, len(regimes)):
            # Detect regime change or end of data
            if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
                # Add shaded region using add_vrect (proper method for background shading)
                fig.add_vrect(
                    x0=dates[start_idx],
                    x1=dates[i] if i < len(regimes) - 1 else dates[i],
                    fillcolor=REGIME_COLORS.get(current_regime, 'rgba(128,128,128,0.2)'),
                    layer='below',  # Below traces and grid
                    line_width=0,
                    row=1, col=1
                )

                # Update for next iteration
                current_regime = regimes.iloc[i]
                start_idx = i

    # ============================================
    # LAYER 2: Price Line (Top)
    # ============================================

    if prices is not None and len(prices) > 0:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color=COLORS['price_line'], width=1.5),
                hovertemplate='<b>Price:</b> $%{y:.2f}<br><b>Date:</b> %{x}<extra></extra>'
            ),
            row=1, col=1
        )
    else:
        # Placeholder if no price data
        fig.add_annotation(
            text="Price data not available",
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['text_secondary']),
            row=1, col=1
        )

    # ============================================
    # LAYER 3: Regime State Indicator (Bottom Subplot)
    # ============================================

    # Map regime names to numeric values for visualization
    regime_numeric = regimes.map({
        'TREND_BULL': 3,
        'TREND_NEUTRAL': 2,
        'TREND_BEAR': 1,
        'CRASH': 0
    }).fillna(2)  # Default to NEUTRAL if unknown

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=regime_numeric,
            mode='lines',
            name='Regime',
            fill='tozeroy',
            line=dict(width=0),
            fillcolor=COLORS['bull_fill'],
            hovertemplate='<b>Regime:</b> %{text}<extra></extra>',
            text=regimes
        ),
        row=2, col=1
    )

    # ============================================
    # Layout Configuration
    # ============================================

    fig.update_layout(
        height=CHART_HEIGHT['regime_timeline'],
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font=dict(
            family='Segoe UI, Arial, sans-serif',
            size=12,
            color=COLORS['text_primary']
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    # Update axes
    fig.update_xaxes(
        title_text='Date',
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title_text='Price ($)',
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=1, col=1
    )

    fig.update_yaxes(
        title_text='Regime',
        ticktext=['CRASH', 'BEAR', 'NEUTRAL', 'BULL'],
        tickvals=[0, 1, 2, 3],
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=2, col=1
    )

    return fig


def create_feature_dashboard(
    dates: pd.DatetimeIndex,
    downside_dev: pd.Series,
    sortino_20d: pd.Series,
    sortino_60d: pd.Series
) -> go.Figure:
    """
    Create feature evolution dashboard showing regime detection features.

    Displays:
    - Downside Deviation (10-day EWMA) with CRASH threshold
    - Sortino Ratio 20-day and 60-day with BULL/BEAR thresholds

    Args:
        dates: DatetimeIndex for x-axis
        downside_dev: Downside deviation values
        sortino_20d: Sortino ratio 20-day values
        sortino_60d: Sortino ratio 60-day values

    Returns:
        Plotly Figure with feature evolution
    """

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Downside Deviation (10d EWMA)', 'Sortino Ratios'),
        vertical_spacing=0.15,
        shared_xaxes=True
    )

    # ============================================
    # Downside Deviation Plot
    # ============================================

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=downside_dev,
            mode='lines',
            name='Downside Dev',
            line=dict(color=COLORS['bear_primary'], width=2),
            hovertemplate='<b>DD:</b> %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add CRASH threshold line
    fig.add_hline(
        y=REGIME_THRESHOLDS['downside_dev_crash'],
        line_dash='dash',
        line_color=COLORS['danger'],
        annotation_text=f"CRASH Threshold ({REGIME_THRESHOLDS['downside_dev_crash']})",
        annotation_position='right',
        row=1, col=1
    )

    # ============================================
    # Sortino Ratios Plot
    # ============================================

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sortino_20d,
            mode='lines',
            name='Sortino 20d',
            line=dict(color=COLORS['info'], width=2),
            hovertemplate='<b>Sortino 20d:</b> %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sortino_60d,
            mode='lines',
            name='Sortino 60d',
            line=dict(color=COLORS['bull_secondary'], width=2),
            hovertemplate='<b>Sortino 60d:</b> %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add threshold lines
    fig.add_hline(
        y=REGIME_THRESHOLDS['sortino_20d_bear'],
        line_dash='dash',
        line_color=COLORS['bear_secondary'],
        annotation_text=f"BEAR Threshold ({REGIME_THRESHOLDS['sortino_20d_bear']})",
        annotation_position='left',
        row=2, col=1
    )

    fig.add_hline(
        y=REGIME_THRESHOLDS['sortino_20d_bull'],
        line_dash='dash',
        line_color=COLORS['bull_secondary'],
        annotation_text=f"BULL Threshold ({REGIME_THRESHOLDS['sortino_20d_bull']})",
        annotation_position='left',
        row=2, col=1
    )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash='solid',
        line_color=COLORS['grid_major'],
        line_width=1,
        row=2, col=1
    )

    # ============================================
    # Layout Configuration
    # ============================================

    fig.update_layout(
        height=CHART_HEIGHT['feature_dashboard'],
        template='plotly_dark',
        paper_bgcolor=COLORS['background_dark'],
        plot_bgcolor=COLORS['background_dark'],
        font=dict(
            family='Segoe UI, Arial, sans-serif',
            size=12,
            color=COLORS['text_primary']
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )

    # Update axes
    fig.update_xaxes(
        title_text='Date',
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=2, col=1
    )

    fig.update_yaxes(
        title_text='Standard Deviation',
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=1, col=1
    )

    fig.update_yaxes(
        title_text='Sortino Ratio',
        gridcolor=COLORS['grid'],
        showgrid=True,
        row=2, col=1
    )

    return fig


def create_regime_statistics_table(regime_data: pd.DataFrame) -> dash_table.DataTable:
    """
    Create summary statistics table for each regime.

    Shows:
    - Regime name
    - Days in regime
    - Average return
    - Volatility
    - Max drawdown
    - Sharpe ratio

    Args:
        regime_data: DataFrame with columns ['regime', 'returns']

    Returns:
        Dash DataTable with regime statistics
    """

    # Calculate statistics per regime
    stats = []

    for regime in ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']:
        regime_mask = regime_data['regime'] == regime

        if regime_mask.sum() == 0:
            # No data for this regime
            stats.append({
                'Regime': regime,
                'Days': 0,
                'Avg Return': 'N/A',
                'Volatility': 'N/A',
                'Max DD': 'N/A',
                'Sharpe': 'N/A'
            })
            continue

        regime_returns = regime_data.loc[regime_mask, 'returns']

        # Calculate metrics
        avg_return = regime_returns.mean()
        volatility = regime_returns.std()
        max_dd = regime_returns.min()
        sharpe = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0

        stats.append({
            'Regime': regime,
            'Days': int(regime_mask.sum()),
            'Avg Return': f"{avg_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Max DD': f"{max_dd:.2%}",
            'Sharpe': f"{sharpe:.2f}"
        })

    # Create DataTable
    return dash_table.DataTable(
        data=stats,
        columns=[{'name': col, 'id': col} for col in stats[0].keys()],
        style_data_conditional=[
            # Highlight CRASH regime
            {
                'if': {'filter_query': '{Regime} = "CRASH"'},
                'backgroundColor': 'rgba(237, 72, 7, 0.2)',
                'color': COLORS['text_primary']
            },
            # Highlight BULL regime
            {
                'if': {'filter_query': '{Regime} = "TREND_BULL"'},
                'backgroundColor': 'rgba(0, 255, 85, 0.2)',
                'color': COLORS['text_primary']
            },
            # Highlight BEAR regime
            {
                'if': {'filter_query': '{Regime} = "TREND_BEAR"'},
                'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                'color': COLORS['text_primary']
            }
        ],
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
        style_table={
            'overflowX': 'auto'
        }
    )


def create_lightweight_chart_example(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    regimes: pd.Series
) -> str:
    """
    Create TradingView-quality chart using lightweight-charts-python.

    NOTE: This is a placeholder/example function. lightweight-charts-python
    typically runs in Jupyter notebooks or standalone applications, not directly
    in Dash. For production, consider:
    - Exporting as HTML and embedding via iframe
    - Using Plotly with advanced styling (already implemented above)
    - Running lightweight-charts in a separate server

    Args:
        dates: DatetimeIndex
        prices: Price series
        regimes: Regime series

    Returns:
        HTML string or save path (implementation-dependent)
    """

    # Placeholder implementation
    # In production, you would:
    # 1. Install: pip install lightweight-charts
    # 2. Create chart with proper styling
    # 3. Export to HTML or serve separately

    implementation_note = """
    # Example lightweight-charts-python implementation:

    from lightweight_charts import Chart

    chart = Chart()

    # Configure chart style (TradingView dark theme)
    chart.layout(
        background_color='{bg}',
        text_color='{text}',
    )

    chart.grid(
        vert_enabled=True,
        horz_enabled=True,
        color='{grid}'
    )

    # Add price data
    chart.set(prices)

    # Add regime markers/shading
    for i, regime in enumerate(regimes):
        if regime == 'CRASH':
            chart.marker(position='below', color='red', shape='arrow_down')

    # Save or show
    chart.show()  # Opens in browser
    # or
    chart.save('regime_chart.html')  # Save to file
    """.format(
        bg=COLORS['background_dark'],
        text=COLORS['text_primary'],
        grid=COLORS['grid']
    )

    return implementation_note


# ============================================
# MPLFINANCE INTEGRATION (Static Charts)
# ============================================

def create_mplfinance_chart(
    ohlc_data: pd.DataFrame,
    regimes: Optional[pd.Series] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Create publication-quality static chart using mplfinance.

    Used for:
    - Exporting charts for reports/presentations
    - High-resolution static visualizations
    - Academic paper figures

    Args:
        ohlc_data: DataFrame with OHLC data (columns: Open, High, Low, Close)
        regimes: Optional regime series for background shading
        save_path: Optional path to save figure

    Returns:
        Path to saved figure or status message
    """

    try:
        import mplfinance as mpf

        # Define custom style matching TradingView theme
        mc = mpf.make_marketcolors(
            up=COLORS['bull_secondary'],
            down=COLORS['bear_secondary'],
            edge='inherit',
            wick='inherit',
            volume=COLORS['volume_up'],
        )

        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridcolor=COLORS['grid'],
            gridstyle='--',
            y_on_right=False,
            facecolor=COLORS['background_dark'],
            edgecolor=COLORS['grid'],
            figcolor=COLORS['background_dark'],
            rc={'axes.labelcolor': COLORS['text_primary'],
                'xtick.color': COLORS['text_primary'],
                'ytick.color': COLORS['text_primary']}
        )

        # Create plot
        kwargs = dict(
            type='candle',
            style=s,
            volume=True,
            title='ATLAS Regime Detection',
            ylabel='Price ($)',
            ylabel_lower='Volume',
            savefig=save_path if save_path else None,
            returnfig=True,
        )

        fig, axes = mpf.plot(ohlc_data, **kwargs)

        return save_path if save_path else "Figure created successfully"

    except ImportError:
        return "mplfinance not available - install with: pip install mplfinance"
    except Exception as e:
        return f"Error creating mplfinance chart: {str(e)}"
