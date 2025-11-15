# ATLAS Plotly Dash Dashboard - Comprehensive Implementation Guide

## Repository Analysis

### System Overview
**ATLAS (Adaptive Trading with Layered Asset System)** is a multi-layer algorithmic trading architecture combining:

- **Layer 1**: Regime Detection using academic statistical jump models (Princeton University research)
- **Layer 2**: STRAT Pattern Recognition for bar-level entry/exit timing
- **Layer 3**: Execution layer supporting both equity and options strategies
- **Layer 4**: Credit Spread Monitoring (future development)

**Key Technologies**:
- VectorBT Pro (backtesting framework)
- NumPy/Pandas (data processing)
- Alpaca API (market data & execution)
- Scikit-learn (statistical models)

**Current Status**: Layer 1 (Regime Detection) validated, Layer 2 (STRAT) in design phase

---

## Dashboard Architecture

### Purpose
The ATLAS dashboard serves multiple critical functions:

1. **Real-time Monitoring**: Track live regime states, portfolio health, and active positions
2. **Historical Analysis**: Visualize backtesting results, performance metrics, and regime transitions
3. **Strategy Comparison**: Compare multiple strategies across different market conditions
4. **Risk Management**: Monitor portfolio heat, position sizing, and drawdown metrics
5. **Debugging & Validation**: Inspect regime classification accuracy and signal generation

### Technical Stack

```python
# Core Dependencies (already in pyproject.toml)
dash>=2.14.0
dash-bootstrap-components>=1.5.0
plotly>=6.2.0
pandas>=2.1.0
numpy>=1.26.0
```

---

## Dashboard Structure

### Recommended Layout

```
dashboard/
├── app.py                      # Main Dash application
├── config.py                   # Configuration and constants
├── callbacks/                  # Dash callback functions
│   ├── __init__.py
│   ├── regime_callbacks.py     # Layer 1 visualizations
│   ├── strategy_callbacks.py   # Strategy performance callbacks
│   ├── portfolio_callbacks.py  # Portfolio monitoring
│   └── risk_callbacks.py       # Risk management callbacks
├── components/                 # Reusable UI components
│   ├── __init__.py
│   ├── header.py              # Top navigation/status bar
│   ├── regime_panel.py        # Regime detection visualizations
│   ├── strategy_panel.py      # Strategy performance panel
│   ├── portfolio_panel.py     # Portfolio overview
│   ├── risk_panel.py          # Risk metrics panel
│   └── strat_panel.py         # STRAT pattern visualizations
├── data_loaders/              # Data integration layer
│   ├── __init__.py
│   ├── regime_loader.py       # Load regime detection data
│   ├── backtest_loader.py     # Load backtesting results
│   ├── live_loader.py         # Real-time data from Alpaca
│   └── cache_manager.py       # Data caching for performance
├── visualizations/            # Plotly figure generators
│   ├── __init__.py
│   ├── regime_viz.py          # Regime timeline, feature plots
│   ├── performance_viz.py     # Equity curves, drawdowns
│   ├── trade_viz.py           # Trade analysis charts
│   ├── risk_viz.py            # Risk heatmaps, correlations
│   └── pattern_viz.py         # STRAT pattern visualizations
└── assets/                    # Static assets
    ├── custom.css             # Custom styling
    └── atlas_logo.png         # Branding assets
```

---

## Core Visualizations

### 1. Regime Detection Dashboard (Layer 1)

#### 1.1 Regime Timeline
**Purpose**: Visualize market regime changes over time

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_regime_timeline(dates, regimes, prices):
    """
    Create stacked visualization showing:
    - Top: Price chart with regime background colors
    - Bottom: Regime state indicator

    Args:
        dates: pd.DatetimeIndex
        regimes: pd.Series with regime labels (TREND_BULL, TREND_BEAR, etc.)
        prices: pd.Series with price data

    Returns:
        plotly.graph_objects.Figure
    """

    # Define regime colors
    REGIME_COLORS = {
        'TREND_BULL': 'rgba(0, 255, 0, 0.2)',
        'TREND_NEUTRAL': 'rgba(128, 128, 128, 0.2)',
        'TREND_BEAR': 'rgba(255, 165, 0, 0.2)',
        'CRASH': 'rgba(255, 0, 0, 0.3)'
    }

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price with Regime Overlay', 'Regime State'),
        vertical_spacing=0.1
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )

    # Add regime background shading
    current_regime = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
            # Add shaded region
            fig.add_vrect(
                x0=dates[start_idx],
                x1=dates[i],
                fillcolor=REGIME_COLORS[current_regime],
                layer='below',
                line_width=0,
                row=1, col=1
            )

            current_regime = regimes.iloc[i]
            start_idx = i

    # Regime state indicator (categorical)
    regime_numeric = regimes.map({
        'TREND_BULL': 3,
        'TREND_NEUTRAL': 2,
        'TREND_BEAR': 1,
        'CRASH': 0
    })

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=regime_numeric,
            mode='lines',
            name='Regime',
            fill='tozeroy',
            line=dict(width=0),
            fillcolor='rgba(100, 100, 100, 0.3)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(
        title_text="Regime",
        ticktext=['CRASH', 'BEAR', 'NEUTRAL', 'BULL'],
        tickvals=[0, 1, 2, 3],
        row=2, col=1
    )

    return fig
```

#### 1.2 Feature Evolution
**Purpose**: Track regime detection features (Downside Deviation, Sortino Ratio)

```python
def create_feature_dashboard(dates, downside_dev, sortino_20d, sortino_60d):
    """
    Visualize regime detection features over time

    Features tracked:
    - Downside Deviation (10-day EWMA)
    - Sortino Ratio 20-day
    - Sortino Ratio 60-day
    """

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Downside Deviation (10d EWMA)', 'Sortino Ratios'),
        vertical_spacing=0.15
    )

    # Downside Deviation
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=downside_dev,
            mode='lines',
            name='Downside Dev',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Add threshold line (example: 0.02 as CRASH threshold)
    fig.add_hline(
        y=0.02,
        line_dash="dash",
        line_color="red",
        annotation_text="CRASH Threshold",
        row=1, col=1
    )

    # Sortino Ratios
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sortino_20d,
            mode='lines',
            name='Sortino 20d',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sortino_60d,
            mode='lines',
            name='Sortino 60d',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )

    # Add threshold lines for regime classification
    fig.add_hline(
        y=-0.5,
        line_dash="dash",
        line_color="orange",
        annotation_text="BEAR Threshold",
        row=2, col=1
    )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="green",
        annotation_text="BULL Threshold",
        row=2, col=1
    )

    fig.update_layout(height=700, template='plotly_white')
    fig.update_yaxes(title_text="Std Dev", row=1, col=1)
    fig.update_yaxes(title_text="Sortino Ratio", row=2, col=1)

    return fig
```

#### 1.3 Regime Statistics Table
**Purpose**: Summary statistics for each regime

```python
import dash_bootstrap_components as dbc
from dash import html, dash_table

def create_regime_statistics_table(regime_data):
    """
    Create table showing:
    - Regime name
    - Days in regime
    - Average return
    - Volatility
    - Max drawdown
    - Sharpe ratio
    """

    # Calculate statistics per regime
    stats = []
    for regime in ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']:
        regime_mask = regime_data['regime'] == regime
        regime_returns = regime_data.loc[regime_mask, 'returns']

        stats.append({
            'Regime': regime,
            'Days': regime_mask.sum(),
            'Avg Return': f"{regime_returns.mean():.2%}",
            'Volatility': f"{regime_returns.std():.2%}",
            'Max DD': f"{(regime_returns.min()):.2%}",
            'Sharpe': f"{(regime_returns.mean() / regime_returns.std() * np.sqrt(252)):.2f}"
        })

    return dash_table.DataTable(
        data=stats,
        columns=[{'name': i, 'id': i} for i in stats[0].keys()],
        style_data_conditional=[
            {
                'if': {'filter_query': '{Regime} = "CRASH"'},
                'backgroundColor': 'rgba(255, 0, 0, 0.2)'
            },
            {
                'if': {'filter_query': '{Regime} = "TREND_BULL"'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
```

---

### 2. Strategy Performance Dashboard

#### 2.1 Equity Curve with Drawdown
**Purpose**: Visualize cumulative returns and underwater equity

```python
def create_equity_curve(portfolio_value, benchmark_value=None):
    """
    Create equity curve with drawdown subplot

    Args:
        portfolio_value: pd.Series with portfolio value over time
        benchmark_value: Optional benchmark (e.g., SPY buy-and-hold)
    """

    # Calculate drawdown
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Portfolio equity curve
    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Benchmark (if provided)
    if benchmark_value is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_value.index,
                y=benchmark_value,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.2)'
        ),
        row=2, col=1
    )

    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()

    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_val * 100,
        text=f"Max DD: {max_dd_val:.2%}",
        showarrow=True,
        arrowhead=2,
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig
```

#### 2.2 Rolling Performance Metrics
**Purpose**: Track rolling Sharpe, Sortino, Win Rate

```python
def create_rolling_metrics(returns, window=60):
    """
    Calculate and visualize rolling performance metrics

    Args:
        returns: pd.Series of daily returns
        window: Rolling window size (default 60 days)
    """

    # Calculate rolling metrics
    rolling_sharpe = (
        returns.rolling(window).mean() /
        returns.rolling(window).std() *
        np.sqrt(252)
    )

    # Rolling Sortino (downside deviation)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    rolling_sortino = (
        returns.rolling(window).mean() /
        downside_returns.rolling(window).std() *
        np.sqrt(252)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=rolling_sharpe,
            mode='lines',
            name=f'Rolling Sharpe ({window}d)',
            line=dict(color='blue', width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=rolling_sortino,
            mode='lines',
            name=f'Rolling Sortino ({window}d)',
            line=dict(color='green', width=2)
        )
    )

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Sharpe = 1.0 (Good)")
    fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="Sharpe = 1.5 (Excellent)")

    fig.update_layout(
        title=f'Rolling Performance Metrics ({window}-Day Window)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Ratio")
    fig.update_xaxes(title_text="Date")

    return fig
```

#### 2.3 Strategy Comparison by Regime
**Purpose**: Compare how different strategies perform in each regime

```python
def create_strategy_regime_comparison(strategies_data):
    """
    Create grouped bar chart comparing strategies across regimes

    Args:
        strategies_data: Dict with structure:
        {
            'Strategy A': {'TREND_BULL': 0.15, 'TREND_BEAR': -0.05, ...},
            'Strategy B': {'TREND_BULL': 0.12, 'TREND_BEAR': 0.02, ...}
        }
    """

    regimes = ['TREND_BULL', 'TREND_NEUTRAL', 'TREND_BEAR', 'CRASH']

    fig = go.Figure()

    for strategy_name, regime_returns in strategies_data.items():
        fig.add_trace(
            go.Bar(
                x=regimes,
                y=[regime_returns[r] * 100 for r in regimes],  # Convert to %
                name=strategy_name
            )
        )

    fig.update_layout(
        title='Strategy Performance by Market Regime',
        barmode='group',
        height=500,
        template='plotly_white',
        yaxis_title='Average Return (%)',
        xaxis_title='Market Regime'
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    return fig
```

---

### 3. Trade Analysis Dashboard

#### 3.1 Trade Distribution
**Purpose**: Visualize P&L distribution and win/loss patterns

```python
def create_trade_distribution(trades_df):
    """
    Create histogram of trade returns

    Args:
        trades_df: DataFrame with columns ['pnl', 'return_pct', 'duration']
    """

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P&L Distribution', 'Return % Distribution')
    )

    # P&L histogram
    fig.add_trace(
        go.Histogram(
            x=trades_df['pnl'],
            nbinsx=30,
            name='P&L',
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )

    # Return % histogram
    fig.add_trace(
        go.Histogram(
            x=trades_df['return_pct'] * 100,
            nbinsx=30,
            name='Return %',
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=2
    )

    # Add mean lines
    fig.add_vline(
        x=trades_df['pnl'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${trades_df['pnl'].mean():.2f}",
        row=1, col=1
    )

    fig.add_vline(
        x=trades_df['return_pct'].mean() * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {trades_df['return_pct'].mean():.2%}",
        row=1, col=2
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)

    return fig
```

#### 3.2 Trade Timeline
**Purpose**: Visualize entry/exit points on price chart

```python
def create_trade_timeline(price_data, trades_df):
    """
    Show trades overlaid on price chart

    Args:
        price_data: pd.Series with price data
        trades_df: DataFrame with columns ['entry_date', 'exit_date', 'entry_price',
                                           'exit_price', 'direction', 'pnl']
    """

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        )
    )

    # Add entry points
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    # Winning entries (green)
    fig.add_trace(
        go.Scatter(
            x=winning_trades['entry_date'],
            y=winning_trades['entry_price'],
            mode='markers',
            name='Winning Entry',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
                line=dict(color='darkgreen', width=2)
            )
        )
    )

    # Losing entries (red)
    fig.add_trace(
        go.Scatter(
            x=losing_trades['entry_date'],
            y=losing_trades['entry_price'],
            mode='markers',
            name='Losing Entry',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red',
                line=dict(color='darkred', width=2)
            )
        )
    )

    # Draw lines from entry to exit for each trade
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['pnl'] > 0 else 'red'
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_date'], trade['exit_date']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                showlegend=False,
                hovertemplate=f"P&L: ${trade['pnl']:.2f}<extra></extra>"
            )
        )

    fig.update_layout(
        title='Trade Timeline',
        height=600,
        template='plotly_white',
        hovermode='closest'
    )

    fig.update_yaxes(title_text="Price ($)")
    fig.update_xaxes(title_text="Date")

    return fig
```

---

### 4. Risk Management Dashboard

#### 4.1 Portfolio Heat Map
**Purpose**: Monitor real-time portfolio exposure

```python
def create_portfolio_heat_gauge(current_heat, max_heat=0.08):
    """
    Create gauge showing current portfolio heat vs limit

    Args:
        current_heat: Current portfolio heat (0-1 scale)
        max_heat: Maximum allowed heat (default 8%)
    """

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_heat * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio Heat", 'font': {'size': 24}},
        delta={'reference': max_heat * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, max_heat * 100], 'ticksuffix': "%"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_heat * 0.5 * 100], 'color': "lightgreen"},
                {'range': [max_heat * 0.5 * 100, max_heat * 0.75 * 100], 'color': "yellow"},
                {'range': [max_heat * 0.75 * 100, max_heat * 100], 'color': "orange"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_heat * 100
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'size': 18}
    )

    return fig
```

#### 4.2 Position Sizing Breakdown
**Purpose**: Show how capital is allocated across positions

```python
def create_position_allocation_chart(positions_df):
    """
    Create sunburst chart showing position allocation

    Args:
        positions_df: DataFrame with columns ['symbol', 'strategy', 'size', 'risk']
    """

    # Prepare hierarchical data
    fig = go.Figure(go.Sunburst(
        labels=positions_df['symbol'].tolist() + positions_df['strategy'].unique().tolist() + ['Portfolio'],
        parents=(
            positions_df['strategy'].tolist() +
            ['Portfolio'] * len(positions_df['strategy'].unique()) +
            ['']
        ),
        values=positions_df['size'].tolist() + [0] * (len(positions_df['strategy'].unique()) + 1),
        branchvalues="total"
    ))

    fig.update_layout(
        title='Position Allocation by Strategy',
        height=600
    )

    return fig
```

#### 4.3 Risk Metrics Table
**Purpose**: Display key risk statistics

```python
def create_risk_metrics_card():
    """
    Create dashboard card with key risk metrics
    """

    metrics = [
        {'Metric': 'Portfolio Heat', 'Current': '4.2%', 'Limit': '8.0%', 'Status': '✓'},
        {'Metric': 'Max Position Size', 'Current': '2.1%', 'Limit': '5.0%', 'Status': '✓'},
        {'Metric': 'Daily Drawdown', 'Current': '-0.8%', 'Limit': '-3.0%', 'Status': '✓'},
        {'Metric': 'VaR (95%)', 'Current': '-2.1%', 'Limit': '-5.0%', 'Status': '✓'},
        {'Metric': 'Open Positions', 'Current': '3', 'Limit': '5', 'Status': '✓'}
    ]

    return dash_table.DataTable(
        data=metrics,
        columns=[{'name': i, 'id': i} for i in metrics[0].keys()],
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Status} = "✗"',
                    'column_id': 'Status'
                },
                'backgroundColor': 'rgba(255, 0, 0, 0.3)',
                'color': 'red'
            },
            {
                'if': {
                    'filter_query': '{Status} = "✓"',
                    'column_id': 'Status'
                },
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'color': 'green'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_cell={'textAlign': 'center'}
    )
```

---

### 5. STRAT Pattern Visualization (Layer 2)

#### 5.1 Bar Classification Chart
**Purpose**: Visualize STRAT bar types and patterns

```python
def create_strat_pattern_chart(candle_data, bar_types, patterns):
    """
    Create candlestick chart with STRAT bar classification overlay

    Args:
        candle_data: DataFrame with OHLC data
        bar_types: Series with bar type classification (1, 2U, 2D, 3)
        patterns: DataFrame with detected patterns (312, 212, 22, etc.)
    """

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=candle_data.index,
            open=candle_data['open'],
            high=candle_data['high'],
            low=candle_data['low'],
            close=candle_data['close'],
            name='Price'
        )
    )

    # Color code by bar type
    BAR_TYPE_COLORS = {
        '1': 'rgba(128, 128, 128, 0.3)',  # Inside - Gray
        '2U': 'rgba(0, 255, 0, 0.3)',      # Up - Green
        '2D': 'rgba(255, 0, 0, 0.3)',      # Down - Red
        '3': 'rgba(255, 255, 0, 0.3)'      # Outside - Yellow
    }

    # Add background shading by bar type
    for i, bar_type in enumerate(bar_types):
        if i < len(candle_data) - 1:
            fig.add_vrect(
                x0=candle_data.index[i],
                x1=candle_data.index[i+1],
                fillcolor=BAR_TYPE_COLORS.get(bar_type, 'rgba(255,255,255,0)'),
                layer='below',
                line_width=0
            )

    # Highlight patterns
    for _, pattern in patterns.iterrows():
        fig.add_annotation(
            x=pattern['date'],
            y=pattern['price'],
            text=pattern['type'],  # e.g., "3-1-2"
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="blue",
            bgcolor="yellow",
            opacity=0.8
        )

    fig.update_layout(
        title='STRAT Pattern Detection',
        height=700,
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )

    return fig
```

#### 5.2 Multi-Timeframe Alignment (4 C's)
**Purpose**: Show timeframe continuity analysis

```python
def create_timeframe_alignment_chart(dates, tf_states):
    """
    Visualize multi-timeframe alignment (Control, Confirm, Conflict, Change)

    Args:
        dates: DatetimeIndex
        tf_states: DataFrame with columns ['daily', 'hourly', '15min', 'alignment']
    """

    # Map alignment to numeric values
    alignment_map = {
        'Control': 4,
        'Confirm': 3,
        'Conflict': 2,
        'Change': 1
    }

    fig = go.Figure()

    # Heatmap showing alignment state
    alignment_numeric = tf_states['alignment'].map(alignment_map)

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=alignment_numeric,
            mode='lines',
            fill='tozeroy',
            name='Alignment State',
            line=dict(width=0),
            fillcolor='rgba(0, 100, 200, 0.3)'
        )
    )

    fig.update_layout(
        title='Multi-Timeframe Alignment (4 C\'s)',
        height=400,
        template='plotly_white',
        yaxis=dict(
            ticktext=['Change', 'Conflict', 'Confirm', 'Control'],
            tickvals=[1, 2, 3, 4]
        )
    )

    return fig
```

---

## Data Integration Layer

### 1. Regime Detection Data Loader

```python
# data_loaders/regime_loader.py

import pandas as pd
import numpy as np
from regime.academic_jump_model import AcademicJumpModel
from regime.academic_features import calculate_features

class RegimeDataLoader:
    """
    Load and format regime detection data for dashboard
    """

    def __init__(self, model_path=None):
        """
        Args:
            model_path: Path to saved regime model (optional)
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, path):
        """Load pre-trained regime model"""
        # Implementation depends on model serialization format
        pass

    def get_regime_timeline(self, start_date, end_date):
        """
        Get regime classification for date range

        Returns:
            DataFrame with columns: ['date', 'regime', 'confidence']
        """
        # Load from model or database
        pass

    def get_regime_features(self, start_date, end_date):
        """
        Get feature values used for regime detection

        Returns:
            DataFrame with columns: ['date', 'downside_dev', 'sortino_20d', 'sortino_60d']
        """
        pass

    def get_regime_statistics(self):
        """
        Calculate aggregate statistics per regime

        Returns:
            DataFrame with regime-level summary stats
        """
        pass
```

### 2. Backtest Results Loader

```python
# data_loaders/backtest_loader.py

import vectorbtpro as vbt
import pandas as pd

class BacktestDataLoader:
    """
    Load VectorBT Pro backtest results for visualization
    """

    def __init__(self, results_path=None):
        self.results_path = results_path
        self.portfolio = None

    def load_backtest(self, strategy_name):
        """
        Load backtest results for specific strategy

        Returns:
            vbt.Portfolio object
        """
        # Load serialized VBT portfolio
        pass

    def get_equity_curve(self):
        """Extract equity curve from portfolio"""
        if self.portfolio:
            return self.portfolio.value()
        return pd.Series()

    def get_trades(self):
        """
        Get individual trade records

        Returns:
            DataFrame with columns: ['entry_date', 'exit_date', 'pnl', 'return', 'duration']
        """
        if self.portfolio:
            trades = self.portfolio.trades.records_readable
            return trades
        return pd.DataFrame()

    def get_performance_metrics(self):
        """
        Calculate key performance metrics

        Returns:
            Dict with Sharpe, Sortino, Max DD, Win Rate, etc.
        """
        if self.portfolio:
            return {
                'total_return': self.portfolio.total_return(),
                'sharpe_ratio': self.portfolio.sharpe_ratio(),
                'sortino_ratio': self.portfolio.sortino_ratio(),
                'max_drawdown': self.portfolio.max_drawdown(),
                'win_rate': self.portfolio.trades.win_rate(),
                'profit_factor': self.portfolio.trades.profit_factor(),
                'total_trades': self.portfolio.trades.count()
            }
        return {}
```

### 3. Live Data Loader (Alpaca Integration)

```python
# data_loaders/live_loader.py

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
import os

class LiveDataLoader:
    """
    Fetch real-time data from Alpaca for live monitoring
    """

    def __init__(self):
        self.data_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )

        self.trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True  # Use paper trading
        )

    def get_current_positions(self):
        """
        Get current open positions

        Returns:
            DataFrame with position details
        """
        positions = self.trading_client.get_all_positions()
        return pd.DataFrame([
            {
                'symbol': p.symbol,
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            }
            for p in positions
        ])

    def get_account_status(self):
        """
        Get account equity, buying power, etc.

        Returns:
            Dict with account metrics
        """
        account = self.trading_client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'last_equity': float(account.last_equity),
            'daytrade_count': int(account.daytrade_count)
        }

    def get_latest_price(self, symbol):
        """Get latest price for symbol"""
        # Implementation using Alpaca data API
        pass
```

---

## Main Dashboard Application

### app.py Structure

```python
# dashboard/app.py

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd

# Import components
from components.header import create_header
from components.regime_panel import create_regime_panel
from components.strategy_panel import create_strategy_panel
from components.portfolio_panel import create_portfolio_panel
from components.risk_panel import create_risk_panel

# Import data loaders
from data_loaders.regime_loader import RegimeDataLoader
from data_loaders.backtest_loader import BacktestDataLoader
from data_loaders.live_loader import LiveDataLoader

# Import visualizations
from visualizations.regime_viz import create_regime_timeline, create_feature_dashboard
from visualizations.performance_viz import create_equity_curve, create_rolling_metrics
from visualizations.trade_viz import create_trade_distribution, create_trade_timeline
from visualizations.risk_viz import create_portfolio_heat_gauge, create_risk_metrics_card

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Initialize data loaders
regime_loader = RegimeDataLoader()
backtest_loader = BacktestDataLoader()
live_loader = LiveDataLoader()

# App layout
app.layout = dbc.Container([

    # Header
    create_header(),

    # Date range selector
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=(datetime.now() - timedelta(days=365)).date(),
                end_date=datetime.now().date(),
                display_format='YYYY-MM-DD'
            )
        ], width=6),

        dbc.Col([
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ], width=6)
    ], className='mb-4'),

    # Tab navigation
    dbc.Tabs([

        # Tab 1: Regime Detection
        dbc.Tab(label='Regime Detection', tab_id='regime-tab', children=[
            create_regime_panel()
        ]),

        # Tab 2: Strategy Performance
        dbc.Tab(label='Strategy Performance', tab_id='strategy-tab', children=[
            create_strategy_panel()
        ]),

        # Tab 3: Live Portfolio
        dbc.Tab(label='Live Portfolio', tab_id='portfolio-tab', children=[
            create_portfolio_panel()
        ]),

        # Tab 4: Risk Management
        dbc.Tab(label='Risk Management', tab_id='risk-tab', children=[
            create_risk_panel()
        ]),

    ], id='tabs', active_tab='regime-tab'),

], fluid=True)


# ============================================
# CALLBACKS
# ============================================

# Regime Timeline Update
@app.callback(
    Output('regime-timeline-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_regime_timeline(start_date, end_date):
    """Update regime detection timeline visualization"""

    # Load regime data
    regime_data = regime_loader.get_regime_timeline(start_date, end_date)

    # Load price data (example: SPY)
    # In production, load from your data source
    dates = regime_data['date']
    regimes = regime_data['regime']
    prices = regime_data['price']  # Assume price column exists

    return create_regime_timeline(dates, regimes, prices)


# Feature Dashboard Update
@app.callback(
    Output('feature-dashboard-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_feature_dashboard(start_date, end_date):
    """Update regime feature visualization"""

    features_data = regime_loader.get_regime_features(start_date, end_date)

    return create_feature_dashboard(
        dates=features_data['date'],
        downside_dev=features_data['downside_dev'],
        sortino_20d=features_data['sortino_20d'],
        sortino_60d=features_data['sortino_60d']
    )


# Equity Curve Update
@app.callback(
    Output('equity-curve-graph', 'figure'),
    [Input('strategy-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_equity_curve(strategy_name, start_date, end_date):
    """Update strategy equity curve"""

    # Load backtest for selected strategy
    backtest_loader.load_backtest(strategy_name)

    portfolio_value = backtest_loader.get_equity_curve()

    # Optionally load benchmark
    # benchmark_value = load_benchmark_data(start_date, end_date)

    return create_equity_curve(portfolio_value)


# Live Portfolio Update (auto-refresh every 30 seconds)
@app.callback(
    [Output('portfolio-value-card', 'children'),
     Output('positions-table', 'data'),
     Output('portfolio-heat-gauge', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_live_portfolio(n):
    """Update live portfolio metrics"""

    # Get account status
    account = live_loader.get_account_status()

    # Get current positions
    positions = live_loader.get_current_positions()

    # Calculate portfolio heat
    # (This would use utils/portfolio_heat.py logic)
    current_heat = 0.042  # Example value

    # Create portfolio value card
    portfolio_card = dbc.Card([
        dbc.CardBody([
            html.H4(f"${account['portfolio_value']:,.2f}", className='text-primary'),
            html.P(f"P&L Today: ${account['equity'] - account['last_equity']:+,.2f}",
                   className='text-success' if account['equity'] > account['last_equity'] else 'text-danger')
        ])
    ])

    # Format positions for table
    positions_data = positions.to_dict('records')

    # Create heat gauge
    heat_gauge = create_portfolio_heat_gauge(current_heat)

    return portfolio_card, positions_data, heat_gauge


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

---

## Component Modules

### Header Component

```python
# components/header.py

import dash_bootstrap_components as dbc
from dash import html

def create_header():
    """Create dashboard header with title and status indicators"""

    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(src='/assets/atlas_logo.png', height='40px'),
                    dbc.NavbarBrand("ATLAS Trading Dashboard", className="ms-2"),
                ], width='auto'),

                dbc.Col([
                    # Status indicators
                    html.Div([
                        html.Span("Live", className="badge bg-success me-2"),
                        html.Span("Connected", className="badge bg-info me-2"),
                        html.Span(id='current-regime-badge', className="badge bg-warning")
                    ])
                ], width='auto')
            ], align='center', className='g-0 w-100')
        ], fluid=True),
        color='dark',
        dark=True,
        className='mb-4'
    )
```

### Regime Panel Component

```python
# components/regime_panel.py

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_regime_panel():
    """Create regime detection visualization panel"""

    return dbc.Container([

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Regime Timeline"),
                    dbc.CardBody([
                        dcc.Graph(id='regime-timeline-graph')
                    ])
                ])
            ], width=12)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Evolution"),
                    dbc.CardBody([
                        dcc.Graph(id='feature-dashboard-graph')
                    ])
                ])
            ], width=8),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Regime Statistics"),
                    dbc.CardBody([
                        html.Div(id='regime-stats-table')
                    ])
                ])
            ], width=4)
        ])

    ], fluid=True)
```

### Strategy Panel Component

```python
# components/strategy_panel.py

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_strategy_panel():
    """Create strategy performance visualization panel"""

    return dbc.Container([

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='strategy-selector',
                    options=[
                        {'label': 'Opening Range Breakout', 'value': 'orb'},
                        {'label': '52-Week High Momentum', 'value': '52w_high'},
                        {'label': 'Portfolio (All Strategies)', 'value': 'portfolio'}
                    ],
                    value='orb',
                    clearable=False
                )
            ], width=4)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Equity Curve & Drawdown"),
                    dbc.CardBody([
                        dcc.Graph(id='equity-curve-graph')
                    ])
                ])
            ], width=12)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rolling Performance Metrics"),
                    dbc.CardBody([
                        dcc.Graph(id='rolling-metrics-graph')
                    ])
                ])
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance by Regime"),
                    dbc.CardBody([
                        dcc.Graph(id='regime-comparison-graph')
                    ])
                ])
            ], width=6)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trade Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id='trade-distribution-graph')
                    ])
                ])
            ], width=12)
        ])

    ], fluid=True)
```

---

## Configuration File

```python
# dashboard/config.py

"""
Dashboard configuration and constants
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
ASSETS_DIR = BASE_DIR / 'dashboard' / 'assets'

# Alpaca Configuration
ALPACA_CONFIG = {
    'api_key': os.getenv('ALPACA_API_KEY'),
    'secret_key': os.getenv('ALPACA_SECRET_KEY'),
    'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
}

# Dashboard Settings
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 8050,
    'debug': True,
    'refresh_interval': 30000,  # 30 seconds
}

# Regime Colors
REGIME_COLORS = {
    'TREND_BULL': 'rgba(0, 255, 0, 0.2)',
    'TREND_NEUTRAL': 'rgba(128, 128, 128, 0.2)',
    'TREND_BEAR': 'rgba(255, 165, 0, 0.2)',
    'CRASH': 'rgba(255, 0, 0, 0.3)'
}

# STRAT Bar Type Colors
STRAT_COLORS = {
    '1': 'rgba(128, 128, 128, 0.3)',   # Inside
    '2U': 'rgba(0, 255, 0, 0.3)',       # Directional Up
    '2D': 'rgba(255, 0, 0, 0.3)',       # Directional Down
    '3': 'rgba(255, 255, 0, 0.3)'       # Outside
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'sharpe_good': 1.0,
    'sharpe_excellent': 1.5,
    'max_drawdown_limit': 0.25,
    'portfolio_heat_limit': 0.08,
    'position_size_limit': 0.05
}

# Strategies
AVAILABLE_STRATEGIES = {
    'orb': {
        'name': 'Opening Range Breakout',
        'file': 'strategies/orb.py',
        'class': 'ORBStrategy'
    },
    '52w_high': {
        'name': '52-Week High Momentum',
        'file': 'strategies/high_momentum_52w.py',
        'class': 'HighMomentum52W'
    }
}
```

---

## Implementation Checklist for Cursor

### Phase 1: Project Setup (Day 1)
- [ ] Create `dashboard/` directory structure
- [ ] Set up `app.py` with basic layout
- [ ] Configure `config.py` with environment variables
- [ ] Create basic header component
- [ ] Test Dash server launches successfully

### Phase 2: Data Integration (Days 2-3)
- [ ] Implement `RegimeDataLoader` class
- [ ] Implement `BacktestDataLoader` class
- [ ] Implement `LiveDataLoader` class (Alpaca integration)
- [ ] Add data caching mechanism
- [ ] Test data loading from all sources

### Phase 3: Regime Visualizations (Days 4-5)
- [ ] Create regime timeline chart
- [ ] Create feature evolution dashboard
- [ ] Create regime statistics table
- [ ] Implement regime panel component
- [ ] Add callbacks for regime tab

### Phase 4: Strategy Performance (Days 6-7)
- [ ] Create equity curve visualization
- [ ] Create rolling metrics chart
- [ ] Create trade distribution plots
- [ ] Create strategy comparison chart
- [ ] Implement strategy panel component
- [ ] Add callbacks for strategy tab

### Phase 5: Live Portfolio Monitoring (Days 8-9)
- [ ] Create portfolio value cards
- [ ] Create positions table
- [ ] Implement portfolio heat gauge
- [ ] Add real-time data updates
- [ ] Implement portfolio panel component
- [ ] Add auto-refresh callbacks

### Phase 6: Risk Management (Day 10)
- [ ] Create risk metrics dashboard
- [ ] Create position allocation charts
- [ ] Implement risk panel component
- [ ] Add risk limit indicators

### Phase 7: STRAT Visualizations (Days 11-12)
- [ ] Create bar classification chart
- [ ] Create pattern detection overlay
- [ ] Create multi-timeframe alignment chart
- [ ] Implement STRAT panel component

### Phase 8: Polish & Testing (Days 13-14)
- [ ] Add custom CSS styling
- [ ] Implement responsive design
- [ ] Add error handling
- [ ] Performance optimization
- [ ] User testing and refinement

---

## Usage Instructions

### Running the Dashboard

```bash
# Navigate to project directory
cd /home/user/ATLAS-Algorithmic-Trading-System-V1

# Activate environment (if using UV)
uv sync

# Set environment variables
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# Run dashboard
uv run python dashboard/app.py

# Access at http://localhost:8050
```

### Development Workflow in Cursor

1. **Start with data loaders**: Get data flowing first before building visualizations
2. **Build incrementally**: Complete one tab at a time, test thoroughly
3. **Use hot reload**: Dash supports hot reload - save files to see changes immediately
4. **Test with real data**: Use actual backtest results and Alpaca data for realistic testing
5. **Parallel development**: Multiple developers can work on different tabs simultaneously

---

## Advanced Features (Future Enhancements)

### 1. WebSocket Live Updates
Replace polling with WebSocket for real-time price and position updates

### 2. Alert System
Add configurable alerts for:
- Regime changes
- Portfolio heat threshold breaches
- Drawdown limits
- Trade signals

### 3. Strategy Optimization Interface
Interactive parameter tuning with live backtest updates

### 4. Paper Trading Control Panel
Submit orders, modify positions directly from dashboard

### 5. Machine Learning Insights
- Regime prediction confidence
- Signal quality scoring
- Anomaly detection

---

## Performance Optimization Tips

1. **Use Dash Bootstrap Components**: Faster than pure HTML/CSS
2. **Implement data caching**: Use `@lru_cache` or Redis for expensive calculations
3. **Lazy loading**: Only load data for active tab
4. **Downsample large datasets**: For charts with >10k points, use datashader
5. **Background callbacks**: Use `dash.long_callback` for slow operations
6. **Clientside callbacks**: Move simple updates to JavaScript

---

## Security Considerations

1. **Authentication**: Add login system for production deployment
2. **API key protection**: Never expose Alpaca keys in frontend code
3. **Rate limiting**: Implement rate limits for Alpaca API calls
4. **HTTPS**: Use SSL/TLS for production deployment
5. **Access control**: Restrict dashboard access to authorized users only

---

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Ensure `dashboard/` is in Python path
2. **Empty visualizations**: Check data loader return values
3. **Slow performance**: Enable caching, reduce update frequency
4. **Callback errors**: Check Input/Output IDs match layout
5. **Alpaca connection issues**: Verify API credentials and network access

---

## Resources

- **Plotly Dash Documentation**: https://dash.plotly.com/
- **Dash Bootstrap Components**: https://dash-bootstrap-components.opensource.faculty.ai/
- **VectorBT Pro Docs**: (in VectorBT Pro Official Documentation/)
- **Alpaca API Docs**: https://alpaca.markets/docs/

---

**Version**: 1.0
**Last Updated**: November 2025
**Status**: Ready for Implementation in Cursor
