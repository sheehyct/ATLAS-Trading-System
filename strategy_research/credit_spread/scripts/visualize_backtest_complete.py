"""
Credit Spread Strategy - Comprehensive VBT Pro Visualization
Feature-rich charting using VectorBT Pro best practices
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREDIT SPREAD STRATEGY - COMPREHENSIVE VBT PRO VISUALIZATION")
print("="*80)
print()

# Load backtest data
print("[STEP 1] Loading backtest data...")
signals = pd.read_csv('credit_spread_signals.csv', index_col=0, parse_dates=True)
trades_df = pd.read_csv('credit_spread_trades.csv')
performance_df = pd.read_csv('credit_spread_performance.csv', index_col=0)
print(f"  Loaded {len(signals)} signal events, {len(trades_df)} trades")
print()

# Download price data
print("[STEP 2] Downloading market data...")
sso = yf.download('SSO', start='2006-01-01', progress=False)
sso_close = sso['Close']
spread_data = pdr.DataReader('BAMLH0A0HYM2', 'fred', start='2003-01-01')
spread_data.columns = ['spread']
print(f"  SSO: {len(sso_close)} days")
print(f"  Credit Spreads: {len(spread_data)} days")
print()

# Prepare signals aligned to SSO index
print("[STEP 3] Aligning signals to price data...")
entries = pd.Series(False, index=sso_close.index)
exits = pd.Series(False, index=sso_close.index)

for date_str in signals[signals['entry']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    matching_dates = sso_close.index[sso_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        entries.loc[matching_dates[0]] = True

for date_str in signals[signals['exit']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    matching_dates = sso_close.index[sso_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        exits.loc[matching_dates[0]] = True

print(f"  Aligned: {entries.sum()} entries, {exits.sum()} exits")
print()

# Create VBT Portfolio
print("[STEP 4] Creating VectorBT Pro Portfolio...")
pf = vbt.Portfolio.from_signals(
    close=sso_close,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,  # 10 bps
    size=1.0,
    size_type='valuepercent',
    freq='1D'
)
print("  Portfolio created successfully")
print()

# Generate VBT Pro Visualizations
print("[STEP 5] Generating VectorBT Pro visualizations...")
print()

# Chart 1: Comprehensive Portfolio Dashboard
print("  [1/6] Portfolio Dashboard (all subplots)...")
try:
    fig1 = pf.plot(
        subplots=[
            'value',
            'trades',
            'trade_pnl',
            'drawdowns',
            'underwater'
        ]
    )
    fig1.update_layout(
        title='Credit Spread Strategy - Portfolio Performance Dashboard',
        height=1200
    )
    fig1.write_html('1_portfolio_dashboard.html')
    print("      [OK] Saved: 1_portfolio_dashboard.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Chart 2: Cumulative Returns with Benchmark
print("  [2/6] Cumulative Returns...")
try:
    # Download SPY as benchmark
    spy = yf.download('SPY', start='2006-01-01', progress=False)
    spy_close = spy['Close']

    fig2 = pf.plot_cumulative_returns(
        pct_scale=False,
    )
    fig2.update_layout(
        title='Credit Spread Strategy - Cumulative Returns',
        height=600
    )
    fig2.write_html('2_cumulative_returns.html')
    print("      [OK] Saved: 2_cumulative_returns.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Chart 3: Underwater Plot (Drawdown Recovery)
print("  [3/6] Underwater Drawdown Plot...")
try:
    fig3 = pf.plot_underwater(
        pct_scale=True
    )
    fig3.update_layout(
        title='Credit Spread Strategy - Underwater Plot (Drawdown Recovery)',
        height=500
    )
    fig3.write_html('3_underwater_drawdown.html')
    print("      [OK] Saved: 3_underwater_drawdown.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Chart 4: Drawdown Analysis
print("  [4/6] Drawdown Analysis...")
try:
    fig4 = pf.plot_drawdowns()
    fig4.update_layout(
        title='Credit Spread Strategy - Drawdown Analysis',
        height=600
    )
    fig4.write_html('4_drawdown_analysis.html')
    print("      [OK] Saved: 4_drawdown_analysis.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Chart 5: Trade Analysis
print("  [5/6] Trade Performance...")
try:
    fig5 = pf.plot_trade_pnl()
    fig5.update_layout(
        title='Credit Spread Strategy - Trade P&L Analysis',
        height=600
    )
    fig5.write_html('5_trade_pnl.html')
    print("      [OK] Saved: 5_trade_pnl.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Chart 6: Credit Spread Signals (Custom Plotly)
print("  [6/6] Credit Spread with Entry/Exit Signals...")
try:
    # Resample spread data to match signal dates
    spread_aligned = spread_data['spread'].reindex(signals.index, method='ffill')

    fig6 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Credit Spread with Signals', 'SSO Price')
    )

    # Top subplot: Credit spread with signals
    fig6.add_trace(go.Scatter(
        x=signals.index,
        y=spread_aligned,
        mode='lines',
        name='Credit Spread',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)

    fig6.add_trace(go.Scatter(
        x=signals.index,
        y=signals['ema_330'],
        mode='lines',
        name='330-day EMA',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ), row=1, col=1)

    # Entry markers
    entry_dates = signals[signals['entry']].index
    entry_values = spread_aligned[entry_dates]
    fig6.add_trace(go.Scatter(
        x=entry_dates,
        y=entry_values,
        mode='markers',
        name='ENTRY',
        marker=dict(color='green', size=15, symbol='triangle-up', line=dict(color='darkgreen', width=2))
    ), row=1, col=1)

    # Exit markers
    exit_dates = signals[signals['exit']].index
    exit_values = spread_aligned[exit_dates]
    fig6.add_trace(go.Scatter(
        x=exit_dates,
        y=exit_values,
        mode='markers',
        name='EXIT',
        marker=dict(color='red', size=15, symbol='triangle-down', line=dict(color='darkred', width=2))
    ), row=1, col=1)

    # Bottom subplot: SSO price - resample to daily for all available data
    sso_daily = sso_close.resample('D').last().dropna()
    fig6.add_trace(go.Scatter(
        x=sso_daily.index,
        y=sso_daily.values,
        mode='lines',
        name='SSO Price',
        line=dict(color='purple', width=2),
        showlegend=True
    ), row=2, col=1)

    # Mark entry/exit on SSO price chart - find nearest dates
    for entry_date in entry_dates:
        entry_date_naive = pd.to_datetime(entry_date).tz_localize(None)
        matching_sso = sso_close.index[sso_close.index.tz_localize(None).date == entry_date_naive.date()]
        if len(matching_sso) > 0:
            fig6.add_trace(go.Scatter(
                x=[matching_sso[0]],
                y=[sso_close.loc[matching_sso[0]]],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                showlegend=False
            ), row=2, col=1)

    for exit_date in exit_dates:
        exit_date_naive = pd.to_datetime(exit_date).tz_localize(None)
        matching_sso = sso_close.index[sso_close.index.tz_localize(None).date == exit_date_naive.date()]
        if len(matching_sso) > 0:
            fig6.add_trace(go.Scatter(
                x=[matching_sso[0]],
                y=[sso_close.loc[matching_sso[0]]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                showlegend=False
            ), row=2, col=1)

    fig6.update_xaxes(title_text="Date", row=2, col=1)
    fig6.update_yaxes(title_text="Spread (%)", row=1, col=1)
    fig6.update_yaxes(title_text="SSO Price ($)", row=2, col=1)

    fig6.update_layout(
        title='Credit Spread Strategy - Signal Analysis',
        hovermode='x unified',
        height=900,
        showlegend=True
    )
    fig6.write_html('6_credit_spread_signals.html')
    print("      [OK] Saved: 6_credit_spread_signals.html")
except Exception as e:
    print(f"      [ERROR] {e}")

# Print comprehensive stats
print()
print("[STEP 6] Performance Statistics...")
print()
stats = pf.stats()
print(stats.to_string())

print()
print("="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print()
print("Generated 6 interactive HTML visualizations:")
print("  1. 1_portfolio_dashboard.html     - Complete portfolio overview")
print("  2. 2_cumulative_returns.html      - Equity curve analysis")
print("  3. 3_underwater_drawdown.html     - Drawdown recovery visualization")
print("  4. 4_drawdown_analysis.html       - Individual drawdown periods")
print("  5. 5_trade_pnl.html               - Trade-by-trade P&L")
print("  6. 6_credit_spread_signals.html   - Credit spread with signals")
print()
print("Open any file in your browser for interactive charts.")
print()
