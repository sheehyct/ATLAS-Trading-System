"""
Credit Spread Strategy - Visualization Script
Generates charts for backtest results using VectorBT Pro
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREDIT SPREAD STRATEGY - VISUALIZATION")
print("="*80)
print()

# Load data
print("[STEP 1] Loading backtest data...")
signals = pd.read_csv('credit_spread_signals.csv', index_col=0, parse_dates=True)
trades = pd.read_csv('credit_spread_trades.csv')
performance = pd.read_csv('credit_spread_performance.csv', index_col=0)

print(f"  Loaded {len(signals)} signal events")
print(f"  Loaded {len(trades)} trades")
print()

# Download SSO price data
print("[STEP 2] Downloading SSO price data...")
sso = yf.download('SSO', start='2006-01-01', progress=False)
sso_close = sso['Close']
print(f"  Downloaded {len(sso_close)} days of SSO data")
print()

# Download credit spread data
print("[STEP 3] Downloading credit spread data...")
spread_data = pdr.DataReader('BAMLH0A0HYM2', 'fred', start='2003-01-01')
spread_data.columns = ['spread']
print(f"  Downloaded {len(spread_data)} days of credit spread data")
print()

# Prepare signals
print("[STEP 4] Preparing signals for portfolio...")
# Align signals to SSO index
entries = pd.Series(False, index=sso_close.index)
exits = pd.Series(False, index=sso_close.index)

# Convert signal dates to timezone-naive for matching
for date_str in signals[signals['entry']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    # Find matching date in SSO index
    matching_dates = sso_close.index[sso_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        entries.loc[matching_dates[0]] = True

for date_str in signals[signals['exit']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    # Find matching date in SSO index
    matching_dates = sso_close.index[sso_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        exits.loc[matching_dates[0]] = True

print(f"  Entries: {entries.sum()}")
print(f"  Exits: {exits.sum()}")
print()

# Create portfolio
print("[STEP 5] Creating portfolio object...")
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

# Generate plots
print("[STEP 6] Generating visualizations...")

# Plot 1: Portfolio performance (equity curve + trades)
print("  [1/4] Creating portfolio performance chart...")
try:
    fig1 = pf.plot(
        subplots=[
            'value',
            'trades',
            'trade_pnl',
            'drawdowns'
        ]
    )
    fig1.write_html('credit_spread_portfolio.html')
    print("      Saved to credit_spread_portfolio.html")
except Exception as e:
    print(f"      Skipped (error: {e})")

# Plot 2: Equity curve with drawdowns
print("  [2/4] Creating equity curve with drawdowns...")
fig2 = pf.plot_cum_returns(
    benchmark_rets=None,
    show_legend=True
)
fig2.write_html('credit_spread_equity.html')
print("      Saved to credit_spread_equity.html")

# Plot 3: Underwater plot (drawdown over time)
print("  [3/4] Creating underwater plot...")
try:
    fig3 = pf.plot_underwater()
    fig3.write_html('credit_spread_drawdown.html')
    print("      Saved to credit_spread_drawdown.html")
except Exception as e:
    print(f"      Skipped (error: {e})")

# Plot 4: Credit spread with signals
print("  [4/4] Creating credit spread chart with signals...")
import plotly.graph_objects as go

# Resample spread data to align with signals
spread_aligned = spread_data['spread'].reindex(signals.index, method='ffill')

fig4 = go.Figure()

# Add credit spread line
fig4.add_trace(go.Scatter(
    x=signals.index,
    y=spread_aligned,
    mode='lines',
    name='Credit Spread',
    line=dict(color='blue', width=2)
))

# Add EMA line
fig4.add_trace(go.Scatter(
    x=signals.index,
    y=signals['ema_330'],
    mode='lines',
    name='330-day EMA',
    line=dict(color='orange', width=2, dash='dash')
))

# Add entry markers
entry_dates = signals[signals['entry']].index
entry_values = spread_aligned[entry_dates]
fig4.add_trace(go.Scatter(
    x=entry_dates,
    y=entry_values,
    mode='markers',
    name='Entry Signals',
    marker=dict(color='green', size=12, symbol='triangle-up')
))

# Add exit markers
exit_dates = signals[signals['exit']].index
exit_values = spread_aligned[exit_dates]
fig4.add_trace(go.Scatter(
    x=exit_dates,
    y=exit_values,
    mode='markers',
    name='Exit Signals',
    marker=dict(color='red', size=12, symbol='triangle-down')
))

fig4.update_layout(
    title='Credit Spread with Entry/Exit Signals',
    xaxis_title='Date',
    yaxis_title='Spread (%)',
    hovermode='x unified',
    height=600
)
fig4.write_html('credit_spread_signals.html')
print("      Saved to credit_spread_signals.html")

print()
print("="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print()
print("Generated files:")
print("  1. credit_spread_portfolio.html - Full portfolio performance")
print("  2. credit_spread_equity.html - Equity curve")
print("  3. credit_spread_drawdown.html - Drawdown underwater plot")
print("  4. credit_spread_signals.html - Credit spread with signals")
print()
print("Open any HTML file in your browser to view the interactive charts.")
print()
