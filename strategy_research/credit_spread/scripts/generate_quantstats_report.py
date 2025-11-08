"""
Generate QuantStats professional tearsheet for Credit Spread Strategy
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import yfinance as yf
import quantstats as qs
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING QUANTSTATS PROFESSIONAL TEARSHEET")
print("="*80)
print()

# Load SSO data
print("[1/4] Loading market data...")
sso = yf.download('SSO', start='2006-01-01', progress=False)
sso_close = sso['Close']

# Download SPY as benchmark
spy = yf.download('SPY', start='2006-01-01', progress=False)
spy_close = spy['Close']

print(f"  SSO: {len(sso_close)} days")
print(f"  SPY (benchmark): {len(spy_close)} days")
print()

# Load signals from CSV
print("[2/4] Loading backtest signals...")
signals = pd.read_csv('credit_spread_signals.csv', index_col=0, parse_dates=True)

# Prepare signals aligned to SSO index
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

print(f"  Signals aligned: {entries.sum()} entries, {exits.sum()} exits")
print()

# Create VBT Portfolio
print("[3/4] Running backtest...")
pf = vbt.Portfolio.from_signals(
    close=sso_close,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,
    size=1.0,
    size_type='valuepercent',
    freq='1D'
)

# Extract returns
strategy_returns = pf.returns
# If returns is a DataFrame, get the first column as Series
if isinstance(strategy_returns, pd.DataFrame):
    strategy_returns = strategy_returns.iloc[:, 0]

benchmark_returns = spy_close.pct_change().reindex(strategy_returns.index).fillna(0)

print(f"  Portfolio created: {len(strategy_returns)} days")
total_ret = pf.total_return.iloc[0] if hasattr(pf.total_return, 'iloc') else pf.total_return
print(f"  Total Return: {total_ret:.2%}")
print()

# Generate QuantStats tearsheet
print("[4/4] Generating QuantStats tearsheet...")

# Extend pandas for QuantStats
qs.extend_pandas()

# Generate full HTML report
qs.reports.html(
    strategy_returns,
    benchmark_returns,
    output='quantstats_credit_spread_tearsheet.html',
    title='Credit Spread Leveraged ETF Strategy',
    download_filename='quantstats_tearsheet.html'
)

print("  [OK] Full tearsheet saved: quantstats_credit_spread_tearsheet.html")
print()

# Skip console metrics (Windows encoding issues)
# The HTML report contains all metrics anyway

print()
print("="*80)
print("QUANTSTATS REPORT COMPLETE")
print("="*80)
print()
print("Generated files:")
print("  1. quantstats_credit_spread_tearsheet.html - Full professional tearsheet")
print()
print("Key features in tearsheet:")
print("  - Cumulative returns vs benchmark")
print("  - Monthly returns heatmap")
print("  - Distribution of monthly returns")
print("  - Drawdown periods (ranked)")
print("  - Rolling volatility & Sharpe")
print("  - Rolling beta & R-squared")
print("  - Daily returns analysis")
print("  - Comprehensive metrics table")
print()
print("Open the HTML file in your browser to view the full report.")
print()
