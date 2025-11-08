"""
RE-TEST Credit Spread Strategy with SPXL (3x) instead of SSO (2x)
CRITICAL FIX: Was using wrong ETF the entire time!
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREDIT SPREAD STRATEGY - SPXL (3X) BACKTEST")
print("="*80)
print()

# Download SPXL data (FIXED: use auto_adjust=False for accurate prices)
print("[1] Downloading SPXL data...")
print("  Using auto_adjust=False to get real tradable prices (not retroactively adjusted)")
spxl = yf.download('SPXL', start='2008-01-01', progress=False, auto_adjust=False)
spxl_close = spxl['Adj Close']  # Use Adj Close (includes dividends, not splits)
print(f"  SPXL: {len(spxl_close)} days ({spxl_close.index[0]} to {spxl_close.index[-1]})")

# Data quality validation
print("\n  Data Quality Checks:")
try:
    june_2025_price = float(spxl.loc['2025-06-30', 'Close'].iloc[0]) if hasattr(spxl.loc['2025-06-30', 'Close'], 'iloc') else float(spxl.loc['2025-06-30', 'Close'])
    print(f"    June 30, 2025 price: ${june_2025_price:.2f}")
    if 150 <= june_2025_price <= 200:
        print(f"    [PASS] Price in expected range ($150-$200)")
    else:
        print(f"    [WARN] Price outside expected range")
except (KeyError, IndexError):
    print(f"    [SKIP] June 30, 2025 not in dataset")

bh_return_raw = float(spxl_close.iloc[-1] / spxl_close.iloc[0])
print(f"    SPXL Buy-and-Hold: {bh_return_raw:.2f}x")
if 60 <= bh_return_raw <= 70:
    print(f"    [PASS] Buy-and-hold return reasonable (60-70x from 2008 bottom)")
elif bh_return_raw > 70:
    print(f"    [WARN] Buy-and-hold seems too high (>70x)")
else:
    print(f"    [WARN] Buy-and-hold seems too low (<60x)")
print()

# Download SPY for benchmark
spy = yf.download('SPY', start='2008-01-01', progress=False, auto_adjust=False)
spy_close = spy['Adj Close']
print(f"  SPY: {len(spy_close)} days")
print()

# Load credit spread signals from CSV
print("[2] Loading credit spread signals...")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
signals_path = os.path.join(script_dir, '..', 'reports', 'credit_spread_signals.csv')
signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
print(f"  Signals loaded: {len(signals)} events")
print()

# Align signals to SPXL index
print("[3] Aligning signals to SPXL trading days...")
entries = pd.Series(False, index=spxl_close.index)
exits = pd.Series(False, index=spxl_close.index)

for date_str in signals[signals['entry']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    matching_dates = spxl_close.index[spxl_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        entries.loc[matching_dates[0]] = True

for date_str in signals[signals['exit']].index:
    signal_date = pd.to_datetime(date_str).tz_localize(None)
    matching_dates = spxl_close.index[spxl_close.index.tz_localize(None).date == signal_date.date()]
    if len(matching_dates) > 0:
        exits.loc[matching_dates[0]] = True

print(f"  Aligned: {entries.sum()} entries, {exits.sum()} exits")
print()

# Run backtest
print("[4] Running backtest with SPXL...")
pf = vbt.Portfolio.from_signals(
    close=spxl_close,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,
    size=1.0,
    size_type='valuepercent',
    freq='1D'
)

# Calculate returns
returns = pf.returns
if isinstance(returns, pd.DataFrame):
    returns = returns.iloc[:, 0]

# Get performance metrics
final_value = float(pf.value.iloc[-1] if hasattr(pf.value, 'iloc') else pf.value.values[-1])
total_return = (final_value / 10000 - 1)
multiple = final_value / 10000

sharpe = float(pf.sharpe_ratio.iloc[0] if hasattr(pf.sharpe_ratio, 'iloc') else pf.sharpe_ratio)
sortino = float(pf.sortino_ratio.iloc[0] if hasattr(pf.sortino_ratio, 'iloc') else pf.sortino_ratio)
max_dd = float(pf.max_drawdown.iloc[0] if hasattr(pf.max_drawdown, 'iloc') else pf.max_drawdown)
win_rate = float(pf.trades.win_rate.iloc[0] if hasattr(pf.trades.win_rate, 'iloc') else pf.trades.win_rate)
num_trades = int(pf.trades.count().iloc[0] if hasattr(pf.trades.count(), 'iloc') else pf.trades.count())

print()
print("="*80)
print("SPXL STRATEGY RESULTS")
print("="*80)
print(f"  Initial Capital: $10,000")
print(f"  Final Value: ${final_value:,.2f}")
print(f"  Total Return: {total_return:.2%}")
print(f"  Multiple: {multiple:.2f}x")
print(f"  Number of Trades: {num_trades}")
print(f"  Win Rate: {win_rate:.2%}")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Sortino Ratio: {sortino:.2f}")
print(f"  Max Drawdown: {max_dd:.2%}")
print()

# Compare to buy-and-hold
print("="*80)
print("BENCHMARK COMPARISON")
print("="*80)

spxl_bh_return = float(spxl_close.iloc[-1] / spxl_close.iloc[0] - 1)
spxl_bh_multiple = spxl_bh_return + 1
spy_bh_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)

print(f"  SPXL Buy & Hold: {spxl_bh_return:.2%} ({spxl_bh_multiple:.2f}x)")
print(f"  SPY Buy & Hold: {spy_bh_return:.2%} ({spy_bh_return+1:.2f}x)")
print()
print(f"  Strategy vs SPXL B&H: {(multiple/spxl_bh_multiple - 1):.1%}")
print(f"  Strategy vs SPY B&H: {(multiple/(spy_bh_return+1) - 1):.1%}")
print()

# Video comparison
print("="*80)
print("VIDEO CLAIM COMPARISON")
print("="*80)
video_multiple = 16.37
print(f"  Video Claim: {(video_multiple-1):.2%} ({video_multiple:.2f}x)")
print(f"  Our Result: {total_return:.2%} ({multiple:.2f}x)")
print(f"  Difference: {(multiple/video_multiple - 1):.1%}")
print()

if multiple > spxl_bh_multiple:
    print("  [SUCCESS] Strategy BEATS SPXL buy-and-hold!")
    print(f"  Alpha: {(multiple/spxl_bh_multiple - 1):.1%}")
else:
    print("  [FAILURE] Strategy loses to SPXL buy-and-hold")
    print(f"  Underperformance: {(multiple/spxl_bh_multiple - 1):.1%}")

print()
print("="*80)
