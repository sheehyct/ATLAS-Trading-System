"""
Check live ATLAS regime detection with current VIX spike.
"""
import vectorbtpro as vbt
import pandas as pd
from datetime import datetime
from regime.academic_jump_model import AcademicJumpModel

# Fetch live SPY data
print('Fetching live SPY data...')
spy_data = vbt.YFData.pull('SPY', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), timeframe='1d', tz='America/New_York')
spy_df = spy_data.get()
print(f'SPY data: {len(spy_df)} days, latest: {spy_df.index[-1]}')
print(f'SPY close: ${spy_df["Close"].iloc[-1]:.2f}')

# Fetch live VIX data
print('\nFetching live VIX data...')
vix_data = vbt.YFData.pull('^VIX', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), timeframe='1d')
vix_df = vix_data.get()
vix_close = vix_df['Close']
print(f'VIX data: {len(vix_df)} days, latest: {vix_df.index[-1]}')
print(f'VIX close: {vix_close.iloc[-1]:.2f}')

# Check VIX spike
vix_1d_change = (vix_close.iloc[-1] / vix_close.iloc[-2] - 1) * 100
vix_3d_change = (vix_close.iloc[-1] / vix_close.iloc[-4] - 1) * 100
print(f'\nVIX SPIKE ANALYSIS:')
print(f'  1-day change: {vix_1d_change:+.1f}%')
print(f'  3-day change: {vix_3d_change:+.1f}%')
print(f'  Flash crash threshold: +20% (1-day) or +50% (3-day)')

if vix_1d_change >= 20:
    print(f'  ⚠️  1-DAY THRESHOLD EXCEEDED ({vix_1d_change:+.1f}% >= 20%)')
if vix_3d_change >= 50:
    print(f'  ⚠️  3-DAY THRESHOLD EXCEEDED ({vix_3d_change:+.1f}% >= 50%)')

# Run ATLAS regime detection
print('\nRunning ATLAS regime detection...')
atlas = AcademicJumpModel()
regimes, lambdas, thetas = atlas.online_inference(
    spy_df,
    lookback=1000,
    default_lambda=1.5,
    vix_data=vix_close
)

# Get current regime
current_regime = regimes.iloc[-1]
print(f'\n{"="*60}')
print(f'CURRENT REGIME: {current_regime}')
print(f'{"="*60}')

if current_regime == 'CRASH':
    print('\n🚨 CRASH REGIME DETECTED 🚨')
    print('System A1 allocation: 0% (100% cash)')
    print('Recommendation: DO NOT DEPLOY - Wait for regime to clear')
else:
    print(f'\nSystem A1 allocation: {{"TREND_BULL": "100%", "TREND_NEUTRAL": "70%", "TREND_BEAR": "30%", "CRASH": "0%"}}[current_regime]')
    print('Deployment status: SAFE TO PROCEED')

# Recent regime history
print('\n' + '='*60)
print('RECENT REGIME HISTORY (Last 10 Days):')
print('='*60)
for i in range(-10, 0):
    date = regimes.index[i]
    regime = regimes.iloc[i]
    spy_close = spy_df.loc[date, 'Close']
    vix_val = vix_close.loc[date] if date in vix_close.index else None

    spy_change = (spy_df.loc[date, 'Close'] / spy_df['Close'].iloc[i-1] - 1) * 100 if i > -10 else 0

    vix_str = f'{vix_val:5.2f}' if vix_val is not None else '  N/A'
    print(f'{date.strftime("%Y-%m-%d")}: {regime:15s} | SPY ${spy_close:7.2f} ({spy_change:+5.1f}%) | VIX {vix_str}')

# Regime distribution
print('\n' + '='*60)
print('REGIME DISTRIBUTION (Full Period):')
print('='*60)
regime_counts = regimes.value_counts()
for regime, count in regime_counts.items():
    pct = 100 * count / len(regimes)
    print(f'{regime:15s}: {count:4d} days ({pct:5.1f}%)')
