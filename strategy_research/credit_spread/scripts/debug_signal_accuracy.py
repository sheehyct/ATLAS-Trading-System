"""
Debug Credit Spread Strategy Signal Accuracy
Investigate the 40% return gap and near-SPY performance
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import vectorbtpro as vbt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEBUGGING CREDIT SPREAD STRATEGY SIGNAL ACCURACY")
print("="*80)
print()

# Load credit spread data
print("[1] Loading credit spread data (FRED)...")
spread_data = pdr.DataReader('BAMLH0A0HYM2', 'fred', start='1996-01-01')
spread_data.columns = ['spread']
print(f"  Credit spread data: {len(spread_data)} days ({spread_data.index[0]} to {spread_data.index[-1]})")
print()

# Calculate 330-day EMA
print("[2] Calculating 330-day EMA...")
spread_data['ema_330'] = spread_data['spread'].ewm(span=330, adjust=False).mean()
print(f"  EMA calculated")
print()

# Video's claimed signal dates
video_signals = {
    'entries': [
        '2003-04-03', '2006-05-04', '2009-04-30', '2012-03-13',
        '2016-07-12', '2019-12-13', '2020-05-21', '2023-07-15'
    ],
    'exits': [
        '1998-08-18', '2005-04-14', '2007-07-19', '2011-08-04',
        '2014-10-09', '2018-12-05', '2020-02-26', '2022-03-14'
    ]
}

print("[3] Analyzing VIDEO signals vs actual spread data...")
print()

for signal_type, dates in video_signals.items():
    print(f"  {signal_type.upper()}:")
    for date_str in dates:
        date = pd.to_datetime(date_str)
        if date in spread_data.index:
            row = spread_data.loc[date]
            spread_val = row['spread']
            ema_val = row['ema_330']

            # Check recent high/low
            lookback_60 = spread_data.loc[:date].tail(60)
            recent_high = lookback_60['spread'].max()
            recent_low = lookback_60['spread'].min()

            if signal_type == 'entries':
                # Entry condition: spread fell 35% from recent high
                pct_from_high = (spread_val - recent_high) / recent_high
                print(f"    {date_str}: Spread={spread_val:.2f}, EMA={ema_val:.2f}, Recent High={recent_high:.2f}")
                print(f"              Down {pct_from_high:.1%} from high (need -35%)")
            else:
                # Exit condition: spread rose 40% from recent low AND above EMA
                pct_from_low = (spread_val - recent_low) / recent_low
                above_ema = spread_val > ema_val
                print(f"    {date_str}: Spread={spread_val:.2f}, EMA={ema_val:.2f}, Recent Low={recent_low:.2f}")
                print(f"              Up {pct_from_low:.1%} from low (need +40%), Above EMA: {above_ema}")
        else:
            print(f"    {date_str}: NOT IN SPREAD DATA (weekend/holiday?)")
        print()

print()
print("="*80)
print("[4] Testing different lookback windows for 'recent high/low'...")
print("="*80)
print()

# Test different lookback periods
lookback_periods = [20, 40, 60, 90, 120, 252]

for lookback in lookback_periods:
    print(f"\nLOOKBACK = {lookback} days:")
    print("-" * 40)

    entries_found = []
    exits_found = []

    for i in range(lookback, len(spread_data)):
        current = spread_data.iloc[i]
        current_date = spread_data.index[i]
        lookback_window = spread_data.iloc[i-lookback:i]

        recent_high = lookback_window['spread'].max()
        recent_low = lookback_window['spread'].min()

        # Entry: 35% fall from recent high
        pct_from_high = (current['spread'] - recent_high) / recent_high
        if pct_from_high <= -0.35:
            # Check if this is a new signal (not consecutive)
            if len(entries_found) == 0 or (current_date - entries_found[-1]).days > 5:
                entries_found.append(current_date)

        # Exit: 40% rise from recent low AND above EMA
        pct_from_low = (current['spread'] - recent_low) / recent_low
        if pct_from_low >= 0.40 and current['spread'] > current['ema_330']:
            if len(exits_found) == 0 or (current_date - exits_found[-1]).days > 5:
                exits_found.append(current_date)

    # Compare to video dates
    video_entry_dates = [pd.to_datetime(d) for d in video_signals['entries']]
    video_exit_dates = [pd.to_datetime(d) for d in video_signals['exits']]

    entry_matches = sum(1 for date in entries_found if any(abs((date - vd).days) <= 5 for vd in video_entry_dates))
    exit_matches = sum(1 for date in exits_found if any(abs((date - vd).days) <= 5 for vd in video_exit_dates))

    total_matches = entry_matches + exit_matches
    total_video_signals = len(video_signals['entries']) + len(video_signals['exits'])
    match_rate = total_matches / total_video_signals

    print(f"  Entries found: {len(entries_found)} (matches: {entry_matches}/{len(video_signals['entries'])})")
    print(f"  Exits found: {len(exits_found)} (matches: {exit_matches}/{len(video_signals['exits'])})")
    print(f"  Total match rate: {match_rate:.1%}")

    if match_rate > 0.7:  # If >70% match, show details
        print(f"\n  Generated ENTRY dates:")
        for date in entries_found[:10]:  # Show first 10
            print(f"    {date.strftime('%Y-%m-%d')}")
        print(f"\n  Generated EXIT dates:")
        for date in exits_found[:10]:
            print(f"    {date.strftime('%Y-%m-%d')}")

print()
print("="*80)
print("[5] Comparing SSO vs SPY performance...")
print("="*80)
print()

# Download data
sso = yf.download('SSO', start='2006-06-21', progress=False)
spy = yf.download('SPY', start='2006-06-21', progress=False)

# Calculate buy and hold returns
sso_close = sso['Close']
spy_close = spy['Close']

sso_bh_return = float(sso_close.iloc[-1] / sso_close.iloc[0] - 1)
spy_bh_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)

print(f"SSO Buy & Hold (2006-2025): {sso_bh_return:.2%} ({sso_bh_return+1:.2f}x)")
print(f"SPY Buy & Hold (2006-2025): {spy_bh_return:.2%} ({spy_bh_return+1:.2f}x)")
print(f"SSO vs SPY ratio: {(sso_bh_return+1)/(spy_bh_return+1):.2f}x")
print()

# Calculate video's claimed return
video_start = 10000
video_end = 163651
video_return = (video_end / video_start - 1)
print(f"Video claim (GBP 10,000 -> GBP 163,651): {video_return:.2%} ({video_return+1:.2f}x)")
print()

# Our result
our_return = 9.6974
print(f"Our backtest result: {(our_return-1)*100:.2%} ({our_return:.2f}x)")
print(f"Gap vs video: {(video_return+1)/our_return - 1:.1%} lower")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Key Findings:")
print("1. Test different lookback periods to match video signals")
print("2. Check if SPY comparison should use SPY or SSO as benchmark")
print("3. Investigate if video used different ETF or date range")
print()
