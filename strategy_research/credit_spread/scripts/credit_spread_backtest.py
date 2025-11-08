"""
Credit Spread Leveraged ETF Strategy - Full Implementation
Strategy: Tier's Credit Spreads Leverage ETF Strategy

Entry: Credit spreads fall 35% from recent highs
Exit: Credit spreads rise 40% from recent lows AND cross above 330-day EMA
Position: 100% in SPXL/SSO when signal active, 100% cash when inactive
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment from config folder
config_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
load_dotenv(config_path)

print("=" * 80)
print("CREDIT SPREAD LEVERAGED ETF STRATEGY BACKTEST")
print("=" * 80)

# ============================================================================
# STEP 1: Download Credit Spread Data from FRED
# ============================================================================
print("\n[STEP 1] Downloading credit spread data from FRED...")

# ICE BofA US High Yield Option-Adjusted Spread
fred_code = 'BAMLH0A0HYM2'

# Download from 1996 to present (max available)
start_date = '1996-12-31'
end_date = datetime.now().strftime('%Y-%m-%d')

try:
    from pandas_datareader import data as pdr
except ImportError:
    print("  [ERROR] pandas_datareader not installed")
    print("  Installing pandas_datareader...")
    import subprocess
    subprocess.run(['uv', 'pip', 'install', 'pandas-datareader'], check=True)
    from pandas_datareader import data as pdr

# Download credit spread data
print(f"  Fetching {fred_code} from {start_date} to {end_date}...")
credit_spreads = pdr.DataReader(fred_code, 'fred', start_date, end_date)
credit_spreads = credit_spreads[fred_code]
credit_spreads.name = 'CreditSpread'

print(f"  [OK] Downloaded {len(credit_spreads)} days of credit spread data")
print(f"  Range: {credit_spreads.index[0].date()} to {credit_spreads.index[-1].date()}")
print(f"  Spread range: {credit_spreads.min():.2f}% to {credit_spreads.max():.2f}%")

# ============================================================================
# STEP 2: Download Price Data (SSO, SPXL, SPY)
# ============================================================================
print("\n[STEP 2] Downloading price data...")

# Use yfinance directly for historical data
print("  Downloading price data with yfinance...")
try:
    import yfinance as yf

    # Download from 2006 (SSO inception) to present
    price_start = '2006-01-01'
    price_end = datetime.now().strftime('%Y-%m-%d')

    symbols = ['SSO', 'SPXL', 'SPY']
    price_data = {}

    for symbol in symbols:
        print(f"  Downloading {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=price_start, end=price_end, auto_adjust=True)

        if len(df) > 0:
            # Keep only Close price (already adjusted)
            price_data[symbol] = df[['Close']].copy()
            price_data[symbol].columns = ['close']
            data_start = df.index[0].date()
            data_end = df.index[-1].date()
            print(f"    {symbol}: {len(df)} days ({data_start} to {data_end})")
        else:
            print(f"    {symbol}: No data available")

except Exception as e:
    print(f"  [ERROR] yfinance download failed: {e}")
    print("  [INFO] Falling back to Alpaca API...")

    # Fallback to Alpaca (requires credentials)
    from data.alpaca import fetch_alpaca_data

    price_data = {}
    for symbol in ['SSO', 'SPXL', 'SPY']:
        try:
            df = fetch_alpaca_data(symbol, timeframe='1D', period_days=365*20)
            price_data[symbol] = df
            print(f"    {symbol}: {len(df)} days via Alpaca")
        except Exception as e:
            print(f"    {symbol}: FAILED - {e}")

# ============================================================================
# STEP 3: Calculate 330-Day EMA on Credit Spreads
# ============================================================================
print("\n[STEP 3] Calculating 330-day EMA on credit spreads...")

ema_period = 330
ema_330 = credit_spreads.ewm(span=ema_period, adjust=False).mean()

print(f"  [OK] EMA calculated with {ema_period}-day span")

# ============================================================================
# STEP 4: Implement Signal Generation with 3 Approaches
# ============================================================================
print("\n[STEP 4] Generating signals with 3 different approaches...")

def generate_signals_approach_a(spreads, ema, window=180):
    """Approach A: Rolling window for recent highs/lows"""
    recent_highs = spreads.rolling(window, min_periods=1).max()
    recent_lows = spreads.rolling(window, min_periods=1).min()

    # Entry: spreads fall 35% from recent high
    entry_threshold = recent_highs * 0.65  # 65% = 100% - 35%
    entries = spreads < entry_threshold

    # Exit: spreads rise 40% from recent low AND cross above EMA
    exit_threshold = recent_lows * 1.40  # 140% = 100% + 40%
    exits = (spreads > exit_threshold) & (spreads > ema)

    return entries, exits, recent_highs, recent_lows


def generate_signals_approach_b(spreads, ema):
    """Approach B: Highest/lowest since last signal change"""
    entries = pd.Series(False, index=spreads.index)
    exits = pd.Series(False, index=spreads.index)

    recent_high = spreads.iloc[0]
    recent_low = spreads.iloc[0]
    in_market = False

    for i in range(len(spreads)):
        current = spreads.iloc[i]
        current_ema = ema.iloc[i]

        if not in_market:
            # Looking for entry: track recent highs
            if current > recent_high:
                recent_high = current

            # Entry signal: 35% fall from recent high
            if current < recent_high * 0.65:
                entries.iloc[i] = True
                in_market = True
                recent_low = current  # Reset recent low when entering
        else:
            # Looking for exit: track recent lows
            if current < recent_low:
                recent_low = current

            # Exit signal: 40% rise from recent low AND above EMA
            if (current > recent_low * 1.40) and (current > current_ema):
                exits.iloc[i] = True
                in_market = False
                recent_high = current  # Reset recent high when exiting

    return entries, exits


def generate_signals_approach_c(spreads, ema):
    """Approach C: Peak detection algorithm"""
    from scipy.signal import find_peaks

    # Find local maxima (peaks) and minima (troughs)
    peaks, _ = find_peaks(spreads.values, distance=30)  # At least 30 days apart
    troughs, _ = find_peaks(-spreads.values, distance=30)

    entries = pd.Series(False, index=spreads.index)
    exits = pd.Series(False, index=spreads.index)

    # For each point, find the most recent peak/trough
    peak_values = pd.Series(index=spreads.index, dtype=float)
    trough_values = pd.Series(index=spreads.index, dtype=float)

    for i in range(len(spreads)):
        # Most recent peak before this point
        prior_peaks = peaks[peaks < i]
        if len(prior_peaks) > 0:
            peak_values.iloc[i] = spreads.iloc[prior_peaks[-1]]
        else:
            peak_values.iloc[i] = spreads.iloc[0]

        # Most recent trough before this point
        prior_troughs = troughs[troughs < i]
        if len(prior_troughs) > 0:
            trough_values.iloc[i] = spreads.iloc[prior_troughs[-1]]
        else:
            trough_values.iloc[i] = spreads.iloc[0]

    # Entry: 35% fall from recent peak
    entries = spreads < (peak_values * 0.65)

    # Exit: 40% rise from recent trough AND above EMA
    exits = (spreads > trough_values * 1.40) & (spreads > ema)

    return entries, exits


# Test all 3 approaches
print("\n  Testing Approach A: Rolling window (180 days)...")
entries_a, exits_a, highs_a, lows_a = generate_signals_approach_a(credit_spreads, ema_330, window=180)
entries_a, exits_a = entries_a.vbt.signals.clean(exits_a)
print(f"    Entries: {entries_a.sum()}, Exits: {exits_a.sum()}")

print("\n  Testing Approach B: Since last signal change...")
entries_b, exits_b = generate_signals_approach_b(credit_spreads, ema_330)
print(f"    Entries: {entries_b.sum()}, Exits: {exits_b.sum()}")

# Approach C requires scipy
try:
    print("\n  Testing Approach C: Peak detection...")
    entries_c, exits_c = generate_signals_approach_c(credit_spreads, ema_330)
    entries_c, exits_c = entries_c.vbt.signals.clean(exits_c)
    print(f"    Entries: {entries_c.sum()}, Exits: {exits_c.sum()}")
except ImportError:
    print("\n  [INFO] Approach C requires scipy (skipping)")
    entries_c, exits_c = None, None

# ============================================================================
# STEP 5: Validate Against Historical Dates from Video
# ============================================================================
print("\n[STEP 5] Validating signals against video historical dates...")

# Historical dates from video (Table in strategy rules document)
video_dates = {
    '1998-08-18': 'EXIT',
    '2003-04-03': 'ENTRY',
    '2005-04-14': 'EXIT',
    '2006-05-04': 'ENTRY',
    '2007-07-19': 'EXIT',
    '2009-04-30': 'ENTRY',
    '2011-08-04': 'EXIT',
    '2012-03-13': 'ENTRY',
    '2014-10-09': 'EXIT',
    '2016-07-12': 'ENTRY',
    '2018-12-05': 'EXIT',
    '2019-12-13': 'ENTRY',
    '2020-02-26': 'EXIT',
    '2020-05-21': 'ENTRY',
    '2022-03-14': 'EXIT',
    '2023-07-15': 'ENTRY'
}

def validate_signals(entries, exits, name=""):
    """Compare generated signals to video dates"""
    print(f"\n  {name}:")
    matches = 0
    total_video_signals = len(video_dates)

    for date_str, expected_action in video_dates.items():
        date = pd.to_datetime(date_str)

        # Check within ±3 days tolerance
        window_start = date - pd.Timedelta(days=3)
        window_end = date + pd.Timedelta(days=3)

        # Find signals in window
        if expected_action == 'ENTRY':
            signals_in_window = entries.loc[window_start:window_end]
            if signals_in_window.any():
                actual_date = signals_in_window[signals_in_window].index[0]
                days_off = (actual_date - date).days
                print(f"    {date_str} ENTRY: MATCH (off by {days_off} days)")
                matches += 1
            else:
                print(f"    {date_str} ENTRY: MISS")
        else:  # EXIT
            signals_in_window = exits.loc[window_start:window_end]
            if signals_in_window.any():
                actual_date = signals_in_window[signals_in_window].index[0]
                days_off = (actual_date - date).days
                print(f"    {date_str} EXIT: MATCH (off by {days_off} days)")
                matches += 1
            else:
                print(f"    {date_str} EXIT: MISS")

    match_rate = matches / total_video_signals
    print(f"\n    Match Rate: {match_rate:.1%} ({matches}/{total_video_signals})")
    return match_rate

# Validate all approaches
match_rate_a = validate_signals(entries_a, exits_a, "Approach A (Rolling Window)")
match_rate_b = validate_signals(entries_b, exits_b, "Approach B (Since Last Signal)")
if entries_c is not None:
    match_rate_c = validate_signals(entries_c, exits_c, "Approach C (Peak Detection)")
else:
    match_rate_c = 0

# Select best approach
best_approach = max(
    [('A', match_rate_a), ('B', match_rate_b), ('C', match_rate_c)],
    key=lambda x: x[1]
)

print(f"\n  [RESULT] Best approach: {best_approach[0]} ({best_approach[1]:.1%} match rate)")

# Use best approach for backtest
if best_approach[0] == 'A':
    final_entries, final_exits = entries_a, exits_a
elif best_approach[0] == 'B':
    final_entries, final_exits = entries_b, exits_b
else:
    final_entries, final_exits = entries_c, exits_c

# ============================================================================
# STEP 6: Run Backtest on SSO (2007-2024)
# ============================================================================
print("\n[STEP 6] Running backtest on SSO (2007-2024)...")

# Get SSO prices
sso_data = price_data['SSO']
sso_close = sso_data['close']

# Align signals with price data
# CRITICAL: Handle initial state - determine if we should be in market at start
sso_start_date = sso_close.index[0]

# Remove timezone for comparison (credit spread data is timezone-naive)
sso_start_date_naive = sso_start_date.tz_localize(None) if hasattr(sso_start_date, 'tz_localize') else sso_start_date

# Find the most recent signal before SSO data starts
# CRITICAL: Filter for where signal is actually True, not just all dates
entries_before_sso = final_entries[final_entries.index < sso_start_date_naive]
signals_before_start = entries_before_sso[entries_before_sso].index  # Only dates where entry==True

exits_before_sso = final_exits[final_exits.index < sso_start_date_naive]
exits_before_start = exits_before_sso[exits_before_sso].index  # Only dates where exit==True

# Determine initial state
should_be_in_market_at_start = False
if len(signals_before_start) > 0 or len(exits_before_start) > 0:
    print(f"  [DEBUG] Entries before SSO start: {len(signals_before_start)}")
    print(f"  [DEBUG] Exits before SSO start: {len(exits_before_start)}")
    if len(signals_before_start) > 0:
        print(f"  [DEBUG] Last entry before SSO: {signals_before_start[-1]}")
    if len(exits_before_start) > 0:
        print(f"  [DEBUG] Last exit before SSO: {exits_before_start[-1]}")

    # Find the most recent signal (entry or exit) before SSO starts
    all_signals_before = pd.concat([
        pd.Series(True, index=signals_before_start),  # Entries
        pd.Series(False, index=exits_before_start)  # Exits (False = exit)
    ]).sort_index()

    if len(all_signals_before) > 0:
        last_signal_was_entry = all_signals_before.iloc[-1]
        last_signal_date = all_signals_before.index[-1]
        should_be_in_market_at_start = last_signal_was_entry
        print(f"  [INFO] Last signal before SSO start: {last_signal_date.date()} - {'ENTRY' if last_signal_was_entry else 'EXIT'}")
        print(f"  [INFO] Initial state: {'IN market' if should_be_in_market_at_start else 'OUT of market'}")

# Align signals to price data using merge_asof for nearest match
# This ensures signals trigger on the nearest available trading day
# Remove timezone from SSO index for alignment
sso_index_naive = sso_close.index.tz_localize(None) if hasattr(sso_close.index, 'tz_localize') else sso_close.index

# Convert signals to DataFrames for merge_asof
signal_dates_entries = final_entries[final_entries].index.to_frame(index=False, name='signal_date')
signal_dates_entries['entry'] = True

signal_dates_exits = final_exits[final_exits].index.to_frame(index=False, name='signal_date')
signal_dates_exits['exit'] = True

price_dates = sso_index_naive.to_frame(index=False, name='price_date')

# Match each signal to the nearest trading day (forward direction only, max 5 days)
entries_matched = pd.merge_asof(
    signal_dates_entries.sort_values('signal_date'),
    price_dates.sort_values('price_date'),
    left_on='signal_date',
    right_on='price_date',
    direction='forward',
    tolerance=pd.Timedelta(days=5)
).dropna()

exits_matched = pd.merge_asof(
    signal_dates_exits.sort_values('signal_date'),
    price_dates.sort_values('price_date'),
    left_on='signal_date',
    right_on='price_date',
    direction='forward',
    tolerance=pd.Timedelta(days=5)
).dropna()

# Create boolean series for entries and exits
entries_aligned = pd.Series(False, index=sso_index_naive)
exits_aligned = pd.Series(False, index=sso_index_naive)

if len(entries_matched) > 0:
    for date in entries_matched['price_date']:
        entries_aligned.loc[date] = True

if len(exits_matched) > 0:
    for date in exits_matched['price_date']:
        exits_aligned.loc[date] = True

# If we should be in market at start, add an entry signal at the first bar
if should_be_in_market_at_start:
    entries_aligned.iloc[0] = True
    print(f"  [INFO] Added initial ENTRY at {sso_start_date.date()} to reflect pre-existing position")

# Final cleanup: ensure alternating entry/exit
entries_aligned, exits_aligned = entries_aligned.vbt.signals.clean(exits_aligned)

print(f"  [INFO] Aligned signals: {entries_aligned.sum()} entries, {exits_aligned.sum()} exits")

# Align all series to the same index (timezone-aware from sso_close)
# Reindex entries/exits back to the original sso_close index with timezone
entries_final = pd.Series(False, index=sso_close.index)
exits_final = pd.Series(False, index=sso_close.index)

for i, date_naive in enumerate(entries_aligned.index):
    if entries_aligned.iloc[i]:
        # Find matching date in sso_close index (with timezone)
        matching_dates = sso_close.index[sso_close.index.date == date_naive.date()]
        if len(matching_dates) > 0:
            entries_final.loc[matching_dates[0]] = True

for i, date_naive in enumerate(exits_aligned.index):
    if exits_aligned.iloc[i]:
        # Find matching date in sso_close index (with timezone)
        matching_dates = sso_close.index[sso_close.index.date == date_naive.date()]
        if len(matching_dates) > 0:
            exits_final.loc[matching_dates[0]] = True

aligned_data = pd.DataFrame({
    'close': sso_close,
    'entries': entries_final,
    'exits': exits_final
})

# Backtest with 100% position sizing
pf = vbt.PF.from_signals(
    close=aligned_data['close'],
    entries=aligned_data['entries'],
    exits=aligned_data['exits'],
    size=1.0,  # 100% of portfolio
    size_type='valuepercent',
    init_cash=10000,  # £10,000 starting capital
    fees=0.0009,  # 0.09% expense ratio (SSO ~0.90% annual)
    freq='D'
)

print(f"\n  Initial Capital: £{pf.init_cash:,.0f}")
print(f"  Final Value: £{pf.final_value:,.0f}")
print(f"  Total Return: {pf.total_return:.2%}")
print(f"  Multiple: {pf.final_value / pf.init_cash:.2f}x")
print(f"\n  Number of Trades: {pf.trades.count()}")
if pf.trades.count() > 0:
    print(f"  Win Rate: {pf.trades.win_rate:.2%}")

print(f"\n  Sharpe Ratio: {pf.sharpe_ratio:.2f}")
print(f"  Sortino Ratio: {pf.sortino_ratio:.2f}")
print(f"  Calmar Ratio: {pf.calmar_ratio:.2f}")
print(f"  Max Drawdown: {pf.max_drawdown:.2%}")

# Time in market
position = pf.position
time_in_market = (position != 0).sum() / len(position)
print(f"  Time in Market: {time_in_market:.2%}")

# ============================================================================
# STEP 7: Compare to Video Claims
# ============================================================================
print("\n[STEP 7] Comparing results to video claims...")

video_claim_multiple = 16.3
actual_multiple = pf.final_value / pf.init_cash
difference_pct = (actual_multiple - video_claim_multiple) / video_claim_multiple * 100

print(f"\n  Video Claim: 16.3x return (£163,651 from £10,000)")
print(f"  Our Result: {actual_multiple:.2f}x return (£{pf.final_value:,.0f} from £{pf.init_cash:,.0f})")
print(f"  Difference: {difference_pct:+.1f}%")

if abs(difference_pct) < 20:
    print(f"  [VALIDATION PASSED] Within 20% tolerance")
else:
    print(f"  [NEEDS INVESTIGATION] Outside 20% tolerance")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n[STEP 8] Saving results...")

# Export signals to CSV for validation
signal_export = pd.DataFrame({
    'date': credit_spreads.index,
    'spread': credit_spreads.values,
    'ema_330': ema_330.values,
    'entry': final_entries.values,
    'exit': final_exits.values
})
signal_export = signal_export[signal_export['entry'] | signal_export['exit']]
signal_export.to_csv('credit_spread_signals.csv', index=False)
print(f"  [OK] Signals saved to credit_spread_signals.csv ({len(signal_export)} signals)")

# Export trades
if pf.trades.count() > 0:
    trades_df = pf.trades.records_readable
    trades_df.to_csv('credit_spread_trades.csv', index=False)
    print(f"  [OK] Trades saved to credit_spread_trades.csv ({len(trades_df)} trades)")

# Save performance stats
stats = pf.stats()
stats.to_csv('credit_spread_performance.csv')
print(f"  [OK] Performance stats saved to credit_spread_performance.csv")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
