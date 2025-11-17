"""
Diagnostic Script: ORB Strategy Signal Decomposition (January 2024)

Purpose: Identify why ORB strategy executed 0 trades in Jan 2024
Approach: Calculate each signal component separately and log TRUE counts

Test Period: 2024-01-01 to 2024-01-31 (same as failing test)
Expected: Find which condition(s) block trades (price/volume/time/direction)

Run: uv run python tests/diagnostic_orb_jan2024.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import time, datetime, timedelta
import pandas_market_calendars as mcal
import os
from dotenv import load_dotenv

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

# Load environment from root .env
load_dotenv()


def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fetch_data_jan2024():
    """Fetch SPY data for January 2024 (same as test period)"""
    print_section("STEP 1: DATA FETCHING")

    # Configure Alpaca
    api_key = os.getenv('ALPACA_MID_KEY')
    api_secret = os.getenv('ALPACA_MID_SECRET')

    if not api_key or not api_secret:
        raise ValueError("Alpaca credentials not found in .env")

    vbt.AlpacaData.set_custom_settings(
        client_config=dict(
            api_key=api_key,
            secret_key=api_secret
        )
    )

    # Fetch 5-minute data
    print("Fetching 5-minute SPY data...")
    data_5min = vbt.AlpacaData.pull(
        'SPY',
        start='2024-01-01',
        end='2024-01-31',
        timeframe='5Min',
        tz='America/New_York'
    ).get()

    # Fetch daily data for ATR
    print("Fetching daily SPY data...")
    data_daily = vbt.AlpacaData.pull(
        'SPY',
        start='2024-01-01',
        end='2024-01-31',
        timeframe='1D',
        tz='America/New_York'
    ).get()

    print(f"\nRaw data fetched:")
    print(f"  5-minute bars: {len(data_5min)}")
    print(f"  Daily bars: {len(data_daily)}")

    # Apply RTH filter
    print(f"\nApplying RTH filter (9:30 AM - 4:00 PM ET)...")
    data_5min_before = len(data_5min)
    data_5min = data_5min.between_time('09:30', '16:00')
    print(f"  Before RTH filter: {data_5min_before} bars")
    print(f"  After RTH filter: {len(data_5min)} bars")

    # Filter NYSE trading days only
    print(f"\nApplying NYSE trading days filter...")
    nyse = mcal.get_calendar('NYSE')
    trading_days = nyse.valid_days(start_date='2024-01-01', end_date='2024-01-31')

    # Date-only comparison (timezone-safe)
    data_5min_dates = pd.Series(data_5min.index.date, index=data_5min.index)
    trading_days_dates = pd.DatetimeIndex(trading_days).date

    data_5min_before = len(data_5min)
    data_5min = data_5min[data_5min_dates.isin(trading_days_dates)]

    data_daily_dates = pd.Series(data_daily.index.date, index=data_daily.index)
    data_daily = data_daily[data_daily_dates.isin(trading_days_dates)]

    print(f"  5min before trading days filter: {data_5min_before} bars")
    print(f"  5min after trading days filter: {len(data_5min)} bars")
    print(f"  Daily bars after filter: {len(data_daily)} bars")
    print(f"  Trading days in Jan 2024: {len(trading_days)}")

    if len(data_5min) == 0:
        print("\n[CRITICAL] Zero bars after filtering! Cannot proceed.")
        return None, None

    print(f"\n[SUCCESS] Data ready for analysis")
    print(f"  Date range: {data_5min.index[0]} to {data_5min.index[-1]}")

    return data_5min, data_daily


def calculate_opening_range(data_5min, opening_minutes=30):
    """Calculate opening range (same logic as ORBStrategy)"""
    print_section("STEP 2: OPENING RANGE CALCULATION")

    # Group by trading day
    daily_groups = data_5min.groupby(data_5min.index.date)

    opening_high_list = []
    opening_low_list = []
    opening_close_list = []
    opening_open_list = []
    dates = []

    n_bars = opening_minutes // 5  # 30 min = 6 bars

    print(f"Opening range: First {opening_minutes} minutes ({n_bars} bars)")

    successful_days = 0
    failed_days = 0

    for date, day_data in daily_groups:
        opening_bars = day_data.iloc[:n_bars]

        if len(opening_bars) < n_bars:
            failed_days += 1
            continue

        opening_high_list.append(opening_bars['High'].max())
        opening_low_list.append(opening_bars['Low'].min())
        opening_close_list.append(opening_bars['Close'].iloc[-1])
        opening_open_list.append(opening_bars['Open'].iloc[0])
        dates.append(pd.Timestamp(date, tz=data_5min.index.tz))
        successful_days += 1

    print(f"\nOpening range calculation results:")
    print(f"  Successful days: {successful_days}")
    print(f"  Failed days (insufficient bars): {failed_days}")

    if successful_days == 0:
        print("[CRITICAL] No opening ranges calculated! Check data.")
        return None

    # Create Series
    opening_high = pd.Series(opening_high_list, index=dates)
    opening_low = pd.Series(opening_low_list, index=dates)
    opening_close = pd.Series(opening_close_list, index=dates)
    opening_open = pd.Series(opening_open_list, index=dates)

    # Broadcast to intraday using map by date pattern (FIXED)
    opening_high_dict = dict(zip(opening_high.index.date, opening_high.values))
    opening_low_dict = dict(zip(opening_low.index.date, opening_low.values))
    opening_close_dict = dict(zip(opening_close.index.date, opening_close.values))
    opening_open_dict = dict(zip(opening_open.index.date, opening_open.values))

    intraday_dates = pd.Series(data_5min.index.date, index=data_5min.index)
    opening_high_ff = intraday_dates.map(opening_high_dict)
    opening_low_ff = intraday_dates.map(opening_low_dict)
    opening_close_ff = intraday_dates.map(opening_close_dict)
    opening_open_ff = intraday_dates.map(opening_open_dict)

    # Check for NaN values after broadcast
    nan_count = opening_high_ff.isna().sum()
    print(f"  NaN values after broadcast: {nan_count} (should be 0)")

    print(f"[SUCCESS] Opening range calculated and broadcast to {len(opening_high_ff)} intraday bars")

    return {
        'opening_high': opening_high_ff,
        'opening_low': opening_low_ff,
        'opening_close': opening_close_ff,
        'opening_open': opening_open_ff
    }


def analyze_signal_conditions(data_5min, data_daily, opening_range):
    """Decompose signal conditions and count TRUE occurrences"""
    print_section("STEP 3: SIGNAL CONDITION DECOMPOSITION")

    opening_high = opening_range['opening_high']
    opening_low = opening_range['opening_low']
    opening_close = opening_range['opening_close']
    opening_open = opening_range['opening_open']

    # Condition 1: Price breakout
    print("\n--- Condition 1: Price Breakout ---")
    price_breakout_long = data_5min['Close'] > opening_high
    price_breakout_short = data_5min['Close'] < opening_low

    print(f"Long breakouts (Close > opening_high): {price_breakout_long.sum()} bars")
    print(f"Short breakouts (Close < opening_low): {price_breakout_short.sum()} bars")

    if price_breakout_long.sum() > 0:
        sample_dates_long = data_5min.index[price_breakout_long][:5]
        print(f"Sample long breakout dates: {list(sample_dates_long)}")

    # Condition 2: Directional bias
    print("\n--- Condition 2: Directional Bias ---")
    bullish_opening = opening_close > opening_open
    bearish_opening = opening_close < opening_open

    print(f"Bullish openings (close > open): {bullish_opening.sum()} bars")
    print(f"Bearish openings (close < open): {bearish_opening.sum()} bars")

    # Condition 3: Volume confirmation
    print("\n--- Condition 3: Volume Confirmation (2.0x) ---")
    volume_ma = data_5min['Volume'].rolling(window=20).mean()
    volume_surge = data_5min['Volume'] > (volume_ma * 2.0)

    print(f"Volume surge (>2.0x MA): {volume_surge.sum()} bars")
    print(f"Volume surge rate: {volume_surge.sum() / len(volume_surge) * 100:.1f}%")

    # Check max volume multiplier
    volume_multiplier = data_5min['Volume'] / volume_ma
    max_mult = volume_multiplier.max()
    print(f"Max volume multiplier achieved: {max_mult:.2f}x")

    if volume_surge.sum() > 0:
        sample_dates_vol = data_5min.index[volume_surge][:5]
        print(f"Sample volume surge dates: {list(sample_dates_vol)}")

    # Condition 4: Time filter
    print("\n--- Condition 4: Time Filter (after 10:00 AM) ---")
    market_open = datetime.strptime("09:30", "%H:%M")
    entry_start = market_open + timedelta(minutes=30)
    entry_start_time = entry_start.time()

    can_enter = data_5min.index.time >= entry_start_time
    print(f"Bars after 10:00 AM: {can_enter.sum()} bars")
    print(f"Time filter rate: {can_enter.sum() / len(can_enter) * 100:.1f}%")

    # Combined conditions
    print("\n" + "=" * 70)
    print("COMBINED SIGNAL ANALYSIS")
    print("=" * 70)

    long_entries = (
        price_breakout_long &
        bullish_opening &
        volume_surge &
        can_enter
    )

    print(f"\nFinal long entry signals (ALL conditions): {long_entries.sum()}")

    if long_entries.sum() > 0:
        print(f"[SUCCESS] Found {long_entries.sum()} valid entry signals")
        sample_entry_dates = data_5min.index[long_entries][:5]
        print(f"Sample entry dates: {list(sample_entry_dates)}")
    else:
        print("[CRITICAL] ZERO trades - analyzing bottleneck...")

        # Find which combination works
        print("\nCombination analysis:")
        combo1 = price_breakout_long & bullish_opening
        combo2 = price_breakout_long & volume_surge
        combo3 = bullish_opening & volume_surge
        combo4 = price_breakout_long & bullish_opening & can_enter
        combo5 = price_breakout_long & volume_surge & can_enter

        print(f"  Price + Direction: {combo1.sum()} bars")
        print(f"  Price + Volume: {combo2.sum()} bars")
        print(f"  Direction + Volume: {combo3.sum()} bars")
        print(f"  Price + Direction + Time: {combo4.sum()} bars")
        print(f"  Price + Volume + Time: {combo5.sum()} bars")

        # Identify the bottleneck
        if combo2.sum() == 0:
            print("\n[BOTTLENECK IDENTIFIED] Price breakouts NEVER coincide with volume surges")
            print("Likely cause: 2.0x volume threshold too strict for Jan 2024 SPY")
        elif combo1.sum() == 0:
            print("\n[BOTTLENECK IDENTIFIED] Price breakouts NEVER occur on bullish opening days")
            print("Likely cause: Directional bias filter too restrictive")
        else:
            print("\n[BOTTLENECK IDENTIFIED] Multi-condition intersection empty")

    return {
        'price_breakout_long': price_breakout_long,
        'bullish_opening': bullish_opening,
        'volume_surge': volume_surge,
        'can_enter': can_enter,
        'long_entries': long_entries
    }


def main():
    """Run complete diagnostic"""
    print("\n" + "=" * 70)
    print("ORB STRATEGY DIAGNOSTIC: JANUARY 2024 SIGNAL DECOMPOSITION")
    print("=" * 70)
    print("\nObjective: Identify why ORB executed 0 trades in Jan 2024")
    print("Method: Calculate each signal condition separately\n")

    # Step 1: Fetch data
    data_5min, data_daily = fetch_data_jan2024()

    if data_5min is None:
        print("\n[FAILED] Cannot fetch data - aborting diagnostic")
        return

    # Step 2: Calculate opening range
    opening_range = calculate_opening_range(data_5min)

    if opening_range is None:
        print("\n[FAILED] Opening range calculation failed - aborting diagnostic")
        return

    # Step 3: Analyze signal conditions
    signals = analyze_signal_conditions(data_5min, data_daily, opening_range)

    # Final summary
    print_section("DIAGNOSTIC SUMMARY")

    print("\nSignal Condition Results:")
    print(f"  [1] Price breakouts: {signals['price_breakout_long'].sum()} bars")
    print(f"  [2] Bullish openings: {signals['bullish_opening'].sum()} bars")
    print(f"  [3] Volume surges (2.0x): {signals['volume_surge'].sum()} bars")
    print(f"  [4] Time filter (>=10:00 AM): {signals['can_enter'].sum()} bars")
    print(f"  [COMBINED] Final entries: {signals['long_entries'].sum()} bars")

    if signals['long_entries'].sum() == 0:
        print("\n[RESULT] ZERO TRADES CONFIRMED")
        print("\nRecommended Actions:")
        print("1. Review volume threshold (2.0x may be too strict)")
        print("2. Check if opening range calculation is correct")
        print("3. Verify directional bias is not over-filtering")
        print("4. Consider testing with 1.5x volume threshold (diagnostic only)")
    else:
        print(f"\n[RESULT] Found {signals['long_entries'].sum()} valid entry signals")
        print("Strategy should execute trades - investigate backtest logic")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
