"""
Parse user's spreadsheet with proper headers and extract key findings.
"""

import pandas as pd
import numpy as np

excel_path = r'c:\Users\sheeh\Downloads\Credit spread strategies.xlsx'

print("="*80)
print("DETAILED ANALYSIS - USER'S CREDIT SPREAD STRATEGY SPREADSHEET")
print("="*80)
print()

# Read Summary with proper parsing
print("[1] SUMMARY TAB - Key Results")
print("="*80)

summary = pd.read_excel(excel_path, sheet_name='Summary', header=None)

print("\nRaw Summary Data:")
print(summary.iloc[0:10].to_string())
print()

# Extract key metrics from Summary (row 2 has headers, rows 3-9 have data)
print("\nKEY PERFORMANCE METRICS (1997-2025):")
print("-"*80)

strategies = {
    'TQQQ -35/+40': summary.iloc[3, 1:9].values,
    'SPXL -35/+40': summary.iloc[4, 1:9].values,
    'TQQQ -30/+35': summary.iloc[5, 1:9].values,
    'TQQQ B&H': summary.iloc[6, 1:9].values,
    'SPXL B&H': summary.iloc[7, 1:9].values,
    'SP500 B&H': summary.iloc[8, 1:9].values,
    'QQQ B&H': summary.iloc[9, 1:9].values,
}

headers = ['Return on $1', '% Gain', 'Return 1997-2009', 'Return 2018-2023',
           'DD 2022', 'DD 2020', 'DD 2008', 'DD 2000']

print(f"\n{'Strategy':<20} {'Return':<12} {'% Gain':<10} {'DD 2000':<10} {'DD 2008':<10} {'DD 2020':<10} {'DD 2022':<10}")
print("-"*95)

for name, values in strategies.items():
    return_val = values[0] if not pd.isna(values[0]) else 'N/A'
    pct_gain = values[1] if not pd.isna(values[1]) else 'N/A'
    dd_2000 = values[7] if not pd.isna(values[7]) else 'N/A'
    dd_2008 = values[6] if not pd.isna(values[6]) else 'N/A'
    dd_2020 = values[5] if not pd.isna(values[5]) else 'N/A'
    dd_2022 = values[4] if not pd.isna(values[4]) else 'N/A'

    print(f"{name:<20} {str(return_val):<12} {str(pct_gain):<10} {str(dd_2000):<10} {str(dd_2008):<10} {str(dd_2020):<10} {str(dd_2022):<10}")

print()
print("="*80)
print("[2] SPXL DETAILED ANALYSIS")
print("="*80)

# Read SPXL sheet with proper header
spxl = pd.read_excel(excel_path, sheet_name='SPXL -35%,+40%', header=6)

# Clean up column names
spxl.columns = ['Empty1', 'Empty2', 'Empty3', 'Date', 'Close', 'Change', 'SPXL Change', 'Return on $1', 'Empty4', 'Empty5']

# Drop empty columns and rows
spxl = spxl[['Date', 'Close', 'Change', 'SPXL Change', 'Return on $1']].dropna(subset=['Date'])

print(f"\nData Range:")
print(f"  Start: {spxl.iloc[0]['Date']}")
print(f"  End: {spxl.iloc[-1]['Date']}")
print(f"  Total days: {len(spxl)}")

print(f"\nPerformance:")
print(f"  Starting value: ${spxl.iloc[0]['Return on $1']:.2f}")
print(f"  Ending value: ${spxl.iloc[-1]['Return on $1']:.2f}")
print(f"  Multiple: {spxl.iloc[-1]['Return on $1'] / spxl.iloc[0]['Return on $1']:.2f}x")

print(f"\nFirst 10 trading days:")
print(spxl.head(10).to_string(index=False))

print(f"\nLast 10 trading days:")
print(spxl.tail(10).to_string(index=False))

# Find major drawdowns
print(f"\nDrawdown Analysis:")
spxl['Peak'] = spxl['Return on $1'].cummax()
spxl['Drawdown'] = (spxl['Return on $1'] / spxl['Peak']) - 1

max_dd = spxl['Drawdown'].min()
max_dd_date = spxl.loc[spxl['Drawdown'].idxmin(), 'Date']

print(f"  Maximum Drawdown: {max_dd:.2%}")
print(f"  Date of Max DD: {max_dd_date}")

# 2000 crash period
crash_2000 = spxl[(spxl['Date'] >= '2000-01-01') & (spxl['Date'] <= '2003-01-01')]
if len(crash_2000) > 0:
    dd_2000 = crash_2000['Drawdown'].min()
    print(f"  2000 Crash DD: {dd_2000:.2%}")

# 2008 crash period
crash_2008 = spxl[(spxl['Date'] >= '2008-01-01') & (spxl['Date'] <= '2009-03-31')]
if len(crash_2008) > 0:
    dd_2008 = crash_2008['Drawdown'].min()
    print(f"  2008 Crash DD: {dd_2008:.2%}")

# 2020 crash period
crash_2020 = spxl[(spxl['Date'] >= '2020-01-01') & (spxl['Date'] <= '2020-06-30')]
if len(crash_2020) > 0:
    dd_2020 = crash_2020['Drawdown'].min()
    print(f"  2020 Crash DD: {dd_2020:.2%}")

# 2022 bear market
crash_2022 = spxl[(spxl['Date'] >= '2022-01-01') & (spxl['Date'] <= '2022-12-31')]
if len(crash_2022) > 0:
    dd_2022 = crash_2022['Drawdown'].min()
    print(f"  2022 Bear DD: {dd_2022:.2%}")

print()
print("="*80)
print("[3] CRITICAL COMPARISON")
print("="*80)

print("\nUSER'S SPREADSHEET (1997-2025 simulated 3x):")
print("  SPXL Strategy (-35/+40): 328x return")
print("  SPXL Buy-and-Hold: 20x return")
print("  Strategy BEATS B&H by: 1540% (16.4x better!)")
print()

print("OUR BACKTEST (2008-2025 actual SPXL):")
print("  SPXL Strategy: 22.62x return")
print("  SPXL Buy-and-Hold: 64.75x return")
print("  Strategy LOSES to B&H by: -65%")
print()

print("WHY THE MASSIVE DIFFERENCE?")
print("  1. Time period: 1997 start vs 2008 start (includes 2000 crash)")
print("  2. Methodology: Simulated 3x vs Actual SPXL")
print("  3. Entry/exit: -35/+40 thresholds vs FRED credit spread signals")
print("  4. Starting at 2000 crash vs 2008 bottom")
print()

print("="*80)
