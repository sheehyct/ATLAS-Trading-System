"""
Analyze user's credit spread strategy Excel spreadsheet.

This uses a different methodology than our SPXL backtest:
- Simulates 3x leverage by multiplying daily SPY/QQQ changes by 3
- Goes back to 1997 (includes 2000 crash + 2008 crash)
- Tests -35% entry / +40% exit thresholds

Purpose: Compare methodologies and reconcile different conclusions.
"""

import pandas as pd
import numpy as np
import os

# File path
excel_path = r'c:\Users\sheeh\Downloads\Credit spread strategies.xlsx'

print("="*80)
print("CREDIT SPREAD STRATEGY - USER SPREADSHEET ANALYSIS")
print("="*80)
print()

# Read all sheets
print("[1] Reading Excel file...")
try:
    # Get sheet names first
    xl_file = pd.ExcelFile(excel_path)
    sheet_names = xl_file.sheet_names
    print(f"  Found {len(sheet_names)} sheets:")
    for i, name in enumerate(sheet_names, 1):
        print(f"    {i}. {name}")
    print()

    # Read each sheet
    sheets = {}
    for name in sheet_names:
        print(f"  Reading sheet: {name}")
        df = pd.read_excel(excel_path, sheet_name=name)
        sheets[name] = df
        print(f"    Shape: {df.shape} (rows x columns)")
        print(f"    Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        print()

except Exception as e:
    print(f"  ERROR reading file: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("="*80)
print("SHEET 1: SUMMARY")
print("="*80)
print()

# Analyze Summary tab
if 'Summary' in sheets or sheet_names[0]:
    summary_name = 'Summary' if 'Summary' in sheets else sheet_names[0]
    summary = sheets[summary_name]

    print("First 20 rows of Summary:")
    print(summary.head(20).to_string())
    print()

    # Look for key metrics
    print("\nKey metrics found in Summary:")
    for col in summary.columns:
        if any(keyword in str(col).lower() for keyword in ['return', 'performance', 'gain', 'result', 'multiple']):
            print(f"  Column: {col}")
            non_null = summary[col].dropna()
            if len(non_null) > 0:
                print(f"    Sample values: {non_null.head(5).tolist()}")
    print()

print("="*80)
print("SHEET 2: STRATEGY APPLIED TO TQQQ vs QQQ")
print("="*80)
print()

# Analyze TQQQ sheet
tqqq_sheet_name = sheet_names[1] if len(sheet_names) > 1 else None
if tqqq_sheet_name:
    tqqq = sheets[tqqq_sheet_name]

    print(f"Sheet name: {tqqq_sheet_name}")
    print(f"Shape: {tqqq.shape}")
    print()

    print("First 10 rows:")
    print(tqqq.head(10).to_string())
    print()

    print("Last 10 rows:")
    print(tqqq.tail(10).to_string())
    print()

    print("Column names:")
    for i, col in enumerate(tqqq.columns):
        print(f"  {i}: {col}")
    print()

    # Look for results
    print("\nLooking for performance results...")
    for col in tqqq.columns:
        if any(keyword in str(col).lower() for keyword in ['result', 'return', 'gain', 'multiple', 'final', 'total']):
            print(f"\n  Column '{col}':")
            non_null = tqqq[col].dropna()
            if len(non_null) > 0:
                print(f"    First value: {non_null.iloc[0]}")
                print(f"    Last value: {non_null.iloc[-1]}")
                print(f"    Max value: {non_null.max()}")
                print(f"    Min value: {non_null.min()}")

print()
print("="*80)
print("SHEET 3: STRATEGY APPLIED TO SPXL vs SPY")
print("="*80)
print()

# Analyze SPXL sheet
spxl_sheet_name = sheet_names[2] if len(sheet_names) > 2 else None
if spxl_sheet_name:
    spxl = sheets[spxl_sheet_name]

    print(f"Sheet name: {spxl_sheet_name}")
    print(f"Shape: {spxl.shape}")
    print()

    print("First 10 rows:")
    print(spxl.head(10).to_string())
    print()

    print("Last 10 rows:")
    print(spxl.tail(10).to_string())
    print()

    print("Column names:")
    for i, col in enumerate(spxl.columns):
        print(f"  {i}: {col}")
    print()

    # Look for results
    print("\nLooking for performance results...")
    for col in spxl.columns:
        if any(keyword in str(col).lower() for keyword in ['result', 'return', 'gain', 'multiple', 'final', 'total']):
            print(f"\n  Column '{col}':")
            non_null = spxl[col].dropna()
            if len(non_null) > 0:
                print(f"    First value: {non_null.iloc[0]}")
                print(f"    Last value: {non_null.iloc[-1]}")
                print(f"    Max value: {non_null.max()}")
                print(f"    Min value: {non_null.min()}")

print()
print("="*80)
print("METHODOLOGY COMPARISON")
print("="*80)
print()

print("USER'S APPROACH (from spreadsheet):")
print("  - Daily SPY/QQQ prices since 1997")
print("  - Multiply daily % change by 3 to simulate 3x leverage")
print("  - Apply -35% entry / +40% exit credit spread signals")
print("  - Start with $1 to track returns")
print("  - Includes 2000 dot-com crash + 2008 financial crisis")
print()

print("OUR APPROACH (from previous backtest):")
print("  - Actual SPXL prices (Nov 2008 - Nov 2025)")
print("  - Real volatility decay from daily rebalancing")
print("  - Credit spread signals from FRED data")
print("  - Started at 2008 market bottom (lucky timing)")
print("  - Result: Strategy 22.62x vs SPXL B&H 64.75x (-65% underperformance)")
print()

print("KEY DIFFERENCES:")
print("  1. Time period: 1997-2025 (user) vs 2008-2025 (ours)")
print("  2. Methodology: Simulated 3x (user) vs Actual SPXL (ours)")
print("  3. Entry/exit: -35/+40 thresholds (user) vs FRED signals (ours)")
print("  4. Includes crashes: 2000 + 2008 (user) vs only 2020/2022 (ours)")
print()

print("="*80)
