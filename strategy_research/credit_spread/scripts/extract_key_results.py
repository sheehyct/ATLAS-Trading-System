"""
Extract key results from user's spreadsheet - simple and clear.
"""

import pandas as pd

excel_path = r'c:\Users\sheeh\Downloads\Credit spread strategies.xlsx'

print("="*80)
print("USER'S CREDIT SPREAD ANALYSIS - KEY RESULTS")
print("="*80)
print()

# Read Summary
summary = pd.read_excel(excel_path, sheet_name='Summary', header=None)

print("PERFORMANCE SUMMARY (1997-2025):")
print("-"*80)
print()

# Row 4: TQQQ -35/+40
print("TQQQ Strategy (-35% entry, +40% exit):")
print(f"  Return on $1: {summary.iloc[4, 1]}")
print(f"  Max DD 2000: {summary.iloc[4, 8]}")
print(f"  Max DD 2008: {summary.iloc[4, 7]}")
print(f"  Max DD 2020: {summary.iloc[4, 6]}")
print(f"  Max DD 2022: {summary.iloc[4, 5]}")
print()

# Row 5: SPXL -35/+40
print("SPXL Strategy (-35% entry, +40% exit):")
print(f"  Return on $1: {summary.iloc[5, 1]}")
print(f"  Max DD 2000: {summary.iloc[5, 8]}")
print(f"  Max DD 2008: {summary.iloc[5, 7]}")
print(f"  Max DD 2020: {summary.iloc[5, 6]}")
print(f"  Max DD 2022: {summary.iloc[5, 5]}")
print()

# Row 7: TQQQ B&H
print("TQQQ Buy-and-Hold:")
print(f"  Return on $1: {summary.iloc[7, 1]}")
print(f"  Max DD 2000: {summary.iloc[7, 8]}")
print(f"  Max DD 2008: {summary.iloc[7, 7]}")
print(f"  Max DD 2020: {summary.iloc[7, 6]}")
print(f"  Max DD 2022: {summary.iloc[7, 5]}")
print()

# Row 8: SPXL B&H
print("SPXL Buy-and-Hold (simulated 3x SPY):")
print(f"  Return on $1: {summary.iloc[8, 1]}")
print(f"  Max DD 2000: {summary.iloc[8, 8]}")
print(f"  Max DD 2008: {summary.iloc[8, 7]}")
print(f"  Max DD 2020: {summary.iloc[8, 6]}")
print(f"  Max DD 2022: {summary.iloc[8, 5]}")
print()

# Row 9: SPY B&H
print("SPY Buy-and-Hold:")
print(f"  Return on $1: {summary.iloc[9, 1]}")
print(f"  Max DD 2000: {summary.iloc[9, 8]}")
print(f"  Max DD 2008: {summary.iloc[9, 7]}")
print(f"  Max DD 2020: {summary.iloc[9, 6]}")
print(f"  Max DD 2022: {summary.iloc[9, 5]}")
print()

print("="*80)
print("CRITICAL COMPARISON")
print("="*80)
print()

spxl_strategy = float(summary.iloc[5, 1])
spxl_bh = float(summary.iloc[8, 1])
outperformance = (spxl_strategy / spxl_bh - 1) * 100

print(f"SPXL STRATEGY vs BUY-AND-HOLD (1997-2025):")
print(f"  Strategy: {spxl_strategy}x return")
print(f"  Buy-and-Hold: {spxl_bh}x return")
print(f"  Strategy BEATS B&H by: {outperformance:.1f}%")
print()

print("vs OUR PREVIOUS ANALYSIS (2008-2025 actual SPXL):")
print(f"  Our Strategy: 22.62x return")
print(f"  Our Buy-and-Hold: 64.75x return")
print(f"  Our Strategy LOSES to B&H by: -65%")
print()

print("WHY THE OPPOSITE CONCLUSIONS?")
print("  1. TIME PERIOD:")
print("     User: 1997-2025 (28 years, includes 2000 + 2008 crashes)")
print("     Ours: 2008-2025 (17 years, started at 2008 bottom)")
print()
print("  2. METHODOLOGY:")
print("     User: Simulated 3x by multiplying SPY daily % changes")
print("     Ours: Actual SPXL price data (includes real decay)")
print()
print("  3. SIGNAL GENERATION:")
print("     User: -35% credit spread decline, +40% credit spread rise")
print("     Ours: FRED credit spread with EMA thresholds")
print()
print("  4. CRASH INCLUSION:")
print("     User: 2000 dot-com crash HELPED strategy (-4% DD vs -91% B&H)")
print("     Ours: Started AFTER 2008 crash (bought at bottom)")
print()

print("="*80)
print("KEY INSIGHT")
print("="*80)
print()
print("The strategy WORKS when it avoids MAJOR CRASHES:")
print("  - 2000 crash: SPXL B&H would lose -91%, strategy only -4%")
print("  - 2008 crash: SPXL B&H would lose -96%, strategy no loss")
print()
print("The strategy FAILS when it starts AFTER a crash:")
print("  - Our backtest started Nov 2008 (market bottom)")
print("  - No major crash to avoid (2020 COVID was brief, already recovered)")
print("  - Time out of market (41%) just misses bull run gains")
print()
print("CONCLUSION:")
print("  Credit spread strategy is CRASH PROTECTION, not bull market optimizer")
print("  It beats buy-and-hold over FULL CYCLES (with crashes)")
print("  It loses to buy-and-hold during RECOVERY PERIODS (post-crash)")
print()

print("="*80)
