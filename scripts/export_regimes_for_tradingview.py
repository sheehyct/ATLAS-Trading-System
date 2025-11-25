"""
Export ATLAS Regime Detection to TradingView-Compatible CSV

Generates CSV files that can be imported into TradingView as custom data.
Works with TradingView Premium/Pro accounts that support custom data import.

Usage:
    python export_regimes_for_tradingview.py

Output:
    - SPY_regimes_tradingview.csv (TradingView import format)
    - QQQ_regimes_tradingview.csv (TradingView import format)
    - regimes_combined_tradingview.csv (multi-symbol)

TradingView Import Steps:
    1. Chart Settings → Symbol → Compare or Add
    2. Import → Upload CSV
    3. Select the generated CSV file
    4. Use Pine Script to read the data and color background

Pine Script Example (see comments at end of this file)
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from datetime import datetime

from regime.academic_jump_model import AcademicJumpModel
from regime.vix_acceleration import fetch_vix_data


def export_regimes_for_tradingview(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2025-11-14',
    output_file=None
):
    """
    Export regime classifications to TradingView-compatible CSV.

    Parameters
    ----------
    symbol : str
        Stock symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    output_file : str, optional
        Output CSV filename (default: {symbol}_regimes_tradingview.csv)

    Returns
    -------
    pd.DataFrame
        Exported regime data
    """
    print(f"\n{'='*70}")
    print(f"EXPORTING ATLAS REGIMES FOR TRADINGVIEW: {symbol}")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}\n")

    # Step 1: Fetch data
    print("[1/4] Fetching market data...")
    data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()
    vix_data = fetch_vix_data(start_date, end_date)
    print(f"   Fetched {len(data)} {symbol} bars")

    # Step 2: Run regime detection
    print("\n[2/4] Running regime detection...")
    model = AcademicJumpModel(lambda_penalty=1.5)
    train_size = int(len(data) * 0.6)
    model.fit(data.iloc[:train_size], n_starts=3, random_seed=42)

    lookback = min(252, len(data) - 100)
    regimes, _, _ = model.online_inference(data, lookback=lookback, vix_data=vix_data)
    print(f"   Classified {len(regimes)} days into regimes")

    # Regime distribution
    regime_counts = regimes.value_counts()
    print(f"\n   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = (count / len(regimes)) * 100
        print(f"      {regime}: {count} days ({pct:.1f}%)")

    # Step 3: Convert to TradingView format
    print("\n[3/4] Converting to TradingView format...")

    # Map regimes to numeric values for TradingView
    regime_map = {
        'CRASH': 0,
        'TREND_BEAR': 1,
        'TREND_NEUTRAL': 2,
        'TREND_BULL': 3
    }

    # Align with full data range
    full_data = data.copy()
    full_data['regime_numeric'] = np.nan
    full_data.loc[regimes.index, 'regime_numeric'] = regimes.map(regime_map)

    # Forward fill regimes (TradingView needs continuous data)
    full_data['regime_numeric'] = full_data['regime_numeric'].fillna(method='ffill')

    # Backfill the initial NaN values (before first classification)
    full_data['regime_numeric'] = full_data['regime_numeric'].fillna(method='bfill')

    # Add regime names as well
    reverse_map = {v: k for k, v in regime_map.items()}
    full_data['regime_name'] = full_data['regime_numeric'].map(reverse_map)

    # TradingView CSV format: Date, Open, High, Low, Close, Volume, Regime
    # We'll use the regime as a custom indicator column
    tv_export = pd.DataFrame({
        'time': full_data.index.strftime('%Y-%m-%d'),
        'open': full_data['Open'],
        'high': full_data['High'],
        'low': full_data['Low'],
        'close': full_data['Close'],
        'volume': full_data['Volume'],
        'regime': full_data['regime_numeric'].astype(int),
        'regime_name': full_data['regime_name']
    })

    # Step 4: Save to CSV (organized in visualization directory)
    if output_file is None:
        output_file = f'visualization/tradingview_exports/{symbol}_regimes_tradingview.csv'

    tv_export.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")
    print(f"   Rows: {len(tv_export)}")
    print(f"   Columns: {list(tv_export.columns)}")

    # Summary
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated file: {output_file}")
    print(f"\nRegime Encoding:")
    print(f"  0 = CRASH (Red)")
    print(f"  1 = BEAR (Orange)")
    print(f"  2 = NEUTRAL (Gray)")
    print(f"  3 = BULL (Green)")
    print(f"\nTradingView Import Instructions:")
    print(f"  1. Open TradingView chart for {symbol}")
    print(f"  2. Click 'Import' button (bottom toolbar)")
    print(f"  3. Upload {output_file}")
    print(f"  4. Chart will show OHLCV data with regime column")
    print(f"  5. Use Pine Script indicator below to color background")
    print(f"{'='*70}\n")

    return tv_export


def export_combined_regimes(symbols=['SPY', 'QQQ'], start_date='2020-01-01', end_date='2025-11-14'):
    """
    Export regimes for multiple symbols in a single comparison file.

    Parameters
    ----------
    symbols : list
        List of symbols to export
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    dict
        Dictionary of DataFrames by symbol
    """
    print(f"\n{'='*70}")
    print(f"EXPORTING COMBINED REGIMES FOR: {', '.join(symbols)}")
    print(f"{'='*70}\n")

    results = {}

    for symbol in symbols:
        df = export_regimes_for_tradingview(symbol, start_date, end_date)
        results[symbol] = df

    # Also create a comparison CSV with just dates and regimes
    print(f"\n{'='*70}")
    print("CREATING REGIME COMPARISON FILE")
    print(f"{'='*70}\n")

    # Start with first symbol's dates
    base_symbol = symbols[0]
    comparison = pd.DataFrame({
        'date': results[base_symbol]['time']
    })

    # Add regime columns for each symbol
    for symbol in symbols:
        comparison[f'{symbol}_regime'] = results[symbol]['regime'].values
        comparison[f'{symbol}_regime_name'] = results[symbol]['regime_name'].values

    comparison_file = 'visualization/tradingview_exports/regimes_combined_tradingview.csv'
    comparison.to_csv(comparison_file, index=False)

    print(f"   Saved: {comparison_file}")
    print(f"   Rows: {len(comparison)}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"{'='*70}\n")

    return results


# Pine Script Template for TradingView
PINE_SCRIPT_TEMPLATE = """
// ============================================================================
// ATLAS Regime Detection - TradingView Pine Script v6
// ============================================================================
// Import Instructions:
// 1. Save this code as a new Pine Script indicator in TradingView
// 2. Import your CSV file using Chart → Import
// 3. The indicator will read the 'regime' column and color the background
// ============================================================================

//@version=6
indicator("ATLAS Regime Detection", overlay=true)

// ============================================================================
// CONFIGURATION
// ============================================================================

// Color settings (matching Python visualizations)
colorCrash = color.new(#DC3545, 85)      // Red (15% opacity)
colorBear = color.new(#FD7E14, 88)       // Orange (12% opacity)
colorNeutral = color.new(#6C757D, 92)    // Gray (8% opacity)
colorBull = color.new(#28A745, 85)       // Green (15% opacity)

// Marker settings
showVixMarkers = input.bool(true, "Show VIX Spike Markers")
vixThreshold1d = input.float(0.20, "VIX 1-Day Threshold (%)", step=0.01)
vixThreshold3d = input.float(0.50, "VIX 3-Day Threshold (%)", step=0.01)

// ============================================================================
// DATA IMPORT
// ============================================================================

// NOTE: After importing your CSV, TradingView creates a custom ticker
// Replace "IMPORTED_DATA" below with your actual imported data ticker
// Example: If you imported SPY_regimes_tradingview.csv,
// it might be named "USER:SPY_REGIMES" or similar

// Get regime data from imported CSV
// IMPORTANT: Update this line to match your imported data ticker name
regimeData = request.security("IMPORTED_DATA", timeframe.period, close[6])  // Regime is column 6

// If using TradingView's custom data import, the regime value will be in a specific column
// Adjust the column index based on your CSV structure:
// Column 0: time
// Column 1: open
// Column 2: high
// Column 3: low
// Column 4: close
// Column 5: volume
// Column 6: regime  <-- This is what we want

// ============================================================================
// VIX FLASH CRASH DETECTION (Optional - Real-time)
// ============================================================================

vixClose = request.security("VIX", timeframe.period, close)
vixChange1d = ta.change(vixClose) / vixClose[1]
vixChange3d = (vixClose - vixClose[3]) / vixClose[3]
vixSpike = (vixChange1d > vixThreshold1d) or (vixChange3d > vixThreshold3d)

// ============================================================================
// REGIME BACKGROUND COLORING
// ============================================================================

regimeColor = regimeData == 0 ? colorCrash :      // CRASH
              regimeData == 1 ? colorBear :        // BEAR
              regimeData == 2 ? colorNeutral :     // NEUTRAL
              colorBull                             // BULL

bgcolor(regimeColor, title="Regime Background")

// ============================================================================
// VIX SPIKE MARKERS
// ============================================================================

plotshape(
    showVixMarkers and vixSpike,
    style=shape.xcross,
    location=location.abovebar,
    color=color.new(#DC3545, 0),
    size=size.small,
    title="VIX Flash Crash"
)

// ============================================================================
// REGIME LEGEND
// ============================================================================

var table legend = table.new(position.top_right, 2, 5,
                              border_width=1,
                              border_color=color.gray)

if barstate.islast
    // Header
    table.cell(legend, 0, 0, "ATLAS",
               bgcolor=color.new(color.white, 0),
               text_color=color.black,
               text_size=size.normal)

    // Regime labels
    table.cell(legend, 0, 1, "CRASH",
               bgcolor=color.new(#DC3545, 70),
               text_color=color.white,
               text_size=size.small)
    table.cell(legend, 0, 2, "BEAR",
               bgcolor=color.new(#FD7E14, 70),
               text_color=color.white,
               text_size=size.small)
    table.cell(legend, 0, 3, "NEUTRAL",
               bgcolor=color.new(#6C757D, 70),
               text_color=color.white,
               text_size=size.small)
    table.cell(legend, 0, 4, "BULL",
               bgcolor=color.new(#28A745, 70),
               text_color=color.white,
               text_size=size.small)

// ============================================================================
// NOTES
// ============================================================================
// 1. Update "IMPORTED_DATA" ticker name to match your CSV import
// 2. Verify column index for regime data (default is column 6)
// 3. Adjust VIX thresholds if needed (defaults: 20% 1-day, 50% 3-day)
// 4. Legend shows in top-right corner
// 5. Background colors match Python visualizations
// ============================================================================
"""


if __name__ == '__main__':
    print("="*70)
    print("ATLAS REGIME EXPORT FOR TRADINGVIEW")
    print("="*70)

    # Export individual symbols
    spy_data = export_regimes_for_tradingview('SPY', '2020-01-01', '2025-11-14')
    qqq_data = export_regimes_for_tradingview('QQQ', '2020-01-01', '2025-11-14')

    # Export combined comparison
    export_combined_regimes(['SPY', 'QQQ'], '2020-01-01', '2025-11-14')

    # Save Pine Script template (organized in visualization directory)
    print("\n" + "="*70)
    print("SAVING PINE SCRIPT TEMPLATE")
    print("="*70)

    with open('visualization/tradingview_exports/ATLAS_TradingView_Indicator.pine', 'w') as f:
        f.write(PINE_SCRIPT_TEMPLATE)

    print("\n   Saved: visualization/tradingview_exports/ATLAS_TradingView_Indicator.pine")
    print("   Copy this code into TradingView Pine Editor")
    print("   Update 'IMPORTED_DATA' ticker name to match your import")

    print("\n" + "="*70)
    print("ALL EXPORTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. visualization/tradingview_exports/SPY_regimes_tradingview.csv")
    print("  2. visualization/tradingview_exports/QQQ_regimes_tradingview.csv")
    print("  3. visualization/tradingview_exports/regimes_combined_tradingview.csv")
    print("  4. visualization/tradingview_exports/ATLAS_TradingView_Indicator.pine")
    print("\nNext steps:")
    print("  1. Import CSV files to TradingView (Chart → Import)")
    print("  2. Copy Pine Script to TradingView Editor")
    print("  3. Update ticker name in Pine Script to match your import")
    print("  4. Apply indicator to chart")
    print("="*70 + "\n")
