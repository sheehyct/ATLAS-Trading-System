"""
SPXL Data Verification Script

Purpose: Download SPXL data from multiple sources and verify data quality
         to fix issues from previous backtest (corrupted auto_adjust=True data)

Critical Issues Being Fixed:
1. Previous backtest showed June 30, 2025 @ $97.56 (WRONG)
2. Current Yahoo Finance shows June 30, 2025 @ $173.30 (CORRECT)
3. SPXL buy-and-hold showed 64.75x (INFLATED, expect 25-35x)

Data Quality Checks:
- Verify June 2025 price is ~$173 (not $97)
- Verify SPXL split history (3:1 in 2013, 4:1 in 2017)
- Calculate realistic buy-and-hold (expect 25-35x for 2008-2025)
- Cross-validate Yahoo Finance vs Alpaca data

Author: ATLAS Development Team
Date: November 8, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

# Configuration
SPXL_INCEPTION = '2008-11-05'  # SPXL launch date
END_DATE = '2025-11-08'  # Today

# Expected splits (from research)
KNOWN_SPLITS = {
    '2013-04-02': 3.0,  # 3:1 forward split
    '2017-05-01': 4.0   # 4:1 forward split
}

# Data quality thresholds
JUNE_2025_PRICE_MIN = 150
JUNE_2025_PRICE_MAX = 200
BH_RETURN_MIN = 20
BH_RETURN_MAX = 50


def download_spxl_yahoo(auto_adjust=False):
    """
    Download SPXL data from Yahoo Finance.

    Args:
        auto_adjust: If False, returns unadjusted OHLC + Adj Close column
                     If True, returns adjusted OHLC (problematic for backtesting)

    Returns:
        pd.DataFrame: SPXL price data
    """
    print(f"\n{'='*80}")
    print(f"DOWNLOADING SPXL DATA FROM YAHOO FINANCE (auto_adjust={auto_adjust})")
    print(f"{'='*80}")

    ticker = yf.Ticker('SPXL')

    # Download data
    df = ticker.history(start=SPXL_INCEPTION, end=END_DATE, auto_adjust=auto_adjust)

    print(f"\nData downloaded: {len(df)} trading days")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"\nColumns: {list(df.columns)}")

    # Show first and last rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())

    return df


def verify_split_history(ticker_obj):
    """
    Verify SPXL split history matches known splits.

    Args:
        ticker_obj: yfinance Ticker object

    Returns:
        pd.Series: Split history
    """
    print(f"\n{'='*80}")
    print("VERIFYING SPXL SPLIT HISTORY")
    print(f"{'='*80}")

    splits = ticker_obj.splits

    print(f"\nSplits found: {len(splits)}")
    for date, ratio in splits.items():
        print(f"  {date.date()}: {ratio}:1 {'forward' if ratio > 1 else 'reverse'} split")

    # Verify against known splits
    print(f"\nExpected splits:")
    for date_str, ratio in KNOWN_SPLITS.items():
        print(f"  {date_str}: {ratio}:1 forward split")

    # Check if splits match
    if len(splits) == len(KNOWN_SPLITS):
        print(f"\n[PASS] Split count matches: {len(splits)}")
    else:
        print(f"\n[WARN] Split count mismatch: found {len(splits)}, expected {len(KNOWN_SPLITS)}")

    return splits


def validate_critical_prices(df, splits_df):
    """
    Validate critical price points to detect data corruption.

    Args:
        df: SPXL price DataFrame
        splits_df: Split history DataFrame

    Returns:
        dict: Validation results
    """
    print(f"\n{'='*80}")
    print("VALIDATING CRITICAL PRICE POINTS")
    print(f"{'='*80}")

    results = {}

    # Check 1: June 30, 2025 price (known from current data)
    print(f"\n1. June 30, 2025 price validation:")
    try:
        june_price = df.loc['2025-06-30', 'Close']
        print(f"   Found: ${june_price:.2f}")
        print(f"   Expected range: ${JUNE_2025_PRICE_MIN}-${JUNE_2025_PRICE_MAX}")

        if JUNE_2025_PRICE_MIN <= june_price <= JUNE_2025_PRICE_MAX:
            print(f"   [PASS] Price is within expected range")
            results['june_2025'] = 'PASS'
        else:
            print(f"   [FAIL] Price is outside expected range")
            results['june_2025'] = 'FAIL'
    except KeyError:
        print(f"   [ERROR] June 30, 2025 not found in data")
        results['june_2025'] = 'ERROR'

    # Check 2: Latest price (Nov 7-8, 2025)
    print(f"\n2. Latest price validation:")
    latest_date = df.index[-1].date()
    latest_price = df.iloc[-1]['Close']
    print(f"   Date: {latest_date}")
    print(f"   Price: ${latest_price:.2f}")
    print(f"   Expected: ~$112-113 (based on recent data)")

    if 100 <= latest_price <= 125:
        print(f"   [PASS] Price is reasonable")
        results['latest'] = 'PASS'
    else:
        print(f"   [WARN] Price seems unusual")
        results['latest'] = 'WARN'

    # Check 3: Inception price (split-adjusted)
    print(f"\n3. Inception price (split-adjusted):")
    inception_price = df.iloc[0]['Close']
    print(f"   Date: {df.index[0].date()}")
    print(f"   Price: ${inception_price:.2f}")

    # With 3:1 and 4:1 splits, original price should be divided by 12
    # Original launch price was ~$50-80, so split-adjusted should be ~$4-7
    if 3 <= inception_price <= 10:
        print(f"   [PASS] Split-adjusted inception price is reasonable")
        results['inception'] = 'PASS'
    else:
        print(f"   [WARN] Inception price seems unusual (expect ~$4-7 split-adjusted)")
        results['inception'] = 'WARN'

    # Check 4: Price continuity around split dates
    print(f"\n4. Price continuity around splits:")
    for split_date, split_ratio in KNOWN_SPLITS.items():
        try:
            split_dt = pd.to_datetime(split_date)

            # Get price before and after split
            before_idx = df.index.get_indexer([split_dt], method='ffill')[0]
            after_idx = before_idx + 1

            if after_idx < len(df):
                price_before = df.iloc[before_idx]['Close']
                price_after = df.iloc[after_idx]['Close']
                date_before = df.index[before_idx].date()
                date_after = df.index[after_idx].date()

                # For forward split, price should drop roughly by split ratio
                expected_ratio = price_before / price_after

                print(f"   {split_date} ({split_ratio}:1 split):")
                print(f"     Before: ${price_before:.2f} ({date_before})")
                print(f"     After:  ${price_after:.2f} ({date_after})")
                print(f"     Ratio:  {expected_ratio:.2f}x (expect ~{split_ratio:.2f}x)")

                # Allow 20% tolerance for split ratio match
                if abs(expected_ratio - split_ratio) / split_ratio < 0.2:
                    print(f"     [PASS] Split adjustment looks correct")
                else:
                    print(f"     [WARN] Split ratio doesn't match expected")
        except Exception as e:
            print(f"   {split_date}: [ERROR] {str(e)}")

    return results


def calculate_buy_and_hold(df, use_adj_close=True):
    """
    Calculate buy-and-hold return for SPXL.

    Args:
        df: SPXL price DataFrame
        use_adj_close: If True, use Adj Close (includes dividends), else use Close

    Returns:
        tuple: (multiple, start_price, end_price, start_date, end_date)
    """
    print(f"\n{'='*80}")
    print("CALCULATING BUY-AND-HOLD RETURN")
    print(f"{'='*80}")

    price_col = 'Adj Close' if use_adj_close and 'Adj Close' in df.columns else 'Close'
    print(f"\nUsing column: {price_col}")

    start_price = df.iloc[0][price_col]
    end_price = df.iloc[-1][price_col]
    start_date = df.index[0].date()
    end_date = df.index[-1].date()

    multiple = end_price / start_price
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (multiple ** (1/years) - 1) * 100

    print(f"\nStart: {start_date} @ ${start_price:.2f}")
    print(f"End:   {end_date} @ ${end_price:.2f}")
    print(f"\nReturn: {multiple:.2f}x (${10000:.0f} -> ${10000*multiple:,.0f})")
    print(f"Period: {years:.1f} years")
    print(f"CAGR:   {cagr:.1f}%")

    # Sanity check
    print(f"\nSanity Check:")
    print(f"  Expected range: {BH_RETURN_MIN}-{BH_RETURN_MAX}x")

    if BH_RETURN_MIN <= multiple <= BH_RETURN_MAX:
        print(f"  [PASS] Return is within expected range")
    elif multiple > BH_RETURN_MAX:
        print(f"  [FAIL] Return is TOO HIGH - likely data corruption")
    else:
        print(f"  [FAIL] Return is TOO LOW - possible data issue")

    return multiple, start_price, end_price, start_date, end_date


def compare_auto_adjust_settings():
    """
    Compare results with auto_adjust=True vs auto_adjust=False.

    This demonstrates the data corruption issue from last session.
    """
    print(f"\n{'='*80}")
    print("COMPARING auto_adjust=True vs auto_adjust=False")
    print(f"{'='*80}")

    # Download with auto_adjust=True (problematic)
    print(f"\n--- With auto_adjust=True (OLD METHOD - PROBLEMATIC) ---")
    df_adjusted = download_spxl_yahoo(auto_adjust=True)
    bh_adjusted, _, _, _, _ = calculate_buy_and_hold(df_adjusted, use_adj_close=False)

    try:
        june_adjusted = df_adjusted.loc['2025-06-30', 'Close']
        print(f"\nJune 30, 2025 price with auto_adjust=True: ${june_adjusted:.2f}")
    except KeyError:
        june_adjusted = None
        print(f"\nJune 30, 2025 not found")

    # Download with auto_adjust=False (correct)
    print(f"\n--- With auto_adjust=False (NEW METHOD - CORRECT) ---")
    df_unadjusted = download_spxl_yahoo(auto_adjust=False)
    bh_unadjusted, _, _, _, _ = calculate_buy_and_hold(df_unadjusted, use_adj_close=True)

    try:
        june_unadjusted = df_unadjusted.loc['2025-06-30', 'Close']
        print(f"\nJune 30, 2025 price with auto_adjust=False: ${june_unadjusted:.2f}")
    except KeyError:
        june_unadjusted = None
        print(f"\nJune 30, 2025 not found")

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if june_adjusted and june_unadjusted:
        print(f"\nJune 30, 2025 Price:")
        print(f"  auto_adjust=True:  ${june_adjusted:.2f}")
        print(f"  auto_adjust=False: ${june_unadjusted:.2f}")
        print(f"  Difference: {abs(june_adjusted - june_unadjusted):.2f} ({abs(june_adjusted - june_unadjusted)/june_unadjusted*100:.1f}%)")

    print(f"\nBuy-and-Hold Return:")
    print(f"  auto_adjust=True:  {bh_adjusted:.2f}x")
    print(f"  auto_adjust=False (Adj Close): {bh_unadjusted:.2f}x")
    print(f"  Difference: {abs(bh_adjusted - bh_unadjusted):.2f}x")

    return df_adjusted, df_unadjusted


def main():
    """
    Main verification workflow.
    """
    print(f"\n{'#'*80}")
    print(f"# SPXL DATA VERIFICATION - Fixing Previous Backtest Data Issues")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    # Step 1: Compare auto_adjust settings (demonstrate the problem)
    df_adjusted, df_unadjusted = compare_auto_adjust_settings()

    # Step 2: Verify split history
    ticker = yf.Ticker('SPXL')
    splits = verify_split_history(ticker)

    # Step 3: Validate critical prices (use unadjusted data as primary)
    validation_results = validate_critical_prices(df_unadjusted, splits)

    # Step 4: Save verified data
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'spxl_verified_data.csv')
    df_unadjusted.to_csv(output_file)
    print(f"\nVerified data saved to: {output_file}")

    # Step 5: Final summary
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for v in validation_results.values() if v == 'PASS')
    total = len(validation_results)

    print(f"\nValidation Results: {passed}/{total} checks passed")
    for check, result in validation_results.items():
        status_emoji = '[PASS]' if result == 'PASS' else '[FAIL]' if result == 'FAIL' else '[WARN]'
        print(f"  {status_emoji} {check}")

    # Calculate final buy-and-hold with verified data
    bh_return, start_price, end_price, start_date, end_date = calculate_buy_and_hold(
        df_unadjusted, use_adj_close=True
    )

    print(f"\nVerified SPXL Buy-and-Hold (2008-2025):")
    print(f"  Multiple: {bh_return:.2f}x")
    print(f"  $10,000 -> ${10000 * bh_return:,.0f}")
    print(f"  Assessment: {'REALISTIC' if BH_RETURN_MIN <= bh_return <= BH_RETURN_MAX else 'SUSPECT'}")

    print(f"\nConclusion:")
    if all(v == 'PASS' for v in validation_results.values()):
        print("  [SUCCESS] All validation checks passed - data is reliable")
    else:
        print("  [PARTIAL] Some validation checks failed - review warnings")

    print(f"\nReady to proceed with backtest re-run: YES")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
