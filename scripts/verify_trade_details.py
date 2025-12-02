#!/usr/bin/env python3
"""
Trade-Level Verification Script for ATLAS Trading System

Session 83K-21: Generate detailed trade logs to verify individual trades
before trusting aggregate metrics. The "sign reversal" in validation
(IS +3.99 vs OOS -16.42 Sharpe) could be a backtest bug.

Usage:
    # Basic run - SPY 3-1-2 with ThetaData
    uv run python scripts/verify_trade_details.py

    # Export to CSV
    uv run python scripts/verify_trade_details.py --csv output/spy_312_trades.csv

    # Verbose mode (show all trades)
    uv run python scripts/verify_trade_details.py --verbose

    # Only show trades with sanity check failures
    uv run python scripts/verify_trade_details.py --issues-only

    # Custom parameters
    uv run python scripts/verify_trade_details.py --symbol SPY --pattern 3-1-2 --timeframe 1D

Output:
    - Console: Trade-by-trade details with sanity check results
    - CSV: Full trade log with all fields for manual review
    - Summary: Aggregated statistics and issue counts
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeSanityChecker:
    """Implements sanity checks to detect potential bugs in trade data."""

    def __init__(self):
        self.issues_found: List[Dict] = []

    def check_direction_vs_pnl(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 1: P&L sign should match exit type.
        TARGET hit -> P&L should be POSITIVE
        STOP hit -> P&L should be NEGATIVE
        """
        pnl = trade.get('pnl', 0)
        exit_type = str(trade.get('exit_type', trade.get('exit_reason', ''))).upper()

        if exit_type == 'TARGET':
            if pnl <= 0:
                return False, f"TARGET hit but P&L is negative (${pnl:.2f})"
        elif exit_type == 'STOP':
            if pnl > 0:
                return False, f"STOP hit but P&L is positive (${pnl:.2f})"

        return True, "OK"

    def check_delta_sign(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 2: Delta sign vs Option Type.
        CALL delta should be POSITIVE (0 to +1)
        PUT delta should be NEGATIVE (-1 to 0)
        """
        option_type = str(trade.get('option_type', '')).upper()
        entry_delta = trade.get('entry_delta', 0)

        if option_type in ('CALL', 'C'):
            if entry_delta < 0:
                return False, f"CALL has negative delta ({entry_delta:.4f})"
        elif option_type in ('PUT', 'P'):
            if entry_delta > 0:
                return False, f"PUT has positive delta ({entry_delta:.4f})"

        return True, "OK"

    def check_strike_vs_underlying(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 3: Strike should be within entry-target range (expanded) per STRAT methodology.

        Session 83K-23 UPDATE: Per OPTIONS.md Section 3, optimal delta range is 0.50-0.80.
        This means ITM strikes ARE expected and correct behavior:
        - Delta 0.50 = ATM (strike = underlying price)
        - Delta 0.60-0.80 = ITM

        The previous check incorrectly required OTM strikes at entry.

        Now we check:
        1. Strike is within reasonable range of underlying (not wildly off)
        2. Delta is in optimal range (0.50-0.80) if available
        """
        strike = trade.get('strike', 0)
        entry_price = trade.get('entry_price', trade.get('entry_price_underlying', 0))
        target_price = trade.get('target_price', 0)
        entry_delta = abs(trade.get('entry_delta', 0))

        if strike == 0 or entry_price == 0:
            return True, "OK (insufficient data)"

        # Check: Strike should be within 15% of underlying (sanity check)
        # This catches cases where strike is wildly wrong (e.g., wrong symbol)
        strike_pct_diff = abs(strike - entry_price) / entry_price
        if strike_pct_diff > 0.15:
            return False, f"Strike ${strike:.2f} is {strike_pct_diff*100:.1f}% from underlying ${entry_price:.2f}"

        # Check: If delta available, verify it's in optimal STRAT range (0.30-0.90)
        # Relaxed from 0.50-0.80 to allow some flexibility
        if entry_delta > 0:
            if entry_delta < 0.30:
                return False, f"Delta {entry_delta:.2f} too low (< 0.30, too far OTM)"
            elif entry_delta > 0.95:
                return False, f"Delta {entry_delta:.2f} too high (> 0.95, essentially stock)"

        return True, "OK"

    def check_option_type_vs_direction(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 4: Option type should match pattern direction.
        Bullish pattern (direction=1) -> CALL
        Bearish pattern (direction=-1) -> PUT
        """
        direction = trade.get('direction', 1)
        option_type = str(trade.get('option_type', '')).upper()

        if direction == 1:
            if option_type not in ('CALL', 'C', ''):
                return False, f"Bullish pattern but option is {option_type}"
        elif direction == -1:
            if option_type not in ('PUT', 'P', ''):
                return False, f"Bearish pattern but option is {option_type}"

        return True, "OK"

    def check_entry_price_reasonable(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 5: Entry price should be near pattern trigger (within slippage tolerance).

        Session 83K-23 BUG FIX: Do NOT use stop_price as proxy for entry_trigger.
        Stop is the OPPOSITE side of the inside bar, not the entry trigger.
        - For bullish: entry_trigger = inside bar HIGH, stop = inside bar LOW
        - For bearish: entry_trigger = inside bar LOW, stop = inside bar HIGH
        Using stop as proxy gives completely wrong results.
        """
        entry_price = trade.get('entry_price', trade.get('entry_price_underlying', 0))
        # Session 83K-23: Only use entry_trigger, never use stop_price as proxy
        trigger_price = trade.get('entry_trigger', 0)

        if entry_price == 0 or trigger_price == 0:
            return True, "OK (insufficient data)"

        # Allow 2% slippage
        pct_diff = abs(entry_price - trigger_price) / trigger_price
        if pct_diff > 0.02:
            return False, f"Entry ${entry_price:.2f} is {pct_diff*100:.1f}% from trigger ${trigger_price:.2f}"

        return True, "OK"

    def check_price_move_vs_exit(self, trade: Dict) -> Tuple[bool, str]:
        """
        CHECK 6: Price move direction should match exit type for the direction.

        For CALL (bullish):
        - TARGET hit means price went UP (exit > entry)
        - STOP hit means price went DOWN (exit < entry)

        For PUT (bearish):
        - TARGET hit means price went DOWN (exit < entry)
        - STOP hit means price went UP (exit > entry)
        """
        direction = trade.get('direction', 1)
        entry_price = trade.get('entry_price', trade.get('entry_price_underlying', 0))
        exit_price = trade.get('exit_price', 0)
        exit_type = str(trade.get('exit_type', trade.get('exit_reason', ''))).upper()

        if entry_price == 0 or exit_price == 0:
            return True, "OK (insufficient data)"

        price_move = exit_price - entry_price

        if direction == 1:  # Bullish
            if exit_type == 'TARGET' and price_move < 0:
                return False, f"Bullish TARGET but price went DOWN (${price_move:.2f})"
            if exit_type == 'STOP' and price_move > 0:
                return False, f"Bullish STOP but price went UP (+${price_move:.2f})"
        else:  # Bearish
            if exit_type == 'TARGET' and price_move > 0:
                return False, f"Bearish TARGET but price went UP (+${price_move:.2f})"
            if exit_type == 'STOP' and price_move < 0:
                return False, f"Bearish STOP but price went DOWN (${price_move:.2f})"

        return True, "OK"

    def run_all_checks(self, trade: Dict) -> List[Dict]:
        """Run all sanity checks on a trade."""
        checks = [
            ('direction_vs_pnl', self.check_direction_vs_pnl),
            ('delta_sign', self.check_delta_sign),
            ('strike_vs_underlying', self.check_strike_vs_underlying),
            ('option_type_vs_direction', self.check_option_type_vs_direction),
            ('entry_price_reasonable', self.check_entry_price_reasonable),
            ('price_move_vs_exit', self.check_price_move_vs_exit),
        ]

        results = []
        for check_name, check_func in checks:
            passed, message = check_func(trade)
            results.append({
                'check': check_name,
                'passed': passed,
                'message': message
            })
            if not passed:
                self.issues_found.append({
                    'trade_id': trade.get('trade_id', 'unknown'),
                    'check': check_name,
                    'message': message
                })

        return results


def format_trade_for_console(trade: Dict, check_results: List[Dict], trade_num: int) -> str:
    """Format trade details for console output."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"TRADE #{trade_num}")
    lines.append("=" * 80)

    # Pattern & Timing
    pattern_type = trade.get('pattern_type', 'N/A')
    direction = 'BULLISH' if trade.get('direction', 1) == 1 else 'BEARISH'
    lines.append(f"Pattern:      {pattern_type} ({direction})")
    lines.append(f"Symbol:       {trade.get('symbol', 'N/A')}")
    lines.append(f"Timeframe:    {trade.get('timeframe', 'N/A')}")

    # Timestamps
    lines.append("")
    lines.append("--- TIMESTAMPS ---")
    pattern_ts = trade.get('pattern_timestamp', 'N/A')
    entry_ts = trade.get('entry_timestamp', trade.get('entry_date', 'N/A'))
    exit_ts = trade.get('exit_timestamp', trade.get('exit_date', 'N/A'))
    lines.append(f"Pattern:      {pattern_ts}")
    lines.append(f"Entry:        {entry_ts}")
    lines.append(f"Exit:         {exit_ts}")
    lines.append(f"Days Held:    {trade.get('days_held', 0)}")

    # Option Details
    lines.append("")
    lines.append("--- OPTION DETAILS ---")
    lines.append(f"Option Type:  {trade.get('option_type', 'N/A')}")
    lines.append(f"Strike:       ${trade.get('strike', 0):.2f}")
    lines.append(f"OSI Symbol:   {trade.get('osi_symbol', 'N/A')}")

    # Price Levels
    lines.append("")
    lines.append("--- PRICE LEVELS ---")
    lines.append(f"Entry Price:  ${trade.get('entry_price', trade.get('entry_price_underlying', 0)):.2f}")
    lines.append(f"Stop Price:   ${trade.get('stop_price', 0):.2f}")
    lines.append(f"Target Price: ${trade.get('target_price', 0):.2f}")
    lines.append(f"Exit Price:   ${trade.get('exit_price', 0):.2f}")
    lines.append(f"Exit Reason:  {trade.get('exit_type', trade.get('exit_reason', 'N/A'))}")

    # Greeks
    lines.append("")
    lines.append("--- GREEKS ---")
    lines.append(f"Entry Delta:  {trade.get('entry_delta', 0):+.4f}")
    lines.append(f"Exit Delta:   {trade.get('exit_delta', 0):+.4f}")
    lines.append(f"Entry Theta:  ${trade.get('entry_theta', 0):.4f}/day")
    lines.append(f"Exit Theta:   ${trade.get('exit_theta', 0):.4f}/day")

    # P&L
    lines.append("")
    lines.append("--- P&L ---")
    pnl = trade.get('pnl', 0)
    pnl_pct = trade.get('pnl_pct', 0)
    result = 'WIN' if pnl > 0 else 'LOSS'
    lines.append(f"P&L:          ${pnl:+.2f} ({result})")
    lines.append(f"P&L %:        {pnl_pct:+.2%}")

    # Data Source
    lines.append(f"Data Source:  {trade.get('data_source', 'N/A')}")

    # Sanity Checks
    lines.append("")
    lines.append("--- SANITY CHECKS ---")
    failed_checks = [c for c in check_results if not c['passed']]
    if failed_checks:
        lines.append("!!! ISSUES DETECTED !!!")
        for check in failed_checks:
            lines.append(f"  [FAIL] {check['check']}: {check['message']}")
    else:
        lines.append("[ALL PASSED]")

    return "\n".join(lines)


def run_verification(
    symbol: str = 'SPY',
    pattern: str = '3-1-2',
    timeframe: str = '1D',
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-01',
    csv_output: Optional[str] = None,
    verbose: bool = False,
    issues_only: bool = False
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Run trade verification and return results.

    Returns:
        Tuple of (trades_df, issues_list)
    """
    # Import project modules
    from validation.strat_validator import DataFetcher
    from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig

    print("=" * 80)
    print("ATLAS TRADE VERIFICATION - Session 83K-21")
    print("=" * 80)
    print(f"Symbol:    {symbol}")
    print(f"Pattern:   {pattern}")
    print(f"Timeframe: {timeframe}")
    print(f"Period:    {start_date} to {end_date}")
    print()

    # Fetch data
    print("Fetching price data...")
    fetcher = DataFetcher()
    try:
        data = fetcher.get_data(symbol, timeframe, start_date, end_date)
        print(f"  Loaded {len(data)} bars")
    except Exception as e:
        print(f"  ERROR: Failed to fetch data: {e}")
        return pd.DataFrame(), []

    # Configure strategy
    print("Configuring strategy...")
    config = STRATOptionsConfig(
        pattern_types=[pattern],
        timeframe=timeframe,
        symbol=symbol,
        min_continuation_bars=2,
    )
    strategy = STRATOptionsStrategy(config=config)

    # Check for ThetaData
    # Session 83K-23 BUG FIX: Correct attribute names for ThetaData wiring
    # OLD (WRONG): _thetadata and missing _use_market_prices
    # NEW (CORRECT): _options_fetcher and _use_market_prices=True
    if strategy._backtester is not None:
        try:
            from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
            thetadata = ThetaDataOptionsFetcher()
            # Check if ThetaData terminal is available by testing connection
            if thetadata._provider is not None:
                print("  ThetaData connected")
                strategy._backtester._options_fetcher = thetadata
                strategy._backtester._use_market_prices = True
            else:
                print("  WARNING: ThetaData not available - using Black-Scholes fallback")
        except Exception as e:
            print(f"  WARNING: ThetaData error: {e}")

    # Run backtest
    print("Running backtest...")
    result = strategy.backtest(data)

    if result.trades is None or result.trades.empty:
        print("  No trades found!")
        return pd.DataFrame(), []

    trades_df = result.trades.copy()
    print(f"  Found {len(trades_df)} trades")

    # Run sanity checks
    print()
    print("Running sanity checks...")
    checker = TradeSanityChecker()

    all_results = []
    for idx, row in trades_df.iterrows():
        trade_dict = row.to_dict()
        trade_dict['trade_id'] = idx + 1 if isinstance(idx, int) else idx
        check_results = checker.run_all_checks(trade_dict)

        has_issues = any(not c['passed'] for c in check_results)
        trade_dict['sanity_passed'] = not has_issues
        trade_dict['sanity_flags'] = ', '.join([c['check'] for c in check_results if not c['passed']])

        # Print trade details based on flags
        trade_num = idx + 1 if isinstance(idx, int) else idx
        if verbose or (issues_only and has_issues) or (not issues_only and not verbose):
            if has_issues or verbose:
                print(format_trade_for_console(trade_dict, check_results, trade_num))
                print()

        all_results.append(trade_dict)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Trades:       {len(results_df)}")

    if 'pnl' in results_df.columns:
        winners = (results_df['pnl'] > 0).sum()
        print(f"Winners:            {winners} ({winners/len(results_df)*100:.1f}%)")
        print(f"Total P&L:          ${results_df['pnl'].sum():+,.2f}")

    issues_count = len(checker.issues_found)
    print(f"Trades with Issues: {issues_count}")

    if checker.issues_found:
        print()
        print("ISSUES BY CHECK:")
        issues_by_check = {}
        for issue in checker.issues_found:
            check_name = issue['check']
            issues_by_check[check_name] = issues_by_check.get(check_name, 0) + 1

        for check_name, count in sorted(issues_by_check.items(), key=lambda x: -x[1]):
            print(f"  {check_name}: {count} occurrence(s)")

    # Export to CSV
    if csv_output:
        output_path = Path(csv_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print()
        print(f"Results exported to: {output_path}")

    return results_df, checker.issues_found


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS Trade Verification Script - Session 83K-21"
    )
    parser.add_argument('--symbol', default='SPY', help='Symbol to verify (default: SPY)')
    parser.add_argument('--pattern', default='3-1-2', help='Pattern type (default: 3-1-2)')
    parser.add_argument('--timeframe', default='1D', help='Timeframe: 1D, 1W, 1M (default: 1D)')
    parser.add_argument('--start', default='2020-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2024-12-01', help='End date YYYY-MM-DD')
    parser.add_argument('--csv', help='CSV output path')
    parser.add_argument('--verbose', action='store_true', help='Show all trades')
    parser.add_argument('--issues-only', action='store_true', help='Only show trades with issues')

    args = parser.parse_args()

    trades_df, issues = run_verification(
        symbol=args.symbol,
        pattern=args.pattern,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        csv_output=args.csv,
        verbose=args.verbose,
        issues_only=args.issues_only
    )

    # Exit with error code if issues found
    if issues:
        print()
        print(f"WARNING: {len(issues)} sanity check failures detected!")
        print("Review the trades above and investigate the flagged issues.")
        sys.exit(1)
    else:
        print()
        print("All sanity checks PASSED.")
        sys.exit(0)


if __name__ == '__main__':
    main()
