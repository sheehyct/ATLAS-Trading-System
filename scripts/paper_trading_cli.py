#!/usr/bin/env python
"""
STRAT Paper Trading CLI - Session 83K-41

Interactive command-line interface for paper trading STRAT patterns.

Usage:
    # Scan for signals
    uv run python scripts/paper_trading_cli.py scan

    # Scan specific symbol
    uv run python scripts/paper_trading_cli.py scan --symbol SPY

    # Add a new trade
    uv run python scripts/paper_trading_cli.py add

    # List open trades
    uv run python scripts/paper_trading_cli.py list --open

    # Close a trade
    uv run python scripts/paper_trading_cli.py close PT_20251204_001

    # Generate report
    uv run python scripts/paper_trading_cli.py report

    # Compare to backtest
    uv run python scripts/paper_trading_cli.py compare
"""

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strat.paper_trading import (
    PaperTrade,
    PaperTradeLog,
    create_paper_trade,
    BASELINE_BACKTEST_STATS,
    HOURLY_BACKTEST_STATS,
)
from strat.paper_signal_scanner import (
    PaperSignalScanner,
    scan_for_signals,
    get_actionable_signals,
)


def cmd_scan(args):
    """Scan for STRAT pattern signals."""
    print("\nScanning for STRAT patterns...")
    print("=" * 60)

    scanner = PaperSignalScanner()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = scanner.DEFAULT_SYMBOLS

    if args.timeframe:
        timeframes = [args.timeframe.upper()]
    else:
        timeframes = scanner.DEFAULT_TIMEFRAMES

    all_signals = []
    for symbol in symbols:
        print(f"\nScanning {symbol}...")
        for tf in timeframes:
            signals = scanner.scan_symbol_timeframe(symbol, tf, lookback_bars=20)
            all_signals.extend(signals)
            if signals:
                print(f"  {tf}: {len(signals)} signals")

    if all_signals:
        print("\n" + "=" * 60)
        print("DETECTED SIGNALS")
        print("=" * 60)
        scanner.print_signals(all_signals)

        # Option to add trade
        if args.interactive:
            add_trade_from_signal(all_signals)
    else:
        print("\nNo actionable signals detected.")

    return all_signals


def cmd_add(args):
    """Add a new paper trade manually."""
    print("\nAdd New Paper Trade")
    print("=" * 40)

    # Get trade details
    symbol = input("Symbol (e.g., SPY): ").upper()
    pattern_type = input("Pattern (e.g., 3-2U, 2-2D): ").upper()
    timeframe = input("Timeframe (1H/1D/1W/1M): ").upper()
    direction = input("Direction (CALL/PUT): ").upper()

    entry_trigger = float(input("Entry trigger price: $"))
    target_price = float(input("Target price: $"))
    stop_price = float(input("Stop price: $"))

    # Optional fields
    strike = input("Option strike (press Enter to skip): ")
    strike = float(strike) if strike else 0.0

    dte = input("DTE (press Enter to skip): ")
    dte = int(dte) if dte else 0

    notes = input("Notes (press Enter to skip): ")

    # Create trade
    trade = create_paper_trade(
        pattern_type=pattern_type,
        timeframe=timeframe,
        symbol=symbol,
        direction=direction,
        pattern_detected_time=datetime.now(),
        entry_trigger=entry_trigger,
        target_price=target_price,
        stop_price=stop_price,
        strike=strike,
        dte=dte,
        notes=notes,
    )

    # Save to log
    log = PaperTradeLog()
    log.add_trade(trade)

    print(f"\nTrade created: {trade.trade_id}")
    print(f"Status: {trade.status}")
    print(f"Magnitude: {trade.magnitude_pct:.2f}%")
    print(f"R:R: {trade.risk_reward:.2f}:1")


def cmd_open(args):
    """Open a pending trade (mark as entered)."""
    log = PaperTradeLog()
    trade = log.get_trade(args.trade_id)

    if not trade:
        print(f"Trade not found: {args.trade_id}")
        return

    entry_price = float(input("Entry price: $"))
    option_price = float(input("Option price paid: $"))
    delta = input("Delta at entry (press Enter to skip): ")
    delta = float(delta) if delta else 0.0

    trade.open_trade(
        entry_time=datetime.now(),
        entry_price=entry_price,
        option_price=option_price,
        delta=delta,
    )

    log._save_trades()
    print(f"\nTrade opened: {args.trade_id}")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Option premium: ${option_price:.2f}")


def cmd_close(args):
    """Close an open trade."""
    log = PaperTradeLog()
    trade = log.get_trade(args.trade_id)

    if not trade:
        print(f"Trade not found: {args.trade_id}")
        return

    if not trade.is_open:
        print(f"Trade is not open. Status: {trade.status}")
        return

    exit_price = float(input("Exit underlying price: $"))
    option_price = float(input("Option price received: $"))
    exit_reason = input("Exit reason (TARGET_HIT/STOP_HIT/TIME_EXIT/MANUAL): ").upper()
    delta = input("Delta at exit (press Enter to skip): ")
    delta = float(delta) if delta else 0.0

    trade.close_trade(
        exit_time=datetime.now(),
        exit_price=exit_price,
        option_price=option_price,
        exit_reason=exit_reason,
        delta=delta,
    )

    log._save_trades()

    print(f"\nTrade closed: {args.trade_id}")
    print(f"P&L: ${trade.pnl_dollars:.2f} ({trade.pnl_percent:.1f}%)")
    print(f"Days held: {trade.days_held}")
    print(f"Exit reason: {trade.exit_reason}")


def cmd_list(args):
    """List paper trades."""
    log = PaperTradeLog()

    if args.open_only:
        trades = log.get_open_trades()
        title = "OPEN TRADES"
    elif args.closed_only:
        trades = log.get_closed_trades()
        title = "CLOSED TRADES"
    else:
        trades = list(log.trades)
        title = "ALL TRADES"

    print(f"\n{title}")
    print("=" * 80)

    if not trades:
        print("No trades found.")
        return

    for t in trades:
        status_icon = "[OPEN]" if t.is_open else "[CLOSED]"
        pnl_str = f"${t.pnl_dollars:+.2f}" if t.is_closed else "pending"

        print(f"\n{status_icon} {t.trade_id}")
        print(f"  {t.pattern_type} {t.direction} on {t.symbol} ({t.timeframe})")
        print(f"  Detected: {t.pattern_detected_time}")
        print(f"  Entry trigger: ${t.entry_trigger:.2f}")
        print(f"  Target: ${t.target_price:.2f}, Stop: ${t.stop_price:.2f}")
        print(f"  P&L: {pnl_str}")


def cmd_report(args):
    """Generate summary report."""
    log = PaperTradeLog()

    print(log.summary_report())

    # Show pattern breakdown
    pattern_breakdown = log.get_pattern_breakdown()
    if pattern_breakdown:
        print("\nPATTERN PERFORMANCE:")
        print("-" * 40)
        for pattern, stats in sorted(pattern_breakdown.items()):
            print(f"{pattern:10} | {stats['trades']:3} trades | "
                  f"{stats['win_rate']:5.1f}% WR | "
                  f"${stats['avg_pnl']:+8.2f} avg")


def cmd_compare(args):
    """Compare paper trading to backtest expectations."""
    log = PaperTradeLog()

    print("\nPAPER TRADING vs BACKTEST COMPARISON")
    print("=" * 60)

    # Overall comparison
    comparison = log.compare_to_backtest(BASELINE_BACKTEST_STATS)

    if 'message' in comparison:
        print(comparison['message'])
        return

    print("\nOVERALL METRICS:")
    print("-" * 40)

    print(f"{'Metric':<15} {'Backtest':>12} {'Paper':>12} {'Diff':>12}")
    print("-" * 51)

    for metric, values in comparison.items():
        if metric == 'trade_count':
            print(f"{'Trades':<15} {'-':>12} {values['paper']:>12}")
        else:
            bt = values['backtest']
            pp = values['paper']
            diff = values['difference']
            suffix = '%' if 'rate' in metric else ''

            if 'pnl' in metric:
                print(f"{metric:<15} ${bt:>11.2f} ${pp:>11.2f} ${diff:>+11.2f}")
            else:
                print(f"{metric:<15} {bt:>11.1f}{suffix} {pp:>11.1f}{suffix} {diff:>+11.1f}")

    # Hourly comparison if applicable
    hourly_trades = log.get_trades_by_timeframe('1H')
    if hourly_trades:
        print("\nHOURLY TIMEFRAME (VALIDATION):")
        print("-" * 40)
        print("Backtest shows hourly is UNPROFITABLE (-$240 avg)")

        closed_hourly = [t for t in hourly_trades if t.is_closed]
        if closed_hourly:
            avg_pnl = sum(t.pnl_dollars for t in closed_hourly) / len(closed_hourly)
            print(f"Paper trading hourly: ${avg_pnl:.2f} avg")

            if avg_pnl < 0:
                print("CONFIRMED: Hourly unprofitable (matches backtest)")
            else:
                print("SURPRISING: Hourly profitable (contradicts backtest)")


def cmd_export(args):
    """Export trades to CSV."""
    log = PaperTradeLog()
    df = log.to_dataframe()

    if df.empty:
        print("No trades to export.")
        return

    output_path = args.output or "paper_trades_export.csv"
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} trades to {output_path}")


def add_trade_from_signal(signals):
    """Helper to add a trade from detected signals."""
    print("\nWould you like to add a trade from these signals? (y/n)")
    choice = input().lower()

    if choice != 'y':
        return

    print("Enter signal number to add as trade:")
    try:
        num = int(input())
        if 1 <= num <= len(signals):
            signal = signals[num - 1]
            trade = signal.to_paper_trade()

            log = PaperTradeLog()
            log.add_trade(trade)

            print(f"\nTrade created: {trade.trade_id}")
            print(f"Pattern: {trade.pattern_type}")
            print(f"Symbol: {trade.symbol} ({trade.timeframe})")
            print(f"Direction: {trade.direction}")
        else:
            print("Invalid signal number.")
    except ValueError:
        print("Invalid input.")


def main():
    parser = argparse.ArgumentParser(
        description="STRAT Paper Trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for signals
  python paper_trading_cli.py scan

  # Scan specific symbol
  python paper_trading_cli.py scan --symbol SPY

  # Add a trade interactively
  python paper_trading_cli.py add

  # List open trades
  python paper_trading_cli.py list --open

  # Close a trade
  python paper_trading_cli.py close PT_20251204_001

  # Generate report
  python paper_trading_cli.py report

  # Compare paper to backtest
  python paper_trading_cli.py compare
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for pattern signals')
    scan_parser.add_argument('--symbol', '-s', help='Symbol to scan')
    scan_parser.add_argument('--timeframe', '-t', help='Timeframe (1H/1D/1W/1M)')
    scan_parser.add_argument('--interactive', '-i', action='store_true',
                            help='Interactive mode to add trades')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new paper trade')

    # Open command
    open_parser = subparsers.add_parser('open', help='Open a pending trade')
    open_parser.add_argument('trade_id', help='Trade ID to open')

    # Close command
    close_parser = subparsers.add_parser('close', help='Close an open trade')
    close_parser.add_argument('trade_id', help='Trade ID to close')

    # List command
    list_parser = subparsers.add_parser('list', help='List paper trades')
    list_parser.add_argument('--open', dest='open_only', action='store_true',
                            help='Show only open trades')
    list_parser.add_argument('--closed', dest='closed_only', action='store_true',
                            help='Show only closed trades')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate summary report')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare to backtest')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export trades to CSV')
    export_parser.add_argument('--output', '-o', help='Output file path')

    args = parser.parse_args()

    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'add':
        cmd_add(args)
    elif args.command == 'open':
        cmd_open(args)
    elif args.command == 'close':
        cmd_close(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'export':
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
