#!/usr/bin/env python
"""
STRAT Signal Daemon - Command Line Entry Point - Session 83K-46/49

Start the autonomous signal detection daemon for paper trading.

Usage:
    # Start daemon (runs continuously)
    uv run python scripts/signal_daemon.py start

    # Start with execution enabled
    uv run python scripts/signal_daemon.py start --execute

    # Run a single scan
    uv run python scripts/signal_daemon.py scan --timeframe 1D

    # Run all scans once
    uv run python scripts/signal_daemon.py scan-all

    # Execute pending signals manually
    uv run python scripts/signal_daemon.py execute

    # Show current option positions
    uv run python scripts/signal_daemon.py positions

    # Close a position
    uv run python scripts/signal_daemon.py close AAPL250117C00200000

    # Check positions for exit conditions (Session 83K-49)
    uv run python scripts/signal_daemon.py monitor

    # Execute exits automatically
    uv run python scripts/signal_daemon.py monitor --execute

    # Show monitoring statistics
    uv run python scripts/signal_daemon.py monitor-stats

    # Test alerter connections
    uv run python scripts/signal_daemon.py test

    # Show status
    uv run python scripts/signal_daemon.py status

Environment Variables:
    DISCORD_WEBHOOK_URL: Discord webhook for alerts
    SIGNAL_SYMBOLS: Comma-separated symbols (default: SPY,QQQ,IWM,DIA,AAPL)
    SIGNAL_TIMEFRAMES: Comma-separated timeframes (default: 1H,1D,1W,1M)
    SIGNAL_LOG_LEVEL: Log level (default: INFO)
    SIGNAL_STORE_PATH: Signal store directory (default: data/signals)
    SIGNAL_EXECUTION_ENABLED: Enable order execution (true/false)
    SIGNAL_EXECUTION_ACCOUNT: Alpaca account (SMALL, MEDIUM, LARGE)
    SIGNAL_MONITORING_ENABLED: Enable position monitoring (true/false)
    SIGNAL_EXIT_DTE: Exit when DTE reaches this value (default: 3)
    SIGNAL_MAX_LOSS_PCT: Max loss as % of premium (default: 0.50)
    SIGNAL_MAX_PROFIT_PCT: Take profit at this % gain (default: 1.00)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strat.signal_automation import (
    SignalAutomationConfig,
    SignalDaemon,
    SignalStore,
)


def setup_logging(level: str = 'INFO') -> None:
    """Configure logging for the daemon."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def cmd_start(args: argparse.Namespace) -> int:
    """Start the daemon."""
    print("=" * 60)
    print("STRAT Signal Daemon")
    print("=" * 60)

    # Load configuration
    config = SignalAutomationConfig.from_env()

    # Override from args
    if args.symbols:
        config.scan.symbols = args.symbols.split(',')
    if args.timeframes:
        config.scan.timeframes = args.timeframes.split(',')
    if args.execute:
        config.execution.enabled = True

    # Validate
    issues = config.validate()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  [!] {issue}")

    print(f"\nConfiguration:")
    print(f"  Symbols: {', '.join(config.scan.symbols)}")
    print(f"  Timeframes: {', '.join(config.scan.timeframes)}")
    print(f"  Discord: {'Enabled' if config.alerts.discord_enabled else 'Disabled'}")
    print(f"  Execution: {'Enabled' if config.execution.enabled else 'Disabled'}")
    print(f"  Store: {config.store_path}")
    print()

    # Create and start daemon
    try:
        daemon = SignalDaemon.from_config(config)

        # Test alerters first
        print("Testing alerters...")
        results = daemon.test_alerters()
        for name, passed in results.items():
            status = "OK" if passed else "FAIL"
            print(f"  {name}: {status}")

        if config.execution.enabled:
            if daemon.executor is not None:
                print("\nExecution: Connected to Alpaca")
            else:
                print("\nExecution: FAILED to connect (alerts only)")

        print("\nStarting daemon (Ctrl+C to stop)...")
        daemon.start(block=True)

    except KeyboardInterrupt:
        print("\nShutdown requested")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Run a single scan."""
    config = SignalAutomationConfig.from_env()

    if args.symbols:
        config.scan.symbols = args.symbols.split(',')

    daemon = SignalDaemon.from_config(config)

    print(f"Running {args.timeframe} scan...")
    print(f"Symbols: {', '.join(config.scan.symbols)}")
    print()

    signals = daemon.run_scan(args.timeframe)

    if signals:
        print(f"\nFound {len(signals)} new signal(s):")
        for s in signals:
            direction = "CALL" if s.direction == "CALL" else "PUT"
            print(
                f"  [{direction}] {s.symbol} {s.pattern_type} "
                f"@ ${s.entry_trigger:.2f} -> ${s.target_price:.2f} "
                f"(R:R {s.risk_reward:.2f})"
            )
    else:
        print("\nNo new signals found")

    return 0


def cmd_scan_all(args: argparse.Namespace) -> int:
    """Run all scans once."""
    config = SignalAutomationConfig.from_env()

    if args.symbols:
        config.scan.symbols = args.symbols.split(',')

    daemon = SignalDaemon.from_config(config)

    print("Running all scans...")
    print(f"Symbols: {', '.join(config.scan.symbols)}")
    print(f"Timeframes: {', '.join(config.scan.timeframes)}")
    print()

    results = daemon.run_all_scans()

    total_signals = 0
    for tf, signals in results.items():
        print(f"\n{tf}: {len(signals)} signal(s)")
        for s in signals:
            direction = "CALL" if s.direction == "CALL" else "PUT"
            print(
                f"  [{direction}] {s.symbol} {s.pattern_type} "
                f"@ ${s.entry_trigger:.2f} -> ${s.target_price:.2f}"
            )
        total_signals += len(signals)

    print(f"\nTotal: {total_signals} signal(s)")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Test alerter connections."""
    config = SignalAutomationConfig.from_env()
    daemon = SignalDaemon.from_config(config)

    print("Testing alerter connections...")
    print()

    results = daemon.test_alerters()

    all_passed = True
    for name, passed in results.items():
        status = "OK" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed")
        return 0
    else:
        print("\nSome tests failed")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show signal store status."""
    config = SignalAutomationConfig.from_env()
    store = SignalStore(config.store_path)

    stats = store.get_stats()

    print("Signal Store Status")
    print("=" * 40)
    print(f"Total signals: {stats.get('total', 0)}")

    if stats.get('by_status'):
        print("\nBy Status:")
        for status, count in stats['by_status'].items():
            print(f"  {status}: {count}")

    if stats.get('by_symbol'):
        print("\nBy Symbol:")
        for symbol, count in stats['by_symbol'].items():
            print(f"  {symbol}: {count}")

    if stats.get('by_timeframe'):
        print("\nBy Timeframe:")
        for tf, count in stats['by_timeframe'].items():
            print(f"  {tf}: {count}")

    if stats.get('by_pattern'):
        print("\nBy Pattern:")
        for pattern, count in stats['by_pattern'].items():
            print(f"  {pattern}: {count}")

    if stats.get('oldest'):
        print(f"\nOldest: {stats['oldest']}")
    if stats.get('newest'):
        print(f"Newest: {stats['newest']}")

    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Clean up old signals."""
    config = SignalAutomationConfig.from_env()
    store = SignalStore(config.store_path)

    print(f"Cleaning up signals older than {args.days} days...")

    removed = store.cleanup_old_signals(days=args.days)

    print(f"Removed {removed} old signal(s)")
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    """Execute pending signals (Session 83K-48)."""
    config = SignalAutomationConfig.from_env()
    config.execution.enabled = True  # Force enable for manual execution

    daemon = SignalDaemon.from_config(config)

    if daemon.executor is None:
        print("ERROR: Failed to connect to Alpaca")
        print("Check your API credentials in .env")
        return 1

    # Get pending signals
    pending = daemon.signal_store.get_pending_signals()

    if not pending:
        print("No pending signals to execute")
        return 0

    print(f"Found {len(pending)} pending signal(s)")
    print()

    for signal in pending:
        direction = "CALL" if signal.direction == "CALL" else "PUT"
        print(
            f"  [{direction}] {signal.symbol} {signal.pattern_type} "
            f"@ ${signal.entry_trigger:.2f} (R:R {signal.risk_reward:.2f})"
        )

    print()

    # Confirm execution
    if not args.yes:
        confirm = input("Execute these signals? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return 0

    # Execute signals
    results = daemon.execute_signals(pending)

    print("\nExecution Results:")
    for result in results:
        if result.state.value == 'submitted':
            print(f"  OK: {result.osi_symbol} - Order {result.order_id}")
        elif result.state.value == 'skipped':
            print(f"  SKIP: {result.signal_key} - {result.error}")
        else:
            print(f"  FAIL: {result.signal_key} - {result.error}")

    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    """Show current option positions (Session 83K-48)."""
    config = SignalAutomationConfig.from_env()
    config.execution.enabled = True

    daemon = SignalDaemon.from_config(config)

    if daemon.executor is None:
        print("ERROR: Failed to connect to Alpaca")
        return 1

    positions = daemon.get_positions()

    if not positions:
        print("No open option positions")
        return 0

    print("Open Option Positions")
    print("=" * 60)

    for pos in positions:
        symbol = pos.get('symbol', 'N/A')
        qty = pos.get('qty', 0)
        avg_price = pos.get('avg_entry_price', 0)
        current_price = pos.get('current_price', 0)
        pnl = pos.get('unrealized_pl', 0)
        pnl_pct = pos.get('unrealized_plpc', 0) * 100 if pos.get('unrealized_plpc') else 0

        print(f"  {symbol}")
        print(f"    Qty: {qty}")
        print(f"    Entry: ${avg_price:.2f}")
        print(f"    Current: ${current_price:.2f}")
        print(f"    P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        print()

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Check positions for exit conditions (Session 83K-49)."""
    config = SignalAutomationConfig.from_env()
    config.execution.enabled = True

    daemon = SignalDaemon.from_config(config)

    if daemon.position_monitor is None:
        print("ERROR: Position monitoring not available")
        print("Check executor connection and monitoring config")
        return 1

    print("Checking positions for exit conditions...")
    print()

    # Check positions
    exit_signals = daemon.check_positions_now()

    if not exit_signals:
        print("No exit conditions detected")
        # Show tracked positions
        tracked = daemon.get_tracked_positions()
        if tracked:
            print(f"\nTracking {len(tracked)} position(s):")
            for pos in tracked:
                if pos.is_active:
                    print(f"  {pos.osi_symbol}")
                    print(f"    Target: ${pos.target_price:.2f} | Stop: ${pos.stop_price:.2f}")
                    print(f"    DTE: {pos.dte} | P&L: ${pos.unrealized_pnl:+.2f}")
        return 0

    print(f"Found {len(exit_signals)} exit condition(s):")
    print()

    for signal in exit_signals:
        reason = signal.reason.value if hasattr(signal.reason, 'value') else str(signal.reason)
        print(f"  [{reason}] {signal.osi_symbol}")
        print(f"    Underlying: ${signal.underlying_price:.2f}")
        print(f"    Option: ${signal.current_option_price:.2f}")
        print(f"    P&L: ${signal.unrealized_pnl:+.2f}")
        print(f"    DTE: {signal.dte}")
        print(f"    Details: {signal.details}")
        print()

    # Execute if --yes flag
    if args.execute:
        if not args.yes:
            confirm = input("Execute these exits? [y/N] ")
            if confirm.lower() != 'y':
                print("Aborted")
                return 0

        print("Executing exits...")
        results = daemon.position_monitor.execute_all_exits(exit_signals)
        print(f"Closed {len(results)} position(s)")

    return 0


def cmd_monitor_stats(args: argparse.Namespace) -> int:
    """Show position monitoring statistics (Session 83K-49)."""
    config = SignalAutomationConfig.from_env()
    config.execution.enabled = True

    daemon = SignalDaemon.from_config(config)

    if daemon.position_monitor is None:
        print("Position monitoring not available")
        return 1

    stats = daemon.position_monitor.get_stats()

    print("Position Monitoring Statistics")
    print("=" * 40)
    print(f"Active positions: {stats.get('active_positions', 0)}")
    print(f"Closed positions: {stats.get('closed_positions', 0)}")
    print(f"Total unrealized P&L: ${stats.get('total_unrealized_pnl', 0):.2f}")
    print(f"Position checks: {stats.get('check_count', 0)}")
    print(f"Exits executed: {stats.get('exit_count', 0)}")
    print(f"Errors: {stats.get('error_count', 0)}")

    return 0


def cmd_close(args: argparse.Namespace) -> int:
    """Close an option position (Session 83K-48)."""
    config = SignalAutomationConfig.from_env()
    config.execution.enabled = True

    daemon = SignalDaemon.from_config(config)

    if daemon.executor is None:
        print("ERROR: Failed to connect to Alpaca")
        return 1

    print(f"Closing position: {args.symbol}")

    result = daemon.close_position(args.symbol)

    if result:
        print(f"Close order submitted: {result.get('id', 'N/A')}")
        return 0
    else:
        print("Failed to close position")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='STRAT Signal Daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the daemon')
    start_parser.add_argument(
        '--symbols',
        help='Comma-separated symbols to scan'
    )
    start_parser.add_argument(
        '--timeframes',
        help='Comma-separated timeframes to enable'
    )
    start_parser.add_argument(
        '--execute', '-x',
        action='store_true',
        help='Enable order execution (paper trading)'
    )

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Run single scan')
    scan_parser.add_argument(
        '--timeframe', '-t',
        default='1D',
        choices=['1H', '1D', '1W', '1M'],
        help='Timeframe to scan'
    )
    scan_parser.add_argument(
        '--symbols',
        help='Comma-separated symbols to scan'
    )

    # Scan-all command
    scan_all_parser = subparsers.add_parser('scan-all', help='Run all scans')
    scan_all_parser.add_argument(
        '--symbols',
        help='Comma-separated symbols to scan'
    )

    # Test command
    subparsers.add_parser('test', help='Test alerter connections')

    # Status command
    subparsers.add_parser('status', help='Show signal store status')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old signals')
    cleanup_parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Remove signals older than N days'
    )

    # Execute command (Session 83K-48)
    execute_parser = subparsers.add_parser(
        'execute',
        help='Execute pending signals'
    )
    execute_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    # Positions command (Session 83K-48)
    subparsers.add_parser('positions', help='Show option positions')

    # Close command (Session 83K-48)
    close_parser = subparsers.add_parser('close', help='Close option position')
    close_parser.add_argument(
        'symbol',
        help='OCC symbol to close (e.g., AAPL250117C00200000)'
    )

    # Monitor command (Session 83K-49)
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Check positions for exit conditions'
    )
    monitor_parser.add_argument(
        '--execute', '-x',
        action='store_true',
        help='Execute detected exits'
    )
    monitor_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    # Monitor-stats command (Session 83K-49)
    subparsers.add_parser('monitor-stats', help='Show monitoring statistics')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Route to command
    if args.command == 'start':
        return cmd_start(args)
    elif args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'scan-all':
        return cmd_scan_all(args)
    elif args.command == 'test':
        return cmd_test(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'cleanup':
        return cmd_cleanup(args)
    elif args.command == 'execute':
        return cmd_execute(args)
    elif args.command == 'positions':
        return cmd_positions(args)
    elif args.command == 'close':
        return cmd_close(args)
    elif args.command == 'monitor':
        return cmd_monitor(args)
    elif args.command == 'monitor-stats':
        return cmd_monitor_stats(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
