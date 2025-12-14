#!/usr/bin/env python
"""
Crypto STRAT Signal Daemon - Command Line Entry Point - Session CRYPTO-4

Start the autonomous crypto signal detection daemon for paper trading.

Usage:
    # Start daemon (runs continuously, 24/7)
    uv run python scripts/run_crypto_daemon.py start

    # Start with execution disabled (signals only)
    uv run python scripts/run_crypto_daemon.py start --no-execute

    # Run a single scan
    uv run python scripts/run_crypto_daemon.py scan

    # Show current daemon status
    uv run python scripts/run_crypto_daemon.py status

    # Show paper trading positions
    uv run python scripts/run_crypto_daemon.py positions

    # Show paper trading performance
    uv run python scripts/run_crypto_daemon.py performance

    # Show current leverage tier
    uv run python scripts/run_crypto_daemon.py leverage

    # Reset paper trading account
    uv run python scripts/run_crypto_daemon.py reset --yes

VPS Deployment:
    # Copy to VPS
    scp scripts/run_crypto_daemon.py atlas@178.156.223.251:~/vectorbt-workspace/scripts/

    # Install systemd service (on VPS)
    sudo cp deploy/atlas-crypto-daemon.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable atlas-crypto-daemon
    sudo systemctl start atlas-crypto-daemon

    # Check status
    sudo systemctl status atlas-crypto-daemon
    sudo journalctl -u atlas-crypto-daemon -f

Environment Variables:
    COINBASE_API_KEY: Coinbase Advanced API key
    COINBASE_API_SECRET: Coinbase Advanced API secret
    CRYPTO_SYMBOLS: Comma-separated symbols (default: BTC-PERP-INTX,ETH-PERP-INTX)
    CRYPTO_PAPER_BALANCE: Starting paper balance (default: 1000.0)
    CRYPTO_LOG_LEVEL: Log level (default: INFO)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / '.env')


def setup_logging(level: str = 'INFO', log_file: str = None) -> None:
    """Configure logging for the daemon."""
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )


def cmd_start(args: argparse.Namespace) -> int:
    """Start the daemon."""
    from crypto import CryptoSignalDaemon, CryptoDaemonConfig
    from crypto import config

    print("=" * 60)
    print("CRYPTO STRAT SIGNAL DAEMON")
    print("=" * 60)
    print("24/7 Operation Mode")
    print()

    # Get symbols from env or use defaults
    symbols_str = os.environ.get('CRYPTO_SYMBOLS', '')
    symbols = symbols_str.split(',') if symbols_str else config.CRYPTO_SYMBOLS

    # Get paper balance from env or use default
    paper_balance = float(os.environ.get('CRYPTO_PAPER_BALANCE', config.DEFAULT_PAPER_BALANCE))

    # Override from args
    if args.symbols:
        symbols = args.symbols.split(',')
    if args.balance:
        paper_balance = args.balance

    # Discord webhook - Session CRYPTO-5
    discord_webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')

    # Create config
    daemon_config = CryptoDaemonConfig(
        symbols=symbols,
        paper_balance=paper_balance,
        enable_execution=not args.no_execute,
        discord_webhook_url=discord_webhook_url,
    )

    print(f"Configuration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Paper Balance: ${paper_balance:,.2f}")
    print(f"  Execution: {'Enabled' if daemon_config.enable_execution else 'Disabled'}")
    print(f"  Discord Alerts: {'Enabled' if discord_webhook_url else 'Disabled'}")
    print(f"  Scan Interval: {daemon_config.scan_interval}s")
    print(f"  Entry Poll Interval: {daemon_config.entry_poll_interval}s")
    print()

    # Show leverage tier info
    from crypto import get_current_leverage_tier, is_intraday_window
    try:
        import pytz
        now_et = datetime.now(pytz.timezone('America/New_York'))
    except ImportError:
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone.utc) - timedelta(hours=5)

    tier = get_current_leverage_tier(now_et)
    print(f"Current Leverage Tier: {tier.upper()} ({'10x' if tier == 'intraday' else '4x'})")
    if is_intraday_window(now_et):
        print(f"  Intraday leverage available")
    else:
        print(f"  In 4-6PM ET gap (swing leverage only)")
    print()

    # Create and start daemon
    try:
        daemon = CryptoSignalDaemon(config=daemon_config)

        print("Starting daemon (Ctrl+C to stop)...")
        print()
        daemon.start(block=True)

    except KeyboardInterrupt:
        print("\nShutdown requested")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("Daemon error")
        return 1

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Run a single scan."""
    from crypto import CryptoSignalDaemon, CryptoDaemonConfig
    from crypto import config

    symbols_str = os.environ.get('CRYPTO_SYMBOLS', '')
    symbols = symbols_str.split(',') if symbols_str else config.CRYPTO_SYMBOLS

    if args.symbols:
        symbols = args.symbols.split(',')

    print(f"Running crypto scan...")
    print(f"Symbols: {', '.join(symbols)}")
    print()

    daemon_config = CryptoDaemonConfig(
        symbols=symbols,
        enable_execution=False,  # Scan only
    )
    daemon = CryptoSignalDaemon(config=daemon_config)

    signals = daemon.run_scan_and_monitor()

    if signals:
        print(f"\nFound {len(signals)} signal(s):")
        for s in signals:
            print(
                f"  [{s.direction}] {s.symbol} {s.pattern_type} ({s.timeframe}) "
                f"@ ${s.entry_trigger:,.2f} -> ${s.target_price:,.2f} "
                f"(R:R {s.risk_reward:.2f}) [{s.signal_type}]"
            )
    else:
        print("\nNo signals found")

    # Show pending setups
    pending = daemon.get_pending_setups()
    if pending:
        print(f"\nPending SETUP signals ({len(pending)}):")
        for s in pending:
            print(f"  {s.symbol} {s.pattern_type} ({s.timeframe}) waiting for trigger")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show daemon status."""
    from crypto import CryptoSignalDaemon

    daemon = CryptoSignalDaemon()
    daemon.print_status()

    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    """Show paper trading positions."""
    from crypto.simulation.paper_trader import PaperTrader

    trader = PaperTrader(account_name="crypto_daemon")

    print("Paper Trading Positions")
    print("=" * 60)

    summary = trader.get_account_summary()
    print(f"Account: {summary['account_name']}")
    print(f"Balance: ${summary['current_balance']:,.2f}")
    print(f"Starting: ${summary['starting_balance']:,.2f}")
    print(f"Realized P&L: ${summary['realized_pnl']:+,.2f}")
    print(f"Return: {summary['return_percent']:+.2f}%")
    print()

    open_trades = trader.account.open_trades
    if open_trades:
        print(f"Open Trades ({len(open_trades)}):")
        for trade in open_trades:
            print(f"  {trade.trade_id}")
            print(f"    {trade.symbol} {trade.side} qty={trade.quantity:.6f}")
            print(f"    Entry: ${trade.entry_price:,.2f} @ {trade.entry_time}")
            print()
    else:
        print("No open trades")

    return 0


def cmd_performance(args: argparse.Namespace) -> int:
    """Show paper trading performance."""
    from crypto.simulation.paper_trader import PaperTrader

    trader = PaperTrader(account_name="crypto_daemon")

    print("Paper Trading Performance")
    print("=" * 60)

    metrics = trader.get_performance_metrics()

    if 'message' in metrics:
        print(metrics['message'])
        return 0

    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print()
    print(f"Total P&L: ${metrics['total_pnl']:+,.2f}")
    print(f"Gross Profit: ${metrics['gross_profit']:,.2f}")
    print(f"Gross Loss: ${metrics['gross_loss']:,.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print()
    print(f"Average Win: ${metrics['avg_win']:,.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
    print(f"Expectancy: ${metrics['expectancy']:,.2f}")
    print()
    print(f"Largest Win: ${metrics['largest_win']:,.2f}")
    print(f"Largest Loss: ${metrics['largest_loss']:,.2f}")

    return 0


def cmd_leverage(args: argparse.Namespace) -> int:
    """Show current leverage tier."""
    from crypto import (
        get_current_leverage_tier,
        is_intraday_window,
        get_max_leverage_for_symbol,
        time_until_intraday_close_et,
    )
    from crypto import config

    try:
        import pytz
        now_et = datetime.now(pytz.timezone('America/New_York'))
    except ImportError:
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone.utc) - timedelta(hours=5)

    print("Leverage Tier Status")
    print("=" * 60)
    print(f"Current Time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tier = get_current_leverage_tier(now_et)
    is_intraday = is_intraday_window(now_et)

    print(f"Current Tier: {tier.upper()}")
    print(f"Intraday Available: {'Yes' if is_intraday else 'No'}")
    print()

    if is_intraday:
        time_remaining = time_until_intraday_close_et(now_et)
        hours = time_remaining.total_seconds() / 3600
        print(f"Time until 4PM ET close: {hours:.1f} hours")
        print(f"  (Must close intraday positions before 4PM ET)")
    else:
        print("Currently in 4-6PM ET gap")
        print("  (Only swing leverage (4x) available)")
    print()

    print("Symbol Leverage Limits:")
    for symbol in config.CRYPTO_SYMBOLS:
        max_lev = get_max_leverage_for_symbol(symbol, now_et)
        print(f"  {symbol}: {max_lev:.0f}x")

    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Reset paper trading account."""
    from crypto.simulation.paper_trader import PaperTrader
    from crypto import config

    if not args.yes:
        confirm = input("Reset paper trading account? This will clear all trades. [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted")
            return 0

    balance = args.balance if args.balance else config.DEFAULT_PAPER_BALANCE

    trader = PaperTrader(account_name="crypto_daemon")
    trader.reset(starting_balance=balance)

    print(f"Paper trading account reset with ${balance:,.2f}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Show trade history."""
    from crypto.simulation.paper_trader import PaperTrader

    trader = PaperTrader(account_name="crypto_daemon")

    print("Trade History")
    print("=" * 60)

    history = trader.get_trade_history(limit=args.limit)

    if not history:
        print("No trades yet")
        return 0

    for trade in history:
        status = trade.get('status', 'UNKNOWN')
        symbol = trade.get('symbol', 'N/A')
        side = trade.get('side', 'N/A')
        qty = trade.get('quantity', 0)
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price')
        pnl = trade.get('pnl', 0)

        print(f"  [{status}] {trade.get('trade_id', 'N/A')}")
        print(f"    {symbol} {side} qty={qty:.6f} @ ${entry:,.2f}")
        if exit_price:
            print(f"    Exit: ${exit_price:,.2f} | P&L: ${pnl:+,.2f}")
        print()

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Crypto STRAT Signal Daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the daemon')
    start_parser.add_argument(
        '--symbols',
        help='Comma-separated symbols (e.g., BTC-PERP-INTX,ETH-PERP-INTX)'
    )
    start_parser.add_argument(
        '--balance',
        type=float,
        help='Paper trading starting balance'
    )
    start_parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Disable trade execution (signals only)'
    )

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Run single scan')
    scan_parser.add_argument(
        '--symbols',
        help='Comma-separated symbols to scan'
    )

    # Status command
    subparsers.add_parser('status', help='Show daemon status')

    # Positions command
    subparsers.add_parser('positions', help='Show paper trading positions')

    # Performance command
    subparsers.add_parser('performance', help='Show paper trading performance')

    # Leverage command
    subparsers.add_parser('leverage', help='Show current leverage tier')

    # History command
    history_parser = subparsers.add_parser('history', help='Show trade history')
    history_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Number of trades to show'
    )

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset paper trading account')
    reset_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    reset_parser.add_argument(
        '--balance',
        type=float,
        help='New starting balance'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Route to command
    if args.command == 'start':
        return cmd_start(args)
    elif args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'positions':
        return cmd_positions(args)
    elif args.command == 'performance':
        return cmd_performance(args)
    elif args.command == 'leverage':
        return cmd_leverage(args)
    elif args.command == 'history':
        return cmd_history(args)
    elif args.command == 'reset':
        return cmd_reset(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
