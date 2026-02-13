"""
Export all trade data from Alpaca SMALL account before paper account reset.

Exports:
1. All fill activities (raw order fills)
2. Closed trades with FIFO P&L matching
3. Open positions (if any)
4. Account summary

Output: data/exports/pre_reset_YYYY-MM-DD/
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from integrations.alpaca_trading_client import AlpacaTradingClient


def serialize(obj):
    """JSON serializer for datetime and UUID objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    import uuid
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    timestamp = datetime.now().strftime('%Y-%m-%d')
    export_dir = Path(__file__).parent.parent / 'data' / 'exports' / f'pre_reset_{timestamp}'
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Export directory: {export_dir}")
    print()

    # Connect to SMALL account
    client = AlpacaTradingClient(account='SMALL')
    if not client.connect():
        print("ERROR: Failed to connect to Alpaca SMALL account")
        sys.exit(1)

    print(f"Connected to Alpaca SMALL account")
    print()

    # 1. Account summary
    print("--- Account Summary ---")
    try:
        account = client.get_account()
        print(f"  Equity: ${account.get('equity', 'N/A')}")
        print(f"  Cash: ${account.get('cash', 'N/A')}")
        print(f"  Buying Power: ${account.get('buying_power', 'N/A')}")
        print(f"  PDT Flagged: {account.get('pattern_day_trader', 'N/A')}")

        with open(export_dir / 'account_summary.json', 'w') as f:
            json.dump(account, f, indent=2, default=serialize)
        print(f"  Saved: account_summary.json")
    except Exception as e:
        print(f"  ERROR getting account info: {e}")
    print()

    # 2. Open positions
    print("--- Open Positions ---")
    try:
        raw_positions = client.client.get_all_positions()
        positions = []
        for pos in raw_positions:
            p = {
                'symbol': pos.symbol,
                'qty': str(pos.qty),
                'side': 'long' if float(pos.qty) > 0 else 'short',
                'avg_entry_price': str(pos.avg_entry_price),
                'market_value': str(pos.market_value),
                'unrealized_pl': str(pos.unrealized_pl),
                'current_price': str(pos.current_price),
            }
            positions.append(p)
            print(f"    {p['symbol']}: {p['qty']} @ ${p['avg_entry_price']}"
                  f" (unrealized P&L: ${p['unrealized_pl']})")
        print(f"  Found {len(positions)} open positions")

        with open(export_dir / 'open_positions.json', 'w') as f:
            json.dump(positions, f, indent=2, default=serialize)
        print(f"  Saved: open_positions.json")
    except Exception as e:
        print(f"  ERROR getting positions: {e}")
    print()

    # 3. All fill activities (raw data)
    print("--- Fill Activities ---")
    try:
        fills = client.get_fill_activities()
        print(f"  Found {len(fills)} total fills")

        # Remove datetime objects for JSON serialization (keep string 'time')
        fills_serializable = []
        for f in fills:
            fill_copy = dict(f)
            fill_copy.pop('time_dt', None)
            fills_serializable.append(fill_copy)

        with open(export_dir / 'all_fills.json', 'w') as f:
            json.dump(fills_serializable, f, indent=2, default=serialize)
        print(f"  Saved: all_fills.json")
    except Exception as e:
        print(f"  ERROR getting fills: {e}")
    print()

    # 4. Closed trades (FIFO matched, options only)
    print("--- Closed Trades (Options, FIFO) ---")
    try:
        closed_options = client.get_closed_trades(options_only=True)
        print(f"  Found {len(closed_options)} closed option trades")

        total_pnl = sum(t['realized_pnl'] for t in closed_options)
        winners = [t for t in closed_options if t['realized_pnl'] > 0]
        losers = [t for t in closed_options if t['realized_pnl'] <= 0]
        win_rate = len(winners) / len(closed_options) * 100 if closed_options else 0

        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Win Rate: {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")

        # Remove datetime objects for serialization
        closed_serializable = []
        for t in closed_options:
            trade_copy = dict(t)
            trade_copy.pop('buy_time_dt', None)
            trade_copy.pop('sell_time_dt', None)
            closed_serializable.append(trade_copy)

        with open(export_dir / 'closed_trades_options.json', 'w') as f:
            json.dump(closed_serializable, f, indent=2, default=serialize)
        print(f"  Saved: closed_trades_options.json")
    except Exception as e:
        print(f"  ERROR getting closed trades: {e}")
    print()

    # 5. Closed trades (ALL instruments)
    print("--- Closed Trades (All Instruments, FIFO) ---")
    try:
        closed_all = client.get_closed_trades(options_only=False)
        print(f"  Found {len(closed_all)} total closed trades")

        closed_all_serializable = []
        for t in closed_all:
            trade_copy = dict(t)
            trade_copy.pop('buy_time_dt', None)
            trade_copy.pop('sell_time_dt', None)
            closed_all_serializable.append(trade_copy)

        with open(export_dir / 'closed_trades_all.json', 'w') as f:
            json.dump(closed_all_serializable, f, indent=2, default=serialize)
        print(f"  Saved: closed_trades_all.json")
    except Exception as e:
        print(f"  ERROR getting all closed trades: {e}")
    print()

    # 6. Copy existing local data files
    print("--- Local Data Backup ---")
    local_files = [
        ('data/enriched_trades.json', 'local_enriched_trades.json'),
        ('data/executions/trade_metadata.json', 'local_trade_metadata.json'),
        ('data/executions/executions.json', 'local_executions.json'),
    ]

    project_root = Path(__file__).parent.parent
    for src_rel, dst_name in local_files:
        src = project_root / src_rel
        if src.exists():
            import shutil
            shutil.copy2(src, export_dir / dst_name)
            print(f"  Copied: {src_rel} -> {dst_name}")
        else:
            print(f"  Skipped (not found): {src_rel}")
    print()

    print("=" * 50)
    print(f"Export complete: {export_dir}")
    print(f"Files: {len(list(export_dir.iterdir()))}")
    print()
    print("Safe to reset the paper account now.")


if __name__ == '__main__':
    main()
