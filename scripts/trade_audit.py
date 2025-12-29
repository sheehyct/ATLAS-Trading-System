#!/usr/bin/env python3
"""
Trade Audit CLI - Correlate trades with code versions and fixes.

PURPOSE:
Answer questions like:
- "Was this trade before or after fix EQUITY-35?"
- "Which trades used the buggy code?"
- "Has this fix been verified with a real trade?"

Usage:
    # Show all trades with their code versions
    python scripts/trade_audit.py list

    # Filter trades after a specific fix
    python scripts/trade_audit.py after EQUITY-35

    # Show which fixes apply to a specific trade
    python scripts/trade_audit.py trace <trade_id>

    # Generate audit report
    python scripts/trade_audit.py report

    # Record a new fix
    python scripts/trade_audit.py fix-add --session EQUITY-36 --desc "..." --components "..."

    # Mark a fix as verified
    python scripts/trade_audit.py fix-verify EQUITY-35 --notes "Verified with trade PT_20251229"

    # Show unverified fixes
    python scripts/trade_audit.py unverified
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct imports using importlib to avoid utils/__init__.py pulling in numpy dependencies
import importlib.util

def _import_module_directly(module_name: str, file_path: Path):
    """Import a module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import our utilities directly
_fix_manifest_module = _import_module_directly(
    'fix_manifest_direct', PROJECT_ROOT / 'utils' / 'fix_manifest.py'
)
_version_tracker_module = _import_module_directly(
    'version_tracker_direct', PROJECT_ROOT / 'utils' / 'version_tracker.py'
)

FixManifest = _fix_manifest_module.FixManifest
record_current_commit_as_fix = _fix_manifest_module.record_current_commit_as_fix
get_version_info = _version_tracker_module.get_version_info
get_affected_systems = _version_tracker_module.get_affected_systems


# Trade data paths
PAPER_TRADES_PATH = Path(__file__).parent.parent / 'paper_trades' / 'paper_trades.json'
SIGNALS_PATH = Path(__file__).parent.parent / 'data' / 'signals' / 'signals.json'


def load_paper_trades() -> List[Dict[str, Any]]:
    """Load paper trades from JSON."""
    if not PAPER_TRADES_PATH.exists():
        return []
    try:
        with open(PAPER_TRADES_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get('trades', [])
    except (json.JSONDecodeError, KeyError):
        return []


def load_signals() -> List[Dict[str, Any]]:
    """Load signals from signal store."""
    if not SIGNALS_PATH.exists():
        return []
    try:
        with open(SIGNALS_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return list(data.get('signals', {}).values())
            return []
    except (json.JSONDecodeError, KeyError):
        return []


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError:
            try:
                return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return None
    return None


def cmd_list(args):
    """List all trades with their code versions."""
    trades = load_paper_trades()
    manifest = FixManifest()

    print("=" * 90)
    print("TRADE LIST WITH CODE VERSIONS")
    print("=" * 90)

    if not trades:
        print("No trades found.")
        return

    # Sort by created_at
    trades = sorted(trades, key=lambda t: t.get('created_at', ''), reverse=True)

    # Limit output
    limit = args.limit or 20
    trades = trades[:limit]

    print(f"{'TRADE_ID':<20} {'SYMBOL':<8} {'STATUS':<10} {'VERSION':<12} {'CREATED':<20}")
    print("-" * 90)

    for trade in trades:
        trade_id = trade.get('trade_id', 'N/A')[:18]
        symbol = trade.get('symbol', 'N/A')
        status = trade.get('status', 'N/A')
        version = trade.get('code_version', 'pre-tracking')
        session = trade.get('code_session', '')
        created = trade.get('created_at', 'N/A')[:19]

        version_display = f"{version}/{session}" if session else version

        print(f"{trade_id:<20} {symbol:<8} {status:<10} {version_display:<12} {created:<20}")

    print("-" * 90)
    print(f"Showing {len(trades)} trades (use --limit to change)")

    # Show how many have version tracking
    with_version = sum(1 for t in trades if t.get('code_version'))
    print(f"Trades with version tracking: {with_version}/{len(trades)}")


def cmd_after(args):
    """Show trades that happened after a specific fix."""
    manifest = FixManifest()
    fix = manifest.get_fix(args.session_id)

    if not fix:
        print(f"Error: Fix '{args.session_id}' not found in manifest.")
        print("\nAvailable fixes:")
        for entry in manifest.entries[-10:]:
            print(f"  - {entry.session_id}: {entry.description[:50]}")
        return

    print("=" * 90)
    print(f"TRADES AFTER FIX: {args.session_id}")
    print(f"Fix deployed: {fix.deployed_at}")
    print(f"Description: {fix.description}")
    print(f"Expected impact: {fix.expected_impact}")
    print("=" * 90)

    fix_timestamp = parse_timestamp(fix.deployed_at)
    if not fix_timestamp:
        print("Error: Could not parse fix deployment timestamp.")
        return

    trades = load_paper_trades()
    after_trades = []

    for trade in trades:
        trade_time = parse_timestamp(trade.get('created_at') or trade.get('entry_time'))
        if trade_time and trade_time > fix_timestamp:
            after_trades.append(trade)

    if not after_trades:
        print("\nNo trades found after this fix.")
        return

    print(f"\n{len(after_trades)} trades after fix:\n")
    print(f"{'TRADE_ID':<20} {'SYMBOL':<8} {'STATUS':<10} {'P&L':<12} {'EXIT_REASON':<15}")
    print("-" * 90)

    for trade in sorted(after_trades, key=lambda t: t.get('created_at', '')):
        trade_id = trade.get('trade_id', 'N/A')[:18]
        symbol = trade.get('symbol', 'N/A')
        status = trade.get('status', 'N/A')
        pnl = trade.get('pnl_dollars', 0)
        exit_reason = trade.get('exit_reason', 'N/A')

        pnl_str = f"${pnl:.2f}" if pnl else "-"
        print(f"{trade_id:<20} {symbol:<8} {status:<10} {pnl_str:<12} {exit_reason:<15}")


def cmd_before(args):
    """Show trades that happened before a specific fix (potentially buggy)."""
    manifest = FixManifest()
    fix = manifest.get_fix(args.session_id)

    if not fix:
        print(f"Error: Fix '{args.session_id}' not found in manifest.")
        return

    print("=" * 90)
    print(f"TRADES BEFORE FIX: {args.session_id} (POTENTIALLY AFFECTED BY BUG)")
    print(f"Fix deployed: {fix.deployed_at}")
    print(f"Description: {fix.description}")
    print("=" * 90)

    fix_timestamp = parse_timestamp(fix.deployed_at)
    if not fix_timestamp:
        print("Error: Could not parse fix deployment timestamp.")
        return

    trades = load_paper_trades()
    before_trades = []

    # Only show trades from a reasonable window before the fix
    for trade in trades:
        trade_time = parse_timestamp(trade.get('created_at') or trade.get('entry_time'))
        if trade_time and trade_time < fix_timestamp:
            before_trades.append(trade)

    if not before_trades:
        print("\nNo trades found before this fix.")
        return

    # Only show last 20
    before_trades = sorted(before_trades, key=lambda t: t.get('created_at', ''), reverse=True)[:20]

    print(f"\n{len(before_trades)} trades before fix (showing most recent 20):\n")
    print(f"{'TRADE_ID':<20} {'SYMBOL':<8} {'STATUS':<10} {'P&L':<12} {'EXIT_REASON':<15}")
    print("-" * 90)

    for trade in before_trades:
        trade_id = trade.get('trade_id', 'N/A')[:18]
        symbol = trade.get('symbol', 'N/A')
        status = trade.get('status', 'N/A')
        pnl = trade.get('pnl_dollars', 0)
        exit_reason = trade.get('exit_reason', 'N/A')

        pnl_str = f"${pnl:.2f}" if pnl else "-"
        print(f"{trade_id:<20} {symbol:<8} {status:<10} {pnl_str:<12} {exit_reason:<15}")

    print("\n⚠️  These trades may have been affected by the bug that was fixed.")


def cmd_trace(args):
    """Trace a specific trade to show which fixes apply to it."""
    trades = load_paper_trades()
    manifest = FixManifest()

    # Find the trade
    trade = None
    for t in trades:
        if t.get('trade_id') == args.trade_id:
            trade = t
            break

    if not trade:
        print(f"Error: Trade '{args.trade_id}' not found.")
        return

    print("=" * 90)
    print(f"TRADE AUDIT TRACE: {args.trade_id}")
    print("=" * 90)

    # Trade details
    print("\nTRADE DETAILS:")
    print(f"  Symbol: {trade.get('symbol')}")
    print(f"  Pattern: {trade.get('pattern_type')} ({trade.get('timeframe')})")
    print(f"  Direction: {trade.get('direction')}")
    print(f"  Status: {trade.get('status')}")
    print(f"  Entry: {trade.get('entry_time')}")
    print(f"  Exit: {trade.get('exit_time')}")
    print(f"  P&L: ${trade.get('pnl_dollars', 0):.2f}")

    # Code version
    print("\nCODE VERSION:")
    if trade.get('code_version'):
        print(f"  Commit: {trade.get('code_version')}")
        print(f"  Session: {trade.get('code_session')}")
        print(f"  Branch: {trade.get('code_branch')}")
        if trade.get('code_dirty'):
            print("  ⚠️  Code had uncommitted changes!")
    else:
        print("  ⚠️  Trade predates version tracking")

    # Find applicable fixes
    trade_time = parse_timestamp(trade.get('created_at') or trade.get('entry_time'))
    if trade_time:
        applicable = manifest.get_applicable_fixes(trade_time, 'equity')
        print(f"\nAPPLICABLE FIXES ({len(applicable)}):")
        if applicable:
            for fix in sorted(applicable, key=lambda f: f.deployed_at, reverse=True)[:10]:
                verified = "✓" if fix.verified else "○"
                print(f"  {verified} [{fix.session_id}] {fix.description[:50]}")
        else:
            print("  No fixes in manifest before this trade.")

        # Show fixes after this trade (trade may be affected by bug)
        after = manifest.get_fixes_after(trade_time)
        if after:
            print(f"\nFIXES DEPLOYED AFTER THIS TRADE ({len(after)}):")
            for fix in sorted(after, key=lambda f: f.deployed_at)[:5]:
                print(f"  ⚠️  [{fix.session_id}] {fix.description[:50]}")
            if len(after) > 5:
                print(f"  ... and {len(after) - 5} more")


def cmd_report(args):
    """Generate full audit report."""
    manifest = FixManifest()
    trades = load_paper_trades()
    version_info = get_version_info()

    print("=" * 90)
    print("ATLAS TRADE AUDIT REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print(f"Current Code: {version_info.version_string}")
    print("=" * 90)

    # Trade statistics
    print("\n## TRADE STATISTICS")
    print(f"Total trades: {len(trades)}")

    with_version = [t for t in trades if t.get('code_version')]
    without_version = [t for t in trades if not t.get('code_version')]
    print(f"With version tracking: {len(with_version)}")
    print(f"Pre-tracking (legacy): {len(without_version)}")

    # Status breakdown
    status_counts = {}
    for t in trades:
        status = t.get('status', 'UNKNOWN')
        status_counts[status] = status_counts.get(status, 0) + 1
    print("\nBy status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Fix manifest report
    print("\n" + manifest.generate_audit_report())

    # Recent session summary
    print("\n## RECENT SESSIONS")
    sessions = {}
    for t in trades:
        session = t.get('code_session', 'pre-tracking')
        if session not in sessions:
            sessions[session] = {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        sessions[session]['count'] += 1
        pnl = t.get('pnl_dollars', 0)
        sessions[session]['pnl'] += pnl
        if pnl > 0:
            sessions[session]['wins'] += 1
        elif pnl < 0:
            sessions[session]['losses'] += 1

    for session, stats in sorted(sessions.items(), reverse=True)[:10]:
        win_rate = stats['wins'] / (stats['wins'] + stats['losses']) * 100 if (stats['wins'] + stats['losses']) > 0 else 0
        print(f"  {session}: {stats['count']} trades, P&L ${stats['pnl']:.2f}, {win_rate:.0f}% win rate")


def cmd_fix_add(args):
    """Add a new fix to the manifest."""
    components = [c.strip() for c in args.components.split(',')]
    systems = [s.strip() for s in args.systems.split(',')]
    fields = [f.strip() for f in args.fields.split(',')] if args.fields else []

    # If session ID provided, use manual approach
    if args.session:
        manifest = FixManifest()
        entry = manifest.add_fix(
            session_id=args.session,
            description=args.desc,
            components=components,
            affected_systems=systems,
            expected_impact=args.impact or "",
            trade_fields_affected=fields
        )
    else:
        entry = record_current_commit_as_fix(
            description=args.desc,
            components=components,
            affected_systems=systems,
            expected_impact=args.impact or "",
            trade_fields_affected=fields
        )

    print(f"✓ Fix recorded: {entry.session_id}")
    print(f"  Commit: {entry.commit_short}")
    print(f"  Description: {entry.description}")
    print(f"  Components: {', '.join(entry.components)}")
    print(f"  Systems: {', '.join(entry.affected_systems)}")
    if entry.expected_impact:
        print(f"  Expected impact: {entry.expected_impact}")


def cmd_fix_verify(args):
    """Mark a fix as verified."""
    manifest = FixManifest()
    success = manifest.mark_verified(args.session_id, args.notes or "")

    if success:
        print(f"✓ Fix {args.session_id} marked as verified.")
        if args.notes:
            print(f"  Notes: {args.notes}")
    else:
        print(f"Error: Fix '{args.session_id}' not found.")


def cmd_unverified(args):
    """Show unverified fixes."""
    manifest = FixManifest()
    unverified = manifest.get_unverified_fixes(args.system)

    print("=" * 70)
    print("UNVERIFIED FIXES")
    if args.system:
        print(f"System: {args.system}")
    print("=" * 70)

    if not unverified:
        print("\nAll fixes have been verified! ✓")
        return

    print(f"\n{len(unverified)} fixes need verification:\n")

    for fix in sorted(unverified, key=lambda f: f.deployed_at, reverse=True):
        print(f"[{fix.session_id}] {fix.commit_short}")
        print(f"  {fix.description}")
        print(f"  Deployed: {fix.deployed_at[:19]}")
        print(f"  Systems: {', '.join(fix.affected_systems)}")
        print(f"  Impact: {fix.expected_impact[:60]}..." if fix.expected_impact else "")
        print()

    print("-" * 70)
    print("To verify a fix after confirming with a trade:")
    print("  python scripts/trade_audit.py fix-verify <SESSION_ID> --notes '...'")


def cmd_version(args):
    """Show current code version."""
    info = get_version_info()

    print("=" * 70)
    print("CURRENT CODE VERSION")
    print("=" * 70)
    print(f"Commit: {info.commit_hash}")
    print(f"Short: {info.commit_short}")
    print(f"Branch: {info.branch}")
    print(f"Session: {info.session_id}")
    print(f"Message: {info.commit_message}")
    print(f"Timestamp: {info.commit_timestamp}")
    print(f"Dirty: {info.is_dirty}")
    if info.tags:
        print(f"Tags: {', '.join(info.tags)}")
    print()
    print("Trade metadata to embed:")
    print(json.dumps(info.trade_metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Trade Audit CLI - Correlate trades with code versions"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list command
    list_parser = subparsers.add_parser('list', help='List trades with versions')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of trades to show')
    list_parser.set_defaults(func=cmd_list)

    # after command
    after_parser = subparsers.add_parser('after', help='Trades after a fix')
    after_parser.add_argument('session_id', help='Fix session ID (e.g., EQUITY-35)')
    after_parser.set_defaults(func=cmd_after)

    # before command
    before_parser = subparsers.add_parser('before', help='Trades before a fix (potentially buggy)')
    before_parser.add_argument('session_id', help='Fix session ID (e.g., EQUITY-35)')
    before_parser.set_defaults(func=cmd_before)

    # trace command
    trace_parser = subparsers.add_parser('trace', help='Trace a specific trade')
    trace_parser.add_argument('trade_id', help='Trade ID to trace')
    trace_parser.set_defaults(func=cmd_trace)

    # report command
    report_parser = subparsers.add_parser('report', help='Generate full audit report')
    report_parser.set_defaults(func=cmd_report)

    # fix-add command
    fix_add_parser = subparsers.add_parser('fix-add', help='Record a new fix')
    fix_add_parser.add_argument('--session', help='Session ID (e.g., EQUITY-35). Auto-detected if not provided.')
    fix_add_parser.add_argument('--desc', required=True, help='Fix description')
    fix_add_parser.add_argument('--components', required=True, help='Affected components (comma-separated)')
    fix_add_parser.add_argument('--systems', default='equity', help='Affected systems (comma-separated)')
    fix_add_parser.add_argument('--impact', help='Expected impact on trades')
    fix_add_parser.add_argument('--fields', help='Affected trade fields (comma-separated)')
    fix_add_parser.set_defaults(func=cmd_fix_add)

    # fix-verify command
    verify_parser = subparsers.add_parser('fix-verify', help='Mark a fix as verified')
    verify_parser.add_argument('session_id', help='Fix session ID')
    verify_parser.add_argument('--notes', help='Verification notes')
    verify_parser.set_defaults(func=cmd_fix_verify)

    # unverified command
    unverified_parser = subparsers.add_parser('unverified', help='Show unverified fixes')
    unverified_parser.add_argument('--system', choices=['equity', 'crypto'], help='Filter by system')
    unverified_parser.set_defaults(func=cmd_unverified)

    # version command
    version_parser = subparsers.add_parser('version', help='Show current code version')
    version_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
