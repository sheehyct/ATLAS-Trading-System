#!/usr/bin/env python3
"""
Pattern Efficiency Analysis Script

Answers the key question: Which patterns leave money on the table?

Analyses:
1. Pattern Win Rate Breakdown - Win rate and P&L by pattern type
2. Pattern Exit Efficiency - Which patterns capture the most profit (requires MFE data)
3. Pattern-Timeframe Matrix - 2D analysis showing performance combinations
4. Losers That Went Green - Patterns where losers frequently went positive first

Usage:
    python scripts/analyze_pattern_efficiency.py

Output:
    - Console report with insights
    - CSV export to output/pattern_efficiency_report.csv

Session: EQUITY-98
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.trade_analytics import TradeStore, TradeAnalyticsEngine
from core.trade_analytics.models import EnrichedTradeRecord


def analyze_pattern_performance(
    analytics: TradeAnalyticsEngine,
) -> List[Dict[str, Any]]:
    """
    Win rate and P&L breakdown by pattern type.

    Returns list of dicts with pattern stats sorted by trade count.
    """
    stats = analytics.win_rate_by_factor("pattern_type")

    results = []
    for s in sorted(stats, key=lambda x: x.trades, reverse=True):
        results.append({
            "pattern": s.segment_value,
            "trades": s.trades,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate": round(s.win_rate, 1),
            "total_pnl": round(s.total_pnl, 2),
            "avg_pnl": round(s.avg_pnl, 2),
            "avg_winner": round(s.avg_winner, 2),
            "avg_loser": round(s.avg_loser, 2),
            "profit_factor": round(s.profit_factor, 2) if s.profit_factor != float('inf') else "N/A",
        })

    return results


def analyze_pattern_exit_efficiency(
    store: TradeStore,
    analytics: TradeAnalyticsEngine,
) -> List[Dict[str, Any]]:
    """
    Exit efficiency breakdown by pattern type.

    Requires MFE data to be populated. Returns stats showing how much
    of available profit each pattern captures.
    """
    trades = store.get_closed_trades()

    # Group trades by pattern type
    pattern_trades: Dict[str, List[EnrichedTradeRecord]] = {}
    for trade in trades:
        pattern = trade.pattern.pattern_type
        if pattern not in pattern_trades:
            pattern_trades[pattern] = []
        pattern_trades[pattern].append(trade)

    results = []
    for pattern, trades_list in sorted(pattern_trades.items()):
        # Count trades with MFE data
        with_mfe = [t for t in trades_list if t.excursion.mfe_pnl != 0]
        winners = [t for t in trades_list if t.is_winner]
        winners_with_mfe = [t for t in winners if t.excursion.mfe_pnl != 0]

        # Calculate efficiency for trades with data
        if winners_with_mfe:
            efficiencies = [t.excursion.exit_efficiency for t in winners_with_mfe]
            avg_efficiency = sum(efficiencies) / len(efficiencies)
            profit_captured = [t.excursion.profit_captured_pct for t in winners_with_mfe]
            avg_captured = sum(profit_captured) / len(profit_captured)
        else:
            avg_efficiency = 0
            avg_captured = 0

        results.append({
            "pattern": pattern,
            "total_trades": len(trades_list),
            "trades_with_mfe": len(with_mfe),
            "winners_with_mfe": len(winners_with_mfe),
            "avg_exit_efficiency": round(avg_efficiency * 100, 1),
            "avg_profit_captured_pct": round(avg_captured, 1),
            "data_coverage_pct": round(len(with_mfe) / len(trades_list) * 100, 1) if trades_list else 0,
        })

    return results


def create_pattern_timeframe_matrix(
    store: TradeStore,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    2D analysis: pattern type vs timeframe.

    Returns nested dict: matrix[pattern][timeframe] = stats
    """
    trades = store.get_closed_trades()

    # Group by pattern and timeframe
    matrix: Dict[str, Dict[str, List[EnrichedTradeRecord]]] = {}

    for trade in trades:
        pattern = trade.pattern.pattern_type
        timeframe = trade.pattern.timeframe

        if pattern not in matrix:
            matrix[pattern] = {}
        if timeframe not in matrix[pattern]:
            matrix[pattern][timeframe] = []
        matrix[pattern][timeframe].append(trade)

    # Calculate stats for each cell
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pattern, tf_trades in sorted(matrix.items()):
        result[pattern] = {}
        for timeframe, trades_list in sorted(tf_trades.items()):
            winners = [t for t in trades_list if t.is_winner]
            total_pnl = sum(t.pnl for t in trades_list)

            result[pattern][timeframe] = {
                "trades": len(trades_list),
                "win_rate": round(len(winners) / len(trades_list) * 100, 1) if trades_list else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(trades_list), 2) if trades_list else 0,
            }

    return result


def analyze_losers_went_green(
    store: TradeStore,
) -> List[Dict[str, Any]]:
    """
    Patterns where losers frequently went positive before losing.

    These are patterns where trailing stops or earlier exits could help.
    Requires excursion data to be populated.
    """
    trades = store.get_closed_trades()

    # Group losers by pattern type
    pattern_losers: Dict[str, List[EnrichedTradeRecord]] = {}
    for trade in trades:
        if not trade.is_winner:
            pattern = trade.pattern.pattern_type
            if pattern not in pattern_losers:
                pattern_losers[pattern] = []
            pattern_losers[pattern].append(trade)

    results = []
    for pattern, losers in sorted(pattern_losers.items()):
        went_green = [l for l in losers if l.excursion.went_green_before_loss]
        had_mfe = [l for l in losers if l.excursion.mfe_pnl > 0]

        # Calculate avg MFE before loss for those that went green
        if had_mfe:
            avg_mfe_before = sum(l.excursion.mfe_pnl for l in had_mfe) / len(had_mfe)
        else:
            avg_mfe_before = 0

        results.append({
            "pattern": pattern,
            "total_losers": len(losers),
            "went_green_count": len(went_green),
            "went_green_pct": round(len(went_green) / len(losers) * 100, 1) if losers else 0,
            "losers_with_mfe_data": len(had_mfe),
            "avg_mfe_before_loss": round(avg_mfe_before, 2),
        })

    return sorted(results, key=lambda x: x["went_green_pct"], reverse=True)


def print_report(
    pattern_performance: List[Dict[str, Any]],
    exit_efficiency: List[Dict[str, Any]],
    matrix: Dict[str, Dict[str, Dict[str, Any]]],
    losers_green: List[Dict[str, Any]],
) -> None:
    """Print formatted console report."""

    print("\n" + "=" * 70)
    print("PATTERN EFFICIENCY ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Section 1: Pattern Performance
    print("\n1. PATTERN WIN RATE BREAKDOWN")
    print("-" * 70)
    print(f"{'Pattern':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'Total P&L':>12} {'Avg P&L':>10} {'PF':>8}")
    print("-" * 70)

    for p in pattern_performance:
        pf_str = str(p['profit_factor']) if p['profit_factor'] != "N/A" else "N/A"
        print(f"{p['pattern']:<12} {p['trades']:>7} {p['wins']:>6} {p['win_rate']:>6.1f}% {p['total_pnl']:>11.2f} {p['avg_pnl']:>10.2f} {pf_str:>8}")

    # Highlight best/worst
    if pattern_performance:
        by_wr = sorted([p for p in pattern_performance if p['trades'] >= 3], key=lambda x: x['win_rate'], reverse=True)
        if by_wr:
            best = by_wr[0]
            worst = by_wr[-1]
            print(f"\nBest Win Rate (min 3 trades): {best['pattern']} at {best['win_rate']}%")
            print(f"Worst Win Rate (min 3 trades): {worst['pattern']} at {worst['win_rate']}%")

    # Section 2: Exit Efficiency
    print("\n\n2. EXIT EFFICIENCY BY PATTERN")
    print("-" * 70)

    has_data = [e for e in exit_efficiency if e['trades_with_mfe'] > 0]
    no_data = [e for e in exit_efficiency if e['trades_with_mfe'] == 0]

    if has_data:
        print(f"{'Pattern':<12} {'Total':>7} {'w/MFE':>7} {'Efficiency':>12} {'Captured%':>12}")
        print("-" * 70)
        for e in has_data:
            print(f"{e['pattern']:<12} {e['total_trades']:>7} {e['trades_with_mfe']:>7} {e['avg_exit_efficiency']:>11.1f}% {e['avg_profit_captured_pct']:>11.1f}%")

    if no_data:
        print(f"\nPatterns awaiting MFE data: {', '.join(e['pattern'] for e in no_data)}")

    # Section 3: Pattern-Timeframe Matrix
    print("\n\n3. PATTERN x TIMEFRAME MATRIX")
    print("-" * 70)

    # Collect all timeframes
    all_tfs = set()
    for pattern_data in matrix.values():
        all_tfs.update(pattern_data.keys())
    all_tfs = sorted(all_tfs, key=lambda x: {'1H': 0, '4H': 1, '1D': 2, '1W': 3, '1M': 4}.get(x, 99))

    # Print header
    header = f"{'Pattern':<12}"
    for tf in all_tfs:
        header += f" {tf:>12}"
    print(header)
    print("-" * 70)

    # Print each pattern row
    for pattern, tf_data in sorted(matrix.items()):
        row = f"{pattern:<12}"
        for tf in all_tfs:
            if tf in tf_data:
                d = tf_data[tf]
                cell = f"{d['trades']}t/{d['win_rate']:.0f}%"
            else:
                cell = "-"
            row += f" {cell:>12}"
        print(row)

    # Section 4: Losers That Went Green
    print("\n\n4. LOSERS THAT WENT GREEN (Opportunity for Earlier Exit)")
    print("-" * 70)

    has_green_data = [l for l in losers_green if l['losers_with_mfe_data'] > 0]

    if has_green_data:
        print(f"{'Pattern':<12} {'Losers':>8} {'Went Green':>12} {'Green %':>10} {'Avg MFE':>12}")
        print("-" * 70)
        for l in has_green_data:
            print(f"{l['pattern']:<12} {l['total_losers']:>8} {l['went_green_count']:>12} {l['went_green_pct']:>9.1f}% {l['avg_mfe_before_loss']:>11.2f}")

        # Insight
        high_green = [l for l in has_green_data if l['went_green_pct'] > 50]
        if high_green:
            print(f"\nINSIGHT: {len(high_green)} pattern(s) have >50% losers that went green first.")
            print("Consider earlier profit-taking or trailing stops for these patterns.")
    else:
        print("Awaiting excursion data from new trades to analyze losers.")

    print("\n" + "=" * 70)


def export_to_csv(
    pattern_performance: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export pattern performance to CSV."""

    if not pattern_performance:
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=pattern_performance[0].keys())
        writer.writeheader()
        writer.writerows(pattern_performance)

    print(f"\nExported to: {output_path}")


def main():
    """Main entry point."""

    # Load trade store
    store_path = project_root / "core" / "trade_analytics" / "data" / "equity_trades.json"

    if not store_path.exists():
        print(f"ERROR: Trade store not found at {store_path}")
        print("Run the trade migration script first.")
        sys.exit(1)

    print(f"Loading trades from: {store_path}")
    store = TradeStore(store_path)
    analytics = TradeAnalyticsEngine(store)

    trades = store.get_closed_trades()
    print(f"Found {len(trades)} closed trades")

    if not trades:
        print("No closed trades available for analysis.")
        sys.exit(0)

    # Run analyses
    print("\nRunning pattern efficiency analyses...")

    pattern_performance = analyze_pattern_performance(analytics)
    exit_efficiency = analyze_pattern_exit_efficiency(store, analytics)
    matrix = create_pattern_timeframe_matrix(store)
    losers_green = analyze_losers_went_green(store)

    # Print report
    print_report(pattern_performance, exit_efficiency, matrix, losers_green)

    # Export CSV
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "pattern_efficiency_report.csv"
    export_to_csv(pattern_performance, csv_path)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
