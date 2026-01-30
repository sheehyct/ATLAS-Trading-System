#!/usr/bin/env python3
"""
Optimal Stop Placement Analysis Script

Answers the key question: What stop distance minimizes getting shaken out
while protecting against large losses?

Analyses:
1. MAE Distribution by Exit Reason - Were stops hit? Did targets get reached?
2. Stop Accuracy - How often were planned stops actually hit?
3. Optimal Stop Simulation - Simulate different stop distances
4. Pattern-Specific Stop Recommendations - Different patterns may need different stops

Note: Requires trades with position data (stops, targets) and excursion data (MAE).
Historical trades may have partial data.

Usage:
    python scripts/analyze_optimal_stops.py

Output:
    - Console report with insights
    - CSV export to output/optimal_stops_report.csv

Session: EQUITY-98
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.trade_analytics import TradeStore
from core.trade_analytics.models import EnrichedTradeRecord


def analyze_mae_by_exit_reason(
    trades: List[EnrichedTradeRecord],
) -> Dict[str, Dict[str, Any]]:
    """
    MAE distribution by exit reason.

    Shows how MAE varies between:
    - Stopped out trades (MAE likely at/near stop)
    - Target hit trades (MAE can be anything)
    - Manual/other exits
    """
    exit_groups: Dict[str, List[EnrichedTradeRecord]] = {}

    for trade in trades:
        reason = trade.exit_reason or "UNKNOWN"
        # Normalize exit reasons
        if "STOP" in reason.upper():
            key = "STOP"
        elif "TARGET" in reason.upper():
            key = "TARGET"
        elif "EOD" in reason.upper() or "DTE" in reason.upper():
            key = "TIME_EXIT"
        else:
            key = "OTHER"

        if key not in exit_groups:
            exit_groups[key] = []
        exit_groups[key].append(trade)

    results = {}
    for reason, group_trades in exit_groups.items():
        with_mae = [t for t in group_trades if t.excursion.mae_pnl != 0]

        if with_mae:
            mae_values = [t.excursion.mae_pnl for t in with_mae]
            mae_pcts = [t.excursion.mae_pct for t in with_mae if t.excursion.mae_pct != 0]

            results[reason] = {
                "count": len(group_trades),
                "with_mae_data": len(with_mae),
                "win_rate": round(sum(1 for t in group_trades if t.is_winner) / len(group_trades) * 100, 1),
                "avg_mae": round(statistics.mean(mae_values), 2),
                "median_mae": round(statistics.median(mae_values), 2),
                "max_mae": round(min(mae_values), 2),  # min because MAE is negative
                "avg_mae_pct": round(statistics.mean(mae_pcts), 2) if mae_pcts else 0,
            }
        else:
            results[reason] = {
                "count": len(group_trades),
                "with_mae_data": 0,
                "win_rate": round(sum(1 for t in group_trades if t.is_winner) / len(group_trades) * 100, 1) if group_trades else 0,
                "message": "No MAE data available",
            }

    return results


def analyze_stop_accuracy(
    trades: List[EnrichedTradeRecord],
) -> Dict[str, Any]:
    """
    How accurate are current stop placements?

    Compares planned stop prices to actual exit prices and MAE.
    """
    # Filter trades with stop data
    with_stops = [t for t in trades if t.position.stop_price != 0]

    if not with_stops:
        return {
            "trades_with_stop_data": 0,
            "message": "No trades with planned stop prices",
        }

    # Analyze stop performance
    stop_hit = 0
    stop_never_reached = 0
    stopped_but_recovered = 0

    for trade in with_stops:
        stop_price = trade.position.stop_price
        exit_price = trade.exit_price
        mae_price = trade.excursion.mae_price if trade.excursion.mae_price != 0 else None

        # Determine if stop was hit
        direction = trade.pattern.direction
        if direction in ["LONG", "CALL"]:
            hit_stop = exit_price <= stop_price
            mae_hit_stop = mae_price <= stop_price if mae_price else False
        else:  # SHORT, PUT
            hit_stop = exit_price >= stop_price
            mae_hit_stop = mae_price >= stop_price if mae_price else False

        if hit_stop:
            stop_hit += 1
        elif mae_hit_stop and not hit_stop:
            stopped_but_recovered += 1
        else:
            stop_never_reached += 1

    total = len(with_stops)
    return {
        "trades_with_stop_data": total,
        "stop_hit": stop_hit,
        "stop_hit_pct": round(stop_hit / total * 100, 1),
        "stop_never_reached": stop_never_reached,
        "stop_never_reached_pct": round(stop_never_reached / total * 100, 1),
        "mae_hit_but_recovered": stopped_but_recovered,
        "mae_hit_but_recovered_pct": round(stopped_but_recovered / total * 100, 1),
    }


def simulate_stop_distances(
    trades: List[EnrichedTradeRecord],
    distances: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Simulate what-if scenarios for different stop distances.

    Uses MAE data to determine if a trade would have been stopped out
    at various distances from entry.

    Args:
        trades: Trades with excursion data
        distances: List of ATR multiples or percentage distances to test

    Note: This is a simplified simulation that assumes:
    - Stop distance is measured from entry price
    - Uses MAE to determine if stop would be hit
    """
    if distances is None:
        distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Filter trades with both entry price and MAE data
    valid_trades = [t for t in trades
                   if t.position.actual_entry_price != 0
                   and t.excursion.mae_pct != 0]

    if not valid_trades:
        return [{
            "stop_pct": d,
            "message": "No trades with entry price and MAE data"
        } for d in distances]

    results = []
    for stop_pct in distances:
        # Count how many trades would be stopped out at this distance
        stopped = 0
        would_win = 0
        would_lose = 0

        for trade in valid_trades:
            # MAE pct is already relative to entry
            mae_magnitude = abs(trade.excursion.mae_pct)

            if mae_magnitude >= stop_pct:
                stopped += 1
                # If stopped, P&L is -stop_pct (simplified)
                would_lose += 1
            else:
                # Kept the position, actual outcome
                if trade.is_winner:
                    would_win += 1
                else:
                    would_lose += 1

        results.append({
            "stop_pct": stop_pct,
            "trades_analyzed": len(valid_trades),
            "stopped_out": stopped,
            "stopped_out_pct": round(stopped / len(valid_trades) * 100, 1),
            "remaining_wins": would_win,
            "remaining_losses": len(valid_trades) - stopped - would_win,
            "effective_win_rate": round(would_win / len(valid_trades) * 100, 1) if valid_trades else 0,
        })

    return results


def recommend_stops_by_pattern(
    trades: List[EnrichedTradeRecord],
) -> List[Dict[str, Any]]:
    """
    Pattern-specific stop recommendations based on MAE distribution.

    Different patterns may have different optimal stop distances.
    """
    # Group by pattern type
    pattern_trades: Dict[str, List[EnrichedTradeRecord]] = {}
    for trade in trades:
        if trade.excursion.mae_pct != 0:
            pattern = trade.pattern.pattern_type
            if pattern not in pattern_trades:
                pattern_trades[pattern] = []
            pattern_trades[pattern].append(trade)

    results = []
    for pattern, ptrades in sorted(pattern_trades.items()):
        if len(ptrades) < 2:
            results.append({
                "pattern": pattern,
                "trades_with_data": len(ptrades),
                "recommendation": "Insufficient data",
            })
            continue

        mae_pcts = [abs(t.excursion.mae_pct) for t in ptrades]
        winners = [t for t in ptrades if t.is_winner]
        losers = [t for t in ptrades if not t.is_winner]

        # Calculate MAE stats
        mean_mae = statistics.mean(mae_pcts)
        median_mae = statistics.median(mae_pcts)

        # Winner vs loser MAE
        winner_mae = [abs(t.excursion.mae_pct) for t in winners] if winners else []
        loser_mae = [abs(t.excursion.mae_pct) for t in losers] if losers else []

        avg_winner_mae = statistics.mean(winner_mae) if winner_mae else 0
        avg_loser_mae = statistics.mean(loser_mae) if loser_mae else 0

        # Recommendation: stop should be > typical winner MAE but < typical loser MAE
        if avg_winner_mae > 0 and avg_loser_mae > avg_winner_mae:
            recommended_stop = (avg_winner_mae + avg_loser_mae) / 2
            confidence = "Medium"
        elif avg_winner_mae > 0:
            recommended_stop = avg_winner_mae * 1.5  # 50% buffer above winner MAE
            confidence = "Low - limited loser data"
        else:
            recommended_stop = median_mae * 1.2
            confidence = "Low - using median"

        results.append({
            "pattern": pattern,
            "trades_with_data": len(ptrades),
            "win_rate": round(len(winners) / len(ptrades) * 100, 1),
            "mean_mae_pct": round(mean_mae, 2),
            "median_mae_pct": round(median_mae, 2),
            "avg_winner_mae": round(avg_winner_mae, 2),
            "avg_loser_mae": round(avg_loser_mae, 2),
            "recommended_stop_pct": round(recommended_stop, 2),
            "confidence": confidence,
        })

    return sorted(results, key=lambda x: x.get("trades_with_data", 0), reverse=True)


def print_report(
    mae_by_exit: Dict[str, Dict[str, Any]],
    stop_accuracy: Dict[str, Any],
    stop_sim: List[Dict[str, Any]],
    pattern_stops: List[Dict[str, Any]],
    data_coverage: Dict[str, int],
) -> None:
    """Print formatted console report."""

    print("\n" + "=" * 70)
    print("OPTIMAL STOP PLACEMENT ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Data coverage
    print("\nDATA COVERAGE")
    print("-" * 70)
    print(f"Total trades: {data_coverage['total']}")
    print(f"With MAE data: {data_coverage['with_mae']} ({data_coverage['mae_coverage_pct']}%)")
    print(f"With stop price: {data_coverage['with_stops']} ({data_coverage['stop_coverage_pct']}%)")

    if data_coverage['with_mae'] == 0:
        print("\n*** INSUFFICIENT MAE DATA ***")
        print("Historical trades lack MAE data. This report will populate as new trades accumulate.")
        print("=" * 70)
        return

    # Section 1: MAE by Exit Reason
    print("\n\n1. MAE BY EXIT REASON")
    print("-" * 70)
    print(f"{'Exit Reason':<15} {'Count':>7} {'Win%':>7} {'Avg MAE':>10} {'Med MAE':>10} {'Max MAE':>10}")
    print("-" * 70)

    for reason, stats in sorted(mae_by_exit.items()):
        if stats.get("with_mae_data", 0) > 0:
            print(f"{reason:<15} {stats['count']:>7} {stats['win_rate']:>6.1f}% ${stats['avg_mae']:>8.2f} ${stats['median_mae']:>8.2f} ${stats['max_mae']:>8.2f}")
        else:
            print(f"{reason:<15} {stats['count']:>7} {stats['win_rate']:>6.1f}% {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    # Section 2: Stop Accuracy
    print("\n\n2. STOP ACCURACY")
    print("-" * 70)

    if stop_accuracy.get("trades_with_stop_data", 0) > 0:
        print(f"Trades with planned stops: {stop_accuracy['trades_with_stop_data']}")
        print(f"\nOutcomes:")
        print(f"  Stop hit (exited at stop): {stop_accuracy['stop_hit']} ({stop_accuracy['stop_hit_pct']}%)")
        print(f"  Stop never reached: {stop_accuracy['stop_never_reached']} ({stop_accuracy['stop_never_reached_pct']}%)")
        print(f"  MAE hit stop but recovered: {stop_accuracy['mae_hit_but_recovered']} ({stop_accuracy['mae_hit_but_recovered_pct']}%)")

        if stop_accuracy['mae_hit_but_recovered'] > stop_accuracy['stop_hit']:
            print("\nINSIGHT: Many trades touched stop level but weren't exited.")
            print("Consider if stops are being managed correctly or if they're too tight.")
    else:
        print(stop_accuracy.get("message", "No stop data available"))

    # Section 3: Stop Distance Simulation
    print("\n\n3. STOP DISTANCE SIMULATION")
    print("-" * 70)

    if stop_sim and stop_sim[0].get("trades_analyzed", 0) > 0:
        print("What-if analysis: How many trades would be stopped at each distance?")
        print(f"\n{'Stop %':<10} {'Stopped':>10} {'Stop %':>10} {'Win Rate':>10}")
        print("-" * 70)

        for sim in stop_sim:
            print(f"{sim['stop_pct']:<10.1f} {sim['stopped_out']:>10} {sim['stopped_out_pct']:>9.1f}% {sim['effective_win_rate']:>9.1f}%")

        # Find optimal point
        best = max(stop_sim, key=lambda x: x.get("effective_win_rate", 0))
        print(f"\nOptimal stop distance: {best['stop_pct']}% (highest effective win rate: {best['effective_win_rate']}%)")
    else:
        print("Insufficient data for stop simulation")

    # Section 4: Pattern-Specific Recommendations
    print("\n\n4. PATTERN-SPECIFIC STOP RECOMMENDATIONS")
    print("-" * 70)

    has_data = [p for p in pattern_stops if p.get("trades_with_data", 0) >= 2]
    if has_data:
        print(f"{'Pattern':<12} {'Trades':>7} {'Win%':>7} {'Avg WinMAE':>11} {'Avg LossMAE':>12} {'Rec Stop':>10} Confidence")
        print("-" * 70)

        for p in has_data:
            print(f"{p['pattern']:<12} {p['trades_with_data']:>7} {p['win_rate']:>6.1f}% {p['avg_winner_mae']:>10.2f}% {p['avg_loser_mae']:>11.2f}% {p['recommended_stop_pct']:>9.2f}% {p['confidence']}")
    else:
        print("Insufficient data for pattern-specific recommendations")

    print("\n" + "=" * 70)


def export_to_csv(
    pattern_stops: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export pattern stop recommendations to CSV."""

    if not pattern_stops:
        return

    fieldnames = [
        'pattern', 'trades_with_data', 'win_rate', 'mean_mae_pct', 'median_mae_pct',
        'avg_winner_mae', 'avg_loser_mae', 'recommended_stop_pct', 'confidence'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pattern_stops)

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

    trades = store.get_closed_trades()
    print(f"Found {len(trades)} closed trades")

    if not trades:
        print("No closed trades available for analysis.")
        sys.exit(0)

    # Data coverage
    with_mae = [t for t in trades if t.excursion.mae_pnl != 0]
    with_stops = [t for t in trades if t.position.stop_price != 0]

    data_coverage = {
        "total": len(trades),
        "with_mae": len(with_mae),
        "mae_coverage_pct": round(len(with_mae) / len(trades) * 100, 1) if trades else 0,
        "with_stops": len(with_stops),
        "stop_coverage_pct": round(len(with_stops) / len(trades) * 100, 1) if trades else 0,
    }

    print(f"\nData coverage: {data_coverage['with_mae']}/{data_coverage['total']} trades have MAE data")

    # Run analyses
    print("Running optimal stop analyses...")

    mae_by_exit = analyze_mae_by_exit_reason(trades)
    stop_accuracy = analyze_stop_accuracy(trades)
    stop_sim = simulate_stop_distances(with_mae)
    pattern_stops = recommend_stops_by_pattern(trades)

    # Print report
    print_report(mae_by_exit, stop_accuracy, stop_sim, pattern_stops, data_coverage)

    # Export CSV
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "optimal_stops_report.csv"
    export_to_csv(pattern_stops, csv_path)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
