#!/usr/bin/env python3
"""
MAE Recovery Analysis Script

Answers the key question: What's the typical MAE before recovery for winners?

Understanding how much drawdown winners experience before recovering helps
set appropriate stop distances that don't get shaken out prematurely.

Analyses:
1. Winner MAE Distribution - Histogram of MAE values for winners
2. MAE to MFE Ratio - Pain-to-gain ratio by pattern
3. Time to MAE - How quickly do winners hit their worst point?
4. Recovery Pattern Analysis - MAE followed by MFE timing

Note: This script requires trades with excursion data (MFE/MAE tracking).
Historical trades migrated without this data will be flagged but not analyzed.

Usage:
    python scripts/analyze_mae_recovery.py

Output:
    - Console report with insights
    - CSV export to output/mae_recovery_report.csv

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


def get_trades_with_excursion_data(trades: List[EnrichedTradeRecord]) -> List[EnrichedTradeRecord]:
    """Filter trades that have MFE/MAE data populated."""
    return [t for t in trades if t.excursion.mfe_pnl != 0 or t.excursion.mae_pnl != 0]


def analyze_winner_mae_distribution(
    winners: List[EnrichedTradeRecord],
) -> Dict[str, Any]:
    """
    MAE distribution for winning trades.

    Shows how much drawdown winners typically experience before recovering.
    """
    with_mae = [w for w in winners if w.excursion.mae_pnl != 0]

    if not with_mae:
        return {
            "trades_analyzed": 0,
            "message": "No winner trades with MAE data available",
        }

    mae_pnls = [w.excursion.mae_pnl for w in with_mae]
    mae_pcts = [w.excursion.mae_pct for w in with_mae if w.excursion.mae_pct != 0]

    # MAE is typically negative (worst drawdown), but stored as negative
    result = {
        "trades_analyzed": len(with_mae),
        "mae_pnl_stats": {
            "min": round(min(mae_pnls), 2),
            "max": round(max(mae_pnls), 2),
            "mean": round(statistics.mean(mae_pnls), 2),
            "median": round(statistics.median(mae_pnls), 2),
            "stdev": round(statistics.stdev(mae_pnls), 2) if len(mae_pnls) > 1 else 0,
        },
    }

    if mae_pcts:
        result["mae_pct_stats"] = {
            "min": round(min(mae_pcts), 2),
            "max": round(max(mae_pcts), 2),
            "mean": round(statistics.mean(mae_pcts), 2),
            "median": round(statistics.median(mae_pcts), 2),
        }

    # Bucketed distribution
    buckets = {"$0-25": 0, "$25-50": 0, "$50-100": 0, "$100-200": 0, "$200+": 0}
    for mae in mae_pnls:
        abs_mae = abs(mae)
        if abs_mae <= 25:
            buckets["$0-25"] += 1
        elif abs_mae <= 50:
            buckets["$25-50"] += 1
        elif abs_mae <= 100:
            buckets["$50-100"] += 1
        elif abs_mae <= 200:
            buckets["$100-200"] += 1
        else:
            buckets["$200+"] += 1

    result["mae_distribution"] = buckets

    return result


def analyze_mae_mfe_ratio(
    trades: List[EnrichedTradeRecord],
) -> List[Dict[str, Any]]:
    """
    Pain-to-gain ratio by pattern type.

    Lower ratio = less pain per unit of gain = better pattern for tight stops.
    Higher ratio = more pain before gain = may need wider stops.
    """
    # Group by pattern type
    pattern_trades: Dict[str, List[EnrichedTradeRecord]] = {}
    for trade in trades:
        if trade.excursion.mfe_pnl != 0 or trade.excursion.mae_pnl != 0:
            pattern = trade.pattern.pattern_type
            if pattern not in pattern_trades:
                pattern_trades[pattern] = []
            pattern_trades[pattern].append(trade)

    results = []
    for pattern, trades_list in sorted(pattern_trades.items()):
        mae_values = [abs(t.excursion.mae_pnl) for t in trades_list if t.excursion.mae_pnl != 0]
        mfe_values = [t.excursion.mfe_pnl for t in trades_list if t.excursion.mfe_pnl > 0]

        if mae_values and mfe_values:
            avg_mae = statistics.mean(mae_values)
            avg_mfe = statistics.mean(mfe_values)
            pain_gain_ratio = avg_mae / avg_mfe if avg_mfe > 0 else float('inf')
        else:
            avg_mae = 0
            avg_mfe = 0
            pain_gain_ratio = 0

        results.append({
            "pattern": pattern,
            "trades_with_data": len(trades_list),
            "avg_mae_abs": round(avg_mae, 2),
            "avg_mfe": round(avg_mfe, 2),
            "pain_gain_ratio": round(pain_gain_ratio, 2),
            "interpretation": _interpret_ratio(pain_gain_ratio),
        })

    return sorted(results, key=lambda x: x["pain_gain_ratio"])


def _interpret_ratio(ratio: float) -> str:
    """Interpret pain-to-gain ratio."""
    if ratio == 0:
        return "No data"
    elif ratio < 0.3:
        return "Low pain - good for tight stops"
    elif ratio < 0.5:
        return "Moderate pain - normal"
    elif ratio < 1.0:
        return "High pain - may need wider stops"
    else:
        return "Very high pain - review strategy"


def analyze_time_to_mae(
    winners: List[EnrichedTradeRecord],
) -> Dict[str, Any]:
    """
    How quickly do winners hit their worst point (MAE)?

    Early MAE = shakeout risk if stops too tight
    Late MAE = may indicate pattern degradation
    """
    with_timing = [w for w in winners if w.excursion.mae_bars_from_entry > 0]

    if not with_timing:
        return {
            "trades_analyzed": 0,
            "message": "No winner trades with MAE timing data available",
        }

    bars_to_mae = [w.excursion.mae_bars_from_entry for w in with_timing]

    result = {
        "trades_analyzed": len(with_timing),
        "bars_to_mae_stats": {
            "min": min(bars_to_mae),
            "max": max(bars_to_mae),
            "mean": round(statistics.mean(bars_to_mae), 1),
            "median": statistics.median(bars_to_mae),
        },
    }

    # Early vs late MAE
    total_bars = [w.bars_in_trade for w in with_timing if w.bars_in_trade > 0]
    if total_bars:
        early_mae = sum(1 for w in with_timing
                       if w.bars_in_trade > 0 and
                       w.excursion.mae_bars_from_entry < w.bars_in_trade * 0.25)
        mid_mae = sum(1 for w in with_timing
                     if w.bars_in_trade > 0 and
                     0.25 <= w.excursion.mae_bars_from_entry / w.bars_in_trade < 0.75)
        late_mae = sum(1 for w in with_timing
                      if w.bars_in_trade > 0 and
                      w.excursion.mae_bars_from_entry >= w.bars_in_trade * 0.75)

        result["mae_timing"] = {
            "early_first_quarter": early_mae,
            "mid_trade": mid_mae,
            "late_last_quarter": late_mae,
        }

    return result


def analyze_recovery_pattern(
    winners: List[EnrichedTradeRecord],
) -> Dict[str, Any]:
    """
    Analyze the pattern of MAE followed by MFE.

    Understanding how trades recover from drawdowns can inform exit timing.
    """
    with_both = [w for w in winners
                if w.excursion.mae_time is not None and w.excursion.mfe_time is not None]

    if not with_both:
        return {
            "trades_analyzed": 0,
            "message": "No winner trades with full timing data available",
        }

    # Check if MAE happens before MFE (normal pattern)
    mae_before_mfe = 0
    mfe_before_mae = 0

    for w in with_both:
        if w.excursion.mae_time < w.excursion.mfe_time:
            mae_before_mfe += 1
        else:
            mfe_before_mae += 1

    return {
        "trades_analyzed": len(with_both),
        "mae_then_recovery": mae_before_mfe,
        "mfe_then_decline": mfe_before_mae,
        "mae_first_pct": round(mae_before_mfe / len(with_both) * 100, 1),
        "insight": _recovery_insight(mae_before_mfe / len(with_both) if with_both else 0),
    }


def _recovery_insight(mae_first_ratio: float) -> str:
    """Generate insight about recovery pattern."""
    if mae_first_ratio > 0.7:
        return "Most winners dip first then recover - hold through initial drawdown"
    elif mae_first_ratio > 0.5:
        return "Mixed pattern - consider trailing stops after initial profit"
    else:
        return "Winners often peak early then decline - take profits quickly"


def print_report(
    winner_mae: Dict[str, Any],
    mae_mfe_ratio: List[Dict[str, Any]],
    time_to_mae: Dict[str, Any],
    recovery: Dict[str, Any],
    data_coverage: Dict[str, int],
) -> None:
    """Print formatted console report."""

    print("\n" + "=" * 70)
    print("MAE RECOVERY ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Data coverage summary
    print("\nDATA COVERAGE")
    print("-" * 70)
    print(f"Total trades: {data_coverage['total']}")
    print(f"With excursion data: {data_coverage['with_excursion']} ({data_coverage['coverage_pct']}%)")
    print(f"Winners with excursion: {data_coverage['winners_with_excursion']}")

    if data_coverage['with_excursion'] == 0:
        print("\n*** INSUFFICIENT DATA ***")
        print("Historical trades lack MFE/MAE data.")
        print("This report will populate as new trades are tracked in real-time.")
        print("=" * 70)
        return

    # Section 1: Winner MAE Distribution
    print("\n\n1. WINNER MAE DISTRIBUTION (How much drawdown do winners see?)")
    print("-" * 70)

    if winner_mae.get("trades_analyzed", 0) > 0:
        stats = winner_mae.get("mae_pnl_stats", {})
        print(f"Trades analyzed: {winner_mae['trades_analyzed']}")
        print(f"\nMAE (Drawdown) Statistics:")
        print(f"  Mean: ${stats.get('mean', 0):.2f}")
        print(f"  Median: ${stats.get('median', 0):.2f}")
        print(f"  Min: ${stats.get('min', 0):.2f}")
        print(f"  Max: ${stats.get('max', 0):.2f}")
        print(f"  StdDev: ${stats.get('stdev', 0):.2f}")

        dist = winner_mae.get("mae_distribution", {})
        if dist:
            print(f"\nMAE Distribution:")
            for bucket, count in dist.items():
                pct = count / winner_mae['trades_analyzed'] * 100
                bar = "#" * int(pct / 5)
                print(f"  {bucket:<10}: {count:>3} ({pct:>5.1f}%) {bar}")
    else:
        print(winner_mae.get("message", "No data available"))

    # Section 2: Pain-to-Gain Ratio by Pattern
    print("\n\n2. PAIN-TO-GAIN RATIO BY PATTERN")
    print("-" * 70)

    if mae_mfe_ratio:
        print(f"{'Pattern':<12} {'Trades':>7} {'Avg MAE':>10} {'Avg MFE':>10} {'Ratio':>8} Interpretation")
        print("-" * 70)
        for r in mae_mfe_ratio:
            if r['trades_with_data'] > 0:
                print(f"{r['pattern']:<12} {r['trades_with_data']:>7} ${r['avg_mae_abs']:>8.2f} ${r['avg_mfe']:>8.2f} {r['pain_gain_ratio']:>7.2f}  {r['interpretation']}")

        # Insight
        low_pain = [r for r in mae_mfe_ratio if r['pain_gain_ratio'] < 0.5 and r['trades_with_data'] >= 3]
        if low_pain:
            print(f"\nLow-pain patterns (ratio < 0.5): {', '.join(r['pattern'] for r in low_pain)}")
            print("These patterns work well with tighter stops.")
    else:
        print("No pattern data with MFE/MAE available")

    # Section 3: Time to MAE
    print("\n\n3. TIME TO MAE (When do winners hit their worst point?)")
    print("-" * 70)

    if time_to_mae.get("trades_analyzed", 0) > 0:
        stats = time_to_mae.get("bars_to_mae_stats", {})
        print(f"Trades analyzed: {time_to_mae['trades_analyzed']}")
        print(f"\nBars to reach MAE:")
        print(f"  Mean: {stats.get('mean', 0):.1f} bars")
        print(f"  Median: {stats.get('median', 0)} bars")
        print(f"  Range: {stats.get('min', 0)} - {stats.get('max', 0)} bars")

        timing = time_to_mae.get("mae_timing", {})
        if timing:
            print(f"\nMAE Timing Distribution:")
            print(f"  Early (first 25%): {timing.get('early_first_quarter', 0)} trades")
            print(f"  Mid-trade (25-75%): {timing.get('mid_trade', 0)} trades")
            print(f"  Late (last 25%): {timing.get('late_last_quarter', 0)} trades")
    else:
        print(time_to_mae.get("message", "No data available"))

    # Section 4: Recovery Pattern
    print("\n\n4. RECOVERY PATTERN ANALYSIS")
    print("-" * 70)

    if recovery.get("trades_analyzed", 0) > 0:
        print(f"Trades analyzed: {recovery['trades_analyzed']}")
        print(f"\nSequence:")
        print(f"  MAE then MFE (dip then recover): {recovery.get('mae_then_recovery', 0)} ({recovery.get('mae_first_pct', 0):.1f}%)")
        print(f"  MFE then MAE (peak then decline): {recovery.get('mfe_then_decline', 0)}")
        print(f"\nInsight: {recovery.get('insight', 'N/A')}")
    else:
        print(recovery.get("message", "No data available"))

    print("\n" + "=" * 70)


def export_to_csv(
    winner_mae: Dict[str, Any],
    mae_mfe_ratio: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export pain-gain ratio data to CSV."""

    if not mae_mfe_ratio:
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pattern', 'trades_with_data', 'avg_mae_abs', 'avg_mfe',
            'pain_gain_ratio', 'interpretation'
        ])
        writer.writeheader()
        writer.writerows(mae_mfe_ratio)

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

    # Data coverage check
    with_excursion = get_trades_with_excursion_data(trades)
    winners = [t for t in trades if t.is_winner]
    winners_with_excursion = [t for t in winners if t.excursion.mfe_pnl != 0 or t.excursion.mae_pnl != 0]

    data_coverage = {
        "total": len(trades),
        "with_excursion": len(with_excursion),
        "coverage_pct": round(len(with_excursion) / len(trades) * 100, 1) if trades else 0,
        "winners_with_excursion": len(winners_with_excursion),
    }

    print(f"\nData coverage: {data_coverage['with_excursion']}/{data_coverage['total']} trades have MFE/MAE data")

    # Run analyses
    print("Running MAE recovery analyses...")

    winner_mae = analyze_winner_mae_distribution(winners_with_excursion)
    mae_mfe_ratio = analyze_mae_mfe_ratio(with_excursion)
    time_to_mae = analyze_time_to_mae(winners_with_excursion)
    recovery = analyze_recovery_pattern(winners_with_excursion)

    # Print report
    print_report(winner_mae, mae_mfe_ratio, time_to_mae, recovery, data_coverage)

    # Export CSV
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "mae_recovery_report.csv"
    export_to_csv(winner_mae, mae_mfe_ratio, csv_path)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
