#!/usr/bin/env python3
"""
Full Trade Analytics Report

Master script that runs all trade analytics analyses and generates a
comprehensive report with actionable insights.

Runs:
1. Pattern Efficiency Analysis
2. MAE Recovery Analysis
3. Optimal Stop Analysis

Generates combined markdown report with all insights.

Usage:
    python scripts/run_full_analytics.py

Output:
    - Console summary
    - Individual CSV reports in output/
    - Combined markdown report: output/analytics_report_YYYYMMDD.md

Session: EQUITY-98
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.trade_analytics import TradeStore, TradeAnalyticsEngine

# Import analysis functions from other scripts
from analyze_pattern_efficiency import (
    analyze_pattern_performance,
    analyze_pattern_exit_efficiency,
    create_pattern_timeframe_matrix,
    analyze_losers_went_green,
)
from analyze_mae_recovery import (
    get_trades_with_excursion_data,
    analyze_winner_mae_distribution,
    analyze_mae_mfe_ratio,
    analyze_time_to_mae,
    analyze_recovery_pattern,
)
from analyze_optimal_stops import (
    analyze_mae_by_exit_reason,
    analyze_stop_accuracy,
    simulate_stop_distances,
    recommend_stops_by_pattern,
)


def generate_markdown_report(
    store: TradeStore,
    analytics: TradeAnalyticsEngine,
    output_path: Path,
) -> None:
    """Generate comprehensive markdown report."""

    trades = store.get_closed_trades()
    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]
    with_excursion = get_trades_with_excursion_data(trades)

    # Run all analyses
    pattern_perf = analyze_pattern_performance(analytics)
    exit_efficiency = analyze_pattern_exit_efficiency(store, analytics)
    matrix = create_pattern_timeframe_matrix(store)
    losers_green = analyze_losers_went_green(store)

    winners_with_exc = [t for t in winners if t.excursion.mfe_pnl != 0 or t.excursion.mae_pnl != 0]
    winner_mae = analyze_winner_mae_distribution(winners_with_exc)
    mae_mfe = analyze_mae_mfe_ratio(with_excursion)
    time_mae = analyze_time_to_mae(winners_with_exc)
    recovery = analyze_recovery_pattern(winners_with_exc)

    mae_by_exit = analyze_mae_by_exit_reason(trades)
    stop_acc = analyze_stop_accuracy(trades)
    stop_sim = simulate_stop_distances(with_excursion)
    pattern_stops = recommend_stops_by_pattern(trades)

    # Build markdown
    md = []
    md.append(f"# ATLAS Trade Analytics Report")
    md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"\n**Trades Analyzed:** {len(trades)}")
    md.append(f"\n**Data Period:** {min(t.entry_time for t in trades).strftime('%Y-%m-%d')} to {max(t.entry_time for t in trades).strftime('%Y-%m-%d')}")

    # Executive Summary
    md.append("\n\n## Executive Summary")
    md.append("\n### Key Metrics")
    md.append(f"\n- **Total Trades:** {len(trades)}")
    md.append(f"- **Win Rate:** {len(winners)/len(trades)*100:.1f}%")
    md.append(f"- **Total P&L:** ${sum(t.pnl for t in trades):,.2f}")
    md.append(f"- **Average P&L:** ${sum(t.pnl for t in trades)/len(trades):,.2f}")
    md.append(f"- **Trades with MFE/MAE Data:** {len(with_excursion)} ({len(with_excursion)/len(trades)*100:.0f}%)")

    # Pattern Performance Table
    md.append("\n\n## 1. Pattern Performance")
    md.append("\n| Pattern | Trades | Wins | Win Rate | Total P&L | Avg P&L | PF |")
    md.append("|---------|--------|------|----------|-----------|---------|-----|")
    for p in pattern_perf:
        pf = p['profit_factor'] if p['profit_factor'] != "N/A" else "-"
        md.append(f"| {p['pattern']} | {p['trades']} | {p['wins']} | {p['win_rate']}% | ${p['total_pnl']:,.2f} | ${p['avg_pnl']:,.2f} | {pf} |")

    # Best/Worst patterns
    by_wr = sorted([p for p in pattern_perf if p['trades'] >= 3], key=lambda x: x['win_rate'], reverse=True)
    if by_wr:
        md.append(f"\n**Best Pattern (min 3 trades):** {by_wr[0]['pattern']} ({by_wr[0]['win_rate']}% WR)")
        md.append(f"\n**Worst Pattern (min 3 trades):** {by_wr[-1]['pattern']} ({by_wr[-1]['win_rate']}% WR)")

    # Pattern-Timeframe Matrix
    md.append("\n\n## 2. Pattern x Timeframe Matrix")
    all_tfs = set()
    for pattern_data in matrix.values():
        all_tfs.update(pattern_data.keys())
    all_tfs = sorted(all_tfs, key=lambda x: {'1H': 0, '4H': 1, '1D': 2, '1W': 3, '1M': 4}.get(x, 99))

    header = "| Pattern |"
    for tf in all_tfs:
        header += f" {tf} |"
    md.append("\n" + header)
    md.append("|---------|" + "------|" * len(all_tfs))

    for pattern, tf_data in sorted(matrix.items()):
        row = f"| {pattern} |"
        for tf in all_tfs:
            if tf in tf_data:
                d = tf_data[tf]
                row += f" {d['trades']}t/{d['win_rate']:.0f}% |"
            else:
                row += " - |"
        md.append(row)

    # Exit Efficiency
    md.append("\n\n## 3. Exit Efficiency Analysis")

    has_eff = [e for e in exit_efficiency if e['trades_with_mfe'] > 0]
    if has_eff:
        md.append("\n| Pattern | Trades | w/MFE | Efficiency | Captured% |")
        md.append("|---------|--------|-------|------------|-----------|")
        for e in has_eff:
            md.append(f"| {e['pattern']} | {e['total_trades']} | {e['trades_with_mfe']} | {e['avg_exit_efficiency']}% | {e['avg_profit_captured_pct']}% |")
    else:
        md.append("\n*Awaiting MFE data from tracked trades*")

    # MAE Recovery
    md.append("\n\n## 4. MAE Recovery Analysis")

    if winner_mae.get("trades_analyzed", 0) > 0:
        stats = winner_mae.get("mae_pnl_stats", {})
        md.append(f"\n**Winners Analyzed:** {winner_mae['trades_analyzed']}")
        md.append(f"\n**Average MAE (Drawdown):** ${stats.get('mean', 0):.2f}")
        md.append(f"\n**Median MAE:** ${stats.get('median', 0):.2f}")

        if mae_mfe:
            md.append("\n\n### Pain-to-Gain Ratio by Pattern")
            md.append("\n| Pattern | Trades | Avg MAE | Avg MFE | Ratio | Interpretation |")
            md.append("|---------|--------|---------|---------|-------|----------------|")
            for r in mae_mfe[:5]:  # Top 5
                if r['trades_with_data'] > 0:
                    md.append(f"| {r['pattern']} | {r['trades_with_data']} | ${r['avg_mae_abs']:.2f} | ${r['avg_mfe']:.2f} | {r['pain_gain_ratio']:.2f} | {r['interpretation']} |")
    else:
        md.append("\n*Awaiting MAE data from tracked trades*")

    # Stop Analysis
    md.append("\n\n## 5. Stop Placement Analysis")

    if stop_acc.get("trades_with_stop_data", 0) > 0:
        md.append(f"\n**Trades with Stop Data:** {stop_acc['trades_with_stop_data']}")
        md.append(f"\n- Stop Hit: {stop_acc['stop_hit']} ({stop_acc['stop_hit_pct']}%)")
        md.append(f"- Stop Never Reached: {stop_acc['stop_never_reached']} ({stop_acc['stop_never_reached_pct']}%)")

    if stop_sim and stop_sim[0].get("trades_analyzed", 0) > 0:
        md.append("\n\n### Stop Distance Simulation")
        md.append("\n| Stop % | Stopped Out | Win Rate |")
        md.append("|--------|-------------|----------|")
        for sim in stop_sim:
            md.append(f"| {sim['stop_pct']}% | {sim['stopped_out']} ({sim['stopped_out_pct']}%) | {sim['effective_win_rate']}% |")

    # Actionable Insights
    md.append("\n\n## 6. Actionable Insights")

    insights = []

    # Generate insights from analytics engine
    full_report = analytics.generate_report(trades)
    if full_report.get("insights"):
        insights.extend(full_report["insights"])

    # Pattern-specific insights
    if by_wr:
        worst = by_wr[-1]
        if worst['win_rate'] < 35:
            insights.append(f"Consider reducing exposure to {worst['pattern']} pattern ({worst['win_rate']}% win rate)")

    # Losers went green insight
    high_green = [l for l in losers_green if l['went_green_pct'] > 50 and l['losers_with_mfe_data'] > 0]
    if high_green:
        patterns = ", ".join(l['pattern'] for l in high_green)
        insights.append(f"Patterns {patterns} have >50% losers that went green first - consider earlier profit-taking")

    if insights:
        for i, insight in enumerate(insights, 1):
            # Remove emojis for ASCII compatibility (Windows console)
            import re
            clean_insight = re.sub(r'[^\x00-\x7F]+', '', insight)
            clean_insight = clean_insight.strip().replace("  ", " ")
            md.append(f"\n{i}. {clean_insight}")
    else:
        md.append("\n*More insights will be generated as trade data accumulates*")

    # Data Notes
    md.append("\n\n## Notes")
    md.append(f"\n- Trades with excursion data: {len(with_excursion)}/{len(trades)}")
    md.append(f"- Historical trades migrated without MFE/MAE will not contribute to excursion analysis")
    md.append(f"- New trades tracked in real-time will populate excursion metrics")

    # Write file with UTF-8 encoding
    output_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown report saved to: {output_path}")


def main():
    """Main entry point."""

    print("\n" + "=" * 70)
    print("ATLAS FULL TRADE ANALYTICS")
    print("=" * 70)

    # Load trade store
    store_path = project_root / "core" / "trade_analytics" / "data" / "equity_trades.json"

    if not store_path.exists():
        print(f"ERROR: Trade store not found at {store_path}")
        print("Run the trade migration script first.")
        sys.exit(1)

    print(f"\nLoading trades from: {store_path}")
    store = TradeStore(store_path)
    analytics = TradeAnalyticsEngine(store)

    trades = store.get_closed_trades()
    print(f"Found {len(trades)} closed trades")

    if not trades:
        print("No closed trades available for analysis.")
        sys.exit(0)

    # Create output directory with date
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Run individual scripts (they print their own reports)
    print("\n" + "-" * 70)
    print("Running Pattern Efficiency Analysis...")
    print("-" * 70)
    from analyze_pattern_efficiency import main as pattern_main
    pattern_main()

    print("\n" + "-" * 70)
    print("Running MAE Recovery Analysis...")
    print("-" * 70)
    from analyze_mae_recovery import main as mae_main
    mae_main()

    print("\n" + "-" * 70)
    print("Running Optimal Stop Analysis...")
    print("-" * 70)
    from analyze_optimal_stops import main as stops_main
    stops_main()

    # Generate combined markdown report
    print("\n" + "-" * 70)
    print("Generating Combined Report...")
    print("-" * 70)

    report_date = datetime.now().strftime("%Y%m%d")
    report_path = output_dir / f"analytics_report_{report_date}.md"
    generate_markdown_report(store, analytics, report_path)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - output/pattern_efficiency_report.csv")
    print(f"  - output/mae_recovery_report.csv")
    print(f"  - output/optimal_stops_report.csv")
    print(f"  - output/analytics_report_{report_date}.md")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
