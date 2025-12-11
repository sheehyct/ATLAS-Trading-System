"""Generate Master Findings Document from all validation data.

Session 83K-37: Enhanced with regime-conditioned statistics and ML Gate 0 review.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_trades():
    """Load all trade CSVs from validation results."""
    trades_dir = Path('validation_results/session_83k/trades')
    all_trades = []

    for csv_file in trades_dir.glob('*.csv'):
        # Parse filename: pattern_timeframe_symbol_trades.csv
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            pattern = parts[0]
            timeframe = parts[1]
            symbol = parts[2]

            df = pd.read_csv(csv_file)
            df['pattern'] = pattern
            df['timeframe'] = timeframe
            df['symbol'] = symbol
            all_trades.append(df)

    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    return pd.DataFrame()


def load_atlas_regime_data():
    """Load pre-computed ATLAS regime data."""
    regime_path = Path('visualization/tradingview_exports/regimes_combined_tradingview.csv')
    if regime_path.exists():
        df = pd.read_csv(regime_path, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.set_index('date')
        return df
    return None


def load_vix_data():
    """Load VIX data via yfinance."""
    try:
        import yfinance as yf
        vix = yf.download('^VIX', start='2020-01-01', end='2025-12-31', progress=False)
        if not vix.empty:
            # Handle both single and multi-level column index
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix_close = vix['Close'].copy()
            vix_close.index = pd.to_datetime(vix_close.index).date
            return vix_close
    except Exception as e:
        print(f"Warning: Could not load VIX data: {e}")
    return None


def add_regime_and_vix_data(trades, atlas_regimes, vix_data):
    """Add regime and VIX classifications to trade data."""
    # Parse entry_date - handle mixed timezones
    trades['entry_date_parsed'] = pd.to_datetime(trades['entry_date'], utc=True).dt.date

    # Add ATLAS regime (use SPY regime as market proxy)
    if atlas_regimes is not None:
        trades['atlas_regime'] = trades['entry_date_parsed'].map(
            lambda d: atlas_regimes.loc[d, 'SPY_regime_name'] if d in atlas_regimes.index else 'UNKNOWN'
        )
    else:
        trades['atlas_regime'] = 'UNKNOWN'

    # Add VIX data and create classifications
    if vix_data is not None:
        trades['vix_level'] = trades['entry_date_parsed'].map(
            lambda d: vix_data.get(d, np.nan)
        )

        # VIX bucket: <15, 15-20, 20-30, 30-40, >40
        def vix_bucket(vix):
            if pd.isna(vix):
                return 'UNKNOWN'
            if vix < 15:
                return '<15'
            elif vix < 20:
                return '15-20'
            elif vix < 30:
                return '20-30'
            elif vix < 40:
                return '30-40'
            else:
                return '>40'

        trades['vix_bucket'] = trades['vix_level'].apply(vix_bucket)

        # Simple VIX-based regime: <15 = BULL, 15-25 = NEUTRAL, 25-35 = BEAR, 35+ = CRASH
        def vix_regime(vix):
            if pd.isna(vix):
                return 'UNKNOWN'
            if vix < 15:
                return 'VIX_BULL'
            elif vix < 25:
                return 'VIX_NEUTRAL'
            elif vix < 35:
                return 'VIX_BEAR'
            else:
                return 'VIX_CRASH'

        trades['vix_regime'] = trades['vix_level'].apply(vix_regime)
    else:
        trades['vix_level'] = np.nan
        trades['vix_bucket'] = 'UNKNOWN'
        trades['vix_regime'] = 'UNKNOWN'

    return trades


def calculate_stats(df, group_cols=None):
    """Calculate key statistics for a dataframe."""
    if df.empty:
        return {}

    stats = {
        'trades': len(df),
        'total_pnl': df['pnl'].sum(),
        'avg_pnl': df['pnl'].mean(),
        'win_rate': (df['pnl'] > 0).mean() * 100,
        'median_pnl': df['pnl'].median(),
    }

    if 'magnitude_pct' in df.columns:
        stats['avg_magnitude'] = df['magnitude_pct'].mean()
        stats['median_magnitude'] = df['magnitude_pct'].median()

    return stats


def calculate_sharpe(pnl_series, periods_per_year=252):
    """Calculate annualized Sharpe ratio from P&L series."""
    if len(pnl_series) < 2:
        return 0.0
    mean_pnl = pnl_series.mean()
    std_pnl = pnl_series.std()
    if std_pnl == 0:
        return 0.0
    return (mean_pnl / std_pnl) * np.sqrt(periods_per_year)


def calculate_gate0_metrics(df, pattern_name):
    """Calculate ML Gate 0 metrics for a pattern family."""
    if df.empty:
        return None

    trades = len(df)
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    win_rate = (df['pnl'] > 0).mean() * 100

    # Win/Loss ratio
    winners = df[df['pnl'] > 0]['pnl']
    losers = df[df['pnl'] < 0]['pnl']
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Sharpe (simplified - using daily P&L proxy)
    sharpe = calculate_sharpe(df['pnl'])

    # Symbol diversity
    symbols = df['symbol'].nunique()

    # Regime diversity
    if 'atlas_regime' in df.columns:
        regimes = df[df['atlas_regime'] != 'UNKNOWN']['atlas_regime'].nunique()
    else:
        regimes = 0

    # Gate 0 checks
    check_trades = trades >= 100
    check_sharpe = sharpe > 0
    check_win_rate = win_rate > 40 or win_loss_ratio > 2.0
    check_pnl = total_pnl > 0
    check_symbols = symbols >= 3
    check_regimes = regimes >= 2

    gate0_passed = all([check_trades, check_sharpe, check_win_rate, check_pnl, check_symbols, check_regimes])

    return {
        'pattern': pattern_name,
        'trades': trades,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'sharpe': sharpe,
        'symbols': symbols,
        'regimes': regimes,
        'check_trades': check_trades,
        'check_sharpe': check_sharpe,
        'check_win_rate': check_win_rate,
        'check_pnl': check_pnl,
        'check_symbols': check_symbols,
        'check_regimes': check_regimes,
        'gate0_passed': gate0_passed
    }


def main():
    print("Loading all trade data...")
    trades = load_all_trades()

    if trades.empty:
        print("No trade data found!")
        return

    print(f"Loaded {len(trades)} trades")

    # Calculate magnitude if not present
    if 'magnitude_pct' not in trades.columns and 'entry_price' in trades.columns and 'target_price' in trades.columns:
        trades['magnitude_pct'] = abs(trades['target_price'] - trades['entry_price']) / trades['entry_price'] * 100

    # Load regime and VIX data
    print("Loading ATLAS regime data...")
    atlas_regimes = load_atlas_regime_data()
    print(f"ATLAS regime data: {'Loaded' if atlas_regimes is not None else 'Not found'}")

    print("Loading VIX data...")
    vix_data = load_vix_data()
    print(f"VIX data: {'Loaded' if vix_data is not None else 'Not found'}")

    # Add regime and VIX classifications
    print("Adding regime and VIX classifications to trades...")
    trades = add_regime_and_vix_data(trades, atlas_regimes, vix_data)

    # Generate master report
    output = []
    output.append("# STRAT Options Validation - Master Findings Report")
    output.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"**Session:** 83K-37 (Regime Statistics + ML Gate 0 Review)")
    output.append(f"**Total Trades Analyzed:** {len(trades):,}")
    output.append(f"**Symbols:** {', '.join(sorted(trades['symbol'].unique()))}")
    output.append(f"**Timeframes:** {', '.join(sorted(trades['timeframe'].unique()))}")
    output.append(f"**Patterns:** {', '.join(sorted(trades['pattern'].unique()))}")

    # Overall Summary
    output.append("\n---\n")
    output.append("## 1. Overall Summary")
    overall = calculate_stats(trades)
    output.append(f"\n- **Total Trades:** {overall['trades']:,}")
    output.append(f"- **Total P&L:** ${overall['total_pnl']:,.0f}")
    output.append(f"- **Average P&L:** ${overall['avg_pnl']:,.0f}")
    output.append(f"- **Win Rate:** {overall['win_rate']:.1f}%")
    if 'avg_magnitude' in overall:
        output.append(f"- **Avg Magnitude:** {overall['avg_magnitude']:.2f}%")

    # By Timeframe
    output.append("\n---\n")
    output.append("## 2. Performance by Timeframe")
    output.append("\n| Timeframe | Trades | Total P&L | Avg P&L | Win Rate | Avg Magnitude |")
    output.append("|-----------|--------|-----------|---------|----------|---------------|")

    for tf in ['1H', '1D', '1W', '1M']:
        tf_trades = trades[trades['timeframe'] == tf]
        if not tf_trades.empty:
            stats = calculate_stats(tf_trades)
            mag = stats.get('avg_magnitude', 0)
            output.append(f"| {tf} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% | {mag:.2f}% |")

    # By Pattern
    output.append("\n---\n")
    output.append("## 3. Performance by Pattern")
    output.append("\n| Pattern | Trades | Total P&L | Avg P&L | Win Rate | Avg Magnitude |")
    output.append("|---------|--------|-----------|---------|----------|---------------|")

    for pattern in ['3-2', '3-2-2', '3-1-2', '2-2', '2-1-2']:
        p_trades = trades[trades['pattern'] == pattern]
        if not p_trades.empty:
            stats = calculate_stats(p_trades)
            mag = stats.get('avg_magnitude', 0)
            output.append(f"| {pattern} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% | {mag:.2f}% |")

    # By Symbol
    output.append("\n---\n")
    output.append("## 4. Performance by Symbol")
    output.append("\n| Symbol | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|--------|--------|-----------|---------|----------|")

    for symbol in sorted(trades['symbol'].unique()):
        s_trades = trades[trades['symbol'] == symbol]
        if not s_trades.empty:
            stats = calculate_stats(s_trades)
            output.append(f"| {symbol} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: By ATLAS Regime
    output.append("\n---\n")
    output.append("## 5. Performance by ATLAS Regime")
    output.append("\n| Regime | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|--------|--------|-----------|---------|----------|")

    for regime in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL']:
        r_trades = trades[trades['atlas_regime'] == regime]
        if not r_trades.empty:
            stats = calculate_stats(r_trades)
            output.append(f"| {regime} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # Unknown regime trades
    unknown = trades[trades['atlas_regime'] == 'UNKNOWN']
    if not unknown.empty:
        output.append(f"\n*Note: {len(unknown)} trades have no regime data (dates not in regime file)*")

    # NEW SECTION: By VIX Regime (Simple)
    output.append("\n---\n")
    output.append("## 6. Performance by VIX Regime (Simple Classification)")
    output.append("\nVIX-based regime: <15 = BULL, 15-25 = NEUTRAL, 25-35 = BEAR, 35+ = CRASH")
    output.append("\n| VIX Regime | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|------------|--------|-----------|---------|----------|")

    for regime in ['VIX_BULL', 'VIX_NEUTRAL', 'VIX_BEAR', 'VIX_CRASH']:
        r_trades = trades[trades['vix_regime'] == regime]
        if not r_trades.empty:
            stats = calculate_stats(r_trades)
            output.append(f"| {regime} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: By VIX Bucket
    output.append("\n---\n")
    output.append("## 7. Performance by VIX Bucket")
    output.append("\n| VIX Level | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|-----------|--------|-----------|---------|----------|")

    for bucket in ['<15', '15-20', '20-30', '30-40', '>40']:
        b_trades = trades[trades['vix_bucket'] == bucket]
        if not b_trades.empty:
            stats = calculate_stats(b_trades)
            output.append(f"| {bucket} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: By Exit Type
    output.append("\n---\n")
    output.append("## 8. Performance by Exit Type")
    output.append("\n| Exit Type | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|-----------|--------|-----------|---------|----------|")

    if 'exit_type' in trades.columns:
        for exit_type in ['TARGET', 'STOP', 'TIME_EXIT']:
            e_trades = trades[trades['exit_type'] == exit_type]
            if not e_trades.empty:
                stats = calculate_stats(e_trades)
                output.append(f"| {exit_type} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: By Direction (CALL vs PUT)
    output.append("\n---\n")
    output.append("## 9. Performance by Direction")
    output.append("\n| Direction | Trades | Total P&L | Avg P&L | Win Rate |")
    output.append("|-----------|--------|-----------|---------|----------|")

    if 'direction' in trades.columns:
        for direction, label in [(1, 'CALL (Bullish)'), (-1, 'PUT (Bearish)')]:
            d_trades = trades[trades['direction'] == direction]
            if not d_trades.empty:
                stats = calculate_stats(d_trades)
                output.append(f"| {label} | {stats['trades']:,} | ${stats['total_pnl']:,.0f} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: Pattern x Regime Matrix
    output.append("\n---\n")
    output.append("## 10. Pattern x ATLAS Regime Performance Matrix")
    output.append("\n### Average P&L by Pattern and Regime")
    output.append("\n| Pattern | CRASH | TREND_BEAR | TREND_NEUTRAL | TREND_BULL |")
    output.append("|---------|-------|------------|---------------|------------|")

    for pattern in ['3-2', '3-2-2', '3-1-2', '2-2', '2-1-2']:
        row = [f"| {pattern}"]
        for regime in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL']:
            subset = trades[(trades['pattern'] == pattern) & (trades['atlas_regime'] == regime)]
            if not subset.empty:
                avg = subset['pnl'].mean()
                count = len(subset)
                row.append(f"${avg:,.0f} ({count})")
            else:
                row.append("-")
        output.append(" | ".join(row) + " |")

    # Pattern x Timeframe Matrix
    output.append("\n---\n")
    output.append("## 11. Pattern x Timeframe Performance Matrix")
    output.append("\n### Average P&L by Pattern and Timeframe")
    output.append("\n| Pattern | 1H | 1D | 1W | 1M |")
    output.append("|---------|-----|-----|-----|-----|")

    for pattern in ['3-2', '3-2-2', '3-1-2', '2-2', '2-1-2']:
        row = [f"| {pattern}"]
        for tf in ['1H', '1D', '1W', '1M']:
            subset = trades[(trades['pattern'] == pattern) & (trades['timeframe'] == tf)]
            if not subset.empty:
                avg = subset['pnl'].mean()
                row.append(f"${avg:,.0f}")
            else:
                row.append("-")
        output.append(" | ".join(row) + " |")

    # Trade count matrix
    output.append("\n### Trade Count by Pattern and Timeframe")
    output.append("\n| Pattern | 1H | 1D | 1W | 1M | Total |")
    output.append("|---------|-----|-----|-----|-----|-------|")

    for pattern in ['3-2', '3-2-2', '3-1-2', '2-2', '2-1-2']:
        row = [f"| {pattern}"]
        total = 0
        for tf in ['1H', '1D', '1W', '1M']:
            subset = trades[(trades['pattern'] == pattern) & (trades['timeframe'] == tf)]
            count = len(subset)
            total += count
            row.append(str(count) if count > 0 else "-")
        row.append(str(total))
        output.append(" | ".join(row) + " |")

    # Magnitude Analysis
    if 'magnitude_pct' in trades.columns:
        output.append("\n---\n")
        output.append("## 12. Magnitude Analysis")
        output.append("\n### P&L by Magnitude Bucket")
        output.append("\n| Magnitude | Trades | Avg P&L | Win Rate |")
        output.append("|-----------|--------|---------|----------|")

        buckets = [
            (0, 0.3, '<0.3%'),
            (0.3, 0.5, '0.3-0.5%'),
            (0.5, 1.0, '0.5-1.0%'),
            (1.0, 2.0, '1.0-2.0%'),
            (2.0, 5.0, '2.0-5.0%'),
            (5.0, float('inf'), '>5.0%'),
        ]

        for low, high, label in buckets:
            subset = trades[(trades['magnitude_pct'] >= low) & (trades['magnitude_pct'] < high)]
            if not subset.empty:
                stats = calculate_stats(subset)
                output.append(f"| {label} | {stats['trades']:,} | ${stats['avg_pnl']:,.0f} | {stats['win_rate']:.1f}% |")

    # NEW SECTION: ML Gate 0 Review
    output.append("\n---\n")
    output.append("## 13. ML Gate 0 Review (Per Pattern Family)")
    output.append("\n**Gate 0 Requirements (per ML_IMPLEMENTATION_GUIDE_STRAT.md):**")
    output.append("- 100+ trades")
    output.append("- Positive Sharpe ratio")
    output.append("- Win rate > 40% OR avg_win > 2x avg_loss")
    output.append("- Positive total P&L")
    output.append("- Tested across 3+ symbols")
    output.append("- Tested across 2+ market regimes")
    output.append("\n### Gate 0 Status by Pattern Family\n")

    gate0_results = []
    for pattern in ['3-2', '3-2-2', '3-1-2', '2-2', '2-1-2']:
        p_trades = trades[trades['pattern'] == pattern]
        if not p_trades.empty:
            metrics = calculate_gate0_metrics(p_trades, pattern)
            if metrics:
                gate0_results.append(metrics)

    for m in gate0_results:
        check_icon = lambda x: 'PASS' if x else 'FAIL'
        output.append(f"#### Pattern Family: {m['pattern']}")
        output.append("")
        output.append("| Metric | Value | Gate 0 Check |")
        output.append("|--------|-------|--------------|")
        output.append(f"| Trades | {m['trades']} | {check_icon(m['check_trades'])} (need 100+) |")
        output.append(f"| Sharpe | {m['sharpe']:.2f} | {check_icon(m['check_sharpe'])} (need > 0) |")
        output.append(f"| Win Rate | {m['win_rate']:.1f}% | {check_icon(m['check_win_rate'])} (need > 40% or W/L > 2) |")
        output.append(f"| Win/Loss Ratio | {m['win_loss_ratio']:.2f} | (Avg Win: ${m['avg_win']:,.0f}, Avg Loss: ${m['avg_loss']:,.0f}) |")
        output.append(f"| Total P&L | ${m['total_pnl']:,.0f} | {check_icon(m['check_pnl'])} (need > 0) |")
        output.append(f"| Symbols | {m['symbols']} | {check_icon(m['check_symbols'])} (need 3+) |")
        output.append(f"| Regimes | {m['regimes']} | {check_icon(m['check_regimes'])} (need 2+) |")
        output.append("")
        gate_status = "**GATE 0: PASSED**" if m['gate0_passed'] else "**GATE 0: FAILED**"
        output.append(f"{gate_status}")
        if m['gate0_passed']:
            output.append("-> Eligible for ML optimization (delta, DTE, position sizing)")
        output.append("")

    # Summary table
    output.append("\n### Gate 0 Summary\n")
    output.append("| Pattern | Trades | Sharpe | Win Rate | Total P&L | Symbols | Regimes | GATE 0 |")
    output.append("|---------|--------|--------|----------|-----------|---------|---------|--------|")
    for m in gate0_results:
        status = "PASS" if m['gate0_passed'] else "FAIL"
        output.append(f"| {m['pattern']} | {m['trades']} | {m['sharpe']:.2f} | {m['win_rate']:.1f}% | ${m['total_pnl']:,.0f} | {m['symbols']} | {m['regimes']} | {status} |")

    # Key Findings
    output.append("\n---\n")
    output.append("## 14. Key Findings")
    output.append("\n### Pattern Recommendations by Timeframe")
    output.append("\n| Timeframe | Best Pattern | Avg P&L | Recommendation |")
    output.append("|-----------|--------------|---------|----------------|")

    for tf in ['1H', '1D', '1W', '1M']:
        tf_trades = trades[trades['timeframe'] == tf]
        if not tf_trades.empty:
            best_pattern = None
            best_avg = float('-inf')
            for pattern in tf_trades['pattern'].unique():
                subset = tf_trades[tf_trades['pattern'] == pattern]
                if len(subset) >= 5:  # Minimum trades for significance
                    avg = subset['pnl'].mean()
                    if avg > best_avg:
                        best_avg = avg
                        best_pattern = pattern

            if best_pattern:
                rec = "PRIMARY" if best_avg > 100 else "SECONDARY" if best_avg > 0 else "AVOID"
                output.append(f"| {tf} | {best_pattern} | ${best_avg:,.0f} | {rec} |")

    # Add hourly-specific findings from Session 83K-35
    output.append("\n---\n")
    output.append("## 15. Hourly Pattern Insights (Session 83K-35)")
    output.append("\nNote: Hourly requires market-open-aligned bars (09:30, 10:30, 11:30), NOT clock-aligned.")
    output.append("\n### Why 3-bar Patterns Outperform 2-bar on Hourly")
    output.append("\n| Pattern | Avg Magnitude | TIME_EXIT Avg | Root Cause |")
    output.append("|---------|--------------|---------------|------------|")
    output.append("| 3-2 | 1.06% | +$68 | 60% larger magnitude, profitable even on forced exit |")
    output.append("| 2-2 | 0.65% | -$331 | Smaller magnitude, loses on TIME_EXIT |")
    output.append("\n### Hourly Recommendations")
    output.append("\n- **3-2, 3-2-2, 3-1-2:** PROFITABLE - Use for hourly options")
    output.append("- **2-2:** BREAKEVEN - Only trade when magnitude >1.0%")
    output.append("- **2-1-2:** SKIP - Too sparse on hourly")

    # Write output
    report_path = Path('docs/MASTER_FINDINGS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nReport generated: {report_path}")
    print('\n'.join(output))


if __name__ == '__main__':
    main()
