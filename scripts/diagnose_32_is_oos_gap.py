#!/usr/bin/env python
"""
Session 83K-61: Diagnostic Script for 3-2 Pattern IS vs OOS Gap

Investigates why 3-2 pattern shows $76k IS P&L but fails OOS validation (Sharpe -6.04).

Usage:
    uv run python scripts/diagnose_32_is_oos_gap.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_trade_data(pattern: str = "3-2", timeframe: str = "1D", symbol: str = "SPY") -> pd.DataFrame:
    """Load trade data from validation results."""
    trades_dir = Path("validation_results/session_83k/trades")
    file_path = trades_dir / f"{pattern}_{timeframe}_{symbol}_trades.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Trade file not found: {file_path}")

    df = pd.read_csv(file_path)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    return df


def split_is_oos(df: pd.DataFrame, is_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split trades into IS and OOS periods based on date (70/30 holdout)."""
    df_sorted = df.sort_values("entry_date").reset_index(drop=True)
    n_total = len(df_sorted)
    n_is = int(n_total * is_ratio)

    is_df = df_sorted.iloc[:n_is].copy()
    oos_df = df_sorted.iloc[n_is:].copy()

    return is_df, oos_df


def calculate_period_stats(df: pd.DataFrame, period_name: str) -> dict:
    """Calculate comprehensive statistics for a period."""
    if len(df) == 0:
        return {"period": period_name, "trades": 0}

    # Basic metrics
    total_pnl = df["pnl"].sum()
    mean_pnl = df["pnl"].mean()
    std_pnl = df["pnl"].std()

    # Win/Loss metrics
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    # R:R calculation (magnitude / risk distance)
    df_valid = df[df["magnitude_pct"].notna() & (df["magnitude_pct"] > 0)]
    if len(df_valid) > 0:
        # Calculate risk as percentage distance from entry to stop
        risk_pct = abs(df_valid["entry_price"] - df_valid["stop_price"]) / df_valid["entry_price"] * 100
        rr_ratio = df_valid["magnitude_pct"] / risk_pct
        mean_rr = rr_ratio.mean()
        median_rr = rr_ratio.median()
    else:
        mean_rr = np.nan
        median_rr = np.nan

    # Exit type distribution
    exit_dist = df["exit_type"].value_counts().to_dict()

    # Date range
    date_range = f"{df['entry_date'].min().strftime('%Y-%m-%d')} to {df['entry_date'].max().strftime('%Y-%m-%d')}"

    # Largest trades
    largest_win = df["pnl"].max() if len(wins) > 0 else 0
    largest_loss = df["pnl"].min() if len(losses) > 0 else 0

    # Direction split
    bullish = df[df["direction"] == 1]
    bearish = df[df["direction"] == -1]

    return {
        "period": period_name,
        "trades": len(df),
        "date_range": date_range,
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "win_rate": win_rate,
        "wins": len(wins),
        "losses": len(losses),
        "mean_rr": mean_rr,
        "median_rr": median_rr,
        "avg_magnitude_pct": df["magnitude_pct"].mean(),
        "avg_days_held": df["days_held"].mean(),
        "exit_target": exit_dist.get("TARGET", 0),
        "exit_stop": exit_dist.get("STOP", 0),
        "exit_time": exit_dist.get("TIME_EXIT", 0),
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "bullish_trades": len(bullish),
        "bearish_trades": len(bearish),
        "bullish_pnl": bullish["pnl"].sum() if len(bullish) > 0 else 0,
        "bearish_pnl": bearish["pnl"].sum() if len(bearish) > 0 else 0,
    }


def calculate_sharpe_estimate(df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Estimate Sharpe ratio from trade returns."""
    if len(df) < 2:
        return np.nan

    returns = df["pnl_pct"].values / 100  # Convert percentage to decimal
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    if excess_returns.std() == 0:
        return np.nan

    # Annualized Sharpe (assuming daily trades, sqrt(252))
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe


def identify_outliers(df: pd.DataFrame, n_top: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify top winners and losers."""
    sorted_df = df.sort_values("pnl")
    top_losers = sorted_df.head(n_top)[["entry_date", "pnl", "pnl_pct", "exit_type", "direction", "magnitude_pct"]]
    top_winners = sorted_df.tail(n_top)[["entry_date", "pnl", "pnl_pct", "exit_type", "direction", "magnitude_pct"]]
    return top_losers, top_winners


def analyze_regime_shift(is_df: pd.DataFrame, oos_df: pd.DataFrame) -> dict:
    """Analyze potential regime shifts between IS and OOS."""
    analysis = {}

    # Average magnitude comparison
    is_mag = is_df["magnitude_pct"].mean()
    oos_mag = oos_df["magnitude_pct"].mean()
    analysis["magnitude_shift"] = (oos_mag - is_mag) / is_mag * 100 if is_mag > 0 else np.nan

    # Win rate comparison
    is_wr = (is_df["pnl"] > 0).mean() * 100
    oos_wr = (oos_df["pnl"] > 0).mean() * 100
    analysis["win_rate_shift"] = oos_wr - is_wr

    # Exit type shift
    is_target_pct = (is_df["exit_type"] == "TARGET").mean() * 100
    oos_target_pct = (oos_df["exit_type"] == "TARGET").mean() * 100
    analysis["target_exit_shift"] = oos_target_pct - is_target_pct

    # Mean P&L shift
    is_mean = is_df["pnl"].mean()
    oos_mean = oos_df["pnl"].mean()
    analysis["mean_pnl_shift"] = oos_mean - is_mean

    # Volatility shift (std of returns)
    is_vol = is_df["pnl_pct"].std()
    oos_vol = oos_df["pnl_pct"].std()
    analysis["volatility_shift"] = (oos_vol - is_vol) / is_vol * 100 if is_vol > 0 else np.nan

    return analysis


def print_report(is_stats: dict, oos_stats: dict, regime_analysis: dict, is_sharpe: float, oos_sharpe: float):
    """Print comprehensive diagnostic report."""
    print("=" * 80)
    print("3-2 PATTERN IS vs OOS DIAGNOSTIC REPORT")
    print("Session 83K-61")
    print("=" * 80)

    # Summary comparison table
    print("\n" + "-" * 40)
    print("SUMMARY COMPARISON")
    print("-" * 40)
    print(f"{'Metric':<25} {'IS':<20} {'OOS':<20}")
    print("-" * 65)
    print(f"{'Trades':<25} {is_stats['trades']:<20} {oos_stats['trades']:<20}")
    print(f"{'Date Range':<25} {is_stats['date_range'][:20]:<20} {oos_stats['date_range'][:20]:<20}")
    print(f"{'Total P&L':<25} ${is_stats['total_pnl']:,.2f}".ljust(46) + f"${oos_stats['total_pnl']:,.2f}")
    print(f"{'Mean P&L':<25} ${is_stats['mean_pnl']:,.2f}".ljust(46) + f"${oos_stats['mean_pnl']:,.2f}")
    print(f"{'Std P&L':<25} ${is_stats['std_pnl']:,.2f}".ljust(46) + f"${oos_stats['std_pnl']:,.2f}")
    print(f"{'Win Rate':<25} {is_stats['win_rate']:.1f}%".ljust(46) + f"{oos_stats['win_rate']:.1f}%")
    print(f"{'Sharpe Estimate':<25} {is_sharpe:.2f}".ljust(46) + f"{oos_sharpe:.2f}")
    print(f"{'Mean R:R':<25} {is_stats['mean_rr']:.2f}".ljust(46) + f"{oos_stats['mean_rr']:.2f}")
    print(f"{'Avg Magnitude %':<25} {is_stats['avg_magnitude_pct']:.2f}%".ljust(46) + f"{oos_stats['avg_magnitude_pct']:.2f}%")
    print(f"{'Avg Days Held':<25} {is_stats['avg_days_held']:.1f}".ljust(46) + f"{oos_stats['avg_days_held']:.1f}")

    # Exit type distribution
    print("\n" + "-" * 40)
    print("EXIT TYPE DISTRIBUTION")
    print("-" * 40)
    print(f"{'Exit Type':<15} {'IS Count':<15} {'OOS Count':<15}")
    print(f"{'TARGET':<15} {is_stats['exit_target']:<15} {oos_stats['exit_target']:<15}")
    print(f"{'STOP':<15} {is_stats['exit_stop']:<15} {oos_stats['exit_stop']:<15}")
    print(f"{'TIME_EXIT':<15} {is_stats['exit_time']:<15} {oos_stats['exit_time']:<15}")

    # Direction analysis
    print("\n" + "-" * 40)
    print("DIRECTION ANALYSIS")
    print("-" * 40)
    print(f"{'Direction':<15} {'IS Trades':<15} {'IS P&L':<15} {'OOS Trades':<12} {'OOS P&L':<15}")
    print(f"{'Bullish':<15} {is_stats['bullish_trades']:<15} ${is_stats['bullish_pnl']:,.0f}".ljust(46) + f"{oos_stats['bullish_trades']:<12} ${oos_stats['bullish_pnl']:,.0f}")
    print(f"{'Bearish':<15} {is_stats['bearish_trades']:<15} ${is_stats['bearish_pnl']:,.0f}".ljust(46) + f"{oos_stats['bearish_trades']:<12} ${oos_stats['bearish_pnl']:,.0f}")

    # Extremes
    print("\n" + "-" * 40)
    print("EXTREME TRADES")
    print("-" * 40)
    print(f"{'Metric':<20} {'IS':<25} {'OOS':<25}")
    print(f"{'Largest Win':<20} ${is_stats['largest_win']:,.2f}".ljust(46) + f"${oos_stats['largest_win']:,.2f}")
    print(f"{'Largest Loss':<20} ${is_stats['largest_loss']:,.2f}".ljust(46) + f"${oos_stats['largest_loss']:,.2f}")

    # Regime shift analysis
    print("\n" + "-" * 40)
    print("REGIME SHIFT ANALYSIS (OOS vs IS)")
    print("-" * 40)
    print(f"Magnitude Shift:      {regime_analysis['magnitude_shift']:+.1f}% (OOS magnitude vs IS)")
    print(f"Win Rate Shift:       {regime_analysis['win_rate_shift']:+.1f} percentage points")
    print(f"Target Exit Shift:    {regime_analysis['target_exit_shift']:+.1f} percentage points")
    print(f"Mean P&L Shift:       ${regime_analysis['mean_pnl_shift']:+,.2f} per trade")
    print(f"Volatility Shift:     {regime_analysis['volatility_shift']:+.1f}% (return std dev)")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Hypothesis testing
    findings = []

    # 1. Sample size issue
    if oos_stats['trades'] < 30:
        findings.append(f"[CRITICAL] OOS has only {oos_stats['trades']} trades (< 30 for statistical significance)")

    # 2. Win rate degradation
    if regime_analysis['win_rate_shift'] < -10:
        findings.append(f"[CONCERN] Win rate dropped {abs(regime_analysis['win_rate_shift']):.1f} percentage points in OOS")

    # 3. Target hit degradation
    if regime_analysis['target_exit_shift'] < -10:
        findings.append(f"[CONCERN] Target exits dropped {abs(regime_analysis['target_exit_shift']):.1f} percentage points in OOS")

    # 4. Sharpe degradation
    if is_sharpe > 0 and oos_sharpe < 0:
        findings.append(f"[CRITICAL] Sharpe went from positive ({is_sharpe:.2f}) to negative ({oos_sharpe:.2f})")

    # 5. Volatility increase
    if regime_analysis['volatility_shift'] > 50:
        findings.append(f"[CONCERN] Return volatility increased {regime_analysis['volatility_shift']:.1f}% in OOS")

    # 6. Mean P&L sign flip
    if is_stats['mean_pnl'] > 0 and oos_stats['mean_pnl'] < 0:
        findings.append(f"[CRITICAL] Mean P&L flipped from +${is_stats['mean_pnl']:.2f} to ${oos_stats['mean_pnl']:.2f}")

    # 7. Large losers in OOS
    if oos_stats['largest_loss'] < -1000 and abs(oos_stats['largest_loss']) > abs(is_stats['largest_loss']):
        findings.append(f"[CONCERN] OOS has larger losses (${oos_stats['largest_loss']:,.2f}) than IS (${is_stats['largest_loss']:,.2f})")

    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")

    if not findings:
        print("No critical issues identified.")

    print("\n" + "=" * 80)


def analyze_multi_symbol(pattern: str = "3-2", timeframe: str = "1D", symbols: list = None) -> pd.DataFrame:
    """Analyze IS vs OOS across multiple symbols."""
    if symbols is None:
        symbols = ["SPY", "QQQ", "AAPL", "IWM", "DIA"]

    results = []
    for symbol in symbols:
        try:
            df = load_trade_data(pattern, timeframe, symbol)
            is_df, oos_df = split_is_oos(df, is_ratio=0.7)

            is_stats = calculate_period_stats(is_df, f"{symbol} IS")
            oos_stats = calculate_period_stats(oos_df, f"{symbol} OOS")
            is_sharpe = calculate_sharpe_estimate(is_df)
            oos_sharpe = calculate_sharpe_estimate(oos_df)

            results.append({
                "symbol": symbol,
                "total_trades": len(df),
                "is_trades": is_stats["trades"],
                "oos_trades": oos_stats["trades"],
                "is_pnl": is_stats["total_pnl"],
                "oos_pnl": oos_stats["total_pnl"],
                "is_win_rate": is_stats["win_rate"],
                "oos_win_rate": oos_stats["win_rate"],
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "is_mean_rr": is_stats["mean_rr"],
                "oos_mean_rr": oos_stats["mean_rr"],
            })
        except FileNotFoundError:
            print(f"No data for {pattern} {timeframe} {symbol}")
            continue

    return pd.DataFrame(results)


def main():
    """Run diagnostic analysis."""
    print("=" * 80)
    print("MULTI-SYMBOL 3-2 PATTERN IS vs OOS ANALYSIS")
    print("=" * 80)

    # Multi-symbol analysis
    summary = analyze_multi_symbol("3-2", "1D")
    print("\n" + "-" * 80)
    print("SUMMARY ACROSS ALL SYMBOLS (3-2 1D)")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Trades':<8} {'IS P&L':>12} {'OOS P&L':>12} {'IS WR':>8} {'OOS WR':>8} {'IS Shrp':>8} {'OOS Shrp':>8}")
    print("-" * 80)

    total_is_pnl = 0
    total_oos_pnl = 0
    for _, row in summary.iterrows():
        print(f"{row['symbol']:<8} {row['total_trades']:<8} ${row['is_pnl']:>10,.0f} ${row['oos_pnl']:>10,.0f} {row['is_win_rate']:>7.1f}% {row['oos_win_rate']:>7.1f}% {row['is_sharpe']:>8.2f} {row['oos_sharpe']:>8.2f}")
        total_is_pnl += row['is_pnl']
        total_oos_pnl += row['oos_pnl']

    print("-" * 80)
    print(f"{'TOTAL':<8} {summary['total_trades'].sum():<8} ${total_is_pnl:>10,.0f} ${total_oos_pnl:>10,.0f}")

    # Profitability check
    print("\n" + "=" * 80)
    print("KEY FINDING: IS vs OOS PROFITABILITY")
    print("=" * 80)
    oos_profitable = (summary['oos_pnl'] > 0).sum()
    oos_total = len(summary)
    print(f"Symbols with POSITIVE OOS P&L: {oos_profitable}/{oos_total}")
    print(f"Total IS P&L:  ${total_is_pnl:,.2f}")
    print(f"Total OOS P&L: ${total_oos_pnl:,.2f}")
    if total_oos_pnl > 0:
        print("CONCLUSION: 3-2 pattern IS PROFITABLE in both IS and OOS periods!")
    else:
        print("CONCLUSION: 3-2 pattern is NOT profitable in OOS period.")

    # Also run detailed SPY analysis
    print("\n" + "=" * 80)
    print("DETAILED SPY ANALYSIS")
    print("=" * 80)

    print("Loading 3-2 1D SPY trade data...")

    # Load data
    df = load_trade_data("3-2", "1D", "SPY")
    print(f"Loaded {len(df)} trades")

    # Split into IS and OOS
    is_df, oos_df = split_is_oos(df, is_ratio=0.7)

    # Calculate statistics
    is_stats = calculate_period_stats(is_df, "In-Sample (70%)")
    oos_stats = calculate_period_stats(oos_df, "Out-of-Sample (30%)")

    # Calculate Sharpe estimates
    is_sharpe = calculate_sharpe_estimate(is_df)
    oos_sharpe = calculate_sharpe_estimate(oos_df)

    # Regime shift analysis
    regime_analysis = analyze_regime_shift(is_df, oos_df)

    # Print report
    print_report(is_stats, oos_stats, regime_analysis, is_sharpe, oos_sharpe)

    # Show OOS trades detail
    print(f"\nOOS TRADE DETAILS (All {oos_stats['trades']} trades):")
    print("-" * 80)
    oos_display = oos_df[["entry_date", "pnl", "pnl_pct", "exit_type", "direction", "magnitude_pct", "days_held"]].copy()
    oos_display["entry_date"] = pd.to_datetime(oos_display["entry_date"], utc=True).dt.strftime("%Y-%m-%d")
    oos_display["pnl"] = oos_display["pnl"].apply(lambda x: f"${x:,.2f}")
    oos_display["pnl_pct"] = oos_display["pnl_pct"].apply(lambda x: f"{x:.1f}%")
    oos_display["magnitude_pct"] = oos_display["magnitude_pct"].apply(lambda x: f"{x:.2f}%")
    oos_display["direction"] = oos_display["direction"].map({1: "BULL", -1: "BEAR"})
    print(oos_display.to_string(index=False))

    # Identify outliers
    print("\n" + "-" * 40)
    print("TOP 5 OOS LOSERS:")
    losers, winners = identify_outliers(oos_df, n_top=5)
    losers_display = losers.copy()
    losers_display["entry_date"] = pd.to_datetime(losers_display["entry_date"], utc=True).dt.strftime("%Y-%m-%d")
    losers_display["direction"] = losers_display["direction"].map({1: "BULL", -1: "BEAR"})
    print(losers_display.to_string(index=False))

    print("\nTOP 5 OOS WINNERS:")
    winners_display = winners.copy()
    winners_display["entry_date"] = pd.to_datetime(winners_display["entry_date"], utc=True).dt.strftime("%Y-%m-%d")
    winners_display["direction"] = winners_display["direction"].map({1: "BULL", -1: "BEAR"})
    print(winners_display.to_string(index=False))

    return is_df, oos_df, is_stats, oos_stats


if __name__ == "__main__":
    main()
