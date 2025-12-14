"""
STRAT Options QuantStats Tearsheet Generator

Generates professional QuantStats tearsheet comparing STRAT options strategy
to SPY buy-and-hold benchmark using ThetaData backtest results.

Session 84: Updated to use ThetaData backtest CSV files.
Session 85: Added support for --thetadata flag to load from ThetaData CSVs.

Usage:
    uv run python scripts/generate_strat_options_quantstats.py --symbol SPY --risk 7 --thetadata

Output:
    reports/strat_options_SPY_7pct_quantstats.html
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# Configuration
CONFIG = {
    'starting_capital': 25000,
    'risk_per_trade_pct': 0.02,  # 2% risk per trade
    'timeframes': ['1D', '1W', '1M'],
    'symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],  # Index ETFs only
    'max_holding_bars': {'1D': 18, '1W': 4, '1M': 2},
    'output_dir': 'reports',
    'output_file': 'strat_options_quantstats_tearsheet.html'
}


def load_thetadata_trades(symbol: str, risk_pct: float, reports_dir: str = 'reports') -> pd.DataFrame:
    """
    Load trades from ThetaData backtest CSV files.

    Args:
        symbol: Symbol (e.g., 'SPY')
        risk_pct: Risk percentage (e.g., 7.0)
        reports_dir: Directory containing CSV files

    Returns:
        DataFrame with all trades from ThetaData backtest
    """
    risk_int = int(risk_pct)
    filepath = Path(reports_dir) / f'options_backtest_{symbol}_{risk_int}pct_thetadata.csv'

    if not filepath.exists():
        raise FileNotFoundError(f"ThetaData backtest file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Parse dates
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])

    # Rename columns for compatibility
    df['symbol'] = symbol
    df['pnl_dollars'] = df['pnl']

    # Map outcome to magnitude_hit for metrics
    df['magnitude_hit'] = df['outcome'] == 'target_hit'

    print(f"  Loaded {len(df)} trades from {filepath}")
    print(f"  Date range: {df['entry_date'].min().date()} to {df['exit_date'].max().date()}")

    return df


def load_trade_data(timeframes: list, symbols: list = None, scripts_dir: str = 'scripts') -> pd.DataFrame:
    """
    Load and merge trades from validation CSV files.

    Args:
        timeframes: List of timeframes ['1D', '1W', '1M']
        symbols: Optional filter for specific symbols
        scripts_dir: Directory containing CSV files

    Returns:
        DataFrame with all trades, parsed dates, and calculated exit dates
    """
    all_trades = []

    for tf in timeframes:
        filepath = Path(scripts_dir) / f'strat_validation_{tf}.csv'
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found")
            continue

        df = pd.read_csv(filepath)

        # Parse entry_date and remove timezone for QuantStats compatibility
        # Use utc=True to handle mixed timezones (EST/EDT), then convert to naive
        df['entry_date'] = pd.to_datetime(df['entry_date'], utc=True).dt.tz_localize(None)
        df['timeframe'] = tf

        # Filter symbols if specified
        if symbols:
            df = df[df['symbol'].isin(symbols)]

        all_trades.append(df)
        print(f"  Loaded {len(df)} trades from {tf}")

    if not all_trades:
        raise ValueError("No trade data found")

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values('entry_date')

    return combined


def calculate_exit_dates(trades_df: pd.DataFrame, max_holding_bars: Dict) -> pd.DataFrame:
    """
    Calculate exit dates based on bars_to_magnitude, bars_to_stop, or max holding.

    For trades that hit magnitude: exit_date = entry_date + bars_to_magnitude
    For trades that hit stop: exit_date = entry_date + bars_to_stop
    For other trades: exit_date = entry_date + max_holding_bars
    """
    trades = trades_df.copy()

    def get_exit_date(row):
        tf = row['timeframe']
        entry = row['entry_date']

        # Time offset per bar based on timeframe
        if tf == '1D':
            offset_func = lambda bars: pd.Timedelta(days=max(1, bars))
        elif tf == '1W':
            offset_func = lambda bars: pd.Timedelta(weeks=max(1, bars))
        else:  # 1M
            offset_func = lambda bars: pd.Timedelta(days=max(30, bars * 30))

        # Determine bars to exit
        if row['magnitude_hit'] and pd.notna(row.get('bars_to_magnitude')):
            bars = int(row['bars_to_magnitude'])
        elif row.get('stop_hit', False) and pd.notna(row.get('bars_to_stop')):
            bars = int(row['bars_to_stop'])
        else:
            bars = max_holding_bars.get(tf, 10)

        # Ensure at least 1 bar (same-day exit for immediate hits)
        bars = max(1, bars)

        return entry + offset_func(bars)

    trades['exit_date'] = trades.apply(get_exit_date, axis=1)
    return trades


def calculate_position_pnl(trades_df: pd.DataFrame, capital: float, risk_pct: float) -> pd.DataFrame:
    """
    Calculate dollar P&L for each trade using 2% risk position sizing.

    Position size = (capital * risk_pct) / (entry_price - stop_price)
    P&L dollars = position_size * entry_price * (actual_pnl_pct / 100)
    """
    trades = trades_df.copy()

    def calc_pnl(row):
        entry = row['entry_price']
        stop = row['stop_price']
        pnl_pct = row['actual_pnl_pct']

        # Risk per share = distance to stop
        risk_per_share = abs(entry - stop)

        if risk_per_share < 0.01:  # Avoid division by near-zero
            risk_per_share = entry * 0.02  # Default 2% of entry price

        # Position size based on fixed risk
        risk_dollars = capital * risk_pct
        shares = risk_dollars / risk_per_share

        # Dollar P&L
        pnl_dollars = shares * entry * (pnl_pct / 100)

        return pnl_dollars

    trades['pnl_dollars'] = trades.apply(calc_pnl, axis=1)
    return trades


def build_daily_returns(trades_df: pd.DataFrame, capital: float) -> pd.Series:
    """
    Construct daily returns series from trade-level data.

    Strategy: Aggregate all trade exits by date to get daily P&L,
    then compute daily percentage returns from equity curve.

    This handles overlapping trades correctly - multiple trades
    exiting on the same day sum their P&L.
    """
    # Aggregate P&L by exit date (normalize to date only, no time)
    daily_pnl = trades_df.groupby(trades_df['exit_date'].dt.normalize())['pnl_dollars'].sum()

    # Get full date range (entry to last exit) - normalize to remove time component
    start_date = trades_df['entry_date'].min().normalize()
    end_date = trades_df['exit_date'].max().normalize()

    # Create full business day index (normalized to midnight)
    full_index = pd.date_range(start=start_date, end=end_date, freq='B').normalize()

    # Reindex to include all business days (0 P&L on days with no exits)
    daily_pnl = daily_pnl.reindex(full_index, fill_value=0)

    # Build equity curve
    equity = capital + daily_pnl.cumsum()

    # Prepend starting capital for return calculation
    equity_with_start = pd.concat([
        pd.Series([capital], index=[equity.index[0] - pd.Timedelta(days=1)]),
        equity
    ])

    # Daily percentage returns
    returns = equity_with_start.pct_change().dropna()
    returns.name = 'STRAT_Options'

    return returns


def get_benchmark_returns(start_date: str, end_date: str) -> pd.Series:
    """
    Download SPY data and calculate buy-and-hold returns for the same period.
    """
    print(f"  Downloading SPY data from {start_date} to {end_date}...")
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

    if len(spy) == 0:
        raise ValueError("Failed to download SPY data")

    # Handle yfinance DataFrame structure (may have MultiIndex columns)
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']

    # Ensure we have a Series
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]

    # Ensure timezone-naive for QuantStats compatibility
    if hasattr(spy_close.index, 'tz') and spy_close.index.tz is not None:
        spy_close.index = spy_close.index.tz_localize(None)

    benchmark_returns = spy_close.pct_change().dropna()
    benchmark_returns.name = 'SPY_BuyHold'

    return benchmark_returns


def generate_tearsheet(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    output_file: str,
    title: str = 'STRAT Options Strategy vs SPY Buy-and-Hold'
) -> str:
    """
    Generate comprehensive QuantStats HTML tearsheet.

    Follows pattern from:
    strategy_research/credit_spread/scripts/generate_quantstats_report.py
    """
    # Align returns to common date range
    common_start = max(strategy_returns.index.min(), benchmark_returns.index.min())
    common_end = min(strategy_returns.index.max(), benchmark_returns.index.max())

    strategy_returns = strategy_returns[common_start:common_end]
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0)

    print(f"  Date range: {common_start.date()} to {common_end.date()}")
    print(f"  Strategy returns: {len(strategy_returns)} days")
    print(f"  Benchmark returns: {len(benchmark_returns)} days")

    # Extend pandas for QuantStats
    qs.extend_pandas()

    # Generate full HTML report
    qs.reports.html(
        strategy_returns,
        benchmark_returns,
        output=output_file,
        title=title,
        download_filename='strat_options_tearsheet.html'
    )

    return output_file


def calculate_trade_metrics(trades_df: pd.DataFrame, capital: float) -> dict:
    """
    Calculate comprehensive trade metrics for the report.

    Returns dict with all metrics for display and HTML embedding.
    """
    metrics = {}

    # Basic metrics
    metrics['total_trades'] = len(trades_df)
    metrics['total_pnl'] = trades_df['pnl_dollars'].sum()
    metrics['final_equity'] = capital + metrics['total_pnl']
    metrics['total_return_pct'] = metrics['total_pnl'] / capital * 100

    # Win/Loss metrics
    winners = trades_df[trades_df['pnl_dollars'] > 0]
    losers = trades_df[trades_df['pnl_dollars'] <= 0]
    metrics['win_count'] = len(winners)
    metrics['loss_count'] = len(losers)
    metrics['win_rate'] = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

    # Best/Worst trades
    best_idx = trades_df['pnl_dollars'].idxmax()
    worst_idx = trades_df['pnl_dollars'].idxmin()
    metrics['best_trade'] = trades_df.loc[best_idx]
    metrics['worst_trade'] = trades_df.loc[worst_idx]

    # Average win/loss
    metrics['avg_win'] = winners['pnl_dollars'].mean() if len(winners) > 0 else 0
    metrics['avg_loss'] = losers['pnl_dollars'].mean() if len(losers) > 0 else 0

    # Profit factor
    gross_profit = winners['pnl_dollars'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl_dollars'].sum()) if len(losers) > 0 else 1
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calls vs Puts (direction: bullish = CALL, bearish = PUT)
    calls = trades_df[trades_df['direction'].isin(['bullish', 1])]
    puts = trades_df[trades_df['direction'].isin(['bearish', -1])]
    metrics['call_count'] = len(calls)
    metrics['put_count'] = len(puts)
    metrics['call_pnl'] = calls['pnl_dollars'].sum() if len(calls) > 0 else 0
    metrics['put_pnl'] = puts['pnl_dollars'].sum() if len(puts) > 0 else 0
    metrics['call_win_rate'] = calls['magnitude_hit'].mean() if len(calls) > 0 else 0
    metrics['put_win_rate'] = puts['magnitude_hit'].mean() if len(puts) > 0 else 0

    # Win/Loss streaks
    trades_df = trades_df.copy()
    trades_df['is_win'] = trades_df['pnl_dollars'] > 0

    # Calculate streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    current_streak_type = None

    for is_win in trades_df['is_win']:
        if current_streak_type is None:
            current_streak_type = is_win
            streak = 1
        elif is_win == current_streak_type:
            streak += 1
        else:
            if current_streak_type:
                max_win_streak = max(max_win_streak, streak)
            else:
                max_loss_streak = max(max_loss_streak, streak)
            current_streak_type = is_win
            streak = 1

    # Don't forget the last streak
    if current_streak_type:
        max_win_streak = max(max_win_streak, streak)
    else:
        max_loss_streak = max(max_loss_streak, streak)

    metrics['max_win_streak'] = max_win_streak
    metrics['max_loss_streak'] = max_loss_streak

    # By pattern type
    metrics['by_pattern'] = {}
    for pattern in trades_df['pattern_type'].unique():
        p_trades = trades_df[trades_df['pattern_type'] == pattern]
        metrics['by_pattern'][pattern] = {
            'count': len(p_trades),
            'pnl': p_trades['pnl_dollars'].sum(),
            'win_rate': p_trades['magnitude_hit'].mean()
        }

    # By timeframe
    metrics['by_timeframe'] = {}
    for tf in trades_df['timeframe'].unique():
        tf_trades = trades_df[trades_df['timeframe'] == tf]
        metrics['by_timeframe'][tf] = {
            'count': len(tf_trades),
            'pnl': tf_trades['pnl_dollars'].sum(),
            'win_rate': tf_trades['magnitude_hit'].mean()
        }

    # By symbol
    metrics['by_symbol'] = {}
    for sym in trades_df['symbol'].unique():
        sym_trades = trades_df[trades_df['symbol'] == sym]
        metrics['by_symbol'][sym] = {
            'count': len(sym_trades),
            'pnl': sym_trades['pnl_dollars'].sum(),
            'win_rate': sym_trades['magnitude_hit'].mean()
        }

    return metrics


def print_summary_metrics(trades_df: pd.DataFrame, capital: float):
    """Print summary metrics before generating tearsheet."""
    metrics = calculate_trade_metrics(trades_df, capital)

    print("\n" + "=" * 60)
    print("STRATEGY SUMMARY METRICS")
    print("=" * 60)

    print(f"\nStarting Capital: ${capital:,.0f}")
    print(f"Final Equity: ${metrics['final_equity']:,.0f}")
    print(f"Total P&L: ${metrics['total_pnl']:,.0f} ({metrics['total_return_pct']:.1f}%)")
    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%} ({metrics['win_count']}W / {metrics['loss_count']}L)")

    print("\n--- Best/Worst Trades ---")
    best = metrics['best_trade']
    worst = metrics['worst_trade']
    print(f"Best Trade:  ${best['pnl_dollars']:,.0f} ({best['symbol']} {best['pattern_type']} on {best['entry_date'].date()})")
    print(f"Worst Trade: ${worst['pnl_dollars']:,.0f} ({worst['symbol']} {worst['pattern_type']} on {worst['entry_date'].date()})")

    print("\n--- Win/Loss Analysis ---")
    print(f"Average Win:  ${metrics['avg_win']:,.0f}")
    print(f"Average Loss: ${metrics['avg_loss']:,.0f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Win Streak:  {metrics['max_win_streak']} trades")
    print(f"Max Loss Streak: {metrics['max_loss_streak']} trades")

    print("\n--- Calls vs Puts ---")
    print(f"CALLs: {metrics['call_count']} trades, ${metrics['call_pnl']:,.0f} P&L, {metrics['call_win_rate']:.1%} win rate")
    print(f"PUTs:  {metrics['put_count']} trades, ${metrics['put_pnl']:,.0f} P&L, {metrics['put_win_rate']:.1%} win rate")

    # By timeframe
    print("\n--- By Timeframe ---")
    for tf, data in metrics['by_timeframe'].items():
        print(f"  {tf}: {data['count']} trades, ${data['pnl']:,.0f} P&L, {data['win_rate']:.1%} win rate")

    # By symbol
    print("\n--- By Symbol ---")
    for sym, data in metrics['by_symbol'].items():
        print(f"  {sym}: {data['count']} trades, ${data['pnl']:,.0f} P&L, {data['win_rate']:.1%} win rate")

    # By pattern (top 5)
    print("\n--- By Pattern Type (Top 5 by P&L) ---")
    sorted_patterns = sorted(metrics['by_pattern'].items(), key=lambda x: x[1]['pnl'], reverse=True)[:5]
    for pattern, data in sorted_patterns:
        print(f"  {pattern}: {data['count']} trades, ${data['pnl']:,.0f} P&L, {data['win_rate']:.1%} win rate")

    return metrics


def append_trade_metrics_to_html(html_path: str, metrics: dict, capital: float):
    """
    Append trade metrics section to the QuantStats HTML report.
    """
    # Read the existing HTML
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Build the trade metrics HTML section
    best = metrics['best_trade']
    worst = metrics['worst_trade']

    trade_metrics_html = f'''
    <div style="margin: 40px auto; max-width: 1000px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h2 style="border-bottom: 2px solid #333; padding-bottom: 10px;">Trade-Level Metrics</h2>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: #28a745;">${metrics['total_pnl']:,.0f}</div>
                <div style="color: #666;">Total P&L ({metrics['total_return_pct']:.1f}%)</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: #007bff;">{metrics['total_trades']}</div>
                <div style="color: #666;">Total Trades</div>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 28px; font-weight: bold; color: #17a2b8;">{metrics['win_rate']:.1%}</div>
                <div style="color: #666;">Win Rate ({metrics['win_count']}W / {metrics['loss_count']}L)</div>
            </div>
        </div>

        <h3 style="margin-top: 30px;">Best & Worst Trades</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #e8f5e9;">
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Best Trade</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd; color: #28a745; font-weight: bold;">${best['pnl_dollars']:,.0f}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{best['symbol']} {best['pattern_type']}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{best['entry_date'].date()}</td>
            </tr>
            <tr style="background: #ffebee;">
                <td style="padding: 12px; border: 1px solid #ddd;"><strong>Worst Trade</strong></td>
                <td style="padding: 12px; border: 1px solid #ddd; color: #dc3545; font-weight: bold;">${worst['pnl_dollars']:,.0f}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{worst['symbol']} {worst['pattern_type']}</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{worst['entry_date'].date()}</td>
            </tr>
        </table>

        <h3 style="margin-top: 30px;">Win/Loss Analysis</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Average Win</td>
                <td style="padding: 10px; border: 1px solid #ddd; color: #28a745; font-weight: bold;">${metrics['avg_win']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">Max Win Streak</td>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{metrics['max_win_streak']} trades</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Average Loss</td>
                <td style="padding: 10px; border: 1px solid #ddd; color: #dc3545; font-weight: bold;">${metrics['avg_loss']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd;">Max Loss Streak</td>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{metrics['max_loss_streak']} trades</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Profit Factor</td>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;" colspan="3">{metrics['profit_factor']:.2f}</td>
            </tr>
        </table>

        <h3 style="margin-top: 30px;">CALLs vs PUTs</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Direction</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">P&L</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">CALLs (Bullish)</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics['call_count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics['call_pnl'] > 0 else '#dc3545'};">${metrics['call_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics['call_win_rate']:.1%}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">PUTs (Bearish)</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics['put_count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics['put_pnl'] > 0 else '#dc3545'};">${metrics['put_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics['put_win_rate']:.1%}</td>
            </tr>
        </table>

        <h3 style="margin-top: 30px;">Performance by Timeframe</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Timeframe</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">P&L</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>'''

    for tf in ['1D', '1W', '1M']:
        if tf in metrics['by_timeframe']:
            data = metrics['by_timeframe'][tf]
            trade_metrics_html += f'''
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{tf}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if data['pnl'] > 0 else '#dc3545'};">${data['pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['win_rate']:.1%}</td>
            </tr>'''

    trade_metrics_html += '''
        </table>

        <h3 style="margin-top: 30px;">Performance by Symbol</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Symbol</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">P&L</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>'''

    # Sort symbols by P&L
    sorted_symbols = sorted(metrics['by_symbol'].items(), key=lambda x: x[1]['pnl'], reverse=True)
    for sym, data in sorted_symbols:
        trade_metrics_html += f'''
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{sym}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if data['pnl'] > 0 else '#dc3545'};">${data['pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['win_rate']:.1%}</td>
            </tr>'''

    trade_metrics_html += '''
        </table>

        <h3 style="margin-top: 30px;">Performance by Pattern Type</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Pattern</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">P&L</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>'''

    # Sort patterns by P&L
    sorted_patterns = sorted(metrics['by_pattern'].items(), key=lambda x: x[1]['pnl'], reverse=True)
    for pattern, data in sorted_patterns:
        trade_metrics_html += f'''
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{pattern}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if data['pnl'] > 0 else '#dc3545'};">${data['pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{data['win_rate']:.1%}</td>
            </tr>'''

    trade_metrics_html += '''
        </table>

        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            Starting Capital: $''' + f"{capital:,.0f}" + ''' |
            Generated by ATLAS STRAT Options Backtest
        </p>
    </div>
    '''

    # Insert before closing body tag
    html_content = html_content.replace('</body>', trade_metrics_html + '\n</body>')

    # Write the updated HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate STRAT Options QuantStats tearsheet'
    )
    parser.add_argument(
        '--risk',
        type=float,
        default=2.0,
        help='Risk percentage per trade (default: 2.0)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Single symbol to analyze (e.g., SPY). Default: all index ETFs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename (default: auto-generated based on parameters)'
    )
    parser.add_argument(
        '--thetadata',
        action='store_true',
        help='Use ThetaData backtest CSV files instead of validation CSVs'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Override CONFIG with CLI arguments
    risk_pct = args.risk / 100.0  # Convert percentage to decimal
    symbols = [args.symbol] if args.symbol else CONFIG['symbols']

    # Generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        symbol_str = args.symbol if args.symbol else 'all'
        source_str = 'thetadata' if args.thetadata else 'validation'
        output_file = f'strat_options_{symbol_str}_{int(args.risk)}pct_{source_str}.html'

    print("=" * 80)
    print("STRAT OPTIONS QUANTSTATS TEARSHEET GENERATOR")
    print("=" * 80)
    print(f"Risk per trade: {args.risk}%")
    print(f"Data source: {'ThetaData backtest' if args.thetadata else 'Validation CSVs'}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {output_file}")

    # Step 1: Load trades
    print("\n[1/6] Loading trade data...")

    if args.thetadata:
        # Load from ThetaData backtest CSV
        if not args.symbol:
            raise ValueError("--symbol is required when using --thetadata")
        trades = load_thetadata_trades(args.symbol, args.risk)
        # ThetaData CSVs already have exit_date and pnl calculated
        print(f"  Total: {len(trades)} trades loaded")
    else:
        # Load from validation CSVs (original behavior)
        trades = load_trade_data(
            timeframes=CONFIG['timeframes'],
            symbols=symbols
        )
        print(f"  Total: {len(trades)} trades loaded")
        print(f"  Date range: {trades['entry_date'].min().date()} to {trades['entry_date'].max().date()}")

        # Step 2: Calculate exit dates (only for validation CSVs)
        print("\n[2/6] Calculating exit dates...")
        trades = calculate_exit_dates(trades, CONFIG['max_holding_bars'])

        # Step 3: Position sizing and P&L (only for validation CSVs)
        print("\n[3/6] Calculating position sizes and P&L...")
        trades = calculate_position_pnl(
            trades,
            capital=CONFIG['starting_capital'],
            risk_pct=risk_pct
        )

    # Print summary metrics and get metrics dict
    metrics = print_summary_metrics(trades, CONFIG['starting_capital'])

    # Step 4: Build daily returns
    print("\n[4/6] Building daily returns series...")
    strategy_returns = build_daily_returns(trades, CONFIG['starting_capital'])
    print(f"  Generated {len(strategy_returns)} daily return observations")

    # Step 5: Get benchmark
    print("\n[5/6] Downloading SPY benchmark...")
    benchmark_returns = get_benchmark_returns(
        start_date=strategy_returns.index.min().strftime('%Y-%m-%d'),
        end_date=strategy_returns.index.max().strftime('%Y-%m-%d')
    )

    # Step 6: Generate tearsheet
    print("\n[6/6] Generating QuantStats tearsheet...")

    # Ensure output directory exists
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file

    generate_tearsheet(
        strategy_returns,
        benchmark_returns,
        str(output_path)
    )

    # Append trade metrics to the HTML report
    print("  Appending trade metrics to report...")
    append_trade_metrics_to_html(str(output_path), metrics, CONFIG['starting_capital'])

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nTearsheet saved to: {output_path}")
    print("\nKey features in tearsheet:")
    print("  - Cumulative returns vs SPY benchmark")
    print("  - Monthly returns heatmap")
    print("  - Drawdown periods (ranked)")
    print("  - Rolling Sharpe ratio")
    print("  - Rolling volatility")
    print("  - Comprehensive metrics table")
    print("\nOpen the HTML file in your browser to view the full report.")


if __name__ == '__main__':
    main()
