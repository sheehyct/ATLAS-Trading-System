#!/usr/bin/env python3
"""
Session 83K-62: Compare 3-2 Magnitude Options A/B/C

Runs comparative backtests for four magnitude calculation strategies across
5 symbols on daily timeframe, with 70/30 IS/OOS split.

Options Compared:
    A: Previous Outside Bar (current implementation)
    B-N2: N-bar Swing Pivot (N=2)
    B-N3: N-bar Swing Pivot (N=3)
    C: Always 1.5x R:R Measured Move

Usage:
    uv run python scripts/compare_32_magnitude_options.py
    uv run python scripts/compare_32_magnitude_options.py --symbols SPY QQQ
    uv run python scripts/compare_32_magnitude_options.py --verbose
"""

import argparse
import logging
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strat.magnitude_calculators import (
    MagnitudeCalculator,
    MagnitudeResult,
    OptionA_PreviousOutsideBar,
    OptionB_SwingPivot,
    OptionC_MeasuredMove,
    get_all_calculators,
)
from strat.bar_classifier import classify_bars_nb

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ComparisonConfig:
    """Configuration for magnitude comparison."""
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'AAPL', 'IWM', 'DIA'])
    start_date: str = '2020-01-01'
    end_date: str = '2024-12-31'
    is_ratio: float = 0.7  # 70% IS, 30% OOS
    max_holding_bars: int = 30  # Max bars to hold position
    output_dir: str = 'validation_results/session_83k_magnitude'


# -----------------------------------------------------------------------------
# Data Fetching
# -----------------------------------------------------------------------------

def fetch_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily price data using cache, Alpaca, or Tiingo."""
    # Try cache first
    cache_path = project_root / 'data_cache' / f'{symbol}_1D.parquet'
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            # Filter to date range
            df = df.loc[start_date:end_date]
            if not df.empty:
                logger.debug(f"Loaded {symbol} from cache ({len(df)} bars)")
                return df
        except Exception as e:
            logger.debug(f"Cache read failed for {symbol}: {e}")

    # Try Alpaca
    try:
        import vectorbtpro as vbt
        alpaca_key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
        alpaca_secret = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')

        if alpaca_key and alpaca_secret:
            data = vbt.AlpacaData.pull(
                symbol,
                start=start_date,
                end=end_date,
                timeframe='1 day',
                adjustment='split',
                tz='America/New_York'
            )
            df = data.get()
            if not df.empty:
                logger.info(f"Fetched {symbol} from Alpaca ({len(df)} bars)")
                return df
    except Exception as e:
        logger.warning(f"Alpaca failed for {symbol}: {e}")

    # Fallback to Tiingo
    try:
        from integrations.tiingo_data_fetcher import TiingoDataFetcher
        fetcher = TiingoDataFetcher()
        df = fetcher.fetch_daily_data(symbol, start_date, end_date)
        if not df.empty:
            logger.info(f"Fetched {symbol} from Tiingo ({len(df)} bars)")
            return df
    except Exception as e:
        logger.warning(f"Tiingo failed for {symbol}: {e}")

    raise ValueError(f"Could not fetch data for {symbol}")


# -----------------------------------------------------------------------------
# Pattern Detection and Outcome Measurement
# -----------------------------------------------------------------------------

@dataclass
class PatternTrade:
    """Represents a single 3-2 pattern trade."""
    pattern_idx: int
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: int  # 1=bullish, -1=bearish
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_type: str  # 'TARGET', 'STOP', 'TIME_EXIT'
    bars_held: int
    pnl: float
    pnl_pct: float
    rr_ratio: float
    magnitude_pct: float
    method_used: str
    lookback_distance: Optional[int]


def detect_32_patterns_with_calculator(
    data: pd.DataFrame,
    calculator: MagnitudeCalculator,
    max_holding_bars: int = 30
) -> List[PatternTrade]:
    """
    Detect 3-2 patterns and measure outcomes using specified magnitude calculator.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC price data with DatetimeIndex
    calculator : MagnitudeCalculator
        Magnitude calculation strategy to use
    max_holding_bars : int
        Maximum bars to hold before time exit

    Returns
    -------
    List[PatternTrade]
        List of completed trades
    """
    # Handle column names (VBT returns lowercase or titlecase)
    if 'High' in data.columns:
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
    else:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

    # Classify bars
    classifications = classify_bars_nb(high, low)

    trades = []
    n = len(classifications)

    for i in range(1, n):
        bar1_class = classifications[i-1]  # Outside bar (3 or -3)
        bar2_class = classifications[i]    # Directional bar (2 or -2)

        # 3-2U: Outside bar followed by 2U (bullish)
        if abs(bar1_class) == 3 and bar2_class == 2:
            entry_price = high[i-1]  # Break above outside bar high
            stop_price = low[i-1]    # Stop at outside bar low
            direction = 1

            # Calculate target using specified calculator
            result = calculator.calculate_target(
                entry_price=entry_price,
                stop_price=stop_price,
                direction=direction,
                high=high,
                low=low,
                classifications=classifications,
                pattern_idx=i
            )

            # Measure outcome
            trade = measure_pattern_outcome(
                data=data,
                pattern_idx=i,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=result.target_price,
                direction=direction,
                max_holding_bars=max_holding_bars,
                high=high,
                low=low,
                close=close,
                method_used=result.method_used,
                lookback_distance=result.lookback_distance,
                rr_ratio=result.rr_ratio
            )
            if trade:
                trades.append(trade)

        # 3-2D: Outside bar followed by 2D (bearish)
        elif abs(bar1_class) == 3 and bar2_class == -2:
            entry_price = low[i-1]   # Break below outside bar low
            stop_price = high[i-1]   # Stop at outside bar high
            direction = -1

            # Calculate target using specified calculator
            result = calculator.calculate_target(
                entry_price=entry_price,
                stop_price=stop_price,
                direction=direction,
                high=high,
                low=low,
                classifications=classifications,
                pattern_idx=i
            )

            # Measure outcome
            trade = measure_pattern_outcome(
                data=data,
                pattern_idx=i,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=result.target_price,
                direction=direction,
                max_holding_bars=max_holding_bars,
                high=high,
                low=low,
                close=close,
                method_used=result.method_used,
                lookback_distance=result.lookback_distance,
                rr_ratio=result.rr_ratio
            )
            if trade:
                trades.append(trade)

    return trades


def measure_pattern_outcome(
    data: pd.DataFrame,
    pattern_idx: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    direction: int,
    max_holding_bars: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    method_used: str,
    lookback_distance: Optional[int],
    rr_ratio: float
) -> Optional[PatternTrade]:
    """
    Measure pattern outcome by simulating trade execution.

    Entry on pattern bar, exit on target/stop/time.
    """
    n = len(high)
    entry_bar = pattern_idx  # Entry on the pattern bar (the 2U/2D bar)

    if entry_bar >= n - 1:
        return None  # Not enough data for exit

    entry_date = data.index[entry_bar]
    exit_bar = None
    exit_price = None
    exit_type = None

    # Scan forward for exit
    for j in range(entry_bar + 1, min(entry_bar + max_holding_bars + 1, n)):
        bar_high = high[j]
        bar_low = low[j]

        if direction == 1:  # Bullish
            # Check stop first (more conservative)
            if bar_low <= stop_price:
                exit_bar = j
                exit_price = stop_price
                exit_type = 'STOP'
                break
            # Check target
            if bar_high >= target_price:
                exit_bar = j
                exit_price = target_price
                exit_type = 'TARGET'
                break
        else:  # Bearish
            # Check stop first
            if bar_high >= stop_price:
                exit_bar = j
                exit_price = stop_price
                exit_type = 'STOP'
                break
            # Check target
            if bar_low <= target_price:
                exit_bar = j
                exit_price = target_price
                exit_type = 'TARGET'
                break

    # Time exit if no target/stop hit
    if exit_bar is None:
        exit_bar = min(entry_bar + max_holding_bars, n - 1)
        exit_price = close[exit_bar]
        exit_type = 'TIME_EXIT'

    # Calculate P&L
    if direction == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    pnl_pct = (pnl / entry_price) * 100

    # Calculate magnitude percentage
    magnitude_pct = abs(target_price - entry_price) / entry_price * 100

    return PatternTrade(
        pattern_idx=pattern_idx,
        entry_date=entry_date,
        exit_date=data.index[exit_bar],
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_price=exit_price,
        exit_type=exit_type,
        bars_held=exit_bar - entry_bar,
        pnl=pnl,
        pnl_pct=pnl_pct,
        rr_ratio=rr_ratio,
        magnitude_pct=magnitude_pct,
        method_used=method_used,
        lookback_distance=lookback_distance
    )


# -----------------------------------------------------------------------------
# Statistics and Comparison
# -----------------------------------------------------------------------------

def trades_to_dataframe(trades: List[PatternTrade], symbol: str, calculator_name: str) -> pd.DataFrame:
    """Convert trades list to DataFrame."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        records.append({
            'symbol': symbol,
            'calculator': calculator_name,
            'pattern_idx': t.pattern_idx,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'stop_price': t.stop_price,
            'target_price': t.target_price,
            'exit_price': t.exit_price,
            'exit_type': t.exit_type,
            'bars_held': t.bars_held,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'rr_ratio': t.rr_ratio,
            'magnitude_pct': t.magnitude_pct,
            'method_used': t.method_used,
            'lookback_distance': t.lookback_distance
        })

    return pd.DataFrame(records)


def split_is_oos(df: pd.DataFrame, is_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split trades into IS and OOS periods by entry date (70/30 holdout)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values('entry_date').reset_index(drop=True)
    n_total = len(df_sorted)
    n_is = int(n_total * is_ratio)

    is_df = df_sorted.iloc[:n_is].copy()
    oos_df = df_sorted.iloc[n_is:].copy()

    return is_df, oos_df


def calculate_period_stats(df: pd.DataFrame) -> Dict:
    """Calculate statistics for a period."""
    if df.empty:
        return {
            'trades': 0, 'total_pnl': 0, 'mean_pnl': 0, 'win_rate': 0,
            'mean_rr': 0, 'sharpe': np.nan, 'target_hits': 0, 'stop_hits': 0
        }

    total_pnl = df['pnl'].sum()
    mean_pnl = df['pnl'].mean()
    std_pnl = df['pnl'].std()

    wins = df[df['pnl'] > 0]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    mean_rr = df['rr_ratio'].mean()

    # Sharpe estimate (annualized)
    if len(df) >= 2 and std_pnl > 0:
        sharpe = (mean_pnl / std_pnl) * np.sqrt(252)
    else:
        sharpe = np.nan

    # Exit type counts
    target_hits = (df['exit_type'] == 'TARGET').sum()
    stop_hits = (df['exit_type'] == 'STOP').sum()
    time_exits = (df['exit_type'] == 'TIME_EXIT').sum()

    return {
        'trades': len(df),
        'total_pnl': total_pnl,
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'win_rate': win_rate,
        'mean_rr': mean_rr,
        'sharpe': sharpe,
        'target_hits': target_hits,
        'stop_hits': stop_hits,
        'time_exits': time_exits,
        'target_pct': target_hits / len(df) * 100 if len(df) > 0 else 0
    }


# -----------------------------------------------------------------------------
# Main Comparison
# -----------------------------------------------------------------------------

def run_comparison(config: ComparisonConfig, verbose: bool = False) -> pd.DataFrame:
    """Run full comparison across all calculators and symbols."""
    calculators = get_all_calculators()
    all_results = []
    all_trades = []

    print("=" * 90)
    print("3-2 MAGNITUDE OPTION COMPARISON")
    print(f"Session 83K-62 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)
    print(f"\nSymbols: {', '.join(config.symbols)}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"IS/OOS Split: {int(config.is_ratio*100)}% / {int((1-config.is_ratio)*100)}%")
    print(f"Max Holding: {config.max_holding_bars} bars")
    print("\nOptions:")
    for calc in calculators:
        print(f"  - {calc.name}")
    print()

    # Process each symbol
    for symbol in config.symbols:
        print(f"\n--- Processing {symbol} ---")

        try:
            data = fetch_price_data(symbol, config.start_date, config.end_date)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue

        # Run each calculator
        for calc in calculators:
            trades = detect_32_patterns_with_calculator(
                data, calc, config.max_holding_bars
            )

            if not trades:
                logger.warning(f"  {calc.name}: No trades detected")
                continue

            # Convert to DataFrame
            trades_df = trades_to_dataframe(trades, symbol, calc.name)
            all_trades.append(trades_df)

            # Split IS/OOS
            is_df, oos_df = split_is_oos(trades_df, config.is_ratio)

            # Calculate stats
            is_stats = calculate_period_stats(is_df)
            oos_stats = calculate_period_stats(oos_df)

            result = {
                'symbol': symbol,
                'calculator': calc.name,
                'is_trades': is_stats['trades'],
                'oos_trades': oos_stats['trades'],
                'is_pnl': is_stats['total_pnl'],
                'oos_pnl': oos_stats['total_pnl'],
                'is_win_rate': is_stats['win_rate'],
                'oos_win_rate': oos_stats['win_rate'],
                'is_mean_rr': is_stats['mean_rr'],
                'oos_mean_rr': oos_stats['mean_rr'],
                'is_sharpe': is_stats['sharpe'],
                'oos_sharpe': oos_stats['sharpe'],
                'is_target_pct': is_stats['target_pct'],
                'oos_target_pct': oos_stats['target_pct'],
            }
            all_results.append(result)

            if verbose:
                print(f"  {calc.name}: {is_stats['trades']}+{oos_stats['trades']} trades, "
                      f"IS ${is_stats['total_pnl']:.2f}, OOS ${oos_stats['total_pnl']:.2f}")

    # Combine results
    results_df = pd.DataFrame(all_results)

    # Save individual trade files
    if all_trades:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades.to_csv(output_dir / 'all_trades.csv', index=False)
        results_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
        print(f"\nSaved {len(combined_trades)} trades to {output_dir}/all_trades.csv")

    return results_df


def print_comparison_report(results_df: pd.DataFrame):
    """Print formatted comparison report."""
    if results_df.empty:
        print("No results to report.")
        return

    print("\n" + "=" * 90)
    print("AGGREGATE BY OPTION (All Symbols)")
    print("=" * 90)

    # Aggregate by calculator
    agg = results_df.groupby('calculator').agg({
        'is_trades': 'sum',
        'oos_trades': 'sum',
        'is_pnl': 'sum',
        'oos_pnl': 'sum',
        'is_win_rate': 'mean',
        'oos_win_rate': 'mean',
        'is_mean_rr': 'mean',
        'oos_mean_rr': 'mean',
        'is_target_pct': 'mean',
        'oos_target_pct': 'mean',
    }).reset_index()

    # Sort by OOS P&L descending
    agg = agg.sort_values('oos_pnl', ascending=False)

    print(f"\n{'Option':<30} {'Trades':<10} {'IS P&L':>12} {'OOS P&L':>12} {'IS WR':>8} {'OOS WR':>8} {'IS R:R':>8} {'OOS R:R':>8}")
    print("-" * 90)

    for _, row in agg.iterrows():
        total_trades = int(row['is_trades'] + row['oos_trades'])
        print(f"{row['calculator']:<30} {total_trades:<10} "
              f"${row['is_pnl']:>10,.2f} ${row['oos_pnl']:>10,.2f} "
              f"{row['is_win_rate']:>7.1f}% {row['oos_win_rate']:>7.1f}% "
              f"{row['is_mean_rr']:>7.2f} {row['oos_mean_rr']:>7.2f}")

    # Per-symbol breakdown
    print("\n" + "=" * 90)
    print("BY SYMBOL")
    print("=" * 90)

    for symbol in results_df['symbol'].unique():
        print(f"\n{symbol}:")
        sym_df = results_df[results_df['symbol'] == symbol].sort_values('oos_pnl', ascending=False)
        for _, row in sym_df.iterrows():
            print(f"  {row['calculator']:<30} IS ${row['is_pnl']:>8,.2f} OOS ${row['oos_pnl']:>8,.2f} "
                  f"WR {row['is_win_rate']:.0f}%/{row['oos_win_rate']:.0f}% "
                  f"R:R {row['is_mean_rr']:.2f}/{row['oos_mean_rr']:.2f}")

    # Recommendation
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    best_oos_pnl = agg.loc[agg['oos_pnl'].idxmax()]
    best_oos_wr = agg.loc[agg['oos_win_rate'].idxmax()]

    # Calculate IS->OOS degradation
    agg['pnl_degradation'] = (agg['is_pnl'] - agg['oos_pnl']) / agg['is_pnl'].abs() * 100
    most_consistent = agg.loc[agg['pnl_degradation'].idxmin()]

    print(f"\nBest OOS P&L:        {best_oos_pnl['calculator']:<30} (${best_oos_pnl['oos_pnl']:,.2f})")
    print(f"Best OOS Win Rate:   {best_oos_wr['calculator']:<30} ({best_oos_wr['oos_win_rate']:.1f}%)")
    print(f"Most Consistent:     {most_consistent['calculator']:<30} ({most_consistent['pnl_degradation']:.1f}% degradation)")


def verify_sample_trades(config: ComparisonConfig):
    """Verify sample trades from each calculator."""
    print("\n" + "=" * 90)
    print("SAMPLE TRADE VERIFICATION")
    print("=" * 90)

    output_dir = Path(config.output_dir)
    trades_file = output_dir / 'all_trades.csv'

    if not trades_file.exists():
        print("No trades file found for verification.")
        return

    trades_df = pd.read_csv(trades_file)

    calculators = trades_df['calculator'].unique()
    for calc in calculators:
        calc_trades = trades_df[trades_df['calculator'] == calc]

        # Get one bullish and one bearish trade
        bullish = calc_trades[calc_trades['direction'] == 1].head(1)
        bearish = calc_trades[calc_trades['direction'] == -1].head(1)

        print(f"\n--- {calc} ---")
        if not bullish.empty:
            t = bullish.iloc[0]
            print(f"  Bullish: {t['symbol']} {t['entry_date'][:10]} | "
                  f"Entry ${t['entry_price']:.2f} -> Target ${t['target_price']:.2f} | "
                  f"Exit: {t['exit_type']} @ ${t['exit_price']:.2f} | "
                  f"P&L ${t['pnl']:.2f} ({t['method_used']})")
        if not bearish.empty:
            t = bearish.iloc[0]
            print(f"  Bearish: {t['symbol']} {t['entry_date'][:10]} | "
                  f"Entry ${t['entry_price']:.2f} -> Target ${t['target_price']:.2f} | "
                  f"Exit: {t['exit_type']} @ ${t['exit_price']:.2f} | "
                  f"P&L ${t['pnl']:.2f} ({t['method_used']})")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare 3-2 magnitude calculation options')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'AAPL', 'IWM', 'DIA'],
                        help='Symbols to test')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date')
    parser.add_argument('--end-date', default='2024-12-31', help='End date')
    parser.add_argument('--max-holding', type=int, default=30, help='Max holding bars')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    config = ComparisonConfig(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        max_holding_bars=args.max_holding
    )

    # Run comparison
    results_df = run_comparison(config, verbose=args.verbose)

    # Print report
    print_comparison_report(results_df)

    # Verify sample trades
    verify_sample_trades(config)

    print("\n" + "=" * 90)
    print("COMPARISON COMPLETE")
    print("=" * 90)


if __name__ == '__main__':
    main()
