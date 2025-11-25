"""
STRAT Visual Trade Verification Script

Session 75: Created to visually verify pattern detection and trade entries/exits
across all timeframes (1H, 1D, 1W, 1M) for TradingView cross-reference.

PURPOSE:
    Generate specific trade examples with entry dates, entry prices, stop prices,
    and target prices for manual visual verification on TradingView charts.
    This helps validate the STRAT pattern detection is working correctly.

USAGE:
    python scripts/visual_trade_verification.py

OUTPUT:
    - Console output with trade details for each pattern/timeframe combination
    - CSV file (reports/visual_verification_trades.csv) for easy reference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

# Suppress future warnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning)


def fetch_data_for_verification(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch OHLC data for verification using paid data sources (Tiingo/Alpaca).

    Priority:
    - Intraday (1H): Alpaca (free tier includes intraday)
    - Daily/Weekly/Monthly: Tiingo (paid, high quality)

    Args:
        symbol: Stock symbol (e.g., 'SPY', 'AAPL')
        timeframe: Timeframe string ('1H', '1D', '1W', '1M')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLC data
    """
    try:
        if timeframe == '1H':
            # Use Alpaca for intraday data
            from data.alpaca import fetch_alpaca_data

            # Calculate period_days from date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            period_days = (end_dt - start_dt).days

            print(f"  Using Alpaca for {symbol} @ {timeframe} ({period_days} days)")
            df = fetch_alpaca_data(symbol, timeframe='1H', period_days=period_days)

            # Normalize column names to uppercase for consistency
            df.columns = [c.title() for c in df.columns]
            return df

        else:
            # Use Tiingo for daily/weekly/monthly data
            from integrations.tiingo_data_fetcher import TiingoDataFetcher

            # Map timeframe to Tiingo frequency
            tf_map = {
                '1D': 'daily',
                '1W': 'weekly',
                '1M': 'monthly',
            }
            frequency = tf_map.get(timeframe, 'daily')

            print(f"  Using Tiingo for {symbol} @ {timeframe} ({frequency})")
            fetcher = TiingoDataFetcher()
            data = fetcher.fetch(symbol, start_date, end_date, timeframe=frequency)
            return data.get()

    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        print(f"  Falling back to VBT Pro YFData...")

        # Fallback to VBT Pro YFData (uses Yahoo Finance internally)
        try:
            import vectorbtpro as vbt

            tf_map = {
                '1H': '1h',
                '1D': '1d',
                '1W': '1wk',
                '1M': '1mo',
            }
            interval = tf_map.get(timeframe, timeframe)

            data = vbt.YFData.pull(
                symbol,
                start=start_date,
                end=end_date,
                timeframe=interval,
                tz='America/New_York'
            )
            return data.get()
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return pd.DataFrame()


def detect_patterns_for_verification(
    data: pd.DataFrame,
    pattern_type: str,
    include_all: bool = False
) -> List[Dict]:
    """
    Detect STRAT patterns and return trade details for verification.

    Args:
        data: OHLC DataFrame
        pattern_type: '3-1-2', '2-1-2', '2-2', or '3-2'
        include_all: If True, include all patterns (no continuation bar filter)

    Returns:
        List of trade dictionaries with entry/stop/target details
    """
    from strat.bar_classifier import classify_bars_nb
    from strat.pattern_detector import (
        detect_312_patterns_nb,
        detect_212_patterns_nb,
        detect_22_patterns_nb,
        detect_32_patterns_nb,
    )

    # Normalize columns
    df = data.copy()
    df.columns = [c.lower() for c in df.columns]

    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        return []

    # Get bar classifications
    classifications = classify_bars_nb(
        df['high'].values,
        df['low'].values
    )

    # Detect patterns based on type
    if pattern_type == '3-1-2':
        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
    elif pattern_type == '2-1-2':
        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
    elif pattern_type == '2-2':
        entries, stops, targets, directions = detect_22_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
    elif pattern_type == '3-2':
        entries, stops, targets, directions = detect_32_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
    else:
        return []

    # Collect trade details
    trades = []
    for i in range(len(entries)):
        if entries[i]:
            direction = directions[i]
            direction_str = 'Bullish' if direction == 1 else 'Bearish'

            # Calculate actual entry price based on direction
            if direction == 1:
                entry_price = df['high'].iloc[i-1] if pattern_type in ['3-1-2', '2-1-2'] else df['high'].iloc[i-1]
            else:
                entry_price = df['low'].iloc[i-1] if pattern_type in ['3-1-2', '2-1-2'] else df['low'].iloc[i-1]

            # Count continuation bars
            continuation_bars = 0
            for j in range(i+1, min(i+6, len(classifications))):
                if direction == 1 and classifications[j] == 2:
                    continuation_bars += 1
                elif direction == -1 and classifications[j] == -2:
                    continuation_bars += 1
                else:
                    break

            # Get bar classifications for context
            bar_sequence = []
            for offset in range(-3, 4):
                idx = i + offset
                if 0 <= idx < len(classifications):
                    bar_sequence.append(f"{classifications[idx]}")
                else:
                    bar_sequence.append("?")

            trades.append({
                'pattern_date': df.index[i],
                'pattern_type': pattern_type,
                'direction': direction_str,
                'entry_price': float(entry_price),
                'stop_price': float(stops[i]),
                'target_price': float(targets[i]),
                'close_at_signal': float(df['close'].iloc[i]),
                'continuation_bars': continuation_bars,
                'bar_sequence': ' -> '.join(bar_sequence),
                'risk': abs(entry_price - stops[i]),
                'reward': abs(targets[i] - entry_price),
                'bar_index': i,
            })

    return trades


def verify_pattern_timeframe_combination(
    symbol: str,
    pattern_type: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    max_examples: int = 3
) -> List[Dict]:
    """
    Verify a specific pattern/timeframe combination.

    Args:
        symbol: Stock symbol
        pattern_type: Pattern type ('3-1-2', '2-1-2', '2-2', '3-2')
        timeframe: Timeframe ('1H', '1D', '1W', '1M')
        start_date: Start date
        end_date: End date
        max_examples: Maximum number of examples to return

    Returns:
        List of trade examples
    """
    print(f"\n{'='*60}")
    print(f"Verifying {pattern_type} on {symbol} @ {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")

    # Fetch data
    data = fetch_data_for_verification(symbol, timeframe, start_date, end_date)

    if data.empty:
        print(f"  No data available")
        return []

    print(f"  Data points: {len(data)}")

    # Detect patterns
    trades = detect_patterns_for_verification(data, pattern_type, include_all=True)

    print(f"  Total patterns detected: {len(trades)}")

    if not trades:
        print(f"  No {pattern_type} patterns found")
        return []

    # Filter for most recent examples
    examples = trades[-max_examples:] if len(trades) > max_examples else trades

    # Print trade details for visual verification
    for i, trade in enumerate(examples, 1):
        rr = trade['reward'] / trade['risk'] if trade['risk'] > 0 else 0

        print(f"\n  Trade #{i}:")
        print(f"    Date:        {trade['pattern_date']}")
        print(f"    Direction:   {trade['direction']}")
        print(f"    Entry:       ${trade['entry_price']:.2f}")
        print(f"    Stop:        ${trade['stop_price']:.2f}")
        print(f"    Target:      ${trade['target_price']:.2f}")
        print(f"    Risk:        ${trade['risk']:.2f}")
        print(f"    Reward:      ${trade['reward']:.2f}")
        print(f"    R:R:         {rr:.2f}")
        print(f"    Cont. Bars:  {trade['continuation_bars']}")
        print(f"    Bar Seq:     {trade['bar_sequence']}")

    # Add metadata
    for trade in examples:
        trade['symbol'] = symbol
        trade['timeframe'] = timeframe

    return examples


def run_full_verification():
    """
    Run verification across all pattern types and timeframes.

    Generates trade examples for manual TradingView verification.
    """
    print("\n" + "="*80)
    print("STRAT VISUAL TRADE VERIFICATION")
    print("Session 75: Pattern Detection Validation")
    print("="*80)

    # Configuration
    symbol = 'SPY'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Timeframe-specific date ranges
    date_ranges = {
        '1H': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),   # 30 days for hourly
        '1D': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),  # 1 year for daily
        '1W': (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'), # 3 years for weekly
        '1M': (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d'), # 10 years for monthly
    }

    # Pattern types to verify
    pattern_types = ['3-1-2', '2-1-2', '2-2', '3-2']

    # Timeframes to test
    timeframes = ['1H', '1D', '1W', '1M']

    all_trades = []

    for pattern in pattern_types:
        for tf in timeframes:
            start_date = date_ranges[tf]

            trades = verify_pattern_timeframe_combination(
                symbol=symbol,
                pattern_type=pattern,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                max_examples=2  # 2 examples per pattern/timeframe
            )

            all_trades.extend(trades)

    # Save all trades to CSV
    if all_trades:
        df = pd.DataFrame(all_trades)

        # Ensure reports directory exists
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)

        output_path = reports_dir / 'visual_verification_trades.csv'
        df.to_csv(output_path, index=False)

        print(f"\n{'='*80}")
        print(f"VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total trade examples: {len(all_trades)}")
        print(f"Output saved to: {output_path}")

        # Summary by pattern type
        print(f"\nBy Pattern Type:")
        for pattern in pattern_types:
            count = len([t for t in all_trades if t['pattern_type'] == pattern])
            print(f"  {pattern}: {count} examples")

        # Summary by timeframe
        print(f"\nBy Timeframe:")
        for tf in timeframes:
            count = len([t for t in all_trades if t['timeframe'] == tf])
            print(f"  {tf}: {count} examples")

        print(f"\n{'='*80}")
        print("TRADINGVIEW VERIFICATION INSTRUCTIONS")
        print("="*80)
        print("""
1. Open TradingView and select SPY
2. For each trade in the CSV:
   a. Set the chart timeframe to match (1H, 1D, 1W, 1M)
   b. Navigate to the pattern_date
   c. Verify the bar classification sequence matches
   d. Check entry/stop/target levels are correct
   e. Note if the trade would have hit target or stop

3. Key things to verify:
   - Bar classifications (1=inside, 2/-2=directional, 3/-3=outside)
   - Entry price is at the correct bar's high/low
   - Stop price is at the correct structural level
   - Target price is calculated correctly (measured move or magnitude)
   - Continuation bars count after pattern

4. Document any discrepancies for investigation
""")

    return all_trades


def run_single_pattern_deep_dive(
    pattern_type: str = '2-2',
    timeframe: str = '1D',
    symbol: str = 'SPY',
    max_examples: int = 10
):
    """
    Deep dive into a single pattern type with more examples.

    Use this for focused verification of a specific pattern.
    """
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: {pattern_type} @ {timeframe}")
    print(f"{'='*80}")

    end_date = datetime.now().strftime('%Y-%m-%d')

    # Extended date range for more examples
    date_ranges = {
        '1H': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        '1D': (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
        '1W': (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'),
        '1M': (datetime.now() - timedelta(days=365*15)).strftime('%Y-%m-%d'),
    }

    trades = verify_pattern_timeframe_combination(
        symbol=symbol,
        pattern_type=pattern_type,
        timeframe=timeframe,
        start_date=date_ranges.get(timeframe, '2020-01-01'),
        end_date=end_date,
        max_examples=max_examples
    )

    return trades


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='STRAT Visual Trade Verification Script'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        choices=['3-1-2', '2-1-2', '2-2', '3-2', 'all'],
        default='all',
        help='Pattern type to verify (default: all)'
    )
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        choices=['1H', '1D', '1W', '1M', 'all'],
        default='all',
        help='Timeframe to verify (default: all)'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='SPY',
        help='Symbol to verify (default: SPY)'
    )
    parser.add_argument(
        '--deep-dive', '-d',
        action='store_true',
        help='Run deep dive with more examples for single pattern/timeframe'
    )
    parser.add_argument(
        '--examples', '-n',
        type=int,
        default=10,
        help='Number of examples for deep dive (default: 10)'
    )

    args = parser.parse_args()

    if args.pattern == 'all' and args.timeframe == 'all':
        # Full verification across all patterns and timeframes
        trades = run_full_verification()
    elif args.deep_dive or (args.pattern != 'all' and args.timeframe != 'all'):
        # Single pattern/timeframe deep dive
        pattern = args.pattern if args.pattern != 'all' else '2-2'
        timeframe = args.timeframe if args.timeframe != 'all' else '1D'

        trades = run_single_pattern_deep_dive(
            pattern_type=pattern,
            timeframe=timeframe,
            symbol=args.symbol,
            max_examples=args.examples
        )
    else:
        # Default to full verification
        trades = run_full_verification()
