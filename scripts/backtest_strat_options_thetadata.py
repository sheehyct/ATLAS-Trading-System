"""
True Options Backtest Using ThetaData Historical Pricing.

This script performs an accurate options backtest by:
1. Loading STRAT pattern signals from validation CSV files
2. Using ThetaData for REAL historical options prices (entry and exit)
3. Calculating actual P&L based on real bid/ask spreads

Unlike the simplified equity-based backtest, this produces:
- Accurate profit factors that vary by risk level
- Real theta decay impact
- Actual bid-ask spread costs
- Professional-grade results suitable for quantitative analysis

Requirements:
- ThetaData Terminal running at localhost:25503
- ThetaData subscription with historical options data

Usage:
    uv run python scripts/backtest_strat_options_thetadata.py --symbol SPY --risk 2
    uv run python scripts/backtest_strat_options_thetadata.py --symbol SPY --risk 5
    uv run python scripts/backtest_strat_options_thetadata.py --symbol SPY --risk 10

Session 83K-84: Initial implementation for true options backtest.
Session 84: Fixed 472 errors by validating strikes/expirations before querying quotes.
            Added ATM fallback when OTM strikes unavailable.
            Limited to 7 years of data (2018-2025).
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas_market_calendars as mcal

from integrations.thetadata_client import ThetaDataRESTClient
from integrations.tiingo_data_fetcher import TiingoDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('options_backtest')

# NYSE market calendar for trading day validation
NYSE_CALENDAR = mcal.get_calendar('NYSE')


# Configuration
CONFIG = {
    'starting_capital': 25000,
    'timeframes': ['1D', '1W', '1M'],
    'max_holding_bars': {'1D': 18, '1W': 4, '1M': 2},
    'output_dir': 'reports',
    'min_date': '2018-01-01',  # Session 84: 7 years of data (ThetaData coverage)
}

# DTE settings by timeframe
DTE_BY_TIMEFRAME = {
    '1H': 7,
    '1D': 21,
    '1W': 35,
    '1M': 75,
}


# Session 84: Progress tracking for diagnostics
class BacktestProgress:
    """Track reasons for skipped trades and other diagnostics."""

    def __init__(self):
        self.total_patterns = 0
        self.processed = 0
        self.skipped_no_expiration = 0
        self.skipped_no_strikes = 0
        self.skipped_no_entry_quote = 0
        self.skipped_no_exit_quote = 0
        self.atm_fallback_used = 0
        self.successful_trades = 0

    def summary(self) -> str:
        """Return summary string."""
        lines = [
            "--- Backtest Progress ---",
            f"Total patterns: {self.total_patterns}",
            f"Processed: {self.processed}",
            f"Successful trades: {self.successful_trades}",
            f"ATM fallback used: {self.atm_fallback_used}",
            f"Skipped - no expiration: {self.skipped_no_expiration}",
            f"Skipped - no strikes: {self.skipped_no_strikes}",
            f"Skipped - no entry quote: {self.skipped_no_entry_quote}",
            f"Skipped - no exit quote: {self.skipped_no_exit_quote}",
        ]
        return "\n".join(lines)


def adjust_to_trading_day(date: datetime, direction: str = 'previous') -> datetime:
    """Adjust a date to a valid trading day."""
    start = date - timedelta(days=10)
    end = date + timedelta(days=10)

    try:
        schedule = NYSE_CALENDAR.schedule(start_date=start, end_date=end)
        trading_days = schedule.index.to_pydatetime()

        if direction == 'previous':
            valid_days = [d for d in trading_days if d.date() <= date.date()]
            if valid_days:
                return valid_days[-1]
        else:
            valid_days = [d for d in trading_days if d.date() >= date.date()]
            if valid_days:
                return valid_days[0]
    except Exception:
        pass

    # Fallback: skip weekends
    while date.weekday() >= 5:
        if direction == 'previous':
            date = date - timedelta(days=1)
        else:
            date = date + timedelta(days=1)

    return date


def get_next_expiration(from_date: datetime, target_dte: int) -> datetime:
    """Get the next monthly expiration with approximately target DTE."""
    # Find third Friday of the target month
    target_date = from_date + timedelta(days=target_dte)

    # Find third Friday
    year = target_date.year
    month = target_date.month

    # First day of month
    first_day = datetime(year, month, 1)
    # Find first Friday (weekday 4)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    # Third Friday
    third_friday = first_friday + timedelta(weeks=2)

    # If third Friday is before our target, try next month
    if third_friday < from_date + timedelta(days=7):
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)

    return third_friday


def select_strike_strat(
    entry_price: float,
    target_price: float,
    stop_price: float,
    direction: int,
    available_strikes: List[float] = None
) -> Tuple[float, str]:
    """
    Select strike using STRAT methodology.

    STRAT Rule: Strike must be within [Entry, Target] range, NOT based on delta.
    Best Strike = Entry + 0.3 × (Target - Entry) for calls
    Best Strike = Entry - 0.3 × (Entry - Target) for puts

    Session 84: Fixed from incorrect delta-based selection to proper STRAT methodology.

    Args:
        entry_price: Entry price from pattern
        target_price: Target price from pattern
        stop_price: Stop price from pattern
        direction: 1 for bullish (call), -1 for bearish (put)
        available_strikes: Optional list of available strikes to choose from

    Returns:
        Tuple of (strike, strike_type)
    """
    if direction == 1:  # Bullish - Call
        # STRAT: Strike within [Entry, Target] range
        strike_range_min = entry_price
        strike_range_max = target_price

        # Optimal strike at 30% from entry toward target
        optimal_strike = entry_price + (0.3 * (target_price - entry_price))
        strike_type = 'call'
    else:  # Bearish - Put
        # STRAT: Strike within [Target, Entry] range
        strike_range_min = target_price
        strike_range_max = entry_price

        # Optimal strike at 30% from entry toward target
        optimal_strike = entry_price - (0.3 * (entry_price - target_price))
        strike_type = 'put'

    # If we have available strikes, find the closest valid one
    if available_strikes:
        # Filter strikes within the valid range
        valid_strikes = [s for s in available_strikes
                        if strike_range_min <= s <= strike_range_max]

        if valid_strikes:
            # Find closest to optimal
            strike = min(valid_strikes, key=lambda x: abs(x - optimal_strike))
        else:
            # If no strikes in range, find closest to entry (ATM)
            strike = min(available_strikes, key=lambda x: abs(x - entry_price))
    else:
        # Round to nearest dollar if no available strikes provided
        strike = round(optimal_strike)

    return strike, strike_type


def select_strike(
    underlying_price: float,
    direction: int,
    delta_target: float = 0.30
) -> float:
    """
    DEPRECATED: Use select_strike_strat() instead.

    This function uses an incorrect formula that results in strikes 6-7% OTM.
    Kept for backwards compatibility.
    """
    if direction == 1:  # Bullish - OTM call
        offset = underlying_price * (1 - delta_target) * 0.1
        strike = round(underlying_price + offset)
    else:  # Bearish - OTM put
        offset = underlying_price * (1 - delta_target) * 0.1
        strike = round(underlying_price - offset)

    return strike


def load_pattern_data(
    timeframes: List[str],
    symbols: List[str] = None,
    scripts_dir: str = 'scripts'
) -> pd.DataFrame:
    """Load pattern data from validation CSV files."""
    all_patterns = []

    for tf in timeframes:
        filepath = Path(scripts_dir) / f'strat_validation_{tf}.csv'
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        df = pd.read_csv(filepath)
        df['entry_date'] = pd.to_datetime(df['entry_date'], utc=True).dt.tz_localize(None)
        df['timeframe'] = tf

        if symbols:
            df = df[df['symbol'].isin(symbols)]

        all_patterns.append(df)
        logger.info(f"Loaded {len(df)} patterns from {tf}")

    if not all_patterns:
        raise ValueError("No pattern data found")

    combined = pd.concat(all_patterns, ignore_index=True)
    combined = combined.sort_values('entry_date')

    return combined


def run_options_backtest(
    patterns_df: pd.DataFrame,
    symbol: str,
    risk_pct: float,
    capital: float,
    thetadata_client: ThetaDataRESTClient,
    price_data: pd.DataFrame
) -> Tuple[pd.DataFrame, BacktestProgress]:
    """
    Run options backtest using ThetaData for real pricing.

    Session 84: Updated to validate strikes/expirations before querying quotes.

    For each pattern:
    1. Find valid expiration from ThetaData
    2. Get available strikes and select best one (with ATM fallback)
    3. Get entry price from ThetaData
    4. Simulate holding period based on pattern outcome
    5. Get exit price from ThetaData
    6. Calculate P&L

    Returns:
        Tuple of (results DataFrame, progress tracker)
    """
    results = []
    running_capital = capital
    progress = BacktestProgress()

    progress.total_patterns = len(patterns_df)

    # Clear client cache for fresh run
    thetadata_client.clear_cache()

    for idx, row in patterns_df.iterrows():
        progress.processed += 1

        entry_date = row['entry_date']
        entry_price = row['entry_price']
        stop_price = row['stop_price']
        target_price = row['target_price']
        pattern_type = row['pattern_type']
        timeframe = row['timeframe']

        # Determine outcome from magnitude_hit and stop_hit columns
        magnitude_hit = row.get('magnitude_hit', False)
        stop_hit = row.get('stop_hit', False)
        bars_to_magnitude = row.get('bars_to_magnitude', 999)
        bars_to_stop = row.get('bars_to_stop', 999)

        if magnitude_hit and (not stop_hit or bars_to_magnitude < bars_to_stop):
            outcome = 'target_hit'
        elif stop_hit and (not magnitude_hit or bars_to_stop < bars_to_magnitude):
            outcome = 'stop_hit'
        else:
            outcome = 'timeout'

        # Determine direction from pattern type or direction column
        direction = 1 if 'Up' in pattern_type or '2U' in pattern_type else -1
        # Note: option_type will be set by select_strike_strat()

        # Get target DTE
        target_dte = DTE_BY_TIMEFRAME.get(timeframe, 21)

        # Adjust entry date to trading day
        query_date = adjust_to_trading_day(entry_date, 'previous')

        # ----- Session 84: VALIDATED EXPIRATION SELECTION -----
        # Use ThetaData to find a valid expiration instead of calculating third Friday
        expiration = thetadata_client.find_valid_expiration(
            underlying=symbol,
            target_date=query_date,
            target_dte=target_dte,
            min_dte=7,
            max_dte=90
        )

        if expiration is None:
            progress.skipped_no_expiration += 1
            continue

        # ----- Session 84: STRAT METHODOLOGY STRIKE SELECTION -----
        # STRAT Rule: Strike must be within [Entry, Target] range
        # Best Strike = Entry + 0.3 × (Target - Entry) for calls
        # Get available strikes from ThetaData for this expiration
        available_strikes = thetadata_client.get_strikes_cached(symbol, expiration)

        if not available_strikes:
            progress.skipped_no_strikes += 1
            continue

        # Use STRAT methodology for strike selection
        strike, option_type = select_strike_strat(
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            direction=direction,
            available_strikes=available_strikes
        )

        if strike is None:
            progress.skipped_no_strikes += 1
            continue

        # Track if we used ATM fallback (strike outside entry-target range)
        if direction == 1:  # Call
            in_range = entry_price <= strike <= target_price
        else:  # Put
            in_range = target_price <= strike <= entry_price

        if not in_range:
            progress.atm_fallback_used += 1

        # ----- Get entry option price from ThetaData -----
        entry_quote = thetadata_client.get_quote(
            underlying=symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            as_of=query_date
        )

        if entry_quote is None or entry_quote.mid <= 0:
            progress.skipped_no_entry_quote += 1
            continue

        # Buy at ask (worst case)
        entry_option_price = entry_quote.ask if entry_quote.ask > 0 else entry_quote.mid

        # Calculate position size
        risk_dollars = running_capital * risk_pct
        max_loss_per_contract = entry_option_price * 100  # Max loss = full premium
        contracts = max(1, min(10, int(risk_dollars / max_loss_per_contract)))

        # Determine exit date based on outcome
        max_holding = CONFIG['max_holding_bars'].get(timeframe, 18)

        if outcome == 'target_hit':
            exit_days = max_holding // 2
        elif outcome == 'stop_hit':
            exit_days = max_holding // 4
        else:  # timeout
            exit_days = max_holding

        # Calculate exit date
        exit_date = entry_date + timedelta(days=exit_days)
        if timeframe == '1W':
            exit_date = entry_date + timedelta(weeks=exit_days)
        elif timeframe == '1M':
            exit_date = entry_date + timedelta(weeks=exit_days * 4)

        # Don't exit after expiration
        if exit_date > expiration:
            exit_date = expiration - timedelta(days=1)

        # Adjust exit date to trading day
        exit_query_date = adjust_to_trading_day(exit_date, 'previous')

        # Get exit option price from ThetaData
        exit_quote = thetadata_client.get_quote(
            underlying=symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            as_of=exit_query_date
        )

        if exit_quote is None or exit_quote.mid <= 0:
            # Still count as successful but with 0 exit price (full loss)
            exit_option_price = 0.0
        else:
            # Sell at bid (worst case)
            exit_option_price = exit_quote.bid if exit_quote.bid > 0 else exit_quote.mid

        # Calculate P&L
        pnl_per_contract = (exit_option_price - entry_option_price) * 100
        total_pnl = pnl_per_contract * contracts

        # Update running capital
        running_capital += total_pnl

        progress.successful_trades += 1

        results.append({
            'entry_date': entry_date,
            'exit_date': exit_query_date,
            'pattern_type': pattern_type,
            'timeframe': timeframe,
            'outcome': outcome,
            'direction': 'bullish' if direction == 1 else 'bearish',
            'underlying_entry': entry_price,
            'underlying_target': target_price,
            'underlying_stop': stop_price,
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type,
            'entry_option_price': entry_option_price,
            'exit_option_price': exit_option_price,
            'contracts': contracts,
            'pnl_per_contract': pnl_per_contract,
            'pnl': total_pnl,
            'capital_after': running_capital,
            'data_source': 'ThetaData',
            'used_atm_fallback': not in_range,  # True if strike outside [Entry, Target] range
        })

        # Progress logging every 50 trades
        if len(results) % 50 == 0:
            logger.info(f"Processed {len(results)} trades...")

    logger.info(f"\n{progress.summary()}")

    return pd.DataFrame(results), progress


def print_results_summary(results_df: pd.DataFrame, capital: float, risk_pct: float):
    """Print summary of backtest results."""
    if results_df.empty:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print(f"STRAT OPTIONS BACKTEST RESULTS - {risk_pct*100:.0f}% RISK (ThetaData)")
    print("=" * 70)

    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['pnl'] > 0])
    losing_trades = len(results_df[results_df['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_pnl = results_df['pnl'].sum()
    final_capital = capital + total_pnl
    total_return = (final_capital - capital) / capital

    print(f"\nStarting Capital: ${capital:,.0f}")
    print(f"Final Capital:    ${final_capital:,.0f}")
    print(f"Total P&L:        ${total_pnl:,.0f} ({total_return:.1%})")

    print(f"\nTotal Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.1%} ({winning_trades}W / {losing_trades}L)")

    winners = results_df[results_df['pnl'] > 0]['pnl']
    losers = results_df[results_df['pnl'] < 0]['pnl']

    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0

    print(f"\nAverage Win:      ${avg_win:,.0f}")
    print(f"Average Loss:     ${avg_loss:,.0f}")

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    print(f"Profit Factor:    {profit_factor:.2f}")

    # By timeframe
    print("\n--- By Timeframe ---")
    for tf in ['1D', '1W', '1M']:
        subset = results_df[results_df['timeframe'] == tf]
        if len(subset) > 0:
            count = len(subset)
            pnl = subset['pnl'].sum()
            wr = len(subset[subset['pnl'] > 0]) / count if count > 0 else 0
            print(f"  {tf}: {count} trades, ${pnl:,.0f} P&L, {wr:.1%} win rate")

    # By pattern type
    print("\n--- By Pattern Type ---")
    pattern_summary = results_df.groupby('pattern_type').agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).mean()]
    }).round(2)
    pattern_summary.columns = ['trades', 'pnl', 'win_rate']
    pattern_summary = pattern_summary.sort_values('pnl', ascending=False)

    for pattern, data in pattern_summary.head(10).iterrows():
        print(f"  {pattern}: {int(data['trades'])} trades, ${data['pnl']:,.0f} P&L, {data['win_rate']:.1%} WR")

    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='True Options Backtest using ThetaData')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to backtest')
    parser.add_argument('--risk', type=float, default=2.0, help='Risk percentage per trade')
    parser.add_argument('--capital', type=float, default=25000, help='Starting capital')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')

    args = parser.parse_args()

    risk_pct = args.risk / 100.0
    symbol = args.symbol.upper()
    capital = args.capital

    print("=" * 70)
    print("STRAT OPTIONS BACKTEST - THETADATA REAL PRICING")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Risk per trade: {args.risk}%")
    print(f"Starting capital: ${capital:,.0f}")
    print(f"Date filter: {CONFIG['min_date']} onwards (7 years)")

    # Step 1: Connect to ThetaData
    print("\n[1/5] Connecting to ThetaData...")
    thetadata = ThetaDataRESTClient()
    if not thetadata.connect():
        print("  ERROR: Could not connect to ThetaData terminal")
        print("  Make sure Theta Terminal is running at localhost:25503")
        return

    print("  Connected to ThetaData terminal")

    # Step 2: Load pattern data
    print("\n[2/5] Loading pattern data...")
    patterns_df = load_pattern_data(
        timeframes=CONFIG['timeframes'],
        symbols=[symbol]
    )
    original_count = len(patterns_df)
    print(f"  Loaded {original_count} patterns")
    print(f"  Date range: {patterns_df['entry_date'].min().date()} to {patterns_df['entry_date'].max().date()}")

    # Session 84: Filter to 7 years (2018-2025) for ThetaData coverage
    print("\n[3/5] Filtering to ThetaData coverage window...")
    min_date = pd.to_datetime(CONFIG['min_date'])
    patterns_df = patterns_df[patterns_df['entry_date'] >= min_date]
    filtered_count = len(patterns_df)
    print(f"  Filtered {original_count - filtered_count} patterns before {CONFIG['min_date']}")
    print(f"  Remaining: {filtered_count} patterns")
    print(f"  Date range: {patterns_df['entry_date'].min().date()} to {patterns_df['entry_date'].max().date()}")

    if filtered_count == 0:
        print("  ERROR: No patterns after date filter")
        return

    # Step 4: Load underlying price data
    print("\n[4/5] Loading underlying price data...")
    fetcher = TiingoDataFetcher()

    start_date = patterns_df['entry_date'].min() - timedelta(days=30)
    end_date = patterns_df['entry_date'].max() + timedelta(days=60)

    price_data = fetcher.fetch(
        symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        timeframe='1D'
    ).get()

    print(f"  Loaded {len(price_data)} price bars")

    # Step 5: Run backtest
    print("\n[5/5] Running options backtest with ThetaData pricing...")
    print("  (Using validated strikes/expirations - may take a few minutes)")
    results_df, progress = run_options_backtest(
        patterns_df=patterns_df,
        symbol=symbol,
        risk_pct=risk_pct,
        capital=capital,
        thetadata_client=thetadata,
        price_data=price_data
    )

    print(f"\n  Completed {len(results_df)} trades with real options pricing")
    success_rate = (progress.successful_trades / progress.total_patterns * 100) if progress.total_patterns > 0 else 0
    print(f"  ThetaData success rate: {success_rate:.1f}%")

    # Print progress summary
    print(f"\n{progress.summary()}")

    # Print results summary
    if not results_df.empty:
        print_results_summary(results_df, capital, risk_pct)

        # Save to CSV
        if args.output:
            output_path = args.output
        else:
            output_path = f"reports/options_backtest_{symbol}_{int(args.risk)}pct_thetadata.csv"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo trades completed - check ThetaData connection and data coverage")

    # Disconnect
    thetadata.disconnect()

    return results_df


if __name__ == '__main__':
    main()
