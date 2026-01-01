"""
Unified STRAT Options Backtest with ThetaData

Single-pass backtest that:
1. Fetches underlying data from Alpaca (split-adjusted)
2. Detects STRAT patterns inline (no CSV dependency)
3. Queries ThetaData for real historical options prices
4. Calculates actual P&L with real bid/ask spreads

Incorporates ALL fixes:
- 83K-19: Split-adjusted prices (not dividend-adjusted)
- 83K-21: Skip trades where entry exceeds target
- 83K-55/56: Correct entry timing (pattern bar, not +1)
- 83K-64: DTE/max_holding alignment, delta range 0.45-0.65
- EQUITY-36: 1H R:R 1.0x, gap-through handling, EOD exit

Usage:
    uv run python scripts/backtest_strat_options_unified.py --symbol SPY --risk 5
    uv run python scripts/backtest_strat_options_unified.py --symbols SPY,QQQ,IWM --risk 10
    uv run python scripts/backtest_strat_options_unified.py --timeframes 1D,1W --risk 5
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

from dotenv import load_dotenv
load_dotenv()

import vectorbtpro as vbt
import pandas_market_calendars as mcal

from integrations.thetadata_client import ThetaDataRESTClient
from strat.unified_pattern_detector import (
    detect_all_patterns,
    PatternDetectionConfig,
    ALL_PATTERNS_CONFIG,
)
# Keep PatternType for backward compatibility in output mapping
from strat.tier1_detector import PatternType
from strat.bar_classifier import classify_bars

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('unified_options_backtest')

# NYSE market calendar
NYSE_CALENDAR = mcal.get_calendar('NYSE')


# =============================================================================
# CONFIGURATION - Single Source of Truth
# =============================================================================

CONFIG = {
    # Backtest period (Alpaca limit ~6 years)
    'start_date': '2019-01-01',
    'end_date': '2025-01-01',

    # Starting capital
    'starting_capital': 25000,

    # Timeframes to test
    'timeframes': ['1H', '1D', '1W', '1M'],

    # DTE by timeframe (83K-64 fix)
    'dte_by_timeframe': {
        '1H': 7,    # Increased from 3 (83K-64)
        '1D': 21,
        '1W': 35,
        '1M': 75,
    },

    # Max holding bars by timeframe (83K-64 fix - aligned with DTE)
    'max_holding_bars': {
        '1H': 28,   # 7 DTE - 3 buffer = 4 days = ~28 hourly bars
        '1D': 18,   # 21 DTE - 3 buffer = 18 trading days
        '1W': 4,    # 35 DTE - 3 buffer = ~4.5 weeks
        '1M': 2,    # 75 DTE - 3 buffer = ~2.4 months
    },

    # Target R:R by timeframe (EQUITY-36)
    'target_rr_by_timeframe': {
        '1H': 1.0,  # Reduced from 1.5x (EQUITY-36)
        '1D': 1.5,
        '1W': 1.5,
        '1M': 1.5,
    },

    # Delta targeting (83K-64 unified)
    'delta_range': (0.45, 0.65),
    'target_delta': 0.55,

    # Pattern detection (EQUITY-38: Now uses unified detector)
    # include_22_down is now True by default via ALL_PATTERNS_CONFIG

    # Entry timing (EQUITY-36)
    # 2-bar patterns can enter at 10:30, 3-bar at 11:30
    # Updated pattern names per CLAUDE.md Section 13 (full bar sequence)
    'two_bar_patterns': ['2D-2U', '2U-2D', '3-2U', '3-2D'],
    'two_bar_min_hour': 10,
    'two_bar_min_minute': 30,
    'three_bar_min_hour': 11,
    'three_bar_min_minute': 30,

    # EOD exit for 1H (EQUITY-36)
    'eod_exit_hour': 15,
    'eod_exit_minute': 55,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_alpaca_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = '1H'
) -> Optional[pd.DataFrame]:
    """
    Fetch data from Alpaca with split adjustment (NOT dividend adjusted).

    83K-19 fix: Use adjustment='split' to match ThetaData options pricing.
    """
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca credentials not found in environment")
        return None

    try:
        data = vbt.AlpacaData.pull(
            symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            adjustment='split',  # CRITICAL: Split-adjusted, not dividend-adjusted
            client_config={
                'api_key': api_key,
                'secret_key': secret_key,
                'paper': True
            },
            tz='America/New_York'
        )
        return data.get()
    except Exception as e:
        logger.error(f"Failed to fetch {symbol} from Alpaca: {e}")
        return None


def resample_data(hourly_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample hourly data to target timeframe."""
    if timeframe == '1H':
        return hourly_data

    return hourly_data.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()


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
    except Exception as e:
        # Session EQUITY-41: Log calendar lookup failures instead of silent pass
        logger.warning(f"Calendar lookup failed for {date}: {e}, using weekend-skip fallback")

    # Fallback: skip weekends
    while date.weekday() >= 5:
        if direction == 'previous':
            date = date - timedelta(days=1)
        else:
            date = date + timedelta(days=1)

    return date


def is_valid_entry_time(timestamp: pd.Timestamp, pattern_type: str, timeframe: str) -> bool:
    """
    Check if entry time is valid based on pattern type (EQUITY-36).

    2-bar patterns (2-2) can enter at 10:30
    3-bar patterns (3-1-2, 2-1-2, 3-2, 3-2-2) need 11:30
    """
    if timeframe != '1H':
        return True  # Only applies to hourly

    hour = timestamp.hour
    minute = timestamp.minute

    # Check if 2-bar pattern
    is_two_bar = any(p in pattern_type for p in CONFIG['two_bar_patterns'])

    if is_two_bar:
        min_hour = CONFIG['two_bar_min_hour']
        min_minute = CONFIG['two_bar_min_minute']
    else:
        min_hour = CONFIG['three_bar_min_hour']
        min_minute = CONFIG['three_bar_min_minute']

    if hour < min_hour:
        return False
    if hour == min_hour and minute < min_minute:
        return False

    # Session EQUITY-41: Block after-hours entries (16:00+)
    # Entries allowed until 15:59, must exit by EOD
    if hour >= 16:
        return False

    return True


def is_eod_exit_time(timestamp: pd.Timestamp, timeframe: str) -> bool:
    """Check if we should exit due to EOD (EQUITY-36 - 1H only)."""
    if timeframe != '1H':
        return False

    return (timestamp.hour >= CONFIG['eod_exit_hour'] and
            timestamp.minute >= CONFIG['eod_exit_minute'])


def detect_gap_through(
    trigger_price: float,
    bar_open: float,
    direction: int
) -> Tuple[float, bool]:
    """
    Detect gap-through entry and return actual entry price (EQUITY-36).

    Gap-through: Bar opens beyond trigger level (entry already happened).
    In this case, actual entry is at bar open, not trigger level.

    Returns:
        (actual_entry_price, is_gap_through)
    """
    if direction == 1:  # Bullish
        if bar_open > trigger_price:
            return bar_open, True
    else:  # Bearish
        if bar_open < trigger_price:
            return bar_open, True

    return trigger_price, False


def select_strike_strat(
    entry_price: float,
    target_price: float,
    stop_price: float,
    direction: int,
    available_strikes: List[float]
) -> Tuple[Optional[float], str]:
    """
    Select strike using STRAT methodology.

    STRAT Rule: Strike must be within [Entry, Target] range.
    Best Strike = Entry + 0.3 × (Target - Entry) for calls
    """
    if not available_strikes:
        return None, ''

    if direction == 1:  # Bullish - Call
        strike_range_min = entry_price
        strike_range_max = target_price
        optimal_strike = entry_price + (0.3 * (target_price - entry_price))
        strike_type = 'call'
    else:  # Bearish - Put
        strike_range_min = target_price
        strike_range_max = entry_price
        optimal_strike = entry_price - (0.3 * (entry_price - target_price))
        strike_type = 'put'

    # Filter strikes within valid range
    valid_strikes = [s for s in available_strikes
                    if strike_range_min <= s <= strike_range_max]

    if valid_strikes:
        strike = min(valid_strikes, key=lambda x: abs(x - optimal_strike))
    else:
        # ATM fallback
        strike = min(available_strikes, key=lambda x: abs(x - entry_price))

    return strike, strike_type


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_patterns_inline(
    data: pd.DataFrame,
    timeframe: str
) -> List[Dict]:
    """
    Detect STRAT patterns using unified detector (EQUITY-38).

    Returns list of pattern dictionaries with entry/stop/target.
    Patterns are returned in CHRONOLOGICAL order (critical bug fix).

    EQUITY-38 Changes:
    - Now uses unified_pattern_detector instead of Tier1Detector
    - Detects ALL 5 pattern types (was missing 3-2 and 3-2-2)
    - Patterns sorted chronologically (was grouped by type)
    - Includes 2-2 Down (2U-2D) patterns by default
    - Uses full bar sequence naming (2D-2U not "2-2 Up")
    """
    if len(data) < 10:
        return []

    # Use unified detector with all patterns enabled (EQUITY-38)
    detected = detect_all_patterns(data, config=ALL_PATTERNS_CONFIG, timeframe=timeframe)

    patterns = []
    for p in detected:
        # Validate entry time (EQUITY-36 timing rules)
        if not is_valid_entry_time(p['timestamp'], p['pattern_type'], timeframe):
            continue

        patterns.append({
            'timestamp': p['timestamp'],
            'pattern_type': p['pattern_type'],  # Full bar sequence (e.g., '2D-2U')
            'trigger_price': p['entry_price'],
            'stop_price': p['stop_price'],
            'target_price': p['target_price'],
            'direction': p['direction'],
            'continuation_bars': 0,  # Unified detector doesn't track continuation
        })

    return patterns


# =============================================================================
# OPTIONS BACKTEST
# =============================================================================

class UnifiedOptionsBacktest:
    """
    Unified options backtest with ThetaData real pricing.
    """

    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.thetadata = None
        self.results = []

    def connect_thetadata(self) -> bool:
        """Connect to ThetaData terminal."""
        self.thetadata = ThetaDataRESTClient()
        if not self.thetadata.connect():
            logger.error("Could not connect to ThetaData terminal")
            return False
        logger.info("Connected to ThetaData terminal")
        return True

    def run_backtest(
        self,
        symbols: List[str],
        timeframes: List[str],
        risk_pct: float,
        capital: float,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Run unified options backtest.

        For each symbol/timeframe:
        1. Fetch underlying from Alpaca (split-adjusted)
        2. Detect patterns inline
        3. For each pattern, get ThetaData options prices
        4. Calculate P&L
        """
        if not self.thetadata:
            if not self.connect_thetadata():
                return pd.DataFrame()

        self.results = []
        running_capital = capital

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*60}")

            # Fetch hourly data from Alpaca
            hourly_data = get_alpaca_data(
                symbol,
                self.config['start_date'],
                self.config['end_date'],
                '1H'
            )

            if hourly_data is None or len(hourly_data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            logger.info(f"Loaded {len(hourly_data)} hourly bars")

            for tf in timeframes:
                logger.info(f"\n--- Timeframe: {tf} ---")

                # Resample to target timeframe
                tf_data = resample_data(hourly_data, tf)
                logger.info(f"Resampled to {len(tf_data)} {tf} bars")

                # Detect patterns
                patterns = detect_patterns_inline(tf_data, tf)
                logger.info(f"Detected {len(patterns)} patterns")

                if not patterns:
                    continue

                # Limit patterns for testing
                if limit:
                    patterns = patterns[:limit]
                    logger.info(f"Limited to {len(patterns)} patterns for testing")

                # Process each pattern
                for idx, pattern in enumerate(patterns):
                    if idx % 10 == 0:
                        logger.info(f"Processing pattern {idx+1}/{len(patterns)}...")
                    result = self._process_pattern(
                        symbol=symbol,
                        timeframe=tf,
                        pattern=pattern,
                        tf_data=tf_data,
                        risk_pct=risk_pct,
                        running_capital=running_capital
                    )

                    if result:
                        running_capital += result['pnl']
                        result['capital_after'] = running_capital
                        self.results.append(result)

        return pd.DataFrame(self.results)

    def _process_pattern(
        self,
        symbol: str,
        timeframe: str,
        pattern: dict,
        tf_data: pd.DataFrame,
        risk_pct: float,
        running_capital: float
    ) -> Optional[dict]:
        """Process a single pattern and return trade result."""

        timestamp = pattern['timestamp']
        trigger_price = pattern['trigger_price']
        stop_price = pattern['stop_price']
        direction = pattern['direction']
        pattern_type = pattern['pattern_type']

        # Get the bar data for gap-through detection
        try:
            bar_idx = tf_data.index.get_loc(timestamp)
            if bar_idx >= len(tf_data) - 1:
                return None  # Need future data
            next_bar = tf_data.iloc[bar_idx + 1]
        except (KeyError, IndexError):
            return None

        # Gap-through detection (EQUITY-36)
        actual_entry, gap_through = detect_gap_through(
            trigger_price, next_bar['Open'], direction
        )

        # EQUITY-39: Use target from unified detector (single source of truth)
        # Target already has timeframe adjustment applied (1.0x for 1H, structural for others)
        target_price = pattern['target_price']

        # Adjust for gap-through: preserve R:R ratio when actual entry differs from trigger
        if gap_through and trigger_price != actual_entry:
            original_risk = abs(trigger_price - stop_price)
            original_reward = abs(target_price - trigger_price)
            original_rr = original_reward / original_risk if original_risk > 0 else 1.0
            actual_risk = abs(actual_entry - stop_price)
            if direction == 1:  # Bullish
                target_price = actual_entry + (actual_risk * original_rr)
            else:  # Bearish
                target_price = actual_entry - (actual_risk * original_rr)

        # EQUITY-39: Calculate effective target_rr from adjusted values
        # (This replaces the old config-based target_rr calculation)
        actual_risk_final = abs(actual_entry - stop_price)
        actual_reward_final = abs(target_price - actual_entry)
        target_rr = actual_reward_final / actual_risk_final if actual_risk_final > 0 else 0.0

        # Skip if entry exceeds target (83K-21 fix)
        if direction == 1 and actual_entry >= target_price:
            return None
        if direction == -1 and actual_entry <= target_price:
            return None

        # Get DTE and find expiration
        target_dte = self.config['dte_by_timeframe'].get(timeframe, 21)
        ts_dt = timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
        if hasattr(ts_dt, 'tzinfo') and ts_dt.tzinfo:
            ts_dt = ts_dt.replace(tzinfo=None)
        query_date = adjust_to_trading_day(ts_dt, 'previous')

        expiration = self.thetadata.find_valid_expiration(
            underlying=symbol,
            target_date=query_date,
            target_dte=target_dte,
            min_dte=7,
            max_dte=90
        )

        if expiration is None:
            return None

        # Get available strikes
        available_strikes = self.thetadata.get_strikes_cached(symbol, expiration)
        if not available_strikes:
            return None

        # Select strike using STRAT methodology
        strike, option_type = select_strike_strat(
            actual_entry, target_price, stop_price, direction, available_strikes
        )

        if strike is None:
            return None

        # Get entry quote from ThetaData
        entry_quote = self.thetadata.get_quote(
            underlying=symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            as_of=query_date
        )

        if entry_quote is None or entry_quote.mid <= 0:
            return None

        # Buy at ask (worst case)
        entry_option_price = entry_quote.ask if entry_quote.ask > 0 else entry_quote.mid

        # Calculate position size
        risk_dollars = running_capital * risk_pct
        max_loss_per_contract = entry_option_price * 100
        contracts = max(1, min(10, int(risk_dollars / max_loss_per_contract)))

        # Simulate holding period
        exit_result = self._simulate_holding(
            symbol=symbol,
            timeframe=timeframe,
            entry_idx=bar_idx + 1,  # Enter on next bar after pattern
            tf_data=tf_data,
            actual_entry=actual_entry,
            target_price=target_price,
            stop_price=stop_price,
            direction=direction,
            expiration=expiration,
            strike=strike,
            option_type=option_type
        )

        if exit_result is None:
            return None

        # Calculate P&L
        pnl_per_contract = (exit_result['exit_option_price'] - entry_option_price) * 100
        total_pnl = pnl_per_contract * contracts

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern_type': pattern_type,
            'entry_date': timestamp,
            'exit_date': exit_result['exit_date'],
            'exit_type': exit_result['exit_type'],
            'direction': 'bullish' if direction == 1 else 'bearish',
            'trigger_price': trigger_price,
            'actual_entry': actual_entry,
            'gap_through': gap_through,
            'target_price': target_price,
            'stop_price': stop_price,
            'target_rr': target_rr,
            'strike': strike,
            'option_type': option_type,
            'expiration': expiration,
            'entry_option_price': entry_option_price,
            'exit_option_price': exit_result['exit_option_price'],
            'contracts': contracts,
            'pnl_per_contract': pnl_per_contract,
            'pnl': total_pnl,
            'data_source': 'ThetaData',
        }

    def _simulate_holding(
        self,
        symbol: str,
        timeframe: str,
        entry_idx: int,
        tf_data: pd.DataFrame,
        actual_entry: float,
        target_price: float,
        stop_price: float,
        direction: int,
        expiration: datetime,
        strike: float,
        option_type: str
    ) -> Optional[dict]:
        """Simulate holding period and determine exit."""

        max_bars = self.config['max_holding_bars'].get(timeframe, 18)
        exit_type = None
        exit_date = None
        exit_underlying_price = None

        for i in range(min(max_bars, len(tf_data) - entry_idx)):
            bar_idx = entry_idx + i
            bar = tf_data.iloc[bar_idx]
            bar_time = tf_data.index[bar_idx]

            # EOD exit check for 1H (EQUITY-36)
            if is_eod_exit_time(bar_time, timeframe):
                exit_type = 'EOD_EXIT'
                exit_date = bar_time
                exit_underlying_price = bar['Close']
                break

            # Check target hit
            if direction == 1:  # Bullish
                if bar['High'] >= target_price:
                    exit_type = 'TARGET'
                    exit_date = bar_time
                    exit_underlying_price = target_price
                    break
                if bar['Low'] <= stop_price:
                    exit_type = 'STOP'
                    exit_date = bar_time
                    exit_underlying_price = stop_price
                    break
            else:  # Bearish
                if bar['Low'] <= target_price:
                    exit_type = 'TARGET'
                    exit_date = bar_time
                    exit_underlying_price = target_price
                    break
                if bar['High'] >= stop_price:
                    exit_type = 'STOP'
                    exit_date = bar_time
                    exit_underlying_price = stop_price
                    break

        # Time exit if no target/stop hit
        if exit_type is None:
            exit_type = 'TIME_EXIT'
            final_idx = min(entry_idx + max_bars - 1, len(tf_data) - 1)
            exit_date = tf_data.index[final_idx]
            exit_underlying_price = tf_data.iloc[final_idx]['Close']

        # Don't exit after expiration (handle timezone)
        exit_date_naive = exit_date.to_pydatetime().replace(tzinfo=None) if hasattr(exit_date, 'to_pydatetime') else exit_date
        expiration_naive = expiration.replace(tzinfo=None) if hasattr(expiration, 'tzinfo') and expiration.tzinfo else expiration

        if exit_date_naive > expiration_naive:
            exit_date = pd.Timestamp(expiration - timedelta(days=1))

        # Get exit quote from ThetaData
        exit_dt = exit_date.to_pydatetime() if hasattr(exit_date, 'to_pydatetime') else exit_date
        if hasattr(exit_dt, 'tzinfo') and exit_dt.tzinfo:
            exit_dt = exit_dt.replace(tzinfo=None)
        exit_query_date = adjust_to_trading_day(exit_dt, 'previous')

        exit_quote = self.thetadata.get_quote(
            underlying=symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            as_of=exit_query_date
        )

        if exit_quote is None:
            exit_option_price = 0.0  # Full loss
        else:
            # Sell at bid (worst case)
            exit_option_price = exit_quote.bid if exit_quote.bid > 0 else exit_quote.mid

        return {
            'exit_date': exit_date,
            'exit_type': exit_type,
            'exit_underlying_price': exit_underlying_price,
            'exit_option_price': exit_option_price,
        }


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(results_df: pd.DataFrame, capital: float, risk_pct: float):
    """Print comprehensive results summary."""
    if results_df.empty:
        print("No results to display")
        return

    print("\n" + "=" * 70)
    print(f"UNIFIED STRAT OPTIONS BACKTEST - {risk_pct*100:.0f}% RISK")
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
    for tf in ['1H', '1D', '1W', '1M']:
        subset = results_df[results_df['timeframe'] == tf]
        if len(subset) > 0:
            count = len(subset)
            pnl = subset['pnl'].sum()
            wr = len(subset[subset['pnl'] > 0]) / count if count > 0 else 0
            print(f"  {tf}: {count} trades, ${pnl:,.0f} P&L, {wr:.1%} win rate")

    # By exit type
    print("\n--- By Exit Type ---")
    for exit_type in ['TARGET', 'STOP', 'TIME_EXIT', 'EOD_EXIT']:
        subset = results_df[results_df['exit_type'] == exit_type]
        if len(subset) > 0:
            count = len(subset)
            pnl = subset['pnl'].sum()
            print(f"  {exit_type}: {count} trades, ${pnl:,.0f} P&L")

    # Gap-through analysis
    gap_trades = results_df[results_df['gap_through'] == True]
    if len(gap_trades) > 0:
        print(f"\n--- Gap-Through Trades ---")
        print(f"  Count: {len(gap_trades)}")
        print(f"  P&L: ${gap_trades['pnl'].sum():,.0f}")

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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified STRAT Options Backtest')
    parser.add_argument('--symbol', type=str, default=None, help='Single symbol')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ,IWM', help='Comma-separated symbols')
    parser.add_argument('--timeframes', type=str, default='1H,1D,1W,1M', help='Comma-separated timeframes')
    parser.add_argument('--risk', type=float, default=5.0, help='Risk percentage per trade')
    parser.add_argument('--capital', type=float, default=25000, help='Starting capital')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit patterns per timeframe (for testing)')

    args = parser.parse_args()

    # Parse symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]

    risk_pct = args.risk / 100.0
    capital = args.capital

    print("=" * 70)
    print("UNIFIED STRAT OPTIONS BACKTEST")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print(f"Risk per trade: {args.risk}%")
    print(f"Starting capital: ${capital:,.0f}")
    print(f"Period: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print("\nConfiguration:")
    print(f"  DTE: {CONFIG['dte_by_timeframe']}")
    print(f"  Max Holding: {CONFIG['max_holding_bars']}")
    print(f"  Target R:R: {CONFIG['target_rr_by_timeframe']}")
    print(f"  Delta Range: {CONFIG['delta_range']}")

    # Run backtest
    backtest = UnifiedOptionsBacktest()
    results_df = backtest.run_backtest(
        symbols=symbols,
        timeframes=timeframes,
        risk_pct=risk_pct,
        capital=capital,
        limit=args.limit
    )

    if results_df.empty:
        print("\nNo trades completed. Check ThetaData connection and data coverage.")
        return

    # Print summary
    print_results_summary(results_df, capital, risk_pct)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"reports/unified_backtest_{'_'.join(symbols)}_{int(args.risk)}pct.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == '__main__':
    main()
