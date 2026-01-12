#!/usr/bin/env python3
"""
Retroactive TFC Backfill for Historical Trades (EQUITY-55)

PURPOSE:
    Calculate Timeframe Continuity (TFC) for all closed trades to enable
    analysis of TFC correlation with trade outcomes.

    Existing signals have tfc_score=0 because they were detected before
    the EQUITY-54 fix. This script calculates TFC retroactively at the
    ENTRY TIME of each trade.

USAGE:
    python scripts/backfill_trade_tfc.py [--days 90] [--output data/enriched_trades.json]

OUTPUT:
    JSON file with enriched trade data and summary statistics comparing
    win rates WITH vs WITHOUT TFC >= 4 alignment.

DESIGN:
    Uses TimeframeContinuityChecker.check_flexible_continuity_at_datetime()
    to evaluate TFC AT ENTRY TIME (not current time) to avoid future data leakage.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_osi_symbol(osi_symbol: str) -> Tuple[str, str]:
    """
    Parse OSI symbol to extract underlying and option type.

    OSI Format: ROOT(1-6 chars) + YYMMDD(6) + C/P(1) + STRIKE*1000(8)
    Example: SPY241220C00600000 -> ('SPY', 'CALL')

    Args:
        osi_symbol: OCC-format option symbol

    Returns:
        Tuple of (underlying_symbol, option_type)
        option_type is 'CALL' or 'PUT'

    Raises:
        ValueError: If symbol is too short or malformed
    """
    if not osi_symbol or len(osi_symbol) < 15:
        raise ValueError(f"Invalid OSI symbol (too short): {osi_symbol}")

    underlying = osi_symbol[:-15]
    if not underlying:
        raise ValueError(f"Could not extract underlying from: {osi_symbol}")

    option_char = osi_symbol[-9]
    if option_char not in ('C', 'P'):
        raise ValueError(f"Invalid option type char '{option_char}' in: {osi_symbol}")

    option_type = 'CALL' if option_char == 'C' else 'PUT'

    return underlying, option_type


def fetch_historical_data(
    symbol: str,
    entry_time: datetime,
    timeframe: str,
    lookback_bars: int = 50
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLC data ending at entry_time.

    CRITICAL: Returns data AS OF entry_time to avoid future data leakage.
    This ensures TFC is calculated as it would have been at entry time.

    Args:
        symbol: Stock symbol (e.g., 'SPY')
        entry_time: Datetime of trade entry
        timeframe: '1H', '1D', '1W', '1M'
        lookback_bars: Number of bars to fetch

    Returns:
        DataFrame with OHLC data ending at or before entry_time,
        or None if fetch fails
    """
    import vectorbtpro as vbt

    # Timeframe to Alpaca format mapping
    tf_map = {
        '1H': '1Hour',
        '1D': '1Day',
        '1W': '1Week',
        '1M': '1Month'
    }

    if timeframe not in tf_map:
        logger.warning(f"Unsupported timeframe: {timeframe}")
        return None

    # Calculate start date based on lookback
    # Add buffer to ensure we get enough bars
    days_map = {
        '1H': lookback_bars // 7 + 60,   # ~7 bars per day + 60 day buffer
        '1D': lookback_bars + 60,
        '1W': lookback_bars * 7 + 90,
        '1M': lookback_bars * 30 + 120
    }

    days = days_map[timeframe]
    start = entry_time - timedelta(days=days)
    # Add 1 day to end to include entry day (Alpaca end is exclusive for intraday)
    end = entry_time + timedelta(days=1)

    try:
        # Setup Alpaca credentials if available
        api_key = os.environ.get('ALPACA_API_KEY', '')
        secret_key = os.environ.get('ALPACA_SECRET_KEY', '')

        if api_key and secret_key:
            vbt.AlpacaData.set_custom_settings(
                client_config=dict(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=True
                )
            )

        data = vbt.AlpacaData.pull(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            timeframe=tf_map[timeframe],
            tz='America/New_York'
        )
        df = data.get()

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return None

        # CRITICAL: Truncate to entry_time to prevent future data leakage
        # Convert entry_time to timezone-aware if needed
        et = pytz.timezone('America/New_York')
        if entry_time.tzinfo is None:
            entry_time = et.localize(entry_time)

        # Filter to only include bars at or before entry time
        df = df[df.index <= entry_time]

        if df.empty:
            logger.warning(f"No bars before entry time for {symbol} {timeframe}")
            return None

        return df

    except Exception as e:
        logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
        return None


def evaluate_historical_tfc(
    symbol: str,
    entry_time: datetime,
    direction: str,
    detection_timeframe: str = '1D'
) -> Dict[str, Any]:
    """
    Evaluate TFC at historical datetime.

    CRITICAL: This evaluates TFC as it was AT THE TIME OF ENTRY,
    not current TFC. This is essential for accurate historical analysis.

    Args:
        symbol: Underlying symbol
        entry_time: Trade entry datetime
        direction: 'bullish' (CALL) or 'bearish' (PUT)
        detection_timeframe: Assumed detection timeframe (default '1D')

    Returns:
        Dict with strength, passes_flexible, aligned_timeframes, etc.
    """
    from strat.timeframe_continuity import TimeframeContinuityChecker

    # TFC timeframes to check (skip 4H for simplicity - matches scanner behavior)
    timeframes = ['1M', '1W', '1D', '1H']

    high_dict = {}
    low_dict = {}
    open_dict = {}
    close_dict = {}

    for tf in timeframes:
        df = fetch_historical_data(symbol, entry_time, tf)
        if df is not None and not df.empty:
            high_dict[tf] = df['High']
            low_dict[tf] = df['Low']
            open_dict[tf] = df['Open']
            close_dict[tf] = df['Close']

    if not high_dict:
        logger.warning(f"No data available for TFC evaluation: {symbol}")
        return {
            'strength': 0,
            'passes_flexible': False,
            'aligned_timeframes': [],
            'direction': direction,
            'required_timeframes': [],
            'error': 'No data available'
        }

    checker = TimeframeContinuityChecker(timeframes=timeframes)

    # Convert entry_time to pandas Timestamp
    et = pytz.timezone('America/New_York')
    if entry_time.tzinfo is None:
        entry_time = et.localize(entry_time)
    target_dt = pd.Timestamp(entry_time)

    try:
        result = checker.check_flexible_continuity_at_datetime(
            high_dict=high_dict,
            low_dict=low_dict,
            target_datetime=target_dt,
            direction=direction,
            detection_timeframe=detection_timeframe,
            open_dict=open_dict,
            close_dict=close_dict
        )
        return result
    except Exception as e:
        logger.warning(f"TFC evaluation failed for {symbol}: {e}")
        return {
            'strength': 0,
            'passes_flexible': False,
            'aligned_timeframes': [],
            'direction': direction,
            'required_timeframes': [],
            'error': str(e)
        }


def calculate_summary_stats(trades: List[Dict]) -> Dict[str, Any]:
    """
    Calculate summary statistics comparing TFC >= 4 vs < 4 performance.

    Args:
        trades: List of enriched trade dictionaries

    Returns:
        Dict with summary statistics
    """
    total_trades = len(trades)

    # Filter trades with valid TFC scores
    trades_with_tfc = [t for t in trades if t.get('tfc_score') is not None]

    # Split by TFC threshold (>= 4 is considered "WITH TFC")
    with_tfc = [t for t in trades_with_tfc if t.get('tfc_score', 0) >= 4]
    without_tfc = [t for t in trades_with_tfc if t.get('tfc_score', 0) < 4]

    def calc_stats(trade_list: List[Dict]) -> Dict:
        if not trade_list:
            return {'count': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                    'total_pnl': 0.0, 'avg_pnl': 0.0}

        wins = sum(1 for t in trade_list if (t.get('pnl') or 0) > 0)
        losses = sum(1 for t in trade_list if (t.get('pnl') or 0) <= 0)
        total_pnl = sum(t.get('pnl') or 0 for t in trade_list)
        avg_pnl = total_pnl / len(trade_list) if trade_list else 0

        return {
            'count': len(trade_list),
            'wins': wins,
            'losses': losses,
            'win_rate': round((wins / len(trade_list)) * 100, 1) if trade_list else 0.0,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2)
        }

    # Calculate pattern breakdown
    pattern_breakdown = {}
    for trade in trades:
        pattern = trade.get('pattern_type') or 'Unknown'
        if pattern not in pattern_breakdown:
            pattern_breakdown[pattern] = []
        pattern_breakdown[pattern].append(trade)

    pattern_stats = {}
    for pattern, pattern_trades in pattern_breakdown.items():
        stats = calc_stats(pattern_trades)
        pattern_stats[pattern] = stats

    # Calculate overall stats
    overall = calc_stats(trades)
    with_tfc_stats = calc_stats(with_tfc)
    without_tfc_stats = calc_stats(without_tfc)

    return {
        'total_trades': total_trades,
        'trades_with_valid_tfc': len(trades_with_tfc),
        'with_tfc_4plus': with_tfc_stats['count'],
        'without_tfc_4plus': without_tfc_stats['count'],
        'win_rate_overall': overall['win_rate'],
        'win_rate_with_tfc_4plus': with_tfc_stats['win_rate'],
        'win_rate_without_tfc_4plus': without_tfc_stats['win_rate'],
        'avg_pnl_overall': overall['avg_pnl'],
        'avg_pnl_with_tfc_4plus': with_tfc_stats['avg_pnl'],
        'avg_pnl_without_tfc_4plus': without_tfc_stats['avg_pnl'],
        'total_pnl_with_tfc_4plus': with_tfc_stats['total_pnl'],
        'total_pnl_without_tfc_4plus': without_tfc_stats['total_pnl'],
        'pattern_breakdown': pattern_stats
    }


def backfill_all_trades(days: int = 90) -> Dict[str, Any]:
    """
    Main backfill function - processes all closed trades.

    Args:
        days: How many days back to look for trades

    Returns:
        Dict with trades list, summary statistics, and errors
    """
    from integrations.alpaca_trading_client import AlpacaTradingClient
    from strat.signal_automation.signal_store import SignalStore

    logger.info(f"Starting TFC backfill for trades in last {days} days")

    # Initialize Alpaca client
    # Try SMALL account first, fall back to LARGE
    client = None
    for account in ['SMALL', 'LARGE']:
        try:
            client = AlpacaTradingClient(account=account)
            if client.connect():
                logger.info(f"Connected to Alpaca {account} account")
                break
        except Exception as e:
            logger.warning(f"Could not connect to {account} account: {e}")
            client = None

    if client is None or not getattr(client, 'connected', False):
        raise RuntimeError("Could not connect to any Alpaca account")

    # Load signal store for pattern lookup
    signal_store = SignalStore()
    logger.info(f"Loaded signal store with {len(signal_store._signals)} signals")

    # Get closed trades (use timezone-aware datetime per CLAUDE.md)
    et = pytz.timezone('America/New_York')
    after = datetime.now(et) - timedelta(days=days)
    closed_trades = client.get_closed_trades(after=after, options_only=True)
    logger.info(f"Found {len(closed_trades)} closed trades")

    enriched = []
    errors = []

    for i, trade in enumerate(closed_trades):
        osi_symbol = trade.get('symbol', '')
        entry_time = trade.get('buy_time_dt')

        logger.info(f"Processing trade {i+1}/{len(closed_trades)}: {osi_symbol}")

        if not entry_time:
            errors.append({
                'osi_symbol': osi_symbol,
                'error': 'Missing entry time'
            })
            continue

        try:
            # Parse OSI symbol
            underlying, option_type = parse_osi_symbol(osi_symbol)
            direction = 'bullish' if option_type == 'CALL' else 'bearish'

            # Look up signal for pattern info (may be None)
            signal = signal_store.get_signal_by_osi_symbol(osi_symbol)
            pattern_type = signal.pattern_type if signal else None
            timeframe = signal.timeframe if signal else '1D'  # Default assumption

            logger.info(f"  Underlying: {underlying}, Direction: {direction}, "
                       f"Pattern: {pattern_type or 'Unknown'}, TF: {timeframe}")

            # Evaluate historical TFC
            tfc_result = evaluate_historical_tfc(
                symbol=underlying,
                entry_time=entry_time,
                direction=direction,
                detection_timeframe=timeframe
            )

            tfc_score = tfc_result.get('strength', 0)
            aligned_tfs = tfc_result.get('aligned_timeframes', [])
            passes_flexible = tfc_result.get('passes_flexible', False)

            # Session EQUITY-56: Detection timeframe is aligned BY DEFINITION at entry
            # Entry triggers when price breaks in expected direction intrabar
            # Historical data truncated to entry_time may not capture intrabar state
            # (bar may show as Type 1 if entry was mid-bar before bar close)
            # Since entry occurred, the bar HAS broken in expected direction
            if timeframe not in aligned_tfs:
                aligned_tfs = aligned_tfs.copy()  # Don't mutate original
                aligned_tfs.append(timeframe)
                tfc_score = len(aligned_tfs)
                # Recalculate passes_flexible with corrected score
                # Using TFC >= 4 as threshold (full timeframe continuity)
                passes_flexible = tfc_score >= 4
                logger.debug(f"  Added detection TF {timeframe} to aligned list (entry = aligned)")

            logger.info(f"  TFC: {tfc_score}/4, Aligned: {aligned_tfs}, Passes: {passes_flexible}")

            # Build enriched trade record
            enriched.append({
                'osi_symbol': osi_symbol,
                'underlying': underlying,
                'direction': option_type,
                'entry_time': entry_time.isoformat() if entry_time else None,
                'exit_time': trade.get('sell_time_dt').isoformat() if trade.get('sell_time_dt') else None,
                'entry_price': trade.get('buy_price'),
                'exit_price': trade.get('sell_price'),
                'pnl': trade.get('realized_pnl'),
                'pnl_pct': trade.get('roi_percent'),
                'pattern_type': pattern_type,
                'timeframe': timeframe,
                'tfc_score': tfc_score,
                'tfc_alignment': ', '.join(aligned_tfs) if aligned_tfs else '',
                'tfc_passes': passes_flexible,
                'duration': trade.get('duration', '')
            })

        except Exception as e:
            logger.error(f"Error processing {osi_symbol}: {e}")
            errors.append({
                'osi_symbol': osi_symbol,
                'error': str(e)
            })

    # Calculate summary statistics
    summary = calculate_summary_stats(enriched)

    logger.info(f"Backfill complete: {len(enriched)} trades enriched, {len(errors)} errors")
    logger.info(f"TFC >= 4: {summary['with_tfc_4plus']} trades, "
               f"win rate {summary['win_rate_with_tfc_4plus']}%")
    logger.info(f"TFC < 4:  {summary['without_tfc_4plus']} trades, "
               f"win rate {summary['win_rate_without_tfc_4plus']}%")

    return {
        'generated_at': datetime.now().isoformat(),
        'version': '1.0',
        'backfill_params': {
            'days_back': days,
            'detection_timeframe_default': '1D',
            'tfc_threshold': 4
        },
        'trades': enriched,
        'summary': summary,
        'errors': errors
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Retroactive TFC Backfill for Historical Trades (EQUITY-55)'
    )
    parser.add_argument(
        '--days', type=int, default=90,
        help='Number of days to look back for trades (default: 90)'
    )
    parser.add_argument(
        '--output', type=str, default='data/enriched_trades.json',
        help='Output file path (default: data/enriched_trades.json)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run backfill
    result = backfill_all_trades(days=args.days)

    # Ensure output directory exists
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Output written to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("TFC BACKFILL SUMMARY")
    print("="*60)
    summary = result['summary']
    print(f"Total trades processed: {summary['total_trades']}")
    print(f"Trades with valid TFC:  {summary['trades_with_valid_tfc']}")
    print()
    print(f"WITH TFC >= 4:          {summary['with_tfc_4plus']} trades")
    print(f"  Win rate:             {summary['win_rate_with_tfc_4plus']}%")
    print(f"  Avg P&L:              ${summary['avg_pnl_with_tfc_4plus']:.2f}")
    print(f"  Total P&L:            ${summary['total_pnl_with_tfc_4plus']:.2f}")
    print()
    print(f"WITHOUT TFC >= 4:       {summary['without_tfc_4plus']} trades")
    print(f"  Win rate:             {summary['win_rate_without_tfc_4plus']}%")
    print(f"  Avg P&L:              ${summary['avg_pnl_without_tfc_4plus']:.2f}")
    print(f"  Total P&L:            ${summary['total_pnl_without_tfc_4plus']:.2f}")
    print()
    if result['errors']:
        print(f"Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:  # Show first 5 errors
            print(f"  - {err['osi_symbol']}: {err['error']}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
