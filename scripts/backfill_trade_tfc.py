#!/usr/bin/env python3
"""
Retroactive TFC Backfill for Historical Trades (EQUITY-55, EQUITY-103)

PURPOSE:
    Calculate Timeframe Continuity (TFC) for all closed trades to enable
    analysis of TFC correlation with trade outcomes.

    EQUITY-55: Original backfill with retroactive TFC calculation.
    EQUITY-103: Added secondary signal matching, smart TFC (reuse signal store
    TFC when available), trade_metadata.json output, --dry-run/--write-metadata.

USAGE:
    # Dry run to verify signal matching quality
    python scripts/backfill_trade_tfc.py --days 45 --dry-run

    # Full run with metadata output
    python scripts/backfill_trade_tfc.py --days 45 --write-metadata

OUTPUT:
    - data/enriched_trades.json: Enriched trade data with TFC and summary stats
    - data/executions/trade_metadata.json: Pattern/TFC metadata keyed by OSI symbol
      (merged with existing entries, used by /trade_metadata API)

DESIGN:
    Smart TFC: Reuses signal_store TFC when tfc_score > 0 (no API calls).
    Falls back to retroactive calculation via AlpacaData only when needed.
    Secondary signal matching finds signals even without executed_osi_symbol set.
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


def _normalize_tz(dt: datetime, reference: datetime) -> datetime:
    """Align timezone awareness of dt to match reference for safe comparison."""
    if dt.tzinfo is None and reference.tzinfo is not None:
        return pytz.UTC.localize(dt)
    if dt.tzinfo is not None and reference.tzinfo is None:
        return dt.replace(tzinfo=None)
    return dt


# Statuses eligible for Tier 2 detected_time matching
_TIER2_ELIGIBLE_STATUSES = ('TRIGGERED', 'HISTORICAL_TRIGGERED', 'ALERTED', 'DETECTED')

# Time windows for Tier 2 matching (seconds)
_TRIGGERED_WINDOW_SECS = 3600    # 60 minutes
_DETECTED_WINDOW_SECS = 86400    # 24 hours


def match_signal_to_trade(
    osi_symbol: str,
    underlying: str,
    option_type: str,
    buy_time: datetime,
    signal_store,
) -> Optional[Any]:
    """
    Match a closed trade to its originating signal using two-tier strategy.

    EQUITY-103: Many historical trades lack executed_osi_symbol on the signal,
    so direct OSI lookup fails. This adds fuzzy matching as a fallback.

    Tier 1: Direct O(1) lookup by executed_osi_symbol (fast path)
    Tier 2: Search all signals matching (underlying + direction + time proximity)

    Args:
        osi_symbol: OCC option symbol of the trade
        underlying: Underlying symbol (e.g., 'SPY')
        option_type: 'CALL' or 'PUT'
        buy_time: Trade entry datetime
        signal_store: SignalStore instance

    Returns:
        StoredSignal if matched, None otherwise
    """
    # Tier 1: Direct OSI symbol lookup (O(1))
    signal = signal_store.get_signal_by_osi_symbol(osi_symbol)
    if signal:
        logger.debug(f"  Tier 1 match (OSI lookup): {signal.signal_key}")
        return signal

    # Tier 2: Fuzzy matching by underlying + direction + time proximity
    all_signals = signal_store.load_signals()
    expected_direction = 'bullish' if option_type == 'CALL' else 'bearish'
    candidates = []

    for sig in all_signals.values():
        if sig.symbol != underlying or sig.direction != expected_direction:
            continue

        # Window A: triggered_at within 60 min (tight, for live triggers)
        if sig.triggered_at:
            triggered_dt = _normalize_tz(sig.triggered_at, buy_time)
            delta = abs((triggered_dt - buy_time).total_seconds())
            if delta < _TRIGGERED_WINDOW_SECS:
                candidates.append((sig, delta, 'triggered'))
                continue

        # Window B: detected_time within 24h for eligible statuses
        if sig.detected_time and sig.status in _TIER2_ELIGIBLE_STATUSES:
            detected_dt = _normalize_tz(sig.detected_time, buy_time)
            delta = abs((detected_dt - buy_time).total_seconds())
            if delta < _DETECTED_WINDOW_SECS:
                candidates.append((sig, delta, 'detected'))

    if not candidates:
        return None

    # Pick closest match by time
    candidates.sort(key=lambda x: x[1])
    best_signal, best_delta, match_type = candidates[0]
    logger.debug(
        f"  Tier 2 match ({match_type}, delta={best_delta:.0f}s): "
        f"{best_signal.signal_key}"
    )
    return best_signal


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


def write_trade_metadata(
    enriched_trades: List[Dict],
    matched_signals: Dict[str, Any],
) -> Dict[str, Dict]:
    """
    Write trade_metadata.json for the /trade_metadata API endpoint.

    EQUITY-103: Merges backfilled metadata with existing trade_metadata.json
    so we don't overwrite entries already there from the executor.

    Args:
        enriched_trades: List of enriched trade dicts from backfill
        matched_signals: Dict mapping osi_symbol -> StoredSignal (or None)

    Returns:
        The final merged metadata dict
    """
    metadata_file = PROJECT_ROOT / 'data' / 'executions' / 'trade_metadata.json'

    # Load existing metadata (don't overwrite executor-written entries)
    existing = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                existing = json.load(f)
            logger.info(f"Loaded {len(existing)} existing entries from trade_metadata.json")
        except Exception as e:
            logger.warning(f"Could not load existing trade_metadata.json: {e}")

    new_count = 0
    for trade in enriched_trades:
        osi_symbol = trade['osi_symbol']

        # Don't overwrite existing entries (executor data is authoritative)
        if osi_symbol in existing:
            continue

        signal = matched_signals.get(osi_symbol)

        entry = {
            'pattern_type': trade.get('pattern_type'),
            'timeframe': trade.get('timeframe', '1D'),
            'tfc_score': trade.get('tfc_score'),
            'tfc_alignment': trade.get('tfc_alignment', ''),
            'direction': trade.get('direction'),
            'symbol': trade.get('underlying'),
            'backfilled_at': datetime.now().isoformat(),
        }

        # Attach signal-specific fields when a matched signal exists
        if signal:
            entry.update({
                'entry_trigger': signal.entry_trigger,
                'stop_price': signal.stop_price,
                'target_price': signal.target_price,
                'magnitude_pct': signal.magnitude_pct,
                'risk_reward': signal.risk_reward,
                'signal_key': signal.signal_key,
            })

        existing[osi_symbol] = entry
        new_count += 1

    # Ensure directory exists and write
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Wrote trade_metadata.json: {new_count} new + {len(existing) - new_count} existing = {len(existing)} total")
    return existing


def backfill_all_trades(
    days: int = 45,
    dry_run: bool = False,
    write_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Main backfill function - processes all closed trades.

    EQUITY-103: Enhanced with secondary signal matching, smart TFC strategy,
    and trade_metadata.json output.

    Args:
        days: How many days back to look for trades
        dry_run: If True, print matches without writing files
        write_metadata: If True, also write trade_metadata.json

    Returns:
        Dict with trades list, summary statistics, and errors
    """
    from integrations.alpaca_trading_client import AlpacaTradingClient
    from strat.signal_automation.signal_store import SignalStore

    logger.info(f"Starting TFC backfill for trades in last {days} days")
    if dry_run:
        logger.info("DRY RUN MODE - no files will be written")

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

    # Count signals with executed_osi_symbol for diagnostics
    osi_count = sum(1 for s in signal_store._signals.values() if s.executed_osi_symbol)
    logger.info(f"  Signals with executed_osi_symbol: {osi_count}")

    # Get closed trades (use timezone-aware datetime per CLAUDE.md)
    et = pytz.timezone('America/New_York')
    after = datetime.now(et) - timedelta(days=days)
    closed_trades = client.get_closed_trades(after=after, options_only=True)
    logger.info(f"Found {len(closed_trades)} closed trades")

    enriched = []
    errors = []
    matched_signals: Dict[str, Any] = {}  # osi_symbol -> StoredSignal or None

    # Tracking for match quality reporting
    tier1_matches = 0
    tier2_matches = 0
    no_matches = 0
    smart_tfc_reused = 0
    retroactive_tfc_calculated = 0

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

            # EQUITY-103: Two-tier signal matching
            signal = match_signal_to_trade(
                osi_symbol=osi_symbol,
                underlying=underlying,
                option_type=option_type,
                buy_time=entry_time,
                signal_store=signal_store,
            )
            matched_signals[osi_symbol] = signal

            # Track match tier for diagnostics
            if not signal:
                no_matches += 1
                match_label = "None"
            elif signal.executed_osi_symbol == osi_symbol:
                tier1_matches += 1
                match_label = "Tier1"
            else:
                tier2_matches += 1
                match_label = "Tier2"

            pattern_type = signal.pattern_type if signal else None
            timeframe = signal.timeframe if signal else '1D'
            logger.info(
                f"  Underlying: {underlying}, Direction: {direction}, "
                f"Pattern: {pattern_type or 'Unknown'}, TF: {timeframe}, "
                f"Match: {match_label}"
            )

            # EQUITY-103: Smart TFC strategy
            # If matched signal already has valid TFC, reuse it (no API calls)
            tfc_score = 0
            aligned_tfs = []
            passes_flexible = False

            if signal and signal.tfc_score:
                # Smart path: reuse signal store TFC
                tfc_score = signal.tfc_score
                aligned_tfs = [tf.strip() for tf in signal.tfc_alignment.split(',') if tf.strip()]
                passes_flexible = tfc_score >= 4
                smart_tfc_reused += 1
                logger.info(f"  Smart TFC: reusing signal store (score={tfc_score})")
            elif not dry_run:
                # Fallback: calculate retroactively (requires API calls)
                retroactive_tfc_calculated += 1
                logger.info(f"  Retroactive TFC: calculating via API...")
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
                if timeframe not in aligned_tfs:
                    aligned_tfs = aligned_tfs.copy()
                    aligned_tfs.append(timeframe)
                    tfc_score = len(aligned_tfs)
                    passes_flexible = tfc_score >= 4
                    logger.debug(f"  Added detection TF {timeframe} to aligned list (entry = aligned)")
            else:
                logger.info(f"  Dry run: would calculate retroactive TFC")

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
                'duration': trade.get('duration', ''),
                'signal_match_tier': match_label,
            })

        except Exception as e:
            logger.error(f"Error processing {osi_symbol}: {e}")
            errors.append({
                'osi_symbol': osi_symbol,
                'error': str(e)
            })

    # Calculate summary statistics
    summary = calculate_summary_stats(enriched)

    # EQUITY-103: Add match quality stats to summary
    total_processed = tier1_matches + tier2_matches + no_matches
    match_rate = ((tier1_matches + tier2_matches) / total_processed * 100) if total_processed > 0 else 0
    summary['signal_matching'] = {
        'tier1_osi_matches': tier1_matches,
        'tier2_fuzzy_matches': tier2_matches,
        'no_match': no_matches,
        'match_rate_pct': round(match_rate, 1),
        'smart_tfc_reused': smart_tfc_reused,
        'retroactive_tfc_calculated': retroactive_tfc_calculated,
    }

    logger.info(f"Backfill complete: {len(enriched)} trades enriched, {len(errors)} errors")
    logger.info(f"Signal matching: {tier1_matches} Tier1 + {tier2_matches} Tier2 = "
                f"{tier1_matches + tier2_matches}/{total_processed} ({match_rate:.0f}%)")
    logger.info(f"Smart TFC reused: {smart_tfc_reused}, Retroactive: {retroactive_tfc_calculated}")
    logger.info(f"TFC >= 4: {summary['with_tfc_4plus']} trades, "
                f"win rate {summary['win_rate_with_tfc_4plus']}%")
    logger.info(f"TFC < 4:  {summary['without_tfc_4plus']} trades, "
                f"win rate {summary['win_rate_without_tfc_4plus']}%")

    # EQUITY-103: Write trade_metadata.json if requested
    if write_metadata and not dry_run:
        write_trade_metadata(enriched, matched_signals)

    return {
        'generated_at': datetime.now().isoformat(),
        'version': '2.0',
        'backfill_params': {
            'days_back': days,
            'detection_timeframe_default': '1D',
            'tfc_threshold': 4,
            'dry_run': dry_run,
            'write_metadata': write_metadata,
        },
        'trades': enriched,
        'summary': summary,
        'errors': errors
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Retroactive TFC Backfill for Historical Trades (EQUITY-55/103)'
    )
    parser.add_argument(
        '--days', type=int, default=45,
        help='Number of days to look back for trades (default: 45)'
    )
    parser.add_argument(
        '--output', type=str, default='data/enriched_trades.json',
        help='Output file path (default: data/enriched_trades.json)'
    )
    parser.add_argument(
        '--write-metadata', action='store_true',
        help='Also write data/executions/trade_metadata.json'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print signal matches without writing files or making API calls'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run backfill
    result = backfill_all_trades(
        days=args.days,
        dry_run=args.dry_run,
        write_metadata=args.write_metadata,
    )

    # Write enriched_trades.json (unless dry run)
    if not args.dry_run:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Output written to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("TFC BACKFILL SUMMARY" + (" (DRY RUN)" if args.dry_run else ""))
    print("="*60)
    summary = result['summary']
    print(f"Total trades processed: {summary['total_trades']}")
    print(f"Trades with valid TFC:  {summary['trades_with_valid_tfc']}")
    print()

    # EQUITY-103: Signal matching quality
    matching = summary.get('signal_matching', {})
    print("SIGNAL MATCHING:")
    print(f"  Tier 1 (OSI lookup):  {matching.get('tier1_osi_matches', 0)}")
    print(f"  Tier 2 (fuzzy):       {matching.get('tier2_fuzzy_matches', 0)}")
    print(f"  No match:             {matching.get('no_match', 0)}")
    print(f"  Match rate:           {matching.get('match_rate_pct', 0)}%")
    print(f"  Smart TFC reused:     {matching.get('smart_tfc_reused', 0)}")
    print(f"  Retroactive TFC:      {matching.get('retroactive_tfc_calculated', 0)}")
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
