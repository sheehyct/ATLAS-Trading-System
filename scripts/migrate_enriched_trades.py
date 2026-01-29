"""
Migrate enriched_trades.json to TradeStore (EQUITY-97)

Imports the 38 completed equity option trades from data/enriched_trades.json
into the trade analytics store for analysis.

Usage:
    uv run python scripts/migrate_enriched_trades.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trade_analytics.models import (
    EnrichedTradeRecord,
    ExcursionData,
    PatternContext,
    MarketContext,
    PositionManagement,
    AssetClass,
)
from core.trade_analytics.trade_store import TradeStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_duration_to_seconds(duration_str: str) -> int:
    """Parse duration string like '23h 48m' to seconds."""
    total_seconds = 0

    if 'd' in duration_str:
        days_part = duration_str.split('d')[0].strip()
        total_seconds += int(days_part) * 86400
        duration_str = duration_str.split('d')[1].strip()

    if 'h' in duration_str:
        hours_part = duration_str.split('h')[0].strip()
        total_seconds += int(hours_part) * 3600
        duration_str = duration_str.split('h')[1].strip()

    if 'm' in duration_str:
        mins_part = duration_str.replace('m', '').strip()
        if mins_part:
            total_seconds += int(mins_part) * 60

    return total_seconds


def convert_enriched_trade(data: dict) -> EnrichedTradeRecord:
    """Convert enriched_trades.json trade to EnrichedTradeRecord."""

    # Parse timestamps (handle timezone-aware strings)
    entry_time = None
    exit_time = None

    if data.get('entry_time'):
        try:
            entry_str = data['entry_time']
            # Remove timezone for naive datetime
            if '+' in entry_str:
                entry_str = entry_str.split('+')[0]
            entry_time = datetime.fromisoformat(entry_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse entry_time: {e}")

    if data.get('exit_time'):
        try:
            exit_str = data['exit_time']
            if '+' in exit_str:
                exit_str = exit_str.split('+')[0]
            exit_time = datetime.fromisoformat(exit_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse exit_time: {e}")

    # Map direction
    direction = data.get('direction', 'CALL')
    direction_norm = 'LONG' if direction in ['CALL', 'BUY'] else 'SHORT'

    # Build pattern context
    pattern = PatternContext(
        pattern_type=data.get('pattern_type', ''),
        timeframe=data.get('timeframe', ''),
        signal_type='COMPLETED',
        direction=direction_norm,
        tfc_score=data.get('tfc_score', 0),
        tfc_alignment=data.get('tfc_alignment', ''),
    )

    # Build market context (minimal - VIX not available in this format)
    market = MarketContext()

    # Build position management
    position = PositionManagement(
        actual_entry_price=data.get('entry_price', 0.0),
        position_size=1,  # Default 1 contract
        option_type=direction,
    )

    # Parse duration
    duration_str = data.get('duration', '0m')
    seconds_in_trade = parse_duration_to_seconds(duration_str)

    # No MFE/MAE data available from historical exports
    excursion = ExcursionData()

    # Build enriched record
    record = EnrichedTradeRecord(
        trade_id=data.get('osi_symbol', ''),
        symbol=data.get('underlying', ''),
        asset_class=AssetClass.EQUITY_OPTION.value,
        entry_time=entry_time,
        exit_time=exit_time,
        seconds_in_trade=seconds_in_trade,
        pnl=data.get('pnl', 0.0),
        pnl_pct=data.get('pnl_pct', 0.0),
        exit_price=data.get('exit_price', 0.0),
        pattern=pattern,
        market=market,
        position=position,
        excursion=excursion,
        notes=f"Migrated from enriched_trades.json - TFC: {data.get('tfc_alignment', 'N/A')}",
        tags=['historical', 'migrated'],
    )

    # Compute derived fields
    record.__post_init__()

    return record


def migrate_enriched_trades():
    """Main migration function."""
    source_path = Path("data/enriched_trades.json")
    store_path = Path("core/trade_analytics/data/equity_trades.json")

    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return 0

    # Load source data
    with open(source_path, 'r') as f:
        data = json.load(f)

    trades_list = data.get('trades', [])
    logger.info(f"Found {len(trades_list)} trades in {source_path}")

    # Initialize store
    store = TradeStore(store_path)
    existing_count = len(store.get_all_trades())
    logger.info(f"Existing trades in store: {existing_count}")

    # Convert and import
    imported = 0
    skipped = 0

    for trade_data in trades_list:
        try:
            trade_id = trade_data.get('osi_symbol', '')

            # Check for duplicates
            if store.get_trade(trade_id):
                logger.debug(f"Skipping duplicate: {trade_id}")
                skipped += 1
                continue

            record = convert_enriched_trade(trade_data)
            store.add_trade(record)
            imported += 1

            logger.info(
                f"Imported: {record.trade_id} - {record.pattern.pattern_type} "
                f"{record.pattern.timeframe} - P&L: ${record.pnl:.2f}"
            )

        except Exception as e:
            logger.error(f"Error converting trade {trade_data.get('osi_symbol')}: {e}")

    # Summary
    final_count = len(store.get_all_trades())
    logger.info("")
    logger.info("=" * 50)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Source trades: {len(trades_list)}")
    logger.info(f"Imported: {imported}")
    logger.info(f"Skipped (duplicates): {skipped}")
    logger.info(f"Total in store: {final_count}")

    # Quick stats
    all_trades = store.get_all_trades()
    winners = [t for t in all_trades if t.is_winner]
    total_pnl = sum(t.pnl for t in all_trades)

    logger.info("")
    if all_trades:
        logger.info(f"Win Rate: {len(winners)}/{len(all_trades)} ({len(winners)/len(all_trades):.1%})")
    logger.info(f"Total P&L: ${total_pnl:.2f}")

    return imported


if __name__ == "__main__":
    migrate_enriched_trades()
