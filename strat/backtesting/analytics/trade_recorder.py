"""
Trade Recorder - Produces EnrichedTradeRecord

Converts closed SimulatedPosition objects into EnrichedTradeRecord
instances (from core/trade_analytics/models.py) for downstream
analysis and comparison with live trade logs.
"""

import logging
from datetime import datetime
from typing import List

from core.trade_analytics.models import (
    EnrichedTradeRecord,
    PatternContext,
    MarketContext,
    PositionManagement,
    ExcursionData,
    AssetClass,
)
from strat.backtesting.simulation.position_tracker import SimulatedPosition

logger = logging.getLogger(__name__)


class TradeRecorder:
    """
    Records closed trades as EnrichedTradeRecord instances.

    Converts SimulatedPosition -> EnrichedTradeRecord, preserving
    all context needed for factor-based analysis.

    Usage:
        recorder = TradeRecorder()
        recorder.record(closed_position)
        records = recorder.get_records()
    """

    def __init__(self):
        self._records: List[EnrichedTradeRecord] = []

    def record(self, pos: SimulatedPosition) -> EnrichedTradeRecord:
        """
        Convert a closed SimulatedPosition to an EnrichedTradeRecord.

        Args:
            pos: Closed SimulatedPosition

        Returns:
            EnrichedTradeRecord with all context populated
        """
        # Calculate timing
        seconds_in_trade = 0
        if pos.entry_time and pos.exit_time:
            seconds_in_trade = int((pos.exit_time - pos.entry_time).total_seconds())

        # Build pattern context
        pattern = PatternContext(
            pattern_type=pos.pattern_type,
            timeframe=pos.timeframe,
            direction=pos.direction,
            setup_bar_high=pos.entry_bar_high,
            setup_bar_low=pos.entry_bar_low,
            entry_bar_type=pos.entry_bar_type,
            entry_bar_high=pos.entry_bar_high,
            entry_bar_low=pos.entry_bar_low,
        )

        # Build position management context
        position_mgmt = PositionManagement(
            entry_trigger=pos.entry_trigger,
            stop_price=pos.stop_price,
            target_price=pos.original_target,
            actual_entry_price=pos.actual_entry_underlying,
            actual_stop_used=pos.stop_price,
            actual_target_used=pos.target_price,
            position_size=float(pos.contracts),
            notional_value=pos.cost_basis,
            option_type=pos.direction,
            strike=pos.strike,
            trailing_stop_activated=pos.trailing_stop_active,
            high_water_mark=pos.high_water_mark,
            trailing_stop_final=pos.trailing_stop_price,
        )

        # Calculate P&L
        pnl = pos.realized_pnl or 0.0
        pnl_pct = 0.0
        if pos.cost_basis > 0:
            pnl_pct = pnl / pos.cost_basis * 100

        # Build the record
        record = EnrichedTradeRecord(
            trade_id=pos.signal_key,
            symbol=pos.symbol,
            asset_class=AssetClass.EQUITY_OPTION.value,
            entry_time=pos.entry_time,
            exit_time=pos.exit_time,
            bars_in_trade=pos.bars_held,
            seconds_in_trade=seconds_in_trade,
            pnl=pnl,
            pnl_pct=pnl_pct,
            gross_pnl=pnl,  # No slippage model in backtest
            exit_reason=pos.exit_reason.value if pos.exit_reason else 'UNKNOWN',
            exit_price=pos.exit_price or 0.0,
            pattern=pattern,
            position=position_mgmt,
            tags=[f"bt:{pos.timeframe}", f"bt:{pos.pattern_type}"],
        )

        self._records.append(record)
        return record

    def get_records(self) -> List[EnrichedTradeRecord]:
        """Return all recorded trades."""
        return list(self._records)

    def clear(self) -> None:
        """Clear all recorded trades."""
        self._records.clear()
