"""
Trade Converter - Import Existing Trades to Enriched Format

Converts existing paper_trades.json and crypto SimulatedTrade objects
to the new EnrichedTradeRecord format for analytics.

Session: Trade Analytics Implementation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.trade_analytics.models import (
    EnrichedTradeRecord,
    ExcursionData,
    PatternContext,
    MarketContext,
    PositionManagement,
    AssetClass,
    ExitReason,
)
from core.trade_analytics.trade_store import TradeStore

logger = logging.getLogger(__name__)


class TradeConverter:
    """
    Convert existing trade formats to EnrichedTradeRecord.
    
    Supports:
    - Equity options paper_trades.json format
    - Crypto SimulatedTrade format
    - Position monitor TrackedPosition format
    """
    
    @staticmethod
    def from_paper_trade(
        data: Dict[str, Any],
        asset_class: str = AssetClass.EQUITY_OPTION.value,
    ) -> EnrichedTradeRecord:
        """
        Convert paper_trades.json record to EnrichedTradeRecord.
        
        Args:
            data: Single trade dict from paper_trades.json
            asset_class: Asset class override
        
        Returns:
            EnrichedTradeRecord
        """
        # Parse timestamps
        entry_time = None
        exit_time = None
        pattern_detected_time = None
        
        if data.get('entry_time'):
            try:
                entry_time = datetime.fromisoformat(data['entry_time'])
            except (ValueError, TypeError):
                pass
        
        if data.get('exit_time'):
            try:
                exit_time = datetime.fromisoformat(data['exit_time'])
            except (ValueError, TypeError):
                pass
        
        if data.get('pattern_detected_time'):
            try:
                pattern_detected_time = datetime.fromisoformat(data['pattern_detected_time'])
            except (ValueError, TypeError):
                pass
        
        # Map direction to standardized format
        direction = data.get('direction', 'LONG')
        if direction in ['CALL', 'BUY']:
            direction = 'LONG'
        elif direction in ['PUT', 'SELL']:
            direction = 'SHORT'
        
        # Map exit reason
        exit_reason_str = data.get('exit_reason', '')
        exit_reason = ExitReason.UNKNOWN.value
        if exit_reason_str:
            try:
                exit_reason = ExitReason(exit_reason_str).value
            except ValueError:
                exit_reason = exit_reason_str
        
        # Build pattern context
        pattern = PatternContext(
            pattern_type=data.get('pattern_type', ''),
            timeframe=data.get('timeframe', ''),
            signal_type='COMPLETED' if data.get('status') == 'CLOSED' else 'SETUP',
            direction=direction,
            magnitude_pct=data.get('magnitude_pct', 0.0),
            tfc_score=data.get('tfc_score', 0),
            tfc_alignment=data.get('tfc_alignment', ''),
        )
        
        # Build market context
        market = MarketContext(
            vix_level=data.get('vix_at_entry', 0.0),
            vix_regime=TradeConverter._vix_to_regime(data.get('vix_at_entry', 0)),
            atr_14=data.get('atr_14', 0.0),
            atr_percent=data.get('atr_percent', 0.0),
            volume_ratio=data.get('volume_ratio', 0.0),
            market_regime=data.get('market_regime', ''),
        )
        
        # Build position management context
        position = PositionManagement(
            entry_trigger=data.get('entry_trigger', 0.0),
            stop_price=data.get('stop_price', 0.0),
            target_price=data.get('target_price', 0.0),
            risk_reward_planned=data.get('risk_reward', 0.0),
            actual_entry_price=data.get('entry_price', 0.0),
            distance_to_target_pct=data.get('distance_to_target', 0.0),
            position_size=data.get('contracts', 1),
            option_type=data.get('direction'),  # CALL/PUT
            strike=data.get('strike'),
            dte_at_entry=data.get('dte_at_entry'),
            delta_at_entry=data.get('delta_at_entry'),
            delta_at_exit=data.get('delta_at_exit'),
            iv_at_entry=data.get('iv_at_entry'),
            iv_at_exit=data.get('iv_at_exit'),
        )
        
        # NOTE: Excursion data not available from paper_trades.json
        # This will need to be populated from position monitor going forward
        excursion = ExcursionData()
        
        # Build enriched record
        record = EnrichedTradeRecord(
            trade_id=data.get('trade_id', ''),
            symbol=data.get('symbol', ''),
            asset_class=asset_class,
            entry_time=entry_time,
            exit_time=exit_time,
            pattern_detected_time=pattern_detected_time,
            pnl=data.get('pnl_dollars', 0.0),
            pnl_pct=data.get('pnl_percent', 0.0),
            exit_reason=exit_reason,
            exit_price=data.get('exit_price', 0.0),
            pattern=pattern,
            market=market,
            position=position,
            excursion=excursion,
            notes=data.get('notes', ''),
        )
        
        # Compute derived fields
        record.__post_init__()
        
        return record
    
    @staticmethod
    def from_crypto_simulated_trade(
        trade: Any,  # SimulatedTrade from crypto paper_trader
    ) -> EnrichedTradeRecord:
        """
        Convert crypto SimulatedTrade to EnrichedTradeRecord.
        
        Args:
            trade: SimulatedTrade object
        
        Returns:
            EnrichedTradeRecord
        """
        # Map direction
        direction = 'LONG' if trade.side == 'BUY' else 'SHORT'
        
        # Map exit reason
        exit_reason = ExitReason.UNKNOWN.value
        if trade.exit_reason:
            try:
                exit_reason = ExitReason(trade.exit_reason).value
            except ValueError:
                exit_reason = trade.exit_reason
        
        # Build pattern context
        pattern = PatternContext(
            pattern_type=getattr(trade, 'pattern_type', ''),
            timeframe=getattr(trade, 'timeframe', ''),
            signal_type='COMPLETED' if trade.status == 'CLOSED' else 'SETUP',
            direction=direction,
            tfc_score=getattr(trade, 'tfc_score', 0),
        )
        
        # Build position management
        position = PositionManagement(
            entry_trigger=trade.entry_price,
            stop_price=getattr(trade, 'stop_price', 0.0) or 0.0,
            target_price=getattr(trade, 'target_price', 0.0) or 0.0,
            actual_entry_price=trade.entry_price,
            position_size=trade.quantity,
        )
        
        # Build excursion from intrabar tracking if available
        excursion = ExcursionData()
        if hasattr(trade, 'intrabar_high') and hasattr(trade, 'intrabar_low'):
            # Calculate MFE/MAE from intrabar extremes
            if direction == 'LONG':
                mfe_price = trade.intrabar_high
                mae_price = trade.intrabar_low if trade.intrabar_low != float('inf') else trade.entry_price
            else:
                mfe_price = trade.intrabar_low if trade.intrabar_low != float('inf') else trade.entry_price
                mae_price = trade.intrabar_high
            
            excursion.mfe_price = mfe_price
            excursion.mae_price = mae_price
            
            # Calculate P&L at extremes
            if direction == 'LONG':
                excursion.mfe_pnl = (mfe_price - trade.entry_price) * trade.quantity
                excursion.mae_pnl = (mae_price - trade.entry_price) * trade.quantity
            else:
                excursion.mfe_pnl = (trade.entry_price - mfe_price) * trade.quantity
                excursion.mae_pnl = (trade.entry_price - mae_price) * trade.quantity
        
        # Build enriched record
        record = EnrichedTradeRecord(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            asset_class=AssetClass.CRYPTO_PERP.value,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            pnl=trade.pnl or 0.0,
            pnl_pct=trade.pnl_percent or 0.0,
            gross_pnl=getattr(trade, 'gross_pnl', trade.pnl) or 0.0,
            total_costs=getattr(trade, 'total_costs', 0.0),
            exit_reason=exit_reason,
            exit_price=trade.exit_price or 0.0,
            pattern=pattern,
            position=position,
            excursion=excursion,
        )
        
        # Compute derived fields and finalize excursion
        record.__post_init__()
        if trade.exit_price:
            record.finalize_excursion()
        
        return record
    
    @staticmethod
    def from_equity_tracked_position(
        pos: Any,  # TrackedPosition from equity position_monitor
        excursion_data: Optional[ExcursionData] = None,
    ) -> EnrichedTradeRecord:
        """
        Convert equity TrackedPosition to EnrichedTradeRecord.
        
        Args:
            pos: TrackedPosition object
            excursion_data: Optional ExcursionData from tracker
        
        Returns:
            EnrichedTradeRecord
        """
        # Map direction
        direction = 'LONG' if pos.direction.upper() in ['CALL', 'LONG', 'BUY'] else 'SHORT'
        
        # Map exit reason
        exit_reason = ExitReason.UNKNOWN.value
        if pos.exit_reason:
            try:
                exit_reason = ExitReason(pos.exit_reason).value
            except ValueError:
                exit_reason = pos.exit_reason
        
        # Build pattern context
        pattern = PatternContext(
            pattern_type=pos.pattern_type,
            timeframe=pos.timeframe,
            signal_type='COMPLETED',
            direction=direction,
            entry_bar_type=getattr(pos, 'entry_bar_type', ''),
            entry_bar_high=getattr(pos, 'entry_bar_high', 0.0),
            entry_bar_low=getattr(pos, 'entry_bar_low', 0.0),
        )
        
        # Build position management
        position = PositionManagement(
            entry_trigger=pos.entry_trigger,
            stop_price=pos.stop_price,
            target_price=pos.original_target if hasattr(pos, 'original_target') else pos.target_price,
            actual_entry_price=getattr(pos, 'actual_entry_underlying', pos.entry_price),
            position_size=pos.contracts,
            option_type=pos.direction,  # CALL/PUT
            trailing_stop_activated=getattr(pos, 'trailing_stop_active', False),
            high_water_mark=getattr(pos, 'high_water_mark', 0.0),
            trailing_stop_final=getattr(pos, 'trailing_stop_price', 0.0),
        )
        
        # Use provided excursion data or build from position
        if excursion_data is None:
            excursion = ExcursionData()
            if hasattr(pos, 'intrabar_high') and hasattr(pos, 'intrabar_low'):
                excursion.mfe_price = pos.intrabar_high
                excursion.mae_price = pos.intrabar_low if pos.intrabar_low != float('inf') else 0.0
        else:
            excursion = excursion_data
        
        # Build enriched record
        record = EnrichedTradeRecord(
            trade_id=pos.signal_key or pos.osi_symbol,
            symbol=pos.symbol,
            asset_class=AssetClass.EQUITY_OPTION.value,
            entry_time=pos.entry_time,
            exit_time=pos.exit_time,
            pnl=pos.realized_pnl or pos.unrealized_pnl or 0.0,
            pnl_pct=pos.unrealized_pct * 100 if hasattr(pos, 'unrealized_pct') else 0.0,
            exit_reason=exit_reason,
            exit_price=pos.exit_price or 0.0,
            pattern=pattern,
            position=position,
            excursion=excursion,
        )
        
        record.__post_init__()
        if pos.exit_price:
            record.finalize_excursion()
        
        return record
    
    @staticmethod
    def _vix_to_regime(vix: float) -> str:
        """Map VIX level to regime string."""
        if vix < 15:
            return "LOW"
        elif vix < 20:
            return "NORMAL"
        elif vix < 25:
            return "ELEVATED"
        elif vix < 30:
            return "HIGH"
        else:
            return "EXTREME"


def import_paper_trades(
    source_path: Path,
    store: TradeStore,
    asset_class: str = AssetClass.EQUITY_OPTION.value,
) -> int:
    """
    Import existing paper_trades.json into TradeStore.
    
    Args:
        source_path: Path to paper_trades.json
        store: TradeStore to import into
        asset_class: Asset class for imported trades
    
    Returns:
        Number of trades imported
    """
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return 0
    
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            trades_list = data
        elif isinstance(data, dict):
            trades_list = data.get('trades', [])
        else:
            logger.error(f"Invalid format in {source_path}")
            return 0
        
        imported = 0
        for trade_data in trades_list:
            try:
                # Only import closed trades
                if trade_data.get('status') != 'CLOSED':
                    continue
                
                record = TradeConverter.from_paper_trade(trade_data, asset_class)
                store.add_trade(record)
                imported += 1
                
            except Exception as e:
                logger.error(f"Error converting trade: {e}")
        
        logger.info(f"Imported {imported} trades from {source_path}")
        return imported
        
    except Exception as e:
        logger.error(f"Error importing trades: {e}")
        return 0
