"""
Trade Store - Persistence Layer for Enriched Trade Records

Stores trade records with full MFE/MAE data for historical analysis.
Supports both JSON file storage and future database backends.

Session: Trade Analytics Implementation
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from threading import Lock
import shutil

from core.trade_analytics.models import EnrichedTradeRecord

logger = logging.getLogger(__name__)


class TradeStore:
    """
    Persistent storage for enriched trade records.
    
    Features:
    - JSON file storage with atomic writes
    - In-memory caching for fast queries
    - Automatic backup on modification
    - Query by various filters
    
    Usage:
        store = TradeStore(Path("trades/analytics_trades.json"))
        
        # Add a trade
        store.add_trade(enriched_trade)
        
        # Query trades
        daily_trades = store.get_trades(timeframe="1D")
        winners = store.get_trades(is_winner=True)
        recent = store.get_trades(after=datetime.now() - timedelta(days=7))
    """
    
    def __init__(
        self,
        store_path: Optional[Path] = None,
        auto_save: bool = True,
        backup_on_save: bool = True,
        max_backups: int = 5,
    ):
        """
        Initialize trade store.
        
        Args:
            store_path: Path to JSON storage file
            auto_save: Automatically save after modifications
            backup_on_save: Create backup before saving
            max_backups: Maximum number of backups to keep
        """
        self.store_path = store_path or Path("core/trade_analytics/data/trades.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._auto_save = auto_save
        self._backup_on_save = backup_on_save
        self._max_backups = max_backups
        self._lock = Lock()
        
        # In-memory cache
        self._trades: Dict[str, EnrichedTradeRecord] = {}
        self._dirty = False
        
        # Load existing data
        self._load()
    
    def _load(self) -> None:
        """Load trades from storage file."""
        if not self.store_path.exists():
            logger.info(f"No existing store at {self.store_path}, starting fresh")
            return
        
        try:
            with open(self.store_path, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                trades_list = data
            elif isinstance(data, dict):
                trades_list = data.get('trades', [])
            else:
                logger.error(f"Invalid store format: {type(data)}")
                return
            
            for trade_data in trades_list:
                try:
                    trade = EnrichedTradeRecord.from_dict(trade_data)
                    self._trades[trade.trade_id] = trade
                except Exception as e:
                    logger.error(f"Error loading trade: {e}")
            
            logger.info(f"Loaded {len(self._trades)} trades from {self.store_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error loading trades: {e}")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    def _save(self) -> bool:
        """Save trades to storage file."""
        if not self._dirty:
            return True
        
        try:
            # Create backup first
            if self._backup_on_save and self.store_path.exists():
                self._create_backup()
            
            # Prepare data
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'count': len(self._trades),
                'trades': [t.to_dict() for t in self._trades.values()],
            }
            
            # Atomic write via temp file
            temp_path = self.store_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Replace original with temp
            temp_path.replace(self.store_path)
            
            self._dirty = False
            logger.debug(f"Saved {len(self._trades)} trades to {self.store_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
            return False
    
    def _create_backup(self) -> None:
        """Create a backup of the current store file."""
        if not self.store_path.exists():
            return
        
        backup_dir = self.store_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{self.store_path.stem}_{timestamp}.json"
        
        shutil.copy2(self.store_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        
        # Clean old backups
        self._cleanup_backups(backup_dir)
    
    def _cleanup_backups(self, backup_dir: Path) -> None:
        """Remove old backups exceeding max_backups."""
        backups = sorted(backup_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(backups) > self._max_backups:
            old_backup = backups.pop(0)
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup}")
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    def add_trade(self, trade: EnrichedTradeRecord) -> bool:
        """
        Add or update a trade record.
        
        Args:
            trade: EnrichedTradeRecord to store
        
        Returns:
            True if successful
        """
        with self._lock:
            trade.updated_at = datetime.utcnow()
            self._trades[trade.trade_id] = trade
            self._dirty = True
            
            if self._auto_save:
                return self._save()
            return True
    
    def get_trade(self, trade_id: str) -> Optional[EnrichedTradeRecord]:
        """Get a trade by ID."""
        with self._lock:
            return self._trades.get(trade_id)
    
    def update_trade(
        self,
        trade_id: str,
        updates: Dict[str, Any]
    ) -> Optional[EnrichedTradeRecord]:
        """
        Update specific fields of a trade.
        
        Args:
            trade_id: Trade to update
            updates: Dict of field -> value updates
        
        Returns:
            Updated trade or None if not found
        """
        with self._lock:
            trade = self._trades.get(trade_id)
            if not trade:
                return None
            
            for key, value in updates.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)
            
            trade.updated_at = datetime.utcnow()
            self._dirty = True
            
            if self._auto_save:
                self._save()
            
            return trade
    
    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by ID."""
        with self._lock:
            if trade_id in self._trades:
                del self._trades[trade_id]
                self._dirty = True
                if self._auto_save:
                    self._save()
                return True
            return False
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_type: Optional[str] = None,
        is_winner: Optional[bool] = None,
        exit_reason: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        asset_class: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_filter: Optional[Callable[[EnrichedTradeRecord], bool]] = None,
    ) -> List[EnrichedTradeRecord]:
        """
        Query trades with various filters.
        
        All filters are AND'd together.
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe ("1H", "1D", etc.)
            pattern_type: Filter by pattern (e.g., "2-1-2U")
            is_winner: Filter winners (True) or losers (False)
            exit_reason: Filter by exit reason
            after: Trades after this datetime
            before: Trades before this datetime
            min_pnl: Minimum P&L
            max_pnl: Maximum P&L
            asset_class: Filter by asset class
            tags: Filter by tags (any match)
            custom_filter: Custom filter function
        
        Returns:
            List of matching trades
        """
        with self._lock:
            results = list(self._trades.values())
        
        # Apply filters
        if symbol:
            results = [t for t in results if t.symbol == symbol]
        
        if timeframe:
            results = [t for t in results if t.pattern.timeframe == timeframe]
        
        if pattern_type:
            results = [t for t in results if pattern_type in t.pattern.pattern_type]
        
        if is_winner is not None:
            results = [t for t in results if t.is_winner == is_winner]
        
        if exit_reason:
            results = [t for t in results if t.exit_reason == exit_reason]
        
        if after:
            results = [t for t in results if t.exit_time and t.exit_time >= after]
        
        if before:
            results = [t for t in results if t.exit_time and t.exit_time <= before]
        
        if min_pnl is not None:
            results = [t for t in results if t.pnl >= min_pnl]
        
        if max_pnl is not None:
            results = [t for t in results if t.pnl <= max_pnl]
        
        if asset_class:
            results = [t for t in results if t.asset_class == asset_class]
        
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]
        
        if custom_filter:
            results = [t for t in results if custom_filter(t)]
        
        return results
    
    def get_all_trades(self) -> List[EnrichedTradeRecord]:
        """Get all trades."""
        with self._lock:
            return list(self._trades.values())
    
    def get_closed_trades(self) -> List[EnrichedTradeRecord]:
        """Get only closed trades (have exit_time)."""
        return self.get_trades(
            custom_filter=lambda t: t.exit_time is not None
        )
    
    def get_recent_trades(self, days: int = 30) -> List[EnrichedTradeRecord]:
        """Get trades from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return self.get_trades(after=cutoff)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        trades = self.get_closed_trades()
        
        if not trades:
            return {'total_trades': 0, 'message': 'No closed trades'}
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        
        # Excursion stats
        avg_exit_efficiency = 0.0
        losers_went_green = 0
        if trades:
            efficiencies = [t.excursion.exit_efficiency for t in trades if t.excursion.mfe_pnl > 0]
            if efficiencies:
                avg_exit_efficiency = sum(efficiencies) / len(efficiencies)
            losers_went_green = sum(1 for t in losers if t.excursion.went_green_before_loss)
        
        return {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': gross_profit / len(winners) if winners else 0,
            'avg_loss': gross_loss / len(losers) if losers else 0,
            'avg_exit_efficiency': avg_exit_efficiency,
            'losers_went_green': losers_went_green,
            'losers_went_green_pct': losers_went_green / len(losers) * 100 if losers else 0,
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def save(self) -> bool:
        """Force save to disk."""
        with self._lock:
            self._dirty = True
            return self._save()
    
    def __len__(self) -> int:
        """Return number of trades."""
        return len(self._trades)
    
    def __contains__(self, trade_id: str) -> bool:
        """Check if trade exists."""
        return trade_id in self._trades
