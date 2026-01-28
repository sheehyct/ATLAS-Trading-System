"""
Excursion Tracker - MFE/MAE Tracking During Trade Lifecycle

Tracks Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
for open positions in real-time. This is THE KEY DATA needed to answer:

- "Did this loser go green first?" (MFE > 0 for losers)
- "Am I leaving profit on table?" (exit_pnl << MFE for winners)
- "Are my stops too tight?" (MAE hits stop before MFE reached)

Usage:
    tracker = ExcursionTracker()
    
    # On position open
    tracker.start_tracking(trade_id, entry_price, direction)
    
    # On each price update (position monitor loop)
    tracker.update(trade_id, current_price, timestamp)
    
    # On position close
    excursion_data = tracker.finalize(trade_id, exit_price)

Session: Trade Analytics Implementation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Any
from threading import Lock

from core.trade_analytics.models import ExcursionData

logger = logging.getLogger(__name__)


@dataclass
class TrackingState:
    """Internal state for tracking a single position's excursion."""
    trade_id: str
    entry_price: float
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    
    # Position sizing for P&L calculation
    quantity: float = 1.0
    multiplier: float = 100.0  # 100 for options (100 shares per contract)
    
    # Running extremes
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    highest_pnl: float = 0.0  # Best P&L seen (MFE candidate)
    lowest_pnl: float = 0.0   # Worst P&L seen (MAE candidate)
    
    # Timestamps for extremes
    mfe_time: Optional[datetime] = None
    mae_time: Optional[datetime] = None
    mfe_price: float = 0.0
    mae_price: float = 0.0
    
    # Bar counting (for "time to MFE/MAE" analysis)
    bar_count: int = 0
    mfe_bar: int = 0
    mae_bar: int = 0
    
    # Price history (optional, for detailed analysis)
    price_samples: List[Dict[str, Any]] = field(default_factory=list)
    sample_interval_seconds: int = 60  # Sample every 60 seconds
    last_sample_time: Optional[datetime] = None
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L at a given price."""
        if self.direction.upper() in ["LONG", "CALL", "BUY"]:
            return (current_price - self.entry_price) * self.quantity * self.multiplier
        else:
            return (self.entry_price - current_price) * self.quantity * self.multiplier
    
    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate P&L percentage at a given price."""
        if self.entry_price <= 0:
            return 0.0
        if self.direction.upper() in ["LONG", "CALL", "BUY"]:
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


class ExcursionTracker:
    """
    Tracks MFE/MAE for open positions in real-time.
    
    Thread-safe implementation for use in position monitor loops.
    
    Example integration with position monitor:
    
        class PositionMonitor:
            def __init__(self):
                self.excursion_tracker = ExcursionTracker()
            
            def on_position_open(self, trade_id, entry_price, direction, qty):
                self.excursion_tracker.start_tracking(
                    trade_id=trade_id,
                    entry_price=entry_price,
                    direction=direction,
                    quantity=qty,
                )
            
            def check_positions(self):
                for pos in self.positions:
                    current_price = self.get_price(pos.symbol)
                    # Update excursion tracking
                    self.excursion_tracker.update(
                        pos.trade_id, 
                        current_price, 
                        datetime.now()
                    )
                    # ... rest of position checks
            
            def on_position_close(self, trade_id, exit_price):
                excursion = self.excursion_tracker.finalize(trade_id, exit_price)
                # excursion now contains MFE/MAE data to store
    """
    
    def __init__(
        self,
        sample_interval_seconds: int = 60,
        max_samples_per_trade: int = 1000,
        store_price_history: bool = True,
    ):
        """
        Initialize excursion tracker.
        
        Args:
            sample_interval_seconds: How often to sample price for history
            max_samples_per_trade: Maximum price samples to store per trade
            store_price_history: Whether to store detailed price history
        """
        self._positions: Dict[str, TrackingState] = {}
        self._lock = Lock()
        self._sample_interval = sample_interval_seconds
        self._max_samples = max_samples_per_trade
        self._store_history = store_price_history
    
    def start_tracking(
        self,
        trade_id: str,
        entry_price: float,
        direction: str,
        quantity: float = 1.0,
        multiplier: float = 100.0,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """
        Start tracking excursion for a new position.
        
        Args:
            trade_id: Unique identifier for the trade
            entry_price: Entry price (underlying for options)
            direction: "LONG", "SHORT", "CALL", "PUT", "BUY", "SELL"
            quantity: Number of shares/contracts
            multiplier: Contract multiplier (100 for options, 1 for stocks)
            entry_time: Entry timestamp (defaults to now)
        """
        with self._lock:
            if trade_id in self._positions:
                logger.warning(f"Already tracking {trade_id}, replacing state")
            
            entry_time = entry_time or datetime.now()
            
            state = TrackingState(
                trade_id=trade_id,
                entry_price=entry_price,
                entry_time=entry_time,
                direction=direction,
                quantity=quantity,
                multiplier=multiplier,
                highest_price=entry_price,
                lowest_price=entry_price,
                highest_pnl=0.0,
                lowest_pnl=0.0,
                mfe_price=entry_price,
                mae_price=entry_price,
                mfe_time=entry_time,
                mae_time=entry_time,
                sample_interval_seconds=self._sample_interval,
                last_sample_time=entry_time,
            )
            
            # Add initial sample
            if self._store_history:
                state.price_samples.append({
                    'time': entry_time.isoformat(),
                    'price': entry_price,
                    'pnl': 0.0,
                    'event': 'ENTRY',
                })
            
            self._positions[trade_id] = state
            logger.debug(f"Started tracking {trade_id} @ {entry_price}")
    
    def update(
        self,
        trade_id: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
        bar_closed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Update excursion tracking with current price.
        
        Call this on each position monitor cycle (e.g., every 60 seconds).
        
        Args:
            trade_id: Trade identifier
            current_price: Current underlying price
            timestamp: Current timestamp (defaults to now)
            bar_closed: Whether a new bar just closed (for bar counting)
        
        Returns:
            Dict with current excursion state, or None if not tracking
        """
        with self._lock:
            state = self._positions.get(trade_id)
            if not state:
                return None
            
            timestamp = timestamp or datetime.now()
            current_pnl = state.calculate_pnl(current_price)
            
            # Update bar count
            if bar_closed:
                state.bar_count += 1
            
            # Track highest/lowest price
            if current_price > state.highest_price:
                state.highest_price = current_price
            if current_price < state.lowest_price:
                state.lowest_price = current_price
            
            # Check for new MFE (Maximum Favorable Excursion)
            if current_pnl > state.highest_pnl:
                state.highest_pnl = current_pnl
                state.mfe_price = current_price
                state.mfe_time = timestamp
                state.mfe_bar = state.bar_count
                logger.debug(
                    f"{trade_id} new MFE: ${current_pnl:.2f} @ {current_price:.2f}"
                )
            
            # Check for new MAE (Maximum Adverse Excursion)
            if current_pnl < state.lowest_pnl:
                state.lowest_pnl = current_pnl
                state.mae_price = current_price
                state.mae_time = timestamp
                state.mae_bar = state.bar_count
                logger.debug(
                    f"{trade_id} new MAE: ${current_pnl:.2f} @ {current_price:.2f}"
                )
            
            # Sample price history (rate-limited)
            if self._store_history and state.last_sample_time:
                elapsed = (timestamp - state.last_sample_time).total_seconds()
                if elapsed >= self._sample_interval:
                    if len(state.price_samples) < self._max_samples:
                        state.price_samples.append({
                            'time': timestamp.isoformat(),
                            'price': current_price,
                            'pnl': current_pnl,
                            'event': 'SAMPLE',
                        })
                    state.last_sample_time = timestamp
            
            return {
                'trade_id': trade_id,
                'current_pnl': current_pnl,
                'mfe_pnl': state.highest_pnl,
                'mae_pnl': state.lowest_pnl,
                'mfe_price': state.mfe_price,
                'mae_price': state.mae_price,
            }
    
    def finalize(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[ExcursionData]:
        """
        Finalize tracking and return ExcursionData for a closed trade.
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit timestamp (defaults to now)
        
        Returns:
            ExcursionData with all MFE/MAE metrics, or None if not tracking
        """
        with self._lock:
            state = self._positions.pop(trade_id, None)
            if not state:
                logger.warning(f"No tracking state for {trade_id}")
                return None
            
            exit_time = exit_time or datetime.now()
            exit_pnl = state.calculate_pnl(exit_price)
            exit_pnl_pct = state.calculate_pnl_pct(exit_price)
            
            # Add final sample
            if self._store_history:
                state.price_samples.append({
                    'time': exit_time.isoformat(),
                    'price': exit_price,
                    'pnl': exit_pnl,
                    'event': 'EXIT',
                })
            
            # Calculate MFE/MAE percentages
            mfe_pct = state.calculate_pnl_pct(state.mfe_price)
            mae_pct = state.calculate_pnl_pct(state.mae_price)
            
            # Calculate exit efficiency
            exit_efficiency = 0.0
            profit_captured_pct = 0.0
            if state.highest_pnl > 0:
                exit_efficiency = exit_pnl / state.highest_pnl
                profit_captured_pct = min(100.0, max(0.0, exit_efficiency * 100))
            
            # Did this loser go green first?
            went_green_before_loss = (exit_pnl <= 0 and state.highest_pnl > 0)
            
            excursion = ExcursionData(
                mfe_pnl=state.highest_pnl,
                mfe_pct=mfe_pct,
                mfe_price=state.mfe_price,
                mfe_time=state.mfe_time,
                mfe_bars_from_entry=state.mfe_bar,
                mae_pnl=state.lowest_pnl,
                mae_pct=mae_pct,
                mae_price=state.mae_price,
                mae_time=state.mae_time,
                mae_bars_from_entry=state.mae_bar,
                exit_efficiency=exit_efficiency,
                profit_captured_pct=profit_captured_pct,
                went_green_before_loss=went_green_before_loss,
                price_samples=state.price_samples if self._store_history else [],
            )
            
            logger.info(
                f"Finalized {trade_id}: MFE=${state.highest_pnl:.2f}, "
                f"MAE=${state.lowest_pnl:.2f}, Exit=${exit_pnl:.2f}, "
                f"Efficiency={exit_efficiency:.1%}"
            )
            
            return excursion
    
    def get_state(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get current tracking state for a position."""
        with self._lock:
            state = self._positions.get(trade_id)
            if not state:
                return None
            
            return {
                'trade_id': state.trade_id,
                'entry_price': state.entry_price,
                'entry_time': state.entry_time.isoformat(),
                'direction': state.direction,
                'highest_price': state.highest_price,
                'lowest_price': state.lowest_price,
                'mfe_pnl': state.highest_pnl,
                'mae_pnl': state.lowest_pnl,
                'mfe_price': state.mfe_price,
                'mae_price': state.mae_price,
                'bar_count': state.bar_count,
            }
    
    def is_tracking(self, trade_id: str) -> bool:
        """Check if a trade is being tracked."""
        with self._lock:
            return trade_id in self._positions
    
    def get_all_tracked(self) -> List[str]:
        """Get list of all tracked trade IDs."""
        with self._lock:
            return list(self._positions.keys())
    
    def clear(self) -> int:
        """Clear all tracking state. Returns number of positions cleared."""
        with self._lock:
            count = len(self._positions)
            self._positions.clear()
            return count
