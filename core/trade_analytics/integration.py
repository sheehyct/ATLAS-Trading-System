"""
Trade Analytics Integration Guide

Shows how to integrate ExcursionTracker into existing position monitors
to capture MFE/MAE data in real-time.

INTEGRATION STEPS:

1. EQUITY DAEMON (strat/signal_automation/daemon.py):
   - Add ExcursionTracker to PositionMonitor
   - Start tracking on position open
   - Update on each position check
   - Finalize and store on position close

2. CRYPTO DAEMON (crypto/scanning/daemon.py):
   - Add ExcursionTracker to CryptoPositionMonitor
   - Same lifecycle: start -> update -> finalize

Session: Trade Analytics Implementation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from core.trade_analytics.excursion_tracker import ExcursionTracker
from core.trade_analytics.trade_store import TradeStore
from core.trade_analytics.converter import TradeConverter
from core.trade_analytics.models import EnrichedTradeRecord

logger = logging.getLogger(__name__)


class TradeAnalyticsIntegration:
    """
    Helper class to integrate trade analytics into existing daemons.
    
    Usage in daemon:
    
        class SignalDaemon:
            def __init__(self):
                # ... existing init ...
                
                # Add analytics integration
                self.analytics = TradeAnalyticsIntegration(
                    store_path=Path("core/trade_analytics/data/equity_trades.json")
                )
            
            def _on_position_open(self, position):
                # Start tracking excursion
                self.analytics.on_position_open(
                    trade_id=position.signal_key,
                    entry_price=position.actual_entry_underlying,
                    direction=position.direction,
                    quantity=position.contracts,
                )
            
            def _check_positions(self):
                for pos in self.positions:
                    # Update excursion tracking
                    self.analytics.on_price_update(
                        trade_id=pos.signal_key,
                        current_price=current_underlying_price,
                    )
            
            def _on_position_exit(self, position, exit_signal):
                # Finalize and store enriched trade
                self.analytics.on_position_close(
                    position=position,
                    exit_price=exit_signal.underlying_price,
                )
    """
    
    def __init__(
        self,
        store_path: Optional[Path] = None,
        sample_interval_seconds: int = 60,
        store_price_history: bool = True,
    ):
        """
        Initialize analytics integration.
        
        Args:
            store_path: Path for trade storage
            sample_interval_seconds: How often to sample price history
            store_price_history: Whether to store detailed price history
        """
        self.store = TradeStore(
            store_path=store_path or Path("core/trade_analytics/data/trades.json")
        )
        self.excursion_tracker = ExcursionTracker(
            sample_interval_seconds=sample_interval_seconds,
            store_price_history=store_price_history,
        )
        
        logger.info(f"Trade analytics initialized, store: {self.store.store_path}")
    
    def on_position_open(
        self,
        trade_id: str,
        entry_price: float,
        direction: str,
        quantity: float = 1.0,
        multiplier: float = 100.0,  # 100 for options
        entry_time: Optional[datetime] = None,
    ) -> None:
        """
        Call when a position is opened.
        
        Args:
            trade_id: Unique identifier (signal_key for equity, trade_id for crypto)
            entry_price: Entry price (underlying for options)
            direction: "LONG", "SHORT", "CALL", "PUT", "BUY", "SELL"
            quantity: Number of contracts/shares
            multiplier: Contract multiplier (100 for options, 1 for crypto perps)
            entry_time: Entry timestamp
        """
        self.excursion_tracker.start_tracking(
            trade_id=trade_id,
            entry_price=entry_price,
            direction=direction,
            quantity=quantity,
            multiplier=multiplier,
            entry_time=entry_time,
        )
        
        logger.debug(f"Started excursion tracking for {trade_id}")
    
    def on_price_update(
        self,
        trade_id: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
        bar_closed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Call on each position check to update excursion tracking.
        
        Args:
            trade_id: Trade identifier
            current_price: Current underlying price
            timestamp: Current timestamp
            bar_closed: Whether a new bar just closed
        
        Returns:
            Current excursion state (mfe/mae)
        """
        return self.excursion_tracker.update(
            trade_id=trade_id,
            current_price=current_price,
            timestamp=timestamp,
            bar_closed=bar_closed,
        )
    
    def on_position_close(
        self,
        position: Any,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        asset_class: str = "equity_option",
    ) -> Optional[EnrichedTradeRecord]:
        """
        Call when a position is closed.
        
        Finalizes excursion tracking, converts to EnrichedTradeRecord,
        and stores in the trade store.
        
        Args:
            position: TrackedPosition (equity) or SimulatedTrade (crypto)
            exit_price: Exit price (underlying for options)
            exit_time: Exit timestamp
            asset_class: "equity_option" or "crypto_perp"
        
        Returns:
            EnrichedTradeRecord that was stored
        """
        # Get trade ID
        trade_id = getattr(position, 'signal_key', None) or getattr(position, 'trade_id', '')
        
        # Finalize excursion tracking
        excursion_data = self.excursion_tracker.finalize(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_time=exit_time,
        )
        
        # Convert position to enriched record
        if asset_class == "crypto_perp":
            record = TradeConverter.from_crypto_simulated_trade(position)
        else:
            record = TradeConverter.from_equity_tracked_position(
                position,
                excursion_data=excursion_data,
            )
        
        # Override excursion if we have tracker data
        if excursion_data:
            record.excursion = excursion_data
        
        # Finalize derived metrics
        record.finalize_excursion()
        
        # Store the enriched record
        self.store.add_trade(record)
        
        logger.info(
            f"Stored enriched trade {trade_id}: "
            f"P&L=${record.pnl:.2f}, MFE=${record.excursion.mfe_pnl:.2f}, "
            f"Efficiency={record.excursion.exit_efficiency:.1%}"
        )
        
        return record
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        return {
            'trades_stored': len(self.store),
            'positions_tracking': len(self.excursion_tracker.get_all_tracked()),
            'store_stats': self.store.get_stats(),
        }


# =============================================================================
# INTEGRATION CODE SNIPPETS
# =============================================================================

EQUITY_DAEMON_INTEGRATION = """
# Add to strat/signal_automation/daemon.py

# In __init__:
from core.trade_analytics.integration import TradeAnalyticsIntegration
from pathlib import Path

self.trade_analytics = TradeAnalyticsIntegration(
    store_path=Path("core/trade_analytics/data/equity_trades.json")
)

# In _on_entry_triggered (after successful execution):
if result.state == ExecutionState.ORDER_SUBMITTED:
    # Start excursion tracking
    self.trade_analytics.on_position_open(
        trade_id=signal.signal_key,
        entry_price=event.current_price,  # Underlying price at entry
        direction=signal.direction,
        quantity=result.contracts or 1,
        multiplier=100.0,  # Options multiplier
    )

# In position_monitor._check_position (add price update):
# Get underlying price
underlying_price = self._get_underlying_price(pos.symbol)
if underlying_price and hasattr(self, '_analytics_integration'):
    self._analytics_integration.on_price_update(
        trade_id=pos.signal_key,
        current_price=underlying_price,
    )

# In _on_position_exit callback:
if self.trade_analytics:
    self.trade_analytics.on_position_close(
        position=tracked_position,
        exit_price=exit_signal.underlying_price,
        asset_class="equity_option",
    )
"""

CRYPTO_DAEMON_INTEGRATION = """
# Add to crypto/scanning/daemon.py

# In __init__:
from core.trade_analytics.integration import TradeAnalyticsIntegration
from pathlib import Path

self.trade_analytics = TradeAnalyticsIntegration(
    store_path=Path("core/trade_analytics/data/crypto_trades.json"),
)

# In _open_position (after paper_trader.open_trade):
if trade:
    self.trade_analytics.on_position_open(
        trade_id=trade.trade_id,
        entry_price=fill_price,
        direction=trade.side,
        quantity=trade.quantity,
        multiplier=1.0,  # Crypto perps no multiplier
    )

# In _monitor_positions (inside position loop):
if self.trade_analytics:
    self.trade_analytics.on_price_update(
        trade_id=trade.trade_id,
        current_price=current_price,
    )

# In _close_position (after paper_trader.close_trade):
if closed_trade and self.trade_analytics:
    self.trade_analytics.on_position_close(
        position=closed_trade,
        exit_price=exit_price,
        asset_class="crypto_perp",
    )
"""

ANALYTICS_CLI_EXAMPLE = """
# Example CLI usage for analyzing trades

from core.trade_analytics import TradeStore, TradeAnalyticsEngine
from pathlib import Path

# Load trade store
store = TradeStore(Path("core/trade_analytics/data/equity_trades.json"))
analytics = TradeAnalyticsEngine(store)

# Question: "What's my win rate on hourly patterns?"
hourly_stats = analytics.win_rate_by_factor("timeframe", filter_value="1H")
for s in hourly_stats:
    print(f"{s.segment_value}: {s.win_rate:.1f}% win rate, {s.trades} trades")

# Question: "Is 1.5% magnitude working?"
mag_analysis = analytics.magnitude_sensitivity()
for m in mag_analysis:
    print(f"Min {m['threshold']}%: WR={m['win_rate']:.1f}%, PF={m['profit_factor']:.2f}")

# Question: "What's my optimal TFC threshold?"
tfc_analysis = analytics.tfc_sensitivity()
for t in tfc_analysis:
    print(f"Min TFC {t['min_tfc']}: WR={t['win_rate']:.1f}%, PF={t['profit_factor']:.2f}")

# Question: "How much profit am I leaving on table?"
efficiency = analytics.exit_efficiency_report()
print(f"Avg Exit Efficiency: {efficiency['winners']['avg_exit_efficiency']:.1%}")
print(f"Profit Left: ${efficiency['winners']['profit_left_on_table']:.2f}")
for insight in efficiency['insights']:
    print(insight)

# Question: "Win rate by VIX level?"
vix_stats = analytics.vix_sensitivity()
for s in vix_stats:
    print(f"VIX {s.segment_value}: {s.win_rate:.1f}% win rate, {s.trades} trades")

# Full report
report = analytics.generate_report()
print("\\n=== INSIGHTS ===")
for insight in report['insights']:
    print(insight)
"""


if __name__ == "__main__":
    # Print integration guides
    print("=" * 60)
    print("EQUITY DAEMON INTEGRATION")
    print("=" * 60)
    print(EQUITY_DAEMON_INTEGRATION)
    
    print("=" * 60)
    print("CRYPTO DAEMON INTEGRATION")
    print("=" * 60)
    print(CRYPTO_DAEMON_INTEGRATION)
    
    print("=" * 60)
    print("ANALYTICS CLI EXAMPLE")
    print("=" * 60)
    print(ANALYTICS_CLI_EXAMPLE)
