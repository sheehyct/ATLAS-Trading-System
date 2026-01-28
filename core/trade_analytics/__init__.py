"""
ATLAS Trade Analytics Engine

Comprehensive trade analysis system for learning from historical trades.
Tracks MFE/MAE, calculates exit efficiency, and provides factor-based
segmentation to answer questions like:

- "What's my win rate on hourly patterns?"
- "Is 1.5% magnitude filter optimal?"
- "What's my optimal TFC threshold?"
- "How much profit am I leaving on the table?"
- "Win rate by VIX level?"

Session: Trade Analytics Implementation
"""

from core.trade_analytics.models import (
    EnrichedTradeRecord,
    ExcursionData,
    MarketContext,
    PatternContext,
    PositionManagement,
)
from core.trade_analytics.excursion_tracker import ExcursionTracker
from core.trade_analytics.analytics_engine import TradeAnalyticsEngine
from core.trade_analytics.trade_store import TradeStore

__all__ = [
    # Models
    "EnrichedTradeRecord",
    "ExcursionData", 
    "MarketContext",
    "PatternContext",
    "PositionManagement",
    # Core components
    "ExcursionTracker",
    "TradeAnalyticsEngine",
    "TradeStore",
]
