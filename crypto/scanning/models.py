"""
Data models for crypto STRAT signal scanning.

Adapted from strat/paper_signal_scanner.py for crypto perpetual futures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CryptoSignalContext:
    """
    Market context captured at signal detection time.

    Similar to equities SignalContext but includes crypto-specific
    metrics like funding rate.
    """

    atr_14: float = 0.0
    atr_percent: float = 0.0
    volume_24h_avg: float = 0.0
    current_volume: float = 0.0
    volume_ratio: float = 0.0
    funding_rate: float = 0.0  # Crypto-specific: current funding rate
    tfc_score: int = 0  # Full Timeframe Continuity score (0-4)
    tfc_alignment: str = ""  # e.g., "4/4 BULLISH"
    tfc_passes: bool = False
    risk_multiplier: float = 1.0
    priority_rank: int = 0


@dataclass
class CryptoDetectedSignal:
    """
    A detected STRAT pattern signal for crypto trading.

    Represents a pattern detected on a specific symbol/timeframe
    with entry, stop, and target levels.
    """

    # Pattern identification
    pattern_type: str  # Full bar sequence: '3-1-2U', '2D-2U', etc.
    direction: str  # 'LONG' or 'SHORT'
    symbol: str  # e.g., 'BTC-PERP-INTX'
    timeframe: str  # '1w', '1d', '4h', '1h', '15m'

    # Timing
    detected_time: datetime

    # Price levels
    entry_trigger: float
    stop_price: float
    target_price: float

    # Metrics
    magnitude_pct: float  # Target move as percentage
    risk_reward: float  # Reward:Risk ratio

    # Context
    context: CryptoSignalContext = field(default_factory=CryptoSignalContext)

    # Signal type: SETUP (waiting for break) or COMPLETED (already triggered)
    signal_type: str = "COMPLETED"

    # Setup-based detection fields (for SETUP signals)
    setup_bar_high: float = 0.0
    setup_bar_low: float = 0.0
    setup_bar_timestamp: Optional[datetime] = None

    # Prior bar info for detecting pattern transitions (Session CRYPTO-8)
    # When SETUP's inside bar breaks, this determines the resulting 2-bar pattern
    prior_bar_type: int = 0  # 2 = 2U, -2 = 2D, 3 = outside
    prior_bar_high: float = 0.0
    prior_bar_low: float = 0.0

    # Maintenance window flag
    has_maintenance_gap: bool = False

    def to_dict(self) -> dict:
        """Convert signal to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "detected_time": self.detected_time.isoformat(),
            "entry_trigger": self.entry_trigger,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "magnitude_pct": self.magnitude_pct,
            "risk_reward": self.risk_reward,
            "signal_type": self.signal_type,
            "setup_bar_high": self.setup_bar_high,
            "setup_bar_low": self.setup_bar_low,
            "has_maintenance_gap": self.has_maintenance_gap,
            "context": {
                "atr_14": self.context.atr_14,
                "atr_percent": self.context.atr_percent,
                "volume_ratio": self.context.volume_ratio,
                "funding_rate": self.context.funding_rate,
                "tfc_score": self.context.tfc_score,
                "tfc_alignment": self.context.tfc_alignment,
                "tfc_passes": self.context.tfc_passes,
                "risk_multiplier": self.context.risk_multiplier,
                "priority_rank": self.context.priority_rank,
            },
        }
