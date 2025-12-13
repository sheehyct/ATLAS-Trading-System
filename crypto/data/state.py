"""
State management for crypto trading system.

Tracks bar classifications, OHLCV data, patterns, positions, and account info.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class CryptoSystemState:
    """
    Manages the state of the crypto trading system.

    Tracks:
    - Bar classifications per timeframe (1, 2U, 2D, 3)
    - Current OHLCV data per timeframe
    - Active STRAT patterns
    - Account and position state
    """

    # Bar classifications by timeframe
    # Keys: '15m', '1h', '4h', '1d', '1w'
    # Values: 1=inside, 2=2U, -2=2D, 3=outside, 0=unclassified
    bar_classifications: Dict[str, int] = field(default_factory=dict)

    # Current OHLCV data by timeframe
    # Keys: '15m', '1h', '4h', '1d', '1w'
    # Values: DataFrame with OHLCV columns
    current_bars: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Active patterns detected
    # Each pattern: {type, direction, entry_price, stop_price, target_price, timeframe, detected_at}
    active_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Account state
    account_equity: float = 0.0
    available_margin: float = 0.0
    current_position: Optional[Dict[str, Any]] = None

    # System status
    last_update_time: Optional[datetime] = None
    is_healthy: bool = True
    last_error: Optional[str] = None

    # Symbols being tracked
    active_symbols: List[str] = field(default_factory=lambda: ["BTC-USD"])

    def update_classification(self, timeframe: str, classification: int) -> None:
        """
        Update bar classification for a timeframe.

        Args:
            timeframe: Timeframe key (e.g., '15m', '1h', '4h')
            classification: Bar type (1=inside, 2=2U, -2=2D, 3=outside)
        """
        self.bar_classifications[timeframe] = classification
        self.last_update_time = datetime.utcnow()

    def update_bar_data(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Update OHLCV data for a timeframe.

        Args:
            timeframe: Timeframe key
            df: DataFrame with OHLCV columns
        """
        self.current_bars[timeframe] = df.copy()
        self.last_update_time = datetime.utcnow()

    def update_account(
        self,
        equity: float,
        margin: float,
        position: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update account details.

        Args:
            equity: Current account equity
            margin: Available margin
            position: Current position dict or None
        """
        self.account_equity = equity
        self.available_margin = margin
        self.current_position = position
        self.last_update_time = datetime.utcnow()

    def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """
        Add a detected pattern.

        Args:
            pattern: Pattern dict with type, direction, prices, etc.
        """
        pattern["detected_at"] = datetime.utcnow().isoformat()
        self.active_patterns.append(pattern)

    def clear_expired_patterns(self, max_age_minutes: int = 60) -> int:
        """
        Remove patterns older than max_age.

        Args:
            max_age_minutes: Maximum pattern age in minutes

        Returns:
            Number of patterns removed
        """
        now = datetime.utcnow()
        initial_count = len(self.active_patterns)

        self.active_patterns = [
            p
            for p in self.active_patterns
            if (now - datetime.fromisoformat(p["detected_at"])).total_seconds()
            < max_age_minutes * 60
        ]

        return initial_count - len(self.active_patterns)

    def get_latest_price(self, timeframe: str = "15m") -> Optional[float]:
        """
        Get latest close price from specified timeframe.

        Args:
            timeframe: Timeframe to get price from

        Returns:
            Latest close price or None
        """
        if timeframe in self.current_bars and not self.current_bars[timeframe].empty:
            return float(self.current_bars[timeframe].iloc[-1]["close"])
        return None

    def get_continuity_score(self, direction: int) -> int:
        """
        Calculate Full Timeframe Continuity (FTFC) score.

        Args:
            direction: Trade direction (1=bullish, -1=bearish)

        Returns:
            Score from 0-4
        """
        score = 0
        target = 2 * direction  # 2 for bull, -2 for bear

        for tf in ["1w", "1d", "4h", "1h"]:
            if self.bar_classifications.get(tf) == target:
                score += 1

        return score

    def check_vetoes(self) -> tuple[bool, str]:
        """
        Check absolute vetoes (Weekly/Daily inside bars).

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        weekly = self.bar_classifications.get("1w", 0)
        daily = self.bar_classifications.get("1d", 0)

        if weekly == 1:
            return False, "Weekly is Scenario 1 (inside bar) - NO TRADES"
        if daily == 1:
            return False, "Daily is Scenario 1 (inside bar) - NO TRADES"

        return True, "No vetoes active"

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current system state.

        Returns:
            Dict with state summary
        """
        can_trade, veto_reason = self.check_vetoes()

        return {
            "symbols": self.active_symbols,
            "last_update": (
                self.last_update_time.isoformat() if self.last_update_time else None
            ),
            "is_healthy": self.is_healthy,
            "bar_classifications": self.bar_classifications.copy(),
            "can_trade": can_trade,
            "veto_reason": veto_reason if not can_trade else None,
            "account_equity": self.account_equity,
            "has_position": self.current_position is not None,
            "active_patterns_count": len(self.active_patterns),
        }

    def reset(self) -> None:
        """Reset state to initial values."""
        self.bar_classifications = {}
        self.current_bars = {}
        self.active_patterns = []
        self.account_equity = 0.0
        self.available_margin = 0.0
        self.current_position = None
        self.last_update_time = None
        self.is_healthy = True
        self.last_error = None
