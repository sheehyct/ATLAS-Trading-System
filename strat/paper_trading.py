"""
STRAT Paper Trading Infrastructure - Session 83K-41

Comprehensive paper trade tracking system with 40+ fields for capturing
all metrics needed for future analysis and comparison with backtests.

Strategic Pivot Decision (Session 83K-40):
- Skip ML optimization, go straight to paper trading
- Baseline already strong (Sharpe 3.97, 65% win rate)
- Paper trading validates ALL backtest assumptions with zero cost

Key Design Principles:
1. Include ALL patterns and timeframes (2-2, 3-2, 3-2-2, 2-1-2, 3-1-2)
2. Capture comprehensive context (VIX, ATR, TFC, ATLAS regime)
3. Enable direct comparison to backtest expectations
4. Persistent storage for long-term data collection
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import json
import csv
import os

import pandas as pd


class TradeDirection(str, Enum):
    """Trade direction enumeration."""
    CALL = 'CALL'
    PUT = 'PUT'


class EntryReason(str, Enum):
    """Entry reason classification."""
    PATTERN_DETECTED = 'PATTERN_DETECTED'
    TRIGGER_HIT = 'TRIGGER_HIT'
    BREAKOUT = 'BREAKOUT'
    GAP_ENTRY = 'GAP_ENTRY'


class ExitReason(str, Enum):
    """Exit reason classification."""
    TARGET_HIT = 'TARGET_HIT'
    STOP_HIT = 'STOP_HIT'
    TIME_EXIT = 'TIME_EXIT'
    MANUAL = 'MANUAL'
    EXPIRATION = 'EXPIRATION'
    PATTERN_INVALIDATED = 'PATTERN_INVALIDATED'


class MarketRegime(str, Enum):
    """ATLAS market regime classification."""
    TREND_BULL = 'TREND_BULL'
    TREND_BEAR = 'TREND_BEAR'
    TREND_NEUTRAL = 'TREND_NEUTRAL'
    CRASH = 'CRASH'


class VixBucket(int, Enum):
    """VIX bucket classification."""
    LOW = 1       # VIX < 15
    MEDIUM = 2    # 15 <= VIX < 20
    ELEVATED = 3  # 20 <= VIX < 30
    HIGH = 4      # 30 <= VIX < 40
    EXTREME = 5   # VIX >= 40


# Session 83K-52: Import from single source of truth
from strat.tier1_detector import PatternType, Timeframe


@dataclass
class PaperTrade:
    """
    Comprehensive paper trade record with 40+ fields.

    Designed for capturing ALL metrics needed for future analysis and
    comparison with backtest expectations.

    Field Categories:
    1. Core Trade Fields - Identification and classification
    2. Entry Fields - When and how the trade was entered
    3. Options Fields - Strike, expiration, Greeks
    4. Target/Stop Fields - Risk/reward parameters
    5. Exit Fields - When and how the trade was closed
    6. P&L Fields - Financial performance
    7. Context Fields - Market conditions at entry
    8. Data Source Fields - Where data came from
    """

    # =========================================================================
    # Core Trade Fields
    # =========================================================================
    trade_id: str                    # Unique identifier (e.g., PT_20251204_001)
    pattern_type: str                # STRAT pattern (2-2U, 3-2D, etc.)
    timeframe: str                   # Chart timeframe (1H, 1D, 1W, 1M)
    symbol: str                      # Underlying symbol (SPY, QQQ, etc.)
    direction: str                   # Trade direction (CALL or PUT)

    # =========================================================================
    # Entry Fields
    # =========================================================================
    pattern_detected_time: datetime  # When pattern completed
    entry_time: Optional[datetime] = None   # Actual entry timestamp
    entry_reason: str = ''           # Why entered (PATTERN_DETECTED, TRIGGER_HIT)
    entry_price: float = 0.0         # Underlying price at entry
    entry_trigger: float = 0.0       # Pattern trigger level

    # =========================================================================
    # Options Fields
    # =========================================================================
    strike: float = 0.0              # Option strike price
    expiration: Optional[date] = None  # Option expiration date
    dte_at_entry: int = 0            # Days to expiration at entry
    option_price_entry: float = 0.0  # Premium paid
    delta_at_entry: float = 0.0      # Delta at entry
    gamma_at_entry: float = 0.0      # Gamma at entry
    theta_at_entry: float = 0.0      # Theta at entry
    iv_at_entry: float = 0.0         # Implied volatility at entry
    contracts: int = 1               # Number of contracts

    # =========================================================================
    # Target/Stop Fields
    # =========================================================================
    target_price: float = 0.0        # Magnitude target (underlying)
    stop_price: float = 0.0          # Stop loss level (underlying)
    magnitude_pct: float = 0.0       # Expected move percentage
    distance_to_target: float = 0.0  # Entry to target distance
    risk_reward: float = 0.0         # Risk/reward ratio

    # =========================================================================
    # Exit Fields
    # =========================================================================
    exit_time: Optional[datetime] = None     # Actual exit timestamp
    exit_reason: str = ''            # Why exited (TARGET_HIT, STOP_HIT, etc.)
    exit_price: float = 0.0          # Underlying price at exit
    option_price_exit: float = 0.0   # Premium received
    delta_at_exit: float = 0.0       # Delta at exit
    iv_at_exit: float = 0.0          # IV at exit
    days_held: int = 0               # Duration of trade

    # =========================================================================
    # P&L Fields
    # =========================================================================
    pnl_dollars: float = 0.0         # Actual P&L in dollars
    pnl_percent: float = 0.0         # Return percentage
    pnl_per_contract: float = 0.0    # P&L per contract

    # =========================================================================
    # Context Fields (Critical for Analysis)
    # =========================================================================
    vix_at_entry: float = 0.0        # VIX level at entry
    vix_bucket: int = 0              # VIX category (1-5)
    atr_14: float = 0.0              # 14-day ATR
    atr_percent: float = 0.0         # ATR as percentage of price
    volume_ratio: float = 0.0        # Volume vs 20-day average
    tfc_score: int = 0               # Timeframe continuity score (0-5)
    tfc_alignment: str = ''          # Higher TF direction (BULLISH, BEARISH, NEUTRAL)
    market_regime: str = ''          # ATLAS regime (TREND_BULL, CRASH, etc.)

    # =========================================================================
    # Data Source Fields
    # =========================================================================
    quote_source: str = ''           # Options data source (ThetaData, Manual)
    underlying_source: str = ''      # Price data source (Alpaca)
    notes: str = ''                  # Free-form notes

    # =========================================================================
    # Metadata
    # =========================================================================
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = 'PENDING'          # PENDING, OPEN, CLOSED, CANCELLED

    def __post_init__(self):
        """Calculate derived fields."""
        # Use entry_price if set, otherwise use entry_trigger for calculations
        reference_price = self.entry_price if self.entry_price > 0 else self.entry_trigger

        # Calculate magnitude percentage if not provided
        if self.magnitude_pct == 0 and reference_price > 0 and self.target_price > 0:
            self.magnitude_pct = abs(self.target_price - reference_price) / reference_price * 100

        # Calculate distance to target
        if self.distance_to_target == 0 and reference_price > 0 and self.target_price > 0:
            self.distance_to_target = abs(self.target_price - reference_price)

        # Calculate risk/reward
        if self.risk_reward == 0 and reference_price > 0 and self.stop_price > 0:
            risk = abs(reference_price - self.stop_price)
            reward = abs(self.target_price - reference_price)
            if risk > 0:
                self.risk_reward = reward / risk

        # Calculate VIX bucket
        if self.vix_bucket == 0 and self.vix_at_entry > 0:
            self.vix_bucket = self._calculate_vix_bucket(self.vix_at_entry)

        # Calculate days held if exit time is set
        if self.days_held == 0 and self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.days_held = max(1, delta.days)

    @staticmethod
    def _calculate_vix_bucket(vix: float) -> int:
        """Calculate VIX bucket from VIX value."""
        if vix < 15:
            return VixBucket.LOW.value
        elif vix < 20:
            return VixBucket.MEDIUM.value
        elif vix < 30:
            return VixBucket.ELEVATED.value
        elif vix < 40:
            return VixBucket.HIGH.value
        else:
            return VixBucket.EXTREME.value

    @staticmethod
    def generate_trade_id() -> str:
        """Generate unique trade ID based on timestamp."""
        now = datetime.now()
        return f"PT_{now.strftime('%Y%m%d_%H%M%S')}"

    @property
    def is_open(self) -> bool:
        """Check if trade is currently open."""
        return self.status == 'OPEN'

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.status == 'CLOSED'

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl_dollars > 0

    @property
    def hit_target(self) -> bool:
        """Check if trade hit target."""
        return self.exit_reason == ExitReason.TARGET_HIT.value

    @property
    def hit_stop(self) -> bool:
        """Check if trade hit stop."""
        return self.exit_reason == ExitReason.STOP_HIT.value

    def open_trade(self, entry_time: datetime, entry_price: float,
                   option_price: float, delta: float = 0.0) -> None:
        """Mark trade as open with entry details."""
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.option_price_entry = option_price
        self.delta_at_entry = delta
        self.status = 'OPEN'
        self.updated_at = datetime.now()

    def close_trade(self, exit_time: datetime, exit_price: float,
                    option_price: float, exit_reason: str,
                    delta: float = 0.0) -> None:
        """Mark trade as closed with exit details."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.option_price_exit = option_price
        self.exit_reason = exit_reason
        self.delta_at_exit = delta
        self.status = 'CLOSED'

        # Calculate P&L
        if self.option_price_entry > 0:
            self.pnl_per_contract = (option_price - self.option_price_entry) * 100
            self.pnl_dollars = self.pnl_per_contract * self.contracts
            self.pnl_percent = (option_price - self.option_price_entry) / self.option_price_entry * 100

        # Calculate days held
        if self.entry_time:
            delta = exit_time - self.entry_time
            self.days_held = max(1, delta.days)

        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, date):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperTrade':
        """Create PaperTrade from dictionary."""
        # Convert ISO strings back to datetime
        datetime_fields = ['pattern_detected_time', 'entry_time', 'exit_time',
                          'created_at', 'updated_at']
        date_fields = ['expiration']

        for field_name in datetime_fields:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])

        for field_name in date_fields:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = date.fromisoformat(data[field_name])

        return cls(**data)

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        data = self.to_dict()
        return data


class PaperTradeLog:
    """
    Persistent storage and management for paper trades.

    Features:
    - CRUD operations for paper trades
    - CSV and JSON persistence
    - Filtering and analysis methods
    - Comparison with backtest expectations
    """

    def __init__(self, storage_dir: str = "paper_trades"):
        """Initialize paper trade log with storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trades: List[PaperTrade] = []
        self._load_trades()

    def _get_csv_path(self) -> Path:
        """Get path to CSV storage file."""
        return self.storage_dir / "paper_trades.csv"

    def _get_json_path(self) -> Path:
        """Get path to JSON storage file."""
        return self.storage_dir / "paper_trades.json"

    def _load_trades(self) -> None:
        """Load trades from storage."""
        json_path = self._get_json_path()
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.trades = [PaperTrade.from_dict(d) for d in data]

    def _save_trades(self) -> None:
        """Save trades to both CSV and JSON."""
        # Save to JSON (primary storage)
        json_path = self._get_json_path()
        with open(json_path, 'w') as f:
            json.dump([t.to_dict() for t in self.trades], f, indent=2)

        # Save to CSV (for easy viewing/analysis)
        csv_path = self._get_csv_path()
        if self.trades:
            df = self.to_dataframe()
            df.to_csv(csv_path, index=False)

    def add_trade(self, trade: PaperTrade) -> None:
        """Add a new paper trade."""
        self.trades.append(trade)
        self._save_trades()

    def update_trade(self, trade_id: str, **kwargs) -> bool:
        """Update an existing trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                for key, value in kwargs.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                trade.updated_at = datetime.now()
                self._save_trades()
                return True
        return False

    def get_trade(self, trade_id: str) -> Optional[PaperTrade]:
        """Get a trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_open_trades(self) -> List[PaperTrade]:
        """Get all open trades."""
        return [t for t in self.trades if t.is_open]

    def get_closed_trades(self) -> List[PaperTrade]:
        """Get all closed trades."""
        return [t for t in self.trades if t.is_closed]

    def get_trades_by_pattern(self, pattern_type: str) -> List[PaperTrade]:
        """Get trades by pattern type."""
        return [t for t in self.trades if t.pattern_type == pattern_type]

    def get_trades_by_symbol(self, symbol: str) -> List[PaperTrade]:
        """Get trades by symbol."""
        return [t for t in self.trades if t.symbol == symbol]

    def get_trades_by_timeframe(self, timeframe: str) -> List[PaperTrade]:
        """Get trades by timeframe."""
        return [t for t in self.trades if t.timeframe == timeframe]

    def get_trades_by_date_range(self, start: datetime, end: datetime) -> List[PaperTrade]:
        """Get trades within date range."""
        return [t for t in self.trades
                if t.pattern_detected_time >= start and t.pattern_detected_time <= end]

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by ID."""
        for i, trade in enumerate(self.trades):
            if trade.trade_id == trade_id:
                del self.trades[i]
                self._save_trades()
                return True
        return False

    def __len__(self) -> int:
        return len(self.trades)

    def __iter__(self):
        return iter(self.trades)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_csv_row() for t in self.trades])

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all closed trades."""
        closed = self.get_closed_trades()
        if not closed:
            return {'total_trades': 0, 'message': 'No closed trades'}

        winners = [t for t in closed if t.is_winner]
        target_hits = [t for t in closed if t.hit_target]
        stop_hits = [t for t in closed if t.hit_stop]

        total_pnl = sum(t.pnl_dollars for t in closed)
        avg_pnl = total_pnl / len(closed) if closed else 0

        return {
            'total_trades': len(closed),
            'winners': len(winners),
            'losers': len(closed) - len(winners),
            'win_rate': len(winners) / len(closed) * 100 if closed else 0,
            'target_hits': len(target_hits),
            'stop_hits': len(stop_hits),
            'target_rate': len(target_hits) / len(closed) * 100 if closed else 0,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_days_held': sum(t.days_held for t in closed) / len(closed) if closed else 0,
        }

    def get_pattern_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by pattern type."""
        closed = self.get_closed_trades()
        patterns = set(t.pattern_type for t in closed)

        breakdown = {}
        for pattern in patterns:
            pattern_trades = [t for t in closed if t.pattern_type == pattern]
            winners = [t for t in pattern_trades if t.is_winner]
            total_pnl = sum(t.pnl_dollars for t in pattern_trades)

            breakdown[pattern] = {
                'trades': len(pattern_trades),
                'win_rate': len(winners) / len(pattern_trades) * 100 if pattern_trades else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(pattern_trades) if pattern_trades else 0,
            }

        return breakdown

    def get_timeframe_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by timeframe."""
        closed = self.get_closed_trades()
        timeframes = set(t.timeframe for t in closed)

        breakdown = {}
        for tf in timeframes:
            tf_trades = [t for t in closed if t.timeframe == tf]
            winners = [t for t in tf_trades if t.is_winner]
            total_pnl = sum(t.pnl_dollars for t in tf_trades)

            breakdown[tf] = {
                'trades': len(tf_trades),
                'win_rate': len(winners) / len(tf_trades) * 100 if tf_trades else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(tf_trades) if tf_trades else 0,
            }

        return breakdown

    def get_vix_bucket_breakdown(self) -> Dict[int, Dict[str, Any]]:
        """Get performance breakdown by VIX bucket."""
        closed = self.get_closed_trades()
        buckets = set(t.vix_bucket for t in closed if t.vix_bucket > 0)

        breakdown = {}
        for bucket in buckets:
            bucket_trades = [t for t in closed if t.vix_bucket == bucket]
            winners = [t for t in bucket_trades if t.is_winner]
            total_pnl = sum(t.pnl_dollars for t in bucket_trades)

            breakdown[bucket] = {
                'trades': len(bucket_trades),
                'win_rate': len(winners) / len(bucket_trades) * 100 if bucket_trades else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(bucket_trades) if bucket_trades else 0,
            }

        return breakdown

    def compare_to_backtest(self, backtest_stats: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Compare paper trading results to backtest expectations.

        Args:
            backtest_stats: Dict with 'win_rate', 'avg_pnl', 'target_rate' from backtest

        Returns:
            Dict with paper vs backtest comparison for each metric
        """
        paper_stats = self.get_summary_stats()

        if paper_stats.get('total_trades', 0) == 0:
            return {'message': 'No paper trades to compare'}

        comparison = {
            'win_rate': {
                'backtest': backtest_stats.get('win_rate', 0),
                'paper': paper_stats.get('win_rate', 0),
                'difference': paper_stats.get('win_rate', 0) - backtest_stats.get('win_rate', 0),
            },
            'avg_pnl': {
                'backtest': backtest_stats.get('avg_pnl', 0),
                'paper': paper_stats.get('avg_pnl', 0),
                'difference': paper_stats.get('avg_pnl', 0) - backtest_stats.get('avg_pnl', 0),
            },
            'target_rate': {
                'backtest': backtest_stats.get('target_rate', 0),
                'paper': paper_stats.get('target_rate', 0),
                'difference': paper_stats.get('target_rate', 0) - backtest_stats.get('target_rate', 0),
            },
            'trade_count': {
                'paper': paper_stats.get('total_trades', 0),
            },
        }

        return comparison

    def summary_report(self) -> str:
        """Generate human-readable summary report."""
        stats = self.get_summary_stats()

        if stats.get('total_trades', 0) == 0:
            return "No trades recorded."

        lines = [
            "=" * 60,
            "PAPER TRADING SUMMARY REPORT",
            "=" * 60,
            f"Total Closed Trades: {stats['total_trades']}",
            f"Open Trades: {len(self.get_open_trades())}",
            "",
            "PERFORMANCE METRICS:",
            f"  Win Rate: {stats['win_rate']:.1f}%",
            f"  Target Hit Rate: {stats['target_rate']:.1f}%",
            f"  Total P&L: ${stats['total_pnl']:,.2f}",
            f"  Avg P&L per Trade: ${stats['avg_pnl']:,.2f}",
            f"  Avg Days Held: {stats['avg_days_held']:.1f}",
            "",
        ]

        # Pattern breakdown
        pattern_breakdown = self.get_pattern_breakdown()
        if pattern_breakdown:
            lines.append("BY PATTERN:")
            for pattern, pstats in sorted(pattern_breakdown.items()):
                lines.append(f"  {pattern}: {pstats['trades']} trades, "
                           f"{pstats['win_rate']:.1f}% WR, "
                           f"${pstats['avg_pnl']:,.2f} avg")
            lines.append("")

        # Timeframe breakdown
        tf_breakdown = self.get_timeframe_breakdown()
        if tf_breakdown:
            lines.append("BY TIMEFRAME:")
            for tf, tstats in sorted(tf_breakdown.items()):
                lines.append(f"  {tf}: {tstats['trades']} trades, "
                           f"{tstats['win_rate']:.1f}% WR, "
                           f"${tstats['avg_pnl']:,.2f} avg")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# Factory Functions
# =============================================================================

def create_paper_trade(
    pattern_type: str,
    timeframe: str,
    symbol: str,
    direction: str,
    pattern_detected_time: datetime,
    entry_trigger: float,
    target_price: float,
    stop_price: float,
    strike: float = 0.0,
    expiration: Optional[date] = None,
    dte: int = 0,
    vix: float = 0.0,
    atr: float = 0.0,
    market_regime: str = '',
    notes: str = ''
) -> PaperTrade:
    """
    Factory function to create a new paper trade with sensible defaults.

    Args:
        pattern_type: STRAT pattern (e.g., '3-2U', '2-2D')
        timeframe: Chart timeframe ('1H', '1D', '1W', '1M')
        symbol: Underlying symbol
        direction: 'CALL' or 'PUT'
        pattern_detected_time: When pattern was detected
        entry_trigger: Trigger price level
        target_price: Magnitude target
        stop_price: Stop loss level
        strike: Option strike (optional, can be set later)
        expiration: Option expiration (optional)
        dte: Days to expiration (optional)
        vix: VIX at pattern detection
        atr: ATR at pattern detection
        market_regime: ATLAS regime
        notes: Free-form notes

    Returns:
        PaperTrade instance
    """
    trade_id = PaperTrade.generate_trade_id()

    return PaperTrade(
        trade_id=trade_id,
        pattern_type=pattern_type,
        timeframe=timeframe,
        symbol=symbol,
        direction=direction,
        pattern_detected_time=pattern_detected_time,
        entry_trigger=entry_trigger,
        target_price=target_price,
        stop_price=stop_price,
        strike=strike,
        expiration=expiration,
        dte_at_entry=dte,
        vix_at_entry=vix,
        atr_14=atr,
        market_regime=market_regime,
        notes=notes,
        status='PENDING',
        quote_source='ThetaData',
        underlying_source='Alpaca',
    )


# =============================================================================
# Backtest Comparison Reference
# =============================================================================

# Baseline metrics from Session 83K-40 (training set)
BASELINE_BACKTEST_STATS = {
    'win_rate': 65.03,      # 65% win rate
    'avg_pnl': 617.12,      # $617 average P&L
    'target_rate': 64.72,   # 65% target hit rate
    'sharpe': 3.97,         # Sharpe ratio
    'total_trades': 652,    # Training set trades
}

# Hourly assumption to validate (NEGATIVE in backtest)
HOURLY_BACKTEST_STATS = {
    'avg_pnl': -240,        # -$240 average P&L
    'win_rate': 33,         # ~33% win rate
    'trades': 1009,         # Hourly trades in backtest
}
