"""
STRAT Trade Execution Log - Explicit Timestamp Tracking

Provides TradeExecutionRecord dataclass for tracking explicit timestamps
and exit reasons per Session 83K requirements:
- Time pattern traded (first bar in pattern)
- Time of entry
- Time of exit/stop
- Exit reason (TARGET, STOP, EXPIRATION, REJECTED)

Session 83K: ATLAS Validation Run for STRAT Strategies
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import pandas as pd


class ExitReason(str, Enum):
    """Exit reason enumeration for trade classification."""
    TARGET = 'TARGET'           # Hit target/magnitude
    STOP = 'STOP'               # Hit stop loss
    EXPIRATION = 'EXPIRATION'   # Options expired
    TIME_EXIT = 'TIME_EXIT'     # Time-based exit (e.g., EOD)
    REJECTED = 'REJECTED'       # Trade rejected by risk manager
    MANUAL = 'MANUAL'           # Manual exit
    UNKNOWN = 'UNKNOWN'         # Unknown reason


@dataclass
class TradeExecutionRecord:
    """
    Explicit timestamp tracking for STRAT pattern trades.

    Per Session 83K requirements:
    - pattern_timestamp: Time first bar in pattern (pattern detection time)
    - entry_timestamp: Time of entry
    - exit_timestamp: Time of exit/stop
    - exit_reason: Why trade was closed

    Attributes:
        trade_id: Unique trade identifier
        symbol: Trading symbol (e.g., 'SPY')
        pattern_type: STRAT pattern type (e.g., '3-1-2U', '2-1-2D')
        timeframe: Pattern timeframe ('1D', '1W', '1M')

        pattern_timestamp: When pattern was detected (trigger bar time)
        entry_timestamp: When entry was triggered
        exit_timestamp: When exit occurred
        exit_reason: How trade was closed (TARGET, STOP, EXPIRATION, REJECTED)

        entry_price: Underlying price at entry
        exit_price: Underlying price at exit
        stop_price: Stop loss level
        target_price: Target/magnitude level

        strike: Option strike price
        option_type: 'CALL' or 'PUT'
        osi_symbol: Full OSI option symbol
        option_entry_price: Option premium at entry
        option_exit_price: Option premium at exit

        entry_delta: Options delta at entry
        exit_delta: Options delta at exit
        entry_theta: Options theta at entry
        exit_theta: Options theta at exit

        pnl: Profit/loss in dollars
        pnl_pct: Profit/loss as percentage
        days_held: Days position was held

        validation_passed: Whether pre-trade validation passed
        validation_reason: Reason if validation failed
        circuit_state: Circuit breaker state at entry
        data_source: 'ThetaData', 'BlackScholes', 'Mixed'
    """
    # Identification
    trade_id: int
    symbol: str
    pattern_type: str
    timeframe: str

    # EXPLICIT TIMESTAMPS (Session 83K requirement)
    pattern_timestamp: datetime      # Time first bar in pattern
    entry_timestamp: datetime        # Time of entry
    exit_timestamp: datetime         # Time of exit/stop
    exit_reason: str                 # TARGET, STOP, EXPIRATION, REJECTED

    # Underlying prices
    entry_price: float               # Underlying at entry
    exit_price: float                # Underlying at exit
    stop_price: float                # Stop loss level
    target_price: float              # Target level

    # Options details
    strike: float = 0.0
    option_type: str = ''            # CALL or PUT
    osi_symbol: str = ''
    option_entry_price: float = 0.0
    option_exit_price: float = 0.0

    # Greeks
    entry_delta: float = 0.0
    exit_delta: float = 0.0
    entry_theta: float = 0.0
    exit_theta: float = 0.0

    # P/L
    pnl: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0

    # Validation
    validation_passed: bool = True
    validation_reason: str = ''
    circuit_state: str = 'NORMAL'

    # Data source
    data_source: str = 'BlackScholes'

    # Optional metadata
    direction: int = 1              # 1=long, -1=short
    continuation_bars: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Code version tracking (for audit traceability)
    code_version: str = ''          # Git commit short hash at trade creation
    code_session: str = ''          # Session ID (e.g., EQUITY-35)
    code_branch: str = ''           # Git branch name

    def __post_init__(self):
        """Calculate derived fields."""
        # Calculate days_held if not provided
        if self.days_held == 0 and self.entry_timestamp and self.exit_timestamp:
            delta = self.exit_timestamp - self.entry_timestamp
            self.days_held = max(1, delta.days)

        # Calculate pnl_pct if pnl is set but pnl_pct is not
        if self.pnl != 0 and self.pnl_pct == 0 and self.option_entry_price > 0:
            self.pnl_pct = self.pnl / (self.option_entry_price * 100)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def hit_target(self) -> bool:
        """Check if trade hit target."""
        return self.exit_reason == ExitReason.TARGET.value

    @property
    def hit_stop(self) -> bool:
        """Check if trade hit stop."""
        return self.exit_reason == ExitReason.STOP.value

    @property
    def was_rejected(self) -> bool:
        """Check if trade was rejected."""
        return self.exit_reason == ExitReason.REJECTED.value

    @property
    def time_to_entry(self) -> Optional[float]:
        """Time from pattern detection to entry (hours)."""
        if self.pattern_timestamp and self.entry_timestamp:
            delta = self.entry_timestamp - self.pattern_timestamp
            return delta.total_seconds() / 3600
        return None

    @property
    def time_in_trade(self) -> Optional[float]:
        """Time from entry to exit (hours)."""
        if self.entry_timestamp and self.exit_timestamp:
            delta = self.exit_timestamp - self.entry_timestamp
            return delta.total_seconds() / 3600
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'pattern_type': self.pattern_type,
            'timeframe': self.timeframe,
            'pattern_timestamp': self.pattern_timestamp,
            'entry_timestamp': self.entry_timestamp,
            'exit_timestamp': self.exit_timestamp,
            'exit_reason': self.exit_reason,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'strike': self.strike,
            'option_type': self.option_type,
            'osi_symbol': self.osi_symbol,
            'option_entry_price': self.option_entry_price,
            'option_exit_price': self.option_exit_price,
            'entry_delta': self.entry_delta,
            'exit_delta': self.exit_delta,
            'entry_theta': self.entry_theta,
            'exit_theta': self.exit_theta,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'days_held': self.days_held,
            'validation_passed': self.validation_passed,
            'validation_reason': self.validation_reason,
            'circuit_state': self.circuit_state,
            'data_source': self.data_source,
            'direction': self.direction,
            'continuation_bars': self.continuation_bars,
            'code_version': self.code_version,
            'code_session': self.code_session,
            'code_branch': self.code_branch,
        }

    def to_backtest_row(self) -> Dict[str, Any]:
        """Convert to BacktestResult.trades DataFrame row format."""
        return {
            # Required BacktestResult.trades columns
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'entry_date': self.entry_timestamp,
            'exit_date': self.exit_timestamp,
            'days_held': self.days_held,

            # Optional pattern columns
            'pattern_type': self.pattern_type,
            'exit_type': self.exit_reason,
            'symbol': self.symbol,
            'direction': self.direction,

            # Extended columns for Session 83K
            'pattern_timestamp': self.pattern_timestamp,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'strike': self.strike,
            'option_type': self.option_type,
            'entry_delta': self.entry_delta,
            'exit_delta': self.exit_delta,
            'validation_passed': self.validation_passed,
            'data_source': self.data_source,
        }


class TradeExecutionLog:
    """
    Collection of TradeExecutionRecord instances with analysis methods.

    Provides:
    - Add/retrieve records
    - Convert to DataFrame for BacktestResult
    - Exit reason breakdown
    - Timestamp analysis
    """

    def __init__(self):
        self.records: List[TradeExecutionRecord] = []

    def add_record(self, record: TradeExecutionRecord) -> None:
        """Add a trade record."""
        self.records.append(record)

    def add_records(self, records: List[TradeExecutionRecord]) -> None:
        """Add multiple trade records."""
        self.records.extend(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for BacktestResult.trades."""
        if not self.records:
            return pd.DataFrame(columns=[
                'pnl', 'pnl_pct', 'entry_date', 'exit_date', 'days_held',
                'pattern_type', 'exit_type', 'symbol', 'direction',
                'pattern_timestamp', 'entry_price', 'exit_price',
                'stop_price', 'target_price', 'validation_passed'
            ])

        rows = [r.to_backtest_row() for r in self.records]
        return pd.DataFrame(rows)

    def to_full_dataframe(self) -> pd.DataFrame:
        """Convert to full DataFrame with all fields."""
        if not self.records:
            return pd.DataFrame()

        rows = [r.to_dict() for r in self.records]
        return pd.DataFrame(rows)

    def get_exit_reason_breakdown(self) -> Dict[str, int]:
        """Count trades by exit reason."""
        breakdown = {}
        for record in self.records:
            reason = record.exit_reason
            breakdown[reason] = breakdown.get(reason, 0) + 1
        return breakdown

    def get_pattern_breakdown(self) -> Dict[str, int]:
        """Count trades by pattern type."""
        breakdown = {}
        for record in self.records:
            pattern = record.pattern_type
            breakdown[pattern] = breakdown.get(pattern, 0) + 1
        return breakdown

    def get_symbol_breakdown(self) -> Dict[str, int]:
        """Count trades by symbol."""
        breakdown = {}
        for record in self.records:
            symbol = record.symbol
            breakdown[symbol] = breakdown.get(symbol, 0) + 1
        return breakdown

    def get_timestamp_analysis(self) -> Dict[str, Any]:
        """Analyze time metrics across trades."""
        if not self.records:
            return {}

        times_to_entry = [r.time_to_entry for r in self.records if r.time_to_entry is not None]
        times_in_trade = [r.time_in_trade for r in self.records if r.time_in_trade is not None]

        return {
            'avg_time_to_entry_hours': sum(times_to_entry) / len(times_to_entry) if times_to_entry else 0,
            'avg_time_in_trade_hours': sum(times_in_trade) / len(times_in_trade) if times_in_trade else 0,
            'min_time_to_entry_hours': min(times_to_entry) if times_to_entry else 0,
            'max_time_to_entry_hours': max(times_to_entry) if times_to_entry else 0,
            'min_time_in_trade_hours': min(times_in_trade) if times_in_trade else 0,
            'max_time_in_trade_hours': max(times_in_trade) if times_in_trade else 0,
        }

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.records:
            return {}

        passed = sum(1 for r in self.records if r.validation_passed)
        rejected = sum(1 for r in self.records if not r.validation_passed)

        return {
            'total_trades': len(self.records),
            'validation_passed': passed,
            'validation_rejected': rejected,
            'pass_rate': passed / len(self.records) if self.records else 0,
        }

    def get_data_source_breakdown(self) -> Dict[str, int]:
        """Count trades by data source."""
        breakdown = {}
        for record in self.records:
            source = record.data_source
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown

    def filter_by_pattern(self, pattern_type: str) -> 'TradeExecutionLog':
        """Return new log filtered by pattern type."""
        filtered = TradeExecutionLog()
        for record in self.records:
            if record.pattern_type == pattern_type:
                filtered.add_record(record)
        return filtered

    def filter_by_symbol(self, symbol: str) -> 'TradeExecutionLog':
        """Return new log filtered by symbol."""
        filtered = TradeExecutionLog()
        for record in self.records:
            if record.symbol == symbol:
                filtered.add_record(record)
        return filtered

    def filter_by_exit_reason(self, reason: str) -> 'TradeExecutionLog':
        """Return new log filtered by exit reason."""
        filtered = TradeExecutionLog()
        for record in self.records:
            if record.exit_reason == reason:
                filtered.add_record(record)
        return filtered

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.records:
            return "No trades recorded."

        exit_breakdown = self.get_exit_reason_breakdown()
        validation = self.get_validation_stats()
        timestamp = self.get_timestamp_analysis()

        lines = [
            "=" * 50,
            "TRADE EXECUTION LOG SUMMARY",
            "=" * 50,
            f"Total Trades: {len(self.records)}",
            "",
            "Exit Reasons:",
        ]

        for reason, count in exit_breakdown.items():
            lines.append(f"  {reason}: {count} ({count/len(self.records)*100:.1f}%)")

        lines.extend([
            "",
            f"Validation Pass Rate: {validation.get('pass_rate', 0)*100:.1f}%",
            f"Avg Time to Entry: {timestamp.get('avg_time_to_entry_hours', 0):.1f} hours",
            f"Avg Time in Trade: {timestamp.get('avg_time_in_trade_hours', 0):.1f} hours",
            "=" * 50,
        ])

        return "\n".join(lines)
