"""
STRAT Pattern Metrics - Trade Result Dataclass

Provides the PatternTradeResult dataclass for representing individual trades
with associated STRAT pattern metadata. Used by PatternMetricsAnalyzer for
per-pattern, per-timeframe, and per-regime performance breakdown.

Session 83G: Pattern metrics implementation per ATLAS Checklist Section 9.2.

Usage:
    from strat.pattern_metrics import PatternTradeResult

    trade = PatternTradeResult(
        trade_id=1,
        symbol='SPY',
        pattern_type='3-1-2U',
        timeframe='1D',
        regime='TREND_BULL',
        entry_date=datetime(2024, 1, 15),
        exit_date=datetime(2024, 1, 18),
        entry_price=100.0,
        exit_price=105.0,
        stop_price=97.0,
        target_price=108.0,
        pnl=500.0,
        pnl_pct=0.05,
        is_winner=True,
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

# Session 83K-52: Import from single source of truth
from strat.tier1_detector import PatternType


@dataclass
class PatternTradeResult:
    """
    Represents a single trade with STRAT pattern metadata.

    Includes entry/exit details, P/L metrics, and pattern classification
    for use in per-pattern, per-timeframe, per-regime analysis.

    Attributes:
        trade_id: Unique trade identifier
        symbol: Trading symbol (e.g., 'SPY')
        pattern_type: STRAT pattern type string
        timeframe: Timeframe of the pattern (e.g., '1D', '1W', '1M')
        regime: ATLAS regime at time of entry
        entry_date: Trade entry datetime
        exit_date: Trade exit datetime
        entry_price: Entry price
        exit_price: Exit price
        stop_price: Stop loss price
        target_price: Target price
        pnl: Profit/loss in dollars
        pnl_pct: Profit/loss as percentage
        is_winner: True if trade was profitable
        days_held: Number of days position held
        hit_target: True if trade hit target price
        hit_stop: True if trade hit stop price
        risk_reward_ratio: Realized risk/reward ratio

        # Options-specific fields
        is_options_trade: Whether this was an options trade
        data_source: 'ThetaData', 'BlackScholes', or 'Mixed'
        entry_delta: Options delta at entry
        exit_delta: Options delta at exit
        entry_theta: Options theta at entry
        theta_cost: Estimated theta decay cost
        entry_iv: Implied volatility at entry
        exit_iv: Implied volatility at exit

        # Additional metadata
        continuation_bars: Number of continuation bars before entry
        mtf_alignment: Multi-timeframe alignment score (0-5)
        metadata: Additional custom metadata
    """
    # Required fields
    trade_id: int
    symbol: str
    pattern_type: str
    timeframe: str

    # Dates
    entry_date: datetime
    exit_date: datetime

    # Prices
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float

    # P/L
    pnl: float
    pnl_pct: float
    is_winner: bool

    # Optional trade details
    regime: str = 'UNKNOWN'
    days_held: int = 0
    hit_target: bool = False
    hit_stop: bool = False
    risk_reward_ratio: float = 0.0

    # Options-specific (Optional)
    is_options_trade: bool = False
    data_source: str = 'Synthetic'
    entry_delta: Optional[float] = None
    exit_delta: Optional[float] = None
    entry_theta: Optional[float] = None
    theta_cost: Optional[float] = None
    entry_iv: Optional[float] = None
    exit_iv: Optional[float] = None

    # Additional metadata
    continuation_bars: int = 0
    mtf_alignment: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Ensure days_held is calculated if not provided
        if self.days_held == 0 and self.entry_date and self.exit_date:
            delta = self.exit_date - self.entry_date
            self.days_held = max(1, delta.days)

        # Ensure is_winner matches pnl
        if self.pnl != 0:
            self.is_winner = self.pnl > 0

    @property
    def pattern_enum(self) -> PatternType:
        """Get PatternType enum from pattern_type string."""
        return PatternType.from_string(self.pattern_type)

    @property
    def is_bullish(self) -> bool:
        """Check if trade was bullish."""
        return self.pattern_enum.is_bullish()

    @property
    def is_bearish(self) -> bool:
        """Check if trade was bearish."""
        return self.pattern_enum.is_bearish()

    @property
    def base_pattern(self) -> str:
        """Get base pattern type without direction."""
        return self.pattern_enum.base_pattern()

    @property
    def risk_amount(self) -> float:
        """Calculate the risk amount (entry to stop distance)."""
        if self.is_bullish:
            return abs(self.entry_price - self.stop_price)
        else:
            return abs(self.stop_price - self.entry_price)

    @property
    def reward_amount(self) -> float:
        """Calculate the reward amount (entry to target distance)."""
        if self.is_bullish:
            return abs(self.target_price - self.entry_price)
        else:
            return abs(self.entry_price - self.target_price)

    @property
    def planned_risk_reward(self) -> float:
        """Calculate planned risk/reward ratio at entry."""
        risk = self.risk_amount
        if risk == 0:
            return 0.0
        return self.reward_amount / risk

    @property
    def actual_risk_reward(self) -> float:
        """Calculate actual risk/reward based on realized P/L."""
        risk = self.risk_amount
        if risk == 0 or self.pnl <= 0:
            return 0.0
        return abs(self.exit_price - self.entry_price) / risk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'pattern_type': self.pattern_type,
            'base_pattern': self.base_pattern,
            'timeframe': self.timeframe,
            'regime': self.regime,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'is_winner': self.is_winner,
            'days_held': self.days_held,
            'hit_target': self.hit_target,
            'hit_stop': self.hit_stop,
            'risk_reward_ratio': self.risk_reward_ratio,
            'planned_risk_reward': self.planned_risk_reward,
            'is_options_trade': self.is_options_trade,
            'data_source': self.data_source,
            'entry_delta': self.entry_delta,
            'exit_delta': self.exit_delta,
            'entry_theta': self.entry_theta,
            'theta_cost': self.theta_cost,
            'entry_iv': self.entry_iv,
            'exit_iv': self.exit_iv,
            'continuation_bars': self.continuation_bars,
            'mtf_alignment': self.mtf_alignment,
            'metadata': self.metadata,
        }


def create_trade_from_backtest_row(
    row: Dict[str, Any],
    trade_id: int,
    is_options: bool = False,
) -> PatternTradeResult:
    """
    Factory function to create PatternTradeResult from backtest DataFrame row.

    Args:
        row: Dictionary containing trade data (from DataFrame.to_dict('records'))
        trade_id: Unique identifier for this trade
        is_options: Whether this is an options trade

    Returns:
        PatternTradeResult instance

    Expected row keys:
        - symbol, pattern_type, timeframe, regime
        - entry_date, exit_date
        - entry_price, exit_price, stop_price, target_price
        - pnl, pnl_pct (or calculate from prices)
        - Optional: data_source, entry_delta, exit_delta, etc.
    """
    # Calculate pnl_pct if not provided
    entry_price = row.get('entry_price', 0)
    exit_price = row.get('exit_price', 0)
    pnl = row.get('pnl', exit_price - entry_price)
    pnl_pct = row.get('pnl_pct', pnl / entry_price if entry_price else 0)

    # Parse dates if strings
    entry_date = row.get('entry_date')
    exit_date = row.get('exit_date')
    if isinstance(entry_date, str):
        entry_date = datetime.fromisoformat(entry_date)
    if isinstance(exit_date, str):
        exit_date = datetime.fromisoformat(exit_date)

    return PatternTradeResult(
        trade_id=trade_id,
        symbol=row.get('symbol', 'UNKNOWN'),
        pattern_type=row.get('pattern_type', 'UNKNOWN'),
        timeframe=row.get('timeframe', '1D'),
        regime=row.get('regime', 'UNKNOWN'),
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=row.get('stop_price', 0),
        target_price=row.get('target_price', 0),
        pnl=pnl,
        pnl_pct=pnl_pct,
        is_winner=pnl > 0,
        days_held=row.get('days_held', 0),
        hit_target=row.get('hit_target', False),
        hit_stop=row.get('hit_stop', False),
        risk_reward_ratio=row.get('risk_reward_ratio', 0),
        is_options_trade=is_options or row.get('is_options_trade', False),
        data_source=row.get('data_source', 'ThetaData' if is_options else 'Synthetic'),
        entry_delta=row.get('entry_delta'),
        exit_delta=row.get('exit_delta'),
        entry_theta=row.get('entry_theta'),
        theta_cost=row.get('theta_cost'),
        entry_iv=row.get('entry_iv'),
        exit_iv=row.get('exit_iv'),
        continuation_bars=row.get('continuation_bars', 0),
        mtf_alignment=row.get('mtf_alignment', 0),
        metadata=row.get('metadata', {}),
    )


def create_trades_from_dataframe(
    df: 'pd.DataFrame',
    is_options: bool = False,
) -> List[PatternTradeResult]:
    """
    Convert a backtest results DataFrame to list of PatternTradeResult.

    Args:
        df: DataFrame with trade data
        is_options: Whether these are options trades

    Returns:
        List of PatternTradeResult instances
    """
    trades = []
    records = df.to_dict('records')

    for idx, row in enumerate(records):
        trade = create_trade_from_backtest_row(
            row=row,
            trade_id=idx + 1,
            is_options=is_options,
        )
        trades.append(trade)

    return trades
