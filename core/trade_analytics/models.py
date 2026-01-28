"""
Trade Analytics Data Models

Rich data models for comprehensive trade analysis.
Captures everything needed to answer questions like:
- "What's my win rate when VIX > 25?"
- "Is 1.5% magnitude working?"
- "How much profit am I leaving on table?"

Session: Trade Analytics Implementation
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json


class AssetClass(str, Enum):
    """Asset class for trade."""
    EQUITY_OPTION = "equity_option"
    CRYPTO_PERP = "crypto_perp"
    EQUITY = "equity"
    CRYPTO_SPOT = "crypto_spot"


class ExitReason(str, Enum):
    """Reason for trade exit."""
    TARGET = "TARGET"
    STOP = "STOP"
    TRAILING_STOP = "TRAILING_STOP"
    DTE_EXIT = "DTE_EXIT"
    MAX_LOSS = "MAX_LOSS"
    MAX_PROFIT = "MAX_PROFIT"
    PATTERN_INVALIDATED = "PATTERN_INVALIDATED"
    EOD_EXIT = "EOD_EXIT"
    TIME_EXIT = "TIME_EXIT"
    MANUAL = "MANUAL"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExcursionData:
    """
    Maximum Favorable/Adverse Excursion tracking.
    
    THE KEY INSIGHT: Tracks the best and worst P&L during a trade.
    This answers critical questions:
    - Did this loser go green first? (MFE > 0 for losers)
    - Am I leaving profit on table? (exit_pnl << MFE for winners)
    - Are my stops too tight? (MAE hits stop before MFE reached)
    """
    # Maximum Favorable Excursion (best P&L seen during trade)
    mfe_pnl: float = 0.0              # Dollar amount
    mfe_pct: float = 0.0              # Percentage of entry
    mfe_price: float = 0.0            # Price at MFE
    mfe_time: Optional[datetime] = None
    mfe_bars_from_entry: int = 0      # How many bars to reach MFE
    
    # Maximum Adverse Excursion (worst P&L seen during trade)
    mae_pnl: float = 0.0              # Dollar amount (negative)
    mae_pct: float = 0.0              # Percentage of entry (negative)
    mae_price: float = 0.0            # Price at MAE
    mae_time: Optional[datetime] = None
    mae_bars_from_entry: int = 0      # How many bars to reach MAE
    
    # Exit efficiency metrics
    exit_efficiency: float = 0.0       # exit_pnl / mfe_pnl (1.0 = perfect exit at MFE)
    profit_captured_pct: float = 0.0   # What % of available profit was captured
    went_green_before_loss: bool = False  # For losers: did it go positive first?
    
    # Price history (optional, for detailed analysis)
    price_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mfe_pnl': self.mfe_pnl,
            'mfe_pct': self.mfe_pct,
            'mfe_price': self.mfe_price,
            'mfe_time': self.mfe_time.isoformat() if self.mfe_time else None,
            'mfe_bars_from_entry': self.mfe_bars_from_entry,
            'mae_pnl': self.mae_pnl,
            'mae_pct': self.mae_pct,
            'mae_price': self.mae_price,
            'mae_time': self.mae_time.isoformat() if self.mae_time else None,
            'mae_bars_from_entry': self.mae_bars_from_entry,
            'exit_efficiency': self.exit_efficiency,
            'profit_captured_pct': self.profit_captured_pct,
            'went_green_before_loss': self.went_green_before_loss,
            # Don't include price_samples in serialization (too large)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExcursionData':
        """Create from dictionary."""
        return cls(
            mfe_pnl=data.get('mfe_pnl', 0.0),
            mfe_pct=data.get('mfe_pct', 0.0),
            mfe_price=data.get('mfe_price', 0.0),
            mfe_time=datetime.fromisoformat(data['mfe_time']) if data.get('mfe_time') else None,
            mfe_bars_from_entry=data.get('mfe_bars_from_entry', 0),
            mae_pnl=data.get('mae_pnl', 0.0),
            mae_pct=data.get('mae_pct', 0.0),
            mae_price=data.get('mae_price', 0.0),
            mae_time=datetime.fromisoformat(data['mae_time']) if data.get('mae_time') else None,
            mae_bars_from_entry=data.get('mae_bars_from_entry', 0),
            exit_efficiency=data.get('exit_efficiency', 0.0),
            profit_captured_pct=data.get('profit_captured_pct', 0.0),
            went_green_before_loss=data.get('went_green_before_loss', False),
        )


@dataclass
class PatternContext:
    """
    STRAT pattern context at trade entry.
    
    Captures the pattern that generated the signal.
    """
    pattern_type: str = ""            # "2-1-2U", "3-2D", "Rev Strat", etc.
    timeframe: str = ""               # "1H", "1D", "1W", "1M"
    signal_type: str = ""             # "SETUP" or "COMPLETED" (triggered)
    direction: str = ""               # "LONG", "SHORT", "CALL", "PUT"
    
    # Setup bar metrics
    magnitude_pct: float = 0.0        # Setup bar range as % of price
    setup_bar_high: float = 0.0
    setup_bar_low: float = 0.0
    
    # Entry bar metrics  
    entry_bar_type: str = ""          # "2U", "2D", "3"
    entry_bar_high: float = 0.0
    entry_bar_low: float = 0.0
    
    # TFC (Timeframe Continuity) at entry
    tfc_score: int = 0                # 0-5 alignment score
    tfc_alignment: str = ""           # "BULLISH", "BEARISH", "MIXED"
    tfc_details: Dict[str, str] = field(default_factory=dict)  # Per-TF breakdown
    
    # Rev Strat specific
    is_rev_strat: bool = False
    ftfc_direction: str = ""          # Full TFC direction if rev strat
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternContext':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketContext:
    """
    Market conditions at trade entry.
    
    For factor-based analysis:
    - "What's my win rate when VIX > 25?"
    - "How do I perform in bull vs bear regimes?"
    """
    # VIX context
    vix_level: float = 0.0
    vix_regime: str = ""              # "LOW", "ELEVATED", "HIGH", "EXTREME"
    vix_percentile: float = 0.0       # VIX percentile (0-100)
    vix_acceleration: bool = False    # VIX acceleration signal active?
    
    # Regime detection
    market_regime: str = ""           # "BULL", "BEAR", "NEUTRAL", "CRASH"
    regime_confidence: float = 0.0    # Jump model confidence
    
    # Broad market trend
    spy_trend: str = ""               # "BULLISH", "BEARISH", "NEUTRAL"
    spy_above_20sma: bool = False
    spy_above_50sma: bool = False
    
    # Sector context
    sector: str = ""                  # "XLK", "XLF", "XLE", etc.
    sector_relative_strength: float = 0.0  # vs SPY
    
    # Symbol-specific technicals
    atr_14: float = 0.0               # 14-period ATR
    atr_percent: float = 0.0          # ATR as % of price
    dollar_volume: float = 0.0        # Average dollar volume
    volume_ratio: float = 0.0         # Current vs average volume
    
    # Calendar context
    day_of_week: int = 0              # 0=Monday, 4=Friday
    hour_of_day: int = 0              # 0-23
    is_opex_week: bool = False        # Options expiration week
    days_to_earnings: Optional[int] = None  # Days to next earnings
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketContext':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PositionManagement:
    """
    Position management parameters and execution.
    
    For analyzing stop/target effectiveness.
    """
    # Original plan
    entry_trigger: float = 0.0        # Signal trigger price
    stop_price: float = 0.0           # Original stop
    target_price: float = 0.0         # Original target (magnitude)
    risk_reward_planned: float = 0.0  # Planned R:R
    
    # Actual execution
    actual_entry_price: float = 0.0   # Where we actually entered
    actual_stop_used: float = 0.0     # May differ if trailing
    actual_target_used: float = 0.0   # May differ if adjusted
    
    # Risk metrics
    distance_to_target_pct: float = 0.0
    distance_to_stop_pct: float = 0.0
    risk_per_trade: float = 0.0       # Dollar risk
    
    # Position sizing
    position_size: float = 0.0        # Shares/contracts
    notional_value: float = 0.0       # Total position value
    leverage_used: float = 1.0        # Leverage multiplier
    risk_multiplier: float = 1.0      # Regime-based sizing adjustment
    
    # Options-specific (if applicable)
    option_type: Optional[str] = None   # "CALL", "PUT"
    strike: Optional[float] = None
    dte_at_entry: Optional[int] = None
    dte_at_exit: Optional[int] = None
    delta_at_entry: Optional[float] = None
    delta_at_exit: Optional[float] = None
    iv_at_entry: Optional[float] = None
    iv_at_exit: Optional[float] = None
    theta_decay_actual: Optional[float] = None
    
    # Trailing stop state
    trailing_stop_activated: bool = False
    high_water_mark: float = 0.0      # Best price seen
    trailing_stop_final: float = 0.0  # Final trailing stop level
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionManagement':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EnrichedTradeRecord:
    """
    Complete trade record with all context for analysis.
    
    This is the central data model that captures EVERYTHING needed
    to learn from our trades.
    """
    # Identifiers
    trade_id: str
    symbol: str
    asset_class: str = AssetClass.EQUITY_OPTION.value
    
    # Timing
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pattern_detected_time: Optional[datetime] = None
    bars_in_trade: int = 0
    seconds_in_trade: int = 0
    
    # P&L
    pnl: float = 0.0                  # Net P&L after costs
    pnl_pct: float = 0.0              # P&L as percentage
    gross_pnl: float = 0.0            # P&L before costs
    total_costs: float = 0.0          # Fees + slippage + funding
    
    # Exit
    exit_reason: str = ExitReason.UNKNOWN.value
    exit_price: float = 0.0
    
    # Nested context objects
    pattern: PatternContext = field(default_factory=PatternContext)
    market: MarketContext = field(default_factory=MarketContext)
    position: PositionManagement = field(default_factory=PositionManagement)
    excursion: ExcursionData = field(default_factory=ExcursionData)
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Derived/computed fields (set after trade closes)
    is_winner: bool = False
    win_loss_category: str = ""       # "BIG_WIN", "SMALL_WIN", "BREAKEVEN", "SMALL_LOSS", "BIG_LOSS"
    
    def __post_init__(self):
        """Compute derived fields."""
        if self.pnl > 0:
            self.is_winner = True
        
        # Categorize win/loss magnitude
        if self.pnl_pct >= 50:
            self.win_loss_category = "BIG_WIN"
        elif self.pnl_pct >= 10:
            self.win_loss_category = "SMALL_WIN"
        elif self.pnl_pct >= -10:
            self.win_loss_category = "BREAKEVEN"
        elif self.pnl_pct >= -30:
            self.win_loss_category = "SMALL_LOSS"
        else:
            self.win_loss_category = "BIG_LOSS"
    
    def finalize_excursion(self) -> None:
        """
        Compute final excursion metrics after trade closes.
        
        Call this when trade is closed to calculate exit efficiency
        and other derived metrics.
        """
        exc = self.excursion
        
        # Exit efficiency: how much of max profit did we capture?
        if exc.mfe_pnl > 0:
            exc.exit_efficiency = self.pnl / exc.mfe_pnl
            exc.profit_captured_pct = min(100.0, max(0.0, exc.exit_efficiency * 100))
        
        # Did this loser go green first?
        if self.pnl <= 0 and exc.mfe_pnl > 0:
            exc.went_green_before_loss = True
        
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pattern_detected_time': self.pattern_detected_time.isoformat() if self.pattern_detected_time else None,
            'bars_in_trade': self.bars_in_trade,
            'seconds_in_trade': self.seconds_in_trade,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'gross_pnl': self.gross_pnl,
            'total_costs': self.total_costs,
            'exit_reason': self.exit_reason,
            'exit_price': self.exit_price,
            'pattern': self.pattern.to_dict(),
            'market': self.market.to_dict(),
            'position': self.position.to_dict(),
            'excursion': self.excursion.to_dict(),
            'notes': self.notes,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_winner': self.is_winner,
            'win_loss_category': self.win_loss_category,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedTradeRecord':
        """Create from dictionary."""
        record = cls(
            trade_id=data['trade_id'],
            symbol=data['symbol'],
            asset_class=data.get('asset_class', AssetClass.EQUITY_OPTION.value),
            entry_time=datetime.fromisoformat(data['entry_time']) if data.get('entry_time') else None,
            exit_time=datetime.fromisoformat(data['exit_time']) if data.get('exit_time') else None,
            pattern_detected_time=datetime.fromisoformat(data['pattern_detected_time']) if data.get('pattern_detected_time') else None,
            bars_in_trade=data.get('bars_in_trade', 0),
            seconds_in_trade=data.get('seconds_in_trade', 0),
            pnl=data.get('pnl', 0.0),
            pnl_pct=data.get('pnl_pct', 0.0),
            gross_pnl=data.get('gross_pnl', 0.0),
            total_costs=data.get('total_costs', 0.0),
            exit_reason=data.get('exit_reason', ExitReason.UNKNOWN.value),
            exit_price=data.get('exit_price', 0.0),
            notes=data.get('notes', ''),
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.utcnow(),
        )
        
        # Restore nested objects
        if 'pattern' in data:
            record.pattern = PatternContext.from_dict(data['pattern'])
        if 'market' in data:
            record.market = MarketContext.from_dict(data['market'])
        if 'position' in data:
            record.position = PositionManagement.from_dict(data['position'])
        if 'excursion' in data:
            record.excursion = ExcursionData.from_dict(data['excursion'])
        
        # Restore derived fields
        record.is_winner = data.get('is_winner', record.pnl > 0)
        record.win_loss_category = data.get('win_loss_category', '')
        
        return record
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EnrichedTradeRecord':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
