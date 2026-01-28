# ATLAS Trade Analytics Engine - Implementation Specification

## Overview

This document specifies the implementation of a comprehensive trade analytics system for ATLAS. The system tracks Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) for all trades, enabling data-driven strategy optimization.

**Problem Statement:** ATLAS currently provides only aggregate P&L metrics. We cannot answer critical questions like:
- "Did that 1H 2-1-2 go 150% of target before reversing and stopping out?"
- "What's my actual win rate on patterns detected when VIX > 25?"
- "Are 1.5% magnitude filters actually working, or would 2% be better?"
- "What's the average time-in-trade for winners vs. losers?"
- "Is my TFC 5/5 filter better than 4/5?"

**Solution:** Build a trade analytics engine that captures rich context at entry, tracks price excursions during the trade, and provides segmented analysis tools.

---

## Architecture

```
core/trade_analytics/
├── __init__.py           ✅ COMPLETE
├── models.py             ✅ COMPLETE  
├── excursion_tracker.py  ❌ TODO
├── analytics_engine.py   ❌ TODO
└── trade_store.py        ❌ TODO
```

### Completed Components

#### models.py
Defines the data structures for enriched trade records:

- **`EnrichedTradeRecord`** - Complete trade record with all context
- **`ExcursionData`** - MFE/MAE tracking (the critical missing piece)
- **`PatternContext`** - STRAT pattern details at entry
- **`MarketContext`** - VIX, regime, sector, technicals at entry
- **`PositionManagement`** - Stops, targets, sizing, options greeks

---

## Components To Implement

### 1. ExcursionTracker (`excursion_tracker.py`)

**Purpose:** Track MFE/MAE in real-time for open positions.

**Integration Point:** Called from position monitor loops (every 60s for equities, every 60s for crypto).

```python
class ExcursionTracker:
    """
    Track Maximum Favorable/Adverse Excursion for open trades.
    
    Usage:
        tracker = ExcursionTracker()
        
        # On each position monitor cycle:
        tracker.update(trade_id, current_price, entry_price, direction, timestamp)
        
        # When trade closes:
        excursion_data = tracker.finalize(trade_id, exit_pnl)
    """
    
    def __init__(self):
        """Initialize tracker with empty state."""
        # Dict[trade_id, ExcursionState]
        self._tracking: Dict[str, ExcursionState] = {}
    
    def start_tracking(
        self,
        trade_id: str,
        entry_price: float,
        direction: str,  # "LONG" or "SHORT" / "CALL" or "PUT"
        entry_time: datetime,
        position_size: float = 1.0,
    ) -> None:
        """
        Begin tracking excursions for a new trade.
        
        Call this when a trade is opened.
        """
        pass
    
    def update(
        self,
        trade_id: str,
        current_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update excursion tracking with current price.
        
        Call this on each position monitor cycle (e.g., every 60 seconds).
        
        Returns:
            Dict with current MFE/MAE if tracking, None if trade_id not found
            {
                'current_pnl': float,
                'current_pnl_pct': float,
                'mfe_pnl': float,
                'mfe_pct': float,
                'mae_pnl': float,
                'mae_pct': float,
            }
        """
        pass
    
    def finalize(
        self,
        trade_id: str,
        exit_price: float,
        exit_pnl: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[ExcursionData]:
        """
        Finalize excursion tracking when trade closes.
        
        Calculates exit efficiency and other derived metrics.
        Removes trade from active tracking.
        
        Returns:
            Complete ExcursionData object, or None if trade_id not found
        """
        pass
    
    def get_active_trades(self) -> List[str]:
        """Get list of trade_ids being tracked."""
        pass
    
    def get_current_excursion(self, trade_id: str) -> Optional[Dict[str, float]]:
        """Get current excursion state for a trade without updating."""
        pass


@dataclass
class ExcursionState:
    """Internal state for tracking a single trade's excursions."""
    trade_id: str
    entry_price: float
    direction: str
    position_size: float
    entry_time: datetime
    
    # Running extremes
    best_price: float = 0.0      # Best price seen (highest for long, lowest for short)
    worst_price: float = 0.0     # Worst price seen
    best_time: Optional[datetime] = None
    worst_time: Optional[datetime] = None
    
    # Sample count for bars calculation
    sample_count: int = 0
    best_sample_idx: int = 0
    worst_sample_idx: int = 0
    
    # Optional: store price history for detailed analysis
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)
```

**Key Implementation Notes:**
- Direction handling: For LONG/CALL, MFE is highest price; for SHORT/PUT, MFE is lowest price
- P&L calculation: `(current - entry) * size` for long, `(entry - current) * size` for short
- Percentage calculation: `pnl / (entry * size) * 100`
- Exit efficiency: `actual_exit_pnl / mfe_pnl` (capped at 1.0, can be negative)

---

### 2. TradeStore (`trade_store.py`)

**Purpose:** Persist enriched trade records and provide query interface.

```python
class TradeStore:
    """
    Persistent storage for enriched trade records.
    
    Stores trades as JSON with indexing for efficient queries.
    Supports both equity options and crypto perpetuals.
    
    Usage:
        store = TradeStore(data_dir="core/trade_analytics/data")
        
        # Add a completed trade
        store.add_trade(enriched_record)
        
        # Query trades
        trades = store.get_trades(
            symbol="SPY",
            timeframe="1H",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
        )
        
        # Get as DataFrame for analysis
        df = store.to_dataframe()
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize store with data directory."""
        pass
    
    def add_trade(self, trade: EnrichedTradeRecord) -> None:
        """Add a completed trade to the store."""
        pass
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing trade record."""
        pass
    
    def get_trade(self, trade_id: str) -> Optional[EnrichedTradeRecord]:
        """Get a single trade by ID."""
        pass
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        asset_class: Optional[str] = None,
        winners_only: bool = False,
        losers_only: bool = False,
    ) -> List[EnrichedTradeRecord]:
        """
        Query trades with optional filters.
        
        All filters are AND conditions.
        """
        pass
    
    def to_dataframe(
        self,
        trades: Optional[List[EnrichedTradeRecord]] = None,
        flatten: bool = True,
    ) -> 'pd.DataFrame':
        """
        Convert trades to pandas DataFrame for analysis.
        
        Args:
            trades: List of trades (uses all trades if None)
            flatten: If True, flatten nested objects into columns
                     (e.g., pattern.timeframe -> pattern_timeframe)
        
        Returns:
            DataFrame with one row per trade
        """
        pass
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all trades."""
        pass
    
    def _save(self) -> None:
        """Persist trades to disk."""
        pass
    
    def _load(self) -> None:
        """Load trades from disk."""
        pass
```

**Storage Format:**
```
core/trade_analytics/data/
├── trades.json           # All enriched trade records
├── trades_backup_YYYYMMDD.json  # Daily backups
└── indexes/
    ├── by_symbol.json    # trade_id lists by symbol
    ├── by_timeframe.json # trade_id lists by timeframe
    └── by_date.json      # trade_id lists by date
```

---

### 3. TradeAnalyticsEngine (`analytics_engine.py`)

**Purpose:** The main analysis interface. Provides segmented win rates, sensitivity analysis, and exit efficiency reports.

```python
class TradeAnalyticsEngine:
    """
    Analyze trade history to find what's actually working.
    
    This is the main interface for answering questions like:
    - "What's my win rate on hourly patterns?"
    - "Is 1.5% magnitude filter optimal?"
    - "What VIX level am I most profitable at?"
    
    Usage:
        engine = TradeAnalyticsEngine(trade_store)
        
        # Segmented analysis
        df = engine.win_rate_by_factor("timeframe")
        df = engine.win_rate_by_factor("vix_level", bins=[15, 20, 25, 30])
        
        # Exit efficiency
        report = engine.exit_efficiency_report()
        
        # Parameter sensitivity
        df = engine.magnitude_sensitivity()
        df = engine.tfc_sensitivity()
    """
    
    def __init__(self, store: TradeStore):
        """Initialize with trade store."""
        self.store = store
    
    # =========================================================================
    # SEGMENTED WIN RATE ANALYSIS
    # =========================================================================
    
    def win_rate_by_factor(
        self,
        factor: str,
        bins: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Calculate win rate segmented by any factor.
        
        Args:
            factor: Field to segment by. Supports:
                - "timeframe" (1H, 1D, 1W, 1M)
                - "pattern_type" (2-1-2U, 3-2D, etc.)
                - "vix_level" (requires bins)
                - "vix_regime" (LOW, ELEVATED, HIGH, EXTREME)
                - "atr_percent" (requires bins)
                - "magnitude_pct" (requires bins)
                - "tfc_score" (0-5)
                - "day_of_week" (0-4)
                - "hour_of_day" (0-23)
                - "exit_reason" (TARGET, STOP, etc.)
                - "market_regime" (BULL, BEAR, etc.)
                
            bins: For continuous factors, bin edges (e.g., [15, 20, 25, 30] for VIX)
            filters: Additional filters to apply before analysis
        
        Returns:
            DataFrame with columns:
                - {factor}: The factor value or bin
                - trades: Number of trades
                - wins: Number of winners
                - losses: Number of losers
                - win_rate: Win rate percentage
                - total_pnl: Sum of P&L
                - avg_pnl: Average P&L per trade
                - avg_winner: Average winning trade
                - avg_loser: Average losing trade
                - profit_factor: Gross profit / gross loss
                - avg_mfe: Average MFE across trades
                - avg_mae: Average MAE across trades
                - avg_exit_efficiency: Average exit efficiency
        
        Example:
            # Win rate by timeframe
            engine.win_rate_by_factor("timeframe")
            
            # Win rate by VIX buckets
            engine.win_rate_by_factor("vix_level", bins=[15, 20, 25, 30])
            
            # Win rate by magnitude, filtered to daily only
            engine.win_rate_by_factor(
                "magnitude_pct",
                bins=[1.0, 1.5, 2.0, 2.5, 3.0],
                filters={"timeframe": "1D"}
            )
        """
        pass
    
    def compare_factors(
        self,
        factor_a: str,
        factor_b: str,
        bins_a: Optional[List[float]] = None,
        bins_b: Optional[List[float]] = None,
    ) -> 'pd.DataFrame':
        """
        Cross-tabulate win rates by two factors.
        
        Example:
            # Win rate by timeframe AND VIX regime
            engine.compare_factors("timeframe", "vix_regime")
        
        Returns:
            Pivot table with factor_a as rows, factor_b as columns,
            values are win rates.
        """
        pass
    
    # =========================================================================
    # EXIT EFFICIENCY ANALYSIS
    # =========================================================================
    
    def exit_efficiency_report(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze how well we're capturing available profit.
        
        THE KEY INSIGHT: If MFE consistently 2x actual exit, we're leaving
        money on table. If MAE consistently hits stop before MFE, stops too tight.
        
        Returns:
            {
                'summary': {
                    'total_trades': int,
                    'avg_exit_efficiency': float,  # 1.0 = perfect
                    'profit_left_on_table': float,  # Sum of (MFE - actual) for winners
                    'could_have_avoided': float,  # Sum of losses that went green first
                },
                'winners': {
                    'count': int,
                    'avg_exit_efficiency': float,
                    'avg_mfe_pct': float,
                    'avg_actual_profit_pct': float,
                    'profit_left_on_table': float,
                    'mfe_vs_target': float,  # How often MFE exceeded target
                },
                'losers': {
                    'count': int,
                    'avg_mfe_before_loss': float,  # Did losers go green?
                    'pct_went_green_first': float,  # % of losers that had positive MFE
                    'avg_mae_pct': float,
                    'avg_actual_loss_pct': float,
                    'mae_vs_stop': float,  # How often MAE hit stop exactly
                },
                'insights': [
                    "X% of losers went green first - consider tighter trailing stops",
                    "Average exit efficiency is Y% - room for improvement",
                    "Leaving $Z on table per trade on average",
                ],
            }
        """
        pass
    
    def mfe_distribution(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze MFE distribution to inform target placement.
        
        Returns:
            {
                'percentiles': {10: x, 25: x, 50: x, 75: x, 90: x},
                'histogram': [(bin_start, bin_end, count), ...],
                'recommendation': "Based on MFE distribution, consider target at X%",
            }
        """
        pass
    
    def mae_distribution(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze MAE distribution to inform stop placement.
        
        Returns:
            {
                'percentiles': {10: x, 25: x, 50: x, 75: x, 90: x},
                'histogram': [(bin_start, bin_end, count), ...],
                'recommendation': "Based on MAE distribution, consider stop at X%",
            }
        """
        pass
    
    # =========================================================================
    # PARAMETER SENSITIVITY ANALYSIS
    # =========================================================================
    
    def magnitude_sensitivity(
        self,
        thresholds: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Answer: "Is 1.5% magnitude filter optimal?"
        
        Simulates different magnitude thresholds on historical trades.
        Shows what results WOULD have been with different filters.
        
        Args:
            thresholds: Magnitude thresholds to test
                        Default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            filters: Additional filters
        
        Returns:
            DataFrame with columns:
                - magnitude_threshold: The threshold tested
                - trades_included: How many trades pass this threshold
                - trades_excluded: How many trades filtered out
                - win_rate: Win rate for included trades
                - avg_pnl: Average P&L for included trades
                - total_pnl: Total P&L for included trades
                - sharpe: Sharpe ratio approximation
        """
        pass
    
    def tfc_sensitivity(
        self,
        min_scores: Optional[List[int]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Answer: "Is TFC 4/5 better than 3/5?"
        
        Args:
            min_scores: TFC score thresholds to test
                        Default: [0, 1, 2, 3, 4, 5]
            filters: Additional filters
        
        Returns:
            DataFrame with columns:
                - min_tfc_score: The minimum score tested
                - trades_included: How many trades pass
                - win_rate: Win rate
                - avg_pnl: Average P&L
                - profit_factor: Profit factor
        """
        pass
    
    def stop_distance_sensitivity(
        self,
        multipliers: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Answer: "Should my stops be tighter or wider?"
        
        Uses MAE data to simulate different stop distances.
        
        Args:
            multipliers: Stop distance multipliers to test
                         Default: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                         1.0 = current stop distance
            filters: Additional filters
        
        Returns:
            DataFrame showing simulated results at each stop level
        """
        pass
    
    # =========================================================================
    # TIME-BASED ANALYSIS
    # =========================================================================
    
    def time_in_trade_analysis(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze relationship between time in trade and outcomes.
        
        Returns:
            {
                'avg_bars_winners': float,
                'avg_bars_losers': float,
                'avg_time_to_mfe': float,
                'avg_time_to_mae': float,
                'optimal_hold_time': str,  # Recommendation
            }
        """
        pass
    
    def performance_by_time(
        self,
        granularity: str = "hour",  # "hour", "day_of_week", "month"
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Show performance by time period.
        
        Example: "Am I more profitable in morning or afternoon?"
        """
        pass
    
    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================
    
    def pattern_performance(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Compare performance across pattern types.
        
        Returns:
            DataFrame with win rate, avg P&L, count per pattern type
        """
        pass
    
    def best_performing_setups(
        self,
        min_trades: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> 'pd.DataFrame':
        """
        Find the best performing setup combinations.
        
        Looks at pattern + timeframe + VIX regime combinations.
        
        Returns:
            DataFrame sorted by profit factor, filtered to min_trades
        """
        pass
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================
    
    def generate_daily_report(
        self,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate daily performance summary."""
        pass
    
    def generate_weekly_report(
        self,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate weekly performance summary with insights."""
        pass
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report.
        
        Includes:
        - Overall statistics
        - Win rate by major factors
        - Exit efficiency analysis
        - Parameter sensitivity
        - Top insights and recommendations
        """
        pass
```

---

## Integration Points

### Equity Options (ATLAS Equities)

**File:** `strat/signal_automation/position_monitor.py`

**Integration Steps:**
1. Import ExcursionTracker in PositionMonitor
2. Call `tracker.start_tracking()` in `_create_tracked_position()`
3. Call `tracker.update()` in `_check_position()` (runs every 60s)
4. Call `tracker.finalize()` when executing exit
5. Create EnrichedTradeRecord from TrackedPosition + ExcursionData
6. Save to TradeStore

**Existing fields to map:**
- `TrackedPosition.pattern_type` → `PatternContext.pattern_type`
- `TrackedPosition.timeframe` → `PatternContext.timeframe`
- `TrackedPosition.tfc_score` → `PatternContext.tfc_score` (needs to be added)
- `TrackedPosition.intrabar_high/low` → Already tracking, use for excursions
- `TrackedPosition.actual_entry_underlying` → `PositionManagement.actual_entry_price`
- `TrackedPosition.high_water_mark` → `ExcursionData.mfe_price`

**Additional data to capture at entry:**
- VIX level (fetch from market data)
- SPY trend (calculate from SPY data)
- ATR percent (from signal detection)
- Market regime (from regime detector if available)

### Crypto Perpetuals (ATLAS Crypto)

**File:** `crypto/simulation/position_monitor.py`

**Integration Steps:**
1. Import ExcursionTracker in CryptoPositionMonitor
2. Call `tracker.start_tracking()` when trade opens
3. Call `tracker.update()` in `_check_trade_exit()` loop
4. Call `tracker.finalize()` when executing exit
5. Create EnrichedTradeRecord from SimulatedTrade + ExcursionData
6. Save to TradeStore

**Existing fields in SimulatedTrade:**
- `intrabar_high/low` - Already tracking!
- `pattern_type`, `timeframe`, `tfc_score` - Already captured
- `entry_bar_high/low` - For pattern invalidation
- Fees and costs - Already tracked

---

## Example Usage (After Implementation)

```python
from core.trade_analytics import TradeAnalyticsEngine, TradeStore

# Initialize
store = TradeStore()
engine = TradeAnalyticsEngine(store)

# Question: "What's my win rate on hourly patterns?"
df = engine.win_rate_by_factor("timeframe")
print(df[df['timeframe'] == '1H'])

# Question: "Is 1.5% magnitude filter optimal?"
df = engine.magnitude_sensitivity()
print(df)

# Question: "Win rate by VIX level?"
df = engine.win_rate_by_factor("vix_level", bins=[15, 20, 25, 30])
print(df)

# Question: "How much profit am I leaving on table?"
report = engine.exit_efficiency_report()
print(f"Average exit efficiency: {report['summary']['avg_exit_efficiency']:.1%}")
print(f"Profit left on table: ${report['summary']['profit_left_on_table']:.2f}")

# Question: "Should I use trailing stops?"
# Look at losers that went green first
print(f"{report['losers']['pct_went_green_first']:.1%} of losers went green first")

# Question: "What's my optimal TFC threshold?"
df = engine.tfc_sensitivity()
print(df)

# Generate full report
full_report = engine.generate_full_report()
```

---

## Testing Strategy

1. **Unit tests for ExcursionTracker:**
   - Test long position MFE/MAE calculation
   - Test short position MFE/MAE calculation
   - Test exit efficiency calculation
   - Test "went green before loss" detection

2. **Unit tests for TradeStore:**
   - Test CRUD operations
   - Test query filters
   - Test DataFrame conversion

3. **Unit tests for TradeAnalyticsEngine:**
   - Test win_rate_by_factor with mock data
   - Test binning for continuous factors
   - Test sensitivity analysis

4. **Integration tests:**
   - Test with real trade history from paper_trades.json
   - Verify excursion tracking in live position monitor

---

## Migration Path

To populate TradeStore with historical data:

1. Parse existing `paper_trades/paper_trades.json`
2. For each trade, create EnrichedTradeRecord with available fields
3. MFE/MAE will be missing for historical trades (set to None/0)
4. Going forward, all new trades will have full excursion data

```python
def migrate_paper_trades(paper_trades_path: Path, store: TradeStore):
    """Migrate existing paper trades to enriched format."""
    with open(paper_trades_path) as f:
        trades = json.load(f)
    
    for trade in trades:
        record = EnrichedTradeRecord(
            trade_id=trade['trade_id'],
            symbol=trade['symbol'],
            # ... map other fields
            # Note: excursion data will be empty for historical trades
        )
        store.add_trade(record)
```

---

## Priority Order

1. **ExcursionTracker** - Core MFE/MAE tracking (highest value)
2. **TradeStore** - Persistence layer
3. **TradeAnalyticsEngine** - Analysis methods
4. **Integration** - Wire into position monitors
5. **Migration** - Import historical trades

---

## References

- Transcript: `/mnt/transcripts/2026-01-27-21-57-13-atlas-sentiment-rag-trade-analytics.txt`
- Existing models: `core/trade_analytics/models.py` (already implemented)
- Equity position monitor: `strat/signal_automation/position_monitor.py`
- Crypto position monitor: `crypto/simulation/position_monitor.py`
- Crypto paper trader: `crypto/simulation/paper_trader.py`
