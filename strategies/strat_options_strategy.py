"""
STRAT Options Strategy - Implements StrategyProtocol for ValidationRunner

Session 83K: Creates a strategy wrapper that bridges STRAT pattern detection
with the ATLAS validation framework. Implements backtest(), optimize(), and
generate_signals() methods required by StrategyProtocol.

Features:
- Supports all 5 STRAT pattern types (3-1-2, 2-1-2, 2-2, 3-2, 3-2-2)
- Options backtesting with explicit timestamp tracking
- Integration with Tier1Detector and OptionsBacktester
- Configurable pattern types, timeframes, and parameters
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from validation.protocols import BacktestResult
from strat.trade_execution_log import (
    TradeExecutionLog,
    TradeExecutionRecord,
    ExitReason,
)
from strat.tier1_detector import (
    Tier1Detector,
    PatternSignal,
    PatternType,
    Timeframe,
)
from strat.bar_classifier import classify_bars_nb
from strat.pattern_detector import (
    detect_312_patterns_nb,
    detect_212_patterns_nb,
    detect_22_patterns_nb,
    detect_32_patterns_nb,
    detect_322_patterns_nb,
)

# Import options module components
try:
    from strat.options_module import (
        OptionsExecutor,
        OptionsBacktester,
        OptionTrade,
    )
    OPTIONS_AVAILABLE = True
except ImportError:
    OPTIONS_AVAILABLE = False


@dataclass
class STRATOptionsConfig:
    """Configuration for STRAT Options Strategy."""
    # Pattern configuration
    pattern_types: List[str] = field(default_factory=lambda: ['3-1-2', '2-1-2', '2-2', '3-2', '3-2-2'])
    timeframe: str = '1W'  # '1D', '1W', '1M'
    min_continuation_bars: int = 2
    include_22_down: bool = False  # Session 69: 2-2 Down has negative expectancy

    # Options configuration
    dte_daily: int = 21
    dte_weekly: int = 35
    dte_monthly: int = 75
    default_iv: float = 0.20

    # Symbol
    symbol: str = 'SPY'

    def get_timeframe_enum(self) -> Timeframe:
        """Convert string timeframe to Timeframe enum."""
        # Session 83K-33: Added '1H' mapping - was MISSING, causing hourly
        # time filters to never activate (fell back to WEEKLY default)
        mapping = {
            '1H': Timeframe.HOURLY,
            '1D': Timeframe.DAILY,
            '1W': Timeframe.WEEKLY,
            '1M': Timeframe.MONTHLY,
        }
        return mapping.get(self.timeframe, Timeframe.WEEKLY)


class STRATOptionsStrategy:
    """
    STRAT pattern-based options strategy implementing StrategyProtocol.

    This class bridges STRAT pattern detection with the ATLAS validation
    framework by implementing the three required methods:
    - backtest(data, params) -> BacktestResult
    - optimize(data, param_grid) -> (best_params, BacktestResult)
    - generate_signals(data, params) -> DataFrame

    Attributes:
        config: STRATOptionsConfig instance
    """

    def __init__(self, config: Optional[STRATOptionsConfig] = None):
        """
        Initialize the strategy.

        Args:
            config: Configuration for the strategy. Uses defaults if None.
        """
        self.config = config or STRATOptionsConfig()

        # Initialize detector with min continuation bars
        # Note: Tier1Detector enforces min 2 continuation bars
        self._detector = Tier1Detector(
            min_continuation_bars=self.config.min_continuation_bars,
            include_22_down=self.config.include_22_down
        )

        # Initialize options components if available
        if OPTIONS_AVAILABLE:
            self._executor = OptionsExecutor()
            self._backtester = OptionsBacktester(default_iv=self.config.default_iv)
        else:
            self._executor = None
            self._backtester = None

    def backtest(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run options backtest on historical data.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            params: Optional strategy parameters to override config

        Returns:
            BacktestResult with trades DataFrame including explicit timestamps
        """
        # Apply parameters
        effective_config = self._apply_params(params)

        # Detect patterns
        signals = self._detect_all_patterns(data, effective_config)

        if not signals:
            return self._empty_result(data)

        # Generate option trades
        if not OPTIONS_AVAILABLE or self._executor is None:
            return self._equity_backtest(data, signals, effective_config)

        # Options backtest
        return self._options_backtest(data, signals, effective_config)

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters on training data.

        Args:
            data: OHLCV DataFrame for parameter optimization
            param_grid: Parameter grid to search. If None, uses default.

        Returns:
            Tuple of (best_params, best_backtest_result)
        """
        if param_grid is None:
            param_grid = {
                'min_continuation_bars': [2, 3],
            }

        best_sharpe = float('-inf')
        best_params = {}
        best_result = None

        # Generate all parameter combinations
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in product(*values):
            params = dict(zip(keys, combo))
            result = self.backtest(data, params)

            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = params.copy()
                best_result = result

        if best_result is None:
            best_result = self._empty_result(data)

        return best_params, best_result

    def generate_signals(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals without backtesting.

        Args:
            data: OHLCV DataFrame
            params: Optional strategy parameters

        Returns:
            DataFrame with entry, exit, direction, pattern_type columns
        """
        effective_config = self._apply_params(params)

        # Initialize output DataFrame
        result = pd.DataFrame(index=data.index)
        result['entry'] = False
        result['exit'] = False
        result['direction'] = 0
        result['stop'] = np.nan
        result['target'] = np.nan
        result['pattern_type'] = ''
        # Session 83K-3 BUG FIX: Initialize with same dtype as index to avoid FutureWarning
        # about incompatible datetime dtype when setting timezone-aware timestamps
        result['pattern_timestamp'] = pd.Series([pd.NaT] * len(data), index=data.index, dtype=data.index.dtype)

        # Detect patterns
        signals = self._detect_all_patterns(data, effective_config)

        # Populate signals
        for signal in signals:
            if signal.timestamp in result.index:
                idx = signal.timestamp
                result.loc[idx, 'entry'] = True
                result.loc[idx, 'direction'] = signal.direction
                result.loc[idx, 'stop'] = signal.stop_price
                result.loc[idx, 'target'] = signal.target_price
                result.loc[idx, 'pattern_type'] = signal.pattern_type.value
                result.loc[idx, 'pattern_timestamp'] = signal.timestamp

        return result

    def _apply_params(self, params: Optional[Dict[str, Any]]) -> STRATOptionsConfig:
        """Apply parameter overrides to create effective config."""
        if params is None:
            return self.config

        # Create copy of config
        config = STRATOptionsConfig(
            pattern_types=params.get('pattern_types', self.config.pattern_types),
            timeframe=params.get('timeframe', self.config.timeframe),
            min_continuation_bars=params.get('min_continuation_bars', self.config.min_continuation_bars),
            include_22_down=params.get('include_22_down', self.config.include_22_down),
            dte_daily=params.get('dte_daily', self.config.dte_daily),
            dte_weekly=params.get('dte_weekly', self.config.dte_weekly),
            dte_monthly=params.get('dte_monthly', self.config.dte_monthly),
            default_iv=params.get('default_iv', self.config.default_iv),
            symbol=params.get('symbol', self.config.symbol),
        )
        return config

    def _detect_all_patterns(
        self,
        data: pd.DataFrame,
        config: STRATOptionsConfig
    ) -> List[PatternSignal]:
        """
        Detect all configured pattern types.

        Uses Tier1Detector for 3-1-2, 2-1-2, 2-2 and direct detection
        functions for 3-2 and 3-2-2.
        """
        all_signals = []

        # Normalize column names
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        # Check required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Classify bars
        classifications = classify_bars_nb(
            df['high'].values,
            df['low'].values
        )

        timeframe = config.get_timeframe_enum()

        # Detect each pattern type
        for pattern_type in config.pattern_types:
            if pattern_type in ['3-1-2', '312']:
                signals = self._detect_312(df, classifications, timeframe)
                all_signals.extend(signals)

            elif pattern_type in ['2-1-2', '212']:
                signals = self._detect_212(df, classifications, timeframe)
                all_signals.extend(signals)

            elif pattern_type in ['2-2', '22']:
                signals = self._detect_22(df, classifications, timeframe, config.include_22_down)
                all_signals.extend(signals)

            elif pattern_type in ['3-2', '32']:
                signals = self._detect_32(df, classifications, timeframe)
                all_signals.extend(signals)

            elif pattern_type in ['3-2-2', '322']:
                signals = self._detect_322(df, classifications, timeframe)
                all_signals.extend(signals)

        # Apply continuation bar filter
        filtered = self._apply_continuation_filter(
            all_signals, df, classifications, config.min_continuation_bars
        )

        # Session 83K-38: Apply hourly time filter (STRAT rules)
        # 2-2 patterns: NOT before 10:30 ET
        # 3-bar patterns: NOT before 11:30 ET
        filtered = self._apply_hourly_time_filter(filtered, timeframe)

        return filtered

    def _apply_hourly_time_filter(self, signals: List[PatternSignal], timeframe: Timeframe) -> List[PatternSignal]:
        """
        Session 83K-38: Filter hourly signals by STRAT time rules.
        
        CRITICAL: Without this filter, patterns are entered at invalid times:
        - 2-2 patterns: NOT before 10:30 ET (need 2 bars to form)
        - 3-bar patterns (3-1-2, 2-1-2, 3-2, 3-2-2): NOT before 11:30 ET
        
        The first bar of the day is 09:30. Patterns need time to form:
        - 2-bar pattern (2-2): Can form after 2nd bar (10:30)
        - 3-bar pattern: Can form after 3rd bar (11:30)
        """
        if timeframe != Timeframe.HOURLY:
            return signals
        
        filtered = []
        for signal in signals:
            ts = signal.timestamp
            if not hasattr(ts, 'hour'):
                filtered.append(signal)
                continue
                
            hour, minute = ts.hour, ts.minute
            signal_minutes = hour * 60 + minute
            
            # Determine pattern type
            ptype = signal.pattern_type.value if hasattr(signal.pattern_type, 'value') else str(signal.pattern_type)
            
            # 2-2 patterns: NOT before 10:30 ET (630 minutes from midnight)
            if '2D-2U' in ptype or '2U-2D' in ptype or ('2-2' in ptype and '3-2-2' not in ptype):
                first_entry_minutes = 10 * 60 + 30  # 10:30
            else:
                # 3-bar patterns: NOT before 11:30 ET (690 minutes from midnight)
                first_entry_minutes = 11 * 60 + 30  # 11:30
            
            if signal_minutes >= first_entry_minutes:
                filtered.append(signal)
        
        return filtered

    def _detect_312(self, df, classifications, timeframe) -> List[PatternSignal]:
        """Detect 3-1-2 patterns."""
        entries, stops, targets, directions = detect_312_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
        return self._to_signals(df, entries, stops, targets, directions, '312', timeframe)

    def _detect_212(self, df, classifications, timeframe) -> List[PatternSignal]:
        """Detect 2-1-2 patterns."""
        entries, stops, targets, directions = detect_212_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
        return self._to_signals(df, entries, stops, targets, directions, '212', timeframe)

    def _detect_22(self, df, classifications, timeframe, include_down: bool) -> List[PatternSignal]:
        """Detect 2-2 patterns."""
        entries, stops, targets, directions = detect_22_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
        signals = self._to_signals(df, entries, stops, targets, directions, '22', timeframe)

        # Filter out 2-2 Down unless explicitly included
        if not include_down:
            signals = [s for s in signals if s.direction == 1]

        return signals

    def _detect_32(self, df, classifications, timeframe) -> List[PatternSignal]:
        """Detect 3-2 patterns."""
        entries, stops, targets, directions = detect_32_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
        return self._to_signals(df, entries, stops, targets, directions, '32', timeframe)

    def _detect_322(self, df, classifications, timeframe) -> List[PatternSignal]:
        """Detect 3-2-2 patterns."""
        entries, stops, targets, directions = detect_322_patterns_nb(
            classifications, df['high'].values, df['low'].values
        )
        return self._to_signals(df, entries, stops, targets, directions, '322', timeframe)

    def _to_signals(
        self,
        df: pd.DataFrame,
        entries: np.ndarray,
        stops: np.ndarray,
        targets: np.ndarray,
        directions: np.ndarray,
        pattern_base: str,
        timeframe: Timeframe
    ) -> List[PatternSignal]:
        """Convert detection arrays to PatternSignal objects."""
        signals = []

        for i in range(len(entries)):
            if entries[i]:
                direction = int(directions[i])

                # Determine pattern type enum
                if pattern_base == '312':
                    ptype = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN
                elif pattern_base == '212':
                    ptype = PatternType.PATTERN_212_UP if direction == 1 else PatternType.PATTERN_212_DOWN
                elif pattern_base == '22':
                    ptype = PatternType.PATTERN_22_UP if direction == 1 else PatternType.PATTERN_22_DOWN
                elif pattern_base == '32':
                    # Session 83K-38: Use proper 3-2 pattern types
                    ptype = PatternType.PATTERN_32_UP if direction == 1 else PatternType.PATTERN_32_DOWN
                elif pattern_base == '322':
                    # Session 83K-38: Use proper 3-2-2 pattern types
                    ptype = PatternType.PATTERN_322_UP if direction == 1 else PatternType.PATTERN_322_DOWN
                else:
                    ptype = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN

                # Get entry price (inside bar high/low for most patterns)
                entry_price = float(df['high'].iloc[i-1] if direction == 1 else df['low'].iloc[i-1])

                signal = PatternSignal(
                    pattern_type=ptype,
                    direction=direction,
                    entry_price=entry_price,
                    stop_price=float(stops[i]),
                    target_price=float(targets[i]),
                    timestamp=df.index[i],
                    timeframe=timeframe,
                )
                signals.append(signal)

        return signals

    def _apply_continuation_filter(
        self,
        signals: List[PatternSignal],
        df: pd.DataFrame,
        classifications: np.ndarray,
        min_bars: int
    ) -> List[PatternSignal]:
        """
        Count continuation bars for signals (analytics only - no filtering).

        Session 83K-8: Continuation bars are now EXIT logic, not ENTRY filter.
        All signals are returned. The continuation_bars field is populated
        for analytics/DTE selection, but no signals are rejected.
        """
        result = []

        for signal in signals:
            try:
                idx = df.index.get_loc(signal.timestamp)
            except KeyError:
                continue

            # Count continuation bars
            count = 0
            lookforward = min(5, len(df) - idx - 1)

            for j in range(1, lookforward + 1):
                bar_class = classifications[idx + j] if idx + j < len(classifications) else 0

                # Count directional bars in pattern direction
                if signal.direction == 1 and bar_class == 2:
                    count += 1
                elif signal.direction == -1 and bar_class == -2:
                    count += 1
                # Break on reversal bars (opposite direction)
                elif signal.direction == 1 and bar_class == -2:
                    break  # Reversal (2D) for bullish pattern
                elif signal.direction == -1 and bar_class == 2:
                    break  # Reversal (2U) for bearish pattern
                # Break on outside bars (exhaustion signal)
                elif bar_class == 3:
                    break
                # Inside bars (1) - continue without counting or breaking

            signal.continuation_bars = count
            # is_filtered indicates whether pattern meets analytics threshold
            # (kept for backward compatibility but no longer used for filtering)
            signal.is_filtered = count >= min_bars

            # Session 83K-8: Return ALL signals (no filtering)
            result.append(signal)

        return result

    def _options_backtest(
        self,
        data: pd.DataFrame,
        signals: List[PatternSignal],
        config: STRATOptionsConfig
    ) -> BacktestResult:
        """Run options backtest with explicit timestamp tracking."""
        # Get underlying price for options calculations
        close_col = 'close' if 'close' in data.columns else 'Close'
        underlying_price = data[close_col].iloc[-1]

        # Generate option trades
        trades = self._executor.generate_option_trades(
            signals=signals,
            underlying=config.symbol,
            underlying_price=underlying_price,
        )

        if not trades:
            return self._empty_result(data)

        # Run backtest
        backtest_df = self._backtester.backtest_trades(trades, data)

        if backtest_df.empty:
            return self._empty_result(data)

        # Create TradeExecutionLog with explicit timestamps
        execution_log = self._create_execution_log(trades, backtest_df, signals, config)

        # Convert to BacktestResult format
        trades_df = execution_log.to_dataframe()

        # Session 83K-27: Preserve magnitude_pct from backtest_df (lost in TradeExecutionLog conversion)
        if 'magnitude_pct' in backtest_df.columns and len(trades_df) == len(backtest_df):
            trades_df['magnitude_pct'] = backtest_df['magnitude_pct'].values

        return self._create_backtest_result(trades_df, data)

    def _equity_backtest(
        self,
        data: pd.DataFrame,
        signals: List[PatternSignal],
        config: STRATOptionsConfig
    ) -> BacktestResult:
        """Fallback equity backtest when options not available."""
        execution_log = TradeExecutionLog()

        close_col = 'close' if 'close' in data.columns else 'Close'
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'

        for i, signal in enumerate(signals):
            try:
                pattern_idx = data.index.get_loc(signal.timestamp)
            except KeyError:
                continue

            # Simulate entry/exit
            entry_hit = False
            entry_idx = None
            entry_price = None
            exit_price = None
            exit_reason = ExitReason.UNKNOWN.value
            exit_idx = None

            for j in range(pattern_idx + 1, min(pattern_idx + 30, len(data))):
                row = data.iloc[j]
                high = row[high_col]
                low = row[low_col]

                if not entry_hit:
                    # Session 83K-33: For hourly, skip entries on the 15:00 bar (no time to exit)
                    if config.timeframe == '1H':
                        bar_time = data.index[j]
                        if hasattr(bar_time, 'hour') and bar_time.hour >= 15:
                            continue  # Skip entry on last bar of day

                    if signal.direction == 1 and high >= signal.entry_price:
                        entry_hit = True
                        entry_idx = j
                        entry_price = signal.entry_price
                    elif signal.direction == -1 and low <= signal.entry_price:
                        entry_hit = True
                        entry_idx = j
                        entry_price = signal.entry_price
                    continue

                # Check exit
                if signal.direction == 1:
                    if high >= signal.target_price:
                        exit_price = signal.target_price
                        exit_reason = ExitReason.TARGET.value
                        exit_idx = j
                        break
                    elif low <= signal.stop_price:
                        exit_price = signal.stop_price
                        exit_reason = ExitReason.STOP.value
                        exit_idx = j
                        break
                else:
                    if low <= signal.target_price:
                        exit_price = signal.target_price
                        exit_reason = ExitReason.TARGET.value
                        exit_idx = j
                        break
                    elif high >= signal.stop_price:
                        exit_price = signal.stop_price
                        exit_reason = ExitReason.STOP.value
                        exit_idx = j
                        break

                # Session 83K-33: Hourly forced exit - no overnight holds
                # For hourly timeframe, must close by 15:30 ET (15:00 bar is last tradeable)
                # Also force exit if we've crossed into a new day (overnight hold)
                if config.timeframe == '1H':
                    bar_time = data.index[j]
                    entry_time = data.index[entry_idx]

                    # Check end of day (15:00+ bar) or next day
                    is_end_of_day = hasattr(bar_time, 'hour') and bar_time.hour >= 15
                    is_next_day = hasattr(bar_time, 'date') and hasattr(entry_time, 'date') and bar_time.date() > entry_time.date()

                    if is_end_of_day or is_next_day:
                        close_col = 'close' if 'close' in data.columns else 'Close'
                        exit_price = row[close_col] if close_col in row.index else row.get('close', high)
                        exit_reason = ExitReason.TIME_EXIT.value
                        exit_idx = j
                        break

            if entry_hit and exit_price is not None and entry_idx is not None and exit_idx is not None:
                # Calculate P/L
                if signal.direction == 1:
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price
                pnl_pct = pnl / entry_price if entry_price else 0

                record = TradeExecutionRecord(
                    trade_id=i + 1,
                    symbol=config.symbol,
                    pattern_type=signal.pattern_type.value,
                    timeframe=config.timeframe,
                    pattern_timestamp=signal.timestamp,
                    entry_timestamp=data.index[entry_idx],
                    exit_timestamp=data.index[exit_idx],
                    exit_reason=exit_reason,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                    pnl=pnl * 100,  # Simulate 100 shares
                    pnl_pct=pnl_pct,
                    direction=signal.direction,
                    continuation_bars=signal.continuation_bars,
                )
                execution_log.add_record(record)

        trades_df = execution_log.to_dataframe()
        return self._create_backtest_result(trades_df, data)

    def _create_execution_log(
        self,
        trades: List,
        backtest_df: pd.DataFrame,
        signals: List[PatternSignal],
        config: STRATOptionsConfig
    ) -> TradeExecutionLog:
        """Create execution log from backtest results."""
        log = TradeExecutionLog()

        for idx, row in backtest_df.iterrows():
            # Find matching signal
            signal = self._find_signal(row, signals)

            # Determine exit reason
            # Session 83K-33: Added TIME_EXIT handling for hourly forced exits
            exit_type = row.get('exit_type', 'UNKNOWN')
            if exit_type == 'TARGET':
                exit_reason = ExitReason.TARGET.value
            elif exit_type == 'STOP':
                exit_reason = ExitReason.STOP.value
            elif exit_type == 'TIME_EXIT':
                exit_reason = ExitReason.TIME_EXIT.value
            elif exit_type == 'REJECTED':
                exit_reason = ExitReason.REJECTED.value
            else:
                exit_reason = ExitReason.UNKNOWN.value

            record = TradeExecutionRecord(
                trade_id=idx + 1 if isinstance(idx, int) else hash(str(idx)) % 10000,
                symbol=config.symbol,
                pattern_type=row.get('pattern_type', signal.pattern_type.value if signal else 'UNKNOWN'),
                timeframe=config.timeframe,
                pattern_timestamp=row.get('timestamp', signal.timestamp if signal else datetime.now()),
                entry_timestamp=row.get('entry_date', row.get('timestamp', datetime.now())),
                exit_timestamp=row.get('exit_date', datetime.now()),
                exit_reason=exit_reason,
                entry_price=row.get('entry_price_underlying', signal.entry_price if signal else 0),
                exit_price=row.get('exit_price', 0),
                stop_price=signal.stop_price if signal else 0,
                target_price=signal.target_price if signal else 0,
                strike=row.get('strike', 0),
                option_type='CALL' if signal and signal.direction == 1 else 'PUT',
                osi_symbol=row.get('osi_symbol', ''),
                option_entry_price=row.get('entry_option_price', 0),
                option_exit_price=row.get('exit_option_price', 0),
                entry_delta=row.get('entry_delta', 0),
                exit_delta=row.get('exit_delta', 0),
                entry_theta=row.get('entry_theta', 0),
                exit_theta=row.get('exit_theta', 0),
                pnl=row.get('pnl', 0),
                pnl_pct=row.get('pnl', 0) / row.get('option_cost', 1) if row.get('option_cost', 0) > 0 else 0,
                days_held=row.get('days_held', 0),
                validation_passed=row.get('validation_passed', True),
                validation_reason=row.get('validation_reason', ''),
                circuit_state=row.get('circuit_state', 'NORMAL'),
                data_source=row.get('data_source', 'BlackScholes'),
                direction=signal.direction if signal else 1,
                continuation_bars=signal.continuation_bars if signal else 0,
            )
            log.add_record(record)

        return log

    def _find_signal(self, row: pd.Series, signals: List[PatternSignal]) -> Optional[PatternSignal]:
        """Find matching signal for a backtest row."""
        timestamp = row.get('timestamp')
        if timestamp is None:
            return signals[0] if signals else None

        for signal in signals:
            if signal.timestamp == timestamp:
                return signal

        return signals[0] if signals else None

    def _calculate_daily_sharpe(
        self,
        trades_df: pd.DataFrame,
        starting_capital: float = 10000.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio from ACTUAL daily returns.

        Session 83K-58: Fixes Sharpe inflation caused by treating each trade
        as a separate period. Multiple trades on the same day are aggregated
        into single daily P&L, then a daily equity curve is built, and
        Sharpe is calculated from daily percentage returns.

        Args:
            trades_df: DataFrame with 'pnl' and 'entry_date' columns
            starting_capital: Initial capital for return calculation
            periods_per_year: Trading periods per year (252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if trades_df is None or len(trades_df) < 2:
            return 0.0

        if 'pnl' not in trades_df.columns or 'entry_date' not in trades_df.columns:
            return 0.0

        # Aggregate trades by calendar date
        daily_pnl = trades_df.groupby(trades_df['entry_date'].dt.date)['pnl'].sum()

        if len(daily_pnl) < 2:
            return 0.0

        # Build daily equity curve from cumulative P&L
        daily_equity = starting_capital + daily_pnl.cumsum()

        # Prepend starting capital to calculate first day's return
        daily_equity_with_start = pd.concat([
            pd.Series([starting_capital], index=[daily_equity.index[0]]),
            daily_equity
        ])

        # Calculate daily percentage returns (same as old pct_change method)
        daily_returns = daily_equity_with_start.pct_change().dropna()

        if len(daily_returns) < 2:
            return 0.0

        mean_return = daily_returns.mean()
        std_return = daily_returns.std(ddof=1)

        # Check for zero or near-zero std (floating point precision issue)
        if std_return == 0 or np.isnan(std_return) or std_return < 1e-10:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        return float(sharpe) if np.isfinite(sharpe) else 0.0

    def _create_backtest_result(
        self,
        trades_df: pd.DataFrame,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Create BacktestResult from trades DataFrame."""
        if trades_df.empty:
            return self._empty_result(data)

        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl / 10000.0  # Assume $10k starting capital

        winners = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winners) / len(trades_df) if len(trades_df) > 0 else 0

        # Calculate equity curve
        # Session 83K-10: Floor equity at zero (long options max loss = premium paid)
        equity = [10000]
        for pnl in trades_df['pnl']:
            new_equity = equity[-1] + pnl
            equity.append(max(0, new_equity))  # Floor at zero - account cannot go negative
        equity_series = pd.Series(equity[1:], index=trades_df['entry_date'] if 'entry_date' in trades_df else range(len(trades_df)))

        # Calculate Sharpe using ACTUAL daily returns
        # Session 83K-58: Fixed Sharpe inflation bug - aggregate trades by date first
        sharpe = self._calculate_daily_sharpe(trades_df)

        # Calculate max drawdown
        # Session 83K-10: Cap MaxDD at 100% (realistic for cash-secured options)
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        # Avoid division by zero when peak is 0
        peak_safe = np.where(peak == 0, 1e-10, peak)
        drawdown = (peak - equity_arr) / peak_safe
        max_dd = min(drawdown.max(), 1.0) if len(drawdown) > 0 else 0  # Cap at 100%

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            trades=trades_df,
            equity_curve=equity_series,
            trade_count=len(trades_df),
            parameters={
                'pattern_types': self.config.pattern_types,
                'timeframe': self.config.timeframe,
                'min_continuation_bars': self.config.min_continuation_bars,
            },
            start_date=data.index[0] if len(data) > 0 else None,
            end_date=data.index[-1] if len(data) > 0 else None,
        )

    def _empty_result(self, data: pd.DataFrame) -> BacktestResult:
        """Return empty BacktestResult when no trades."""
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            trades=pd.DataFrame(columns=['pnl', 'pnl_pct', 'entry_date', 'exit_date', 'days_held']),
            equity_curve=pd.Series([10000], index=[data.index[0]] if len(data) > 0 else [0]),
            trade_count=0,
            parameters={},
            start_date=data.index[0] if len(data) > 0 else None,
            end_date=data.index[-1] if len(data) > 0 else None,
        )
