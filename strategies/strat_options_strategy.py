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
        mapping = {
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
        result['pattern_timestamp'] = pd.NaT

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
                    # Create custom type for 3-2
                    ptype = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN
                elif pattern_base == '322':
                    # Create custom type for 3-2-2
                    ptype = PatternType.PATTERN_312_UP if direction == 1 else PatternType.PATTERN_312_DOWN
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
        """Apply continuation bar filter to signals."""
        filtered = []

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

                if signal.direction == 1 and bar_class == 2:
                    count += 1
                elif signal.direction == -1 and bar_class == -2:
                    count += 1
                else:
                    break

            signal.continuation_bars = count
            signal.is_filtered = count >= min_bars

            if signal.is_filtered:
                filtered.append(signal)

        return filtered

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
            exit_type = row.get('exit_type', 'UNKNOWN')
            if exit_type == 'TARGET':
                exit_reason = ExitReason.TARGET.value
            elif exit_type == 'STOP':
                exit_reason = ExitReason.STOP.value
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
        equity = [10000]
        for pnl in trades_df['pnl']:
            equity.append(equity[-1] + pnl)
        equity_series = pd.Series(equity[1:], index=trades_df['entry_date'] if 'entry_date' in trades_df else range(len(trades_df)))

        # Calculate Sharpe (simplified - daily returns approximation)
        if len(equity) > 2:
            returns = pd.Series(equity).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

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
