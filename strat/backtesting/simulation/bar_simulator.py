"""
Bar Simulator - Core Event Loop

Processes OHLCV bars chronologically, managing:
1. Signal detection on each bar
2. Entry evaluation for pending signals
3. Exit evaluation for open positions
4. Capital constraint enforcement
5. Position lifecycle (open -> monitor -> close)

This is the central simulation engine that ties together
signal generation, entry simulation, exit evaluation,
and capital tracking.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import pandas as pd

from strat.backtesting.config import BacktestConfig
from strat.backtesting.signals.signal_generator import BacktestSignal, BacktestSignalGenerator
from strat.backtesting.signals.entry_simulator import EntrySimulator
from strat.backtesting.exits.exit_evaluator import BacktestExitEvaluator, ExitEvalResult
from strat.backtesting.simulation.position_tracker import SimulatedPosition, ExitReason
from strat.backtesting.simulation.capital_simulator import CapitalSimulator
from strat.backtesting.data_providers.base import OptionsPriceProvider, OptionsQuoteResult

logger = logging.getLogger(__name__)


class BarSimulator:
    """
    Bar-by-bar simulation engine.

    Processes bars chronologically for a single symbol/timeframe
    combination, managing the full lifecycle from signal detection
    through position exit.

    Usage:
        sim = BarSimulator(config, price_provider)
        trades = sim.run(df, 'SPY', '1D')
    """

    def __init__(
        self,
        config: BacktestConfig,
        price_provider: Optional[OptionsPriceProvider] = None,
        capital_sim: Optional[CapitalSimulator] = None,
    ):
        self._config = config
        self._price_provider = price_provider
        self._capital_sim = capital_sim

        self._signal_gen = BacktestSignalGenerator(config)
        self._entry_sim = EntrySimulator(config)
        self._exit_eval = BacktestExitEvaluator(config)

        # State
        self._pending_signals: List[BacktestSignal] = []
        self._open_positions: List[SimulatedPosition] = []
        self._closed_trades: List[SimulatedPosition] = []
        self._bar_count = 0

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> List[SimulatedPosition]:
        """
        Run bar-by-bar simulation over the entire DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            symbol: Underlying symbol
            timeframe: Timeframe string

        Returns:
            List of closed SimulatedPosition trades
        """
        self._pending_signals = []
        self._open_positions = []
        self._closed_trades = []
        self._bar_count = 0

        # Detect all signals upfront (they're based on pattern completion)
        all_signals = self._signal_gen.detect_signals(df, symbol, timeframe)
        # Group signals by their detection bar index for efficient lookup
        signals_by_bar: Dict[int, List[BacktestSignal]] = {}
        for sig in all_signals:
            idx = sig.detected_bar_index
            signals_by_bar.setdefault(idx, []).append(sig)

        logger.info("Starting simulation: %s %s, %d bars, %d signals detected",
                     symbol, timeframe, len(df), len(all_signals))

        for bar_idx in range(len(df)):
            bar = df.iloc[bar_idx]
            bar_time = df.index[bar_idx]
            if hasattr(bar_time, 'to_pydatetime'):
                bar_time = bar_time.to_pydatetime()

            # Settle pending capital
            if self._capital_sim:
                self._capital_sim.settle_pending(bar_time)

            # Add newly detected signals to pending queue
            if bar_idx in signals_by_bar:
                for sig in signals_by_bar[bar_idx]:
                    self._pending_signals.append(sig)

            # Check exits on open positions FIRST (existing positions have priority)
            self._process_exits(bar, bar_time, df, bar_idx)

            # Then check entries for pending signals
            self._process_entries(bar, bar_time, symbol, timeframe, df, bar_idx)

            self._bar_count += 1

        # Force-close any remaining open positions at last bar
        self._close_remaining(df)

        logger.info("Simulation complete: %d trades closed", len(self._closed_trades))
        return self._closed_trades

    def _process_exits(
        self,
        bar: pd.Series,
        bar_time: datetime,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        """Evaluate exit conditions for all open positions."""
        still_open = []

        for pos in self._open_positions:
            bar_high = bar.get('High', bar.get('high', 0))
            bar_low = bar.get('Low', bar.get('low', 0))
            bar_close = bar.get('Close', bar.get('close', 0))

            # Update option price estimate (simplified: use underlying move as proxy)
            self._update_option_price_estimate(pos, bar_close, bar_time)

            result = self._exit_eval.evaluate(pos, bar_time, bar_high, bar_low, bar_close)

            if result.should_exit:
                self._execute_exit(pos, result, bar_time, bar_close, df, bar_idx)
                if not pos.is_active:
                    self._closed_trades.append(pos)
                    continue
                # Partial exit: position still open with fewer contracts

            still_open.append(pos)

        self._open_positions = still_open

    def _process_entries(
        self,
        bar: pd.Series,
        bar_time: datetime,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        """Evaluate pending signals for entry on this bar."""
        still_pending = []

        for signal in self._pending_signals:
            # Skip signals that are too old (expired)
            bars_since_detection = bar_idx - signal.detected_bar_index
            max_bars = self._config.get_max_holding(timeframe)
            if bars_since_detection > max_bars:
                continue  # Signal expired, drop it

            # Capital check
            if self._capital_sim and self._config.capital_tracking_enabled:
                budget = self._capital_sim.get_trade_budget()
                if budget <= 0 or not self._capital_sim.can_open_trade(budget):
                    still_pending.append(signal)
                    continue

            # Evaluate entry
            entry_result = self._entry_sim.evaluate_entry(signal, bar, bar_time)

            if entry_result is not None:
                pos = self._open_position(signal, entry_result, bar_time, symbol, timeframe, df, bar_idx)
                if pos:
                    self._open_positions.append(pos)
                    if signal.timeframe == '1H':
                        self._entry_sim.record_hourly_entry(bar_time)
                    continue  # Don't re-add to pending

            still_pending.append(signal)

        self._pending_signals = still_pending

    def _open_position(
        self,
        signal: BacktestSignal,
        entry_result: dict,
        bar_time: datetime,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> Optional[SimulatedPosition]:
        """Create a new SimulatedPosition from a triggered signal."""
        actual_entry = entry_result['actual_entry_price']

        # Get option quote for entry pricing
        entry_option_price = self._get_entry_option_price(
            signal, actual_entry, bar_time,
        )
        if entry_option_price is None or entry_option_price <= 0:
            logger.debug("No option price for %s, skipping", signal.signal_key if hasattr(signal, 'signal_key') else symbol)
            return None

        # Position sizing
        budget = (self._capital_sim.get_trade_budget()
                  if self._capital_sim and self._config.capital_tracking_enabled
                  else self._config.fixed_dollar_amount)

        max_loss_per_contract = entry_option_price * 100
        if max_loss_per_contract > 0:
            contracts = max(1, min(10, int(budget / max_loss_per_contract)))
        else:
            contracts = 1

        cost = entry_option_price * contracts * 100
        signal_key = f"{symbol}_{timeframe}_{signal.pattern_type}_{bar_time.strftime('%Y%m%d_%H%M')}"

        # Capital reservation
        if self._capital_sim and self._config.capital_tracking_enabled:
            if not self._capital_sim.can_open_trade(cost):
                return None
            self._capital_sim.reserve_capital(signal_key, cost)

        # Calculate 1x R:R target
        risk = abs(actual_entry - signal.stop_price)
        if signal.direction.upper() in ('CALL', 'BULL', 'UP'):
            target_1x = actual_entry + risk
        else:
            target_1x = actual_entry - risk

        # Adjusted target for hourly
        target_rr = self._config.get_target_rr(timeframe)
        adjusted_target = signal.target_price
        if timeframe == '1H' and target_rr == 1.0:
            adjusted_target = target_1x

        # Determine entry bar type from pattern
        entry_bar_type = ''
        if signal.direction.upper() in ('CALL', 'BULL', 'UP'):
            entry_bar_type = '2U'
        else:
            entry_bar_type = '2D'

        # Build option expiration
        expiration = ''
        if self._price_provider:
            target_dte = self._config.get_target_dte(timeframe)
            exp = self._price_provider.find_expiration(
                symbol, target_dte, bar_time,
                min_dte=self._config.min_dte,
                max_dte=self._config.max_dte,
            )
            if exp:
                expiration = exp

        # Is this a 3-2 pattern? (for ATR trailing)
        is_32 = '3-2' in signal.pattern_type and '3-2-2' not in signal.pattern_type

        pos = SimulatedPosition(
            symbol=symbol,
            signal_key=signal_key,
            direction=signal.direction,
            entry_trigger=signal.entry_trigger,
            target_price=adjusted_target,
            stop_price=signal.stop_price,
            pattern_type=signal.pattern_type,
            timeframe=timeframe,
            entry_price=entry_option_price,
            contracts=contracts,
            entry_time=bar_time,
            expiration=expiration,
            actual_entry_underlying=actual_entry,
            target_1x=target_1x,
            original_target=signal.target_price,
            entry_bar_type=entry_bar_type,
            entry_bar_high=signal.setup_bar_high,
            entry_bar_low=signal.setup_bar_low,
            high_water_mark=actual_entry,
            atr_at_detection=signal.atr_at_detection if is_32 else 0.0,
            use_atr_trailing=is_32 and self._config.use_atr_trailing_for_32,
            atr_trail_distance=signal.atr_at_detection * self._config.atr_trailing_distance_multiple if is_32 else 0.0,
            entry_bar_index=bar_idx,
        )

        logger.debug("Opened position: %s %s %s @ $%.2f (option $%.4f x%d)",
                     symbol, signal.pattern_type, signal.direction,
                     actual_entry, entry_option_price, contracts)

        return pos

    def _execute_exit(
        self,
        pos: SimulatedPosition,
        result: ExitEvalResult,
        bar_time: datetime,
        bar_close: float,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        """Execute an exit for a position."""
        # Get option price at exit
        exit_option_price = self._get_exit_option_price(pos, bar_close, bar_time)

        contracts_to_close = result.contracts_to_close

        pnl = pos.close(
            reason=result.reason,
            exit_time=bar_time,
            exit_option_price=exit_option_price,
            exit_underlying_price=bar_close,
            contracts_closed=contracts_to_close,
        )

        # Capital release
        if self._capital_sim and self._config.capital_tracking_enabled:
            if contracts_to_close and pos.is_active:
                # Partial close
                fraction = contracts_to_close / (contracts_to_close + pos.contracts_remaining)
                proceeds = exit_option_price * contracts_to_close * 100
                self._capital_sim.release_capital_partial(
                    pos.signal_key, fraction, proceeds, bar_time,
                )
            else:
                # Full close
                proceeds = exit_option_price * (contracts_to_close or pos.contracts) * 100
                self._capital_sim.release_capital(pos.signal_key, proceeds, bar_time)

        logger.debug("Closed %s: %s, PnL=$%.2f (%s)",
                     pos.symbol, result.reason.value if result.reason else 'UNKNOWN',
                     pnl, result.details)

    def _get_entry_option_price(
        self,
        signal: BacktestSignal,
        underlying_price: float,
        bar_time: datetime,
    ) -> Optional[float]:
        """Get option price at entry (buy at ask)."""
        if self._price_provider is None:
            # Fallback: estimate as 3% of underlying
            return underlying_price * 0.03

        option_type = 'C' if signal.direction.upper() in ('CALL', 'BULL', 'UP') else 'P'
        target_dte = self._config.get_target_dte(signal.timeframe)

        # Find expiration
        expiration = self._price_provider.find_expiration(
            signal.symbol, target_dte, bar_time,
            min_dte=self._config.min_dte,
            max_dte=self._config.max_dte,
        )
        if not expiration:
            return underlying_price * 0.03  # Fallback

        # Select strike (STRAT methodology: entry + 0.3*(target-entry))
        strike = self._select_strike(
            signal.symbol, expiration, underlying_price,
            signal.entry_trigger, signal.target_price, option_type,
        )
        if strike is None:
            return underlying_price * 0.03  # Fallback

        # Get quote
        quote = self._price_provider.get_quote(
            signal.symbol, expiration, strike, option_type, bar_time,
        )
        if quote and quote.is_valid:
            return quote.fill_price_buy  # Buy at ask
        return underlying_price * 0.03  # Fallback

    def _get_exit_option_price(
        self,
        pos: SimulatedPosition,
        underlying_price: float,
        bar_time: datetime,
    ) -> float:
        """Get option price at exit (sell at bid)."""
        if self._price_provider is None or not pos.expiration or not pos.strike:
            # Rough estimate using underlying move
            if pos.entry_price > 0 and pos.actual_entry_underlying > 0:
                move_pct = (underlying_price - pos.actual_entry_underlying) / pos.actual_entry_underlying
                if pos.is_bullish:
                    return max(0.01, pos.entry_price * (1 + move_pct * 2))
                else:
                    return max(0.01, pos.entry_price * (1 - move_pct * 2))
            return pos.entry_price  # Breakeven fallback

        quote = self._price_provider.get_quote(
            pos.symbol, pos.expiration, pos.strike,
            pos.option_type or ('C' if pos.is_bullish else 'P'),
            bar_time,
        )
        if quote and quote.is_valid:
            return quote.fill_price_sell  # Sell at bid
        return max(0.01, pos.entry_price * 0.5)  # Conservative fallback

    def _select_strike(
        self,
        symbol: str,
        expiration: str,
        underlying_price: float,
        entry_price: float,
        target_price: float,
        option_type: str,
    ) -> Optional[float]:
        """
        Select strike using STRAT methodology.

        Strike = Entry + 0.3 * (Target - Entry) for calls
        Fallback to ATM if no matching strike available.
        """
        if not self._price_provider:
            return round(underlying_price)

        strikes = self._price_provider.get_strikes(symbol, expiration)
        if not strikes:
            return round(underlying_price)

        # STRAT optimal: slightly in-the-money
        if option_type == 'C':
            optimal = entry_price + 0.3 * (target_price - entry_price)
        else:
            optimal = entry_price - 0.3 * (entry_price - target_price)

        # Find nearest available strike
        nearest = min(strikes, key=lambda s: abs(s - optimal))

        # If too far, fall back to ATM
        if abs(nearest - optimal) > abs(optimal - underlying_price) * 0.5:
            nearest = min(strikes, key=lambda s: abs(s - underlying_price))

        return nearest

    def _update_option_price_estimate(
        self,
        pos: SimulatedPosition,
        underlying_close: float,
        bar_time: datetime,
    ) -> None:
        """Update option price estimate for unrealized P&L tracking."""
        if self._price_provider and pos.expiration and pos.strike:
            quote = self._price_provider.get_quote(
                pos.symbol, pos.expiration, pos.strike,
                pos.option_type or ('C' if pos.is_bullish else 'P'),
                bar_time,
            )
            if quote and quote.is_valid:
                pos.update_option_price(quote.mid if quote.mid > 0 else quote.fill_price_sell)
                return

        # Simple delta-based estimate when no provider
        if pos.entry_price > 0 and pos.actual_entry_underlying > 0:
            move = underlying_close - pos.actual_entry_underlying
            delta_est = 0.55  # Approximate delta
            if not pos.is_bullish:
                move = -move
            price_change = move * delta_est
            estimated_price = max(0.01, pos.entry_price + price_change)
            pos.update_option_price(estimated_price)

    def _close_remaining(self, df: pd.DataFrame) -> None:
        """Force-close any positions still open at end of data."""
        if not self._open_positions:
            return

        last_bar = df.iloc[-1]
        last_time = df.index[-1]
        if hasattr(last_time, 'to_pydatetime'):
            last_time = last_time.to_pydatetime()

        bar_close = last_bar.get('Close', last_bar.get('close', 0))

        for pos in self._open_positions:
            exit_price = self._get_exit_option_price(pos, bar_close, last_time)
            pos.close(
                reason=ExitReason.TIME_EXIT,
                exit_time=last_time,
                exit_option_price=exit_price,
                exit_underlying_price=bar_close,
            )
            if self._capital_sim and self._config.capital_tracking_enabled:
                proceeds = exit_price * pos.contracts * 100
                self._capital_sim.release_capital(pos.signal_key, proceeds, last_time)
            self._closed_trades.append(pos)

        self._open_positions = []
