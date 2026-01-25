"""
Live signal generator for StatArb pairs trading.

Monitors Z-score for configured pairs and generates entry/exit signals.
Designed to run alongside STRAT pattern trading in CryptoSignalDaemon.

Session EQUITY-91: Initial implementation for ADA/XRP pair.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatArbSignalType(Enum):
    """Type of StatArb signal."""

    LONG_SPREAD = "long_spread"   # Long asset1, short asset2
    SHORT_SPREAD = "short_spread"  # Short asset1, long asset2
    EXIT = "exit"                  # Close all positions


@dataclass
class StatArbConfig:
    """Configuration for StatArb signal generation."""

    # Z-score calculation
    zscore_window: int = 15  # Rolling window for Z-score (bars)

    # Entry/exit thresholds (validated from walk-forward)
    entry_threshold: float = 2.0  # Enter when |Z| > threshold
    exit_threshold: float = 0.0   # Exit when Z crosses toward 0
    stop_threshold: float = 3.0   # Stop out if |Z| exceeds this

    # Position sizing
    max_position_pct: float = 0.20  # Max 20% of account per leg
    leverage_tier: str = "swing"     # "intraday" or "swing"

    # Risk management
    max_loss_pct: float = 0.10  # Max 10% loss before forced exit
    min_hold_bars: int = 1      # Minimum bars to hold before exit

    # Data requirements
    min_history_bars: int = 30  # Minimum bars needed for calculations


@dataclass
class StatArbSignal:
    """A StatArb trading signal."""

    signal_type: StatArbSignalType
    long_symbol: str    # Symbol to go long
    short_symbol: str   # Symbol to go short
    zscore: float       # Current Z-score
    spread: float       # Current spread value
    timestamp: datetime

    # Position sizing
    long_notional: float = 0.0
    short_notional: float = 0.0

    # Context
    hedge_ratio: float = 1.0
    entry_zscore: Optional[float] = None  # Z-score at entry (for exits)
    bars_held: int = 0                     # How long position held


@dataclass
class StatArbPosition:
    """Tracks an open StatArb position."""

    pair: Tuple[str, str]  # (long_symbol, short_symbol)
    direction: str         # "long_spread" or "short_spread"
    entry_zscore: float
    entry_spread: float
    entry_time: datetime
    hedge_ratio: float
    long_notional: float
    short_notional: float
    bars_held: int = 0

    # P/L tracking (updated each bar)
    current_zscore: float = 0.0
    unrealized_pnl: float = 0.0


class StatArbSignalGenerator:
    """
    Generates StatArb trading signals from live price data.

    Maintains rolling price history and calculates Z-scores
    for configured pairs. Signals entry when Z-score exceeds
    threshold and exit when it reverts to mean.

    Usage:
        generator = StatArbSignalGenerator(
            pairs=[("ADA-USD", "XRP-USD")],
            config=StatArbConfig(),
        )

        # In polling loop:
        signals = generator.check_for_signals(price_fetcher)
        for signal in signals:
            if signal.signal_type == StatArbSignalType.LONG_SPREAD:
                # Open long spread position
                pass
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        config: Optional[StatArbConfig] = None,
        account_value: float = 1000.0,
    ) -> None:
        """
        Initialize signal generator.

        Args:
            pairs: List of (symbol1, symbol2) tuples to trade
            config: StatArb configuration
            account_value: Account value for position sizing
        """
        self.pairs = pairs
        self.config = config or StatArbConfig()
        self.account_value = account_value

        # Price history for each symbol
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Calculated values per pair
        self._hedge_ratios: Dict[Tuple[str, str], float] = {}
        self._last_zscore: Dict[Tuple[str, str], float] = {}

        # Open positions
        self._positions: Dict[Tuple[str, str], StatArbPosition] = {}

        # Initialize price history containers
        for sym1, sym2 in pairs:
            self._price_history[sym1] = []
            self._price_history[sym2] = []
            self._hedge_ratios[(sym1, sym2)] = 1.0
            self._last_zscore[(sym1, sym2)] = 0.0

    def update_account_value(self, value: float) -> None:
        """Update account value for position sizing."""
        self.account_value = value

    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update price history with new prices.

        Args:
            prices: Dict of symbol -> current price
            timestamp: Timestamp for prices (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        for symbol, price in prices.items():
            if symbol in self._price_history:
                self._price_history[symbol].append((timestamp, price))

                # Trim to max history
                max_history = self.config.zscore_window * 3
                if len(self._price_history[symbol]) > max_history:
                    self._price_history[symbol] = self._price_history[symbol][-max_history:]

    def _get_price_series(self, symbol: str) -> Optional[pd.Series]:
        """Get price series for a symbol."""
        history = self._price_history.get(symbol, [])
        if len(history) < self.config.min_history_bars:
            return None

        timestamps, prices = zip(*history)
        return pd.Series(prices, index=pd.DatetimeIndex(timestamps))

    def _calculate_zscore(self, pair: Tuple[str, str]) -> Optional[float]:
        """
        Calculate current Z-score for a pair.

        Returns None if insufficient data.
        """
        sym1, sym2 = pair

        prices1 = self._get_price_series(sym1)
        prices2 = self._get_price_series(sym2)

        if prices1 is None or prices2 is None:
            return None

        # Align indices
        common_idx = prices1.index.intersection(prices2.index)
        if len(common_idx) < self.config.min_history_bars:
            return None

        p1 = prices1.loc[common_idx]
        p2 = prices2.loc[common_idx]

        # Calculate log spread
        log_p1 = np.log(p1)
        log_p2 = np.log(p2)

        # Calculate hedge ratio (rolling OLS)
        if len(log_p1) >= self.config.zscore_window:
            window_p1 = log_p1[-self.config.zscore_window:]
            window_p2 = log_p2[-self.config.zscore_window:]

            cov = np.cov(window_p2, window_p1)[0, 1]
            var = np.var(window_p2)
            hedge_ratio = cov / var if var > 0 else 1.0

            self._hedge_ratios[pair] = hedge_ratio
        else:
            hedge_ratio = self._hedge_ratios.get(pair, 1.0)

        # Calculate spread
        spread = log_p1 - hedge_ratio * log_p2

        # Calculate Z-score
        window = min(len(spread), self.config.zscore_window)
        spread_window = spread[-window:]

        mean = spread_window.mean()
        std = spread_window.std()

        if std <= 0:
            return 0.0

        zscore = (spread.iloc[-1] - mean) / std
        self._last_zscore[pair] = zscore

        return zscore

    def check_for_signals(
        self,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[StatArbSignal]:
        """
        Check for entry/exit signals on all pairs.

        Args:
            prices: Current prices (if not provided, uses cached)

        Returns:
            List of signals (may be empty)
        """
        signals = []

        if prices:
            self.update_prices(prices)

        for pair in self.pairs:
            sym1, sym2 = pair

            zscore = self._calculate_zscore(pair)
            if zscore is None:
                continue

            # Get previous Z-score for crossover detection
            prev_zscore = self._last_zscore.get(pair, 0.0)

            # Check for existing position
            position = self._positions.get(pair)

            if position is not None:
                # Update position
                position.bars_held += 1
                position.current_zscore = zscore

                # Check for exit signals
                exit_signal = self._check_exit(pair, position, zscore)
                if exit_signal:
                    signals.append(exit_signal)
                    del self._positions[pair]
            else:
                # Check for entry signals
                entry_signal = self._check_entry(pair, zscore, prev_zscore)
                if entry_signal:
                    signals.append(entry_signal)

                    # Create position tracking
                    self._positions[pair] = StatArbPosition(
                        pair=pair,
                        direction=entry_signal.signal_type.value,
                        entry_zscore=zscore,
                        entry_spread=0.0,  # Would need spread value
                        entry_time=entry_signal.timestamp,
                        hedge_ratio=self._hedge_ratios.get(pair, 1.0),
                        long_notional=entry_signal.long_notional,
                        short_notional=entry_signal.short_notional,
                    )

        return signals

    def _check_entry(
        self,
        pair: Tuple[str, str],
        zscore: float,
        prev_zscore: float,
    ) -> Optional[StatArbSignal]:
        """Check for entry signal."""
        sym1, sym2 = pair
        hedge_ratio = self._hedge_ratios.get(pair, 1.0)

        # Long spread: Z crosses below -entry_threshold
        if prev_zscore > -self.config.entry_threshold and zscore <= -self.config.entry_threshold:
            # Calculate position sizes
            notional = self.account_value * self.config.max_position_pct

            return StatArbSignal(
                signal_type=StatArbSignalType.LONG_SPREAD,
                long_symbol=sym1,
                short_symbol=sym2,
                zscore=zscore,
                spread=0.0,
                timestamp=datetime.utcnow(),
                long_notional=notional,
                short_notional=notional * abs(hedge_ratio),
                hedge_ratio=hedge_ratio,
            )

        # Short spread: Z crosses above entry_threshold
        if prev_zscore < self.config.entry_threshold and zscore >= self.config.entry_threshold:
            notional = self.account_value * self.config.max_position_pct

            return StatArbSignal(
                signal_type=StatArbSignalType.SHORT_SPREAD,
                long_symbol=sym2,  # Swap direction
                short_symbol=sym1,
                zscore=zscore,
                spread=0.0,
                timestamp=datetime.utcnow(),
                long_notional=notional * abs(hedge_ratio),
                short_notional=notional,
                hedge_ratio=hedge_ratio,
            )

        return None

    def _check_exit(
        self,
        pair: Tuple[str, str],
        position: StatArbPosition,
        zscore: float,
    ) -> Optional[StatArbSignal]:
        """Check for exit signal."""
        # Minimum hold check
        if position.bars_held < self.config.min_hold_bars:
            return None

        # Exit on mean reversion (Z crosses exit_threshold toward 0)
        if position.direction == "long_spread":
            # Entered at Z < -entry, exit when Z > exit (default 0)
            if zscore >= self.config.exit_threshold:
                return StatArbSignal(
                    signal_type=StatArbSignalType.EXIT,
                    long_symbol=position.pair[0],
                    short_symbol=position.pair[1],
                    zscore=zscore,
                    spread=0.0,
                    timestamp=datetime.utcnow(),
                    hedge_ratio=position.hedge_ratio,
                    entry_zscore=position.entry_zscore,
                    bars_held=position.bars_held,
                )
        else:
            # Entered at Z > entry, exit when Z < exit (default 0)
            if zscore <= self.config.exit_threshold:
                return StatArbSignal(
                    signal_type=StatArbSignalType.EXIT,
                    long_symbol=position.pair[1],
                    short_symbol=position.pair[0],
                    zscore=zscore,
                    spread=0.0,
                    timestamp=datetime.utcnow(),
                    hedge_ratio=position.hedge_ratio,
                    entry_zscore=position.entry_zscore,
                    bars_held=position.bars_held,
                )

        # Stop out on extreme Z-score
        if abs(zscore) >= self.config.stop_threshold:
            return StatArbSignal(
                signal_type=StatArbSignalType.EXIT,
                long_symbol=position.pair[0] if position.direction == "long_spread" else position.pair[1],
                short_symbol=position.pair[1] if position.direction == "long_spread" else position.pair[0],
                zscore=zscore,
                spread=0.0,
                timestamp=datetime.utcnow(),
                hedge_ratio=position.hedge_ratio,
                entry_zscore=position.entry_zscore,
                bars_held=position.bars_held,
            )

        return None

    def get_current_zscore(self, pair: Tuple[str, str]) -> Optional[float]:
        """Get current Z-score for a pair."""
        return self._last_zscore.get(pair)

    def get_position(self, pair: Tuple[str, str]) -> Optional[StatArbPosition]:
        """Get current position for a pair."""
        return self._positions.get(pair)

    def get_all_positions(self) -> Dict[Tuple[str, str], StatArbPosition]:
        """Get all open positions."""
        return dict(self._positions)

    def has_position(self, pair: Tuple[str, str]) -> bool:
        """Check if position exists for pair."""
        return pair in self._positions

    def get_active_symbols(self) -> set:
        """Get symbols currently in use by open positions."""
        symbols = set()
        for (sym1, sym2), pos in self._positions.items():
            symbols.add(sym1)
            symbols.add(sym2)
        return symbols

    def close_position(self, pair: Tuple[str, str]) -> Optional[StatArbPosition]:
        """
        Manually close a position.

        Returns the closed position or None if not found.
        """
        return self._positions.pop(pair, None)

    def get_status(self) -> Dict:
        """Get generator status for monitoring."""
        status = {
            "pairs": [f"{s1}/{s2}" for s1, s2 in self.pairs],
            "positions": len(self._positions),
            "zscores": {},
            "history_bars": {},
        }

        for pair in self.pairs:
            sym1, sym2 = pair
            key = f"{sym1}/{sym2}"

            z = self._last_zscore.get(pair)
            status["zscores"][key] = round(z, 3) if z is not None else None

            bars1 = len(self._price_history.get(sym1, []))
            bars2 = len(self._price_history.get(sym2, []))
            status["history_bars"][key] = min(bars1, bars2)

        return status
