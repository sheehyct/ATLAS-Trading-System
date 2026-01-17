"""
Position monitoring for crypto paper trades - Session CRYPTO-4

Monitors open positions and closes them when stop or target is hit.
Designed to run within the crypto daemon polling loop.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crypto.exchange.coinbase_client import CoinbaseClient
from crypto.simulation.paper_trader import PaperTrader, SimulatedTrade

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Represents an exit condition being triggered."""

    trade: SimulatedTrade
    reason: str  # 'STOP', 'TARGET', 'MANUAL'
    current_price: float
    trigger_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


class CryptoPositionMonitor:
    """
    Monitors open paper trading positions for stop/target exits.

    Checks current market price against stop_price and target_price
    for each open trade, and closes trades when conditions are met.

    Usage:
        monitor = CryptoPositionMonitor(client, paper_trader)
        exit_signals = monitor.check_exits()
        for signal in exit_signals:
            monitor.execute_exit(signal)

    Integration with daemon:
        The daemon's entry monitor loop can call check_exits()
        periodically (e.g., every 60 seconds along with trigger polling).
    """

    def __init__(
        self,
        client: CoinbaseClient,
        paper_trader: PaperTrader,
    ):
        """
        Initialize position monitor.

        Args:
            client: CoinbaseClient for fetching current prices
            paper_trader: PaperTrader instance to monitor
        """
        self.client = client
        self.paper_trader = paper_trader
        self._check_count = 0
        self._exit_count = 0
        self._error_count = 0

    def check_exits(self) -> List[ExitSignal]:
        """
        Check all open trades for exit conditions.

        Returns:
            List of ExitSignal objects for trades that should be closed
        """
        self._check_count += 1
        exit_signals: List[ExitSignal] = []

        open_trades = self.paper_trader.account.open_trades
        if not open_trades:
            return exit_signals

        # Get current prices for all symbols
        symbols = list(set(t.symbol for t in open_trades))
        prices = self._get_current_prices(symbols)

        for trade in open_trades:
            current_price = prices.get(trade.symbol)
            if current_price is None:
                logger.warning(f"No price available for {trade.symbol}")
                continue

            exit_signal = self._check_trade_exit(trade, current_price)
            if exit_signal:
                exit_signals.append(exit_signal)

        return exit_signals

    def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols."""
        prices = {}
        for symbol in symbols:
            try:
                price = self.client.get_current_price(symbol)
                if price:
                    prices[symbol] = price
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                self._error_count += 1
        return prices

    def _check_trade_exit(
        self, trade: SimulatedTrade, current_price: float
    ) -> Optional[ExitSignal]:
        """
        Check if a trade should be exited.

        Exit Priority (per STRAT methodology - Session EQUITY-67):
        1. Target Hit (highest priority)
        2. Pattern Invalidated (Type 3 evolution)
        3. Stop Hit (lowest priority)

        Args:
            trade: Trade to check
            current_price: Current market price

        Returns:
            ExitSignal if exit condition met, None otherwise
        """
        # Calculate unrealized P&L (needed for all exit types)
        if trade.side == "BUY":
            unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

        unrealized_pnl_percent = (
            unrealized_pnl / (trade.entry_price * trade.quantity)
        ) * 100

        # Priority 1: Check target (highest priority exit)
        if trade.target_price:
            target_hit = False
            if trade.side == "BUY" and current_price >= trade.target_price:
                target_hit = True
            elif trade.side == "SELL" and current_price <= trade.target_price:
                target_hit = True

            if target_hit:
                logger.info(
                    f"TARGET HIT: {trade.trade_id} {trade.symbol} @ {current_price:.2f} "
                    f"(target: {trade.target_price:.2f}, P&L: ${unrealized_pnl:.2f})"
                )
                return ExitSignal(
                    trade=trade,
                    reason="TARGET",
                    current_price=current_price,
                    trigger_price=trade.target_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                )

        # Priority 2: Check pattern invalidation (Type 3 evolution)
        invalidation_signal = self._check_pattern_invalidation(trade, current_price)
        if invalidation_signal:
            return invalidation_signal

        # Priority 3: Check stop loss (lowest priority)
        if trade.stop_price:
            stop_hit = False
            if trade.side == "BUY" and current_price <= trade.stop_price:
                stop_hit = True
            elif trade.side == "SELL" and current_price >= trade.stop_price:
                stop_hit = True

            if stop_hit:
                logger.info(
                    f"STOP HIT: {trade.trade_id} {trade.symbol} @ {current_price:.2f} "
                    f"(stop: {trade.stop_price:.2f}, P&L: ${unrealized_pnl:.2f})"
                )
                return ExitSignal(
                    trade=trade,
                    reason="STOP",
                    current_price=current_price,
                    trigger_price=trade.stop_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                )

        return None

    def _check_pattern_invalidation(
        self, trade: SimulatedTrade, current_price: float
    ) -> Optional[ExitSignal]:
        """
        Check if entry bar evolved to Type 3 (pattern invalidation).

        Session EQUITY-67: Ported from equity position monitor (EQUITY-44/48).

        Per STRAT methodology, if entry bar breaks BOTH high AND low,
        the pattern premise is invalidated - exit immediately.

        Args:
            trade: Trade to check
            current_price: Current market price

        Returns:
            ExitSignal if pattern invalidated, None otherwise
        """
        # Skip if not a Type 2 entry or missing setup bar data
        if trade.entry_bar_type not in ("2U", "2D"):
            return None

        if trade.entry_bar_high <= 0 or trade.entry_bar_low <= 0:
            return None

        # Update intrabar extremes
        if current_price > trade.intrabar_high:
            trade.intrabar_high = current_price
        if current_price < trade.intrabar_low:
            trade.intrabar_low = current_price

        # Check for Type 3 evolution: broke BOTH setup bar high AND low
        broke_high = trade.intrabar_high > trade.entry_bar_high
        broke_low = trade.intrabar_low < trade.entry_bar_low

        if broke_high and broke_low:
            # Pattern invalidated! Entry bar evolved to Type 3
            if trade.side == "BUY":
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

            unrealized_pnl_percent = (
                unrealized_pnl / (trade.entry_price * trade.quantity)
            ) * 100

            logger.warning(
                f"PATTERN INVALIDATED: {trade.trade_id} {trade.symbol} - "
                f"Entry bar evolved to Type 3: Setup H=${trade.entry_bar_high:.2f} L=${trade.entry_bar_low:.2f}, "
                f"Intrabar H=${trade.intrabar_high:.2f} L=${trade.intrabar_low:.2f}"
            )

            return ExitSignal(
                trade=trade,
                reason="PATTERN",  # Pattern invalidated
                current_price=current_price,
                trigger_price=0.0,  # Not applicable for pattern invalidation
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
            )

        return None

    def execute_exit(self, signal: ExitSignal) -> Optional[SimulatedTrade]:
        """
        Execute an exit based on exit signal.

        Args:
            signal: ExitSignal from check_exits()

        Returns:
            Closed SimulatedTrade or None if failed
        """
        try:
            # Update exit reason before closing
            signal.trade.exit_reason = signal.reason

            # Close the trade
            closed = self.paper_trader.close_trade(
                trade_id=signal.trade.trade_id,
                exit_price=signal.current_price,
            )

            if closed:
                self._exit_count += 1
                logger.info(
                    f"EXIT EXECUTED: {signal.trade.trade_id} ({signal.reason}) "
                    f"P&L: ${closed.pnl:.2f} ({closed.pnl_percent:.1f}%)"
                )
                return closed
            else:
                logger.warning(f"Failed to close trade: {signal.trade.trade_id}")
                return None

        except Exception as e:
            logger.error(f"Error executing exit for {signal.trade.trade_id}: {e}")
            self._error_count += 1
            return None

    def execute_all_exits(
        self, signals: List[ExitSignal]
    ) -> List[SimulatedTrade]:
        """
        Execute all exit signals.

        Args:
            signals: List of ExitSignal objects

        Returns:
            List of closed trades
        """
        closed_trades = []
        for signal in signals:
            closed = self.execute_exit(signal)
            if closed:
                closed_trades.append(closed)
        return closed_trades

    def get_open_positions_with_pnl(self) -> List[Dict[str, Any]]:
        """
        Get all open positions with current P&L.

        Returns:
            List of position dicts with current P&L info
        """
        positions = []
        open_trades = self.paper_trader.account.open_trades

        if not open_trades:
            return positions

        # Get current prices
        symbols = list(set(t.symbol for t in open_trades))
        prices = self._get_current_prices(symbols)

        for trade in open_trades:
            current_price = prices.get(trade.symbol, trade.entry_price)

            # Calculate P&L
            if trade.side == "BUY":
                unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.quantity

            unrealized_pnl_percent = (
                unrealized_pnl / (trade.entry_price * trade.quantity)
            ) * 100

            # Calculate distance to stop/target
            stop_distance_pct = None
            target_distance_pct = None

            if trade.stop_price:
                stop_distance_pct = (
                    (current_price - trade.stop_price) / current_price
                ) * 100
                if trade.side == "SELL":
                    stop_distance_pct = -stop_distance_pct

            if trade.target_price:
                target_distance_pct = (
                    (trade.target_price - current_price) / current_price
                ) * 100
                if trade.side == "SELL":
                    target_distance_pct = -target_distance_pct

            positions.append({
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "current_price": current_price,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "stop_distance_pct": stop_distance_pct,
                "target_distance_pct": target_distance_pct,
                "timeframe": trade.timeframe,
                "pattern_type": trade.pattern_type,
                "entry_time": trade.entry_time.isoformat(),
            })

        return positions

    def get_stats(self) -> Dict[str, Any]:
        """Get position monitoring statistics."""
        return {
            "check_count": self._check_count,
            "exit_count": self._exit_count,
            "error_count": self._error_count,
            "open_positions": len(self.paper_trader.account.open_trades),
            "closed_positions": len(self.paper_trader.account.closed_trades),
        }
