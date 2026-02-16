"""
Capital Simulator - Backtest equivalent of VirtualBalanceTracker

Pure simulation of capital constraints without threading, file I/O,
or Alpaca sync. Mirrors the live VirtualBalanceTracker's gating logic:
- Virtual capital budget ($3,000 default)
- Fixed dollar or % of capital sizing
- Max concurrent positions (5)
- Portfolio heat limit (8%)
- T+1 settlement for cash accounts
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from strat.backtesting.config import BacktestConfig

logger = logging.getLogger(__name__)


class CapitalSimulator:
    """
    Simulates capital constraints during backtesting.

    Tracks virtual capital, deployed capital, and pending settlements
    to gate position entries exactly as the live system does.

    Usage:
        sim = CapitalSimulator(config)
        if sim.can_open_trade(300.0):
            sim.reserve_capital('trade_1', 300.0)
        sim.release_capital('trade_1', proceeds=450.0, trade_date=date)
        sim.settle_pending(current_date)
    """

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._starting_capital = config.virtual_capital
        self._virtual_capital = config.virtual_capital
        self._deployed_capital = 0.0
        self._positions: Dict[str, float] = {}  # signal_key -> cost_basis
        self._pending_settlements: List[Dict] = []
        self._realized_pnl = 0.0

    @property
    def available_capital(self) -> float:
        """Capital available for new trades (excludes deployed + unsettled)."""
        unsettled = sum(
            s['amount'] for s in self._pending_settlements
            if not s.get('settled', False)
        )
        return self._virtual_capital - self._deployed_capital - unsettled

    @property
    def position_count(self) -> int:
        """Number of currently open positions."""
        return len(self._positions)

    def can_open_trade(self, cost: float) -> bool:
        """
        Check if a new trade is permitted.

        Validates:
        1. Sufficient available capital
        2. Under max concurrent positions
        3. Portfolio heat within limit

        Args:
            cost: Dollar cost of the trade

        Returns:
            True if trade is permitted
        """
        if cost <= 0:
            return False

        if self.available_capital < cost:
            logger.debug("Insufficient capital: need $%.2f, available $%.2f",
                         cost, self.available_capital)
            return False

        if len(self._positions) >= self._config.max_concurrent_positions:
            logger.debug("Position limit reached: %d/%d",
                         len(self._positions), self._config.max_concurrent_positions)
            return False

        # Portfolio heat check
        total_risk = sum(self._positions.values()) + cost
        if self._virtual_capital > 0:
            heat = total_risk / self._virtual_capital
            if heat > self._config.max_portfolio_heat:
                logger.debug("Portfolio heat %.1f%% > max %.1f%%",
                             heat * 100, self._config.max_portfolio_heat * 100)
                return False

        return True

    def get_trade_budget(self) -> float:
        """Return the dollar budget for the next trade."""
        if self._config.sizing_mode == 'pct_capital':
            raw = self._config.pct_of_capital * self._virtual_capital
        else:
            raw = self._config.fixed_dollar_amount
        return min(raw, max(0.0, self.available_capital))

    def reserve_capital(self, signal_key: str, cost: float) -> None:
        """Reserve capital for a new position."""
        self._positions[signal_key] = cost
        self._deployed_capital += cost
        logger.debug("Reserved $%.2f for %s (deployed=$%.2f, positions=%d)",
                     cost, signal_key, self._deployed_capital, len(self._positions))

    def release_capital(
        self,
        signal_key: str,
        proceeds: float,
        trade_date: datetime,
    ) -> None:
        """
        Release capital for a fully closed position.

        Args:
            signal_key: Position identifier
            proceeds: Cash received from closing
            trade_date: Date of the close (for settlement calculation)
        """
        original_cost = self._positions.pop(signal_key, 0.0)
        self._deployed_capital = max(0.0, self._deployed_capital - original_cost)
        pnl = proceeds - original_cost
        self._virtual_capital += pnl
        self._realized_pnl += pnl

        # T+1 settlement
        settlement_date = self._next_trading_day(trade_date)
        self._pending_settlements.append({
            'amount': proceeds,
            'settlement_date': settlement_date,
            'signal_key': signal_key,
            'settled': False,
        })

        logger.debug("Released %s: cost=$%.2f proceeds=$%.2f pnl=$%.2f",
                     signal_key, original_cost, proceeds, pnl)

    def release_capital_partial(
        self,
        signal_key: str,
        fraction: float,
        proceeds: float,
        trade_date: datetime,
    ) -> None:
        """Release a fraction of a position (partial exit)."""
        original_cost = self._positions.get(signal_key, 0.0)
        partial_cost = original_cost * fraction
        self._positions[signal_key] = original_cost - partial_cost
        self._deployed_capital = max(0.0, self._deployed_capital - partial_cost)
        pnl = proceeds - partial_cost
        self._virtual_capital += pnl
        self._realized_pnl += pnl

        settlement_date = self._next_trading_day(trade_date)
        self._pending_settlements.append({
            'amount': proceeds,
            'settlement_date': settlement_date,
            'signal_key': signal_key,
            'settled': False,
        })

    def settle_pending(self, current_date: datetime) -> None:
        """Clear settlements that have reached their settlement date."""
        current_str = current_date.strftime('%Y-%m-%d')
        for settlement in self._pending_settlements:
            if not settlement['settled'] and settlement['settlement_date'] <= current_str:
                settlement['settled'] = True

        # Clean up old settled entries
        self._pending_settlements = [
            s for s in self._pending_settlements
            if not s['settled']
        ]

    def get_summary(self) -> dict:
        """Return a snapshot of the capital state."""
        return {
            'virtual_capital': self._virtual_capital,
            'starting_capital': self._starting_capital,
            'deployed_capital': self._deployed_capital,
            'available_capital': self.available_capital,
            'realized_pnl': self._realized_pnl,
            'position_count': len(self._positions),
            'total_return_pct': (self._virtual_capital - self._starting_capital)
                                / self._starting_capital * 100
                                if self._starting_capital > 0 else 0.0,
        }

    def _next_trading_day(self, from_date: datetime) -> str:
        """Calculate next trading day (simple weekday heuristic)."""
        offset = self._config.settlement_days
        nxt = from_date + timedelta(days=offset)
        while nxt.weekday() >= 5:  # Skip weekends
            nxt += timedelta(days=1)
        return nxt.strftime('%Y-%m-%d')
